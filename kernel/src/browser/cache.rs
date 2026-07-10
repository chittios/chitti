//! **HTTP memory cache** for browser resources.
//!
//! Reference (not linked — C++ cannot enter this `no_std` kernel):
//! - Ladybird `Libraries/LibHTTP/Cache/MemoryCache.{h,cpp}`
//! - Ladybird `Libraries/LibHTTP/Cache/CacheMode.h`
//! - Ladybird `Libraries/LibHTTP/Cache/Utilities.{h,cpp}` (RFC 9111 freshness)
//! - Ladybird `Libraries/LibWeb/ServiceWorker/Cache.h` (Cache API shape for named
//!   request→response maps; we keep a single process-wide memory cache plus a
//!   lightweight named `AssetStore` for page-session assets)
//!
//! Clone for local reading: `../ladybird-ref` (or any Ladybird checkout).
//! Behaviour here is a **subset**: key = method + URL (no Vary multi-entry),
//! freshness from `Cache-Control: max-age` / heuristic default, bounded
//! entry count + total bytes (first-fit eviction of oldest).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Fetch cache policy — mirrors Ladybird / Fetch `cache` mode names.
/// See Ladybird `LibHTTP/Cache/CacheMode.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMode {
    /// Search cache; update on network response.
    Default,
    /// Never search; never store.
    NoStore,
    /// Never search; store network response.
    Reload,
    /// Search but always revalidate conceptually (we treat as network-first
    /// then store — no conditional GET yet).
    NoCache,
    /// Prefer cache even if stale.
    ForceCache,
    /// Cache only; miss is an error.
    OnlyIfCached,
}

impl CacheMode {
    pub fn permits_lookup(self) -> bool {
        !matches!(self, CacheMode::NoStore | CacheMode::Reload)
    }

    pub fn permits_store(self) -> bool {
        !matches!(self, CacheMode::NoStore | CacheMode::OnlyIfCached)
    }

    pub fn permits_stale(self) -> bool {
        matches!(self, CacheMode::ForceCache | CacheMode::OnlyIfCached)
    }
}

/// One complete cached HTTP response (Ladybird `MemoryCache::Entry` subset).
#[derive(Clone, Debug)]
pub struct Entry {
    pub url: String,
    pub method: String,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// MIME type from `Content-Type` (type/subtype only) or empty.
    pub content_type: String,
    pub request_time_ms: u64,
    pub response_time_ms: u64,
    /// Seconds of freshness from `Cache-Control: max-age=N`, or default.
    pub freshness_lifetime_secs: u64,
}

impl Entry {
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.response_time_ms)
    }

    pub fn is_fresh(&self, now_ms: u64) -> bool {
        let age_secs = self.age_ms(now_ms) / 1000;
        age_secs <= self.freshness_lifetime_secs
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Process-wide in-memory cache (Ladybird `MemoryCache`).
pub struct MemoryCache {
    /// Keyed by `cache_key(method, url)`.
    map: BTreeMap<u64, Entry>,
    /// Insertion order for LRU-ish eviction (oldest first).
    order: Vec<u64>,
    total_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    hits: u64,
    misses: u64,
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new(128, 8 * 1024 * 1024)
    }
}

impl MemoryCache {
    /// `const` so the process-wide static can be initialized without lazy locks.
    pub const fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            map: BTreeMap::new(),
            order: Vec::new(),
            total_bytes: 0,
            max_entries: if max_entries == 0 { 1 } else { max_entries },
            max_bytes: if max_bytes < 64 * 1024 {
                64 * 1024
            } else {
                max_bytes
            },
            hits: 0,
            misses: 0,
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn misses(&self) -> u64 {
        self.misses
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.total_bytes = 0;
    }

    /// Open a matching entry if the mode and freshness allow (Ladybird
    /// `MemoryCache::open_entry`).
    pub fn open_entry(
        &mut self,
        url: &str,
        method: &str,
        mode: CacheMode,
        now_ms: u64,
    ) -> Option<Entry> {
        if !mode.permits_lookup() {
            return None;
        }
        if !is_method_cacheable(method) {
            return None;
        }
        let key = cache_key(method, url);
        let Some(entry) = self.map.get(&key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        if entry.is_fresh(now_ms) || mode.permits_stale() {
            self.hits = self.hits.saturating_add(1);
            // Touch: move key to end of order.
            if let Some(i) = self.order.iter().position(|k| *k == key) {
                self.order.remove(i);
                self.order.push(key);
            }
            return Some(entry.clone());
        }
        // Expired under Default/NoCache — drop so network revalidates.
        self.remove_key(key);
        self.misses = self.misses.saturating_add(1);
        None
    }

    /// Store a completed response (Ladybird create_entry + finalize_entry).
    pub fn put(&mut self, entry: Entry) {
        if !is_method_cacheable(&entry.method) {
            return;
        }
        if !is_status_cacheable(entry.status) {
            return;
        }
        if entry.header("cache-control").is_some_and(|cc| {
            contains_cc_directive(cc, "no-store") || contains_cc_directive(cc, "private")
        }) {
            // Skip no-store; `private` is fine for a single-user OS cache but we
            // still honour no-store strictly.
            if entry
                .header("cache-control")
                .is_some_and(|cc| contains_cc_directive(cc, "no-store"))
            {
                return;
            }
        }
        // Oversized single entry: refuse rather than thrashing.
        if entry.body.len() > self.max_bytes {
            return;
        }
        let key = cache_key(&entry.method, &entry.url);
        if let Some(old) = self.map.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.body.len());
            self.order.retain(|k| *k != key);
        }
        self.total_bytes = self.total_bytes.saturating_add(entry.body.len());
        self.map.insert(key, entry);
        self.order.push(key);
        self.evict_if_needed();
    }

    fn remove_key(&mut self, key: u64) {
        if let Some(old) = self.map.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.body.len());
        }
        self.order.retain(|k| *k != key);
    }

    fn evict_if_needed(&mut self) {
        while self.map.len() > self.max_entries || self.total_bytes > self.max_bytes {
            let Some(oldest) = self.order.first().copied() else {
                break;
            };
            self.remove_key(oldest);
        }
    }
}

/// Named asset map for a page session (Service Worker `Cache` name → entries
/// idea: hold URL → body for the current document's subresources without
/// fighting the shared MemoryCache TTL).
#[derive(Default)]
pub struct AssetStore {
    /// Absolute URL → raw bytes + content type.
    pub(crate) map: BTreeMap<String, (String, Vec<u8>)>,
}

impl AssetStore {
    pub const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn put(&mut self, url: &str, content_type: &str, body: Vec<u8>) {
        self.map
            .insert(url.to_string(), (content_type.to_string(), body));
    }

    pub fn get(&self, url: &str) -> Option<(&str, &[u8])> {
        self.map
            .get(url)
            .map(|(ct, b)| (ct.as_str(), b.as_slice()))
    }

    pub fn contains(&self, url: &str) -> bool {
        self.map.contains_key(url)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn urls(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

// ── pure helpers (unit-tested; Ladybird Utilities.cpp equivalents) ──────────

/// FNV-1a 64-bit over `METHOD\0URL` (Ladybird uses SHA-1 of url+method).
pub fn cache_key(method: &str, url: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in method.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= 0;
    h = h.wrapping_mul(0x100000001b3);
    for b in url.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn is_method_cacheable(method: &str) -> bool {
    method.eq_ignore_ascii_case("GET") || method.eq_ignore_ascii_case("HEAD")
}

/// RFC 9111: cacheable by default for 200, 203, 204, 206, 300, 301, 404, 405,
/// 410, 414, 501 — we allow the common success/redirect set.
pub fn is_status_cacheable(status: u16) -> bool {
    matches!(
        status,
        200 | 203 | 204 | 206 | 300 | 301 | 404 | 405 | 410 | 414 | 501
    )
}

/// Extract MIME type (before `;`) from a Content-Type header value.
pub fn mime_from_content_type(ct: &str) -> String {
    let main = ct.split(';').next().unwrap_or(ct).trim();
    main.to_ascii_lowercase()
}

/// Sniff a few magic numbers when Content-Type is missing (images only).
pub fn sniff_mime(body: &[u8]) -> &'static str {
    if body.len() >= 8 && body[0..8] == [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'] {
        return "image/png";
    }
    if body.len() >= 3 && body[0] == 0xff && body[1] == 0xd8 && body[2] == 0xff {
        return "image/jpeg";
    }
    if body.len() >= 6
        && (body.starts_with(b"GIF87a") || body.starts_with(b"GIF89a"))
    {
        return "image/gif";
    }
    if body.len() >= 12 && body.starts_with(b"RIFF") && &body[8..12] == b"WEBP" {
        return "image/webp";
    }
    if body.starts_with(b"{") || body.starts_with(b"[") {
        return "application/json";
    }
    if looks_like_html(body) {
        return "text/html";
    }
    if looks_like_css(body) {
        return "text/css";
    }
    "application/octet-stream"
}

fn looks_like_html(body: &[u8]) -> bool {
    let n = body.len().min(256);
    let s = core::str::from_utf8(&body[..n]).unwrap_or("");
    let t = s.trim_start();
    t.starts_with("<!DOCTYPE")
        || t.starts_with("<!doctype")
        || t.starts_with("<html")
        || t.starts_with("<HTML")
}

fn looks_like_css(body: &[u8]) -> bool {
    let n = body.len().min(128);
    let s = core::str::from_utf8(&body[..n]).unwrap_or("");
    s.contains('{') && (s.contains(':') || s.contains('@'))
}

/// Parse `Cache-Control: max-age=N` (seconds).
pub fn parse_max_age(cache_control: &str) -> Option<u64> {
    for part in cache_control.split(',') {
        let p = part.trim();
        let lower = p.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("max-age=") {
            let num = rest
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap_or("");
            if let Ok(n) = num.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

pub fn contains_cc_directive(cache_control: &str, directive: &str) -> bool {
    let want = directive.to_ascii_lowercase();
    for part in cache_control.split(',') {
        let p = part.trim().to_ascii_lowercase();
        if p == want || p.starts_with(&alloc::format!("{want}=")) {
            return true;
        }
    }
    false
}

/// Freshness lifetime: max-age if present, else heuristic 300s for 200 GET.
pub fn freshness_lifetime_secs(status: u16, headers: &[(String, String)]) -> u64 {
    if let Some((_, cc)) = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("cache-control"))
    {
        if contains_cc_directive(cc, "no-cache") || contains_cc_directive(cc, "must-revalidate") {
            return 0;
        }
        if let Some(ma) = parse_max_age(cc) {
            return ma;
        }
    }
    if status == 200 || status == 203 || status == 301 {
        300
    } else {
        0
    }
}

/// Build an [`Entry`] from a network response.
pub fn entry_from_response(
    url: &str,
    method: &str,
    status: u16,
    headers: &[(String, String)],
    body: Vec<u8>,
    request_time_ms: u64,
    response_time_ms: u64,
) -> Entry {
    let ct = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| mime_from_content_type(v))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| sniff_mime(&body).to_string());
    let freshness = freshness_lifetime_secs(status, headers);
    Entry {
        url: url.to_string(),
        method: method.to_string(),
        status,
        headers: headers.to_vec(),
        body,
        content_type: ct,
        request_time_ms,
        response_time_ms,
        freshness_lifetime_secs: freshness,
    }
}

/// Global memory cache (process-wide, like Ladybird `MemoryCache` singleton usage).
pub static MEMORY_CACHE: crate::mm::Locked<MemoryCache> =
    crate::mm::Locked::new(MemoryCache::new(128, 8 * 1024 * 1024));

/// Clear the global cache (tests / `/browse` hard reload).
pub fn clear_global() {
    MEMORY_CACHE.with(|c| c.clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    #[test_case]
    fn cache_key_stable_and_distinct() {
        let a = cache_key("GET", "https://ex.com/a");
        let b = cache_key("GET", "https://ex.com/a");
        let c = cache_key("GET", "https://ex.com/b");
        let d = cache_key("HEAD", "https://ex.com/a");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test_case]
    fn mime_and_sniff() {
        assert_eq!(mime_from_content_type("text/html; charset=utf-8"), "text/html");
        assert_eq!(
            sniff_mime(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0]),
            "image/png"
        );
        assert_eq!(sniff_mime(&[0xff, 0xd8, 0xff, 0xe0]), "image/jpeg");
        assert_eq!(sniff_mime(b"<!DOCTYPE html><html>"), "text/html");
    }

    #[test_case]
    fn max_age_and_cc() {
        assert_eq!(parse_max_age("public, max-age=60"), Some(60));
        assert_eq!(parse_max_age("max-age=0, no-cache"), Some(0));
        assert!(contains_cc_directive("no-store, max-age=1", "no-store"));
        assert!(!contains_cc_directive("max-age=1", "no-store"));
    }

    #[test_case]
    fn memory_cache_hit_miss_fresh_stale() {
        let mut c = MemoryCache::new(4, 1024 * 1024);
        let headers = vec![(
            String::from("cache-control"),
            String::from("max-age=10"),
        )];
        let e = entry_from_response(
            "https://ex.com/x",
            "GET",
            200,
            &headers,
            b"hello".to_vec(),
            1000,
            1000,
        );
        assert_eq!(e.freshness_lifetime_secs, 10);
        c.put(e);
        assert_eq!(c.len(), 1);

        // Fresh at t=5000 (age 4s < 10).
        let hit = c.open_entry("https://ex.com/x", "GET", CacheMode::Default, 5000);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().body, b"hello");
        assert_eq!(c.hits(), 1);

        // Stale at t=20000 under Default → miss + evict.
        let miss = c.open_entry("https://ex.com/x", "GET", CacheMode::Default, 20_000);
        assert!(miss.is_none());
        assert_eq!(c.len(), 0);

        // ForceCache serves stale.
        let e2 = entry_from_response(
            "https://ex.com/y",
            "GET",
            200,
            &headers,
            b"y".to_vec(),
            0,
            0,
        );
        c.put(e2);
        let forced = c.open_entry("https://ex.com/y", "GET", CacheMode::ForceCache, 999_999);
        assert!(forced.is_some());
    }

    #[test_case]
    fn only_if_cached_and_no_store() {
        let mut c = MemoryCache::new(4, 1024);
        assert!(c
            .open_entry("https://a", "GET", CacheMode::OnlyIfCached, 0)
            .is_none());

        let e = entry_from_response(
            "https://a",
            "GET",
            200,
            &[(String::from("cache-control"), String::from("no-store"))],
            b"x".to_vec(),
            0,
            0,
        );
        c.put(e);
        assert_eq!(c.len(), 0, "no-store must not enter cache");
    }

    #[test_case]
    fn asset_store_roundtrip() {
        let mut a = AssetStore::new();
        a.put("https://ex.com/i.png", "image/png", b"\x89PNG".to_vec());
        assert!(a.contains("https://ex.com/i.png"));
        let (ct, body) = a.get("https://ex.com/i.png").unwrap();
        assert_eq!(ct, "image/png");
        assert_eq!(body, b"\x89PNG");
        assert_eq!(a.len(), 1);
        a.clear();
        assert!(a.is_empty() || a.len() == 0);
    }

    #[test_case]
    fn eviction_by_count() {
        let mut c = MemoryCache::new(2, 1024 * 1024);
        for i in 0..3 {
            let url = alloc::format!("https://ex.com/{i}");
            c.put(entry_from_response(
                &url,
                "GET",
                200,
                &[],
                b"ab".to_vec(),
                0,
                0,
            ));
        }
        assert!(c.len() <= 2);
        assert!(c.total_bytes() <= 4);
    }
}
