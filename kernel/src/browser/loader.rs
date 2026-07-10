//! **Resource loader** — browser fetch front-end with cache + MIME.
//!
//! Reference (not linked):
//! - Ladybird `Libraries/LibWeb/Loader/ResourceLoader.{h,cpp}`
//! - Ladybird `Libraries/LibWeb/Loader/LoadRequest.h`
//! - Fetch infrastructure under `Libraries/LibWeb/Fetch/`
//!
//! Ladybird's loader is async over RequestServer IPC. Here the kernel already
//! owns the NIC (`net::http`); loads are **synchronous but cooperative**
//! (`shell::upkeep` / Ctrl+C via `get_follow`). Cache lookup/store goes through
//! [`super::cache::MemoryCache`] the same way Ladybird consults `HTTP::MemoryCache`.

use super::cache::{
    self, entry_from_response, mime_from_content_type, sniff_mime, AssetStore, CacheMode, Entry,
    MEMORY_CACHE,
};
use super::cors::{self, CorsResult, Origin, RequestMode};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI32, Ordering};

/// What the load is for (Fetch `destination` / Ladybird LoadRequest destination).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Destination {
    Document,
    Image,
    Script,
    Style,
    Worker,
    Font,
    Other,
}

impl Destination {
    pub fn as_str(self) -> &'static str {
        match self {
            Destination::Document => "document",
            Destination::Image => "image",
            Destination::Script => "script",
            Destination::Style => "style",
            Destination::Worker => "worker",
            Destination::Font => "font",
            Destination::Other => "other",
        }
    }
}

/// A load request — Ladybird `Web::LoadRequest` subset.
#[derive(Clone, Debug)]
pub struct LoadRequest {
    pub url: String,
    pub method: String,
    pub cache_mode: CacheMode,
    pub destination: Destination,
    pub timeout_ms: u64,
    /// Document URL that initiated this load (for relative resolve upstream).
    pub source_url: Option<String>,
    /// CORS mode (Fetch / Ladybird Request mode).
    pub cors_mode: RequestMode,
    /// Initiator origin for CORS (document URL).
    pub initiator_origin: Option<String>,
    /// Include credentials in CORS sense.
    pub credentials: bool,
}

impl LoadRequest {
    pub fn get(url: &str) -> Self {
        Self {
            url: url.trim().to_string(),
            method: String::from("GET"),
            cache_mode: CacheMode::Default,
            destination: Destination::Other,
            timeout_ms: 60_000,
            source_url: None,
            cors_mode: RequestMode::NoCors,
            initiator_origin: None,
            credentials: false,
        }
    }

    pub fn document(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Document;
        r.cors_mode = RequestMode::Navigate;
        r.timeout_ms = 60_000;
        r
    }

    pub fn image(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Image;
        r.cors_mode = RequestMode::NoCors;
        r.timeout_ms = 20_000;
        r
    }

    pub fn script(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Script;
        r.cors_mode = RequestMode::NoCors;
        r.timeout_ms = 30_000;
        r
    }

    pub fn style(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Style;
        r.cors_mode = RequestMode::NoCors;
        r.timeout_ms = 20_000;
        r
    }

    pub fn worker(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Worker;
        r.cors_mode = RequestMode::Cors;
        r.timeout_ms = 30_000;
        r
    }

    pub fn iframe(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Document;
        r.cors_mode = RequestMode::Navigate; // nested nav; sandboxed CORS later
        r.timeout_ms = 45_000;
        r
    }

    pub fn fetch_cors(url: &str) -> Self {
        let mut r = Self::get(url);
        r.destination = Destination::Other;
        r.cors_mode = RequestMode::Cors;
        r.timeout_ms = 30_000;
        r
    }

    pub fn with_cache_mode(mut self, mode: CacheMode) -> Self {
        self.cache_mode = mode;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source_url = Some(source.to_string());
        self.initiator_origin = Origin::from_url(source).serialized.into();
        if self.initiator_origin.as_deref() == Some("null") {
            self.initiator_origin = None;
        }
        self
    }

    pub fn with_cors(mut self, mode: RequestMode) -> Self {
        self.cors_mode = mode;
        self
    }

    pub fn with_credentials(mut self, yes: bool) -> Self {
        self.credentials = yes;
        self
    }
}

/// Result of a successful load (network or cache).
#[derive(Clone, Debug)]
pub struct LoadedResource {
    /// Final URL after redirects.
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub from_cache: bool,
    pub redirects: u32,
    pub destination: Destination,
    /// CORS outcome (`Ok` / `Opaque` / error already returned).
    pub cors_opaque: bool,
}

impl LoadedResource {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    pub fn is_html(&self) -> bool {
        self.content_type.starts_with("text/html")
            || self.content_type.is_empty() && looks_html_bytes(&self.body)
    }

    pub fn is_image(&self) -> bool {
        self.content_type.starts_with("image/")
    }
}

fn looks_html_bytes(body: &[u8]) -> bool {
    sniff_mime(body) == "text/html"
}

static PENDING_LOADS: AtomicI32 = AtomicI32::new(0);

/// Ladybird `ResourceLoader::pending_loads`.
pub fn pending_loads() -> i32 {
    PENDING_LOADS.load(Ordering::Relaxed)
}

fn now_ms() -> u64 {
    #[cfg(test)]
    {
        // Deterministic clock in unit tests (pure path never hits network).
        1_000_000
    }
    #[cfg(not(test))]
    {
        crate::arch::now_ms()
    }
}

/// Load a resource: cache open → network (`get_follow`) → cache put.
///
/// `OnlyIfCached` never touches the network. `NoStore` / `Reload` skip lookup.
pub fn load(req: &LoadRequest) -> Result<LoadedResource, String> {
    if req.url.is_empty() {
        return Err("empty url".into());
    }
    if !super::url::is_http_url(&req.url) {
        return Err("url must be http:// or https://".into());
    }

    PENDING_LOADS.fetch_add(1, Ordering::Relaxed);
    let result = load_inner(req);
    PENDING_LOADS.fetch_sub(1, Ordering::Relaxed);
    result
}

fn load_inner(req: &LoadRequest) -> Result<LoadedResource, String> {
    let now = now_ms();
    let method = if req.method.is_empty() {
        "GET"
    } else {
        req.method.as_str()
    };

    // ── cache lookup (Ladybird MemoryCache::open_entry) ───────────────────
    if req.cache_mode.permits_lookup() {
        let hit = MEMORY_CACHE.with(|c| {
            c.open_entry(&req.url, method, req.cache_mode, now)
        });
        if let Some(entry) = hit {
            crate::ktrace::log_fmt(format_args!(
                "browser:loader cache hit {} ({}) {}b",
                req.url,
                req.destination.as_str(),
                entry.body.len()
            ));
            return Ok(resource_from_entry(entry, req.destination, true, 0));
        }
        if req.cache_mode == CacheMode::OnlyIfCached {
            return Err(format!("only-if-cached miss: {}", req.url));
        }
    }

    // ── network ───────────────────────────────────────────────────────────
    crate::ktrace::log_fmt(format_args!(
        "browser:loader fetch {} ({})",
        req.url,
        req.destination.as_str()
    ));

    // CORS preflight for non-simple CORS requests (Fetch + Ladybird).
    if req.cors_mode == RequestMode::Cors {
        let initiator = req
            .initiator_origin
            .as_deref()
            .map(|s| Origin {
                serialized: s.to_string(),
            })
            .unwrap_or_else(Origin::null);
        let extra: Vec<(String, String)> = Vec::new();
        if cors::needs_preflight(&req.method, &extra) {
            let pkey = cors::PreflightCache::key(
                &initiator.serialized,
                &req.url,
                &req.method,
            );
            let cached = cors::PREFLIGHT_CACHE.with(|c| c.allows(&pkey));
            if !cached {
                let ph = cors::preflight_request_headers(
                    &initiator,
                    &req.method,
                    &extra,
                    req.credentials,
                );
                let pref: Vec<(&str, &str)> =
                    ph.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                match crate::net::http::request(
                    "OPTIONS",
                    &req.url,
                    &pref,
                    &[],
                    req.timeout_ms.min(15_000),
                ) {
                    Ok(resp) => {
                        match cors::check_preflight(
                            &initiator,
                            &resp.headers,
                            &req.method,
                            &extra,
                            req.credentials,
                        ) {
                            cors::PreflightResult::Ok => {
                                let max_age = resp
                                    .headers
                                    .iter()
                                    .find(|(k, _)| {
                                        k.eq_ignore_ascii_case("access-control-max-age")
                                    })
                                    .and_then(|(_, v)| v.parse().ok())
                                    .unwrap_or(600u64);
                                cors::PREFLIGHT_CACHE.with(|c| c.insert(pkey, max_age));
                            }
                            cors::PreflightResult::Failed(m) => {
                                return Err(format!("CORS preflight: {m}"));
                            }
                        }
                    }
                    Err(e) => {
                        // Soft-fail preflight network errors only when method is simple GET.
                        if !req.method.eq_ignore_ascii_case("GET")
                            && !req.method.eq_ignore_ascii_case("HEAD")
                        {
                            return Err(format!("CORS preflight network: {e}"));
                        }
                    }
                }
            }
        }
    }

    // Attach cookies for this URL.
    let cookie_hdr = {
        let is_https = req.url.starts_with("https://");
        super::storage::STORAGE.with(|s| {
            s.cookies()
                .cookie_header_for(&req.url, now_ms(), is_https)
        })
    };

    let t0 = now_ms();
    let got = if cookie_hdr.is_empty() {
        crate::net::http::get_follow(&req.url, req.timeout_ms)?
    } else {
        // One-shot GET with Cookie (no multi-hop cookie update mid-redirect for now).
        let headers = [("Cookie", cookie_hdr.as_str())];
        let resp = crate::net::http::request("GET", &req.url, &headers, &[], req.timeout_ms)?;
        crate::net::http::FollowedGet {
            response: resp,
            final_url: req.url.clone(),
            redirects: 0,
        }
    };
    let t1 = now_ms();

    let status = got.response.status;
    let redirects = got.redirects;
    let final_url = got.final_url;
    let headers = got.response.headers;
    let mut body = got.response.body;
    // Soft cap for documents is applied by the shell; here only a hard safety
    // bound so one giant asset cannot OOM the heap.
    const HARD_MAX: usize = 16 * 1024 * 1024;
    if body.len() > HARD_MAX {
        body.truncate(HARD_MAX);
    }

    let content_type = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| mime_from_content_type(v))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| sniff_mime(&body).to_string());

    // CORS gate (Ladybird Fetch CORS check on response).
    let initiator = req
        .initiator_origin
        .as_deref()
        .map(|s| Origin {
            serialized: s.to_string(),
        })
        .or_else(|| {
            req.source_url
                .as_deref()
                .map(Origin::from_url)
        })
        .unwrap_or_else(Origin::null);
    let cors = cors::check_response(
        req.cors_mode,
        &initiator,
        &final_url,
        &headers,
        req.credentials,
    );
    let (body, cors_opaque) = match cors {
        CorsResult::Ok => (body, false),
        CorsResult::Opaque => {
            // Opaque: body not exposed to JS; keep for image/no-cors paint.
            (body, true)
        }
        CorsResult::Blocked(msg) => {
            return Err(format!("CORS blocked: {msg}"));
        }
    };

    let loaded = LoadedResource {
        url: final_url.clone(),
        status,
        content_type: content_type.clone(),
        headers: headers.clone(),
        body: body.clone(),
        from_cache: false,
        redirects,
        destination: req.destination,
        cors_opaque,
    };

    // Store Set-Cookie headers into the jar.
    for (k, v) in &headers {
        if k.eq_ignore_ascii_case("set-cookie") {
            if let Some((_, host, path)) = super::url::split_http(&final_url) {
                let host = host.split(':').next().unwrap_or(&host);
                let now = now_ms();
                super::storage::STORAGE.with(|s| {
                    s.cookies().set_from_header(v, host, &path, now);
                });
            }
        }
    }

    if req.cache_mode.permits_store() && status < 400 && !cors_opaque {
        let entry = entry_from_response(
            &final_url,
            method,
            status,
            &headers,
            body,
            t0,
            t1,
        );
        // Also key under the request URL if redirects rewrote it, so the next
        // open_entry on the original URL hits (Ladybird keys on final URI;
        // for navigation UX we store both).
        MEMORY_CACHE.with(|c| {
            c.put(entry.clone());
            if final_url != req.url {
                let mut e2 = entry;
                e2.url = req.url.clone();
                c.put(e2);
            }
        });
    }

    Ok(loaded)
}

fn resource_from_entry(
    entry: Entry,
    destination: Destination,
    from_cache: bool,
    redirects: u32,
) -> LoadedResource {
    LoadedResource {
        url: entry.url,
        status: entry.status,
        content_type: entry.content_type,
        headers: entry.headers,
        body: entry.body,
        from_cache,
        redirects,
        destination,
        cors_opaque: false,
    }
}

/// Load into an [`AssetStore`] (page session assets). Returns whether network
/// was used (`false` = cache or asset store hit).
pub fn load_into_assets(
    store: &mut AssetStore,
    req: &LoadRequest,
) -> Result<LoadedResource, String> {
    if let Some((ct, body)) = store.get(&req.url) {
        return Ok(LoadedResource {
            url: req.url.clone(),
            status: 200,
            content_type: ct.to_string(),
            headers: Vec::new(),
            body: body.to_vec(),
            from_cache: true,
            redirects: 0,
            destination: req.destination,
            cors_opaque: false,
        });
    }
    let loaded = load(req)?;
    if loaded.status < 400 {
        store.put(&loaded.url, &loaded.content_type, loaded.body.clone());
        if loaded.url != req.url {
            store.put(&req.url, &loaded.content_type, loaded.body.clone());
        }
    }
    Ok(loaded)
}

/// Convenience: document navigation with optional hard reload.
pub fn load_document(url: &str, reload: bool) -> Result<LoadedResource, String> {
    let mode = if reload {
        CacheMode::Reload
    } else {
        CacheMode::Default
    };
    load(&LoadRequest::document(url).with_cache_mode(mode))
}

/// Convenience: image subresource.
pub fn load_image(url: &str) -> Result<LoadedResource, String> {
    load(&LoadRequest::image(url))
}

/// Cache statistics for diagnostics (`browser` status / tests).
pub fn cache_stats() -> (usize, usize, u64, u64) {
    MEMORY_CACHE.with(|c| (c.len(), c.total_bytes(), c.hits(), c.misses()))
}

/// Pure helper: choose content type from headers + body (no I/O).
pub fn resolve_content_type(headers: &[(String, String)], body: &[u8]) -> String {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| mime_from_content_type(v))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| sniff_mime(body).to_string())
}

/// Pure: decide whether a LoadRequest mode should attempt network after a miss.
pub fn should_network(mode: CacheMode, had_hit: bool) -> bool {
    if had_hit {
        return false;
    }
    mode != CacheMode::OnlyIfCached
}

// Re-export cache mode for callers that only import loader.
pub use cache::CacheMode as Mode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::cache::{entry_from_response, CacheMode, MemoryCache};

    #[test_case]
    fn load_request_builders() {
        let d = LoadRequest::document("https://ex.com/");
        assert_eq!(d.destination, Destination::Document);
        assert_eq!(d.cache_mode, CacheMode::Default);
        let i = LoadRequest::image("https://ex.com/a.png").with_cache_mode(CacheMode::ForceCache);
        assert_eq!(i.destination, Destination::Image);
        assert_eq!(i.cache_mode, CacheMode::ForceCache);
        assert_eq!(i.timeout_ms, 20_000);
    }

    #[test_case]
    fn resolve_content_type_header_and_sniff() {
        let headers = alloc::vec![(
            String::from("Content-Type"),
            String::from("text/html; charset=UTF-8")
        )];
        assert_eq!(resolve_content_type(&headers, b"xxx"), "text/html");
        assert_eq!(
            resolve_content_type(&[], &[0xff, 0xd8, 0xff, 0xdb]),
            "image/jpeg"
        );
    }

    #[test_case]
    fn should_network_matrix() {
        assert!(should_network(CacheMode::Default, false));
        assert!(!should_network(CacheMode::Default, true));
        assert!(!should_network(CacheMode::OnlyIfCached, false));
        assert!(should_network(CacheMode::Reload, false));
    }

    #[test_case]
    fn resource_from_cache_entry_fields() {
        let e = entry_from_response(
            "https://ex.com/",
            "GET",
            200,
            &[(String::from("content-type"), String::from("text/html"))],
            b"<html>".to_vec(),
            0,
            0,
        );
        let mut c = MemoryCache::new(8, 1024 * 1024);
        c.put(e);
        let hit = c
            .open_entry("https://ex.com/", "GET", CacheMode::Default, 1)
            .unwrap();
        let r = resource_from_entry(hit, Destination::Document, true, 0);
        assert!(r.from_cache);
        assert_eq!(r.status, 200);
        assert_eq!(r.content_type, "text/html");
        assert_eq!(r.body, b"<html>");
    }

    #[test_case]
    fn load_into_assets_session_hit() {
        let mut store = AssetStore::new();
        store.put("https://ex.com/a.png", "image/png", b"\x89PNG".to_vec());
        let req = LoadRequest::image("https://ex.com/a.png");
        // Session hit path does not need network.
        let got = load_into_assets(&mut store, &req).unwrap();
        assert!(got.from_cache);
        assert_eq!(got.body, b"\x89PNG");
        assert_eq!(got.content_type, "image/png");
    }

    #[test_case]
    fn destination_labels() {
        assert_eq!(Destination::Worker.as_str(), "worker");
        assert_eq!(Destination::Style.as_str(), "style");
    }
}
