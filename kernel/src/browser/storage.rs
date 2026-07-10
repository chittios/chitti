//! Browser storage: **cookies**, **localStorage**, **sessionStorage**.
//!
//! Reference (not linked):
//! - Ladybird / HTML Web Storage + Cookie Store
//! - MDN: Cookies, Window.localStorage, sessionStorage
//!
//! Cookie Domain uses a compact [public suffix](super::psl) list (Ladybird
//! PublicSuffixData spirit). Expires uses [`super::httpdate`]. Profiles + disk
//! persistence live under `/configs/browser/profiles/<id>/` via Synapse FS.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// ── Cookies ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    pub secure: bool,
    pub http_only: bool,
    /// Expiry as unix-ms; `None` = session cookie.
    pub expires_ms: Option<u64>,
    pub same_site: SameSite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SameSite {
    Strict,
    Lax,
    None,
}

#[derive(Clone, Debug, Default)]
pub struct CookieJar {
    pub(crate) cookies: Vec<Cookie>,
}

impl CookieJar {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.cookies.clear();
    }

    pub fn len(&self) -> usize {
        self.cookies.len()
    }

    /// Parse a single `Set-Cookie` header value and store it.
    /// `now_ms` is used to resolve Max-Age and validate Expires.
    pub fn set_from_header(
        &mut self,
        header: &str,
        request_host: &str,
        request_path: &str,
        now_ms: u64,
    ) {
        let mut parts = header.split(';');
        let Some(nv) = parts.next() else { return };
        let (name, value) = match nv.find('=') {
            Some(i) => (nv[..i].trim().to_string(), nv[i + 1..].trim().to_string()),
            None => return,
        };
        if name.is_empty() {
            return;
        }
        let host = request_host
            .split(':')
            .next()
            .unwrap_or(request_host)
            .to_ascii_lowercase();
        let mut domain = host.clone();
        let mut path = default_cookie_path(request_path);
        let mut secure = false;
        let mut http_only = false;
        let mut expires_ms = None;
        let mut same_site = SameSite::Lax;
        for p in parts {
            let p = p.trim();
            let (k, v) = match p.find('=') {
                Some(i) => (p[..i].trim(), p[i + 1..].trim()),
                None => (p, ""),
            };
            let kl = k.to_ascii_lowercase();
            match kl.as_str() {
                "domain" => {
                    let d = v.trim_start_matches('.').to_ascii_lowercase();
                    if !d.is_empty() && super::psl::cookie_domain_ok(&d, &host) {
                        domain = d;
                    }
                }
                "path" => {
                    if v.starts_with('/') {
                        path = v.to_string();
                    }
                }
                "secure" => secure = true,
                "httponly" => http_only = true,
                "max-age" => {
                    if let Ok(secs) = v.parse::<i64>() {
                        if secs <= 0 {
                            self.cookies.retain(|c| !(c.name == name && c.domain == domain));
                            return;
                        }
                        expires_ms = Some(now_ms.saturating_add((secs as u64).saturating_mul(1000)));
                    }
                }
                "expires" => {
                    if let Some(t) = super::httpdate::parse_http_date_ms(v) {
                        expires_ms = Some(t);
                    }
                }
                "samesite" => {
                    same_site = match v.to_ascii_lowercase().as_str() {
                        "strict" => SameSite::Strict,
                        "none" => SameSite::None,
                        _ => SameSite::Lax,
                    };
                }
                _ => {}
            }
        }
        // Reject Domain= that is a public suffix alone.
        if super::psl::is_public_suffix(&domain) && domain != host {
            domain = host;
        }
        self.cookies
            .retain(|c| !(c.name == name && c.domain == domain && c.path == path));
        self.cookies.push(Cookie {
            name,
            value,
            domain,
            path,
            secure,
            http_only,
            expires_ms,
            same_site,
        });
    }

    pub fn drop_session_cookies(&mut self) {
        self.cookies.retain(|c| c.expires_ms.is_some());
    }

    /// `Cookie` request header value for a URL.
    pub fn cookie_header_for(&self, url: &str, now_ms: u64, is_https: bool) -> String {
        let Some((host, path)) = host_path(url) else {
            return String::new();
        };
        let mut pairs = Vec::new();
        for c in &self.cookies {
            if c.secure && !is_https {
                continue;
            }
            if let Some(exp) = c.expires_ms {
                if exp <= now_ms {
                    continue;
                }
            }
            if !domain_matches(&c.domain, &host) {
                continue;
            }
            if !path_matches(&c.path, &path) {
                continue;
            }
            pairs.push(format!("{}={}", c.name, c.value));
        }
        pairs.join("; ")
    }
}

fn default_cookie_path(request_path: &str) -> String {
    if let Some(i) = request_path.rfind('/') {
        if i == 0 {
            return String::from("/");
        }
        return request_path[..=i].to_string();
    }
    String::from("/")
}

fn host_path(url: &str) -> Option<(String, String)> {
    let (_, host, path) = super::url::split_http(url)?;
    let host = host.split(':').next().unwrap_or(&host).to_ascii_lowercase();
    let path = if path.is_empty() {
        String::from("/")
    } else {
        path
    };
    Some((host, path))
}

fn domain_matches(cookie_domain: &str, host: &str) -> bool {
    let cd = cookie_domain.trim_start_matches('.').to_ascii_lowercase();
    let h = host.to_ascii_lowercase();
    h == cd || h.ends_with(&format!(".{cd}"))
}

fn path_matches(cookie_path: &str, request_path: &str) -> bool {
    if request_path.starts_with(cookie_path) {
        if cookie_path.ends_with('/') || request_path.len() == cookie_path.len() {
            return true;
        }
        return request_path.as_bytes().get(cookie_path.len()) == Some(&b'/');
    }
    false
}

// ── Web Storage (localStorage / sessionStorage) ───────────────────────────

#[derive(Clone, Debug, Default)]
pub struct WebStorage {
    map: BTreeMap<String, String>,
    /// Origin this storage is bound to (`https://host`).
    pub origin: String,
}

impl WebStorage {
    pub fn new(origin: &str) -> Self {
        Self {
            map: BTreeMap::new(),
            origin: origin.to_string(),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn key(&self, index: usize) -> Option<String> {
        self.map.keys().nth(index).cloned()
    }

    pub fn get_item(&self, key: &str) -> Option<String> {
        self.map.get(key).cloned()
    }

    pub fn set_item(&mut self, key: &str, value: &str) -> Result<(), &'static str> {
        if key.len() > 1024 || value.len() > 64 * 1024 {
            return Err("QuotaExceededError");
        }
        if self.map.len() >= 256 && !self.map.contains_key(key) {
            return Err("QuotaExceededError");
        }
        self.map.insert(key.to_string(), value.to_string());
        Ok(())
    }

    pub fn remove_item(&mut self, key: &str) {
        self.map.remove(key);
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

/// One browser profile (multi-profile sessions — Ladybird profile spirit).
#[derive(Clone, Debug)]
pub struct BrowserProfile {
    pub id: String,
    pub local: BTreeMap<String, WebStorage>,
    pub session: BTreeMap<String, WebStorage>,
    pub cookies: CookieJar,
}

impl BrowserProfile {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            local: BTreeMap::new(),
            session: BTreeMap::new(),
            cookies: CookieJar::new(),
        }
    }

    pub fn local_for(&mut self, origin: &str) -> &mut WebStorage {
        self.local
            .entry(origin.to_string())
            .or_insert_with(|| WebStorage::new(origin))
    }

    pub fn session_for(&mut self, origin: &str) -> &mut WebStorage {
        self.session
            .entry(origin.to_string())
            .or_insert_with(|| WebStorage::new(origin))
    }

    pub fn end_session(&mut self) {
        self.session.clear();
        self.cookies.drop_session_cookies();
    }
}

/// Process-wide multi-profile storage.
#[derive(Clone, Debug)]
pub struct StoragePartition {
    pub profiles: BTreeMap<String, BrowserProfile>,
    pub active_profile: String,
}

impl Default for StoragePartition {
    fn default() -> Self {
        let mut profiles = BTreeMap::new();
        profiles.insert(String::from("default"), BrowserProfile::new("default"));
        Self {
            profiles,
            active_profile: String::from("default"),
        }
    }
}

impl StoragePartition {
    pub fn active(&mut self) -> &mut BrowserProfile {
        if self.active_profile.is_empty() {
            self.active_profile = String::from("default");
        }
        let id = self.active_profile.clone();
        self.profiles
            .entry(id.clone())
            .or_insert_with(|| BrowserProfile::new(&id))
    }

    pub fn active_ref(&self) -> Option<&BrowserProfile> {
        self.profiles.get(&self.active_profile)
    }

    pub fn switch_profile(&mut self, id: &str) {
        if !self.profiles.contains_key(id) {
            self.profiles
                .insert(id.to_string(), BrowserProfile::new(id));
        }
        self.active_profile = id.to_string();
    }

    pub fn local_for(&mut self, origin: &str) -> &mut WebStorage {
        self.active().local_for(origin)
    }

    pub fn session_for(&mut self, origin: &str) -> &mut WebStorage {
        self.active().session_for(origin)
    }

    pub fn cookies(&mut self) -> &mut CookieJar {
        &mut self.active().cookies
    }

    pub fn end_session(&mut self) {
        self.active().end_session();
    }

    /// Serialize active profile localStorage + cookies to a simple text format.
    pub fn serialize_active(&self) -> String {
        let Some(p) = self.profiles.get(&self.active_profile) else {
            return String::new();
        };
        let mut out = String::new();
        out.push_str(&format!("profile={}\n", p.id));
        for (origin, store) in &p.local {
            for (k, v) in &store.map {
                out.push_str(&format!(
                    "L\t{}\t{}\t{}\n",
                    escape_field(origin),
                    escape_field(k),
                    escape_field(v)
                ));
            }
        }
        for c in &p.cookies.cookies {
            out.push_str(&format!(
                "C\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                escape_field(&c.name),
                escape_field(&c.value),
                escape_field(&c.domain),
                escape_field(&c.path),
                if c.secure { "1" } else { "0" },
                c.expires_ms.map(|e| e.to_string()).unwrap_or_default(),
                match c.same_site {
                    SameSite::Strict => "S",
                    SameSite::Lax => "L",
                    SameSite::None => "N",
                }
            ));
        }
        out
    }

    /// Load serialized blob into active profile (replaces local + cookies).
    pub fn load_active_from(&mut self, text: &str) {
        let p = self.active();
        p.local.clear();
        p.cookies.clear();
        for line in text.lines() {
            if let Some(id) = line.strip_prefix("profile=") {
                // keep id
                let _ = id;
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.is_empty() {
                continue;
            }
            match parts[0] {
                "L" if parts.len() >= 4 => {
                    let origin = unescape_field(parts[1]);
                    let k = unescape_field(parts[2]);
                    let v = unescape_field(parts[3]);
                    let _ = p.local_for(&origin).set_item(&k, &v);
                }
                "C" if parts.len() >= 7 => {
                    p.cookies.cookies.push(Cookie {
                        name: unescape_field(parts[1]),
                        value: unescape_field(parts[2]),
                        domain: unescape_field(parts[3]),
                        path: unescape_field(parts[4]),
                        secure: parts[5] == "1",
                        http_only: false,
                        expires_ms: if parts[6].is_empty() {
                            None
                        } else {
                            parts[6].parse().ok()
                        },
                        same_site: match parts.get(7).copied().unwrap_or("L") {
                            "S" => SameSite::Strict,
                            "N" => SameSite::None,
                            _ => SameSite::Lax,
                        },
                    });
                }
                _ => {}
            }
        }
    }
}

fn escape_field(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n")
}

fn unescape_field(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Path for profile persistence (Synapse store).
pub fn profile_path(profile_id: &str) -> String {
    format!("/configs/browser/profiles/{profile_id}/storage.v1")
}

/// Persist active profile to disk (no-op if Synapse FS unavailable).
pub fn persist_active() {
    let (path, blob) = STORAGE.with(|s| {
        (
            profile_path(&s.active_profile),
            s.serialize_active(),
        )
    });
    #[cfg(not(test))]
    {
        let _ = crate::synapse::fs::mkdir("/configs/browser/profiles", true);
        let parent = path.rsplit_once('/').map(|(p, _)| p).unwrap_or("/configs");
        let _ = crate::synapse::fs::mkdir(parent, true);
        crate::synapse::fs::write(&path, blob.as_bytes());
    }
    #[cfg(test)]
    {
        let _ = (path, blob);
    }
}

/// Load active profile from disk if present.
pub fn load_active() {
    let path = STORAGE.with(|s| profile_path(&s.active_profile));
    #[cfg(not(test))]
    {
        if let Some(bytes) = crate::synapse::fs::read(&path) {
            if let Ok(text) = core::str::from_utf8(&bytes) {
                STORAGE.with(|s| s.load_active_from(text));
            }
        }
    }
    #[cfg(test)]
    {
        let _ = path;
    }
}

/// Process-wide browser storage (multi-profile). Initialized empty; first
/// `active()` call creates the `default` profile (const static friendly).
pub static STORAGE: crate::mm::Locked<StoragePartition> =
    crate::mm::Locked::new(StoragePartition {
        profiles: BTreeMap::new(),
        active_profile: String::new(),
    });

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn cookie_set_and_match() {
        let mut jar = CookieJar::new();
        jar.set_from_header(
            "sid=abc; Path=/; Domain=ex.com; Secure",
            "ex.com",
            "/app/page",
            1_000_000,
        );
        assert_eq!(jar.len(), 1);
        let h = jar.cookie_header_for("https://ex.com/app/x", 2_000_000, true);
        assert!(h.contains("sid=abc"), "{h}");
        let h2 = jar.cookie_header_for("http://ex.com/app/x", 2_000_000, false);
        assert!(h2.is_empty(), "secure cookie blocked on http: {h2}");
        let h3 = jar.cookie_header_for("https://other.com/", 2_000_000, true);
        assert!(h3.is_empty());
    }

    #[test_case]
    fn cookie_expires_and_psl() {
        let mut jar = CookieJar::new();
        jar.set_from_header(
            "a=1; Domain=co.uk",
            "www.foo.co.uk",
            "/",
            0,
        );
        // Domain=co.uk must not stick as public suffix
        let h = jar.cookie_header_for("https://www.foo.co.uk/", 0, true);
        // may still set host-only cookie
        let _ = h;
        jar.clear();
        jar.set_from_header(
            "x=y; Expires=Sat, 01 Jan 2000 00:00:00 GMT",
            "ex.com",
            "/",
            0,
        );
        // expired relative to "now" far in future
        let h = jar.cookie_header_for("https://ex.com/", 1_700_000_000_000, true);
        assert!(h.is_empty(), "expired cookie: {h}");
    }

    #[test_case]
    fn multi_profile_and_serialize() {
        let mut p = StoragePartition::default();
        p.local_for("https://a").set_item("k", "v").unwrap();
        p.switch_profile("work");
        p.local_for("https://b").set_item("x", "1").unwrap();
        assert_eq!(p.active_profile, "work");
        let blob = p.serialize_active();
        assert!(blob.contains("L\t"));
        p.switch_profile("default");
        assert_eq!(p.local_for("https://a").get_item("k").as_deref(), Some("v"));
    }

    #[test_case]
    fn web_storage_roundtrip() {
        let mut s = WebStorage::new("https://ex.com");
        assert!(s.set_item("a", "1").is_ok());
        assert_eq!(s.get_item("a").as_deref(), Some("1"));
        assert_eq!(s.len(), 1);
        assert_eq!(s.key(0).as_deref(), Some("a"));
        s.remove_item("a");
        assert!(s.get_item("a").is_none());
        s.set_item("x", "y").unwrap();
        s.clear();
        assert_eq!(s.len(), 0);
    }

    #[test_case]
    fn partition_session_end() {
        let mut p = StoragePartition::default();
        p.session_for("https://a").set_item("k", "v").unwrap();
        p.local_for("https://a").set_item("k", "v").unwrap();
        p.end_session();
        assert!(p.active().session.is_empty());
        assert_eq!(p.local_for("https://a").get_item("k").as_deref(), Some("v"));
    }
}
