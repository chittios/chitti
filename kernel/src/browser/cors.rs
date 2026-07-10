//! **CORS** (Cross-Origin Resource Sharing) — pure checks.
//!
//! Reference (not linked):
//! - Fetch living standard CORS protocol
//! - Ladybird `Libraries/LibWeb/Fetch/Infrastructure/HTTP/CORS.{h,cpp}`
//! - Ladybird `Libraries/LibWeb/HTML/CORSSettingAttribute.*`
//! - MDN: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
//!
//! We implement the **response check** + request mode matrix used by the
//! browser loader and `fetch()`. Full preflight (OPTIONS) is a future step;
//! simple requests (GET/HEAD/POST + safelisted headers) validate
//! `Access-Control-Allow-Origin` / `Credentials` against the initiator origin.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Fetch request mode (Fetch standard / Ladybird Request mode subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestMode {
    /// Same-origin only; cross-origin fails.
    SameOrigin,
    /// Opaque response for cross-origin (no CORS headers required).
    NoCors,
    /// CORS protocol; response must allow origin.
    Cors,
    /// Navigate documents (no CORS filter on main document load).
    Navigate,
}

/// Result of applying CORS to a completed HTTP response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CorsResult {
    /// Usable by the caller (same-origin or CORS-allowed).
    Ok,
    /// Opaque filtered response (no-cors cross-origin): body may be kept but
    /// JS sees opaque; we mark `opaque` for the host.
    Opaque,
    /// Blocked (CORS failure or same-origin mode on cross-origin).
    Blocked(String),
}

/// Tuple origin `scheme://host` (port included in host if non-default).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    pub serialized: String,
}

impl Origin {
    pub fn null() -> Self {
        Self {
            serialized: String::from("null"),
        }
    }

    pub fn from_url(url: &str) -> Self {
        super::url::origin(url).map(|s| Self { serialized: s }).unwrap_or_else(Self::null)
    }

    pub fn is_null(&self) -> bool {
        self.serialized == "null"
    }

    pub fn same(&self, other: &Origin) -> bool {
        !self.is_null() && self.serialized == other.serialized
    }
}

/// True if method is CORS-safelisted (Fetch).
pub fn is_cors_safelisted_method(method: &str) -> bool {
    matches!(
        method.to_ascii_uppercase().as_str(),
        "GET" | "HEAD" | "POST"
    )
}

/// Safelisted request header names (Fetch CORS-safelisted + client hints subset).
pub fn is_cors_safelisted_request_header_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "accept"
            | "accept-language"
            | "content-language"
            | "content-type"
            | "range"
            | "dpr"
            | "downlink"
            | "save-data"
            | "viewport-width"
            | "width"
    )
}

/// Content-Type values allowed on a simple CORS request.
pub fn is_cors_safelisted_content_type(value: &str) -> bool {
    let v = value
        .split(';')
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase();
    matches!(
        v.as_str(),
        "application/x-www-form-urlencoded" | "multipart/form-data" | "text/plain"
    )
}

/// Whether a simple (non-preflight) CORS request is possible for this method+headers.
pub fn is_simple_cors_request(method: &str, headers: &[(String, String)]) -> bool {
    if !is_cors_safelisted_method(method) {
        return false;
    }
    for (n, v) in headers {
        if !is_cors_safelisted_request_header_name(n) {
            return false;
        }
        if n.eq_ignore_ascii_case("content-type") && !is_cors_safelisted_content_type(v) {
            return false;
        }
    }
    true
}

/// Check a response against initiator origin and mode (Ladybird-style gate).
pub fn check_response(
    mode: RequestMode,
    initiator: &Origin,
    response_url: &str,
    response_headers: &[(String, String)],
    credentials: bool,
) -> CorsResult {
    let resp_origin = Origin::from_url(response_url);
    let cross = !initiator.same(&resp_origin);

    match mode {
        RequestMode::Navigate => CorsResult::Ok,
        RequestMode::SameOrigin => {
            if cross {
                CorsResult::Blocked(alloc::format!(
                    "same-origin mode blocked cross-origin {} → {}",
                    initiator.serialized,
                    resp_origin.serialized
                ))
            } else {
                CorsResult::Ok
            }
        }
        RequestMode::NoCors => {
            if cross {
                CorsResult::Opaque
            } else {
                CorsResult::Ok
            }
        }
        RequestMode::Cors => {
            if !cross {
                return CorsResult::Ok;
            }
            // Cross-origin: require ACAO
            let acao = header_get(response_headers, "access-control-allow-origin");
            let Some(acao) = acao else {
                return CorsResult::Blocked(String::from(
                    "missing Access-Control-Allow-Origin",
                ));
            };
            let acao = acao.trim();
            if acao == "*" {
                if credentials {
                    return CorsResult::Blocked(String::from(
                        "ACAO * not allowed with credentials",
                    ));
                }
                return CorsResult::Ok;
            }
            if acao == initiator.serialized {
                if credentials {
                    let acac = header_get(response_headers, "access-control-allow-credentials");
                    if acac.map(|s| s.eq_ignore_ascii_case("true")).unwrap_or(false) {
                        return CorsResult::Ok;
                    }
                    return CorsResult::Blocked(String::from(
                        "missing Access-Control-Allow-Credentials",
                    ));
                }
                return CorsResult::Ok;
            }
            CorsResult::Blocked(alloc::format!(
                "ACAO {acao} does not match origin {}",
                initiator.serialized
            ))
        }
    }
}

fn header_get<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Headers the loader should add for a CORS request.
pub fn request_headers_for_cors(
    mode: RequestMode,
    initiator: &Origin,
) -> Vec<(String, String)> {
    let mut h = Vec::new();
    if matches!(mode, RequestMode::Cors) && !initiator.is_null() {
        h.push((String::from("Origin"), initiator.serialized.clone()));
    }
    h
}

/// Default mode for a destination (Ladybird PotentialCORSRequest spirit).
pub fn mode_for_destination(dest: &str) -> RequestMode {
    match dest {
        "document" => RequestMode::Navigate,
        "image" | "style" | "script" | "font" | "worker" => RequestMode::NoCors,
        "fetch" | "iframe" => RequestMode::Cors,
        _ => RequestMode::Cors,
    }
}

// ── Preflight (OPTIONS) ───────────────────────────────────────────────────

/// Whether a CORS request needs a preflight (non-simple method or headers).
pub fn needs_preflight(method: &str, headers: &[(String, String)]) -> bool {
    !is_simple_cors_request(method, headers)
}

/// Result of validating an OPTIONS preflight response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreflightResult {
    Ok,
    Failed(String),
}

/// Check Access-Control-Allow-Methods / Allow-Headers on a preflight response.
/// When `credentials` is true, ACAO must be an exact origin (not `*`) and
/// Access-Control-Allow-Credentials: true is required.
pub fn check_preflight(
    initiator: &Origin,
    response_headers: &[(String, String)],
    request_method: &str,
    request_headers: &[(String, String)],
    credentials: bool,
) -> PreflightResult {
    let acao = response_headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-origin"))
        .map(|(_, v)| v.as_str().trim());
    match acao {
        Some("*") if credentials => {
            return PreflightResult::Failed(String::from(
                "credentialed preflight cannot use ACAO *",
            ));
        }
        Some("*") => {}
        Some(o) if o == initiator.serialized => {}
        Some(o) => {
            return PreflightResult::Failed(alloc::format!("preflight ACAO {o}"));
        }
        None => {
            return PreflightResult::Failed(String::from(
                "preflight missing Access-Control-Allow-Origin",
            ));
        }
    }
    if credentials {
        let acac = response_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-credentials"))
            .map(|(_, v)| v.as_str());
        if !acac.map(|s| s.eq_ignore_ascii_case("true")).unwrap_or(false) {
            return PreflightResult::Failed(String::from(
                "credentialed preflight missing Allow-Credentials",
            ));
        }
    }
    // Empty Allow-Methods: default allows GET/HEAD/POST only (Fetch).
    let methods = response_headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-methods"))
        .map(|(_, v)| v.as_str());
    match methods {
        None | Some("") => {
            if !is_cors_safelisted_method(request_method) {
                return PreflightResult::Failed(alloc::format!(
                    "method {request_method} not allowed (no Allow-Methods)"
                ));
            }
        }
        Some("*") => {}
        Some(methods) => {
            let ok = methods
                .split(',')
                .any(|m| m.trim().eq_ignore_ascii_case(request_method));
            if !ok {
                return PreflightResult::Failed(alloc::format!(
                    "method {request_method} not in Allow-Methods"
                ));
            }
        }
    }
    // Full request header list must be covered by Allow-Headers.
    let allowed_h = response_headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("access-control-allow-headers"))
        .map(|(_, v)| v.as_str());
    let mut unsafe_names: Vec<String> = request_headers
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| !is_cors_safelisted_request_header_name(n))
        .collect();
    // Case-insensitive unique
    unsafe_names.sort_by(|a, b| a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase()));
    unsafe_names.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    match allowed_h {
        None | Some("") if !unsafe_names.is_empty() => {
            return PreflightResult::Failed(String::from(
                "preflight missing Allow-Headers for non-safelisted headers",
            ));
        }
        Some("*") => {}
        Some(allowed_h) => {
            for n in &unsafe_names {
                let ok = allowed_h
                    .split(',')
                    .any(|h| h.trim().eq_ignore_ascii_case(n));
                if !ok {
                    return PreflightResult::Failed(alloc::format!("header {n} not allowed"));
                }
            }
        }
        _ => {}
    }
    PreflightResult::Ok
}

/// Build OPTIONS request headers for preflight (full non-safelisted list).
pub fn preflight_request_headers(
    initiator: &Origin,
    method: &str,
    headers: &[(String, String)],
    credentials: bool,
) -> Vec<(String, String)> {
    let mut h = Vec::new();
    if !initiator.is_null() {
        h.push((String::from("Origin"), initiator.serialized.clone()));
    }
    h.push((
        String::from("Access-Control-Request-Method"),
        method.to_ascii_uppercase(),
    ));
    let mut names: Vec<String> = headers
        .iter()
        .filter(|(n, _)| !is_cors_safelisted_request_header_name(n))
        .map(|(n, _)| n.clone())
        .collect();
    names.sort_by(|a, b| a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase()));
    names.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    if !names.is_empty() {
        h.push((
            String::from("Access-Control-Request-Headers"),
            names.join(", "),
        ));
    }
    if credentials {
        // Fetch: credentials mode include → cookies on actual request; preflight
        // itself is without cookies typically, but we mark intent.
        h.push((String::from("Access-Control-Request-Credentials"), String::from("true")));
    }
    h
}

/// In-memory preflight cache key → max-age seconds remaining (logical).
#[derive(Clone, Debug, Default)]
pub struct PreflightCache {
    /// key = origin|url|method
    map: alloc::collections::BTreeMap<String, u64>,
}

impl PreflightCache {
    pub fn key(origin: &str, url: &str, method: &str) -> String {
        alloc::format!("{origin}|{url}|{}", method.to_ascii_uppercase())
    }

    pub fn insert(&mut self, key: String, max_age_secs: u64) {
        self.map.insert(key, max_age_secs);
    }

    pub fn allows(&self, key: &str) -> bool {
        self.map.get(key).copied().unwrap_or(0) > 0
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

pub static PREFLIGHT_CACHE: crate::mm::Locked<PreflightCache> =
    crate::mm::Locked::new(PreflightCache {
        map: alloc::collections::BTreeMap::new(),
    });

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    #[test_case]
    fn same_origin_ok() {
        let o = Origin::from_url("https://a.com/x");
        assert_eq!(o.serialized, "https://a.com");
        let r = check_response(
            RequestMode::Cors,
            &o,
            "https://a.com/y",
            &[],
            false,
        );
        assert_eq!(r, CorsResult::Ok);
    }

    #[test_case]
    fn cors_star_and_exact() {
        let o = Origin::from_url("https://app.ex/");
        let headers = vec![(
            String::from("Access-Control-Allow-Origin"),
            String::from("*"),
        )];
        assert_eq!(
            check_response(RequestMode::Cors, &o, "https://api.ex/d", &headers, false),
            CorsResult::Ok
        );
        assert!(matches!(
            check_response(RequestMode::Cors, &o, "https://api.ex/d", &headers, true),
            CorsResult::Blocked(_)
        ));
        let headers2 = vec![(
            String::from("Access-Control-Allow-Origin"),
            String::from("https://app.ex"),
        )];
        assert_eq!(
            check_response(RequestMode::Cors, &o, "https://api.ex/d", &headers2, false),
            CorsResult::Ok
        );
    }

    #[test_case]
    fn no_cors_opaque() {
        let o = Origin::from_url("https://a.com/");
        let r = check_response(RequestMode::NoCors, &o, "https://b.com/i.png", &[], false);
        assert_eq!(r, CorsResult::Opaque);
    }

    #[test_case]
    fn same_origin_mode_blocks() {
        let o = Origin::from_url("https://a.com/");
        assert!(matches!(
            check_response(RequestMode::SameOrigin, &o, "https://b.com/", &[], false),
            CorsResult::Blocked(_)
        ));
    }

    #[test_case]
    fn simple_request_helpers() {
        assert!(is_cors_safelisted_method("get"));
        assert!(!is_cors_safelisted_method("PUT"));
        assert!(is_simple_cors_request("POST", &[]));
        assert!(!is_simple_cors_request(
            "POST",
            &[(String::from("X-Custom"), String::from("1"))]
        ));
    }

    #[test_case]
    fn preflight_needs_and_check() {
        assert!(needs_preflight("PUT", &[]));
        assert!(needs_preflight(
            "POST",
            &[(String::from("X-Token"), String::from("a"))]
        ));
        assert!(!needs_preflight("GET", &[]));
        let o = Origin::from_url("https://app.ex/");
        let headers = vec![
            (
                String::from("Access-Control-Allow-Origin"),
                String::from("https://app.ex"),
            ),
            (
                String::from("Access-Control-Allow-Methods"),
                String::from("PUT, POST"),
            ),
            (
                String::from("Access-Control-Allow-Headers"),
                String::from("X-Token"),
            ),
        ];
        assert_eq!(
            check_preflight(
                &o,
                &headers,
                "PUT",
                &[(String::from("X-Token"), String::from("1"))],
                false,
            ),
            PreflightResult::Ok
        );
        assert!(matches!(
            check_preflight(&o, &headers, "DELETE", &[], false),
            PreflightResult::Failed(_)
        ));
        // Credentialed: * ACAO fails
        let star = vec![(
            String::from("Access-Control-Allow-Origin"),
            String::from("*"),
        )];
        assert!(matches!(
            check_preflight(&o, &star, "GET", &[], true),
            PreflightResult::Failed(_)
        ));
        let cred = vec![
            (
                String::from("Access-Control-Allow-Origin"),
                String::from("https://app.ex"),
            ),
            (
                String::from("Access-Control-Allow-Credentials"),
                String::from("true"),
            ),
            (
                String::from("Access-Control-Allow-Methods"),
                String::from("GET"),
            ),
        ];
        assert_eq!(
            check_preflight(&o, &cred, "GET", &[], true),
            PreflightResult::Ok
        );
    }
}
