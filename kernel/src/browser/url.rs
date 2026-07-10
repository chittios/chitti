//! URL helpers for the browser agent: absolute HTTP(S) checks and relative
//! `href` resolution against a base URL. Pure — no network.

use alloc::format;
use alloc::string::{String, ToString};

/// True if `url` is an absolute `http://` or `https://` URL.
pub fn is_http_url(url: &str) -> bool {
    let u = url.trim();
    u.starts_with("http://") || u.starts_with("https://")
}

/// Tuple origin `scheme://host[:port]` for CORS (no path). `None` if not http(s).
pub fn origin(url: &str) -> Option<String> {
    let (tls, host, _path) = split_http(url)?;
    let scheme = if tls { "https" } else { "http" };
    // Drop default ports for stable origin serialisation.
    let host = if tls && host.ends_with(":443") {
        host.trim_end_matches(":443").to_string()
    } else if !tls && host.ends_with(":80") {
        host.trim_end_matches(":80").to_string()
    } else {
        host
    };
    Some(format!("{scheme}://{host}"))
}

/// Same-origin check (scheme + host + port).
pub fn same_origin(a: &str, b: &str) -> bool {
    match (origin(a), origin(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Split `http[s]://host[:port]/path[?query][#frag]` into scheme host path.
/// Path includes query but not fragment. Returns `None` if not absolute http(s).
pub fn split_http(url: &str) -> Option<(bool, String, String)> {
    let u = url.trim();
    let (tls, rest) = if let Some(r) = u.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = u.strip_prefix("http://") {
        (false, r)
    } else {
        return None;
    };
    let rest = rest.split('#').next().unwrap_or(rest);
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if hostport.is_empty() {
        return None;
    }
    Some((tls, hostport.to_string(), path.to_string()))
}

/// Resolve `href` against absolute `base` (http/https). Handles absolute href,
/// protocol-relative `//host/…`, root-relative `/…`, and path-relative.
pub fn resolve(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') || href.starts_with("javascript:") {
        return None;
    }
    if is_http_url(href) {
        return Some(href.split('#').next().unwrap_or(href).to_string());
    }
    let (tls, host, base_path) = split_http(base)?;
    let scheme = if tls { "https" } else { "http" };
    if let Some(rest) = href.strip_prefix("//") {
        let rest = rest.split('#').next().unwrap_or(rest);
        return Some(format!("{scheme}://{rest}"));
    }
    if href.starts_with('/') {
        let p = href.split('#').next().unwrap_or(href);
        return Some(format!("{scheme}://{host}{p}"));
    }
    // Path-relative: join against base directory.
    let dir = match base_path.rfind('/') {
        Some(i) => &base_path[..=i],
        None => "/",
    };
    let mut joined = String::from(dir);
    joined.push_str(href.split('#').next().unwrap_or(href));
    // Collapse "/./" and simple ".." (best-effort).
    let norm = normalize_path(&joined);
    Some(format!("{scheme}://{host}{norm}"))
}

fn normalize_path(path: &str) -> String {
    let mut segs: alloc::vec::Vec<&str> = alloc::vec![];
    for s in path.split('/') {
        if s.is_empty() || s == "." {
            continue;
        }
        if s == ".." {
            segs.pop();
            continue;
        }
        segs.push(s);
    }
    let mut out = String::from("/");
    for (i, s) in segs.iter().enumerate() {
        if i > 0 {
            out.push('/');
        }
        out.push_str(s);
    }
    if path.ends_with('/') && out != "/" {
        out.push('/');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn absolute_and_relative_resolve() {
        assert!(is_http_url("https://ex.com/a"));
        assert!(!is_http_url("/a"));
        assert_eq!(
            resolve("https://ex.com/dir/page.html", "https://other/x"),
            Some("https://other/x".into())
        );
        assert_eq!(
            resolve("https://ex.com/dir/page.html", "/root"),
            Some("https://ex.com/root".into())
        );
        assert_eq!(
            resolve("https://ex.com/dir/page.html", "next.html"),
            Some("https://ex.com/dir/next.html".into())
        );
        assert_eq!(
            resolve("https://ex.com/dir/page.html", "../up"),
            Some("https://ex.com/up".into())
        );
        assert_eq!(
            resolve("http://ex.com/", "//cdn.ex/a.js"),
            Some("http://cdn.ex/a.js".into())
        );
        assert!(resolve("https://ex.com/", "#frag").is_none());
        assert_eq!(origin("https://ex.com:443/a"), Some("https://ex.com".into()));
        assert_eq!(origin("http://ex.com:80/"), Some("http://ex.com".into()));
        assert!(same_origin("https://a.com/x", "https://a.com/y"));
        assert!(!same_origin("https://a.com/", "https://b.com/"));
    }
}
