//! URL helpers for the browser agent: absolute HTTP(S) / `file:///` checks and
//! relative `href` resolution against a base URL. Pure — no network.

use alloc::format;
use alloc::string::{String, ToString};

/// True if `url` is an absolute `http://` or `https://` URL.
pub fn is_http_url(url: &str) -> bool {
    let u = url.trim();
    u.starts_with("http://") || u.starts_with("https://")
}

/// True if `url` is an absolute `file:` URL (`file:///path` or `file://localhost/path`).
pub fn is_file_url(url: &str) -> bool {
    url.trim().starts_with("file:")
}

/// True if `/browse` can open this URL (http(s) or local `file:`).
pub fn is_browse_url(url: &str) -> bool {
    is_http_url(url) || is_file_url(url)
}

/// Extract the store/mount path from a `file:` URL.
///
/// - `file:///samples/html/x.html` → `/samples/html/x.html`
/// - `file://localhost/samples/x` → `/samples/x`
/// - `file:/samples/x` → `/samples/x`
///
/// Returns `None` if not a file URL or the path is empty.
pub fn file_path(url: &str) -> Option<String> {
    let u = url.trim();
    let rest = u.strip_prefix("file:")?;
    // Optional `//` authority.
    let path = if let Some(after) = rest.strip_prefix("//") {
        // `file:///abs` → after is `/abs`; `file://localhost/abs` → drop host.
        if after.starts_with('/') {
            after
        } else {
            // host[/path]
            after.find('/').map(|i| &after[i..]).unwrap_or("/")
        }
    } else if rest.starts_with('/') {
        rest
    } else {
        return Some(format!("/{rest}"));
    };
    let path = path.split('#').next().unwrap_or(path);
    let path = path.split('?').next().unwrap_or(path);
    if path.is_empty() {
        return None;
    }
    // Ensure leading slash.
    if path.starts_with('/') {
        Some(path.to_string())
    } else {
        Some(format!("/{path}"))
    }
}

/// Build a `file:///` URL from an absolute store path (`/samples/…`).
pub fn file_url_from_path(path: &str) -> String {
    let p = path.trim();
    if p.starts_with("file:") {
        return p.split('#').next().unwrap_or(p).to_string();
    }
    let p = if p.starts_with('/') { p } else { return format!("file:///{p}") };
    // `file://` + `/samples/...` → `file:///samples/...`
    format!("file://{p}")
}

/// Normalize a `/browse` argument: accept `file:///…`, absolute store paths
/// (`/samples/html/…`), and http(s).
pub fn normalize_browse_arg(url: &str) -> String {
    let u = url.trim();
    if u.is_empty() {
        return String::new();
    }
    if is_http_url(u) || is_file_url(u) {
        return u.split('#').next().unwrap_or(u).to_string();
    }
    if u.starts_with('/') {
        return file_url_from_path(u);
    }
    u.to_string()
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

/// Resolve `href` against absolute `base` (http/https **or** `file:///`).
/// Handles absolute href, protocol-relative `//host/…`, root-relative `/…`,
/// and path-relative.
pub fn resolve(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') || href.starts_with("javascript:") {
        return None;
    }
    if is_http_url(href) || is_file_url(href) {
        return Some(href.split('#').next().unwrap_or(href).to_string());
    }
    // file:/// base — keep everything on the local store.
    if is_file_url(base) {
        let base_path = file_path(base)?;
        if href.starts_with('/') {
            let p = href.split('#').next().unwrap_or(href);
            return Some(file_url_from_path(p));
        }
        let dir = match base_path.rfind('/') {
            Some(i) => &base_path[..=i],
            None => "/",
        };
        let mut joined = String::from(dir);
        joined.push_str(href.split('#').next().unwrap_or(href));
        let norm = normalize_path(&joined);
        return Some(file_url_from_path(&norm));
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

    #[test_case]
    fn file_url_path_and_resolve() {
        assert!(is_file_url("file:///samples/html/x.html"));
        assert!(!is_browse_url("ftp://x"));
        assert!(is_browse_url("file:///samples/html/x.html"));
        assert_eq!(
            file_path("file:///samples/html/x.html").as_deref(),
            Some("/samples/html/x.html")
        );
        assert_eq!(
            file_path("file://localhost/samples/a").as_deref(),
            Some("/samples/a")
        );
        assert_eq!(
            normalize_browse_arg("/samples/html/index.html"),
            "file:///samples/html/index.html"
        );
        assert_eq!(
            resolve("file:///samples/html/index.html", "styles.css"),
            Some("file:///samples/html/styles.css".into())
        );
        assert_eq!(
            resolve("file:///samples/html/index.html", "/samples/js/hello.js"),
            Some("file:///samples/js/hello.js".into())
        );
        assert_eq!(
            resolve("file:///samples/html/js-demo.html", "./app.js"),
            Some("file:///samples/html/app.js".into())
        );
    }
}
