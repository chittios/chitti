//! Pure HTTP path → JSON response policy for the Doc agent.
//!
//! This is the SOUL routing table as deterministic code. Host only frames
//! bytes and reads the named `file` through the capability gate.

use alloc::format;
use alloc::string::{String, ToString};

/// Route one request. Returns a JSON response object string (same contract as
/// the model SOUL: `status`, optional `content_type`/`file`/`body`).
pub fn route_request(method: &str, path: &str) -> String {
    let method = method.trim();
    let path = normalize_path(path);

    // POST /echo — echo is body-less at the pipeline today; still a valid 200.
    if method.eq_ignore_ascii_case("POST") && path == "/echo" {
        return String::from(
            r#"{"status":200,"content_type":"text/plain; charset=utf-8","body":""}"#,
        );
    }

    // Static site map (was SOUL.md prose).
    match path.as_str() {
        "/" | "/index.html" => ok_file("index.html", "text/html; charset=utf-8"),
        "/docs" | "/docs/" | "/docs.html" => ok_file("docs.html", "text/html; charset=utf-8"),
        "/logo.svg" => ok_file("logo.svg", "image/svg+xml"),
        _ => String::from(r#"{"status":404}"#),
    }
}

fn ok_file(file: &str, ctype: &str) -> String {
    format!(
        r#"{{"status":200,"content_type":"{ctype}","file":"{file}"}}"#
    )
}

/// Strip query string, collapse trailing slash except for `/`.
fn normalize_path(path: &str) -> String {
    let path = path.split('?').next().unwrap_or(path).trim();
    if path.is_empty() {
        return String::from("/");
    }
    // Keep leading slash.
    let mut p = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    // /docs/ → /docs (but keep /)
    if p.len() > 1 && p.ends_with('/') {
        p.pop();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    // Host-side unit tests when built with std (not used in wasm cdylib).
    #[test]
    fn routes_home_docs_logo_and_404() {
        let r = route_request("GET", "/");
        assert!(r.contains("index.html"));
        let r = route_request("GET", "/docs");
        assert!(r.contains("docs.html"));
        let r = route_request("GET", "/logo.svg");
        assert!(r.contains("logo.svg"));
        let r = route_request("GET", "/nope");
        assert!(r.contains("404"));
    }
}
