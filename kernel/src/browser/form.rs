//! HTML form submission helpers (Ladybird `HTMLFormElement` / form-urlencoded).
//! Pure — no network. Host builds the request then uses `loader` / `http`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One successful name/value pair for application/x-www-form-urlencoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormField {
    pub name: String,
    pub value: String,
}

/// Encode fields as `a=1&b=2` (WHATWG application/x-www-form-urlencoded subset).
pub fn encode_urlencoded(fields: &[FormField]) -> String {
    let mut out = String::new();
    for (i, f) in fields.iter().enumerate() {
        if f.name.is_empty() {
            continue;
        }
        if i > 0 && !out.is_empty() {
            out.push('&');
        }
        out.push_str(&percent_encode(&f.name));
        out.push('=');
        out.push_str(&percent_encode(&f.value));
    }
    out
}

/// Percent-encode for form-urlencoded: space → `+`, unreserved pass through.
pub fn percent_encode(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                o.push(b as char);
            }
            b' ' => o.push('+'),
            _ => {
                o.push('%');
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                o.push(HEX[(b >> 4) as usize] as char);
                o.push(HEX[(b & 0xf) as usize] as char);
            }
        }
    }
    o
}

/// Decode a form-urlencoded component (`+` → space, `%HH`).
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h = hex_nibble(bytes[i + 1]);
                let l = hex_nibble(bytes[i + 2]);
                if let (Some(h), Some(l)) = (h, l) {
                    out.push((h << 4) | l);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Resolved form submission target.
#[derive(Clone, Debug)]
pub struct FormSubmit {
    pub method: String, // "GET" or "POST"
    pub url: String,
    /// Empty for GET (query is in `url`); body for POST.
    pub body: String,
    pub content_type: &'static str,
}

/// Build a submit request from action/method + fields (Ladybird form construct).
/// `base` is the document URL for resolving relative `action`.
pub fn build_submit(
    base: &str,
    action: &str,
    method: &str,
    fields: &[FormField],
) -> FormSubmit {
    let method = if method.eq_ignore_ascii_case("post") {
        "POST"
    } else {
        "GET"
    };
    let action = action.trim();
    let target = if action.is_empty() {
        // Submit to document URL without fragment.
        base.split('#').next().unwrap_or(base).to_string()
    } else {
        super::url::resolve(base, action).unwrap_or_else(|| action.to_string())
    };
    let encoded = encode_urlencoded(fields);
    if method == "GET" {
        let url = append_query(&target, &encoded);
        FormSubmit {
            method: String::from("GET"),
            url,
            body: String::new(),
            content_type: "application/x-www-form-urlencoded",
        }
    } else {
        FormSubmit {
            method: String::from("POST"),
            url: target,
            body: encoded,
            content_type: "application/x-www-form-urlencoded",
        }
    }
}

fn append_query(url: &str, query: &str) -> String {
    if query.is_empty() {
        return url.to_string();
    }
    let (base, frag) = match url.find('#') {
        Some(i) => (&url[..i], &url[i..]),
        None => (url, ""),
    };
    if base.contains('?') {
        format!("{base}&{query}{frag}")
    } else {
        format!("{base}?{query}{frag}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn encode_simple_fields() {
        let fields = [
            FormField {
                name: String::from("q"),
                value: String::from("hello world"),
            },
            FormField {
                name: String::from("a"),
                value: String::from("1&2"),
            },
        ];
        let s = encode_urlencoded(&fields);
        assert_eq!(s, "q=hello+world&a=1%262");
        assert_eq!(percent_decode("hello+world"), "hello world");
        assert_eq!(percent_decode("1%262"), "1&2");
    }

    #[test_case]
    fn get_submit_appends_query() {
        let fields = [FormField {
            name: String::from("q"),
            value: String::from("x"),
        }];
        let s = build_submit("https://ex.com/search", "", "get", &fields);
        assert_eq!(s.method, "GET");
        assert_eq!(s.url, "https://ex.com/search?q=x");
        assert!(s.body.is_empty());
    }

    #[test_case]
    fn post_submit_has_body() {
        let fields = [FormField {
            name: String::from("user"),
            value: String::from("a"),
        }];
        let s = build_submit("https://ex.com/", "/login", "POST", &fields);
        assert_eq!(s.method, "POST");
        assert_eq!(s.url, "https://ex.com/login");
        assert_eq!(s.body, "user=a");
    }

    #[test_case]
    fn relative_action_resolves() {
        let s = build_submit("https://ex.com/a/b", "../c", "get", &[]);
        assert!(s.url.contains("ex.com"), "{}", s.url);
    }
}
