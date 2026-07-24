//! A **public package registry** client: fetch a signed index of installable
//! agents over HTTP(S), search it, and resolve an entry for install. The index
//! is JSON:
//!
//! ```json
//! { "schema": 1, "entries": [
//!     { "name": "report-writer", "version": "1.0.0",
//!       "description": "…", "download": "https://…/pkg/report-writer",
//!       "key_id": "chitti-publisher-test" } ] }
//! ```
//!
//! Discovery (search/resolve) is over the network; the install pipeline then
//! runs the same verify → consent → capability-subset flow as any package
//! (`skills::install`). Registry packages are authenticated with **P-256 ECDSA**
//! against the baked publisher trust store (`skills::crypto`), not the local MAC.
//!
//! **Index authenticity:** when the JSON root carries `key_id` + `sig`
//! (base64 DER ECDSA-P256 over the raw body with those two fields stripped is
//! not implemented yet — we verify over the concatenation of each entry's
//! `name\\0version\\0download\\0key_id\\n` in order), the signature is checked
//! against the trust store and unsigned/invalid indexes are **refused**. An
//! index with no `sig` field is refused unless `allow_unsigned` is true
//! (test/dev only).
//!
//! Note: in this pre-release, an entry's `download` payload is resolved from the
//! built-in signed catalog (the kernel can't yet consume a foreign postcard
//! blob it didn't build). Fetching + `SkillPackage::from_bytes` + P-256 verify
//! of a downloaded blob is the next increment — `crypto::verify_p256` is already
//! implemented and tested, and `SkillPackage::verify` already dispatches on it.

use crate::json::Json;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One installable agent listed in the registry index.
#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub name: String,
    pub version: String,
    pub description: String,
    pub download: String,
    pub key_id: String,
}

/// Fetch and parse the registry index at `url`. Unsigned indexes are refused
/// (see module docs); use [`parse_index_allow_unsigned`] only in tests.
pub fn fetch_index(url: &str) -> Result<Vec<IndexEntry>, String> {
    let resp = crate::net::http::get(url, 15_000).map_err(|e| alloc::format!("fetch failed: {e}"))?;
    if resp.status != 200 {
        return Err(alloc::format!("registry returned status {}", resp.status));
    }
    parse_index(&resp.text())
}

/// Parse an index document (pulled out for unit testing without the network).
/// Requires a valid `key_id` + `sig` (base64 DER) over the entry binding string.
pub fn parse_index(text: &str) -> Result<Vec<IndexEntry>, String> {
    parse_index_ex(text, false)
}

/// Like [`parse_index`] but allows an unsigned document (unit tests / offline fixtures).
pub fn parse_index_allow_unsigned(text: &str) -> Result<Vec<IndexEntry>, String> {
    parse_index_ex(text, true)
}

fn parse_index_ex(text: &str, allow_unsigned: bool) -> Result<Vec<IndexEntry>, String> {
    let j = Json::parse(text).ok_or_else(|| "index is not valid JSON".to_string())?;
    let arr = j.get("entries").and_then(|e| e.as_array()).ok_or_else(|| "index has no `entries` array".to_string())?;
    let field = |e: &Json, k: &str| e.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let mut out = Vec::new();
    for e in arr {
        let name = field(e, "name");
        if name.is_empty() {
            continue;
        }
        out.push(IndexEntry {
            name,
            version: field(e, "version"),
            description: field(e, "description"),
            download: field(e, "download"),
            key_id: field(e, "key_id"),
        });
    }
    let key_id = j.get("key_id").and_then(|v| v.as_str()).unwrap_or("");
    let sig_b64 = j.get("sig").and_then(|v| v.as_str()).unwrap_or("");
    if key_id.is_empty() || sig_b64.is_empty() {
        if allow_unsigned {
            crate::ktrace::log("registry", "index unsigned (allowed for this call)");
            return Ok(out);
        }
        return Err("registry index is unsigned (need key_id + sig)".into());
    }
    let msg = index_binding_bytes(&out);
    let sig = b64_decode(sig_b64).ok_or_else(|| "registry sig is not valid base64".to_string())?;
    if !crate::skills::crypto::verify_p256(key_id, &msg, &sig) {
        return Err("registry index signature verification failed".into());
    }
    Ok(out)
}

/// Canonical bytes signed by the registry publisher: one line per entry
/// `name\\0version\\0download\\0key_id\\n` (description is display-only and
/// excluded so it can be localised without re-signing).
pub fn index_binding_bytes(entries: &[IndexEntry]) -> Vec<u8> {
    let mut msg = Vec::new();
    for e in entries {
        msg.extend_from_slice(e.name.as_bytes());
        msg.push(0);
        msg.extend_from_slice(e.version.as_bytes());
        msg.push(0);
        msg.extend_from_slice(e.download.as_bytes());
        msg.push(0);
        msg.extend_from_slice(e.key_id.as_bytes());
        msg.push(b'\n');
    }
    msg
}

/// Minimal base64 decode (std alphabet, ignores whitespace). `None` on bad input.
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let (a, b, c, d) = (val(chunk[0])?, val(chunk[1])?, val(chunk[2])?, val(chunk[3])?);
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            out.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            out.push((c << 6) | d);
        }
    }
    Some(out)
}

/// Fetch the index and return entries whose name or description contains `query`
/// (case-insensitive; empty query returns all).
pub fn search(url: &str, query: &str) -> Result<Vec<IndexEntry>, String> {
    let q = query.to_ascii_lowercase();
    Ok(fetch_index(url)?
        .into_iter()
        .filter(|e| q.is_empty() || e.name.to_ascii_lowercase().contains(&q) || e.description.to_ascii_lowercase().contains(&q))
        .collect())
}

/// Look up a single entry by exact name in the index at `url`.
pub fn resolve(url: &str, name: &str) -> Result<Option<IndexEntry>, String> {
    Ok(fetch_index(url)?.into_iter().find(|e| e.name == name))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"schema":1,"entries":[
        {"name":"report-writer","version":"1.0.0","description":"Write reports from facts","download":"http://h/pkg/rw","key_id":"chitti-publisher-test"},
        {"name":"note-summarizer","version":"2.1.0","description":"Summarize notes","download":"http://h/pkg/ns","key_id":"chitti-publisher-test"}
    ]}"#;

    #[test_case]
    fn parses_index_entries() {
        // Fixture has no sig — only allow_unsigned for unit tests.
        let e = parse_index_allow_unsigned(SAMPLE).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].name, "report-writer");
        assert_eq!(e[0].version, "1.0.0");
        assert_eq!(e[1].name, "note-summarizer");
        assert_eq!(e[1].key_id, "chitti-publisher-test");
    }

    #[test_case]
    fn rejects_non_json_missing_entries_and_unsigned() {
        assert!(parse_index("not json").is_err());
        assert!(parse_index(r#"{"schema":1}"#).is_err());
        // Production path refuses unsigned indexes.
        let err = parse_index(SAMPLE).unwrap_err();
        assert!(err.contains("unsigned"), "{err}");
    }

    #[test_case]
    fn index_binding_bytes_stable() {
        let e = parse_index_allow_unsigned(SAMPLE).unwrap();
        let b = index_binding_bytes(&e);
        assert!(b.windows(b"report-writer".len()).any(|w| w == b"report-writer"));
        assert_eq!(b.iter().filter(|&&c| c == b'\n').count(), 2);
    }
}
