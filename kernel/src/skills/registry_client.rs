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

/// Fetch and parse the registry index at `url`.
pub fn fetch_index(url: &str) -> Result<Vec<IndexEntry>, String> {
    let resp = crate::net::http::get(url, 15_000).map_err(|e| alloc::format!("fetch failed: {e}"))?;
    if resp.status != 200 {
        return Err(alloc::format!("registry returned status {}", resp.status));
    }
    parse_index(&resp.text())
}

/// Parse an index document (pulled out for unit testing without the network).
pub fn parse_index(text: &str) -> Result<Vec<IndexEntry>, String> {
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
    Ok(out)
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
        let e = parse_index(SAMPLE).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].name, "report-writer");
        assert_eq!(e[0].version, "1.0.0");
        assert_eq!(e[1].name, "note-summarizer");
        assert_eq!(e[1].key_id, "chitti-publisher-test");
    }

    #[test_case]
    fn rejects_non_json_and_missing_entries() {
        assert!(parse_index("not json").is_err());
        assert!(parse_index(r#"{"schema":1}"#).is_err());
    }
}
