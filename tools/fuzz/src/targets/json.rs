//! Fuzz target: the kernel's JSON parser (`kernel/src/json.rs`).
//!
//! The JSON parser consumes **model output** — tool-call arguments, skill
//! payloads, registry indices — so it is the highest-value panic target in the
//! OS: a panic here is a kernel abort triggered by untrusted text. The harness
//! drives a parse → pretty-print → re-parse round trip, which exercises both
//! the parse tree and the serializer.

// `kernel/src/json.rs` is `no_std` and names `alloc` explicitly; on the host
// `alloc` is re-exported by `std`, but the `use` paths still need the name.

#[path = "../../../../kernel/src/json.rs"]
#[allow(dead_code)]
pub mod json;

pub fn run(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    if let Some(v) = json::Json::parse(&text) {
        let pretty = v.to_pretty();
        // Re-parse the pretty form: a serializer bug that emits unparseable
        // JSON is as real a panic as a parser bug.
        if let Some(v2) = json::Json::parse(&pretty) {
            let _ = v2.to_pretty();
        }
    }
}
