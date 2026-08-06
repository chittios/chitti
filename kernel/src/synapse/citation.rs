//! **Citation-constrained arguments** — the paper §6 answer to turn-granular taint.
//!
//! A plan argument is either a *literal* (model-authored, inherits turn taint) or a
//! *citation* into the resident context (an offset + length the executor verifies).
//! The kernel checks that the span exists, reads the provenance already attached to
//! that region, and requires the argument to *be* the cited span (or a whitelisted
//! transform of it). A quote is verifiable; a claim is not.
//!
//! This module is the checkable half. Wiring it into the constraint grammar and
//! the executor gate is the next step; the shape and the predicates live here so
//! they cannot silently drift from the paper's strawman.

use crate::agent::types::Provenance;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One resident context region a citation may name.
#[derive(Clone, Debug)]
pub struct ContextSpan {
    /// Absolute byte offset in the concatenated resident transcript.
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
    /// Provenance already attached to this region.
    pub provenance: Provenance,
    /// The bytes themselves.
    pub bytes: String,
}

/// A plan argument: either model-authored text, or a verified citation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CitedArg {
    /// Model wrote these bytes. Carries the turn's resident max taint.
    Literal(String),
    /// Citation into the context: `(offset, len)`. The executor verifies.
    Cite { offset: usize, len: usize },
}

/// Outcome of verifying one cited argument against the resident context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CiteCheck {
    /// Literal — no span to verify; caller applies turn taint.
    Literal { text: String },
    /// Citation matched a span exactly.
    Matched {
        text: String,
        provenance: Provenance,
    },
    /// Citation named a span that does not exist (or is truncated).
    Missing,
}

/// Concatenate resident messages into the byte stream citations index into.
///
/// Offsets are absolute in this concatenation (message order, no separators),
/// so a citation never silently skips a boundary the model did not name.
pub fn flatten_context(spans: &[ContextSpan]) -> String {
    let mut out = String::new();
    for s in spans {
        out.push_str(&s.bytes);
    }
    out
}

/// Verify `arg` against `spans`. Literals pass through; citations must name an
/// exact existing region.
pub fn verify(arg: &CitedArg, spans: &[ContextSpan]) -> CiteCheck {
    match arg {
        CitedArg::Literal(t) => CiteCheck::Literal { text: t.clone() },
        CitedArg::Cite { offset, len } => {
            let flat = flatten_context(spans);
            let end = offset.saturating_add(*len);
            if *len == 0 || end > flat.len() || *offset >= flat.len() {
                return CiteCheck::Missing;
            }
            let slice = &flat[*offset..end];
            let mut prov = Provenance::UntrustedIngested;
            let mut covered = false;
            for s in spans {
                let s_end = s.offset.saturating_add(s.len);
                if *offset >= s.offset && end <= s_end {
                    prov = s.provenance;
                    covered = true;
                    break;
                }
            }
            if !covered {
                return CiteCheck::Matched {
                    text: slice.to_string(),
                    provenance: Provenance::UntrustedIngested,
                };
            }
            CiteCheck::Matched {
                text: slice.to_string(),
                provenance: prov,
            }
        }
    }
}

/// Parse a citation object from a tool-call JSON fragment.
///
/// Accepted shapes:
/// - `{"$cite":{"offset":N,"len":M}}`
/// - `{"offset":N,"len":M}` when `cite:true` is also present
///
/// A plain JSON string becomes a literal.
pub fn parse_cite_json(obj_text: &str) -> Option<CitedArg> {
    let j = crate::json::Json::parse(obj_text)?;
    if let Some(c) = j.get("$cite") {
        let offset = c.get("offset").and_then(|v| v.as_f64()).map(|n| n as u64)? as usize;
        let len = c.get("len").and_then(|v| v.as_f64()).map(|n| n as u64)? as usize;
        return Some(CitedArg::Cite { offset, len });
    }
    if j.get("cite").and_then(|v| v.as_bool()) == Some(true) {
        let offset = j.get("offset").and_then(|v| v.as_f64()).map(|n| n as u64)? as usize;
        let len = j.get("len").and_then(|v| v.as_f64()).map(|n| n as u64)? as usize;
        return Some(CitedArg::Cite { offset, len });
    }
    if let Some(s) = j.as_str() {
        return Some(CitedArg::Literal(s.to_string()));
    }
    None
}

/// Build a [`ContextSpan`] list from `(text, provenance)` pairs in order.
pub fn spans_from_messages(msgs: &[(String, Provenance)]) -> Vec<ContextSpan> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for (text, prov) in msgs {
        let len = text.len();
        out.push(ContextSpan {
            offset,
            len,
            provenance: *prov,
            bytes: text.clone(),
        });
        offset = offset.saturating_add(len);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_literal_passes_through_without_a_span() {
        let arg = CitedArg::Literal(String::from("rm -rf /"));
        assert_eq!(
            verify(&arg, &[]),
            CiteCheck::Literal {
                text: String::from("rm -rf /")
            }
        );
    }

    #[test_case]
    fn a_citation_must_name_an_existing_span() {
        let spans = spans_from_messages(&[(
            String::from("delete /tmp/sandbox/victim"),
            Provenance::UntrustedIngested,
        )]);
        let ok = CitedArg::Cite {
            offset: 7,
            len: 19,
        };
        match verify(&ok, &spans) {
            CiteCheck::Matched { text, provenance } => {
                assert_eq!(text, "/tmp/sandbox/victim");
                assert_eq!(provenance, Provenance::UntrustedIngested);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
        assert_eq!(
            verify(&CitedArg::Cite { offset: 0, len: 9999 }, &spans),
            CiteCheck::Missing
        );
    }

    #[test_case]
    fn user_typed_citation_keeps_user_provenance() {
        let spans = spans_from_messages(&[(
            String::from("please remove notes.txt"),
            Provenance::UserTyped,
        )]);
        let arg = CitedArg::Cite {
            offset: 14,
            len: 9,
        };
        match verify(&arg, &spans) {
            CiteCheck::Matched { text, provenance } => {
                assert_eq!(text, "notes.txt");
                assert_eq!(provenance, Provenance::UserTyped);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }

    #[test_case]
    fn parse_cite_json_reads_the_wrapper_and_flag_forms() {
        let a = parse_cite_json(r#"{"$cite":{"offset":3,"len":4}}"#).unwrap();
        assert_eq!(a, CitedArg::Cite { offset: 3, len: 4 });
        let b = parse_cite_json(r#"{"cite":true,"offset":1,"len":2}"#).unwrap();
        assert_eq!(b, CitedArg::Cite { offset: 1, len: 2 });
        let c = parse_cite_json(r#""hello""#).unwrap();
        assert_eq!(c, CitedArg::Literal(String::from("hello")));
    }
}
