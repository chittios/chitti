//! Intent **planner** (`CHITTI_OS_HANDOFF.md` Phase 5): the piece that turns
//! a natural-language intent into a concrete plan of actions -- Synapse tool
//! calls and memory operations -- the agent then executes.
//!
//! The planner is the *stochastic* layer, above the determinism boundary
//! (Part 2). In the shipping design it is the Cortex model, emitting tool
//! calls constrained by the Phase 4 grammar. The default implementation here
//! is instead a small, deterministic, rule-based parser: the real 0.8B model
//! is far too slow under QEMU TCG (~tens of seconds per token) to drive a
//! multi-step plan inside a test, and the fast test suite is model-free by
//! design. The [`Planner`] trait is the seam a Cortex-backed planner drops
//! into unchanged -- what the runtime below it does with the plan (execute,
//! checkpoint, recall, coordinate) is identical either way, and that runtime
//! is what Phase 5 delivers and tests.

use super::actions::Action;
use alloc::string::String;
use alloc::vec::Vec;

/// Produces a plan (a sequence of [`Action`]s) for an intent. A Cortex-backed
/// planner and the deterministic [`RulePlanner`] are interchangeable behind
/// this trait.
pub trait Planner {
    fn plan(&mut self, intent: &str) -> Vec<Action>;
    fn name(&self) -> &'static str;
}

/// The deterministic rule-based planner (see the module doc for why). It
/// recognises a deliberately small set of intent shapes; anything else falls
/// back to echoing the intent as the result. Fully reproducible: the same
/// intent always yields the same plan.
pub struct RulePlanner;

impl Planner for RulePlanner {
    fn name(&self) -> &'static str {
        "rule-based (deterministic)"
    }

    fn plan(&mut self, intent: &str) -> Vec<Action> {
        let low = intent.to_ascii_lowercase();

        // "write a file called X with the text Y[, then read it back]"
        if low.contains("write") && low.contains("text") {
            if let (Some(path), Some(text)) = (extract_path(intent, &low), extract_text(intent, &low)) {
                let mut steps = alloc::vec![Action::call_write(&path, &text)];
                if low.contains("read") {
                    steps.push(Action::call_read(&path));
                }
                return steps;
            }
        }

        // "remember (that) K is/= V"
        if let Some(rest) = after(intent, &low, "remember ") {
            let rest = strip_prefix_ci(rest, "that ");
            if let Some((key, value)) = split_kv(rest) {
                return alloc::vec![Action::Remember(key, value)];
            }
        }

        // "recall K" / "what is K" / "what's K"
        if let Some(rest) = after(intent, &low, "recall ") {
            return alloc::vec![Action::Recall(clean_key(rest))];
        }
        for marker in ["what is ", "what's ", "whats "] {
            if let Some(rest) = after(intent, &low, marker) {
                return alloc::vec![Action::Recall(clean_key(rest))];
            }
        }

        // "list files" / "list"
        if low.starts_with("list") {
            return alloc::vec![Action::call_list()];
        }

        // "say TEXT" -> console
        if let Some(rest) = after(intent, &low, "say ") {
            return alloc::vec![Action::call_console(rest.trim())];
        }

        // Fallback: report the intent verbatim as the result.
        alloc::vec![Action::call_emit(intent.trim())]
    }
}

// --- intent-parsing helpers (ASCII, case-insensitive matching) ------------

/// Return the slice of `orig` following the first occurrence of `marker` in
/// its lowercased form `low` (indices coincide for ASCII).
fn after<'a>(orig: &'a str, low: &str, marker: &str) -> Option<&'a str> {
    low.find(marker).map(|i| &orig[i + marker.len()..])
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> &'a str {
    if s.to_ascii_lowercase().starts_with(prefix) {
        &s[prefix.len()..]
    } else {
        s
    }
}

/// Extract the file path from a "...called X with..." (or "named X") phrase.
fn extract_path(orig: &str, low: &str) -> Option<String> {
    let rest = after(orig, low, "called ").or_else(|| after(orig, low, "named "))?;
    let rest_low = rest.to_ascii_lowercase();
    let end = rest_low.find(" with").unwrap_or(rest.len());
    let path = rest[..end].trim();
    (!path.is_empty()).then(|| String::from(path))
}

/// Extract the text payload from a "...with the text Y..." phrase, cut at the
/// first comma (so a trailing ", then read it back" is dropped).
fn extract_text(orig: &str, low: &str) -> Option<String> {
    let rest = after(orig, low, "text ")?;
    let end = rest.find(',').unwrap_or(rest.len());
    let text = rest[..end].trim();
    (!text.is_empty()).then(|| String::from(text))
}

/// Split "K is V" / "K = V" / "K: V" into (key, value).
fn split_kv(s: &str) -> Option<(String, String)> {
    for sep in [" is ", " = ", "=", ": ", ":"] {
        if let Some(i) = s.find(sep) {
            let key = clean_key(&s[..i]);
            let value = s[i + sep.len()..].trim().trim_end_matches(['.', '!']).trim();
            if !key.is_empty() && !value.is_empty() {
                return Some((key, String::from(value)));
            }
        }
    }
    None
}

/// Normalise a memory key: trim whitespace and trailing punctuation.
fn clean_key(s: &str) -> String {
    String::from(s.trim().trim_end_matches(['?', '.', '!']).trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn plans_write_then_read() {
        let plan = RulePlanner.plan("write a file called notes with the text hello, then read it back");
        assert_eq!(plan.len(), 2);
        match &plan[0] {
            Action::Call(json) => {
                assert!(json.contains("mem_fs_write"));
                assert!(json.contains("\"path\":\"notes\""));
                assert!(json.contains("\"text\":\"hello\""));
            }
            other => panic!("expected a write call, got {other:?}"),
        }
        match &plan[1] {
            Action::Call(json) => assert!(json.contains("mem_fs_read") && json.contains("\"path\":\"notes\"")),
            other => panic!("expected a read call, got {other:?}"),
        }
    }

    #[test_case]
    fn plans_remember_and_recall() {
        match &RulePlanner.plan("remember that capital_of_france is Paris")[0] {
            Action::Remember(k, v) => {
                assert_eq!(k, "capital_of_france");
                assert_eq!(v, "Paris");
            }
            other => panic!("expected Remember, got {other:?}"),
        }
        match &RulePlanner.plan("what is capital_of_france?")[0] {
            Action::Recall(k) => assert_eq!(k, "capital_of_france"),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test_case]
    fn unrecognised_intent_falls_back_to_emit() {
        match &RulePlanner.plan("ponder the meaning of everything")[0] {
            Action::Call(json) => assert!(json.contains("emit_result")),
            other => panic!("expected emit_result fallback, got {other:?}"),
        }
    }
}
