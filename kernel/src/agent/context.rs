//! **Context management** (`CHITTI_AGENTIC_HANDOFF.md` Phase D): keep a long
//! session within its KV/context budget by *auto-compaction* — when live tokens
//! approach the budget, the oldest turns are summarized, their full text pushed
//! to the persistent store, and only a compact summary kept resident. A later
//! reference pulls the full text back on demand ([`recall`]) — demand-paging of
//! conversation history over the two-tier memory (`synapse::fs`).
//!
//! Deterministic: the summarizer is a fixed textual condensation (the real
//! Cortex summarizer plugs in behind the same seam), so compaction is
//! reproducible for the temp-0 tests.

use crate::agent::types::*;
use crate::session::session::est_tokens;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Keep the system prompt (message 0) and this many most-recent messages
/// resident; everything older is eligible for compaction.
const KEEP_RECENT: usize = 3;

/// Recompute `live_tokens` from what is actually resident: resident message
/// tokens + compaction-summary tokens.
fn recompute_live_tokens(session: &mut Session) {
    let msg: u32 = session.messages.iter().filter(|m| m.resident).map(|m| m.tokens).sum();
    let sum: u32 = session.context.compactions.iter().map(|c| c.tokens).sum();
    session.context.live_tokens = msg + sum;
}

/// If the session is at/over its compaction threshold, summarize the oldest
/// compactible messages out to the store and keep a summary resident. Returns
/// true if a compaction happened. Idempotent below threshold.
pub fn maybe_compact(session: &mut Session, now: Ticks) -> bool {
    let threshold = session.budget.limits.compact_threshold;
    if threshold == 0 || session.context.live_tokens < threshold {
        return false;
    }
    let n = session.messages.len();
    if n <= KEEP_RECENT + 1 {
        return false; // nothing but system + recent
    }
    let upper = n - KEEP_RECENT; // exclusive; keep [upper, n) resident
    let mut first: Option<MsgId> = None;
    let mut last: Option<MsgId> = None;
    let mut parts: Vec<String> = Vec::new();
    for i in 1..upper {
        if !session.messages[i].resident {
            continue;
        }
        let (mid, snippet) = {
            let m = &session.messages[i];
            let snip: String = m.content.chars().take(24).collect();
            (m.id, snip)
        };
        // Persist the full text to the store, keyed by session + message id.
        let key = StoreKey(format!("sess/{}/cmp/{}", session.id.0, mid.0));
        crate::synapse::fs::write(&key.0, session.messages[i].content.as_bytes());
        session.messages[i].resident = false;
        session.messages[i].store_ref = Some(key);
        if first.is_none() {
            first = Some(mid);
        }
        last = Some(mid);
        if !snippet.is_empty() {
            parts.push(snippet);
        }
    }
    let (Some(first), Some(last)) = (first, last) else {
        return false; // nothing was compacted
    };
    let summary = format!("[compacted {}..{}: {}]", first.0, last.0, parts.join(" | "));
    let summary_ref = StoreKey(format!("sess/{}/cmpsum/{}", session.id.0, session.context.compactions.len()));
    crate::synapse::fs::write(&summary_ref.0, summary.as_bytes());
    let tokens = est_tokens(&summary);
    session.context.compactions.push(CompactionRecord {
        covers: (first, last),
        summary,
        summary_ref,
        tokens,
        at_ticks: now,
    });
    recompute_live_tokens(session);
    crate::ktrace::log_fmt(format_args!(
        "context.compact: session {} summarized {}..{} -> {} live tokens now {}",
        session.id.0, first.0, last.0, session.context.compactions.len(), session.context.live_tokens
    ));
    true
}

/// Recall a compacted message's full text on demand, paging it back into live
/// context (marks it resident and restores its token cost). Returns the text,
/// or the resident text if it was never compacted, or `None` if unknown.
pub fn recall(session: &mut Session, id: MsgId) -> Option<String> {
    let idx = session.messages.iter().position(|m| m.id == id)?;
    if session.messages[idx].resident {
        return Some(session.messages[idx].content.clone());
    }
    let key = session.messages[idx].store_ref.clone()?;
    let bytes = crate::synapse::fs::read(&key.0)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    session.messages[idx].resident = true;
    recompute_live_tokens(session);
    crate::ktrace::log_fmt(format_args!("context.recall: session {} paged message {} back into context", session.id.0, id.0));
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::manifest;
    use crate::session::todo::{self, TodoInput};

    /// (a) A session driven past its budget compacts older turns (event
    /// recorded, messages evicted to the store) and a compacted fact can be
    /// recalled on demand.
    #[test_case]
    fn compaction_evicts_and_recall_pages_back() {
        let mut m = manifest::orchestrator_manifest();
        m.budgets.compact_threshold = 30; // tiny, to force compaction
        let mut s = Session::new(&m, 1, alloc::vec![], 0);
        // Push several sizable turns; remember the first user message's id + text.
        let key_text = "the secret account number is 8675309 for reference";
        let first_id = s.push_message(Role::User, key_text.into(), Provenance::UserTyped, 1);
        for i in 0..8 {
            s.push_message(Role::Assistant, alloc::format!("assistant turn number {i} with some filler text"), Provenance::SystemTrusted, 2);
        }
        assert!(s.context.live_tokens >= 30);
        let did = maybe_compact(&mut s, 10);
        assert!(did, "compaction should have triggered");
        assert!(!s.context.compactions.is_empty(), "a compaction event is recorded");
        // The early user message is now evicted (not resident) but has a store_ref.
        let m0 = s.messages.iter().find(|mm| mm.id == first_id).unwrap();
        assert!(!m0.resident, "old message evicted from live context");
        assert!(m0.store_ref.is_some(), "evicted text lives in the store");
        // Recall pages the exact original text back in.
        let recalled = recall(&mut s, first_id).expect("recall");
        assert_eq!(recalled, key_text, "recall returns the original text verbatim");
        assert!(s.messages.iter().find(|mm| mm.id == first_id).unwrap().resident, "recalled message is resident again");
    }

    /// (b) A 5+ step task is tracked and completed via the todo list.
    #[test_case]
    fn five_step_task_tracked_via_todos() {
        let m = manifest::orchestrator_manifest();
        let mut s = Session::new(&m, 2, alloc::vec![], 0);
        let steps = ["scan", "read", "analyze", "summarize", "write report"];
        let items: alloc::vec::Vec<TodoInput> = steps
            .iter()
            .enumerate()
            .map(|(i, t)| TodoInput { id: (i + 1) as u32, text: (*t).into(), status: TodoStatus::Pending })
            .collect();
        let remaining = todo::write(&mut s, items, 1);
        assert_eq!(remaining, 5, "five pending steps");
        // Work them down: mark each done.
        let done: alloc::vec::Vec<TodoInput> = steps
            .iter()
            .enumerate()
            .map(|(i, t)| TodoInput { id: (i + 1) as u32, text: (*t).into(), status: TodoStatus::Done })
            .collect();
        let remaining = todo::write(&mut s, done, 2);
        assert_eq!(remaining, 0, "all steps completed");
        assert_eq!(s.todos.iter().filter(|t| t.status == TodoStatus::Done).count(), 5);
    }

    /// (c) A forked session diverges without altering the parent.
    #[test_case]
    fn fork_diverges_without_mutating_parent() {
        let m = manifest::orchestrator_manifest();
        let mut parent = Session::new(&m, 3, alloc::vec![], 0);
        parent.push_message(Role::User, "original".into(), Provenance::UserTyped, 1);
        let parent_len = parent.messages.len();
        let parent_id = parent.id;

        let mut fork = crate::session::fork(&parent, 5);
        assert_ne!(fork.id, parent_id, "fork gets a new id");
        // Mutate only the fork.
        fork.push_message(Role::User, "only-in-fork".into(), Provenance::UserTyped, 6);
        assert_eq!(fork.messages.len(), parent_len + 1);
        assert_eq!(parent.messages.len(), parent_len, "parent is untouched by the fork");
        assert!(!parent.messages.iter().any(|mm| mm.content == "only-in-fork"));
    }
}
