//! **Effect journal** — what an agent did, and how to take it back.
//!
//! An agent that writes twelve files and gets the eleventh wrong leaves the
//! human to work out which twelve. This module records each effectful primitive
//! together with the information needed to invert it, so a whole turn can be
//! rolled back as a unit.
//!
//! It lives here rather than in the agent layer for the same reason the audit
//! log does: **every effect already funnels through
//! [`crate::synapse::executor`]**, so there is exactly one place to record them
//! and no way for a caller to bypass the recording by taking a different route.
//!
//! ## Recording the inverse, not the action
//!
//! Rollback needs the *previous* state, and the only moment it exists is just
//! before the write. So [`Undo`] carries the prior bytes (or the fact that there
//! were none), captured by the executor before it calls the primitive. Deriving
//! the inverse afterwards is impossible for exactly the cases that matter — once
//! a file is overwritten its old contents are gone.
//!
//! ## What is and is not reversible, stated rather than assumed
//!
//! A store write is reversible. A network send is not: the bytes have left. A
//! console write is not: the human read it. Pretending otherwise would be worse
//! than not offering rollback at all, because the user would believe a turn had
//! been undone when half of it had escaped. So [`Undo::Irreversible`] is a real
//! variant, [`Turn::fully_reversible`] reports honestly, and `/undo-turn` tells
//! the user precisely which effects it could not take back.
//!
//! ## Bounded, like every other history in this kernel
//!
//! Turns are capped ([`MAX_TURNS`]) and oldest-dropped. An unbounded journal of
//! file contents is a memory leak that grows with exactly the activity you most
//! want to be able to undo.

use crate::mm::Locked;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// How many completed turns stay rollback-able.
pub const MAX_TURNS: usize = 16;

/// How to reverse one recorded effect.
#[derive(Clone, PartialEq, Debug)]
pub enum Undo {
    /// The path held these bytes before the write; restore them.
    RestoreFile { path: String, prior: Vec<u8> },
    /// The path did not exist before the write; delete it.
    DeleteFile { path: String },
    /// The effect left the machine or reached the human. Named so the report
    /// can say what it could not undo.
    Irreversible { what: &'static str, detail: String },
}

/// One recorded effect: the primitive that caused it and how to undo it.
#[derive(Clone, PartialEq, Debug)]
pub struct Entry {
    pub primitive: &'static str,
    pub undo: Undo,
}

/// A group of effects from one agent turn.
#[derive(Clone, PartialEq, Debug)]
pub struct Turn {
    pub id: u64,
    pub agent: u64,
    pub label: String,
    pub entries: Vec<Entry>,
}

impl Turn {
    /// Whether every recorded effect can be taken back.
    pub fn fully_reversible(&self) -> bool {
        !self.entries.iter().any(|e| matches!(e.undo, Undo::Irreversible { .. }))
    }

    /// The inverse operations, **newest first** — order matters: a turn that
    /// created a file and then wrote it again must undo the write before the
    /// creation, or the delete removes a file the restore just repopulated.
    pub fn inverse(&self) -> Vec<&Undo> {
        self.entries.iter().rev().map(|e| &e.undo).collect()
    }

    /// One-line summary for `/undo-turn` to show before it acts.
    pub fn summary(&self) -> String {
        let files = self
            .entries
            .iter()
            .filter(|e| matches!(e.undo, Undo::RestoreFile { .. } | Undo::DeleteFile { .. }))
            .count();
        let irreversible = self.entries.len() - files;
        alloc::format!(
            "turn {} (agent {}): {} — {} file change(s), {} irreversible",
            self.id,
            self.agent,
            self.label,
            files,
            irreversible
        )
    }
}

#[derive(Default)]
struct Journal {
    /// The turn currently being recorded, if a turn is open.
    open: Option<Turn>,
    done: Vec<Turn>,
    next_id: u64,
}

static JOURNAL: Locked<Journal> = Locked::new(Journal { open: None, done: Vec::new(), next_id: 1 });

/// Begin recording a turn. A turn already open is closed first — a caller that
/// forgot to end one must not silently merge two turns into an unrollbackable
/// blob.
pub fn begin(agent: u64, label: &str) -> u64 {
    JOURNAL.with(|j| {
        if let Some(t) = j.open.take() {
            push_done(j, t);
        }
        let id = j.next_id;
        j.next_id += 1;
        j.open = Some(Turn { id, agent, label: label.to_string(), entries: Vec::new() });
        id
    })
}

/// Record one effect against the open turn. A no-op when no turn is open, so
/// kernel-internal calls outside an agent turn are not journalled.
pub fn record(primitive: &'static str, undo: Undo) {
    JOURNAL.with(|j| {
        if let Some(t) = j.open.as_mut() {
            t.entries.push(Entry { primitive, undo });
        }
    });
}

/// Whether a turn is currently recording.
pub fn is_open() -> bool {
    JOURNAL.with(|j| j.open.is_some())
}

/// Close the open turn. An empty turn is discarded rather than stored — a turn
/// that changed nothing is not something anyone wants to undo, and keeping them
/// would push real turns out of the bounded history.
pub fn end() -> Option<u64> {
    JOURNAL.with(|j| {
        let t = j.open.take()?;
        if t.entries.is_empty() {
            return None;
        }
        let id = t.id;
        push_done(j, t);
        Some(id)
    })
}

fn push_done(j: &mut Journal, t: Turn) {
    if t.entries.is_empty() {
        return;
    }
    j.done.push(t);
    if j.done.len() > MAX_TURNS {
        j.done.remove(0);
    }
}

/// The most recent completed turn, without removing it.
pub fn last() -> Option<Turn> {
    JOURNAL.with(|j| j.done.last().cloned())
}

/// Take the most recent completed turn for rollback.
pub fn take_last() -> Option<Turn> {
    JOURNAL.with(|j| j.done.pop())
}

/// Every completed turn, newest first.
pub fn list() -> Vec<Turn> {
    JOURNAL.with(|j| j.done.iter().rev().cloned().collect())
}

/// Drop all history. Used when switching agents so one agent's turns cannot be
/// rolled back from another's session.
pub fn clear() {
    JOURNAL.with(|j| {
        j.open = None;
        j.done.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn reset() {
        clear();
    }

    #[test_case]
    fn a_turn_records_and_closes() {
        reset();
        let id = begin(7, "write config");
        assert!(is_open());
        record("mem_fs_write", Undo::DeleteFile { path: "/a".to_string() });
        assert_eq!(end(), Some(id));
        assert!(!is_open());
        let t = last().unwrap();
        assert_eq!(t.id, id);
        assert_eq!(t.agent, 7);
        assert_eq!(t.entries.len(), 1);
    }

    #[test_case]
    fn an_empty_turn_is_discarded() {
        // A turn that changed nothing is not undoable, and keeping it would push
        // a real turn out of the bounded history.
        reset();
        begin(1, "read only");
        assert_eq!(end(), None);
        assert!(last().is_none());
    }

    #[test_case]
    fn inverse_order_is_newest_first() {
        // Create-then-write must undo the write before the create, or the delete
        // removes a file the restore just repopulated.
        reset();
        begin(1, "two steps");
        record("mem_fs_write", Undo::DeleteFile { path: "/f".to_string() });
        record("mem_fs_write", Undo::RestoreFile { path: "/f".to_string(), prior: vec![1] });
        end();
        let t = last().unwrap();
        let inv = t.inverse();
        assert!(matches!(inv[0], Undo::RestoreFile { .. }), "the later write undoes first");
        assert!(matches!(inv[1], Undo::DeleteFile { .. }));
    }

    #[test_case]
    fn an_irreversible_effect_is_reported_not_hidden() {
        // The user must not believe a turn was fully undone when a network send
        // had already left the machine.
        reset();
        begin(1, "post and write");
        record("mem_fs_write", Undo::DeleteFile { path: "/f".to_string() });
        record(
            "net_http_post",
            Undo::Irreversible { what: "network send", detail: "POST example.com".to_string() },
        );
        end();
        let t = last().unwrap();
        assert!(!t.fully_reversible());
        assert!(t.summary().contains("1 irreversible"));
    }

    #[test_case]
    fn a_turn_of_only_file_writes_is_fully_reversible() {
        reset();
        begin(1, "files");
        record("mem_fs_write", Undo::RestoreFile { path: "/a".to_string(), prior: vec![] });
        record("mem_fs_delete", Undo::RestoreFile { path: "/b".to_string(), prior: vec![9] });
        end();
        assert!(last().unwrap().fully_reversible());
    }

    #[test_case]
    fn opening_a_turn_closes_a_forgotten_one() {
        // Otherwise a caller that forgets `end` silently merges unrelated work
        // into one blob that cannot be rolled back meaningfully.
        reset();
        begin(1, "first");
        record("mem_fs_write", Undo::DeleteFile { path: "/a".to_string() });
        begin(1, "second");
        record("mem_fs_write", Undo::DeleteFile { path: "/b".to_string() });
        end();
        let all = list();
        assert_eq!(all.len(), 2, "the forgotten turn was closed, not merged");
    }

    #[test_case]
    fn recording_outside_a_turn_is_a_no_op() {
        // Kernel-internal calls happen constantly and are not agent effects.
        reset();
        record("mem_fs_write", Undo::DeleteFile { path: "/x".to_string() });
        assert!(last().is_none());
    }

    #[test_case]
    fn history_is_bounded() {
        reset();
        for i in 0..MAX_TURNS + 5 {
            begin(1, "t");
            record("mem_fs_write", Undo::DeleteFile { path: alloc::format!("/{i}") });
            end();
        }
        assert_eq!(list().len(), MAX_TURNS);
    }

    #[test_case]
    fn take_last_pops_so_a_turn_cannot_be_undone_twice() {
        reset();
        begin(1, "once");
        record("mem_fs_write", Undo::DeleteFile { path: "/a".to_string() });
        end();
        assert!(take_last().is_some());
        assert!(take_last().is_none(), "undoing twice would double-apply the inverse");
    }
}
