//! The Synapse **audit log** (`CHITTI_OS_HANDOFF.md` Phase 4): an
//! append-only record of every attempted capability invocation -- who
//! called, which primitive, a hash of the arguments, the outcome, and a
//! hash of the result. It is the tamper-evident history the determinism
//! boundary promises: every effect (and every *denied* or *rejected*
//! attempt) leaves a permanent, ordered trace.
//!
//! Append-only is enforced structurally: the only mutating operation this
//! module exposes is `record`, which pushes to the end. There is no API to
//! edit, reorder, or remove an entry, and entries are `Copy` value types, so
//! a snapshot taken now can never be invalidated by later activity -- the
//! property the phase's acceptance test asserts directly.

use crate::mm::Locked;
use alloc::vec::Vec;

/// What happened to an attempted invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Grammar-valid, capability held, primitive ran.
    Executed,
    /// Grammar-valid, but the caller lacked the required capability.
    DeniedNoCapability,
    /// Rejected by the grammar; never reached a primitive.
    RejectedMalformed,
    /// A destructive primitive refused by the taint gate (Phase 6): its
    /// justification traced to untrusted ingested content and no human
    /// confirmed it. The primitive did not run.
    RefusedTainted,
    /// Capability held, but the concrete target (path/host/port) fell outside
    /// the granted scope (Gate 2.5). The primitive did not run.
    DeniedScope,
}

/// One immutable audit record. Every field is `Copy` (the primitive name is
/// a `'static` registry string; args/result are summarised as FNV-1a
/// hashes), so a cloned snapshot is a permanent, independent value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Monotonic per-log index (also the entry's position).
    pub seq: u64,
    /// The invoking task.
    pub caller: crate::sched::TaskId,
    /// Registered primitive name, or `"<malformed>"` for a rejected call.
    pub primitive: &'static str,
    /// FNV-1a hash of the raw call text (arguments included).
    pub args_hash: u64,
    pub outcome: Outcome,
    /// FNV-1a hash of the structured result the caller received.
    pub result_hash: u64,
}

static LOG: Locked<Vec<Entry>> = Locked::new(Vec::new());

/// ktrace-mirror coalescing state: how many consecutive entries identical to
/// the last-printed one (all fields but `seq`) have been suppressed. The
/// **log itself records every entry** — only the serial/ring mirror collapses
/// runs, so a polling loop can't drown the human-facing trace (a UI event
/// pump once emitted ~1000 identical lines/second; see `service::package_ui`).
static REPEATS: Locked<(Option<Entry>, u64)> = Locked::new((None, 0));

/// Append one record and return its sequence number. The *only* way to
/// modify the log -- there is no edit or delete path.
pub fn record(caller: crate::sched::TaskId, primitive: &'static str, args_hash: u64, outcome: Outcome, result_hash: u64) -> u64 {
    let seq = LOG.with(|log| {
        let seq = log.len() as u64;
        log.push(Entry { seq, caller, primitive, args_hash, outcome, result_hash });
        seq
    });
    let e = Entry { seq, caller, primitive, args_hash, outcome, result_hash };
    REPEATS.with(|(last, n)| {
        let same = last.map(|l| {
            l.caller == e.caller && l.primitive == e.primitive && l.args_hash == e.args_hash && l.outcome == e.outcome && l.result_hash == e.result_hash
        }).unwrap_or(false);
        if same && *n < 4096 {
            *n += 1;
            return; // suppressed; summarised when the run breaks
        }
        if *n > 0 {
            crate::ktrace::log_fmt(format_args!("synapse.audit: (previous entry repeated {n} more times, through #{})", seq - 1));
        }
        *last = Some(e);
        *n = 0;
        crate::ktrace::log_fmt(format_args!(
            "synapse.audit #{seq}: caller={caller} primitive={primitive} args={args_hash:#018x} outcome={outcome:?} result={result_hash:#018x}"
        ));
    });
    seq
}

/// Number of entries recorded so far.
pub fn len() -> usize {
    LOG.with(|log| log.len())
}

/// A point-in-time copy of the whole log. Because `Entry: Copy`, the
/// returned `Vec` is fully independent of future appends.
pub fn snapshot() -> Vec<Entry> {
    LOG.with(|log| log.clone())
}

/// FNV-1a (64-bit) -- the same hash `cortex` uses for provenance, kept local
/// so `synapse` needn't reach down into the inference layer for it.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn record_appends_and_is_immutable() {
        let before = snapshot();
        let s0 = record(7, "console_write", 0x11, Outcome::Executed, 0x22);
        let s1 = record(7, "list", 0x33, Outcome::Executed, 0x44);
        assert_eq!(s1, s0 + 1, "sequence numbers must be monotonic");

        // The snapshot taken *before* the two records must be a strict
        // prefix of the log now, byte-for-byte -- proving past entries are
        // never mutated or reordered by later appends.
        let after = snapshot();
        assert_eq!(after.len(), before.len() + 2);
        assert_eq!(&after[..before.len()], &before[..], "existing entries changed after append");
        assert_eq!(after[before.len()].primitive, "console_write");
        assert_eq!(after[before.len() + 1].outcome, Outcome::Executed);
    }

    #[test_case]
    fn repeated_entries_all_recorded_despite_ktrace_coalescing() {
        // The ktrace mirror collapses identical runs, but the log itself must
        // record every entry — a polling loop's audit trail stays complete.
        let before = len();
        for _ in 0..10 {
            record(9, "ui_event_poll", 0xaa, Outcome::Executed, 0xbb);
        }
        assert_eq!(len(), before + 10, "every repeat must append to the log");
        let snap = snapshot();
        for (i, e) in snap[before..].iter().enumerate() {
            assert_eq!(e.seq, (before + i) as u64);
            assert_eq!(e.primitive, "ui_event_poll");
        }
    }
}
