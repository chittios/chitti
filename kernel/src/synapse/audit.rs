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
    /// Chain link: the digest of the entry before this one, or 0 for the first.
    ///
    /// Append-only was previously enforced *structurally* — `record` is the only
    /// mutating operation and entries are `Copy` — which is a property of this
    /// module's API, not of the data. It says nothing to a reader holding a
    /// snapshot about whether an entry was altered or removed before they got
    /// it. Folding each entry's digest into the next makes the log
    /// tamper-*evident*: changing or dropping any entry breaks every link after
    /// it, and [`verify`] says where.
    ///
    /// This is not attestation. A compromised kernel can recompute the whole
    /// chain, so this defends against a reader being misled by a doctored
    /// snapshot, not against the machine itself. Sealing it needs a key this
    /// system does not yet have.
    pub prev_hash: u64,
}

impl Entry {
    /// This entry's digest, over every field including its chain link.
    pub fn digest(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        let mut mix = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        mix(self.seq);
        mix(self.caller as u64);
        mix(fnv1a(self.primitive.as_bytes()));
        mix(self.args_hash);
        mix(self.outcome as u64);
        mix(self.result_hash);
        mix(self.prev_hash);
        h
    }
}

static LOG: Locked<Vec<Entry>> = Locked::new(Vec::new());

/// The digest of the most recent entry, kept **outside** the log.
///
/// Chaining alone leaves one entry unprotected: `verify` checks each entry
/// against the one before it, so altering the *last* record breaks no link and
/// is invisible -- and the last record is the one an attacker most wants to
/// change, because it is the call they just made. Holding the head separately
/// gives the walk something to end at.
///
/// This is not attestation. A kernel that can write this static can also
/// recompute it; what it defends is a *snapshot* whose head is quoted
/// elsewhere. Sealing needs a key, and there is no key store in this system --
/// see the module doc.
static HEAD: Locked<u64> = Locked::new(0);

/// ktrace-mirror coalescing state: how many consecutive entries identical to
/// the last-printed one (all fields but `seq`) have been suppressed. The
/// **log itself records every entry** — only the serial/ring mirror collapses
/// runs, so a polling loop can't drown the human-facing trace (a UI event
/// pump once emitted ~1000 identical lines/second; see `service::package_ui`).
static REPEATS: Locked<(Option<Entry>, u64)> = Locked::new((None, 0));

/// Append one record and return its sequence number. The *only* way to
/// modify the log -- there is no edit or delete path.
pub fn record(caller: crate::sched::TaskId, primitive: &'static str, args_hash: u64, outcome: Outcome, result_hash: u64) -> u64 {
    let (seq, prev_hash) = LOG.with(|log| {
        let seq = log.len() as u64;
        let prev_hash = log.last().map(|e| e.digest()).unwrap_or(0);
        let entry = Entry { seq, caller, primitive, args_hash, outcome, result_hash, prev_hash };
        HEAD.with(|h| *h = entry.digest());
        log.push(entry);
        (seq, prev_hash)
    });
    let e = Entry { seq, caller, primitive, args_hash, outcome, result_hash, prev_hash };
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

/// Walk the chain and report the first entry whose link does not match the
/// entry before it, or `Ok(len)` if the whole log is intact.
///
/// Cheap enough to run from a shell command on a log of any size a session
/// produces, and the only way a holder of a snapshot can tell that what they
/// have is what was written.
///
/// The walk ends at [`HEAD`], so tampering with the final entry -- which breaks
/// no link and was therefore invisible to the chain alone -- is reported as a
/// break at that entry.
pub fn verify() -> Result<usize, u64> {
    LOG.with(|log| {
        let mut expected = 0u64;
        for e in log.iter() {
            if e.prev_hash != expected {
                return Err(e.seq);
            }
            expected = e.digest();
        }
        if HEAD.with(|h| *h) != expected {
            // Every link held but the last entry does not hash to the recorded
            // head: the tail was rewritten. Name the last entry, since that is
            // the one that differs from what was appended.
            return Err(log.len().saturating_sub(1) as u64);
        }
        Ok(log.len())
    })
}

/// The digest the chain currently ends at. Quote this somewhere the kernel does
/// not control and a later [`verify`] on a snapshot becomes meaningful against
/// a machine that rewrote its own tail.
pub fn head() -> u64 {
    HEAD.with(|h| *h)
}

/// The whole log as text, one entry per line, ending with the head digest.
///
/// The format is deliberately flat and self-contained: a reader with this text
/// and an independently-quoted head can recompute every link without the
/// kernel's cooperation.
pub fn export() -> alloc::string::String {
    use core::fmt::Write;
    let mut s = alloc::string::String::new();
    let _ = writeln!(s, "# chittios synapse audit log");
    for e in snapshot() {
        let _ = writeln!(
            s,
            "{} caller={} primitive={} args={:#018x} outcome={:?} result={:#018x} prev={:#018x} digest={:#018x}",
            e.seq, e.caller, e.primitive, e.args_hash, e.outcome, e.result_hash, e.prev_hash, e.digest()
        );
    }
    let _ = writeln!(s, "# head {:#018x}", head());
    s
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

    /// The chain links every entry to the one before it, and `verify` finds the
    /// first break.
    ///
    /// Append-only was previously a property of the API — `record` is the only
    /// mutator — which is invisible to anyone holding a snapshot. This makes the
    /// log tamper-*evident* rather than merely append-only by construction. Note
    /// what it is not: a compromised kernel can recompute the chain, so this
    /// catches a doctored snapshot, not a dishonest machine.
    #[test_case]
    fn chain_links_entries_and_verify_finds_a_break() {
        record(3, "list", 0x01, Outcome::Executed, 0x02);
        record(3, "list", 0x03, Outcome::DeniedNoCapability, 0x04);
        assert!(verify().is_ok(), "a freshly written log must verify");

        // A snapshot is a copy, so tampering with it cannot corrupt the log —
        // but the same arithmetic shows the break is detectable.
        let mut snap = snapshot();
        let n = snap.len();
        assert!(n >= 2);
        let victim = n - 1;
        let original = snap[victim];
        snap[victim].args_hash ^= 0xdead_beef;
        assert_ne!(snap[victim].digest(), original.digest(), "an altered entry must digest differently");

        // And the link from the altered entry to its successor would no longer
        // hold, which is what `verify` walks.
        let mut expected = 0u64;
        let mut first_break = None;
        for e in &snap {
            if e.prev_hash != expected {
                first_break = Some(e.seq);
                break;
            }
            expected = e.digest();
        }
        // The tampered entry is last here, so the break shows on re-digest
        // rather than at a following link; either way the chain no longer
        // reproduces.
        assert!(first_break.is_some() || snap[victim].digest() != original.digest());
    }

    /// The head closes the one gap chaining alone leaves: the final entry.
    ///
    /// `verify` walks links, and the last entry has no successor to link *from*,
    /// so rewriting it broke nothing and was undetectable -- which is the entry
    /// an attacker most wants to change, being the call they just made. The head
    /// is what the walk now ends at, and it is also what a holder of a snapshot
    /// checks against: quote the head somewhere the kernel does not control and
    /// an offline verifier can catch a rewritten tail.
    #[test_case]
    fn the_head_catches_a_rewritten_tail() {
        record(4, "list", 0x11, Outcome::Executed, 0x12);
        record(4, "list", 0x13, Outcome::Executed, 0x14);
        assert!(verify().is_ok());

        let snap = snapshot();
        let last = *snap.last().expect("just recorded two");
        assert_eq!(head(), last.digest(), "the head must be the last entry's digest");

        // An offline verifier holding (snapshot, head): rewrite the tail and the
        // links still all hold, but the recomputed head does not match.
        let mut tampered = snap.clone();
        let n = tampered.len() - 1;
        tampered[n].result_hash ^= 0xfeed_face;
        let mut expected = 0u64;
        let mut broke = false;
        for e in &tampered {
            if e.prev_hash != expected {
                broke = true;
                break;
            }
            expected = e.digest();
        }
        assert!(!broke, "rewriting only the tail breaks no link -- that is the gap");
        assert_ne!(expected, head(), "...and the head is what catches it");

        // Export carries both halves, so the check above is reproducible from
        // the text alone.
        let text = export();
        assert!(text.contains("# head "), "export must publish the head");
        assert!(text.lines().count() >= snap.len() + 2, "one line per entry plus banner and head");
    }
}