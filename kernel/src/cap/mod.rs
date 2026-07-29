//! Unforgeable capability tokens (seL4-inspired), the security substrate
//! of `CHITTI_OS_HANDOFF.md` Phase 2. A `Cap` is nothing but an opaque
//! index into the *owning task's own* capability table; there is no way
//! to construct one from a raw integer or to reach into another task's
//! table. A task can only ever exercise a `Right` the kernel explicitly
//! `grant`ed it -- no ambient authority, no capability by convention.

use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::mm::Locked;
use crate::sched::{self, TaskId};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Identifies an IPC endpoint (`ipc::create_endpoint`). Defined here
/// rather than in `ipc` so `Right` can name one structurally without a
/// circular module dependency.
pub type EndpointId = u64;

/// Identifies a `synapse` capability primitive (`synapse::registry`).
/// Defined here for the same reason as `EndpointId`: `Right` can name a
/// primitive structurally (as a small opaque id) without `cap` -- a Phase 2
/// module -- having to depend upward on the Phase 4 `synapse` layer.
pub type PrimitiveId = u16;

/// Identifies an inter-agent byte/datagram channel (`channel::create`).
/// Defined here (not in `channel`) for the same structural-naming reason as
/// [`EndpointId`]: `Right` can name a channel without a circular dependency on
/// the `channel` module. A channel end is granted per-direction, so a channel
/// handle held by a task is unforgeable in exactly the way an IPC endpoint cap
/// is — see [`Right::ChannelWrite`]/[`Right::ChannelRead`].
pub type ChannelId = u64;

/// Identifies a network listener (`net::listen`). Same structural-naming
/// rationale as [`EndpointId`]/[`ChannelId`].
pub type ListenerId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Right {
    IpcSend(EndpointId),
    IpcReceive(EndpointId),
    /// Authority to invoke one Synapse primitive (Phase 4). An agent may
    /// only call a primitive it holds the matching `InvokePrimitive` right
    /// for -- no ambient authority over the capability ABI.
    InvokePrimitive(PrimitiveId),
    /// Authority to push bytes/datagrams into a channel. Held per-direction:
    /// granting the write end never confers the read end, so attenuation is
    /// structural (exactly like `IpcSend` vs `IpcReceive`). A channel handle
    /// argument the model emits is resolved as a `Cap` slot in the caller's
    /// own table (`synapse::executor`), so a guessed integer only ever indexes
    /// the caller's own capability space — no ambient authority over channels.
    ChannelWrite(ChannelId),
    /// Authority to pull bytes/datagrams out of a channel. See
    /// [`Right::ChannelWrite`].
    ChannelRead(ChannelId),
    /// Authority to accept inbound connections on a network listener
    /// (`net::listen`/`net_accept`). Minted by `net_listen` into the caller's
    /// own table, resolved as a `Cap` slot the same way channel ends are.
    NetListen(ListenerId),
}

/// An opaque handle into the holder's own `CapTable`. The index is
/// `pub(crate)` (not `pub`): within the kernel it's plumbed around like
/// seL4's CPtr (a plain integer, meaningful only relative to its owning
/// task's own table), but no code outside this crate could construct or
/// inspect one, and no API anywhere takes "some other task's table" --
/// only ever the caller's own (`sched::current_task_id()`). That's what
/// makes a `Cap` unforgeable in practice: guessing an index only ever
/// searches your own capability space, never anyone else's.
#[derive(Clone, Copy, Debug)]
pub struct Cap(pub(crate) u32);

#[derive(Default)]
pub struct CapTable {
    slots: Vec<Option<Right>>,
}

impl CapTable {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn insert(&mut self, right: Right) -> Cap {
        self.slots.push(Some(right));
        Cap((self.slots.len() - 1) as u32)
    }

    fn lookup(&self, cap: Cap) -> Option<Right> {
        self.slots.get(cap.0 as usize).copied().flatten()
    }

    /// Whether this table contains a slot granting exactly `right`.
    ///
    /// Unlike `lookup` (which resolves a specific `Cap` index the holder
    /// already possesses), this answers "does the holder hold *some*
    /// capability conferring this authority?" -- the shape the Synapse tool
    /// ABI needs, where an agent names a primitive rather than a slot index.
    /// It still only ever scans the caller's *own* table (see `holds`), so
    /// it grants no way to reach into another task's capability space.
    fn grants(&self, right: Right) -> bool {
        self.slots.iter().any(|slot| *slot == Some(right))
    }

    /// Clear a specific slot (used by transfer-style `channel_grant`).
    fn revoke_slot(&mut self, cap: Cap) -> Option<Right> {
        let i = cap.0 as usize;
        if i >= self.slots.len() {
            return None;
        }
        self.slots[i].take()
    }
}

/// Count of denied capability checks, incremented (and `ktrace`d)
/// alongside every failed lookup. The Phase 2 acceptance test asserts
/// this to prove a missing capability is actually denied, not silently
/// allowed.
static DENIALS: AtomicU64 = AtomicU64::new(0);

pub fn denials() -> u64 {
    DENIALS.load(Ordering::SeqCst)
}

pub(crate) fn record_denial(task: TaskId, operation: &str) {
    DENIALS.fetch_add(1, Ordering::SeqCst);
    crate::ktrace::log_fmt(format_args!("cap: task {task} denied {operation} -- no matching capability"));
}

/// Kernel-mediated grant. Only code with access to this module (kernel
/// subsystems and test/setup code, not arbitrary task logic) can hand a
/// task a new right; ordinary task code has no API to mint or copy a
/// `Cap` on its own.
pub fn grant(task: TaskId, right: Right) -> Cap {
    sched::with_cap_table_mut(task, |table| table.insert(right))
}

/// Revoke a specific capability slot in `task`'s own table. Returns the
/// right that was held, or `None` if the slot was empty / OOB. Used by
/// transfer-style operations (`channel_grant`) so authority is moved, not
/// duplicated.
pub fn revoke(task: TaskId, cap: Cap) -> Option<Right> {
    sched::with_cap_table_mut(task, |table| table.revoke_slot(cap))
}

/// Look up what `cap` grants in `task`'s own table. Used by `ipc` to gate
/// send/receive; not exposed as a way to inspect another task's table
/// (`task` here is always `sched::current_task_id()` at the call site).
pub(crate) fn lookup(task: TaskId, cap: Cap) -> Option<Right> {
    sched::with_cap_table_mut(task, |table| table.lookup(cap))
}

/// Whether `task` holds a capability conferring `right`. `synapse` uses
/// this to gate every tool call: an agent may invoke a primitive only if
/// its own capability table grants `Right::InvokePrimitive(id)`. As with
/// `lookup`, `task` is always the caller (`sched::current_task_id()`), so
/// this never inspects another task's authority.
pub fn holds(task: TaskId, right: Right) -> bool {
    sched::with_cap_table_mut(task, |table| table.grants(right))
}

// --- Scope ledger (fine-grained, per-task) ----------------------------------
//
// The live `Right`/`CapTable` is primitive-granularity ("may you call
// mem_fs_write at all"). The *scope* of a granted capability (which paths, which
// hosts/ports) is recorded here alongside it, and the Synapse executor's Gate
// 2.5 consults it to enforce path/host/port limits the manifest declared. Kept
// as a side table so the `Copy` `Right` enum stays small and unchanged.
//
// Enforcement is deny-only-when-recorded: a task with NO ledger entry for a
// domain is not scope-constrained for it (preserves the behaviour of the many
// call sites that `grant` an `InvokePrimitive` directly without scopes). A
// domain present in the ledger is enforced — a `Scope::Any` entry (the common
// grant) covers everything, so it passes; a narrow entry bites.

static SCOPES: Locked<BTreeMap<TaskId, Vec<CapabilityRequest>>> = Locked::new(BTreeMap::new());

/// Record the granted capability scopes for `task` (called by
/// `agent::manifest::grant_to_task`, the one place declarative caps become live
/// authority). Appends, so multiple grants accumulate.
pub fn grant_scopes(task: TaskId, caps: &[CapabilityRequest]) {
    if caps.is_empty() {
        return;
    }
    SCOPES.with(|m| m.entry(task).or_default().extend_from_slice(caps));
}

/// Gate 2.5 predicate: may `task` exercise `want` rights in `domain` on the
/// concrete `target` scope? Returns `true` (allow) unless `domain` is present in
/// the task's ledger and no recorded entry covers the target with the needed
/// rights. See the module note on deny-only-when-recorded.
pub fn scope_check(task: TaskId, domain: CapDomain, want: Rights, target: &Scope) -> bool {
    SCOPES.with(|m| match m.get(&task) {
        None => true,
        Some(caps) => {
            let mut constrained = false;
            for c in caps.iter().filter(|c| c.domain == domain) {
                constrained = true;
                if c.rights.contains(want) && c.scope.covers(target) {
                    return true;
                }
            }
            !constrained // no entry for this domain -> not scope-constrained
        }
    })
}

/// Drop a task's scope ledger (called on `kill`, alongside the cap-table wipe).
pub fn clear_scopes(task: TaskId) {
    SCOPES.with(|m| {
        m.remove(&task);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capability gate, pinned in isolation.
    ///
    /// It had no test of its own until a per-component count (paper E5) made
    /// that visible. It was never *untested* -- the acceptance suite and every
    /// corpus run go through it -- but nothing here pinned the one gate whose
    /// entire job is "holds this right, or does not", and an aggregate test
    /// count hid that. The asymmetry is what matters: granting one right must
    /// not confer a neighbouring one, and revocation must actually revoke.
    #[test_case]
    fn a_grant_confers_exactly_one_right_and_revocation_takes_it_back() {
        let task = crate::sched::spawn_parked("cap-test");

        assert!(!holds(task, Right::InvokePrimitive(7)), "a fresh table grants nothing");
        let cap = grant(task, Right::InvokePrimitive(7));
        assert!(holds(task, Right::InvokePrimitive(7)));
        // Holding one primitive must not imply the next one along.
        assert!(!holds(task, Right::InvokePrimitive(8)));
        assert!(!holds(task, Right::InvokePrimitive(6)));

        assert_eq!(lookup(task, cap), Some(Right::InvokePrimitive(7)));
        assert_eq!(revoke(task, cap), Some(Right::InvokePrimitive(7)));
        assert!(!holds(task, Right::InvokePrimitive(7)), "revocation must actually revoke");
        assert_eq!(revoke(task, cap), None, "revoking twice is not a second grant");

        let _ = crate::sched::kill(task);
    }

    /// Deny-only-when-recorded, which is the rule the scope ledger is built on
    /// and the one most likely to be broken by a well-meaning change: a task
    /// with no ledger entry is unconstrained, and a recorded narrow entry bites.
    #[test_case]
    fn the_scope_ledger_constrains_only_what_it_records() {
        use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
        let task = crate::sched::spawn_parked("scope-test");
        let target = Scope::Path(alloc::string::String::from("/elsewhere/x"));

        // No ledger entry for the domain: unconstrained.
        assert!(scope_check(task, CapDomain::Fs, Rights::READ, &target));

        grant_scopes(
            task,
            &[CapabilityRequest::new(
                CapDomain::Fs,
                Rights::READ,
                Scope::Path(alloc::string::String::from("/home/**")),
            )],
        );
        assert!(!scope_check(task, CapDomain::Fs, Rights::READ, &target), "a recorded narrow scope must bite");
        assert!(scope_check(
            task,
            CapDomain::Fs,
            Rights::READ,
            &Scope::Path(alloc::string::String::from("/home/notes.txt"))
        ));
        // A domain the ledger says nothing about stays unconstrained even once
        // another domain is recorded.
        assert!(scope_check(task, CapDomain::Net, Rights::READ, &target));

        clear_scopes(task);
        assert!(scope_check(task, CapDomain::Fs, Rights::READ, &target), "clearing restores unconstrained");
        let _ = crate::sched::kill(task);
    }
}
