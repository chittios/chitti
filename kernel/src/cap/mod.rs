//! Unforgeable capability tokens (seL4-inspired), the security substrate
//! of `CHITTI_OS_HANDOFF.md` Phase 2. A `Cap` is nothing but an opaque
//! index into the *owning task's own* capability table; there is no way
//! to construct one from a raw integer or to reach into another task's
//! table. A task can only ever exercise a `Right` the kernel explicitly
//! `grant`ed it -- no ambient authority, no capability by convention.

use crate::sched::{self, TaskId};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Identifies an IPC endpoint (`ipc::create_endpoint`). Defined here
/// rather than in `ipc` so `Right` can name one structurally without a
/// circular module dependency.
pub type EndpointId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Right {
    IpcSend(EndpointId),
    IpcReceive(EndpointId),
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

/// Look up what `cap` grants in `task`'s own table. Used by `ipc` to gate
/// send/receive; not exposed as a way to inspect another task's table
/// (`task` here is always `sched::current_task_id()` at the call site).
pub(crate) fn lookup(task: TaskId, cap: Cap) -> Option<Right> {
    sched::with_cap_table_mut(task, |table| table.lookup(cap))
}
