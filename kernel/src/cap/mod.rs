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

/// Whether `task` holds a capability conferring `right`. `synapse` uses
/// this to gate every tool call: an agent may invoke a primitive only if
/// its own capability table grants `Right::InvokePrimitive(id)`. As with
/// `lookup`, `task` is always the caller (`sched::current_task_id()`), so
/// this never inspects another task's authority.
pub fn holds(task: TaskId, right: Right) -> bool {
    sched::with_cap_table_mut(task, |table| table.grants(right))
}
