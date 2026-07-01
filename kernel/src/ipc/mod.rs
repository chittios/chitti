//! Capability-gated message passing between tasks (`CHITTI_OS_HANDOFF.md`
//! Phase 2). Modeled as seL4-style endpoints: a task can `send`/`receive`
//! on an endpoint only by holding a matching `cap::Right` naming it --
//! there is no API that lets a task name another task directly, so every
//! IPC path is capability-gated by construction, not by a runtime check
//! bolted on afterward.

use crate::cap::{self, Cap, EndpointId, Right};
use crate::mm::Locked;
use crate::sched::{self, TaskId};
use alloc::collections::{BTreeMap, VecDeque};
use core::sync::atomic::{AtomicU64, Ordering};

pub struct Message {
    pub sender: TaskId,
    pub data: u64,
}

struct Endpoint {
    queue: VecDeque<Message>,
}

static NEXT_ENDPOINT: AtomicU64 = AtomicU64::new(0);
static ENDPOINTS: Locked<BTreeMap<EndpointId, Endpoint>> = Locked::new(BTreeMap::new());

/// Create a fresh endpoint with an empty queue. Kernel/setup-mediated,
/// same as `cap::grant`: creating an endpoint says nothing about who may
/// use it -- that's decided entirely by who subsequently gets a `Cap`
/// naming it.
pub fn create_endpoint() -> EndpointId {
    let id = NEXT_ENDPOINT.fetch_add(1, Ordering::SeqCst);
    ENDPOINTS.with(|eps| {
        eps.insert(id, Endpoint { queue: VecDeque::new() });
    });
    id
}

#[derive(Debug, PartialEq, Eq)]
pub enum IpcError {
    CapabilityDenied,
}

/// Send `data` on the endpoint named by `cap`, which must be a
/// `Right::IpcSend` held in the *calling* task's own capability table.
pub fn send(cap: Cap, data: u64) -> Result<(), IpcError> {
    let task = sched::current_task_id();
    match cap::lookup(task, cap) {
        Some(Right::IpcSend(ep)) => {
            ENDPOINTS.with(|eps| {
                eps.get_mut(&ep)
                    .expect("ipc::send: endpoint disappeared")
                    .queue
                    .push_back(Message { sender: task, data });
            });
            Ok(())
        }
        _ => {
            cap::record_denial(task, "ipc::send");
            Err(IpcError::CapabilityDenied)
        }
    }
}

/// Non-blocking receive: `Ok(None)` if the endpoint is empty right now.
pub fn try_receive(cap: Cap) -> Result<Option<Message>, IpcError> {
    let task = sched::current_task_id();
    match cap::lookup(task, cap) {
        Some(Right::IpcReceive(ep)) => {
            Ok(ENDPOINTS.with(|eps| eps.get_mut(&ep).expect("ipc::receive: endpoint disappeared").queue.pop_front()))
        }
        _ => {
            cap::record_denial(task, "ipc::receive");
            Err(IpcError::CapabilityDenied)
        }
    }
}

/// Blocking receive: cooperatively yields while the endpoint is empty.
/// Bounded so a broken scheduler/IPC path fails the test loudly instead
/// of hanging the whole QEMU run forever (same pattern as the Phase 1
/// timer test).
pub fn receive(cap: Cap) -> Result<Message, IpcError> {
    let mut spins = 0u64;
    loop {
        if let Some(msg) = try_receive(cap)? {
            return Ok(msg);
        }
        sched::yield_now();
        spins += 1;
        assert!(spins < 100_000_000, "ipc::receive: no message ever arrived");
    }
}
