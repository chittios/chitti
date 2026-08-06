//! **Synapse** -- the capability ABI (`CHITTI_OS_HANDOFF.md` Phase 4), the
//! syscall layer of Chitti. It turns an *untrusted plan* (tokens a model
//! emitted) into *deterministic, capability-checked effects*, and is the
//! concrete enforcement of the determinism boundary from Part 2: above
//! Synapse everything is stochastic; at and below it, everything is
//! deterministic, gated, and audited.
//!
//! The pieces, in the order a call flows through them:
//!
//! * [`registry`] -- the fixed, MCP-shaped catalogue of primitives (name +
//!   input schema). The single source of truth the grammar and executor read.
//! * [`grammar`] -- the constraint grammar *generated from* the registry.
//!   It both validates a completed call (`grammar::parse`) and, via
//!   [`grammar::ConstrainedDecoder`], masks a model's token stream so only a
//!   well-formed call can ever be emitted (the tie to `cortex::sampler`).
//! * [`executor`] -- the one path that runs a call: grammar gate, then
//!   capability gate (`cap::Right::InvokePrimitive`), then isolated native
//!   execution. Model output reaches a primitive *only* through here.
//! * [`fs`] -- the in-memory store the FS primitives mutate.
//! * [`audit`] -- the append-only log every attempt (executed, denied, or
//!   rejected) is recorded in.
//!
//! Phase 4 is capability checks only; provenance/taint gating is Phase 6.

pub mod abi;
pub mod citation;
pub mod chunked;
pub mod tenant;
pub mod audit;
pub mod bench;
pub mod executor;
pub mod fs;
pub mod grammar;
pub mod policy;
pub mod registry;
pub mod ui;
pub mod vpath;

pub use executor::{execute, execute_current, execute_with_justification, Invocation};

use crate::cap::{self, Right};
use crate::sched;

/// Boot-time demonstration of the Phase 4 deliverable, driven from the
/// bootstrap task: a scripted agent grants itself a subset of primitives,
/// emits tool calls as canonical JSON, and the kernel validates, gates,
/// executes, and audits each one. Fast and model-free, so it runs on every
/// boot (unlike the slow Cortex inference demo) to show the ABI end to end
/// on serial. Returns nothing; every step is visible via `ktrace`/serial.
pub fn demo() {
    let me = sched::current_task_id();
    crate::serial_println!("Chitti: --- Synapse capability ABI ---");

    // Grant this agent authority over a subset of primitives. Note we do
    // *not* grant `spawn_agent`: the denial below proves the gate bites.
    for id in [registry::CONSOLE_WRITE, registry::MEM_FS_WRITE, registry::MEM_FS_READ, registry::LIST] {
        cap::grant(me, Right::InvokePrimitive(id));
    }

    let calls = [
        r#"{"name":"mem_fs_write","arguments":{"path":"notes","text":"hello from an agent"}}"#,
        r#"{"name":"mem_fs_read","arguments":{"path":"notes"}}"#,
        r#"{"name":"list","arguments":{}}"#,
        r#"{"name":"console_write","arguments":{"text":"agent says hi"}}"#,
        // Denied: capability for spawn_agent was never granted.
        r#"{"name":"spawn_agent","arguments":{"persona":"helper"}}"#,
        // Rejected by the grammar: unregistered primitive.
        r#"{"name":"delete_everything","arguments":{}}"#,
    ];

    for raw in calls {
        match execute(me, raw) {
            Invocation::Executed { primitive, result } => {
                crate::serial_println!("Chitti: synapse [{}] -> {}", primitive, result);
            }
            Invocation::Denied { primitive } => {
                crate::serial_println!("Chitti: synapse [{}] DENIED (no capability)", primitive);
            }
            Invocation::Rejected(err) => {
                crate::serial_println!("Chitti: synapse rejected malformed call: {:?}", err);
            }
            Invocation::RefusedTainted { primitive } => {
                crate::serial_println!("Chitti: synapse [{}] REFUSED (tainted justification)", primitive);
            }
            Invocation::NeedsApproval { primitive } => {
                crate::serial_println!("Chitti: synapse [{}] NEEDS HUMAN APPROVAL", primitive);
            }
            Invocation::DeniedScope { primitive } => {
                crate::serial_println!("Chitti: synapse [{}] DENIED (target outside granted scope)", primitive);
            }
        }
    }
    crate::serial_println!("Chitti: synapse audit log has {} entries", audit::len());
}
