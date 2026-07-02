//! **Persona** -- the agent runtime (`CHITTI_OS_HANDOFF.md` Phase 5): agents
//! as first-class processes. This is the layer that sits at the top of the
//! stack, above the determinism boundary: an agent reasons (stochastically,
//! via a planner) about an intent, then drives deterministic effects through
//! Synapse (Phase 4), which in turn may use Cortex (Phase 3), the scheduler
//! and IPC (Phase 2), and the microkernel (Phase 1).
//!
//! The pieces:
//!
//! * [`manifest`] -- what an agent *is*: model ref, persona prompt,
//!   capability set, memory policy.
//! * [`memory`] -- two-tier memory: a bounded live context (tier 1) and a
//!   durable persistent store with demand-paging recall (tier 2).
//! * [`planner`] -- intent -> plan. The stochastic layer; a deterministic
//!   rule planner stands in for the Cortex model in the test suite.
//! * [`actions`] -- the plan vocabulary (Synapse calls + memory ops).
//! * [`agent`] -- the process itself: lifecycle (spawn/suspend/resume/kill)
//!   and the plan/act loop.
//!
//! The intent **shell** that drives all of this over serial lives in the
//! sibling `crate::shell` module.

pub mod actions;
pub mod agent;
pub mod manifest;
pub mod memory;
pub mod planner;

pub use agent::{Agent, AgentState};
pub use manifest::{Manifest, MemoryPolicy, ModelRef};
pub use planner::{Planner, RulePlanner};

use crate::synapse::registry;
use alloc::vec;

/// A manifest for a general-purpose agent, granting the everyday primitive
/// set (in-memory FS, console, and result reporting). This is what the shell
/// spawns its session agent from.
pub fn default_manifest(name: &str) -> Manifest {
    Manifest::new(
        name,
        "You are a Chitti OS agent. Plan an intent as a short sequence of capability calls and report the result.",
        vec![
            registry::CONSOLE_WRITE,
            registry::MEM_FS_READ,
            registry::MEM_FS_WRITE,
            registry::LIST,
            registry::EMIT_RESULT,
        ],
    )
}
