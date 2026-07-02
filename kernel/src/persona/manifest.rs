//! Agent **manifest** (`CHITTI_OS_HANDOFF.md` Phase 5): the static
//! declaration of what an agent *is*, analogous to an executable's header.
//! An agent is spawned *from* a manifest, and the manifest is exactly the
//! part of an agent that survives a suspend/resume (its identity, authority,
//! and policy) -- as opposed to the derived, recomputable KV/live state,
//! which is not.
//!
//! Four fields, matching the phase spec: the model it reasons with, its
//! persona/system prompt, the capability set it may exercise, and its memory
//! policy.

use crate::cap::PrimitiveId;
use alloc::string::String;
use alloc::vec::Vec;

/// Which inference backend an agent plans with. The planner is the
/// *stochastic* layer above the determinism boundary; below it everything
/// (Synapse, Cortex numerics) is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// The Cortex-attached GGUF model (Phase 3). Real inference; used when a
    /// model boot module is present.
    Cortex,
    /// No model attached: the agent plans with the deterministic rule-based
    /// planner. This is what the fast, model-free test suite and the boot
    /// demo use (the real 0.8B model is far too slow under QEMU TCG to drive
    /// a multi-step plan in a test).
    Deterministic,
}

/// How an agent manages its two-tier memory (`persona::memory`).
#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    /// Soft cap on messages kept in the *live* working set (tier 1). Older
    /// messages beyond this are evicted from live context; anything the
    /// agent wants to keep must be written to the persistent store (tier 2).
    pub working_set_limit: usize,
    /// Whether the agent may demand-page (recall) facts from the persistent
    /// store that are not currently in its live context.
    pub recall_enabled: bool,
}

impl MemoryPolicy {
    pub const DEFAULT: MemoryPolicy = MemoryPolicy { working_set_limit: 32, recall_enabled: true };
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The static description an agent is spawned from.
pub struct Manifest {
    /// Stable agent name; also the namespace for its persistent memory.
    pub name: String,
    pub model: ModelRef,
    /// The persona / system prompt that frames the agent's behaviour.
    pub persona_prompt: String,
    /// The primitives this agent is authorised to invoke. `Agent::spawn`
    /// turns each into a `cap::Right::InvokePrimitive` grant on the agent's
    /// own task -- no ambient authority beyond this set.
    pub capabilities: Vec<PrimitiveId>,
    pub memory: MemoryPolicy,
}

impl Manifest {
    /// A manifest for `name` with an explicit capability set, the
    /// deterministic planner, and the default memory policy.
    pub fn new(name: &str, persona_prompt: &str, capabilities: Vec<PrimitiveId>) -> Self {
        Self {
            name: String::from(name),
            model: ModelRef::Deterministic,
            persona_prompt: String::from(persona_prompt),
            capabilities,
            memory: MemoryPolicy::DEFAULT,
        }
    }
}
