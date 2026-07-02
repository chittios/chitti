//! Two-tier agent memory (`CHITTI_OS_HANDOFF.md` Phase 5).
//!
//! * **Tier 1 -- live context ([`Context`]).** The agent's working set: the
//!   messages currently "in RAM". In a model-backed agent this is what the
//!   Cortex KV cache is derived from. It is bounded by the manifest's
//!   `working_set_limit`; messages beyond it are evicted from live context.
//!
//! * **Tier 2 -- persistent store.** The agent's "disk": durable key/value
//!   facts, backed by the Synapse in-memory FS (`synapse::fs`) under a
//!   per-agent namespace. It is *not* bounded by the working set and
//!   survives suspend/resume and context eviction.
//!
//! The bridge between them is **demand-paging / RAG-style recall**: when an
//! agent references a fact that is not in its live context, [`recall`] pages
//! it in from the persistent store. That is the behaviour the phase's
//! acceptance test (c) exercises -- an agent answering from a fact it had to
//! fetch, because it was never in its live context.

use crate::synapse::fs;
use alloc::string::String;
use alloc::vec::Vec;

/// Who produced a context message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The persona / system prompt.
    System,
    /// A human (or upstream agent) intent.
    User,
    /// The agent's own reasoning/output.
    Agent,
    /// A tool result or a recalled fact paged in from tier 2.
    Tool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// Tier-1 live working set. Cheap to clone (it is what a checkpoint keeps),
/// and deliberately bounded so live context stays small while the durable
/// store grows without limit.
#[derive(Clone, Debug)]
pub struct Context {
    pub messages: Vec<Message>,
    /// Keys currently paged in from the persistent store (tier 2 -> tier 1).
    pub paged_keys: Vec<String>,
    limit: usize,
}

impl Context {
    pub fn new(limit: usize) -> Self {
        Self { messages: Vec::new(), paged_keys: Vec::new(), limit }
    }

    /// Append a message, evicting the oldest live message(s) if the working
    /// set would exceed its limit. Eviction only drops the *live* copy;
    /// durable facts live in tier 2 and are unaffected.
    pub fn push(&mut self, role: Role, text: &str) {
        self.messages.push(Message { role, text: String::from(text) });
        while self.messages.len() > self.limit {
            self.messages.remove(0);
        }
    }

    /// Whether any live message contains `needle`. Used by the acceptance
    /// test to prove a recalled fact was genuinely absent from live context.
    pub fn contains(&self, needle: &str) -> bool {
        self.messages.iter().any(|m| m.text.contains(needle))
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Persistent-store key for `agent`'s fact `key`. Namespaced per agent so
/// two agents' stores never collide.
fn store_path(agent: &str, key: &str) -> String {
    let mut p = String::from("mem/");
    p.push_str(agent);
    p.push('/');
    p.push_str(key);
    p
}

/// Write a durable fact to tier 2 (survives context eviction and
/// suspend/resume).
pub fn remember(agent: &str, key: &str, value: &str) {
    fs::write(&store_path(agent, key), value.as_bytes());
    crate::ktrace::log_fmt(format_args!("persona.memory: agent {agent} stored fact '{key}' to persistent store"));
}

/// Read a durable fact from tier 2 without paging it into any context.
pub fn persisted(agent: &str, key: &str) -> Option<String> {
    fs::read(&store_path(agent, key)).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// **Demand-page** a fact from the persistent store into `ctx` (tier 2 ->
/// tier 1), returning its value. This is the RAG-style recall the phase
/// requires: the agent references something not in live context, and it is
/// fetched on demand. Returns `None` if no such fact exists.
pub fn recall(agent: &str, key: &str, ctx: &mut Context) -> Option<String> {
    let value = persisted(agent, key)?;
    ctx.push(Role::Tool, &alloc::format!("recall {key} = {value}"));
    ctx.paged_keys.push(String::from(key));
    crate::ktrace::log_fmt(format_args!(
        "persona.memory: agent {agent} recalled '{key}' from persistent store into live context"
    ));
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn context_evicts_beyond_working_set_limit() {
        let mut ctx = Context::new(3);
        for i in 0..5 {
            ctx.push(Role::User, &alloc::format!("msg{i}"));
        }
        assert_eq!(ctx.len(), 3, "live context should be bounded by its limit");
        // Oldest were evicted; newest retained.
        assert!(ctx.contains("msg4"));
        assert!(!ctx.contains("msg0"));
    }

    #[test_case]
    fn recall_pages_persistent_fact_into_context() {
        remember("mem_test_agent", "colour", "teal");
        let mut ctx = Context::new(8);
        assert!(!ctx.contains("teal"), "fact must not start in live context");
        let got = recall("mem_test_agent", "colour", &mut ctx);
        assert_eq!(got.as_deref(), Some("teal"));
        assert!(ctx.contains("teal"), "recall must page the fact into live context");
        assert!(ctx.paged_keys.iter().any(|k| k == "colour"));
        // A miss returns None and pages nothing.
        assert_eq!(recall("mem_test_agent", "absent", &mut ctx), None);
    }
}
