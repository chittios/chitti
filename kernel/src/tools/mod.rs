//! **Tools** — the first-class, MCP-shaped tool layer
//! (`CHITTI_AGENTIC_HANDOFF.md` Phase B). A *tool* is a Synapse primitive
//! wrapped with an agent-facing definition (name, description, JSON-schema
//! input). Tools are the presentation/validation layer over Synapse — they
//! never touch hardware or memory directly (locked invariant #1); a tool call
//! is validated (shape → capability → taint) and then executed by the
//! deterministic Synapse executor, and the structured result is formatted back
//! into the agent's context.
//!
//! * [`registry`] — the tool catalogue (builtin toolset + provider-registered
//!   tools) and per-agent discovery.
//! * [`dispatch`] — the [`Router`], the real
//!   [`ToolDispatch`](crate::agent::agent_loop::ToolDispatch): shape-validate →
//!   Synapse (capability + taint gate) → `tool_result`.
//! * [`provider`] — in-kernel tool-provider modules ("MCP servers") that
//!   register additional toolsets (used by Phase F skill-bundled tools).

pub mod dispatch;
pub mod provider;
pub mod registry;

pub use dispatch::Router;
pub use registry::{ToolBinding, ToolDef};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::agent_loop::ToolDispatch;
    use crate::agent::rule_steps::{args, tool};
    use crate::agent::types::TodoStatus;
    use crate::agent::{manifest, orchestrator};
    use crate::synapse::audit;

    /// (a) A malformed call (missing a required arg) is refused before any
    /// effect — the Synapse audit log does not grow.
    #[test_case]
    fn malformed_call_rejected_before_dispatch() {
        let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 1);
        let mut router = Router::new();
        let before = audit::len();
        // `write` requires {path, content}; omit content.
        let call = tool("write", args(&[("path", "b_x")]));
        let out = router.call(&mut orch.session, orch.caller, &call);
        assert!(out.is_error, "malformed call must be an error");
        assert!(out.result.contains("malformed"));
        assert_eq!(audit::len(), before, "malformed call must never reach the executor/audit");
    }

    /// (b) A tool whose underlying primitive the agent lacks the capability for
    /// is denied by the capability gate and audited.
    #[test_case]
    fn ungranted_tool_denied_and_audited() {
        // The reader role holds READ|LIST but NOT write.
        let mut orch = orchestrator::Orchestrator::spawn(manifest::reader_subagent_manifest(), 2);
        let mut router = Router::new();
        let before = audit::len();
        let call = tool("write", args(&[("path", "b_denied"), ("content", "nope")]));
        let out = router.call(&mut orch.session, orch.caller, &call);
        assert!(out.is_error);
        assert!(out.result.contains("denied"), "expected capability denial, got: {}", out.result);
        // Exactly one audit entry, a capability denial.
        assert_eq!(audit::len(), before + 1);
        assert_eq!(audit::snapshot().last().unwrap().outcome, audit::Outcome::DeniedNoCapability);
        assert!(!crate::synapse::fs::exists("b_denied"), "denied write must not touch the FS");
    }

    /// (c) A valid write+read round-trips through the memory FS and appears in
    /// the audit log.
    #[test_case]
    fn write_read_roundtrips_and_is_audited() {
        let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 3);
        let mut router = Router::new();
        let before = audit::len();
        let w = router.call(&mut orch.session, orch.caller, &tool("write", args(&[("path", "b_rt"), ("content", "roundtrip")])));
        assert!(!w.is_error);
        let r = router.call(&mut orch.session, orch.caller, &tool("read", args(&[("path", "b_rt")])));
        assert!(!r.is_error);
        assert!(r.result.contains("roundtrip"), "read got: {}", r.result);
        assert_eq!(crate::synapse::fs::read("b_rt").as_deref(), Some(&b"roundtrip"[..]));
        assert_eq!(audit::len(), before + 2, "both effects audited");
    }

    /// (d) `todo_write` updates the session todo list.
    #[test_case]
    fn todo_write_updates_session_todos() {
        let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 4);
        let mut router = Router::new();
        let payload = r#"{"todos":[{"id":1,"text":"read the file","status":"done"},{"id":2,"text":"write summary","status":"in_progress"}]}"#;
        let out = router.call(&mut orch.session, orch.caller, &tool("todo_write", payload.into()));
        assert!(!out.is_error);
        assert_eq!(orch.session.todos.len(), 2);
        assert_eq!(orch.session.todos[0].status, TodoStatus::Done);
        assert_eq!(orch.session.todos[1].status, TodoStatus::InProgress);
        assert_eq!(orch.session.todos[1].text, "write summary");
    }

    /// Discovery: an agent only sees the tools its manifest lists.
    #[test_case]
    fn discovery_intersects_toolset() {
        let reader = manifest::reader_subagent_manifest(); // read, list, emit_result
        let seen = registry::for_agent(&reader.toolset);
        assert!(seen.iter().any(|t| t.name == "read"));
        assert!(seen.iter().any(|t| t.name == "list"));
        assert!(!seen.iter().any(|t| t.name == "write"), "reader should not see write");
        assert!(!seen.iter().any(|t| t.name == "delete"));
    }
}
