//! The **orchestrator** (`CHITTI_AGENTIC_HANDOFF.md` Phase A): the session's
//! foreground main agent. It owns a [`Session`], holds the live capabilities,
//! and drives the [`agent_loop`](super::agent_loop) over a [`StepSource`] and a
//! [`ToolDispatch`]. The Intent shell talks to the orchestrator; it never talks
//! to tools or sub-agents directly (locked decision).
//!
//! Phase A ships a compact [`SynapseTools`] dispatcher covering the core
//! builtin tools over Synapse (write/read/list/delete/console/emit_result, plus
//! session-local `todo_write`). Phase B promotes this to the full MCP-shaped
//! `tools/` registry; the [`ToolDispatch`] trait is the stable seam so the loop
//! is unchanged.

use crate::agent::agent_loop::{self, LoopResult, StepSource, ToolDispatch, ToolOutcome};
use crate::agent::manifest;
use crate::agent::types::*;
use crate::security::taint::{self, Justification};
use crate::session::{self, todo};
use crate::synapse::{self, executor::Invocation};
use alloc::string::{String, ToString};

/// Monotonic tick source (kernel ms). Ticks only order events; tests don't
/// assert on their absolute value.
pub fn now() -> Ticks {
    crate::arch::now_ms() as Ticks
}

/// Map a session-level [`Provenance`] to the taint gate's 3-variant provenance.
/// `SkillInstalled` is trusted to *steer* the agent (so it is not treated as
/// untrusted-ingested), while its authority stays bounded by the install grant
/// at the capability layer (handoff invariant #3). See DECISIONS.md.
pub fn to_taint(p: Provenance) -> taint::Provenance {
    match p {
        Provenance::UserTyped => taint::Provenance::UserTyped,
        Provenance::SystemTrusted | Provenance::SkillInstalled(_) => taint::Provenance::SystemTrusted,
        Provenance::UntrustedIngested => taint::Provenance::UntrustedIngested,
    }
}

/// The Phase-A tool dispatcher: builtin tools → Synapse primitives, every
/// effect capability-checked and audited by the executor.
pub struct SynapseTools {
    /// When true, the justification for each call is derived from the session's
    /// current worst provenance (the Phase E injection defense). When false
    /// (Phase A default), calls are trusted.
    pub taint_aware: bool,
    /// Set by the shell when a human has explicitly confirmed a destructive
    /// action at the prompt.
    pub human_confirmed: bool,
}

impl SynapseTools {
    pub fn new() -> Self {
        Self { taint_aware: false, human_confirmed: false }
    }
    pub fn taint_aware() -> Self {
        Self { taint_aware: true, human_confirmed: false }
    }

    fn justification(&self, session: &Session) -> Justification {
        if !self.taint_aware {
            return Justification::trusted();
        }
        let j = Justification::from_context(to_taint(session.resident_max_taint()));
        if self.human_confirmed {
            j.confirmed()
        } else {
            j
        }
    }

    /// Run a canonical Synapse call and turn its `Invocation` into a
    /// `ToolOutcome`. FS/tool output is tagged `UntrustedIngested` — that taint
    /// is what a later destructive call would be justified by.
    fn run_synapse(&self, session: &Session, caller: crate::sched::TaskId, raw: &str) -> ToolOutcome {
        match synapse::execute_with_justification(caller, raw, self.justification(session)) {
            Invocation::Executed { result, .. } => ToolOutcome::ok(result, Provenance::UntrustedIngested),
            Invocation::Denied { primitive } => {
                ToolOutcome::error(alloc::format!("denied: no capability for {primitive}"))
            }
            Invocation::Rejected(err) => ToolOutcome::error(alloc::format!("rejected: {err:?}")),
            Invocation::RefusedTainted { primitive } => {
                ToolOutcome::error(alloc::format!("refused: destructive '{primitive}' justified by untrusted content"))
            }
        }
    }
}

impl Default for SynapseTools {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolDispatch for SynapseTools {
    fn call(&mut self, session: &mut Session, caller: crate::sched::TaskId, call: &ToolCall) -> ToolOutcome {
        let a = &call.args;
        match call.tool.as_str() {
            "write" => {
                let (Some(path), Some(content)) = (todo::json_str(a, "path"), todo::json_str(a, "content")) else {
                    return ToolOutcome::error("write: needs {path, content}");
                };
                self.run_synapse(session, caller, &synapse_call("mem_fs_write", &[("path", &path), ("text", &content)]))
            }
            "read" => {
                let Some(path) = todo::json_str(a, "path") else {
                    return ToolOutcome::error("read: needs {path}");
                };
                self.run_synapse(session, caller, &synapse_call("mem_fs_read", &[("path", &path)]))
            }
            "list" => self.run_synapse(session, caller, &synapse_call("list", &[])),
            "delete" => {
                let Some(path) = todo::json_str(a, "path") else {
                    return ToolOutcome::error("delete: needs {path}");
                };
                self.run_synapse(session, caller, &synapse_call("mem_fs_delete", &[("path", &path)]))
            }
            "console" => {
                let text = todo::json_str(a, "text").unwrap_or_default();
                self.run_synapse(session, caller, &synapse_call("console_write", &[("text", &text)]))
            }
            "emit_result" => {
                let text = todo::json_str(a, "text").unwrap_or_default();
                self.run_synapse(session, caller, &synapse_call("emit_result", &[("text", &text)]))
            }
            "todo_write" => {
                let items = todo::parse_args(a);
                let remaining = todo::write(session, items, now());
                ToolOutcome::ok(alloc::format!("ok:{remaining} remaining"), Provenance::SystemTrusted)
            }
            other => ToolOutcome::error(alloc::format!("unknown or not-yet-wired tool: {other}")),
        }
    }
}

/// Build a canonical Synapse tool-call JSON: `{"name":..,"arguments":{..}}`,
/// with string values escaped for the grammar.
pub fn synapse_call(name: &str, args: &[(&str, &str)]) -> String {
    let mut s = String::from("{\"name\":\"");
    s.push_str(name);
    s.push_str("\",\"arguments\":{");
    for (i, (k, v)) in args.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(k);
        s.push_str("\":\"");
        json_escape_into(&mut s, v);
        s.push('"');
    }
    s.push_str("}}");
    s
}

fn json_escape_into(out: &mut String, v: &str) {
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
}

/// The session's foreground main agent.
pub struct Orchestrator {
    pub manifest: AgentManifest,
    pub session: Session,
    /// The dedicated task whose capability table gates every tool call.
    pub caller: crate::sched::TaskId,
}

impl Orchestrator {
    /// Spawn an orchestrator for `manifest` on its own capability-owning task,
    /// granting its full capability set there (the effective live caps == the
    /// manifest's — the orchestrator is the root, nothing above it to attenuate
    /// against).
    pub fn spawn(manifest: AgentManifest, seed: u64) -> Self {
        let caller = crate::sched::spawn_parked("orchestrator");
        let live = manifest::grant_to_task(caller, &manifest.capabilities);
        let session = Session::new(&manifest, seed, live, now());
        crate::ktrace::log_fmt(format_args!(
            "orchestrator.spawn: session {} manifest '{}' caps={} task={}",
            session.id.0,
            manifest.name,
            manifest.capabilities.len(),
            caller
        ));
        Self { manifest, session, caller }
    }

    /// Re-attach an orchestrator to a resumed `session`: spawn a fresh
    /// capability-owning task and re-grant the session's live caps to it (the
    /// KV cache is recomputed on demand, not restored). This is how a persisted
    /// session continues.
    pub fn from_session(manifest: AgentManifest, session: Session) -> Self {
        let caller = crate::sched::spawn_parked("orchestrator");
        let granted: alloc::vec::Vec<CapabilityRequest> = session.capabilities.iter().map(|c| c.req.clone()).collect();
        manifest::grant_to_task(caller, &granted);
        crate::ktrace::log_fmt(format_args!(
            "orchestrator.resume: session {} re-attached on task {}",
            session.id.0, caller
        ));
        Self { manifest, session, caller }
    }

    /// Handle one user intent: record it (as `UserTyped`), run the loop to a
    /// stop condition, persist the session, and return the loop result.
    pub fn handle(&mut self, intent: &str, steps: &mut dyn StepSource, tools: &mut dyn ToolDispatch) -> LoopResult {
        self.session.push_message(Role::User, intent.to_string(), Provenance::UserTyped, now());
        let result = agent_loop::run(&mut self.session, steps, tools, self.caller, now);
        let _ = session::save(&self.session);
        result
    }

    /// Tear down the agent: mark it done (its parked task's stack is reclaimed
    /// by the scheduler's existing dead-task policy, as with `persona`).
    pub fn kill(&mut self) {
        crate::ktrace::log_fmt(format_args!("orchestrator.kill: session {} (task {})", self.session.id.0, self.caller));
    }
}
