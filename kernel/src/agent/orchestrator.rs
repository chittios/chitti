//! The **orchestrator** (`CHITTI_AGENTIC_HANDOFF.md` Phase A): the session's
//! foreground main agent. It owns a [`Session`], holds the live capabilities,
//! and drives the [`agent_loop`](super::agent_loop) over a [`StepSource`] and a
//! [`ToolDispatch`]. The Intent shell talks to the orchestrator; it never talks
//! to tools or sub-agents directly (locked decision).
//!
//! Tool execution is delegated to the [`tools::Router`](crate::tools::Router)
//! (Phase B) via the [`ToolDispatch`] seam, so the loop is agnostic to how
//! tools are catalogued and validated. This module keeps only the small shared
//! helpers the router needs: the canonical Synapse call builder
//! ([`synapse_call`]) and the provenance bridge ([`to_taint`]).

use crate::agent::agent_loop::{self, LoopResult, StepSource, ToolDispatch};
use crate::agent::manifest;
use crate::agent::types::*;
use crate::security::taint;
use crate::session;
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
