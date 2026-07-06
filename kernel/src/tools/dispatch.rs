//! The [`Router`]: the real [`ToolDispatch`](crate::agent::agent_loop::ToolDispatch).
//! Every tool call flows: **shape-validate** (tool known + required args
//! present) → lower to a Synapse canonical call → **capability + taint gate**
//! (the Synapse executor) → execute → format `tool_result`. Malformed or
//! ungranted calls are refused before any effect, and every attempt is audited
//! by the executor.
//!
//! Bindings the loop can't satisfy on its own — `spawn_subagent`, `load_skill`,
//! `run` — are delegated to optional hooks the orchestrator installs as later
//! phases land (C/E/F). Until a hook is set, the tool returns a clean "not
//! wired yet" error rather than doing anything unsafe.

use crate::agent::agent_loop::{ToolDispatch, ToolOutcome};
use crate::agent::orchestrator::{synapse_call, to_taint};
use crate::agent::types::{Provenance, Session, ToolCall};
use crate::security::taint::Justification;
use crate::session::todo;
use crate::synapse::{self, executor::Invocation};
use crate::tools::registry::{self, ToolBinding};
use alloc::boxed::Box;
use alloc::string::String;

/// A hook the orchestrator installs to service an agent-layer tool. Takes the
/// session, the caller task, and the parsed tool args; returns the outcome.
pub type ToolHook = Box<dyn FnMut(&mut Session, crate::sched::TaskId, &ToolCall) -> ToolOutcome>;

/// The tool router. Holds the taint policy and the optional agent-layer hooks.
pub struct Router {
    /// When true, each call's justification is derived from the session's worst
    /// resident provenance (the Phase E injection defense).
    pub taint_aware: bool,
    /// Set when a human has confirmed a destructive action at the shell.
    pub human_confirmed: bool,
    /// Phase C: dispatch a sub-agent.
    pub spawn_hook: Option<ToolHook>,
    /// Phase F: load a skill body.
    pub load_skill_hook: Option<ToolHook>,
    /// Phase E: run an intent through the compiled path.
    pub run_hook: Option<ToolHook>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            taint_aware: false,
            human_confirmed: false,
            spawn_hook: None,
            load_skill_hook: None,
            run_hook: None,
        }
    }

    /// A taint-aware router (Phase E): destructive calls justified by untrusted
    /// ingested content are refused at the Synapse gate.
    pub fn taint_aware() -> Self {
        let mut r = Self::new();
        r.taint_aware = true;
        r
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

    fn run_synapse(&self, session: &Session, caller: crate::sched::TaskId, raw: &str) -> ToolOutcome {
        match synapse::execute_with_justification(caller, raw, self.justification(session)) {
            Invocation::Executed { result, .. } => ToolOutcome::ok(result, Provenance::UntrustedIngested),
            Invocation::Denied { primitive } => ToolOutcome::error(alloc::format!("denied: no capability for {primitive}")),
            Invocation::Rejected(err) => ToolOutcome::error(alloc::format!("rejected: {err:?}")),
            Invocation::RefusedTainted { primitive } => {
                ToolOutcome::error(alloc::format!("refused: destructive '{primitive}' justified by untrusted content"))
            }
            Invocation::DeniedScope { primitive } => {
                ToolOutcome::error(alloc::format!("denied: '{primitive}' target outside granted scope"))
            }
        }
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolDispatch for Router {
    fn call(&mut self, session: &mut Session, caller: crate::sched::TaskId, call: &ToolCall) -> ToolOutcome {
        // Shape gate 1: the tool must be a registered tool.
        let Some(def) = registry::get(&call.tool) else {
            return ToolOutcome::error(alloc::format!("unknown tool: {}", call.tool));
        };
        // Shape gate 2: required args must be present (never dispatched otherwise).
        for key in &def.required {
            if todo::json_str(&call.args, key).is_none() && !call.args.contains(&alloc::format!("\"{key}\"")) {
                return ToolOutcome::error(alloc::format!("malformed {}: missing required arg '{key}'", call.tool));
            }
        }

        match &def.binding {
            ToolBinding::Synapse { primitive, arg_map } => {
                // Map the tool's JSON keys to the primitive's parameter keys.
                let mut pairs: alloc::vec::Vec<(&str, String)> = alloc::vec::Vec::new();
                for (tool_key, prim_key) in arg_map.iter() {
                    let v = todo::json_str(&call.args, tool_key).unwrap_or_default();
                    pairs.push((prim_key.as_str(), v));
                }
                let borrowed: alloc::vec::Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
                self.run_synapse(session, caller, &synapse_call(primitive, &borrowed))
            }
            ToolBinding::SessionTodo => {
                let items = todo::parse_args(&call.args);
                let remaining = todo::write(session, items, crate::agent::orchestrator::now());
                ToolOutcome::ok(alloc::format!("ok:{remaining} remaining"), Provenance::SystemTrusted)
            }
            ToolBinding::SpawnSubagent => match self.spawn_hook.as_mut() {
                Some(h) => h(session, caller, call),
                None => ToolOutcome::error("spawn_subagent: not available for this agent"),
            },
            ToolBinding::LoadSkill => match self.load_skill_hook.as_mut() {
                Some(h) => h(session, caller, call),
                None => ToolOutcome::error("load_skill: no skill subsystem wired"),
            },
            ToolBinding::RunIntent => match self.run_hook.as_mut() {
                Some(h) => h(session, caller, call),
                None => ToolOutcome::error("run: no compiled-intent path wired"),
            },
            ToolBinding::Shell { command, destructive } => {
                // Destructive system commands (format/install) are gated exactly
                // like a DELETE: refused when justified by untrusted content and
                // not human-confirmed.
                if *destructive && self.justification(session).blocks_destructive() {
                    return ToolOutcome::error(alloc::format!(
                        "refused: destructive command '/{command}' justified by untrusted content"
                    ));
                }
                let arg = todo::json_str(&call.args, "args").unwrap_or_default();
                let out = crate::shell::run_tool_command(command, &arg);
                ToolOutcome::ok(out, Provenance::UntrustedIngested)
            }
        }
    }
}
