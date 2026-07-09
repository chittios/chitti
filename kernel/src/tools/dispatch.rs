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
use crate::agent::types::{CapDomain, Provenance, Rights, Scope, Session, ToolCall};
use crate::cap;
use crate::security::taint::Justification;
use crate::session::todo;
use crate::synapse::{self, executor::Invocation, fs as store};
use crate::tools::pathutil;
use crate::tools::registry::{self, McpResourceKind, StoreQueryKind, ToolBinding};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

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
                // `read` with optional line range: full-file through Synapse,
                // then slice in the tools layer (grammar stays path-only).
                if call.tool == "read" {
                    return self.call_read_ranged(session, caller, call);
                }
                // `edit` with replace_all=true: unique-match gate is bypassed
                // only when the model explicitly asks to replace every hit.
                if call.tool == "edit" {
                    let replace_all = todo::json_str(&call.args, "replace_all")
                        .map(|v| v == "true" || v == "1")
                        .unwrap_or_else(|| call.args.contains("\"replace_all\":true") || call.args.contains("\"replace_all\": true"));
                    if replace_all {
                        return self.call_edit_replace_all(session, caller, call);
                    }
                }
                // Map the tool's JSON keys to the primitive's parameter keys.
                let mut pairs: alloc::vec::Vec<(&str, String)> = alloc::vec::Vec::new();
                for (tool_key, prim_key) in arg_map.iter() {
                    let v = todo::json_str(&call.args, tool_key).unwrap_or_default();
                    pairs.push((prim_key.as_str(), v));
                }
                let borrowed: alloc::vec::Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
                self.run_synapse(session, caller, &synapse_call(primitive, &borrowed))
            }
            ToolBinding::StoreQuery { kind } => self.call_store_query(session, caller, call, *kind),
            ToolBinding::SessionTodo => {
                // Plan-mode toggles reuse this binding only as a placeholder in
                // the registry; the real implementation is shell-side.
                if call.tool == "enter_plan_mode" {
                    crate::shell::set_plan_mode(true);
                    return ToolOutcome::ok(
                        "ok: plan mode on — read-only tools only until exit_plan_mode",
                        Provenance::SystemTrusted,
                    );
                }
                if call.tool == "exit_plan_mode" {
                    crate::shell::set_plan_mode(false);
                    return ToolOutcome::ok("ok: plan mode off", Provenance::SystemTrusted);
                }
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
                // Default invoke path (chat Router has no orch hook installed):
                // progressive L0→L1 (+ optional L2 asset) via skills::loader.
                None => default_skill_invoke(session, call),
            },
            ToolBinding::McpResources { kind } => match kind {
                McpResourceKind::List => {
                    let server = todo::json_str(&call.args, "server").unwrap_or_default();
                    let text = crate::mcp::list_resources(if server.is_empty() { None } else { Some(server.as_str()) });
                    ToolOutcome::ok(text, Provenance::UntrustedIngested)
                }
                McpResourceKind::Read => {
                    let server = todo::json_str(&call.args, "server").unwrap_or_default();
                    let uri = todo::json_str(&call.args, "uri").unwrap_or_default();
                    if server.is_empty() || uri.is_empty() {
                        return ToolOutcome::error("mcp_read_resource needs server + uri");
                    }
                    match crate::mcp::read_resource(&server, &uri) {
                        Ok(t) => ToolOutcome::ok(t, Provenance::UntrustedIngested),
                        Err(e) => ToolOutcome::error(e),
                    }
                }
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
            ToolBinding::Mcp { server, tool } => {
                // Forward to the connected MCP server. The whole `arguments`
                // object is passed through as JSON. The result is external
                // content, so it enters context as UntrustedIngested (taint
                // gate applies to anything it later justifies).
                match crate::mcp::call(server, tool, &call.args) {
                    Ok(text) => ToolOutcome::ok(text, Provenance::UntrustedIngested),
                    Err(e) => ToolOutcome::error(e),
                }
            }
            ToolBinding::AgentMemory => {
                // The session's agent owns the memory namespace
                // (`/agent/<id>/memory/`).
                let agent_id = session.agent.manifest_id.0;
                let out = crate::agent::home::run_memory_tool(&call.tool, agent_id, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Recalled facts are durable agent state the human installed
                    // into the agent's home — treat as system-trusted content
                    // (same as a skill body), not untrusted web ingest.
                    ToolOutcome::ok(out, Provenance::SystemTrusted)
                }
            }
        }
    }
}

/// Default `skill` / `load_skill` implementation when no orchestrator hook is set.
fn default_skill_invoke(session: &mut Session, call: &ToolCall) -> ToolOutcome {
    use crate::session::todo::json_str;
    let name = json_str(&call.args, "name").unwrap_or_default();
    if name.is_empty() {
        return ToolOutcome::error("skill: missing required arg 'name' (see /skills)");
    }
    let asset = json_str(&call.args, "asset");
    match crate::skills::loader::invoke(
        session,
        &name,
        asset.as_deref().filter(|s| !s.is_empty()),
        crate::agent::orchestrator::now(),
    ) {
        Ok(text) => {
            let sid = crate::skills::index::by_name(&name).map(|m| m.id).unwrap_or(crate::agent::types::SkillId(0));
            ToolOutcome::ok(text, Provenance::SkillInstalled(sid))
        }
        Err(e) => ToolOutcome::error(e),
    }
}

impl Router {
    /// Readable paths for `caller` (Gate 2.5), sorted.
    fn readable_paths(caller: crate::sched::TaskId) -> Vec<String> {
        store::list()
            .into_iter()
            .filter(|p| cap::scope_check(caller, CapDomain::Fs, Rights::READ, &Scope::Path(p.clone())))
            .collect()
    }

    fn call_store_query(
        &self,
        _session: &Session,
        caller: crate::sched::TaskId,
        call: &ToolCall,
        kind: StoreQueryKind,
    ) -> ToolOutcome {
        let paths = Self::readable_paths(caller);
        match kind {
            StoreQueryKind::Glob => {
                let pattern = todo::json_str(&call.args, "pattern").unwrap_or_default();
                if pattern.is_empty() {
                    return ToolOutcome::error("malformed glob: missing required arg 'pattern'");
                }
                let hits = pathutil::glob_filter(&pattern, &paths);
                ToolOutcome::ok(alloc::format!("ok:[{}]", hits.join(",")), Provenance::UntrustedIngested)
            }
            StoreQueryKind::Grep => {
                let query = todo::json_str(&call.args, "query").unwrap_or_default();
                if query.is_empty() {
                    return ToolOutcome::error("malformed grep: missing required arg 'query'");
                }
                let path_glob = todo::json_str(&call.args, "path_glob").unwrap_or_default();
                let paths = if path_glob.is_empty() {
                    paths
                } else {
                    pathutil::glob_filter(&path_glob, &paths)
                };
                let mut files: Vec<(String, String)> = Vec::new();
                for p in paths {
                    if let Some(bytes) = store::read(&p) {
                        files.push((p, String::from_utf8_lossy(&bytes).into_owned()));
                    }
                }
                let hits = pathutil::grep_files(&query, &files, 50);
                if hits.is_empty() {
                    return ToolOutcome::ok("ok:[]", Provenance::UntrustedIngested);
                }
                let mut out = String::from("ok:\n");
                for h in hits {
                    out.push_str(&alloc::format!("{}:{}:{}\n", h.path, h.line, h.text));
                }
                ToolOutcome::ok(out, Provenance::UntrustedIngested)
            }
        }
    }

    fn call_read_ranged(&self, session: &Session, caller: crate::sched::TaskId, call: &ToolCall) -> ToolOutcome {
        let path = todo::json_str(&call.args, "path").unwrap_or_default();
        if path.is_empty() {
            return ToolOutcome::error("malformed read: missing required arg 'path'");
        }
        let raw = synapse_call("mem_fs_read", &[("path", path.as_str())]);
        let outcome = self.run_synapse(session, caller, &raw);
        if outcome.is_error {
            return outcome;
        }
        // Strip the `ok:` prefix the executor adds.
        let body = outcome.result.strip_prefix("ok:").unwrap_or(&outcome.result);
        let start = todo::json_u32(&call.args, "start_line")
            .or_else(|| todo::json_str(&call.args, "start_line").and_then(|s| s.parse().ok()))
            .unwrap_or(0);
        let end = todo::json_u32(&call.args, "end_line")
            .or_else(|| todo::json_str(&call.args, "end_line").and_then(|s| s.parse().ok()))
            .unwrap_or(0);
        if start == 0 && end == 0 {
            return ToolOutcome::ok(outcome.result, outcome.provenance);
        }
        const MAX_BYTES: usize = 32 * 1024;
        let sliced = pathutil::line_range(body, start, end, MAX_BYTES);
        ToolOutcome::ok(alloc::format!("ok:{sliced}"), Provenance::UntrustedIngested)
    }

    fn call_edit_replace_all(&self, session: &Session, caller: crate::sched::TaskId, call: &ToolCall) -> ToolOutcome {
        let path = todo::json_str(&call.args, "path").unwrap_or_default();
        let old = todo::json_str(&call.args, "old").unwrap_or_default();
        let new = todo::json_str(&call.args, "new").unwrap_or_default();
        if path.is_empty() || old.is_empty() {
            return ToolOutcome::error("malformed edit: need path + old (+ new)");
        }
        // Cap-gated read.
        let read = self.run_synapse(session, caller, &synapse_call("mem_fs_read", &[("path", path.as_str())]));
        if read.is_error {
            return read;
        }
        let body = read.result.strip_prefix("ok:").unwrap_or(&read.result);
        let edited = match pathutil::safe_edit(body, &old, &new, true) {
            Ok(s) => s,
            Err(e) => return ToolOutcome::error(alloc::format!("error:{e}:{path}")),
        };
        // Cap-gated write of the whole file.
        self.run_synapse(
            session,
            caller,
            &synapse_call("mem_fs_write", &[("path", path.as_str()), ("text", edited.as_str())]),
        )
    }
}
