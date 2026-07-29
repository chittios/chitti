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
use crate::agent::orchestrator::{synapse_call, synapse_call_for, to_taint};
use crate::agent::types::{CapDomain, Provenance, Rights, Scope, Session, ToolCall};
use crate::cap;
use crate::security::taint::{Effect, Justification};
use crate::session::todo;
use crate::synapse::{self, executor::Invocation, fs as store};
use crate::tools::pathutil;
use crate::tools::registry::{self, McpResourceKind, StoreQueryKind, ToolBinding};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
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
        // Default ON: agent tool paths must never silently drop taint. Callers
        // that intentionally want a fully-trusted kernel path set
        // `taint_aware = false` explicitly (rare).
        Self {
            taint_aware: true,
            human_confirmed: false,
            spawn_hook: None,
            load_skill_hook: None,
            run_hook: None,
        }
    }

    /// A taint-aware router (Phase E): destructive calls justified by untrusted
    /// ingested content are refused at the Synapse gate. Alias of [`Self::new`]
    /// kept for call-site clarity.
    pub fn taint_aware() -> Self {
        Self::new()
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

/// What this call would do, classified by binding.
///
/// **This is the chokepoint.** It is an exhaustive `match` over `ToolBinding`
/// with no wildcard arm, so adding a binding does not compile until someone
/// says what it does — which is the difference between a policy that is
/// enforced and one that is remembered. The seven scattered
/// `blocks_destructive()` checks this replaces were each correct and each
/// invisible from the others; the audit that found four unguarded arms
/// (`security::redteam`) is what that arrangement costs.
///
/// Note the classification depends on the *call*, not only the binding: `/http`
/// is egress but `/ls` is not, `memory_add` mutates but `memory_get` does not,
/// and `run_shell_command` has to look inside its own argument to find the
/// command it would run.
pub fn effect_of(def: &registry::ToolDef, call: &ToolCall) -> Effect {
    match &def.binding {
        // Lowered to a primitive: the executor's gate 3 is the real decision,
        // and it sees the primitive's own effect flags. Classifying here as
        // inert would be wrong (the router would let a tainted delete reach the
        // executor), so mirror the registry's view of the primitive.
        ToolBinding::Synapse { primitive, .. } => crate::synapse::registry::by_name(primitive)
            .map(|p| p.effect())
            .unwrap_or(Effect::INERT),

        // Shell commands: destructive by table, plus `/http` in any form —
        // a GET is an exfiltration channel even though it destroys nothing.
        ToolBinding::Shell { command, destructive } => {
            let egress = command == "http" || shell_cmd_is_destructive_http(command, &call.args);
            Effect { irreversible: *destructive, egress }
        }
        ToolBinding::RunShellCommand => {
            let cmdline = todo::json_str(&call.args, "command")
                .or_else(|| todo::json_str(&call.args, "cmd"))
                .unwrap_or_default();
            match crate::tools::shell_cmd::parse_command_line(&cmdline) {
                Ok((name, rest)) => {
                    let extra = todo::json_str(&call.args, "args").unwrap_or_default();
                    let full = if extra.is_empty() {
                        rest
                    } else if rest.is_empty() {
                        extra
                    } else {
                        format!("{rest} {extra}")
                    };
                    Effect {
                        irreversible: crate::tools::shell_cmd::is_destructive_cmd(&name),
                        egress: name == "http" || shell_http_args_destructive(&full),
                    }
                }
                // Unparseable: treat as effectful. A command we cannot read is
                // not a command we may assume is harmless.
                Err(_) => Effect::BOTH,
            }
        }

        // Remote calls and network fetches: egress by construction.
        ToolBinding::Mcp { .. } => Effect::EGRESS,
        ToolBinding::Web => Effect::EGRESS,
        ToolBinding::Download => Effect::BOTH, // fetches *and* writes the store
        ToolBinding::Browser => Effect {
            irreversible: false,
            egress: matches!(call.tool.as_str(), "browser_open" | "browser_navigate" | "browser_goto"),
        },

        // Durable state that re-enters the agent later. Mutations only.
        ToolBinding::AgentMemory => Effect {
            irreversible: matches!(call.tool.as_str(), "memory_add" | "remember" | "memory_md_append"),
            egress: false,
        },
        ToolBinding::AgentStorage => Effect {
            irreversible: matches!(call.tool.as_str(), "storage_set" | "storage_remove"),
            egress: false,
        },

        // Inert with respect to the provenance policy. Each of these either
        // touches nothing outside the turn, or reaches an effect only by
        // lowering to a primitive that is gated in its own right.
        ToolBinding::StoreQuery { .. } => Effect::INERT, // reads, result-filtered by scope
        ToolBinding::SessionTodo => Effect::INERT,       // session-local list
        ToolBinding::SpawnSubagent => Effect::INERT,     // delegation only attenuates
        ToolBinding::LoadSkill => Effect::INERT,         // loads text into context
        ToolBinding::RunIntent => Effect::INERT,         // its steps are gated individually
        ToolBinding::McpResources { .. } => Effect::INERT, // resource *reads*
        ToolBinding::AgentWasm => Effect::INERT,         // sandboxed; effects go via tools
        ToolBinding::Media => Effect::INERT,             // playback state
        ToolBinding::BgTask => Effect::INERT,            // inspects tasks; spawning is elsewhere
        ToolBinding::AskUser => Effect::INERT,           // asks a human
        ToolBinding::SearchReplace => Effect::INERT,     // lowers to gated mem_fs_read/write
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

        // THE gate. One call, before the dispatch match, classified by
        // `effect_of` — which the compiler forces to stay exhaustive. Everything
        // below this line has already been through the provenance policy.
        //
        // The executor gates again for anything that lowers to a primitive. That
        // redundancy is deliberate and cheap: this check is a property of the
        // router, and the router is not in the TCB.
        let effect = effect_of(&def, call);
        if effect.is_effectful() && self.justification(session).blocks_destructive() {
            let what = match (effect.irreversible, effect.egress) {
                (true, true) => "irreversible and leaves the machine",
                (true, false) => "irreversible",
                _ => "leaves the machine",
            };
            return ToolOutcome::error(alloc::format!(
                "refused: '{}' is {what}, justified by untrusted content",
                call.tool
            ));
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
                // UINT surface ids etc. must be bare digits in the wire JSON.
                self.run_synapse(session, caller, &synapse_call_for(primitive, &borrowed))
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
                    // Must not silently drop plan mode: shell `execute_chat_tool`
                    // shows a human confirm modal. This Router path is only a
                    // fallback — refuse unless the shell already confirmed.
                    if !self.human_confirmed {
                        return ToolOutcome::error(
                            "exit_plan_mode: needs human approval (use the chat plan-exit confirm)",
                        );
                    }
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
                // not human-confirmed. Any `/http` under taint is also refused
                // (GET is an egress/exfil channel, same as net_http_get).
                let arg = todo::json_str(&call.args, "args").unwrap_or_default();
                let out = crate::shell::run_tool_command(command, &arg);
                ToolOutcome::ok(out, Provenance::UntrustedIngested)
            }
            ToolBinding::Mcp { server, tool } => {
                // MCP tools/call is remote side-effectful egress. Gate it like a
                // destructive primitive: untrusted justification cannot fire it
                // without human confirmation (results stay UntrustedIngested).
                match crate::mcp::call(server, tool, &call.args) {
                    Ok(text) => ToolOutcome::ok(text, Provenance::UntrustedIngested),
                    Err(e) => ToolOutcome::error(e),
                }
            }
            ToolBinding::AgentMemory => {
                // The session's agent owns the memory namespace
                // (`/agent/<id>/memory/`). Mutating tools re-enter the system
                // prompt (MEMORY.md / stored facts), so under taint they are
                // refused — same defence as SOUL.md writes at the Synapse gate.
                let agent_id = session.agent.manifest_id.0;
                let mutating = matches!(
                    call.tool.as_str(),
                    "memory_add" | "remember" | "memory_md_append"
                );
                if mutating && self.justification(session).blocks_destructive() {
                    return ToolOutcome::error(alloc::format!(
                        "refused: '{}' justified by untrusted content (would launder into system prompt)",
                        call.tool
                    ));
                }
                let out = crate::agent::home::run_memory_tool(&call.tool, agent_id, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Mutating under taint is refused above, so recalls cannot
                    // launder *new* untrusted content. Tag as system-trusted
                    // (durable agent state), matching skill-body steering.
                    ToolOutcome::ok(out, Provenance::SystemTrusted)
                }
            }
            ToolBinding::AgentStorage => {
                // `storage_*` is DURABLE (`/agent/<id>/storage/<key>`), so it
                // crosses turns and sessions — which makes it both an effect site
                // and, read back, a laundering channel. Both halves were missing.
                //
                // Writing is a durable mutation and gates like agent memory: an
                // injection must not be able to leave a "preference" behind.
                let agent_id = session.agent.manifest_id.0;
                let out = crate::agent::storage::run_tool(agent_id, &call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Reading returns whatever was stored, by whoever stored it.
                    // Tagging that trusted completed the cycle: store under an
                    // injection, read back clean, and the turn has no reason left
                    // to refuse anything.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::Media => {
                let out = crate::shell::run_media_tool(&call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Kernel-authored status ("ok:image_open <path>"), which may
                    // echo an argument but carries no bytes the context did not
                    // already hold. An echo cannot *lower* a turn's taint --
                    // `join` is monotone within a turn -- so this one stays
                    // trusted. The channels that had to change are the ones that
                    // outlive the turn: durable storage, background tasks, wasm.
                    ToolOutcome::ok(out, Provenance::SystemTrusted)
                }
            }
            ToolBinding::AgentWasm => {
                // Prefer the package that declares this export (autostart notes/
                // paint/… tools work while chat is still the shell agent).
                let agent_id = crate::agent::system::owner_agent_for_tool(&call.tool)
                    .unwrap_or(session.agent.manifest_id.0);
                match crate::service::package_ui::call_agent_export(
                    agent_id,
                    &call.tool,
                    &call.args,
                ) {
                    Ok(out) if out.starts_with("error:") => ToolOutcome::error(out),
                    // A package's wasm tool digests whatever it was handed -- a
                    // PDF, a note, an HTTP request -- so its output is a function
                    // of ingested bytes and re-enters as ingested bytes. The
                    // module is sandboxed under fuel and memory limits; that
                    // bounds what it can *do*, not how far its output can be
                    // trusted.
                    Ok(out) => ToolOutcome::ok(out, Provenance::UntrustedIngested),
                    Err(e) => ToolOutcome::error(alloc::format!("wasm:{e}")),
                }
            }
            ToolBinding::Download => {
                // Network egress + store write: refuse under untrusted
                // justification (exfil / overwrite via injected "download …").
                let out = crate::shell::run_download_tool(&call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Download body is external content; path metadata is system.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::Browser => {
                // Navigation / open under untrusted context is ambient network
                // egress (and runs page scripts) — refuse like net_http_get.
                let out = crate::shell::run_browser_tool(&call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Page content is external / untrusted-ingested.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::RunShellCommand => {
                // Same taint gate as Shell bindings: destructive first tokens
                // (rm/install/…) and HTTP POST/body cannot run under untrusted
                // justification without human confirmation.
                let command = todo::json_str(&call.args, "command")
                    .or_else(|| todo::json_str(&call.args, "cmd"))
                    .unwrap_or_default();
                if let Ok((name, rest)) = crate::tools::shell_cmd::parse_command_line(&command) {
                    let extra = todo::json_str(&call.args, "args").unwrap_or_default();
                    let full = if extra.is_empty() {
                        rest
                    } else if rest.is_empty() {
                        extra
                    } else {
                        format!("{rest} {extra}")
                    };
                    let dest = crate::tools::shell_cmd::is_destructive_cmd(&name)
                        || name == "http"
                        || shell_http_args_destructive(&full);
                }
                let out = crate::tools::shell_cmd::run_from_tool_args(&call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::BgTask => {
                let out = dispatch_bg_tool(&call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Task output is the output of a *command*, and a task
                    // outlives the turn that started it -- so the untrusted
                    // message that prompted it may be long gone by the time the
                    // output is collected. Ingested, not trusted.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::AskUser => {
                let out = dispatch_ask_user(&call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    ToolOutcome::ok(out, Provenance::UserTyped)
                }
            }
            ToolBinding::Web => {
                let out = dispatch_web_tool_gated(session, self, &call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::SearchReplace => self.call_search_replace(session, caller, call),
        }
    }
}

fn dispatch_bg_tool(name: &str, args: &str) -> String {
    use crate::session::todo::{json_str, json_u32};
    match name {
        "task_output" => {
            let id = json_u32(args, "task_id")
                .or_else(|| json_str(args, "task_id").and_then(|s| s.parse().ok()))
                .unwrap_or(0) as u64;
            let wait = json_u32(args, "timeout_ms")
                .or_else(|| json_str(args, "timeout_ms").and_then(|s| s.parse().ok()))
                .unwrap_or(0) as u64;
            crate::tools::bg::task_output(id, wait)
        }
        "kill_task" => {
            let id = json_u32(args, "task_id")
                .or_else(|| json_str(args, "task_id").and_then(|s| s.parse().ok()))
                .unwrap_or(0) as u64;
            crate::tools::bg::kill_task(id)
        }
        "list_tasks" => crate::tools::bg::list_tasks(),
        "monitor" => {
            let command = json_str(args, "command").unwrap_or_default();
            if command.is_empty() {
                return String::from("error: need command");
            }
            let interval = json_u32(args, "interval_ms")
                .or_else(|| json_str(args, "interval_ms").and_then(|s| s.parse().ok()))
                .unwrap_or(5000) as u64;
            let (cmd, rest) = match crate::tools::shell_cmd::parse_command_line(&command) {
                Ok(x) => x,
                Err(e) => return format!("error:{e}"),
            };
            let id = crate::tools::bg::spawn_monitor(&cmd, &rest, interval);
            format!("ok:monitor task_id={id} every {interval}ms /{cmd} {rest}")
        }
        _ => format!("error: unknown bg tool {name}"),
    }
}

/// Parse a JSON array of strings from a tool arg: `["a","b"]` or newline list.
fn parse_option_list(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let s = raw.trim();
    if s.starts_with('[') {
        // Extract quoted strings.
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                i += 1;
                let mut t = String::new();
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        t.push(bytes[i + 1] as char);
                        i += 2;
                    } else {
                        t.push(bytes[i] as char);
                        i += 1;
                    }
                }
                if !t.is_empty() {
                    out.push(t);
                }
            }
            i += 1;
        }
    } else {
        for part in s.split(|c| c == '\n' || c == '|' || c == ',') {
            let p = part.trim();
            if !p.is_empty() {
                out.push(p.to_string());
            }
        }
    }
    out
}

fn dispatch_ask_user(args: &str) -> String {
    use crate::session::todo::json_str;
    let question = json_str(args, "question")
        .or_else(|| json_str(args, "prompt"))
        .unwrap_or_default();
    if question.is_empty() {
        return String::from("error: need question");
    }
    let opt_raw = json_str(args, "options").unwrap_or_default();
    let options = parse_option_list(&opt_raw);
    if options.len() < 2 {
        return String::from("error: need at least 2 options (JSON array or comma-separated)");
    }
    if options.len() > 8 {
        return String::from("error: at most 8 options");
    }
    let refs: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
    match crate::modal::choose("Agent question", &question, &refs) {
        Some(i) => format!("ok:selected={} index={} label={}", i + 1, i, options[i]),
        None => String::from("error: user cancelled the question"),
    }
}

fn strip_html_rough(html: &str, max: usize) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            continue;
        }
        if c == '>' {
            in_tag = false;
            out.push(' ');
            continue;
        }
        if !in_tag {
            out.push(c);
            if out.len() >= max {
                out.push_str("…");
                break;
            }
        }
    }
    // Collapse whitespace.
    let mut compact = String::new();
    let mut sp = false;
    for c in out.chars() {
        if c.is_whitespace() {
            if !sp {
                compact.push(' ');
                sp = true;
            }
        } else {
            sp = false;
            compact.push(c);
        }
    }
    compact
}

/// Whether a shell `/http` invocation is side-effectful (POST/body/download).
fn shell_http_args_destructive(args: &str) -> bool {
    let lower = args.to_ascii_lowercase();
    // -X POST|PUT|PATCH|DELETE, -d body, -O/-o download (writes store + egress).
    if lower.contains("-x post")
        || lower.contains("-x put")
        || lower.contains("-x patch")
        || lower.contains("-x delete")
    {
        return true;
    }
    let mut tokens = lower.split_whitespace();
    while let Some(t) = tokens.next() {
        match t {
            "-d" | "--data" | "--data-raw" | "-O" | "-o" | "--stream" => return true,
            t if t.starts_with("-d") && t.len() > 2 => return true, // -dBODY
            _ => {}
        }
    }
    false
}

/// Shell binding helper: `http` with POST/body counts as destructive for taint.
fn shell_cmd_is_destructive_http(command: &str, call_args: &str) -> bool {
    if command != "http" {
        return false;
    }
    let arg = todo::json_str(call_args, "args").unwrap_or_default();
    shell_http_args_destructive(&arg)
}

/// Web tools always hit the network; refuse when the Router's session
/// justification is tainted (callers pass session via `call` wrapper).
/// Kept as a thin alias: the web tools' taint check now happens once, in
/// `Router::call`, classified by `effect_of` as `Effect::EGRESS`.
fn dispatch_web_tool_gated(_session: &Session, _router: &Router, name: &str, args: &str) -> String {
    dispatch_web_tool(name, args)
}

fn dispatch_web_tool(name: &str, args: &str) -> String {
    use crate::session::todo::{json_str, json_u32};
    match name {
        "web_search" => {
            let q = json_str(args, "query").unwrap_or_default();
            if q.is_empty() {
                return String::from("error: need query");
            }
            // DuckDuckGo HTML (no API key). Results are untrusted.
            let mut enc = String::new();
            for b in q.bytes() {
                match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        enc.push(b as char)
                    }
                    b' ' => enc.push_str("%20"),
                    _ => enc.push_str(&format!("%{b:02X}")),
                }
            }
            let url = format!("https://html.duckduckgo.com/html/?q={enc}");
            match crate::net::http::get(&url, 30_000) {
                Ok(resp) => {
                    let text = strip_html_rough(&resp.text(), 4000);
                    format!("ok:web_search q={q}\n{text}")
                }
                Err(e) => format!("error:web_search: {e}"),
            }
        }
        "web_fetch" => {
            let url = json_str(args, "url").unwrap_or_default();
            if url.is_empty() {
                return String::from("error: need url");
            }
            let max = json_u32(args, "max_bytes")
                .or_else(|| json_str(args, "max_bytes").and_then(|s| s.parse().ok()))
                .unwrap_or(8000) as usize;
            match crate::net::http::get(&url, 30_000) {
                Ok(resp) => {
                    let body = resp.text();
                    let text = if body.contains('<') {
                        strip_html_rough(&body, max)
                    } else {
                        let n = body.len().min(max);
                        let mut t = body[..n].to_string();
                        if body.len() > max {
                            t.push_str("…");
                        }
                        t
                    };
                    format!("ok:web_fetch status={} url={url}\n{text}", resp.status)
                }
                Err(e) => format!("error:web_fetch: {e}"),
            }
        }
        _ => format!("error: unknown web tool {name}"),
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
            let sid = crate::skills::index::by_name(&name)
                .map(|m| m.id)
                .unwrap_or(crate::agent::types::SkillId(0));
            // bordered `<skill name path>` envelope for model grounding.
            let path = alloc::format!("/skills/{}/SKILL.md", name);
            let wrapped = crate::agent::prompt::skill_result_envelope(&name, &path, &text);
            ToolOutcome::ok(wrapped, Provenance::SkillInstalled(sid))
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
                let query = todo::json_str(&call.args, "query")
                    .or_else(|| todo::json_str(&call.args, "pattern"))
                    .unwrap_or_default();
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
                let head = todo::json_u32(&call.args, "head_limit")
                    .or_else(|| todo::json_str(&call.args, "head_limit").and_then(|s| s.parse().ok()))
                    .unwrap_or(50)
                    .clamp(1, 200) as usize;
                let ci = todo::json_str(&call.args, "case_insensitive")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or_else(|| {
                        call.args.contains("\"case_insensitive\":true")
                            || call.args.contains("\"case_insensitive\": true")
                    });
                let hits = pathutil::grep_files_ex(&query, &files, head, ci);
                if hits.is_empty() {
                    return ToolOutcome::ok("ok:[]", Provenance::UntrustedIngested);
                }
                let mut out = String::from("ok:\n");
                for h in hits {
                    out.push_str(&alloc::format!("{}:{}:{}\n", h.path, h.line, h.text));
                }
                ToolOutcome::ok(out, Provenance::UntrustedIngested)
            }
            StoreQueryKind::ListDir => {
                let path = todo::json_str(&call.args, "path").unwrap_or_else(|| "/".into());
                let kids = pathutil::list_dir_children(&path, &paths);
                if kids.is_empty() {
                    return ToolOutcome::ok(
                        alloc::format!("ok:list_dir {path}\n(empty)"),
                        Provenance::UntrustedIngested,
                    );
                }
                let mut out = alloc::format!("ok:list_dir {path}\n");
                for k in kids {
                    out.push_str(&k);
                    out.push('\n');
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

    /// `search_replace` — same engine as `edit`, accepts old_string/new_string.
    fn call_search_replace(
        &self,
        session: &Session,
        caller: crate::sched::TaskId,
        call: &ToolCall,
    ) -> ToolOutcome {
        let path = todo::json_str(&call.args, "path").unwrap_or_default();
        let old = todo::json_str(&call.args, "old_string")
            .or_else(|| todo::json_str(&call.args, "old"))
            .unwrap_or_default();
        let new = todo::json_str(&call.args, "new_string")
            .or_else(|| todo::json_str(&call.args, "new"))
            .unwrap_or_default();
        if path.is_empty() || old.is_empty() {
            return ToolOutcome::error("malformed search_replace: need path + old_string (+ new_string)");
        }
        let replace_all = todo::json_str(&call.args, "replace_all")
            .map(|v| v == "true" || v == "1")
            .unwrap_or_else(|| {
                call.args.contains("\"replace_all\":true") || call.args.contains("\"replace_all\": true")
            });
        let read = self.run_synapse(session, caller, &synapse_call("mem_fs_read", &[("path", path.as_str())]));
        if read.is_error {
            return read;
        }
        let body = read.result.strip_prefix("ok:").unwrap_or(&read.result);
        let edited = match pathutil::safe_edit(body, &old, &new, replace_all) {
            Ok(s) => s,
            Err(e) => return ToolOutcome::error(alloc::format!("error:{e}:{path}")),
        };
        self.run_synapse(
            session,
            caller,
            &synapse_call("mem_fs_write", &[("path", path.as_str()), ("text", edited.as_str())]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn def_and_call(tool: &str, args: &str) -> (registry::ToolDef, ToolCall) {
        let def = registry::get(tool).unwrap_or_else(|| panic!("no such tool: {tool}"));
        (def, ToolCall { call_id: 1, tool: tool.to_string(), args: args.to_string() })
    }

    /// Every tool the attack corpus targets must classify as effectful, and the
    /// read-only ones must not.
    ///
    /// `effect_of` is exhaustive over `ToolBinding`, so the compiler guarantees a
    /// *new binding* gets classified. It cannot guarantee the classification is
    /// right, and the seven checks it replaced each encoded a condition (`/http`
    /// is egress though it destroys nothing; only `memory_add` mutates, not
    /// `memory_get`; `run_shell_command` has to read its own argument). This pins
    /// those conditions so a refactor cannot quietly relax one.
    #[test_case]
    fn effect_of_classifies_the_conditions_the_scattered_checks_encoded() {
        let cases: &[(&str, &str, bool, bool)] = &[
            ("rm", r#"{"args":"/tmp/x"}"#, true, false),
            ("ls", r#"{"args":"/"}"#, false, false),
            ("http", r#"{"args":"http://127.0.0.1:9/"}"#, false, true),
            ("delete", r#"{"path":"/x"}"#, true, false),
            ("read", r#"{"path":"/x"}"#, false, false),
            ("download", r#"{"url":"http://127.0.0.1:9/","path":"/x"}"#, true, true),
            ("web_fetch", r#"{"url":"http://127.0.0.1:9/"}"#, false, true),
            ("browser_open", r#"{"url":"http://127.0.0.1:9/"}"#, false, true),
            ("browser_status", r#"{}"#, false, false),
            ("memory_add", r#"{"key":"k","value":"v"}"#, true, false),
            ("memory_get", r#"{"key":"k"}"#, false, false),
            ("storage_set", r#"{"key":"k","value":"v"}"#, true, false),
            ("storage_get", r#"{"key":"k"}"#, false, false),
            ("run_shell_command", r#"{"command":"rm /tmp/x"}"#, true, false),
            ("run_shell_command", r#"{"command":"ls /"}"#, false, false),
        ];
        for (tool, args, irr, egr) in cases {
            let (def, call) = def_and_call(tool, args);
            let e = effect_of(&def, &call);
            assert_eq!((e.irreversible, e.egress), (*irr, *egr), "{tool} {args} classified {:?}", e);
        }
    }
}
