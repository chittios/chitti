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
use crate::security::taint::Justification;
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
            ToolBinding::AgentStorage => {
                let agent_id = session.agent.manifest_id.0;
                let out = crate::agent::storage::run_tool(agent_id, &call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    ToolOutcome::ok(out, Provenance::SystemTrusted)
                }
            }
            ToolBinding::Media => {
                let out = crate::shell::run_media_tool(&call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Paths + playback status are host-side facts (system-trusted).
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
                    Ok(out) => ToolOutcome::ok(out, Provenance::SystemTrusted),
                    Err(e) => ToolOutcome::error(alloc::format!("wasm:{e}")),
                }
            }
            ToolBinding::Download => {
                let out = crate::shell::run_download_tool(&call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Download body is external content; path metadata is system.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::Browser => {
                let out = crate::shell::run_browser_tool(&call.tool, &call.args);
                if out.starts_with("error:") {
                    ToolOutcome::error(out)
                } else {
                    // Page content is external / untrusted-ingested.
                    ToolOutcome::ok(out, Provenance::UntrustedIngested)
                }
            }
            ToolBinding::RunShellCommand => {
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
                    ToolOutcome::ok(out, Provenance::SystemTrusted)
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
                let out = dispatch_web_tool(&call.tool, &call.args);
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
