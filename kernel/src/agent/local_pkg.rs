//! **Local agent packages** — an agent authored on this machine, installed from
//! the store.
//!
//! # Why
//!
//! Every `SkillPackage` in the tree is built from `include_str!`/`include_bytes!`,
//! and `/agents install` resolves a name against three hardcoded samples. So an
//! agent scaffolded by `/agents new`, edited in the editor and compiled by
//! `/agents build` had nowhere to go. This is the missing half: read a package's
//! files back out of the store at runtime, turn them into a `SkillPackage`, and put
//! it through the same consent-and-grant path a registry package takes.
//! `InstallSource::Store { key }` has existed unconstructed since the types were
//! written; this finally constructs it.
//!
//! # The invariant that shapes everything here
//!
//! A local package's files live in the **store**, which any agent with a broad `Fs`
//! scope can write. So an install must be a *human* decision, and a re-install must
//! not become a way to acquire authority nobody approved:
//!
//! > **Re-install and boot-reinstall grant `manifest ∩ recorded grant`. They never
//! > re-read the manifest's requests as an authority. Widening needs consent again.**
//!
//! Without that, editing `manifest.json` in the store and triggering a reload is
//! privilege escalation with extra steps. CLAUDE.md invariant 5 — a skill is bounded
//! by its install-time grant, forever — has to hold for a package whose manifest is
//! writable, and `InstallRecord.granted_capabilities` is what makes it hold.
//!
//! # Ids have to survive a reboot
//!
//! An agent's home is `/agent/<id>/`: its SOUL, its assets, its memory. If the id
//! moved on the next boot, all of that would be orphaned. `next_agent_id()` is an
//! in-memory counter from 1 and the system roster bypasses it with fixed `9000+`
//! ids, so local packages get their own **persisted** counter well clear of both.

use crate::agent::types::*;
use crate::skills::package::SkillPackage;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where local packages are authored: `~/agents/<name>/`.
pub fn dir_for(name: &str) -> String {
    format!("{}/agents/{}", crate::agent::home::USER_HOME, name)
}

/// Store key holding the next free local id.
const NEXT_ID_KEY: &str = "agents/local/next_id";
/// Store key holding the name → ids index, one `name\tagent\tskill\t` line each.
const INDEX_KEY: &str = "agents/local/index";

/// First id handed to a local package.
///
/// Clear of the 47 hand-assigned system ids at `9000 + offset` and of the in-memory
/// minter that starts at 1, so a local agent's home can never land on another
/// agent's. 20000 leaves room for the system roster to grow by thousands first.
pub const LOCAL_ID_BASE: u64 = 20_000;

/// One installed local package.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LocalEntry {
    pub name: String,
    pub agent: AgentId,
    pub skill: SkillId,
}

// --- diagnostics ------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Error,
    Warning,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic {
    pub level: Level,
    pub field: String,
    pub message: String,
}

impl Diagnostic {
    fn error(field: &str, message: impl Into<String>) -> Self {
        Diagnostic { level: Level::Error, field: field.to_string(), message: message.into() }
    }
    fn warn(field: &str, message: impl Into<String>) -> Self {
        Diagnostic { level: Level::Warning, field: field.to_string(), message: message.into() }
    }
}

/// Keys a manifest may carry. An unknown key is a warning, not an error: it is
/// usually a typo (`capabilites`), and the parser's silence about it is how a
/// capability request goes missing without anyone noticing.
const KNOWN_KEYS: &[&str] = &[
    "name",
    "version",
    "kind",
    "description",
    "toolset",
    "capabilities",
    "default_port",
    "autostart",
    "mcp_servers",
    "command_hooks",
    "wasm",
    "package_ui",
];
const KNOWN_KINDS: &[&str] = &["service", "skill_agent", "subagent"];
const KNOWN_DOMAINS: &[&str] =
    &["fs", "console", "spawn", "todo", "inference", "ipc", "skill_manage", "channel", "net", "ui"];
const KNOWN_RIGHTS: &[&str] = &["read", "write", "exec", "delete", "list"];
/// Ceiling on `wasm.memory_pages` (4096 × 64 KiB = 256 MiB). The runtime enforces
/// whatever a manifest asks for as a *limit*, so an absurd value is not itself
/// dangerous — but a guest that can ask for more linear memory than the kernel
/// heap holds turns an out-of-memory into a guest trap at an arbitrary moment,
/// and a typo'd extra digit should be a diagnostic rather than that.
const MAX_MEMORY_PAGES: i64 = 4096;

/// Check a manifest, reporting what [`crate::agent::system::parse_manifest`] would
/// silently swallow.
///
/// The parser is deliberately forgiving — it has to be, since it also reads the
/// compiled-in manifests at boot and a hard failure there would cost the machine an
/// agent. That forgiveness is wrong for a *human editing a file*: `kind` defaults to
/// `Service` on any unrecognized value, an unknown capability `domain` is skipped
/// entirely, and unknown `rights` are dropped. Each of those changes what the agent
/// is or may do, with no message anywhere.
pub fn lint(json: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let Some(j) = crate::json::Json::parse(json) else {
        out.push(Diagnostic::error("(file)", "not valid JSON"));
        return out;
    };
    let Some(obj) = j.as_object() else {
        out.push(Diagnostic::error("(file)", "the manifest must be a JSON object"));
        return out;
    };

    for (key, _) in obj {
        if !KNOWN_KEYS.contains(&key.as_str()) {
            out.push(Diagnostic::warn(key, "unknown key -- ignored by the parser (typo?)"));
        }
    }

    // Required, because everything downstream is named by them.
    for req in ["name", "version", "description"] {
        match j.get(req).and_then(|v| v.as_str()) {
            None => out.push(Diagnostic::error(req, "required")),
            Some(s) if s.trim().is_empty() => out.push(Diagnostic::error(req, "must not be empty")),
            Some(_) => {}
        }
    }

    match j.get("kind").and_then(|v| v.as_str()) {
        None => out.push(Diagnostic::error("kind", "required -- one of service, skill_agent, subagent")),
        Some(k) if !KNOWN_KINDS.contains(&k) => out.push(Diagnostic::error(
            "kind",
            format!("'{k}' is not a kind; the parser would silently install this as a service"),
        )),
        Some(_) => {}
    }

    match j.get("toolset").and_then(|v| v.as_array()) {
        None => out.push(Diagnostic::warn("toolset", "absent -- this agent gets no tools")),
        Some(items) => {
            if items.is_empty() {
                out.push(Diagnostic::warn("toolset", "empty -- this agent gets no tools"));
            }
            for it in items {
                match it.as_str() {
                    None => out.push(Diagnostic::error("toolset", "entries must be strings")),
                    Some(name) => {
                        // A name matching no registered tool is invisible at runtime,
                        // with no error — the agent simply cannot call it.
                        if name != "*" && !tool_exists(name) {
                            out.push(Diagnostic::warn(
                                "toolset",
                                format!("'{name}' matches no registered tool -- it will be invisible"),
                            ));
                        }
                    }
                }
            }
        }
    }

    if let Some(caps) = j.get("capabilities").and_then(|v| v.as_array()) {
        for (i, c) in caps.iter().enumerate() {
            let field = format!("capabilities[{i}]");
            match c.get("domain").and_then(|v| v.as_str()) {
                None => out.push(Diagnostic::error(&field, "missing 'domain'")),
                Some(d) if !KNOWN_DOMAINS.contains(&d) => out.push(Diagnostic::error(
                    &field,
                    format!("unknown domain '{d}' -- the whole capability is skipped silently"),
                )),
                Some(_) => {}
            }
            match c.get("rights").and_then(|v| v.as_array()) {
                None => out.push(Diagnostic::warn(&field, "no 'rights' -- grants nothing")),
                Some(rs) => {
                    for r in rs {
                        match r.as_str() {
                            None => out.push(Diagnostic::error(&field, "rights entries must be strings")),
                            Some(r) if !KNOWN_RIGHTS.contains(&r) => {
                                out.push(Diagnostic::warn(&field, format!("unknown right '{r}' -- dropped")))
                            }
                            Some(_) => {}
                        }
                    }
                }
            }
            // Scope syntax: anything unrecognized becomes `Scope::Any`, which is the
            // *widest* possible reading of a typo. Worth an error, not a warning.
            if let Some(scope) = c.get("scope").and_then(|v| v.as_str()) {
                let ok = scope == "home"
                    || scope == "any"
                    || scope.starts_with("path:")
                    || scope.starts_with("net:");
                if !ok {
                    out.push(Diagnostic::error(
                        &field,
                        format!("scope '{scope}' is not recognised, and would widen to ANY"),
                    ));
                }
            }
        }
    }

    if let Some(w) = j.get("wasm") {
        match w.get("module").and_then(|v| v.as_str()) {
            None => out.push(Diagnostic::error("wasm.module", "declared 'wasm' block without a 'module'")),
            Some(m) if !m.starts_with("assets/") => out.push(Diagnostic::warn(
                "wasm.module",
                "conventionally 'assets/tools.wasm' -- install only places files under assets/",
            )),
            Some(_) => {}
        }
        if let Some(f) = w.get("fuel").and_then(|v| v.as_i64()) {
            if f <= 0 {
                out.push(Diagnostic::error("wasm.fuel", "must be positive"));
            }
        }
        if let Some(p) = w.get("memory_pages").and_then(|v| v.as_i64()) {
            if p <= 0 {
                out.push(Diagnostic::error("wasm.memory_pages", "must be positive"));
            } else if p > MAX_MEMORY_PAGES {
                out.push(Diagnostic::error(
                    "wasm.memory_pages",
                    "over the 256 MiB ceiling a guest may ask for",
                ));
            }
        }
    }
    out
}

/// Does the registry hold a tool by this name? `for_agent` is the existing
/// intersection helper, so asking it for one name answers exactly the question the
/// runtime will ask later.
fn tool_exists(name: &str) -> bool {
    !crate::tools::registry::for_agent(&[name.to_string()]).is_empty()
}

/// True if any diagnostic is fatal.
pub fn has_errors(d: &[Diagnostic]) -> bool {
    d.iter().any(|x| x.level == Level::Error)
}

// --- id allocation ----------------------------------------------------------

fn read_u64(key: &str) -> Option<u64> {
    let bytes = crate::synapse::fs::read(key)?;
    core::str::from_utf8(&bytes).ok()?.trim().parse().ok()
}

/// Ids for a *new* local package, from a counter that survives a reboot.
fn mint_ids() -> (AgentId, SkillId) {
    let next = read_u64(NEXT_ID_KEY).unwrap_or(LOCAL_ID_BASE).max(LOCAL_ID_BASE);
    crate::synapse::fs::write(NEXT_ID_KEY, format!("{}", next + 1).as_bytes());
    // Keep the in-memory minter ahead too, so a session-minted id can never collide
    // with one of ours.
    notice_agent_id(AgentId(next));
    notice_skill_id(SkillId(next));
    (AgentId(next), SkillId(next))
}

// --- the name → ids index ---------------------------------------------------

/// Every local package this machine knows about.
pub fn index() -> Vec<LocalEntry> {
    let Some(bytes) = crate::synapse::fs::read(INDEX_KEY) else {
        return Vec::new();
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let mut f = line.split('\t');
            let name = f.next()?.trim();
            let agent = f.next()?.trim().parse().ok()?;
            let skill = f.next()?.trim().parse().ok()?;
            (!name.is_empty()).then(|| LocalEntry {
                name: name.to_string(),
                agent: AgentId(agent),
                skill: SkillId(skill),
            })
        })
        .collect()
}

pub fn lookup(name: &str) -> Option<LocalEntry> {
    index().into_iter().find(|e| e.name == name)
}

fn write_index(entries: &[LocalEntry]) {
    let mut out = String::new();
    for e in entries {
        out.push_str(&format!("{}\t{}\t{}\n", e.name, e.agent.0, e.skill.0));
    }
    crate::synapse::fs::write(INDEX_KEY, out.as_bytes());
}

/// Ids for `name`, reusing them if it was installed before.
///
/// Reuse is the point: the same name keeps the same `/agent/<id>/`, so a re-install
/// after an edit does not orphan the agent's memory.
pub fn ids_for(name: &str) -> (AgentId, SkillId) {
    if let Some(e) = lookup(name) {
        return (e.agent, e.skill);
    }
    let (agent, skill) = mint_ids();
    let mut all = index();
    all.push(LocalEntry { name: name.to_string(), agent, skill });
    write_index(&all);
    (agent, skill)
}

pub fn forget(name: &str) {
    let mut all = index();
    all.retain(|e| e.name != name);
    write_index(&all);
}

// --- building a package from the store --------------------------------------

/// Read `~/agents/<name>/` and build a signable, installable package.
pub fn package_from_store(name: &str) -> Result<(SkillPackage, Vec<Diagnostic>), Vec<Diagnostic>> {
    let dir = dir_for(name);
    let manifest_path = format!("{dir}/manifest.json");
    let Some(mbytes) = crate::synapse::fs::read(&manifest_path) else {
        return Err(alloc::vec![Diagnostic::error(
            "(dir)",
            format!("no {manifest_path} -- /agents new {name} first")
        )]);
    };
    let Ok(mtext) = core::str::from_utf8(&mbytes) else {
        return Err(alloc::vec![Diagnostic::error("manifest.json", "not valid UTF-8")]);
    };
    let mut diags = lint(mtext);
    if has_errors(&diags) {
        return Err(diags);
    }
    let Some(m) = crate::agent::system::parse_manifest(mtext) else {
        return Err(alloc::vec![Diagnostic::error("manifest.json", "could not be parsed")]);
    };
    if m.name != name {
        diags.push(Diagnostic::warn(
            "name",
            format!("manifest says '{}' but the folder is '{name}'; the folder wins", m.name),
        ));
    }

    let (agent_id, skill_id) = ids_for(name);
    let soul = crate::synapse::fs::read(&format!("{dir}/SOUL.md"))
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_default();
    if soul.trim().is_empty() {
        diags.push(Diagnostic::warn("SOUL.md", "empty -- the agent has no judgment of its own"));
    }

    // Assets. **Every payload must also be declared**, because `place_trusted`
    // silently drops an undeclared one — and a `tools.wasm` that never reaches the
    // agent's home fails later as "tools.wasm missing", far from the cause.
    let mut payloads: Vec<(String, Vec<u8>)> = Vec::new();
    if let Some(module) = m.wasm_module.as_deref() {
        let file = module.rsplit('/').next().unwrap_or(module);
        match crate::synapse::fs::read(&format!("{dir}/{module}")) {
            Some(bytes) if !bytes.is_empty() => {
                // Refuse a module that is not wasm at all, here rather than at the
                // first tool call.
                if crate::agent::wasm_rt::validate_module(&bytes).is_err() {
                    return Err(alloc::vec![Diagnostic::error(
                        "wasm.module",
                        format!("{module} is not a valid wasm module -- rebuild it (/agents build {name})")
                    )]);
                }
                // A JS-derived module built against another engine would fail inside
                // QuickJS; name it now.
                if let Some(ns) = crate::agent::js_rt::JsSession::namespace() {
                    if crate::agent::jsmod::links_plugin(&bytes, ns) {
                        let want = crate::agent::js_rt::JsSession::plugin_stamp();
                        match crate::agent::jsmod::plugin_stamp(&bytes) {
                            Some(got) if got != want => {
                                return Err(alloc::vec![Diagnostic::error(
                                    "wasm.module",
                                    format!("built against JS engine '{got}', this machine has '{want}' -- /agents build {name}")
                                )]);
                            }
                            _ => {}
                        }
                    }
                }
                payloads.push((file.to_string(), bytes));
            }
            _ => {
                return Err(alloc::vec![Diagnostic::error(
                    "wasm.module",
                    format!("{dir}/{module} is missing or empty -- /agents build {name}")
                )])
            }
        }
    }

    let asset_meta: Vec<Asset> = payloads
        .iter()
        .map(|(file, bytes)| Asset {
            name: file.clone(),
            store_ref: StoreKey(format!("/agent/{}/assets/{file}", agent_id.0)),
            bytes: bytes.len() as u32,
        })
        .collect();

    // `$HOME` in a scope means the agent's own folder, as for a system agent.
    let home = crate::agent::home::path(agent_id.0);
    let caps: Vec<CapabilityRequest> = m
        .capabilities
        .iter()
        .cloned()
        .map(|mut c| {
            if let Scope::Path(p) = &c.scope {
                if p.contains("$HOME") {
                    c.scope = Scope::Path(p.replace("$HOME", &home));
                }
            }
            c
        })
        .collect();

    let agent = AgentManifest {
        schema_version: 1,
        id: agent_id,
        name: name.to_string(),
        version: m.version.clone(),
        kind: m.kind,
        description: m.description.clone(),
        system_prompt: soul.clone(),
        toolset: m.toolset.clone(),
        capabilities: caps.clone(),
        skills: Vec::new(),
        sampling: Sampling::deterministic(1),
        budgets: Budgets {
            max_turns: 8,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 256,
            max_subagents: 2,
            max_depth: 1,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Installed { skill: skill_id },
        mcp_servers: m.mcp_servers.clone(),
    };
    let manifest = SkillManifest {
        schema_version: 2,
        id: skill_id,
        name: name.to_string(),
        version: m.version.clone(),
        description: m.description.clone(),
        kind: SkillKind::SkillAgent,
        requested_capabilities: caps,
        body_ref: StoreKey(format!("skills/{}/body.md", skill_id.0)),
        bundled_tools: Vec::new(),
        assets: asset_meta,
        agent: Some(agent),
        soul_ref: Some(StoreKey(format!("/agent/{}/SOUL.md", agent_id.0))),
        skill_docs: Vec::new(),
        signature: SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        },
    };
    let mut pkg = SkillPackage {
        manifest,
        body: format!("Local {name} agent, authored on this machine."),
        soul: Some(soul),
        skill_docs: Vec::new(),
        assets: payloads,
    };
    // Signed with the machine's own dev key. The signature is **not** the trust
    // decision here — it makes `install`'s verify gate pass for a package this
    // machine built. The trust decision is the human at the consent modal, which is
    // the only thing standing between a store-written manifest and a grant.
    pkg.sign();
    Ok((pkg, diags))
}

/// Store key remembering which tool names a package registered.
fn tools_key(name: &str) -> String {
    format!("agents/local/{name}/tools")
}

/// Register a local package's tools so its agent can actually call them.
///
/// **A manifest's `toolset` does not create tools.** `registry::for_agent` only
/// *filters* a compiled-in static list, so a name with no matching `ToolDef` is
/// silently invisible — which is what "matches no registered tool" in the lint is
/// warning about, and why a freshly installed local agent could see none of its own
/// tools. Runtime registration is the existing mechanism (`register_replace`, used by
/// bundled tools and by `/mcp connect`); this points it at a package's own module.
///
/// Which names are the package's own is decided by **the module's exports**, not by
/// the manifest alone: a `toolset` also lists borrowed tools like `memory_add`, and
/// registering those would shadow the real ones. Anything the manifest declares and
/// the module exports is this package's; anything else is left alone.
///
/// Previously-registered names are deregistered first, so a tool deleted from the
/// script does not linger in the registry — the registry has no per-agent index, so
/// the set is remembered in the store.
pub fn register_tools(name: &str, module: &[u8], toolset: &[String], description: &str) -> Vec<String> {
    use crate::tools::registry::{self, ToolBinding, ToolDef};

    // Drop whatever this package registered last time.
    if let Some(prev) = crate::synapse::fs::read(&tools_key(name)) {
        if let Ok(text) = core::str::from_utf8(&prev) {
            for old in text.split_whitespace() {
                registry::deregister(old);
            }
        }
    }

    let exports = crate::agent::jsmod::export_names(module);
    let mut registered: Vec<String> = Vec::new();
    for tool in toolset {
        if !exports.iter().any(|e| e == tool) {
            continue;
        }
        registry::register_replace(ToolDef {
            name: tool.clone(),
            // No per-tool description in the manifest yet, so say where it comes
            // from: the agent's own name is the useful part for a model choosing.
            description: format!("{description} (tool of the local '{name}' agent)"),
            // Permissive: the script validates its own arguments, and a wrong schema
            // would refuse calls the tool would have handled.
            input_schema: String::from(r#"{"type":"object"}"#),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        });
        registered.push(tool.clone());
    }
    crate::synapse::fs::write(&tools_key(name), registered.join(" ").as_bytes());
    registered
}

/// The capabilities to grant on a re-install: what the manifest asks for,
/// intersected with what a human already approved.
///
/// This is the invariant in the module docs, as code. A first install has no record
/// and returns `None` — the caller must ask.
pub fn regrant(skill: SkillId, requested: &[CapabilityRequest]) -> Option<Vec<CapabilityRequest>> {
    let rec = crate::skills::install::load_record(skill)?;
    Some(intersect_caps(requested, &rec.granted_capabilities))
}

/// Re-install every local package at boot.
///
/// Needed because install records are written to the store and never read back:
/// without this a local agent's files survive a reboot while its role and tools do
/// not. Grants come from [`regrant`], so a manifest edited between boots cannot gain
/// authority — it can only lose it.
pub fn reinstall_all(now: Ticks) {
    let entries = index();
    if entries.is_empty() {
        return;
    }
    for e in &entries {
        let (pkg, _diags) = match package_from_store(&e.name) {
            Ok(v) => v,
            Err(d) => {
                let first = d.first().map(|x| x.message.clone()).unwrap_or_default();
                crate::ktrace::log_fmt(format_args!(
                    "agents.local: '{}' not reinstalled: {first}",
                    e.name
                ));
                continue;
            }
        };
        let requested = pkg.manifest.requested_capabilities.clone();
        let Some(granted) = regrant(e.skill, &requested) else {
            crate::ktrace::log_fmt(format_args!(
                "agents.local: '{}' has no install record; leaving it uninstalled",
                e.name
            ));
            continue;
        };
        if granted.len() < requested.len() {
            crate::ktrace::log_fmt(format_args!(
                "agents.local: '{}' now requests {} caps but was granted {}; keeping the smaller grant",
                e.name,
                requested.len(),
                granted.len()
            ));
        }
        let source = InstallSource::Store { key: StoreKey(dir_for(&e.name)) };
        let module: Vec<u8> = pkg.assets.first().map(|(_, b)| b.clone()).unwrap_or_default();
        let toolset = pkg.manifest.agent.as_ref().map(|a| a.toolset.clone()).unwrap_or_default();
        let description = pkg.manifest.description.clone();
        match crate::skills::install::install(&pkg, &granted, "boot", source, now) {
            Ok(_) => {
                let tools = register_tools(&e.name, &module, &toolset, &description);
                crate::ktrace::log_fmt(format_args!(
                    "agents.local: reinstalled '{}' as agent {} ({} caps, {} tools)",
                    e.name,
                    e.agent.0,
                    granted.len(),
                    tools.len()
                ));
            }
            Err(err) => crate::ktrace::log_fmt(format_args!(
                "agents.local: '{}' failed to reinstall: {err:?}",
                e.name
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{
      "name": "demo", "version": "0.1.0", "kind": "skill_agent",
      "description": "a demo",
      "toolset": ["demo_echo"],
      "capabilities": [{ "domain": "fs", "rights": ["read"], "scope": "home" }],
      "wasm": { "module": "assets/tools.wasm", "fuel": 400000000 }
    }"#;

    #[test_case]
    fn a_good_manifest_lints_clean() {
        let d = lint(GOOD);
        assert!(!has_errors(&d), "unexpected errors: {d:?}");
    }

    #[test_case]
    fn the_lint_catches_what_the_parser_swallows() {
        // `kind` unrecognized: the parser defaults to Service, so the agent would be
        // installed as something else entirely, silently.
        let d = lint(&GOOD.replace("\"skill_agent\"", "\"skillagent\""));
        assert!(has_errors(&d), "a bad kind must be an error: {d:?}");
        assert!(d.iter().any(|x| x.field == "kind" && x.message.contains("service")));

        // An unknown domain means the whole capability is dropped.
        let d = lint(&GOOD.replace("\"domain\": \"fs\"", "\"domain\": \"files\""));
        assert!(has_errors(&d));
        assert!(d.iter().any(|x| x.message.contains("skipped silently")));

        // An unrecognised scope widens to ANY, which is the *worst* reading of a typo.
        let d = lint(&GOOD.replace("\"scope\": \"home\"", "\"scope\": \"hom\""));
        assert!(has_errors(&d));
        assert!(d.iter().any(|x| x.message.contains("widen")));

        // An unknown right is quietly dropped.
        let d = lint(&GOOD.replace("[\"read\"]", "[\"read\", \"append\"]"));
        assert!(d.iter().any(|x| x.level == Level::Warning && x.message.contains("append")));

        // A misspelled top-level key takes a whole section with it.
        let d = lint(&GOOD.replace("\"capabilities\"", "\"capabilites\""));
        assert!(d.iter().any(|x| x.field == "capabilites"));
    }

    #[test_case]
    fn required_fields_are_required() {
        for missing in ["name", "version", "description", "kind"] {
            let bad = GOOD.replace(&format!("\"{missing}\""), &format!("\"x_{missing}\""));
            let d = lint(&bad);
            assert!(
                d.iter().any(|x| x.level == Level::Error && x.field == missing),
                "{missing} should be reported: {d:?}"
            );
        }
    }

    #[test_case]
    fn malformed_manifests_do_not_panic() {
        for bad in ["", "{", "[]", "null", "\"a\"", "{\"name\":1}", "{\"capabilities\":[{}]}"] {
            let d = lint(bad);
            assert!(!d.is_empty(), "{bad:?} should produce a diagnostic");
        }
    }

    #[test_case]
    fn wasm_block_must_name_a_module() {
        let d = lint(&GOOD.replace("\"module\": \"assets/tools.wasm\"", "\"modul\": \"assets/tools.wasm\""));
        assert!(d.iter().any(|x| x.field == "wasm.module" && x.level == Level::Error));
        let d = lint(&GOOD.replace("400000000", "0"));
        assert!(d.iter().any(|x| x.field == "wasm.fuel" && x.level == Level::Error));
    }

    /// Ids are reused per name and never collide with the system range — an agent's
    /// home, and therefore its memory, depends on it.
    #[test_case]
    fn local_ids_are_stable_per_name_and_clear_of_the_system_range() {
        forget("t_alpha");
        forget("t_beta");
        let (a1, s1) = ids_for("t_alpha");
        let (a2, _) = ids_for("t_beta");
        let (a1b, s1b) = ids_for("t_alpha");
        assert_eq!((a1, s1), (a1b, s1b), "the same name must keep its id");
        assert_ne!(a1, a2, "different names must not share an id");
        assert!(a1.0 >= LOCAL_ID_BASE && a2.0 >= LOCAL_ID_BASE, "must be clear of 9000+ system ids");
        assert!(lookup("t_alpha").is_some());
        forget("t_alpha");
        assert!(lookup("t_alpha").is_none());
        forget("t_beta");
    }

    /// **The security invariant**: a re-install cannot widen a grant.
    #[test_case]
    fn a_reinstall_cannot_widen_the_recorded_grant() {
        use crate::agent::types::{CapDomain, Rights};
        let skill = SkillId(29_001);
        // A human once approved read-only on the agent's home.
        let granted = alloc::vec![CapabilityRequest::new(
            CapDomain::Fs,
            Rights::READ,
            Scope::Path(String::from("/agent/29001/**"))
        )];
        let record = InstallRecord {
            skill,
            installed_ticks: 0,
            granted_capabilities: granted.clone(),
            approved_by: String::from("test"),
            source: InstallSource::Store { key: StoreKey(String::from("x")) },
            verified: true,
            key_id: String::from("k"),
        };
        crate::synapse::fs::write(
            &format!("skills/{}/install", skill.0),
            &postcard::to_allocvec(&record).unwrap(),
        );

        // The manifest is then edited to ask for the whole filesystem, read+write.
        let widened = alloc::vec![CapabilityRequest::new(
            CapDomain::Fs,
            Rights::READ | Rights::WRITE,
            Scope::Any
        )];
        let out = regrant(skill, &widened).expect("a record exists");
        // The edit buys nothing: the intersection is still the old narrow grant.
        assert!(
            !out.iter().any(|c| c.scope == Scope::Any),
            "an edited manifest must not acquire Scope::Any: {out:?}"
        );
        assert!(
            !out.iter().any(|c| c.rights.contains(Rights::WRITE)),
            "nor a right that was never approved: {out:?}"
        );
        // With no record at all, the caller is told to ask rather than handed a grant.
        assert!(regrant(SkillId(29_999), &widened).is_none());
    }
}
