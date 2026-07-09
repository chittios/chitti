//! **System agents** — the built-in Network / HTTP / SSH agents, defined in the
//! same installable format as any other agent (a markdown `SOUL.md` persona + a
//! JSON manifest declaring the toolset and capabilities), living under the
//! repo's `agents/` folder. Their definitions are compiled into the kernel image
//! via `include_str!`, and at boot [`install_all`] signs each into a
//! `SkillPackage` and installs it through the normal permissioned flow — so
//! their SOUL lands in `/agent/<id>/SOUL.md`, their capability grant is recorded,
//! and their dispatchable role is registered, exactly like a package fetched
//! from the registry (but pre-trusted, since they ship with the OS).
//!
//! Each service agent is wired to its native serve loop (`service::{network,
//! http, ssh}`); `/agents start <name> <port>` brings one up. The protocol logic
//! is native code below the determinism boundary — the markdown supplies the
//! persona, capability grant, and policy, never the wire handling.

use crate::agent::types::*;
use crate::json::Json;
use crate::skills::package::SkillPackage;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A built-in system agent: its markdown SOUL + JSON manifest (compiled in),
/// stable ids, and any bundled assets that land in its install folder. The
/// web agents (network/http/doc) run together as the [`super::super::service::pipeline`];
/// ssh runs standalone. Wiring is done by the `/agents start` command.
struct SystemAgentDef {
    name: &'static str,
    soul: &'static str,
    manifest_json: &'static str,
    skill_id: SkillId,
    agent_id: AgentId,
    /// `(filename, contents)` written to `/agent/<id>/assets/` on install (e.g.
    /// the Doc agent's HTML + logo). Text, compiled in via `include_str!`.
    assets: &'static [(&'static str, &'static str)],
    /// Binary assets (e.g. `tools.wasm`), via `include_bytes!`.
    binary_assets: &'static [(&'static str, &'static [u8])],
}

// Stable ids so `/agent/<id>/` is consistent across boots and doesn't collide
// with runtime-minted ids (which start at 1).
const SYSTEM_SKILL_BASE: u64 = 9000;
const SYSTEM_AGENT_BASE: u64 = 9000;

// Only agents that actually reason from a SOUL are installed agents. The
// `network` and `http` stages are pure mechanical plumbing (relay bytes, parse a
// protocol) — no judgment, no SOUL — so they live entirely in `crate::service`,
// not here. `doc` routes via `assets/tools.wasm` (deterministic); SOUL remains
// for model fallback. `ssh` carries a login/tunnel persona.
static SYSTEM_AGENTS: &[SystemAgentDef] = &[
    SystemAgentDef {
        name: "doc",
        soul: include_str!("../../../agents/doc/SOUL.md"),
        manifest_json: include_str!("../../../agents/doc/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 1),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 1),
        assets: &[
            ("index.html", include_str!("../../../agents/doc/assets/index.html")),
            ("docs.html", include_str!("../../../agents/doc/assets/docs.html")),
            ("logo.svg", include_str!("../../../agents/doc/assets/logo.svg")),
        ],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/doc/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "ssh",
        soul: include_str!("../../../agents/ssh/SOUL.md"),
        manifest_json: include_str!("../../../agents/ssh/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 2),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 2),
        assets: &[],
        binary_assets: &[],
    },
    // UI agent: paints a chess board via board_set/board_mark; rules live in
    // assets/tools.wasm (built by tools/chess-wasm/).
    SystemAgentDef {
        name: "chess",
        soul: include_str!("../../../agents/chess/SOUL.md"),
        manifest_json: include_str!("../../../agents/chess/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 3),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 3),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/chess/assets/tools.wasm"),
        )],
    },
];

/// The install-folder path (`/agent/<id>`) of a system agent by name.
pub fn home_for(name: &str) -> Option<String> {
    SYSTEM_AGENTS.iter().find(|d| d.name == name).map(|d| crate::agent::home::path(d.agent_id.0))
}

/// The parsed, portable part of a system agent's `manifest.json`.
pub struct ParsedManifest {
    pub name: String,
    pub version: String,
    pub kind: AgentKind,
    pub description: String,
    pub toolset: Vec<String>,
    pub capabilities: Vec<CapabilityRequest>,
    pub default_port: u16,
    pub autostart: bool,
    pub mcp_servers: Vec<crate::agent::types::McpServerSpec>,
}

fn parse_domain(s: &str) -> Option<CapDomain> {
    Some(match s {
        "fs" => CapDomain::Fs,
        "console" => CapDomain::Console,
        "spawn" => CapDomain::Spawn,
        "todo" => CapDomain::Todo,
        "inference" => CapDomain::Inference,
        "ipc" => CapDomain::Ipc,
        "skill_manage" => CapDomain::SkillManage,
        "channel" => CapDomain::Channel,
        "net" => CapDomain::Net,
        "ui" => CapDomain::Ui,
        _ => return None,
    })
}

fn parse_rights(arr: &[Json]) -> Rights {
    let mut r = Rights::empty();
    for v in arr {
        match v.as_str() {
            Some("read") => r |= Rights::READ,
            Some("write") => r |= Rights::WRITE,
            Some("exec") => r |= Rights::EXEC,
            Some("delete") => r |= Rights::DELETE,
            Some("list") => r |= Rights::LIST,
            _ => {}
        }
    }
    r
}

fn parse_scope(s: &str) -> Scope {
    // "home" = read/write only within the agent's own install folder + memory.
    // Resolved to the concrete `/agent/<id>/**` path in `build_package` (which
    // knows the id) via the `$HOME` sentinel.
    if s == "home" {
        return Scope::Path("$HOME/**".to_string());
    }
    if let Some(rest) = s.strip_prefix("path:") {
        return Scope::Path(rest.to_string());
    }
    if let Some(rest) = s.strip_prefix("net:") {
        // net:host:lo-hi  or  net:host:port
        let mut it = rest.rsplitn(2, ':');
        let ports = it.next().unwrap_or("");
        let host = it.next().unwrap_or("*").to_string();
        let (lo, hi) = match ports.split_once('-') {
            Some((a, b)) => (a.parse().unwrap_or(0), b.parse().unwrap_or(0)),
            None => {
                let p = ports.parse().unwrap_or(0);
                (p, p)
            }
        };
        return Scope::Net { host, port_lo: lo, port_hi: hi };
    }
    Scope::Any
}

/// Parse a system agent's `manifest.json` into its portable manifest.
pub fn parse_manifest(json: &str) -> Option<ParsedManifest> {
    let j = Json::parse(json)?;
    let kind = match j.get("kind").and_then(|k| k.as_str()) {
        Some("service") => AgentKind::Service,
        Some("skill_agent") => AgentKind::SkillAgent,
        Some("subagent") => AgentKind::Subagent,
        _ => AgentKind::Service,
    };
    let toolset = j
        .get("toolset")
        .and_then(|t| t.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let mut caps = Vec::new();
    if let Some(arr) = j.get("capabilities").and_then(|c| c.as_array()) {
        for c in arr {
            if let Some(domain) = c.get("domain").and_then(|d| d.as_str()).and_then(parse_domain) {
                let rights = c.get("rights").and_then(|r| r.as_array()).map(parse_rights).unwrap_or_else(Rights::empty);
                let scope = c.get("scope").and_then(|s| s.as_str()).map(parse_scope).unwrap_or(Scope::Any);
                caps.push(CapabilityRequest::new(domain, rights, scope));
            }
        }
    }
    // Optional `mcp_servers`: [{ "name", "url", "bearer"? }, …].
    let mut mcp_servers = Vec::new();
    if let Some(arr) = j.get("mcp_servers").and_then(|m| m.as_array()) {
        for s in arr {
            if let (Some(name), Some(url)) =
                (s.get("name").and_then(|v| v.as_str()), s.get("url").and_then(|v| v.as_str()))
            {
                mcp_servers.push(crate::agent::types::McpServerSpec {
                    name: name.to_string(),
                    url: url.to_string(),
                    bearer: s.get("bearer").and_then(|v| v.as_str()).map(|b| b.to_string()),
                });
            }
        }
    }
    Some(ParsedManifest {
        name: j.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        version: j.get("version").and_then(|v| v.as_str()).unwrap_or("0.0.0").to_string(),
        kind,
        description: j.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        toolset,
        capabilities: caps,
        default_port: j.get("default_port").and_then(|v| v.as_i64()).unwrap_or(0) as u16,
        autostart: j.get("autostart").and_then(|v| v.as_bool()).unwrap_or(false),
        mcp_servers,
    })
}

/// The agent's declared capabilities with the `$HOME` sentinel resolved to its
/// concrete install folder (so a `scope: "home"` grant becomes a real path).
fn resolved_caps(def: &SystemAgentDef, m: &ParsedManifest) -> Vec<CapabilityRequest> {
    let home = crate::agent::home::path(def.agent_id.0);
    m.capabilities
        .iter()
        .map(|c| {
            let scope = match &c.scope {
                Scope::Path(p) if p.contains("$HOME") => Scope::Path(p.replace("$HOME", &home)),
                other => other.clone(),
            };
            CapabilityRequest::new(c.domain, c.rights, scope)
        })
        .collect()
}

/// Build the signed `SkillPackage` for a system agent from its parsed manifest,
/// SOUL, resolved capabilities, and bundled assets.
fn build_package(def: &SystemAgentDef, m: &ParsedManifest, caps: &[CapabilityRequest]) -> SkillPackage {
    let agent = AgentManifest {
        schema_version: 1,
        id: def.agent_id,
        name: m.name.clone(),
        version: m.version.clone(),
        kind: m.kind,
        description: m.description.clone(),
        system_prompt: def.soul.to_string(),
        toolset: m.toolset.clone(),
        capabilities: caps.to_vec(),
        skills: Vec::new(),
        sampling: Sampling::deterministic(1),
        budgets: Budgets {
            max_turns: 8,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 64,
            max_subagents: 2,
            max_depth: 1,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Installed { skill: def.skill_id },
        mcp_servers: m.mcp_servers.clone(),
    };
    // Bundled assets → manifest.assets (declared) + package payload (placed into
    // the agent's install folder by `place_agent_home`).
    let mut asset_meta: Vec<Asset> = def
        .assets
        .iter()
        .map(|(name, content)| Asset {
            name: (*name).to_string(),
            store_ref: StoreKey(alloc::format!("/agent/{}/assets/{name}", def.agent_id.0)),
            bytes: content.len() as u32,
        })
        .collect();
    for (name, content) in def.binary_assets {
        asset_meta.push(Asset {
            name: (*name).to_string(),
            store_ref: StoreKey(alloc::format!("/agent/{}/assets/{name}", def.agent_id.0)),
            bytes: content.len() as u32,
        });
    }
    let mut asset_payload: Vec<(String, Vec<u8>)> = def
        .assets
        .iter()
        .map(|(name, content)| ((*name).to_string(), content.as_bytes().to_vec()))
        .collect();
    for (name, content) in def.binary_assets {
        asset_payload.push(((*name).to_string(), content.to_vec()));
    }
    let manifest = SkillManifest {
        schema_version: 2,
        id: def.skill_id,
        name: m.name.clone(),
        version: m.version.clone(),
        description: m.description.clone(),
        kind: SkillKind::SkillAgent,
        requested_capabilities: caps.to_vec(),
        body_ref: StoreKey(alloc::format!("skills/{}/body.md", def.skill_id.0)),
        bundled_tools: Vec::new(),
        assets: asset_meta,
        agent: Some(agent),
        soul_ref: Some(StoreKey(alloc::format!("/agent/{}/SOUL.md", def.agent_id.0))),
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
        body: alloc::format!("System {} agent.", m.name),
        soul: Some(def.soul.to_string()),
        skill_docs: Vec::new(),
        assets: asset_payload,
    };
    pkg.sign();
    pkg
}

/// Install every built-in system agent (idempotent per boot): sign its package
/// and install it granting its full declared (home-resolved) capability set
/// (system agents are pre-trusted), landing its SOUL + assets in `/agent/<id>/`
/// and registering its role. Called once from `run_os` after the FS + net are up.
pub fn install_all(now: Ticks) {
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            crate::ktrace::log_fmt(format_args!("system-agent: '{}' manifest.json failed to parse", def.name));
            continue;
        };
        let caps = resolved_caps(def, &m);
        let pkg = build_package(def, &m, &caps);
        match crate::skills::install::install(&pkg, &caps, "system", InstallSource::BootModule { name: alloc::format!("system:{}", def.name) }, now) {
            Ok(rec) => crate::ktrace::log_fmt(format_args!(
                "system-agent: installed '{}' -> /agent/{} ({} caps, {} asset(s))",
                def.name,
                def.agent_id.0,
                rec.granted_capabilities.len(),
                def.assets.len() + def.binary_assets.len()
            )),
            Err(e) => crate::ktrace::log_fmt(format_args!("system-agent: install '{}' failed: {:?}", def.name, e)),
        }
    }
    crate::serial_println!("Chitti: system agents installed (doc, ssh, chess) in /agent/");
}

/// The installed system agents, for `/agents list`/display: (name, agent_id).
pub fn list() -> Vec<(&'static str, u64)> {
    SYSTEM_AGENTS.iter().map(|d| (d.name, d.agent_id.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_the_bundled_doc_manifest() {
        let doc = SYSTEM_AGENTS.iter().find(|d| d.name == "doc").expect("doc agent");
        let m = parse_manifest(doc.manifest_json).expect("doc manifest parses");
        assert_eq!(m.name, "doc");
        assert_eq!(m.kind, AgentKind::Service);
        // The Doc agent reads files (channel + fs read), scoped to its home.
        assert!(m.capabilities.iter().any(|c| c.domain == CapDomain::Fs && c.rights.contains(Rights::READ)));
    }

    #[test_case]
    fn all_system_manifests_parse_and_declare_caps() {
        for def in SYSTEM_AGENTS {
            let m = parse_manifest(def.manifest_json).unwrap_or_else(|| panic!("{} manifest", def.name));
            assert_eq!(m.name, def.name);
            assert!(!m.capabilities.is_empty(), "{} declares capabilities", def.name);
        }
        // SOUL-backed system agents: doc + ssh + chess (UI). network/http are
        // pure service-layer plumbing, not installed agents.
        assert_eq!(SYSTEM_AGENTS.len(), 3);
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "chess"));
        assert!(!SYSTEM_AGENTS.iter().any(|d| d.name == "network" || d.name == "http"));
    }

    #[test_case]
    fn doc_home_scope_resolves_to_the_agent_folder() {
        let doc = SYSTEM_AGENTS.iter().find(|d| d.name == "doc").unwrap();
        let m = parse_manifest(doc.manifest_json).unwrap();
        let caps = resolved_caps(doc, &m);
        // The doc agent's Fs READ scope is its own install folder, not "$HOME".
        let fs = caps.iter().find(|c| c.domain == CapDomain::Fs).expect("doc has an Fs cap");
        assert_eq!(fs.rights, Rights::READ);
        match &fs.scope {
            Scope::Path(p) => assert_eq!(p, &alloc::format!("/agent/{}/**", doc.agent_id.0)),
            other => panic!("expected a resolved path scope, got {other:?}"),
        }
    }

    #[test_case]
    fn parses_scope_variants() {
        assert_eq!(parse_scope("any"), Scope::Any);
        assert_eq!(parse_scope("path:/work/**"), Scope::Path("/work/**".into()));
        assert_eq!(parse_scope("net:*.example.com:80-443"), Scope::Net { host: "*.example.com".into(), port_lo: 80, port_hi: 443 });
    }
}
