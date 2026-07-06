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
use crate::service::ServiceSpec;
use crate::skills::package::SkillPackage;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A built-in system agent: its markdown SOUL + JSON manifest (compiled in),
/// stable ids, and the native service it fronts.
struct SystemAgentDef {
    name: &'static str,
    soul: &'static str,
    manifest_json: &'static str,
    skill_id: SkillId,
    agent_id: AgentId,
    /// The native serve loop + its port setter, if this is a service agent.
    service: Option<(&'static ServiceSpec, fn(u16))>,
}

// Stable ids so `/agent/<id>/` is consistent across boots and doesn't collide
// with runtime-minted ids (which start at 1).
const SYSTEM_SKILL_BASE: u64 = 9000;
const SYSTEM_AGENT_BASE: u64 = 9000;

static SYSTEM_AGENTS: &[SystemAgentDef] = &[
    SystemAgentDef {
        name: "network",
        soul: include_str!("../../../agents/network/SOUL.md"),
        manifest_json: include_str!("../../../agents/network/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 1),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 1),
        service: Some((&crate::service::network::ECHO_SERVICE, crate::service::network::set_echo_port)),
    },
    SystemAgentDef {
        name: "http",
        soul: include_str!("../../../agents/http/SOUL.md"),
        manifest_json: include_str!("../../../agents/http/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 2),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 2),
        service: Some((&crate::service::http::HTTP_SERVICE, crate::service::http::set_port)),
    },
    SystemAgentDef {
        name: "ssh",
        soul: include_str!("../../../agents/ssh/SOUL.md"),
        manifest_json: include_str!("../../../agents/ssh/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 3),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 3),
        service: Some((&crate::service::ssh::SSH_SERVICE, crate::service::ssh::set_port)),
    },
];

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
    Some(ParsedManifest {
        name: j.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        version: j.get("version").and_then(|v| v.as_str()).unwrap_or("0.0.0").to_string(),
        kind,
        description: j.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        toolset,
        capabilities: caps,
        default_port: j.get("default_port").and_then(|v| v.as_i64()).unwrap_or(0) as u16,
        autostart: j.get("autostart").and_then(|v| v.as_bool()).unwrap_or(false),
    })
}

/// Build the signed `SkillPackage` for a system agent from its parsed manifest,
/// SOUL, and stable ids.
fn build_package(def: &SystemAgentDef, m: &ParsedManifest) -> SkillPackage {
    let agent = AgentManifest {
        schema_version: 1,
        id: def.agent_id,
        name: m.name.clone(),
        version: m.version.clone(),
        kind: m.kind,
        description: m.description.clone(),
        system_prompt: def.soul.to_string(),
        toolset: m.toolset.clone(),
        capabilities: m.capabilities.clone(),
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
    };
    let manifest = SkillManifest {
        schema_version: 2,
        id: def.skill_id,
        name: m.name.clone(),
        version: m.version.clone(),
        description: m.description.clone(),
        kind: SkillKind::SkillAgent,
        requested_capabilities: m.capabilities.clone(),
        body_ref: StoreKey(alloc::format!("skills/{}/body.md", def.skill_id.0)),
        bundled_tools: Vec::new(),
        assets: Vec::new(),
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
        assets: Vec::new(),
    };
    pkg.sign();
    pkg
}

/// Install every built-in system agent (idempotent per boot): sign its package,
/// install it granting its full declared capability set (system agents are
/// pre-trusted), land its SOUL in `/agent/<id>/`, register its role, and set its
/// service's default port. Called once from `run_os` after the FS + net are up.
pub fn install_all(now: Ticks) {
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            crate::ktrace::log_fmt(format_args!("system-agent: '{}' manifest.json failed to parse", def.name));
            continue;
        };
        let pkg = build_package(def, &m);
        // Grant the full declared set (pre-trusted); consent-subsetting applies to
        // third-party packages, not the OS's own system agents.
        let approved = m.capabilities.clone();
        match crate::skills::install::install(&pkg, &approved, "system", InstallSource::BootModule { name: alloc::format!("system:{}", def.name) }, now) {
            Ok(rec) => crate::ktrace::log_fmt(format_args!(
                "system-agent: installed '{}' -> /agent/{} ({} caps)",
                def.name,
                def.agent_id.0,
                rec.granted_capabilities.len()
            )),
            Err(e) => crate::ktrace::log_fmt(format_args!("system-agent: install '{}' failed: {:?}", def.name, e)),
        }
        // Pre-load the service's default port so `/agents start <name>` works
        // with no explicit port.
        if let (Some((_, set_port)), true) = (def.service, m.default_port != 0) {
            set_port(m.default_port);
        }
    }
    crate::serial_println!("Chitti: system agents installed (network, http, ssh) in /agent/");
}

/// Look up a system agent's native service by name, for `/agents start <name>`.
pub fn service_for(name: &str) -> Option<(&'static ServiceSpec, fn(u16))> {
    SYSTEM_AGENTS.iter().find(|d| d.name == name).and_then(|d| d.service)
}

/// The installed system agents, for `/agents list`/display: (name, agent_id).
pub fn list() -> Vec<(&'static str, u64)> {
    SYSTEM_AGENTS.iter().map(|d| (d.name, d.agent_id.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_the_bundled_network_manifest() {
        let m = parse_manifest(SYSTEM_AGENTS[0].manifest_json).expect("network manifest parses");
        assert_eq!(m.name, "network");
        assert_eq!(m.kind, AgentKind::Service);
        assert!(m.toolset.iter().any(|t| t == "channel_grant"));
        // Declares Net EXEC and Channel READ|WRITE.
        assert!(m
            .capabilities
            .iter()
            .any(|c| c.domain == CapDomain::Net && c.rights.contains(Rights::EXEC)));
        assert!(m
            .capabilities
            .iter()
            .any(|c| c.domain == CapDomain::Channel && c.rights.contains(Rights::READ | Rights::WRITE)));
    }

    #[test_case]
    fn all_system_manifests_parse_and_bind_a_service() {
        for def in SYSTEM_AGENTS {
            let m = parse_manifest(def.manifest_json).unwrap_or_else(|| panic!("{} manifest", def.name));
            assert_eq!(m.name, def.name);
            assert!(!m.capabilities.is_empty(), "{} declares capabilities", def.name);
            assert!(def.service.is_some(), "{} binds a native service", def.name);
        }
    }

    #[test_case]
    fn parses_scope_variants() {
        assert_eq!(parse_scope("any"), Scope::Any);
        assert_eq!(parse_scope("path:/work/**"), Scope::Path("/work/**".into()));
        assert_eq!(parse_scope("net:*.example.com:80-443"), Scope::Net { host: "*.example.com".into(), port_lo: 80, port_hi: 443 });
    }
}
