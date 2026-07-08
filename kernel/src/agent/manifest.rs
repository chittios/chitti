//! Builtin agent roles and the bridge from the declarative capability model
//! (`CapabilityRequest`) down to the live, unforgeable kernel capabilities
//! (`cap::Right::InvokePrimitive`).
//!
//! The declarative `CapDomain`/`Rights`/`Scope` in a manifest is *portable and
//! human-readable* — what a role asks for, shown at an install prompt. The
//! runtime authority is the existing seL4-style per-task capability table. At
//! spawn we lower the granted requests into `InvokePrimitive` grants on the
//! agent's task, so Synapse's capability gate (unchanged) enforces them with no
//! ambient authority (DECISIONS.md #13).

use crate::agent::types::*;
use crate::cap::PrimitiveId;
use crate::synapse::registry;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// A fixed AgentId for the builtin orchestrator (id 1, matching the schema
/// example). Sub-agent roles get minted ids.
pub const ORCHESTRATOR_ID: AgentId = AgentId(1);

/// The interactive main agent behind the intent shell (schema Part 1 example).
pub fn orchestrator_manifest() -> AgentManifest {
    AgentManifest {
        schema_version: 1,
        id: ORCHESTRATOR_ID,
        name: "orchestrator".to_string(),
        version: "1.0.0".to_string(),
        kind: AgentKind::Orchestrator,
        description: "Interactive main agent behind the intent shell.".to_string(),
        system_prompt: "You are Chitti's main agent. Plan, use tools, delegate to sub-agents \
                        when a task is self-contained, and keep the user informed."
            .to_string(),
        toolset: vec![
            // Agent-layer builtins.
            "read".into(),
            "write".into(),
            "edit".into(),
            "list".into(),
            "search".into(),
            "run".into(),
            "spawn_subagent".into(),
            "todo_write".into(),
            "load_skill".into(),
            "emit_result".into(),
            // System `/command` tools — the root agent drives the machine like a
            // human at the shell (see tools::registry::shell_commands).
            "help".into(),
            "disks".into(),
            "ls".into(),
            "mount".into(),
            "umount".into(),
            "mounts".into(),
            "cat".into(),
            "network".into(),
            "ping".into(),
            "wifi".into(),
            "http".into(),
            "datetime".into(),
            "ui".into(),
            "shortcuts".into(),
            "skills".into(),
            "ktrace".into(),
            "close".into(),
            "bench".into(),
            "perf".into(),
            "infer".into(),
            "mkext4".into(),
            "install".into(),
        ],
        capabilities: vec![
            CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE | Rights::LIST, Scope::Any),
            CapabilityRequest::new(CapDomain::Console, Rights::READ | Rights::WRITE, Scope::Any),
            CapabilityRequest::new(CapDomain::Spawn, Rights::EXEC, Scope::Any),
            CapabilityRequest::new(CapDomain::Todo, Rights::READ | Rights::WRITE, Scope::Any),
            CapabilityRequest::new(CapDomain::Inference, Rights::EXEC, Scope::Any),
        ],
        skills: Vec::new(),
        sampling: Sampling { temperature: 0.2, top_p: 0.9, seed: 42, max_output_tokens: 1024 },
        budgets: Budgets {
            max_turns: 64,
            max_context_tokens: 8192,
            compact_threshold: 6500,
            max_tool_calls: 256,
            max_subagents: 8,
            max_depth: 2,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 512, style: SummaryStyle::Structured },
        origin: Origin::Builtin,
        mcp_servers: alloc::vec::Vec::new(),
    }
}

/// A general **worker** sub-agent role for model-driven delegation from the
/// shell agent: it runs its own inference-backed agentic loop and may call the
/// system `/command` tools, with authority strictly attenuated from its parent
/// (read/list files, console, inference — no writes, no skill management).
/// `max_depth: 0` so it cannot itself delegate.
pub fn worker_subagent_manifest() -> AgentManifest {
    AgentManifest {
        schema_version: 1,
        id: next_agent_id(),
        name: "worker".to_string(),
        version: "1.0.0".to_string(),
        kind: AgentKind::Subagent,
        description: "Complete a self-contained task with tools and report a summary.".to_string(),
        system_prompt: "You complete the delegated task using tools and report the result.".to_string(),
        // Read-only system tools: a worker can inspect the machine but not
        // format/install/write — mirroring its attenuated capability set.
        toolset: vec![
            "help".into(),
            "disks".into(),
            "ls".into(),
            "mounts".into(),
            "cat".into(),
            "network".into(),
            "ping".into(),
            "wifi".into(),
            "http".into(),
            "datetime".into(),
            "skills".into(),
        ],
        capabilities: vec![
            CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::LIST, Scope::Any),
            CapabilityRequest::new(CapDomain::Console, Rights::READ | Rights::WRITE, Scope::Any),
            CapabilityRequest::new(CapDomain::Inference, Rights::EXEC, Scope::Any),
        ],
        skills: Vec::new(),
        sampling: Sampling { temperature: 0.7, top_p: 0.8, seed: 7, max_output_tokens: 1024 },
        budgets: Budgets {
            max_turns: 8,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 8,
            max_subagents: 0,
            max_depth: 0,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Builtin,
        mcp_servers: alloc::vec::Vec::new(),
    }
}

/// A minimal read-only sub-agent role for Phase C tests/demos: it can read and
/// list files and emit a result, nothing else. `max_depth: 0` so it cannot
/// itself delegate.
pub fn reader_subagent_manifest() -> AgentManifest {
    AgentManifest {
        schema_version: 1,
        id: next_agent_id(),
        name: "reader".to_string(),
        version: "1.0.0".to_string(),
        kind: AgentKind::Subagent,
        description: "Read files and report their contents. Use for self-contained lookups.".to_string(),
        system_prompt: "You read the requested files and report exactly what they contain.".to_string(),
        toolset: vec!["read".into(), "list".into(), "emit_result".into()],
        capabilities: vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::LIST, Scope::Any)],
        skills: Vec::new(),
        sampling: Sampling::deterministic(7),
        budgets: Budgets {
            max_turns: 12,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 32,
            max_subagents: 0,
            max_depth: 0,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Builtin,
        mcp_servers: alloc::vec::Vec::new(),
    }
}

/// Lower a set of granted `CapabilityRequest`s to the concrete Synapse
/// `PrimitiveId`s they authorize. `emit_result` is always included — an agent
/// can always report its own result. Domains with no direct primitive
/// (`Inference` — internal to Cortex; `Todo` — session-local; `Ipc`) map to no
/// primitive here.
pub fn primitives_for(caps: &[CapabilityRequest]) -> Vec<PrimitiveId> {
    let mut prims = vec![registry::EMIT_RESULT];
    for c in caps {
        match c.domain {
            CapDomain::Fs => {
                if c.rights.contains(Rights::READ) {
                    prims.push(registry::MEM_FS_READ);
                    prims.push(registry::MEM_FS_SEARCH); // search is a read-only query
                }
                if c.rights.contains(Rights::WRITE) {
                    prims.push(registry::MEM_FS_WRITE);
                    prims.push(registry::MEM_FS_EDIT); // edit is a guarded write
                }
                if c.rights.contains(Rights::LIST) {
                    prims.push(registry::LIST);
                }
                if c.rights.contains(Rights::DELETE) {
                    prims.push(registry::MEM_FS_DELETE);
                }
            }
            CapDomain::Console => prims.push(registry::CONSOLE_WRITE),
            CapDomain::Spawn => prims.push(registry::SPAWN_AGENT),
            CapDomain::Channel => {
                // Coarse "may touch the channel ABI" authority. Which specific
                // channel/direction is a separate per-end `ChannelRead/Write`
                // grant resolved by the executor, so this stays broad on purpose.
                prims.push(registry::CHANNEL_CREATE);
                prims.push(registry::CHANNEL_CLOSE);
                prims.push(registry::CHANNEL_GRANT);
                if c.rights.contains(Rights::WRITE) {
                    prims.push(registry::CHANNEL_WRITE);
                }
                if c.rights.contains(Rights::READ) {
                    prims.push(registry::CHANNEL_READ);
                }
            }
            CapDomain::Net => {
                // EXEC = listen/accept (a server); READ = http GET; WRITE = http POST.
                if c.rights.contains(Rights::EXEC) {
                    prims.push(registry::NET_LISTEN);
                    prims.push(registry::NET_ACCEPT);
                }
                if c.rights.contains(Rights::READ) {
                    prims.push(registry::NET_HTTP_GET);
                }
                if c.rights.contains(Rights::WRITE) {
                    prims.push(registry::NET_HTTP_POST);
                }
            }
            CapDomain::Ui => {
                // EXEC = request/own a surface + poll its input; WRITE = draw;
                // DELETE = close. Which surface is gated by ownership, not scope.
                if c.rights.contains(Rights::EXEC) {
                    prims.push(registry::UI_SURFACE_REQUEST);
                    prims.push(registry::UI_EVENT_POLL);
                }
                if c.rights.contains(Rights::WRITE) {
                    prims.push(registry::UI_DRAW);
                }
                if c.rights.contains(Rights::DELETE) {
                    prims.push(registry::UI_SURFACE_CLOSE);
                }
            }
            CapDomain::Inference | CapDomain::Todo | CapDomain::Ipc | CapDomain::SkillManage => {}
        }
    }
    prims.sort_unstable();
    prims.dedup();
    prims
}

/// Mint live capability tokens for `granted` and grant the corresponding
/// `InvokePrimitive` rights on `task`. Returns the display/audit `Capability`
/// list to store on the session. This is the one place declarative caps become
/// live authority.
pub fn grant_to_task(task: crate::sched::TaskId, granted: &[CapabilityRequest]) -> Vec<Capability> {
    for prim in primitives_for(granted) {
        crate::cap::grant(task, crate::cap::Right::InvokePrimitive(prim));
    }
    // Record the granted scopes so the executor's Gate 2.5 can enforce
    // path/host/port limits (not just primitive-granularity authority).
    crate::cap::grant_scopes(task, granted);
    granted
        .iter()
        .map(|req| Capability { id: next_cap_id(), req: req.clone() })
        .collect()
}

/// Human-readable one-line render of a capability request (for install/consent
/// prompts and audit).
pub fn render_cap(c: &CapabilityRequest) -> String {
    let scope = match &c.scope {
        Scope::Any => "any".to_string(),
        Scope::Path(p) => alloc::format!("path:{p}"),
        Scope::Resource(r) => alloc::format!("resource:{r}"),
        Scope::Net { host, port_lo, port_hi } => {
            if port_lo == port_hi {
                alloc::format!("net:{host}:{port_lo}")
            } else {
                alloc::format!("net:{host}:{port_lo}-{port_hi}")
            }
        }
    };
    alloc::format!("{:?} {:?} @ {}", c.domain, c.rights, scope)
}
