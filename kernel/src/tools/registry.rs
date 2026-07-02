//! The MCP-shaped tool catalogue: the builtin toolset plus any tools registered
//! by providers (Phase F skill-bundled tools). Each [`ToolDef`] carries the
//! agent-facing name/description/JSON-schema and how the tool *binds* to a
//! deterministic executor ([`ToolBinding`]).
//!
//! Per-agent **discovery** ([`for_agent`]) intersects an agent's declared
//! toolset (manifest `toolset`) with what's registered, so the agent only ever
//! sees the tools it may use — its authority is still separately enforced by
//! the Synapse capability gate at dispatch.

use crate::mm::Locked;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// How a tool's validated call reaches a deterministic executor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ToolBinding {
    /// Lower to this Synapse primitive (by wire name) and execute through the
    /// capability/taint-gated, audited executor. `arg_map` pairs the tool's
    /// JSON keys with the primitive's parameter keys (in primitive order).
    Synapse { primitive: &'static str, arg_map: &'static [(&'static str, &'static str)] },
    /// A session-local effect (no FS/console side effect): the todo list.
    SessionTodo,
    /// Dispatch a sub-agent (Phase C) — routed to the agent layer, audited.
    SpawnSubagent,
    /// Load a skill body into context (Phase F).
    LoadSkill,
    /// Run an intent through the compiled-intent path (Phase E).
    RunIntent,
}

/// An agent-facing tool definition (MCP shape).
#[derive(Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON-schema text (shown to the model; used for required-field checks).
    pub input_schema: String,
    /// Required argument keys, extracted from the schema for fast validation.
    pub required: Vec<String>,
    pub binding: ToolBinding,
}

impl ToolDef {
    fn synapse(
        name: &str,
        description: &str,
        schema: &str,
        required: &[&str],
        primitive: &'static str,
        arg_map: &'static [(&'static str, &'static str)],
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: schema.to_string(),
            required: required.iter().map(|s| s.to_string()).collect(),
            binding: ToolBinding::Synapse { primitive, arg_map },
        }
    }
}

/// The builtin toolset (schema Part 1 orchestrator example). Each maps to a
/// Synapse primitive except the session-local / agent-layer ones.
fn builtins() -> Vec<ToolDef> {
    alloc::vec![
        ToolDef::synapse(
            "read",
            "Read a file's contents from the store.",
            r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
            &["path"],
            "mem_fs_read",
            &[("path", "path")],
        ),
        ToolDef::synapse(
            "write",
            "Write text to a file, creating or replacing it.",
            r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#,
            &["path", "content"],
            "mem_fs_write",
            &[("path", "path"), ("content", "text")],
        ),
        ToolDef::synapse(
            "edit",
            "Replace the first occurrence of `old` with `new` in a file.",
            r#"{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}"#,
            &["path", "old", "new"],
            "mem_fs_edit",
            &[("path", "path"), ("old", "old"), ("new", "new")],
        ),
        ToolDef::synapse(
            "list",
            "List the file paths present in the store.",
            r#"{"type":"object","properties":{}}"#,
            &[],
            "list",
            &[],
        ),
        ToolDef::synapse(
            "search",
            "List paths of files whose contents contain `query`.",
            r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#,
            &["query"],
            "mem_fs_search",
            &[("query", "query")],
        ),
        ToolDef::synapse(
            "delete",
            "Delete a file. Destructive — gated on provenance.",
            r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
            &["path"],
            "mem_fs_delete",
            &[("path", "path")],
        ),
        ToolDef::synapse(
            "console",
            "Write a line to the system console.",
            r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#,
            &["text"],
            "console_write",
            &[("text", "text")],
        ),
        ToolDef::synapse(
            "emit_result",
            "Report the agent's final result for this intent.",
            r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#,
            &["text"],
            "emit_result",
            &[("text", "text")],
        ),
        ToolDef {
            name: "todo_write".to_string(),
            description: "Replace the session todo list from a structured payload.".to_string(),
            input_schema: r#"{"type":"object","properties":{"todos":{"type":"array"}},"required":["todos"]}"#.to_string(),
            required: alloc::vec!["todos".to_string()],
            binding: ToolBinding::SessionTodo,
        },
        ToolDef {
            name: "spawn_subagent".to_string(),
            description: "Delegate a self-contained task to an isolated sub-agent; get back a summary.".to_string(),
            input_schema: r#"{"type":"object","properties":{"role":{"type":"string"},"task":{"type":"string"}},"required":["role","task"]}"#.to_string(),
            required: alloc::vec!["role".to_string(), "task".to_string()],
            binding: ToolBinding::SpawnSubagent,
        },
        ToolDef {
            name: "load_skill".to_string(),
            description: "Load an installed skill's instructions when a task matches it.".to_string(),
            input_schema: r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#.to_string(),
            required: alloc::vec!["name".to_string()],
            binding: ToolBinding::LoadSkill,
        },
        ToolDef {
            name: "run".to_string(),
            description: "Run an intent through the compiled-intent path (deterministic replay when cached).".to_string(),
            input_schema: r#"{"type":"object","properties":{"intent":{"type":"string"}},"required":["intent"]}"#.to_string(),
            required: alloc::vec!["intent".to_string()],
            binding: ToolBinding::RunIntent,
        },
    ]
}

/// The live catalogue: builtins + provider-registered tools. Lazily seeded with
/// the builtins on first access.
static REGISTRY: Locked<Vec<ToolDef>> = Locked::new(Vec::new());

/// Access the catalogue under the lock, seeding builtins on first use. All
/// accessors go through here so seeding is atomic and never re-entrant.
fn with_registry<R>(f: impl FnOnce(&mut Vec<ToolDef>) -> R) -> R {
    REGISTRY.with(|reg| {
        if reg.is_empty() {
            *reg = builtins();
        }
        f(reg)
    })
}

/// Register a provider tool (Phase F/G). Ignored if a tool with the same name
/// already exists (builtins win; duplicate providers are idempotent).
pub fn register(def: ToolDef) {
    with_registry(|reg| {
        if reg.iter().any(|t| t.name == def.name) {
            return;
        }
        crate::ktrace::log_fmt(format_args!("tools.register: provider tool '{}'", def.name));
        reg.push(def);
    });
}

/// Look up a tool by name (clones the def — small, and avoids holding the lock).
pub fn get(name: &str) -> Option<ToolDef> {
    with_registry(|reg| reg.iter().find(|t| t.name == name).cloned())
}

/// The tools an agent whose manifest lists `toolset` can see. `"*"` means all
/// registered tools. Authority is still gated per call by Synapse capabilities.
pub fn for_agent(toolset: &[String]) -> Vec<ToolDef> {
    with_registry(|reg| {
        if toolset.iter().any(|t| t == "*") {
            reg.clone()
        } else {
            reg.iter().filter(|t| toolset.iter().any(|n| n == &t.name)).cloned().collect()
        }
    })
}

/// Render a toolset as a compact description block for a model prompt (the
/// "discover its granted tools" surface a real Cortex StepSource injects).
pub fn describe(defs: &[ToolDef]) -> String {
    let mut s = String::new();
    for d in defs {
        s.push_str("- ");
        s.push_str(&d.name);
        s.push_str(": ");
        s.push_str(&d.description);
        s.push_str(" schema=");
        s.push_str(&d.input_schema);
        s.push('\n');
    }
    s
}
