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
    /// Owned (not `&'static`) so skill-bundled tools registered at runtime bind
    /// the same way builtins do.
    Synapse { primitive: String, arg_map: Vec<(String, String)> },
    /// A session-local effect (no FS/console side effect): the todo list.
    SessionTodo,
    /// Dispatch a sub-agent (Phase C) — routed to the agent layer, audited.
    SpawnSubagent,
    /// Load a skill body into context (Phase F).
    LoadSkill,
    /// Run an intent through the compiled-intent path (Phase E).
    RunIntent,
    /// Run a stateless system `/command` (the shell OS commands: disks, mount,
    /// datetime, install, …) and return its printed output. `destructive`
    /// commands are taint-gated at dispatch, like a `DELETE`.
    Shell { command: String, destructive: bool },
    /// Call a tool on a connected MCP server (JSON-RPC `tools/call` over HTTP).
    /// Registered dynamically by `/mcp connect`; the model calls it like any
    /// other tool and the arguments are forwarded verbatim as the JSON-RPC
    /// `arguments` object. `server` is the local connection name.
    Mcp { server: String, tool: String },
    /// Durable per-agent memory under `/agent/<id>/memory/` — `memory_add`,
    /// `memory_get`, `memory_list`, `memory_search`. No Synapse primitive; the
    /// store is the agent's own home (already sandboxed for non-orchestrator
    /// agents).
    AgentMemory,
    /// Agent session/durable storage (`storage_*`) — localStorage-shaped host
    /// API for UI/WASM agents ([`crate::agent::storage`]).
    AgentStorage,
    /// Image / audio / video player tools (`draw_image`, `audio_player`,
    /// `video_player`, …) — host media runtime in the shell action pane.
    Media,
    /// Path/content query over the capability-scoped store listing (`glob` /
    /// `grep`). Pure matching after a scope-filtered `list`/`read` — no new
    /// Synapse primitive (Gate 2.5 applied per path).
    StoreQuery { kind: StoreQueryKind },
    /// MCP resources list / read (not tools/call).
    McpResources { kind: McpResourceKind },
}

/// Which store query the [`ToolBinding::StoreQuery`] tool performs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreQueryKind {
    Glob,
    Grep,
}

/// MCP resource tool kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum McpResourceKind {
    List,
    Read,
}

impl ToolDef {
    /// A tool bound to a connected MCP server tool. `name` is the namespaced
    /// registry name (`mcp__<server>__<tool>`); `server`/`tool` identify the
    /// remote. `schema` is the server-provided JSON input schema.
    pub fn mcp(name: &str, description: &str, schema: &str, server: &str, tool: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: schema.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Mcp { server: server.to_string(), tool: tool.to_string() },
        }
    }
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
        primitive: &str,
        arg_map: &[(&str, &str)],
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: schema.to_string(),
            required: required.iter().map(|s| s.to_string()).collect(),
            binding: ToolBinding::Synapse {
                primitive: primitive.to_string(),
                arg_map: arg_map.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            },
        }
    }

    /// A tool bound to a stateless shell system command. Its single optional
    /// `args` string is passed verbatim as the command's argument line.
    fn shell(name: &str, description: &str, destructive: bool) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: r#"{"type":"object","properties":{"args":{"type":"string","description":"the command's argument line"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Shell { command: name.to_string(), destructive },
        }
    }
}

/// The system `/command` toolset: the OS commands the shell exposes, so the root
/// agent can operate the machine exactly as a human can. Destructive ones
/// (format/install) are taint-gated at dispatch. Agent-layer commands (`/do`,
/// `/agent`, `/subagent`) are already covered by `run` and `spawn_subagent`.
fn shell_commands() -> Vec<ToolDef> {
    alloc::vec![
        ToolDef::shell("help", "Open the Commands browser (search + scroll). args: empty=modal, 'text'=flat list.", false),
        ToolDef::shell("disks", "List every block device and its detected filesystems (read-only).", false),
        ToolDef::shell("ls", "List a store directory (Linux-like). args: '[path] [-l]' (default /). Numeric arg lists a disk volume root.", false),
        ToolDef::shell("cat", "Print a store or mounted file. args: a /path.", false),
        ToolDef::shell("grep", "Search store file contents. args: '<query> [path_glob]'.", false),
        ToolDef::shell("glob", "List store paths matching a glob. args: a pattern (e.g. **/*.md).", false),
        ToolDef::shell("mkdir", "Create a store directory. args: '[-p] <path>'.", false),
        ToolDef::shell("cp", "Copy a store file or tree. args: '[-r] <src> <dst>'.", false),
        ToolDef::shell("mv", "Rename/move a store file or tree. args: '<src> <dst>'.", false),
        ToolDef::shell("rm", "Remove a store file or tree. args: '[-r] <path>'.", true),
        ToolDef::shell("touch", "Create an empty store file. args: a /path.", false),
        ToolDef::shell("pwd", "Print the working directory (always /).", false),
        ToolDef::shell("mount", "Mount a disk volume. args: '<disk> [volume] [/path]' (default /mnt).", false),
        ToolDef::shell("umount", "Unmount a mounted path. args: the /path.", false),
        ToolDef::shell("mounts", "List the currently mounted volumes.", false),
        ToolDef::shell("network", "Show the network status (ip/gw/dns), or configure it. args: empty=status, 'dhcp', 'static <ip/prefix> [gw]', 'dns <ip>'.", false),
        ToolDef::shell("ping", "ICMP-ping a host to check connectivity. args: a hostname or IPv4 address, e.g. 'www.google.com'.", false),
        ToolDef::shell("wifi", "Wi-Fi facade over the NIC. args: 'scan' | 'connect <ssid>' | 'info'.", false),
        ToolDef::shell(
            "channel",
            "Messaging channels (Telegram…). args: list|add telegram <name> <token>|start|stop|send|reply|pair|allow|status.",
            false,
        ),
        ToolDef::shell("http", "curl-like HTTP client. args: [-X M] [-H \"K: V\"] [-d body] [--stream] <url> (http/https).", false),
        ToolDef::shell("datetime", "Show or set the wall clock. args: empty=show, 'YYYY-MM-DD HH:MM'=set, 'tz +5:30'=zone.", false),
        #[cfg(not(feature = "server"))] // GUI tool: absent from server builds
        ToolDef::shell("ui", "View or manage the UI config. args: 'config' | 'reload' | 'reset'.", false),
        ToolDef::shell("shortcuts", "List the keyboard shortcuts.", false),
        ToolDef::shell("skills", "List installed skills.", false),
        ToolDef::shell("ktrace", "Toggle the ktrace log stream in the action pane.", false),
        ToolDef::shell("close", "Close the action pane (chat becomes full-width).", false),
        ToolDef::shell("bench", "Benchmark the matvec kernel throughput.", false),
        ToolDef::shell("perf", "Benchmark end-to-end prefill/decode tokens-per-second.", false),
        ToolDef::shell("infer", "Run the reference-inference parity check.", false),
        ToolDef::shell("mkext4", "Format a disk with ext4. DESTRUCTIVE. args: '<disk> yes'.", true),
        ToolDef::shell("install", "Install Chitti to a disk (GPT: ESP + ext4). DESTRUCTIVE. args: '[<disk>] yes'.", true),
    ]
}

/// The builtin toolset (schema Part 1 orchestrator example). Each maps to a
/// Synapse primitive except the session-local / agent-layer ones.
fn builtins() -> Vec<ToolDef> {
    alloc::vec![
        ToolDef::synapse(
            "read",
            "Read a file from the store. Optional start_line/end_line (1-based) limit the excerpt.",
            r#"{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}},"required":["path"]}"#,
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
            "Replace `old` with `new` in a file. Fails if `old` is empty, missing, or matches more than once unless replace_all is true.",
            r#"{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old","new"]}"#,
            &["path", "old", "new"],
            "mem_fs_edit",
            &[("path", "path"), ("old", "old"), ("new", "new")],
        ),
        ToolDef::synapse(
            "list",
            "List store file paths (flat; use glob for patterns, or the /ls shell command for a directory tree).",
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
        ToolDef {
            name: "glob".to_string(),
            description: "List store paths matching a glob (e.g. *.md, /agent/1/**, **/memory/*).".to_string(),
            input_schema: r#"{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}"#.to_string(),
            required: alloc::vec!["pattern".to_string()],
            binding: ToolBinding::StoreQuery { kind: StoreQueryKind::Glob },
        },
        ToolDef {
            name: "grep".to_string(),
            description: "Search file contents for a substring; returns path:line:text hits (scoped to readable paths).".to_string(),
            input_schema: r#"{"type":"object","properties":{"query":{"type":"string"},"path_glob":{"type":"string"}},"required":["query"]}"#.to_string(),
            required: alloc::vec!["query".to_string()],
            binding: ToolBinding::StoreQuery { kind: StoreQueryKind::Grep },
        },
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
        // UI surfaces — Chess / board agents (ownership-gated at the executor).
        ToolDef::synapse(
            "ui_surface_request",
            "Request a drawing surface in the action pane. kind: canvas|board|image|video|html.",
            r#"{"type":"object","properties":{"kind":{"type":"string"}},"required":["kind"]}"#,
            &["kind"],
            "ui_surface_request",
            &[("kind", "kind")],
        ),
        ToolDef::synapse(
            "ui_draw",
            "Paint a surface you own. ops: 'clear <hex>; rect x y w h <hex>; line …; pixel …'.",
            r#"{"type":"object","properties":{"surface":{"type":"integer"},"ops":{"type":"string"}},"required":["surface","ops"]}"#,
            &["surface", "ops"],
            "ui_draw",
            &[("surface", "surface"), ("ops", "ops")],
        ),
        ToolDef::synapse(
            "ui_event_poll",
            "Poll one click/key event for a surface you own.",
            r#"{"type":"object","properties":{"surface":{"type":"integer"}},"required":["surface"]}"#,
            &["surface"],
            "ui_event_poll",
            &[("surface", "surface")],
        ),
        ToolDef::synapse(
            "ui_surface_close",
            "Close a surface you own.",
            r#"{"type":"object","properties":{"surface":{"type":"integer"}},"required":["surface"]}"#,
            &["surface"],
            "ui_surface_close",
            &[("surface", "surface")],
        ),
        ToolDef::synapse(
            "board_set",
            "Paint an 8×8 chess board from a FEN string onto a surface you own (prefer this over raw ui_draw for chess).",
            r#"{"type":"object","properties":{"surface":{"type":"integer"},"fen":{"type":"string"}},"required":["surface","fen"]}"#,
            &["surface", "fen"],
            "board_set",
            &[("surface", "surface"), ("fen", "fen")],
        ),
        ToolDef::synapse(
            "board_mark",
            "Highlight squares on a board surface (e.g. squares='e2,e4', color='cc785c').",
            r#"{"type":"object","properties":{"surface":{"type":"integer"},"squares":{"type":"string"},"color":{"type":"string"}},"required":["surface","squares"]}"#,
            &["surface", "squares"],
            "board_mark",
            &[("surface", "surface"), ("squares", "squares"), ("color", "color")],
        ),
        // Agent storage (localStorage-shaped; host side of the WASM import ABI).
        ToolDef {
            name: "storage_get".to_string(),
            description: "Read agent storage. args: key, optional scope=session|durable.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"},"scope":{"type":"string"}},"required":["key"]}"#.to_string(),
            required: alloc::vec!["key".to_string()],
            binding: ToolBinding::AgentStorage,
        },
        ToolDef {
            name: "storage_set".to_string(),
            description: "Write agent storage. args: key, value, optional scope=session|durable.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"},"scope":{"type":"string"}},"required":["key","value"]}"#.to_string(),
            required: alloc::vec!["key".to_string(), "value".to_string()],
            binding: ToolBinding::AgentStorage,
        },
        ToolDef {
            name: "storage_list".to_string(),
            description: "List agent storage keys. optional scope=session|durable.".to_string(),
            input_schema: r#"{"type":"object","properties":{"scope":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentStorage,
        },
        ToolDef {
            name: "storage_remove".to_string(),
            description: "Remove an agent storage key. optional scope=session|durable.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"},"scope":{"type":"string"}},"required":["key"]}"#.to_string(),
            required: alloc::vec!["key".to_string()],
            binding: ToolBinding::AgentStorage,
        },
        // Media players (action pane) — image / audio / video. Paths may be
        // store keys, /downloads/…, or mount paths (/mnt/…).
        ToolDef {
            name: "draw_image".to_string(),
            description: "Open an image (.png/.jpg) in the action pane viewer. args: path.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#.to_string(),
            required: alloc::vec!["path".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "image_control".to_string(),
            description: "Control the image viewer: cmd=zoom_in|zoom_out|rotate_cw|rotate_ccw|reset|pan_up|pan_down|pan_left|pan_right.".to_string(),
            input_schema: r#"{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}"#.to_string(),
            required: alloc::vec!["cmd".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "audio_player".to_string(),
            description: "Play audio (.wav/.mp3/.aac) in the action pane. args: path.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#.to_string(),
            required: alloc::vec!["path".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "audio_control".to_string(),
            description: "Control audio playback: cmd=pause|seek|restart|stop; optional ms for seek (e.g. 5000 or -5000).".to_string(),
            input_schema: r#"{"type":"object","properties":{"cmd":{"type":"string"},"ms":{"type":"integer"}},"required":["cmd"]}"#.to_string(),
            required: alloc::vec!["cmd".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "video_player".to_string(),
            description: "Play video (.mp4/.mov/.mkv/.webm H.264) in the action pane. args: path.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#.to_string(),
            required: alloc::vec!["path".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "video_control".to_string(),
            description: "Control video: cmd=pause|seek|restart|mute|volume; frames for seek; delta for volume (±).".to_string(),
            input_schema: r#"{"type":"object","properties":{"cmd":{"type":"string"},"frames":{"type":"integer"},"delta":{"type":"integer"}},"required":["cmd"]}"#.to_string(),
            required: alloc::vec!["cmd".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "media_status".to_string(),
            description: "Report which media (image/audio/video) is loaded in the action pane.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "todo_write".to_string(),
            description: "Replace the session todo list from a structured payload.".to_string(),
            input_schema: r#"{"type":"object","properties":{"todos":{"type":"array"}},"required":["todos"]}"#.to_string(),
            required: alloc::vec!["todos".to_string()],
            binding: ToolBinding::SessionTodo,
        },
        ToolDef {
            name: "spawn_subagent".to_string(),
            description: "Delegate a task to an isolated sub-agent. role: explore|plan|worker|reader (default worker).".to_string(),
            input_schema: r#"{"type":"object","properties":{"role":{"type":"string","description":"explore|plan|worker|reader"},"task":{"type":"string"}},"required":["task"]}"#.to_string(),
            required: alloc::vec!["task".to_string()],
            binding: ToolBinding::SpawnSubagent,
        },
        ToolDef {
            name: "load_skill".to_string(),
            description: "Load an installed skill's L1 instructions into context (optional L2 asset). Alias: skill.".to_string(),
            input_schema: r#"{"type":"object","properties":{"name":{"type":"string","description":"skill name from /skills"},"asset":{"type":"string","description":"optional L2 asset name"}},"required":["name"]}"#.to_string(),
            required: alloc::vec!["name".to_string()],
            binding: ToolBinding::LoadSkill,
        },
        ToolDef {
            name: "skill".to_string(),
            description: "Invoke an installed skill by name: loads L1 body into context; optional asset for L2 progressive disclosure.".to_string(),
            input_schema: r#"{"type":"object","properties":{"name":{"type":"string"},"asset":{"type":"string"}},"required":["name"]}"#.to_string(),
            required: alloc::vec!["name".to_string()],
            binding: ToolBinding::LoadSkill,
        },
        ToolDef {
            name: "enter_plan_mode".to_string(),
            description: "Enter plan mode: only read-only tools + todos/skills until exit_plan_mode.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::SessionTodo, // handled specially in execute_chat_tool before Router
        },
        ToolDef {
            name: "exit_plan_mode".to_string(),
            description: "Leave plan mode (requires human confirmation); re-enables write tools.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::SessionTodo,
        },
        ToolDef {
            name: "mcp_resources".to_string(),
            description: "List resources on a connected MCP server (or all servers if name omitted).".to_string(),
            input_schema: r#"{"type":"object","properties":{"server":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::McpResources { kind: McpResourceKind::List },
        },
        ToolDef {
            name: "mcp_read_resource".to_string(),
            description: "Read one MCP resource by URI from a connected server.".to_string(),
            input_schema: r#"{"type":"object","properties":{"server":{"type":"string"},"uri":{"type":"string"}},"required":["server","uri"]}"#.to_string(),
            required: alloc::vec!["server".to_string(), "uri".to_string()],
            binding: ToolBinding::McpResources { kind: McpResourceKind::Read },
        },
        ToolDef {
            name: "run".to_string(),
            description: "Run an intent through the compiled-intent path (deterministic replay when cached).".to_string(),
            input_schema: r#"{"type":"object","properties":{"intent":{"type":"string"}},"required":["intent"]}"#.to_string(),
            required: alloc::vec!["intent".to_string()],
            binding: ToolBinding::RunIntent,
        },
        ToolDef {
            name: "memory_add".to_string(),
            description: "Store a durable key/value fact in this agent's memory (survives context compaction).".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string","description":"fact name ([A-Za-z0-9._-])"},"value":{"type":"string","description":"fact contents"}},"required":["key","value"]}"#.to_string(),
            required: alloc::vec!["key".to_string(), "value".to_string()],
            binding: ToolBinding::AgentMemory,
        },
        ToolDef {
            name: "memory_get".to_string(),
            description: "Retrieve a previously stored fact by key (exact or short suffix, e.g. name → user.name). On miss, lists closest keys — or use memory_list / memory_search.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string","description":"fact name to look up (prefer exact key from memory_list)"}},"required":["key"]}"#.to_string(),
            required: alloc::vec!["key".to_string()],
            binding: ToolBinding::AgentMemory,
        },
        ToolDef {
            name: "memory_list".to_string(),
            description: "List the keys currently stored in this agent's durable memory.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentMemory,
        },
        ToolDef {
            name: "memory_search".to_string(),
            description: "Search this agent's durable memory keys and values for a substring.".to_string(),
            input_schema: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#.to_string(),
            required: alloc::vec!["query".to_string()],
            binding: ToolBinding::AgentMemory,
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
            reg.extend(shell_commands());
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

/// Register or replace a tool by name — for dynamic sources (MCP servers) that
/// re-connect and refresh their tool set. Unlike [`register`], it overwrites an
/// existing entry of the same name.
pub fn register_replace(def: ToolDef) {
    with_registry(|reg| {
        reg.retain(|t| t.name != def.name);
        reg.push(def);
    });
}

/// Remove a tool by name (e.g. `/mcp disconnect` dropping a server's tools).
/// No-op if absent.
pub fn deregister(name: &str) {
    with_registry(|reg| reg.retain(|t| t.name != name));
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
