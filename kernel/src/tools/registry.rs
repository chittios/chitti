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
    /// Call an export of the **active session agent's** `assets/tools.wasm`
    /// (string ABI). Package SOUL + wasm only — no per-app host match arms.
    AgentWasm,
    /// HTTP(S) fetch + save body to the store (`download` tool).
    Download,
    /// Browser agent tools (`browser_open`, scroll, click, …) — host HTML engine.
    Browser,
    /// Path/content query over the capability-scoped store listing (`glob` /
    /// `grep`). Pure matching after a scope-filtered `list`/`read` — no new
    /// Synapse primitive (Gate 2.5 applied per path).
    StoreQuery { kind: StoreQueryKind },
    /// MCP resources list / read (not tools/call).
    McpResources { kind: McpResourceKind },
    /// Freeform Chitti shell line (`run_shell_command`) — first token is a
    /// system `/command` name, not POSIX sh.
    RunShellCommand,
    /// Background task control (`task_output`, `kill_task`, `monitor`, …).
    BgTask,
    /// Multi-option human question via modal (`ask_user_question`).
    AskUser,
    /// Packaged web tools (`web_search`, `web_fetch`) over the HTTP client.
    Web,
    /// `search_replace` — edit with old_string/new_string (or old/new) keys.
    SearchReplace,
}

/// Which store query the [`ToolBinding::StoreQuery`] tool performs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreQueryKind {
    Glob,
    Grep,
    ListDir,
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

    /// Package `tools.wasm` export (name == export).
    fn wasm(name: &str, description: &str, schema: &str, required: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: schema.to_string(),
            required: required.iter().map(|s| s.to_string()).collect(),
            binding: ToolBinding::AgentWasm,
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
        ToolDef::shell(
            "about",
            "About ChittiOS (logo, version, build) — same as clicking the status-bar brand. args: empty=modal, 'text'=serial.",
            false,
        ),
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
        ToolDef::shell("wifi", "Wi-Fi: real Broadcom on Apple (chitti.wifi); facade on QEMU. args: info|scan|connect <ssid>|load.", false),
        ToolDef::shell(
            "display",
            "Screen size. args: empty=status, 'list', 'scale <1-4>|auto' (text size — USE THIS for \"too small\"), 'set <WxH>|native' (logical desktop, letterboxes), 'boot <WxH>|auto' (panel mode, next boot).",
            false,
        ),
        ToolDef::shell(
            "statusbar",
            "Move the OS status bar to a screen edge. args: empty=current position, or one of 'top' | 'bottom' | 'left' | 'right'. Applies instantly and persists; left/right make it a column that stacks its fields.",
            false,
        ),
        ToolDef::shell(
            "channel",
            "Messaging channels (Telegram…). args: list|add telegram <name> <token>|start|stop|send|reply|pair|allow|status.",
            false,
        ),
        ToolDef::shell("http", "curl-like HTTP client. args: [-X M] [-H \"K: V\"] [-d body] [--stream] [-O|-o file] <url> (http/https). Prefer download tool to save files.", false),
        ToolDef {
            name: "download".to_string(),
            description: "HTTP(S) GET a URL and save the body to the store. args: url (required), optional path (default /downloads/<basename>).".to_string(),
            input_schema: r#"{"type":"object","properties":{"url":{"type":"string"},"path":{"type":"string"}},"required":["url"]}"#.to_string(),
            required: alloc::vec!["url".to_string()],
            binding: ToolBinding::Download,
        },
        ToolDef {
            name: "browser_open".to_string(),
            description: "Open/render a web page in the action pane. args: url (http/https).".to_string(),
            input_schema: r#"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"#.to_string(),
            required: alloc::vec!["url".to_string()],
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_navigate".to_string(),
            description: "Navigate the browser tab to a URL (pushes history). args: url.".to_string(),
            input_schema: r#"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"#.to_string(),
            required: alloc::vec!["url".to_string()],
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_back".to_string(),
            description: "Go back in browser history.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_scroll".to_string(),
            description: "Scroll the browser page. args: dy (pixels) or page (+1/-1 page).".to_string(),
            input_schema: r#"{"type":"object","properties":{"dy":{"type":"integer"},"page":{"type":"integer"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_click".to_string(),
            description: "Click at surface coords (hits links). args: x, y.".to_string(),
            input_schema: r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"]}"#.to_string(),
            required: alloc::vec!["x".to_string(), "y".to_string()],
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_status".to_string(),
            description: "Current browser URL, title, scroll, size.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_links".to_string(),
            description: "List links on the current page (href + text).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Browser,
        },
        ToolDef {
            name: "browser_text".to_string(),
            description: "Plain-text extract of the current page (for answering questions). optional max chars.".to_string(),
            input_schema: r#"{"type":"object","properties":{"max":{"type":"integer"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::Browser,
        },
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
        ToolDef::shell("install", "Install ChittiOS to a disk (GPT: ESP + ext4). DESTRUCTIVE -- repartitions the WHOLE disk, erasing any existing OS. args: '[<disk>] yes|format|plan|alongside'; 'plan' is READ-ONLY and reports whether installing next to an existing OS is possible; 'alongside' does it non-destructively (adds our loader to the existing ESP, backing up the current one; no partition is modified).", true),
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
            description: "Search file contents for a substring; returns path:line:text hits (scoped to readable paths). Optional path_glob, head_limit (default 50), case_insensitive.".to_string(),
            input_schema: r#"{"type":"object","properties":{"query":{"type":"string"},"pattern":{"type":"string"},"path_glob":{"type":"string"},"head_limit":{"type":"integer"},"case_insensitive":{"type":"boolean"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::StoreQuery { kind: StoreQueryKind::Grep },
        },
        ToolDef {
            name: "list_dir".to_string(),
            description: "List direct children of a store directory (files and subdirs/). path defaults to /.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::StoreQuery { kind: StoreQueryKind::ListDir },
        },
        ToolDef {
            name: "search_replace".to_string(),
            description: "Replace old_string with new_string in a file (unique match unless replace_all). Alias of edit with Grok-style arg names.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path"]}"#.to_string(),
            required: alloc::vec!["path".to_string()],
            binding: ToolBinding::SearchReplace,
        },
        ToolDef {
            name: "run_shell_command".to_string(),
            description: "Run a Chitti system command (not POSIX sh). command is e.g. \"ls /\", \"ping 1.1.1.1\", \"http https://…\". Optional background:true returns task_id.".to_string(),
            input_schema: r#"{"type":"object","properties":{"command":{"type":"string"},"args":{"type":"string"},"background":{"type":"boolean"}},"required":["command"]}"#.to_string(),
            required: alloc::vec!["command".to_string()],
            binding: ToolBinding::RunShellCommand,
        },
        ToolDef {
            name: "task_output".to_string(),
            description: "Get output/status of a background task. task_id required; optional timeout_ms to wait.".to_string(),
            input_schema: r#"{"type":"object","properties":{"task_id":{"type":"integer"},"timeout_ms":{"type":"integer"}},"required":["task_id"]}"#.to_string(),
            required: alloc::vec!["task_id".to_string()],
            binding: ToolBinding::BgTask,
        },
        ToolDef {
            name: "kill_task".to_string(),
            description: "Stop a background task by task_id.".to_string(),
            input_schema: r#"{"type":"object","properties":{"task_id":{"type":"integer"}},"required":["task_id"]}"#.to_string(),
            required: alloc::vec!["task_id".to_string()],
            binding: ToolBinding::BgTask,
        },
        ToolDef {
            name: "monitor".to_string(),
            description: "Poll a Chitti command on an interval (ms, min 1000). Returns task_id; lines accumulate for task_output.".to_string(),
            input_schema: r#"{"type":"object","properties":{"command":{"type":"string"},"interval_ms":{"type":"integer"}},"required":["command"]}"#.to_string(),
            required: alloc::vec!["command".to_string()],
            binding: ToolBinding::BgTask,
        },
        ToolDef {
            name: "list_tasks".to_string(),
            description: "List background agent tasks.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::BgTask,
        },
        ToolDef {
            name: "ask_user_question".to_string(),
            description: "Ask the human a multiple-choice question via popup modal. options is a JSON array of short labels (2–8).".to_string(),
            input_schema: r#"{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"string"}},"required":["question","options"]}"#.to_string(),
            required: alloc::vec!["question".to_string(), "options".to_string()],
            binding: ToolBinding::AskUser,
        },
        ToolDef {
            name: "web_search".to_string(),
            description: "Search the web for a query; returns a short text summary of top results (via HTTP). Content is untrusted.".to_string(),
            input_schema: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#.to_string(),
            required: alloc::vec!["query".to_string()],
            binding: ToolBinding::Web,
        },
        ToolDef {
            name: "web_fetch".to_string(),
            description: "Fetch a URL and return truncated plain text (HTML tags stripped when present). Prefer over raw http for reading pages.".to_string(),
            input_schema: r#"{"type":"object","properties":{"url":{"type":"string"},"max_bytes":{"type":"integer"}},"required":["url"]}"#.to_string(),
            required: alloc::vec!["url".to_string()],
            binding: ToolBinding::Web,
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
            name: "pdf_preview".to_string(),
            description: "Preview a PDF: deterministic wasm digest (pdf agent), text in an editor tab. args: path.".to_string(),
            input_schema: r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#.to_string(),
            required: alloc::vec!["path".to_string()],
            binding: ToolBinding::Media,
        },
        ToolDef {
            name: "git_command".to_string(),
            description: "Run a git command (init|status|add|commit|log|branch|checkout|clone|push). Deterministic wasm (git agent). args: the full `/git …` line.".to_string(),
            input_schema: r#"{"type":"object","properties":{"args":{"type":"string"}},"required":["args"]}"#.to_string(),
            required: alloc::vec!["args".to_string()],
            binding: ToolBinding::AgentWasm,
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
        // Package WASM app tools (export name == tool name in agent tools.wasm).
        ToolDef {
            name: "notes_start".to_string(),
            description: "Open the Notes package UI (list/read durable notes).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "notes_list".to_string(),
            description: "List durable note keys (notes agent WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "notes_get".to_string(),
            description: "Read a note by key (notes agent WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#.to_string(),
            required: alloc::vec!["key".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "notes_set".to_string(),
            description: "Write a note. args: key, body.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"},"body":{"type":"string"}},"required":["key","body"]}"#.to_string(),
            required: alloc::vec!["key".to_string(), "body".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "notes_remove".to_string(),
            description: "Delete a note by key.".to_string(),
            input_schema: r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#.to_string(),
            required: alloc::vec!["key".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_start".to_string(),
            description: "Init paint canvas (package UI / WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_clear".to_string(),
            description: "Clear paint canvas. optional color=.".to_string(),
            input_schema: r#"{"type":"object","properties":{"color":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_rect".to_string(),
            description: "Draw rect. args: x,y,w,h, optional color.".to_string(),
            input_schema: r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"},"color":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_line".to_string(),
            description: "Draw line. args: x0,y0,x1,y1, optional color.".to_string(),
            input_schema: r#"{"type":"object","properties":{"x0":{"type":"integer"},"y0":{"type":"integer"},"x1":{"type":"integer"},"y1":{"type":"integer"},"color":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_pixel".to_string(),
            description: "Draw pixel. args: x,y, optional color.".to_string(),
            input_schema: r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"color":{"type":"string"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_draw".to_string(),
            description: "Raw draw-op string. args: ops.".to_string(),
            input_schema: r#"{"type":"object","properties":{"ops":{"type":"string"}},"required":["ops"]}"#.to_string(),
            required: alloc::vec!["ops".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "paint_status".to_string(),
            description: "Paint agent status.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "slides_start".to_string(),
            description: "Start slide deck (WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "slides_next".to_string(),
            description: "Next slide.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "slides_prev".to_string(),
            description: "Previous slide.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "slides_goto".to_string(),
            description: "Go to slide n (1-based).".to_string(),
            input_schema: r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}"#.to_string(),
            required: alloc::vec!["n".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "slides_status".to_string(),
            description: "Current slide status.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "mines_start".to_string(),
            description: "Start minesweeper (WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "mines_click".to_string(),
            description: "Reveal cell. args: row, col (0..8).".to_string(),
            input_schema: r#"{"type":"object","properties":{"row":{"type":"integer"},"col":{"type":"integer"}},"required":["row","col"]}"#.to_string(),
            required: alloc::vec!["row".to_string(), "col".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "mines_flag".to_string(),
            description: "Toggle flag. args: row, col.".to_string(),
            input_schema: r#"{"type":"object","properties":{"row":{"type":"integer"},"col":{"type":"integer"}},"required":["row","col"]}"#.to_string(),
            required: alloc::vec!["row".to_string(), "col".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "mines_status".to_string(),
            description: "Minesweeper status.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "snake_start".to_string(),
            description: "Start snake (WASM).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "snake_dir".to_string(),
            description: "Snake direction. args: dir=up|down|left|right.".to_string(),
            input_schema: r#"{"type":"object","properties":{"dir":{"type":"string"}},"required":["dir"]}"#.to_string(),
            required: alloc::vec!["dir".to_string()],
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "snake_tick".to_string(),
            description: "Advance snake one step.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "snake_status".to_string(),
            description: "Snake score/status.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "synth_tone".to_string(),
            description: "Play tone. args: hz, ms.".to_string(),
            input_schema: r#"{"type":"object","properties":{"hz":{"type":"integer"},"ms":{"type":"integer"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "synth_beep".to_string(),
            description: "Short beep. optional hz.".to_string(),
            input_schema: r#"{"type":"object","properties":{"hz":{"type":"integer"}}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "synth_stop".to_string(),
            description: "Stop synth (device drains).".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        ToolDef {
            name: "synth_status".to_string(),
            description: "Synth status.".to_string(),
            input_schema: r#"{"type":"object","properties":{}}"#.to_string(),
            required: Vec::new(),
            binding: ToolBinding::AgentWasm,
        },
        // Chess package tools (tools/chess-wasm).
        ToolDef::wasm(
            "chess_legal",
            "Legal destinations from a square (optional fen, from).",
            r#"{"type":"object","properties":{"fen":{"type":"string"},"from":{"type":"string"}},"required":["from"]}"#,
            &["from"],
        ),
        ToolDef::wasm(
            "chess_try_move",
            "Apply a chess move (from,to; optional fen). Paints board on success.",
            r#"{"type":"object","properties":{"fen":{"type":"string"},"from":{"type":"string"},"to":{"type":"string"}},"required":["from","to"]}"#,
            &["from", "to"],
        ),
        // Default-OS UI package tools (tools/apps-wasm).
        ToolDef::wasm("calc_start", "Start calculator UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("calc_eval", "Evaluate expression (expr like 12+3).", r#"{"type":"object","properties":{"expr":{"type":"string"}}}"#, &[]),
        ToolDef::wasm("calc_status", "Calculator display status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("clock_start", "Start clock UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("clock_set_timer", "Set timer seconds.", r#"{"type":"object","properties":{"seconds":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("clock_status", "Clock status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("files_start", "Start files browser UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("files_list", "List virtual files.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("files_get", "Read virtual file by key.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("files_set", "Write virtual file. key+body.", r#"{"type":"object","properties":{"key":{"type":"string"},"body":{"type":"string"}},"required":["key","body"]}"#, &["key", "body"]),
        ToolDef::wasm("files_remove", "Remove virtual file.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("files_status", "Files browser status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("gallery_start", "Start gallery UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("gallery_list", "List gallery keys.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("gallery_get", "Get gallery entry.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("gallery_set", "Set gallery entry meta.", r#"{"type":"object","properties":{"key":{"type":"string"},"body":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("gallery_status", "Gallery status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sheets_start", "Start sheets UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sheets_get", "Get cell. col,row 0-based.", r#"{"type":"object","properties":{"col":{"type":"integer"},"row":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("sheets_set", "Set cell value.", r#"{"type":"object","properties":{"col":{"type":"integer"},"row":{"type":"integer"},"value":{"type":"string"}}}"#, &[]),
        ToolDef::wasm("sheets_status", "Sheets status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("calendar_start", "Start calendar UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("calendar_add", "Add event date=YYYY-MM-DD body=.", r#"{"type":"object","properties":{"date":{"type":"string"},"body":{"type":"string"}},"required":["date"]}"#, &["date"]),
        ToolDef::wasm("calendar_list", "List calendar events.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("calendar_status", "Calendar status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("contacts_start", "Start contacts UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("contacts_list", "List contacts.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("contacts_get", "Get contact by name.", r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#, &["name"]),
        ToolDef::wasm("contacts_set", "Set contact name+body.", r#"{"type":"object","properties":{"name":{"type":"string"},"body":{"type":"string"}},"required":["name"]}"#, &["name"]),
        ToolDef::wasm("contacts_remove", "Remove contact.", r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#, &["name"]),
        ToolDef::wasm("contacts_status", "Contacts status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("writer_start", "Start writer UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("writer_get", "Get document body.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("writer_set", "Set document body.", r#"{"type":"object","properties":{"body":{"type":"string"}},"required":["body"]}"#, &["body"]),
        ToolDef::wasm("writer_status", "Writer status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("archive_start", "Start archive UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("archive_pack", "Pack keys into named archive.", r#"{"type":"object","properties":{"name":{"type":"string"},"keys":{"type":"string"}},"required":["name"]}"#, &["name"]),
        ToolDef::wasm("archive_unpack", "Unpack named archive.", r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#, &["name"]),
        ToolDef::wasm("archive_list", "List archives.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("archive_status", "Archive status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("hex_start", "Start hex viewer UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("hex_open", "Open storage key in hex viewer.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("hex_dump", "Hex dump of key.", r#"{"type":"object","properties":{"key":{"type":"string"}}}"#, &[]),
        ToolDef::wasm("hex_status", "Hex viewer status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("game2048_start", "Start 2048 game UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("game2048_status", "2048 score/status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("activity_start", "Start activity panel UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("activity_set", "Set activity tasks/mem.", r#"{"type":"object","properties":{"tasks":{"type":"integer"},"mem":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("activity_status", "Activity status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("weather_start", "Start weather card UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("weather_set", "Set weather temp/cond/place.", r#"{"type":"object","properties":{"temp":{"type":"integer"},"cond":{"type":"string"},"place":{"type":"string"}}}"#, &[]),
        ToolDef::wasm("weather_status", "Weather status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("settings_start", "Start settings UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("settings_get", "Get settings prefs.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("settings_set", "Set settings prefs.", r#"{"type":"object","properties":{"theme":{"type":"integer"},"opacity":{"type":"integer"},"mode":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("settings_status", "Settings status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("dict_start", "Start dictionary UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("dict_lookup", "Lookup word definition.", r#"{"type":"object","properties":{"word":{"type":"string"}},"required":["word"]}"#, &["word"]),
        ToolDef::wasm("dict_set", "Define a word.", r#"{"type":"object","properties":{"word":{"type":"string"},"def":{"type":"string"}},"required":["word","def"]}"#, &["word", "def"]),
        ToolDef::wasm("dict_status", "Dict status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("diff_start", "Start diff UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("diff_set", "Set diff side a|b body.", r#"{"type":"object","properties":{"side":{"type":"string"},"body":{"type":"string"}},"required":["body"]}"#, &["body"]),
        ToolDef::wasm("diff_status", "Diff status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("breakout_start", "Start breakout game UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("breakout_status", "Breakout score/lives.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("tetris_start", "Start tetris game UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("tetris_status", "Tetris score/status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("console_start", "Start console log viewer UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("console_log", "Append a console log line.", r#"{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}"#, &["msg"]),
        ToolDef::wasm("console_list", "List console log lines.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("console_status", "Console status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("maps_start", "Start maps pin UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("maps_set", "Set map lat/lon/place/zoom.", r#"{"type":"object","properties":{"lat":{"type":"integer"},"lon":{"type":"integer"},"place":{"type":"string"},"zoom":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("maps_list", "List saved map places.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("maps_status", "Maps status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("radio_start", "Start radio UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("radio_tune", "Tune station by name or index.", r#"{"type":"object","properties":{"station":{"type":"string"},"index":{"type":"integer"}}}"#, &[]),
        ToolDef::wasm("radio_status", "Radio status.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sandbox_start", "Start sandbox-lab UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sandbox_home_write", "Write a key inside the sandbox home store.", r#"{"type":"object","properties":{"key":{"type":"string"},"body":{"type":"string"}}}"#, &[]),
        ToolDef::wasm(
            "sandbox_try_escape",
            "Attempt a path outside the sandbox (must be denied for absolute paths).",
            r#"{"type":"object","properties":{"path":{"type":"string"}}}"#,
            &[],
        ),
        ToolDef::wasm("sandbox_child", "Toggle simulated attenuated child in sandbox-lab UI.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sandbox_list", "List sandbox_* storage keys.", r#"{"type":"object","properties":{}}"#, &[]),
        ToolDef::wasm("sandbox_get", "Read a sandbox storage key.", r#"{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}"#, &["key"]),
        ToolDef::wasm("sandbox_status", "Sandbox lab counters/status.", r#"{"type":"object","properties":{}}"#, &[]),
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
