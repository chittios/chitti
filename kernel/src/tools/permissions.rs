//! Optional **permission rules** (`/configs/core/permissions.json`) — allow /
//! ask / deny patterns over tool names. Applied *before* the approval-mode
//! modal; the capability gate still owns real authority.
//!
//! ```json
//! {
//!   "allow": ["datetime", "read", "list", "glob", "grep", "memory_*"],
//!   "ask":   ["write", "edit", "http", "mcp__*"],
//!   "deny":  ["install", "mkext4", "delete"]
//! }
//! ```
//!
//! Patterns: exact name, trailing `*` (prefix), or full `*`. Empty / missing
//! file = no rules (approval mode alone decides).

use crate::json::Json;
use crate::mm::Locked;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const PATH: &str = "/configs/core/permissions.json";

/// Outcome of a rule check. `None` from [`check`] means "no rule matched".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

struct Rules {
    allow: Vec<String>,
    ask: Vec<String>,
    deny: Vec<String>,
}

static RULES: Locked<Option<Rules>> = Locked::new(None);

/// Wildcard match: `*` matches anything; `foo*` matches a prefix; otherwise exact.
pub fn pattern_match(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    pattern == name
}

fn any_match(pats: &[String], name: &str) -> bool {
    pats.iter().any(|p| pattern_match(p, name))
}

/// Load (or reload) permissions from the store. Missing file clears rules.
pub fn load() {
    let rules = match crate::synapse::fs::read(PATH) {
        Some(bytes) => match core::str::from_utf8(&bytes).ok().and_then(Json::parse) {
            Some(j) => Some(Rules {
                allow: string_list(&j, "allow"),
                ask: string_list(&j, "ask"),
                deny: string_list(&j, "deny"),
            }),
            None => {
                crate::ktrace::log_fmt(format_args!("permissions: {PATH} is not valid JSON — ignoring"));
                None
            }
        },
        None => None,
    };
    let n = rules
        .as_ref()
        .map(|r| r.allow.len() + r.ask.len() + r.deny.len())
        .unwrap_or(0);
    RULES.with(|slot| *slot = rules);
    crate::ktrace::log_fmt(format_args!("permissions: loaded {n} rule(s) from {PATH}"));
}

fn string_list(j: &Json, key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(arr) = j.get(key).and_then(|v| v.as_array()) else {
        return out;
    };
    for v in arr {
        if let Some(s) = v.as_str() {
            out.push(s.to_string());
        }
    }
    out
}

/// Look up a tool name against deny → allow → ask (deny wins). `None` if no
/// rule matches (fall through to approval mode).
pub fn check(name: &str) -> Option<Decision> {
    RULES.with(|slot| {
        let Some(r) = slot.as_ref() else { return None };
        if any_match(&r.deny, name) {
            return Some(Decision::Deny);
        }
        if any_match(&r.allow, name) {
            return Some(Decision::Allow);
        }
        if any_match(&r.ask, name) {
            return Some(Decision::Ask);
        }
        None
    })
}

/// Whether any rules are loaded.
pub fn is_active() -> bool {
    RULES.with(|s| s.is_some())
}

/// Human-readable summary for `/permissions`.
pub fn summary() -> String {
    RULES.with(|slot| match slot.as_ref() {
        None => String::from("(no permissions.json — approval mode only)"),
        Some(r) => alloc::format!(
            "allow=[{}] ask=[{}] deny=[{}]",
            r.allow.join(", "),
            r.ask.join(", "),
            r.deny.join(", ")
        ),
    })
}

/// Write a starter permissions.json if absent (human can edit).
pub fn ensure_default() {
    if crate::synapse::fs::exists(PATH) {
        return;
    }
    let body = r#"{
  "allow": ["datetime", "disks", "list", "read", "glob", "grep", "memory_get", "memory_list", "memory_search", "search_tools", "skill", "todo_write"],
  "ask": ["write", "edit", "http", "mcp__*", "memory_add", "spawn_subagent"],
  "deny": ["install", "mkext4", "delete"]
}
"#;
    crate::synapse::fs::write(PATH, body.as_bytes());
}

/// True if a tool is **read-only** (safe in plan mode and concurrent batches).
pub fn is_readonly_tool(name: &str) -> bool {
    use crate::tools::registry::{self, McpResourceKind, StoreQueryKind, ToolBinding};
    // Structural names always treated as read-only discovery / planning.
    if matches!(
        name,
        "search_tools"
            | "list"
            | "read"
            | "glob"
            | "grep"
            | "search"
            | "memory_get"
            | "memory_list"
            | "memory_search"
            | "datetime"
            | "disks"
            | "ls"
            | "cat"
            | "grep"
            | "glob"
            | "pwd"
            | "mounts"
            | "help"
            | "skills"
            | "skill"
            | "load_skill"
            | "todo_write"
            | "mcp_resources"
            | "emit_result"
            | "network"
            | "enter_plan_mode"
            | "exit_plan_mode"
    ) {
        return true;
    }
    match registry::get(name).map(|d| d.binding) {
        Some(ToolBinding::Synapse { primitive, .. }) => {
            matches!(
                primitive.as_str(),
                "mem_fs_read" | "list" | "mem_fs_search" | "emit_result" | "console_write"
            )
        }
        Some(ToolBinding::Shell { destructive, .. }) => !destructive
            && matches!(
                name,
                "datetime" | "disks" | "ls" | "mounts" | "help" | "skills" | "network" | "ping"
                    | "cat" | "grep" | "glob" | "pwd" | "shortcuts"
            ),
        Some(ToolBinding::StoreQuery { kind: StoreQueryKind::Glob | StoreQueryKind::Grep }) => true,
        Some(ToolBinding::AgentMemory) => {
            matches!(name, "memory_get" | "memory_list" | "memory_search")
        }
        Some(ToolBinding::AgentStorage) => {
            matches!(name, "storage_get" | "storage_list")
        }
        // Media open/control is local UI (not network); auto-allow like list.
        Some(ToolBinding::Media) => {
            matches!(
                name,
                "draw_image"
                    | "image_control"
                    | "audio_player"
                    | "audio_control"
                    | "video_player"
                    | "video_control"
                    | "media_status"
            )
        }
        // Package WASM tools: gated by agent toolset; treat as auto when read-ish.
        Some(ToolBinding::Download) => name == "download",
        Some(ToolBinding::Browser) => {
            matches!(
                name,
                "browser_status" | "browser_links" | "browser_text" | "browser_scroll"
            )
        }
        Some(ToolBinding::AgentWasm) => {
            matches!(
                name,
                "notes_list"
                    | "notes_get"
                    | "notes_set"
                    | "notes_remove"
                    | "paint_start"
                    | "paint_clear"
                    | "paint_rect"
                    | "paint_line"
                    | "paint_pixel"
                    | "paint_draw"
                    | "paint_status"
                    | "slides_start"
                    | "slides_next"
                    | "slides_prev"
                    | "slides_goto"
                    | "slides_status"
                    | "mines_start"
                    | "mines_click"
                    | "mines_flag"
                    | "mines_status"
                    | "snake_start"
                    | "snake_dir"
                    | "snake_tick"
                    | "snake_status"
                    | "synth_tone"
                    | "synth_beep"
                    | "synth_stop"
                    | "synth_status"
            )
        }
        Some(ToolBinding::SessionTodo) | Some(ToolBinding::LoadSkill) => true,
        Some(ToolBinding::McpResources { kind: McpResourceKind::List }) => true,
        Some(ToolBinding::McpResources { kind: McpResourceKind::Read }) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn pattern_match_basics() {
        assert!(pattern_match("*", "anything"));
        assert!(pattern_match("mcp__*", "mcp__harness__echo"));
        assert!(!pattern_match("mcp__*", "write"));
        assert!(pattern_match("*_get", "memory_get"));
        assert!(pattern_match("read", "read"));
        assert!(!pattern_match("read", "write"));
    }

    #[test_case]
    fn rules_deny_allow_ask_order() {
        RULES.with(|s| {
            *s = Some(Rules {
                allow: alloc::vec![String::from("read"), String::from("memory_*")],
                ask: alloc::vec![String::from("write"), String::from("mcp__*")],
                deny: alloc::vec![String::from("install"), String::from("delete")],
            });
        });
        assert_eq!(check("install"), Some(Decision::Deny));
        assert_eq!(check("read"), Some(Decision::Allow));
        assert_eq!(check("memory_get"), Some(Decision::Allow));
        assert_eq!(check("write"), Some(Decision::Ask));
        assert_eq!(check("mcp__x__y"), Some(Decision::Ask));
        assert_eq!(check("datetime"), None); // no rule
        // deny wins over allow if both match
        RULES.with(|s| {
            *s = Some(Rules {
                allow: alloc::vec![String::from("*")],
                ask: Vec::new(),
                deny: alloc::vec![String::from("delete")],
            });
        });
        assert_eq!(check("delete"), Some(Decision::Deny));
        assert_eq!(check("read"), Some(Decision::Allow));
        RULES.with(|s| *s = None);
    }

    #[test_case]
    fn readonly_classification() {
        assert!(is_readonly_tool("read"));
        assert!(is_readonly_tool("glob"));
        assert!(is_readonly_tool("todo_write"));
        assert!(!is_readonly_tool("write"));
        assert!(!is_readonly_tool("delete"));
        assert!(!is_readonly_tool("install"));
    }
}
