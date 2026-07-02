//! The session todo list + `todo_write` semantics
//! (`CHITTI_AGENTIC_HANDOFF.md` Phase A/D). `todo_write` replaces the whole
//! list from a structured payload — the same shape Claude Code's TodoWrite tool
//! uses — so the orchestrator can plan and work down a multi-step task.

use crate::agent::types::*;
use alloc::string::String;
use alloc::vec::Vec;

/// One entry in a `todo_write` payload.
pub struct TodoInput {
    pub id: u32,
    pub text: String,
    pub status: TodoStatus,
}

/// Replace the session's todo list wholesale (idempotent, matches TodoWrite).
/// Returns the number of items now pending/in-progress (the remaining work).
pub fn write(session: &mut Session, items: Vec<TodoInput>, now: Ticks) -> usize {
    session.todos = items
        .into_iter()
        .map(|t| Todo { id: t.id, text: t.text, status: t.status, created_ticks: now })
        .collect();
    session.updated_ticks = now;
    let remaining = session
        .todos
        .iter()
        .filter(|t| matches!(t.status, TodoStatus::Pending | TodoStatus::InProgress))
        .count();
    crate::ktrace::log_fmt(format_args!(
        "session.todo_write: {} items, {} remaining",
        session.todos.len(),
        remaining
    ));
    remaining
}

/// Parse a `todo_write` tool's JSON args into `TodoInput`s. Expects
/// `{"todos":[{"id":1,"text":"...","status":"pending"},...]}`. Tolerant of
/// whitespace; unknown status strings default to `Pending`.
pub fn parse_args(json: &str) -> Vec<TodoInput> {
    let mut out = Vec::new();
    // Walk each object between braces after the "todos" array marker.
    let body = match json.find("\"todos\"").and_then(|i| json[i..].find('[')).map(|j| j) {
        Some(_) => json,
        None => return out,
    };
    let mut rest = &body[body.find('[').map(|i| i + 1).unwrap_or(0)..];
    while let Some(open) = rest.find('{') {
        let close = match rest[open..].find('}') {
            Some(c) => open + c,
            None => break,
        };
        let obj = &rest[open..=close];
        let id = json_u32(obj, "id").unwrap_or((out.len() + 1) as u32);
        let text = json_str(obj, "text").unwrap_or_default();
        let status = match json_str(obj, "status").as_deref() {
            Some("in_progress") => TodoStatus::InProgress,
            Some("done") => TodoStatus::Done,
            Some("cancelled") => TodoStatus::Cancelled,
            _ => TodoStatus::Pending,
        };
        out.push(TodoInput { id, text, status });
        rest = &rest[close + 1..];
    }
    out
}

/// Minimal `"key":"value"` string extractor for flat JSON objects. Handles
/// `\"`, `\\`, `\n`, `\t` escapes. Sufficient for the controlled tool-args
/// shapes the loop emits; full JSON validation lives in the tools registry.
pub fn json_str(json: &str, key: &str) -> Option<String> {
    let pat = alloc::format!("\"{key}\"");
    let start = json.find(&pat)? + pat.len();
    let after = json[start..].find(':')? + start + 1;
    let bytes = json.as_bytes();
    let mut i = after;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;
    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                i += 1;
                out.push(match bytes[i] {
                    b'n' => '\n',
                    b't' => '\t',
                    b'"' => '"',
                    b'\\' => '\\',
                    other => other as char,
                });
            }
            b'"' => break,
            b => out.push(b as char),
        }
        i += 1;
    }
    Some(out)
}

/// Extract an unsigned integer field `"key": <n>`.
pub fn json_u32(json: &str, key: &str) -> Option<u32> {
    let pat = alloc::format!("\"{key}\"");
    let start = json.find(&pat)? + pat.len();
    let after = json[start..].find(':')? + start + 1;
    let tail = json[after..].trim_start();
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}
