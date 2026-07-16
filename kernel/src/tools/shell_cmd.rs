//! `run_shell_command` — freeform **Chitti** shell invocation for agents.
//!
//! Not POSIX `/bin/sh`. The first token is a registered system command name
//! (`ls`, `http`, `ping`, …); the remainder is the argument line. Execution
//! goes through [`crate::shell::run_tool_command`] (same path as per-command
//! Shell tools), so capability, taint, and capture semantics are unchanged.

use crate::session::todo;
use crate::tools::registry::{self, ToolBinding};
use alloc::format;
use alloc::string::{String, ToString};

/// Known destructive first-tokens (even if args look soft).
fn is_destructive_cmd(name: &str) -> bool {
    matches!(
        name,
        "rm" | "mkext4" | "install" | "umount" | "mv" | "cp"
    ) || registry::get(name)
        .map(|d| matches!(d.binding, ToolBinding::Shell { destructive: true, .. }))
        .unwrap_or(false)
}

/// Split `command` into (name, args). Accepts optional leading `/`.
pub fn parse_command_line(command: &str) -> Result<(String, String), &'static str> {
    let s = command.trim();
    if s.is_empty() {
        return Err("empty command");
    }
    let s = s.strip_prefix('/').unwrap_or(s);
    let (name, rest) = match s.split_once(char::is_whitespace) {
        Some((n, r)) => (n.trim(), r.trim()),
        None => (s, ""),
    };
    if name.is_empty() {
        return Err("empty command name");
    }
    // Only allow [A-Za-z0-9_-] names — no shell metacharacters.
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err("invalid command name (use a Chitti /command, not shell syntax)");
    }
    Ok((name.to_string(), rest.to_string()))
}

/// Run immediately (foreground). Returns tool result text.
pub fn run_foreground(command: &str) -> String {
    let (name, args) = match parse_command_line(command) {
        Ok(x) => x,
        Err(e) => return format!("error:{e}"),
    };
    if registry::get(&name).is_none() {
        // Still try dispatch_system via run_tool_command — it reports unavailable.
        return crate::shell::run_tool_command(&name, &args);
    }
    // Prefer Shell-bound tools; also allow other names that dispatch_system knows.
    crate::shell::run_tool_command(&name, &args)
}

/// Parse tool args JSON for `run_shell_command`.
pub fn run_from_tool_args(args_json: &str) -> String {
    let command = todo::json_str(args_json, "command")
        .or_else(|| todo::json_str(args_json, "cmd"))
        .unwrap_or_default();
    if command.is_empty() {
        return String::from("error: need command (e.g. \"ls /\" or \"ping 1.1.1.1\")");
    }
    let background = todo::json_str(args_json, "background")
        .map(|v| v == "true" || v == "1")
        .unwrap_or_else(|| {
            args_json.contains("\"background\":true") || args_json.contains("\"background\": true")
        });
    let (name, rest) = match parse_command_line(&command) {
        Ok(x) => x,
        Err(e) => return format!("error:{e}"),
    };
    // Optional separate args field merges after the command line rest.
    let extra = todo::json_str(args_json, "args").unwrap_or_default();
    let full_args = if extra.is_empty() {
        rest
    } else if rest.is_empty() {
        extra
    } else {
        format!("{rest} {extra}")
    };
    if background {
        if is_destructive_cmd(&name) {
            return String::from(
                "error: destructive commands cannot run in background (run foreground with approval)",
            );
        }
        let id = crate::tools::bg::spawn_shell(&name, &full_args);
        return format!(
            "ok:background task_id={id} cmd=/{name} {full_args}\n\
             Use task_output with task_id={id} to read output; kill_task to stop."
        );
    }
    let _ = is_destructive_cmd(&name); // approval is at execute_chat_tool layer
    run_foreground(&if full_args.is_empty() {
        name.clone()
    } else {
        format!("{name} {full_args}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_strips_slash_and_splits() {
        let (n, a) = parse_command_line("/ls -l /agent").unwrap();
        assert_eq!(n, "ls");
        assert_eq!(a, "-l /agent");
        let (n, a) = parse_command_line("ping 1.1.1.1").unwrap();
        assert_eq!(n, "ping");
        assert_eq!(a, "1.1.1.1");
    }

    #[test_case]
    fn parse_rejects_metachar() {
        assert!(parse_command_line("ls;rm").is_err());
        assert!(parse_command_line("").is_err());
    }
}
