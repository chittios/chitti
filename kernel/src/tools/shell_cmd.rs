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
/// Public so the Router can taint-gate `run_shell_command` the same way as
/// per-command Shell bindings.
pub fn is_destructive_cmd(name: &str) -> bool {
    matches!(
        name,
        "rm" | "mkext4" | "install" | "umount" | "mv" | "cp"
    ) || registry::get(name)
        .map(|d| matches!(d.binding, ToolBinding::Shell { destructive: true, .. }))
        .unwrap_or(false)
}

/// Every stage of a shell line as `(name, args)`.
///
/// **A pipeline is as effectful as its worst stage**, so callers that gate on
/// effect must see all of them. `ls / | rm /x` reads as a harmless `ls` if only
/// the first token is examined, which is exactly the smuggling shape the
/// `channel send` and `schedule add` classifications already had to learn.
///
/// An empty result means the line could not be read at all — callers must treat
/// that as effectful, not as inert.
pub fn stages(command: &str) -> alloc::vec::Vec<(String, String)> {
    let body = command.trim();
    let body = body.strip_prefix('/').unwrap_or(body);
    if crate::shell::pipeline::has_operator(body) {
        return match crate::shell::pipeline::parse(body) {
            Ok(script) => script.stages().map(|s| (s.name.clone(), s.arg.clone())).collect(),
            Err(_) => alloc::vec::Vec::new(),
        };
    }
    match parse_command_line(command) {
        Ok(pair) => alloc::vec![pair],
        Err(_) => alloc::vec::Vec::new(),
    }
}

/// Whether **any** stage of a shell line is destructive.
pub fn line_is_destructive(command: &str) -> bool {
    let st = stages(command);
    // Unreadable is not harmless.
    st.is_empty() || st.iter().any(|(n, _)| is_destructive_cmd(n))
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

/// Run a composed line (`|`, `&&`, `>`, …) and return what it printed.
///
/// **`enter_tool_call` is set explicitly here, and that is load-bearing.** The
/// pipeline runner calls `dispatch_system` directly rather than going through
/// `run_tool_command`, so without this the stages would run with `in_tool_call`
/// at zero — indistinguishable from a human typing at the console, which is the
/// one signal the human-only refusals rely on. Getting this wrong would be a
/// privilege escalation dressed as a refactor.
fn run_pipeline_line(command: &str) -> String {
    let body = command.trim();
    let body = body.strip_prefix('/').unwrap_or(body);
    let script = match crate::shell::pipeline::parse(body) {
        Ok(s) => s,
        Err(e) => return format!("error:{e}"),
    };
    crate::shell::enter_tool_call();
    crate::serial::capture_begin();
    crate::shell::pipeline::run(&script);
    let out = crate::serial::capture_end();
    crate::shell::leave_tool_call();
    out
}

/// Run immediately (foreground). Returns tool result text.
pub fn run_foreground(command: &str) -> String {
    if crate::shell::pipeline::has_operator(command) {
        return run_pipeline_line(command);
    }
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
    if crate::shell::pipeline::has_operator(&command) {
        if background {
            // A background job is stored as one (name, args) pair and replayed
            // through `run_tool_command`; a pipeline does not fit that shape,
            // and pretending it did would run only the first stage.
            return String::from("error: background jobs cannot be pipelines — run it in the foreground, or background a single command");
        }
        return run_pipeline_line(&command);
    }
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

    #[test_case]
    fn destructive_cmd_names() {
        assert!(is_destructive_cmd("rm"));
        assert!(is_destructive_cmd("install"));
        assert!(is_destructive_cmd("mkext4"));
        assert!(!is_destructive_cmd("ls"));
        assert!(!is_destructive_cmd("ping"));
    }
}
