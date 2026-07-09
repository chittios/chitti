//! **Per-agent home directories** on the filesystem: every agent gets
//! `/agent/<id>/` holding its **SOUL.md** (persona — prepended to the agent's
//! system prompt), a **skills/** area (agent-local skill state), and a
//! **memory/** area (durable notes the agent reads/writes with its fs tools
//! or the dedicated `memory_add` / `memory_get` / `memory_list` tool calls).
//! The store is flat (path → bytes), so the "directories" are path prefixes
//! seeded with a `.keep` marker; on an installed system these persist on ext4
//! like the rest of `synapse::fs`.

use crate::synapse::fs;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The home path for agent `id` (no trailing slash).
pub fn path(id: u64) -> String {
    format!("/agent/{}", id)
}

/// Ensure agent `id`'s home exists: `SOUL.md` (seeded with a default persona on
/// first boot), `skills/` and `memory/` markers. Idempotent; cheap when present.
/// A SOUL.md placed by an installed package (see `skills::package::place_agent_home`)
/// already exists here, so `ensure` never overwrites a packaged persona.
pub fn ensure(id: u64, name: &str) {
    let home = path(id);
    let soul = format!("{}/SOUL.md", home);
    if !fs::exists(&soul) {
        // A concise *persona* the model adopts — not documentation about the
        // file (a chatty meta description gets parroted back as an answer).
        let default = format!(
            "You are {name}, the shell agent of Chitti OS. You are concise, direct, and \
             practical. You operate the machine through tools and answer in plain prose."
        );
        fs::write(&soul, default.as_bytes());
        crate::ktrace::log_fmt(format_args!("agent.home: created {} (SOUL.md + skills/ + memory/)", home));
    }
    for marker in [format!("{}/skills/.keep", home), format!("{}/memory/.keep", home)] {
        if !fs::exists(&marker) {
            fs::write(&marker, b"");
        }
    }
}

/// Agent `id`'s SOUL.md contents, if present (lossy UTF-8).
pub fn soul(id: u64) -> Option<String> {
    fs::read(&format!("{}/SOUL.md", path(id))).map(|b| String::from_utf8_lossy(&b).into_owned())
}

// --- durable agent memory (tool-call surface) -----------------------------
//
// Keys land at `/agent/<id>/memory/<key>`. Keys are sanitised so a model cannot
// path-escape the memory folder (no `/`, `..`, or control bytes).

/// Sanitize a memory key: allow `[A-Za-z0-9._-]`, max 64 chars. Rejects empty
/// / traversal / anything that would leave the agent's memory prefix.
pub fn sanitize_memory_key(key: &str) -> Option<String> {
    let k = key.trim();
    if k.is_empty() || k.len() > 64 {
        return None;
    }
    if k == "." || k == ".." || k.starts_with('.') && k == ".keep" {
        return None;
    }
    if !k.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
        return None;
    }
    Some(k.to_string())
}

/// Absolute store path for agent `id`'s memory fact `key` (after sanitise).
pub fn memory_path(id: u64, key: &str) -> Option<String> {
    let k = sanitize_memory_key(key)?;
    Some(format!("{}/memory/{}", path(id), k))
}

/// Store `key = value` in agent `id`'s durable memory. Ensures the home exists.
/// Returns an error string on a bad key.
pub fn memory_add(id: u64, key: &str, value: &str) -> Result<(), &'static str> {
    let Some(p) = memory_path(id, key) else {
        return Err("invalid memory key (use [A-Za-z0-9._-], max 64 chars)");
    };
    // Seed the home if this agent never wrote before (e.g. first tool call).
    if !fs::exists(&format!("{}/memory/.keep", path(id))) {
        ensure(id, "agent");
    }
    fs::write(&p, value.as_bytes());
    crate::ktrace::log_fmt(format_args!("agent.memory: agent {} stored '{}'", id, key.trim()));
    Ok(())
}

/// Read a durable fact for agent `id`. `None` if the key is invalid or absent.
pub fn memory_get(id: u64, key: &str) -> Option<String> {
    let p = memory_path(id, key)?;
    fs::read(&p).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// List memory keys stored for agent `id` (sorted, no `.keep` marker).
pub fn memory_list(id: u64) -> Vec<String> {
    let prefix = format!("{}/memory/", path(id));
    let mut keys: Vec<String> = fs::list()
        .into_iter()
        .filter_map(|p| {
            let rest = p.strip_prefix(&prefix)?;
            if rest.is_empty() || rest == ".keep" || rest.contains('/') {
                return None;
            }
            Some(rest.to_string())
        })
        .collect();
    keys.sort();
    keys
}

/// Run a `memory_add` / `memory_get` / `memory_list` tool for `agent_id`.
/// Accepts either:
/// * chat-flattened args — `key`, `key\\x1fvalue`, or `key value…`
/// * full JSON object args from the Router — `{"key":"…","value":"…"}`
///
/// Returns a human-readable result; failures start with `error:`.
pub fn run_memory_tool(name: &str, agent_id: u64, args: &str) -> String {
    // Prefer JSON fields when the Router (or a well-formed model call) sent them.
    let json_key = crate::session::todo::json_str(args, "key");
    let json_val = crate::session::todo::json_str(args, "value");
    let (key, value) = if let Some(k) = json_key {
        (k, json_val.unwrap_or_default())
    } else if let Some((k, v)) = args.split_once('\u{1f}') {
        (String::from(k), String::from(v))
    } else if let Some((k, v)) = args.split_once(char::is_whitespace) {
        (String::from(k), String::from(v.trim_start()))
    } else {
        (String::from(args.trim()), String::new())
    };
    match name {
        "memory_add" | "remember" => {
            if key.is_empty() {
                return String::from("error: memory_add needs a key (and a value)");
            }
            if value.is_empty() {
                return String::from("error: memory_add needs a value");
            }
            match memory_add(agent_id, &key, &value) {
                Ok(()) => format!("ok: stored '{key}' ({} bytes)", value.len()),
                Err(e) => format!("error: {e}"),
            }
        }
        "memory_get" | "recall" => {
            if key.is_empty() {
                return String::from("error: memory_get needs a key");
            }
            match memory_get(agent_id, &key) {
                Some(v) => v,
                None => format!("(no memory for key '{key}')"),
            }
        }
        "memory_list" => {
            let keys = memory_list(agent_id);
            if keys.is_empty() {
                String::from("(no memories stored)")
            } else {
                keys.join("\n")
            }
        }
        other => format!("error: unknown memory tool '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn memory_key_sanitise_rejects_traversal() {
        assert!(sanitize_memory_key("ok_key-1").is_some());
        assert!(sanitize_memory_key("note.txt").is_some());
        assert!(sanitize_memory_key("").is_none());
        assert!(sanitize_memory_key("../etc").is_none());
        assert!(sanitize_memory_key("a/b").is_none());
        assert!(sanitize_memory_key("has space").is_none());
        assert!(sanitize_memory_key(&"x".repeat(65)).is_none());
    }

    #[test_case]
    fn memory_add_get_list_roundtrip() {
        let id = 42_u64;
        ensure(id, "mem_test");
        assert!(memory_add(id, "colour", "teal").is_ok());
        assert_eq!(memory_get(id, "colour").as_deref(), Some("teal"));
        assert!(memory_list(id).iter().any(|k| k == "colour"));
        assert!(memory_add(id, "../escape", "nope").is_err());
        assert!(memory_get(id, "missing").is_none());
    }

    #[test_case]
    fn memory_tool_json_and_flat_args() {
        let id = 77_u64;
        let out = run_memory_tool("memory_add", id, r#"{"key":"fav","value":"orange"}"#);
        assert!(out.starts_with("ok:"), "json add: {out}");
        assert_eq!(run_memory_tool("memory_get", id, r#"{"key":"fav"}"#), "orange");
        // Chat-flattened form uses unit-separator between key and value.
        let out = run_memory_tool("memory_add", id, "city\u{1f}Chennai");
        assert!(out.starts_with("ok:"), "flat add: {out}");
        assert_eq!(run_memory_tool("memory_get", id, "city"), "Chennai");
        // Space-separated flat form (human `/memory add` and shell args).
        let out = run_memory_tool("memory_add", id, "greeting hello world");
        assert!(out.starts_with("ok:"), "space add: {out}");
        assert_eq!(run_memory_tool("memory_get", id, "greeting"), "hello world");
        let list = run_memory_tool("memory_list", id, "");
        assert!(list.contains("fav") && list.contains("city") && list.contains("greeting"), "list: {list}");
    }

    #[test_case]
    fn memory_tool_errors_and_miss() {
        let id = 88_u64;
        assert!(run_memory_tool("memory_add", id, "").starts_with("error:"));
        assert!(run_memory_tool("memory_add", id, "onlykey").starts_with("error:"));
        assert!(run_memory_tool("memory_add", id, r#"{"key":"x"}"#).starts_with("error:"));
        assert!(run_memory_tool("memory_get", id, "").starts_with("error:"));
        assert!(run_memory_tool("memory_get", id, "nope").contains("no memory"));
        // Bad keys never write outside the agent's memory prefix.
        let bad = run_memory_tool("memory_add", id, "../escape secret");
        assert!(bad.starts_with("error:"), "traversal must fail: {bad}");
        assert!(memory_get(id, "../escape").is_none());
        assert!(run_memory_tool("memory_list", id, "").contains("no memories")
            || !run_memory_tool("memory_list", id, "").contains("escape"));
    }

    #[test_case]
    fn memory_overwrite_and_isolation() {
        let a = 91_u64;
        let b = 92_u64;
        assert!(memory_add(a, "shared_name", "from-a").is_ok());
        assert!(memory_add(b, "shared_name", "from-b").is_ok());
        assert_eq!(memory_get(a, "shared_name").as_deref(), Some("from-a"));
        assert_eq!(memory_get(b, "shared_name").as_deref(), Some("from-b"));
        // Overwrite in place.
        assert!(memory_add(a, "shared_name", "updated").is_ok());
        assert_eq!(memory_get(a, "shared_name").as_deref(), Some("updated"));
        assert_eq!(memory_get(b, "shared_name").as_deref(), Some("from-b"));
    }

    #[test_case]
    fn memory_path_is_under_agent_home() {
        let p = memory_path(7, "note").expect("valid key");
        assert_eq!(p, "/agent/7/memory/note");
        assert!(memory_path(7, "a/b").is_none());
        assert!(memory_path(7, "").is_none());
    }
}
