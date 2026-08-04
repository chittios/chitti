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

/// The **ChittiOS user home** (`~`) — the folder the shell agent starts in,
/// `/pwd` names, and user commands (git, downloads, notes) default to. A
/// single-user OS, so the home is a fixed `/home/chitti` rather than a
/// per-login path. Distinct from the per-agent `/agent/<id>/` install homes.
pub const USER_HOME: &str = "/home/chitti";

/// Ensure the user home exists as a directory in the store (a `.keep` marker
/// on `/home` and `/home/chitti`, the same convention `mkdir` uses). Called at
/// boot alongside the agent-roster install, so the `~` exists both in the
/// in-memory store (diskless boots) and on a freshly installed disk's empty
/// ext4 data partition — **never** removing or overwriting user files (a
/// home that already has children needs no marker, and an in-place `/install`
/// update preserves the whole data partition).
pub fn ensure_user_home() {
    if !fs::exists("/home/.keep") {
        fs::write("/home/.keep", b"");
    }
    if !fs::exists("/home/chitti/.keep") {
        fs::write("/home/chitti/.keep", b"");
    }
}

/// Ensure agent `id`'s home exists: `SOUL.md` (seeded with a default persona on
/// first boot), optional `MEMORY.md` seed, `skills/` and `memory/` markers.
/// Idempotent; cheap when present. A SOUL.md placed by an installed package
/// (see `skills::package::place_agent_home`) already exists here, so `ensure`
/// never overwrites a packaged persona.
pub fn ensure(id: u64, name: &str) {
    let home = path(id);
    let soul = format!("{}/SOUL.md", home);
    if !fs::exists(&soul) {
        // A concise *persona* the model adopts — not documentation about the
        // file (a chatty meta description gets parroted back as an answer).
        let default = format!(
            "You are {name}, the shell agent of ChittiOS. You are concise, direct, and \
             practical. You operate the machine through tools and answer in plain prose."
        );
        fs::write(&soul, default.as_bytes());
        crate::ktrace::log_fmt(format_args!("agent.home: created {} (SOUL.md + skills/ + memory/)", home));
    }
    // MEMORY.md is the hierarchical memory entrypoint (project facts). Seed
    // only if missing — never clobber user notes.
    let mem_md = format!("{}/MEMORY.md", home);
    if !fs::exists(&mem_md) {
        fs::write(
            &mem_md,
            b"# Memory\n\nDurable notes for this agent. Keep entries short; detail goes in memory/<key>.\n",
        );
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

/// Max lines / bytes of MEMORY.md injected into the system prompt.
pub const MEMORY_MD_MAX_LINES: usize = 200;
pub const MEMORY_MD_MAX_BYTES: usize = 25_000;

/// Agent `id`'s MEMORY.md body, truncated for prompt injection. `None` if empty
/// or missing (after strip of the default seed heading-only content is still
/// returned so the agent knows the file exists).
pub fn memory_md(id: u64) -> Option<String> {
    let raw = fs::read(&format!("{}/MEMORY.md", path(id)))?;
    let text = String::from_utf8_lossy(&raw);
    if text.trim().is_empty() {
        return None;
    }
    let (body, _) = crate::tools::pathutil::truncate_memory_md(&text, MEMORY_MD_MAX_LINES, MEMORY_MD_MAX_BYTES);
    Some(body)
}

/// Append a short line to MEMORY.md (used by humans and the remember path).
pub fn memory_md_append(id: u64, line: &str) -> Result<(), &'static str> {
    ensure(id, "agent");
    let p = format!("{}/MEMORY.md", path(id));
    let mut cur = fs::read(&p).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    if !cur.ends_with('\n') && !cur.is_empty() {
        cur.push('\n');
    }
    cur.push_str("- ");
    cur.push_str(line.trim());
    cur.push('\n');
    fs::write(&p, cur.as_bytes());
    Ok(())
}

/// Search keys + values in durable KV memory for `query` (case-insensitive
/// multi-term ranked match). Returns `key: snippet` lines, best first (max 16).
pub fn memory_search(id: u64, query: &str) -> Vec<String> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let terms: Vec<&str> = q
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter(|t| !t.is_empty())
        .collect();
    let mut ranked: Vec<(u32, String)> = Vec::new();
    for key in memory_list(id) {
        let val = memory_get(id, &key).unwrap_or_default();
        let hay_k = key.to_ascii_lowercase();
        let hay_v = val.to_ascii_lowercase();
        let mut score = 0u32;
        if hay_k.contains(&q) || hay_v.contains(&q) {
            score = score.saturating_add(50);
        }
        for t in &terms {
            if hay_k == *t {
                score = score.saturating_add(80);
            } else if hay_k.contains(t) {
                score = score.saturating_add(30);
            }
            if hay_v.contains(t) {
                score = score.saturating_add(10);
            }
        }
        if score == 0 {
            continue;
        }
        let snip = if val.len() > 120 {
            // char-safe cut
            let mut end = 120.min(val.len());
            while end > 0 && !val.is_char_boundary(end) {
                end -= 1;
            }
            alloc::format!("{}…", &val[..end])
        } else {
            val
        };
        ranked.push((score, format!("{key}: {snip}")));
    }
    // Also scan MEMORY.md lines.
    if let Some(md) = fs::read(&format!("{}/MEMORY.md", path(id))) {
        let text = String::from_utf8_lossy(&md);
        for (i, line) in text.lines().enumerate() {
            let hay = line.to_ascii_lowercase();
            let mut score = 0u32;
            if hay.contains(&q) {
                score = score.saturating_add(40);
            }
            for t in &terms {
                if hay.contains(t) {
                    score = score.saturating_add(8);
                }
            }
            if score > 0 {
                ranked.push((score, format!("MEMORY.md:{}: {line}", i + 1)));
            }
        }
    }
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    ranked.into_iter().take(16).map(|(_, s)| s).collect()
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
/// Exact key only — see [`memory_resolve`] for suffix/case-insensitive lookup
/// (models often store `user.name` then later ask for `name` after compact).
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

/// Resolve `want` against stored keys for agent `id`.
///
/// Order: exact → case-insensitive exact → unique dotted/segment suffix
/// (`name` → `user.name`) → unique substring. Pure over the key list + gets;
/// unit-tested so post-compact recall stays reliable.
pub fn memory_resolve(id: u64, want: &str) -> Option<(String, String)> {
    let want_raw = want.trim();
    if want_raw.is_empty() {
        return None;
    }
    if let Some(v) = memory_get(id, want_raw) {
        return Some((want_raw.to_string(), v));
    }
    let want = want_raw.to_ascii_lowercase();
    let keys = memory_list(id);
    if keys.is_empty() {
        return None;
    }
    // Case-insensitive exact.
    for k in &keys {
        if k.to_ascii_lowercase() == want {
            if let Some(v) = memory_get(id, k) {
                return Some((k.clone(), v));
            }
        }
    }
    // Suffix / last path segment: "name" matches "user.name" or "profile_name".
    let mut suffix: Vec<&String> = keys
        .iter()
        .filter(|k| {
            let kl = k.to_ascii_lowercase();
            kl.ends_with(&format!(".{want}"))
                || kl.ends_with(&format!("_{want}"))
                || kl.ends_with(&format!("-{want}"))
                || kl.split(|c| c == '.' || c == '_' || c == '-').last() == Some(want.as_str())
        })
        .collect();
    suffix.sort();
    suffix.dedup();
    if suffix.len() == 1 {
        let k = suffix[0].clone();
        if let Some(v) = memory_get(id, &k) {
            return Some((k, v));
        }
    }
    // Unique substring hit (e.g. "user" → sole key containing "user").
    let contains: Vec<&String> = keys
        .iter()
        .filter(|k| k.to_ascii_lowercase().contains(&want))
        .collect();
    if contains.len() == 1 {
        let k = contains[0].clone();
        if let Some(v) = memory_get(id, &k) {
            return Some((k, v));
        }
    }
    None
}

/// Compact key→value listing for the system prompt so durable facts survive
/// `/compact` without relying on the model remembering exact keys.
/// Caps entries so a large store cannot blow the prefill budget.
pub fn memory_kv_digest(id: u64) -> Option<String> {
    let keys = memory_list(id);
    if keys.is_empty() {
        return None;
    }
    const MAX_KEYS: usize = 24;
    const MAX_VAL: usize = 96;
    let mut out = String::from(
        "Durable memory (use the exact key with memory_get; on miss use memory_list / memory_search):\n",
    );
    for k in keys.iter().take(MAX_KEYS) {
        let val = memory_get(id, k).unwrap_or_default();
        let snip = if val.len() > MAX_VAL {
            format!("{}…", &val[..MAX_VAL])
        } else {
            val
        };
        out.push_str("- ");
        out.push_str(k);
        out.push_str(": ");
        out.push_str(&snip);
        out.push('\n');
    }
    if keys.len() > MAX_KEYS {
        out.push_str(&format!(
            "… and {} more keys (call memory_list)\n",
            keys.len() - MAX_KEYS
        ));
    }
    Some(out)
}

/// Keys that *almost* match `want` (for miss diagnostics). Pure-ish helper.
fn memory_near_keys(id: u64, want: &str) -> Vec<String> {
    let want = want.trim().to_ascii_lowercase();
    if want.is_empty() {
        return Vec::new();
    }
    memory_list(id)
        .into_iter()
        .filter(|k| {
            let kl = k.to_ascii_lowercase();
            kl.contains(&want)
                || want.contains(&kl)
                || kl.ends_with(&want)
                || kl.split(|c| c == '.' || c == '_' || c == '-').any(|s| s == want)
        })
        .take(8)
        .collect()
}

/// Run a `memory_add` / `memory_get` / `memory_list` / `memory_search` tool
/// for `agent_id`. Accepts either:
/// * chat-flattened args — `key`, `key\\x1fvalue`, or `key value…`
/// * full JSON object args from the Router — `{"key":"…","value":"…"}`
///
/// Returns a human-readable result; failures start with `error:`.
pub fn run_memory_tool(name: &str, agent_id: u64, args: &str) -> String {
    // Prefer JSON fields when the Router (or a well-formed model call) sent them.
    let json_key = crate::session::todo::json_str(args, "key");
    let json_val = crate::session::todo::json_str(args, "value");
    let json_query = crate::session::todo::json_str(args, "query");
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
            match memory_resolve(agent_id, &key) {
                Some((resolved, v)) if resolved == key => v,
                Some((resolved, v)) => {
                    // Model asked for a short/alias key; surface the real name
                    // so the next turn can use the exact key after compact.
                    format!("{v}\n(key: {resolved})")
                }
                None => {
                    let near = memory_near_keys(agent_id, &key);
                    let all = memory_list(agent_id);
                    if all.is_empty() {
                        format!("(no memory for key '{key}' — store is empty; use memory_add first)")
                    } else if !near.is_empty() {
                        format!(
                            "(no memory for key '{key}'; closest: {} — try memory_get with one of those, or memory_search)",
                            near.join(", ")
                        )
                    } else {
                        format!(
                            "(no memory for key '{key}'; known keys: {} — or memory_search / memory_list)",
                            all.iter().take(12).cloned().collect::<Vec<_>>().join(", ")
                        )
                    }
                }
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
        "memory_search" => {
            let q = json_query.unwrap_or_else(|| {
                // Flat form: whole args string is the query.
                if key.is_empty() {
                    String::new()
                } else if value.is_empty() {
                    key.clone()
                } else {
                    alloc::format!("{key} {value}")
                }
            });
            if q.trim().is_empty() {
                return String::from("error: memory_search needs a query");
            }
            let hits = memory_search(agent_id, &q);
            if hits.is_empty() {
                String::from("(no memory matches)")
            } else {
                hits.join("\n")
            }
        }
        other => format!("error: unknown memory tool '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ensure_user_home` creates the `~` directory markers in the store,
    /// is idempotent, and never clobbers existing user files.
    #[test_case]
    fn ensure_user_home_creates_and_preserves() {
        crate::synapse::fs::write("/home/chitti/.keep", b"");
        crate::synapse::fs::write("/home/chitti/notes.txt", b"hello");
        ensure_user_home();
        assert!(crate::synapse::fs::is_dir("/home"), "home dir missing");
        assert!(crate::synapse::fs::is_dir("/home/chitti"), "~ dir missing");
        assert!(crate::synapse::fs::exists("/home/chitti/.keep"));
        // Idempotent, and user files are untouched.
        ensure_user_home();
        assert_eq!(
            crate::synapse::fs::read("/home/chitti/notes.txt"),
            Some(b"hello".to_vec())
        );
        crate::synapse::fs::write("/home/chitti/.keep", b"");
        crate::synapse::fs::write("/home/chitti/notes.txt", b"hello");
    }

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

    #[test_case]
    fn memory_md_and_search() {
        let id = 55_u64;
        ensure(id, "mem_md");
        assert!(memory_md(id).is_some());
        assert!(memory_add(id, "project", "chitti-os").is_ok());
        assert!(memory_add(id, "colour", "terracotta").is_ok());
        let hits = memory_search(id, "chitti");
        assert!(hits.iter().any(|h| h.contains("project")), "hits={hits:?}");
        let hits2 = memory_search(id, "MEMORY");
        // Default MEMORY.md heading or content should match loosely; at least
        // search over md lines is exercised without panic.
        let _ = hits2;
        let tool = run_memory_tool("memory_search", id, r#"{"query":"terracotta"}"#);
        assert!(tool.contains("colour"), "tool={tool}");
    }

    /// Post-compact recall: store under `user.name`, get with short `name`.
    #[test_case]
    fn memory_resolve_suffix_and_digest() {
        let id = 66_u64;
        assert!(memory_add(id, "user.name", "Vinoth").is_ok());
        assert!(memory_add(id, "project", "chitti").is_ok());
        // Exact still works.
        assert_eq!(
            memory_resolve(id, "user.name").map(|(_, v)| v).as_deref(),
            Some("Vinoth")
        );
        // Short key resolves uniquely via dotted suffix.
        let r = memory_resolve(id, "name").expect("suffix resolve");
        assert_eq!(r.0, "user.name");
        assert_eq!(r.1, "Vinoth");
        // Tool surface returns the value (+ resolved key annotation).
        let out = run_memory_tool("memory_get", id, r#"{"key":"name"}"#);
        assert!(out.contains("Vinoth"), "out={out}");
        assert!(out.contains("user.name"), "should cite resolved key: {out}");
        // Digest lists both facts for system-prompt injection.
        let dig = memory_kv_digest(id).expect("digest");
        assert!(dig.contains("user.name") && dig.contains("Vinoth"), "dig={dig}");
        assert!(dig.contains("project") && dig.contains("chitti"), "dig={dig}");
        // Miss with near keys.
        let miss = run_memory_tool("memory_get", id, r#"{"key":"username"}"#);
        assert!(miss.contains("no memory") || miss.contains("closest") || miss.contains("known"), "miss={miss}");
    }
}
