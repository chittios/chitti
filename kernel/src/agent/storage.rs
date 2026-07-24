//! **Agent storage** — browser-like localStorage for installable agents.
//!
//! Two scopes, both key-sandboxed to one agent:
//!
//! * **session** — in-RAM map, wiped when the agent stops / process reboots
//! * **durable** — `/agent/<id>/storage/<key>` on the Synapse store (persists
//!   on ext4 like memory/)
//!
//! This is the host-side of the future WASM import ABI (`storage_get` /
//! `storage_set` / …). Kernel UI agents use it today so chess FEN and other
//! state is not special-cased through `memory_*` helpers.
//!
//! Security: keys use the same sanitizer as durable memory (`[A-Za-z0-9._-]`,
//! max 64). Values are size-capped. No path traversal out of the agent home.

use crate::agent::home::{self, sanitize_memory_key};
use crate::mm::Locked;
use crate::synapse::fs;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Max value size per key (bytes).
pub const MAX_VALUE_BYTES: usize = 64 * 1024;
/// Max total bytes of session storage per agent.
pub const MAX_SESSION_TOTAL: usize = 1024 * 1024;
/// Max total bytes of durable storage per agent (sum of value sizes).
pub const MAX_DURABLE_TOTAL: usize = 4 * 1024 * 1024;
/// Max number of durable keys per agent.
pub const MAX_DURABLE_KEYS: usize = 256;

/// Storage scope: ephemeral session vs durable home folder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    Session,
    Durable,
}

impl Scope {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "" | "session" | "s" => Some(Scope::Session),
            "durable" | "d" | "persistent" => Some(Scope::Durable),
            _ => None,
        }
    }
}

type SessionMap = BTreeMap<(u64, String), Vec<u8>>;

static SESSION: Locked<SessionMap> = Locked::new(BTreeMap::new());

fn storage_path(id: u64, key: &str) -> Option<String> {
    let k = sanitize_memory_key(key)?;
    Some(format!("{}/storage/{}", home::path(id), k))
}

fn ensure_storage_dir(id: u64) {
    home::ensure(id, "agent");
    let marker = format!("{}/storage/.keep", home::path(id));
    if !fs::exists(&marker) {
        fs::write(&marker, b"");
    }
}

fn session_total(id: u64, map: &SessionMap) -> usize {
    map.iter()
        .filter(|((aid, _), _)| *aid == id)
        .map(|(_, v)| v.len())
        .sum()
}

/// `storage_set`: write `value` under `key` for agent `id`.
pub fn set(id: u64, scope: Scope, key: &str, value: &[u8]) -> Result<(), &'static str> {
    let k = sanitize_memory_key(key).ok_or("invalid storage key (use [A-Za-z0-9._-], max 64)")?;
    if value.len() > MAX_VALUE_BYTES {
        return Err("value too large (max 64 KiB per key)");
    }
    match scope {
        Scope::Session => SESSION.with(|m| {
            let prev = m.get(&(id, k.clone())).map(|v| v.len()).unwrap_or(0);
            let total = session_total(id, m) - prev + value.len();
            if total > MAX_SESSION_TOTAL {
                return Err("session storage full (max 1 MiB per agent)");
            }
            m.insert((id, k), value.to_vec());
            Ok(())
        }),
        Scope::Durable => {
            ensure_storage_dir(id);
            let p = storage_path(id, &k).ok_or("invalid storage key")?;
            let prev = fs::size_of(&p).unwrap_or(0);
            let keys = list(id, Scope::Durable);
            let is_new = !keys.iter().any(|x| x == &k);
            if is_new && keys.len() >= MAX_DURABLE_KEYS {
                return Err("durable storage key limit (max 256 keys per agent)");
            }
            let total = durable_total(id).saturating_sub(prev).saturating_add(value.len());
            if total > MAX_DURABLE_TOTAL {
                return Err("durable storage full (max 4 MiB per agent)");
            }
            fs::write(&p, value);
            Ok(())
        }
    }
}

/// Sum of durable value sizes for `id` (keys under `/agent/<id>/storage/`).
fn durable_total(id: u64) -> usize {
    let prefix = format!("{}/storage/", home::path(id));
    fs::list()
        .into_iter()
        .filter(|p| p.starts_with(&prefix) && !p.ends_with("/.keep") && !p.ends_with(".keep"))
        .filter_map(|p| fs::size_of(&p))
        .sum()
}

/// `storage_get`: read value, or `None` if missing.
pub fn get(id: u64, scope: Scope, key: &str) -> Option<Vec<u8>> {
    let k = sanitize_memory_key(key)?;
    match scope {
        Scope::Session => SESSION.with(|m| m.get(&(id, k)).cloned()),
        Scope::Durable => {
            let p = storage_path(id, &k)?;
            fs::read(&p)
        }
    }
}

/// Convenience: UTF-8 lossy get.
pub fn get_str(id: u64, scope: Scope, key: &str) -> Option<String> {
    get(id, scope, key).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Convenience: UTF-8 set.
pub fn set_str(id: u64, scope: Scope, key: &str, value: &str) -> Result<(), &'static str> {
    set(id, scope, key, value.as_bytes())
}

/// Remove one key.
pub fn remove(id: u64, scope: Scope, key: &str) -> Result<bool, &'static str> {
    let k = sanitize_memory_key(key).ok_or("invalid storage key")?;
    match scope {
        Scope::Session => SESSION.with(|m| Ok(m.remove(&(id, k)).is_some())),
        Scope::Durable => {
            let p = storage_path(id, &k).ok_or("invalid storage key")?;
            Ok(fs::delete(&p))
        }
    }
}

/// List keys for an agent in the given scope.
pub fn list(id: u64, scope: Scope) -> Vec<String> {
    match scope {
        Scope::Session => SESSION.with(|m| {
            let mut keys: Vec<String> = m
                .iter()
                .filter(|((aid, _), _)| *aid == id)
                .map(|((_, k), _)| k.clone())
                .collect();
            keys.sort();
            keys
        }),
        Scope::Durable => {
            let prefix = format!("{}/storage/", home::path(id));
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
    }
}

/// Drop all **session** keys for `id` (call on UI agent stop).
pub fn clear_session(id: u64) {
    SESSION.with(|m| {
        let doomed: Vec<_> = m
            .keys()
            .filter(|(aid, _)| *aid == id)
            .cloned()
            .collect();
        for k in doomed {
            m.remove(&k);
        }
    });
}

/// Tool-facing entry: parse scope string, run op, return observation text.
pub fn run_tool(agent_id: u64, name: &str, args_json: &str) -> String {
    use crate::session::todo::json_str;
    let scope = Scope::parse(&json_str(args_json, "scope").unwrap_or_default()).unwrap_or(Scope::Session);
    let key = json_str(args_json, "key").unwrap_or_default();
    let value = json_str(args_json, "value").unwrap_or_default();
    match name {
        "storage_get" => {
            if key.is_empty() {
                return String::from("error: missing key");
            }
            match get_str(agent_id, scope, &key) {
                Some(v) => v,
                None => format!("error:no such key '{key}'"),
            }
        }
        "storage_set" => {
            if key.is_empty() {
                return String::from("error: missing key");
            }
            match set_str(agent_id, scope, &key, &value) {
                Ok(()) => format!("ok:stored {key}"),
                Err(e) => format!("error:{e}"),
            }
        }
        "storage_remove" => {
            if key.is_empty() {
                return String::from("error: missing key");
            }
            match remove(agent_id, scope, &key) {
                Ok(true) => format!("ok:removed {key}"),
                Ok(false) => format!("ok:absent {key}"),
                Err(e) => format!("error:{e}"),
            }
        }
        "storage_list" => {
            let keys = list(agent_id, scope);
            if keys.is_empty() {
                String::from("(empty)")
            } else {
                keys.join("\n")
            }
        }
        other => format!("error:unknown storage tool {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn session_roundtrip_and_clear() {
        let id = 4242u64;
        clear_session(id);
        set_str(id, Scope::Session, "fen", "start").unwrap();
        assert_eq!(get_str(id, Scope::Session, "fen").as_deref(), Some("start"));
        assert!(list(id, Scope::Session).iter().any(|k| k == "fen"));
        clear_session(id);
        assert!(get_str(id, Scope::Session, "fen").is_none());
    }

    #[test_case]
    fn rejects_path_escape_keys() {
        assert!(set_str(1, Scope::Session, "../x", "no").is_err());
        assert!(set_str(1, Scope::Session, "a/b", "no").is_err());
        assert!(set_str(1, Scope::Durable, "ok_key", "yes").is_ok());
    }

    #[test_case]
    fn durable_roundtrip() {
        let id = 4243u64;
        set_str(id, Scope::Durable, "board", "rnbq").unwrap();
        assert_eq!(get_str(id, Scope::Durable, "board").as_deref(), Some("rnbq"));
        assert!(list(id, Scope::Durable).iter().any(|k| k == "board"));
    }
}
