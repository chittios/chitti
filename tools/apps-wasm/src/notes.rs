use crate::guest::{json_str, storage_get_durable, storage_list_durable, storage_remove_durable, storage_set_durable};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const PREFIX: &str = "note_";

fn key_ok(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= 64
        && k.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

pub fn list(_: &str) -> String {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    if n < 0 {
        return String::from("error:list failed");
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    let mut keys: Vec<String> = raw
        .split('\n')
        .filter(|k| k.starts_with(PREFIX))
        .map(|k| k[PREFIX.len()..].to_string())
        .collect();
    keys.sort();
    if keys.is_empty() {
        String::from("(empty)")
    } else {
        keys.join("\n")
    }
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let mut buf = [0u8; 8192];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no such note '{key}'");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn set(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "value"))
        .or_else(|| json_str(args, "content"))
        .unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage_set failed");
    }
    format!("ok:note {key} ({} bytes)", body.len())
}

pub fn remove(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let _ = storage_remove_durable(&sk);
    format!("ok:removed {key}")
}
