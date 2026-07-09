//! **Telegram Bot API** adapter for [`super`] messaging channels.
//!
//! Uses HTTPS long-poll style `getUpdates` (short timeout so the shell upkeep
//! stays cooperative) and `sendMessage` for outbound text. No grammY — pure
//! HTTP + JSON over Chitti's existing net stack.

use super::Instance;
use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Parsed inbound Telegram message (text only for v1).
#[derive(Clone, Debug)]
pub struct TgMessage {
    pub update_id: i64,
    pub chat_id: String,
    pub from_id: String,
    pub from_name: String,
    pub text: String,
}

fn api_url(token: &str, method: &str) -> String {
    format!("https://api.telegram.org/bot{token}/{method}")
}

/// `getMe` → bot username (without @), for startup probe.
/// Short timeout so a bad route never freezes the shell for tens of seconds.
pub fn get_me(token: &str) -> Result<String, String> {
    let url = api_url(token, "getMe");
    let resp = crate::net::http::get(&url, 8_000).map_err(|e| e)?;
    if resp.status != 200 {
        return Err(format!("HTTP {}", resp.status));
    }
    let text = resp.text();
    let j = Json::parse(&text).ok_or_else(|| String::from("bad json"))?;
    if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(j.get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("getMe not ok")
            .to_string());
    }
    let uname = j
        .get("result")
        .and_then(|r| r.get("username"))
        .and_then(|v| v.as_str())
        .unwrap_or("bot");
    Ok(uname.to_string())
}

/// Send a plain-text message to `chat_id`.
pub fn send_message(token: &str, chat_id: &str, text: &str) -> Result<(), String> {
    // Telegram hard limit ~4096; chunk if needed.
    let chunks = chunk_text(text, 3500);
    for chunk in chunks {
        let body = format!(
            "{{\"chat_id\":{},\"text\":{}}}",
            json_str_or_num(chat_id),
            json_escape(&chunk)
        );
        let url = api_url(token, "sendMessage");
        let resp = crate::net::http::post_json(&url, &body, None, 10_000)?;
        if resp.status / 100 != 2 {
            return Err(format!("sendMessage HTTP {}", resp.status));
        }
        let t = resp.text();
        if let Some(j) = Json::parse(&t) {
            if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                return Err(j
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sendMessage failed")
                    .to_string());
            }
        }
    }
    Ok(())
}

/// Poll `getUpdates` once (Telegram `timeout=0` = return immediately).
/// Uses a short client deadline so a stuck TLS path cannot wedge the UI;
/// caller rate-limits via [`super::tick`]. Advances `inst.offset`.
pub fn poll(inst: &mut Instance) -> Result<Vec<TgMessage>, String> {
    let url = format!(
        "{}?offset={}&timeout=0&limit=10&allowed_updates=%5B%22message%22%5D",
        api_url(&inst.token, "getUpdates"),
        inst.offset
    );
    // 6s client budget — enough for TLS+getUpdates empty body; Ctrl+C cancels.
    let resp = crate::net::http::get(&url, 6_000).map_err(|e| e)?;
    if resp.status != 200 {
        return Err(format!("getUpdates HTTP {}", resp.status));
    }
    let text = resp.text();
    let j = Json::parse(&text).ok_or_else(|| String::from("bad json"))?;
    if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(j
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("getUpdates not ok")
            .to_string());
    }
    let mut out = Vec::new();
    let Some(arr) = j.get("result").and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    for upd in arr {
        let update_id = upd.get("update_id").and_then(|v| v.as_i64()).unwrap_or(0);
        if update_id + 1 > inst.offset {
            inst.offset = update_id + 1;
        }
        let Some(msg) = upd.get("message") else {
            continue;
        };
        let text = msg
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            continue; // stickers/photos etc. later
        }
        let chat_id = msg
            .get("chat")
            .and_then(|c| c.get("id"))
            .map(json_id)
            .unwrap_or_default();
        let from = msg.get("from");
        let from_id = from
            .and_then(|f| f.get("id"))
            .map(json_id)
            .unwrap_or_else(|| chat_id.clone());
        let from_name = from
            .and_then(|f| {
                let first = f.get("first_name").and_then(|v| v.as_str()).unwrap_or("");
                let user = f.get("username").and_then(|v| v.as_str());
                if let Some(u) = user {
                    Some(format!("{first} (@{u})"))
                } else if !first.is_empty() {
                    Some(first.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| from_id.clone());
        out.push(TgMessage {
            update_id,
            chat_id,
            from_id,
            from_name,
            text,
        });
    }
    Ok(out)
}

fn json_id(v: &Json) -> String {
    match v {
        Json::Num(n) => format!("{}", *n as i64),
        Json::Str(s) => s.clone(),
        _ => String::new(),
    }
}

/// Chat ids are numeric; still accept string form.
fn json_str_or_num(id: &str) -> String {
    if id.chars().all(|c| c == '-' || c.is_ascii_digit()) {
        id.to_string()
    } else {
        json_escape(id)
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn chunk_text(s: &str, max: usize) -> Vec<String> {
    if s.len() <= max {
        return alloc::vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        if rest.len() <= max {
            out.push(rest.to_string());
            break;
        }
        // Prefer break at newline.
        let mut cut = max;
        if let Some(i) = rest[..max].rfind('\n') {
            if i > max / 4 {
                cut = i + 1;
            }
        }
        out.push(rest[..cut].to_string());
        rest = &rest[cut..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn json_escape_quotes() {
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert!(json_escape("hi\n").contains("\\n"));
    }

    #[test_case]
    fn chunk_splits_long() {
        let s = "x".repeat(100);
        let c = chunk_text(&s, 30);
        assert!(c.len() >= 3);
        assert_eq!(c.iter().map(|p| p.len()).sum::<usize>(), 100);
    }
}
