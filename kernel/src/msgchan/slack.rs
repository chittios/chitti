//! **Slack Web API** adapter for messaging channels.
//!
//! Polls `conversations.history` and posts via `chat.postMessage`. Token form:
//! `BOT_TOKEN#CHANNEL_ID` (channel id like `C01234567`).

use super::{Instance, RawMessage};
use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const API: &str = "https://slack.com/api";

/// Split `token#channel_id` stored on the instance.
pub fn split_cred(token: &str) -> Result<(&str, &str), String> {
    let (bot, chan) = token
        .split_once('#')
        .ok_or_else(|| String::from("slack token must be BOT_TOKEN#CHANNEL_ID"))?;
    if bot.is_empty() || chan.is_empty() {
        return Err(String::from("slack token must be BOT_TOKEN#CHANNEL_ID"));
    }
    Ok((bot, chan))
}

fn auth_header(bot: &str) -> String {
    format!("Bearer {bot}")
}

/// `auth.test` → bot user id / team for startup probe.
pub fn get_me(token: &str) -> Result<String, String> {
    let (bot, _) = split_cred(token)?;
    let auth = auth_header(bot);
    let url = format!("{API}/auth.test");
    let resp = crate::net::http::request(
        "POST",
        &url,
        &[
            ("Authorization", auth.as_str()),
            ("Content-Type", "application/x-www-form-urlencoded"),
        ],
        &[],
        8_000,
    )?;
    if resp.status != 200 {
        return Err(format!("HTTP {}", resp.status));
    }
    let j = Json::parse(&resp.text()).ok_or_else(|| String::from("bad json"))?;
    if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(j
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("auth.test failed")
            .to_string());
    }
    let user = j
        .get("user")
        .and_then(|v| v.as_str())
        .or_else(|| j.get("user_id").and_then(|v| v.as_str()))
        .unwrap_or("bot");
    Ok(user.to_string())
}

/// Send a plain-text message to `channel_id`.
pub fn send_message(token: &str, channel_id: &str, text: &str) -> Result<(), String> {
    let (bot, default_chan) = split_cred(token)?;
    let chan = if channel_id.is_empty() {
        default_chan
    } else {
        channel_id
    };
    let auth = auth_header(bot);
    let body = format!(
        "{{\"channel\":{},\"text\":{}}}",
        json_escape(chan),
        json_escape(text)
    );
    let url = format!("{API}/chat.postMessage");
    let resp = crate::net::http::request(
        "POST",
        &url,
        &[
            ("Authorization", auth.as_str()),
            ("Content-Type", "application/json"),
        ],
        body.as_bytes(),
        10_000,
    )?;
    if resp.status / 100 != 2 {
        return Err(format!("chat.postMessage HTTP {}", resp.status));
    }
    let t = resp.text();
    if let Some(j) = Json::parse(&t) {
        if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            return Err(j
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("chat.postMessage failed")
                .to_string());
        }
    }
    Ok(())
}

/// Poll `conversations.history`. `inst.offset` stores the last `ts` as
/// micros (ts * 1e6) so we can order without floats in the cursor.
pub fn poll(inst: &mut Instance) -> Result<Vec<RawMessage>, String> {
    let (bot, chan) = split_cred(&inst.token)?;
    let auth = auth_header(bot);
    // Slack `oldest` is a string timestamp; we store micros in offset.
    let oldest = if inst.offset > 0 {
        let secs = inst.offset / 1_000_000;
        let micros = inst.offset % 1_000_000;
        format!("{secs}.{micros:06}")
    } else {
        String::from("0")
    };
    let url = format!(
        "{API}/conversations.history?channel={chan}&oldest={oldest}&limit=10&inclusive=false"
    );
    let resp = crate::net::http::request(
        "GET",
        &url,
        &[("Authorization", auth.as_str())],
        &[],
        6_000,
    )?;
    if resp.status != 200 {
        return Err(format!("conversations.history HTTP {}", resp.status));
    }
    let j = Json::parse(&resp.text()).ok_or_else(|| String::from("bad json"))?;
    if j.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(j
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("history not ok")
            .to_string());
    }
    let mut out = Vec::new();
    let Some(arr) = j.get("messages").and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    // Newest-first from Slack; reverse for chronological handling.
    for msg in arr.iter().rev() {
        // Skip bot_message / subtypes we do not want to answer.
        if msg.get("bot_id").is_some() || msg.get("subtype").is_some() {
            continue;
        }
        let ts_str = msg.get("ts").and_then(|v| v.as_str()).unwrap_or("");
        let ts_micros = parse_ts_micros(ts_str);
        if ts_micros <= inst.offset {
            continue;
        }
        let text = msg
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            if ts_micros > inst.offset {
                inst.offset = ts_micros;
            }
            continue;
        }
        let from_id = msg
            .get("user")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if ts_micros > inst.offset {
            inst.offset = ts_micros;
        }
        out.push(RawMessage {
            update_id: ts_micros,
            chat_id: chan.to_string(),
            from_id: from_id.clone(),
            from_name: from_id,
            text,
        });
    }
    Ok(out)
}

/// Parse Slack `ts` ("1672531200.000100") into integer micros.
pub fn parse_ts_micros(ts: &str) -> i64 {
    let (secs, frac) = match ts.split_once('.') {
        Some((s, f)) => (s, f),
        None => (ts, "0"),
    };
    let s: i64 = secs.parse().unwrap_or(0);
    let mut f = frac.as_bytes().to_vec();
    while f.len() < 6 {
        f.push(b'0');
    }
    f.truncate(6);
    let micros: i64 = core::str::from_utf8(&f)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    s.saturating_mul(1_000_000).saturating_add(micros)
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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn split_cred_requires_hash() {
        assert!(split_cred("xoxb-only").is_err());
        let (b, c) = split_cred("xoxb-abc#C0123").unwrap();
        assert_eq!(b, "xoxb-abc");
        assert_eq!(c, "C0123");
    }

    #[test_case]
    fn parse_ts_micros_pads_fraction() {
        assert_eq!(parse_ts_micros("1672531200.000100"), 1_672_531_200_000_100);
        assert_eq!(parse_ts_micros("100"), 100_000_000);
        assert_eq!(parse_ts_micros("100.5"), 100_500_000);
    }
}
