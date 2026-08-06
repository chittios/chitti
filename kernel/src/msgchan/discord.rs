//! **Discord Bot API** adapter for messaging channels.
//!
//! Uses HTTPS REST (no Gateway WebSocket): poll a single channel's message
//! history and post replies. Token form: `BOT_TOKEN#CHANNEL_ID` (the channel
//! snowflake after `#` is the inbox to watch).

use super::{Instance, RawMessage};
use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const API: &str = "https://discord.com/api/v10";

/// Split `token#channel_id` stored on the instance.
pub fn split_cred(token: &str) -> Result<(&str, &str), String> {
    let (bot, chan) = token
        .split_once('#')
        .ok_or_else(|| String::from("discord token must be BOT_TOKEN#CHANNEL_ID"))?;
    if bot.is_empty() || chan.is_empty() {
        return Err(String::from("discord token must be BOT_TOKEN#CHANNEL_ID"));
    }
    Ok((bot, chan))
}

fn auth_header(bot: &str) -> String {
    format!("Bot {bot}")
}

/// `GET /users/@me` → username for startup probe.
pub fn get_me(token: &str) -> Result<String, String> {
    let (bot, _) = split_cred(token)?;
    let auth = auth_header(bot);
    let url = format!("{API}/users/@me");
    let resp = crate::net::http::request(
        "GET",
        &url,
        &[("Authorization", auth.as_str())],
        &[],
        8_000,
    )?;
    if resp.status != 200 {
        return Err(format!("HTTP {}", resp.status));
    }
    let j = Json::parse(&resp.text()).ok_or_else(|| String::from("bad json"))?;
    let uname = j
        .get("username")
        .and_then(|v| v.as_str())
        .unwrap_or("bot");
    Ok(uname.to_string())
}

/// Send a plain-text message to `channel_id` (peer).
pub fn send_message(token: &str, channel_id: &str, text: &str) -> Result<(), String> {
    let (bot, default_chan) = split_cred(token)?;
    let chan = if channel_id.is_empty() {
        default_chan
    } else {
        channel_id
    };
    let auth = auth_header(bot);
    let body = format!("{{\"content\":{}}}", json_escape(text));
    let url = format!("{API}/channels/{chan}/messages");
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
        return Err(format!("sendMessage HTTP {}", resp.status));
    }
    Ok(())
}

/// Poll channel messages after `inst.offset` (snowflake as i64).
pub fn poll(inst: &mut Instance) -> Result<Vec<RawMessage>, String> {
    let (bot, chan) = split_cred(&inst.token)?;
    let auth = auth_header(bot);
    let url = if inst.offset > 0 {
        format!(
            "{API}/channels/{chan}/messages?after={}&limit=10",
            inst.offset
        )
    } else {
        format!("{API}/channels/{chan}/messages?limit=10")
    };
    let resp = crate::net::http::request(
        "GET",
        &url,
        &[("Authorization", auth.as_str())],
        &[],
        6_000,
    )?;
    if resp.status != 200 {
        return Err(format!("messages HTTP {}", resp.status));
    }
    let j = Json::parse(&resp.text()).ok_or_else(|| String::from("bad json"))?;
    let arr = j.as_array().ok_or_else(|| String::from("expected array"))?;
    // Discord returns newest-first; reverse so we process oldest→newest.
    let mut out = Vec::new();
    for msg in arr.iter().rev() {
        let id = msg
            .get("id")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);
        if id <= inst.offset {
            continue;
        }
        // Skip our own bot messages (author.bot).
        if msg
            .get("author")
            .and_then(|a| a.get("bot"))
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            if id > inst.offset {
                inst.offset = id;
            }
            continue;
        }
        let text = msg
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            if id > inst.offset {
                inst.offset = id;
            }
            continue;
        }
        let author = msg.get("author");
        let from_id = author
            .and_then(|a| a.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let from_name = author
            .and_then(|a| a.get("username"))
            .and_then(|v| v.as_str())
            .unwrap_or(from_id.as_str())
            .to_string();
        if id > inst.offset {
            inst.offset = id;
        }
        out.push(RawMessage {
            update_id: id,
            chat_id: chan.to_string(),
            from_id,
            from_name,
            text,
        });
    }
    Ok(out)
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
        assert!(split_cred("onlytoken").is_err());
        assert!(split_cred("#noid").is_err());
        let (b, c) = split_cred("abc.def#1234567890").unwrap();
        assert_eq!(b, "abc.def");
        assert_eq!(c, "1234567890");
    }

    #[test_case]
    fn json_escape_quotes() {
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
    }
}
