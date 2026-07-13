//! Session transcript in the Claude-Code JSONL shape: one append-only JSON
//! record per message, each carrying a `uuid` / `parentUuid` thread link, an
//! ISO-8601 timestamp, the `sessionId` + `cwd`, and a `message` with a role and
//! typed content blocks (`text` / `tool_use{id,name,input}` /
//! `tool_result{tool_use_id,content}`).
//!
//! This is a **presentation** artifact written at `/sessions/<id>.jsonl` alongside
//! the authoritative postcard snapshot (which drives deterministic resume). A
//! human — or an external tool — reads the JSONL to follow a session, exactly
//! as Claude Code's `~/.claude/projects/*/*.jsonl` transcripts do. It is
//! rewritten in full on each save (kernel sessions are small; append needs no
//! separate store API).

use crate::agent::types::{Message, Role, Session};
use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

fn key_for(id: u64) -> String {
    format!("/sessions/{}.jsonl", id)
}

/// A stable per-message thread key, `<session>-<msgid>`.
fn uuid(session_id: u64, msg_id: u64) -> String {
    format!("{}-{}", session_id, msg_id)
}

/// The record `type` / message `role` string. A tool result is threaded as a
/// `user` record (its block is a `tool_result`), matching the Claude shape.
fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "user",
    }
}

/// The typed `content` blocks for one message.
fn content_blocks(m: &Message) -> Json {
    let mut blocks: Vec<Json> = Vec::new();
    if m.role == Role::Tool {
        blocks.push(Json::Obj(vec![
            ("type".to_string(), Json::Str("tool_result".to_string())),
            (
                "tool_use_id".to_string(),
                Json::Str(format!("toolu_{}", m.tool_call_id.unwrap_or(0))),
            ),
            ("content".to_string(), Json::Str(m.content.clone())),
        ]));
        return Json::Arr(blocks);
    }
    // Prose (omitted for a pure tool-call assistant turn).
    if !m.content.is_empty() {
        blocks.push(Json::Obj(vec![
            ("type".to_string(), Json::Str("text".to_string())),
            ("text".to_string(), Json::Str(m.content.clone())),
        ]));
    }
    // One tool_use block per call; `input` is the tool's JSON args embedded as
    // an object when parseable, else kept as a raw string.
    for tc in &m.tool_calls {
        let input = Json::parse(&tc.args).unwrap_or_else(|| Json::Str(tc.args.clone()));
        blocks.push(Json::Obj(vec![
            ("type".to_string(), Json::Str("tool_use".to_string())),
            ("id".to_string(), Json::Str(format!("toolu_{}", tc.call_id))),
            ("name".to_string(), Json::Str(tc.tool.clone())),
            ("input".to_string(), input),
        ]));
    }
    Json::Arr(blocks)
}

/// Build one compact JSONL record line for a message (no trailing newline).
fn record(session: &Session, m: &Message, parent: Option<u64>, ts: &str) -> String {
    let sid = session.id.0;
    let message = Json::Obj(vec![
        ("role".to_string(), Json::Str(role_str(m.role).to_string())),
        ("content".to_string(), content_blocks(m)),
    ]);
    Json::Obj(vec![
        ("type".to_string(), Json::Str(role_str(m.role).to_string())),
        ("uuid".to_string(), Json::Str(uuid(sid, m.id.0))),
        (
            "parentUuid".to_string(),
            match parent {
                Some(p) => Json::Str(uuid(sid, p)),
                None => Json::Null,
            },
        ),
        ("timestamp".to_string(), Json::Str(ts.to_string())),
        ("sessionId".to_string(), Json::Str(format!("{}", sid))),
        ("cwd".to_string(), Json::Str(session.env.cwd.clone())),
        ("message".to_string(), message),
    ])
    .to_compact()
}

/// Serialize the whole session to its Claude-Code-style JSONL transcript and
/// write it to `/sessions/<id>.jsonl` in the store.
pub fn write_transcript(session: &Session) {
    let ts = crate::clock::now_iso8601();
    let mut out = String::new();
    let mut parent: Option<u64> = None;
    for m in &session.messages {
        out.push_str(&record(session, m, parent, &ts));
        out.push('\n');
        parent = Some(m.id.0);
    }
    crate::synapse::fs::write(&key_for(session.id.0), out.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::{MsgId, Provenance, ToolCall};

    fn msg(id: u64, role: Role, content: &str, calls: Vec<ToolCall>, tc: Option<u64>) -> Message {
        Message {
            id: MsgId(id),
            role,
            content: content.to_string(),
            provenance: Provenance::UserTyped,
            tool_calls: calls,
            tool_call_id: tc,
            tokens: 0,
            ticks: 0,
            resident: true,
            store_ref: None,
        }
    }

    #[test_case]
    fn user_record_is_text_block_with_thread_link() {
        let s = Session::new(&crate::agent::manifest::orchestrator_manifest(), 42, alloc::vec![], 0);
        let m = msg(1, Role::User, "hello", alloc::vec![], None);
        let line = record(&s, &m, None, "2026-07-13T00:00:00Z");
        assert!(line.contains("\"type\":\"user\""), "{line}");
        assert!(line.contains("\"parentUuid\":null"), "{line}");
        assert!(line.contains("\"type\":\"text\""), "{line}");
        assert!(line.contains("\"text\":\"hello\""), "{line}");
        assert!(!line.contains('\n'), "record must be a single line");
    }

    #[test_case]
    fn assistant_tool_use_and_tool_result_blocks() {
        let s = Session::new(&crate::agent::manifest::orchestrator_manifest(), 42, alloc::vec![], 0);
        // Assistant turn that calls a tool (no prose) → a single tool_use block.
        let a = msg(
            2,
            Role::Assistant,
            "",
            alloc::vec![ToolCall { call_id: 7, tool: "read".into(), args: "{\"path\":\"/x\"}".into() }],
            None,
        );
        let la = record(&s, &a, Some(1), "2026-07-13T00:00:01Z");
        assert!(la.contains("\"type\":\"tool_use\""), "{la}");
        assert!(la.contains("\"id\":\"toolu_7\""), "{la}");
        assert!(la.contains("\"name\":\"read\""), "{la}");
        assert!(la.contains("\"input\":{\"path\":\"/x\"}"), "input embeds parsed args: {la}");
        assert!(la.contains("\"parentUuid\":\"42-1\""), "threads to the parent: {la}");
        assert!(!la.contains("\"type\":\"text\""), "empty prose omits the text block: {la}");

        // The tool result → a user record with a tool_result block.
        let t = msg(3, Role::Tool, "file body", alloc::vec![], Some(7));
        let lt = record(&s, &t, Some(2), "2026-07-13T00:00:02Z");
        assert!(lt.contains("\"type\":\"user\""), "tool result is a user record: {lt}");
        assert!(lt.contains("\"type\":\"tool_result\""), "{lt}");
        assert!(lt.contains("\"tool_use_id\":\"toolu_7\""), "{lt}");
        assert!(lt.contains("\"content\":\"file body\""), "{lt}");
    }
}
