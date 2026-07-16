//! Session persistence over the memory store (`synapse::fs`, the "disk").
//! A session serializes with `postcard` to `/sessions/<id>` and resumes
//! deterministically: same seed + same messages ⇒ identical continuation (the
//! KV cache is recomputed, never stored). `fork` branches a session under a new
//! id so an exploration can diverge without mutating the original (Phase D).

use crate::agent::types::*;
use alloc::format;
use alloc::string::ToString;

fn key_for(id: SessionId) -> alloc::string::String {
    format!("/sessions/{}", id.0)
}

/// Persist `session` to the memory store. Overwrites any prior snapshot at the
/// same id (write-through).
pub fn save(session: &Session) -> Result<(), postcard::Error> {
    let bytes = postcard::to_allocvec(session)?;
    crate::synapse::fs::write(&key_for(session.id), &bytes);
    // Also write the human-readable JSONL transcript
    // (presentation only; postcard above is the source of truth for resume).
    super::jsonl::write_transcript(session);
    // bordered summary index for list/search UX.
    write_summary(session, None);
    crate::ktrace::log_fmt(format_args!(
        "session.save: id={} messages={} bytes={}",
        session.id.0,
        session.messages.len(),
        bytes.len()
    ));
    Ok(())
}

/// Write `/sessions/<id>/summary.json` (title, counts, optional parent).
fn write_summary(session: &Session, parent: Option<u64>) {
    let mut user_texts: alloc::vec::Vec<&str> = alloc::vec::Vec::new();
    for m in &session.messages {
        if matches!(m.role, Role::User) {
            user_texts.push(m.content.as_str());
        }
    }
    let title = crate::agent::prompt::title_from_messages(&user_texts);
    let body = crate::agent::prompt::session_summary_json(
        session.id.0,
        &title,
        session.messages.len(),
        "chitti",
        parent,
        session.updated_ticks,
    );
    let path = alloc::format!("/sessions/{}/summary.json", session.id.0);
    crate::synapse::fs::write(&path, body.as_bytes());
}

/// List session summary lines for `/session list` (id + title).
pub fn list_summaries() -> alloc::vec::Vec<(u64, alloc::string::String)> {
    let mut out = alloc::vec::Vec::new();
    for p in crate::synapse::fs::list() {
        // postcard keys are `/sessions/<id>` with no further slash.
        let Some(rest) = p.strip_prefix("/sessions/") else {
            continue;
        };
        if rest.contains('/') {
            continue;
        }
        let Ok(id) = rest.parse::<u64>() else {
            continue;
        };
        let title = crate::synapse::fs::read(&alloc::format!("/sessions/{id}/summary.json"))
            .and_then(|b| {
                let t = alloc::string::String::from_utf8_lossy(&b);
                // crude "title":"…" extract
                let key = "\"title\":\"";
                let i = t.find(key)?;
                let after = &t[i + key.len()..];
                let end = after.find('"')?;
                Some(after[..end].to_string())
            })
            .unwrap_or_else(|| alloc::string::String::from("(no summary)"));
        out.push((id, title));
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out
}

/// Substring search over session summaries + jsonl transcripts.
pub fn search_sessions(query: &str) -> alloc::vec::Vec<(u64, alloc::string::String)> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return list_summaries();
    }
    let mut out = alloc::vec::Vec::new();
    for (id, title) in list_summaries() {
        let mut hay = title.to_ascii_lowercase();
        if let Some(bytes) = crate::synapse::fs::read(&alloc::format!("/sessions/{id}.jsonl")) {
            hay.push_str(&alloc::string::String::from_utf8_lossy(&bytes).to_ascii_lowercase());
        }
        if hay.contains(&q) {
            out.push((id, title));
        }
    }
    out
}

/// Reconstruct a session previously [`save`]d. Returns `None` if absent, or the
/// postcard error if the stored bytes are corrupt.
///
/// Also advances the process-global id minters past any ids already present in
/// the snapshot, so a later `Session::new` / `push_message` cannot collide with
/// a resumed id (counters otherwise restart at 1 each boot).
pub fn resume(id: SessionId) -> Option<Session> {
    let bytes = crate::synapse::fs::read(&key_for(id))?;
    match postcard::from_bytes::<Session>(&bytes) {
        Ok(s) => {
            notice_ids(&s);
            crate::ktrace::log_fmt(format_args!(
                "session.resume: id={} messages={} (KV recomputed from seed {})",
                s.id.0,
                s.messages.len(),
                s.seed
            ));
            Some(s)
        }
        Err(_) => None,
    }
}

/// Bump id minters so they sit past every id already living in `session`.
fn notice_ids(session: &Session) {
    notice_session_id(session.id);
    for m in &session.messages {
        notice_msg_id(m.id);
        for c in &m.tool_calls {
            // call_ids are a separate sequence in practice; still advance the
            // msg minter past them so ids stay unique across kinds in display.
            notice_msg_id(MsgId(c.call_id));
        }
        if let Some(cid) = m.tool_call_id {
            notice_msg_id(MsgId(cid));
        }
    }
}

/// Branch `session` into a new session under a fresh id, copying its state.
/// The fork is independent: mutating it never touches the parent (Phase D
/// "explore without mutating the original"). Not persisted until `save`d.
/// Writes a summary with `parent` linkage when later saved via [`save_fork`].
pub fn fork(session: &Session, now: Ticks) -> Session {
    let mut f = session.clone();
    f.id = next_session_id();
    f.created_ticks = now;
    f.updated_ticks = now;
    crate::ktrace::log_fmt(format_args!("session.fork: {} -> {}", session.id.0, f.id.0));
    f
}

/// Persist a forked session and record `parent` in summary.json.
pub fn save_fork(session: &Session, parent: u64) -> Result<(), postcard::Error> {
    let bytes = postcard::to_allocvec(session)?;
    crate::synapse::fs::write(&key_for(session.id), &bytes);
    super::jsonl::write_transcript(session);
    write_summary(session, Some(parent));
    Ok(())
}
