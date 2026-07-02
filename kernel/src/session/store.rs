//! Session persistence over the memory store (`synapse::fs`, the "disk").
//! A session serializes with `postcard` to `sess/<id>` and resumes
//! deterministically: same seed + same messages ⇒ identical continuation (the
//! KV cache is recomputed, never stored). `fork` branches a session under a new
//! id so an exploration can diverge without mutating the original (Phase D).

use crate::agent::types::*;
use alloc::format;

fn key_for(id: SessionId) -> alloc::string::String {
    format!("sess/{}", id.0)
}

/// Persist `session` to the memory store. Overwrites any prior snapshot at the
/// same id (write-through).
pub fn save(session: &Session) -> Result<(), postcard::Error> {
    let bytes = postcard::to_allocvec(session)?;
    crate::synapse::fs::write(&key_for(session.id), &bytes);
    crate::ktrace::log_fmt(format_args!(
        "session.save: id={} messages={} bytes={}",
        session.id.0,
        session.messages.len(),
        bytes.len()
    ));
    Ok(())
}

/// Reconstruct a session previously [`save`]d. Returns `None` if absent, or the
/// postcard error if the stored bytes are corrupt.
pub fn resume(id: SessionId) -> Option<Session> {
    let bytes = crate::synapse::fs::read(&key_for(id))?;
    match postcard::from_bytes::<Session>(&bytes) {
        Ok(s) => {
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

/// Branch `session` into a new session under a fresh id, copying its state.
/// The fork is independent: mutating it never touches the parent (Phase D
/// "explore without mutating the original"). Not persisted until `save`d.
pub fn fork(session: &Session, now: Ticks) -> Session {
    let mut f = session.clone();
    f.id = next_session_id();
    f.created_ticks = now;
    f.updated_ticks = now;
    crate::ktrace::log_fmt(format_args!("session.fork: {} -> {}", session.id.0, f.id.0));
    f
}
