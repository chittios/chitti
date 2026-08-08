//! Host-side SSH sessions, addressed by a small integer.
//!
//! A wasm agent cannot hold a live connection: the git module's protocol is
//! interactive (the server speaks first with its ref advertisement, then waits
//! for the client's wants), so it needs to read, write and read again across
//! several host calls. The connection therefore lives here and the guest holds
//! only an opaque id.
//!
//! Ids are **not reused within a boot**: a stale id from an earlier call would
//! otherwise address someone else's connection, which is a confused-deputy bug
//! rather than a leak of a number.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use super::client::{Client, ExecStream};
use crate::mm::Locked;

struct Entry {
    id: u32,
    /// The agent that opened it, so one agent cannot read another's stream.
    agent_id: u64,
    client: Client,
    stream: ExecStream,
}

static SESSIONS: Locked<Vec<Entry>> = Locked::new(Vec::new());
static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/// How many sessions one agent may hold open at once. A guest that leaks
/// handles would otherwise hold TCP connections and DMA-backed buffers open
/// until the machine is rebooted.
const MAX_PER_AGENT: usize = 8;

/// Connect, authenticate and start `command`; returns the session id.
pub fn open(
    agent_id: u64,
    user: &str,
    host: &str,
    port: u16,
    command: &str,
    key: Option<&super::auth::PrivateKey>,
) -> Result<u32, String> {
    let live = SESSIONS.with(|s| s.iter().filter(|e| e.agent_id == agent_id).count());
    if live >= MAX_PER_AGENT {
        return Err("ssh: too many open sessions for this agent".into());
    }
    let mut client = Client::connect(host, port)?;
    client.authenticate(user, key, None)?;
    let stream = client.exec_stream(command)?;
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    SESSIONS.with(|s| {
        s.push(Entry {
            id,
            agent_id,
            client,
            stream,
        })
    });
    Ok(id)
}

/// Run `f` against a session the caller owns.
fn with<R>(agent_id: u64, id: u32, f: impl FnOnce(&mut Entry) -> R) -> Option<R> {
    SESSIONS.with(|s| {
        let e = s
            .iter_mut()
            .find(|e| e.id == id && e.agent_id == agent_id)?;
        Some(f(e))
    })
}

/// Read the next chunk; an empty vector means end of stream.
///
/// The client and the stream are borrowed **disjointly** out of the entry, which
/// is what lets a method on one take `&mut` the other without moving anything
/// out and putting it back.
pub fn read(agent_id: u64, id: u32) -> Option<Result<Vec<u8>, String>> {
    with(agent_id, id, |e| {
        let Entry { client, stream, .. } = e;
        client.stream_read(stream)
    })
}

pub fn write(agent_id: u64, id: u32, data: &[u8]) -> Option<Result<(), String>> {
    with(agent_id, id, |e| {
        let Entry { client, stream, .. } = e;
        client.stream_write(stream, data)
    })
}

pub fn send_eof(agent_id: u64, id: u32) -> Option<Result<(), String>> {
    with(agent_id, id, |e| {
        let Entry { client, stream, .. } = e;
        client.stream_eof(stream)
    })
}

pub fn close(agent_id: u64, id: u32) {
    let entry = SESSIONS.with(|s| {
        let i = s.iter().position(|e| e.id == id && e.agent_id == agent_id)?;
        Some(s.remove(i))
    });
    if let Some(mut e) = entry {
        e.client.disconnect();
    }
}

/// Drop every session an agent left behind.
pub fn close_all(agent_id: u64) {
    loop {
        let entry = SESSIONS.with(|s| {
            let i = s.iter().position(|e| e.agent_id == agent_id)?;
            Some(s.remove(i))
        });
        match entry {
            Some(mut e) => e.client.disconnect(),
            None => break,
        }
    }
}
