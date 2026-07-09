//! Wiring for the web **request pipeline**: three service agents connected by
//! datagram channels, each with a single responsibility —
//!
//! ```text
//!   client ── TCP ──▶ Network agent ──req──▶ HTTP agent ──req──▶ Doc agent
//!                        ▲   (owns socket)      (protocol)     (reads a file
//!                        └──resp──────────────────┘◀──body─────  via a tool call)
//! ```
//!
//! The Network agent owns the socket and relays raw bytes; the HTTP agent parses
//! the request and formats the response but never touches the socket or the FS;
//! the Doc agent maps a path to a file and reads it with a capability-gated
//! `mem_fs_read` tool call, scoped read-only to its own install folder. Each
//! stage is a native service task (deterministic code below the determinism
//! boundary); they hand data across on the channels below.
//!
//! Processing is serial (single request in flight) — correct on the cooperative
//! single-core scheduler, where the shared channel pairs never interleave.

use crate::cap::ChannelId;
use crate::channel::{self, ChannelKind};
use crate::mm::Locked;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};

const UNSET: u64 = u64::MAX;

// The four datagram channels linking the stages (raw request/response bytes and
// the http↔server request/body handoff). Created once by `wire`.
static NET_TO_HTTP: AtomicU64 = AtomicU64::new(UNSET); // raw request bytes
static HTTP_TO_NET: AtomicU64 = AtomicU64::new(UNSET); // raw response bytes
static HTTP_TO_SERVER: AtomicU64 = AtomicU64::new(UNSET); // "METHOD path"
static SERVER_TO_HTTP: AtomicU64 = AtomicU64::new(UNSET); // "status\nctype\n" + body

static NET_PORT: AtomicU16 = AtomicU16::new(0);
// The install folder of the content agent currently being served (its SOUL +
// assets). Set per `start` — the pipeline serves whichever agent you point it at.
static CONTENT_HOME: Locked<Option<String>> = Locked::new(None);
static WIRED: AtomicBool = AtomicBool::new(false);

fn get(a: &AtomicU64) -> Option<ChannelId> {
    let v = a.load(Ordering::SeqCst);
    if v == UNSET {
        None
    } else {
        Some(v)
    }
}

pub fn net_to_http() -> Option<ChannelId> {
    get(&NET_TO_HTTP)
}
pub fn http_to_net() -> Option<ChannelId> {
    get(&HTTP_TO_NET)
}
pub fn http_to_server() -> Option<ChannelId> {
    get(&HTTP_TO_SERVER)
}
pub fn server_to_http() -> Option<ChannelId> {
    get(&SERVER_TO_HTTP)
}
pub fn net_port() -> u16 {
    NET_PORT.load(Ordering::SeqCst)
}
pub fn content_home() -> Option<String> {
    CONTENT_HOME.with(|d| d.clone())
}

/// Create the four pipeline channels (idempotent — once per boot).
fn wire() {
    if WIRED.swap(true, Ordering::SeqCst) {
        return;
    }
    NET_TO_HTTP.store(channel::create(ChannelKind::Datagram, 16), Ordering::SeqCst);
    HTTP_TO_NET.store(channel::create(ChannelKind::Datagram, 16), Ordering::SeqCst);
    HTTP_TO_SERVER.store(channel::create(ChannelKind::Datagram, 16), Ordering::SeqCst);
    SERVER_TO_HTTP.store(channel::create(ChannelKind::Datagram, 16), Ordering::SeqCst);
}

/// Bring up the whole web pipeline serving the content agent installed at
/// `content_home`: wire the channels, record the listen port + that agent's
/// install folder, and start the three generic stage tasks (Network relay, HTTP
/// protocol, generic content Server). The Server stage is granted read-only
/// scope to the served agent's home so its `mem_fs_read` tool calls pass the
/// executor's scope gate. Returns the Network stage's task id.
///
/// The Network and HTTP stages are the same reusable plumbing for every server
/// agent; only `content_home` changes — so serving a different agent (a user's
/// own SOUL + assets) needs no new code, just a different home.
pub fn start(port: u16, content_home: &str) -> crate::sched::TaskId {
    use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
    wire();
    NET_PORT.store(port, Ordering::SeqCst);
    CONTENT_HOME.with(|d| *d = Some(String::from(content_home)));

    let server_task = super::start(&super::server::SERVER_STAGE);
    // The served agent may read only within its own install folder (+ memory).
    crate::cap::grant_scopes(
        server_task,
        &[CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Path(alloc::format!("{content_home}/**")))],
    );
    super::start(&super::http::HTTP_STAGE);
    super::start(&super::network::NETWORK_STAGE)
}

/// Cooperatively wait for the next datagram on `id`, up to `deadline_ms`
/// (`arch::now_ms`). Pumps upkeep + yields every empty spin. `None` on timeout.
pub fn recv_deadline(id: ChannelId, deadline_ms: u64) -> Option<Vec<u8>> {
    loop {
        if let Ok(Some(frame)) = channel::try_recv_dgram(id) {
            return Some(frame);
        }
        if crate::arch::now_ms() >= deadline_ms {
            return None;
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// Send a whole datagram frame, retrying under back-pressure up to a deadline.
pub fn send_frame(id: ChannelId, data: &[u8], deadline_ms: u64) -> bool {
    loop {
        match channel::try_send_dgram(id, data) {
            Ok(()) => return true,
            Err(channel::ChannelError::WouldBlock) => {}
            Err(_) => return false,
        }
        if crate::arch::now_ms() >= deadline_ms {
            return false;
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// Per-request budget for a stage waiting on the next stage. Generous because
/// the Doc agent plans each route with a live model turn (and pays a one-time
/// model load on the first request) — routing is a judgment, not a table lookup.
pub const STAGE_DEADLINE_MS: u64 = 60_000;
