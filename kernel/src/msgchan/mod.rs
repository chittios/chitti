//! **Messaging channels** — external chat platforms (Telegram now; Discord /
//! Slack / webhooks later) that deliver inbound messages into Chitti and send
//! replies back out.
//!
//! Distinct from [`crate::channel`] (cap-gated inter-agent byte pipes). These
//! are OpenClaw-style *inbox adapters*: a named instance + backend + access
//! policy, polled cooperatively from the shell idle loop, config on the
//! Synapse store at [`CONFIG_PATH`].
//!
//! Shell surface: `/channel …` (list / add / remove / start / stop / send /
//! allow / pair / reply / status).

pub mod telegram;

use crate::json::Json;
use crate::mm::Locked;
use crate::synapse::fs as store;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// On-disk config path (JSON array of channel instances).
pub const CONFIG_PATH: &str = "/configs/core/channels.json";

/// Backend kind. New platforms add a variant + match arm — the shell command
/// and config schema stay the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Telegram,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Telegram => "telegram",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "telegram" | "tg" => Some(Kind::Telegram),
            _ => None,
        }
    }
}

/// Who may talk to a channel instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmPolicy {
    /// First unknown sender gets a pairing code; approve with `/channel pair`.
    Pairing,
    /// Only numeric IDs in `allow_from`.
    Allowlist,
    /// Anyone (requires explicit `allow_from` containing `*`).
    Open,
}

impl DmPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            DmPolicy::Pairing => "pairing",
            DmPolicy::Allowlist => "allowlist",
            DmPolicy::Open => "open",
        }
    }
    pub fn parse(s: &str) -> Option<DmPolicy> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pairing" | "pair" => Some(DmPolicy::Pairing),
            "allowlist" | "allow" => Some(DmPolicy::Allowlist),
            "open" => Some(DmPolicy::Open),
            _ => None,
        }
    }
}

/// One configured messaging channel instance.
#[derive(Clone, Debug)]
pub struct Instance {
    pub name: String,
    pub kind: Kind,
    /// Opaque credential (Telegram bot token, etc.).
    pub token: String,
    pub policy: DmPolicy,
    /// Allowed sender IDs as strings (`"12345"` or `"*"` for open).
    pub allow_from: Vec<String>,
    /// When true, shell `tick` polls the backend.
    pub running: bool,
    /// Backend-private cursor (Telegram `getUpdates` offset).
    pub offset: i64,
    /// Pending pairing: (code, sender_id, display name).
    pub pending_pair: Option<(String, String, String)>,
    /// Last inbound peer for `/channel reply`.
    pub last_peer: Option<String>,
    pub last_error: Option<String>,
    /// When true, inbound text is queued for the shell agent to answer.
    pub auto_agent: bool,
}

/// Normalised inbound message handed to the shell / agent.
#[derive(Clone, Debug)]
pub struct Inbound {
    pub channel: String,
    pub kind: Kind,
    pub from_id: String,
    pub from_name: String,
    pub peer_id: String,
    pub text: String,
}

static INSTANCES: Locked<Vec<Instance>> = Locked::new(Vec::new());
static INBOUND: Locked<Vec<Inbound>> = Locked::new(Vec::new());

/// Reentrancy / mutual exclusion for any Telegram (or other) HTTP from this
/// module. HTTP drivers call `shell::upkeep` while waiting, and upkeep used to
/// call `tick` again → nested HTTPS → frozen shell. One in-flight op at a time.
static BUSY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// Wall-clock of last successful poll attempt (rate-limit tick).
static LAST_POLL_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// Minimum gap between poll rounds (ms). Keeps the idle loop free.
const POLL_INTERVAL_MS: u64 = 2500;

/// Run `f` only if no other msgchan HTTP is in flight. Nested calls return
/// `None` without running `f` (safe for upkeep reentry).
fn with_busy<T>(f: impl FnOnce() -> T) -> Option<T> {
    use core::sync::atomic::Ordering;
    if BUSY
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return None;
    }
    let out = f();
    BUSY.store(false, Ordering::Release);
    Some(out)
}

/// Load config from the store (call once at boot after the store is mounted).
pub fn load() {
    let Some(bytes) = store::read(CONFIG_PATH) else {
        return;
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return;
    };
    let Some(Json::Arr(arr)) = Json::parse(text) else {
        return;
    };
    let mut out = Vec::new();
    for item in arr {
        let Json::Obj(pairs) = item else { continue };
        let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, v)| v);
        let name = get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        let kind = get("kind")
            .and_then(|v| v.as_str())
            .and_then(Kind::parse)
            .unwrap_or(Kind::Telegram);
        let token = get("token").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let policy = get("policy")
            .and_then(|v| v.as_str())
            .and_then(DmPolicy::parse)
            .unwrap_or(DmPolicy::Pairing);
        let mut allow_from = Vec::new();
        if let Some(Json::Arr(a)) = get("allow_from") {
            for e in a {
                if let Some(s) = e.as_str() {
                    allow_from.push(s.to_string());
                }
            }
        }
        let running = get("running").and_then(|v| v.as_bool()).unwrap_or(false);
        let offset = get("offset").and_then(|v| v.as_i64()).unwrap_or(0);
        let auto_agent = get("auto_agent").and_then(|v| v.as_bool()).unwrap_or(true);
        out.push(Instance {
            name,
            kind,
            token,
            policy,
            allow_from,
            running,
            offset,
            pending_pair: None,
            last_peer: None,
            last_error: None,
            auto_agent,
        });
    }
    INSTANCES.with(|v| *v = out);
    crate::ktrace::log_fmt(format_args!(
        "msgchan: loaded {} channel(s) from {}",
        INSTANCES.with(|v| v.len()),
        CONFIG_PATH
    ));
}

/// Persist the instance table.
pub fn save() {
    let arr: Vec<Json> = INSTANCES.with(|v| {
        v.iter()
            .map(|i| {
                let allows: Vec<Json> = i.allow_from.iter().map(|s| Json::Str(s.clone())).collect();
                Json::Obj(alloc::vec![
                    (String::from("name"), Json::Str(i.name.clone())),
                    (String::from("kind"), Json::Str(String::from(i.kind.as_str()))),
                    (String::from("token"), Json::Str(i.token.clone())),
                    (String::from("policy"), Json::Str(String::from(i.policy.as_str()))),
                    (String::from("allow_from"), Json::Arr(allows)),
                    (String::from("running"), Json::Bool(i.running)),
                    (String::from("offset"), Json::Num(i.offset as f64)),
                    (String::from("auto_agent"), Json::Bool(i.auto_agent)),
                ])
            })
            .collect()
    });
    let text = Json::Arr(arr).to_pretty();
    store::write(CONFIG_PATH, text.as_bytes());
}

/// Snapshot of all instances (for listing).
pub fn list() -> Vec<Instance> {
    INSTANCES.with(|v| v.clone())
}

/// Available backend type names.
pub fn types() -> &'static [&'static str] {
    &["telegram"]
}

/// Add a channel instance. Fails if the name is taken or kind unknown.
pub fn add(name: &str, kind: Kind, token: &str, policy: DmPolicy) -> Result<(), &'static str> {
    let name = name.trim();
    if name.is_empty() {
        return Err("empty name");
    }
    if token.trim().is_empty() {
        return Err("empty token");
    }
    let exists = INSTANCES.with(|v| v.iter().any(|i| i.name == name));
    if exists {
        return Err("name already exists");
    }
    INSTANCES.with(|v| {
        v.push(Instance {
            name: name.to_string(),
            kind,
            token: token.trim().to_string(),
            policy,
            allow_from: Vec::new(),
            running: false,
            offset: 0,
            pending_pair: None,
            last_peer: None,
            last_error: None,
            auto_agent: true,
        });
    });
    save();
    Ok(())
}

/// Remove by name.
pub fn remove(name: &str) -> Result<(), &'static str> {
    let n = INSTANCES.with(|v| {
        let before = v.len();
        v.retain(|i| i.name != name);
        before - v.len()
    });
    if n == 0 {
        return Err("no such channel");
    }
    save();
    Ok(())
}

/// Start polling for `name`.
///
/// Marks the instance running immediately, then optionally probes identity
/// under [`with_busy`] so we never nest HTTPS inside `upkeep`. A failed probe
/// leaves the channel **running** so the next `tick` can retry; the error is
/// recorded on `last_error`.
pub fn start(name: &str) -> Result<(), &'static str> {
    let ok = INSTANCES.with(|v| {
        if let Some(i) = v.iter_mut().find(|i| i.name == name) {
            i.running = true;
            i.last_error = None;
            true
        } else {
            false
        }
    });
    if !ok {
        return Err("no such channel");
    }
    save();
    // Probe identity for Telegram (short timeout; never nest with tick).
    if let Some(inst) = INSTANCES.with(|v| v.iter().find(|i| i.name == name).cloned()) {
        if inst.kind == Kind::Telegram {
            match with_busy(|| telegram::get_me(&inst.token)) {
                Some(Ok(bot)) => {
                    crate::ktrace::log_fmt(format_args!("msgchan: {} online as @{}", name, bot));
                    crate::serial_println!("channel> online as @{}", bot);
                }
                Some(Err(e)) => {
                    INSTANCES.with(|v| {
                        if let Some(i) = v.iter_mut().find(|i| i.name == name) {
                            i.last_error = Some(e.clone());
                        }
                    });
                    // Still running — poll will keep trying; do not freeze the shell.
                    crate::serial_println!(
                        "channel> getMe failed ({e}) — still started; will retry on poll (Ctrl+C to cancel waits)"
                    );
                    save();
                }
                None => {
                    crate::serial_println!(
                        "channel> probe busy; '{name}' is started — poll will probe later"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Stop polling.
pub fn stop(name: &str) -> Result<(), &'static str> {
    let ok = INSTANCES.with(|v| {
        if let Some(i) = v.iter_mut().find(|i| i.name == name) {
            i.running = false;
            true
        } else {
            false
        }
    });
    if !ok {
        return Err("no such channel");
    }
    save();
    Ok(())
}

/// Allow a sender id (or `*` for open wildcard).
pub fn allow(name: &str, user_id: &str) -> Result<(), &'static str> {
    let uid = user_id.trim();
    if uid.is_empty() {
        return Err("empty user id");
    }
    let ok = INSTANCES.with(|v| {
        if let Some(i) = v.iter_mut().find(|i| i.name == name) {
            if !i.allow_from.iter().any(|a| a == uid) {
                i.allow_from.push(uid.to_string());
            }
            true
        } else {
            false
        }
    });
    if !ok {
        return Err("no such channel");
    }
    save();
    Ok(())
}

/// Approve a pending pairing code for `name`. Returns the approved sender id.
pub fn pair_approve(name: &str, code: &str) -> Result<String, &'static str> {
    let code = code.trim().to_ascii_uppercase();
    INSTANCES.with(|v| {
        let i = v.iter_mut().find(|i| i.name == name).ok_or("no such channel")?;
        let (c, uid, _disp) = i.pending_pair.clone().ok_or("no pending pair")?;
        if c != code {
            return Err("bad pairing code");
        }
        if !i.allow_from.iter().any(|a| a == &uid) {
            i.allow_from.push(uid.clone());
        }
        i.pending_pair = None;
        // Persist after release — call save outside.
        Ok(uid)
    })
    .map(|uid| {
        save();
        uid
    })
}

/// Send text to `peer` on channel `name`.
pub fn send(name: &str, peer: &str, text: &str) -> Result<(), String> {
    let inst = INSTANCES
        .with(|v| v.iter().find(|i| i.name == name).cloned())
        .ok_or_else(|| String::from("no such channel"))?;
    match inst.kind {
        Kind::Telegram => with_busy(|| telegram::send_message(&inst.token, peer, text))
            .unwrap_or_else(|| Err(String::from("channel busy (try again)"))),
    }
}

/// Reply to the last inbound peer on `name`.
pub fn reply(name: &str, text: &str) -> Result<(), String> {
    let peer = INSTANCES
        .with(|v| {
            v.iter()
                .find(|i| i.name == name)
                .and_then(|i| i.last_peer.clone())
        })
        .ok_or_else(|| String::from("no last peer (receive a message first)"))?;
    send(name, &peer, text)
}

/// Pop one queued inbound message (for the shell agent loop).
pub fn take_inbound() -> Option<Inbound> {
    INBOUND.with(|q| {
        if q.is_empty() {
            None
        } else {
            Some(q.remove(0))
        }
    })
}

/// How many inbound messages are waiting.
pub fn inbound_len() -> usize {
    INBOUND.with(|q| q.len())
}

/// Cooperative poll of every running channel. Call from `shell::upkeep`.
///
/// **Must not re-enter:** HTTP wait loops call `upkeep`, which calls `tick`.
/// Nested entry is a no-op. Polls are also **rate-limited** so the prompt stays
/// interactive between Telegram round-trips. Config is only written when the
/// offset or error state changes (avoids rewrite-on-sync thrash on ext4).
///
/// **Skips while the user is typing** — a multi-second HTTPS `getUpdates` on
/// every idle pulse made the composer feel laggy after every keystroke.
pub fn tick() {
    use core::sync::atomic::Ordering;
    let now = crate::arch::now_ms();
    // If a key was pressed recently, yield the CPU to the line editor.
    let key_age = now.saturating_sub(crate::console::input_activity_ms());
    if key_age < 400 {
        return;
    }
    let last = LAST_POLL_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < POLL_INTERVAL_MS {
        return;
    }
    poll_wave(/*force*/ false);
}

/// Force one poll round now (ignores the rate-limit interval). Used by
/// `/channel poll` and after `allow` so pending Telegram updates are fetched.
pub fn poll_now(only_name: Option<&str>) {
    poll_wave_named(only_name, /*force*/ true);
}

fn poll_wave(force: bool) {
    poll_wave_named(None, force);
}

fn poll_wave_named(only_name: Option<&str>, force: bool) {
    use core::sync::atomic::Ordering;
    // If something else holds BUSY (e.g. start's getMe), skip this pulse.
    if BUSY.load(Ordering::Acquire) {
        return;
    }

    let names: Vec<String> = INSTANCES.with(|v| {
        v.iter()
            .filter(|i| i.running)
            .filter(|i| only_name.map(|n| n == i.name).unwrap_or(true))
            .map(|i| i.name.clone())
            .collect()
    });
    if names.is_empty() {
        return;
    }

    if BUSY
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    if !force {
        // only rate-limit background ticks; force always records "now"
    }
    LAST_POLL_MS.store(crate::arch::now_ms(), Ordering::Relaxed);

    let mut dirty = false;
    for name in names {
        let Some(mut inst) = INSTANCES.with(|v| v.iter().find(|i| i.name == name).cloned()) else {
            continue;
        };
        let prev_offset = inst.offset;
        let prev_err = inst.last_error.clone();
        let result = match inst.kind {
            Kind::Telegram => telegram::poll(&mut inst),
        };
        match result {
            Ok(msgs) => {
                if !msgs.is_empty() {
                    crate::ktrace::log_fmt(format_args!(
                        "msgchan: {} got {} update(s)",
                        name,
                        msgs.len()
                    ));
                }
                for m in msgs {
                    handle_inbound(&mut inst, m);
                }
                inst.last_error = None;
            }
            Err(e) => {
                if inst.last_error.as_deref() != Some(e.as_str()) {
                    crate::ktrace::log_fmt(format_args!("msgchan: {} poll: {e}", name));
                    crate::serial_println!("channel[{name}] poll error: {e}");
                }
                inst.last_error = Some(e);
            }
        }
        if inst.offset != prev_offset || inst.last_error != prev_err {
            dirty = true;
        }
        INSTANCES.with(|v| {
            if let Some(slot) = v.iter_mut().find(|i| i.name == name) {
                *slot = inst;
            }
        });
    }
    BUSY.store(false, Ordering::Release);

    if dirty {
        save();
    }
}

fn handle_inbound(inst: &mut Instance, msg: telegram::TgMessage) {
    let from_id = msg.from_id;
    let from_name = msg.from_name;
    let peer = msg.chat_id;
    let text = msg.text;

    // Access control.
    let allowed = match inst.policy {
        DmPolicy::Open => inst.allow_from.iter().any(|a| a == "*") || inst.allow_from.is_empty(),
        DmPolicy::Allowlist => inst.allow_from.iter().any(|a| a == &from_id || a == "*"),
        DmPolicy::Pairing => {
            if inst.allow_from.iter().any(|a| a == &from_id || a == "*") {
                true
            } else {
                // Issue a pairing code once.
                if inst.pending_pair.is_none() {
                    let code = pair_code(&from_id);
                    inst.pending_pair = Some((code.clone(), from_id.clone(), from_name.clone()));
                    let notice = format!(
                        "Chitti pairing code: {code}\nOn the console: /channel pair {} {code}",
                        inst.name
                    );
                    let _ = telegram::send_message(&inst.token, &peer, &notice);
                    crate::serial_println!(
                        "channel[{}]: pairing request from {} ({}) code={}",
                        inst.name,
                        from_name,
                        from_id,
                        code
                    );
                }
                false
            }
        }
    };
    if !allowed {
        return;
    }

    inst.last_peer = Some(peer.clone());
    crate::serial_println!(
        "channel[{}/{}]: {} ({}): {}",
        inst.name,
        inst.kind.as_str(),
        from_name,
        from_id,
        text
    );

    // Built-in remote commands (no agent).
    let t = text.trim();
    if t.eq_ignore_ascii_case("/ping") || t.eq_ignore_ascii_case("ping") {
        let _ = telegram::send_message(&inst.token, &peer, "pong — Chitti OS channel is live");
        return;
    }
    if t.eq_ignore_ascii_case("/whoami") {
        let body = format!("you are {from_name} id={from_id} chat={peer}");
        let _ = telegram::send_message(&inst.token, &peer, &body);
        return;
    }
    if t.eq_ignore_ascii_case("/help") {
        let _ = telegram::send_message(
            &inst.token,
            &peer,
            "Chitti channel commands:\n/ping — liveness\n/whoami — your ids\n/help — this text\n\nOther messages go to the shell agent when auto_agent is on.",
        );
        return;
    }

    if inst.auto_agent && !t.is_empty() {
        let queued = INBOUND.with(|q| {
            if q.len() < 32 {
                q.push(Inbound {
                    channel: inst.name.clone(),
                    kind: inst.kind,
                    from_id,
                    from_name,
                    peer_id: peer.clone(),
                    text: text.clone(),
                });
                true
            } else {
                false
            }
        });
        if queued {
            crate::ktrace::log_fmt(format_args!(
                "msgchan: queued for agent (channel={} peer={})",
                inst.name, peer
            ));
            // Immediate ack so Telegram users know the bot is working while
            // the shell wakes and runs inference (can take seconds).
            let _ = telegram::send_message(
                &inst.token,
                &peer,
                "… Chitti is thinking",
            );
        } else {
            let _ = telegram::send_message(
                &inst.token,
                &peer,
                "Chitti is busy (inbound queue full). Try again in a moment.",
            );
        }
    }
}

fn pair_code(seed: &str) -> String {
    // Short deterministic-looking code from time + id (not crypto).
    let t = crate::arch::now_ms();
    let mut h = t ^ 0x9e37_79b9;
    for b in seed.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u64);
    }
    format!("{:04X}", (h % 0x10000) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn kind_and_policy_parse() {
        assert_eq!(Kind::parse("tg"), Some(Kind::Telegram));
        assert_eq!(Kind::parse("telegram"), Some(Kind::Telegram));
        assert!(Kind::parse("discord").is_none());
        assert_eq!(DmPolicy::parse("pairing"), Some(DmPolicy::Pairing));
        assert_eq!(DmPolicy::parse("open"), Some(DmPolicy::Open));
    }

    #[test_case]
    fn add_list_remove_roundtrip() {
        // Isolate: wipe instances.
        INSTANCES.with(|v| v.clear());
        assert!(add("home", Kind::Telegram, "123:abc", DmPolicy::Pairing).is_ok());
        assert!(add("home", Kind::Telegram, "x", DmPolicy::Pairing).is_err());
        let all = list();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "home");
        assert!(remove("home").is_ok());
        assert!(list().is_empty());
    }
}
