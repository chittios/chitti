//! **Generic package UI runtime** — load an agent's `assets/tools.wasm`, own a
//! surface, forward tool calls, ticks, clicks, and keys to guest exports. **No
//! app-specific logic** lives here; that is entirely in the package WASM +
//! SOUL.md.
//!
//! ## State model
//! The running app holds **one persistent [`Session`]** (a live wasm instance):
//! game state lives in the guest's own statics and survives across calls —
//! snake's body keeps moving between `tick`s, minesweeper's field survives
//! between clicks. (The old fresh-instance-per-call path reset the guest memory
//! on every call, so no app could hold state at all — snake never moved.) The
//! guest bump allocator resets itself at the start of each host call cycle, so
//! the persistent instance does not leak its heap. Chat tool calls on a package
//! whose UI is **not** running still use a fresh instance (stateless tools).
//!
//! ## Input
//! The compositor pushes clicks on the app's surface into its Synapse event
//! queue; [`tick`] drains them into the guest `on_click` export. The shell
//! routes keys (arrows + printables) for the focused surface into [`nav`] /
//! [`key`], forwarded as the guest `on_key` export. Both are generic: the wasm
//! decides what a click or key means for its app.
//!
//! ## The model-ask protocol (generic agent turns)
//! Any guest export may return **`ask:<prompt>`**: the runtime then runs ONE
//! model turn — the agent's own `SOUL.md` as system prompt, the wasm-built
//! prompt as the user message, no tools — and hands the reply text back via the
//! guest `on_reply {app, text}` export. The wasm builds the prompt (e.g. chess
//! enumerates its legal moves natively) and validates whatever comes back, so
//! the model only ever *chooses*; deterministic package code decides what that
//! choice means. While an ask is in flight the runtime drops surface input and
//! keeps ticking the guest (loader animation); nested asks are refused.

use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::agent::wasm_rt::{self, HostBindings, Limits, Session};
use crate::cap::Right;
use crate::mm::Locked;
use crate::sched::TaskId;
use alloc::format;
use alloc::string::{String, ToString};
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

struct Running {
    name: String,
    agent_id: u64,
    task: TaskId,
    surface: u32,
    /// The live wasm instance — guest statics persist for the app's lifetime.
    session: Session,
    /// Per-call fuel refilled before every export call.
    fuel: u64,
}

static RUN: Locked<Option<Running>> = Locked::new(None);
static ACTIVE_SURFACE: AtomicU32 = AtomicU32::new(0);
static LAST_TICK_MS: AtomicU64 = AtomicU64::new(0);
/// A model ask is in flight: surface input is dropped; nested asks refused.
static ASKING: AtomicBool = AtomicBool::new(false);

/// Fuel for one guest export call (refilled per call on the live session).
const CALL_FUEL: u64 = 2_000_000;

fn ui_caps() -> alloc::vec::Vec<Right> {
    let reqs = [CapabilityRequest::new(
        CapDomain::Ui,
        Rights::EXEC | Rights::WRITE | Rights::DELETE,
        Scope::Any,
    )];
    crate::agent::manifest::primitives_for(&reqs)
        .into_iter()
        .map(Right::InvokePrimitive)
        .collect()
}

fn syn_exec(task: TaskId, raw: &str) -> String {
    match crate::synapse::execute(task, raw) {
        crate::synapse::Invocation::Executed { result, .. } => result,
        other => format!("{other:?}"),
    }
}

/// Surface id for the running package UI, if any.
pub fn active_surface() -> Option<u32> {
    let id = ACTIVE_SURFACE.load(Ordering::Relaxed);
    if id == 0 {
        None
    } else {
        Some(id)
    }
}

pub fn owns_surface(id: u32) -> bool {
    active_surface() == Some(id)
}

pub fn is_running() -> bool {
    RUN.with(|r| r.is_some())
}

pub fn running_name() -> Option<String> {
    RUN.with(|r| r.as_ref().map(|x| x.name.clone()))
}

/// Host bindings for the active package UI (or agent-only defaults).
pub fn bindings_for(agent_id: u64) -> HostBindings {
    RUN.with(|r| {
        if let Some(run) = r.as_ref() {
            if run.agent_id == agent_id {
                return HostBindings {
                    agent_id,
                    task: run.task,
                    surface: run.surface,
                };
            }
        }
        HostBindings {
            agent_id,
            task: 0,
            surface: 0,
        }
    })
}

/// Invoke a guest export on `agent_id`'s `assets/tools.wasm`.
///
/// When the package UI is running for this agent the call goes through its
/// **live session** — required so tool calls see (and mutate) the running app's
/// state. Otherwise a fresh instance runs the (stateless) tool.
pub fn call_agent_export(agent_id: u64, export: &str, args_json: &str) -> Result<String, &'static str> {
    let live = RUN.with(|r| {
        if let Some(run) = r.as_mut() {
            if run.agent_id == agent_id {
                run.session.set_fuel(run.fuel)?;
                return run.session.call_string(export, args_json).map(Some);
            }
        }
        Ok(None)
    })?;
    if let Some(out) = live {
        return Ok(out);
    }
    let path = format!("{}/assets/tools.wasm", crate::agent::home::path(agent_id));
    let wasm = crate::synapse::fs::read(&path).ok_or("tools.wasm missing")?;
    if wasm.is_empty() {
        return Err("tools.wasm empty");
    }
    wasm_rt::call_string_bound(
        &wasm,
        export,
        args_json,
        Limits::default().with_fuel(CALL_FUEL),
        bindings_for(agent_id),
    )
}

/// Call a guest export on the RUNNING app (no-op error when none).
fn call_running(export: &str, args_json: &str) -> Result<String, &'static str> {
    RUN.with(|r| {
        let run = r.as_mut().ok_or("no package UI running")?;
        run.session.set_fuel(run.fuel)?;
        run.session.call_string(export, args_json)
    })
}

/// Process a guest export's result: an `ask:<prompt>` runs one model turn and
/// feeds the text back via `on_reply` (see the module doc); anything else
/// passes through untouched.
fn handle_result(out: String) -> String {
    let Some(prompt) = out.strip_prefix("ask:") else {
        return out;
    };
    if ASKING.swap(true, Ordering::SeqCst) {
        crate::ktrace::log("package_ui", "nested model ask refused");
        return out;
    }
    let (name, agent_id, surface) = match RUN.with(|r| {
        r.as_ref().map(|x| (x.name.clone(), x.agent_id, x.surface))
    }) {
        Some(x) => x,
        None => {
            ASKING.store(false, Ordering::SeqCst);
            return out;
        }
    };
    // The agent's own persona is the system prompt; the wasm built the user
    // message (and will validate the reply — the model only chooses).
    let soul_path = format!("{}/SOUL.md", crate::agent::home::path(agent_id));
    let soul = crate::synapse::fs::read(&soul_path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_else(|| String::from("You are a package app agent. Answer the request precisely."));
    crate::ktrace::log_fmt(format_args!("package_ui: model ask from '{name}' ({} chars)", prompt.len()));
    let reply = crate::shell::ui_agent_reply(&soul, prompt, surface, |_cmd, _args| {
        // No tools during an ask — the wasm owns all state transitions.
        String::from("error:tools are unavailable here - answer directly")
    });
    ASKING.store(false, Ordering::SeqCst);
    let text = reply.unwrap_or_default();
    // Keep the reply JSON-safe for the guest's simple parser: strip quotes,
    // backslashes, and control characters (apps match plain substrings).
    let clean: String = text
        .chars()
        .map(|c| if c == '"' || c == '\\' || (c as u32) < 0x20 { ' ' } else { c })
        .collect();
    let args = format!(r#"{{"app":"{name}","text":"{clean}"}}"#);
    match call_running("on_reply", &args) {
        Ok(r2) if r2.starts_with("ask:") => {
            crate::ktrace::log("package_ui", "on_reply returned another ask; refused");
            r2
        }
        Ok(r2) => {
            crate::serial_println!("package_ui> {name}: {r2}");
            r2
        }
        Err(e) => {
            crate::serial_println!("package_ui> {name} on_reply: {e}");
            String::from("error:on_reply failed")
        }
    }
}

pub fn stop() {
    if let Some(run) = RUN.with(|r| r.take()) {
        ACTIVE_SURFACE.store(0, Ordering::Relaxed);
        let raw = format!(
            r#"{{"name":"ui_surface_close","arguments":{{"surface":{}}}}}"#,
            run.surface
        );
        let _ = syn_exec(run.task, &raw);
        let _ = crate::sched::kill(run.task);
        crate::serial_println!("package_ui> stopped '{}'", run.name);
    }
}

/// Start package UI for a system agent name that ships `assets/tools.wasm`
/// (`chess`, `paint`, `calc`, `files`, `game2048`, …). Init calls `{name}_start`
/// (or historical aliases like `mines_start`).
pub fn start(name: &str) -> Result<u32, &'static str> {
    stop();
    let home = crate::agent::system::home_for(name).ok_or("unknown package agent")?;
    let agent_id = home
        .rsplit('/')
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or("bad agent home")?;
    let wasm_path = format!("{home}/assets/tools.wasm");
    let wasm = crate::synapse::fs::read(&wasm_path).ok_or("package has no tools.wasm")?;

    let task_name: &'static str = match name {
        "paint" => "pkg-paint",
        "slides" => "pkg-slides",
        "minesweeper" => "pkg-mines",
        "snake" => "pkg-snake",
        "synth" => "pkg-synth",
        other => alloc::boxed::Box::leak(format!("pkg-{other}").into_boxed_str()),
    };
    let task = crate::arch::interrupts::without_interrupts(|| {
        let task = crate::sched::spawn_parked(task_name);
        for r in ui_caps() {
            crate::cap::grant(task, r);
        }
        task
    });
    let out = syn_exec(
        task,
        r#"{"name":"ui_surface_request","arguments":{"kind":"canvas"}}"#,
    );
    let surface = out
        .strip_prefix("ok:surface=")
        .and_then(|s| s.trim().parse().ok())
        .ok_or("ui_surface_request failed")?;

    // One persistent instance for the app's whole run: guest statics ARE the
    // game state (board, snake body, brush colour).
    let bind = HostBindings {
        agent_id,
        task,
        surface,
    };
    let session = Session::instantiate(&wasm, Limits::default().with_fuel(CALL_FUEL), bind)
        .map_err(|e| {
            let raw = format!(
                r#"{{"name":"ui_surface_close","arguments":{{"surface":{surface}}}}}"#
            );
            let _ = syn_exec(task, &raw);
            let _ = crate::sched::kill(task);
            e
        })?;

    ACTIVE_SURFACE.store(surface, Ordering::Relaxed);
    RUN.with(|r| {
        *r = Some(Running {
            name: name.into(),
            agent_id,
            task,
            surface,
            session,
            fuel: CALL_FUEL,
        });
    });

    // Guest init export: most packages export `{name}_start`; a few keep
    // historical aliases (minesweeper → mines_start). The generic `model` flag
    // tells the app whether the runtime can serve `ask:` (e.g. chess opponent).
    let model = crate::shell::planner_available() || crate::shell::remote::is_remote_active();
    let init_args = format!(r#"{{"app":"{name}","surface":{surface},"model":{model}}}"#);
    let start_owned = match name {
        "minesweeper" => String::from("mines_start"),
        "chess" => String::from("chess_start"),
        // Hyphenated package names cannot be wasm export identifiers.
        "sandbox-lab" => String::from("sandbox_start"),
        other => format!("{other}_start"),
    };
    let init = call_running(&start_owned, &init_args)
        .or_else(|_| call_running("app_start", &init_args));
    match init {
        Ok(s) => {
            let s = handle_result(s);
            crate::serial_println!("package_ui> {name}: {s}");
        }
        Err(e) => crate::serial_println!("package_ui> {name} init: {e}"),
    }
    Ok(surface)
}

/// Parse a Synapse `ui_event_poll` result (`ok:click=X,Y`) into coordinates.
/// Pure — unit-tested.
pub fn parse_click_event(out: &str) -> Option<(u16, u16)> {
    let rest = out.strip_prefix("ok:click=")?;
    let mut xy = rest.split(',');
    let x = xy.next()?.trim().parse().ok()?;
    let y = xy.next()?.trim().parse().ok()?;
    Some((x, y))
}

/// CSI arrow final byte → the key name the guest `on_key` export receives.
/// Pure — unit-tested.
pub fn nav_key_name(fin: u8) -> Option<&'static str> {
    match fin {
        b'A' => Some("up"),
        b'B' => Some("down"),
        b'C' => Some("right"),
        b'D' => Some("left"),
        _ => None,
    }
}

/// Upkeep: drain surface clicks into the guest `on_click` export (every call),
/// and run the optional guest `tick` export (throttled — snake's step rate /
/// the thinking-loader animation). While a model ask is in flight, clicks are
/// discarded (the guest also guards) and only the tick animation runs.
pub fn tick() {
    let (name, task, surface) = match RUN.with(|r| {
        r.as_ref().map(|x| (x.name.clone(), x.task, x.surface))
    }) {
        Some(x) => x,
        None => return,
    };
    let asking = ASKING.load(Ordering::Relaxed);
    // Clicks: unthrottled so input feels immediate. Drain via the audited
    // primitive only when the native peek says events exist (audit-log safe).
    if crate::synapse::ui::has_events(surface) {
        for _ in 0..8 {
            let raw = format!(r#"{{"name":"ui_event_poll","arguments":{{"surface":{surface}}}}}"#);
            let out = syn_exec(task, &raw);
            if let Some((x, y)) = parse_click_event(&out) {
                if asking {
                    continue; // discard input while the model is thinking
                }
                let args = format!(r#"{{"app":"{name}","x":{x},"y":{y}}}"#);
                if let Ok(out) = call_running("on_click", &args) {
                    let _ = handle_result(out);
                }
            } else {
                break; // ok:none / key (shell routes keys directly) / error
            }
        }
    }
    // Guest tick (snake steps, thinking dots): throttled to the app step rate.
    let now = crate::arch::now_ms();
    if now.saturating_sub(LAST_TICK_MS.load(Ordering::Relaxed)) < 180 {
        return;
    }
    LAST_TICK_MS.store(now, Ordering::Relaxed);
    if let Ok(out) = call_running("tick", &format!(r#"{{"app":"{name}"}}"#)) {
        if !asking {
            let _ = handle_result(out);
        }
    }
}

/// Forward a named key to the running app's `on_key`. Returns true only when
/// the app actually HANDLED it (any reply other than the bare `"ok"` the guest
/// dispatch returns for keys it ignores) — unhandled keys fall through to the
/// normal shell behaviour, so an app never swallows typing it has no use for.
fn forward_key(key: &str) -> bool {
    let name = match RUN.with(|r| r.as_ref().map(|x| x.name.clone())) {
        Some(n) => n,
        None => return false,
    };
    if ASKING.load(Ordering::Relaxed) {
        return true; // swallow input while the model is thinking
    }
    let args = format!(r#"{{"app":"{name}","key":"{key}"}}"#);
    match call_running("on_key", &args) {
        Ok(out) => handle_result(out) != "ok",
        Err(_) => false,
    }
}

/// Forward an arrow key (CSI final byte) to the running app. Returns true if
/// the app handled it (snake steers, mines moves its cursor, slides navigate).
pub fn nav(fin: u8) -> bool {
    match nav_key_name(fin) {
        Some(key) => forward_key(key),
        None => false,
    }
}

/// Forward a printable / Enter / Esc key to the running app. Returns true if
/// the app handled it. Control chords (Ctrl+C etc.) never reach here — they
/// stay with the shell.
pub fn key(c: u8) -> bool {
    let key: String = match c {
        b'\r' | b'\n' => "enter".to_string(),
        b' ' => "space".to_string(),
        0x1b => "esc".to_string(),
        0x21..=0x7e => (c as char).to_string(),
        _ => return false,
    };
    forward_key(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn click_event_parses_coords_and_rejects_noise() {
        assert_eq!(parse_click_event("ok:click=12,34"), Some((12, 34)));
        assert_eq!(parse_click_event("ok:click=0,191"), Some((0, 191)));
        assert_eq!(parse_click_event("ok:none"), None);
        assert_eq!(parse_click_event("ok:key=13"), None);
        assert_eq!(parse_click_event("error:no events"), None);
        assert_eq!(parse_click_event("ok:click=12"), None);
        assert_eq!(parse_click_event("ok:click=a,b"), None);
    }

    #[test_case]
    fn nav_final_bytes_map_to_key_names() {
        assert_eq!(nav_key_name(b'A'), Some("up"));
        assert_eq!(nav_key_name(b'B'), Some("down"));
        assert_eq!(nav_key_name(b'C'), Some("right"));
        assert_eq!(nav_key_name(b'D'), Some("left"));
        assert_eq!(nav_key_name(b'Z'), None);
    }
}
