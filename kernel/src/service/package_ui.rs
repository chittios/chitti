//! **Generic package UI runtime** — load an agent's `assets/tools.wasm`, own a
//! surface, forward tool calls, ticks, clicks, and keys to guest exports. **No
//! app-specific logic** lives here; that is entirely in the package WASM +
//! SOUL.md.
//!
//! ## State model
//! **Multiple apps run in parallel.** Each running package holds **one
//! persistent [`Session`]** (a live wasm instance) keyed by surface id: game
//! state lives in the guest's own statics and survives across calls — snake's
//! body keeps moving between `tick`s, minesweeper's field survives between
//! clicks. Starting a new package does **not** stop the others; each gets its
//! own action-pane tab. Closing a tab (Ctrl+W / `[x]`) kills only that agent.
//!
//! The guest bump allocator resets itself at the start of each host call
//! cycle, so the persistent instance does not leak its heap. Chat tool calls
//! on a package whose UI is **not** running still use a fresh instance
//! (stateless tools).
//!
//! ## Input
//! The compositor pushes clicks on each app's surface into its Synapse event
//! queue; [`tick`] drains them into that guest's `on_click` export. The shell
//! routes keys for the **focused** surface into [`nav`] / [`key`].
//!
//! ## The model-ask protocol (generic agent turns)
//! Any guest export may return **`ask:<prompt>`**: the runtime then runs ONE
//! model turn — the agent's own `SOUL.md` as system prompt, the wasm-built
//! prompt as the user message, no tools — and hands the reply text back via the
//! guest `on_reply {app, text}` export. Nested asks (globally) are refused
//! while one is in flight.

use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::agent::wasm_rt::{self, HostBindings, Limits, Session};
use crate::cap::Right;
use crate::mm::Locked;
use crate::sched::TaskId;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
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
    /// Model ask in flight for this app: surface input is dropped.
    asking: bool,
}

/// All live package-UI apps, keyed by surface id.
static APPS: Locked<BTreeMap<u32, Running>> = Locked::new(BTreeMap::new());
/// Tab titles (surface → agent name). Safe to read while a session is taken out
/// of [`APPS`] for a guest call — never requires the session map.
static TITLES: Locked<BTreeMap<u32, String>> = Locked::new(BTreeMap::new());
/// Surfaces the user dismissed (Ctrl+W) while a guest export was still running.
/// Blocks compositor presents so the tab cannot reopen; teardown finishes when
/// the export returns.
static STOP_PENDING: Locked<alloc::vec::Vec<u32>> = Locked::new(alloc::vec::Vec::new());
/// Most recently started / focused package surface (for stop() without a tab).
static FOCUS_SURFACE: AtomicU32 = AtomicU32::new(0);
static LAST_TICK_MS: AtomicU64 = AtomicU64::new(0);
/// True while any guest export is running (prevents re-entrant map take-out).
static IN_GUEST: AtomicBool = AtomicBool::new(false);
/// Global gate: only one model ask at a time (ui_agent_reply is shell-global).
static ASKING_GLOBAL: AtomicBool = AtomicBool::new(false);

/// Fuel for one guest export call (refilled per call on the live session).
const CALL_FUEL: u64 = 2_000_000;

fn set_title(surface: u32, name: &str) {
    TITLES.with(|t| {
        t.insert(surface, name.into());
    });
    FOCUS_SURFACE.store(surface, Ordering::Relaxed);
}

fn clear_title(surface: u32) {
    TITLES.with(|t| {
        t.remove(&surface);
    });
    if FOCUS_SURFACE.load(Ordering::Relaxed) == surface {
        // Fall back to another live title, if any.
        let next = TITLES.with(|t| t.keys().next().copied().unwrap_or(0));
        FOCUS_SURFACE.store(next, Ordering::Relaxed);
    }
}

fn mark_stop_pending(surface: u32) {
    STOP_PENDING.with(|v| {
        if !v.iter().any(|&x| x == surface) {
            v.push(surface);
        }
    });
}

fn take_stop_pending(surface: u32) -> bool {
    STOP_PENDING.with(|v| {
        if let Some(i) = v.iter().position(|&x| x == surface) {
            v.remove(i);
            true
        } else {
            false
        }
    })
}

fn is_stop_pending(surface: u32) -> bool {
    STOP_PENDING.with(|v| v.iter().any(|&x| x == surface))
}

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
    // Package UI host path: effects are ownership-gated UI primitives only
    // (surface draw/event/close), not FS/net. Justification is system-trusted
    // because the call is native code below the determinism boundary, not model
    // output — the model never invents these strings.
    match crate::synapse::execute_with_justification(
        task,
        raw,
        crate::security::Justification::trusted(),
    ) {
        crate::synapse::Invocation::Executed { result, .. } => result,
        other => format!("{other:?}"),
    }
}

/// Most recently focused package-UI surface, if any.
pub fn active_surface() -> Option<u32> {
    let id = FOCUS_SURFACE.load(Ordering::Relaxed);
    if id == 0 || !owns_surface(id) {
        // FOCUS may lag after close — pick any live app.
        TITLES.with(|t| t.keys().next().copied())
    } else {
        Some(id)
    }
}

/// True if `id` is a live package-UI surface (tab title still registered).
pub fn owns_surface(id: u32) -> bool {
    TITLES.with(|t| t.contains_key(&id))
}

/// Whether compositor presents for this package surface should reach the FB.
/// False after the user closes that canvas so a late guest draw cannot reopen it.
pub fn should_present(id: u32) -> bool {
    !is_stop_pending(id) && owns_surface(id)
}

/// Tab title for package-UI surface `id` (agent name). Safe to call from the
/// compositor while it holds `SCREEN` — does **not** touch [`APPS`].
pub fn surface_tab_name(id: u32) -> Option<String> {
    TITLES.with(|t| t.get(&id).cloned())
}

pub fn is_running() -> bool {
    TITLES.with(|t| !t.is_empty()) || APPS.with(|m| !m.is_empty())
}

/// Name of the focused package UI, if any (Agents catalogue "run" badge).
pub fn running_name() -> Option<String> {
    let id = active_surface()?;
    surface_tab_name(id)
}

/// All live package-UI agent names (for catalogue badges).
pub fn running_names() -> Vec<String> {
    TITLES.with(|t| t.values().cloned().collect())
}

pub fn is_name_running(name: &str) -> bool {
    TITLES.with(|t| t.values().any(|n| n == name))
}

/// Host bindings for a package agent (or defaults when its UI is not running).
pub fn bindings_for(agent_id: u64) -> HostBindings {
    APPS.with(|m| {
        if let Some(run) = m.values().find(|r| r.agent_id == agent_id) {
            return HostBindings {
                agent_id,
                task: run.task,
                surface: run.surface,
            };
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
/// When that agent's package UI is running the call goes through its **live
/// session**. Otherwise a fresh instance runs the (stateless) tool.
pub fn call_agent_export(agent_id: u64, export: &str, args_json: &str) -> Result<String, &'static str> {
    if !IN_GUEST.load(Ordering::Relaxed) {
        let taken = APPS.with(|m| {
            let sid = m.values().find(|r| r.agent_id == agent_id).map(|r| r.surface);
            sid.and_then(|s| m.remove(&s))
        });
        if let Some(run) = taken {
            return finish_guest_call(run, export, args_json);
        }
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

/// Call a guest export on the app that owns `surface`.
fn call_surface(surface: u32, export: &str, args_json: &str) -> Result<String, &'static str> {
    if IN_GUEST.load(Ordering::Relaxed) {
        return Err("reentrant package_ui call");
    }
    let run = APPS.with(|m| m.remove(&surface)).ok_or("no package UI for surface")?;
    finish_guest_call(run, export, args_json)
}

/// Run one guest export with the session taken out of [`APPS`]; honour a
/// per-surface stop requested mid-call.
fn finish_guest_call(
    mut run: Running,
    export: &str,
    args_json: &str,
) -> Result<String, &'static str> {
    let surface = run.surface;
    if !is_stop_pending(surface) {
        FOCUS_SURFACE.store(surface, Ordering::Relaxed);
    }
    IN_GUEST.store(true, Ordering::SeqCst);
    let result = (|| {
        run.session.set_fuel(run.fuel)?;
        run.session.call_string(export, args_json)
    })();
    IN_GUEST.store(false, Ordering::SeqCst);

    if take_stop_pending(surface) || is_stop_pending(surface) {
        // User closed this canvas during the call — tear down now.
        let _ = take_stop_pending(surface);
        clear_title(surface);
        drop_running(run);
        return result;
    }
    APPS.with(|m| {
        m.insert(surface, run);
    });
    result
}

fn drop_running(run: Running) {
    let raw = format!(
        r#"{{"name":"ui_surface_close","arguments":{{"surface":{}}}}}"#,
        run.surface
    );
    let _ = syn_exec(run.task, &raw);
    let _ = crate::sched::kill(run.task);
    crate::serial_println!("package_ui> stopped '{}'", run.name);
}

/// Process a guest export's result for the app on `surface`.
fn handle_result(surface: u32, out: String) -> String {
    let Some(prompt) = out.strip_prefix("ask:") else {
        return out;
    };
    if ASKING_GLOBAL.swap(true, Ordering::SeqCst) {
        crate::ktrace::log("package_ui", "nested model ask refused");
        return out;
    }
    let meta = APPS.with(|m| {
        m.get(&surface)
            .map(|x| (x.name.clone(), x.agent_id, x.surface))
    });
    let Some((name, agent_id, surface)) = meta else {
        ASKING_GLOBAL.store(false, Ordering::SeqCst);
        return out;
    };
    APPS.with(|m| {
        if let Some(r) = m.get_mut(&surface) {
            r.asking = true;
        }
    });
    let soul_path = format!("{}/SOUL.md", crate::agent::home::path(agent_id));
    let soul = crate::synapse::fs::read(&soul_path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_else(|| {
            String::from("You are a package app agent. Answer the request precisely.")
        });
    crate::ktrace::log_fmt(format_args!(
        "package_ui: model ask from '{name}' ({} chars)",
        prompt.len()
    ));
    let reply = crate::shell::ui_agent_reply(&soul, prompt, surface, |_cmd, _args| {
        String::from("error:tools are unavailable here - answer directly")
    });
    ASKING_GLOBAL.store(false, Ordering::SeqCst);
    APPS.with(|m| {
        if let Some(r) = m.get_mut(&surface) {
            r.asking = false;
        }
    });
    let text = reply.unwrap_or_default();
    let clean: String = text
        .chars()
        .map(|c| {
            if c == '"' || c == '\\' || (c as u32) < 0x20 {
                ' '
            } else {
                c
            }
        })
        .collect();
    let args = format!(r#"{{"app":"{name}","text":"{clean}"}}"#);
    match call_surface(surface, "on_reply", &args) {
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

/// Stop the package UI on `surface` (kill agent, close Synapse surface).
/// Returns the surface id if something was stopped.
pub fn stop_surface(surface: u32) -> Option<u32> {
    if !owns_surface(surface) && !APPS.with(|m| m.contains_key(&surface)) {
        return None;
    }
    // Hide immediately so late presents cannot reopen this tab.
    clear_title(surface);
    APPS.with(|m| {
        if let Some(r) = m.get_mut(&surface) {
            r.asking = false;
        }
    });

    if IN_GUEST.load(Ordering::Relaxed) {
        // Session may be taken out of APPS mid-call — mark pending either way.
        mark_stop_pending(surface);
        crate::serial_println!("package_ui> stop surface={surface} (guest export in flight)");
        // If the session is still in the map (another app's guest is running),
        // tear down now.
        if let Some(run) = APPS.with(|m| m.remove(&surface)) {
            let _ = take_stop_pending(surface);
            drop_running(run);
        }
        return Some(surface);
    }

    let _ = take_stop_pending(surface);
    if let Some(run) = APPS.with(|m| m.remove(&surface)) {
        drop_running(run);
        return Some(surface);
    }
    Some(surface)
}

/// Stop the focused package UI (or the only one). Returns its surface id.
pub fn stop() -> Option<u32> {
    let sid = active_surface().or_else(|| APPS.with(|m| m.keys().next().copied()))?;
    stop_surface(sid)
}

/// Stop every running package UI. Returns the surface ids that were stopped
/// (so the shell can remove their tabs).
pub fn stop_all() -> Vec<u32> {
    let mut ids: Vec<u32> = TITLES.with(|t| t.keys().copied().collect());
    let leftover: Vec<u32> = APPS.with(|m| m.keys().copied().collect());
    for id in leftover {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut stopped = Vec::new();
    for id in ids {
        if stop_surface(id).is_some() {
            stopped.push(id);
        }
    }
    stopped
}

/// Stop the first package UI named `name`, if any.
pub fn stop_named(name: &str) -> Option<u32> {
    let sid = TITLES.with(|t| {
        t.iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(&s, _)| s)
    })?;
    stop_surface(sid)
}

/// Surface id of any live package session (compat — prefer [`owns_surface`]).
pub fn running_surface() -> Option<u32> {
    active_surface().or_else(|| APPS.with(|m| m.keys().next().copied()))
}

/// True if closing this action-pane surface tab should kill a package agent.
pub fn close_kills_agent(tab_surface: u32) -> bool {
    owns_surface(tab_surface) || APPS.with(|m| m.contains_key(&tab_surface))
}

/// Start package UI for a system agent name that ships `assets/tools.wasm`.
///
/// Does **not** stop other running package UIs — apps run in parallel, each
/// with its own tab. If `name` is already running, focuses that instance.
pub fn start(name: &str) -> Result<u32, &'static str> {
    // Focus an existing instance of the same app instead of spawning a twin.
    if let Some(sid) = TITLES.with(|t| {
        t.iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(&s, _)| s)
    }) {
        FOCUS_SURFACE.store(sid, Ordering::Relaxed);
        #[cfg(not(test))]
        {
            crate::framebuffer::set_right(crate::framebuffer::RightMode::Surface(sid));
            crate::framebuffer::focus_set(true);
            crate::synapse::ui::represent(sid);
        }
        crate::serial_println!("package_ui> {name}: already running (surface {sid}) — focused");
        return Ok(sid);
    }

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
    crate::serial_println!(
        "package_ui> {name}: surface request (wasm {} KiB)…",
        wasm.len() / 1024
    );
    let out = syn_exec(
        task,
        r#"{"name":"ui_surface_request","arguments":{"kind":"canvas"}}"#,
    );
    let surface = out
        .strip_prefix("ok:surface=")
        .and_then(|s| s.trim().parse().ok())
        .ok_or("ui_surface_request failed")?;
    crate::serial_println!("package_ui> {name}: surface={surface} — opening action pane…");

    set_title(surface, name);

    // Eager blank present: open the Surface tab *now*, before wasmi instantiate.
    #[cfg(not(test))]
    {
        let blank = format!(
            r#"{{"name":"ui_draw","arguments":{{"surface":{surface},"ops":"clear 1a1a2e"}}}}"#
        );
        let _ = syn_exec(task, &blank);
        crate::framebuffer::focus_set(true);
    }
    crate::serial_println!("package_ui> {name}: pane open, instantiating wasm…");

    let bind = HostBindings {
        agent_id,
        task,
        surface,
    };
    let limits = Limits::default()
        .with_fuel(CALL_FUEL)
        .with_pages(64);
    let session = Session::instantiate(&wasm, limits, bind).map_err(|e| {
        crate::serial_println!("package_ui> {name}: wasm instantiate failed: {e}");
        let raw = format!(
            r#"{{"name":"ui_surface_close","arguments":{{"surface":{surface}}}}}"#
        );
        let _ = syn_exec(task, &raw);
        let _ = crate::sched::kill(task);
        clear_title(surface);
        e
    })?;
    let start_owned = start_export_name(name);
    crate::serial_println!("package_ui> {name}: wasm ready, calling {start_owned}…");

    APPS.with(|m| {
        m.insert(
            surface,
            Running {
                name: name.into(),
                agent_id,
                task,
                surface,
                session,
                fuel: CALL_FUEL,
                asking: false,
            },
        );
    });
    set_title(surface, name);

    let model = crate::shell::planner_available() || crate::shell::remote::is_remote_active();
    let init_args = format!(r#"{{"app":"{name}","surface":{surface},"model":{model}}}"#);
    let mut init_result: Result<String, &'static str> = Err("init not run");
    crate::synapse::ui::with_deferred_present(|| {
        init_result = call_surface(surface, &start_owned, &init_args)
            .or_else(|_| call_surface(surface, "app_start", &init_args));
    });
    match init_result {
        Ok(s) if s.starts_with("ask:") => {
            crate::serial_println!(
                "package_ui> {name}: init deferred model ask; surface {surface} ready"
            );
        }
        Ok(s) => {
            let s = handle_result(surface, s);
            crate::serial_println!("package_ui> {name}: {s}");
        }
        Err(e) => crate::serial_println!("package_ui> {name} init: {e}"),
    }
    #[cfg(not(test))]
    {
        crate::framebuffer::focus_set(true);
        crate::synapse::ui::represent(surface);
    }
    crate::serial_println!("package_ui> {name}: surface {surface} ready");
    Ok(surface)
}

/// Guest init export for a package name. Pure — unit-tested.
pub fn start_export_name(name: &str) -> String {
    match name {
        "minesweeper" => String::from("mines_start"),
        "chess" => String::from("chess_start"),
        "sandbox-lab" => String::from("sandbox_start"),
        other => format!("{other}_start"),
    }
}

/// Parse a Synapse `ui_event_poll` result (`ok:click=X,Y`) into coordinates.
pub fn parse_click_event(out: &str) -> Option<(u16, u16)> {
    let rest = out.strip_prefix("ok:click=")?;
    let mut xy = rest.split(',');
    let x = xy.next()?.trim().parse().ok()?;
    let y = xy.next()?.trim().parse().ok()?;
    Some((x, y))
}

/// CSI arrow final byte → the key name the guest `on_key` export receives.
pub fn nav_key_name(fin: u8) -> Option<&'static str> {
    match fin {
        b'A' => Some("up"),
        b'B' => Some("down"),
        b'C' => Some("right"),
        b'D' => Some("left"),
        _ => None,
    }
}

/// Upkeep: for **every** running package UI, drain surface clicks and run the
/// optional guest `tick` export (throttled).
pub fn tick() {
    if IN_GUEST.load(Ordering::Relaxed) {
        return;
    }
    let apps: Vec<(String, TaskId, u32, bool)> = APPS.with(|m| {
        m.values()
            .map(|x| (x.name.clone(), x.task, x.surface, x.asking))
            .collect()
    });
    if apps.is_empty() {
        return;
    }

    for (name, task, surface, asking) in &apps {
        if is_stop_pending(*surface) {
            continue;
        }
        if crate::synapse::ui::has_events(*surface) {
            for _ in 0..8 {
                let raw = format!(
                    r#"{{"name":"ui_event_poll","arguments":{{"surface":{surface}}}}}"#
                );
                let out = syn_exec(*task, &raw);
                if let Some((x, y)) = parse_click_event(&out) {
                    if *asking {
                        continue;
                    }
                    let args = format!(r#"{{"app":"{name}","x":{x},"y":{y}}}"#);
                    if let Ok(out) = call_surface(*surface, "on_click", &args) {
                        let _ = handle_result(*surface, out);
                    }
                } else {
                    break;
                }
            }
        }
    }

    let now = crate::arch::now_ms();
    if now.saturating_sub(LAST_TICK_MS.load(Ordering::Relaxed)) < 180 {
        return;
    }
    LAST_TICK_MS.store(now, Ordering::Relaxed);

    // Re-read after click handling (map may have changed).
    let apps: Vec<(String, u32, bool)> = APPS.with(|m| {
        m.values()
            .map(|x| (x.name.clone(), x.surface, x.asking))
            .collect()
    });
    for (name, surface, asking) in apps {
        if is_stop_pending(surface) {
            continue;
        }
        if let Ok(out) = call_surface(surface, "tick", &format!(r#"{{"app":"{name}"}}"#)) {
            if !asking {
                let _ = handle_result(surface, out);
            }
        }
    }
}

/// Surface that should receive keyboard input: the focused action-pane package
/// tab, if any.
fn input_target_surface() -> Option<u32> {
    #[cfg(not(test))]
    {
        if let crate::framebuffer::RightMode::Surface(id) = crate::framebuffer::right_mode() {
            if owns_surface(id) {
                FOCUS_SURFACE.store(id, Ordering::Relaxed);
                return Some(id);
            }
        }
    }
    active_surface()
}

fn forward_key(key: &str) -> bool {
    let surface = match input_target_surface() {
        Some(s) => s,
        None => return false,
    };
    let (name, asking) = match APPS.with(|m| {
        m.get(&surface)
            .map(|x| (x.name.clone(), x.asking))
    }) {
        Some(x) => x,
        None => return false,
    };
    if asking {
        return true;
    }
    let args = format!(r#"{{"app":"{name}","key":"{key}"}}"#);
    match call_surface(surface, "on_key", &args) {
        Ok(out) => handle_result(surface, out) != "ok",
        Err(_) => false,
    }
}

/// Forward an arrow key (CSI final byte) to the focused package app.
pub fn nav(fin: u8) -> bool {
    match nav_key_name(fin) {
        Some(key) => forward_key(key),
        None => false,
    }
}

/// Forward a printable / Enter / Esc key to the focused package app.
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

    #[test_case]
    fn start_export_names_cover_all_package_ui_apps() {
        assert_eq!(start_export_name("chess"), "chess_start");
        assert_eq!(start_export_name("minesweeper"), "mines_start");
        assert_eq!(start_export_name("sandbox-lab"), "sandbox_start");
        for name in [
            "paint", "slides", "snake", "synth", "calc", "clock", "files",
            "gallery", "sheets", "calendar", "contacts", "writer", "archive",
            "hex", "game2048", "activity", "weather", "settings", "dict",
            "diff", "breakout", "tetris", "console", "maps", "radio",
        ] {
            assert_eq!(
                start_export_name(name),
                format!("{name}_start"),
                "default export naming for {name}"
            );
        }
    }

    #[test_case]
    fn should_present_tracks_per_surface_titles_and_stop() {
        // Clean slate for this test (other tests may leave titles).
        let leftovers: Vec<u32> = TITLES.with(|t| t.keys().copied().collect());
        for id in leftovers {
            clear_title(id);
        }
        STOP_PENDING.with(|v| v.clear());

        assert!(!should_present(1));
        set_title(7, "chess");
        set_title(8, "paint");
        assert!(should_present(7));
        assert!(should_present(8));
        assert!(!should_present(9));
        mark_stop_pending(7);
        assert!(!should_present(7), "stop pending blocks present for that surface");
        assert!(should_present(8), "other apps still present");
        let _ = take_stop_pending(7);
        clear_title(7);
        clear_title(8);
    }

    #[test_case]
    fn multi_titles_report_parallel_running_names() {
        let leftovers: Vec<u32> = TITLES.with(|t| t.keys().copied().collect());
        for id in leftovers {
            clear_title(id);
        }
        set_title(1, "chess");
        set_title(2, "calc");
        let names = running_names();
        assert!(names.iter().any(|n| n == "chess"));
        assert!(names.iter().any(|n| n == "calc"));
        assert!(is_name_running("chess"));
        assert!(owns_surface(1) && owns_surface(2));
        clear_title(1);
        clear_title(2);
    }
}
