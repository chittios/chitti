//! **Generic package UI runtime** — load an agent's `assets/tools.wasm`, own a
//! surface, forward tool calls and ticks to guest exports. **No app-specific
//! logic** lives here; that is entirely in the package WASM + SOUL.md.

use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::agent::wasm_rt::{self, HostBindings, Limits, Session};
use crate::cap::Right;
use crate::mm::Locked;
use crate::sched::TaskId;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

struct Running {
    name: String,
    agent_id: u64,
    task: TaskId,
    surface: u32,
    /// Raw module bytes (re-instantiate per call so host bind stays fresh).
    wasm: Vec<u8>,
}

static RUN: Locked<Option<Running>> = Locked::new(None);
static ACTIVE_SURFACE: AtomicU32 = AtomicU32::new(0);
static LAST_TICK_MS: AtomicU64 = AtomicU64::new(0);

fn ui_caps() -> Vec<Right> {
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
pub fn call_agent_export(agent_id: u64, export: &str, args_json: &str) -> Result<String, &'static str> {
    let path = format!("{}/assets/tools.wasm", crate::agent::home::path(agent_id));
    let wasm = crate::synapse::fs::read(&path).ok_or("tools.wasm missing")?;
    if wasm.is_empty() {
        return Err("tools.wasm empty");
    }
    let bind = bindings_for(agent_id);
    // If package UI is running for this agent, prefer its in-memory bytes.
    let bytes = RUN.with(|r| {
        r.as_ref()
            .filter(|x| x.agent_id == agent_id)
            .map(|x| x.wasm.clone())
            .unwrap_or(wasm)
    });
    wasm_rt::call_string_bound(
        &bytes,
        export,
        args_json,
        Limits::default().with_fuel(2_000_000),
        bind,
    )
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

/// Start package UI for a system agent name (`paint`, `slides`, `minesweeper`, `snake`).
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

    ACTIVE_SURFACE.store(surface, Ordering::Relaxed);
    RUN.with(|r| {
        *r = Some(Running {
            name: name.into(),
            agent_id,
            task,
            surface,
            wasm: wasm.clone(),
        });
    });

    // Guest init export (manifest tool names; minesweeper → mines_start).
    let init_args = format!(r#"{{"app":"{name}","surface":{surface}}}"#);
    let start_export = match name {
        "minesweeper" => "mines_start",
        "paint" => "paint_start",
        "slides" => "slides_start",
        "snake" => "snake_start",
        other => other, // fall through to app_start below
    };
    let init = call_agent_export(agent_id, start_export, &init_args)
        .or_else(|_| call_agent_export(agent_id, "app_start", &init_args));
    match init {
        Ok(s) => crate::serial_println!("package_ui> {name}: {s}"),
        Err(e) => crate::serial_println!("package_ui> {name} init: {e}"),
    }
    Ok(surface)
}

/// Upkeep: optional guest `tick` export (snake, etc.).
pub fn tick() {
    let now = crate::arch::now_ms();
    if now.saturating_sub(LAST_TICK_MS.load(Ordering::Relaxed)) < 180 {
        return;
    }
    LAST_TICK_MS.store(now, Ordering::Relaxed);
    let (aid, name) = match RUN.with(|r| r.as_ref().map(|x| (x.agent_id, x.name.clone()))) {
        Some(x) => x,
        None => return,
    };
    let _ = call_agent_export(aid, "tick", &format!(r#"{{"app":"{name}"}}"#));
}
