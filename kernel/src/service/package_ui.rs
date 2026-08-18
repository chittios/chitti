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
    /// Target frame interval. `HOUSEKEEPING_MS` for an ordinary event-driven app;
    /// a smaller value for a **realtime** one that declared `wasm.frame_ms`.
    frame_ms: u64,
    /// When this app last ticked. Per-app, not global: one shared timestamp meant
    /// the first app to tick set it for everyone, so N apps each ticked at 1/N of
    /// the intended rate — invisible with one app open, which is the usual case.
    last_tick_ms: u64,
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
/// Housekeeping cadence for an ordinary, event-driven package UI.
///
/// Unchanged at 180 ms: these apps redraw on input, and a faster poll would cost
/// a ring-3 crossing, an audit entry and a full pane-scale present per app per
/// tick for no benefit. A **realtime** app opts out via `wasm.frame_ms`.
const HOUSEKEEPING_MS: u64 = 180;
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
    //
    // **Performed from ring 3**, like every other effect an installed agent has. A
    // running app is an agent: it just reaches its surface through a wasm module rather
    // than through the tool router, and which code path an effect takes is not a reason
    // for it to keep kernel privilege.
    match crate::synapse::tenant::invoke_in_userspace(task, raw, crate::security::Justification::trusted()) {
        Some(crate::synapse::Invocation::Executed { result, .. }) => result,
        Some(other) => format!("{other:?}"),
        None => alloc::string::String::from("error: the userspace call never reached the gates"),
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
    // **Which ABI the module speaks is a property of the module**, not of the
    // manifest: a JavaScript-derived `tools.wasm` imports the QuickJS engine, and
    // nothing else in the system does. So a package written in JS and one written
    // in Rust are the same artifact at the same path with the same manifest, and
    // they cannot be confused or mislabelled — there is no flag to get wrong.
    if let Some(ns) = crate::agent::js_rt::JsSession::namespace() {
        if crate::agent::jsmod::links_plugin(&wasm, ns) {
            return call_js_export(agent_id, &wasm, export, args_json);
        }
    }
    let mut limits = Limits::default()
        .with_fuel(crate::agent::system::manifest_fuel(agent_id).unwrap_or(CALL_FUEL));
    if let Some(pages) = crate::agent::system::manifest_pages(agent_id) {
        limits = limits.with_pages(pages);
    }
    wasm_rt::call_string_bound(&wasm, export, args_json, limits, bindings_for(agent_id))
}

/// Run one tool of a JavaScript-derived module: arguments as JSON on fd 0, result
/// as JSON on fd 1.
///
/// A fresh instance per call, deliberately. The engine's module top level re-runs
/// on every `invoke`, so JS globals cannot carry state between calls anyway —
/// caching an instance would buy the ~50 ms of engine start-up at the cost of a
/// staleness bug the moment `/agents build` rewrites the module underneath it.
fn call_js_export(
    agent_id: u64,
    wasm: &[u8],
    export: &str,
    args_json: &str,
) -> Result<String, &'static str> {
    // A module built against a different engine fails inside QuickJS with
    // `invalid version`, which reads as a broken tool rather than a stale build.
    // Name it before running, while we still can.
    let want = crate::agent::js_rt::JsSession::plugin_stamp();
    if let Some(got) = crate::agent::jsmod::plugin_stamp(wasm) {
        if got != want {
            crate::ktrace::log_fmt(format_args!(
                "js: agent {agent_id} tools.wasm built against '{got}', engine is '{want}'"
            ));
            return Err("tools.wasm was built against another JS engine -- rebuild it");
        }
    }
    let limits = wasm_rt::Limits::default()
        // A manifest may raise the budget; it may not lower it below what the
        // engine needs to start (see JS_MIN_CALL_FUEL).
        .with_fuel(
            crate::agent::system::manifest_fuel(agent_id)
                .unwrap_or(crate::agent::js_rt::JS_CALL_FUEL)
                .max(crate::agent::js_rt::JS_MIN_CALL_FUEL),
        )
        .with_pages(crate::agent::js_rt::JS_MEM_PAGES)
        .with_table_elems(crate::agent::js_rt::JS_TABLE_ELEMS);
    let mut js = crate::agent::js_rt::JsSession::with_module(wasm, limits, bindings_for(agent_id))?;
    js.call(export, args_json)
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

/// A [`Session`](crate::agent::types::Session) for one model-ask turn, carrying
/// **the app agent's own identity** — built once per ask, not per tool call.
///
/// The identity matters: `storage_*` and `memory_*` are keyed on
/// `session.agent.manifest_id`, so running them under the chat agent's session
/// would put a game's saved position in the shell agent's store. Caps are the
/// manifest's, narrowed by the install grant, exactly as `resolve_chat_agent`
/// does for a chat agent — an app's model turn must not be able to do more than
/// the human approved at install.
///
/// Per-ask rather than per-call because `grant_to_task` mints cap slots in the
/// task's table; doing that on every tool call would grow it for the life of a
/// long game.
fn ask_session(agent_id: u64, task: TaskId) -> Option<crate::agent::types::Session> {
    use crate::agent::types::AgentId;
    let m = crate::skills::agent_skill::by_id(AgentId(agent_id))?;
    let grant = crate::skills::agent_skill::install_grant(AgentId(agent_id))
        .unwrap_or_else(|| m.capabilities.clone());
    let bounded = crate::agent::types::intersect_caps(&m.capabilities, &grant);
    let caps = crate::skills::install::with_home_sandbox(&bounded, AgentId(agent_id), m.kind);
    let live = crate::agent::manifest::grant_to_task(task, &caps);
    Some(crate::agent::types::Session::new(
        &m,
        agent_id,
        live,
        crate::agent::orchestrator::now(),
    ))
}

/// Execute one tool call from an app agent's model turn, **as that agent**,
/// through the ordinary [`Router`](crate::tools::Router).
///
/// Going through the Router rather than reimplementing the dispatch is the whole
/// point: it already knows that `ui_draw`/`board_set` lower to Synapse
/// primitives (gated + audited under `task`'s own caps, in ring 3),
/// `storage_*`/`memory_*` are durable per-agent state, and a wasm-export tool
/// like `chess_legal` runs in the owning package's module. This function
/// therefore contains no tool names — the previous version contained a closure
/// that refused everything, and before that a list that had drifted from both
/// the prompt and the manifest.
fn run_app_tool(
    session: &mut crate::agent::types::Session,
    task: TaskId,
    surface: u32,
    call_id: u64,
    tool: &str,
    args: &str,
) -> String {
    use crate::agent::agent_loop::{format_tool_result, ToolDispatch};
    use crate::agent::types::ToolCall;
    let args = with_surface(tool, args, surface);
    let call = ToolCall {
        call_id,
        tool: tool.to_string(),
        args,
    };
    let mut router = crate::tools::Router::taint_aware();
    let out = router.call(session, task, &call);
    let text = format_tool_result(out.is_error, out.result);
    crate::ktrace::log_fmt(format_args!(
        "package_ui: ask tool {tool} -> {}",
        crate::tools::pathutil::truncate_on_char_boundary(&text, 80)
    ));
    text
}

/// Fill in `surface` when the tool requires one and the model left it out.
///
/// Not a guess: a running app owns exactly one surface, and the Synapse UI
/// primitives are ownership-gated anyway, so the only surface this call could
/// legally name is the one it was told about in the prompt. Without this, a
/// model that forgets the argument gets "missing required arg 'surface'" and
/// usually forgets it again — a whole class of dead turns for a field the kernel
/// already knows.
fn with_surface(tool: &str, args: &str, surface: u32) -> String {
    let requires_surface = crate::tools::registry::get(tool)
        .map(|d| d.required.iter().any(|r| r == "surface"))
        .unwrap_or(false);
    if !requires_surface || args.contains("\"surface\"") {
        return args.to_string();
    }
    let t = args.trim();
    match t.strip_prefix('{').map(str::trim_start) {
        // `{}` / empty → a fresh object; otherwise splice the field in front.
        Some(rest) if rest.starts_with('}') || rest.is_empty() => {
            format!("{{\"surface\":{surface}}}")
        }
        Some(rest) => format!("{{\"surface\":{surface},{rest}"),
        None => format!("{{\"surface\":{surface}}}"),
    }
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
            .map(|x| (x.name.clone(), x.agent_id, x.surface, x.task))
    });
    let Some((name, agent_id, surface, task)) = meta else {
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
    // What this agent may call comes from its own manifest ∩ the registry, and
    // the same list drives the prompt and the gate (`shell::ui_agent_reply`).
    let tools = crate::shell::ui_agent_toolset(agent_id);
    let reply = match ask_session(agent_id, task) {
        Some(mut session) => {
            crate::ktrace::log_fmt(format_args!(
                "package_ui: ask from '{name}' with {} tool(s) as agent {agent_id} on task {task}",
                tools.len()
            ));
            let mut call_id = 0u64;
            crate::shell::ui_agent_reply(&soul, prompt, surface, &tools, |cmd, args| {
                call_id += 1;
                run_app_tool(&mut session, task, surface, call_id, cmd, args)
            })
        }
        // No manifest (or none installed): the agent has no identity to run a
        // tool under, so say that rather than pretending tools exist.
        None => crate::shell::ui_agent_reply(&soul, prompt, surface, &[], |_cmd, _args| {
            String::from("error: this app has no installed manifest, so it has no tools here")
        }),
    };
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
        // 64 pages (4 MiB) unless the manifest asks for more — every existing app
        // declares none and so keeps exactly its old ceiling.
        .with_pages(crate::agent::system::manifest_pages(agent_id).unwrap_or(64));
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
    // A package UI's native resolution and frame budget come from its **signed
    // manifest**, not from the running guest — so the kernel applies both here
    // and `ui_surface_request` keeps taking only a `kind`. That is why neither
    // needed a grammar change or a new primitive: the guest's authority is
    // exactly what it was.
    if let Some((mw, mh)) = crate::agent::system::manifest_surface(agent_id) {
        match crate::synapse::ui::resize(task, surface, mw as usize, mh as usize) {
            Ok((w, h)) => {
                crate::serial_println!("package_ui> {name}: surface {surface} sized {w}x{h}")
            }
            Err(e) => crate::serial_println!(
                "package_ui> {name}: surface resize to {mw}x{mh} refused ({e:?}) — keeping default"
            ),
        }
    }
    let frame_ms = crate::agent::system::manifest_frame_ms(agent_id)
        .map(|v| v as u64)
        .unwrap_or(HOUSEKEEPING_MS);
    if frame_ms != HOUSEKEEPING_MS {
        crate::serial_println!(
            "package_ui> {name}: realtime, {frame_ms} ms/frame ({} fps target)",
            1000 / frame_ms.max(1)
        );
    }

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
                frame_ms,
                // Zero, not `now`, so the first frame runs on the very next pump
                // rather than after a full frame of nothing.
                last_tick_ms: 0,
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

    // Re-read after click handling (map may have changed). Each app carries its
    // own cadence and its own last-tick stamp, so a realtime game runs at its
    // frame rate without dragging every other app up to it — and, more to the
    // point, without every other app dragging *it* down.
    let apps: Vec<(String, u32, bool, u64)> = APPS.with(|m| {
        m.values()
            .filter(|x| now.saturating_sub(x.last_tick_ms) >= x.frame_ms)
            .map(|x| (x.name.clone(), x.surface, x.asking, x.frame_ms))
            .collect()
    });
    for (name, surface, asking, frame_ms) in apps {
        if is_stop_pending(surface) {
            continue;
        }
        // Stamp *before* the call, not after. A frame that takes longer than its
        // budget would otherwise schedule the next one immediately on return, so
        // a heavy app would monopolise `upkeep` and starve the clock, the mouse
        // and the net stack — the cooperative-scheduler failure this whole file
        // is careful about.
        APPS.with(|m| {
            if let Some(r) = m.get_mut(&surface) {
                r.last_tick_ms = now;
            }
        });
        // The guest is handed the elapsed time so it can advance its own
        // simulation honestly. `dt` rather than a frame counter because the pump
        // is driven by `upkeep()`, whose cadence is opportunistic — a fixed step
        // would drift against the wall clock whenever the machine is busy.
        let dt = frame_ms;
        let args = format!(r#"{{"app":"{name}","dt":{dt}}}"#);
        if let Ok(out) = call_surface(surface, "tick", &args) {
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

    /// **A JavaScript-derived `tools.wasm` is reachable as an ordinary agent tool.**
    ///
    /// This is the whole claim of the JS path: the artifact, the path and the
    /// manifest are the ones a Rust package uses, so the tool router needs no new
    /// binding and `call_agent_export` is the single entry point for both. What
    /// decides the ABI is the module's own imports, checked here by going through
    /// the *real* entry point rather than calling the JS runtime directly.
    #[test_case]
    fn a_javascript_module_answers_through_the_ordinary_tool_path() {
        const SRC: &str = r#"
            function readArgs() {
              const chunks = []; const buf = new Uint8Array(1024); let n;
              while ((n = Javy.IO.readSync(0, buf)) > 0) chunks.push(buf.slice(0, n));
              let total = 0; for (const c of chunks) total += c.length;
              const all = new Uint8Array(total); let at = 0;
              for (const c of chunks) { all.set(c, at); at += c.length; }
              return JSON.parse(new TextDecoder().decode(all) || "{}");
            }
            function reply(v) { Javy.IO.writeSync(1, new TextEncoder().encode(JSON.stringify(v))); }
            export function probe_double() { const a = readArgs(); reply({ doubled: (a.n || 0) * 2 }); }
        "#;
        // A throwaway agent id, outside the system range, with a module in the
        // place `place_agent_home` would have written one.
        let agent_id = 31_337u64;
        let wasm = crate::agent::js_rt::build_module(SRC, &["probe_double"]).expect("build");
        let path = format!("{}/assets/tools.wasm", crate::agent::home::path(agent_id));
        crate::synapse::fs::write(&path, &wasm);

        let out = call_agent_export(agent_id, "probe_double", r#"{"n":21}"#)
            .expect("a JS module must answer through call_agent_export");
        assert!(out.contains(r#""doubled":42"#), "got {out}");

        // A tool the module does not export is an error, not a panic or an empty
        // success — the caller has to be able to tell.
        assert!(call_agent_export(agent_id, "no_such_tool", "{}").is_err());

        // And the ordinary Rust path is untouched: a module that does not import the
        // engine still goes through the string ABI.
        let rust_path = format!("{}/assets/tools.wasm", crate::agent::home::path(agent_id + 1));
        crate::synapse::fs::write(rust_path.as_str(), crate::agent::wasm_rt::FIXTURE_ECHO);
        let echoed = call_agent_export(agent_id + 1, "echo", r#"{"a":1}"#).expect("string ABI");
        assert_eq!(echoed, r#"{"a":1}"#);
    }

    /// A module built by a different engine is refused with a rebuild hint rather
    /// than failing deep inside QuickJS as `invalid version`.
    #[test_case]
    fn a_stale_javascript_module_is_named_not_run() {
        let agent_id = 31_339u64;
        let bc = alloc::vec![1u8, 2, 3, 4];
        let ns = crate::agent::js_rt::JsSession::namespace().expect("namespace");
        let stale = crate::agent::jsmod::emit(ns, "some-older-engine@1", &bc, &["t"]).expect("emit");
        let path = format!("{}/assets/tools.wasm", crate::agent::home::path(agent_id));
        crate::synapse::fs::write(&path, &stale);
        let err = call_agent_export(agent_id, "t", "{}").unwrap_err();
        assert!(err.contains("rebuild"), "got {err}");
    }

    /// An app's model turn is told its surface id; when the model omits the
    /// argument anyway the runtime fills it in. A running app owns exactly one
    /// surface and the UI primitives are ownership-gated, so there is only one
    /// value this call could legally name — and without it the turn dies on
    /// "missing required arg 'surface'", usually repeatedly.
    #[test_case]
    fn with_surface_fills_only_when_required_and_absent() {
        // `ui_draw` requires a surface: an object without one gets it, in front,
        // with the rest of the JSON intact.
        let out = with_surface("ui_draw", r#"{"ops":"clear 000000"}"#, 7);
        assert_eq!(out, r#"{"surface":7,"ops":"clear 000000"}"#, "got {out}");
        // Already present → untouched (never overwrite the model's own value:
        // if it named another surface, the ownership gate must refuse it, not
        // have it silently rewritten into a legal one).
        let given = r#"{"surface":9,"ops":"clear 000000"}"#;
        assert_eq!(with_surface("ui_draw", given, 7), given);
        // Empty / `{}` args become a fresh object.
        assert_eq!(with_surface("ui_draw", "{}", 3), r#"{"surface":3}"#);
        assert_eq!(with_surface("ui_draw", "", 3), r#"{"surface":3}"#);
        assert_eq!(with_surface("ui_draw", "{ }", 3), r#"{"surface":3}"#);
        // A tool that takes no surface is never given one…
        let a = r#"{"key":"fen","value":"8/8"}"#;
        assert_eq!(with_surface("storage_set", a, 7), a);
        // …nor is an unknown tool (no schema to consult → do not invent args).
        assert_eq!(with_surface("no_such_tool", a, 7), a);
    }

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
