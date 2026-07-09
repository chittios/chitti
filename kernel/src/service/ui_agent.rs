//! **UI agent runtime** — the action-pane counterpart of [`super::server`].
//!
//! A UI agent is installable markdown (`agents/<name>/{SOUL.md,manifest.json}`)
//! that owns a Synapse surface and paints it through tool calls. The runtime:
//!
//! 1. Spawns a cap-owning task, grants `CapDomain::Ui`, requests a `board`
//!    surface, and paints an initial FEN via `board_set`.
//! 2. Persists FEN in agent memory (`memory/fen`).
//! 3. On each compositor click **or keyboard activation**: maps to a square,
//!    then either a **SOUL ReAct** turn (model present) with `board_set` /
//!    `board_mark` / `chess_legal`, or a **native two-click** path using the
//!    same legal-move engine.
//!
//! Keyboard (action pane focused on the board surface): **arrows** move a
//! cursor, **Enter** selects / moves (same as a click). Esc clears selection.
//!
//! Determinism boundary: the model only *decides*; every paint is a
//! capability-checked Synapse call. Rules validation is pure native code
//! ([`super::chess_rules`]).

use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::agent::wasm_abi::{self, ToolBackend};
use crate::agent::{storage, home};
use crate::cap::Right;
use crate::mm::Locked;
use crate::sched::TaskId;
use crate::service::chess_rules;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

struct UiAgentState {
    name: String,
    home: String,
    soul: String,
    task: TaskId,
    surface: u32,
    agent_id: u64,
    /// Last board FEN; mirrored via [`storage`] (session + durable).
    fen: String,
    /// First-click / Enter selection for the native (no-model) path.
    selected: Option<String>,
    /// Keyboard cursor on the board: (file 0..7, rank 0..7), rank 0 = white's first.
    cursor: (u8, u8),
    use_model: bool,
    /// Guard: a model turn is in flight (ignore nested ticks).
    busy: bool,
    /// Host / future WASM tool table for this agent instance.
    tools: Vec<ToolBackend>,
}

/// Cursor outline colour (cool blue — distinct from selection terracotta).
const CURSOR_COLOR: &str = "6688cc";
/// Selection / legal-move highlight (brand terracotta).
const SELECT_COLOR: &str = "cc785c";

static AGENT: Locked<Option<UiAgentState>> = Locked::new(None);
static RUNNING: AtomicBool = AtomicBool::new(false);
static ACTIVE_SURFACE: AtomicU32 = AtomicU32::new(0);
static ACTIVE_TASK: AtomicU64 = AtomicU64::new(0);

/// Surface id the compositor should push events into, if any.
pub fn active_surface() -> Option<u32> {
    let id = ACTIVE_SURFACE.load(Ordering::Relaxed);
    if id == 0 {
        None
    } else {
        Some(id)
    }
}

/// True when `id` is the running UI agent's surface (so shell media keys must
/// not treat it as an image viewer).
pub fn owns_surface(id: u32) -> bool {
    active_surface() == Some(id)
}

/// True while a model turn is in flight (shell routes keys here to ignore them).
pub fn is_busy() -> bool {
    RUNNING.load(Ordering::Relaxed)
        && AGENT.with(|a| a.as_ref().map(|s| s.busy).unwrap_or(false))
}

/// Nudge the board cursor by `(df, dr)` and repaint. Pure clamp: stays on board.
pub fn nudge_cursor(df: i8, dr: i8) {
    if !RUNNING.load(Ordering::Relaxed) || is_busy() {
        return;
    }
    AGENT.with(|a| {
        if let Some(s) = a.as_mut() {
            s.cursor = step_cursor(s.cursor, df, dr);
        }
    });
    repaint_board_chrome();
}

/// Activate the square under the keyboard cursor (same as a click / Enter).
pub fn activate_cursor() {
    if !RUNNING.load(Ordering::Relaxed) || is_busy() {
        return;
    }
    let sq = AGENT.with(|a| {
        a.as_ref()
            .map(|s| chess_rules::square_name(s.cursor.0, s.cursor.1))
    });
    if let Some(sq) = sq {
        activate_square(&sq);
    }
}

/// Clear selection and repaint (Esc while the board is focused).
pub fn clear_selection() {
    if !RUNNING.load(Ordering::Relaxed) || is_busy() {
        return;
    }
    AGENT.with(|a| {
        if let Some(s) = a.as_mut() {
            s.selected = None;
        }
    });
    repaint_board_chrome();
}

/// Step `(file, rank)` by deltas, clamped to 0..7. Pure — unit-tested.
pub fn step_cursor(cur: (u8, u8), df: i8, dr: i8) -> (u8, u8) {
    let f = (cur.0 as i16 + df as i16).clamp(0, 7) as u8;
    let r = (cur.1 as i16 + dr as i16).clamp(0, 7) as u8;
    (f, r)
}

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

fn call_board_set(task: TaskId, surface: u32, fen: &str) -> String {
    let fen_esc = fen.replace('\\', "\\\\").replace('"', "\\\"");
    let raw = format!(r#"{{"name":"board_set","arguments":{{"surface":{surface},"fen":"{fen_esc}"}}}}"#);
    syn_exec(task, &raw)
}

fn call_board_mark(task: TaskId, surface: u32, squares: &str, color: &str) -> String {
    let raw = format!(
        r#"{{"name":"board_mark","arguments":{{"surface":{surface},"squares":"{squares}","color":"{color}"}}}}"#
    );
    syn_exec(task, &raw)
}

fn call_surface_request(task: TaskId, kind: &str) -> Option<u32> {
    let raw = format!(r#"{{"name":"ui_surface_request","arguments":{{"kind":"{kind}"}}}}"#);
    let out = syn_exec(task, &raw);
    out.strip_prefix("ok:surface=").and_then(|s| s.trim().parse().ok())
}

fn agent_id_from_home(home_path: &str) -> u64 {
    home_path.rsplit('/').next().and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn persist_fen(agent_id: u64, fen: &str) {
    let _ = storage::set_str(agent_id, storage::Scope::Session, "fen", fen);
    let _ = storage::set_str(agent_id, storage::Scope::Durable, "fen", fen);
}

fn load_persisted_fen(agent_id: u64) -> Option<String> {
    storage::get_str(agent_id, storage::Scope::Session, "fen")
        .or_else(|| storage::get_str(agent_id, storage::Scope::Durable, "fen"))
        .filter(|s| !s.is_empty())
}

/// Start (or replace) a UI agent by package name (`chess`).
pub fn start(name: &str) -> Result<u32, &'static str> {
    stop();

    let home = crate::agent::system::home_for(name).ok_or("unknown UI agent (try: chess)")?;
    let soul_path = format!("{home}/SOUL.md");
    let soul = crate::synapse::fs::read(&soul_path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_else(|| {
            String::from("You are a UI agent. Paint your board with board_set/board_mark.")
        });

    let task_name: &'static str = match name {
        "chess" => "ui-chess",
        other => alloc::boxed::Box::leak(format!("ui-{other}").into_boxed_str()),
    };
    let task = crate::arch::interrupts::without_interrupts(|| {
        let task = crate::sched::spawn_parked(task_name);
        for r in ui_caps() {
            crate::cap::grant(task, r);
        }
        task
    });

    let agent_id = agent_id_from_home(&home);
    let surface = call_surface_request(task, "board").ok_or("ui_surface_request failed")?;
    let fen = load_persisted_fen(agent_id).unwrap_or_else(|| chess_rules::START_FEN.to_string());
    let paint = call_board_set(task, surface, &fen);
    if !paint.starts_with("ok:") {
        return Err("board_set initial paint failed");
    }
    persist_fen(agent_id, &fen);

    // Same backend policy as shell chat: remote and/or local GGUF.
    let use_model = crate::shell::planner_available();
    // Chess ships tools.wasm; other UI agents use host-only table until packaged.
    let tools = if name == "chess" {
        wasm_abi::chess_package_tools()
    } else {
        wasm_abi::default_ui_host_tools()
    };
    AGENT.with(|a| {
        *a = Some(UiAgentState {
            name: name.to_string(),
            home,
            soul,
            task,
            surface,
            agent_id,
            fen,
            selected: None,
            // Start cursor on white's e2 (common first-move square).
            cursor: (4, 1),
            use_model,
            busy: false,
            tools,
        });
    });
    ACTIVE_SURFACE.store(surface, Ordering::Relaxed);
    ACTIVE_TASK.store(task, Ordering::Relaxed);
    RUNNING.store(true, Ordering::Relaxed);
    repaint_board_chrome();

    let backend = if crate::shell::remote::is_remote_active() {
        "remote"
    } else if use_model {
        "local"
    } else {
        "native"
    };
    crate::ktrace::log_fmt(format_args!(
        "ui_agent: started '{name}' surface={surface} task={task} backend={backend}"
    ));
    crate::serial_println!(
        "ui_agent> ready ({}). Click or arrows+Enter.{}",
        if use_model {
            if backend == "remote" {
                "SOUL ReAct via remote model"
            } else {
                "SOUL ReAct via local model"
            }
        } else {
            "native legal moves — /model remote or load a GGUF for agent turns"
        },
        if use_model {
            " Events blocked while thinking (loader on board)."
        } else {
            ""
        }
    );
    Ok(surface)
}

/// Stop the running UI agent and close its surface.
pub fn stop() {
    if !RUNNING.swap(false, Ordering::Relaxed) {
        AGENT.with(|a| *a = None);
        ACTIVE_SURFACE.store(0, Ordering::Relaxed);
        ACTIVE_TASK.store(0, Ordering::Relaxed);
        return;
    }
    if let Some(st) = AGENT.with(|a| a.take()) {
        let raw = format!(
            r#"{{"name":"ui_surface_close","arguments":{{"surface":{}}}}}"#,
            st.surface
        );
        let _ = syn_exec(st.task, &raw);
        storage::clear_session(st.agent_id);
        let _ = crate::sched::kill(st.task);
        crate::ktrace::log_fmt(format_args!("ui_agent: stopped '{}'", st.name));
    }
    ACTIVE_SURFACE.store(0, Ordering::Relaxed);
    ACTIVE_TASK.store(0, Ordering::Relaxed);
}

pub fn is_running() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

pub fn status_line() -> Option<String> {
    AGENT.with(|a| {
        a.as_ref().map(|s| {
            let backend = if s.busy {
                " [thinking…]"
            } else if s.use_model && crate::shell::remote::is_remote_active() {
                " [remote]"
            } else if s.use_model {
                " [local]"
            } else {
                " [native]"
            };
            format!(
                "ui-agent {} surface={} fen={}{}",
                s.name,
                s.surface,
                s.fen.split_whitespace().next().unwrap_or("?"),
                backend
            )
        })
    })
}

/// Pump events from the surface queue (called from shell upkeep).
pub fn tick() {
    if !RUNNING.load(Ordering::Relaxed) {
        return;
    }
    // While a model turn is in flight: animate the loader and **drop** all
    // input so clicks/keys cannot nest turns.
    if AGENT.with(|a| a.as_ref().map(|s| s.busy).unwrap_or(false)) {
        paint_loader_overlay();
        drain_events_discard();
        return;
    }
    let (task, surface) = match AGENT.with(|a| a.as_ref().map(|s| (s.task, s.surface))) {
        Some(x) => x,
        None => return,
    };
    for _ in 0..4 {
        let raw = format!(r#"{{"name":"ui_event_poll","arguments":{{"surface":{surface}}}}}"#);
        let out = syn_exec(task, &raw);
        if out == "ok:none" || out.starts_with("error:") {
            break;
        }
        if let Some(rest) = out.strip_prefix("ok:click=") {
            let mut xy = rest.split(',');
            let x: u16 = xy.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let y: u16 = xy.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            handle_click(x, y);
        } else if let Some(k) = out.strip_prefix("ok:key=") {
            // Surface-routed keys (if compositor pushes them). Shell path uses
            // nudge_cursor / activate_cursor directly for lower latency.
            if let Ok(code) = k.trim().parse::<u8>() {
                handle_key_byte(code);
            }
        }
    }
}

/// Discard queued surface events without acting (busy / loader state).
fn drain_events_discard() {
    let (task, surface) = match AGENT.with(|a| a.as_ref().map(|s| (s.task, s.surface))) {
        Some(x) => x,
        None => return,
    };
    for _ in 0..8 {
        let raw = format!(r#"{{"name":"ui_event_poll","arguments":{{"surface":{surface}}}}}"#);
        let out = syn_exec(task, &raw);
        if out == "ok:none" || out.starts_with("error:") {
            break;
        }
    }
}

/// Animated bottom strip while the agent model turn runs (loader tool chrome).
fn paint_loader_overlay() {
    let (task, surface) = match AGENT.with(|a| a.as_ref().map(|s| (s.task, s.surface))) {
        Some(x) => x,
        None => return,
    };
    let phase = ((crate::arch::now_ms() / 120) % 5) as i32;
    // Board is 192px tall (8×24); use the right margin strip under the board
    // for a simple indeterminate progress bar (no text draw op).
    let y = 192 - 20;
    let mut ops = format!("rect 0 {y} 256 20 1a1816; ");
    let bx = 24 + phase * 44;
    ops.push_str(&format!("rect {bx} {} 36 10 cc785c; ", y + 5));
    // Subtle track
    ops.push_str(&format!("rect 20 {} 216 2 3a3632; ", y + 9));
    call_ui_draw(task, surface, &ops);
}

fn call_ui_draw(task: TaskId, surface: u32, ops: &str) {
    let ops_esc = ops.replace('\\', "\\\\").replace('"', "\\\"");
    let raw = format!(
        r#"{{"name":"ui_draw","arguments":{{"surface":{surface},"ops":"{ops_esc}"}}}}"#
    );
    let _ = syn_exec(task, &raw);
}

fn handle_key_byte(code: u8) {
    match code {
        b'\r' | b'\n' => activate_cursor(),
        0x1b => clear_selection(),
        // Direct WASD / hjkl fallbacks if a driver ever pushes raw letters.
        b'w' | b'W' | b'k' | b'K' => nudge_cursor(0, 1),
        b's' | b'S' | b'j' | b'J' => nudge_cursor(0, -1),
        b'a' | b'A' | b'h' | b'H' => nudge_cursor(-1, 0),
        b'd' | b'D' | b'l' | b'L' => nudge_cursor(1, 0),
        _ => {}
    }
}

fn handle_click(x: u16, y: u16) {
    if is_busy() {
        return;
    }
    let sq = match crate::synapse::ui::click_to_square(x, y) {
        Some(s) => s,
        None => {
            crate::serial_println!("ui_agent> click ({x},{y}) off-board");
            return;
        }
    };
    crate::serial_println!("ui_agent> click ({x},{y}) → {sq}");
    // Keep keyboard cursor under the pointer so Enter continues from here.
    if let Some((f, r)) = chess_rules::parse_square(&sq) {
        AGENT.with(|a| {
            if let Some(s) = a.as_mut() {
                s.cursor = (f, r);
            }
        });
    }
    activate_square(&sq);
}

fn activate_square(sq: &str) {
    if is_busy() {
        return;
    }
    let use_model = AGENT.with(|a| a.as_ref().map(|s| s.use_model).unwrap_or(false));
    if use_model {
        model_turn_on_click(sq);
    } else {
        native_two_click(sq);
    }
}

/// Redraw FEN + selection/legal marks + keyboard cursor outline.
fn repaint_board_chrome() {
    let (task, surface, fen, selected, cursor) = match AGENT.with(|a| {
        a.as_ref().map(|s| {
            (
                s.task,
                s.surface,
                s.fen.clone(),
                s.selected.clone(),
                s.cursor,
            )
        })
    }) {
        Some(x) => x,
        None => return,
    };
    call_board_set(task, surface, &fen);
    let cur = chess_rules::square_name(cursor.0, cursor.1);
    if let Some(ref sel) = selected {
        let legal = chess_rules::legal_moves(&fen, sel);
        if legal != "none" {
            let marks = format!("{sel},{legal}");
            call_board_mark(task, surface, &marks, SELECT_COLOR);
        } else {
            call_board_mark(task, surface, sel, SELECT_COLOR);
        }
    }
    // Cursor last so it stays visible over legal highlights.
    call_board_mark(task, surface, &cur, CURSOR_COLOR);
}

/// SOUL ReAct on a click: model may call board_set / board_mark / chess_legal.
/// Falls back to native two-click if the planner is unavailable.
fn model_turn_on_click(sq: &str) {
    let (soul, fen, surface, task, selected) = match AGENT.with(|a| {
        a.as_ref().map(|s| {
            (
                s.soul.clone(),
                s.fen.clone(),
                s.surface,
                s.task,
                s.selected.clone(),
            )
        })
    }) {
        Some(x) => x,
        None => return,
    };

    AGENT.with(|a| {
        if let Some(s) = a.as_mut() {
            s.busy = true;
        }
    });

    // Loader chrome: repaint board + selection, then animated strip (tick keeps
    // animating while remote/local inference runs via upkeep).
    call_board_set(task, surface, &fen);
    call_board_mark(task, surface, sq, "6688cc");
    paint_loader_overlay();
    crate::serial_println!("ui_agent> thinking… (Ctrl+C to cancel; input blocked)");
    crate::shell::upkeep();

    let legal = chess_rules::legal_moves(&fen, sq);
    let user = format!(
        "event: click\nsquare: {sq}\nsurface: {surface}\ncurrent_fen: {fen}\n\
         selected: {}\nlegal_from_square: {legal}\n\
         Use chess_legal if unsure. Then board_mark and/or board_set. Finish with a short status line.",
        selected.as_deref().unwrap_or("-")
    );

    let mut last_fen = fen.clone();
    let mut last_selected = selected.clone();

    let tools = AGENT.with(|a| a.as_ref().map(|s| s.tools.clone()).unwrap_or_default());
    let agent_id = AGENT.with(|a| a.as_ref().map(|s| s.agent_id).unwrap_or(0));
    let answer = crate::shell::ui_agent_reply(&soul, &user, surface, |cmd, args| {
        // Keep the loader alive between tool hops.
        paint_loader_overlay();
        crate::shell::upkeep();
        dispatch_ui_tool(
            task,
            surface,
            agent_id,
            &tools,
            cmd,
            args,
            &mut last_fen,
            &mut last_selected,
        )
    });

    AGENT.with(|a| {
        if let Some(s) = a.as_mut() {
            s.busy = false;
            s.fen = last_fen.clone();
            s.selected = last_selected.clone();
            persist_fen(s.agent_id, &s.fen);
        }
    });

    match answer {
        Some(text) => {
            let t = text.trim();
            if !t.is_empty() {
                crate::serial_println!("ui_agent> {t}");
            }
            // Ensure board + cursor chrome match runtime state after the turn.
            repaint_board_chrome();
        }
        None => {
            crate::serial_println!("ui_agent> no planner (remote or local) — falling back to native moves");
            native_two_click(sq);
        }
    }
}

/// Execute one UI-agent tool via the agent's [`ToolBackend`] table.
///
/// Host tools: Synapse draw / storage / chess_rules helpers (until W3 moves
/// chess logic into package WASM). Wasm tools: call export (W1+ stub errors).
fn dispatch_ui_tool(
    task: TaskId,
    surface: u32,
    agent_id: u64,
    tools: &[ToolBackend],
    cmd: &str,
    args: &str,
    fen: &mut String,
    selected: &mut Option<String>,
) -> String {
    use crate::session::todo::json_str;
    let Some(backend) = wasm_abi::lookup(tools, cmd) else {
        return format!("error:unknown tool {cmd}");
    };
    match backend {
        ToolBackend::Wasm {
            module_path,
            export,
            fuel,
            ..
        } => {
            // Inject current FEN when the model omits it (runtime state is source of truth).
            let mut call_args = args.to_string();
            if json_str(&call_args, "fen").unwrap_or_default().is_empty() && !fen.is_empty() {
                let fen_esc = fen.replace('\\', "\\\\").replace('"', "\\\"");
                if call_args.trim().starts_with('{') {
                    call_args = format!(
                        r#"{{"fen":"{fen_esc}",{}"#,
                        call_args.trim().trim_start_matches('{')
                    );
                } else {
                    // bare from-square → wrap
                    let from = call_args.trim().trim_matches('"');
                    call_args = format!(r#"{{"fen":"{fen_esc}","from":"{from}"}}"#);
                }
            }
            let path = format!(
                "{}/{}",
                home::path(agent_id),
                module_path.trim_start_matches('/')
            );
            let bytes = match crate::synapse::fs::read(&path) {
                Some(b) => b,
                None => return format!("error:wasm module missing at {path}"),
            };
            let bind = crate::agent::wasm_rt::HostBindings {
                agent_id,
                task,
                surface,
            };
            match wasm_abi::call_wasm_export(&bytes, export, &call_args, *fuel, bind) {
                Ok(s) => {
                    // Keep host FEN cache in sync when the guest reports a new position.
                    if let Some(rest) = s.strip_prefix("ok:fen=") {
                        *fen = rest.to_string();
                        *selected = None;
                        persist_fen(agent_id, fen);
                    } else if s.starts_with("ok:") {
                        // guest painted via host_board_set; refresh from storage if present
                        if let Some(f) = storage::get_str(agent_id, storage::Scope::Session, "fen") {
                            *fen = f;
                            *selected = None;
                        }
                    }
                    s
                }
                Err(e) => format!("error:{e}"),
            }
        }
        ToolBackend::Host { name } => {
            let arg_fen = json_str(args, "fen").unwrap_or_default();
            let arg_from = json_str(args, "from").unwrap_or_default();
            let arg_to = json_str(args, "to").unwrap_or_default();
            let arg_squares = json_str(args, "squares").unwrap_or_default();
            let arg_color = json_str(args, "color").unwrap_or_else(|| "cc785c".into());
            match name.as_str() {
                "board_mark" => {
                    let squares = if !arg_squares.is_empty() {
                        arg_squares
                    } else {
                        args.trim().to_string()
                    };
                    let r = call_board_mark(task, surface, &squares, &arg_color);
                    if squares.split(|c: char| c == ',' || c.is_whitespace()).count() == 1 {
                        let sq = squares.trim().to_string();
                        if chess_rules::parse_square(&sq).is_some() {
                            *selected = Some(sq);
                        }
                    }
                    r
                }
                "board_set" => {
                    let mut new_fen = if !arg_fen.is_empty() {
                        arg_fen
                    } else if args.contains('/') {
                        args.trim().to_string()
                    } else {
                        String::new()
                    };
                    if !arg_from.is_empty() && !arg_to.is_empty() {
                        match chess_rules::try_move(fen, &arg_from, &arg_to) {
                            Ok(f) => new_fen = f,
                            Err(e) => {
                                call_board_set(task, surface, fen);
                                call_board_mark(task, surface, &arg_to, "aa3333");
                                return e;
                            }
                        }
                    }
                    if new_fen.is_empty() {
                        return String::from("error: missing fen");
                    }
                    if chess_rules::Board::from_fen(&new_fen).is_none() {
                        return String::from("error:bad_fen");
                    }
                    let r = call_board_set(task, surface, &new_fen);
                    if r.starts_with("ok:") {
                        *fen = new_fen;
                        *selected = None;
                        persist_fen(agent_id, fen);
                    }
                    r
                }
                "storage_get" | "storage_set" | "storage_list" | "storage_remove" => {
                    storage::run_tool(agent_id, name, args)
                }
                "memory_add" | "memory_get" | "memory_list" | "memory_search" => {
                    home::run_memory_tool(name, agent_id, args)
                }
                "ui_draw" => {
                    // Raw draw passthrough for freeform UI agents.
                    let ops = json_str(args, "ops").unwrap_or_else(|| args.to_string());
                    let ops_esc = ops.replace('\\', "\\\\").replace('"', "\\\"");
                    let raw = format!(
                        r#"{{"name":"ui_draw","arguments":{{"surface":{surface},"ops":"{ops_esc}"}}}}"#
                    );
                    syn_exec(task, &raw)
                }
                other => format!("error:host tool '{other}' not wired in UI runtime"),
            }
        }
    }
}

/// Native two-click path (no GGUF): legal-move gated select/move.
fn native_two_click(sq: &str) {
    let (fen, selected) = match AGENT.with(|a| {
        a.as_ref()
            .map(|s| (s.fen.clone(), s.selected.clone()))
    }) {
        Some(x) => x,
        None => return,
    };

    match selected {
        None => {
            let legal = chess_rules::legal_moves(&fen, sq);
            if legal == "none" {
                AGENT.with(|a| {
                    if let Some(s) = a.as_mut() {
                        s.selected = None;
                    }
                });
                // Flash illegal square, keep cursor.
                let (task, surface) = match AGENT.with(|a| a.as_ref().map(|s| (s.task, s.surface))) {
                    Some(x) => x,
                    None => return,
                };
                call_board_set(task, surface, &fen);
                call_board_mark(task, surface, sq, "aa3333");
                let cur = AGENT.with(|a| {
                    a.as_ref()
                        .map(|s| chess_rules::square_name(s.cursor.0, s.cursor.1))
                });
                if let Some(c) = cur {
                    call_board_mark(task, surface, &c, CURSOR_COLOR);
                }
                crate::serial_println!("ui_agent> no legal moves from {sq}");
                return;
            }
            AGENT.with(|a| {
                if let Some(s) = a.as_mut() {
                    s.selected = Some(sq.to_string());
                }
            });
            repaint_board_chrome();
            crate::serial_println!("ui_agent> selected {sq} → legal {legal}");
        }
        Some(from) if from == sq => {
            AGENT.with(|a| {
                if let Some(s) = a.as_mut() {
                    s.selected = None;
                }
            });
            repaint_board_chrome();
            crate::serial_println!("ui_agent> deselected");
        }
        Some(from) => match chess_rules::try_move(&fen, &from, sq) {
            Ok(new_fen) => {
                let check = if chess_rules::in_check(&new_fen) {
                    " (check!)"
                } else {
                    ""
                };
                AGENT.with(|a| {
                    if let Some(s) = a.as_mut() {
                        s.fen = new_fen;
                        s.selected = None;
                        // Park cursor on the destination square after a move.
                        if let Some((f, r)) = chess_rules::parse_square(sq) {
                            s.cursor = (f, r);
                        }
                        persist_fen(s.agent_id, &s.fen);
                    }
                });
                repaint_board_chrome();
                crate::serial_println!("ui_agent> moved {from} → {sq}{check}");
            }
            Err(e) => {
                AGENT.with(|a| {
                    if let Some(s) = a.as_mut() {
                        s.selected = None;
                    }
                });
                let (task, surface) = match AGENT.with(|a| a.as_ref().map(|s| (s.task, s.surface))) {
                    Some(x) => x,
                    None => return,
                };
                call_board_set(task, surface, &fen);
                call_board_mark(task, surface, sq, "aa3333");
                let cur = AGENT.with(|a| {
                    a.as_ref()
                        .map(|s| chess_rules::square_name(s.cursor.0, s.cursor.1))
                });
                if let Some(c) = cur {
                    call_board_mark(task, surface, &c, CURSOR_COLOR);
                }
                crate::serial_println!("ui_agent> {e}");
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn step_cursor_clamps_to_board() {
        assert_eq!(step_cursor((0, 0), -1, -1), (0, 0));
        assert_eq!(step_cursor((7, 7), 1, 1), (7, 7));
        assert_eq!(step_cursor((4, 1), 0, 1), (4, 2)); // e2 → e3
        assert_eq!(step_cursor((4, 1), -1, 0), (3, 1)); // e2 → d2
        assert_eq!(step_cursor((0, 3), 1, 0), (1, 3)); // a4 → b4
    }
}
