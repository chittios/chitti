//! The chess **game/UI** — everything app-specific lives here in the package
//! wasm, driven by the kernel's *generic* package-UI runtime:
//!
//! * `chess_start` paints the board and restores a saved game.
//! * `on_click {x,y}` / `on_key {key}` implement select → move (two-click or
//!   arrows+Enter), `n` new game, Esc deselect.
//! * After a legal human (White) move with a model available, the export
//!   returns an **`ask:` request**: the runtime runs one model turn over the
//!   agent's SOUL and hands the text back via `on_reply` — where WE parse it
//!   against the native legal-move list and validate with the rules engine.
//!   The model only ever *chooses from moves this module generated*; an
//!   unusable reply falls back to the first legal move.
//! * `tick` animates the "thinking" dots while an ask is in flight.
//!
//! Painting goes through the host board primitives (`host_board_set` /
//! `host_board_mark` — the kernel owns board pixels) plus `host_ui_draw` text
//! ops for the HUD strip. State lives in plain-array statics (the runtime keeps
//! one persistent instance per app session; the bump heap resets per call, so
//! no static may hold a heap type).

use crate::rules;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// Board geometry — mirrors `kernel/src/synapse/ui.rs` (BOARD_SQ/OX/OY); only
// click mapping uses these, the board itself is painted host-side by
// `host_board_set`. The HUD is a reserved pane-space strip (`hud_set`), not in
// the surface, so the board fills the full 192-px height.
const SQ: i32 = 24;
const OX: i32 = 32;
const OY: i32 = 0;

/// The static shortcut line shown under the status in the HUD.
const SHORTCUTS: &str = "arrows/click move  enter select  esc clear  n new game";

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SELECT_COLOR: &str = "cc785c";
const CURSOR_COLOR: &str = "6688cc";
const ILLEGAL_COLOR: &str = "aa3333";

// --- persistent state (plain arrays only — see module doc) ------------------
static mut FEN: [u8; 100] = [0; 100];
static mut FEN_LEN: usize = 0;
/// Selected square (file, rank), 0xff = none.
static mut SEL: (u8, u8) = (0xff, 0xff);
/// Keyboard cursor (file, rank).
static mut CUR: (u8, u8) = (4, 1);
/// A model opponent is available (from the runtime's `model` start flag).
static mut MODEL: u8 = 0;
/// An `ask:` (agent move) is in flight — inputs are ignored until `on_reply`.
static mut WAITING: u8 = 0;

fn fen_get() -> String {
    unsafe {
        if FEN_LEN == 0 {
            return START_FEN.to_string();
        }
        core::str::from_utf8(&FEN[..FEN_LEN]).unwrap_or(START_FEN).to_string()
    }
}

fn fen_set(fen: &str) {
    let b = fen.as_bytes();
    let n = b.len().min(unsafe { FEN.len() });
    unsafe {
        FEN[..n].copy_from_slice(&b[..n]);
        FEN_LEN = n;
    }
    // Persist across restarts (durable) and expose to chat tools (session).
    crate::storage_set(1, "fen", fen);
    crate::storage_set(0, "fen", fen);
}

pub fn current_fen() -> String {
    fen_get()
}

/// Record a new position arriving from the chat tools (`chess_try_move`).
pub fn note_external_fen(fen: &str) {
    fen_set(fen);
    unsafe { SEL = (0xff, 0xff) };
}

fn white_to_move(fen: &str) -> bool {
    fen.split_whitespace().nth(1) != Some("b")
}

/// Every legal move for the side to move, as (from, to) square names.
fn all_legal_moves(fen: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(board) = rules::Board::from_fen(fen) else {
        return out;
    };
    for rank in 0..8u8 {
        for file in 0..8u8 {
            let from = rules::square_name(file, rank);
            for to in board.legal_to_squares(&from) {
                out.push((from.clone(), to));
            }
        }
    }
    out
}

/// First legal move named in `reply` (uci `e7e5`, dashed `e7-e5`, or spaced).
fn parse_move_reply(reply: &str, legal: &[(String, String)]) -> Option<(String, String)> {
    let lower = reply.to_ascii_lowercase();
    for (from, to) in legal {
        if lower.contains(&format!("{from}{to}"))
            || lower.contains(&format!("{from}-{to}"))
            || lower.contains(&format!("{from} {to}"))
        {
            return Some((from.clone(), to.clone()));
        }
    }
    None
}

// --- painting ----------------------------------------------------------------

/// Set the HUD (status line + shortcut line). Rendered crisp in the reserved
/// pane-space strip by the compositor — `dots` animates a thinking indicator.
fn paint_hud(status: &str, dots: &str) {
    crate::hud_set(&format!("{status}{dots}\n{SHORTCUTS}"));
}

/// Repaint everything: board (host primitive), selection + legal marks,
/// cursor, HUD.
fn paint(status: &str) {
    let fen = fen_get();
    crate::board_set(&fen);
    unsafe {
        if SEL.0 != 0xff {
            let sel = rules::square_name(SEL.0, SEL.1);
            let legal = rules::legal_moves(&fen, &sel);
            if legal != "none" {
                crate::board_mark(&format!("{sel},{legal}"), SELECT_COLOR);
            } else {
                crate::board_mark(&sel, SELECT_COLOR);
            }
        }
        let cur = rules::square_name(CUR.0, CUR.1);
        crate::board_mark(&cur, CURSOR_COLOR);
    }
    paint_hud(status, "");
}

fn turn_status(fen: &str) -> String {
    let model = unsafe { MODEL != 0 };
    match (white_to_move(fen), model) {
        (true, true) => "Your move (White)".to_string(),
        (false, true) => "Agent to move (Black)".to_string(),
        (true, false) => "White to move (hotseat)".to_string(),
        (false, false) => "Black to move (hotseat)".to_string(),
    }
}

// --- game flow -----------------------------------------------------------------

pub fn start(args: &str) -> String {
    unsafe {
        MODEL = if args.contains("\"model\":true") { 1 } else { 0 };
        WAITING = 0;
        SEL = (0xff, 0xff);
        CUR = (4, 1);
    }
    // Restore a saved game (durable storage), else the start position.
    let mut buf = [0u8; 128];
    let n = crate::storage_get(1, "fen", &mut buf);
    let fen = if n > 0 {
        core::str::from_utf8(&buf[..n as usize]).unwrap_or(START_FEN).to_string()
    } else {
        START_FEN.to_string()
    };
    fen_set(&fen);
    paint(&turn_status(&fen));
    // A restored game may already be on the agent's turn.
    if unsafe { MODEL != 0 } && !white_to_move(&fen) {
        return agent_ask(&fen);
    }
    format!("ok:chess you play White{}", if unsafe { MODEL != 0 } { ", agent answers as Black" } else { " and Black (hotseat)" })
}

fn new_game() -> String {
    unsafe {
        SEL = (0xff, 0xff);
        CUR = (4, 1);
        WAITING = 0;
    }
    fen_set(START_FEN);
    paint("New game - your move (White)");
    String::from("ok:new game")
}

/// Build the `ask:` request for the agent's (Black's) reply move.
fn agent_ask(fen: &str) -> String {
    let legal = all_legal_moves(fen);
    if legal.is_empty() {
        let over = if rules::in_check(fen) {
            "Checkmate - you win! (n = new game)"
        } else {
            "Stalemate - draw (n = new game)"
        };
        paint(over);
        return format!("ok:{over}");
    }
    unsafe { WAITING = 1 };
    paint_hud("Agent thinking", "");
    let mut menu = String::new();
    for (i, (f, t)) in legal.iter().enumerate() {
        if i > 0 {
            menu.push(' ');
        }
        menu.push_str(f);
        menu.push_str(t);
    }
    format!(
        "ask:Position (FEN): {fen}\nYou play Black. Legal moves (from+to): {menu}\n\
         Reply with exactly ONE move from that list in the same 4-character form (e.g. e7e5). No other text."
    )
}

/// The model's reply to an `ask:` — parse, validate, apply, repaint.
pub fn on_reply(args: &str) -> String {
    let text = crate::json_str(args, "text").unwrap_or_default();
    unsafe { WAITING = 0 };
    let fen = fen_get();
    let legal = all_legal_moves(&fen);
    if legal.is_empty() {
        paint(&turn_status(&fen));
        return String::from("ok:no moves");
    }
    let (from, to, fell_back) = match parse_move_reply(&text, &legal) {
        Some((f, t)) => (f, t, false),
        None => (legal[0].0.clone(), legal[0].1.clone(), true),
    };
    match rules::try_move(&fen, &from, &to) {
        Ok(new_fen) => {
            fen_set(&new_fen);
            unsafe { SEL = (0xff, 0xff) };
            let check = rules::in_check(&new_fen);
            let over = all_legal_moves(&new_fen).is_empty();
            let status = if over && check {
                format!("Agent: {from}{to} - CHECKMATE, agent wins (n = new)")
            } else if over {
                format!("Agent: {from}{to} - stalemate, draw (n = new)")
            } else if check {
                format!("Agent: {from}{to} - CHECK! Your move")
            } else if fell_back {
                format!("Agent: {from}{to} (fallback) - your move")
            } else {
                format!("Agent: {from}{to} - your move")
            };
            paint(&status);
            format!("ok:agent {from}{to}")
        }
        Err(e) => {
            paint(&turn_status(&fen));
            format!("error:agent move rejected {e}")
        }
    }
}

/// Select / move on `sq` — the shared core of clicks and Enter.
fn activate(sq: &str) -> String {
    let fen = fen_get();
    let sel = unsafe { SEL };
    if let Some((f, r)) = rules::parse_square(sq) {
        unsafe { CUR = (f, r) };
    }
    if sel.0 == 0xff {
        // First pick: select if the square has legal moves.
        let legal = rules::legal_moves(&fen, sq);
        if legal == "none" {
            paint(&format!("No legal moves from {sq}"));
            crate::board_mark(sq, ILLEGAL_COLOR);
            return format!("ok:none {sq}");
        }
        if let Some((f, r)) = rules::parse_square(sq) {
            unsafe { SEL = (f, r) };
        }
        paint(&format!("{sq} selected - pick a highlighted square"));
        return format!("ok:selected {sq}");
    }
    let from = rules::square_name(sel.0, sel.1);
    if from == sq {
        unsafe { SEL = (0xff, 0xff) };
        paint(&turn_status(&fen));
        return String::from("ok:deselected");
    }
    match rules::try_move(&fen, &from, sq) {
        Ok(new_fen) => {
            fen_set(&new_fen);
            unsafe { SEL = (0xff, 0xff) };
            let check = if rules::in_check(&new_fen) { " CHECK!" } else { "" };
            if unsafe { MODEL != 0 } {
                // Show the human move, then hand the position to the agent.
                paint(&format!("You: {from}{sq}{check}"));
                return agent_ask(&new_fen);
            }
            let over = all_legal_moves(&new_fen).is_empty();
            let status = if over && !check.is_empty() {
                format!("You: {from}{sq} - CHECKMATE (n = new)")
            } else if over {
                format!("You: {from}{sq} - stalemate (n = new)")
            } else {
                format!("You: {from}{sq}{check} - {}", turn_status(&new_fen))
            };
            paint(&status);
            format!("ok:moved {from}{sq}")
        }
        Err(e) => {
            unsafe { SEL = (0xff, 0xff) };
            paint(&format!("Illegal: {from} to {sq}"));
            crate::board_mark(sq, ILLEGAL_COLOR);
            e
        }
    }
}

pub fn on_click(x: i32, y: i32) -> String {
    if unsafe { WAITING != 0 } {
        return String::from("ok:busy");
    }
    let fx = (x - OX) / SQ;
    let ry = (y - OY) / SQ;
    if x < OX || y < OY || !(0..8).contains(&fx) || !(0..8).contains(&ry) {
        return String::from("ok:off-board");
    }
    // Board is drawn rank 8 at top.
    let sq = rules::square_name(fx as u8, (7 - ry) as u8);
    activate(&sq)
}

pub fn on_key(key: &str) -> String {
    if unsafe { WAITING != 0 } {
        return String::from("ok:busy");
    }
    let (df, dr): (i8, i8) = match key {
        "up" => (0, 1),
        "down" => (0, -1),
        "left" => (-1, 0),
        "right" => (1, 0),
        "enter" | "space" => {
            let (f, r) = unsafe { CUR };
            return activate(&rules::square_name(f, r));
        }
        "esc" => {
            unsafe { SEL = (0xff, 0xff) };
            let fen = fen_get();
            paint(&turn_status(&fen));
            return String::from("ok:deselected");
        }
        "n" => return new_game(),
        _ => return String::from("ok"),
    };
    unsafe {
        let f = (CUR.0 as i16 + df as i16).clamp(0, 7) as u8;
        let r = (CUR.1 as i16 + dr as i16).clamp(0, 7) as u8;
        CUR = (f, r);
    }
    let fen = fen_get();
    let sel = unsafe { SEL };
    let status = if sel.0 != 0xff {
        format!("{} selected - pick a highlighted square", rules::square_name(sel.0, sel.1))
    } else {
        turn_status(&fen)
    };
    paint(&status);
    String::from("ok:cursor")
}

/// Animate the thinking dots while an ask is in flight (runtime tick).
pub fn tick() -> String {
    if unsafe { WAITING == 0 } {
        return String::from("ok:idle");
    }
    let phase = (crate::now_ms() / 300 % 4) as usize;
    paint_hud("Agent thinking", &"...."[..phase]);
    String::from("ok:thinking")
}
