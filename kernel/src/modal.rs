//! Approval / input **modals** — a blocking, framebuffer-drawn dialog the shell
//! (and, through it, agents/tools) use to ask the human for a yes/no decision or
//! a line of text (optionally masked, for passwords). Driven by **keyboard and
//! mouse**: a modal is a small event loop that draws the dialog, then polls the
//! console + pointer until the user resolves it, and repaints the UI on exit.
//!
//! Used by the destructive-action taint gate, skill-install consent, and the
//! `/wifi` password prompt; `confirm`/`input` are the primitives any caller
//! (including a tool the agent invokes) uses to get a human decision.
//!
//! In the test build (no framebuffer) these are safe stubs: `confirm` denies,
//! `input` returns empty.

use alloc::string::String;

/// Ask a yes/no question. Returns `true` only on an explicit yes (default No —
/// safe for destructive confirmations). Keyboard: `y`/`n`, Enter (the focused
/// button), Tab to switch focus, Esc/Ctrl-C = no. Mouse: click a button.
#[cfg(not(test))]
pub fn confirm(title: &str, msg: &str) -> bool {
    use crate::framebuffer::{self, ModalHit};
    let mut focus_yes = false;
    framebuffer::draw_confirm(title, msg, focus_yes);
    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'y' | b'Y' => return finish(true),
                b'n' | b'N' | 0x1b | 0x03 => return finish(false),
                b'\r' | b'\n' => return finish(focus_yes),
                b'\t' => {
                    focus_yes = !focus_yes;
                    framebuffer::draw_confirm(title, msg, focus_yes);
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Yes => return finish(true),
                ModalHit::No => return finish(false),
                _ => {}
            }
        }
        crate::shell::status_tick(); // status bar + net stay alive under the modal
        crate::sched::yield_now();
    }
}

/// Prompt for a line of text (masked = password dots). Enter/OK submits, Esc
/// cancels (returns empty). Keyboard types; mouse can click OK.
#[cfg(not(test))]
pub fn input(title: &str, prompt: &str, masked: bool) -> String {
    use crate::framebuffer::{self, ModalHit};
    let mut buf = String::new();
    let mut caret_on = true;
    let mut last_blink = crate::arch::now_ms();
    framebuffer::draw_input(title, prompt, &buf, masked, caret_on);
    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'\r' | b'\n' => {
                    framebuffer::modal_dismiss();
                    return buf;
                }
                0x1b | 0x03 => {
                    framebuffer::modal_dismiss();
                    return String::new();
                }
                0x7f | 0x08 => {
                    buf.pop();
                    framebuffer::draw_input(title, prompt, &buf, masked, caret_on);
                }
                0x20..=0x7e => {
                    buf.push(b as char);
                    framebuffer::draw_input(title, prompt, &buf, masked, caret_on);
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
        }
        if t.pressed && framebuffer::modal_hit(t.x, t.y) == ModalHit::Ok {
            framebuffer::modal_dismiss();
            return buf;
        }
        // Blink the field caret ~2 Hz.
        let now = crate::arch::now_ms();
        if now.saturating_sub(last_blink) >= 500 {
            last_blink = now;
            caret_on = !caret_on;
            framebuffer::draw_input(title, prompt, &buf, masked, caret_on);
        }
        crate::shell::status_tick(); // status bar + net stay alive under the modal
        crate::sched::yield_now();
    }
}

#[cfg(not(test))]
fn finish(v: bool) -> bool {
    crate::framebuffer::modal_dismiss();
    v
}

/// Test stub: no framebuffer, so deny by default.
#[cfg(test)]
pub fn confirm(_title: &str, _msg: &str) -> bool {
    false
}

/// Test stub: no framebuffer, so return empty.
#[cfg(test)]
pub fn input(_title: &str, _prompt: &str, _masked: bool) -> String {
    String::new()
}
