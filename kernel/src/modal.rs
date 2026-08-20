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

/// Pump audio/video/agents without stealing this loop's mouse. Returns true
/// when a present painted over the popup, so the caller must redraw it.
#[cfg(not(test))]
fn pump_under_popup() -> bool {
    crate::shell::background_tick();
    crate::framebuffer::take_modal_dirty()
}

/// Ask a yes/no question. Returns `true` only on an explicit yes (default No —
/// safe for destructive confirmations). Keyboard: **Enter** confirms the
/// focused button, **Esc** (or Ctrl-C) cancels, **arrows/Tab** move focus
/// between Yes/No (`y`/`n` are also quick shortcuts). Mouse: click a button.
#[cfg(not(test))]
pub fn confirm(title: &str, msg: &str) -> bool {
    use crate::framebuffer::{self, ModalHit};
    let mut focus_yes = false;
    framebuffer::draw_confirm(title, msg, focus_yes);
    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                // ESC is ambiguous: a bare Esc keypress (cancel), or the start
                // of a CSI arrow sequence. Peek — an arrow just moves focus; a
                // real Esc cancels. This stops arrow keys from reading as "No".
                0x1b => match esc_seq() {
                    // Left/Right/Up/Down toggle which button is focused.
                    Some(b'C') | Some(b'D') | Some(b'A') | Some(b'B') => {
                        focus_yes = !focus_yes;
                        framebuffer::draw_confirm(title, msg, focus_yes);
                    }
                    Some(_) => {} // other CSI (Home/End/PgUp…): ignore
                    None => return finish(false), // bare Esc = cancel
                },
                0x03 => return finish(false), // Ctrl+C = cancel
                b'y' | b'Y' => return finish(true),
                b'n' | b'N' => return finish(false),
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
        if pump_under_popup() {
            framebuffer::draw_confirm(title, msg, focus_yes);
        }
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
                // A bare Esc cancels; an arrow-key CSI is consumed + ignored (so
                // arrows don't cancel a half-typed field).
                0x1b => {
                    if esc_seq().is_none() {
                        framebuffer::modal_dismiss();
                        return String::new();
                    }
                }
                0x03 => {
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
        if pump_under_popup() {
            framebuffer::draw_input(title, prompt, &buf, masked, caret_on);
        }
        crate::sched::yield_now();
    }
}

#[cfg(not(test))]
fn finish(v: bool) -> bool {
    crate::framebuffer::modal_dismiss();
    v
}

/// After a `0x1b` byte, decode a CSI arrow/nav sequence: returns the final byte
/// (`A`/`B`/`C`/`D`/…) if the ESC was `ESC [ … <final>`, or `None` for a bare
/// Esc keypress. Bounded busy-wait (the continuation bytes of an arrow key are
/// still in flight over serial), matching the shell/editor decoders.
#[cfg(not(test))]
fn esc_seq() -> Option<u8> {
    // Is the next byte a '[' (CSI introducer)? If nothing arrives, it was a
    // bare Esc.
    let next = seq_byte()?;
    if next != b'[' {
        return None;
    }
    // Consume params up to the final byte (0x40..=0x7e).
    loop {
        match seq_byte() {
            Some(b @ 0x40..=0x7e) => return Some(b),
            Some(_) => {}    // parameter/intermediate byte
            None => return None,
        }
    }
}

/// Bounded read of one console byte (for coalescing an ANSI escape sequence).
#[cfg(not(test))]
fn seq_byte() -> Option<u8> {
    for _ in 0..2000 {
        if let Some(b) = crate::console::read_byte() {
            return Some(b);
        }
        crate::sched::yield_now();
    }
    None
}

/// Test stub: no framebuffer, so deny by default.
#[cfg(test)]
pub fn confirm(_title: &str, _msg: &str) -> bool {
    false
}

/// Whether a modal is on screen. Anything that would consume console input or
/// start work behind one must defer rather than race it.
#[cfg(not(test))]
pub fn is_open() -> bool {
    crate::framebuffer::modal_is_open()
}

/// Test stub: no framebuffer, so never.
#[cfg(test)]
pub fn is_open() -> bool {
    false
}

/// Test stub: no framebuffer, so return empty.
#[cfg(test)]
pub fn input(_title: &str, _prompt: &str, _masked: bool) -> String {
    String::new()
}

/// macOS-style **About ChittiOS** dialog (logo, version, build, arch, tagline).
/// Dismiss with Enter / Esc / Ctrl+C / OK click / close mark. Used by `/about`
/// and a click on the status-bar brand.
#[cfg(not(test))]
pub fn about() {
    use crate::framebuffer::{self, ModalHit};
    framebuffer::draw_about();
    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'\r' | b'\n' | 0x03 => {
                    framebuffer::modal_dismiss();
                    return;
                }
                0x1b => {
                    // Bare Esc dismisses; arrow CSI is ignored.
                    if esc_seq().is_none() {
                        framebuffer::modal_dismiss();
                        return;
                    }
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
            // Hover chrome on the popup's interactive rows (repaint on change).
            if framebuffer::update_popup_hover(t.x, t.y) {
                framebuffer::draw_about();
            }
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Ok | ModalHit::Close => {
                    framebuffer::modal_dismiss();
                    return;
                }
                _ => {}
            }
        }
        if pump_under_popup() {
            framebuffer::draw_about();
        }
        crate::sched::yield_now();
    }
}

/// Test stub: no framebuffer About dialog.
#[cfg(test)]
pub fn about() {}

/// macOS-style **status-bar dropdown** for a chip (net / clock / mem / volume …).
/// Live-refreshes the clock once a second; the volume menu repaints on change.
/// Dismiss with Esc, Ctrl+C, close mark, or a click outside the panel.
#[cfg(not(test))]
pub fn status_menu(chip: crate::framebuffer::StatusChip) {
    use crate::framebuffer::{self, ModalHit, StatusChip};
    // Brand is About, not a dropdown.
    if chip == StatusChip::Brand {
        about();
        return;
    }
    let mut last_paint = 0u64;
    let mut need_repaint = true;
    loop {
        let now = crate::arch::now_ms();
        // Clock ticks every second; volume / hover / a present that painted
        // over us set `need_repaint`. Other chips used to paint only once,
        // so the audio analyser (and any agent UI) sat on top of Keyboard
        // / Net / … until the menu was dismissed — Help/Agents already
        // redrew immediately after `pump_under_popup`.
        let due = need_repaint
            || last_paint == 0
            || (chip == StatusChip::Clock && now.saturating_sub(last_paint) >= 1000);
        if due {
            framebuffer::draw_status_menu(chip);
            last_paint = now;
            need_repaint = false;
        }
        if let Some(b) = crate::console::read_byte() {
            match b {
                0x03 => {
                    framebuffer::modal_dismiss();
                    return;
                }
                0x1b => {
                    if let Some(fin) = esc_seq() {
                        // Arrows adjust volume while the Sound menu is open.
                        if chip == StatusChip::Volume {
                            match fin {
                                b'C' | b'A' => {
                                    crate::sound::volume_adjust(5);
                                    need_repaint = true;
                                }
                                b'D' | b'B' => {
                                    crate::sound::volume_adjust(-5);
                                    need_repaint = true;
                                }
                                _ => {}
                            }
                        }
                    } else {
                        framebuffer::modal_dismiss();
                        return;
                    }
                }
                b'm' | b'M' if chip == StatusChip::Volume => {
                    crate::sound::toggle_mute();
                    need_repaint = true;
                }
                b'+' | b'=' if chip == StatusChip::Volume => {
                    crate::sound::volume_adjust(5);
                    need_repaint = true;
                }
                b'-' | b'_' if chip == StatusChip::Volume => {
                    crate::sound::volume_adjust(-5);
                    need_repaint = true;
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
            // Hover chrome on the dropdown's close mark / action rows.
            if framebuffer::update_popup_hover(t.x, t.y) {
                need_repaint = true;
            }
        }
        // Scroll wheel over the volume menu (or while it is open) adjusts level.
        if chip == StatusChip::Volume && t.wheel != 0 {
            // One notch = 5% (wheel > 0 = up / louder).
            let delta = t.wheel * 5;
            crate::sound::volume_adjust(delta);
            need_repaint = true;
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Close | ModalHit::Ok => {
                    framebuffer::modal_dismiss();
                    return;
                }
                ModalHit::Choose(i) if chip == StatusChip::Volume => {
                    match i {
                        0 => {
                            crate::sound::toggle_mute();
                        }
                        1 => {
                            crate::sound::volume_adjust(-5);
                        }
                        2 => {
                            crate::sound::volume_adjust(5);
                        }
                        _ => {}
                    }
                    need_repaint = true;
                }
                // Clicks on the panel body (slot 1) keep the menu open; anywhere
                // else (including the status bar) dismisses.
                ModalHit::No if framebuffer::status_menu_contains(t.x, t.y) => {}
                _ => {
                    if !framebuffer::status_menu_contains(t.x, t.y) {
                        framebuffer::modal_dismiss();
                        return;
                    }
                }
            }
        }
        if pump_under_popup() {
            framebuffer::draw_status_menu(chip);
            last_paint = crate::arch::now_ms();
            need_repaint = false;
        }
        crate::sched::yield_now();
    }
}

/// Test stub: no status-bar dropdowns (framebuffer is not linked under `--test`).
#[cfg(test)]
pub fn status_menu(_chip: u8) {}

/// Multi-option question for agents (`ask_user_question`). Returns the chosen
/// option index, or `None` if cancelled. Keyboard: arrows/Tab move, Enter
/// selects, 1..9 jump, Esc cancels. Mouse: click a row to select it.
#[cfg(not(test))]
pub fn choose(title: &str, question: &str, options: &[&str]) -> Option<usize> {
    use crate::framebuffer::{self, ModalHit};
    if options.is_empty() {
        return None;
    }
    let mut focus = 0usize;
    let n = options.len();
    framebuffer::draw_choose(title, question, options, focus);
    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                0x1b => match esc_seq() {
                    Some(b'A') | Some(b'D') => {
                        focus = focus.checked_sub(1).unwrap_or(n - 1);
                        framebuffer::draw_choose(title, question, options, focus);
                    }
                    Some(b'B') | Some(b'C') => {
                        focus = (focus + 1) % n;
                        framebuffer::draw_choose(title, question, options, focus);
                    }
                    Some(_) => {}
                    None => {
                        framebuffer::modal_dismiss();
                        return None;
                    }
                },
                0x03 => {
                    framebuffer::modal_dismiss();
                    return None;
                }
                b'\t' => {
                    focus = (focus + 1) % n;
                    framebuffer::draw_choose(title, question, options, focus);
                }
                b'\r' | b'\n' => {
                    framebuffer::modal_dismiss();
                    return Some(focus);
                }
                b'1'..=b'9' => {
                    let i = (b - b'1') as usize;
                    if i < n {
                        framebuffer::modal_dismiss();
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
            // Hover chrome on the option rows (repaint on change).
            if framebuffer::update_popup_hover(t.x, t.y) {
                framebuffer::draw_choose(title, question, options, focus);
            }
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Choose(i) if i < n => {
                    framebuffer::modal_dismiss();
                    return Some(i);
                }
                _ => {}
            }
        }
        if pump_under_popup() {
            framebuffer::draw_choose(title, question, options, focus);
        }
        crate::sched::yield_now();
    }
}

#[cfg(test)]
pub fn choose(_title: &str, _question: &str, options: &[&str]) -> Option<usize> {
    // Tests auto-pick the first option when present.
    if options.is_empty() {
        None
    } else {
        Some(0)
    }
}

/// Open the searchable **Commands** browser (the `/help` modal). Returns the
/// selected command **name** (without `/`) on Enter or a list-row click, or
/// `None` if dismissed. Typing filters the list; ↑/↓ move the highlight
/// (skipping category headers); PgUp/PgDn / mouse wheel page; Esc / Ctrl+C /
/// `[x]` cancel.
#[cfg(not(test))]
pub fn browse_commands() -> Option<String> {
    use crate::framebuffer::{self, CommandsRow, ModalHit};
    use crate::shell::catalog::{self, Row};
    use alloc::string::String;
    use alloc::vec::Vec;

    const VIEW: usize = 12;
    let mut query = String::new();
    let mut rows = catalog::filter_rows("");
    let mut sel = catalog::first_sel(&rows);
    let mut scroll = 0usize;
    let mut caret_on = true;
    let mut last_blink = crate::arch::now_ms();

    fn paint(query: &str, rows: &[Row], sel: usize, scroll: usize, caret_on: bool) {
        let scroll = scroll.min(rows.len());
        let end = (scroll + VIEW).min(rows.len());
        let slice = &rows[scroll..end];
        let mut slash_buf: Vec<String> = Vec::new();
        for r in slice {
            if let Row::Item { name, .. } = r {
                slash_buf.push(alloc::format!("/{name}"));
            }
        }
        let mut si = 0usize;
        let mut view: Vec<CommandsRow<'_>> = Vec::new();
        for (i, r) in slice.iter().enumerate() {
            let abs = scroll + i;
            match r {
                Row::Header(h) => view.push(CommandsRow::Header(h.as_str())),
                Row::Item { title, shortcut, .. } => {
                    let slash = slash_buf[si].as_str();
                    si += 1;
                    view.push(CommandsRow::Item {
                        title: title.as_str(),
                        slash,
                        shortcut: shortcut.as_str(),
                        selected: abs == sel,
                    });
                }
            }
        }
        framebuffer::draw_commands_browser(query, &view, scroll, rows.len(), caret_on);
    }

    fn refilter(query: &str) -> (Vec<Row>, usize, usize) {
        let rows = catalog::filter_rows(query);
        let sel = catalog::first_sel(&rows);
        let scroll = catalog::clamp_scroll(sel, 0, VIEW, rows.len());
        (rows, sel, scroll)
    }

    paint(&query, &rows, sel, scroll, caret_on);

    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'\r' | b'\n' => {
                    if let Some(name) = catalog::name_at(&rows, sel) {
                        let n = String::from(name);
                        crate::framebuffer::modal_dismiss();
                        return Some(n);
                    }
                }
                0x1b => match esc_seq_param() {
                    // ↑ / ↓
                    Some((0, b'A')) => {
                        sel = catalog::move_sel(&rows, sel, -1);
                        scroll = catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some((0, b'B')) => {
                        sel = catalog::move_sel(&rows, sel, 1);
                        scroll = catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    // PgUp / PgDn
                    Some((5, b'~')) => {
                        sel = catalog::move_sel(&rows, sel, -(VIEW as i32));
                        scroll = catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some((6, b'~')) => {
                        sel = catalog::move_sel(&rows, sel, VIEW as i32);
                        scroll = catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some(_) => {}
                    None => {
                        crate::framebuffer::modal_dismiss();
                        return None;
                    }
                },
                0x03 => {
                    crate::framebuffer::modal_dismiss();
                    return None;
                }
                0x7f | 0x08 => {
                    query.pop();
                    let r = refilter(&query);
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                0x15 => {
                    // Ctrl+U: clear search
                    query.clear();
                    let r = refilter("");
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                0x20..=0x7e => {
                    query.push(b as char);
                    let r = refilter(&query);
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
            // Hover chrome on the command rows (repaint on change).
            if framebuffer::update_popup_hover(t.x, t.y) {
                paint(&query, &rows, sel, scroll, caret_on);
            }
        }
        if t.wheel != 0 {
            // Wheel up (+): previous items; wheel down (−): next.
            let steps = t.wheel.clamp(-3, 3);
            sel = catalog::move_sel(&rows, sel, -steps);
            scroll = catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
            paint(&query, &rows, sel, scroll, caret_on);
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Close => {
                    crate::framebuffer::modal_dismiss();
                    return None;
                }
                ModalHit::ListRow(abs) => {
                    // Click a selectable item → accept immediately (mouse-first).
                    // Headers are ignored so a mis-click doesn't dismiss.
                    if let Some(name) = catalog::name_at(&rows, abs) {
                        let n = String::from(name);
                        crate::framebuffer::modal_dismiss();
                        return Some(n);
                    }
                }
                _ => {}
            }
        }
        let now = crate::arch::now_ms();
        if now.saturating_sub(last_blink) >= 500 {
            last_blink = now;
            caret_on = !caret_on;
            paint(&query, &rows, sel, scroll, caret_on);
        }
        if pump_under_popup() {
            paint(&query, &rows, sel, scroll, caret_on);
        }
        crate::sched::yield_now();
    }
}

/// Like [`esc_seq`] but also returns the numeric CSI parameter (for PgUp=5,
/// PgDn=6). Bare Esc → `None`.
#[cfg(not(test))]
fn esc_seq_param() -> Option<(u64, u8)> {
    let next = seq_byte()?;
    if next != b'[' {
        return None;
    }
    let mut param: u64 = 0;
    loop {
        match seq_byte() {
            Some(b @ 0x40..=0x7e) => return Some((param, b)),
            Some(d @ b'0'..=b'9') => param = param.saturating_mul(10) + (d - b'0') as u64,
            Some(_) => {}
            None => return None,
        }
    }
}

/// Test stub: no framebuffer browser.
#[cfg(test)]
pub fn browse_commands() -> Option<String> {
    None
}

/// Open the searchable **Agents** browser (the `/agents` modal). Returns an
/// encoded pick on Enter or a list-row click:
/// * `switch:<id>` — rebind chat to a live task
/// * `ui:<name>` — start package UI (`/agents start <name>`)
/// * `shell:<name>` — rebind chat to that SOUL package
///
/// `None` if dismissed (Esc / Ctrl+C / `[x]`).
#[cfg(not(test))]
pub fn browse_agents() -> Option<String> {
    use crate::framebuffer::{self, CommandsRow, ModalHit};
    use crate::shell::agents_catalog;
    use crate::shell::catalog::Row;
    use alloc::string::String;
    use alloc::vec::Vec;

    const VIEW: usize = 12;

    // Drop residual keys (trailing CR from the `/agents` Enter, host paste
    // crumbs, etc.) so the modal does not auto-select or auto-dismiss.
    for _ in 0..64 {
        if crate::console::read_byte().is_none() {
            break;
        }
    }

    let mut query = String::new();
    let mut rows = agents_catalog::filter_rows("");
    if rows.is_empty() {
        // Should never happen (system agents are compiled in) — surface loudly.
        crate::serial_println!("agents> browser: empty catalogue (bug)");
        return None;
    }
    let mut sel = agents_catalog::first_sel(&rows);
    let mut scroll = 0usize;
    let mut caret_on = true;
    let mut last_blink = crate::arch::now_ms();
    // Ignore Enter for a short window so the same key that submitted `/agents`
    // cannot immediately pick the first row. Mouse clicks are armed immediately
    // (they are not residual from the command submit).
    let opened_ms = crate::arch::now_ms();
    let mut arm_enter = false;

    fn paint(query: &str, rows: &[Row], sel: usize, scroll: usize, caret_on: bool) {
        let scroll = scroll.min(rows.len());
        let end = (scroll + VIEW).min(rows.len());
        let slice = &rows[scroll..end];
        let mut kind_buf: Vec<String> = Vec::new();
        for r in slice {
            if let Row::Item { shortcut, .. } = r {
                kind_buf.push(shortcut.clone());
            }
        }
        let mut si = 0usize;
        let mut view: Vec<CommandsRow<'_>> = Vec::new();
        for (i, r) in slice.iter().enumerate() {
            let abs = scroll + i;
            match r {
                Row::Header(h) => view.push(CommandsRow::Header(h.as_str())),
                Row::Item { title, name, .. } => {
                    let kind = kind_buf.get(si).map(|s| s.as_str()).unwrap_or("");
                    si += 1;
                    let _ = name;
                    view.push(CommandsRow::Item {
                        title: title.as_str(),
                        slash: "",
                        shortcut: kind,
                        selected: abs == sel,
                    });
                }
            }
        }
        framebuffer::draw_list_browser("Agents", query, &view, scroll, rows.len(), caret_on);
    }

    fn refilter(query: &str) -> (Vec<Row>, usize, usize) {
        let rows = agents_catalog::filter_rows(query);
        let sel = agents_catalog::first_sel(&rows);
        let scroll = agents_catalog::clamp_scroll(sel, 0, VIEW, rows.len());
        (rows, sel, scroll)
    }

    crate::serial_println!(
        "agents> browser open ({} rows) — type to search, Enter/click select, Esc close",
        rows.len()
    );
    paint(&query, &rows, sel, scroll, caret_on);

    loop {
        // Arm Enter only after ~200 ms so the submit key can't auto-pick.
        if !arm_enter && crate::arch::now_ms().saturating_sub(opened_ms) >= 200 {
            arm_enter = true;
        }
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'\r' | b'\n' => {
                    if !arm_enter {
                        continue;
                    }
                    if let Some(name) = agents_catalog::name_at(&rows, sel) {
                        let n = String::from(name);
                        crate::framebuffer::modal_dismiss();
                        return Some(n);
                    }
                }
                0x1b => match esc_seq_param() {
                    Some((0, b'A')) => {
                        sel = agents_catalog::move_sel(&rows, sel, -1);
                        scroll = agents_catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some((0, b'B')) => {
                        sel = agents_catalog::move_sel(&rows, sel, 1);
                        scroll = agents_catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some((5, b'~')) => {
                        sel = agents_catalog::move_sel(&rows, sel, -(VIEW as i32));
                        scroll = agents_catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some((6, b'~')) => {
                        sel = agents_catalog::move_sel(&rows, sel, VIEW as i32);
                        scroll = agents_catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
                        paint(&query, &rows, sel, scroll, caret_on);
                    }
                    Some(_) => {}
                    None => {
                        crate::framebuffer::modal_dismiss();
                        return None;
                    }
                },
                0x03 => {
                    crate::framebuffer::modal_dismiss();
                    return None;
                }
                0x7f | 0x08 => {
                    query.pop();
                    let r = refilter(&query);
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                0x15 => {
                    query.clear();
                    let r = refilter("");
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                0x20..=0x7e => {
                    query.push(b as char);
                    let r = refilter(&query);
                    rows = r.0;
                    sel = r.1;
                    scroll = r.2;
                    paint(&query, &rows, sel, scroll, caret_on);
                }
                _ => {}
            }
        }
        let t = crate::mouse::tick();
        if t.moved {
            framebuffer::cursor_move(t.x, t.y);
            // Hover chrome on the agent rows (repaint on change).
            if framebuffer::update_popup_hover(t.x, t.y) {
                paint(&query, &rows, sel, scroll, caret_on);
            }
        }
        if t.wheel != 0 {
            let steps = t.wheel.clamp(-3, 3);
            sel = agents_catalog::move_sel(&rows, sel, -steps);
            scroll = agents_catalog::clamp_scroll(sel, scroll, VIEW, rows.len());
            paint(&query, &rows, sel, scroll, caret_on);
        }
        if t.pressed {
            match framebuffer::modal_hit(t.x, t.y) {
                ModalHit::Close => {
                    crate::framebuffer::modal_dismiss();
                    return None;
                }
                ModalHit::ListRow(abs) => {
                    if let Some(name) = agents_catalog::name_at(&rows, abs) {
                        let n = String::from(name);
                        crate::framebuffer::modal_dismiss();
                        return Some(n);
                    }
                }
                _ => {}
            }
        }
        // Blink the search caret less often than /help did for full repaints —
        // full modal redraws every 500 ms felt like flicker on large FB.
        let now = crate::arch::now_ms();
        if now.saturating_sub(last_blink) >= 800 {
            last_blink = now;
            caret_on = !caret_on;
            paint(&query, &rows, sel, scroll, caret_on);
        }
        if pump_under_popup() {
            paint(&query, &rows, sel, scroll, caret_on);
        }
        crate::sched::yield_now();
    }
}

/// Test stub: no framebuffer agents browser.
#[cfg(test)]
pub fn browse_agents() -> Option<String> {
    None
}
