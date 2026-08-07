//! The blocking **login gate** — the screen that will not go away until the
//! right password is typed.
//!
//! `#[cfg(not(test))]` with a deny-by-default stub, matching [`crate::modal`]:
//! there is no framebuffer in the test build, and a gate that returned `true`
//! there would be a gate that opens itself.
//!
//! # Why this is not `modal::input`
//!
//! Two reasons, either of which alone would be enough:
//!
//! - **`modal::input` renders only to the framebuffer.** Nothing reaches the
//!   UART. `cargo xtask run` drives this OS over `-serial mon:stdio` and so does
//!   the whole e2e harness, and to both of them a framebuffer-only gate is a
//!   machine that has silently stopped responding — no prompt, no echo, nothing
//!   to wait for. The gate must mirror to serial or it bricks the dev loop.
//! - **It returns an empty string on Esc.** For a text field that is a cancel;
//!   for an auth prompt it is an *empty password attempt*, and the distinction
//!   is the whole security property.
//!
//! # Ctrl+C does not dismiss this
//!
//! The standing rule is that Ctrl+C interrupts every command and every
//! long-running loop. **This loop is the documented exception**, because a gate
//! you can Ctrl+C out of is not a gate. Ctrl+C and Esc stay *responsive* — they
//! clear the typed buffer and repaint immediately, so the key is never dead —
//! but neither returns. Do not "fix" this.
//!
//! # The pump is `status_tick`, and that is load-bearing
//!
//! Like every modal, this loop consumes its own input, so it pumps
//! [`crate::shell::status_tick`] rather than `upkeep()` (whose `mouse::tick()`
//! would steal the clicks). A security property falls out of that and should not
//! be optimised away: the shell task stays **runnable** throughout, so the
//! scheduler never reaches the idle pump task — which is the only thing that
//! calls `upkeep()` — so `msgchan::tick`, `schedule::tick`,
//! `service::supervise_tick` and `tools::bg::pump` do not fire behind the lock
//! screen. A Telegram DM cannot drive an agent turn while the console is locked.
//! Swapping in `upkeep()` would look like a harmless cleanup and would silently
//! open that door.

use super::Reason;

/// Longest password the field will accept. Matches
/// [`super::MAX_PASSWORD_LEN`]; a longer paste is truncated rather than
/// allocating without bound from a held key.
const FIELD_MAX: usize = super::MAX_PASSWORD_LEN;

/// The banner line, per reason. `const` strings rather than a formatted message
/// so an e2e scenario can `wait_for` an exact literal.
fn banner(reason: Reason) -> &'static str {
    match reason {
        Reason::Boot => "chitti login: locked (boot)",
        Reason::Manual => "chitti login: locked (manual)",
        Reason::Idle => "chitti login: locked (idle)",
        Reason::Resume => "chitti login: locked (resume)",
    }
}

/// Block until the correct password is entered.
///
/// Returns `true` when unlocked. Returns `false` **only** when there is nothing
/// to authenticate against (no password enrolled, or a record that will not
/// parse) — in which case the caller carries on, because refusing to boot a
/// machine whose credential file got corrupted would be unrecoverable.
#[cfg(not(test))]
pub fn gate(reason: Reason) -> bool {
    use alloc::string::String;

    let Some(rec) = super::store::load() else {
        // `store::refresh` has already ktraced the corrupt-record case.
        return false;
    };

    super::set_locked(true);
    crate::ktrace::log_fmt(format_args!("auth: console locked ({})", reason.as_str()));

    // Serial side, once. `write_str_raw` rather than `serial_println!` because
    // the latter also paints into the chat grid, which is not the input surface
    // while a modal owns the screen (the composer prompt uses raw for the same
    // reason).
    crate::serial::write_str_raw("\r\n");
    crate::serial::write_str_raw(banner(reason));
    crate::serial::write_str_raw("\r\npassword: ");

    // Framebuffer side, once. Blank the desktop *before* the card so the old
    // transcript is not legible around it.
    crate::framebuffer::lock_cover();

    let title = banner(reason);
    let mut buf = String::new();
    let mut caret_on = true;
    let mut last_blink = crate::arch::now_ms();
    crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);

    loop {
        if let Some(b) = crate::console::read_byte() {
            match b {
                b'\r' | b'\n' => {
                    if try_unlock(&rec, &buf) {
                        return true;
                    }
                    buf.clear();
                    crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
                    crate::serial::write_str_raw("password: ");
                }
                // A bare Esc clears the field; an arrow-key CSI is consumed and
                // ignored. Neither leaves the gate — see the module doc.
                0x1b => {
                    if esc_seq().is_none() {
                        buf.clear();
                        crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
                    }
                }
                0x03 => {
                    buf.clear();
                    crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
                }
                0x7f | 0x08 => {
                    buf.pop();
                    crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
                }
                // Printable ASCII only — the same range `validate_new` enforces
                // at enrolment, so anything enrolled can always be typed back.
                // Serial echo is suppressed (the sudo convention); the
                // framebuffer shows a dot per character via `masked`.
                0x20..=0x7e => {
                    if buf.len() < FIELD_MAX {
                        buf.push(b as char);
                        crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
                    }
                }
                _ => {}
            }
        }
        // Blink the field caret ~2 Hz. Only the card is repainted — never
        // `lock_cover`, which is a whole-panel fill.
        let now = crate::arch::now_ms();
        if now.saturating_sub(last_blink) >= 500 {
            last_blink = now;
            caret_on = !caret_on;
            crate::framebuffer::draw_input(title, "password:", &buf, true, caret_on);
        }
        crate::shell::status_tick();
        crate::sched::yield_now();
    }
}

/// Check one attempt, applying the backoff on failure. `true` unlocks.
#[cfg(not(test))]
fn try_unlock(rec: &super::Credential, attempt: &str) -> bool {
    // The KDF is deliberately slow, so pump while it runs or the clock, caret
    // and net stack freeze for the duration of every attempt.
    if super::verify(rec, attempt, &mut crate::shell::status_tick) {
        super::reset_failures();
        super::set_locked(false);
        crate::ktrace::log("auth", "console unlocked");
        crate::serial::write_str_raw("\r\nlogin> unlocked\r\n");
        crate::framebuffer::modal_dismiss();
        return true;
    }
    let n = super::note_failure();
    super::store::note_failed_attempt();
    crate::ktrace::log_fmt(format_args!("auth: failed login attempt ({n} consecutive)"));
    crate::serial::write_str_raw("\r\nlogin> incorrect password\r\n");
    let wait = super::backoff_ms(n);
    if wait > 0 {
        crate::serial::write_str_raw("login> too many attempts, waiting\r\n");
        backoff(wait);
    }
    false
}

/// Sleep out the backoff, pumping the UI and **discarding** anything typed
/// meanwhile — otherwise typing ahead during the delay buys the attacker their
/// attempts back the moment it ends.
#[cfg(not(test))]
fn backoff(ms: u64) {
    let until = crate::arch::now_ms().saturating_add(ms);
    while crate::arch::now_ms() < until {
        while crate::console::read_byte().is_some() {}
        crate::shell::status_tick();
        crate::sched::yield_now();
    }
}

/// After a `0x1b`, decode a CSI sequence: the final byte if this was
/// `ESC [ … <final>`, or `None` for a bare Esc. Bounded busy-wait, matching the
/// shell/editor/modal decoders — the continuation bytes of an arrow key are
/// still in flight over serial when the ESC arrives.
#[cfg(not(test))]
fn esc_seq() -> Option<u8> {
    let next = seq_byte()?;
    if next != b'[' {
        return None;
    }
    loop {
        match seq_byte() {
            Some(b @ 0x40..=0x7e) => return Some(b),
            Some(_) => {}
            None => return None,
        }
    }
}

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

/// Test stub: no framebuffer, so the gate can never be satisfied. Deny by
/// default, exactly as `modal::confirm` does — a stub that returned `true` would
/// make every test run against an unlocked machine and prove nothing.
#[cfg(test)]
pub fn gate(_reason: Reason) -> bool {
    false
}
