//! A single global clipboard shared by the editor (yank/paste) and the shell
//! (paste into the input line). `linewise` mirrors vim's register kind: a
//! line-wise yank (`yy`/`dd`) pastes as whole lines, a char-wise yank (visual
//! `y`) pastes inline.
//!
//! **Host clipboard bridge.** The clipboard also syncs with the host machine's
//! clipboard over the serial console, so you can copy in Chitti and paste on
//! the host (macOS/Linux) and vice-versa — no guest-additions driver required,
//! and it works the same on QEMU and VirtualBox:
//!
//! * **Copy out** — [`set`] emits an **OSC 52** escape (`ESC ] 52 ; c ; <b64>
//!   BEL`) to the serial terminal; a terminal with OSC-52 enabled (iTerm2,
//!   kitty, WezTerm, tmux with `set -g set-clipboard on`, …) copies it to the
//!   host clipboard.
//! * **Paste in** — the shell enables **bracketed paste** (`ESC[?2004h`); a
//!   host paste arrives wrapped in `ESC[200~ … ESC[201~`, which the line editor
//!   captures into the clipboard via [`set_from_host`] and inserts.
//!
//! (This rides the console serial line, so it needs the VM's console attached
//! to a terminal — `qemu … -serial mon:stdio`, or a VBox serial port routed to
//! a host terminal. The graphical window itself has no clipboard channel; that
//! would need a SPICE vdagent / VirtualBox Guest Additions agent.)

use crate::mm::Locked;
use alloc::string::String;

struct Clip {
    text: String,
    linewise: bool,
}

static CLIP: Locked<Option<Clip>> = Locked::new(None);

/// Replace the clipboard contents **and** push them to the host clipboard
/// (OSC 52). Used by every in-OS copy (editor yank, chat drag-select).
pub fn set(text: String, linewise: bool) {
    emit_osc52(&text);
    CLIP.with(|c| *c = Some(Clip { text, linewise }));
}

/// Replace the clipboard contents **without** echoing back to the host — used
/// when a *host* paste populates the clipboard (bracketed paste), so we don't
/// bounce it straight back out via OSC 52.
pub fn set_from_host(text: String) {
    CLIP.with(|c| *c = Some(Clip { text, linewise: false }));
}

/// The clipboard `(text, linewise)`, or `None` if empty.
pub fn get() -> Option<(String, bool)> {
    CLIP.with(|c| c.as_ref().map(|x| (x.text.clone(), x.linewise)))
}

/// Enable bracketed paste on the host terminal (`ESC[?2004h`), so a paste
/// arrives wrapped in `ESC[200~ … ESC[201~` and can be captured whole. Called
/// once at shell start. No-op in the test build (no real terminal).
pub fn enable_host_paste() {
    #[cfg(not(test))]
    for &b in b"\x1b[?2004h" {
        crate::serial::put_byte(b);
    }
}

/// Build the OSC 52 "set clipboard" escape for `text`: `ESC ] 52 ; c ; <base64>
/// BEL`. Pure (base64 + framing), so it is unit-tested. `c` selects the
/// clipboard (as opposed to the primary selection).
pub fn osc52_sequence(text: &str) -> String {
    let mut s = String::with_capacity(text.len() * 4 / 3 + 8);
    s.push_str("\x1b]52;c;");
    s.push_str(&crate::net::ws::base64_encode(text.as_bytes()));
    s.push('\x07'); // BEL terminates the OSC
    s
}

/// Write the OSC 52 sequence for `text` to the serial console only (bypassing
/// the framebuffer pane, whose ANSI parser doesn't handle OSC). Terminals with
/// OSC-52 support copy it to the host clipboard. No-op in the test build and
/// for an empty selection.
fn emit_osc52(text: &str) {
    #[cfg(not(test))]
    {
        if text.is_empty() {
            return;
        }
        for b in osc52_sequence(text).bytes() {
            crate::serial::put_byte(b);
        }
    }
    #[cfg(test)]
    let _ = text;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test_case]
    fn osc52_framing_and_base64() {
        // "hi" -> base64 "aGk=" wrapped in the OSC-52 set-clipboard escape.
        assert_eq!(osc52_sequence("hi"), "\x1b]52;c;aGk=\x07");
        // Round-trips through the shared base64 encoder for arbitrary text.
        let seq = osc52_sequence("ChittiOS");
        assert!(seq.starts_with("\x1b]52;c;") && seq.ends_with('\x07'));
    }

    #[test_case]
    fn set_from_host_does_not_emit() {
        // set_from_host stores without an OSC-52 echo; get() round-trips it.
        set_from_host("pasted".to_string());
        assert_eq!(get(), Some(("pasted".to_string(), false)));
    }
}
