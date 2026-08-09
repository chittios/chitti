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
//! a host terminal.)
//!
//! **Graphical-window bridge.** The serial route above needs a terminal. For
//! the VM *window*, [`vdagent`] speaks the SPICE agent protocol over a
//! virtio-serial port ([`crate::drivers::virtio_serial`]) — what
//! `-chardev qemu-vdagent,clipboard=on` connects to. Both routes feed the same
//! clipboard, and both are best-effort: whichever is present is used.
//!
//! **Whether that reaches the host's system clipboard depends on the host UI,
//! and on macOS today it does not.** QEMU bridges its internal clipboard to a
//! real one only through a display backend that registers a clipboard peer —
//! `gtk` and `dbus` do; **`cocoa` does not**. So on a macOS host with the
//! default `-display cocoa` the guest and QEMU agree on a clipboard that
//! nothing copies to the Mac pasteboard. `/clip` says so rather than implying
//! it worked. On a Linux/GTK host it works window-to-window; on macOS, use the
//! OSC-52 serial route above, or VirtualBox, whose host process does reach the
//! pasteboard.

pub mod vdagent;

use crate::mm::Locked;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

struct Clip {
    text: String,
    linewise: bool,
}

static CLIP: Locked<Option<Clip>> = Locked::new(None);

/// Replace the clipboard contents **and** push them to the host clipboard
/// (OSC 52). Used by every in-OS copy (editor yank, chat drag-select).
pub fn set(text: String, linewise: bool) {
    emit_osc52(&text);
    // Tell the host we now own the clipboard. `grab` only announces — the host
    // asks for the bytes when something actually pastes, which is what
    // CAP_CLIPBOARD_BY_DEMAND means.
    if agent_ready() {
        let _ = crate::drivers::virtio_serial::write(&vdagent::grab(&[vdagent::FMT_UTF8_TEXT]));
    }
    // Same announcement to VirtualBox, whose host process owns the clipboard
    // itself — the route that reaches a Mac pasteboard from a VM window.
    if crate::drivers::vbox::clipboard::connected() {
        let _ = crate::drivers::vbox::clipboard::report_formats(
            crate::drivers::vbox::clipboard::FMT_UNICODETEXT,
        );
    }
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


// =====================================================================
// SPICE agent loop
// =====================================================================

/// Reassembly state plus what we know about the host end.
struct Agent {
    rx: vdagent::Reassembler,
    /// Capabilities the host announced. Zero until it answers.
    host_caps: u32,
    /// Whether we have announced ours (once per open port).
    announced: bool,
}

static AGENT: Locked<Option<Agent>> = Locked::new(None);

/// Whether the SPICE agent port is open and the host has answered.
pub fn agent_ready() -> bool {
    crate::drivers::virtio_serial::spice_ready()
        && AGENT.with(|a| a.as_ref().map(|a| a.announced).unwrap_or(false))
}

/// Whether a host clipboard channel exists at all (the port is open, whatever
/// the host UI then does with it).
pub fn agent_present() -> bool {
    crate::drivers::virtio_serial::spice_ready()
}

/// Bring up the virtio-serial clipboard channel. Quiet when absent, which is
/// the common case.
pub fn agent_init() {
    if !crate::drivers::virtio_serial::init() {
        return;
    }
    AGENT.with(|a| *a = Some(Agent { rx: vdagent::Reassembler::new(), host_caps: 0, announced: false }));
    tick();
}

/// Pump the agent: drain the port, act on whatever arrived. Called from the UI
/// pump, so it must never block.
pub fn tick() {
    if !crate::drivers::virtio_serial::present() {
        return;
    }
    // Announce as soon as the port opens, asking the host to answer. Until
    // both sides have announced, neither will send clipboard traffic.
    let need_announce = crate::drivers::virtio_serial::spice_ready()
        && AGENT.with(|a| a.as_ref().map(|a| !a.announced).unwrap_or(false));
    if need_announce
        && crate::drivers::virtio_serial::write(&vdagent::announce(true))
    {
        AGENT.with(|a| {
            if let Some(a) = a.as_mut() {
                a.announced = true;
            }
        });
        crate::ktrace::log("clipboard", "SPICE agent: announced capabilities to the host");
    }

    let bytes = crate::drivers::virtio_serial::pump();
    if bytes.is_empty() {
        return;
    }
    // Decode inside the lock, act outside it: handling a message calls back
    // into `set_from_host` and the transport, and `Locked` is not reentrant.
    let msgs: Vec<vdagent::Incoming> = AGENT
        .with(|a| a.as_mut().map(|a| a.rx.feed(&bytes)))
        .unwrap_or_default();
    for m in msgs {
        handle(m);
    }
}

fn handle(m: vdagent::Incoming) {
    match m {
        vdagent::Incoming::Caps { caps, request } => {
            AGENT.with(|a| {
                if let Some(a) = a.as_mut() {
                    a.host_caps = caps;
                }
            });
            // A `request` flag means answer with ours — and must NOT itself set
            // the flag, or the two ends announce at each other forever.
            if request {
                let _ = crate::drivers::virtio_serial::write(&vdagent::announce(false));
            }
        }
        vdagent::Incoming::Grab(formats) => {
            // The host copied something. Ask for it now rather than at paste
            // time: there is one clipboard here and no owner to ask later.
            if formats.contains(&vdagent::FMT_UTF8_TEXT) {
                let _ = crate::drivers::virtio_serial::write(&vdagent::request(vdagent::FMT_UTF8_TEXT));
            }
        }
        vdagent::Incoming::Clipboard { format, data } => {
            if format == vdagent::FMT_UTF8_TEXT {
                // Lossy rather than refused: a host clipboard may hold bytes we
                // cannot represent, and dropping the whole paste is worse than
                // one replacement character.
                let text = alloc::string::String::from_utf8_lossy(&data).into_owned();
                // `set_from_host`, never `set` — `set` would grab the clipboard
                // straight back and bounce it to the host in a loop.
                set_from_host(text);
            }
        }
        vdagent::Incoming::Request(format) => {
            // The host is pasting and wants our contents.
            let text = get().map(|(t, _)| t).unwrap_or_default();
            let reply = if format == vdagent::FMT_UTF8_TEXT && !text.is_empty() {
                vdagent::clipboard(vdagent::FMT_UTF8_TEXT, text.as_bytes())
            } else {
                // Answering a request we cannot satisfy with silence leaves the
                // host's paste hanging; a release says "nothing here".
                vdagent::release()
            };
            let _ = crate::drivers::virtio_serial::write(&reply);
        }
        vdagent::Incoming::Release | vdagent::Incoming::Other(_) => {}
    }
}

/// One line describing the host clipboard channel, for `/clip`.
pub fn agent_status() -> String {
    // VirtualBox first: where it is connected it is the route that actually
    // reaches the host's own clipboard, including on macOS.
    if let Some(s) = crate::drivers::vbox::clipboard::status() {
        return s;
    }
    if !crate::drivers::virtio_serial::present() {
        return "host channel: OSC-52 over serial only (no virtio-serial device)".to_string();
    }
    if !crate::drivers::virtio_serial::spice_ready() {
        return "host channel: virtio-serial present, no SPICE agent port open".to_string();
    }
    let caps = AGENT.with(|a| a.as_ref().map(|a| a.host_caps).unwrap_or(0));
    if caps == 0 {
        return "host channel: SPICE agent port open, host has not answered yet".to_string();
    }
    // Be honest about the macOS case rather than reporting a working link the
    // user cannot observe: QEMU only bridges its clipboard to a real one via a
    // display backend that registers a clipboard peer, and cocoa does not.
    "host channel: SPICE agent connected (on a macOS/cocoa host QEMU has no \
clipboard peer, so this does not reach the Mac pasteboard — use OSC-52 over \
serial, or VirtualBox)"
        .to_string()
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
