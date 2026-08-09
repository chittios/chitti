//! system
//!
//! The **power / peripherals / display / theme** command surface carved out
//! of the former 16k-line `shell/mod.rs` monolith: `/power`, `/suspend`,
//! `/battery`, `/datetime`, `/ntp`, `/bluetooth`, `/camera`, `/touch`,
//! `/screenshot`, `/record`, `/ui`, `/statusbar`, `/theme` and the ktrace/close
//! actions. Moved verbatim; `use super::*` keeps the parent's statics visible,
//! and the parent re-imports this module's items with `pub(crate) use system::*`.

use super::*;

#[cfg(test)]
pub(super) fn update_composer_hint(_remote_on: bool, _remote_cfg: Option<&remote::RemoteConfig>) {}

/// `/suspend` — suspend the machine, or report why it cannot.
///
/// `/suspend plan` is read-only and enumerates every precondition, which is the useful
/// form on a machine that will not suspend: a laptop with no `\_S3` package, ACPI mode
/// still owned by firmware, or no FACS to publish a resume address in each fail for a
/// different reason and need a different fix.
///
/// The transition itself confirms first. A suspend that does not resume loses
/// everything unsaved and can only be escaped by holding the power button, so this is
/// exactly the class of action the permission modal exists for.
/// `/power [status|mode …]` — idle counters + energy policy.
pub(super) fn run_power(arg: &str) {
    let a = arg.trim();
    if a.is_empty() || a == "status" || a == "info" {
        run_power_status();
        return;
    }
    let mut parts = a.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    match cmd {
        "mode" => {
            let Some(name) = parts.next() else {
                serial_println!(
                    "power> mode is {} — usage: /power mode performance|powersave|auto",
                    crate::power::cpu::mode().as_str()
                );
                return;
            };
            let Some(m) = crate::power::cpu::Mode::parse(name) else {
                serial_println!("power> unknown mode '{name}' (performance|powersave|auto)");
                return;
            };
            match crate::power::cpu::set_mode(m) {
                Ok(()) => {
                    serial_println!(
                        "power> mode {} (effective {})",
                        m.as_str(),
                        crate::power::cpu::last_effective()
                            .map(|e| e.as_str())
                            .unwrap_or("?")
                    );
                }
                Err(why) => {
                    // Policy is still recorded; hardware may not support it.
                    serial_println!(
                        "power> mode {} recorded — {why}",
                        m.as_str()
                    );
                }
            }
        }
        _ => serial_println!("power> usage: /power [status|mode <performance|powersave|auto>]"),
    }
}

/// `/bluetooth` — HCI transport, scan, PIN pair, HID host.
pub(super) fn run_bluetooth(arg: &str) {
    let a = arg.trim();
    if a.is_empty() || a == "status" || a == "info" {
        serial_println!("bluetooth> host + HCI USB:");
        for line in crate::drivers::bluetooth::status_lines() {
            serial_println!("  {line}");
        }
        serial_println!(
            "  cmds: status|reset|scan [n]|pair <AA:BB:…>|hid|bonds|disconnect"
        );
        return;
    }
    let mut parts = a.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    match cmd {
        "reset" | "up" => match crate::drivers::bluetooth::host::reset_and_info() {
            Ok(s) => serial_println!("bluetooth> {s}"),
            Err(e) => serial_println!("bluetooth> {e}"),
        },
        "scan" => {
            let slots: u8 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            serial_println!("bluetooth> inquiry ({slots}×1.28s)…");
            match crate::drivers::bluetooth::host::scan(slots) {
                Ok(list) if list.is_empty() => {
                    serial_println!("bluetooth> no devices (dongle powered? transport up?)")
                }
                Ok(list) => {
                    for (i, e) in list.iter().enumerate() {
                        let major = crate::drivers::bluetooth::hci::cod_major(e.class_of_device);
                        serial_println!(
                            "  [{i}] {}  CoD={:#x} major={major}",
                            crate::drivers::bluetooth::hci::format_bd_addr(&e.bd_addr),
                            e.class_of_device
                        );
                    }
                }
                Err(e) => serial_println!("bluetooth> scan: {e}"),
            }
        }
        "pair" => {
            let Some(addr) = parts.next() else {
                serial_println!("bluetooth> usage: /bluetooth pair <AA:BB:CC:DD:EE:FF>");
                return;
            };
            let pin = if let Some(p) = parts.next() {
                p.to_string()
            } else {
                let p = crate::modal::input(
                    "Bluetooth PIN",
                    &alloc::format!("PIN for {addr} (default 0000 if empty)"),
                    true,
                );
                if p.is_empty() {
                    "0000".into()
                } else {
                    p
                }
            };
            match crate::drivers::bluetooth::host::pair(addr, Some(&pin)) {
                Ok(s) => serial_println!("bluetooth> {s}"),
                Err(e) => serial_println!("bluetooth> pair: {e}"),
            }
        }
        "hid" => match crate::drivers::bluetooth::host::open_hid() {
            Ok(s) => serial_println!("bluetooth> {s}"),
            Err(e) => serial_println!("bluetooth> hid: {e}"),
        },
        "bonds" => {
            let bonds = crate::drivers::bluetooth::bond::load();
            if bonds.is_empty() {
                serial_println!("bluetooth> no bonds stored");
            } else {
                for b in bonds {
                    serial_println!("  {}  {}", b.addr, b.name);
                }
            }
        }
        "disconnect" | "down" => match crate::drivers::bluetooth::host::disconnect() {
            Ok(()) => serial_println!("bluetooth> disconnected"),
            Err(e) => serial_println!("bluetooth> {e}"),
        },
        _ => serial_println!(
            "bluetooth> usage: /bluetooth [status|reset|scan|pair <addr>|hid|bonds|disconnect]"
        ),
    }
}

/// `/camera [status|grab]` — UVC still capture to `/downloads/`.
pub(super) fn run_camera(arg: &str) {
    let a = arg.trim();
    if a.is_empty() || a == "status" || a == "info" {
        serial_println!("camera> UVC:");
        for line in crate::drivers::uvc::status_lines() {
            serial_println!("  {line}");
        }
        serial_println!("  cmds: status | grab [path]");
        return;
    }
    let mut parts = a.split_whitespace();
    match parts.next().unwrap_or("") {
        "grab" | "capture" | "snap" => {
            let dest = parts.next().map(|p| {
                if p.starts_with('/') {
                    p.to_string()
                } else {
                    alloc::format!("/downloads/{p}")
                }
            });
            camera_grab(dest.as_deref());
        }
        _ => serial_println!("camera> usage: /camera [status|grab [path]]"),
    }
}

pub(super) fn camera_grab(dest: Option<&str>) {
    if !crate::arch::uvc_ready() {
        serial_println!("camera> no stream transport — plug a UVC webcam and reboot/re-enum");
        return;
    }
    serial_println!("camera> grabbing still (up to 8 s)…");
    let Some((plan, frame)) = crate::arch::uvc_grab(8_000) else {
        serial_println!("camera> grab failed (timeout or empty frame)");
        return;
    };
    let ext = match plan.format {
        crate::drivers::uvc::PixelFormat::Mjpeg => "jpg",
        crate::drivers::uvc::PixelFormat::Yuy2 => "yuy2",
        crate::drivers::uvc::PixelFormat::UncompressedOther => "bin",
    };
    let path = dest
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let t = crate::arch::now_ms();
            alloc::format!("/downloads/camera-{t}.{ext}")
        });
    // MJPEG should start with JPEG SOI; still save whatever we assembled.
    if matches!(plan.format, crate::drivers::uvc::PixelFormat::Mjpeg)
        && (frame.len() < 2 || frame[0] != 0xff || frame[1] != 0xd8)
    {
        serial_println!(
            "camera> warning: frame is not a JPEG SOI ({} bytes) — saving anyway",
            frame.len()
        );
    }
    crate::synapse::fs::write(&path, &frame);
    crate::drivers::uvc::mark_grab_ok();
    serial_println!(
        "camera> saved {} ({} bytes, {} {}x{}) — /open {}",
        path,
        frame.len(),
        plan.format.name(),
        plan.width,
        plan.height,
        path
    );
    // Best-effort: if JPEG, try decoding to prove the frame is real.
    if matches!(plan.format, crate::drivers::uvc::PixelFormat::Mjpeg) {
        match crate::image::decode(&frame) {
            Ok(img) => serial_println!("camera> jpeg decode ok {}x{}", img.w, img.h),
            Err(e) => serial_println!("camera> jpeg decode: {e}"),
        }
    }
}

/// The framebuffer read, behind a `cfg` twin because `framebuffer` is
/// `#[cfg(not(test))]` — the same shape `refresh_todos` and `modal::confirm`
/// use. The test build has no panel, so it reports exactly what a serial-only
/// boot reports.
/// Shared by `/screenshot` and `/record` (sibling modules see only `pub(super)`).
#[cfg(not(test))]
pub(super) fn capture_screen(
    extent: crate::screenshot::Extent,
    cursor: bool,
) -> Result<(u32, u32, alloc::vec::Vec<u32>), &'static str> {
    crate::framebuffer::capture(extent, cursor)
}

#[cfg(test)]
pub(super) fn capture_screen(
    _extent: crate::screenshot::Extent,
    _cursor: bool,
) -> Result<(u32, u32, alloc::vec::Vec<u32>), &'static str> {
    Err("no framebuffer (serial-only boot)")
}

/// `/screenshot [extent] [after <d>] [--cursor] [dest]` — capture the screen to
/// a PNG in the store.
///
/// The framebuffer read is `framebuffer::capture` (ring 0: it is device memory,
/// so per the standing rule the driver reads and an agent asks) and every
/// geometry/naming decision is in the pure, unit-tested [`crate::screenshot`].
/// This function is the glue: parse, wait, capture, encode, write, verify.
pub(super) fn run_screenshot(arg: &str) {
    let a = arg.trim();
    if a == "help" || a == "-h" || a == "--help" {
        serial_println!("screenshot> usage: /screenshot [desktop|panel|chat|pane <n>|region <x>,<y>,<w>,<h>]");
        serial_println!("            [after <n>[s|ms]] [--cursor] [<dest>]");
        serial_println!("            default: the logical desktop, to /downloads/screenshot-<ms>.png");
        return;
    }
    let mut req = match crate::screenshot::parse(a) {
        Ok(r) => r,
        Err(e) => {
            serial_println!("screenshot> {e}");
            serial_println!("screenshot> try /screenshot help");
            return;
        }
    };

    // A screenshot reads whatever is on the panel, which crosses every agent
    // boundary the scope gate defends: another agent's surface, the human's
    // chat, a password modal. So a non-root agent may capture **only its own
    // surface**.
    //
    // The gate is `in_tool_call()` — "a model chose this call" — and *not*
    // `active_agent_id()` alone. Those are different questions, and conflating
    // them was a real bug: `active_agent_id` reports which agent the **chat is
    // homed to**, so after `/agents switch git` a human typing `/screenshot` at
    // the console was refused as though they were the git agent. A human at the
    // keyboard is the trust root whatever the chat is currently talking to.
    //
    // Enforced by narrowing the extent rather than refusing, because "your own
    // window" is what an app agent actually wants and a flat refusal would just
    // be worked around with a `ui_draw` read-back.
    let agent = active_agent_id();
    if crate::shell::in_tool_call() && agent != crate::agent::manifest::ORCHESTRATOR_ID.0 {
        let task = crate::sched::current_task_id();
        match crate::synapse::ui::surface_of_owner(task) {
            Some(id) => {
                if req.extent != crate::screenshot::Extent::Desktop {
                    serial_println!(
                        "screenshot> confined to your own surface (agent {agent}); \
                         ignoring the requested extent"
                    );
                }
                req.extent = crate::screenshot::Extent::Surface(id);
            }
            None => {
                serial_println!(
                    "screenshot> refused: agent {agent} has no surface of its own, and \
                     only the shell agent may capture the whole screen"
                );
                return;
            }
        }
    }

    // A capture taken the instant the command is submitted shows the command
    // still in the composer. `after` is how you photograph a menu or a hover
    // state; the wait pumps the UI and answers Ctrl+C, per the standing rules.
    if req.delay_ms > 0 {
        serial_println!("screenshot> capturing in {} ms…", req.delay_ms);
        let until = crate::arch::now_ms() + req.delay_ms;
        while crate::arch::now_ms() < until {
            if crate::shell::poll_interrupt() {
                serial_println!("screenshot> cancelled");
                return;
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
    }

    let (w, h, pixels) = match capture_screen(req.extent, req.cursor) {
        Ok(v) => v,
        Err(e) => {
            serial_println!("screenshot> {e}");
            return;
        }
    };

    let bytes = match crate::image::png::encode_rgb32(w as usize, h as usize, &pixels) {
        Ok(b) => b,
        Err(e) => {
            serial_println!("screenshot> encode failed: {e}");
            return;
        }
    };
    // The pixel buffer is up to tens of megabytes; drop it before the write so
    // the encoded PNG and the raw frame are not both resident (perf trap #3 —
    // this allocator punishes holding two large buffers at once).
    drop(pixels);

    let dest = req
        .dest
        .as_deref()
        .map(|d| crate::shell::resolve_path(&crate::screenshot::normalize_dest(d)))
        .unwrap_or_else(|| crate::screenshot::default_path(crate::arch::now_ms()));

    if crate::synapse::fs::exists(&dest)
        && !crate::modal::confirm("Overwrite file?", &alloc::format!("{dest} already exists."))
    {
        serial_println!("screenshot> cancelled (not overwritten)");
        return;
    }

    crate::synapse::fs::write(&dest, &bytes);
    serial_println!(
        "screenshot> {}",
        crate::screenshot::saved_line(&dest, w, h, bytes.len())
    );

    // Round-trip the file we just wrote, exactly as `/camera grab` does: an
    // encoder bug that produces a plausible-looking file is otherwise only
    // discovered later, by a human, on a different machine.
    match crate::image::png::decode(&bytes) {
        Ok(img) if img.w as u32 == w && img.h as u32 == h => {}
        Ok(img) => serial_println!(
            "screenshot> warning: re-decoded as {}x{}, expected {w}x{h}",
            img.w,
            img.h
        ),
        Err(e) => serial_println!("screenshot> warning: re-decode failed: {e}"),
    }
}

/// `/touch [status]` — digitizer path shares HID pointer decode (Tip Switch → click).
pub(super) fn run_touch(arg: &str) {
    let a = arg.trim();
    if !(a.is_empty() || a == "status" || a == "info") {
        serial_println!("touch> usage: /touch [status]");
        return;
    }
    serial_println!("touch> HID digitizer:");
    serial_println!(
        "  path: same as USB/I2C pointer — Tip Switch (0x0D/0x42) → left click,"
    );
    serial_println!("         absolute X/Y scaled to the framebuffer via mouse::set_abs");
    #[cfg(target_arch = "x86_64")]
    let has_usb = crate::arch::x86_64::xhci::has_mouse();
    #[cfg(target_arch = "aarch64")]
    let has_usb = crate::arch::aarch64::xhci::has_mouse();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let has_usb = false;
    serial_println!(
        "  usb pointer: {}",
        if has_usb {
            "enumerated (touch panels use this path when their report layout parses)"
        } else {
            "none enumerated yet"
        }
    );
    serial_println!("  note: multi-touch / gestures not implemented; first contact only");
}

pub(super) fn run_power_status() {
    // Keep auto mode tracking idle/battery without requiring a human to re-type.
    crate::power::cpu::tick();
    let halts = crate::power::idle::halt_count();
    let idle_ms = crate::power::idle::idle_ms();
    let up = crate::arch::now_ms();
    let pct = if up > 0 {
        (idle_ms.saturating_mul(100) / up).min(100)
    } else {
        0
    };
    serial_println!("power> CPU idle (hlt/wfi):");
    serial_println!("  halts     {halts}");
    serial_println!(
        "  skipped   {} (no timer IRQ — cooperative poll, not WFI)",
        crate::power::idle::skipped_count()
    );
    serial_println!(
        "  halt ok   {}",
        if crate::arch::idle_halt_ok() {
            "yes (timer IRQ live)"
        } else {
            "no (cooperative — WFI would freeze)"
        }
    );
    serial_println!("  idle time ~{idle_ms} ms ({pct}% of uptime {up} ms)");
    serial_println!("power> energy policy:");
    for line in crate::power::cpu::status_lines() {
        serial_println!("  {line}");
    }
    serial_println!("  set       /power mode performance|powersave|auto");
    serial_println!("  suspend   /suspend plan");
}

pub(super) fn run_suspend(arg: &str) {
    let plan = crate::power::plan();
    let a = arg.trim();
    // `--yes` skips the modal, the same escape hatch `/agents install` has, so the e2e
    // harness can drive a real suspend over serial. Human-typed input still confirms.
    let assume_yes = a.split_whitespace().any(|w| w == "--yes" || w == "-y");
    let a = a
        .split_whitespace()
        .filter(|w| *w != "--yes" && *w != "-y")
        .next()
        .unwrap_or("");
    let planning = matches!(a, "plan" | "status" | "info" | "");
    for line in &plan.report {
        serial_println!("suspend> {line}");
    }
    match plan.kind {
        Some(k) => serial_println!(
            "suspend> mechanism: {} -- {}",
            k.label(),
            if plan.ready { "ready" } else { "NOT ready" }
        ),
        None => serial_println!("suspend> this machine cannot suspend"),
    }
    if planning {
        // Say what comes back and what does not, because the one thing a human
        // needs to decide here is whether losing it is acceptable — and finding
        // out after the machine has resumed is too late.
        serial_println!("suspend> on resume: xHCI, the NIC and the sound device are re-probed;");
        serial_println!("suspend>            disks re-probe on first access; the IP address is NOT");
        serial_println!("suspend>            re-asserted -- run `/network dhcp` afterwards");
        serial_println!("suspend> `/suspend now` to actually suspend (`--yes` skips the prompt)");
        return;
    }
    if !plan.ready {
        serial_println!("suspend> refusing: preconditions above are not met");
        return;
    }
    // `/suspend plan` stays agent-readable, but actually suspending does not.
    // With a login password set the machine resumes into a lock screen, and an
    // agent-initiated suspend would put that prompt inside
    // `serial::capture_begin()`'s buffer rather than on the console — invisible to
    // the human, who sees a machine that went to sleep and came back hung.
    if crate::shell::in_tool_call() {
        serial_println!("suspend> refused: only a human at the console may suspend the machine");
        return;
    }
    #[cfg(not(test))]
    if !assume_yes
        && !crate::modal::confirm(
        "Suspend the machine?",
        "The machine will sleep. If it does not resume, unsaved state is lost and \
         the only way back is holding the power button.",
        )
    {
        serial_println!("suspend> cancelled");
        return;
    }
    match crate::power::suspend() {
        Ok(()) => serial_println!("suspend> resumed"),
        Err(e) => serial_println!("suspend> not attempted: {e}"),
    }
}

/// `/battery` — the ACPI battery, and which step failed if there is no reading.
///
/// A diagnostic more than a display (the percentage lives in the status bar via
/// `${battery}`): none of the battery path can be verified in a VM, so on a real
/// laptop this is what says whether the namespace, the embedded controller or the
/// `_BST` evaluation is the thing to fix. Read-only.
pub(super) fn run_battery() {
    for line in crate::drivers::battery::diagnose() {
        serial_println!("battery> {line}");
    }
}

/// `/datetime` — show or set the wall clock and timezone.
pub(super) fn run_datetime(arg: &str) {
    use crate::clock;
    if arg.is_empty() {
        serial_println!(
            "datetime> {}  {}  (source={})",
            clock::format_datetime(),
            clock::format_tz(),
            clock::source().as_str()
        );
        serial_println!("  set time: /datetime 2026-07-04 13:45[:00]");
        serial_println!("  set zone: /datetime tz America/New_York | +5:30 | list");
        return;
    }
    if let Some(tz) = arg.strip_prefix("tz") {
        let tz = tz.trim();
        if tz == "list" || tz == "ls" {
            serial_println!("datetime> zones:");
            for n in clock::tz::list_names() {
                if let Some(d) = clock::tz::describe(n, clock::now_unix()) {
                    serial_println!("  {d}");
                }
            }
            return;
        }
        // IANA name first (contains '/'), then fixed offset.
        if tz.contains('/') || clock::tz::lookup(tz).is_some() {
            if clock::set_tz_name(tz) {
                crate::ui_config::persist_tz_name(tz, clock::fixed_tz_offset());
                update_status();
                serial_println!(
                    "datetime> timezone {}  (now {})",
                    clock::format_tz(),
                    clock::format_datetime()
                );
            } else {
                serial_println!("datetime> unknown zone '{tz}' — try /datetime tz list");
            }
            return;
        }
        match parse_tz(tz) {
            Some(secs) => {
                // Keep the displayed wall time fixed when relabeling the zone.
                clock::set_tz_keep_local(secs);
                crate::ui_config::persist_tz(secs);
                update_status();
                serial_println!(
                    "datetime> timezone {}  (now {})",
                    clock::format_tz(),
                    clock::format_datetime()
                );
            }
            None => serial_println!("usage: /datetime tz America/New_York | +5:30 | list"),
        }
        return;
    }
    match parse_datetime(arg) {
        Some((y, mo, d, h, mi, s)) => {
            clock::set_local(y, mo, d, h, mi, s);
            update_status();
            serial_println!(
                "datetime> set to {}  {}",
                clock::format_datetime(),
                clock::format_tz()
            );
        }
        None => serial_println!("usage: /datetime YYYY-MM-DD HH:MM[:SS]"),
    }
}

/// `/ntp [host|ip]` — SNTP sync (default `pool.ntp.org`). Human-only.
pub(super) fn run_ntp(arg: &str) {
    let host = arg.trim();
    if !crate::net::is_up() {
        serial_println!("ntp> no network (try /network dhcp)");
        return;
    }
    serial_println!(
        "ntp> querying {} …",
        if host.is_empty() { "pool.ntp.org" } else { host }
    );
    match crate::net::ntp_sync_host(host, 8_000) {
        Ok(unix) => {
            update_status();
            serial_println!(
                "ntp> ok — unix {unix}  {}  {}  (source=ntp)",
                crate::clock::format_datetime(),
                crate::clock::format_tz()
            );
        }
        Err(e) => serial_println!("ntp> {e}"),
    }
}

/// `/ui` — show or manage the UI config (`/configs/core/ui.json`).
pub(super) fn run_ui(arg: &str) {
    #[cfg(feature = "server")]
    {
        let _ = arg;
        serial_println!("ui> unavailable in the server build (no GUI)");
        return;
    }
    #[cfg(not(feature = "server"))]
    run_ui_inner(arg);
}

#[cfg(not(feature = "server"))]
pub(super) fn run_ui_inner(arg: &str) {
    use crate::ui_config;
    match arg {
        "" | "config" | "show" => {
            serial_println!("ui> {} (edit with /open {}, then /ui reload)", ui_config::ui_path(), ui_config::ui_path());
            for line in ui_config::ui_json_text().lines() {
                serial_println!("{}", line);
            }
        }
        "reload" => {
            ui_config::reload_and_apply();
            update_status();
            serial_println!("ui> reloaded {} and re-applied the layout", ui_config::ui_path());
        }
        "reset" => {
            ui_config::reset();
            update_status();
            serial_println!("ui> reset to defaults and re-applied");
        }
        // `status_pos` is the one ui.json field with a dedicated command, since
        // editing JSON to move a bar is a poor trade; keep them discoverable
        // from each other.
        "statusbar" | "bar" => run_statusbar(""),
        _ => {
            serial_println!("usage: /ui [config|reload|reset]   (edit {} via /open)", ui_config::ui_path());
            serial_println!("  /statusbar <top|bottom|left|right>   move the OS status bar");
        }
    }
}

/// `/statusbar` — which desktop edge the OS status bar sits on.
///
/// Applies instantly and persists to `ui.json` (`status_pos`), so it survives a
/// reboot and is also editable by hand via `/open`. Reversible and purely
/// cosmetic, which is why the settings agent may set it directly.
pub(super) fn run_statusbar(arg: &str) {
    #[cfg(any(feature = "server", test))]
    {
        let _ = arg;
        serial_println!("statusbar> unavailable in the server build (no GUI)");
    }
    #[cfg(all(not(feature = "server"), not(test)))]
    run_statusbar_inner(arg);
}

#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn run_statusbar_inner(arg: &str) {
    use crate::panes_layout::StatusPos;
    let arg = arg.trim();
    if arg.is_empty() || arg == "status" {
        let cur = crate::framebuffer::status_pos().unwrap_or_default();
        serial_println!("statusbar> {} (top | bottom | left | right)", cur.as_str());
        if cur.vertical() {
            serial_println!(
                "statusbar>   side bar: {} cols, fields stack top→bottom (one token per row)",
                crate::panes_layout::STATUS_V_COLS
            );
        }
        serial_println!("statusbar>   default is top; icons paint ~1.5× body text");
        serial_println!("statusbar> usage: /statusbar <top|bottom|left|right>");
        return;
    }
    // A typo must not move the bar somewhere unasked-for, so parse strictly and
    // say what was accepted rather than silently defaulting.
    let Some(pos) = StatusPos::parse(arg) else {
        serial_println!("statusbar> unknown position '{}' (expected top | bottom | left | right)", arg);
        return;
    };
    let now = crate::framebuffer::set_status_pos(pos);
    let mut cfg = crate::ui_config::current();
    cfg.status_pos = pos.as_str().to_string();
    crate::ui_config::set_config(cfg);
    update_status();
    match now {
        Some(p) => serial_println!("statusbar> moved to the {} edge", p.as_str()),
        // No console (serial-only boot): the preference is still saved for next time.
        None => serial_println!("statusbar> recorded {} (no framebuffer to apply it to)", pos.as_str()),
    }
}

/// `/theme` — list / set / save / install UI themes (colours, syntax, cursor,
/// wallpaper, opacity). A theme is a preset that populates `ui.json`; see
/// [`crate::theme`].
pub(super) fn run_theme(arg: &str) {
    #[cfg(feature = "server")]
    {
        let _ = arg;
        serial_println!("theme> unavailable in the server build (no GUI)");
    }
    #[cfg(not(feature = "server"))]
    run_theme_inner(arg);
}

#[cfg(not(feature = "server"))]
pub(super) fn run_theme_inner(arg: &str) {
    let (sub, rest) = match arg.split_once(' ') {
        Some((a, b)) => (a, b.trim()),
        None => (arg, ""),
    };
    match sub {
        "" | "list" => {
            let cur = crate::ui_config::current().theme_name;
            serial_println!("themes (bundled + installed; * = current):");
            for n in crate::theme::list() {
                serial_println!("  {}{}", n, if n == cur { "  *" } else { "" });
            }
            serial_println!("/theme set <name> · current · save <name> · install <url>");
            serial_println!("/theme wallpaper <none|gradient:#a,#b|/path|https://url> · opacity <0-255>");
        }
        "current" => serial_println!("theme> current: {}", crate::ui_config::current().theme_name),
        "wallpaper" | "bg" | "wp" => set_wallpaper_cmd(rest),
        "opacity" => match rest.parse::<u64>() {
            Ok(n) => {
                let op = n.min(255);
                let mut cfg = crate::ui_config::current();
                cfg.opacity = op;
                crate::ui_config::set_config(cfg);
                serial_println!("theme> opacity {} (255 = opaque; lower = more see-through)", op);
            }
            Err(_) => serial_println!("usage: /theme opacity <0-255>"),
        },
        "set" => {
            if rest.is_empty() {
                serial_println!("usage: /theme set <name>");
                return;
            }
            match crate::theme::apply(rest) {
                Ok(()) => {
                    update_status();
                    serial_println!("theme> set: {}", rest);
                }
                Err(e) => serial_println!("theme> error: {}", e),
            }
        }
        "save" => {
            if rest.is_empty() {
                serial_println!("usage: /theme save <name>");
                return;
            }
            match crate::theme::save(rest) {
                Ok(p) => serial_println!("theme> saved current appearance -> {}", p),
                Err(e) => serial_println!("theme> error: {}", e),
            }
        }
        "install" => {
            if rest.is_empty() {
                serial_println!("usage: /theme install <url>");
                return;
            }
            match crate::theme::install(rest) {
                Ok(n) => serial_println!("theme> installed '{}' — /theme set {}", n, n),
                Err(e) => serial_println!("theme> error: {}", e),
            }
        }
        _ => serial_println!("usage: /theme [list | set <name> | current | save <name> | install <url> | wallpaper <spec> | opacity <n>]"),
    }
}

/// `/theme wallpaper <spec>` — set the desktop backdrop. `spec` is one of:
/// `none` (solid theme bg), `gradient:#aabbcc,#112233` (two-stop vertical
/// gradient), a store path to a PNG/JPEG (`/downloads/pic.png`), or an
/// `http(s)://` URL — which is downloaded into the store, sniffed, then mapped.
/// The image is decoded once and cover-scaled to the screen by the compositor.
#[cfg(not(feature = "server"))]
pub(super) fn set_wallpaper_cmd(rest: &str) {
    use alloc::string::{String, ToString};
    if rest.is_empty() {
        serial_println!("usage: /theme wallpaper <none | gradient:#aabbcc,#112233 | /path/img | https://url>");
        serial_println!("  tip: request a screen-sized image (e.g. Unsplash '?w=2560') — large photos decode slowly in-kernel");
        return;
    }
    let spec = if rest.eq_ignore_ascii_case("none") {
        String::new()
    } else if rest.starts_with("http://") || rest.starts_with("https://") {
        serial_println!("theme> downloading wallpaper (large images decode slowly)…");
        // Fixed store name — the decoder sniffs PNG/JPEG magic, so the
        // extension is irrelevant; overwrite so repeated sets don't pile up.
        let args = alloc::format!(
            r#"{{"url":"{}","path":"wallpaper","overwrite":"true"}}"#,
            rest
        );
        let r = run_download_tool(&args);
        match r.strip_prefix("ok:path=").and_then(|s| s.split(' ').next()) {
            Some(p) => {
                let looks_ok = crate::synapse::fs::read(p)
                    .map(|b| is_image_bytes(&b))
                    .unwrap_or(false);
                if !looks_ok {
                    serial_println!("theme> downloaded but it isn't a PNG/JPEG image ({}); not applied", p);
                    return;
                }
                serial_println!("theme> saved {}", p);
                p.to_string()
            }
            None => {
                serial_println!("theme> download failed: {}", r);
                return;
            }
        }
    } else {
        // A store path (image) or a `gradient:` / `""` spec — pass through.
        rest.to_string()
    };
    let mut cfg = crate::ui_config::current();
    cfg.wallpaper = spec;
    crate::ui_config::set_config(cfg);
    let now = crate::ui_config::current().wallpaper;
    if now.is_empty() {
        serial_println!("theme> wallpaper cleared (solid background)");
        return;
    }
    serial_println!("theme> wallpaper set: {}", now);
    // A translucent backdrop only reads if the image is bright enough. Probe an
    // image wallpaper's mean luma and nudge the user when it'll be near-black —
    // otherwise "I set a wallpaper and nothing changed" looks like a bug.
    let op = crate::ui_config::current().opacity;
    if !now.starts_with("gradient:") {
        if let Some(luma) = crate::synapse::fs::read(&now)
            .and_then(|b| crate::image::decode(&b).ok())
            .map(|img| crate::image::mean_luma(&img))
        {
            if luma < 40 {
                serial_println!(
                    "theme> note: this image is very dark (mean brightness {}/255) — blended at opacity {} \
                     it will look near-black. Try a brighter image, or a lower opacity to let more of it through.",
                    luma, op
                );
            }
        }
    }
}

/// Cheap magic-byte sniff so a 404/HTML error page isn't mapped as a backdrop.
#[cfg(not(feature = "server"))]
pub(super) fn is_image_bytes(b: &[u8]) -> bool {
    b.starts_with(&[0x89, b'P', b'N', b'G']) // PNG
        || b.starts_with(&[0xff, 0xd8, 0xff]) // JPEG
}

/// `/ktrace` — toggle the ktrace log stream in the action (right) pane.
pub(super) fn toggle_ktrace() {
    #[cfg(not(test))]
    {
        use crate::framebuffer::{self, RightMode};
        if framebuffer::has_tab(RightMode::Ktrace) {
            framebuffer::close_tab_mode(RightMode::Ktrace);
            repaint_active_tab();
            serial_println!("ktrace> tab closed");
        } else {
            framebuffer::open_ktrace();
            serial_println!("ktrace> showing as an action tab (Ctrl+Tab cycles focus, /close closes it)");
        }
    }
}

/// `/close` (also Ctrl+W) — close the **active** action tab; the pane collapses
/// once the last tab closes. Tears down that tab's process (stops audio,
/// drops the editor buffer, **kills package-UI agents**).
pub(super) fn close_action() {
    #[cfg(not(test))]
    {
        close_active_tab();
        serial_println!("(closed the active tab)");
    }
}
