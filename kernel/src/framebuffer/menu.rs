//! Status-bar dropdown menus — one body painter per chip.

use super::*;

/// macOS-style status-bar dropdown for `chip`. Anchored under/above the bar
/// near the chip's hit rect. Click outside, Esc, or Close dismisses.
pub fn draw_status_menu(chip: StatusChip) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
        clear_modal_close_rect();
        LIST_BROWSER_GEOM.with(|g| *g = None);
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        // Repaint the bar so chip rects / live values are current, then overlay.
        sc.draw_status();

        let cw = sc.cw();
        let ch = sc.ch();
        let anchor = STATUS_CHIP_RECTS.with(|a| a[chip as usize]);
        let bar = STATUS_BAR_RECT.with(|r| *r);
        let pos = sc.layout.status_pos;

        // Menu size: clock is larger (analog face); volume is a compact control.
        let (cols, rows) = match chip {
            StatusChip::Clock => (32u64, 16u64),
            StatusChip::Volume => (28, 12),
            StatusChip::Net => (36, 12),
            StatusChip::Mem | StatusChip::Cpu => (34, 10),
            StatusChip::Battery => (30, 9),
            StatusChip::Kbd | StatusChip::Mouse => (28, 8),
            StatusChip::Brand => (28, 8),
            // Wider and taller: it lists actual rows, not a couple of fields.
            StatusChip::Notifications => (44, 12),
        };
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = rows * ch + 2 * (BORDER + PAD);

        // Anchor: prefer under a top bar, above a bottom bar, else to the side.
        let mut bx = if anchor.2 > 0 {
            anchor.0.saturating_add(anchor.2 / 2).saturating_sub(bw / 2)
        } else {
            (sc.width - bw) / 2
        };
        if bx + bw > sc.width.saturating_sub(OUTER) {
            bx = sc.width.saturating_sub(bw + OUTER);
        }
        if bx < OUTER {
            bx = OUTER;
        }
        let by = match pos {
            crate::panes_layout::StatusPos::Top => bar.1 + bar.3 + 4,
            crate::panes_layout::StatusPos::Bottom => bar.1.saturating_sub(bh + 4),
            crate::panes_layout::StatusPos::Left => {
                bx = bar.0 + bar.2 + 4;
                if anchor.2 > 0 {
                    anchor.1
                } else {
                    (sc.height - bh) / 2
                }
            }
            crate::panes_layout::StatusPos::Right => {
                bx = bar.0.saturating_sub(bw + 4);
                if anchor.2 > 0 {
                    anchor.1
                } else {
                    (sc.height - bh) / 2
                }
            }
        };
        let by = by.min(sc.height.saturating_sub(bh + OUTER)).max(OUTER);

        let bg = sc.theme.status_bg;
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let mut y = by + BORDER + PAD;
        let content_w = cols * cw;

        // Title row + close.
        let title = match chip {
            StatusChip::Brand => "ChittiOS",
            StatusChip::Kbd => "Keyboard",
            StatusChip::Mouse => "Mouse",
            StatusChip::Net => "Network",
            StatusChip::Mem => "Memory",
            StatusChip::Cpu => "Processor",
            StatusChip::Battery => "Battery",
            StatusChip::Volume => "Sound",
            StatusChip::Clock => "Clock",
            StatusChip::Notifications => "Notifications",
        };
        let icon = match chip {
            StatusChip::Brand => crate::icons::fa::HOUSE,
            StatusChip::Kbd => crate::icons::fa::KEYBOARD,
            StatusChip::Mouse => crate::icons::fa::MOUSE,
            StatusChip::Net => crate::icons::fa::WIFI,
            StatusChip::Mem => crate::icons::fa::MEMORY,
            StatusChip::Cpu => crate::icons::fa::MICROCHIP,
            StatusChip::Battery => crate::icons::fa::BATTERY,
            StatusChip::Volume => {
                crate::icons::volume_icon(crate::sound::muted(), crate::sound::volume())
            }
            StatusChip::Clock => crate::icons::fa::CLOCK,
            StatusChip::Notifications => crate::icons::fa::BELL,
        };
        sc.draw_str(
            ix,
            y,
            &alloc::format!("{icon} {title}"),
            sc.theme.accent,
            bg,
        );
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        let cx = ix + content_w.saturating_sub(close_w);
        let (iw, _) = sc.glyph_cell(mark);
        // Hover chrome (same chip as the pane close button): fill an accent
        // square so the mark reads as a button while the pointer is on it.
        let hovered_close = POPUP_HOVER.with(|h| *h == Some(ModalHit::Close));
        let close_bg = if hovered_close { sc.mix(bg, sc.theme.accent, 0.18) } else { bg };
        if hovered_close {
            sc.fill_rect(cx, y, close_w, ch, close_bg);
        }
        let close_ink = if hovered_close {
            sc.lighten(sc.theme.accent, 0.35)
        } else {
            sc.theme.accent
        };
        sc.blit_glyph(
            cx + close_w.saturating_sub(iw) / 2,
            y,
            mark,
            close_ink,
            close_bg,
        );
        // Dedicated close rect (must not share slot 0 with Yes / menu body).
        set_modal_close_rect((cx, y, close_w, ch));
        y += ch + 4;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += ch / 2;

        match chip {
            StatusChip::Clock => {
                y = draw_clock_menu_body(sc, ix, y, content_w, ch, cw, bg);
            }
            StatusChip::Net => {
                y = draw_net_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Mem => {
                y = draw_mem_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Cpu => {
                y = draw_cpu_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Battery => {
                y = draw_battery_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Volume => {
                y = draw_volume_menu_body(sc, ix, y, content_w, ch, cw, bg);
            }
            StatusChip::Kbd => {
                y = draw_input_menu_body(sc, ix, y, ch, bg, true);
            }
            StatusChip::Mouse => {
                y = draw_input_menu_body(sc, ix, y, ch, bg, false);
            }
            StatusChip::Notifications => {
                y = draw_notify_menu_body(sc, ix, y, content_w, ch, cw, bg);
            }
            StatusChip::Brand => {
                sc.draw_str(ix, y, "Click for About…", sc.theme.chat_fg, bg);
                y += ch;
                sc.draw_str(ix, y, &alloc::format!("v{}", crate::VERSION), sc.theme.title_dim, bg);
            }
        }

        // Footer hint.
        let foot_y = by + bh - BORDER - PAD - ch;
        sc.draw_str(ix, foot_y, "Esc / click outside to close", sc.theme.title_dim, bg);
        // Full card is clickable for "inside" hit tests (slot 1 = menu body).
        MODAL_RECTS.with(|m| m[1] = (bx, by, bw, bh));
        sc.cursor_overlay();
    });
}

/// True if `(x,y)` is inside the open status menu panel (not the close mark).
pub fn status_menu_contains(x: u64, y: u64) -> bool {
    MODAL_RECTS.with(|m| in_rect(x, y, m[1]))
}

fn draw_clock_menu_body(
    sc: &Screen,
    ix: u64,
    mut y: u64,
    content_w: u64,
    ch: u64,
    cw: u64,
    bg: Rgb,
) -> u64 {
    let (yy, mo, d, h, mi, s, wd) =
        crate::clock::civil_from_unix(crate::clock::now_unix() + crate::clock::tz_offset() as i64);
    let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let date = alloc::format!(
        "{}, {} {} {}",
        weekdays[wd as usize],
        d,
        months[(mo as usize).saturating_sub(1).min(11)],
        yy
    );
    sc.draw_str(ix, y, &date, sc.theme.chat_fg, bg);
    y += ch + 4;

    // Analog face centred in the content width.
    let face_r = (ch * 3).max(28).min(content_w / 2 - 4);
    let fcx = ix + content_w / 2;
    let fcy = y + face_r + 4;
    draw_analog_clock(sc, fcx as i64, fcy as i64, face_r as i64, cw, ch, h, mi, s);
    y = fcy + face_r + ch;

    let time = alloc::format!("{:02}:{:02}:{:02}", h, mi, s);
    let tx = ix + content_w.saturating_sub(time.len() as u64 * cw) / 2;
    sc.draw_str(tx, y, &time, sc.theme.accent, bg);
    y += ch;
    let tz = crate::clock::format_tz();
    let tzx = ix + content_w.saturating_sub(tz.chars().count() as u64 * cw) / 2;
    sc.draw_str(tzx, y, &tz, sc.theme.title_dim, bg);
    y += ch + ch / 2;
    sc.draw_str(ix, y, "Timezone via /datetime tz …", sc.theme.title_dim, bg);
    y + ch
}

/// Volume dropdown body. Registers three clickable [`ModalHit::Choose`] rows:
/// `0` = mute toggle, `1` = −5%, `2` = +5%. Also paints a level bar.
fn draw_volume_menu_body(
    sc: &Screen,
    ix: u64,
    mut y: u64,
    content_w: u64,
    ch: u64,
    cw: u64,
    bg: Rgb,
) -> u64 {
    let muted = crate::sound::muted();
    let pct = crate::sound::volume();
    let icon = crate::icons::volume_icon(muted, pct);
    let label = if muted {
        alloc::format!("{icon}  Muted  ({pct}%)")
    } else {
        alloc::format!("{icon}  Output  {pct}%")
    };
    sc.draw_str(ix, y, &label, sc.theme.chat_fg, bg);
    y += ch + 4;

    // Level bar.
    let bar_h = (ch / 2).max(6);
    let bar_w = content_w;
    sc.fill_rect(ix, y, bar_w, bar_h, sc.theme.border_dim);
    let fill = (bar_w as u32 * pct / 100) as u64;
    if fill > 0 && !muted {
        sc.fill_rect(ix, y, fill, bar_h, sc.theme.accent);
    } else if fill > 0 && muted {
        sc.fill_rect(ix, y, fill, bar_h, sc.theme.title_dim);
    }
    y += bar_h + ch / 2;

    // Device line.
    let dev = if crate::sound::is_up() {
        "Device  PCM ready"
    } else {
        "Device  none (software gain still applies)"
    };
    sc.draw_str(ix, y, dev, sc.theme.title_dim, bg);
    y += ch + 4;

    // Clickable action rows → Choose(0..2).
    CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
    let actions: [(&str, bool); 3] = [
        (
            if muted { "Unmute" } else { "Mute" },
            true,
        ),
        ("Volume  −5%", true),
        ("Volume  +5%", true),
    ];
    CHOOSE_COUNT.store(actions.len(), core::sync::atomic::Ordering::Relaxed);
    for (i, (text, _)) in actions.iter().enumerate() {
        // Hover chrome: accent-tinted row highlight + brighter label, matching
        // the list browsers and tab bar (distinct from a pressed state).
        let hovered = POPUP_HOVER.with(|h| *h == Some(ModalHit::Choose(i)));
        let row_bg = if hovered {
            sc.mix(sc.theme.chat_bg, sc.theme.accent, 0.14)
        } else {
            sc.theme.chat_bg
        };
        sc.fill_rect(ix, y, content_w, ch, row_bg);
        let prefix = match i {
            0 => crate::icons::fa::VOLUME_XMARK,
            1 => crate::icons::fa::MINUS,
            _ => crate::icons::fa::PLUS,
        };
        sc.draw_str(
            ix,
            y,
            &alloc::format!("{prefix}  {text}"),
            if hovered {
                sc.lighten(sc.theme.chat_fg, 0.35)
            } else {
                sc.theme.chat_fg
            },
            row_bg,
        );
        CHOOSE_RECTS.with(|c| c[i] = (ix, y, content_w, ch));
        y += ch + 2;
    }
    y += ch / 2;
    sc.draw_str(
        ix,
        y,
        "Wheel / ←→  adjust · m mute",
        sc.theme.title_dim,
        bg,
    );
    // Silence unused cw (kept for API symmetry with other drawers).
    let _ = cw;
    y + ch
}

fn draw_net_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let up = crate::net::is_up();
    sc.draw_str(
        ix,
        y,
        if up { "Status   Connected" } else { "Status   Offline" },
        if up { sc.theme.accent } else { sc.theme.title_dim },
        bg,
    );
    y += ch;
    if let Some(info) = crate::net::info() {
        sc.draw_str(ix, y, &alloc::format!("Interface  {}", info.ifname), sc.theme.chat_fg, bg);
        y += ch;
        let mac = alloc::format!(
            "MAC  {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            info.mac[0], info.mac[1], info.mac[2], info.mac[3], info.mac[4], info.mac[5]
        );
        sc.draw_str(ix, y, &mac, sc.theme.title_dim, bg);
        y += ch;
        if let Some(ip) = info.ip {
            sc.draw_str(ix, y, &alloc::format!("IPv4  {}", ip), sc.theme.chat_fg, bg);
            y += ch;
        }
        if let Some(gw) = info.gateway {
            sc.draw_str(ix, y, &alloc::format!("Gateway  {}", gw), sc.theme.title_dim, bg);
            y += ch;
        }
        if !info.dns.is_empty() {
            sc.draw_str(ix, y, &alloc::format!("DNS  {}", info.dns[0]), sc.theme.title_dim, bg);
            y += ch;
        }
        sc.draw_str(
            ix,
            y,
            if info.dhcp { "Config  DHCP" } else { "Config  Static" },
            sc.theme.title_dim,
            bg,
        );
        y += ch;
    } else {
        sc.draw_str(ix, y, "No network device bound", sc.theme.title_dim, bg);
        y += ch;
    }
    y += ch / 2;
    sc.draw_str(ix, y, "Shell: /network  /wifi  /ping", sc.theme.title_dim, bg);
    y + ch
}

fn draw_mem_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let m = crate::mm::mem_stats();
    let mib = 1024 * 1024;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Heap used   {} MiB", m.heap_used / mib),
        sc.theme.chat_fg,
        bg,
    );
    y += ch;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Heap total  {} MiB", m.heap_total / mib),
        sc.theme.title_dim,
        bg,
    );
    y += ch;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("RAM total   {} MiB", m.ram_total / mib),
        sc.theme.title_dim,
        bg,
    );
    y += ch;
    let reserved = m.ram_reserved.saturating_sub(m.heap_total);
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Reserved    {} MiB", reserved / mib),
        sc.theme.title_dim,
        bg,
    );
    y + ch
}

fn draw_cpu_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let pct = crate::shell::cpu_percent();
    let cores = crate::arch::cpu_count();
    sc.draw_str(ix, y, &alloc::format!("Load     {pct}%"), sc.theme.chat_fg, bg);
    y += ch;
    sc.draw_str(ix, y, &alloc::format!("Cores    {cores}"), sc.theme.title_dim, bg);
    y += ch;
    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";
    sc.draw_str(ix, y, &alloc::format!("Arch     {arch}"), sc.theme.title_dim, bg);
    y += ch + ch / 2;
    sc.draw_str(ix, y, "Shell: /top  /perf", sc.theme.title_dim, bg);
    y + ch
}

fn draw_battery_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    if let Some(b) = crate::drivers::battery::cached() {
        sc.draw_str(
            ix,
            y,
            &alloc::format!("Charge   {}", crate::drivers::battery::format(&b)),
            sc.theme.chat_fg,
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "Source   ACPI _BST / EC", sc.theme.title_dim, bg);
        y += ch;
        sc.draw_str(ix, y, "Shell: /battery", sc.theme.title_dim, bg);
    } else {
        sc.draw_str(ix, y, "No battery reported", sc.theme.title_dim, bg);
        y += ch;
        sc.draw_str(ix, y, "(desktop / no ACPI pack)", sc.theme.title_dim, bg);
    }
    y + ch
}

/// The notification dropdown: the newest few entries at a glance.
///
/// A *glance*, deliberately — not a scrollable list. Anything longer than this
/// belongs in the action pane (`/notify`), and building a second scrollable list
/// widget in a popover would be a second thing to keep themed.
fn draw_notify_menu_body(
    sc: &Screen,
    ix: u64,
    mut y: u64,
    content_w: u64,
    ch: u64,
    cw: u64,
    bg: Rgb,
) -> u64 {
    let all = crate::notify::list();
    let cols = (content_w / cw).max(1) as usize;
    if all.is_empty() {
        sc.draw_str(ix, y, "Nothing to report", sc.theme.title_dim, bg);
        y += ch;
        sc.draw_str(ix, y, "Shell: /notify", sc.theme.title_dim, bg);
        return y + ch;
    }
    let unread = crate::notify::unread_count();
    sc.draw_str(
        ix,
        y,
        &alloc::format!("{} unread of {}", unread, all.len()),
        sc.theme.accent,
        bg,
    );
    y += ch;
    let now = crate::clock::now_unix();
    const SHOWN: usize = 6;
    for n in all.iter().take(SHOWN) {
        // `summary_line` is in `crate::notify` (which is testable) rather than
        // here, so the row's fitting and truncation are covered by a unit test.
        let row = crate::notify::summary_line(n, now, cols);
        let fg = if n.read { sc.theme.title_dim } else { sc.theme.chat_fg };
        sc.draw_str(ix, y, &row, fg, bg);
        y += ch;
    }
    if all.len() > SHOWN {
        sc.draw_str(
            ix,
            y,
            &alloc::format!("… {} more — /notify", all.len() - SHOWN),
            sc.theme.title_dim,
            bg,
        );
        y += ch;
    }
    y + ch / 2
}

fn draw_input_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb, kbd: bool) -> u64 {
    if kbd {
        let last = crate::console::input_activity_ms();
        let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
        sc.draw_str(
            ix,
            y,
            if active { "Keyboard  Active" } else { "Keyboard  Idle" },
            if active { sc.theme.accent } else { sc.theme.chat_fg },
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "USB HID / virtio / PS-2", sc.theme.title_dim, bg);
    } else {
        let last = crate::mouse::activity_ms();
        let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
        sc.draw_str(
            ix,
            y,
            if active { "Mouse  Active" } else { "Mouse  Idle" },
            if active { sc.theme.accent } else { sc.theme.chat_fg },
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "Pointer + wheel scroll", sc.theme.title_dim, bg);
    }
    y + ch
}
