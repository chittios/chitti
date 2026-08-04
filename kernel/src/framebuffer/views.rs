//! Action-pane views: the `/top` dashboard, audio/video/browser HUDs, the
//! todo list, and the editor surface.

use super::*;

/// One row in the `/top` process table (kernel tasks / agents / services).
pub struct TopTask<'a> {
    pub id: u64,
    pub name: &'a str,
    pub state: &'a str,
    /// Optional tree prefix (`|- `, `` ` - ``) for a light process-tree look.
    pub tree: &'a str,
}

/// A snapshot for [`draw_top`] — the `/top` dashboard's inputs, gathered by the
/// shell so the framebuffer layer stays free of `mm`/`smp` coupling.
pub struct TopView<'a> {
    /// Per-core busy percentage (index = core id).
    pub cores: &'a [u64],
    pub cores_online: u64,
    pub ram_used: u64,
    pub ram_total: u64,
    pub heap_used: u64,
    pub heap_total: u64,
    pub model_bytes: u64,
    pub uptime: &'a str,
    pub arch: &'a str,
    pub allocs: u64,
    pub datetime: &'a str,
    /// Process table rows (already sorted by the shell).
    pub tasks: &'a [TopTask<'a>],
    pub tasks_total: u64,
    pub tasks_running: u64,
    /// Average core utilisation (0..=100), shown as a load stand-in.
    pub load_pct: u64,
    pub net_up: bool,
    pub model_name: &'a str,
}

/// Render the `/top` dashboard in an **htop-like** layout into the action pane:
/// dual-column header (CPU/Mem meters | Tasks/Load/Uptime), a process table,
/// and an F-key footer. No-op unless the pane is in [`RightMode::Top`].
/// Refreshed ~1 Hz from the shell idle tick.
pub fn draw_top(v: &TopView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Top) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, iy, iw) = (d.ix, d.iy, d.iw);
        let bottom = iy + d.ih;
        // NO full-interior clear: overwrite in place (padded strings + self-
        // filling bars) so the 1 Hz refresh does not flicker.
        let bg = d.bg;
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let fmt = |b: u64| -> String {
            if b >= gib {
                alloc::format!("{}.{}G", b / gib, (b % gib) * 10 / gib)
            } else if b >= mib {
                alloc::format!("{}M", b / mib)
            } else {
                alloc::format!("{}K", b / 1024)
            }
        };
        let cols = (iw / cw).max(1) as usize;
        let gap = 2 * cw;
        // Two-column header when the pane is wide enough (≥ 48 cells).
        let two_col = cols >= 48;
        let left_w = if two_col { (iw - gap) / 2 } else { iw };
        let right_x = px + left_w + gap;
        let right_w = iw.saturating_sub(left_w + gap);

        // --- left: CPU + Mem + Heap meters (htop style) ---------------------
        let mut y_l = iy;
        // Compact meters: `1 [████░░░░] 34%` — 1-based like htop.
        for (i, &pct) in v.cores.iter().enumerate() {
            if y_l + ch > bottom {
                break;
            }
            let online = (i as u64) < v.cores_online;
            let lab = alloc::format!("{}", i + 1);
            let detail = if online {
                crate::textsel::fit_width(&alloc::format!("{}%", pct.min(100)), 4)
            } else {
                crate::textsel::fit_width("--%", 4)
            };
            y_l = sc.htop_meter(px, y_l, left_w, &lab, if online { pct } else { 0 }, &detail, bg);
        }
        let ram_pct = if v.ram_total > 0 { v.ram_used * 100 / v.ram_total } else { 0 };
        let heap_pct = if v.heap_total > 0 { v.heap_used * 100 / v.heap_total } else { 0 };
        if y_l + ch <= bottom {
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Mem",
                ram_pct,
                &crate::textsel::fit_width(&alloc::format!("{}/{}", fmt(v.ram_used), fmt(v.ram_total)), 11),
                bg,
            );
        }
        if y_l + ch <= bottom {
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Heap",
                heap_pct,
                &crate::textsel::fit_width(&alloc::format!("{}/{}", fmt(v.heap_used), fmt(v.heap_total)), 11),
                bg,
            );
        }
        // Model as a third "resource" bar (htop's Swp analog for our OS).
        if y_l + ch <= bottom && v.model_bytes > 0 {
            let model_pct = if v.ram_total > 0 {
                (v.model_bytes * 100 / v.ram_total).min(100)
            } else {
                0
            };
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Mdl",
                model_pct,
                &crate::textsel::fit_width(&fmt(v.model_bytes), 11),
                bg,
            );
        }

        // --- right: Tasks / Load / Uptime (htop info column) ---------------
        let mut y_r = iy;
        if two_col {
            let rcols = (right_w / cw).max(1) as usize;
            // Load as "N.NN" from average core % (htop's loadavg stand-in).
            let load_i = v.load_pct.min(999);
            let info = [
                alloc::format!("Tasks: {}, {} running", v.tasks_total, v.tasks_running),
                alloc::format!("Load average: {}.{:02}", load_i / 100, load_i % 100),
                alloc::format!("CPU avg: {}%", v.load_pct.min(100)),
                alloc::format!("Uptime: {}", v.uptime),
                alloc::format!("{}", v.datetime),
                alloc::format!("Arch: {}  ({} cores)", v.arch, v.cores_online),
                alloc::format!("Network: {}", if v.net_up { "up" } else { "down" }),
                alloc::format!(
                    "Model: {}",
                    crate::textsel::ellipsize(v.model_name, rcols.saturating_sub(8))
                ),
                alloc::format!("Heap allocs: {}", v.allocs),
            ];
            for s in &info {
                if y_r + ch > bottom {
                    break;
                }
                sc.draw_str_bg(
                    right_x,
                    y_r,
                    &crate::textsel::fit_width(s, rcols),
                    sc.theme.logs_fg,
                    bg,
                );
                y_r += ch;
            }
        }

        // Header block ends at the taller of the two columns.
        let mut y = y_l.max(y_r) + ch / 2;
        if y + 3 * ch > bottom {
            // Still paint footer if almost full.
            sc.cursor_overlay();
            return;
        }

        // --- process table header (htop green bar) -------------------------
        let hdr_bg = (0, 140, 80); // classic htop green
        let hdr_fg = (0, 0, 0);
        let hdr = if cols >= 60 {
            "  PID  STATE    NAME / COMMAND"
        } else if cols >= 40 {
            "  PID  STATE  COMMAND"
        } else {
            "  PID  COMMAND"
        };
        sc.fill_rect(px, y, iw, ch, hdr_bg);
        sc.draw_str_bg(px, y, &crate::textsel::fit_width(hdr, cols), hdr_fg, hdr_bg);
        y += ch;

        // --- process rows --------------------------------------------------
        let footer_h = ch; // reserve one line for F-keys
        // **How many rows fit, decided up front so truncation can be *reported*.**
        // This used to just `break` when it ran out of room, which reads as "these are
        // all the tasks" — and in a small pane that meant three rows standing in for a
        // dozen, so a running agent looked absent. Same rule as everywhere else here: no
        // silent caps.
        let room = if bottom > y + footer_h { ((bottom - y - footer_h) / ch) as usize } else { 0 };
        let truncated = v.tasks.len() > room;
        // Give up one row to the "+N more" marker when there is something to say.
        let show = if truncated { room.saturating_sub(1) } else { v.tasks.len() };
        let mut first_running_painted = false;
        for t in v.tasks.iter().take(show) {
            if y + ch + footer_h > bottom {
                break;
            }
            let is_run = t.state == "running";
            let sel = is_run && !first_running_painted;
            if sel {
                first_running_painted = true;
            }
            let row_bg = if sel {
                (0, 180, 200) // htop cyan selection
            } else {
                bg
            };
            let row_fg = if sel { (0, 0, 0) } else { sc.theme.chat_fg };
            let state_fg = if sel {
                (0, 0, 0)
            } else {
                match t.state {
                    "running" => (126, 214, 150),
                    "ready" => (240, 200, 120),
                    "parked" => sc.theme.title_dim,
                    "dead" => (255, 106, 110),
                    _ => sc.theme.logs_fg,
                }
            };
            sc.fill_rect(px, y, iw, ch, row_bg);
            // Columns: PID (5) STATE (8) tree+name
            let pid = crate::textsel::fit_width(&alloc::format!("{}", t.id), 5);
            let st = crate::textsel::fit_width(t.state, 8);
            let name_cols = cols.saturating_sub(5 + 1 + 8 + 1);
            let name = crate::textsel::fit_width(
                &alloc::format!("{}{}", t.tree, t.name),
                name_cols,
            );
            let mut xx = px;
            xx = sc.draw_str(xx, y, &pid, row_fg, row_bg);
            xx = sc.draw_str(xx, y, " ", row_fg, row_bg);
            xx = sc.draw_str(xx, y, &st, state_fg, row_bg);
            xx = sc.draw_str(xx, y, " ", row_fg, row_bg);
            let _ = sc.draw_str(xx, y, &name, row_fg, row_bg);
            y += ch;
        }
        // Say what was left out. The count is the honest one — total tasks the scheduler
        // holds, not just the ones that fit — so a full list in a taller pane and a
        // clipped one here describe the same system.
        if truncated && y + ch + footer_h <= bottom {
            let hidden = v.tasks.len().saturating_sub(show);
            let more = alloc::format!("  +{hidden} more of {} tasks -- taller pane to see all", v.tasks_total);
            sc.draw_str_bg(px, y, &crate::textsel::fit_width(&more, cols), sc.theme.title_dim, bg);
            y += ch;
        }
        // Blank any leftover process-area rows so a shrinking task list
        // leaves no ghost lines.
        let blank = crate::textsel::fit_width("", cols);
        while y + ch + footer_h <= bottom {
            sc.draw_str_bg(px, y, &blank, bg, bg);
            y += ch;
        }

        // --- F-key footer (htop style) -------------------------------------
        let foot_y = bottom.saturating_sub(ch);
        let foot_bg = sc.theme.status_bg;
        sc.fill_rect(px, foot_y, iw, ch, foot_bg);
        // Number in reverse / label dim — approximate with accent digits.
        let keys = [
            ("F1", "Help"),
            ("F2", "Setup"),
            ("F3", "Search"),
            ("F4", "Filter"),
            ("F5", "Tree"),
            ("F6", "Sort"),
            ("F9", "Kill"),
            ("F10", "Quit"),
        ];
        let mut fx = px;
        for (k, lab) in keys {
            if fx + (k.len() + lab.len() + 2) as u64 * cw > px + iw {
                break;
            }
            fx = sc.draw_str(fx, foot_y, k, foot_bg, sc.theme.accent); // reverse-ish
            fx = sc.draw_str(fx, foot_y, lab, sc.theme.logs_fg, foot_bg);
            fx += cw; // gap
        }
        // Right-align a short quit hint if room.
        let hint = " /close ";
        if cols > 20 {
            let hx = px + iw.saturating_sub(hint.len() as u64 * cw);
            if hx > fx {
                sc.draw_str(hx, foot_y, hint, sc.theme.title_dim, foot_bg);
            }
        }
        sc.cursor_overlay();
    });
}

/// A snapshot of the background audio player, for [`draw_audio`].
pub struct AudioView<'a> {
    pub name: &'a str,
    pub pos_ms: u64,
    pub total_ms: u64,
    pub rate: u32,
    pub playing: bool,
    pub paused: bool,
    /// Peak envelope `0..=255` for the wave visualizer (see `audio::waveform_peaks`).
    pub peaks: &'a [u8],
    /// Software volume percent (`0..=100`) and mute (from `sound::volume`/`muted`).
    pub volume: u32,
    pub muted: bool,
}

/// Paint the audio-player tab in the **same HUD layout as the video player**:
/// a centre **wave visualizer** (played = accent, remaining = dim) plus a
/// bottom control strip (status line, scrubber, shortcut hints). No-op unless
/// the audio tab is active. Called ~4 Hz from the shell while the tab is on top.
pub fn draw_audio(v: &AudioView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Audio) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let bg = d.bg;
        // Same HUD height as the video player so the two tabs feel identical.
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);

        // --- centre: waveform visualizer ---------------------------------
        // Compact band (about 1/3 of the content height, capped) centred in
        // the area above the HUD — not full-pane tall.
        let content_h = by.saturating_sub(py);
        let wave_h = (content_h / 3).clamp(ch * 2, ch * 5);
        let wave_top = py + content_h.saturating_sub(wave_h) / 2;
        let wave_x = px + cw * 2;
        let wave_w = pw.saturating_sub(4 * cw).max(1);
        // Clear the full content band once so a taller previous paint (or
        // leftover glyphs) never sticks around when the wave shrinks —
        // wallpaper-aware so the translucent desktop shows behind it.
        sc.paint_surface(px, py, pw, content_h.saturating_sub(1), bg);
        let mid = wave_top + wave_h / 2;
        let n_peaks = v.peaks.len().max(1);
        let play_x = if v.total_ms > 0 {
            (wave_w * v.pos_ms.min(v.total_ms) / v.total_ms).min(wave_w.saturating_sub(1))
        } else {
            0
        };
        for col in 0..wave_w {
            let pi = ((col as usize) * n_peaks) / (wave_w as usize).max(1);
            let peak = v.peaks.get(pi).copied().unwrap_or(0) as u64;
            // Half-height bar (mirrored above/below centre); min 1px when energy.
            let half = if peak == 0 {
                0
            } else {
                ((wave_h / 2 - 1) * peak / 255).max(1)
            };
            let color = if col <= play_x { sc.theme.accent } else { sc.theme.title_dim };
            // Clear the column then draw the bar.
            sc.fill_rect(wave_x + col, wave_top, 1, wave_h, bg);
            if half > 0 {
                sc.fill_rect(wave_x + col, mid.saturating_sub(half), 1, half * 2, color);
            } else {
                // Quiet: a 1px centre tick so the track silhouette stays visible.
                sc.fill_rect(wave_x + col, mid, 1, 1, sc.theme.sep_dim);
            }
        }
        // Playhead: thin bright line at the current position.
        sc.fill_rect(wave_x + play_x, wave_top, 2.max(sc.scale), wave_h, sc.theme.chat_fg);

        // --- bottom HUD (mirrors `draw_video_status`) --------------------
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let state = if v.paused {
            "||"
        } else if v.playing {
            ">"
        } else {
            "="
        };
        let time = alloc::format!("{} / {}", mmss(v.pos_ms), mmss(v.total_ms));
        let vol = if v.muted {
            String::from("muted")
        } else {
            alloc::format!("vol {}%", v.volume.min(100))
        };
        // Drop less-critical fields as the pane narrows so the line always fits.
        let candidates = [
            alloc::format!("{} {}  {}  {}", state, v.name, time, vol),
            alloc::format!("{} {}  {}", state, v.name, time),
            alloc::format!("{} {}", state, v.name),
            alloc::format!("{} {}", state, crate::textsel::ellipsize(v.name, cols.saturating_sub(3).max(1))),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(&alloc::format!("{} {}", state, v.name), cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;
        // Scrubber in the control strip (video-style), not the main area.
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if v.total_ms > 0 {
            (track_w * v.pos_ms.min(v.total_ms) / v.total_ms).min(track_w)
        } else {
            0
        };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;
        // Shortcut hints: wrap; drop tokens that can't fit even alone.
        let hints = [
            "space play/pause",
            "<-/-> seek",
            "up/dn volume",
            "0 restart",
            "m mute",
            "Ctrl+C stop",
        ];
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue; // too wide even alone — hide
            }
            let cand = if linebuf.is_empty() {
                String::from(h)
            } else {
                alloc::format!("{}{}{}", linebuf, sep, h)
            };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    break; // no room for another hint row
                }
                sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
                y += ch;
                linebuf = String::from(h);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom + ch {
            sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
        }
        sc.cursor_overlay();
    });
}

/// Overlay the **browser** control/status bar (same layout family as the video
/// player HUD): title + URL, a scroll scrubber, and keyboard shortcut hints.
/// Call *after* [`present_surface_reserve`] so the strip sits on the reserved
/// bottom region. No-op unless the browser surface tab is active.
pub fn draw_browser_status(
    title: &str,
    url: &str,
    scroll_y: i32,
    content_h: i32,
    view_h: i32,
    focused_input: bool,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(BROWSER_SURFACE)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let bg = d.bg;
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);

        let max_scroll = (content_h - view_h).max(0);
        let scroll_pct = if max_scroll > 0 {
            ((scroll_y as i64 * 100) / max_scroll as i64).clamp(0, 100) as u32
        } else {
            0
        };
        let mode = if focused_input { "input" } else { "nav" };
        // Drop fields as the pane narrows (video HUD pattern).
        let scroll_s = if max_scroll > 0 {
            alloc::format!("scroll {}%  {}/{}", scroll_pct, scroll_y, max_scroll)
        } else {
            String::from("top")
        };
        let candidates = [
            alloc::format!("{}  {}  {}  [{}]", title, url, scroll_s, mode),
            alloc::format!("{}  {}  [{}]", title, scroll_s, mode),
            alloc::format!("{}  {}", title, scroll_s),
            alloc::format!(
                "{}  {}",
                crate::textsel::ellipsize(title, cols.saturating_sub(12).max(4)),
                scroll_s
            ),
            crate::textsel::ellipsize(title, cols),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(title, cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;

        // Scroll scrubber (full-width track, filled = position).
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if max_scroll > 0 {
            (track_w * scroll_y as u64 / max_scroll as u64).min(track_w)
        } else {
            0
        };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;

        // Shortcut hints — wrap like the video player.
        let hints = if focused_input {
            [
                "type text",
                "Bksp erase",
                "Tab next",
                "Enter submit",
                "Esc unfocus",
                "wheel scroll",
            ]
        } else {
            [
                "j/k scroll",
                "space page",
                "wheel scroll",
                "b back",
                "r reload",
                "click link/form",
            ]
        };
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue;
            }
            let cand = if linebuf.is_empty() {
                String::from(h)
            } else {
                alloc::format!("{}{}{}", linebuf, sep, h)
            };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    break;
                }
                sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
                y += ch;
                linebuf = String::from(h);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom + ch {
            sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
        }
        let _ = url; // included in line1 when the pane is wide enough
        sc.cursor_overlay();
    });
}

/// Height in px the browser HUD reserves (same formula as the video player).
pub fn browser_hud_height() -> u64 {
    video_hud_height()
}

/// Overlay the video player's control/status bar along the bottom of the video
/// surface pane: playback state, mm:ss / mm:ss, frame counter, mute, a scrubber,
/// and the key-shortcut hints. Drawn *after* the frame blit (present_surface
/// clears the pane each present), so it sits on top like a real player's HUD.
/// No-op unless the video surface tab is active.
///
/// `fps` is the instantaneous / smoothed decode+present FPS (0 = unknown / paused).
#[allow(clippy::too_many_arguments)]
pub fn draw_video_status(
    name: &str,
    playing: bool,
    muted: bool,
    has_audio: bool,
    frame: usize,
    frames: usize,
    pos_ms: u64,
    total_ms: u64,
    volume: u32,
    fps: u32,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(VIDEO_SURFACE)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        // Reserved HUD strip (below the video frame). Fill the whole strip once
        // so time/fps string length changes never leave glyph trails; the strip
        // is small (~4 lines) so this is cheap and does not flash the picture.
        let bg = d.bg;
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        // Usable text width in whole glyph cells, with a one-cell left margin.
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let state = if playing { ">" } else { "||" };
        let vol = if !has_audio {
            String::from("no audio")
        } else if muted {
            String::from("muted")
        } else {
            alloc::format!("vol {}%", volume.min(100))
        };
        let fps_s = if fps > 0 {
            alloc::format!("{} fps", fps)
        } else {
            String::from("-- fps")
        };
        // Drop fields as the pane narrows so the status line never overflows.
        let time = alloc::format!("{} / {}", mmss(pos_ms), mmss(total_ms));
        let fr = alloc::format!("{}/{}", frame, frames);
        let candidates = [
            alloc::format!("{} {}  {}  {}  {}  {}", state, name, time, fr, fps_s, vol),
            alloc::format!("{} {}  {}  {}  {}", state, name, time, fps_s, vol),
            alloc::format!("{} {}  {}  {}", state, name, time, fps_s),
            alloc::format!("{} {}  {}", state, name, time),
            alloc::format!("{} {}", state, name),
            alloc::format!("{} {}", state, crate::textsel::ellipsize(name, cols.saturating_sub(3).max(1))),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(&alloc::format!("{} {}", state, name), cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;
        // Scrubber: a full-width track with a filled portion for progress
        // (self-filling — the whole track is overwritten each refresh).
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if total_ms > 0 { (track_w * pos_ms.min(total_ms) / total_ms).min(track_w) } else { 0 };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;
        // Shortcuts: wrap; hide tokens that don't fit; stop when HUD is full.
        let hints = ["space play/pause", "<-/-> seek", "up/dn volume", "0 restart", "m mute", "Ctrl+C stop"];
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue;
            }
            let cand = if linebuf.is_empty() { String::from(h) } else { alloc::format!("{}{}{}", linebuf, sep, h) };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    break;
                }
                sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
                y += ch;
                linebuf = String::from(h);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom + ch {
            sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
        }
        sc.cursor_overlay();
    });
}

/// The editor pane text-area geometry `(ix, iy, cw, ch, cols, text_rows)` so the
/// editor can map a click to a (row, col). `None` unless the editor is open.
pub fn editor_pane_geom() -> Option<(u64, u64, u64, u64, u64, u64)> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            let d = sc.mode_dims(RightMode::Editor)?;
            Some((d.ix, d.iy, d.cw, d.ch, d.cols, d.rows.saturating_sub(1)))
        })
    })
}

/// The editor viewport size `(cols, rows)` inside the right pane — `rows` is the
/// text area (the bottom row is reserved for the editor's mode line). `None` if
/// the console isn't up.
pub fn editor_dims() -> Option<(usize, usize)> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            // The editor's own column when its tab is up (it may not be the
            // focused one), else the focused column for a not-yet-opened editor.
            let (cols, rows) = match sc.mode_dims(RightMode::Editor) {
                Some(d) => (d.cols, d.rows),
                None => (sc.logs().cols, sc.logs().rows),
            };
            (cols as usize, (rows.saturating_sub(1)).max(1) as usize)
        })
    })
}

/// Height in px the video HUD reserves at the bottom of the action pane — the
/// player blits its frame above this, and [`draw_video_status`] fills it.
pub fn video_hud_height() -> u64 {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.ch() * 4 + sc.ch() / 2).unwrap_or(0))
}

/// Height in px the PDF viewer's status strip reserves — two text rows: where
/// you are in the document, and the keys. Shorter than the video HUD because a
/// document needs no scrubber or volume.
pub fn pdf_hud_height() -> u64 {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.ch() * 2 + sc.ch() / 2).unwrap_or(0))
}

/// Overlay the PDF viewer's status strip along the bottom of its pane. Drawn
/// *after* the page blit, in the strip `present_surface_reserve` left alone, so
/// it updates in place rather than flickering under each re-render.
///
/// `line` comes from [`crate::pdfview::hud`] — built there, not here, so the
/// wording is unit-tested; this function only lays it out. It is split on the
/// double spaces the author used to group fields, then packed into rows that fit
/// the pane, which is the same trick the vertical status bar uses: a document
/// pane can be narrow, and an ellipsized status line loses the page number,
/// which is the one field a reader actually needs.
pub fn draw_pdf_status(line: &str) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(PDF_SURFACE)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let (ch, cw) = (sc.ch(), sc.cw());
        let (px, py, pw, ph) = (d.ix, d.iy, d.iw, d.ih);
        let bg = d.bg;
        let barh = ch * 2 + ch / 2;
        let by = py + ph.saturating_sub(barh);
        // Fill the whole strip once so a shorter status never leaves glyph
        // trails, then a hairline to separate it from the page.
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent);
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let mut y = by + ch / 3;
        let hud_bottom = py + ph;
        // The first row is the position/zoom group (accent), the rest hints.
        let mut first = true;
        let mut linebuf = String::new();
        let flush = |sc: &mut Screen, y: &mut u64, s: &str, first: bool| {
            let colour = if first { sc.theme.accent } else { sc.theme.logs_fg };
            sc.draw_str_bg(px + cw, *y, &crate::textsel::fit_width(s, cols), colour, bg);
            *y += ch;
        };
        for group in line.split("  ").filter(|s| !s.trim().is_empty()) {
            let group = group.trim();
            let cand = if linebuf.is_empty() {
                String::from(group)
            } else {
                alloc::format!("{}  {}", linebuf, group)
            };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    return;
                }
                flush(sc, &mut y, &linebuf, first);
                first = false;
                linebuf = String::from(group);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom {
            flush(sc, &mut y, &linebuf, first);
        }
        sc.cursor_overlay();
    });
}

/// Open the `/top` dashboard in the action pane (filled by the shell's idle
/// tick). Returns true if it is now open (false if it was already).
pub fn open_top() {
    set_right(RightMode::Top);
}

/// Whether the action pane currently shows `/top`.
pub fn is_top() -> bool {
    right_mode() == RightMode::Top
}

/// Open the live todos pane.
pub fn open_todos() {
    set_right(RightMode::Todos);
}

/// Whether the action pane shows todos.
pub fn is_todos() -> bool {
    right_mode() == RightMode::Todos
}

/// One row for [`draw_todos`].
pub struct TodoViewItem<'a> {
    pub id: u32,
    pub text: &'a str,
    pub status: &'a str,
}

/// Render the session todo list into the action pane (checklist view).
pub fn draw_todos(items: &[TodoViewItem<'_>], title: &str) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Todos) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let (px, iy, iw) = (d.ix, d.iy, d.iw);
        let bg = d.bg;
        let rows = d.rows;
        let cols = (iw / sc.cw()).max(1) as usize;
        let mut y = iy;
        let head = if title.is_empty() {
            alloc::format!("{} Todos", crate::icons::fa::LIST_CHECK)
        } else {
            alloc::format!("{} {title}", crate::icons::fa::LIST_CHECK)
        };
        let head_fmt = pad_trunc(&head, cols);
        sc.draw_str_bg(px, y, &head_fmt, sc.theme.accent, bg);
        y += ch + ch / 4;
        if items.is_empty() {
            sc.draw_str_bg(px, y, &pad_trunc("(no todos — agent todo_write)", cols), sc.theme.title_dim, bg);
            sc.cursor_overlay();
            return;
        }
        for it in items {
            use crate::icons::fa;
            let mark = match it.status {
                "done" => fa::SQUARE_CHECK,
                "in_progress" => fa::CHEVRON_RIGHT,
                "cancelled" => fa::BAN,
                _ => fa::SQUARE,
            };
            let row = alloc::format!("{mark} {}: {}", it.id, it.text);
            let fg = match it.status {
                "done" => sc.theme.title_dim,
                "in_progress" => sc.theme.accent,
                _ => sc.theme.logs_fg,
            };
            sc.draw_str_bg(px, y, &pad_trunc(&row, cols), fg, bg);
            y += ch;
            if y + ch > iy + rows * ch {
                break;
            }
        }
        let blank = pad_trunc("", cols);
        while y + ch <= iy + rows * ch {
            sc.draw_str_bg(px, y, &blank, bg, bg);
            y += ch;
        }
        sc.cursor_overlay();
    });
}

/// Open (or focus) the `/open` editor tab.
pub fn editor_enter() {
    set_right(RightMode::Editor);
}

/// Close the editor tab (the editor quit); the active tab falls back to a
/// sibling, or the pane collapses if the editor was the only tab.
pub fn editor_leave() {
    close_tab_mode(RightMode::Editor);
}

/// Render the editor into the right pane: title `editor: <file>`, the visible
/// slice of `lines` from `top`, a reverse-video block cursor at
/// `(cur_row, cur_col)`, and a bottom mode line. Soft-wraps long lines so the
/// full buffer is reachable (vim-like; previously clipped mid-line).
///
/// `top` is the first **visual** row (soft-wrap aware). `hl` is optional per-byte
/// syntax colours for logical lines starting at `hl_base` (index 0 = that line;
/// `None` entries fall back to the theme's `editor_fg`).
#[allow(clippy::too_many_arguments)]
pub fn editor_render(
    title: &str,
    lines: &[alloc::string::String],
    top: usize,
    cur_row: usize,
    cur_col: usize,
    modeline: &str,
    sel: Option<((usize, usize), (usize, usize))>,
    hl: Option<&[Vec<Option<Rgb>>]>,
    hl_base: usize,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        // Paint into the column that holds the editor tab, not the focused one.
        let Some(ei) = sc.mode_column(RightMode::Editor) else {
            return;
        };
        let d = PaneDims::of(&sc.actions[ei].pane);
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, pw, cw, ch, cols, rows) = (d.x, d.w, d.cw, d.ch, d.cols, d.rows);
        let (ix, iy) = (d.ix, d.iy);
        sc.draw_frame_titled(&sc.actions[ei].pane, true, title);
        // Clear the interior to the editor background — wallpaper-aware so the
        // translucent desktop shows behind the editor too (glyphs blend via
        // `blit_glyph`/`bg_at`).
        sc.paint_surface(ix, iy, cols * cw, rows * ch, sc.theme.editor_bg);
        let text_rows = rows.saturating_sub(1);
        // Is text (row, col) inside the inclusive selection range?
        let in_sel = |row: usize, col: usize| -> bool {
            let Some(((r1, c1), (r2, c2))) = sel else { return false };
            if row < r1 || row > r2 {
                return false;
            }
            let after_start = row > r1 || col >= c1;
            let before_end = row < r2 || col <= c2;
            after_start && before_end
        };
        // Line-number gutter width (digits + 1 space).
        let gutter = {
            let mut n = lines.len().max(1);
            let mut w = 1;
            while n >= 10 {
                n /= 10;
                w += 1;
            }
            (w + 1) as u64
        };
        let tw = (cols.saturating_sub(gutter) as usize).max(1);
        let lenses: alloc::vec::Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
        // Walk visual rows from `top`, painting one soft-wrap segment per screen row.
        for i in 0..text_rows {
            let vis = top + i as usize;
            let (li, seg) = crate::editor_wrap::unvis(&lenses, vis, tw);
            if li >= lines.len() {
                break;
            }
            // Past the end of the buffer's visual extent — stop filling.
            let total_vis = {
                let mut t = 0usize;
                for &len in &lenses {
                    t += crate::editor_wrap::soft_wraps(len, tw);
                }
                t
            };
            if vis >= total_vis {
                break;
            }
            let y = iy + i * ch;
            // Gutter: number only on the first wrap segment of a logical line.
            let mut x = ix;
            if seg == 0 {
                let num = alloc::format!("{:>width$} ", li + 1, width = (gutter - 1) as usize);
                for chr in num.chars() {
                    sc.blit_glyph(x, y, chr, sc.theme.editor_lineno, sc.theme.editor_bg);
                    x += cw;
                }
            } else {
                // Continuation marker gutter (spaces) so wrap segments stay aligned.
                for _ in 0..gutter {
                    sc.blit_glyph(x, y, ' ', sc.theme.editor_lineno, sc.theme.editor_bg);
                    x += cw;
                }
            }
            let start = seg * tw;
            let line = &lines[li];
            let hl_row = li.saturating_sub(hl_base);
            for (off, chr) in line.chars().enumerate().skip(start).take(tw) {
                let col = off;
                let bg = if in_sel(li, col) {
                    sc.theme.editor_sel
                } else {
                    sc.theme.editor_bg
                };
                let fg = hl
                    .and_then(|h| h.get(hl_row))
                    .and_then(|v| v.get(col).copied().flatten())
                    .unwrap_or(sc.theme.editor_fg);
                sc.blit_glyph(x, y, chr, fg, bg);
                x += cw;
            }
        }
        // Reverse-video block cursor on the soft-wrap cell that holds (cur_row, cur_col).
        {
            let cur_vis = crate::editor_wrap::vis_index(&lenses, cur_row, cur_col, tw);
            if cur_vis >= top && (cur_vis - top) < text_rows as usize {
                let scr = (cur_vis - top) as u64;
                let col_in_seg = (cur_col % tw) as u64;
                let col_on_screen = gutter + col_in_seg;
                if col_on_screen < cols {
                    let y = iy + scr * ch;
                    let x = ix + col_on_screen * cw;
                    let chr = lines
                        .get(cur_row)
                        .and_then(|l| l.chars().nth(cur_col))
                        .unwrap_or(' ');
                    let chr = if chr.is_control() { ' ' } else { chr };
                    sc.blit_glyph(x, y, chr, sc.theme.editor_bg, sc.theme.accent);
                }
            }
        }
        // Mode line across the bottom interior row — ellipsize so a long path
        // never paints past the pane edge.
        let sy = iy + text_rows * ch;
        sc.paint_surface(px + BORDER, sy, pw - 2 * BORDER, ch, sc.theme.status_bg);
        let ml = crate::textsel::ellipsize(modeline, cols as usize);
        let mut x = ix;
        for chr in ml.chars() {
            sc.blit_glyph(x, sy, chr, sc.theme.title_active, sc.theme.status_bg);
            x += cw;
        }
        sc.cursor_overlay();
    });
}

impl Screen {
    /// A labelled horizontal usage bar filled proportional to `pct` (0..=100)
    /// and coloured green/amber/red by load, over background `bg`. Returns the
    /// y below the bar.
    fn usage_bar_bg(&self, x: u64, y: u64, w: u64, label: &str, pct: u64, detail: &str, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let lab_w = 7 * cw; // fixed label column
        self.draw_str(x, y, label, self.theme.chat_fg, bg);
        let bx = x + lab_w;
        // Reserve gap(1) + detail(11) + margin(1) = 13 cells after the bar, so
        // the padded detail text never runs past the pane's right edge/border.
        let bw = w.saturating_sub(lab_w + 13 * cw);
        let bh = ch;
        // Border (static — re-stroking the same pixels never blanks).
        self.rect_outline(bx, y, bw, bh, 1, self.theme.border_dim);
        let p = pct.min(100);
        let inner = bw.saturating_sub(2);
        let fill = inner * p / 100;
        let color = if p < 60 {
            (126, 214, 150) // green
        } else if p < 85 {
            (240, 200, 120) // amber
        } else {
            (255, 106, 110) // red
        };
        // Repaint in place: colour the filled span, then background the rest —
        // the coloured region is never blanked to bg first, so no flicker.
        if fill > 0 {
            self.fill_rect(bx + 1, y + 1, fill, bh.saturating_sub(2), color);
        }
        if inner > fill {
            self.fill_rect(bx + 1 + fill, y + 1, inner - fill, bh.saturating_sub(2), self.theme.chat_bg);
        }
        // Detail (e.g. "512M/6.0G") after the bar — ellipsize + pad so it never
        // overflows the pane and shrinking values leave no residue.
        let detail_x = bx + bw + cw;
        let avail = ((x + w).saturating_sub(detail_x) / cw) as usize;
        let d = crate::textsel::fit_width(detail, avail);
        self.draw_str(detail_x, y, &d, self.theme.title_dim, bg);
        y + ch + ch / 3
    }

    /// Compact htop-style meter: `1 [████░░░░] 34%` — short label, thin bar
    /// with green→amber→red fill, fixed-width detail. Returns next y.
    fn htop_meter(&self, x: u64, y: u64, w: u64, label: &str, pct: u64, detail: &str, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let lab = crate::textsel::fit_width(label, 4);
        self.draw_str(x, y, &lab, self.theme.chat_fg, bg);
        let bx = x + 5 * cw;
        // Reserve space for detail (already padded by caller, ~4–11 cells).
        let det_cols = detail.chars().count().max(4) as u64 + 1;
        let bw = w.saturating_sub(5 * cw + det_cols * cw);
        let bh = ch.saturating_sub(2).max(ch * 3 / 4);
        let by = y + (ch.saturating_sub(bh)) / 2;
        // Track outline + fill (htop-like).
        self.rect_outline(bx, by, bw, bh, 1, self.theme.border_dim);
        let p = pct.min(100);
        let inner = bw.saturating_sub(2);
        let fill = inner * p / 100;
        // Gradient-ish: green low, amber mid, red high — solid for simplicity.
        let color = if p < 50 {
            (80, 200, 120) // htop green
        } else if p < 75 {
            (220, 180, 60) // yellow
        } else if p < 90 {
            (230, 120, 50) // orange
        } else {
            (230, 60, 60) // red
        };
        if fill > 0 {
            self.fill_rect(bx + 1, by + 1, fill, bh.saturating_sub(2), color);
        }
        if inner > fill {
            self.fill_rect(
                bx + 1 + fill,
                by + 1,
                inner - fill,
                bh.saturating_sub(2),
                self.theme.chat_bg,
            );
        }
        let detail_x = bx + bw + cw;
        let avail = ((x + w).saturating_sub(detail_x) / cw) as usize;
        let d = crate::textsel::fit_width(detail, avail);
        self.draw_str(detail_x, y, &d, self.theme.title_dim, bg);
        y + ch
    }
}

#[cfg(test)]
mod hud_tests {
    use super::{hud_strip_height, wrapped_hint_lines};

    #[test_case]
    fn hud_hint_wrapping_counts_lines() {
        // No hint text (status only) → zero hint rows.
        assert_eq!(wrapped_hint_lines("just a status", 40), 0);
        // One short hint fits on one row.
        assert_eq!(wrapped_hint_lines("status\nn new", 40), 1);
        // A long single hint wraps by words to fit narrow columns.
        let hud = "status\narrows move  enter select  esc clear  n new game";
        assert_eq!(wrapped_hint_lines(hud, 80), 1, "fits one row when wide");
        assert!(wrapped_hint_lines(hud, 20) >= 3, "narrow pane wraps to several rows");
        // Two explicit hint lines are at least two rows.
        assert!(wrapped_hint_lines("s\nfirst line\nsecond line", 80) >= 2);
    }

    #[test_case]
    fn hud_strip_height_matches_status_plus_hints() {
        // Chess-shaped HUD: status + one shortcut line. Pure — must stay free
        // of SCREEN so present_surface_hud can compute reserve outside the
        // draw critical section (re-entering SCREEN deadlocks).
        let hud = "Your move (White)\narrows/click move  enter select  esc clear  n new game";
        let ch = 16u64;
        let cols = 40usize;
        let hints = wrapped_hint_lines(hud, cols);
        let expect = (1 + hints) * ch + ch / 2 + 2;
        assert_eq!(hud_strip_height(hud, ch, cols), expect);
        assert!(hud_strip_height(hud, ch, cols) > ch, "strip taller than one cell");
        // Status-only is shorter than status+hints.
        assert!(hud_strip_height(hud, ch, cols) > hud_strip_height("status only", ch, cols));
    }
}
