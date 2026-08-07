//! Live screen-recording session — start/stop, frame pump, status chip.
//!
//! macOS shape: you start, work, stop. No fixed 5-second default. The global
//! shortcut (`Cmd+Shift+5` / `Ctrl+Shift+5`) toggles; the status-bar chip
//! shows elapsed time and clicking it stops. A safety auto-stop at
//! [`crate::screencast::MAX_DURATION_MS`] ends a forgotten take.
//!
//! Frames are captured from [`crate::shell::upkeep`] so the UI stays live —
//! the standing cooperative rule.

use super::*;
use crate::mm::Locked;
use crate::screencast::{self, Request, Verb};
use crate::video::h264::encoder::Encoder;
use crate::video::mp4_mux::{self, Sample};
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(not(test))]
fn capture(extent: crate::screenshot::Extent, cursor: bool) -> Result<(u32, u32, Vec<u32>), &'static str> {
    // Same path as `/screenshot` — framebuffer is device memory (ring 0).
    crate::framebuffer::capture(extent, cursor)
}

#[cfg(test)]
fn capture(
    _extent: crate::screenshot::Extent,
    _cursor: bool,
) -> Result<(u32, u32, Vec<u32>), &'static str> {
    Err("no framebuffer (serial-only boot)")
}

struct Live {
    extent: crate::screenshot::Extent,
    cursor: bool,
    scale_pct: u32,
    fps: u32,
    dest: Option<String>,
    /// Optional auto-stop (from `for <d>`). Distinct from the hard safety max.
    timed_ms: Option<u64>,
    encoder: Encoder,
    samples: Vec<Sample>,
    out_w: u32,
    out_h: u32,
    started_ms: u64,
    next_due_ms: u64,
    frame_i: usize,
    keyint: usize,
}

static LIVE: Locked<Option<Live>> = Locked::new(None);

/// True while a take is in progress.
pub fn is_recording() -> bool {
    LIVE.with(|s| s.is_some())
}

/// Elapsed ms of the live take, or 0.
pub fn elapsed_ms() -> u64 {
    LIVE.with(|s| {
        s.as_ref()
            .map(|l| crate::arch::now_ms().saturating_sub(l.started_ms))
            .unwrap_or(0)
    })
}

/// Status-bar chip text — empty when idle so the template drops the separator.
pub fn chip_text() -> String {
    screencast::chip_text(is_recording(), elapsed_ms())
}

/// `/record …` entry point.
pub(super) fn run_record(arg: &str) {
    let a = arg.trim();
    let req = match screencast::parse(a) {
        Ok(r) => r,
        Err(e) => {
            serial_println!("record> {e}");
            serial_println!("record> try /record help");
            return;
        }
    };
    match req.verb {
        Verb::Help => {
            serial_println!("record> usage: /record                  start or stop (toggle)");
            serial_println!("                /record start [options] [dest]");
            serial_println!("                /record stop");
            serial_println!("                /record status");
            serial_println!("                /record for <n>[s|m] …  optional timed take");
            serial_println!(
                "            options: fps <n>  scale <pct>  desktop|panel|chat|pane n|region x,y,w,h  --cursor"
            );
            serial_println!(
                "            default: until you stop · {} fps · {}% scale · H.264/MP4",
                screencast::DEFAULT_FPS,
                screencast::DEFAULT_SCALE_PCT
            );
            serial_println!(
                "            shortcut: Cmd+Shift+5 / Ctrl+Shift+5  (toggle, like macOS)"
            );
            serial_println!("            status bar: red ● while recording — click it to stop");
        }
        Verb::Status => {
            if is_recording() {
                let e = elapsed_ms() / 1000;
                serial_println!(
                    "record> recording… {:}:{:02}  ({} frame(s) so far) — /record stop or Cmd+Shift+5",
                    e / 60,
                    e % 60,
                    LIVE.with(|s| s.as_ref().map(|l| l.samples.len()).unwrap_or(0))
                );
            } else {
                serial_println!("record> idle — /record start  or  Cmd+Shift+5");
            }
        }
        Verb::Stop => {
            stop_and_save();
        }
        Verb::Start => {
            if is_recording() {
                serial_println!("record> already recording — /record stop first");
                return;
            }
            start(req);
        }
        Verb::Toggle => {
            if is_recording() {
                stop_and_save();
            } else {
                start(req);
            }
        }
    }
}

/// Global shortcut entry (Cmd+Shift+5): always toggle.
pub(super) fn toggle_shortcut() {
    if is_recording() {
        stop_and_save();
    } else {
        start(Request::default());
    }
}

fn confine_extent(mut req: Request) -> Option<Request> {
    let agent = active_agent_id();
    if crate::shell::in_tool_call() && agent != crate::agent::manifest::ORCHESTRATOR_ID.0 {
        let task = crate::sched::current_task_id();
        match crate::synapse::ui::surface_of_owner(task) {
            Some(id) => {
                if req.extent != crate::screenshot::Extent::Desktop {
                    serial_println!(
                        "record> confined to your own surface (agent {agent}); \
                         ignoring the requested extent"
                    );
                }
                req.extent = crate::screenshot::Extent::Surface(id);
            }
            None => {
                serial_println!(
                    "record> refused: agent {agent} has no surface of its own, and \
                     only the shell agent may capture the whole screen"
                );
                return None;
            }
        }
    }
    Some(req)
}

fn start(req: Request) {
    let Some(req) = confine_extent(req) else {
        return;
    };
    // First frame now so "started" is real and we know geometry.
    let (w0, h0, pixels) = match capture(req.extent, req.cursor) {
        Ok(v) => v,
        Err(e) => {
            serial_println!("record> {e}");
            return;
        }
    };
    let (sw, sh) = screencast::scaled_size(w0, h0, req.scale_pct);
    let (fw0, fh0, frame_px) = if sw == w0 && sh == h0 {
        (w0, h0, pixels)
    } else {
        let img = crate::image::Image {
            w: w0 as usize,
            h: h0 as usize,
            pixels,
        };
        let scaled = crate::image::resize(&img, sw as usize, sh as usize);
        (scaled.w as u32, scaled.h as u32, scaled.pixels)
    };
    let (fw, fh) = crate::video::h264::encoder::align_mb(fw0, fh0);
    let frame_px = if fw == fw0 && fh == fh0 {
        frame_px
    } else {
        crate::video::h264::encoder::crop_rgb32(
            fw0 as usize,
            fh0 as usize,
            &frame_px,
            fw as usize,
            fh as usize,
        )
    };
    let mut encoder = match Encoder::new(fw as usize, fh as usize, screencast::DEFAULT_QP) {
        Ok(e) => e,
        Err(e) => {
            serial_println!("record> encoder init failed: {e}");
            return;
        }
    };
    let au = match encoder.encode_rgb32(&frame_px, true) {
        Ok(a) => a,
        Err(e) => {
            serial_println!("record> encode failed: {e}");
            return;
        }
    };
    let sample_dur = screencast::sample_duration_ms(req.fps);
    let now = crate::arch::now_ms();
    let interval = screencast::frame_interval_ms(req.fps);
    let keyint = (req.fps.max(1) * 2) as usize;
    let live = Live {
        extent: req.extent,
        cursor: req.cursor,
        scale_pct: req.scale_pct,
        fps: req.fps,
        dest: req.dest,
        timed_ms: req.duration_ms,
        encoder,
        samples: {
            let mut v = Vec::new();
            v.push(Sample {
                bytes: au,
                duration: sample_dur,
                sync: true,
            });
            v
        },
        out_w: fw,
        out_h: fh,
        started_ms: now,
        next_due_ms: now.saturating_add(interval),
        frame_i: 1,
        keyint,
    };
    LIVE.with(|s| *s = Some(live));
    // Force a status-bar repaint so the ● chip appears immediately.
    update_status();
    serial_println!(
        "record> ● recording  ({} fps, {}% scale) — stop with /record stop, the status-bar chip, or Cmd+Shift+5",
        req.fps,
        req.scale_pct
    );
}

/// Pump: capture + encode at most one frame. Called from `upkeep`.
pub fn tick() {
    let now = crate::arch::now_ms();
    // Auto-stop decisions outside the lock (stop_and_save takes it).
    let should_stop = LIVE.with(|slot| {
        let Some(live) = slot.as_ref() else {
            return false;
        };
        let elapsed = now.saturating_sub(live.started_ms);
        if elapsed >= screencast::MAX_DURATION_MS {
            return true;
        }
        if let Some(t) = live.timed_ms {
            if elapsed >= t {
                return true;
            }
        }
        if live.samples.len() >= screencast::MAX_FRAMES {
            return true;
        }
        false
    });
    if should_stop {
        serial_println!("record> auto-stop");
        stop_and_save();
        return;
    }

    // One frame if due.
    let mut frame_job: Option<(crate::screenshot::Extent, bool, u32, u32, usize, u32)> = None;
    LIVE.with(|slot| {
        let Some(live) = slot.as_mut() else {
            return;
        };
        if now < live.next_due_ms {
            return;
        }
        frame_job = Some((
            live.extent,
            live.cursor,
            live.scale_pct,
            live.fps,
            live.frame_i,
            live.keyint as u32,
        ));
        let interval = screencast::frame_interval_ms(live.fps);
        live.next_due_ms = now.saturating_add(interval);
    });
    let Some((extent, cursor, scale_pct, fps, frame_i, keyint)) = frame_job else {
        return;
    };

    let (w0, h0, pixels) = match capture(extent, cursor) {
        Ok(v) => v,
        Err(_) => return, // skip this tick; try again next interval
    };
    let (sw, sh) = screencast::scaled_size(w0, h0, scale_pct);
    let (fw0, fh0, frame_px) = if sw == w0 && sh == h0 {
        (w0, h0, pixels)
    } else {
        let img = crate::image::Image {
            w: w0 as usize,
            h: h0 as usize,
            pixels,
        };
        let scaled = crate::image::resize(&img, sw as usize, sh as usize);
        (scaled.w as u32, scaled.h as u32, scaled.pixels)
    };
    let (fw, fh) = crate::video::h264::encoder::align_mb(fw0, fh0);
    let frame_px = if fw == fw0 && fh == fh0 {
        frame_px
    } else {
        crate::video::h264::encoder::crop_rgb32(
            fw0 as usize,
            fh0 as usize,
            &frame_px,
            fw as usize,
            fh as usize,
        )
    };

    LIVE.with(|slot| {
        let Some(live) = slot.as_mut() else {
            return;
        };
        if fw != live.out_w || fh != live.out_h {
            // Geometry change: leave the sample list as-is; stop on next tick.
            return;
        }
        let force_idr = keyint > 0 && frame_i % keyint as usize == 0;
        match live.encoder.encode_rgb32(&frame_px, force_idr) {
            Ok(au) => {
                live.samples.push(Sample {
                    bytes: au,
                    duration: screencast::sample_duration_ms(fps),
                    sync: force_idr,
                });
                live.frame_i += 1;
            }
            Err(_) => {}
        }
    });
    // Refresh the elapsed chip about once a second.
    if frame_i % fps.max(1) as usize == 0 {
        update_status();
    }
}

pub fn stop_and_save() {
    let live = LIVE.with(|s| s.take());
    let Some(live) = live else {
        serial_println!("record> not recording");
        return;
    };
    update_status();
    if live.samples.is_empty() {
        serial_println!("record> cancelled (no frames)");
        return;
    }
    let n_saved = live.samples.len();
    let out_w = live.out_w;
    let out_h = live.out_h;
    let bytes = match mp4_mux::mux_avc(
        out_w,
        out_h,
        1000,
        &live.encoder.sps_nal,
        &live.encoder.pps_nal,
        &live.samples,
    ) {
        Ok(b) => b,
        Err(e) => {
            serial_println!("record> mux failed: {e}");
            return;
        }
    };
    let dest = live
        .dest
        .as_deref()
        .map(|d| crate::shell::resolve_path(&screencast::normalize_dest(d)))
        .unwrap_or_else(|| screencast::default_path(crate::arch::now_ms()));

    if crate::synapse::fs::exists(&dest)
        && !crate::modal::confirm("Overwrite file?", &alloc::format!("{dest} already exists."))
    {
        serial_println!("record> cancelled (not overwritten)");
        return;
    }
    crate::synapse::fs::write(&dest, &bytes);
    serial_println!(
        "record> {}",
        screencast::saved_line(&dest, out_w, out_h, n_saved, bytes.len())
    );
    match crate::video::probe(&bytes) {
        Ok(info) => serial_println!(
            "record> probe ok: {} {}x{}, {} frame(s)",
            info.codec,
            info.width,
            info.height,
            info.frame_count
        ),
        Err(e) => serial_println!("record> warning: probe failed: {e}"),
    }
}
