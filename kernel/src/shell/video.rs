//! video
//!
//! The **video player** subsystem carved out of the former 16k-line
//! `shell/mod.rs` monolith: the `VideoPlayer` static, stream decode/seek,
//! the SMP decode-ahead worker (`vjob`), transport/media-key controls and
//! the on-screen HUD. Moved verbatim; `use super::*` keeps the parent's
//! statics visible, and the parent re-imports this module's items with
//! `pub(crate) use video::*`.

use super::*;

/// Surface id the video player presents frames on (== framebuffer::VIDEO_SURFACE).
#[cfg(not(feature = "server"))]
pub(super) const VIDEO_SURFACE: u32 = u32::MAX - 1;

/// A background video player: decoded keyframes plus a playback clock. Frames
/// are advanced by presentation timestamp from `ui_tick` (`pump_video`), so it
/// keeps playing/advancing across tab switches like the audio player. Baseline
/// H.264 decodes every frame (I keyframes + P inter frames), so playback is
/// full-motion, not keyframe-only.
#[cfg(not(feature = "server"))]
pub(super) struct VideoPlayer {
    dec: crate::video::StreamDecoder,
    frame_count: usize,
    idx: usize,
    playing: bool,
    base_ms: u64,     // wall-clock at which pts 0 plays
    paused_at: u64,   // playback-time when paused
    total_ms: u64,
    name: String,
    finished_announced: bool,
    muted: bool,
    has_audio: bool,
    /// Decoded mono S16 PCM for the video's audio track (option B: owned by
    /// the video player so closing the tab stops audio without touching the
    /// standalone audio tab).
    audio_pcm: Option<alloc::vec::Vec<i16>>,
    audio_rate: u32,
    /// Next PCM sample index to queue (advanced by `pump_video` audio path).
    audio_at: usize,
    /// FPS meter: frames presented in the current 1 s window + EMA display value.
    fps_window_start_ms: u64,
    fps_window_frames: u32,
    fps_display: u32,
    /// Wall-clock of last successful present (for instant ms/frame → fps).
    last_present_ms: u64,
    /// A decode-ahead job is running on an SMP worker: `dec` is on loan to
    /// that core and MUST NOT be touched until [`video_job_collect`] returns
    /// it (reading the immutable sample table — `pts_ms`/`frame_count` — is
    /// fine). See the SAFETY notes at `vjob`.
    pending_job: bool,
    /// A frame decoded ahead of its pts, held until due (`dec.cur` is it).
    ahead: Option<usize>,
}
#[cfg(not(feature = "server"))]
pub(super) static VIDEO: crate::mm::Locked<Option<VideoPlayer>> = crate::mm::Locked::new(None);

/// `/open <path>.mp4|.mov|.m3u8` or `http(s)://…` — demux + decode and play in
/// a video action-pane tab. Non-blocking: `pump_video` advances frames from the
/// idle tick. `/close` or Ctrl+C stops it.
#[cfg(not(feature = "server"))]
pub(super) fn play_video(path: &str) -> Result<(), alloc::string::String> {
    let path = path.trim();
    if crate::browser::url::is_http_url(path) {
        return play_video_url(path);
    }
    let Some(bytes) = read_mounted(path).or_else(|| crate::synapse::fs::read(path)) else {
        serial_println!("open> {} not found under any mount or in the store (see /mounts)", path);
        return Err(alloc::format!("{path} not found under any mount or in the store"));
    };
    if crate::video::hls::looks_like_playlist(&bytes)
        || path_ends_with_ci(path, ".m3u8")
        || path_ends_with_ci(path, ".m3u")
    {
        return play_hls(path, &bytes);
    }
    play_video_bytes(path, bytes)
}

fn path_ends_with_ci(path: &str, ext: &str) -> bool {
    path.len() >= ext.len()
        && path
            .as_bytes()
            .iter()
            .rev()
            .zip(ext.as_bytes().iter().rev())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Fetch an `http(s)` URL (playlist or progressive file) and play it.
#[cfg(not(feature = "server"))]
pub(super) fn play_video_url(url: &str) -> Result<(), alloc::string::String> {
    serial_println!("open> fetching {url}…");
    crate::shell::upkeep();
    if crate::shell::poll_interrupt() {
        return Err(alloc::string::String::from("cancelled"));
    }
    let got = crate::net::http::get_follow(url, 60_000).map_err(|e| e.to_string())?;
    if got.response.status >= 400 {
        return Err(alloc::format!("HTTP {} for {url}", got.response.status));
    }
    let bytes = got.response.body;
    // The **final** URL after redirects, not the one typed: a playlist's segment
    // URIs are relative to where the playlist actually came from, and CDNs
    // redirect these constantly. Resolving against the original URL points every
    // segment at a host that never served the playlist.
    let base = got.final_url;
    if crate::video::hls::looks_like_playlist(&bytes)
        || path_ends_with_ci(&base, ".m3u8")
        || path_ends_with_ci(&base, ".m3u")
    {
        return play_hls(&base, &bytes);
    }
    play_video_bytes(&base, bytes)
}

/// HLS VOD over the default transport: HTTP for a URL, the store/mounts for a
/// local playlist.
#[cfg(not(feature = "server"))]
fn play_hls(base: &str, playlist_bytes: &[u8]) -> Result<(), alloc::string::String> {
    play_hls_with(base, playlist_bytes, &mut default_fetch)
}

/// Fetch one HLS resource: an absolute URL over HTTP, anything else off a mount
/// or the store (a playlist saved by `/http -O` names its segments relatively).
#[cfg(not(feature = "server"))]
fn default_fetch(url: &str) -> Result<alloc::vec::Vec<u8>, alloc::string::String> {
    if crate::browser::url::is_http_url(url) {
        let got = crate::net::http::get_follow(url, 60_000).map_err(|e| e.to_string())?;
        if got.response.status >= 400 {
            return Err(alloc::format!("HTTP {} for {url}", got.response.status));
        }
        return Ok(got.response.body);
    }
    read_mounted(url)
        .or_else(|| crate::synapse::fs::read(url))
        .ok_or_else(|| alloc::format!("segment not found: {url}"))
}

/// HLS VOD: parse playlist, download segments, open the player.
///
/// The transport is injected because **who is loading matters**: a `<video>`
/// clicked in a page must fetch its playlist *and every segment* through the
/// browser's own loader, which owns the memory cache, redirect handling,
/// referrer and instrumentation. Fetching the playlist through the browser and
/// then its segments through a bare HTTP client would be a per-request policy
/// split invisible from the outside.
///
/// UI pumping and Ctrl+C are added **here**, around whichever transport is
/// supplied, so no caller can forget them (the standing rule: a long loop pumps
/// `upkeep()` and answers Ctrl+C).
#[cfg(not(feature = "server"))]
pub(super) fn play_hls_with(
    base: &str,
    playlist_bytes: &[u8],
    fetch: &mut dyn FnMut(&str) -> Result<alloc::vec::Vec<u8>, alloc::string::String>,
) -> Result<(), alloc::string::String> {
    let text = core::str::from_utf8(playlist_bytes)
        .map_err(|_| alloc::string::String::from("hls: playlist is not UTF-8"))?;
    serial_println!("open> HLS playlist {base}");
    match crate::video::probe(playlist_bytes) {
        Ok(info) => {
            serial_println!(
                "open>   {} — {}  {}:{:02}",
                info.container,
                info.codec,
                info.duration_ms / 60000,
                (info.duration_ms % 60000) / 1000
            );
            if !info.decodable {
                serial_println!("open>   cannot decode: {}", info.unsupported_reason);
                return Err(alloc::string::String::from(info.unsupported_reason));
            }
        }
        Err(e) => return Err(alloc::string::String::from(e)),
    }

    let mut pumped = |url: &str| -> Result<alloc::vec::Vec<u8>, alloc::string::String> {
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            return Err(alloc::string::String::from("cancelled"));
        }
        fetch(url)
    };

    let (vod, _) = crate::video::hls::resolve_vod(text, base, &mut pumped)?;
    serial_println!(
        "open>   {} segment(s), ~{} ms — downloading…",
        vod.segments.len(),
        vod.duration_ms
    );
    let loaded = crate::video::hls::load_vod(&vod, &mut pumped, |i, n| {
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            return Err(alloc::string::String::from("cancelled"));
        }
        if i == 0 || i + 1 == n || (i + 1) % 4 == 0 {
            serial_println!("open>   segment {}/{}", i + 1, n);
        }
        Ok(())
    })?;
    serial_println!(
        "open>   demuxed {} sample(s) ({})",
        loaded.samples.len(),
        loaded.container
    );
    let t0 = crate::arch::now_ms();
    match crate::video::StreamDecoder::open_hls(loaded) {
        Ok(mut dec) => {
            let frame_count = dec.frame_count();
            let total_ms = dec.duration_ms;
            dec.seek_decode(0);
            serial_println!(
                "open>   {}x{}  {} frame(s)  decoder={}  ready in {} ms (HLS) — Ctrl+Tab focus, space=pause",
                dec.src_w,
                dec.src_h,
                frame_count,
                dec.backend,
                crate::arch::now_ms().saturating_sub(t0)
            );
            install_player(base, dec, frame_count, total_ms, false, None, 0)
        }
        Err(e) => {
            serial_println!("open> HLS decode failed: {}", e);
            Err(alloc::string::String::from(e))
        }
    }
}

/// Start the video player from already-loaded bytes (browser `<video>` click).
#[cfg(not(feature = "server"))]
pub(super) fn play_video_bytes(name: &str, bytes: alloc::vec::Vec<u8>) -> Result<(), alloc::string::String> {
    if crate::video::hls::looks_like_playlist(&bytes) {
        return play_hls(name, &bytes);
    }
    let t0 = crate::arch::now_ms();
    // Probe first so we can report clearly and handle unsupported streams.
    match crate::video::probe(&bytes) {
        Ok(info) => {
            serial_println!("open> {} — {} {}x{} {} frames {}:{:02}", name, info.codec, info.width, info.height, info.frame_count, info.duration_ms / 60000, info.duration_ms % 60000 / 1000);
            if !info.decodable {
                serial_println!("open>   cannot decode: {}", info.unsupported_reason);
                return Err(alloc::string::String::from(info.unsupported_reason));
            }
        }
        Err(e) => {
            serial_println!("open> cannot open {}: {}", name, e);
            return Err(alloc::string::String::from(e));
        }
    }
    // Demux/describe audio; if AAC-LC, decode PCM now so pump_video can sync it.
    let (has_audio, audio_pcm, audio_rate) = match crate::video::audio_info(&bytes) {
        Some(a) if a.decodable => {
            serial_println!(
                "open>   audio: {} {} Hz {}ch — decoding…",
                a.codec,
                a.sample_rate,
                a.channels
            );
            match crate::video::decode_audio(&bytes) {
                Ok(audio) => {
                    serial_println!(
                        "open>   audio ready: {}:{:02} mono @ {} Hz ({} KiB PCM)",
                        audio.duration_ms() / 60000,
                        (audio.duration_ms() % 60000) / 1000,
                        audio.rate,
                        audio.pcm.len() * 2 / 1024
                    );
                    (true, Some(audio.pcm), audio.rate)
                }
                Err(e) => {
                    serial_println!("open>   audio decode failed ({}) — video plays silently", e);
                    (true, None, 0)
                }
            }
        }
        Some(a) => {
            serial_println!(
                "open>   audio: {} {} Hz {}ch (unsupported profile — video plays silently)",
                a.codec,
                a.sample_rate,
                a.channels
            );
            (true, None, 0)
        }
        None => (false, None, 0),
    };
    match crate::video::StreamDecoder::open(bytes) {
        Ok(mut dec) => {
            let frame_count = dec.frame_count();
            let total_ms = dec.duration_ms;
            // Stream like VLC: demux + first frame only — no full-clip RGB cache.
            dec.seek_decode(0);
            serial_println!(
                "open>   {}x{}  {} frame(s)  decoder={}  ready in {} ms (streaming) — Ctrl+Tab cycles focus, space=pause",
                dec.src_w,
                dec.src_h,
                frame_count,
                dec.backend,
                crate::arch::now_ms().saturating_sub(t0)
            );
            install_player(name, dec, frame_count, total_ms, has_audio, audio_pcm, audio_rate)
        }
        Err(e) => {
            serial_println!("open> decode failed: {}", e);
            Err(alloc::string::String::from(e))
        }
    }
}

#[cfg(not(feature = "server"))]
fn install_player(
    name: &str,
    dec: crate::video::StreamDecoder,
    frame_count: usize,
    total_ms: u64,
    has_audio: bool,
    audio_pcm: Option<alloc::vec::Vec<i16>>,
    audio_rate: u32,
) -> Result<(), alloc::string::String> {
    let name = name.rsplit('/').next().unwrap_or(name).to_string();
    let now = crate::arch::now_ms();
    VIDEO.with(|v| {
        *v = Some(VideoPlayer {
            dec,
            frame_count,
            idx: 0,
            playing: true,
            base_ms: now,
            paused_at: 0,
            total_ms,
            name,
            finished_announced: false,
            muted: false,
            has_audio,
            audio_pcm,
            audio_rate,
            audio_at: 0,
            fps_window_start_ms: now,
            fps_window_frames: 0,
            fps_display: 0,
            last_present_ms: 0,
            pending_job: false,
            ahead: None,
        })
    });
    #[cfg(not(test))]
    {
        crate::framebuffer::set_right(crate::framebuffer::RightMode::Surface(VIDEO_SURFACE));
        present_video_frame();
    }
    Ok(())
}

/// Present the current video frame into the video tab (no-op if not active).
/// Updates the rolling FPS meter (frames presented per wall-clock second).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn present_video_frame() {
    let now = crate::arch::now_ms();
    VIDEO.with(|v| {
        if let Some(p) = v.as_mut() {
            if p.pending_job {
                // `dec` (including `cur`) is on loan to the decode worker;
                // the pump presents this frame when it collects the job.
                return;
            }
            if let Some(f) = p.dec.cur_frame() {
                // Reserve the bottom strip for the HUD so the per-frame blit
                // never repaints under it (no flicker); the HUD lives there.
                let hud = crate::framebuffer::video_hud_height();
                crate::framebuffer::present_surface_reserve(VIDEO_SURFACE, f.w, f.h, &f.pixels, hud);
                // FPS: count presents in 1 s windows; show last completed window.
                p.fps_window_frames = p.fps_window_frames.saturating_add(1);
                p.last_present_ms = now;
                let elapsed = now.saturating_sub(p.fps_window_start_ms);
                if elapsed >= 1000 {
                    // frames in this window → fps (scale if window > 1 s)
                    let fps = if elapsed > 0 {
                        (p.fps_window_frames as u64 * 1000 / elapsed) as u32
                    } else {
                        0
                    };
                    p.fps_display = fps;
                    p.fps_window_start_ms = now;
                    p.fps_window_frames = 0;
                }
            }
        }
    });
    present_video_status();
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn present_video_frame() {}

/// Whether a video is loaded.
#[cfg(not(feature = "server"))]
pub(super) fn video_loaded() -> bool {
    VIDEO.with(|v| v.is_some())
}

/// Stop + unload the video (Ctrl+C / closing the video tab).
#[cfg(not(feature = "server"))]
pub(super) fn stop_video() {
    let stopped = VIDEO.with(|v| {
        // Reclaim `dec` from any decode-ahead worker before dropping it.
        #[cfg(not(test))]
        if let Some(p) = v.as_mut() {
            video_job_join(p);
        }
        v.take().is_some()
    });
    if stopped {
        serial_println!("\ropen> video stopped");
    }
}
#[cfg(feature = "server")]
pub(super) fn stop_video() {}

/// Re-entrancy guard: `upkeep` → `pump_video` must never nest (VIDEO lock).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) static PUMPING_VIDEO: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Advance the video by presentation time; the idle-tick heartbeat.
/// Also queues ~200 ms audio chunks from the video's own PCM (when present),
/// gated on play state and device drain — never steals the standalone audio tab.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pump_video() {
    use core::sync::atomic::Ordering;
    // Bail if already pumping (e.g. a nested upkeep from a mistaken yield
    // inside decode). Nested VIDEO.with would spin forever.
    if PUMPING_VIDEO.swap(true, Ordering::AcqRel) {
        return;
    }
    let result = pump_video_inner();
    PUMPING_VIDEO.store(false, Ordering::Release);
    let _ = result;
}

#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pump_video_inner() {
    use core::sync::atomic::Ordering;
    let now = crate::arch::now_ms();
    // Audio chunk (copied out so VIDEO lock isn't held across sound::play).
    let audio_chunk = VIDEO.with(|v| {
        let p = v.as_mut()?;
        if !p.playing {
            return None;
        }
        let pcm = p.audio_pcm.as_ref()?;
        if p.audio_rate == 0 || p.audio_at >= pcm.len() {
            return None;
        }
        if crate::sound::playing() {
            return None; // still draining previous chunk
        }
        // Snap cursor to the current video pts so seek/pause recover cleanly.
        let t = now.saturating_sub(p.base_ms);
        let want = ((t as u128) * p.audio_rate as u128 / 1000) as usize;
        // Only jump forward/back if we drifted > ~50 ms (avoid tiny jitter).
        let slop = (p.audio_rate as usize / 20).max(1);
        if want > p.audio_at + slop || p.audio_at > want + slop {
            p.audio_at = want.min(pcm.len());
        }
        if p.audio_at >= pcm.len() {
            return None;
        }
        let chunk = (p.audio_rate as usize / 5).max(256); // ~200 ms
        let end = (p.audio_at + chunk).min(pcm.len());
        let slice = pcm[p.audio_at..end].to_vec();
        p.audio_at = end;
        Some((slice, p.audio_rate))
    });
    if let Some((slice, rate)) = audio_chunk {
        let _ = crate::sound::play(&slice, rate);
    }

    // Phase A: collect a finished decode-ahead job and decide whether the
    // held frame is due. NO job submission here — the blit below reads
    // `dec.cur`, so `dec` must not go back on loan until after it runs
    // (submitting first made `present_video_frame`'s loan-guard skip every
    // blit: audio + counters advanced, the picture froze on frame one).
    let present = VIDEO.with(|v| {
        let Some(p) = v.as_mut() else { return false };
        // Collect a finished decode-ahead job. The decoded frame is *held*
        // (`ahead`) until its pts is due — never shown early.
        if p.pending_job {
            match video_job_collect(p) {
                Some((goal, changed)) => {
                    if changed {
                        p.ahead = Some(goal);
                    } else {
                        p.idx = goal; // decode failed/no-op: skip past, don't loop
                    }
                }
                None => {} // still decoding on the worker
            }
        }
        if !p.playing || p.frame_count == 0 {
            return false;
        }
        let t = now.saturating_sub(p.base_ms);
        // Present the held frame the moment it is due (already decoded —
        // this is the cheap path that keeps presentation at clip rate).
        let mut presented = false;
        if let Some(a) = p.ahead {
            if a <= p.idx {
                p.ahead = None; // stale (a seek moved us past it)
            } else if p.dec.pts_ms(a) <= t {
                p.ahead = None;
                p.idx = a;
                presented = true;
                let pts = p.dec.pts_ms(a);
                if t > pts.saturating_add(100) {
                    // Behind the wall clock — snap media time forward (drop
                    // backlog), never snap backward to a previous keyframe.
                    p.base_ms = now.saturating_sub(pts);
                }
                // Content signature for the perf line: proves the *picture*
                // advances, not just the counters (a present-ordering bug once
                // froze the image while every metric kept ticking).
                if let Some(f) = p.dec.cur_frame() {
                    let mut sig = 0u32;
                    let step = (f.pixels.len() / 16).max(1);
                    for px in f.pixels.iter().step_by(step) {
                        sig = sig.wrapping_mul(31).wrapping_add(*px);
                    }
                    VIDEO_SIG.store(((p.idx as u64) << 32) | sig as u64, Ordering::Relaxed);
                }
            }
        }
        presented
    });
    if present {
        let t0 = crate::arch::now_ms();
        present_video_frame();
        VIDEO_PRESENT_MS.fetch_add(crate::arch::now_ms().saturating_sub(t0), Ordering::Relaxed);
        let n = VIDEO_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 32 == 0 {
            let d = VIDEO_DECODE_MS.swap(0, Ordering::Relaxed);
            let pr = VIDEO_PRESENT_MS.swap(0, Ordering::Relaxed);
            let sig = VIDEO_SIG.load(Ordering::Relaxed);
            crate::ktrace::log_fmt(format_args!(
                "video: perf: last 32 frames: decode {} ms ({}/frame), present {} ms ({}/frame), at frame {} sig {:08x}",
                d,
                d / 32,
                pr,
                pr / 32,
                sig >> 32,
                sig as u32
            ));
        }
    }

    // Phase B: with the blit done, keep the decode pipeline fed — pick the
    // next goal and put `dec` back on loan. Decoding runs one frame AHEAD of
    // its due time, so the ~30 ms 1080p decode overlaps the current frame's
    // display.
    let finished = VIDEO.with(|v| {
        let Some(p) = v.as_mut() else { return false };
        if p.pending_job || p.ahead.is_some() || !p.playing || p.frame_count == 0 {
            return false;
        }
        let t = now.saturating_sub(p.base_ms);
        let mut target = p.idx;
        while target + 1 < p.frame_count && p.dec.pts_ms(target + 1) <= t {
            target += 1;
        }
        // End of clip: stop on the last frame.
        if t >= p.total_ms && target + 1 >= p.frame_count {
            p.playing = false;
            // **Record where we stopped.** Without this `paused_at` keeps whatever the
            // last manual pause left in it (0 if there never was one), so pressing play
            // afterwards anchored the clock to a stale time while `idx` sat on the final
            // frame — and since the goal is forward-only, nothing was ever decoded again.
            p.paused_at = p.total_ms;
            if !p.finished_announced {
                p.finished_announced = true;
                p.idx = target;
                p.dec.seek_decode(target);
                return true;
            }
            return false;
        }
        // Forward only; when behind, catch up in SMALL steps (the hurry
        // flag frame-drops non-reference backlog, and the clock re-anchor
        // absorbs the rest). Small jobs = frequent presents: one giant
        // jump would decode every backlog reference in one job and starve
        // presentation (4K went from ~8 to ~3 fps that way).
        let goal = if target > p.idx { target.min(p.idx + 2) } else { p.idx + 1 };
        let goal = goal.min(p.frame_count.saturating_sub(1));
        if goal > p.idx {
            let hurry = t > p.dec.pts_ms(goal).saturating_add(100);
            // Prefer an SMP worker (BSP keeps pumping UI/audio); fall back
            // to synchronous decode-and-hold when none is available.
            if video_job_submit(&mut p.dec, goal, hurry) {
                p.pending_job = true;
            } else {
                let t0 = crate::arch::now_ms();
                let changed = p.dec.seek_decode_hurry(goal, hurry);
                VIDEO_DECODE_MS.fetch_add(crate::arch::now_ms().saturating_sub(t0), Ordering::Relaxed);
                if changed {
                    p.ahead = Some(goal);
                } else {
                    p.idx = goal;
                }
            }
        }
        false
    });
    if finished {
        let t0 = crate::arch::now_ms();
        present_video_frame();
        VIDEO_PRESENT_MS.fetch_add(crate::arch::now_ms().saturating_sub(t0), Ordering::Relaxed);
        // Stage accounting, one ktrace line per 32 presented frames: where the
        // per-frame budget goes (decode vs present), per the measure-first rule.
        let n = VIDEO_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 32 == 0 {
            let d = VIDEO_DECODE_MS.swap(0, Ordering::Relaxed);
            let pr = VIDEO_PRESENT_MS.swap(0, Ordering::Relaxed);
            crate::ktrace::log_fmt(format_args!(
                "video: perf: last 32 frames: decode {} ms ({}/frame), present {} ms ({}/frame)",
                d,
                d / 32,
                pr,
                pr / 32
            ));
        }
    }
}

/// Per-stage wall-time accumulators for the `video: perf:` ktrace (32-frame
/// windows; see `pump_video_inner`).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) static VIDEO_DECODE_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) static VIDEO_PRESENT_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) static VIDEO_FRAMES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// `(presented frame idx << 32) | pixel signature` — the perf line's proof
/// that the displayed content is advancing.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) static VIDEO_SIG: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Decode-ahead job plumbing: the pump loans `dec` to an SMP worker
/// (`smp::async_submit`) so the ~30 ms 1080p decode overlaps UI/audio work on
/// the BSP instead of blocking it. Exclusive access is handed over whole: the
/// BSP sets `pending_job` and must not touch `dec` (beyond the immutable
/// sample table) until the job completes; every other `dec` toucher goes
/// through [`video_job_join`] first.
#[cfg(all(target_arch = "aarch64", not(feature = "server"), not(test)))]
mod vjob {
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    pub static DEC: AtomicUsize = AtomicUsize::new(0);
    pub static GOAL: AtomicUsize = AtomicUsize::new(0);
    pub static HURRY: AtomicBool = AtomicBool::new(false);
    pub static CHANGED: AtomicBool = AtomicBool::new(false);
    /// Worker-measured decode wall time (ms) for the stage accounting.
    pub static MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

    /// Worker-side entry: decode to `GOAL` on the loaned decoder.
    ///
    /// # Safety
    /// Called only via `smp::async_submit`; the BSP published DEC/GOAL/HURRY
    /// before submitting (release) and reads CHANGED only after observing
    /// completion (acquire), so this core has exclusive access to the decoder
    /// for the whole run. `Rc` inside the decoder is safe under whole-object
    /// single-core handoff.
    pub unsafe fn run(_ctx: *mut u8) {
        let dec = DEC.load(Ordering::Acquire) as *mut crate::video::StreamDecoder;
        if dec.is_null() {
            return;
        }
        let goal = GOAL.load(Ordering::Relaxed);
        let hurry = HURRY.load(Ordering::Relaxed);
        let t0 = crate::arch::now_ms();
        // SAFETY: exclusive loan per above.
        let changed = unsafe { (*dec).seek_decode_hurry(goal, hurry) };
        MS.store(crate::arch::now_ms().saturating_sub(t0), Ordering::Relaxed);
        CHANGED.store(changed, Ordering::Release);
    }
}

/// Try to start a decode-ahead job for `goal`. `false` → caller decodes
/// synchronously (x86, no workers, degraded fleet, or a job already active).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_job_submit(dec: &mut crate::video::StreamDecoder, goal: usize, hurry: bool) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        vjob::DEC.store(dec as *mut _ as usize, core::sync::atomic::Ordering::Release);
        vjob::GOAL.store(goal, core::sync::atomic::Ordering::Relaxed);
        vjob::HURRY.store(hurry, core::sync::atomic::Ordering::Relaxed);
        vjob::CHANGED.store(false, core::sync::atomic::Ordering::Relaxed);
        // SAFETY: dec stays in the VIDEO player (stable address) and the BSP
        // honours the loan via `pending_job` until async_take_done.
        unsafe { crate::arch::aarch64::smp::async_submit(vjob::run, core::ptr::null_mut()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (dec, goal, hurry);
        false
    }
}

/// Poll a pending decode-ahead job; on completion return `Some((goal,
/// changed))` and return ownership of the decoder to the BSP. The caller
/// decides whether the frame is held (`ahead`) or skipped.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_job_collect(p: &mut VideoPlayer) -> Option<(usize, bool)> {
    if !p.pending_job {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::sync::atomic::Ordering;
        if crate::arch::aarch64::smp::async_take_done() {
            p.pending_job = false;
            VIDEO_DECODE_MS.fetch_add(vjob::MS.load(Ordering::Relaxed), Ordering::Relaxed);
            return Some((vjob::GOAL.load(Ordering::Relaxed), vjob::CHANGED.load(Ordering::Acquire)));
        }
        None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        p.pending_job = false;
        Some((p.idx, false))
    }
}

/// Block (bounded by one frame's decode) until no job is on loan — required
/// before any mutable `dec` access outside the pump (seek, restart, close).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_job_join(p: &mut VideoPlayer) {
    while p.pending_job {
        if video_job_collect(p).is_some() {
            break;
        }
        core::hint::spin_loop();
    }
    // Whatever the join was for (seek/restart/close) invalidates a held frame.
    p.ahead = None;
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn pump_video() {}

/// Toggle play/pause on the video (space on the video tab).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_toggle_pause() {
    let now = crate::arch::now_ms();
    VIDEO.with(|v| {
        if let Some(p) = v.as_mut() {
            if p.playing {
                p.paused_at = now.saturating_sub(p.base_ms);
                p.playing = false;
            } else {
                // Never resume the clock behind the picture, and treat play-at-the-end as
                // a replay — see `video::resume_action` for what each case froze.
                let frame_pts = p.dec.pts_ms(p.idx);
                match crate::video::resume_action(p.paused_at, frame_pts, p.total_ms) {
                    crate::video::Resume::Restart => {
                        video_job_join(p); // reclaim `dec` before seeking it
                        p.idx = 0;
                        p.base_ms = now;
                        p.paused_at = 0;
                        p.dec.seek_decode(0);
                        p.audio_at = 0;
                    }
                    crate::video::Resume::At(ms) => {
                        p.paused_at = ms;
                        p.base_ms = now.saturating_sub(ms);
                    }
                }
                p.playing = true;
                p.finished_announced = false;
            }
        }
    });
    present_video_status();
}

/// Shared mute for audio + video tabs (`m`). Uses the global software mute so
/// the next PCM chunk is silence; also mirrors into `VideoPlayer.muted` for the
/// HUD when a video is loaded.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn media_toggle_mute() {
    let m = crate::sound::toggle_mute();
    VIDEO.with(|v| {
        if let Some(p) = v.as_mut() {
            p.muted = m;
        }
    });
    present_video_status();
    repaint_audio();
}

/// Shared volume adjust for audio + video tabs (↑/↓). Steps are percent points.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn media_volume_adjust(delta: i32) {
    let v = crate::sound::volume_adjust(delta);
    // Keep video HUD mute flag in sync if volume-up unmuted.
    VIDEO.with(|vp| {
        if let Some(p) = vp.as_mut() {
            p.muted = crate::sound::muted();
        }
    });
    let _ = v;
    present_video_status();
    repaint_audio();
}

/// Draw the video player's status bar into the surface tab: playback state,
/// position/duration, mute, and the key-shortcut hints (mirrors the audio
/// player's footer). No-op when the video tab isn't the focused surface.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn present_video_status() {
    VIDEO.with(|v| {
        if let Some(p) = v.as_ref() {
            let pos = p.dec.pts_ms(p.idx);
            // Prefer completed 1 s window; if still filling, show instant
            // estimate from last inter-present gap.
            let fps = if p.fps_display > 0 {
                p.fps_display
            } else if p.fps_window_frames > 0 {
                let elapsed = crate::arch::now_ms().saturating_sub(p.fps_window_start_ms).max(1);
                (p.fps_window_frames as u64 * 1000 / elapsed) as u32
            } else {
                0
            };
            crate::framebuffer::draw_video_status(
                &p.name,
                p.playing,
                crate::sound::muted() || p.muted,
                p.has_audio,
                p.idx + 1,
                p.frame_count,
                pos,
                p.total_ms,
                crate::sound::volume(),
                fps,
            );
        }
    });
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn present_video_status() {}

/// Seek the video by whole frames (arrows on the video tab).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_seek(delta: i64) {
    let now = crate::arch::now_ms();
    VIDEO.with(|v| {
        if let Some(p) = v.as_mut() {
            video_job_join(p); // reclaim `dec` from any decode-ahead worker
            let n = p.frame_count as i64;
            let ni = (p.idx as i64 + delta).clamp(0, n - 1) as usize;
            p.idx = ni;
            // Re-anchor the clock to the sought frame's pts.
            let pts = p.dec.pts_ms(ni);
            p.base_ms = now.saturating_sub(pts);
            p.paused_at = pts;
            p.finished_announced = false;
            p.dec.seek_decode(ni);
            // Keep audio cursor in lockstep with video pts.
            if p.audio_rate > 0 {
                p.audio_at = ((pts as u128) * p.audio_rate as u128 / 1000) as usize;
                if let Some(pcm) = p.audio_pcm.as_ref() {
                    p.audio_at = p.audio_at.min(pcm.len());
                }
            }
        }
    });
    present_video_frame();
}

/// Restart the video from the first frame (0 / Home).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn video_restart() {
    let now = crate::arch::now_ms();
    VIDEO.with(|v| {
        if let Some(p) = v.as_mut() {
            video_job_join(p); // reclaim `dec` from any decode-ahead worker
            p.idx = 0;
            p.base_ms = now;
            p.paused_at = 0;
            p.playing = true;
            p.finished_announced = false;
            p.dec.seek_decode(0);
            p.audio_at = 0;
        }
    });
    present_video_frame();
}
