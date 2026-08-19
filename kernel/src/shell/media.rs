//! media
//!
//! The **image viewer + audio player** tabs carved out of the former
//! 16k-line `shell/mod.rs` monolith: the `ImageTab` and `AudioPlayer`
//! statics, decode/present, transport controls and repaint paths. Moved
//! verbatim; `use super::*` keeps the parent's statics visible, and the
//! parent re-imports this module's items with `pub(crate) use media::*`.

use super::*;

/// Surface id the `/open` image viewer presents on (labelled "image" in the
/// tab bar; distinct from any agent-allocated `synapse::ui` surface).
#[cfg(not(feature = "server"))]
pub(super) const VIEWER_SURFACE: u32 = u32::MAX; // == framebuffer::IMAGE_SURFACE (labelled "image")

/// The retained source image plus interactive view state (zoom / rotation /
/// pan), so the image tab can be zoomed, rotated, and panned and repaints when
/// you switch back to it (surfaces aren't otherwise backed). The source is
/// capped to `IMAGE_MAX_PX` at load so a huge photo can't exhaust the heap
/// while still holding enough detail for a few zoom steps.
#[cfg(not(feature = "server"))]
pub(super) struct ImageTab {
    src: crate::image::Image,
    zoom: u32, // percent of fit-to-pane; 100 = fit
    rot: u32,  // 90° quadrants clockwise
    pan_x: i64,
    pan_y: i64,
}
#[cfg(not(feature = "server"))]
pub(super) static IMAGE: crate::mm::Locked<Option<ImageTab>> = crate::mm::Locked::new(None);
/// Cap the retained source image (≈16 MiB of u32) — bounds heap use for a huge
/// photo; box-downscaled once at load, aspect preserved by halving.
#[cfg(not(feature = "server"))]
pub(super) const IMAGE_MAX_PX: usize = 4_000_000;

/// `/open <path>.png|.jpg` — decode and show an image in an action-pane tab.
/// Reads from a mounted volume (`/mnt/...`) or the Synapse store; the decoded
/// image is box-downscaled to the pane, then integer-upscaled/letterboxed by
/// the compositor, and retained so switching back to the tab repaints it.
#[cfg(not(feature = "server"))]
pub(super) fn view_image(path: &str) {
    let Some(bytes) = read_mounted(path).or_else(|| crate::synapse::fs::read(path)) else {
        serial_println!("open> {} not found under any mount or in the store (see /mounts)", path);
        return;
    };
    let t0 = crate::arch::now_ms();
    // **Decoded in ring 3.** The bytes are attacker-supplied and the decoder needs no authority
    // to turn them into pixels, so it runs as a tenant that holds nothing and can be discarded —
    // see `/decoder`, which switches back to the in-kernel path for an A/B.
    match crate::synapse::tenant::decode_image_for_view(&bytes) {
        Ok(img) => {
            let (iw, ih) = (img.w, img.h);
            #[cfg(not(test))]
            {
                // Cap the retained source (halving preserves aspect) so a huge
                // photo can't exhaust the heap.
                let mut src = img;
                let (mut nw, mut nh) = (src.w, src.h);
                while nw * nh > IMAGE_MAX_PX {
                    nw = nw.div_ceil(2);
                    nh = nh.div_ceil(2);
                }
                if (nw, nh) != (src.w, src.h) {
                    src = crate::image::resize(&src, nw, nh);
                }
                IMAGE.with(|s| *s = Some(ImageTab { src, zoom: 100, rot: 0, pan_x: 0, pan_y: 0 }));
                // Open the tab and render the fitted view. Controls activate
                // once the action pane is focused (Ctrl+Tab / click) — the same
                // gating as pane scroll, so typing at the prompt is never eaten.
                crate::framebuffer::set_right(crate::framebuffer::RightMode::Surface(VIEWER_SURFACE));
                render_image();
            }
            #[cfg(test)]
            drop(img);
            serial_println!(
                "open> {} — {}x{} px, {} KiB, decoded in {} ms  (Ctrl+Tab to focus pane, then +/- zoom, r/l rotate, arrows pan, 0 reset; Ctrl+Tab again returns to shell; /close hides)",
                path,
                iw,
                ih,
                bytes.len() / 1024,
                crate::arch::now_ms().saturating_sub(t0)
            );
        }
        Err(e) => serial_println!("open> cannot decode {}: {}", path, e),
    }
}

/// Render the retained image at its current zoom/rotation/pan into the tab.
/// Also the repaint-on-switch path (surfaces aren't otherwise backed).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn render_image() {
    let (pw, ph) = crate::framebuffer::surface_dims_px(VIEWER_SURFACE).unwrap_or((640, 480));
    let bg = crate::framebuffer::pane_bg().unwrap_or(0);
    IMAGE.with(|s| {
        if let Some(t) = s.as_ref() {
            let v = crate::image::render_view(&t.src, pw as usize, ph as usize, t.zoom, t.rot, t.pan_x, t.pan_y, bg);
            crate::framebuffer::present_surface(VIEWER_SURFACE, v.w, v.h, &v.pixels);
        }
    });
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn render_image() {}
#[cfg(not(test))]
pub(super) fn repaint_image() {
    render_image();
}

/// Apply an interactive image-viewer command (zoom/rotate/pan/reset) and
/// re-render the tab. No-op unless the image tab is loaded.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn image_cmd(c: u8) {
    let (pw, ph) = crate::framebuffer::surface_dims_px(VIEWER_SURFACE).unwrap_or((640, 480));
    let (pw, ph) = (pw as i64, ph as i64);
    IMAGE.with(|s| {
        let Some(t) = s.as_mut() else { return };
        // Pan step scales with the pane so it feels the same at any resolution.
        let step = (pw / 6).max(16);
        match c {
            b'+' | b'=' => t.zoom = (t.zoom + 25).min(800),
            b'-' | b'_' => t.zoom = t.zoom.saturating_sub(25).max(10),
            b'r' | b'R' => {
                t.rot = (t.rot + 1) % 4;
                t.pan_x = 0;
                t.pan_y = 0;
            }
            b'l' | b'L' => {
                t.rot = (t.rot + 3) % 4;
                t.pan_x = 0;
                t.pan_y = 0;
            }
            b'0' => {
                t.zoom = 100;
                t.rot = 0;
                t.pan_x = 0;
                t.pan_y = 0;
            }
            // Arrow bytes forwarded as A/B/C/D: pan the image (only meaningful
            // once zoomed past the pane).
            b'A' => t.pan_y += step,
            b'B' => t.pan_y -= step,
            b'C' => t.pan_x += step,
            b'D' => t.pan_x -= step,
            _ => return,
        }
        // Clamp pan so the image can't be dragged entirely off the pane.
        let (ow, oh) = if t.rot % 2 == 1 { (t.src.h, t.src.w) } else { (t.src.w, t.src.h) };
        let (fw, fh) = crate::image::fit(ow, oh, pw as usize, ph as usize);
        let dw = (fw as i64 * t.zoom as i64 / 100).max(1);
        let dh = (fh as i64 * t.zoom as i64 / 100).max(1);
        let maxx = (dw - pw).max(0) / 2 + pw / 4;
        let maxy = (dh - ph).max(0) / 2 + ph / 4;
        t.pan_x = t.pan_x.clamp(-maxx, maxx);
        t.pan_y = t.pan_y.clamp(-maxy, maxy);
    });
    render_image();
}
#[cfg(not(all(not(feature = "server"), not(test))))]
#[allow(dead_code)]
pub(super) fn image_cmd(_c: u8) {}

/// A background audio player: the decoded PCM plus a cursor. Lives in a static
/// so playback continues while you switch tabs or run other commands — it is
/// pumped one chunk at a time from `ui_tick` (`pump_audio`), like `/top`
/// refreshes. `done` latches at end-of-track; auto-advance then consults the
/// sibling playlist (same folder) and the repeat / shuffle flags.
#[cfg(not(feature = "server"))]
pub(super) struct AudioPlayer {
    /// Interleaved when `channels > 1`, so this is `frames * channels` long and
    /// **is not a sample count you may divide by `rate`**.
    pcm: alloc::vec::Vec<i16>,
    rate: u32,
    /// 1 = mono, 2 = stereo interleaved.
    channels: u8,
    /// Play cursor, an index into `pcm` — always a multiple of `channels`, so
    /// it never lands mid-frame. A cursor that drifts off a frame boundary
    /// swaps left and right for the rest of the track.
    at: usize,
    /// Full store / mount path, used to rebuild the sibling playlist.
    path: String,
    name: String,
    total_ms: u64,
    done: bool,
    paused: bool,
    finished_announced: bool,
    /// Peak envelope for the whole-track silhouette (`audio::waveform_peaks`).
    peaks: alloc::vec::Vec<u8>,
    /// Live octave-band spectrum, peak-held across refreshes.
    spectrum: alloc::vec::Vec<u8>,
    /// Sibling audio files in the same folder (always contains `path`).
    playlist: alloc::vec::Vec<String>,
    playlist_idx: usize,
    repeat: crate::audio::hud::Repeat,
    shuffle: bool,
}
#[cfg(not(feature = "server"))]
pub(super) static AUDIO: crate::mm::Locked<Option<AudioPlayer>> = crate::mm::Locked::new(None);

/// `/open <path>.wav|.mp3|.aac` — decode (RIFF/WAVE, MPEG Layer III, or ADTS
/// AAC) and play in the background at the file's own sample rate, in an
/// "audio" action-pane tab.
/// Non-blocking: it starts playback and returns; `pump_audio` (idle tick) feeds
/// the device chunk by chunk, so switching tabs, editing, or running other
/// commands never interrupts the track. `/close` (or Ctrl+C at the prompt)
/// stops it.
#[cfg(not(feature = "server"))]
pub(super) fn play_audio(path: &str) {
    let Some(bytes) = read_mounted(path).or_else(|| crate::synapse::fs::read(path)) else {
        serial_println!("open> {} not found under any mount or in the store (see /mounts)", path);
        return;
    };
    let t0 = crate::arch::now_ms();
    let audio = match crate::audio::decode(&bytes) {
        Ok(a) => a,
        Err(e) => {
            serial_println!("open> cannot decode {}: {}", path, e);
            return;
        }
    };
    let total_ms = audio.duration_ms();
    serial_println!(
        "open> playing {} — {}:{:02} at {} Hz ({} KiB, decoded in {} ms)",
        path,
        total_ms / 60000,
        total_ms % 60000 / 1000,
        audio.rate,
        bytes.len() / 1024,
        crate::arch::now_ms().saturating_sub(t0)
    );
    // `ensure_up`, not `is_up`: discovery otherwise ran once at boot, so a USB
    // audio device plugged in afterwards was never adopted and this said "no
    // sound device" for the rest of the session.
    if !crate::sound::ensure_up() {
        serial_println!("open> no sound device — decoded OK but cannot play (see the sound: lines in /ktrace)");
        #[cfg(target_arch = "aarch64")]
        if crate::arch::aarch64::is_apple() {
            serial_println!("open>   built-in speaker did not come up — `/audio dump` then `/audio up` to retry");
        }
        return;
    }
    serial_println!("open>   switch tabs freely, it keeps playing; Ctrl+Tab to focus then space=pause <-/->=seek n/p=next/prev r=repeat s=shuffle up/dn=volume 0=restart m=mute; Ctrl+Tab again returns to shell; Ctrl+C or /close stops");
    let name = path.rsplit('/').next().unwrap_or(path).to_string();
    // The waveform is one envelope for the whole track, so it is built from a
    // mono fold rather than from the interleaved buffer — peaks taken straight
    // off `L R L R` alternate between the channels and read as a rougher,
    // noisier envelope than the audio actually has.
    let peaks = crate::audio::waveform_peaks(
        &crate::audio::to_mono(&audio.pcm, audio.channels),
        crate::audio::WAVEFORM_BINS,
    );
    let playlist = audio_siblings(path);
    let playlist_idx = playlist.iter().position(|p| p == path).unwrap_or(0);
    if playlist.len() > 1 {
        serial_println!("open>   playlist {}/{} in {}", playlist_idx + 1, playlist.len(), crate::audio::hud::parent_dir(path));
    }
    AUDIO.with(|a| {
        *a = Some(AudioPlayer {
            channels: audio.channels,
            pcm: audio.pcm,
            rate: audio.rate,
            at: 0,
            path: path.to_string(),
            name,
            total_ms,
            done: false,
            paused: false,
            finished_announced: false,
            peaks,
            spectrum: alloc::vec![0u8; crate::audio::hud::SPECTRUM_BARS],
            playlist,
            playlist_idx,
            repeat: crate::audio::hud::Repeat::Off,
            shuffle: false,
        })
    });
    #[cfg(not(test))]
    {
        crate::framebuffer::set_right(crate::framebuffer::RightMode::Audio);
        repaint_audio();
        super::update_status();
    }
}

/// Sibling `.wav`/`.mp3`/`.aac`/`.m4a` files in the same folder as `path`.
/// Always contains `path` itself, even when the directory cannot be listed.
#[cfg(not(feature = "server"))]
fn audio_siblings(path: &str) -> alloc::vec::Vec<String> {
    use crate::audio::hud;
    let parent = hud::parent_dir(path);
    let names: alloc::vec::Vec<String> = crate::fs::vfs::readdir(parent)
        .map(|ents| {
            ents.into_iter()
                .filter(|e| !e.is_dir && hud::is_audio_filename(&e.name))
                .map(|e| e.name)
                .collect()
        })
        .unwrap_or_default();
    let refs: alloc::vec::Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    hud::playlist_from_names(path, &refs)
}

/// Whether a track is loaded (playing or paused at end).
#[cfg(not(feature = "server"))]
pub(super) fn audio_loaded() -> bool {
    AUDIO.with(|a| a.is_some())
}

/// Status-bar now-playing chip. Empty when idle so the template swallows the
/// separator (same posture as `${notifications}` / `${recording}`).
pub(crate) fn now_playing_chip() -> String {
    #[cfg(all(not(feature = "server"), not(test)))]
    {
        AUDIO.with(|a| {
            let Some(p) = a.as_ref() else {
                return String::new();
            };
            crate::audio::hud::chip_text(true, !p.done && !p.paused, &p.name)
        })
    }
    #[cfg(not(all(not(feature = "server"), not(test))))]
    {
        String::new()
    }
}

/// Stop + unload the background track (Ctrl+C / closing the audio tab).
#[cfg(not(feature = "server"))]
pub(super) fn stop_audio() {
    let was = AUDIO.with(|a| a.take().is_some());
    if was {
        serial_println!("\ropen> audio stopped");
        #[cfg(not(test))]
        super::update_status();
    }
}
/// Headless build has no `/open` media player; the tab-close path still calls
/// this generically, so provide a no-op.
#[cfg(feature = "server")]
pub(super) fn stop_audio() {}

/// Toggle play/pause on the background track (space key on the audio tab,
/// or a click on the status-bar now-playing chip).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_toggle_pause() {
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            p.paused = !p.paused;
        }
    });
    repaint_audio();
    super::update_status();
}

/// Step to the next / previous sibling. Wraps; shuffle hops.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_step(forward: bool) {
    let next = AUDIO.with(|a| {
        let p = a.as_ref()?;
        Some(crate::audio::hud::user_step(
            p.playlist_idx,
            p.playlist.len(),
            p.shuffle,
            crate::arch::now_ms(),
            forward,
        ))
    });
    if let Some(i) = next {
        audio_play_index(i);
    }
}

/// Cycle Off → All → One → Off.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_cycle_repeat() {
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            p.repeat = p.repeat.cycle();
        }
    });
    repaint_audio();
}

/// Toggle shuffle.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_toggle_shuffle() {
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            p.shuffle = !p.shuffle;
        }
    });
    repaint_audio();
}

/// Decode and swap in playlist slot `idx`, keeping repeat / shuffle / the
/// sibling list. No-op when the slot is already the loaded track (rewinds)
/// or the file cannot be read.
#[cfg(all(not(feature = "server"), not(test)))]
fn audio_play_index(idx: usize) {
    let path = AUDIO.with(|a| a.as_ref().and_then(|p| p.playlist.get(idx).cloned()));
    let Some(path) = path else { return };
    let already = AUDIO.with(|a| a.as_ref().map(|p| p.path == path).unwrap_or(false));
    if already {
        audio_restart();
        return;
    }
    let Some(bytes) = read_mounted(&path).or_else(|| crate::synapse::fs::read(&path)) else {
        serial_println!("open> {} not found", path);
        return;
    };
    let audio = match crate::audio::decode(&bytes) {
        Ok(a) => a,
        Err(e) => {
            serial_println!("open> cannot decode {}: {}", path, e);
            return;
        }
    };
    let total_ms = audio.duration_ms();
    let name = path.rsplit('/').next().unwrap_or(path.as_str()).to_string();
    let peaks = crate::audio::waveform_peaks(
        &crate::audio::to_mono(&audio.pcm, audio.channels),
        crate::audio::WAVEFORM_BINS,
    );
    serial_println!(
        "open> playing {} — {}:{:02} at {} Hz",
        path,
        total_ms / 60000,
        total_ms % 60000 / 1000,
        audio.rate
    );
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            p.channels = audio.channels;
            p.pcm = audio.pcm;
            p.rate = audio.rate;
            p.at = 0;
            p.path = path;
            p.name = name;
            p.total_ms = total_ms;
            p.done = false;
            p.paused = false;
            p.finished_announced = false;
            p.peaks = peaks;
            p.spectrum.fill(0);
            p.playlist_idx = idx;
        }
    });
    repaint_audio();
    super::update_status();
}

#[cfg(not(all(not(feature = "server"), not(test))))]
#[allow(dead_code)]
pub(super) fn audio_step(_forward: bool) {}
#[cfg(not(all(not(feature = "server"), not(test))))]
#[allow(dead_code)]
pub(super) fn audio_cycle_repeat() {}
#[cfg(not(all(not(feature = "server"), not(test))))]
#[allow(dead_code)]
pub(super) fn audio_toggle_shuffle() {}

/// Seek the background track by `delta_ms` (negative = rewind), clamped to the
/// track. Takes effect after the device drains its already-queued ~200 ms.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_seek(delta_ms: i64) {
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            let ch = p.channels.max(1) as i64;
            // Frames, then scaled to samples — and snapped back to a frame
            // boundary, since an odd offset in a stereo track plays the right
            // channel into the left for the remainder.
            let samples = delta_ms * p.rate as i64 / 1000 * ch;
            let n = (p.at as i64 + samples).clamp(0, p.pcm.len() as i64) as usize;
            p.at = n / ch as usize * ch as usize;
            if p.at < p.pcm.len() {
                p.done = false;
                p.finished_announced = false;
            }
        }
    });
    repaint_audio();
}

/// Restart the background track from the beginning (0 / Home on the audio tab).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn audio_restart() {
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            p.at = 0;
            p.done = false;
            p.finished_announced = false;
        }
    });
    repaint_audio();
}

/// Feed the next chunk to the sound device when it has drained the previous
/// one — the background-player heartbeat, called every idle tick. Copies the
/// chunk out before playing so the `AUDIO` lock isn't held across the device
/// enqueue. No-op when nothing is loaded or the device is still draining.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pump_audio() {
    if crate::sound::playing() {
        return; // still draining the last chunk
    }
    let next = AUDIO.with(|a| {
        let p = a.as_mut()?;
        if p.paused {
            return None; // hold position; the device drains its last chunk to silence
        }
        if p.done || p.at >= p.pcm.len() {
            p.done = true;
            return None;
        }
        let ch = p.channels.max(1) as usize;
        let chunk = (p.rate as usize / 5).max(256) * ch; // ~200 ms of frames
        let end = ((p.at + chunk).min(p.pcm.len())) / ch * ch;
        let slice = p.pcm[p.at..end].to_vec();
        p.at = end;
        Some((slice, p.rate, p.channels))
    });
    if let Some((slice, rate, ch)) = next {
        let _ = crate::sound::play_ch(&slice, rate, ch);
    }
    // Live spectrum from the window around the playhead, peak-held so a
    // transient does not vanish in one 4 Hz refresh.
    AUDIO.with(|a| {
        if let Some(p) = a.as_mut() {
            if p.paused || p.done {
                return;
            }
            let ch = p.channels.max(1) as usize;
            let frames = p.pcm.len() / ch;
            if frames == 0 {
                return;
            }
            let at = (p.at / ch).min(frames.saturating_sub(1));
            let win = 1024.min(frames);
            let start = at.saturating_sub(win / 8);
            let end = (start + win).min(frames);
            let mut mono = alloc::vec![0i16; end.saturating_sub(start)];
            for (i, slot) in mono.iter_mut().enumerate() {
                let f = start + i;
                let mut acc: i32 = 0;
                for c in 0..ch {
                    acc += p.pcm[f * ch + c] as i32;
                }
                *slot = (acc / ch as i32) as i16;
            }
            let now = crate::audio::hud::spectrum_bands(&mono, crate::audio::hud::SPECTRUM_BARS);
            p.spectrum = crate::audio::hud::decay_spectrum(
                &p.spectrum,
                &now,
                crate::audio::hud::SPECTRUM_DECAY,
            );
        }
    });
    // Auto-advance (repeat / next sibling) or announce end-of-track once.
    let advance = AUDIO.with(|a| {
        let p = a.as_mut()?;
        if !(p.done && !p.finished_announced && !crate::sound::playing()) {
            return None;
        }
        crate::audio::hud::auto_next(
            p.playlist_idx,
            p.playlist.len(),
            p.repeat,
            p.shuffle,
            crate::arch::now_ms(),
        )
    });
    if let Some(i) = advance {
        audio_play_index(i);
    } else {
        let finished = AUDIO.with(|a| {
            a.as_mut()
                .map(|p| p.done && !p.finished_announced && !crate::sound::playing())
                .unwrap_or(false)
        });
        if finished {
            AUDIO.with(|a| {
                if let Some(p) = a.as_mut() {
                    p.finished_announced = true;
                }
            });
            serial_println!("\ropen> audio finished");
            super::update_status();
        }
    }
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn pump_audio() {}

/// Repaint the audio tab (progress). Called on switch + ~4 Hz while active.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn repaint_audio() {
    AUDIO.with(|a| {
        if let Some(p) = a.as_ref() {
            // Frames, not samples: on a stereo track the sample count runs at
            // twice real time, so the clock would reach 2x the duration.
            let pos_ms = (p.at / p.channels.max(1) as usize) as u64 * 1000 / p.rate.max(1) as u64;
            crate::framebuffer::draw_audio(&crate::framebuffer::AudioView {
                name: &p.name,
                path: &p.path,
                pos_ms: pos_ms.min(p.total_ms),
                total_ms: p.total_ms,
                rate: p.rate,
                playing: !p.done && !p.paused,
                paused: p.paused,
                peaks: &p.peaks,
                spectrum: &p.spectrum,
                volume: crate::sound::volume(),
                muted: crate::sound::muted(),
                playlist: &p.playlist,
                playlist_idx: p.playlist_idx,
                repeat: p.repeat,
                shuffle: p.shuffle,
            });
        }
    });
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn repaint_audio() {}

