//! voice
//!
//! The **voice subsystem** carved out of the former 16k-line `shell/mod.rs`
//! monolith: `/voice` routing, the ONNX VAD/STT/TTS model loading and the
//! mic->VAD->STT->LLM->TTS conversation loop (`voice_talk`), plus the
//! chunked `speech_pump` speech queue. Moved verbatim; `use super::*` keeps
//! the parent's statics visible, and the parent re-imports this module's
//! items with `pub(crate) use voice::*`.

use super::*;

/// True if `/voice <arg>` names a stateless subcommand (test/models/stt/say) —
/// i.e. not the bare conversation loop. Used by the shell to route bare `/voice`
/// to the chat-driven `voice_talk` and everything else to `dispatch_system`.
pub(super) fn voice_is_subcommand(arg: &str) -> bool {
    let a = arg.trim();
    a == "test" || a == "models" || a.starts_with("models load") || a.starts_with("stt ") || a.starts_with("say ")
}

pub(super) fn run_voice(arg: &str) {
    // Only audio playback/capture needs a sound device; model loading and STT
    // (which reads a WAV file) do not — so the check is per-branch, not here.
    let arg = arg.trim();
    if arg == "test" {
        voice_test();
    } else if arg == "models" {
        voice_models();
    } else if let Some(rest) = arg.strip_prefix("models load") {
        // /voice models load <which> [path]   (path optional → default search)
        let mut it = rest.trim().splitn(2, ' ');
        match it.next().filter(|s| !s.is_empty()) {
            Some(which) => {
                let path = it.next().map(|s| s.trim());
                match voice_load(which, path) {
                    Ok((n, src)) => serial_println!("voice> loaded {} ({} bytes) from {}", which, n, src),
                    Err(e) => serial_println!("voice> {}", e),
                }
            }
            None => serial_println!("voice> usage: /voice models load parakeet|kitten [path]"),
        }
    } else if arg == "remote" || arg.starts_with("remote ") {
        voice_remote_cmd(arg.strip_prefix("remote").unwrap_or("").trim());
    } else if let Some(path) = arg.strip_prefix("stt ") {
        voice_stt_file(path.trim());
    } else if let Some(text) = arg.strip_prefix("say ") {
        voice_say(text.trim());
    } else {
        // Bare `/voice` is the interactive conversation loop, which needs the
        // shell's live ChatSession; the interactive loop intercepts it before
        // reaching here (see `run_os`). Reaching this arm means the agent tool
        // layer invoked it, where there is no chat to drive.
        serial_println!("voice> conversation mode runs from the shell prompt (type /voice there); subcommands: test|models|stt <wav>|say <text>");
    }
}

/// `/voice remote …` — configure a hosted TTS/STT provider (human-only, like
/// `/model remote`). Subcommands: `tts <provider> <key> [voice] [model]`,
/// `stt <provider> <key> [model]`, `off [tts|stt]`, or bare = show.
pub(super) fn voice_remote_cmd(rest: &str) {
    use voice_remote::{Endpoint, Provider};
    let mut cfg = voice_remote::load();
    let show = |cfg: &voice_remote::VoiceConfig| {
        let dir = |e: &Option<Endpoint>| match e {
            Some(x) => alloc::format!("{} (voice='{}' model='{}')", x.provider.name(), x.voice, x.model),
            None => "local".into(),
        };
        serial_println!("voice> remote tts: {}", dir(&cfg.tts));
        serial_println!("voice> remote stt: {}", dir(&cfg.stt));
        serial_println!("voice>   providers: elevenlabs cartesia inworld sarvam openai");
        serial_println!("voice>   set: /voice remote tts <provider> <key> [voice] [model]");
        serial_println!("voice>        /voice remote stt <provider> <key> [model]   |   /voice remote off [tts|stt]");
    };
    if rest.is_empty() {
        show(&cfg);
        return;
    }
    let mut it = rest.split_whitespace();
    match it.next() {
        Some("off") => {
            match it.next() {
                Some("tts") => cfg.tts = None,
                Some("stt") => cfg.stt = None,
                _ => {
                    cfg.tts = None;
                    cfg.stt = None;
                }
            }
            voice_remote::save(&cfg);
            serial_println!("voice> remote voice off (using local ONNX models)");
        }
        Some(dir @ ("tts" | "stt")) => {
            let (Some(prov), Some(key)) = (it.next(), it.next()) else {
                serial_println!("voice> usage: /voice remote {dir} <provider> <key> [voice] [model]");
                return;
            };
            let Some(provider) = Provider::parse(prov) else {
                serial_println!("voice> unknown provider '{prov}' (elevenlabs|cartesia|inworld|sarvam|openai)");
                return;
            };
            // TTS: [voice] [model];  STT: [model].
            let (voice, model) = if dir == "tts" {
                (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
            } else {
                (String::new(), it.next().unwrap_or("").to_string())
            };
            let ep = Endpoint { provider, key: key.to_string(), voice, model };
            if dir == "tts" {
                cfg.tts = Some(ep);
            } else {
                cfg.stt = Some(ep);
            }
            voice_remote::save(&cfg);
            serial_println!("voice> remote {dir} → {} (key hidden). `/voice {}` now goes through it.", provider.name(), if dir == "tts" { "say" } else { "stt" });
            serial_println!("voice>   NB: HTTPS via the in-kernel TLS client; a provider that won't handshake reports a TLS error, not a wrong result.");
        }
        _ => show(&cfg),
    }
}

/// Pending synthesized speech: PCM waiting for device slots. Fed by
/// [`speech_pump`] from `ui_tick` (non-blocking — sized to the device's free
/// periods), so playback continues while the next chunk synthesizes on the
/// SMP fleet. Bounded: `voice_say` only enqueues one utterance.
pub(super) static SPEECH_Q: crate::mm::Locked<alloc::collections::VecDeque<i16>> = crate::mm::Locked::new(alloc::collections::VecDeque::new());

/// Feed queued speech into free device periods (never blocks). Called from
/// `ui_tick`, which the ONNX per-node loop pumps — that's what makes chunked
/// TTS gapless: synthesis of chunk k+1 keeps chunk k's audio flowing.
///
/// Drivers that cannot report free slots (`out_free_bytes() == 0`, e.g. HDA /
/// AC'97 single-shot DMA) fall back to: wait until not playing, then push a
/// bounded batch. Without that fallback the queue never drains and
/// `/voice say` hangs the shell forever after "speaking…".
pub(crate) fn speech_pump() {
    if SPEECH_Q.with(|q| q.is_empty()) {
        return;
    }
    let free_bytes = crate::sound::out_free_bytes();
    let free_samples = if free_bytes == 0 {
        // Unknown free-slot accounting: only start a new play when the device
        // is idle, then push up to ~1 s of 24 kHz mono.
        if crate::sound::playing() {
            return;
        }
        crate::sound::tts::RATE as usize // 1 second of samples
    } else {
        free_bytes / 2
    };
    if free_samples == 0 {
        return;
    }
    let slice: alloc::vec::Vec<i16> = SPEECH_Q.with(|q| {
        if q.is_empty() {
            return alloc::vec::Vec::new();
        }
        let n = free_samples.min(q.len());
        q.drain(..n).collect()
    });
    if !slice.is_empty() {
        let _ = crate::sound::play(&slice, crate::sound::tts::RATE);
    }
}

/// Split text into speakable chunks at sentence/clause boundaries, so
/// synthesis can pipeline with playback: the first clause plays while the
/// rest is still synthesizing. Pure — unit-tested. Chunks shorter than
/// `MIN` merge forward (tiny fragments sound choppy and waste per-run cost).
pub(crate) fn split_speech(text: &str) -> alloc::vec::Vec<alloc::string::String> {
    // A sentence ender always splits (first audio = first sentence's synth
    // time); long comma clauses split too so no chunk grows unbounded. Only
    // near-empty fragments merge (they sound choppy and waste a graph run).
    const MIN: usize = 8;
    let mut out: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    let mut cur = alloc::string::String::new();
    for ch in text.chars() {
        cur.push(ch);
        let boundary = matches!(ch, '.' | '!' | '?' | ';' | ':') || (ch == ',' && cur.len() >= 48);
        if boundary && cur.trim().len() >= MIN {
            out.push(core::mem::take(&mut cur).trim().into());
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        match out.last_mut() {
            Some(last) if tail.len() < MIN => {
                last.push(' ');
                last.push_str(tail);
            }
            _ => out.push(tail.into()),
        }
    }
    out
}

/// Chunked TTS core shared by `/voice say` and the conversation loop.
/// Returns total samples synthesized (0 on early failure).
pub(super) fn speak_text(text: &str) -> usize {
    if !crate::sound::is_up() || text.trim().is_empty() {
        return 0;
    }
    let remote_tts = voice_remote::load().tts;
    if remote_tts.is_none() && !ensure_voice_model("kitten") {
        serial_println!(
            "voice> no kitten model found (bundle it in the image, /voice models load kitten <path>, or /voice remote tts …)"
        );
        return 0;
    }
    let chunks = split_speech(text);
    let t0 = crate::arch::now_ms();
    let mut total = 0usize;
    let mut cancelled = false;
    for (i, chunk) in chunks.iter().enumerate() {
        let synth = match &remote_tts {
            Some(e) => voice_remote::synth(e, chunk),
            None => crate::sound::tts::synth(chunk),
        };
        match synth {
            Ok(pcm) => {
                total += pcm.len();
                if i == 0 {
                    serial_println!(
                        "voice> speaking ({} chunk(s); first in {} ms)\u{2026}",
                        chunks.len(),
                        crate::arch::now_ms().saturating_sub(t0)
                    );
                }
                SPEECH_Q.with(|q| q.extend(pcm.iter().copied()));
                speech_pump();
            }
            Err(e) => {
                serial_println!("voice> tts: {}", e);
                break;
            }
        }
        if poll_cancel() {
            cancelled = true;
            break;
        }
    }
    // Drain: keep feeding until the queue and device empty (or Ctrl+C).
    // Bound the wait so a stuck DMA completion (or free-bytes==0 bug) can never
    // freeze the shell: audio_duration + 5 s grace, min 3 s.
    let ms_audio = if total == 0 {
        0
    } else {
        (total as u64)
            .saturating_mul(1000)
            .saturating_div(crate::sound::tts::RATE as u64)
    };
    let deadline = crate::arch::now_ms()
        .saturating_add(ms_audio.saturating_add(5_000).max(3_000));
    while SPEECH_Q.with(|q| !q.is_empty()) || crate::sound::playing() {
        speech_pump();
        // Full upkeep so USB keyboard / net / UI keep working during playback.
        upkeep();
        crate::sched::yield_now();
        if poll_cancel() || poll_interrupt() {
            cancelled = true;
            SPEECH_Q.with(|q| q.clear());
            break;
        }
        if crate::arch::now_ms() >= deadline {
            serial_println!(
                "voice> playback timeout (queue still {} samples) — giving up",
                SPEECH_Q.with(|q| q.len())
            );
            SPEECH_Q.with(|q| q.clear());
            break;
        }
    }
    if cancelled {
        serial_println!("voice> {} samples; cancelled", total);
    }
    total
}

/// `/voice say <text>` — text-to-speech via KittenTTS (G2P → model → playback),
/// **chunked**: each clause plays as soon as it is synthesized while the next
/// one runs on the SMP fleet, so speech starts in ~a second instead of after
/// the whole utterance. Ctrl+C stops between chunks and drains the queue.
pub(super) fn voice_say(text: &str) {
    // `ensure_up`, not `is_up`: a USB audio device plugged in after boot was
    // otherwise never adopted (discovery ran once, at boot).
    if !crate::sound::ensure_up() {
        serial_println!("voice> no sound device");
        return;
    }
    let remote_tts = voice_remote::load().tts;
    match &remote_tts {
        Some(e) => serial_println!(
            "voice> synthesizing via {} \u{201c}{}\u{201d}\u{2026}",
            e.provider.name(),
            text
        ),
        None => serial_println!("voice> synthesizing \u{201c}{}\u{201d}\u{2026}", text),
    }
    let total = speak_text(text);
    if total > 0 {
        serial_println!("voice> {} samples; done", total);
    }
}

/// `/onnx info|run <path>` — the generic ONNX runtime surface: inspect or
/// execute **any** ONNX model from a mounted volume. `run` feeds each graph
/// input a zero tensor of its declared shape (dynamic dims → 1) unless the
/// model needs real inputs, and reports each output's shape + a value preview.
pub(super) fn run_onnx(arg: &str) {
    if arg.trim() == "bench" {
        // Raw dot_f32 throughput: the inner kernel of every conv/matmul the
        // voice models run — isolates SIMD/memory speed from graph overhead.
        let n = 1408usize;
        let a = alloc::vec![1.0f32; n];
        let b = alloc::vec![0.5f32; n];
        let iters = 200_000u64;
        let t0 = crate::arch::now_ms();
        let mut acc = 0f32;
        for _ in 0..iters {
            acc += crate::cortex::tensor::dot_f32(&a, &b);
        }
        let ms = crate::arch::now_ms().saturating_sub(t0).max(1);
        let gmacs = (iters as f64 * n as f64) / (ms as f64 * 1e6);
        serial_println!("onnx> dot_f32 (NEON) len {}: {} ms = {:.2} GMAC/s (acc {})", n, ms, gmacs, acc);
        // Scalar f32 baseline: distinguishes "NEON is slow" from "all FP is slow".
        let t1 = crate::arch::now_ms();
        let mut acc2 = 0f32;
        for _ in 0..iters / 10 {
            let mut s = 0f32;
            for i in 0..n {
                s += a[i] * b[i];
            }
            acc2 += s;
        }
        let ms2 = crate::arch::now_ms().saturating_sub(t1).max(1);
        let gm2 = (iters as f64 / 10.0 * n as f64) / (ms2 as f64 * 1e6);
        serial_println!("onnx> dot scalar len {}: {} ms = {:.2} GMAC/s (acc {})", n, ms2, gm2, acc2);
        return;
    }
    let (sub, path) = match arg.trim().split_once(' ') {
        Some((s, p)) => (s, p.trim()),
        None => {
            serial_println!("onnx> usage: /onnx info <path> | /onnx run <path> | /onnx bench");
            return;
        }
    };
    let bytes = match crate::synapse::fs::read(path) {
        Some(b) => b,
        None => {
            serial_println!("onnx> file not found: {} (mount a volume first, e.g. /mount 0)", path);
            return;
        }
    };
    let model = match crate::onnx::parse(&bytes) {
        Some(m) => m,
        None => {
            serial_println!("onnx> failed to parse {} as ONNX", path);
            return;
        }
    };
    serial_println!("onnx> {}", crate::onnx::summary(&model));
    if sub == "info" {
        serial_println!("  ir_version {}", model.ir_version);
        for i in &model.graph.inputs {
            serial_println!("  input:  {}", i);
        }
        for o in &model.graph.outputs {
            serial_println!("  output: {}", o);
        }
        return;
    }
    if sub != "run" {
        serial_println!("onnx> unknown '{}' — use info|run", sub);
        return;
    }
    // Feed zero tensors for graph inputs not already covered by initializers.
    use crate::onnx::exec::Val;
    let init_names: alloc::vec::Vec<&str> = model.graph.initializers.iter().map(|t| t.name).collect();
    let mut feeds: alloc::vec::Vec<(&str, Val)> = alloc::vec::Vec::new();
    for name in &model.graph.inputs {
        if init_names.contains(name) {
            continue;
        }
        // Without declared shapes here we default to a scalar zero; models with
        // real input needs should be driven by their own command (e.g. /voice).
        feeds.push((name, Val::new(alloc::vec![1], alloc::vec![0.0])));
    }
    serial_println!("onnx> running (zero inputs)\u{2026}");
    match crate::onnx::exec::run(&model, &feeds) {
        Ok(out) => {
            for (name, v) in out.iter() {
                let preview: alloc::vec::Vec<f32> = v.f.iter().take(4).copied().collect();
                serial_println!("  {} {:?} = {:?}\u{2026}", name, v.dims, preview);
            }
        }
        Err(e) => serial_println!("onnx> run error: {}", e),
    }
}

/// Default filenames a voice model may be shipped under (checked in order,
/// across the mounted `/` and common voice dirs, plus x86 Limine boot modules).
pub(super) fn voice_candidates(which: &str) -> &'static [&'static str] {
    match which {
        "kitten" => &["/voice/kitten_tts_mini.onnx", "/kitten_tts_mini.onnx", "/kitten.onnx", "/voice/kitten.onnx", "/mnt/kitten.onnx"],
        "parakeet" => &["/voice/parakeet_ctc_int8.onnx", "/parakeet.onnx", "/voice/parakeet.onnx", "/mnt/parakeet.onnx"],
        _ => &[],
    }
}

/// Load a voice model. With an explicit `path`, read it from the mounts;
/// otherwise search the default locations — a bundled x86 Limine boot module
/// first, then the known filesystem paths on whatever is mounted. Returns
/// `(bytes, source)`.
pub(super) fn voice_load(which: &str, path: Option<&str>) -> Result<(usize, alloc::string::String), alloc::string::String> {
    if which != "kitten" && which != "parakeet" {
        return Err("unknown model (parakeet|kitten)".into());
    }
    if let Some(p) = path {
        serial_println!("voice> reading {} \u{2026}", p);
        let bytes = read_mounted(p).ok_or_else(|| alloc::format!("{} not found on any mount (see /mounts)", p))?;
        let n = crate::sound::model_store::load_bytes(which, bytes)?;
        return Ok((n, p.into()));
    }
    // No path: a bundled boot module (x86 Limine) first, then any disk volume
    // (the FAT ESP / ext4 data partition — aarch64 image), then the mounts.
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::cortex::find_module(which) {
        let n = crate::sound::model_store::load_bytes(which, m.to_vec())?;
        return Ok((n, alloc::format!("boot module ({which})")));
    }
    let fname = alloc::format!("{which}.onnx");
    if let Some(bytes) = find_on_disks(&[&fname]) {
        let n = crate::sound::model_store::load_bytes(which, bytes)?;
        return Ok((n, alloc::format!("{fname} (disk)")));
    }
    for cand in voice_candidates(which) {
        if let Some(bytes) = read_mounted(cand) {
            let n = crate::sound::model_store::load_bytes(which, bytes)?;
            return Ok((n, (*cand).into()));
        }
    }
    Err(alloc::format!("no {which} model bundled or on disk (pass a path, or bundle via the image)"))
}

/// Ensure a voice model is loaded, searching the default locations on first use
/// (lazy — reading the 78/131 MB models at boot would stall the shell). Returns
/// true if loaded (already or just now).
pub(super) fn ensure_voice_model(which: &str) -> bool {
    let loaded = match which {
        "kitten" => crate::sound::model_store::kitten().is_some(),
        "parakeet" => crate::sound::model_store::parakeet().is_some(),
        _ => false,
    };
    if loaded {
        return true;
    }
    match voice_load(which, None) {
        Ok((n, src)) => {
            serial_println!("voice> {} loaded ({} bytes) from {}", which, n, src);
            true
        }
        Err(_) => false,
    }
}

/// `/voice models` — show which voice models are loaded + how to get them.
pub(super) fn voice_models() {
    let mk = |b: bool| if b { "\x1b[32mloaded\x1b[0m" } else { "not loaded" };
    serial_println!("voice> models:");
    serial_println!("  silero-vad   \x1b[32membedded\x1b[0m (VAD, 630 KB)");
    serial_println!("  parakeet-stt {} (STT; /voice models load parakeet [path])", mk(crate::sound::model_store::parakeet().is_some()));
    serial_println!("  kitten-tts   {} (TTS; /voice models load kitten [path])", mk(crate::sound::model_store::kitten().is_some()));
    serial_println!("  (no path = search boot module + any disk for <model>.onnx; loaded on first /voice use)");
    serial_println!("  host: cargo xtask voice-assets  (downloads into assets/voice/)");
}

/// `/voice stt </path/file.wav>` — transcribe a WAV from a mounted volume
/// through the STT front-end. Mic-independent, so the mel + CTC path is
/// exercisable without microphone hardware/permission. Any PCM rate is
/// resampled to 16 kHz for the local parakeet front-end.
pub(super) fn voice_stt_file(path: &str) {
    let remote_stt = voice_remote::load().stt;
    if remote_stt.is_none() && !ensure_voice_model("parakeet") {
        serial_println!("voice> no parakeet model found (bundle it in the image, /voice models load parakeet <path>, or /voice remote stt …)");
        return;
    }
    let bytes = match read_mounted(path) {
        Some(b) => b,
        None => {
            serial_println!("voice> file not found: {} (mount a volume first, e.g. /mount 0)", path);
            return;
        }
    };
    let (pcm_src, rate) = match wav_to_pcm16(&bytes) {
        Some(p) => p,
        None => {
            serial_println!("voice> not a 16-bit PCM WAV: {}", path);
            return;
        }
    };
    // Local STT is hard-wired to 16 kHz mel; remote providers accept the
    // source rate (they resample server-side) but we still normalise for
    // consistent local behaviour.
    let pcm = if rate != 16_000 {
        serial_println!("voice> resampling {} Hz → 16 kHz ({} samples)", rate, pcm_src.len());
        crate::sound::resample(&pcm_src, rate, 16_000)
    } else {
        pcm_src
    };
    match &remote_stt {
        Some(e) => {
            serial_println!(
                "voice> {}: {} samples @16k; transcribing via {}\u{2026}",
                path,
                pcm.len(),
                e.provider.name()
            );
            match voice_remote::transcribe(e, &pcm, 16_000) {
                Ok(text) => serial_println!("voice> stt> {}", text),
                Err(err) => serial_println!("voice> {}", err),
            }
        }
        None => {
            serial_println!("voice> {}: {} samples @16k; transcribing\u{2026}", path, pcm.len());
            let text = crate::sound::stt::transcribe(&pcm);
            serial_println!("voice> stt> {}", text);
        }
    }
}

/// Minimal RIFF/WAVE parser: returns mono S16LE samples (averaging stereo) and
/// the source sample rate. Handles the standard 44-byte header; scans chunks
/// for `fmt ` + `data`. Only 16-bit PCM is supported.
pub(super) fn wav_to_pcm16(b: &[u8]) -> Option<(alloc::vec::Vec<i16>, u32)> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12;
    let mut channels = 1u16;
    let mut rate = 16_000u32;
    let mut bits = 16u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let sz = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = b.get(pos + 8..pos + 8 + sz)?;
        if id == b"fmt " && body.len() >= 16 {
            // audio format 1 = PCM
            let fmt = u16::from_le_bytes([body[0], body[1]]);
            if fmt != 1 {
                return None;
            }
            channels = u16::from_le_bytes([body[2], body[3]]).max(1);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]).max(1);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            data = Some(body);
        }
        pos += 8 + sz + (sz & 1); // chunks are word-aligned
    }
    if bits != 16 {
        return None;
    }
    let data = data?;
    let ch = channels as usize;
    let mut out = alloc::vec::Vec::with_capacity(data.len() / 2 / ch);
    for frame in data.chunks_exact(2 * ch) {
        let mut acc = 0i32;
        for c in 0..ch {
            acc += i16::from_le_bytes([frame[c * 2], frame[c * 2 + 1]]) as i32;
        }
        out.push((acc / ch as i32) as i16);
    }
    Some((out, rate))
}

/// Sound self-test: play a short tone, then sample the mic for 2 s and report
/// the peak level — proves playback and capture end-to-end.
pub(super) fn voice_test() {
    // A USB headset is the commonest reason someone expects audio and gets
    // none, so say what was found on the bus before saying what plays. It is
    // reported whether or not a device came up: a machine with *both* HDA and a
    // USB headset plays through HDA, and knowing that is the difference between
    // "no audio" and "audio, wrong output".
    if crate::drivers::uac::present() {
        serial_println!("voice> USB audio device on the bus:");
        for line in crate::drivers::uac::status_lines() {
            serial_println!("voice>   {line}");
        }
    }
    if !crate::sound::ensure_up() {
        serial_println!("voice> no sound device found");
        return;
    }
    serial_println!(
        "voice> output: {} channel(s)",
        crate::sound::out_channels()
    );
    serial_println!("voice> playing test tone\u{2026}");
    let tone = crate::sound::test_tone(440, 600, 16000);
    match crate::sound::play(&tone, 16000) {
        Ok(()) => {
            while crate::sound::playing() {
                ui_tick();
                crate::sched::yield_now();
            }
            serial_println!("voice> tone done");
        }
        Err(e) => {
            serial_println!("voice> play failed: {}", e);
            return;
        }
    }
    serial_println!("voice> capturing 2 s from the mic\u{2026}");
    if let Err(e) = crate::sound::capture_start(16000) {
        serial_println!("voice> capture failed: {}", e);
        return;
    }
    let mut frame = [0i16; 1600]; // 100 ms at 16 kHz
    let mut peak = 0f32;
    let mut got = 0usize;
    let t0 = crate::arch::now_ms();
    while crate::arch::now_ms().saturating_sub(t0) < 2000 {
        let n = crate::sound::capture_read(&mut frame);
        if n > 0 {
            got += n;
            let r = crate::sound::rms(&frame[..n]);
            if r > peak {
                peak = r;
            }
        }
        ui_tick();
        crate::sched::yield_now();
    }
    crate::sound::capture_stop();
    serial_println!("voice> captured {} samples, peak level {}%", got, (peak * 100.0) as u32);
}

/// The interactive voice session: live waveform modal driven by mic RMS, with
/// level-based endpointing (speech starts above the threshold, an utterance
/// ends after ~800 ms of silence). Esc / q / Ctrl+C or the Stop button end the
/// session. The LLM backend matches shell chat: **remote** when `/model remote`
/// is active, otherwise the local GGUF.
pub(super) fn voice_talk(
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
    remote_on: bool,
    remote_cfg: &Option<remote::RemoteConfig>,
    remote_chat: &mut Option<remote::RemoteChat>,
) {
    if !crate::sound::ensure_up() {
        serial_println!("voice> no sound device found");
        return;
    }
    // The conversation loop needs STT (hear) and TTS (speak). Prefer a
    // human-configured remote endpoint; otherwise load the local ONNX models.
    // Missing backends degrade the loop rather than abort it.
    let voice_cfg = voice_remote::load();
    let have_stt = voice_cfg.stt.is_some() || ensure_voice_model("parakeet");
    let have_tts = voice_cfg.tts.is_some() || ensure_voice_model("kitten");
    if !have_stt {
        serial_println!(
            "voice> no STT backend — load parakeet (`/voice models load parakeet`) or `/voice remote stt …`"
        );
    } else if voice_cfg.stt.is_some() {
        serial_println!("voice> STT: remote ({})", voice_cfg.stt.as_ref().unwrap().provider.name());
    }
    if !have_tts {
        serial_println!(
            "voice> no TTS backend — load kitten (`/voice models load kitten`) or `/voice remote tts …`"
        );
    } else if voice_cfg.tts.is_some() {
        serial_println!("voice> TTS: remote ({})", voice_cfg.tts.as_ref().unwrap().provider.name());
    }
    // LLM: same backend as shell chat.
    if remote_on {
        if let Some(cfg) = remote_cfg {
            let rc = remote_chat.get_or_insert_with(|| remote::RemoteChat::new(cfg.clone()));
            if rc.is_empty() && session.messages.len() > 1 {
                rc.hydrate_from_session(session);
            }
            serial_println!("voice> LLM backend: remote ({})", cfg.model);
        } else {
            serial_println!("voice> remote mode but no endpoint — /model remote <url>");
            return;
        }
    } else if chat.is_none() {
        let mut spin = Spinner::new("loading model");
        *chat = ChatSession::load(&mut spin);
        spin.clear();
        if let Some(sess) = chat.as_mut() {
            if session.messages.len() > 1 {
                sess.hydrate_from_session(session);
            }
        }
        if chat.is_none() {
            serial_println!("voice> no local model — try /model remote or bundle a GGUF");
            return;
        }
    }
    serial_println!("voice> listening \u{2014} Esc (or the Stop button) ends the session");
    if let Err(e) = crate::sound::capture_start(16000) {
        serial_println!("voice> capture failed: {}", e);
        return;
    }
    #[cfg(not(test))]
    {
        let mut levels: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
        let mut frame = [0i16; 1600];
        // VAD works on 512-sample windows; capture chunks are re-framed here.
        let mut vadbuf: alloc::vec::Vec<i16> = alloc::vec::Vec::new();
        let mut utter: alloc::vec::Vec<i16> = alloc::vec::Vec::new();
        let mut in_speech = false;
        let mut silent_ms = 0u32;
        crate::sound::vad::reset();
        crate::framebuffer::draw_voice(&levels, "listening\u{2026}");
        loop {
            if let Some(b) = crate::console::read_byte() {
                if b == 0x1b || b == b'q' || b == 3 {
                    break;
                }
            }
            let t = crate::mouse::tick();
            if t.moved {
                crate::framebuffer::cursor_move(t.x, t.y);
            }
            if t.pressed && matches!(crate::framebuffer::modal_hit(t.x, t.y), crate::framebuffer::ModalHit::Ok) {
                break;
            }
            let n = crate::sound::capture_read(&mut frame);
            if n > 0 {
                let r = crate::sound::rms(&frame[..n]);
                levels.push(r);
                if levels.len() > 256 {
                    levels.remove(0);
                }
                vadbuf.extend_from_slice(&frame[..n]);
                // Run silero VAD over each complete 512-sample window (32 ms).
                while vadbuf.len() >= 512 {
                    let win: alloc::vec::Vec<i16> = vadbuf.drain(..512).collect();
                    // Falls back to a simple level gate if the model failed.
                    let speech = match crate::sound::vad::prob(&win) {
                        Some(p) => p > 0.5,
                        None => crate::sound::rms(&win) > 0.02,
                    };
                    if speech {
                        in_speech = true;
                        silent_ms = 0;
                        utter.extend_from_slice(&win);
                    } else if in_speech {
                        silent_ms += 32;
                        utter.extend_from_slice(&win);
                        if silent_ms > 800 {
                            let ms = utter.len() as u32 / 16;
                            serial_println!("voice> utterance captured: {} ms ({} samples, silero-gated)", ms, utter.len());
                            let clip = core::mem::take(&mut utter);
                            in_speech = false;
                            silent_ms = 0;
                            // Full pipeline: hear -> think -> speak. Playback and
                            // capture share the device, so stop capture first, run
                            // the turn, then resume listening (VAD reset).
                            crate::sound::capture_stop();
                            voice_converse_turn(
                                chat,
                                session,
                                remote_on,
                                remote_cfg,
                                remote_chat,
                                &clip,
                                have_stt,
                                have_tts,
                                &mut levels,
                            );
                            crate::sound::vad::reset();
                            vadbuf.clear();
                            let _ = crate::sound::capture_start(16000);
                            crate::framebuffer::draw_voice(&levels, "listening\u{2026}");
                        }
                    }
                }
                let status = if in_speech { "listening\u{2026} (speech detected)" } else { "listening\u{2026} (Esc or Stop to end)" };
                crate::framebuffer::draw_voice(&levels, status);
            }
            crate::net::poll();
            crate::sched::yield_now();
        }
        crate::framebuffer::modal_dismiss();
    }
    crate::sound::capture_stop();
    serial_println!("voice> session ended");
}

/// One voice-conversation turn: transcribe the captured `clip` (STT), feed the
/// transcript to the LLM (remote or local, same as shell chat), then speak the
/// reply (TTS). Each stage degrades independently.
#[cfg(not(test))]
pub(super) fn voice_converse_turn(
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
    remote_on: bool,
    remote_cfg: &Option<remote::RemoteConfig>,
    remote_chat: &mut Option<remote::RemoteChat>,
    clip: &[i16],
    have_stt: bool,
    have_tts: bool,
    levels: &mut alloc::vec::Vec<f32>,
) {
    // 1. Hear — remote STT if configured, else local parakeet.
    let heard = if have_stt {
        crate::framebuffer::draw_voice(levels, "transcribing\u{2026}");
        let t = match voice_remote::load().stt {
            Some(e) => match voice_remote::transcribe(&e, clip, 16_000) {
                Ok(s) => s,
                Err(err) => {
                    serial_println!("voice> remote stt: {err}");
                    alloc::string::String::new()
                }
            },
            None => crate::sound::stt::transcribe(clip),
        };
        serial_println!("voice> you: {}", t);
        t
    } else {
        alloc::string::String::new()
    };
    let heard = heard.trim();
    // Treat diagnostic placeholders as "nothing heard".
    if heard.is_empty() || (heard.starts_with('(') && heard.ends_with(')')) {
        serial_println!("voice> (nothing to transcribe \u{2014} continuing to listen)");
        return;
    }
    // 2. Think — same backend as shell chat.
    crate::framebuffer::draw_voice(levels, "thinking\u{2026}");
    let reply = if remote_on {
        match (remote_cfg, remote_chat.as_mut()) {
            (Some(cfg), Some(rc)) => {
                let _ = cfg;
                rc.turn(heard, session)
            }
            (Some(cfg), None) => {
                let mut rc = remote::RemoteChat::new(cfg.clone());
                if session.messages.len() > 1 {
                    rc.hydrate_from_session(session);
                }
                let out = rc.turn(heard, session);
                *remote_chat = Some(rc);
                out
            }
            _ => {
                serial_println!("voice> (remote mode misconfigured \u{2014} cannot reply)");
                return;
            }
        }
    } else {
        match chat.as_mut() {
            Some(sess) => sess.turn(heard, session),
            None => {
                serial_println!("voice> (no LLM loaded \u{2014} cannot reply)");
                return;
            }
        }
    };
    let reply = reply.trim();
    if reply.is_empty() {
        return;
    }
    // 3. Speak — same chunked/remote path as `/voice say` (first audio ASAP).
    if have_tts {
        crate::framebuffer::draw_voice(levels, "speaking\u{2026}");
        let _ = speak_text(reply);
    }
}
