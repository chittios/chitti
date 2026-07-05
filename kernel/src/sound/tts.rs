//! **Text-to-speech** — the KittenTTS pipeline: text → [`g2p`] phoneme ids →
//! the KittenTTS ONNX model (run by [`crate::onnx`]) → a 24 kHz waveform played
//! through the [`SndDevice`](super::SndDevice).
//!
//! The 78 MB model is loaded at runtime (`/voice models load kitten <path>` →
//! [`super::model_store`]); it is far too big to embed. One speaker style vector
//! is embedded (`kitten_voice.bin`, [400,256] — indexed by token count as the
//! reference does) so `/voice say` works as soon as the model is loaded.

use alloc::vec::Vec;

/// Embedded speaker style table `[400][256]` (expr-voice-2-f), LE f32.
static VOICE: &[u8] = include_bytes!("testdata/kitten_voice.bin");

fn voice_row(ntok: usize) -> Vec<f32> {
    let row = ntok.min(399);
    let base = row * 256 * 4;
    (0..256).map(|i| f32::from_le_bytes([VOICE[base + i * 4], VOICE[base + i * 4 + 1], VOICE[base + i * 4 + 2], VOICE[base + i * 4 + 3]])).collect()
}

/// Synthesize `text` to a 24 kHz mono waveform via KittenTTS. `None` if the
/// model isn't loaded or the run fails.
pub fn synth(text: &str) -> Result<Vec<i16>, alloc::string::String> {
    use crate::onnx::exec::Val;
    let bytes = super::model_store::kitten().ok_or("kitten model not loaded (/voice models load kitten <path>)")?;
    let ids = super::g2p::text_to_ids(text);
    if ids.len() <= 2 {
        return Err("nothing to say (no phonemes)".into());
    }
    let t0 = crate::arch::now_ms();
    let model = crate::onnx::parse(bytes).ok_or("kitten model parse failed")?;
    let t_parse = crate::arch::now_ms();
    let n = ids.len();
    let input_ids = Val { dims: alloc::vec![1, n], f: ids.iter().map(|&x| x as f32).collect(), i: Some(ids), seq: None };
    let style = Val::new(alloc::vec![1, 256], voice_row(n));
    let speed = Val::new(alloc::vec![1], alloc::vec![1.0]);
    crate::ktrace::log_fmt(format_args!("tts: running kitten on {n} tokens (slow on the scalar interpreter)"));
    let (a0, s0) = crate::mm::heap::alloc_stats();
    let out = crate::onnx::exec::run(&model, &[("input_ids", input_ids), ("style", style), ("speed", speed)]).map_err(|e| alloc::format!("kitten run failed: {e}"))?;
    let (a1, s1) = crate::mm::heap::alloc_stats();
    crate::ktrace::log_fmt(format_args!(
        "tts: parse {} ms, run {} ms, {} allocs, {} scan steps ({} avg)",
        t_parse.saturating_sub(t0),
        crate::arch::now_ms().saturating_sub(t_parse),
        a1 - a0,
        s1 - s0,
        (s1 - s0) / (a1 - a0).max(1)
    ));
    // The waveform is the largest float output (the model also emits short
    // per-token tensors like durations); pick by length, not map order.
    let wav = out.values().max_by_key(|v| v.f.len()).ok_or("kitten produced no output")?;
    // The StyleTTS2 decoder currently overflows to NaN/inf on the scalar int8
    // interpreter (its intermediate activations run ~30x hot, compounding
    // through the residual/upsample chain) — refuse to "play" garbage.
    if wav.f.iter().any(|s| !s.is_finite()) {
        return Err("kitten waveform is non-finite (decoder int8 numerics overflow — see parakeet-stt-numerics)".into());
    }
    let pcm: Vec<i16> = wav.f.iter().map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
    Ok(pcm)
}

/// KittenTTS output sample rate.
pub const RATE: u32 = 24000;
