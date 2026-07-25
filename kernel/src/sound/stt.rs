//! **Speech-to-text** — the CTC decode + tokenizer side of the parakeet STT
//! pipeline, plus the glue from PCM → log-mel features ([`super::mel`]) → model
//! logprobs → text.
//!
//! The acoustic model (NeMo parakeet-tdt-ctc-110m, int8, 131 MB) is **not**
//! embedded in the kernel — it is far too large for `include_bytes!` and the
//! scalar ONNX interpreter would run its 4,733-node conformer well below real
//! time. It is loaded from disk when present (`/voice`'s STT path), and the
//! ONNX executor grows the int8 conformer op-set incrementally. What lands
//! here and is unit-tested is the deterministic decode half: the SentencePiece
//! token table and the CTC greedy collapse that turns `[T, vocab]` logprobs
//! into text — the exact algorithm that produced the correct reference
//! transcriptions on the host.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use crate::onnx::exec::Val;

/// CTC blank id for this model = last vocab index (1024; vocab 1025).
pub const BLANK: usize = 1024;
/// Expected SentencePiece vocab size (ids 0..=1024 inclusive).
pub const VOCAB: usize = 1025;

/// Parakeet's SentencePiece vocab (`<piece> <id>` per line). Only ~10 KB, so it
/// is embedded rather than loaded from disk — the CTC decode always has its
/// id→text table the moment the (large, on-disk) acoustic model is present.
static TOKENS: &str = include_str!("testdata/parakeet_tokens.txt");

/// Parse a sherpa/NeMo `tokens.txt` (`<piece> <id>` per line) into an
/// id-indexed table. SentencePiece word-boundary marker `▁` (U+2581) becomes a
/// leading space at join time.
pub fn parse_tokens(text: &str) -> Vec<String> {
    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut max_id = 0usize;
    for line in text.lines() {
        let mut it = line.rsplitn(2, ' ');
        let id = it.next().and_then(|s| s.parse::<usize>().ok());
        let piece = it.next().unwrap_or("");
        if let Some(id) = id {
            max_id = max_id.max(id);
            pairs.push((id, piece.into()));
        }
    }
    // Index by id, filling any gaps (the table must be dense to `max_id`).
    let mut out = alloc::vec![String::new(); max_id + 1];
    for (id, p) in pairs {
        out[id] = p;
    }
    out
}

/// Greedy CTC decode of `[T][vocab]` log-probabilities: argmax per frame,
/// collapse runs of the same id, drop blanks — then map ids through `tokens`,
/// turning the `▁` word marker into spaces.
pub fn ctc_greedy(logprobs: &[Vec<f32>], tokens: &[String]) -> String {
    let mut ids: Vec<usize> = Vec::new();
    let mut prev = usize::MAX;
    for frame in logprobs {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in frame.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        if best != BLANK && best != prev {
            ids.push(best);
        }
        prev = best;
    }
    let mut out = String::new();
    for id in ids {
        if let Some(p) = tokens.get(id) {
            // SentencePiece '▁' (U+2581) marks a word start → space.
            out.push_str(&p.replace('\u{2581}', " "));
        }
    }
    out.trim().into()
}

/// Pick the CTC logprob tensor from the graph outputs.
///
/// **Critical:** `BTreeMap::values()` is alphabetical by output *name*, not
/// execution order — grabbing `.next()` was the same class of bug that made
/// KittenTTS play a 14-sample duration tensor instead of the waveform. Prefer
/// a tensor whose last (or middle) dim is the vocab size, else the largest
/// float tensor.
pub fn pick_logprobs<'a>(out: &'a BTreeMap<String, Val>) -> Option<&'a Val> {
    // Name hints first.
    for (k, v) in out.iter() {
        let kl = k.to_ascii_lowercase();
        if (kl.contains("log") || kl.contains("logit") || kl.contains("prob") || kl.contains("output"))
            && v.f.len() >= VOCAB
        {
            return Some(v);
        }
    }
    // Dimensional: last or second-to-last ≈ vocab.
    let by_vocab = out.values().filter(|v| {
        let d = v.dims.as_slice();
        d.last() == Some(&VOCAB)
            || d.last() == Some(&(VOCAB - 1))
            || (d.len() >= 2 && (d[d.len() - 2] == VOCAB || d[d.len() - 2] == VOCAB - 1))
    });
    if let Some(v) = by_vocab.max_by_key(|v| v.f.len()) {
        return Some(v);
    }
    out.values().max_by_key(|v| v.f.len())
}

/// Reshape a logprob `Val` into `[T][vocab]` rows, accepting NeMo layouts
/// `[1,T,V]`, `[1,V,T]`, or flat `[T*V]` with last-dim = V.
pub fn frames_from_logprobs(lp: &Val) -> Result<Vec<Vec<f32>>, &'static str> {
    let d = lp.dims.as_slice();
    let (t, v, time_major) = match d {
        // [B, A, B2] — prefer the dim that matches vocab as V.
        &[1, a, b] if a == VOCAB || a == VOCAB - 1 => (b, a, false), // [B, V, T]
        &[1, a, b] if b == VOCAB || b == VOCAB - 1 || b > 256 => (a, b, true), // [B, T, V]
        &[1, a, b] if a > 256 && a > b => (b, a, false), // large middle → V
        &[1, a, b] => (a, b, true),
        &[t, v] if v == VOCAB || v == VOCAB - 1 || v > 256 => (t, v, true),
        _ => {
            // Fall back: assume last dim is vocab.
            let v = *d.last().unwrap_or(&VOCAB);
            if v == 0 || lp.f.len() % v != 0 {
                return Err("parakeet output shape unexpected");
            }
            (lp.f.len() / v, v, true)
        }
    };
    if t == 0 || v == 0 || lp.f.len() < t * v {
        return Err("parakeet output too small for declared shape");
    }
    let mut rows = Vec::with_capacity(t);
    if time_major {
        // row t = f[t*v .. (t+1)*v]
        for ti in 0..t {
            let s = ti * v;
            rows.push(lp.f[s..s + v].to_vec());
        }
    } else {
        // column-major in time: f[vi*t + ti]
        for ti in 0..t {
            let mut row = Vec::with_capacity(v);
            for vi in 0..v {
                row.push(lp.f[vi * t + ti]);
            }
            rows.push(row);
        }
    }
    Ok(rows)
}

/// End-to-end transcription of 16 kHz mono PCM. Computes the log-mel features
/// (always), then runs the acoustic model **if it has been loaded** from disk;
/// otherwise returns a diagnostic with the feature shape so the front half is
/// observable without the (large, slow) model.
pub fn transcribe(pcm: &[i16]) -> String {
    let feat = super::mel::features(pcm);
    if feat.is_empty() {
        return String::from("(clip too short)");
    }
    let (mels, frames) = (feat.len(), feat[0].len());
    let model = match super::model_store::parakeet_model() {
        Some(m) => m,
        None => {
            // Bytes present but unparsed, or neither loaded.
            if super::model_store::parakeet().is_none() {
                return alloc::format!(
                    "(STT front-end ready: {mels}x{frames} log-mel features; load the parakeet model to decode — see /voice models)"
                );
            }
            return String::from("(parakeet model failed to parse)");
        }
    };
    let tokens = parse_tokens(TOKENS);

    // audio_signal [1, 80, T] (row-major: mel-major then time).
    let mut sig = alloc::vec::Vec::with_capacity(mels * frames);
    for m in &feat {
        sig.extend_from_slice(m);
    }
    let x = Val::new(alloc::vec![1, mels, frames], sig);
    // NeMo length is int64 frames; keep f mirror for ops that only read f.
    let len = Val {
        dims: alloc::vec![1],
        f: alloc::vec![frames as f32],
        i: Some(alloc::vec![frames as i64]),
        seq: None,
    };
    crate::ktrace::log_fmt(format_args!(
        "stt: running parakeet on {mels}x{frames} features"
    ));
    let (a0, s0) = crate::mm::heap::alloc_stats();
    let t0 = crate::arch::now_ms();
    let out = match crate::onnx::exec::run(model, &[("audio_signal", x), ("length", len)]) {
        Ok(o) => o,
        Err(e) => return alloc::format!("(parakeet run failed: {e})"),
    };
    let (a1, s1) = crate::mm::heap::alloc_stats();
    crate::ktrace::log_fmt(format_args!(
        "stt: run {} ms, {} allocs, {} scan steps ({} avg)",
        crate::arch::now_ms().saturating_sub(t0),
        a1 - a0,
        s1 - s0,
        (s1 - s0) / (a1 - a0).max(1)
    ));
    let lp = match pick_logprobs(&out) {
        Some(v) => v,
        None => return String::from("(parakeet produced no output)"),
    };
    crate::ktrace::log_fmt(format_args!(
        "stt: logprobs dims={:?} n={}",
        lp.dims,
        lp.f.len()
    ));
    let rows = match frames_from_logprobs(lp) {
        Ok(r) => r,
        Err(e) => return alloc::format!("({e}: dims={:?})", lp.dims),
    };
    crate::ktrace::log_fmt(format_args!(
        "stt: decoded {} conformer frames (vocab {})",
        rows.len(),
        rows.first().map(|r| r.len()).unwrap_or(0)
    ));
    let text = ctc_greedy(&rows, &tokens);
    if text.is_empty() {
        // Distinguish "model ran but all-blank" from silence.
        String::from("(no speech recognised)")
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test_case]
    fn ctc_collapse_and_tokens() {
        // tokens.txt style table.
        let toks = parse_tokens("\u{2581}hi 0\n\u{2581}there 1\n! 2\n<blk> 1024\n");
        assert_eq!(toks.len(), 1025);
        assert_eq!(toks[0], "\u{2581}hi");
        // Frames: hi, hi (repeat→collapse), blank, there, !, ! (collapse).
        let mk = |id: usize| {
            let mut f = vec![0f32; 1025];
            f[id] = 1.0;
            f
        };
        let lp = vec![mk(0), mk(0), mk(BLANK), mk(1), mk(2), mk(2)];
        assert_eq!(ctc_greedy(&lp, &toks), "hi there!".to_string());
    }

    #[test_case]
    fn pick_logprobs_prefers_vocab_shaped() {
        let mut m = BTreeMap::new();
        m.insert(
            "duration".into(),
            Val::new(alloc::vec![1, 14], alloc::vec![0.0; 14]),
        );
        m.insert(
            "logprobs".into(),
            Val::new(alloc::vec![1, 3, VOCAB], alloc::vec![0.0; 3 * VOCAB]),
        );
        let v = pick_logprobs(&m).unwrap();
        assert_eq!(v.f.len(), 3 * VOCAB);
    }

    #[test_case]
    fn frames_btv_and_bvt() {
        // [1, T=2, V=VOCAB] time-major (NeMo default).
        let mut f = vec![0f32; 2 * VOCAB];
        f[0] = 1.0; // t0 v0
        f[VOCAB + 1] = 2.0; // t1 v1
        let v = Val::new(alloc::vec![1, 2, VOCAB], f);
        let rows = frames_from_logprobs(&v).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[1][1], 2.0);

        // [1, V=VOCAB, T=2] vocab-major.
        let mut f2 = vec![0f32; 2 * VOCAB];
        // f[vi*t + ti]: v0 t0 → 0; v1 t1 → 1*2+1 = 3
        f2[0] = 1.0;
        f2[3] = 2.0;
        let v2 = Val::new(alloc::vec![1, VOCAB, 2], f2);
        let rows2 = frames_from_logprobs(&v2).unwrap();
        assert_eq!(rows2.len(), 2);
        assert_eq!(rows2[0][0], 1.0);
        assert_eq!(rows2[1][1], 2.0);
    }
}
