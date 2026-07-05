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

use alloc::string::String;
use alloc::vec::Vec;

/// CTC blank id for this model = last vocab index (1024; vocab 1025).
pub const BLANK: usize = 1024;

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
    let bytes = match super::model_store::parakeet() {
        Some(b) => b,
        None => {
            return alloc::format!(
                "(STT front-end ready: {mels}x{frames} log-mel features; load the parakeet model to decode — see /voice models)"
            );
        }
    };
    let tokens = parse_tokens(TOKENS);

    let model = match crate::onnx::parse(bytes) {
        Some(m) => m,
        None => return String::from("(parakeet model failed to parse)"),
    };
    // audio_signal [1, 80, T] (row-major: mel-major then time).
    let mut sig = alloc::vec::Vec::with_capacity(mels * frames);
    for m in &feat {
        sig.extend_from_slice(m);
    }
    use crate::onnx::exec::Val;
    let x = Val::new(alloc::vec![1, mels, frames], sig);
    let len = Val { dims: alloc::vec![1], f: alloc::vec![frames as f32], i: Some(alloc::vec![frames as i64]), seq: None };
    crate::ktrace::log_fmt(format_args!("stt: running parakeet on {mels}x{frames} features (this is slow on the scalar interpreter)"));
    let out = match crate::onnx::exec::run(&model, &[("audio_signal", x), ("length", len)]) {
        Ok(o) => o,
        Err(e) => return alloc::format!("(parakeet run failed: {e})"),
    };
    let lp = match out.values().next() {
        Some(v) => v,
        None => return String::from("(parakeet produced no output)"),
    };
    // logprobs [1, T', vocab] → [T'][vocab].
    let vocab = *lp.dims.last().unwrap_or(&0);
    if vocab == 0 {
        return String::from("(parakeet output shape unexpected)");
    }
    let tprime = lp.f.len() / vocab;
    let rows: alloc::vec::Vec<alloc::vec::Vec<f32>> = (0..tprime).map(|t| lp.f[t * vocab..(t + 1) * vocab].to_vec()).collect();
    crate::ktrace::log_fmt(format_args!("stt: decoded {tprime} conformer frames (vocab {vocab})"));
    ctc_greedy(&rows, &tokens)
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
}
