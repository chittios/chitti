//! **Voice-activity detection** — the silero-vad v5 model run by the in-kernel
//! ONNX executor. The model (630 KB) is embedded in the kernel image; state
//! (the two LSTM layers' h/c) is kept across frames so probabilities stream
//! like the reference implementation: feed 512-sample 16 kHz mono frames, get
//! a speech probability per frame.

use crate::mm::Locked;
use crate::onnx::{self, exec};
use alloc::vec;
use alloc::vec::Vec;

// The silero model is embedded only when present at build time (gitignored;
// fetched by `cargo xtask voice-assets`). Absent → an empty slice and `ensure`
// reports VAD unavailable, so the kernel still builds without the voice assets
// (CI, a fresh clone). See `build.rs` for the `voice_vad_embedded` cfg.
#[cfg(voice_vad_embedded)]
static MODEL_BYTES: &[u8] = include_bytes!("../../../assets/voice/silero_vad.onnx");
#[cfg(not(voice_vad_embedded))]
static MODEL_BYTES: &[u8] = &[];

struct Vad {
    model: onnx::Model<'static>,
    h: exec::Val,
    c: exec::Val,
}

static VAD: Locked<Option<Vad>> = Locked::new(None);

fn zeros() -> (exec::Val, exec::Val) {
    (exec::Val::new(vec![2, 1, 64], vec![0.0; 128]), exec::Val::new(vec![2, 1, 64], vec![0.0; 128]))
}

/// Parse the model on first use. Returns false if the embedded model is bad.
fn ensure() -> bool {
    VAD.with(|v| {
        if v.is_some() {
            return true;
        }
        if MODEL_BYTES.is_empty() {
            crate::ktrace::log("vad", "silero not bundled (build without assets/voice/) — VAD unavailable");
            return false;
        }
        match onnx::parse(MODEL_BYTES) {
            Some(model) => {
                crate::ktrace::log_fmt(format_args!("vad: silero loaded — {}", onnx::summary(&model)));
                let (h, c) = zeros();
                *v = Some(Vad { model, h, c });
                true
            }
            None => false,
        }
    })
}

/// Reset the streaming state (call at the start of a listening session).
pub fn reset() {
    VAD.with(|v| {
        if let Some(vad) = v.as_mut() {
            let (h, c) = zeros();
            vad.h = h;
            vad.c = c;
        }
    });
}

/// Speech probability for one 512-sample 16 kHz mono frame (state carries
/// across calls). `None` if the model failed to load or inference errored.
pub fn prob(frame: &[i16]) -> Option<f32> {
    if !ensure() {
        return None;
    }
    let mut x: Vec<f32> = Vec::with_capacity(512);
    for i in 0..512 {
        x.push(frame.get(i).map(|&s| s as f32 / 32768.0).unwrap_or(0.0));
    }
    VAD.with(|v| {
        let vad = v.as_mut()?;
        let xin = exec::Val::new(vec![1, 512], x);
        let out = exec::run(&vad.model, &[("x", xin), ("h", vad.h.clone()), ("c", vad.c.clone())]).ok()?;
        vad.h = out.get("new_h")?.clone();
        vad.c = out.get("new_c")?.clone();
        Some(out.get("prob")?.f[0])
    })
}
