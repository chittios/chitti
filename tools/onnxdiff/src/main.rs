//! **onnxdiff** — host-side calibration harness for the kernel's bare-metal
//! ONNX interpreter. Mounts `kernel/src/onnx` natively (`#[path]`), stubs its
//! three kernel dependencies with the kernel's *exact* math (so the kernel's
//! transcendental approximations are part of what's under test), runs a model
//! with fixed inputs, and dumps per-node stats (`NODE ...` lines) for diffing
//! against an onnxruntime reference (`tools/onnxdiff/ref.py`).
//!
//! Usage:
//!   onnxdiff kitten   <kitten.onnx>  [out.txt]   # "hello world" fixed input
//!   onnxdiff parakeet <parakeet.onnx> <mel.bin>  [out.txt]  # 80xT f32 LE mel

extern crate alloc;

use std::io::Write as _;
use std::sync::Mutex;

/// Where `serial_println!` lines go: stdout by default, or a file (huge dumps).
pub static SINK: Mutex<Option<std::fs::File>> = Mutex::new(None);

#[macro_export]
macro_rules! serial_println {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        // Microsecond timestamp prefix: per-node time = delta between
        // consecutive NODE lines (the op executes between the prints).
        let us = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
        let mut sink = $crate::SINK.lock().unwrap();
        match sink.as_mut() {
            Some(f) => { use std::io::Write as _; writeln!(f, "[{}] {}", us, line).unwrap(); }
            None => println!("[{}] {}", us, line),
        }
    }};
}

/// The kernel's exact scalar math, copied from `kernel/src/cortex/tensor.rs`
/// and `kernel/src/sound/mel.rs` so the harness reproduces kernel numerics
/// bit-for-bit (std `exp`/`cos` would hide approximation error).
pub mod cortex {
    pub mod tensor {
        /// The kernel's exact NEON dot (4 independent accumulators) on aarch64
        /// hosts — same silicon as the HVF VM, so host timing ≈ kernel timing.
        #[cfg(target_arch = "aarch64")]
        pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
            use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
            let n = a.len().min(b.len());
            let mut i = 0;
            unsafe {
                let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                while i + 16 <= n {
                    let (pa, pb) = (a.as_ptr().add(i), b.as_ptr().add(i));
                    a0 = vfmaq_f32(a0, vld1q_f32(pa), vld1q_f32(pb));
                    a1 = vfmaq_f32(a1, vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
                    a2 = vfmaq_f32(a2, vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
                    a3 = vfmaq_f32(a3, vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
                    i += 16;
                }
                let mut acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
                while i + 4 <= n {
                    acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
                    i += 4;
                }
                let mut sum = vaddvq_f32(acc);
                while i < n {
                    sum += a[i] * b[i];
                    i += 1;
                }
                sum
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
            let n = a.len().min(b.len());
            let mut acc = [0f32; 4];
            let chunks = n / 4;
            for i in 0..chunks {
                for l in 0..4 {
                    acc[l] += a[i * 4 + l] * b[i * 4 + l];
                }
            }
            let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for i in chunks * 4..n {
                s += a[i] * b[i];
            }
            s
        }
        const LN2: f32 = core::f32::consts::LN_2;
        pub fn expf(x: f32) -> f32 {
            if x.is_nan() {
                return x;
            }
            if x > 88.0 {
                return f32::INFINITY;
            }
            if x < -88.0 {
                return 0.0;
            }
            let k = (x / LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
            let r = x - k as f32 * LN2;
            let r2 = r * r;
            let p = 1.0
                + r
                + 0.5 * r2
                + r2 * r * (1.0 / 6.0)
                + r2 * r2 * (1.0 / 24.0)
                + r2 * r2 * r * (1.0 / 120.0);
            let two_k = f32::from_bits((((k + 127) as u32) & 0xff) << 23);
            p * two_k
        }
    }
}
pub mod sound {
    pub mod mel {
        fn floorf(x: f32) -> f32 {
            let t = x as i64 as f32;
            if t > x {
                t - 1.0
            } else {
                t
            }
        }
        fn cos_sin_small(r: f32) -> (f32, f32) {
            let r2 = r * r;
            let c = 1.0 - r2 * (0.5 - r2 * (1.0 / 24.0 - r2 / 720.0));
            let s = r * (1.0 - r2 * (1.0 / 6.0 - r2 * (1.0 / 120.0 - r2 / 5040.0)));
            (c, s)
        }
        pub fn cosf_pub(x: f32) -> f32 {
            use core::f32::consts::FRAC_PI_2;
            let k = floorf(x / FRAC_PI_2 + 0.5) as i32;
            let r = x - k as f32 * FRAC_PI_2;
            let (c, s) = cos_sin_small(r);
            match k.rem_euclid(4) {
                0 => c,
                1 => -s,
                2 => -c,
                _ => s,
            }
        }
    }
}

#[path = "../../../kernel/src/onnx/mod.rs"]
pub mod onnx;

use onnx::exec::Val;

/// "hello world" phoneme ids from the kernel G2P (g2p.rs reference test).
const HELLO_IDS: &[i64] = &[0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 0];

fn voice_row(bin: &[u8], ntok: usize) -> Vec<f32> {
    let row = ntok.min(399);
    let base = row * 256 * 4;
    (0..256)
        .map(|i| f32::from_le_bytes([bin[base + i * 4], bin[base + i * 4 + 1], bin[base + i * 4 + 2], bin[base + i * 4 + 3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: onnxdiff kitten <kitten.onnx> [out.txt] | onnxdiff parakeet <parakeet.onnx> <mel.bin> [out.txt]";
    let mode = args.get(1).expect(usage).as_str();
    let model_path = args.get(2).expect(usage);
    let bytes = std::fs::read(model_path).expect("read model");
    let model = onnx::parse(&bytes).expect("parse model");
    eprintln!("parsed: {}", onnx::summary(&model));

    let feeds: Vec<(&str, Val)> = match mode {
        "kitten" => {
            if let Some(out) = args.get(3) {
                *SINK.lock().unwrap() = Some(std::fs::File::create(out).expect("create out"));
            }
            let vb = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernel/src/sound/testdata/kitten_voice.bin"))
                .expect("kitten_voice.bin");
            let n = HELLO_IDS.len();
            let input_ids = Val { dims: vec![1, n], f: HELLO_IDS.iter().map(|&x| x as f32).collect(), i: Some(HELLO_IDS.to_vec()), seq: None };
            let style = Val::new(vec![1, 256], voice_row(&vb, n));
            let speed = Val::new(vec![1], vec![1.0]);
            vec![("input_ids", input_ids), ("style", style), ("speed", speed)]
        }
        "parakeet" => {
            let mel_path = args.get(3).expect(usage);
            if let Some(out) = args.get(4) {
                *SINK.lock().unwrap() = Some(std::fs::File::create(out).expect("create out"));
            }
            let mb = std::fs::read(mel_path).expect("read mel.bin");
            let f: Vec<f32> = mb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let frames = f.len() / 80;
            let x = Val::new(vec![1, 80, frames], f);
            let len = Val { dims: vec![1], f: vec![frames as f32], i: Some(vec![frames as i64]), seq: None };
            eprintln!("mel: 80x{frames}");
            vec![("audio_signal", x), ("length", len)]
        }
        _ => panic!("{usage}"),
    };

    onnx::exec::NODE_TRACE.store(true, core::sync::atomic::Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    match onnx::exec::run(&model, &feeds) {
        Ok(out) => {
            eprintln!("ran in {:.1?}", t0.elapsed());
            for (name, v) in out.iter() {
                let mx = v.f.iter().fold(0f32, |m, &x| m.max(x.abs()));
                eprintln!("OUTPUT '{}' dims={:?} n={} maxabs={:.6}", name, v.dims, v.f.len(), mx);
            }
            // Audible check: write the waveform (largest output) as raw f32 LE.
            if let Some(wav) = out.values().max_by_key(|v| v.f.len()) {
                let mut buf = Vec::with_capacity(wav.f.len() * 4);
                for &s in &wav.f {
                    buf.extend_from_slice(&s.to_le_bytes());
                }
                std::fs::write("/tmp/onnxdiff_wave.f32", &buf).unwrap();
                eprintln!("wrote /tmp/onnxdiff_wave.f32 ({} samples)", wav.f.len());
            }
        }
        Err(e) => eprintln!("run error: {e}"),
    }
    if let Some(f) = SINK.lock().unwrap().as_mut() {
        f.flush().unwrap();
    }
}
