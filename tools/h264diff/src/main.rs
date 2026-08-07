//! Decode an MP4's first keyframe with the KERNEL's own H.264 decoder (mounted
//! via #[path]) and write its YUV planes to stdout, so diff.py can compare them
//! bit-for-bit against PyAV/ffmpeg. The onnxdiff/cortexdiff pattern.
extern crate alloc;

// The real kernel audio tree (default; the audio bring-up harness uses it).
#[cfg(feature = "real-audio")]
mod sound {
    pub mod mel {
        pub fn cosf_pub(x: f32) -> f32 {
            x.cos()
        }
    }
}
#[cfg(feature = "real-audio")]
mod shell {
    pub fn upkeep() {}
}
#[cfg(feature = "real-audio")]
mod cortex {
    pub mod tensor {
        pub fn libm_sqrtf(x: f32) -> f32 {
            x.sqrt()
        }
    }
}
#[cfg(feature = "real-audio")]
#[path = "../../../kernel/src/audio/mod.rs"]
mod audio;

// Stub of the kernel `audio` API surface `video/mod.rs` touches — build with
// `--no-default-features` to exercise only the video decoder without coupling
// to the (independently evolving) audio tree.
#[cfg(not(feature = "real-audio"))]
mod audio {
    extern crate alloc;
    pub struct Audio {
        pub rate: u32,
        pub pcm: alloc::vec::Vec<i16>,
    }
    impl Audio {
        pub fn duration_ms(&self) -> u64 {
            if self.rate == 0 { 0 } else { self.pcm.len() as u64 * 1000 / self.rate as u64 }
        }
    }
    pub fn decode(_bytes: &[u8]) -> Result<Audio, &'static str> {
        Err("h264diff stub")
    }
    pub mod aac {
        pub struct Asc {
            pub sample_rate: u32,
            pub channels: u8,
            pub sbr: bool,
            pub ps: bool,
            pub aot: u8,
        }
        impl Asc {
            pub fn output_rate(&self) -> u32 {
                self.sample_rate
            }
        }
        pub fn parse_asc(_asc: &[u8]) -> Result<Asc, &'static str> {
            Err("h264diff stub")
        }
        #[allow(clippy::too_many_arguments)]
        pub fn decode_track(_rate: u32, _ch: u8, _asc: &[u8], _bytes: &[u8], _samples: &[(usize, usize)]) -> Result<super::Audio, &'static str> {
            Err("h264diff stub")
        }
    }
}
#[path = "../../../kernel/src/video/mod.rs"]
mod video;

// Kernel `video::frame_from_yuv` optionally fans out over aarch64 SMP. The
// host harness is a single-process std binary — provide a no-op arch facade
// so the path compiles and always takes the single-threaded convert.
#[allow(dead_code)]
mod arch {
    pub fn online_cpus() -> usize {
        1
    }
    /// # Safety
    /// Never called when `online_cpus() == 1`.
    pub unsafe fn parallel_for(
        _n: usize,
        _chunk: usize,
        _f: unsafe fn(usize, usize, *mut u8),
        _ctx: *mut u8,
    ) {
    }
}

use std::io::Write;
use video::h264;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: h264diff <file.mp4> [--stream[=jump]]");
    let bytes = std::fs::read(path).expect("read file");
    if args.iter().any(|a| a == "--bench") {
        // Streaming throughput: decode on demand (no full-clip RGB cache).
        let mut dec = video::StreamDecoder::open(bytes).expect("open");
        eprintln!(
            "bench: {}x{} frames={} (stream decode)",
            dec.src_w,
            dec.src_h,
            dec.frame_count()
        );
        let t0 = std::time::Instant::now();
        let n = dec.frame_count().min(60);
        let mut ok = 0usize;
        for i in 0..n {
            if dec.seek_decode(i) {
                ok += 1;
            }
            if i % 5 == 0 {
                eprint!(".");
            }
        }
        let ms = t0.elapsed().as_millis().max(1);
        eprintln!(
            "\nbench: {} frames in {} ms ({:.1} ms/frame, {:.2} fps)",
            ok,
            ms,
            ms as f64 / ok.max(1) as f64,
            ok as f64 * 1000.0 / ms as f64
        );
        return;
    }

    if args.iter().any(|a| a == "--adts") {
        let t0 = std::time::Instant::now();
        let a = audio::decode(&bytes).expect("audio::decode ADTS");
        let peak = a.pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        let nz = a.pcm.iter().filter(|&&s| s != 0).count();
        eprintln!("adts: {} samples @ {} Hz peak={} nonzero={} in {:?}",
            a.pcm.len(), a.rate, peak, nz, t0.elapsed());
        return;
    }
    if args.iter().any(|a| a == "--audio") {
        let info = video::audio_info(&bytes).expect("audio track");
        eprintln!("audio: {} {} Hz {}ch decodable={}", info.codec, info.sample_rate, info.channels, info.decodable);
        let t0 = std::time::Instant::now();
        let a = video::decode_audio(&bytes).expect("decode_audio");
        let peak = a.pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
        let nz = a.pcm.iter().filter(|&&s| s != 0).count();
        eprintln!("pcm: {} samples @ {} Hz ({:.1}s) peak={} nonzero={} in {:?}",
            a.pcm.len(), a.rate, a.duration_ms() as f64 / 1000.0, peak, nz, t0.elapsed());
        return;
    }

    // --stream[=N]: decode via the kernel's StreamDecoder exactly as the player
    // does (on-demand seek_decode), optionally jumping N frames per step to
    // mimic pump_video pacing. Outputs the displayed RGB (0x00RRGGBB, LE u32).
    if let Some(sa) = args.iter().find(|a| a.starts_with("--stream")) {
        let jump: usize = sa.split('=').nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut dec = video::StreamDecoder::open(bytes).expect("open stream");
        let n = dec.frame_count();
        eprintln!("h264diff: STREAM frames={} jump={}", n, jump);
        let mut out = std::io::stdout();
        let mut i = 0usize;
        while i < n {
            dec.seek_decode(i);
            let f = dec.cur_frame().expect("frame");
            for &p in &f.pixels {
                out.write_all(&p.to_le_bytes()).unwrap();
            }
            i += jump;
        }
        return;
    }
    if args.iter().any(|a| a == "--idct8test") {
        // stdin: lines of 64 raster dequantised coeffs -> stdout: 64 residuals.
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line.unwrap();
            let vals: Vec<i32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if vals.len() != 64 {
                break;
            }
            let mut blk = [0i32; 64];
            blk.copy_from_slice(&vals);
            h264::transform::idct8_residual(&mut blk);
            let out: Vec<String> = blk.iter().map(|v| v.to_string()).collect();
            println!("{}", out.join(" "));
        }
        return;
    }
    // Demux either container → (avcc, samples).
    let (config, samples): (video::mp4::CodecConfig, Vec<video::mp4::Sample>) = if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        let t = video::mkv::parse(&bytes).expect("demux mkv");
        (t.config, t.samples)
    } else {
        let t = video::mp4::parse(&bytes).expect("demux mp4");
        (t.config, t.samples)
    };
    let avcc = match config {
        video::mp4::CodecConfig::Avc(a) => a,
        other => panic!("h264diff: not an H.264 track ({}) — use videodiff", other.codec_name()),
    };
    let sps = h264::parse_sps(&video::bits::unescape_rbsp(&avcc.sps[0][1..])).expect("sps");
    let pps = h264::parse_pps(&video::bits::unescape_rbsp(&avcc.pps[0][1..])).expect("pps");
    eprintln!("h264diff: {}x{} samples={} lenSize={} cabac={}", sps.width(), sps.height(), samples.len(), avcc.length_size, pps.entropy_coding_mode);
    if pps.entropy_coding_mode {
        // CABAC path: decode all AUs in decode order via H264Dec, then write
        // YUV in display (cts) order for the PyAV frame-by-frame diff.
        let mut dec = h264::decoder_cabac::H264Dec::new(sps.clone(), pps.clone()).expect("cabac init");
        dec.trace = std::env::var("H264_TRACE").is_ok();
        dec.no_deblock = std::env::var("H264_NO_DEBLOCK").is_ok();
        let mut decoded: Vec<(u64, std::rc::Rc<h264::decoder_cabac::Pic>)> = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            let data = &bytes[s.offset..s.offset + s.size];
            let mut slices: Vec<(Vec<u8>, bool, u8)> = Vec::new();
            for nal in h264::split_avcc(data, avcc.length_size) {
                if nal.kind.is_slice() {
                    slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr, nal.ref_idc));
                }
            }
            if slices.is_empty() {
                continue;
            }
            match dec.decode_au(&slices) {
                Ok(pic) => {
                    eprintln!("h264diff: sample {} ok poc={} dpb={}", i, pic.poc, dec.dpb_len());
                    if dec.trace && std::env::var("H264_TRACE_N").map(|v| i < v.parse().unwrap_or(0)).unwrap_or(false) {
                        for l in &dec.trace_log {
                            eprintln!("T {}", l);
                        }
                    }
                    decoded.push((s.cts, pic));
                }
                Err(e) => {
                    eprintln!("h264diff: decode error at sample {}: {}", i, e);
                    break;
                }
            }
        }
        if std::env::var("H264_DECODE_ORDER").is_err() {
            decoded.sort_by_key(|(cts, _)| *cts);
        }
        let mut out = std::io::stdout();
        for (_, pic) in &decoded {
            out.write_all(&pic.f.y).unwrap();
            out.write_all(&pic.f.cb).unwrap();
            out.write_all(&pic.f.cr).unwrap();
        }
        eprintln!("h264diff: wrote {} frame(s) (display order)", decoded.len());
        return;
    }
    let mut out = std::io::stdout();
    let mut n = 0;
    let mut reference: Option<h264::decoder::DecodedFrame> = None;
    for s in &samples {
        let data = &bytes[s.offset..s.offset + s.size];
        let mut slices: Vec<(Vec<u8>, bool)> = Vec::new();
        for nal in h264::split_avcc(data, avcc.length_size) {
            if nal.kind.is_slice() {
                slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr));
            }
        }
        if slices.is_empty() {
            continue;
        }
        let df = h264::decoder::decode_access_unit(&sps, &pps, &slices, reference.as_ref()).expect("decode");
        out.write_all(&df.y).unwrap();
        out.write_all(&df.cb).unwrap();
        out.write_all(&df.cr).unwrap();
        reference = Some(df);
        n += 1;
    }
    eprintln!("h264diff: wrote {} frame(s)", n);
}
