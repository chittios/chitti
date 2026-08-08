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
    // Encode a synthetic desktop-like clip with the kernel recorder encoder,
    // mux it, and open it through StreamDecoder (the same path as `/open`).
    // Exit 0 only when frame 0 paints non-black.
    if args.iter().any(|a| a == "--record-selftest") {
        record_selftest(&args);
        return;
    }
    let path = args.get(1).expect("usage: h264diff <file.mp4> [--stream[=jump]] | --record-selftest");
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

fn record_selftest(args: &[String]) {
    use video::h264::encoder::Encoder;
    use video::mp4_mux::{self, Sample};

    // Match a real `/record` geometry (50% of a typical panel, MB-aligned).
    let (w, h) = args
        .iter()
        .find(|a| a.starts_with("--size="))
        .and_then(|a| {
            let s = a.trim_start_matches("--size=");
            let mut it = s.split('x');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .unwrap_or((64usize, 48usize));
    eprintln!("record-selftest: encoding {w}x{h}");
    let mut enc = Encoder::new(w, h, 28).expect("encoder");
    let mut px = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = if y < 8.max(h / 16) {
                0x00ff_ffff // white bar
            } else if (x / 16 + y / 16) % 2 == 0 {
                0x00cc_785c // brand terracotta
            } else {
                0x00_30_40_80
            };
        }
    }
    let mut samples = Vec::new();
    for i in 0..3 {
        let au = enc.encode_rgb32(&px, i == 0).expect("encode");
        eprintln!(
            "record-selftest: frame {i} au_len={} first8={:02x?}",
            au.len(),
            &au[..au.len().min(8)]
        );
        samples.push(Sample {
            bytes: au,
            duration: 200,
            sync: i == 0,
        });
    }

    // Native CAVLC path on raw sample0.
    {
        let sps = h264::parse_sps(&video::bits::unescape_rbsp(&enc.sps_nal[1..])).unwrap();
        let pps = h264::parse_pps(&video::bits::unescape_rbsp(&enc.pps_nal[1..])).unwrap();
        let nals = h264::split_avcc(&samples[0].bytes, 4);
        assert!(!nals.is_empty(), "no NAL in sample0");
        let df = h264::decoder::decode_access_unit(&sps, &pps, &[(nals[0].rbsp(), true)], None)
            .expect("native decode sample0");
        let mean_y = df.y.iter().map(|&p| p as u32).sum::<u32>() / df.y.len() as u32;
        let bright = df.y.iter().filter(|&&p| p > 200).count();
        eprintln!(
            "record-selftest: native mean_y={mean_y} y0={} bright={}/{} (want white bar)",
            df.y[0],
            bright,
            df.y.len()
        );
        // Pure mid-gray prediction has no bright pixels; a white bar must survive.
        if bright < df.y.len() / 64 {
            panic!(
                "native decode lost the white bar (bright={bright}, mean_y={mean_y}) — residual empty?"
            );
        }
    }

    let file = mp4_mux::mux_avc(
        w as u32,
        h as u32,
        1000,
        &enc.sps_nal,
        &enc.pps_nal,
        &samples,
    )
    .expect("mux");
    eprintln!("record-selftest: muxed {} bytes", file.len());

    // Demux sample offsets must land inside mdat, not at file start (ftyp).
    {
        let track = video::mp4::parse(&file).expect("demux");
        let s0 = &track.samples[0];
        eprintln!(
            "record-selftest: sample0 offset={} size={} file_len={}",
            s0.offset,
            s0.size,
            file.len()
        );
        assert!(s0.offset > 8, "sample0 offset {} looks like stco=0 bug", s0.offset);
        let head = &file[s0.offset..s0.offset + s0.size.min(8)];
        eprintln!("record-selftest: sample0 first8={:02x?}", head);
        // Must not be 'ftyp'
        assert!(
            !(head.len() >= 8 && &head[4..8] == b"ftyp"),
            "sample0 is ftyp — stco still wrong"
        );
    }

    let mut dec = video::StreamDecoder::open(file).expect("StreamDecoder open");
    eprintln!("record-selftest: backend={}", dec.backend);
    assert!(
        dec.seek_decode(0),
        "seek_decode(0) failed backend={}",
        dec.backend
    );
    let f = dec.cur_frame().expect("cur_frame");
    let nonzero = f.pixels.iter().filter(|&&p| p & 0x00ff_ffff != 0).count();
    let mean_r: u64 = f
        .pixels
        .iter()
        .map(|&p| ((p >> 16) & 0xff) as u64)
        .sum::<u64>()
        / f.pixels.len() as u64;
    let mean_g: u64 = f
        .pixels
        .iter()
        .map(|&p| ((p >> 8) & 0xff) as u64)
        .sum::<u64>()
        / f.pixels.len() as u64;
    let mean_b: u64 = f
        .pixels
        .iter()
        .map(|&p| (p & 0xff) as u64)
        .sum::<u64>()
        / f.pixels.len() as u64;
    eprintln!(
        "record-selftest: frame0 {}x{} nonzero={}/{} mean_rgb=({},{},{}) p0={:08x}",
        f.w,
        f.h,
        nonzero,
        f.pixels.len(),
        mean_r,
        mean_g,
        mean_b,
        f.pixels[0]
    );
    if nonzero <= f.pixels.len() / 4 {
        panic!(
            "frame is nearly black ({nonzero}/{}) — same failure as /record → /open",
            f.pixels.len()
        );
    }
    assert!(
        dec.seek_decode(1),
        "seek_decode(1) failed after P frame"
    );
    eprintln!("record-selftest: OK");
}
