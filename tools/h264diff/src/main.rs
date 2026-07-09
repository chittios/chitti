//! Decode an MP4's first keyframe with the KERNEL's own H.264 decoder (mounted
//! via #[path]) and write its YUV planes to stdout, so diff.py can compare them
//! bit-for-bit against PyAV/ffmpeg. The onnxdiff/cortexdiff pattern.
extern crate alloc;

#[path = "../../../kernel/src/video/mod.rs"]
mod video;

use std::io::Write;
use video::h264;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: h264diff <file.mp4> [--stream[=jump]]");
    let bytes = std::fs::read(path).expect("read file");
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
    // Demux either container → (avcc, samples).
    let (avcc, samples): (video::mp4::AvcC, Vec<video::mp4::Sample>) = if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        let t = video::mkv::parse(&bytes).expect("demux mkv");
        (t.avcc, t.samples)
    } else {
        let t = video::mp4::parse(&bytes).expect("demux mp4");
        (t.avcc, t.samples)
    };
    let sps = h264::parse_sps(&video::bits::unescape_rbsp(&avcc.sps[0][1..])).expect("sps");
    let pps = h264::parse_pps(&video::bits::unescape_rbsp(&avcc.pps[0][1..])).expect("pps");
    eprintln!("h264diff: {}x{} samples={} lenSize={}", sps.width(), sps.height(), samples.len(), avcc.length_size);
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
