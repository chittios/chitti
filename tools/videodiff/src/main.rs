//! Run the **kernel's own** video stack natively on the host, so its output can
//! be checked against PyAV/ffmpeg without a QEMU round trip — the
//! onnxdiff/cortexdiff pattern, generalised past H.264.
//!
//! The kernel modules are *mounted*, not copied (`#[path]`), so there is one
//! implementation and a bring-up fix here cannot diverge from the shipped one.
//! What this harness therefore tests is the decoder; what it cannot test is the
//! kernel plumbing around it (heap, SMP, framebuffer).
//!
//! Usage:
//!   videodiff probe   <file>          — demux + parameter sets, no pixels
//!   videodiff headers <file> [n]      — per-frame header dump (n frames)
//!   videodiff yuv     <file> [n]      — decode n frames, raw I420 to stdout
//!   videodiff hls     <x.m3u8> [n]    — fetch a local HLS VOD (segments beside
//!                                       the playlist), demux + decode n frames;
//!                                       raw RGB24 to stdout when redirected
//!
//! `diff.py` drives the pixel dumps against PyAV.

extern crate alloc;

// --- Stubs for the kernel surface `video/mod.rs` reaches outside itself. -----
//
// Only the *shape* is needed: the audio track is a separate decoder with its own
// harness, and the SMP facade collapses to single-threaded on the host.

mod audio {
    pub struct Audio {
        pub rate: u32,
        pub pcm: Vec<i16>,
    }
    impl Audio {
        pub fn duration_ms(&self) -> u64 {
            if self.rate == 0 {
                0
            } else {
                self.pcm.len() as u64 * 1000 / self.rate as u64
            }
        }
    }
    pub fn decode(_bytes: &[u8]) -> Result<Audio, &'static str> {
        Err("videodiff: audio decode is out of scope here (see h264diff --adts)")
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
            Err("videodiff stub")
        }
        #[allow(clippy::too_many_arguments)]
        pub fn decode_track(
            _rate: u32,
            _ch: u8,
            _asc: &[u8],
            _bytes: &[u8],
            _samples: &[(usize, usize)],
        ) -> Result<super::Audio, &'static str> {
            Err("videodiff stub")
        }
    }
}

/// The arch-neutral SMP facade `video/mt.rs` calls. One core here, so
/// `parallel_rows` always takes its single-threaded path.
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

#[path = "../../../kernel/src/video/mod.rs"]
mod video;

use std::io::Write;
use video::mp4::CodecConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: videodiff <probe|headers|yuv|vp9frame|vp9seq|hevcseq|hevcsh|hls> <file> [frames]");
            std::process::exit(2);
        }
    };
    let bytes = std::fs::read(path).expect("read file");
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    match cmd {
        "probe" => probe(&bytes),
        "headers" => headers(&bytes, n),
        "yuv" => yuv(bytes, n),
        "vp9frame" => vp9frame(&bytes, n),
        "vp9seq" => vp9seq(&bytes, n),
        "hevcseq" => hevcseq(&bytes, n),
        "hevcsh" => hevcsh(&bytes, n),
        "hls" => hls_open(path, &bytes, n),
        _ => {
            eprintln!("usage: videodiff <probe|headers|yuv|vp9frame|vp9seq|hevcseq|hevcsh|hls> <file> [frames]");
            std::process::exit(2);
        }
    }
}

/// Load an HLS VOD playlist from disk (segments resolved relative to the
/// playlist path) and open it with StreamDecoder — the same path `/open` uses
/// after network fetch.
fn hls_open(playlist_path: &str, bytes: &[u8], limit: usize) {
    let text = std::str::from_utf8(bytes).expect("utf8 playlist");
    // Base URL = file path for relative segment joins.
    let base = playlist_path.to_string();
    let base_dir = std::path::Path::new(playlist_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let fetch = |url: &str| -> Result<Vec<u8>, String> {
        let p = if url.starts_with('/') || url.contains("://") {
            std::path::PathBuf::from(url)
        } else {
            base_dir.join(url)
        };
        std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))
    };
    let (vod, _) = video::hls::resolve_vod(text, &base, fetch).expect("resolve_vod");
    eprintln!(
        "hls: {} segment(s), duration {} ms",
        vod.segments.len(),
        vod.duration_ms
    );
    let loaded = video::hls::load_vod(&vod, fetch, |i, n| {
        eprintln!("  segment {}/{}", i + 1, n);
        Ok(())
    })
    .expect("load_vod");
    eprintln!(
        "hls: demuxed {} samples, {} bytes ({})",
        loaded.samples.len(),
        loaded.bytes.len(),
        loaded.container
    );
    let mut dec = video::StreamDecoder::open_hls(loaded).expect("open_hls");
    let n = dec.frame_count().min(limit);
    eprintln!(
        "hls: {}x{} frames={} backend={}",
        dec.src_w, dec.src_h, dec.frame_count(), dec.backend
    );
    // Frames go to stdout as raw RGB24 in *display* order when stdout is
    // redirected, so `diff.py` can compare them against PyAV's decode of the
    // same segments — the only way to catch an access-unit split that still
    // decodes. A terminal gets the summary alone rather than a screen of bytes.
    use std::io::{IsTerminal, Write};
    let out = std::io::stdout();
    let dump = !out.is_terminal();
    let mut w = out.lock();
    for i in 0..n {
        if !dec.seek_decode(i) {
            eprintln!("frame {} did not decode", i);
            std::process::exit(1);
        }
        if dump {
            let f = dec.cur_frame().expect("decoded frame");
            let mut rgb = Vec::with_capacity(f.pixels.len() * 3);
            for px in &f.pixels {
                rgb.push((px >> 16) as u8);
                rgb.push((px >> 8) as u8);
                rgb.push(*px as u8);
            }
            w.write_all(&rgb).expect("write rgb");
        }
    }
    let _ = w.flush();
    eprintln!("hls: decoded {} frame(s) ok", n);
}

fn probe(bytes: &[u8]) {
    match video::probe(bytes) {
        Ok(i) => {
            println!("container   {}", i.container);
            println!("codec       {}", i.codec);
            println!("size        {}x{}", i.width, i.height);
            println!("profile     {}", i.profile_idc);
            println!("level       {}", i.level_idc);
            println!("frames      {}", i.frame_count);
            println!("duration_ms {}", i.duration_ms);
            println!("decodable   {}", i.decodable);
        }
        Err(e) => {
            eprintln!("probe failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// Dump per-frame header facts. This is the bring-up view: it answers "did the
/// demuxer find the frames, and does each one's header parse" separately from
/// "do the pixels match", which are different failures with the same symptom.
fn headers(bytes: &[u8], limit: usize) {
    let (config, samples) = demux(bytes);
    println!("codec {} samples {}", config.codec_name(), samples.len());
    match &config {
        CodecConfig::Vp9(_) => {
            let mut refs: video::vp9::RefSizes = [(0, 0); video::vp9::NUM_REF_FRAMES];
            for (i, s) in samples.iter().take(limit).enumerate() {
                let data = &bytes[s.offset..s.offset + s.size];
                let parts = match video::vp9::split_superframe(data) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("{:4} SUPERFRAME ERROR {}", i, e);
                        continue;
                    }
                };
                for (k, &(a, b)) in parts.iter().enumerate() {
                    match video::vp9::parse_frame_header(&data[a..b], &refs) {
                        Ok(h) => {
                            if let Some(slot) = h.show_existing_frame {
                                println!("{:4}.{} show_existing slot={}", i, k, slot);
                                continue;
                            }
                            println!(
                                "{:4}.{} {} {}x{} show={} q={} lf={} tiles={}x{} refresh={:#04x} hdr={}+{}",
                                i,
                                k,
                                if h.key_frame { "KEY  " } else if h.intra_only { "INTRA" } else { "INTER" },
                                h.width,
                                h.height,
                                h.show_frame as u8,
                                h.quant.base_q_idx,
                                h.loop_filter.level,
                                1 << h.tiles.cols_log2,
                                1 << h.tiles.rows_log2,
                                h.refresh_frame_flags,
                                h.uncompressed_header_bytes,
                                h.header_size_in_bytes,
                            );
                            // A frame updates the slots it refreshes, which is
                            // what later frames read their size from.
                            for slot in 0..video::vp9::NUM_REF_FRAMES {
                                if h.refresh_frame_flags & (1 << slot) != 0 {
                                    refs[slot] = (h.width, h.height);
                                }
                            }
                        }
                        Err(e) => println!("{:4}.{} HEADER ERROR {}", i, k, e),
                    }
                }
            }
        }
        CodecConfig::Hevc(hvcc) => {
            let sps = video::hevc::parse_sps(&video::bits::unescape_rbsp(&hvcc.sps[0][2..])).expect("sps");
            let pps = video::hevc::parse_pps(&video::bits::unescape_rbsp(&hvcc.pps[0][2..])).expect("pps");
            println!(
                "sps {}x{} ctb={} profile={} level={} sao={} amp={} tmvp={} | pps wpp={} tiles={} qp={}",
                sps.width(),
                sps.height(),
                sps.ctb_size(),
                sps.ptl.profile_name(),
                sps.ptl.level_idc,
                sps.sao_enabled as u8,
                sps.amp_enabled as u8,
                sps.temporal_mvp_enabled as u8,
                pps.entropy_coding_sync_enabled as u8,
                pps.tiles_enabled as u8,
                pps.init_qp,
            );
            for (i, s) in samples.iter().take(limit).enumerate() {
                let data = &bytes[s.offset..s.offset + s.size];
                for nal in video::hevc::split_hvcc(data, hvcc.length_size) {
                    if !nal.kind.is_slice() {
                        continue;
                    }
                    match video::hevc::parse_slice_header(&nal.rbsp(), nal.kind, &sps, &pps) {
                        Ok(h) => println!(
                            "{:4} nal={:2} {:?} poc_lsb={:3} qp={:3} first={} addr={} sao={}{} rps(-{},+{}) merge={} data@{} eps={} l0={} l1={}",
                            i,
                            nal.kind.code(),
                            h.slice_type,
                            h.pic_order_cnt_lsb,
                            h.qp,
                            h.first_slice_in_pic as u8,
                            h.segment_address,
                            h.sao_luma as u8,
                            h.sao_chroma as u8,
                            h.st_rps.num_negative(),
                            h.st_rps.num_positive(),
                            h.max_num_merge_cand(),
                            h.data_byte_offset,
                            h.entry_point_offsets.len(),
                            h.num_ref_idx_l0,
                            h.num_ref_idx_l1,
                        ),
                        Err(e) => println!("{:4} nal={:2} SLICE HEADER ERROR {}", i, nal.kind.code(), e),
                    }
                }
            }
        }
        CodecConfig::Avc(avcc) => {
            for (i, s) in samples.iter().take(limit).enumerate() {
                let data = &bytes[s.offset..s.offset + s.size];
                for nal in video::h264::split_avcc(data, avcc.length_size) {
                    println!("{:4} nal={:?} ref_idc={}", i, nal.kind, nal.ref_idc);
                }
            }
        }
    }
}

fn yuv(bytes: Vec<u8>, limit: usize) {
    let mut dec = match video::StreamDecoder::open(bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open failed: {}", e);
            std::process::exit(1);
        }
    };
    let n = dec.frame_count().min(limit);
    eprintln!("videodiff: {}x{} frames={} backend={}", dec.src_w, dec.src_h, dec.frame_count(), dec.backend);
    let out = std::io::stdout();
    let mut w = out.lock();
    for i in 0..n {
        if !dec.seek_decode(i) {
            eprintln!("frame {} did not decode", i);
            break;
        }
        if let Some(f) = dec.cur_frame() {
            for p in &f.pixels {
                w.write_all(&p.to_le_bytes()).unwrap();
            }
        }
    }
}

/// Decode VP9 frames with the kernel's own decoder and write raw I420 to
/// stdout, so `diff.py` can compare them plane-for-plane against PyAV. This is
/// the bring-up path: it reports *where* a frame stopped decoding rather than
/// failing the whole run, because during bring-up most frames do not decode.
fn vp9frame(bytes: &[u8], limit: usize) {
    let (config, samples) = demux(bytes);
    if !matches!(config, CodecConfig::Vp9(_)) {
        eprintln!("not a VP9 track");
        std::process::exit(1);
    }
    let mut refs: video::vp9::RefSizes = [(0, 0); video::vp9::NUM_REF_FRAMES];
    let mut fc = video::vp9::header::FrameContext::default();
    let out = std::io::stdout();
    let mut w = out.lock();
    let mut done = 0usize;
    for (i, s) in samples.iter().enumerate() {
        if done >= limit {
            break;
        }
        let data = &bytes[s.offset..s.offset + s.size];
        let parts = match video::vp9::split_superframe(data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}: superframe: {}", i, e);
                break;
            }
        };
        for &(a, b) in &parts {
            let h = match video::vp9::parse_frame_header(&data[a..b], &refs) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("{}: header: {}", i, e);
                    return;
                }
            };
            for slot in 0..video::vp9::NUM_REF_FRAMES {
                if h.refresh_frame_flags & (1 << slot) != 0 {
                    refs[slot] = (h.width, h.height);
                }
            }
            if std::env::var("VP9_MODES").is_ok() {
                match video::vp9::decoder::decode_intra_frame_state(&data[a..b], &h, &mut fc) {
                    Ok(st) => {
                        eprintln!("{}: mode grid {}x{} (bs/ymode/uvmode/tx/skip)", i, st.mi_cols, st.mi_rows);
                        for row in 0..st.mi_rows {
                            let mut line = String::new();
                            for col in 0..st.mi_cols {
                                let m = &st.mi[row * st.mi_cols + col];
                                if m.block_size < 3 {
                                    line.push_str(&format!(
                                        "{:>2}/[{}{}{}{}]/{}/{}/{} ",
                                        m.block_size, m.sub_modes[0], m.sub_modes[1],
                                        m.sub_modes[2], m.sub_modes[3],
                                        m.uv_mode, m.tx_size, m.skip as u8
                                    ));
                                } else {
                                    line.push_str(&format!(
                                        "{:>2}/{}/{}/{}/{} ",
                                        m.block_size, m.y_mode, m.uv_mode, m.tx_size, m.skip as u8
                                    ));
                                }
                            }
                            eprintln!("  {}", line);
                        }
                    }
                    Err(e) => eprintln!("{}: decode: {}", i, e),
                }
                return;
            }
            match video::vp9::decoder::decode_intra_frame(&data[a..b], &h, &mut fc) {
                Ok(f) => {
                    eprintln!("{}: decoded {}x{} ({} luma bytes)", i, f.w, f.h, f.y.len());
                    w.write_all(&f.y).unwrap();
                    w.write_all(&f.cb).unwrap();
                    w.write_all(&f.cr).unwrap();
                    done += 1;
                }
                Err(e) => {
                    eprintln!("{}: decode: {}", i, e);
                    return;
                }
            }
        }
    }
}

/// Decode a whole VP9 sequence with the kernel's `Vp9Decoder` and write every
/// **shown** frame as raw I420, so `diff.py` can compare the sequence — which
/// is the only way inter prediction can be checked at all.
fn vp9seq(bytes: &[u8], limit: usize) {
    let (config, samples) = demux(bytes);
    if !matches!(config, CodecConfig::Vp9(_)) {
        eprintln!("not a VP9 track");
        std::process::exit(1);
    }
    let mut dec = video::vp9::decoder::Vp9Decoder::new();
    let out = std::io::stdout();
    let mut w = out.lock();
    let mut shown = 0usize;
    for (i, s) in samples.iter().enumerate() {
        if shown >= limit {
            break;
        }
        let data = &bytes[s.offset..s.offset + s.size];
        let parts = match video::vp9::split_superframe(data) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}: superframe: {}", i, e);
                return;
            }
        };
        for &(a, b) in &parts {
            let mut mi_out = None;
            let res = if std::env::var("VP9_MODES").is_ok() {
                dec.decode_frame_debug(&data[a..b], &mut mi_out)
            } else {
                dec.decode_frame(&data[a..b])
            };
            if let Some((cols, rows, mi)) = mi_out {
                eprintln!("--- frame {} modes {}x{} (bs/ymode/ref/tx/skip/filt)", i, cols, rows);
                let r0: usize = std::env::var("VP9_R0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                let c0: usize = std::env::var("VP9_C0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                for row in r0..(r0 + 5).min(rows) {
                    let mut line = String::new();
                    for col in c0..(c0 + 6).min(cols) {
                        let m = &mi[row * cols + col];
                        line.push_str(&format!(
                            "{:>2}/{:>2}/{}/{}/{}/{} ",
                            m.block_size, m.y_mode, m.ref_frame[0], m.tx_size, m.skip as u8, m.interp_filter
                        ));
                    }
                    eprintln!("  {}", line);
                }
            }
            match res {
                Ok(Some(f)) => {
                    eprintln!("{}: shown {}x{}", i, f.w, f.h);
                    w.write_all(&f.y).unwrap();
                    w.write_all(&f.cb).unwrap();
                    w.write_all(&f.cr).unwrap();
                    shown += 1;
                }
                Ok(None) => eprintln!("{}: hidden frame", i),
                Err(e) => {
                    eprintln!("{}: decode: {}", i, e);
                    return;
                }
            }
        }
    }
}

fn demux(bytes: &[u8]) -> (CodecConfig, Vec<video::mp4::Sample>) {
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        let t = video::mkv::parse(bytes).expect("demux mkv");
        (t.config, t.samples)
    } else {
        let t = video::mp4::parse(bytes).expect("demux mp4");
        (t.config, t.samples)
    }
}

/// Write one plane: for 8-bit, raw bytes; for 10/12-bit, little-endian u16
/// samples (yuv420p{10,12}le / yuv422p* / yuv444p* layout matching plane size).
fn write_hevc_plane(w: &mut impl Write, plane: &[u16], bit_depth: u32) {
    if bit_depth <= 8 {
        let bytes: Vec<u8> = plane.iter().map(|&s| s as u8).collect();
        w.write_all(&bytes).unwrap();
    } else {
        let mut bytes = Vec::with_capacity(plane.len() * 2);
        for &s in plane {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        w.write_all(&bytes).unwrap();
    }
}

/// Decode an HEVC track and write raw I420 (or 10/12-bit LE) to stdout, so
/// a host harness can compare it sample-for-sample against PyAV.
fn hevcseq(bytes: &[u8], limit: usize) {
    let (config, samples) = demux(bytes);
    let hvcc = match &config {
        CodecConfig::Hevc(h) => h.clone(),
        _ => {
            eprintln!("not an HEVC track");
            std::process::exit(1);
        }
    };
    let mut dec = video::hevc::decoder::HevcDecoder::new();
    dec.trace_on = std::env::var("HEVC_TRACE").is_ok();
    dec.skip_loop_filter = std::env::var("HEVC_NO_FILTER").is_ok();
    if let Err(e) = dec.set_parameter_sets(&hvcc.vps, &hvcc.sps, &hvcc.pps) {
        eprintln!("parameter sets: {}", e);
        std::process::exit(1);
    }
    let out = std::io::stdout();
    let mut w = out.lock();
    let mut shown = 0usize;
    for (i, s) in samples.iter().enumerate() {
        if shown >= limit {
            break;
        }
        let data = &bytes[s.offset..s.offset + s.size];
        let nals = video::hevc::split_hvcc(data, hvcc.length_size);
        match dec.decode_au(&nals) {
            Ok(frames) => {
                for f in frames {
                    write_hevc_plane(&mut w, &f.y, f.bit_depth);
                    write_hevc_plane(&mut w, &f.cb, f.bit_depth);
                    write_hevc_plane(&mut w, &f.cr, f.bit_depth);
                    shown += 1;
                    if shown >= limit {
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("sample {}: {}", i, e);
                if dec.trace_on {
                    eprintln!("--- last {} trace entries ---", dec.trace.len().min(20));
                    for (x, y, l2, c, mode, cbf) in dec.trace.iter().rev().take(20).collect::<Vec<_>>().into_iter().rev() {
                        if *x == 0xFFFC {
                            eprintln!("CU ({:2},{:2}) {}x{} d{} @byte {}", l2, c, 1u32 << mode, 1u32 << mode, cbf, y);
                        } else if *x == 0xFFF0 {
                            eprintln!("QP_FAIL @byte {} cu=({},{}) log2={}", y, l2, c, mode);
                        } else {
                            eprintln!("tu ({:3},{:3}) {}x{} c{} mode {:2} cbf {:03b}", x, y, 1 << l2, 1 << l2, c, mode, cbf);
                        }
                    }
                }
                std::process::exit(1);
            }
        }
    }
    for f in dec.flush() {
        if shown >= limit {
            break;
        }
        write_hevc_plane(&mut w, &f.y, f.bit_depth);
        write_hevc_plane(&mut w, &f.cb, f.bit_depth);
        write_hevc_plane(&mut w, &f.cr, f.bit_depth);
        shown += 1;
    }
    eprintln!("hevcseq: wrote {} frame(s), {} CTU(s)", shown, dec.ctus);
    {
        let rb = video::bits::unescape_rbsp(&hvcc.sps[0][2..]);
        let sp = video::hevc::parse_sps(&rb).unwrap();
        let rp = video::bits::unescape_rbsp(&hvcc.pps[0][2..]);
        let pp = video::hevc::parse_pps(&rp).unwrap();
        eprintln!("SPS coded={}x{} disp={}x{} win={},{},{},{}", sp.pic_width_in_luma_samples, sp.pic_height_in_luma_samples, sp.width(), sp.height(), sp.conf_win_left, sp.conf_win_right, sp.conf_win_top, sp.conf_win_bottom);
        eprintln!("SPS mincb={} ctb={} mintb={} maxtb={} hdepth_i={} hdepth_p={} amp={} sao={} pcm={} sis={} scal={} tmvp={}",
            sp.log2_min_cb_size, sp.log2_ctb_size, sp.log2_min_tb_size, sp.log2_max_tb_size,
            sp.max_transform_hierarchy_depth_intra, sp.max_transform_hierarchy_depth_inter,
            sp.amp_enabled as u8, sp.sao_enabled as u8, sp.pcm_enabled as u8,
            sp.strong_intra_smoothing as u8, sp.scaling_list_enabled as u8, sp.temporal_mvp_enabled as u8);
        eprintln!("PPS tskip={} sdh={} tqb={} cuqp={} qpdepth={} cbo={} cro={} wp={} wbp={} sign={} lmpl={} dep={}",
            pp.transform_skip_enabled as u8, pp.sign_data_hiding_enabled as u8,
            pp.transquant_bypass_enabled as u8, pp.cu_qp_delta_enabled as u8, pp.diff_cu_qp_delta_depth,
            pp.cb_qp_offset, pp.cr_qp_offset, pp.weighted_pred as u8, pp.weighted_bipred as u8,
            pp.sign_data_hiding_enabled as u8, pp.log2_parallel_merge_level, pp.dependent_slice_segments_enabled as u8);

    }
    if dec.trace_on {
        for (x, y, l2, c, mode, cbf) in dec.trace.iter().take(4000) {
            if *x == 0xFFFC {
                eprintln!("CU ({:2},{:2}) {}x{} d{} @byte {}", l2, c, 1u32 << mode, 1u32 << mode, cbf, y);
            } else if *x == 0xFFFD {
                eprintln!("SLICE type={} consumed {} of {} bytes", mode, y, (*l2 as usize) | ((*c as usize) << 8));
            } else if *x == 0xFFFE {
                eprintln!("  pu ({:2},{:2}) flag={} mv=({},{}) merge={}", y & 0xff, y >> 8, l2, *c as i8, *mode as i8, cbf);
            } else if *x == 0xFFFF {
                eprintln!("  coef dc={} maxxy={} c{} scan={} ncoeff={}", *y as i16, l2, c, mode>>4, cbf);
            } else if *x == 0xFFF1 {
                eprintln!("  last=({},{}) scan={} c{} log2={}", y, l2, c, mode, cbf);
            } else if *x == 0xFFA0 {
                eprintln!("  mode@({},{}) mode={} candL={} candA={} prev={}", y&0xff, y>>8, l2, c, mode, cbf);
            } else {
                eprintln!("tu ({:3},{:3}) {}x{} c{} mode {:2} cbf {:03b}", x, y, 1 << l2, 1 << l2, c, mode, cbf);
            }
        }
    }
}

/// Dump parsed HEVC slice headers with entry points and data offset.
fn hevcsh(bytes: &[u8], limit: usize) {
    let (config, samples) = demux(bytes);
    let hvcc = match &config {
        CodecConfig::Hevc(h) => h.clone(),
        _ => { eprintln!("not HEVC"); std::process::exit(1); }
    };
    let sps_rbsp = video::bits::unescape_rbsp(&hvcc.sps[0][2..]);
    let sps = video::hevc::parse_sps(&sps_rbsp).expect("sps");
    let pps_rbsp = video::bits::unescape_rbsp(&hvcc.pps[0][2..]);
    let pps = video::hevc::parse_pps(&pps_rbsp).expect("pps");
    eprintln!(
        "SPS {}x{} mincb={} ctb={} mintb={} maxtb={} h_i={} h_p={} amp={} sao={} pcm={} sis={} scal={} tmvp={} lt={} st_rps={} max_sub={}",
        sps.pic_width_in_luma_samples, sps.pic_height_in_luma_samples,
        sps.log2_min_cb_size, sps.log2_ctb_size, sps.log2_min_tb_size, sps.log2_max_tb_size,
        sps.max_transform_hierarchy_depth_intra, sps.max_transform_hierarchy_depth_inter,
        sps.amp_enabled as u8, sps.sao_enabled as u8, sps.pcm_enabled as u8,
        sps.strong_intra_smoothing as u8, sps.scaling_list_enabled as u8,
        sps.temporal_mvp_enabled as u8, sps.long_term_ref_pics_present as u8,
        sps.st_rps.len(), sps.max_sub_layers
    );
    eprintln!(
        "PPS id={} extra_bits={} dep={} wpp={} tiles={} wp={} wbp={} lists_mod={} cuqp={} qpdepth={} init_qp={} sdh={} tskip={} cbo={} cro={} deblock_off={} beta={} tc={} ext={} lmpl={}",
        pps.id, pps.num_extra_slice_header_bits, pps.dependent_slice_segments_enabled as u8,
        pps.entropy_coding_sync_enabled as u8, pps.tiles_enabled as u8,
        pps.weighted_pred as u8, pps.weighted_bipred as u8, pps.lists_modification_present as u8,
        pps.cu_qp_delta_enabled as u8, pps.diff_cu_qp_delta_depth, pps.init_qp,
        pps.sign_data_hiding_enabled as u8, pps.transform_skip_enabled as u8,
        pps.cb_qp_offset, pps.cr_qp_offset, pps.deblocking_filter_disabled as u8,
        pps.beta_offset_div2, pps.tc_offset_div2, pps.slice_segment_header_extension_present as u8,
        pps.log2_parallel_merge_level
    );
    for (i, s) in samples.iter().enumerate().take(limit) {
        let data = &bytes[s.offset..s.offset + s.size];
        let nals = video::hevc::split_hvcc(data, hvcc.length_size);
        for nal in nals {
            if !nal.kind.is_slice() { continue; }
            let rbsp = nal.rbsp();
            match video::hevc::parse_slice_header(&rbsp, nal.kind, &sps, &pps) {
                Ok(h) => {
                    let eps: Vec<u32> = h.entry_point_offsets.clone();
                    let sum: u32 = eps.iter().sum();
                    eprintln!(
                        "sample{} nal={} type={:?} first={} dep={} addr={} poc_lsb={} qp={} sao={}{} tmvp={} merge={} data@{} eps={} sum_eps={} rbsp={} cabac_bytes={} mvd1z={} col_l0={} col_idx={} l0={} l1={} list_l0={:?} list_l1={:?} st_rps_s0={:?} st_rps_s1={:?} used0={:?} used1={:?} cabac_init={}",
                        i, nal.kind.code(), h.slice_type, h.first_slice_in_pic as u8,
                        h.dependent_slice_segment as u8, h.segment_address,
                        h.pic_order_cnt_lsb, h.qp, h.sao_luma as u8, h.sao_chroma as u8,
                        h.temporal_mvp_enabled as u8, h.max_num_merge_cand(),
                        h.data_byte_offset, eps.len(), sum, rbsp.len(),
                        rbsp.len().saturating_sub(h.data_byte_offset),
                        h.mvd_l1_zero as u8, h.collocated_from_l0 as u8, h.collocated_ref_idx,
                        h.num_ref_idx_l0, h.num_ref_idx_l1,
                        h.list_entry_l0, h.list_entry_l1,
                        h.st_rps.delta_poc_s0, h.st_rps.delta_poc_s1,
                        h.st_rps.used_s0, h.st_rps.used_s1,
                        h.cabac_init as u8
                    );
                    if !eps.is_empty() {
                        let mut starts = vec![0u32];
                        for &o in &eps { starts.push(starts.last().unwrap() + o); }
                        eprintln!("  entry_starts={:?}", starts);
                        // first few cabac data bytes
                        if h.data_byte_offset < rbsp.len() {
                            let d = &rbsp[h.data_byte_offset..];
                            let n = d.len().min(16);
                            eprint!("  cabac_head=");
                            for b in &d[..n] { eprint!("{:02x}", b); }
                            eprintln!();
                            for (k, &st) in starts.iter().enumerate().take(4) {
                                let off = st as usize;
                                if off < d.len() {
                                    let n = (d.len()-off).min(8);
                                    eprint!("  sub{}@{}=", k, off);
                                    for b in &d[off..off+n] { eprint!("{:02x}", b); }
                                    eprintln!();
                                }
                            }
                        }
                    }
                }
                Err(e) => eprintln!("sample{}: {}", i, e),
            }
        }
    }
}
