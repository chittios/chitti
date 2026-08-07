//! Minimal ISO-BMFF (MP4) **muxer** for a single H.264/AVC video track.
//!
//! The inverse of [`super::mp4`]'s demuxer, scoped to what screen recording
//! needs: one progressive `avc1` track, SPS/PPS in the sample description
//! (`avcC`), length-prefixed VCL NALs as samples, a sync-sample table for
//! IDR frames. Pure over in-memory buffers — unit-tested by round-tripping
//! through our own demuxer.

use alloc::vec;
use alloc::vec::Vec;

/// One compressed access unit (typically one length-prefixed slice NAL).
pub struct Sample {
    pub bytes: Vec<u8>,
    /// Duration in timescale ticks.
    pub duration: u32,
    /// True for IDR / keyframes.
    pub sync: bool,
}

/// Mux an AVC stream into a self-contained `.mp4` file.
///
/// `sps_nal` / `pps_nal` are full NAL units (1-byte header + payload), as
/// produced by the encoder. `timescale` is ticks per second (e.g. 1000).
pub fn mux_avc(
    width: u32,
    height: u32,
    timescale: u32,
    sps_nal: &[u8],
    pps_nal: &[u8],
    samples: &[Sample],
) -> Result<Vec<u8>, &'static str> {
    if samples.is_empty() {
        return Err("mp4mux: no samples");
    }
    if width == 0 || height == 0 || timescale == 0 {
        return Err("mp4mux: bad geometry/timescale");
    }
    if sps_nal.is_empty() || pps_nal.is_empty() {
        return Err("mp4mux: missing parameter sets");
    }

    let mut mdat_payload = Vec::new();
    let mut chunk_offsets: Vec<u32> = Vec::new();
    let mut sample_sizes: Vec<u32> = Vec::new();
    let mut sync_samples: Vec<u32> = Vec::new(); // 1-based indices
    let mut total_duration: u64 = 0;

    // mdat header is 8 bytes; sample data starts at file offset after ftyp+mdat hdr.
    // We'll patch offsets after building ftyp.
    for (i, s) in samples.iter().enumerate() {
        chunk_offsets.push(mdat_payload.len() as u32); // relative; patched later
        sample_sizes.push(s.bytes.len() as u32);
        if s.sync {
            sync_samples.push((i + 1) as u32);
        }
        total_duration += s.duration as u64;
        mdat_payload.extend_from_slice(&s.bytes);
    }

    let avcc = build_avcc(sps_nal, pps_nal)?;
    let stsd = build_stsd(width, height, &avcc);
    let stts = build_stts(samples);
    let stsc = box_full(b"stsc", 0, 0, &{
        let mut b = Vec::new();
        // one entry: first_chunk=1, samples_per_chunk=1, sample_desc=1
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
        b
    });
    let stsz = {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 → table
        b.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
        for &sz in &sample_sizes {
            b.extend_from_slice(&sz.to_be_bytes());
        }
        box_full(b"stsz", 0, 0, &b)
    };
    let stss = {
        let mut b = Vec::new();
        b.extend_from_slice(&(sync_samples.len() as u32).to_be_bytes());
        for &s in &sync_samples {
            b.extend_from_slice(&s.to_be_bytes());
        }
        box_full(b"stss", 0, 0, &b)
    };

    // Placeholder stco — patched once we know mdat offset.
    let stco_placeholder = {
        let mut b = Vec::new();
        b.extend_from_slice(&(chunk_offsets.len() as u32).to_be_bytes());
        for _ in &chunk_offsets {
            b.extend_from_slice(&0u32.to_be_bytes());
        }
        box_full(b"stco", 0, 0, &b)
    };

    let stbl = box_container(
        b"stbl",
        &[stsd, stts, stsc, stsz, stss, stco_placeholder],
    );
    let minf = box_container(
        b"minf",
        &[
            box_full(b"vmhd", 0, 1, &[0u8; 8]),
            box_container(
                b"dinf",
                &[box_full(b"dref", 0, 0, &{
                    let mut b = Vec::new();
                    b.extend_from_slice(&1u32.to_be_bytes());
                    // Self-contained url entry (full box, flags bit 0 set).
                    b.extend_from_slice(&box_full(b"url ", 0, 1, &[]));
                    b
                })],
            ),
            stbl,
        ],
    );
    let mdhd = {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // creation
        b.extend_from_slice(&0u32.to_be_bytes()); // modification
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&(total_duration as u32).to_be_bytes());
        b.extend_from_slice(&0x55c4_0000u32.to_be_bytes()); // language 'und' + quality
        box_full(b"mdhd", 0, 0, &b)
    };
    let hdlr = {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        b.extend_from_slice(b"vide");
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(b"VideoHandler\0");
        box_full(b"hdlr", 0, 0, &b)
    };
    let mdia = box_container(b"mdia", &[mdhd, hdlr, minf]);
    let tkhd = {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // creation
        b.extend_from_slice(&0u32.to_be_bytes()); // modification
        b.extend_from_slice(&1u32.to_be_bytes()); // track_id
        b.extend_from_slice(&0u32.to_be_bytes()); // reserved
        b.extend_from_slice(&(total_duration as u32).to_be_bytes());
        b.extend_from_slice(&[0u8; 8]); // reserved
        b.extend_from_slice(&0u16.to_be_bytes()); // layer
        b.extend_from_slice(&0u16.to_be_bytes()); // alternate
        b.extend_from_slice(&0u16.to_be_bytes()); // volume
        b.extend_from_slice(&0u16.to_be_bytes()); // reserved
        // identity matrix
        b.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        ]);
        b.extend_from_slice(&(width << 16).to_be_bytes());
        b.extend_from_slice(&(height << 16).to_be_bytes());
        box_full(b"tkhd", 0, 3, &b) // flags: enabled+in_movie
    };
    let trak = box_container(b"trak", &[tkhd, mdia]);
    let mvhd = {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&(total_duration as u32).to_be_bytes());
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        b.extend_from_slice(&0x0100u16.to_be_bytes()); // volume
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&[0u8; 8]);
        b.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        ]);
        b.extend_from_slice(&[0u8; 24]);
        b.extend_from_slice(&2u32.to_be_bytes()); // next_track_id
        box_full(b"mvhd", 0, 0, &b)
    };
    let moov = box_container(b"moov", &[mvhd, trak]);

    // ftyp
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(&0x200u32.to_be_bytes());
    ftyp.extend_from_slice(b"isomiso2avc1mp41");
    let ftyp_box = box_raw(b"ftyp", &ftyp);

    // mdat
    let mdat_box = box_raw(b"mdat", &mdat_payload);

    // Absolute sample offsets = ftyp.len + 8 (mdat header) + relative
    let mdat_data_off = (ftyp_box.len() + 8) as u32;
    let mut out = Vec::with_capacity(ftyp_box.len() + mdat_box.len() + moov.len() + 256);
    out.extend_from_slice(&ftyp_box);
    out.extend_from_slice(&mdat_box);
    // Patch stco inside moov: find 'stco' and rewrite offsets.
    let mut moov_bytes = moov;
    patch_stco(&mut moov_bytes, mdat_data_off, &chunk_offsets);
    out.extend_from_slice(&moov_bytes);
    Ok(out)
}

fn build_avcc(sps_nal: &[u8], pps_nal: &[u8]) -> Result<Vec<u8>, &'static str> {
    // avcC: configurationVersion, profile, compat, level, lengthSizeMinusOne,
    // numSPS, SPS, numPPS, PPS.
    if sps_nal.len() < 4 || pps_nal.len() < 1 {
        return Err("mp4mux: short parameter set");
    }
    let profile = sps_nal[1];
    let compat = sps_nal[2];
    let level = sps_nal[3];
    let mut b = Vec::new();
    b.push(1); // version
    b.push(profile);
    b.push(compat);
    b.push(level);
    b.push(0xff); // 6 bits reserved + lengthSizeMinusOne=3
    b.push(0xe1); // 3 bits reserved + numOfSequenceParameterSets=1
    b.extend_from_slice(&(sps_nal.len() as u16).to_be_bytes());
    b.extend_from_slice(sps_nal);
    b.push(1); // numOfPictureParameterSets
    b.extend_from_slice(&(pps_nal.len() as u16).to_be_bytes());
    b.extend_from_slice(pps_nal);
    Ok(b)
}

fn build_stsd(width: u32, height: u32, avcc: &[u8]) -> Vec<u8> {
    let mut avc1 = Vec::new();
    avc1.extend_from_slice(&[0u8; 6]); // reserved
    avc1.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    avc1.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
    avc1.extend_from_slice(&(width as u16).to_be_bytes());
    avc1.extend_from_slice(&(height as u16).to_be_bytes());
    avc1.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // hres 72 dpi
    avc1.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vres
    avc1.extend_from_slice(&0u32.to_be_bytes()); // reserved
    avc1.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    avc1.extend_from_slice(&[0u8; 32]); // compressor name
    avc1.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    avc1.extend_from_slice(&(-1i16 as u16).to_be_bytes()); // pre_defined
    avc1.extend_from_slice(&box_raw(b"avcC", avcc));
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    body.extend_from_slice(&box_raw(b"avc1", &avc1));
    box_full(b"stsd", 0, 0, &body)
}

fn build_stts(samples: &[Sample]) -> Vec<u8> {
    // Run-length compress equal durations.
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for s in samples {
        if let Some(last) = entries.last_mut() {
            if last.1 == s.duration {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, s.duration));
    }
    let mut b = Vec::new();
    b.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (count, dur) in entries {
        b.extend_from_slice(&count.to_be_bytes());
        b.extend_from_slice(&dur.to_be_bytes());
    }
    box_full(b"stts", 0, 0, &b)
}

fn box_raw(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
    out.extend_from_slice(typ);
    out.extend_from_slice(body);
    out
}

fn box_full(typ: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + body.len());
    inner.push(version);
    inner.push(((flags >> 16) & 0xff) as u8);
    inner.push(((flags >> 8) & 0xff) as u8);
    inner.push((flags & 0xff) as u8);
    inner.extend_from_slice(body);
    box_raw(typ, &inner)
}

fn box_container(typ: &[u8; 4], children: &[Vec<u8>]) -> Vec<u8> {
    let mut body = Vec::new();
    for c in children {
        body.extend_from_slice(c);
    }
    box_raw(typ, &body)
}

fn patch_stco(moov: &mut [u8], mdat_data_off: u32, rel_offsets: &[u32]) {
    // Find 'stco' fourcc and rewrite the offset table.
    let mut i = 0;
    while i + 8 <= moov.len() {
        let size = u32::from_be_bytes([moov[i], moov[i + 1], moov[i + 2], moov[i + 3]]) as usize;
        if size < 8 || i + size > moov.len() {
            break;
        }
        if &moov[i + 4..i + 8] == b"stco" {
            // full box: ver/flags at i+8, entry_count at i+12, entries at i+16
            if i + 16 + rel_offsets.len() * 4 <= i + size {
                let count = u32::from_be_bytes([
                    moov[i + 12],
                    moov[i + 13],
                    moov[i + 14],
                    moov[i + 15],
                ]) as usize;
                let n = count.min(rel_offsets.len());
                for (k, &rel) in rel_offsets.iter().take(n).enumerate() {
                    let off = mdat_data_off + rel;
                    let p = i + 16 + k * 4;
                    moov[p..p + 4].copy_from_slice(&off.to_be_bytes());
                }
            }
            return;
        }
        // Recurse into containers by continuing the linear scan — stco is a
        // leaf so a flat scan of the whole moov blob finds it.
        i += if size == 0 { 8 } else { size };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::encoder::Encoder;
    use crate::video::mp4;

    #[test_case]
    fn muxed_recording_demuxes_with_our_parser() {
        let mut enc = Encoder::new(32, 32, 28).unwrap();
        let px = alloc::vec![0x00_20_40_80u32; 32 * 32];
        let mut samples = Vec::new();
        for i in 0..3 {
            let au = enc.encode_rgb32(&px, i == 0).unwrap();
            samples.push(Sample {
                bytes: au,
                duration: 200, // 5 fps at timescale 1000
                sync: i == 0,
            });
        }
        let mp4 = mux_avc(32, 32, 1000, &enc.sps_nal, &enc.pps_nal, &samples).unwrap();
        let track = mp4::parse(&mp4).expect("demux");
        assert_eq!((track.width, track.height), (32, 32));
        assert_eq!(track.samples.len(), 3);
        assert!(track.samples[0].is_sync);
    }
}
