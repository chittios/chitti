//! **Chitti UEFI stub bootloader** (aarch64). AAVMF launches this as
//! `\EFI\BOOT\BOOTAA64.EFI` from the ESP; it loads the normal `-kernel` Chitti
//! ELF off the same volume, loads its PT_LOAD segments to their physical link
//! addresses, optionally loads the model to the fixed model address, exits boot
//! services, and jumps to the kernel entry **with the MMU still on** (UEFI's
//! identity map).
//!
//! Why MMU-on: the aarch64 kernel is an identity-map kernel (`arch::aarch64::mmu`
//! builds its own low-4 GiB identity map in `mmu::init`, which `enable_mmu`
//! installs via TTBR0 + `tlbi` — a map-to-map switch that works whether the MMU
//! was on or off). Handing off with UEFI's identity map active + caches on lets
//! the kernel run its normal `_start` -> `mmu::init` path unchanged, with no
//! cache-maintenance dance and no HHDM/`dma_to_phys` retrofit — so the whole
//! proven `-kernel` code path (incl. the virtio-blk-mmio driver) runs as-is,
//! now booted from a disk's ESP via firmware.

#![no_main]
#![no_std]

extern crate alloc;

use uefi::boot::{self, AllocateType, MemoryType};
use uefi::prelude::*;

/// The most heap the kernel will ever want (its largest per-model tier is 1 GiB
/// for the 9B). The stub reserves this much in free RAM and reports the base; the
/// kernel uses its own (<= this) tier within the reservation.
const HEAP_MAX: u64 = 1 << 30;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le64(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

/// Load `\model.gguf.000` off the ESP via **raw Block I/O with our own FAT32
/// walk**, straight into freshly allocated pages. The firmware's FAT driver
/// (`fs.read`) reads a ~774 MB model cluster-by-cluster through small internal
/// buffers — minutes on VirtualBox — and costs two extra full copies (the read
/// Vec + the copy into place). Here: the whole FAT is read in one Block I/O
/// call, the cluster chain is coalesced into contiguous runs, and each run is
/// read directly into the destination (capped at 16 MiB per request). Returns
/// `(base, len)`, or `None` for any surprise (caller falls back to `fs.read`).
fn load_model_blockio() -> Option<(u64, u64)> {
    use uefi::proto::loaded_image::LoadedImage;
    use uefi::proto::media::block::BlockIO;

    // GetProtocol: the firmware owns LoadedImage; we only read `.device()`.
    // SAFETY: read-only access to our own image's protocol.
    let li = unsafe {
        boot::open_protocol::<LoadedImage>(
            boot::OpenProtocolParams {
                handle: boot::image_handle(),
                agent: boot::image_handle(),
                controller: None,
            },
            boot::OpenProtocolAttributes::GetProtocol,
        )
        .ok()?
    };
    let dev = li.device()?;
    // GetProtocol (non-exclusive): the firmware's FAT stack keeps its own open.
    // SAFETY: read-only Block I/O alongside the firmware driver; we never write.
    let bio = unsafe {
        boot::open_protocol::<BlockIO>(
            boot::OpenProtocolParams {
                handle: dev,
                agent: boot::image_handle(),
                controller: None,
            },
            boot::OpenProtocolAttributes::GetProtocol,
        )
        .ok()?
    };
    let media_id = bio.media().media_id();
    let bs = bio.media().block_size() as usize;
    if bs != 512 {
        return None; // only 512-byte sectors (matches the image builder)
    }

    // BPB → FAT32 geometry (partition-relative LBA 0).
    let mut bpb = [0u8; 512];
    bio.read_blocks(media_id, 0, &mut bpb).ok()?;
    if bpb[510] != 0x55 || bpb[511] != 0xAA || le16(&bpb, 11) as usize != 512 {
        return None;
    }
    let spc = bpb[13] as u64;
    let reserved = le16(&bpb, 14) as u64;
    let nfats = bpb[16] as u64;
    let fat_size = {
        let f16 = le16(&bpb, 22) as u64;
        if f16 != 0 {
            f16
        } else {
            le32(&bpb, 36) as u64
        }
    };
    let root_entries = le16(&bpb, 17) as u64;
    if spc == 0 || fat_size == 0 || root_entries != 0 {
        return None; // FAT32 only (root as cluster chain)
    }
    let root_clus = le32(&bpb, 44);
    let fat_lba = reserved;
    let data_lba = reserved + nfats * fat_size;
    let cluster_lba = |c: u32| data_lba + (c as u64 - 2) * spc;

    // The whole FAT in one read (a few hundred KiB for our ESP sizes), so the
    // chain walk below is pure memory.
    if fat_size * 512 > 64 << 20 {
        return None; // implausible FAT size — don't trust the parse
    }
    let mut fat = alloc::vec![0u8; (fat_size * 512) as usize];
    bio.read_blocks(media_id, fat_lba, &mut fat).ok()?;
    let next = |c: u32| -> Option<u32> {
        // Bounds-checked: a corrupt chain must fall back, not fault the stub.
        let off = c as usize * 4;
        if off + 4 > fat.len() {
            return None;
        }
        let n = le32(&fat, off) & 0x0fff_ffff;
        (n >= 2 && n < 0x0fff_fff8).then_some(n)
    };

    // Walk the root directory for `model.gguf.000` (VFAT long name: accumulate
    // LFN entries; 8.3 fallback compares the padded short name).
    const WANT: &str = "model.gguf.000";
    let bpc = (spc * 512) as usize;
    let mut dirbuf = alloc::vec![0u8; bpc];
    let (mut start, mut size) = (0u32, 0u32);
    let mut lfn = alloc::string::String::new();
    let mut c = Some(root_clus);
    let mut dir_clusters = 0u32;
    'outer: while let Some(cl) = c {
        // Cycle guard: a corrupt chain must not spin the stub forever.
        dir_clusters += 1;
        if dir_clusters > 65_536 {
            return None;
        }
        bio.read_blocks(media_id, cluster_lba(cl), &mut dirbuf)
            .ok()?;
        for e in dirbuf.chunks_exact(32) {
            match e[0] {
                0 => break 'outer, // end of directory
                0xe5 => {
                    lfn.clear();
                    continue;
                }
                _ => {}
            }
            if e[11] == 0x0f {
                // LFN entry: 13 UCS-2 chars at offsets 1..11, 14..26, 28..32.
                let mut part = alloc::string::String::new();
                for &o in &[1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30] {
                    let ch = le16(e, o);
                    if ch == 0 || ch == 0xffff {
                        break;
                    }
                    part.push(char::from_u32(ch as u32).unwrap_or('?'));
                }
                // Entries arrive last-part-first; prepend.
                part.push_str(&lfn);
                lfn = part;
                continue;
            }
            let name = if !lfn.is_empty() {
                core::mem::take(&mut lfn)
            } else {
                short_name(e)
            };
            if name.eq_ignore_ascii_case(WANT) && e[11] & 0x10 == 0 {
                start = (le16(e, 20) as u32) << 16 | le16(e, 26) as u32;
                size = le32(e, 28);
                break 'outer;
            }
        }
        c = next(cl);
    }
    if start < 2 || size == 0 {
        return None;
    }

    // Destination pages, then the chain as coalesced runs, ≤16 MiB per read.
    let pages = (size as usize).div_ceil(4096);
    let ptr = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages).ok()?;
    // SAFETY: freshly allocated `pages * 4096` bytes.
    let dst = unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), pages * 4096) };
    // 1 MiB per request: comfortably under any firmware Block I/O limit —
    // VirtualBox's EFI NVMe driver does not split oversized transfers the way
    // QEMU's EDK2 does (NVMe MDTS), and a too-big read hangs it.
    const MAX_RUN: usize = 1 << 20;
    let mut done = 0usize;
    let mut cur = Some(start);
    let mut chain_guard = 0u64;
    let mut next_progress = 256usize << 20;
    while done < size as usize {
        // Cycle guard: never walk more clusters than the file can hold.
        chain_guard += 1;
        if chain_guard > (size as u64 / bpc as u64) + 16 {
            return None;
        }
        let c0 = cur?;
        let mut run = 1u64;
        let mut tail = c0;
        cur = None;
        while done + (run as usize) * bpc < size as usize && (run as usize) * bpc < MAX_RUN {
            match next(tail) {
                Some(n) if n == tail + 1 => {
                    tail = n;
                    run += 1;
                }
                other => {
                    cur = other;
                    break;
                }
            }
        }
        if cur.is_none() && done + (run as usize) * bpc < size as usize {
            cur = next(tail); // run was cut by MAX_RUN, not by the chain
        }
        let want = (size as usize - done).min(run as usize * bpc);
        let sectors = want.div_ceil(512);
        bio.read_blocks(
            media_id,
            cluster_lba(c0),
            &mut dst[done..done + sectors * 512],
        )
        .ok()?;
        done += want;
        if done >= next_progress {
            log::info!("chitti-stub: model {} / {} MiB…", done >> 20, size >> 20);
            next_progress += 256 << 20;
        }
    }
    log::info!(
        "chitti-stub: model {} bytes at {:#x} (Block I/O fast path)",
        size,
        ptr.as_ptr() as u64
    );
    Some((ptr.as_ptr() as u64, size as u64))
}

/// The 8.3 short name of a directory entry, dot-joined and lowercased-ish
/// (enough for a case-insensitive compare).
fn short_name(e: &[u8]) -> alloc::string::String {
    let base: alloc::string::String = e[0..8].iter().map(|&b| b as char).collect();
    let ext: alloc::string::String = e[8..11].iter().map(|&b| b as char).collect();
    let (base, ext) = (base.trim_end(), ext.trim_end());
    if ext.is_empty() {
        alloc::string::String::from(base)
    } else {
        alloc::format!("{base}.{ext}")
    }
}

/// Allocate `bytes` physical pages at page-aligned `paddr`, returning a slice
/// over them. Used once for the whole kernel span and once for the model.
fn alloc_at(paddr: u64, bytes: usize) -> &'static mut [u8] {
    let base = paddr & !0xfff;
    let pages = (bytes as u64 + (paddr - base)).div_ceil(4096) as usize;
    let ptr = boot::allocate_pages(AllocateType::Address(base), MemoryType::LOADER_DATA, pages)
        .unwrap_or_else(|e| panic!("allocate_pages at {base:#x} ({pages} pages) failed: {e:?}"));
    // SAFETY: freshly allocated, `pages * 4096` bytes at `base`.
    unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), pages * 4096) }
}

/// The ACPI 2.0 RSDP physical address from the UEFI configuration table, or 0.
/// Days since 1970-01-01 for a proleptic-Gregorian date (Hinnant's algorithm;
/// the kernel's `clock` uses the same math).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// The current UTC time as Unix seconds from the UEFI runtime clock, or 0 if the
/// firmware has no RTC. Converts the (possibly timezone-local) EFI time to UTC.
fn efi_unix() -> u64 {
    let Ok(t) = uefi::runtime::get_time() else {
        return 0;
    };
    let secs = days_from_civil(t.year() as i64, t.month() as i64, t.day() as i64) * 86400
        + t.hour() as i64 * 3600
        + t.minute() as i64 * 60
        + t.second() as i64;
    // EFI time may carry a timezone offset (minutes east of UTC); 2047 = unspecified.
    let secs = match t.time_zone() {
        Some(tz) if tz != 2047 => secs - tz as i64 * 60,
        _ => secs,
    };
    if secs > 0 {
        secs as u64
    } else {
        0
    }
}

fn acpi_rsdp() -> u64 {
    use uefi::table::cfg::{ACPI2_GUID, ACPI_GUID};
    uefi::system::with_config_table(|entries| {
        entries
            .iter()
            .find(|e| e.guid == ACPI2_GUID)
            .or_else(|| entries.iter().find(|e| e.guid == ACPI_GUID))
            .map(|e| e.address as u64)
            .unwrap_or(0)
    })
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().expect("uefi init");
    log::info!("chitti-stub: loading kernel off the ESP");

    // Read the kernel ELF (and the model, if present) from the boot volume.
    let mut fs = uefi::fs::FileSystem::new(
        boot::get_image_file_system(boot::image_handle()).expect("no ESP filesystem"),
    );
    let kernel = fs
        .read(cstr16!("\\chitti-kernel"))
        .expect("read \\chitti-kernel");
    log::info!("chitti-stub: kernel {} bytes", kernel.len());

    // Parse the ELF64 header; find the PT_LOAD span, allocate it once (segments
    // can share page boundaries, so per-segment AllocateAddress conflicts), then
    // copy each segment to its physical address.
    assert_eq!(&kernel[0..4], b"\x7fELF", "not an ELF");
    let entry = le64(&kernel, 24);
    let phoff = le64(&kernel, 32) as usize;
    let phentsize = le16(&kernel, 54) as usize;
    let phnum = le16(&kernel, 56) as usize;
    let loads = || {
        (0..phnum)
            .map(|i| phoff + i * phentsize)
            .filter(|&ph| le32(&kernel, ph) == 1)
    };
    let min_pa = loads()
        .map(|ph| le64(&kernel, ph + 24))
        .min()
        .expect("no PT_LOAD");
    let max_end = loads()
        .map(|ph| le64(&kernel, ph + 24) + le64(&kernel, ph + 40))
        .max()
        .unwrap();
    log::info!("chitti-stub: kernel span {min_pa:#x}..{max_end:#x} entry={entry:#x}");
    let region = alloc_at(min_pa, (max_end - min_pa) as usize);
    for ph in loads() {
        let off = le64(&kernel, ph + 8) as usize;
        let paddr = le64(&kernel, ph + 24);
        let filesz = le64(&kernel, ph + 32) as usize;
        let memsz = le64(&kernel, ph + 40) as usize;
        let dst = (paddr - min_pa) as usize;
        region[dst..dst + filesz].copy_from_slice(&kernel[off..off + filesz]);
        for b in region[dst + filesz..dst + memsz].iter_mut() {
            *b = 0; // .bss tail (kernel also zeroes __bss, but be safe)
        }
    }

    // Load the model into free RAM (AnyPages), if bundled. A *fixed* physical
    // address is not reliable under UEFI firmware — VirtualBox/AAVMF reserves
    // regions at fixed addresses and AllocateAddress there returns NOT_FOUND — so
    // we let the firmware pick a free run and report where it landed in the
    // boot-info; the kernel reads the model there (not at a hardcoded address).
    //
    // Fast path first: raw Block I/O + our own FAT32 walk reads the ~774 MB
    // model in big contiguous requests with zero extra copies; the firmware's
    // FAT driver (fallback) reads it cluster-by-cluster — minutes on VBox.
    let model_region: Option<(u64, u64)> = match load_model_blockio() {
        Some(r) => Some(r),
        None => {
            log::info!(
                "chitti-stub: Block I/O fast path unavailable; firmware FAT fallback (slow)"
            );
            match fs.read(cstr16!("\\model.gguf.000")) {
                Ok(model) => {
                    let pages = model.len().div_ceil(4096);
                    match boot::allocate_pages(
                        AllocateType::AnyPages,
                        MemoryType::LOADER_DATA,
                        pages,
                    ) {
                        Ok(ptr) => {
                            let base = ptr.as_ptr() as u64;
                            // SAFETY: freshly allocated, `pages * 4096` bytes at `base`.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(ptr.as_ptr(), pages * 4096)
                            };
                            dst[..model.len()].copy_from_slice(&model);
                            log::info!(
                                "chitti-stub: model {} bytes at {base:#x} (firmware FAT path)",
                                model.len()
                            );
                            Some((base, model.len() as u64))
                        }
                        Err(e) => {
                            log::warn!("chitti-stub: model alloc ({pages} pages) failed: {e:?} -- booting without a model (need more VM RAM)");
                            None
                        }
                    }
                }
                Err(_) => {
                    log::info!("chitti-stub: no model on ESP (kernel will report no model)");
                    None
                }
            }
        }
    };

    // Reserve the kernel heap in free RAM (AnyPages, HEAP_MAX) and mark it
    // LOADER_DATA so it survives ExitBootServices. Report its base; the kernel
    // places its heap here. AnyPages (not a fixed address, and not the top of RAM
    // where UEFI parks ACPI/runtime data) is what makes this robust across
    // firmwares.
    let heap_region: Option<(u64, u64)> = {
        let pages = (HEAP_MAX / 4096) as usize;
        match boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages) {
            Ok(ptr) => {
                let base = ptr.as_ptr() as u64;
                log::info!(
                    "chitti-stub: reserved kernel heap {base:#x} ({} MiB)",
                    HEAP_MAX >> 20
                );
                Some((base, HEAP_MAX))
            }
            Err(e) => {
                log::warn!(
                    "chitti-stub: heap reserve ({pages} pages) failed: {e:?} (need more VM RAM)"
                );
                None
            }
        }
    };

    // Capture the UEFI GOP framebuffer and publish it in a boot-info page at a
    // fixed address the kernel checks. This is what makes Chitti's console
    // visible on ANY UEFI platform (VirtualBox-ARM, UTM, real hardware) — the
    // kernel's own ramfb device is QEMU-only. Best-effort: absent GOP or a
    // blt-only mode just means no boot-info (kernel falls back to ramfb/serial).
    let bootinfo: Option<u64> = (|| {
        use uefi::proto::console::gop::{GraphicsOutput, PixelFormat};
        let h = match boot::get_handle_for_protocol::<GraphicsOutput>() {
            Ok(h) => h,
            Err(e) => {
                log::info!("chitti-stub: no GOP handle: {e:?}");
                return None;
            }
        };
        // Non-exclusive open: the firmware console (ConSplitter) already owns
        // GOP, so an exclusive open is denied. GetProtocol just reads it.
        let mut gop = match unsafe {
            boot::open_protocol::<GraphicsOutput>(
                boot::OpenProtocolParams {
                    handle: h,
                    agent: boot::image_handle(),
                    controller: None,
                },
                boot::OpenProtocolAttributes::GetProtocol,
            )
        } {
            Ok(g) => g,
            Err(e) => {
                log::info!("chitti-stub: GOP open failed: {e:?}");
                return None;
            }
        };
        // Select the LARGEST available mode so an HDMI monitor runs at its full
        // native resolution instead of whatever (often 800x600/1024x768) mode
        // the firmware left GOP in. `Mode` is `Copy`, so pick the best while the
        // read-only `modes()` iterator is alive, then `set_mode` it.
        let mut best: Option<(uefi::proto::console::gop::Mode, usize)> = None;
        for m in gop.modes() {
            let mi = m.info();
            if mi.pixel_format() == PixelFormat::BltOnly {
                continue;
            }
            let (mw, mh) = mi.resolution();
            let area = mw * mh;
            if best.as_ref().map_or(true, |&(_, ba)| area > ba) {
                best = Some((m, area));
            }
        }
        if let Some((ref m, _)) = best {
            let (mw, mh) = m.info().resolution();
            if let Err(e) = gop.set_mode(m) {
                log::info!("chitti-stub: GOP set_mode {mw}x{mh} failed: {e:?} (keeping current)");
            }
        }

        let mode = gop.current_mode_info();
        if mode.pixel_format() == PixelFormat::BltOnly {
            return None;
        }
        let (w, hgt) = mode.resolution();
        let pitch = mode.stride() as u64 * 4;
        // Pixel-format shifts (bit position of each channel in the LE u32). GOP
        // "Rgb" stores bytes R,G,B,X → red at bit 0; "Bgr" → red at bit 16 (the
        // common XRGB8888). A real HDMI panel can report either, and swapping
        // red/blue would tint the whole UI, so carry the shifts to the kernel.
        let (rs, gs, bs): (u8, u8, u8) = match mode.pixel_format() {
            PixelFormat::Rgb => (0, 8, 16),
            PixelFormat::Bgr => (16, 8, 0),
            PixelFormat::Bitmask => match mode.pixel_bitmask() {
                Some(m) => (
                    m.red.trailing_zeros() as u8,
                    m.green.trailing_zeros() as u8,
                    m.blue.trailing_zeros() as u8,
                ),
                None => (16, 8, 0),
            },
            PixelFormat::BltOnly => return None,
        };
        let fb = gop.frame_buffer().as_mut_ptr() as u64;
        // Boot-info page: AnyPages (fixed low addresses aren't reliably
        // allocatable — they vary per firmware/platform). Its address is passed
        // to the kernel in x1 at handoff.
        let p = boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, 1).ok()?;
        let addr = p.as_ptr() as u64;
        let page = unsafe { core::slice::from_raw_parts_mut(p.as_ptr(), 4096) };
        page[0..8].copy_from_slice(b"CHITTIBI");
        page[8..16].copy_from_slice(&fb.to_le_bytes());
        page[16..24].copy_from_slice(&(w as u64).to_le_bytes());
        page[24..32].copy_from_slice(&(hgt as u64).to_le_bytes());
        page[32..40].copy_from_slice(&pitch.to_le_bytes());
        // ACPI RSDP (from the UEFI config table) at offset 40 — the kernel walks
        // it to find the PCIe ECAM base (MCFG), so PCIe is discovered, not
        // hardcoded. 0 if absent.
        let rsdp = acpi_rsdp();
        page[40..48].copy_from_slice(&rsdp.to_le_bytes());
        // Pixel format at 48..52: r_shift, g_shift, b_shift, bytes-per-pixel.
        page[48] = rs;
        page[49] = gs;
        page[50] = bs;
        page[51] = 4;
        // Real wall-clock time (UTC Unix seconds) at 52..60, from UEFI GetTime.
        // This is the only reliable clock on VirtualBox-ARM, whose generic timer
        // doesn't advance for the guest; 0 if the firmware has no RTC.
        page[52..60].copy_from_slice(&efi_unix().to_le_bytes());
        // Kernel heap region at 60..76 (base@60, size@68) and model region at
        // 76..92 (base@76, size@84), both AnyPages-allocated above. The kernel
        // reads the model at the reported base and places its heap at the heap
        // base — no fixed physical addresses on the UEFI path.
        let (hb, hs) = heap_region.unwrap_or((0, 0));
        page[60..68].copy_from_slice(&hb.to_le_bytes());
        page[68..76].copy_from_slice(&hs.to_le_bytes());
        let (mb, ms) = model_region.unwrap_or((0, 0));
        page[76..84].copy_from_slice(&mb.to_le_bytes());
        page[84..92].copy_from_slice(&ms.to_le_bytes());
        log::info!("chitti-stub: GOP {w}x{hgt} at {fb:#x} (shifts {rs}/{gs}/{bs}), ACPI RSDP {rsdp:#x}, heap {hb:#x}, model {mb:#x} -> boot-info {addr:#x}");
        Some(addr)
    })();

    // (The kernel heap was reserved above, just past the model, before the
    // boot-info page was allocated.)

    // Hand off through a TRAMPOLINE in identity RAM. The kernel is an
    // identity-map kernel that expects the QEMU `-kernel` entry state: EL1,
    // MMU off, image coherent in RAM. We can't disable the MMU while executing
    // the stub itself (its code is UEFI-mapped; the PC would fault), so we copy
    // a tiny MMU-off-and-branch trampoline into a page of identity RAM (UEFI
    // maps allocated RAM at VA == PA on the `virt` machine) and jump through
    // it. After ExitBootServices we clean the D-cache for everything we wrote
    // (kernel image, model, trampoline) so RAM is coherent once caches go off.
    let tramp = alloc_at_any(4096);
    let tsrc = trampoline as extern "C" fn(u64) -> ! as usize as *const u8;
    // SAFETY: the trampoline fn is a short flat code sequence (< 128 bytes,
    // ends in `br`); copying its bytes to RAM and executing them there is the
    // whole point.
    unsafe { core::ptr::copy_nonoverlapping(tsrc, tramp.as_mut_ptr(), 128) };
    let tramp_pa = tramp.as_ptr() as u64;
    log::info!("chitti-stub: exiting boot services; MMU-off handoff via trampoline {tramp_pa:#x} -> {entry:#x}");

    // SAFETY: done with boot services; clean caches, then enter the trampoline
    // (identity address, so it survives the MMU turning off) with x0 = entry.
    unsafe {
        let _ = boot::exit_boot_services(Some(MemoryType::LOADER_DATA));
        clean_dcache(min_pa, max_end);
        if let Some((m, n)) = model_region {
            clean_dcache(m, m + n);
        }
        clean_dcache(tramp_pa, tramp_pa + 128);
        if let Some(bi) = bootinfo {
            clean_dcache(bi, bi + 4096);
        }
        // x0 = kernel entry, x1 = boot-info page (0 if none); the trampoline
        // preserves both (it scratches only x8) and branches to x0. The kernel
        // `_start` stashes x1 for `bootinfo_framebuffer`.
        core::arch::asm!(
            "ic iallu",
            "dsb sy",
            "isb",
            "mov x0, {e}",
            "mov x1, {bi}",
            "br {t}",
            e = in(reg) entry,
            bi = in(reg) bootinfo.unwrap_or(0),
            t = in(reg) tramp_pa,
            options(noreturn),
        );
    }
}

/// The MMU-off trampoline. Runs from identity RAM (copied there), so the PC
/// stays valid across the MMU switch-off. x0 = kernel entry. Position-
/// independent by construction (no loads, no branches except the final `br`).
#[unsafe(naked)]
extern "C" fn trampoline(_entry: u64) -> ! {
    core::arch::naked_asm!(
        "mrs x8, sctlr_el1",
        "bic x8, x8, #1",    // M = 0 (MMU off)
        "bic x8, x8, #4",    // C = 0 (data cache off)
        "bic x8, x8, #4096", // I = 0 (instruction cache off)
        "msr sctlr_el1, x8",
        "isb",
        "tlbi vmalle1",
        "dsb sy",
        "isb",
        "br x0",
    )
}

/// Allocate one+ pages of identity RAM anywhere below 4 GiB.
fn alloc_at_any(bytes: usize) -> &'static mut [u8] {
    let pages = bytes.div_ceil(4096);
    let ptr = boot::allocate_pages(
        AllocateType::MaxAddress(0xFFFF_F000),
        MemoryType::LOADER_DATA,
        pages,
    )
    .unwrap_or_else(|e| panic!("allocate_pages (any, {pages} pages) failed: {e:?}"));
    // SAFETY: freshly allocated pages.
    unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), pages * 4096) }
}

/// Clean the D-cache for `[start, end)` by VA (64-byte lines), so writes reach
/// RAM before the trampoline disables caches.
unsafe fn clean_dcache(start: u64, end: u64) {
    let line = 64u64;
    let mut a = start & !(line - 1);
    while a < end {
        // SAFETY: `dc cvac` on a mapped VA is safe.
        unsafe { core::arch::asm!("dc cvac, {}", in(reg) a, options(nostack, preserves_flags)) };
        a += line;
    }
    unsafe { core::arch::asm!("dsb sy", options(nostack, preserves_flags)) };
}
