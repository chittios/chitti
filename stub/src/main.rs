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

/// The kernel's EDID parser, mounted directly so the stub and the kernel cannot
/// disagree about a display's native resolution (and so the bit-packing is
/// covered by `cargo xtask test`, which the stub crate has no harness for).
#[path = "../../kernel/src/edid.rs"]
mod edid;

/// `EFI_EDID_ACTIVE_PROTOCOL` / `EFI_EDID_DISCOVERED_PROTOCOL` — the EDID the
/// firmware is currently using for a display, and the raw one it read from the
/// monitor. Not in the `uefi` crate, so declared here; both protocols share this
/// layout, sitting on the same handle as the `GraphicsOutput` protocol.
#[repr(C)]
struct EdidProtocol {
    size_of_edid: u32,
    edid: *const u8,
}

/// `EFI_EDID_ACTIVE_PROTOCOL_GUID`.
const EDID_ACTIVE_GUID: uefi::Guid = uefi::guid!("bd8c1056-9f36-44ec-92a8-a6337f817986");
/// `EFI_EDID_DISCOVERED_PROTOCOL_GUID`.
const EDID_DISCOVERED_GUID: uefi::Guid = uefi::guid!("1c0c34f6-d380-41fa-a049-8ad06c1a66aa");

/// The EDID the firmware settled on for this display.
#[repr(transparent)]
#[uefi::proto::unsafe_protocol(EDID_ACTIVE_GUID)]
struct EdidActive(EdidProtocol);

/// The raw EDID the firmware read from the monitor.
#[repr(transparent)]
#[uefi::proto::unsafe_protocol(EDID_DISCOVERED_GUID)]
struct EdidDiscovered(EdidProtocol);

/// Both EDID protocols share a layout; this lets one reader serve both.
trait AsEdid {
    fn as_edid(&self) -> &EdidProtocol;
}
impl AsEdid for EdidActive {
    fn as_edid(&self) -> &EdidProtocol {
        &self.0
    }
}
impl AsEdid for EdidDiscovered {
    fn as_edid(&self) -> &EdidProtocol {
        &self.0
    }
}

/// Read one EDID protocol off `handle` and return its **base block**.
fn edid_from<T: uefi::proto::ProtocolPointer + AsEdid>(
    handle: uefi::Handle,
) -> Option<alloc::vec::Vec<u8>> {
    // SAFETY: GetProtocol only reads the interface pointer — the firmware console
    // already owns this handle, so an exclusive open would be denied.
    let p = unsafe {
        boot::open_protocol::<T>(
            boot::OpenProtocolParams { handle, agent: boot::image_handle(), controller: None },
            boot::OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    let e = p.as_edid();
    let (len, ptr) = (e.size_of_edid as usize, e.edid);
    // A protocol that exists but describes no data is common on virtual GPUs.
    if ptr.is_null() || len < edid::BASE_BLOCK_LEN {
        return None;
    }
    // SAFETY: the firmware owns this buffer and reports its length; we read only
    // `len` bytes, and the borrow ends before boot services are exited.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    // Copy it out: the firmware's buffer is gone after ExitBootServices, and the
    // kernel needs these bytes to identify the display.
    edid::is_valid(bytes).then(|| bytes[..edid::BASE_BLOCK_LEN].to_vec())
}

/// `EFI_CONSOLE_OUT_DEVICE_GUID` — a marker (no interface) the firmware installs
/// on the handle(s) it uses for console output.
const CONSOLE_OUT_GUID: uefi::Guid = uefi::guid!("d3b36f2c-d551-11d4-9a46-0090273fc14d");

/// A zero-sized stand-in for the console-out marker, which carries no interface —
/// only its presence on a handle is meaningful.
#[repr(transparent)]
#[uefi::proto::unsafe_protocol(CONSOLE_OUT_GUID)]
struct ConsoleOutDevice(u8);

/// Whether the firmware draws its own console on `handle`.
///
/// This is the signal that identifies *the display the user is looking at* on a
/// machine with more than one output: the firmware's boot messages went there.
/// `test_protocol` only asks whether the marker is installed — it opens nothing,
/// which matters because the console splitter already owns these handles.
fn has_console_out(handle: uefi::Handle) -> bool {
    boot::test_protocol::<ConsoleOutDevice>(boot::OpenProtocolParams {
        handle,
        agent: boot::image_handle(),
        controller: None,
    })
    .unwrap_or(false)
}

/// The display's EDID base block on `handle`, if it publishes a valid one.
///
/// Tries the **active** EDID first (what the firmware settled on) and falls back
/// to the **discovered** one (what the monitor actually said). Absent on most
/// hypervisors and on headless boots, which is a legitimate answer — the caller
/// must then keep the firmware's mode rather than invent one.
fn edid_block(handle: uefi::Handle) -> Option<alloc::vec::Vec<u8>> {
    edid_from::<EdidActive>(handle).or_else(|| edid_from::<EdidDiscovered>(handle))
}

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

/// Allocate `pages` from conventional RAM **at or above `min_addr`**, falling
/// back to AnyPages when no such region exists. The kernel's identity map
/// types GiB block 0 as **Device MMIO**; on platforms whose RAM includes
/// low physical memory (VirtualBox-ARM), an AnyPages model/heap allocation
/// can land there — byte reads work but the NEON matvec's unaligned vector
/// loads take an alignment fault (Device memory), and everything is uncached.
/// So the big regions the kernel computes over must sit >= 1 GiB.
fn alloc_pages_min_addr(pages: usize, min_addr: u64) -> uefi::Result<core::ptr::NonNull<u8>> {
    use uefi::mem::memory_map::MemoryMap;
    let need = pages as u64 * 4096;
    if let Ok(mm) = boot::memory_map(MemoryType::LOADER_DATA) {
        // Pick the highest fitting conventional region top >= min_addr (the
        // top stays clear of the kernel image and other low allocations).
        let mut best: Option<u64> = None;
        for d in mm.entries() {
            if d.ty != MemoryType::CONVENTIONAL {
                continue;
            }
            let start = d.phys_start.max(min_addr);
            let end = d.phys_start + d.page_count * 4096;
            if end > start && end - start >= need {
                let base = (end - need) & !0xfff;
                if base >= start {
                    best = Some(best.map_or(base, |b: u64| b.max(base)));
                }
            }
        }
        if let Some(base) = best {
            if let Ok(p) = boot::allocate_pages(AllocateType::Address(base), MemoryType::LOADER_DATA, pages) {
                return Ok(p);
            }
        }
    }
    // No suitable high region (tiny VM): AnyPages keeps small setups booting.
    boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages)
}

/// Reassemble the model from split parts `\model.gguf.000`, `.001`, … into one
/// contiguous allocation, returning `(base, len)`. FAT32 caps a single file at
/// 4 GiB, so the image build splits a large model (the 9B) into <= 1 GiB parts;
/// every loader path concatenates the sorted parts, and this is that path for
/// the UEFI/ESP boot. A lone `\model.gguf.000` is just the one-part case.
/// Returns `None` (boot without a model) if no parts are present or the
/// allocation fails.
fn load_model(fs: &mut uefi::fs::FileSystem) -> Option<(u64, u64)> {
    use uefi::fs::PathBuf;
    use uefi::CString16;
    // Pass 1: total the sizes of the consecutive parts (metadata only — no data
    // read yet), so we can allocate one contiguous region up front.
    let mut parts: alloc::vec::Vec<PathBuf> = alloc::vec::Vec::new();
    let mut total: usize = 0;
    for idx in 0.. {
        let name = alloc::format!("\\model.gguf.{idx:03}");
        let path = PathBuf::from(CString16::try_from(name.as_str()).expect("model part path"));
        match fs.metadata(&path) {
            Ok(info) => {
                total += info.file_size() as usize;
                parts.push(path);
            }
            Err(_) => break,
        }
    }
    if parts.is_empty() || total == 0 {
        log::info!("chitti-stub: no model on ESP (kernel will report no model)");
        return None;
    }
    let pages = total.div_ceil(4096);
    // >= 1 GiB: the kernel maps GiB block 0 as Device — NEON over a model
    // placed there alignment-faults (see `alloc_pages_min_addr`).
    let ptr = match alloc_pages_min_addr(pages, 1 << 30) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("chitti-stub: model alloc ({pages} pages) failed: {e:?} -- booting without a model (need more VM RAM)");
            return None;
        }
    };
    let base = ptr.as_ptr() as u64;
    // SAFETY: freshly allocated, `pages * 4096` contiguous bytes at `base`.
    let dst = unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), pages * 4096) };
    // Pass 2: read each part (one at a time — the transient buffer is one part,
    // <= 1 GiB) and copy it into the contiguous region, in order.
    let mut off = 0usize;
    for path in &parts {
        match fs.read(path) {
            Ok(bytes) => {
                dst[off..off + bytes.len()].copy_from_slice(&bytes);
                off += bytes.len();
            }
            Err(e) => {
                log::warn!("chitti-stub: reading model part failed: {e:?} -- booting without a model");
                return None;
            }
        }
    }
    log::info!("chitti-stub: model {off} bytes at {base:#x} ({} part(s))", parts.len());
    Some((base, off as u64))
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
    let Ok(t) = uefi::runtime::get_time() else { return 0 };
    let secs = days_from_civil(t.year() as i64, t.month() as i64, t.day() as i64) * 86400
        + t.hour() as i64 * 3600
        + t.minute() as i64 * 60
        + t.second() as i64;
    // EFI time may carry a timezone offset (minutes east of UTC); 2047 = unspecified.
    let secs = match t.time_zone() {
        Some(tz) if tz != 2047 => secs - tz as i64 * 60,
        _ => secs,
    };
    if secs > 0 { secs as u64 } else { 0 }
}

/// Total installed physical RAM, from the UEFI memory map: the span from the
/// lowest to the highest RAM-backed descriptor (everything except the two MMIO
/// types). Approximates the machine's DRAM (e.g. 6 GiB) closely enough for a
/// status display; 0 if the map can't be read. Must be called before
/// `exit_boot_services`.
fn total_ram_bytes() -> u64 {
    use uefi::boot::MemoryType;
    use uefi::mem::memory_map::MemoryMap; // brings `entries()` into scope
    let Ok(map) = boot::memory_map(MemoryType::LOADER_DATA) else {
        return 0;
    };
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for d in map.entries() {
        // Skip memory-mapped I/O (11) and I/O port space (12): not DRAM.
        if matches!(d.ty, MemoryType::MMIO | MemoryType::MMIO_PORT_SPACE) {
            continue;
        }
        let start = d.phys_start;
        let end = start + d.page_count * 4096;
        if start < lo {
            lo = start;
        }
        if end > hi {
            hi = end;
        }
    }
    if hi > lo {
        hi - lo
    } else {
        0
    }
}

/// The machine's actual RAM extents from the UEFI memory map: every DRAM-backed
/// descriptor (everything except the two MMIO types and firmware-reserved),
/// sorted and merged into `(base, size)` clumps. Real machines and
/// VirtualBox-ARM interleave RAM and MMIO inside the same GiB, so the kernel
/// needs the true extents — not just a total — to type its identity map
/// (Normal over MMIO breaks the framebuffer/ECAM; Device over RAM
/// alignment-faults NEON loads). Must run before `exit_boot_services`.
fn ram_regions() -> alloc::vec::Vec<(u64, u64)> {
    use uefi::boot::MemoryType;
    use uefi::mem::memory_map::MemoryMap;
    let Ok(map) = boot::memory_map(MemoryType::LOADER_DATA) else {
        return alloc::vec::Vec::new();
    };
    let mut ext: alloc::vec::Vec<(u64, u64)> = map
        .entries()
        .filter(|d| !matches!(d.ty, MemoryType::MMIO | MemoryType::MMIO_PORT_SPACE | MemoryType::RESERVED))
        .map(|d| (d.phys_start, d.page_count * 4096))
        .filter(|&(_, s)| s > 0)
        .collect();
    ext.sort_unstable();
    let mut merged: alloc::vec::Vec<(u64, u64)> = alloc::vec::Vec::new();
    for (b, s) in ext {
        match merged.last_mut() {
            Some((mb, ms)) if b <= *mb + *ms => {
                let end = (b + s).max(*mb + *ms);
                *ms = end - *mb;
            }
            _ => merged.push((b, s)),
        }
    }
    merged
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
    let mut fs = uefi::fs::FileSystem::new(boot::get_image_file_system(boot::image_handle()).expect("no ESP filesystem"));
    let kernel = fs.read(cstr16!("\\chitti-kernel")).expect("read \\chitti-kernel");
    log::info!("chitti-stub: kernel {} bytes", kernel.len());

    // The human's mode preference, if they left one on the ESP. This is the only
    // channel that exists this early: the rest of the display settings live on the
    // ext4 data partition, which nothing here can read, and the framebuffer's
    // dimensions are fixed the moment GOP hands it over. Absent file → None, and
    // the firmware's own configuration decides as before.
    //
    // It has to be honoured even where a hypervisor's resolution knob exists,
    // because some of them are ignored: VirtualBox-ARM accepts
    // `VBoxInternal2/EfiGraphicsResolution` and boots at its own resolution anyway.
    let boot_res: Option<(u32, u32)> = fs
        .read(cstr16!("\\chitti-display.cfg"))
        .ok()
        .and_then(|b| core::str::from_utf8(&b).ok().and_then(edid::parse_boot_cfg));
    if let Some((w, h)) = boot_res {
        log::info!("chitti-stub: {} asks for {w}x{h}", edid::BOOT_CFG_PATH);
    }

    // Parse the ELF64 header; find the PT_LOAD span, allocate it once (segments
    // can share page boundaries, so per-segment AllocateAddress conflicts), then
    // copy each segment to its physical address.
    assert_eq!(&kernel[0..4], b"\x7fELF", "not an ELF");
    let entry = le64(&kernel, 24);
    let phoff = le64(&kernel, 32) as usize;
    let phentsize = le16(&kernel, 54) as usize;
    let phnum = le16(&kernel, 56) as usize;
    let loads = || (0..phnum).map(|i| phoff + i * phentsize).filter(|&ph| le32(&kernel, ph) == 1);
    let min_pa = loads().map(|ph| le64(&kernel, ph + 24)).min().expect("no PT_LOAD");
    let max_end = loads().map(|ph| le64(&kernel, ph + 24) + le64(&kernel, ph + 40)).max().unwrap();
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
    let model_region: Option<(u64, u64)> = load_model(&mut fs);

    // Reserve the kernel heap in free RAM (>= 1 GiB, else AnyPages; HEAP_MAX)
    // and mark it LOADER_DATA so it survives ExitBootServices. Report its
    // base; the kernel places its heap here. A firmware-chosen address (not a
    // fixed one, and not the top of RAM where UEFI parks ACPI/runtime data)
    // is what makes this robust across firmwares; the >= 1 GiB floor keeps it
    // out of the kernel's Device-typed GiB block 0 (VirtualBox-ARM has RAM
    // there — a heap in Device memory is uncached and NEON-hostile).
    let heap_region: Option<(u64, u64)> = {
        let pages = (HEAP_MAX / 4096) as usize;
        match alloc_pages_min_addr(pages, 1 << 30) {
            Ok(ptr) => {
                let base = ptr.as_ptr() as u64;
                log::info!("chitti-stub: reserved kernel heap {base:#x} ({} MiB)", HEAP_MAX >> 20);
                Some((base, HEAP_MAX))
            }
            Err(e) => {
                log::warn!("chitti-stub: heap reserve ({pages} pages) failed: {e:?} (need more VM RAM)");
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
        // Enumerate **every** graphics output, not just the first. A real laptop
        // with an external monitor attached exposes one GOP handle per display, so
        // taking `get_handle_for_protocol` (= handle[0]) read the EDID of, and set
        // the mode on, whichever output the firmware happened to list first — a
        // coin flip between the built-in panel and the monitor.
        let handles = match boot::locate_handle_buffer(
            boot::SearchType::from_proto::<GraphicsOutput>(),
        ) {
            Ok(hs) if !hs.is_empty() => hs,
            _ => {
                log::info!("chitti-stub: no GOP handle");
                return None;
            }
        };
        // Describe each output, then let the shared policy choose (see
        // `edid::pick_output`: firmware console-out first, then any output with a
        // connected display, then output 0).
        let mut infos = alloc::vec::Vec::with_capacity(handles.len());
        let mut blocks: alloc::vec::Vec<Option<alloc::vec::Vec<u8>>> =
            alloc::vec::Vec::with_capacity(handles.len());
        for (i, &hh) in handles.iter().enumerate() {
            let block = edid_block(hh);
            let info = edid::OutputInfo {
                console_out: has_console_out(hh),
                edid_native: block.as_deref().and_then(edid::preferred_resolution),
            };
            log::info!(
                "chitti-stub: GOP output {i}: console_out={} edid={:?} name={:?}",
                info.console_out,
                info.edid_native,
                block.as_deref().and_then(edid::monitor_name)
            );
            infos.push(info);
            blocks.push(block);
        }
        let out_idx = edid::pick_output(&infos)?;
        let h = handles[out_idx];
        if handles.len() > 1 {
            log::info!("chitti-stub: driving GOP output {out_idx} of {}", handles.len());
        }
        // Non-exclusive open: the firmware console (ConSplitter) already owns
        // GOP, so an exclusive open is denied. GetProtocol just reads it.
        let mut gop = match unsafe {
            boot::open_protocol::<GraphicsOutput>(
                boot::OpenProtocolParams { handle: h, agent: boot::image_handle(), controller: None },
                boot::OpenProtocolAttributes::GetProtocol,
            )
        } {
            Ok(g) => g,
            Err(e) => {
                log::info!("chitti-stub: GOP open failed: {e:?}");
                return None;
            }
        };
        // Choose the display mode. Three cases, in order — the ordering is the
        // whole point, because "biggest mode advertised" is NOT "native":
        //
        //  1. The display reports an EDID → use its **preferred timing**, the
        //     panel's real native resolution. This is what firmware and every
        //     other OS loader do.
        //  2. No EDID → keep whatever mode the firmware is already in. On a
        //     hypervisor that is the resolution the VM was *configured* for
        //     (VirtualBox `VBoxInternal2/EfiGraphicsResolution`, UTM's display
        //     setting), so overriding it ignored the user's choice — this stub
        //     used to always jump to the largest advertised mode, which is why
        //     a VirtualBox guest came up at a fixed 2560x1440 no matter what the
        //     VM was set to.
        //  3. No EDID *and* the firmware left a mode nobody would choose
        //     (< 1024x768 — the classic real-hardware "UEFI came up at 800x600")
        //     → then, and only then, take the largest advertised mode.
        //
        // `Mode` is `Copy`, so collect candidates while the read-only `modes()`
        // iterator is alive, then `set_mode` outside it.
        let native = infos[out_idx].edid_native;
        let cur = gop.current_mode_info().resolution();
        let modes: alloc::vec::Vec<uefi::proto::console::gop::Mode> = gop
            .modes()
            .filter(|m| m.info().pixel_format() != PixelFormat::BltOnly)
            .collect();
        let dims: alloc::vec::Vec<(usize, u32, u32)> = modes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let (w, hh) = m.info().resolution();
                (i, w as u32, hh as u32)
            })
            .collect();
        // Log the candidate list. Without it, "the resolution I asked for did not
        // happen" has two indistinguishable causes — the mode was not offered, or it
        // was offered and `set_mode` failed — and on a machine that will not boot
        // right, that distinction is the whole diagnosis.
        log::info!(
            "chitti-stub: GOP current {}x{}, {} usable mode(s): {:?}",
            cur.0,
            cur.1,
            dims.len(),
            dims.iter().map(|&(_, w, h)| (w, h)).collect::<alloc::vec::Vec<_>>()
        );
        let pick = match boot_res.or(native) {
            // A resolution asked for on the ESP outranks the display's own native
            // mode: it is the one preference here a human typed on purpose, and it
            // exists precisely for the cases where the automatic answer was wrong
            // (a hypervisor that ignores its resolution setting, a panel whose EDID
            // native mode is larger than the window showing it).
            Some((nw, nh)) => {
                if boot_res.is_some() {
                    log::info!("chitti-stub: requested {nw}x{nh}");
                } else {
                    log::info!("chitti-stub: EDID native {nw}x{nh}");
                }
                edid::best_mode_for((nw, nh), dims.iter().copied())
            }
            None if edid::is_implausibly_small(cur.0 as u32, cur.1 as u32) => {
                log::info!(
                    "chitti-stub: no EDID and firmware mode {}x{} is a default nobody chose — taking the largest mode",
                    cur.0,
                    cur.1
                );
                dims.iter()
                    .max_by_key(|(_, w, hh)| *w as u64 * *hh as u64)
                    .map(|&(i, _, _)| i)
            }
            None => {
                // Respect the platform's configured resolution.
                log::info!("chitti-stub: no EDID — keeping the firmware's {}x{} mode", cur.0, cur.1);
                None
            }
        };
        if let Some(i) = pick {
            let m = &modes[i];
            let (mw, mh) = m.info().resolution();
            // Say so when the request could not be met exactly. `best_mode_for` never
            // exceeds what was asked for — a request usually means "no bigger than
            // this", e.g. it has to fit a window — so the answer can be a good deal
            // smaller, and silently landing there looks like the file was ignored.
            if let Some((rw, rh)) = boot_res {
                if (mw as u32, mh as u32) != (rw, rh) {
                    log::info!(
                        "chitti-stub: {rw}x{rh} is not offered; closest that fits is {mw}x{mh} \
                         (the firmware's mode list is above)"
                    );
                }
            }
            if (mw, mh) == cur {
                log::info!("chitti-stub: GOP already at {mw}x{mh}");
            } else if let Err(e) = gop.set_mode(m) {
                log::info!("chitti-stub: GOP set_mode {mw}x{mh} failed: {e:?} (keeping current)");
            } else {
                log::info!("chitti-stub: GOP set_mode {mw}x{mh}");
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
                Some(m) => (m.red.trailing_zeros() as u8, m.green.trailing_zeros() as u8, m.blue.trailing_zeros() as u8),
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
        // Total physical RAM at 92..100: the span of RAM-backed descriptors in
        // the UEFI memory map. This is the machine's installed RAM (e.g. 6 GiB)
        // the kernel shows in the status bar / `/top`, distinct from its fixed
        // heap. 0 if the map can't be read.
        let ram = total_ram_bytes();
        page[92..100].copy_from_slice(&ram.to_le_bytes());
        // RAM extents at 104.. : count@104, then up to 16 (base@0, size@8)
        // pairs from 112 — the kernel types its identity map from these (RAM
        // and MMIO interleave inside GiB blocks on VirtualBox/real hardware).
        let regions = ram_regions();
        let n = regions.len().min(16);
        if regions.len() > 16 {
            log::info!("chitti-stub: {} RAM extents; passing first 16", regions.len());
        }
        page[104..112].copy_from_slice(&(n as u64).to_le_bytes());
        for (i, &(rb, rsz)) in regions.iter().take(n).enumerate() {
            page[112 + i * 16..120 + i * 16].copy_from_slice(&rb.to_le_bytes());
            page[120 + i * 16..128 + i * 16].copy_from_slice(&rsz.to_le_bytes());
        }
        // The chosen output's EDID base block: length at 384..388, then the 128
        // bytes at 388..516 (RAM extents end at 368, so this is free space).
        //
        // Passing the raw block — rather than a resolution the stub already
        // digested — is the same handoff Linux's EFI stub does: the kernel needs
        // the display's *identity* (vendor/product/serial) to keep per-monitor
        // settings, and its name to show which output it is talking about. The
        // firmware's buffer is gone after ExitBootServices, so it has to be copied
        // here or not at all. Zero length = this display published no EDID.
        let edid_bytes: &[u8] = blocks
            .get(out_idx)
            .and_then(|b| b.as_deref())
            .unwrap_or(&[]);
        let elen = edid_bytes.len().min(edid::BASE_BLOCK_LEN);
        page[384..388].copy_from_slice(&(elen as u32).to_le_bytes());
        page[388..388 + elen].copy_from_slice(&edid_bytes[..elen]);
        log::info!("chitti-stub: GOP {w}x{hgt} at {fb:#x} (shifts {rs}/{gs}/{bs}), ACPI RSDP {rsdp:#x}, heap {hb:#x}, model {mb:#x}, RAM {} MiB in {n} extent(s), EDID {elen}B -> boot-info {addr:#x}", ram >> 20);
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
        "bic x8, x8, #1",      // M = 0 (MMU off)
        "bic x8, x8, #4",      // C = 0 (data cache off)
        "bic x8, x8, #4096",   // I = 0 (instruction cache off)
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
    let ptr = boot::allocate_pages(AllocateType::MaxAddress(0xFFFF_F000), MemoryType::LOADER_DATA, pages)
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
