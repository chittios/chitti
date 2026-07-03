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

/// The aarch64 model load address the kernel's `cortex::model_module` reads
/// (mirrors QEMU `-device loader` on the `-kernel` path). 0.8B layout.
const MODEL_ADDR: u64 = 0x4800_0000;

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

#[entry]
fn main() -> Status {
    uefi::helpers::init().expect("uefi init");
    log::info!("chitti-stub: loading kernel off the ESP");

    // Read the kernel ELF (and the model, if present) from the boot volume.
    let mut fs = uefi::fs::FileSystem::new(boot::get_image_file_system(boot::image_handle()).expect("no ESP filesystem"));
    let kernel = fs.read(cstr16!("\\chitti-kernel")).expect("read \\chitti-kernel");
    log::info!("chitti-stub: kernel {} bytes", kernel.len());

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

    // Load the model to the fixed address the kernel reads, if bundled.
    match fs.read(cstr16!("\\model.gguf.000")) {
        Ok(model) => {
            let dst = alloc_at(MODEL_ADDR, model.len());
            dst[..model.len()].copy_from_slice(&model);
            log::info!("chitti-stub: model {} bytes at {MODEL_ADDR:#x}", model.len());
        }
        Err(_) => log::info!("chitti-stub: no model on ESP (kernel will report no model)"),
    }

    // Reserve the kernel's FIXED regions so UEFI hasn't parked runtime/ACPI data
    // where the `-kernel` layout puts them (the kernel hardcodes these, as it
    // does under QEMU `-kernel` where they're guaranteed free RAM): the 256 MiB
    // heap at 0x80000000, and the model window at 0x48000000 (already backed if
    // a model was loaded above). Marking them LOADER_DATA keeps them ours across
    // ExitBootServices and stops UEFI reusing them.
    const HEAP_BASE: u64 = 0x8000_0000;
    const HEAP_PAGES: usize = 256 * 1024 * 1024 / 4096;
    match boot::allocate_pages(AllocateType::Address(HEAP_BASE), MemoryType::LOADER_DATA, HEAP_PAGES) {
        Ok(_) => log::info!("chitti-stub: reserved kernel heap {HEAP_BASE:#x} (256 MiB)"),
        Err(e) => log::warn!("chitti-stub: could NOT reserve heap {HEAP_BASE:#x}: {e:?} (UEFI may collide)"),
    }

    log::info!("chitti-stub: exiting boot services, jumping to kernel at {entry:#x} (MMU on)");
    // SAFETY: we are done with boot services; the memory map is discarded (the
    // kernel builds its own via mmu::init). MMU + caches stay on (UEFI identity
    // map), so the loaded image is coherent when the kernel reads it.
    unsafe {
        let _ = boot::exit_boot_services(Some(MemoryType::LOADER_DATA));
        core::arch::asm!("br {}", in(reg) entry, options(noreturn));
    }
}
