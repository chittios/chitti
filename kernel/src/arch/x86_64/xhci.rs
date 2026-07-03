//! **xHCI USB host controller** + a **USB HID boot-keyboard** driver, so a real
//! USB keyboard (or QEMU's `-device qemu-xhci -device usb-kbd`) can drive the
//! shell on real x86 hardware — the modern counterpart to the PS/2 keyboard.
//!
//! Scope: a polled, single-device bring-up sufficient for a boot-protocol
//! keyboard — not a general USB stack. Stages (each gated in QEMU):
//!   C1 controller bring-up: find the PCI xHCI, map its MMIO, reset, set up the
//!      DCBAA + command ring + event ring, and start the controller.
//!   C2 enumerate: detect the port, reset it, enable a device slot, address it.
//!   C3 configure: read descriptors, pick the HID boot-keyboard interface,
//!      set the configuration, arm its interrupt IN endpoint.
//!   C4 input: poll the interrupt endpoint for 8-byte boot reports, map USB HID
//!      usage codes -> ASCII, and push them into the console ring.
//!
//! DMA memory comes from `mm::alloc_dma` (page-aligned, so the 64-byte xHCI
//! structure alignment is satisfied); the CPU reaches it through the HHDM.
// Staged bring-up: some fields/helpers are wired in later stages (C2-C4).
#![allow(dead_code)]

use crate::arch::x86_64::port::{inl, outl};
use crate::mm::{alloc_dma, map_mmio_page, Locked};
use core::ptr::{read_volatile, write_volatile};

/// The system's USB keyboard controller, brought up at boot. `console` polls it
/// for keystrokes alongside PS/2 + serial.
static XHCI: Locked<Option<Xhci>> = Locked::new(None);

/// Probe + bring up the xHCI controller and (later stages) enumerate a HID
/// keyboard. No-op if absent. Called once at boot on x86.
pub fn init_global() {
    if let Some(mut x) = Xhci::init() {
        x.enumerate_keyboard();
        XHCI.with(|s| *s = Some(x));
    }
}

/// The next byte from a USB keyboard, if any (drains HID reports). `None` if no
/// controller/keyboard or nothing pending.
pub fn poll_key() -> Option<u8> {
    XHCI.with(|s| s.as_mut().and_then(|x| x.poll_key()))
}

// --- minimal PCI config access (port 0xCF8/0xCFC) ----------------------
fn pci_addr(bus: u8, slot: u8, func: u8, off: u8) -> u32 {
    0x8000_0000 | ((bus as u32) << 16) | ((slot as u32) << 11) | ((func as u32) << 8) | ((off as u32) & 0xfc)
}
fn cfg_read32(bus: u8, slot: u8, func: u8, off: u8) -> u32 {
    // SAFETY: standard PCI config ports.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, off));
        inl(0xcfc)
    }
}
fn cfg_write32(bus: u8, slot: u8, func: u8, off: u8, v: u32) {
    // SAFETY: standard PCI config ports.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, off));
        outl(0xcfc, v);
    }
}

// --- xHCI capability register offsets (from BAR base) ------------------
const CAP_CAPLENGTH: usize = 0x00; // u8 caplength, u16 version at +2
const CAP_HCSPARAMS1: usize = 0x04;
const CAP_HCSPARAMS2: usize = 0x08;
const CAP_DBOFF: usize = 0x14;
const CAP_RTSOFF: usize = 0x18;

// --- operational register offsets (from BAR + caplength) ---------------
const OP_USBCMD: usize = 0x00;
const OP_USBSTS: usize = 0x04;
const OP_CRCR: usize = 0x18; // 64-bit
const OP_DCBAAP: usize = 0x30; // 64-bit
const OP_CONFIG: usize = 0x38;
const OP_PORTS: usize = 0x400; // PORTSC[0] at op+0x400, stride 0x10

const USBCMD_RUN: u32 = 1 << 0;
const USBCMD_HCRST: u32 = 1 << 1;
const USBSTS_HCH: u32 = 1 << 0; // HCHalted
const USBSTS_CNR: u32 = 1 << 11; // Controller Not Ready

// TRB types we use.
const TRB_LINK: u32 = 6;

/// A 16-byte Transfer Request Block.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct Trb {
    param: u64,
    status: u32,
    control: u32,
}

/// The controller + its rings. Addresses are stored both as physical (for the
/// device) and virtual (HHDM, for the CPU).
pub struct Xhci {
    mmio: usize,     // virtual base of the mapped register space
    op: usize,       // virtual base of operational registers
    rt: usize,       // virtual base of runtime registers
    db: usize,       // virtual base of the doorbell array
    max_ports: u8,
    max_slots: u8,
    // Command ring.
    cmd_ring_va: usize,
    cmd_ring_pa: u64,
    cmd_enqueue: usize,
    cmd_cycle: u32,
    // Event ring.
    evt_ring_va: usize,
    evt_ring_pa: u64,
    evt_dequeue: usize,
    evt_cycle: u32,
    // Device context base address array.
    dcbaa_va: usize,
    dcbaa_pa: u64,
}

const RING_TRBS: usize = 64; // TRBs per ring (last is a Link on the command ring)

unsafe fn r32(addr: usize) -> u32 {
    unsafe { read_volatile(addr as *const u32) }
}
unsafe fn w32(addr: usize, v: u32) {
    unsafe { write_volatile(addr as *mut u32, v) };
}
unsafe fn w64(addr: usize, v: u64) {
    // xHCI 64-bit registers: write low then high dword.
    unsafe {
        write_volatile(addr as *mut u32, v as u32);
        write_volatile((addr + 4) as *mut u32, (v >> 32) as u32);
    }
}

impl Xhci {
    /// Find the PCI xHCI controller (class 0x0C / subclass 0x03 / prog-if 0x30),
    /// map its MMIO, and bring the controller up. Returns `None` if absent.
    pub fn init() -> Option<Xhci> {
        let (bus, slot, func) = find_xhci()?;
        crate::ktrace::log_fmt(format_args!("xhci: controller at {bus:02x}:{slot:02x}.{func}"));

        // BAR0 (memory, possibly 64-bit).
        let bar0 = cfg_read32(bus, slot, func, 0x10);
        if bar0 & 0x1 != 0 {
            return None; // I/O BAR — not xHCI MMIO
        }
        let mut phys = (bar0 & 0xffff_fff0) as u64;
        if (bar0 >> 1) & 0x3 == 0x2 {
            // 64-bit BAR: high half in BAR1.
            phys |= (cfg_read32(bus, slot, func, 0x14) as u64) << 32;
        }
        // Enable memory space (bit1) + bus master DMA (bit2).
        let cmd = cfg_read32(bus, slot, func, 0x04);
        cfg_write32(bus, slot, func, 0x04, cmd | 0b110);

        // Map the register window (32 KiB is plenty for qemu-xhci) uncached.
        let mmio = map_mmio_page(phys) as usize;
        for i in 1u64..8 {
            map_mmio_page(phys + i * 0x1000);
        }

        // SAFETY: `mmio` is the mapped xHCI register block.
        unsafe {
            let caplen = (r32(mmio + CAP_CAPLENGTH) & 0xff) as usize;
            let op = mmio + caplen;
            let rt = mmio + (r32(mmio + CAP_RTSOFF) & !0x1f) as usize;
            let db = mmio + (r32(mmio + CAP_DBOFF) & !0x3) as usize;
            let hcs1 = r32(mmio + CAP_HCSPARAMS1);
            let max_slots = (hcs1 & 0xff) as u8;
            let max_ports = ((hcs1 >> 24) & 0xff) as u8;

            // Wait for CNR to clear, then reset.
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {}
            w32(op + OP_USBCMD, USBCMD_HCRST);
            while r32(op + OP_USBCMD) & USBCMD_HCRST != 0 {}
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {}

            // Program enabled device slots.
            w32(op + OP_CONFIG, max_slots as u32);

            // Scratchpad buffers, if the controller wants them (HCSPARAMS2).
            let hcs2 = r32(mmio + CAP_HCSPARAMS2);
            let nscratch = (((hcs2 >> 21) & 0x1f) | ((hcs2 >> 27) & 0x1f) << 5) as usize;

            // Device Context Base Address Array: (max_slots + 1) * 8 bytes.
            let (dcbaa_pa, dcbaa_va) = alloc_dma(4096)?;
            let dcbaa_va = dcbaa_va as usize;
            if nscratch > 0 {
                // Scratchpad buffer array + buffers (one page each).
                let (sp_arr_pa, sp_arr_va) = alloc_dma(nscratch * 8)?;
                for i in 0..nscratch {
                    let (buf_pa, _) = alloc_dma(4096)?;
                    write_volatile((sp_arr_va as *mut u64).add(i), buf_pa);
                }
                write_volatile(dcbaa_va as *mut u64, sp_arr_pa); // DCBAA[0] -> scratchpad array
            }
            w64(op + OP_DCBAAP, dcbaa_pa);

            // Command ring.
            let (cmd_ring_pa, cmd_ring_va) = alloc_dma(RING_TRBS * 16)?;
            let cmd_ring_va = cmd_ring_va as usize;
            // Last TRB is a Link back to the start (keeps the ring circular).
            let link = (cmd_ring_va + (RING_TRBS - 1) * 16) as *mut Trb;
            write_volatile(
                link,
                Trb { param: cmd_ring_pa, status: 0, control: (TRB_LINK << 10) | 0x2 /*TC*/ | 0x1 /*cycle*/ },
            );
            // CRCR = ring pointer | RCS(1).
            w64(op + OP_CRCR, cmd_ring_pa | 1);

            // Event ring: one segment + a 1-entry ERST.
            let (evt_ring_pa, evt_ring_va) = alloc_dma(RING_TRBS * 16)?;
            let evt_ring_va = evt_ring_va as usize;
            let (erst_pa, erst_va) = alloc_dma(64)?;
            let erst_va = erst_va as usize;
            write_volatile(erst_va as *mut u64, evt_ring_pa); // segment base
            write_volatile((erst_va + 8) as *mut u32, RING_TRBS as u32); // segment size
            // Interrupter 0 registers at rt + 0x20.
            let ir0 = rt + 0x20;
            w32(ir0 + 0x08, 1); // ERSTSZ = 1
            w64(ir0 + 0x10, erst_pa); // ERSTBA -- actually ERDP first per some HCs; set ERDP then ERSTBA
            w64(ir0 + 0x18, evt_ring_pa); // ERDP
            w64(ir0 + 0x10, erst_pa); // ERSTBA (writing this arms the interrupter)

            // Run.
            let c = r32(op + OP_USBCMD);
            w32(op + OP_USBCMD, c | USBCMD_RUN);
            // Wait for HCHalted to clear.
            let mut spins = 0u32;
            while r32(op + OP_USBSTS) & USBSTS_HCH != 0 {
                spins += 1;
                if spins > 1_000_000 {
                    crate::ktrace::log("xhci", "controller did not leave halted state");
                    return None;
                }
            }

            crate::ktrace::log_fmt(format_args!(
                "xhci: running (caplen {caplen}, {max_slots} slots, {max_ports} ports, {nscratch} scratchpad)"
            ));
            Some(Xhci {
                mmio,
                op,
                rt,
                db,
                max_ports,
                max_slots,
                cmd_ring_va,
                cmd_ring_pa,
                cmd_enqueue: 0,
                cmd_cycle: 1,
                evt_ring_va,
                evt_ring_pa,
                evt_dequeue: 0,
                evt_cycle: 1,
                dcbaa_va,
                dcbaa_pa,
            })
        }
    }

    /// PORTSC of 1-based `port`.
    fn portsc(&self, port: u8) -> usize {
        self.op + OP_PORTS + (port as usize - 1) * 0x10
    }

    /// C2/C3: find an attached keyboard, address + configure it. Filled in the
    /// next stages; a no-op keeps C1 self-contained.
    fn enumerate_keyboard(&mut self) {}

    /// C4: drain any pending HID boot report into an ASCII byte.
    fn poll_key(&mut self) -> Option<u8> {
        None
    }
}

/// Scan PCI (buses 0..=1, all slots/funcs) for an xHCI controller.
fn find_xhci() -> Option<(u8, u8, u8)> {
    for bus in 0u8..=1 {
        for slot in 0u8..32 {
            let id = cfg_read32(bus, slot, 0, 0x00);
            if id == 0xffff_ffff {
                continue;
            }
            let header = (cfg_read32(bus, slot, 0, 0x0c) >> 16) & 0xff;
            let nfuncs = if header & 0x80 != 0 { 8 } else { 1 };
            for func in 0..nfuncs {
                let id = cfg_read32(bus, slot, func, 0x00);
                if id == 0xffff_ffff {
                    continue;
                }
                let class = cfg_read32(bus, slot, func, 0x08);
                let (base, sub, progif) = ((class >> 24) & 0xff, (class >> 16) & 0xff, (class >> 8) & 0xff);
                if base == 0x0c && sub == 0x03 && progif == 0x30 {
                    return Some((bus, slot, func));
                }
            }
        }
    }
    None
}
