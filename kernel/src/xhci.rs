//! **xHCI USB host controller** + a **USB HID boot-keyboard** driver, so a real
//! USB keyboard (on real hardware, or QEMU's `-device qemu-xhci -device
//! usb-kbd`) can drive the shell — the modern counterpart to PS/2. This is the
//! **arch-neutral core**; the per-arch wrappers (`arch::x86_64::xhci`,
//! `arch::aarch64::xhci`) discover the controller, map its MMIO, and hand this
//! core the register base + a DMA allocator, then reuse it verbatim.
//!
//! Scope: a polled, single-device bring-up sufficient for a boot-protocol
//! keyboard — not a general USB stack. Stages:
//!   C1 controller bring-up: reset, set up the DCBAA + command ring + event
//!      ring, start the controller (from a wrapper-supplied MMIO base).
//!   C2 enumerate: detect the port, reset it, enable a device slot, address it.
//!   C3 configure: read descriptors, pick the HID boot-keyboard interface,
//!      set the configuration, arm its interrupt IN endpoint.
//!   C4 input: poll the interrupt endpoint for 8-byte boot reports, map USB HID
//!      usage codes -> ASCII, and push them into the console ring.
//!
//! DMA memory comes from the wrapper's `Alloc` (returns `(physical, virtual)`):
//! the device is handed the physical address, the CPU uses the virtual one
//! (identical on the aarch64 identity map; HHDM on x86).
// Staged bring-up: some fields/helpers are wired in later stages (C2-C4).
#![allow(dead_code)]

use core::ptr::{read_volatile, write_volatile};

/// A page-aligned DMA allocator: `bytes -> (physical_addr, virtual_addr)`.
pub type Alloc = fn(usize) -> Option<(u64, usize)>;

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
    // Context structure size: 32 or 64 bytes (HCCPARAMS1.CSZ).
    ctx_size: usize,
    // Wrapper-supplied DMA allocator (arch-specific phys/virt mapping).
    alloc: Alloc,
    // The enumerated keyboard, if one was found + configured.
    kbd: Option<Kbd>,
    // The enumerated pointer (USB tablet / mouse), if one was found.
    mouse: Option<Ptr>,
    // Ports that already produced a working device. On an enumeration retry
    // (VirtualBox's async VUSB reset makes the *other* device time out) these
    // are skipped so we don't reset a port whose device already enumerated —
    // re-resetting port 1 after the keyboard was ready is what killed it.
    done_ports: u32,
    // Small ring of decoded keyboard bytes (the shared event ring is drained by
    // `pump_events`, which routes reports; `poll_key` pops from here).
    key_buf: [u8; 16],
    key_head: usize,
    key_tail: usize,
}

/// A configured HID boot keyboard.
struct Kbd {
    slot: u8,
    int_dci: u8,       // doorbell target for the interrupt IN endpoint
    int_ring_va: usize,
    int_ring_pa: u64,
    int_enqueue: usize,
    int_cycle: u32,
    report_pa: u64,    // 8-byte HID boot report buffer
    report_va: usize,
    prev: [u8; 8],     // last report, to detect newly-pressed keys
    // USB HID boot keyboards report only press/release edges, so a held key
    // would never repeat; synthesize accelerating typematic in software.
    rep: crate::keyrepeat::Typematic,
    rep_usage: u8, // the held usage the typematic is armed for (0 = none)
}

/// A configured HID pointer (USB tablet or mouse). The absolute-tablet report
/// (what QEMU `usb-tablet` and VirtualBox's USB pointing device emit) is
/// `[buttons, X:u16le, Y:u16le, wheel]` with X/Y in `0..=0x7fff`; a relative
/// boot mouse is `[buttons, dx:i8, dy:i8]`. `absolute` picks how `poll_mouse`
/// interprets the report.
struct Ptr {
    slot: u8,
    int_dci: u8,
    int_ring_va: usize,
    int_ring_pa: u64,
    int_enqueue: usize,
    int_cycle: u32,
    report_pa: u64,
    report_va: usize,
    report_len: u32,
    layout: PtrLayout,
}

/// Where the fields sit in a pointer's input report, parsed from its HID report
/// descriptor (so it works for any tablet/mouse layout, not a hardcoded guess).
/// Byte offsets are absolute in the received report (they already include the
/// leading report-ID byte when one is present).
#[derive(Clone, Copy)]
struct PtrLayout {
    btn_byte: u8,
    x_byte: u8,
    y_byte: u8,
    field_bytes: u8, // 1 or 2
    relative: bool,
    scale_max: i32,         // logical max for absolute scaling
    wheel_byte: Option<u8>, // byte offset of the scroll-wheel field (Usage 0x38), if present
}

impl PtrLayout {
    /// The standard relative boot-mouse report: `[buttons, dx:i8, dy:i8]`
    /// (plus a wheel byte at offset 3 on wheel mice).
    const BOOT_MOUSE: PtrLayout =
        PtrLayout { btn_byte: 0, x_byte: 1, y_byte: 2, field_bytes: 1, relative: true, scale_max: 0, wheel_byte: Some(3) };
}

/// Where a device sits in the USB topology — a root-port device or one behind a
/// hub. The slot context needs the route string, the root-hub port, and (for a
/// LS/FS device behind a HS hub) the Transaction Translator parent, so the xHC
/// can address and route to it. `ROOT` is the default (a direct root-port
/// device, route 0, no TT).
#[derive(Clone, Copy)]
struct DevLoc {
    root_port: u8,   // root-hub port the device (or its top-tier hub) hangs off
    route: u32,      // xHCI route string (0 for a root-port device)
    speed: u32,      // xHCI PSIV (1=FS,2=LS,3=HS,4=SS) of THIS device
    parent_slot: u8, // TT hub's slot id (0 → no TT / on a root port)
    parent_port: u8, // TT hub's downstream port number (1-based)
    tt: bool,        // LS/FS device behind a HS hub → needs the parent's TT
}

/// Per-device handles produced by `enumerate_common` (an addressed device with
/// its config descriptor read), consumed by `finish_keyboard`/`finish_pointer`.
struct Common {
    slot: u8,
    /// Output Device Context (DCBAA[slot]) — source of truth for Slot Context
    /// when building Configure Endpoint (must not re-zero port/speed).
    dev_ctx_va: usize,
    ep0_va: usize,
    ep0_pa: u64,
    ep0_enq: usize,
    ep0_cycle: u32,
    in_ctx_va: usize,
    in_ctx_pa: u64,
    buf_va: usize,
    buf_pa: u64,
    total: u32,
    cfg_val: u8,
    /// bDeviceClass from the device descriptor (9 = USB hub → recurse).
    dev_class: u8,
    /// Topology location (root port / route / TT), for downstream re-enumeration.
    loc: DevLoc,
    /// SET_CONFIGURATION already issued on this slot (so a composite device's
    /// second interface doesn't re-configure and reset the first endpoint).
    configured: bool,
}

const RING_TRBS: usize = 64; // TRBs per ring (last is a Link on the command ring)

// TRB types.
const TRB_NORMAL: u32 = 1;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_CONFIGURE_ENDPOINT: u32 = 12;
const TRB_EVALUATE_CONTEXT: u32 = 13;
const TRB_RESET_ENDPOINT: u32 = 14;
const TRB_SET_TR_DEQUEUE: u32 = 16;
// Event TRB types.
const EVT_TRANSFER: u32 = 32;
const EVT_CMD_COMPLETION: u32 = 33;
const CC_SUCCESS: u32 = 1;
const CC_STALL: u32 = 6;

// PORTSC bits.
const PORTSC_CCS: u32 = 1 << 0; // current connect status
const PORTSC_PED: u32 = 1 << 1; // port enabled
const PORTSC_PR: u32 = 1 << 4; // port reset
// RW1C/RW1CS bits to preserve (write 0) when doing a read-modify-write.
const PORTSC_RW1CS: u32 = PORTSC_PED | (0x7f << 17);

/// Full barrier between CPU writes to (cacheable) DMA memory and the following
/// device MMIO write. On aarch64 a `dsb sy` orders the Normal-memory TRB writes
/// ahead of the Device-memory doorbell (a `dmb ish` would not order across the
/// Normal/Device boundary); on x86 a `SeqCst` fence suffices.
#[inline]
fn dma_barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `dsb sy` is a barrier with no memory-safety implications.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Invalidate the CPU cache over `[va, va+len)` so a re-read of a buffer the
/// controller DMA-wrote sees fresh data on a **non-coherent** DMA host (real
/// Apple DWC3). Coherent hosts (QEMU, x86) already snoop, so this is just a
/// forced reload there — harmless. Uses `dc civac` (clean+invalidate to PoC);
/// we never dirty controller-written buffers, so nothing is lost. Call it
/// *before* reading a same-address buffer the device rewrites in place (the HID
/// report buffer, the event ring TRBs).
#[inline]
fn dma_invalidate(va: usize, len: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: cache maintenance over a mapped Normal-memory DMA buffer.
    unsafe {
        const LINE: usize = 64; // Apple/ARM64 cache line
        let mut p = va & !(LINE - 1);
        let end = va + len;
        while p < end {
            core::arch::asm!("dc civac, {}", in(reg) p, options(nostack, preserves_flags));
            p += LINE;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = (va, len);
}

/// Clean (write back) the CPU cache over `[va, va+len)` to the Point of
/// Coherency so a buffer WE wrote (a TRB the controller will DMA-read) is
/// actually in RAM before we ring the doorbell — required on non-coherent DMA
/// (real Apple DWC3). Coherent hosts (QEMU, x86) snoop, so this is a harmless
/// no-op there. `dc cvac` + `dsb sy`.
#[inline]
fn dma_clean(va: usize, len: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: cache maintenance over a mapped Normal-memory DMA buffer.
    unsafe {
        const LINE: usize = 64;
        let mut p = va & !(LINE - 1);
        let end = va + len;
        while p < end {
            core::arch::asm!("dc cvac, {}", in(reg) p, options(nostack, preserves_flags));
            p += LINE;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = (va, len);
}

/// Crude busy-wait (~`iters` volatile reads); used for the USB reset-recovery
/// settle where we have no timer in the arch-neutral core. Order of a few ms.
fn spin_delay(iters: u64) {
    let mut sink = 0u64;
    for i in 0..iters {
        sink = sink.wrapping_add(unsafe { read_volatile(&i) });
    }
    let _ = sink;
}

// MMIO accessors. On aarch64 these are single-instruction register-offset
// loads/stores via inline asm — the same guarantee Linux's readl()/writel()
// make. `read_volatile` lets LLVM pick the addressing mode, and in a loop it
// can emit a post-indexed `ldr` (writeback); an MMIO fault from a writeback
// access has no instruction syndrome (ESR.ISV=0), which a hypervisor cannot
// emulate — QEMU/HVF aborts with "Assertion failed: (isv)". Plain `[reg]`
// addressing always faults with ISV=1.
unsafe fn r32(addr: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v: u32;
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) addr, options(nostack, preserves_flags));
        v
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        read_volatile(addr as *const u32)
    }
}
unsafe fn w32(addr: usize, v: u32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) v, a = in(reg) addr, options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        write_volatile(addr as *mut u32, v)
    };
}
unsafe fn w64(addr: usize, v: u64) {
    // xHCI 64-bit registers: write low then high dword.
    unsafe {
        w32(addr, v as u32);
        w32(addr + 4, (v >> 32) as u32);
    }
}

impl Xhci {
    /// Bring the controller up from a wrapper-supplied `mmio` register base and
    /// DMA allocator: reset, set up the DCBAA + command ring + event ring, and
    /// start it. Discovery + MMIO mapping are the wrapper's job (arch-specific).
    pub fn bringup(mmio: usize, alloc: Alloc) -> Option<Xhci> {
        // SAFETY: `mmio` is the mapped xHCI register block.
        unsafe {
            let caplen = (r32(mmio + CAP_CAPLENGTH) & 0xff) as usize;
            let op = mmio + caplen;
            let rt = mmio + (r32(mmio + CAP_RTSOFF) & !0x1f) as usize;
            let db = mmio + (r32(mmio + CAP_DBOFF) & !0x3) as usize;
            let hcs1 = r32(mmio + CAP_HCSPARAMS1);
            let max_slots = (hcs1 & 0xff) as u8;
            let max_ports = ((hcs1 >> 24) & 0xff) as u8;
            // Context size: 64 bytes if HCCPARAMS1.CSZ (bit 2), else 32.
            let ctx_size = if r32(mmio + 0x10) & (1 << 2) != 0 { 64 } else { 32 };

            // Wait for CNR to clear, then reset. All three waits are BOUNDED: on
            // QEMU/VBox/PCIe they clear in microseconds, but a controller that
            // never becomes ready (e.g. the Apple DWC3 xHCI when the ATC-PHY isn't
            // fully host-ready) must fail cleanly, not hang the boot.
            let mut s = 0u32;
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {
                s += 1;
                if s > 2_000_000 {
                    crate::ktrace::log("xhci", "CNR never cleared before reset — controller not ready");
                    return None;
                }
            }
            w32(op + OP_USBCMD, USBCMD_HCRST);
            let mut s = 0u32;
            while r32(op + OP_USBCMD) & USBCMD_HCRST != 0 {
                s += 1;
                if s > 2_000_000 {
                    crate::ktrace::log("xhci", "HCRST never cleared — reset stuck");
                    return None;
                }
            }
            let mut s = 0u32;
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {
                s += 1;
                if s > 2_000_000 {
                    crate::ktrace::log("xhci", "CNR never cleared after reset — controller not ready");
                    return None;
                }
            }

            // Program enabled device slots.
            w32(op + OP_CONFIG, max_slots as u32);

            // Scratchpad buffers, if the controller wants them (HCSPARAMS2).
            let hcs2 = r32(mmio + CAP_HCSPARAMS2);
            let nscratch = (((hcs2 >> 21) & 0x1f) | ((hcs2 >> 27) & 0x1f) << 5) as usize;

            // Device Context Base Address Array: (max_slots + 1) * 8 bytes.
            let (dcbaa_pa, dcbaa_va) = alloc(4096)?;
            let dcbaa_va = dcbaa_va as usize;
            if nscratch > 0 {
                // Scratchpad buffer array + buffers (one page each).
                let (sp_arr_pa, sp_arr_va) = alloc(nscratch * 8)?;
                for i in 0..nscratch {
                    let (buf_pa, _) = alloc(4096)?;
                    write_volatile((sp_arr_va as *mut u64).add(i), buf_pa);
                }
                write_volatile(dcbaa_va as *mut u64, sp_arr_pa); // DCBAA[0] -> scratchpad array
            }
            w64(op + OP_DCBAAP, dcbaa_pa);

            // Command ring.
            let (cmd_ring_pa, cmd_ring_va) = alloc(RING_TRBS * 16)?;
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
            let (evt_ring_pa, evt_ring_va) = alloc(RING_TRBS * 16)?;
            let evt_ring_va = evt_ring_va as usize;
            let (erst_pa, erst_va) = alloc(64)?;
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
                ctx_size,
                alloc,
                kbd: None,
                mouse: None,
                done_ports: 0,
                key_buf: [0; 16],
                key_head: 0,
                key_tail: 0,
            })
        }
    }

    /// PORTSC of 1-based `port`.
    fn portsc(&self, port: u8) -> usize {
        self.op + OP_PORTS + (port as usize - 1) * 0x10
    }

    /// Ring a doorbell (slot 0 = command ring; slot N, target = endpoint DCI).
    /// A full DMA barrier first ensures the TRBs we just wrote to (cacheable)
    /// memory are globally visible before the controller sees the doorbell —
    /// required on real hardware / VirtualBox (QEMU's coherent emulation hid it).
    unsafe fn doorbell(&self, slot: u8, target: u32) {
        dma_barrier();
        unsafe { w32(self.db + slot as usize * 4, target) };
    }

    /// Push a TRB onto a command/transfer ring and advance, wrapping at the
    /// Link TRB in the last slot. Associated (no `self`) so it can borrow a
    /// ring's fields directly.
    unsafe fn ring_push(va: usize, pa: u64, enq: &mut usize, cycle: &mut u32, param: u64, status: u32, control: u32) {
        let control = (control & !1) | *cycle;
        let pos = *enq;
        unsafe { write_volatile((va + pos * 16) as *mut Trb, Trb { param, status, control }) };
        // Clean the written TRB to RAM so the controller's DMA read sees it on a
        // non-coherent host (real Apple); no-op on coherent QEMU/x86.
        dma_clean(va + pos * 16, 16);
        *enq += 1;
        if *enq == RING_TRBS - 1 {
            let link = (TRB_LINK << 10) | 0x2 | *cycle;
            unsafe { write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: link }) };
            dma_clean(va + (RING_TRBS - 1) * 16, 16);
            *enq = 0;
            *cycle ^= 1;
        }
    }

    /// Consume the next event TRB if one is ready (cycle bit matches), updating
    /// the dequeue pointer + ERDP. Non-blocking.
    unsafe fn next_event(&mut self) -> Option<Trb> {
        // Non-coherent DMA (real Apple): invalidate this TRB slot so we observe
        // the controller's write rather than a stale cached line (the ring wraps,
        // so a given slot is re-read across cycles).
        dma_invalidate(self.evt_ring_va + self.evt_dequeue * 16, 16);
        let trb = unsafe { read_volatile((self.evt_ring_va + self.evt_dequeue * 16) as *const Trb) };
        if (trb.control & 1) != self.evt_cycle {
            return None;
        }
        self.evt_dequeue += 1;
        if self.evt_dequeue == RING_TRBS {
            self.evt_dequeue = 0;
            self.evt_cycle ^= 1;
        }
        let erdp = self.evt_ring_pa + (self.evt_dequeue as u64) * 16;
        unsafe { w64(self.rt + 0x20 + 0x18, erdp | (1 << 3)) }; // ERDP, EHB=1 to ack
        Some(trb)
    }

    /// Block (bounded) for the next event of `want_type`; returns its TRB.
    /// Non-matching events are discarded (port-status etc.) — only one waiter.
    /// `max_spins` caps how long we poll; VirtualBox can stall a control TD
    /// forever if EP0 is Halted, so optional requests must use a tight bound.
    unsafe fn wait_event_bounded(&mut self, want_type: u32, max_spins: u64) -> Option<Trb> {
        let mut spins = 0u64;
        loop {
            if let Some(ev) = unsafe { self.next_event() } {
                if (ev.control >> 10) & 0x3f == want_type {
                    return Some(ev);
                }
            } else {
                spins += 1;
                if spins > max_spins {
                    unsafe { self.dump_event_ring(want_type) };
                    return None;
                }
            }
        }
    }

    unsafe fn wait_event(&mut self, want_type: u32) -> Option<Trb> {
        // ~seconds under a hypervisor; enough for a healthy control TD.
        unsafe { self.wait_event_bounded(want_type, 5_000_000) }
    }

    /// After a STALL or timed-out control TD, EP0 is Halted and its transfer
    /// ring may have an unfinished TD. VirtualBox then never completes further
    /// EP0 traffic (and host KeyboardQueue fills → VERR_PDM_NO_QUEUE_ITEMS).
    /// Reset Endpoint + Set TR Dequeue Pointer rewinds EP0 so enum can continue.
    unsafe fn recover_ep0(&mut self, slot: u8, ep0_va: usize, ep0_pa: u64, enq: &mut usize, cycle: &mut u32) {
        // Reset Endpoint, DCI 1 (EP0). TSP=0 → drop halted transfer state.
        let _ = unsafe {
            self.command(0, (TRB_RESET_ENDPOINT << 10) | (1u32 << 16) | ((slot as u32) << 24))
        };
        // Software ring rewind + Link TRB at the end.
        *enq = 0;
        *cycle = 1;
        unsafe {
            core::ptr::write_bytes(ep0_va as *mut u8, 0, RING_TRBS * 16);
            self.init_link(ep0_va, ep0_pa);
        }
        // Point the HC at the fresh ring (DCS=1 matches cycle 1).
        let _ = unsafe {
            self.command(
                ep0_pa | 1,
                (TRB_SET_TR_DEQUEUE << 10) | (1u32 << 16) | ((slot as u32) << 24),
            )
        };
        crate::ktrace::log_fmt(format_args!("xhci: recovered EP0 on slot {slot} after stall/timeout"));
    }

    /// Diagnostic dumped when `wait_event` times out: the current dequeue
    /// position + cycle, and every event-ring TRB's (control, status) so we can
    /// tell "controller posted nothing" from "event present, cycle mismatch".
    unsafe fn dump_event_ring(&self, want_type: u32) {
        crate::ktrace::log_fmt(format_args!(
            "xhci: wait_event({want_type}) TIMEOUT deq={} cyc={} erdp-based",
            self.evt_dequeue, self.evt_cycle
        ));
        for i in 0..RING_TRBS {
            let t = unsafe { read_volatile((self.evt_ring_va + i * 16) as *const Trb) };
            // Only log non-empty slots (control != 0) to keep it short.
            if t.control != 0 || t.status != 0 {
                let ty = (t.control >> 10) & 0x3f;
                let cc = (t.status >> 24) & 0xff;
                crate::ktrace::log_fmt(format_args!("  evt[{i}] type={ty} cc={cc} c={} ctrl={:#x}", t.control & 1, t.control));
            }
        }
    }

    /// Run a command TRB; return (completion_code, slot_id).
    unsafe fn command(&mut self, param: u64, control: u32) -> Option<(u32, u8)> {
        unsafe {
            Self::ring_push(self.cmd_ring_va, self.cmd_ring_pa, &mut self.cmd_enqueue, &mut self.cmd_cycle, param, 0, control);
            self.doorbell(0, 0);
            let ev = self.wait_event(EVT_CMD_COMPLETION)?;
            Some(((ev.status >> 24) & 0xff, ((ev.control >> 24) & 0xff) as u8))
        }
    }

    /// A control transfer on EP0 of `slot` over ring `(va,pa,enq,cycle)`.
    /// `setup` is the packed 8-byte setup packet; `buf_pa`/`len` an optional
    /// data stage (`data_in` = device→host). Returns true on success.
    #[allow(clippy::too_many_arguments)]
    unsafe fn control(
        &mut self,
        slot: u8,
        va: usize,
        pa: u64,
        enq: &mut usize,
        cycle: &mut u32,
        setup: u64,
        buf_pa: u64,
        len: u32,
        data_in: bool,
    ) -> bool {
        unsafe {
            let trt = if len == 0 { 0 } else if data_in { 3 } else { 2 };
            // Setup stage (immediate data).
            Self::ring_push(va, pa, enq, cycle, setup, 8, (TRB_SETUP << 10) | (trt << 16) | (1 << 6));
            // Data stage.
            if len > 0 {
                let dir = if data_in { 1 << 16 } else { 0 };
                Self::ring_push(va, pa, enq, cycle, buf_pa, len, (TRB_DATA << 10) | dir);
            }
            // Status stage (opposite direction; IOC to get an event).
            let sdir = if data_in && len > 0 { 0 } else { 1 << 16 };
            Self::ring_push(va, pa, enq, cycle, 0, 0, (TRB_STATUS << 10) | sdir | (1 << 5));
            self.doorbell(slot, 1); // EP0 = DCI 1
            match self.wait_event(EVT_TRANSFER) {
                Some(ev) => {
                    let cc = (ev.status >> 24) & 0xff;
                    let ok = cc == CC_SUCCESS || cc == 13 /* SHORT_PACKET */;
                    if !ok {
                        crate::ktrace::log_fmt(format_args!("xhci: control transfer cc={cc} (len {len}, in={data_in})"));
                        // STALL (6) leaves EP0 Halted — must Reset Endpoint before
                        // the next setup packet or every later control TD times out.
                        if cc == CC_STALL {
                            unsafe { self.recover_ep0(slot, va, pa, enq, cycle) };
                        }
                    }
                    ok
                }
                None => {
                    crate::ktrace::log_fmt(format_args!("xhci: control transfer TIMEOUT (len {len}, in={data_in})"));
                    unsafe { self.recover_ep0(slot, va, pa, enq, cycle) };
                    false
                }
            }
        }
    }

    /// C2/C3: find a connected keyboard, reset its port, address the device,
    /// read its descriptors, set the config + boot protocol, and arm the
    /// interrupt IN endpoint.
    pub fn enumerate_keyboard(&mut self) -> bool {
        self.enumerate_input()
    }

    /// Enumerate the connected USB HID devices — a boot keyboard and/or a
    /// pointer (USB tablet / mouse), which may sit on separate ports.
    pub fn enumerate_input(&mut self) -> bool {
        // After our HCRST, VirtualBox re-attaches HidKeyboard / HidMouse
        // asynchronously and often races a guest port-reset with its own
        // device-level reset ("reset request is ignored, already resetting").
        // Wait before the first scan so the first pass has a chance; QEMU is
        // instantaneous so the extra settle is just a few ms of busy-wait.
        spin_delay(80_000_000);
        // Fast path out: nothing connected on any port -> no retries, no delays.
        let any_port = (1..=self.max_ports).any(|p| unsafe { r32(self.portsc(p)) } & PORTSC_CCS != 0);
        if !any_port {
            // Dump each port's raw PORTSC so a bring-up boot distinguishes an
            // unpowered/dead PHY (0 / all-ones) from a powered port awaiting a
            // device (PP set, CCS clear).
            for p in 1..=self.max_ports {
                let psc = unsafe { r32(self.portsc(p)) };
                crate::ktrace::log_fmt(format_args!("xhci: port {p} PORTSC={psc:#010x}"));
            }
            crate::ktrace::log("xhci", "no USB device connected");
            return false;
        }
        // Enumerate with RETRIES. VirtualBox resets the attached VUSB device
        // *asynchronously* after our controller reset (~20 ms); a control
        // transfer submitted in that window is silently dropped. A settle +
        // fresh attempt succeeds once the device-level reset has finished. QEMU
        // is instantaneous and never needs the retry.
        // Up to 8 passes with a settle between them. `done_ports` keeps a device
        // that already came up from being re-reset, so the extra passes only
        // re-try whatever is still missing and never disturb a working device.
        // QEMU gets both on the first pass and exits at once.
        for attempt in 0..8 {
            if attempt > 0 {
                spin_delay(300_000_000);
                crate::ktrace::log_fmt(format_args!("xhci: retrying enumeration (attempt {})", attempt + 1));
            }
            // SAFETY: single-threaded boot; all DMA regions are freshly allocated.
            unsafe { self.scan_ports() };
            if self.kbd.is_some() && self.mouse.is_some() {
                break;
            }
        }
        if self.kbd.is_none() {
            crate::ktrace::log("xhci", "no HID keyboard enumerated");
        }
        if self.mouse.is_none() {
            crate::ktrace::log("xhci", "no HID pointer enumerated");
        }
        self.kbd.is_some() || self.mouse.is_some()
    }

    /// Whether a HID keyboard / pointer was enumerated (boot diagnostic).
    pub fn has_keyboard(&self) -> bool {
        self.kbd.is_some()
    }
    pub fn has_mouse(&self) -> bool {
        self.mouse.is_some()
    }
    /// Root-hub port count (HCSPARAMS1.MaxPorts) — a bring-up diagnostic.
    pub fn port_count(&self) -> u8 {
        self.max_ports
    }

    /// Enumerate every connected root port once, classifying each device as a
    /// keyboard, a pointer, or a **hub** (recurse into it — the Mac mini's USB-A
    /// ports hang off an internal hub). A device on the first port no longer
    /// starves the others (each port is tried, not just the first).
    unsafe fn scan_ports(&mut self) {
        for port in 1..=self.max_ports {
            if self.kbd.is_some() && self.mouse.is_some() {
                break;
            }
            // Never re-touch a port that already gave us a device: resetting it
            // on a retry (to enumerate the *other*, still-missing device) would
            // knock the working one offline. This is the keyboard-dies-when-the-
            // mouse-retries bug.
            if port < 32 && self.done_ports & (1 << port) != 0 {
                continue;
            }
            let psc = unsafe { r32(self.portsc(port)) };
            if psc & PORTSC_CCS == 0 {
                continue;
            }
            crate::ktrace::log_fmt(format_args!("xhci: root port {port} device present (portsc={psc:#x})"));
            let Some(mut c) = (unsafe { self.enumerate_common(port) }) else { continue };
            crate::ktrace::log_fmt(format_args!("xhci: root port {port} slot {} dev_class={}", c.slot, c.dev_class));
            // A USB hub (bDeviceClass 9): recurse — the keyboard on a USB-A port
            // lives behind the Mac mini's internal hub, not on a root port.
            if c.dev_class == 9 {
                unsafe { self.enumerate_hub(&mut c) };
                if port < 32 { self.done_ports |= 1 << port; }
                continue;
            }
            let got = unsafe { self.classify_and_finish(&mut c) };
            if got && port < 32 {
                self.done_ports |= 1 << port;
            }
        }
    }

    /// Classify an addressed device by its config descriptor and finish whatever
    /// HID role(s) we still need — BOTH a keyboard and a pointer from the same
    /// device when it is composite (a wireless dongle exposing both), sharing the
    /// one slot. Returns whether anything was registered.
    unsafe fn classify_and_finish(&mut self, c: &mut Common) -> bool {
        let mut got = false;
        if self.kbd.is_none() {
            if let Some((iface, ep, mps, ivl)) = unsafe { parse_hid_keyboard(c.buf_va, c.total as usize) } {
                if let Some(k) = unsafe { self.finish_keyboard(c, iface, ep, mps, ivl) } {
                    self.kbd = Some(k);
                    got = true;
                }
            }
        }
        if self.mouse.is_none() {
            if let Some((iface, ep, mps, ivl, proto)) = unsafe { parse_hid_pointer(c.buf_va, c.total as usize) } {
                if let Some(p) = unsafe { self.finish_pointer(c, iface, ep, mps, ivl, proto) } {
                    self.mouse = Some(p);
                    got = true;
                }
            }
        }
        got
    }

    /// Bring up a USB **hub** `c` (already addressed) and enumerate the devices
    /// behind it — this is how a keyboard on the Mac mini's USB-A ports (which
    /// hang off an internal hub) is reached. Configure the hub, mark its slot as
    /// a hub in the xHC, power + reset each downstream port, and for every
    /// connected one address the device with the right route string + TT and
    /// classify it. One tier deep (the Mac mini's topology); nested hubs are
    /// logged and skipped.
    unsafe fn enumerate_hub(&mut self, c: &mut Common) {
        // Standard-request byte for hub class requests (recipient = other/port
        // uses bmRequestType 0x23 set / 0xA3 get).
        const REQ_GET_STATUS: u8 = 0;
        const REQ_CLEAR_FEATURE: u8 = 1;
        const REQ_SET_FEATURE: u8 = 3;
        const FEAT_PORT_RESET: u16 = 4;
        const FEAT_PORT_POWER: u16 = 8;
        const FEAT_C_PORT_RESET: u16 = 20;
        // Port status bits (USB 2.0 hub wPortStatus).
        const PS_CONNECTION: u16 = 1 << 0;
        const PS_ENABLE: u16 = 1 << 1;
        const PS_RESET: u16 = 1 << 4;
        const PS_LOW_SPEED: u16 = 1 << 9;
        const PS_HIGH_SPEED: u16 = 1 << 10;

        // 1. Configure the hub (once), so it answers class requests.
        if !c.configured {
            if !unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x00, 9, c.cfg_val as u16, 0, 0), 0, 0, false)
            } {
                crate::ktrace::log("xhci", "hub SET_CONFIGURATION failed");
                return;
            }
            c.configured = true;
        }
        // 2. Read the hub descriptor (class GET_DESCRIPTOR, type 0x29).
        if !unsafe {
            self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0xA0, 6, 0x2900, 0, 15), c.buf_pa, 15, true)
        } {
            crate::ktrace::log("xhci", "hub GET_DESCRIPTOR failed");
            return;
        }
        let nbr_ports = unsafe { read_volatile((c.buf_va + 2) as *const u8) };
        let characteristics = unsafe { read_u16(c.buf_va + 3) };
        let pwr_on_2_good = unsafe { read_volatile((c.buf_va + 5) as *const u8) } as u64; // 2ms units
        let ttt = ((characteristics >> 5) & 0x3) as u32; // TT Think Time
        crate::ktrace::log_fmt(format_args!(
            "xhci: hub slot {} ports={nbr_ports} char={characteristics:#06x} ttt={ttt}",
            c.slot
        ));
        // 3. Tell the xHC this slot is a hub (Hub=1, #ports, TT think time) so it
        //    will schedule + route downstream transfers.
        unsafe { self.mark_hub(c, nbr_ports, ttt) };
        // 4. Power all ports, wait the descriptor's power-on-to-good, then reset
        //    + enumerate each connected one.
        for p in 1..=nbr_ports {
            let _ = unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x23, REQ_SET_FEATURE, FEAT_PORT_POWER, p as u16, 0), 0, 0, false)
            };
        }
        spin_delay(2_000_000 + pwr_on_2_good * 400_000);
        for p in 1..=nbr_ports {
            if self.kbd.is_some() && self.mouse.is_some() {
                break;
            }
            if !unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0xA3, REQ_GET_STATUS, 0, p as u16, 4), c.buf_pa, 4, true)
            } {
                continue;
            }
            let status = unsafe { read_u16(c.buf_va) };
            crate::ktrace::log_fmt(format_args!("xhci: hub port {p} status={status:#06x}"));
            if status & PS_CONNECTION == 0 {
                continue; // nothing plugged into this downstream port
            }
            // Reset the downstream port, poll until enabled + reset-done.
            let _ = unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x23, REQ_SET_FEATURE, FEAT_PORT_RESET, p as u16, 0), 0, 0, false)
            };
            let mut reset_ok = false;
            for _ in 0..50 {
                spin_delay(1_000_000);
                if !unsafe {
                    self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0xA3, REQ_GET_STATUS, 0, p as u16, 4), c.buf_pa, 4, true)
                } {
                    continue;
                }
                let st = unsafe { read_u16(c.buf_va) };
                if st & PS_ENABLE != 0 && st & PS_RESET == 0 {
                    reset_ok = true;
                    break;
                }
            }
            if !reset_ok {
                crate::ktrace::log_fmt(format_args!("xhci: hub port {p} reset never completed"));
                continue;
            }
            // Clear the reset-change latch, then let the device settle.
            let _ = unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x23, REQ_CLEAR_FEATURE, FEAT_C_PORT_RESET, p as u16, 0), 0, 0, false)
            };
            spin_delay(4_000_000); // USB reset-recovery
            let status = unsafe { read_u16(c.buf_va) };
            let speed = if status & PS_LOW_SPEED != 0 {
                2 // low speed (PSIV 2)
            } else if status & PS_HIGH_SPEED != 0 {
                3 // high speed (PSIV 3)
            } else {
                1 // full speed (PSIV 1)
            };
            let hub_hs = c.loc.speed >= 3;
            let loc = DevLoc {
                root_port: c.loc.root_port,
                route: push_route(c.loc.route, p),
                speed,
                parent_slot: c.slot,
                parent_port: p,
                // A LS/FS device behind a HS hub uses the hub's transaction translator.
                tt: (speed == 1 || speed == 2) && hub_hs,
            };
            crate::ktrace::log_fmt(format_args!(
                "xhci: hub port {p} device speed={speed} route={:#x} tt={} — addressing",
                loc.route, loc.tt
            ));
            let Some(mut dc) = (unsafe { self.address_and_config(loc) }) else {
                crate::ktrace::log_fmt(format_args!("xhci: hub port {p} device enumeration failed"));
                continue;
            };
            crate::ktrace::log_fmt(format_args!("xhci: hub port {p} slot {} dev_class={}", dc.slot, dc.dev_class));
            if dc.dev_class == 9 {
                crate::ktrace::log_fmt(format_args!("xhci: nested hub on hub port {p} not descended"));
                continue;
            }
            unsafe { self.classify_and_finish(&mut dc) };
        }
    }

    /// Update a hub's Slot Context in the xHC (Configure Endpoint, Add Slot
    /// only): set the Hub flag, the port count, and the TT Think Time so the
    /// controller will route to + schedule for the devices behind it.
    unsafe fn mark_hub(&mut self, c: &Common, nbr_ports: u8, ttt: u32) {
        unsafe {
            let in_ctx_va = c.in_ctx_va;
            w32(in_ctx_va, 0); // Drop none
            w32(in_ctx_va + 4, 0b1); // Add A0 (slot) only
            let in_slot = in_ctx_va + self.ctx_size;
            core::ptr::copy_nonoverlapping(c.dev_ctx_va as *const u8, in_slot as *mut u8, self.ctx_size);
            w32(in_slot, r32(in_slot) | (1 << 26)); // dword0 Hub=1
            let d1 = (r32(in_slot + 4) & 0x00ff_ffff) | ((nbr_ports as u32) << 24); // #ports [31:24]
            w32(in_slot + 4, d1);
            let d2 = (r32(in_slot + 8) & !(0b11 << 16)) | ((ttt & 0x3) << 16); // TTT [17:16]
            w32(in_slot + 8, d2);
            match self.command(c.in_ctx_pa, (TRB_CONFIGURE_ENDPOINT << 10) | ((c.slot as u32) << 24)) {
                Some((cc, _)) if cc == CC_SUCCESS => {}
                other => crate::ktrace::log_fmt(format_args!("xhci: mark-hub configure endpoint failed ({other:?})")),
            }
        }
    }

    /// Enumerate a device on **root** port `port`: reset the port, read its
    /// speed, then address + configure it at the root of the topology (route 0,
    /// no TT). The location-independent work is in [`address_and_config`], which
    /// the hub path reuses for downstream devices.
    unsafe fn enumerate_common(&mut self, port: u8) -> Option<Common> {
        let psc = self.portsc(port);
        unsafe {
            // Reset the port (USB2/FullSpeed needs it; preserve RW1C status bits).
            let before = r32(psc);
            crate::ktrace::log_fmt(format_args!("xhci: resetting port {port} (portsc={before:#x})"));
            let v = before & !PORTSC_RW1CS;
            w32(psc, v | PORTSC_PR);
            // Wait for the port to come back enabled. Each read is an MMIO trap
            // (slow under a hypervisor), so the bound is in reads, not time:
            // ~2M reads is seconds of real time, far beyond a real port reset
            // (~10 ms on VirtualBox; instant on QEMU).
            let mut spins = 0u64;
            while r32(psc) & PORTSC_PED == 0 {
                spins += 1;
                if spins > 2_000_000 {
                    crate::ktrace::log_fmt(format_args!("xhci: port {port} did not re-enable after reset (portsc={:#x})", r32(psc)));
                    return None;
                }
            }
            // USB reset-recovery: a device needs time after reset before it will
            // answer control transfers (the spec allows up to 10 ms). QEMU is
            // instant, but slower/FullSpeed models (e.g. VirtualBox's HID
            // keyboard) can NAK Address Device / GET_DESCRIPTOR without this.
            spin_delay(2_000_000);
        }
        let speed = (unsafe { r32(psc) } >> 10) & 0xf;
        crate::ktrace::log_fmt(format_args!("xhci: port {port} connected, speed {speed}, resetting -> enabled"));
        let loc = DevLoc { root_port: port, route: 0, speed, parent_slot: 0, parent_port: 0, tt: false };
        unsafe { self.address_and_config(loc) }
    }

    /// Steps 2–8 shared by every device (root or behind a hub): enable a slot,
    /// address the device at its [`DevLoc`] (route string + TT), learn EP0
    /// max-packet, capture bDeviceClass, and read its full config descriptor.
    /// The caller then classifies it (keyboard / pointer / hub).
    unsafe fn address_and_config(&mut self, loc: DevLoc) -> Option<Common> {
        let speed = loc.speed;
        let ep0_mps: u32 = match speed {
            3 => 64,      // high
            4 | 5 => 512, // super/super+
            _ => 8,       // full/low
        };

        // 2. Enable a slot.
        let (cc, slot) = unsafe { self.command(0, TRB_ENABLE_SLOT << 10) }?;
        if cc != CC_SUCCESS {
            crate::ktrace::log_fmt(format_args!("xhci: enable slot failed cc={cc}"));
            return None;
        }

        // 3. Device context + DCBAA entry.
        let (dev_ctx_pa, dev_ctx_va) = (self.alloc)(4096)?;
        let dev_ctx_va = dev_ctx_va as usize;
        unsafe { write_volatile((self.dcbaa_va as *mut u64).add(slot as usize), dev_ctx_pa) };

        // 4. EP0 transfer ring.
        let (ep0_pa, ep0_va) = (self.alloc)(RING_TRBS * 16)?;
        let ep0_va = ep0_va as usize;
        unsafe { self.init_link(ep0_va, ep0_pa) };
        let mut ep0_enq = 0usize;
        let mut ep0_cycle = 1u32;

        // 5. Input context for Address Device (add slot + EP0).
        let (in_ctx_pa, in_ctx_va) = (self.alloc)(4096)?;
        let in_ctx_va = in_ctx_va as usize;
        unsafe {
            self.build_input_slot_ep0(in_ctx_va, loc, ep0_mps, ep0_pa);
            // 6. Address Device.
            let (cc, _) = self.command(in_ctx_pa, (TRB_ADDRESS_DEVICE << 10) | ((slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: address device failed cc={cc}"));
                return None;
            }
        }
        // Diagnostic: the output device context after Address Device — slot
        // state (dword3 bits 31:27) + EP0 state (EP-context dword0 bits 2:0) —
        // so a "transfer never runs" failure can be told from EP0 not being in
        // the Running state (1). QEMU: slot=2(addressed), ep0=1(running).
        unsafe {
            let slot_state = (read_volatile((dev_ctx_va + 12) as *const u32) >> 27) & 0x1f;
            let ep0_state = read_volatile((dev_ctx_va + self.ctx_size) as *const u32) & 0x7;
            crate::ktrace::log_fmt(format_args!("xhci: post-address slot_state={slot_state} ep0_state={ep0_state}"));
            let _ = dev_ctx_pa;
        }
        crate::ktrace::log_fmt(format_args!("xhci: device addressed on slot {slot}"));
        // SET_ADDRESS recovery: the USB spec gives a device up to 2 ms after
        // being addressed before it must answer EP0 requests. QEMU replies
        // instantly, but a real-timed device model (VirtualBox) NAKs a control
        // transfer issued immediately after Address Device, so wait first.
        spin_delay(4_000_000);

        // 7. Learn the real EP0 max packet size before any multi-byte read.
        // We addressed the device assuming MPS=8 (full/low) or 64 (high). A
        // full-speed device may actually use MPS 16/32/64; reading the 9-byte
        // config descriptor with an under-sized MPS makes the device pack all 9
        // bytes into one packet → the host sees a packet > its configured MPS →
        // BABBLE, and enumeration fails (exactly what VirtualBox's full-speed
        // keyboard hit). Reading only the first 8 bytes of the DEVICE descriptor
        // is always safe (8 ≤ any MPS), and byte 7 is bMaxPacketSize0.
        let (buf_pa, buf_va) = (self.alloc)(4096)?;
        let buf_va = buf_va as usize;
        let mut dev_class = 0u8;
        let got_dev = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0100, 0, 8), buf_pa, 8, true)
        };
        if !got_dev {
            // Diagnostic: did the controller consume the EP0 TRBs? Read the EP0
            // TR-dequeue pointer back from the output device context — if it
            // advanced past the ring base, the transfer executed but the event
            // was lost; if not, the controller never processed the doorbell.
            // Plus USBSTS (HSE = a DMA/host error reading the ring).
            unsafe {
                let deq = read_volatile((dev_ctx_va + self.ctx_size + 8) as *const u64);
                let usbsts = r32(self.op + OP_USBSTS);
                crate::ktrace::log_fmt(format_args!("xhci: EP0 stuck: tr_deq={:#x} ring_base={:#x} usbsts={:#x}", deq, ep0_pa, usbsts));
            }
        }
        if got_dev {
            // bDeviceClass@4: 9 = USB hub (the caller recurses into it).
            dev_class = unsafe { read_volatile((buf_va + 4) as *const u8) };
            let b = unsafe { read_volatile((buf_va + 7) as *const u8) } as u32;
            // full/low (speed 1/2): bMaxPacketSize0 is the byte count (8/16/32/64).
            // super-speed (4): it's an exponent (9 => 512). high (3) is always 64.
            let real_mps = match speed {
                1 | 2 => b,
                4 => 1u32 << b,
                _ => 64,
            };
            if real_mps >= 8 && real_mps != ep0_mps {
                crate::ktrace::log_fmt(format_args!("xhci: EP0 max packet {ep0_mps} -> {real_mps}; re-evaluating"));
                unsafe {
                    self.build_input_eval_mps(in_ctx_va, real_mps);
                    let _ = self.command(in_ctx_pa, (TRB_EVALUATE_CONTEXT << 10) | ((slot as u32) << 24));
                }
            }
        }

        // 8. Read the configuration descriptor: first 9 bytes for wTotalLength.
        let ok = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0200, 0, 9), buf_pa, 9, true)
        };
        if !ok {
            return None;
        }
        let total = unsafe { read_u16(buf_va + 2) } as u32;
        let cfg_val = unsafe { read_volatile((buf_va + 5) as *const u8) };
        let ok = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0200, 0, total as u16), buf_pa, total.min(4096), true)
        };
        if !ok {
            return None;
        }
        Some(Common {
            slot,
            dev_ctx_va,
            ep0_va,
            ep0_pa,
            ep0_enq,
            ep0_cycle,
            in_ctx_va,
            in_ctx_pa,
            buf_va,
            buf_pa,
            total,
            cfg_val,
            dev_class,
            loc,
            configured: false,
        })
    }

    /// Set the configuration, put the interface into a protocol, and configure
    /// the interrupt IN endpoint. Shared tail of keyboard/pointer setup; returns
    /// the endpoint's DCI + ring/report allocations.
    #[allow(clippy::too_many_arguments)]
    unsafe fn finish_endpoint(
        &mut self,
        c: &mut Common,
        iface: u8,
        ep_addr: u8,
        ep_mps: u32,
        interval: u8,
        set_boot: bool,
    ) -> Option<(u8, usize, u64, u64, usize)> {
        unsafe {
            // Set configuration ONCE per device (required before it answers
            // interrupt IN). A composite device (keyboard + mouse on one slot)
            // finishes two interfaces; re-issuing SET_CONFIGURATION for the
            // second would reset the first interface's endpoints, so guard it.
            if !c.configured {
                if !self.control(
                    c.slot,
                    c.ep0_va,
                    c.ep0_pa,
                    &mut c.ep0_enq,
                    &mut c.ep0_cycle,
                    setup(0x00, 9, c.cfg_val as u16, 0, 0),
                    0,
                    0,
                    false,
                ) {
                    crate::ktrace::log_fmt(format_args!(
                        "xhci: SET_CONFIGURATION({}) failed — retrying once after EP0 recover",
                        c.cfg_val
                    ));
                    // recover_ep0 already ran inside control(); try once more.
                    if !self.control(
                        c.slot,
                        c.ep0_va,
                        c.ep0_pa,
                        &mut c.ep0_enq,
                        &mut c.ep0_cycle,
                        setup(0x00, 9, c.cfg_val as u16, 0, 0),
                        0,
                        0,
                        false,
                    ) {
                        crate::ktrace::log("xhci", "SET_CONFIGURATION failed twice");
                        return None;
                    }
                }
                c.configured = true;
            }
            if set_boot {
                // SET_PROTOCOL(boot=0). VBox often stalls when already in boot
                // mode — control() recovers EP0 so later GET_DESCRIPTOR still works.
                if !self.control(
                    c.slot,
                    c.ep0_va,
                    c.ep0_pa,
                    &mut c.ep0_enq,
                    &mut c.ep0_cycle,
                    setup(0x21, 0x0b, 0, iface as u16, 0),
                    0,
                    0,
                    false,
                ) {
                    crate::ktrace::log("xhci", "SET_PROTOCOL(boot) stalled (recovered EP0; continuing)");
                }
            }
            // Do NOT issue SET_IDLE here: on VBox a stall mid-enum left EP0 Halted
            // and the follow-up control TD hung until wait_event timed out, while
            // the host KeyboardQueue filled (VERR_PDM_NO_QUEUE_ITEMS). Idle default
            // is fine for boot keyboards/tablets.
        }
        let (int_pa, int_va) = (self.alloc)(RING_TRBS * 16)?;
        let int_va = int_va as usize;
        unsafe { self.init_link(int_va, int_pa) };
        let (report_pa, report_va) = (self.alloc)(4096)?;
        let report_va = report_va as usize;
        let int_dci = (((ep_addr & 0x0f) as u32) * 2 + 1) as u8; // IN endpoint DCI
        unsafe {
            self.build_input_configure(c, int_dci, ep_mps, interval, int_pa);
            let (cc, _) = self.command(c.in_ctx_pa, (TRB_CONFIGURE_ENDPOINT << 10) | ((c.slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: configure endpoint failed cc={cc}"));
                return None;
            }
            // EP state in Output Context: 1 = Running. Anything else means the
            // interrupt IN will never complete (VBox used to show READY with
            // a wiped Slot Context → port/speed 0 after Evaluate Context).
            // OUTPUT device context: slot @ index 0, EP DCI d @ index d (no Input
            // Control Context prefix — that +1 shift applies only to the INPUT
            // context in build_input_configure). Reading (dci+1) here was off by
            // one (an unused entry → always 0).
            let ep_ctx = c.dev_ctx_va + self.ctx_size * int_dci as usize;
            // Output device context is controller-written — invalidate before read
            // on non-coherent DMA (real Apple).
            dma_invalidate(c.dev_ctx_va, self.ctx_size * (int_dci as usize + 1));
            let ep_state = read_volatile(ep_ctx as *const u32) & 0x7;
            if ep_state != 1 {
                crate::ktrace::log_fmt(format_args!("xhci: interrupt EP dci={int_dci} not Running (state {ep_state})"));
            }
        }
        Some((int_dci, int_va, int_pa, report_pa, report_va))
    }

    /// Finish a boot keyboard: set config + boot protocol, configure its
    /// interrupt endpoint, and arm the first transfer.
    unsafe fn finish_keyboard(&mut self, c: &mut Common, iface: u8, ep_addr: u8, ep_mps: u32, interval: u8) -> Option<Kbd> {
        let (int_dci, int_va, int_pa, report_pa, report_va) =
            unsafe { self.finish_endpoint(c, iface, ep_addr, ep_mps, interval, true) }?;
        let mut kbd = Kbd {
            slot: c.slot, int_dci, int_ring_va: int_va, int_ring_pa: int_pa,
            int_enqueue: 0, int_cycle: 1, report_pa, report_va,
            prev: [0; 8],
            rep: crate::keyrepeat::Typematic::new(), rep_usage: 0,
        };
        unsafe { self.queue_interrupt(&mut kbd) };
        crate::ktrace::log_fmt(format_args!("xhci: HID keyboard ready (slot {}, ep {ep_addr:#x}, dci {int_dci})", c.slot));
        Some(kbd)
    }

    /// Finish a pointer: a boot mouse (proto 2, relative) gets boot protocol; a
    /// tablet (proto 0) keeps report protocol (absolute). Configure the endpoint
    /// and arm the first transfer.
    unsafe fn finish_pointer(&mut self, c: &mut Common, iface: u8, ep_addr: u8, ep_mps: u32, interval: u8, proto: u8) -> Option<Ptr> {
        let boot_mouse = proto == 2;
        // Report-descriptor length from the HID (0x21) descriptor in the config.
        let rdlen = unsafe { parse_hid_report_len(c.buf_va, c.total as usize, iface) }.unwrap_or(0);
        let (int_dci, int_va, int_pa, report_pa, report_va) =
            unsafe { self.finish_endpoint(c, iface, ep_addr, ep_mps, interval, boot_mouse) }?;
        // Determine the report layout. A boot mouse (SET_PROTOCOL boot) uses the
        // fixed 3-byte report; a tablet keeps report protocol, so parse its HID
        // report descriptor to locate X/Y/buttons (VirtualBox and QEMU tablets
        // differ, e.g. a leading report-ID byte).
        let layout = if boot_mouse {
            PtrLayout::BOOT_MOUSE
        } else if rdlen > 0 {
            let got = unsafe {
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x81, 6, 0x2200, iface as u16, rdlen), c.buf_pa, (rdlen as u32).min(4096), true)
            };
            (got.then(|| unsafe { parse_report_layout(c.buf_va, (rdlen as usize).min(4096)) }).flatten())
                // Fallback: the common absolute tablet report [buttons, X:u16, Y:u16].
                .unwrap_or(PtrLayout { btn_byte: 0, x_byte: 1, y_byte: 3, field_bytes: 2, relative: false, scale_max: 0x7fff, wheel_byte: None })
        } else {
            PtrLayout { btn_byte: 0, x_byte: 1, y_byte: 3, field_bytes: 2, relative: false, scale_max: 0x7fff, wheel_byte: None }
        };
        let mut ptr = Ptr {
            slot: c.slot, int_dci, int_ring_va: int_va, int_ring_pa: int_pa,
            int_enqueue: 0, int_cycle: 1, report_pa, report_va,
            report_len: ep_mps.clamp(4, 16),
            layout,
        };
        unsafe {
            self.arm_int(ptr.int_ring_va, ptr.int_ring_pa, &mut ptr.int_enqueue, &mut ptr.int_cycle, ptr.report_pa, ptr.report_len, ptr.slot, ptr.int_dci);
        }
        crate::ktrace::log_fmt(format_args!(
            "xhci: HID pointer ready (slot {}, ep {ep_addr:#x}, dci {int_dci}, {}, x@{} y@{} {}B btn@{} max={})",
            c.slot,
            if layout.relative { "relative" } else { "absolute" },
            layout.x_byte, layout.y_byte, layout.field_bytes, layout.btn_byte, layout.scale_max
        ));
        Some(ptr)
    }

    /// Initialize a transfer ring's trailing Link TRB (back to its own start).
    unsafe fn init_link(&self, va: usize, pa: u64) {
        unsafe {
            write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: (TRB_LINK << 10) | 0x2 | 1 });
        }
    }

    /// Build the input context for Address Device: add flags A0|A1, a slot
    /// context (route string, root-hub port, speed, 1 context entry — plus the
    /// TT parent for a LS/FS device behind a HS hub) and the EP0 control
    /// endpoint context.
    unsafe fn build_input_slot_ep0(&self, in_ctx_va: usize, loc: DevLoc, ep0_mps: u32, ep0_pa: u64) {
        unsafe {
            // zero the control + slot + ep0 area
            core::ptr::write_bytes(in_ctx_va as *mut u8, 0, self.ctx_size * 3);
            w32(in_ctx_va + 4, 0b11); // Add flags: A0 (slot) | A1 (EP0)
            let slot_ctx = in_ctx_va + self.ctx_size;
            // dword0: Route String[19:0] | Speed[23:20] | Context Entries[31:27]=1.
            w32(slot_ctx, (1 << 27) | (loc.speed << 20) | (loc.route & 0xf_ffff));
            // dword1: Root Hub Port Number[23:16].
            w32(slot_ctx + 4, (loc.root_port as u32) << 16);
            // dword2: TT Hub Slot ID[7:0] | TT Port Number[15:8] for a LS/FS
            // device behind a HS hub (the hub does the transaction translation).
            if loc.tt {
                w32(slot_ctx + 8, (loc.parent_slot as u32) | ((loc.parent_port as u32) << 8));
            }
            let ep0 = in_ctx_va + self.ctx_size * 2;
            w32(ep0 + 4, (4 << 3) | (ep0_mps << 16) | (3 << 1)); // type=Control, MPS, CErr=3
            w64(ep0 + 8, ep0_pa | 1); // TR dequeue ptr | DCS
            w32(ep0 + 16, 8); // avg TRB length
        }
    }

    /// Build the input context for Evaluate Context to update EP0's max packet
    /// size (learned from the device descriptor). Add flag A1 (EP0) only.
    ///
    /// **Must not wipe the Slot Context** — a prior version zeroed control+slot+EP0
    /// here, so the later Configure Endpoint inherited port=0/speed=0 and VBox
    /// never scheduled interrupt INs (control still worked; kbd queue overflowed).
    unsafe fn build_input_eval_mps(&self, in_ctx_va: usize, ep0_mps: u32) {
        unsafe {
            w32(in_ctx_va, 0); // Drop flags
            w32(in_ctx_va + 4, 0b10); // Add flags: A1 (EP0) only — leave Slot alone
            let ep0 = in_ctx_va + self.ctx_size * 2;
            core::ptr::write_bytes(ep0 as *mut u8, 0, self.ctx_size);
            w32(ep0 + 4, (4 << 3) | (ep0_mps << 16) | (3 << 1)); // type=Control, MPS, CErr=3
        }
    }

    /// Input Context for Configure Endpoint: copy the live Output Slot Context
    /// (port, speed, route — still valid after Address Device), bump Context
    /// Entries, and install the interrupt IN endpoint at `dci`.
    unsafe fn build_input_configure(&self, c: &Common, dci: u8, mps: u32, interval: u8, int_pa: u64) {
        unsafe {
            let in_ctx_va = c.in_ctx_va;
            // Control: Drop none; Add Slot (A0) + interrupt EP (A_dci).
            w32(in_ctx_va, 0);
            w32(in_ctx_va + 4, 0b1 | (1u32 << dci));
            // Copy Output Slot Context → Input Slot Context, then set Context
            // Entries to max(current, dci): a composite device's second endpoint
            // may have a LOWER DCI than the first, and shrinking the count would
            // drop the already-installed endpoint from the slot.
            let in_slot = in_ctx_va + self.ctx_size;
            core::ptr::copy_nonoverlapping(c.dev_ctx_va as *const u8, in_slot as *mut u8, self.ctx_size);
            let cur_entries = (r32(in_slot) >> 27) & 0x1f;
            let entries = cur_entries.max(dci as u32);
            let d0 = r32(in_slot) & !(0x1f << 27);
            w32(in_slot, d0 | (entries << 27));
            // Interrupt IN endpoint context at index dci+1.
            let ep = in_ctx_va + self.ctx_size * (dci as usize + 1);
            core::ptr::write_bytes(ep as *mut u8, 0, self.ctx_size);
            // Interval: 2^(Interval)*125µs; FS bInterval is in frames → log2+3.
            let biv = interval.max(1) as u32;
            let ivl = (biv.ilog2()).min(12) + 3; // 3..=15
            w32(ep, ivl << 16);
            w32(ep + 4, (7 << 3) | (mps << 16) | (3 << 1)); // Interrupt IN, MPS, CErr=3
            w64(ep + 8, int_pa | 1); // TR Dequeue | DCS=1
            // Avg TRB Length | Max ESIT Payload (FS INT requires ESIT == MPS).
            let esit = mps.min(0xffff);
            w32(ep + 16, esit | (esit << 16));
        }
    }

    /// Queue one interrupt IN transfer for the keyboard's 8-byte boot report.
    unsafe fn queue_interrupt(&self, kbd: &mut Kbd) {
        unsafe {
            self.arm_int(kbd.int_ring_va, kbd.int_ring_pa, &mut kbd.int_enqueue, &mut kbd.int_cycle, kbd.report_pa, 8, kbd.slot, kbd.int_dci);
        }
    }

    /// Queue one interrupt IN transfer of `len` bytes on `(ring, slot, dci)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn arm_int(&self, ring_va: usize, ring_pa: u64, enq: &mut usize, cycle: &mut u32, report_pa: u64, len: u32, slot: u8, dci: u8) {
        unsafe {
            Self::ring_push(ring_va, ring_pa, enq, cycle, report_pa, len, (TRB_NORMAL << 10) | (1 << 5));
            self.doorbell(slot, dci as u32);
        }
    }

    /// Drain the shared event ring once, routing each transfer event to the
    /// keyboard (→ decoded bytes into `key_buf`) or the pointer (→ `crate::mouse`)
    /// by slot + endpoint DCI, and re-arm that endpoint. Both `poll_key` and
    /// `poll_mouse` call this; a single drainer avoids the two stealing each
    /// other's events off the one ring.
    fn pump_events(&mut self) {
        let mut kbd = self.kbd.take();
        let mut mouse = self.mouse.take();
        // SAFETY: rings/buffers are live for the controller's lifetime.
        unsafe {
            while let Some(ev) = self.next_event() {
                if (ev.control >> 10) & 0x3f != EVT_TRANSFER {
                    continue;
                }
                let slot = ((ev.control >> 24) & 0xff) as u8;
                let dci = ((ev.control >> 16) & 0x1f) as u8;
                if let Some(k) = kbd.as_mut() {
                    if k.slot == slot && k.int_dci == dci {
                        // Non-coherent DMA (real Apple): drop the stale cached
                        // copy so we read the report the controller just wrote.
                        dma_invalidate(k.report_va, 8);
                        let mut report = [0u8; 8];
                        for (i, b) in report.iter_mut().enumerate() {
                            *b = read_volatile((k.report_va + i) as *const u8);
                        }
                        let shift = report[0] & 0x22 != 0;
                        let ctrl = report[0] & 0x11 != 0;
                        for &usage in &report[2..8] {
                            if usage != 0 && !k.prev[2..8].contains(&usage) {
                                // Arrow/nav keys become the ANSI sequences a
                                // serial terminal sends, so the shell/editor
                                // decode one encoding for every input path.
                                // Ctrl+Tab = pane-focus toggle (`ESC [ T`).
                                let mut seq = [0u8; crate::keyrepeat::SEQ_MAX];
                                let mut n = 0usize;
                                if let Some(s) = match usage {
                                    0x52 => Some(&b"[A"[..]),  // Up
                                    0x51 => Some(&b"[B"[..]),  // Down
                                    0x4f => Some(&b"[C"[..]),  // Right
                                    0x50 => Some(&b"[D"[..]),  // Left
                                    0x4a => Some(&b"[H"[..]),  // Home
                                    0x4d => Some(&b"[F"[..]),  // End
                                    0x4b => Some(&b"[5~"[..]), // PgUp
                                    0x4e => Some(&b"[6~"[..]), // PgDn
                                    0x4c => Some(&b"[3~"[..]), // Delete
                                    0x2b if ctrl => Some(&b"[T"[..]), // Ctrl+Tab
                                    _ => None,
                                } {
                                    seq[0] = 0x1b;
                                    n = 1;
                                    for &b in s.iter().take(seq.len() - 1) {
                                        seq[n] = b;
                                        n += 1;
                                    }
                                } else if let Some(a) = hid_to_ascii(usage, shift, ctrl) {
                                    seq[0] = a;
                                    n = 1;
                                }
                                for &b in &seq[..n] {
                                    self.key_push(b);
                                }
                                // Arm software typematic for the newest press.
                                if n > 0 {
                                    k.rep.press(&seq[..n], crate::arch::now_ms());
                                    k.rep_usage = usage;
                                }
                            }
                        }
                        // The armed key was released: stop repeating it.
                        if k.rep_usage != 0 && !report[2..8].contains(&k.rep_usage) {
                            k.rep.release();
                            k.rep_usage = 0;
                        }
                        k.prev = report;
                        self.queue_interrupt(k);
                        continue;
                    }
                }
                if let Some(m) = mouse.as_mut() {
                    if m.slot == slot && m.int_dci == dci {
                        let n = (m.report_len as usize).min(16);
                        dma_invalidate(m.report_va, n);
                        let mut rep = [0u8; 16];
                        for (i, b) in rep.iter_mut().enumerate().take(n) {
                            *b = read_volatile((m.report_va + i) as *const u8);
                        }
                        let lo = m.layout;
                        let field = |off: u8| -> u32 {
                            let o = off as usize;
                            if lo.field_bytes >= 2 {
                                (rep[o] as u32) | ((*rep.get(o + 1).unwrap_or(&0) as u32) << 8)
                            } else {
                                rep[o] as u32
                            }
                        };
                        let (x, y) = (field(lo.x_byte), field(lo.y_byte));
                        if lo.relative {
                            let sext = |v: u32| if lo.field_bytes >= 2 { v as u16 as i16 as i32 } else { v as u8 as i8 as i32 };
                            crate::mouse::move_rel(sext(x), sext(y));
                        } else {
                            crate::mouse::set_abs(x as i32, y as i32, lo.scale_max);
                        }
                        crate::mouse::set_left(rep[lo.btn_byte as usize] & 1 != 0);
                        // Scroll wheel (HID Usage 0x38): a signed 8-bit delta,
                        // +1 = wheel away from the user (scroll up). Feed it
                        // straight through — mouse::add_wheel treats + as up.
                        if let Some(wb) = lo.wheel_byte {
                            if let Some(&z) = rep.get(wb as usize) {
                                let dz = z as i8 as i32;
                                if dz != 0 {
                                    crate::mouse::add_wheel(dz);
                                }
                            }
                        }
                        self.arm_int(m.int_ring_va, m.int_ring_pa, &mut m.int_enqueue, &mut m.int_cycle, m.report_pa, m.report_len, m.slot, m.int_dci);
                        continue;
                    }
                }
            }
        }
        self.kbd = kbd;
        self.mouse = mouse;
    }

    fn key_push(&mut self, b: u8) {
        let n = (self.key_head + 1) % self.key_buf.len();
        if n != self.key_tail {
            self.key_buf[self.key_head] = b;
            self.key_head = n;
        }
    }

    /// The next decoded keyboard byte, if any (drains + routes events first).
    /// Also where held-key repeats are synthesized: USB HID reports only
    /// press/release edges, so the armed [`crate::keyrepeat::Typematic`]
    /// re-emits the held key's bytes at an accelerating rate.
    pub fn poll_key(&mut self) -> Option<u8> {
        self.pump_events();
        let rep = self.kbd.as_mut().and_then(|k| k.rep.poll(crate::arch::now_ms()));
        if let Some((seq, n)) = rep {
            for &b in &seq[..n] {
                self.key_push(b);
            }
        }
        if self.key_head == self.key_tail {
            None
        } else {
            let b = self.key_buf[self.key_tail];
            self.key_tail = (self.key_tail + 1) % self.key_buf.len();
            Some(b)
        }
    }

    /// Drain pending pointer reports into [`crate::mouse`] (no-op without a USB
    /// pointer). Shares the event drain with `poll_key`.
    pub fn poll_mouse(&mut self) {
        self.pump_events();
    }
}

/// Append downstream hub `port` to a route string at the first empty nibble.
/// xHCI route strings pack one 4-bit hub-port number per tier below the root
/// hub (`bits[3:0]` = the first-tier hub's downstream port, etc.), up to 5 tiers.
/// A root-port device has route 0, so a device on `port` of a root-port hub gets
/// `port` in the low nibble.
fn push_route(hub_route: u32, port: u8) -> u32 {
    for tier in 0..5 {
        if (hub_route >> (4 * tier)) & 0xf == 0 {
            return hub_route | (((port as u32) & 0xf) << (4 * tier));
        }
    }
    hub_route
}

/// Pack a USB setup packet into the little-endian u64 a Setup Stage TRB expects.
fn setup(bm_request_type: u8, b_request: u8, w_value: u16, w_index: u16, w_length: u16) -> u64 {
    (bm_request_type as u64)
        | ((b_request as u64) << 8)
        | ((w_value as u64) << 16)
        | ((w_index as u64) << 32)
        | ((w_length as u64) << 48)
}

unsafe fn read_u16(addr: usize) -> u16 {
    unsafe { (read_volatile(addr as *const u8) as u16) | ((read_volatile((addr + 1) as *const u8) as u16) << 8) }
}

/// Walk a configuration descriptor looking for a HID boot-keyboard interface
/// (class 3, subclass 1, protocol 1) and its interrupt IN endpoint. Returns
/// `(interface number, endpoint address, max packet size, bInterval)`.
unsafe fn parse_hid_keyboard(buf: usize, len: usize) -> Option<(u8, u8, u32, u8)> {
    let mut i = 0usize;
    let mut in_kbd_iface = false;
    let mut iface_num = 0u8;
    while i + 2 <= len {
        let blen = unsafe { read_volatile((buf + i) as *const u8) } as usize;
        let btype = unsafe { read_volatile((buf + i + 1) as *const u8) };
        if blen < 2 {
            break;
        }
        match btype {
            0x04 => {
                // Interface: bInterfaceNumber@2, class@5, subclass@6, protocol@7.
                iface_num = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let class = unsafe { read_volatile((buf + i + 5) as *const u8) };
                let sub = unsafe { read_volatile((buf + i + 6) as *const u8) };
                let proto = unsafe { read_volatile((buf + i + 7) as *const u8) };
                in_kbd_iface = class == 3 && sub == 1 && proto == 1;
            }
            0x05 if in_kbd_iface => {
                // Endpoint: address@2, attributes@3, wMaxPacketSize@4, interval@6.
                let addr = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let attr = unsafe { read_volatile((buf + i + 3) as *const u8) };
                if addr & 0x80 != 0 && attr & 0x3 == 3 {
                    let mps = unsafe { read_u16(buf + i + 4) } as u32 & 0x7ff;
                    let interval = unsafe { read_volatile((buf + i + 6) as *const u8) };
                    return Some((iface_num, addr, mps.max(8), interval));
                }
            }
            _ => {}
        }
        i += blen;
    }
    None
}

/// Walk a config descriptor for a HID **pointer** interface — a USB tablet
/// (class 3, protocol 0) or a boot mouse (class 3, protocol 2) — and its
/// interrupt IN endpoint. Returns `(interface, endpoint addr, max packet,
/// bInterval, bInterfaceProtocol)`. Skips the keyboard (protocol 1).
unsafe fn parse_hid_pointer(buf: usize, len: usize) -> Option<(u8, u8, u32, u8, u8)> {
    let mut i = 0usize;
    let mut in_ptr_iface = false;
    let mut iface_num = 0u8;
    let mut proto = 0u8;
    while i + 2 <= len {
        let blen = unsafe { read_volatile((buf + i) as *const u8) } as usize;
        let btype = unsafe { read_volatile((buf + i + 1) as *const u8) };
        if blen < 2 {
            break;
        }
        match btype {
            0x04 => {
                iface_num = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let class = unsafe { read_volatile((buf + i + 5) as *const u8) };
                let p = unsafe { read_volatile((buf + i + 7) as *const u8) };
                // HID pointer: mouse (proto 2) or tablet/absolute (proto 0). Not
                // the keyboard (proto 1).
                in_ptr_iface = class == 3 && (p == 2 || p == 0);
                proto = p;
            }
            0x05 if in_ptr_iface => {
                let addr = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let attr = unsafe { read_volatile((buf + i + 3) as *const u8) };
                if addr & 0x80 != 0 && attr & 0x3 == 3 {
                    let mps = unsafe { read_u16(buf + i + 4) } as u32 & 0x7ff;
                    let interval = unsafe { read_volatile((buf + i + 6) as *const u8) };
                    return Some((iface_num, addr, mps.max(4), interval, proto));
                }
            }
            _ => {}
        }
        i += blen;
    }
    None
}

/// Find the report-descriptor length for interface `iface` from its HID (0x21)
/// descriptor in the config buffer (wDescriptorLength at offset 7).
unsafe fn parse_hid_report_len(buf: usize, len: usize, iface: u8) -> Option<u16> {
    let mut i = 0usize;
    let mut in_iface = false;
    while i + 2 <= len {
        let blen = unsafe { read_volatile((buf + i) as *const u8) } as usize;
        let btype = unsafe { read_volatile((buf + i + 1) as *const u8) };
        if blen < 2 {
            break;
        }
        if btype == 0x04 {
            in_iface = unsafe { read_volatile((buf + i + 2) as *const u8) } == iface;
        } else if btype == 0x21 && in_iface && i + 9 <= len {
            return Some(unsafe { read_u16(buf + i + 7) });
        }
        i += blen;
    }
    None
}

/// Parse a HID **report descriptor** to locate the pointer's X/Y/button fields
/// (works for any tablet/mouse layout, incl. a leading report-ID byte). Walks
/// the item stream tracking usage page / report size / count / id and, on each
/// Input item, assigns bit offsets to its usages.
unsafe fn parse_report_layout(buf: usize, len: usize) -> Option<PtrLayout> {
    let (mut usage_page, mut report_size, mut report_count) = (0u16, 0u32, 0u32);
    let mut logical_max = 0i32;
    let mut x_scale = 0i32; // logical max in effect when the X field is seen
    let mut usages = [0u16; 16];
    let mut nusg = 0usize;
    let mut bit = 0u32; // current bit offset within the input report
    let (mut x, mut y, mut btn, mut rel): (Option<(u32, u32)>, Option<(u32, u32)>, Option<u32>, bool) = (None, None, None, false);
    let mut wheel: Option<u32> = None; // bit offset of the Wheel field (Usage 0x38)
    let mut i = 0usize;
    while i < len {
        let b = unsafe { read_volatile((buf + i) as *const u8) };
        i += 1;
        let size = match b & 3 {
            3 => 4,
            n => n as usize,
        };
        if i + size > len {
            break;
        }
        let mut data = 0u32;
        for k in 0..size {
            data |= (unsafe { read_volatile((buf + i + k) as *const u8) } as u32) << (8 * k);
        }
        i += size;
        let (typ, tag) = ((b >> 2) & 3, b >> 4);
        match (typ, tag) {
            (1, 0x0) => usage_page = data as u16,
            (1, 0x7) => report_size = data,
            (1, 0x9) => report_count = data,
            (1, 0x8) => bit = 8, // Report ID declared → reports carry a 1-byte ID prefix
            (1, 0x2) => logical_max = data as i32,
            (2, 0x0) => {
                if nusg < usages.len() {
                    usages[nusg] = data as u16;
                    nusg += 1;
                }
            }
            (0, 0x8) => {
                // Input: assign report_count fields of report_size bits.
                if data & 1 == 0 {
                    for f in 0..report_count {
                        let u = if (f as usize) < nusg {
                            usages[f as usize]
                        } else if nusg > 0 {
                            usages[nusg - 1]
                        } else {
                            0
                        };
                        let fb = bit + f * report_size;
                        if usage_page == 0x01 && u == 0x30 {
                            x = Some((fb, report_size));
                            rel = data & 4 != 0;
                            x_scale = logical_max; // capture the max in effect for X
                        } else if usage_page == 0x01 && u == 0x31 {
                            y = Some((fb, report_size));
                        } else if usage_page == 0x01 && u == 0x38 && wheel.is_none() {
                            wheel = Some(fb); // Generic Desktop / Wheel
                        } else if usage_page == 0x09 && btn.is_none() {
                            btn = Some(fb);
                        }
                    }
                }
                bit += report_count * report_size;
                nusg = 0;
            }
            (0, _) => nusg = 0, // other main items (Output/Feature/Collection): clear locals
            _ => {}
        }
    }
    let (xb, xs) = x?;
    let (yb, _) = y?;
    Some(PtrLayout {
        btn_byte: btn.map(|b| (b / 8) as u8).unwrap_or(0),
        x_byte: (xb / 8) as u8,
        y_byte: (yb / 8) as u8,
        field_bytes: (xs / 8).max(1) as u8,
        relative: rel,
        scale_max: if x_scale > 0 { x_scale } else { 0x7fff },
        wheel_byte: wheel.map(|w| (w / 8) as u8),
    })
}

/// Map a USB HID keyboard usage id to an ASCII byte (US layout). Handles
/// letters, digits, common punctuation, space/enter/tab/backspace; Shift for
/// the upper register; Ctrl+letter -> control code (Ctrl+C=3, Ctrl+D=4).
fn hid_to_ascii(usage: u8, shift: bool, ctrl: bool) -> Option<u8> {
    let base: u8 = match usage {
        0x04..=0x1d => b'a' + (usage - 0x04),         // a..z
        0x1e..=0x26 => b'1' + (usage - 0x1e),         // 1..9
        0x27 => b'0',
        0x28 => b'\r', // Enter
        0x29 => 0x1b,  // Esc
        0x2a => 0x08,  // Backspace
        0x2b => b'\t', // Tab
        0x2c => b' ',  // Space
        0x2d => b'-',
        0x2e => b'=',
        0x2f => b'[',
        0x30 => b']',
        0x31 => b'\\',
        0x33 => b';',
        0x34 => b'\'',
        0x35 => b'`',
        0x36 => b',',
        0x37 => b'.',
        0x38 => b'/',
        _ => return None,
    };
    let ch = if shift { shift_ascii(base) } else { base };
    if ctrl && ch.is_ascii_alphabetic() {
        Some(ch.to_ascii_uppercase() & 0x1f)
    } else {
        Some(ch)
    }
}

/// The shifted form of a US-layout key.
fn shift_ascii(c: u8) -> u8 {
    match c {
        b'a'..=b'z' => c.to_ascii_uppercase(),
        b'1' => b'!',
        b'2' => b'@',
        b'3' => b'#',
        b'4' => b'$',
        b'5' => b'%',
        b'6' => b'^',
        b'7' => b'&',
        b'8' => b'*',
        b'9' => b'(',
        b'0' => b')',
        b'-' => b'_',
        b'=' => b'+',
        b'[' => b'{',
        b']' => b'}',
        b'\\' => b'|',
        b';' => b':',
        b'\'' => b'"',
        b'`' => b'~',
        b',' => b'<',
        b'.' => b'>',
        b'/' => b'?',
        other => other,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A boot mouse with a scroll wheel: 3 button bits + 5 pad, then X, Y,
    /// Wheel as three 8-bit relative fields — the classic descriptor VBox's
    /// USB pointer and most USB mice emit. Verifies the parser locates all four
    /// fields (the wheel offset is what makes scroll work).
    #[test_case]
    fn parses_boot_mouse_with_wheel() {
        #[rustfmt::skip]
        let desc: &[u8] = &[
            0x05,0x01, 0x09,0x02, 0xA1,0x01, 0x09,0x01, 0xA1,0x00,
            0x05,0x09, 0x19,0x01, 0x29,0x03, 0x15,0x00, 0x25,0x01,
            0x95,0x03, 0x75,0x01, 0x81,0x02,             // 3 button bits
            0x95,0x01, 0x75,0x05, 0x81,0x03,             // 5 pad bits
            0x05,0x01, 0x09,0x30, 0x09,0x31, 0x09,0x38,  // X, Y, Wheel
            0x75,0x08, 0x95,0x03, 0x81,0x06,             // 3 x 8-bit relative
            0xC0, 0xC0,
        ];
        let lo = unsafe { parse_report_layout(desc.as_ptr() as usize, desc.len()) }.expect("layout");
        assert_eq!(lo.btn_byte, 0);
        assert_eq!(lo.x_byte, 1);
        assert_eq!(lo.y_byte, 2);
        assert_eq!(lo.field_bytes, 1);
        assert!(lo.relative);
        assert_eq!(lo.wheel_byte, Some(3), "the scroll wheel field must be located");
    }

    /// Endpoint Context dword 4 packing: Average TRB Length + Max ESIT Payload
    /// (the VBox interrupt-IN fix — payload must be non-zero for FS INT).
    #[test_case]
    fn ep_context_esit_payload_packing() {
        // Mirror build_input_configure's dword-4 formula: both halves = mps.
        let mps = 8u32;
        let esit = mps.min(0xffff);
        let dw4 = esit | (esit << 16);
        assert_eq!(dw4 & 0xffff, 8, "Average TRB Length");
        assert_eq!(dw4 >> 16, 8, "Max ESIT Payload Lo");
        // Zero payload is the bug we ship against.
        assert_ne!(0u32, dw4 >> 16);
    }

    /// An absolute tablet (16-bit X/Y, no wheel): the fallback layout QEMU's
    /// usb-tablet uses. Verifies absolute mode, 2-byte fields, and that a
    /// missing wheel stays `None` (so we never inject phantom scroll).
    #[test_case]
    fn parses_absolute_tablet_without_wheel() {
        #[rustfmt::skip]
        let desc: &[u8] = &[
            0x05,0x01, 0x09,0x02, 0xA1,0x01, 0x09,0x01, 0xA1,0x00,
            0x05,0x09, 0x19,0x01, 0x29,0x03, 0x15,0x00, 0x25,0x01,
            0x95,0x03, 0x75,0x01, 0x81,0x02,             // 3 button bits
            0x95,0x01, 0x75,0x05, 0x81,0x03,             // 5 pad bits
            0x05,0x01, 0x09,0x30, 0x09,0x31,             // X, Y
            0x75,0x10, 0x95,0x02, 0x81,0x02,             // 2 x 16-bit absolute
        ];
        let lo = unsafe { parse_report_layout(desc.as_ptr() as usize, desc.len()) }.expect("layout");
        assert!(!lo.relative);
        assert_eq!(lo.field_bytes, 2);
        assert_eq!(lo.x_byte, 1); // byte 0 = buttons, X at byte 1
        assert_eq!(lo.wheel_byte, None);
    }

    #[test_case]
    fn hid_keyboard_ascii_mapping() {
        assert_eq!(hid_to_ascii(0x04, false, false), Some(b'a')); // 'a'
        assert_eq!(hid_to_ascii(0x04, true, false), Some(b'A')); // Shift+a
        assert_eq!(hid_to_ascii(0x06, false, true), Some(3)); // Ctrl+c = 0x03
        assert_eq!(hid_to_ascii(0x28, false, false), Some(b'\r')); // Enter
        assert_eq!(hid_to_ascii(0x2c, false, false), Some(b' ')); // Space
        assert_eq!(hid_to_ascii(0x00, false, false), None); // no key
    }

    /// Route strings pack one hub-port nibble per tier below the root hub: a
    /// root-port device is route 0; a device on port P of a root-port hub is P
    /// in the low nibble; a second tier fills the next nibble.
    #[test_case]
    fn route_string_packs_one_nibble_per_tier() {
        assert_eq!(push_route(0, 3), 0x3); // tier-1 device on hub port 3
        assert_eq!(push_route(0, 7), 0x7);
        assert_eq!(push_route(0x3, 5), 0x53); // tier-2 device behind port 3 then 5
        assert_eq!(push_route(0x53, 1), 0x153); // tier-3
        // Port numbers are masked to a nibble (hubs have ≤15 ports on this path).
        assert_eq!(push_route(0, 0xf), 0xf);
    }

    /// The Address-Device slot-context word-0 packing used by
    /// `build_input_slot_ep0`: route string in [19:0], speed in [23:20],
    /// context-entries in [31:27]; and word-2 carries the TT parent for a LS/FS
    /// device behind a HS hub.
    #[test_case]
    fn slot_context_route_speed_and_tt_packing() {
        let route = 0x53u32;
        let speed = 2u32; // low speed
        let d0 = (1 << 27) | (speed << 20) | (route & 0xf_ffff);
        assert_eq!(d0 & 0xf_ffff, 0x53, "route string");
        assert_eq!((d0 >> 20) & 0xf, 2, "speed");
        assert_eq!((d0 >> 27) & 0x1f, 1, "context entries");
        // Word-2 TT: hub slot id [7:0] | hub port [15:8].
        let (parent_slot, parent_port) = (4u32, 3u32);
        let d2 = parent_slot | (parent_port << 8);
        assert_eq!(d2 & 0xff, 4);
        assert_eq!((d2 >> 8) & 0xff, 3);
    }

    /// `mark_hub`'s slot-context bit math: Hub flag at bit 26, number of ports in
    /// [31:24], TT think time in [17:16] — set without disturbing the copied
    /// route/speed/port fields.
    #[test_case]
    fn hub_slot_context_bit_math() {
        // Start from a slot context with speed=HS, root port 2, entries=1.
        let d0_in = (1u32 << 27) | (3 << 20);
        let d1_in = (2u32) << 16;
        let d2_in = 0u32;
        let nbr_ports = 4u32;
        let ttt = 0b10u32;
        let d0 = d0_in | (1 << 26);
        let d1 = (d1_in & 0x00ff_ffff) | (nbr_ports << 24);
        let d2 = (d2_in & !(0b11 << 16)) | ((ttt & 0x3) << 16);
        assert_eq!((d0 >> 26) & 1, 1, "Hub flag");
        assert_eq!((d0 >> 20) & 0xf, 3, "speed preserved");
        assert_eq!((d1 >> 24) & 0xff, 4, "number of ports");
        assert_eq!((d1 >> 16) & 0xff, 2, "root hub port preserved");
        assert_eq!((d2 >> 16) & 0x3, 0b10, "TT think time");
    }
}
