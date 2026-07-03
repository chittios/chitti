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
    // Context structure size: 32 or 64 bytes (HCCPARAMS1.CSZ).
    ctx_size: usize,
    // The enumerated keyboard, if one was found + configured.
    kbd: Option<Kbd>,
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
// Event TRB types.
const EVT_TRANSFER: u32 = 32;
const EVT_CMD_COMPLETION: u32 = 33;
const CC_SUCCESS: u32 = 1;

// PORTSC bits.
const PORTSC_CCS: u32 = 1 << 0; // current connect status
const PORTSC_PED: u32 = 1 << 1; // port enabled
const PORTSC_PR: u32 = 1 << 4; // port reset
// RW1C/RW1CS bits to preserve (write 0) when doing a read-modify-write.
const PORTSC_RW1CS: u32 = PORTSC_PED | (0x7f << 17);

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
            // Context size: 64 bytes if HCCPARAMS1.CSZ (bit 2), else 32.
            let ctx_size = if r32(mmio + 0x10) & (1 << 2) != 0 { 64 } else { 32 };

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
                ctx_size,
                kbd: None,
            })
        }
    }

    /// PORTSC of 1-based `port`.
    fn portsc(&self, port: u8) -> usize {
        self.op + OP_PORTS + (port as usize - 1) * 0x10
    }

    /// Ring a doorbell (slot 0 = command ring; slot N, target = endpoint DCI).
    unsafe fn doorbell(&self, slot: u8, target: u32) {
        unsafe { w32(self.db + slot as usize * 4, target) };
    }

    /// Push a TRB onto a command/transfer ring and advance, wrapping at the
    /// Link TRB in the last slot. Associated (no `self`) so it can borrow a
    /// ring's fields directly.
    unsafe fn ring_push(va: usize, pa: u64, enq: &mut usize, cycle: &mut u32, param: u64, status: u32, control: u32) {
        let control = (control & !1) | *cycle;
        unsafe { write_volatile((va + *enq * 16) as *mut Trb, Trb { param, status, control }) };
        *enq += 1;
        if *enq == RING_TRBS - 1 {
            let link = (TRB_LINK << 10) | 0x2 | *cycle;
            unsafe { write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: link }) };
            *enq = 0;
            *cycle ^= 1;
        }
    }

    /// Consume the next event TRB if one is ready (cycle bit matches), updating
    /// the dequeue pointer + ERDP. Non-blocking.
    unsafe fn next_event(&mut self) -> Option<Trb> {
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
    unsafe fn wait_event(&mut self, want_type: u32) -> Option<Trb> {
        let mut spins = 0u64;
        loop {
            if let Some(ev) = unsafe { self.next_event() } {
                if (ev.control >> 10) & 0x3f == want_type {
                    return Some(ev);
                }
            } else {
                spins += 1;
                if spins > 20_000_000 {
                    return None;
                }
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
                Some(ev) => (ev.status >> 24) & 0xff == CC_SUCCESS || (ev.status >> 24) & 0xff == 13, /* SHORT_PACKET ok */
                None => false,
            }
        }
    }

    /// C2/C3: find a connected keyboard, reset its port, address the device,
    /// read its descriptors, set the config + boot protocol, and arm the
    /// interrupt IN endpoint.
    fn enumerate_keyboard(&mut self) {
        // SAFETY: single-threaded boot; all DMA regions are freshly allocated.
        unsafe {
            let Some(kbd) = self.try_enumerate() else {
                crate::ktrace::log("xhci", "no HID keyboard enumerated");
                return;
            };
            self.kbd = Some(kbd);
        }
    }

    unsafe fn try_enumerate(&mut self) -> Option<Kbd> {
        // 1. Find + reset a connected port.
        let mut port = 0u8;
        for p in 1..=self.max_ports {
            if unsafe { r32(self.portsc(p)) } & PORTSC_CCS != 0 {
                port = p;
                break;
            }
        }
        if port == 0 {
            return None;
        }
        let psc = self.portsc(port);
        unsafe {
            // Reset the port (USB2 needs it; preserve RW1C status bits).
            let v = r32(psc) & !PORTSC_RW1CS;
            w32(psc, v | PORTSC_PR);
            let mut spins = 0;
            while r32(psc) & PORTSC_PED == 0 {
                spins += 1;
                if spins > 5_000_000 {
                    return None;
                }
            }
        }
        let speed = (unsafe { r32(psc) } >> 10) & 0xf;
        let ep0_mps: u32 = match speed {
            3 => 64,      // high
            4 | 5 => 512, // super/super+
            _ => 8,       // full/low
        };
        crate::ktrace::log_fmt(format_args!("xhci: port {port} connected, speed {speed}, resetting -> enabled"));

        // 2. Enable a slot.
        let (cc, slot) = unsafe { self.command(0, TRB_ENABLE_SLOT << 10) }?;
        if cc != CC_SUCCESS {
            crate::ktrace::log_fmt(format_args!("xhci: enable slot failed cc={cc}"));
            return None;
        }

        // 3. Device context + DCBAA entry.
        let (dev_ctx_pa, _dev_ctx_va) = alloc_dma(4096)?;
        unsafe { write_volatile((self.dcbaa_va as *mut u64).add(slot as usize), dev_ctx_pa) };

        // 4. EP0 transfer ring.
        let (ep0_pa, ep0_va) = alloc_dma(RING_TRBS * 16)?;
        let ep0_va = ep0_va as usize;
        unsafe { self.init_link(ep0_va, ep0_pa) };
        let mut ep0_enq = 0usize;
        let mut ep0_cycle = 1u32;

        // 5. Input context for Address Device (add slot + EP0).
        let (in_ctx_pa, in_ctx_va) = alloc_dma(4096)?;
        let in_ctx_va = in_ctx_va as usize;
        unsafe {
            self.build_input_slot_ep0(in_ctx_va, port, speed, ep0_mps, ep0_pa);
            // 6. Address Device.
            let (cc, _) = self.command(in_ctx_pa, (TRB_ADDRESS_DEVICE << 10) | ((slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: address device failed cc={cc}"));
                return None;
            }
        }
        crate::ktrace::log_fmt(format_args!("xhci: device addressed on slot {slot}"));

        // 7-8. Read the configuration descriptor into a DMA buffer + parse.
        let (buf_pa, buf_va) = alloc_dma(4096)?;
        let buf_va = buf_va as usize;
        // GET_DESCRIPTOR(Configuration), first 9 bytes for wTotalLength.
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
        let Some((iface, ep_addr, ep_mps, interval)) = (unsafe { parse_hid_keyboard(buf_va, total as usize) }) else {
            crate::ktrace::log("xhci", "no HID boot-keyboard interface in config descriptor");
            return None;
        };

        // 9. Set configuration, 10. boot protocol.
        unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x00, 9, cfg_val as u16, 0, 0), 0, 0, false);
            // SET_PROTOCOL(boot=0) to the HID interface (class request).
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x21, 0x0b, 0, iface as u16, 0), 0, 0, false);
        }

        // 11. Interrupt IN ring + report buffer.
        let (int_pa, int_va) = alloc_dma(RING_TRBS * 16)?;
        let int_va = int_va as usize;
        unsafe { self.init_link(int_va, int_pa) };
        let (report_pa, report_va) = alloc_dma(4096)?;
        let report_va = report_va as usize;
        let epnum = (ep_addr & 0x0f) as u32;
        let int_dci = (epnum * 2 + 1) as u8; // IN endpoint

        // 12. Configure Endpoint (add slot + the interrupt endpoint).
        unsafe {
            self.build_input_configure(in_ctx_va, int_dci, ep_mps, interval, int_pa);
            let (cc, _) = self.command(in_ctx_pa, (TRB_CONFIGURE_ENDPOINT << 10) | ((slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: configure endpoint failed cc={cc}"));
                return None;
            }
        }

        let mut kbd = Kbd {
            slot,
            int_dci,
            int_ring_va: int_va,
            int_ring_pa: int_pa,
            int_enqueue: 0,
            int_cycle: 1,
            report_pa,
            report_va,
            prev: [0; 8],
        };
        // 13. Arm the first interrupt transfer.
        unsafe { self.queue_interrupt(&mut kbd) };
        crate::ktrace::log_fmt(format_args!(
            "xhci: HID keyboard ready (slot {slot}, ep {ep_addr:#x}, dci {int_dci}) -- type in the window"
        ));
        Some(kbd)
    }

    /// Initialize a transfer ring's trailing Link TRB (back to its own start).
    unsafe fn init_link(&self, va: usize, pa: u64) {
        unsafe {
            write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: (TRB_LINK << 10) | 0x2 | 1 });
        }
    }

    /// Build the input context for Address Device: add flags A0|A1, a slot
    /// context (route 0, this root port, speed, 1 context entry) and the EP0
    /// control endpoint context.
    unsafe fn build_input_slot_ep0(&self, in_ctx_va: usize, port: u8, speed: u32, ep0_mps: u32, ep0_pa: u64) {
        unsafe {
            // zero the control + slot + ep0 area
            core::ptr::write_bytes(in_ctx_va as *mut u8, 0, self.ctx_size * 3);
            w32(in_ctx_va + 4, 0b11); // Add flags: A0 (slot) | A1 (EP0)
            let slot_ctx = in_ctx_va + self.ctx_size;
            w32(slot_ctx, (1 << 27) | (speed << 20)); // ctx entries=1, speed, route=0
            w32(slot_ctx + 4, (port as u32) << 16); // root hub port number
            let ep0 = in_ctx_va + self.ctx_size * 2;
            w32(ep0 + 4, (4 << 3) | (ep0_mps << 16) | (3 << 1)); // type=Control, MPS, CErr=3
            w64(ep0 + 8, ep0_pa | 1); // TR dequeue ptr | DCS
            w32(ep0 + 16, 8); // avg TRB length
        }
    }

    /// Extend the input context for Configure Endpoint: add the interrupt IN
    /// endpoint at `dci` and bump the slot's context-entries count.
    unsafe fn build_input_configure(&self, in_ctx_va: usize, dci: u8, mps: u32, interval: u8, int_pa: u64) {
        unsafe {
            w32(in_ctx_va, 0); // drop flags
            w32(in_ctx_va + 4, 0b1 | (1 << dci)); // add A0 (slot) + the endpoint
            let slot_ctx = in_ctx_va + self.ctx_size;
            let d0 = r32(slot_ctx) & !(0x1f << 27);
            w32(slot_ctx, d0 | ((dci as u32) << 27)); // context entries = dci
            let ep = in_ctx_va + self.ctx_size * (dci as usize + 1);
            core::ptr::write_bytes(ep as *mut u8, 0, self.ctx_size);
            // Interrupt interval: xHCI encodes it as 2^(Interval) * 125us. For a
            // full-speed 1ms endpoint that's ~8; clamp bInterval into range.
            let ivl = (interval.max(1).min(16) as u32).ilog2() + 3;
            w32(ep, ivl << 16);
            w32(ep + 4, (7 << 3) | (mps << 16) | (3 << 1)); // type=Interrupt IN, MPS, CErr=3
            w64(ep + 8, int_pa | 1);
            w32(ep + 16, mps); // avg TRB length ~ max packet
        }
    }

    /// Queue one interrupt IN transfer for the keyboard's 8-byte boot report.
    unsafe fn queue_interrupt(&self, kbd: &mut Kbd) {
        unsafe {
            Self::ring_push(
                kbd.int_ring_va,
                kbd.int_ring_pa,
                &mut kbd.int_enqueue,
                &mut kbd.int_cycle,
                kbd.report_pa,
                8,
                (TRB_NORMAL << 10) | (1 << 5), // IOC
            );
            self.doorbell(kbd.slot, kbd.int_dci as u32);
        }
    }

    /// C4: drain a pending HID boot report into ASCII bytes (there may be
    /// several newly-pressed keys in one report). Returns the first, buffering
    /// the rest is unnecessary for a keyboard (one key/report in practice).
    fn poll_key(&mut self) -> Option<u8> {
        let mut kbd = self.kbd.take()?;
        let mut out = None;
        // SAFETY: kbd's rings/buffer are live for the controller's lifetime.
        unsafe {
            while let Some(ev) = self.next_event() {
                if (ev.control >> 10) & 0x3f != EVT_TRANSFER {
                    continue;
                }
                // A boot report arrived. Parse newly-pressed keys.
                let mut report = [0u8; 8];
                for (i, b) in report.iter_mut().enumerate() {
                    *b = read_volatile((kbd.report_va + i) as *const u8);
                }
                let shift = report[0] & 0x22 != 0;
                let ctrl = report[0] & 0x11 != 0;
                for &usage in &report[2..8] {
                    if usage != 0 && !kbd.prev[2..8].contains(&usage) {
                        if let Some(a) = hid_to_ascii(usage, shift, ctrl) {
                            out = out.or(Some(a));
                        }
                    }
                }
                kbd.prev = report;
                self.queue_interrupt(&mut kbd); // re-arm
                if out.is_some() {
                    break;
                }
            }
        }
        self.kbd = Some(kbd);
        out
    }
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
