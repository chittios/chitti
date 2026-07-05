//! **Intel AC'97 audio controller** (PCI class 04:01) — VirtualBox's classic
//! audio adapter and countless real PC chipsets (ICH/ICH2…). AC'97 is an
//! inherently x86 device (two I/O-port BAR register files: the NAM mixer and
//! the NABM bus-master engine), so it lives under `arch::x86_64` alongside SB16;
//! aarch64 gets virtio-snd + HDA.
//!
//! Poll-driven, exposed through the arch-neutral [`crate::sound::SndDevice`]
//! trait. Fixed 48 kHz (the QEMU/VBox AC'97 has no variable-rate), so the driver
//! resamples 3:1 to keep the 16 kHz contract. Playback (PCM Out) and capture
//! (PCM In) each run a Buffer Descriptor List of DMA buffers.

use crate::arch::x86_64::pci::{self, PciDevice};
use crate::arch::x86_64::port::{inb, inw, outb, outl, outw};
use crate::sound::SndDevice;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

// NAM (mixer) register offsets, relative to BAR0.
const NAM_RESET: u16 = 0x00;
const NAM_MASTER_VOL: u16 = 0x02;
const NAM_PCM_VOL: u16 = 0x18;
const NAM_REC_SEL: u16 = 0x1a;
const NAM_REC_GAIN: u16 = 0x1c;
const NAM_EXT_AUDIO: u16 = 0x28;

// NABM (bus master) per-stream register block bases, relative to BAR1.
const NABM_PI: u16 = 0x00; // PCM in
const NABM_PO: u16 = 0x10; // PCM out
const NABM_GLOB_CNT: u16 = 0x2c;
// Per-stream offsets within a NABM block.
const BDBAR: u16 = 0x00; // buffer descriptor list base (phys)
const LVI: u16 = 0x05; // last valid index
const SR: u16 = 0x06; // status
const CR: u16 = 0x0b; // control (byte)

const CR_RPBM: u8 = 0x01; // run/pause bus master
const CR_RR: u8 = 0x02; // reset registers
const SR_DCH: u16 = 0x01; // DMA controller halted

const RATE: usize = 48000;
const CHUNK: usize = 9600; // 100 ms S16 mono @ 48 kHz
const NBUF: usize = 4;

/// A BDL entry: 32-bit buffer phys addr + 16-bit sample count + control.
#[repr(C)]
#[derive(Clone, Copy)]
struct Bdl {
    addr: u32,
    samples: u16,
    ctrl: u16,
}

pub struct Ac97 {
    nam: u16, // BAR0 I/O base (mixer)
    nabm: u16, // BAR1 I/O base (bus master)
    po_bdl: (u64, u64),
    po_buf: (u64, u64),
    pi_bdl: (u64, u64),
    pi_ring: (u64, u64),
    play_open: bool,
    cap_on: bool,
    cap_pos: usize,
    pending: VecDeque<i16>,
    decim: [i32; 2],
}

fn io_bar(d: &PciDevice, idx: u8) -> u16 {
    // I/O BARs have bit0 set; base is bits [15:2].
    let off = 0x10 + idx as u16 * 4;
    (pci::read32(d.bus, d.dev, d.func, off) & 0xfffc) as u16
}

impl Ac97 {
    pub fn probe() -> Option<Box<dyn SndDevice>> {
        let d = match pci::find_class_sub(0x04, 0x01) {
            Some(d) => d,
            None => {
                crate::ktrace::log("ac97", "probe: no PCI class 04.01 device");
                return None;
            }
        };
        crate::ktrace::log_fmt(format_args!("ac97: found {:04x}:{:04x} at {}:{}.{}", d.vendor, d.device, d.bus, d.dev, d.func));
        // Enable I/O space + bus master (COMMAND bits 0 and 2).
        let cmd = pci::read32(d.bus, d.dev, d.func, 0x04);
        pci::write32(d.bus, d.dev, d.func, 0x04, cmd | 0b101);
        let nam = io_bar(&d, 0);
        let nabm = io_bar(&d, 1);
        if nam == 0 || nabm == 0 {
            return None;
        }
        // SAFETY: I/O ports of the located AC'97 function.
        unsafe {
            // Cold reset the mixer + enable the bus master.
            outl(nabm + NABM_GLOB_CNT, 0x02); // cold reset
            outw(nam + NAM_RESET, 0); // any write resets the mixer
            // Unmute + full volume on master and PCM; select mic for record.
            outw(nam + NAM_MASTER_VOL, 0x0000);
            outw(nam + NAM_PCM_VOL, 0x0000);
            outw(nam + NAM_EXT_AUDIO, inw(nam + NAM_EXT_AUDIO)); // leave VRA off (48k)
            outw(nam + NAM_REC_SEL, 0x0000); // mic in? 0=mic, per codec; 0x0404=line
            outw(nam + NAM_REC_GAIN, 0x0000); // 0 dB, unmuted
        }
        let po_bdl = crate::mm::alloc_dma(NBUF * 8)?;
        let po_buf = crate::mm::alloc_dma(NBUF * CHUNK)?;
        let pi_bdl = crate::mm::alloc_dma(NBUF * 8)?;
        let pi_ring = crate::mm::alloc_dma(NBUF * CHUNK)?;
        crate::ktrace::log_fmt(format_args!("ac97: up (vendor {:04x} dev {:04x}), NAM {:#x} NABM {:#x}", d.vendor, d.device, nam, nabm));
        Some(Box::new(Ac97 {
            nam,
            nabm,
            po_bdl,
            po_buf,
            pi_bdl,
            pi_ring,
            play_open: false,
            cap_on: false,
            cap_pos: 0,
            pending: VecDeque::new(),
            decim: [0; 2],
        }))
    }

    unsafe fn write_bdl(bdl_virt: u64, i: usize, addr: u32, samples: u16, ctrl: u16) {
        let e = Bdl { addr, samples, ctrl };
        unsafe { core::ptr::write_volatile((bdl_virt + (i * 8) as u64) as *mut Bdl, e) };
    }
}

impl SndDevice for Ac97 {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        // AC'97 PCM-out runs at a fixed 48 kHz; resample whatever the caller
        // provides (16 kHz tones, 24 kHz TTS) instead of assuming 16 kHz.
        let up: Vec<i16> = crate::sound::resample(pcm, hz, 48_000);
        // SAFETY: DMA buffers + AC'97 I/O ports are the driver's.
        unsafe {
            // Reset PCM Out engine.
            outb(self.nabm + NABM_PO + CR, CR_RR);
            while inb(self.nabm + NABM_PO + CR) & CR_RR != 0 {
                core::hint::spin_loop();
            }
            // Chunk the audio across up to NBUF BDL entries.
            let per = CHUNK / 2; // samples per buffer
            let mut off = 0usize;
            let mut idx = 0usize;
            while off < up.len() && idx < NBUF {
                let n = (up.len() - off).min(per);
                let dst = self.po_buf.1 + (idx * CHUNK) as u64;
                core::ptr::copy_nonoverlapping(up.as_ptr().add(off), dst as *mut i16, n);
                let ioc = if off + n >= up.len() { 0x8000 } else { 0 }; // IOC on last
                Self::write_bdl(self.po_bdl.1, idx, (self.po_buf.0 + (idx * CHUNK) as u64) as u32, n as u16, ioc);
                off += n;
                idx += 1;
            }
            if idx == 0 {
                return Ok(());
            }
            outl(self.nabm + NABM_PO + BDBAR, self.po_bdl.0 as u32);
            outb(self.nabm + NABM_PO + LVI, (idx - 1) as u8);
            outb(self.nabm + NABM_PO + CR, CR_RPBM); // run
        }
        self.play_open = true;
        Ok(())
    }

    fn playing(&mut self) -> bool {
        if !self.play_open {
            return false;
        }
        // SAFETY: reading the PCM Out status register.
        unsafe {
            if inw(self.nabm + NABM_PO + SR) & SR_DCH != 0 {
                self.play_open = false;
                return false;
            }
        }
        true
    }

    fn capture_start(&mut self, _hz: u32) -> Result<(), &'static str> {
        if self.cap_on {
            return Ok(());
        }
        // SAFETY: DMA + I/O ports.
        unsafe {
            outb(self.nabm + NABM_PI + CR, CR_RR);
            while inb(self.nabm + NABM_PI + CR) & CR_RR != 0 {
                core::hint::spin_loop();
            }
            for i in 0..NBUF {
                Self::write_bdl(self.pi_bdl.1, i, (self.pi_ring.0 + (i * CHUNK) as u64) as u32, (CHUNK / 2) as u16, 0);
            }
            outl(self.nabm + NABM_PI + BDBAR, self.pi_bdl.0 as u32);
            outb(self.nabm + NABM_PI + LVI, (NBUF - 1) as u8);
            outb(self.nabm + NABM_PI + CR, CR_RPBM);
        }
        self.cap_on = true;
        self.cap_pos = 0;
        self.pending.clear();
        self.decim = [0; 2];
        Ok(())
    }

    fn capture_read(&mut self, out: &mut [i16]) -> usize {
        if self.cap_on {
            // The current index register (CIV) tells us how far DMA has filled;
            // we read the completed buffers between cap_pos and CIV.
            // SAFETY: reading I/O + the capture ring DMA.
            unsafe {
                let civ = inb(self.nabm + NABM_PI + 0x04) as usize % NBUF; // CIV
                while self.cap_pos != civ {
                    let base = self.pi_ring.1 + (self.cap_pos * CHUNK) as u64;
                    for k in 0..CHUNK / 2 {
                        let s = core::ptr::read_volatile((base + (k * 2) as u64) as *const i16);
                        self.decim[0] += s as i32;
                        self.decim[1] += 1;
                        if self.decim[1] == 3 {
                            self.pending.push_back((self.decim[0] / 3) as i16);
                            self.decim = [0; 2];
                        }
                    }
                    // Re-arm this buffer and advance LVI so DMA keeps running.
                    outb(self.nabm + NABM_PI + LVI, ((self.cap_pos + NBUF - 1) % NBUF) as u8);
                    self.cap_pos = (self.cap_pos + 1) % NBUF;
                }
            }
        }
        let mut n = 0;
        while n < out.len() {
            match self.pending.pop_front() {
                Some(s) => {
                    out[n] = s;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    fn capture_stop(&mut self) {
        if self.cap_on {
            // SAFETY: stopping the PCM In engine.
            unsafe { outb(self.nabm + NABM_PI + CR, 0) };
            self.cap_on = false;
            self.pending.clear();
        }
    }
}
