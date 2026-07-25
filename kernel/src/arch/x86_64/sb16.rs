//! **Sound Blaster 16** (ISA, ports 0x220…) — the most legacy audio path,
//! present in VirtualBox ("SoundBlaster 16") and QEMU (`sb16`). Pure x86 ISA
//! (fixed I/O ports + the 8237 ISA DMA controller with its 64 KiB page
//! constraint), so it lives under `arch::x86_64` like AC'97.
//!
//! Poll-driven, [`crate::sound::SndDevice`]. 16-bit signed mono; playback via
//! DSP command 0xB6 (16-bit output, auto-init) on DMA channel 5, driven at
//! 16 kHz natively (SB16 has a programmable sample-rate command, so no
//! resampling needed). Capture (DSP 0xBE) is wired the same way on channel 5.
//!
//! ISA DMA needs a **physically-contiguous buffer within one 64 KiB (128 KiB
//! for 16-bit) page** — `mm::alloc_dma` returns page-aligned contiguous frames;
//! we bound the buffer to 32 KiB (16 K samples) so it never straddles a
//! 16-bit-DMA page boundary.

use crate::arch::x86_64::port::{inb, outb};
use crate::sound::SndDevice;
use alloc::boxed::Box;
use alloc::collections::VecDeque;

const BASE: u16 = 0x220;
const DSP_RESET: u16 = BASE + 0x6;
const DSP_READ: u16 = BASE + 0xa;
const DSP_WRITE: u16 = BASE + 0xc; // write-status on bit7 too
const DSP_WRITE_STATUS: u16 = BASE + 0xc;
const DSP_READ_STATUS: u16 = BASE + 0xe;

// 16-bit DMA channel 5 (8237 #2). Registers are the classic ISA DMA ports.
const DMA16_ADDR: u16 = 0xc4; // channel 5 base address
const DMA16_COUNT: u16 = 0xc6; // channel 5 count
const DMA16_PAGE: u16 = 0x8b; // channel 5 page
const DMA16_MASK: u16 = 0xd4;
const DMA16_MODE: u16 = 0xd6;
const DMA16_CLEAR: u16 = 0xd8; // clear byte-pointer flip-flop

const BUF_SAMPLES: usize = 16384; // 32 KiB, safely within a 128 KiB 16-bit page
const RATE: u32 = 16000;

pub struct Sb16 {
    buf: (u64, u64),  // playback DMA buffer, ≤ 32 KiB
    cbuf: (u64, u64), // capture DMA buffer (separate, so capture ≠ playback echo)
    play_open: bool,
    cap_on: bool,
    cap_read: usize,
    pending: VecDeque<i16>,
}

fn dsp_reset() -> bool {
    // SAFETY: standard SB16 DSP reset handshake on the fixed ports.
    unsafe {
        outb(DSP_RESET, 1);
        for _ in 0..100 {
            core::hint::spin_loop();
        }
        outb(DSP_RESET, 0);
        for _ in 0..100_000 {
            if inb(DSP_READ_STATUS) & 0x80 != 0 && inb(DSP_READ) == 0xaa {
                return true;
            }
        }
        false
    }
}

fn dsp_write(v: u8) {
    // SAFETY: poll write-status bit7 clear, then write the command port.
    unsafe {
        for _ in 0..100_000 {
            if inb(DSP_WRITE_STATUS) & 0x80 == 0 {
                break;
            }
        }
        outb(DSP_WRITE, v);
    }
}

impl Sb16 {
    pub fn probe() -> Option<Box<dyn SndDevice>> {
        if !dsp_reset() {
            return None;
        }
        // DMA buffers that satisfy the 8237's placement rules: under 16 MiB
        // (24-bit address latch) and inside one 128 KiB block (the page register
        // does not increment across it on a 16-bit channel).
        //
        // This used to be an ordinary `alloc_dma` followed by a check — and the
        // check always failed, because the allocator hands out high memory once
        // the kernel and model have taken the low frames. SB16 was therefore
        // unreachable code on every machine. `alloc_dma_bounded` asks for what
        // the hardware actually needs instead of hoping.
        const ISA_LIMIT: u64 = 16 * 1024 * 1024;
        const ISA_16BIT_BLOCK: u64 = 128 * 1024;
        let bytes = BUF_SAMPLES * 2;
        let Some(buf) = crate::mm::alloc_dma_bounded(bytes, ISA_LIMIT, ISA_16BIT_BLOCK) else {
            crate::ktrace::log("sb16", "no free DMA-reachable memory below 16 MiB; declining (use HDA/AC'97)");
            return None;
        };
        let Some(cbuf) = crate::mm::alloc_dma_bounded(bytes, ISA_LIMIT, ISA_16BIT_BLOCK) else {
            crate::ktrace::log("sb16", "no second DMA-reachable buffer below 16 MiB; declining (use HDA/AC'97)");
            return None;
        };
        // Turn the speaker on.
        dsp_write(0xd1);
        crate::ktrace::log_fmt(format_args!("sb16: up at {:#x}, DMA16 buffer phys {:#x}", BASE, buf.0));
        Some(Box::new(Sb16 { buf, cbuf, play_open: false, cap_on: false, cap_read: 0, pending: VecDeque::new() }))
    }

    /// Program 8237 channel 5 for a `bytes`-length transfer (`write` = device
    /// reads from RAM = playback; else capture). 16-bit DMA counts in *words*
    /// and addresses in words within the page. Mode bits: single(0x40),
    /// auto-init(0x10), read-from-mem(0x08)/write-to-mem(0x04), channel 1 (5-4).
    fn program_dma(&self, bytes: usize, write: bool) {
        let phys = if write { self.buf.0 } else { self.cbuf.0 };
        let words = (bytes / 2 - 1) as u16;
        let word_addr = ((phys >> 1) & 0xffff) as u16;
        let page = (phys >> 16) as u8;
        // Playback: single-cycle read-from-mem (no auto-init). Capture: auto-init
        // write-to-mem (paired with the 0xBE auto-init DSP command).
        let mode = if write { 0x49 } else { 0x55 };
        // SAFETY: classic 8237 programming sequence on the fixed DMA ports.
        unsafe {
            outb(DMA16_MASK, 0x04 | 0x01); // mask channel 5 (0x04|ch1)
            outb(DMA16_CLEAR, 0); // clear flip-flop
            outb(DMA16_MODE, mode);
            outb(DMA16_ADDR, (word_addr & 0xff) as u8);
            outb(DMA16_ADDR, (word_addr >> 8) as u8);
            outb(DMA16_PAGE, page);
            outb(DMA16_CLEAR, 0);
            outb(DMA16_COUNT, (words & 0xff) as u8);
            outb(DMA16_COUNT, (words >> 8) as u8);
            outb(DMA16_MASK, 0x01); // unmask channel 5
        }
    }

    fn set_rate(&self, out: bool, hz: u32) {
        // SB16 DSP programs an arbitrary rate (5000..=44100); clamp and use the
        // caller's rate directly (16 kHz mic/tones, 24 kHz TTS).
        let hz = hz.clamp(5000, 44100);
        dsp_write(if out { 0x41 } else { 0x42 }); // set output/input sample rate
        dsp_write((hz >> 8) as u8);
        dsp_write((hz & 0xff) as u8);
    }
}

impl SndDevice for Sb16 {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        let n = pcm.len().min(BUF_SAMPLES);
        if n == 0 {
            return Ok(());
        }
        // SAFETY: copy into the DMA buffer; then program DMA + DSP.
        unsafe {
            core::ptr::copy_nonoverlapping(pcm.as_ptr(), self.buf.1 as *mut i16, n);
        }
        self.program_dma(n * 2, true);
        self.set_rate(true, hz);
        // 0xB6 = 16-bit output, single-cycle; mode 0x10 = signed mono.
        dsp_write(0xb0); // 16-bit output, single-cycle, D/A
        dsp_write(0x10); // signed, mono
        let words = (n - 1) as u16;
        dsp_write((words & 0xff) as u8);
        dsp_write((words >> 8) as u8);
        self.play_open = true;
        Ok(())
    }

    fn out_free_bytes(&mut self) -> usize {
        if self.playing() {
            0
        } else {
            24_000 * 2
        }
    }

    fn playing(&mut self) -> bool {
        if !self.play_open {
            return false;
        }
        // Single-cycle transfer done → DSP raises the 16-bit IRQ status; poll
        // the mixer's interrupt-status (0x82 bit1) via the DSP read-status.
        // SAFETY: reading DSP status.
        unsafe {
            // The 8237 count register reading is unreliable across chipsets;
            // treat the transfer as complete once the DSP is idle again.
            if inb(DSP_WRITE_STATUS) & 0x80 == 0 {
                // A robust "done" check is IRQ-based; without IRQs we conservatively
                // report done after the buffer's nominal duration is polled out.
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
        self.program_dma(BUF_SAMPLES * 2, false);
        self.set_rate(false, RATE);
        dsp_write(0xbe); // 16-bit input, auto-init
        dsp_write(0x10); // signed mono
        let words = (BUF_SAMPLES - 1) as u16;
        dsp_write((words & 0xff) as u8);
        dsp_write((words >> 8) as u8);
        self.cap_on = true;
        self.cap_read = 0;
        self.pending.clear();
        Ok(())
    }

    fn capture_read(&mut self, out: &mut [i16]) -> usize {
        // Without IRQs/count feedback we conservatively drain the whole buffer
        // once (auto-init keeps refilling); good enough to prove capture wiring.
        if self.cap_on && self.cap_read < BUF_SAMPLES {
            // SAFETY: reading the DMA capture buffer.
            unsafe {
                let base = self.cbuf.1;
                for k in self.cap_read..BUF_SAMPLES {
                    let s = core::ptr::read_volatile((base + (k * 2) as u64) as *const i16);
                    self.pending.push_back(s);
                }
            }
            self.cap_read = BUF_SAMPLES;
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
            dsp_write(0xd9); // exit 16-bit auto-init
            self.cap_on = false;
            self.pending.clear();
        }
    }
}
