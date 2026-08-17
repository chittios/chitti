//! **ADMAC** — the audio DMA engine that feeds the I2S serialiser.
//!
//! The `dmas` phandle of the machine's `apple,mca` node points here (resolve it,
//! do not assume: an M2 also has a SIO coprocessor, and picking the engine by
//! SoC generation gets it backwards on this machine).
//!
//! The model is two small hardware rings per channel. The host pushes
//! **descriptors** — a 64-bit address, a length, and flags — four words at a
//! time into a write port, and the engine pushes **reports** back the same way
//! as each one completes. Neither ring is in memory; both are read and written
//! through a single register, four words per entry, and reading a partial entry
//! desynchronises the ring for good. That is why the pure encoding lives here
//! and is tested: a descriptor is 4 words in a fixed order, and getting the
//! address halves the wrong way round points the engine at a plausible address
//! in the wrong half of memory.
//!
//! Ported from m1n1 `proxyclient/m1n1/hw/admac.py`.

/// Global registers.
pub mod reg {
    pub const TX_EN: u64 = 0x00;
    pub const TX_EN_CLR: u64 = 0x04;
    pub const RX_EN: u64 = 0x08;
    pub const RX_EN_CLR: u64 = 0x0c;
    /// Per-channel block base and stride.
    pub const CHAN_BASE: u64 = 0x8000;
    pub const CHAN_STRIDE: u64 = 0x200;
    /// Offsets within a channel block.
    pub const CHAN_CTL: u64 = 0x00;
    pub const CHAN_STATUS: u64 = 0x10; // + line*4
    pub const CHAN_INTMASK: u64 = 0x20; // + line*4
    pub const CHAN_BUSWIDTH: u64 = 0x40;
    pub const CHAN_BURSTSIZE: u64 = 0x54;
    pub const CHAN_RESIDUE: u64 = 0x64;
    pub const CHAN_DESC_RING: u64 = 0x70;
    pub const CHAN_REPORT_RING: u64 = 0x74;
    /// Descriptor/report ports, indexed by `channel / 2`.
    pub const TX_DESC_WRITE: u64 = 0x10000;
    pub const TX_REPORT_READ: u64 = 0x10100;
}

/// `CHAN_CTL` bits.
pub const CTL_RESET_RINGS: u32 = 1 << 0;
pub const CTL_CLEAR_OF_UF: u32 = 1 << 1;

/// `CHAN_STATUS` / `CHAN_INTMASK` bits.
pub const ST_DESC_DONE: u32 = 1 << 0;
pub const ST_DESC_RING_EMPTY: u32 = 1 << 4;
pub const ST_REPORT_RING_FULL: u32 = 1 << 5;
pub const ST_RING_ERR: u32 = 1 << 6;

/// Ring-status bits (`R_RING`).
pub const RING_EMPTY: u32 = 1 << 8;
pub const RING_FULL: u32 = 1 << 9;
pub const RING_ERR: u32 = 1 << 10;

/// Descriptor flags.
pub const DESC_NOTIFY: u32 = 1 << 16;
/// **Not used, and the reason is a hardware bug worth recording**: once a
/// descriptor with `REPEAT` is loaded, every descriptor loaded afterwards
/// repeats too, and only a power-domain reset stops it (m1n1's note). A stream
/// built on it cannot be ended.
pub const DESC_REPEAT: u32 = 1 << 17;

/// Bus word width (`E_BUSWIDTH`): the size of one sample as the engine moves it.
pub const BUSWIDTH_32BIT: u32 = 2;
/// Words per frame (`E_FRAME`).
pub const FRAME_1_WORD: u32 = 0;

/// The burst size m1n1 programs for the audio path. Undocumented; carried over.
pub const BURSTSIZE: u32 = 0x00c0_0060;

/// A descriptor as the four words the write port takes, in order.
///
/// `[addr_lo, addr_hi, length, flags]` — the address is split across two 32-bit
/// words low half first. `length` is in **bytes**.
pub fn descriptor(iova: u64, len: u32, id: u8, notify: bool) -> [u32; 4] {
    [
        (iova & 0xffff_ffff) as u32,
        (iova >> 32) as u32,
        len,
        (id as u32) | if notify { DESC_NOTIFY } else { 0 },
    ]
}

/// The descriptor id carried back in a report's flags.
pub fn report_id(flags: u32) -> u8 {
    (flags & 0xff) as u8
}

/// `CHAN_BUSWIDTH` value for a word size and frame size.
pub fn buswidth(word: u32, frame: u32) -> u32 {
    (word & 0x7) | ((frame & 0x7) << 4)
}

/// Physical offset of a channel's register block.
pub fn chan_reg(channel: u32, off: u64) -> u64 {
    reg::CHAN_BASE + channel as u64 * reg::CHAN_STRIDE + off
}

/// The enable-register bit for a channel. Channels are paired — even are
/// transmit, odd receive — and the bit is per *pair*, so channel 4's bit is 2,
/// not 4. Using the channel number enables somebody else's stream.
pub fn enable_bit(channel: u32) -> u32 {
    1 << (channel / 2)
}

/// Whether a channel number is a transmit channel.
pub fn is_tx(channel: u32) -> bool {
    channel % 2 == 0
}

/// The descriptor/report port for a transmit channel.
pub fn tx_desc_port(channel: u32) -> u64 {
    reg::TX_DESC_WRITE + (channel as u64 / 2) * 4
}
pub fn tx_report_port(channel: u32) -> u64 {
    reg::TX_REPORT_READ + (channel as u64 / 2) * 4
}

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;

    /// One transmit channel of the engine.
    pub struct AdmacTx {
        base: u64,
        channel: u32,
        next_id: u8,
    }

    impl AdmacTx {
        /// # Safety
        /// `base` must be the ADMAC node's `reg` window, its power domain on,
        /// and `channel` an even (transmit) channel the machine's I2S owns.
        pub unsafe fn new(base: u64, size: usize, channel: u32) -> Self {
            AdmacTx { base: crate::mm::map_mmio(base, size), channel, next_id: 0 }
        }

        fn w(&self, off: u64, v: u32) {
            // SAFETY: inside the mapped ADMAC window.
            unsafe {
                core::arch::asm!("str {0:w}, [{1}]", in(reg) v, in(reg) self.base + off, options(nostack))
            };
        }

        fn r(&self, off: u64) -> u32 {
            let v: u32;
            // SAFETY: inside the mapped ADMAC window.
            unsafe {
                core::arch::asm!("ldr {0:w}, [{1}]", out(reg) v, in(reg) self.base + off, options(nostack))
            };
            v
        }

        /// Reset the channel's rings and set the transfer geometry.
        pub fn reset(&self) {
            self.w(chan_reg(self.channel, reg::CHAN_CTL), CTL_RESET_RINGS | CTL_CLEAR_OF_UF);
            self.w(chan_reg(self.channel, reg::CHAN_CTL), 0);
            self.w(chan_reg(self.channel, reg::CHAN_BURSTSIZE), BURSTSIZE);
            self.w(
                chan_reg(self.channel, reg::CHAN_BUSWIDTH),
                buswidth(BUSWIDTH_32BIT, FRAME_1_WORD),
            );
            // Drain any report the previous owner left behind, so the first real
            // report is not attributed to a descriptor we never submitted.
            while !self.reports_empty() {
                self.take_report();
            }
        }

        /// Whether another descriptor fits in the ring.
        pub fn can_submit(&self) -> bool {
            self.r(chan_reg(self.channel, reg::CHAN_DESC_RING)) & RING_FULL == 0
        }

        /// Hand the engine a buffer at device address `iova`, `len` bytes.
        /// Returns the descriptor id, or `None` when the ring is full.
        pub fn submit(&mut self, iova: u64, len: u32) -> Option<u8> {
            if !self.can_submit() {
                return None;
            }
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            // All four words go to the same port, in order. A partial write
            // leaves the ring's internal read-out counter mid-entry, and every
            // later descriptor is then interpreted one word out.
            for w in descriptor(iova, len, id, true) {
                self.w(tx_desc_port(self.channel), w);
            }
            Some(id)
        }

        pub fn reports_empty(&self) -> bool {
            self.r(chan_reg(self.channel, reg::CHAN_REPORT_RING)) & RING_EMPTY != 0
        }

        /// Read one report (four words) and return its descriptor id.
        pub fn take_report(&self) -> u8 {
            let mut w = [0u32; 4];
            for slot in w.iter_mut() {
                *slot = self.r(tx_report_port(self.channel));
            }
            report_id(w[3])
        }

        /// Start the channel.
        pub fn enable(&self) {
            self.w(
                chan_reg(self.channel, reg::CHAN_INTMASK),
                ST_DESC_DONE | ST_DESC_RING_EMPTY | ST_REPORT_RING_FULL | ST_RING_ERR,
            );
            self.w(reg::TX_EN, enable_bit(self.channel));
        }

        pub fn disable(&self) {
            self.w(reg::TX_EN_CLR, enable_bit(self.channel));
        }

        /// Channel status, for diagnosis.
        pub fn status(&self) -> u32 {
            self.r(chan_reg(self.channel, reg::CHAN_STATUS))
        }

        /// Acknowledge a completed descriptor and clear a ring error if one is
        /// latched (both rings have to be told, per m1n1).
        pub fn ack(&self) {
            let st = self.status();
            self.w(chan_reg(self.channel, reg::CHAN_STATUS), ST_DESC_DONE);
            if st & ST_RING_ERR != 0 {
                self.w(chan_reg(self.channel, reg::CHAN_DESC_RING), RING_ERR);
                self.w(chan_reg(self.channel, reg::CHAN_REPORT_RING), RING_ERR);
            }
        }

        /// Bytes of the current descriptor still to be moved.
        pub fn residue(&self) -> u32 {
            self.r(chan_reg(self.channel, reg::CHAN_RESIDUE))
        }

        /// Channel state, for `/audio dump`.
        pub fn dump(&self) {
            crate::ktrace::log_fmt(format_args!(
                "audio: admac ch{} TX_EN={:#x} status={:#010x} desc_ring={:#010x} report_ring={:#010x}",
                self.channel,
                self.r(reg::TX_EN),
                self.status(),
                self.r(chan_reg(self.channel, reg::CHAN_DESC_RING)),
                self.r(chan_reg(self.channel, reg::CHAN_REPORT_RING)),
            ));
            crate::ktrace::log_fmt(format_args!(
                "audio: admac ch{} buswidth={:#x} burstsize={:#x} residue={}",
                self.channel,
                self.r(chan_reg(self.channel, reg::CHAN_BUSWIDTH)),
                self.r(chan_reg(self.channel, reg::CHAN_BURSTSIZE)),
                self.residue(),
            ));
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::AdmacTx;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_descriptor_is_four_words_low_half_first() {
        // The address is split across two words. Swapping them points the engine
        // at an address that is usually still mapped — so it DMAs, and plays
        // whatever happens to be there.
        let d = descriptor(0x1_2345_6000, 0x4000, 7, true);
        assert_eq!(d[0], 0x2345_6000, "low half");
        assert_eq!(d[1], 0x1, "high half");
        assert_eq!(d[2], 0x4000, "length in bytes");
        assert_eq!(d[3] & 0xff, 7, "descriptor id");
        assert_ne!(d[3] & DESC_NOTIFY, 0);
        // Without NOTIFY the engine completes the descriptor silently and the
        // report never arrives, so a queue never drains from the host's view.
        assert_eq!(descriptor(0, 1, 0, false)[3] & DESC_NOTIFY, 0);
    }

    #[test_case]
    fn the_descriptor_id_round_trips_through_a_report() {
        for id in [0u8, 1, 127, 255] {
            let d = descriptor(0x1000, 0x100, id, true);
            assert_eq!(report_id(d[3]), id);
        }
        // The report's upper flag bits must not leak into the id.
        assert_eq!(report_id(0x0f00_0042), 0x42);
    }

    #[test_case]
    fn the_enable_bit_is_per_channel_pair() {
        // Channels are TX/RX pairs sharing one enable bit. Using the channel
        // number would start a different stream — channel 4 is bit 2.
        assert_eq!(enable_bit(0), 1);
        assert_eq!(enable_bit(1), 1, "the RX half of the same pair");
        assert_eq!(enable_bit(4), 1 << 2);
        assert_eq!(enable_bit(8), 1 << 4);
        assert!(is_tx(0) && is_tx(4) && !is_tx(1));
    }

    #[test_case]
    fn channel_registers_and_ports_are_where_the_reference_puts_them() {
        assert_eq!(chan_reg(0, reg::CHAN_CTL), 0x8000);
        assert_eq!(chan_reg(0, reg::CHAN_DESC_RING), 0x8070);
        assert_eq!(chan_reg(1, reg::CHAN_CTL), 0x8200, "0x200 stride");
        assert_eq!(chan_reg(4, reg::CHAN_BUSWIDTH), 0x8000 + 4 * 0x200 + 0x40);
        // The descriptor and report ports are indexed by pair, not by channel.
        assert_eq!(tx_desc_port(0), 0x10000);
        assert_eq!(tx_desc_port(4), 0x10008);
        assert_eq!(tx_report_port(4), 0x10108);
    }

    #[test_case]
    fn buswidth_packs_word_and_frame_in_separate_fields() {
        // WORD is bits 2:0 and FRAME bits 6:4; packing them into one field makes
        // the engine read samples at a plausible wrong size.
        let v = buswidth(BUSWIDTH_32BIT, FRAME_1_WORD);
        assert_eq!(v & 0x7, 2);
        assert_eq!((v >> 4) & 0x7, 0);
        assert_eq!(buswidth(2, 1), 0x12);
    }
}
