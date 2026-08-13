//! **MCA** — the I2S controller. It takes the NCO's master clock, divides out a
//! bit clock and a frame sync, and shifts the DMA engine's samples onto the pins
//! the amplifier listens to.
//!
//! One *cluster* per port; each cluster has a clock generator (`MCLK` +
//! `SYNCGEN`), two transmit serialisers and two receive ones. The bring-up
//! sequence and every register here is m1n1's
//! `proxyclient/m1n1/hw/mca.py` plus the order in
//! `proxyclient/experiments/speaker_amp.py`.
//!
//! The register *encodings* live here as pure functions because each one is a
//! bit field whose failure mode is silence rather than an error: a serialiser
//! configured for the wrong slot width still runs, still consumes samples, and
//! puts nothing audible on the wire.

/// Cluster register block stride within the MCA's first `reg` window.
pub const CLUSTER_STRIDE: u64 = 0x4000;
/// Second `reg` window ("switch"), one 0x8000 block per cluster.
pub const SWITCH_STRIDE: u64 = 0x8000;

/// Offsets inside a cluster (m1n1 `MCAClusterRegs`).
pub mod reg {
    pub const MCLK_STATUS: u64 = 0x000;
    pub const MCLK_CONF: u64 = 0x004;
    pub const SYNCGEN_STATUS: u64 = 0x100;
    pub const SYNCGEN_MCLK_SEL: u64 = 0x104;
    pub const SYNCGEN_HI_PERIOD: u64 = 0x108;
    pub const SYNCGEN_LO_PERIOD: u64 = 0x10c;
    pub const PORT_ENABLES: u64 = 0x600;
    pub const PORT_CLK_SEL: u64 = 0x604;
    pub const PORT_DATA_SEL: u64 = 0x608;
    /// Transmit serialiser A, relative to the cluster base.
    pub const TXA: u64 = 0x300;
    /// Within a serialiser block.
    pub const SERDES_STATUS: u64 = 0x0;
    pub const SERDES_CONF: u64 = 0x4;
    pub const SERDES_BITDELAY: u64 = 0x8;
    pub const SERDES_CHANMASK: u64 = 0xc; // four words
}

/// `STATUS` bits shared by the clock generator and the serialisers.
pub const STATUS_EN: u32 = 1 << 0;
pub const STATUS_RST: u32 = 1 << 1;

/// `PORT_ENABLES` bits.
pub const PORT_CLOCK1: u32 = 1 << 1;
pub const PORT_CLOCK2: u32 = 1 << 2;
pub const PORT_DATA: u32 = 1 << 3;

/// Slot widths the serialiser understands (`E_SLOT_WIDTH`). The value is the
/// *encoding*, not the bit count — 32-bit slots are `0x10`, not 32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotWidth {
    W16 = 0x4,
    W20 = 0x8,
    W24 = 0xc,
    W32 = 0x10,
}

/// Pack a transmit serialiser's `CONF` register.
///
/// The fields are m1n1's `R_SERDES_CONF`: `NSLOTS` (3:0), `SLOT_WIDTH` (8:4),
/// `BCLK_POL` (10), `UNK1` (12), `UNK2` (13), `IDLE_UNDRIVEN` (14),
/// `SYNC_SEL` (18:16).
///
/// `IDLE_UNDRIVEN` is the one worth naming: with it clear the serialiser keeps
/// driving the data pin between frames, which an amplifier reads as a continuous
/// stream of whatever was last in the shift register.
pub fn serdes_conf(nslots: u32, width: SlotWidth, bclk_pol: bool, sync_sel: u32) -> u32 {
    (nslots & 0xf)
        | ((width as u32) << 4)
        | ((bclk_pol as u32) << 10)
        | (1 << 12) // UNK1, set by every known-good configuration
        | (1 << 13) // UNK2
        | (1 << 14) // IDLE_UNDRIVEN
        | ((sync_sel & 0x7) << 16)
}

/// The `SYNCGEN_HI_PERIOD` / `SYNCGEN_LO_PERIOD` pair for a frame of
/// `bits_per_frame` bit-clock periods.
///
/// m1n1's speaker path uses `HI = 0`, `LO = bits - 2`: the generator counts a
/// high period of one and a low period of the rest, and both registers are
/// "count minus one". Writing `bits` rather than `bits - 2` stretches every
/// frame by two bit clocks, which detunes the sample rate by ~3% — audible as
/// wrong pitch, not as a failure.
pub fn syncgen_periods(bits_per_frame: u32) -> (u32, u32) {
    (0, bits_per_frame.saturating_sub(2))
}

/// I2S FSYNC: even duty cycle, both registers count-minus-one.
///
/// Linux `mca_fe_hw_params` writes `HI = ratio/2 - 1`, `LO = (ratio+1)/2 - 1`.
/// A one-clock pulse (`syncgen_periods`) is TDM DSP_B, and the SN012776
/// latches `tdm_clock_error` against it — the Mac mini's machine driver is
/// `SND_SOC_DAIFMT_I2S | IB_IF`, which is a square FSYNC, not a pulse.
pub fn syncgen_i2s_periods(bits_per_frame: u32) -> (u32, u32) {
    let hi = bits_per_frame / 2;
    let lo = (bits_per_frame + 1) / 2;
    (hi.saturating_sub(1), lo.saturating_sub(1))
}

/// `MCLK_CONF` bits 11:8 — Linux `FIELD_PREP(MCLK_CONF_DIV, 1)`.
///
/// The dump on the silent Mac mini read `0x200` (div 2) because nothing wrote
/// this register. A leftover divider makes BCLK/FSYNC the right *ratio* at the
/// wrong *rate*, which the amplifier reports as a TDM clock error.
pub fn mclk_conf_div(div: u32) -> u32 {
    (div & 0xf) << 8
}

/// Linux `REG_DMA_ADAPTER_A` for a playback stream.
///
/// `nchans` is the number of audio channels (capped at 4); `sample_bits` is
/// the width already in each DMA word. We left-justify 16-bit samples into
/// 32-bit slots in software, so `sample_bits` is 32 and the pad is 0.
/// m1n1's `0x102048` is pad 8 (24-in-32) and is the wrong word here.
pub fn dma_adapter(nchans: u32, sample_bits: u32) -> u32 {
    let pad = 32u32.saturating_sub(sample_bits).min(31);
    let n = nchans.min(4);
    (n << 20) | (2 << 13) | (2 << 5) | pad
}

/// `PORT_CLK_SEL`'s selector is a **field at bits 11:8**, not the whole word.
///
/// m1n1 writes it as `PORT_CLK_SEL.set(SEL=cluster + 1)`, which lands at
/// `0x100` for cluster 0. Writing the plain number leaves `SEL = 0` — and that
/// is not a broken register, it is a port with **no clock routed to its pins**.
/// Everything inside the controller then works perfectly: the serialiser runs,
/// the DMA engine drains at real time, every status register reads correct. Only
/// the amplifier notices, by never seeing a bit clock, refusing to leave
/// shutdown, and latching no fault — because an absent clock is not a clock
/// error. That was the silence.
pub fn port_clk_sel(cluster: u32) -> u32 {
    ((cluster + 1) & 0xf) << 8
}

/// The channel mask that enables slot 0 only.
///
/// The mask is inverted — a *clear* bit enables the slot — so "channel 0 on" is
/// `0xffff_fffe`, and the intuitive `1` would enable every slot except the one
/// wanted.
pub const CHANMASK_SLOT0: u32 = 0xffff_fffe;

/// The magic word m1n1 writes to the switch block for a cluster before starting
/// the serialiser. Undocumented; carried over verbatim because the path does not
/// work without it and nothing here can derive it.
pub const SWITCH_MAGIC: u32 = 0x102048;

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;

    /// One MCA cluster: its clock generator and transmit serialiser A.
    pub struct Mca {
        cluster: u64,
        switch: u64,
        index: u64,
    }

    impl Mca {
        /// # Safety
        /// `clusters`/`switch` must be the MCA node's two `reg` windows, and the
        /// power domains for `audio_p` and this cluster must already be on.
        pub unsafe fn new(
            clusters: u64,
            clusters_size: usize,
            switch: u64,
            switch_size: usize,
            index: u64,
        ) -> Self {
            let c = crate::mm::map_mmio(clusters, clusters_size);
            let s = crate::mm::map_mmio(switch, switch_size);
            Mca { cluster: c + index * CLUSTER_STRIDE, switch: s + index * SWITCH_STRIDE, index }
        }

        fn w(&self, off: u64, v: u32) {
            // SAFETY: inside the mapped cluster window.
            unsafe {
                core::arch::asm!("str {0:w}, [{1}]", in(reg) v, in(reg) self.cluster + off, options(nostack))
            };
        }

        fn r(&self, off: u64) -> u32 {
            let v: u32;
            // SAFETY: inside the mapped cluster window.
            unsafe {
                core::arch::asm!("ldr {0:w}, [{1}]", out(reg) v, in(reg) self.cluster + off, options(nostack))
            };
            v
        }

        /// Configure the cluster for a mono 32-bit-slot I2S stream and start its
        /// clocks. The serialiser itself stays **off** — it is enabled only once
        /// the DMA engine has data queued, because a running serialiser with an
        /// empty FIFO underruns immediately.
        pub fn configure(&self, bits_per_frame: u32) {
            // Clock generator: reset, select this cluster's MCLK, program an
            // I2S (50 %) frame, then enable.
            self.w(reg::SYNCGEN_STATUS, STATUS_RST);
            self.w(reg::SYNCGEN_STATUS, 0);
            self.w(reg::SYNCGEN_MCLK_SEL, 1 + self.index as u32);
            let (hi, lo) = syncgen_i2s_periods(bits_per_frame);
            self.w(reg::SYNCGEN_HI_PERIOD, hi);
            self.w(reg::SYNCGEN_LO_PERIOD, lo);
            self.w(reg::MCLK_CONF, mclk_conf_div(1));

            // Transmit serialiser A: two 32-bit I2S slots, data delayed one
            // bit (I2S, not left-justified), BCLK not inverted. Linux
            // `MACAUDIO_DAI_FMT = I2S | IB_IF` sets BITSTART=1 and does
            // **not** set `BCLK_POL` on the MCA — the edge lives on the
            // codec. The M1 script's inverted BCLK + zero delay is TAS5770L.
            let tx = reg::TXA;
            self.w(tx + reg::SERDES_STATUS, 0);
            self.w(
                tx + reg::SERDES_CONF,
                serdes_conf(1, SlotWidth::W32, false, 1 + self.index as u32),
            );
            self.w(tx + reg::SERDES_BITDELAY, 1);
            self.w(tx + reg::SERDES_CHANMASK, CHANMASK_SLOT0);
            self.w(tx + reg::SERDES_CHANMASK + 4, CHANMASK_SLOT0);

            // Route the cluster's clocks and data out of the port.
            self.w(reg::PORT_ENABLES, PORT_CLOCK1 | PORT_CLOCK2 | PORT_DATA);
            self.w(reg::PORT_CLK_SEL, port_clk_sel(self.index as u32));
            // `PORT_DATA_SEL` genuinely is a bitmask of serialiser outputs
            // (`TXA0` is bit 0, `TXB0` bit 1, `TXA1` bit 2 …), so `cluster + 1`
            // is the raw value the reference writes and not a field like
            // `PORT_CLK_SEL`. Checked rather than assumed after the clock
            // selector turned out to be the reason for the silence.
            self.w(reg::PORT_DATA_SEL, self.index as u32 + 1);
            self.w(reg::MCLK_STATUS, STATUS_EN);
            self.w(reg::SYNCGEN_STATUS, STATUS_EN);

            // SAFETY: the switch block, sized from the MCA node's second `reg`.
            unsafe {
                core::arch::asm!(
                    "str {0:w}, [{1}]",
                    in(reg) dma_adapter(1, 32),
                    in(reg) self.switch,
                    options(nostack)
                )
            };
        }

        /// Start shifting. Call **after** the DMA engine has descriptors queued.
        ///
        /// Linux `mca_fe_early_trigger` resets the serdes with `SYNC_SEL=7`
        /// before enabling it. Skipping that leaves the first frames
        /// unsynchronised, which the SN012776 reports as a TDM clock error.
        pub fn start_tx(&self) {
            let tx = reg::TXA;
            let conf = self.r(tx + reg::SERDES_CONF);
            let sync = ((1 + self.index as u32) & 7) << 16;
            self.w(tx + reg::SERDES_CONF, conf & !(7 << 16));
            self.w(tx + reg::SERDES_CONF, (conf & !(7 << 16)) | (7 << 16));
            self.w(tx + reg::SERDES_STATUS, STATUS_RST);
            for _ in 0..2_000 {
                core::hint::spin_loop();
            }
            self.w(tx + reg::SERDES_CONF, (conf & !(7 << 16)) | sync);
            self.w(tx + reg::SERDES_STATUS, STATUS_EN);
        }

        /// Stop shifting and park the clocks.
        pub fn stop(&self) {
            self.w(reg::TXA + reg::SERDES_STATUS, 0);
            self.w(reg::SYNCGEN_STATUS, 0);
            self.w(reg::MCLK_STATUS, 0);
        }

        /// Whether the transmit serialiser reports itself enabled.
        pub fn tx_running(&self) -> bool {
            self.r(reg::TXA + reg::SERDES_STATUS) & STATUS_EN != 0
        }

        /// Every register this driver programmed, read back. Silence with a
        /// correctly-configured chain and a correctly-configured amplifier means
        /// one of them is not what it was told to be, and that is only ever
        /// answerable by reading it.
        pub fn dump(&self) {
            for (name, off) in [
                ("MCLK_STATUS", reg::MCLK_STATUS),
                ("MCLK_CONF", reg::MCLK_CONF),
                ("SYNCGEN_STATUS", reg::SYNCGEN_STATUS),
                ("SYNCGEN_MCLK_SEL", reg::SYNCGEN_MCLK_SEL),
                ("SYNCGEN_HI_PERIOD", reg::SYNCGEN_HI_PERIOD),
                ("SYNCGEN_LO_PERIOD", reg::SYNCGEN_LO_PERIOD),
                ("PORT_ENABLES", reg::PORT_ENABLES),
                ("PORT_CLK_SEL", reg::PORT_CLK_SEL),
                ("PORT_DATA_SEL", reg::PORT_DATA_SEL),
            ] {
                crate::ktrace::log_fmt(format_args!("audio: mca {name:<18} = {:#010x}", self.r(off)));
            }
            for (name, off) in [
                ("TXA STATUS", reg::SERDES_STATUS),
                ("TXA CONF", reg::SERDES_CONF),
                ("TXA BITDELAY", reg::SERDES_BITDELAY),
                ("TXA CHANMASK0", reg::SERDES_CHANMASK),
            ] {
                crate::ktrace::log_fmt(format_args!(
                    "audio: mca {name:<18} = {:#010x}",
                    self.r(reg::TXA + off)
                ));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::Mca;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_32_bit_slot_is_encoded_not_counted() {
        // `SLOT_WIDTH` is an enum in bits 8:4, so a 32-bit slot is 0x10 << 4.
        // Writing the bit *count* there selects 32 -> a reserved encoding, and
        // the serialiser runs anyway.
        let c = serdes_conf(0, SlotWidth::W32, true, 1);
        assert_eq!((c >> 4) & 0x1f, 0x10);
        assert_eq!(SlotWidth::W16 as u32, 0x4);
        assert_eq!(SlotWidth::W24 as u32, 0xc);
    }

    #[test_case]
    fn the_serdes_conf_fields_land_where_the_reference_puts_them() {
        // Reproduces speaker_amp.py's configuration exactly: NSLOTS=0,
        // SLOT_WIDTH=W_32BIT, BCLK_POL=1, UNK1=1, UNK2=1, IDLE_UNDRIVEN=1,
        // SYNC_SEL=1.
        let c = serdes_conf(0, SlotWidth::W32, true, 1);
        assert_eq!(c & 0xf, 0, "NSLOTS");
        assert_ne!(c & (1 << 10), 0, "BCLK_POL");
        assert_ne!(c & (1 << 12), 0, "UNK1");
        assert_ne!(c & (1 << 13), 0, "UNK2");
        assert_ne!(c & (1 << 14), 0, "IDLE_UNDRIVEN — the pin must idle, not repeat");
        assert_eq!((c >> 16) & 7, 1, "SYNC_SEL");
        // SLOT_WIDTH 0x10 at bit 4, BCLK_POL/UNK1/UNK2/IDLE_UNDRIVEN, SYNC_SEL 1.
        assert_eq!(c, 0x100 | 0x400 | 0x1000 | 0x2000 | 0x4000 | 0x1_0000);
    }

    #[test_case]
    fn the_frame_period_is_the_count_minus_two() {
        // A 256-bit TDM pulse is programmed as 0xfe, which is what the
        // reference writes for a 48 kHz stream off a 12.288 MHz master clock.
        assert_eq!(syncgen_periods(256), (0, 0xfe));
        assert_eq!(syncgen_periods(64), (0, 62));
        // Never underflows into a huge period on a nonsense frame size.
        assert_eq!(syncgen_periods(1), (0, 0));
        assert_eq!(syncgen_periods(0), (0, 0));
    }

    #[test_case]
    fn i2s_fsync_is_an_even_duty_cycle() {
        // Linux: HI = ratio/2 - 1, LO = (ratio+1)/2 - 1. A 64-bit I2S frame
        // is 31/31, not the TDM pulse (0, 62).
        assert_eq!(syncgen_i2s_periods(64), (31, 31));
        assert_eq!(syncgen_i2s_periods(256), (127, 127));
        assert_eq!(syncgen_i2s_periods(0), (0, 0));
        assert_ne!(syncgen_i2s_periods(64), syncgen_periods(64));
    }

    #[test_case]
    fn mclk_div_lands_in_bits_11_8() {
        assert_eq!(mclk_conf_div(1), 0x100);
        assert_eq!(mclk_conf_div(2), 0x200);
        assert_ne!(mclk_conf_div(1), 0x200, "the leftover the dump showed");
    }

    #[test_case]
    fn the_dma_adapter_is_the_linux_word_not_the_m1n1_pad() {
        // 1 channel, 32-bit words already padded: NCHANS=1, TX/RX_NCHANS=2, pad=0.
        assert_eq!(dma_adapter(1, 32), 0x104040);
        assert_ne!(dma_adapter(1, 32), SWITCH_MAGIC);
        // 24-in-32 would be pad 8, which is how m1n1's 0x102048 is built.
        assert_eq!(dma_adapter(1, 24) & 0x1f, 8);
    }

    #[test_case]
    fn the_port_clock_selector_is_a_field_not_a_number() {
        // Bits 11:8. Writing the cluster number raw leaves SEL=0, which routes
        // no clock to the port's pins — invisible in every controller register
        // and audible only as total silence.
        assert_eq!(port_clk_sel(0), 0x100);
        assert_eq!(port_clk_sel(1), 0x200);
        assert_eq!(port_clk_sel(5), 0x600);
        assert_eq!(port_clk_sel(0) & 0xff, 0, "nothing below bit 8");
        assert_ne!(port_clk_sel(0), 1, "the mistake this exists to prevent");
    }

    #[test_case]
    fn the_channel_mask_is_inverted() {
        // Clear enables. The intuitive value would enable all 31 other slots and
        // mute the one being used.
        assert_eq!(CHANMASK_SLOT0 & 1, 0, "slot 0 enabled");
        assert_eq!(CHANMASK_SLOT0 & 2, 2, "slot 1 masked off");
    }
}
