//! **Built-in audio on Apple Silicon**: NCO → MCA → ADMAC → TAS2764 over I2C.
//!
//! Five blocks have to agree before a sample reaches the speaker, and each one
//! is silent when it is wrong:
//!
//! ```text
//!   PMGR      each block's own power domains, resolved by phandle   [pmgr]
//!   NCO       900 MHz reference -> 3.072 MHz master clock         [nco]
//!   MCA       master clock -> bit clock + frame sync, serialises  [mca]
//!   ADMAC     memory -> the serialiser, via a DART                [admac]
//!   TAS2764   I2S -> the speaker, configured over I2C             [tas2764]
//! ```
//!
//! The path is a port of m1n1's `proxyclient/experiments/speaker_amp.py`, which
//! plays audio through a Mac mini's built-in speaker in 148 lines. Every address,
//! channel, I2C address and power-domain name is **discovered from the device
//! tree** rather than taken from that script: the reference is an M1 and this is
//! an M2, and the two differ in exactly the places a script hardcodes (the
//! amplifier is at `0x38` here, not `0x31`; the reset GPIO is a different pin).
//!
//! **The serial format is Linux's, not the script's**, and that was the last
//! thing standing between a chain that read back perfect and a speaker that
//! stayed silent. The script drives a one-clock TDM pulse at a 12.288 MHz master
//! clock; this machine's amplifier is driven by Linux as
//! `SND_SOC_DAIFMT_I2S | IB_IF` — 64 bit-clocks per frame (two 32-bit slots), a
//! square FSYNC, one bit of data delay, `BCLK_POL` clear. Given the pulse it
//! accepts the clock, latches `tdm_clock_error`, and shuts itself down.
//!
//! # Boot bring-up, gated on the tree
//!
//! `sound::autodetect` calls [`up`] at boot **when the device tree names
//! `ti,tas2764` and `apple,mca`**. That is not a hunt through undiscovered
//! MMIO — the tree states the path, and a missing node is a no-op. `/audio
//! up` remains as a retry. `/audio probe` is still the read-only first step
//! if a human wants to inspect the amp without taking it out of shutdown.
//!
//! `/audio probe` is the read-only half — power on, then read the amplifier's
//! registers over I2C and report. It writes nothing to the amplifier, so it is
//! the safe first thing to run on a machine, and it splits "the I2C bus works
//! and the chip is there" from every failure further down the chain.
//!
//! # What is not implemented
//!
//! Capture (the microphone is behind the always-on processor, a different
//! device entirely), the headphone jack (a CS42L84 on the same bus — its codec
//! driver is a separate port), and stereo (the built-in speaker is mono, and the
//! I2S serialiser here is configured for one slot).

pub mod admac;
pub mod i2c;
pub mod mca;
pub mod nco;
pub mod pmgr;
pub mod tas2764;

/// Sample rate the stream runs at. Every caller's PCM is resampled to this.
pub const RATE: u32 = 48_000;
/// Bit clocks per I2S frame. Linux macaudio's **primary** FE uses
/// `bclk_ratio = 64` (two 32-bit slots) for the Mac mini's single speaker.
/// The M1 `speaker_amp.py` TDM pulse is 256 clocks; that ratio with a
/// one-clock FSYNC is what the SN012776 latched as `tdm_clock_error`.
pub const BITS_PER_FRAME: u32 = 64;

/// The power domains m1n1's reference script enables, by name.
///
/// **Documentation, not the mechanism.** Powering these four by label was the
/// first version and it left the DMA engine's IOMMU gated — reading that block's
/// lock register then took a synchronous external abort, because a gated block
/// does not read as zero, it does not decode at all. The bring-up now enables
/// the domains **each node declares** (`pmgr::enable_domains_of`), which on this
/// machine is seven for the MCA alone. Kept because it names what the reference
/// touches, which is the first thing to compare against when a domain is
/// missing.
pub const REFERENCE_DOMAINS: &[&[u8]] = &[b"i2c1", b"sio_adma", b"audio_p", b"mca0"];

/// Convert one signed 16-bit sample to the word the serialiser expects.
///
/// The stream is 32-bit slots carrying a left-justified sample, so the 16 bits
/// go in the **top** half. Putting them in the bottom half is not silence — it
/// is the sample scaled down by 65536, i.e. inaudible output from a path that
/// otherwise looks like it is working.
pub fn sample_word(s: i16) -> u32 {
    ((s as i32) << 16) as u32
}

/// GPIO register offset for `pin` in an `apple,pinctrl` block.
pub fn gpio_reg(pin: u32) -> u64 {
    pin as u64 * 4
}

/// A pinctrl register value that drives `pin` as an output at `level`,
/// preserving the pull/peripheral configuration the firmware left.
///
/// Bit 0 is the output level, bits 3:1 the mode (1 = output), bit 9 marks the
/// configuration complete. Writing a whole word copied from another machine's
/// script also rewrites the pull configuration, which is how a reset line ends
/// up floating.
pub fn gpio_out_value(current: u32, level: bool) -> u32 {
    const MODE_OUT: u32 = 1;
    const CFG_DONE: u32 = 1 << 9;
    (current & !0xf) | (MODE_OUT << 1) | CFG_DONE | level as u32
}

/// True when this machine's device tree names the built-in speaker path.
///
/// Read-only: a missing `ti,tas2764` or `apple,mca` is "this box has no
/// speaker amp", not a failed probe. Used so boot can bring the path up
/// without guessing, and so a machine without the part never touches it.
pub fn builtin_present() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if !crate::arch::aarch64::is_apple() {
            return false;
        }
        let dtb = crate::arch::aarch64::boot::boot_x0();
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        unsafe {
            crate::fdt::reg_of_compatible(dtb, b"ti,tas2764").is_some()
                && crate::fdt::reg_of_compatible(dtb, b"apple,mca").is_some()
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(target_arch = "aarch64")]
mod hw;
#[cfg(target_arch = "aarch64")]
pub use hw::{command, dump, probe_summary, set_level, up};

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_sample_is_left_justified_in_its_slot() {
        // 32-bit slots carrying a 16-bit sample in the top half. In the bottom
        // half the same audio plays at 1/65536 amplitude — silence you can only
        // distinguish from a dead path with an oscilloscope.
        assert_eq!(sample_word(0), 0);
        assert_eq!(sample_word(1), 0x0001_0000);
        assert_eq!(sample_word(i16::MAX), 0x7fff_0000);
        assert_eq!(sample_word(-1), 0xffff_0000);
        assert_eq!(sample_word(i16::MIN), 0x8000_0000);
    }

    #[test_case]
    fn a_gpio_write_keeps_the_configuration_it_did_not_come_for() {
        // Only the level and the mode are ours; the pull configuration in the
        // upper bits belongs to whatever set the pin up.
        let firmware = 0x0007_6a00 | (0b101 << 5);
        let hi = gpio_out_value(firmware, true);
        let lo = gpio_out_value(firmware, false);
        assert_eq!(hi & 1, 1);
        assert_eq!(lo & 1, 0);
        assert_eq!((hi >> 1) & 0x7, 1, "mode = output");
        assert_ne!(hi & (1 << 9), 0, "config done");
        assert_eq!(hi & (0b111 << 5), firmware & (0b111 << 5), "pull config preserved");
        assert_eq!(hi & !0xf, lo & !0xf, "level is the only difference");
    }

    #[test_case]
    fn the_gpio_register_is_one_word_per_pin() {
        assert_eq!(gpio_reg(0), 0);
        assert_eq!(gpio_reg(88), 0x160);
    }

    #[test_case]
    fn the_clock_chain_is_self_consistent() {
        // The three numbers that have to agree: the master clock is the frame
        // rate times the frame length, and the NCO must be able to make it.
        assert_eq!(RATE * BITS_PER_FRAME, 3_072_000);
        assert!(nco::calc_regvals(900_000_000, (RATE * BITS_PER_FRAME) as u64).is_some());
        // The 256-clock TDM rate is still a legal NCO target — it is just not
        // the frame this machine's amplifier will lock to.
        assert!(nco::calc_regvals(900_000_000, 48_000 * 256).is_some());
    }
}
