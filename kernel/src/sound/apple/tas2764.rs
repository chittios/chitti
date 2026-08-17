//! The **TAS2764 speaker amplifier** (`ti,sn012776` / `ti,tas2764`) — the last
//! link in the chain, and the only one attached to something physical.
//!
//! It is configured over I2C and does nothing until told to leave software
//! shutdown. Two properties of this part shape the code:
//!
//! * **It shuts itself down when its clock disappears.** The I2S serialiser must
//!   already be running before the amplifier is taken out of shutdown, or it
//!   goes straight back in and stays there, silently.
//! * **The gain is a real amplifier gain into a real speaker.** The register
//!   sequence here sets the *minimum* level, exactly as m1n1's `speaker_amp.py`
//!   does, and nothing raises it. Output loudness is handled in software by
//!   [`crate::sound::apply_output_gain`], which cannot damage anything. If a
//!   future change wants the hardware gain raised, that is a decision about
//!   somebody's speaker and needs a measurement, not a guess.
//!
//! The M1 mini's part is a TAS5770L (`speaker_amp.py`'s target). The M2 mini
//! (j473) is a TAS2764 / `ti,sn012776` at `0x38`. They are **not** the same
//! register file: m1n1's 0x40/0x06 TDM writes are TAS5770L encodings, and on
//! this part they program a reserved word-width and a sample-rate field of
//! zero. The chip then acknowledges `PWR_CTRL = ACTIVE` and stays in shutdown,
//! with nothing in `INT_LTCH0` — an absent-looking clock that is actually a
//! configuration the part will not lock. The sequence below is Linux's
//! `tas2764.c` / `tas2764-quirks.h` for `DEVID_SN012776`, not the M1 script.
//!
//! **What the script does not need and this does**: a reset. It runs on a
//! machine where nothing has configured the amplifier yet. Here the part was
//! driven by macOS minutes earlier and m1n1 left it untouched, so it can be
//! sitting on any *book* — and then every register write is acknowledged and
//! lands somewhere else, while reads return that book's contents. The symptom is
//! a device that answers every transaction correctly and keeps none of it, which
//! is why [`configure`] now reads one register back and says whether it took.

/// Registers used here. Page 0 of book 0, which is where the device starts.
pub mod reg {
    /// Page select — must read back 0 for the rest of these to mean anything.
    pub const PAGE: u8 = 0x00;
    /// Software reset: writing bit 0 returns the register file to defaults.
    pub const SW_RESET: u8 = 0x01;
    /// Book select. **Only reachable from page 0**, and the reason this driver
    /// could not configure anything: the device had been left on another book by
    /// whatever ran before us (macOS, then m1n1), so every write landed on some
    /// other book's register and was acknowledged and discarded, while reads
    /// returned that book's contents — a register file that answers plausibly
    /// and ignores you.
    pub const BOOK: u8 = 0x7f;
    /// Power control: mode in bits 1:0, sense blocks in 3:2.
    pub const PWR_CTRL: u8 = 0x02;
    /// Playback configuration, including the amplifier level.
    pub const CHNL_0: u8 = 0x03;
    /// TDM configuration block. Offsets are Linux `tas2764.h`, **not** the
    /// TAS5770L map m1n1's `speaker_amp.py` writes — `TDM_CFG5` is `0x0e` here.
    pub const TDM_CFG0: u8 = 0x08;
    pub const TDM_CFG1: u8 = 0x09;
    pub const TDM_CFG2: u8 = 0x0a;
    pub const TDM_CFG3: u8 = 0x0c;
    pub const TDM_CFG4: u8 = 0x0d;
    pub const TDM_CFG5: u8 = 0x0e;
    /// Clock / IRQ config; bit 2 is write-1-to-clear of the latched faults.
    pub const INT_CLK_CFG: u8 = 0x5c;
    /// **Digital** volume, 8 bits — a separate level from the amplifier gain,
    /// applied before it. m1n1's `SN012776Regs` names it and the reference
    /// script never writes it, so it sits at whatever the part reset to.
    pub const DVC: u8 = 0x1a;
    /// First of the 24-byte SN012776 brownout-prevention table.
    pub const BOP_CFG0: u8 = 0x1d;
}

/// `PWR_CTRL` mode in bits 1:0. Linux `TAS2764_PWR_CTRL_*` — **not** m1n1's
/// `0x0c`/`0x0d`/`0x0e`, which also set the sense-PD bits. On the SN012776 a
/// write of `0x0c` is rejected and the part stays at `0x0e` (software
/// shutdown), which is the silent-but-playing Mac mini.
pub const PWR_ACTIVE: u8 = 0x00;
pub const PWR_MUTE: u8 = 0x01;
pub const PWR_SHUTDOWN: u8 = 0x02;
/// `PWR_CTRL` bit 7: Linux sets this on `DEVID_SN012776` so the BOP table
/// is the one that actually runs.
pub const PWR_BOP_SRC: u8 = 1 << 7;

/// Minimum amplifier level, and the largest the part accepts.
///
/// `CHNL_0` bits 5:1 are the **analog** gain into the speaker — Linux's control
/// for this part runs 0..=0x14 with 0 the quietest — so this is the one number
/// here that can do physical damage and the one this driver never raises on its
/// own. [`GAIN_MAX`] is the part's own ceiling, not a judgement about what is
/// safe for any particular speaker.
pub const GAIN_MIN: u8 = 0x00;
pub const GAIN_MAX: u8 = 0x14;

/// Loudest **digital** volume. Inverted, per Linux's `SOC_SINGLE_TLV(...,
/// TAS2764_DVC, 0, 0xc9, 1, ...)`: 0 is unity and 0xc9 is near-mute.
///
/// Written explicitly because leaving it alone means inheriting whatever the
/// part reset to, and an attenuator sitting at 0xc9 is silence produced by a
/// chain that is otherwise working perfectly. Digital, and before the amplifier,
/// so unlike [`GAIN_MIN`] it cannot damage anything.
pub const DVC_UNITY: u8 = 0x00;

/// The `CHNL_0` value for amplifier level `n`, clamped to the part's range.
///
/// The level is bits **5:1**, not 5:0 — writing `n` unshifted halves it and
/// leaves bit 0 in the `CDS_MODE` field's neighbour, which is a quieter and
/// subtly different configuration rather than an error.
pub fn chnl_0_for_gain(n: u8) -> u8 {
    (n.min(GAIN_MAX)) << 1
}

/// `TDM_CFG0` for 48 kHz: Linux `TAS2764_TDM_CFG0_SMP_48KHZ |
/// TAS2764_TDM_CFG0_44_1_48KHZ`. The TAS5770L script's `0x40` is bit 6, which
/// this part does not use as a sample-rate bit — it leaves the rate field at
/// zero, and the amplifier will not leave shutdown.
pub const TDM_CFG0_48K: u8 = 0x08;
/// `TDM_CFG1` for I2S `IB_IF`: RX start slot 1 (I2S 1-bit delay) and sample
/// on the falling edge. Linux `tas2764_set_fmt` for `I2S | IB_IF`.
pub const TDM_CFG1_I2S: u8 = 0x03;
/// `TDM_CFG2`: 32-bit word in a 32-bit slot, ASI source = I2C offset (slots
/// come from `TDM_CFG3`). Linux `RXW_32BITS | RXS_32BITS`. The TAS5770L `0x06`
/// is a reserved word-width on this map.
pub const TDM_CFG2_32_32: u8 = 0x0e;
/// `TDM_CFG3`: left on slot 0, right on slot 1 — Linux `set_tdm_slot` for a
/// 2-slot I2S frame. The Mac mini only plays slot 0.
pub const TDM_CFG3_I2S: u8 = 0x10;
/// `TDM_CFG4` TX rising (`TAS2764_TDM_CFG4_TX_RISING`) for `IB_IF`.
pub const TDM_CFG4_TX_RISING: u8 = 0x00;
/// `INT_CLK_CFG` bit 2: write-1-to-clear the latched fault. Power-up has to
/// drop a previous `tdm_clock_error` or the next ACTIVE request is refused
/// for a reason that is no longer true.
pub const INT_CLK_CFG_IRQZ_CLR: u8 = 1 << 2;

/// The configuration sequence, as `(register, bytes)`.
///
/// Ordered, and the order matters: the TDM block describes the stream before the
/// device is asked to play it. The gain write is deliberately the last thing
/// before power-up so it cannot be missed by a short-circuiting error path.
pub const INIT: &[(u8, &[u8])] = &[
    (reg::TDM_CFG0, &[TDM_CFG0_48K]),
    (reg::TDM_CFG1, &[TDM_CFG1_I2S]),
    (reg::TDM_CFG2, &[TDM_CFG2_32_32]),
    (reg::TDM_CFG3, &[TDM_CFG3_I2S]),
    (reg::TDM_CFG4, &[TDM_CFG4_TX_RISING]),
    (reg::DVC, &[DVC_UNITY]),
    (reg::CHNL_0, &[GAIN_MIN]),
];

/// Linux `sn012776_bop_presets` — 24 bytes at [`reg::BOP_CFG0`]. Applied
/// only for the SN012776, which is every Apple speaker this driver binds.
pub const SN012776_BOP: &[u8] = &[
    0x01, 0x32, 0x02, 0x22, 0x83, 0x2d, 0x80, 0x02, 0x06, 0x32, 0x46, 0x30, 0x02, 0x06, 0x38,
    0x40, 0x30, 0x02, 0x06, 0x3e, 0x37, 0x30, 0xff, 0xe6,
];

/// Linux `tas2764-quirks.h` sequences enabled by `ENABLED_APPLE_QUIRKS = 0x3f`,
/// as `(page, register, value)`. Page 0xfd is a hidden window; the 0x0d writes
/// there are a *second* paging step inside that window, transcribed verbatim.
pub const SN012776_QUIRKS: &[(u8, u8, u8)] = &[
    (0x00, 0x35, 0xb0), // noise gate disable (NS_CFG0)
    (0x00, 0x76, 0x00), // DAC-modulator reset when DSP is off
    (0x00, 0x6b, 0x41), // CONV_VBAT_PVDD_MODE
    (0x01, 0x33, 0x80), // undocumented TDM-adjacent
    (0x01, 0x37, 0x3a),
    (0x06, 0x14, 0x00), // Apple undocumented 0x614–0x61f
    (0x06, 0x15, 0x13),
    (0x06, 0x16, 0x52),
    (0x06, 0x17, 0x00),
    (0x06, 0x18, 0xe4),
    (0x06, 0x19, 0x0c),
    (0x06, 0x16, 0xaa),
    (0x06, 0x1b, 0x00),
    (0x06, 0x1c, 0x12),
    (0x06, 0x1d, 0xa0),
    (0x06, 0x1e, 0xd8),
    (0x06, 0x1f, 0x00),
    (0xfd, 0x0d, 0x0d), // hidden-page window
    (0xfd, 0x6c, 0x02),
    (0xfd, 0x6d, 0x0f),
    (0xfd, 0x0d, 0x00),
];

/// Whether a `PWR_CTRL` value would leave the amplifier driving the speaker.
///
/// Pure, and worth having: every path that stops audio has to end in shutdown,
/// and "left it in mute" looks identical from the host while the output stage
/// stays powered.
pub fn is_driving(pwr_ctrl: u8) -> bool {
    pwr_ctrl & 0x3 == 0
}

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;
    use crate::sound::apple::i2c::{I2c, I2cError};

    /// Read a register without writing anything — the identification step, and
    /// the only thing `/audio probe` does to the amplifier.
    pub fn read_reg(bus: &I2c, addr: u8, r: u8) -> Result<u8, I2cError> {
        bus.read_reg8(addr, r)
    }

    /// Select page 0 of book 0 and software-reset the device.
    ///
    /// **This is what a register file that acknowledges writes and ignores them
    /// needs.** The reference script omits it because it runs on a machine where
    /// nothing has touched the amplifier yet; here macOS configured this part
    /// minutes earlier and m1n1 left it as it found it, so it can be sitting on
    /// any book. The order is forced: `BOOK` is only reachable from page 0, so
    /// page comes first, then book, and only then does register `0x02` mean
    /// `MODE_CTRL` again.
    ///
    /// `settle_ms` waits after the reset — the part needs a moment before it
    /// accepts configuration, and writing into that window is indistinguishable
    /// from writing to the wrong book.
    pub fn select_and_reset(bus: &I2c, addr: u8, settle_ms: impl Fn(u64)) -> Result<(), I2cError> {
        bus.write_reg(addr, reg::PAGE, &[0])?;
        bus.write_reg(addr, reg::BOOK, &[0])?;
        bus.write_reg(addr, reg::PAGE, &[0])?;
        bus.write_reg(addr, reg::SW_RESET, &[1])?;
        settle_ms(2);
        // The reset returns the device to book 0 page 0, but say so explicitly
        // rather than assuming what a reset leaves behind.
        bus.write_reg(addr, reg::PAGE, &[0])?;
        bus.write_reg(addr, reg::BOOK, &[0])?;
        bus.write_reg(addr, reg::PAGE, &[0])?;
        Ok(())
    }

    /// Apply [`SN012776_QUIRKS`], keeping the page selected across consecutive
    /// writes. The 0xfd block is a hidden window: writing 0x0d = 0x0d opens it
    /// and the following 0x6c/0x6d stores only mean anything *while it is
    /// open*, so bouncing back to page 0 between them would discard the lot.
    fn apply_quirks(bus: &I2c, addr: u8) -> Result<(), I2cError> {
        let mut page = 0u8;
        for &(p, r, v) in SN012776_QUIRKS {
            if p != page {
                bus.write_reg(addr, reg::PAGE, &[p])?;
                page = p;
            }
            bus.write_reg(addr, r, &[v])?;
        }
        if page != 0 {
            bus.write_reg(addr, reg::PAGE, &[0])?;
        }
        Ok(())
    }

    /// Apply the configuration sequence. Leaves the device in **shutdown**;
    /// [`power_up`] is separate so the caller can start the I2S clock first.
    ///
    /// Returns whether the writes actually **took**: one register is read back
    /// and compared. A configuration that is acknowledged and discarded is the
    /// failure this part actually has, and it is invisible unless something
    /// checks.
    pub fn configure(bus: &I2c, addr: u8) -> Result<bool, I2cError> {
        bus.write_reg(addr, reg::PWR_CTRL, &[PWR_SHUTDOWN | PWR_BOP_SRC])?;
        for (r, v) in INIT {
            bus.write_reg(addr, *r, v)?;
        }
        // SN012776 brownout table, then the Apple quirk pages. Order matches
        // Linux `tas2764_codec_probe` for `DEVID_SN012776`.
        for (i, b) in SN012776_BOP.iter().enumerate() {
            let r = reg::BOP_CFG0.checked_add(i as u8).ok_or(I2cError::Timeout)?;
            bus.write_reg(addr, r, &[*b])?;
        }
        apply_quirks(bus, addr)?;
        let readback = bus.read_reg8(addr, reg::TDM_CFG0)?;
        Ok(readback == TDM_CFG0_48K)
    }

    /// Leave software shutdown, and **report whether it stayed out**.
    ///
    /// The part shuts itself back down when it is unhappy with its clock, so
    /// writing ACTIVE is a request, not a result. Returning the register as it
    /// reads afterwards is the difference between "we powered the amplifier" and
    /// "we asked, and it declined" — which are indistinguishable from the host
    /// otherwise, and produce identical silence.
    pub fn power_up(bus: &I2c, addr: u8) -> Result<u8, I2cError> {
        // Quirk writes walk off page 0. Say so before MODE_CTRL, or the
        // ACTIVE request lands on whatever page they finished on.
        bus.write_reg(addr, reg::PAGE, &[0])?;
        // Drop a latched TDM clock error from the previous attempt. Leaving
        // it set makes the next ACTIVE look like it failed for the same
        // reason even after the clocks were fixed.
        let cfg = bus.read_reg8(addr, reg::INT_CLK_CFG).unwrap_or(0x19);
        let _ = bus.write_reg(addr, reg::INT_CLK_CFG, &[cfg | INT_CLK_CFG_IRQZ_CLR]);
        bus.write_reg(addr, reg::PWR_CTRL, &[PWR_ACTIVE | PWR_BOP_SRC])?;
        bus.read_reg8(addr, reg::PWR_CTRL)
    }

    /// The latched fault register (`INT_LTCH0`), which says *why* the amplifier
    /// refused to drive: bit 2 is a TDM clock error, bit 1 over-current, bit 0
    /// over-temperature. Latched, so it survives long enough to be read.
    pub fn faults(bus: &I2c, addr: u8) -> Result<u8, I2cError> {
        bus.read_reg8(addr, 0x49)
    }

    /// Set the analog amplifier level (0..=[`GAIN_MAX`]).
    ///
    /// Deliberately a separate, explicitly-invoked function rather than part of
    /// the configuration: raising it is a decision about somebody's speaker.
    ///
    /// **The write is bracketed by shutdown.** `CHNL_0` is acknowledged and
    /// discarded while the amplifier is driving — every other register on this
    /// part took its write on the same bus in the same session, so the
    /// difference is the register, not the path. So: drop to software shutdown,
    /// write, restore whatever power state was in force. The caller sees only
    /// the read-back.
    pub fn set_gain(bus: &I2c, addr: u8, n: u8) -> Result<u8, I2cError> {
        let v = chnl_0_for_gain(n);
        let prev = bus.read_reg8(addr, reg::PWR_CTRL)?;
        bus.write_reg(addr, reg::PWR_CTRL, &[PWR_SHUTDOWN])?;
        bus.write_reg(addr, reg::CHNL_0, &[v])?;
        let got = bus.read_reg8(addr, reg::CHNL_0)?;
        bus.write_reg(addr, reg::PWR_CTRL, &[prev])?;
        Ok(got)
    }

    /// Set the digital volume (0 = unity, 0xc9 = near-mute).
    pub fn set_dvc(bus: &I2c, addr: u8, v: u8) -> Result<u8, I2cError> {
        bus.write_reg(addr, reg::DVC, &[v])?;
        bus.read_reg8(addr, reg::DVC)
    }

    /// Mute, then shut down. Both, in that order, as the reference does: muting
    /// first stops the output stage before the clock goes away.
    pub fn power_down(bus: &I2c, addr: u8) {
        let _ = bus.write_reg(addr, reg::PWR_CTRL, &[PWR_MUTE]);
        let _ = bus.write_reg(addr, reg::PWR_CTRL, &[PWR_SHUTDOWN]);
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::{configure, faults, power_down, power_up, read_reg, select_and_reset, set_dvc, set_gain};

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_gain_field_is_bits_5_to_1() {
        // Writing the level unshifted halves it and puts its low bit next to
        // CDS_MODE: a quieter, subtly different configuration rather than an
        // error. And the part's own ceiling is enforced here, not at the caller.
        assert_eq!(chnl_0_for_gain(0), 0x00);
        assert_eq!(chnl_0_for_gain(1), 0x02);
        assert_eq!(chnl_0_for_gain(10), 0x14);
        assert_eq!(chnl_0_for_gain(GAIN_MAX), GAIN_MAX << 1);
        assert_eq!(chnl_0_for_gain(255), GAIN_MAX << 1, "clamped to the part's range");
        // The level must never spill into CDS_MODE (bits 7:6).
        for n in 0..=255u8 {
            assert_eq!(chnl_0_for_gain(n) & 0xc0, 0, "gain {n} reached CDS_MODE");
        }
    }

    #[test_case]
    fn the_amplifier_comes_up_at_its_lowest_gain() {
        // This is a real amplifier into a real speaker: the sequence must never
        // contain a gain write other than the minimum. Software volume does the
        // rest, where the worst case is a quiet mistake.
        let gain: alloc::vec::Vec<_> = INIT.iter().filter(|(r, _)| *r == reg::CHNL_0).collect();
        assert_eq!(gain.len(), 1, "exactly one gain write");
        assert_eq!(gain[0].1, &[GAIN_MIN]);
    }

    #[test_case]
    fn only_the_active_power_state_drives_the_speaker() {
        assert!(is_driving(PWR_ACTIVE));
        assert!(is_driving(PWR_ACTIVE | PWR_BOP_SRC));
        assert!(!is_driving(PWR_MUTE));
        assert!(!is_driving(PWR_SHUTDOWN));
        // m1n1's TAS5770L values: 0x0c looks "active" in bits 1:0, but it is
        // not what this part accepts as a request to leave shutdown.
        assert_ne!(PWR_ACTIVE, 0x0c);
        assert_eq!(PWR_ACTIVE & 0x3, 0);
        assert_eq!(PWR_SHUTDOWN & 0x3, 2);
    }

    #[test_case]
    fn tdm_cfg0_is_the_tas2764_48khz_encoding_not_the_tas5770l_one() {
        // 0x40 is what speaker_amp.py writes for a TAS5770L. On this map bit 6
        // is unused and the sample-rate field stays zero, which is how the
        // Mac mini's amp sat in shutdown with no latched clock error.
        assert_eq!(TDM_CFG0_48K, 0x08);
        assert_ne!(TDM_CFG0_48K, 0x40);
        assert_eq!(INIT[0].0, reg::TDM_CFG0);
        assert_eq!(INIT[0].1, &[TDM_CFG0_48K]);
        assert_eq!(TDM_CFG2_32_32, 0x0e, "32-bit word in a 32-bit slot");
        assert_ne!(TDM_CFG2_32_32, 0x06, "0x06 is a reserved word-width here");
        assert_eq!(TDM_CFG1_I2S, 0x03, "I2S start slot 1, RX falling");
        assert_eq!(TDM_CFG3_I2S, 0x10, "left=0 right=1");
        assert_eq!(reg::TDM_CFG5, 0x0e, "not the TAS5770L 0x0d");
    }

    #[test_case]
    fn the_sn012776_quirk_pages_stay_grouped() {
        // apply_quirks keeps the page selected across a run. A 0xfd write
        // sandwiched between page-0 writes would close the hidden window
        // before the 0x6c/0x6d stores landed.
        assert_eq!(SN012776_BOP.len(), 24);
        let fd: alloc::vec::Vec<_> = SN012776_QUIRKS.iter().filter(|(p, _, _)| *p == 0xfd).copied().collect();
        assert_eq!(fd.len(), 4);
        let first = SN012776_QUIRKS.iter().position(|(p, _, _)| *p == 0xfd).unwrap();
        for (i, (p, _, _)) in fd.iter().enumerate() {
            assert_eq!(*p, 0xfd);
            assert_eq!(SN012776_QUIRKS[first + i].0, 0xfd);
        }
    }

    #[test_case]
    fn the_init_sequence_touches_no_register_twice() {
        // A duplicate write means one of them is dead code and the order of the
        // two decides the outcome — which is how a transcribed sequence rots.
        for (i, (r, _)) in INIT.iter().enumerate() {
            assert!(
                !INIT[i + 1..].iter().any(|(o, _)| o == r),
                "register {r:#x} written twice"
            );
        }
        // And the power register is not in it: powering up is a separate step,
        // after the clock is running.
        assert!(!INIT.iter().any(|(r, _)| *r == reg::PWR_CTRL));
    }
}
