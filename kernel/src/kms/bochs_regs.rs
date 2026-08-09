//! Bochs VBE ("dispi") register layout and the pure logic over it.
//!
//! The interface QEMU's **standard VGA** presents (`-vga std`, PCI `1234:1111`),
//! which is also what Bochs and several other emulators implement. Linux drives
//! it with `drivers/gpu/drm/tiny/bochs.c`; this is the same device.
//!
//! Split out of [`super::bochs`] for the reason `virtio_gpu_proto` is split out
//! of `virtio_gpu`: the hardware module is `cfg(not(test))`, so a `#[test_case]`
//! written inside it is never compiled. Everything here is pure and therefore
//! actually tested — the register offsets, the identification rule, the VRAM
//! arithmetic, and above all the **order** of a mode set.
//!
//! ## Why this device matters more than the two backends we already have
//!
//! virtio-gpu and VMSVGA both have to be asked for. `1234:1111` is what a plain
//! `qemu-system-x86_64 -M q35` hands you — including from our own RUN.md, which
//! names no display device. Without a driver for it the compositor keeps the
//! loader's framebuffer and `/display` can only letterbox a *smaller* desktop
//! inside it, which is precisely Linux's `nomodeset` position.

/// Dispi register indices. 16-bit registers, addressed by index.
pub const INDEX_ID: u16 = 0x0;
pub const INDEX_XRES: u16 = 0x1;
pub const INDEX_YRES: u16 = 0x2;
pub const INDEX_BPP: u16 = 0x3;
pub const INDEX_ENABLE: u16 = 0x4;
pub const INDEX_BANK: u16 = 0x5;
pub const INDEX_VIRT_WIDTH: u16 = 0x6;
pub const INDEX_VIRT_HEIGHT: u16 = 0x7;
pub const INDEX_X_OFFSET: u16 = 0x8;
pub const INDEX_Y_OFFSET: u16 = 0x9;
pub const INDEX_VIDEO_MEMORY_64K: u16 = 0xa;

/// `ENABLE` bits.
pub const ENABLE_DISABLED: u16 = 0x00;
pub const ENABLE_ENABLED: u16 = 0x01;
/// Scan out of the linear framebuffer (BAR0) rather than the banked VGA window.
pub const ENABLE_LFB: u16 = 0x40;

/// Lowest version id the device reports; the low nibble is the revision.
pub const ID0: u16 = 0xB0C0;

/// x86 I/O port pair, for a device whose MMIO BAR is absent.
pub const IOPORT_INDEX: u16 = 0x01CE;
pub const IOPORT_DATA: u16 = 0x01CF;

/// The dispi registers inside the MMIO BAR, as a byte offset. Each register is
/// 16 bits, so the index is doubled.
pub const MMIO_DISPI_BASE: u64 = 0x500;
/// QEMU publishes the display's EDID at the **base** of the MMIO window when
/// `edid=on`, not up with the registers.
///
/// This was first written as `0x600` from memory and read nothing on any
/// configuration, including `edid=on,xres=…,yres=…` where QEMU certainly does
/// publish one. `0x600` is the **QEXT** region (Linux reads `qext_size` from
/// exactly there); the EDID lives at offset 0, in the space below `0x400` where
/// the VGA registers begin — `bochs_get_edid_block` reads `mmio + block * 128`
/// and refuses anything reaching `0x400`. Taken from
/// `drivers/gpu/drm/tiny/bochs.c`, fetched rather than recalled, because recalling
/// it is what produced the wrong offset in the first place.
pub const MMIO_EDID_BASE: u64 = 0x0;
/// How much of the MMIO window this driver maps. Everything it touches — EDID at
/// 0, dispi registers at 0x500 — is inside the first page.
pub const MMIO_WINDOW_LEN: u64 = 0x1000;

/// PCI ids. Deliberately **only** QEMU/Bochs std VGA.
///
/// VirtualBox's adapter (`80ee:beef`) also implements these registers, but this
/// driver does not claim it: getting a real VirtualBox display wrong is exactly
/// what the VMSVGA backend already did once, and that cost a working console
/// rather than a feature. Claim it only after verifying on that target.
pub const VENDOR_QEMU: u16 = 0x1234;
pub const DEVICE_STDVGA: u16 = 0x1111;

/// Byte offset of dispi register `index` within the MMIO BAR.
pub const fn mmio_offset(index: u16) -> u64 {
    MMIO_DISPI_BASE + (index as u64) * 2
}

/// Whether an `INDEX_ID` readback identifies a Bochs VBE device.
///
/// The low nibble is a revision that varies by emulator and version, so only the
/// top twelve bits identify the family. Comparing the whole word rejects every
/// device except the one revision that was hard-coded.
pub const fn is_bochs_id(id: u16) -> bool {
    (id & 0xfff0) == ID0
}

/// VRAM size from `INDEX_VIDEO_MEMORY_64K`, in bytes.
pub const fn vram_bytes(mem_64k: u16) -> u64 {
    (mem_64k as u64) * 64 * 1024
}

/// Bytes per scanline for a packed (unpadded) framebuffer.
pub const fn pitch_bytes(w: u32, bpp_bytes: u32) -> u64 {
    (w as u64) * (bpp_bytes as u64)
}

/// Whether a mode's framebuffer fits in `vram`.
///
/// The device does not refuse an oversized mode — it will happily program
/// geometry whose framebuffer runs off the end of VRAM, and the result is a
/// scanout reading memory that is not there. Checking is the driver's job.
pub const fn mode_fits(w: u32, h: u32, bpp_bytes: u32, vram: u64) -> bool {
    if w == 0 || h == 0 {
        return false;
    }
    // The registers are 16-bit, so a larger dimension cannot be expressed at all.
    if w > u16::MAX as u32 || h > u16::MAX as u32 {
        return false;
    }
    match pitch_bytes(w, bpp_bytes).checked_mul(h as u64) {
        Some(need) => need <= vram,
        None => false,
    }
}

/// The register writes a mode set consists of, **in order**.
///
/// Returned as data rather than performed inline so the ordering is testable:
/// the rule that bit everyone who has written one of these is that geometry
/// written to an *enabled* device is ignored or latched inconsistently, leaving
/// the scanout on the previous configuration while the registers read back the
/// new one. VMSVGA in this same tree produced a console drawn four times side by
/// side that way. So `ENABLE` is cleared first and set last, and a test pins it.
///
/// `VIRT_WIDTH` is the stride **in pixels**, not bytes.
pub fn modeset_sequence(w: u32, h: u32, bpp: u16) -> [(u16, u16); 10] {
    [
        (INDEX_ENABLE, ENABLE_DISABLED),
        (INDEX_BPP, bpp),
        (INDEX_XRES, w as u16),
        (INDEX_YRES, h as u16),
        (INDEX_BANK, 0),
        (INDEX_VIRT_WIDTH, w as u16),
        (INDEX_VIRT_HEIGHT, h as u16),
        (INDEX_X_OFFSET, 0),
        (INDEX_Y_OFFSET, 0),
        (INDEX_ENABLE, ENABLE_ENABLED | ENABLE_LFB),
    ]
}

/// The modes to advertise: everything standard that fits in VRAM, largest first,
/// with `preferred` promoted to the front when it is itself usable.
pub fn usable_modes(vram: u64, bpp_bytes: u32, preferred: Option<(u32, u32)>) -> alloc::vec::Vec<(u32, u32)> {
    let mut out = alloc::vec::Vec::new();
    if let Some(p) = preferred {
        if mode_fits(p.0, p.1, bpp_bytes, vram) {
            out.push(p);
        }
    }
    for &(w, h) in crate::display::STANDARD_MODES {
        if Some((w, h)) != preferred && mode_fits(w, h, bpp_bytes, vram) {
            out.push((w, h));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Register i lives at 0x500 + 2i. Off-by-one here reads a neighbouring
    /// register, which returns a plausible number rather than an error — an
    /// XRES read landing on BPP yields 32, a believable width for nothing.
    #[test_case]
    fn dispi_registers_are_16_bit_slots_from_0x500() {
        assert_eq!(mmio_offset(INDEX_ID), 0x500);
        assert_eq!(mmio_offset(INDEX_XRES), 0x502);
        assert_eq!(mmio_offset(INDEX_YRES), 0x504);
        assert_eq!(mmio_offset(INDEX_BPP), 0x506);
        assert_eq!(mmio_offset(INDEX_ENABLE), 0x508);
        assert_eq!(mmio_offset(INDEX_VIDEO_MEMORY_64K), 0x514);
    }

    /// **EDID is below the registers, not above them.** It occupies the window
    /// from 0 up to the VGA register block at 0x400; the dispi registers start
    /// at 0x500 and 0x600 is QEXT. Writing this from memory put EDID at 0x600,
    /// which reads QEXT's size word — a small integer that fails the EDID header
    /// check, so it silently looked like "this device publishes no EDID" on every
    /// configuration rather than like a wrong address.
    #[test_case]
    fn edid_lives_below_the_register_block() {
        assert_eq!(MMIO_EDID_BASE, 0);
        // A whole base block fits below the VGA register window at 0x400.
        assert!(MMIO_EDID_BASE + crate::edid::BASE_BLOCK_LEN as u64 <= 0x400);
        // And it must not collide with the dispi registers or QEXT.
        assert!(MMIO_EDID_BASE + crate::edid::BASE_BLOCK_LEN as u64 <= MMIO_DISPI_BASE);
        assert_eq!(MMIO_DISPI_BASE, 0x500);
        // Everything touched is inside the single page mapped.
        assert!(mmio_offset(INDEX_VIDEO_MEMORY_64K) < MMIO_WINDOW_LEN);
    }

    /// Only the top twelve bits identify the family: the low nibble is a
    /// revision that differs between emulators and releases, so an exact
    /// comparison binds to one build of one emulator.
    #[test_case]
    fn identification_ignores_the_revision_nibble() {
        for rev in 0..=0xf {
            assert!(is_bochs_id(ID0 | rev), "B0C{rev:X} is a Bochs id");
        }
        assert!(!is_bochs_id(0x0000), "an unclaimed register reads 0");
        assert!(!is_bochs_id(0xffff), "a floating bus reads all-ones");
        assert!(!is_bochs_id(0xB1C0));
    }

    /// **`ENABLE` must bracket the geometry writes.** Programming an enabled
    /// device leaves it scanning out the previous configuration while the
    /// registers read back the new one — the failure that made VMSVGA draw the
    /// console four times across.
    #[test_case]
    fn a_mode_set_disables_first_and_enables_last() {
        let seq = modeset_sequence(1920, 1080, 32);
        assert_eq!(seq[0], (INDEX_ENABLE, ENABLE_DISABLED), "must disable first");
        let last = seq[seq.len() - 1];
        assert_eq!(last.0, INDEX_ENABLE);
        assert_eq!(last.1, ENABLE_ENABLED | ENABLE_LFB, "and enable the LFB last");
        // Nothing may touch ENABLE in between, or the bracket is not a bracket.
        for w in &seq[1..seq.len() - 1] {
            assert_ne!(w.0, INDEX_ENABLE);
        }
        // Every geometry register is written while disabled.
        let mid: alloc::vec::Vec<u16> = seq[1..seq.len() - 1].iter().map(|w| w.0).collect();
        for r in [INDEX_XRES, INDEX_YRES, INDEX_BPP, INDEX_VIRT_WIDTH] {
            assert!(mid.contains(&r), "register {r:#x} must be set inside the bracket");
        }
    }

    /// The stride register counts **pixels**, not bytes. Writing a byte count
    /// gives a scanout striding four times too far: a picture squeezed into the
    /// left quarter with three columns of garbage, which reads as a corrupt
    /// framebuffer rather than a wrong register.
    #[test_case]
    fn virt_width_is_a_pixel_stride() {
        let seq = modeset_sequence(1280, 800, 32);
        let vw = seq.iter().find(|w| w.0 == INDEX_VIRT_WIDTH).unwrap().1;
        assert_eq!(vw, 1280);
        assert_ne!(vw as u64, pitch_bytes(1280, 4), "not the byte pitch");
    }

    /// A mode whose framebuffer exceeds VRAM must be refused. The device
    /// programs it regardless and then scans out memory that is not there.
    #[test_case]
    fn a_mode_must_fit_in_vram() {
        let vram = vram_bytes(256); // QEMU's default 16 MiB
        assert_eq!(vram, 16 * 1024 * 1024);
        assert!(mode_fits(1920, 1080, 4, vram), "1080p needs 8.3 MiB");
        assert!(!mode_fits(3840, 2160, 4, vram), "4K needs 33 MiB");
        assert!(mode_fits(3840, 2160, 4, vram_bytes(1024)), "…and fits in 64 MiB");
        assert!(!mode_fits(0, 1080, 4, vram));
        assert!(!mode_fits(1920, 0, 4, vram));
    }

    /// The registers are 16 bits, so a dimension above 65535 cannot be
    /// expressed — and truncating it would program a *valid-looking* small mode
    /// rather than failing.
    #[test_case]
    fn a_dimension_too_large_for_the_register_is_refused() {
        assert!(!mode_fits(70_000, 100, 4, u64::MAX));
        assert!(!mode_fits(100, 70_000, 4, u64::MAX));
        // And the truncation that would otherwise happen is visibly wrong.
        assert_eq!(70_000u32 as u16, 4464);
    }

    /// The advertised list is VRAM-bounded and leads with the preferred mode.
    #[test_case]
    fn usable_modes_are_bounded_by_vram_and_lead_with_preferred() {
        let modes = usable_modes(vram_bytes(256), 4, Some((1920, 1080)));
        assert_eq!(modes[0], (1920, 1080), "preferred first");
        assert_eq!(modes.iter().filter(|&&m| m == (1920, 1080)).count(), 1, "not duplicated");
        assert!(!modes.contains(&(3840, 2160)), "4K does not fit 16 MiB");
        assert!(modes.contains(&(1280, 720)));

        // A preferred mode that does not fit is dropped, not promoted.
        let small = usable_modes(vram_bytes(4), 4, Some((3840, 2160)));
        assert!(!small.contains(&(3840, 2160)));
    }

    /// A machine reporting no VRAM must advertise nothing rather than every
    /// mode — `mode_fits` against 0 is what keeps the list honest.
    #[test_case]
    fn no_vram_advertises_no_modes() {
        assert!(usable_modes(0, 4, None).is_empty());
        assert!(usable_modes(0, 4, Some((640, 480))).is_empty());
    }
}
