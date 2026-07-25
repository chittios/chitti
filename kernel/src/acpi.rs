//! Minimal ACPI table walking — just enough to discover, from firmware tables
//! (not hardcoded per-machine addresses), the PCIe ECAM base (**MCFG**) and the
//! console UART base (**SPCR**). The stub captures the RSDP from the UEFI
//! configuration table and hands it to the kernel; from there: RSDP → XSDT →
//! {MCFG, SPCR}. This is what lets aarch64 run on platforms whose device map
//! differs from QEMU `virt` (e.g. VirtualBox puts the PL011 at 0xFFDDF000, not
//! QEMU's 0x09000000).
//!
//! Read-only, no allocation, no interpreter — table headers + one field each.

/// Make an ACPI table at physical address `phys` readable, returning the address
/// to read it at.
///
/// **x86 must map these pages explicitly.** Limine's HHDM covers *usable RAM*,
/// and ACPI tables live in firmware-reserved regions outside it — so both the raw
/// physical address and its HHDM translation are unmapped, and reading either is a
/// page fault that halts the boot rather than a garbage read a signature check
/// could reject. (Both faults happened: `0xf52e0`, then `0xffff8000000f52e0`.)
///
/// aarch64 keeps its previous behaviour exactly: it runs on a flat identity map
/// where the tables are already reachable, and `init_uart` reads SPCR before the
/// frame allocator exists, so calling into `mm` there would be worse than useless.
#[inline]
fn map_table(phys: u64, len: usize) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        crate::mm::map_mmio(phys, len.max(0x1000))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = len;
        phys
    }
}

fn le16(p: *const u8, o: usize) -> u16 {
    unsafe { u16::from_le_bytes([*p.add(o), *p.add(o + 1)]) }
}
fn le32(p: *const u8, o: usize) -> u32 {
    unsafe { u32::from_le_bytes([*p.add(o), *p.add(o + 1), *p.add(o + 2), *p.add(o + 3)]) }
}
fn le64(p: *const u8, o: usize) -> u64 {
    let mut v = [0u8; 8];
    for (i, b) in v.iter_mut().enumerate() {
        *b = unsafe { *p.add(o + i) };
    }
    u64::from_le_bytes(v)
}

/// A PCIe ECAM segment discovered from MCFG.
#[derive(Clone, Copy)]
pub struct EcamSegment {
    pub base: u64,
    pub bus_start: u8,
    pub bus_end: u8,
}

/// Find the ACPI table with 4-byte signature `sig` by walking `rsdp` → XSDT
/// (or RSDT), returning a pointer to its header. Every address is identity-
/// mapped physical (the caller ensures the map covers the tables, which live in
/// low reserved RAM). `None` if no valid ACPI or no such table.
fn find_table(rsdp: u64, sig: &[u8; 4]) -> Option<*const u8> {
    if rsdp == 0 {
        return None;
    }
    let r = rsdp as *const u8;
    // RSDP: "RSD PTR " signature; revision @15; XsdtAddress @24 (8 bytes).
    let mut rsig = [0u8; 8];
    for (i, b) in rsig.iter_mut().enumerate() {
        *b = unsafe { *r.add(i) };
    }
    if &rsig != b"RSD PTR " {
        return None;
    }
    let revision = unsafe { *r.add(15) };
    // XSDT (rev>=2) preferred; else RSDT (32-bit entries).
    let (table_ptr, entry_size) = if revision >= 2 { (le64(r, 24), 8usize) } else { (le32(r, 16) as u64, 4usize) };
    if table_ptr == 0 {
        return None;
    }
    // Map a page to read the header's length, then remap covering the whole table
    // (an XSDT with many entries can exceed one page).
    let t = map_table(table_ptr, 0x1000) as *const u8;
    let t = map_table(table_ptr, le32(t, 4) as usize) as *const u8;
    // System description table header: length @4 (u32). Entries follow the
    // 36-byte header, each a pointer to another table.
    let len = le32(t, 4) as usize;
    let n = len.saturating_sub(36) / entry_size;
    for i in 0..n {
        let off = 36 + i * entry_size;
        let tbl = if entry_size == 8 { le64(t, off) } else { le32(t, off) as u64 };
        if tbl == 0 {
            continue;
        }
        // Each entry points at another table in reserved memory: map before reading.
        let p = map_table(tbl, 0x1000) as *const u8;
        let mut s = [0u8; 4];
        for (k, b) in s.iter_mut().enumerate() {
            *b = unsafe { *p.add(k) };
        }
        if &s == sig {
            let len = le32(p, 4) as usize;
            return Some(map_table(tbl, len) as *const u8);
        }
    }
    None
}

/// Walk `rsdp` → XSDT → MCFG and return the first ECAM segment, if any.
pub fn ecam_from_rsdp(rsdp: u64) -> Option<EcamSegment> {
    let p = find_table(rsdp, b"MCFG")?;
    // MCFG: 44-byte header (incl. 8 reserved), then allocation entries of 16
    // bytes: base@0 (u64), segment@8 (u16), bus_start@10, bus_end@11.
    let base = le64(p, 44);
    let bus_start = unsafe { *p.add(44 + 10) };
    let bus_end = unsafe { *p.add(44 + 11) };
    let _ = le16(p, 44 + 8);
    Some(EcamSegment { base, bus_start, bus_end })
}

/// Walk `rsdp` -> XSDT -> **HPET** and return the event-timer block's base
/// address, if the platform has one.
///
/// The HPET is the reference clock used to calibrate the local-APIC timer on a
/// machine whose legacy PIT is absent or non-functional — increasingly common on
/// UEFI-only hardware, where the 8254 may simply not be wired up.
pub fn hpet_from_rsdp(rsdp: u64) -> Option<u64> {
    let p = find_table(rsdp, b"HPET")?;
    // HPET table: 36-byte header, event-timer-block id @36 (u32), then a 12-byte
    // Generic Address Structure @40 whose address field is at +4.
    let space = unsafe { *p.add(40) };
    let base = le64(p, 44);
    // Only a memory-mapped (space 0) block is usable.
    if space != 0 || base == 0 {
        return None;
    }
    Some(base)
}

/// The DSDT named by the FADT, mapped for reading.
///
/// Exposed because the DSDT is where devices that cannot be enumerated are described
/// — the I2C connections in [`i2c_resources`], and eventually anything an AML
/// interpreter would evaluate.
pub fn dsdt_from_rsdp(rsdp: u64, map: impl Fn(u64, usize) -> u64) -> Option<*const u8> {
    let f = find_table(rsdp, b"FACP")?;
    let len = le32(f, 4) as usize;
    let dsdt_phys = if len >= 156 && le64(f, 148) != 0 { le64(f, 148) } else { le32(f, 40) as u64 };
    // Refuse an implausible physical address instead of mapping it. A value with bits
    // above the platform's physical-address width produces a page-table entry with
    // reserved bits set, and the resulting fault (error 0x8) is far harder to read
    // than this check is to write. 1 TiB is well past any real DSDT.
    if dsdt_phys == 0 || dsdt_phys >= (1 << 40) {
        return None;
    }
    // Map the header, learn the real length, then map exactly that much. Mapping a
    // fixed guess and scanning to the declared length reads past the mapping on any
    // table larger than the guess.
    let hdr = map(dsdt_phys, 0x1000) as *const u8;
    let total = le32(hdr, 4) as usize;
    if total < 36 || total > 0x40_0000 {
        return None;
    }
    Some(map(dsdt_phys, total) as *const u8)
}

/// An **I2C serial-bus connection** described by an ACPI resource descriptor.
///
/// This is how an I2C-attached device — notably a HID-over-I2C touchpad — is
/// located. Such a device cannot be probed for: unlike PCI it has no enumerable
/// identity, so its bus and 7-bit address come from its `_CRS`. Blind-scanning a
/// laptop's I2C bus is not an acceptable substitute, because the same controller
/// commonly hosts the embedded controller and the USB-C PD controller, and writing
/// to the wrong address can misconfigure real hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cResource {
    /// 7-bit (or 10-bit) target address.
    pub address: u16,
    /// Bus speed in Hz, as declared (typically 100_000 or 400_000).
    pub speed_hz: u32,
    /// True when the descriptor marks this device as the *controller* rather than
    /// the target; such an entry does not describe something we can talk to.
    pub controller_mode: bool,
}

/// ACPI large-resource tag for a Serial Bus connection.
const RES_SERIAL_BUS: u8 = 0x8e;
/// Serial-bus type 1 = I2C (2 = SPI, 3 = UART).
const SERIAL_BUS_I2C: u8 = 0x01;

/// Parse one **I2C Serial Bus** resource descriptor from the start of `d`.
///
/// Layout (ACPI 6.x, Large Resource 0x8E): tag, 16-bit length, revision, resource
/// source index, bus type, general flags, 16-bit type flags, type revision, 16-bit
/// type-data length, then for I2C a 32-bit connection speed and a 16-bit target
/// address.
///
/// `None` if `d` is not an I2C serial-bus descriptor or is truncated — this parses
/// firmware tables, so a short or foreign buffer must be refused rather than read
/// past.
pub fn parse_i2c_serial_bus(d: &[u8]) -> Option<I2cResource> {
    if d.len() < 18 || d[0] != RES_SERIAL_BUS {
        return None;
    }
    // Declared length covers everything after the 3-byte header.
    let len = u16::from_le_bytes([d[1], d[2]]) as usize;
    if 3 + len > d.len() || len < 15 {
        return None;
    }
    if d[5] != SERIAL_BUS_I2C {
        return None;
    }
    // General flags bit 1: 0 = this device is the target ("slave"), 1 = controller.
    let controller_mode = d[6] & 0x02 != 0;
    let speed_hz = u32::from_le_bytes([d[12], d[13], d[14], d[15]]);
    let address = u16::from_le_bytes([d[16], d[17]]);
    Some(I2cResource { address, speed_hz, controller_mode })
}

/// Every I2C connection described anywhere in `dsdt`.
///
/// Scans for Serial Bus descriptors rather than walking the ACPI namespace, because
/// `_CRS` is in practice a static `ResourceTemplate` buffer — a fixed blob in the
/// table — so it can be found without evaluating AML. That is what makes an
/// I2C-attached device reachable before there is an AML interpreter.
///
/// **What this cannot do:** it does not associate a connection with the device that
/// declared it, so it cannot tell a touchpad from a sensor. The caller must confirm
/// identity over the bus (a HID-over-I2C device answers a HID descriptor read at its
/// descriptor register), and must not write blindly to an address just because it
/// appears here.
pub fn i2c_resources(dsdt: *const u8) -> alloc::vec::Vec<I2cResource> {
    let mut out = alloc::vec::Vec::new();
    // SAFETY: `dsdt` points at a mapped ACPI table whose length is at offset 4.
    let len = le32(dsdt, 4) as usize;
    if len < 36 || len > 0x40_0000 {
        return out;
    }
    let byte = |i: usize| -> u8 { unsafe { *dsdt.add(i) } };
    let mut i = 36;
    while i + 18 <= len {
        if byte(i) == RES_SERIAL_BUS {
            // Copy the candidate window out before parsing, so the parser works on
            // a bounded slice rather than a raw pointer.
            let want = (u16::from_le_bytes([byte(i + 1), byte(i + 2)]) as usize) + 3;
            if want >= 18 && i + want <= len {
                let mut buf = alloc::vec::Vec::with_capacity(want);
                for k in 0..want {
                    buf.push(byte(i + k));
                }
                if let Some(r) = parse_i2c_serial_bus(&buf) {
                    if !out.contains(&r) {
                        out.push(r);
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// Everything needed to drive an ACPI **S5 (soft-off)** transition.
///
/// The values come from two different tables, which is the awkward part: the
/// FADT gives the *register* (`PM1a_CNT_BLK`), but the *value* to write
/// (`SLP_TYPa`) lives in the DSDT's `\_S5_` package and is normally obtained by
/// evaluating AML.
#[derive(Clone, Copy, Debug)]
pub struct SleepInfo {
    /// `PM1a_CNT` I/O port (system-I/O space; the only space this implements).
    pub pm1a_cnt: u16,
    /// `PM1b_CNT`, or 0 when the platform has no second block.
    pub pm1b_cnt: u16,
    /// Sleep type for S5, from the DSDT `\_S5_` package.
    pub slp_typa: u8,
    pub slp_typb: u8,
    /// `SMI_CMD` port and the `ACPI_ENABLE` value, for taking ownership from
    /// firmware when `SCI_EN` is clear (legacy BIOS boots; UEFI boots hand ACPI
    /// over already enabled).
    pub smi_cmd: u32,
    pub acpi_enable: u8,
}

/// `PM1x_CNT` bit 13: start the sleep transition.
pub const SLP_EN: u16 = 1 << 13;
/// `PM1x_CNT` bit 0: ACPI mode is enabled (SCI_EN).
pub const SCI_EN: u16 = 1 << 0;

/// Locate the RSDP, validating the signature rather than trusting the caller's
/// idea of what kind of address it has.
///
/// `candidates` are tried in order — a Limine RSDP response is a physical address
/// on newer revisions and an HHDM-relative virtual one on older builds, so the
/// caller passes both interpretations and this picks whichever actually has
/// `"RSD PTR "` at it. Returns the usable (mapped) address.
pub fn find_rsdp(candidates: &[u64]) -> Option<u64> {
    for &c in candidates {
        if c == 0 {
            continue;
        }
        let p = c as *const u8;
        let mut sig = [0u8; 8];
        for (i, b) in sig.iter_mut().enumerate() {
            // SAFETY: reading 8 bytes at a candidate RSDP address. Callers only
            // pass addresses the bootloader reported or the legacy BIOS window,
            // both of which are mapped.
            *b = unsafe { *p.add(i) };
        }
        if &sig == b"RSD PTR " {
            return Some(c);
        }
    }
    None
}

/// Walk `rsdp` -> XSDT -> **FADT** ("FACP") and the DSDT it names, and return the
/// S5 sleep parameters. `None` if either table is missing or the DSDT has no
/// `\_S5_` package (a machine that genuinely cannot soft-off this way).
///
/// `phys_to_virt` maps a physical table address for reading — identity on the
/// aarch64 map, HHDM on x86.
pub fn s5_from_rsdp(rsdp: u64, phys_to_virt: impl Fn(u64) -> u64) -> Option<SleepInfo> {
    let f = find_table(rsdp, b"FACP")?;
    // FADT: SMI_CMD@48, ACPI_ENABLE@52, PM1a_CNT_BLK@64, PM1b_CNT_BLK@68,
    // DSDT@40, X_DSDT@148, X_PM1a_CNT_BLK@180 (a 12-byte Generic Address
    // Structure: space@0, width@1, offset@2, access@3, address@4).
    let len = le32(f, 4) as usize;
    let smi_cmd = le32(f, 48);
    let acpi_enable = unsafe { *f.add(52) };
    let mut pm1a = le32(f, 64) as u64;
    let mut pm1b = le32(f, 68) as u64;
    // Prefer the extended GAS forms when present and in system-I/O space (1).
    if len >= 192 {
        let space = unsafe { *f.add(180) };
        let addr = le64(f, 184);
        if addr != 0 && space == 1 {
            pm1a = addr;
        }
        let space_b = unsafe { *f.add(192) };
        let addr_b = le64(f, 196);
        if len >= 204 && addr_b != 0 && space_b == 1 {
            pm1b = addr_b;
        }
    }
    if pm1a == 0 || pm1a > u16::MAX as u64 {
        return None; // not an I/O-port PM1a_CNT — memory-mapped PM blocks unsupported
    }
    // DSDT, for the `\_S5_` package.
    let dsdt_phys = if len >= 156 && le64(f, 148) != 0 { le64(f, 148) } else { le32(f, 40) as u64 };
    if dsdt_phys == 0 {
        return None;
    }
    let dsdt = phys_to_virt(dsdt_phys) as *const u8;
    let (slp_typa, slp_typb) = parse_s5_package(dsdt)?;
    Some(SleepInfo {
        pm1a_cnt: pm1a as u16,
        pm1b_cnt: if pm1b <= u16::MAX as u64 { pm1b as u16 } else { 0 },
        slp_typa,
        slp_typb,
        smi_cmd,
        acpi_enable,
    })
}

/// Find `\_S5_` in the DSDT and decode the two sleep-type values out of its
/// package, **without** an AML interpreter.
///
/// This is the well-worn bytecode-scan shortcut: locate the `_S5_` name, confirm
/// it is introduced by a `NameOp` (0x08, optionally preceded by a root-scope
/// backslash), then step over `PackageOp` (0x12), its `PkgLength` (whose leading
/// two bits give the encoded byte count) and the element count, and read the
/// first two elements. Each element is `ZeroOp` (0x00), `OneOp` (0x01) or a
/// `BytePrefix` (0x0A) followed by the value.
///
/// It is a heuristic, not evaluation: a DSDT that computes `_S5_` dynamically, or
/// hides it behind a method, will not match — and then soft-off simply reports
/// unavailable rather than writing a wrong sleep type.
fn parse_s5_package(dsdt: *const u8) -> Option<(u8, u8)> {
    // SAFETY: `dsdt` points at a mapped ACPI table whose own header declares its
    // length at offset 4; every read below stays inside that length.
    let len = le32(dsdt, 4) as usize;
    if len < 36 || len > 0x40_0000 {
        return None; // implausible: not a DSDT we should scan
    }
    let byte = |i: usize| -> u8 { unsafe { *dsdt.add(i) } };
    let mut i = 36; // skip the table header
    while i + 4 < len {
        if byte(i) == b'_' && byte(i + 1) == b'S' && byte(i + 2) == b'5' && byte(i + 3) == b'_' {
            // Must be a NameOp definition: `08 '_S5_'`, or `08 '\' '_S5_'`.
            let named = (i >= 1 && byte(i - 1) == 0x08)
                || (i >= 2 && byte(i - 2) == 0x08 && byte(i - 1) == b'\\');
            if !named {
                i += 1;
                continue;
            }
            let mut p = i + 4; // just past the name
            if p >= len || byte(p) != 0x12 {
                i += 1;
                continue; // not a package after all
            }
            p += 1; // PackageOp
            if p >= len {
                return None;
            }
            // PkgLength: the top two bits say how many extra length bytes follow.
            p += ((byte(p) >> 6) as usize) + 1;
            if p >= len {
                return None;
            }
            p += 1; // NumElements
            // Decode up to two elements as sleep types.
            let mut vals = [0u8; 2];
            for slot in vals.iter_mut() {
                if p >= len {
                    break;
                }
                match byte(p) {
                    0x00 => {
                        *slot = 0;
                        p += 1;
                    }
                    0x01 => {
                        *slot = 1;
                        p += 1;
                    }
                    0x0a => {
                        if p + 1 >= len {
                            break;
                        }
                        *slot = byte(p + 1);
                        p += 2;
                    }
                    _ => break, // something we don't decode; keep what we have
                }
            }
            return Some((vals[0], vals[1]));
        }
        i += 1;
    }
    None
}

/// GICv3 register bases discovered from ACPI — for an ARM platform booted by UEFI
/// with **no device tree** (VirtualBox-ARM, UTM, real SBSA servers/laptops).
pub struct GicInfo {
    /// Distributor base, from the MADT GICD structure (type 0x0D).
    pub gicd: u64,
    /// This CPU's redistributor base.
    pub gicr: u64,
    /// GIC architecture version reported by the GICD structure (3 = GICv3).
    pub version: u8,
}

/// Walk `rsdp` → XSDT → **MADT** ("APIC") and return the GICv3 distributor and
/// redistributor bases for the CPU whose `MPIDR_EL1` is `mpidr`.
///
/// This is the ARM-SBSA counterpart to reading the device tree's `arm,gic-v3`
/// node. Without it a UEFI-booted ARM machine has no way to locate the GIC, and
/// hardcoding QEMU `virt`'s `0x0800_0000` would poke whatever happens to live
/// there on a real server.
///
/// Redistributor resolution follows the ACPI spec's own precedence:
/// 1. the **GICC** structure (type 0x0C) whose `MPIDR` field matches this CPU —
///    its `GicrBaseAddress` is exactly this core's redistributor frame;
/// 2. failing an MPIDR match, the first GICC with a non-zero `GicrBaseAddress`;
/// 3. failing that, the base of the **GICR** discovery range (type 0x0F), whose
///    first frame belongs to the boot CPU on every platform that uses this form.
///
/// `None` if there is no MADT, no GICD, or no redistributor by any of those
/// routes — the caller then stays on cooperative scheduling rather than guessing.
pub fn gic_from_rsdp(rsdp: u64, mpidr: u64) -> Option<GicInfo> {
    let p = find_table(rsdp, b"APIC")?;
    let total = le32(p, 4) as usize;
    // MADT header is 36 bytes + LocalApicAddress(u32) + Flags(u32); the
    // interrupt-controller structures start at 44. Each is {type u8, length u8}.
    let mut off = 44usize;
    let mut gicd = 0u64;
    let mut version = 0u8;
    let mut gicr_matched = 0u64; // GICC with our MPIDR
    let mut gicr_first = 0u64; // first GICC with a redistributor
    let mut gicr_range = 0u64; // GICR discovery range base
    while off + 2 <= total {
        // SAFETY: `p` points at a mapped ACPI table; `off + 2 <= total` and
        // `total` is the table's own declared length.
        let (ty, len) = unsafe { (*p.add(off), *p.add(off + 1) as usize) };
        if len < 2 || off + len > total {
            break; // malformed length would otherwise loop or read past the table
        }
        match ty {
            0x0c => {
                // GICC: GicrBaseAddress @64, MPIDR @72 (both u64).
                if len >= 80 {
                    let base = le64(p, off + 64);
                    // MPIDR comparison uses the affinity bits only (bits 63:40
                    // are reserved/flags in MPIDR_EL1).
                    if base != 0 {
                        if le64(p, off + 72) & 0xff_00ff_ffff == mpidr & 0xff_00ff_ffff {
                            gicr_matched = base;
                        }
                        if gicr_first == 0 {
                            gicr_first = base;
                        }
                    }
                }
            }
            0x0d => {
                // GICD: PhysicalBaseAddress @8 (u64), GicVersion @20 (u8).
                if len >= 24 {
                    gicd = le64(p, off + 8);
                    // SAFETY: `off + 20 < off + len <= total`, inside the table.
                    version = unsafe { *p.add(off + 20) };
                }
            }
            0x0f => {
                // GICR: DiscoveryRangeBaseAddress @4 (u64).
                if len >= 16 && gicr_range == 0 {
                    gicr_range = le64(p, off + 4);
                }
            }
            _ => {}
        }
        off += len;
    }
    let gicr = if gicr_matched != 0 {
        gicr_matched
    } else if gicr_first != 0 {
        gicr_first
    } else {
        gicr_range
    };
    if gicd == 0 || gicr == 0 {
        return None;
    }
    Some(GicInfo { gicd, gicr, version })
}

/// Walk `rsdp` → XSDT → **SPCR** (Serial Port Console Redirection) and return
/// the console UART's `(base_address, interface_type)`. Interface type 0x03 is
/// ARM PL011; 0x0e is ARM SBSA (PL011-compatible DR/FR layout). This is how the
/// UART base is found on a platform whose map differs from QEMU `virt` (e.g.
/// VirtualBox's PL011 at 0xFFDDF000). `None` if no SPCR or a non-MMIO address.
pub fn uart_from_rsdp(rsdp: u64) -> Option<(u64, u8)> {
    let p = find_table(rsdp, b"SPCR")?;
    // SPCR: 36-byte header, then Interface Type @36 (1 byte), 3 reserved, then a
    // 12-byte Generic Address Structure @40: AddressSpaceId@0 (0 = system
    // memory), ..., Address@4 (8 bytes).
    let iface = unsafe { *p.add(36) };
    let addr_space = unsafe { *p.add(40) };
    let base = le64(p, 44);
    // Only a memory-mapped (addr_space 0) UART with a non-zero base is usable.
    if addr_space != 0 || base == 0 {
        return None;
    }
    Some((base, iface))
}

#[cfg(test)]
mod tests {
    use super::*;
    // --- I2C serial-bus resources ----------------------------------------

    /// Build an I2C Serial Bus descriptor for `addr` at `speed`.
    fn i2c_desc(addr: u16, speed: u32, controller: bool) -> alloc::vec::Vec<u8> {
        let mut d = alloc::vec::Vec::new();
        d.push(0x8e); // Serial Bus large resource
        d.extend_from_slice(&15u16.to_le_bytes()); // length after the 3-byte header
        d.push(1); // revision
        d.push(0); // resource source index
        d.push(0x01); // bus type: I2C
        d.push(if controller { 0x02 } else { 0x00 }); // general flags
        d.extend_from_slice(&0u16.to_le_bytes()); // type flags
        d.push(1); // type revision
        d.extend_from_slice(&6u16.to_le_bytes()); // type data length
        d.extend_from_slice(&speed.to_le_bytes());
        d.extend_from_slice(&addr.to_le_bytes());
        d
    }

    #[test_case]
    fn parses_an_i2c_touchpad_connection() {
        // 0x2c at 400 kHz is the shape a HID-over-I2C touchpad declares.
        let d = i2c_desc(0x2c, 400_000, false);
        let r = parse_i2c_serial_bus(&d).unwrap();
        assert_eq!(r.address, 0x2c);
        assert_eq!(r.speed_hz, 400_000);
        assert!(!r.controller_mode);
    }

    #[test_case]
    fn distinguishes_controller_mode_from_a_target() {
        // General-flags bit 1 marks the descriptor as describing the controller,
        // which is not something we can talk to — treating it as a device address
        // would mean writing to the bus master itself.
        let r = parse_i2c_serial_bus(&i2c_desc(0x2c, 100_000, true)).unwrap();
        assert!(r.controller_mode);
    }

    #[test_case]
    fn refuses_non_i2c_and_truncated_descriptors() {
        // SPI (bus type 2) must not be read as I2C: the type-specific payload
        // differs, so the "address" would be garbage.
        let mut spi = i2c_desc(0x2c, 100_000, false);
        spi[5] = 0x02;
        assert_eq!(parse_i2c_serial_bus(&spi), None);
        // A different large-resource tag entirely.
        let mut other = i2c_desc(0x2c, 100_000, false);
        other[0] = 0x87;
        assert_eq!(parse_i2c_serial_bus(&other), None);
        // Truncated: these are firmware bytes, so a short buffer must be refused
        // rather than read past.
        let d = i2c_desc(0x2c, 100_000, false);
        for cut in 0..18 {
            assert_eq!(parse_i2c_serial_bus(&d[..cut]), None, "cut {cut}");
        }
        // Declared length longer than the buffer.
        let mut lying = i2c_desc(0x2c, 100_000, false);
        lying[1] = 0xff;
        assert_eq!(parse_i2c_serial_bus(&lying), None);
    }

    #[test_case]
    fn scans_a_dsdt_for_every_distinct_connection() {
        // Two devices at different addresses, plus a duplicate that must collapse.
        let mut body = alloc::vec::Vec::new();
        body.extend_from_slice(&[0x08, b'X']); // filler
        body.extend_from_slice(&i2c_desc(0x2c, 400_000, false));
        body.extend_from_slice(&[0x08, b'Y']);
        body.extend_from_slice(&i2c_desc(0x15, 100_000, false));
        body.extend_from_slice(&i2c_desc(0x2c, 400_000, false)); // duplicate
        let t = dsdt(&body);
        let got = i2c_resources(t.as_ptr());
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got.iter().any(|r| r.address == 0x2c && r.speed_hz == 400_000));
        assert!(got.iter().any(|r| r.address == 0x15 && r.speed_hz == 100_000));
    }

    #[test_case]
    fn scan_of_a_table_with_no_i2c_finds_nothing() {
        assert!(i2c_resources(dsdt(&[]).as_ptr()).is_empty());
        // A plausible-looking table body with no serial-bus descriptors.
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x02, 0x0a, 0x05]);
        assert!(i2c_resources(t.as_ptr()).is_empty());
    }

    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a minimal fake DSDT: 36-byte header (with a correct length field)
    /// followed by `body` bytes.
    fn dsdt(body: &[u8]) -> Vec<u8> {
        let mut t = vec![0u8; 36];
        t[0..4].copy_from_slice(b"DSDT");
        t.extend_from_slice(body);
        let len = t.len() as u32;
        t[4..8].copy_from_slice(&len.to_le_bytes());
        t
    }

    #[test_case]
    fn s5_package_byte_prefix_values() {
        // NameOp '_S5_' PackageOp PkgLen NumElem BytePrefix 5 BytePrefix 5
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x02, 0x0a, 0x05, 0x0a, 0x05]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((5, 5)));
    }

    #[test_case]
    fn s5_package_root_scoped_and_zero_one_ops() {
        // The other common encoding: `08 '\' '_S5_'`, elements ZeroOp / OneOp.
        let t = dsdt(&[0x08, b'\\', b'_', b'S', b'5', b'_', 0x12, 0x05, 0x02, 0x00, 0x01]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((0, 1)));
    }

    #[test_case]
    fn s5_package_multibyte_pkglength() {
        // PkgLength with the top bits set encodes extra length bytes, which must
        // be stepped over or the element read lands on the wrong byte.
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x41, 0x00, 0x02, 0x0a, 0x07, 0x0a, 0x07]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((7, 7)));
    }

    #[test_case]
    fn s5_name_without_nameop_is_not_matched() {
        // `_S5_` appearing as a plain string / method reference must NOT be
        // decoded — writing a sleep type read out of unrelated bytes is worse
        // than reporting soft-off unavailable.
        let t = dsdt(&[0xff, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x02, 0x0a, 0x05, 0x0a, 0x05]);
        assert_eq!(parse_s5_package(t.as_ptr()), None);
    }

    #[test_case]
    fn s5_absent_or_truncated_returns_none() {
        assert_eq!(parse_s5_package(dsdt(&[]).as_ptr()), None);
        // NameOp + name but the package is cut off at the table end.
        assert_eq!(parse_s5_package(dsdt(&[0x08, b'_', b'S', b'5', b'_']).as_ptr()), None);
        // Name present, but followed by something that isn't a PackageOp.
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x14, 0x06, 0x02]);
        assert_eq!(parse_s5_package(t.as_ptr()), None);
    }

    #[test_case]
    fn find_rsdp_picks_the_candidate_with_the_signature() {
        let good: [u8; 8] = *b"RSD PTR ";
        let bad: [u8; 8] = *b"NOPENOPE";
        let g = good.as_ptr() as u64;
        let b = bad.as_ptr() as u64;
        // Order matters: the first *valid* candidate wins, junk is skipped.
        assert_eq!(find_rsdp(&[b, g]), Some(g));
        assert_eq!(find_rsdp(&[g, b]), Some(g));
        assert_eq!(find_rsdp(&[0, b]), None);
        assert_eq!(find_rsdp(&[]), None);
    }
}
