//! **VirtualBox Guest Additions transport (VMMDev)** — identification and the
//! request mechanism, staged deliberately.
//!
//! VirtualBox's clipboard *and* its shared folders both ride HGCM, which rides
//! this device: PCI `80ee:cafe`, whose whole protocol is "write the physical
//! address of a request structure to one register, and the host fills the
//! structure in synchronously".
//!
//! What exists here is the device, the register, and the request framing —
//! with every offset pinned by a test against the layouts in VirtualBox's own
//! `include/VBox/VMMDev.h`, **fetched, not recalled**. What does *not* exist is
//! HGCM and the two services on top of it, and that is the same staging
//! [`super::wifi::iwl`] uses for the same reason: those are large
//! per-service protocols, no emulator here presents this device, and code
//! written from memory would look complete, send well-formed garbage to a real
//! hypervisor and report success.
//!
//! ## The two transports, and why the driver must discover which
//!
//! The request register is reachable two ways, and they are **not** a
//! per-architecture constant:
//!
//! * **BAR0** is 32 bytes of **PCI I/O space** — the original path, and the
//!   only one on x86.
//! * **BAR3** is a 4 KiB **MMIO** window with the *same* register offsets
//!   (`VMMDEV_MMIO_OFF_REQUEST` is 0, exactly like
//!   `VMMDEV_PORT_OFF_REQUEST`), added for hosts with no I/O ports — which is
//!   what an ARM guest needs.
//!
//! But BAR3 is created only when the device's `MmioReq` option is set
//! (`VMMDev.cpp`, `if (pThis->fMmioReq)`), so its presence is a property of the
//! VM, not of the architecture. Hardcoding "I/O on x86, MMIO on ARM" would be
//! wrong on any VM configured the other way, so both are probed and whichever
//! is present is used. Note this is the *opposite* of the VMSVGA situation
//! documented in `kms/vmsvga.rs`, where the MMIO register layout was a guess:
//! here the offsets are published and identical between the two windows.
//!
//! ## The 32-bit address constraint is real
//!
//! The request register takes a **32-bit physical address**, so every request
//! buffer must live below 4 GiB. On a VM with more RAM than that, an ordinary
//! allocation can land above the line — and the failure mode is not an error,
//! it is the host reading a truncated address and processing whatever happens
//! to be there. [`request_addr_fits`] is the check, and bring-up refuses
//! rather than truncating.

use crate::mm::Locked;
use alloc::string::String;

/// PCI identity of the VirtualBox Guest Service device.
pub const VENDOR: u16 = 0x80ee;
pub const DEVICE: u16 = 0xcafe;

/// Register offsets, identical in the I/O (BAR0) and MMIO (BAR3) windows.
pub const OFF_REQUEST: u64 = 0;
pub const OFF_REQUEST_FAST: u64 = 8;

/// `VMMDEV_VERSION` — the version every request header must carry.
pub const VMMDEV_VERSION: u32 = 0x0001_0004;

/// `VMMDevRequestHeader`: `size, version, requestType, rc, reserved1,
/// fRequestor` — six 32-bit fields, 24 bytes.
pub const REQ_HDR: usize = 24;
pub const HDR_SIZE: usize = 0;
pub const HDR_VERSION: usize = 4;
pub const HDR_TYPE: usize = 8;
pub const HDR_RC: usize = 12;
pub const HDR_RESERVED1: usize = 16;
pub const HDR_REQUESTOR: usize = 20;

// --- request types (VMMDevRequestType) ---
pub const REQ_GET_HOST_VERSION: u32 = 4;
pub const REQ_REPORT_GUEST_INFO: u32 = 50;
pub const REQ_REPORT_GUEST_INFO2: u32 = 58;
pub const REQ_HGCM_CONNECT: u32 = 60;
pub const REQ_HGCM_DISCONNECT: u32 = 61;
pub const REQ_HGCM_CALL32: u32 = 62;
pub const REQ_HGCM_CALL64: u32 = 63;

/// `VMMDevHGCMRequestHeader` adds `fu32Flags` and `result` — 32 bytes total.
pub const HGCM_HDR: usize = REQ_HDR + 8;
/// `VBOX_HGCM_REQ_DONE` — bit 0 of `fu32Flags`, set by the host when an
/// asynchronous HGCM request has completed.
pub const HGCM_REQ_DONE: u32 = 1;

/// `VMMDevReportGuestInfo` body: `additionsVersion`, `osType`.
pub const GUEST_INFO_LEN: usize = REQ_HDR + 8;
/// The additions version we claim. Reported so the host knows a guest agent is
/// present at all; it gates whether the host offers HGCM services.
pub const ADDITIONS_VERSION: u32 = 0x0006_0000; // 6.0
/// `VBOXOSTYPE_Linux26_x64` / `_Linux26` — "a modern Linux-ish guest", which is
/// the closest published value and the one that unlocks the most services. The
/// host uses it for presentation and service gating, not for behaviour we
/// depend on.
pub const OS_TYPE_LINUX26: u32 = 0x5_3100;

/// One page of `.bss`, page-aligned, used as the request buffer on aarch64.
///
/// See [`request_buffer`] for why this is not a heap allocation.
#[repr(C, align(4096))]
struct ReqPage([u8; 4096]);
static mut REQ_PAGE: ReqPage = ReqPage([0; 4096]);

/// `(phys, virt)` of a request buffer the 32-bit request register can reach.
///
/// **The obvious `alloc_dma` does not work here, and it fails on a real VM.**
/// aarch64 places its heap at the *top of discovered RAM*, so on a guest with
/// more than 4 GiB every allocation is out of the register's reach — an 8 GiB
/// VirtualBox-ARM guest reported exactly `request buffer landed above 4 GiB`
/// and `/vbox up` could not make its first request. The kernel image is loaded
/// low by the firmware and RAM is identity-mapped, so a page of `.bss` is both
/// below the line and its own physical address.
///
/// x86 frame-allocates and *can* express the constraint, so it asks for it
/// directly; the caller's [`request_addr_fits`] check still has the last word
/// on either arch.
fn request_buffer() -> Option<(u64, u64)> {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: taking the address of a static; no reference is created, and
        // this driver is the only user of the page.
        let p = core::ptr::addr_of_mut!(REQ_PAGE) as u64;
        Some((p, p))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // A single 4 KiB page can never cross a 4 KiB boundary, so the boundary
        // constraint is satisfied trivially.
        crate::mm::alloc_dma_bounded(4096, u32::MAX as u64, 4096)
    }
}

/// Whether a request buffer's physical address fits the 32-bit register.
///
/// A truncated address is not an error the host can report — it would read a
/// request structure from whatever lives at the low 32 bits — so this is
/// checked before any request is submitted.
pub fn request_addr_fits(phys: u64) -> bool {
    phys <= u32::MAX as u64
}

/// Fill a `VMMDevRequestHeader` into `buf`. Returns false if `buf` is too
/// small for the declared size.
pub fn write_header(buf: &mut [u8], size: u32, req_type: u32) -> bool {
    if buf.len() < REQ_HDR || (size as usize) > buf.len() {
        return false;
    }
    buf[HDR_SIZE..HDR_SIZE + 4].copy_from_slice(&size.to_le_bytes());
    buf[HDR_VERSION..HDR_VERSION + 4].copy_from_slice(&VMMDEV_VERSION.to_le_bytes());
    buf[HDR_TYPE..HDR_TYPE + 4].copy_from_slice(&req_type.to_le_bytes());
    // `rc` is an OUT field; the host overwrites it. Seeded with a value that
    // is not a success code so a request the host never touched cannot read as
    // having succeeded.
    buf[HDR_RC..HDR_RC + 4].copy_from_slice(&(-1i32).to_le_bytes());
    buf[HDR_RESERVED1..HDR_RESERVED1 + 4].copy_from_slice(&0u32.to_le_bytes());
    buf[HDR_REQUESTOR..HDR_REQUESTOR + 4].copy_from_slice(&0u32.to_le_bytes());
    true
}

/// Read the `rc` a completed request came back with. VirtualBox status codes
/// are >= 0 on success.
pub fn header_rc(buf: &[u8]) -> i32 {
    if buf.len() < REQ_HDR {
        return -1;
    }
    i32::from_le_bytes([buf[HDR_RC], buf[HDR_RC + 1], buf[HDR_RC + 2], buf[HDR_RC + 3]])
}

/// How the request register is reached on this VM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// BAR0, PCI I/O space (x86 only — there are no I/O instructions on ARM).
    Port(u16),
    /// BAR3, a 4 KiB MMIO window with the same register offsets.
    Mmio(u64),
}

/// What was found on the bus.
pub struct VBoxGuest {
    pub window: Window,
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    /// Host version string, once [`bring_up`] has read it.
    pub host_version: Option<String>,
}

static DEV: Locked<Option<VBoxGuest>> = Locked::new(None);

#[cfg(target_arch = "aarch64")]
use crate::pci;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci;

/// Locate the VMMDev device, **reading only**.
///
/// Identification never writes: the same rule the I2C and EC drivers follow,
/// and the one the VMSVGA driver had to learn after a probe-time write moved a
/// live display's scanout.
pub fn probe() -> bool {
    let mut found: Option<pci::PciDevice> = None;
    pci::for_each(&mut |d: pci::PciDevice| {
        if d.vendor == VENDOR && d.device == DEVICE {
            found = Some(d);
            return false;
        }
        true
    });
    let Some(d) = found else {
        return false;
    };

    // BAR3 is an MMIO window created only when the VM sets the device's
    // MmioReq option, so its presence is a property of the VM rather than of
    // the architecture — probe, do not assume.
    let bar3 = d.bar(3);
    // BAR0 is I/O space, which `PciDevice::bar` deliberately reports as 0
    // (every other driver here wants memory BARs), so read it raw.
    let bar0_raw = pci::read32(d.bus, d.dev, d.func, 0x10);
    let io_port = if bar0_raw & 1 != 0 { (bar0_raw & 0xffff_fffc) as u16 } else { 0 };

    let window = if bar3 != 0 {
        Window::Mmio(crate::mm::map_mmio(bar3, 0x1000))
    } else if io_port != 0 && cfg!(target_arch = "x86_64") {
        Window::Port(io_port)
    } else {
        // An ARM guest whose VM did not enable MmioReq has no way to reach the
        // register: there is no I/O instruction. Say which case this is rather
        // than reporting "no VirtualBox device", which is a different fact.
        crate::ktrace::log(
            "vbox",
            "VMMDev found but unreachable: no MMIO BAR and no I/O ports on this arch \
             (enable the device's MmioReq option)",
        );
        return false;
    };

    crate::ktrace::log_fmt(format_args!(
        "vbox: VMMDev at {:02x}:{:02x}.{} via {}",
        d.bus,
        d.dev,
        d.func,
        match window {
            Window::Port(p) => alloc::format!("I/O port {p:#06x}"),
            Window::Mmio(a) => alloc::format!("MMIO {a:#x}"),
        }
    ));
    DEV.with(|s| {
        *s = Some(VBoxGuest { window, bus: d.bus, dev: d.dev, func: d.func, host_version: None })
    });
    true
}

/// Whether a VMMDev device was found and is reachable.
pub fn present() -> bool {
    DEV.with(|d| d.is_some())
}

/// Submit a request whose buffer is at `phys` (and already framed).
///
/// The whole protocol: write the request's physical address to the register.
/// The host processes it synchronously and writes its answer back into the
/// same buffer, so there is nothing to wait for on this path — only HGCM calls
/// are asynchronous, and they signal through `fu32Flags`.
fn submit(phys: u64) -> bool {
    if !request_addr_fits(phys) {
        crate::ktrace::log_fmt(format_args!(
            "vbox: request buffer at {phys:#x} is above 4 GiB; the register takes 32 bits"
        ));
        return false;
    }
    DEV.with(|d| {
        let Some(g) = d.as_ref() else {
            return false;
        };
        match g.window {
            #[cfg(target_arch = "x86_64")]
            Window::Port(p) => {
                // SAFETY: `p` is this device's I/O BAR; the register is 32-bit.
                unsafe { crate::arch::x86_64::port::outl(p + OFF_REQUEST as u16, phys as u32) };
                true
            }
            #[cfg(not(target_arch = "x86_64"))]
            Window::Port(_) => false,
            Window::Mmio(base) => {
                // SAFETY: `base` is this device's mapped MMIO BAR; the request
                // register is 32-bit at offset 0.
                unsafe {
                    core::ptr::write_volatile((base + OFF_REQUEST) as *mut u32, phys as u32)
                };
                true
            }
        }
    })
}

/// Announce this guest to the host and read its version back.
///
/// **Command-gated, never automatic at boot** — the same posture `/wifi up`
/// takes. This is the first thing here that *writes* to a real hypervisor's
/// device, and an untested driver should not touch one just because the
/// machine started.
pub fn bring_up() -> Result<String, &'static str> {
    if !present() && !probe() {
        return Err("no VirtualBox VMMDev device on this machine");
    }
    let Some((phys, virt)) = request_buffer() else {
        return Err("could not allocate a request buffer below 4 GiB");
    };
    if !request_addr_fits(phys) {
        return Err("request buffer landed above 4 GiB (the register takes a 32-bit address)");
    }
    // SAFETY: a 4 KiB DMA region this function owns for the call's duration.
    let buf = unsafe { core::slice::from_raw_parts_mut(virt as *mut u8, 4096) };

    // 1. Tell the host a guest agent exists. Until this lands the host offers
    //    no HGCM services at all, so it is the gate on everything above.
    buf[..GUEST_INFO_LEN].fill(0);
    if !write_header(buf, GUEST_INFO_LEN as u32, REQ_REPORT_GUEST_INFO) {
        return Err("could not frame the guest-info request");
    }
    buf[REQ_HDR..REQ_HDR + 4].copy_from_slice(&ADDITIONS_VERSION.to_le_bytes());
    buf[REQ_HDR + 4..REQ_HDR + 8].copy_from_slice(&OS_TYPE_LINUX26.to_le_bytes());
    if !submit(phys) {
        return Err("the request register is unreachable");
    }
    let rc = header_rc(buf);
    if rc < 0 {
        crate::ktrace::log_fmt(format_args!("vbox: ReportGuestInfo failed rc={rc}"));
        return Err("the host rejected our guest-info report");
    }

    // 2. Read something back, which is what proves the transport works in both
    //    directions rather than merely that a write was accepted.
    const HOST_VERSION_LEN: usize = REQ_HDR + 12; // major, minor, build
    buf[..HOST_VERSION_LEN].fill(0);
    if !write_header(buf, HOST_VERSION_LEN as u32, REQ_GET_HOST_VERSION) {
        return Err("could not frame the host-version request");
    }
    if !submit(phys) {
        return Err("the request register is unreachable");
    }
    let rc = header_rc(buf);
    if rc < 0 {
        return Err("the host would not report its version");
    }
    let rd = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    let v = alloc::format!("{}.{}.{}", rd(REQ_HDR), rd(REQ_HDR + 4), rd(REQ_HDR + 8));
    DEV.with(|d| {
        if let Some(g) = d.as_mut() {
            g.host_version = Some(v.clone());
        }
    });
    crate::ktrace::log_fmt(format_args!("vbox: host version {v}; guest info accepted"));
    Ok(v)
}

/// One line for `/vbox`.
pub fn status() -> String {
    DEV.with(|d| match d.as_ref() {
        None => "no VirtualBox VMMDev device found (not running under VirtualBox?)".into(),
        Some(g) => {
            let w = match g.window {
                Window::Port(p) => alloc::format!("I/O port {p:#06x}"),
                Window::Mmio(a) => alloc::format!("MMIO {a:#x}"),
            };
            match &g.host_version {
                Some(v) => alloc::format!(
                    "VMMDev at {:02x}:{:02x}.{} via {w}; host {v}. HGCM (clipboard, shared \
                     folders) is NOT implemented yet.",
                    g.bus, g.dev, g.func
                ),
                None => alloc::format!(
                    "VMMDev at {:02x}:{:02x}.{} via {w}; not brought up (try /vbox up)",
                    g.bus, g.dev, g.func
                ),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_request_header_is_six_words_in_this_order() {
        // size, version, requestType, rc, reserved1, fRequestor. Every field is
        // a plain u32, so a wrong offset yields a valid-looking request rather
        // than an error — the host would act on whatever landed in `type`.
        assert_eq!(REQ_HDR, 24);
        assert_eq!((HDR_SIZE, HDR_VERSION, HDR_TYPE), (0, 4, 8));
        assert_eq!((HDR_RC, HDR_RESERVED1, HDR_REQUESTOR), (12, 16, 20));
        // The HGCM header adds fu32Flags + result on top.
        assert_eq!(HGCM_HDR, 32);

        let mut buf = [0u8; 64];
        assert!(write_header(&mut buf, GUEST_INFO_LEN as u32, REQ_REPORT_GUEST_INFO));
        assert_eq!(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 32);
        assert_eq!(u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]), VMMDEV_VERSION);
        assert_eq!(u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]), REQ_REPORT_GUEST_INFO);
        // `rc` is seeded non-success, so a request the host never touched
        // cannot be read as having succeeded.
        assert!(header_rc(&buf) < 0);
    }

    #[test_case]
    fn a_request_that_does_not_fit_its_buffer_is_refused() {
        let mut small = [0u8; 16];
        assert!(!write_header(&mut small, 24, REQ_GET_HOST_VERSION));
        let mut buf = [0u8; 32];
        // A declared size beyond the buffer is refused rather than handing the
        // host a length that runs past what we allocated.
        assert!(!write_header(&mut buf, 64, REQ_GET_HOST_VERSION));
        assert!(write_header(&mut buf, 32, REQ_GET_HOST_VERSION));
    }

    #[test_case]
    fn the_request_register_takes_a_32_bit_address() {
        // A guest with more than 4 GiB of RAM can allocate above the line, and
        // a truncated address is not an error the host can report — it would
        // read a request from whatever lives at the low 32 bits.
        assert!(request_addr_fits(0x4000_0000));
        assert!(request_addr_fits(u32::MAX as u64));
        assert!(!request_addr_fits(u32::MAX as u64 + 1));
        assert!(!request_addr_fits(0x1_8000_0000));
    }

    #[test_case]
    fn both_windows_use_the_same_register_offsets() {
        // VMMDEV_PORT_OFF_REQUEST and VMMDEV_MMIO_OFF_REQUEST are both 0, which
        // is what lets one submit path serve the I/O and MMIO transports. This
        // is the opposite of the VMSVGA case, where the MMIO layout was a guess.
        assert_eq!(OFF_REQUEST, 0);
        assert_eq!(OFF_REQUEST_FAST, 8);
    }

    #[test_case]
    fn the_pci_identity_is_the_guest_service_device() {
        assert_eq!((VENDOR, DEVICE), (0x80ee, 0xcafe));
    }
}
