//! **KMS — kernel mode setting.** The device-independent half of a real display
//! driver, modelled on Linux's DRM/KMS split.
//!
//! Everything above this line is firmware-framebuffer territory: the loader picks
//! a mode, hands over a linear surface, and the mode is fixed for the boot — the
//! position Linux is in with `efifb`/`simpledrm` (i.e. `nomodeset`). A *driver*
//! changes that by owning the display device itself, which is what this module
//! plus a [`DisplayDriver`] backend provides:
//!
//! * **Connectors** — one per physical output, each with its own mode list,
//!   preferred mode, and EDID. This mirrors DRM's connector object.
//! * **Mode setting** — program a mode and get back the [`Scanout`] describing the
//!   new framebuffer, which the compositor then re-inits onto.
//! * **Damage/flush** — some devices scan out of guest memory directly, others
//!   need dirty rectangles pushed to the host. The trait covers both.
//! * **Events** — a display change (resize/hotplug) is the analogue of DRM's
//!   hot-plug-detect, polled from the shell's idle pump.
//!
//! The split matters for the same reason it does in Linux: the policy (which mode,
//! which output, what scale) is written once here, and each device only has to
//! know how to enumerate and program itself.
//!
//! With no backend bound this module is inert and the compositor keeps using the
//! loader's framebuffer — so a platform we have no driver for degrades to exactly
//! today's behaviour rather than losing its console.

/// virtio-gpu wire protocol: pure encode/decode, unit-tested off hardware. The
/// transport (virtqueue over PCI) is separate, per the `agx/proto.rs` split.
pub mod virtio_gpu_proto;
/// virtio-gpu over virtio-PCI — the hardware half. Needs a device, so not
/// unit-testable; `cfg(not(test))` because it reaches the compositor and PCI.
#[cfg(not(test))]
pub mod virtio_gpu;
/// VMware SVGA II (`vmsvga`) — what VirtualBox and QEMU's `vmware-svga` present.
#[cfg(not(test))]
pub mod vmsvga;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// A display mode. Refresh is carried for reporting only — nothing here picks a
/// mode on refresh, and a device that doesn't report one uses 0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mode {
    pub w: u32,
    pub h: u32,
    /// Refresh in mHz (60000 = 60 Hz), or 0 when unknown.
    pub refresh_mhz: u32,
}

impl Mode {
    pub const fn new(w: u32, h: u32) -> Mode {
        Mode { w, h, refresh_mhz: 0 }
    }

    /// Pixel area, for "largest mode" comparisons.
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }
}

/// One physical output — DRM's connector.
#[derive(Clone, Debug)]
pub struct Connector {
    /// Device-local id, used to address this output in [`DisplayDriver::set_mode`].
    pub id: u32,
    /// Stable-ish name in the DRM style (`Virtual-1`, `SVGA-1`).
    pub name: String,
    /// Whether something is attached. A disconnected output is kept in the list
    /// (like DRM does) so a later hotplug is a state change, not a new object.
    pub connected: bool,
    /// Modes this output reports, best first.
    pub modes: Vec<Mode>,
    /// The output's preferred mode, if it declares one.
    pub preferred: Option<Mode>,
    /// Raw EDID base block, when the device exposes it.
    pub edid: Option<Vec<u8>>,
}

/// The framebuffer a mode set produced: exactly what the compositor needs to
/// start drawing, and nothing else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Scanout {
    pub addr: usize,
    pub pitch: u64,
    pub w: u32,
    pub h: u32,
    pub bpp_bytes: u64,
    pub r_shift: u32,
    pub g_shift: u32,
    pub b_shift: u32,
}

/// What a display device must be able to do. One implementation per device
/// family; the policy above is shared.
pub trait DisplayDriver: Send {
    /// Short driver name for logs and `/display` (`virtio-gpu`, `vmsvga`).
    fn name(&self) -> &'static str;

    /// Current outputs. Re-read after [`Self::poll_events`] reports a change.
    fn connectors(&mut self) -> Vec<Connector>;

    /// Program `mode` on output `connector` and return the resulting framebuffer.
    fn set_mode(&mut self, connector: u32, mode: Mode) -> Result<Scanout, &'static str>;

    /// Push a dirty rectangle to the display.
    ///
    /// A device that scans out of guest memory directly implements this as a
    /// no-op; one that keeps a host-side copy (virtio-gpu) must transfer and
    /// present. The compositor accumulates damage and calls this from the idle
    /// pump rather than per pixel.
    fn flush(&mut self, _x: u32, _y: u32, _w: u32, _h: u32) {}

    /// True when the outputs changed (resize or hotplug) since the last call —
    /// the analogue of a DRM hot-plug-detect interrupt, polled because we have no
    /// interrupt to hang it on.
    fn poll_events(&mut self) -> bool {
        false
    }
}

/// The bound driver, if any. `None` means "firmware framebuffer only", which is a
/// supported state, not a failure.
static DRIVER: crate::mm::Locked<Option<Box<dyn DisplayDriver>>> = crate::mm::Locked::new(None);

/// Probe for a supported display device and bind the first that answers.
///
/// Called once at boot, **after** PCI is up. Finding nothing is the normal case on
/// most platforms and is not an error — the compositor keeps the loader's
/// framebuffer, which is exactly today's behaviour.
#[cfg(not(test))]
pub fn probe() {
    probe_bind_only();
    adopt_console_if_needed();
}

/// Bind a driver without touching the console.
///
/// Split from [`probe`] because the two must happen at different points on
/// aarch64: PCIe (and so binding) is ready before the platform framebuffer is
/// initialised, and adopting the console in between meant the later framebuffer
/// init replaced it.
#[cfg(not(test))]
pub fn probe_bind_only() {
    if has_driver() {
        return;
    }
    // VMware SVGA II (VirtualBox, QEMU `vmware-svga`) is **detected but not bound**:
    // the driver is incomplete and binding it corrupts the display.
    //
    // SVGA II ignores the mode registers until the **FIFO** has been set up and
    // `SVGA_REG_CONFIG_DONE` written — until then the device stays in its VGA mode.
    // So a mode set appears to succeed (the registers read back what was written)
    // while the scanout keeps its old geometry, and the compositor draws at a pitch
    // the device is not using: verified against `qemu-system-x86_64 -device
    // vmware-svga`, which rendered the console four times side by side. Reporting
    // the device and declining is strictly better than a scrambled screen — the
    // firmware framebuffer still works.
    //
    // To finish it: map BAR2, program FIFO_MIN/MAX/NEXT_CMD/STOP, write
    // CONFIG_DONE, then re-test the mode set against that same QEMU device.
    if vmsvga::VmSvga::probe().is_some() {
        crate::serial_println!(
            "kms> vmsvga detected but NOT enabled (needs FIFO/CONFIG_DONE init) -- using the firmware framebuffer"
        );
        crate::ktrace::log("kms", "vmsvga present, driver incomplete: not bound");
    }
    if let Some(g) = virtio_gpu::VirtioGpu::probe() {
        bind(Box::new(g));
        return;
    }
    crate::ktrace::log("kms", "no display driver (firmware framebuffer only)");
}

/// If nothing has given us a framebuffer yet, this device *is* the display: set its
/// preferred mode so the console has somewhere to draw.
///
/// On a machine that also has a firmware framebuffer the mode is left alone — the
/// loader already picked one, and changing it at boot would be a surprise.
#[cfg(not(test))]
pub fn adopt_console_if_needed() {
    if crate::framebuffer::physical_size().is_some() {
        return;
    }
    let cs = connectors();
    let Some(m) = cs
        .iter()
        .find(|c| c.connected)
        .or_else(|| cs.first())
        .and_then(|c| c.preferred.or_else(|| c.modes.first().copied()))
    else {
        return;
    };
    let drv = driver_name().unwrap_or("?");
    match set_mode((m.w, m.h)) {
        Some(got) => crate::serial_println!("kms> console up on {drv} at {}x{}", got.w, got.h),
        None => crate::serial_println!("kms> {drv} present but the mode set failed"),
    }
}

/// Bind `driver` as the display driver. Called once, from probe.
pub fn bind(driver: Box<dyn DisplayDriver>) {
    let name = driver.name();
    DRIVER.with(|d| *d = Some(driver));
    crate::ktrace::log("kms", "driver bound");
    crate::serial_println!("kms> display driver: {name} (real mode setting available)");
}

/// The bound driver's name, or `None` when running on the loader's framebuffer.
pub fn driver_name() -> Option<&'static str> {
    DRIVER.with(|d| d.as_ref().map(|x| x.name()))
}

/// Whether a real mode set is possible. When false, `/display set` letterboxes
/// instead — the documented `nomodeset` equivalent.
pub fn has_driver() -> bool {
    DRIVER.with(|d| d.is_some())
}

/// The outputs the driver reports (empty without a driver).
pub fn connectors() -> Vec<Connector> {
    DRIVER.with(|d| d.as_mut().map(|x| x.connectors()).unwrap_or_default())
}

/// Modes available on the first connected output, best first.
///
/// This is what `/display list` should show once a driver is bound: the modes the
/// *display* reports, rather than a generic table of sizes that merely fit.
pub fn modes() -> Vec<Mode> {
    let cs = connectors();
    cs.iter()
        .find(|c| c.connected)
        .or_else(|| cs.first())
        .map(|c| c.modes.clone())
        .unwrap_or_default()
}

/// Choose the mode to honour a `(w, h)` request from a connector's list.
///
/// Exact match wins; otherwise the largest mode that fits inside the request, so
/// a request is never satisfied by something bigger than asked for. `None` when
/// the list is empty or nothing fits — the caller then leaves the mode alone.
pub fn match_mode(modes: &[Mode], want: (u32, u32)) -> Option<Mode> {
    if let Some(m) = modes.iter().find(|m| m.w == want.0 && m.h == want.1) {
        return Some(*m);
    }
    modes
        .iter()
        .filter(|m| m.w <= want.0 && m.h <= want.1)
        .max_by_key(|m| m.area())
        .copied()
}

/// Program a mode on the first connected output and hand the new framebuffer to
/// the compositor.
///
/// Returns the mode actually set. `None` without a driver, or when the device
/// refuses — in both cases nothing has changed and the caller can fall back.
pub fn set_mode(want: (u32, u32)) -> Option<Mode> {
    let picked = DRIVER.with(|d| {
        let drv = d.as_mut()?;
        let cs = drv.connectors();
        let c = cs.iter().find(|c| c.connected).or_else(|| cs.first())?;
        let mode = match_mode(&c.modes, want).or(c.preferred)?;
        let scanout = drv.set_mode(c.id, mode).ok()?;
        Some((mode, scanout))
    })?;
    let (mode, s) = picked;
    // Re-init the console onto the new surface. Done outside the driver lock: the
    // compositor's repaint path can call back into `flush`, and re-entering the
    // lock would deadlock.
    //
    // `framebuffer` is `cfg(not(test))`, so the compositor hand-off is gated — the
    // policy above stays testable, which is the whole reason it lives here.
    #[cfg(not(test))]
    crate::framebuffer::reinit_scanout(s.addr, s.w as u64, s.h as u64, s.pitch, s.bpp_bytes, s.r_shift, s.g_shift, s.b_shift);
    let _ = s;
    damage_all();
    Some(mode)
}

/// Accumulated dirty rectangle, in scanout pixels: `(x0, y0, x1, y1)`.
static DAMAGE: crate::mm::Locked<Option<(u32, u32, u32, u32)>> = crate::mm::Locked::new(None);

/// Mark a region dirty. Cheap and lock-local — safe to call from paint paths.
///
/// Damage is *accumulated* rather than flushed here because a redraw touches the
/// screen in hundreds of small pieces; presenting each one would be a queue round
/// trip per glyph.
pub fn damage(x: u32, y: u32, w: u32, h: u32) {
    if w == 0 || h == 0 || !has_driver() {
        return;
    }
    DAMAGE.with(|d| {
        let r = (x, y, x + w, y + h);
        *d = Some(match *d {
            None => r,
            Some(o) => (o.0.min(r.0), o.1.min(r.1), o.2.max(r.2), o.3.max(r.3)),
        });
    });
}

/// Mark the whole screen dirty (a mode set, a full redraw).
pub fn damage_all() {
    #[cfg(not(test))]
    if let Some((w, h)) = crate::framebuffer::physical_size() {
        damage(0, 0, w, h);
    }
}

/// Present accumulated damage, if any. Called from the shell's idle pump, so the
/// cost is one flush per tick rather than one per draw call.
pub fn flush_damage() {
    let Some((x0, y0, x1, y1)) = DAMAGE.with(|d| d.take()) else { return };
    DRIVER.with(|d| {
        if let Some(drv) = d.as_mut() {
            drv.flush(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0));
        }
    });
}

/// Poll the driver for a display change (resize/hotplug) — DRM's HPD, polled.
/// Returns true when the outputs changed and the caller should re-apply policy.
pub fn poll_events() -> bool {
    DRIVER.with(|d| d.as_mut().map(|x| x.poll_events()).unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes() -> Vec<Mode> {
        alloc::vec![
            Mode::new(2560, 1440),
            Mode::new(1920, 1080),
            Mode::new(1280, 720),
            Mode::new(1024, 768),
        ]
    }

    #[test_case]
    fn match_mode_prefers_exact_then_largest_that_fits() {
        let m = modes();
        assert_eq!(match_mode(&m, (1920, 1080)), Some(Mode::new(1920, 1080)));
        // No exact match → the largest that fits *inside* the request, never bigger.
        assert_eq!(match_mode(&m, (2000, 1200)), Some(Mode::new(1920, 1080)));
        assert_eq!(match_mode(&m, (1300, 800)), Some(Mode::new(1280, 720)));
        // Nothing fits → leave the mode alone rather than pick something wrong.
        assert_eq!(match_mode(&m, (640, 480)), None);
        assert_eq!(match_mode(&[], (1920, 1080)), None);
    }

    #[test_case]
    fn match_mode_never_returns_something_larger_than_asked() {
        let m = modes();
        for &(w, h) in &[(800u32, 600u32), (1024, 768), (1280, 720), (1920, 1080), (4096, 2160)] {
            if let Some(got) = match_mode(&m, (w, h)) {
                assert!(got.w <= w && got.h <= h, "asked {w}x{h}, got {}x{}", got.w, got.h);
            }
        }
    }

    #[test_case]
    fn mode_area_orders_by_pixels() {
        assert!(Mode::new(1920, 1080).area() > Mode::new(1280, 720).area());
        assert_eq!(Mode::new(1920, 1080).area(), 1920 * 1080);
        assert_eq!(Mode::new(0, 0).area(), 0);
    }

    #[test_case]
    fn no_driver_is_a_supported_state_not_a_failure() {
        // The whole module must be inert without a backend: this is the
        // firmware-framebuffer path, which has to keep working.
        assert!(!has_driver());
        assert_eq!(driver_name(), None);
        assert!(connectors().is_empty());
        assert_eq!(set_mode((1920, 1080)), None);
        assert!(!poll_events());
        // Damage accumulation and flushing are no-ops, not panics.
        damage(0, 0, 100, 100);
        flush_damage();
    }
}
