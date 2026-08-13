//! The aarch64 half of Apple built-in audio: device-tree discovery, the
//! bring-up order, and the [`SndDevice`] the rest of the OS plays through.
//!
//! Split from the parent module so everything there stays arch-neutral and
//! testable — this file is MMIO and ordering, which only hardware can check.

use super::{admac, i2c, mca, nco, pmgr, tas2764};
use super::{BITS_PER_FRAME, RATE};
use crate::mm::Locked;
use crate::sound::SndDevice;
use alloc::boxed::Box;
use alloc::string::String;

/// Everything the tree has to tell us before a single register is written.
struct Topology {
    i2c_base: u64,
    i2c_size: u64,
    amp_addr: u8,
    amp_gpio: Option<(u64, u64, u32)>, // (pinctrl base, size, pin)
    nco_base: u64,
    nco_size: u64,
    nco_fin: u64,
    mca_clusters: u64,
    mca_clusters_size: u64,
    mca_switch: u64,
    mca_switch_size: u64,
    admac_base: u64,
    admac_size: u64,
    dart_base: u64,
    dart_phandle: u32,
    dart_stream: u32,
}

/// Read the whole audio topology out of the device tree.
///
/// Every one of these is discovered rather than assumed, and the failure is
/// named: "this machine has no `apple,mca`" and "the amplifier is not on the bus"
/// are different problems, and a single "audio init failed" would hide which.
fn discover() -> Result<Topology, &'static str> {
    let dtb = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
    unsafe {
        let (mca_clusters, mca_clusters_size) =
            crate::fdt::reg_of_compatible(dtb, b"apple,mca").ok_or("no apple,mca node")?;
        let (mca_switch, mca_switch_size) =
            crate::fdt::reg_of_nth_node(dtb, b"apple,mca", 0, 1).ok_or("the mca node has no second reg window")?;
        // **The DMA engine is the one this machine's I2S names**, not the one its
        // SoC generation suggests: a t8112 has both an ADMAC and a SIO, and
        // guessing picks the wrong one.
        let mut dmas = [0u32; 2];
        let n = crate::fdt::prop_cells_of_compatible(dtb, b"apple,mca", b"dmas", &mut dmas);
        if n == 0 {
            return Err("the mca node names no dma engine");
        }
        let (admac_base, admac_size) =
            crate::fdt::reg_by_phandle(dtb, dmas[0]).ok_or("the mca node's dma phandle resolves to nothing")?;
        // That engine must be an ADMAC; a SIO would need firmware and a mailbox,
        // which is a different driver entirely and not this one. Compared
        // against *an* `apple,admac` rather than *the first* one — this machine
        // has two, and which one comes first in the tree is not a fact worth
        // depending on.
        match crate::fdt::reg_of_compatible(dtb, b"apple,admac") {
            None => return Err("no apple,admac node (a SIO dma path is unimplemented)"),
            Some((first, _)) if first != admac_base => crate::ktrace::log_fmt(format_args!(
                "audio: the i2s dma engine is {admac_base:#x}; the first apple,admac in the tree is {first:#x}"
            )),
            _ => {}
        }
        let (nco_base, nco_size) =
            crate::fdt::reg_of_compatible(dtb, b"apple,nco").ok_or("no apple,nco node")?;
        // **The amplifier's own bus, not the first `apple,i2c` in the tree.**
        // This machine has five controllers and the amp hangs off the second;
        // talking to the first meant every register write was acknowledged by
        // something else and discarded, and every read returned that bus's idea
        // of the world. Third instance of the same mistake in this driver, after
        // the DMA engine and its IOMMU — the tree states the relationship, so
        // resolve it.
        let (i2c_base, i2c_size) = crate::fdt::parent_reg_of_compatible(dtb, b"ti,tas2764")
            .ok_or("the amplifier's i2c bus is not in the tree")?;
        // **The DART is the one the engine names, resolved by phandle.** This
        // machine has twenty-one `apple,t8110-dart` nodes and the first one in
        // the tree belongs to something else entirely — its power domain is off,
        // so reading its lock register took a synchronous external abort at
        // `dart_base + 0x200`. `iommus = <&dart stream>`: the first cell is the
        // node, the second the stream id. Taking one from the phandle and the
        // other from a compatible search is how the two came from different
        // devices.
        let mut iommus = [0u32; 2];
        let n = crate::fdt::prop_cells_of_compatible(dtb, b"apple,admac", b"iommus", &mut iommus);
        if n < 2 {
            return Err("the dma engine names no iommu");
        }
        let (dart_base, _) =
            crate::fdt::reg_by_phandle(dtb, iommus[0]).ok_or("the dma engine's iommu phandle resolves to nothing")?;
        let dart_stream = iommus[1];
        Ok(Topology {
            i2c_base,
            i2c_size,
            // The amplifier's I2C address is its node's `reg`, and the bus it is
            // on is that node's parent — both from the tree, because this
            // machine's amp is at 0x38 where the reference M1's is at 0x31.
            amp_addr: amp_address(dtb).ok_or("no ti,tas2764 amplifier in the tree")?,
            amp_gpio: amp_reset_gpio(dtb),
            nco_base,
            nco_size,
            nco_fin: nco_input_hz(dtb).unwrap_or(900_000_000),
            mca_clusters,
            mca_clusters_size,
            mca_switch,
            mca_switch_size,
            admac_base,
            admac_size,
            dart_base,
            dart_phandle: iommus[0],
            dart_stream,
        })
    }
}

/// The amplifier's 7-bit I2C address, from its node's `reg`.
fn amp_address(dtb: u64) -> Option<u8> {
    // SAFETY: as `discover`.
    let (addr, _) = unsafe { crate::fdt::reg_of_compatible(dtb, b"ti,tas2764") }?;
    Some((addr & 0x7f) as u8)
}

/// `(pinctrl base, size, pin)` for the amplifier's `shutdown-gpios`, if it has
/// one. `None` simply means the part is not held in reset by a GPIO here.
fn amp_reset_gpio(dtb: u64) -> Option<(u64, u64, u32)> {
    let mut cells = [0u32; 3];
    // SAFETY: as `discover`.
    let n = unsafe { crate::fdt::prop_cells_of_compatible(dtb, b"ti,tas2764", b"shutdown-gpios", &mut cells) };
    if n < 2 {
        return None;
    }
    // SAFETY: as `discover`.
    let (base, size) = unsafe { crate::fdt::reg_by_phandle(dtb, cells[0]) }?;
    Some((base, size, cells[1]))
}

/// The NCO's reference frequency, from the fixed-clock its `clocks` names.
fn nco_input_hz(dtb: u64) -> Option<u64> {
    let mut clk = [0u32; 1];
    // SAFETY: as `discover`.
    let n = unsafe { crate::fdt::prop_cells_of_compatible(dtb, b"apple,nco", b"clocks", &mut clk) };
    if n == 0 {
        return None;
    }
    // `clock-frequency` is a plain u32 on the referenced node; reuse the
    // phandle walker and read the property off it.
    let mut hz = [0u32; 1];
    // SAFETY: as `discover`.
    let n = unsafe { crate::fdt::prop_cells_by_phandle(dtb, clk[0], b"clock-frequency", &mut hz) };
    (n > 0).then(|| hz[0] as u64)
}

/// The live device, once `/audio up` has built it.
pub struct AppleAudio {
    admac: admac::AdmacTx,
    mca: mca::Mca,
    bus: i2c::I2c,
    amp: u8,
    /// The DMA ring: one contiguous buffer, handed to the engine in chunks.
    dma_va: u64,
    dma_iova: u64,
    dma_bytes: usize,
    /// Byte cursor into the ring for the next chunk.
    pos: usize,
    /// Descriptors submitted but not yet reported complete.
    inflight: usize,
    /// Whether the amplifier is currently out of shutdown.
    driving: bool,
    /// When a descriptor last completed, so a silent stall can be named.
    last_retire: u64,
    stall_reported: bool,
    /// Whether the amplifier's refusal to power up has been reported.
    power_reported: bool,
}

/// Chunk handed to the engine per descriptor: 64 ms at 48 kHz mono, 4 bytes a
/// sample.
///
/// **Sized so a whole pump chunk fits in the ring without blocking.** The media
/// player feeds ~200 ms at a time and paces itself on `playing()` rather than on
/// `out_free_bytes`, so a ring shorter than that would make every feed wait for
/// the engine — with the `SND` lock held, which is time the whole machine spends
/// doing nothing. Four 64 ms descriptors is 256 ms, comfortably more.
const CHUNK_FRAMES: usize = RATE as usize * 64 / 1000;
const CHUNK_BYTES: usize = CHUNK_FRAMES * 4;
/// Ring capacity in chunks. The hardware descriptor ring is four slots — its
/// read/write cursors are 2-bit fields — so this cannot usefully be raised.
const RING_CHUNKS: usize = 4;
/// How long the engine may go without retiring a descriptor before the call
/// gives up. Generous next to one 64 ms descriptor; the point is that it is
/// finite, since the alternative to a bound here is a wedged machine.
const STALL_MS: u64 = 500;

impl AppleAudio {
    /// Reclaim any completed descriptors, so `out_free_bytes` and `playing`
    /// reflect the engine rather than what was submitted.
    fn reap(&mut self) {
        while !self.admac.reports_empty() {
            let _ = self.admac.take_report();
            self.inflight = self.inflight.saturating_sub(1);
        }
        self.admac.ack();
    }

    /// Copy `pcm` into the ring and hand it to the engine, one chunk per
    /// descriptor, stopping when the ring is full.
    fn submit(&mut self, pcm: &[i16]) -> usize {
        let mut done = 0usize;
        while done < pcm.len() && self.inflight < RING_CHUNKS && self.admac.can_submit() {
            let n = (pcm.len() - done).min(CHUNK_FRAMES);
            let off = self.pos;
            // SAFETY: `dma_va` is a `dma_bytes` DMA allocation this owns, and
            // `off + n*4` stays inside it by the wrap below.
            unsafe {
                let dst = (self.dma_va as *mut u32).add(off / 4);
                for (i, s) in pcm[done..done + n].iter().enumerate() {
                    dst.add(i).write_volatile(super::sample_word(*s));
                }
            }
            // The engine reads this memory without going through our caches.
            clean_dcache(self.dma_va + off as u64, n * 4);
            if self.admac.submit(self.dma_iova + off as u64, (n * 4) as u32).is_none() {
                break;
            }
            self.inflight += 1;
            self.pos = (off + CHUNK_BYTES) % self.dma_bytes;
            done += n;
        }
        done
    }
}

/// Busy-wait `ms` milliseconds off the generic timer.
///
/// Deliberately not `upkeep()`-pumping: these waits are single-digit
/// milliseconds inside a bring-up sequence, and the one thing a device-settling
/// delay must not do is run other code that might touch the device.
fn mdelay(ms: u64) {
    let end = crate::arch::now_ms() + ms;
    while crate::arch::now_ms() < end {
        core::hint::spin_loop();
    }
}

/// Clean a range out of the data cache to the point of coherency.
///
/// The DMA engine is **not** cache-coherent with the CPU here, so samples
/// written through a normal cached mapping can still be sitting in this core's
/// cache when the engine reads the buffer. The symptom is not silence: it is
/// the *previous* contents of that memory being played, which sounds like a
/// glitching or repeating stream rather than a missing barrier.
fn clean_dcache(va: u64, len: usize) {
    // SAFETY: cache maintenance over a mapped normal-memory buffer.
    unsafe {
        let mut p = va & !63;
        let end = va + len as u64;
        while p < end {
            core::arch::asm!("dc cvac, {}", in(reg) p, options(nostack, preserves_flags));
            p += 64;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

impl SndDevice for AppleAudio {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        if pcm.is_empty() {
            return Ok(());
        }
        let resampled;
        let src = if hz == RATE {
            pcm
        } else {
            resampled = crate::sound::resample(pcm, hz, RATE);
            &resampled[..]
        };
        // **Nothing in here may call `upkeep()`.** This runs inside
        // `sound::play_ch`, which holds the `SND` lock — and `upkeep` pumps the
        // media player, which calls `sound::play_ch` again. `Locked` is not
        // reentrant, so that is a spin with interrupts disabled: the machine
        // stops dead, with no output and no Ctrl+C. It is the exact failure
        // `Locked::try_with`'s documentation warns about, reached from a new
        // direction. So the wait below spins on the engine alone, and it is
        // bounded, because a DMA engine that never retires a descriptor must
        // report that rather than hang.
        self.reap();
        let mut fed = 0usize;
        let mut last_progress = crate::arch::now_ms();
        while fed < src.len() {
            let n = self.submit(&src[fed..]);
            if n > 0 {
                fed += n;
                last_progress = crate::arch::now_ms();
            }
            // The serialiser starts only once there is data queued (an empty
            // FIFO underruns immediately), and the amplifier only after the
            // serialiser is running — it shuts itself back down without a clock.
            if !self.mca.tx_running() {
                self.mca.start_tx();
                // The amp samples the bit clock as it leaves shutdown. Starting
                // the serialiser and writing ACTIVE in the same breath is how
                // it sees "no clock" and sits at PWR_CTRL=0x0e with an empty
                // fault latch — identical to a wrong register map.
                mdelay(2);
            }
            if !self.driving {
                match tas2764::power_up(&self.bus, self.amp) {
                    Ok(pwr) if tas2764::is_driving(pwr) => self.driving = true,
                    Ok(pwr) => {
                        // It declined. Say why once, from its own latched fault
                        // register — a clock error means the amplifier is
                        // rejecting our I2S framing, which is a completely
                        // different repair from anything on this side of the bus.
                        if !self.power_reported {
                            self.power_reported = true;
                            let f = tas2764::faults(&self.bus, self.amp).unwrap_or(0);
                            crate::ktrace::log_fmt(format_args!(
                                "audio: the amplifier refused to leave shutdown (PWR_CTRL={pwr:#04x}); \
                                 latched faults {f:#04x} -- tdm_clock_error={} over_current={} over_temp={}",
                                (f >> 2) & 1,
                                (f >> 1) & 1,
                                f & 1
                            ));
                        }
                    }
                    Err(_) => {}
                }
            }
            if fed < src.len() {
                if crate::arch::now_ms().saturating_sub(last_progress) > STALL_MS {
                    return Err("the audio dma engine stopped retiring descriptors");
                }
                core::hint::spin_loop();
                self.reap();
            }
        }
        Ok(())
    }

    fn playing(&mut self) -> bool {
        let before = self.inflight;
        self.reap();
        if self.inflight != before || self.inflight == 0 {
            self.last_retire = crate::arch::now_ms();
            return self.inflight > 0;
        }
        // Nothing retired, and the ring holds at most 256 ms. The media player
        // refills only when this returns false, so an engine that stops
        // reporting stops playback dead — with no error anywhere, because every
        // register still reads correct. Say it once; the next `play` turns it
        // into a real error via `STALL_MS`.
        if !self.stall_reported && crate::arch::now_ms().saturating_sub(self.last_retire) > 2_000 {
            self.stall_reported = true;
            crate::ktrace::log_fmt(format_args!(
                "audio: the dma engine has not retired a descriptor in 2 s ({} in flight, status {:#x}, residue {}) \
                 -- descriptors are being accepted but never completing",
                self.inflight,
                self.admac.status(),
                self.admac.residue()
            ));
        }
        true
    }

    fn out_free_bytes(&mut self) -> usize {
        self.reap();
        RING_CHUNKS.saturating_sub(self.inflight) * CHUNK_FRAMES * 2
    }

    fn capture_start(&mut self, _hz: u32) -> Result<(), &'static str> {
        Err("the built-in microphone is behind the always-on processor, which has no driver")
    }

    fn capture_read(&mut self, _out: &mut [i16]) -> usize {
        0
    }

    fn capture_stop(&mut self) {}
}

impl Drop for AppleAudio {
    fn drop(&mut self) {
        tas2764::power_down(&self.bus, self.amp);
        self.mca.stop();
        self.admac.disable();
    }
}

/// What the last `/audio probe` or `up` found, for `/audio status`.
static SUMMARY: Locked<Option<String>> = Locked::new(None);

/// The summary line, if anything has run.
pub fn probe_summary() -> Option<String> {
    SUMMARY.with(|s| s.clone())
}

fn set_summary(s: String) {
    SUMMARY.with(|slot| *slot = Some(s));
}

/// Power the blocks and read the amplifier's registers. **Writes nothing to the
/// amplifier** — the safe first step on a machine.
fn probe() -> Result<(), String> {
    let t = discover().map_err(String::from)?;
    // **Power every block from its own node, not from a list of names.** The
    // first version enabled four domains by label and left the DMA engine's
    // IOMMU gated, which is not a bad read — a gated block does not decode at
    // all, so the first access is a fault. Each node states the domains it
    // needs; the MCA alone names seven.
    // NB `enable_domains_of` matches by compatible, so for the buses this
    // machine has several of it powers the *first*. That is right for the ones
    // below and wrong for I2C, whose real controller is powered separately after
    // this loop from the amplifier's own parent node.
    for (what, compat) in [
        ("i2c", &b"apple,i2c"[..]),
        ("iommu", &b"apple,t8110-dart"[..]),
        ("dma engine", &b"apple,admac"[..]),
        ("i2s controller", &b"apple,mca"[..]),
        ("clock generator", &b"apple,nco"[..]),
        ("gpio", &b"apple,pinctrl"[..]),
    ] {
        if !pmgr::enable_domains_of(compat) {
            return Err(alloc::format!("a power domain for the {what} would not come up"));
        }
    }
    // The amplifier's own I2C controller. `discover` resolved its base from the
    // amp's parent; its power domain has to come from the same node, or the
    // controller we actually drive is the one left gated.
    {
        let dtb = crate::arch::aarch64::boot::boot_x0();
        let mut doms = [0u32; 4];
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        let n = unsafe { crate::fdt::parent_prop_cells_of_compatible(dtb, b"ti,tas2764", b"power-domains", &mut doms) };
        for d in &doms[..n.min(doms.len())] {
            if *d != 0 && !pmgr::enable_phandle(*d, 0) {
                return Err(String::from("the amplifier's i2c controller power domain would not come up"));
            }
        }
    }
    // The IOMMU's domain is named by *its own* node, but `enable_domains_of`
    // above matched the first `apple,t8110-dart` in the tree — which is not
    // ours. Power the one the engine actually points at.
    {
        let dtb = crate::arch::aarch64::boot::boot_x0();
        let mut doms = [0u32; 4];
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        let n = unsafe { crate::fdt::prop_cells_by_phandle(dtb, t.dart_phandle, b"power-domains", &mut doms) };
        for d in &doms[..n.min(doms.len())] {
            if *d != 0 && !pmgr::enable_phandle(*d, 0) {
                return Err(String::from("the dma engine's iommu power domain would not come up"));
            }
        }
    }
    // SAFETY: the I2C block's power domain is on (just enabled) and the window
    // is the one the tree published.
    let bus = unsafe { i2c::I2c::new(t.i2c_base, t.i2c_size as usize) };
    // Release the amplifier's reset before talking to it: held in reset it does
    // not answer, which is indistinguishable from absent.
    if let Some((gbase, gsize, pin)) = t.amp_gpio {
        set_amp_reset(gbase, gsize, pin, true);
    }
    // Select book 0 page 0 before reading anything, or the register file
    // reported is whichever book the previous owner of this chip left selected —
    // which is exactly how the first bring-up read a plausible file full of
    // values it had never written. No software reset here: `probe` is the
    // read-only step, and a reset is a change.
    let _ = bus.write_reg(t.amp_addr, tas2764::reg::PAGE, &[0]);
    let _ = bus.write_reg(t.amp_addr, tas2764::reg::BOOK, &[0]);
    let _ = bus.write_reg(t.amp_addr, tas2764::reg::PAGE, &[0]);
    let page = tas2764::read_reg(&bus, t.amp_addr, tas2764::reg::PAGE);
    let pwr = tas2764::read_reg(&bus, t.amp_addr, tas2764::reg::PWR_CTRL);
    match (page, pwr) {
        (Ok(p), Ok(w)) => {
            let s = alloc::format!(
                "amplifier at i2c {:#04x} on bus {:#x} answered: page={:#04x} power={:#04x} ({}); \
                 nco {:#x} mca {:#x} admac {:#x} dart {:#x} stream {}",
                t.amp_addr,
                t.i2c_base,
                p,
                w,
                if tas2764::is_driving(w) { "driving" } else { "shut down" },
                t.nco_base,
                t.mca_clusters,
                t.admac_base,
                t.dart_base,
                t.dart_stream
            );
            crate::ktrace::log_fmt(format_args!("audio: {s}"));
            set_summary(s);
            Ok(())
        }
        (Err(e), _) | (_, Err(e)) => Err(alloc::format!(
            "the amplifier at i2c {:#04x} did not answer: {}",
            t.amp_addr,
            e.as_str()
        )),
    }
}

/// Drive the amplifier's reset line. `assert_high` releases it (the line is
/// active-low shutdown).
fn set_amp_reset(base: u64, size: u64, pin: u32, release: bool) {
    let va = crate::mm::map_mmio(base, size as usize) + super::gpio_reg(pin);
    // SAFETY: `va` is one pin's register inside the pinctrl window the tree
    // sized; single 32-bit accesses.
    unsafe {
        let cur: u32;
        core::arch::asm!("ldr {0:w}, [{1}]", out(reg) cur, in(reg) va, options(nostack));
        let next = super::gpio_out_value(cur, release);
        core::arch::asm!("str {0:w}, [{1}]", in(reg) next, in(reg) va, options(nostack));
        crate::ktrace::log_fmt(format_args!(
            "audio: amp reset gpio pin {pin}: {cur:#x} -> {next:#x} ({})",
            if release { "released" } else { "held" }
        ));
    }
}

/// Bring the whole path up and register it as the machine's sound device.
pub fn up() -> Result<(), String> {
    if crate::sound::is_up() {
        return Err(String::from("a sound device is already up"));
    }
    probe()?;
    let t = discover().map_err(String::from)?;

    // The DMA engine reaches memory through a DART. Bypass is what this path
    // wants: the kernel's DMA allocations are identity-mapped (VA == PA here),
    // so with the stream in bypass the device address *is* the physical address
    // and there is no table to keep in step.
    // SAFETY: the DART in front of the audio DMA engine, from the tree.
    let dart = unsafe { crate::arch::aarch64::dart::Dart::new(t.dart_base as usize, t.dart_stream) };
    if dart.is_locked() {
        return Err(String::from(
            "the audio DART is locked by the bootloader; its translation cannot be reprogrammed",
        ));
    }
    if !dart.set_bypass() {
        return Err(String::from("the audio DART would not enter bypass"));
    }
    dart.flush_tlb();

    let dma_bytes = CHUNK_BYTES * RING_CHUNKS;
    let (dma_pa, dma_va) = crate::mm::alloc_dma(dma_bytes).ok_or_else(|| String::from("no DMA memory for the audio ring"))?;

    // SAFETY: every base below came from the device tree and every power domain
    // is on (checked in `probe`).
    unsafe {
        let n = nco::Nco::new(t.nco_base, t.nco_size as usize, 0);
        if !n.set_rate(t.nco_fin, (RATE * BITS_PER_FRAME) as u64) {
            return Err(alloc::format!(
                "the nco cannot make {} Hz from a {} Hz reference",
                RATE * BITS_PER_FRAME,
                t.nco_fin
            ));
        }
        let m = mca::Mca::new(
            t.mca_clusters,
            t.mca_clusters_size as usize,
            t.mca_switch,
            t.mca_switch_size as usize,
            0,
        );
        m.configure(BITS_PER_FRAME);

        let a = admac::AdmacTx::new(t.admac_base, t.admac_size as usize, 0);
        a.reset();
        a.enable();

        let bus = i2c::I2c::new(t.i2c_base, t.i2c_size as usize);
        // **A full reset first.** The amplifier was configured by macOS minutes
        // ago and m1n1 left it alone, so it can be on any book — where every
        // write is acknowledged and lands on a register that is not the one
        // named. Pulse the shutdown line the way the part's own driver does
        // (low, settle, high, settle), then select book 0 page 0 and software
        // reset.
        if let Some((gbase, gsize, pin)) = t.amp_gpio {
            set_amp_reset(gbase, gsize, pin, false);
            mdelay(2);
            set_amp_reset(gbase, gsize, pin, true);
            mdelay(2);
        }
        tas2764::select_and_reset(&bus, t.amp_addr, mdelay).map_err(|e| {
            alloc::format!("the amplifier would not accept a reset: {}", e.as_str())
        })?;
        let took = tas2764::configure(&bus, t.amp_addr).map_err(|e| {
            alloc::format!("the amplifier refused its configuration: {}", e.as_str())
        })?;
        if !took {
            return Err(String::from(
                "the amplifier acknowledged its configuration and did not keep it (read-back \
                 mismatch) -- it is not on book 0 page 0, or it is still held in shutdown",
            ));
        }

        let dev = AppleAudio {
            admac: a,
            mca: m,
            bus,
            amp: t.amp_addr,
            dma_va,
            dma_iova: dma_pa,
            dma_bytes,
            pos: 0,
            inflight: 0,
            driving: false,
            last_retire: crate::arch::now_ms(),
            stall_reported: false,
            power_reported: false,
        };
        crate::sound::init(Box::new(dev));
    }
    let s = alloc::format!(
        "built-in audio up: {} Hz mono, nco {:#x} -> mca {:#x} -> admac {:#x} -> amp {:#04x} \
         (the amplifier leaves shutdown when the first samples are queued; \
         `/audio gain <n>` sets the analog level, which starts at the minimum)",
        RATE,
        t.nco_base,
        t.mca_clusters,
        t.admac_base,
        t.amp_addr
    );
    crate::ktrace::log_fmt(format_args!("audio: {s}"));
    set_summary(s);
    Ok(())
}


/// `/audio dump` — read every layer back and print it.
///
/// **The one question a silent-but-running chain poses is "which of these is not
/// what I told it to be", and only the hardware can answer it.** Everything here
/// is a read: the amplifier's register file, the I2S clock generator and
/// serialiser, the DMA channel, the clock generator. It is the `/wifi diag`
/// shape — one boot that distinguishes causes that otherwise look identical from
/// the outside, all of which present as "it says it is playing and nothing comes
/// out".
pub fn dump() -> Result<(), String> {
    let t = discover().map_err(String::from)?;
    // SAFETY: every base came from the device tree; `up`/`probe` powered the
    // domains, and these are reads.
    unsafe {
        let n = nco::Nco::new(t.nco_base, t.nco_size as usize, 0);
        let r = n.regs();
        crate::ktrace::log_fmt(format_args!(
            "audio: nco enabled={} regs={:#010x} {:#010x} {:#010x} {:#010x} (want _ {:#x} {:#x} {:#x})",
            n.enabled(),
            r[0], r[1], r[2], r[3],
            super::nco::calc_regvals(t.nco_fin, (RATE * BITS_PER_FRAME) as u64).map(|v| v[1]).unwrap_or(0),
            super::nco::calc_regvals(t.nco_fin, (RATE * BITS_PER_FRAME) as u64).map(|v| v[2]).unwrap_or(0),
            super::nco::calc_regvals(t.nco_fin, (RATE * BITS_PER_FRAME) as u64).map(|v| v[3]).unwrap_or(0),
        ));
        let m = mca::Mca::new(
            t.mca_clusters,
            t.mca_clusters_size as usize,
            t.mca_switch,
            t.mca_switch_size as usize,
            0,
        );
        m.dump();
        let a = admac::AdmacTx::new(t.admac_base, t.admac_size as usize, 0);
        a.dump();
        let bus = i2c::I2c::new(t.i2c_base, t.i2c_size as usize);
        // Book 0 page 0 first — see `probe`.
        let _ = bus.write_reg(t.amp_addr, tas2764::reg::PAGE, &[0]);
        let _ = bus.write_reg(t.amp_addr, tas2764::reg::BOOK, &[0]);
        let _ = bus.write_reg(t.amp_addr, tas2764::reg::PAGE, &[0]);
        // The amplifier's register file, sixteen at a time. A configuration that
        // never landed — a NAK swallowed, a page left selected, a device that
        // does not auto-increment across a multi-byte write — reads back as the
        // reset defaults, which is the difference between "it is misconfigured"
        // and "it was never configured".
        for base in [0x00u8, 0x10] {
            let mut row = [0u8; 16];
            let mut ok = true;
            for (i, slot) in row.iter_mut().enumerate() {
                match bus.read_reg8(t.amp_addr, base + i as u8) {
                    Ok(v) => *slot = v,
                    Err(e) => {
                        crate::ktrace::log_fmt(format_args!(
                            "audio: amp reg {:#04x}: {}",
                            base + i as u8,
                            e.as_str()
                        ));
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                crate::ktrace::log_fmt(format_args!(
                    "audio: amp {base:#04x}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} \
                     {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                    row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15],
                ));
            }
        }
        // What the configuration sequence asked for, so the two can be compared
        // without a datasheet to hand.
        crate::ktrace::log_fmt(format_args!(
            "audio: amp expected after configure: 02=80(active+bop) 03={:02x}(min gain) 08={:02x} 09={:02x} 0a={:02x} 0c={:02x}",
            tas2764::GAIN_MIN,
            tas2764::TDM_CFG0_48K,
            tas2764::TDM_CFG1_I2S,
            tas2764::TDM_CFG2_32_32,
            tas2764::TDM_CFG3_I2S
        ));
        // The registers m1n1's own `SN012776Regs` map names — this machine's amp
        // is `ti,sn012776`, and that map is the only description of the part
        // available here. `INT_LTCH0` is the one worth the round trip: it latches
        // **TDM clock error**, over-current and over-temperature, so it answers
        // "did the amplifier reject the stream" directly instead of by inference.
        for (name, r, decode) in [
            ("MODE_CTRL", 0x02u8, true),
            ("CHNL_0   ", 0x03, true),
            ("DVC      ", 0x1a, false),
            ("INT_LTCH0", 0x49, true),
            ("INT_CLK_CFG", 0x5c, false),
        ] {
            match bus.read_reg8(t.amp_addr, r) {
                Ok(v) if decode && r == 0x02 => crate::ktrace::log_fmt(format_args!(
                    "audio: amp {name} ({r:#04x}) = {v:#04x} -- mode={} isns_pd={} vsns_pd={}",
                    match v & 3 {
                        0 => "ACTIVE",
                        1 => "MUTE",
                        2 => "SHUTDOWN",
                        _ => "?",
                    },
                    (v >> 3) & 1,
                    (v >> 2) & 1
                )),
                Ok(v) if decode && r == 0x03 => crate::ktrace::log_fmt(format_args!(
                    "audio: amp {name} ({r:#04x}) = {v:#04x} -- amp_level={} cds_mode={}",
                    (v >> 1) & 0x1f,
                    (v >> 6) & 3
                )),
                Ok(v) if decode && r == 0x49 => crate::ktrace::log_fmt(format_args!(
                    "audio: amp {name} ({r:#04x}) = {v:#04x} -- latched: tdm_clock_error={} over_current={} over_temp={}",
                    (v >> 2) & 1,
                    (v >> 1) & 1,
                    v & 1
                )),
                Ok(v) => crate::ktrace::log_fmt(format_args!("audio: amp {name} ({r:#04x}) = {v:#04x}")),
                Err(e) => crate::ktrace::log_fmt(format_args!("audio: amp {name} ({r:#04x}): {}", e.as_str())),
            }
        }
    }
    Ok(())
}


/// `/audio amp [reg]` — is the I2C register round-trip faithful?
///
/// The dump raised a question it cannot answer: the amplifier's registers read
/// back as none of the values written to them. Two possibilities need opposite
/// fixes — **the writes are not landing** (acknowledged but ignored: a paging or
/// sequencing problem) or **the reads are not returning register contents** (an
/// off-by-one in the FIFO, a missing repeated START, an extra count byte, as
/// SMBus block reads use). Guessing costs a boot each; a write-read-restore on
/// one register settles it in one.
///
/// The register used is digital volume (`0x1a`), an 8-bit field that stores
/// whatever is written. An earlier version used `0x0d` (TAS2764 `TDM_CFG4`, one
/// edge-select bit) and then printed a hardcoded "map is wrong" even when the
/// write could not have stuck.
pub fn amp_roundtrip(reg_arg: Option<u8>) -> Result<(), String> {
    let t = discover().map_err(String::from)?;
    // SAFETY: the I2C block is powered (probe/up ran) and the window is the
    // tree's.
    let bus = unsafe { i2c::I2c::new(t.i2c_base, t.i2c_size as usize) };
    let rd = |r: u8| -> String {
        match bus.read_reg8(t.amp_addr, r) {
            Ok(v) => alloc::format!("{v:#04x}"),
            Err(e) => alloc::format!("<{}>", e.as_str()),
        }
    };
    if let Some(r) = reg_arg {
        crate::serial_println!("audio> amp[{r:#04x}] = {}", rd(r));
        return Ok(());
    }
    // Digital volume is an 8-bit field that takes any value. TDM_CFG4 (0x0d)
    // only implements a single edge-select bit, so writing 0x01 and reading
    // 0x00 is *success* on that register — which is how the first diagnostic
    // printed a hardcoded "map is wrong" over a write that never could stick.
    const SCRATCH: u8 = tas2764::reg::DVC;
    const PROBE_VALUE: u8 = 0x01;
    let _ = bus.write_reg(t.amp_addr, tas2764::reg::PAGE, &[0]);
    let _ = bus.write_reg(t.amp_addr, tas2764::reg::BOOK, &[0]);
    crate::serial_println!("audio> amp round-trip on register {SCRATCH:#04x}:");
    let before = bus.read_reg8(t.amp_addr, SCRATCH).ok();
    crate::serial_println!("audio>   before          = {}", rd(SCRATCH));
    let w1 = bus.write_reg(t.amp_addr, SCRATCH, &[PROBE_VALUE]);
    crate::serial_println!(
        "audio>   write {PROBE_VALUE:#04x}      = {}",
        match w1 {
            Ok(()) => "acknowledged",
            Err(e) => e.as_str(),
        }
    );
    let after = bus.read_reg8(t.amp_addr, SCRATCH).ok();
    crate::serial_println!("audio>   read back       = {}", rd(SCRATCH));
    let restore = before.unwrap_or(0);
    let w2 = bus.write_reg(t.amp_addr, SCRATCH, &[restore]);
    crate::serial_println!(
        "audio>   restore {restore:#04x}    = {}",
        match w2 {
            Ok(()) => "acknowledged",
            Err(e) => e.as_str(),
        }
    );
    crate::serial_println!("audio>   read back       = {}", rd(SCRATCH));
    match after {
        Some(v) if v == PROBE_VALUE => crate::serial_println!(
            "audio>   verdict: write stuck — I2C both ways work"
        ),
        Some(v) => crate::serial_println!(
            "audio>   verdict: wrote {PROBE_VALUE:#04x}, read {v:#04x} — acknowledged but ignored, \
             or this is not page 0"
        ),
        None => crate::serial_println!("audio>   verdict: the read failed"),
    }
    Ok(())
}


/// `/audio gain [n]` and `/audio dvc [n]` — the two levels this part has.
///
/// **Separate from the bring-up on purpose.** `CHNL_0` is an analog gain into a
/// real speaker, so nothing raises it implicitly; the configuration leaves it at
/// the minimum and this is how a human steps it up, one deliberate number at a
/// time, with the value read back so "I set it" and "it took" are not the same
/// claim.
pub fn set_level(analog: bool, n: Option<u8>) -> Result<(), String> {
    let t = discover().map_err(String::from)?;
    // SAFETY: the I2C block is powered by `probe`/`up`, and this is the bus the
    // amplifier's own node hangs off.
    let bus = unsafe { i2c::I2c::new(t.i2c_base, t.i2c_size as usize) };
    let reg = if analog { tas2764::reg::CHNL_0 } else { tas2764::reg::DVC };
    let Some(n) = n else {
        let v = bus
            .read_reg8(t.amp_addr, reg)
            .map_err(|e| alloc::format!("read: {}", e.as_str()))?;
        if analog {
            crate::serial_println!("audio> amp gain = {} (0..={}), CHNL_0={v:#04x}", (v >> 1) & 0x1f, tas2764::GAIN_MAX);
        } else {
            crate::serial_println!("audio> digital volume = {v:#04x} (0 = unity, 0xc9 = near-mute)");
        }
        return Ok(());
    };
    let got = if analog {
        tas2764::set_gain(&bus, t.amp_addr, n)
    } else {
        tas2764::set_dvc(&bus, t.amp_addr, n)
    }
    .map_err(|e| alloc::format!("write: {}", e.as_str()))?;
    if analog {
        let pwr = bus.read_reg8(t.amp_addr, tas2764::reg::PWR_CTRL).unwrap_or(0xff);
        crate::serial_println!(
            "audio> amp gain -> {} (CHNL_0 reads back {got:#04x}, power {pwr:#04x} {}); analog gain \
             into the built-in speaker — step it, do not jump to {}",
            (got >> 1) & 0x1f,
            if tas2764::is_driving(pwr) { "driving" } else { "not driving" },
            tas2764::GAIN_MAX
        );
        if (got >> 1) & 0x1f != n.min(tas2764::GAIN_MAX) {
            crate::serial_println!(
                "audio>   the amplifier did not keep it. Every other register took its write on this \
                 bus, so this is the register, not the path — try again with playback stopped."
            );
        }
    } else {
        crate::serial_println!("audio> digital volume -> {got:#04x}");
    }
    Ok(())
}

/// `/audio [probe|up|status]`.
pub fn command(arg: &str) {
    if !crate::arch::aarch64::is_apple() {
        crate::serial_println!("audio> built-in audio bring-up is Apple Silicon only");
        return;
    }
    match arg.trim() {
        "" | "status" => match probe_summary() {
            Some(s) => crate::serial_println!("audio> {s}"),
            None => crate::serial_println!(
                "audio> nothing brought up yet — `/audio probe` reads the amplifier (writes nothing), \
                 `/audio up` starts the whole path, `/audio dump` reads every layer back"
            ),
        },
        "dump" | "diag" => match dump() {
            Ok(()) => crate::serial_println!("audio> dump written to /ktrace"),
            Err(e) => crate::serial_println!("audio> dump failed: {e}"),
        },
        a if a.starts_with("gain") || a.starts_with("dvc") => {
            let analog = a.starts_with("gain");
            let arg = a.trim_start_matches("gain").trim_start_matches("dvc").trim();
            let n = if arg.is_empty() {
                None
            } else if let Some(h) = arg.strip_prefix("0x") {
                u8::from_str_radix(h, 16).ok()
            } else {
                arg.parse::<u8>().ok()
            };
            if !arg.is_empty() && n.is_none() {
                crate::serial_println!("audio> not a number: '{arg}'");
            } else if let Err(e) = set_level(analog, n) {
                crate::serial_println!("audio> {e}");
            }
        }
        a if a.starts_with("amp") => {
            let arg = a.trim_start_matches("amp").trim();
            let reg = if arg.is_empty() {
                None
            } else {
                u8::from_str_radix(arg.trim_start_matches("0x"), 16).ok()
            };
            if let Err(e) = amp_roundtrip(reg) {
                crate::serial_println!("audio> amp: {e}");
            }
        }
        "probe" => match probe() {
            Ok(()) => crate::serial_println!("audio> probe ok — see the audio: line in /ktrace"),
            Err(e) => crate::serial_println!("audio> probe failed: {e}"),
        },
        "up" | "init" => match up() {
            Ok(()) => crate::serial_println!(
                "audio> up — try `/open /samples/audios/sample.aac` (mono {} Hz, lowest amplifier gain)",
                RATE
            ),
            Err(e) => crate::serial_println!("audio> bring-up failed: {e}"),
        },
        other => crate::serial_println!("audio> unknown subcommand '{other}' (probe|up|dump|gain [n]|dvc [n]|amp [reg]|status)"),
    }
}
