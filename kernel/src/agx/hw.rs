//! AGX hardware orchestration (aarch64 / real Apple Silicon): FDT discovery, the
//! minimal PMGR power-domain enable, and the RTKit boot handshake that ties the
//! pure [`super::proto`] codecs to the [`super::asc`] mailbox transport. Ported
//! from m1n1's `rtkit_boot` (`third_party/m1n1/src/rtkit.c:496-657`) but
//! cooperative + bounded: every wait pumps the UI/clock/net and answers Ctrl+C.

use super::asc::{Asc, Message};
use super::proto::{self, Action, BufferKind, EndpointSet, RtkitState};
use alloc::alloc::{alloc_zeroed, Layout};

/// t8112 GPU node: `gpu@206400000`, compatible `apple,agx-t8112`. Its `reg[0]`
/// ("asc") is the coprocessor CPU base; the mailbox FIFO sits at +0x8000
/// (`asc.rs` folds that in). Verified against `third_party/dtb/t8112-j473.dtb`.
const AGX_COMPAT: &[u8] = b"apple,agx-t8112";

/// The `gfx` power domain's PMGR pwrstate register, absolute. From
/// `third_party/dtb/t8112-j473.dtb`: `power-management@23b700000` (the t8112
/// PMGR, `reg = <0x2 0x3b700000 …>`) + its child `power-controller@430`
/// (`label = "gfx"`). t8112-specific — only ever reached after the
/// `apple,agx-t8112` GPU node is confirmed present, so it can't misfire on
/// another SoC. Mandatory before touching ASC MMIO (m1n1 `kboot_gpu.c`).
const GFX_PWRSTATE: u64 = 0x2_3b70_0430;
const PMGR_PS_TARGET: u32 = 0xf; // bits[3:0]  — TARGET field
const PMGR_PS_ACTUAL_SHIFT: u32 = 4; // bits[7:4] — ACTUAL field
const PMGR_PS_ACTIVE: u32 = 0xf;
/// Bits `pmgr_set_mode` clears before setting TARGET (auto-enable + the sticky
/// was-clock/power-gated bits): `AUTO_ENABLE(28) | WAS_CLKGATED(9) | WAS_PWRGATED(8) | TARGET(3:0)`.
const PMGR_CLEAR: u32 = (1 << 28) | (1 << 9) | (1 << 8) | 0xf;

/// No-IOMMU device-address base for RTKit shared buffers. The GPU node has no
/// `iommus`, so shared DRAM is addressed directly; Apple ORs an `asc-dram-mask`
/// that isn't in this FDT. Start at 0 and log every buffer request so the mask
/// (if any) shows up in the trace.
// TODO(asc-dram-mask): cross-check against drm/asahi's t8112 constant.
const DVA_BASE: u64 = 0;

/// Timeouts (ms) — generous, cooperative, always bounded.
const HELLO_TIMEOUT_MS: u64 = 1000;
const STEP_TIMEOUT_MS: u64 = 1000;
const SEND_TIMEOUT_MS: u64 = 200;
/// Overall budget for the "wait until the IOP powers ON" phase. The IOP may take
/// several seconds to init its logs + power on after we grant its buffers, so
/// this is a *total* deadline (not per-message) — a slow-but-progressing IOP is
/// not cut off, and only true silence for the whole budget is a timeout.
const POWERUP_BUDGET_MS: u64 = 15000;
/// Per-poll recv window inside the power-up budget (short, so the heartbeat +
/// interrupt stay responsive).
const POWERUP_POLL_MS: u64 = 500;

/// The outcome of a bring-up attempt (drives the `/agx` report + message).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Outcome {
    #[default]
    NotRun,
    /// No `apple,agx-t8112` node — not a t8112, or no FDT (QEMU/VBox).
    NoGpu,
    /// `cpu_start` done but no HELLO in ~1 s — **firmware not resident** (the
    /// external Asahi GPU-firmware-provisioning blocker).
    NoHello,
    BadHello,
    VersionMismatch,
    EpmapFail,
    SendFail,
    /// Timed out waiting for the IOP to reach power ON.
    Timeout,
    /// The IOP reported a crash during boot.
    Crashed,
    /// Full handshake complete — coprocessor RUNNING.
    Running,
}

/// Snapshot of the last bring-up, for `/agx status`.
#[derive(Clone, Copy, Default)]
struct Report {
    outcome: Outcome,
    asc_base: u64,
    version: u64,
    eps: EndpointSet,
    iop_power: u64,
    ap_power: u64,
    n_buffers: u32,
    cpu_running: bool,
}

static REPORT: crate::mm::Locked<Report> = crate::mm::Locked::new(Report {
    outcome: Outcome::NotRun,
    asc_base: 0,
    version: 0,
    eps: EndpointSet { crashlog: false, debug: false, ioreport: false, syslog: false, oslog: false },
    iop_power: 0,
    ap_power: 0,
    n_buffers: 0,
    cpu_running: false,
});

/// True if `needle` appears in the FDT `/chosen bootargs` (the m1n1 `-b` command
/// line). Gates the bring-up on `chitti.agx`, mirroring `apple_usb`'s
/// `chitti.usb`.
fn bootarg_present(needle: &[u8]) -> bool {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    if let Some(c) = unsafe { crate::fdt::chosen(fdt) } {
        if !c.bootargs_ptr.is_null() && c.bootargs_len >= needle.len() {
            // SAFETY: `[bootargs_ptr, +len)` views the still-mapped FDT.
            let s = unsafe { core::slice::from_raw_parts(c.bootargs_ptr, c.bootargs_len) };
            return s.windows(needle.len()).any(|w| w == needle);
        }
    }
    false
}

/// Discover the AGX ASC CPU base from the boot FDT (`reg[0]` of the
/// `apple,agx-t8112` GPU node). `None` when absent (QEMU/VBox/other SoC).
fn discover_asc_base() -> Option<u64> {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let (base, _size) = unsafe { crate::fdt::reg_of_compatible(fdt, AGX_COMPAT) }?;
    Some(base)
}

/// Read/modify/write a 32-bit PMGR register (single `ldr`/`str`, the MMIO rule).
fn pmgr_r32(addr: u64) -> u32 {
    // SAFETY: single 32-bit MMIO read of a mapped PMGR register.
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}
fn pmgr_w32(addr: u64, v: u32) {
    // SAFETY: single 32-bit MMIO write of a mapped PMGR register.
    unsafe { core::ptr::write_volatile(addr as *mut u32, v) }
}

/// Enable the `gfx` power domain (m1n1 `pmgr_set_mode` → ACTIVE): clear the
/// auto/gated bits + TARGET, set TARGET=ACTIVE, poll ACTUAL==ACTIVE. Bounded.
/// Returns false if the domain never reaches ACTIVE.
fn pmgr_enable_gfx() -> bool {
    crate::arch::aarch64::mmu::map_device_gib(GFX_PWRSTATE);
    let cur = pmgr_r32(GFX_PWRSTATE);
    let next = (cur & !PMGR_CLEAR) | PMGR_PS_TARGET;
    pmgr_w32(GFX_PWRSTATE, next);
    let deadline = crate::arch::now_ms() + 100;
    while crate::arch::now_ms() < deadline {
        let actual = (pmgr_r32(GFX_PWRSTATE) >> PMGR_PS_ACTUAL_SHIFT) & 0xf;
        if actual == PMGR_PS_ACTIVE {
            crate::ktrace::log_fmt(format_args!("agx: gfx power domain ACTIVE (pmgr {:#x}={:#x})", GFX_PWRSTATE, pmgr_r32(GFX_PWRSTATE)));
            return true;
        }
        core::hint::spin_loop();
    }
    crate::ktrace::log_fmt(format_args!("agx: gfx power domain did NOT reach ACTIVE (pmgr {:#x}={:#x})", GFX_PWRSTATE, pmgr_r32(GFX_PWRSTATE)));
    false
}

/// Allocate a zeroed, 16 KiB-aligned DMA buffer of `pages`×4 KiB for an RTKit
/// shared region. aarch64 identity map ⇒ phys == the returned pointer. Leaked
/// for the coprocessor's lifetime. `None` on OOM.
fn alloc_shared(pages: u64) -> Option<u64> {
    let bytes = (pages.max(1) as usize) * 4096;
    let layout = Layout::from_size_align(bytes, 16384).ok()?;
    // SAFETY: nonzero, 16 KiB-aligned; leaked as device-shared DMA (VA == PA).
    let p = unsafe { alloc_zeroed(layout) } as u64;
    (p != 0).then_some(p)
}

/// Latched Ctrl+C during a bring-up (reset at the start of every `up()`). Once
/// set, every subsequent `pump()` reports abort, so a multi-second wait bails
/// promptly on the first Ctrl+C rather than only shortening the current poll.
static ABORT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// The cooperative pump run between every mailbox poll: keep the UI/clock/net
/// alive and report Ctrl+C (returns true to abort). Mirrors the inference/IO
/// loops' `upkeep()` + interrupt-poll discipline (CLAUDE.md standing rule).
fn pump() -> bool {
    crate::shell::upkeep();
    if crate::shell::poll_interrupt() {
        ABORT.store(true, core::sync::atomic::Ordering::Relaxed);
    }
    ABORT.load(core::sync::atomic::Ordering::Relaxed)
}

/// Send `(msg0, ep)`, logging + returning false on failure.
fn send(asc: &Asc, msg0: u64, ep: u8) -> bool {
    let ok = asc.send(&Message { msg0, msg1: ep as u32 }, SEND_TIMEOUT_MS, &mut pump);
    if !ok {
        crate::ktrace::log_fmt(format_args!("agx: send failed (ep {ep:#x}, msg {msg0:#018x}) — mailbox stuck?"));
    }
    ok
}

/// Drive the RTKit handshake to RUNNING (port of `rtkit_boot`). Fills `rep` and
/// returns the outcome.
fn rtkit_boot(asc: &Asc, rep: &mut Report) -> Outcome {
    use proto::*;

    // Boot the IOP and nudge it awake (INIT). Unconditional, per m1n1.
    asc.cpu_start();
    if !send(asc, msg_iop_pwr_state(POWER_INIT), EP_MGMT) {
        return Outcome::SendFail;
    }

    // --- Phase A decisive test: HELLO within ~1 s ⇒ firmware resident. --------
    let Some(hello) = asc.recv_blocking(HELLO_TIMEOUT_MS, &mut pump) else {
        crate::ktrace::log("agx", "no HELLO within 1s after cpu_start — gfx-asc firmware not resident (blocked on Asahi GPU-firmware provisioning)");
        return Outcome::NoHello;
    };
    crate::ktrace::log_fmt(format_args!("agx: got mailbox msg0={:#018x} ep={:#x}", hello.msg0, hello.msg1));
    if hello.msg1 != EP_MGMT as u32 || mgmt_type(hello.msg0) != MGMT_MSG_HELLO {
        crate::ktrace::log("agx", "first message was not HELLO on EP_MGMT");
        return Outcome::BadHello;
    }
    let (min_ver, max_ver) = hello_versions(hello.msg0);
    let Some(ver) = negotiate_version(min_ver, max_ver) else {
        crate::ktrace::log_fmt(format_args!("agx: version mismatch (IOP [{min_ver},{max_ver}] vs [{RTKIT_MIN_VERSION},{RTKIT_MAX_VERSION}])"));
        return Outcome::VersionMismatch;
    };
    rep.version = ver;
    crate::ktrace::log_fmt(format_args!("agx: HELLO ok — booting RTKit version {ver}"));
    if !send(asc, msg_hello_ack(ver), EP_MGMT) {
        return Outcome::SendFail;
    }

    // --- endpoint map loop ---------------------------------------------------
    let mut eps = EndpointSet::default();
    loop {
        let Some(m) = asc.recv_blocking(STEP_TIMEOUT_MS, &mut pump) else {
            crate::ktrace::log("agx", "timeout waiting for endpoint map");
            return Outcome::EpmapFail;
        };
        if m.msg1 != EP_MGMT as u32 || mgmt_type(m.msg0) != MGMT_MSG_EPMAP {
            crate::ktrace::log_fmt(format_args!("agx: expected EPMAP, got msg0={:#018x} ep={:#x}", m.msg0, m.msg1));
            return Outcome::EpmapFail;
        }
        let em = epmap(m.msg0);
        crate::ktrace::log_fmt(format_args!("agx: EPMAP raw msg0={:#018x} bitmap={:#010x} base={} done={}", m.msg0, em.bitmap, em.base, em.done));
        eps.record_bitmap(em.bitmap, em.base);
        if !send(asc, msg_epmap_reply(em.base, em.done), EP_MGMT) {
            return Outcome::SendFail;
        }
        if em.done {
            break;
        }
    }
    rep.eps = eps;
    crate::ktrace::log_fmt(format_args!(
        "agx: endpoints — crashlog={} debug={} syslog={} ioreport={} oslog={}",
        eps.crashlog, eps.debug, eps.syslog, eps.ioreport, eps.oslog
    ));

    // --- start every present system endpoint ---------------------------------
    for ep in eps.to_start() {
        if !send(asc, msg_start_ep(ep), EP_MGMT) {
            return Outcome::SendFail;
        }
    }

    // --- pump until the IOP reaches power ON, servicing buffer requests -------
    // Total-budget wait: the IOP may take several seconds to finish its own init
    // (writing crashlog/syslog, powering rails) before it volunteers the power
    // ACK, so we poll in short windows against one overall deadline rather than
    // bailing on the first quiet gap. Every received message is logged so a
    // silent IOP (→ buffers unreachable, the GPU-UAT boundary) is distinguishable
    // from one sending something we mishandle.
    let mut state = RtkitState::default();
    let powerup_deadline = crate::arch::now_ms() + POWERUP_BUDGET_MS;
    let mut last_beat = crate::arch::now_ms();
    while state.iop_power != POWER_ON {
        if crate::arch::now_ms() >= powerup_deadline {
            crate::ktrace::log_fmt(format_args!(
                "agx: timed out after {POWERUP_BUDGET_MS}ms waiting for power ON (iop_power={:#x}, buffers granted={}) — IOP silent; likely needs GPU-UAT-addressable shared memory (Milestone 2)",
                state.iop_power, rep.n_buffers
            ));
            return Outcome::Timeout;
        }
        let Some(m) = asc.recv_blocking(POWERUP_POLL_MS, &mut pump) else {
            if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
                crate::ktrace::log("agx", "power-up wait cancelled (Ctrl+C)");
                return Outcome::Timeout;
            }
            // No message this window. Heartbeat ~1/s so the human sees progress,
            // then keep waiting until the overall deadline.
            let now = crate::arch::now_ms();
            if now.saturating_sub(last_beat) >= 1000 {
                last_beat = now;
                crate::ktrace::log_fmt(format_args!("agx: waiting for power ON… ({}ms left, iop_power={:#x})", powerup_deadline.saturating_sub(now), state.iop_power));
            }
            continue;
        };
        crate::ktrace::log_fmt(format_args!("agx: pump msg0={:#018x} ep={:#x}", m.msg0, m.msg1));
        let ep = m.msg1 as u8;
        if ep >= EP_APP_BASE {
            crate::ktrace::log_fmt(format_args!("agx: unexpected app-endpoint msg during boot (ep {ep:#x})"));
            continue;
        }
        match handle_system_msg(&mut state, m.msg0, ep) {
            Action::None => {}
            Action::Send(msg0, e) => {
                if !send(asc, msg0, e) {
                    return Outcome::SendFail;
                }
            }
            Action::AllocBuffer { ep, kind, n_pages, addr } => {
                crate::ktrace::log_fmt(format_args!("agx: buffer request ep={ep:#x} kind={kind:?} pages={n_pages} addr={addr:#x}"));
                if addr != 0 {
                    // IOP pre-allocated (SRAM/handoff) — record, no reply.
                    rep.n_buffers += 1;
                    if kind == BufferKind::Crashlog {
                        state.have_crashlog_buffer = true;
                    }
                } else {
                    let Some(phys) = alloc_shared(n_pages) else {
                        crate::ktrace::log("agx", "shared-buffer allocation failed (OOM)");
                        return Outcome::Timeout;
                    };
                    let dva = phys | DVA_BASE;
                    if !send(asc, msg_buffer_reply(n_pages, dva), ep) {
                        return Outcome::SendFail;
                    }
                    rep.n_buffers += 1;
                    if kind == BufferKind::Crashlog {
                        state.have_crashlog_buffer = true;
                    }
                }
            }
            Action::Crashed => {
                crate::ktrace::log("agx", "IOP crashed during boot (second crashlog buffer request)");
                return Outcome::Crashed;
            }
            Action::Unhandled => {
                crate::ktrace::log_fmt(format_args!("agx: unhandled system msg ep={ep:#x} msg0={:#018x}", m.msg0));
            }
        }
    }
    rep.iop_power = state.iop_power;

    // --- final: AP power → ON (RUNNING) --------------------------------------
    if !send(asc, msg_ap_pwr_state(POWER_ON), EP_MGMT) {
        return Outcome::SendFail;
    }
    rep.ap_power = POWER_ON;
    Outcome::Running
}

/// Bring up the AGX coprocessor. Gated on `is_apple()` + the `chitti.agx`
/// bootarg; a clean no-op otherwise. Returns the human-facing summary line.
fn up() -> Outcome {
    if !crate::arch::aarch64::is_apple() {
        crate::serial_println!("agx> not Apple Silicon — nothing to bring up");
        return Outcome::NoGpu;
    }
    if !bootarg_present(b"chitti.agx") {
        crate::serial_println!("agx> gated: add the `chitti.agx` bootarg on a BARE boot (never under the m1n1 hv)");
        return Outcome::NotRun;
    }
    let Some(base) = discover_asc_base() else {
        crate::serial_println!("agx> no `apple,agx-t8112` GPU node in the device tree");
        return Outcome::NoGpu;
    };
    ABORT.store(false, core::sync::atomic::Ordering::Relaxed);
    // Bare Apple boot has no serial; mirror the handshake trace to the chat pane.
    crate::ktrace::set_console_echo(true);
    crate::ktrace::log_fmt(format_args!("agx: GPU ASC base {base:#x}"));

    // PMGR gfx power on, then map the ASC MMIO window.
    if !pmgr_enable_gfx() {
        crate::serial_println!("agx> gfx power domain would not enable — aborting");
        return Outcome::SendFail;
    }
    crate::arch::aarch64::mmu::map_device_gib(base);

    let mut rep = Report { asc_base: base, ..Default::default() };
    // SAFETY: `base` is the FDT-discovered, Device-mapped ASC CPU window.
    let asc = unsafe { Asc::new(base as usize) };
    let outcome = rtkit_boot(&asc, &mut rep);
    rep.outcome = outcome;
    rep.cpu_running = asc.cpu_running();
    REPORT.with(|r| *r = rep);

    match outcome {
        Outcome::Running => crate::serial_println!("agx> coprocessor RUNNING — RTKit v{} up, {} endpoint(s) started", rep.version, count_eps(&rep.eps)),
        Outcome::NoHello => crate::serial_println!(
            "agx> NO HELLO after cpu_start — the gfx-asc control firmware is not resident.\n\
             agx> Milestone 1 is BLOCKED on Asahi GPU-firmware provisioning for this machine."
        ),
        Outcome::VersionMismatch => crate::serial_println!("agx> RTKit version mismatch — see ktrace"),
        Outcome::BadHello => crate::serial_println!("agx> first mailbox message was not HELLO — see ktrace"),
        Outcome::EpmapFail => crate::serial_println!("agx> endpoint-map exchange failed — see ktrace"),
        Outcome::SendFail => crate::serial_println!("agx> mailbox send failed (ASC stuck?) — see ktrace"),
        Outcome::Timeout => crate::serial_println!("agx> timed out reaching power ON — see ktrace"),
        Outcome::Crashed => crate::serial_println!("agx> IOP crashed during boot — see ktrace"),
        Outcome::NoGpu | Outcome::NotRun => {}
    }
    outcome
}

fn count_eps(e: &EndpointSet) -> u32 {
    e.crashlog as u32 + e.debug as u32 + e.syslog as u32 + e.ioreport as u32 + e.oslog as u32
}

/// `/agx status` — dump the last bring-up snapshot.
fn status() {
    let r = REPORT.with(|r| *r);
    crate::serial_println!("agx status:");
    crate::serial_println!("  outcome:     {:?}", r.outcome);
    crate::serial_println!("  asc base:    {:#x}", r.asc_base);
    crate::serial_println!("  cpu running: {}", r.cpu_running);
    crate::serial_println!("  rtkit ver:   {}", r.version);
    crate::serial_println!(
        "  endpoints:   crashlog={} debug={} syslog={} ioreport={} oslog={}",
        r.eps.crashlog, r.eps.debug, r.eps.syslog, r.eps.ioreport, r.eps.oslog
    );
    crate::serial_println!("  iop power:   {:#x}", r.iop_power);
    crate::serial_println!("  ap power:    {:#x}", r.ap_power);
    crate::serial_println!("  buffers:     {}", r.n_buffers);
}

/// Print one FDT property: name, byte length, and the value decoded as
/// big-endian u32 cells (addresses/phandles) plus an ASCII rendering when it
/// looks like a string list.
fn dump_prop(name: &[u8], val: &[u8]) {
    let n = core::str::from_utf8(name).unwrap_or("<name?>");
    // ASCII string-ish values (all printable/NUL) — show them as text.
    let stringish = !val.is_empty() && val.iter().all(|&b| b == 0 || (0x20..0x7f).contains(&b));
    if stringish && val.len() <= 96 {
        let s = core::str::from_utf8(&val[..val.len().saturating_sub(1)]).unwrap_or("");
        crate::serial_println!("  {n} ({} B) = \"{}\"", val.len(), s.replace('\0', "\",\""));
        return;
    }
    if val.len() % 4 == 0 && !val.is_empty() {
        // Decode as up to 16 BE u32 cells (pairs read as u64 for addresses).
        let mut line = alloc::string::String::new();
        use core::fmt::Write;
        let ncells = (val.len() / 4).min(16);
        for i in 0..ncells {
            let c = u32::from_be_bytes([val[i * 4], val[i * 4 + 1], val[i * 4 + 2], val[i * 4 + 3]]);
            let _ = write!(line, " {c:#x}");
        }
        crate::serial_println!("  {n} ({} B) =<{} >", val.len(), line);
    } else {
        crate::serial_println!("  {n} ({} B) = <{} bytes>", val.len(), val.len());
    }
}

/// `/agx sgx` — dump the AGX GPU node's every property + the resolved
/// `memory-region` carveouts (ttbs/pagetables/handoff/…) AS FILLED on the live
/// machine. This is the discovery step for Milestone 2's GPU-VA layout: it shows
/// whether m1n1 populated the `uat-*` carveouts (they are `reg=<0,0,0,0>` in the
/// static DTB) and surfaces any `rtkit-private-vm-region`-style property.
fn sgx_dump() {
    use alloc::vec::Vec;
    if !crate::arch::aarch64::is_apple() {
        crate::serial_println!("agx> not Apple Silicon");
        return;
    }
    let fdt = crate::arch::aarch64::boot::boot_x0();
    crate::serial_println!("agx sgx: GPU node ({})", core::str::from_utf8(AGX_COMPAT).unwrap_or("?"));
    // reg[0]=asc, reg[1]=sgx.
    for (i, nm) in [(0usize, "asc"), (1, "sgx")] {
        // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
        if let Some((b, s)) = unsafe { crate::fdt::reg_nth_of_compatible(fdt, AGX_COMPAT, i) } {
            crate::serial_println!("  reg[{i}] {nm}: base={b:#x} size={s:#x}");
        }
    }
    crate::serial_println!("agx sgx: properties —");
    let mut memregion: Vec<u32> = Vec::new();
    let mut memnames: Vec<u8> = Vec::new();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let found = unsafe {
        crate::fdt::for_each_prop_of_compatible(fdt, AGX_COMPAT, &mut |name, val| {
            dump_prop(name, val);
            if name == b"memory-region" {
                for ch in val.chunks_exact(4) {
                    memregion.push(u32::from_be_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
            } else if name == b"memory-region-names" {
                memnames = val.to_vec();
            }
        })
    };
    if !found {
        crate::serial_println!("agx sgx: no {} node in the device tree", core::str::from_utf8(AGX_COMPAT).unwrap_or("?"));
        return;
    }
    // Resolve each memory-region phandle → its carveout (base,size) as filled.
    if !memregion.is_empty() {
        crate::serial_println!("agx sgx: memory-region carveouts (resolved) —");
        let names: Vec<&[u8]> = memnames.split(|&b| b == 0).filter(|s| !s.is_empty()).collect();
        for (i, &ph) in memregion.iter().enumerate() {
            let nm = names.get(i).and_then(|s| core::str::from_utf8(s).ok()).unwrap_or("?");
            // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
            match unsafe { crate::fdt::reg_by_phandle(fdt, ph) } {
                Some((b, s)) => crate::serial_println!("  [{i}] {nm:14} phandle={ph:#x} base={b:#x} size={s:#x}"),
                None => crate::serial_println!("  [{i}] {nm:14} phandle={ph:#x} <unresolved>"),
            }
        }
    }
}

/// `/agx` command entry (aarch64).
pub fn command(arg: &str) {
    match arg.trim() {
        "" | "up" | "init" => {
            up();
        }
        "status" | "info" => status(),
        "sgx" | "dump" => sgx_dump(),
        other => {
            crate::serial_println!("agx> unknown subcommand '{other}' (try: up | status | sgx)");
        }
    }
}
