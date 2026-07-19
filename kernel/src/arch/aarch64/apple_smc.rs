//! Minimal **Apple SMC** client (RTKit over ASC) for GPIO power keys.
//!
//! Used to assert WiFi/BT module power: key `gP0d` (GPIO 13) ← `0x800001`
//! (m1n1 `pcie_enable_devices.py` / Asahi `pwren-gpios = <&smc_gpio 13>`).
//!
//! Boot order matches **proxyclient `mgmt.py` / our AGX path**, not the generic
//! m1n1 `rtkit.c` wait-IOP-then-AP sequence. SMC (like AGX) stalls after
//! buffer grants until `SetAPPower(ON)` — waiting for IOP ON first deadlocks
//! with `buffer@ep4 pages=0` forever. Correct sequence:
//!
//! 1. IOP_PWR=INIT → HELLO → EPMAP (accumulate all fragments)
//! 2. START system endpoints (crashlog/syslog/debug/ioreport)
//! 3. **AP_PWR=ON immediately** (`boot_done`)
//! 4. Pump until BOTH iop and ap power == ON, ACKing ioreport 0x8/0xc
//! 5. START_EP 0x20 → INITIALIZE → shmem pointer
//! 6. WRITE_KEY into shmem
//!
//! SMC is SRAM-backed (`rtkit_init(..., sram=true)`): BUFFER_REQUESTs with a
//! non-zero addr are IOP-provided — **do not reply**.

use crate::agx::asc::{Asc, Message};
use crate::agx::proto::{
    self, EP_MGMT, MGMT_MSG_EPMAP, MGMT_MSG_HELLO, POWER_INIT, POWER_ON,
};

const SMC_COMPAT: &[&[u8]] = &[b"apple,t8112-smc", b"apple,smc"];

const SMC_EP: u8 = 0x20;
const SMC_INITIALIZE: u64 = 0x17;
const SMC_WRITE_KEY: u64 = 0x11;
const SMC_READ_KEY: u64 = 0x10;

/// FourCC `gP0d` (GPIO pin 13) — WiFi/BT module power enable.
pub const KEY_GP0D: u32 = u32::from_be_bytes(*b"gP0d");
/// FourCC `gP1a` (GPIO pin 26) — companion enable written by m1n1
/// `pcie_enable_devices.py` next to gP0d.
pub const KEY_GP1A: u32 = u32::from_be_bytes(*b"gP1a");
/// Power-on value (output enable + high). m1n1: `0x800001`.
pub const GPIO_ON: u32 = 0x8000_01;
pub const GPIO_OFF: u32 = 0x8000_00;
/// Simple flag/on for 1-byte-ish GPIO keys (m1n1 writes `1` to gP1a).
pub const GPIO_FLAG_ON: u32 = 1;

struct SmcState {
    ready: bool,
    asc_base: u64,
    shmem: u64,
    msgid: u8,
}

static SMC: crate::mm::Locked<SmcState> = crate::mm::Locked::new(SmcState {
    ready: false,
    asc_base: 0,
    shmem: 0,
    msgid: 0,
});

fn pump() -> bool {
    crate::shell::status_tick();
    crate::shell::poll_interrupt()
}

fn mdelay(ms: u64) {
    let end = crate::arch::now_ms() + ms;
    while crate::arch::now_ms() < end {
        let _ = pump();
        core::hint::spin_loop();
    }
}

fn discover_smc_base() -> Option<u64> {
    let fdt = super::boot::boot_x0();
    for &compat in SMC_COMPAT {
        // SAFETY: FDT from boot.
        if let Some((base, _)) = unsafe { crate::fdt::reg_of_compatible(fdt, compat) } {
            return Some(base);
        }
    }
    None
}

fn send(asc: &Asc, msg0: u64, ep: u8) -> bool {
    asc.send(&Message { msg0, msg1: ep as u32 }, 2000, &mut pump)
}

/// Handle a system-endpoint message during boot (m1n1 `rtkit_recv` switch +
/// proxyclient `ioreporting.py`).
///
/// - BUFFER_REQUEST with non-zero addr (SRAM): record only, **no reply**
/// - ioreport type 0x8 / 0xc: **must ACK** or the IOP never reaches power-ON
/// - syslog LOG: ACK by echo
fn handle_system_ep(asc: &Asc, ep: u8, msg0: u64) {
    let ty = proto::mgmt_type(msg0); // bits [59:52]
    match ep {
        // crashlog
        0x1 => {
            if ty == proto::MSG_BUFFER_REQUEST {
                let br = proto::buffer_request(msg0);
                crate::ktrace::log_fmt(format_args!(
                    "smc: crashlog buf pages={} addr={:#x} (SRAM, no reply)",
                    br.n_pages, br.addr
                ));
            } else {
                crate::ktrace::log_fmt(format_args!(
                    "smc: crashlog ty={ty:#x} msg0={msg0:#018x}"
                ));
            }
        }
        // syslog
        0x2 => match ty {
            proto::MSG_BUFFER_REQUEST => {
                let br = proto::buffer_request(msg0);
                crate::ktrace::log_fmt(format_args!(
                    "smc: syslog buf pages={} addr={:#x} (SRAM, no reply)",
                    br.n_pages, br.addr
                ));
            }
            proto::MSG_SYSLOG_INIT => {
                crate::ktrace::log("smc", "syslog INIT");
            }
            proto::MSG_SYSLOG_LOG => {
                let _ = send(asc, msg0, ep);
            }
            _ => {
                crate::ktrace::log_fmt(format_args!(
                    "smc: syslog unhandled ty={ty:#x} — echo"
                ));
                let _ = send(asc, msg0, ep);
            }
        },
        // debug
        0x3 => {
            if ty == proto::MSG_BUFFER_REQUEST {
                let br = proto::buffer_request(msg0);
                crate::ktrace::log_fmt(format_args!(
                    "smc: debug buf pages={} addr={:#x} (SRAM, no reply)",
                    br.n_pages, br.addr
                ));
            }
        }
        // ioreport — 0x8 (Report) and 0xc (Start) must be ACKed (m1n1 rtkit.c)
        0x4 => match ty {
            proto::MSG_BUFFER_REQUEST => {
                let br = proto::buffer_request(msg0);
                if br.addr != 0 {
                    crate::ktrace::log_fmt(format_args!(
                        "smc: ioreport buf pages={} addr={:#x} (SRAM, no reply)",
                        br.n_pages, br.addr
                    ));
                } else if br.n_pages != 0 {
                    // Non-SRAM path: IOP wants us to allocate. We have no DMA
                    // allocator here — log and skip (SMC uses SRAM in practice).
                    crate::ktrace::log_fmt(format_args!(
                        "smc: ioreport wants alloc pages={} — cannot (SRAM-only)",
                        br.n_pages
                    ));
                } else {
                    // pages=0 addr=0 with type 1 is odd; still echo as safety.
                    crate::ktrace::log("smc", "ioreport empty buffer req — echo");
                    let _ = send(asc, msg0, ep);
                }
            }
            0x8 | 0xc => {
                crate::ktrace::log_fmt(format_args!(
                    "smc: ioreport ACK ty={ty:#x} msg0={msg0:#018x}"
                ));
                let _ = send(asc, msg0, ep);
            }
            _ => {
                crate::ktrace::log_fmt(format_args!(
                    "smc: ioreport unhandled ty={ty:#x} msg0={msg0:#018x} — echo"
                ));
                let _ = send(asc, msg0, ep);
            }
        },
        // oslog — m1n1 logs only; some firmwares want 0x30 ACK of 0x10 INIT
        0x8 => {
            let ty_hi = (msg0 >> 0) & 0xff; // oslog uses low byte in some revs
            crate::ktrace::log_fmt(format_args!(
                "smc: oslog ty={ty:#x} low={ty_hi:#x} msg0={msg0:#018x}"
            ));
            if ty == 0x10 || (msg0 & 0xff) == 0x10 {
                // MSG_OSLOG_INIT → ACK with 0x30 (Linux/asahi)
                let _ = send(asc, 0x30, ep);
            }
        }
        _ => {
            crate::ktrace::log_fmt(format_args!(
                "smc: sys ep={ep:#x} ty={ty:#x} msg0={msg0:#018x}"
            ));
            if ty == 0x8 || ty == 0xc {
                let _ = send(asc, msg0, ep);
            }
        }
    }
}

/// Soft-reset ASC only when the outbox is stuck FULL before we can talk.
fn asc_soft_reset(asc: &Asc) {
    let (cpu, a2i, i2a) = asc.diag();
    crate::ktrace::log_fmt(format_args!(
        "smc: soft-reset (cpu={cpu:#x} a2i={a2i:#x} i2a={i2a:#x})"
    ));
    asc.cpu_stop();
    mdelay(10);
    let mut n = 0u32;
    while let Some(m) = asc.try_recv() {
        n += 1;
        if n > 64 {
            break;
        }
        let _ = m;
    }
    asc.mbox_enable();
    asc.cpu_start();
    mdelay(20);
}

/// Process one inbound message; update power flags. Returns true if it was
/// an app-endpoint message (caller may want the payload).
fn handle_msg(asc: &Asc, m: &Message, iop_on: &mut bool, ap_on: &mut bool) -> Option<u64> {
    let ep = m.msg1 as u8;
    if ep == EP_MGMT {
        let ty = proto::mgmt_type(m.msg0);
        if ty == 7 {
            // IOP_PWR_STATE_ACK — exact match (0x220 INIT must NOT count as ON)
            let st = proto::pwr_state(m.msg0);
            crate::ktrace::log_fmt(format_args!("smc: IOP_PWR ACK {st:#x}"));
            if st == POWER_ON {
                *iop_on = true;
            }
        } else if ty == 0xb {
            let st = proto::pwr_state(m.msg0);
            crate::ktrace::log_fmt(format_args!("smc: AP_PWR ACK {st:#x}"));
            if st == POWER_ON {
                *ap_on = true;
            }
        } else {
            crate::ktrace::log_fmt(format_args!(
                "smc: mgmt ty={ty:#x} msg0={:#018x}",
                m.msg0
            ));
        }
        None
    } else if ep == SMC_EP {
        Some(m.msg0)
    } else if ep < 0x20 {
        handle_system_ep(asc, ep, m.msg0);
        None
    } else {
        crate::ktrace::log_fmt(format_args!(
            "smc: app ep={ep:#x} msg0={:#018x}",
            m.msg0
        ));
        None
    }
}

/// m1n1/proxyclient RTKit boot for SMC, then INITIALIZE → shmem.
///
/// Marker string `smc: boot v2` appears in ktrace so a stale binary is obvious.
fn rtkit_wake_and_init(asc: &Asc) -> Option<u64> {
    crate::ktrace::log("smc", "boot v2 (AP-first, ioreport ACK)");

    asc.mbox_enable();
    if !asc.cpu_running() {
        asc.cpu_start();
    }
    mdelay(5);

    // If the outbox is stuck FULL (asleep IOP with a stale queue), soft-reset.
    if !asc.can_send() {
        crate::ktrace::log("smc", "A2I FULL at start — soft-reset");
        asc_soft_reset(asc);
    }

    crate::ktrace::log("smc", "sending IOP_PWR=INIT");
    if !send(asc, proto::msg_iop_pwr_state(POWER_INIT), EP_MGMT) {
        crate::ktrace::log("smc", "IOP INIT send failed — soft-reset retry");
        asc_soft_reset(asc);
        if !send(asc, proto::msg_iop_pwr_state(POWER_INIT), EP_MGMT) {
            crate::ktrace::log("smc", "IOP INIT send failed after reset");
            return None;
        }
    }

    // --- Phase A: HELLO + full EPMAP ----------------------------------------
    let mut sys_eps = [false; 0x20];
    let mut epmap_done = false;
    let mut iop_on = false;
    let mut ap_on = false;
    let deadline_epmap = crate::arch::now_ms() + 3000;

    while crate::arch::now_ms() < deadline_epmap && !epmap_done {
        let Some(m) = asc.recv_blocking(150, &mut pump) else {
            if pump() {
                return None;
            }
            continue;
        };
        let ep = m.msg1 as u8;
        if ep == EP_MGMT {
            let ty = proto::mgmt_type(m.msg0);
            if ty == MGMT_MSG_HELLO {
                let (min_v, max_v) = proto::hello_versions(m.msg0);
                let ver = proto::negotiate_version(min_v, max_v).unwrap_or(12);
                crate::ktrace::log_fmt(format_args!(
                    "smc: HELLO v{ver} (iop [{min_v},{max_v}])"
                ));
                let _ = send(asc, proto::msg_hello_ack(ver), EP_MGMT);
            } else if ty == MGMT_MSG_EPMAP {
                let em = proto::epmap(m.msg0);
                crate::ktrace::log_fmt(format_args!(
                    "smc: EPMAP base={} bitmap={:#x} done={}",
                    em.base, em.bitmap, em.done
                ));
                for bit in 0..32u8 {
                    if em.bitmap & (1u32 << bit) != 0 {
                        let ep_idx = (em.base as u32 * 32 + bit as u32) as u8;
                        if (ep_idx as usize) < 0x20 {
                            sys_eps[ep_idx as usize] = true;
                        } else {
                            crate::ktrace::log_fmt(format_args!(
                                "smc: app endpoint {ep_idx:#x} advertised"
                            ));
                        }
                    }
                }
                let _ = send(asc, proto::msg_epmap_reply(em.base, em.done), EP_MGMT);
                if em.done {
                    epmap_done = true;
                }
            } else {
                let _ = handle_msg(asc, &m, &mut iop_on, &mut ap_on);
            }
        } else {
            let _ = handle_msg(asc, &m, &mut iop_on, &mut ap_on);
        }
    }
    if !epmap_done {
        crate::ktrace::log("smc", "EPMAP not completed");
        return None;
    }

    // --- Phase B: START system endpoints (NOT app 0x20 yet) -----------------
    for ep in 1u8..0x20 {
        if sys_eps[ep as usize] {
            crate::ktrace::log_fmt(format_args!("smc: START_EP system {ep:#x}"));
            let _ = send(asc, proto::msg_start_ep(ep), EP_MGMT);
        }
    }

    // --- Phase C: AP power ON *immediately* (boot_done) ---------------------
    // Same ordering as AGX / proxyclient mgmt.boot_done(). Waiting for IOP ON
    // first deadlocks SMC after the buffer grants.
    crate::ktrace::log("smc", "sending AP_PWR=ON (boot_done)");
    if !send(asc, proto::msg_ap_pwr_state(POWER_ON), EP_MGMT) {
        crate::ktrace::log("smc", "AP_PWR send failed");
        return None;
    }

    // --- Phase D: pump until BOTH powers ON, servicing system messages ------
    crate::ktrace::log("smc", "waiting for iop+ap power ON…");
    let deadline_pwr = crate::arch::now_ms() + 5000;
    while !(iop_on && ap_on) && crate::arch::now_ms() < deadline_pwr {
        let Some(m) = asc.recv_blocking(100, &mut pump) else {
            if pump() {
                return None;
            }
            continue;
        };
        let _ = handle_msg(asc, &m, &mut iop_on, &mut ap_on);
    }
    crate::ktrace::log_fmt(format_args!(
        "smc: power state iop_on={iop_on} ap_on={ap_on}"
    ));
    if !iop_on || !ap_on {
        crate::ktrace::log("smc", "power ON incomplete — continuing to START 0x20 anyway");
    } else {
        crate::ktrace::log("smc", "iop+ap power ON");
    }

    // --- Phase E: START SMC app endpoint + INITIALIZE ----------------------
    crate::ktrace::log("smc", "START_EP 0x20");
    if !send(asc, proto::msg_start_ep(SMC_EP), EP_MGMT) {
        crate::ktrace::log("smc", "START_EP 0x20 send failed");
        return None;
    }

    // Drain any late system messages for a short settle window.
    let settle = crate::arch::now_ms() + 300;
    while crate::arch::now_ms() < settle {
        if let Some(m) = asc.try_recv() {
            if let Some(payload) = handle_msg(asc, &m, &mut iop_on, &mut ap_on) {
                if payload >= 0x10_0000 {
                    crate::ktrace::log_fmt(format_args!("smc: early shmem={payload:#x}"));
                    return Some(payload);
                }
            }
        } else {
            mdelay(5);
        }
    }

    crate::ktrace::log("smc", "sending INITIALIZE");
    // msgid 0 is fine for the first command (m1n1/Python do the same).
    if !send(asc, SMC_INITIALIZE, SMC_EP) {
        crate::ktrace::log("smc", "INITIALIZE send failed");
        return None;
    }

    let shmem_deadline = crate::arch::now_ms() + 3000;
    while crate::arch::now_ms() < shmem_deadline {
        let Some(m) = asc.recv_blocking(100, &mut pump) else {
            if pump() {
                return None;
            }
            continue;
        };
        crate::ktrace::log_fmt(format_args!(
            "smc: rx ep={:#x} msg0={:#018x}",
            m.msg1, m.msg0
        ));
        if let Some(payload) = handle_msg(asc, &m, &mut iop_on, &mut ap_on) {
            if payload >= 0x10_0000 {
                return Some(payload);
            }
        }
    }
    crate::ktrace::log("smc", "no shmem from INITIALIZE");
    None
}

/// Bring up the SMC RTKit endpoint and obtain the shared-memory pointer.
pub fn init() -> bool {
    if SMC.with(|s| s.ready) {
        return true;
    }
    if !super::is_apple() {
        return false;
    }
    let Some(base) = discover_smc_base() else {
        crate::ktrace::log("smc", "no apple,smc node in FDT");
        return false;
    };
    crate::arch::aarch64::mmu::map_device_gib(base);
    crate::arch::aarch64::mmu::map_device_gib(base + 0x8000);
    let fdt = super::boot::boot_x0();
    // SAFETY: FDT from boot; second reg is SMC SRAM window on t8112.
    if let Some((sram, _)) =
        unsafe { crate::fdt::reg_nth_of_compatible(fdt, b"apple,t8112-smc", 1) }
    {
        crate::arch::aarch64::mmu::map_device_gib(sram);
        crate::ktrace::log_fmt(format_args!("smc: SRAM {sram:#x}"));
    }
    crate::ktrace::log_fmt(format_args!("smc: ASC base {base:#x}"));

    // SAFETY: FDT-discovered, Device-mapped ASC window.
    let asc = unsafe { Asc::new(base as usize) };
    let Some(shmem) = rtkit_wake_and_init(&asc) else {
        return false;
    };
    crate::ktrace::log_fmt(format_args!("smc: shmem={shmem:#x}"));
    crate::arch::aarch64::mmu::map_device_gib(shmem);

    SMC.with(|s| {
        s.ready = true;
        s.asc_base = base;
        s.shmem = shmem;
        s.msgid = 1;
    });
    crate::ktrace::log("smc", "ready");
    true
}

/// Write a 32-bit SMC key (FourCC).
pub fn write_u32(key: u32, value: u32) -> bool {
    if !SMC.with(|s| s.ready) && !init() {
        return false;
    }
    let (base, shmem, id) = SMC.with(|s| {
        let id = s.msgid & 0xf;
        s.msgid = s.msgid.wrapping_add(1);
        (s.asc_base, s.shmem, id)
    });
    // SAFETY: shmem from INITIALIZE; mapped Device/Normal as needed.
    unsafe {
        core::ptr::write_volatile(shmem as *mut u32, value);
        core::arch::asm!(
            "dc cvac, {p}",
            "dsb sy",
            p = in(reg) shmem,
            options(nostack, preserves_flags)
        );
    }
    let msg0 = SMC_WRITE_KEY
        | ((id as u64) << 12)
        | ((4u64) << 16)
        | ((key as u64) << 32);
    // SAFETY: ASC base from init.
    let asc = unsafe { Asc::new(base as usize) };
    if !send(&asc, msg0, SMC_EP) {
        crate::ktrace::log("smc", "WRITE_KEY send failed");
        return false;
    }
    let mut iop_on = true;
    let mut ap_on = true;
    let deadline = crate::arch::now_ms() + 2000;
    while crate::arch::now_ms() < deadline {
        let Some(m) = asc.recv_blocking(50, &mut pump) else {
            if pump() {
                return false;
            }
            continue;
        };
        if let Some(payload) = handle_msg(&asc, &m, &mut iop_on, &mut ap_on) {
            let rid = ((payload >> 12) & 0xf) as u8;
            let result = payload & 0xff;
            crate::ktrace::log_fmt(format_args!(
                "smc: WRITE reply id={rid} result={result} msg0={payload:#018x}"
            ));
            if rid == id {
                if result != 0 {
                    crate::ktrace::log_fmt(format_args!(
                        "smc: WRITE_KEY key={key:#x} err={result}"
                    ));
                    return false;
                }
                return true;
            }
        }
    }
    crate::ktrace::log("smc", "WRITE_KEY timeout");
    false
}

/// Power the WiFi/BT module on (GPIO 13 / `gP0d`, plus `gP1a` like m1n1).
pub fn wifi_power_on() -> bool {
    crate::ktrace::log("smc", "asserting WiFi power gP0d=0x800001");
    if !write_u32(KEY_GP0D, GPIO_ON) {
        crate::ktrace::log("smc", "gP0d write failed");
        return false;
    }
    crate::ktrace::log("smc", "gP0d write OK");
    // Companion rail/enable used by m1n1 pcie_enable_devices.py. Non-fatal if
    // the key is missing on some boards.
    if write_u32(KEY_GP1A, GPIO_FLAG_ON) {
        crate::ktrace::log("smc", "gP1a write OK");
    } else {
        crate::ktrace::log("smc", "gP1a write skipped/failed (non-fatal)");
    }
    mdelay(150);
    true
}

/// Power the WiFi/BT module **off** (GPIO 13 / `gP0d` output-low).
pub fn wifi_power_off() -> bool {
    crate::ktrace::log("smc", "de-asserting WiFi power gP0d=0x800000");
    let ok = write_u32(KEY_GP0D, GPIO_OFF);
    mdelay(50);
    ok
}

/// **Full power-cycle** of the WiFi/BT module: rail off, settle, rail on. Unlike
/// PERST# (which resets the PCIe link but leaves the dongle PMU's resource state
/// intact), a rail cycle cold-boots the chip so its PMU reloads defaults with the
/// SYS_MEM/RAM domain **powered** — required for TCM/firmware download on a bare
/// boot where nothing power-cycled the module.
pub fn wifi_power_cycle() -> bool {
    crate::ktrace::log("smc", "power-cycling WiFi module (gP0d off→on)");
    let _ = wifi_power_off();
    mdelay(250);
    wifi_power_on()
}

#[allow(dead_code)]
pub fn read_u32(key: u32) -> Option<u32> {
    if !SMC.with(|s| s.ready) && !init() {
        return None;
    }
    let (base, id) = SMC.with(|s| {
        let id = s.msgid & 0xf;
        s.msgid = s.msgid.wrapping_add(1);
        (s.asc_base, id)
    });
    let msg0 = SMC_READ_KEY | ((id as u64) << 12) | ((4u64) << 16) | ((key as u64) << 32);
    // SAFETY: ASC base from init.
    let asc = unsafe { Asc::new(base as usize) };
    if !send(&asc, msg0, SMC_EP) {
        return None;
    }
    let mut iop_on = true;
    let mut ap_on = true;
    let deadline = crate::arch::now_ms() + 1000;
    while crate::arch::now_ms() < deadline {
        let Some(m) = asc.recv_blocking(50, &mut pump) else {
            continue;
        };
        if let Some(payload) = handle_msg(&asc, &m, &mut iop_on, &mut ap_on) {
            let rid = ((payload >> 12) & 0xf) as u8;
            if rid == id {
                if (payload & 0xff) != 0 {
                    return None;
                }
                return Some(((payload >> 32) & 0xffff_ffff) as u32);
            }
        }
    }
    None
}
