//! **CPU frequency / energy policy (P-state lite)**.
//!
//! Goal: avoid permanent “max effort” when the machine is idle or on battery.
//! This is not a full ACPI CPPC / intel_pstate stack — three human modes and a
//! small set of knobs the OS can actually write:
//!
//! | Mode | Intent |
//! |------|--------|
//! | `performance` | Prefer highest throughput (EPB = 0 on x86). |
//! | `powersave` | Prefer lowest energy (EPB = 15). |
//! | `auto` | Powersave when idle fraction is high, battery is discharging, or the pack is critical; else performance. |
//!
//! ## x86 path
//!
//! [`IA32_ENERGY_PERF_BIAS`](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
//! (MSR `0x1B0`) when CPUID.06H:ECX[3] is set. HWP / `_PSS` are later.
//!
//! ## aarch64 path
//!
//! No portable OPP control yet (Apple Silicon is special-cased elsewhere; PSCI
//! does not expose a standard “set OPP” for guests). [`set_mode`] records policy
//! and reports honestly; apply is a no-op until a platform path lands.
//!
//! ## Thermal
//!
//! Full ACPI `_TMP` zones are not wired yet. `auto` treats a **critical battery**
//! as a hard powersave signal (the one thermal-adjacent input we already have).

use core::sync::atomic::{AtomicU8, Ordering};

/// User-selected policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    Performance = 0,
    Powersave = 1,
    Auto = 2,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Performance => "performance",
            Mode::Powersave => "powersave",
            Mode::Auto => "auto",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "performance" | "perf" | "max" => Some(Mode::Performance),
            "powersave" | "save" | "eco" | "low" => Some(Mode::Powersave),
            "auto" | "balanced" | "default" => Some(Mode::Auto),
            _ => None,
        }
    }
}

/// Resolved target after policy inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Effective {
    Performance = 0,
    Powersave = 1,
}

impl Effective {
    pub fn as_str(self) -> &'static str {
        match self {
            Effective::Performance => "performance",
            Effective::Powersave => "powersave",
        }
    }
}

/// Snapshot of inputs for pure policy (unit-tested).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyInput {
    pub mode: Mode,
    /// 0–100: fraction of uptime spent in [`crate::power::idle::halt`].
    pub idle_pct: u8,
    pub battery_discharging: bool,
    /// Force powersave (critical battery today; ACPI `_TMP` later).
    pub thermal_hot: bool,
}

/// Idle fraction at or above this → `auto` picks powersave.
pub const AUTO_IDLE_PCT: u8 = 40;

/// Pure policy: map mode + sensors → effective target.
pub fn effective(input: PolicyInput) -> Effective {
    if input.thermal_hot {
        return Effective::Powersave;
    }
    match input.mode {
        Mode::Performance => Effective::Performance,
        Mode::Powersave => Effective::Powersave,
        Mode::Auto => {
            if input.battery_discharging || input.idle_pct >= AUTO_IDLE_PCT {
                Effective::Powersave
            } else {
                Effective::Performance
            }
        }
    }
}

// ── live state ───────────────────────────────────────────────────────────

static MODE: AtomicU8 = AtomicU8::new(Mode::Auto as u8);
static LAST_EFF: AtomicU8 = AtomicU8::new(0xff); // invalid → first tick always applies
static APPLIES: AtomicU64Counter = AtomicU64Counter::new();

/// Tiny atomic u64 without pulling more imports at the top for tests on no_std.
struct AtomicU64Counter(core::sync::atomic::AtomicU64);
impl AtomicU64Counter {
    const fn new() -> Self {
        Self(core::sync::atomic::AtomicU64::new(0))
    }
    fn fetch_add(&self, n: u64) -> u64 {
        self.0.fetch_add(n, Ordering::Relaxed)
    }
    fn load(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Current user mode.
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::Performance,
        1 => Mode::Powersave,
        _ => Mode::Auto,
    }
}

/// Set the user mode and apply immediately.
pub fn set_mode(m: Mode) -> Result<(), &'static str> {
    MODE.store(m as u8, Ordering::Relaxed);
    // Force re-apply even if effective is unchanged (user asked explicitly).
    LAST_EFF.store(0xff, Ordering::Relaxed);
    tick();
    if !backend_available() {
        return Err(backend_unavailable_reason());
    }
    Ok(())
}

/// Recompute policy and apply if the effective target changed.
pub fn tick() {
    let eff = effective(current_input());
    let prev = LAST_EFF.load(Ordering::Relaxed);
    if prev == eff as u8 {
        return;
    }
    apply(eff);
    LAST_EFF.store(eff as u8, Ordering::Relaxed);
    APPLIES.fetch_add(1);
}

/// Last applied effective mode (for `/power status`), or `None` before first tick.
pub fn last_effective() -> Option<Effective> {
    match LAST_EFF.load(Ordering::Relaxed) {
        0 => Some(Effective::Performance),
        1 => Some(Effective::Powersave),
        _ => None,
    }
}

/// How many times the backend was programmed since boot.
pub fn apply_count() -> u64 {
    APPLIES.load()
}

fn current_input() -> PolicyInput {
    let up = crate::arch::now_ms();
    let idle = crate::power::idle::idle_ms();
    let idle_pct = if up > 0 {
        (idle.saturating_mul(100) / up).min(100) as u8
    } else {
        0
    };
    let bat = crate::drivers::battery::cached();
    PolicyInput {
        mode: mode(),
        idle_pct,
        battery_discharging: bat.map(|b| b.discharging).unwrap_or(false),
        thermal_hot: bat.map(|b| b.critical).unwrap_or(false),
    }
}

// ── backend ──────────────────────────────────────────────────────────────

/// True when this arch can program a frequency/energy control.
pub fn backend_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        x86_epb_supported()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn backend_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_epb_supported() {
            "IA32_ENERGY_PERF_BIAS"
        } else {
            "none (CPUID.06H:ECX[3] clear)"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "none (no portable OPP yet)"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "none"
    }
}

fn backend_unavailable_reason() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "CPU does not expose ENERGY_PERF_BIAS (policy recorded only)"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "no frequency control on this aarch64 platform yet (policy recorded only)"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unsupported architecture"
    }
}

/// ENERGY_PERF_BIAS values (Intel SDM): 0 = performance, 15 = max powersave.
pub const EPB_PERFORMANCE: u64 = 0;
pub const EPB_POWERSAVE: u64 = 15;
/// Mid bias used only for documentation / tests (auto uses 0 or 15).
pub const EPB_BALANCE: u64 = 6;

const MSR_IA32_ENERGY_PERF_BIAS: u32 = 0x1b0;

fn apply(eff: Effective) {
    #[cfg(target_arch = "x86_64")]
    {
        if !x86_epb_supported() {
            return;
        }
        let bias = match eff {
            Effective::Performance => EPB_PERFORMANCE,
            Effective::Powersave => EPB_POWERSAVE,
        };
        // Preserve upper bits; EPB is in the low 4 bits.
        let cur = unsafe { rdmsr(MSR_IA32_ENERGY_PERF_BIAS) };
        let next = (cur & !0xf) | (bias & 0xf);
        // SAFETY: MSR is present per CPUID; writing EPB only changes the energy hint.
        unsafe { wrmsr(MSR_IA32_ENERGY_PERF_BIAS, next) };
        crate::ktrace::log_fmt(format_args!(
            "power.cpu: applied {} (EPB={bias})",
            eff.as_str()
        ));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = eff;
        // Policy is recorded; hardware path not available.
    }
}

/// Current EPB low nibble if the MSR is present, for status lines.
pub fn read_epb() -> Option<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if !x86_epb_supported() {
            return None;
        }
        // SAFETY: gated on CPUID.
        Some(unsafe { rdmsr(MSR_IA32_ENERGY_PERF_BIAS) } & 0xf)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn x86_epb_supported() -> bool {
    // CPUID leaf 6, ECX bit 3 = Energy Performance Bias preference.
    let ecx = cpuid_ecx(6);
    ecx & (1 << 3) != 0
}

#[cfg(target_arch = "x86_64")]
fn cpuid_ecx(leaf: u32) -> u32 {
    let ecx: u32;
    // SAFETY: CPUID has no memory side effects; leaf 6 exists on modern x86_64.
    unsafe {
        core::arch::asm!(
            "push rbx",
            "mov eax, {leaf:e}",
            "xor ecx, ecx",
            "cpuid",
            "mov {out:e}, ecx",
            "pop rbx",
            leaf = in(reg) leaf,
            out = out(reg) ecx,
            options(nostack, preserves_flags),
        );
    }
    ecx
}

#[cfg(target_arch = "x86_64")]
unsafe fn rdmsr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: caller ensures the MSR is valid on this CPU.
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") msr,
            out("eax") lo,
            out("edx") hi,
            options(nostack, preserves_flags),
        );
    }
    ((hi as u64) << 32) | lo as u64
}

#[cfg(target_arch = "x86_64")]
unsafe fn wrmsr(msr: u32, value: u64) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    // SAFETY: caller ensures the MSR is valid; only EPB bits are changed by apply.
    unsafe {
        core::arch::asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") lo,
            in("edx") hi,
            options(nostack, preserves_flags),
        );
    }
}

/// Multi-line status for `/power status`.
pub fn status_lines() -> alloc::vec::Vec<alloc::string::String> {
    use alloc::format;
    use alloc::string::String;
    let mut v = alloc::vec::Vec::new();
    let inp = current_input();
    let eff = effective(inp);
    v.push(format!("mode      {}", mode().as_str()));
    v.push(format!(
        "effective {} (idle {}%, battery discharging={}, critical={})",
        eff.as_str(),
        inp.idle_pct,
        inp.battery_discharging,
        inp.thermal_hot
    ));
    v.push(format!("backend   {}", backend_name()));
    if let Some(epb) = read_epb() {
        v.push(format!("EPB       {epb} (0=perf … 15=save)"));
    }
    v.push(format!("applies   {}", apply_count()));
    if !backend_available() {
        v.push(String::from(backend_unavailable_reason()));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn mode_parse_accepts_aliases() {
        assert_eq!(Mode::parse("performance"), Some(Mode::Performance));
        assert_eq!(Mode::parse("PERF"), Some(Mode::Performance));
        assert_eq!(Mode::parse("powersave"), Some(Mode::Powersave));
        assert_eq!(Mode::parse("eco"), Some(Mode::Powersave));
        assert_eq!(Mode::parse("auto"), Some(Mode::Auto));
        assert_eq!(Mode::parse("nope"), None);
    }

    #[test_case]
    fn performance_and_powersave_are_absolute() {
        let base = PolicyInput {
            mode: Mode::Performance,
            idle_pct: 99,
            battery_discharging: true,
            thermal_hot: false,
        };
        assert_eq!(effective(base), Effective::Performance);
        assert_eq!(
            effective(PolicyInput {
                mode: Mode::Powersave,
                idle_pct: 0,
                battery_discharging: false,
                thermal_hot: false
            }),
            Effective::Powersave
        );
    }

    #[test_case]
    fn auto_uses_idle_and_battery() {
        let cool = PolicyInput {
            mode: Mode::Auto,
            idle_pct: 10,
            battery_discharging: false,
            thermal_hot: false,
        };
        assert_eq!(effective(cool), Effective::Performance);
        assert_eq!(
            effective(PolicyInput {
                idle_pct: AUTO_IDLE_PCT,
                ..cool
            }),
            Effective::Powersave
        );
        assert_eq!(
            effective(PolicyInput {
                battery_discharging: true,
                idle_pct: 0,
                ..cool
            }),
            Effective::Powersave
        );
    }

    #[test_case]
    fn thermal_hot_overrides_performance_mode() {
        // Critical battery / future _TMP: even an explicit performance request
        // yields when the machine is thermally (or energy) critical.
        assert_eq!(
            effective(PolicyInput {
                mode: Mode::Performance,
                idle_pct: 0,
                battery_discharging: false,
                thermal_hot: true,
            }),
            Effective::Powersave
        );
    }

    #[test_case]
    fn epb_constants_match_intel_sdm_range() {
        assert_eq!(EPB_PERFORMANCE, 0);
        assert_eq!(EPB_POWERSAVE, 15);
        assert!(EPB_BALANCE < EPB_POWERSAVE);
    }
}
