//! **What the determinism boundary costs** — a microbenchmark for the Synapse
//! gate chain (`/bench synapse`).
//!
//! The security argument for putting the boundary in the kernel is only
//! interesting if crossing it is cheap relative to the thing it guards. A token
//! of inference costs tens of milliseconds (`/perf`); the claim this module
//! exists to check is that the whole four-gate authorization decision costs
//! *microseconds*, i.e. that there is no security/performance tradeoff here to
//! argue about. It is the harness behind experiment E1 in `paper/`.
//!
//! Method, and why it is shaped this way — three of these are corrections, and
//! each was a wrong number before it was a rule:
//!
//! * Timing is **amortized over a batch**, sized by doubling until the batch
//!   takes at least [`TARGET_MS`]. `arch::now_ms` has millisecond granularity on
//!   both arches (1 kHz PIT/APIC on x86, the generic timer on aarch64), which is
//!   six orders of magnitude coarser than one gate — so a single call cannot be
//!   timed, and nothing here tries to. `arch::cycle_count` (TSC / `CNTVCT_EL0`)
//!   is read across the same batch as a cross-check; it counts **constant-rate
//!   ticks, not CPU cycles** ([`ticks_per_ms`]), and calling them cycles invites
//!   a reader to divide by a GHz figure and land two orders of magnitude out.
//! * Every batch is preceded by a **warm-up** ([`WARMUP_ITERS`]) and the four
//!   gate prefixes share **one** batch size. Neither was true at first, and the
//!   cumulative curve came out *decreasing* (gates 1..1 slower than gates 1..3),
//!   because the first batch paid to grow the heap: the grammar allocates a
//!   `String` per string argument and the scope gate normalizes a path into
//!   another. Differencing batches of different sizes and heap states produced
//!   marginal costs that were pure artifact.
//! * Every timed call goes through `core::hint::black_box`. Without it the FNV
//!   hash row reported **0 ns over 16.7M iterations** — the loop had been
//!   eliminated as dead code, and a free operation and a deleted one print
//!   identically. [`Row::is_suspect`] now says so out loud.
//! * Measurement runs through [`super::executor::gate_prefix`], which applies
//!   the real gate predicates in the real order but executes no primitive and
//!   writes no audit entry. That keeps the number honest (it prices the
//!   authorization decision, not `console_write`'s UART) and keeps a
//!   million-iteration loop from mutating the store or appending a million audit
//!   records.
//! * The benchmark's authority is a **synthetic parked task** with an explicit
//!   capability table and scope ledger, killed at the end. Measuring against the
//!   shell agent's own table would make the numbers a function of whatever the
//!   session happens to hold, and — worse — pricing a *denied* call would mean
//!   granting rights to an agent as a side effect of a benchmark.
//!
//! The arithmetic (batch → per-call figures, cumulative → per-gate deltas) is
//! pure and unit-tested; only the loop that drives it needs a machine.

use super::executor::{self, GATE_CAPABILITY, GATE_COUNT, GATE_GRAMMAR, GATE_SCOPE, GATE_TAINT};
use super::{audit, registry};
use crate::agent::types::{CapDomain, CapabilityRequest, Rights, Scope};
use crate::cap::{self, Right};
use crate::sched::{self, TaskId};
use crate::security::{Justification, Provenance};
use alloc::vec::Vec;

/// A batch must run at least this long for the millisecond clock to be worth
/// reading (60 ms => ~1.7% quantization error from a 1 ms tick).
pub const TARGET_MS: u64 = 60;

/// Ceiling on batch size, so a pathologically cheap operation (an FNV hash at
/// ~20 ns) terminates instead of doubling toward the heat death of the machine.
pub const MAX_ITERS: u64 = 1 << 24;

/// Batch size for the one measurement that has a side effect per iteration
/// (appending an audit entry). Fixed and small for two reasons: 16M entries
/// would be ~800 MB of log, and the log is a security artifact — a benchmark
/// should leave a footnote in it, not a chapter. The entries are recorded under
/// the name `<bench>`, following the `<malformed>` convention for a record that
/// names no primitive, so nothing reading the log later mistakes them for
/// invocations. There is no delete path (by design), so this cannot be cleaned
/// up afterwards; that is the reason for the small number.
pub const AUDIT_ITERS: u64 = 1024;

/// One timed row: what was measured, over how many iterations, and how long that
/// took by both clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub label: &'static str,
    /// How many gates were run (`0` for rows that are not a gate prefix).
    pub upto: u8,
    /// The gate that refused during measurement, or 0 if the call passed every
    /// gate that ran. Reported because a row that stops early is measuring less
    /// work than its label implies, and a silently-denied benchmark row is a
    /// wrong number rather than a missing one.
    pub stop: u8,
    pub iters: u64,
    pub ms: u64,
    /// Constant-rate counter ticks (TSC / `CNTVCT_EL0`) across the same batch —
    /// **not** CPU cycles. See [`ticks_per_ms`].
    pub ticks: u64,
}

impl Row {
    pub fn ns_per_call(&self) -> u64 {
        per_call_ns(self.ms, self.iters)
    }

    pub fn ticks_per_call(&self) -> u64 {
        per_call(self.ticks, self.iters)
    }

    /// Whether this row should be disbelieved rather than reported. Two cases,
    /// both of which mean the optimizer got to the loop (or the clock is not
    /// advancing) rather than that the work is fast:
    ///
    /// * no time elapsed at all, and
    /// * a per-call figure that rounds to **zero nanoseconds** — sub-nanosecond
    ///   means under a cycle, which nothing that touches memory achieves.
    ///
    /// The second case is why this is not just `ms == 0`: the FNV row once
    /// reported 6 ms over 16.7M iterations, which is 0.36 ns/call. That printed
    /// as a plausible `0 ns/call` while actually being a hoisted call the loop
    /// never made.
    pub fn is_suspect(&self) -> bool {
        self.ms == 0 || self.ns_per_call() == 0
    }
}

/// Nanoseconds per call from a batch's millisecond total. `ms * 1e6` cannot
/// overflow for any batch a human waits for.
pub fn per_call_ns(ms: u64, iters: u64) -> u64 {
    per_call(ms.saturating_mul(1_000_000), iters)
}

/// Integer per-iteration average, rounded to nearest, with a zero-iteration
/// guard (a cancelled or clamped run reports 0 rather than dividing by zero).
pub fn per_call(total: u64, iters: u64) -> u64 {
    if iters == 0 {
        return 0;
    }
    (total + iters / 2) / iters
}

/// Convert a counter-tick count to nanoseconds using the ticks-per-millisecond
/// factor observed in a long row. Used for the audit row, whose fixed iteration
/// count is far too short for the millisecond clock but ample for the counter.
/// Returns `None` when no factor could be established (a cancelled run).
pub fn ticks_to_ns(ticks: u64, ticks_per_ms: u64) -> Option<u64> {
    if ticks_per_ms == 0 {
        return None;
    }
    Some(per_call(ticks.saturating_mul(1_000_000), ticks_per_ms))
}

/// Counter ticks per millisecond, from a row long enough to have a meaningful
/// `ms`. This is also the number that tells the reader what a "tick" is: it is
/// **not** a CPU cycle. `CNTVCT_EL0` is a fixed ~24 MHz architectural counter on
/// Apple silicon and 62.5 MHz under QEMU, and the x86 TSC runs at a nominal
/// invariant rate — so ticks are a constant-rate cross-check on the millisecond
/// clock, not a measure of work done, and a per-call tick figure is only as fine
/// as the rate printed beside it (~42 ns at 24 MHz).
pub fn ticks_per_ms(rows: &[Row]) -> u64 {
    rows.iter()
        .filter(|r| r.ms >= TARGET_MS / 2 && r.ticks > 0)
        .map(|r| per_call(r.ticks, r.ms))
        .next()
        .unwrap_or(0)
}

/// Turn cumulative gate-prefix rows into the marginal cost of each gate.
///
/// The rows must be ordered by `upto` ascending, each measuring gates `1..=upto`
/// of the *same* call **at the same iteration count** — otherwise the differences
/// are between batches with different heap and cache states and mean nothing.
///
/// `None` for a gate means the difference was non-positive: the longer prefix
/// measured no slower than the shorter one, so this gate's cost is below the
/// noise floor of the method. That is reported as unmeasurable rather than as
/// zero, because "+0 ns" reads as a claim that a gate is free, and a saturating
/// subtraction is not evidence for it.
pub fn gate_deltas(cumulative: &[Row]) -> Vec<(u8, Option<u64>)> {
    let mut out = Vec::new();
    let mut prev = 0u64;
    for r in cumulative {
        let ns = r.ns_per_call();
        out.push((r.upto, if ns > prev { Some(ns - prev) } else { None }));
        prev = prev.max(ns);
    }
    out
}

/// Human name for a gate index, for the printed table.
pub fn gate_name(gate: u8) -> &'static str {
    match gate {
        0 => "passed",
        GATE_GRAMMAR => "grammar",
        GATE_CAPABILITY => "capability",
        GATE_TAINT => "taint",
        GATE_SCOPE => "scope",
        _ => "?",
    }
}

/// Iterations run and discarded before each timed batch. The gate path allocates
/// (the grammar builds a `String` per string argument, the scope gate normalizes
/// a path into another), and the kernel's first-fit allocator makes the *first*
/// batch pay for growing the heap. Without this, cumulative prefixes came out
/// **non-monotonic** — gates 1..1 measured slower than gates 1..3 — which made
/// every marginal cost after the first meaningless.
pub const WARMUP_ITERS: u64 = 8192;

/// Time one batch of exactly `iters` calls, after a warm-up. The timed region
/// contains nothing but the calls: upkeep and the Ctrl+C poll happen around it,
/// so the command stays interruptible (the standing rule) without upkeep landing
/// inside a measurement.
fn measure_at(label: &'static str, upto: u8, iters: u64, f: &mut impl FnMut() -> u8) -> Row {
    for _ in 0..WARMUP_ITERS {
        core::hint::black_box(f());
    }
    let t0 = crate::arch::now_ms();
    let c0 = crate::arch::cycle_count();
    let mut stop = 0u8;
    for _ in 0..iters {
        // black_box so a pure, result-discarded call cannot be optimized away.
        // Without it the FNV row reported 0 ns over 16.7M iterations — the loop
        // had been eliminated entirely, and a free operation and a deleted one
        // look identical in the output.
        stop = core::hint::black_box(f());
    }
    let ms = crate::arch::now_ms().saturating_sub(t0);
    let ticks = crate::arch::cycle_count().saturating_sub(c0);
    Row { label, upto, stop, iters, ms, ticks }
}

/// Find a batch size whose run takes at least [`TARGET_MS`], by doubling. Returns
/// `None` if the user cancelled.
fn calibrate(f: &mut impl FnMut() -> u8) -> Option<u64> {
    let mut iters: u64 = 1024;
    loop {
        let t0 = crate::arch::now_ms();
        for _ in 0..iters {
            core::hint::black_box(f());
        }
        let ms = crate::arch::now_ms().saturating_sub(t0);
        if ms >= TARGET_MS || iters >= MAX_ITERS {
            return Some(iters);
        }
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            return None;
        }
        iters *= 2;
    }
}

/// Calibrate, then time one batch. For rows measured on their own; the gate
/// prefixes share one batch size instead (see [`run`]).
fn measure(label: &'static str, upto: u8, mut f: impl FnMut() -> u8) -> Option<Row> {
    let iters = calibrate(&mut f)?;
    crate::shell::upkeep();
    if crate::shell::poll_interrupt() {
        return None;
    }
    Some(measure_at(label, upto, iters, &mut f))
}

/// The measurement subject: a task whose authority is exactly what the benchmark
/// says it is. Killed by [`teardown`], which revokes the table and the ledger.
struct Subject {
    /// Holds primitives *and* a narrow FS scope ledger — the enforced path.
    scoped: TaskId,
    /// Holds the same primitives with **no** ledger entry — the
    /// deny-only-when-recorded path most tasks actually take.
    unscoped: TaskId,
}

/// Primitives the subject holds. `net_http_post` is deliberately absent: pricing
/// a capability denial needs a primitive that is genuinely not held, and the one
/// thing a benchmark must not do is widen an agent's authority to measure it.
const GRANTED: &[u16] = &[registry::LIST, registry::MEM_FS_READ, registry::MEM_FS_WRITE, registry::MEM_FS_DELETE];

fn setup() -> Subject {
    let scoped = sched::spawn_parked("synapse-bench-scoped");
    let unscoped = sched::spawn_parked("synapse-bench-unscoped");
    for &p in GRANTED {
        cap::grant(scoped, Right::InvokePrimitive(p));
        cap::grant(unscoped, Right::InvokePrimitive(p));
    }
    cap::grant_scopes(
        scoped,
        &[CapabilityRequest::new(
            CapDomain::Fs,
            Rights::READ | Rights::WRITE | Rights::LIST | Rights::DELETE,
            Scope::Path(alloc::string::String::from("/bench/**")),
        )],
    );
    Subject { scoped, unscoped }
}

fn teardown(s: Subject) {
    let _ = sched::kill(s.scoped);
    let _ = sched::kill(s.unscoped);
}

/// Representative call that clears all four gates: a scoped-path read, so the
/// scope gate does real work (normalize + ledger walk) rather than being skipped.
const READ_IN_SCOPE: &str = r#"{"name":"mem_fs_read","arguments":{"path":"/bench/probe.txt"}}"#;

/// `/bench synapse` — price the gate chain. Prints one line per row plus the
/// per-gate marginal costs. Returns `false` if the user cancelled.
pub fn run() -> bool {
    let subject = setup();
    let trusted = Justification::trusted();
    let tainted = Justification::from_context(Provenance::UntrustedIngested);

    crate::serial_println!(
        "bench> synapse gate chain: {} gates, {} primitives ({} destructive), batch >= {} ms",
        GATE_COUNT,
        registry::REGISTRY.len(),
        registry::REGISTRY.iter().filter(|p| p.effect.is_effectful()).count(),
        TARGET_MS
    );

    // --- cumulative prefixes of the same call: 1, 1-2, 1-3, 1-4 ---
    //
    // All four share ONE batch size, calibrated on the cheapest prefix (gates
    // 1..1, so every row still clears TARGET_MS). Differencing rows measured at
    // different iteration counts compares different heap and cache states, which
    // is how the first version produced a *decreasing* cumulative curve.
    let Some(iters) = calibrate(&mut || executor::gate_prefix(subject.scoped, READ_IN_SCOPE, trusted, GATE_GRAMMAR)) else {
        teardown(subject);
        crate::serial_println!("bench> cancelled");
        return false;
    };
    let mut prefixes: Vec<Row> = Vec::new();
    for upto in [GATE_GRAMMAR, GATE_CAPABILITY, GATE_TAINT, GATE_SCOPE] {
        let label = match upto {
            GATE_GRAMMAR => "gates 1..1   (grammar)",
            GATE_CAPABILITY => "gates 1..2   (+capability)",
            GATE_TAINT => "gates 1..3   (+taint)",
            _ => "gates 1..4   (+scope, enforced)",
        };
        let row = measure_at(label, upto, iters, &mut || {
            executor::gate_prefix(subject.scoped, READ_IN_SCOPE, trusted, upto)
        });
        report(&row);
        prefixes.push(row);
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            teardown(subject);
            crate::serial_println!("bench> cancelled");
            return false;
        }
    }

    // --- variants, all four gates ---
    let variants: &[(&'static str, TaskId, &'static str, Justification)] = &[
        ("all gates, no scope ledger", subject.unscoped, READ_IN_SCOPE, trusted),
        ("all gates, no-arg call", subject.scoped, r#"{"name":"list","arguments":{}}"#, trusted),
        ("refused: malformed", subject.scoped, r#"{"name":"exfiltrate","arguments":{}}"#, trusted),
        (
            "refused: no capability",
            subject.scoped,
            r#"{"name":"net_http_post","arguments":{"url":"http://x/","body":"y"}}"#,
            trusted,
        ),
        (
            "refused: tainted destructive",
            subject.scoped,
            r#"{"name":"mem_fs_delete","arguments":{"path":"/bench/probe.txt"}}"#,
            tainted,
        ),
        (
            "refused: outside scope",
            subject.scoped,
            r#"{"name":"mem_fs_read","arguments":{"path":"/etc/shadow"}}"#,
            trusted,
        ),
    ];
    let mut rows = prefixes.clone();
    for &(label, task, raw, j) in variants {
        let Some(row) = measure(label, GATE_COUNT, || executor::gate_prefix(task, raw, j, GATE_COUNT)) else {
            teardown(subject);
            crate::serial_println!("bench> cancelled");
            return false;
        };
        report(&row);
        rows.push(row);
    }

    // --- the two costs the gate chain adds around itself ---
    // Both ends need the barrier. Wrapping only the *result* left the row at
    // 0.36 ns/call — a pure function of a `const` input is hoisted out of the
    // loop entirely, so the black_box was guarding a value the compiler had
    // already computed once. black_box the input to defeat constant folding,
    // and the output to defeat dead-code elimination.
    let hash = measure("args hash (fnv1a)", 0, || {
        core::hint::black_box(audit::fnv1a(core::hint::black_box(READ_IN_SCOPE).as_bytes()));
        0
    });
    let Some(hash) = hash else {
        teardown(subject);
        crate::serial_println!("bench> cancelled");
        return false;
    };
    report(&hash);
    rows.push(hash);

    // One audit entry per iteration really appends, so this row is short and
    // priced in counter ticks, converted with the factor from the long rows above.
    let tpms = ticks_per_ms(&rows);
    let c0 = crate::arch::cycle_count();
    for _ in 0..AUDIT_ITERS {
        audit::record(subject.scoped, "<bench>", 0, audit::Outcome::Executed, 0);
    }
    let audit_ticks = crate::arch::cycle_count().saturating_sub(c0);
    let per = per_call(audit_ticks, AUDIT_ITERS);
    match ticks_to_ns(per, tpms) {
        Some(ns) => crate::serial_println!(
            "bench>   {:<28} {:>7} ns/call {:>8} tick/call ({} iters, appends; ktrace coalesced)",
            "audit::record",
            ns,
            per,
            AUDIT_ITERS
        ),
        None => crate::serial_println!("bench>   {:<28} {:>8} tick/call ({} iters)", "audit::record", per, AUDIT_ITERS),
    }

    // --- marginal cost per gate ---
    crate::serial_println!("bench> marginal cost per gate (ns/call, same batch size):");
    for (gate, ns) in gate_deltas(&prefixes) {
        match ns {
            Some(ns) => crate::serial_println!("bench>   gate {} {:<12} +{} ns", gate, gate_name(gate), ns),
            None => crate::serial_println!("bench>   gate {} {:<12} below noise floor", gate, gate_name(gate)),
        }
    }
    let total = prefixes.last().map(|r| r.ns_per_call()).unwrap_or(0);
    crate::serial_println!(
        "bench> full authorization decision: {} ns/call ({} ticks @ {} ticks/ms). \
         Compare against per-token decode cost from /perf.",
        total,
        prefixes.last().map(|r| r.ticks_per_call()).unwrap_or(0),
        tpms
    );

    teardown(subject);
    true
}

fn report(r: &Row) {
    crate::serial_println!(
        "bench>   {:<28} {:>7} ns/call {:>8} tick/call ({} iters in {} ms, stop={}){}",
        r.label,
        r.ns_per_call(),
        r.ticks_per_call(),
        r.iters,
        r.ms,
        gate_name(r.stop),
        if r.is_suspect() { "  [SUSPECT: no time elapsed -- loop elided or clock stopped]" } else { "" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &'static str, upto: u8, ms: u64, iters: u64) -> Row {
        Row { label, upto, stop: 0, iters, ms, ticks: 0 }
    }

    #[test_case]
    fn per_call_rounds_to_nearest_and_guards_zero() {
        assert_eq!(per_call(0, 0), 0); // cancelled run: no divide by zero
        assert_eq!(per_call(100, 10), 10);
        assert_eq!(per_call(105, 10), 11); // .5 rounds up
        assert_eq!(per_call(104, 10), 10);
        // 60 ms over 1M iterations is 60 ns/call.
        assert_eq!(per_call_ns(60, 1_000_000), 60);
        assert_eq!(per_call_ns(0, 1_000_000), 0);
    }

    #[test_case]
    fn gate_deltas_are_marginal_and_report_noise_as_unmeasured() {
        // Cumulative 40/60/70/120 ns over 1M-iteration batches.
        let rows = alloc::vec![
            row("g1", GATE_GRAMMAR, 40, 1_000_000),
            row("g2", GATE_CAPABILITY, 60, 1_000_000),
            row("g3", GATE_TAINT, 70, 1_000_000),
            row("g4", GATE_SCOPE, 120, 1_000_000),
        ];
        assert_eq!(
            gate_deltas(&rows),
            alloc::vec![(1, Some(40)), (2, Some(20)), (3, Some(10)), (4, Some(50))]
        );

        // Noise: a longer prefix measured *faster*. That gate is reported as
        // unmeasurable (`None`), never as a zero cost -- and the running maximum
        // is what the next gate differences against, so one fast batch cannot
        // inflate the gate after it.
        let noisy = alloc::vec![
            row("g1", GATE_GRAMMAR, 50, 1_000_000),
            row("g2", GATE_CAPABILITY, 45, 1_000_000),
            row("g3", GATE_TAINT, 55, 1_000_000),
        ];
        assert_eq!(gate_deltas(&noisy), alloc::vec![(1, Some(50)), (2, None), (3, Some(5))]);
    }

    #[test_case]
    fn tick_conversion_needs_a_factor() {
        // A 2.4 MHz-per-ms counter: 240 ticks is 100 ns.
        assert_eq!(ticks_to_ns(240, 2_400_000), Some(100));
        assert_eq!(ticks_to_ns(240, 0), None); // no long row: refuse to invent one
    }

    #[test_case]
    fn ticks_per_ms_only_trusts_a_long_row() {
        let short = Row { label: "short", upto: 0, stop: 0, iters: 1, ms: 1, ticks: 1_000_000 };
        let long = Row { label: "long", upto: 0, stop: 0, iters: 1, ms: TARGET_MS, ticks: TARGET_MS * 2_400_000 };
        assert_eq!(ticks_per_ms(&alloc::vec![short]), 0, "a 1 ms row is quantization noise, not a clock");
        assert_eq!(ticks_per_ms(&alloc::vec![short, long]), 2_400_000);
    }

    /// The measurement path must agree with the real path about which gate
    /// refuses, or the benchmark is pricing something the executor does not do.
    /// This is what keeps `gate_prefix`'s copy of the chain from drifting.
    #[test_case]
    fn gate_prefix_agrees_with_execute() {
        use crate::synapse::{self, Invocation};

        let subject = setup();
        let trusted = Justification::trusted();
        let tainted = Justification::from_context(Provenance::UntrustedIngested);

        // A file to read, inside the granted scope, so the "passes everything"
        // case really executes rather than failing on a missing path.
        synapse::fs::write("/bench/probe.txt", b"x");

        let cases: &[(&str, Justification, TaskId)] = &[
            // Executes: all four gates pass.
            (READ_IN_SCOPE, trusted, subject.scoped),
            (r#"{"name":"list","arguments":{}}"#, trusted, subject.scoped),
            // Gate 1: not a registered primitive.
            (r#"{"name":"exfiltrate","arguments":{}}"#, trusted, subject.scoped),
            // Gate 1: valid name, broken shape.
            (r#"{"name":"list","arguments":"#, trusted, subject.scoped),
            // Gate 2: never granted.
            (r#"{"name":"net_http_post","arguments":{"url":"http://x/","body":"y"}}"#, trusted, subject.scoped),
            // Gate 3: destructive under untrusted justification.
            (r#"{"name":"mem_fs_delete","arguments":{"path":"/bench/probe.txt"}}"#, tainted, subject.scoped),
            // Gate 3: identity file, destructive by re-entry.
            (r#"{"name":"mem_fs_write","arguments":{"path":"/bench/SOUL.md","text":"p"}}"#, tainted, subject.scoped),
            // Gate 4: outside the granted path scope.
            (r#"{"name":"mem_fs_read","arguments":{"path":"/etc/shadow"}}"#, trusted, subject.scoped),
            // Gate 4: `..` cannot walk out of the prefix grant.
            (r#"{"name":"mem_fs_read","arguments":{"path":"/bench/../etc/shadow"}}"#, trusted, subject.scoped),
            // Gate 4: the login credential record, unreachable however the call is
            // justified or scoped. All four verbs, including read.
            (r#"{"name":"mem_fs_read","arguments":{"path":"/configs/core/auth.json"}}"#, trusted, subject.scoped),
            (r#"{"name":"mem_fs_write","arguments":{"path":"/configs/core/auth.json","text":"{}"}}"#, trusted, subject.scoped),
            (r#"{"name":"mem_fs_delete","arguments":{"path":"/configs/core/auth.json"}}"#, trusted, subject.scoped),
            // …and through a path that only normalises to it.
            (r#"{"name":"mem_fs_read","arguments":{"path":"/configs/core/../core/auth.json"}}"#, trusted, subject.scoped),
        ];

        for &(raw, j, task) in cases {
            let predicted = executor::gate_prefix(task, raw, j, GATE_COUNT);
            let actual = synapse::execute_with_justification(task, raw, j);
            assert_eq!(
                predicted,
                executor::gate_of_outcome(&actual),
                "gate_prefix said {} ({}), executor said {actual:?} for {raw}",
                predicted,
                gate_name(predicted)
            );
        }

        // A prefix shorter than the refusing gate reports "passed": measuring
        // gates 1..2 of a taint-refused call must not charge for gate 3.
        let del = r#"{"name":"mem_fs_delete","arguments":{"path":"/bench/probe.txt"}}"#;
        assert_eq!(executor::gate_prefix(subject.scoped, del, tainted, GATE_CAPABILITY), 0);
        assert_eq!(executor::gate_prefix(subject.scoped, del, tainted, GATE_TAINT), GATE_TAINT);

        synapse::fs::delete("/bench/probe.txt");
        teardown(subject);
    }

    /// The subject's authority is exactly what the module claims: the four
    /// granted primitives and nothing else, narrowed to `/bench/**`, and gone
    /// after teardown. A benchmark that leaked a capability would be a worse bug
    /// than a wrong number.
    #[test_case]
    fn bench_subject_holds_only_what_it_declares_and_gives_it_back() {
        let subject = setup();
        let scoped = subject.scoped;
        for &p in GRANTED {
            assert!(cap::holds(scoped, Right::InvokePrimitive(p)), "missing granted primitive {p}");
        }
        assert!(
            !cap::holds(scoped, Right::InvokePrimitive(registry::NET_HTTP_POST)),
            "the denial row needs net_http_post to be genuinely absent"
        );
        // The ledger bites for the scoped task and is absent for the other.
        let inside = Scope::Path(alloc::string::String::from("/bench/a.txt"));
        let outside = Scope::Path(alloc::string::String::from("/etc/shadow"));
        assert!(cap::scope_check(scoped, CapDomain::Fs, Rights::READ, &inside));
        assert!(!cap::scope_check(scoped, CapDomain::Fs, Rights::READ, &outside));
        assert!(
            cap::scope_check(subject.unscoped, CapDomain::Fs, Rights::READ, &outside),
            "no ledger entry means unconstrained (deny-only-when-recorded)"
        );

        teardown(subject);
        assert!(!cap::holds(scoped, Right::InvokePrimitive(registry::MEM_FS_DELETE)), "teardown must revoke");
        assert!(
            cap::scope_check(scoped, CapDomain::Fs, Rights::READ, &outside),
            "teardown must drop the scope ledger too"
        );
    }
}
