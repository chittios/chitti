//! **OOM reclaim registry** — when the heap cannot grow, drop what is safe to
//! drop and retry the allocation before killing a task.
//!
//! The kernel heap used to grow until frames ran out and then panic in
//! `handle_alloc_error`. That is the wrong policy for an agentic OS: a runaway
//! agent or a crafted media file should not take down the shell. The order is
//! now:
//!
//! 1. first-fit in the current arena  
//! 2. [`crate::mm::heap::grow`] more physical frames  
//! 3. **reclaim** — every registered hook runs and frees what it can  
//! 4. retry the allocation once  
//! 5. if still out of memory, **OOM-kill** the current non-bootstrap task  
//!    (bootstrap / shell still panics — there is nowhere safe to land)
//!
//! Hooks are `fn() -> usize` returning approximate bytes freed. They must not
//! allocate (or only allocate less than they free), or reclaim can nest and
//! fail closed without progress. Register at boot from subsystems that hold
//! large caches (prefix store, image tabs, speech queues, …).

use super::Locked;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// A reclaim hook: free what is safe, return an estimate of bytes released.
pub type Hook = fn() -> usize;

static HOOKS: Locked<Vec<Hook>> = Locked::new(Vec::new());
static RECLAIM_RUNS: AtomicU64 = AtomicU64::new(0);
static RECLAIM_BYTES: AtomicU64 = AtomicU64::new(0);
static OOM_KILLS: AtomicU64 = AtomicU64::new(0);
/// Re-entrancy guard: a hook that allocates re-enters `run` and would recurse.
static IN_RECLAIM: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Register a reclaim hook. Idempotent for the same function pointer.
pub fn register(hook: Hook) {
    HOOKS.with(|h| {
        if !h.iter().any(|f| core::ptr::fn_addr_eq(*f, hook)) {
            h.push(hook);
        }
    });
}

/// Run every registered hook once. Returns total estimated bytes freed.
///
/// Safe to call from the allocator (the free path does not re-enter reclaim).
/// Nested calls (a hook that allocates) return 0 immediately so we never
/// recurse. Hooks should free more than they allocate.
pub fn run() -> usize {
    if IN_RECLAIM.swap(true, Ordering::Acquire) {
        return 0;
    }
    RECLAIM_RUNS.fetch_add(1, Ordering::Relaxed);
    let hooks: Vec<Hook> = HOOKS.with(|h| h.clone());
    let mut freed = 0usize;
    for f in hooks {
        freed = freed.saturating_add(f());
    }
    RECLAIM_BYTES.fetch_add(freed as u64, Ordering::Relaxed);
    if freed > 0 {
        crate::ktrace::log_fmt(format_args!(
            "mm: reclaimed ~{} KiB across {} hook(s)",
            freed / 1024,
            HOOKS.with(|h| h.len())
        ));
    }
    IN_RECLAIM.store(false, Ordering::Release);
    freed
}

/// `(reclaim runs, bytes freed estimate, oom kills)`.
pub fn stats() -> (u64, u64, u64) {
    (
        RECLAIM_RUNS.load(Ordering::Relaxed),
        RECLAIM_BYTES.load(Ordering::Relaxed),
        OOM_KILLS.load(Ordering::Relaxed),
    )
}

/// Record that a task was killed for OOM (called from the OOM path).
pub fn note_oom_kill() {
    OOM_KILLS.fetch_add(1, Ordering::Relaxed);
}

/// Built-in reclaims that are always safe and free measurable heap.
///
/// Registered from [`crate::mm::init`] so the registry is never empty even when
/// higher layers have not started.
pub fn register_builtin() {
    register(reclaim_audit_compact);
    register(reclaim_session_storage);
}

/// Drop half the audit log when it is large — the chain is re-sealed to a
/// truncation marker so verify still works (see `synapse::audit::compact`).
fn reclaim_audit_compact() -> usize {
    crate::synapse::audit::compact_if_over_budget()
}

/// Wipe **session** (ephemeral) agent storage maps — durable store is left alone.
fn reclaim_session_storage() -> usize {
    crate::agent::storage::reclaim_all_session()
}
