//! ChittiOS: cooperative host hook so a long-running script can't monopolize
//! the (cooperatively-scheduled) kernel thread.
//!
//! The tree-walking interpreter runs entirely on the caller's stack with no
//! yield points, so evaluating a heavy page's scripts would freeze the UI
//! clock/mouse and swallow Ctrl+C. The kernel installs a [`TickHook`] via
//! [`set_tick_hook`]; the interpreter calls [`host_tick`] from its hot loops
//! (function calls + loop iterations), which every ~2048 invocations runs the
//! hook. The hook pumps the kernel UI and returns `true` to request
//! cancellation, whereupon the interpreter aborts with the uncatchable
//! [`interrupt_error`] (a `try/catch` in the script cannot swallow it — see
//! `execute_try_statement`).

use crate::runner::ds::error::JErrorType;
use core::sync::atomic::{AtomicUsize, Ordering};

/// A host callback: pump the UI and return `true` to request cancellation.
pub type TickHook = fn() -> bool;

static TICK_HOOK: AtomicUsize = AtomicUsize::new(0);
static TICK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Sentinel message identifying the interrupt error so `try/catch` can refuse
/// to catch it. Begins with a NUL so no real error message collides.
pub const INTERRUPT_SENTINEL: &str = "\u{0}__chitti_interrupt__";

/// Install (or clear, with `None`) the host tick hook. Setting it also resets
/// the interval counter so the first tick after install fires promptly.
pub fn set_tick_hook(hook: Option<TickHook>) {
    TICK_COUNT.store(0, Ordering::Relaxed);
    TICK_HOOK.store(hook.map_or(0, |f| f as usize), Ordering::Relaxed);
}

/// Called from the interpreter's hot loops. Every 2048th call runs the host
/// hook; returns `true` when the host requests cancellation.
#[inline]
pub fn host_tick() -> bool {
    let n = TICK_COUNT.fetch_add(1, Ordering::Relaxed);
    if n & 0x7FF != 0 {
        return false;
    }
    let p = TICK_HOOK.load(Ordering::Relaxed);
    if p == 0 {
        return false;
    }
    // SAFETY: `p` is either 0 (handled above) or a `TickHook` fn pointer stored
    // by `set_tick_hook`; the transmute reconstructs that same pointer type.
    let f: TickHook = unsafe { core::mem::transmute::<usize, TickHook>(p) };
    f()
}

/// The error raised to abort execution on a host cancel request.
pub fn interrupt_error() -> JErrorType {
    JErrorType::RangeError(INTERRUPT_SENTINEL.into())
}

/// True if `err` is the interrupt sentinel (so `try/catch` won't swallow it).
pub fn is_interrupt(err: &JErrorType) -> bool {
    matches!(err, JErrorType::RangeError(m) if m == INTERRUPT_SENTINEL)
}
