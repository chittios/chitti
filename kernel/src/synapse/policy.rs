//! Permission policy and human approval, **below** the determinism boundary.
//!
//! # Why this is here and not in the tool layer
//!
//! The allow/ask/deny rules ([`crate::tools::permissions`]) and the approval
//! modal were consulted in exactly one place: `shell`'s tool dispatch. That works
//! while every agent runs as kernel code on the shell's own stack, because the
//! shell *is* the only path to a primitive. It stops working the moment an agent
//! is a tenant with its own address space: a tenant reaching
//! [`crate::synapse::executor::execute`] directly — which is the whole point of a
//! syscall ABI — would simply never run the check. Policy enforced by the thing
//! being governed is not enforcement.
//!
//! So the *decision* lives here, checked inside the executor between the
//! capability gate and the taint gate, and the tool layer keeps its copy only for
//! what it is actually good for: asking the human early, with context, before a
//! call is even attempted.
//!
//! **This is not a fifth Synapse gate.** The four gates — grammar, capability,
//! taint, scope — are the capability/provenance architecture, and
//! the design paper publishes them by number: a figure with four boxes, an attack
//! table citing "Gate 3 (taint)" and "Gate 4 (scope)", and a cost methodology
//! measuring cumulative prefixes 1, 1--2, 1--3, 1--4. Numbering this check into
//! that chain would invalidate all three. It is human policy layered over the
//! chain, so it is deliberately absent from `GATE_COUNT` and from `gate_prefix`.
//!
//! # Why a tenant cannot forge an approval
//!
//! Two things stay out of the caller's hands. The **mode** is kernel state set by
//! a human at the shell, not a parameter. And an **approval** is a one-shot entry
//! the kernel records against `(task, primitive)`; the caller passes nothing, so
//! there is nothing to fabricate. A tenant can ask for a primitive and be told
//! it needs approval; it cannot tell the kernel that approval was given.
//!
//! That is also why approvals are consumed rather than sticky: "the human said
//! yes" is a statement about one operation, and a tenant that could reuse it
//! would hold standing authority the human never granted.

use crate::mm::Locked;
use crate::sched::TaskId;
use alloc::vec::Vec;

/// How much the human wants to be asked. Set by `/mode` at the shell; read by
/// the executor's approval check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Every effectful primitive needs an explicit approval.
    Manual,
    /// Rules decide; unmatched effectful primitives run. The default, and what
    /// the shell has always done.
    Auto,
    /// Nothing is asked. Identical to `Auto` at this layer today, and kept
    /// distinct because the tool layer still separates them (it asks on `Auto`
    /// for `ask`-rule tools and does not on `Bypass`).
    Bypass,
}

/// What the policy gate concluded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// Proceed to the next gate.
    Allow,
    /// A human must approve this exact call first; none is on record.
    NeedsApproval,
}

static MODE: Locked<Mode> = Locked::new(Mode::Auto);

/// One-shot approvals, as `(task, primitive)` pairs. A `Vec` because the list is
/// never more than a handful long — a human can only approve so fast.
static APPROVALS: Locked<Vec<(TaskId, &'static str)>> = Locked::new(Vec::new());

/// Read the current mode.
pub fn mode() -> Mode {
    MODE.with(|m| *m)
}

/// Set the mode. **Human-only**: this is deliberately not reachable as a Synapse
/// primitive or an agent tool, because a tenant that could set its own mode to
/// `Bypass` would be deciding how much it gets asked about its own actions.
pub fn set_mode(m: Mode) {
    MODE.with(|slot| *slot = m);
    crate::ktrace::log_fmt(format_args!("synapse.policy: mode = {m:?}"));
}

/// Record that a human approved `primitive` for `task`, once.
///
/// Called from the shell's confirmation path. Duplicate grants collapse — two
/// yeses to the same question are still one permission.
pub fn approve(task: TaskId, primitive: &'static str) {
    APPROVALS.with(|list| {
        if !list.iter().any(|&(t, p)| t == task && p == primitive) {
            list.push((task, primitive));
        }
    });
    crate::ktrace::log_fmt(format_args!("synapse.policy: human approved '{primitive}' for task {task} (one shot)"));
}

/// Take the approval for `(task, primitive)` if there is one.
fn consume(task: TaskId, primitive: &'static str) -> bool {
    APPROVALS.with(|list| {
        match list.iter().position(|&(t, p)| t == task && p == primitive) {
            Some(i) => {
                list.remove(i);
                true
            }
            None => false,
        }
    })
}

/// Drop every approval held for `task`. Called when a task is killed, so a
/// recycled task id cannot inherit a previous tenant's permission.
pub fn clear_task(task: TaskId) {
    APPROVALS.with(|list| list.retain(|&(t, _)| t != task));
}

/// How many approvals are outstanding, for diagnostics and tests.
pub fn pending() -> usize {
    APPROVALS.with(|list| list.len())
}

/// Decide whether `caller` may invoke `primitive`, consuming an approval if one
/// is needed and available.
///
/// `effectful` is the registry's own view of whether the primitive changes
/// anything — a read is never gated here, whatever the mode, because asking a
/// human to confirm a read trains them to click yes.
pub fn decide(caller: TaskId, primitive: &'static str, effectful: bool, human_confirmed: bool) -> Verdict {
    // An explicit confirmation at the shell satisfies the requirement directly.
    // Reusing `Justification::human_confirmed` rather than inventing a parallel
    // signal matters: it is already the one thing that lets a tainted
    // justification past the taint gate, it is already kernel-set, and a second
    // mechanism for "the human said yes" would be a second thing to get wrong.
    //
    // A tenant cannot set it. The syscall ABI takes only the call text; the
    // kernel builds the `Justification` from its own record of the caller's
    // session, so `human_confirmed` is not reachable from user memory.
    if human_confirmed {
        return Verdict::Allow;
    }
    verdict(caller, primitive, effectful, true)
}

/// The same decision **without consuming** an approval.
///
/// Exists for [`crate::synapse::executor::gate_prefix`], the benchmark's
/// measurement path, whose contract is that it runs the real predicates in the
/// real order while executing no primitive and writing no audit entry. Consuming
/// an approval there would be a side effect — and worse, a *silent* one: pricing
/// the gate chain would spend the human's permission, so the next real call would
/// ask again for no reason anyone could trace to a benchmark.
pub fn peek(caller: TaskId, primitive: &'static str, effectful: bool, human_confirmed: bool) -> Verdict {
    if human_confirmed {
        return Verdict::Allow;
    }
    verdict(caller, primitive, effectful, false)
}

fn verdict(caller: TaskId, primitive: &'static str, effectful: bool, take: bool) -> Verdict {
    let approved = |primitive: &'static str| {
        if take {
            consume(caller, primitive)
        } else {
            APPROVALS.with(|list| list.iter().any(|&(t, p)| t == caller && p == primitive))
        }
    };
    // **`tools::permissions` is deliberately NOT consulted here.** Its patterns
    // are authored against *tool* names — the defaults are `read`, `write`,
    // `install`, `delete` — while this function sees *primitive* names like
    // `mem_fs_write`. Matching one namespace's rules against the other's names is
    // a category error whose results would be arbitrary: today `deny: ["delete"]`
    // happens not to match `mem_fs_delete`, and a user writing `deny: ["*"]` would
    // silently disable every primitive in the kernel. Tool-name rules stay in the
    // tool layer, where the names line up. Primitive-granularity deny rules would
    // need their own config and are not pretended to exist.
    if !effectful {
        return Verdict::Allow;
    }
    match mode() {
        Mode::Bypass | Mode::Auto => Verdict::Allow,
        Mode::Manual => {
            if approved(primitive) {
                Verdict::Allow
            } else {
                Verdict::NeedsApproval
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suite shares one kernel, so leave the mode as we found it.
    fn with_mode<R>(m: Mode, f: impl FnOnce() -> R) -> R {
        let prev = mode();
        set_mode(m);
        let r = f();
        set_mode(prev);
        r
    }

    #[test_case]
    fn auto_mode_allows_effectful_primitives() {
        // The default, and what the shell has always done — so introducing this
        // check must not change any existing behaviour.
        with_mode(Mode::Auto, || {
            assert_eq!(decide(7, "mem_fs_write", true, false), Verdict::Allow);
            assert_eq!(decide(7, "mem_fs_read", false, false), Verdict::Allow);
        });
    }

    #[test_case]
    fn manual_mode_needs_an_approval_for_an_effectful_primitive() {
        with_mode(Mode::Manual, || {
            assert_eq!(decide(7, "mem_fs_write", true, false), Verdict::NeedsApproval);
            // A read is never gated: asking a human to confirm reads trains them
            // to click yes, which is worse than not asking.
            assert_eq!(decide(7, "mem_fs_read", false, false), Verdict::Allow);
            // With an approval on record it proceeds — once.
            approve(7, "mem_fs_write");
            assert_eq!(decide(7, "mem_fs_write", true, false), Verdict::Allow);
            assert_eq!(
                decide(7, "mem_fs_write", true, false),
                Verdict::NeedsApproval,
                "an approval is one shot: reuse would be standing authority nobody granted"
            );
        });
    }

    #[test_case]
    fn an_approval_is_scoped_to_one_task_and_one_primitive() {
        with_mode(Mode::Manual, || {
            approve(11, "mem_fs_write");
            // Not another task's...
            assert_eq!(decide(12, "mem_fs_write", true, false), Verdict::NeedsApproval);
            // ...and not another primitive's.
            assert_eq!(decide(11, "mem_fs_delete", true, false), Verdict::NeedsApproval);
            assert_eq!(decide(11, "mem_fs_write", true, false), Verdict::Allow);
        });
    }

    #[test_case]
    fn killing_a_task_drops_its_approvals() {
        // Task ids are reused, so a stale approval would hand a future tenant a
        // permission a human gave to something else entirely.
        with_mode(Mode::Manual, || {
            approve(21, "mem_fs_write");
            assert!(pending() > 0);
            clear_task(21);
            assert_eq!(decide(21, "mem_fs_write", true, false), Verdict::NeedsApproval);
        });
    }

    #[test_case]
    fn duplicate_approvals_collapse() {
        with_mode(Mode::Manual, || {
            let before = pending();
            approve(31, "mem_fs_write");
            approve(31, "mem_fs_write");
            assert_eq!(pending(), before + 1, "two yeses to one question is one permission");
            clear_task(31);
        });
    }
}
