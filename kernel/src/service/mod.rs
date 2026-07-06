//! **Service agents** — long-running daemons (Network/SSH/HTTP/Doc/…) that run
//! as real scheduled tasks with a native `serve()` event loop, in contrast to
//! the request/response reasoning agents driven by the model. A service is
//! declared by a [`ServiceSpec`] (name, entry loop, autostart flag, and the
//! capability rights it runs with), started by [`start`], supervised by
//! [`supervise_tick`] (called from the shell idle loop), and stopped by [`stop`].
//!
//! Determinism boundary: a service's hot loop is native, deterministic code —
//! there is no model in it. The markdown "agent" for a service supplies its
//! identity/persona and the *capability grant* it runs under; the loop itself
//! (accept a connection, copy bytes, parse a protocol) is native and audited at
//! the Synapse boundary exactly like any other effect.

pub mod network;

use crate::cap::{self, Right};
use crate::mm::Locked;
use crate::sched::{self, TaskId};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A daemon declaration. `entry` is a native serve loop; it should treat any
/// `CapabilityDenied`/`Closed` from a channel/net op as a shutdown signal and
/// `return`, since [`stop`]/`kill` revokes the task's authority.
pub struct ServiceSpec {
    pub name: &'static str,
    pub entry: extern "C" fn(u64),
    /// Start at boot and keep restarting on death (bounded), vs. start on demand.
    pub autostart: bool,
    /// Authority granted to the task at spawn (e.g. `NetListen`, `InvokePrimitive`).
    pub caps: &'static [Right],
}

struct Supervised {
    spec: &'static ServiceSpec,
    task: TaskId,
    restarts: u32,
}

static SERVICES: Locked<BTreeMap<&'static str, Supervised>> = Locked::new(BTreeMap::new());

/// Max automatic restarts before the supervisor gives up on an autostart
/// service (a crash loop should fail visibly, not spin forever).
const MAX_RESTARTS: u32 = 5;

/// Start (or restart) a service: spawn its task and grant its capabilities
/// *under interrupts-off*, so a timer preemption can't run `serve()` before its
/// authority is in place — the same grant-before-run guard the IPC test uses.
/// Returns the task id. Idempotent-ish: if already running, returns the existing
/// task without respawning.
pub fn start(spec: &'static ServiceSpec) -> TaskId {
    if let Some(existing) = SERVICES.with(|m| m.get(spec.name).map(|s| s.task)) {
        if sched::is_alive(existing) {
            return existing;
        }
    }
    crate::arch::interrupts::without_interrupts(|| {
        let task = sched::spawn(spec.name, spec.entry, 0);
        for r in spec.caps {
            cap::grant(task, *r);
        }
        SERVICES.with(|m| {
            let restarts = m.get(spec.name).map(|s| s.restarts).unwrap_or(0);
            m.insert(spec.name, Supervised { spec, task, restarts });
        });
        crate::ktrace::log_fmt(format_args!("service: started '{}' as task {task}", spec.name));
        task
    })
}

/// Stop a service: revoke its authority and terminate its task (`sched::kill`
/// wipes the cap table). The serve loop, being cooperative, observes the
/// revocation at its next channel/net op and returns. Removes it from
/// supervision so it is not auto-restarted.
pub fn stop(name: &str) -> Result<(), &'static str> {
    let task = SERVICES.with(|m| m.remove(name).map(|s| s.task));
    match task {
        Some(t) => sched::kill(t),
        None => Err("no such service"),
    }
}

/// The task id currently backing service `name`, if running. Used by
/// `channel_grant` to resolve a forward target by service name.
pub fn task_for(name: &str) -> Option<TaskId> {
    SERVICES.with(|m| m.get(name).and_then(|s| if sched::is_alive(s.task) { Some(s.task) } else { None }))
}

/// A snapshot of the service table for `/agents services`: (name, task, alive).
pub fn list() -> Vec<(&'static str, TaskId, bool)> {
    SERVICES.with(|m| m.iter().map(|(&name, s)| (name, s.task, sched::is_alive(s.task))).collect())
}

/// Supervision tick — call from `shell::upkeep`. Restarts a dead `autostart`
/// service with a bounded restart count so a crash loop fails visibly rather
/// than spinning forever.
pub fn supervise_tick() {
    // Collect dead autostart services first (don't respawn while holding the lock).
    let dead: Vec<&'static ServiceSpec> = SERVICES.with(|m| {
        m.values()
            .filter(|s| s.spec.autostart && !sched::is_alive(s.task) && s.restarts < MAX_RESTARTS)
            .map(|s| s.spec)
            .collect()
    });
    for spec in dead {
        SERVICES.with(|m| {
            if let Some(s) = m.get_mut(spec.name) {
                s.restarts += 1;
            }
        });
        crate::ktrace::log_fmt(format_args!("service: restarting dead autostart '{}'", spec.name));
        start(spec);
    }
}

#[cfg(test)]
pub fn reset() {
    SERVICES.with(|m| m.clear());
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    static RUNS: AtomicU32 = AtomicU32::new(0);

    // A one-shot service: bumps a counter and exits, so the supervisor sees it
    // die and (because autostart) restarts it.
    extern "C" fn oneshot(_arg: u64) {
        RUNS.fetch_add(1, Ordering::SeqCst);
    }

    static ONESHOT: ServiceSpec = ServiceSpec { name: "test-oneshot", entry: oneshot, autostart: true, caps: &[] };

    #[test_case]
    fn service_starts_and_supervisor_restarts_it() {
        reset();
        RUNS.store(0, Ordering::SeqCst);
        let t0 = start(&ONESHOT);
        assert!(task_for("test-oneshot").is_some(), "service should be registered");
        // Let it run to completion (it just bumps RUNS and exits).
        let mut spins = 0;
        while sched::is_alive(t0) && spins < 10_000_000 {
            sched::yield_now();
            spins += 1;
        }
        assert_eq!(RUNS.load(Ordering::SeqCst), 1, "the service loop should have run once");
        // Supervisor sees it dead and restarts it (bounded).
        supervise_tick();
        let mut spins = 0;
        while RUNS.load(Ordering::SeqCst) < 2 && spins < 10_000_000 {
            sched::yield_now();
            spins += 1;
        }
        assert_eq!(RUNS.load(Ordering::SeqCst), 2, "the supervisor should have restarted the dead service");
        reset();
    }

    extern "C" fn park(_arg: u64) {
        loop {
            sched::yield_now();
        }
    }

    #[test_case]
    fn stop_removes_from_supervision() {
        reset();
        static PARK: ServiceSpec = ServiceSpec { name: "test-park", entry: park, autostart: false, caps: &[] };
        start(&PARK);
        assert!(task_for("test-park").is_some());
        stop("test-park").unwrap();
        assert!(task_for("test-park").is_none(), "stopped service must leave supervision");
        assert_eq!(stop("test-park"), Err("no such service"));
        reset();
    }
}
