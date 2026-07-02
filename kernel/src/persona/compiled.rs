//! Self-compiling agents / **compiled intents** (`CHITTI_OS_HANDOFF.md`
//! Phase 6, feature 2): the `/bin` analogue.
//!
//! The first time an intent is satisfied, its validated capability trace is
//! recorded, keyed by an *intent signature* and guarded by a set of
//! *preconditions* -- a snapshot of the external state the trace depended on.
//! On a later matching intent whose preconditions still hold, the trace is
//! **replayed deterministically and inference is skipped entirely** (a
//! "compiled intent" -- a cached, deterministic capability trace ≈ a binary).
//! If a precondition no longer holds, the compiled intent is stale: the agent
//! falls back to planning, and recompiles from the fresh trace.
//!
//! This is the payoff of the determinism boundary: because every effect below
//! it is deterministic and audited, a validated trace can be re-run as-is,
//! turning a repeated agent task from an inference problem into a lookup.

use super::actions::Action;
use super::agent::Agent;
use super::memory;
use super::planner::Planner;
use crate::mm::Locked;
use crate::synapse::audit::fnv1a;
use crate::synapse::grammar::{self, ArgValue};
use crate::synapse::registry;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// A condition on external state that must still hold for a compiled trace to
/// be safe to replay. Captured (as a hash of the current value) at compile
/// time and re-checked before every replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Precondition {
    /// The file at `path` must have this exact content (`None` = must still be
    /// absent). Guards traces that *read* a file they did not themselves write.
    FileContent { path: String, expected: Option<u64> },
    /// The persistent fact `agent/key` must have this exact value (`None` =
    /// must still be absent). Guards traces that *recall* a fact.
    Fact { agent: String, key: String, expected: Option<u64> },
}

impl Precondition {
    /// Whether this precondition holds against current state.
    fn satisfied(&self) -> bool {
        match self {
            Precondition::FileContent { path, expected } => {
                crate::synapse::fs::read(path).map(|b| fnv1a(&b)) == *expected
            }
            Precondition::Fact { agent, key, expected } => {
                memory::persisted(agent, key).map(|v| fnv1a(v.as_bytes())) == *expected
            }
        }
    }
}

/// A compiled intent: the validated trace plus the preconditions that make it
/// safe to replay.
struct CompiledIntent {
    signature: u64,
    sample_intent: String,
    preconditions: Vec<Precondition>,
    trace: Vec<Action>,
    uses: u64,
}

static STORE: Locked<Vec<CompiledIntent>> = Locked::new(Vec::new());
static REPLAYS: AtomicU64 = AtomicU64::new(0);

/// Total compiled-intent replays (cache hits) so far.
pub fn replays() -> u64 {
    REPLAYS.load(Ordering::SeqCst)
}

/// Number of distinct compiled intents in the store (the "/bin" size).
pub fn count() -> usize {
    STORE.with(|s| s.len())
}

/// Stable signature of an intent: normalised (lowercased, whitespace
/// collapsed, trimmed) then hashed, so trivial phrasing differences map to the
/// same compiled intent.
pub fn signature(intent: &str) -> u64 {
    let mut normalized = String::new();
    for (i, word) in intent.split_whitespace().enumerate() {
        if i > 0 {
            normalized.push(' ');
        }
        normalized.push_str(&word.to_ascii_lowercase());
    }
    fnv1a(normalized.as_bytes())
}

/// Outcome of consulting the store for a signature.
enum Lookup {
    /// A compiled intent exists and all its preconditions still hold: replay.
    Hit(Vec<Action>),
    /// A compiled intent exists but a precondition failed: stale, re-plan.
    Stale,
    /// No compiled intent for this signature.
    Absent,
}

/// Consult the store without holding its lock across the (separately locked)
/// state reads the precondition checks perform.
fn lookup(sig: u64) -> Lookup {
    let found = STORE.with(|s| {
        s.iter()
            .find(|c| c.signature == sig)
            .map(|c| (c.preconditions.clone(), c.trace.clone()))
    });
    match found {
        None => Lookup::Absent,
        Some((preconds, trace)) => {
            if preconds.iter().all(|p| p.satisfied()) {
                Lookup::Hit(trace)
            } else {
                Lookup::Stale
            }
        }
    }
}

/// Record (or replace) the compiled intent for `sig`.
fn record(sig: u64, intent: &str, preconditions: Vec<Precondition>, trace: Vec<Action>) {
    STORE.with(|s| {
        s.retain(|c| c.signature != sig);
        let n = trace.len();
        s.push(CompiledIntent {
            signature: sig,
            sample_intent: intent.to_string(),
            preconditions,
            trace,
            uses: 0,
        });
        crate::ktrace::log_fmt(format_args!(
            "persona.compiled: recorded intent sig={sig:#018x} ({n}-step trace) -- '{intent}'"
        ));
    });
}

fn bump_uses(sig: u64) {
    STORE.with(|s| {
        if let Some(c) = s.iter_mut().find(|c| c.signature == sig) {
            c.uses += 1;
            crate::ktrace::log_fmt(format_args!(
                "persona.compiled: '{}' replayed (used {} time(s) total)",
                c.sample_intent, c.uses
            ));
        }
    });
}

/// Execute `intent` on `agent`, using the compiled-intent cache: replay a
/// still-valid compiled trace with **zero inference**, or plan (invoking the
/// planner), execute, and compile the fresh trace. Returns the final result.
pub fn run(agent: &mut Agent, intent: &str, planner: &mut dyn Planner) -> String {
    let sig = signature(intent);
    match lookup(sig) {
        Lookup::Hit(trace) => {
            REPLAYS.fetch_add(1, Ordering::SeqCst);
            crate::ktrace::log_fmt(format_args!(
                "persona.compiled: CACHE HIT sig={sig:#018x} -- replaying {} step(s), skipping inference",
                trace.len()
            ));
            agent.begin_with_plan(intent, trace);
            let result = agent.run_to_completion().to_string();
            bump_uses(sig);
            result
        }
        stale_or_absent => {
            crate::ktrace::log_fmt(format_args!(
                "persona.compiled: CACHE {} sig={sig:#018x} -- planning",
                if matches!(stale_or_absent, Lookup::Stale) { "STALE" } else { "MISS" }
            ));
            agent.begin(intent, planner);
            let result = agent.run_to_completion().to_string();
            // Only compile a *validated* trace. A run that was refused by the
            // taint gate, denied for lack of a capability, or rejected by the
            // grammar is not a trace worth replaying -- and compiling it could
            // let a later replay fire an action the gate had just blocked.
            if is_authorized(&result) {
                let preconds = infer_preconditions(&agent.manifest.name, agent.plan());
                record(sig, intent, preconds, agent.plan().to_vec());
            } else {
                crate::ktrace::log_fmt(format_args!(
                    "persona.compiled: NOT compiling sig={sig:#018x} -- run was not authorized ({result})"
                ));
            }
            result
        }
    }
}

/// Derive the preconditions a trace depends on: the external state it *reads*
/// (a file it didn't write earlier in the trace; a fact it didn't remember
/// earlier), snapshotted as content hashes. Effects the trace itself produces
/// need no precondition -- replaying re-produces them.
fn infer_preconditions(agent: &str, trace: &[Action]) -> Vec<Precondition> {
    let mut written: Vec<String> = Vec::new();
    let mut remembered: Vec<String> = Vec::new();
    let mut preconds: Vec<Precondition> = Vec::new();

    for action in trace {
        match action {
            Action::Call(raw) => {
                let Ok(call) = grammar::parse(raw) else { continue };
                match call.id {
                    registry::MEM_FS_WRITE | registry::MEM_FS_DELETE => {
                        written.push(arg0(&call.args));
                    }
                    registry::MEM_FS_READ => {
                        let path = arg0(&call.args);
                        if !written.iter().any(|w| *w == path) {
                            let expected = crate::synapse::fs::read(&path).map(|b| fnv1a(&b));
                            preconds.push(Precondition::FileContent { path, expected });
                        }
                    }
                    _ => {}
                }
            }
            Action::Remember(key, _) => remembered.push(key.clone()),
            Action::Recall(key) => {
                if !remembered.iter().any(|r| r == key) {
                    let expected = memory::persisted(agent, key).map(|v| fnv1a(v.as_bytes()));
                    preconds.push(Precondition::Fact {
                        agent: String::from(agent),
                        key: key.clone(),
                        expected,
                    });
                }
            }
        }
    }
    preconds
}

/// Whether a run's final result reflects an *authorized* execution (worth
/// compiling), as opposed to a taint refusal, capability denial, or grammar
/// rejection.
fn is_authorized(result: &str) -> bool {
    !(result.starts_with("refused:") || result.starts_with("denied:") || result.starts_with("rejected:"))
}

/// First argument of a parsed call as a string (the FS/recall primitives all
/// take their path/key first, as a string, guaranteed by the grammar).
fn arg0(args: &[ArgValue]) -> String {
    match args.first() {
        Some(ArgValue::Str(s)) => s.clone(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn signature_is_stable_under_trivial_rephrasing() {
        assert_eq!(signature("What is  X "), signature("what is x"));
        assert_ne!(signature("what is x"), signature("what is y"));
    }
}
