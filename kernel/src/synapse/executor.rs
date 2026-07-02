//! The Synapse **executor** (`CHITTI_OS_HANDOFF.md` Phase 4): the single
//! path from a model-emitted tool call to a real, deterministic effect. It
//! is the load-bearing crossing of the determinism boundary -- every call
//! runs the same three gates, in order:
//!
//! 1. **Grammar.** `grammar::parse` must accept the call as complete and
//!    well-formed. A malformed call is rejected here and no primitive runs.
//! 2. **Capability.** The caller's own capability table must grant
//!    `Right::InvokePrimitive(id)`. No ambient authority: holding no
//!    capability means the call is denied and audited.
//! 3. **Execution.** Only then does native code run the primitive, in an
//!    isolated context (it sees nothing but its typed arguments and the
//!    subsystem the registry entry names -- never the caller's memory,
//!    stack, or capability table).
//!
//! Whatever the outcome -- executed, denied, or rejected -- exactly one
//! append-only audit entry is written. Model output never reaches a
//! primitive except through this function.

use super::grammar::{self, ArgValue, Call, GrammarError};
use super::registry::{self, PrimitiveSpec};
use super::{audit, fs};
use crate::cap::{self, Right};
use crate::sched::{self, TaskId};
use crate::security::Justification;
use alloc::format;
use alloc::string::{String, ToString};
use core::sync::atomic::{AtomicU64, Ordering};

/// The structured result of an invocation attempt, returned to the caller
/// (and, in later phases, fed back to the model as the tool's output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invocation {
    /// The primitive ran; `result` is its structured output text.
    Executed { primitive: &'static str, result: String },
    /// Grammar-valid but the caller lacked `InvokePrimitive` authority.
    Denied { primitive: &'static str },
    /// The grammar rejected the call; no primitive ran.
    Rejected(GrammarError),
    /// A destructive primitive refused by the taint gate (Phase 6): the
    /// justification traced to untrusted ingested content and no human
    /// confirmed it. No primitive ran.
    RefusedTainted { primitive: &'static str },
}

/// Deterministic id source for `spawn_agent` requests. Real agent lifecycle
/// is Phase 5; here the primitive only mints a stable, auditable request id.
static NEXT_AGENT_ID: AtomicU64 = AtomicU64::new(1);

/// Run a tool call emitted by `caller`, with a fully-trusted justification.
/// This is the pre-Phase-6 entry point; it never trips the taint gate, so
/// system/kernel-internal callers keep their prior behaviour. Callers that
/// carry untrusted context (agents) use [`execute_with_justification`].
pub fn execute(caller: TaskId, raw: &str) -> Invocation {
    execute_with_justification(caller, raw, Justification::trusted())
}

/// Run a tool call emitted by `caller`, gating destructive primitives on the
/// provenance of `justification`. See the module doc for the ordered gates;
/// Phase 6 inserts a fourth (taint) gate before execution. Every path writes
/// exactly one audit entry.
pub fn execute_with_justification(caller: TaskId, raw: &str, justification: Justification) -> Invocation {
    let args_hash = audit::fnv1a(raw.as_bytes());

    // Gate 1: grammar.
    let call: Call = match grammar::parse(raw) {
        Ok(call) => call,
        Err(err) => {
            audit::record(caller, "<malformed>", args_hash, audit::Outcome::RejectedMalformed, 0);
            return Invocation::Rejected(err);
        }
    };
    let spec = registry::by_id(call.id).expect("grammar accepted an unregistered id");

    // Gate 2: capability. The caller must hold InvokePrimitive for this id
    // in its *own* table -- no ambient authority over the ABI.
    if !cap::holds(caller, Right::InvokePrimitive(call.id)) {
        cap::record_denial(caller, spec.name);
        audit::record(caller, spec.name, args_hash, audit::Outcome::DeniedNoCapability, 0);
        return Invocation::Denied { primitive: spec.name };
    }

    // Gate 3: taint (Phase 6). A destructive primitive whose justification
    // traces to untrusted ingested content is refused -- this is the
    // prompt-injection-as-privilege-escalation defence, enforced at the OS
    // boundary regardless of how the agent phrased the call. Only an explicit
    // human confirmation at the shell can let it through.
    if spec.destructive && justification.blocks_destructive() {
        crate::ktrace::log_fmt(format_args!(
            "synapse.taint: REFUSED destructive '{}' by task {caller} -- justification is untrusted ingested content ({:?})",
            spec.name, justification.provenance
        ));
        audit::record(caller, spec.name, args_hash, audit::Outcome::RefusedTainted, 0);
        return Invocation::RefusedTainted { primitive: spec.name };
    }

    // Gate 4: execute in isolation, then audit the effect.
    let result = run_primitive(spec, &call.args);
    let result_hash = audit::fnv1a(result.as_bytes());
    audit::record(caller, spec.name, args_hash, audit::Outcome::Executed, result_hash);
    Invocation::Executed { primitive: spec.name, result }
}

/// Convenience wrapper for the currently-running task (trusted justification).
pub fn execute_current(raw: &str) -> Invocation {
    execute(sched::current_task_id(), raw)
}

/// Dispatch a validated call to its native implementation. Each arm sees
/// only its typed arguments and the one subsystem its registry entry names;
/// none can reach the caller's memory or another task's state. The grammar
/// has already guaranteed argument arity and types, so the accessors below
/// are total for any call the grammar accepted.
fn run_primitive(spec: &PrimitiveSpec, args: &[ArgValue]) -> String {
    match spec.id {
        registry::CONSOLE_WRITE => {
            let text = arg_str(args, 0);
            crate::serial_println!("synapse.console: {}", text);
            "ok".to_string()
        }
        registry::MEM_FS_READ => {
            let path = arg_str(args, 0);
            match fs::read(path) {
                Some(bytes) => format!("ok:{}", String::from_utf8_lossy(&bytes)),
                None => format!("error:not_found:{path}"),
            }
        }
        registry::MEM_FS_WRITE => {
            let path = arg_str(args, 0);
            let text = arg_str(args, 1);
            fs::write(path, text.as_bytes());
            format!("ok:wrote {} bytes to {path}", text.len())
        }
        registry::LIST => {
            let paths = fs::list();
            format!("ok:[{}]", paths.join(","))
        }
        registry::SPAWN_AGENT => {
            // Lifecycle is Phase 5; here we only mint an auditable request id.
            let persona = arg_str(args, 0);
            let id = NEXT_AGENT_ID.fetch_add(1, Ordering::SeqCst);
            format!("ok:agent_requested id={id} persona={persona}")
        }
        registry::SLEEP => {
            // Deterministic no-op in Phase 4: records intent without coupling
            // to the scheduler (which would make test timing nondeterministic).
            let ticks = arg_uint(args, 0);
            format!("ok:slept {ticks} ticks")
        }
        registry::EMIT_RESULT => {
            let text = arg_str(args, 0);
            format!("ok:result={text}")
        }
        registry::MEM_FS_DELETE => {
            // Destructive: only reached once the taint gate (above) has let
            // this call through.
            let path = arg_str(args, 0);
            if fs::delete(path) {
                format!("ok:deleted {path}")
            } else {
                format!("error:not_found:{path}")
            }
        }
        other => format!("error:unimplemented primitive id {other}"),
    }
}

fn arg_str<'a>(args: &'a [ArgValue], i: usize) -> &'a str {
    match &args[i] {
        ArgValue::Str(s) => s,
        ArgValue::Uint(_) => unreachable!("grammar guaranteed a string at arg {i}"),
    }
}

fn arg_uint(args: &[ArgValue], i: usize) -> u64 {
    match &args[i] {
        ArgValue::Uint(u) => *u,
        ArgValue::Str(_) => unreachable!("grammar guaranteed a uint at arg {i}"),
    }
}
