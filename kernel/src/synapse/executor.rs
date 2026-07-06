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
use crate::cap::{self, Cap, ChannelId, Right};
use crate::channel;
use crate::sched::{self, TaskId};
use crate::security::Justification;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
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
    let result = run_primitive(caller, spec, &call.args);
    let result_hash = audit::fnv1a(result.as_bytes());
    audit::record(caller, spec.name, args_hash, audit::Outcome::Executed, result_hash);
    Invocation::Executed { primitive: spec.name, result }
}

/// Convenience wrapper for the currently-running task (trusted justification).
pub fn execute_current(raw: &str) -> Invocation {
    execute(sched::current_task_id(), raw)
}

/// Cooperative-blocking budget for `channel_read`: the model never sees an
/// infinite hang — after this many ms with no data it gets `ok:blocked` and
/// re-plans. Native service loops use `channel::read_blocking` directly with
/// their own budget.
const CHANNEL_READ_DEADLINE_MS: u64 = 30_000;

/// Resolve a model-emitted channel handle (`chan`/`max` arg `i`, a `Uint`) as a
/// `Cap` slot index into the CALLER'S OWN table. Returns the channel id and
/// whether the end is the write side. This is what keeps channel handles
/// unforgeable: a guessed integer only ever indexes the caller's own capability
/// space (mirrors `ipc::send`'s `cap::lookup`), never a global channel id.
fn resolve_channel_end(caller: TaskId, args: &[ArgValue], i: usize) -> Option<(ChannelId, bool)> {
    let slot = arg_uint(args, i) as u32;
    match cap::lookup(caller, Cap(slot)) {
        Some(Right::ChannelWrite(c)) => Some((c, true)),
        Some(Right::ChannelRead(c)) => Some((c, false)),
        _ => None,
    }
}

/// Dispatch a validated call to its native implementation. Each arm sees
/// only its typed arguments and the one subsystem its registry entry names;
/// none can reach the caller's memory or another task's state. The grammar
/// has already guaranteed argument arity and types, so the accessors below
/// are total for any call the grammar accepted. `caller` is threaded in so the
/// channel primitives can resolve a handle arg against the caller's own cap
/// table (the fine-grained second gate, on top of `InvokePrimitive`).
fn run_primitive(caller: TaskId, spec: &PrimitiveSpec, args: &[ArgValue]) -> String {
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
        registry::MEM_FS_EDIT => {
            let path = arg_str(args, 0);
            let old = arg_str(args, 1);
            let new = arg_str(args, 2);
            match fs::read(path) {
                Some(bytes) => {
                    let content = String::from_utf8_lossy(&bytes).into_owned();
                    match content.find(old) {
                        Some(at) => {
                            let mut edited = String::with_capacity(content.len() - old.len() + new.len());
                            edited.push_str(&content[..at]);
                            edited.push_str(new);
                            edited.push_str(&content[at + old.len()..]);
                            fs::write(path, edited.as_bytes());
                            format!("ok:edited {path}")
                        }
                        None => format!("error:not_found_substring:{path}"),
                    }
                }
                None => format!("error:not_found:{path}"),
            }
        }
        registry::MEM_FS_SEARCH => {
            let query = arg_str(args, 0);
            let mut hits: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
            for path in fs::list() {
                if let Some(bytes) = fs::read(&path) {
                    if String::from_utf8_lossy(&bytes).contains(query) {
                        hits.push(path);
                    }
                }
            }
            format!("ok:[{}]", hits.join(","))
        }
        registry::CHANNEL_CREATE => {
            let kind = match arg_str(args, 0) {
                "stream" => channel::ChannelKind::Stream,
                "datagram" => channel::ChannelKind::Datagram,
                other => return format!("error:bad_kind:{other}"),
            };
            let id = channel::create(kind, 64 * 1024);
            // Mint both ends into the CALLER'S OWN table and hand back the two
            // slot indices; the caller reads them out of this result and uses
            // them as the `chan` arg on later calls — a capability round-trip
            // that never leaves its own table.
            let read_slot = cap::grant(caller, Right::ChannelRead(id)).0;
            let write_slot = cap::grant(caller, Right::ChannelWrite(id)).0;
            format!("ok:channel_read={read_slot} channel_write={write_slot}")
        }
        registry::CHANNEL_WRITE => {
            match resolve_channel_end(caller, args, 0) {
                Some((c, true)) => {
                    let text = arg_str(args, 1);
                    match channel::try_write(c, text.as_bytes()) {
                        Ok(0) => "ok:blocked".to_string(),
                        Ok(n) => format!("ok:wrote={n}"),
                        Err(e) => format!("error:{e:?}"),
                    }
                }
                _ => {
                    cap::record_denial(caller, "channel_write (handle)");
                    "error:denied_channel_handle".to_string()
                }
            }
        }
        registry::CHANNEL_READ => {
            match resolve_channel_end(caller, args, 0) {
                Some((c, false)) => {
                    let max = (arg_uint(args, 1) as usize).clamp(1, 64 * 1024);
                    let mut buf = vec![0u8; max];
                    let deadline = crate::arch::now_ms() + CHANNEL_READ_DEADLINE_MS;
                    match channel::read_blocking(c, &mut buf, deadline) {
                        Ok(0) => "ok:eof".to_string(),
                        Ok(n) => format!("ok:data={}", String::from_utf8_lossy(&buf[..n])),
                        Err(channel::ChannelError::WouldBlock) => "ok:blocked".to_string(),
                        Err(e) => format!("error:{e:?}"),
                    }
                }
                _ => {
                    cap::record_denial(caller, "channel_read (handle)");
                    "error:denied_channel_handle".to_string()
                }
            }
        }
        registry::CHANNEL_CLOSE => {
            match resolve_channel_end(caller, args, 0) {
                Some((c, is_write)) => {
                    if is_write {
                        channel::close_write(c);
                    } else {
                        channel::close_read(c);
                    }
                    channel::close_end(c);
                    "ok:closed".to_string()
                }
                _ => {
                    cap::record_denial(caller, "channel_close (handle)");
                    "error:denied_channel_handle".to_string()
                }
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
