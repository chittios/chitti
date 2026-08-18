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
use super::{audit, fs, journal, policy};
use crate::agent::types::{CapDomain, Rights, Scope};
use crate::cap::{self, Cap, ChannelId, Right};
use crate::channel;
use crate::sched::{self, TaskId};
use crate::service;
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
    /// The caller held the primitive but the concrete target (path/host/port)
    /// fell outside its granted scope (Gate 2.5). No primitive ran.
    DeniedScope { primitive: &'static str },
    /// The policy requires a human to approve this exact call and none is on
    /// record. No primitive ran. The caller may ask a human and retry; it cannot
    /// assert approval itself.
    NeedsApproval { primitive: &'static str },
}

/// Deterministic id source for `spawn_agent` requests. Real agent lifecycle
/// is Phase 5; here the primitive only mints a stable, auditable request id.
static NEXT_AGENT_ID: AtomicU64 = AtomicU64::new(1);


/// **Dry-run mode.** When set, the gate chain runs in full and the *decision* is
/// reached exactly as it would be, but [`run_primitive`] is never called and the
/// intended call is recorded instead.
///
/// This is the honest way to answer "what would this agent do": the alternative
/// — asking the model to describe its plan — is the model's account of itself,
/// which is precisely the untrusted thing the determinism boundary exists to
/// distrust. Here the *executor* reports, so what you see is what would have run.
///
/// Deliberately a mode on the executor rather than a flag threaded through every
/// caller: an agent must not be able to opt out of it, and a caller that forgot
/// to pass the flag would silently perform real effects during a preview.
static DRY_RUN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Calls recorded during the current dry run, in order.
static PLAN: crate::mm::Locked<alloc::vec::Vec<alloc::string::String>> =
    crate::mm::Locked::new(alloc::vec::Vec::new());

pub fn dry_run_enabled() -> bool {
    DRY_RUN.load(core::sync::atomic::Ordering::Relaxed)
}

/// Begin a dry run, clearing any previous plan.
pub fn begin_dry_run() {
    PLAN.with(|p| p.clear());
    DRY_RUN.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// End the dry run and return what the agent would have done.
pub fn end_dry_run() -> alloc::vec::Vec<alloc::string::String> {
    DRY_RUN.store(false, core::sync::atomic::Ordering::Relaxed);
    PLAN.with(|p| core::mem::take(p))
}

/// Which agent the executor should charge for the calls it is running.
///
/// Set by the agent loop around a turn. Zero means "not an agent" -- kernel and
/// shell calls are not billed to anyone.
static BILL_TO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn bill_to(agent: u64) {
    BILL_TO.store(agent, core::sync::atomic::Ordering::Relaxed);
}

pub fn billed_agent() -> u64 {
    BILL_TO.load(core::sync::atomic::Ordering::Relaxed)
}

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
    // Charged for every outcome below, refused ones included: the tokens that
    // produced a denied call were still spent. See `agent::cost`.
    let billed = billed_agent();

    // Gate 1: grammar.
    let call: Call = match grammar::parse(raw) {
        Ok(call) => call,
        Err(err) => {
            audit::record(caller, "<malformed>", args_hash, audit::Outcome::RejectedMalformed, 0);
            charge(billed, true);
            return Invocation::Rejected(err);
        }
    };
    let spec = registry::by_id(call.id).expect("grammar accepted an unregistered id");

    // Gate 2: capability. The caller must hold InvokePrimitive for this id
    // in its *own* table -- no ambient authority over the ABI.
    if !cap::holds(caller, Right::InvokePrimitive(call.id)) {
        cap::record_denial(caller, spec.name);
        audit::record(caller, spec.name, args_hash, audit::Outcome::DeniedNoCapability, 0);
            charge(billed, true);
        return Invocation::Denied { primitive: spec.name };
    }

    // Consult policy once; `decide` consumes an approval when it uses one, so it
    // must not be called twice for one invocation.
    let policy_verdict = policy::decide(caller, spec.name, spec.effect.is_effectful(), justification.human_confirmed);

    // **Approval check** — deliberately *not* one of the four gates. The four
    // (grammar, capability, taint, scope) are the capability/provenance
    // architecture; this is human policy layered over it, and conflating the two
    // would renumber a chain that the design paper publishes by number.
    //
    // It is here rather than in the tool layer because that layer is the thing
    // being governed. `tools::permissions` and the approval modal were consulted
    // only by the shell's tool dispatch, which suffices while every agent is
    // kernel code reached through the shell and is nothing at all once an agent is
    // a tenant that can call this function directly. See `synapse::policy`.
    //
    // Placed after the capability gate so "you may not do this at all" still wins
    // over "a human must confirm this", and before the taint gate so an untrusted
    // justification is refused outright rather than turned into a question.
    if policy_verdict == policy::Verdict::NeedsApproval {
        crate::ktrace::log_fmt(format_args!(
            "synapse.policy: '{}' by task {caller} needs human approval",
            spec.name
        ));
        audit::record(caller, spec.name, args_hash, audit::Outcome::NeedsApproval, 0);
            charge(billed, true);
        return Invocation::NeedsApproval { primitive: spec.name };
    }

    // Gate 3: taint (Phase 6). A destructive primitive whose justification
    // traces to untrusted ingested content is refused -- this is the
    // prompt-injection-as-privilege-escalation defence, enforced at the OS
    // boundary regardless of how the agent phrased the call. Only an explicit
    // human confirmation at the shell can let it through.
    //
    // Identity-file writes (SOUL.md / MEMORY.md) are treated as destructive
    // even though mem_fs_write is not: those files re-enter the system prompt
    // as trusted persona, so an untrusted write is taint-laundering.
    let identity_write = is_identity_file_mutation(spec, &call.args);
    if (spec.effect.is_effectful() || identity_write) && justification.blocks_destructive() {
        crate::ktrace::log_fmt(format_args!(
            "synapse.taint: REFUSED destructive '{}' by task {caller} -- justification is untrusted ingested content ({:?})",
            spec.name, justification.provenance
        ));
        audit::record(caller, spec.name, args_hash, audit::Outcome::RefusedTainted, 0);
            charge(billed, true);
        return Invocation::RefusedTainted { primitive: spec.name };
    }

    // Gate 3.5: scope. If the caller was granted a *narrow* scope for this
    // domain (a path glob, a host/port range), the concrete target this call
    // touches must fall within it. Tasks with no scope ledger for the domain are
    // unconstrained (preserves primitive-granularity behaviour); a Scope::Any
    // grant covers everything. See `cap::scope_check`.
    //
    // Layered inside Gate 4 (not numbered as a fifth gate — see `GATE_COUNT`):
    // the **login credential record** is unreachable from any primitive. Not
    // writable (it is the gate), not deletable (deleting it removes the gate),
    // not even readable — it is a salt plus a PBKDF2 digest, i.e. an offline
    // cracking target.
    //
    // This has to be an *unconditional* deny rather than a `Scope` returned from
    // `scope_target`, because `cap::scope_check` is deny-only-when-recorded: a
    // task with no scope ledger for the domain — which the orchestrator has not —
    // is unconstrained, so a scope rule here would refuse nothing.
    //
    // NB this is the policy, not the enforcement. `/cat`, `/rm`, `/cp`, `/mv`,
    // `/grep` and `/glob` are shell-bound tools that reach the store without ever
    // entering this function, so the refusal that actually holds is the facade
    // guard in `synapse::fs`.
    if is_credential_access(spec, &call.args) {
        crate::ktrace::log_fmt(format_args!(
            "synapse.scope: DENIED '{}' by task {caller} -- the login credential record",
            spec.name
        ));
        audit::record(caller, spec.name, args_hash, audit::Outcome::DeniedScope, 0);
        charge(billed, true);
        return Invocation::DeniedScope { primitive: spec.name };
    }
    if let Some((domain, want, target)) = scope_target(spec, &call.args) {
        if !cap::scope_check(caller, domain, want, &target) {
            crate::ktrace::log_fmt(format_args!(
                "synapse.scope: DENIED '{}' by task {caller} -- target outside granted scope",
                spec.name
            ));
            audit::record(caller, spec.name, args_hash, audit::Outcome::DeniedScope, 0);
            charge(billed, true);
            return Invocation::DeniedScope { primitive: spec.name };
        }
    }

    // Gate 4: execute in isolation, then audit the effect. The justification's
    // provenance goes in so a write can record *who* authorised the bytes it
    // stored -- see `run_primitive`'s MEM_FS_WRITE/EDIT arms.
    // Dry run: the decision above was reached in full, so this call *would* have
    // happened. Record it and return without touching anything. The audit entry
    // is still written -- a preview is an event worth having in the trail.
    if dry_run_enabled() {
        let rendered = alloc::format!("{} {}", spec.name, render_args(&call.args));
        PLAN.with(|p| p.push(rendered.clone()));
        audit::record(caller, spec.name, args_hash, audit::Outcome::Executed, 0);
        charge(billed, false);
        return Invocation::Executed {
            primitive: spec.name,
            result: alloc::format!("dry-run: would call {rendered}"),
        };
    }

    // Capture the inverse *before* the effect: once a file is overwritten its
    // prior contents are gone, so there is exactly one moment to record them.
    let undo = journal_undo(spec, &call.args);

    let result = run_primitive(caller, spec, &call.args, justification.provenance);
    let result_hash = audit::fnv1a(result.as_bytes());
    audit::record(caller, spec.name, args_hash, audit::Outcome::Executed, result_hash);
    charge(billed, false);
    if let Some(u) = undo {
        journal::record(spec.name, u);
    }
    Invocation::Executed { primitive: spec.name, result }
}

/// Bill one invocation to the agent the loop named, if any.
fn charge(agent: u64, refused: bool) {
    if agent != 0 {
        crate::agent::cost::add_call(agent, refused);
    }
}

/// Render a call's arguments for the dry-run plan, truncating long ones so a
/// preview of a file write does not print the file.
fn render_args(args: &[grammar::ArgValue]) -> alloc::string::String {
    let mut out = alloc::string::String::new();
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        match a {
            grammar::ArgValue::Uint(n) => out.push_str(&alloc::format!("{n}")),
            grammar::ArgValue::Str(s) => {
                if s.len() > 60 {
                    out.push_str(&alloc::format!("\"{}…\" ({} bytes)", &s[..60], s.len()));
                } else {
                    out.push_str(&alloc::format!("\"{s}\""));
                }
            }
        }
    }
    out
}

/// How to reverse this call, or `None` if it changes nothing durable.
///
/// Only store mutations are reversible. A network send or a console write has
/// already left; those are recorded as [`journal::Undo::Irreversible`] so a
/// rollback can say what it could not take back rather than implying it did.
fn journal_undo(
    spec: &registry::PrimitiveSpec,
    args: &[grammar::ArgValue],
) -> Option<journal::Undo> {
    if !journal::is_open() {
        return None;
    }
    let path = match args.first() {
        Some(grammar::ArgValue::Str(p)) => p.clone(),
        _ => alloc::string::String::new(),
    };
    match spec.id {
        registry::MEM_FS_WRITE | registry::MEM_FS_EDIT | registry::MEM_FS_DELETE => {
            Some(match crate::synapse::fs::read(&path) {
                Some(prior) => journal::Undo::RestoreFile { path, prior },
                None => journal::Undo::DeleteFile { path },
            })
        }
        registry::NET_HTTP_GET | registry::NET_HTTP_POST => Some(journal::Undo::Irreversible {
            what: "network request",
            detail: path,
        }),
        registry::CONSOLE_WRITE => Some(journal::Undo::Irreversible {
            what: "console output",
            detail: alloc::string::String::new(),
        }),
        _ => None,
    }
}

/// Convenience wrapper for the currently-running task (trusted justification).
pub fn execute_current(raw: &str) -> Invocation {
    execute(sched::current_task_id(), raw)
}

// --- Gate introspection (the measurement path) -------------------------------

/// The gates a call clears before a primitive runs, in the order
/// [`execute_with_justification`] applies them. 1-based, so 0 can mean "nothing
/// refused".
pub const GATE_GRAMMAR: u8 = 1;
pub const GATE_CAPABILITY: u8 = 2;
pub const GATE_TAINT: u8 = 3;
pub const GATE_SCOPE: u8 = 4;
/// How many gates guard the boundary. Bumping this is the deliberate edit a new
/// gate requires (see [`gate_prefix`]).
///
/// **This number is published.** The design paper states "four gates in a fixed
/// order", draws them, and prices them as cumulative prefixes 1, 1--2, 1--3,
/// 1--4; the attack table cites "Gate 3 (taint)" and "Gate 4 (scope)" by number.
/// Renumbering these silently invalidates a figure, a table and a measurement
/// methodology, so `gate_numbering_is_the_published_contract` pins it — and a new
/// *architectural* gate means updating the paper, not just this constant.
pub const GATE_COUNT: u8 = 4;

/// Run the first `upto` gates of the real chain and report the 1-based gate that
/// refused, or 0 if every gate it ran passed.
///
/// This is the **measurement** path, used by [`super::bench`] to price the
/// authorization decision: it applies exactly the predicates
/// [`execute_with_justification`] applies, in the same order, but never runs a
/// primitive and never writes an audit entry — so timing it prices the gates and
/// nothing else, and a benchmark loop cannot mutate the store or flood the log.
///
/// It is a second copy of the gate order, which is a real drift risk, so it is
/// pinned: `gate_prefix_agrees_with_execute` asserts that for every outcome the
/// real path produces, this function names the matching gate. A gate added above
/// without being added here fails that test.
pub fn gate_prefix(caller: TaskId, raw: &str, justification: Justification, upto: u8) -> u8 {
    // Gate 1: grammar.
    let call: Call = match grammar::parse(raw) {
        Ok(call) => call,
        Err(_) => return GATE_GRAMMAR,
    };
    if upto < GATE_CAPABILITY {
        return 0;
    }
    // The real path `expect`s here; an unregistered id that the grammar accepted
    // is a bug, and a measurement path must not panic the machine over it.
    let Some(spec) = registry::by_id(call.id) else {
        return GATE_GRAMMAR;
    };

    // Gate 2: capability.
    if !cap::holds(caller, Right::InvokePrimitive(call.id)) {
        return GATE_CAPABILITY;
    }
    if upto < GATE_TAINT {
        return 0;
    }

    // Gate 3: taint.
    let identity_write = is_identity_file_mutation(spec, &call.args);
    if (spec.effect.is_effectful() || identity_write) && justification.blocks_destructive() {
        return GATE_TAINT;
    }
    if upto < GATE_SCOPE {
        return 0;
    }

    // Gate 3.5 ("4" here): scope. The credential deny is layered inside it, in
    // the same order as the real path — `gate_prefix_agrees_with_execute` fails
    // if this copy drifts from that one.
    if is_credential_access(spec, &call.args) {
        return GATE_SCOPE;
    }
    if let Some((domain, want, target)) = scope_target(spec, &call.args) {
        if !cap::scope_check(caller, domain, want, &target) {
            return GATE_SCOPE;
        }
    }
    0
}

#[cfg(test)]
mod gate_contract_tests {
    use super::*;

    #[test_case]
    fn gate_numbering_is_the_published_contract() {
        // The design paper publishes these by number: a figure with four boxes,
        // an attack table citing "Gate 3 (taint)" and "Gate 4 (scope)", and a cost
        // methodology measuring cumulative prefixes 1, 1--2, 1--3, 1--4.
        // Renumbering them invalidates a figure, a table and a measurement — so a
        // new *architectural* gate is a paper edit, not just a constant bump.
        //
        // This is not hypothetical: adding the approval check as "Gate 2.75"
        // renumbered taint to 4 and scope to 5, which `synapse::bench`'s own test
        // caught. The approval check is therefore layered over the chain rather
        // than numbered into it, and `gate_of_outcome` reports it as no gate.
        assert_eq!(GATE_GRAMMAR, 1);
        assert_eq!(GATE_CAPABILITY, 2);
        assert_eq!(GATE_TAINT, 3);
        assert_eq!(GATE_SCOPE, 4);
        assert_eq!(GATE_COUNT, 4, "four gates: see the design paper");
        assert_eq!(
            gate_of_outcome(&Invocation::NeedsApproval { primitive: "x" }),
            0,
            "the approval check is not one of the four gates"
        );
    }

    /// No primitive may read, write, edit or delete the login credential record —
    /// with a **fully-granted, trusted** caller, which is the worst case: if the
    /// orchestrator (which holds everything and has no scope ledger) cannot reach
    /// it, nothing can.
    #[test_case]
    fn a_primitive_cannot_touch_the_credential_record() {
        let task = crate::sched::current_task_id();
        for id in [
            registry::MEM_FS_READ,
            registry::MEM_FS_WRITE,
            registry::MEM_FS_EDIT,
            registry::MEM_FS_DELETE,
        ] {
            cap::grant(task, Right::InvokePrimitive(id));
        }
        // Plant a record so a permitted read would actually return bytes — a test
        // against an absent file would pass even with the deny removed.
        {
            let _a = crate::synapse::fs::CredentialAccess::new();
            crate::synapse::fs::write(crate::auth::PATH, b"{\"salt\":\"deadbeef\"}");
        }

        let calls = [
            alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{}"}}}}"#, crate::auth::PATH),
            alloc::format!(
                r#"{{"name":"mem_fs_write","arguments":{{"path":"{}","text":"pwned"}}}}"#,
                crate::auth::PATH
            ),
            alloc::format!(
                r#"{{"name":"mem_fs_edit","arguments":{{"path":"{}","old":"a","new":"b"}}}}"#,
                crate::auth::PATH
            ),
            alloc::format!(r#"{{"name":"mem_fs_delete","arguments":{{"path":"{}"}}}}"#, crate::auth::PATH),
            // Normalisation must not be a way round it.
            alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"//configs//core/auth.json"}}}}"#),
        ];
        for raw in &calls {
            let out = execute_with_justification(task, raw, Justification::trusted());
            assert!(
                matches!(out, Invocation::DeniedScope { .. }),
                "the credential record was reachable via {raw}: {out:?}"
            );
        }

        // The record survived every attempt, and its bytes never leaked.
        let still = {
            let _a = crate::synapse::fs::CredentialAccess::new();
            crate::synapse::fs::read(crate::auth::PATH)
        };
        assert_eq!(still.as_deref(), Some(&b"{\"salt\":\"deadbeef\"}"[..]));

        // Enumeration is an oracle too: neither `list` nor `search` may reveal it.
        let listed = execute_with_justification(task, r#"{"name":"list","arguments":{}}"#, Justification::trusted());
        if let Invocation::Executed { result, .. } = &listed {
            assert!(!result.contains("auth.json"), "list enumerated the credential record: {result}");
        }
        assert!(
            !crate::synapse::fs::list().iter().any(|p| crate::auth::is_credential_path(p)),
            "fs::list leaked the credential record, so glob/grep/readable_paths do too"
        );

        {
            let _a = crate::synapse::fs::CredentialAccess::new();
            crate::synapse::fs::delete(crate::auth::PATH);
        }
    }
}

/// The gate an [`Invocation`] outcome says refused the call — the inverse of
/// [`gate_prefix`]'s return value, and the mapping the equivalence test uses.
pub fn gate_of_outcome(inv: &Invocation) -> u8 {
    match inv {
        Invocation::Rejected(_) => GATE_GRAMMAR,
        Invocation::Denied { .. } => GATE_CAPABILITY,
        Invocation::RefusedTainted { .. } => GATE_TAINT,
        Invocation::DeniedScope { .. } => GATE_SCOPE,
        // Not one of the four gates: the approval check is human policy layered
        // over the chain, not part of the capability/provenance architecture the
        // paper measures. Reported as "no gate refused" because none did.
        Invocation::NeedsApproval { .. } => 0,
        Invocation::Executed { .. } => 0,
    }
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

/// Like [`resolve_channel_end`], but also returns the caller's cap slot so
/// transfer ops can revoke it after granting the target.
fn resolve_channel_end_slot(caller: TaskId, args: &[ArgValue], i: usize) -> Option<(Cap, ChannelId, bool)> {
    let slot = arg_uint(args, i) as u32;
    let cap_h = Cap(slot);
    match cap::lookup(caller, cap_h) {
        Some(Right::ChannelWrite(c)) => Some((cap_h, c, true)),
        Some(Right::ChannelRead(c)) => Some((cap_h, c, false)),
        _ => None,
    }
}

/// Map a validated call to the concrete (domain, rights, target scope) it
/// touches, for the executor's scope gate. `None` = the primitive names no
/// scopeable resource (console, sleep, emit_result, channel ops — channels are
/// gated by their per-end cap, not a scope). This is the single place a
/// primitive's arguments become the scope target the ledger is checked against.
///
/// FS paths are always [`vpath::normalize`]d first so `..` / `//` / `.` cannot
/// slip past a prefix grant like `/agent/7/**`.
fn scope_target(spec: &PrimitiveSpec, args: &[ArgValue]) -> Option<(CapDomain, Rights, Scope)> {
    match spec.id {
        registry::MEM_FS_READ => Some((CapDomain::Fs, Rights::READ, Scope::Path(fs_path(args, 0)))),
        registry::MEM_FS_WRITE => Some((CapDomain::Fs, Rights::WRITE, Scope::Path(fs_path(args, 0)))),
        registry::MEM_FS_EDIT => Some((CapDomain::Fs, Rights::WRITE, Scope::Path(fs_path(args, 0)))),
        registry::MEM_FS_DELETE => Some((CapDomain::Fs, Rights::DELETE, Scope::Path(fs_path(args, 0)))),
        registry::NET_HTTP_GET => net_scope(arg_str(args, 0), Rights::READ),
        registry::NET_HTTP_POST => net_scope(arg_str(args, 0), Rights::WRITE),
        registry::NET_LISTEN => {
            let port = arg_uint(args, 0) as u16;
            Some((
                CapDomain::Net,
                Rights::EXEC,
                Scope::Net {
                    host: String::from("*"),
                    port_lo: port,
                    port_hi: port,
                },
            ))
        }
        _ => None,
    }
}

/// Normalised FS path argument (index `i`) for scope + I/O.
fn fs_path(args: &[ArgValue], i: usize) -> String {
    super::vpath::normalize(arg_str(args, i))
}

/// Paths that re-enter the agent as trusted persona / durable memory. Writing
/// them under untrusted justification is prompt-injection taint-laundering.
fn is_identity_path(path: &str) -> bool {
    let n = super::vpath::normalize(path);
    let base = n.rsplit('/').next().unwrap_or(n.as_str());
    base.eq_ignore_ascii_case("SOUL.md") || base.eq_ignore_ascii_case("MEMORY.md")
}

/// Whether this call mutates an identity file (SOUL.md / MEMORY.md).
fn is_identity_file_mutation(spec: &PrimitiveSpec, args: &[ArgValue]) -> bool {
    match spec.id {
        registry::MEM_FS_WRITE | registry::MEM_FS_EDIT | registry::MEM_FS_DELETE => {
            is_identity_path(arg_str(args, 0))
        }
        _ => false,
    }
}

/// Whether this call touches the login credential record in *any* way.
///
/// Unlike [`is_identity_file_mutation`] this includes **read**: the record is a
/// salt plus a PBKDF2 digest, so handing it to an agent is handing over an
/// offline cracking target. `MEM_FS_LIST` and `MEM_FS_SEARCH` are not listed
/// because they take no path argument — they are covered at the source, by
/// `synapse::fs::list` filtering the record out of every enumeration.
fn is_credential_access(spec: &PrimitiveSpec, args: &[ArgValue]) -> bool {
    match spec.id {
        registry::MEM_FS_READ | registry::MEM_FS_WRITE | registry::MEM_FS_EDIT | registry::MEM_FS_DELETE => {
            crate::auth::is_credential_path(arg_str(args, 0))
        }
        _ => false,
    }
}

/// Build a `Net` scope *target* (a single host:port point) from a URL, for the
/// scope gate. Returns `None` (unconstrained) if the URL doesn't parse — the
/// primitive itself will then reject the malformed URL.
fn net_scope(url: &str, want: Rights) -> Option<(CapDomain, Rights, Scope)> {
    let (_https, host, port, _path) = crate::net::http::parse_url(url).ok()?;
    Some((CapDomain::Net, want, Scope::Net { host, port_lo: port, port_hi: port }))
}

/// Resolve a `net_accept` listener handle (a `Uint` = the caller's cap slot) as
/// a `NetListen` right in the caller's own table.
fn resolve_listener(caller: TaskId, args: &[ArgValue], i: usize) -> Option<crate::cap::ListenerId> {
    let slot = arg_uint(args, i) as u32;
    match cap::lookup(caller, Cap(slot)) {
        Some(Right::NetListen(l)) => Some(l),
        _ => None,
    }
}

/// Resolve a `channel_grant` target agent: either a numeric task id, or a
/// running service name. Returns the target task id, or `None` if it names no
/// live task/service.
fn resolve_agent_target(name: &str) -> Option<TaskId> {
    if let Ok(id) = name.parse::<TaskId>() {
        if sched::is_alive(id) {
            return Some(id);
        }
        return None;
    }
    service::task_for(name)
}

/// Dispatch a validated call to its native implementation. Each arm sees
/// only its typed arguments and the one subsystem its registry entry names;
/// none can reach the caller's memory or another task's state. The grammar
/// has already guaranteed argument arity and types, so the accessors below
/// are total for any call the grammar accepted. `caller` is threaded in so the
/// channel primitives can resolve a handle arg against the caller's own cap
/// table (the fine-grained second gate, on top of `InvokePrimitive`).
///
/// `prov` is the justification this call cleared the taint gate under. The two
/// arms that put bytes in the store record it (`fs::write_tagged`), because a
/// stored object's integrity is a property of *who authorised writing it*, and
/// this is the only place in the kernel that knows both at once.
fn run_primitive(
    caller: TaskId,
    spec: &PrimitiveSpec,
    args: &[ArgValue],
    prov: crate::security::Provenance,
) -> String {
    match spec.id {
        registry::CONSOLE_WRITE => {
            let text = arg_str(args, 0);
            crate::serial_println!("synapse.console: {}", text);
            "ok".to_string()
        }
        registry::MEM_FS_READ => {
            let path = fs_path(args, 0);
            match fs::read(&path) {
                Some(bytes) => format!("ok:{}", String::from_utf8_lossy(&bytes)),
                None => format!("error:not_found:{path}"),
            }
        }
        registry::MEM_FS_WRITE => {
            let path = fs_path(args, 0);
            let text = arg_str(args, 1);
            fs::write_tagged(&path, text.as_bytes(), prov);
            format!("ok:wrote {} bytes to {path}", text.len())
        }
        registry::LIST => {
            // Scope-filtered: a task constrained to a home path (an installed
            // agent) sees only paths it may read, so `list` can't enumerate the
            // whole store. `scope_target` returns None for LIST (no single
            // target), so the confinement is applied here, per result.
            let paths: alloc::vec::Vec<String> = fs::list()
                .into_iter()
                .filter(|p| cap::scope_check(caller, CapDomain::Fs, Rights::READ, &Scope::Path(p.clone())))
                .collect();
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
            let path = fs_path(args, 0);
            if fs::delete(&path) {
                format!("ok:deleted {path}")
            } else {
                format!("error:not_found:{path}")
            }
        }
        registry::MEM_FS_EDIT => {
            // Safer edit: refuse empty `old` and refuse multi-match unless the
            // tools Router rewrote to a single unique occurrence. Default path:
            // unique match only.
            let path = fs_path(args, 0);
            let old = arg_str(args, 1);
            let new = arg_str(args, 2);
            match fs::read(&path) {
                Some(bytes) => {
                    let content = String::from_utf8_lossy(&bytes).into_owned();
                    match crate::tools::pathutil::safe_edit(&content, old, new, false) {
                        Ok(edited) => {
                            fs::write_tagged(&path, edited.as_bytes(), prov);
                            format!("ok:edited {path}")
                        }
                        Err(e) => format!("error:{e}:{path}"),
                    }
                }
                None => format!("error:not_found:{path}"),
            }
        }
        registry::MEM_FS_SEARCH => {
            // Content search, confined to the caller's readable scope (same
            // gate as LIST): an installed agent searches only its own home.
            let query = arg_str(args, 0);
            let mut hits: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
            for path in fs::list() {
                if !cap::scope_check(caller, CapDomain::Fs, Rights::READ, &Scope::Path(path.clone())) {
                    continue;
                }
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
        registry::CHANNEL_GRANT => {
            // Transfer-authority op: the caller may only grant an end it actually
            // holds, to a target it can name. After grant, the caller's slot is
            // revoked so authority is moved (not duplicated). Destructive, so
            // the taint gate already refused it if the justification is untrusted.
            let (caller_cap, chan, is_write) = match resolve_channel_end_slot(caller, args, 0) {
                Some(x) => x,
                None => {
                    cap::record_denial(caller, "channel_grant (handle)");
                    return "error:denied_channel_handle".to_string();
                }
            };
            let target_name = arg_str(args, 1);
            let target = match resolve_agent_target(target_name) {
                Some(t) => t,
                None => return format!("error:no_such_agent:{target_name}"),
            };
            // Target receives the same directional right; caller's slot is
            // cleared so a granted end is not ambiently shared.
            let right = if is_write { Right::ChannelWrite(chan) } else { Right::ChannelRead(chan) };
            let slot = cap::grant(target, right).0;
            let _ = cap::revoke(caller, caller_cap);
            let dir = if is_write { "write" } else { "read" };
            format!("ok:transferred {dir} end to task {target} (slot {slot})")
        }
        registry::NET_LISTEN => {
            let port = arg_uint(args, 0);
            let proto = arg_str(args, 1);
            if proto != "tcp" {
                return format!("error:unsupported_proto:{proto}");
            }
            if port == 0 || port > 65535 {
                return format!("error:bad_port:{port}");
            }
            match crate::net::listen(port as u16) {
                Ok(listener) => {
                    let slot = cap::grant(caller, Right::NetListen(listener)).0;
                    format!("ok:listener={slot} port={port}")
                }
                Err(e) => format!("error:{e}"),
            }
        }
        registry::NET_ACCEPT => match resolve_listener(caller, args, 0) {
            Some(listener) => {
                // Cooperative accept, bounded (the model never hangs forever).
                let deadline = crate::arch::now_ms() + CHANNEL_READ_DEADLINE_MS;
                loop {
                    if let Some(handle) = crate::net::try_accept(listener) {
                        let chan = channel::adopt_tcp(handle);
                        let read_slot = cap::grant(caller, Right::ChannelRead(chan)).0;
                        let write_slot = cap::grant(caller, Right::ChannelWrite(chan)).0;
                        break format!("ok:channel_read={read_slot} channel_write={write_slot}");
                    }
                    if crate::arch::now_ms() >= deadline {
                        break "ok:no_connection".to_string();
                    }
                    crate::shell::upkeep();
                    crate::sched::yield_now();
                }
            }
            _ => {
                cap::record_denial(caller, "net_accept (listener handle)");
                "error:denied_listener_handle".to_string()
            }
        },
        registry::NET_HTTP_GET => {
            let url = arg_str(args, 0);
            match crate::net::http::get(url, 15_000) {
                Ok(resp) => {
                    let body = resp.text();
                    // Char-boundary safe: this is a **remote server's** bytes, so a raw
                    // `&body[..4096]` is a kernel panic any host can trigger by putting a
                    // multi-byte character across byte 4096.
                    let capped = crate::tools::pathutil::truncate_on_char_boundary(&body, 4096);
                    format!("ok:status={} body={}", resp.status, capped)
                }
                Err(e) => format!("error:{e}"),
            }
        }
        registry::NET_HTTP_POST => {
            let url = arg_str(args, 0);
            let body = arg_str(args, 1);
            match crate::net::http::post_json(url, body, None, 15_000) {
                Ok(resp) => format!("ok:status={}", resp.status),
                Err(e) => format!("error:{e}"),
            }
        }
        registry::UI_SURFACE_REQUEST => {
            let kind = match super::ui::SurfaceKind::parse(arg_str(args, 0)) {
                Some(k) => k,
                None => return format!("error:bad_surface_kind:{}", arg_str(args, 0)),
            };
            let id = super::ui::request(caller, kind);
            format!("ok:surface={id}")
        }
        registry::UI_DRAW => {
            let surface = arg_uint(args, 0) as u32;
            let ops = arg_str(args, 1);
            match super::ui::draw(caller, surface, ops) {
                Ok(n) => format!("ok:drew={n}"),
                Err(super::ui::DrawErr::NotOwner) => {
                    cap::record_denial(caller, "ui_draw (not surface owner)");
                    "error:not_surface_owner".to_string()
                }
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::UI_EVENT_POLL => {
            let surface = arg_uint(args, 0) as u32;
            match super::ui::poll(caller, surface) {
                Ok(Some(super::ui::UiEvent::Click { x, y })) => format!("ok:click={x},{y}"),
                Ok(Some(super::ui::UiEvent::Key(k))) => format!("ok:key={k}"),
                Ok(None) => "ok:none".to_string(),
                Err(super::ui::DrawErr::NotOwner) => "error:not_surface_owner".to_string(),
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::UI_SURFACE_CLOSE => {
            let surface = arg_uint(args, 0) as u32;
            match super::ui::close(caller, surface) {
                Ok(()) => "ok:closed".to_string(),
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::BOARD_SET => {
            let surface = arg_uint(args, 0) as u32;
            let fen = arg_str(args, 1);
            match super::ui::board_set(caller, surface, fen) {
                Ok(n) => format!("ok:drew={n}"),
                Err(super::ui::DrawErr::NotOwner) => {
                    cap::record_denial(caller, "board_set (not surface owner)");
                    "error:not_surface_owner".to_string()
                }
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::BOARD_MARK => {
            let surface = arg_uint(args, 0) as u32;
            let squares = arg_str(args, 1);
            let color = arg_str(args, 2);
            match super::ui::board_mark(caller, surface, squares, color) {
                Ok(n) => format!("ok:drew={n}"),
                Err(super::ui::DrawErr::NotOwner) => {
                    cap::record_denial(caller, "board_mark (not surface owner)");
                    "error:not_surface_owner".to_string()
                }
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::UI_HUD => {
            let surface = arg_uint(args, 0) as u32;
            let text = arg_str(args, 1);
            match super::ui::set_hud(caller, surface, text) {
                Ok(()) => "ok:hud".to_string(),
                Err(super::ui::DrawErr::NotOwner) => {
                    cap::record_denial(caller, "ui_hud (not surface owner)");
                    "error:not_surface_owner".to_string()
                }
                Err(e) => format!("error:{e:?}"),
            }
        }
        registry::UI_PRESENT => {
            let surface = arg_uint(args, 0) as u32;
            let w = arg_uint(args, 1) as usize;
            let h = arg_uint(args, 2) as usize;
            match super::ui::present_pixels(caller, surface, w, h) {
                Ok(n) => format!("ok:presented={n}"),
                Err(super::ui::DrawErr::NotOwner) => {
                    cap::record_denial(caller, "ui_present (not surface owner)");
                    "error:not_surface_owner".to_string()
                }
                Err(e) => format!("error:{e:?}"),
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
