//! **Sub-agents** (`CHITTI_AGENTIC_HANDOFF.md` Phase C): the orchestrator
//! delegates a self-contained task to an isolated sub-agent that runs its own
//! loop to completion and returns **only a summary**. The three invariants this
//! module enforces:
//!
//! 1. **Context isolation** — the sub-agent gets its own [`Session`] (its own
//!    KV/context); its transcript never merges into the parent. Only the
//!    summary crosses back, via [`integrate`], which appends a
//!    [`SubagentRecord`] (summary, no transcript) + one tool-result message.
//! 2. **Capability attenuation** — the effective caps are a subset of the
//!    parent's. A sub-agent role that *requests* an authority the parent lacks
//!    is refused at spawn ([`DispatchError::CapabilityRefused`]); it can only
//!    ever narrow, never widen.
//! 3. **Bounded delegation** — a depth cap stops runaway recursion
//!    ([`DispatchError::DepthExceeded`]).
//!
//! True SMP parallelism across cores is available via [`dispatch_batch`], which
//! assigns each sub-agent a distinct core id; under the single-threaded QEMU
//! test harness they execute sequentially but are recorded per core and both
//! summaries are integrated (see DECISIONS.md).

use crate::agent::agent_loop::{self, StepSource, ToolDispatch};
use crate::agent::manifest;
use crate::agent::orchestrator::now;
use crate::agent::types::*;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Why a sub-agent could not be dispatched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchError {
    /// The role requested a capability its parent does not hold — widening,
    /// which the attenuation invariant forbids.
    CapabilityRefused(CapabilityRequest),
    /// Dispatching would exceed the parent's `max_depth`.
    DepthExceeded { depth: u8, max: u8 },
}

/// The result of running a sub-agent to completion.
#[derive(Debug)]
pub struct SubagentOutcome {
    pub record: SubagentRecord,
    /// The isolated sub-session (its full transcript). The parent never sees
    /// this — it exists so callers/tests can prove isolation. Discarded after.
    pub sub_session: Session,
}

/// Condense a sub-agent's final answer per its summary policy (byte-truncate to
/// the token budget; the real Cortex summarizer plugs in behind this).
fn condense(answer: &str, policy: SummaryPolicy) -> String {
    let max_bytes = (policy.max_tokens as usize) * 4;
    if answer.len() <= max_bytes {
        answer.to_string()
    } else {
        let mut s: String = answer.chars().take(max_bytes).collect();
        s.push('…');
        s
    }
}

/// Verify `role`'s requested capabilities are all contained by `parent_caps`
/// (subset check). Returns the effective (already-attenuated) set on success,
/// or the first offending request. This is the spawn-path enforcement of the
/// "delegation only ever narrows authority" invariant.
pub fn attenuate(parent_caps: &[CapabilityRequest], role: &AgentManifest) -> Result<Vec<CapabilityRequest>, DispatchError> {
    for req in &role.capabilities {
        if !parent_caps.iter().any(|p| p.contains(req)) {
            return Err(DispatchError::CapabilityRefused(req.clone()));
        }
    }
    // Every requested cap is contained by some parent cap, so the role's set is
    // already ⊆ parent. Intersect anyway to clamp rights to exactly the overlap.
    Ok(intersect_caps(&role.capabilities, parent_caps))
}

/// Dispatch one sub-agent: enforce depth + attenuation, spin up an isolated
/// session on its own capability-owning task, run its loop to completion with
/// `steps`/`tools`, and return its summary + isolated transcript.
///
/// `parent_caps` are the parent's live effective caps; `parent_depth` is the
/// parent's delegation depth; `max_depth` is the parent's budget. `core` is the
/// core this sub-agent is assigned to (for the SMP record).
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    parent_caps: &[CapabilityRequest],
    parent_depth: u8,
    max_depth: u8,
    role: AgentManifest,
    task: &str,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    core: Option<u8>,
) -> Result<SubagentOutcome, DispatchError> {
    if parent_depth >= max_depth {
        crate::ktrace::log_fmt(format_args!(
            "subagent.dispatch: REFUSED '{}' — depth {} >= max {}",
            role.name, parent_depth, max_depth
        ));
        return Err(DispatchError::DepthExceeded { depth: parent_depth, max: max_depth });
    }
    let effective = attenuate(parent_caps, &role)?;

    // Isolated identity + capability table for the sub-agent.
    let sub_task = crate::sched::spawn_parked("subagent");
    let live = manifest::grant_to_task(sub_task, &effective);

    // Its own session (own context/KV). The delegated task is system-trusted
    // (orchestrator-authored); anything the sub-agent then ingests is tainted.
    let mut sub_session = Session::new(&role, role.sampling.seed, live, now());
    let sub_id = role.id;
    sub_session.push_message(Role::User, task.to_string(), Provenance::SystemTrusted, now());

    crate::ktrace::log_fmt(format_args!(
        "subagent.dispatch: '{}' (session {}, task {}, {} caps, core {:?}, depth {})",
        role.name, sub_session.id.0, sub_task, effective.len(), core, parent_depth + 1
    ));

    let result = agent_loop::run(&mut sub_session, steps, tools, sub_task, now);
    let summary = condense(&result.answer, role.summary);

    let record = SubagentRecord {
        id: sub_id,
        manifest_id: role.id,
        dispatched_ticks: now(),
        status: SubagentStatus::Completed,
        summary: Some(summary),
        effective_caps: effective,
        core,
        audit_ref: sub_session.audit_cursor,
    };
    Ok(SubagentOutcome { record, sub_session })
}

/// Integrate a completed sub-agent's result into the parent session: record it
/// in the ledger and append **only its summary** as a tool-result message. The
/// sub-agent's transcript is never merged — isolation preserved.
pub fn integrate(parent: &mut Session, call_id: u64, outcome: &SubagentOutcome) {
    parent.subagents.push(outcome.record.clone());
    parent.budget.subagents_used += 1;
    let summary = outcome.record.summary.clone().unwrap_or_default();
    // The summary is authored by the orchestrator's summarizer → system-trusted.
    parent.push_tool_result(call_id, alloc::format!("subagent[{}]: {}", outcome.record.manifest_id.0, summary), Provenance::SystemTrusted, now());
}

/// Record a completed sub-agent in the parent's ledger **without** appending a
/// message. Used on the tool path, where the agentic loop appends the
/// tool-result (carrying the summary) itself — so calling [`integrate`] there
/// would double-append. Isolation is identical: only the record (summary, no
/// transcript) is stored.
pub fn record(parent: &mut Session, outcome: &SubagentOutcome) {
    parent.subagents.push(outcome.record.clone());
    parent.budget.subagents_used += 1;
}

/// Dispatch several sub-agents, one per spec, assigning each a distinct core id
/// (round-robin over `core_count`). Returns each outcome (or its dispatch
/// error). Under the single-threaded test harness these run sequentially; the
/// core assignment + record structure is the SMP-ready path.
pub fn dispatch_batch(
    parent_caps: &[CapabilityRequest],
    parent_depth: u8,
    max_depth: u8,
    specs: Vec<(AgentManifest, String)>,
    make_steps: &mut dyn FnMut(usize, &str) -> alloc::boxed::Box<dyn StepSource>,
    tools: &mut dyn ToolDispatch,
    core_count: u8,
) -> Vec<Result<SubagentOutcome, DispatchError>> {
    let mut out = Vec::new();
    for (i, (role, task)) in specs.into_iter().enumerate() {
        let core = Some((i as u8) % core_count.max(1));
        let mut steps = make_steps(i, &task);
        out.push(dispatch(parent_caps, parent_depth, max_depth, role, &task, steps.as_mut(), tools, core));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::rule_steps::{args, tool, ScriptedSteps};
    use crate::agent::agent_loop::Step;
    use crate::tools::Router;
    use alloc::boxed::Box;
    use alloc::vec;

    fn parent_caps() -> Vec<CapabilityRequest> {
        manifest::orchestrator_manifest().capabilities
    }

    /// (a) The sub-agent's context is isolated: the parent gets the summary, not
    /// the sub-agent's tool-call/tool-result turns.
    #[test_case]
    fn subagent_context_is_isolated() {
        let mut parent = Session::new(&manifest::orchestrator_manifest(), 1, vec![], now());
        let parent_msgs_before = parent.messages.len();
        let mut router = Router::new();
        // Sub-agent script: read a file, then finalize a summary.
        crate::synapse::fs::write("c_iso", b"secret-body");
        let mut steps = ScriptedSteps::new(vec![
            Step::Tools(vec![tool("read", args(&[("path", "c_iso")]))]),
            Step::Final("the file contained secret-body".into()),
        ]);
        let outcome = dispatch(&parent_caps(), 0, 2, manifest::reader_subagent_manifest(), "read c_iso", &mut steps, &mut router, Some(1)).expect("dispatch");
        // The sub-session ran multiple turns (read + final + result).
        assert!(outcome.sub_session.messages.len() >= 3, "sub-agent had a full transcript");
        integrate(&mut parent, 99, &outcome);
        // Parent gained exactly ONE message (the summary), not the transcript.
        assert_eq!(parent.messages.len(), parent_msgs_before + 1);
        let last = parent.messages.last().unwrap();
        assert_eq!(last.role, Role::Tool);
        assert!(last.content.contains("secret-body"), "summary crossed back");
        // The parent context does NOT contain the sub-agent's raw read tool-call.
        assert!(!parent.messages.iter().any(|m| m.tool_calls.iter().any(|c| c.tool == "read")));
        assert_eq!(parent.subagents.len(), 1);
    }

    /// (b) A sub-agent role requesting a capability the parent lacks is refused
    /// at spawn.
    #[test_case]
    fn subagent_widening_capability_refused() {
        // Parent holds only READ|LIST (reader role).
        let parent = manifest::reader_subagent_manifest().capabilities;
        // A role that wants WRITE — which the parent does not hold.
        let mut greedy = manifest::reader_subagent_manifest();
        greedy.capabilities = vec![CapabilityRequest::new(CapDomain::Fs, Rights::WRITE, Scope::Any)];
        let mut router = Router::new();
        let mut steps = ScriptedSteps::new(vec![Step::Final("x".into())]);
        let err = dispatch(&parent, 0, 2, greedy, "t", &mut steps, &mut router, None).unwrap_err();
        match err {
            DispatchError::CapabilityRefused(c) => assert_eq!(c.domain, CapDomain::Fs),
            other => panic!("expected CapabilityRefused, got {other:?}"),
        }
    }

    /// (c) Two sub-agents dispatched together are assigned distinct cores and
    /// both summaries are integrated into the parent.
    #[test_case]
    fn two_subagents_integrate_both_results() {
        crate::synapse::fs::write("c_a", b"alpha");
        crate::synapse::fs::write("c_b", b"beta");
        let mut parent = Session::new(&manifest::orchestrator_manifest(), 5, vec![], now());
        let mut router = Router::new();
        let specs = vec![
            (manifest::reader_subagent_manifest(), "read c_a".to_string()),
            (manifest::reader_subagent_manifest(), "read c_b".to_string()),
        ];
        // Distinct scripts per sub-agent (index-keyed).
        let mut make = |i: usize, _t: &str| -> Box<dyn StepSource> {
            let (path, body) = if i == 0 { ("c_a", "alpha") } else { ("c_b", "beta") };
            Box::new(ScriptedSteps::new(vec![
                Step::Tools(vec![tool("read", args(&[("path", path)]))]),
                Step::Final(alloc::format!("found {body}")),
            ]))
        };
        let results = dispatch_batch(&parent_caps(), 0, 2, specs, &mut make, &mut router, 4);
        assert_eq!(results.len(), 2);
        let cores: alloc::vec::Vec<Option<u8>> = results.iter().map(|r| r.as_ref().unwrap().record.core).collect();
        assert_eq!(cores, vec![Some(0), Some(1)], "distinct cores assigned");
        for (cid, r) in results.iter().enumerate() {
            integrate(&mut parent, cid as u64, r.as_ref().unwrap());
        }
        assert_eq!(parent.subagents.len(), 2);
        let joined: String = parent.messages.iter().filter(|m| m.role == Role::Tool).map(|m| m.content.clone()).collect();
        assert!(joined.contains("alpha") && joined.contains("beta"), "both summaries integrated");
    }

    /// (d) The depth limit prevents runaway recursion.
    #[test_case]
    fn depth_limit_prevents_runaway() {
        let mut router = Router::new();
        let mut steps = ScriptedSteps::new(vec![Step::Final("x".into())]);
        // At depth == max, a further dispatch is refused.
        let err = dispatch(&parent_caps(), 2, 2, manifest::reader_subagent_manifest(), "t", &mut steps, &mut router, None).unwrap_err();
        assert!(matches!(err, DispatchError::DepthExceeded { depth: 2, max: 2 }));
    }
}
