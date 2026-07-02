//! The **agentic loop** (`CHITTI_AGENTIC_HANDOFF.md` Phase A): the Claude-Code
//! control cycle `model → (tool_calls) → Synapse validate+execute →
//! tool_results → model`, repeated until the agent emits a final result or a
//! budget stops it.
//!
//! Two seams keep the loop testable and the real model a drop-in:
//!
//! * [`StepSource`] — produces the next [`Step`] (a batch of tool calls, or the
//!   final answer) given the session so far. The real implementation wraps
//!   Cortex with grammar-constrained decoding; the deterministic
//!   [`rule_steps`](super::rule_steps) implementation drives the tests and boot
//!   demo (temp-0-equivalent, reproducible — see DECISIONS.md #4/#6).
//! * [`ToolDispatch`] — validates + executes one tool call and returns its
//!   result. Every effect flows through Synapse here (locked invariant #1); the
//!   loop never touches the FS/console directly.
//!
//! Budgets (`max_turns`, `max_tool_calls`) are enforced every iteration, so a
//! runaway model can never loop or spend forever.

use crate::agent::types::*;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What the model decided to do this turn.
pub enum Step {
    /// Emit one or more tool calls (executed, their results appended, loop
    /// continues).
    Tools(Vec<ToolCall>),
    /// Final answer — the loop appends it as an assistant message and stops.
    Final(String),
}

/// The result of one tool call: its output text and the provenance that output
/// carries into context (file/tool output is `UntrustedIngested`; a trusted
/// internal op is `SystemTrusted`).
pub struct ToolOutcome {
    pub result: String,
    pub provenance: Provenance,
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn ok(result: impl Into<String>, provenance: Provenance) -> Self {
        Self { result: result.into(), provenance, is_error: false }
    }
    pub fn error(result: impl Into<String>) -> Self {
        Self { result: result.into(), provenance: Provenance::SystemTrusted, is_error: true }
    }
}

/// Produces the next [`Step`]. `session` is read-only here — the loop owns all
/// mutation so a StepSource can't corrupt budget/message accounting.
pub trait StepSource {
    fn next(&mut self, session: &Session) -> Step;
}

/// Validates and executes one tool call, returning its outcome. Implemented by
/// the Phase B tools layer over Synapse.
pub trait ToolDispatch {
    fn call(&mut self, session: &mut Session, caller: crate::sched::TaskId, call: &ToolCall) -> ToolOutcome;
}

/// Why the loop stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    /// The model emitted a final answer.
    Final,
    /// Hit `budgets.max_turns`.
    BudgetTurns,
    /// Hit `budgets.max_tool_calls`.
    BudgetToolCalls,
}

/// The outcome of running the loop to a stop condition.
pub struct LoopResult {
    pub answer: String,
    pub stop: StopReason,
    pub turns: u32,
    pub tool_calls: u32,
}

/// Drive `session` with `steps` + `tools` until a final answer or a budget
/// stop. `caller` is the task whose capability table gates every tool call
/// (no ambient authority). `now` is a tick source (monotonic).
pub fn run(
    session: &mut Session,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    caller: crate::sched::TaskId,
    now: impl Fn() -> Ticks,
) -> LoopResult {
    let limits = session.budget.limits;
    loop {
        // Turn budget: stop before asking the model again.
        if session.budget.turns_used >= limits.max_turns {
            let answer = "stopped: turn budget exhausted".to_string();
            session.push_message(Role::Assistant, answer.clone(), Provenance::SystemTrusted, now());
            crate::ktrace::log_fmt(format_args!("agent.loop: session {} hit max_turns", session.id.0));
            return LoopResult {
                answer,
                stop: StopReason::BudgetTurns,
                turns: session.budget.turns_used,
                tool_calls: session.budget.tool_calls_used,
            };
        }
        session.budget.turns_used += 1;

        match steps.next(session) {
            Step::Final(answer) => {
                session.push_message(Role::Assistant, answer.clone(), Provenance::SystemTrusted, now());
                crate::ktrace::log_fmt(format_args!(
                    "agent.loop: session {} final after {} turns / {} tool calls",
                    session.id.0, session.budget.turns_used, session.budget.tool_calls_used
                ));
                return LoopResult {
                    answer,
                    stop: StopReason::Final,
                    turns: session.budget.turns_used,
                    tool_calls: session.budget.tool_calls_used,
                };
            }
            Step::Tools(calls) => {
                // Record the assistant's tool-call turn, then execute each call
                // and append its result. Tool-call budget is checked per call.
                session.push_assistant_tool_calls(String::new(), calls.clone(), now());
                for call in &calls {
                    if session.budget.tool_calls_used >= limits.max_tool_calls {
                        let answer = "stopped: tool-call budget exhausted".to_string();
                        session.push_message(Role::Assistant, answer.clone(), Provenance::SystemTrusted, now());
                        return LoopResult {
                            answer,
                            stop: StopReason::BudgetToolCalls,
                            turns: session.budget.turns_used,
                            tool_calls: session.budget.tool_calls_used,
                        };
                    }
                    session.budget.tool_calls_used += 1;
                    let outcome = tools.call(session, caller, call);
                    session.push_tool_result(call.call_id, outcome.result, outcome.provenance, now());
                }
                // Keep the context within budget: compact older turns once the
                // live-token count approaches the window (Phase D).
                crate::agent::context::maybe_compact(session, now());
            }
        }
    }
}
