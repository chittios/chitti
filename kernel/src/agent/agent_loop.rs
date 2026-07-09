//! The **agentic loop** (`CHITTI_AGENTIC_HANDOFF.md` Phase A): the control
//! cycle `model → (tool_calls) → Synapse validate+execute → tool_results →
//! model`, repeated until the agent emits a final result or a budget stops it.
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
    /// Human cancel (Ctrl+C / Esc) polled via [`run_with_cancel`]'s cancel hook.
    Cancelled,
}

/// The outcome of running the loop to a stop condition.
pub struct LoopResult {
    pub answer: String,
    pub stop: StopReason,
    pub turns: u32,
    pub tool_calls: u32,
}

/// Format a tool outcome for the session transcript. Errors always carry a
/// stable `error:` / `denied:` / `refused:` prefix so a StepSource (or the
/// chat model) can tell success from failure without a separate schema field.
pub fn format_tool_result(is_error: bool, result: String) -> String {
    if !is_error {
        return result;
    }
    if result.starts_with("error:")
        || result.starts_with("denied:")
        || result.starts_with("refused:")
        || result.starts_with("Denied:")
    {
        result
    } else {
        alloc::format!("error: {result}")
    }
}

/// Drive `session` with `steps` + `tools` until a final answer or a budget
/// stop. `caller` is the task whose capability table gates every tool call
/// (no ambient authority). `now` is a tick source (monotonic).
///
/// Does not poll for human cancel — use [`run_with_cancel`] when the loop
/// should honour Ctrl+C / Esc (interactive shell / sub-agent with a console).
pub fn run(
    session: &mut Session,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    caller: crate::sched::TaskId,
    now: impl Fn() -> Ticks,
) -> LoopResult {
    run_with_cancel(session, steps, tools, caller, now, || false)
}

/// Like [`run`], but polls `cancel` before each model step and between tool
/// calls. When `cancel` returns true the loop appends a cancelled assistant
/// message and returns [`StopReason::Cancelled`] — no further tools run.
pub fn run_with_cancel(
    session: &mut Session,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    caller: crate::sched::TaskId,
    now: impl Fn() -> Ticks,
    mut cancel: impl FnMut() -> bool,
) -> LoopResult {
    let limits = session.budget.limits;
    loop {
        if cancel() {
            return cancelled(session, &now);
        }
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
                // Concurrent-safe (read-only) batches: still run on one core
                // (cooperative scheduler) but skip intermediate cancel/budget
                // checks between pure reads — one compact after the batch.
                let all_readonly = !calls.is_empty()
                    && calls
                        .iter()
                        .all(|c| crate::tools::permissions::is_readonly_tool(&c.tool));
                if all_readonly && calls.len() > 1 {
                    crate::ktrace::log_fmt(format_args!(
                        "agent.loop: concurrent-safe batch of {} read-only tools",
                        calls.len()
                    ));
                }
                for call in &calls {
                    if !all_readonly {
                        if cancel() {
                            return cancelled(session, &now);
                        }
                    }
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
                    // Preserve `is_error` in the content the model sees: the
                    // Message schema has no separate error bit (postcard v1),
                    // so a stable `error:` prefix is the contract.
                    let text = format_tool_result(outcome.is_error, outcome.result);
                    session.push_tool_result(call.call_id, text, outcome.provenance, now());
                }
                if all_readonly && cancel() {
                    return cancelled(session, &now);
                }
                // Keep the context within budget: compact older turns once the
                // live-token count approaches the window (Phase D).
                crate::agent::context::maybe_compact(session, now());
            }
        }
    }
}

fn cancelled(session: &mut Session, now: &impl Fn() -> Ticks) -> LoopResult {
    let answer = "stopped: cancelled".to_string();
    session.push_message(Role::Assistant, answer.clone(), Provenance::SystemTrusted, now());
    crate::ktrace::log_fmt(format_args!(
        "agent.loop: session {} cancelled after {} turns / {} tool calls",
        session.id.0, session.budget.turns_used, session.budget.tool_calls_used
    ));
    LoopResult {
        answer,
        stop: StopReason::Cancelled,
        turns: session.budget.turns_used,
        tool_calls: session.budget.tool_calls_used,
    }
}
