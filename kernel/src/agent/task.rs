//! Running an agent's loop on **its own scheduler task**.
//!
//! This is what "agents are processes" has to mean to be true. Until now an agent
//! had a task id and a capability table, but its plan/act loop executed on the
//! *caller's* stack — the shell's, or a test's — so the task was an identity holder
//! and nothing else. Two consequences, both of which this module removes:
//!
//! * The identity that the gates check and the stack the work runs on were different
//!   things, and keeping them consistent was a convention rather than a mechanism.
//!   Here they are the same task by construction: the loop runs *as* the identity
//!   whose capabilities gate its calls, so `current_task_id()` inside a tool is the
//!   agent, not whoever invoked it.
//! * An agent's depth was charged to its caller's stack. A sub-agent nested inside an
//!   agent inside the shell shared one stack, and the overflow that produces is a
//!   silent triple fault (see [`crate::sched::spawn_with_stack`]). Now each agent
//!   brings its own.
//!
//! # The loan, and why it is sound
//!
//! A task entry is `extern "C" fn(u64)`, so the loop's four `&mut` borrows — session,
//! step source, tool dispatch, clock — cannot be passed as arguments. They are packed
//! into a [`Job`] on the caller's frame and reached through a raw pointer.
//!
//! That is sound because the caller **blocks until the task has finished**: it hands
//! the borrows over, touches nothing until [`crate::sched::join`] returns, and only
//! then reads the result back. So there is exactly one accessor at any moment and the
//! borrows outlive the task, which is the same loan the video player makes when it
//! hands its `StreamDecoder` to an SMP worker.
//!
//! The dangerous version of this is a caller that returns early — on a cancel, say —
//! while the task is still running, leaving it writing through a dangling pointer.
//! [`run_on_own_task`] therefore has no early return between spawn and join, and
//! cancellation is delivered *into* the loop (which already polls for it) instead of
//! by abandoning it.

use super::agent_loop::{self, LoopResult, StepSource, ToolDispatch};
use super::types::{Session, Ticks};

/// How much stack an agent's task gets.
///
/// Deliberately the same as the scheduler's default for now. A deep chain (the loop, a
/// model forward, the ONNX interpreter's wide dispatch frames, a tool, possibly a
/// nested sub-agent) argues for more, and 1 MiB was tried — but that is the first
/// thing in this kernel to ask the first-fit allocator for a megabyte repeatedly, and
/// it takes the P5 heap-growth path that nothing else exercises this hard. Raising it
/// is a change that needs its own validation rather than a ride along with this one.
const AGENT_STACK: usize = 256 * 1024;

/// The borrows an agent task runs against, plus somewhere to leave its answer.
///
/// `'a` is the caller's frame. Never sent anywhere, never stored beyond the join —
/// see the module doc on why the loan holds.
struct Job<'a> {
    session: &'a mut Session,
    steps: &'a mut dyn StepSource,
    tools: &'a mut dyn ToolDispatch,
    now: &'a dyn Fn() -> Ticks,
    cancel: &'a mut dyn FnMut() -> bool,
    /// Written by the task, read by the caller after the join.
    out: Option<LoopResult>,
}

/// Entry point for an agent's task: run the loop the caller loaned us, leave the
/// result in the job, and exit.
extern "C" fn agent_task_entry(arg: u64) {
    // SAFETY: `arg` is the `&mut Job` the caller parked for us. The caller blocks
    // from before this task can be scheduled until after it is dead, so this is the
    // only live reference to it for as long as this function runs.
    let job = unsafe { &mut *(arg as *mut Job) };
    let result = agent_loop::run_with_cancel(
        job.session,
        job.steps,
        job.tools,
        // **Our own id, not the caller's.** This is the whole point: the task running
        // the work is the principal the capability gate checks.
        crate::sched::current_task_id(),
        || (job.now)(),
        || (job.cancel)(),
    );
    job.out = Some(result);
}

/// Run an agent loop on a fresh task and return its result.
///
/// `grant` is called once with the new task id before it can run, so a caller can
/// give the agent exactly the capabilities it should have — the task exists first,
/// which is what lets authority be attached to the thing that will actually spend it.
///
/// The task is reaped before returning, so its capability table is revoked by
/// `TaskControlBlock::reclaim` and the agent leaves no standing authority behind.
pub fn run_on_own_task(
    name: &'static str,
    session: &mut Session,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    now: &dyn Fn() -> Ticks,
    cancel: &mut dyn FnMut() -> bool,
    grant: &mut dyn FnMut(crate::sched::TaskId),
) -> LoopResult {
    let mut job = Job { session, steps, tools, now, cancel, out: None };
    let arg = &mut job as *mut Job as u64;

    // **Spawn and grant must be atomic against the scheduler.** A task becomes
    // runnable the moment `spawn` enqueues it, and the timer can preempt between two
    // ordinary statements — so granting afterwards leaves a window where the agent's
    // first tool call races its own authority and is refused for reasons that depend
    // on timing. Scheduling only happens on a yield or a timer tick, so masking
    // interrupts closes the window exactly, without needing a new task state.
    let task = crate::arch::interrupts::without_interrupts(|| {
        let task = crate::sched::spawn_with_stack(name, agent_task_entry, arg, AGENT_STACK);
        grant(task);
        task
    });

    // **No early return between here and the join.** See the module doc: abandoning
    // the task would leave it writing into a frame that no longer exists.
    crate::sched::join(task);
    let _ = crate::sched::kill(task);

    job.out.take().unwrap_or_else(|| {
        // The task did not record a result: it was killed, or it faulted. Reported as
        // a stop rather than a panic, because an agent dying is an operational event
        // — the OS survives it — and the caller needs something to show a user.
        crate::ktrace::log_fmt(format_args!("agent.task: '{name}' (task {task}) produced no result"));
        LoopResult {
            answer: alloc::string::String::from("stopped: the agent's task ended without answering"),
            stop: agent_loop::StopReason::Cancelled,
            turns: 0,
            tool_calls: 0,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::*;
    use alloc::string::{String, ToString};

    /// A step source that answers immediately, recording which task ran it.
    struct RecordingSteps {
        ran_as: u64,
    }
    impl StepSource for RecordingSteps {
        fn next(&mut self, _session: &Session) -> agent_loop::Step {
            self.ran_as = crate::sched::current_task_id();
            agent_loop::Step::Final("done".to_string())
        }
    }

    struct NoTools;
    impl ToolDispatch for NoTools {
        fn call(&mut self, _session: &mut Session, _caller: crate::sched::TaskId, _call: &ToolCall) -> agent_loop::ToolOutcome {
            agent_loop::ToolOutcome {
                result: String::from("no tools"),
                provenance: Provenance::SystemTrusted,
                is_error: true,
                origin: None,
            }
        }
    }

    fn a_session() -> Session {
        let role = crate::agent::manifest::orchestrator_manifest();
        Session::new(&role, 7, alloc::vec::Vec::new(), 0)
    }

    #[test_case]
    fn an_agent_loop_runs_on_its_own_task_not_the_callers() {
        // The claim that makes an agent a process: the loop does not execute on the
        // caller's stack, and the identity the gates would check is the task that ran
        // the work rather than whoever asked for it.
        let caller = crate::sched::current_task_id();
        let mut session = a_session();
        let mut steps = RecordingSteps { ran_as: u64::MAX };
        let mut tools = NoTools;
        let mut granted_to = 0u64;

        let result = run_on_own_task(
            "test-agent",
            &mut session,
            &mut steps,
            &mut tools,
            &|| 0,
            &mut || false,
            &mut |t| granted_to = t,
        );

        assert_eq!(result.answer, "done", "the loop must have run to a final answer");
        assert_ne!(steps.ran_as, caller, "the loop must not run on the caller's task");
        assert_ne!(steps.ran_as, u64::MAX, "the step source was never reached");
        assert_eq!(granted_to, steps.ran_as, "authority must be granted to the task that runs the work");
        // Reaped, so it holds nothing: the P8 property, now structural rather than
        // dependent on a caller remembering to retire an identity.
        assert!(!crate::sched::is_alive(steps.ran_as), "the agent task must be reaped");
    }

    #[test_case]
    fn the_agent_task_writes_its_result_back_through_the_loan() {
        // The loan is the only channel out of the task, so a result arriving at all is
        // what proves the caller's frame was still valid when the task wrote to it.
        let mut session = a_session();
        let mut steps = RecordingSteps { ran_as: 0 };
        let mut tools = NoTools;
        let result = run_on_own_task(
            "test-agent-loan",
            &mut session,
            &mut steps,
            &mut tools,
            &|| 0,
            &mut || false,
            &mut |_| {},
        );
        assert_eq!(result.stop, agent_loop::StopReason::Final, "a final answer is a clean stop, got {:?}", result.stop);
        // And the session it was loaned really was mutated, not a copy.
        assert!(
            session.messages.iter().any(|m| m.content.contains("done")),
            "the agent's answer must be in the caller's own session"
        );
    }
}
