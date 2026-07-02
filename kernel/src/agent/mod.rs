//! **Agent** — the Claude-Code-style agent layer (`CHITTI_AGENTIC_HANDOFF.md`).
//! Replaces the flat `persona` model with an orchestrator that runs a tool-use
//! loop over a first-class [`tool`](crate::tools) layer, dispatches isolated
//! sub-agents, and persists to first-class [`session`](crate::session)s.
//!
//! * [`types`] — the shared contract (`CHITTI_SCHEMAS.md`): `AgentManifest`,
//!   `Session`, `SkillManifest`, and the Part-0 primitives.
//! * [`agent_loop`] — the agentic loop (model → tool → result → repeat) with
//!   budgets and stop conditions, over the `StepSource`/`ToolDispatch` seams.
//! * [`manifest`] — builtin roles + the bridge from declarative caps to live
//!   kernel `InvokePrimitive` grants.
//! * [`orchestrator`] — the session's foreground main agent + the Phase-A
//!   Synapse-backed tool dispatcher.
//! * [`rule_steps`] — a deterministic `StepSource` for tests and the boot demo.
//!
//! Everything above the determinism boundary lives here; every *effect* still
//! flows down through Synapse (locked invariant #1).

pub mod agent_loop;
pub mod manifest;
pub mod orchestrator;
pub mod rule_steps;
pub mod types;

pub use types::*;

/// Boot-time demonstration of the Phase A deliverable: a real orchestrator loop
/// completing a task with a tool, then a session save→resume that continues
/// coherently. Deterministic (rule StepSource), so it runs on every boot.
#[cfg(not(test))]
pub fn demo() {
    use crate::serial_println;
    serial_println!("Chitti: --- Agentic loop + sessions (Phase A) ---");

    let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 42);
    let mut tools = crate::tools::Router::new();

    let intent = "write a file called notes with the text hello world, then read it back";
    serial_println!("Chitti: intent> {}", intent);
    let mut steps = rule_steps::for_intent(intent);
    let r = orch.handle(intent, &mut steps, &mut tools);
    serial_println!(
        "Chitti: loop> {} (stop={:?}, turns={}, tool_calls={})",
        r.answer, r.stop, r.turns, r.tool_calls
    );

    // Save + resume: the session round-trips through the store and continues.
    let sid = orch.session.id;
    let _ = crate::session::save(&orch.session);
    if let Some(resumed) = crate::session::resume(sid) {
        serial_println!(
            "Chitti: resume> session {} reconstructed with {} messages",
            resumed.id.0,
            resumed.messages.len()
        );
        let mut orch2 = orchestrator::Orchestrator::from_session(manifest::orchestrator_manifest(), resumed);
        let follow = "list";
        let mut steps2 = rule_steps::for_intent(follow);
        let r2 = orch2.handle(follow, &mut steps2, &mut tools);
        serial_println!("Chitti: resume-cont> {} ({} messages now)", r2.answer, orch2.session.messages.len());
    }
}

#[cfg(test)]
mod tests {
    use super::agent_loop::{Step, StopReason};
    use super::rule_steps::{args, tool, ScriptedSteps};
    use super::types::*;
    use super::{manifest, orchestrator};
    use alloc::vec;

    /// (a) A multi-turn loop completes a task using ≥1 tool, and the effect is
    /// real (the file exists in the memory FS after the loop).
    #[test_case]
    fn loop_completes_task_using_a_tool() {
        let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 42);
        let mut tools = crate::tools::Router::new();

        let steps = vec![
            Step::Tools(vec![tool("write", args(&[("path", "phaseA_notes"), ("content", "hello world")]))]),
            Step::Tools(vec![tool("read", args(&[("path", "phaseA_notes")]))]),
            Step::Final("wrote and verified phaseA_notes".into()),
        ];
        let mut src = ScriptedSteps::new(steps);
        let r = orch.handle("write phaseA_notes then read it", &mut src, &mut tools);

        assert_eq!(r.stop, StopReason::Final);
        assert_eq!(r.tool_calls, 2, "two tool calls (write, read) executed");
        // The effect is real and observable from outside the loop.
        assert_eq!(crate::synapse::fs::read("phaseA_notes").as_deref(), Some(&b"hello world"[..]));
        // The read tool-result is in context and tagged untrusted-ingested.
        let read_result = orch.session.messages.iter().rev().find(|m| m.role == Role::Tool).unwrap();
        assert!(read_result.content.contains("hello world"));
        assert_eq!(read_result.provenance, Provenance::UntrustedIngested);
    }

    /// (b) `session save` then `session resume` reconstructs the history and the
    /// agent continues the conversation coherently.
    #[test_case]
    fn session_save_resume_and_continue() {
        let mut orch = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 7);
        let mut tools = crate::tools::Router::new();

        let steps = vec![Step::Tools(vec![tool("write", args(&[("path", "resume_f"), ("content", "v1")]))]), Step::Final("done".into())];
        let mut src = ScriptedSteps::new(steps);
        orch.handle("write resume_f", &mut src, &mut tools);
        let sid = orch.session.id;
        let msgs_before = orch.session.messages.len();
        crate::session::save(&orch.session).expect("save");

        // Fresh reconstruction from the store — no in-memory carryover.
        let resumed = crate::session::resume(sid).expect("resume");
        assert_eq!(resumed.id, sid);
        assert_eq!(resumed.messages.len(), msgs_before, "full history reconstructed");
        assert_eq!(resumed.seed, 7);

        // Continue the resumed session with another turn.
        let mut orch2 = orchestrator::Orchestrator::from_session(manifest::orchestrator_manifest(), resumed);
        let mut src2 = ScriptedSteps::new(vec![Step::Tools(vec![tool("list", "{}".into())]), Step::Final("listed".into())]);
        let r2 = orch2.handle("list", &mut src2, &mut tools);
        assert_eq!(r2.stop, StopReason::Final);
        assert!(orch2.session.messages.len() > msgs_before, "conversation continued after resume");
    }

    /// A turn budget stops a runaway loop instead of spinning forever.
    #[test_case]
    fn turn_budget_stops_the_loop() {
        let mut m = manifest::orchestrator_manifest();
        m.budgets.max_turns = 2;
        let mut orch = orchestrator::Orchestrator::spawn(m, 1);
        let mut tools = crate::tools::Router::new();
        // A source that always asks for another tool call, never finalizes.
        struct Never;
        impl super::agent_loop::StepSource for Never {
            fn next(&mut self, _s: &Session) -> Step {
                Step::Tools(vec![tool("list", "{}".into())])
            }
        }
        let mut never = Never;
        let r = orch.handle("loop forever", &mut never, &mut tools);
        assert_eq!(r.stop, StopReason::BudgetTurns);
        assert!(r.turns <= 2);
    }
}
