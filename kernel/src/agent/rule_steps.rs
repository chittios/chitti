//! A deterministic [`StepSource`](super::agent_loop::StepSource) for the test
//! suite and boot demo. Real inference (Cortex + grammar-constrained decoding)
//! is too slow for the fast unit suite and non-deterministic to assert on, so —
//! exactly as the existing `persona::RulePlanner` stands in for the model — a
//! rule/scripted step source drives the loop reproducibly (DECISIONS.md #4/#6).
//! The real Cortex-backed `StepSource` plugs into the same trait for `run`.

use crate::agent::agent_loop::{Step, StepSource};
use crate::agent::types::*;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

static CALL_CTR: AtomicU64 = AtomicU64::new(1);

/// Build a tool call with a fresh call id and JSON args.
pub fn tool(name: &str, args: String) -> ToolCall {
    ToolCall { call_id: CALL_CTR.fetch_add(1, Ordering::Relaxed), tool: name.to_string(), args }
}

/// A `{ "k":"v", ... }` object from string pairs (escapes `"`/`\`).
pub fn args(pairs: &[(&str, &str)]) -> String {
    let mut s = String::from("{");
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(k);
        s.push_str("\":\"");
        for c in v.chars() {
            match c {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                c => s.push(c),
            }
        }
        s.push('"');
    }
    s.push('}');
    s
}

/// A fixed script of [`Step`]s, popped in order. Once exhausted it emits a
/// terminal `Final` so the loop always stops.
pub struct ScriptedSteps {
    queue: VecDeque<Step>,
    exhausted_answer: String,
}

impl ScriptedSteps {
    pub fn new(steps: Vec<Step>) -> Self {
        Self { queue: steps.into_iter().collect(), exhausted_answer: "done".to_string() }
    }
}

impl StepSource for ScriptedSteps {
    fn next(&mut self, _session: &Session) -> Step {
        self.queue.pop_front().unwrap_or_else(|| Step::Final(self.exhausted_answer.clone()))
    }
}

/// Map a small set of demo intents to a deterministic tool-use script — the
/// StepSource analogue of `persona::RulePlanner`. Used by the boot demo so a
/// real orchestrator loop runs without waiting on slow inference. Unrecognized
/// intents just echo back.
pub fn for_intent(intent: &str) -> ScriptedSteps {
    let lower = intent.to_ascii_lowercase();
    // "write a file called X with the text Y, then read it back"
    if lower.contains("write a file called") {
        if let Some((name, text)) = parse_write_intent(intent) {
            let mut steps = Vec::new();
            steps.push(Step::Tools(alloc::vec![tool("write", args(&[("path", &name), ("content", &text)]))]));
            if lower.contains("read it back") || lower.contains("read back") {
                steps.push(Step::Tools(alloc::vec![tool("read", args(&[("path", &name)]))]));
            }
            steps.push(Step::Final(alloc::format!("wrote and verified '{name}'")));
            return ScriptedSteps::new(steps);
        }
    }
    if lower.starts_with("list") {
        return ScriptedSteps::new(alloc::vec![
            Step::Tools(alloc::vec![tool("list", "{}".to_string())]),
            Step::Final("listed the store".to_string()),
        ]);
    }
    ScriptedSteps::new(alloc::vec![Step::Final(alloc::format!("ack: {intent}"))])
}

/// Parse `... called <name> with the text <text>` → (name, text).
fn parse_write_intent(intent: &str) -> Option<(String, String)> {
    let after = intent.split("called ").nth(1)?;
    let name = after.split_whitespace().next()?.trim_matches(',').to_string();
    let text = intent.split("with the text ").nth(1)?.trim_end_matches(", then read it back").trim().to_string();
    Some((name, text))
}
