//! The plan vocabulary (`CHITTI_OS_HANDOFF.md` Phase 5): a single [`Action`]
//! a planner emits and an agent executes. Two kinds cross the determinism
//! boundary differently:
//!
//! * [`Action::Call`] is a Synapse tool call as canonical MCP JSON. It goes
//!   through the full Phase 4 pipeline (grammar -> capability -> execute ->
//!   audit); the agent never touches an effect except through it.
//! * [`Action::Remember`] / [`Action::Recall`] are tier-2 memory operations
//!   on the agent's persistent store (`persona::memory`).
//!
//! The `call_*` constructors emit exactly the grammar's canonical shape, so a
//! plan is always accepted by `synapse::grammar::parse`.

use alloc::string::String;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// A Synapse tool call, as canonical JSON, to run through the ABI.
    Call(String),
    /// Store `key = value` in the agent's persistent memory (tier 2).
    Remember(String, String),
    /// Demand-page `key` from the persistent store into live context.
    Recall(String),
}

impl Action {
    pub fn call_write(path: &str, text: &str) -> Action {
        Action::Call(alloc::format!(
            r#"{{"name":"mem_fs_write","arguments":{{"path":"{}","text":"{}"}}}}"#,
            escape(path),
            escape(text)
        ))
    }

    pub fn call_read(path: &str) -> Action {
        Action::Call(alloc::format!(r#"{{"name":"mem_fs_read","arguments":{{"path":"{}"}}}}"#, escape(path)))
    }

    pub fn call_list() -> Action {
        Action::Call(String::from(r#"{"name":"list","arguments":{}}"#))
    }

    pub fn call_console(text: &str) -> Action {
        Action::Call(alloc::format!(r#"{{"name":"console_write","arguments":{{"text":"{}"}}}}"#, escape(text)))
    }

    pub fn call_emit(text: &str) -> Action {
        Action::Call(alloc::format!(r#"{{"name":"emit_result","arguments":{{"text":"{}"}}}}"#, escape(text)))
    }
}

/// Escape a string for embedding in a canonical tool-call value, matching the
/// escapes `synapse::grammar` accepts (`\" \\ \n \t \r`).
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::grammar;

    #[test_case]
    fn constructed_calls_are_grammar_valid() {
        // Every constructor must emit something the Phase 4 grammar accepts.
        let cases = [
            Action::call_write("notes", "hello world"),
            Action::call_read("notes"),
            Action::call_list(),
            Action::call_console("hi"),
            Action::call_emit("done"),
        ];
        for c in cases {
            match c {
                Action::Call(json) => {
                    assert!(grammar::parse(&json).is_ok(), "planner emitted a grammar-invalid call: {json}");
                }
                _ => unreachable!(),
            }
        }
    }

    #[test_case]
    fn escaping_survives_the_grammar_roundtrip() {
        // A payload with characters that must be escaped still parses back to
        // the original bytes.
        let payload = "a\"b\\c";
        let Action::Call(json) = Action::call_write("p", payload) else { unreachable!() };
        let call = grammar::parse(&json).expect("escaped call must parse");
        assert_eq!(call.args[1], grammar::ArgValue::Str(String::from(payload)));
    }
}
