//! `cargo xtask ring-check` — enforce the ring-3 standing rule mechanically.
//!
//! CLAUDE.md says only the kernel and drivers run in ring 0, and that an agent's effects
//! go through `synapse::tenant::invoke_in_userspace` rather than
//! `synapse::executor::execute`. That rule is easy to state and invisible to break: a
//! direct executor call from agent code still *works*, it just quietly keeps kernel
//! privilege, so nothing fails and no test turns red.
//!
//! Four such bypasses existed at once (`agent/wasm_rt.rs`, `service/server.rs`,
//! `service/package_ui.rs`, `persona/agent.rs`), and every one was found by grepping for
//! the call rather than by reading the code around it. So this replaces the grep: a new
//! call site outside the allowlist fails the check and has to justify itself.
//!
//! The scanner is deliberately textual. A type-level version (sealing the executor behind
//! a private module) would be stronger, but the executor is legitimately public for the
//! ABI, the bench and the gate-chain tests, and a marker trait would be one more thing to
//! forget. A list that fails loudly is worth more than a mechanism that is elegant.

use std::path::Path;

/// Files permitted to call the executor directly, with why. Anything else is a bypass.
///
/// Paths are repo-relative and matched by suffix, so the list reads the way a human would
/// write it.
pub const ALLOWED: &[(&str, &str)] = &[
    ("kernel/src/synapse/executor.rs", "defines it"),
    ("kernel/src/synapse/mod.rs", "re-exports it"),
    ("kernel/src/synapse/abi.rs", "the kernel side of the tenant trap -- this IS the ring-3 path"),
    ("kernel/src/synapse/bench.rs", "measures the gate chain; kernel-internal by definition"),
    ("kernel/src/synapse/policy.rs", "approval policy consulted by the executor"),
    ("kernel/src/tools/dispatch.rs", "the in-kernel arm, taken only for the orchestrator"),
    ("kernel/src/lib.rs", "#[test_case] tests of the gate chain itself"),
    ("kernel/src/cap/mod.rs", "capability tests"),
    ("kernel/src/security/redteam.rs", "the attack census runs through the real router"),
];

/// A direct executor call found in the tree.
#[derive(Debug, PartialEq, Eq)]
pub struct Hit {
    pub file: String,
    pub line: usize,
    pub text: String,
}

/// Whether `line` is a *call* to the executor rather than prose about one.
///
/// Comments are the whole difficulty here: CLAUDE.md's rule is explained in doc comments
/// that name `synapse::execute` repeatedly, and counting those would make the check cry
/// wolf until somebody deleted it. So a line whose first non-space characters start a
/// comment is prose, and only the rest can be a call.
pub fn is_executor_call(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("//") || t.starts_with("*") || t.starts_with("#!") {
        return false;
    }
    // `use` lines import the name without exercising the authority.
    if t.starts_with("use ") || t.starts_with("pub use ") {
        return false;
    }
    for pat in ["execute_with_justification(", "execute_current(", "execute("] {
        if let Some(i) = t.find(pat) {
            // Require the call to be qualified as the executor's, so an unrelated
            // `execute(` on some other type does not trip the check.
            let before = &t[..i];
            if before.ends_with("synapse::")
                || before.ends_with("executor::")
                || before.ends_with("super::executor::")
                || before.ends_with("crate::synapse::")
            {
                return true;
            }
        }
    }
    false
}

/// Whether `path` is on the allowlist.
pub fn is_allowed(path: &str) -> bool {
    let norm = path.replace('\\', "/");
    ALLOWED.iter().any(|(allowed, _)| norm.ends_with(allowed))
}

/// Scan `dir` recursively for direct executor calls outside the allowlist.
pub fn scan(dir: &Path) -> std::io::Result<Vec<Hit>> {
    let mut hits = Vec::new();
    walk(dir, &mut hits)?;
    hits.sort_by(|a, b| (a.file.clone(), a.line).cmp(&(b.file.clone(), b.line)));
    Ok(hits)
}

fn walk(dir: &Path, hits: &mut Vec<Hit>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk(&path, hits)?;
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let display = path.to_string_lossy().replace('\\', "/");
        if is_allowed(&display) {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        for (i, line) in text.lines().enumerate() {
            if is_executor_call(line) {
                hits.push(Hit { file: display.clone(), line: i + 1, text: line.trim().to_string() });
            }
        }
    }
    Ok(())
}

/// Run the check, returning the offending call sites.
pub fn check(repo: &Path) -> std::io::Result<Vec<Hit>> {
    scan(&repo.join("kernel/src"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_about_the_executor_is_not_a_call() {
        // The reason this check needs care at all: the rule is documented in comments
        // that name the function, and a scanner that counted those would be deleted
        // within a week for crying wolf.
        assert!(!is_executor_call("/// used to call `synapse::execute` directly"));
        assert!(!is_executor_call("// match crate::synapse::execute(task, &raw) {"));
        assert!(!is_executor_call("//! [`crate::synapse::executor::execute`] directly"));
        assert!(!is_executor_call(" * synapse::execute("));
        assert!(!is_executor_call("use synapse::executor::execute;"));
        assert!(!is_executor_call("pub use executor::{execute, execute_current};"));
    }

    #[test]
    fn a_real_call_is_a_call() {
        assert!(is_executor_call("    let inv = synapse::execute(task, &raw);"));
        assert!(is_executor_call("    match crate::synapse::execute_with_justification(t, r, j) {"));
        assert!(is_executor_call("        let inv = synapse::execute_current(BAD);"));
        assert!(is_executor_call("    let x = super::executor::execute_with_justification(c, &r, j);"));
    }

    #[test]
    fn an_unrelated_execute_is_not_the_executor() {
        // Qualification is required, so a same-named method elsewhere does not trip it.
        assert!(!is_executor_call("    self.execute(plan);"));
        assert!(!is_executor_call("    runtime.execute(args)?;"));
    }

    #[test]
    fn the_allowlist_matches_by_suffix_and_nothing_else() {
        assert!(is_allowed("kernel/src/synapse/abi.rs"));
        assert!(is_allowed("/abs/path/kernel/src/tools/dispatch.rs"));
        assert!(!is_allowed("kernel/src/agent/wasm_rt.rs"));
        // Not fooled by a similar name: the bypasses that existed were in files whose
        // names look adjacent to allowed ones.
        assert!(!is_allowed("kernel/src/service/server.rs"));
        assert!(!is_allowed("kernel/src/persona/agent.rs"));
    }

    #[test]
    fn the_scan_actually_finds_a_planted_bypass() {
        // Guards against the worst failure mode for a check like this: passing because it
        // looks at nothing. A green `ring-check` has to mean "searched and found none",
        // not "the walk quietly matched no files".
        let dir = std::env::temp_dir().join("chitti-ring-check-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("agent")).expect("temp dir");
        std::fs::write(
            dir.join("agent/rogue.rs"),
            "fn f(t: u64) {\n    let _ = crate::synapse::execute(t, \"x\");\n}\n",
        )
        .expect("write");
        // And a file that only *mentions* it, which must not be reported.
        std::fs::write(dir.join("agent/innocent.rs"), "/// see crate::synapse::execute(t, r)\nfn g() {}\n")
            .expect("write");

        let hits = scan(&dir).expect("scan");
        assert_eq!(hits.len(), 1, "expected exactly the planted call, got {hits:?}");
        assert!(hits[0].file.ends_with("agent/rogue.rs"));
        assert_eq!(hits[0].line, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_allowlist_entry_carries_a_reason() {
        // A bare path would become "somebody added this once"; the reason is what a
        // future reader needs in order to challenge it.
        for (path, why) in ALLOWED {
            assert!(!why.trim().is_empty(), "{path} is allowed with no stated reason");
            assert!(path.starts_with("kernel/src/"), "{path} is not a kernel source path");
        }
    }
}
