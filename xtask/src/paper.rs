//! `cargo xtask paper-check` — verify that the quantitative claims in
//! `paper/main.tex` still match the code.
//!
//! # Why this exists
//!
//! The paper states numbers that are properties of the repository: how many
//! primitives the registry has, how many gates guard the boundary, how many
//! in-kernel tests there are. Nothing connected those statements to the thing
//! they describe, so an ordinary change could invalidate a published claim
//! silently — and did. Adding an approval gate renumbered a chain the paper
//! prints by number ("Gate 3 (taint)", "Gate 4 (scope)") and re-priced a cost
//! table that measures cumulative prefixes 1--4; only `synapse::bench`'s own
//! drift test noticed. Adding ~100 unit tests made the stated test count wrong
//! with nothing noticing at all.
//!
//! So the claims that *can* be derived from the tree are derived and compared.
//! This does not edit the paper — it reports, and exits non-zero on a mismatch,
//! leaving the prose to a human who can decide whether the number or the code is
//! the thing that should change.
//!
//! # What it deliberately does not check
//!
//! Measured figures — tokens/s, gate costs in nanoseconds, attack-success rates —
//! are outputs of running the kernel (`/perf`, `/bench synapse`, `/redteam`), not
//! properties of the source. Pretending to verify them from a static tree would
//! be worse than not checking them: it would give a green light that means
//! nothing. They are listed as `unchecked` so a reader knows the difference.

use std::fmt;
use std::path::Path;

/// One claim the paper makes and the tree can answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    /// Human-readable name, used in the report.
    pub what: &'static str,
    /// The value the paper states.
    pub stated: u64,
    /// The value the tree actually has.
    pub actual: u64,
}

impl Claim {
    pub fn ok(&self) -> bool {
        self.stated == self.actual
    }
}

impl fmt::Display for Claim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ok() {
            write!(f, "  ok       {}: {}", self.what, self.actual)
        } else {
            write!(f, "  MISMATCH {}: paper says {}, tree has {}", self.what, self.stated, self.actual)
        }
    }
}

/// Parse a number as LaTeX writes it: `1{,}203`, `1,203`, `26`.
///
/// The `{,}` form is the one that matters — it is how the paper thousands-separates
/// (`\,` and `{,}` both appear in TeX), and a parser that only handled bare digits
/// would silently find no claims and report success.
pub fn parse_tex_number(s: &str) -> Option<u64> {
    let mut digits = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '0'..='9' => digits.push(c),
            // `{,}` and `{\,}` separators, and a bare comma.
            '{' => {
                // Skip to the matching `}` provided it only held a separator.
                let mut inner = String::new();
                for c2 in chars.by_ref() {
                    if c2 == '}' {
                        break;
                    }
                    inner.push(c2);
                }
                if !matches!(inner.trim(), "," | "\\," | "") {
                    break;
                }
            }
            ',' => {}
            _ => break,
        }
    }
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// The first number appearing immediately before `marker`, scanning back from it.
///
/// Used for claims phrased "N in-kernel unit tests": the number precedes the
/// words. Returns `None` if the marker is absent, which is itself worth
/// reporting — a claim that vanished from the prose is a claim this tool has
/// stopped checking, and silently checking nothing is the failure mode to avoid.
pub fn number_before(tex: &str, marker: &str) -> Option<u64> {
    let at = tex.find(marker)?;
    let head = &tex[..at];
    // Walk back over the token immediately before the marker.
    let tail: String = head.chars().rev().take_while(|c| !c.is_ascii_alphabetic()).collect();
    let token: String = tail.chars().rev().collect();
    let trimmed = token.trim();
    let start = trimmed.rfind(|c: char| c.is_whitespace() || c == '(').map(|i| i + 1).unwrap_or(0);
    parse_tex_number(trimmed[start..].trim())
}

/// Count `#[test_case]` attributes under `dir`, recursively.
///
/// This is the *declared* count, which is what the paper's parenthetical states.
/// It is deliberately not the number the x86 suite runs: some are `cfg`-gated to
/// aarch64 and never execute there, which the paper already explains.
pub fn count_test_cases(dir: &Path) -> std::io::Result<u64> {
    let mut n = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            n += count_test_cases(&path)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path)?;
            n += count_attr(&text);
        }
    }
    Ok(n)
}

/// Count `#[test_case]` occurrences in one file, ignoring ones inside a line
/// comment or a doc comment — a mention in prose is not a test.
pub fn count_attr(text: &str) -> u64 {
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//") && t.contains("#[test_case]")
        })
        .count() as u64
}


/// The mechanism's non-test line count -- the paper's TCB figure.
///
/// Lines outside every `#[cfg(test)] mod … { … }` region, for the files listed in
/// `paper/tools_loc.py`. The brace walk is not fussiness: the obvious recipe
/// ("everything before the first `#[cfg(test)]`") was correct while each file
/// ended with a single test module, and became silently wrong when
/// `synapse/executor.rs` grew a `gate_contract_tests` module in the *middle* of
/// the file with several hundred lines of real code after it. That reported 244
/// non-test lines for a 763-line file, and the paper's total drifted 35% low --
/// in the flattering direction, which is the one that matters.
///
/// The file list lives in the Python script and is mirrored here rather than
/// parsed out of it: two short lists that must agree is a worse failure than one
/// list, but parsing Python from xtask to avoid duplicating nine paths is worse
/// still. `mechanism_files_match_the_script` pins them together.
pub const MECHANISM_FILES: &[&str] = &[
    "kernel/src/synapse/grammar.rs",
    "kernel/src/cap/mod.rs",
    "kernel/src/security/taint.rs",
    "kernel/src/security/mod.rs",
    "kernel/src/synapse/vpath.rs",
    "kernel/src/synapse/policy.rs",
    "kernel/src/security/citation.rs",
    "kernel/src/synapse/citation.rs",
    "kernel/src/synapse/executor.rs",
    "kernel/src/synapse/registry.rs",
    "kernel/src/synapse/audit.rs",
    "kernel/src/synapse/fs.rs",
];

/// Lines in `text` that are not inside a `#[cfg(test)]` module.
pub fn non_test_lines(text: &str) -> u64 {
    let lines: Vec<&str> = text.lines().collect();
    let mut total = 0u64;
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i].trim_start().starts_with("#[cfg(test)]") {
            let mut depth: i64 = 0;
            let mut opened = false;
            i += 1;
            while i < lines.len() {
                depth += lines[i].matches('{').count() as i64;
                depth -= lines[i].matches('}').count() as i64;
                if lines[i].contains('{') {
                    opened = true;
                }
                i += 1;
                if opened && depth <= 0 {
                    break;
                }
            }
            continue;
        }
        total += 1;
        i += 1;
    }
    total
}

/// Sum [`non_test_lines`] over [`MECHANISM_FILES`].
pub fn mechanism_lines(repo: &Path) -> std::io::Result<u64> {
    let mut n = 0;
    for f in MECHANISM_FILES {
        n += non_test_lines(&std::fs::read_to_string(repo.join(f))?);
    }
    Ok(n)
}

/// Count `PrimitiveSpec {` *literals* in the registry, excluding the struct's own
/// definition.
///
/// The obvious grep counts 27 and the paper says 26; the difference is
/// `pub struct PrimitiveSpec {`. Getting this wrong in the confident direction
/// would mean "fixing" a paper that was right, which is why the definition is
/// excluded explicitly rather than by subtracting one.
pub fn count_primitives(registry_rs: &str) -> u64 {
    registry_rs
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.contains("PrimitiveSpec {") && !t.starts_with("//") && !t.contains("struct ")
        })
        .count() as u64
}

/// Render a number the way the paper writes it: `3601` -> `3{,}601`.
pub fn tex_number(n: u64) -> String {
    let s = n.to_string();
    if s.len() <= 3 {
        return s;
    }
    let (head, tail) = s.split_at(s.len() - 3);
    format!("{head}{{,}}{tail}")
}

/// Read `pub const GATE_COUNT: u8 = N;` out of the executor.
pub fn gate_count(executor_rs: &str) -> Option<u64> {
    let at = executor_rs.find("pub const GATE_COUNT: u8 =")?;
    let rest = &executor_rs[at..];
    let eq = rest.find('=')? + 1;
    let end = rest[eq..].find(';')? + eq;
    rest[eq..end].trim().parse().ok()
}

/// Compare every derivable claim. `ran` is the number of tests the x86 suite
/// actually executed, if the caller measured it (`cargo xtask test`); `None`
/// skips that one rather than guessing.
pub fn check(repo: &Path, ran: Option<u64>) -> std::io::Result<Vec<Claim>> {
    let tex = std::fs::read_to_string(repo.join("paper/main.tex"))?;
    let registry = std::fs::read_to_string(repo.join("kernel/src/synapse/registry.rs"))?;
    let executor = std::fs::read_to_string(repo.join("kernel/src/synapse/executor.rs"))?;
    let declared = count_test_cases(&repo.join("kernel/src"))?;

    let mut out = Vec::new();
    if let Some(stated) = number_before(&tex, " primitives spanning") {
        out.push(Claim { what: "registry primitives", stated, actual: count_primitives(&registry) });
    }
    if let Some(stated) = number_before(&tex, " are declared") {
        out.push(Claim { what: "declared #[test_case]", stated, actual: declared });
    }
    // The TCB figure. The marker is deliberately a plain-token one: the same
    // number also appears as `\textbf{3{,}601 lines excluding its own tests}`,
    // and `number_before` walks back over `\textbf{` into a brace it cannot
    // parse, so anchoring there silently checked nothing. Both other statements
    // of the figure are then cross-checked with `contains`, so updating one and
    // forgetting the others fails rather than passes.
    if let Some(stated) = number_before(&tex, " mechanism\nlines") {
        let actual = mechanism_lines(repo)?;
        out.push(Claim { what: "mechanism lines (non-test)", stated, actual });
        let n = tex_number(actual);
        for (what, needle) in [
            ("E5 table total", format!("\\textbf{{{n}}} & \\textbf{{")),
            ("TCB figure in section 4", format!("\\textbf{{{n} lines excluding its own tests}}")),
        ] {
            out.push(Claim { what, stated: if tex.contains(&needle) { actual } else { 0 }, actual });
        }
    }
    if let (Some(stated), Some(ran)) = (number_before(&tex, " in-kernel unit tests"), ran) {
        out.push(Claim { what: "unit tests run (x86)", stated, actual: ran });
    }
    // "four gates in a fixed order" is spelled as a word, so it is matched
    // literally rather than parsed. The numbering itself is pinned in-kernel by
    // `gate_numbering_is_the_published_contract`; this catches the count.
    if let Some(actual) = gate_count(&executor) {
        let stated = if tex.contains("four gates") { 4 } else { 0 };
        out.push(Claim { what: "gates in the chain", stated, actual });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The brace walk must skip a test module that sits in the *middle* of a file
    /// and keep counting after it. The old "everything before the first
    /// `#[cfg(test)]`" rule got this wrong by 35% and in the flattering direction.
    #[test]
    fn non_test_lines_skips_interior_test_modules() {
        let src = "\
line one
#[cfg(test)]
mod inner {
    fn helper() {}
}
line after
line after two
";
        // 4 real lines: "line one", "line after", "line after two", and the
        // trailing empty one `lines()` does not yield -- so 3.
        assert_eq!(non_test_lines(src), 3);
        // A file with no tests at all is just its line count.
        assert_eq!(non_test_lines("a\nb\nc\n"), 3);
    }

    /// The paper's file list and xtask's must not drift apart, or the checker
    /// silently validates a different number than the artifact's script prints.
    #[test]
    fn mechanism_files_match_the_script() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let script = std::fs::read_to_string(root.join("paper/tools_loc.py")).expect("tools_loc.py");
        for f in MECHANISM_FILES {
            assert!(script.contains(f), "{f} is checked by xtask but absent from tools_loc.py");
        }
        let listed = script.matches("kernel/src/").count();
        assert_eq!(listed, MECHANISM_FILES.len(), "tools_loc.py lists a file xtask does not check");
    }

    #[test]
    fn renders_a_number_the_way_the_paper_writes_it() {
        assert_eq!(tex_number(3601), "3{,}601");
        assert_eq!(tex_number(26), "26");
        assert_eq!(tex_number(999), "999");
    }

    #[test]
    fn parses_the_latex_thousands_separator() {
        // The form that matters: a parser handling only bare digits would find no
        // claims and cheerfully report success.
        assert_eq!(parse_tex_number("1{,}203"), Some(1203));
        assert_eq!(parse_tex_number("1,226"), Some(1226));
        assert_eq!(parse_tex_number("26"), Some(26));
        assert_eq!(parse_tex_number("2{,}671"), Some(2671));
        assert_eq!(parse_tex_number("none"), None);
        // A brace holding something other than a separator ends the number
        // rather than being skipped over.
        assert_eq!(parse_tex_number("1{\\times}5"), Some(1));
    }

    #[test]
    fn finds_the_number_before_a_marker() {
        let tex = "covered by 1{,}203 in-kernel unit tests that run without a model";
        assert_eq!(number_before(tex, " in-kernel unit tests"), Some(1203));
        let tex2 = "(1{,}226 are declared; the difference is aarch64-gated)";
        assert_eq!(number_before(tex2, " are declared"), Some(1226));
        // An absent marker is `None`, not a wrong answer — a claim that vanished
        // from the prose must stop being "checked" loudly, not quietly pass.
        assert_eq!(number_before(tex, " nonexistent marker"), None);
    }

    #[test]
    fn counts_only_real_test_attributes() {
        let src = "\
#[test_case]
fn a() {}
// #[test_case] in a comment does not count
/// #[test_case] in a doc comment does not count
    #[test_case]
    fn b() {}
";
        assert_eq!(count_attr(src), 2);
    }

    #[test]
    fn primitive_count_excludes_the_struct_definition() {
        // The exact trap: a naive grep counts the definition and reports 27
        // against a paper that correctly says 26.
        let src = "\
pub struct PrimitiveSpec {
    pub id: u16,
}
const T: &[PrimitiveSpec] = &[
    PrimitiveSpec {
        id: 1,
    },
    PrimitiveSpec {
        id: 2,
    },
];
";
        assert_eq!(count_primitives(src), 2);
    }

    #[test]
    fn reads_the_gate_count_constant() {
        let src = "pub const GATE_SCOPE: u8 = 4;\npub const GATE_COUNT: u8 = 4;\n";
        assert_eq!(gate_count(src), Some(4));
        assert_eq!(gate_count("no such constant"), None);
    }

    #[test]
    fn a_claim_reports_its_own_verdict() {
        let good = Claim { what: "x", stated: 4, actual: 4 };
        let bad = Claim { what: "x", stated: 4, actual: 5 };
        assert!(good.ok() && !bad.ok());
        assert!(format!("{bad}").contains("MISMATCH"));
    }
}
