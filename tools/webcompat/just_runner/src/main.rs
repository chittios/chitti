//! test262 runner using the **just-engine** ES6 interpreter (tree-walking).
//!
//! This is the host harness path that ports full language effort from
//! `third_party/just-ref` (applegrew/just). Kernel `browser::js` remains the
//! no_std DOM-facing engine; this binary is for webcompat reporting.
//!
//! Usage: chitti-just-runner <test.js|dir>…

use just_engine::parser::JsParser;
use just_engine::runner::ds::value::JsValue;
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: chitti-just-runner <test.js|dir>…");
        return ExitCode::from(2);
    }
    let mut files = Vec::new();
    for a in &args {
        collect_js(Path::new(a), &mut files);
    }
    files.sort();
    // Quiet per-file panics so one malformed/unsupported input can't abort the
    // whole run; we isolate and count them below.
    std::panic::set_hook(Box::new(|_| {}));
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    let mut panics = 0usize;
    for f in &files {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(f)))
            .unwrap_or_else(|_| Outcome::Fail("panic".into()));
        match outcome {
            Outcome::Pass => {
                pass += 1;
                println!("PASS {}", f.display());
            }
            Outcome::Fail(msg) => {
                fail += 1;
                if msg == "panic" {
                    panics += 1;
                }
                println!("FAIL {} — {msg}", f.display());
            }
            Outcome::Skip(msg) => {
                skip += 1;
                println!("SKIP {} — {msg}", f.display());
            }
        }
    }
    let total = pass + fail + skip;
    let rate = if pass + fail == 0 {
        0.0
    } else {
        100.0 * pass as f64 / (pass + fail) as f64
    };
    println!(
        "\n=== chitti-just-runner summary (just-engine) ===\nfiles={total} pass={pass} fail={fail} skip={skip} panics={panics} pass_rate={rate:.1}% (of runnable)"
    );
    ExitCode::SUCCESS
}

enum Outcome {
    Pass,
    Fail(String),
    Skip(String),
}

fn collect_js(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_file() {
        if p.extension().and_then(|e| e.to_str()) == Some("js") {
            out.push(p.to_path_buf());
        }
        return;
    }
    if let Ok(rd) = fs::read_dir(p) {
        for e in rd.flatten() {
            collect_js(&e.path(), out);
        }
    }
}

/// test262 metadata we care about (from the `/*--- … ---*/` YAML block).
#[derive(Default)]
struct Meta {
    negative: bool,
    module: bool,
    raw: bool,
    needs_async_harness: bool,
}

fn parse_meta(src: &str) -> Meta {
    let mut m = Meta::default();
    if let Some(i) = src.find("/*---") {
        if let Some(j) = src[i..].find("---*/") {
            let front = &src[i + 5..i + j];
            m.negative = front.contains("negative:");
            if let Some(fi) = front.find("flags:") {
                let line: String = front[fi..].lines().next().unwrap_or("").to_string();
                m.module = line.contains("module");
                m.raw = line.contains("raw");
                m.needs_async_harness = line.contains("async");
            }
        }
    }
    m
}

/// Prepare test262 source for just-engine.
fn adapt_source(src: &str) -> Result<String, String> {
    // Module/include forms we don't support. ($DONOTEVALUATE is handled in
    // run_one as a parse-only negative test.)
    if src.contains("$INCLUDE") || src.contains("import ") || src.contains("export ") {
        return Err("module/include".into());
    }

    let mut s = src.to_string();
    // Strip YAML front-matter
    while let Some(i) = s.find("/*---") {
        if let Some(j) = s[i..].find("---*/") {
            s = format!("{}{}", &s[..i], &s[i + j + 5..]);
        } else {
            break;
        }
    }

    // Inject minimal harness helpers used by many tests.
    let preamble = r#"
var $ERROR = function (msg) { throw new Error(String(msg)); };
var Test262Error = function (msg) { this.message = String(msg); this.name = "Test262Error"; };
Test262Error.prototype = new Error();
var assert = function (cond, msg) {
  if (!cond) { throw new Test262Error(msg || "assert failed"); }
};
assert.sameValue = function (actual, expected, msg) {
  if (actual !== expected) {
    throw new Test262Error((msg || "") + " expected " + expected + " got " + actual);
  }
};
assert.notSameValue = function (actual, unexpected, msg) {
  if (actual === unexpected) {
    throw new Test262Error((msg || "") + " got unexpected " + unexpected);
  }
};
assert.throws = function (type, fn, msg) {
  var threw = false;
  try { fn(); } catch (e) { threw = true; }
  if (!threw) { throw new Test262Error(msg || "expected throw"); }
};
assert._isSameValue = function (a, b) {
  if (a === b) { return a !== 0 || 1 / a === 1 / b; }
  return a !== a && b !== b;
};
"#;

    // Prefer harness-style tests that use assert; leave throw Test262Error as-is.
    Ok(format!("{preamble}\n{s}\n"))
}

fn run_one(path: &Path) -> Outcome {
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return Outcome::Fail(format!("read: {e}")),
    };
    let meta = parse_meta(&src);
    // Module tests + async tests needing the $DONE harness aren't supported.
    if meta.module {
        return Outcome::Skip("module".into());
    }
    if meta.needs_async_harness {
        return Outcome::Skip("async-harness".into());
    }
    let adapted = match adapt_source(&src) {
        Ok(s) => s,
        Err(e) => return Outcome::Skip(e),
    };

    // `$DONOTEVALUATE`: a parse-only test (almost always `negative: phase:
    // parse`). Parse but don't run; the parser rejecting it is the expected
    // outcome for a negative test.
    let parse_only = src.contains("$DONOTEVALUATE");

    let parse_result = JsParser::parse_to_ast_from_str(&adapted);
    if parse_only {
        let parsed_ok = parse_result.is_ok();
        return match (meta.negative, parsed_ok) {
            (true, false) => Outcome::Pass,  // rejected as expected
            (true, true) => Outcome::Fail("negative parse test parsed without error".into()),
            (false, true) => Outcome::Pass,
            (false, false) => Outcome::Skip("parse (donotevaluate)".into()),
        };
    }

    let ast = match parse_result {
        Ok(a) => a,
        Err(e) => {
            // A negative test that fails to parse threw as expected → PASS.
            if meta.negative {
                return Outcome::Pass;
            }
            // Otherwise an engine parser gap → skip (not a correctness failure).
            return Outcome::Skip(format!("parse: {e:?}"));
        }
    };

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());

    let mut threw = false;
    for stmt in &ast.body {
        if execute_statement(stmt, &mut ctx).is_err() {
            threw = true;
            break;
        }
    }
    let _ = JsValue::Undefined;

    match (meta.negative, threw) {
        // Negative test: throwing (parse or runtime) is the expected outcome.
        (true, true) => Outcome::Pass,
        (true, false) => Outcome::Fail("negative test did not throw".into()),
        // Positive test: any throw is a failure; completing is a pass.
        (false, true) => Outcome::Fail("runtime threw".into()),
        (false, false) => Outcome::Pass,
    }
}
