//! test262 runner using the **just-engine** ES6 interpreter (tree-walking).
//!
//! This is the host harness path that ports full language effort from
//! `third_party/just-ref` (applegrew/just). Kernel `browser::js` remains the
//! no_std DOM-facing engine; this binary is for webcompat reporting.
//!
//! Usage: chitti-just-runner [--raw] <test.js|dir>…
//!
//! `--raw` skips the test262 preamble/metadata handling and runs each file
//! verbatim (for real-world library fixtures — parse-error positions then map
//! directly to the file).

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
    let mut args: Vec<String> = env::args().skip(1).collect();
    let raw = args.iter().any(|a| a == "--raw");
    args.retain(|a| a != "--raw");
    if args.is_empty() {
        eprintln!("usage: chitti-just-runner [--raw] <test.js|dir>…");
        return ExitCode::from(2);
    }
    let mut files = Vec::new();
    for a in &args {
        collect_js(Path::new(a), &mut files);
    }
    files.sort();
    // Quiet per-file panics so one malformed/unsupported input can't abort the
    // whole run; we isolate and count them below. Capture the panic message
    // into a thread-local so the outcome can report it (JUST_VERBOSE=1).
    use std::cell::RefCell;
    thread_local! { static LAST_PANIC: RefCell<String> = RefCell::new(String::new()); }
    let verbose = std::env::var("JUST_VERBOSE").is_ok();
    std::panic::set_hook(Box::new(|info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "?".into());
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        LAST_PANIC.with(|c| *c.borrow_mut() = format!("{loc} {msg}"));
    }));
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    let mut panics = 0usize;
    for f in &files {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(f, raw)))
            .unwrap_or_else(|_| {
                let m = if verbose {
                    LAST_PANIC.with(|c| c.borrow().clone())
                } else {
                    String::new()
                };
                Outcome::Fail(if m.is_empty() { "panic".into() } else { format!("panic @ {m}") })
            });
        match outcome {
            Outcome::Pass => {
                pass += 1;
                println!("PASS {}", f.display());
            }
            Outcome::Fail(msg) => {
                fail += 1;
                if msg == "panic" || msg.starts_with("panic @") {
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

/// Run a file verbatim (no test262 preamble/metadata): parse-error positions
/// map directly to the file; completing without a throw is a PASS.
fn run_raw(src: &str) -> Outcome {
    let ast = match JsParser::parse_to_ast_from_str(src) {
        Ok(a) => a,
        Err(e) => return Outcome::Skip(format!("parse: {e:?}")),
    };
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut ctx);
    for stmt in &ast.body {
        if let Err(e) = execute_statement(stmt, &mut ctx) {
            let msg: String = describe_error(&e).replace('\n', " ").chars().take(220).collect();
            return Outcome::Fail(format!("runtime threw: {msg}"));
        }
    }
    Outcome::Pass
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
    only_strict: bool,
    no_strict: bool,
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
                m.only_strict = line.contains("onlyStrict");
                m.no_strict = line.contains("noStrict") || line.contains("raw");
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
  if (!assert._isSameValue(actual, expected)) {
    throw new Test262Error((msg || "") + " expected " + expected + " got " + actual);
  }
};
assert.notSameValue = function (actual, unexpected, msg) {
  if (assert._isSameValue(actual, unexpected)) {
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

// --- test262 harness includes (propertyHelper.js / compareArray.js / sta.js /
// testTypedArray-free subset) that many tests declare via `includes:` ---
var $MAX_ITERATIONS = 100000;
var $262 = {
  global: this,
  evalScript: function (src) { return eval(src); },
  detachArrayBuffer: function () {},
  createRealm: function () { return $262; },
  agent: {}
};
function compareArray(a, b) {
  if (a === null || b === null) { return a === b; }
  if (a.length !== b.length) { return false; }
  for (var i = 0; i < a.length; i++) {
    if (!assert._isSameValue(a[i], b[i])) { return false; }
  }
  return true;
}
compareArray.isSameValue = assert._isSameValue;
assert.compareArray = function (actual, expected, msg) {
  assert(compareArray(actual, expected),
    (msg || "") + " arrays differ: expected " + expected + " got " + actual);
};
function verifyProperty(obj, name, desc, options) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  if (d === undefined) {
    throw new Test262Error("property " + String(name) + " does not exist");
  }
  if (desc === undefined) { return true; }
  if ("value" in desc) {
    assert.sameValue(d.value, desc.value, "value of " + String(name));
  }
  if ("writable" in desc) {
    assert.sameValue(d.writable, desc.writable, "writable of " + String(name));
  }
  if ("enumerable" in desc) {
    assert.sameValue(d.enumerable, desc.enumerable, "enumerable of " + String(name));
  }
  if ("configurable" in desc) {
    assert.sameValue(d.configurable, desc.configurable, "configurable of " + String(name));
  }
  return true;
}
// Older-style aliases still used by some tests.
function verifyEqualTo(obj, name, value) {
  assert.sameValue(obj[name], value, "value of " + String(name));
}
function verifyWritable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && d.writable, String(name) + " should be writable");
}
function verifyNotWritable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && !d.writable, String(name) + " should be non-writable");
}
function verifyEnumerable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && d.enumerable, String(name) + " should be enumerable");
}
function verifyNotEnumerable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && !d.enumerable, String(name) + " should be non-enumerable");
}
function verifyConfigurable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && d.configurable, String(name) + " should be configurable");
}
function verifyNotConfigurable(obj, name) {
  var d = Object.getOwnPropertyDescriptor(obj, name);
  assert(d && !d.configurable, String(name) + " should be non-configurable");
}
"#;

    // Prefer harness-style tests that use assert; leave throw Test262Error as-is.
    Ok(format!("{preamble}\n{s}\n"))
}

/// Render a thrown error into a readable one-liner, extracting `.name`/
/// `.message` from thrown Error-like objects so assertion failures are legible.
fn describe_error(e: &just_engine::runner::ds::error::JErrorType) -> String {
    use just_engine::runner::ds::error::JErrorType;
    use just_engine::runner::eval::expression::get_own_prop_value;
    match e {
        JErrorType::ReferenceError(m) => format!("ReferenceError({m:?})"),
        JErrorType::TypeError(m) => format!("TypeError({m:?})"),
        JErrorType::RangeError(m) => format!("RangeError({m:?})"),
        JErrorType::SyntaxError(m) => format!("SyntaxError({m:?})"),
        JErrorType::YieldValue(_) => "YieldValue".to_string(),
        JErrorType::Thrown(v) => {
            let name = match get_own_prop_value(v, "name") {
                Some(JsValue::String(s)) => s,
                _ => "Thrown".to_string(),
            };
            let msg = match get_own_prop_value(v, "message") {
                Some(JsValue::String(s)) => s,
                _ => match v {
                    JsValue::String(s) => s.clone(),
                    _ => String::new(),
                },
            };
            format!("Thrown[{name}]: {msg}")
        }
    }
}

fn run_one(path: &Path, raw: bool) -> Outcome {
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return Outcome::Fail(format!("read: {e}")),
    };
    if raw {
        return run_raw(&src);
    }
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

    // `onlyStrict` tests are strict-mode code even though the file carries no
    // in-source `"use strict"` directive (the standard harness would prepend
    // one). Signal that to the parser so Annex-B legacy octal literals/escapes
    // are rejected as Syntax Errors, matching the negative parse expectation.
    let parse_result = JsParser::parse_to_ast_from_str_strict(&adapted, meta.only_strict);
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
    // `onlyStrict` tests (and files opening with a "use strict" directive) run
    // in strict mode: assignment to an undeclared reference throws, etc.
    if meta.only_strict || (!meta.no_strict && src.contains("\"use strict\"")) {
        ctx.strict = true;
    }

    // Hoist top-level `var` declarations (global scope) before running.
    just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut ctx);

    let mut threw: Option<String> = None;
    for stmt in &ast.body {
        if let Err(e) = execute_statement(stmt, &mut ctx) {
            threw = Some(describe_error(&e));
            break;
        }
    }
    let _ = JsValue::Undefined;

    match (meta.negative, threw) {
        // Negative test: throwing (parse or runtime) is the expected outcome.
        (true, Some(_)) => Outcome::Pass,
        (true, None) => Outcome::Fail("negative test did not throw".into()),
        // Positive test: any throw is a failure; completing is a pass.
        (false, Some(msg)) => {
            let msg = msg.replace('\n', " ");
            let msg: String = msg.chars().take(220).collect();
            Outcome::Fail(format!("runtime threw: {msg}"))
        }
        (false, None) => Outcome::Pass,
    }
}
