//! ES6 JavaScript tier backed by the vendored, no_std'd `just-engine`
//! (applegrew/just — see `third_party/just-ref` and THIRDPARTY-LICENSES.md).
//!
//! This is the browser's higher-capability JS runtime: a full ES6 tree-walking
//! interpreter (classes + inheritance, prototypes, closures, generators,
//! `instanceof`, destructuring, getters/setters) — the surface the hand-rolled
//! [`super::js`] tree-walker and [`super::js_bc`] bytecode VM don't cover.
//!
//! `just` sits **below the determinism boundary**: it is native, deterministic
//! Rust that only manipulates values and (via a host `PluginResolver`, added in
//! the DOM stage) the sandboxed browser DOM. It never touches Synapse
//! capabilities, the filesystem, or the network directly — page JS stays
//! `untrusted_ingested` and confined exactly as it is under [`super::js`].

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use just_engine::parser::JsParser;
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::ds::value::JsValue;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;
use just_engine::runner::std_lib::console::drain_console_log;

/// The outcome of evaluating a program: the string form of the last statement's
/// completion value, plus every line written to `console.*` during the run.
pub struct JsEval {
    pub value: String,
    pub log: Vec<String>,
}

/// Parse + execute a JavaScript program with the core built-ins installed
/// (`Math`, `JSON`, `Array`, `Object`, `String`, `Number`, `console`, the
/// `Error` constructors). No DOM bindings yet — that arrives with the
/// `DomResolver` tier. Returns `Err(msg)` on a parse or runtime error.
pub fn eval_program(src: &str) -> Result<JsEval, String> {
    // Drain any console residue from a previous run so `log` is scoped to this call.
    let _ = drain_console_log();

    let ast = JsParser::parse_to_ast_from_str(src)
        .map_err(|e| format!("SyntaxError: {:?}", e))?;

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());

    let mut last = JsValue::Undefined;
    for stmt in &ast.body {
        match execute_statement(stmt, &mut ctx) {
            Ok(completion) => {
                if let Some(v) = completion.value {
                    last = v;
                }
            }
            Err(e) => {
                let log = drain_console_log();
                let mut msg = format!("{:?}", e);
                if !log.is_empty() {
                    msg.push_str(" (console: ");
                    msg.push_str(&log.join(" | "));
                    msg.push(')');
                }
                return Err(msg);
            }
        }
    }

    Ok(JsEval {
        value: display_value(&last),
        log: drain_console_log(),
    })
}

/// Human-facing rendering of a top-level result value (REPL-style). Distinct
/// from `console.log` formatting: strings are shown quoted here.
fn display_value(v: &JsValue) -> String {
    use just_engine::runner::ds::value::JsNumberType;
    match v {
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Null => "null".to_string(),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::String(s) => format!("\"{}\"", s),
        JsValue::Number(n) => match n {
            JsNumberType::Integer(i) => i.to_string(),
            JsNumberType::Float(f) => f.to_string(),
            JsNumberType::NaN => "NaN".to_string(),
            JsNumberType::PositiveInfinity => "Infinity".to_string(),
            JsNumberType::NegativeInfinity => "-Infinity".to_string(),
        },
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val(src: &str) -> String {
        eval_program(src).expect("eval ok").value
    }

    // NB: these exercise `just`'s genuinely-working ES surface. `just` is an
    // experimental engine with real upstream gaps (arrow functions, `class …
    // extends`, most Array/String *instance* methods, `JSON.parse` of objects,
    // `Object.keys`, and a grammar that wants whitespace after `while`/`for`
    // before `(`). Those are being addressed incrementally; tests here assert
    // only what the ported engine reliably does today, so the suite is a true
    // regression gate rather than a wish list.

    #[test_case]
    fn arithmetic_and_precedence() {
        assert_eq!(val("1 + 2 * 3"), "7");
    }

    #[test_case]
    fn closures_capture_state() {
        assert_eq!(
            val("function mk(){let n=0; return function(){ n = n + 1; return n; };} let c = mk(); c(); c(); c()"),
            "3"
        );
    }

    #[test_case]
    fn single_class_with_method() {
        let src = "class A { constructor(x){ this.x = x; } dbl(){ return this.x * 2; } } \
                   let a = new A(21); a.dbl()";
        assert_eq!(val(src), "42");
    }

    #[test_case]
    fn objects_members_and_assignment() {
        assert_eq!(val("let o = { a: 1 }; o.b = 2; o.a + o.b"), "3");
    }

    #[test_case]
    fn arrays_index_and_length() {
        assert_eq!(val("let a = [10, 20, 30]; a[0] + a[2] + a.length"), "43");
    }

    #[test_case]
    fn iife_and_ternary() {
        assert_eq!(val("(function(x){ return x > 3 ? x * x : 0; })(6)"), "36");
    }

    #[test_case]
    fn try_catch_throw() {
        assert_eq!(val("try { throw 'e'; } catch (e) { 'caught:' + e }"), "\"caught:e\"");
    }

    #[test_case]
    fn destructuring_and_typeof() {
        assert_eq!(val("let { a, b } = { a: 1, b: 2 }; typeof (a + b)"), "\"number\"");
    }

    #[test_case]
    fn for_loop_accumulates() {
        assert_eq!(val("let s = 0; for (var i = 0; i < 5; i = i + 1) { s = s + i; } s"), "10");
    }

    #[test_case]
    fn console_log_captured() {
        let out = eval_program("console.log('hello', 42); 0").expect("eval");
        assert_eq!(out.log.len(), 1);
        assert_eq!(out.log[0], "hello 42");
    }

    #[test_case]
    fn syntax_error_is_reported() {
        assert!(eval_program("function (").is_err());
    }

    // --- interpreter gaps fixed in the ChittiOS port (arrows, extends/super,
    // instance methods, Object statics, JSON objects, String()/Number(),
    // minified loops). These are the features `just` upstream could not run.

    #[test_case]
    fn arrow_functions() {
        assert_eq!(val("let f = x => x + 1; f(9)"), "10");
        assert_eq!(val("let g = (a, b) => { return a * b; }; g(6, 7)"), "42");
    }

    #[test_case]
    fn class_extends_and_super() {
        let src = "class A { constructor(x){ this.x = x; } g(){ return this.x * 2; } } \
                   class B extends A { constructor(x){ super(x); } h(){ return this.x * 3; } } \
                   let b = new B(5); b.g() + b.h()";
        assert_eq!(val(src), "25");
    }

    #[test_case]
    fn array_instance_methods() {
        assert_eq!(
            val("[1,2,3,4].map(x => x*x).filter(x => x > 3).reduce((a,b) => a+b, 0)"),
            "29"
        );
        assert_eq!(val("let a=[1,2]; a.push(3); a.join('-')"), "\"1-2-3\"");
    }

    #[test_case]
    fn string_instance_methods() {
        assert_eq!(val("'hello'.toUpperCase()"), "\"HELLO\"");
        assert_eq!(val("'a,b,c'.split(',').length"), "3");
    }

    #[test_case]
    fn object_statics() {
        assert_eq!(val("Object.keys({a:1, b:2}).length"), "2");
        assert_eq!(val("let t = {}; Object.assign(t, {x: 5}); t.x"), "5");
    }

    #[test_case]
    fn json_parse_objects() {
        assert_eq!(val("JSON.parse('{\"a\":41}').a + 1"), "42");
        assert_eq!(val("JSON.parse('{\"a\":[1,2],\"b\":{\"c\":3}}').b.c"), "3");
    }

    #[test_case]
    fn string_and_number_callable() {
        assert_eq!(val("String(42)"), "\"42\"");
        assert_eq!(val("Number('42') + 1"), "43");
    }

    #[test_case]
    fn minified_loops() {
        assert_eq!(val("var s=0; for(var i=0;i<5;i++){s=s+i;} s"), "10");
        assert_eq!(val("var i=0; while(i<3){i=i+1;} i"), "3");
    }

    // --- Stage F built-ins added by the ChittiOS port ---------------------

    #[test_case]
    fn map_and_set() {
        assert_eq!(val("let m=new Map(); m.set('a',1); m.set('a',9); m.get('a')"), "9");
        assert_eq!(val("let m=new Map(); m.set('a',1); m.set('b',2); m.size"), "2");
        assert_eq!(val("let s=new Set([1,2,2,3]); s.size"), "3");
    }

    #[test_case]
    fn date_from_epoch() {
        assert_eq!(val("let d=new Date(1700000000000); d.getFullYear()"), "2023");
        assert_eq!(val("let d=new Date(42); d.getTime()"), "42");
    }

    #[test_case]
    fn regexp_test_and_match() {
        assert_eq!(val("let r=/\\d+/; r.test('x42')"), "true");
        assert_eq!(val("'a1b2c3'.match(/\\d/g).length"), "3");
        assert_eq!(
            val("'2023-11-14'.replace(/(\\d+)-(\\d+)-(\\d+)/, '$3/$2/$1')"),
            "\"14/11/2023\""
        );
    }

    #[test_case]
    fn promise_then_chain() {
        assert_eq!(val("let out=0; Promise.resolve(10).then(v=>v*2).then(v=>{out=v;}); out"), "20");
        assert_eq!(val("let out=0; let p=new Promise((res)=>{res(7);}); p.then(v=>{out=v;}); out"), "7");
        assert_eq!(val("let out=0; Promise.reject('e').catch(x=>{out=x;}); out"), "\"e\"");
    }
}
