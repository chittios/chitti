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

// ============================================================================
// Stage D: DOM bindings via a `PluginResolver`.
//
// The DOM lives Rust-side in `JsDom`/`ElemRef` (integer-handle model). To run a
// DOM script on `just` we (1) move the `JsDom` into an `Rc<RefCell<>>` the
// resolver shares, (2) expose `document`/`window`/`location`/`localStorage` and
// element wrappers, (3) let the script read/write wrapper properties as normal
// JS objects, then (4) **sync** each wrapper's fields back into the `JsDom`.
// This needs no interpreter changes — wrappers are ordinary `just` objects.
//
// Security: the resolver exposes ONLY the sandboxed `JsDom` (same surface the
// hand-rolled engine touches). No Synapse/fs/net — the determinism/taint
// boundary is unchanged.
// ============================================================================

use alloc::rc::Rc;
use alloc::vec;
use core::cell::RefCell;
use just_engine::runner::ds::error::JErrorType;
use just_engine::runner::ds::value::JsNumberType;
use just_engine::runner::eval::expression::{get_own_prop_value, make_array, make_object, set_own_prop};
use just_engine::runner::plugin::resolver::PluginResolver;
use super::js::{empty_elem, ElemRef, JsDom};

/// Records of element wrappers handed to the script, to be synced back.
type Wrappers = Rc<RefCell<Vec<(usize, JsValue)>>>;

struct DomResolver {
    dom: Rc<RefCell<JsDom>>,
    wrappers: Wrappers,
    document: JsValue,
    window: JsValue,
    location: JsValue,
    storage: JsValue,
}

fn s(v: &str) -> JsValue {
    JsValue::String(v.to_string())
}
fn as_str(v: &JsValue) -> String {
    match v {
        JsValue::String(x) => x.clone(),
        JsValue::Undefined | JsValue::Null => String::new(),
        other => other.to_string(),
    }
}

/// Build a wrapper object mirroring `dom.elements[idx]`, and record it.
fn make_elem_wrapper(dom: &JsDom, idx: usize, wrappers: &Wrappers) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s("Element"), false);
    set_own_prop(&w, "__elem_index__", JsValue::Number(JsNumberType::Integer(idx as i64)), false);
    if let Some(e) = dom.elements.get(idx) {
        set_own_prop(&w, "tagName", s(&e.tag.to_uppercase()), true);
        set_own_prop(&w, "innerText", s(&e.text), true);
        set_own_prop(&w, "textContent", s(&e.text), true);
        set_own_prop(&w, "value", s(&e.value), true);
        set_own_prop(&w, "id", s(e.id.as_deref().unwrap_or("")), true);
        set_own_prop(&w, "className", s(e.class.as_deref().unwrap_or("")), true);
        set_own_prop(&w, "style", s(&e.style), true);
    }
    wrappers.borrow_mut().push((idx, w.clone()));
    w
}

impl DomResolver {
    fn new(dom: Rc<RefCell<JsDom>>, wrappers: Wrappers) -> Self {
        let document = make_object(vec![]);
        set_own_prop(&document, "__builtin_name__", s("document"), false);
        set_own_prop(&document, "title", s(&dom.borrow().title), true);
        let location = make_object(vec![]);
        set_own_prop(&location, "__builtin_name__", s("location"), false);
        set_own_prop(&location, "href", s(&dom.borrow().location_href), true);
        let storage = make_object(vec![]);
        set_own_prop(&storage, "__builtin_name__", s("localStorage"), false);
        let window = make_object(vec![]);
        set_own_prop(&window, "__builtin_name__", s("window"), false);
        set_own_prop(&window, "location", location.clone(), true);
        set_own_prop(&window, "document", document.clone(), true);
        DomResolver { dom, wrappers, document, window, location, storage }
    }

    fn find_by_id(&self, id: &str) -> Option<usize> {
        self.dom.borrow().elements.iter().position(|e| e.id.as_deref() == Some(id))
    }
    fn find_by_selector(&self, sel: &str) -> Option<usize> {
        let dom = self.dom.borrow();
        if let Some(id) = sel.strip_prefix('#') {
            dom.elements.iter().position(|e| e.id.as_deref() == Some(id))
        } else if let Some(cls) = sel.strip_prefix('.') {
            dom.elements.iter().position(|e| e.class.as_deref().map_or(false, |c| c.split_whitespace().any(|w| w == cls)))
        } else {
            let t = sel.to_ascii_lowercase();
            dom.elements.iter().position(|e| e.tag == t)
        }
    }

}

impl PluginResolver for DomResolver {
    fn has_binding(&self, name: &str) -> bool {
        matches!(name, "document" | "window" | "location" | "localStorage" | "Element" | "navigator")
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        Ok(match name {
            "document" => self.document.clone(),
            "window" => self.window.clone(),
            "location" => self.location.clone(),
            "localStorage" => self.storage.clone(),
            "navigator" => {
                let n = make_object(vec![]);
                set_own_prop(&n, "userAgent", s("ChittiOS/just"), true);
                n
            }
            _ => JsValue::Undefined,
        })
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        _ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let arg0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        match (object_name, method_name) {
            ("document", "getElementById") => {
                let id = as_str(&arg0);
                Some(Ok(match self.find_by_id(&id) {
                    Some(i) => make_elem_wrapper(&self.dom.borrow(), i, &self.wrappers),
                    None => JsValue::Null,
                }))
            }
            ("document", "querySelector") => {
                let sel = as_str(&arg0);
                Some(Ok(match self.find_by_selector(&sel) {
                    Some(i) => make_elem_wrapper(&self.dom.borrow(), i, &self.wrappers),
                    None => JsValue::Null,
                }))
            }
            ("document", "querySelectorAll") | ("document", "getElementsByTagName") => {
                let sel = as_str(&arg0);
                let want = sel.to_ascii_lowercase();
                let idxs: Vec<usize> = {
                    let dom = self.dom.borrow();
                    dom.elements.iter().enumerate()
                        .filter(|(_, e)| want == "*" || e.tag == want)
                        .map(|(i, _)| i).collect()
                };
                let items = idxs.into_iter().map(|i| make_elem_wrapper(&self.dom.borrow(), i, &self.wrappers)).collect();
                Some(Ok(make_array(items)))
            }
            ("document", "createElement") => {
                let tag = as_str(&arg0);
                let idx = {
                    let mut dom = self.dom.borrow_mut();
                    dom.elements.push(empty_elem(&tag));
                    dom.elements.len() - 1
                };
                Some(Ok(make_elem_wrapper(&self.dom.borrow(), idx, &self.wrappers)))
            }
            ("Element", "setAttribute") => {
                if let JsValue::Object(_) = this {
                    if let Some(JsValue::Number(JsNumberType::Integer(i))) = get_own_prop_value(&this, "__elem_index__") {
                        let key = as_str(&arg0);
                        let val = as_str(&args.get(1).cloned().unwrap_or(JsValue::Undefined));
                        let mut dom = self.dom.borrow_mut();
                        if let Some(e) = dom.elements.get_mut(i as usize) {
                            match key.as_str() {
                                "id" => { e.id = Some(val.clone()); set_own_prop(&this, "id", s(&val), true); }
                                "class" => { e.class = Some(val.clone()); set_own_prop(&this, "className", s(&val), true); }
                                _ => { e.attrs.insert(key, val); }
                            }
                        }
                    }
                }
                Some(Ok(JsValue::Undefined))
            }
            ("Element", "getAttribute") => {
                let key = as_str(&arg0);
                if let Some(JsValue::Number(JsNumberType::Integer(i))) = get_own_prop_value(&this, "__elem_index__") {
                    let dom = self.dom.borrow();
                    if let Some(e) = dom.elements.get(i as usize) {
                        return Some(Ok(e.attrs.get(&key).map(|v| s(v)).unwrap_or(JsValue::Null)));
                    }
                }
                Some(Ok(JsValue::Null))
            }
            ("Element", "appendChild") => {
                let child_idx = get_own_prop_value(&arg0, "__elem_index__");
                let parent_idx = get_own_prop_value(&this, "__elem_index__");
                if let (Some(JsValue::Number(JsNumberType::Integer(p))), Some(JsValue::Number(JsNumberType::Integer(c)))) = (parent_idx, child_idx) {
                    let mut dom = self.dom.borrow_mut();
                    let (p, c) = (p as usize, c as usize);
                    if p < dom.elements.len() && c < dom.elements.len() {
                        dom.elements[p].children.push(c);
                        dom.elements[c].parent = Some(p);
                    }
                }
                Some(Ok(arg0))
            }
            ("Element", "addEventListener") => {
                // Recorded but not dispatched here (no event loop on this tier).
                if let Some(JsValue::Number(JsNumberType::Integer(i))) = get_own_prop_value(&this, "__elem_index__") {
                    let ev = as_str(&arg0);
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i as usize) {
                        e.listeners.entry(ev).or_default();
                    }
                }
                Some(Ok(JsValue::Undefined))
            }
            ("localStorage", "getItem") => {
                let key = as_str(&arg0);
                Some(Ok(get_own_prop_value(&self.storage, &key).unwrap_or(JsValue::Null)))
            }
            ("localStorage", "setItem") => {
                let key = as_str(&arg0);
                let val = args.get(1).cloned().unwrap_or(JsValue::Undefined);
                set_own_prop(&self.storage, &key, val, true);
                Some(Ok(JsValue::Undefined))
            }
            _ => None,
        }
    }

    fn name(&self) -> &str {
        "chitti_dom"
    }
}

/// Run DOM-touching scripts on `just` with DOM bindings. Returns `false` (having
/// left `dom` untouched) if any script fails to parse/run, so the caller can
/// fall back to the legacy engine. On success, DOM mutations are synced back.
pub fn run_scripts_via_just(dom: &mut JsDom, scripts: &[String]) -> bool {
    // Parse everything FIRST — if any script can't parse, bail before touching
    // the DOM so the caller can fall back cleanly (no partial/double execution).
    let mut asts = Vec::with_capacity(scripts.len());
    for src in scripts {
        match JsParser::parse_to_ast_from_str(src) {
            Ok(ast) => asts.push(ast),
            Err(_) => return false,
        }
    }

    let _ = drain_console_log();
    // Move the JsDom into a shared cell for the resolver.
    let placeholder = JsDom::from_document(&super::html::parse(""));
    let taken = core::mem::replace(dom, placeholder);
    let shared = Rc::new(RefCell::new(taken));
    let wrappers: Wrappers = Rc::new(RefCell::new(Vec::new()));
    let resolver = DomResolver::new(shared.clone(), wrappers.clone());
    let document = resolver.document.clone();
    let location = resolver.location.clone();

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.add_resolver(alloc::boxed::Box::new(resolver));

    // Execute; a runtime error just ends that script (like the legacy engine).
    for ast in &asts {
        for stmt in &ast.body {
            if execute_statement(stmt, &mut ctx).is_err() {
                break;
            }
        }
    }

    // Sync wrapper/document/location mutations back into the shared JsDom.
    sync_back_all(&shared, &wrappers, &document, &location);
    drop(ctx); // releases the resolver's Rc clones

    // Reclaim the JsDom.
    match Rc::try_unwrap(shared) {
        Ok(cell) => *dom = cell.into_inner(),
        Err(_still_shared) => return false, // shouldn't happen
    }
    for line in drain_console_log() {
        dom.log.push(line);
    }
    true
}

/// Free-function sync (the resolver's clones live inside `ctx`; we hold our own).
fn sync_back_all(dom: &Rc<RefCell<JsDom>>, wrappers: &Wrappers, document: &JsValue, location: &JsValue) {
    let mut d = dom.borrow_mut();
    for (idx, w) in wrappers.borrow().iter() {
        let Some(e) = d.elements.get_mut(*idx) else { continue };
        if let Some(v) = get_own_prop_value(w, "innerText") {
            e.text = as_str(&v);
        }
        if let Some(v) = get_own_prop_value(w, "value") {
            e.value = as_str(&v);
        }
        if let Some(v) = get_own_prop_value(w, "style") {
            e.style = as_str(&v);
        }
        if let Some(v) = get_own_prop_value(w, "className") {
            e.class = Some(as_str(&v));
        }
        if let Some(v) = get_own_prop_value(w, "id") {
            e.id = Some(as_str(&v));
        }
    }
    if let Some(v) = get_own_prop_value(document, "title") {
        d.title = as_str(&v);
    }
    if let Some(v) = get_own_prop_value(location, "href") {
        let href = as_str(&v);
        if href != d.location_href {
            d.navigate = Some(href);
        }
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

    #[test_case]
    fn async_await() {
        assert_eq!(val("async function f(){ return 41; } let out = await f(); out + 1"), "42");
        assert_eq!(val("async function g(){ return await Promise.resolve(100); } let out = await g(); out"), "100");
        // async fn returns a Promise usable with .then
        assert_eq!(val("async function f(){ return 7; } let out=0; let p=f(); p.then(v=>{out=v*2;}); out"), "14");
    }

    #[test_case]
    fn proxy_traps() {
        assert_eq!(val("let p=new Proxy({}, {get:(t,k)=>'got:'+k}); p.foo"), "\"got:foo\"");
        assert_eq!(val("let log=''; let p=new Proxy({}, {set:(t,k,v)=>{log=k+'='+v;}}); p.a=9; log"), "\"a=9\"");
        assert_eq!(val("let t={x:5}; let p=new Proxy(t, {}); p.x"), "5");
    }

    #[test_case]
    fn reflect_static() {
        assert_eq!(val("Reflect.get({x:7}, 'x')"), "7");
        assert_eq!(val("let o={}; Reflect.set(o,'k',3); o.k"), "3");
        assert_eq!(val("Reflect.has({a:1}, 'a')"), "true");
        assert_eq!(val("let f=(a,b)=>a+b; Reflect.apply(f, null, [4,5])"), "9");
    }

    // --- Stage D: DOM bindings via the DomResolver (run_scripts_via_just) ---

    #[test_case]
    fn dom_get_element_and_set_text() {
        use crate::browser::html;
        let doc = html::parse(r#"<html><body><p id="msg">old</p></body></html>"#);
        let mut dom = JsDom::from_document(&doc);
        let ok = run_scripts_via_just(
            &mut dom,
            &[String::from("let el = document.getElementById('msg'); el.innerText = 'hello ' + (1 + 1);")],
        );
        assert!(ok, "just DOM run should succeed");
        let el = dom.elements.iter().find(|e| e.id.as_deref() == Some("msg")).unwrap();
        assert_eq!(el.text, "hello 2");
    }

    #[test_case]
    fn dom_title_and_create_element() {
        use crate::browser::html;
        let doc = html::parse("<html><head><title>Old</title></head><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        let ok = run_scripts_via_just(
            &mut dom,
            &[String::from(
                "document.title = 'New'; let d = document.createElement('div'); d.className = 'box'; console.log('made', d.tagName);",
            )],
        );
        assert!(ok);
        assert_eq!(dom.title, "New");
        assert!(dom.elements.iter().any(|e| e.tag == "div" && e.class.as_deref() == Some("box")));
        assert!(dom.log.iter().any(|l| l.contains("made") && l.contains("DIV")));
    }
}
