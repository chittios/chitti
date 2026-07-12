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
    just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut ctx);

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
// Stage D: LIVE DOM bindings — the `just` DOM tier is primary for DOM scripts.
//
// `DomProps` (a `NativeProps`) backs element / document / window / location /
// style property get/set directly against a shared `Rc<RefCell<JsDom>>`, so
// reads and writes are LIVE (a `querySelector` after `el.className='x'` sees
// it, `childElementCount` reflects `appendChild` immediately). Methods route
// through `DomResolver` (`PluginResolver`) keyed by `__builtin_name__`. Element
// wrappers carry `__native_node__` = element index + `__builtin_name__`. There
// is NO end-of-run sync — every mutation hits the JsDom immediately.
//
// Security: only the sandboxed JsDom is exposed — no Synapse / fs / net. The
// determinism/taint boundary is unchanged.
// ============================================================================

use alloc::rc::Rc;
use alloc::vec;
use core::cell::RefCell;
use just_engine::runner::ds::error::JErrorType;
use just_engine::runner::ds::value::JsNumberType;
use just_engine::runner::eval::expression::{
    call_value, get_own_prop_value, make_array, make_object, set_own_prop, to_boolean,
    value_is_callable,
};
use just_engine::runner::plugin::resolver::PluginResolver;
use just_engine::runner::plugin::types::NativeProps;
use just_engine::runner::std_lib::promise;
use super::js::{empty_elem, JsDom};

const STYLE_OFFSET: i64 = 1_000_000;
const CANVAS_OFFSET: i64 = 2_000_000;
const DOC_NODE: i64 = -1;
const WIN_NODE: i64 = -2;
const LOC_NODE: i64 = -3;

fn s(v: &str) -> JsValue {
    JsValue::String(v.to_string())
}
fn num(n: i64) -> JsValue {
    JsValue::Number(JsNumberType::Integer(n))
}
fn as_str(v: &JsValue) -> String {
    match v {
        JsValue::String(x) => x.clone(),
        JsValue::Undefined | JsValue::Null => String::new(),
        other => other.to_string(),
    }
}
fn truthy(v: &JsValue) -> bool {
    match v {
        JsValue::Undefined | JsValue::Null => false,
        JsValue::Boolean(b) => *b,
        JsValue::Number(JsNumberType::Integer(0)) => false,
        JsValue::String(x) => !x.is_empty(),
        _ => true,
    }
}

fn elem_wrapper(i: usize) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s("Element"), false);
    set_own_prop(&w, "__native_node__", num(i as i64), false);
    w
}
fn style_wrapper(i: usize) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s("Style"), false);
    set_own_prop(&w, "__native_node__", num(STYLE_OFFSET + i as i64), false);
    set_own_prop(&w, "__style_elem__", num(i as i64), false);
    w
}
fn classlist_wrapper(i: usize) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s("ClassList"), false);
    set_own_prop(&w, "__cl_node__", num(i as i64), false);
    w
}
fn canvas_ctx_wrapper(i: usize) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s("Canvas2d"), false);
    set_own_prop(&w, "__native_node__", num(CANVAS_OFFSET + i as i64), false);
    set_own_prop(&w, "__canvas_elem__", num(i as i64), false);
    w
}
fn response_wrapper(body: &str) -> JsValue {
    let r = make_object(vec![]);
    set_own_prop(&r, "__builtin_name__", s("Response"), false);
    set_own_prop(&r, "__body__", s(body), false);
    set_own_prop(&r, "ok", JsValue::Boolean(true), true);
    set_own_prop(&r, "status", num(200), true);
    set_own_prop(&r, "statusText", s("OK"), true);
    r
}
/// Bare window global functions (called as `foo(...)`, not `window.foo(...)`).
fn is_bare_global(name: &str) -> bool {
    matches!(
        name,
        "scrollTo" | "scrollBy" | "scroll" | "alert" | "confirm" | "prompt"
            | "setTimeout" | "setInterval" | "clearTimeout" | "clearInterval"
            | "requestAnimationFrame" | "cancelAnimationFrame" | "queueMicrotask"
            | "encodeURIComponent" | "decodeURIComponent" | "encodeURI" | "decodeURI"
            // Bare event-target methods at global scope == `window.X(…)`.
            | "addEventListener" | "removeEventListener" | "dispatchEvent"
    )
}

fn percent_encode(input: &str, keep: &str) -> String {
    let mut out = String::new();
    for b in input.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || keep.contains(c) {
            out.push(c);
        } else {
            out.push('%');
            out.push_str(&alloc::format!("{:02X}", b));
        }
    }
    out
}
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = core::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn num_arg(v: &JsValue) -> i32 {
    match v {
        JsValue::Number(JsNumberType::Integer(n)) => *n as i32,
        JsValue::Number(JsNumberType::Float(f)) => *f as i32,
        JsValue::String(x) => x.trim().parse().unwrap_or(0),
        _ => 0,
    }
}
fn fnum_arg(v: &JsValue) -> f32 {
    match v {
        JsValue::Number(JsNumberType::Integer(n)) => *n as f32,
        JsValue::Number(JsNumberType::Float(f)) => *f as f32,
        JsValue::String(x) => x.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

// --- CSS inline-style string helpers ("a: b; c: d") --------------------------

fn camel_to_kebab(p: &str) -> String {
    let mut out = String::new();
    for c in p.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}
fn style_get(decl: &str, prop: &str) -> String {
    for part in decl.split(';') {
        if let Some((k, v)) = part.split_once(':') {
            if k.trim() == prop {
                return v.trim().to_string();
            }
        }
    }
    String::new()
}
fn style_set(decl: &str, prop: &str, val: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut found = false;
    for part in decl.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((k, _)) = part.split_once(':') {
            if k.trim() == prop {
                found = true;
                if !val.is_empty() {
                    out.push(format!("{}: {}", prop, val));
                }
                continue;
            }
        }
        out.push(part.to_string());
    }
    if !found && !val.is_empty() {
        out.push(format!("{}: {}", prop, val));
    }
    out.join("; ")
}

// --- selector matching -------------------------------------------------------

fn matches_selector(e: &super::js::ElemRef, sel: &str) -> bool {
    if let Some(id) = sel.strip_prefix('#') {
        e.id.as_deref() == Some(id)
    } else if let Some(cls) = sel.strip_prefix('.') {
        e.class.as_deref().map_or(false, |c| c.split_whitespace().any(|w| w == cls))
    } else {
        e.tag == sel.to_ascii_lowercase()
    }
}

// ============================================================================
// DomProps: live property backing (NativeProps).
// ============================================================================

struct DomProps {
    dom: Rc<RefCell<JsDom>>,
}

impl DomProps {
    fn elem_get(&self, i: usize, prop: &str) -> Option<JsValue> {
        let dom = self.dom.borrow();
        let e = dom.elements.get(i)?;
        Some(match prop {
            "innerText" | "textContent" | "innerHTML" | "outerHTML" => s(&e.text),
            "value" => s(&e.value),
            "id" => s(e.id.as_deref().unwrap_or("")),
            "className" => s(e.class.as_deref().unwrap_or("")),
            "tagName" | "nodeName" => s(&e.tag.to_uppercase()),
            "childElementCount" => num(e.children.len() as i64),
            "nodeType" => num(1),
            "checked" => JsValue::Boolean(e.checked),
            "disabled" => JsValue::Boolean(e.disabled),
            "hidden" => JsValue::Boolean(e.hidden),
            "href" => s(&e.href),
            "src" => s(&e.src),
            "type" => s(&e.type_attr),
            "name" => s(&e.name_attr),
            "placeholder" => s(&e.placeholder),
            "clientWidth" | "clientHeight" | "offsetWidth" | "offsetHeight" | "scrollTop"
            | "scrollHeight" => num(0),
            "style" => style_wrapper(i),
            "classList" => classlist_wrapper(i),
            "children" => {
                let ws: Vec<JsValue> = e.children.iter().map(|&c| elem_wrapper(c)).collect();
                make_array(ws)
            }
            "firstChild" | "firstElementChild" => {
                e.children.first().map(|&c| elem_wrapper(c)).unwrap_or(JsValue::Null)
            }
            "lastChild" | "lastElementChild" => {
                e.children.last().map(|&c| elem_wrapper(c)).unwrap_or(JsValue::Null)
            }
            "parentNode" | "parentElement" => {
                e.parent.map(elem_wrapper).unwrap_or(JsValue::Null)
            }
            _ => return None, // not a property → let the call path route methods
        })
    }

    fn elem_set(&self, i: usize, prop: &str, value: JsValue) -> bool {
        let mut dom = self.dom.borrow_mut();
        let Some(e) = dom.elements.get_mut(i) else { return false };
        match prop {
            "innerText" | "textContent" | "innerHTML" => e.text = as_str(&value),
            "value" => e.value = as_str(&value),
            "id" => e.id = Some(as_str(&value)),
            "className" => e.class = Some(as_str(&value)),
            "checked" => e.checked = truthy(&value),
            "disabled" => e.disabled = truthy(&value),
            "hidden" => e.hidden = truthy(&value),
            "href" => e.href = as_str(&value),
            "src" => e.src = as_str(&value),
            _ => return false,
        }
        true
    }

    fn style_get(&self, i: usize, prop: &str) -> Option<JsValue> {
        if matches!(prop, "setProperty" | "getPropertyValue" | "removeProperty") {
            return None; // methods
        }
        let dom = self.dom.borrow();
        let e = dom.elements.get(i)?;
        if prop == "cssText" {
            return Some(s(&e.style));
        }
        Some(s(&style_get(&e.style, &camel_to_kebab(prop))))
    }

    fn style_set(&self, i: usize, prop: &str, value: JsValue) -> bool {
        let mut dom = self.dom.borrow_mut();
        let Some(e) = dom.elements.get_mut(i) else { return false };
        if prop == "cssText" {
            e.style = as_str(&value);
        } else {
            e.style = style_set(&e.style, &camel_to_kebab(prop), &as_str(&value));
        }
        true
    }

    fn doc_get(&self, prop: &str) -> Option<JsValue> {
        let dom = self.dom.borrow();
        Some(match prop {
            "title" => s(&dom.title),
            "cookie" => s(""),
            "readyState" => s("complete"),
            "body" => dom.elements.iter().position(|e| e.tag == "body").map(elem_wrapper).unwrap_or(JsValue::Null),
            "head" => dom.elements.iter().position(|e| e.tag == "head").map(elem_wrapper).unwrap_or(JsValue::Null),
            "documentElement" => dom.elements.iter().position(|e| e.tag == "html").map(elem_wrapper).unwrap_or(JsValue::Null),
            _ => return None,
        })
    }
}

impl DomProps {
    fn canvas_get(&self, i: usize, prop: &str) -> Option<JsValue> {
        match prop {
            "lineWidth" => {
                let dom = self.dom.borrow();
                Some(num(dom.canvases.get(&i).map(|c| c.line_width as i64).unwrap_or(1)))
            }
            // fillStyle/strokeStyle/font aren't read back; methods fall through.
            _ => None,
        }
    }
    fn canvas_set(&self, i: usize, prop: &str, value: JsValue) -> bool {
        let mut dom = self.dom.borrow_mut();
        let c = dom.ensure_canvas(i);
        match prop {
            "fillStyle" => c.set_fill_style_css(&as_str(&value)),
            "strokeStyle" => c.set_stroke_style_css(&as_str(&value)),
            "lineWidth" => c.line_width = num_arg(&value),
            "font" => {
                let n: i32 = as_str(&value)
                    .split(|ch: char| !ch.is_ascii_digit())
                    .find(|t| !t.is_empty())
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(14);
                c.font_size = n as f32;
            }
            _ => return false,
        }
        true
    }
}

impl NativeProps for DomProps {
    fn get(&self, node: i64, prop: &str) -> Option<JsValue> {
        if node >= CANVAS_OFFSET {
            return self.canvas_get((node - CANVAS_OFFSET) as usize, prop);
        }
        if node >= STYLE_OFFSET {
            return self.style_get((node - STYLE_OFFSET) as usize, prop);
        }
        match node {
            DOC_NODE => self.doc_get(prop),
            WIN_NODE => match prop {
                "innerWidth" => Some(num(self.dom.borrow().inner_width as i64)),
                "innerHeight" => Some(num(self.dom.borrow().inner_height as i64)),
                "name" => Some(s("")),
                _ => None,
            },
            LOC_NODE => match prop {
                "href" => Some(s(&self.dom.borrow().location_href)),
                "pathname" | "search" | "hash" | "host" | "hostname" | "protocol" => Some(s("")),
                _ => None,
            },
            n if n >= 0 => self.elem_get(n as usize, prop),
            _ => None,
        }
    }

    fn set(&self, node: i64, prop: &str, value: JsValue) -> bool {
        if node >= CANVAS_OFFSET {
            return self.canvas_set((node - CANVAS_OFFSET) as usize, prop, value);
        }
        if node >= STYLE_OFFSET {
            return self.style_set((node - STYLE_OFFSET) as usize, prop, value);
        }
        match node {
            DOC_NODE => {
                if prop == "title" {
                    self.dom.borrow_mut().title = as_str(&value);
                    true
                } else {
                    prop == "cookie" // swallow cookie writes
                }
            }
            LOC_NODE => {
                if prop == "href" {
                    let href = as_str(&value);
                    let mut dom = self.dom.borrow_mut();
                    if href != dom.location_href {
                        dom.navigate = Some(href);
                    }
                    true
                } else {
                    false
                }
            }
            n if n >= 0 => self.elem_set(n as usize, prop, value),
            _ => false,
        }
    }
}

// ============================================================================
// DomResolver: globals + methods (PluginResolver).
// ============================================================================

/// A page-JS event listener: target (element index, or the `DOC_NODE`/
/// `WIN_NODE` sentinels), event type, callback function value, capture flag.
pub(crate) struct PageListener {
    target: i64,
    type_: String,
    cb: JsValue,
    capture: bool,
}

/// Listener registry shared between the resolver (which registers) and the
/// page dispatcher. Lives inside [`JsPage`] for the page lifetime.
type ListenerReg = Rc<RefCell<Vec<PageListener>>>;

/// True when `a` and `b` are the same function object (Rc identity).
fn same_fn(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Object(x), JsValue::Object(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

struct DomResolver {
    dom: Rc<RefCell<JsDom>>,
    /// addEventListener registrations (shared with the JsPage dispatcher).
    listeners: ListenerReg,
    document: JsValue,
    window: JsValue,
    location: JsValue,
    storage: JsValue,
    session: JsValue,
    parent: JsValue,
    web_assembly: JsValue,
    fetch_fn: JsValue,
}

fn global_wrapper(name: &str, node: i64) -> JsValue {
    let w = make_object(vec![]);
    set_own_prop(&w, "__builtin_name__", s(name), false);
    set_own_prop(&w, "__native_node__", num(node), false);
    w
}

impl DomResolver {
    fn new(dom: Rc<RefCell<JsDom>>) -> Self {
        Self::with_listeners(dom, Rc::new(RefCell::new(Vec::new())))
    }

    /// Build a resolver sharing an externally-owned listener registry (the
    /// persistent-page path, so the dispatcher can invoke stored callbacks).
    fn with_listeners(dom: Rc<RefCell<JsDom>>, listeners: ListenerReg) -> Self {
        let document = global_wrapper("document", DOC_NODE);
        let window = global_wrapper("window", WIN_NODE);
        let location = global_wrapper("location", LOC_NODE);
        let storage = make_object(vec![]);
        set_own_prop(&storage, "__builtin_name__", s("localStorage"), false);
        let session = make_object(vec![]);
        set_own_prop(&session, "__builtin_name__", s("sessionStorage"), false);
        // `parent`/`self`/`top`/`window` all reference a window-like object that
        // routes postMessage to the parent frame.
        let parent = make_object(vec![]);
        set_own_prop(&parent, "__builtin_name__", s("parent"), false);
        set_own_prop(&parent, "postMessage", JsValue::Undefined, false); // marker so member-call routes
        let web_assembly = make_object(vec![]);
        set_own_prop(&web_assembly, "__builtin_name__", s("WebAssembly"), false);
        let fetch_fn = make_object(vec![]);
        set_own_prop(&fetch_fn, "__builtin_name__", s("fetch"), false);
        set_own_prop(&window, "document", document.clone(), true);
        set_own_prop(&window, "location", location.clone(), true);
        set_own_prop(&window, "parent", parent.clone(), true);
        set_own_prop(&window, "self", window.clone(), true);
        set_own_prop(&window, "top", parent.clone(), true);
        set_own_prop(&window, "localStorage", storage.clone(), true);
        set_own_prop(&window, "sessionStorage", session.clone(), true);
        // DOM interface constructors (HTMLElement/Node/EventTarget/…). Real
        // sites and libraries (bliss, jQuery) read `X.prototype` and hijack it,
        // or test `"method" in X.prototype`. Provide each as an object carrying
        // an own empty `prototype` object so those reads and
        // `Object.defineProperty(Node.prototype, …)` calls don't throw a
        // ReferenceError (which used to abort the whole page script). Left on
        // `window`, so `has_binding`/`resolve`'s window fallback serves them and
        // they stay stable across accesses. The prototypes are deliberately
        // empty — `"addEventListener" in EventTarget.prototype` is then false,
        // so bliss skips its addEventListener hijack rather than crashing in it.
        for iface in [
            "EventTarget", "Node", "Element", "HTMLElement", "HTMLDocument",
            "Document", "Window", "CharacterData", "Text", "Comment",
            "DocumentFragment", "Event", "CustomEvent", "UIEvent", "MouseEvent",
            "KeyboardEvent", "MutationObserver", "HTMLCollection", "NodeList",
            "DOMTokenList", "CSSStyleDeclaration", "ShadowRoot", "DOMParser",
            "XMLHttpRequest", "HTMLInputElement", "HTMLDivElement",
            "HTMLSpanElement", "HTMLAnchorElement", "HTMLStyleElement",
        ] {
            let ctor = make_object(vec![]);
            set_own_prop(&ctor, "prototype", make_object(vec![]), false);
            set_own_prop(&ctor, "name", s(iface), false);
            set_own_prop(&window, iface, ctor, true);
        }
        DomResolver { dom, listeners, document, window, location, storage, session, parent, web_assembly, fetch_fn }
    }

    /// Register an event listener: `target` is an element index or the
    /// `DOC_NODE`/`WIN_NODE` sentinel; `args` = (type, callback, capture?).
    fn register_listener(&self, target: i64, args: &[JsValue]) {
        let type_ = as_str(args.get(0).unwrap_or(&JsValue::Undefined));
        let cb = args.get(1).cloned().unwrap_or(JsValue::Undefined);
        if type_.is_empty() || !value_is_callable(&cb) {
            return;
        }
        // Third arg: bool `useCapture` or an options object `{capture: true}`.
        let capture = match args.get(2) {
            Some(JsValue::Boolean(b)) => *b,
            Some(o @ JsValue::Object(_)) => get_own_prop_value(o, "capture")
                .map(|v| to_boolean(&v))
                .unwrap_or(false),
            _ => false,
        };
        self.listeners.borrow_mut().push(PageListener { target, type_, cb, capture });
    }

    /// Remove listeners matching (target, type[, same callback object]).
    fn unregister_listener(&self, target: i64, args: &[JsValue]) {
        let type_ = as_str(args.get(0).unwrap_or(&JsValue::Undefined));
        let cb = args.get(1).cloned();
        self.listeners.borrow_mut().retain(|l| {
            if l.target != target || l.type_ != type_ {
                return true;
            }
            match &cb {
                Some(f) if value_is_callable(f) => !same_fn(&l.cb, f),
                _ => false, // no callback given: drop all of this type
            }
        });
    }

    /// Parse a `fetch(url, opts)` call: (method, absolute url, body).
    fn fetch_args(&self, args: &[JsValue]) -> (String, String, String) {
        let url = as_str(args.get(0).unwrap_or(&JsValue::Undefined));
        let mut method = String::from("GET");
        let mut body = String::new();
        if let Some(opts) = args.get(1) {
            if let Some(m) = get_own_prop_value(opts, "method") {
                let m = as_str(&m);
                if !m.is_empty() {
                    method = m.to_uppercase();
                }
            }
            if let Some(b) = get_own_prop_value(opts, "body") {
                body = as_str(&b);
            }
        }
        let base = self.dom.borrow().location_href.clone();
        let abs = if super::url::is_http_url(&url) {
            url.clone()
        } else {
            super::url::resolve(&base, &url).unwrap_or(url)
        };
        (method, abs, body)
    }

    fn node_of(&self, this: &JsValue) -> Option<usize> {
        // Prefer the "real element index" markers (classList/style/canvas
        // wrappers) over `__native_node__` (which is offset for style/canvas).
        let raw = get_own_prop_value(this, "__cl_node__")
            .or_else(|| get_own_prop_value(this, "__style_elem__"))
            .or_else(|| get_own_prop_value(this, "__canvas_elem__"))
            .or_else(|| get_own_prop_value(this, "__native_node__"));
        match raw {
            Some(JsValue::Number(JsNumberType::Integer(n))) if n >= 0 && n < STYLE_OFFSET => Some(n as usize),
            _ => None,
        }
    }
}

impl PluginResolver for DomResolver {
    fn has_binding(&self, name: &str) -> bool {
        is_bare_global(name)
            || matches!(
                name,
                "document" | "window" | "location" | "localStorage" | "sessionStorage"
                    | "navigator" | "parent" | "self" | "top" | "fetch" | "WebAssembly"
                    | "postMessage" | "Element" | "ClassList" | "Style" | "Canvas2d" | "Response"
                    | "Event" | "globalThis" | "performance"
            )
            // `window` IS the global object: after `window.google = {}` a bare
            // `google` must resolve to that window property (real sites — google,
            // gbar, etc. — assign to `window.X` then read `X` bare).
            || get_own_prop_value(&self.window, name).is_some()
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        Ok(match name {
            "document" => self.document.clone(),
            // `globalThis === window === self` in a browser — all resolve to the
            // one stable window object (so `globalThis.X = …` persists; the
            // engine's default `globalThis` hands back a fresh empty object).
            "window" | "self" | "globalThis" => self.window.clone(),
            "top" | "parent" => self.parent.clone(),
            "location" => self.location.clone(),
            "localStorage" => self.storage.clone(),
            "sessionStorage" => self.session.clone(),
            "fetch" => self.fetch_fn.clone(),
            "WebAssembly" => self.web_assembly.clone(),
            "postMessage" => {
                let f = make_object(vec![]);
                set_own_prop(&f, "__builtin_name__", s("postMessage"), false);
                f
            }
            "navigator" => {
                let n = make_object(vec![]);
                // Match the HTTP `User-Agent` the loader sends, so UA-sniffing
                // page scripts agree with the server's content negotiation.
                set_own_prop(&n, "userAgent", s(super::loader::BROWSER_USER_AGENT), true);
                // Firefox-consistent (Gecko): empty vendor, Linux platform.
                set_own_prop(&n, "platform", s("Linux x86_64"), true);
                set_own_prop(&n, "vendor", s(""), true);
                set_own_prop(&n, "language", s("en-US"), true);
                set_own_prop(&n, "appName", s("Netscape"), true);
                set_own_prop(&n, "product", s("Gecko"), true);
                set_own_prop(&n, "onLine", JsValue::Boolean(true), true);
                n
            }
            "performance" => {
                // `performance.now()` is dispatched via call_method; `timing`/
                // `navigation`/`timeOrigin` are read as plain properties. Real
                // sites gate feature code on `performance` existing, so defining
                // it (with a working clock) clears a very common ReferenceError.
                let p = make_object(vec![]);
                set_own_prop(&p, "__builtin_name__", s("performance"), false);
                set_own_prop(&p, "timing", make_object(vec![]), true);
                set_own_prop(&p, "navigation", make_object(vec![]), true);
                set_own_prop(&p, "timeOrigin", num(0), true);
                p
            }
            n if is_bare_global(n) => {
                let f = make_object(vec![]);
                set_own_prop(&f, "__builtin_name__", s(n), false);
                f
            }
            // Bare global read → the matching `window` property (window is the
            // global object). `has_binding` only routes here when the property
            // exists, so a truly-unknown name still falls through to Undefined.
            n => get_own_prop_value(&self.window, n).unwrap_or(JsValue::Undefined),
        })
    }

    fn call_constructor(
        &self,
        object_name: &str,
        ctx: &mut EvalContext,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        // Bare window globals invoked as functions.
        if is_bare_global(object_name) {
            let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
            let a1 = args.get(1).cloned().unwrap_or(JsValue::Undefined);
            let res = match object_name {
                "scrollTo" | "scroll" => {
                    self.dom.borrow_mut().scroll_to = Some(num_arg(&a1));
                    JsValue::Undefined
                }
                "scrollBy" => {
                    let mut dom = self.dom.borrow_mut();
                    let cur = dom.scroll_to.unwrap_or(0);
                    dom.scroll_to = Some(cur + num_arg(&a1));
                    JsValue::Undefined
                }
                "alert" => {
                    self.dom.borrow_mut().log.push(as_str(&a0));
                    JsValue::Undefined
                }
                "confirm" => JsValue::Boolean(true),
                "prompt" => JsValue::Null,
                // No event loop: run the callback synchronously (best effort).
                "setTimeout" | "setInterval" | "requestAnimationFrame" | "queueMicrotask" => {
                    if value_is_callable(&a0) {
                        let _ = call_value(&a0, JsValue::Undefined, vec![], ctx);
                    }
                    num(0)
                }
                "clearTimeout" | "clearInterval" | "cancelAnimationFrame" => JsValue::Undefined,
                "encodeURIComponent" => s(&percent_encode(&as_str(&a0), "-_.!~*'()")),
                "encodeURI" => s(&percent_encode(&as_str(&a0), "-_.!~*'();,/?:@&=+$#")),
                "decodeURIComponent" | "decodeURI" => s(&percent_decode(&as_str(&a0))),
                _ => JsValue::Undefined,
            };
            return Some(Ok(res));
        }
        // `fetch(url, opts)` called as a bare function → record + fetch + Promise.
        if object_name == "fetch" {
            let (method, url, body) = self.fetch_args(&args);
            self.dom.borrow_mut().fetch_log.push((method.clone(), url.clone(), body.clone()));
            let text = super::js::host_fetch(&method, &url, &body);
            return Some(Ok(promise::resolve_value(response_wrapper(&text))));
        }
        // Bare `postMessage(data, targetOrigin)` == `window.postMessage` (self).
        if object_name == "postMessage" {
            let data = as_str(args.get(0).unwrap_or(&JsValue::Undefined));
            let target_origin = as_str(args.get(1).unwrap_or(&JsValue::Undefined));
            let origin = self.dom.borrow().location_href.clone();
            self.dom.borrow_mut().outbound_messages.push(super::js::Message {
                data,
                origin,
                target_origin,
                target: "self".to_string(),
            });
            return Some(Ok(JsValue::Undefined));
        }
        None
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        _ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        let a1 = args.get(1).cloned().unwrap_or(JsValue::Undefined);
        let res = match (object_name, method_name) {
            ("document", "getElementById") => {
                let id = as_str(&a0);
                match self.dom.borrow().elements.iter().position(|e| e.id.as_deref() == Some(id.as_str())) {
                    Some(i) => elem_wrapper(i),
                    None => JsValue::Null,
                }
            }
            ("document", "querySelector") => {
                let sel = as_str(&a0);
                match self.dom.borrow().elements.iter().position(|e| matches_selector(e, &sel)) {
                    Some(i) => elem_wrapper(i),
                    None => JsValue::Null,
                }
            }
            ("document", "querySelectorAll")
            | ("document", "getElementsByTagName")
            | ("document", "getElementsByClassName") => {
                let sel = as_str(&a0);
                let sel = if method_name == "getElementsByClassName" {
                    format!(".{}", sel)
                } else if method_name == "getElementsByTagName" {
                    sel.to_ascii_lowercase()
                } else {
                    sel
                };
                let idxs: Vec<usize> = {
                    let dom = self.dom.borrow();
                    dom.elements.iter().enumerate()
                        .filter(|(_, e)| sel == "*" || matches_selector(e, &sel))
                        .map(|(i, _)| i).collect()
                };
                make_array(idxs.into_iter().map(elem_wrapper).collect())
            }
            ("document", "createElement") => {
                let tag = as_str(&a0);
                let mut dom = self.dom.borrow_mut();
                dom.elements.push(empty_elem(&tag));
                elem_wrapper(dom.elements.len() - 1)
            }
            ("document", "createTextNode") => {
                let mut dom = self.dom.borrow_mut();
                let mut e = empty_elem("#text");
                e.text = as_str(&a0);
                dom.elements.push(e);
                elem_wrapper(dom.elements.len() - 1)
            }
            ("Element", "appendChild") | ("Element", "insertBefore") => {
                if let (Some(p), Some(c)) = (self.node_of(&this), self.node_of(&a0)) {
                    let mut dom = self.dom.borrow_mut();
                    if p < dom.elements.len() && c < dom.elements.len() && p != c {
                        dom.elements[p].children.retain(|&x| x != c);
                        dom.elements[p].children.push(c);
                        dom.elements[c].parent = Some(p);
                    }
                }
                a0
            }
            ("Element", "removeChild") => {
                if let (Some(p), Some(c)) = (self.node_of(&this), self.node_of(&a0)) {
                    let mut dom = self.dom.borrow_mut();
                    if p < dom.elements.len() {
                        dom.elements[p].children.retain(|&x| x != c);
                    }
                    if c < dom.elements.len() {
                        dom.elements[c].parent = None;
                    }
                }
                a0
            }
            ("Element", "remove") => {
                if let Some(c) = self.node_of(&this) {
                    let mut dom = self.dom.borrow_mut();
                    if let Some(p) = dom.elements.get(c).and_then(|e| e.parent) {
                        dom.elements[p].children.retain(|&x| x != c);
                    }
                    if let Some(e) = dom.elements.get_mut(c) {
                        e.parent = None;
                    }
                }
                JsValue::Undefined
            }
            ("Element", "setAttribute") => {
                if let Some(i) = self.node_of(&this) {
                    let (k, v) = (as_str(&a0), as_str(&a1));
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        match k.as_str() {
                            "id" => e.id = Some(v),
                            "class" => e.class = Some(v),
                            "href" => e.href = v,
                            "src" => e.src = v,
                            _ => { e.attrs.insert(k, v); }
                        }
                    }
                }
                JsValue::Undefined
            }
            ("Element", "getAttribute") => {
                let k = as_str(&a0);
                if let Some(i) = self.node_of(&this) {
                    let dom = self.dom.borrow();
                    if let Some(e) = dom.elements.get(i) {
                        return Some(Ok(match k.as_str() {
                            "id" => e.id.as_deref().map(s).unwrap_or(JsValue::Null),
                            "class" => e.class.as_deref().map(s).unwrap_or(JsValue::Null),
                            _ => e.attrs.get(&k).map(|v| s(v)).unwrap_or(JsValue::Null),
                        }));
                    }
                }
                JsValue::Null
            }
            ("Element", "hasAttribute") => {
                let k = as_str(&a0);
                let has = self.node_of(&this).map_or(false, |i| {
                    let dom = self.dom.borrow();
                    dom.elements.get(i).map_or(false, |e| match k.as_str() {
                        "id" => e.id.is_some(),
                        "class" => e.class.is_some(),
                        _ => e.attrs.contains_key(&k),
                    })
                });
                JsValue::Boolean(has)
            }
            ("Element", "removeAttribute") => {
                if let Some(i) = self.node_of(&this) {
                    let k = as_str(&a0);
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        e.attrs.remove(&k);
                    }
                }
                JsValue::Undefined
            }
            ("Element", "addEventListener") => {
                if let Some(i) = self.node_of(&this) {
                    // Store the CALLBACK for real dispatch (JsPage), and keep
                    // the name-key marker used by the interactive-element set.
                    self.register_listener(i as i64, &args);
                    let ev = as_str(&a0);
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        e.listeners.entry(ev).or_default();
                    }
                }
                JsValue::Undefined
            }
            ("Element", "removeEventListener") => {
                if let Some(i) = self.node_of(&this) {
                    self.unregister_listener(i as i64, &args);
                }
                JsValue::Undefined
            }
            ("Event", "preventDefault") => {
                set_own_prop(&this, "__default_prevented__", JsValue::Boolean(true), false);
                JsValue::Undefined
            }
            ("Event", "stopPropagation") | ("Event", "stopImmediatePropagation") => {
                set_own_prop(&this, "__stopped__", JsValue::Boolean(true), false);
                JsValue::Undefined
            }
            ("document", "addEventListener") => {
                self.register_listener(DOC_NODE, &args);
                JsValue::Undefined
            }
            ("document", "removeEventListener") => {
                self.unregister_listener(DOC_NODE, &args);
                JsValue::Undefined
            }
            ("Element", "cloneNode") => {
                if let Some(i) = self.node_of(&this) {
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get(i).cloned() {
                        dom.elements.push(e);
                        return Some(Ok(elem_wrapper(dom.elements.len() - 1)));
                    }
                }
                JsValue::Null
            }
            ("Element", "contains") => {
                let target = self.node_of(&a0);
                let root = self.node_of(&this);
                let mut found = false;
                if let (Some(r), Some(t)) = (root, target) {
                    let dom = self.dom.borrow();
                    // walk up from target to root
                    let mut cur = Some(t);
                    while let Some(c) = cur {
                        if c == r {
                            found = true;
                            break;
                        }
                        cur = dom.elements.get(c).and_then(|e| e.parent);
                    }
                }
                JsValue::Boolean(found)
            }
            ("Element", "focus") | ("Element", "blur") | ("Element", "click")
            | ("Element", "scrollIntoView") => JsValue::Undefined,
            ("Element", "getContext") => {
                match self.node_of(&this) {
                    Some(i) => {
                        self.dom.borrow_mut().ensure_canvas(i);
                        canvas_ctx_wrapper(i)
                    }
                    None => JsValue::Null,
                }
            }
            ("ClassList", "contains") => {
                let cls = as_str(&a0);
                let has = self.node_of(&this).map_or(false, |i| {
                    let dom = self.dom.borrow();
                    dom.elements.get(i).map_or(false, |e| {
                        e.class.as_deref().map_or(false, |c| c.split_whitespace().any(|w| w == cls))
                    })
                });
                JsValue::Boolean(has)
            }
            ("ClassList", "add") | ("ClassList", "remove") | ("ClassList", "toggle") => {
                if let Some(i) = self.node_of(&this) {
                    let cls = as_str(&a0);
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        let mut set: Vec<String> = e.class.as_deref().unwrap_or("").split_whitespace().map(|x| x.to_string()).collect();
                        let present = set.iter().any(|x| x == &cls);
                        match method_name {
                            "add" => { if !present { set.push(cls); } }
                            "remove" => set.retain(|x| x != &cls),
                            _ => { if present { set.retain(|x| x != &cls); } else { set.push(cls); } }
                        }
                        e.class = Some(set.join(" "));
                    }
                }
                JsValue::Undefined
            }
            ("Style", "setProperty") => {
                if let Some(i) = self.node_of(&this) {
                    let (k, v) = (as_str(&a0), as_str(&a1));
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        e.style = style_set(&e.style, &k, &v);
                    }
                }
                JsValue::Undefined
            }
            ("Style", "getPropertyValue") => {
                let k = as_str(&a0);
                let v = self.node_of(&this).map(|i| {
                    let dom = self.dom.borrow();
                    dom.elements.get(i).map(|e| style_get(&e.style, &k)).unwrap_or_default()
                }).unwrap_or_default();
                s(&v)
            }
            ("Style", "removeProperty") => {
                if let Some(i) = self.node_of(&this) {
                    let k = as_str(&a0);
                    let mut dom = self.dom.borrow_mut();
                    if let Some(e) = dom.elements.get_mut(i) {
                        e.style = style_set(&e.style, &k, "");
                    }
                }
                JsValue::Undefined
            }
            ("window", "addEventListener") => {
                self.register_listener(WIN_NODE, &args);
                JsValue::Undefined
            }
            ("window", "removeEventListener") => {
                self.unregister_listener(WIN_NODE, &args);
                JsValue::Undefined
            }
            ("window", "scrollTo") | ("window", "requestAnimationFrame") => JsValue::Undefined,
            // High-resolution time: the monotonic kernel clock in ms.
            ("performance", "now") => num(crate::arch::now_ms() as i64),
            ("performance", "getEntriesByType") | ("performance", "getEntries")
            | ("performance", "getEntriesByName") => make_array(vec![]),
            ("performance", "mark") | ("performance", "measure")
            | ("performance", "clearMarks") | ("performance", "clearMeasures")
            | ("performance", "clearResourceTimings") | ("performance", "setResourceTimingBufferSize") => {
                JsValue::Undefined
            }
            ("window", "alert") => {
                self.dom.borrow_mut().log.push(as_str(&a0));
                JsValue::Undefined
            }
            ("window", "getComputedStyle") => {
                match self.node_of(&a0) {
                    Some(i) => style_wrapper(i),
                    None => make_object(vec![]),
                }
            }
            ("localStorage" | "sessionStorage", "getItem") => {
                let store = if object_name == "sessionStorage" { &self.session } else { &self.storage };
                get_own_prop_value(store, &as_str(&a0)).unwrap_or(JsValue::Null)
            }
            ("localStorage" | "sessionStorage", "setItem") => {
                let store = if object_name == "sessionStorage" { &self.session } else { &self.storage };
                set_own_prop(store, &as_str(&a0), a1, true);
                JsValue::Undefined
            }
            ("localStorage" | "sessionStorage", "removeItem") => {
                let store = if object_name == "sessionStorage" { &self.session } else { &self.storage };
                set_own_prop(store, &as_str(&a0), JsValue::Null, true);
                JsValue::Undefined
            }

            // --- Canvas 2D context ---
            ("Canvas2d", m) => {
                if let Some(i) = self.node_of(&this) {
                    let mut dom = self.dom.borrow_mut();
                    let c = dom.ensure_canvas(i);
                    match m {
                        "fillRect" => c.fill_rect(num_arg(&a0), num_arg(&a1), num_arg(args.get(2).unwrap_or(&JsValue::Undefined)), num_arg(args.get(3).unwrap_or(&JsValue::Undefined))),
                        "strokeRect" => c.stroke_rect(num_arg(&a0), num_arg(&a1), num_arg(args.get(2).unwrap_or(&JsValue::Undefined)), num_arg(args.get(3).unwrap_or(&JsValue::Undefined))),
                        "clearRect" => c.clear_rect(num_arg(&a0), num_arg(&a1), num_arg(args.get(2).unwrap_or(&JsValue::Undefined)), num_arg(args.get(3).unwrap_or(&JsValue::Undefined))),
                        "fillText" | "strokeText" => c.fill_text(&as_str(&a0), num_arg(&a1), num_arg(args.get(2).unwrap_or(&JsValue::Undefined))),
                        "beginPath" => c.begin_path(),
                        "closePath" => c.close_path(),
                        "moveTo" => c.move_to(fnum_arg(&a0), fnum_arg(&a1)),
                        "lineTo" => c.line_to(fnum_arg(&a0), fnum_arg(&a1)),
                        "stroke" => c.stroke(),
                        "fill" => c.fill(),
                        "arc" => c.arc(fnum_arg(&a0), fnum_arg(&a1), fnum_arg(args.get(2).unwrap_or(&JsValue::Undefined)), fnum_arg(args.get(3).unwrap_or(&JsValue::Undefined)), fnum_arg(args.get(4).unwrap_or(&JsValue::Undefined))),
                        "save" | "restore" | "translate" | "rotate" | "scale" | "setTransform" | "rect" => {}
                        _ => {}
                    }
                }
                JsValue::Undefined
            }

            // --- fetch Response ---
            ("Response", "text") => {
                let body = get_own_prop_value(&this, "__body__").map(|v| as_str(&v)).unwrap_or_default();
                promise::resolve_value(s(&body))
            }
            ("Response", "json") => {
                let body = get_own_prop_value(&this, "__body__").map(|v| as_str(&v)).unwrap_or_default();
                match just_engine::runner::std_lib::json::parse_str(&body) {
                    Ok(v) => promise::resolve_value(v),
                    Err(e) => promise::reject_value(s(&alloc::format!("{:?}", e))),
                }
            }

            // --- postMessage (window/self/parent/top) ---
            ("window" | "parent" | "self" | "top", "postMessage") => {
                let target = if object_name == "parent" || object_name == "top" { "parent" } else { "self" };
                let data = as_str(&a0);
                let target_origin = as_str(&a1);
                let origin = self.dom.borrow().location_href.clone();
                self.dom.borrow_mut().outbound_messages.push(super::js::Message {
                    data,
                    origin,
                    target_origin,
                    target: target.to_string(),
                });
                JsValue::Undefined
            }

            // --- WebAssembly (stub: compiles/instantiates to empty exports) ---
            ("WebAssembly", "instantiate") | ("WebAssembly", "instantiateStreaming") => {
                let instance = make_object(vec![]);
                set_own_prop(&instance, "exports", make_object(vec![]), true);
                let result = make_object(vec![]);
                set_own_prop(&result, "instance", instance, true);
                set_own_prop(&result, "module", make_object(vec![]), true);
                promise::resolve_value(result)
            }
            ("WebAssembly", "compile") | ("WebAssembly", "compileStreaming") => {
                promise::resolve_value(make_object(vec![]))
            }
            ("WebAssembly", "validate") => JsValue::Boolean(true),

            _ => return None,
        };
        Some(Ok(res))
    }

    fn name(&self) -> &str {
        "chitti_dom"
    }
}

/// Run DOM-touching scripts on `just` with LIVE DOM bindings. Returns `false`
/// (DOM untouched) if any script fails to parse, so the caller can fall back to
/// the legacy engine; parse happens before any mutation.
pub fn run_scripts_via_just(dom: &mut JsDom, scripts: &[String]) -> bool {
    let mut asts = Vec::with_capacity(scripts.len());
    for src in scripts {
        match JsParser::parse_to_ast_from_str(src) {
            Ok(ast) => asts.push(ast),
            Err(_) => return false,
        }
    }
    let _ = drain_console_log();

    let placeholder = JsDom::from_document(&super::html::parse(""));
    let taken = core::mem::replace(dom, placeholder);
    let shared = Rc::new(RefCell::new(taken));

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.native_props = Some(Rc::new(DomProps { dom: shared.clone() }));
    ctx.add_resolver(alloc::boxed::Box::new(DomResolver::new(shared.clone())));

    let mut errors: alloc::vec::Vec<String> = alloc::vec::Vec::new();
    for ast in &asts {
        just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut ctx);
        for stmt in &ast.body {
            if let Err(e) = execute_statement(stmt, &mut ctx) {
                // Surface the runtime error (it lands in dom.log → serial) —
                // silently swallowing it made page-script failures invisible.
                errors.push(alloc::format!("Uncaught {}", e.to_string()));
                break;
            }
        }
    }
    drop(ctx); // releases the resolver + native_props Rc clones of `shared`

    match Rc::try_unwrap(shared) {
        Ok(cell) => *dom = cell.into_inner(),
        Err(_) => return false,
    }
    for line in drain_console_log() {
        dom.log.push(line);
    }
    dom.log.append(&mut errors);
    true
}

// ============================================================================
// Persistent page context — JsPage
//
// The one-shot `run_scripts_via_just` re-executes every page script on every
// re-layout and cannot keep closures alive (they hold `*const` pointers into
// the run-local ASTs). `JsPage` keeps the whole JS world — the parsed ASTs,
// the `EvalContext` (every closure + captured environment), the listener
// registry, and the shared `JsDom` — alive for the page lifetime, so:
//   • scripts run ONCE per navigation (repaints just re-read the DOM),
//   • `addEventListener` callbacks can be invoked on real UI events,
//   • DOM state (counters, toggles, created elements) persists across clicks.
// ============================================================================

/// The persistent page JS context. **Field order is load-bearing**: Rust drops
/// fields in declaration order, and `ctx` (which transitively owns every
/// closure) must drop before `asts`/`attr_asts` — `SimpleFunctionObject`s hold
/// `*const` pointers into the AST heap those vectors own.
///
/// SAFETY (pointer validity): the pointers target heap allocations *inside*
/// `ProgramData` (statement `Vec` buffers and `Box`ed bodies), which do not
/// move when the owning `ProgramData` values are moved (e.g. by a `Vec` push
/// realloc). The ASTs are never mutated after parse.
pub struct JsPage {
    ctx: EvalContext,
    listeners: ListenerReg,
    shared: Rc<RefCell<JsDom>>,
    asts: Vec<just_engine::parser::ast::ProgramData>,
    /// Mini-ASTs parsed at dispatch time for inline `on*` attribute source —
    /// kept for the page lifetime (handlers may create closures into them).
    attr_asts: Vec<just_engine::parser::ast::ProgramData>,
}

/// The live page. SAFETY (`Sync`): `mm::Locked` is unconditionally `Sync`; the
/// page is only ever created/dispatched/dropped from the shell task (the
/// single-threaded UI loop), and `.with()` serializes access.
static JS_PAGE: crate::mm::Locked<Option<JsPage>> = crate::mm::Locked::new(None);

/// A UI event to deliver into page JS.
pub struct PageEvent {
    /// Element index in `JsDom.elements` (the click target).
    pub target: usize,
    /// Event type ("click", "input", "keydown", "change", "submit").
    pub type_: String,
    pub x: i32,
    pub y: i32,
}

/// Boot the persistent page context: build the DOM from `doc`, run `scripts`
/// (per-script parse tolerance — a script that fails to parse logs `Uncaught
/// SyntaxError` and is skipped, it cannot take the rest of the page down), and
/// keep everything alive for later [`page_dispatch`] calls. Replaces any
/// previous page. Returns the number of scripts that parsed.
pub fn page_boot(
    doc: &super::html::Document,
    location_href: &str,
    inner_w: i32,
    inner_h: i32,
    scripts: &[String],
) -> usize {
    page_close();
    let mut dom = JsDom::from_document(doc);
    if !location_href.is_empty() {
        dom.location_href = location_href.to_string();
    }
    dom.inner_width = inner_w;
    dom.inner_height = inner_h;

    let _ = drain_console_log();
    let shared = Rc::new(RefCell::new(dom));
    let listeners: ListenerReg = Rc::new(RefCell::new(Vec::new()));

    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.native_props = Some(Rc::new(DomProps { dom: shared.clone() }));
    ctx.add_resolver(alloc::boxed::Box::new(DomResolver::with_listeners(
        shared.clone(),
        listeners.clone(),
    )));

    let mut asts = Vec::with_capacity(scripts.len());
    for src in scripts {
        match JsParser::parse_to_ast_from_str(src) {
            Ok(ast) => asts.push(ast),
            Err(e) => {
                let msg = alloc::format!("{:?}", e);
                let msg: String = msg.chars().take(160).collect();
                shared
                    .borrow_mut()
                    .log
                    .push(alloc::format!("Uncaught SyntaxError: {}", msg));
            }
        }
    }
    let parsed = asts.len();

    for ast in &asts {
        just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut ctx);
        for stmt in &ast.body {
            if let Err(e) = execute_statement(stmt, &mut ctx) {
                shared
                    .borrow_mut()
                    .log
                    .push(alloc::format!("Uncaught {}", e.to_string()));
                break;
            }
        }
    }
    for line in drain_console_log() {
        shared.borrow_mut().log.push(line);
    }

    JS_PAGE.with(|slot| {
        *slot = Some(JsPage { ctx, listeners, shared, asts, attr_asts: Vec::new() })
    });
    parsed
}

/// True when a persistent page context is live.
pub fn page_active() -> bool {
    JS_PAGE.with(|slot| slot.is_some())
}

/// Drop the page context (navigation / reload / tab close). Field order in
/// [`JsPage`] guarantees closures die before the ASTs they point into.
pub fn page_close() {
    JS_PAGE.with(|slot| *slot = None);
}

/// Run `f` against the live page DOM (commit/layout/effects reads and writes).
/// Returns `None` when no page is active.
pub fn page_with_dom<R>(f: impl FnOnce(&mut JsDom) -> R) -> Option<R> {
    JS_PAGE.with(|slot| slot.as_mut().map(|p| f(&mut *p.shared.borrow_mut())))
}

/// Element indices that should be hit-testable: everything with a registered
/// listener or an inline `on*` attribute.
pub fn page_interactive_elems() -> alloc::vec::Vec<usize> {
    JS_PAGE.with(|slot| {
        let Some(p) = slot.as_mut() else { return Vec::new() };
        let mut out: Vec<usize> = p
            .listeners
            .borrow()
            .iter()
            .filter(|l| l.target >= 0)
            .map(|l| l.target as usize)
            .collect();
        let dom = p.shared.borrow();
        for (i, e) in dom.elements.iter().enumerate() {
            if !e.listeners.is_empty() || e.attrs.keys().any(|k| k.starts_with("on")) {
                out.push(i);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    })
}

/// Dispatch UI events into page JS: capture (window → document → ancestors)
/// → target (inline `on*` attr first, then listeners) → bubble (ancestors →
/// document → window). Returns one `default_prevented` flag per event.
pub fn page_dispatch(events: &[PageEvent]) -> alloc::vec::Vec<bool> {
    let _ = drain_console_log();
    let out = JS_PAGE.with(|slot| {
        let Some(page) = slot.as_mut() else {
            return events.iter().map(|_| false).collect::<alloc::vec::Vec<bool>>();
        };
        let mut prevented = alloc::vec::Vec::with_capacity(events.len());
        for ev in events {
            prevented.push(dispatch_one(page, ev));
        }
        prevented
    });
    // Console output produced by handlers lands in the page log.
    let lines = drain_console_log();
    if !lines.is_empty() {
        JS_PAGE.with(|slot| {
            if let Some(p) = slot.as_mut() {
                p.shared.borrow_mut().log.extend(lines);
            }
        });
    }
    out
}

fn dispatch_one(page: &mut JsPage, ev: &PageEvent) -> bool {
    // Event object: preventDefault/stopPropagation route through the
    // DomResolver's ("Event", …) method arms and set marker props.
    let eobj = make_object(vec![]);
    set_own_prop(&eobj, "__builtin_name__", s("Event"), false);
    set_own_prop(&eobj, "type", s(&ev.type_), true);
    set_own_prop(&eobj, "clientX", num(ev.x as i64), true);
    set_own_prop(&eobj, "clientY", num(ev.y as i64), true);
    let target_w = elem_wrapper(ev.target);
    set_own_prop(&eobj, "target", target_w.clone(), true);
    set_own_prop(&eobj, "currentTarget", target_w.clone(), true);

    // Propagation path root→target: window, document, ancestors, target.
    // Snapshot it (and everything else we need) BEFORE running JS — handler
    // code re-enters the resolver, which borrows the same RefCell<JsDom>.
    let mut chain: Vec<i64> = vec![WIN_NODE, DOC_NODE];
    {
        let dom = page.shared.borrow();
        let mut anc: Vec<i64> = Vec::new();
        let mut cur = dom.elements.get(ev.target).and_then(|e| e.parent);
        while let Some(p) = cur {
            anc.push(p as i64);
            if anc.len() > 64 {
                break; // cycle guard
            }
            cur = dom.elements.get(p).and_then(|e| e.parent);
        }
        anc.reverse();
        chain.extend(anc);
    }

    let stopped = |eobj: &JsValue| {
        get_own_prop_value(eobj, "__stopped__")
            .map(|v| to_boolean(&v))
            .unwrap_or(false)
    };

    // Capture phase: root → target's parent, capture listeners only.
    for &t in &chain {
        if stopped(&eobj) {
            break;
        }
        invoke_listeners(page, t, &ev.type_, &eobj, true);
    }

    // Target phase: inline `on<type>` attribute source first, then listeners
    // (both capture and bubble registrations fire at the target).
    if !stopped(&eobj) {
        run_on_attr(page, ev.target, &ev.type_, &eobj);
        invoke_listeners(page, ev.target as i64, &ev.type_, &eobj, true);
        invoke_listeners(page, ev.target as i64, &ev.type_, &eobj, false);
    }

    // Bubble phase: target's parent → root, bubble listeners only.
    for &t in chain.iter().rev() {
        if stopped(&eobj) {
            break;
        }
        invoke_listeners(page, t, &ev.type_, &eobj, false);
    }

    get_own_prop_value(&eobj, "__default_prevented__")
        .map(|v| to_boolean(&v))
        .unwrap_or(false)
}

/// Invoke registered listeners for (target, type, phase). Snapshots matching
/// callbacks first — a handler may add/remove listeners while running.
fn invoke_listeners(page: &mut JsPage, target: i64, type_: &str, eobj: &JsValue, capture: bool) {
    let cbs: Vec<JsValue> = page
        .listeners
        .borrow()
        .iter()
        .filter(|l| l.target == target && l.type_ == type_ && l.capture == capture)
        .map(|l| l.cb.clone())
        .collect();
    if cbs.is_empty() {
        return;
    }
    let this = if target >= 0 {
        elem_wrapper(target as usize)
    } else {
        global_wrapper(if target == DOC_NODE { "document" } else { "window" }, target)
    };
    set_own_prop(eobj, "currentTarget", this.clone(), true);
    for cb in cbs {
        if get_own_prop_value(eobj, "__stopped__").map(|v| to_boolean(&v)).unwrap_or(false) {
            break;
        }
        if let Err(e) = call_value(&cb, this.clone(), vec![eobj.clone()], &mut page.ctx) {
            page.shared
                .borrow_mut()
                .log
                .push(alloc::format!("Uncaught {}", e.to_string()));
        }
    }
}

/// Run an element's inline `on<type>` attribute source (e.g.
/// `onclick="count++; render()"`) with `this` bound to the element and
/// `event` in scope. The parsed mini-AST is kept alive on the page (handlers
/// may create closures into it).
fn run_on_attr(page: &mut JsPage, target: usize, type_: &str, eobj: &JsValue) {
    let src = {
        let dom = page.shared.borrow();
        let key = alloc::format!("on{}", type_);
        dom.elements.get(target).and_then(|e| e.attrs.get(&key).cloned())
    };
    let Some(src) = src else { return };
    let ast = match JsParser::parse_to_ast_from_str(&src) {
        Ok(a) => a,
        Err(e) => {
            let msg = alloc::format!("{:?}", e);
            let msg: String = msg.chars().take(120).collect();
            page.shared
                .borrow_mut()
                .log
                .push(alloc::format!("Uncaught SyntaxError (on{}): {}", type_, msg));
            return;
        }
    };
    page.attr_asts.push(ast);
    // Re-borrow the freshly pushed AST for execution (kept alive on the page).
    let n_attr = page.attr_asts.len() - 1;

    // `this` = the element; `event` bound in a fresh block scope.
    let saved_this = page.ctx.global_this.clone();
    page.ctx.global_this = Some(elem_wrapper(target));
    page.ctx.push_block_scope();
    let _ = page.ctx.create_binding("event", false);
    let _ = page.ctx.initialize_binding("event", eobj.clone());
    let body: Vec<_> = page.attr_asts[n_attr].body.iter().collect();
    for stmt in body {
        if let Err(e) = execute_statement(stmt, &mut page.ctx) {
            page.shared
                .borrow_mut()
                .log
                .push(alloc::format!("Uncaught {}", e.to_string()));
            break;
        }
    }
    page.ctx.pop_block_scope();
    page.ctx.global_this = saved_this;
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
    fn new_expression_method_chaining() {
        // `new X(args).m()` — member/call directly on a `new` expression.
        assert_eq!(
            val("class A { constructor(x){ this.x = x; } g(){ return this.x * 2; } } new A(21).g()"),
            "42"
        );
        assert_eq!(val("class P { constructor(){ this.v = 7; } } new P().v"), "7");
    }

    #[test_case]
    fn async_expressions_and_arrows() {
        // async function expression returns a Promise usable with await + .then
        assert_eq!(val("let f = async function(){ return 9; }; await f()"), "9");
        assert_eq!(val("let f = async function(){ return 3; }; let o=0; f().then(v=>{o=v*2;}); o"), "6");
        // async arrow (concise + block body) — await and .then both work
        assert_eq!(val("let f = async () => 41; await f() + 1"), "42");
        assert_eq!(val("let f = async () => 5; let o=0; f().then(v=>{o=v*2;}); o"), "10");
        assert_eq!(val("let g = async (a,b) => { return a+b; }; await g(4,5)"), "9");
        // `async` still usable as an identifier / function name (no regression)
        assert_eq!(val("let async = 7; async + 1"), "8");
    }

    #[test_case]
    fn response_json_parses() {
        // `Response.json()` really parses the body (host_fetch returns
        // {"ok":true,"url":"…"} under cfg(test)).
        use crate::browser::{html, js};
        let doc = html::parse("<html><body></body></html>");
        let mut dom = js::JsDom::from_document(&doc);
        let _ = js::run_scripts(
            &mut dom,
            &[String::from(
                "var out=''; fetch('https://ex.com/api').then(function(r){ return r.json(); }).then(function(j){ out = j.ok; }); console.log(out);",
            )],
        );
        assert!(dom.log.iter().any(|l| l == "true"), "json().ok should be true: {:?}", dom.log);
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
    fn dom_primary_tier_rich_surface() {
        // Drives the PRIMARY path (`js::run_scripts` routes DOM scripts to the
        // just DOM tier): style get/set, classList, querySelectorAll on parsed
        // elements. (childElementCount/children are only tracked for JS-built
        // trees via appendChild — see `js::tests::dom_create_append_query`.)
        use crate::browser::{html, js};
        let doc = html::parse(
            r#"<html><body><div id="box" class="a"></div><span class="x">1</span><span class="x">2</span></body></html>"#,
        );
        let mut dom = js::JsDom::from_document(&doc);
        let logs = js::run_scripts(
            &mut dom,
            &[String::from(
                r##"
                var box = document.getElementById("box");
                box.style.color = "red";
                box.classList.add("active");
                console.log(box.style.color);
                console.log(box.className);
                console.log(document.querySelectorAll(".x").length);
                "##,
            )],
        );
        assert!(logs.iter().any(|l| l == "red"), "style.color: {logs:?}");
        assert!(logs.iter().any(|l| l.contains("active") && l.contains("a")), "classList.add: {logs:?}");
        assert!(logs.iter().any(|l| l == "2"), "querySelectorAll: {logs:?}");
        // Live: writes reflected in the JsDom immediately.
        let b = dom.elements.iter().find(|e| e.id.as_deref() == Some("box")).unwrap();
        assert!(b.style.contains("color") && b.style.contains("red"), "{}", b.style);
        assert!(b.class.as_deref().unwrap_or("").contains("active"), "{:?}", b.class);
    }

    #[test_case]
    fn dom_canvas_2d() {
        use crate::browser::{html, js};
        let doc = html::parse(r#"<html><body><canvas id="c"></canvas></body></html>"#);
        let mut dom = js::JsDom::from_document(&doc);
        let ok = js::run_scripts(
            &mut dom,
            &[String::from(
                "var c = document.getElementById('c'); var ctx = c.getContext('2d'); ctx.fillStyle = 'red'; ctx.fillRect(0, 0, 10, 10); ctx.beginPath();",
            )],
        );
        let _ = ok;
        assert!(!dom.canvases.is_empty(), "getContext should create a canvas");
    }

    #[test_case]
    fn dom_fetch_logs_and_resolves() {
        use crate::browser::{html, js};
        let doc = html::parse("<html><body></body></html>");
        let mut dom = js::JsDom::from_document(&doc);
        let _ = js::run_scripts(
            &mut dom,
            &[String::from(
                "var out=''; fetch('https://ex.com/api').then(function(r){ return r.text(); }).then(function(t){ out = t; }); console.log(out);",
            )],
        );
        assert!(!dom.fetch_log.is_empty(), "fetch should record");
        assert_eq!(dom.fetch_log[0].1, "https://ex.com/api");
        // synchronous settlement: the .then chain ran, so `out` was logged non-empty
        assert!(dom.log.iter().any(|l| l.contains("ok")), "{:?}", dom.log);
    }

    #[test_case]
    fn dom_post_message() {
        use crate::browser::{html, js};
        let doc = html::parse("<html><body></body></html>");
        let mut dom = js::JsDom::from_document(&doc);
        let _ = js::run_scripts(&mut dom, &[String::from("window.postMessage('hi', '*');")]);
        assert_eq!(dom.outbound_messages.len(), 1);
        assert_eq!(dom.outbound_messages[0].data, "hi");
    }

    #[test_case]
    fn dom_webassembly_stub() {
        use crate::browser::{html, js};
        let doc = html::parse("<html><body></body></html>");
        let mut dom = js::JsDom::from_document(&doc);
        let logs = js::run_scripts(
            &mut dom,
            &[String::from(
                "var ok = 0; WebAssembly.instantiate('bytes').then(function(m){ ok = m.instance ? 1 : 0; }); console.log(ok);",
            )],
        );
        assert!(logs.iter().any(|l| l == "1"), "WebAssembly.instantiate should resolve: {logs:?}");
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

    // ── Persistent page context (JsPage) ────────────────────────────────

    fn idx_of(id: &str) -> usize {
        page_with_dom(|dom| {
            dom.elements
                .iter()
                .position(|e| e.id.as_deref() == Some(id))
                .expect("element present")
        })
        .expect("page active")
    }

    fn click(target: usize) -> bool {
        page_dispatch(&[PageEvent { target, type_: String::from("click"), x: 1, y: 1 }])
            .first()
            .copied()
            .unwrap_or(false)
    }

    #[test_case]
    fn page_onclick_attr_runs_at_target() {
        use crate::browser::html;
        let doc = html::parse(
            r#"<html><body><button id="b" onclick="document.getElementById('out').innerText = 'clicked'">go</button><div id="out">x</div></body></html>"#,
        );
        page_boot(&doc, "https://t.example/", 640, 400, &[]);
        let b = idx_of("b");
        click(b);
        let out = page_with_dom(|dom| {
            dom.elements
                .iter()
                .find(|e| e.id.as_deref() == Some("out"))
                .map(|e| e.text.clone())
                .unwrap_or_default()
        })
        .unwrap();
        assert_eq!(out, "clicked");
        page_close();
    }

    #[test_case]
    fn page_listener_state_persists_across_clicks() {
        // The flagship persistent-context property: a closure's captured
        // counter survives between dispatches (impossible with the old
        // stateless re-run model).
        use crate::browser::html;
        let doc = html::parse(
            r#"<html><body><button id="b">go</button><div id="out">0</div></body></html>"#,
        );
        page_boot(
            &doc,
            "https://t.example/",
            640,
            400,
            &[String::from(
                "var n = 0; document.getElementById('b').addEventListener('click', function (e) { n = n + 1; document.getElementById('out').innerText = 'n=' + n; });",
            )],
        );
        let b = idx_of("b");
        click(b);
        click(b);
        let out = page_with_dom(|dom| {
            dom.elements
                .iter()
                .find(|e| e.id.as_deref() == Some("out"))
                .map(|e| e.text.clone())
                .unwrap_or_default()
        })
        .unwrap();
        assert_eq!(out, "n=2");
        page_close();
    }

    #[test_case]
    fn page_prevent_default_and_bubble() {
        use crate::browser::html;
        let doc = html::parse(
            r#"<html><body><div id="outer"><a id="lnk" href="/x">l</a></div></body></html>"#,
        );
        page_boot(
            &doc,
            "https://t.example/",
            640,
            400,
            &[String::from(
                "document.getElementById('lnk').addEventListener('click', function (e) { e.preventDefault(); });\
                 document.getElementById('outer').addEventListener('click', function (e) { document.getElementById('outer').innerText = 'bubbled'; });",
            )],
        );
        let lnk = idx_of("lnk");
        // preventDefault reported to the host (suppresses native link follow)…
        assert!(click(lnk), "default should be prevented");
        // …and the event bubbled to the parent listener via the parent links.
        let outer = page_with_dom(|dom| {
            dom.elements
                .iter()
                .find(|e| e.id.as_deref() == Some("outer"))
                .map(|e| e.text.clone())
                .unwrap_or_default()
        })
        .unwrap();
        assert_eq!(outer, "bubbled");
        page_close();
    }

    #[test_case]
    fn page_boot_tolerates_bad_script() {
        // A script that fails to parse logs a SyntaxError and is skipped;
        // the good script still runs (external minified libs must not take
        // down inline page scripts).
        use crate::browser::html;
        let doc = html::parse(r#"<html><body><p id="m">x</p></body></html>"#);
        let parsed = page_boot(
            &doc,
            "https://t.example/",
            640,
            400,
            &[
                String::from("this is not (((( javascript"),
                String::from("document.getElementById('m').innerText = 'ran';"),
            ],
        );
        assert_eq!(parsed, 1, "one of two scripts parses");
        let (text, has_err) = page_with_dom(|dom| {
            let t = dom
                .elements
                .iter()
                .find(|e| e.id.as_deref() == Some("m"))
                .map(|e| e.text.clone())
                .unwrap_or_default();
            let e = dom.log.iter().any(|l| l.contains("SyntaxError"));
            (t, e)
        })
        .unwrap();
        assert_eq!(text, "ran");
        assert!(has_err, "the bad script's SyntaxError is logged");
        page_close();
    }

    #[test_case]
    fn stamp_matches_collect_elems_indices() {
        use crate::browser::{html, js};
        let mut doc = html::parse(
            r#"<html><head><style>p{}</style></head><body><div id="a"><p id="b">t</p></div><span id="c">s</span></body></html>"#,
        );
        let dom = JsDom::from_document(&doc);
        js::stamp_elem_indices(&mut doc.root);
        // Every stamped node's ordinal must agree with collect_elems' index
        // (checked via the id attribute both sides carry).
        fn walk(n: &crate::browser::html::Node, dom: &JsDom) {
            if let crate::browser::html::NodeKind::Element { id: Some(id), .. } = &n.kind {
                let i = n.elem_idx.expect("stamped");
                assert_eq!(dom.elements[i].id.as_deref(), Some(id.as_str()), "idx {i}");
            }
            for c in &n.children {
                walk(c, dom);
            }
        }
        walk(&doc.root, &dom);
    }
}
