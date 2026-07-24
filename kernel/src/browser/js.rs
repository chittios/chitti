//! **JavaScript engine (subset)** — pure no_std interpreter for common page
//! scripts. Not ECMAScript-complete; intentionally small and sandboxed.
//!
//! Ladybird reference: `Libraries/LibJS` + `LibWeb/HTML/Window*` +
//! `LibWeb/Fetch/FetchMethod` — we expose the same host names as a thin subset.
//!
//! Supported:
//! - `var` / `let` / `const` bindings, numbers, strings, booleans, `null`
//! - Binary ops `+ - * / == != < > <= >=`, unary `!` / `typeof` / `+` / `-`
//! - `if` / `else`, `while`, `for` (bounded), blocks, `return` / `throw` / `break`
//! - **User functions** `function f(a){…}`, arrow `a => …`, calls
//! - **try / catch / finally**
//! - **`new`** for Object / Array / RegExp / Error / String / Number / Boolean
//! - **class** (constructor + methods as plain functions)
//! - **BigInt** literals (`1n`) + `BigInt(...)`
//! - **RegExp** literals `/pat/flags` + `.test` / `.exec`
//! - Objects `{a:1}`, arrays `[1,2]`, property get/set
//! - Host objects:
//!   - `console.log(...)` / `alert(...)`
//!   - `document.title` get/set, `document.body`, `getElementById`, `querySelector`
//!   - element `.innerText` / `.textContent` / `.value` / `.style.*` / canvas 2d
//!   - `window` / `self` / `location` / `location.href` / `location.assign`
//!   - `window.scrollTo` / `scrollBy` / `innerWidth` / `innerHeight`
//!   - `fetch(url)` / `fetch(url, {method, body})` — host network when not in unit tests
//!   - `encodeURIComponent` / `decodeURIComponent`
//!
//! Structure follows Ladybird LibJS host objects + WebIDL bindings for Window /
//! Document / Location / Fetch — not a full
//! [MDN JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript) VM.
//!
//! Security: no eval of host code, instruction budget, no prototype pollution.
//! Scripts that fail are skipped (best-effort). `fetch` goes through the
//! browser loader (capability path at the shell); unit tests get a stub body.

use super::html::{Node, NodeKind};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::rc::Rc;

const MAX_STEPS: u32 = 200_000;
const MAX_LOOP: u32 = 10_000;
const MAX_CALL_DEPTH: u32 = 64;

#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Num(f64),
    /// ECMAScript BigInt (bounded to i128 for no_std practicality).
    BigInt(i128),
    Str(String),
    /// Index into the flat element table in [`JsDom`].
    Elem(usize),
    /// Plain object / map.
    Obj(Rc<BTreeMap<String, Val>>),
    /// Array (dense vector).
    Arr(Rc<Vec<Val>>),
    /// User / host function.
    Fun(Rc<FunVal>),
    /// RegExp pattern + flags.
    RegExp(Rc<RegExpVal>),
    Undefined,
}

#[derive(Clone, Debug)]
pub struct FunVal {
    pub name: String,
    pub params: Vec<String>,
    pub body: String,
    /// When true, body is a single expression (arrow).
    pub is_expr: bool,
}

#[derive(Clone, Debug)]
pub struct RegExpVal {
    pub pattern: String,
    pub flags: String,
}

impl Val {
    fn as_bool(&self) -> bool {
        match self {
            Val::Null | Val::Undefined => false,
            Val::Bool(b) => *b,
            Val::Num(n) => *n != 0.0 && !n.is_nan(),
            Val::BigInt(n) => *n != 0,
            Val::Str(s) => !s.is_empty(),
            Val::Elem(_) | Val::Obj(_) | Val::Arr(_) | Val::Fun(_) | Val::RegExp(_) => true,
        }
    }

    fn as_str(&self) -> String {
        match self {
            Val::Null => String::from("null"),
            Val::Undefined => String::from("undefined"),
            Val::Bool(b) => b.to_string(),
            Val::Num(n) => {
                if n.is_nan() {
                    String::from("NaN")
                } else if *n == (*n as i64) as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Val::BigInt(n) => format!("{n}"),
            Val::Str(s) => s.clone(),
            Val::Elem(i) => format!("[Element #{i}]"),
            Val::Obj(_) => String::from("[object Object]"),
            Val::Arr(a) => a
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(","),
            Val::Fun(f) => format!("function {}() {{ [native code] }}", f.name),
            Val::RegExp(r) => format!("/{}/{}", r.pattern, r.flags),
        }
    }

    fn as_num(&self) -> f64 {
        match self {
            Val::Num(n) => *n,
            Val::BigInt(n) => *n as f64,
            Val::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Val::Str(s) => s.parse().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Val::Undefined => "undefined",
            Val::Null => "object",
            Val::Bool(_) => "boolean",
            Val::Num(_) => "number",
            Val::BigInt(_) => "bigint",
            Val::Str(_) => "string",
            Val::Fun(_) => "function",
            Val::Obj(_) | Val::Arr(_) | Val::Elem(_) | Val::RegExp(_) => "object",
        }
    }
}

/// Control-flow signal for return / throw / break.
#[derive(Clone, Debug)]
enum Flow {
    None,
    Return(Val),
    Throw(Val),
    Break,
}

/// Mutable DOM surface the JS engine can touch.
pub struct JsDom {
    pub title: String,
    pub log: Vec<String>,
    /// Flat list of element pointers as indices into a parallel `nodes` vector
    /// of owned element snapshots we mutate via the document tree.
    pub elements: Vec<ElemRef>,
    /// `window.location.href` seed + mutable.
    pub location_href: String,
    pub inner_width: i32,
    pub inner_height: i32,
    /// Host should navigate after scripts if set.
    pub navigate: Option<String>,
    /// Absolute scroll position requested by script.
    pub scroll_to: Option<i32>,
    /// Recorded fetch calls (method, url, body) for diagnostics.
    pub fetch_log: Vec<(String, String, String)>,
    /// `postMessage` outbound queue (Ladybird MessageEvent delivery).
    pub outbound_messages: Vec<Message>,
    /// Inbound messages delivered by the host before/after scripts.
    pub inbound_messages: Vec<Message>,
    /// When true, `parent` resolves to a proxy that posts to parent.
    pub is_nested: bool,
    pub parent_origin: String,
    /// Canvas 2D contexts keyed by element index (`getContext('2d')`).
    pub canvases: alloc::collections::BTreeMap<usize, super::canvas::Canvas2d>,
}

/// One `postMessage` payload (data is stringified).
#[derive(Clone, Debug)]
pub struct Message {
    pub data: String,
    pub origin: String,
    /// Target origin filter (`*` or exact).
    pub target_origin: String,
    /// Logical channel: `parent` | `self` | frame name.
    pub target: String,
}

#[derive(Clone, Debug)]
pub struct ElemRef {
    pub tag: String,
    pub id: Option<String>,
    pub class: Option<String>,
    /// Mutable text content (replaces first text child on commit).
    pub text: String,
    /// Form control value (`input.value`).
    pub value: String,
    /// Inline style overrides (CSS declaration text).
    pub style: String,
    /// Canvas width/height attributes (or defaults).
    pub canvas_w: Option<i32>,
    pub canvas_h: Option<i32>,
    /// HTML attributes (id/class/style mirrored when set).
    pub attrs: BTreeMap<String, String>,
    /// Tree links (indices into JsDom.elements). Detached: parent = None.
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Event listeners: type → handler names / source snippets.
    pub listeners: BTreeMap<String, Vec<String>>,
    /// dataset-* keys without prefix.
    pub dataset: BTreeMap<String, String>,
    pub checked: bool,
    pub disabled: bool,
    pub hidden: bool,
    pub href: String,
    pub src: String,
    pub type_attr: String,
    pub name_attr: String,
    pub placeholder: String,
}

impl JsDom {
    pub fn from_document(doc: &super::html::Document) -> Self {
        let mut elements = Vec::new();
        collect_elems(&doc.root, &mut elements);
        // Parent/child links (same pre-order walk as collect_elems) so event
        // dispatch can bubble target → ancestors → document → window.
        let mut next = 0usize;
        link_parents_walk(&doc.root, None, &mut next, &mut elements);
        JsDom {
            title: doc.title.clone(),
            log: Vec::new(),
            elements,
            location_href: String::from("about:blank"),
            inner_width: 640,
            inner_height: 400,
            navigate: None,
            scroll_to: None,
            fetch_log: Vec::new(),
            outbound_messages: Vec::new(),
            inbound_messages: Vec::new(),
            is_nested: false,
            parent_origin: String::new(),
            canvases: alloc::collections::BTreeMap::new(),
        }
    }

    /// Ensure a canvas 2d buffer exists for element `i` (default 300×150).
    pub(crate) fn ensure_canvas(&mut self, i: usize) -> &mut super::canvas::Canvas2d {
        if !self.canvases.contains_key(&i) {
            let (w, h) = self
                .elements
                .get(i)
                .map(|e| (e.canvas_w.unwrap_or(300), e.canvas_h.unwrap_or(150)))
                .unwrap_or((300, 150));
            self.canvases
                .insert(i, super::canvas::Canvas2d::new(w, h));
        }
        self.canvases.get_mut(&i).unwrap()
    }

    fn find_id(&self, id: &str) -> Option<usize> {
        self.elements.iter().position(|e| e.id.as_deref() == Some(id))
    }

    fn find_class(&self, class: &str) -> Option<usize> {
        self.elements.iter().position(|e| {
            e.class
                .as_deref()
                .map(|c| c.split_whitespace().any(|x| x == class))
                .unwrap_or(false)
        })
    }

    fn find_tag(&self, tag: &str) -> Option<usize> {
        self.elements
            .iter()
            .position(|e| e.tag.eq_ignore_ascii_case(tag))
    }
}

/// Stamp each tree node with its `collect_elems` ordinal (`Node.elem_idx`), so
/// layout can associate boxes with `JsDom.elements` indices without a fragile
/// synced counter — identity travels with the node. Pre-order, the exact
/// `collect_elems` visit rule. Nodes that already carry an index (inserted for
/// JS-created elements by [`commit_to_tree`]) keep it, and still consume their
/// ordinal slot so following siblings stay aligned.
pub fn stamp_elem_indices(root: &mut Node) {
    let mut next = 0usize;
    stamp_walk(root, &mut next);
}

fn stamp_walk(n: &mut Node, next: &mut usize) {
    if let NodeKind::Element { tag, .. } = &n.kind {
        if !matches!(tag.as_str(), "script" | "style" | "noscript") {
            // Only PARSED nodes consume ordinals — nodes pre-stamped by the
            // created-element insertion pass sit outside the parse order, and
            // counting them would shift every parsed element after them.
            if n.elem_idx.is_none() {
                n.elem_idx = Some(*next);
                *next += 1;
            }
        }
    }
    for c in &mut n.children {
        stamp_walk(c, next);
    }
}

/// Populate `ElemRef.parent`/`children` links by walking the tree with the
/// exact `collect_elems` visit rule (pre-order, skipping script/style/
/// noscript), assigning the same indices `collect_elems` produced.
fn link_parents_walk(
    n: &Node,
    parent: Option<usize>,
    next: &mut usize,
    elements: &mut [ElemRef],
) {
    let mut my_idx = parent;
    if let NodeKind::Element { tag, .. } = &n.kind {
        if !matches!(tag.as_str(), "script" | "style" | "noscript") {
            let i = *next;
            *next += 1;
            if let Some(e) = elements.get_mut(i) {
                e.parent = parent;
            }
            if let Some(p) = parent {
                if let Some(pe) = elements.get_mut(p) {
                    pe.children.push(i);
                }
            }
            my_idx = Some(i);
        }
    }
    for c in &n.children {
        link_parents_walk(c, my_idx, next, elements);
    }
}

fn collect_elems(n: &Node, out: &mut Vec<ElemRef>) {
    if let NodeKind::Element {
        tag,
        id,
        class,
        style_attr,
        value,
        width_attr,
        height_attr,
        on_attrs,
        ..
    } = &n.kind
    {
        if !matches!(tag.as_str(), "script" | "style" | "noscript") {
            let text = super::html::collect_text(n);
            let val = value.clone().unwrap_or_else(|| {
                if matches!(tag.as_str(), "input" | "textarea" | "button") {
                    text.clone()
                } else {
                    String::new()
                }
            });
            let mut attrs = BTreeMap::new();
            if let Some(i) = id {
                attrs.insert(String::from("id"), i.clone());
            }
            if let Some(c) = class {
                attrs.insert(String::from("class"), c.clone());
            }
            if let Some(s) = style_attr {
                attrs.insert(String::from("style"), s.clone());
            }
            // Inline event-handler attributes (`onclick="…"`) — dispatch runs
            // these at the target phase.
            for (k, v) in on_attrs {
                attrs.insert(k.clone(), v.clone());
            }
            out.push(ElemRef {
                tag: tag.clone(),
                id: id.clone(),
                class: class.clone(),
                text,
                value: val,
                style: style_attr.clone().unwrap_or_default(),
                canvas_w: if tag.eq_ignore_ascii_case("canvas") {
                    *width_attr
                } else {
                    None
                },
                canvas_h: if tag.eq_ignore_ascii_case("canvas") {
                    *height_attr
                } else {
                    None
                },
                attrs,
                parent: None,
                children: Vec::new(),
                listeners: BTreeMap::new(),
                dataset: BTreeMap::new(),
                checked: false,
                disabled: false,
                hidden: false,
                href: String::new(),
                src: String::new(),
                type_attr: String::new(),
                name_attr: String::new(),
                placeholder: String::new(),
            });
        }
    }
    for c in &n.children {
        collect_elems(c, out);
    }
}

pub(crate) fn empty_elem(tag: &str) -> ElemRef {
    let tag_l = tag.to_ascii_lowercase();
    ElemRef {
        tag: tag_l.clone(),
        id: None,
        class: None,
        text: String::new(),
        value: String::new(),
        style: String::new(),
        canvas_w: if tag_l == "canvas" { Some(300) } else { None },
        canvas_h: if tag_l == "canvas" { Some(150) } else { None },
        attrs: BTreeMap::new(),
        parent: None,
        children: Vec::new(),
        listeners: BTreeMap::new(),
        dataset: BTreeMap::new(),
        checked: false,
        disabled: false,
        hidden: false,
        href: String::new(),
        src: String::new(),
        type_attr: String::new(),
        name_attr: String::new(),
        placeholder: String::new(),
    }
}

/// Run all scripts; returns log lines. Mutates `dom` (title, element text/style).
/// Tries the bytecode VM (`js_bc`) first for simple scripts, then the AST engine.
pub fn run_scripts(dom: &mut JsDom, scripts: &[String]) -> Vec<String> {
    // Seed last inbound message for scripts (`messageData` / `messageOrigin`).
    let (msg_data, msg_origin) = dom
        .inbound_messages
        .last()
        .map(|m| (Some(m.data.clone()), Some(m.origin.clone())))
        .unwrap_or((None, None));

    // Primary DOM tier: run the whole batch on `just` with LIVE DOM bindings —
    // now including canvas 2D (getContext/fillRect/…), fetch (→ fetch_log +
    // Promise<Response>), window/parent postMessage, and a WebAssembly stub, in
    // addition to element/document/window/location/style/classList. It parses
    // everything first, so a parse failure returns false with the DOM untouched
    // → clean fallback to the legacy engine below.
    let uses_dom = scripts.iter().any(|s| {
        s.contains("document")
            || s.contains("localStorage")
            || s.contains("sessionStorage")
            || s.contains("window")
            || s.contains("location")
            || s.contains("fetch")
            || s.contains("postMessage")
            || s.contains("parent")
            || s.contains("getContext")
            || s.contains("canvas")
            || s.contains("WebAssembly")
    });
    if uses_dom && super::js_just::run_scripts_via_just(dom, scripts) {
        super::events::EVENT_LOOP.with(|el| {
            el.drain(16);
        });
        return dom.log.clone();
    }

    for s in scripts {
        // A script that touches DOM / storage / fetch / postMessage / canvas
        // needs the host engine (which owns the DOM bindings). Everything else
        // is DOM-free computation and can run on the richer `just` ES6 tier.
        // NB: unlike the old heuristic, plain `function`/`class`/`for`/`try`
        // do NOT force the host engine — those are exactly what `just` handles
        // better than the bytecode VM.
        let needs_dom = s.contains("document")
            || s.contains("localStorage")
            || s.contains("sessionStorage")
            || s.contains("fetch")
            || s.contains("postMessage")
            || s.contains("parent")
            || s.contains("location")
            || s.contains("window")
            || s.contains("addEventListener")
            || s.contains("WebAssembly")
            || s.contains("getContext")
            || s.contains("canvas")
            || s.contains("fillRect")
            || s.contains("strokeRect");
        if !needs_dom {
            // Tier 1: the `just` ES6 interpreter — closures, single classes,
            // try/catch, destructuring, ternary, for/while loops. Its only
            // observable effect for a DOM-free script is `console.*`, captured
            // into `dom.log`. Any parse/runtime error or unsupported ES feature
            // falls through to the bytecode VM and then the full tree-walker, so
            // this can never regress a script that used to run.
            if let Ok(out) = super::js_just::eval_program(s) {
                for line in out.log {
                    dom.log.push(line);
                }
                continue;
            }
            // Tier 2: the bytecode VM fast path (pure arithmetic/console).
            if let Ok(chunk) = super::js_bc::compile(s) {
                let mut host = super::js_bc::MapHost::default();
                if super::js_bc::run(&chunk, &mut host).is_ok() {
                    for line in host.log {
                        dom.log.push(line);
                    }
                    continue;
                }
            }
        }
        let mut eng = Engine {
            dom,
            vars: Vec::new(),
            steps: 0,
            call_depth: 0,
            flow: Flow::None,
        };
        if let Some(d) = msg_data.clone() {
            eng.set_var("messageData", Val::Str(d));
        }
        if let Some(o) = msg_origin.clone() {
            eng.set_var("messageOrigin", Val::Str(o));
        }
        let _ = eng.exec_program(s);
    }
    // Drain a few event-loop tasks after scripts (HTML checkpoint).
    super::events::EVENT_LOOP.with(|el| {
        el.drain(16);
    });
    dom.log.clone()
}

/// Deliver a message into `dom` (parent ← iframe or iframe ← parent).
pub fn deliver_message(dom: &mut JsDom, msg: Message) {
    dom.inbound_messages.push(msg);
}

/// Flush script-drawn canvas buffers onto layout [`FrameBox`]es (same pre-order
/// canvas element order as `getContext` element indices).
pub fn apply_canvases_to_layout(dom: &JsDom, lay: &mut super::layout::Layout) {
    if dom.canvases.is_empty() {
        return;
    }
    // Map element index → canvas pixels by matching canvas tags in element order
    // to FrameBox::Canvas order.
    let canvas_elems: alloc::vec::Vec<usize> = dom
        .elements
        .iter()
        .enumerate()
        .filter(|(_, e)| e.tag.eq_ignore_ascii_case("canvas"))
        .map(|(i, _)| i)
        .collect();
    let mut frame_i = 0usize;
    for fr in lay.frames.iter_mut() {
        if fr.kind != super::layout::EmbedKind::Canvas {
            continue;
        }
        let elem_idx = canvas_elems.get(frame_i).copied();
        frame_i += 1;
        let Some(ei) = elem_idx else {
            continue;
        };
        if let Some(c2d) = dom.canvases.get(&ei) {
            fr.src_w = c2d.w;
            fr.src_h = c2d.h;
            fr.pixels = Some(c2d.pixels.clone());
            fr.canvas_id = Some(ei);
        }
    }
}

/// Apply JS DOM mutations back onto the live tree (text + style_attr).
/// Matches elements by `id` first (stable), then by pre-order index — same
/// spirit as LibWeb binding a JS wrapper to a DOM node identity.
/// Full DOM→tree commit for the persistent-page path: stamp parse-order
/// element indices, apply element state (text/style), then INSERT nodes for
/// JS-created elements (`createElement` + `appendChild`) so they lay out,
/// paint, and hit-test like parsed ones.
pub fn commit_full(root: &mut Node, dom: &JsDom) {
    stamp_elem_indices(root);
    commit_to_tree(root, dom);
    insert_created_elems(root, dom);
}

/// Append tree nodes for elements that exist in `dom.elements` but have no
/// stamped node in the parsed tree (JS-created). Parents may themselves be
/// created, so iterate until stable (bounded).
fn insert_created_elems(root: &mut Node, dom: &JsDom) {
    for _round in 0..4 {
        let mut inserted = false;
        for (i, er) in dom.elements.iter().enumerate() {
            let Some(p) = er.parent else { continue };
            if find_mut_by_elem_idx(root, i).is_some() {
                continue; // already in the tree (parsed or previously inserted)
            }
            let Some(parent_node) = find_mut_by_elem_idx(root, p) else { continue };
            let mut n = Node {
                kind: NodeKind::Element {
                    tag: er.tag.clone(),
                    href: (!er.href.is_empty()).then(|| er.href.clone()),
                    alt: None,
                    src: (!er.src.is_empty()).then(|| er.src.clone()),
                    id: er.id.clone(),
                    class: er.class.clone(),
                    style_attr: (!er.style.is_empty()).then(|| er.style.clone()),
                    name: None,
                    value: (!er.value.is_empty()).then(|| er.value.clone()),
                    input_type: (!er.type_attr.is_empty()).then(|| er.type_attr.clone()),
                    action: None,
                    method: None,
                    placeholder: None,
                    srcdoc: None,
                    target: None,
                    sandbox: None,
                    width_attr: None,
                    height_attr: None,
                    rel: None,
                    srcset: None,
                    on_attrs: er
                        .attrs
                        .iter()
                        .filter(|(k, _)| k.starts_with("on"))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                },
                children: Vec::new(),
                elem_idx: Some(i),
            };
            if !er.text.is_empty() {
                n.children.push(Node::text(er.text.clone()));
            }
            parent_node.children.push(n);
            inserted = true;
        }
        if !inserted {
            break;
        }
    }
}

/// Locate the tree node stamped with element index `i`.
fn find_mut_by_elem_idx(n: &mut Node, i: usize) -> Option<&mut Node> {
    if n.elem_idx == Some(i) {
        return Some(n);
    }
    let n_ptr = n as *mut Node;
    // SAFETY: single-threaded tree walk; at most one exclusive ref returned
    // (same pattern as `find_mut_by_id` above).
    let n = unsafe { &mut *n_ptr };
    for c in n.children.iter_mut() {
        if let Some(hit) = find_mut_by_elem_idx(c, i) {
            return Some(hit);
        }
    }
    None
}

pub fn commit_to_tree(root: &mut Node, dom: &JsDom) {
    for er in &dom.elements {
        if let Some(id) = er.id.as_deref() {
            if let Some(n) = find_mut_by_id(root, id) {
                apply_elem_ref(n, er);
            }
        }
    }
    // Index-based fallback for nodes without id (in pre-order element order).
    let mut idx = 0usize;
    commit_walk_index(root, dom, &mut idx);
}

fn apply_elem_ref(n: &mut Node, er: &ElemRef) {
    if let NodeKind::Element { style_attr, .. } = &mut n.kind {
        if !er.style.is_empty() {
            *style_attr = Some(er.style.clone());
        }
    }
    // Replace visible text content.
    let has_element_child = n.children.iter().any(|c| matches!(c.kind, NodeKind::Element { .. }));
    if !has_element_child {
        n.children.clear();
        if !er.text.is_empty() {
            n.children.push(Node::text(er.text.clone()));
        }
    } else if n.children.len() == 1 {
        if let NodeKind::Text(t) = &mut n.children[0].kind {
            *t = er.text.clone();
        }
    }
}

fn find_mut_by_id<'a>(n: &'a mut Node, id: &str) -> Option<&'a mut Node> {
    if matches!(&n.kind, NodeKind::Element { id: Some(i), .. } if i.as_str() == id) {
        return Some(n);
    }
    let n_ptr = n as *mut Node;
    // SAFETY: single-threaded tree walk; at most one exclusive ref returned.
    let n = unsafe { &mut *n_ptr };
    for i in 0..n.children.len() {
        let child_ptr = &mut n.children[i] as *mut Node;
        if let Some(found) = find_mut_by_id(unsafe { &mut *child_ptr }, id) {
            return Some(found);
        }
    }
    None
}

fn commit_walk_index(n: &mut Node, dom: &JsDom, idx: &mut usize) {
    if let NodeKind::Element { tag, id, .. } = &n.kind {
        if !matches!(tag.as_str(), "script" | "style" | "noscript") {
            // Prefer id-based path above; only fill nodes that still need it.
            let skip = id.is_some();
            if !skip {
                if let Some(er) = dom.elements.get(*idx) {
                    apply_elem_ref(n, er);
                }
            }
            *idx += 1;
        }
    }
    // Re-borrow for children after mutating n.
    let n_ptr = n as *mut Node;
    let n = unsafe { &mut *n_ptr };
    for i in 0..n.children.len() {
        let c = &mut n.children[i];
        commit_walk_index(c, dom, idx);
    }
}

struct Engine<'a> {
    dom: &'a mut JsDom,
    vars: Vec<(String, Val)>,
    steps: u32,
    call_depth: u32,
    /// Pending flow control (return/throw/break).
    flow: Flow,
}

impl<'a> Engine<'a> {
    fn step(&mut self) -> Result<(), ()> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            Err(())
        } else {
            Ok(())
        }
    }

    fn get_var(&self, name: &str) -> Val {
        for (k, v) in self.vars.iter().rev() {
            if k == name {
                return v.clone();
            }
        }
        // Host globals (Ladybird Window + WorkerGlobalScope subset)
        match name {
            "undefined" => Val::Undefined,
            "null" => Val::Null,
            "true" => Val::Bool(true),
            "false" => Val::Bool(false),
            "NaN" => Val::Num(f64::NAN),
            "Infinity" => Val::Num(f64::INFINITY),
            "console" => Val::Str(String::from("__console__")),
            "document" => Val::Str(String::from("__document__")),
            "window" | "self" | "globalThis" => Val::Str(String::from("__window__")),
            "location" => Val::Str(String::from("__location__")),
            "fetch" => Val::Str(String::from("__fetch__")),
            "alert" => Val::Str(String::from("__alert__")),
            "encodeURIComponent" => Val::Str(String::from("__encodeURIComponent__")),
            "decodeURIComponent" => Val::Str(String::from("__decodeURIComponent__")),
            "scrollTo" => Val::Str(String::from("__scrollTo__")),
            "scrollBy" => Val::Str(String::from("__scrollBy__")),
            "postMessage" => Val::Str(String::from("__postMessage__")),
            "parent" => {
                if self.dom.is_nested {
                    Val::Str(String::from("__parent__"))
                } else {
                    Val::Str(String::from("__window__"))
                }
            }
            "top" => Val::Str(String::from("__window__")),
            "frames" => Val::Str(String::from("__frames__")),
            "JSON" => Val::Str(String::from("__JSON__")),
            "Array" => Val::Str(String::from("__Array__")),
            "Object" => Val::Str(String::from("__Object__")),
            "String" => Val::Str(String::from("__String__")),
            "Number" => Val::Str(String::from("__Number__")),
            "Boolean" => Val::Str(String::from("__Boolean__")),
            "Error" | "TypeError" | "ReferenceError" | "SyntaxError" | "RangeError" => {
                Val::Str(String::from("__Error__"))
            }
            "RegExp" => Val::Str(String::from("__RegExp__")),
            "BigInt" => Val::Str(String::from("__BigInt__")),
            "Math" => Val::Str(String::from("__Math__")),
            "parseInt" => Val::Str(String::from("__parseInt__")),
            "parseFloat" => Val::Str(String::from("__parseFloat__")),
            "isNaN" => Val::Str(String::from("__isNaN__")),
            "isFinite" => Val::Str(String::from("__isFinite__")),
            "localStorage" => Val::Str(String::from("__localStorage__")),
            "sessionStorage" => Val::Str(String::from("__sessionStorage__")),
            "addEventListener" => Val::Str(String::from("__addEventListener__")),
            "dispatchEvent" => Val::Str(String::from("__dispatchEvent__")),
            "WebAssembly" => Val::Str(String::from("__WebAssembly__")),
            _ => Val::Undefined,
        }
    }

    fn set_var(&mut self, name: &str, v: Val) {
        for (k, slot) in self.vars.iter_mut().rev() {
            if k == name {
                *slot = v;
                return;
            }
        }
        self.vars.push((name.to_string(), v));
    }

    fn exec_program(&mut self, src: &str) -> Result<(), ()> {
        let mut p = Parser::new(src);
        while !p.eof() {
            self.step()?;
            p.skip_ws_and_semi();
            if p.eof() {
                break;
            }
            self.exec_stmt(&mut p)?;
            if matches!(self.flow, Flow::Throw(_) | Flow::Return(_)) {
                // Top-level throw: log and stop.
                if let Flow::Throw(v) = core::mem::replace(&mut self.flow, Flow::None) {
                    self.dom.log.push(format!("Uncaught {}", v.as_str()));
                }
                break;
            }
            self.flow = Flow::None;
        }
        Ok(())
    }

    fn exec_stmt(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        self.step()?;
        p.skip_ws();
        if matches!(self.flow, Flow::Return(_) | Flow::Throw(_) | Flow::Break) {
            return Ok(());
        }
        if p.eat_kw("var") || p.eat_kw("let") || p.eat_kw("const") {
            p.skip_ws();
            let name = p.ident().ok_or(())?;
            p.skip_ws();
            if p.eat('=') {
                p.skip_ws();
                let v = self.eval_expr(p)?;
                self.set_var(&name, v);
            } else {
                self.set_var(&name, Val::Undefined);
            }
            p.eat(';');
            return Ok(());
        }
        if p.eat_kw("function") {
            return self.decl_function(p, false);
        }
        if p.eat_kw("class") {
            return self.decl_class(p);
        }
        if p.eat_kw("return") {
            p.skip_ws();
            if p.peek() == ';' || p.peek() == '}' || p.eof() {
                self.flow = Flow::Return(Val::Undefined);
            } else {
                let v = self.eval_expr(p)?;
                self.flow = Flow::Return(v);
            }
            p.eat(';');
            return Ok(());
        }
        if p.eat_kw("throw") {
            p.skip_ws();
            let v = self.eval_expr(p)?;
            self.flow = Flow::Throw(v);
            p.eat(';');
            return Ok(());
        }
        if p.eat_kw("break") {
            self.flow = Flow::Break;
            p.eat(';');
            return Ok(());
        }
        if p.eat_kw("try") {
            return self.exec_try(p);
        }
        if p.eat_kw("if") {
            p.skip_ws();
            p.eat('(');
            let cond = self.eval_expr(p)?;
            p.eat(')');
            p.skip_ws();
            if cond.as_bool() {
                self.exec_block_or_stmt(p)?;
                p.skip_ws();
                if p.eat_kw("else") {
                    self.skip_block_or_stmt(p)?;
                }
            } else {
                self.skip_block_or_stmt(p)?;
                p.skip_ws();
                if p.eat_kw("else") {
                    self.exec_block_or_stmt(p)?;
                }
            }
            return Ok(());
        }
        if p.eat_kw("while") {
            p.skip_ws();
            p.eat('(');
            let cond_start = p.pos;
            let mut d = 1i32;
            while !p.eof() && d > 0 {
                let c = p.peek();
                if c == '(' {
                    d += 1;
                } else if c == ')' {
                    d -= 1;
                    if d == 0 {
                        break;
                    }
                }
                p.bump();
            }
            let cond_end = p.pos;
            p.eat(')');
            p.skip_ws();
            let body_start = p.pos;
            self.skip_block_or_stmt(p)?;
            let body_end = p.pos;
            let full = p.src;
            let cond_slice = &full[cond_start..cond_end];
            let body_slice = &full[body_start..body_end];
            for _ in 0..MAX_LOOP {
                self.step()?;
                let mut cond_parser = Parser::new(cond_slice);
                let cond = self.eval_expr(&mut cond_parser)?;
                if !cond.as_bool() {
                    break;
                }
                let mut bp = Parser::new(body_slice);
                self.exec_block_or_stmt(&mut bp)?;
                if matches!(self.flow, Flow::Return(_) | Flow::Throw(_)) {
                    break;
                }
                if matches!(self.flow, Flow::Break) {
                    self.flow = Flow::None;
                    break;
                }
            }
            return Ok(());
        }
        if p.eat_kw("for") {
            return self.exec_for(p);
        }
        // Expression statement (assignments, calls).
        let _ = self.eval_expr(p)?;
        p.eat(';');
        Ok(())
    }

    fn decl_function(&mut self, p: &mut Parser<'_>, _is_method: bool) -> Result<(), ()> {
        p.skip_ws();
        let name = p.ident().unwrap_or_else(|| String::from("anonymous"));
        p.skip_ws();
        p.eat('(');
        let mut params = Vec::new();
        p.skip_ws();
        if !p.eat(')') {
            loop {
                p.skip_ws();
                if let Some(param) = p.ident() {
                    params.push(param);
                }
                p.skip_ws();
                if p.eat(')') {
                    break;
                }
                p.eat(',');
            }
        }
        p.skip_ws();
        // capture body block source
        let body = capture_block(p)?;
        let fun = Val::Fun(Rc::new(FunVal {
            name: name.clone(),
            params,
            body,
            is_expr: false,
        }));
        self.set_var(&name, fun);
        Ok(())
    }

    fn decl_class(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        p.skip_ws();
        let name = p.ident().ok_or(())?;
        p.skip_ws();
        // optional extends — skip for now
        if p.eat_kw("extends") {
            p.skip_ws();
            let _ = p.ident();
        }
        p.skip_ws();
        p.eat('{');
        let mut ctor_body = String::from("{}");
        let mut methods: BTreeMap<String, Val> = BTreeMap::new();
        while !p.eof() {
            p.skip_ws();
            if p.eat('}') {
                break;
            }
            if p.eat_kw("constructor") {
                p.skip_ws();
                p.eat('(');
                while !p.eof() && !p.eat(')') {
                    p.bump();
                }
                p.skip_ws();
                ctor_body = capture_block(p)?;
                continue;
            }
            // method name()
            if let Some(mname) = p.ident() {
                p.skip_ws();
                if p.eat('(') {
                    let mut params = Vec::new();
                    p.skip_ws();
                    if !p.eat(')') {
                        loop {
                            p.skip_ws();
                            if let Some(param) = p.ident() {
                                params.push(param);
                            }
                            p.skip_ws();
                            if p.eat(')') {
                                break;
                            }
                            p.eat(',');
                        }
                    }
                    p.skip_ws();
                    let body = capture_block(p)?;
                    methods.insert(
                        mname.clone(),
                        Val::Fun(Rc::new(FunVal {
                            name: mname,
                            params,
                            body,
                            is_expr: false,
                        })),
                    );
                    continue;
                }
            }
            // skip unknown
            p.bump();
        }
        // Class as constructor function that returns object with methods.
        let mut body = String::from("{ var __self = {}; ");
        // run constructor body with `this` aliased — simplified: inject methods
        for (k, _) in &methods {
            body.push_str(&format!(
                "__self.{k} = __cls_{name}_{k}; "
            ));
        }
        // strip outer braces of ctor
        let inner = ctor_body
            .trim()
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or(&ctor_body);
        body.push_str(inner);
        body.push_str("; return __self; }");
        // register methods as free functions for wiring
        for (k, v) in methods {
            self.set_var(&format!("__cls_{name}_{k}"), v);
        }
        self.set_var(
            &name,
            Val::Fun(Rc::new(FunVal {
                name: name.clone(),
                params: Vec::new(),
                body,
                is_expr: false,
            })),
        );
        Ok(())
    }

    fn exec_try(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        p.skip_ws();
        let try_body = capture_block(p)?;
        p.skip_ws();
        let mut catch_param = None;
        let mut catch_body = None;
        if p.eat_kw("catch") {
            p.skip_ws();
            if p.eat('(') {
                p.skip_ws();
                catch_param = p.ident();
                p.skip_ws();
                p.eat(')');
            }
            p.skip_ws();
            catch_body = Some(capture_block(p)?);
        }
        p.skip_ws();
        let mut finally_body = None;
        if p.eat_kw("finally") {
            p.skip_ws();
            finally_body = Some(capture_block(p)?);
        }
        // Run try
        let saved_flow = core::mem::replace(&mut self.flow, Flow::None);
        let mut tp = Parser::new(&try_body);
        // strip braces
        let try_inner = strip_braces(&try_body);
        let mut tp = Parser::new(try_inner);
        let _ = self.exec_block_or_stmt(&mut tp);
        let threw = matches!(self.flow, Flow::Throw(_));
        if threw {
            if let Flow::Throw(err) = core::mem::replace(&mut self.flow, Flow::None) {
                if let Some(cb) = catch_body.as_ref() {
                    if let Some(param) = catch_param.as_ref() {
                        self.set_var(param, err);
                    }
                    let inner = strip_braces(cb);
                    let mut cp = Parser::new(inner);
                    let _ = self.exec_block_or_stmt(&mut cp);
                } else {
                    self.flow = Flow::Throw(err);
                }
            }
        }
        if let Some(fb) = finally_body.as_ref() {
            let pending = core::mem::replace(&mut self.flow, Flow::None);
            let inner = strip_braces(fb);
            let mut fp = Parser::new(inner);
            let _ = self.exec_block_or_stmt(&mut fp);
            if matches!(self.flow, Flow::None) {
                self.flow = pending;
            }
        }
        let _ = saved_flow;
        let _ = tp;
        Ok(())
    }

    fn exec_for(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        // for (init; cond; step) body
        p.skip_ws();
        p.eat('(');
        // init
        p.skip_ws();
        if !p.eat(';') {
            if p.eat_kw("var") || p.eat_kw("let") || p.eat_kw("const") {
                p.skip_ws();
                let name = p.ident().ok_or(())?;
                p.skip_ws();
                if p.eat('=') {
                    let v = self.eval_expr(p)?;
                    self.set_var(&name, v);
                }
            } else {
                let _ = self.eval_expr(p)?;
            }
            p.eat(';');
        }
        // cond slice
        let cond_start = p.pos;
        while !p.eof() && p.peek() != ';' {
            p.bump();
        }
        let cond_end = p.pos;
        p.eat(';');
        // step slice
        let step_start = p.pos;
        let mut depth = 0i32;
        while !p.eof() {
            let c = p.peek();
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            p.bump();
        }
        let step_end = p.pos;
        p.eat(')');
        p.skip_ws();
        let body_start = p.pos;
        self.skip_block_or_stmt(p)?;
        let body_end = p.pos;
        let full = p.src;
        let cond_slice = full[cond_start..cond_end].trim();
        let step_slice = full[step_start..step_end].trim();
        let body_slice = &full[body_start..body_end];
        for _ in 0..MAX_LOOP {
            self.step()?;
            if !cond_slice.is_empty() {
                let mut cp = Parser::new(cond_slice);
                let cond = self.eval_expr(&mut cp)?;
                if !cond.as_bool() {
                    break;
                }
            }
            let mut bp = Parser::new(body_slice);
            self.exec_block_or_stmt(&mut bp)?;
            if matches!(self.flow, Flow::Return(_) | Flow::Throw(_)) {
                break;
            }
            if matches!(self.flow, Flow::Break) {
                self.flow = Flow::None;
                break;
            }
            if !step_slice.is_empty() {
                let mut sp = Parser::new(step_slice);
                let _ = self.eval_expr(&mut sp);
            }
        }
        Ok(())
    }

    fn call_user_fun(&mut self, fun: &FunVal, args: &[Val]) -> Result<Val, ()> {
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(());
        }
        self.call_depth += 1;
        let scope_mark = self.vars.len();
        for (i, param) in fun.params.iter().enumerate() {
            let v = args.get(i).cloned().unwrap_or(Val::Undefined);
            self.vars.push((param.clone(), v));
        }
        let prev_flow = core::mem::replace(&mut self.flow, Flow::None);
        let result = if fun.is_expr {
            let mut p = Parser::new(&fun.body);
            self.eval_expr(&mut p)
        } else {
            let inner = strip_braces(&fun.body);
            let mut p = Parser::new(inner);
            while !p.eof() {
                self.step()?;
                p.skip_ws_and_semi();
                if p.eof() {
                    break;
                }
                self.exec_stmt(&mut p)?;
                if matches!(self.flow, Flow::Return(_) | Flow::Throw(_)) {
                    break;
                }
            }
            match core::mem::replace(&mut self.flow, Flow::None) {
                Flow::Return(v) => Ok(v),
                Flow::Throw(v) => {
                    self.flow = Flow::Throw(v);
                    Ok(Val::Undefined)
                }
                _ => Ok(Val::Undefined),
            }
        };
        self.vars.truncate(scope_mark);
        self.call_depth -= 1;
        if matches!(self.flow, Flow::None) {
            self.flow = prev_flow;
        }
        result
    }

    fn exec_block_or_stmt(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        p.skip_ws();
        if p.eat('{') {
            while !p.eof() {
                p.skip_ws();
                if p.eat('}') {
                    break;
                }
                self.exec_stmt(p)?;
            }
            Ok(())
        } else {
            self.exec_stmt(p)
        }
    }

    fn skip_block_or_stmt(&mut self, p: &mut Parser<'_>) -> Result<(), ()> {
        p.skip_ws();
        if p.eat('{') {
            let mut d = 1i32;
            while !p.eof() && d > 0 {
                let c = p.peek();
                if c == '{' {
                    d += 1;
                } else if c == '}' {
                    d -= 1;
                }
                p.bump();
            }
            Ok(())
        } else {
            // skip one stmt roughly until ;
            while !p.eof() {
                let c = p.peek();
                p.bump();
                if c == ';' {
                    break;
                }
                if c == '{' {
                    p.pos -= 1;
                    return self.skip_block_or_stmt(p);
                }
            }
            Ok(())
        }
    }

    fn eval_expr(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        self.eval_assign(p)
    }

    fn eval_assign(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let left_pos = p.pos;
        let left = self.eval_or(p)?;
        p.skip_ws();
        if p.eat('=') && p.peek() != '=' {
            // Assignment: re-parse left as lvalue
            p.skip_ws();
            let right = self.eval_assign(p)?;
            // Simple ident
            let mut lp = Parser {
                src: p.src,
                pos: left_pos,
            };
            lp.skip_ws();
            if let Some(name) = lp.ident() {
                lp.skip_ws();
                if lp.eof() || lp.peek() == ';' {
                    self.set_var(&name, right.clone());
                    return Ok(right);
                }
            }
            // document.title = …
            // elem.innerText = …
            // elem.style.color = …
            let path = parse_path_at(p.src, left_pos);
            self.assign_path(&path, right.clone())?;
            let _ = left;
            return Ok(right);
        }
        Ok(left)
    }

    fn assign_path(&mut self, path: &[String], v: Val) -> Result<(), ()> {
        if path.is_empty() {
            return Err(());
        }
        if path.len() == 2 && path[0] == "document" && path[1] == "title" {
            self.dom.title = v.as_str();
            return Ok(());
        }
        if path.len() == 2 && path[0] == "location" && path[1] == "href" {
            self.dom.location_href = v.as_str();
            self.dom.navigate = Some(v.as_str());
            return Ok(());
        }
        if path.len() == 1 && path[0] == "location" {
            // location = "url"
            self.dom.location_href = v.as_str();
            self.dom.navigate = Some(v.as_str());
            return Ok(());
        }
        if path.len() == 3 && path[0] == "window" && path[1] == "location" && path[2] == "href" {
            self.dom.location_href = v.as_str();
            self.dom.navigate = Some(v.as_str());
            return Ok(());
        }
        if path.len() >= 2 {
            // resolve base element
            // vars: x.innerText / canvas2d props
            let base = self.get_var(&path[0]);
            if let Val::Elem(i) = base {
                if path.len() == 2 && (path[1] == "innerText" || path[1] == "textContent") {
                    if let Some(e) = self.dom.elements.get_mut(i) {
                        e.text = v.as_str();
                    }
                    return Ok(());
                }
                if path.len() == 2 && path[1] == "value" {
                    if let Some(e) = self.dom.elements.get_mut(i) {
                        e.value = v.as_str();
                        e.text = v.as_str();
                    }
                    return Ok(());
                }
                if path.len() == 3 && path[1] == "style" {
                    if let Some(e) = self.dom.elements.get_mut(i) {
                        let prop = css_prop_name(&path[2]);
                        let decl = format!("{prop}: {};", v.as_str());
                        if !e.style.is_empty() && !e.style.ends_with(';') {
                            e.style.push(';');
                        }
                        e.style.push_str(&decl);
                    }
                    return Ok(());
                }
                if path.len() == 2 && (path[1] == "width" || path[1] == "height") {
                    if let Some(e) = self.dom.elements.get_mut(i) {
                        let n = v.as_num() as i32;
                        if path[1] == "width" {
                            e.canvas_w = Some(n);
                        } else {
                            e.canvas_h = Some(n);
                        }
                        if let Some(c) = self.dom.canvases.get_mut(&i) {
                            // Resize by allocating a fresh buffer (content not preserved).
                            let w = e.canvas_w.unwrap_or(c.w as i32);
                            let h = e.canvas_h.unwrap_or(c.h as i32);
                            *c = super::canvas::Canvas2d::new(w, h);
                        }
                    }
                    return Ok(());
                }
                if path.len() == 2 {
                    match path[1].as_str() {
                        "className" => {
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                let s = v.as_str();
                                e.class = Some(s.clone());
                                e.attrs.insert(String::from("class"), s);
                            }
                            return Ok(());
                        }
                        "id" => {
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                let s = v.as_str();
                                e.id = Some(s.clone());
                                e.attrs.insert(String::from("id"), s);
                            }
                            return Ok(());
                        }
                        "innerHTML" | "outerHTML" => {
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                e.text = v.as_str();
                                e.children.clear();
                            }
                            return Ok(());
                        }
                        "checked" | "disabled" | "hidden" => {
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                let b = v.as_bool();
                                match path[1].as_str() {
                                    "checked" => e.checked = b,
                                    "disabled" => e.disabled = b,
                                    "hidden" => e.hidden = b,
                                    _ => {}
                                }
                            }
                            return Ok(());
                        }
                        "href" | "src" | "type" | "name" | "placeholder" => {
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                let s = v.as_str();
                                match path[1].as_str() {
                                    "href" => e.href = s.clone(),
                                    "src" => e.src = s.clone(),
                                    "type" => e.type_attr = s.clone(),
                                    "name" => e.name_attr = s.clone(),
                                    "placeholder" => e.placeholder = s.clone(),
                                    _ => {}
                                }
                                e.attrs.insert(path[1].clone(), s);
                            }
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
            // canvas2d.fillStyle / strokeStyle / lineWidth / font
            if let Val::Str(s) = self.get_var(&path[0]) {
                if let Some(rest) = s.strip_prefix("__canvas2d__") {
                    if let Ok(i) = rest.parse::<usize>() {
                        if path.len() == 2 {
                            let c = self.dom.ensure_canvas(i);
                            match path[1].as_str() {
                                "fillStyle" => c.set_fill_style_css(&v.as_str()),
                                "strokeStyle" => c.set_stroke_style_css(&v.as_str()),
                                "lineWidth" => c.line_width = v.as_num() as i32,
                                "font" => {
                                    // parse leading number from "16px sans-serif"
                                    let n: i32 = v
                                        .as_str()
                                        .split(|ch: char| !ch.is_ascii_digit())
                                        .find(|t| !t.is_empty())
                                        .and_then(|t| t.parse().ok())
                                        .unwrap_or(14);
                                    c.font_size = n as f32;
                                }
                                _ => {}
                            }
                            return Ok(());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn eval_or(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_and(p)?;
        loop {
            p.skip_ws();
            if p.eat_str("||") {
                let r = self.eval_and(p)?;
                v = Val::Bool(v.as_bool() || r.as_bool());
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_and(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_eq(p)?;
        loop {
            p.skip_ws();
            if p.eat_str("&&") {
                let r = self.eval_eq(p)?;
                v = Val::Bool(v.as_bool() && r.as_bool());
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_eq(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_rel(p)?;
        loop {
            p.skip_ws();
            if p.eat_str("===") || p.eat_str("==") {
                let r = self.eval_rel(p)?;
                v = Val::Bool(v.as_str() == r.as_str() || (matches!((&v, &r), (Val::Num(a), Val::Num(b)) if a == b)));
            } else if p.eat_str("!==") || p.eat_str("!=") {
                let r = self.eval_rel(p)?;
                v = Val::Bool(v.as_str() != r.as_str());
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_rel(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_add(p)?;
        loop {
            p.skip_ws();
            if p.eat_str("<=") {
                let r = self.eval_add(p)?;
                v = Val::Bool(v.as_num() <= r.as_num());
            } else if p.eat_str(">=") {
                let r = self.eval_add(p)?;
                v = Val::Bool(v.as_num() >= r.as_num());
            } else if p.eat('<') {
                let r = self.eval_add(p)?;
                v = Val::Bool(v.as_num() < r.as_num());
            } else if p.eat('>') {
                let r = self.eval_add(p)?;
                v = Val::Bool(v.as_num() > r.as_num());
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_add(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_mul(p)?;
        loop {
            p.skip_ws();
            if p.eat('+') {
                let r = self.eval_mul(p)?;
                if matches!(&v, Val::Str(_)) || matches!(&r, Val::Str(_)) {
                    v = Val::Str(format!("{}{}", v.as_str(), r.as_str()));
                } else {
                    v = Val::Num(v.as_num() + r.as_num());
                }
            } else if p.eat('-') {
                let r = self.eval_mul(p)?;
                v = Val::Num(v.as_num() - r.as_num());
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_mul(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_unary(p)?;
        loop {
            p.skip_ws();
            if p.eat('*') {
                let r = self.eval_unary(p)?;
                v = Val::Num(v.as_num() * r.as_num());
            } else if p.eat('/') {
                let r = self.eval_unary(p)?;
                let d = r.as_num();
                v = Val::Num(if d == 0.0 { 0.0 } else { v.as_num() / d });
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn eval_unary(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        p.skip_ws();
        if p.eat('!') {
            let v = self.eval_unary(p)?;
            return Ok(Val::Bool(!v.as_bool()));
        }
        if p.eat('-') {
            let v = self.eval_unary(p)?;
            return Ok(Val::Num(-v.as_num()));
        }
        if p.eat_kw("typeof") {
            let v = self.eval_unary(p)?;
            return Ok(Val::Str(v.type_name().into()));
        }
        if p.eat_kw("new") {
            p.skip_ws();
            let ctor = self.eval_postfix(p)?;
            // eval_postfix already consumed call args if present: `new Foo()` 
            // When `new Foo` without call, invoke with no args.
            return match ctor {
                Val::Fun(f) => self.call_user_fun(&f, &[]),
                Val::Str(ref s)
                    if s.starts_with("__")
                        && (s.contains("Object")
                            || s.contains("Array")
                            || s.contains("Error")
                            || s.contains("RegExp")
                            || s.contains("BigInt")
                            || s.contains("String")
                            || s.contains("Number")
                            || s.contains("Boolean")) =>
                {
                    // Already called if postfix ate (); otherwise call empty.
                    if s.ends_with("__") && !s.contains("result") {
                        self.call_ctor(s, &[])
                    } else {
                        Ok(ctor)
                    }
                }
                other => Ok(other),
            };
        }
        if p.eat('+') {
            let v = self.eval_unary(p)?;
            return Ok(Val::Num(v.as_num()));
        }
        self.eval_postfix(p)
    }

    fn eval_postfix(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        let mut v = self.eval_primary(p)?;
        loop {
            p.skip_ws();
            if p.eat('.') {
                p.skip_ws();
                let prop = p.ident().ok_or(())?;
                v = self.prop_get(v, &prop)?;
            } else if p.eat('[') {
                let key = self.eval_expr(p)?;
                p.eat(']');
                let prop = key.as_str();
                v = self.prop_get(v, &prop)?;
            } else if p.eat('(') {
                let mut args = Vec::new();
                p.skip_ws();
                if !p.eat(')') {
                    loop {
                        args.push(self.eval_expr(p)?);
                        p.skip_ws();
                        if p.eat(')') {
                            break;
                        }
                        p.eat(',');
                        p.skip_ws();
                    }
                }
                v = self.call(v, &args)?;
            } else {
                break;
            }
        }
        Ok(v)
    }

    fn prop_get(&mut self, base: Val, prop: &str) -> Result<Val, ()> {
        match base {
            Val::Str(s) if s == "__document__" => match prop {
                "title" => Ok(Val::Str(self.dom.title.clone())),
                "body" => Ok(self
                    .dom
                    .find_tag("body")
                    .map(Val::Elem)
                    .unwrap_or(Val::Null)),
                "URL" | "documentURI" => Ok(Val::Str(self.dom.location_href.clone())),
                "getElementById" => Ok(Val::Str("__getElementById__".into())),
                "querySelector" => Ok(Val::Str("__querySelector__".into())),
                "querySelectorAll" => Ok(Val::Str("__querySelectorAll__".into())),
                "getElementsByTagName" => Ok(Val::Str("__getElementsByTagName__".into())),
                "getElementsByClassName" => Ok(Val::Str("__getElementsByClassName__".into())),
                "createElement" => Ok(Val::Str("__createElement__".into())),
                "createTextNode" => Ok(Val::Str("__createTextNode__".into())),
                "createDocumentFragment" => Ok(Val::Str("__createDocumentFragment__".into())),
                "createComment" => Ok(Val::Str("__createComment__".into())),
                "createEvent" => Ok(Val::Str("__createEvent__".into())),
                "write" | "writeln" => Ok(Val::Str("__document.write__".into())),
                "open" | "close" => Ok(Val::Str("__noop__".into())),
                "cookie" => {
                    let origin = super::url::origin(&self.dom.location_href)
                        .unwrap_or_else(|| String::from("null"));
                    let c = super::storage::STORAGE.with(|s| {
                        s.cookies()
                            .cookie_header_for(&self.dom.location_href, 0, true)
                    });
                    let _ = origin;
                    Ok(Val::Str(c))
                }
                "readyState" => Ok(Val::Str(String::from("complete"))),
                "compatMode" => Ok(Val::Str(String::from("CSS1Compat"))),
                "characterSet" | "charset" | "inputEncoding" => {
                    Ok(Val::Str(String::from("UTF-8")))
                }
                "contentType" => Ok(Val::Str(String::from("text/html"))),
                "doctype" => Ok(Val::Null),
                "documentElement" => Ok(self
                    .dom
                    .find_tag("html")
                    .map(Val::Elem)
                    .unwrap_or(Val::Null)),
                "head" => Ok(self
                    .dom
                    .find_tag("head")
                    .map(Val::Elem)
                    .unwrap_or(Val::Null)),
                "forms" | "images" | "links" | "scripts" | "embeds" | "plugins" => {
                    Ok(Val::Arr(Rc::new(Vec::new())))
                }
                "activeElement" => Ok(self
                    .dom
                    .find_tag("body")
                    .map(Val::Elem)
                    .unwrap_or(Val::Null)),
                "defaultView" => Ok(Val::Str(String::from("__window__"))),
                "location" => Ok(Val::Str("__location__".into())),
                "addEventListener" => Ok(Val::Str("__addEventListener__".into())),
                "removeEventListener" => Ok(Val::Str("__noop__".into())),
                "dispatchEvent" => Ok(Val::Str("__dispatchEvent__".into())),
                "hasFocus" => Ok(Val::Str("__document.hasFocus__".into())),
                "elementFromPoint" => Ok(Val::Str("__noop__".into())),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__localStorage__" => match prop {
                "getItem" => Ok(Val::Str("__localStorage.getItem__".into())),
                "setItem" => Ok(Val::Str("__localStorage.setItem__".into())),
                "removeItem" => Ok(Val::Str("__localStorage.removeItem__".into())),
                "clear" => Ok(Val::Str("__localStorage.clear__".into())),
                "key" => Ok(Val::Str("__localStorage.key__".into())),
                "length" => {
                    let origin = super::url::origin(&self.dom.location_href)
                        .unwrap_or_else(|| String::from("null"));
                    let n = super::storage::STORAGE.with(|s| s.local_for(&origin).len());
                    Ok(Val::Num(n as f64))
                }
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__sessionStorage__" => match prop {
                "getItem" => Ok(Val::Str("__sessionStorage.getItem__".into())),
                "setItem" => Ok(Val::Str("__sessionStorage.setItem__".into())),
                "removeItem" => Ok(Val::Str("__sessionStorage.removeItem__".into())),
                "clear" => Ok(Val::Str("__sessionStorage.clear__".into())),
                "key" => Ok(Val::Str("__sessionStorage.key__".into())),
                "length" => {
                    let origin = super::url::origin(&self.dom.location_href)
                        .unwrap_or_else(|| String::from("null"));
                    let n = super::storage::STORAGE.with(|s| s.session_for(&origin).len());
                    Ok(Val::Num(n as f64))
                }
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__window__" => match prop {
                "document" => Ok(Val::Str("__document__".into())),
                "location" => Ok(Val::Str("__location__".into())),
                "console" => Ok(Val::Str("__console__".into())),
                "localStorage" => Ok(Val::Str("__localStorage__".into())),
                "sessionStorage" => Ok(Val::Str("__sessionStorage__".into())),
                "addEventListener" => Ok(Val::Str("__addEventListener__".into())),
                "dispatchEvent" => Ok(Val::Str("__dispatchEvent__".into())),
                "innerWidth" => Ok(Val::Num(self.dom.inner_width as f64)),
                "innerHeight" => Ok(Val::Num(self.dom.inner_height as f64)),
                "scrollX" | "pageXOffset" => Ok(Val::Num(0.0)),
                "scrollY" | "pageYOffset" => Ok(Val::Num(
                    self.dom.scroll_to.unwrap_or(0) as f64,
                )),
                "fetch" => Ok(Val::Str("__fetch__".into())),
                "alert" => Ok(Val::Str("__alert__".into())),
                "scrollTo" => Ok(Val::Str("__scrollTo__".into())),
                "scrollBy" => Ok(Val::Str("__scrollBy__".into())),
                "encodeURIComponent" => Ok(Val::Str("__encodeURIComponent__".into())),
                "decodeURIComponent" => Ok(Val::Str("__decodeURIComponent__".into())),
                "postMessage" => Ok(Val::Str("__postMessage__".into())),
                "parent" => Ok(Val::Str(if self.dom.is_nested {
                    "__parent__".into()
                } else {
                    "__window__".into()
                })),
                "self" | "window" | "globalThis" => Ok(Val::Str("__window__".into())),
                "navigator" => Ok(Val::Str("__navigator__".into())),
                "frames" => Ok(Val::Str("__frames__".into())),
                "length" => Ok(Val::Num(0.0)), // frame count host-filled later
                "origin" => Ok(Val::Str(
                    super::url::origin(&self.dom.location_href).unwrap_or_else(|| {
                        String::from("null")
                    }),
                )),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__parent__" => match prop {
                "postMessage" => Ok(Val::Str("__parentPostMessage__".into())),
                "location" => Ok(Val::Str("__location__".into())),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__JSON__" => match prop {
                "stringify" => Ok(Val::Str("__JSON.stringify__".into())),
                "parse" => Ok(Val::Str("__JSON.parse__".into())),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__Math__" => match prop {
                "floor" | "ceil" | "round" | "abs" | "min" | "max" | "sqrt" | "pow" => {
                    Ok(Val::Str(format!("__Math.{prop}__")))
                }
                "PI" => Ok(Val::Num(3.141592653589793)),
                "E" => Ok(Val::Num(2.718281828459045)),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__location__" => match prop {
                "href" => Ok(Val::Str(self.dom.location_href.clone())),
                "toString" => Ok(Val::Str("__location.toString__".into())),
                "assign" | "replace" => Ok(Val::Str("__location.assign__".into())),
                "reload" => Ok(Val::Str("__location.reload__".into())),
                "pathname" => {
                    let path = self
                        .dom
                        .location_href
                        .split("://")
                        .nth(1)
                        .and_then(|r| r.find('/').map(|i| &r[i..]))
                        .unwrap_or("/");
                    Ok(Val::Str(path.into()))
                }
                "host" | "hostname" => {
                    let host = self
                        .dom
                        .location_href
                        .split("://")
                        .nth(1)
                        .map(|r| r.split('/').next().unwrap_or(r))
                        .unwrap_or("");
                    Ok(Val::Str(host.into()))
                }
                "protocol" => {
                    let p = if self.dom.location_href.starts_with("https") {
                        "https:"
                    } else if self.dom.location_href.starts_with("http") {
                        "http:"
                    } else {
                        ""
                    };
                    Ok(Val::Str(p.into()))
                }
                "origin" => {
                    // scheme://host
                    let href = &self.dom.location_href;
                    if let Some(rest) = href.split("://").nth(1) {
                        let host = rest.split('/').next().unwrap_or(rest);
                        let scheme = href.split("://").next().unwrap_or("http");
                        Ok(Val::Str(format!("{scheme}://{host}")))
                    } else {
                        Ok(Val::Str(href.clone()))
                    }
                }
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__navigator__" => match prop {
                "userAgent" => Ok(Val::Str("ChittiBrowser/0.1".into())),
                "language" => Ok(Val::Str("en".into())),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__console__" => match prop {
                "log" | "info" | "warn" | "error" => Ok(Val::Str("__console.log__".into())),
                _ => Ok(Val::Undefined),
            },
            Val::Obj(map) => {
                if let Some(v) = map.get(prop) {
                    Ok(v.clone())
                } else {
                    Ok(Val::Undefined)
                }
            }
            Val::Arr(arr) => match prop {
                "length" => Ok(Val::Num(arr.len() as f64)),
                "push" => Ok(Val::Str("__array.push__".into())),
                "join" => Ok(Val::Str("__array.join__".into())),
                "pop" => Ok(Val::Str("__array.pop__".into())),
                n if n.chars().all(|c| c.is_ascii_digit()) => {
                    let i: usize = n.parse().unwrap_or(usize::MAX);
                    Ok(arr.get(i).cloned().unwrap_or(Val::Undefined))
                }
                _ => Ok(Val::Undefined),
            },
            Val::RegExp(re) => match prop {
                "source" => Ok(Val::Str(re.pattern.clone())),
                "flags" => Ok(Val::Str(re.flags.clone())),
                "test" => Ok(Val::Str(format!(
                    "__regexp.test__{}__{}",
                    re.pattern, re.flags
                ))),
                "exec" => Ok(Val::Str(format!(
                    "__regexp.exec__{}__{}",
                    re.pattern, re.flags
                ))),
                _ => Ok(Val::Undefined),
            },
            Val::Fun(f) => match prop {
                "name" => Ok(Val::Str(f.name.clone())),
                "length" => Ok(Val::Num(f.params.len() as f64)),
                _ => Ok(Val::Undefined),
            },
            Val::Str(s) if s == "__Array__" => match prop {
                "isArray" => Ok(Val::Str("__Array.isArray__".into())),
                _ => Ok(Val::Str("__Array__".into())), // new Array via call
            },
            Val::Str(s) if s == "__Object__" => match prop {
                "keys" => Ok(Val::Str("__Object.keys__".into())),
                "assign" => Ok(Val::Str("__Object.assign__".into())),
                _ => Ok(Val::Str("__Object__".into())),
            },
            Val::Elem(i) => {
                let e = self.dom.elements.get(i).ok_or(())?;
                match prop {
                    "innerText" | "textContent" => Ok(Val::Str(e.text.clone())),
                    "value" => Ok(Val::Str(e.value.clone())),
                    "id" => Ok(Val::Str(e.id.clone().unwrap_or_default())),
                    "style" => Ok(Val::Str(format!("__style__{i}"))),
                    "tagName" => Ok(Val::Str(e.tag.to_ascii_uppercase())),
                    "width" => {
                        if e.tag.eq_ignore_ascii_case("canvas") {
                            Ok(Val::Num(
                                e.canvas_w.unwrap_or(300) as f64,
                            ))
                        } else {
                            Ok(Val::Undefined)
                        }
                    }
                    "height" => {
                        if e.tag.eq_ignore_ascii_case("canvas") {
                            Ok(Val::Num(
                                e.canvas_h.unwrap_or(150) as f64,
                            ))
                        } else {
                            Ok(Val::Undefined)
                        }
                    }
                    "getContext" => Ok(Val::Str(format!("__getContext__{i}"))),
                    "getAttribute" => Ok(Val::Str(format!("__getAttribute__{i}"))),
                    "setAttribute" => Ok(Val::Str(format!("__setAttribute__{i}"))),
                    "removeAttribute" => Ok(Val::Str(format!("__removeAttribute__{i}"))),
                    "hasAttribute" => Ok(Val::Str(format!("__hasAttribute__{i}"))),
                    "getAttributeNS" => Ok(Val::Str(format!("__getAttribute__{i}"))),
                    "setAttributeNS" => Ok(Val::Str(format!("__setAttribute__{i}"))),
                    "appendChild" => Ok(Val::Str(format!("__appendChild__{i}"))),
                    "removeChild" => Ok(Val::Str(format!("__removeChild__{i}"))),
                    "insertBefore" => Ok(Val::Str(format!("__insertBefore__{i}"))),
                    "replaceChild" => Ok(Val::Str(format!("__replaceChild__{i}"))),
                    "cloneNode" => Ok(Val::Str(format!("__cloneNode__{i}"))),
                    "contains" => Ok(Val::Str(format!("__contains__{i}"))),
                    "remove" => Ok(Val::Str(format!("__remove__{i}"))),
                    "append" => Ok(Val::Str(format!("__appendChild__{i}"))),
                    "prepend" => Ok(Val::Str(format!("__appendChild__{i}"))),
                    "before" | "after" | "replaceWith" => Ok(Val::Str("__noop__".into())),
                    "querySelector" => Ok(Val::Str(format!("__elemQuery__{i}"))),
                    "querySelectorAll" => Ok(Val::Str(format!("__elemQueryAll__{i}"))),
                    "getElementsByTagName" => Ok(Val::Str("__getElementsByTagName__".into())),
                    "getElementsByClassName" => Ok(Val::Str("__getElementsByClassName__".into())),
                    "matches" | "webkitMatchesSelector" | "closest" => {
                        Ok(Val::Str(format!("__matches__{i}")))
                    }
                    "addEventListener" => Ok(Val::Str(format!("__elemAddEvent__{i}"))),
                    "removeEventListener" => Ok(Val::Str("__noop__".into())),
                    "dispatchEvent" => Ok(Val::Str(format!("__elemDispatch__{i}"))),
                    "focus" | "blur" | "click" | "submit" | "play" | "pause" | "load"
                    | "scrollIntoView" | "requestFullscreen" => Ok(Val::Str("__noop__".into())),
                    "className" => Ok(Val::Str(e.class.clone().unwrap_or_default())),
                    "classList" => Ok(Val::Str(format!("__classList__{i}"))),
                    "dataset" => Ok(Val::Str(format!("__dataset__{i}"))),
                    "innerHTML" | "outerHTML" => Ok(Val::Str(e.text.clone())),
                    "children" | "childNodes" => {
                        let kids: Vec<Val> =
                            e.children.iter().copied().map(Val::Elem).collect();
                        Ok(Val::Arr(Rc::new(kids)))
                    }
                    "firstChild" | "firstElementChild" => Ok(e
                        .children
                        .first()
                        .copied()
                        .map(Val::Elem)
                        .unwrap_or(Val::Null)),
                    "lastChild" | "lastElementChild" => Ok(e
                        .children
                        .last()
                        .copied()
                        .map(Val::Elem)
                        .unwrap_or(Val::Null)),
                    "parentNode" | "parentElement" => {
                        Ok(e.parent.map(Val::Elem).unwrap_or(Val::Null))
                    }
                    "nextSibling" | "nextElementSibling" | "previousSibling"
                    | "previousElementSibling" => Ok(Val::Null),
                    "ownerDocument" => Ok(Val::Str(String::from("__document__"))),
                    "nodeType" => Ok(Val::Num(1.0)), // ELEMENT_NODE
                    "nodeName" => Ok(Val::Str(e.tag.to_ascii_uppercase())),
                    "nodeValue" => Ok(Val::Null),
                    "childElementCount" => Ok(Val::Num(e.children.len() as f64)),
                    "isConnected" => Ok(Val::Bool(e.parent.is_some() || e.tag == "html" || e.tag == "body")),
                    "namespaceURI" => Ok(Val::Str(String::from(
                        "http://www.w3.org/1999/xhtml",
                    ))),
                    "localName" => Ok(Val::Str(e.tag.clone())),
                    "checked" => Ok(Val::Bool(e.checked)),
                    "disabled" => Ok(Val::Bool(e.disabled)),
                    "hidden" => Ok(Val::Bool(e.hidden)),
                    "href" => Ok(Val::Str(e.href.clone())),
                    "src" => Ok(Val::Str(e.src.clone())),
                    "type" => Ok(Val::Str(e.type_attr.clone())),
                    "name" => Ok(Val::Str(e.name_attr.clone())),
                    "placeholder" => Ok(Val::Str(e.placeholder.clone())),
                    "clientWidth" | "offsetWidth" | "scrollWidth" => Ok(Val::Num(100.0)),
                    "clientHeight" | "offsetHeight" | "scrollHeight" => Ok(Val::Num(40.0)),
                    "clientTop" | "clientLeft" | "offsetTop" | "offsetLeft" | "scrollTop"
                    | "scrollLeft" => Ok(Val::Num(0.0)),
                    "getBoundingClientRect" => Ok(Val::Str("__getBoundingClientRect__".into())),
                    "getClientRects" => Ok(Val::Str("__getClientRects__".into())),
                    _ => Ok(Val::Undefined),
                }
            }
            Val::Str(s) if s.starts_with("__classList__") => match prop {
                "add" => Ok(Val::Str(format!(
                    "__classList.add__{}",
                    s.strip_prefix("__classList__").unwrap_or("0")
                ))),
                "remove" => Ok(Val::Str(format!(
                    "__classList.remove__{}",
                    s.strip_prefix("__classList__").unwrap_or("0")
                ))),
                "toggle" => Ok(Val::Str(format!(
                    "__classList.toggle__{}",
                    s.strip_prefix("__classList__").unwrap_or("0")
                ))),
                "contains" => Ok(Val::Str(format!(
                    "__classList.contains__{}",
                    s.strip_prefix("__classList__").unwrap_or("0")
                ))),
                "length" => {
                    let i: usize = s
                        .strip_prefix("__classList__")
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(0);
                    let n = self
                        .dom
                        .elements
                        .get(i)
                        .and_then(|e| e.class.as_ref())
                        .map(|c| c.split_whitespace().count())
                        .unwrap_or(0);
                    Ok(Val::Num(n as f64))
                }
                _ => Ok(Val::Undefined),
            }
            Val::Str(s) if s.starts_with("__dataset__") => {
                let i: usize = s
                    .strip_prefix("__dataset__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                Ok(self
                    .dom
                    .elements
                    .get(i)
                    .and_then(|e| e.dataset.get(prop).cloned())
                    .map(Val::Str)
                    .unwrap_or(Val::Undefined))
            }
            Val::Str(s) if s.starts_with("__canvas2d__") => {
                let i: usize = s
                    .strip_prefix("__canvas2d__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                match prop {
                    "fillRect" => Ok(Val::Str(format!("__c2d.fillRect__{i}"))),
                    "strokeRect" => Ok(Val::Str(format!("__c2d.strokeRect__{i}"))),
                    "clearRect" => Ok(Val::Str(format!("__c2d.clearRect__{i}"))),
                    "fillText" => Ok(Val::Str(format!("__c2d.fillText__{i}"))),
                    "strokeText" => Ok(Val::Str(format!("__c2d.fillText__{i}"))),
                    "beginPath" => Ok(Val::Str(format!("__c2d.beginPath__{i}"))),
                    "moveTo" => Ok(Val::Str(format!("__c2d.moveTo__{i}"))),
                    "lineTo" => Ok(Val::Str(format!("__c2d.lineTo__{i}"))),
                    "closePath" => Ok(Val::Str(format!("__c2d.closePath__{i}"))),
                    "stroke" => Ok(Val::Str(format!("__c2d.stroke__{i}"))),
                    "fill" => Ok(Val::Str(format!("__c2d.fill__{i}"))),
                    "arc" => Ok(Val::Str(format!("__c2d.arc__{i}"))),
                    "fillStyle" => {
                        let c = self
                            .dom
                            .canvases
                            .get(&i)
                            .map(|c| c.fill_style)
                            .unwrap_or(0);
                        Ok(Val::Str(format!("#{c:06x}")))
                    }
                    "strokeStyle" => {
                        let c = self
                            .dom
                            .canvases
                            .get(&i)
                            .map(|c| c.stroke_style)
                            .unwrap_or(0);
                        Ok(Val::Str(format!("#{c:06x}")))
                    }
                    "lineWidth" => {
                        let w = self
                            .dom
                            .canvases
                            .get(&i)
                            .map(|c| c.line_width)
                            .unwrap_or(1);
                        Ok(Val::Num(w as f64))
                    }
                    "canvas" => Ok(Val::Elem(i)),
                    _ => Ok(Val::Undefined),
                }
            }
            Val::Str(s) if s.starts_with("__style__") => {
                Ok(Val::Str(String::new()))
            }
            Val::Str(s) if s.starts_with("__fetch_result__") => match prop {
                "ok" => Ok(Val::Bool(true)),
                "status" => Ok(Val::Num(200.0)),
                "text" | "json" => Ok(Val::Str("__fetch_body__".into())),
                "url" => Ok(Val::Str(
                    s.strip_prefix("__fetch_result__").unwrap_or("").into(),
                )),
                _ => Ok(Val::Undefined),
            },
            _ => Ok(Val::Undefined),
        }
    }

    fn call(&mut self, callee: Val, args: &[Val]) -> Result<Val, ()> {
        match callee {
            Val::Fun(f) => self.call_user_fun(&f, args),
            Val::Str(s) if s == "__console.log__" || s == "__alert__" => {
                let msg = args
                    .iter()
                    .map(|a| a.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                self.dom.log.push(msg);
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__getElementById__" => {
                let id = args.first().map(|a| a.as_str()).unwrap_or_default();
                Ok(self.dom.find_id(&id).map(Val::Elem).unwrap_or(Val::Null))
            }
            Val::Str(s) if s == "__querySelector__" => {
                let sel = args.first().map(|a| a.as_str()).unwrap_or_default();
                let sel = sel.trim();
                if let Some(id) = sel.strip_prefix('#') {
                    Ok(self.dom.find_id(id).map(Val::Elem).unwrap_or(Val::Null))
                } else if let Some(c) = sel.strip_prefix('.') {
                    Ok(self.dom.find_class(c).map(Val::Elem).unwrap_or(Val::Null))
                } else {
                    Ok(self.dom.find_tag(sel).map(Val::Elem).unwrap_or(Val::Null))
                }
            }
            Val::Str(s) if s == "__createElement__" => {
                let tag = args
                    .first()
                    .map(|a| a.as_str())
                    .unwrap_or_else(|| String::from("div"));
                let i = self.dom.elements.len();
                self.dom.elements.push(empty_elem(&tag));
                Ok(Val::Elem(i))
            }
            Val::Str(s) if s == "__createTextNode__" || s == "__createComment__" => {
                let text = args.first().map(|a| a.as_str()).unwrap_or_default();
                let i = self.dom.elements.len();
                let mut el = empty_elem("#text");
                el.text = text;
                self.dom.elements.push(el);
                Ok(Val::Elem(i))
            }
            Val::Str(s) if s == "__createDocumentFragment__" => {
                let i = self.dom.elements.len();
                self.dom.elements.push(empty_elem("#document-fragment"));
                Ok(Val::Elem(i))
            }
            Val::Str(s) if s == "__querySelectorAll__" => {
                let sel = args.first().map(|a| a.as_str()).unwrap_or_default();
                let list = self.dom_query_all(&sel);
                Ok(Val::Arr(Rc::new(list.into_iter().map(Val::Elem).collect())))
            }
            Val::Str(s) if s == "__getElementsByTagName__" => {
                let tag = args.first().map(|a| a.as_str()).unwrap_or_default();
                let list: Vec<Val> = self
                    .dom
                    .elements
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| tag == "*" || e.tag.eq_ignore_ascii_case(&tag))
                    .map(|(i, _)| Val::Elem(i))
                    .collect();
                Ok(Val::Arr(Rc::new(list)))
            }
            Val::Str(s) if s == "__getElementsByClassName__" => {
                let cls = args.first().map(|a| a.as_str()).unwrap_or_default();
                let list: Vec<Val> = self
                    .dom
                    .elements
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| {
                        e.class
                            .as_deref()
                            .map(|c| c.split_whitespace().any(|x| x == cls))
                            .unwrap_or(false)
                    })
                    .map(|(i, _)| Val::Elem(i))
                    .collect();
                Ok(Val::Arr(Rc::new(list)))
            }
            Val::Str(s) if s == "__document.hasFocus__" => Ok(Val::Bool(true)),
            Val::Str(s) if s == "__document.write__" => {
                let t = args.iter().map(|a| a.as_str()).collect::<Vec<_>>().join("");
                self.dom.log.push(format!("document.write:{t}"));
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__getBoundingClientRect__" => {
                let mut m = BTreeMap::new();
                for (k, v) in [
                    ("x", 0.0),
                    ("y", 0.0),
                    ("top", 0.0),
                    ("left", 0.0),
                    ("right", 100.0),
                    ("bottom", 40.0),
                    ("width", 100.0),
                    ("height", 40.0),
                ] {
                    m.insert(String::from(k), Val::Num(v));
                }
                Ok(Val::Obj(Rc::new(m)))
            }
            Val::Str(s) if s == "__getClientRects__" => Ok(Val::Arr(Rc::new(Vec::new()))),
            Val::Str(s) if s.starts_with("__appendChild__") => {
                let parent: usize = s
                    .strip_prefix("__appendChild__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                if let Some(Val::Elem(child)) = args.first() {
                    self.dom_append_child(parent, *child);
                    Ok(Val::Elem(*child))
                } else {
                    Ok(Val::Null)
                }
            }
            Val::Str(s) if s.starts_with("__removeChild__") => {
                let parent: usize = s
                    .strip_prefix("__removeChild__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                if let Some(Val::Elem(child)) = args.first() {
                    self.dom_remove_child(parent, *child);
                    Ok(Val::Elem(*child))
                } else {
                    Ok(Val::Null)
                }
            }
            Val::Str(s) if s.starts_with("__remove__") => {
                let i: usize = s
                    .strip_prefix("__remove__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                if let Some(p) = self.dom.elements.get(i).and_then(|e| e.parent) {
                    self.dom_remove_child(p, i);
                }
                Ok(Val::Undefined)
            }
            Val::Str(s) if s.starts_with("__cloneNode__") => {
                let i: usize = s
                    .strip_prefix("__cloneNode__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let clone = self.dom.elements.get(i).cloned().unwrap_or_else(|| empty_elem("div"));
                let ni = self.dom.elements.len();
                let mut c = clone;
                c.parent = None;
                c.children.clear();
                self.dom.elements.push(c);
                Ok(Val::Elem(ni))
            }
            Val::Str(s) if s.starts_with("__contains__") => {
                let _i: usize = s
                    .strip_prefix("__contains__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                Ok(Val::Bool(args.first().map(|a| matches!(a, Val::Elem(_))).unwrap_or(false)))
            }
            Val::Str(s) if s.starts_with("__elemAddEvent__") => {
                let i: usize = s
                    .strip_prefix("__elemAddEvent__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let typ = args.first().map(|a| a.as_str()).unwrap_or_default();
                let handler = args.get(1).map(|a| a.as_str()).unwrap_or_default();
                if let Some(e) = self.dom.elements.get_mut(i) {
                    e.listeners.entry(typ).or_default().push(handler);
                }
                Ok(Val::Undefined)
            }
            Val::Str(s) if s.starts_with("__classList.") => {
                self.call_class_list(&s, args)
            }
            Val::Str(s) if s.starts_with("__hasAttribute__") => {
                let i: usize = s
                    .strip_prefix("__hasAttribute__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let name = args.first().map(|a| a.as_str()).unwrap_or_default();
                let ok = self
                    .dom
                    .elements
                    .get(i)
                    .map(|e| e.attrs.contains_key(&name) || (name == "id" && e.id.is_some()) || (name == "class" && e.class.is_some()))
                    .unwrap_or(false);
                Ok(Val::Bool(ok))
            }
            Val::Str(s) if s.starts_with("__removeAttribute__") => {
                let i: usize = s
                    .strip_prefix("__removeAttribute__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let name = args.first().map(|a| a.as_str()).unwrap_or_default();
                if let Some(e) = self.dom.elements.get_mut(i) {
                    e.attrs.remove(&name);
                    if name == "id" {
                        e.id = None;
                    }
                    if name == "class" {
                        e.class = None;
                    }
                }
                Ok(Val::Undefined)
            }
            Val::Str(s) if s.starts_with("__getContext__") => {
                let i: usize = s
                    .strip_prefix("__getContext__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let kind = args
                    .first()
                    .map(|a| a.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if kind == "2d" || kind.is_empty() {
                    let _ = self.dom.ensure_canvas(i);
                    Ok(Val::Str(format!("__canvas2d__{i}")))
                } else {
                    Ok(Val::Null)
                }
            }
            Val::Str(s) if s.starts_with("__c2d.") => {
                self.call_canvas2d(&s, args)
            }
            Val::Str(s) if s == "__fetch__" => self.call_fetch(args),
            Val::Str(s) if s == "__postMessage__" || s == "__parentPostMessage__" => {
                let data = args.first().map(|a| a.as_str()).unwrap_or_default();
                let target_origin = args
                    .get(1)
                    .map(|a| a.as_str())
                    .unwrap_or_else(|| String::from("*"));
                let origin =
                    super::url::origin(&self.dom.location_href).unwrap_or_else(|| String::from("null"));
                let target = if s == "__parentPostMessage__" {
                    String::from("parent")
                } else {
                    String::from("self")
                };
                self.dom.outbound_messages.push(Message {
                    data,
                    origin,
                    target_origin,
                    target,
                });
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__JSON.stringify__" => {
                let v = args.first().cloned().unwrap_or(Val::Null);
                Ok(Val::Str(json_stringify(&v)))
            }
            Val::Str(s) if s == "__JSON.parse__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                Ok(json_parse_simple(&s))
            }
            Val::Str(s) if s.starts_with("__Math.") => {
                let name = s
                    .strip_prefix("__Math.")
                    .and_then(|x| x.strip_suffix("__"))
                    .unwrap_or("");
                let a = args.first().map(|v| v.as_num()).unwrap_or(0.0);
                let b = args.get(1).map(|v| v.as_num()).unwrap_or(0.0);
                // no_std: avoid f64::floor/sqrt (need libm); use integer-ish helpers.
                let r = match name {
                    "floor" => {
                        let i = a as i64;
                        if (a < 0.0) && (a != i as f64) {
                            (i - 1) as f64
                        } else {
                            i as f64
                        }
                    }
                    "ceil" => {
                        let i = a as i64;
                        if (a > 0.0) && (a != i as f64) {
                            (i + 1) as f64
                        } else {
                            i as f64
                        }
                    }
                    "round" => {
                        if a >= 0.0 {
                            (a + 0.5) as i64 as f64
                        } else {
                            (a - 0.5) as i64 as f64
                        }
                    }
                    "abs" => {
                        if a < 0.0 {
                            -a
                        } else {
                            a
                        }
                    }
                    "min" => {
                        if a < b {
                            a
                        } else {
                            b
                        }
                    }
                    "max" => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    "sqrt" => {
                        // Newton for non-negative; 0 for negatives.
                        if a <= 0.0 {
                            0.0
                        } else {
                            let mut x = a;
                            for _ in 0..12 {
                                x = 0.5 * (x + a / x);
                            }
                            x
                        }
                    }
                    "pow" => {
                        // Integer exponent only.
                        let exp = b as i32;
                        let mut r = 1.0f64;
                        let mut base = a;
                        let mut e = exp.unsigned_abs();
                        while e > 0 {
                            if e & 1 == 1 {
                                r *= base;
                            }
                            base *= base;
                            e >>= 1;
                        }
                        if exp < 0 {
                            if r == 0.0 {
                                0.0
                            } else {
                                1.0 / r
                            }
                        } else {
                            r
                        }
                    }
                    _ => 0.0,
                };
                Ok(Val::Num(r))
            }
            Val::Str(s) if s == "__parseInt__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                let n: f64 = s
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+')
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0.0);
                Ok(Val::Num(n))
            }
            Val::Str(s) if s == "__parseFloat__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                Ok(Val::Num(s.trim().parse().unwrap_or(0.0)))
            }
            Val::Str(s) if s == "__isNaN__" => {
                let n = args.first().map(|a| a.as_num()).unwrap_or(0.0);
                Ok(Val::Bool(n != n))
            }
            Val::Str(s) if s == "__fetch_body__" => {
                // Last fetch body stored as a special log line prefix.
                let body = self
                    .dom
                    .log
                    .iter()
                    .rev()
                    .find_map(|l| l.strip_prefix("__fetch_body:"))
                    .unwrap_or("{}")
                    .to_string();
                Ok(Val::Str(body))
            }
            Val::Str(s) if s == "__encodeURIComponent__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                Ok(Val::Str(super::form::percent_encode(&s)))
            }
            Val::Str(s) if s == "__decodeURIComponent__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_default();
                // form percent_decode treats + as space; for URI use a light path.
                Ok(Val::Str(super::form::percent_decode(
                    &s.replace('+', "%2B"),
                )))
            }
            Val::Str(s) if s == "__scrollTo__" => {
                // scrollTo(x, y) or scrollTo({top:y})
                let y = if args.len() >= 2 {
                    args[1].as_num() as i32
                } else {
                    args.first().map(|a| a.as_num() as i32).unwrap_or(0)
                };
                self.dom.scroll_to = Some(y.max(0));
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__scrollBy__" => {
                let dy = if args.len() >= 2 {
                    args[1].as_num() as i32
                } else {
                    args.first().map(|a| a.as_num() as i32).unwrap_or(0)
                };
                let cur = self.dom.scroll_to.unwrap_or(0);
                self.dom.scroll_to = Some((cur + dy).max(0));
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__location.assign__" => {
                let url = args.first().map(|a| a.as_str()).unwrap_or_default();
                self.dom.location_href = url.clone();
                self.dom.navigate = Some(url);
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__location.reload__" => {
                self.dom.navigate = Some(self.dom.location_href.clone());
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__location.toString__" => {
                Ok(Val::Str(self.dom.location_href.clone()))
            }
            Val::Str(s) if s == "__noop__" => Ok(Val::Undefined),
            Val::Str(s) if s == "__Object__" || s == "__Array__" || s == "__Error__"
                || s == "__RegExp__" || s == "__BigInt__" || s == "__String__"
                || s == "__Number__" || s == "__Boolean__" =>
            {
                self.call_ctor(&s, args)
            }
            Val::Str(s) if s == "__isFinite__" => {
                let n = args.first().map(|a| a.as_num()).unwrap_or(0.0);
                Ok(Val::Bool(n.is_finite()))
            }
            Val::Str(s) if s.starts_with("__regexp.test__") => {
                let rest = s.strip_prefix("__regexp.test__").unwrap_or("");
                let (pat, _flags) = rest.split_once("__").unwrap_or((rest, ""));
                let text = args.first().map(|a| a.as_str()).unwrap_or_default();
                Ok(Val::Bool(simple_re_test(pat, &text)))
            }
            Val::Str(s) if s.starts_with("__regexp.exec__") => {
                let rest = s.strip_prefix("__regexp.exec__").unwrap_or("");
                let (pat, _flags) = rest.split_once("__").unwrap_or((rest, ""));
                let text = args.first().map(|a| a.as_str()).unwrap_or_default();
                if simple_re_test(pat, &text) {
                    Ok(Val::Arr(Rc::new(alloc::vec![Val::Str(text)])))
                } else {
                    Ok(Val::Null)
                }
            }
            Val::Str(s) if s == "__array.push__" => {
                // Cannot mutate Rc array easily — return new length stub
                Ok(Val::Num(args.len() as f64))
            }
            Val::Str(s) if s == "__array.join__" => {
                Ok(Val::Str(String::new()))
            }
            Val::Str(s) if s == "__Array.isArray__" => {
                Ok(Val::Bool(matches!(args.first(), Some(Val::Arr(_)))))
            }
            Val::Str(s) if s == "__Object.keys__" => {
                if let Some(Val::Obj(m)) = args.first() {
                    let keys: Vec<Val> = m.keys().cloned().map(Val::Str).collect();
                    Ok(Val::Arr(Rc::new(keys)))
                } else {
                    Ok(Val::Arr(Rc::new(Vec::new())))
                }
            }
            Val::RegExp(re) => {
                let _ = re;
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__localStorage__" || s.starts_with("__localStorage.") => {
                self.storage_call("local", &s, args)
            }
            Val::Str(s) if s == "__sessionStorage__" || s.starts_with("__sessionStorage.") => {
                self.storage_call("session", &s, args)
            }
            Val::Str(s) if s == "__addEventListener__" => {
                let typ = args.first().map(|a| a.as_str()).unwrap_or_default();
                let handler = args
                    .get(1)
                    .map(|a| a.as_str())
                    .unwrap_or_else(|| String::from("console"));
                super::events::EVENT_LOOP.with(|el| {
                    el.add_event_listener(
                        super::events::TARGET_WINDOW,
                        &typ,
                        &handler,
                        false,
                        false,
                    );
                });
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__dispatchEvent__" => {
                let typ = args.first().map(|a| a.as_str()).unwrap_or_default();
                let data = args.get(1).map(|a| a.as_str()).unwrap_or_default();
                super::events::EVENT_LOOP.with(|el| {
                    el.queue_event(
                        super::events::Event::new(&typ, super::events::TARGET_WINDOW)
                            .with_data(&data),
                    );
                    el.drain(8);
                });
                Ok(Val::Undefined)
            }
            Val::Str(s) if s == "__WebAssembly__" => Ok(Val::Str("__WebAssembly__".into())),
            Val::Str(s) if s.starts_with("__getAttribute__") => {
                let i: usize = s
                    .strip_prefix("__getAttribute__")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
                let name = args.first().map(|a| a.as_str()).unwrap_or_default();
                let v = self.dom.elements.get(i).and_then(|e| {
                    e.attrs
                        .get(&name)
                        .cloned()
                        .or_else(|| {
                            if name == "id" {
                                e.id.clone()
                            } else if name == "class" {
                                e.class.clone()
                            } else if name == "style" {
                                Some(e.style.clone())
                            } else {
                                None
                            }
                        })
                });
                Ok(v.map(Val::Str).unwrap_or(Val::Null))
            }
            Val::Str(s) if s.starts_with("__setAttribute__") => {
                if let Some(rest) = s.strip_prefix("__setAttribute__") {
                    if let Ok(i) = rest.parse::<usize>() {
                        if let (Some(name), Some(val)) = (args.first(), args.get(1)) {
                            let n = name.as_str();
                            let v = val.as_str();
                            if let Some(e) = self.dom.elements.get_mut(i) {
                                e.attrs.insert(n.clone(), v.clone());
                                match n.as_str() {
                                    "style" => e.style = v,
                                    "id" => e.id = Some(v),
                                    "class" => e.class = Some(v),
                                    "href" => e.href = v,
                                    "src" => e.src = v,
                                    "type" => e.type_attr = v,
                                    "name" => e.name_attr = v,
                                    "placeholder" => e.placeholder = v,
                                    "value" => {
                                        e.value = v.clone();
                                        e.text = v;
                                    }
                                    _ => {
                                        if let Some(ds) = n.strip_prefix("data-") {
                                            e.dataset.insert(ds.to_string(), v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Val::Undefined)
            }
            _ => Ok(Val::Undefined),
        }
    }

    fn call_ctor(&mut self, which: &str, args: &[Val]) -> Result<Val, ()> {
        match which {
            "__Object__" => {
                let mut m = BTreeMap::new();
                // Object({...}) shallow — if arg is obj, return it
                if let Some(Val::Obj(o)) = args.first() {
                    return Ok(Val::Obj(o.clone()));
                }
                Ok(Val::Obj(Rc::new(m)))
            }
            "__Array__" => {
                if args.len() == 1 {
                    if let Val::Num(n) = &args[0] {
                        let len = (*n as usize).min(10_000);
                        return Ok(Val::Arr(Rc::new(alloc::vec![Val::Undefined; len])));
                    }
                }
                Ok(Val::Arr(Rc::new(args.to_vec())))
            }
            "__Error__" => {
                let msg = args.first().map(|a| a.as_str()).unwrap_or_default();
                let mut m = BTreeMap::new();
                m.insert(String::from("message"), Val::Str(msg));
                m.insert(String::from("name"), Val::Str(String::from("Error")));
                Ok(Val::Obj(Rc::new(m)))
            }
            "__RegExp__" => {
                let pat = args.first().map(|a| a.as_str()).unwrap_or_default();
                let flags = args.get(1).map(|a| a.as_str()).unwrap_or_default();
                Ok(Val::RegExp(Rc::new(RegExpVal { pattern: pat, flags })))
            }
            "__BigInt__" => {
                let s = args.first().map(|a| a.as_str()).unwrap_or_else(|| String::from("0"));
                let n: i128 = s.trim().parse().unwrap_or(0);
                Ok(Val::BigInt(n))
            }
            "__String__" => Ok(Val::Str(
                args.first().map(|a| a.as_str()).unwrap_or_default(),
            )),
            "__Number__" => Ok(Val::Num(
                args.first().map(|a| a.as_num()).unwrap_or(0.0),
            )),
            "__Boolean__" => Ok(Val::Bool(
                args.first().map(|a| a.as_bool()).unwrap_or(false),
            )),
            _ => Ok(Val::Undefined),
        }
    }

    fn dom_append_child(&mut self, parent: usize, child: usize) {
        if parent == child || parent >= self.dom.elements.len() || child >= self.dom.elements.len()
        {
            return;
        }
        // Detach from previous parent
        if let Some(op) = self.dom.elements.get(child).and_then(|e| e.parent) {
            if let Some(pe) = self.dom.elements.get_mut(op) {
                pe.children.retain(|&c| c != child);
            }
        }
        if let Some(pe) = self.dom.elements.get_mut(parent) {
            if !pe.children.contains(&child) {
                pe.children.push(child);
            }
        }
        if let Some(ce) = self.dom.elements.get_mut(child) {
            ce.parent = Some(parent);
        }
    }

    fn dom_remove_child(&mut self, parent: usize, child: usize) {
        if let Some(pe) = self.dom.elements.get_mut(parent) {
            pe.children.retain(|&c| c != child);
        }
        if let Some(ce) = self.dom.elements.get_mut(child) {
            if ce.parent == Some(parent) {
                ce.parent = None;
            }
        }
    }

    fn dom_query_all(&self, sel: &str) -> Vec<usize> {
        let sel = sel.trim();
        if sel.is_empty() {
            return Vec::new();
        }
        if let Some(id) = sel.strip_prefix('#') {
            return self.dom.find_id(id).into_iter().collect();
        }
        if let Some(c) = sel.strip_prefix('.') {
            return self
                .dom
                .elements
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.class
                        .as_deref()
                        .map(|cl| cl.split_whitespace().any(|x| x == c))
                        .unwrap_or(false)
                })
                .map(|(i, _)| i)
                .collect();
        }
        // tag or tag.class
        if let Some((tag, cls)) = sel.split_once('.') {
            return self
                .dom
                .elements
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    e.tag.eq_ignore_ascii_case(tag)
                        && e.class
                            .as_deref()
                            .map(|cl| cl.split_whitespace().any(|x| x == cls))
                            .unwrap_or(false)
                })
                .map(|(i, _)| i)
                .collect();
        }
        self.dom
            .elements
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tag.eq_ignore_ascii_case(sel))
            .map(|(i, _)| i)
            .collect()
    }

    fn call_class_list(&mut self, callee: &str, args: &[Val]) -> Result<Val, ()> {
        // __classList.add__{i}
        let rest = callee.strip_prefix("__classList.").unwrap_or(callee);
        let (method, idx_s) = rest.split_once("__").unwrap_or((rest, "0"));
        let i: usize = idx_s.parse().unwrap_or(0);
        let token = args.first().map(|a| a.as_str()).unwrap_or_default();
        let Some(e) = self.dom.elements.get_mut(i) else {
            return Ok(Val::Undefined);
        };
        let mut classes: Vec<String> = e
            .class
            .as_deref()
            .unwrap_or("")
            .split_whitespace()
            .map(String::from)
            .collect();
        match method {
            "add" => {
                if !token.is_empty() && !classes.iter().any(|c| c == &token) {
                    classes.push(token);
                }
            }
            "remove" => classes.retain(|c| c != &token),
            "toggle" => {
                if classes.iter().any(|c| c == &token) {
                    classes.retain(|c| c != &token);
                } else if !token.is_empty() {
                    classes.push(token);
                }
            }
            "contains" => {
                return Ok(Val::Bool(classes.iter().any(|c| c == &token)));
            }
            _ => {}
        }
        let joined = classes.join(" ");
        e.class = if joined.is_empty() {
            None
        } else {
            Some(joined.clone())
        };
        e.attrs.insert(String::from("class"), joined);
        Ok(Val::Undefined)
    }

    fn call_canvas2d(&mut self, callee: &str, args: &[Val]) -> Result<Val, ()> {
        // __c2d.fillRect__{i}
        let rest = callee.strip_prefix("__c2d.").unwrap_or(callee);
        let (method, idx_s) = rest.split_once("__").unwrap_or((rest, "0"));
        let i: usize = idx_s.parse().unwrap_or(0);
        let c = self.dom.ensure_canvas(i);
        let n = |a: Option<&Val>| a.map(|v| v.as_num() as i32).unwrap_or(0);
        let f = |a: Option<&Val>| a.map(|v| v.as_num() as f32).unwrap_or(0.0);
        match method {
            "fillRect" => {
                c.fill_rect(n(args.first()), n(args.get(1)), n(args.get(2)), n(args.get(3)));
            }
            "strokeRect" => {
                c.stroke_rect(n(args.first()), n(args.get(1)), n(args.get(2)), n(args.get(3)));
            }
            "clearRect" => {
                c.clear_rect(n(args.first()), n(args.get(1)), n(args.get(2)), n(args.get(3)));
            }
            "fillText" => {
                let text = args.first().map(|a| a.as_str()).unwrap_or_default();
                c.fill_text(&text, n(args.get(1)), n(args.get(2)));
            }
            "beginPath" => c.begin_path(),
            "moveTo" => c.move_to(f(args.first()), f(args.get(1))),
            "lineTo" => c.line_to(f(args.first()), f(args.get(1))),
            "closePath" => c.close_path(),
            "stroke" => c.stroke(),
            "fill" => c.fill(),
            "arc" => c.arc(
                f(args.first()),
                f(args.get(1)),
                f(args.get(2)),
                f(args.get(3)),
                f(args.get(4)).max(0.01),
            ),
            _ => {}
        }
        Ok(Val::Undefined)
    }

    fn storage_call(&mut self, kind: &str, callee: &str, args: &[Val]) -> Result<Val, ()> {
        // localStorage.getItem via prop_get returns __localStorage.getItem__ style —
        // handle both bare storage object methods when called as storage_call after prop.
        let method = if let Some(rest) = callee.strip_prefix("__localStorage.") {
            rest.strip_suffix("__").unwrap_or(rest)
        } else if let Some(rest) = callee.strip_prefix("__sessionStorage.") {
            rest.strip_suffix("__").unwrap_or(rest)
        } else {
            // Bare: treat first arg pattern from chained call — use prop path
            ""
        };
        let origin = super::url::origin(&self.dom.location_href)
            .unwrap_or_else(|| String::from("null"));
        let method = if method.is_empty() {
            // call on storage object directly unsupported
            "getItem"
        } else {
            method
        };
        match method {
            "getItem" => {
                let key = args.first().map(|a| a.as_str()).unwrap_or_default();
                let v = super::storage::STORAGE.with(|s| {
                    let store = if kind == "session" {
                        s.session_for(&origin)
                    } else {
                        s.local_for(&origin)
                    };
                    store.get_item(&key)
                });
                Ok(v.map(Val::Str).unwrap_or(Val::Null))
            }
            "setItem" => {
                let key = args.first().map(|a| a.as_str()).unwrap_or_default();
                let val = args.get(1).map(|a| a.as_str()).unwrap_or_default();
                let _ = super::storage::STORAGE.with(|s| {
                    let store = if kind == "session" {
                        s.session_for(&origin)
                    } else {
                        s.local_for(&origin)
                    };
                    store.set_item(&key, &val)
                });
                Ok(Val::Undefined)
            }
            "removeItem" => {
                let key = args.first().map(|a| a.as_str()).unwrap_or_default();
                super::storage::STORAGE.with(|s| {
                    let store = if kind == "session" {
                        s.session_for(&origin)
                    } else {
                        s.local_for(&origin)
                    };
                    store.remove_item(&key);
                });
                Ok(Val::Undefined)
            }
            "clear" => {
                super::storage::STORAGE.with(|s| {
                    let store = if kind == "session" {
                        s.session_for(&origin)
                    } else {
                        s.local_for(&origin)
                    };
                    store.clear();
                });
                Ok(Val::Undefined)
            }
            "key" => {
                let i = args.first().map(|a| a.as_num() as usize).unwrap_or(0);
                let v = super::storage::STORAGE.with(|s| {
                    let store = if kind == "session" {
                        s.session_for(&origin)
                    } else {
                        s.local_for(&origin)
                    };
                    store.key(i)
                });
                Ok(v.map(Val::Str).unwrap_or(Val::Null))
            }
            _ => Ok(Val::Undefined),
        }
    }

    fn call_fetch(&mut self, args: &[Val]) -> Result<Val, ()> {
        let url = args.first().map(|a| a.as_str()).unwrap_or_default();
        let mut method = String::from("GET");
        let mut body = String::new();
        // Second arg object is not fully parsed; accept string body as arg2.
        if let Some(a1) = args.get(1) {
            // If it's a plain string, treat as method shorthand-less body for POST.
            match a1 {
                Val::Str(s) if s.starts_with('{') || s.contains("method") => {
                    // Minimal parse: look for method and body in the stringified object.
                    let low = s.to_ascii_lowercase();
                    if low.contains("post") {
                        method = String::from("POST");
                    }
                    if let Some(i) = s.find("body") {
                        // body: "..." or body:"..."
                        let rest = &s[i..];
                        if let Some(q) = rest.find('"') {
                            let rest2 = &rest[q + 1..];
                            // skip optional :
                            let rest2 = rest2.trim_start_matches(|c: char| c == ':' || c.is_whitespace() || c == '"');
                            if let Some(end) = rest2.find('"') {
                                body = rest2[..end].to_string();
                            }
                        }
                    }
                }
                Val::Str(s) => {
                    method = String::from("POST");
                    body = s.clone();
                }
                _ => {}
            }
        }
        let abs = if super::url::is_http_url(&url) {
            url.clone()
        } else {
            super::url::resolve(&self.dom.location_href, &url).unwrap_or(url.clone())
        };
        self.dom
            .fetch_log
            .push((method.clone(), abs.clone(), body.clone()));
        let response_text = host_fetch(&method, &abs, &body);
        self.dom
            .log
            .push(format!("__fetch_body:{response_text}"));
        // Thenable-ish: return result object; also support .then via ignored.
        Ok(Val::Str(format!("__fetch_result__{abs}")))
    }

    fn eval_primary(&mut self, p: &mut Parser<'_>) -> Result<Val, ()> {
        self.step()?;
        p.skip_ws();
        // Arrow / paren group — if ( params ) => body
        if p.eat('(') {
            let save = p.pos;
            // Try parse as arrow params: (a, b) =>
            let mut params = Vec::new();
            p.skip_ws();
            if !p.eat(')') {
                loop {
                    p.skip_ws();
                    if let Some(name) = p.ident() {
                        params.push(name);
                    } else {
                        // not arrow — reparse as group expr
                        p.pos = save;
                        let v = self.eval_expr(p)?;
                        p.eat(')');
                        return Ok(v);
                    }
                    p.skip_ws();
                    if p.eat(')') {
                        break;
                    }
                    if !p.eat(',') {
                        p.pos = save;
                        let v = self.eval_expr(p)?;
                        p.eat(')');
                        return Ok(v);
                    }
                }
            }
            p.skip_ws();
            if p.eat_str("=>") {
                p.skip_ws();
                let body = if p.peek() == '{' {
                    capture_block(p)?
                } else {
                    // expression body until ; or ,
                    let start = p.pos;
                    let mut depth = 0i32;
                    while !p.eof() {
                        let c = p.peek();
                        if c == '(' || c == '[' || c == '{' {
                            depth += 1;
                        } else if c == ')' || c == ']' || c == '}' {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        } else if (c == ';' || c == ',') && depth == 0 {
                            break;
                        }
                        p.bump();
                    }
                    p.src[start..p.pos].to_string()
                };
                let is_expr = !body.trim_start().starts_with('{');
                return Ok(Val::Fun(Rc::new(FunVal {
                    name: String::from("arrow"),
                    params,
                    body,
                    is_expr,
                })));
            }
            // grouped expression already consumed `(` — reparse
            p.pos = save;
            let v = self.eval_expr(p)?;
            p.eat(')');
            return Ok(v);
        }
        // Object literal
        if p.eat('{') {
            let mut map = BTreeMap::new();
            p.skip_ws();
            if !p.eat('}') {
                loop {
                    p.skip_ws();
                    let key = if p.peek() == '"' || p.peek() == '\'' {
                        p.string().ok_or(())?
                    } else {
                        p.ident().ok_or(())?
                    };
                    p.skip_ws();
                    p.eat(':');
                    p.skip_ws();
                    let val = self.eval_expr(p)?;
                    map.insert(key, val);
                    p.skip_ws();
                    if p.eat('}') {
                        break;
                    }
                    p.eat(',');
                    p.skip_ws();
                    if p.eat('}') {
                        break;
                    }
                }
            }
            return Ok(Val::Obj(Rc::new(map)));
        }
        // Array literal
        if p.eat('[') {
            let mut arr = Vec::new();
            p.skip_ws();
            if !p.eat(']') {
                loop {
                    p.skip_ws();
                    if p.eat(']') {
                        break;
                    }
                    arr.push(self.eval_expr(p)?);
                    p.skip_ws();
                    if p.eat(']') {
                        break;
                    }
                    p.eat(',');
                }
            }
            return Ok(Val::Arr(Rc::new(arr)));
        }
        // RegExp literal /pattern/flags
        if p.peek() == '/' {
            // Ambiguous with division — only if previous was start/op (heuristic: always here in primary).
            p.bump();
            let start = p.pos;
            while !p.eof() {
                let c = p.peek();
                if c == '\\' {
                    p.bump();
                    p.bump();
                    continue;
                }
                if c == '/' {
                    break;
                }
                p.bump();
            }
            let pattern = p.src[start..p.pos].to_string();
            p.eat('/');
            let mut flags = String::new();
            while p.peek().is_ascii_alphabetic() {
                flags.push(p.peek());
                p.bump();
            }
            return Ok(Val::RegExp(Rc::new(RegExpVal { pattern, flags })));
        }
        if p.peek() == '"' || p.peek() == '\'' {
            return Ok(Val::Str(p.string().ok_or(())?));
        }
        if p.peek().is_ascii_digit() {
            let n = p.number().ok_or(())?;
            // BigInt suffix
            if p.peek() == 'n' {
                p.bump();
                return Ok(Val::BigInt(n as i128));
            }
            return Ok(Val::Num(n));
        }
        // single-param arrow: x => ...
        if let Some(name) = p.ident() {
            p.skip_ws();
            if p.eat_str("=>") {
                p.skip_ws();
                let body = if p.peek() == '{' {
                    capture_block(p)?
                } else {
                    let start = p.pos;
                    let mut depth = 0i32;
                    while !p.eof() {
                        let c = p.peek();
                        if c == '(' || c == '[' || c == '{' {
                            depth += 1;
                        } else if c == ')' || c == ']' || c == '}' {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        } else if (c == ';' || c == ',') && depth == 0 {
                            break;
                        }
                        p.bump();
                    }
                    p.src[start..p.pos].to_string()
                };
                let is_expr = !body.trim_start().starts_with('{');
                return Ok(Val::Fun(Rc::new(FunVal {
                    name: String::from("arrow"),
                    params: alloc::vec![name],
                    body,
                    is_expr,
                })));
            }
            return Ok(self.get_var(&name));
        }
        Err(())
    }
}

fn capture_block(p: &mut Parser<'_>) -> Result<String, ()> {
    p.skip_ws();
    if p.peek() != '{' {
        // single statement as block
        let start = p.pos;
        while !p.eof() && p.peek() != ';' {
            p.bump();
        }
        p.eat(';');
        return Ok(format!("{{ {} }}", &p.src[start..p.pos]));
    }
    let start = p.pos;
    p.eat('{');
    let mut depth = 1i32;
    while !p.eof() && depth > 0 {
        let c = p.peek();
        if c == '"' || c == '\'' {
            let q = c;
            p.bump();
            while !p.eof() {
                let c2 = p.peek();
                p.bump();
                if c2 == '\\' {
                    p.bump();
                    continue;
                }
                if c2 == q {
                    break;
                }
            }
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
        }
        p.bump();
    }
    Ok(p.src[start..p.pos].to_string())
}

fn strip_braces(s: &str) -> &str {
    let t = s.trim();
    t.strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .unwrap_or(t)
}

/// Tiny regexp subset: literal substring, `.` any, `^`/`$` anchors, simple `*`.
fn simple_re_test(pat: &str, text: &str) -> bool {
    if pat.is_empty() {
        return true;
    }
    // Escape-aware literal-ish match: if no metachar, substring search.
    let meta = pat.contains('.')
        || pat.contains('*')
        || pat.contains('+')
        || pat.contains('?')
        || pat.contains('[')
        || pat.contains('(')
        || pat.contains('^')
        || pat.contains('$');
    if !meta {
        return text.contains(pat);
    }
    // Very small matcher: ^pat$ or . * support via recursive NFA-ish
    match_re(pat.as_bytes(), text.as_bytes())
}

fn match_re(pat: &[u8], text: &[u8]) -> bool {
    fn go(p: &[u8], t: &[u8]) -> bool {
        if p.is_empty() {
            return t.is_empty();
        }
        if p[0] == b'^' {
            return go(&p[1..], t);
        }
        if p[0] == b'$' {
            return t.is_empty() && (p.len() == 1 || go(&p[1..], t));
        }
        let (atom, rest) = if p.len() >= 2 && p[1] == b'*' {
            (&p[0..1], &p[2..])
        } else {
            (&p[0..1], &p[1..])
        };
        let is_star = p.len() >= 2 && p[1] == b'*';
        if is_star {
            // zero or more
            let mut i = 0;
            loop {
                if go(rest, &t[i..]) {
                    return true;
                }
                if i >= t.len() {
                    return false;
                }
                if atom[0] != b'.' && atom[0] != t[i] {
                    return false;
                }
                i += 1;
            }
        } else {
            if t.is_empty() {
                return false;
            }
            if atom[0] != b'.' && atom[0] != t[0] {
                return false;
            }
            go(rest, &t[1..])
        }
    }
    if pat.first() == Some(&b'^') {
        go(pat, text)
    } else {
        // search
        for i in 0..=text.len() {
            if go(pat, &text[i..]) {
                return true;
            }
        }
        false
    }
}

fn json_stringify(v: &Val) -> String {
    match v {
        Val::Null => String::from("null"),
        Val::Undefined => String::from("null"),
        Val::Bool(true) => String::from("true"),
        Val::Bool(false) => String::from("false"),
        Val::Num(n) => {
            if *n == (*n as i64) as f64 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Val::BigInt(n) => format!("{n}"),
        Val::Str(s) => {
            let mut o = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    _ => o.push(c),
                }
            }
            o.push('"');
            o
        }
        Val::Elem(i) => format!("{{\"tag\":\"Element\",\"i\":{i}}}"),
        Val::Obj(m) => {
            let parts: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("\"{k}\":{}", json_stringify(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Val::Arr(a) => {
            let parts: Vec<String> = a.iter().map(json_stringify).collect();
            format!("[{}]", parts.join(","))
        }
        Val::Fun(f) => format!("\"function {}(){{}}\"", f.name),
        Val::RegExp(r) => format!("\"/{}/{}\"", r.pattern, r.flags),
    }
}

/// Minimal JSON value parse: null/true/false/number/string only (no objects/arrays).
fn json_parse_simple(s: &str) -> Val {
    let s = s.trim();
    if s == "null" {
        return Val::Null;
    }
    if s == "true" {
        return Val::Bool(true);
    }
    if s == "false" {
        return Val::Bool(false);
    }
    if let Ok(n) = s.parse::<f64>() {
        return Val::Num(n);
    }
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return Val::Str(s[1..s.len() - 1].to_string());
    }
    Val::Str(s.to_string())
}

/// Network for `fetch` — real loader outside unit tests; stub inside.
///
/// Hardening (prompt-injection / SSRF defence for untrusted page scripts):
/// - only `http://` / `https://`
/// - no loopback / link-local / RFC1918 private destinations
/// - non-GET methods refused (no ambient POST exfil from page JS)
pub(crate) fn host_fetch(method: &str, url: &str, body: &str) -> String {
    let _ = body;
    if let Err(e) = fetch_policy_ok(method, url) {
        return format!("{{\"error\":\"{e}\"}}");
    }
    #[cfg(test)]
    {
        return format!("{{\"ok\":true,\"url\":\"{url}\"}}");
    }
    #[cfg(not(test))]
    {
        use super::cache::CacheMode;
        use super::loader::{Destination, LoadRequest};
        // Non-GET already refused by `fetch_policy_ok`.
        let mut req = LoadRequest::get(url)
            .with_cache_mode(CacheMode::Default)
            .with_timeout(30_000);
        req.destination = Destination::Other;
        match super::loader::load(&req) {
            Ok(r) => String::from_utf8_lossy(&r.body).into_owned(),
            Err(e) => format!("{{\"error\":\"{e}\"}}"),
        }
    }
}

/// Pure policy for page-script `fetch` (unit-tested).
pub(crate) fn fetch_policy_ok(method: &str, url: &str) -> Result<(), &'static str> {
    let m = method.trim();
    if !(m.eq_ignore_ascii_case("GET") || m.eq_ignore_ascii_case("HEAD")) {
        return Err("page fetch: only GET/HEAD allowed (no POST from untrusted scripts)");
    }
    if !super::url::is_http_url(url) {
        return Err("page fetch: only http(s) URLs");
    }
    let host = super::url::split_http(url)
        .map(|(_, h, _)| h)
        .ok_or("page fetch: bad url")?;
    // strip port for IP checks
    let host_only = host.split(':').next().unwrap_or(&host);
    if is_blocked_fetch_host(host_only) {
        return Err("page fetch: blocked host (loopback/private/link-local)");
    }
    Ok(())
}

/// True for hosts page scripts must not reach (SSRF / local-service probe).
fn is_blocked_fetch_host(host: &str) -> bool {
    let h = host.trim().trim_matches(|c| c == '[' || c == ']').to_ascii_lowercase();
    if h == "localhost" || h.ends_with(".localhost") || h == "0.0.0.0" {
        return true;
    }
    // IPv4 dotted-quad private/link-local/loopback.
    let mut parts = [0u8; 4];
    let mut i = 0usize;
    for seg in h.split('.') {
        if i >= 4 {
            return false; // not a simple IPv4
        }
        if let Ok(n) = seg.parse::<u8>() {
            parts[i] = n;
            i += 1;
        } else {
            return false; // hostname — allow (no DNS-rebinding defence here)
        }
    }
    if i != 4 {
        return false;
    }
    let [a, b, _, _] = parts;
    // 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16, 0/8
    a == 127
        || a == 10
        || a == 0
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 169 && b == 254)
}

fn css_prop_name(js: &str) -> String {
    // backgroundColor → background-color
    let mut out = String::new();
    for c in js.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_path_at(src: &str, pos: usize) -> Vec<String> {
    let mut p = Parser { src, pos };
    p.skip_ws();
    let mut path = Vec::new();
    if let Some(n) = p.ident() {
        path.push(n);
    }
    loop {
        p.skip_ws();
        if p.eat('.') {
            p.skip_ws();
            if let Some(n) = p.ident() {
                path.push(n);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    path
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> char {
        self.src[self.pos..].chars().next().unwrap_or('\0')
    }

    fn bump(&mut self) {
        if let Some(c) = self.src[self.pos..].chars().next() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_ws(&mut self) {
        while !self.eof() {
            let c = self.peek();
            if c.is_whitespace() {
                self.bump();
                continue;
            }
            // // comment
            if c == '/' && self.src[self.pos..].starts_with("//") {
                while !self.eof() && self.peek() != '\n' {
                    self.bump();
                }
                continue;
            }
            // /* */
            if c == '/' && self.src[self.pos..].starts_with("/*") {
                self.pos += 2;
                while !self.eof() && !self.src[self.pos..].starts_with("*/") {
                    self.bump();
                }
                if self.src[self.pos..].starts_with("*/") {
                    self.pos += 2;
                }
                continue;
            }
            break;
        }
    }

    fn skip_ws_and_semi(&mut self) {
        loop {
            self.skip_ws();
            if self.eat(';') {
                continue;
            }
            break;
        }
    }

    fn eat(&mut self, c: char) -> bool {
        self.skip_ws();
        if self.peek() == c {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, s: &str) -> bool {
        self.skip_ws();
        if self.src[self.pos..].starts_with(s) {
            // don't match === as ==
            if s == "==" && self.src[self.pos..].starts_with("===") {
                return false;
            }
            if s == "!=" && self.src[self.pos..].starts_with("!==") {
                return false;
            }
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        self.skip_ws();
        if self.src[self.pos..].starts_with(kw) {
            let after = self.pos + kw.len();
            let next = self.src[after..].chars().next().unwrap_or('\0');
            if next.is_ascii_alphanumeric() || next == '_' || next == '$' {
                return false;
            }
            self.pos = after;
            true
        } else {
            false
        }
    }

    fn ident(&mut self) -> Option<String> {
        self.skip_ws();
        let c = self.peek();
        if !(c.is_ascii_alphabetic() || c == '_' || c == '$') {
            return None;
        }
        let start = self.pos;
        self.bump();
        while !self.eof() {
            let c = self.peek();
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                self.bump();
            } else {
                break;
            }
        }
        Some(self.src[start..self.pos].to_string())
    }

    fn string(&mut self) -> Option<String> {
        self.skip_ws();
        let q = self.peek();
        if q != '"' && q != '\'' {
            return None;
        }
        self.bump();
        let mut out = String::new();
        while !self.eof() {
            let c = self.peek();
            self.bump();
            if c == q {
                break;
            }
            if c == '\\' {
                let n = self.peek();
                self.bump();
                out.push(match n {
                    'n' => '\n',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    o => o,
                });
            } else {
                out.push(c);
            }
        }
        Some(out)
    }

    fn number(&mut self) -> Option<f64> {
        self.skip_ws();
        let start = self.pos;
        while !self.eof() && (self.peek().is_ascii_digit() || self.peek() == '.') {
            self.bump();
        }
        self.src[start..self.pos].parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::html;

    #[test_case]
    fn fetch_policy_blocks_private_and_post() {
        assert!(super::fetch_policy_ok("GET", "https://example.com/x").is_ok());
        assert!(super::fetch_policy_ok("HEAD", "http://cdn.example.org/a").is_ok());
        assert!(super::fetch_policy_ok("POST", "https://example.com/x").is_err());
        assert!(super::fetch_policy_ok("GET", "http://127.0.0.1/secret").is_err());
        assert!(super::fetch_policy_ok("GET", "http://localhost/x").is_err());
        assert!(super::fetch_policy_ok("GET", "http://192.168.1.1/").is_err());
        assert!(super::fetch_policy_ok("GET", "http://10.0.0.5/").is_err());
        assert!(super::fetch_policy_ok("GET", "http://169.254.169.254/").is_err());
        assert!(super::fetch_policy_ok("GET", "file:///etc/passwd").is_err());
    }

    #[test_case]
    fn dom_free_script_runs_on_just_tier() {
        // A closure-based script (which the bytecode VM can't compile) is
        // DOM-free, so `run_scripts` routes it to the `just` ES6 tier; its
        // console output must land in dom.log.
        let doc = html::parse("<html><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        let logs = run_scripts(
            &mut dom,
            &[String::from(
                "function adder(n){ return function(x){ return x + n; }; } \
                 let add10 = adder(10); console.log(add10(5));",
            )],
        );
        assert!(logs.iter().any(|l| l.contains("15")), "{logs:?}");
    }

    #[test_case]
    fn console_log_and_title() {
        let mut doc = html::parse("<html><head><title>Old</title></head><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        let logs = run_scripts(
            &mut dom,
            &[String::from(
                r#"document.title = "New"; console.log("hi", 1+2);"#,
            )],
        );
        assert_eq!(dom.title, "New");
        assert!(logs.iter().any(|l| l.contains("hi") && l.contains("3")), "{logs:?}");
        commit_to_tree(&mut doc.root, &dom);
        let _ = doc;
    }

    #[test_case]
    fn get_element_by_id_sets_text_and_style() {
        let mut doc = html::parse(
            r#"<html><body><p id="msg">hello</p></body></html>"#,
        );
        let mut dom = JsDom::from_document(&doc);
        let _ = run_scripts(
            &mut dom,
            &[String::from(
                r##"
                var el = document.getElementById("msg");
                el.innerText = "world";
                el.style.color = "#ff0000";
                "##,
            )],
        );
        let el = dom.elements.iter().find(|e| e.id.as_deref() == Some("msg")).unwrap();
        assert_eq!(el.text, "world");
        assert!(el.style.contains("color"), "{}", el.style);
        commit_to_tree(&mut doc.root, &dom);
    }

    #[test_case]
    fn window_location_fetch_and_scroll() {
        let doc = html::parse("<html><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        dom.location_href = String::from("https://ex.com/page");
        dom.inner_width = 640;
        let logs = run_scripts(
            &mut dom,
            &[String::from(
                r##"
                console.log(window.innerWidth);
                console.log(location.href);
                var r = fetch("/api");
                console.log(typeof r);
                scrollTo(0, 120);
                encodeURIComponent("a b");
                "##,
            )],
        );
        assert!(logs.iter().any(|l| l.contains("640")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("https://ex.com/page")), "{logs:?}");
        assert_eq!(dom.scroll_to, Some(120));
        assert!(!dom.fetch_log.is_empty(), "fetch should log");
        assert_eq!(dom.fetch_log[0].1, "https://ex.com/api");
    }

    #[test_case]
    fn postmessage_and_json() {
        let doc = html::parse("<html><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        dom.location_href = String::from("https://parent.ex/");
        dom.is_nested = true;
        let _ = run_scripts(
            &mut dom,
            &[String::from(
                r##"
                parent.postMessage("hello", "*");
                postMessage("self-msg", "https://parent.ex");
                console.log(JSON.stringify("x"));
                "##,
            )],
        );
        assert!(
            dom.outbound_messages.iter().any(|m| m.data == "hello" && m.target == "parent"),
            "{:?}",
            dom.outbound_messages
        );
        assert!(
            dom.outbound_messages.iter().any(|m| m.data == "self-msg"),
            "{:?}",
            dom.outbound_messages
        );
    }

    #[test_case]
    fn dom_create_append_query() {
        let doc = html::parse("<html><body><div id=\"root\"></div></body></html>");
        let mut dom = JsDom::from_document(&doc);
        let logs = run_scripts(
            &mut dom,
            &[String::from(
                r##"
                var root = document.getElementById("root");
                var el = document.createElement("span");
                el.className = "hi";
                el.setAttribute("data-x", "1");
                root.appendChild(el);
                console.log(root.childElementCount);
                console.log(document.querySelector(".hi").tagName);
                console.log(el.getAttribute("data-x"));
                console.log(el.classList.contains("hi"));
                "##,
            )],
        );
        assert!(logs.iter().any(|l| l.contains("1")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("SPAN")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("true") || l == "true"), "{logs:?}");
    }

    #[test_case]
    fn function_try_new_bigint_regexp() {
        let doc = html::parse("<html><body></body></html>");
        let mut dom = JsDom::from_document(&doc);
        let logs = run_scripts(
            &mut dom,
            &[String::from(
                r##"
                function add(a, b) { return a + b; }
                console.log(add(2, 3));
                try {
                  throw "boom";
                } catch (e) {
                  console.log(e);
                }
                var o = new Object();
                var arr = new Array(1, 2, 3);
                console.log(typeof add);
                var bi = 10n;
                console.log(typeof bi);
                var re = /foo/;
                console.log(re.test("foobar"));
                var f = (x) => x * 2;
                console.log(f(4));
                "##,
            )],
        );
        assert!(logs.iter().any(|l| l.contains("5")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("boom")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("function")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("bigint")), "{logs:?}");
        assert!(logs.iter().any(|l| l == "true" || l.contains("true")), "{logs:?}");
        assert!(logs.iter().any(|l| l.contains("8")), "{logs:?}");
    }

    #[test_case]
    fn canvas_get_context_fill_rect() {
        let html = r##"<html><body><canvas id="c" width="64" height="48"></canvas>
            <script>
            var el = document.getElementById("c");
            var ctx = el.getContext("2d");
            ctx.fillStyle = "#ff0000";
            ctx.fillRect(0, 0, 10, 10);
            console.log("canvas-ok");
            </script></body></html>"##;
        let frame = crate::browser::render_html(html, 320, 200, 0);
        assert!(
            frame.js_log.iter().any(|l| l.contains("canvas-ok")),
            "{:?}",
            frame.js_log
        );
        // Canvas frame should have red pixels from fillRect.
        let fr = frame
            .layout
            .frames
            .iter()
            .find(|f| f.kind == crate::browser::layout::EmbedKind::Canvas)
            .expect("canvas frame");
        let px = fr.pixels.as_ref().expect("canvas pixels");
        assert!(px.iter().any(|&p| p == 0xff0000), "expected red pixels");
    }
}
