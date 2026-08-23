//! Compact live DOM inside the jsint tenant.
//!
//! The kernel serialises the page's element table + scripts as a `JDOM`
//! envelope. Scripts run against this copy; mutations come back as `JMUT`.
//! Click handlers stay in this process for the page lifetime (Continue/Event).

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use just_engine::parser::JsParser;
use just_engine::runner::ds::error::JErrorType;
use just_engine::runner::ds::value::{JsNumberType, JsValue};
use just_engine::runner::eval::expression::{
    call_value, get_own_prop_value, make_object, set_own_prop, value_is_callable,
};
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::plugin::resolver::PluginResolver;
use just_engine::runner::plugin::types::{EvalContext, NativeProps};

use crate::chunked::{
    get_str, get_u16, get_u32, put_str, put_u16, put_u32, JS_DOM_MAGIC, JS_EV_CLICK,
    JS_EV_INPUT, JS_EV_KEYDOWN, JS_MUT_MAGIC, JS_OP_HIDDEN, JS_OP_INTERACTIVE, JS_OP_LOG,
    JS_OP_NAV, JS_OP_SCROLL, JS_OP_SNAPSHOT, JS_OP_STYLE, JS_OP_TEXT, JS_OP_TITLE,
    JS_OP_VALUE,
};

const DOC_NODE: i64 = -1;
const WIN_NODE: i64 = -2;
const LOC_NODE: i64 = -3;

pub struct Elem {
    pub tag: String,
    pub id: String,
    pub class: String,
    pub text: String,
    pub value: String,
    pub style: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub attrs: Vec<(String, String)>,
    pub hidden: bool,
}

pub struct World {
    pub title: String,
    pub href: String,
    pub inner_w: i32,
    pub inner_h: i32,
    pub elems: Vec<Elem>,
    pub navigate: Option<String>,
    pub scroll: Option<i32>,
    pub extra_log: String,
    pub listeners: Vec<(usize, String, JsValue)>,
    wrappers: RefCell<alloc::collections::BTreeMap<usize, JsValue>>,
    document: JsValue,
    window: JsValue,
    location: JsValue,
}

impl World {
    pub fn unpack(input: &[u8]) -> Option<(Self, Vec<String>)> {
        let mut off = 0usize;
        let mag = get_u32(input, &mut off)?;
        if mag != JS_DOM_MAGIC {
            return None;
        }
        let inner_w = get_u32(input, &mut off)? as i32;
        let inner_h = get_u32(input, &mut off)? as i32;
        let href = get_str(input, &mut off)?.to_string();
        let n = get_u32(input, &mut off)? as usize;
        let mut elems = Vec::with_capacity(n);
        for _ in 0..n {
            let tag = get_str(input, &mut off)?.to_string();
            let id = get_str(input, &mut off)?.to_string();
            let class = get_str(input, &mut off)?.to_string();
            let text = get_str(input, &mut off)?.to_string();
            let value = get_str(input, &mut off)?.to_string();
            let style = get_str(input, &mut off)?.to_string();
            let parent = get_u32(input, &mut off)?;
            let flags = input.get(off).copied()?;
            off += 1;
            let nc = get_u16(input, &mut off)? as usize;
            let mut children = Vec::with_capacity(nc);
            for _ in 0..nc {
                children.push(get_u32(input, &mut off)? as usize);
            }
            let na = get_u16(input, &mut off)? as usize;
            let mut attrs = Vec::with_capacity(na);
            for _ in 0..na {
                let k = get_str(input, &mut off)?.to_string();
                let v = get_str(input, &mut off)?.to_string();
                attrs.push((k, v));
            }
            elems.push(Elem {
                tag,
                id,
                class,
                text,
                value,
                style,
                parent: if parent == u32::MAX { None } else { Some(parent as usize) },
                children,
                attrs,
                hidden: flags & 1 != 0,
            });
        }
        let ns = get_u32(input, &mut off)? as usize;
        let mut scripts = Vec::with_capacity(ns);
        for _ in 0..ns {
            scripts.push(get_str(input, &mut off)?.to_string());
        }
        let document = make_host("document", DOC_NODE);
        let window = make_host("window", WIN_NODE);
        let location = make_host("location", LOC_NODE);
        Some((
            Self {
                title: String::new(),
                href,
                inner_w,
                inner_h,
                elems,
                navigate: None,
                scroll: None,
                extra_log: String::new(),
                listeners: Vec::new(),
                wrappers: RefCell::new(alloc::collections::BTreeMap::new()),
                document,
                window,
                location,
            },
            scripts,
        ))
    }

    fn pack_elem(out: &mut Vec<u8>, e: &Elem) {
        put_str(out, &e.tag);
        put_str(out, &e.id);
        put_str(out, &e.class);
        put_str(out, &e.text);
        put_str(out, &e.value);
        put_str(out, &e.style);
        put_u32(out, e.parent.map(|p| p as u32).unwrap_or(u32::MAX));
        out.push(if e.hidden { 1 } else { 0 });
        put_u16(out, e.children.len() as u16);
        for &c in &e.children {
            put_u32(out, c as u32);
        }
        put_u16(out, e.attrs.len() as u16);
        for (k, v) in &e.attrs {
            put_str(out, k);
            put_str(out, v);
        }
    }

    pub fn pack_mutations(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, JS_MUT_MAGIC);
        let mut ops: Vec<(u8, u32, Vec<u8>)> = Vec::new();
        let mut snap = Vec::new();
        put_u32(&mut snap, self.elems.len() as u32);
        for e in &self.elems {
            Self::pack_elem(&mut snap, e);
        }
        ops.push((JS_OP_SNAPSHOT, 0, snap));
        if !self.title.is_empty() {
            ops.push((JS_OP_TITLE, 0, self.title.as_bytes().to_vec()));
        }
        if let Some(n) = &self.navigate {
            ops.push((JS_OP_NAV, 0, n.as_bytes().to_vec()));
        }
        if let Some(s) = self.scroll {
            ops.push((JS_OP_SCROLL, 0, s.to_le_bytes().to_vec()));
        }
        if !self.extra_log.is_empty() {
            ops.push((JS_OP_LOG, 0, self.extra_log.as_bytes().to_vec()));
        }
        for (i, e) in self.elems.iter().enumerate() {
            if !e.text.is_empty() {
                ops.push((JS_OP_TEXT, i as u32, e.text.as_bytes().to_vec()));
            }
            if !e.value.is_empty() {
                ops.push((JS_OP_VALUE, i as u32, e.value.as_bytes().to_vec()));
            }
            if !e.style.is_empty() {
                ops.push((JS_OP_STYLE, i as u32, e.style.as_bytes().to_vec()));
            }
            if e.hidden {
                ops.push((JS_OP_HIDDEN, i as u32, alloc::vec![1]));
            }
        }
        let mut interactive: Vec<u8> = Vec::new();
        let mut idxs: Vec<u32> = Vec::new();
        for (i, e) in self.elems.iter().enumerate() {
            if e.attrs.iter().any(|(k, _)| k.starts_with("on"))
                || self.listeners.iter().any(|(t, _, _)| *t == i)
            {
                idxs.push(i as u32);
            }
        }
        put_u32(&mut interactive, idxs.len() as u32);
        for i in idxs {
            put_u32(&mut interactive, i);
        }
        ops.push((JS_OP_INTERACTIVE, 0, interactive));
        put_u32(&mut out, ops.len() as u32);
        for (op, idx, payload) in ops {
            out.push(op);
            put_u32(&mut out, idx);
            put_u32(&mut out, payload.len() as u32);
            out.extend_from_slice(&payload);
        }
        out
    }

    fn wrapper(&self, i: usize) -> JsValue {
        let mut map = self.wrappers.borrow_mut();
        if let Some(w) = map.get(&i) {
            return w.clone();
        }
        let w = make_host("Element", i as i64);
        map.insert(i, w.clone());
        w
    }

    fn find_id(&self, id: &str) -> Option<usize> {
        self.elems.iter().position(|e| e.id == id)
    }

    fn matches(e: &Elem, sel: &str) -> bool {
        let sel = sel.trim();
        if sel.is_empty() {
            return false;
        }
        if let Some(id) = sel.strip_prefix('#') {
            return e.id == id;
        }
        if let Some(c) = sel.strip_prefix('.') {
            return e.class.split_whitespace().any(|x| x == c);
        }
        e.tag.eq_ignore_ascii_case(sel)
    }
}

fn make_host(name: &str, node: i64) -> JsValue {
    let o = make_object(vec![]);
    set_own_prop(&o, "__builtin_name__", JsValue::String(name.to_string()), false);
    set_own_prop(
        &o,
        "__native_node__",
        JsValue::Number(JsNumberType::Integer(node)),
        false,
    );
    o
}

fn as_str(v: &JsValue) -> String {
    match v {
        JsValue::String(s) => s.clone(),
        JsValue::Undefined | JsValue::Null => String::new(),
        other => other.to_string(),
    }
}

pub struct DomHost {
    pub world: Rc<RefCell<World>>,
}

fn style_wrapper(i: usize) -> JsValue {
    make_host("Style", STYLE_OFFSET + i as i64)
}

fn empty_elem(tag: &str) -> Elem {
    Elem {
        tag: tag.to_ascii_lowercase(),
        id: String::new(),
        class: String::new(),
        text: String::new(),
        value: String::new(),
        style: String::new(),
        parent: None,
        children: Vec::new(),
        attrs: Vec::new(),
        hidden: false,
    }
}

fn is_ancestor(elems: &[Elem], anc: usize, node: usize) -> bool {
    let mut p = elems.get(node).and_then(|e| e.parent);
    while let Some(i) = p {
        if i == anc {
            return true;
        }
        p = elems.get(i).and_then(|e| e.parent);
    }
    false
}

fn truthy(v: &JsValue) -> bool {
    match v {
        JsValue::Boolean(b) => *b,
        JsValue::Null | JsValue::Undefined => false,
        JsValue::Number(JsNumberType::Integer(0)) => false,
        JsValue::String(s) if s.is_empty() => false,
        _ => true,
    }
}

const STYLE_OFFSET: i64 = 1_000_000;

fn host_fn(name: &str) -> JsValue {
    let f = make_object(vec![]);
    set_own_prop(&f, "__builtin_name__", JsValue::String(name.to_string()), false);
    set_own_prop(&f, "__host_fn__", JsValue::Boolean(true), false);
    f
}

fn html_ctor(name: &str) -> JsValue {
    let c = make_object(vec![]);
    set_own_prop(&c, "name", JsValue::String(name.to_string()), false);
    set_own_prop(&c, "prototype", make_object(vec![]), false);
    c
}

fn is_timer(name: &str) -> bool {
    matches!(
        name,
        "setTimeout"
            | "setInterval"
            | "clearTimeout"
            | "clearInterval"
            | "requestAnimationFrame"
            | "cancelAnimationFrame"
            | "queueMicrotask"
    )
}

fn is_html_iface(name: &str) -> bool {
    matches!(
        name,
        "EventTarget"
            | "Node"
            | "Element"
            | "HTMLElement"
            | "HTMLDocument"
            | "Document"
            | "Window"
            | "HTMLIFrameElement"
            | "HTMLInputElement"
            | "HTMLTextAreaElement"
            | "HTMLSelectElement"
            | "HTMLButtonElement"
            | "HTMLFormElement"
            | "HTMLDivElement"
            | "HTMLSpanElement"
            | "HTMLAnchorElement"
            | "SVGElement"
            | "MutationObserver"
            | "NodeList"
            | "HTMLCollection"
            | "DOMTokenList"
            | "CSSStyleDeclaration"
    )
}

impl NativeProps for DomHost {
    fn get(&self, node: i64, prop: &str) -> Option<JsValue> {
        if node >= STYLE_OFFSET {
            let i = (node - STYLE_OFFSET) as usize;
            let w = self.world.borrow();
            let e = w.elems.get(i)?;
            return Some(match prop {
                "cssText" => JsValue::String(e.style.clone()),
                _ => JsValue::String(String::new()),
            });
        }
        let w = self.world.borrow();
        Some(match (node, prop) {
            (DOC_NODE, "title") => JsValue::String(w.title.clone()),
            (DOC_NODE, "nodeType") => JsValue::Number(JsNumberType::Integer(9)),
            (DOC_NODE, "body") => w
                .elems
                .iter()
                .position(|e| e.tag == "body")
                .map(|i| w.wrapper(i))
                .unwrap_or(JsValue::Null),
            (DOC_NODE, "documentElement") => w
                .elems
                .iter()
                .position(|e| e.tag == "html")
                .map(|i| w.wrapper(i))
                .unwrap_or(JsValue::Null),
            (WIN_NODE, "innerWidth") => JsValue::Number(JsNumberType::Integer(w.inner_w as i64)),
            (WIN_NODE, "innerHeight") => JsValue::Number(JsNumberType::Integer(w.inner_h as i64)),
            (WIN_NODE, "location") => w.location.clone(),
            (WIN_NODE, "document") => w.document.clone(),
            (WIN_NODE, n) if is_timer(n) || n == "getComputedStyle" => host_fn(n),
            (WIN_NODE, n) if is_html_iface(n) => html_ctor(n),
            (LOC_NODE, "href") => JsValue::String(w.href.clone()),
            (n, "innerText" | "textContent" | "innerHTML" | "outerHTML") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::String(e.text.clone()))
                .unwrap_or(JsValue::Undefined),
            (n, "value") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::String(e.value.clone()))
                .unwrap_or(JsValue::Undefined),
            (n, "id") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::String(e.id.clone()))
                .unwrap_or(JsValue::Undefined),
            (n, "className") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::String(e.class.clone()))
                .unwrap_or(JsValue::Undefined),
            (n, "tagName" | "nodeName") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::String(e.tag.to_ascii_uppercase()))
                .unwrap_or(JsValue::Undefined),
            (n, "nodeType") if n >= 0 => w.elems.get(n as usize).map(|e| {
                JsValue::Number(JsNumberType::Integer(match e.tag.as_str() {
                    "#text" => 3,
                    "#comment" => 8,
                    _ => 1,
                }))
            }).unwrap_or(JsValue::Undefined),
            (n, "childElementCount") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::Number(JsNumberType::Integer(e.children.len() as i64)))
                .unwrap_or(JsValue::Undefined),
            (n, "children") if n >= 0 => {
                let kids = w.elems.get(n as usize).map(|e| e.children.clone()).unwrap_or_default();
                just_engine::runner::eval::expression::make_array(
                    kids.into_iter().map(|i| w.wrapper(i)).collect(),
                )
            }
            (n, "firstChild" | "firstElementChild") if n >= 0 => w
                .elems
                .get(n as usize)
                .and_then(|e| e.children.first().copied())
                .map(|i| w.wrapper(i))
                .unwrap_or(JsValue::Null),
            (n, "lastChild" | "lastElementChild") if n >= 0 => w
                .elems
                .get(n as usize)
                .and_then(|e| e.children.last().copied())
                .map(|i| w.wrapper(i))
                .unwrap_or(JsValue::Null),
            (n, "parentNode" | "parentElement") if n >= 0 => w
                .elems
                .get(n as usize)
                .and_then(|e| e.parent)
                .map(|i| w.wrapper(i))
                .unwrap_or(JsValue::Null),
            (n, "style") if n >= 0 => style_wrapper(n as usize),
            (n, "hidden") if n >= 0 => w
                .elems
                .get(n as usize)
                .map(|e| JsValue::Boolean(e.hidden))
                .unwrap_or(JsValue::Undefined),
            (n, "ownerDocument") if n >= 0 => w.document.clone(),
            (n, "constructor") if n >= 0 => {
                let c = make_object(vec![]);
                set_own_prop(&c, "name", JsValue::String(String::from("HTMLElement")), false);
                set_own_prop(&c, "prototype", make_object(vec![]), false);
                c
            }
            _ => return None,
        })
    }

    fn set(&self, node: i64, prop: &str, value: JsValue) -> bool {
        if node >= STYLE_OFFSET {
            let i = (node - STYLE_OFFSET) as usize;
            let mut w = self.world.borrow_mut();
            let Some(e) = w.elems.get_mut(i) else { return false };
            if prop == "cssText" {
                e.style = as_str(&value);
            } else {
                if !e.style.is_empty() && !e.style.ends_with(';') {
                    e.style.push(';');
                }
                e.style.push_str(prop);
                e.style.push(':');
                e.style.push_str(&as_str(&value));
                e.style.push(';');
            }
            return true;
        }
        let mut w = self.world.borrow_mut();
        match (node, prop) {
            (DOC_NODE, "title") => {
                w.title = as_str(&value);
                true
            }
            (LOC_NODE, "href") => {
                w.navigate = Some(as_str(&value));
                true
            }
            (n, "innerText" | "textContent" | "innerHTML") if n >= 0 => {
                let i = n as usize;
                if i >= w.elems.len() {
                    return false;
                }
                let kids = core::mem::take(&mut w.elems[i].children);
                w.elems[i].text = as_str(&value);
                for c in kids {
                    if let Some(ch) = w.elems.get_mut(c) {
                        ch.parent = None;
                    }
                }
                true
            }
            (n, "value") if n >= 0 => {
                if let Some(e) = w.elems.get_mut(n as usize) {
                    e.value = as_str(&value);
                }
                true
            }
            (n, "id") if n >= 0 => {
                if let Some(e) = w.elems.get_mut(n as usize) {
                    e.id = as_str(&value);
                }
                true
            }
            (n, "className") if n >= 0 => {
                if let Some(e) = w.elems.get_mut(n as usize) {
                    e.class = as_str(&value);
                }
                true
            }
            (n, "style") if n >= 0 => {
                if let Some(e) = w.elems.get_mut(n as usize) {
                    e.style = as_str(&value);
                }
                true
            }
            (n, "hidden") if n >= 0 => {
                if let Some(e) = w.elems.get_mut(n as usize) {
                    e.hidden = truthy(&value);
                }
                true
            }
            _ => false,
        }
    }
}

impl PluginResolver for DomHost {
    fn has_binding(&self, name: &str) -> bool {
        matches!(
            name,
            "document"
                | "window"
                | "self"
                | "globalThis"
                | "location"
                | "Element"
                | "Style"
                | "Event"
                | "getComputedStyle"
        ) || is_timer(name)
            || is_html_iface(name)
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        let w = self.world.borrow();
        Ok(match name {
            "document" => w.document.clone(),
            "window" | "self" | "globalThis" => w.window.clone(),
            "location" => w.location.clone(),
            n if is_timer(n) || n == "getComputedStyle" => host_fn(n),
            n if is_html_iface(n) => html_ctor(n),
            _ => JsValue::Undefined,
        })
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        let a1 = args.get(1).cloned().unwrap_or(JsValue::Undefined);
        let node = match get_own_prop_value(&this, "__native_node__") {
            Some(JsValue::Number(JsNumberType::Integer(n))) => n,
            _ if object_name == "document" => DOC_NODE,
            _ if object_name == "window" => WIN_NODE,
            _ => -99,
        };
        let elem = if node >= 0 && node < STYLE_OFFSET {
            Some(node as usize)
        } else {
            None
        };

        if is_timer(method_name) {
            if matches!(
                method_name,
                "setTimeout" | "setInterval" | "requestAnimationFrame" | "queueMicrotask"
            ) && value_is_callable(&a0)
            {
                let _ = call_value(&a0, JsValue::Undefined, vec![], ctx);
            }
            return Some(Ok(JsValue::Number(JsNumberType::Integer(0))));
        }

        if node == DOC_NODE && method_name == "getElementById" {
            let id = as_str(&a0);
            let w = self.world.borrow();
            return Some(Ok(w.find_id(&id).map(|i| w.wrapper(i)).unwrap_or(JsValue::Null)));
        }
        if node == DOC_NODE && (method_name == "querySelector" || method_name == "querySelectorAll")
        {
            let sel = as_str(&a0);
            let w = self.world.borrow();
            let mut hits = Vec::new();
            for (i, e) in w.elems.iter().enumerate() {
                if World::matches(e, &sel) {
                    hits.push(w.wrapper(i));
                    if method_name == "querySelector" {
                        break;
                    }
                }
            }
            if method_name == "querySelector" {
                return Some(Ok(hits.into_iter().next().unwrap_or(JsValue::Null)));
            }
            return Some(Ok(just_engine::runner::eval::expression::make_array(hits)));
        }
        if node == DOC_NODE && method_name == "createElement" {
            let tag = as_str(&a0);
            let mut w = self.world.borrow_mut();
            w.elems.push(empty_elem(&tag));
            let i = w.elems.len() - 1;
            return Some(Ok(w.wrapper(i)));
        }
        if node == DOC_NODE && method_name == "createElementNS" {
            let tag = as_str(&a1);
            let mut w = self.world.borrow_mut();
            w.elems.push(empty_elem(&tag));
            let i = w.elems.len() - 1;
            return Some(Ok(w.wrapper(i)));
        }
        if node == DOC_NODE && method_name == "createTextNode" {
            let mut w = self.world.borrow_mut();
            let mut e = empty_elem("#text");
            e.text = as_str(&a0);
            w.elems.push(e);
            let i = w.elems.len() - 1;
            return Some(Ok(w.wrapper(i)));
        }
        if method_name == "appendChild" {
            if let Some(p) = elem {
                if let Some(JsValue::Number(JsNumberType::Integer(c))) =
                    get_own_prop_value(&a0, "__native_node__")
                {
                    if c >= 0 {
                        let c = c as usize;
                        let mut w = self.world.borrow_mut();
                        if p < w.elems.len()
                            && c < w.elems.len()
                            && p != c
                            && !is_ancestor(&w.elems, c, p)
                        {
                            if let Some(old) = w.elems[c].parent {
                                if old < w.elems.len() {
                                    w.elems[old].children.retain(|&x| x != c);
                                }
                            }
                            w.elems[p].children.retain(|&x| x != c);
                            w.elems[p].children.push(c);
                            w.elems[c].parent = Some(p);
                        }
                    }
                }
            }
            return Some(Ok(a0));
        }
        if method_name == "insertBefore" {
            if let Some(p) = elem {
                if let Some(JsValue::Number(JsNumberType::Integer(c))) =
                    get_own_prop_value(&a0, "__native_node__")
                {
                    if c >= 0 {
                        let c = c as usize;
                        let ref_i = match get_own_prop_value(&a1, "__native_node__") {
                            Some(JsValue::Number(JsNumberType::Integer(r))) if r >= 0 => {
                                Some(r as usize)
                            }
                            _ => None,
                        };
                        let mut w = self.world.borrow_mut();
                        if p < w.elems.len()
                            && c < w.elems.len()
                            && p != c
                            && !is_ancestor(&w.elems, c, p)
                        {
                            if let Some(old) = w.elems[c].parent {
                                if old < w.elems.len() {
                                    w.elems[old].children.retain(|&x| x != c);
                                }
                            }
                            w.elems[p].children.retain(|&x| x != c);
                            let at = ref_i
                                .and_then(|r| w.elems[p].children.iter().position(|&x| x == r))
                                .unwrap_or(w.elems[p].children.len());
                            w.elems[p].children.insert(at, c);
                            w.elems[c].parent = Some(p);
                        }
                    }
                }
            }
            return Some(Ok(a0));
        }
        if method_name == "removeChild" {
            if let Some(p) = elem {
                if let Some(JsValue::Number(JsNumberType::Integer(c))) =
                    get_own_prop_value(&a0, "__native_node__")
                {
                    if c >= 0 {
                        let c = c as usize;
                        let mut w = self.world.borrow_mut();
                        if p < w.elems.len() {
                            w.elems[p].children.retain(|&x| x != c);
                        }
                        if c < w.elems.len() {
                            w.elems[c].parent = None;
                        }
                    }
                }
            }
            return Some(Ok(a0));
        }
        if method_name == "setAttribute" {
            if let Some(i) = elem {
                let (k, v) = (as_str(&a0), as_str(&a1));
                let mut w = self.world.borrow_mut();
                if let Some(e) = w.elems.get_mut(i) {
                    match k.as_str() {
                        "id" => e.id = v.clone(),
                        "class" => e.class = v.clone(),
                        _ => {}
                    }
                    if let Some(slot) = e.attrs.iter_mut().find(|(ak, _)| ak == &k) {
                        slot.1 = v;
                    } else {
                        e.attrs.push((k, v));
                    }
                }
            }
            return Some(Ok(JsValue::Undefined));
        }
        if method_name == "getAttribute" {
            if let Some(i) = elem {
                let k = as_str(&a0);
                let w = self.world.borrow();
                let v = w.elems.get(i).and_then(|e| match k.as_str() {
                    "id" if !e.id.is_empty() => Some(e.id.clone()),
                    "class" if !e.class.is_empty() => Some(e.class.clone()),
                    _ => e.attrs.iter().find(|(ak, _)| ak == &k).map(|(_, v)| v.clone()),
                });
                return Some(Ok(v.map(JsValue::String).unwrap_or(JsValue::Null)));
            }
        }
        if method_name == "hasAttribute" {
            if let Some(i) = elem {
                let k = as_str(&a0);
                let w = self.world.borrow();
                let has = w.elems.get(i).map(|e| match k.as_str() {
                    "id" => !e.id.is_empty(),
                    "class" => !e.class.is_empty(),
                    _ => e.attrs.iter().any(|(ak, _)| ak == &k),
                }).unwrap_or(false);
                return Some(Ok(JsValue::Boolean(has)));
            }
        }
        if method_name == "removeAttribute" {
            if let Some(i) = elem {
                let k = as_str(&a0);
                let mut w = self.world.borrow_mut();
                if let Some(e) = w.elems.get_mut(i) {
                    e.attrs.retain(|(ak, _)| ak != &k);
                }
            }
            return Some(Ok(JsValue::Undefined));
        }
        if method_name == "addEventListener" && (elem.is_some() || node == DOC_NODE || node == WIN_NODE)
        {
            let ty = as_str(&a0);
            let cb = a1;
            let t = elem.unwrap_or(usize::MAX);
            self.world.borrow_mut().listeners.push((t, ty, cb));
            return Some(Ok(JsValue::Undefined));
        }
        if method_name == "querySelector" || method_name == "querySelectorAll" {
            if let Some(root) = elem {
                let sel = as_str(&a0);
                let w = self.world.borrow();
                let mut hits = Vec::new();
                fn walk(w: &World, i: usize, sel: &str, all: bool, hits: &mut Vec<JsValue>) {
                    if let Some(e) = w.elems.get(i) {
                        for &c in &e.children {
                            if w.elems.get(c).map(|e| World::matches(e, sel)).unwrap_or(false) {
                                hits.push(w.wrapper(c));
                                if !all {
                                    return;
                                }
                            }
                            walk(w, c, sel, all, hits);
                            if !all && !hits.is_empty() {
                                return;
                            }
                        }
                    }
                }
                walk(&w, root, &sel, method_name == "querySelectorAll", &mut hits);
                if method_name == "querySelector" {
                    return Some(Ok(hits.into_iter().next().unwrap_or(JsValue::Null)));
                }
                return Some(Ok(just_engine::runner::eval::expression::make_array(hits)));
            }
        }
        if method_name == "hasOwnProperty" {
            return Some(Ok(JsValue::Boolean(
                get_own_prop_value(&this, &as_str(&a0)).is_some(),
            )));
        }
        if method_name == "closest" || method_name == "matches" {
            if let Some(start) = elem {
                let sel = as_str(&a0);
                let w = self.world.borrow();
                let only_self = method_name == "matches";
                let mut cur = Some(start);
                let mut found = if only_self {
                    JsValue::Boolean(false)
                } else {
                    JsValue::Null
                };
                while let Some(i) = cur {
                    let Some(e) = w.elems.get(i) else { break };
                    if World::matches(e, &sel) {
                        found = if only_self {
                            JsValue::Boolean(true)
                        } else {
                            w.wrapper(i)
                        };
                        break;
                    }
                    if only_self {
                        break;
                    }
                    cur = e.parent;
                }
                return Some(Ok(found));
            }
        }
        if method_name == "getComputedStyle" {
            let i = match get_own_prop_value(&a0, "__native_node__") {
                Some(JsValue::Number(JsNumberType::Integer(n))) if n >= 0 => n as usize,
                _ => 0,
            };
            return Some(Ok(style_wrapper(i)));
        }
        let _ = ctx;
        None
    }

    fn has_method(&self, object_name: &str, method_name: &str) -> bool {
        matches!(
            (object_name, method_name),
            ("document", "getElementById")
                | ("document", "querySelector")
                | ("document", "querySelectorAll")
                | ("document", "createElement")
                | ("document", "createElementNS")
                | ("document", "createTextNode")
                | ("document", "addEventListener")
                | ("Element", "appendChild")
                | ("Element", "insertBefore")
                | ("Element", "removeChild")
                | ("Element", "setAttribute")
                | ("Element", "getAttribute")
                | ("Element", "hasAttribute")
                | ("Element", "removeAttribute")
                | ("Element", "addEventListener")
                | ("Element", "querySelector")
                | ("Element", "querySelectorAll")
                | ("Element", "hasOwnProperty")
                | ("Element", "closest")
                | ("Element", "matches")
                | ("window", "addEventListener")
                | ("window", "getComputedStyle")
                | ("window", "setTimeout")
                | ("window", "queueMicrotask")
                | ("window", "requestAnimationFrame")
        )
    }

    fn call_constructor(
        &self,
        object_name: &str,
        ctx: &mut EvalContext,
        _this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        if is_timer(object_name) {
            if matches!(
                object_name,
                "setTimeout" | "setInterval" | "requestAnimationFrame" | "queueMicrotask"
            ) && value_is_callable(&a0)
            {
                let _ = call_value(&a0, JsValue::Undefined, vec![], ctx);
            }
            return Some(Ok(JsValue::Number(JsNumberType::Integer(0))));
        }
        if object_name == "getComputedStyle" {
            let i = match get_own_prop_value(&a0, "__native_node__") {
                Some(JsValue::Number(JsNumberType::Integer(n))) if n >= 0 => n as usize,
                _ => 0,
            };
            return Some(Ok(style_wrapper(i)));
        }
        None
    }

    fn name(&self) -> &str {
        "dom"
    }
}

pub fn dispatch(world: &Rc<RefCell<World>>, ctx: &mut EvalContext, ty: u64, target: usize, _x: i32, _y: i32) {
    let ev = match ty {
        JS_EV_CLICK => "click",
        JS_EV_INPUT => "input",
        JS_EV_KEYDOWN => "keydown",
        _ => "click",
    };
    let onclick = {
        let w = world.borrow();
        w.elems.get(target).and_then(|e| {
            e.attrs
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("onclick") || k.eq_ignore_ascii_case(&format!("on{ev}")))
                .map(|(_, v)| v.clone())
        })
    };
    if let Some(src) = onclick {
        if let Ok(ast) = JsParser::parse_to_ast_from_str(&src) {
            for stmt in &ast.body {
                let _ = execute_statement(stmt, ctx);
            }
        }
    }
    let cbs: Vec<JsValue> = world
        .borrow()
        .listeners
        .iter()
        .filter(|(t, k, _)| *t == target && k == ev)
        .map(|(_, _, c)| c.clone())
        .collect();
    for cb in cbs {
        if value_is_callable(&cb) {
            let _ = call_value(&cb, JsValue::Undefined, vec![], ctx);
        }
    }
}
