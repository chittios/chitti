//! A userspace tenant: runs in **ring 3 / EL0**, loaded by `kernel/src/synapse/tenant.rs`.
//!
//! This is where attacker-facing **page JavaScript** belongs — the pest/just parser
//! and tree-walking interpreter. A malformed script becomes a status word from a
//! tenant the kernel discards, instead of a wild write in ring 0.
//!
//! Parse + `EvalContext` + `execute_statement`. A `JDOM` envelope carries the
//! page's element table so `document`/`window` bindings run here too; click
//! [`ChunkCmd::Event`]s keep the same session so listeners survive.
//!
//! `fetch` is stubbed (empty / error JSON); a host round-trip is v1.1. Policy stays
//! in the kernel when that is wired.
//!
//! # Contract with the loader
//!
//! Entered at `_start` with the startup page in the C ABI's first argument register.
//! The page is [`chunked::ChunkBlock`] (cmd/status, input, units, output, heap,
//! `kind`/`meta*`). This tenant writes `kind = KIND_JS` (3). `fetch` is stubbed, so
//! it never asks for a host-op via `meta0`.
//!
//! **This tenant is stateful.** [`ChunkCmd::Init`] / [`ChunkCmd::Cancel`] reset the
//! bump cursor; [`ChunkCmd::Continue`] must not — the ASTs and `EvalContext` live
//! in the arena, and closures hold `*const` pointers into that heap.
//!
//! Input (Init): a length-prefixed envelope (`u32le n`, then `n` times `u32le len`
//! + bytes), or the whole buffer as one UTF-8 script.
//!
//! Output: UTF-8 diagnostics — optional `title=` line, `console.*` lines, and a
//! `value=` line for the last completion. Reported via `output_ptr`/`output_len`
//! (heap-output: the string lives in the arena).

#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

extern crate alloc;

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;

use just_engine::parser::ast::ProgramData;
use just_engine::parser::JsParser;
use just_engine::runner::ds::error::JErrorType;
use just_engine::runner::ds::value::{JsNumberType, JsValue};
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::eval::types::CompletionType;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::resolver::PluginResolver;
use just_engine::runner::plugin::types::EvalContext;
use just_engine::runner::std_lib::console::drain_console_log;

/// Shared chunked-tenant ABI. Same file the kernel mounts, so the command/status
/// vocabulary cannot drift.
#[allow(dead_code)]
#[path = "../../../kernel/src/synapse/chunked.rs"]
mod chunked;

/// HTML tokenizer — same source as the kernel, so a port cannot regress it.
#[allow(dead_code)]
#[path = "../../../kernel/src/browser/elements.rs"]
mod elements;
#[allow(dead_code)]
#[path = "../../../kernel/src/browser/html.rs"]
mod html;

mod dom;

use chunked::{ChunkBlock, ChunkCmd, ChunkStatus, KIND_JS};

/// ABI entry numbers, from `synapse::abi::Entry`.
const ENTRY_EXIT: u64 = 2;

/// Longest a single log line may be (page-controlled text).
const MAX_LOG_LINE: usize = 240;

/// Leave EL0 for good. Never returns.
fn exit() -> ! {
    // SAFETY: the one ABI call this tenant makes. Reply registers are zeroed because a
    // trap clobbers the caller-saved set, and handing the kernel stale values as a reply
    // buffer is the defect that once had a tenant overwrite its own code page.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!(
            "syscall",
            in("rdi") ENTRY_EXIT, in("r8") 0u64, in("r9") 0u64,
            options(noreturn, nostack)
        );
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!(
            "svc #0",
            in("x0") ENTRY_EXIT, in("x3") 0u64, in("x4") 0u64,
            options(noreturn, nostack)
        );
    }
}

// ---------------------------------------------------------------------------
// Bump arena — copied from imgdec, with a different reset rule.
// ---------------------------------------------------------------------------

static mut HEAP_BASE: usize = 0;
static mut HEAP_LEN_BYTES: usize = 0;
/// Bytes handed out so far. **Reset on Init/Cancel only.** Continue must leave
/// this alone: the session's ASTs and `EvalContext` sit in the arena, and
/// closures hold pointers into that heap (`js_just` documents the same rule).
static mut CURSOR: usize = 0;
static mut OOM: bool = false;
static mut BLOCK: *mut u8 = core::ptr::null_mut();

struct Bump;

// SAFETY: single-threaded tenant. `alloc` hands out disjoint aligned slices from
// the loader's heap; the only reuse is of the top allocation. Every returned
// pointer is inside `[HEAP_BASE, HEAP_BASE + HEAP_LEN_BYTES)`.
unsafe impl core::alloc::GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: core::alloc::Layout) -> *mut u8 {
        unsafe {
            let base = HEAP_BASE;
            let start = (base + CURSOR).next_multiple_of(l.align().max(1));
            let end = start.saturating_sub(base).saturating_add(l.size());
            if base == 0 || end > HEAP_LEN_BYTES {
                OOM = true;
                return core::ptr::null_mut();
            }
            CURSOR = end;
            start as *mut u8
        }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: core::alloc::Layout) {
        unsafe {
            if HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR {
                CURSOR = (p as usize) - HEAP_BASE;
            }
        }
    }

    unsafe fn realloc(&self, p: *mut u8, l: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let top = HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR;
            if top {
                let end = (p as usize).saturating_sub(HEAP_BASE).saturating_add(new_size);
                if end <= HEAP_LEN_BYTES {
                    CURSOR = end;
                    return p;
                }
                OOM = true;
                return core::ptr::null_mut();
            }
            let new = self.alloc(core::alloc::Layout::from_size_align_unchecked(new_size, l.align()));
            if !new.is_null() {
                core::ptr::copy_nonoverlapping(p, new, l.size().min(new_size));
                self.dealloc(p, l);
            }
            new
        }
    }
}

#[global_allocator]
static ALLOC: Bump = Bump;

#[alloc_error_handler]
fn on_alloc_failure(_layout: core::alloc::Layout) -> ! {
    // SAFETY: single-threaded; `BLOCK` is the loader's page or null.
    unsafe {
        if !BLOCK.is_null() {
            let want = (HEAP_LEN_BYTES as u64).saturating_mul(2).max(arena_estimate(0));
            finish(BLOCK, ChunkStatus::OutOfMemory, 0, 0, 0, want);
        }
    }
    exit()
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        if !BLOCK.is_null() {
            let status = if OOM { ChunkStatus::OutOfMemory } else { ChunkStatus::Failed };
            let want = if OOM {
                (HEAP_LEN_BYTES as u64).saturating_mul(2).max(arena_estimate(0))
            } else {
                0
            };
            finish(BLOCK, status, 0, 0, 0, want);
        }
    }
    exit()
}

/// # Safety
/// `blk` must point at the startup page, mapped read-write for this run.
unsafe fn read_block(blk: *mut u8) -> ChunkBlock {
    let mut header = [0u8; ChunkBlock::BYTES];
    unsafe {
        core::ptr::copy_nonoverlapping(blk, header.as_mut_ptr(), ChunkBlock::BYTES);
    }
    ChunkBlock::unpack(&header).unwrap_or_else(|| ChunkBlock::new_init(0, 0, 0))
}

/// # Safety
/// As [`read_block`]. Does not change the loader's heap mapping fields unless
/// `b` itself does — callers copy the live block first.
unsafe fn write_block(blk: *mut u8, b: &ChunkBlock) {
    let mut out = [0u8; ChunkBlock::BYTES];
    let _ = b.pack(&mut out);
    unsafe {
        core::ptr::copy_nonoverlapping(out.as_ptr(), blk, ChunkBlock::BYTES);
    }
}

/// Write a result. Preserves `heap_ptr`/`heap_len` (loader-owned) and input.
unsafe fn finish(
    blk: *mut u8,
    status: ChunkStatus,
    units_done: u64,
    output_ptr: u64,
    output_len: u64,
    heap_want: u64,
) {
    let mut b = unsafe { read_block(blk) };
    b.cmd_or_status = status as u64;
    b.units_done = units_done;
    b.output_ptr = output_ptr;
    b.output_len = output_len;
    b.heap_want = heap_want;
    b.kind = KIND_JS;
    unsafe { write_block(blk, &b) }
}

fn arena_estimate(input_len: usize) -> u64 {
    (input_len as u64).saturating_mul(32).saturating_add(16 << 20)
}

// ---------------------------------------------------------------------------
// Persistent JS session — field order is load-bearing.
// ---------------------------------------------------------------------------

/// The live JS world. **`ctx` must drop before `asts`**: `SimpleFunctionObject`s
/// hold `*const` pointers into the AST heap those vectors own (same rule as
/// `kernel/src/browser/js_just.rs`'s `JsPage`).
struct Session {
    ctx: EvalContext,
    asts: Vec<ProgramData>,
    next: usize,
    log: String,
    last: String,
    /// Rendered bytes the kernel reads back (UTF-8 log or JMUT). Lives as long as the session.
    output: Vec<u8>,
    world: Option<Rc<RefCell<dom::World>>>,
}

static mut SESSION: Option<Session> = None;

/// `fetch` is not a host round-trip in v1. Page JS that calls it gets a string
/// error body — policy stays in the kernel when the real fetch is wired.
struct FetchStub;

impl PluginResolver for FetchStub {
    fn has_binding(&self, name: &str) -> bool {
        name == "fetch"
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        if name == "fetch" {
            Ok(JsValue::String(String::from("{\"error\":\"fetch stub\"}")))
        } else {
            Ok(JsValue::Undefined)
        }
    }

    fn call_method(
        &self,
        _object_name: &str,
        _method_name: &str,
        _ctx: &mut EvalContext,
        _this: JsValue,
        _args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        None
    }

    fn name(&self) -> &str {
        "fetch_stub"
    }
}

fn parse_scripts(input: &[u8]) -> Result<Vec<String>, String> {
    if let Some(v) = try_envelope(input) {
        return Ok(v);
    }
    let s = core::str::from_utf8(input).map_err(|_| String::from("input is not UTF-8"))?;
    Ok(alloc::vec![s.to_string()])
}

/// Length-prefixed envelope: `u32le n` then `n` × (`u32le len` + bytes).
/// Refused (returns `None`, caller treats the whole buffer as one script) unless
/// `n` is in 1..=4096 and the sizes consume the buffer exactly.
fn try_envelope(input: &[u8]) -> Option<Vec<String>> {
    if input.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes(input[0..4].try_into().ok()?) as usize;
    if n == 0 || n > 4096 {
        return None;
    }
    let mut off = 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if off + 4 > input.len() {
            return None;
        }
        let len = u32::from_le_bytes(input[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let end = off.checked_add(len)?;
        if end > input.len() {
            return None;
        }
        let s = core::str::from_utf8(&input[off..end]).ok()?;
        out.push(s.to_string());
        off = end;
    }
    if off != input.len() {
        return None;
    }
    Some(out)
}

fn display_value(v: &JsValue) -> String {
    match v {
        JsValue::Undefined => String::from("undefined"),
        JsValue::Null => String::from("null"),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::String(s) => format!("\"{}\"", s),
        JsValue::Number(n) => match n {
            JsNumberType::Integer(i) => i.to_string(),
            JsNumberType::Float(f) => f.to_string(),
            JsNumberType::NaN => String::from("NaN"),
            JsNumberType::PositiveInfinity => String::from("Infinity"),
            JsNumberType::NegativeInfinity => String::from("-Infinity"),
        },
        other => other.to_string(),
    }
}

fn log_line(prefix: &str, body: &str) -> String {
    let mut out = String::from(prefix);
    if body.chars().count() <= MAX_LOG_LINE {
        out.push_str(body);
        return out;
    }
    out.extend(body.chars().take(MAX_LOG_LINE));
    out.push_str(" … (truncated)");
    out
}

fn eval_one(session: &mut Session) {
    let i = session.next;
    let Some(ast) = session.asts.get(i) else {
        return;
    };
    just_engine::runner::eval::statement::hoist_var_declarations(&ast.body, &mut session.ctx);
    for stmt in &ast.body {
        match execute_statement(stmt, &mut session.ctx) {
            Ok(completion) => {
                if let Some(v) = completion.value {
                    session.last = display_value(&v);
                }
                if completion.completion_type == CompletionType::Return {
                    break;
                }
            }
            Err(e) => {
                session.log.push_str(&log_line("Uncaught ", &e.to_string()));
                session.log.push('\n');
                break;
            }
        }
    }
    for line in drain_console_log() {
        session.log.push_str(&log_line("", &line));
        session.log.push('\n');
    }
    session.next = i + 1;
}

fn render(session: &mut Session) {
    if let Some(w) = &session.world {
        w.borrow_mut().extra_log = session.log.clone();
        session.output = w.borrow().pack_mutations();
        return;
    }
    let mut s = String::new();
    if let Ok(JsValue::String(t)) = session.ctx.get_binding("title") {
        if !t.is_empty() {
            s.push_str("title=");
            s.push_str(&t);
            s.push('\n');
        }
    }
    s.push_str(&session.log);
    if !session.last.is_empty() {
        s.push_str("value=");
        s.push_str(&session.last);
        s.push('\n');
    }
    session.output = s.into_bytes();
}

fn report(blk: *mut u8, session: &Session, status: ChunkStatus, want: u64) {
    unsafe {
        finish(
            blk,
            status,
            session.next as u64,
            session.output.as_ptr() as u64,
            session.output.len() as u64,
            want,
        );
    }
}

fn run_units(session: &mut Session, max_units: u64) {
    let n = max_units.max(1) as usize;
    for _ in 0..n {
        if session.next >= session.asts.len() {
            break;
        }
        eval_one(session);
    }
    render(session);
}

fn status_after(session: &Session) -> ChunkStatus {
    if session.next >= session.asts.len() {
        ChunkStatus::Done
    } else {
        ChunkStatus::NeedMore
    }
}

fn boot_session(sources: Vec<String>) -> Result<Session, String> {
    let _ = drain_console_log();
    let mut asts = Vec::with_capacity(sources.len());
    let mut log = String::new();
    for src in &sources {
        match JsParser::parse_to_ast_from_str(src) {
            Ok(ast) => asts.push(ast),
            Err(e) => {
                let msg = format!("{:?}", e);
                log.push_str(&log_line("Uncaught SyntaxError: ", &msg));
                log.push('\n');
            }
        }
    }
    if asts.is_empty() && !sources.is_empty() {
        return Err(if log.is_empty() {
            String::from("no scripts parsed")
        } else {
            log
        });
    }
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.add_resolver(alloc::boxed::Box::new(FetchStub));
    Ok(Session {
        ctx,
        asts,
        next: 0,
        log,
        last: String::new(),
        output: Vec::new(),
        world: None,
    })
}

fn boot_from_input(input: &[u8]) -> Result<Session, String> {
    if let Some((world, sources)) = dom::World::from_html(input) {
        return boot_world(world, sources);
    }
    if let Some((world, sources)) = dom::World::unpack(input) {
        return boot_world(world, sources);
    }
    parse_scripts(input).and_then(boot_session)
}

fn boot_world(world: dom::World, sources: Vec<String>) -> Result<Session, String> {
    let world = Rc::new(RefCell::new(world));
    let mut session = boot_session(sources)?;
    session.ctx.native_props = Some(Rc::new(dom::DomHost { world: world.clone() }));
    session.ctx.add_resolver(alloc::boxed::Box::new(dom::DomHost { world: world.clone() }));
    session.world = Some(world);
    Ok(session)
}

fn fail_text(blk: *mut u8, msg: &str, want: u64) {
    // A one-shot diagnostic in the arena. The kernel copies it before the next
    // Init; we do not keep a session.
    let mut s = String::from(msg);
    if !s.ends_with('\n') {
        s.push('\n');
    }
    let ptr = s.as_ptr() as u64;
    let len = s.len() as u64;
    core::mem::forget(s);
    unsafe {
        finish(blk, ChunkStatus::Failed, 0, ptr, len, want);
    }
}

fn adopt_heap(blk: *mut u8) {
    // SAFETY: `blk` is the loader's page; `heap_ptr`/`heap_len` name mapped RW memory.
    unsafe {
        let b = read_block(blk);
        HEAP_BASE = b.heap_ptr as usize;
        HEAP_LEN_BYTES = b.heap_len as usize;
    }
}

fn reset_arena(blk: *mut u8) {
    // Drop the session **before** rolling the cursor: Drop walks heap objects.
    // SAFETY: single-threaded tenant; `addr_of_mut` rather than a place-ref to
    // the static (edition 2024 denies `static_mut_refs`).
    unsafe {
        *core::ptr::addr_of_mut!(SESSION) = None;
        adopt_heap(blk);
        CURSOR = 0;
        OOM = false;
    }
}

fn session_slot() -> *mut Option<Session> {
    // SAFETY: single-threaded tenant; the pointer is only deref'd on this core.
    unsafe { core::ptr::addr_of_mut!(SESSION) }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start(blk: *mut u8) -> ! {
    unsafe {
        BLOCK = blk;
        let chunk = read_block(blk);
        let cmd = match chunk.cmd() {
            Some(c) => c,
            None => {
                finish(blk, ChunkStatus::Failed, 0, 0, 0, 0);
                exit();
            }
        };

        match cmd {
            ChunkCmd::Init => {
                reset_arena(blk);
                let want = arena_estimate(chunk.input_len as usize);
                {
                    let mut b = read_block(blk);
                    b.heap_want = want;
                    b.kind = KIND_JS;
                    write_block(blk, &b);
                }
                if HEAP_BASE == 0 || HEAP_LEN_BYTES == 0 {
                    finish(blk, ChunkStatus::OutOfMemory, 0, 0, 0, want);
                    exit();
                }
                let input = core::slice::from_raw_parts(
                    chunk.input_ptr as *const u8,
                    chunk.input_len as usize,
                );
                match boot_from_input(input) {
                    Ok(mut session) => {
                        run_units(&mut session, chunk.max_units);
                        let st = status_after(&session);
                        report(blk, &session, st, want);
                        *session_slot() = Some(session);
                    }
                    Err(msg) => fail_text(blk, &msg, want),
                }
            }
            ChunkCmd::Continue => {
                // Heap may have grown in place; the cursor and the session stay.
                adopt_heap(blk);
                OOM = false;
                match (*session_slot()).as_mut() {
                    Some(session) => {
                        run_units(session, chunk.max_units);
                        let st = status_after(session);
                        report(blk, session, st, 0);
                    }
                    None => fail_text(blk, "Continue without Init", 0),
                }
            }
            ChunkCmd::Event => {
                adopt_heap(blk);
                OOM = false;
                match (*session_slot()).as_mut() {
                    Some(session) => {
                        if let Some(w) = &session.world {
                            dom::dispatch(
                                w,
                                &mut session.ctx,
                                chunk.meta0,
                                chunk.meta1 as usize,
                                chunk.meta2 as i32,
                                chunk.meta3 as i32,
                            );
                        }
                        for line in drain_console_log() {
                            session.log.push_str(&log_line("", &line));
                            session.log.push('\n');
                        }
                        render(session);
                        report(blk, session, ChunkStatus::NeedMore, 0);
                    }
                    None => fail_text(blk, "Event without Init", 0),
                }
            }
            ChunkCmd::Seek => fail_text(blk, "Seek is not a JS command", 0),
            ChunkCmd::Cancel => {
                reset_arena(blk);
                finish(blk, ChunkStatus::Cancelled, 0, 0, 0, 0);
            }
        }
    }
    exit()
}
