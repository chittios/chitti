//! **WASM runtime (W1/W2)** — wasmi interpreter for agent tool modules.
//!
//! Loads install-time module bytes, instantiates with fuel + memory page
//! caps, and calls exports. Guest code is **deterministic untrusted logic**;
//! side effects only through capability-gated **host imports** on the linker.
//!
//! # String ABI v1
//!
//! Tool exports take UTF-8 args via linear memory:
//!
//! * Signature: `(param i32 i32) (result i32 i32)` — `(args_ptr, args_len) →
//!   (result_ptr, result_len)`
//! * Optional export `chitti_alloc(i32) -> i32` for host scratch; if missing,
//!   the host writes at offset 0 (fixtures / tiny modules only)
//! * Guest must export linear `memory`
//!
//! # Host imports (module `chitti`) — W2
//!
//! | Import | Semantics |
//! |--------|-----------|
//! | `host_storage_{get,set,remove,list}` | [`crate::agent::storage`] for `agent_id` |
//! | `host_board_set` / `host_board_mark` | Synapse UI (task + surface ownership) |
//! | `host_ui_draw` | raw draw-ops string |
//! | `host_hud_set` | reserved pane HUD strip |
//! | `host_surface_id` | active surface for this binding |
//! | `host_now_ms` / `host_log` | clock + ktrace |
//! | `host_sys_{get,set}` | shell theme / approval mode / opacity (settings UI) | |
//!
//! # Security
//!
//! * Fuel budget per call (infinite loops trap as fuel exhaustion)
//! * [`ResourceLimiter`] caps linear memory growth (default 16 pages = 1 MiB)
//! * No ambient FS/net; draw/storage go through existing gates
//! * Modules load at install/start only — never from model output

use crate::agent::storage::{self, Scope as StorageScope};
use crate::sched::TaskId;
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use wasmi_core::LimiterError;
use wasmi::{Caller, Config, Engine, Extern, Linker, Memory, Module, ResourceLimiter, Store};

/// Default instruction fuel when a tool entry does not specify one.
pub const DEFAULT_FUEL: u64 = 5_000_000;
/// Default max linear memory: 32 × 64 KiB = 2 MiB (chess tools.wasm starts at 18 pages).
pub const DEFAULT_MAX_MEMORY_PAGES: u32 = 32;

/// Default function-table ceiling. Unchanged from the value that was hardcoded
/// in the limiter, so every existing agent module keeps exactly its old bound.
pub const DEFAULT_MAX_TABLE_ELEMS: u32 = 256;
/// Wasm page size in bytes.
const PAGE_SIZE: usize = 64 * 1024;

/// Limits applied to one module instance / store.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub fuel: u64,
    pub max_memory_pages: u32,
    /// How many instances may live in one store.
    ///
    /// One, for every guest that is a single module. The JavaScript path is the
    /// exception and the reason this is configurable: a JS-derived `tools.wasm`
    /// imports the QuickJS engine, so the engine and the module are two instances
    /// sharing a store. wasmi enforces this through `ResourceLimiter`, and the
    /// symptom of getting it wrong is the flat refusal `"tried to instantiate too
    /// many instances"` at the *second* instantiation — which reads like a problem
    /// with the module rather than with our own bookkeeping.
    pub max_instances: usize,
    /// Ceiling on function-table elements. An indirect-call table is how a
    /// module does dynamic dispatch, so its size tracks how much abstraction the
    /// guest was compiled with: a hand-written app tool needs tens of entries,
    /// while a renderer built on trait objects and closures needs hundreds (the
    /// PDF rasterizer declares 693). This used to be a hardcoded 256 in the
    /// limiter, which is why such a module failed at *instantiate* with a
    /// "missing imports?" error — the one message that does not point at tables.
    pub max_table_elems: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_FUEL,
            max_memory_pages: DEFAULT_MAX_MEMORY_PAGES,
            max_table_elems: DEFAULT_MAX_TABLE_ELEMS,
            max_instances: 1,
        }
    }
}

impl Limits {
    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel = if fuel == 0 { DEFAULT_FUEL } else { fuel };
        self
    }

    pub fn with_pages(mut self, pages: u32) -> Self {
        self.max_memory_pages = pages.max(1);
        self
    }

    pub fn with_table_elems(mut self, elems: u32) -> Self {
        self.max_table_elems = elems.max(1);
        self
    }

    pub fn with_instances(mut self, n: usize) -> Self {
        self.max_instances = n.max(1);
        self
    }
}

/// Capability context injected by the UI agent runtime for host imports.
#[derive(Clone, Copy, Debug)]
pub struct HostBindings {
    /// Agent id for storage scope (`/agent/<id>/storage/…`).
    pub agent_id: u64,
    /// Cap-owning task for Synapse UI calls (`0` = no UI / unit tests).
    pub task: TaskId,
    /// Owned surface id for board_set / board_mark / ui_draw.
    pub surface: u32,
}

impl Default for HostBindings {
    fn default() -> Self {
        Self {
            agent_id: 0,
            task: 0,
            surface: 0,
        }
    }
}

/// Host state carried in the wasmi [`Store`].
pub struct HostState {
    limiter: PageLimiter,
    pub bind: HostBindings,
    /// Last host-import diagnostic (for tests / debugging).
    pub last_error: Option<&'static str>,
    /// Rate-limit host_log spam (count of logs this store).
    log_count: u32,
    /// The three standard file descriptors, for guests that speak WASI stdio
    /// instead of the string ABI (a QuickJS module built by Javy: args JSON
    /// arrives on fd 0, the result leaves on fd 1). Empty and untouched for
    /// every other guest.
    fds: Fds,
    /// The one file a WASI guest may have open, if this surface grants any.
    ///
    /// One slot rather than a table because the only guest on this surface opens
    /// exactly one file, once, to check it exists — see [`register_wasi_fs_imports`].
    /// A table would be more general and would need an eviction policy for a case
    /// that does not arise.
    wasi_file: Option<(String, u64)>,
}

/// fd 0/1/2 backing for a WASI guest, held host-side.
///
/// Deliberately three plain buffers rather than a filesystem: the whole point of
/// this surface is that a guest gets a JSON in and a JSON out and can reach
/// nothing else. `stdin` is **rewindable** because a Javy plugin reads a runtime
/// config off fd 0 during `initialize-runtime` and the tool's own arguments have
/// to be presented afterwards as a fresh stream — feeding one where the other is
/// expected fails with `unknown field`, which reads like a bad argument rather
/// than a protocol mistake.
#[derive(Default)]
pub struct Fds {
    stdin: Vec<u8>,
    stdin_at: usize,
    stdout: Vec<u8>,
    /// Bytes already forwarded from fd 2 to ktrace, so a chatty guest cannot
    /// flood the trace (the same reason `host_log` counts).
    stderr_bytes: u32,
    /// The most recent line the guest wrote to fd 2.
    ///
    /// Kept because a JavaScript exception arrives as a **trap**, and a trap's own
    /// message is `wasm unreachable` — true and useless. The reason ("refused: this
    /// agent has no such capability bound") is what the guest printed to stderr just
    /// before dying, so holding on to it is the difference between a usable error and
    /// a shrug.
    last_stderr: String,
}

/// Cap on fd 2 forwarded to ktrace per store.
const STDERR_TRACE_CAP: u32 = 4096;

impl Fds {
    /// Present `bytes` as the guest's stdin, from the beginning.
    pub fn set_stdin(&mut self, bytes: &[u8]) {
        self.stdin.clear();
        self.stdin.extend_from_slice(bytes);
        self.stdin_at = 0;
    }

    /// Take everything the guest wrote to stdout, clearing it for the next call.
    pub fn take_stdout(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.stdout)
    }

    /// The last thing the guest said on stderr, if anything.
    pub fn last_stderr(&self) -> Option<&str> {
        let t = self.last_stderr.trim();
        (!t.is_empty()).then_some(t)
    }
}

impl HostState {
    pub fn fds(&mut self) -> &mut Fds {
        &mut self.fds
    }

    pub fn fds_ref(&self) -> &Fds {
        &self.fds
    }
}

struct PageLimiter {
    max_bytes: usize,
    max_table_elems: u32,
    max_instances: usize,
}

impl PageLimiter {
    fn new(limits: &Limits) -> Self {
        Self {
            max_bytes: (limits.max_memory_pages as usize).saturating_mul(PAGE_SIZE),
            max_table_elems: limits.max_table_elems,
            max_instances: limits.max_instances,
        }
    }
}

impl ResourceLimiter for PageLimiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool, LimiterError> {
        Ok(desired <= self.max_bytes)
    }

    // wasmi 1.x counts table elements in `usize`, not `u32`.
    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool, LimiterError> {
        Ok(desired <= self.max_table_elems as usize)
    }

    fn instances(&self) -> usize {
        self.max_instances
    }

    fn tables(&self) -> usize {
        self.max_instances
    }

    fn memories(&self) -> usize {
        self.max_instances
    }
}

fn make_engine() -> Engine {
    let mut config = Config::default();
    config.consume_fuel(true);
    config.ignore_custom_sections(true);
    Engine::new(&config)
}

/// A store and linker on the **JS** import surface, for a caller that has to
/// instantiate more than one module into the same store.
///
/// [`Session`] deliberately owns exactly one instance, so the QuickJS path cannot
/// use it: a JS-derived `tools.wasm` imports the engine's `cabi_realloc`, `invoke`
/// and `memory`, which means the plugin must be instantiated first and linked in.
/// Handing out the pieces keeps that assembly in [`crate::agent::js_rt`] rather
/// than growing a second shape into `Session`.
pub fn js_store(
    limits: Limits,
    bind: HostBindings,
) -> Result<(Store<HostState>, Linker<HostState>), &'static str> {
    // Two instances share this store: the QuickJS engine and the module that
    // imports it. Without this the second instantiation is refused outright.
    let limits = limits.with_instances(2);
    let engine = make_engine();
    let host = HostState {
        limiter: PageLimiter::new(&limits),
        bind,
        last_error: None,
        log_count: 0,
        fds: Fds::default(),
        wasi_file: None,
    };
    let mut store = Store::new(&engine, host);
    store.limiter(|s| &mut s.limiter);
    let fuel = if limits.fuel == 0 { DEFAULT_FUEL } else { limits.fuel };
    store.set_fuel(fuel).map_err(|_| "fuel metering unavailable")?;
    let mut linker = Linker::<HostState>::new(&engine);
    register_host_imports(&mut linker)?;
    register_wasi_imports(&mut linker)?;
    Ok((store, linker))
}

/// Normalise a wasmi call error the way [`Session`] does, for callers outside it.
pub fn map_trap_pub(err: wasmi::Error) -> &'static str {
    map_trap(err)
}

/// Compile + validate module bytes. Pure: no execution.
pub fn validate_module(wasm: &[u8]) -> Result<(), &'static str> {
    let engine = make_engine();
    Module::new(&engine, wasm)
        .map(|_| ())
        .map_err(|_| "invalid wasm module")
}

/// Call an export of type `(i32, i32) -> i32` (fixture / pure math tools).
pub fn call_i32_i32(
    wasm: &[u8],
    export: &str,
    a: i32,
    b: i32,
    limits: Limits,
) -> Result<i32, &'static str> {
    let mut session = Session::instantiate(wasm, limits, HostBindings::default())?;
    session.call_i32_i32(export, a, b)
}

/// Call an export of type `() -> ()` (used to prove fuel traps infinite loops).
pub fn call_void(wasm: &[u8], export: &str, limits: Limits) -> Result<(), &'static str> {
    let mut session = Session::instantiate(wasm, limits, HostBindings::default())?;
    session.call_void(export)
}

/// String-ABI tool call with default (no-UI) bindings.
pub fn call_string(
    wasm: &[u8],
    export: &str,
    args: &str,
    limits: Limits,
) -> Result<String, &'static str> {
    call_string_bound(wasm, export, args, limits, HostBindings::default())
}

/// String-ABI tool call with full host bindings (UI agent path).
pub fn call_string_bound(
    wasm: &[u8],
    export: &str,
    args: &str,
    limits: Limits,
    bind: HostBindings,
) -> Result<String, &'static str> {
    let mut session = Session::instantiate(wasm, limits, bind)?;
    session.call_string(export, args)
}

/// String-ABI call for **browser page WASM**: only `env` / WASI stubs, no
/// `chitti` storage/UI/sound imports (prevents host-import bleed from untrusted
/// page modules).
pub fn call_string_page(
    wasm: &[u8],
    export: &str,
    args: &str,
    limits: Limits,
) -> Result<String, &'static str> {
    let mut session = Session::instantiate_page(wasm, limits)?;
    session.call_string(export, args)
}

/// A live module instance bound to one store (one per UI agent).
pub struct Session {
    store: Store<HostState>,
    instance: wasmi::Instance,
}

impl Session {
    /// Compile, link **agent** host imports, instantiate with limits + bindings.
    pub fn instantiate(
        wasm: &[u8],
        limits: Limits,
        bind: HostBindings,
    ) -> Result<Self, &'static str> {
        Self::instantiate_with(wasm, limits, bind, HostImportSet::Agent)
    }

    /// Page / untrusted module path: no `chitti.*` agent host imports.
    pub fn instantiate_page(wasm: &[u8], limits: Limits) -> Result<Self, &'static str> {
        Self::instantiate_with(wasm, limits, HostBindings::default(), HostImportSet::Page)
    }

    /// A JavaScript guest: the agent surface **plus** WASI stdio. See
    /// [`HostImportSet::Js`] — the `chitti.*` half is identical to what a wasm
    /// agent gets, so this widens the ABI, not the authority.
    pub fn instantiate_js(
        wasm: &[u8],
        limits: Limits,
        bind: HostBindings,
    ) -> Result<Self, &'static str> {
        Self::instantiate_with(wasm, limits, bind, HostImportSet::Js)
    }

    /// Instantiate a guest built against a libc — [`HostImportSet::Native`].
    ///
    /// The `chitti.*` half is identical to [`Self::instantiate`]; the addition is
    /// the WASI surface a libc imports whether or not the program uses it. Opt-in
    /// per package (`wasm.native`) rather than always-on, so an app that does not
    /// need a filesystem view does not get one.
    pub fn instantiate_native(
        wasm: &[u8],
        limits: Limits,
        bind: HostBindings,
    ) -> Result<Self, &'static str> {
        Self::instantiate_with(wasm, limits, bind, HostImportSet::Native)
    }

    /// Present `bytes` as the guest's stdin, from the start. A WASI guest reads
    /// its arguments here; rewindable because a Javy plugin consumes one stream
    /// during `initialize-runtime` and the tool's arguments must follow as a new
    /// one.
    pub fn set_stdin(&mut self, bytes: &[u8]) {
        self.store.data_mut().fds.set_stdin(bytes);
    }

    /// Take what the guest wrote to stdout, clearing it for the next call.
    pub fn take_stdout(&mut self) -> Vec<u8> {
        self.store.data_mut().fds.take_stdout()
    }

    fn instantiate_with(
        wasm: &[u8],
        limits: Limits,
        bind: HostBindings,
        imports: HostImportSet,
    ) -> Result<Self, &'static str> {
        if wasm.is_empty() {
            return Err("empty wasm module");
        }
        let engine = make_engine();
        let module = Module::new(&engine, wasm).map_err(|_| "wasm parse/validate failed")?;

        let host = HostState {
            limiter: PageLimiter::new(&limits),
            bind,
            last_error: None,
            log_count: 0,
            fds: Fds::default(),
            wasi_file: None,
        };
        let mut store = Store::new(&engine, host);
        store.limiter(|s| &mut s.limiter);
        let fuel = if limits.fuel == 0 {
            DEFAULT_FUEL
        } else {
            limits.fuel
        };
        store
            .set_fuel(fuel)
            .map_err(|_| "fuel metering unavailable")?;

        let mut linker = Linker::<HostState>::new(&engine);
        match imports {
            HostImportSet::Agent => register_host_imports(&mut linker)?,
            HostImportSet::Page => register_page_imports(&mut linker)?,
            HostImportSet::Js => {
                register_host_imports(&mut linker)?;
                register_wasi_imports(&mut linker)?;
            }
            HostImportSet::Native => {
                register_host_imports(&mut linker)?;
                register_wasi_imports(&mut linker)?;
                register_wasi_fs_imports(&mut linker)?;
            }
        }
        // wasmi 1.x runs the start function as part of instantiation.
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|_| "wasm instantiate failed (missing imports/limits?)")?;

        Ok(Self { store, instance })
    }

    pub fn set_fuel(&mut self, fuel: u64) -> Result<(), &'static str> {
        let f = if fuel == 0 { DEFAULT_FUEL } else { fuel };
        self.store.set_fuel(f).map_err(|_| "set_fuel failed")
    }

    pub fn bindings(&self) -> HostBindings {
        self.store.data().bind
    }

    pub fn call_i32_i32(&mut self, export: &str, a: i32, b: i32) -> Result<i32, &'static str> {
        let func = self
            .instance
            .get_typed_func::<(i32, i32), i32>(&self.store, export)
            .map_err(|_| "export missing or wrong type (want (i32,i32)->i32)")?;
        func.call(&mut self.store, (a, b)).map_err(map_trap)
    }

    pub fn call_i32(&mut self, export: &str) -> Result<i32, &'static str> {
        let func = self
            .instance
            .get_typed_func::<(), i32>(&self.store, export)
            .map_err(|_| "export missing or wrong type (want ()->i32)")?;
        func.call(&mut self.store, ()).map_err(map_trap)
    }

    pub fn call_void(&mut self, export: &str) -> Result<(), &'static str> {
        let func = self
            .instance
            .get_typed_func::<(), ()>(&self.store, export)
            .map_err(|_| "export missing or wrong type (want ()->())")?;
        func.call(&mut self.store, ()).map_err(map_trap)
    }

    /// String ABI: write args, call export, read result.
    ///
    /// Supports both multi-value `(i32,i32)->(i32,i32)` and Rust-cdylib packed
    /// `(i32,i32)->i64` where `i64 = (ptr << 32) | len`.
    pub fn call_string(&mut self, export: &str, args: &str) -> Result<String, &'static str> {
        let args_bytes = args.as_bytes();
        let ptr = self.guest_alloc(args_bytes.len())?;
        self.write_guest(ptr, args_bytes)?;
        let a = (ptr as i32, args_bytes.len() as i32);

        let (rptr, rlen) = if let Ok(func) = self
            .instance
            .get_typed_func::<(i32, i32), (i32, i32)>(&self.store, export)
        {
            func.call(&mut self.store, a).map_err(map_trap)?
        } else if let Ok(func) = self
            .instance
            .get_typed_func::<(i32, i32), i64>(&self.store, export)
        {
            let packed = func.call(&mut self.store, a).map_err(map_trap)?;
            let p = ((packed as u64) >> 32) as i32;
            let l = (packed as u64 & 0xffff_ffff) as i32;
            (p, l)
        } else {
            return Err("export missing or wrong type (want string ABI)");
        };
        if rptr < 0 || rlen < 0 {
            return Err("guest returned negative ptr/len");
        }
        self.read_guest_string(rptr as usize, rlen as usize)
    }

    /// Stage `bytes` in guest memory and return their address — the binary-in
    /// half of a bulk ABI, for a guest whose input is megabytes (a PDF) rather
    /// than a JSON string. Base64-through-`call_string` would cost a 4/3-sized
    /// string on both sides of the boundary plus the guest's decode.
    pub fn put_bytes(&mut self, bytes: &[u8]) -> Result<usize, &'static str> {
        let ptr = self.guest_alloc(bytes.len())?;
        self.write_guest(ptr, bytes)?;
        Ok(ptr)
    }

    /// Copy `len` bytes out of guest memory at `ptr` — the binary-out half, for
    /// a guest that produces a pixel buffer.
    ///
    /// **Both arguments come from the guest**, which is the untrusted side, so
    /// this bounds-checks them against the live memory size rather than trusting
    /// the report — the same rule the image tenant's loader follows for every
    /// number a tenant hands back.
    pub fn get_bytes(&self, ptr: usize, len: usize) -> Result<Vec<u8>, &'static str> {
        let mem = self
            .instance
            .get_memory(&self.store, "memory")
            .ok_or("guest did not export memory")?;
        let data = mem.data(&self.store);
        let end = ptr.checked_add(len).ok_or("guest read overflow")?;
        if end > data.len() {
            return Err("guest read out of bounds");
        }
        Ok(data[ptr..end].to_vec())
    }

    /// Read `n` little-endian `u32`s out of guest memory — a guest-reported
    /// header (`[w, h, ptr, len]`), bounds-checked like [`Self::get_bytes`].
    pub fn get_u32s(&self, ptr: usize, n: usize) -> Result<Vec<u32>, &'static str> {
        let bytes = self.get_bytes(ptr, n * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn guest_alloc(&mut self, len: usize) -> Result<usize, &'static str> {
        if let Ok(alloc) = self
            .instance
            .get_typed_func::<i32, i32>(&self.store, "chitti_alloc")
        {
            let p = alloc
                .call(&mut self.store, len as i32)
                .map_err(map_trap)?;
            if p < 0 {
                return Err("chitti_alloc failed");
            }
            return Ok(p as usize);
        }
        if len > PAGE_SIZE {
            return Err("args too large without chitti_alloc");
        }
        let mem = self
            .instance
            .get_memory(&self.store, "memory")
            .ok_or("guest did not export memory")?;
        // Page counts are `u64` in wasmi 1.x (memory64 support).
        let need_pages = len.div_ceil(PAGE_SIZE) as u64;
        let cur = mem.size(&self.store);
        if cur < need_pages {
            mem.grow(&mut self.store, need_pages - cur)
                .map_err(|_| "memory grow denied by limiter")?;
        }
        Ok(0)
    }

    fn write_guest(&mut self, ptr: usize, bytes: &[u8]) -> Result<(), &'static str> {
        let mem = self
            .instance
            .get_memory(&self.store, "memory")
            .ok_or("guest did not export memory")?;
        let data = mem.data_mut(&mut self.store);
        let end = ptr.checked_add(bytes.len()).ok_or("guest write overflow")?;
        if end > data.len() {
            return Err("guest write out of bounds");
        }
        data[ptr..end].copy_from_slice(bytes);
        Ok(())
    }

    fn read_guest_string(&self, ptr: usize, len: usize) -> Result<String, &'static str> {
        let mem = self
            .instance
            .get_memory(&self.store, "memory")
            .ok_or("guest did not export memory")?;
        let data = mem.data(&self.store);
        let end = ptr.checked_add(len).ok_or("guest read overflow")?;
        if end > data.len() {
            return Err("guest read out of bounds");
        }
        core::str::from_utf8(&data[ptr..end])
            .map(|s| s.into())
            .map_err(|_| "guest result is not utf-8")
    }
}

// ---------------------------------------------------------------------------
// Host imports
// ---------------------------------------------------------------------------

/// Which host-import surface a wasmi instance gets.
#[derive(Clone, Copy, Debug)]
enum HostImportSet {
    /// Full agent package surface (`chitti.*` storage/UI/sound).
    Agent,
    /// Browser page surface: `env` + WASI stubs only (no agent effects).
    Page,
    /// A native guest built against a libc: the agent surface, WASI stdio, and a
    /// **read-only** WASI file view of the store.
    ///
    /// This exists for ported C programs, which reach a file through `fopen`
    /// whatever else they are given. Doom is the case in hand: its WAD arrives in
    /// linear memory via `host_fs_read` and its lumps never touch WASI at all, but
    /// `D_FindWADByName` still checks the file *exists* before opening it, and
    /// without that it reports "Game mode indeterminate" and refuses to start.
    Native,
    /// A JavaScript guest: the **whole** agent surface plus WASI stdio, because
    /// a QuickJS module reads its arguments from fd 0 and writes its result to
    /// fd 1. The `chitti.*` half is the identical, already-gated set a wasm
    /// agent gets — the JS reaches nothing a hand-written module could not.
    Js,
}

/// Minimal imports for untrusted page WASM — no storage, UI, or sound.
fn register_page_imports(linker: &mut Linker<HostState>) -> Result<(), &'static str> {
    linker
        .func_wrap("env", "abort", |_caller: Caller<'_, HostState>| {
            // Trap-like: wasmi will surface as a call error if the guest aborts via this.
        })
        .map_err(|_| "define env.abort")?;
    linker
        .func_wrap(
            "env",
            "log",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                if caller.data().log_count >= 16 {
                    return;
                }
                caller.data_mut().log_count += 1;
                if let Some(msg) = read_guest_str(&caller, ptr, len) {
                    let preview: String = msg.chars().take(120).collect();
                    crate::serial_println!("page.wasm.log> {preview}");
                }
            },
        )
        .map_err(|_| "define env.log")?;
    register_wasi_imports(linker)
}

/// WASI errno: bad file descriptor.
const WASI_EBADF: i32 = 8;

/// The `wasi_snapshot_preview1` surface, which is exactly the ten functions a
/// Javy-compiled module imports — and no more. Only `fd_read` and `fd_write`
/// move data, against the host-side [`Fds`]; the rest exist so a module that
/// links WASI can instantiate at all.
///
/// **These signatures are load-bearing.** wasmi resolves an import by name *and*
/// `FuncType`, so a stub of the wrong arity does not degrade gracefully — the
/// module fails at instantiation with `"wasm instantiate failed (missing
/// imports/limits?)"`, a message that says nothing about types. This replaced
/// five stubs that were all declared `() -> i32` while real preview1 takes
/// parameters, so no genuine WASI module could ever have loaded. Nothing noticed
/// because the only module on this surface (`pdfrender.wasm`) has no import
/// section at all, and no test instantiated a WASI importer — `a_wasi_module_can_
/// instantiate_and_do_stdio` is now that test.
///
/// The clock reads zero and the RNG returns zeros. That is not laziness: it is
/// what `javy build -C deterministic` assumes, and a tool below the determinism
/// boundary should not be able to observe wall-clock time or entropy anyway.
fn register_wasi_imports(linker: &mut Linker<HostState>) -> Result<(), &'static str> {
    const W: &str = "wasi_snapshot_preview1";
    // fd_read(fd, iovs, iovs_len, nread) -> errno. Drains stdin across the iovec
    // list; a short read (0 bytes) is EOF, which is how the guest stops looping.
    linker
        .func_wrap(
            W,
            "fd_read",
            |mut caller: Caller<'_, HostState>, fd: i32, iovs: i32, iovs_len: i32, nread: i32| -> i32 {
                if fd != 0 {
                    return WASI_EBADF;
                }
                let mut wrote = 0u32;
                for i in 0..iovs_len.max(0) {
                    let Some((ptr, len)) = read_iovec(&caller, iovs, i) else {
                        return WASI_EBADF;
                    };
                    let at = caller.data().fds.stdin_at;
                    let take = caller.data().fds.stdin.len().saturating_sub(at).min(len);
                    if take == 0 {
                        break;
                    }
                    let chunk = caller.data().fds.stdin[at..at + take].to_vec();
                    if write_guest_bytes(&mut caller, ptr as i32, &chunk).is_err() {
                        return WASI_EBADF;
                    }
                    caller.data_mut().fds.stdin_at += take;
                    wrote += take as u32;
                }
                if write_guest_bytes(&mut caller, nread, &wrote.to_le_bytes()).is_err() {
                    return WASI_EBADF;
                }
                0
            },
        )
        .map_err(|_| "define wasi.fd_read")?;
    // fd_write(fd, iovs, iovs_len, nwritten) -> errno. fd 1 is the result; fd 2
    // is diagnostics and goes to ktrace, capped so a chatty guest cannot flood it.
    linker
        .func_wrap(
            W,
            "fd_write",
            |mut caller: Caller<'_, HostState>, fd: i32, iovs: i32, iovs_len: i32, nwritten: i32| -> i32 {
                if fd != 1 && fd != 2 {
                    return WASI_EBADF;
                }
                let mut total = 0u32;
                for i in 0..iovs_len.max(0) {
                    let Some((ptr, len)) = read_iovec(&caller, iovs, i) else {
                        return WASI_EBADF;
                    };
                    let Some(bytes) = read_guest_bytes(&caller, ptr as i32, len as i32) else {
                        return WASI_EBADF;
                    };
                    if fd == 1 {
                        caller.data_mut().fds.stdout.extend_from_slice(&bytes);
                    } else {
                        let used = caller.data().fds.stderr_bytes;
                        if used < STDERR_TRACE_CAP {
                            caller.data_mut().fds.stderr_bytes = used.saturating_add(bytes.len() as u32);
                            let msg = String::from_utf8_lossy(&bytes);
                            let preview: String = msg.chars().take(200).collect();
                            crate::ktrace::log_fmt(format_args!("js.stderr> {}", preview.trim_end()));
                        }
                        // Kept regardless of the trace cap: this is what a failed call
                        // reports back, and the *last* line is the interesting one.
                        {
                            let msg = String::from_utf8_lossy(&bytes);
                            let line = msg.trim();
                            if !line.is_empty() && !line.starts_with("note: run with") {
                                caller.data_mut().fds.last_stderr = String::from(line);
                            }
                        }
                    }
                    total += len as u32;
                }
                if write_guest_bytes(&mut caller, nwritten, &total.to_le_bytes()).is_err() {
                    return WASI_EBADF;
                }
                0
            },
        )
        .map_err(|_| "define wasi.fd_write")?;
    linker
        .func_wrap(W, "fd_close", |_c: Caller<'_, HostState>, _fd: i32| -> i32 { 0 })
        .map_err(|_| "define wasi.fd_close")?;
    linker
        .func_wrap(
            W,
            "fd_seek",
            |_c: Caller<'_, HostState>, _fd: i32, _off: i64, _whence: i32, _out: i32| -> i32 { WASI_EBADF },
        )
        .map_err(|_| "define wasi.fd_seek")?;
    linker
        .func_wrap(
            W,
            "fd_fdstat_get",
            |_c: Caller<'_, HostState>, _fd: i32, _out: i32| -> i32 { 0 },
        )
        .map_err(|_| "define wasi.fd_fdstat_get")?;
    // No environment: both counts written as zero, so `environ_get` has nothing
    // to fill in.
    linker
        .func_wrap(
            W,
            "environ_sizes_get",
            |mut caller: Caller<'_, HostState>, count: i32, size: i32| -> i32 {
                if write_guest_bytes(&mut caller, count, &0u32.to_le_bytes()).is_err()
                    || write_guest_bytes(&mut caller, size, &0u32.to_le_bytes()).is_err()
                {
                    return WASI_EBADF;
                }
                0
            },
        )
        .map_err(|_| "define wasi.environ_sizes_get")?;
    linker
        .func_wrap(W, "environ_get", |_c: Caller<'_, HostState>, _a: i32, _b: i32| -> i32 { 0 })
        .map_err(|_| "define wasi.environ_get")?;
    linker
        .func_wrap(
            W,
            "clock_time_get",
            |mut caller: Caller<'_, HostState>, _id: i32, _prec: i64, out: i32| -> i32 {
                if write_guest_bytes(&mut caller, out, &0u64.to_le_bytes()).is_err() {
                    return WASI_EBADF;
                }
                0
            },
        )
        .map_err(|_| "define wasi.clock_time_get")?;
    linker
        .func_wrap(
            W,
            "random_get",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| -> i32 {
                let zeros = alloc::vec![0u8; len.max(0) as usize];
                if write_guest_bytes(&mut caller, ptr, &zeros).is_err() {
                    return WASI_EBADF;
                }
                0
            },
        )
        .map_err(|_| "define wasi.random_get")?;
    // A guest that exits is done, not broken: record nothing and return. The
    // caller reads the result off fd 1 either way.
    linker
        .func_wrap(W, "proc_exit", |_c: Caller<'_, HostState>, _code: i32| {})
        .map_err(|_| "define wasi.proc_exit")?;
    Ok(())
}

/// Read the `i`-th `iovec` (two little-endian u32s: buffer pointer, length) out
/// of a guest-supplied array.
fn read_iovec(caller: &Caller<'_, HostState>, iovs: i32, i: i32) -> Option<(usize, usize)> {
    let base = iovs.checked_add(i.checked_mul(8)?)?;
    let raw = read_guest_bytes(caller, base, 8)?;
    let ptr = u32::from_le_bytes(raw[0..4].try_into().ok()?) as usize;
    let len = u32::from_le_bytes(raw[4..8].try_into().ok()?) as usize;
    Some((ptr, len))
}

fn register_host_imports(linker: &mut Linker<HostState>) -> Result<(), &'static str> {
    // storage_set(scope, key_ptr, key_len, val_ptr, val_len) -> i32
    linker
        .func_wrap(
            "chitti",
            "host_storage_set",
            |caller: Caller<'_, HostState>,
             scope: i32,
             key_ptr: i32,
             key_len: i32,
             val_ptr: i32,
             val_len: i32|
             -> i32 {
                let id = caller.data().bind.agent_id;
                if id == 0 {
                    return -3; // no ambient agent-0 storage
                }
                let Some(key) = read_guest_str(&caller, key_ptr, key_len) else {
                    return -1;
                };
                let Some(val) = read_guest_bytes(&caller, val_ptr, val_len) else {
                    return -1;
                };
                let sc = scope_from_i32(scope);
                match storage::set(id, sc, &key, &val) {
                    Ok(()) => 0,
                    Err(_) => -1,
                }
            },
        )
        .map_err(|_| "define host_storage_set")?;

    // storage_get(scope, key_ptr, key_len, out_ptr, out_cap) -> i32  (len or -1/-2)
    linker
        .func_wrap(
            "chitti",
            "host_storage_get",
            |mut caller: Caller<'_, HostState>,
             scope: i32,
             key_ptr: i32,
             key_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                let Some(key) = read_guest_str(&caller, key_ptr, key_len) else {
                    return -2;
                };
                let sc = scope_from_i32(scope);
                let id = caller.data().bind.agent_id;
                let Some(val) = storage::get(id, sc, &key) else {
                    return -1;
                };
                if out_cap < 0 || (val.len() as i32) > out_cap {
                    return -2;
                }
                if write_guest_bytes(&mut caller, out_ptr, &val).is_err() {
                    return -2;
                }
                val.len() as i32
            },
        )
        .map_err(|_| "define host_storage_get")?;

    linker
        .func_wrap(
            "chitti",
            "host_storage_remove",
            |caller: Caller<'_, HostState>, scope: i32, key_ptr: i32, key_len: i32| -> i32 {
                let Some(key) = read_guest_str(&caller, key_ptr, key_len) else {
                    return -1;
                };
                let sc = scope_from_i32(scope);
                let id = caller.data().bind.agent_id;
                match storage::remove(id, sc, &key) {
                    Ok(true) => 1,
                    Ok(false) => 0,
                    Err(_) => -1,
                }
            },
        )
        .map_err(|_| "define host_storage_remove")?;

    linker
        .func_wrap(
            "chitti",
            "host_storage_list",
            |mut caller: Caller<'_, HostState>, scope: i32, out_ptr: i32, out_cap: i32| -> i32 {
                let sc = scope_from_i32(scope);
                let id = caller.data().bind.agent_id;
                let joined = storage::list(id, sc).join("\n");
                let bytes = joined.as_bytes();
                if out_cap < 0 || (bytes.len() as i32) > out_cap {
                    return -2;
                }
                if write_guest_bytes(&mut caller, out_ptr, bytes).is_err() {
                    return -2;
                }
                bytes.len() as i32
            },
        )
        .map_err(|_| "define host_storage_list")?;

    // board_set(fen_ptr, fen_len) -> i32
    linker
        .func_wrap(
            "chitti",
            "host_board_set",
            |caller: Caller<'_, HostState>, fen_ptr: i32, fen_len: i32| -> i32 {
                let Some(fen) = read_guest_str(&caller, fen_ptr, fen_len) else {
                    return -1;
                };
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2; // no UI binding
                }
                match syn_board_set(bind.task, bind.surface, &fen) {
                    true => 0,
                    false => -1,
                }
            },
        )
        .map_err(|_| "define host_board_set")?;

    linker
        .func_wrap(
            "chitti",
            "host_board_mark",
            |caller: Caller<'_, HostState>,
             sq_ptr: i32,
             sq_len: i32,
             color_ptr: i32,
             color_len: i32|
             -> i32 {
                let Some(sq) = read_guest_str(&caller, sq_ptr, sq_len) else {
                    return -1;
                };
                let color = read_guest_str(&caller, color_ptr, color_len)
                    .unwrap_or_else(|| "cc785c".into());
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2;
                }
                match syn_board_mark(bind.task, bind.surface, &sq, &color) {
                    true => 0,
                    false => -1,
                }
            },
        )
        .map_err(|_| "define host_board_mark")?;

    linker
        .func_wrap(
            "chitti",
            "host_ui_draw",
            |caller: Caller<'_, HostState>, ops_ptr: i32, ops_len: i32| -> i32 {
                let Some(ops) = read_guest_str(&caller, ops_ptr, ops_len) else {
                    return -1;
                };
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2;
                }
                // -3 is distinct from -1 on purpose: "one of your ops is too long"
                // is a guest bug it can fix, "refused" is not.
                match syn_ui_draw(bind.task, bind.surface, &ops) {
                    Ok(()) => 0,
                    Err(UiDrawErr::Refused) => -1,
                    Err(UiDrawErr::Unsplittable) => -3,
                }
            },
        )
        .map_err(|_| "define host_ui_draw")?;

    // host_ui_present(ptr, len, w, h) -> 0 ok | -1 refused | -2 no binding |
    //                                    -3 the numbers do not describe a frame
    //
    // The pixel counterpart of `host_ui_draw`, for a renderer that produces a
    // frame rather than draw ops. Pixels are 0xRRGGBB in the low 24 bits of a
    // little-endian u32, `w * h` of them, row-major.
    //
    // Pixels do **not** cross the ring-3 boundary: a tenant cannot reach guest
    // linear memory and the call block is 1920 bytes, so a frame could never fit.
    // Instead the frame is read here — where wasmi bounds-checks the read against
    // live linear memory — parked against the calling task, and then the ordinary
    // gated call carries only `{surface, w, h}`. Reading is not an effect;
    // *applying* is, and only the gate can cause that. A refused frame is dropped,
    // so a denial costs a copy and changes nothing on screen.
    linker
        .func_wrap(
            "chitti",
            "host_ui_present",
            |caller: Caller<'_, HostState>, ptr: i32, len: i32, w: i32, h: i32| -> i32 {
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2;
                }
                // Every number is checked before it is used to size anything —
                // "a caller's claimed length must never size a buffer".
                if w <= 0 || h <= 0 || len <= 0 {
                    return -3;
                }
                let (w, h, len) = (w as usize, h as usize, len as usize);
                let Some(px_count) = w.checked_mul(h) else {
                    return -3;
                };
                if px_count > crate::synapse::ui::MAX_SURF_PIXELS
                    || len != px_count.saturating_mul(4)
                {
                    return -3;
                }
                let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) else {
                    return -1;
                };
                let data = mem.data(&caller);
                let start = ptr as usize;
                let Some(end) = start.checked_add(len) else {
                    return -3;
                };
                if end > data.len() {
                    return -3;
                }
                let px: alloc::vec::Vec<u32> = data[start..end]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0x00ff_ffff)
                    .collect();
                crate::synapse::ui::stage_pixels(bind.task, w, h, px);
                if syn_ui_present(bind.task, bind.surface, w, h) {
                    0
                } else {
                    // The gate refused, or userspace never ran it. Either way the
                    // stage must not survive to be picked up by a later call.
                    crate::synapse::ui::discard_staged(bind.task);
                    -1
                }
            },
        )
        .map_err(|_| "define host_ui_present")?;

    linker
        .func_wrap(
            "chitti",
            "host_hud_set",
            |caller: Caller<'_, HostState>, text_ptr: i32, text_len: i32| -> i32 {
                // Empty (len 0) clears the HUD; a null/oob ptr is an error.
                let text = if text_len == 0 {
                    String::new()
                } else {
                    match read_guest_str(&caller, text_ptr, text_len) {
                        Some(t) => t,
                        None => return -1,
                    }
                };
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2;
                }
                match syn_ui_hud(bind.task, bind.surface, &text) {
                    true => 0,
                    false => -1,
                }
            },
        )
        .map_err(|_| "define host_hud_set")?;

    linker
        .func_wrap(
            "chitti",
            "host_audio_submit",
            |caller: Caller<'_, HostState>, ptr: i32, frames: i32, rate: i32, channels: i32| -> i32 {
                // Signed 16-bit PCM, `frames * channels` samples, interleaved.
                // 0 ok | -1 refused | -2 no binding | -3 the numbers do not
                // describe a block.
                //
                // There is deliberately no `host_audio_rate`: the guest names its
                // rate here and every driver takes it per call, so there is no
                // single device rate to report and a function that returned one
                // would be inventing it.
                //
                // The audio counterpart of `host_ui_present`, and the same
                // staging contract for the same reason: samples cannot cross the
                // 1920-byte ring-3 call block, so they are read here (bounds-
                // checked by wasmi), parked against the task, and the gated call
                // carries only their shape. Reading is not an effect; playing is.
                //
                // This replaces `host_sound_play(hz, ms)` for anything real. That
                // one is a tone generator sharing a 32-call *lifetime* budget with
                // logging and notifications, which `synth.rs` exhausts after 32
                // key presses — fine for a jingle, useless for game audio.
                let bind = caller.data().bind;
                if bind.task == 0 {
                    return -2;
                }
                if frames <= 0 || channels <= 0 || channels > 2 || rate <= 0 {
                    return -3;
                }
                let (frames, channels) = (frames as usize, channels as usize);
                let Some(samples) = frames.checked_mul(channels) else {
                    return -3;
                };
                if samples > crate::sound::MAX_SUBMIT_SAMPLES {
                    return -3;
                }
                let Some(bytes) = samples.checked_mul(2) else {
                    return -3;
                };
                let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) else {
                    return -1;
                };
                let data = mem.data(&caller);
                let start = ptr as usize;
                let Some(end) = start.checked_add(bytes) else {
                    return -3;
                };
                if end > data.len() {
                    return -3;
                }
                let pcm: alloc::vec::Vec<i16> = data[start..end]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                crate::sound::stage_pcm(bind.task, pcm);
                if syn_audio_submit(bind.task, frames, rate as u32, channels as u8) {
                    0
                } else {
                    crate::sound::discard_staged_pcm(bind.task);
                    -1
                }
            },
        )
        .map_err(|_| "define host_audio_submit")?;

    linker
        .func_wrap(
            "chitti",
            "host_audio_free",
            |_: Caller<'_, HostState>| -> i32 {
                // Room in the output queue, in samples. Inert and un-gated for the
                // same reason `host_now_ms` is: it observes our own device state
                // and reveals only queue depth. It must stay a plain import rather
                // than a primitive — a game polls it every frame, and an audited
                // call there would double the audit and ktrace volume the frame
                // pump already has to keep down.
                crate::sound::out_free_samples().min(i32::MAX as usize) as i32
            },
        )
        .map_err(|_| "define host_audio_free")?;


    linker
        .func_wrap(
            "chitti",
            "host_keys_held",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| -> i32 {
                // Writes the 8-word (32-byte) physical held-key bitmap, indexed
                // by HID usage: word `usage / 32`, bit `usage % 32`, LSB-first.
                // Returns the bytes written, or negative on refusal.
                //
                // **Gated on focus, and that is the whole security story.** Held-key
                // state is keyboard input: an unfocused app that could read it
                // would see everything typed at the shell prompt, in the editor and
                // into the password modal — a keylogger reachable from any
                // installed package. Focus is the same boundary `forward_key`
                // already enforces for pushed keys; this makes the pull-based
                // reader honour it too. An unfocused caller gets **zeros rather
                // than an error**, because "nothing is held" is the truth from its
                // point of view and an error would only tell it that someone else
                // is typing.
                let bind = caller.data().bind;
                if bind.task == 0 || bind.surface == 0 {
                    return -2;
                }
                const BYTES: usize = 32;
                if len < BYTES as i32 {
                    return -3;
                }
                let focused = crate::service::package_ui::is_input_target(bind.surface);
                let words = if focused {
                    crate::keymap::held_snapshot()
                } else {
                    [0u32; 8]
                };
                let mut buf = [0u8; BYTES];
                for (i, w) in words.iter().enumerate() {
                    buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                let Some(mem) = caller.get_export("memory").and_then(|e| e.into_memory()) else {
                    return -1;
                };
                match mem.write(&mut caller, ptr as usize, &buf) {
                    Ok(()) => BYTES as i32,
                    Err(_) => -3,
                }
            },
        )
        .map_err(|_| "define host_keys_held")?;

    linker
        .func_wrap(
            "chitti",
            "host_surface_size",
            |caller: Caller<'_, HostState>| -> i64 {
                // (w << 32) | h, or 0 if this binding owns no surface. Inert: it
                // reports the geometry the kernel already chose for this guest
                // from its own manifest, so there is nothing to gate — the same
                // reason `host_surface_id` and `host_now_ms` are plain imports.
                let sid = caller.data().bind.surface;
                match crate::synapse::ui::surface_dims(sid) {
                    Some((w, h)) => ((w as i64) << 32) | (h as i64 & 0xffff_ffff),
                    None => 0,
                }
            },
        )
        .map_err(|_| "define host_surface_size")?;

    linker
        .func_wrap(
            "chitti",
            "host_surface_id",
            |caller: Caller<'_, HostState>| -> i32 { caller.data().bind.surface as i32 },
        )
        .map_err(|_| "define host_surface_id")?;

    linker
        .func_wrap("chitti", "host_now_ms", |_caller: Caller<'_, HostState>| -> i64 {
            crate::arch::now_ms() as i64
        })
        .map_err(|_| "define host_now_ms")?;

    linker
        .func_wrap(
            "chitti",
            "host_log",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                if caller.data().log_count >= 64 {
                    return; // rate limit
                }
                caller.data_mut().log_count += 1;
                if let Some(msg) = read_guest_str(&caller, ptr, len) {
                    crate::serial_println!("wasm.host_log> {msg}");
                }
            },
        )
        .map_err(|_| "define host_log")?;

    // host_sound_play(hz, ms) -> i32  — tone via the sound device (synth agents).
    // Gated: requires a real agent binding (agent_id != 0) and a short rate
    // limit so page/default instances cannot spam audio.
    linker
        .func_wrap(
            "chitti",
            "host_sound_play",
            |mut caller: Caller<'_, HostState>, hz: i32, ms: i32| -> i32 {
                let bind = caller.data().bind;
                if bind.agent_id == 0 {
                    caller.data_mut().last_error = Some("sound: no agent binding");
                    return -3;
                }
                // Reuse log_count as a crude rate limit shared with host_log.
                if caller.data().log_count >= 32 {
                    return -4;
                }
                caller.data_mut().log_count = caller.data().log_count.saturating_add(1);
                if !crate::sound::is_up() {
                    return -1;
                }
                let hz = (hz as u32).clamp(20, 4000);
                let ms = (ms as u32).clamp(20, 500); // tighter per-call cap
                let rate = 16_000u32;
                let pcm = crate::sound::test_tone(hz, ms, rate);
                match crate::sound::play(&pcm, rate) {
                    Ok(()) => 0,
                    Err(_) => -2,
                }
            },
        )
        .map_err(|_| "define host_sound_play")?;

    // host_notify(severity, title_ptr, title_len, body_ptr, body_len) -> i32
    //
    // The agent-facing notification API. Same three rules the `notify` shell tool
    // follows, and for the same reasons:
    //
    //   1. **The source is stamped from the binding**, never taken from the
    //      guest. A guest that could name its own source could post as `kernel`,
    //      which is a transfer of authority wearing a label.
    //   2. **Write-only.** There is no `host_notify_list`, so a notification an
    //      agent posts cannot be read back — which removes the laundering channel
    //      (untrusted content re-entering its own context with the tag stripped)
    //      for zero policy rather than a tagging path.
    //   3. **Rate-limited**, sharing the `log_count` budget with `host_log` and
    //      `host_sound_play`. A notification pane must not be a guest-controlled
    //      spam surface, and the chime makes that louder than usual.
    //
    // Requires a real agent binding: an unbound page instance has no identity to
    // stamp, so it has nothing to say.
    linker
        .func_wrap(
            "chitti",
            "host_notify",
            |mut caller: Caller<'_, HostState>,
             sev: i32,
             tptr: i32,
             tlen: i32,
             bptr: i32,
             blen: i32|
             -> i32 {
                let bind = caller.data().bind;
                if bind.agent_id == 0 {
                    caller.data_mut().last_error = Some("notify: no agent binding");
                    return -3;
                }
                if caller.data().log_count >= 32 {
                    return -4;
                }
                caller.data_mut().log_count = caller.data().log_count.saturating_add(1);
                let Some(title) = read_guest_str(&caller, tptr, tlen) else {
                    return -1;
                };
                if title.trim().is_empty() {
                    caller.data_mut().last_error = Some("notify: a notification needs a title");
                    return -1;
                }
                let body = if blen > 0 {
                    read_guest_str(&caller, bptr, blen).unwrap_or_default()
                } else {
                    alloc::string::String::new()
                };
                let sev = match sev {
                    1 => crate::notify::Severity::Success,
                    2 => crate::notify::Severity::Warn,
                    3 => crate::notify::Severity::Error,
                    // `Action` is deliberately **not** reachable from a guest: it
                    // means "a human decision is waiting", which only the kernel's
                    // own unattended-approval path is entitled to claim. A guest
                    // asking for it gets `Warn`.
                    4 => crate::notify::Severity::Warn,
                    _ => crate::notify::Severity::Info,
                };
                let source = alloc::format!("agent:{}", bind.agent_id);
                crate::notify::post(sev, &source, &title, &body);
                0
            },
        )
        .map_err(|_| "define host_notify")?;

    // host_sys_set(key, val) / host_sys_get(key, out) — apply or read OS UI
    // preferences from the settings package. Keys: theme, mode, opacity.
    // Requires a real agent binding so unbound page instances cannot flip mode.
    linker
        .func_wrap(
            "chitti",
            "host_sys_set",
            |mut caller: Caller<'_, HostState>,
             k_ptr: i32,
             k_len: i32,
             v_ptr: i32,
             v_len: i32|
             -> i32 {
                if caller.data().bind.agent_id == 0 {
                    caller.data_mut().last_error = Some("sys_set: no agent binding");
                    return -3;
                }
                let Some(key) = read_guest_str(&caller, k_ptr, k_len) else {
                    return -1;
                };
                let Some(val) = read_guest_str(&caller, v_ptr, v_len) else {
                    return -1;
                };
                match apply_sys_pref(&key, &val) {
                    Ok(()) => 0,
                    Err(e) => {
                        caller.data_mut().last_error = Some(e);
                        -2
                    }
                }
            },
        )
        .map_err(|_| "define host_sys_set")?;

    linker
        .func_wrap(
            "chitti",
            "host_sys_get",
            |mut caller: Caller<'_, HostState>,
             k_ptr: i32,
             k_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                let Some(key) = read_guest_str(&caller, k_ptr, k_len) else {
                    return -1;
                };
                let Some(val) = read_sys_pref(&key) else {
                    return -2;
                };
                if out_ptr < 0 || out_cap < 0 {
                    return -1;
                }
                let bytes = val.as_bytes();
                let n = bytes.len().min(out_cap as usize);
                if write_guest_bytes(&mut caller, out_ptr, &bytes[..n]).is_err() {
                    return -1;
                }
                n as i32
            },
        )
        .map_err(|_| "define host_sys_get")?;

    // host_tasks_list(out, cap) → n bytes written. Lines: `id\tname\tstate\n`.
    // Read-only snapshot of the scheduler for the Activity package UI so it is
    // not a fake progress toy. Unbound (agent_id 0) callers get an empty list.
    linker
        .func_wrap(
            "chitti",
            "host_tasks_list",
            |mut caller: Caller<'_, HostState>, out_ptr: i32, out_cap: i32| -> i32 {
                if out_ptr < 0 || out_cap <= 0 {
                    return -1;
                }
                if caller.data().bind.agent_id == 0 {
                    return 0;
                }
                let mut s = alloc::string::String::new();
                for (id, name, state) in crate::sched::list() {
                    use core::fmt::Write;
                    let _ = write!(s, "{id}\t{name}\t{state}\n");
                    if s.len() >= out_cap as usize {
                        s.truncate(out_cap as usize);
                        break;
                    }
                }
                // Also surface heap / reclaim pressure so Activity's "mem" bar
                // is real rather than a random walk.
                let (_, free, used) = crate::mm::heap::stats();
                let total = free.saturating_add(used).max(1);
                let pct = ((used as u64 * 100) / total as u64).min(100);
                use core::fmt::Write;
                let _ = write!(s, "heap\t{used}/{total}\t{pct}%\n");
                let bytes = s.as_bytes();
                let n = bytes.len().min(out_cap as usize);
                if write_guest_bytes(&mut caller, out_ptr, &bytes[..n]).is_err() {
                    return -1;
                }
                n as i32
            },
        )
        .map_err(|_| "define host_tasks_list")?;

    // host_fs_list(path, out, cap) → n. Lines `name\tkind\tsize\n` for children
    // of `path` (Synapse virtual FS, same view as shell `/ls`). kind is `d` or `f`.
    // `n` is the listing's **full** length; `n > cap` means it was truncated and
    // the guest should retry with a bigger buffer (see `fill_guest`).
    linker
        .func_wrap(
            "chitti",
            "host_fs_list",
            |mut caller: Caller<'_, HostState>,
             path_ptr: i32,
             path_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                if caller.data().bind.agent_id == 0 {
                    return -3;
                }
                if out_ptr < 0 || out_cap <= 0 {
                    return -1;
                }
                let Some(path) = read_guest_str(&caller, path_ptr, path_len) else {
                    return -1;
                };
                let path = if path.is_empty() { "/" } else { path.as_str() };
                let mut s = alloc::string::String::new();
                use core::fmt::Write;
                for e in crate::synapse::fs::list_dir(path) {
                    let kind = if e.is_dir { 'd' } else { 'f' };
                    let name = e.name.as_str();
                    // Skip empty / weird names.
                    if name.is_empty() || name == "." || name == ".." {
                        continue;
                    }
                    let _ = write!(s, "{name}\t{kind}\t{}\n", e.size);
                }
                // The whole listing is built and its **full** length reported, so a
                // guest with too small a buffer can grow and ask again. Stopping at
                // `out_cap` instead would also cut the last line mid-field, which
                // parses as a plausible bogus entry rather than as an error.
                match fill_guest(&mut caller, out_ptr, out_cap, s.as_bytes()) {
                    Some(n) => n as i32,
                    None => -1,
                }
            },
        )
        .map_err(|_| "define host_fs_list")?;

    // host_fs_read(path, out, cap) → n, the file's **full** length (up to `cap`
    // bytes of it are written; `n > cap` means retry bigger — see `fill_guest`).
    linker
        .func_wrap(
            "chitti",
            "host_fs_read",
            |mut caller: Caller<'_, HostState>,
             path_ptr: i32,
             path_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                if caller.data().bind.agent_id == 0 {
                    return -3;
                }
                if out_ptr < 0 || out_cap <= 0 {
                    return -1;
                }
                let Some(path) = read_guest_str(&caller, path_ptr, path_len) else {
                    return -1;
                };
                let Some(data) = crate::synapse::fs::read(&path) else {
                    return -2;
                };
                // The **file's** length, not the written length — see `fill_guest`.
                match fill_guest(&mut caller, out_ptr, out_cap, &data) {
                    Some(n) => n as i32,
                    None => -1,
                }
            },
        )
        .map_err(|_| "define host_fs_read")?;

        // host_user_home(out_ptr, out_cap) -> i32 — the **ChittiOS user home**
    // (`~`, `/home/chitti`) the shell agent starts in. Distinct from
    // `host_home` (the calling agent's `/agent/<id>` install folder): a tool
    // like git defaults its working tree to the user's home, not its own.
    linker
        .func_wrap(
            "chitti",
            "host_user_home",
            |mut caller: Caller<'_, HostState>, out_ptr: i32, out_cap: i32| -> i32 {
                let home = crate::agent::home::USER_HOME;
                let b = home.as_bytes();
                let n = b.len().min(out_cap.max(0) as usize);
                if write_guest_bytes(&mut caller, out_ptr, &b[..n]).is_err() {
                    return -1;
                }
                n as i32
            },
        )
        .map_err(|_| "define host_user_home")?;

    // host_home(out_ptr, out_cap) -> i32 — the calling agent's install folder
    // (`/agent/<id>`), so a wasm tool can build absolute store paths for the
    // unscoped `host_fs_read`/`host_fs_list` and home-scoped `host_fs_write`.
    linker
        .func_wrap(
            "chitti",
            "host_home",
            |mut caller: Caller<'_, HostState>, out_ptr: i32, out_cap: i32| -> i32 {
                let id = caller.data().bind.agent_id;
                if id == 0 {
                    return -3;
                }
                let home = crate::agent::home::path(id);
                let b = home.as_bytes();
                let n = b.len().min(out_cap.max(0) as usize);
                if write_guest_bytes(&mut caller, out_ptr, &b[..n]).is_err() {
                    return -1;
                }
                n as i32
            },
        )
        .map_err(|_| "define host_home")?;

    // host_fs_write(path, data_ptr, data_len) -> i32  (0 ok; -1 bad args, -2
    // outside the agent's home, -3 no agent). Unlike the read side, **write is
    // scoped to the calling agent's own `/agent/<id>/` home** — a wasm tool is
    // untrusted input and must not scribble over arbitrary store keys. The git
    // agent's repos live in its home.
    linker
        .func_wrap(
            "chitti",
            "host_fs_write",
            |caller: Caller<'_, HostState>,
             path_ptr: i32,
             path_len: i32,
             data_ptr: i32,
             data_len: i32|
             -> i32 {
                let id = caller.data().bind.agent_id;
                if id == 0 {
                    return -3;
                }
                let Some(path) = read_guest_str(&caller, path_ptr, path_len) else {
                    return -1;
                };
                let Some(data) = read_guest_bytes(&caller, data_ptr, data_len) else {
                    return -1;
                };
                let home = crate::agent::home::path(id);
                let p = crate::synapse::vpath::normalize(&path);
                let in_home = p == home || p.starts_with(&alloc::format!("{home}/"));
                if !in_home {
                    // Only agents whose manifest grants `Scope::Any` fs may
                    // write outside their home — and never into another
                    // agent's private `/agent/<n>/` folder (SOULs, storage).
                    if !crate::agent::system::fs_any_scope(id) {
                        return -2;
                    }
                    if p.starts_with("/agent/") {
                        return -2;
                    }
                }
                crate::synapse::fs::write(&p, &data);
                0
            },
        )
        .map_err(|_| "define host_fs_write")?;

    // host_fs_remove(path) -> i32 (0 removed, -1 bad args, -2 refused).
    //
    // Gated **exactly like `host_fs_write`**, and deliberately so: deleting a
    // file is a write in every sense that matters here, so an agent that may
    // not create a path may not remove one either, and the same
    // `/agent/<n>/` exclusion keeps one agent out of another's SOUL and
    // storage. `git rm`, `git clean` and `git tag -d` are the callers.
    linker
        .func_wrap(
            "chitti",
            "host_fs_remove",
            |caller: Caller<'_, HostState>, path_ptr: i32, path_len: i32| -> i32 {
                let id = caller.data().bind.agent_id;
                if id == 0 {
                    return -3;
                }
                let Some(path) = read_guest_str(&caller, path_ptr, path_len) else {
                    return -1;
                };
                let home = crate::agent::home::path(id);
                let p = crate::synapse::vpath::normalize(&path);
                let in_home = p == home || p.starts_with(&alloc::format!("{home}/"));
                if !in_home {
                    if !crate::agent::system::fs_any_scope(id) {
                        return -2;
                    }
                    if p.starts_with("/agent/") {
                        return -2;
                    }
                }
                crate::synapse::fs::delete(&p);
                0
            },
        )
        .map_err(|_| "define host_fs_remove")?;

    // host_fs_exists(path) -> i32 (1 exists, 0 missing, -1 bad args).
    linker
        .func_wrap(
            "chitti",
            "host_fs_exists",
            |caller: Caller<'_, HostState>, path_ptr: i32, path_len: i32| -> i32 {
                let Some(path) = read_guest_str(&caller, path_ptr, path_len) else {
                    return -1;
                };
                i32::from(crate::synapse::fs::exists(&path))
            },
        )
        .map_err(|_| "define host_fs_exists")?;

    // host_now_unix() -> i64  — wall-clock Unix seconds (commit timestamps).
    linker
        .func_wrap("chitti", "host_now_unix", |_caller: Caller<'_, HostState>| -> i64 {
            crate::clock::now_unix()
        })
        .map_err(|_| "define host_now_unix")?;

    // host_sha1(src_ptr, src_len, out_ptr) -> i32 (writes 20 bytes; 0 ok).
    linker
        .func_wrap(
            "chitti",
            "host_sha1",
            |mut caller: Caller<'_, HostState>,
             src_ptr: i32,
             src_len: i32,
             out_ptr: i32|
             -> i32 {
                let Some(src) = read_guest_bytes(&caller, src_ptr, src_len) else {
                    return -1;
                };
                let digest = crate::net::sha1::sha1(&src);
                if write_guest_bytes(&mut caller, out_ptr, &digest).is_err() {
                    return -1;
                }
                0
            },
        )
        .map_err(|_| "define host_sha1")?;

    // host_inflate(src, srclen, out, outcap) -> i64  (zlib decompress; low 32
    // bits = the **decompressed** length, high 32 bits = **input bytes consumed**
    // — a packfile packs one zlib stream per object, so the caller must know
    // where each stream ends; -1/-2 on error). Trailing bytes after the stream
    // are fine, which is what lets a caller hand it the rest of the pack and use
    // `consumed` to walk to the next object instead of searching for the end.
    linker
        .func_wrap(
            "chitti",
            "host_inflate",
            |mut caller: Caller<'_, HostState>,
             src_ptr: i32,
             src_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i64 {
                let Some(src) = read_guest_bytes(&caller, src_ptr, src_len) else {
                    return -1;
                };
                let Ok((dec, consumed)) = crate::image::inflate::zlib_decompress_len(&src) else {
                    return -2;
                };
                // The **decompressed** length, not the written one — see `fill_guest`.
                match fill_guest(&mut caller, out_ptr, out_cap, &dec) {
                    Some(n) => ((consumed as i64) << 32) | n as i64,
                    None => -1,
                }
            },
        )
        .map_err(|_| "define host_inflate")?;

    // host_deflate(src, srclen, out, outcap) -> i32  (zlib **stored-block**
    // compress — valid zlib real git accepts; the **compressed** length, or -1).
    linker
        .func_wrap(
            "chitti",
            "host_deflate",
            |mut caller: Caller<'_, HostState>,
             src_ptr: i32,
             src_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                let Some(src) = read_guest_bytes(&caller, src_ptr, src_len) else {
                    return -1;
                };
                let enc = zlib_deflate_stored(&src);
                // The **compressed** length, not the written one — see `fill_guest`.
                match fill_guest(&mut caller, out_ptr, out_cap, &enc) {
                    Some(n) => n as i32,
                    None => -1,
                }
            },
        )
        .map_err(|_| "define host_deflate")?;

    // host_http(req_ptr, req_len, out_ptr, out_cap) -> i64 = (status << 32) | len.
    // `req` JSON: {"m":"GET|POST","u":"url","h":"K: V; K: V","b":"<base64 body>"}.
    // Up to `out_cap` bytes of the response body are written to `out`; the low 32
    // bits of the return are the body's **full** length (so `len > out_cap` means
    // "retry with a buffer this big" — see `fill_guest`), the high bits the HTTP
    // status. Gated: the calling agent must declare a `net` capability in its
    // manifest.
    linker
        .func_wrap(
            "chitti",
            "host_http",
            |mut caller: Caller<'_, HostState>,
             req_ptr: i32,
             req_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i64 {
                if !crate::agent::system::has_net_cap(caller.data().bind.agent_id) {
                    return -1;
                }
                let Some(req) = read_guest_str(&caller, req_ptr, req_len) else {
                    return -2;
                };
                let Some(j) = crate::json::Json::parse(&req) else {
                    return -3;
                };
                let method = j.get("m").and_then(|v| v.as_str()).unwrap_or("GET").to_string();
                let url = j.get("u").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if url.is_empty() {
                    return -4;
                }
                let mut header_owns: alloc::vec::Vec<(String, String)> = alloc::vec::Vec::new();
                if let Some(h) = j.get("h").and_then(|v| v.as_str()) {
                    for pair in h.split(';') {
                        if let Some((k, v)) = pair.split_once(':') {
                            header_owns.push((k.trim().to_string(), v.trim().to_string()));
                        }
                    }
                }
                let headers: alloc::vec::Vec<(&str, &str)> = header_owns
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                let body = match j.get("b").and_then(|v| v.as_str()) {
                    Some(b64) => crate::net::ws::base64_decode(b64).unwrap_or_default(),
                    None => alloc::vec::Vec::new(),
                };
                let resp = match crate::net::http::request(&method, &url, &headers, &body, 30_000) {
                    Ok(r) => r,
                    Err(_) => return -5,
                };
                // The **body's** length, not the written one — see `fill_guest`. This
                // is the one that mattered: a git packfile is hundreds of KiB and the
                // guest's buffer was 64, so every clone of a real repository came back
                // as a truncated pack and died inside the decompressor.
                match fill_guest(&mut caller, out_ptr, out_cap, &resp.body) {
                    Some(n) => ((resp.status as i64) << 32) | n as i64,
                    None => -1,
                }
            },
        )
        .map_err(|_| "define host_http")?;

    // host_ssh(req_ptr, req_len, out_ptr, out_cap) -> i64
    //
    // One import, sub-commanded by the JSON's `op`, because SSH needs a *stateful*
    // stream and the guest can only hold an integer: git's protocol is
    // interactive — the server sends its ref advertisement, then waits for the
    // client's wants — so a single request/response call like `host_http` cannot
    // express it. The connection lives in `net::ssh::table` and the guest holds
    // an id.
    //
    //   {"op":"open","u":user,"h":host,"p":port,"c":command,"k":keypath} -> id
    //   {"op":"read","s":id}   -> up to out_cap bytes; low 32 bits = full length
    //   {"op":"write","s":id,"b":"<base64>"} -> 0
    //   {"op":"eof","s":id}    -> 0
    //   {"op":"close","s":id}  -> 0
    //
    // Gated exactly like `host_http`: the calling agent must declare a `net`
    // capability. Errors are negative and their text is written to `out` so the
    // guest can report *why* rather than a bare code.
    linker
        .func_wrap(
            "chitti",
            "host_ssh",
            |mut caller: Caller<'_, HostState>,
             req_ptr: i32,
             req_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i64 {
                let agent_id = caller.data().bind.agent_id;
                if !crate::agent::system::has_net_cap(agent_id) {
                    return -1;
                }
                let Some(req) = read_guest_str(&caller, req_ptr, req_len) else {
                    return -2;
                };
                let Some(j) = crate::json::Json::parse(&req) else {
                    return -3;
                };
                let op = j.get("op").and_then(|v| v.as_str()).unwrap_or("");
                let sid = j.get("s").and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;

                // Report an error's text through `out` — a bare -5 tells the user
                // nothing, and "permission denied" vs "no route" vs "bad key" are
                // the three things they need to tell apart.
                let mut fail = |caller: &mut Caller<'_, HostState>, msg: &str| -> i64 {
                    let _ = fill_guest(caller, out_ptr, out_cap, msg.as_bytes());
                    -5
                };

                match op {
                    "open" => {
                        let user = j.get("u").and_then(|v| v.as_str()).unwrap_or("git").to_string();
                        let host = j.get("h").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let port = j.get("p").and_then(|v| v.as_f64()).unwrap_or(22.0) as u16;
                        let cmd = j.get("c").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        if host.is_empty() || cmd.is_empty() {
                            return -4;
                        }
                        let keypath = j.get("k").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let key = match crate::shell::ssh::load_identity(keypath.as_deref()) {
                            Ok(k) => k,
                            Err(e) => return fail(&mut caller, &e),
                        };
                        match crate::net::ssh::table::open(
                            agent_id,
                            &user,
                            &host,
                            port,
                            &cmd,
                            key.as_ref(),
                        ) {
                            Ok(id) => id as i64,
                            Err(e) => fail(&mut caller, &e),
                        }
                    }
                    "read" => match crate::net::ssh::table::read(agent_id, sid) {
                        None => -6,
                        Some(Err(e)) => fail(&mut caller, &e),
                        Some(Ok(data)) => match fill_guest(&mut caller, out_ptr, out_cap, &data) {
                            Some(n) => n as i64,
                            None => -1,
                        },
                    },
                    "write" => {
                        let body = match j.get("b").and_then(|v| v.as_str()) {
                            Some(b64) => crate::net::ws::base64_decode(b64).unwrap_or_default(),
                            None => alloc::vec::Vec::new(),
                        };
                        match crate::net::ssh::table::write(agent_id, sid, &body) {
                            None => -6,
                            Some(Err(e)) => fail(&mut caller, &e),
                            Some(Ok(())) => 0,
                        }
                    }
                    "eof" => match crate::net::ssh::table::send_eof(agent_id, sid) {
                        None => -6,
                        Some(Err(e)) => fail(&mut caller, &e),
                        Some(Ok(())) => 0,
                    },
                    "close" => {
                        crate::net::ssh::table::close(agent_id, sid);
                        0
                    }
                    _ => -4,
                }
            },
        )
        .map_err(|_| "define host_ssh")?;

    Ok(())
}

/// zlib-compress `data` for the git agent's `host_deflate`.
///
/// Deliberately still **stored blocks** ([`crate::image::deflate::zlib_stored`])
/// and not the real compressor next to it: git objects written by this path are
/// read back by real `git`, so the bytes-are-in-there-literally property is
/// worth keeping while nothing has verified our fixed-Huffman output against
/// libz. Switching this to `zlib_compress` is a one-line change and a
/// deliberate one — a smaller `.git` is not worth a packfile no other git can
/// read, until an interop test says otherwise.
fn zlib_deflate_stored(data: &[u8]) -> alloc::vec::Vec<u8> {
    crate::image::deflate::zlib_stored(data)
}

/// Apply a settings-app preference to live shell/UI state.
fn apply_sys_pref(key: &str, val: &str) -> Result<(), &'static str> {
    match key {
        "theme" => {
            let name = val.trim();
            if name.is_empty() {
                return Err("empty theme");
            }
            crate::theme::apply(name).map_err(|_| "theme apply failed")
        }
        "mode" => {
            if crate::shell::set_approval_mode_name(val) {
                Ok(())
            } else {
                Err("unknown mode")
            }
        }
        "opacity" => {
            let parsed: u64 = val.trim().parse().map_err(|_| "bad opacity")?;
            let o = parsed.min(255);
            let mut cfg = crate::ui_config::current();
            cfg.opacity = o;
            crate::ui_config::set_config(cfg);
            Ok(())
        }
        _ => Err("unknown sys key"),
    }
}

fn read_sys_pref(key: &str) -> Option<String> {
    match key {
        "theme" => {
            let name = crate::ui_config::current().theme_name;
            Some(if name.is_empty() {
                String::from("dark")
            } else {
                name
            })
        }
        "mode" => Some(String::from(crate::shell::approval_mode_name())),
        "opacity" => Some(format!("{}", crate::ui_config::current().opacity)),
        _ => None,
    }
}

fn scope_from_i32(s: i32) -> StorageScope {
    if s == 1 {
        StorageScope::Durable
    } else {
        StorageScope::Session
    }
}

fn guest_memory(caller: &Caller<'_, HostState>) -> Option<Memory> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Some(m),
        _ => None,
    }
}

fn read_guest_bytes(caller: &Caller<'_, HostState>, ptr: i32, len: i32) -> Option<Vec<u8>> {
    if ptr < 0 || len < 0 {
        return None;
    }
    let mem = guest_memory(caller)?;
    let data = mem.data(caller);
    let p = ptr as usize;
    let l = len as usize;
    let end = p.checked_add(l)?;
    if end > data.len() {
        return None;
    }
    Some(data[p..end].to_vec())
}

fn read_guest_str(caller: &Caller<'_, HostState>, ptr: i32, len: i32) -> Option<String> {
    let b = read_guest_bytes(caller, ptr, len)?;
    String::from_utf8(b).ok()
}

/// Write as much of `data` as fits in the guest's `cap`-byte buffer at `out_ptr`
/// and report **how long `data` actually was** — not how much was written.
///
/// A guest cannot size a buffer before it knows the answer's size, and a host that
/// replies `min(len, cap)` makes "it fit" and "it did not" indistinguishable: a
/// 182 KiB packfile handed to a 64 KiB buffer came back as a perfectly plausible
/// complete 64 KiB one, and the git agent failed inflating the object that
/// straddled the cut — reported as `object inflate failed`, which points at the
/// decompressor rather than at the truncation two layers up.
///
/// So the contract is the image tenant's: the callee reports what it needed, the
/// caller grows its arena and asks again. A caller that does not care about
/// truncation must clamp the returned length to its own capacity before slicing.
fn fill_guest(
    caller: &mut Caller<'_, HostState>,
    out_ptr: i32,
    cap: i32,
    data: &[u8],
) -> Option<usize> {
    let n = data.len().min(cap.max(0) as usize);
    write_guest_bytes(caller, out_ptr, &data[..n]).ok()?;
    Some(data.len())
}

fn write_guest_bytes(
    caller: &mut Caller<'_, HostState>,
    ptr: i32,
    bytes: &[u8],
) -> Result<(), ()> {
    if ptr < 0 {
        return Err(());
    }
    let mem = guest_memory(caller).ok_or(())?;
    let data = mem.data_mut(&mut *caller);
    let p = ptr as usize;
    let end = p.checked_add(bytes.len()).ok_or(())?;
    if end > data.len() {
        return Err(());
    }
    data[p..end].copy_from_slice(bytes);
    Ok(())
}

/// Submit one Synapse call **from ring 3**, as `task`, and report whether it ran.
///
/// A wasm app's UI effects used to call `synapse::execute` directly, which meant they
/// skipped the tool router — and so skipped the userspace migration that routes an
/// agent's other effects through a tenant. The result was a split with nothing behind
/// it: chess's *filesystem* calls crossed the privilege boundary while its *board* calls
/// did not, for no reason except which code path they happened to take.
///
/// `Justification::trusted()` because that is what `synapse::execute` supplies by
/// default, and the rule for a migration is that it must not change what a caller may
/// do. The tenant does not choose this — the kernel sets it before entering ring 3.
fn syn_in_userspace(task: TaskId, raw: &str) -> bool {
    matches!(
        crate::synapse::tenant::invoke_in_userspace(task, raw, crate::security::taint::Justification::trusted()),
        Some(crate::synapse::Invocation::Executed { result, .. }) if result.starts_with("ok:")
    )
}

fn syn_board_set(task: TaskId, surface: u32, fen: &str) -> bool {
    let fen_esc = fen.replace('\\', "\\\\").replace('"', "\\\"");
    let raw = format!(
        r#"{{"name":"board_set","arguments":{{"surface":{surface},"fen":"{fen_esc}"}}}}"#
    );
    syn_in_userspace(task, &raw)
}

fn syn_board_mark(task: TaskId, surface: u32, squares: &str, color: &str) -> bool {
    let raw = format!(
        r#"{{"name":"board_mark","arguments":{{"surface":{surface},"squares":"{squares}","color":"{color}"}}}}"#
    );
    syn_in_userspace(task, &raw)
}

fn syn_ui_hud(task: TaskId, surface: u32, text: &str) -> bool {
    let esc = text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let raw = format!(r#"{{"name":"ui_hud","arguments":{{"surface":{surface},"text":"{esc}"}}}}"#);
    syn_in_userspace(task, &raw)
}

/// JSON-escape the two characters that can appear inside an op program's text.
fn esc_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Render one `ui_draw` call envelope around **already-escaped** ops.
fn render_ui_draw(surface: u32, ops_esc: &str) -> String {
    format!(r#"{{"name":"ui_draw","arguments":{{"surface":{surface},"ops":"{ops_esc}"}}}}"#)
}

/// Split a `;`-separated draw program into the fewest calls whose rendered
/// envelopes each fit `cap` bytes. Returns escaped op runs, ready for
/// [`render_ui_draw`].
///
/// This exists because a guest `ui_draw` crosses into ring 3, and
/// [`crate::synapse::tenant::block::CAPACITY`] (1920) bounds the whole call
/// text — envelope included. Before chunking, a program over that limit was
/// **silently not drawn**: `invoke_in_userspace` returned `None`, `host_ui_draw`
/// returned -1, and the guest had no way to tell that from any other failure. A
/// fresh 9×9 minesweeper board is ~2.2 KB of ops, so it never rendered at all,
/// and snake stopped repainting past roughly length 65.
///
/// Splitting is safe because ops are independent statements applied in order
/// (`apply_surface` in a loop), so N calls leave `back` in exactly the state one
/// call would have. The caller defers presents so the frame still presents once.
///
/// The envelope overhead is **measured, not counted by hand** — a miscounted
/// constant here would reintroduce the silent-drop bug at a slightly different
/// threshold, which is strictly harder to find than the original.
///
/// `None` means a *single* statement cannot fit even alone. That is reported
/// rather than truncated: a partial statement is either a parse error or, worse,
/// a valid statement with different arguments.
fn split_ops(surface: u32, ops: &str, cap: usize) -> Option<Vec<String>> {
    let budget = cap.checked_sub(render_ui_draw(surface, "").len())?;
    let mut out = Vec::new();
    let mut cur = String::new();
    for stmt in ops.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        let e = esc_json(stmt);
        if e.len() > budget {
            return None;
        }
        // +1 for the `;` that would rejoin it.
        let need = if cur.is_empty() { e.len() } else { cur.len() + 1 + e.len() };
        if need > budget {
            out.push(core::mem::take(&mut cur));
        } else if !cur.is_empty() {
            cur.push(';');
        }
        cur.push_str(&e);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Some(out)
}

/// What `host_ui_draw` reports, so a guest can tell the three cases apart.
enum UiDrawErr {
    /// One statement is too long to cross the boundary even by itself.
    Unsplittable,
    /// A chunk was refused by the gates, or userspace never ran it.
    Refused,
}

fn syn_ui_draw(task: TaskId, surface: u32, ops: &str) -> Result<(), UiDrawErr> {
    let cap = crate::synapse::tenant::block::CAPACITY;
    let Some(chunks) = split_ops(surface, ops, cap) else {
        crate::ktrace::log_fmt(format_args!(
            "ui_draw: a single op is longer than the {cap}-byte ring-3 call block — not drawn"
        ));
        return Err(UiDrawErr::Unsplittable);
    };
    // One present for the whole program, not one per chunk: presenting re-rasterizes
    // every label at pane scale, so a chunked frame would pay that N times.
    let mut ok = true;
    crate::synapse::ui::with_deferred_present(|| {
        for c in &chunks {
            if !syn_in_userspace(task, &render_ui_draw(surface, c)) {
                ok = false;
                break;
            }
        }
    });
    if ok {
        Ok(())
    } else {
        Err(UiDrawErr::Refused)
    }
}

fn syn_ui_present(task: TaskId, surface: u32, w: usize, h: usize) -> bool {
    let raw = format!(
        r#"{{"name":"ui_present","arguments":{{"surface":{surface},"w":{w},"h":{h}}}}}"#
    );
    syn_in_userspace(task, &raw)
}

fn syn_audio_submit(task: TaskId, frames: usize, rate: u32, channels: u8) -> bool {
    let raw = format!(
        r#"{{"name":"audio_submit","arguments":{{"frames":{frames},"rate":{rate},"channels":{channels}}}}}"#
    );
    syn_in_userspace(task, &raw)
}

/// The WASI file surface a ported C program needs, **read-only** and backed by
/// the Synapse store.
///
/// `register_wasi_imports` covers stdio; this adds the eight preview1 functions a
/// libc reaches for `fopen`. Together they are exactly the 14 imports a
/// doomgeneric build emits — a list obtained by parsing the module's import
/// section rather than by guessing, because wasmi resolves an import by name
/// **and `FuncType`** and a single wrong arity makes the whole module fail with
/// `missing imports`, a message that names nothing.
///
/// **Read-only, and every mutator fails honestly.** `path_rename`,
/// `path_unlink_file`, `path_create_directory` and `path_remove_directory` return
/// `NOENT`. A guest that needs to write has `host_fs_write`, which is home-scoped;
/// routing writes through a second, unscoped path would widen authority by way of
/// a compatibility shim. A C program treats a failed config write as "no config"
/// and carries on, which is the behaviour wanted here.
///
/// **One open file, one offset.** The guest this exists for opens one file once.
/// A descriptor table would need an eviction policy for a case that does not
/// arise.
///
/// **`path_open` succeeds but reading through it does not.** `fd_read`/`fd_seek`/
/// `fd_close` belong to the stdio set and answer `BADF` for any descriptor they do
/// not own, so this surface answers *"does this file exist"* and nothing more.
/// That is the whole requirement — a ported program gets its actual bytes through
/// `host_fs_read` into linear memory, where it can address them directly, rather
/// than streaming them a `fread` at a time across the boundary. Stated because the
/// asymmetry is surprising: an `fopen` that works followed by an `fread` that
/// returns nothing. If a guest ever genuinely needs to stream a file, give this
/// its own `fd_read` arm keyed on `FILE_FD` and serve it from `synapse::fs`;
/// do not widen the stdio one.
fn register_wasi_fs_imports(linker: &mut Linker<HostState>) -> Result<(), &'static str> {
    const W: &str = "wasi_snapshot_preview1";
    const NOENT: i32 = 44;
    const BADF: i32 = 8;
    /// The preopened directory. A libc will not resolve any path — absolute or
    /// relative — without at least one preopen; it fails inside its own path
    /// resolution before ever calling `path_open`.
    const PREOPEN_FD: i32 = 3;
    /// The single file descriptor this surface can hand out. Above stdio and the
    /// preopen so it cannot collide with either.
    const FILE_FD: i32 = 4;

    fn read_path(caller: &Caller<'_, HostState>, ptr: i32, len: i32) -> Option<String> {
        let mem = caller.get_export("memory").and_then(|e| e.into_memory())?;
        let data = mem.data(caller);
        let start = ptr.max(0) as usize;
        let end = start.checked_add(len.max(0) as usize)?;
        if end > data.len() {
            return None;
        }
        core::str::from_utf8(&data[start..end]).ok().map(String::from)
    }

    /// Resolve a guest path against the store.
    ///
    /// A guest path arrives relative to the preopen, so a leading `/` is stripped
    /// and one added back: the store is rooted, and a path that kept the guest's
    /// own notion of `/` would name a different file on every libc.
    fn store_path(p: &str) -> String {
        let t = p.trim_start_matches('/');
        format!("/{t}")
    }

    linker
        .func_wrap(
            W,
            "fd_prestat_get",
            |mut caller: Caller<'_, HostState>, fd: i32, out: i32| -> i32 {
                if fd != PREOPEN_FD {
                    return BADF;
                }
                // { u8 tag = dir, pad[3], u32 name_len }. The padding is real:
                // name_len is at offset 4, not 1.
                if let Some(m) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                    let mut buf = [0u8; 8];
                    buf[4..8].copy_from_slice(&1u32.to_le_bytes());
                    let _ = m.write(&mut caller, out as usize, &buf);
                }
                0
            },
        )
        .map_err(|_| "define wasi.fd_prestat_get")?;

    linker
        .func_wrap(
            W,
            "fd_prestat_dir_name",
            |mut caller: Caller<'_, HostState>, fd: i32, path: i32, len: i32| -> i32 {
                if fd != PREOPEN_FD || len < 1 {
                    return BADF;
                }
                if let Some(m) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                    let _ = m.write(&mut caller, path as usize, b"/");
                }
                0
            },
        )
        .map_err(|_| "define wasi.fd_prestat_dir_name")?;

    linker
        .func_wrap(
            W,
            "path_open",
            |mut caller: Caller<'_, HostState>,
             _dirfd: i32,
             _dirflags: i32,
             path: i32,
             path_len: i32,
             oflags: i32,
             _base: i64,
             _inherit: i64,
             _fdflags: i32,
             out_fd: i32|
             -> i32 {
                // OFLAGS_CREAT (1) | OFLAGS_TRUNC (8) mean the guest intends to
                // write. Refused rather than silently opened read-only, which
                // would look like success and then lose the data.
                if oflags & 0b1001 != 0 {
                    return NOENT;
                }
                let Some(p) = read_path(&caller, path, path_len) else {
                    return NOENT;
                };
                let full = store_path(&p);
                if !crate::synapse::fs::exists(&full) {
                    return NOENT;
                }
                caller.data_mut().wasi_file = Some((full, 0));
                if let Some(m) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                    let _ = m.write(&mut caller, out_fd as usize, &FILE_FD.to_le_bytes());
                }
                0
            },
        )
        .map_err(|_| "define wasi.path_open")?;

    linker
        .func_wrap(
            W,
            "fd_fdstat_set_flags",
            |_: Caller<'_, HostState>, _: i32, _: i32| -> i32 { 0 },
        )
        .map_err(|_| "define wasi.fd_fdstat_set_flags")?;

    // Every mutator: refused. See the function doc — writes go through the
    // home-scoped `host_fs_write`, not through a compatibility shim.
    for name in [
        "path_create_directory",
        "path_remove_directory",
        "path_unlink_file",
    ] {
        linker
            .func_wrap(W, name, |_: Caller<'_, HostState>, _: i32, _: i32, _: i32| -> i32 {
                NOENT
            })
            .map_err(|_| "define wasi path mutator")?;
    }
    linker
        .func_wrap(
            W,
            "path_rename",
            |_: Caller<'_, HostState>, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32| -> i32 {
                NOENT
            },
        )
        .map_err(|_| "define wasi.path_rename")?;

    Ok(())
}

fn map_trap(err: wasmi::Error) -> &'static str {
    let msg = format!("{err}");
    if msg.contains("fuel") || msg.contains("Fuel") {
        "wasm fuel exhausted"
    } else if msg.contains("out of bounds") || msg.contains("OutOfBounds") {
        "wasm memory out of bounds"
    } else if msg.contains("Unreachable") || msg.contains("unreachable") {
        "wasm unreachable"
    } else {
        "wasm trap"
    }
}

// ---------------------------------------------------------------------------
// Hand-written WASM fixtures (no wat crate in the kernel)
// ---------------------------------------------------------------------------

/// `(module (func (export "add") (param i32 i32) (result i32)
///    local.get 0 local.get 1 i32.add))`
pub const FIXTURE_ADD: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, // magic
    0x01, 0x00, 0x00, 0x00, // version
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
];

/// `(module (func (export "spin") (loop br 0)))`
pub const FIXTURE_SPIN: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x08, 0x01, 0x04, 0x73, 0x70, 0x69, 0x6e, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x03, 0x40, 0x0c, 0x00, 0x0b, 0x0b,
];

/// Echo string ABI fixture (memory + echo export).
pub const FIXTURE_ECHO: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x08, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x02, 0x7f, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x05, 0x03, 0x01, 0x00, 0x01,
    0x07, 0x11, 0x02, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00, 0x04, 0x65, 0x63,
    0x68, 0x6f, 0x00, 0x00,
    0x0a, 0x08, 0x01, 0x06, 0x00, 0x20, 0x00, 0x20, 0x01, 0x0b,
];

/// A module importing the **real** `wasi_snapshot_preview1` signatures: reads
/// stdin into a buffer and echoes it to stdout, returning the byte count.
/// 
/// ```wat
/// (import "wasi_snapshot_preview1" "fd_read"  (func (param i32 i32 i32 i32) (result i32)))
/// (import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))
/// (memory (export "memory") 1)
/// (func (export "echo") (result i32)  ;; iovecs at 0/16, counts at 8/24, buffer at 64
///   ...)
/// ```
/// 
/// Generated by `tools/wasmgen`-style emission and verified by running it under
/// wasmi before being pasted here — the same out-of-band route the other fixtures
/// in this file took (there is no `wat` crate in the kernel).
pub const FIXTURE_WASI_ECHO: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0d, 0x02, 0x60,
    0x04, 0x7f, 0x7f, 0x7f, 0x7f, 0x01, 0x7f, 0x60, 0x00, 0x01, 0x7f, 0x02,
    0x44, 0x02, 0x16, 0x77, 0x61, 0x73, 0x69, 0x5f, 0x73, 0x6e, 0x61, 0x70,
    0x73, 0x68, 0x6f, 0x74, 0x5f, 0x70, 0x72, 0x65, 0x76, 0x69, 0x65, 0x77,
    0x31, 0x07, 0x66, 0x64, 0x5f, 0x72, 0x65, 0x61, 0x64, 0x00, 0x00, 0x16,
    0x77, 0x61, 0x73, 0x69, 0x5f, 0x73, 0x6e, 0x61, 0x70, 0x73, 0x68, 0x6f,
    0x74, 0x5f, 0x70, 0x72, 0x65, 0x76, 0x69, 0x65, 0x77, 0x31, 0x08, 0x66,
    0x64, 0x5f, 0x77, 0x72, 0x69, 0x74, 0x65, 0x00, 0x00, 0x03, 0x02, 0x01,
    0x01, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x11, 0x02, 0x06, 0x6d, 0x65,
    0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00, 0x04, 0x65, 0x63, 0x68, 0x6f, 0x00,
    0x02, 0x0a, 0x40, 0x01, 0x3e, 0x00, 0x41, 0x00, 0x41, 0xc0, 0x00, 0x36,
    0x02, 0x00, 0x41, 0x04, 0x41, 0x20, 0x36, 0x02, 0x00, 0x41, 0x00, 0x41,
    0x00, 0x41, 0x01, 0x41, 0x08, 0x10, 0x00, 0x1a, 0x41, 0x10, 0x41, 0xc0,
    0x00, 0x36, 0x02, 0x00, 0x41, 0x14, 0x41, 0x08, 0x28, 0x02, 0x00, 0x36,
    0x02, 0x00, 0x41, 0x01, 0x41, 0x10, 0x41, 0x01, 0x41, 0x18, 0x10, 0x01,
    0x1a, 0x41, 0x08, 0x28, 0x02, 0x00, 0x0b,
];

/// `memory.grow` bomb for page-limiter tests.
pub const FIXTURE_GROW: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x05, 0x03, 0x01, 0x00, 0x01,
    0x07, 0x11, 0x02, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00, 0x04, 0x67, 0x72,
    0x6f, 0x77, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0xe8, 0x07, 0x40, 0x00, 0x0b,
];

/// Guest that imports host storage + surface (see `/tmp/storage_host.wat`).
pub const FIXTURE_HOST_STORAGE: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x16, 0x04, 0x60, 0x05, 0x7f, 0x7f,
    0x7f, 0x7f, 0x7f, 0x01, 0x7f, 0x60, 0x02, 0x7f, 0x7f, 0x00, 0x60, 0x00, 0x01, 0x7f, 0x60,
    0x00, 0x00, 0x02, 0x60, 0x04, 0x06, 0x63, 0x68, 0x69, 0x74, 0x74, 0x69, 0x10, 0x68, 0x6f,
    0x73, 0x74, 0x5f, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x5f, 0x73, 0x65, 0x74, 0x00,
    0x00, 0x06, 0x63, 0x68, 0x69, 0x74, 0x74, 0x69, 0x10, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x73,
    0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x5f, 0x67, 0x65, 0x74, 0x00, 0x00, 0x06, 0x63, 0x68,
    0x69, 0x74, 0x74, 0x69, 0x08, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x6c, 0x6f, 0x67, 0x00, 0x01,
    0x06, 0x63, 0x68, 0x69, 0x74, 0x74, 0x69, 0x0f, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x73, 0x75,
    0x72, 0x66, 0x61, 0x63, 0x65, 0x5f, 0x69, 0x64, 0x00, 0x02, 0x03, 0x04, 0x03, 0x02, 0x02,
    0x03, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x29, 0x04, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72,
    0x79, 0x02, 0x00, 0x09, 0x72, 0x6f, 0x75, 0x6e, 0x64, 0x74, 0x72, 0x69, 0x70, 0x00, 0x04,
    0x07, 0x73, 0x75, 0x72, 0x66, 0x61, 0x63, 0x65, 0x00, 0x05, 0x06, 0x6c, 0x6f, 0x67, 0x5f,
    0x68, 0x69, 0x00, 0x06, 0x0a, 0x2c, 0x03, 0x1c, 0x00, 0x41, 0x00, 0x41, 0x00, 0x41, 0x01,
    0x41, 0x10, 0x41, 0x05, 0x10, 0x00, 0x1a, 0x41, 0x00, 0x41, 0x00, 0x41, 0x01, 0x41, 0xc0,
    0x00, 0x41, 0x20, 0x10, 0x01, 0x0b, 0x04, 0x00, 0x10, 0x03, 0x0b, 0x08, 0x00, 0x41, 0x10,
    0x41, 0x05, 0x10, 0x02, 0x0b, 0x0b, 0x11, 0x02, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x6b, 0x00,
    0x41, 0x10, 0x0b, 0x05, 0x68, 0x65, 0x6c, 0x6c, 0x6f,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every chunk must actually fit the ring-3 call block once enveloped —
    /// which is the whole point, and the thing a hand-counted overhead
    /// constant would get subtly wrong.
    fn assert_chunks_fit(surface: u32, ops: &str, cap: usize) -> Vec<String> {
        let chunks = split_ops(surface, ops, cap).expect("splittable");
        for c in &chunks {
            let rendered = render_ui_draw(surface, c);
            assert!(
                rendered.len() <= cap,
                "chunk renders to {} bytes, over the {cap}-byte block: {rendered}",
                rendered.len()
            );
        }
        chunks
    }

    /// Rejoining the chunks must reproduce the original program, statement for
    /// statement — a chunker that dropped or duplicated one would still produce
    /// a plausible picture, which is exactly the failure mode being fixed.
    #[test_case]
    fn split_ops_preserves_every_statement() {
        let ops = (0..200)
            .map(|i| format!("rect {i} {i} 4 4 ff0000"))
            .collect::<Vec<_>>()
            .join(";");
        let chunks = assert_chunks_fit(7, &ops, crate::synapse::tenant::block::CAPACITY);
        assert!(chunks.len() > 1, "200 rects must need more than one call");
        assert_eq!(chunks.join(";"), ops);
    }

    /// A program that already fits must stay exactly one call, so the common
    /// case pays nothing for chunking existing.
    #[test_case]
    fn a_small_program_is_one_call() {
        let ops = "clear 000000;rect 1 2 3 4 ffffff";
        let chunks = assert_chunks_fit(1, ops, crate::synapse::tenant::block::CAPACITY);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], ops);
    }

    /// The two shipped regressions, pinned as cases: a fresh 9x9 minesweeper
    /// board (~2.2 KB) and a long snake body both exceed the block and must now
    /// split rather than vanish.
    #[test_case]
    fn the_boards_that_silently_did_not_draw_now_split() {
        let cap = crate::synapse::tenant::block::CAPACITY;
        // Shape of mines::paint — a cell rect plus a label per cell.
        let mut ops = String::from("clear 1a1a2e");
        for r in 0..9 {
            for c in 0..9 {
                ops.push_str(&format!(";rect {} {} 18 18 303048", 16 + c * 20, 6 + r * 20));
            }
        }
        assert!(ops.len() > cap, "fixture must exceed the block to be a regression test");
        let chunks = assert_chunks_fit(3, &ops, cap);
        assert_eq!(chunks.join(";"), ops);

        // Shape of snake::paint at a length that used to stop repainting.
        let mut snake = String::from("clear 000000");
        for i in 0..90 {
            snake.push_str(&format!(";rect {} {} 6 6 00ff00", i % 40 * 6, i / 40 * 6));
        }
        assert!(snake.len() > cap);
        let chunks = assert_chunks_fit(4, &snake, cap);
        assert_eq!(chunks.join(";"), snake);
    }

    /// A single unsplittable statement is reported, never truncated: half a
    /// statement is either a parse error or a valid statement with different
    /// arguments, and both are worse than an error the guest can see.
    #[test_case]
    fn one_oversized_statement_is_refused_not_truncated() {
        let huge = format!("text 0 0 12 ffffff {}", "x".repeat(4096));
        assert!(split_ops(1, &huge, crate::synapse::tenant::block::CAPACITY).is_none());
        // And it is refused even when other statements would have fit.
        let mixed = format!("clear 000000;{huge};rect 0 0 1 1 ffffff");
        assert!(split_ops(1, &mixed, crate::synapse::tenant::block::CAPACITY).is_none());
    }

    /// Escaping happens before the fit check, so a program of quotes cannot
    /// pass the budget and then overflow the block once escaped.
    #[test_case]
    fn escaping_is_counted_against_the_budget() {
        let cap = crate::synapse::tenant::block::CAPACITY;
        // Each `"` becomes two bytes; unescaped this fits, escaped it must split.
        let one = "text 0 0 12 ffffff ".to_string() + &"\"".repeat(64);
        let ops = (0..20).map(|_| one.clone()).collect::<Vec<_>>().join(";");
        assert!(ops.len() < cap * 4);
        let _ = assert_chunks_fit(2, &ops, cap);
    }

    /// Empty and separator-only programs are not calls at all. `draw` with an
    /// empty program would still present, so emitting zero chunks is correct.
    #[test_case]
    fn an_empty_program_makes_no_calls() {
        let cap = crate::synapse::tenant::block::CAPACITY;
        assert_eq!(split_ops(1, "", cap).expect("ok").len(), 0);
        assert_eq!(split_ops(1, ";;  ;", cap).expect("ok").len(), 0);
    }

    /// A capacity smaller than the envelope itself must not underflow into a
    /// vast budget — `checked_sub` is load-bearing here.
    #[test_case]
    fn a_capacity_below_the_envelope_is_none() {
        assert!(split_ops(1, "clear 000000", 4).is_none());
    }

    /// The Freedoom package's real module must instantiate against the real import
    /// set. This is the test that would otherwise not exist until someone booted
    /// the OS and typed `/agents start doom`.
    ///
    /// It is worth its cost because the failure mode is uniquely unhelpful: wasmi
    /// resolves an import by name **and** `FuncType`, so one missing function or
    /// one wrong arity fails the whole module with `missing imports/limits?` —
    /// naming neither the function nor the type. A libc-built guest imports ~14
    /// WASI functions it may never call, and every one of them has to be present.
    #[test_case]
    fn the_freedoom_module_instantiates_against_the_native_import_set() {
        const DOOM: &[u8] = include_bytes!("../../../agents/freedoom/assets/tools.wasm");
        // Same limits the package manifest declares, so this fails here rather
        // than at `/agents start` if a limit is too tight — the table-elems trap
        // the PDF renderer hit at 693 against a hardcoded 256.
        // 2048 pages = the module's own `--initial-memory` of 128 MiB. A limit
        // below a module's *initial* memory is refused at instantiation, and the
        // message says `missing imports/limits?` without saying which — this test
        // caught exactly that at 1024.
        let limits = Limits::default()
            .with_fuel(200_000_000)
            .with_pages(2048)
            .with_table_elems(4096);
        let s = Session::instantiate_native(DOOM, limits, HostBindings::default())
            .expect("freedoom module must instantiate against HostImportSet::Native");
        drop(s);

        // And it must *not* instantiate on the narrow set — otherwise `wasm.native`
        // is decorative and a future guest would silently lose its WASI imports.
        let limits = Limits::default()
            .with_fuel(200_000_000)
            .with_pages(2048)
            .with_table_elems(4096);
        assert!(
            Session::instantiate(DOOM, limits, HostBindings::default()).is_err(),
            "a libc guest must need the native set; if this passes, the opt-in is meaningless"
        );
    }

    #[test_case]
    fn add_i32_fixture() {
        let r = call_i32_i32(FIXTURE_ADD, "add", 2, 3, Limits::default().with_fuel(10_000))
            .expect("add");
        assert_eq!(r, 5);
        let r = call_i32_i32(FIXTURE_ADD, "add", -1, 1, Limits::default().with_fuel(10_000))
            .expect("add neg");
        assert_eq!(r, 0);
    }

    #[test_case]
    fn fuel_exhausts_infinite_loop() {
        let err = call_void(FIXTURE_SPIN, "spin", Limits::default().with_fuel(50))
            .expect_err("spin must run out of fuel");
        assert!(err.contains("fuel"), "expected fuel error, got: {err}");
    }

    #[test_case]
    fn string_echo_abi() {
        let out = call_string(
            FIXTURE_ECHO,
            "echo",
            r#"{"from":"e2"}"#,
            Limits::default().with_fuel(50_000),
        )
        .expect("echo");
        assert_eq!(out, r#"{"from":"e2"}"#);
    }

    /// A live instance can be **moved to another core**, which is the only form
    /// of SMP available to a wasm guest here.
    ///
    /// wasmi is a single-threaded interpreter and the kernel implements none of
    /// the wasm `threads` proposal, so nothing inside one call can be split
    /// across cores. What *is* possible is loaning a whole `Session` to an SMP
    /// worker the way the video player loans its decoder (`smp::async_submit`):
    /// the UI keeps pumping on the BSP while a page renders elsewhere, and the
    /// next page can be rendered ahead while the reader looks at this one.
    ///
    /// That requires `Session: Send`, which is a property of wasmi's types plus
    /// our `HostState` and is easy to lose by accident — one `Rc` in host state
    /// would end it. This asserts it at compile time so the groundwork cannot rot
    /// before the render-ahead path is built on it.
    #[test_case]
    fn a_session_is_send_so_it_can_be_loaned_to_an_smp_worker() {
        fn assert_send<T: Send>() {}
        assert_send::<Session>();
        assert_send::<HostState>();
    }

    /// A module that imports the **real** WASI signatures instantiates, and fd 0
    /// / fd 1 actually move bytes.
    ///
    /// Nothing tested this surface before, which is exactly why five stubs
    /// declared `() -> i32` — the wrong arity for every one of them — sat here
    /// unnoticed. wasmi resolves an import by name *and* type, so the symptom was
    /// not a wrong result but `"wasm instantiate failed (missing imports/limits?)"`,
    /// a message that never mentions types. The only module on the old surface
    /// (`pdfrender.wasm`) has no imports at all, so nothing ever asked.
    #[test_case]
    fn a_wasi_module_can_instantiate_and_do_stdio() {
        let mut s = Session::instantiate_js(
            FIXTURE_WASI_ECHO,
            Limits::default().with_fuel(200_000),
            HostBindings::default(),
        )
        .expect("a real WASI importer must instantiate");
        let msg = br#"{"hello":"wasi"}"#;
        s.set_stdin(msg);
        let n = s.call_i32("echo").expect("echo");
        assert_eq!(n as usize, msg.len(), "fd_read must report what it handed over");
        assert_eq!(s.take_stdout(), msg.to_vec(), "fd_write must carry the bytes through");
        // Draining is real: a second call reads EOF and echoes nothing.
        let n2 = s.call_i32("echo").expect("echo again");
        assert_eq!(n2, 0, "stdin is consumed, not re-served");
        assert!(s.take_stdout().is_empty());
    }

    /// The same guest instantiates on the **Page** surface too, because both sets
    /// register one shared shim.
    ///
    /// So the fd plumbing is *present* for a browser module, not absent — worth
    /// stating plainly rather than implying isolation that is not implemented.
    /// What makes it inert there is that nothing on the page path ever calls
    /// `set_stdin`/`take_stdout`: stdin is empty and stdout is dropped, and the
    /// buffers live in the per-`Store` `HostState`, so no two guests share them.
    /// If a page ever needs to be denied stdio outright, that is a separate,
    /// deliberate change — a distinct shim for the Page set.
    #[test_case]
    fn page_surface_links_the_same_wasi_shim() {
        let mut s = Session::instantiate_page(FIXTURE_WASI_ECHO, Limits::default().with_fuel(200_000))
            .expect("page surface must still instantiate a WASI importer");
        s.set_stdin(b"ignored");
        let n = s.call_i32("echo").expect("echo");
        assert_eq!(n, 7, "the shim is shared, so fd 0 still reads");
        assert_eq!(s.take_stdout(), b"ignored".to_vec());
    }

    #[test_case]
    fn memory_grow_denied_by_limiter() {
        let mut s = Session::instantiate(
            FIXTURE_GROW,
            Limits {
                fuel: 50_000,
                max_memory_pages: 2,
                ..Limits::default()
            },
            HostBindings::default(),
        )
        .expect("instantiate");
        let func = s
            .instance
            .get_typed_func::<(), i32>(&s.store, "grow")
            .expect("grow export");
        let r = func.call(&mut s.store, ()).expect("grow call");
        assert_eq!(r, -1, "memory.grow must fail when over page cap");
    }

    #[test_case]
    fn validate_rejects_garbage() {
        assert!(validate_module(b"not wasm").is_err());
        assert!(validate_module(FIXTURE_ADD).is_ok());
    }

    #[test_case]
    fn empty_module_refused() {
        assert!(Session::instantiate(&[], Limits::default(), HostBindings::default()).is_err());
    }

    #[test_case]
    fn host_storage_roundtrip_via_import() {
        let agent_id = 7777u64;
        storage::clear_session(agent_id);
        let bind = HostBindings {
            agent_id,
            task: 0,
            surface: 42,
        };
        let mut s = Session::instantiate(
            FIXTURE_HOST_STORAGE,
            Limits::default().with_fuel(100_000),
            bind,
        )
        .expect("instantiate host fixture");
        // surface export should see binding
        let sid = s.call_i32("surface").expect("surface");
        assert_eq!(sid, 42);
        // roundtrip: set k=hello, get → len 5
        let n = s.call_i32("roundtrip").expect("roundtrip");
        assert_eq!(n, 5, "host_storage_get should return value length");
        assert_eq!(
            storage::get_str(agent_id, StorageScope::Session, "k").as_deref(),
            Some("hello")
        );
        s.call_void("log_hi").expect("log");
        storage::clear_session(agent_id);
    }

    #[test_case]
    fn chess_tools_wasm_legal_from_start() {
        // Real package module (tools/chess-wasm → agents/chess/assets/tools.wasm).
        let wasm = include_bytes!("../../../agents/chess/assets/tools.wasm");
        assert!(validate_module(wasm).is_ok());
        let args = format!(
            r#"{{"fen":"{}","from":"e2"}}"#,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        let out = call_string_bound(
            wasm,
            "chess_legal",
            &args,
            Limits::default().with_fuel(500_000),
            HostBindings {
                agent_id: 9003,
                task: 0,
                surface: 0,
            },
        )
        .expect("chess_legal");
        assert!(
            out.contains("e3") && out.contains("e4"),
            "expected pawn doubles, got {out}"
        );
        assert!(out.starts_with("legal:e2->"), "got {out}");
    }

    #[test_case]
    fn host_board_set_without_task_returns_error_code() {
        // (module
        //   (import "chitti" "host_board_set" (func $bs (param i32 i32) (result i32)))
        //   (memory (export "memory") 1)
        //   (data (i32.const 0) "fen")
        //   (func (export "try") (result i32)
        //     (call $bs (i32.const 0) (i32.const 3))))
        // Generated via wat::parse (no custom name section).
        const MOD: &[u8] = &[
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x0b, 0x02, 0x60, 0x02, 0x7f,
            0x7f, 0x01, 0x7f, 0x60, 0x00, 0x01, 0x7f, 0x02, 0x19, 0x01, 0x06, 0x63, 0x68, 0x69,
            0x74, 0x74, 0x69, 0x0e, 0x68, 0x6f, 0x73, 0x74, 0x5f, 0x62, 0x6f, 0x61, 0x72, 0x64,
            0x5f, 0x73, 0x65, 0x74, 0x00, 0x00, 0x03, 0x02, 0x01, 0x01, 0x05, 0x03, 0x01, 0x00,
            0x01, 0x07, 0x10, 0x02, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00, 0x03,
            0x74, 0x72, 0x79, 0x00, 0x01, 0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0x41, 0x03,
            0x10, 0x00, 0x0b, 0x0b, 0x09, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x03, 0x66, 0x65, 0x6e,
        ];
        let mut s = Session::instantiate(
            MOD,
            Limits::default().with_fuel(10_000),
            HostBindings {
                agent_id: 1,
                task: 0,
                surface: 0,
            },
        )
        .expect("mod");
        let code = s.call_i32("try").expect("try");
        assert_eq!(code, -2, "no task/surface must return -2");
    }
}
