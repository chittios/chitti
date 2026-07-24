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
//! | `host_surface_id` | active surface for this binding |
//! | `host_now_ms` / `host_log` | clock + ktrace |
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
use alloc::vec::Vec;
use wasmi::errors::{MemoryError, TableError};
use wasmi::{Caller, Config, Engine, Extern, Linker, Memory, Module, ResourceLimiter, Store};

/// Default instruction fuel when a tool entry does not specify one.
pub const DEFAULT_FUEL: u64 = 5_000_000;
/// Default max linear memory: 32 × 64 KiB = 2 MiB (chess tools.wasm starts at 18 pages).
pub const DEFAULT_MAX_MEMORY_PAGES: u32 = 32;
/// Wasm page size in bytes.
const PAGE_SIZE: usize = 64 * 1024;

/// Limits applied to one module instance / store.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub fuel: u64,
    pub max_memory_pages: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_FUEL,
            max_memory_pages: DEFAULT_MAX_MEMORY_PAGES,
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
}

struct PageLimiter {
    max_bytes: usize,
}

impl PageLimiter {
    fn new(max_pages: u32) -> Self {
        Self {
            max_bytes: (max_pages as usize).saturating_mul(PAGE_SIZE),
        }
    }
}

impl ResourceLimiter for PageLimiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool, MemoryError> {
        Ok(desired <= self.max_bytes)
    }

    fn table_growing(
        &mut self,
        _current: u32,
        desired: u32,
        _maximum: Option<u32>,
    ) -> Result<bool, TableError> {
        Ok(desired <= 256)
    }

    fn instances(&self) -> usize {
        1
    }

    fn tables(&self) -> usize {
        1
    }

    fn memories(&self) -> usize {
        1
    }
}

fn make_engine() -> Engine {
    let mut config = Config::default();
    config.consume_fuel(true);
    config.ignore_custom_sections(true);
    Engine::new(&config)
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
            limiter: PageLimiter::new(limits.max_memory_pages),
            bind,
            last_error: None,
            log_count: 0,
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
        }
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|_| "wasm instantiate failed (missing imports?)")?
            .start(&mut store)
            .map_err(|_| "wasm start function failed")?;

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
        let need_pages = ((len + PAGE_SIZE - 1) / PAGE_SIZE) as u32;
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
    // WASI stubs: return ENOSYS (-1) so modules that probe WASI don't get FS.
    for (mod_name, field) in [
        ("wasi_snapshot_preview1", "fd_write"),
        ("wasi_snapshot_preview1", "fd_close"),
        ("wasi_snapshot_preview1", "environ_get"),
        ("wasi_snapshot_preview1", "environ_sizes_get"),
        ("wasi_snapshot_preview1", "proc_exit"),
    ] {
        let _ = linker.func_wrap(mod_name, field, |_caller: Caller<'_, HostState>| -> i32 { -1 });
    }
    Ok(())
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
                match syn_ui_draw(bind.task, bind.surface, &ops) {
                    true => 0,
                    false => -1,
                }
            },
        )
        .map_err(|_| "define host_ui_draw")?;

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

    Ok(())
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

fn syn_board_set(task: TaskId, surface: u32, fen: &str) -> bool {
    let fen_esc = fen.replace('\\', "\\\\").replace('"', "\\\"");
    let raw = format!(
        r#"{{"name":"board_set","arguments":{{"surface":{surface},"fen":"{fen_esc}"}}}}"#
    );
    matches!(
        crate::synapse::execute(task, &raw),
        crate::synapse::Invocation::Executed { result, .. } if result.starts_with("ok:")
    )
}

fn syn_board_mark(task: TaskId, surface: u32, squares: &str, color: &str) -> bool {
    let raw = format!(
        r#"{{"name":"board_mark","arguments":{{"surface":{surface},"squares":"{squares}","color":"{color}"}}}}"#
    );
    matches!(
        crate::synapse::execute(task, &raw),
        crate::synapse::Invocation::Executed { result, .. } if result.starts_with("ok:")
    )
}

fn syn_ui_hud(task: TaskId, surface: u32, text: &str) -> bool {
    let esc = text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let raw = format!(r#"{{"name":"ui_hud","arguments":{{"surface":{surface},"text":"{esc}"}}}}"#);
    matches!(
        crate::synapse::execute(task, &raw),
        crate::synapse::Invocation::Executed { result, .. } if result.starts_with("ok:")
    )
}

fn syn_ui_draw(task: TaskId, surface: u32, ops: &str) -> bool {
    let ops_esc = ops.replace('\\', "\\\\").replace('"', "\\\"");
    let raw = format!(
        r#"{{"name":"ui_draw","arguments":{{"surface":{surface},"ops":"{ops_esc}"}}}}"#
    );
    matches!(
        crate::synapse::execute(task, &raw),
        crate::synapse::Invocation::Executed { result, .. } if result.starts_with("ok:")
    )
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

    #[test_case]
    fn memory_grow_denied_by_limiter() {
        let mut s = Session::instantiate(
            FIXTURE_GROW,
            Limits {
                fuel: 50_000,
                max_memory_pages: 2,
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
