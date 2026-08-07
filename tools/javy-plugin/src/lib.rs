//! **ChittiOS's QuickJS plugin** — the JS engine plus the kernel's gated host
//! surface, exposed to scripts as a `Chitti` global.
//!
//! # Why this exists
//!
//! The stock Javy plugin gives a script stdio and nothing else. An agent's tools
//! need what a hand-written Rust tool module already has: the `chitti.host_*`
//! imports — storage, its own filesystem home, the clock, logging, hashing, HTTP
//! when the manifest grants it. Upstream's answer to "I need more than stdio" is to
//! build your own plugin (`docs/docs-using-extending.md`), so this is that.
//!
//! **The authority does not widen.** Every function below is the *same* host import
//! a wasm agent calls, gated the same way by the same code in
//! `kernel/src/agent/wasm_rt.rs`: `host_fs_write` is still confined to the agent's
//! home unless its manifest grants a wider scope, `host_http` still refuses without
//! a `net` capability, and the UI imports still refuse without a bound surface.
//! What changes is only that JavaScript can reach them.
//!
//! # The ABI, in one paragraph
//!
//! Core wasm passes numbers, so each host import takes `(ptr, len)` pairs into the
//! plugin's own linear memory and structured data is JSON — the convention upstream
//! recommends and the kernel already uses. A JS string therefore becomes bytes in a
//! scratch buffer before the call, and a result is read back out of one. Return
//! codes are negative for failure and the meanings are per-import: `-3` generally
//! means "not bound" (no agent identity or no surface), `-2` "refused or out of
//! bounds", `-1` "missing or bad arguments".
//!
//! **A refusal becomes a JS exception, never an empty success.** A script that asks
//! for something it has no capability for has to be able to tell that apart from a
//! key that genuinely holds no value.

use javy_plugin_api::javy::quickjs::prelude::{Func, Opt};
use javy_plugin_api::javy::quickjs::{Ctx, Exception, IntoJs, Object, Result as JsResult, Value};
use javy_plugin_api::javy::Runtime;
use javy_plugin_api::{import_namespace, Config};

/// Our namespace. A module built against this plugin imports from it, which is
/// exactly how the kernel recognises a JS-derived `tools.wasm`
/// (`jsmod::links_plugin`). The `_v1` suffix is the compatibility boundary: bump it
/// when the host surface below changes shape, so old artifacts are refused rather
/// than mis-linked.
import_namespace!("chitti_js_v1");

/// Scratch space for marshalling strings across the (ptr, len) boundary. One buffer
/// per direction, sized for a JSON payload; a host import that needs more says so
/// through its return code rather than growing this.
const SCRATCH: usize = 64 * 1024;

// The kernel's host imports, declared exactly as `register_host_imports` defines
// them. Signatures must match by arity *and* type: wasmi resolves imports by both,
// and a mismatch fails at instantiation with a message that does not mention types.
#[link(wasm_import_module = "chitti")]
unsafe extern "C" {
    fn host_storage_set(scope: i32, kp: i32, kl: i32, vp: i32, vl: i32) -> i32;
    fn host_storage_get(scope: i32, kp: i32, kl: i32, out: i32, cap: i32) -> i32;
    fn host_storage_remove(scope: i32, kp: i32, kl: i32) -> i32;
    fn host_storage_list(scope: i32, out: i32, cap: i32) -> i32;
    fn host_fs_read(pp: i32, pl: i32, out: i32, cap: i32) -> i32;
    fn host_fs_write(pp: i32, pl: i32, dp: i32, dl: i32) -> i32;
    fn host_fs_list(pp: i32, pl: i32, out: i32, cap: i32) -> i32;
    fn host_fs_exists(pp: i32, pl: i32) -> i32;
    fn host_ui_draw(op: i32, ol: i32) -> i32;
    fn host_hud_set(tp: i32, tl: i32) -> i32;
    fn host_log(mp: i32, ml: i32);
    fn host_notify(sev: i32, tp: i32, tl: i32, bp: i32, bl: i32) -> i32;
    fn host_now_ms() -> i64;
    fn host_now_unix() -> i64;
    fn host_sha1(sp: i32, sl: i32, out: i32) -> i32;
    fn host_home(out: i32, cap: i32) -> i32;
    fn host_user_home(out: i32, cap: i32) -> i32;
    fn host_http(rp: i32, rl: i32, out: i32, cap: i32) -> i64;
}

/// A leaked scratch buffer, so its address is stable for the life of the instance.
/// Leaked rather than stack-allocated because a host import writes into it while
/// the guest is suspended inside the call.
fn scratch() -> &'static mut [u8] {
    use std::sync::OnceLock;
    static ADDR: OnceLock<usize> = OnceLock::new();
    let addr = *ADDR.get_or_init(|| {
        let b = vec![0u8; SCRATCH].leak();
        b.as_ptr() as usize
    });
    // SAFETY: the allocation is leaked and never freed, and the runtime is
    // single-threaded, so this is the only live reference at any moment.
    unsafe { core::slice::from_raw_parts_mut(addr as *mut u8, SCRATCH) }
}

/// Copy `s` into a second scratch region and return `(ptr, len)` for a host call.
fn stage(s: &str) -> (i32, i32) {
    use std::sync::OnceLock;
    static ADDR: OnceLock<usize> = OnceLock::new();
    let addr = *ADDR.get_or_init(|| {
        let b = vec![0u8; SCRATCH].leak();
        b.as_ptr() as usize
    });
    let n = s.len().min(SCRATCH);
    // SAFETY: as `scratch` — leaked, single-threaded, bounded by `n`.
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), addr as *mut u8, n);
    }
    (addr as i32, n as i32)
}

/// Stage two strings for calls that take a key and a value.
fn stage2(a: &str, b: &str) -> ((i32, i32), (i32, i32)) {
    let (ap, al) = stage(a);
    // The second string goes into the read buffer, which is free at call time for
    // any import that only writes it afterwards. Every two-string import here
    // (storage_set, fs_write) reads both and writes neither.
    let out = scratch();
    let n = b.len().min(out.len());
    out[..n].copy_from_slice(&b.as_bytes()[..n]);
    ((ap, al), (out.as_ptr() as i32, n as i32))
}

/// Read `n` bytes back out of the scratch buffer as a string.
fn taken(n: i32) -> String {
    if n <= 0 {
        return String::new();
    }
    let b = scratch();
    let n = (n as usize).min(b.len());
    String::from_utf8_lossy(&b[..n]).into_owned()
}

/// A value that is either a string or JS `null`.
///
/// Exists to keep a lifetime out of the closure signatures below: writing
/// `|ctx: Ctx<'_>| -> JsResult<Value<'_>>` gives the argument and the return two
/// *independent* anonymous lifetimes, which will not compile, and a closure cannot
/// introduce a named one. Implementing `IntoJs` on an owned type moves that problem
/// to where the context is already in hand.
///
/// `null` and not `undefined` deliberately — see the note at `storageGet`.
enum MaybeStr {
    Null,
    Str(String),
}

impl<'js> IntoJs<'js> for MaybeStr {
    fn into_js(self, ctx: &Ctx<'js>) -> JsResult<Value<'js>> {
        match self {
            MaybeStr::Null => Ok(Value::new_null(ctx.clone())),
            MaybeStr::Str(s) => s.into_js(ctx),
        }
    }
}

/// Storage scope: 0 = session, 1 = durable. Mirrors `agent::storage::Scope`.
fn scope_of(durable: bool) -> i32 {
    if durable {
        1
    } else {
        0
    }
}

/// Turn a negative host return code into a **thrown JS exception**.
///
/// Thrown rather than returned as a value so a refusal cannot be mistaken for an
/// empty success: a script that asks for something it has no capability for must be
/// able to tell that apart from a key that genuinely holds no value.
fn refuse(ctx: &Ctx<'_>, what: &str, code: i32) -> javy_plugin_api::javy::quickjs::Error {
    let why = match code {
        -1 => "bad arguments, or nothing there",
        -2 => "refused: out of bounds, or outside this agent's scope",
        -3 => "refused: this agent has no such capability bound",
        -4 => "rate limited",
        _ => "refused",
    };
    Exception::throw_message(ctx, &format!("{what}: {why} (code {code})"))
}

fn config() -> Config {
    // `javy_stream_io` is what gives the script `Javy.IO.readSync/writeSync`, which
    // is how arguments and results travel; `text_encoding` gives it TextEncoder /
    // TextDecoder, which every tool needs to touch that JSON.
    let mut config = Config::default();
    config
        .text_encoding(true)
        .javy_stream_io(true)
        .simd_json_builtins(true);
    config
}

fn modify_runtime(runtime: Runtime) -> Runtime {
    runtime.context().with(|ctx| {
        let g = ctx.globals();
        let chitti = Object::new(ctx.clone()).unwrap();

        // --- storage: the agent's own key/value space -----------------------
        chitti
            .set(
                "storageGet",
                Func::from(|ctx: Ctx<'_>, durable: bool, key: String| -> JsResult<MaybeStr> {
                    let (kp, kl) = stage(&key);
                    let out = scratch();
                    let n = unsafe {
                        host_storage_get(scope_of(durable), kp, kl, out.as_ptr() as i32, out.len() as i32)
                    };
                    match n {
                        // `null`, like `localStorage.getItem` — and deliberately not
                        // `undefined`, which `JSON.stringify` drops from an object, so
                        // a caller echoing the value back would lose the distinction
                        // between "no value" and "no such field".
                        -1 => Ok(MaybeStr::Null),
                        n if n < 0 => Err(refuse(&ctx, "storageGet", n)),
                        n => Ok(MaybeStr::Str(taken(n))),
                    }
                }),
            )
            .unwrap();
        chitti
            .set(
                "storageSet",
                Func::from(|ctx: Ctx<'_>, durable: bool, key: String, value: String| -> JsResult<()> {
                    let ((kp, kl), (vp, vl)) = stage2(&key, &value);
                    let r = unsafe { host_storage_set(scope_of(durable), kp, kl, vp, vl) };
                    if r < 0 {
                        return Err(refuse(&ctx, "storageSet", r));
                    }
                    Ok(())
                }),
            )
            .unwrap();
        chitti
            .set(
                "storageRemove",
                Func::from(|ctx: Ctx<'_>, durable: bool, key: String| -> JsResult<bool> {
                    let (kp, kl) = stage(&key);
                    let r = unsafe { host_storage_remove(scope_of(durable), kp, kl) };
                    if r < 0 {
                        return Err(refuse(&ctx, "storageRemove", r));
                    }
                    Ok(r == 1)
                }),
            )
            .unwrap();
        chitti
            .set(
                "storageList",
                Func::from(|ctx: Ctx<'_>, durable: bool| -> JsResult<String> {
                    let out = scratch();
                    let n = unsafe {
                        host_storage_list(scope_of(durable), out.as_ptr() as i32, out.len() as i32)
                    };
                    if n < 0 {
                        return Err(refuse(&ctx, "storageList", n));
                    }
                    Ok(taken(n))
                }),
            )
            .unwrap();

        // --- filesystem: home-scoped unless the manifest widens it ----------
        chitti
            .set(
                "fsRead",
                Func::from(|ctx: Ctx<'_>, path: String| -> JsResult<MaybeStr> {
                    let (pp, pl) = stage(&path);
                    let out = scratch();
                    let n = unsafe { host_fs_read(pp, pl, out.as_ptr() as i32, out.len() as i32) };
                    match n {
                        // Absent, as opposed to refused — `null` for the same reason
                        // as `storageGet`.
                        -2 => Ok(MaybeStr::Null),
                        n if n < 0 => Err(refuse(&ctx, "fsRead", n)),
                        n => Ok(MaybeStr::Str(taken(n))),
                    }
                }),
            )
            .unwrap();
        chitti
            .set(
                "fsWrite",
                Func::from(|ctx: Ctx<'_>, path: String, data: String| -> JsResult<()> {
                    let ((pp, pl), (dp, dl)) = stage2(&path, &data);
                    let r = unsafe { host_fs_write(pp, pl, dp, dl) };
                    if r < 0 {
                        return Err(refuse(&ctx, "fsWrite", r));
                    }
                    Ok(())
                }),
            )
            .unwrap();
        chitti
            .set(
                "fsList",
                Func::from(|ctx: Ctx<'_>, path: String| -> JsResult<String> {
                    let (pp, pl) = stage(&path);
                    let out = scratch();
                    let n = unsafe { host_fs_list(pp, pl, out.as_ptr() as i32, out.len() as i32) };
                    if n < 0 {
                        return Err(refuse(&ctx, "fsList", n));
                    }
                    Ok(taken(n))
                }),
            )
            .unwrap();
        chitti
            .set(
                "fsExists",
                Func::from(|path: String| -> bool {
                    let (pp, pl) = stage(&path);
                    unsafe { host_fs_exists(pp, pl) == 1 }
                }),
            )
            .unwrap();

        // --- the agent's own paths ------------------------------------------
        chitti
            .set(
                "home",
                Func::from(|ctx: Ctx<'_>| -> JsResult<String> {
                    let out = scratch();
                    let n = unsafe { host_home(out.as_ptr() as i32, out.len() as i32) };
                    if n < 0 {
                        return Err(refuse(&ctx, "home", n));
                    }
                    Ok(taken(n))
                }),
            )
            .unwrap();
        chitti
            .set(
                "userHome",
                Func::from(|| -> String {
                    let out = scratch();
                    let n = unsafe { host_user_home(out.as_ptr() as i32, out.len() as i32) };
                    taken(n)
                }),
            )
            .unwrap();

        // --- UI, for an agent that owns a surface ---------------------------
        chitti
            .set(
                "uiDraw",
                Func::from(|ctx: Ctx<'_>, ops: String| -> JsResult<()> {
                    let (p, l) = stage(&ops);
                    let r = unsafe { host_ui_draw(p, l) };
                    if r < 0 {
                        return Err(refuse(&ctx, "uiDraw", r));
                    }
                    Ok(())
                }),
            )
            .unwrap();
        chitti
            .set(
                "hud",
                Func::from(|ctx: Ctx<'_>, text: String| -> JsResult<()> {
                    let (p, l) = stage(&text);
                    let r = unsafe { host_hud_set(p, l) };
                    if r < 0 {
                        return Err(refuse(&ctx, "hud", r));
                    }
                    Ok(())
                }),
            )
            .unwrap();

        // --- network, only with a `net` capability --------------------------
        chitti
            .set(
                "http",
                Func::from(|ctx: Ctx<'_>, request: String| -> JsResult<String> {
                    let (p, l) = stage(&request);
                    let out = scratch();
                    let packed = unsafe { host_http(p, l, out.as_ptr() as i32, out.len() as i32) };
                    if packed < 0 {
                        return Err(Exception::throw_message(
                            &ctx,
                            "http: refused -- this agent's manifest declares no `net` capability",
                        ));
                    }
                    // (status << 32) | len, as the host packs it.
                    let status = (packed >> 32) as i32;
                    let len = (packed & 0xffff_ffff) as i32;
                    let body = taken(len);
                    Ok(format!("{{\"status\":{status},\"body\":{}}}", json_string(&body)))
                }),
            )
            .unwrap();

        // --- notifications ---------------------------------------------------
        //
        // `Chitti.notify(severity, title, body?)` — tell the human something that
        // outlives this call. Write-only by design: there is no `notifyList`, so a
        // notification an agent posts cannot be read back, which removes the
        // laundering channel for zero policy. The `source` is stamped by the host
        // from this agent's binding and cannot be set from here.
        //
        // `severity` is "info" | "ok" | "warn" | "error". `action` is deliberately
        // not reachable: it means "a human decision is waiting", which only the
        // kernel's own unattended-approval path is entitled to claim.
        chitti
            .set(
                "notify",
                Func::from(
                    |ctx: Ctx<'_>, severity: String, title: String, body: Opt<String>| -> JsResult<()> {
                        let sev = match severity.trim().to_ascii_lowercase().as_str() {
                            "ok" | "success" | "done" => 1,
                            "warn" | "warning" => 2,
                            "error" | "err" | "fail" => 3,
                            _ => 0,
                        };
                        let body = body.0.unwrap_or_default();
                        let ((tp, tl), (bp, bl)) = stage2(&title, &body);
                        let r = unsafe { host_notify(sev, tp, tl, bp, bl) };
                        if r < 0 {
                            return Err(refuse(&ctx, "notify", r));
                        }
                        Ok(())
                    },
                ),
            )
            .unwrap();

        // --- diagnostics and time ------------------------------------------
        chitti
            .set(
                "log",
                Func::from(|msg: String| {
                    let (p, l) = stage(&msg);
                    unsafe { host_log(p, l) };
                }),
            )
            .unwrap();
        // Both clocks are the host's. A tool below the determinism boundary should
        // not normally need them; they are here because a log line with no
        // timestamp is hard to correlate.
        chitti.set("nowMs", Func::from(|| -> f64 { (unsafe { host_now_ms() }) as f64 })).unwrap();
        chitti.set("nowUnix", Func::from(|| -> f64 { (unsafe { host_now_unix() }) as f64 })).unwrap();
        chitti
            .set(
                "sha1",
                Func::from(|data: String| -> String {
                    let (p, l) = stage(&data);
                    let out = scratch();
                    let r = unsafe { host_sha1(p, l, out.as_ptr() as i32) };
                    if r < 0 {
                        return String::new();
                    }
                    let b = scratch();
                    let mut hex = String::with_capacity(40);
                    for byte in &b[..20] {
                        hex.push_str(&format!("{byte:02x}"));
                    }
                    hex
                }),
            )
            .unwrap();

        g.set("Chitti", chitti).unwrap();
    });
    runtime
}

/// Minimal JSON string escaping, for wrapping an HTTP body into a JSON reply.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[export_name = "initialize-runtime"]
pub extern "C" fn initialize_runtime() {
    javy_plugin_api::initialize_runtime(config, modify_runtime).unwrap();
}
