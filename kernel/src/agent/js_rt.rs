//! **JavaScript runtime** — QuickJS in wasm, for compiling and running JS on the
//! machine.
//!
//! # What this is for
//!
//! Writing an agent's deterministic tool logic should not require a host
//! toolchain. This module holds the QuickJS engine (a [Javy] plugin, in the kernel
//! image) and drives its two jobs:
//!
//! * **compile** — `compile-src` turns JavaScript source into QuickJS bytecode,
//!   which [`crate::agent::jsmod::emit`] then wraps into an ordinary
//!   `assets/tools.wasm`. That is what makes `/agents build` possible with no
//!   compiler on the machine but this one.
//! * **run** — a module emitted that way imports `{cabi_realloc, invoke, memory}`
//!   from the plugin, so running it means instantiating the plugin *and* the
//!   module in one store and linking the first into the second. This is the
//!   kernel's only multi-module link.
//!
//! Arguments and results travel as JSON over fd 0 / fd 1 (the WASI shim in
//! [`crate::agent::wasm_rt`]), because that is the only channel Javy gives a
//! guest: its exported functions take no parameters and their return values are
//! dropped.
//!
//! # Three protocol facts, each of which fails silently if got wrong
//!
//! 1. **`initialize-runtime` reads a config JSON off fd 0.** It must be fed `{}`
//!    and fd 0 rewound before the tool's arguments are presented, or it consumes
//!    them and fails with `unknown field 'q', expected one of 'javy-stream-io',…`
//!    — which reads like a bad argument rather than a protocol mistake.
//! 2. **`compile-src` returns a 3-word area** `[discriminant, ptr, len]`
//!    (`result<list<u8>, string>` in its WIT). Reading it as `(ptr, len)` yields a
//!    plausible pointer and QuickJS then rejects the "bytecode" with
//!    `invalid version (0 expected=26)`.
//! 3. **Bytecode is version-tied to the plugin.** A module built against another
//!    plugin fails the same way, which is why `jsmod` stamps the plugin identity
//!    into the artifact and the runtime checks it before blaming the script.
//!
//! # Cost
//!
//! Measured under this kernel's interpreter: `compile-src` is ~1 ms and ~1 Mfuel
//! for a small tool; a call is ~2 ms and ~3 Mfuel. **That 3 Mfuel is the floor
//! before the script does anything**, which is why the budgets here are far above
//! `wasm_rt::DEFAULT_FUEL` (5 M) and `package_ui`'s per-call 2 M — those would trap
//! on the engine's own start-up.
//!
//! [Javy]: https://github.com/bytecodealliance/javy

use crate::agent::jsmod;
use crate::agent::wasm_rt::{self, HostBindings, HostState, Limits};
use alloc::string::String;
use alloc::vec::Vec;
use wasmi::{Instance, Module, Store};

/// The QuickJS engine, in the kernel image.
///
/// Kernel infrastructure rather than an agent asset, for the same reason as the
/// PDF rasterizer: it is 1.3 MiB shared by every JS agent, and an agent asset is
/// written into the store at every boot and read back on every open.
#[cfg(not(feature = "server"))]
pub static PLUGIN: &[u8] = include_bytes!("../../../assets/wasm/javy-plugin.wasm");
#[cfg(feature = "server")]
pub static PLUGIN: &[u8] = &[];

/// Fuel for compiling one script. Compilation is ~1 Mfuel for a small tool; this
/// leaves room for a large one while still bounding a pathological input.
pub const JS_COMPILE_FUEL: u64 = 2_000_000_000;

/// Fuel for one tool call. The engine's own start-up is ~3 Mfuel before the
/// script runs, so this is not a tight budget — it is a runaway bound.
pub const JS_CALL_FUEL: u64 = 400_000_000;

/// Floor on per-call fuel for a JS guest, whatever a manifest says.
///
/// The engine's own start-up is ~3-4 Mfuel before the script runs, and existing
/// package manifests declare 2 M — a number chosen for a hand-written Rust tool. A
/// manifest like that would make every JS call trap during start-up, which looks
/// like a broken tool rather than a budget set for a different language. So a
/// manifest can raise the budget but not lower it past the point of working.
pub const JS_MIN_CALL_FUEL: u64 = 50_000_000;

/// Linear-memory ceiling for a JS guest, in 64 KiB pages (16 MiB). QuickJS plus a
/// compiled script sits around 3 MiB; this leaves room for the data a tool builds.
pub const JS_MEM_PAGES: u32 = 256;

/// Function-table ceiling. The plugin's own table is well under this; it exists so
/// a rebuilt plugin does not silently need a bump (the PDF renderer's 693-entry
/// table is what taught us the default 256 is not always enough).
pub const JS_TABLE_ELEMS: u32 = 4096;

/// Cap on script size, so `/agents build` refuses a runaway file rather than
/// feeding megabytes of source into the compiler.
pub const MAX_SCRIPT_BYTES: usize = 512 * 1024;

/// A live QuickJS engine, optionally with a JS-derived module linked into it.
pub struct JsSession {
    store: Store<HostState>,
    plugin: Instance,
    module: Option<Instance>,
}

impl JsSession {
    /// Start the engine on its own — enough to compile.
    pub fn new(limits: Limits, bind: HostBindings) -> Result<Self, &'static str> {
        Self::build(None, limits, bind)
    }

    /// Start the engine with a JS-derived `tools.wasm` linked against it.
    pub fn with_module(
        module_wasm: &[u8],
        limits: Limits,
        bind: HostBindings,
    ) -> Result<Self, &'static str> {
        Self::build(Some(module_wasm), limits, bind)
    }

    fn build(
        module_wasm: Option<&[u8]>,
        limits: Limits,
        bind: HostBindings,
    ) -> Result<Self, &'static str> {
        if PLUGIN.is_empty() {
            return Err("no javascript engine in this build");
        }
        let namespace = jsmod::import_namespace(PLUGIN).ok_or("plugin has no import namespace")?;
        let (mut store, mut linker) = wasm_rt::js_store(limits, bind)?;
        let engine = store.engine().clone();

        let plugin_module = Module::new(&engine, PLUGIN).map_err(|_| "plugin failed to validate")?;
        let plugin = linker
            .instantiate_and_start(&mut store, &plugin_module)
            .map_err(|_| "plugin failed to instantiate")?;

        // Protocol fact 1: this reads a config JSON off fd 0, so give it `{}` and
        // leave fd 0 rewound for whatever the caller presents next.
        store.data_mut().fds().set_stdin(b"{}");
        plugin
            .get_typed_func::<(), ()>(&store, "initialize-runtime")
            .map_err(|_| "plugin lacks initialize-runtime")?
            .call(&mut store, ())
            .map_err(|_| "plugin initialize-runtime failed")?;
        store.data_mut().fds().set_stdin(b"");

        let module = match module_wasm {
            None => None,
            Some(wasm) => {
                // The module imports the plugin's exports under its namespace, so
                // define them before instantiating. First multi-module link in the
                // kernel; `Memory` crosses too, which is what lets the guest's
                // pointers and the plugin's agree.
                for field in ["cabi_realloc", "invoke", "memory"] {
                    let ext = plugin
                        .get_export(&store, field)
                        .ok_or("plugin is missing an export the module imports")?;
                    linker
                        .define(namespace, field, ext)
                        .map_err(|_| "failed to link the plugin into the module")?;
                }
                let m = Module::new(&engine, wasm).map_err(|_| "module failed to validate")?;
                Some(linker.instantiate_and_start(&mut store, &m).map_err(|e| {
                    crate::ktrace::log_fmt(format_args!("js: module instantiate failed: {e}"));
                    "module failed to instantiate (built for another plugin?)"
                })?)
            }
        };
        Ok(JsSession { store, plugin, module })
    }

    /// The plugin's import namespace — what a JS-derived module imports from, and
    /// what [`jsmod::links_plugin`] matches to recognise one.
    pub fn namespace() -> Option<&'static str> {
        jsmod::import_namespace(PLUGIN)
    }

    /// An identity for the running plugin, stamped into modules built against it
    /// so a stale artifact is detectable. The namespace is version-suffixed by
    /// Javy itself (`…-v4`), which is exactly the compatibility boundary that
    /// matters, plus the size to distinguish two builds of the same generation.
    pub fn plugin_stamp() -> String {
        alloc::format!(
            "{}@{}",
            Self::namespace().unwrap_or("unknown"),
            PLUGIN.len()
        )
    }

    pub fn set_fuel(&mut self, fuel: u64) -> Result<(), &'static str> {
        self.store.set_fuel(fuel).map_err(|_| "set_fuel failed")
    }

    /// Compile JavaScript to QuickJS bytecode.
    pub fn compile(&mut self, src: &str) -> Result<Vec<u8>, &'static str> {
        if src.is_empty() {
            return Err("empty script");
        }
        if src.len() > MAX_SCRIPT_BYTES {
            return Err("script too large");
        }
        let realloc = self
            .plugin
            .get_typed_func::<(i32, i32, i32, i32), i32>(&self.store, "cabi_realloc")
            .map_err(|_| "plugin lacks cabi_realloc")?;
        let compile = self
            .plugin
            .get_typed_func::<(i32, i32), i32>(&self.store, "compile-src")
            .map_err(|_| "plugin lacks compile-src")?;
        let mem = self
            .plugin
            .get_memory(&self.store, "memory")
            .ok_or("plugin lacks memory")?;

        let len = src.len() as i32;
        let ptr = realloc
            .call(&mut self.store, (0, 0, 1, len))
            .map_err(|_| "plugin allocation failed")?;
        mem.write(&mut self.store, ptr as usize, src.as_bytes())
            .map_err(|_| "writing the script into the engine failed")?;
        let ret = compile
            .call(&mut self.store, (ptr, len))
            .map_err(|_| "the script did not compile")?;
        if ret <= 0 {
            return Err("the script did not compile");
        }
        // Protocol fact 2: three words, and the first is a discriminant.
        let mut words = [0u8; 12];
        mem.read(&self.store, ret as usize, &mut words)
            .map_err(|_| "compile result out of bounds")?;
        let w = |i: usize| u32::from_le_bytes([words[i * 4], words[i * 4 + 1], words[i * 4 + 2], words[i * 4 + 3]]);
        if w(0) != 0 {
            // The error arm carries a string; the message is not worth plumbing
            // through when the actionable part is "this script is not valid JS".
            return Err("the script has a syntax error");
        }
        let (bc_ptr, bc_len) = (w(1) as usize, w(2) as usize);
        let data = mem.data(&self.store);
        let end = bc_ptr.checked_add(bc_len).ok_or("compile result overflow")?;
        if bc_len == 0 || end > data.len() {
            return Err("compile result out of bounds");
        }
        Ok(data[bc_ptr..end].to_vec())
    }

    /// Call one export of the linked module with `args_json`, returning what it
    /// wrote to stdout.
    pub fn call(&mut self, export: &str, args_json: &str) -> Result<String, &'static str> {
        let module = self.module.ok_or("no module linked")?;
        let f = module
            .get_typed_func::<(), ()>(&self.store, export)
            .map_err(|_| "no such tool in this module")?;
        self.store.data_mut().fds().set_stdin(args_json.as_bytes());
        let _ = self.store.data_mut().fds().take_stdout();
        f.call(&mut self.store, ()).map_err(wasm_rt::map_trap_pub)?;
        let out = self.store.data_mut().fds().take_stdout();
        String::from_utf8(out).map_err(|_| "tool output is not utf-8")
    }

    /// The guest's own last words on stderr — the reason behind a trap.
    pub fn last_stderr(&self) -> Option<&str> {
        self.store.data().fds_ref().last_stderr()
    }

    /// Tool names the linked module exports.
    pub fn exports(&self) -> Vec<String> {
        let Some(m) = self.module else { return Vec::new() };
        let names: Vec<String> = m.exports(&self.store).map(|e| String::from(e.name())).collect();
        names
            .into_iter()
            .filter(|n| m.get_typed_func::<(), ()>(&self.store, n).is_ok())
            .collect()
    }
}

/// Compile `src` and wrap it in a `tools.wasm` exporting `tools` — the whole
/// on-machine build, in one call.
///
/// Returns the module bytes. The caller writes them wherever the package keeps
/// its artifact; nothing here touches the store.
pub fn build_module(src: &str, tools: &[&str]) -> Result<Vec<u8>, &'static str> {
    let namespace = JsSession::namespace().ok_or("plugin has no import namespace")?;
    let stamp = JsSession::plugin_stamp();
    let limits = Limits::default()
        .with_fuel(JS_COMPILE_FUEL)
        .with_pages(JS_MEM_PAGES)
        .with_table_elems(JS_TABLE_ELEMS);
    let mut js = JsSession::new(limits, HostBindings::default())?;
    let bytecode = js.compile(src)?;
    jsmod::emit(namespace, &stamp, &bytecode, tools).map_err(|e| match e {
        jsmod::EmitError::NoExports => "no tools declared for this module",
        jsmod::EmitError::TooManyExports => "too many tools in one module",
        jsmod::EmitError::BadName => "a tool name is not a valid JS identifier",
        jsmod::EmitError::EmptyBytecode => "the script compiled to nothing",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool script in the shape an agent would ship: read args JSON from fd 0,
    /// write a result JSON to fd 1, one ESM export per tool.
    const SCRIPT: &str = r#"
        function readIn() {
          const chunks = []; const buf = new Uint8Array(1024); let n;
          while ((n = Javy.IO.readSync(0, buf)) > 0) chunks.push(buf.slice(0, n));
          let total = 0; for (const c of chunks) total += c.length;
          const all = new Uint8Array(total); let o = 0;
          for (const c of chunks) { all.set(c, o); o += c.length; }
          return JSON.parse(new TextDecoder().decode(all) || "{}");
        }
        function writeOut(v) { Javy.IO.writeSync(1, new TextEncoder().encode(JSON.stringify(v))); }
        export function demo_echo() { writeOut({ tool: "demo_echo", got: readIn() }); }
        export function demo_sum() { const a = readIn(); writeOut({ sum: (a.xs || []).reduce((s, x) => s + x, 0) }); }
    "#;

    fn limits() -> Limits {
        Limits::default()
            .with_fuel(JS_COMPILE_FUEL)
            .with_pages(JS_MEM_PAGES)
            .with_table_elems(JS_TABLE_ELEMS)
    }

    /// **The end-to-end proof**: JavaScript compiled on this machine, wrapped in a
    /// real wasm module, linked against the engine, and called — twice, with
    /// different arguments, returning correct JSON both times.
    ///
    /// This is the test that would catch a wrong LEB128 length, a mis-ordered
    /// section, the fd-0 config trap, or the 3-word compile result being misread:
    /// each of those produces a module that *validates* and then misbehaves.
    #[test_case]
    fn javascript_compiles_and_runs_on_this_machine() {
        let wasm = build_module(SCRIPT, &["demo_echo", "demo_sum"]).expect("build");
        assert!(jsmod::links_plugin(&wasm, JsSession::namespace().unwrap()));
        assert_eq!(jsmod::plugin_stamp(&wasm).as_deref(), Some(JsSession::plugin_stamp().as_str()));

        let mut js = JsSession::with_module(&wasm, limits(), HostBindings::default()).expect("link");
        let mut names = js.exports();
        names.sort();
        assert_eq!(names, alloc::vec!["demo_echo", "demo_sum"]);

        let out = js.call("demo_echo", r#"{"q":"hi"}"#).expect("call demo_echo");
        assert!(out.contains(r#""tool":"demo_echo""#), "got {out}");
        assert!(out.contains(r#""q":"hi""#), "arguments must reach the script: {out}");

        let out = js.call("demo_sum", r#"{"xs":[10,20,30]}"#).expect("call demo_sum");
        assert!(out.contains(r#""sum":60"#), "the script must compute: {out}");
    }

    /// Arguments are a fresh stream per call — the fd-0 rewind is real, and a
    /// second call does not see the first call's input.
    #[test_case]
    fn each_call_gets_its_own_arguments() {
        let wasm = build_module(SCRIPT, &["demo_echo"]).expect("build");
        let mut js = JsSession::with_module(&wasm, limits(), HostBindings::default()).expect("link");
        let a = js.call("demo_echo", r#"{"n":1}"#).expect("first");
        let b = js.call("demo_echo", r#"{"n":2}"#).expect("second");
        assert!(a.contains(r#""n":1"#), "{a}");
        assert!(b.contains(r#""n":2"#) && !b.contains(r#""n":1"#), "{b}");
    }

    /// A syntax error is reported as an error — **and costs the engine instance.**
    ///
    /// The plugin does not return the error arm of `compile-src` for bad input: it
    /// panics inside its own WASI shim, which under `panic = "abort"` is a trap.
    /// So `compile` correctly yields `Err`, but the session is poisoned and every
    /// later call on it fails too. That is why [`build_module`] constructs a fresh
    /// session per build rather than keeping one warm — a compiler that dies on
    /// bad input is fine, a *cached* compiler that dies on bad input would reject
    /// every subsequent good script.
    #[test_case]
    fn a_syntax_error_is_reported_and_burns_the_session() {
        let mut js = JsSession::new(limits(), HostBindings::default()).expect("engine");
        let e = js.compile("export function t( {{{ bad").unwrap_err();
        assert!(e.contains("compile") || e.contains("syntax"), "got {e}");
        // Pinned, not wished away: the same session cannot compile again.
        assert!(
            js.compile("export function t() {}").is_err(),
            "a trapped instance stays trapped -- build_module must start fresh"
        );
        // ...and starting fresh works, which is the path users actually take.
        assert!(build_module("export function t() {}", &["t"]).is_ok());
    }

    /// **JavaScript reaches the kernel's gated host surface**, through the very
    /// same `chitti.host_*` imports a hand-written Rust tool module calls.
    ///
    /// This is what our own plugin buys over the stock one, which gives a script
    /// stdio and nothing else. The authority does not widen: these are the same
    /// imports, gated by the same code, so what a JS tool may do is exactly what a
    /// wasm tool with the same manifest may do.
    #[test_case]
    fn javascript_reaches_the_gated_host_surface() {
        const SRC: &str = r#"
            function reply(v) { Javy.IO.writeSync(1, new TextEncoder().encode(JSON.stringify(v))); }
            export function probe() {
              Chitti.log("hello from a js tool");
              Chitti.storageSet(false, "k", "v1");
              const got = Chitti.storageGet(false, "k");
              const missing = Chitti.storageGet(false, "nope");
              reply({ got, missing, home: Chitti.home(), sha1: Chitti.sha1("abc") });
            }
        "#;
        let wasm = build_module(SRC, &["probe"]).expect("build");
        // A real agent identity: storage is scoped to it, and `host_home` needs it.
        let bind = HostBindings { agent_id: 4242, task: 0, surface: 0 };
        let mut js = JsSession::with_module(&wasm, limits(), bind).expect("link");
        let out = js.call("probe", "{}").expect("probe");
        assert!(out.contains(r#""got":"v1""#), "storage round-trip failed: {out}");
        // A key that holds nothing is `null`, *not* an exception — the script has to
        // be able to tell "no value" from "not allowed".
        assert!(out.contains(r#""missing":null"#), "a missing key must read as null: {out}");
        assert!(out.contains("/agent/4242"), "the agent's own home should be visible: {out}");
        // SHA-1("abc") — a known digest, so this is the host's hasher and not a stub.
        assert!(
            out.contains("a9993e364706816aba3e25717850c26c9cd0d89d"),
            "host_sha1 did not produce the known digest: {out}"
        );
        crate::agent::storage::clear_session(4242);
    }

    /// A capability the agent does not hold becomes a **thrown JS exception**, not an
    /// empty success. Drawing needs a bound surface; this identity has none.
    #[test_case]
    fn a_refused_capability_throws_into_the_script() {
        const SRC: &str = r#"
            function reply(v) { Javy.IO.writeSync(1, new TextEncoder().encode(JSON.stringify(v))); }
            export function probe() {
              try { Chitti.uiDraw("clear 000000"); reply({ drew: true }); }
              catch (e) { reply({ refused: String(e) }); }
            }
        "#;
        let wasm = build_module(SRC, &["probe"]).expect("build");
        let mut js = JsSession::with_module(&wasm, limits(), HostBindings::default()).expect("link");
        let out = js.call("probe", "{}").expect("probe");
        assert!(out.contains("refused"), "an unbound surface must refuse: {out}");
        assert!(!out.contains(r#""drew":true"#), "and must not look like it worked: {out}");
    }

    #[test_case]
    fn scripts_are_bounded() {
        let mut js = JsSession::new(limits(), HostBindings::default()).expect("engine");
        assert!(js.compile("").is_err());
        let big = alloc::string::String::from_utf8(alloc::vec![b'x'; MAX_SCRIPT_BYTES + 1]).unwrap();
        assert_eq!(js.compile(&big).unwrap_err(), "script too large");
    }

    /// A module whose declared tool has no matching JS export builds fine — the
    /// mismatch shows up as a failed call, so the message has to point at the tool.
    #[test_case]
    fn calling_a_tool_the_script_does_not_export_fails_clearly() {
        let wasm = build_module(SCRIPT, &["demo_echo", "not_in_script"]).expect("build");
        let mut js = JsSession::with_module(&wasm, limits(), HostBindings::default()).expect("link");
        assert!(js.call("demo_echo", "{}").is_ok());
        assert!(js.call("not_in_script", "{}").is_err(), "a missing JS export must fail");
        assert!(js.call("never_declared", "{}").is_err(), "and so must a missing wasm export");
    }
}
