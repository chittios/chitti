//! `doombench` — price the wasm interpreter on a real game loop, and check that
//! both builds render the same thing.
//!
//! Usage:
//! ```text
//! cargo run --release -- <freedoom1.wad> [frames]
//! ```
//!
//! Reports, for the native build and for the same sources under wasmi:
//! frames/sec, ms/frame, and — first — whether the two **agree** on the frame
//! content and the game clock. A faster build that renders something else has
//! changed the program, not won; that is the order `/html bench` reports in and
//! the reason it restores the previous engine afterwards.
//!
//! The decision this feeds: whether a Doom port can be an ordinary wasm app
//! package (signed, capability-gated, like every other app here) or has to be a
//! ring-3 tenant (native speed, but it means writing the first chunked stateful
//! tenant). CLAUDE.md's existing figures bracket the interpreter tax between 3x
//! and 67x depending on the workload, which is far too wide to design against.

use std::time::Instant;

// ---------------------------------------------------------------------------
// native side
// ---------------------------------------------------------------------------

extern "C" {
    fn dg_wad_storage() -> *mut u8;
    fn dg_wad_capacity() -> u32;
    fn dg_set_wad(base: *const u8, len: u32);
    fn dg_iwad_path_buf() -> *mut u8;
    fn dg_iwad_path_cap() -> u32;
    fn dg_create();
    fn dg_tick();
    fn dg_frame_hash() -> u64;
    fn dg_gametic() -> i32;
}

/// What one run reports. Both backends produce this, so they are comparable by
/// construction rather than by two hand-written print statements.
struct Run {
    label: &'static str,
    frames: u32,
    secs: f64,
    /// Frame hash after the last frame — the agreement check.
    hash: u64,
    /// Doom's own simulation clock. Two builds that agree here ran the same tics,
    /// which is a much stronger claim than agreeing on one frame's pixels.
    gametic: i32,
}

impl Run {
    fn fps(&self) -> f64 {
        self.frames as f64 / self.secs
    }
    fn ms(&self) -> f64 {
        self.secs * 1000.0 / self.frames as f64
    }
}

fn run_native(wad: &[u8], wad_path: &str, frames: u32) -> Run {
    // SAFETY: single-threaded, and the C side owns no state until `dg_create`.
    // `wad_storage` is a `.bss` array of known capacity, checked before the copy.
    unsafe {
        let cap = dg_wad_capacity() as usize;
        assert!(
            wad.len() <= cap,
            "WAD is {} bytes, staging buffer is {cap}",
            wad.len()
        );
        let dst = dg_wad_storage();
        std::ptr::copy_nonoverlapping(wad.as_ptr(), dst, wad.len());
        dg_set_wad(dst, wad.len() as u32);

        // The path is how Doom picks its game mode; see platform.c.
        let pcap = dg_iwad_path_cap() as usize;
        let p = wad_path.as_bytes();
        assert!(p.len() + 1 <= pcap, "iwad path longer than {pcap}");
        let pbuf = dg_iwad_path_buf();
        std::ptr::copy_nonoverlapping(p.as_ptr(), pbuf, p.len());
        *pbuf.add(p.len()) = 0;

        // `dg_create` renders one frame itself (D_DoomLoop ends with a tick), so
        // it is startup and is deliberately outside the timed region.
        dg_create();

        let t = Instant::now();
        for _ in 0..frames {
            dg_tick();
        }
        let secs = t.elapsed().as_secs_f64();
        Run {
            label: "native",
            frames,
            secs,
            hash: dg_frame_hash(),
            gametic: dg_gametic(),
        }
    }
}

// ---------------------------------------------------------------------------
// wasm side
// ---------------------------------------------------------------------------

/// `None` when the wasm toolchain was absent at build time — `build.rs` then
/// leaves `DOOM_WASM` unset and warns, so the native side still runs.
fn run_wasm(wad: &[u8], wad_path: &str, frames: u32) -> Option<Run> {
    let path = option_env!("DOOM_WASM")?;
    let bytes = std::fs::read(path).ok()?;
    Some(wasm::run(&bytes, wad, wad_path, frames))
}

mod wasm {
    use super::Run;
    use std::time::Instant;
    use wasmi::*;

    /// Host state: the exit status, so a guest `proc_exit` is a fact the harness
    /// can report rather than a process that vanished — plus the one virtual file
    /// the guest is allowed to see.
    #[derive(Default)]
    pub struct Host {
        pub exited: Option<i32>,
        /// The WAD, served read-only through a single synthetic fd.
        ///
        /// This exists **only** so Doom's game-mode identification works the same
        /// way it does natively: `D_FindWADByName` checks the file exists, and
        /// `D_IdentifyIWADByName` reads the game mode out of the *filename*. The
        /// lump data never comes through here — `w_file_memory.c` sets
        /// `wad_file_t::mapped`, so Doom reads lumps straight out of guest memory.
        /// Keeping the two builds on the same identification path is the point;
        /// otherwise the wasm side would be running a differently-configured game
        /// and the ratio would not mean anything.
        pub wad: Vec<u8>,
        pub basename: String,
        /// Offset of the open fd, if any. One file, one reader — Doom opens the
        /// IWAD once during identification.
        pub open: Option<u64>,
    }

    /// The synthetic descriptors. 3 is the preopened directory (wasi-libc will not
    /// resolve a relative *or* absolute path without at least one preopen — it
    /// fails in `__wasilibc_find_relpath` before ever calling `path_open`), and 4
    /// is the WAD.
    const PREOPEN_FD: i32 = 3;
    const WAD_FD: i32 = 4;

    /// The WASI surface Doom actually reaches.
    ///
    /// Deliberately minimal and deliberately *not* a filesystem: the WAD is
    /// already in guest memory (see `w_file_memory.c`), so every path operation
    /// can honestly report "no such file" and Doom falls back to its defaults —
    /// which is the same position the real ChittiOS port is in. Doom tolerates a
    /// missing `default.cfg`; it does not tolerate being lied to about a file it
    /// then tries to parse.
    ///
    /// Arities matter here more than they look. wasmi resolves an import by name
    /// **and `FuncType`**, so a stub with the wrong signature makes the whole
    /// module fail to instantiate with `missing imports` — a message that names
    /// nothing. That exact trap already cost this tree a debugging cycle when the
    /// kernel's five WASI stubs were all declared `() -> i32`.
    fn link(linker: &mut Linker<Host>) -> Result<(), Box<dyn std::error::Error>> {
        const W: &str = "wasi_snapshot_preview1";
        // ENOSYS/EBADF-ish: 8 is WASI `ERRNO_BADF`, 44 is `ERRNO_NOENT`.
        const NOENT: i32 = 44;
        const BADF: i32 = 8;

        // Diagnostics. Doom prints its banner and level load messages; forwarding
        // them is how a failure inside init is diagnosable at all.
        linker.func_wrap(
            W,
            "fd_write",
            |mut caller: Caller<'_, Host>, fd: i32, iovs: i32, n: i32, out: i32| -> i32 {
                let Some(Extern::Memory(mem)) = caller.get_export("memory") else {
                    return BADF;
                };
                let mut total = 0u32;
                let mut buf = Vec::new();
                for i in 0..n.max(0) {
                    let mut hdr = [0u8; 8];
                    if mem
                        .read(&caller, (iovs + i * 8) as usize, &mut hdr)
                        .is_err()
                    {
                        return BADF;
                    }
                    let p = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
                    let l = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
                    let mut b = vec![0u8; l];
                    if mem.read(&caller, p, &mut b).is_err() {
                        return BADF;
                    }
                    buf.extend_from_slice(&b);
                    total += l as u32;
                }
                if fd == 1 || fd == 2 {
                    print!("{}", String::from_utf8_lossy(&buf));
                }
                let _ = mem.write(&mut caller, out as usize, &total.to_le_bytes());
                0
            },
        )?;

        linker.func_wrap(W, "proc_exit", |mut caller: Caller<'_, Host>, code: i32| {
            caller.data_mut().exited = Some(code);
        })?;

        // One preopened directory, "/", so wasi-libc can resolve a path at all.
        // `fd_prestat_get` writes { u8 tag = dir, u32 name_len }; the padding
        // matters (name_len is at offset 4, not 1).
        linker.func_wrap(
            W,
            "fd_prestat_get",
            |mut caller: Caller<'_, Host>, fd: i32, out: i32| -> i32 {
                if fd != PREOPEN_FD {
                    return BADF;
                }
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let mut buf = [0u8; 8];
                    buf[0] = 0; // preopentype: dir
                    buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // len("/")
                    let _ = m.write(&mut caller, out as usize, &buf);
                }
                0
            },
        )?;
        linker.func_wrap(
            W,
            "fd_prestat_dir_name",
            |mut caller: Caller<'_, Host>, fd: i32, path: i32, len: i32| -> i32 {
                if fd != PREOPEN_FD || len < 1 {
                    return BADF;
                }
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let _ = m.write(&mut caller, path as usize, b"/");
                }
                0
            },
        )?;

        // Open: the WAD, matched by basename, and nothing else.
        linker.func_wrap(
            W,
            "path_open",
            |mut caller: Caller<'_, Host>,
             _dirfd: i32,
             _dirflags: i32,
             path: i32,
             path_len: i32,
             _oflags: i32,
             _base: i64,
             _inherit: i64,
             _fdflags: i32,
             out_fd: i32|
             -> i32 {
                let Some(Extern::Memory(m)) = caller.get_export("memory") else {
                    return BADF;
                };
                let mut b = vec![0u8; path_len.max(0) as usize];
                if m.read(&caller, path as usize, &mut b).is_err() {
                    return NOENT;
                }
                let name = String::from_utf8_lossy(&b).to_string();
                let want = caller.data().basename.clone();
                // Basename match: the guest sees a path relative to the preopen,
                // and only the final component is meaningful to us.
                if !name.rsplit('/').next().is_some_and(|n| n == want) {
                    return NOENT;
                }
                caller.data_mut().open = Some(0);
                let _ = m.write(&mut caller, out_fd as usize, &WAD_FD.to_le_bytes());
                0
            },
        )?;
        linker.func_wrap(
            W,
            "fd_read",
            |mut caller: Caller<'_, Host>, fd: i32, iovs: i32, n: i32, out: i32| -> i32 {
                if fd != WAD_FD {
                    return BADF;
                }
                let Some(Extern::Memory(m)) = caller.get_export("memory") else {
                    return BADF;
                };
                let Some(mut off) = caller.data().open else {
                    return BADF;
                };
                let wad = caller.data().wad.clone();
                let mut total = 0u32;
                for i in 0..n.max(0) {
                    let mut hdr = [0u8; 8];
                    if m.read(&caller, (iovs + i * 8) as usize, &mut hdr).is_err() {
                        return BADF;
                    }
                    let p = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
                    let l = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
                    let start = (off as usize).min(wad.len());
                    let end = (start + l).min(wad.len());
                    if end > start {
                        let _ = m.write(&mut caller, p, &wad[start..end]);
                    }
                    let got = (end - start) as u64;
                    off += got;
                    total += got as u32;
                    if got < l as u64 {
                        break; // short read: at EOF
                    }
                }
                caller.data_mut().open = Some(off);
                let _ = m.write(&mut caller, out as usize, &total.to_le_bytes());
                0
            },
        )?;
        linker.func_wrap(
            W,
            "fd_seek",
            |mut caller: Caller<'_, Host>, fd: i32, offset: i64, whence: i32, out: i32| -> i32 {
                if fd != WAD_FD {
                    return BADF;
                }
                let len = caller.data().wad.len() as i64;
                let cur = caller.data().open.unwrap_or(0) as i64;
                // WASI whence: 0 = set, 1 = cur, 2 = end.
                let base = match whence {
                    0 => 0,
                    1 => cur,
                    2 => len,
                    _ => return BADF,
                };
                let next = (base + offset).clamp(0, len) as u64;
                caller.data_mut().open = Some(next);
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let _ = m.write(&mut caller, out as usize, &next.to_le_bytes());
                }
                0
            },
        )?;
        linker.func_wrap(W, "fd_close", |mut caller: Caller<'_, Host>, fd: i32| -> i32 {
            if fd == WAD_FD {
                caller.data_mut().open = None;
                return 0;
            }
            BADF
        })?;
        linker.func_wrap(
            W,
            "fd_fdstat_get",
            |mut caller: Caller<'_, Host>, fd: i32, out: i32| -> i32 {
                // filetype at 0, flags at 2, rights at 8 and 16.
                let ty: u8 = match fd {
                    PREOPEN_FD => 3, // directory
                    WAD_FD => 4,     // regular file
                    1 | 2 => 2,      // character device
                    _ => return BADF,
                };
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let mut buf = [0u8; 24];
                    buf[0] = ty;
                    // Grant every right: this is a harness, and the guest is our
                    // own code. The kernel's equivalent must not do this.
                    buf[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
                    buf[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
                    let _ = m.write(&mut caller, out as usize, &buf);
                }
                0
            },
        )?;
        linker.func_wrap(
            W,
            "fd_fdstat_set_flags",
            |_: Caller<'_, Host>, _: i32, _: i32| -> i32 { 0 },
        )?;
        linker.func_wrap(
            W,
            "path_filestat_get",
            |mut caller: Caller<'_, Host>,
             _dirfd: i32,
             _flags: i32,
             path: i32,
             path_len: i32,
             out: i32|
             -> i32 {
                let Some(Extern::Memory(m)) = caller.get_export("memory") else {
                    return BADF;
                };
                let mut b = vec![0u8; path_len.max(0) as usize];
                if m.read(&caller, path as usize, &mut b).is_err() {
                    return NOENT;
                }
                let name = String::from_utf8_lossy(&b).to_string();
                let want = caller.data().basename.clone();
                if !name.rsplit('/').next().is_some_and(|n| n == want) {
                    return NOENT;
                }
                let size = caller.data().wad.len() as u64;
                // filestat: dev(8) ino(8) filetype(1)+pad(7) nlink(8) size(8) ...
                let mut buf = [0u8; 64];
                buf[16] = 4; // regular file
                buf[24..32].copy_from_slice(&1u64.to_le_bytes());
                buf[32..40].copy_from_slice(&size.to_le_bytes());
                let _ = m.write(&mut caller, out as usize, &buf);
                0
            },
        )?;
        linker.func_wrap(
            W,
            "path_unlink_file",
            |_: Caller<'_, Host>, _: i32, _: i32, _: i32| -> i32 { NOENT },
        )?;
        linker.func_wrap(
            W,
            "path_rename",
            |_: Caller<'_, Host>, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32| -> i32 {
                NOENT
            },
        )?;
        linker.func_wrap(
            W,
            "path_create_directory",
            |_: Caller<'_, Host>, _: i32, _: i32, _: i32| -> i32 { NOENT },
        )?;
        linker.func_wrap(
            W,
            "path_remove_directory",
            |_: Caller<'_, Host>, _: i32, _: i32, _: i32| -> i32 { NOENT },
        )?;
        // Zero args / zero environment. Doom takes its argv from `dg_create`.
        linker.func_wrap(W, "args_sizes_get", |mut caller: Caller<'_, Host>, c: i32, s: i32| -> i32 {
            if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                let _ = m.write(&mut caller, c as usize, &0u32.to_le_bytes());
                let _ = m.write(&mut caller, s as usize, &0u32.to_le_bytes());
            }
            0
        })?;
        linker.func_wrap(W, "args_get", |_: Caller<'_, Host>, _: i32, _: i32| -> i32 { 0 })?;
        linker.func_wrap(
            W,
            "environ_sizes_get",
            |mut caller: Caller<'_, Host>, c: i32, s: i32| -> i32 {
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let _ = m.write(&mut caller, c as usize, &0u32.to_le_bytes());
                    let _ = m.write(&mut caller, s as usize, &0u32.to_le_bytes());
                }
                0
            },
        )?;
        linker.func_wrap(W, "environ_get", |_: Caller<'_, Host>, _: i32, _: i32| -> i32 { 0 })?;
        // A fixed clock: the guest uses its own virtual clock for frame timing
        // (see platform.c), so anything here must not reintroduce nondeterminism.
        linker.func_wrap(
            W,
            "clock_time_get",
            |mut caller: Caller<'_, Host>, _: i32, _: i64, out: i32| -> i32 {
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let _ = m.write(&mut caller, out as usize, &0u64.to_le_bytes());
                }
                0
            },
        )?;
        linker.func_wrap(
            W,
            "random_get",
            |mut caller: Caller<'_, Host>, buf: i32, len: i32| -> i32 {
                // Deterministic on purpose: a benchmark that reseeds differently
                // per run is not comparable to itself, let alone to native.
                if let Some(Extern::Memory(m)) = caller.get_export("memory") {
                    let zeros = vec![0u8; len.max(0) as usize];
                    let _ = m.write(&mut caller, buf as usize, &zeros);
                }
                0
            },
        )?;
        Ok(())
    }

    /// Fuel consumed by one frame, averaged over `frames`.
    ///
    /// A separate pass from the timing one on purpose: metering costs a measured
    /// 3.7%, and folding it into the speed figure makes both numbers harder to
    /// read. This one exists to size the manifest — `package_ui`'s `CALL_FUEL` is
    /// 2,000,000 per export call today, and a guest that runs out mid-frame traps,
    /// which for a persistent-instance app means losing the game rather than
    /// dropping a frame.
    pub fn fuel_per_frame(
        module_bytes: &[u8],
        wad: &[u8],
        wad_path: &str,
        frames: u32,
    ) -> Option<u64> {
        let mut cfg = Config::default();
        cfg.compilation_mode(CompilationMode::Eager);
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg);
        let module = Module::new(&engine, module_bytes).ok()?;
        let basename = wad_path.rsplit('/').next().unwrap_or(wad_path).to_string();
        let mut store = Store::new(
            &engine,
            Host {
                wad: wad.to_vec(),
                basename,
                ..Host::default()
            },
        );
        let mut linker = Linker::new(&engine);
        link(&mut linker).ok()?;
        // Startup alone is far more than a frame; give it plenty and re-arm after.
        store.set_fuel(u64::MAX / 4).ok()?;
        let inst = linker
            .instantiate_and_start(&mut store, &module)
            .ok()?;
        let mem = match inst.get_export(&store, "memory") {
            Some(Extern::Memory(m)) => m,
            _ => return None,
        };
        let call_u32 = |n: &str, s: &mut Store<Host>| -> u32 {
            inst.get_typed_func::<(), u32>(&*s, n)
                .ok()
                .and_then(|f| f.call(&mut *s, ()).ok())
                .unwrap_or(0)
        };
        let dst = call_u32("dg_wad_storage", &mut store) as usize;
        mem.write(&mut store, dst, wad).ok()?;
        let pbuf = call_u32("dg_iwad_path_buf", &mut store) as usize;
        mem.write(&mut store, pbuf, wad_path.as_bytes()).ok()?;
        mem.write(&mut store, pbuf + wad_path.len(), &[0u8]).ok()?;
        inst.get_typed_func::<(u32, u32), ()>(&store, "dg_set_wad")
            .ok()?
            .call(&mut store, (dst as u32, wad.len() as u32))
            .ok()?;
        inst.get_typed_func::<(), ()>(&store, "dg_create")
            .ok()?
            .call(&mut store, ())
            .ok()?;

        let tick = inst.get_typed_func::<(), ()>(&store, "dg_tick").ok()?;
        let before = store.get_fuel().ok()?;
        for _ in 0..frames {
            tick.call(&mut store, ()).ok()?;
        }
        let after = store.get_fuel().ok()?;
        Some(before.saturating_sub(after) / frames.max(1) as u64)
    }

    pub fn run(module_bytes: &[u8], wad: &[u8], wad_path: &str, frames: u32) -> Run {
        // Fuel metering is OFF here, and that is deliberate: the kernel charges
        // fuel, but this harness is measuring the interpreter's *speed*, and
        // metering costs a measured 3.7%. The fuel *cost* of a frame is a separate
        // number worth having — it sizes the manifest budget — but mixing it into
        // the speed figure makes both harder to read.
        let engine = Engine::default();
        let module = Module::new(&engine, module_bytes).expect("wasm module");
        let basename = wad_path.rsplit('/').next().unwrap_or(wad_path).to_string();
        let mut store = Store::new(
            &engine,
            Host { wad: wad.to_vec(), basename, ..Host::default() },
        );
        let mut linker = Linker::new(&engine);
        link(&mut linker).expect("link wasi stubs");
        // `instantiate_and_start` runs the module's own start section, which for a
        // wasi-libc module is what initialises its heap and statics — Doom's zone
        // allocator is unusable without it.
        let inst = linker
            .instantiate_and_start(&mut store, &module)
            .expect("instantiate (missing import or wrong import arity?)");

        let mem = match inst.get_export(&store, "memory") {
            Some(Extern::Memory(m)) => m,
            _ => panic!("module exports no memory"),
        };
        let f_u32 = |n: &str, s: &mut Store<Host>| -> u32 {
            let f = inst
                .get_typed_func::<(), u32>(&*s, n)
                .unwrap_or_else(|_| panic!("export {n}"));
            f.call(&mut *s, ()).expect(n)
        };

        let cap = f_u32("dg_wad_capacity", &mut store) as usize;
        assert!(wad.len() <= cap, "WAD {} > staging {cap}", wad.len());
        let dst = f_u32("dg_wad_storage", &mut store) as usize;
        mem.write(&mut store, dst, wad).expect("stage wad");

        // The IWAD path, for game-mode identification (see platform.c).
        let pcap = f_u32("dg_iwad_path_cap", &mut store) as usize;
        let pbuf = f_u32("dg_iwad_path_buf", &mut store) as usize;
        let pbytes = wad_path.as_bytes();
        assert!(pbytes.len() + 1 <= pcap, "iwad path longer than {pcap}");
        mem.write(&mut store, pbuf, pbytes).expect("stage iwad path");
        mem.write(&mut store, pbuf + pbytes.len(), &[0u8]).expect("nul");

        inst.get_typed_func::<(u32, u32), ()>(&store, "dg_set_wad")
            .expect("dg_set_wad")
            .call(&mut store, (dst as u32, wad.len() as u32))
            .expect("dg_set_wad call");

        // Startup, outside the timed region — same as native.
        inst.get_typed_func::<(), ()>(&store, "dg_create")
            .expect("dg_create")
            .call(&mut store, ())
            .expect("dg_create call");

        let tick = inst
            .get_typed_func::<(), ()>(&store, "dg_tick")
            .expect("dg_tick");
        let t = Instant::now();
        for _ in 0..frames {
            tick.call(&mut store, ()).expect("dg_tick call");
        }
        let secs = t.elapsed().as_secs_f64();

        let hash = inst
            .get_typed_func::<(), u64>(&store, "dg_frame_hash")
            .expect("dg_frame_hash")
            .call(&mut store, ())
            .expect("hash");
        let gametic = inst
            .get_typed_func::<(), i32>(&store, "dg_gametic")
            .expect("dg_gametic")
            .call(&mut store, ())
            .expect("gametic");

        Run {
            label: "wasmi",
            frames,
            secs,
            hash,
            gametic,
        }
    }
}

// ---------------------------------------------------------------------------

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(wad_path) = args.next() else {
        eprintln!("usage: doombench <freedoom1.wad> [frames]");
        eprintln!();
        eprintln!("Freedoom (3-clause BSD, freely redistributable):");
        eprintln!("  https://github.com/freedoom/freedoom/releases");
        std::process::exit(2);
    };
    let frames: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let wad = std::fs::read(&wad_path).expect("read WAD");
    println!(
        "doombench: {} ({:.1} MB), {}x{}, {frames} frames\n",
        wad_path,
        wad.len() as f64 / (1024.0 * 1024.0),
        option_env!("DOOM_RESX").unwrap_or("?"),
        option_env!("DOOM_RESY").unwrap_or("?"),
    );

    // Native is the denominator, so it runs first and its output is the reference.
    let n = run_native(&wad, &wad_path, frames);
    let w = run_wasm(&wad, &wad_path, frames);

    // Agreement before speed. A faster build that renders something else has
    // changed the program.
    if let Some(w) = &w {
        let px_ok = n.hash == w.hash;
        let tic_ok = n.gametic == w.gametic;
        println!(
            "agreement: frame {} (native {:#018x}, wasmi {:#018x})",
            if px_ok { "MATCH" } else { "DIFFER" },
            n.hash,
            w.hash
        );
        println!(
            "agreement: gametic {} (native {}, wasmi {})\n",
            if tic_ok { "MATCH" } else { "DIFFER" },
            n.gametic,
            w.gametic
        );
        if !px_ok || !tic_ok {
            println!(
                "  NOTE: the two builds are not running the same program, so the\n\
                         ratio below is not a measurement of the interpreter.\n"
            );
        }
    }

    for r in [Some(&n), w.as_ref()].into_iter().flatten() {
        println!(
            "{:>7}: {:8.2} fps  {:7.2} ms/frame  ({:.2}s for {} frames)",
            r.label,
            r.fps(),
            r.ms(),
            r.secs,
            r.frames
        );
    }

    // Sizes the manifest's `wasm.fuel`; see `fuel_per_frame`.
    if option_env!("DOOM_WASM").is_some() {
        if let Some(bytes) = option_env!("DOOM_WASM").and_then(|p| std::fs::read(p).ok()) {
            // A short pass: this only needs an average, and metering is slow.
            if let Some(f) = wasm::fuel_per_frame(&bytes, &wad, &wad_path, 60) {
                println!(
                    "\nfuel: {f} per frame  (package_ui's CALL_FUEL is 2,000,000 — \
                     a Doom frame needs {:.1}x that)",
                    f as f64 / 2_000_000.0
                );
            }
        }
    }

    if let Some(w) = &w {
        let ratio = w.ms() / n.ms();
        println!("\ninterpreter tax: {ratio:.1}x native");
        // The gate this harness exists to answer.
        println!(
            "verdict: {}",
            if w.fps() >= 35.0 {
                "wasm is playable (>= 35 fps, Doom's own tic rate) — a wasm app package works"
            } else if w.fps() >= 20.0 {
                "wasm is marginal (20-35 fps) — playable but with no headroom on slower hardware"
            } else {
                "wasm is too slow (< 20 fps) — the port needs a ring-3 tenant"
            }
        );
        println!(
            "\nNB in-kernel will be slower than this host figure. The PDF renderer\n\
             measured ~3x on heavy pages, scaling with working set (suspected guest\n\
             TLB pressure under stage-2 translation), so treat the number above as\n\
             an upper bound rather than a prediction."
        );
    } else {
        println!("\n(wasm side unavailable — see the build warning)");
    }
}
