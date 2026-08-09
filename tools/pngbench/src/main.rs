//! Is wasmi fast enough to decode images?
//!
//! The plan's gate before porting the image viewer: if the interpreter is ~50x the in-kernel
//! decoder on a multi-megapixel PNG, then wasm is the right sandbox only for small images and
//! the plan changes. Measured rather than assumed, because the answer decides the design.
//!
//! Both sides run **the same source**: `kernel/src/image/{inflate,png}.rs` mounted by `#[path]`
//! natively here, and the identical files compiled to wasm in `tools/png-wasm`. So the ratio is
//! the interpreter's overhead and nothing else — not two implementations, not two algorithms.
//!
//! Host rather than in-kernel deliberately (the `cortexdiff`/`onnxdiff`/`h264diff` pattern):
//! the overhead is a property of wasmi and this code, not of QEMU, and measuring here takes
//! seconds and needs no kernel build.

// The mounted kernel modules are `no_std` and name `alloc` explicitly; on the host `alloc` is
// re-exported by std but must still be linked under that name for their `use` paths to resolve.
extern crate alloc;

use std::time::Instant;

/// Mirrors `kernel/src/image/mod.rs`'s `Image`; `png.rs` constructs it.
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
}

#[path = "../../../kernel/src/image/inflate.rs"]
pub mod inflate;
#[path = "../../../kernel/src/image/png.rs"]
pub mod png;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "assets/samples/images/sudoku.png".into());
    let wasm_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tools/png-wasm/target/wasm32-unknown-unknown/release/chitti_png_wasm.wasm".into());

    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    println!("input: {path} ({} KiB)", bytes.len() / 1024);

    // --- native ------------------------------------------------------------------------
    let t = Instant::now();
    let native = png::decode(&bytes).expect("native decode failed");
    let native_ms = t.elapsed().as_secs_f64() * 1e3;
    let mp = (native.w * native.h) as f64 / 1e6;
    println!("native: {:.1} ms   {}x{} ({mp:.2} MP)", native_ms, native.w, native.h);

    // --- wasmi -------------------------------------------------------------------------
    let wasm = std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {wasm_path}: {e} -- build tools/png-wasm first"));
    let engine = wasmi::Engine::default();
    let module = wasmi::Module::new(&engine, &wasm[..]).expect("module");
    let mut store = wasmi::Store::new(&engine, ());
    let instance = wasmi::Linker::<()>::new(&engine)
        .instantiate(&mut store, &module)
        .expect("instantiate")
        .start(&mut store)
        .expect("start");
    let alloc = instance.get_typed_func::<i32, i32>(&store, "chitti_alloc").expect("chitti_alloc");
    let decode = instance.get_typed_func::<(i32, i32), i64>(&store, "png_decode").expect("png_decode");
    let mem = instance.get_memory(&store, "memory").expect("memory");

    let p = alloc.call(&mut store, bytes.len() as i32).expect("alloc");
    mem.write(&mut store, p as usize, &bytes).expect("write input");
    let t = Instant::now();
    let packed = decode.call(&mut store, (p, bytes.len() as i32)).expect("png_decode");
    let wasm_ms = t.elapsed().as_secs_f64() * 1e3;

    // Unpack the raw ABI: [w, h, ok] then pixels.
    let (rp, rn) = (((packed >> 32) & 0xffff_ffff) as usize, (packed & 0xffff_ffff) as usize);
    let mut hdr = [0u8; 12];
    mem.read(&store, rp, &mut hdr).expect("read header");
    let g = |i: usize| u32::from_le_bytes([hdr[i], hdr[i + 1], hdr[i + 2], hdr[i + 3]]) as usize;
    let (ww, wh, ok) = (g(0), g(4), g(8));
    println!("wasmi : {:.1} ms   {ww}x{wh} ok={ok}   ({rn} bytes out)", wasm_ms);

    // --- the differential, which is the point of mounting the same source ---------------
    assert_eq!(ok, 1, "wasm decode reported failure");
    assert_eq!((ww, wh), (native.w, native.h), "dimensions disagree");
    let mut pix = vec![0u8; native.pixels.len() * 4];
    mem.read(&store, rp + 12, &mut pix).expect("read pixels");
    let same = native
        .pixels
        .iter()
        .zip(pix.chunks_exact(4))
        .all(|(a, b)| *a == u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    println!("pixels: {}", if same { "byte-identical" } else { "DIFFER" });
    assert!(same, "same source, different pixels -- the harness is wrong");

    println!("\nwasmi is {:.1}x native  ({:.1} ms vs {:.1} ms)", wasm_ms / native_ms, wasm_ms, native_ms);
    println!("verdict: {}", if wasm_ms / native_ms < 10.0 { "usable for the viewer" } else { "too slow for large images -- see the plan" });
}
