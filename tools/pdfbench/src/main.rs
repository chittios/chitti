//! Is the wasm PDF rasterizer fast enough, and does it draw the same pixels?
//!
//! Two questions, one harness, because they have the same setup:
//!
//! 1. **The interpreter's tax.** The renderer cannot run in a ring-3 tenant
//!    (hayro's tree is `std`-bound), so its sandbox is wasm and the kernel pays
//!    wasmi's interpretation cost. That cost decided the build profile: at
//!    `opt-level = "s"` a dense LaTeX page took ~7.0 s, at `3` ~0.82 s. A number
//!    that moves a design decision by 8.5x has to be measured, not assumed.
//! 2. **The boundary.** Native and wasm run *the same crate* (`rlib` here,
//!    `cdylib` there), so any pixel difference is the ABI — the header struct,
//!    the pointer/length report, the premultiplied-RGBA byte order — and not two
//!    implementations drifting. That is the half that can actually be wrong,
//!    exactly as the image tenant's differential found.
//!
//! Host rather than in-kernel on purpose (the `pngbench`/`onnxdiff` pattern):
//! both answers are properties of wasmi and this code, not of QEMU.
//!
//! ```text
//! cargo run --release -- <file.pdf> [scale] [pages]
//! ```
//!
//! Writes `/tmp/pdfbench-<n>.ppm` per page (native pixels) for eyeballing, and
//! prints a per-page native/wasm timing and byte-comparison table.

use std::time::Instant;

const MODULE: &str = "../pdfrender-wasm/target/wasm32-unknown-unknown/release/chitti_pdfrender_wasm.wasm";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: pdfbench <file.pdf> [scale=1.5] [pages=3]");
        std::process::exit(2);
    }
    let pdf = std::fs::read(&args[1]).expect("read pdf");
    let scale: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let pages: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    println!("{}: {} KiB, scale {scale}", args[1], pdf.len() / 1024);

    let native = render_native(&pdf, scale, pages);
    let wasm = render_wasm(&pdf, scale, pages);

    println!("\n page      native      wasmi   ratio   pixels       native-vs-wasm");
    for (i, (n, w)) in native.iter().zip(wasm.iter()).enumerate() {
        let ratio = w.ms as f64 / n.ms.max(1) as f64;
        println!(
            "  {i:>3}  {:>8} ms  {:>7} ms  {ratio:>5.0}x  {}x{}   {}",
            n.ms,
            w.ms,
            n.w,
            n.h,
            compare(n, w)
        );
    }
    println!(
        "\nA channel delta of +-1 on a minority of pixels is expected, not a bug: vello_cpu picks\n\
         a SIMD level at runtime, so the native build blends with NEON while the wasm build takes\n\
         the scalar fallback. Geometry must match exactly — a differing *pixel count*, a large\n\
         max delta, or a whole shifted region is the ABI or the renderer, and that is the thing\n\
         this harness is here to catch."
    );
}

/// Summarize a native/wasm pixel difference: identical, rounding-level, or not.
fn compare(n: &Rendered, w: &Rendered) -> String {
    if (n.w, n.h) != (w.w, w.h) {
        return format!("GEOMETRY DIFFERS ({}x{} vs {}x{})", n.w, n.h, w.w, w.h);
    }
    if n.pixels.len() != w.pixels.len() {
        return format!("LENGTH DIFFERS ({} vs {})", n.pixels.len(), w.pixels.len());
    }
    let mut differing = 0u64;
    let mut max_delta = 0u8;
    for (a, b) in n.pixels.chunks(4).zip(w.pixels.chunks(4)) {
        let d = (0..4).map(|i| a[i].abs_diff(b[i])).max().unwrap_or(0);
        if d > 0 {
            differing += 1;
            max_delta = max_delta.max(d);
        }
    }
    if differing == 0 {
        return "identical".into();
    }
    let pct = 100.0 * differing as f64 / (n.pixels.len() / 4) as f64;
    let verdict = if max_delta <= 2 { "rounding" } else { "SUSPECT" };
    format!("{pct:.2}% of px differ, max delta {max_delta} ({verdict})")
}

struct Rendered {
    ms: u128,
    w: u32,
    h: u32,
    pixels: Vec<u8>,
}

/// The renderer called directly — the speed the kernel would get if hayro's
/// dependency tree were ever `no_std` enough for a ring-3 tenant.
fn render_native(pdf: &[u8], scale: f32, pages: u32) -> Vec<Rendered> {
    use chitti_pdfrender_wasm as r;
    // Through the module's inner API, not its `extern "C"` exports: those report
    // addresses as `i32`, which is the whole pointer in wasm and a truncated one
    // on a 64-bit host. Same functions the exports wrap, so the comparison below
    // still covers everything but the pointer packing itself.
    let n = r::open(pdf.to_vec()).expect("pdf_open");
    println!("native: {n} pages");
    let mut out = Vec::new();
    for page in 0..pages.min(n as u32) {
        let t = Instant::now();
        let (w, h, pixels) = r::render_page(page as usize, scale)
            .unwrap_or_else(|e| panic!("render_page({page}) failed: {e}"));
        let ms = t.elapsed().as_millis();
        write_ppm(page, w, h, pixels);
        out.push(Rendered { ms, w, h, pixels: pixels.to_vec() });
    }
    out
}

/// The renderer under wasmi 0.40 with fuel metering on — the kernel's setup
/// (`agent::wasm_rt::make_engine`), so the timings transfer.
fn render_wasm(pdf: &[u8], scale: f32, pages: u32) -> Vec<Rendered> {
    let bytes = std::fs::read(MODULE).unwrap_or_else(|e| {
        panic!("read {MODULE}: {e}\nbuild it: cd ../pdfrender-wasm && cargo build --release --target wasm32-unknown-unknown")
    });
    let mut cfg = wasmi::Config::default();
    cfg.consume_fuel(true);
    let engine = wasmi::Engine::new(&cfg);
    let t = Instant::now();
    let module = wasmi::Module::new(&engine, &bytes[..]).expect("validate");
    println!("wasm: {} KiB module, validated in {:?}", bytes.len() / 1024, t.elapsed());
    let mut store = wasmi::Store::new(&engine, ());
    store.set_fuel(u64::MAX / 2).unwrap();
    let linker = wasmi::Linker::<()>::new(&engine);
    let inst = linker
        .instantiate(&mut store, &module)
        .expect("instantiate")
        .start(&mut store)
        .expect("start");
    let mem = inst.get_memory(&store, "memory").expect("memory export");
    let alloc = inst.get_typed_func::<i32, i32>(&store, "chitti_alloc").unwrap();
    let open = inst.get_typed_func::<(i32, i32), i32>(&store, "pdf_open").unwrap();
    let render = inst.get_typed_func::<(i32, i32), i32>(&store, "pdf_render").unwrap();
    let last_err = inst.get_typed_func::<(), i32>(&store, "pdf_last_error").unwrap();

    let staged = alloc.call(&mut store, pdf.len() as i32).unwrap();
    mem.write(&mut store, staged as usize, pdf).unwrap();
    let n = open.call(&mut store, (staged, pdf.len() as i32)).unwrap();
    assert!(n > 0, "pdf_open failed: {n}");

    let mut out = Vec::new();
    for page in 0..pages.min(n as u32) {
        let before = store.get_fuel().unwrap();
        let t = Instant::now();
        let hdr = render.call(&mut store, (page as i32, (scale * 1000.0) as i32)).unwrap();
        let ms = t.elapsed().as_millis();
        let fuel = before - store.get_fuel().unwrap();
        assert!(hdr != 0, "pdf_render failed: {}", last_err.call(&mut store, ()).unwrap());
        let mut words = [0u8; 16];
        mem.read(&store, hdr as usize, &mut words).unwrap();
        let g = |i: usize| u32::from_le_bytes(words[i * 4..i * 4 + 4].try_into().unwrap());
        let (w, h, ptr, len) = (g(0), g(1), g(2), g(3));
        let mut pixels = vec![0u8; len as usize];
        mem.read(&store, ptr as usize, &mut pixels).unwrap();
        println!(
            "  wasm page {page}: {ms} ms, {w}x{h}, {} MiB linear memory, {} Mfuel",
            mem.size(&store) * 64 / 1024,
            fuel / 1_000_000
        );
        out.push(Rendered { ms, w, h, pixels });
    }
    out
}

fn write_ppm(page: u32, w: u32, h: u32, rgba: &[u8]) {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for px in rgba.chunks(4) {
        out.extend_from_slice(&px[..3]);
    }
    let path = format!("/tmp/pdfbench-{page}.ppm");
    std::fs::write(&path, out).ok();
    println!("  wrote {path}");
}
