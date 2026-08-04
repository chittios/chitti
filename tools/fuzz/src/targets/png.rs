//! Fuzz target: the kernel's PNG decoder (`kernel/src/image/{inflate,png}.rs`).
//!
//! PNGs are attacker-supplied files decoded in ring 3 by `synapse::tenant`
//! (the `ImageTenant`), so a panic here is contained — but it is still the
//! kernel's own decoder, and the ring-3 differential (`cargo xtask test`) is
//! the defence against it going wrong in-kernel. Fuzzing it on the host is the
//! cheapest way to find the panic first. Mounted the same way `pngbench` does:
//! the real `inflate.rs` + `png.rs` under `#[path]`, with a shim `Image`.

// `kernel/src/image/*` are `no_std` and name `alloc` explicitly.

/// Mirrors `kernel/src/image/mod.rs`'s `Image`; `png.rs` constructs it.
#[allow(dead_code)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
}

#[path = "../../../../kernel/src/image/inflate.rs"]
pub mod inflate;
#[path = "../../../../kernel/src/image/png.rs"]
pub mod png;

pub fn run(data: &[u8]) {
    let _ = png::decode(data);
}
