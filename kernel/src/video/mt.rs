//! Multi-core helpers for the video path (aarch64 SMP).
//!
//! H.264 CABAC itself is serial within a slice; the wins are **data-parallel**
//! stages after a picture exists: YUV→RGB convert and framebuffer letterbox
//! scale. Same barrier `parallel_for` Cortex matvecs use.

/// Run `f(row0, row1)` over `[0, n_rows)` split across online cores when
/// profitable; otherwise single-threaded.
///
/// # Safety
/// `f` must be safe for concurrent calls on **disjoint** row ranges.
#[inline]
pub unsafe fn parallel_rows(n_rows: usize, min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
    #[cfg(not(test))]
    {
        // Arch-neutral: was `cfg(aarch64)`, so x86 converted every video frame's
        // rows on one core.
        if crate::arch::online_cpus() > 1 && n_rows >= min_chunk.saturating_mul(2).max(16) {
            // SAFETY: caller contract.
            unsafe {
                crate::arch::parallel_for(n_rows, min_chunk.max(8), f, ctx);
            }
            return;
        }
    }
    let _ = min_chunk;
    // SAFETY: single range, caller contract.
    unsafe {
        f(0, n_rows, ctx);
    }
}

/// Context for parallel letterbox scale of an RGB frame into a destination
/// buffer (used by present). Each worker writes disjoint output rows.
#[repr(C)]
pub struct ScaleCtx {
    pub src: *const u32,
    pub src_w: usize,
    pub src_h: usize,
    pub src_len: usize,
    pub dst: *mut u32,
    pub dst_w: usize,
    pub dst_h: usize,
    pub dst_len: usize,
}

/// Scale rows `[row0, row1)` nearest-neighbour from src into dst.
///
/// # Safety
/// `ctx` must point to a live [`ScaleCtx`]; row ranges must not overlap.
pub unsafe fn scale_worker(row0: usize, row1: usize, ctx: *mut u8) {
    let c = &*(ctx as *const ScaleCtx);
    let src = core::slice::from_raw_parts(c.src, c.src_len);
    let dst = core::slice::from_raw_parts_mut(c.dst, c.dst_len);
    let row1 = row1.min(c.dst_h);
    if row0 >= row1 || c.dst_w == 0 || c.src_w == 0 || c.src_h == 0 {
        return;
    }
    for dy in row0..row1 {
        let sy = dy * c.src_h / c.dst_h;
        let srow = sy * c.src_w;
        let drow = dy * c.dst_w;
        for dx in 0..c.dst_w {
            let sx = dx * c.src_w / c.dst_w;
            dst[drow + dx] = src[srow + sx];
        }
    }
}
