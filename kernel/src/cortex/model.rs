//! Qwen3.5-0.8B hybrid decoder forward pass (`CHITTI_OS_HANDOFF.md` Phase
//! 3). This is a gated-DeltaNet + gated-attention hybrid, reconstructed
//! against llama.cpp's `qwen35` graph and the HF `Qwen3_5` modeling code:
//!
//! - 3 of every 4 layers are **gated-DeltaNet** (linear attention / SSM):
//!   a causal depthwise conv1d + SiLU over the q/k/v projection, L2-norm'd
//!   q/k, and the recurrent gated delta rule
//!   `S = g·S + β·kᵀ(v − Sᵀk); o = Sᵀq` with `g = exp(−exp(A)·softplus(α+dt))`,
//!   then a gated RMSNorm `RMSNorm(o)·SiLU(z)` and an output projection.
//! - every 4th layer is **full attention** with QK-norm, partial (64-dim)
//!   NeoX RoPE, GQA (8 q / 2 kv heads, head_dim 256), and a per-head
//!   sigmoid output gate (query and gate are interleaved in `attn_q`).
//! - SwiGLU FFN and tied output embeddings throughout.
//!
//! One token per `forward` call; deltanet layers carry a recurrent state +
//! conv ring in `Cache`, attention layers carry a KV history. Deterministic.

use super::gguf::{Config, Family, Gguf, GgufError, GGML_TYPE_F32};
use super::tensor::{self, QK};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Cycles spent in each phase of [`Model::prefill_batched`], accumulated across
/// every chunk since the last [`reset_prefill_phases`]. Prefill has exactly
/// three kinds of work and they scale differently — the projections are
/// weight-stationary matmuls, attention is quadratic in position, and the
/// DeltaNet recurrence is linear but only parallel across heads — so "prefill is
/// slow" is three different diagnoses. `/perf` prints the split.
///
/// Ordering is `Relaxed` throughout: these are counters read after the fact,
/// never a synchronisation signal.
static PHASE_PROJ: AtomicU64 = AtomicU64::new(0);
static PHASE_ATTN: AtomicU64 = AtomicU64::new(0);
static PHASE_DELTA: AtomicU64 = AtomicU64::new(0);
static PHASE_ELEM: AtomicU64 = AtomicU64::new(0);

/// Zero the prefill phase counters (call before a measured run).
pub fn reset_prefill_phases() {
    for c in [&PHASE_PROJ, &PHASE_ATTN, &PHASE_DELTA, &PHASE_ELEM] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Prefill cycles by phase since the last reset: projections, attention core,
/// DeltaNet core, elementwise (norms/SwiGLU/residuals).
pub fn prefill_phases() -> [u64; 4] {
    [
        PHASE_PROJ.load(Ordering::Relaxed),
        PHASE_ATTN.load(Ordering::Relaxed),
        PHASE_DELTA.load(Ordering::Relaxed),
        PHASE_ELEM.load(Ordering::Relaxed),
    ]
}

/// Add the cycles elapsed since `t0` to `c`, and return a fresh timestamp — so
/// a phase boundary is one call and no interval is double-counted or dropped.
fn phase_mark(c: &AtomicU64, t0: u64) -> u64 {
    let now = crate::arch::cycle_count();
    c.fetch_add(now.wrapping_sub(t0), Ordering::Relaxed);
    now
}

/// Where tap `j` of window position `mi` reads from, in a causal depthwise
/// conv1d of kernel width `ck` run over a whole prefill window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvTap {
    /// Row `p` of the window's own qkv batch.
    Window(usize),
    /// Slot `x` of the conv ring as it stood *entering* the window.
    Ring(usize),
}

/// Resolve one tap of the windowed conv1d — pure, so the mapping can be pinned
/// off-hardware.
///
/// The per-position form shifted the ring down a slot and appended the current
/// vector *before* convolving, so its tap `j` read post-shift slot `j`. Over a
/// window there is no shift: tap `j` wants the vector at `p = mi + j - (ck-1)`,
/// which is window row `p` when `p >= 0` and otherwise entering-ring slot
/// `ck + p` — **not** `ck - 1 + p`, which is where the shift's `-1` tries to
/// follow you. That off-by-one reads every tap one position too old, which
/// still produces fluent text, so it is a bug you find by fixture, not by eye.
pub fn conv_tap(mi: usize, j: usize, ck: usize) -> ConvTap {
    let p = mi as isize + j as isize - (ck as isize - 1);
    if p >= 0 {
        ConvTap::Window(p as usize)
    } else {
        ConvTap::Ring((ck as isize + p) as usize)
    }
}

/// Advance the conv ring past an `m`-position window, in place.
///
/// New slot `j` must hold the qkv vector at window index `m - ck + j`: inside
/// the window when that is non-negative, else the old ring's slot `m + j`.
/// Reading from a strictly higher index makes the carry-over a forward
/// in-place copy.
pub fn advance_conv_ring(ring: &mut [f32], qkv: &[f32], m: usize, conv_dim: usize, ck: usize) {
    if m >= ck {
        ring.copy_from_slice(&qkv[(m - ck) * conv_dim..m * conv_dim]);
    } else {
        let keep = ck - m; // slots that survive from the old ring
        ring.copy_within(m * conv_dim.., 0);
        ring[keep * conv_dim..].copy_from_slice(&qkv[..m * conv_dim]);
    }
}

/// `parallel_for`'s `min_chunk` for a fan-out of roughly `work` multiply-adds:
/// `1` (split freely) once the job outweighs a fleet wake, else a value large
/// enough that `parallel_for` keeps the whole range inline.
///
/// This exists because the DeltaNet recurrence is *always* parallel across
/// heads, at every window size — but at a decode step's one position it is only
/// ~0.8M MACs a layer, and fanning that out measured **slower** than running it
/// inline (35 -> 20 tok/s of decode: 36 extra fleet wakes a token bought less
/// work than they cost). Parallelism that is available is not automatically
/// parallelism that is worth taking.
fn fanout_chunk(work: usize) -> usize {
    /// Measured on the 0.8B under HVF: below roughly this much work per
    /// dispatch, the wake + barrier dominates.
    const WORTH_A_WAKE: usize = 4 << 20;
    if work >= WORTH_A_WAKE {
        1
    } else {
        usize::MAX
    }
}

/// A quantized weight tensor: its raw bytes plus the GGUF quant type, so the
/// matvec can pick the right kernel per tensor (the 9B GGUF mixes Q4_0, Q4_1,
/// Q8_0, Q5_K and Q6_K; the 0.8B is all Q8_0).
#[derive(Clone, Copy)]
struct QWeight<'a> {
    data: &'a [u8],
    qt: u32,
}

fn f32_tensor<'a>(g: &Gguf<'a>, name: &str, n: usize) -> Result<&'a [f32], GgufError> {
    let info = g.tensor(name)?;
    if info.ggml_type != GGML_TYPE_F32 {
        return Err(GgufError::MissingTensor);
    }
    let bytes = g.tensor_bytes(name, n * 4)?;
    // SAFETY: GGUF tensor data is 32-byte aligned; host is little-endian.
    Ok(unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) })
}

/// Load a quantized weight of any supported type: read its `ggml_type`, compute
/// the on-disk byte size from the type's block layout, validate, and return the
/// bytes tagged with the type.
fn qtensor<'a>(g: &Gguf<'a>, name: &str, n_cols: usize, n_rows: usize) -> Result<QWeight<'a>, GgufError> {
    let qt = g.tensor(name)?.ggml_type;
    let (block_bytes, elems) = tensor::block_layout(qt);
    if block_bytes == 0 || n_cols % elems != 0 {
        return Err(GgufError::MissingTensor); // unsupported quant / bad shape
    }
    let bytes = n_rows * (n_cols / elems) * block_bytes;
    Ok(QWeight { data: g.tensor_bytes(name, bytes)?, qt })
}

/// Dispatch one matvec `y = W · x` by the weight's quant type. Q8_0 and Q4_0
/// take the fast int8-activation SDOT path (SMP + batched-capable); the
/// remaining k-quant types (Q4_1/Q5_K/Q6_K, a minority of tensors) take the
/// generic exact-f32 dequant-and-dot path (still row-split across cores). The
/// int8 activation is fine for both Q8_0 and Q4_0 (validated: 9B greedy output
/// matches llama.cpp byte-for-byte once the DeltaNet head grouping is correct).
fn matvec_qw(qw: QWeight, x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    if qw.qt == tensor::QT_Q8_0 {
        tensor::matvec_q8_0_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    if qw.qt == tensor::QT_Q4_0 {
        tensor::matvec_q4_0_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    if qw.qt == tensor::QT_Q4_K && n_cols % tensor::QK_K == 0 {
        tensor::matvec_q4_k_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    if qw.qt == tensor::QT_Q6_K && n_cols % tensor::QK_K == 0 {
        tensor::matvec_q6_k_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    if qw.qt == tensor::QT_Q2_0 && n_cols % tensor::QK2_0 == 0 {
        tensor::matvec_q2_0_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    if qw.qt == tensor::QT_Q1_0 && n_cols % tensor::QK1_0 == 0 {
        tensor::matvec_q1_0_fast(qw.data, x, y, xq, xs, n_rows, n_cols);
        return;
    }
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    let _ = (&mut *xq, &mut *xs);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `qw.data` holds `n_rows` rows for `qw.qt`/`n_cols`; `x`/`y` sized.
    unsafe {
        crate::arch::aarch64::smp::matvec_quant(qw.qt, qw.data.as_ptr(), x.as_ptr(), y.as_mut_ptr(), n_rows, n_cols);
    }
    // x86 (and any other arch): split the row range across the compute fleet.
    // `matvec_quant_rows` was already row-ranged — this used to pass `0..n_rows`,
    // i.e. the whole matrix on one core, which is what made inference single-core
    // on x86 while aarch64 fanned the identical work across every core.
    //
    // Each worker writes only `y[start..end]` and reads only its own weight rows
    // plus all of `x`, so the ranges are disjoint and the result is independent of
    // how the split falls (no cross-range reassociation).
    #[cfg(not(target_arch = "aarch64"))]
    {
        struct Ctx {
            qt: u32,
            w: *const u8,
            x: *const f32,
            y: *mut f32,
            n_cols: usize,
        }
        /// # Safety
        /// `ctx` is a live `Ctx` for the duration of the fan-out, and `[start,
        /// end)` is disjoint from every other worker's range.
        unsafe fn rows(start: usize, end: usize, ctx: *mut u8) {
            // SAFETY: the caller passes the `Ctx` published below, which outlives
            // the `parallel_for` call.
            let c = unsafe { &*(ctx as *const Ctx) };
            // SAFETY: forwarded contract; this worker owns rows `start..end`.
            unsafe { tensor::matvec_quant_rows(c.qt, c.w, c.x, c.y, start, end, c.n_cols) };
        }
        let mut ctx =
            Ctx { qt: qw.qt, w: qw.data.as_ptr(), x: x.as_ptr(), y: y.as_mut_ptr(), n_cols };
        // Below this many rows the barrier costs more than the work.
        const MIN_ROWS: usize = 32;
        // SAFETY: `rows` is safe on disjoint sub-ranges sharing `ctx`, and `ctx`
        // lives until `parallel_for` returns.
        unsafe {
            crate::arch::parallel_for(n_rows, MIN_ROWS, rows, &mut ctx as *mut Ctx as *mut u8)
        };
    }
}

/// Per-layer weights; one of the two variants depending on layer type.
enum LayerKind<'a> {
    Attn {
        q: QWeight<'a>, // [dim -> n_head*head_dim*2] (query+gate interleaved)
        k: QWeight<'a>, // [dim -> n_kv*head_dim]
        v: QWeight<'a>,
        o: QWeight<'a>, // [n_head*head_dim -> dim]
        q_norm: &'a [f32],
        k_norm: &'a [f32],
    },
    Delta {
        qkv: QWeight<'a>,    // [dim -> conv_dim]
        gate: QWeight<'a>,   // [dim -> value_dim]  (z)
        conv1d: &'a [f32],   // [conv_dim * conv_kernel], tap j of channel c at c*K+j
        dt_bias: &'a [f32],  // [n_v_heads]
        a_log: &'a [f32],    // [n_v_heads]
        alpha: QWeight<'a>,  // [dim -> n_v_heads]
        beta: QWeight<'a>,   // [dim -> n_v_heads]
        norm: &'a [f32],     // [head_v_dim]
        out: QWeight<'a>,    // [value_dim -> dim]
    },
    /// Gemma-4 attention (llama.cpp `gemma4.cpp` graph): per-layer geometry
    /// (sliding layers GQA with a windowed KV ring; global layers MQA with a
    /// larger head_dim, `v` absent → K reused as V, p-RoPE freq factors),
    /// per-head QK-norms with an *unweighted* RMS on V, attention scale 1.0.
    GemmaAttn {
        q: QWeight<'a>,         // [dim -> n_head*head_dim]
        k: QWeight<'a>,         // [dim -> n_kv*head_dim]
        v: Option<QWeight<'a>>, // absent on global layers (V = K)
        o: QWeight<'a>,         // [n_head*head_dim -> dim]
        q_norm: &'a [f32],      // [head_dim]
        k_norm: &'a [f32],      // [head_dim]
        n_kv: usize,
        head_dim: usize,
        /// `Some(W)` = sliding-window layer (KV ring of W); `None` = global.
        window: Option<usize>,
        rope_base: f32,
        /// p-RoPE frequency divisors (global layers; `rope_freqs.weight`).
        freq_factors: Option<&'a [f32]>,
        /// Layer whose KV cache this layer attends over. Normally itself; for a
        /// **shared-KV** layer (`il >= n_layer_kv_from_start`) it is an earlier
        /// layer of the same kind, and this layer computes no K/V at all — its
        /// `k`/`v` tensors exist in the file but are unused, exactly as
        /// llama.cpp marks them `TENSOR_NOT_REQUIRED`. Getting this wrong is
        /// silent: the layer still produces plausible activations from the wrong
        /// history, which is what made 18 of Gemma-4-E4B's 42 layers wrong.
        kv_src: usize,
    },
    /// LFM2 attention (llama.cpp `lfm2.cpp`): plain GQA, RMS QK-norm + full
    /// per-head RoPE, attention scale `1/sqrt(head_dim)`, no gate.
    Lfm2Attn {
        q: QWeight<'a>,        // [dim -> n_head*head_dim]
        k: QWeight<'a>,        // [dim -> n_kv*head_dim]
        v: QWeight<'a>,        // [dim -> n_kv*head_dim]
        o: QWeight<'a>,        // [n_head*head_dim -> dim]
        q_norm: &'a [f32],     // [head_dim]
        k_norm: &'a [f32],     // [head_dim]
        n_kv: usize,
    },
    /// LFM2 recurrent **shortconv** block (llama.cpp `lfm2.cpp`): a 3-way
    /// projection split into `b`/`c`/`x`, the gated product `b·x` through a
    /// **causal depthwise conv1d** over a cached per-layer state (d_conv =
    /// `shortconv.l_cache - 1` columns), then `c · conv(b·x)` via an out
    /// projection. The state is the recurrent half of the hybrid.
    ShortConv {
        conv: &'a [f32],         // [l_cache * n_embd], tap j of channel c at c*K+j
        in_proj: QWeight<'a>,    // [dim -> 3*dim]
        out_proj: QWeight<'a>,   // [dim -> dim]
    },
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    /// The norm feeding the FFN (qwen `post_attention_norm` / gemma `ffn_norm`).
    post_norm: &'a [f32],
    /// Gemma sandwich norms: applied to the attention/FFN block *output*
    /// before its residual add. `None` (qwen) leaves the math untouched.
    post_attn_norm: Option<&'a [f32]>,
    ffn_post_norm: Option<&'a [f32]>,
    /// Gemma per-layer output scalar (`layer_output_scale`); 1.0 = absent.
    out_scale: f32,
    kind: LayerKind<'a>,
    ffn_gate: QWeight<'a>,
    ffn_up: QWeight<'a>,
    ffn_down: QWeight<'a>,
    /// Gemma E-series per-layer-embedding block, applied after the FFN residual:
    /// `x += pl_post_norm(pl_proj @ (gelu(pl_inp_gate @ x) * ple[layer]))`.
    /// All three are present together or not at all.
    pl_inp_gate: Option<QWeight<'a>>, // [dim -> E]
    pl_proj: Option<QWeight<'a>>,     // [E -> dim]
    pl_post_norm: Option<&'a [f32]>,  // [dim]
}

pub struct Model<'a> {
    pub config: Config,
    gguf: Gguf<'a>,
    token_embd: QWeight<'a>, // [dim, vocab] -- also the tied output when `output` is None
    output: Option<QWeight<'a>>, // separate (untied) output projection, if present
    output_norm: &'a [f32],
    layers: Vec<Layer<'a>>,
    vocab: usize,
    /// Gemma E-series per-layer embeddings: a second, per-layer token table
    /// (`[E*n_layer, vocab]`) plus the projection of the *scaled* input
    /// embedding that is added to it. `None` on every other model.
    pl_tok_embd: Option<QWeight<'a>>,
    pl_model_proj: Option<QWeight<'a>>,
    pl_proj_norm: Option<&'a [f32]>,
    /// Whether prefill takes the window-batched path. Deliberately **not** a
    /// function of the quant mix: [`Model::batched_proj`] batches the tensors
    /// whose type has a weight-stationary kernel and applies the rest a position
    /// at a time, so a mixed file loses only the amortization on its
    /// non-batchable tensors instead of dropping the whole model to the
    /// per-token path. Gemma is excluded because `prefill_batched` has no
    /// implementation for its attention layer shape.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    batched: bool,
    /// Percent of quantized *projection* bytes whose quant type has a batched
    /// kernel. Reported by `/perf`, because "batched prefill" stopped being a
    /// yes/no once batching became per tensor — and because this number is the
    /// one that predicts prefill throughput on a real-world GGUF.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    batch_pct: u8,
}

/// Whether `qt` has a weight-stationary batched matmul kernel, i.e. whether a
/// projection with this quant type can be applied to a whole prefill window in
/// one pass over the weights.
///
/// Q8_0/Q1_0/Q2_0 have SDOT kernels on any aarch64; Q4_0's batched kernel is
/// i8mm-only, so it depends on a runtime CPU capability (and a hypervisor can
/// withhold it — VirtualBox does). Everything else — the K-quants, Q4_1, the
/// i-quants — has none, and falls back per tensor.
#[cfg(target_arch = "aarch64")]
fn has_batched_kernel(qt: u32) -> bool {
    match qt {
        tensor::QT_Q8_0 | tensor::QT_Q1_0 | tensor::QT_Q2_0 => true,
        tensor::QT_Q4_0 => crate::arch::has_i8mm(),
        _ => false,
    }
}
#[cfg(not(target_arch = "aarch64"))]
fn has_batched_kernel(_qt: u32) -> bool {
    false
}

/// Reusable per-forward scratch (no per-token allocation).
pub struct State {
    hidden: Vec<f32>,
    residual: Vec<f32>,
    norm: Vec<f32>,
    // attention scratch
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    // deltanet scratch
    qkv: Vec<f32>,
    z: Vec<f32>,
    gates: Vec<f32>, // per-head g decay
    betas: Vec<f32>,
    conv: Vec<f32>,
    delta_o: Vec<f32>,
    // ffn scratch
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_act: Vec<f32>,
    proj: Vec<f32>,
    pub logits: Vec<f32>,
    // int8-activation scratch for the fast (SDOT) matvec path, sized to the
    // widest matvec input so it is reused across every projection.
    xq: Vec<i8>,
    xs: Vec<f32>,
    /// Gemma E-series per-layer inputs for the current token (`E*n_layer`), and
    /// the `E`-wide gate scratch. Empty on every other model.
    ple: Vec<f32>,
    /// The per-layer *table* row for the current token, kept here rather than
    /// allocated per token.
    ple_tok: Vec<f32>,
    pl_gate: Vec<f32>,
}

impl State {
    pub fn new(c: &Config, vocab: usize) -> Self {
        let dim = c.embedding_length;
        let v = |n| alloc::vec![0.0f32; n];
        // DeltaNet scratch exists only for hybrid families; zero-sized else.
        let (ssm_inner, ssm_conv, ssm_heads) =
            c.ssm.map(|s| (s.inner, s.conv_dim(), s.dt_rank)).unwrap_or((0, 0, 0));
        // SWA families have a second (global-layer) attention geometry; the
        // scratch must cover the wider of the two.
        let hd_g = c.swa.map(|s| s.head_dim_global).unwrap_or(0);
        let kv_g = c.swa.map(|s| s.head_count_kv_global * s.head_dim_global).unwrap_or(0);
        // Gemma E-series per-layer-embedding scratch; zero-sized elsewhere.
        let ple_e = c.swa.map(|s| s.n_embd_per_layer).unwrap_or(0);
        let ple_len = ple_e * c.block_count;
        let q_width = (c.head_count * c.head_dim * 2).max(c.head_count * hd_g);
        let kv_width = (c.head_count_kv * c.head_dim).max(kv_g);
        // LFM2 shortconv scratch: the 3-way in_proj needs 3*dim, and the conv
        // uses `dim`-wide bx / conv_out staging (reusing the DeltaNet scratch,
        // which is zero-sized when the family has no ssm).
        let sc_qkv = if c.family == Family::Lfm2 { 3 * c.embedding_length } else { 0 };
        // Widest matvec input across all projections (columns): the norm-fed
        // projections use `dim`, ffn_down uses the FFN width, o_proj uses
        // head_count*head_dim (per geometry), and the DeltaNet output uses
        // ssm_inner.
        let max_cols = dim
            .max(c.feed_forward_length)
            .max(c.head_count * c.head_dim)
            .max(c.head_count * hd_g)
            .max(ssm_inner);
        Self {
            hidden: v(dim),
            residual: v(dim),
            norm: v(dim),
            q: v(q_width),
            k: v(kv_width),
            v: v(kv_width),
            attn_out: v(q_width),
            scores: v(c.context_length),
            qkv: v(ssm_conv.max(sc_qkv)),
            z: v(ssm_inner),
            gates: v(ssm_heads),
            betas: v(ssm_heads.max(c.embedding_length)),
            conv: v(ssm_conv),
            delta_o: v(ssm_inner.max(c.embedding_length)),
            ffn_gate: v(c.feed_forward_length),
            ffn_up: v(c.feed_forward_length),
            ffn_act: v(c.feed_forward_length),
            proj: v(dim),
            logits: v(vocab),
            xq: alloc::vec![0i8; max_cols],
            xs: alloc::vec![0.0f32; max_cols / QK],
            ple: v(ple_len),
            ple_tok: v(ple_len),
            pl_gate: v(ple_e),
        }
    }
}

/// Per-stream cache: KV history for attention layers (a fixed W-slot ring on
/// sliding-window layers), recurrent state + conv ring for gated-DeltaNet
/// layers.
///
/// `Clone` is what makes the prefix cache (`cortex::prefix`) possible: a
/// prefilled system prompt is kept as a snapshot and cloned back in, rather than
/// prefilled again. It has to be a copy, not a borrow — decoding mutates the KV
/// in place, so the snapshot would not survive the turn it seeded.
#[derive(Clone)]
pub struct Cache {
    attn_k: Vec<Vec<f32>>, // [layer] -> flattened [pos * (n_kv*head_dim)] (or a W-slot ring)
    attn_v: Vec<Vec<f32>>,
    /// Per layer: true when attn_k/attn_v is a preallocated sliding-window
    /// ring (fixed length; evict zeroes it) rather than a growing history.
    ring: Vec<bool>,
    delta_s: Vec<Vec<f32>>, // [layer] -> [n_v_heads * state * head_v_dim]
    conv: Vec<Vec<f32>>,    // [layer] -> conv ring [conv_kernel * conv_dim]
    /// LFM2 shortconv recurrent state: `[layer] -> [d_conv * n_embd]` (the
    /// last `d_conv` columns of `b·x`, prepended causally on the next token).
    /// Empty on every other family / attention layer.
    shortconv_state: Vec<Vec<f32>>,
    positions: usize,
}

impl Cache {
    pub fn new(c: &Config) -> Self {
        let n = c.block_count;
        // DeltaNet state sizes; attention-only families never take the else
        // branch below, so zero sizes are never allocated.
        let (s_size, conv_size) = c
            .ssm
            .map(|s| (s.dt_rank * s.state * s.head_dim(), s.conv_kernel * s.conv_dim()))
            .unwrap_or((0, 0));
        let mut attn_k = Vec::with_capacity(n);
        let mut attn_v = Vec::with_capacity(n);
        let mut ring = Vec::with_capacity(n);
        let mut delta_s = Vec::with_capacity(n);
        let mut conv = Vec::with_capacity(n);
        let mut shortconv_state = Vec::with_capacity(n);
        let sc_d_conv = c.shortconv_l_cache.saturating_sub(1);
        for l in 0..n {
            if c.is_attention_layer(l) {
                // Sliding-window layers bound their KV to a W-slot ring (the
                // base config geometry is the sliding-layer one); global /
                // full-history layers grow with the actual sequence.
                // A shared-KV layer allocates nothing: it attends over `kv_src`'s
                // cache (see `LayerKind::GemmaAttn::kv_src`).
                if !c.owns_kv(l) {
                    attn_k.push(Vec::new());
                    attn_v.push(Vec::new());
                    ring.push(c.is_sliding(l));
                } else if c.is_sliding(l) {
                    // `ring_slots`, not `window`: a shared layer reads this ring
                    // *after* the window was committed, so a bare W-slot ring
                    // would have already evicted the oldest entries that the
                    // earliest position in the chunk still needs. See
                    // `Config::ring_slots`.
                    let slots = c.ring_slots();
                    let kv_dim = c.head_count_kv * c.head_dim;
                    attn_k.push(alloc::vec![0.0f32; slots * kv_dim]);
                    attn_v.push(alloc::vec![0.0f32; slots * kv_dim]);
                    ring.push(true);
                } else {
                    attn_k.push(Vec::new());
                    attn_v.push(Vec::new());
                    ring.push(false);
                }
                delta_s.push(Vec::new());
                conv.push(Vec::new());
            } else {
                attn_k.push(Vec::new());
                attn_v.push(Vec::new());
                ring.push(false);
                delta_s.push(alloc::vec![0.0f32; s_size]);
                conv.push(alloc::vec![0.0f32; conv_size]);
            }
            // LFM2 shortconv recurrent layers keep a per-layer conv state.
            shortconv_state.push(if c.is_shortconv(l) {
                alloc::vec![0.0f32; sc_d_conv * c.embedding_length]
            } else {
                Vec::new()
            });
        }
        Self { attn_k, attn_v, ring, delta_s, conv, shortconv_state, positions: 0 }
    }

    pub fn len(&self) -> usize {
        self.positions
    }

    /// Heap bytes this cache holds — what a snapshot of it costs to keep.
    ///
    /// Measured from `len`, not `capacity`: a clone allocates exactly the live
    /// length, so this is the size of the copy the prefix cache would store, and
    /// budgeting on capacity would over-charge a cache that had grown and been
    /// rebuilt. The attention halves scale with position; the DeltaNet state and
    /// conv ring are fixed per layer.
    pub fn bytes(&self) -> usize {
        let f32s: usize = self
            .attn_k
            .iter()
            .chain(self.attn_v.iter())
            .chain(self.delta_s.iter())
            .chain(self.conv.iter())
            .map(|v| v.len())
            .sum();
        f32s * core::mem::size_of::<f32>() + self.ring.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Reset all recurrent state / KV history (KV evict). The continuation
    /// is reproduced by replaying tokens through the deterministic pass.
    /// Ring layers keep their fixed length and are zeroed; growing layers
    /// are truncated.
    pub fn evict(&mut self) {
        for (l, k) in self.attn_k.iter_mut().enumerate() {
            if self.ring[l] {
                k.iter_mut().for_each(|x| *x = 0.0);
            } else {
                k.clear();
            }
        }
        for (l, v) in self.attn_v.iter_mut().enumerate() {
            if self.ring[l] {
                v.iter_mut().for_each(|x| *x = 0.0);
            } else {
                v.clear();
            }
        }
        for s in &mut self.delta_s {
            s.iter_mut().for_each(|x| *x = 0.0);
        }
        for c in &mut self.conv {
            c.iter_mut().for_each(|x| *x = 0.0);
        }
        self.positions = 0;
        crate::ktrace::log("cortex.kv", "cache evicted (KV + recurrent state reset)");
    }
}

impl<'a> Model<'a> {
    pub fn load(gguf: Gguf<'a>) -> Result<Self, GgufError> {
        let c = &gguf.config;
        let dim = c.embedding_length;
        let ffn = c.feed_forward_length;
        let vocab = gguf.tokens.len().max(1);
        let head_dim = c.head_dim;
        let attn_q_dim = c.head_count * head_dim * 2;
        let kv_dim = c.head_count_kv * head_dim;
        let attn_o_in = c.head_count * head_dim;

        let token_embd = qtensor(&gguf, "token_embd.weight", dim, vocab)?;
        // Untied output projection if the GGUF has one (the 9B); else tied.
        let output = qtensor(&gguf, "output.weight", dim, vocab).ok();
        // LFM2 ships its final RMS norm under `token_embd_norm.weight` (a wrong
        // name llama.cpp special-cases via LLM_TENSOR_OUTPUT_NORM_LFM2); every
        // other family calls it `output_norm.weight`.
        let output_norm = match c.family {
            Family::Lfm2 => f32_tensor(&gguf, "token_embd_norm.weight", dim)?,
            _ => f32_tensor(&gguf, "output_norm.weight", dim)?,
        };
        // Gemma p-RoPE frequency divisors for global layers (one shared root
        // tensor, `head_dim_global/2` entries). Absent elsewhere.
        let rope_freqs: Option<&[f32]> = c
            .swa
            .and_then(|swa| f32_tensor(&gguf, "rope_freqs.weight", swa.rope_dim_global / 2).ok());

        let mut layers = Vec::with_capacity(c.block_count);
        for l in 0..c.block_count {
            let n = |s: &str| format!("blk.{l}.{s}");
            // Gemma sandwich norms + per-layer output scalar; None/1.0 for
            // hybrid families keeps their forward math untouched.
            let mut post_attn_norm = None;
            let mut ffn_post_norm = None;
            let mut out_scale = 1.0f32;
            let (kind, post_norm) = match c.family {
                Family::Gemma4 => {
                    // Per-layer geometry: base config carries the sliding-layer
                    // values; global layers override from SwaConfig.
                    let swa = c.swa.expect("gemma requires swa config");
                    let sliding = c.is_sliding(l);
                    let (hd, n_kv, rope_base, window, freq_factors) = if sliding {
                        (c.head_dim, c.head_count_kv, c.rope_freq_base, Some(swa.window), None)
                    } else {
                        (swa.head_dim_global, swa.head_count_kv_global, swa.freq_base_global, None, rope_freqs)
                    };
                    post_attn_norm = Some(f32_tensor(&gguf, &n("post_attention_norm.weight"), dim)?);
                    ffn_post_norm = Some(f32_tensor(&gguf, &n("post_ffw_norm.weight"), dim)?);
                    out_scale = f32_tensor(&gguf, &n("layer_output_scale.weight"), 1).map(|s| s[0]).unwrap_or(1.0);
                    let kind = LayerKind::GemmaAttn {
                        q: qtensor(&gguf, &n("attn_q.weight"), dim, c.head_count * hd)?,
                        k: qtensor(&gguf, &n("attn_k.weight"), dim, n_kv * hd)?,
                        // Global layers have no V projection: V = K.
                        v: qtensor(&gguf, &n("attn_v.weight"), dim, n_kv * hd).ok(),
                        o: qtensor(&gguf, &n("attn_output.weight"), c.head_count * hd, dim)?,
                        q_norm: f32_tensor(&gguf, &n("attn_q_norm.weight"), hd)?,
                        k_norm: f32_tensor(&gguf, &n("attn_k_norm.weight"), hd)?,
                        n_kv,
                        head_dim: hd,
                        window,
                        rope_base,
                        freq_factors,
                        // Shared-KV layers reuse an earlier layer of the same
                        // kind: sliding -> `kv_from_start - 2`, global -> `- 1`
                        // (llama.cpp's `layer_reuse_cb` for GEMMA3N/GEMMA4).
                        kv_src: if l < swa.n_layer_kv_from_start {
                            l
                        } else {
                            swa.n_layer_kv_from_start - if sliding { 2 } else { 1 }
                        },
                    };
                    (kind, f32_tensor(&gguf, &n("ffn_norm.weight"), dim)?)
                }
                Family::QwenHybrid if c.is_attention_layer(l) => {
                    let kind = LayerKind::Attn {
                        q: qtensor(&gguf, &n("attn_q.weight"), dim, attn_q_dim)?,
                        k: qtensor(&gguf, &n("attn_k.weight"), dim, kv_dim)?,
                        v: qtensor(&gguf, &n("attn_v.weight"), dim, kv_dim)?,
                        o: qtensor(&gguf, &n("attn_output.weight"), attn_o_in, dim)?,
                        q_norm: f32_tensor(&gguf, &n("attn_q_norm.weight"), head_dim)?,
                        k_norm: f32_tensor(&gguf, &n("attn_k_norm.weight"), head_dim)?,
                    };
                    (kind, f32_tensor(&gguf, &n("post_attention_norm.weight"), dim)?)
                }
                Family::QwenHybrid => {
                    // Delta layers exist only in hybrid families, so `ssm` is
                    // present here.
                    let s = c.ssm.expect("delta layer requires ssm config");
                    let (conv_dim, value_dim) = (s.conv_dim(), s.inner);
                    let kind = LayerKind::Delta {
                        qkv: qtensor(&gguf, &n("attn_qkv.weight"), dim, conv_dim)?,
                        gate: qtensor(&gguf, &n("attn_gate.weight"), dim, value_dim)?,
                        conv1d: f32_tensor(&gguf, &n("ssm_conv1d.weight"), conv_dim * s.conv_kernel)?,
                        dt_bias: f32_tensor(&gguf, &n("ssm_dt.bias"), s.dt_rank)?,
                        a_log: f32_tensor(&gguf, &n("ssm_a"), s.dt_rank)?,
                        alpha: qtensor(&gguf, &n("ssm_alpha.weight"), dim, s.dt_rank)?,
                        beta: qtensor(&gguf, &n("ssm_beta.weight"), dim, s.dt_rank)?,
                        norm: f32_tensor(&gguf, &n("ssm_norm.weight"), s.head_dim())?,
                        out: qtensor(&gguf, &n("ssm_out.weight"), value_dim, dim)?,
                    };
                    (kind, f32_tensor(&gguf, &n("post_attention_norm.weight"), dim)?)
                }
                Family::Lfm2 if c.is_attention_layer(l) => {
                    // Per-layer KV heads: 0 marks a recurrent layer, else the
                    // layer's own GQA count.
                    let n_kv = c
                        .kv_heads_per_layer
                        .get(l)
                        .copied()
                        .unwrap_or(c.head_count_kv as u32) as usize;
                    let kv_dim = n_kv * head_dim;
                    let kind = LayerKind::Lfm2Attn {
                        q: qtensor(&gguf, &n("attn_q.weight"), dim, c.head_count * head_dim)?,
                        k: qtensor(&gguf, &n("attn_k.weight"), dim, kv_dim)?,
                        v: qtensor(&gguf, &n("attn_v.weight"), dim, kv_dim)?,
                        o: qtensor(&gguf, &n("attn_output.weight"), c.head_count * head_dim, dim)?,
                        q_norm: f32_tensor(&gguf, &n("attn_q_norm.weight"), head_dim)?,
                        k_norm: f32_tensor(&gguf, &n("attn_k_norm.weight"), head_dim)?,
                        n_kv,
                    };
                    (kind, f32_tensor(&gguf, &n("ffn_norm.weight"), dim)?)
                }
                Family::Lfm2 => {
                    // Recurrent shortconv layer: the cached-state conv block.
                    let k = c.shortconv_l_cache;
                    let kind = LayerKind::ShortConv {
                        conv: f32_tensor(&gguf, &n("shortconv.conv.weight"), k * dim)?,
                        in_proj: qtensor(&gguf, &n("shortconv.in_proj.weight"), dim, 3 * dim)?,
                        out_proj: qtensor(&gguf, &n("shortconv.out_proj.weight"), dim, dim)?,
                    };
                    (kind, f32_tensor(&gguf, &n("ffn_norm.weight"), dim)?)
                }
            };
            // Gemma E-series per-layer-embedding block. All-or-nothing: a file
            // with only some of the three is malformed, and silently running
            // without the block is the failure this whole path exists to fix.
            let (pl_inp_gate, pl_proj, pl_post_norm) = if c.family == Family::Gemma4
                && c.swa.map(|s| s.n_embd_per_layer).unwrap_or(0) > 0
            {
                let e = c.swa.expect("gemma swa").n_embd_per_layer;
                (
                    Some(qtensor(&gguf, &n("inp_gate.weight"), dim, e)?),
                    Some(qtensor(&gguf, &n("proj.weight"), e, dim)?),
                    Some(f32_tensor(&gguf, &n("post_norm.weight"), dim)?),
                )
            } else {
                (None, None, None)
            };
            layers.push(Layer {
                attn_norm: f32_tensor(&gguf, &n("attn_norm.weight"), dim)?,
                post_norm,
                post_attn_norm,
                ffn_post_norm,
                out_scale,
                kind,
                ffn_gate: qtensor(&gguf, &n("ffn_gate.weight"), dim, ffn)?,
                ffn_up: qtensor(&gguf, &n("ffn_up.weight"), dim, ffn)?,
                ffn_down: qtensor(&gguf, &n("ffn_down.weight"), ffn, dim)?,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }
        // Model-level halves of the per-layer-embedding stack.
        let (pl_tok_embd, pl_model_proj, pl_proj_norm) =
            match c.swa.map(|s| s.n_embd_per_layer).unwrap_or(0) {
                0 => (None, None, None),
                e => {
                    let el = e * c.block_count;
                    (
                        Some(qtensor(&gguf, "per_layer_token_embd.weight", el, vocab)?),
                        Some(qtensor(&gguf, "per_layer_model_proj.weight", dim, el)?),
                        Some(f32_tensor(&gguf, "per_layer_proj_norm.weight", e)?),
                    )
                }
            };

        // Batching is decided **per tensor**, not per model. The old gate
        // required one uniform quant type across every weight and anchored it on
        // `token_embd.qt` — which is the worst possible anchor, because prefill
        // never batches `token_embd` at all (it is a row lookup via
        // `dequant_embed_row`, plus one `matvec_qw` for the final position when
        // the output is tied).
        //
        // Real GGUFs are almost never uniform. `llama-quantize` upcasts selected
        // tensors unless `--pure` is passed, so a file *labelled* Q4_0 arrives as
        // Q4_0 + Q8_0 + Q5_K + Q4_1 with `token_embd` at Q6_K — and the old gate
        // read that one Q6_K tensor and put the entire 4B model on the per-token
        // path, forfeiting batching for the 89% of its projection bytes that do
        // have a kernel. Now each projection takes the batched kernel when its own
        // type has one and falls back to per-position matvecs when it does not.
        let mut batchable = 0usize;
        let mut quantized = 0usize;
        {
            let mut acct = |w: QWeight| {
                quantized += w.data.len();
                if has_batched_kernel(w.qt) {
                    batchable += w.data.len();
                }
            };
            for ly in &layers {
                acct(ly.ffn_gate);
                acct(ly.ffn_up);
                acct(ly.ffn_down);
                match ly.kind {
                    LayerKind::Attn { q, k, v, o, .. } => {
                        acct(q);
                        acct(k);
                        acct(v);
                        acct(o);
                    }
                    LayerKind::Delta { qkv, gate, alpha, beta, out, .. } => {
                        acct(qkv);
                        acct(gate);
                        acct(alpha);
                        acct(beta);
                        acct(out);
                    }
                    // Gemma's projections go through the same `batched_proj`, so
                    // they count exactly like the hybrid's. `v` is absent on a
                    // global layer (V = K), hence the `Option`.
                    LayerKind::GemmaAttn { q, k, v, o, .. } => {
                        acct(q);
                        acct(k);
                        if let Some(v) = v {
                            acct(v);
                        }
                        acct(o);
                    }
                    LayerKind::Lfm2Attn { q, k, v, o, .. } => {
                        acct(q);
                        acct(k);
                        acct(v);
                        acct(o);
                    }
                    LayerKind::ShortConv { in_proj, out_proj, .. } => {
                        acct(in_proj);
                        acct(out_proj);
                    }
                }
            }
        }
        let batch_pct = if quantized == 0 { 0 } else { (batchable * 100 / quantized) as u8 };
        // The window-batched path is worth taking even at 0% batchable weights:
        // the attention core fans out over positions and the DeltaNet core over
        // heads (with the position loop inside), the conv ring advances once per
        // window instead of once per position, and the matvec work in the
        // fallback is exactly what the per-token path would have done anyway.
        let batched = cfg!(target_arch = "aarch64");

        Ok(Self {
            config: c.clone(),
            gguf,
            token_embd,
            output,
            output_norm,
            layers,
            vocab,
            pl_tok_embd,
            pl_model_proj,
            pl_proj_norm,
            batched,
            batch_pct,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// Whether [`Model::prefill`] takes the weight-stationary batched path on
    /// this model (uniform batch-capable quant type + an arch with the batched
    /// matmul kernels). Callers feeding long prompts token-by-token should
    /// instead feed chunks through `prefill` when this is true — each weight
    /// is then read once per chunk instead of once per token.
    pub fn batched_prefill_supported(&self) -> bool {
        self.batched
    }

    /// Percent of quantized projection bytes with a batched kernel — see
    /// [`Model::batch_pct`]'s field docs. 100 means every projection is
    /// weight-stationary; 0 means every one falls back to per-position matvecs
    /// (the window-parallel attention/DeltaNet cores still apply).
    pub fn batch_pct(&self) -> u8 {
        self.batch_pct
    }
    pub fn token_str(&self, id: usize) -> &str {
        self.gguf.tokens.get(id).copied().unwrap_or("")
    }
    /// Build the BPE text encoder from this model's vocab + merges (owns its
    /// maps, so it outlives borrows of the model). ~40 MiB / ~200 ms for the 9B.
    pub fn tokenizer(&self) -> crate::cortex::tokenizer::Tokenizer<'a> {
        crate::cortex::tokenizer::Tokenizer::build(&self.gguf)
    }
    /// The model's EOS token id (generation stops here).
    pub fn eos(&self) -> usize {
        self.config.eos_token_id as usize
    }
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.config)
    }
    pub fn new_state(&self) -> State {
        State::new(&self.config, self.vocab)
    }

    /// Prefill a prompt: run `prompt[i]` at position `pos0 + i` for every token,
    /// leaving the recurrent/KV cache advanced and `state.logits` holding the
    /// logits *after the last* prompt token (the only position whose logits are
    /// needed to pick the first generated token). The single entry point for
    /// prompt ingestion, so prefill optimizations land here.
    pub fn prefill(&self, prompt: &[usize], pos0: usize, cache: &mut Cache, state: &mut State) {
        // On aarch64, prefill a multi-token prompt with the weight-stationary
        // batched path (each weight read once for the whole prompt instead of
        // once per token). It is bit-identical to the sequential loop -- same
        // int8 quantization, same per-(row,position) SDOT, same sequential
        // cores -- just far less weight bandwidth. x86 (TCG, no batched matmul
        // kernel) uses the sequential path; prefill still works, just slower.
        #[cfg(target_arch = "aarch64")]
        if prompt.len() >= 2 && self.batched {
            self.prefill_batched(prompt, pos0, cache, state);
            return;
        }
        let last = prompt.len();
        for (i, &tok) in prompt.iter().enumerate() {
            self.forward(tok, pos0 + i, cache, state, i + 1 == last);
        }
    }

    /// Batched (weight-stationary) prefill: process all `M = prompt.len()`
    /// positions together so each projection weight is read once and applied to
    /// every position (`matmul_q8_0_sdot_rows`), while the order-dependent
    /// recurrence/attention still runs per position through the shared
    /// `attn_core`/`delta_core`. Only the last position needs logits.
    #[cfg(target_arch = "aarch64")]
    fn prefill_batched(&self, prompt: &[usize], pos0: usize, cache: &mut Cache, s: &mut State) {
        self.prefill_batched_inner(prompt, pos0, cache, s, None)
    }

    /// The batched prefill, optionally emitting logits for **every** position.
    ///
    /// Speculative decoding needs exactly this: the expensive part — all layers
    /// over all `m` positions — is already batched here, and the only thing done
    /// once is the final `vocab x dim` projection. Verifying a window of drafts
    /// is therefore one batched pass plus `m` projections, not `m` full forwards,
    /// which is where the entire speedup comes from.
    ///
    /// `all_logits` is filled with one vector per position when supplied. The
    /// last position is still written to `s.logits` either way, so every existing
    /// caller is bit-identical.
    #[cfg(target_arch = "aarch64")]
    fn prefill_batched_inner(
        &self,
        prompt: &[usize],
        pos0: usize,
        cache: &mut Cache,
        s: &mut State,
        mut all_logits: Option<&mut alloc::vec::Vec<alloc::vec::Vec<f32>>>,
    ) {
        let c = &self.config;
        let dim = c.embedding_length;
        let ffn = c.feed_forward_length;
        let hd = c.head_dim;
        let nq = c.head_count;
        let kv_dim = c.head_count_kv * hd;
        let qdim = nq * hd * 2; // query+gate interleaved
        let ao = nq * hd; // attn_out / o-proj input width
        let (conv_dim, value_dim, nh) =
            c.ssm.map(|s| (s.conv_dim(), s.inner, s.dt_rank)).unwrap_or((0, 0, 0));
        // LFM2 shortconv staging: the 3-way in_proj needs 3*dim, and the conv
        // output is dim-wide (reuses the DeltaNet qkv/delta_o slots).
        let lfm2_sc = c.family == Family::Lfm2;
        let qkv_w = conv_dim.max(if lfm2_sc { 3 * dim } else { 0 });
        let delta_w = value_dim.max(if lfm2_sc { dim } else { 0 });
        let sc_d = c.shortconv_l_cache.saturating_sub(1);
        let m = prompt.len();
        // Gemma has *two* attention geometries and every buffer must cover the
        // wider one — global layers carry `head_dim_global` (512 on E4B) against
        // the sliding layers' 256, so a buffer sized from `c.head_dim` alone
        // overflows on every global layer. Mirrors `State::new`.
        let hd_g = c.swa.map(|s| s.head_dim_global).unwrap_or(0);
        let kv_g = c.swa.map(|s| s.head_count_kv_global * s.head_dim_global).unwrap_or(0);
        let q_width = qdim.max(nq * hd_g);
        let kv_width = kv_dim.max(kv_g);
        let ao_width = ao.max(nq * hd_g);
        let max_cols = dim.max(ffn).max(ao_width).max(value_dim);

        // M-wide buffers (one prefill's worth; freed on return).
        let mut hidden = alloc::vec![0.0f32; m * dim];
        let mut norm = alloc::vec![0.0f32; m * dim];
        let mut q = alloc::vec![0.0f32; m * q_width];
        let mut k = alloc::vec![0.0f32; m * kv_width];
        let mut v = alloc::vec![0.0f32; m * kv_width];
        let mut attn_out = alloc::vec![0.0f32; m * ao_width];
        // One scores row per position, each `pos0 + mi + 1` long — the rows must
        // be separate because every position's attention runs concurrently.
        let mut scores = alloc::vec![0.0f32; m * (pos0 + m)];
        let mut qkv = alloc::vec![0.0f32; m * qkv_w];
        let mut conv_act = alloc::vec![0.0f32; m * conv_dim];
        let mut z = alloc::vec![0.0f32; m * value_dim];
        let mut gates = alloc::vec![0.0f32; m * nh];
        let mut betas = alloc::vec![0.0f32; m * nh];
        let mut delta_o = alloc::vec![0.0f32; m * delta_w];
        let mut conv_in = alloc::vec![0.0f32; (sc_d + m) * dim];
        let mut ffn_gate = alloc::vec![0.0f32; m * ffn];
        let mut ffn_up = alloc::vec![0.0f32; m * ffn];
        let mut ffn_act = alloc::vec![0.0f32; m * ffn];
        let mut proj_out = alloc::vec![0.0f32; m * dim];
        // Batched int8-activation scratch (packed tightly per projection's cols).
        let mut xq = alloc::vec![0i8; m * max_cols];
        let mut xs = alloc::vec![0.0f32; m * (max_cols / QK)];

        // Embeddings for all positions.
        for (mi, &tok) in prompt.iter().enumerate() {
            dequant_embed_row(self.token_embd, tok, &mut hidden[mi * dim..(mi + 1) * dim]);
        }
        // Gemma scales embeddings by sqrt(dim) — same as `forward`, applied to
        // every position in the window.
        if c.family == Family::Gemma4 {
            let es = tensor_sqrtf(dim as f32);
            hidden.iter_mut().for_each(|x| *x *= es);
        }
        // Gemma E-series per-layer inputs, one row of `E*n_layer` per position.
        // Derived from the *scaled* embedding, so this follows the scale above.
        let ple_e = c.swa.map(|w| w.n_embd_per_layer).unwrap_or(0);
        let el = ple_e * c.block_count;
        let mut ple_all = alloc::vec![0.0f32; m * el];
        let mut pl_gate_all = alloc::vec![0.0f32; m * ple_e];
        if el > 0 {
            // `hidden` is this call's own buffer, so its row can be handed over
            // directly — no per-position copy.
            let State { ple, ple_tok, xq: sxq, xs: sxs, .. } = &mut *s;
            for mi in 0..m {
                self.per_layer_inputs(
                    prompt[mi],
                    &hidden[mi * dim..(mi + 1) * dim],
                    ple,
                    ple_tok,
                    sxq,
                    sxs,
                );
                ple_all[mi * el..(mi + 1) * el].copy_from_slice(&ple[..el]);
            }
        }

        // Phase timing: see `prefill_phases`. `t` is advanced at every boundary
        // by `phase_mark`, so the four counters partition the loop exactly.
        let mut t = crate::arch::cycle_count();
        for l in 0..self.layers.len() {
            // rmsnorm(attn_norm) per position.
            for mi in 0..m {
                tensor::rmsnorm(&hidden[mi * dim..(mi + 1) * dim], self.layers[l].attn_norm, c.rms_eps, &mut norm[mi * dim..(mi + 1) * dim]);
            }
            t = phase_mark(&PHASE_ELEM, t);

            match &self.layers[l].kind {
                LayerKind::Attn { q: q_w, k: k_w, v: v_w, o: o_w, .. } => {
                    let (q_w, k_w, v_w, o_w) = (*q_w, *k_w, *v_w, *o_w);
                    self.batched_proj(q_w, &norm, &mut q, &mut xq, &mut xs, m, qdim, dim);
                    self.batched_proj(k_w, &norm, &mut k, &mut xq, &mut xs, m, kv_dim, dim);
                    self.batched_proj(v_w, &norm, &mut v, &mut xq, &mut xs, m, kv_dim, dim);
                    t = phase_mark(&PHASE_PROJ, t);
                    self.attn_core_batched(l, pos0, m, &mut q, &mut k, &v, cache, &mut scores, &mut attn_out);
                    t = phase_mark(&PHASE_ATTN, t);
                    self.batched_proj(o_w, &attn_out, &mut proj_out, &mut xq, &mut xs, m, dim, ao);
                }
                // Gemma: per-layer geometry (a sliding layer's `head_dim` is not
                // a global layer's), and V is the *pre-norm, pre-RoPE* K on
                // global layers, which is why the copy happens here rather than
                // inside the core — exactly the order `gemma_attn_layer` uses.
                LayerKind::GemmaAttn { q: q_w, k: k_w, v: v_w, o: o_w, n_kv, head_dim, .. } => {
                    let (q_w, k_w, v_w, o_w, n_kv, ghd) =
                        (*q_w, *k_w, *v_w, *o_w, *n_kv, *head_dim);
                    let gkv = n_kv * ghd;
                    let gao = nq * ghd;
                    self.batched_proj(q_w, &norm, &mut q, &mut xq, &mut xs, m, gao, dim);
                    // Shared-KV layer: Q only (see `GemmaAttn::kv_src`).
                    if c.owns_kv(l) {
                        self.batched_proj(k_w, &norm, &mut k, &mut xq, &mut xs, m, gkv, dim);
                        match v_w {
                            Some(vw) => self.batched_proj(vw, &norm, &mut v, &mut xq, &mut xs, m, gkv, dim),
                            None => v[..m * gkv].copy_from_slice(&k[..m * gkv]),
                        }
                    }
                    t = phase_mark(&PHASE_PROJ, t);
                    self.gemma_attn_core_batched(
                        l, pos0, m, &mut q, &mut k, &mut v, cache, &mut scores, &mut attn_out,
                    );
                    t = phase_mark(&PHASE_ATTN, t);
                    self.batched_proj(o_w, &attn_out, &mut proj_out, &mut xq, &mut xs, m, dim, gao);
                }
                LayerKind::Delta { qkv: qkv_w, gate: gate_w, alpha: alpha_w, beta: beta_w, out: out_w, .. } => {
                    let (qkv_w, gate_w, alpha_w, beta_w, out_w) = (*qkv_w, *gate_w, *alpha_w, *beta_w, *out_w);
                    self.batched_proj(qkv_w, &norm, &mut qkv, &mut xq, &mut xs, m, conv_dim, dim);
                    self.batched_proj(gate_w, &norm, &mut z, &mut xq, &mut xs, m, value_dim, dim);
                    self.batched_proj(alpha_w, &norm, &mut gates, &mut xq, &mut xs, m, nh, dim);
                    self.batched_proj(beta_w, &norm, &mut betas, &mut xq, &mut xs, m, nh, dim);
                    t = phase_mark(&PHASE_PROJ, t);
                    self.delta_core_batched(
                        l, cache, m, &qkv, &z, &mut gates, &mut betas, &mut conv_act, &mut delta_o,
                    );
                    t = phase_mark(&PHASE_DELTA, t);
                    self.batched_proj(out_w, &delta_o, &mut proj_out, &mut xq, &mut xs, m, dim, value_dim);
                }
                // LFM2: batched projections (Q4_0 → i8mm) + window-wide
                // attention / windowed shortconv cores, so prefill runs at
                // batched speed instead of one position at a time.
                LayerKind::Lfm2Attn { q: q_w, k: k_w, v: v_w, o: o_w, n_kv, .. } => {
                    let (q_w, k_w, v_w, o_w, n_kv) = (*q_w, *k_w, *v_w, *o_w, *n_kv);
                    let lkv = n_kv * hd;
                    let lq = nq * hd;
                    self.batched_proj(q_w, &norm, &mut q, &mut xq, &mut xs, m, lq, dim);
                    self.batched_proj(k_w, &norm, &mut k, &mut xq, &mut xs, m, lkv, dim);
                    self.batched_proj(v_w, &norm, &mut v, &mut xq, &mut xs, m, lkv, dim);
                    t = phase_mark(&PHASE_PROJ, t);
                    self.lfm2_attn_core_batched(l, pos0, m, &mut q, &mut k, &v, cache, &mut scores, &mut attn_out);
                    t = phase_mark(&PHASE_ATTN, t);
                    self.batched_proj(o_w, &attn_out, &mut proj_out, &mut xq, &mut xs, m, dim, lq);
                }
                LayerKind::ShortConv { in_proj, out_proj, .. } => {
                    let (in_proj, out_proj) = (*in_proj, *out_proj);
                    let plen = 3 * dim;
                    self.batched_proj(in_proj, &norm, &mut qkv, &mut xq, &mut xs, m, plen, dim);
                    t = phase_mark(&PHASE_PROJ, t);
                    self.shortconv_core_batched(l, cache, m, &qkv, &mut conv_in, &mut delta_o);
                    t = phase_mark(&PHASE_DELTA, t);
                    self.batched_proj(out_proj, &delta_o, &mut proj_out, &mut xq, &mut xs, m, dim, dim);
                }
            }
            t = phase_mark(&PHASE_PROJ, t);
            // Gemma sandwich: normalize the block output before its residual.
            if let Some(w) = self.layers[l].post_attn_norm {
                for mi in 0..m {
                    tensor::rmsnorm_inplace(&mut proj_out[mi * dim..(mi + 1) * dim], w, c.rms_eps);
                }
            }
            // Residual after the attn/delta block.
            for i in 0..m * dim {
                hidden[i] += proj_out[i];
            }

            // FFN (SwiGLU) with its own residual.
            for mi in 0..m {
                tensor::rmsnorm(&hidden[mi * dim..(mi + 1) * dim], self.layers[l].post_norm, c.rms_eps, &mut norm[mi * dim..(mi + 1) * dim]);
            }
            t = phase_mark(&PHASE_ELEM, t);
            self.batched_proj(self.layers[l].ffn_gate, &norm, &mut ffn_gate, &mut xq, &mut xs, m, ffn, dim);
            self.batched_proj(self.layers[l].ffn_up, &norm, &mut ffn_up, &mut xq, &mut xs, m, ffn, dim);
            t = phase_mark(&PHASE_PROJ, t);
            match c.family {
                // Gemma is GELU-gated, not SwiGLU. Same elementwise shape, so
                // the batched form is the per-position one over `m * ffn`.
                Family::Gemma4 => tensor::gelu_mul(&ffn_gate, &ffn_up, &mut ffn_act),
                _ => {
                    for i in 0..m * ffn {
                        ffn_act[i] = tensor::silu(ffn_gate[i]) * ffn_up[i];
                    }
                }
            }
            t = phase_mark(&PHASE_ELEM, t);
            self.batched_proj(self.layers[l].ffn_down, &ffn_act, &mut proj_out, &mut xq, &mut xs, m, dim, ffn);
            t = phase_mark(&PHASE_PROJ, t);
            if let Some(w) = self.layers[l].ffn_post_norm {
                for mi in 0..m {
                    tensor::rmsnorm_inplace(&mut proj_out[mi * dim..(mi + 1) * dim], w, c.rms_eps);
                }
            }
            for i in 0..m * dim {
                hidden[i] += proj_out[i];
            }
            // Gemma E-series per-layer-embedding block, per position. Its two
            // matrices are F32/BF16, neither of which has a batched kernel, so a
            // position-at-a-time loop costs exactly what `batched_proj` would
            // have fallen back to anyway.
            if el > 0 {
                if let (Some(gw), Some(pw), Some(pn)) = (
                    self.layers[l].pl_inp_gate,
                    self.layers[l].pl_proj,
                    self.layers[l].pl_post_norm,
                ) {
                    // Both projections go through `batched_proj` like every other
                    // one: weight-stationary when the type has a kernel, a matvec
                    // per position when it does not. They are F32/BF16 on the
                    // published E4B (so today this is the fallback), but a
                    // requantized file makes them batchable and a hand-rolled
                    // per-position loop here would silently forfeit that.
                    self.batched_proj(gw, &hidden, &mut pl_gate_all, &mut xq, &mut xs, m, ple_e, dim);
                    for mi in 0..m {
                        let (go, po) = (mi * ple_e, mi * el + l * ple_e);
                        for j in 0..ple_e {
                            pl_gate_all[go + j] =
                                tensor::gelu(pl_gate_all[go + j]) * ple_all[po + j];
                        }
                    }
                    // `proj_out` was just folded into `hidden`, so it is free and
                    // doubles as this block's output scratch.
                    self.batched_proj(pw, &pl_gate_all, &mut proj_out, &mut xq, &mut xs, m, dim, ple_e);
                    for mi in 0..m {
                        tensor::rmsnorm_inplace(&mut proj_out[mi * dim..(mi + 1) * dim], pn, c.rms_eps);
                    }
                    for i in 0..m * dim {
                        hidden[i] += proj_out[i];
                    }
                }
            }
            t = phase_mark(&PHASE_ELEM, t);
            // Gemma per-layer output scalar (scales the whole stream).
            if self.layers[l].out_scale != 1.0 {
                let os = self.layers[l].out_scale;
                hidden.iter_mut().for_each(|x| *x *= os);
            }
            t = phase_mark(&PHASE_ELEM, t);
        }

        let out_w = self.output.unwrap_or(self.token_embd);

        // Speculative verification wants a distribution at every position, not
        // just the last. Done before the last-position projection below so the
        // existing path is untouched when nobody asked.
        if let Some(sink) = all_logits.as_deref_mut() {
            sink.clear();
            for mi in 0..m {
                tensor::rmsnorm(&hidden[mi * dim..(mi + 1) * dim], self.output_norm, c.rms_eps, &mut s.norm);
                matvec_qw(out_w, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
                self.logit_tail(c, s);
                sink.push(s.logits.clone());
            }
        }

        // Only the final position's logits are needed to pick the first token.
        let last = m - 1;
        tensor::rmsnorm(&hidden[last * dim..(last + 1) * dim], self.output_norm, c.rms_eps, &mut s.norm);
        matvec_qw(out_w, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
        // Gemma final-logit softcap + suppressed-token bias — the same tail
        // `forward` applies. Omitting it here would make the first sampled token
        // after a batched prefill come from an uncapped distribution while every
        // later one is capped.
        if let Some(swa) = c.swa {
            if swa.logit_softcap != 0.0 {
                let cap = swa.logit_softcap;
                s.logits.iter_mut().for_each(|x| *x = cap * tensor::tanhf(*x / cap));
            }
        }
        for &id in &self.gguf.suppress_tokens {
            if let Some(lg) = s.logits.get_mut(id as usize) {
                *lg = f32::NEG_INFINITY;
            }
        }
        cache.positions = cache.positions.max(pos0 + m);
    }

    /// **Speculative verification.** Run `tokens` through the model starting at
    /// `pos0` and return the next-token logits at **every** position.
    ///
    /// `tokens` is the accepted prefix's last token followed by the γ drafted
    /// ones, so the returned `tokens.len()` distributions line up as: index `i`
    /// is the target's opinion of what should follow `tokens[i]`. That makes
    /// index 0 the check on the first draft and the final index the free bonus
    /// position.
    ///
    /// **The cache is advanced by the whole window**, including positions whose
    /// drafts are about to be rejected. The caller must snapshot the cache
    /// before calling and restore it on a partial accept — see
    /// [`Cache::clone`]. That is not an optimisation detail: a DeltaNet layer's
    /// recurrent state is *stepped*, not appended, so it cannot be rewound by
    /// truncating anything. Skipping the snapshot leaves the model conditioned
    /// on tokens it never emitted, and the output stays fluent while silently
    /// diverging from what unassisted decode would have produced.
    ///
    /// On aarch64 this is one batched pass plus `m` output projections. On x86
    /// there is no batched matmul kernel, so it degrades to `m` sequential
    /// forwards and speculation is a **net loss** — `spec::Stats::speedup_estimate`
    /// will report below 1.0, and the caller should leave it off there.
    pub fn verify_window(
        &self,
        tokens: &[usize],
        pos0: usize,
        cache: &mut Cache,
        s: &mut State,
    ) -> alloc::vec::Vec<alloc::vec::Vec<f32>> {
        let mut out = alloc::vec::Vec::with_capacity(tokens.len());
        if tokens.is_empty() {
            return out;
        }
        #[cfg(target_arch = "aarch64")]
        if tokens.len() >= 2 && self.batched {
            self.prefill_batched_inner(tokens, pos0, cache, s, Some(&mut out));
            return out;
        }
        // Portable fallback: correct, but one pass per position.
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward(tok, pos0 + i, cache, s, true);
            out.push(s.logits.clone());
        }
        out
    }

    /// Gemma final-logit softcap + suppressed-token bias, applied to
    /// `s.logits`. Shared so a per-position projection cannot drift from the
    /// last-position one.
    #[cfg(target_arch = "aarch64")]
    fn logit_tail(&self, c: &Config, s: &mut State) {
        if let Some(swa) = c.swa {
            if swa.logit_softcap != 0.0 {
                let cap = swa.logit_softcap;
                s.logits.iter_mut().for_each(|x| *x = cap * tensor::tanhf(*x / cap));
            }
        }
        for &id in &self.gguf.suppress_tokens {
            if let Some(lg) = s.logits.get_mut(id as usize) {
                *lg = f32::NEG_INFINITY;
            }
        }
    }

    /// Weight-stationary batched projection: quantize each of the `m` input
    /// vectors (`input[mi*cols..]`) to int8, then run one matmul that reads the
    /// weight once and writes `out[mi*rows + r]`. `xq`/`xs` are packed tightly
    /// with stride `cols`/`cols/QK` (the layout `matmul_q8_0_sdot_rows` expects).
    ///
    /// **Per tensor, not per model.** A quant type with no batched kernel is
    /// applied a position at a time instead of disqualifying the whole model —
    /// see the note in [`Model::load`]. The fallback is bit-identical to the
    /// per-token path (same `matvec_qw`, same order), and `matvec_qw` still
    /// splits its rows across the fleet, so the only thing lost on those tensors
    /// is the weight-read amortization.
    #[cfg(target_arch = "aarch64")]
    fn batched_proj(&self, w: QWeight, input: &[f32], out: &mut [f32], xq: &mut [i8], xs: &mut [f32], m: usize, rows: usize, cols: usize) {
        if !has_batched_kernel(w.qt) {
            // Row-major `out` means position `mi`'s outputs are the contiguous
            // run `out[mi*rows .. (mi+1)*rows]` — the same layout a matvec
            // writes, so this needs no gather.
            for mi in 0..m {
                matvec_qw(
                    w,
                    &input[mi * cols..(mi + 1) * cols],
                    &mut out[mi * rows..(mi + 1) * rows],
                    xq,
                    xs,
                    rows,
                    cols,
                );
            }
            return;
        }
        let nb = cols / QK;
        for mi in 0..m {
            tensor::quantize_activations_q8(
                &input[mi * cols..(mi + 1) * cols],
                &mut xq[mi * cols..(mi + 1) * cols],
                &mut xs[mi * nb..(mi + 1) * nb],
            );
        }
        // SAFETY: `w` is `rows` rows of `w.qt` blocks; `xq`/`xs` hold `m`
        // activations of `cols`/`nb`; `out` has `m*rows` slots. Each matmul
        // splits `[0,rows)` across cores, each writing a disjoint row range.
        unsafe {
            match w.qt {
                // Q1_0: the i8mm (vmmlaq_s32) 2×2-tile GEMM does 2× the MAC/instr
                // of SDOT — used for real batches (m≥2) when the CPU has
                // FEAT_I8MM; else the SDOT matmul. (m==1 has no 2nd column for a
                // 2×2 tile, so i8mm gives nothing there — keep SDOT.)
                tensor::QT_Q1_0 if m >= 2 && crate::arch::has_i8mm() => crate::arch::aarch64::smp::matmul_q1_0_i8mm(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                tensor::QT_Q1_0 => crate::arch::aarch64::smp::matmul_q1_0_sdot(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                tensor::QT_Q2_0 => crate::arch::aarch64::smp::matmul_q2_0_sdot(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                // Q8_0: weights are already int8, so i8mm needs no unpack — the
                // cleanest 2×2 tile. Used for real batches (m≥2) with FEAT_I8MM.
                tensor::QT_Q8_0 if m >= 2 && crate::arch::has_i8mm() => crate::arch::aarch64::smp::matmul_q8_0_i8mm(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                // Q4_0's only batched kernel is the i8mm one, and
                // `has_batched_kernel` reports Q4_0 batchable *only* when the CPU
                // has FEAT_I8MM — so reaching here means m == 1 (a one-position
                // window), which the matvec below handles.
                tensor::QT_Q4_0 if m >= 2 => crate::arch::aarch64::smp::matmul_q4_0_i8mm(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                // Q8_0 without i8mm: the SDOT batched matmul. Named explicitly —
                // this arm used to be a catch-all `_`, which would have silently
                // read *any* quant type's bytes as Q8_0 once batching stopped
                // requiring a uniform model. Anything else cannot reach here
                // (`has_batched_kernel` gated it into the fallback above), so an
                // unreachable is the honest arm rather than a wrong answer.
                tensor::QT_Q8_0 => crate::arch::aarch64::smp::matmul_sdot(
                    w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols,
                ),
                // m == 1 for a type whose batched kernel needs a column pair.
                _ => {
                    for mi in 0..m {
                        matvec_qw(
                            w,
                            &input[mi * cols..(mi + 1) * cols],
                            &mut out[mi * rows..(mi + 1) * rows],
                            xq,
                            xs,
                            rows,
                            cols,
                        );
                    }
                }
            }
        }
    }

    /// One decoder step for `token` at position `pos` (== `cache.len()`).
    /// When `need_logits`, writes next-token logits into `state.logits`;
    /// otherwise skips the final norm + output projection. The output
    /// projection is ~254M MACs (the vocab-sized matmul dominates a step),
    /// and prefill only needs the *last* prompt position's logits, so
    /// skipping it on the others speeds prefill roughly in proportion to
    /// the prompt length -- for free, since those logits are discarded.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut Cache, s: &mut State, need_logits: bool) {
        let c = &self.config;
        let dim = c.embedding_length;

        dequant_embed_row(self.token_embd, token, &mut s.hidden);
        // Gemma scales embeddings by sqrt(dim); skipped entirely elsewhere so
        // the hybrid path's math is untouched.
        if c.family == Family::Gemma4 {
            let es = tensor_sqrtf(dim as f32);
            s.hidden.iter_mut().for_each(|x| *x *= es);
        }
        // Per-layer inputs are derived from the *scaled* embedding, so this must
        // follow the scale above and precede the layer loop.
        if !s.ple.is_empty() {
            // Disjoint field borrows — no clone of the hidden row.
            let State { hidden, ple, ple_tok, xq, xs, .. } = s;
            self.per_layer_inputs(token, hidden, ple, ple_tok, xq, xs);
        }

        for l in 0..self.layers.len() {
            let ly = &self.layers[l];
            s.residual.copy_from_slice(&s.hidden);
            tensor::rmsnorm(&s.hidden, ly.attn_norm, c.rms_eps, &mut s.norm);
            match &ly.kind {
                LayerKind::Attn { .. } => self.attn_layer(l, pos, cache, s),
                LayerKind::Delta { .. } => self.delta_layer(l, cache, s),
                LayerKind::GemmaAttn { .. } => self.gemma_attn_layer(l, pos, cache, s),
                LayerKind::Lfm2Attn { .. } => self.lfm2_attn_layer(l, pos, cache, s),
                LayerKind::ShortConv { .. } => self.shortconv_layer(l, cache, s),
            }
            // Gemma sandwich: normalize the block output before its residual.
            if let Some(w) = ly.post_attn_norm {
                tensor::rmsnorm_inplace(&mut s.proj, w, c.rms_eps);
            }
            // attn residual: hidden = residual + proj
            for i in 0..dim {
                s.hidden[i] = s.residual[i] + s.proj[i];
            }
            // FFN block (SwiGLU / GELU-par) with its own residual.
            s.residual.copy_from_slice(&s.hidden);
            tensor::rmsnorm(&s.hidden, ly.post_norm, c.rms_eps, &mut s.norm);
            let ffn = c.feed_forward_length;
            matvec_qw(ly.ffn_gate, &s.norm, &mut s.ffn_gate, &mut s.xq, &mut s.xs, ffn, dim);
            matvec_qw(ly.ffn_up, &s.norm, &mut s.ffn_up, &mut s.xq, &mut s.xs, ffn, dim);
            match c.family {
                Family::Gemma4 => tensor::gelu_mul(&s.ffn_gate, &s.ffn_up, &mut s.ffn_act),
                _ => tensor::silu_mul(&s.ffn_gate, &s.ffn_up, &mut s.ffn_act),
            }
            matvec_qw(ly.ffn_down, &s.ffn_act, &mut s.proj, &mut s.xq, &mut s.xs, dim, ffn);
            if let Some(w) = ly.ffn_post_norm {
                tensor::rmsnorm_inplace(&mut s.proj, w, c.rms_eps);
            }
            for i in 0..dim {
                s.hidden[i] = s.residual[i] + s.proj[i];
            }
            // Gemma E-series per-layer-embedding block: after the FFN residual,
            // before the layer scalar (llama.cpp `gemma4.cpp` order).
            if !s.ple.is_empty() {
                let e = s.pl_gate.len();
                // Disjoint field borrows: `ple` is read, the rest are scratch.
                let State { hidden, ple, pl_gate, proj, xq, xs, .. } = s;
                self.per_layer_block(l, hidden, &ple[l * e..(l + 1) * e], pl_gate, proj, xq, xs);
            }
            // Gemma per-layer output scalar (scales the whole stream).
            if ly.out_scale != 1.0 {
                let os = ly.out_scale;
                s.hidden.iter_mut().for_each(|x| *x *= os);
            }
        }

        if need_logits {
            tensor::rmsnorm(&s.hidden, self.output_norm, c.rms_eps, &mut s.norm);
            // Untied output projection if present (9B), else the tied embeddings.
            let out_w = self.output.unwrap_or(self.token_embd);
            matvec_qw(out_w, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
            // Gemma final-logit softcap + suppressed-token bias.
            if let Some(swa) = c.swa {
                if swa.logit_softcap != 0.0 {
                    let cap = swa.logit_softcap;
                    s.logits.iter_mut().for_each(|x| *x = cap * tensor::tanhf(*x / cap));
                }
            }
            for &id in &self.gguf.suppress_tokens {
                if let Some(lg) = s.logits.get_mut(id as usize) {
                    *lg = f32::NEG_INFINITY;
                }
            }
        }

        cache.positions = cache.positions.max(pos + 1);
    }

    fn gemma_attn_layer(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;
        let (q_w, k_w, v_w, o_w, n_kv, hd) = match &self.layers[l].kind {
            LayerKind::GemmaAttn { q, k, v, o, n_kv, head_dim, .. } => (*q, *k, *v, *o, *n_kv, *head_dim),
            _ => unreachable!(),
        };
        let nq = c.head_count;
        let kv_dim = n_kv * hd;

        // Exact-length slices: the scratch is sized for the widest geometry
        // (`State::new`), and the int8-quantize path requires x.len == n_cols.
        matvec_qw(q_w, &s.norm, &mut s.q[..nq * hd], &mut s.xq, &mut s.xs, nq * hd, dim);
        // A shared-KV layer projects **Q only** and attends over `kv_src`'s
        // cache; its own k/v tensors are unused (llama.cpp: TENSOR_NOT_REQUIRED).
        if c.owns_kv(l) {
            matvec_qw(k_w, &s.norm, &mut s.k[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim);
            match v_w {
                Some(v) => matvec_qw(v, &s.norm, &mut s.v[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim),
                // Global layers have no V projection: V = K (pre-norm/rope copy).
                None => {
                    let (k, v) = (&s.k[..kv_dim], &mut s.v[..kv_dim]);
                    v.copy_from_slice(k);
                }
            }
        }

        self.gemma_attn_core(l, pos, cache, s);

        matvec_qw(o_w, &s.attn_out[..nq * hd], &mut s.proj, &mut s.xq, &mut s.xs, dim, nq * hd);
    }

    /// The position-sequential core of a Gemma-4 attention layer (llama.cpp
    /// `gemma4.cpp`): per-head weighted QK-norms + an *unweighted* RMS on V,
    /// full-dim NeoX RoPE (per-layer base; p-RoPE freq divisors on global
    /// layers), KV append (sliding layers write a W-slot ring), causal
    /// attention at scale 1.0 over the layer's window.
    fn gemma_attn_core(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let (q_norm, k_norm, n_kv, hd, window, rope_base, freq_factors, kv_src) = match &self.layers[l].kind {
            LayerKind::GemmaAttn { q_norm, k_norm, n_kv, head_dim, window, rope_base, freq_factors, kv_src, .. } => {
                (*q_norm, *k_norm, *n_kv, *head_dim, *window, *rope_base, *freq_factors, *kv_src)
            }
            _ => unreachable!(),
        };
        let nq = c.head_count;
        let kv_dim = n_kv * hd;
        let group = nq / n_kv;
        let owns = c.owns_kv(l);

        // Per-head norms + full-dim RoPE. V gets a bare (unweighted) RMS norm.
        for h in 0..nq {
            let q = &mut s.q[h * hd..(h + 1) * hd];
            tensor::rmsnorm_inplace(q, q_norm, c.rms_eps);
            tensor::rope_ext(q, pos, hd, rope_base, freq_factors);
        }
        if owns {
            for h in 0..n_kv {
                let k = &mut s.k[h * hd..(h + 1) * hd];
                tensor::rmsnorm_inplace(k, k_norm, c.rms_eps);
                let v = &mut s.v[h * hd..(h + 1) * hd];
                rms_scale(v, c.rms_eps);
                tensor::rope_ext(k, pos, hd, rope_base, freq_factors);
            }
        }

        // KV store: sliding layers keep a ring (keys roped at absolute position
        // before insertion, so eviction never re-ropes); global layers append
        // full history. A shared-KV layer stores nothing — it only reads.
        let slots = c.ring_slots();
        let (t_lo, ring) = match window {
            Some(w) => {
                if owns {
                    let slot = (pos % slots) * kv_dim;
                    cache.attn_k[l][slot..slot + kv_dim].copy_from_slice(&s.k[..kv_dim]);
                    cache.attn_v[l][slot..slot + kv_dim].copy_from_slice(&s.v[..kv_dim]);
                }
                ((pos + 1).saturating_sub(w), Some(slots))
            }
            None => {
                if owns {
                    cache.attn_k[l].extend_from_slice(&s.k[..kv_dim]);
                    cache.attn_v[l].extend_from_slice(&s.v[..kv_dim]);
                }
                (0, None)
            }
        };
        // Read side: this layer's own cache, or its source's when shared. Named
        // separately from `l` so a later edit cannot confuse "the layer I am"
        // with "the layer whose history I read".
        let src = kv_src;

        // Causal attention, scale 1.0 (Gemma-4: the QK-norms replace 1/sqrt(d)).
        for h in 0..nq {
            let kvh = h / group;
            let q_head = &s.q[h * hd..(h + 1) * hd];
            for t in t_lo..=pos {
                let off = match ring {
                    Some(w) => (t % w) * kv_dim + kvh * hd,
                    None => t * kv_dim + kvh * hd,
                };
                s.scores[t - t_lo] = tensor::dot_f32(q_head, &cache.attn_k[src][off..off + hd]);
            }
            tensor::softmax(&mut s.scores[0..=pos - t_lo]);
            let out = &mut s.attn_out[h * hd..(h + 1) * hd];
            out.iter_mut().for_each(|x| *x = 0.0);
            for t in t_lo..=pos {
                let off = match ring {
                    Some(w) => (t % w) * kv_dim + kvh * hd,
                    None => t * kv_dim + kvh * hd,
                };
                let v_t = &cache.attn_v[src][off..off + hd];
                let w = s.scores[t - t_lo];
                for i in 0..hd {
                    out[i] += w * v_t[i];
                }
            }
        }
    }

    fn attn_layer(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;
        let hd = c.head_dim;
        let nq = c.head_count;
        let nkv = c.head_count_kv;
        let kv_dim = nkv * hd;
        let (q_w, k_w, v_w, o_w) = match &self.layers[l].kind {
            LayerKind::Attn { q, k, v, o, .. } => (*q, *k, *v, *o),
            _ => unreachable!(),
        };

        // Projections into s.q (query+gate interleaved), s.k, s.v.
        // Exact-length slices: the scratch is sized for the widest geometry
        // (`State::new`), and the int8-quantize path requires x.len == n_cols.
        matvec_qw(q_w, &s.norm, &mut s.q[..nq * hd * 2], &mut s.xq, &mut s.xs, nq * hd * 2, dim);
        matvec_qw(k_w, &s.norm, &mut s.k[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim);
        matvec_qw(v_w, &s.norm, &mut s.v[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim);

        // Sequential (recurrent/causal) core: consumes s.q/s.k/s.v, writes s.attn_out.
        self.attn_core(l, pos, cache, s);

        matvec_qw(o_w, &s.attn_out[..nq * hd], &mut s.proj, &mut s.xq, &mut s.xs, dim, nq * hd);
    }

    /// The position-sequential part of an attention layer for a single position
    /// — decode's entry point. A one-position window through
    /// [`Model::attn_core_batched`], so decode and prefill run the same code and
    /// cannot drift.
    fn attn_core(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        self.attn_core_batched(l, pos, 1, &mut s.q, &mut s.k, &s.v, cache, &mut s.scores, &mut s.attn_out);
    }

    /// LFM2 attention (llama.cpp `lfm2.cpp` `build_attn_block`): RMS QK-norm,
    /// full per-head RoPE, plain GQA causal attention at `1/sqrt(head_dim)`,
    /// out projection — no gate (unlike QwenHybrid). Decode-position core.
    fn lfm2_attn_layer(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;
        let hd = c.head_dim;
        let nq = c.head_count;
        let (q_w, k_w, v_w, o_w, q_norm, k_norm, n_kv) = match &self.layers[l].kind {
            LayerKind::Lfm2Attn { q, k, v, o, q_norm, k_norm, n_kv } => (*q, *k, *v, *o, *q_norm, *k_norm, *n_kv),
            _ => unreachable!(),
        };
        let kv_dim = n_kv * hd;
        let scale = 1.0 / tensor_sqrtf(hd as f32);

        matvec_qw(q_w, &s.norm, &mut s.q[..nq * hd], &mut s.xq, &mut s.xs, nq * hd, dim);
        matvec_qw(k_w, &s.norm, &mut s.k[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim);
        matvec_qw(v_w, &s.norm, &mut s.v[..kv_dim], &mut s.xq, &mut s.xs, kv_dim, dim);

        // Per-head QK norms + full RoPE (V is left bare, as llama.cpp does).
        for h in 0..nq {
            let qh = &mut s.q[h * hd..(h + 1) * hd];
            tensor::rmsnorm_inplace(qh, q_norm, c.rms_eps);
            tensor::rope_ext(qh, pos, hd, c.rope_freq_base, None);
        }
        for h in 0..n_kv {
            let kh = &mut s.k[h * hd..(h + 1) * hd];
            tensor::rmsnorm_inplace(kh, k_norm, c.rms_eps);
            tensor::rope_ext(kh, pos, hd, c.rope_freq_base, None);
        }

        cache.attn_k[l].extend_from_slice(&s.k[..kv_dim]);
        cache.attn_v[l].extend_from_slice(&s.v[..kv_dim]);
        let group = nq / n_kv;
        let o = &mut s.attn_out[..nq * hd];
        o.iter_mut().for_each(|x| *x = 0.0);
        for h in 0..nq {
            let kvh = h / group;
            let qh = &s.q[h * hd..(h + 1) * hd];
            let sc = &mut s.scores[..pos + 1];
            for t in 0..=pos {
                let base = t * kv_dim + kvh * hd;
                let kt = &cache.attn_k[l][base..base + hd];
                sc[t] = tensor::dot_f32(qh, kt) * scale;
            }
            tensor::softmax(sc);
            for t in 0..=pos {
                let base = t * kv_dim + kvh * hd;
                let vt = &cache.attn_v[l][base..base + hd];
                let w = sc[t];
                for i in 0..hd {
                    o[h * hd + i] += w * vt[i];
                }
            }
        }
        matvec_qw(o_w, &s.attn_out[..nq * hd], &mut s.proj, &mut s.xq, &mut s.xs, dim, nq * hd);
    }

    /// LFM2 recurrent **shortconv** block (llama.cpp `lfm2.cpp`
    /// `build_shortconv_block`): in_proj → split `b`/`c`/`x` → `bx = b·x` →
    /// causal depthwise conv1d over the cached per-layer state (d_conv
    /// columns, prepended causally) → `y = c·conv(bx)` → out_proj. Decode is
    /// one token per call, so the conv consumes exactly state ++ [current].
    /// Window-wide LFM2 attention: the batched twin of
    /// [`Self::lfm2_attn_layer`], fanning out over **positions** (each
    /// `(position, head)` is independent once the window's K/V exist). No
    /// gate — unlike QwenHybrid — so `q` rows are `nq*head_dim` wide.
    fn lfm2_attn_core_batched(
        &self,
        l: usize,
        pos0: usize,
        m: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        cache: &mut Cache,
        scores: &mut [f32],
        out: &mut [f32],
    ) {
        let c = &self.config;
        let hd = c.head_dim;
        let nq = c.head_count;
        let (q_norm, k_norm, n_kv) = match &self.layers[l].kind {
            LayerKind::Lfm2Attn { q_norm, k_norm, n_kv, .. } => (*q_norm, *k_norm, *n_kv),
            _ => unreachable!(),
        };
        let kv_dim = n_kv * hd;
        let qdim = nq * hd;
        let ao = nq * hd;
        let group = nq / n_kv;
        let scale = 1.0 / tensor_sqrtf(hd as f32);

        // Per-head QK norms + full RoPE (V left bare), order-independent per
        // absolute position.
        for mi in 0..m {
            let pos = pos0 + mi;
            for h in 0..nq {
                let qh = &mut q[mi * qdim + h * hd..mi * qdim + (h + 1) * hd];
                tensor::rmsnorm_inplace(qh, q_norm, c.rms_eps);
                tensor::rope_ext(qh, pos, hd, c.rope_freq_base, None);
            }
            for h in 0..n_kv {
                let kh = &mut k[mi * kv_dim + h * hd..mi * kv_dim + (h + 1) * hd];
                tensor::rmsnorm_inplace(kh, k_norm, c.rms_eps);
                tensor::rope_ext(kh, pos, hd, c.rope_freq_base, None);
            }
        }

        cache.attn_k[l].extend_from_slice(&k[..m * kv_dim]);
        cache.attn_v[l].extend_from_slice(&v[..m * kv_dim]);

        struct Ctx {
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            scores: *mut f32,
            out: *mut f32,
            pos0: usize,
            nq: usize,
            hd: usize,
            kv_dim: usize,
            group: usize,
            qdim: usize,
            ao: usize,
            sl: usize,
            scale: f32,
        }
        unsafe fn positions(start: usize, end: usize, ctx: *mut u8) {
            let c = unsafe { &*(ctx as *const Ctx) };
            for mi in start..end {
                let pos = c.pos0 + mi;
                let sc = unsafe { core::slice::from_raw_parts_mut(c.scores.add(mi * c.sl), pos + 1) };
                for h in 0..c.nq {
                    let kvh = h / c.group;
                    let q_head =
                        unsafe { core::slice::from_raw_parts(c.q.add(mi * c.qdim + h * c.hd), c.hd) };
                    for t in 0..=pos {
                        let k_t =
                            unsafe { core::slice::from_raw_parts(c.kc.add(t * c.kv_dim + kvh * c.hd), c.hd) };
                        sc[t] = tensor::dot_f32(q_head, k_t) * c.scale;
                    }
                    tensor::softmax(sc);
                    let o = unsafe { core::slice::from_raw_parts_mut(c.out.add(mi * c.ao + h * c.hd), c.hd) };
                    o.iter_mut().for_each(|x| *x = 0.0);
                    for t in 0..=pos {
                        let v_t =
                            unsafe { core::slice::from_raw_parts(c.vc.add(t * c.kv_dim + kvh * c.hd), c.hd) };
                        let w = sc[t];
                        for i in 0..c.hd {
                            o[i] += w * v_t[i];
                        }
                    }
                }
            }
        }
        let mut ctx = Ctx {
            q: q.as_ptr(),
            kc: cache.attn_k[l].as_ptr(),
            vc: cache.attn_v[l].as_ptr(),
            scores: scores.as_mut_ptr(),
            out: out.as_mut_ptr(),
            pos0,
            nq,
            hd,
            kv_dim,
            group,
            qdim,
            ao,
            sl: pos0 + m,
            scale,
        };
        // SAFETY: `positions` is safe on disjoint position ranges sharing `ctx`,
        // and `ctx` lives until `parallel_for` returns.
        unsafe { crate::arch::parallel_for(m, 1, positions, &mut ctx as *mut Ctx as *mut u8) };
    }

    /// Window-wide LFM2 **shortconv**: the batched twin of
    /// [`Self::shortconv_layer`]. The window's `b·x` product is padded with the
    /// cached per-layer state columns, run through the causal depthwise conv,
    /// gated by `c`, and the state advanced — one pass over the whole window.
    fn shortconv_core_batched(
        &self,
        l: usize,
        cache: &mut Cache,
        m: usize,
        qkv: &[f32],
        conv_in: &mut [f32],
        out: &mut [f32],
    ) {
        let c = &self.config;
        let dim = c.embedding_length;
        let k = c.shortconv_l_cache;
        let d = k.saturating_sub(1);
        let conv = match &self.layers[l].kind {
            LayerKind::ShortConv { conv, .. } => *conv,
            _ => unreachable!(),
        };
        let state = &mut cache.shortconv_state[l];
        // bx = b·x per position; the window is padded with the cached state.
        for mi in 0..m {
            let base = mi * 3 * dim;
            for i in 0..dim {
                conv_in[(d + mi) * dim + i] = qkv[base + i] * qkv[base + 2 * dim + i];
            }
        }
        for j in 0..d {
            for i in 0..dim {
                conv_in[j * dim + i] = state[j * dim + i];
            }
        }
        // Causal depthwise conv: position `mi` reads taps `conv_in[mi..mi+k]`.
        for mi in 0..m {
            let base = mi * 3 * dim;
            for i in 0..dim {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += conv[i * k + j] * conv_in[(mi + j) * dim + i];
                }
                // y = c · conv_out.
                out[mi * dim + i] = qkv[base + dim + i] * acc;
            }
        }
        // New state = last `d` columns of the padded window.
        for j in 0..d {
            for i in 0..dim {
                state[j * dim + i] = conv_in[(m + j) * dim + i];
            }
        }
    }

    fn shortconv_layer(&self, l: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;        let (conv, in_proj, out_proj) = match &self.layers[l].kind {
            LayerKind::ShortConv { conv, in_proj, out_proj } => (*conv, *in_proj, *out_proj),
            _ => unreachable!(),
        };
        let k = c.shortconv_l_cache; // kernel width
        let d = k.saturating_sub(1); // state columns
        let proj_len = 3 * dim;

        // 3-way projection: [b | c | x], each `dim` wide.
        matvec_qw(in_proj, &s.norm, &mut s.qkv[..proj_len], &mut s.xq, &mut s.xs, proj_len, dim);
        let (b, cg, x) = (&s.qkv[..dim], &s.qkv[dim..2 * dim], &s.qkv[2 * dim..]);
        let state = &mut cache.shortconv_state[l];
        // bx = b · x, staged in `betas`.
        for i in 0..dim {
            s.betas[i] = b[i] * x[i];
        }
        let bx = &s.betas[..dim];
        // Causal depthwise conv: window[j] = state[i*d+j] for j<d, then bx[i].
        for i in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += conv[i * k + j] * state[i * d + j];
            }
            acc += conv[i * k + d] * bx[i];
            s.delta_o[i] = acc;
        }
        // y = c · conv_out, then out_proj.
        for i in 0..dim {
            s.delta_o[i] *= cg[i];
        }
        matvec_qw(out_proj, &s.delta_o[..dim], &mut s.proj, &mut s.xq, &mut s.xs, dim, dim);
        // Update the state: shift each channel column left, append bx.
        for i in 0..dim {
            for j in 0..d.saturating_sub(1) {
                state[i * d + j] = state[i * d + j + 1];
            }
            if d > 0 {
                state[i * d + d - 1] = bx[i];
            }
        }
    }

    /// The "position-sequential" part of an attention layer over a whole
    /// `m`-position prefill window: QK-norm + partial RoPE, append K/V to this
    /// layer's history, GQA causal attention, per-head sigmoid gate. Reads the
    /// batched projections in `q`/`k`/`v` (row stride `qdim`/`kv_dim`); writes
    /// `out` (row stride `n_head*head_dim`).
    ///
    /// **It is not actually sequential.** Attention at position `mi` reads the
    /// K/V prefix `0..=pos0+mi`, all of which is known once the batched K/V
    /// projections have been written — so every `(position, head)` pair is
    /// independent and the window fans out across the fleet. Doing it a position
    /// at a time (as this did) left the whole causal core on one core while the
    /// projections used all of them, which is most of why prefill throughput sat
    /// near decode throughput instead of an order above it.
    ///
    /// The arithmetic per `(position, head)` is unchanged and in the same order,
    /// so the result is bit-identical to the per-position loop — the reference
    /// fixtures (`cargo xtask ref-check`) hold it to that.
    #[allow(clippy::too_many_arguments)]
    fn attn_core_batched(
        &self,
        l: usize,
        pos0: usize,
        m: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        cache: &mut Cache,
        scores: &mut [f32],
        out: &mut [f32],
    ) {
        let c = &self.config;
        let hd = c.head_dim;
        let nq = c.head_count;
        let nkv = c.head_count_kv;
        let kv_dim = nkv * hd;
        let qdim = nq * hd * 2; // query+gate interleaved
        let ao = nq * hd;
        let group = nq / nkv;
        let scale = 1.0 / tensor_sqrtf(hd as f32);
        let (q_norm, k_norm) = match &self.layers[l].kind {
            LayerKind::Attn { q_norm, k_norm, .. } => (*q_norm, *k_norm),
            _ => unreachable!(),
        };

        // QK-norm per head, partial RoPE on the first rope_dim dims. Position
        // `mi`'s rotation depends only on its own absolute position, so this is
        // order-independent even though it is written as a loop.
        for mi in 0..m {
            let pos = pos0 + mi;
            for h in 0..nq {
                let qh = &mut q[mi * qdim + h * 2 * hd..mi * qdim + h * 2 * hd + hd];
                tensor::rmsnorm_inplace(qh, q_norm, c.rms_eps);
                tensor::rope(&mut qh[0..c.rope_dim], pos, c.rope_dim, c.rope_freq_base);
            }
            for h in 0..nkv {
                let kh = &mut k[mi * kv_dim + h * hd..mi * kv_dim + (h + 1) * hd];
                tensor::rmsnorm_inplace(kh, k_norm, c.rms_eps);
                tensor::rope(&mut kh[0..c.rope_dim], pos, c.rope_dim, c.rope_freq_base);
            }
        }

        // Append the window's K/V in one extend: the batch buffers already carry
        // the cache's `[pos][kv_dim]` layout, so this is one copy, not `m`.
        cache.attn_k[l].extend_from_slice(&k[..m * kv_dim]);
        cache.attn_v[l].extend_from_slice(&v[..m * kv_dim]);

        /// Shared, immutable view for the fan-out below. Raw pointers because
        /// `parallel_for` takes a plain `fn` — every worker reads `q`/`kc`/`vc`
        /// and writes only its own positions' `scores`/`out` rows.
        struct Ctx {
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            scores: *mut f32,
            out: *mut f32,
            pos0: usize,
            nq: usize,
            hd: usize,
            kv_dim: usize,
            group: usize,
            qdim: usize,
            ao: usize,
            sl: usize,
            scale: f32,
        }
        /// # Safety
        /// `ctx` is the live `Ctx` published below and `[start, end)` is a range
        /// of positions disjoint from every other worker's, so the `scores` and
        /// `out` rows written here are touched by no one else.
        unsafe fn positions(start: usize, end: usize, ctx: *mut u8) {
            // SAFETY: the caller passes the `Ctx` published below, which outlives
            // the `parallel_for` call.
            let c = unsafe { &*(ctx as *const Ctx) };
            for mi in start..end {
                let pos = c.pos0 + mi;
                // SAFETY: this worker owns position `mi`, and `sl >= pos + 1`.
                let sc = unsafe { core::slice::from_raw_parts_mut(c.scores.add(mi * c.sl), pos + 1) };
                for h in 0..c.nq {
                    let kvh = h / c.group;
                    // SAFETY: `q` holds `m` rows of `qdim`; head `h`'s query half
                    // is `hd` wide at `h * 2 * hd`.
                    let q_head =
                        unsafe { core::slice::from_raw_parts(c.q.add(mi * c.qdim + h * 2 * c.hd), c.hd) };
                    for t in 0..=pos {
                        // SAFETY: the cache holds at least `pos + 1` rows of `kv_dim`.
                        let k_t =
                            unsafe { core::slice::from_raw_parts(c.kc.add(t * c.kv_dim + kvh * c.hd), c.hd) };
                        sc[t] = tensor::dot_f32(q_head, k_t) * c.scale;
                    }
                    tensor::softmax(sc);
                    // SAFETY: `out` holds `m` rows of `ao`; this worker owns row `mi`.
                    let o = unsafe { core::slice::from_raw_parts_mut(c.out.add(mi * c.ao + h * c.hd), c.hd) };
                    o.iter_mut().for_each(|x| *x = 0.0);
                    for t in 0..=pos {
                        // SAFETY: as `k_t` above.
                        let v_t =
                            unsafe { core::slice::from_raw_parts(c.vc.add(t * c.kv_dim + kvh * c.hd), c.hd) };
                        let w = sc[t];
                        for i in 0..c.hd {
                            o[i] += w * v_t[i];
                        }
                    }
                    // gate is the second half of head h's slice in `q`
                    // SAFETY: the gate half sits `hd` past the query half.
                    let gate = unsafe {
                        core::slice::from_raw_parts(c.q.add(mi * c.qdim + h * 2 * c.hd + c.hd), c.hd)
                    };
                    for i in 0..c.hd {
                        o[i] *= tensor::sigmoid(gate[i]);
                    }
                }
            }
        }
        let mut ctx = Ctx {
            q: q.as_ptr(),
            kc: cache.attn_k[l].as_ptr(),
            vc: cache.attn_v[l].as_ptr(),
            scores: scores.as_mut_ptr(),
            out: out.as_mut_ptr(),
            pos0,
            nq,
            hd,
            kv_dim,
            group,
            qdim,
            ao,
            sl: pos0 + m,
            scale,
        };
        // One position is a real unit of work here (`nq * pos` dot products), so
        // a chunk of 1 is worth handing out; `parallel_for` still runs a whole
        // window inline when it is too small to split (decode's `m == 1`). The
        // static split is by position count, not by cost — work grows with
        // `pos0 + mi`, so the split is slightly uneven on the very first chunk
        // (`pos0 == 0`, where attention is cheapest anyway) and within a few
        // percent once `pos0 >> m`.
        // SAFETY: `positions` is safe on disjoint position ranges sharing `ctx`,
        // and `ctx` lives until `parallel_for` returns.
        unsafe { crate::arch::parallel_for(m, 1, positions, &mut ctx as *mut Ctx as *mut u8) };
    }

    /// Window-wide Gemma-4 attention: the batched twin of [`Self::gemma_attn_core`],
    /// fanning out over **positions** (each `(position, head)` is independent once
    /// the window's K/V exist). Decode enters it as a one-position window so the
    /// two paths cannot drift.
    ///
    /// **The sliding ring forces read-before-commit.** A local layer keeps a fixed
    /// `W`-slot ring, so slot `t % W` is shared by `t` and `t - W` — and `t - W`
    /// is *inside* the window of the earliest position in this chunk for any
    /// `m >= 2`. Committing the window's K/V up front (what the Qwen path does,
    /// safely, because its cache is append-only) would therefore overwrite history
    /// that positions in this very chunk still have to attend to, and the damage
    /// grows with chunk size — fluent output that quietly loses the oldest `m`
    /// tokens of context. So attention reads history from the **cache** for
    /// `t < pos0` and the in-flight window from the **chunk buffers** for
    /// `t >= pos0`, and the ring is written only afterwards.
    #[cfg(target_arch = "aarch64")]
    fn gemma_attn_core_batched(
        &self,
        l: usize,
        pos0: usize,
        m: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        cache: &mut Cache,
        scores: &mut [f32],
        out: &mut [f32],
    ) {
        let c = &self.config;
        let (q_norm, k_norm, n_kv, hd, window, rope_base, freq_factors, kv_src) = match &self.layers[l].kind {
            LayerKind::GemmaAttn { q_norm, k_norm, n_kv, head_dim, window, rope_base, freq_factors, kv_src, .. } => {
                (*q_norm, *k_norm, *n_kv, *head_dim, *window, *rope_base, *freq_factors, *kv_src)
            }
            _ => unreachable!(),
        };
        let nq = c.head_count;
        let kv_dim = n_kv * hd;
        let ao = nq * hd;
        let group = nq / n_kv;
        let owns = c.owns_kv(l);

        // Per-head norms + full-dim RoPE, per position. Order-independent: a
        // position's rotation depends only on its own absolute index. V takes a
        // bare (unweighted) RMS and is never roped.
        for mi in 0..m {
            let pos = pos0 + mi;
            for h in 0..nq {
                let qh = &mut q[mi * ao + h * hd..mi * ao + (h + 1) * hd];
                tensor::rmsnorm_inplace(qh, q_norm, c.rms_eps);
                tensor::rope_ext(qh, pos, hd, rope_base, freq_factors);
            }
            if owns {
                for h in 0..n_kv {
                    let kh = &mut k[mi * kv_dim + h * hd..mi * kv_dim + (h + 1) * hd];
                    tensor::rmsnorm_inplace(kh, k_norm, c.rms_eps);
                    let vh = &mut v[mi * kv_dim + h * hd..mi * kv_dim + (h + 1) * hd];
                    rms_scale(vh, c.rms_eps);
                    tensor::rope_ext(kh, pos, hd, rope_base, freq_factors);
                }
            }
        }

        /// Shared, immutable view for the fan-out. `kc`/`vc` are the committed
        /// cache (history only, `t < pos0`); `kb`/`vb` are this window's rows.
        struct Ctx {
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            kb: *const f32,
            vb: *const f32,
            scores: *mut f32,
            out: *mut f32,
            pos0: usize,
            nq: usize,
            hd: usize,
            kv_dim: usize,
            group: usize,
            ao: usize,
            sl: usize,
            /// Attention span of a sliding layer, 0 for a global layer. Distinct
            /// from `slots`: this bounds *what is attended to*, `slots` is only
            /// the ring's modulus.
            win: usize,
            /// Ring modulus for a sliding layer, 0 for a full-history global
            /// layer (whose cache is indexed by absolute `t`).
            slots: usize,
            /// Whether this layer computed the window's K/V itself. False on a
            /// shared-KV layer, whose chunk buffers are empty — every `t` must
            /// then come from `kv_src`'s cache, which its source already
            /// committed.
            own_window: bool,
        }
        /// # Safety
        /// `ctx` is the live `Ctx` published below; `[start, end)` is a range of
        /// positions disjoint from every other worker's, so the `scores`/`out`
        /// rows written here are touched by no one else.
        unsafe fn positions(start: usize, end: usize, ctx: *mut u8) {
            // SAFETY: the caller passes the `Ctx` published below.
            let c = unsafe { &*(ctx as *const Ctx) };
            // Offset of time `t`'s K/V row, and which buffer it lives in.
            let row = |t: usize, kvh: usize| -> (bool, usize) {
                if c.own_window && t >= c.pos0 {
                    (true, (t - c.pos0) * c.kv_dim + kvh * c.hd)
                } else if c.slots != 0 {
                    (false, (t % c.slots) * c.kv_dim + kvh * c.hd)
                } else {
                    (false, t * c.kv_dim + kvh * c.hd)
                }
            };
            for mi in start..end {
                let pos = c.pos0 + mi;
                // A sliding layer attends only over its window.
                let t_lo = if c.win != 0 { (pos + 1).saturating_sub(c.win) } else { 0 };
                let n = pos + 1 - t_lo;
                // SAFETY: this worker owns position `mi`, and `sl >= n`.
                let sc = unsafe { core::slice::from_raw_parts_mut(c.scores.add(mi * c.sl), n) };
                for h in 0..c.nq {
                    let kvh = h / c.group;
                    // SAFETY: `q` holds `m` rows of `ao`; head `h` is `hd` wide.
                    let q_head =
                        unsafe { core::slice::from_raw_parts(c.q.add(mi * c.ao + h * c.hd), c.hd) };
                    for t in t_lo..=pos {
                        let (in_win, off) = row(t, kvh);
                        let base = if in_win { c.kb } else { c.kc };
                        // SAFETY: `off` is in range for the buffer `in_win` picked
                        // — the window holds `m` rows, the cache at least `pos0`.
                        let k_t = unsafe { core::slice::from_raw_parts(base.add(off), c.hd) };
                        sc[t - t_lo] = tensor::dot_f32(q_head, k_t);
                    }
                    // Scale 1.0: Gemma-4's QK-norms replace the 1/sqrt(d).
                    tensor::softmax(sc);
                    // SAFETY: this worker owns position `mi`'s `out` row.
                    let o = unsafe {
                        core::slice::from_raw_parts_mut(c.out.add(mi * c.ao + h * c.hd), c.hd)
                    };
                    o.iter_mut().for_each(|x| *x = 0.0);
                    for t in t_lo..=pos {
                        let (in_win, off) = row(t, kvh);
                        let base = if in_win { c.vb } else { c.vc };
                        // SAFETY: as above, for V.
                        let v_t = unsafe { core::slice::from_raw_parts(base.add(off), c.hd) };
                        let w = sc[t - t_lo];
                        for i in 0..c.hd {
                            o[i] += w * v_t[i];
                        }
                    }
                }
            }
        }

        let mut ctx = Ctx {
            q: q.as_ptr(),
            // Read side is `kv_src`: itself normally, an earlier layer when shared.
            kc: cache.attn_k[kv_src].as_ptr(),
            vc: cache.attn_v[kv_src].as_ptr(),
            kb: k.as_ptr(),
            vb: v.as_ptr(),
            scores: scores.as_mut_ptr(),
            out: out.as_mut_ptr(),
            pos0,
            nq,
            hd,
            kv_dim,
            group,
            ao,
            sl: pos0 + m,
            win: window.unwrap_or(0),
            slots: if window.is_some() { c.ring_slots() } else { 0 },
            own_window: owns,
        };
        // SAFETY: `positions` is safe on disjoint position ranges sharing `ctx`,
        // which lives until `parallel_for` returns.
        unsafe { crate::arch::parallel_for(m, 1, positions, &mut ctx as *mut Ctx as *mut u8) };

        // Commit the window now that every position has read the history it
        // needed (see the read-before-commit note above). A shared-KV layer has
        // nothing to commit.
        if !owns {
            return;
        }
        match window {
            Some(_) => {
                let w = c.ring_slots();
                for mi in 0..m {
                    let slot = ((pos0 + mi) % w) * kv_dim;
                    cache.attn_k[l][slot..slot + kv_dim]
                        .copy_from_slice(&k[mi * kv_dim..(mi + 1) * kv_dim]);
                    cache.attn_v[l][slot..slot + kv_dim]
                        .copy_from_slice(&v[mi * kv_dim..(mi + 1) * kv_dim]);
                }
            }
            None => {
                cache.attn_k[l].extend_from_slice(&k[..m * kv_dim]);
                cache.attn_v[l].extend_from_slice(&v[..m * kv_dim]);
            }
        }
    }

    /// Gemma E-series **per-layer inputs** for one token (llama.cpp
    /// `build_inp_per_layer` + `project_per_layer_inputs`): writes `E*n_layer`
    /// floats, layer `il`'s slice being `out[il*E..(il+1)*E]`.
    ///
    /// Two terms are summed and scaled by `1/sqrt(2)`: the token's row of a second,
    /// per-layer embedding table (scaled by `sqrt(E)`), and a projection of the
    /// *already sqrt(dim)-scaled* input embedding (scaled by `1/sqrt(dim)`, then
    /// RMS-normed per layer). The `1/sqrt(2)` is the reference's
    /// `per_layer_input_scale` — it is what makes the sum an average rather than
    /// a doubling, and dropping it silently doubles every layer's injected
    /// signal.
    /// Takes explicit slices rather than `&mut State` so the caller can pass its
    /// own hidden row as `scaled_embd` without cloning it, and so the `E*n_layer`
    /// scratch is preallocated instead of heap-allocated per token (43 KiB per
    /// token on the E4B — see the allocator-churn rule in CLAUDE.md).
    fn per_layer_inputs(
        &self,
        token: usize,
        scaled_embd: &[f32],
        ple: &mut [f32],
        ple_tok: &mut [f32],
        xq: &mut [i8],
        xs: &mut [f32],
    ) {
        let c = &self.config;
        let e = match c.swa.map(|w| w.n_embd_per_layer).unwrap_or(0) {
            0 => return,
            e => e,
        };
        let (tok_w, proj_w, norm_w) = match (self.pl_tok_embd, self.pl_model_proj, self.pl_proj_norm) {
            (Some(a), Some(b), Some(n)) => (a, b, n),
            _ => return,
        };
        let el = e * c.block_count;
        // Term 1: the per-layer embedding row for this token.
        let tok_row = &mut ple_tok[..el];
        dequant_embed_row(tok_w, token, tok_row);
        let ts = tensor_sqrtf(e as f32);
        tok_row.iter_mut().for_each(|x| *x *= ts);
        // Term 2: project the scaled input embedding, then RMS-norm per layer.
        let out = &mut ple[..el];
        matvec_qw(proj_w, scaled_embd, out, xq, xs, el, c.embedding_length);
        let ps = 1.0 / tensor_sqrtf(c.embedding_length as f32);
        out.iter_mut().for_each(|x| *x *= ps);
        for il in 0..c.block_count {
            tensor::rmsnorm_inplace(&mut out[il * e..(il + 1) * e], norm_w, c.rms_eps);
        }
        // Combine. `1/sqrt(2)`, not `1/2` — the reference scales the sum, and the
        // two terms are not independent draws.
        let is = 1.0 / tensor_sqrtf(2.0);
        for (o, p) in out.iter_mut().zip(tok_row.iter()) {
            *o = (*o + *p) * is;
        }
    }

    /// The per-layer-embedding block for layer `l`, applied to `x` in place
    /// after the FFN residual (and before the layer output scalar):
    /// `x += pl_post_norm(pl_proj @ (gelu(pl_inp_gate @ x) * ple_l))`.
    /// A no-op on models without the stack.
    fn per_layer_block(&self, l: usize, x: &mut [f32], ple_l: &[f32], gate: &mut [f32], proj: &mut [f32], xq: &mut [i8], xs: &mut [f32]) {
        let c = &self.config;
        let (gate_w, proj_w, post) = match (
            self.layers[l].pl_inp_gate,
            self.layers[l].pl_proj,
            self.layers[l].pl_post_norm,
        ) {
            (Some(g), Some(p), Some(n)) => (g, p, n),
            _ => return,
        };
        let e = ple_l.len();
        let dim = c.embedding_length;
        let g = &mut gate[..e];
        matvec_qw(gate_w, x, g, xq, xs, e, dim);
        for (v, p) in g.iter_mut().zip(ple_l.iter()) {
            *v = tensor::gelu(*v) * *p;
        }
        let o = &mut proj[..dim];
        matvec_qw(proj_w, g, o, xq, xs, dim, e);
        tensor::rmsnorm_inplace(o, post, c.rms_eps);
        for i in 0..dim {
            x[i] += o[i];
        }
    }

    fn delta_layer(&self, l: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;
        let ssm = c.ssm.expect("delta layer requires ssm config");
        let conv_dim = ssm.conv_dim();
        let value_dim = ssm.inner;
        // Value heads. NOT `n_group` — they are equal on the 0.8B/2B but the 4B
        // has 32 value heads over 16 key/query groups (see `delta_core_batched`).
        let nh = ssm.dt_rank;
        let (qkv_w, gate_w, alpha_w, beta_w, out_w) = match &self.layers[l].kind {
            LayerKind::Delta { qkv, gate, alpha, beta, out, .. } => (*qkv, *gate, *alpha, *beta, *out),
            _ => unreachable!(),
        };

        matvec_qw(qkv_w, &s.norm, &mut s.qkv, &mut s.xq, &mut s.xs, conv_dim, dim);
        matvec_qw(gate_w, &s.norm, &mut s.z, &mut s.xq, &mut s.xs, value_dim, dim);
        matvec_qw(alpha_w, &s.norm, &mut s.gates, &mut s.xq, &mut s.xs, nh, dim); // reuse gates as alpha
        matvec_qw(beta_w, &s.norm, &mut s.betas, &mut s.xq, &mut s.xs, nh, dim);

        // Sequential (recurrent) core: consumes s.qkv/s.z/s.gates/s.betas, writes s.delta_o.
        self.delta_core(l, cache, s);

        matvec_qw(out_w, &s.delta_o, &mut s.proj, &mut s.xq, &mut s.xs, dim, value_dim);
    }

    /// One position through the DeltaNet recurrence — decode's entry point. A
    /// one-position window through [`Model::delta_core_batched`], so decode and
    /// prefill run the same code and cannot drift.
    fn delta_core(&self, l: usize, cache: &mut Cache, s: &mut State) {
        self.delta_core_batched(
            l,
            cache,
            1,
            &s.qkv,
            &s.z,
            &mut s.gates,
            &mut s.betas,
            &mut s.conv,
            &mut s.delta_o,
        );
    }

    /// The recurrent part of a DeltaNet layer over an `m`-position prefill
    /// window: gate/beta activation, causal conv1d, and the gated delta rule
    /// that advances the recurrent state. Reads the batched projections in
    /// `qkv`/`z`/`gates` (=alpha)/`betas`; writes `out` (row stride `inner`).
    /// `conv` is `m * conv_dim` of scratch.
    ///
    /// The recurrence is sequential **in position** but completely independent
    /// **per head** — head `h` owns its own `hk×hv` state slice and never reads
    /// another's. So the parallel axis is the head, with the position loop
    /// *inside* it: one `parallel_for` per layer rather than one per position,
    /// which is what makes the fan-out worth its barrier (a 64-token window is
    /// 64 barriers per layer the other way round). Everything before it — the
    /// activations and the conv — is position-independent.
    ///
    /// Two things the per-position version paid for that the window does not:
    /// the conv ring was **shifted down a slot per position** (`(ck-1)*conv_dim`
    /// floats — 72 KiB a token on the 0.8B, per layer), and each head
    /// re-allocated its `q`/`k` copies on the heap every position. The window
    /// reads its history straight out of the batch and updates the ring once,
    /// and the per-head copies live on the stack.
    #[allow(clippy::too_many_arguments)]
    fn delta_core_batched(
        &self,
        l: usize,
        cache: &mut Cache,
        m: usize,
        qkv: &[f32],
        z: &[f32],
        gates: &mut [f32],
        betas: &mut [f32],
        conv: &mut [f32],
        out: &mut [f32],
    ) {
        let c = &self.config;
        let ssm = c.ssm.expect("delta core requires ssm config");
        let conv_dim = ssm.conv_dim();
        let value_dim = ssm.inner;
        let nh = ssm.dt_rank; // value heads
        let hk = ssm.state;
        let hv = ssm.head_dim();
        let key_dim = hk * ssm.n_group;
        // GQA over the recurrent heads: `n_group` key/query heads, `nh` value
        // heads. llama.cpp maps them with `ggml_repeat` (tiling), i.e. value
        // head h uses key head `h % n_group` -- NOT repeat-interleave. (When
        // n_group == nh, e.g. the 0.8B, both are the identity.)
        let ck = ssm.conv_kernel;
        let (conv1d, dt_bias, a_log, norm_w) = match &self.layers[l].kind {
            LayerKind::Delta { conv1d, dt_bias, a_log, norm, .. } => (*conv1d, *dt_bias, *a_log, *norm),
            _ => unreachable!(),
        };

        // g = -exp(A_log) * softplus(alpha + dt_bias)  (log-decay); beta = sigmoid.
        for mi in 0..m {
            for h in 0..nh {
                let a = -tensor::expf(a_log[h]);
                let g = a * tensor::softplus(gates[mi * nh + h] + dt_bias[h]);
                gates[mi * nh + h] = tensor::expf(g); // decay multiplier in (0,1)
                betas[mi * nh + h] = tensor::sigmoid(betas[mi * nh + h]);
            }
        }

        // Causal depthwise conv1d + SiLU over the window. Tap `j` of position
        // `mi` is the qkv vector at `p = mi + j - (ck-1)`: inside the window for
        // `p >= 0`, else out of the carried-in ring. The ring runs slot 0 oldest
        // .. ck-1 newest, and entering the window slot `x` holds `qkv(x - ck)` —
        // so a negative `p` is slot `ck + p`, **not** `ck - 1 + p`. (The
        // per-position form shifted the ring down a slot *before* convolving,
        // which is where that extra `-1` hides; getting it wrong reads each tap
        // one position too old and still produces plausible text.) Same `j`
        // order as the per-position form, so the sum is bit-identical.
        {
            /// Shared view for the channel fan-out. Each worker writes only its
            /// own channels of `out` and reads `qkv`/`ring`/`w`.
            struct ConvCtx {
                qkv: *const f32,
                ring: *const f32,
                w: *const f32,
                out: *mut f32,
                m: usize,
                conv_dim: usize,
                ck: usize,
            }
            /// # Safety
            /// `ctx` is the live `ConvCtx` published below; `[start, end)` is a
            /// channel range disjoint from every other worker's.
            unsafe fn channels(start: usize, end: usize, ctx: *mut u8) {
                // SAFETY: the caller passes the `ConvCtx` published below.
                let c = unsafe { &*(ctx as *const ConvCtx) };
                for mi in 0..c.m {
                    for ci in start..end {
                        let mut acc = 0.0f32;
                        for j in 0..c.ck {
                            // SAFETY: `conv_tap` returns a window row `< m` or a
                            // ring slot in `1..ck`, both in bounds.
                            let x = unsafe {
                                match conv_tap(mi, j, c.ck) {
                                    ConvTap::Window(p) => *c.qkv.add(p * c.conv_dim + ci),
                                    ConvTap::Ring(x) => *c.ring.add(x * c.conv_dim + ci),
                                }
                            };
                            // SAFETY: `w` is `conv_dim` rows of `ck` taps.
                            acc += x * unsafe { *c.w.add(ci * c.ck + j) };
                        }
                        // SAFETY: this worker owns channel `ci` of every row.
                        unsafe { *c.out.add(mi * c.conv_dim + ci) = tensor::silu(acc) };
                    }
                }
            }
            let mut ctx = ConvCtx {
                qkv: qkv.as_ptr(),
                ring: cache.conv[l].as_ptr(),
                w: conv1d.as_ptr(),
                out: conv.as_mut_ptr(),
                m,
                conv_dim,
                ck,
            };
            // SAFETY: `channels` is safe on disjoint channel ranges sharing
            // `ctx`, and `ctx` lives until `parallel_for` returns.
            unsafe {
                crate::arch::parallel_for(
                    conv_dim,
                    fanout_chunk(m * conv_dim * ck),
                    channels,
                    &mut ctx as *mut ConvCtx as *mut u8,
                )
            };
        }

        // Advance the ring past the window, once instead of per position.
        advance_conv_ring(&mut cache.conv[l], qkv, m, conv_dim, ck);

        // Recurrent gated delta rule: sequential in position, independent per
        // head, so the fan-out is over heads and each worker walks the window.
        let scale = 1.0 / tensor_sqrtf(hk as f32);
        debug_assert!(hv <= 256, "DeltaNet head_v dim {hv} exceeds the fixed delta scratch");
        debug_assert!(hk <= 256, "DeltaNet head_k dim {hk} exceeds the fixed delta scratch");
        /// Shared view for the head fan-out: head `h` owns state slice `h` and
        /// the `h`-th span of every output row, so no two workers overlap.
        struct DeltaCtx {
            conv: *const f32,
            z: *const f32,
            gates: *const f32,
            betas: *const f32,
            state: *mut f32,
            out: *mut f32,
            norm_w: *const f32,
            m: usize,
            nh: usize,
            hk: usize,
            hv: usize,
            n_group: usize,
            key_dim: usize,
            conv_dim: usize,
            value_dim: usize,
            scale: f32,
            rms_eps: f32,
        }
        /// # Safety
        /// `ctx` is the live `DeltaCtx` published below; `[start, end)` is a head
        /// range disjoint from every other worker's.
        unsafe fn heads(start: usize, end: usize, ctx: *mut u8) {
            // SAFETY: the caller passes the `DeltaCtx` published below.
            let c = unsafe { &*(ctx as *const DeltaCtx) };
            // Per-head scratch, reused across the window (the per-position form
            // heap-allocated these; hk/hv <= 256 is asserted by the caller).
            let mut qbuf = [0.0f32; 256];
            let mut kbuf = [0.0f32; 256];
            let mut dbuf = [0.0f32; 256];
            // SAFETY: `norm_w` is the layer's `hv`-wide DeltaNet norm weight.
            let norm_w = unsafe { core::slice::from_raw_parts(c.norm_w, c.hv) };
            for h in start..end {
                // Key/query head for value head `h`: tiling (`ggml_repeat`), per
                // llama.cpp. NB the two candidate mappings are indistinguishable
                // whenever `n_group == nh` (the 0.8B and the 2B), and the 4B
                // (n_group 16, dt_rank 32 — ratio 2) is the first model that can
                // tell them apart: on it, repeat-interleave (`h / g_rep`, what
                // Qwen3-Next's own Python does) decodes pure gibberish while this
                // decodes fluent-but-degenerating text. So the mapping is not the
                // whole story — see the 4B long-context divergence vs llama.cpp.
                let g = h % c.n_group;
                // SAFETY: this worker owns head `h`'s `hk*hv` state slice.
                let sh = unsafe { core::slice::from_raw_parts_mut(c.state.add(h * c.hk * c.hv), c.hk * c.hv) };
                for mi in 0..c.m {
                    // SAFETY: `conv` holds `m` rows of `conv_dim`.
                    let cv = unsafe { core::slice::from_raw_parts(c.conv.add(mi * c.conv_dim), c.conv_dim) };
                    let q = &mut qbuf[..c.hk];
                    q.copy_from_slice(&cv[g * c.hk..(g + 1) * c.hk]);
                    let k = &mut kbuf[..c.hk];
                    k.copy_from_slice(&cv[c.key_dim + g * c.hk..c.key_dim + (g + 1) * c.hk]);
                    let vv = &cv[2 * c.key_dim + h * c.hv..2 * c.key_dim + (h + 1) * c.hv];
                    tensor::l2norm(q, 1e-6);
                    tensor::l2norm(k, 1e-6);
                    for x in q.iter_mut() {
                        *x *= c.scale;
                    }
                    // SAFETY: `gates`/`betas` hold `m` rows of `nh`.
                    let gd = unsafe { *c.gates.add(mi * c.nh + h) };
                    let beta = unsafe { *c.betas.add(mi * c.nh + h) };
                    tensor::scale_f32(sh, gd);
                    // SAFETY: `out` holds `m` rows of `value_dim`; head `h`'s span
                    // within a row belongs to this worker alone.
                    let o = unsafe {
                        core::slice::from_raw_parts_mut(c.out.add(mi * c.value_dim + h * c.hv), c.hv)
                    };
                    // kv_mem[v] = sum_k S[k,v]*k[k]; delta = (v - kv_mem)*beta;
                    // S[k,v] += k[k]*delta[v]; o[v] = sum_k S[k,v]*q[k]
                    //
                    // Iterated **row-wise** (ki outer): each S row `sh[ki*hv..]`
                    // is a contiguous [hv] slice, so every pass is a SIMD AXPY
                    // instead of a stride-hv scalar walk — this delta rule
                    // (18 layers × 16 heads × 128×128 state, every token) was
                    // the decode-time hot spot.
                    let delta = &mut dbuf[..c.hv];
                    delta.copy_from_slice(vv);
                    for ki in 0..c.hk {
                        // delta[v] -= k[ki] * S[ki, v]  (accumulating -kv_mem into vv)
                        tensor::axpy_f32(delta, &sh[ki * c.hv..(ki + 1) * c.hv], -k[ki]);
                    }
                    for d in delta.iter_mut() {
                        *d *= beta;
                    }
                    o.fill(0.0);
                    for ki in 0..c.hk {
                        let row = &mut sh[ki * c.hv..(ki + 1) * c.hv];
                        tensor::axpy_f32(row, delta, k[ki]);
                        tensor::axpy_f32(o, row, q[ki]);
                    }
                    // gated RMSNorm: RMSNorm(out over hv) * SiLU(z_head)
                    tensor::rmsnorm_inplace(o, norm_w, c.rms_eps);
                    for vi in 0..c.hv {
                        // SAFETY: `z` holds `m` rows of `value_dim`.
                        o[vi] *= tensor::silu(unsafe { *c.z.add(mi * c.value_dim + h * c.hv + vi) });
                    }
                }
            }
        }
        let mut ctx = DeltaCtx {
            conv: conv.as_ptr(),
            z: z.as_ptr(),
            gates: gates.as_ptr(),
            betas: betas.as_ptr(),
            state: cache.delta_s[l].as_mut_ptr(),
            out: out.as_mut_ptr(),
            norm_w: norm_w.as_ptr(),
            m,
            nh,
            hk,
            hv,
            n_group: ssm.n_group,
            key_dim,
            conv_dim,
            value_dim,
            scale,
            rms_eps: c.rms_eps,
        };
        // The recurrence is head-parallel at every window size, but a single
        // decode position is too little work to pay for the wake — see
        // `fanout_chunk`.
        // SAFETY: `heads` is safe on disjoint head ranges sharing `ctx`, and
        // `ctx` lives until `parallel_for` returns.
        unsafe {
            crate::arch::parallel_for(
                nh,
                fanout_chunk(m * nh * hk * hv * 3),
                heads,
                &mut ctx as *mut DeltaCtx as *mut u8,
            )
        };
    }
}

/// Dequantize embedding row `tok` of `qw` (`n=out.len()` columns) into `out`,
/// for any supported quant type (the 9B's token_embd is Q4_0, the 0.8B's is
/// Q8_0). Reads exactly one row from the weight bytes.
fn dequant_embed_row(qw: QWeight, tok: usize, out: &mut [f32]) {
    let n = out.len();
    let (block_bytes, elems) = tensor::block_layout(qw.qt);
    let blocks = n / elems;
    let row_bytes = blocks * block_bytes;
    let row = &qw.data[tok * row_bytes..(tok + 1) * row_bytes];
    let mut buf = [0.0f32; tensor::QK_K];
    for b in 0..blocks {
        tensor::dequant_block(qw.qt, &row[b * block_bytes..(b + 1) * block_bytes], &mut buf[..elems]);
        out[b * elems..(b + 1) * elems].copy_from_slice(&buf[..elems]);
    }
}

/// Unweighted in-place RMS normalization (`x / sqrt(mean(x²)+eps)`) — Gemma-4
/// applies this to V per head (a bare `ggml_rms_norm`, no weight tensor).
fn rms_scale(x: &mut [f32], eps: f32) {
    let mut ms = 0.0f32;
    for &v in x.iter() {
        ms += v * v;
    }
    ms = ms / x.len() as f32 + eps;
    let inv = 1.0 / tensor_sqrtf(ms);
    x.iter_mut().for_each(|v| *v *= inv);
}

fn tensor_sqrtf(x: f32) -> f32 {
    // Hardware sqrt per arch (`core` has no `f32::sqrt`): `sqrtss` on x86,
    // `fsqrt` on aarch64, Newton-Raphson elsewhere. Argument is positive.
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `sqrtss` has no side effects.
    unsafe {
        let mut r = x;
        core::arch::asm!("sqrtss {r}, {r}", r = inout(xmm_reg) r, options(nomem, nostack, preserves_flags));
        r
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `fsqrt` has no side effects.
    unsafe {
        let r: f32;
        core::arch::asm!("fsqrt {r:s}, {x:s}", r = out(vreg) r, x = in(vreg) x, options(nomem, nostack, preserves_flags));
        r
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        let mut g = x;
        for _ in 0..20 {
            g = 0.5 * (g + x / g);
        }
        g
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// FNV-1a hash of the model bytes, for the per-inference `ktrace`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn detokenize(model: &Model, ids: &[usize]) -> String {
    // Flavor-aware unmapping: GPT-2 byte-BPE tokens carry the Ġ/Ċ/ĉ byte
    // aliases; gemma tokens are raw UTF-8 with ▁ whitespace + <0xXX> bytes.
    let gemma = model.gguf.tokenizer_model == "gemma4";
    let mut out = String::new();
    for &id in ids {
        let t = model.token_str(id);
        if gemma {
            if let Some(b) = crate::cortex::tokenizer::parse_byte_token(t) {
                out.push(b as char);
                continue;
            }
            for ch in t.chars() {
                out.push(if ch == '\u{2581}' { ' ' } else { ch });
            }
        } else {
            for ch in t.chars() {
                match ch {
                    'Ġ' => out.push(' '),
                    'Ċ' => out.push('\n'),
                    'ĉ' => out.push('\t'),
                    other => out.push(other),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Reference for the windowed conv: the per-position form this replaced —
    /// shift the ring down a slot, append the current vector, then convolve
    /// post-shift slots. Kept verbatim as the oracle.
    fn conv_per_position(
        qkv: &[f32],
        ring0: &[f32],
        w: &[f32],
        m: usize,
        conv_dim: usize,
        ck: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut ring = ring0.to_vec();
        let mut out = alloc::vec![0.0f32; m * conv_dim];
        for mi in 0..m {
            for j in 0..ck - 1 {
                let (a, b) = ring.split_at_mut((j + 1) * conv_dim);
                a[j * conv_dim..(j + 1) * conv_dim].copy_from_slice(&b[..conv_dim]);
            }
            ring[(ck - 1) * conv_dim..].copy_from_slice(&qkv[mi * conv_dim..(mi + 1) * conv_dim]);
            for ci in 0..conv_dim {
                let mut acc = 0.0f32;
                for j in 0..ck {
                    acc += ring[j * conv_dim + ci] * w[ci * ck + j];
                }
                out[mi * conv_dim + ci] = acc;
            }
        }
        (out, ring)
    }

    /// The window form of the conv (`conv_tap` + `advance_conv_ring`) must equal
    /// the per-position shift-and-convolve it replaced, for every window size
    /// either side of the kernel width — including `m == 1` (decode) and
    /// `m < ck`, where part of the history survives in the ring.
    ///
    /// This is the test the off-by-one needed: reading the ring at `ck - 1 + p`
    /// instead of `ck + p` takes every tap one position too old, which changes
    /// no shape, trips no assertion, and still decodes fluent text.
    #[test_case]
    fn conv_window_matches_the_per_position_ring() {
        const CK: usize = 4;
        const CD: usize = 5; // small but not a multiple of anything convenient
        let w: Vec<f32> = (0..CD * CK).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let ring0: Vec<f32> = (0..CK * CD).map(|i| (i as f32) * 0.5 - 2.0).collect();
        for m in [1usize, 2, 3, 4, 5, 9] {
            let qkv: Vec<f32> = (0..m * CD).map(|i| 1.0 - (i as f32) * 0.125).collect();
            let (want_out, want_ring) = conv_per_position(&qkv, &ring0, &w, m, CD, CK);

            let mut got_out = alloc::vec![0.0f32; m * CD];
            for mi in 0..m {
                for ci in 0..CD {
                    let mut acc = 0.0f32;
                    for j in 0..CK {
                        let x = match conv_tap(mi, j, CK) {
                            ConvTap::Window(p) => qkv[p * CD + ci],
                            ConvTap::Ring(s) => ring0[s * CD + ci],
                        };
                        acc += x * w[ci * CK + j];
                    }
                    got_out[mi * CD + ci] = acc;
                }
            }
            let mut got_ring = ring0.clone();
            advance_conv_ring(&mut got_ring, &qkv, m, CD, CK);

            assert_eq!(got_out, want_out, "conv output diverges at window m={m}");
            assert_eq!(got_ring, want_ring, "conv ring diverges after window m={m}");
        }
    }

    /// A window never reaches further back than `ck-1` positions, so ring slot 0
    /// — the oldest, which the per-position shift would have discarded first —
    /// is never a tap source. Pins the boundary the `ck + p` mapping turns on.
    #[test_case]
    fn conv_taps_never_reach_the_discarded_ring_slot() {
        for ck in [2usize, 3, 4, 8] {
            for mi in 0..6 {
                for j in 0..ck {
                    match conv_tap(mi, j, ck) {
                        ConvTap::Ring(s) => {
                            assert!(s >= 1 && s < ck, "ck={ck} mi={mi} j={j}: ring slot {s} out of 1..{ck}");
                        }
                        ConvTap::Window(p) => assert!(p <= mi, "ck={ck} mi={mi} j={j}: tap {p} is not causal"),
                    }
                }
            }
            // The newest tap is always the current position.
            assert_eq!(conv_tap(3, ck - 1, ck), ConvTap::Window(3));
        }
    }

    /// Batching is a property of each tensor's quant type, never of the model.
    /// The regression this guards: the old gate demanded one uniform type across
    /// every weight *and* read it from `token_embd`, so a single Q6_K embedding —
    /// a tensor prefill never batches — put a whole 4B model on the per-token
    /// path and forfeited batching for 89% of its projection bytes.
    #[test_case]
    fn batchable_types_are_per_tensor_and_k_quants_fall_back() {
        // K-quants, Q4_1 and the i-quants have no weight-stationary kernel.
        for qt in [tensor::QT_Q4_K, tensor::QT_Q6_K, tensor::QT_Q5_K, tensor::QT_Q4_1, tensor::QT_Q2_K] {
            assert!(!has_batched_kernel(qt), "quant {qt} must fall back, not claim a batched kernel");
        }
        // On aarch64 these do; on other arches there are no batched kernels at
        // all, so the predicate is uniformly false and everything falls back.
        for qt in [tensor::QT_Q8_0, tensor::QT_Q1_0, tensor::QT_Q2_0] {
            assert_eq!(has_batched_kernel(qt), cfg!(target_arch = "aarch64"), "quant {qt}");
        }
    }

    /// A decode step's DeltaNet layer must stay inline: it is head-parallel in
    /// principle, but fanning ~0.8M MACs across the fleet measured slower than
    /// running it on one core (35 -> 20 tok/s). A prefill window must fan out.
    #[test_case]
    fn fanout_chunk_keeps_a_decode_step_inline_and_splits_a_window() {
        // 0.8B DeltaNet geometry: nh=16, hk=hv=128, 3 passes.
        let per_pos = 16 * 128 * 128 * 3;
        let inline = fanout_chunk(per_pos);
        assert!(inline > 16, "one decode position must not split across 16 heads (got {inline})");
        assert_eq!(fanout_chunk(per_pos * 64), 1, "a 64-token window must split");
    }
}
