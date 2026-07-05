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

use super::gguf::{Config, Gguf, GgufError, GGML_TYPE_F32};
use super::tensor::{self, QK};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

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
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    let _ = (&mut *xq, &mut *xs);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `qw.data` holds `n_rows` rows for `qw.qt`/`n_cols`; `x`/`y` sized.
    unsafe {
        crate::arch::aarch64::smp::matvec_quant(qw.qt, qw.data.as_ptr(), x.as_ptr(), y.as_mut_ptr(), n_rows, n_cols);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        tensor::matvec_quant_rows(qw.qt, qw.data.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols);
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
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    post_norm: &'a [f32],
    kind: LayerKind<'a>,
    ffn_gate: QWeight<'a>,
    ffn_up: QWeight<'a>,
    ffn_down: QWeight<'a>,
}

pub struct Model<'a> {
    pub config: Config,
    gguf: Gguf<'a>,
    token_embd: QWeight<'a>, // [dim, vocab] -- also the tied output when `output` is None
    output: Option<QWeight<'a>>, // separate (untied) output projection, if present
    output_norm: &'a [f32],
    layers: Vec<Layer<'a>>,
    vocab: usize,
    /// True when every weight tensor is Q8_0 (the 0.8B) -- enables the
    /// Q8_0-only batched prefill; mixed-quant models (9B) prefill sequentially.
    /// Only consulted on aarch64 (the arch with batched prefill).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    all_q8_0: bool,
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
}

impl State {
    pub fn new(c: &Config, vocab: usize) -> Self {
        let dim = c.embedding_length;
        let v = |n| alloc::vec![0.0f32; n];
        // Widest matvec input across all projections (columns): the norm-fed
        // projections use `dim`, ffn_down uses the FFN width, o_proj uses
        // head_count*head_dim, and the DeltaNet output uses ssm_inner.
        let max_cols =
            dim.max(c.feed_forward_length).max(c.head_count * c.head_dim).max(c.ssm_inner);
        Self {
            hidden: v(dim),
            residual: v(dim),
            norm: v(dim),
            q: v(c.head_count * c.head_dim * 2),
            k: v(c.head_count_kv * c.head_dim),
            v: v(c.head_count_kv * c.head_dim),
            attn_out: v(c.head_count * c.head_dim),
            scores: v(c.context_length),
            qkv: v(c.ssm_conv_dim()),
            z: v(c.ssm_inner),
            gates: v(c.ssm_dt_rank),
            betas: v(c.ssm_dt_rank),
            conv: v(c.ssm_conv_dim()),
            delta_o: v(c.ssm_inner),
            ffn_gate: v(c.feed_forward_length),
            ffn_up: v(c.feed_forward_length),
            ffn_act: v(c.feed_forward_length),
            proj: v(dim),
            logits: v(vocab),
            xq: alloc::vec![0i8; max_cols],
            xs: alloc::vec![0.0f32; max_cols / QK],
        }
    }
}

/// Hybrid per-stream cache: KV history for attention layers, recurrent
/// state + conv ring for gated-DeltaNet layers.
pub struct Cache {
    attn_k: Vec<Vec<f32>>, // [layer] -> flattened [pos * (n_kv*head_dim)]
    attn_v: Vec<Vec<f32>>,
    delta_s: Vec<Vec<f32>>, // [layer] -> [n_v_heads * state * head_v_dim]
    conv: Vec<Vec<f32>>,    // [layer] -> conv ring [conv_kernel * conv_dim]
    positions: usize,
}

impl Cache {
    pub fn new(c: &Config) -> Self {
        let n = c.block_count;
        let head_k = c.ssm_state;
        let head_v = c.ssm_head_dim();
        let s_size = c.ssm_dt_rank * head_k * head_v;
        let conv_size = c.ssm_conv_kernel * c.ssm_conv_dim();
        let mut attn_k = Vec::with_capacity(n);
        let mut attn_v = Vec::with_capacity(n);
        let mut delta_s = Vec::with_capacity(n);
        let mut conv = Vec::with_capacity(n);
        for l in 0..n {
            if c.is_attention_layer(l) {
                attn_k.push(Vec::new());
                attn_v.push(Vec::new());
                delta_s.push(Vec::new());
                conv.push(Vec::new());
            } else {
                attn_k.push(Vec::new());
                attn_v.push(Vec::new());
                delta_s.push(alloc::vec![0.0f32; s_size]);
                conv.push(alloc::vec![0.0f32; conv_size]);
            }
        }
        Self { attn_k, attn_v, delta_s, conv, positions: 0 }
    }

    pub fn len(&self) -> usize {
        self.positions
    }
    pub fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Reset all recurrent state / KV history (KV evict). The continuation
    /// is reproduced by replaying tokens through the deterministic pass.
    pub fn evict(&mut self) {
        for k in &mut self.attn_k {
            k.clear();
        }
        for v in &mut self.attn_v {
            v.clear();
        }
        for s in &mut self.delta_s {
            s.iter_mut().for_each(|x| *x = 0.0);
        }
        for c in &mut self.conv {
            c.iter_mut().for_each(|x| *x = 0.0);
        }
        self.positions = 0;
        crate::ktrace::log("cortex.kv", "hybrid cache evicted (KV + recurrent state reset)");
    }
}

impl<'a> Model<'a> {
    pub fn load(gguf: Gguf<'a>) -> Result<Self, GgufError> {
        let c = gguf.config;
        let dim = c.embedding_length;
        let ffn = c.feed_forward_length;
        let vocab = gguf.tokens.len().max(1);
        let head_dim = c.head_dim;
        let attn_q_dim = c.head_count * head_dim * 2;
        let kv_dim = c.head_count_kv * head_dim;
        let attn_o_in = c.head_count * head_dim;
        let conv_dim = c.ssm_conv_dim();
        let value_dim = c.ssm_inner;
        let head_v = c.ssm_head_dim();

        let token_embd = qtensor(&gguf, "token_embd.weight", dim, vocab)?;
        // Untied output projection if the GGUF has one (the 9B); else tied.
        let output = qtensor(&gguf, "output.weight", dim, vocab).ok();
        let output_norm = f32_tensor(&gguf, "output_norm.weight", dim)?;

        let mut layers = Vec::with_capacity(c.block_count);
        for l in 0..c.block_count {
            let n = |s: &str| format!("blk.{l}.{s}");
            let kind = if c.is_attention_layer(l) {
                LayerKind::Attn {
                    q: qtensor(&gguf, &n("attn_q.weight"), dim, attn_q_dim)?,
                    k: qtensor(&gguf, &n("attn_k.weight"), dim, kv_dim)?,
                    v: qtensor(&gguf, &n("attn_v.weight"), dim, kv_dim)?,
                    o: qtensor(&gguf, &n("attn_output.weight"), attn_o_in, dim)?,
                    q_norm: f32_tensor(&gguf, &n("attn_q_norm.weight"), head_dim)?,
                    k_norm: f32_tensor(&gguf, &n("attn_k_norm.weight"), head_dim)?,
                }
            } else {
                LayerKind::Delta {
                    qkv: qtensor(&gguf, &n("attn_qkv.weight"), dim, conv_dim)?,
                    gate: qtensor(&gguf, &n("attn_gate.weight"), dim, value_dim)?,
                    conv1d: f32_tensor(&gguf, &n("ssm_conv1d.weight"), conv_dim * c.ssm_conv_kernel)?,
                    dt_bias: f32_tensor(&gguf, &n("ssm_dt.bias"), c.ssm_dt_rank)?,
                    a_log: f32_tensor(&gguf, &n("ssm_a"), c.ssm_dt_rank)?,
                    alpha: qtensor(&gguf, &n("ssm_alpha.weight"), dim, c.ssm_dt_rank)?,
                    beta: qtensor(&gguf, &n("ssm_beta.weight"), dim, c.ssm_dt_rank)?,
                    norm: f32_tensor(&gguf, &n("ssm_norm.weight"), head_v)?,
                    out: qtensor(&gguf, &n("ssm_out.weight"), value_dim, dim)?,
                }
            };
            layers.push(Layer {
                attn_norm: f32_tensor(&gguf, &n("attn_norm.weight"), dim)?,
                post_norm: f32_tensor(&gguf, &n("post_attention_norm.weight"), dim)?,
                kind,
                ffn_gate: qtensor(&gguf, &n("ffn_gate.weight"), dim, ffn)?,
                ffn_up: qtensor(&gguf, &n("ffn_up.weight"), dim, ffn)?,
                ffn_down: qtensor(&gguf, &n("ffn_down.weight"), ffn, dim)?,
            });
        }

        // All-Q8_0 (0.8B) unlocks the batched-prefill fast path; mixed-quant
        // models fall back to sequential prefill.
        let q8 = |w: QWeight| w.qt == tensor::QT_Q8_0;
        let all_q8_0 = q8(token_embd)
            && output.map(q8).unwrap_or(true)
            && layers.iter().all(|ly| {
                q8(ly.ffn_gate)
                    && q8(ly.ffn_up)
                    && q8(ly.ffn_down)
                    && match ly.kind {
                        LayerKind::Attn { q, k, v, o, .. } => q8(q) && q8(k) && q8(v) && q8(o),
                        LayerKind::Delta { qkv, gate, alpha, beta, out, .. } => {
                            q8(qkv) && q8(gate) && q8(alpha) && q8(beta) && q8(out)
                        }
                    }
            });

        Ok(Self { config: c, gguf, token_embd, output, output_norm, layers, vocab, all_q8_0 })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn token_str(&self, id: usize) -> &str {
        self.gguf.tokens.get(id).copied().unwrap_or("")
    }
    /// Build the BPE text encoder from this model's vocab + merges (owns its
    /// maps, so it outlives borrows of the model). ~40 MiB / ~200 ms for the 9B.
    pub fn tokenizer(&self) -> crate::cortex::tokenizer::Tokenizer {
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
        if prompt.len() >= 2 && self.all_q8_0 {
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
        let c = &self.config;
        let dim = c.embedding_length;
        let ffn = c.feed_forward_length;
        let hd = c.head_dim;
        let nq = c.head_count;
        let kv_dim = c.head_count_kv * hd;
        let qdim = nq * hd * 2; // query+gate interleaved
        let ao = nq * hd; // attn_out / o-proj input width
        let conv_dim = c.ssm_conv_dim();
        let value_dim = c.ssm_inner;
        let nh = c.ssm_dt_rank;
        let m = prompt.len();
        let max_cols = dim.max(ffn).max(ao).max(value_dim);

        // M-wide buffers (one prefill's worth; freed on return).
        let mut hidden = alloc::vec![0.0f32; m * dim];
        let mut norm = alloc::vec![0.0f32; m * dim];
        let mut q = alloc::vec![0.0f32; m * qdim];
        let mut k = alloc::vec![0.0f32; m * kv_dim];
        let mut v = alloc::vec![0.0f32; m * kv_dim];
        let mut attn_out = alloc::vec![0.0f32; m * ao];
        let mut qkv = alloc::vec![0.0f32; m * conv_dim];
        let mut z = alloc::vec![0.0f32; m * value_dim];
        let mut gates = alloc::vec![0.0f32; m * nh];
        let mut betas = alloc::vec![0.0f32; m * nh];
        let mut delta_o = alloc::vec![0.0f32; m * value_dim];
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

        for l in 0..self.layers.len() {
            // rmsnorm(attn_norm) per position.
            for mi in 0..m {
                tensor::rmsnorm(&hidden[mi * dim..(mi + 1) * dim], self.layers[l].attn_norm, c.rms_eps, &mut norm[mi * dim..(mi + 1) * dim]);
            }

            match &self.layers[l].kind {
                LayerKind::Attn { q: q_w, k: k_w, v: v_w, o: o_w, .. } => {
                    let (q_w, k_w, v_w, o_w) = (*q_w, *k_w, *v_w, *o_w);
                    self.batched_proj(q_w, &norm, &mut q, &mut xq, &mut xs, m, qdim, dim);
                    self.batched_proj(k_w, &norm, &mut k, &mut xq, &mut xs, m, kv_dim, dim);
                    self.batched_proj(v_w, &norm, &mut v, &mut xq, &mut xs, m, kv_dim, dim);
                    for mi in 0..m {
                        s.q.copy_from_slice(&q[mi * qdim..(mi + 1) * qdim]);
                        s.k.copy_from_slice(&k[mi * kv_dim..(mi + 1) * kv_dim]);
                        s.v.copy_from_slice(&v[mi * kv_dim..(mi + 1) * kv_dim]);
                        self.attn_core(l, pos0 + mi, cache, s);
                        attn_out[mi * ao..(mi + 1) * ao].copy_from_slice(&s.attn_out[..ao]);
                    }
                    self.batched_proj(o_w, &attn_out, &mut proj_out, &mut xq, &mut xs, m, dim, ao);
                }
                LayerKind::Delta { qkv: qkv_w, gate: gate_w, alpha: alpha_w, beta: beta_w, out: out_w, .. } => {
                    let (qkv_w, gate_w, alpha_w, beta_w, out_w) = (*qkv_w, *gate_w, *alpha_w, *beta_w, *out_w);
                    self.batched_proj(qkv_w, &norm, &mut qkv, &mut xq, &mut xs, m, conv_dim, dim);
                    self.batched_proj(gate_w, &norm, &mut z, &mut xq, &mut xs, m, value_dim, dim);
                    self.batched_proj(alpha_w, &norm, &mut gates, &mut xq, &mut xs, m, nh, dim);
                    self.batched_proj(beta_w, &norm, &mut betas, &mut xq, &mut xs, m, nh, dim);
                    for mi in 0..m {
                        s.qkv.copy_from_slice(&qkv[mi * conv_dim..(mi + 1) * conv_dim]);
                        s.z.copy_from_slice(&z[mi * value_dim..(mi + 1) * value_dim]);
                        s.gates[..nh].copy_from_slice(&gates[mi * nh..(mi + 1) * nh]);
                        s.betas[..nh].copy_from_slice(&betas[mi * nh..(mi + 1) * nh]);
                        self.delta_core(l, cache, s);
                        delta_o[mi * value_dim..(mi + 1) * value_dim].copy_from_slice(&s.delta_o[..value_dim]);
                    }
                    self.batched_proj(out_w, &delta_o, &mut proj_out, &mut xq, &mut xs, m, dim, value_dim);
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
            self.batched_proj(self.layers[l].ffn_gate, &norm, &mut ffn_gate, &mut xq, &mut xs, m, ffn, dim);
            self.batched_proj(self.layers[l].ffn_up, &norm, &mut ffn_up, &mut xq, &mut xs, m, ffn, dim);
            for i in 0..m * ffn {
                ffn_act[i] = tensor::silu(ffn_gate[i]) * ffn_up[i];
            }
            self.batched_proj(self.layers[l].ffn_down, &ffn_act, &mut proj_out, &mut xq, &mut xs, m, dim, ffn);
            for i in 0..m * dim {
                hidden[i] += proj_out[i];
            }
        }

        // Only the final position's logits are needed to pick the first token.
        let last = m - 1;
        tensor::rmsnorm(&hidden[last * dim..(last + 1) * dim], self.output_norm, c.rms_eps, &mut s.norm);
        let out_w = self.output.unwrap_or(self.token_embd);
        matvec_qw(out_w, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
        cache.positions = cache.positions.max(pos0 + m);
    }

    /// Weight-stationary batched projection: quantize each of the `m` input
    /// vectors (`input[mi*cols..]`) to int8, then run one matmul that reads the
    /// weight once and writes `out[mi*rows + r]`. `xq`/`xs` are packed tightly
    /// with stride `cols`/`cols/QK` (the layout `matmul_q8_0_sdot_rows` expects).
    #[cfg(target_arch = "aarch64")]
    fn batched_proj(&self, w: QWeight, input: &[f32], out: &mut [f32], xq: &mut [i8], xs: &mut [f32], m: usize, rows: usize, cols: usize) {
        debug_assert_eq!(w.qt, tensor::QT_Q8_0); // batched prefill is Q8_0-only (all_q8_0)
        let nb = cols / QK;
        for mi in 0..m {
            tensor::quantize_activations_q8(
                &input[mi * cols..(mi + 1) * cols],
                &mut xq[mi * cols..(mi + 1) * cols],
                &mut xs[mi * nb..(mi + 1) * nb],
            );
        }
        // SAFETY: `w` is `rows` Q8_0 rows of `cols/QK` blocks; `xq`/`xs` hold `m`
        // activations of `cols`/`nb`; `out` has `m*rows` slots. matmul_sdot
        // splits `[0,rows)` across cores, each writing a disjoint row range.
        unsafe {
            crate::arch::aarch64::smp::matmul_sdot(w.data.as_ptr(), xq.as_ptr(), xs.as_ptr(), out.as_mut_ptr(), m, rows, cols);
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

        for l in 0..self.layers.len() {
            s.residual.copy_from_slice(&s.hidden);
            tensor::rmsnorm(&s.hidden, self.layers[l].attn_norm, c.rms_eps, &mut s.norm);
            match &self.layers[l].kind {
                LayerKind::Attn { .. } => self.attn_layer(l, pos, cache, s),
                LayerKind::Delta { .. } => self.delta_layer(l, cache, s),
            }
            // attn residual: hidden = residual + proj
            for i in 0..dim {
                s.hidden[i] = s.residual[i] + s.proj[i];
            }
            // FFN block (SwiGLU) with its own residual.
            s.residual.copy_from_slice(&s.hidden);
            tensor::rmsnorm(&s.hidden, self.layers[l].post_norm, c.rms_eps, &mut s.norm);
            let ffn = c.feed_forward_length;
            matvec_qw(self.layers[l].ffn_gate, &s.norm, &mut s.ffn_gate, &mut s.xq, &mut s.xs, ffn, dim);
            matvec_qw(self.layers[l].ffn_up, &s.norm, &mut s.ffn_up, &mut s.xq, &mut s.xs, ffn, dim);
            tensor::silu_mul(&s.ffn_gate, &s.ffn_up, &mut s.ffn_act);
            matvec_qw(self.layers[l].ffn_down, &s.ffn_act, &mut s.proj, &mut s.xq, &mut s.xs, dim, ffn);
            for i in 0..dim {
                s.hidden[i] = s.residual[i] + s.proj[i];
            }
        }

        if need_logits {
            tensor::rmsnorm(&s.hidden, self.output_norm, c.rms_eps, &mut s.norm);
            // Untied output projection if present (9B), else the tied embeddings.
            let out_w = self.output.unwrap_or(self.token_embd);
            matvec_qw(out_w, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
        }

        cache.positions = cache.positions.max(pos + 1);
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
        matvec_qw(q_w, &s.norm, &mut s.q, &mut s.xq, &mut s.xs, nq * hd * 2, dim);
        matvec_qw(k_w, &s.norm, &mut s.k, &mut s.xq, &mut s.xs, kv_dim, dim);
        matvec_qw(v_w, &s.norm, &mut s.v, &mut s.xq, &mut s.xs, kv_dim, dim);

        // Sequential (recurrent/causal) core: consumes s.q/s.k/s.v, writes s.attn_out.
        self.attn_core(l, pos, cache, s);

        matvec_qw(o_w, &s.attn_out, &mut s.proj, &mut s.xq, &mut s.xs, dim, nq * hd);
    }

    /// The position-sequential part of an attention layer, shared by decode
    /// (`attn_layer`) and batched prefill: QK-norm + partial RoPE, append K/V to
    /// this layer's history, GQA causal attention, per-head sigmoid gate. Reads
    /// the projections in `s.q`/`s.k`/`s.v`; writes `s.attn_out`. Kept separate
    /// so the projections above can be batched across positions while this
    /// (cheap, order-dependent) core stays identical -- parity by construction.
    fn attn_core(&self, l: usize, pos: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let hd = c.head_dim;
        let nq = c.head_count;
        let nkv = c.head_count_kv;
        let kv_dim = nkv * hd;
        let group = nq / nkv;
        let scale = 1.0 / tensor_sqrtf(hd as f32);
        let (q_norm, k_norm) = match &self.layers[l].kind {
            LayerKind::Attn { q_norm, k_norm, .. } => (*q_norm, *k_norm),
            _ => unreachable!(),
        };

        // QK-norm per head, partial RoPE on the first rope_dim dims.
        for h in 0..nq {
            let q = &mut s.q[h * 2 * hd..h * 2 * hd + hd]; // query half of this head
            tensor::rmsnorm_inplace(q, q_norm, c.rms_eps);
            tensor::rope(&mut q[0..c.rope_dim], pos, c.rope_dim, c.rope_freq_base);
        }
        for h in 0..nkv {
            let k = &mut s.k[h * hd..(h + 1) * hd];
            tensor::rmsnorm_inplace(k, k_norm, c.rms_eps);
            tensor::rope(&mut k[0..c.rope_dim], pos, c.rope_dim, c.rope_freq_base);
        }

        // Append k,v to this layer's KV history.
        cache.attn_k[l].extend_from_slice(&s.k);
        cache.attn_v[l].extend_from_slice(&s.v);

        // GQA causal attention with per-head sigmoid output gate.
        for h in 0..nq {
            let kvh = h / group;
            let q_head = &s.q[h * 2 * hd..h * 2 * hd + hd];
            for t in 0..=pos {
                let k_t = &cache.attn_k[l][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                s.scores[t] = tensor::dot_f32(q_head, k_t) * scale;
            }
            tensor::softmax(&mut s.scores[0..=pos]);
            let out = &mut s.attn_out[h * hd..(h + 1) * hd];
            out.iter_mut().for_each(|x| *x = 0.0);
            for t in 0..=pos {
                let v_t = &cache.attn_v[l][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                let w = s.scores[t];
                for i in 0..hd {
                    out[i] += w * v_t[i];
                }
            }
            // gate is the second half of head h's slice in s.q
            for i in 0..hd {
                out[i] *= tensor::sigmoid(s.q[h * 2 * hd + hd + i]);
            }
        }
    }

    fn delta_layer(&self, l: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let dim = c.embedding_length;
        let conv_dim = c.ssm_conv_dim();
        let value_dim = c.ssm_inner;
        let nh = c.ssm_dt_rank; // = n_group; k and v head counts are equal
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

    /// The position-sequential (recurrent) part of a DeltaNet layer, shared by
    /// decode (`delta_layer`) and batched prefill: gate/beta activation, causal
    /// conv1d over the ring, and the gated delta rule that advances the
    /// recurrent state. Reads the projections in `s.qkv`/`s.z`/`s.gates`
    /// (=alpha)/`s.betas`; writes `s.delta_o`. Split out so the projections can
    /// be batched while this order-dependent recurrence stays identical.
    fn delta_core(&self, l: usize, cache: &mut Cache, s: &mut State) {
        let c = &self.config;
        let conv_dim = c.ssm_conv_dim();
        let nh = c.ssm_dt_rank; // value heads
        let hk = c.ssm_state;
        let hv = c.ssm_head_dim();
        let key_dim = hk * c.ssm_n_group;
        // GQA over the recurrent heads: `n_group` key/query heads, `nh` value
        // heads. llama.cpp maps them with `ggml_repeat` (tiling), i.e. value
        // head h uses key head `h % n_group` -- NOT repeat-interleave. (When
        // n_group == nh, e.g. the 0.8B, both are the identity.)
        let ck = c.ssm_conv_kernel;
        let (conv1d, dt_bias, a_log, norm_w) = match &self.layers[l].kind {
            LayerKind::Delta { conv1d, dt_bias, a_log, norm, .. } => (*conv1d, *dt_bias, *a_log, *norm),
            _ => unreachable!(),
        };

        // g = -exp(A_log) * softplus(alpha + dt_bias)  (log-decay); beta = sigmoid.
        for h in 0..nh {
            let a = -tensor::expf(a_log[h]);
            let g = a * tensor::softplus(s.gates[h] + dt_bias[h]);
            s.gates[h] = tensor::expf(g); // decay multiplier in (0,1)
            s.betas[h] = tensor::sigmoid(s.betas[h]);
        }

        // Causal depthwise conv1d over the conv ring (last `ck` qkv vectors)
        // + SiLU. Ring layout: conv[j*conv_dim + c], j=0 oldest .. ck-1 newest.
        let ring = &mut cache.conv[l];
        // shift older, append current qkv at the newest slot
        for j in 0..ck - 1 {
            let (a, b) = ring.split_at_mut((j + 1) * conv_dim);
            a[j * conv_dim..(j + 1) * conv_dim].copy_from_slice(&b[..conv_dim]);
        }
        ring[(ck - 1) * conv_dim..].copy_from_slice(&s.qkv);
        for cidx in 0..conv_dim {
            let mut acc = 0.0f32;
            for j in 0..ck {
                acc += ring[j * conv_dim + cidx] * conv1d[cidx * ck + j];
            }
            s.conv[cidx] = tensor::silu(acc);
        }

        // Recurrent gated delta rule per head.
        let scale = 1.0 / tensor_sqrtf(hk as f32);
        let s_state = &mut cache.delta_s[l];
        for h in 0..nh {
            let g = h % c.ssm_n_group; // key/query head (ggml_repeat tiling, per llama.cpp)
            let q = &mut s.conv[g * hk..(g + 1) * hk].to_vec();
            let k = &mut s.conv[key_dim + g * hk..key_dim + (g + 1) * hk].to_vec();
            let vv = &s.conv[2 * key_dim + h * hv..2 * key_dim + (h + 1) * hv];
            tensor::l2norm(q, 1e-6);
            tensor::l2norm(k, 1e-6);
            for x in q.iter_mut() {
                *x *= scale;
            }
            let sh = &mut s_state[h * hk * hv..(h + 1) * hk * hv]; // S[k*hv + v]
            let gd = s.gates[h];
            let beta = s.betas[h];
            tensor::scale_f32(sh, gd);
            let out = &mut s.delta_o[h * hv..(h + 1) * hv];
            // kv_mem[v] = sum_k S[k,v]*k[k]; delta = (v - kv_mem)*beta;
            // S[k,v] += k[k]*delta[v]; o[v] = sum_k S[k,v]*q[k]
            //
            // Iterated **row-wise** (ki outer): each S row `sh[ki*hv..]` is a
            // contiguous [hv] slice, so every pass is a SIMD AXPY instead of a
            // stride-hv scalar walk — this delta rule (18 layers × 16 heads ×
            // 128×128 state, every token) was the decode-time hot spot.
            debug_assert!(hv <= 256, "DeltaNet head_v dim {hv} exceeds the fixed delta scratch");
            let mut delta = [0.0f32; 256]; // hv <= 256 for the supported models
            let delta = &mut delta[..hv];
            delta.copy_from_slice(vv);
            for ki in 0..hk {
                // delta[v] -= k[ki] * S[ki, v]  (accumulating -kv_mem into vv)
                tensor::axpy_f32(delta, &sh[ki * hv..(ki + 1) * hv], -k[ki]);
            }
            for d in delta.iter_mut() {
                *d *= beta;
            }
            out.fill(0.0);
            for ki in 0..hk {
                let row = &mut sh[ki * hv..(ki + 1) * hv];
                tensor::axpy_f32(row, delta, k[ki]);
                tensor::axpy_f32(out, row, q[ki]);
            }
            // gated RMSNorm: RMSNorm(out over hv) * SiLU(z_head)
            tensor::rmsnorm_inplace(out, norm_w, c.rms_eps);
            for vi in 0..hv {
                out[vi] *= tensor::silu(s.z[h * hv + vi]);
            }
        }
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
    let mut out = String::new();
    for &id in ids {
        for ch in model.token_str(id).chars() {
            match ch {
                'Ġ' => out.push(' '),
                'Ċ' => out.push('\n'),
                'ĉ' => out.push('\t'),
                other => out.push(other),
            }
        }
    }
    out
}
