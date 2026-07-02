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

use super::gguf::{Config, Gguf, GgufError, GGML_TYPE_F32, GGML_TYPE_Q8_0};
use super::tensor::{self, QK};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

fn q8_0_bytes(n_cols: usize, n_rows: usize) -> usize {
    n_rows * (n_cols / QK) * tensor::Q8_0_BLOCK_BYTES
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

fn q8_tensor<'a>(g: &Gguf<'a>, name: &str, n_cols: usize, n_rows: usize) -> Result<&'a [u8], GgufError> {
    let info = g.tensor(name)?;
    if info.ggml_type != GGML_TYPE_Q8_0 {
        return Err(GgufError::MissingTensor);
    }
    g.tensor_bytes(name, q8_0_bytes(n_cols, n_rows))
}

/// Per-layer weights; one of the two variants depending on layer type.
enum LayerKind<'a> {
    Attn {
        q: &'a [u8],   // [dim -> n_head*head_dim*2] (query+gate interleaved)
        k: &'a [u8],   // [dim -> n_kv*head_dim]
        v: &'a [u8],
        o: &'a [u8],   // [n_head*head_dim -> dim]
        q_norm: &'a [f32],
        k_norm: &'a [f32],
    },
    Delta {
        qkv: &'a [u8],       // [dim -> conv_dim]
        gate: &'a [u8],      // [dim -> value_dim]  (z)
        conv1d: &'a [f32],   // [conv_dim * conv_kernel], tap j of channel c at c*K+j
        dt_bias: &'a [f32],  // [n_v_heads]
        a_log: &'a [f32],    // [n_v_heads]
        alpha: &'a [u8],     // [dim -> n_v_heads]
        beta: &'a [u8],      // [dim -> n_v_heads]
        norm: &'a [f32],     // [head_v_dim]
        out: &'a [u8],       // [value_dim -> dim]
    },
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    post_norm: &'a [f32],
    kind: LayerKind<'a>,
    ffn_gate: &'a [u8],
    ffn_up: &'a [u8],
    ffn_down: &'a [u8],
}

pub struct Model<'a> {
    pub config: Config,
    gguf: Gguf<'a>,
    token_embd: &'a [u8], // Q8_0 [dim, vocab] -- also the tied output
    output_norm: &'a [f32],
    layers: Vec<Layer<'a>>,
    vocab: usize,
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

        let token_embd = q8_tensor(&gguf, "token_embd.weight", dim, vocab)?;
        let output_norm = f32_tensor(&gguf, "output_norm.weight", dim)?;

        let mut layers = Vec::with_capacity(c.block_count);
        for l in 0..c.block_count {
            let n = |s: &str| format!("blk.{l}.{s}");
            let kind = if c.is_attention_layer(l) {
                LayerKind::Attn {
                    q: q8_tensor(&gguf, &n("attn_q.weight"), dim, attn_q_dim)?,
                    k: q8_tensor(&gguf, &n("attn_k.weight"), dim, kv_dim)?,
                    v: q8_tensor(&gguf, &n("attn_v.weight"), dim, kv_dim)?,
                    o: q8_tensor(&gguf, &n("attn_output.weight"), attn_o_in, dim)?,
                    q_norm: f32_tensor(&gguf, &n("attn_q_norm.weight"), head_dim)?,
                    k_norm: f32_tensor(&gguf, &n("attn_k_norm.weight"), head_dim)?,
                }
            } else {
                LayerKind::Delta {
                    qkv: q8_tensor(&gguf, &n("attn_qkv.weight"), dim, conv_dim)?,
                    gate: q8_tensor(&gguf, &n("attn_gate.weight"), dim, value_dim)?,
                    conv1d: f32_tensor(&gguf, &n("ssm_conv1d.weight"), conv_dim * c.ssm_conv_kernel)?,
                    dt_bias: f32_tensor(&gguf, &n("ssm_dt.bias"), c.ssm_dt_rank)?,
                    a_log: f32_tensor(&gguf, &n("ssm_a"), c.ssm_dt_rank)?,
                    alpha: q8_tensor(&gguf, &n("ssm_alpha.weight"), dim, c.ssm_dt_rank)?,
                    beta: q8_tensor(&gguf, &n("ssm_beta.weight"), dim, c.ssm_dt_rank)?,
                    norm: f32_tensor(&gguf, &n("ssm_norm.weight"), head_v)?,
                    out: q8_tensor(&gguf, &n("ssm_out.weight"), value_dim, dim)?,
                }
            };
            layers.push(Layer {
                attn_norm: f32_tensor(&gguf, &n("attn_norm.weight"), dim)?,
                post_norm: f32_tensor(&gguf, &n("post_attention_norm.weight"), dim)?,
                kind,
                ffn_gate: q8_tensor(&gguf, &n("ffn_gate.weight"), dim, ffn)?,
                ffn_up: q8_tensor(&gguf, &n("ffn_up.weight"), dim, ffn)?,
                ffn_down: q8_tensor(&gguf, &n("ffn_down.weight"), ffn, dim)?,
            });
        }

        Ok(Self { config: c, gguf, token_embd, output_norm, layers, vocab })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn token_str(&self, id: usize) -> &str {
        self.gguf.tokens.get(id).copied().unwrap_or("")
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
        let last = prompt.len();
        for (i, &tok) in prompt.iter().enumerate() {
            self.forward(tok, pos0 + i, cache, state, i + 1 == last);
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

        let row_bytes = (dim / QK) * tensor::Q8_0_BLOCK_BYTES;
        dequant_row_q8_0(&self.token_embd[token * row_bytes..(token + 1) * row_bytes], &mut s.hidden);

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
            tensor::matvec_q8_0_fast(self.layers[l].ffn_gate, &s.norm, &mut s.ffn_gate, &mut s.xq, &mut s.xs, ffn, dim);
            tensor::matvec_q8_0_fast(self.layers[l].ffn_up, &s.norm, &mut s.ffn_up, &mut s.xq, &mut s.xs, ffn, dim);
            tensor::silu_mul(&s.ffn_gate, &s.ffn_up, &mut s.ffn_act);
            tensor::matvec_q8_0_fast(self.layers[l].ffn_down, &s.ffn_act, &mut s.proj, &mut s.xq, &mut s.xs, dim, ffn);
            for i in 0..dim {
                s.hidden[i] = s.residual[i] + s.proj[i];
            }
        }

        if need_logits {
            tensor::rmsnorm(&s.hidden, self.output_norm, c.rms_eps, &mut s.norm);
            tensor::matvec_q8_0_fast(self.token_embd, &s.norm, &mut s.logits, &mut s.xq, &mut s.xs, self.vocab, dim);
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
        tensor::matvec_q8_0_fast(q_w, &s.norm, &mut s.q, &mut s.xq, &mut s.xs, nq * hd * 2, dim);
        tensor::matvec_q8_0_fast(k_w, &s.norm, &mut s.k, &mut s.xq, &mut s.xs, kv_dim, dim);
        tensor::matvec_q8_0_fast(v_w, &s.norm, &mut s.v, &mut s.xq, &mut s.xs, kv_dim, dim);

        // Sequential (recurrent/causal) core: consumes s.q/s.k/s.v, writes s.attn_out.
        self.attn_core(l, pos, cache, s);

        tensor::matvec_q8_0_fast(o_w, &s.attn_out, &mut s.proj, &mut s.xq, &mut s.xs, dim, nq * hd);
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

        tensor::matvec_q8_0_fast(qkv_w, &s.norm, &mut s.qkv, &mut s.xq, &mut s.xs, conv_dim, dim);
        tensor::matvec_q8_0_fast(gate_w, &s.norm, &mut s.z, &mut s.xq, &mut s.xs, value_dim, dim);
        tensor::matvec_q8_0_fast(alpha_w, &s.norm, &mut s.gates, &mut s.xq, &mut s.xs, nh, dim); // reuse gates as alpha
        tensor::matvec_q8_0_fast(beta_w, &s.norm, &mut s.betas, &mut s.xq, &mut s.xs, nh, dim);

        // Sequential (recurrent) core: consumes s.qkv/s.z/s.gates/s.betas, writes s.delta_o.
        self.delta_core(l, cache, s);

        tensor::matvec_q8_0_fast(out_w, &s.delta_o, &mut s.proj, &mut s.xq, &mut s.xs, dim, value_dim);
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
        let nh = c.ssm_dt_rank;
        let hk = c.ssm_state;
        let hv = c.ssm_head_dim();
        let key_dim = hk * c.ssm_n_group;
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
            let q = &mut s.conv[h * hk..(h + 1) * hk].to_vec();
            let k = &mut s.conv[key_dim + h * hk..key_dim + (h + 1) * hk].to_vec();
            let vv = &s.conv[2 * key_dim + h * hv..2 * key_dim + (h + 1) * hv];
            tensor::l2norm(q, 1e-6);
            tensor::l2norm(k, 1e-6);
            for x in q.iter_mut() {
                *x *= scale;
            }
            let sh = &mut s_state[h * hk * hv..(h + 1) * hk * hv]; // S[k*hv + v]
            let gd = s.gates[h];
            let beta = s.betas[h];
            for x in sh.iter_mut() {
                *x *= gd;
            }
            let out = &mut s.delta_o[h * hv..(h + 1) * hv];
            // kv_mem[v] = sum_k S[k,v]*k[k]; delta = (v - kv_mem)*beta;
            // S[k,v] += k[k]*delta[v]; o[v] = sum_k S[k,v]*q[k]
            for vi in 0..hv {
                let mut kv_mem = 0.0f32;
                for ki in 0..hk {
                    kv_mem += sh[ki * hv + vi] * k[ki];
                }
                let delta = (vv[vi] - kv_mem) * beta;
                for ki in 0..hk {
                    sh[ki * hv + vi] += k[ki] * delta;
                }
            }
            for vi in 0..hv {
                let mut acc = 0.0f32;
                for ki in 0..hk {
                    acc += sh[ki * hv + vi] * q[ki];
                }
                out[vi] = acc;
            }
            // gated RMSNorm: RMSNorm(out over hv) * SiLU(z_head)
            tensor::rmsnorm_inplace(out, norm_w, c.rms_eps);
            for vi in 0..hv {
                out[vi] *= tensor::silu(s.z[h * hv + vi]);
            }
        }
    }
}

fn dequant_row_q8_0(row: &[u8], out: &mut [f32]) {
    let blocks = out.len() / QK;
    for b in 0..blocks {
        let block = &row[b * tensor::Q8_0_BLOCK_BYTES..(b + 1) * tensor::Q8_0_BLOCK_BYTES];
        tensor::dequant_q8_0_block(block, &mut out[b * QK..(b + 1) * QK]);
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
