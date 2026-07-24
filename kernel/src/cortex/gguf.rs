//! Minimal GGUF (v3) parser. Reads the model Limine/the stub hands us as a
//! boot module, entirely zero-copy: tensor data and token strings are returned
//! as slices *into* the module memory (reachable through the HHDM), never
//! copied onto the heap -- models are hundreds of MiB and copying would be
//! absurd.
//!
//! **Architecture-dynamic**: `general.architecture` names the metadata key
//! prefix (`qwen35.block_count`, `gemma4.attention.sliding_window`, ...), so
//! the hyperparameters are discovered per model file, the way the ONNX reader
//! discovers a graph — nothing numeric is compiled in. The architecture string
//! resolves to a [`Family`] (the forward-pass shape); a finetune that renames
//! the arch still loads via the key-shape sniff. Only the subset Cortex needs
//! is parsed: the numeric config, the tensor directory (name → dims/type/
//! offset), and the tokenizer tables (tokens, merges, scores, token types).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element types (named ids; the full quant set lives in
/// `tensor::block_layout`).
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q8_0: u32 = 8;

#[derive(Debug)]
pub enum GgufError {
    BadMagic,
    UnsupportedVersion(u32),
    /// `general.architecture` names a family we have no forward pass for
    /// (and the key shape matched nothing known).
    UnsupportedArch,
    Truncated,
    BadString,
    UnknownValueType(u32),
    /// A required metadata key is absent (the suffix under the arch prefix,
    /// or the full key for `tokenizer.*`/`general.*`).
    MissingKey(&'static str),
    /// A key is present but structurally unusable (e.g. a per-layer array
    /// that doesn't reduce to the two-geometry SWA scheme).
    BadValue(&'static str),
    MissingTensor,
}

/// The forward-pass family an architecture string resolves to. The *prefix*
/// stays the raw string (key lookup); the family picks the layer graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen3.5/3.6 hybrid: gated-DeltaNet layers with every
    /// `full_attn_interval`-th layer full attention (QK-norm, gated).
    QwenHybrid,
    /// Gemma 4: attention-only, sliding-window/global interleave with
    /// per-kind geometry, sandwich norms, GELU, logit softcap.
    Gemma4,
}

/// Gated-DeltaNet (SSM) hyperparameters — present for [`Family::QwenHybrid`].
#[derive(Clone, Copy, Debug)]
pub struct SsmConfig {
    pub n_group: usize,     // linear-attn key heads (16)
    pub dt_rank: usize,     // linear-attn value heads
    pub inner: usize,       // value_dim
    pub state: usize,       // per-head state / key dim (128)
    pub conv_kernel: usize, // causal conv width (4)
}

impl SsmConfig {
    /// Width of the DeltaNet qkv projection / conv (`key_dim*2 + value_dim`).
    pub fn conv_dim(&self) -> usize {
        self.state * self.n_group * 2 + self.inner
    }
    pub fn head_dim(&self) -> usize {
        self.inner / self.dt_rank
    }
}

/// Sliding-window attention scheme — present for [`Family::Gemma4`]. The base
/// `Config` geometry (`head_count_kv`/`head_dim`/`rope_dim`/`rope_freq_base`)
/// carries the *sliding*-layer values; global layers override with these.
#[derive(Clone, Copy, Debug)]
pub struct SwaConfig {
    /// Sliding window length in tokens.
    pub window: usize,
    /// Bit `l` set → layer `l` is a sliding-window layer (block_count <= 64).
    pub pattern: u64,
    pub head_count_kv_global: usize,
    pub head_dim_global: usize,
    pub rope_dim_global: usize,
    pub freq_base_global: f32,
    /// Final-logit tanh softcap (0.0 = none).
    pub logit_softcap: f32,
}

/// The numeric hyperparameters Cortex's forward pass needs, pulled from the
/// `{general.architecture}.*` / `tokenizer.*` metadata keys.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub family: Family,
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub context_length: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// QwenHybrid: full-attention layers occur where
    /// `(layer + 1) % full_attn_interval == 0`. 1 for attention-only families.
    pub full_attn_interval: usize,
    // --- attention geometry (sliding-layer values for SWA families) ---
    pub head_count: usize,    // query heads
    pub head_count_kv: usize, // kv heads
    pub head_dim: usize,      // attention head dim = key_length
    pub rope_dim: usize,      // rotary dims per head (partial RoPE)
    /// Gated-DeltaNet hyperparameters (QwenHybrid).
    pub ssm: Option<SsmConfig>,
    /// Sliding-window scheme (Gemma4).
    pub swa: Option<SwaConfig>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    /// Whether the template should prepend BOS once at context start.
    pub add_bos: bool,
}

impl Config {
    /// Is layer `l` a self-attention layer (vs a gated-DeltaNet layer)?
    pub fn is_attention_layer(&self, l: usize) -> bool {
        match self.family {
            Family::QwenHybrid => (l + 1) % self.full_attn_interval == 0,
            Family::Gemma4 => true,
        }
    }
    /// Is layer `l` a sliding-window layer (SWA families; false otherwise)?
    pub fn is_sliding(&self, l: usize) -> bool {
        self.swa.map(|s| (s.pattern >> l) & 1 == 1).unwrap_or(false)
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset from the start of the tensor-data section (already aligned).
    pub offset: u64,
}

pub struct Gguf<'a> {
    data: &'a [u8],
    tensor_data_base: usize,
    pub config: Config,
    /// The raw `general.architecture` string (also the metadata key prefix).
    pub arch: &'a str,
    /// `general.name` — the human model name, for display/refcheck keying.
    pub name: Option<&'a str>,
    tensors: BTreeMap<&'a str, TensorInfo>,
    /// Token id → token string, sliced from the module. Empty if the model
    /// carried no token table.
    pub tokens: Vec<&'a str>,
    /// BPE merge rules in priority order (`"<left> <right>"` per entry), for
    /// the text encoder (`cortex::tokenizer`). Empty if none were present.
    pub merges: Vec<&'a str>,
    /// Per-token scores (SPM-style vocabs); empty for merge-ranked vocabs.
    pub scores: Vec<f32>,
    /// Per-token type (`tokenizer.ggml.token_type`: 1=normal, 2=unknown,
    /// 3=control, 6=byte, ...); empty if absent.
    pub token_type: Vec<i32>,
    /// `tokenizer.ggml.model` — the tokenizer flavor ("gpt2", "gemma4", ...).
    pub tokenizer_model: &'a str,
    /// `tokenizer.chat_template` (raw Jinja source) — retained for reference,
    /// not interpreted (the shell's ChatFormat renders per family).
    pub chat_template: Option<&'a str>,
    /// Whether the tokenizer prepends a space before encoding (SPM flavors).
    pub add_space_prefix: bool,
    /// Token ids the sampler must never emit (`tokenizer.ggml.suppress_tokens`).
    pub suppress_tokens: Vec<u32>,
}

/// Bounds-checked forward cursor over the module bytes.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated)?;
        let slice = self.data.get(self.pos..end).ok_or(GgufError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// A GGUF string: `u64` length + UTF-8 bytes, borrowed from the module.
    fn gstr(&mut self) -> Result<&'a str, GgufError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes).map_err(|_| GgufError::BadString)
    }
}

/// A GGUF metadata value — enough to read the scalars/arrays Cortex retains.
enum Value<'a> {
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(&'a str),
    /// Array of strings (tokens / merges / tags).
    StrArray(Vec<&'a str>),
    /// Array of i32/u32 (token types, per-layer head counts, suppress lists).
    I32Array(Vec<i32>),
    /// Array of f32 (token scores).
    F32Array(Vec<f32>),
    /// Array of bool (per-layer sliding-window pattern).
    BoolArray(Vec<bool>),
    /// Any other value/array, skipped (cursor already advanced past it).
    Other,
}

fn read_value<'a>(c: &mut Cursor<'a>, value_type: u32) -> Result<Value<'a>, GgufError> {
    Ok(match value_type {
        0 | 1 => {
            c.take(1)?;
            Value::Other
        } // u8/i8
        2 | 3 => {
            c.take(2)?;
            Value::Other
        } // u16/i16
        4 => Value::U32(c.u32()?),
        5 => Value::U32(c.u32()?), // i32 read as bits; only used for ids/counts
        6 => Value::F32(c.f32()?),
        7 => Value::Bool(c.take(1)?[0] != 0),
        10 => Value::U64(c.u64()?),
        11 => Value::U64(c.u64()?), // i64
        12 => {
            c.take(8)?;
            Value::Other
        } // f64
        8 => Value::Str(c.gstr()?),
        9 => {
            let elem_type = c.u32()?;
            let count = c.u64()? as usize;
            // Allocation-bomb guard: never `with_capacity` from an untrusted
            // count alone. Cap absolute size; fixed-width arrays also check
            // against remaining file bytes when possible.
            const MAX_META_ARRAY: usize = 1_000_000;
            if count > MAX_META_ARRAY {
                return Err(GgufError::Truncated);
            }
            match elem_type {
                8 => {
                    let mut v = Vec::new();
                    v.try_reserve(count.min(4096)).map_err(|_| GgufError::Truncated)?;
                    for _ in 0..count {
                        v.push(c.gstr()?);
                    }
                    Value::StrArray(v)
                }
                4 | 5 => {
                    let mut v = Vec::new();
                    v.try_reserve(count.min(65_536)).map_err(|_| GgufError::Truncated)?;
                    for _ in 0..count {
                        v.push(c.u32()? as i32);
                    }
                    Value::I32Array(v)
                }
                6 => {
                    let mut v = Vec::new();
                    v.try_reserve(count.min(65_536)).map_err(|_| GgufError::Truncated)?;
                    for _ in 0..count {
                        v.push(c.f32()?);
                    }
                    Value::F32Array(v)
                }
                7 => {
                    let mut v = Vec::new();
                    v.try_reserve(count.min(65_536)).map_err(|_| GgufError::Truncated)?;
                    for _ in 0..count {
                        v.push(c.take(1)?[0] != 0);
                    }
                    Value::BoolArray(v)
                }
                _ => {
                    // Skip a fixed-width-element array without materializing it.
                    for _ in 0..count {
                        read_value(c, elem_type)?;
                    }
                    Value::Other
                }
            }
        }
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

impl<'a> Gguf<'a> {
    /// Parse `data` (the entire GGUF boot module) into a directory of
    /// tensors + config + tokenizer tables. Zero-copy: all returned slices
    /// borrow `data`.
    pub fn parse(data: &'a [u8]) -> Result<Self, GgufError> {
        let mut c = Cursor::new(data);
        if c.u32()? != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = c.u32()?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;

        // --- metadata ---
        let mut get_u32: BTreeMap<&str, u32> = BTreeMap::new();
        let mut get_f32: BTreeMap<&str, f32> = BTreeMap::new();
        let mut get_str: BTreeMap<&str, &'a str> = BTreeMap::new();
        let mut get_bool: BTreeMap<&str, bool> = BTreeMap::new();
        let mut alignment = DEFAULT_ALIGNMENT;
        let mut tokens: Vec<&str> = Vec::new();
        let mut merges: Vec<&str> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut token_type: Vec<i32> = Vec::new();
        let mut suppress_tokens: Vec<u32> = Vec::new();
        // Per-layer arrays some architectures use where others have scalars.
        let mut kv_heads_per_layer: Option<Vec<i32>> = None;
        let mut sliding_pattern: Option<Vec<bool>> = None;

        for _ in 0..n_kv {
            let key = c.gstr()?;
            let value_type = c.u32()?;
            let value = read_value(&mut c, value_type)?;
            match value {
                Value::U32(v) => {
                    get_u32.insert(key, v);
                    if key == "general.alignment" {
                        alignment = v as usize;
                    }
                }
                Value::U64(v) => {
                    get_u32.insert(key, v as u32);
                }
                Value::F32(v) => {
                    get_f32.insert(key, v);
                }
                Value::Bool(v) => {
                    get_bool.insert(key, v);
                }
                Value::Str(v) => {
                    get_str.insert(key, v);
                }
                Value::StrArray(v) if key == "tokenizer.ggml.tokens" => tokens = v,
                Value::StrArray(v) if key == "tokenizer.ggml.merges" => merges = v,
                Value::F32Array(v) if key == "tokenizer.ggml.scores" => scores = v,
                Value::I32Array(v) => match key {
                    "tokenizer.ggml.token_type" => token_type = v,
                    "tokenizer.ggml.suppress_tokens" => suppress_tokens = v.into_iter().map(|x| x as u32).collect(),
                    k if k.ends_with(".attention.head_count_kv") => kv_heads_per_layer = Some(v),
                    _ => {}
                },
                Value::BoolArray(v) if key.ends_with(".attention.sliding_window_pattern") => sliding_pattern = Some(v),
                _ => {}
            }
        }

        // --- architecture → key prefix + family ---
        let arch = *get_str.get("general.architecture").ok_or(GgufError::MissingKey("general.architecture"))?;
        // The arch string is the hyperparameter key prefix (llama.cpp
        // convention). Unknown strings fall back to a key-shape sniff so a
        // renamed finetune of a known family still loads.
        let ak = |s: &str| format!("{arch}.{s}");
        let family = match arch {
            "qwen35" | "qwen36" => Family::QwenHybrid,
            "gemma4" => Family::Gemma4,
            _ if get_u32.contains_key(ak("ssm.inner_size").as_str()) => Family::QwenHybrid,
            _ if get_u32.contains_key(ak("attention.sliding_window").as_str()) => Family::Gemma4,
            _ => return Err(GgufError::UnsupportedArch),
        };

        let u32_of = |k: &'static str| -> Result<u32, GgufError> {
            get_u32.get(ak(k).as_str()).copied().ok_or(GgufError::MissingKey(k))
        };
        let f32_of = |k: &'static str| -> Result<f32, GgufError> {
            get_f32.get(ak(k).as_str()).copied().ok_or(GgufError::MissingKey(k))
        };
        let block_count = u32_of("block_count")? as usize;

        let ssm = match family {
            Family::QwenHybrid => Some(SsmConfig {
                n_group: u32_of("ssm.group_count")? as usize,
                dt_rank: u32_of("ssm.time_step_rank")? as usize,
                inner: u32_of("ssm.inner_size")? as usize,
                state: u32_of("ssm.state_size")? as usize,
                conv_kernel: u32_of("ssm.conv_kernel")? as usize,
            }),
            Family::Gemma4 => None,
        };

        // Attention geometry. SWA families carry two variants: the base
        // fields take the sliding-layer values (`*_swa` keys), and SwaConfig
        // carries the global-layer overrides.
        let (head_count_kv, head_dim, rope_dim, rope_freq_base, swa) = match family {
            Family::QwenHybrid => (
                u32_of("attention.head_count_kv")? as usize,
                u32_of("attention.key_length")? as usize,
                u32_of("rope.dimension_count")? as usize,
                f32_of("rope.freq_base").unwrap_or(10_000_000.0),
                None,
            ),
            Family::Gemma4 => {
                if block_count > 64 {
                    return Err(GgufError::BadValue("sliding pattern exceeds 64 layers"));
                }
                let pattern_arr = sliding_pattern.ok_or(GgufError::MissingKey("attention.sliding_window_pattern"))?;
                if pattern_arr.len() != block_count {
                    return Err(GgufError::BadValue("sliding_window_pattern length != block_count"));
                }
                let mut pattern = 0u64;
                for (l, &s) in pattern_arr.iter().enumerate() {
                    pattern |= (s as u64) << l;
                }
                // Per-layer kv heads must reduce to one value per layer kind
                // (sliding vs global) — the two-geometry scheme.
                // Accept either a per-layer I32 array *or* a scalar U32 (Gemma 4
                // E4B ships the scalar form: one n_kv for every layer).
                let kv: Vec<i32> = if let Some(v) = kv_heads_per_layer {
                    if v.len() != block_count {
                        return Err(GgufError::BadValue("head_count_kv length != block_count"));
                    }
                    v
                } else if let Some(n) = get_u32.get(ak("attention.head_count_kv").as_str()) {
                    alloc::vec![*n as i32; block_count]
                } else {
                    return Err(GgufError::MissingKey("attention.head_count_kv"));
                };
                let (mut kv_swa, mut kv_global) = (None, None);
                for (l, &n) in kv.iter().enumerate() {
                    let slot = if (pattern >> l) & 1 == 1 { &mut kv_swa } else { &mut kv_global };
                    match *slot {
                        None => *slot = Some(n as usize),
                        Some(prev) if prev == n as usize => {}
                        Some(_) => return Err(GgufError::BadValue("head_count_kv varies within a layer kind")),
                    }
                }
                let kv_swa = kv_swa.ok_or(GgufError::BadValue("no sliding layers in pattern"))?;
                // Global layers may share the same scalar n_kv as sliding (E4B);
                // if the pattern has no global bit, fall back to the sliding value.
                let kv_global = kv_global.unwrap_or(kv_swa);
                let head_dim_global = u32_of("attention.key_length")? as usize;
                let head_dim_swa = u32_of("attention.key_length_swa")? as usize;
                let rope_dim_global = u32_of("rope.dimension_count").map(|v| v as usize).unwrap_or(head_dim_global);
                let rope_dim_swa = u32_of("rope.dimension_count_swa").map(|v| v as usize).unwrap_or(head_dim_swa);
                let swa = SwaConfig {
                    window: u32_of("attention.sliding_window")? as usize,
                    pattern,
                    head_count_kv_global: kv_global,
                    head_dim_global,
                    rope_dim_global,
                    freq_base_global: f32_of("rope.freq_base").unwrap_or(1_000_000.0),
                    logit_softcap: f32_of("final_logit_softcapping").unwrap_or(0.0),
                };
                (kv_swa, head_dim_swa, rope_dim_swa, f32_of("rope.freq_base_swa").unwrap_or(10_000.0), Some(swa))
            }
        };

        let config = Config {
            family,
            block_count,
            embedding_length: u32_of("embedding_length")? as usize,
            feed_forward_length: u32_of("feed_forward_length")? as usize,
            context_length: u32_of("context_length")? as usize,
            rms_eps: f32_of("attention.layer_norm_rms_epsilon")?,
            rope_freq_base,
            full_attn_interval: match family {
                Family::QwenHybrid => u32_of("full_attention_interval").unwrap_or(4) as usize,
                Family::Gemma4 => 1,
            },
            head_count: u32_of("attention.head_count")? as usize,
            head_count_kv,
            head_dim,
            rope_dim,
            ssm,
            swa,
            bos_token_id: get_u32.get("tokenizer.ggml.bos_token_id").copied(),
            eos_token_id: get_u32
                .get("tokenizer.ggml.eos_token_id")
                .copied()
                .ok_or(GgufError::MissingKey("tokenizer.ggml.eos_token_id"))?,
            add_bos: get_bool.get("tokenizer.ggml.add_bos_token").copied().unwrap_or(false),
        };

        // --- tensor directory ---
        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = c.gstr()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.insert(name, TensorInfo { dims, ggml_type, offset });
        }

        let tensor_data_base = align_up(c.pos, alignment);

        Ok(Self {
            data,
            tensor_data_base,
            config,
            arch,
            name: get_str.get("general.name").copied(),
            tensors,
            tokens,
            merges,
            scores,
            token_type,
            tokenizer_model: get_str.get("tokenizer.ggml.model").copied().unwrap_or("gpt2"),
            chat_template: get_str.get("tokenizer.chat_template").copied(),
            add_space_prefix: get_bool.get("tokenizer.ggml.add_space_prefix").copied().unwrap_or(false),
            suppress_tokens,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.tensors.get(name).ok_or(GgufError::MissingTensor)
    }

    /// Raw bytes of `name`'s tensor data, sliced from the module. `len` is
    /// the caller-computed byte length (depends on dims × element size).
    pub fn tensor_bytes(&self, name: &str, len: usize) -> Result<&'a [u8], GgufError> {
        let info = self.tensor(name)?;
        let start = self.tensor_data_base + info.offset as usize;
        self.data.get(start..start + len).ok_or(GgufError::Truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The model-load flow's first guardrails: reject a non-GGUF blob and an
    /// unsupported container version, rather than reading garbage as a model.
    #[test_case]
    fn rejects_bad_magic() {
        let not_gguf = [0u8, 1, 2, 3, 4, 5, 6, 7];
        assert!(matches!(Gguf::parse(&not_gguf), Err(GgufError::BadMagic)));
    }

    #[test_case]
    fn rejects_unsupported_version() {
        // "GGUF" magic (LE) + version 2 (we only support v3).
        let mut buf = vec![0x47u8, 0x47, 0x55, 0x46];
        buf.extend_from_slice(&2u32.to_le_bytes());
        assert!(matches!(Gguf::parse(&buf), Err(GgufError::UnsupportedVersion(2))));
    }

    #[test_case]
    fn rejects_truncated_header() {
        // Correct magic + v3 but nothing after → must error, not panic.
        let mut buf = vec![0x47u8, 0x47, 0x55, 0x46];
        buf.extend_from_slice(&3u32.to_le_bytes());
        assert!(Gguf::parse(&buf).is_err());
    }

    // --- synthetic GGUF builder: a tiny in-memory writer so the dynamic
    // config path is testable without a real multi-GB model file. ---

    struct B {
        buf: Vec<u8>,
        n_kv: u64,
    }

    impl B {
        fn new() -> Self {
            let mut buf = vec![0x47u8, 0x47, 0x55, 0x46];
            buf.extend_from_slice(&3u32.to_le_bytes());
            // n_tensors=0; n_kv patched in finish().
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
            Self { buf, n_kv: 0 }
        }
        fn s(&mut self, s: &str) {
            self.buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(s.as_bytes());
        }
        fn kv_u32(&mut self, k: &str, v: u32) {
            self.s(k);
            self.buf.extend_from_slice(&4u32.to_le_bytes());
            self.buf.extend_from_slice(&v.to_le_bytes());
            self.n_kv += 1;
        }
        fn kv_f32(&mut self, k: &str, v: f32) {
            self.s(k);
            self.buf.extend_from_slice(&6u32.to_le_bytes());
            self.buf.extend_from_slice(&v.to_le_bytes());
            self.n_kv += 1;
        }
        fn kv_bool(&mut self, k: &str, v: bool) {
            self.s(k);
            self.buf.extend_from_slice(&7u32.to_le_bytes());
            self.buf.push(v as u8);
            self.n_kv += 1;
        }
        fn kv_str(&mut self, k: &str, v: &str) {
            self.s(k);
            self.buf.extend_from_slice(&8u32.to_le_bytes());
            self.s(v);
            self.n_kv += 1;
        }
        fn kv_i32_array(&mut self, k: &str, v: &[i32]) {
            self.s(k);
            self.buf.extend_from_slice(&9u32.to_le_bytes());
            self.buf.extend_from_slice(&5u32.to_le_bytes());
            self.buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                self.buf.extend_from_slice(&x.to_le_bytes());
            }
            self.n_kv += 1;
        }
        fn kv_bool_array(&mut self, k: &str, v: &[bool]) {
            self.s(k);
            self.buf.extend_from_slice(&9u32.to_le_bytes());
            self.buf.extend_from_slice(&7u32.to_le_bytes());
            self.buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                self.buf.push(*x as u8);
            }
            self.n_kv += 1;
        }
        fn finish(mut self) -> Vec<u8> {
            self.buf[16..24].copy_from_slice(&self.n_kv.to_le_bytes());
            self.buf
        }
    }

    /// The common attention keys every synthetic model needs, under `prefix`.
    fn base_keys(b: &mut B, prefix: &str) {
        let k = |s: &str| format!("{prefix}.{s}");
        b.kv_u32(&k("block_count"), 4);
        b.kv_u32(&k("embedding_length"), 64);
        b.kv_u32(&k("feed_forward_length"), 128);
        b.kv_u32(&k("context_length"), 512);
        b.kv_f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
        b.kv_u32(&k("attention.head_count"), 4);
    }

    fn qwen_keys(b: &mut B, prefix: &str) {
        let k = |s: &str| format!("{prefix}.{s}");
        base_keys(b, prefix);
        b.kv_u32(&k("attention.head_count_kv"), 2);
        b.kv_u32(&k("attention.key_length"), 16);
        b.kv_u32(&k("rope.dimension_count"), 8);
        b.kv_f32(&k("rope.freq_base"), 10_000_000.0);
        b.kv_u32(&k("full_attention_interval"), 4);
        b.kv_u32(&k("ssm.group_count"), 2);
        b.kv_u32(&k("ssm.time_step_rank"), 2);
        b.kv_u32(&k("ssm.inner_size"), 32);
        b.kv_u32(&k("ssm.state_size"), 8);
        b.kv_u32(&k("ssm.conv_kernel"), 4);
        b.kv_u32("tokenizer.ggml.eos_token_id", 7);
    }

    #[test_case]
    fn parses_qwen_hybrid_config_dynamically() {
        let mut b = B::new();
        b.kv_str("general.architecture", "qwen35");
        b.kv_str("general.name", "Tiny-Qwen");
        qwen_keys(&mut b, "qwen35");
        let bytes = b.finish();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.arch, "qwen35");
        assert_eq!(g.name, Some("Tiny-Qwen"));
        assert_eq!(g.config.family, Family::QwenHybrid);
        assert_eq!(g.config.block_count, 4);
        let ssm = g.config.ssm.unwrap();
        assert_eq!((ssm.inner, ssm.state, ssm.conv_kernel), (32, 8, 4));
        assert_eq!(ssm.conv_dim(), 8 * 2 * 2 + 32);
        // bos absent (the real 4B has none) → Option::None, eos required.
        assert_eq!(g.config.bos_token_id, None);
        assert_eq!(g.config.eos_token_id, 7);
        // Hybrid layer schedule: (l+1) % 4 == 0.
        assert!(!g.config.is_attention_layer(0));
        assert!(g.config.is_attention_layer(3));
        assert!(!g.config.is_sliding(0));
    }

    #[test_case]
    fn sniffs_renamed_hybrid_finetune_by_key_shape() {
        // A finetune that renamed the arch but keeps the qwen-hybrid key
        // shape (e.g. a repackaged Ornith) must still resolve to QwenHybrid.
        let mut b = B::new();
        b.kv_str("general.architecture", "mystery9b");
        qwen_keys(&mut b, "mystery9b");
        let bytes = b.finish();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.config.family, Family::QwenHybrid);
        assert_eq!(g.arch, "mystery9b");
    }

    #[test_case]
    fn rejects_unknown_architecture() {
        let mut b = B::new();
        b.kv_str("general.architecture", "novel-arch");
        base_keys(&mut b, "novel-arch");
        b.kv_u32("tokenizer.ggml.eos_token_id", 1);
        let bytes = b.finish();
        assert!(matches!(Gguf::parse(&bytes), Err(GgufError::UnsupportedArch)));
    }

    #[test_case]
    fn requires_eos_and_architecture() {
        // No general.architecture at all.
        let b = B::new();
        let bytes = b.finish();
        assert!(matches!(Gguf::parse(&bytes), Err(GgufError::MissingKey("general.architecture"))));
        // Qwen keys but no eos.
        let mut b = B::new();
        b.kv_str("general.architecture", "qwen35");
        base_keys(&mut b, "qwen35");
        let k = |s: &str| format!("qwen35.{s}");
        b.kv_u32(&k("attention.head_count_kv"), 2);
        b.kv_u32(&k("attention.key_length"), 16);
        b.kv_u32(&k("rope.dimension_count"), 8);
        b.kv_u32(&k("ssm.group_count"), 2);
        b.kv_u32(&k("ssm.time_step_rank"), 2);
        b.kv_u32(&k("ssm.inner_size"), 32);
        b.kv_u32(&k("ssm.state_size"), 8);
        b.kv_u32(&k("ssm.conv_kernel"), 4);
        let bytes = b.finish();
        assert!(matches!(Gguf::parse(&bytes), Err(GgufError::MissingKey("tokenizer.ggml.eos_token_id"))));
    }

    #[test_case]
    fn parses_gemma_swa_config() {
        // The real gemma-4-12b shape scaled down: per-layer kv-head array
        // (8 sliding / 1 global), bool sliding pattern, *_swa geometry keys,
        // dual rope bases, logit softcap, gemma4 tokenizer flavor.
        let mut b = B::new();
        b.kv_str("general.architecture", "gemma4");
        base_keys(&mut b, "gemma4");
        b.kv_u32("gemma4.attention.sliding_window", 128);
        b.kv_bool_array("gemma4.attention.sliding_window_pattern", &[true, true, true, false]);
        b.kv_i32_array("gemma4.attention.head_count_kv", &[8, 8, 8, 1]);
        b.kv_u32("gemma4.attention.key_length", 32);
        b.kv_u32("gemma4.attention.key_length_swa", 16);
        b.kv_u32("gemma4.rope.dimension_count", 32);
        b.kv_u32("gemma4.rope.dimension_count_swa", 16);
        b.kv_f32("gemma4.rope.freq_base", 1_000_000.0);
        b.kv_f32("gemma4.rope.freq_base_swa", 10_000.0);
        b.kv_f32("gemma4.final_logit_softcapping", 30.0);
        b.kv_u32("tokenizer.ggml.bos_token_id", 2);
        b.kv_u32("tokenizer.ggml.eos_token_id", 106);
        b.kv_bool("tokenizer.ggml.add_bos_token", true);
        b.kv_str("tokenizer.ggml.model", "gemma4");
        let bytes = b.finish();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.config.family, Family::Gemma4);
        assert!(g.config.ssm.is_none());
        // Base geometry = sliding-layer values; global overrides in SwaConfig.
        assert_eq!(g.config.head_count_kv, 8);
        assert_eq!(g.config.head_dim, 16);
        assert_eq!(g.config.rope_dim, 16);
        assert_eq!(g.config.rope_freq_base, 10_000.0);
        let swa = g.config.swa.unwrap();
        assert_eq!(swa.window, 128);
        assert_eq!(swa.pattern, 0b0111);
        assert_eq!(swa.head_count_kv_global, 1);
        assert_eq!(swa.head_dim_global, 32);
        assert_eq!(swa.freq_base_global, 1_000_000.0);
        assert_eq!(swa.logit_softcap, 30.0);
        // Every layer is attention; only 0..3 slide.
        assert!(g.config.is_attention_layer(0) && g.config.is_attention_layer(3));
        assert!(g.config.is_sliding(0) && !g.config.is_sliding(3));
        assert_eq!(g.config.bos_token_id, Some(2));
        assert!(g.config.add_bos);
        assert_eq!(g.tokenizer_model, "gemma4");
    }

    #[test_case]
    fn rejects_inconsistent_swa_kv_array() {
        // kv-head count varying *within* the sliding kind is unsupported.
        let mut b = B::new();
        b.kv_str("general.architecture", "gemma4");
        base_keys(&mut b, "gemma4");
        b.kv_u32("gemma4.attention.sliding_window", 128);
        b.kv_bool_array("gemma4.attention.sliding_window_pattern", &[true, true, true, false]);
        b.kv_i32_array("gemma4.attention.head_count_kv", &[8, 4, 8, 1]);
        b.kv_u32("gemma4.attention.key_length", 32);
        b.kv_u32("gemma4.attention.key_length_swa", 16);
        b.kv_u32("tokenizer.ggml.eos_token_id", 106);
        let bytes = b.finish();
        assert!(matches!(Gguf::parse(&bytes), Err(GgufError::BadValue(_))));
    }

    /// Gemma 4 E4B (unsloth Q4_K_M) ships `attention.head_count_kv` as a
    /// *scalar* U32 (same n_kv for every layer), not a per-layer I32 array.
    /// Parse must accept that form — the silent failure mode for
    /// `make run-uefi MODEL=e4b` before this landed.
    #[test_case]
    fn parses_gemma_scalar_head_count_kv() {
        let mut b = B::new();
        b.kv_str("general.architecture", "gemma4");
        base_keys(&mut b, "gemma4");
        b.kv_u32("gemma4.attention.sliding_window", 512);
        b.kv_bool_array("gemma4.attention.sliding_window_pattern", &[true, true, true, false]);
        // Scalar, not array — the E4B shape.
        b.kv_u32("gemma4.attention.head_count_kv", 2);
        b.kv_u32("gemma4.attention.key_length", 512);
        b.kv_u32("gemma4.attention.key_length_swa", 256);
        b.kv_u32("gemma4.rope.dimension_count", 512);
        b.kv_u32("gemma4.rope.dimension_count_swa", 256);
        b.kv_f32("gemma4.rope.freq_base", 1_000_000.0);
        b.kv_f32("gemma4.rope.freq_base_swa", 10_000.0);
        b.kv_f32("gemma4.final_logit_softcapping", 30.0);
        b.kv_u32("tokenizer.ggml.eos_token_id", 1);
        b.kv_str("tokenizer.ggml.model", "gemma4");
        let bytes = b.finish();
        let g = Gguf::parse(&bytes).expect("scalar head_count_kv must parse");
        assert_eq!(g.config.family, Family::Gemma4);
        assert_eq!(g.config.head_count_kv, 2); // sliding-layer base
        let swa = g.config.swa.unwrap();
        assert_eq!(swa.head_count_kv_global, 2); // same scalar for global
        assert_eq!(swa.head_dim_global, 512);
        assert_eq!(g.config.head_dim, 256); // swa key_length
        assert!(g.config.is_sliding(0) && !g.config.is_sliding(3));
    }
}
