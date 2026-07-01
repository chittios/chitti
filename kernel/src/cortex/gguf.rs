//! Minimal GGUF (v3) parser (`CHITTI_OS_HANDOFF.md` Phase 3). Reads the
//! Qwen2.5 model Limine hands us as a boot module, entirely zero-copy:
//! tensor data and token strings are returned as slices *into* the module
//! memory (reachable through the HHDM), never copied onto the heap -- the
//! model is ~675 MiB and copying it would be absurd.
//!
//! Only the subset Cortex needs is parsed: the numeric config, the tensor
//! directory (name → dims/type/offset), and the token table (id → string,
//! for detokenizing output). Merges and the chat template are skipped.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element types (the subset present in a Q8_0 Qwen2 GGUF).
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q8_0: u32 = 8;

#[derive(Debug)]
pub enum GgufError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    BadString,
    UnknownValueType(u32),
    MissingKey(&'static str),
    MissingTensor,
}

/// The numeric hyperparameters Cortex's forward pass needs, pulled from
/// the `qwen35.*` / `tokenizer.*` metadata keys. Qwen3.5-0.8B is a hybrid:
/// most layers are gated-DeltaNet (linear attention / SSM), and every
/// `full_attn_interval`-th layer is full attention with QK-norm.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub context_length: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// Full-attention layers occur where `(layer + 1) % full_attn_interval == 0`.
    pub full_attn_interval: usize,
    // --- full-attention layers ---
    pub head_count: usize,    // query heads (8)
    pub head_count_kv: usize, // kv heads (2)
    pub head_dim: usize,      // attention head dim = key_length (256)
    pub rope_dim: usize,      // rotary dims per head (partial RoPE, 64)
    // --- gated-DeltaNet (SSM) layers ---
    pub ssm_n_group: usize,  // linear-attn key heads (16)
    pub ssm_dt_rank: usize,  // linear-attn value heads (16)
    pub ssm_inner: usize,    // value_dim (2048)
    pub ssm_state: usize,    // per-head state / key dim (128)
    pub ssm_conv_kernel: usize, // causal conv width (4)
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Config {
    /// Is layer `l` a full-attention layer (vs a gated-DeltaNet layer)?
    pub fn is_attention_layer(&self, l: usize) -> bool {
        (l + 1) % self.full_attn_interval == 0
    }
    /// Width of the DeltaNet qkv projection / conv (`key_dim*2 + value_dim`).
    pub fn ssm_conv_dim(&self) -> usize {
        self.ssm_state * self.ssm_n_group * 2 + self.ssm_inner
    }
    pub fn ssm_head_dim(&self) -> usize {
        self.ssm_inner / self.ssm_dt_rank
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
    tensors: BTreeMap<&'a str, TensorInfo>,
    /// Token id → token string (GPT-2 byte-level encoding), sliced from the
    /// module. Empty if the model carried no token table.
    pub tokens: Vec<&'a str>,
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

/// A GGUF metadata value, only enough to read the scalars/arrays we need.
enum Value<'a> {
    U32(u32),
    U64(u64),
    F32(f32),
    /// Array of token strings (the only array type we retain).
    StrArray(Vec<&'a str>),
    /// Any other value/array, skipped (cursor already advanced past it).
    Other,
}

fn read_value<'a>(c: &mut Cursor<'a>, value_type: u32) -> Result<Value<'a>, GgufError> {
    Ok(match value_type {
        0 | 1 | 7 => {
            c.take(1)?;
            Value::Other
        } // u8/i8/bool
        2 | 3 => {
            c.take(2)?;
            Value::Other
        } // u16/i16
        4 => Value::U32(c.u32()?),
        5 => Value::U32(c.u32()? as u32), // i32 read as bits; only used for ids
        6 => Value::F32(c.f32()?),
        10 => Value::U64(c.u64()?),
        11 => Value::U64(c.u64()?), // i64
        12 => {
            c.take(8)?;
            Value::Other
        } // f64
        8 => {
            c.gstr()?;
            Value::Other
        }
        9 => {
            let elem_type = c.u32()?;
            let count = c.u64()? as usize;
            if elem_type == 8 {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(c.gstr()?);
                }
                Value::StrArray(v)
            } else {
                // Skip a fixed-width-element array without materializing it.
                for _ in 0..count {
                    read_value(c, elem_type)?;
                }
                Value::Other
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
    /// tensors + config + token table. Zero-copy: all returned slices
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
        let mut alignment = DEFAULT_ALIGNMENT;
        let mut tokens: Vec<&str> = Vec::new();

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
                Value::StrArray(v) if key == "tokenizer.ggml.tokens" => tokens = v,
                _ => {}
            }
        }

        let u32_of = |k: &'static str| get_u32.get(k).copied().ok_or(GgufError::MissingKey(k));
        let config = Config {
            block_count: u32_of("qwen35.block_count")? as usize,
            embedding_length: u32_of("qwen35.embedding_length")? as usize,
            feed_forward_length: u32_of("qwen35.feed_forward_length")? as usize,
            context_length: u32_of("qwen35.context_length")? as usize,
            rms_eps: get_f32
                .get("qwen35.attention.layer_norm_rms_epsilon")
                .copied()
                .ok_or(GgufError::MissingKey("qwen35.attention.layer_norm_rms_epsilon"))?,
            rope_freq_base: get_f32.get("qwen35.rope.freq_base").copied().unwrap_or(10_000_000.0),
            full_attn_interval: u32_of("qwen35.full_attention_interval").unwrap_or(4) as usize,
            head_count: u32_of("qwen35.attention.head_count")? as usize,
            head_count_kv: u32_of("qwen35.attention.head_count_kv")? as usize,
            head_dim: u32_of("qwen35.attention.key_length")? as usize,
            rope_dim: u32_of("qwen35.rope.dimension_count")? as usize,
            ssm_n_group: u32_of("qwen35.ssm.group_count")? as usize,
            ssm_dt_rank: u32_of("qwen35.ssm.time_step_rank")? as usize,
            ssm_inner: u32_of("qwen35.ssm.inner_size")? as usize,
            ssm_state: u32_of("qwen35.ssm.state_size")? as usize,
            ssm_conv_kernel: u32_of("qwen35.ssm.conv_kernel")? as usize,
            bos_token_id: u32_of("tokenizer.ggml.bos_token_id").unwrap_or(151643),
            eos_token_id: u32_of("tokenizer.ggml.eos_token_id").unwrap_or(151645),
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

        Ok(Self { data, tensor_data_base, config, tensors, tokens })
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
