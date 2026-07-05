//! Runtime store for the **large** voice models (parakeet STT ~131 MB, Kitten
//! TTS ~78 MB) that are too big to `include_bytes!` into the kernel. They are
//! loaded on demand from the filesystem (`/voice models load`), where an
//! installed system keeps them on its ext4 data partition. Until a model is
//! loaded these return `None`, and the pipelines report their front-end status
//! rather than failing.
//!
//! The bytes are leaked to `'static` on load so the zero-copy [`crate::onnx`]
//! reader can borrow tensor `raw_data` for the model's lifetime.

use crate::mm::Locked;

static PARAKEET: Locked<Option<&'static [u8]>> = Locked::new(None);
static KITTEN: Locked<Option<&'static [u8]>> = Locked::new(None);

/// The loaded parakeet STT model bytes, if any.
pub fn parakeet() -> Option<&'static [u8]> {
    PARAKEET.with(|p| *p)
}

/// The loaded Kitten TTS model bytes, if any.
pub fn kitten() -> Option<&'static [u8]> {
    KITTEN.with(|k| *k)
}

/// Load a voice model from a mounted filesystem path into the store. Returns
/// the byte length on success. `which` is `"parakeet"` or `"kitten"`.
pub fn load(which: &str, path: &str) -> Result<usize, &'static str> {
    let bytes = crate::synapse::fs::read(path).ok_or("model file not found")?;
    let leaked: &'static [u8] = alloc::boxed::Box::leak(bytes.into_boxed_slice());
    let n = leaked.len();
    match which {
        "parakeet" => PARAKEET.with(|p| *p = Some(leaked)),
        "kitten" => KITTEN.with(|k| *k = Some(leaked)),
        _ => return Err("unknown model (parakeet|kitten)"),
    }
    crate::ktrace::log_fmt(format_args!("model_store: loaded {which} ({n} bytes) from {path}"));
    Ok(n)
}
