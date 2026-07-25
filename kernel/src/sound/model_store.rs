//! Runtime store for the **large** voice models (parakeet STT ~131 MB, Kitten
//! TTS ~78 MB) that are too big to `include_bytes!` into the kernel. They are
//! loaded on demand from the filesystem (`/voice models load`), where an
//! installed system keeps them on its ext4 data partition. Until a model is
//! loaded these return `None`, and the pipelines report their front-end status
//! rather than failing.
//!
//! The bytes are leaked to `'static` on load so the zero-copy [`crate::onnx`]
//! reader can borrow tensor `raw_data` for the model's lifetime. Parsed
//! [`crate::onnx::Model`] graphs are cached after the first successful parse
//! so `/voice say` (multi-chunk) and repeated STT do not re-walk a 78–131 MB
//! protobuf on every utterance.

use crate::mm::Locked;
use crate::onnx::Model;

static PARAKEET: Locked<Option<&'static [u8]>> = Locked::new(None);
static KITTEN: Locked<Option<&'static [u8]>> = Locked::new(None);
static PARAKEET_MODEL: Locked<Option<&'static Model<'static>>> = Locked::new(None);
static KITTEN_MODEL: Locked<Option<&'static Model<'static>>> = Locked::new(None);

/// The loaded parakeet STT model bytes, if any.
pub fn parakeet() -> Option<&'static [u8]> {
    PARAKEET.with(|p| *p)
}

/// The loaded Kitten TTS model bytes, if any.
pub fn kitten() -> Option<&'static [u8]> {
    KITTEN.with(|k| *k)
}

/// Parsed Kitten graph (cached). Prefer this over re-`parse`ing every synth.
pub fn kitten_model() -> Option<&'static Model<'static>> {
    ensure_parsed("kitten")
}

/// Parsed parakeet graph (cached).
pub fn parakeet_model() -> Option<&'static Model<'static>> {
    ensure_parsed("parakeet")
}

fn ensure_parsed(which: &str) -> Option<&'static Model<'static>> {
    let (bytes_slot, model_slot) = match which {
        "kitten" => (&KITTEN, &KITTEN_MODEL),
        "parakeet" => (&PARAKEET, &PARAKEET_MODEL),
        _ => return None,
    };
    model_slot.with(|ms| {
        if let Some(m) = *ms {
            return Some(m);
        }
        let bytes = bytes_slot.with(|b| *b)?;
        let parsed = crate::onnx::parse(bytes)?;
        // Model only borrows the leaked model bytes; both live for the rest of
        // the boot. Leaking the Model keeps a stable `'static` ref for reuse.
        let leaked: &'static Model<'static> =
            alloc::boxed::Box::leak(alloc::boxed::Box::new(parsed));
        *ms = Some(leaked);
        crate::ktrace::log_fmt(format_args!("model_store: parsed {which} graph (cached)"));
        Some(leaked)
    })
}

/// Store pre-read model `bytes` under `which` (`"parakeet"` or `"kitten"`),
/// leaking them to `'static` for the zero-copy ONNX reader. The shell reads the
/// file from a mounted disk (`read_mounted`) and hands the bytes here.
/// Clears any prior parse cache for that model.
pub fn load_bytes(which: &str, bytes: alloc::vec::Vec<u8>) -> Result<usize, &'static str> {
    let leaked: &'static [u8] = alloc::boxed::Box::leak(bytes.into_boxed_slice());
    let n = leaked.len();
    match which {
        "parakeet" => {
            PARAKEET.with(|p| *p = Some(leaked));
            PARAKEET_MODEL.with(|m| *m = None);
        }
        "kitten" => {
            KITTEN.with(|k| *k = Some(leaked));
            KITTEN_MODEL.with(|m| *m = None);
        }
        _ => return Err("unknown model (parakeet|kitten)"),
    }
    crate::ktrace::log_fmt(format_args!("model_store: loaded {which} ({n} bytes)"));
    Ok(n)
}
