//! **Page WASM** — instantiate + call with a limited host import surface.
//!
//! Reference: Ladybird LibWasm / WebAssembly JS API (subset).
//! Uses [`crate::agent::wasm_rt`] (wasmi). **No ambient FS** — only pure
//! compute imports: `env.abort`, `env.log` (length-capped), `wasi_snapshot_preview1`
//! stubs that return 0 / ENOSYS.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const MAX_MODULE_BYTES: usize = 512 * 1024;
pub const PAGE_FUEL: u64 = 50_000_000;

#[derive(Clone, Debug)]
pub struct PageModule {
    pub url: String,
    pub bytes: Vec<u8>,
}

/// Instantiated page module (bytes + optional last log from host import).
#[derive(Clone, Debug)]
pub struct PageInstance {
    pub module: PageModule,
    pub last_log: String,
}

pub fn is_wasm_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0..4] == [0x00, b'a', b's', b'm']
}

pub fn validate(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.len() > MAX_MODULE_BYTES {
        return Err("wasm module too large");
    }
    if !is_wasm_magic(bytes) {
        return Err("not a wasm module");
    }
    crate::agent::wasm_abi::validate_wasm_module(bytes)
}

/// Instantiate: validate only (wasmi instance is created per-call for fuel isolation).
pub fn instantiate(url: &str, bytes: &[u8]) -> Result<PageInstance, String> {
    validate(bytes).map_err(|e| String::from(e))?;
    Ok(PageInstance {
        module: PageModule {
            url: url.to_string(),
            bytes: bytes.to_vec(),
        },
        last_log: String::new(),
    })
}

/// Call a string-in/string-out export on the **page** host-import surface
/// (no agent storage/UI/sound — see [`crate::agent::wasm_rt::call_string_page`]).
pub fn call_export(
    module_bytes: &[u8],
    export: &str,
    arg: &str,
) -> Result<String, String> {
    validate(module_bytes).map_err(|e| String::from(e))?;
    crate::agent::wasm_rt::call_string_page(
        module_bytes,
        export,
        arg,
        crate::agent::wasm_rt::Limits::default().with_fuel(PAGE_FUEL),
    )
    .map_err(|e| format!("{e}"))
}

/// Call via instance handle.
pub fn instance_call(inst: &PageInstance, export: &str, arg: &str) -> Result<String, String> {
    call_export(&inst.module.bytes, export, arg)
}

/// Host import names we acknowledge (for diagnostics / future wasmi linker).
pub fn supported_imports() -> &'static [&'static str] {
    &[
        "env.abort",
        "env.log",
        "env.seed",
        "wasi_snapshot_preview1.fd_write",
        "wasi_snapshot_preview1.fd_close",
        "wasi_snapshot_preview1.environ_get",
        "wasi_snapshot_preview1.environ_sizes_get",
        "wasi_snapshot_preview1.proc_exit",
    ]
}

pub fn is_supported_import(module: &str, field: &str) -> bool {
    let key = format!("{module}.{field}");
    supported_imports().iter().any(|s| *s == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn magic_and_reject_large() {
        assert!(is_wasm_magic(&[0x00, b'a', b's', b'm', 1, 0, 0, 0]));
        assert!(!is_wasm_magic(b"notwasm"));
        let big = alloc::vec![0u8; MAX_MODULE_BYTES + 1];
        assert!(validate(&big).is_err());
    }

    #[test_case]
    fn imports_list() {
        assert!(is_supported_import("env", "abort"));
        assert!(is_supported_import("wasi_snapshot_preview1", "fd_write"));
        assert!(!is_supported_import("env", "eval"));
    }

    #[test_case]
    fn instantiate_rejects_garbage() {
        assert!(instantiate("x.wasm", b"nope").is_err());
    }
}
