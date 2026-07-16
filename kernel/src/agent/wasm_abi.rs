//! **WASM tool ABI (design + host stubs)** — agent-authored tools.
//!
//! # Why
//!
//! App-specific `match` arms in kernel runtimes do not scale. Agents ship
//! deterministic guest code (`assets/tools.wasm`) that implements tools like
//! `chess_legal`, while **all effects** (draw, storage) go through capability-
//! gated **host imports** — same determinism boundary as Synapse. The generic
//! runtime is `service::package_ui` (one persistent instance per running app).
//!
//! # Call graph
//!
//! ```text
//!   model tool_call ──▶ ToolBackend::Wasm { export }
//!                            │
//!                            ▼
//!                      wasmi instance (fuel, memory cap)   [W1+]
//!                            │ host imports only
//!              ┌─────────────┼─────────────┐
//!              ▼             ▼             ▼
//!         storage_*     board_set      ui_draw / log
//!         (this crate)  (Synapse)      (Synapse)
//! ```
//!
//! # String ABI v1 (planned)
//!
//! Guest exports used as tools take UTF-8 args via linear memory:
//!
//! * Host writes args at guest heap, passes `(ptr, len)` i32 pairs
//! * Guest returns `(ptr, len)` to a result string in its memory
//! * Host copies result out before the next call
//!
//! Host imports (guest → kernel), all checked against the binding `TaskId`:
//!
//! | Import | Effect |
//! |--------|--------|
//! | `host_storage_get/set/list/remove` | [`crate::agent::storage`] |
//! | `host_board_set` / `host_board_mark` / `host_ui_draw` | Synapse UI |
//! | `host_surface_id` | active UI surface for this agent |
//! | `host_now_ms` / `host_log` | clock + rate-limited ktrace |
//!
//! # Security
//!
//! * Modules load at **install/start** only (never from model output)
//! * Fuel + memory page limits
//! * No ambient FS/net; no raw framebuffer
//! * Surface ownership enforced on every draw import
//!
//! # Status
//!
//! * **W0 (this module):** [`ToolBackend`] table + host storage (no interpreter)
//! * **W1:** wasmi load/call via [`crate::agent::wasm_rt`] (fuel + memory limits)
//! * **W2+:** host imports + package `wasm.exports` registration

use crate::agent::wasm_rt::{self, HostBindings, Limits};
use alloc::string::String;

/// Invoke a WASM tool export with the string ABI (`(i32,i32)->(i32,i32)`).
///
/// `module_bytes` is the raw `.wasm` payload (loaded from the agent package at
/// install/start — never from model output). `fuel == 0` uses
/// [`wasm_rt::DEFAULT_FUEL`]. Host imports use `bind` (agent id + UI task).
pub fn call_wasm_export(
    module_bytes: &[u8],
    export: &str,
    args_json: &str,
    fuel: u64,
    bind: HostBindings,
) -> Result<String, &'static str> {
    wasm_rt::call_string_bound(
        module_bytes,
        export,
        args_json,
        Limits::default().with_fuel(fuel),
        bind,
    )
}

/// Compile-only check used at package install time.
pub fn validate_wasm_module(module_bytes: &[u8]) -> Result<(), &'static str> {
    wasm_rt::validate_module(module_bytes)
}
