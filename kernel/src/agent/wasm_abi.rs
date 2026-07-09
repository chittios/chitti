//! **WASM tool ABI (design + host stubs)** — agent-authored tools.
//!
//! # Why
//!
//! Chess-specific `match` arms in `ui_agent` do not scale. Agents should ship
//! deterministic guest code (`assets/tools.wasm`) that implements tools like
//! `chess_legal`, while **all effects** (draw, storage) go through capability-
//! gated **host imports** — same determinism boundary as Synapse.
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
use alloc::vec::Vec;

/// How a named tool is implemented for a running UI / package agent.
#[derive(Clone, Debug)]
pub enum ToolBackend {
    /// Built-in host tool (Synapse primitive or storage API).
    Host {
        /// Canonical tool name (`board_set`, `storage_get`, …).
        name: String,
    },
    /// Guest export (W1+). `module_path` is under the agent home.
    Wasm {
        name: String,
        /// e.g. `assets/tools.wasm`
        module_path: String,
        /// Export function name inside the module.
        export: String,
        /// Instruction fuel budget for one call (0 = engine default).
        fuel: u64,
    },
}

impl ToolBackend {
    pub fn name(&self) -> &str {
        match self {
            ToolBackend::Host { name } | ToolBackend::Wasm { name, .. } => name,
        }
    }
}

/// Default host tool table for UI agents (generic surface + storage).
pub fn default_ui_host_tools() -> Vec<ToolBackend> {
    [
        "ui_surface_request",
        "ui_draw",
        "ui_event_poll",
        "ui_surface_close",
        "board_set",
        "board_mark",
        "storage_get",
        "storage_set",
        "storage_list",
        "storage_remove",
        "memory_add",
        "memory_get",
        "memory_list",
    ]
    .into_iter()
    .map(|n| ToolBackend::Host { name: n.into() })
    .collect()
}

/// Chess package tool table: host UI/storage + WASM rule exports.
pub fn chess_package_tools() -> Vec<ToolBackend> {
    let mut t = default_ui_host_tools();
    t.push(ToolBackend::Wasm {
        name: "chess_legal".into(),
        module_path: "assets/tools.wasm".into(),
        export: "chess_legal".into(),
        fuel: 2_000_000,
    });
    t.push(ToolBackend::Wasm {
        name: "chess_try_move".into(),
        module_path: "assets/tools.wasm".into(),
        export: "chess_try_move".into(),
        fuel: 2_000_000,
    });
    t
}

/// Resolve a tool name against a backend table.
pub fn lookup<'a>(table: &'a [ToolBackend], name: &str) -> Option<&'a ToolBackend> {
    table.iter().find(|b| b.name() == name)
}

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
