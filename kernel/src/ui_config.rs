//! UI / shortcuts configuration, persisted as JSON under `/configs/core/`
//! (`ui.json`, `shortcuts.json`) in the Synapse store. Phase 1 provides just the
//! status-bar text resolution + timezone persistence hook; Phase 2 fills in the
//! full JSON schema (pane layout, statusbar templates with variables) and the
//! `/ui` and `/shortcuts` editors.

use alloc::format;
use alloc::string::String;

/// Resolve the status-bar `(left, right)` strings. Phase 1: brand + datetime;
/// Phase 2 resolves configurable templates with `${var}` substitution.
pub fn status_strings() -> (String, String) {
    let left = format!("Chitti OS v{}", crate::VERSION);
    let right = format!("{}  {}", crate::clock::format_datetime(), crate::clock::format_tz());
    (left, right)
}

/// Persist the timezone offset so it survives a reboot. Phase 1: no-op (the
/// clock keeps it in memory); Phase 2 writes it into `/configs/core/ui.json`.
pub fn persist_tz(_offset_secs: i32) {}
