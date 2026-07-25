//! Catalogue for the `/agents` searchable browser (same shape as
//! [`super::catalog`] for `/help`): category headers + selectable items with a
//! kind badge (`ui/canvas` vs `shell`).

use crate::agent::system::{self, AgentUiClass};
use crate::shell::catalog::{self, Row};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Safe short prefix of `s` by **chars**, never panicking on UTF-8 boundaries.
fn take_chars(s: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max {
            out.push_str("...");
            break;
        }
        out.push(ch);
    }
    out
}

/// Build filtered rows for the agents popup.
///
/// Groups (UI first so Enter opens a canvas by default, not a process switch):
/// * **UI (canvas)** — installable package-UI apps
/// * **Shell** — chat/SOUL packages (and content agents)
/// * **Running** — live scheduler tasks with agent identity
pub fn filter_rows(query: &str) -> Vec<Row> {
    let q = query.trim().to_ascii_lowercase();
    let mut out = Vec::new();
    let running_ui = crate::service::package_ui::running_names();

    // --- Installable system agents (UI first — this is an app launcher) ---
    let mut ui_items: Vec<Row> = Vec::new();
    let mut shell_items: Vec<Row> = Vec::new();
    for (name, agent_id) in system::list() {
        let class = system::ui_class(name);
        let desc = system::description_of(name);
        let hay = format!("{name} {desc}").to_ascii_lowercase();
        if !q.is_empty() && !hay.contains(&q) {
            continue;
        }
        let badge = system::ui_class_label(class);
        let live = match class {
            AgentUiClass::UiCanvas if running_ui.iter().any(|n| n == name) => " run",
            _ => "",
        };
        // Keep titles short and ASCII-safe for the modal paint path.
        let title = if desc.is_empty() {
            name.to_string()
        } else {
            format!("{name} - {}", take_chars(&desc, 36))
        };
        let item = Row::Item {
            title,
            // Encoded pick: `ui:name` or `shell:name`
            name: match class {
                AgentUiClass::UiCanvas => format!("ui:{name}"),
                AgentUiClass::Shell => format!("shell:{name}"),
            },
            shortcut: format!("{badge}{live} id{agent_id}"),
        };
        match class {
            AgentUiClass::UiCanvas => ui_items.push(item),
            AgentUiClass::Shell => shell_items.push(item),
        }
    }
    if !ui_items.is_empty() {
        out.push(Row::Header(String::from("UI (canvas)")));
        out.extend(ui_items);
    }
    if !shell_items.is_empty() {
        out.push(Row::Header(String::from("Shell agents")));
        out.extend(shell_items);
    }

    // --- Running processes (after apps — avoid mis-Enter on orchestrator) ---
    // Skip bootstrap (#0): switching to it is not useful.
    let mut running_items: Vec<Row> = Vec::new();
    for (id, name, state) in crate::sched::list() {
        if name == "bootstrap" || id == 0 {
            continue;
        }
        let title = format!("{name}  #{id}");
        let hay = format!("{title} {state}").to_ascii_lowercase();
        if !q.is_empty() && !hay.contains(&q) {
            continue;
        }
        let kind = if name == "orchestrator" || id == 1 {
            "shell"
        } else if name.starts_with("pkg-") {
            "ui/canvas"
        } else {
            "process"
        };
        running_items.push(Row::Item {
            title,
            // Encoded pick: `switch:<id>`
            name: format!("switch:{id}"),
            shortcut: format!("{kind} {state}"),
        });
    }
    if !running_items.is_empty() {
        out.push(Row::Header(String::from("Running")));
        out.extend(running_items);
    }

    out
}

/// First selectable row, preferring a `ui:` canvas app when present.
pub fn first_sel(rows: &[Row]) -> usize {
    // Prefer the first canvas app so Enter opens UI without arrowing past
    // headers / processes.
    for (i, r) in rows.iter().enumerate() {
        if let Row::Item { name, .. } = r {
            if name.starts_with("ui:") {
                return i;
            }
        }
    }
    catalog::first_sel(rows)
}

// Re-export nav helpers so the modal can share catalog::move_sel etc.
// `first_sel` is defined above (prefers ui: canvas rows).
pub use catalog::{clamp_scroll, move_sel, name_at};

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn filter_has_ui_and_shell_groups() {
        let rows = filter_rows("");
        assert!(
            !rows.is_empty(),
            "agents catalog must not be empty"
        );
        assert!(
            rows.iter()
                .any(|r| matches!(r, Row::Header(h) if h.contains("UI"))),
            "expected UI header"
        );
        assert!(
            rows.iter()
                .any(|r| matches!(r, Row::Header(h) if h.contains("Shell"))),
            "expected Shell header"
        );
        assert!(
            rows.iter()
                .any(|r| matches!(r, Row::Item { name, .. } if name == "ui:chess")),
            "chess should be ui:chess"
        );
        assert!(
            rows.iter()
                .any(|r| matches!(r, Row::Item { name, .. } if name == "shell:notes")),
            "notes should be shell:notes"
        );
        // UI section must come before Running so default Enter starts a canvas.
        let ui_pos = rows.iter().position(|r| matches!(r, Row::Header(h) if h.contains("UI")));
        let run_pos = rows.iter().position(|r| matches!(r, Row::Header(h) if h.contains("Running")));
        if let (Some(u), Some(r)) = (ui_pos, run_pos) {
            assert!(u < r, "UI (canvas) group must precede Running");
        }
        let sel = first_sel(&rows);
        assert!(
            matches!(name_at(&rows, sel), Some(n) if n.starts_with("ui:")),
            "first_sel must prefer a ui: canvas row, got {:?}",
            name_at(&rows, sel)
        );
    }

    #[test_case]
    fn search_filters_by_name() {
        let rows = filter_rows("chess");
        assert!(
            rows.iter()
                .any(|r| matches!(r, Row::Item { name, .. } if name.contains("chess"))),
            "chess match"
        );
        assert!(
            !rows
                .iter()
                .any(|r| matches!(r, Row::Item { name, .. } if name.contains("notes"))),
            "notes should not match chess query"
        );
    }

    #[test_case]
    fn take_chars_never_panics_on_utf8() {
        let s = "café—unicode";
        let t = take_chars(s, 3);
        assert!(t.starts_with("caf"));
    }
}
