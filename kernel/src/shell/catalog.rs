//! Categorised command catalogue for the **Commands** help browser
//! (searchable, scrollable modal opened by `/help`).
//!
//! Pure data + filter/scroll helpers — no framebuffer. Keep entries in sync
//! with `COMMANDS` in `shell/mod.rs` and the slash-suggest list.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One installable / slash command in the browser.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub category: &'static str,
    /// Friendly title shown on the left (e.g. "Compact History").
    pub title: &'static str,
    /// Canonical command name without `/` (e.g. `compact`).
    pub name: &'static str,
    /// Optional shortcut hint (e.g. `Ctrl+W`), else empty.
    pub shortcut: &'static str,
}

/// A rendered row in the filtered list (category header or command).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Header(String),
    Item {
        title: String,
        name: String,
        shortcut: String,
    },
}

/// Full catalogue — grouped like the Grok-style Commands palette.
pub const ENTRIES: &[Entry] = &[
    // Session
    Entry { category: "Session", title: "Session Info", name: "session", shortcut: "" },
    Entry { category: "Session", title: "Clear Chat", name: "clear", shortcut: "" },
    Entry { category: "Session", title: "Compact History", name: "compact", shortcut: "" },
    Entry { category: "Session", title: "Agents", name: "agents", shortcut: "" },
    Entry { category: "Session", title: "Exit / Power Off", name: "exit", shortcut: "Ctrl+D" },
    Entry { category: "Session", title: "Restart", name: "restart", shortcut: "" },
    // Context & memory
    Entry { category: "Context", title: "Memory", name: "memory", shortcut: "" },
    Entry { category: "Context", title: "Todos", name: "todos", shortcut: "" },
    Entry { category: "Context", title: "Skills", name: "skills", shortcut: "" },
    Entry { category: "Context", title: "Permissions", name: "permissions", shortcut: "" },
    // Model & agent
    Entry { category: "Model & Agent", title: "Switch Model", name: "model", shortcut: "" },
    Entry { category: "Model & Agent", title: "Approval Mode", name: "mode", shortcut: "" },
    Entry { category: "Model & Agent", title: "Thinking", name: "think", shortcut: "" },
    Entry { category: "Model & Agent", title: "Infer (parity)", name: "infer", shortcut: "" },
    Entry { category: "Model & Agent", title: "Perf Benchmark", name: "perf", shortcut: "" },
    Entry { category: "Model & Agent", title: "Matvec Bench", name: "bench", shortcut: "" },
    // Files
    Entry { category: "Files", title: "List Directory", name: "ls", shortcut: "" },
    Entry { category: "Files", title: "Print File", name: "cat", shortcut: "" },
    Entry { category: "Files", title: "Open / Edit / Play", name: "open", shortcut: "" },
    Entry { category: "Files", title: "Make Directory", name: "mkdir", shortcut: "" },
    Entry { category: "Files", title: "Copy", name: "cp", shortcut: "" },
    Entry { category: "Files", title: "Move / Rename", name: "mv", shortcut: "" },
    Entry { category: "Files", title: "Remove", name: "rm", shortcut: "" },
    Entry { category: "Files", title: "Touch File", name: "touch", shortcut: "" },
    Entry { category: "Files", title: "Glob Paths", name: "glob", shortcut: "" },
    Entry { category: "Files", title: "Grep Contents", name: "grep", shortcut: "" },
    Entry { category: "Files", title: "Working Directory", name: "pwd", shortcut: "" },
    // Storage
    Entry { category: "Storage", title: "Disks", name: "disks", shortcut: "" },
    Entry { category: "Storage", title: "Mounts", name: "mounts", shortcut: "" },
    Entry { category: "Storage", title: "Mount Volume", name: "mount", shortcut: "" },
    Entry { category: "Storage", title: "Unmount", name: "umount", shortcut: "" },
    Entry { category: "Storage", title: "Install Chitti", name: "install", shortcut: "" },
    Entry { category: "Storage", title: "Format ext4", name: "mkext4", shortcut: "" },
    // Network
    Entry { category: "Network", title: "Network Status", name: "network", shortcut: "" },
    Entry { category: "Network", title: "Ping Host", name: "ping", shortcut: "" },
    Entry { category: "Network", title: "Wi-Fi", name: "wifi", shortcut: "" },
    Entry { category: "Network", title: "HTTP Client", name: "http", shortcut: "" },
    Entry { category: "Network", title: "WebSocket", name: "ws", shortcut: "" },
    Entry { category: "Network", title: "MCP Client", name: "mcp", shortcut: "" },
    // System & UI
    Entry { category: "System & UI", title: "System Info", name: "info", shortcut: "" },
    Entry { category: "System & UI", title: "Process Monitor", name: "top", shortcut: "" },
    Entry { category: "System & UI", title: "Kernel Trace", name: "ktrace", shortcut: "" },
    Entry { category: "System & UI", title: "Date / Time", name: "datetime", shortcut: "" },
    Entry { category: "System & UI", title: "PCI Devices", name: "lspci", shortcut: "" },
    Entry { category: "System & UI", title: "UI Config", name: "ui", shortcut: "" },
    Entry { category: "System & UI", title: "Shortcuts", name: "shortcuts", shortcut: "" },
    Entry { category: "System & UI", title: "Clipboard", name: "clip", shortcut: "" },
    Entry { category: "System & UI", title: "Close Action Pane", name: "close", shortcut: "Ctrl+W" },
    Entry { category: "System & UI", title: "Commands Help", name: "help", shortcut: "" },
    // Media
    Entry { category: "Media", title: "Voice", name: "voice", shortcut: "" },
    Entry { category: "Media", title: "ONNX Models", name: "onnx", shortcut: "" },
];

/// Build the filtered flat row list for `query` (case-insensitive substring on
/// title, name, category, shortcut).
pub fn filter_rows(query: &str) -> Vec<Row> {
    let q = query.trim().to_ascii_lowercase();
    let mut out = Vec::new();
    let mut last_cat = "";
    for e in ENTRIES {
        if !q.is_empty() {
            let hay = alloc::format!(
                "{} {} {} {}",
                e.category, e.title, e.name, e.shortcut
            )
            .to_ascii_lowercase();
            if !hay.contains(&q) {
                continue;
            }
        }
        if e.category != last_cat {
            out.push(Row::Header(String::from(e.category)));
            last_cat = e.category;
        }
        out.push(Row::Item {
            title: String::from(e.title),
            name: String::from(e.name),
            shortcut: String::from(e.shortcut),
        });
    }
    out
}

/// Indices of selectable (Item) rows in `rows`.
pub fn selectable(rows: &[Row]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter_map(|(i, r)| match r {
            Row::Item { .. } => Some(i),
            Row::Header(_) => None,
        })
        .collect()
}

/// Move selection among selectable rows. `dir` is -1/up or +1/down (or more).
/// Returns the new absolute row index.
pub fn move_sel(rows: &[Row], sel: usize, dir: i32) -> usize {
    let opts = selectable(rows);
    if opts.is_empty() {
        return 0;
    }
    let pos = opts.iter().position(|&i| i == sel).unwrap_or(0);
    let n = opts.len() as i32;
    let mut np = pos as i32 + dir;
    while np < 0 {
        np += n;
    }
    opts[(np as usize) % opts.len()]
}

/// First selectable row index, or 0.
pub fn first_sel(rows: &[Row]) -> usize {
    selectable(rows).into_iter().next().unwrap_or(0)
}

/// Ensure `scroll` keeps `sel` visible in a viewport of `view_rows` lines.
pub fn clamp_scroll(sel: usize, scroll: usize, view_rows: usize, total: usize) -> usize {
    if view_rows == 0 || total == 0 {
        return 0;
    }
    let max_scroll = total.saturating_sub(view_rows);
    let mut s = scroll.min(max_scroll);
    if sel < s {
        s = sel;
    } else if sel >= s + view_rows {
        s = sel + 1 - view_rows;
    }
    s.min(max_scroll)
}

/// Command name at absolute row `sel`, if that row is an item.
pub fn name_at(rows: &[Row], sel: usize) -> Option<&str> {
    match rows.get(sel) {
        Some(Row::Item { name, .. }) => Some(name.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn filter_groups_and_search() {
        let all = filter_rows("");
        assert!(all.iter().any(|r| matches!(r, Row::Header(h) if h == "Files")));
        assert!(all.iter().any(|r| matches!(r, Row::Item { name, .. } if name == "ls")));

        let f = filter_rows("comp");
        // compact + maybe others
        assert!(f.iter().any(|r| matches!(r, Row::Item { name, .. } if name == "compact")));
        // Headers only for matching categories
        assert!(f.iter().any(|r| matches!(r, Row::Header(_))));
    }

    #[test_case]
    fn nav_skips_headers() {
        let rows = filter_rows("");
        let s0 = first_sel(&rows);
        assert!(matches!(rows[s0], Row::Item { .. }));
        let s1 = move_sel(&rows, s0, 1);
        assert!(s1 > s0 || s1 == s0);
        assert!(matches!(rows[s1], Row::Item { .. }));
        let up = move_sel(&rows, s0, -1);
        assert!(matches!(rows[up], Row::Item { .. }));
    }

    #[test_case]
    fn scroll_keeps_selection_visible() {
        assert_eq!(clamp_scroll(0, 0, 5, 20), 0);
        assert_eq!(clamp_scroll(7, 0, 5, 20), 3); // 7 visible when scroll=3 (3..8)
        assert_eq!(clamp_scroll(2, 5, 5, 20), 2); // scroll back
    }
}
