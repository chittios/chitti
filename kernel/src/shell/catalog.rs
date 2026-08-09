//! Categorised command catalogue — **single source of truth** for:
//!
//! * the **Commands** help browser (`/help` modal)
//! * **slash-command suggestions** in the composer (`shell::suggest`)
//! * tab-completion name discovery (canonical names; aliases live in
//!   [`COMMAND_ALIASES`])
//!
//! When you add or rename a shell `/command`, **update [`ENTRIES`] here**
//! (and the `dispatch_system` match arm). Do not maintain a parallel list in
//! `suggest.rs`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One installable / slash command in the browser **and** composer popup.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub category: &'static str,
    /// Friendly title — help modal left column **and** slash-suggestion detail.
    pub title: &'static str,
    /// Canonical command name without `/` (e.g. `compact`, `channel`).
    pub name: &'static str,
    /// Optional shortcut hint (e.g. `Ctrl+W`), else empty.
    pub shortcut: &'static str,
}

/// Extra names accepted by Tab completion that are **aliases** of a catalogue
/// entry (not listed as their own `ENTRIES` row).
pub const COMMAND_ALIASES: &[&str] = &[
    "channels", // → channel
    "date",     // → datetime
    "net",      // → network
    "keys",     // → shortcuts
    "logs",     // → ktrace
    "todo",     // → todos
    "quit",     // → exit
    "reboot",   // → restart
    "perms",    // → permissions
    "panes",      // → pane
    "resolution", // → display
    "res",        // → display
    "bt",         // → bluetooth
    "uvc",        // → camera
    "touchscreen", // → touch
    "bat",        // → battery
    "screencap",  // → screenshot
    "shot",       // → screenshot
    "screencast", // → record
    "rec",        // → record
    "notifications", // → notify
    "notif",      // → notify
    "cron",       // → schedule
    "kbd",        // → keyboard
    "password",   // → passwd
];

/// Commands a **scheduled run** may never install, and a package manifest may
/// never claim as a `command_hooks` name.
///
/// These are the interactive-only login commands. A schedule fires with no human
/// at the keyboard (and under `IN_TOOL_CALL`), so a scheduled `/passwd` could
/// only ever be refused — leaving a job that fails forever and silently. A
/// manifest hook is dead code for the same reason `dispatch_system` never sees
/// these names, but reserving them says so out loud rather than relying on it.
// `ssh-keygen` is here because *overwriting* a private key locks the human out
// of every server that trusts it — so no schedule may fire it and no package may
// claim the name, exactly as for the login commands.
pub const RESERVED_HUMAN_ONLY: &[&str] = &["passwd", "password", "lock", "ssh-keygen"];

/// Whether `name` names a human-only command that automation must not install.
pub fn is_human_only(name: &str) -> bool {
    RESERVED_HUMAN_ONLY.iter().any(|&r| r == name)
}

/// Whether `name` is a known slash command or alias (for completion).
pub fn is_command_name(name: &str) -> bool {
    ENTRIES.iter().any(|e| e.name == name) || COMMAND_ALIASES.iter().any(|&a| a == name)
}

/// Names matching `prefix` for Tab completion (catalogue + aliases).
pub fn complete_names(prefix: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    for e in ENTRIES {
        if e.name.starts_with(prefix) {
            out.push(e.name);
        }
    }
    for &a in COMMAND_ALIASES {
        if a.starts_with(prefix) && !out.iter().any(|&n| n == a) {
            out.push(a);
        }
    }
    out
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

/// Full catalogue — grouped like the bordered Commands palette.
pub const ENTRIES: &[Entry] = &[
    // Session
    Entry { category: "Session", title: "Session Info", name: "session", shortcut: "" },
    Entry { category: "Session", title: "Clear Chat", name: "clear", shortcut: "" },
    Entry { category: "Session", title: "Compact History", name: "compact", shortcut: "" },
    Entry { category: "Session", title: "Agents browser", name: "agents", shortcut: "" },
    Entry { category: "Session", title: "Exit / Power Off", name: "exit", shortcut: "Ctrl+D" },
    Entry { category: "Session", title: "Restart", name: "restart", shortcut: "" },
    // Context & memory
    Entry { category: "Context", title: "Memory", name: "memory", shortcut: "" },
    Entry { category: "Context", title: "Todos", name: "todos", shortcut: "" },
    Entry { category: "Context", title: "Skills", name: "skills", shortcut: "" },
    Entry { category: "Context", title: "Invoke Skill", name: "skill", shortcut: "" },
    Entry { category: "Context", title: "Plan Mode", name: "plan", shortcut: "" },
    Entry { category: "Context", title: "Permissions", name: "permissions", shortcut: "" },
    // Model & agent
    Entry { category: "Model & Agent", title: "Switch Model", name: "model", shortcut: "" },
    Entry { category: "Model & Agent", title: "Approval Mode", name: "mode", shortcut: "" },
    Entry { category: "Model & Agent", title: "Scheduled Runs", name: "schedule", shortcut: "" },
    Entry { category: "Model & Agent", title: "Thinking", name: "think", shortcut: "" },
    Entry { category: "Model & Agent", title: "Infer (parity)", name: "infer", shortcut: "" },
    Entry { category: "Model & Agent", title: "Perf Benchmark", name: "perf", shortcut: "" },
    Entry { category: "Model & Agent", title: "Bench: matvec | synapse gates", name: "bench", shortcut: "" },
    Entry { category: "Model & Agent", title: "Red-team the agent boundary", name: "redteam", shortcut: "" },
    Entry { category: "Model & Agent", title: "Audit log: verify | export", name: "audit", shortcut: "" },
    // Files
    Entry { category: "Files", title: "List Directory", name: "ls", shortcut: "" },
    Entry { category: "Files", title: "Print File", name: "cat", shortcut: "" },
    Entry { category: "Files", title: "First Lines", name: "head", shortcut: "" },
    Entry { category: "Files", title: "Last Lines", name: "tail", shortcut: "" },
    Entry { category: "Files", title: "Copy File To Clipboard", name: "pbcopy", shortcut: "" },
    Entry { category: "Files", title: "Open / Edit / Play", name: "open", shortcut: "" },
    Entry { category: "Files", title: "Edit File", name: "edit", shortcut: "" },
    Entry { category: "Files", title: "Image Decoder (ring 3)", name: "decoder", shortcut: "" },
    Entry { category: "System", title: "Heap Allocator (A/B)", name: "heap", shortcut: "" },
    Entry { category: "Web", title: "HTML Engine (A/B)", name: "html", shortcut: "" },
    Entry {
        category: "Files",
        title: "JavaScript (Node-style -e / file)",
        name: "js",
        shortcut: "",
    },
    Entry { category: "Files", title: "Make Directory", name: "mkdir", shortcut: "" },
    Entry { category: "Files", title: "Copy", name: "cp", shortcut: "" },
    Entry { category: "Files", title: "Move / Rename", name: "mv", shortcut: "" },
    Entry { category: "Files", title: "Remove", name: "rm", shortcut: "" },
    Entry { category: "Files", title: "Touch File", name: "touch", shortcut: "" },
    Entry { category: "Files", title: "Glob Paths", name: "glob", shortcut: "" },
    Entry { category: "Files", title: "Grep Contents", name: "grep", shortcut: "" },
    Entry { category: "Files", title: "Working Directory", name: "pwd", shortcut: "" },
    Entry { category: "Files", title: "Change Directory", name: "cd", shortcut: "" },
    // Storage
    Entry { category: "Storage", title: "Disks", name: "disks", shortcut: "" },
    Entry { category: "Storage", title: "Mounts", name: "mounts", shortcut: "" },
    Entry { category: "Storage", title: "Mount Volume", name: "mount", shortcut: "" },
    Entry { category: "Storage", title: "Unmount", name: "umount", shortcut: "" },
    Entry { category: "Storage", title: "Install Chitti", name: "install", shortcut: "" },
    Entry { category: "Storage", title: "Format ext4", name: "mkext4", shortcut: "" },
    Entry { category: "Storage", title: "Format exFAT", name: "mkexfat", shortcut: "" },
    // Network
    Entry { category: "Network", title: "Network Status", name: "network", shortcut: "" },
    Entry { category: "Network", title: "Ping Host", name: "ping", shortcut: "" },
    Entry { category: "Network", title: "SSH Client", name: "ssh", shortcut: "" },
    Entry { category: "Network", title: "Generate an SSH Key", name: "ssh-keygen", shortcut: "" },
    Entry { category: "Network", title: "Wi-Fi", name: "wifi", shortcut: "" },
    Entry { category: "Network", title: "HTTP Client", name: "http", shortcut: "" },
    Entry {
        category: "Network",
        title: "Browse URL / file:///samples/html/…",
        name: "browse",
        shortcut: "",
    },
    Entry { category: "Network", title: "WebSocket", name: "ws", shortcut: "" },
    Entry { category: "Network", title: "MCP Client", name: "mcp", shortcut: "" },
    Entry {
        category: "Network",
        title: "Messaging channels (Telegram…)",
        name: "channel",
        shortcut: "",
    },
    // System & UI
    Entry { category: "System & UI", title: "System Info", name: "info", shortcut: "" },
    Entry { category: "System & UI", title: "Process Monitor", name: "top", shortcut: "" },
    Entry { category: "System & UI", title: "Kernel Trace", name: "ktrace", shortcut: "" },
    Entry { category: "System & UI", title: "Date / Time", name: "datetime", shortcut: "" },
    Entry { category: "System & UI", title: "PCI Devices", name: "lspci", shortcut: "" },
    Entry { category: "System & UI", title: "Battery", name: "battery", shortcut: "" },
    Entry { category: "System & UI", title: "Power / Energy Mode", name: "power", shortcut: "" },
    Entry { category: "System & UI", title: "Bluetooth", name: "bluetooth", shortcut: "" },
    Entry {
        category: "System & UI",
        title: "Camera grab (UVC still)",
        name: "camera",
        shortcut: "",
    },
    Entry { category: "System & UI", title: "Screenshot", name: "screenshot", shortcut: "Cmd+Shift+3 / PrtSc" },
    Entry { category: "System & UI", title: "Screen record (start/stop)", name: "record", shortcut: "Cmd+Shift+5" },
    Entry { category: "System & UI", title: "Keyboard Layout & IME", name: "keyboard", shortcut: "" },
    Entry { category: "System & UI", title: "Notifications", name: "notify", shortcut: "" },
    Entry { category: "System & UI", title: "Touchscreen", name: "touchscreen", shortcut: "" },
    // Login. Listed here so `/help` and Tab completion find them — they are
    // **interactive-only** (matched in the REPL, never in `dispatch_system`), so
    // being in this catalogue does not make them reachable by an agent. See
    // `shell::auth` for why that distinction is the whole security property.
    Entry { category: "System & UI", title: "Login Password", name: "passwd", shortcut: "" },
    Entry { category: "System & UI", title: "Lock Console", name: "lock", shortcut: "" },
    Entry { category: "System & UI", title: "Suspend", name: "suspend", shortcut: "" },
    Entry { category: "System & UI", title: "UI Config", name: "ui", shortcut: "" },
    Entry { category: "System & UI", title: "Theme", name: "theme", shortcut: "" },
    Entry { category: "System & UI", title: "Status Bar Position", name: "statusbar", shortcut: "" },
    Entry { category: "System & UI", title: "Panes & Grid", name: "pane", shortcut: "Ctrl+F" },
    Entry { category: "System & UI", title: "Display / Resolution", name: "display", shortcut: "" },
    Entry { category: "System & UI", title: "Shortcuts", name: "shortcuts", shortcut: "" },
    Entry { category: "System & UI", title: "Clipboard", name: "clip", shortcut: "" },
    Entry { category: "System & UI", title: "VirtualBox guest transport", name: "vbox", shortcut: "" },
    Entry { category: "System & UI", title: "Close Action Pane", name: "close", shortcut: "Ctrl+W" },
    Entry { category: "System & UI", title: "About ChittiOS", name: "about", shortcut: "" },
    Entry { category: "System & UI", title: "Commands Help", name: "help", shortcut: "Cmd+/ or F1" },
    // Media
    Entry { category: "Media", title: "Voice", name: "voice", shortcut: "" },
    Entry { category: "Media", title: "PDF Viewer", name: "pdf", shortcut: "" },
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
            let icon = crate::icons::for_command_category(e.category);
            out.push(Row::Header(alloc::format!("{icon} {}", e.category)));
            last_cat = e.category;
        }
        let icon = crate::icons::for_command(e.name);
        out.push(Row::Item {
            title: alloc::format!("{icon} {}", e.title),
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
        // Headers carry an FA glyph prefix (e.g. "📁 Files").
        assert!(all.iter().any(|r| matches!(r, Row::Header(h) if h.contains("Files"))));
        assert!(all.iter().any(|r| matches!(r, Row::Item { name, .. } if name == "ls")));
        assert!(
            all.iter()
                .any(|r| matches!(r, Row::Item { title, name, .. } if name == "ls" && title.contains("List"))),
            "item titles keep the friendly label after the FA icon"
        );

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

    #[test_case]
    fn channel_in_catalogue_and_complete() {
        assert!(ENTRIES.iter().any(|e| e.name == "channel"));
        assert!(is_command_name("channel"));
        assert!(is_command_name("channels"));
        let m = complete_names("chan");
        assert!(m.iter().any(|n| *n == "channel"));
        assert!(m.iter().any(|n| *n == "channels"));
    }

    /// Catalogue names must be unique (no duplicate rows).
    #[test_case]
    fn entry_names_unique() {
        for (i, e) in ENTRIES.iter().enumerate() {
            for (j, o) in ENTRIES.iter().enumerate() {
                if i != j {
                    assert_ne!(e.name, o.name, "duplicate catalogue name {}", e.name);
                }
            }
        }
    }
}
