//! Slash-command and `@file` mention suggestions for the shell composer.
//!
//! Pure filtering + ranking (no framebuffer). The line editor drives this on
//! every edit; the compositor paints the popup. Matches the Grok-style
//! command menu: a filtered list of `/command` + description, and `@/path`
//! file mentions from the Synapse store.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Maximum rows shown in the popup.
pub const MAX_ITEMS: usize = 8;

/// One row in the suggestion menu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// Text that replaces the in-progress token (includes leading `/` or `@`).
    pub insert: String,
    /// Left column (e.g. `/help`, `@/agent/1/SOUL.md`).
    pub label: String,
    /// Right column — short description or path context.
    pub detail: String,
}

/// What the caret is currently completing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Leading `/command` (no args yet).
    Command,
    /// `@path` mention anywhere in the line.
    File,
}

/// Active completion context: kind, the typed prefix (without `/` or `@`),
/// and the byte offset in the line where the token starts (`/` or `@`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub kind: Kind,
    pub prefix: String,
    pub token_start: usize,
}

/// Slash commands with short descriptions (canonical names only).
/// Keep in sync with `COMMANDS` in `shell/mod.rs`.
const COMMAND_HELP: &[(&str, &str)] = &[
    ("agents", "List / switch / kill agent processes"),
    ("bench", "Matvec kernel throughput"),
    ("cat", "Print a store or mounted file"),
    ("clear", "Reset chat context + clear the pane"),
    ("clip", "Show or set the shared clipboard"),
    ("close", "Close the action pane (or Ctrl+W)"),
    ("compact", "Compact chat context (model summary)"),
    ("cp", "Copy a store file or tree (-r)"),
    ("datetime", "Show or set the wall clock"),
    ("disks", "List block devices + filesystems"),
    ("exit", "Power off the machine"),
    ("glob", "List store paths matching a pattern"),
    ("grep", "Search store file contents"),
    ("help", "Browse commands and usage"),
    ("http", "curl-like HTTP client"),
    ("infer", "Reference inference parity check"),
    ("info", "CPU / memory / model / OS status"),
    ("install", "Install or update Chitti on a disk"),
    ("ktrace", "Toggle the ktrace log stream"),
    ("ls", "List a store directory (Linux-like)"),
    ("lspci", "List PCI devices"),
    ("mcp", "Model Context Protocol client"),
    ("mkdir", "Create a store directory (-p)"),
    ("mkext4", "Format a disk with ext4 (destructive)"),
    ("mode", "Agent tool approvals: manual|auto|bypass|plan"),
    ("model", "Local or remote chat backend"),
    ("mount", "Mount a disk volume at a path"),
    ("mounts", "List mounted volumes"),
    ("mv", "Rename or move a store path"),
    ("network", "Net status / dhcp / static / dns"),
    ("onnx", "Inspect or run an ONNX model"),
    ("open", "Edit / preview / play a file"),
    ("perf", "Prefill/decode tokens per second"),
    ("ping", "ICMP echo a host or IP"),
    ("pwd", "Print working directory"),
    ("rm", "Remove a store file or tree (-r)"),
    ("session", "Show / save / resume a session"),
    ("shortcuts", "List keyboard shortcuts"),
    ("skills", "List installed skills"),
    ("think", "Toggle model thinking (on|off)"),
    ("top", "Live CPU + memory monitor"),
    ("touch", "Create an empty store file"),
    ("ui", "View or reload the UI config"),
    ("umount", "Unmount a path"),
    ("voice", "Voice: test, stt, say, conversation"),
    ("wifi", "Wi-Fi scan / connect / info"),
    ("ws", "WebSocket client"),
];

/// Detect a slash-command or `@file` token ending at the caret.
pub fn context(buf: &str, cur: usize) -> Option<Context> {
    let cur = cur.min(buf.len());
    let before = &buf[..cur];
    // @file mention: last `@` that starts a token, no spaces after it.
    if let Some(i) = before.rfind('@') {
        let ok_start = i == 0 || before.as_bytes().get(i.wrapping_sub(1)) == Some(&b' ');
        if ok_start {
            let rest = &before[i + 1..];
            if !rest.contains(' ') && !rest.contains('\t') {
                return Some(Context {
                    kind: Kind::File,
                    prefix: rest.to_string(),
                    token_start: i,
                });
            }
        }
    }
    // /command only when the line (before caret) is a single leading command.
    let trimmed_start = before.trim_start();
    let lead = before.len() - trimmed_start.len();
    if let Some(rest) = trimmed_start.strip_prefix('/') {
        if !rest.contains(' ') && !rest.contains('\t') {
            return Some(Context {
                kind: Kind::Command,
                prefix: rest.to_string(),
                token_start: lead,
            });
        }
    }
    None
}

/// Ranked command suggestions for `prefix` (without the leading `/`).
pub fn command_items(prefix: &str, max: usize) -> Vec<Item> {
    let p = prefix.to_ascii_lowercase();
    let mut starts: Vec<Item> = Vec::new();
    let mut contains: Vec<Item> = Vec::new();
    for &(name, detail) in COMMAND_HELP {
        let nlow = name.to_ascii_lowercase();
        if p.is_empty() || nlow.starts_with(&p) {
            starts.push(cmd_item(name, detail));
        } else if nlow.contains(&p) || detail.to_ascii_lowercase().contains(&p) {
            contains.push(cmd_item(name, detail));
        }
    }
    starts.extend(contains);
    starts.truncate(max.max(1).min(MAX_ITEMS));
    starts
}

fn cmd_item(name: &str, detail: &str) -> Item {
    Item {
        insert: alloc::format!("/{name} "),
        label: alloc::format!("/{name}"),
        detail: detail.to_string(),
    }
}

/// File-mention suggestions: store paths matching `prefix` (may include a
/// leading `/`). `paths` is the flat store key list.
pub fn file_items(prefix: &str, paths: &[String], max: usize) -> Vec<Item> {
    let max = max.max(1).min(MAX_ITEMS);
    let p = prefix;
    let mut starts: Vec<Item> = Vec::new();
    let mut contains: Vec<Item> = Vec::new();
    for path in paths {
        let show = path.as_str();
        if p.is_empty() || show.starts_with(p) {
            starts.push(file_item(show));
        } else if show.contains(p) {
            contains.push(file_item(show));
        }
    }
    starts.sort_by(|a, b| a.label.len().cmp(&b.label.len()).then(a.label.cmp(&b.label)));
    contains.sort_by(|a, b| a.label.len().cmp(&b.label.len()).then(a.label.cmp(&b.label)));
    starts.extend(contains);
    starts.truncate(max);
    starts
}

fn file_item(path: &str) -> Item {
    // No secondary detail column — long paths need the full row width, and a
    // basename on the right was overflowing the popup border. The label is
    // the full `@path` (ellipsized at draw time, keeping the trailing end).
    Item {
        insert: alloc::format!("@{path}"),
        label: alloc::format!("@{path}"),
        detail: String::new(),
    }
}

/// Build the suggestion list for the current context.
pub fn items_for(ctx: &Context, store_paths: &[String]) -> Vec<Item> {
    match ctx.kind {
        Kind::Command => command_items(&ctx.prefix, MAX_ITEMS),
        Kind::File => file_items(&ctx.prefix, store_paths, MAX_ITEMS),
    }
}

/// Apply the selected item: replace `buf[token_start..cur]` with `item.insert`.
/// Returns the new cursor position (end of the inserted text).
pub fn apply(buf: &mut String, cur: usize, token_start: usize, item: &Item) -> usize {
    let cur = cur.min(buf.len());
    let start = token_start.min(cur);
    buf.replace_range(start..cur, &item.insert);
    start + item.insert.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    #[test_case]
    fn context_slash_command() {
        let c = context("/he", 3).unwrap();
        assert_eq!(c.kind, Kind::Command);
        assert_eq!(c.prefix, "he");
        assert_eq!(c.token_start, 0);

        let c = context("/help ", 6);
        assert!(c.is_none(), "space ends command token");

        let c = context("  /ls", 5).unwrap();
        assert_eq!(c.prefix, "ls");
        assert_eq!(c.token_start, 2);
    }

    #[test_case]
    fn context_file_mention() {
        let c = context("see @/agent/1", 13).unwrap();
        assert_eq!(c.kind, Kind::File);
        assert_eq!(c.prefix, "/agent/1");
        assert_eq!(c.token_start, 4);

        let c = context("@", 1).unwrap();
        assert_eq!(c.prefix, "");
        assert_eq!(c.kind, Kind::File);

        assert!(context("x@y", 3).is_none(), "mid-word @ is not a mention");
    }

    #[test_case]
    fn command_filter_and_apply() {
        let items = command_items("hel", 8);
        assert!(!items.is_empty());
        assert!(items.iter().any(|i| i.label == "/help"));
        assert!(items[0].insert.starts_with('/'));

        let mut buf = String::from("/he");
        let cur = apply(&mut buf, 3, 0, &items.iter().find(|i| i.label == "/help").unwrap());
        assert_eq!(buf, "/help ");
        assert_eq!(cur, buf.len());
    }

    #[test_case]
    fn file_filter() {
        let paths = vec![
            String::from("/agent/1/SOUL.md"),
            String::from("/agent/1/MEMORY.md"),
            String::from("/configs/core/ui.json"),
            String::from("/downloads/pic.png"),
        ];
        let items = file_items("/agent", &paths, 8);
        assert!(items.iter().all(|i| i.label.contains("agent")));
        assert!(items.iter().any(|i| i.insert == "@/agent/1/SOUL.md"));
        // File rows carry no secondary detail (avoids right-column overflow).
        assert!(items.iter().all(|i| i.detail.is_empty()));

        let items = file_items("", &paths, 3);
        assert_eq!(items.len(), 3);
    }

    #[test_case]
    fn empty_prefix_lists_commands() {
        let items = command_items("", 8);
        assert_eq!(items.len(), 8);
        assert!(items.iter().any(|i| i.label.starts_with('/')));
    }
}
