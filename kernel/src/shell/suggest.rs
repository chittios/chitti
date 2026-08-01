//! Slash-command and `@file` mention suggestions for the shell composer.
//!
//! Pure filtering + ranking (no framebuffer). The line editor drives this on
//! every edit; the compositor paints the popup.
//!
//! **Single source of truth for slash commands:** [`crate::shell::catalog::ENTRIES`].
//! When you add or rename a `/command`, update that catalogue (title =
//! suggestion detail). Do **not** maintain a parallel list here — unit tests
//! fail if a catalogue entry is missing from suggestions.

use crate::shell::catalog;
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
/// Sourced from [`catalog::ENTRIES`] so `/channel` and every other shell
/// command stay in the popup when the catalogue is updated.
pub fn command_items(prefix: &str, max: usize) -> Vec<Item> {
    let p = prefix.to_ascii_lowercase();
    let mut starts: Vec<Item> = Vec::new();
    let mut contains: Vec<Item> = Vec::new();
    for e in catalog::ENTRIES {
        let nlow = e.name.to_ascii_lowercase();
        let title_low = e.title.to_ascii_lowercase();
        let cat_low = e.category.to_ascii_lowercase();
        if p.is_empty() || nlow.starts_with(&p) {
            starts.push(cmd_item(e.name, e.title));
        } else if nlow.contains(&p) || title_low.contains(&p) || cat_low.contains(&p) {
            contains.push(cmd_item(e.name, e.title));
        }
    }
    // A **fully-typed command must come first**, whatever the catalog's
    // declaration order. The menu highlights item 0, and the line editor submits
    // on Enter only when accepting the highlighted item would not change the line
    // (`shell::suggest_would_complete`). `model` is declared before `mode`, so
    // typing `/mode` highlighted `/model`: Enter "accepted" it, the line became
    // `/model `, and the next command was appended onto it — `/mode` could not be
    // run at all, and every later line was one out of step. That is the same
    // shape as the `/todos open` swallow this gate was originally added for; the
    // gate was right, the candidate order was not.
    //
    // Done before `truncate` so the exact match cannot be cut, and as a
    // move-to-front so everything else keeps its declaration order.
    if let Some(i) = starts.iter().position(|it| it.label.len() > 1 && it.label[1..].eq_ignore_ascii_case(&p)) {
        let exact = starts.remove(i);
        starts.insert(0, exact);
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

    /// Every catalogue command must surface in slash suggestions (catches
    /// forgetting to wire a new `/command` into the composer popup).
    #[test_case]
    fn catalog_commands_are_all_suggestable() {
        for e in catalog::ENTRIES {
            let items = command_items(e.name, 64);
            assert!(
                items.iter().any(|i| i.label == alloc::format!("/{}", e.name)),
                "catalogue command /{} missing from suggestions — update catalog::ENTRIES only (suggest reads it)",
                e.name
            );
            // Exact-name filter should put this command first or alone.
            assert!(
                items.iter().any(|i| i.detail == e.title),
                "/{} should show title {:?} as detail",
                e.name,
                e.title
            );
        }
        // Regression: messaging channels must appear when typing /chan…
        let ch = command_items("chan", 8);
        assert!(
            ch.iter().any(|i| i.label == "/channel"),
            "/channel must appear for prefix 'chan': {:?}",
            ch.iter().map(|i| i.label.as_str()).collect::<alloc::vec::Vec<_>>()
        );
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
