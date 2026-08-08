//! Slash-command, `@file` mention, and **filesystem path-argument** suggestions
//! for the shell composer.
//!
//! Pure filtering + ranking (no framebuffer). The line editor drives this on
//! every edit; the compositor paints the popup.
//!
//! **Single source of truth for slash commands:** [`crate::shell::catalog::ENTRIES`].
//! When you add or rename a `/command`, update that catalogue (title =
//! suggestion detail). Do **not** maintain a parallel list here — unit tests
//! fail if a catalogue entry is missing from suggestions.
//!
//! Path-argument completion (`/ls /configs`, `/open /samples/images/…`) is
//! keyed on [`PATH_COMMANDS`] — the commands whose arguments are store/mount
//! paths. The caller supplies the parent directory's `vfs::readdir` listing so
//! this module stays pure (see [`path_items`] / [`path_parts`]).

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
    /// A filesystem path argument after a path-taking command (`/ls /configs/`).
    Path,
}

/// Commands whose whitespace-delimited arguments are filesystem paths
/// (`/ls /configs`, `/open /samples/images/x.png`, `/cp /a /b`, …). Typing an
/// argument of one of these auto-suggests store / mount paths. Deliberately a
/// small hand-list: every entry here must actually take a path (the catalogue
/// is the source of truth for *names*, this is about *argument shape*).
pub const PATH_COMMANDS: &[&str] = &[
    "ls", "cd", "cat", "head", "tail", "pbcopy", "open", "edit", "browse", "mkdir", "rm", "cp",
    "mv", "touch", "glob", "grep",
];

/// Commands whose path argument can only ever be a **directory**, so the popup
/// must not offer files. Listing `README.md` under `/cd ` is not merely noise
/// now that `/cd` refuses a non-directory — it offers a completion that is
/// guaranteed to fail.
pub const DIR_ONLY_COMMANDS: &[&str] = &["cd", "mkdir"];

/// The slash command a line begins with, if any (`"/cd sub"` -> `Some("cd")`).
///
/// Pure, so the caller can decide what to feed the popup without this module
/// having to carry the command through [`Context`].
pub fn leading_command(buf: &str) -> Option<&str> {
    let rest = buf.trim_start().strip_prefix('/')?;
    let cmd = rest.split_whitespace().next()?;
    (!cmd.is_empty()).then_some(cmd)
}

/// The command of the stage the caret is in — the last one on the line, not
/// the first. `"/ls / | /cd su"` -> `Some("cd")`.
pub fn active_command(buf: &str) -> Option<&str> {
    let (off, tail) = crate::shell::pipeline::completion_tail(buf);
    if tail == crate::shell::pipeline::TailKind::RedirectPath {
        // A redirect target belongs to no command's argument shape.
        return None;
    }
    let seg = buf[off..].trim_start();
    // `match`, not `unwrap_or`: its argument is evaluated eagerly, so the
    // `return None` in the else branch fired even when the `/` stripped fine —
    // which made every dirs-only completion stop working. Caught by
    // `cd_completes_directories_only`, which is why that test exists.
    let seg = match seg.strip_prefix('/') {
        Some(r) => r,
        None if off > 0 => seg,
        None => return None,
    };
    let cmd = seg.split_whitespace().next()?;
    (!cmd.is_empty()).then_some(cmd)
}

/// Whether the caret's command completes directories only (see
/// [`DIR_ONLY_COMMANDS`]).
///
/// Keyed on [`active_command`], not the first command on the line: after
/// `/ls / | /cd ` the popup must offer directories for `cd`, not files for
/// `ls`.
pub fn wants_dirs_only(buf: &str) -> bool {
    active_command(buf).is_some_and(|c| DIR_ONLY_COMMANDS.contains(&c))
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
    // Completion applies to the LAST stage of the line, not the first command
    // on it. Without this, `/head -n 3 /x | /gr` took `head` as the command,
    // found it in PATH_COMMANDS, and offered *folders* for `/gr` — the first
    // stage's argument shape applied to a completely different command.
    let (seg_off, tail) = crate::shell::pipeline::completion_tail(before);
    let seg = &before[seg_off..];

    // A redirection target is a path whatever the command was: `/ls / > /tm`
    // completes a file, not a directory listing of `/ls`'s argument.
    if tail == crate::shell::pipeline::TailKind::RedirectPath {
        return path_context(seg, seg_off);
    }

    let trimmed_start = seg.trim_start();
    let lead = seg.len() - trimmed_start.len();
    // A later stage may be typed with or without its own `/` — the parser
    // accepts both, so the popup must too. Accepting a bare word at the start
    // of the *whole* line would turn ordinary chat into a command popup, which
    // is why the no-slash form is allowed only after an operator.
    let rest = match trimmed_start.strip_prefix('/') {
        Some(r) => r,
        None if seg_off > 0 => trimmed_start,
        None => return None,
    };
    if !rest.contains(' ') && !rest.contains('\t') {
        return Some(Context {
            kind: Kind::Command,
            prefix: rest.to_string(),
            token_start: seg_off + lead,
        });
    }
    // Path argument after a path-taking command: `/ls /co`, `/ls /configs/`,
    // `/cp /a /b`. The token is whatever follows the last whitespace.
    if let Some(cmd) = rest.split_whitespace().next() {
        let after_cmd = rest.as_bytes().get(cmd.len()).map(|&b| b == b' ' || b == b'\t');
        if PATH_COMMANDS.contains(&cmd) && after_cmd == Some(true) {
            return path_context(seg, seg_off);
        }
    }
    None
}

/// Path completion for the token at the end of `seg`, reported at its absolute
/// offset in the line.
fn path_context(seg: &str, seg_off: usize) -> Option<Context> {
    let start = seg.rfind(|c: char| c == ' ' || c == '\t').map(|i| i + 1).unwrap_or(0);
    let token = &seg[start..];
    if token.contains(' ') || token.contains('\t') {
        return None;
    }
    Some(Context {
        kind: Kind::Path,
        prefix: token.to_string(),
        token_start: seg_off + start,
    })
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

/// Split a typed path prefix into the **parent directory to list** and the
/// **partial final component**. `/configs/co` → (`/configs`, `co`);
/// `/configs/` → (`/configs`, ``); `` → (`/`, ``); bare `co` → (`/`, `co`).
/// Pure — the caller reads the parent via the VFS.
pub fn path_parts(prefix: &str) -> (String, String) {
    if prefix.is_empty() {
        return (String::from("/"), String::new());
    }
    if prefix.ends_with('/') {
        let parent = prefix.trim_end_matches('/');
        return (
            if parent.is_empty() { String::from("/") } else { parent.to_string() },
            String::new(),
        );
    }
    match prefix.rfind('/') {
        Some(i) => (prefix[..i].to_string(), prefix[i + 1..].to_string()),
        None => (String::from("/"), prefix.to_string()),
    }
}

/// Path completions for a typed `prefix` against the parent directory's
/// listing. Directories first (labelled with a trailing `/` and a `dir`
/// detail), then files — same order as `/ls`. The insert keeps the directory
/// suffix so Tab/Enter drills one level at a time. When the prefix is already
/// an **exact file**, the item is dropped so the popup closes instead of
/// echoing the completed path back at the caret (the shell analogue of bash's
/// "nothing left to complete").
pub fn path_items(prefix: &str, entries: &[crate::fs::vfs::DirEntry], max: usize) -> Vec<Item> {
    let max = max.max(1).min(MAX_ITEMS);
    let (parent, partial) = path_parts(prefix);
    let norm = crate::synapse::vpath::normalize(prefix);
    // The completed path keeps the **typed form**: absolute for `/…`, `~/` for
    // the home, bare-relative for anything else (so `docs` completes to
    // `docs/…`, not an absolute path).
    let absolute = prefix.starts_with('/');
    let tilde = prefix.starts_with("~/");
    let mut out: Vec<Item> = Vec::new();
    for e in entries {
        if !e.name.starts_with(&partial) {
            continue;
        }
        let name = &e.name;
        let full = if absolute {
            if parent == "/" {
                alloc::format!("/{name}")
            } else {
                alloc::format!("{parent}/{name}")
            }
        } else if tilde {
            if parent == "~" {
                alloc::format!("~/{name}")
            } else {
                alloc::format!("{parent}/{name}")
            }
        } else if parent == "/" {
            name.clone() // bare token → relative to the pwd
        } else {
            alloc::format!("{parent}/{name}")
        };
        if !e.is_dir && norm == full {
            continue; // already typed the whole file — nothing to complete
        }
        let display = if e.is_dir {
            alloc::format!("{full}/")
        } else {
            full
        };
        out.push(Item {
            insert: display.clone(),
            label: display,
            detail: if e.is_dir { String::from("dir") } else { String::new() },
        });
    }
    out.sort_by(|a, b| {
        let ad = a.detail == "dir";
        let bd = b.detail == "dir";
        bd.cmp(&ad).then_with(|| a.label.cmp(&b.label))
    });
    out.truncate(max);
    out
}

/// Build the suggestion list for the current context.
///
/// `store_paths` feeds `@file` mentions; `dir_entries` feeds path-argument
/// completion (`Kind::Path`) and is the caller's `vfs::readdir` of
/// [`path_parts`]'s parent (fetched outside so this stays pure + testable).
pub fn items_for(
    ctx: &Context,
    store_paths: &[String],
    dir_entries: &[crate::fs::vfs::DirEntry],
) -> Vec<Item> {
    match ctx.kind {
        Kind::Command => command_items(&ctx.prefix, MAX_ITEMS),
        Kind::File => file_items(&ctx.prefix, store_paths, MAX_ITEMS),
        Kind::Path => path_items(&ctx.prefix, dir_entries, MAX_ITEMS),
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
    fn after_a_pipe_the_popup_completes_a_command_not_the_first_stage_s_paths() {
        // The reported bug: `context` took `rest.split_whitespace().next()` as
        // THE command for the whole line, so `/head … /x | /gr` matched `head`
        // in PATH_COMMANDS and offered *folders* for `/gr` — the first stage's
        // argument shape applied to a different command entirely.
        let line = "/head -n 3 /samples/README.md | /gr";
        let c = context(line, line.len()).unwrap();
        assert_eq!(c.kind, Kind::Command, "after `|` a new command begins");
        assert_eq!(c.prefix, "gr");
        // The token starts at the `/`, so accepting replaces `/gr` and leaves
        // the rest of the line alone.
        assert_eq!(&line[c.token_start..], "/gr");

        // Every stage separator behaves the same way.
        for sep in ["|", "|&", "||", "&&", ";"] {
            let line = alloc::format!("/ls / {sep} /he");
            let c = context(&line, line.len()).unwrap();
            assert_eq!(c.kind, Kind::Command, "after `{sep}`");
            assert_eq!(c.prefix, "he");
        }
    }

    #[test_case]
    fn a_later_stage_completes_its_own_path_arguments() {
        // Once the second stage HAS a command, its own argument shape applies —
        // grep takes a path, so a path is offered, at the right offset.
        let line = "/cat /x | /grep foo /sam";
        let c = context(line, line.len()).unwrap();
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "/sam");
        assert_eq!(&line[c.token_start..], "/sam");

        // And a stage whose command takes no path offers nothing, rather than
        // inheriting the first stage's path-ness.
        let line = "/cat /x | /mounts ";
        assert!(context(line, line.len()).is_none());
    }

    #[test_case]
    fn a_stage_typed_without_its_slash_still_completes() {
        // The parser accepts `| grep y`, so the popup must too. But a bare word
        // at the start of the whole line is ordinary chat and must NOT open a
        // command popup.
        let line = "/ls / | gr";
        let c = context(line, line.len()).unwrap();
        assert_eq!((c.kind, c.prefix.as_str()), (Kind::Command, "gr"));
        assert_eq!(&line[c.token_start..], "gr");
        assert!(context("hello wor", 9).is_none(), "plain chat must not complete");
    }

    #[test_case]
    fn a_redirect_target_completes_a_path_whatever_the_command_is() {
        // `>` is a path position even after a command that takes none, and even
        // after a dirs-only one.
        let line = "/mounts > /tm";
        let c = context(line, line.len()).unwrap();
        assert_eq!((c.kind, c.prefix.as_str()), (Kind::Path, "/tm"));
        let line = "/ls / >> /var/lo";
        let c = context(line, line.len()).unwrap();
        assert_eq!((c.kind, c.prefix.as_str()), (Kind::Path, "/var/lo"));
        // A redirect target is a file, so it is never dirs-only even when the
        // stage's command is `/mkdir`.
        assert!(!wants_dirs_only("/mkdir /a > /b"));
    }

    #[test_case]
    fn dirs_only_follows_the_active_stage() {
        // `/cd` completes directories; `/ls` does not. After a pipe it must be
        // the *last* stage that decides.
        assert!(wants_dirs_only("/cd /con"));
        assert!(!wants_dirs_only("/ls /con"));
        assert!(wants_dirs_only("/ls / | /cd /con"));
        assert!(!wants_dirs_only("/cd /a | /ls /con"));
        assert_eq!(active_command("/ls / | /cd su"), Some("cd"));
        assert_eq!(active_command("/ls /x"), Some("ls"));
    }

    #[test_case]
    fn a_quoted_operator_does_not_start_a_new_completion_stage() {
        // The quote rules must match the parser's, or the popup and the runner
        // disagree about where a stage begins.
        let line = r#"/grep "a | b" /sam"#;
        let c = context(line, line.len()).unwrap();
        assert_eq!(c.kind, Kind::Path, "the `|` is inside quotes");
        assert_eq!(c.prefix, "/sam");
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

    /// The other direction: a command that takes a **path** must also be in the
    /// catalogue, or it completes its arguments while being invisible in `/help`
    /// and in the slash popup.
    ///
    /// `/cd` was exactly that hole in reverse — absent from *both* lists, so it
    /// had no suggestion entry, no path completion and no help row, while working
    /// perfectly when typed blind. The existing test above only checks
    /// catalogue → suggestions, which cannot see a command that is in neither.
    #[test_case]
    fn path_taking_commands_are_in_the_catalogue() {
        for name in PATH_COMMANDS {
            assert!(
                catalog::ENTRIES.iter().any(|e| e.name == *name),
                "/{name} takes a path but is not in catalog::ENTRIES — it would have \
                 argument completion while being absent from /help and the popup"
            );
        }
    }

    /// `/cd` completes directories only, and is otherwise an ordinary path
    /// command.
    #[test_case]
    fn cd_completes_directories_only() {
        assert_eq!(leading_command("/cd sub"), Some("cd"));
        assert_eq!(leading_command("/ls -lah ."), Some("ls"));
        assert_eq!(leading_command("hello"), None);
        assert!(wants_dirs_only("/cd re"));
        assert!(wants_dirs_only("/mkdir -p re"));
        // Every other path command still offers files — `/cat dir/` would be
        // useless with them filtered out.
        assert!(!wants_dirs_only("/ls re"));
        assert!(!wants_dirs_only("/cat re"));

        // And `/cd ` is a path context at all, which is what was missing.
        let c = context("/cd sub", 7).expect("`/cd <path>` must be a path context");
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "sub");
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

    #[test_case]
    fn context_path_argument() {
        // `/ls /co` — caret at end of the path token.
        let c = context("/ls /co", 7).unwrap();
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "/co");
        assert_eq!(c.token_start, 4);

        // `/ls /configs/` — empty partial lists the dir's children.
        let c = context("/ls /configs/", 13).unwrap();
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "/configs/");
        assert_eq!(c.token_start, 4);

        // Bare space after the command → root listing.
        let c = context("/ls ", 4).unwrap();
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "");

        // Second argument of a multi-arg command.
        let c = context("/cp /a /b", 9).unwrap();
        assert_eq!(c.kind, Kind::Path);
        assert_eq!(c.prefix, "/b");

        // Command name alone (no space) stays a command.
        let c = context("/ls", 3).unwrap();
        assert_eq!(c.kind, Kind::Command);

        // A non-path command's arguments do not complete paths.
        assert!(!matches!(context("/help thing", 11), Some(Context { kind: Kind::Path, .. })));

        // Caret back in the command name — still a command.
        let c = context("/ls /co", 3).unwrap();
        assert_eq!(c.kind, Kind::Command);
    }

    #[test_case]
    fn path_parts_split() {
        assert_eq!(path_parts(""), (String::from("/"), String::new()));
        assert_eq!(path_parts("/"), (String::from("/"), String::new()));
        assert_eq!(path_parts("/configs/co"), (String::from("/configs"), String::from("co")));
        assert_eq!(path_parts("/configs/"), (String::from("/configs"), String::new()));
        assert_eq!(path_parts("co"), (String::from("/"), String::from("co")));
        assert_eq!(path_parts("/agent/1/SOUL.m"), (String::from("/agent/1"), String::from("SOUL.m")));
    }

    /// Build `vfs::DirEntry` fixtures.
    fn de(name: &str, is_dir: bool) -> crate::fs::vfs::DirEntry {
        crate::fs::vfs::DirEntry {
            name: String::from(name),
            is_dir,
            size: 0,
        }
    }

    #[test_case]
    fn path_items_filters_dirs_first_and_drills() {
        let entries = [
            de("agent", true),
            de("configs", true),
            de("downloads", true),
            de("samples", true),
        ];
        let items = path_items("/sa", &entries, 8);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "/samples/");
        assert_eq!(items[0].insert, "/samples/");
        assert_eq!(items[0].detail, "dir");

        // A bare space (empty prefix) suggests the **current directory's**
        // children as relative names (the pwd, not the store root).
        let items = path_items("", &entries, 8);
        let labels: alloc::vec::Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, ["agent/", "configs/", "downloads/", "samples/"]);

        // Files sort after dirs.
        let mixed = [
            de("note.txt", false),
            de("img", true),
            de("a.bin", false),
        ];
        let items = path_items("/", &mixed, 8);
        let labels: alloc::vec::Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, ["/img/", "/a.bin", "/note.txt"]);
        assert!(items[1].detail.is_empty(), "files carry no detail");
    }

    #[test_case]
    fn path_items_relative_and_tilde_keep_the_typed_form() {
        let entries = [de("docs", true), de("dots.txt", false)];
        // `~/doc` → the home, completed as `~/…`.
        let items = path_items("~/doc", &entries, 8);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].insert, "~/docs/");
        // `doc` → the pwd, completed **relative** (no leading slash).
        let items = path_items("doc", &entries, 8);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].insert, "docs/");
        // `work/doc` → relative with a subdir.
        let items = path_items("work/doc", &entries, 8);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].insert, "work/docs/");
        // Absolute keeps the leading slash.
        let items = path_items("/doc", &entries, 8);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].insert, "/docs/");
    }

    #[test_case]
    fn path_items_drop_exact_typed_file() {
        let entries = [de("ui.json", false), de("ui_old.json", false), de("core", true)];
        // A prefix that is still partial keeps the candidate.
        let items = path_items("/configs/ui", &entries, 8);
        assert!(items.iter().any(|i| i.label == "/configs/ui.json"));
        // The exact file path closes the menu (no item).
        let items = path_items("/configs/ui.json", &entries, 8);
        assert!(items.iter().all(|i| i.label != "/configs/ui.json"), "exact file must drop");
        // A directory with the exact name still offers to drill in.
        let items = path_items("/configs/core", &entries, 8);
        assert!(items.iter().any(|i| i.label == "/configs/core/"));
    }

    #[test_case]
    fn path_items_ignores_non_prefix_entries() {
        let entries = [de("agent", true), de("configs", true)];
        let items = path_items("/zz", &entries, 8);
        assert!(items.is_empty());
    }
}
