//! **Shell composition** — `|`, `|&`, `;`, `&&`, `||`, `>` and `>>`, parsed
//! into a structure the REPL can run. Pure: no I/O, no dispatch, so the part
//! that is easy to get wrong is testable (`shell/` compiles in the test build).
//!
//! ## Two levels, because precedence is not flat
//!
//! `|` binds tighter than `&&`/`||`/`;`, so `a && b | c` is `a && (b | c)` and
//! not `(a && b) | c`. A flat list of stages and operators cannot express that,
//! and the runner would have to rediscover it — so the shape here is a
//! [`Script`] of [`Pipeline`]s joined by [`Join`]s, which makes the grouping a
//! property of the parse rather than of whoever walks it.
//!
//! ## stdout vs stderr, without touching a single command
//!
//! There is no `serial_eprintln!` in this tree: every command prints through
//! `serial_println!`. But they already separate the two by convention —
//! `/cat`, `/head`, `/ls`, `/grep` and `/mounts` all print `name> …` status
//! lines and unprefixed data lines:
//!
//! ```text
//! cat> /samples/README.md (10669 bytes):   <- diagnostic
//! # ChittiOS sample files                   <- data
//! ```
//!
//! Since a pipeline knows each stage's command name, `"{name}> "` is a
//! **precise** classifier rather than a heuristic, and [`split_streams`] needs
//! no cooperation from any handler. The limit, stated rather than discovered: a
//! command that emits *data* lines prefixed `name> ` would misclassify them.
//! None do today. If one ever does, the fix is an explicit diagnostic macro for
//! that command, not a rewrite of every call site.
//!
//! ## Quotes protect operators; they are NOT word splitting
//!
//! `'` and `"` stop `|` and friends from splitting a line, so
//! `/grep "a | b" file` is one stage. The quotes are then passed through to the
//! command **verbatim**, because commands here parse their own arguments with
//! `split_whitespace` and have never understood quoting. Stripping them would
//! silently change what an existing command receives.

use alloc::string::{String, ToString};
use alloc::borrow::ToOwned;
use alloc::vec::Vec;

/// How one stage's output reaches the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipeKind {
    /// `|` — stdout only.
    Stdout,
    /// `|&` — stdout and the `name> ` diagnostics, interleaved in the order
    /// they were printed.
    Both,
}

/// How one pipeline's result decides whether the next runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Join {
    /// `;` — always run the next.
    Seq,
    /// `&&` — run the next only if this one succeeded.
    And,
    /// `||` — run the next only if this one failed.
    Or,
}

/// Where a stage's stdout goes instead of the console.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Redir {
    None,
    /// `>` — replace the file.
    Write(String),
    /// `>>` — append to it.
    Append(String),
}

/// One command invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    /// Command name with any leading `/` removed, as `dispatch_system` wants.
    pub name: String,
    /// Everything after the name, verbatim (quotes included — see the module
    /// doc).
    pub arg: String,
    pub redir: Redir,
}

/// Stages joined by `|` / `|&`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pipeline {
    pub stages: Vec<Stage>,
    /// `pipes.len() == stages.len() - 1`.
    pub pipes: Vec<PipeKind>,
}

/// Pipelines joined by `;` / `&&` / `||`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Script {
    pub pipelines: Vec<Pipeline>,
    /// `joins.len() == pipelines.len() - 1`.
    pub joins: Vec<Join>,
}

impl Script {
    /// Whether this is a plain single command — in which case the REPL should
    /// take its ordinary path and none of this machinery applies.
    pub fn is_single_command(&self) -> bool {
        self.pipelines.len() == 1
            && self.pipelines[0].stages.len() == 1
            && self.pipelines[0].stages[0].redir == Redir::None
    }

    /// Every stage, in order.
    pub fn stages(&self) -> impl Iterator<Item = &Stage> {
        self.pipelines.iter().flat_map(|p| p.stages.iter())
    }
}

/// A token boundary found outside quotes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sep {
    Pipe(PipeKind),
    Join(Join),
}

/// Whether `line` contains any composition operator outside quotes.
///
/// Cheap pre-check for the REPL: a line with none of these must take exactly
/// the path it took before, byte for byte.
///
/// **Redirection counts.** `>` is not one of `scan`'s separators (it binds
/// inside a stage, not between stages), so checking only those made
/// `/head -n 3 f > out` skip the pipeline path entirely — and `/head` then got
/// `f > out` as its argument and reported "only one path at a time".
pub fn has_operator(line: &str) -> bool {
    if scan(line).iter().any(|(sep, _)| sep.is_some()) {
        return true;
    }
    let mut quote: Option<char> = None;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == '>' => return true,
            None => {}
        }
    }
    false
}

/// What the text after the last unquoted operator is completing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailKind {
    /// A new command begins here — after `|`, `|&`, `||`, `&&` or `;`.
    Command,
    /// A redirection target, which is a **path whatever the command is** —
    /// after `>` or `>>`.
    RedirectPath,
}

/// Byte offset where the completable tail of `line` begins, and what it is.
///
/// This is what the composer's suggestion popup needs: after `|` a *new*
/// command starts, so offering the first command's path arguments is wrong.
/// Returns `(0, Command)` when there is no unquoted operator — the whole line —
/// so a caller that always consults this behaves exactly as before on a plain
/// command.
///
/// Scans bytes, which is safe because every operator and quote is ASCII and no
/// UTF-8 continuation byte can collide with one; the offsets are therefore
/// already byte offsets into `line`.
pub fn completion_tail(line: &str) -> (usize, TailKind) {
    let b = line.as_bytes();
    let mut quote: Option<u8> = None;
    let mut out = (0usize, TailKind::Command);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => {
                quote = Some(c);
                i += 1;
            }
            b'|' => {
                // `|`, `|&` and `||` all start a new command; only the width
                // differs.
                let w = if matches!(b.get(i + 1), Some(b'|') | Some(b'&')) { 2 } else { 1 };
                out = (i + w, TailKind::Command);
                i += w;
            }
            b'&' if b.get(i + 1) == Some(&b'&') => {
                out = (i + 2, TailKind::Command);
                i += 2;
            }
            b';' => {
                out = (i + 1, TailKind::Command);
                i += 1;
            }
            b'>' => {
                let w = if b.get(i + 1) == Some(&b'>') { 2 } else { 1 };
                out = (i + w, TailKind::RedirectPath);
                i += w;
            }
            _ => i += 1,
        }
    }
    out
}

/// Remove ANSI escape sequences.
///
/// Commands colourise for the console — `/cat` and `/head` syntax-highlight —
/// and those escapes land in the capture verbatim. Forwarding them would make a
/// downstream `/grep` match against text with colour codes embedded in it,
/// which fails for no visible reason, and would put them in a redirected file.
/// Piped and redirected data is therefore plain.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: ends at the first byte in @..~.
            Some('[') => {
                chars.next();
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: ends at BEL or ESC \.
            Some(']') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            // A lone ESC: drop it rather than emitting a stray control byte.
            _ => {}
        }
    }
    out
}

/// Split `line` at unquoted operators into `(separator_before, text)` pieces.
/// The first piece's separator is `None`.
fn scan(line: &str) -> Vec<(Option<Sep>, String)> {
    let mut out: Vec<(Option<Sep>, String)> = Vec::new();
    let mut cur = String::new();
    let mut pending: Option<Sep> = None;
    let mut quote: Option<char> = None;
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            cur.push(c);
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
            cur.push(c);
            i += 1;
            continue;
        }
        let next = bytes.get(i + 1).copied();
        let sep = match (c, next) {
            ('|', Some('&')) => Some((Sep::Pipe(PipeKind::Both), 2)),
            ('|', Some('|')) => Some((Sep::Join(Join::Or), 2)),
            ('|', _) => Some((Sep::Pipe(PipeKind::Stdout), 1)),
            ('&', Some('&')) => Some((Sep::Join(Join::And), 2)),
            (';', _) => Some((Sep::Join(Join::Seq), 1)),
            _ => None,
        };
        if let Some((s, width)) = sep {
            out.push((pending, cur.clone()));
            cur.clear();
            pending = Some(s);
            i += width;
            continue;
        }
        cur.push(c);
        i += 1;
    }
    out.push((pending, cur));
    out
}

/// Parse a command line into a [`Script`].
///
/// `line` is the text after the leading `/` of the first command; later stages
/// may be written with or without their own `/`.
pub fn parse(line: &str) -> Result<Script, &'static str> {
    if line.contains('\0') {
        return Err("invalid character in command line");
    }
    // An unterminated quote means the operator scan was guessing about where
    // the quoted run ended, so the split cannot be trusted.
    if unbalanced_quotes(line) {
        return Err("unterminated quote");
    }
    let pieces = scan(line);

    let mut pipelines: Vec<Pipeline> = Vec::new();
    let mut joins: Vec<Join> = Vec::new();
    let mut cur = Pipeline { stages: Vec::new(), pipes: Vec::new() };

    for (sep, text) in pieces {
        match sep {
            // First piece.
            None => {}
            Some(Sep::Pipe(k)) => cur.pipes.push(k),
            Some(Sep::Join(j)) => {
                pipelines.push(core::mem::replace(
                    &mut cur,
                    Pipeline { stages: Vec::new(), pipes: Vec::new() },
                ));
                joins.push(j);
            }
        }
        cur.stages.push(parse_stage(&text)?);
    }
    pipelines.push(cur);

    for p in &pipelines {
        if p.pipes.len() + 1 != p.stages.len() {
            return Err("malformed pipeline");
        }
    }
    Ok(Script { pipelines, joins })
}

fn unbalanced_quotes(line: &str) -> bool {
    let mut quote: Option<char> = None;
    for c in line.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None => {}
        }
    }
    quote.is_some()
}

/// Parse one stage: `[/]name [args] [> path | >> path]`.
fn parse_stage(text: &str) -> Result<Stage, &'static str> {
    let (body, redir) = split_redirect(text)?;
    let body = body.trim();
    if body.is_empty() {
        return Err("empty command between operators");
    }
    let body = body.strip_prefix('/').unwrap_or(body);
    let (name, arg) = match body.find(char::is_whitespace) {
        Some(i) => (&body[..i], body[i..].trim()),
        None => (body, ""),
    };
    if name.is_empty() {
        return Err("empty command name");
    }
    // Same rule `tools::shell_cmd::parse_command_line` applies: a command name
    // is a plain identifier. Anything else here means the split went wrong.
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        return Err("invalid command name");
    }
    // A pipeline is automation, and these commands exist only for a human at
    // this console. They are absent from `dispatch_system` so a stage could not
    // reach them anyway — refusing here says why instead of "not available".
    if crate::shell::catalog::is_human_only(name) {
        return Err("that command cannot be used in a pipeline");
    }
    Ok(Stage { name: name.to_string(), arg: arg.to_string(), redir })
}

/// Pull a trailing `>`/`>>` redirection out of a stage's text.
fn split_redirect(text: &str) -> Result<(String, Redir), &'static str> {
    let chars: Vec<char> = text.chars().collect();
    let mut quote: Option<char> = None;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
            i += 1;
            continue;
        }
        if c == '>' {
            let append = chars.get(i + 1) == Some(&'>');
            let body: String = chars[..i].iter().collect();
            let rest: String = chars[i + if append { 2 } else { 1 }..].iter().collect();
            let path = rest.trim();
            if path.is_empty() {
                return Err("redirection needs a path");
            }
            // One redirection per stage. A second `>` is far more likely a typo
            // than an intent, and silently honouring the last one hides it.
            if path.contains('>') {
                return Err("only one redirection per command");
            }
            let r = if append {
                Redir::Append(path.to_string())
            } else {
                Redir::Write(path.to_string())
            };
            return Ok((body, r));
        }
        i += 1;
    }
    Ok((text.to_string(), Redir::None))
}

/// Split a stage's captured output into `(stdout, stderr)` by the `name> `
/// convention. Line terminators are preserved on both sides.
pub fn split_streams(out: &str, cmd: &str) -> (String, String) {
    let mut prefix = String::with_capacity(cmd.len() + 2);
    prefix.push_str(cmd);
    prefix.push_str("> ");
    let mut stdout = String::new();
    let mut stderr = String::new();
    for line in out.split_inclusive('\n') {
        if line.starts_with(&prefix) {
            stderr.push_str(line);
        } else {
            stdout.push_str(line);
        }
    }
    (stdout, stderr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn names(s: &Script) -> Vec<&str> {
        s.stages().map(|st| st.name.as_str()).collect()
    }

    #[test_case]
    fn redirection_counts_as_an_operator() {
        // `>` binds inside a stage rather than between stages, so it is not one
        // of `scan`'s separators — and checking only those made
        // `/head -n 3 f > out` skip the pipeline path entirely, after which
        // /head saw `f > out` as its argument and said "only one path at a
        // time". Found on a real boot, not in review.
        assert!(has_operator("head -n 3 f > out"));
        assert!(has_operator("ls >> log"));
        // A quoted `>` is an argument, so the line must still take the plain
        // path.
        assert!(!has_operator(r#"grep ">" f"#));
        assert!(!has_operator("grep '>' f"));
    }

    #[test_case]
    fn ansi_is_stripped_from_data_that_leaves_a_stage() {
        // `/cat` and `/head` syntax-highlight for the console. Forwarding those
        // escapes makes a downstream `/grep` match against text with colour
        // codes embedded in it — it fails to match and nothing says why — and
        // puts them in a redirected file. Also found on a real boot: a piped
        // `/pbcopy` captured `\x1b[38;2;204;120;92m# ChittiOS…`.
        let coloured = "\x1b[38;2;204;120;92m# ChittiOS\x1b[0m sample files\n";
        assert_eq!(strip_ansi(coloured), "# ChittiOS sample files\n");
        // OSC (the clipboard push) terminates at BEL or ESC-backslash.
        assert_eq!(strip_ansi("a\x1b]52;c;aGk=\x07b"), "ab");
        assert_eq!(strip_ansi("a\x1b]0;title\x1b\\b"), "ab");
        // Plain text is untouched, and a lone ESC does not leak through.
        assert_eq!(strip_ansi("plain"), "plain");
        assert_eq!(strip_ansi("a\x1bb"), "ab");
        // A CSI ends at the first byte in @..~, so the text after it survives.
        assert_eq!(strip_ansi("\x1b[0mkept"), "kept");
    }

    #[test_case]
    fn a_plain_command_is_recognised_as_needing_none_of_this() {
        // The REPL must take its ordinary path byte for byte when there is no
        // operator, so this is the gate that keeps every existing command
        // unaffected.
        assert!(!has_operator("head -n 3 /x"));
        assert!(!has_operator("clip hello world"));
        let s = parse("head -n 3 /x").unwrap();
        assert!(s.is_single_command());
        assert_eq!(s.pipelines[0].stages[0].name, "head");
        assert_eq!(s.pipelines[0].stages[0].arg, "-n 3 /x");
    }

    #[test_case]
    fn pipes_bind_tighter_than_and_or_and_semicolon() {
        // `a && b | c` is `a && (b | c)`, not `(a && b) | c`. A flat list of
        // stages cannot say that, which is why the parse is two levels.
        let s = parse("a && b | c").unwrap();
        assert_eq!(s.pipelines.len(), 2);
        assert_eq!(s.joins, vec![Join::And]);
        assert_eq!(s.pipelines[0].stages.len(), 1);
        assert_eq!(s.pipelines[1].stages.len(), 2);
        assert_eq!(s.pipelines[1].pipes, vec![PipeKind::Stdout]);

        // And the mirror: `a | b && c` groups the pipe first.
        let s = parse("a | b && c").unwrap();
        assert_eq!(s.pipelines.len(), 2);
        assert_eq!(s.pipelines[0].stages.len(), 2);
        assert_eq!(s.pipelines[1].stages.len(), 1);
        assert_eq!(names(&s), vec!["a", "b", "c"]);
    }

    #[test_case]
    fn every_operator_is_distinguished_from_its_lookalike() {
        // `|` vs `|&` vs `||` differ by one character and mean entirely
        // different things; so do `&&` and a bare `&`.
        assert_eq!(parse("a | b").unwrap().pipelines[0].pipes, vec![PipeKind::Stdout]);
        assert_eq!(parse("a |& b").unwrap().pipelines[0].pipes, vec![PipeKind::Both]);
        assert_eq!(parse("a || b").unwrap().joins, vec![Join::Or]);
        assert_eq!(parse("a && b").unwrap().joins, vec![Join::And]);
        assert_eq!(parse("a ; b").unwrap().joins, vec![Join::Seq]);
        // Mixed, in one line.
        let s = parse("a | b |& c ; d && e || f").unwrap();
        assert_eq!(names(&s), vec!["a", "b", "c", "d", "e", "f"]);
        assert_eq!(s.pipelines[0].pipes, vec![PipeKind::Stdout, PipeKind::Both]);
        assert_eq!(s.joins, vec![Join::Seq, Join::And, Join::Or]);
    }

    #[test_case]
    fn quotes_protect_operators_and_are_passed_through_verbatim() {
        // The quote handling exists to stop the SPLIT, not to do word
        // splitting: commands here parse their own arguments and have never
        // understood quotes, so stripping them would change what an existing
        // command receives.
        assert!(!has_operator(r#"grep "a | b" file"#));
        let s = parse(r#"grep "a | b" file"#).unwrap();
        assert!(s.is_single_command());
        assert_eq!(s.pipelines[0].stages[0].arg, r#""a | b" file"#);

        // Single quotes too, and an operator outside the quotes still splits.
        let s = parse("grep 'x && y' f && ls").unwrap();
        assert_eq!(names(&s), vec!["grep", "ls"]);
        assert_eq!(s.pipelines[0].stages[0].arg, "'x && y' f");
    }

    #[test_case]
    fn an_unterminated_quote_is_refused_rather_than_guessed_at() {
        // With an open quote the scanner cannot know where the quoted run
        // ended, so any split it produced would be a guess.
        assert_eq!(parse(r#"grep "a | b file"#), Err("unterminated quote"));
        assert_eq!(parse("grep 'oops"), Err("unterminated quote"));
    }

    #[test_case]
    fn redirection_is_parsed_per_stage_and_needs_a_path() {
        let s = parse("ls / > /tmp/out").unwrap();
        assert_eq!(s.pipelines[0].stages[0].redir, Redir::Write("/tmp/out".to_string()));
        assert_eq!(s.pipelines[0].stages[0].arg, "/");
        let s = parse("ls / >> /tmp/out").unwrap();
        assert_eq!(s.pipelines[0].stages[0].redir, Redir::Append("/tmp/out".to_string()));
        // `>>` must not be read as `>` followed by a path starting with `>`.
        assert!(!matches!(s.pipelines[0].stages[0].redir, Redir::Write(_)));
        // A redirection binds to its own stage, not the whole line.
        let s = parse("cat a > x | grep b").unwrap();
        assert_eq!(s.pipelines[0].stages[0].redir, Redir::Write("x".to_string()));
        assert_eq!(s.pipelines[0].stages[1].redir, Redir::None);
        // A quoted `>` is an argument, not a redirection.
        let s = parse(r#"grep ">" f"#).unwrap();
        assert_eq!(s.pipelines[0].stages[0].redir, Redir::None);

        assert_eq!(parse("ls >"), Err("redirection needs a path"));
        assert_eq!(parse("ls > a > b"), Err("only one redirection per command"));
    }

    #[test_case]
    fn malformed_lines_say_what_is_wrong() {
        assert_eq!(parse("ls |"), Err("empty command between operators"));
        assert_eq!(parse("| ls"), Err("empty command between operators"));
        assert_eq!(parse("ls && "), Err("empty command between operators"));
        assert_eq!(parse("ls | | ls"), Err("empty command between operators"));
    }

    #[test_case]
    fn a_human_only_command_cannot_appear_in_a_pipeline() {
        // `/passwd` and `/lock` are absent from `dispatch_system` entirely, so
        // a stage could never reach them — this refuses at the parse with a
        // reason rather than letting it run and print "not available".
        for name in crate::shell::catalog::RESERVED_HUMAN_ONLY {
            let line = alloc::format!("ls | {name}");
            assert_eq!(
                parse(&line),
                Err("that command cannot be used in a pipeline"),
                "{name} was accepted as a pipeline stage"
            );
        }
        // And it is the *name* that is refused, not the substring: a command
        // merely containing one must still parse.
        assert!(parse("ls | grep passwd").is_ok());
    }

    #[test_case]
    fn a_leading_slash_on_later_stages_is_accepted() {
        // Nobody types `/head x | grep y`; they type `| /grep y`.
        let s = parse("head /x | /grep y").unwrap();
        assert_eq!(names(&s), vec!["head", "grep"]);
        assert_eq!(s.pipelines[0].stages[1].arg, "y");
    }

    #[test_case]
    fn streams_split_on_the_command_prefix() {
        // The whole stdout/stderr design in one assertion: `name> ` lines are
        // diagnostics, everything else is data — which is what lets pipes work
        // with no change to any command handler.
        let out = "cat> /x (12 bytes):\nhello\nworld\ncat> done\n";
        let (o, e) = split_streams(out, "cat");
        assert_eq!(o, "hello\nworld\n");
        assert_eq!(e, "cat> /x (12 bytes):\ncat> done\n");

        // Another command's prefix is data as far as this stage is concerned —
        // which is exactly right when one stage's output was piped into it.
        let (o, e) = split_streams("ls> /\ndata\n", "cat");
        assert_eq!(o, "ls> /\ndata\n");
        assert_eq!(e, "");
    }

    #[test_case]
    fn splitting_preserves_line_endings_and_an_unterminated_tail() {
        // A file whose last line has no newline must not gain one, and must not
        // be dropped.
        let (o, e) = split_streams("a\nb", "cat");
        assert_eq!(o, "a\nb");
        assert_eq!(e, "");
        let (o, e) = split_streams("cat> only", "cat");
        assert_eq!(o, "");
        assert_eq!(e, "cat> only");
        // Empty in, empty out — not a stray newline.
        assert_eq!(split_streams("", "cat"), (String::new(), String::new()));
        // A prefix that is not followed by a space is NOT a diagnostic:
        // `cat>x` is data. The convention is `name> `, with the space.
        let (o, _) = split_streams("cat>x\n", "cat");
        assert_eq!(o, "cat>x\n");
    }
}

// =====================================================================
// Piped input
// =====================================================================

/// The upstream stage's stdout, waiting for the current stage to read it.
///
/// Commands here take a **path**, not stdin, so "piped input" is offered rather
/// than injected: a stage reads it by naming `-` or by omitting its path, the
/// way `head` and `grep` read stdin. Appending the upstream text to the
/// downstream argument instead would make `/ls | /grep foo` pass a directory
/// listing as grep's *path*, which is the kind of plausible-and-wrong this
/// codebase avoids.
static PIPED: crate::mm::Locked<Option<String>> = crate::mm::Locked::new(None);
/// Whether the stage that was offered piped input actually read it, so the
/// runner can say so instead of silently discarding data.
static PIPED_TAKEN: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Offer `text` to the next stage (or clear the offer with `None`).
pub fn set_piped(text: Option<String>) {
    PIPED.with(|p| *p = text);
    PIPED_TAKEN.store(false, core::sync::atomic::Ordering::Relaxed);
}

/// Read the piped input, if a pipeline is feeding this stage.
pub fn take_piped() -> Option<String> {
    let v = PIPED.with(|p| p.clone());
    if v.is_some() {
        PIPED_TAKEN.store(true, core::sync::atomic::Ordering::Relaxed);
    }
    v
}

/// Whether piped input is on offer (without marking it read).
pub fn piped_available() -> bool {
    PIPED.with(|p| p.is_some())
}

/// Bytes offered but never read by the stage, if any.
pub fn piped_unread() -> Option<usize> {
    if PIPED_TAKEN.load(core::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    PIPED.with(|p| p.as_ref().map(|s| s.len()))
}

// =====================================================================
// Runner
// =====================================================================

/// Run one stage and return its captured output.
///
/// `Redirect` rather than `Tee`: an intermediate stage must not paint the
/// console with output the user never asked to see. Sets the status to
/// [`crate::shell::status::NOT_FOUND`] when the name is not a system command,
/// so `&&` after a typo does not proceed.
fn run_stage(stage: &Stage, piped: Option<String>) -> String {
    crate::shell::status::reset();
    set_piped(piped);
    crate::serial::sink_push(crate::serial::SinkMode::Redirect);
    let handled = crate::shell::dispatch_system(&stage.name, &stage.arg);
    let out = crate::serial::sink_pop().unwrap_or_default();
    if !handled {
        crate::shell::status::fail(crate::shell::status::NOT_FOUND);
    }
    let unread = piped_unread();
    set_piped(None);
    let mut out = out;
    if !handled {
        out.push_str(&alloc::format!(
            "{}> not a system command (interactive-only commands cannot be piped)\n",
            stage.name
        ));
    } else if let Some(n) = unread {
        // Never silently discard upstream data: a command that takes a path and
        // was given one ignores the pipe, and the user needs to know their
        // first stage did nothing.
        out.push_str(&alloc::format!(
            "{}> ignored {n} byte(s) of piped input (it takes a path; pass `-` to read the pipe)\n",
            stage.name
        ));
    }
    out
}

/// Write a stage's stdout to a file for `>` / `>>`.
fn apply_redirect(redir: &Redir, data: &str) -> Result<(), &'static str> {
    let (path, append) = match redir {
        Redir::None => return Ok(()),
        Redir::Write(p) => (p, false),
        Redir::Append(p) => (p, true),
    };
    let full = super::resolve_path(path);
    let mut bytes = alloc::vec::Vec::new();
    if append {
        // `vfs::write` replaces, so append is read-modify-write. On a `/host`
        // 9P file that costs a full re-read of the file each time — real, and
        // worth knowing before using `>>` in a loop.
        if let Ok(existing) = crate::fs::vfs::read(&full) {
            bytes.extend_from_slice(&existing);
        }
    }
    bytes.extend_from_slice(data.as_bytes());
    // The same split every write command here makes: a path on a *mounted*
    // volume goes through the VFS, everything else through the Synapse store.
    // Routing a store path to `vfs::write` gets `NotMounted` — on a dev boot
    // with no data partition that is every path, so `>` failed for everything.
    if super::fs::path_on_mount(&full) {
        crate::fs::vfs::write(&full, &bytes).map_err(|_| "could not write the redirect target")
    } else {
        crate::synapse::fs::write(&full, &bytes);
        Ok(())
    }
}

/// Run one pipeline (stages joined by `|` / `|&`), printing the last stage's
/// stdout unless it is redirected. Returns the last stage's status.
fn run_pipeline(pl: &Pipeline) -> i32 {
    let mut carry: Option<String> = None;
    for (i, stage) in pl.stages.iter().enumerate() {
        // Every stage is interruptible as a whole, not only inside itself:
        // Ctrl+C must abandon the rest of the pipeline, not just the command
        // that happened to be running.
        if crate::shell::poll_interrupt() {
            crate::serial_println!("pipeline> cancelled");
            return crate::shell::status::FAILURE;
        }
        let out = run_stage(stage, carry.take());
        let (stdout, stderr) = split_streams(&out, &stage.name);
        // Diagnostics always reach the human, at every stage — a pipeline that
        // swallowed "not found" would look like it produced nothing for no
        // reason.
        if !stderr.is_empty() {
            crate::serial_print!("{}", stderr);
        }
        let last = i + 1 == pl.stages.len();
        // Data leaving this stage is stripped; data going to the console is
        // not. `/cat` and `/head` syntax-highlight, and those escapes would
        // otherwise make a downstream `/grep` match against text with colour
        // codes in it — and land in a redirected file.
        if let Some(redir) = Some(&stage.redir).filter(|r| **r != Redir::None) {
            if let Err(e) = apply_redirect(redir, &strip_ansi(&stdout)) {
                crate::serial_println!("pipeline> {e}");
                crate::shell::status::fail1();
            }
            // A redirect consumes this stage's stdout, so a following stage
            // gets nothing — exactly as sh does.
            if !last {
                carry = Some(String::new());
            }
            continue;
        }
        if last {
            crate::serial_print!("{}", stdout);
            continue;
        }
        let forward = strip_ansi(match pl.pipes.get(i) {
            Some(PipeKind::Both) => &out,
            _ => &stdout,
        });
        if forward.len() > crate::serial::SINK_MAX {
            crate::serial_println!("pipeline> too much data between stages (over 1 MiB)");
            return crate::shell::status::FAILURE;
        }
        carry = Some(forward);
    }
    crate::shell::status::get()
}

/// Run a whole script, honouring `;`, `&&` and `||`.
pub fn run(script: &Script) {
    let mut status = 0;
    let mut warned_unreported = false;
    for (i, pl) in script.pipelines.iter().enumerate() {
        if i > 0 {
            let join = script.joins[i - 1];
            let run_it = match join {
                Join::Seq => true,
                Join::And => status == 0,
                Join::Or => status != 0,
            };
            // `&&`/`||` after a command that never reports failure would imply
            // a check that is not happening. Say so once per line rather than
            // per stage, and only for the conditional joins.
            if matches!(join, Join::And | Join::Or) && !warned_unreported {
                if let Some(prev) = script.pipelines[i - 1].stages.last() {
                    if !crate::shell::status::reports_status(&prev.name) {
                        crate::serial_println!(
                            "pipeline> note: /{} does not report failure, so it always counts as success",
                            prev.name
                        );
                        warned_unreported = true;
                    }
                }
            }
            if !run_it {
                continue;
            }
        }
        status = run_pipeline(pl);
    }
}
