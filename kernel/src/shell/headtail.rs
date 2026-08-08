//! **`/head` and `/tail` argument parsing and selection** — pure, so the part
//! that is easy to get wrong is testable without a filesystem.
//!
//! The selection is byte-oriented rather than `str`-oriented on purpose: these
//! commands are pointed at whatever is on disk, including files that are not
//! UTF-8, and slicing a `String` at a line boundary computed from bytes is a
//! panic waiting for the first file with a multi-byte character near the cut.
//!
//! **The trap is `tail`, and it is the trailing newline.** A file ending in
//! `\n` has *n* lines, not *n + 1* — the final newline terminates the last
//! line, it does not begin an empty one. Counting `\n` from the end without
//! skipping that terminator makes `tail -n 1` return an empty slice for every
//! well-formed text file, which reads as "the file is empty" rather than as a
//! bug. [`select`] skips it; [`tail_of_a_newline_terminated_file_is_the_last_real_line`]
//! pins it.

/// Lines shown when no count is given, matching every `head`/`tail` a user has
/// typed before.
pub const DEFAULT_LINES: usize = 10;

/// What to take, and from which end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub count: usize,
    /// `-c`: count bytes rather than lines.
    pub bytes: bool,
    /// `tail` takes from the end; `head` from the start.
    pub from_end: bool,
}

impl Spec {
    pub fn lines(count: usize, from_end: bool) -> Spec {
        Spec { count, bytes: false, from_end }
    }
}

/// Parse `[-n N | -c N | -N] [path]` in any order.
///
/// Returns `(spec, path)` where **`path` is optional**: with no path the input
/// is whatever was piped in, exactly as `head` reads stdin. The caller decides
/// whether an absent path is an error, because only it knows whether a pipeline
/// is feeding this stage. `-` names piped input explicitly.
///
/// Flags may precede or follow the path because a user who types
/// `/tail file.log -n 50` after seeing the output should not have to retype the
/// line.
pub fn parse(arg: &str, from_end: bool) -> Result<(Spec, Option<&str>), &'static str> {
    let mut spec = Spec::lines(DEFAULT_LINES, from_end);
    let mut path: Option<&str> = None;
    let mut want_count: Option<bool> = None; // Some(bytes?) — a flag awaiting its value

    for tok in arg.split_whitespace() {
        // A pending `-n` / `-c` consumes this token as its count.
        if let Some(bytes) = want_count.take() {
            spec.count = parse_count(tok).ok_or("count must be a number")?;
            spec.bytes = bytes;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("-n").filter(|_| tok.starts_with("-n")) {
            if rest.is_empty() {
                want_count = Some(false);
            } else {
                spec.count = parse_count(rest).ok_or("count must be a number")?;
                spec.bytes = false;
            }
            continue;
        }
        if let Some(rest) = tok.strip_prefix("-c").filter(|_| tok.starts_with("-c")) {
            if rest.is_empty() {
                want_count = Some(true);
            } else {
                spec.count = parse_count(rest).ok_or("count must be a number")?;
                spec.bytes = true;
            }
            continue;
        }
        // A bare `-20`, which is how most people actually type it.
        if let Some(rest) = tok.strip_prefix('-') {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                spec.count = parse_count(rest).ok_or("count must be a number")?;
                spec.bytes = false;
                continue;
            }
            return Err("unknown option (try -n <lines> or -c <bytes>)");
        }
        if path.replace(tok).is_some() {
            // Two paths is far more likely a typo than a request to
            // concatenate, and silently using one of them hides it.
            return Err("only one path at a time");
        }
    }
    if want_count.is_some() {
        return Err("-n / -c needs a count");
    }
    Ok((spec, path))
}

/// Whether `path` names piped input rather than a file.
pub fn is_stdin(path: Option<&str>) -> bool {
    matches!(path, None | Some("-"))
}

fn parse_count(s: &str) -> Option<usize> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Take the selected slice of `data`.
pub fn select<'a>(data: &'a [u8], spec: &Spec) -> &'a [u8] {
    if spec.count == 0 {
        return &[];
    }
    if spec.bytes {
        return if spec.from_end {
            &data[data.len().saturating_sub(spec.count)..]
        } else {
            &data[..spec.count.min(data.len())]
        };
    }
    if spec.from_end {
        tail_lines(data, spec.count)
    } else {
        head_lines(data, spec.count)
    }
}

/// The first `n` lines, including their terminating newlines.
fn head_lines(data: &[u8], n: usize) -> &[u8] {
    let mut seen = 0;
    for (i, b) in data.iter().enumerate() {
        if *b == b'\n' {
            seen += 1;
            if seen == n {
                return &data[..=i];
            }
        }
    }
    // Fewer than `n` lines in the file: all of it.
    data
}

/// The last `n` lines.
fn tail_lines(data: &[u8], n: usize) -> &[u8] {
    // A trailing newline TERMINATES the last line; it does not begin a new
    // empty one. Skipping it here is the whole difference between `tail -n 1`
    // returning the last line and returning nothing.
    let mut i = data.len();
    if i > 0 && data[i - 1] == b'\n' {
        i -= 1;
    }
    let mut found = 0;
    while i > 0 {
        i -= 1;
        if data[i] == b'\n' {
            found += 1;
            if found == n {
                return &data[i + 1..];
            }
        }
    }
    data
}

/// How many lines `data` holds, by the same rule [`select`] uses — for the
/// "showing N of M" line, which is the only way to tell a truncated view from a
/// short file.
pub fn count_lines(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let newlines = data.iter().filter(|b| **b == b'\n').count();
    // An unterminated final line still counts.
    if data[data.len() - 1] == b'\n' {
        newlines
    } else {
        newlines + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THREE: &[u8] = b"alpha\nbeta\ngamma\n";
    const NO_NL: &[u8] = b"alpha\nbeta\ngamma";

    #[test_case]
    fn tail_of_a_newline_terminated_file_is_the_last_real_line() {
        // THE bug this module exists to avoid: a file ending in `\n` has three
        // lines, not four. Counting newlines from the end without skipping the
        // terminator makes `tail -n 1` return "" for every well-formed text
        // file — which reads as an empty file, not as a bug.
        assert_eq!(select(THREE, &Spec::lines(1, true)), b"gamma\n");
        assert_eq!(select(THREE, &Spec::lines(2, true)), b"beta\ngamma\n");
        assert_eq!(select(THREE, &Spec::lines(3, true)), THREE);
        // A file with no trailing newline behaves identically.
        assert_eq!(select(NO_NL, &Spec::lines(1, true)), b"gamma");
        assert_eq!(select(NO_NL, &Spec::lines(2, true)), b"beta\ngamma");
    }

    #[test_case]
    fn head_keeps_the_newline_that_terminates_each_line() {
        assert_eq!(select(THREE, &Spec::lines(1, false)), b"alpha\n");
        assert_eq!(select(THREE, &Spec::lines(2, false)), b"alpha\nbeta\n");
        // Asking for more lines than exist yields the whole file, not an error.
        assert_eq!(select(THREE, &Spec::lines(99, false)), THREE);
        assert_eq!(select(NO_NL, &Spec::lines(99, false)), NO_NL);
        // The last line of a file with no trailing newline is still included.
        assert_eq!(select(NO_NL, &Spec::lines(3, false)), NO_NL);
    }

    #[test_case]
    fn degenerate_inputs_do_not_panic_or_over_read() {
        for from_end in [false, true] {
            // Zero lines is empty, not "everything".
            assert_eq!(select(THREE, &Spec::lines(0, from_end)), b"");
            // An empty file yields an empty slice at any count.
            assert_eq!(select(b"", &Spec::lines(5, from_end)), b"");
            // A file that is only newlines has that many lines.
            assert_eq!(select(b"\n\n\n", &Spec::lines(1, from_end)).len(), 1);
            // A single unterminated line comes back whole from both ends.
            assert_eq!(select(b"solo", &Spec::lines(1, from_end)), b"solo");
        }
    }

    #[test_case]
    fn byte_mode_slices_bytes_and_clamps() {
        assert_eq!(select(THREE, &Spec { count: 5, bytes: true, from_end: false }), b"alpha");
        assert_eq!(select(THREE, &Spec { count: 6, bytes: true, from_end: true }), b"gamma\n");
        // A count past the end clamps rather than panicking.
        assert_eq!(select(THREE, &Spec { count: 9999, bytes: true, from_end: false }), THREE);
        assert_eq!(select(THREE, &Spec { count: 9999, bytes: true, from_end: true }), THREE);
    }

    #[test_case]
    fn selection_is_byte_oriented_so_non_utf8_is_safe() {
        // These commands are pointed at whatever is on disk. Slicing a String
        // at a byte offset computed from newlines panics on the first file with
        // a multi-byte character near the cut, so the whole path stays bytes.
        let data = b"caf\xc3\xa9 latte\n\xff\xfe binary\nlast\n";
        assert_eq!(select(data, &Spec::lines(1, false)), "café latte\n".as_bytes());
        assert_eq!(select(data, &Spec::lines(1, true)), b"last\n");
        // A cut landing inside a multi-byte character is a legal byte slice.
        assert_eq!(select(data, &Spec { count: 4, bytes: true, from_end: false }), b"caf\xc3");
    }

    #[test_case]
    fn counts_are_accepted_in_every_form_people_type() {
        let p = |s| Some(s);
        assert_eq!(parse("f.txt", false), Ok((Spec::lines(DEFAULT_LINES, false), p("f.txt"))));
        assert_eq!(parse("-n 5 f.txt", false), Ok((Spec::lines(5, false), p("f.txt"))));
        assert_eq!(parse("-n5 f.txt", false), Ok((Spec::lines(5, false), p("f.txt"))));
        assert_eq!(parse("-5 f.txt", false), Ok((Spec::lines(5, false), p("f.txt"))));
        // Flags after the path, because a user re-running with a bigger count
        // appends it rather than retyping the line.
        assert_eq!(parse("f.txt -n 5", false), Ok((Spec::lines(5, false), p("f.txt"))));
        assert_eq!(
            parse("-c 100 f.txt", true),
            Ok((Spec { count: 100, bytes: true, from_end: true }, p("f.txt")))
        );
        // `from_end` comes from which command was typed, never from the args.
        assert_eq!(parse("f.txt", true).unwrap().0.from_end, true);
    }

    #[test_case]
    fn an_absent_path_means_piped_input_not_an_error() {
        // The parser cannot know whether a pipeline is feeding this stage, so
        // it reports "no path" as a fact and the caller decides. `-` names
        // piped input explicitly, as it does everywhere else.
        assert_eq!(parse("", false), Ok((Spec::lines(DEFAULT_LINES, false), None)));
        assert_eq!(parse("-n 5", false), Ok((Spec::lines(5, false), None)));
        assert!(is_stdin(None));
        assert!(is_stdin(Some("-")));
        assert!(!is_stdin(Some("f.txt")));
    }

    #[test_case]
    fn a_malformed_invocation_says_what_is_wrong() {
        // Each of these used to be easy to write as "usage:" and leave the user
        // guessing which part was rejected.
        assert_eq!(parse("-n f.txt", false), Err("count must be a number"));
        assert_eq!(parse("f.txt -n", false), Err("-n / -c needs a count"));
        assert_eq!(parse("-z f.txt", false), Err("unknown option (try -n <lines> or -c <bytes>)"));
        assert_eq!(parse("a.txt b.txt", false), Err("only one path at a time"));
    }

    #[test_case]
    fn line_counting_agrees_with_selection() {
        assert_eq!(count_lines(THREE), 3);
        assert_eq!(count_lines(NO_NL), 3);
        assert_eq!(count_lines(b""), 0);
        assert_eq!(count_lines(b"\n"), 1);
        assert_eq!(count_lines(b"solo"), 1);
        // Asking for every line by count returns the whole file, which is what
        // makes "showing N of M" trustworthy.
        assert_eq!(select(THREE, &Spec::lines(count_lines(THREE), true)), THREE);
        assert_eq!(select(NO_NL, &Spec::lines(count_lines(NO_NL), false)), NO_NL);
    }
}
