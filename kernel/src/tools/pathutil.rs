//! Pure path/content helpers for agent tools (`glob`, `grep`, line-range
//! `read`, safer `edit`). No I/O — callers supply path lists and file bytes so
//! unit tests exercise the logic off-hardware, and the Router applies
//! capability scope filters before calling in.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The longest prefix of `s` that is at most `max` **bytes** and ends on a character
/// boundary.
///
/// `&s[..max]` panics when `max` lands inside a multi-byte character, and the byte that
/// gets cut is not under our control: it is a byte of somebody's file. `grep` truncated
/// its preview with a raw slice at 200 and **panicked the kernel** on a line whose `…`
/// occupied bytes 198..201 — a crash reachable by grepping any file containing
/// non-ASCII text near a 200-byte offset, which includes this repo's own docs.
///
/// The walk-back loop existed correctly in four other places, hand-written each time
/// ([`crate::agent::prompt::bound_tool_result`], `agent::home`, `synapse::abi`). Five
/// copies is how one comes to be missing, so this is the shared one.
pub fn truncate_on_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Glob match against a full path. Supports `*`, `?`, and `**` (any path
/// segment sequence including `/`). Patterns without `/` match the basename
/// or the full path (so `*.md` matches `/agent/1/MEMORY.md`).
pub fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    // Basename convenience: `*.rs` matches any path ending with a matching name.
    if !pattern.contains('/') {
        let base = path.rsplit('/').next().unwrap_or(path);
        if glob_match_segments(pattern, base) {
            return true;
        }
    }
    glob_match_segments(pattern, path)
}

fn glob_match_segments(pattern: &str, path: &str) -> bool {
    // Split on `**` for recursive directory match.
    if let Some(i) = pattern.find("**") {
        let (pre, rest) = pattern.split_at(i);
        let post = rest.trim_start_matches('*').trim_start_matches('/');
        let pre = pre.trim_end_matches('/');
        if !pre.is_empty() {
            // Path must start with pre (as prefix with optional trailing match).
            if !path.starts_with(pre) {
                // Also allow pre as path prefix of a segment boundary.
                if !(path.len() > pre.len() && path.as_bytes().get(pre.len()) == Some(&b'/') && path.starts_with(pre)) {
                    // Try segment-level: pre may itself contain * ?
                    if !starts_with_glob(pre, path) {
                        return false;
                    }
                }
            }
        }
        if post.is_empty() {
            return true;
        }
        // Find any suffix of path that matches post.
        if path.is_empty() {
            return glob_match_simple(post, "");
        }
        // Walk every position (including 0 and end).
        let bytes = path.as_bytes();
        for start in 0..=bytes.len() {
            if start > 0 && bytes[start - 1] != b'/' && start != bytes.len() {
                // Prefer starting at segment boundaries, but also allow mid-path
                // for patterns like `**/*.md`.
                if !post.starts_with('*') && start < bytes.len() {
                    continue;
                }
            }
            let suffix = &path[start..];
            let s = suffix.trim_start_matches('/');
            if glob_match_simple(post, s) || glob_match_simple(post, suffix) {
                return true;
            }
        }
        return false;
    }
    glob_match_simple(pattern, path)
}

fn starts_with_glob(pattern: &str, path: &str) -> bool {
    // Match pattern as a prefix of path (path may continue after).
    if pattern.is_empty() {
        return true;
    }
    // Try every prefix of path as a match for pattern.
    for end in 0..=path.len() {
        if end < path.len() && path.as_bytes()[end] != b'/' && end != 0 {
            // only cut on segment boundary for multi-seg patterns with /
            if pattern.contains('/') {
                continue;
            }
        }
        if glob_match_simple(pattern, &path[..end]) {
            // remainder should start with / or be empty when pattern had structure
            if end == path.len() || path.as_bytes()[end] == b'/' {
                return true;
            }
            if !pattern.contains('/') {
                return true;
            }
        }
    }
    false
}

/// Single-level glob: `*` = any run of non-`/` (or any if pattern has no slash
/// constraint handled by caller), `?` = one char. For full paths, `*` does not
/// cross `/`.
fn glob_match_simple(pattern: &str, text: &str) -> bool {
    glob_match_star(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_star(pat: &[u8], text: &[u8]) -> bool {
    let mut pi = 0usize;
    let mut ti = 0usize;
    let mut star_pi = usize::MAX;
    let mut star_ti = 0usize;
    while ti < text.len() {
        if pi < pat.len() && (pat[pi] == b'?' || (pat[pi] != b'*' && pat[pi] == text[ti])) {
            // `?` does not match `/`
            if pat[pi] == b'?' && text[ti] == b'/' {
                if star_pi != usize::MAX {
                    star_ti += 1;
                    ti = star_ti;
                    pi = star_pi + 1;
                    continue;
                }
                return false;
            }
            if pat[pi] != b'*' {
                pi += 1;
                ti += 1;
                continue;
            }
        }
        if pi < pat.len() && pat[pi] == b'*' {
            // `*` stops at `/` for path segments.
            star_pi = pi;
            star_ti = ti;
            pi += 1;
            continue;
        }
        if star_pi != usize::MAX {
            // Grow the star match, but not across `/` unless the star is meant
            // to absorb non-slash only — so stop at slash: fail this branch.
            if text[star_ti] == b'/' {
                return false;
            }
            star_ti += 1;
            ti = star_ti;
            pi = star_pi + 1;
            continue;
        }
        return false;
    }
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len()
}

/// Filter `paths` by a glob pattern. Returns matching paths (sorted by input order).
pub fn glob_filter(pattern: &str, paths: &[String]) -> Vec<String> {
    paths.iter().filter(|p| glob_match(pattern, p)).cloned().collect()
}

/// One grep hit: path, 1-based line number, line text (trimmed to `max_line`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrepHit {
    pub path: String,
    pub line: u32,
    pub text: String,
}

/// Search `files` (path, utf-8 content) for lines containing `query` (substring).
/// Caps total hits at `max_hits`. When `case_insensitive`, both sides are
/// lowercased for the match (hit text stays original).
pub fn grep_files(query: &str, files: &[(String, String)], max_hits: usize) -> Vec<GrepHit> {
    grep_files_ex(query, files, max_hits, false)
}

/// Extended grep with case folding.
pub fn grep_files_ex(
    query: &str,
    files: &[(String, String)],
    max_hits: usize,
    case_insensitive: bool,
) -> Vec<GrepHit> {
    let mut out = Vec::new();
    if query.is_empty() || max_hits == 0 {
        return out;
    }
    let q = if case_insensitive {
        query.to_ascii_lowercase()
    } else {
        query.to_string()
    };
    for (path, content) in files {
        for (i, line) in content.lines().enumerate() {
            let hay = if case_insensitive {
                line.to_ascii_lowercase()
            } else {
                line.to_string()
            };
            if hay.contains(&q) {
                let text = if line.len() > 200 {
                    let mut t = truncate_on_char_boundary(line, 200).to_string();
                    t.push('…');
                    t
                } else {
                    line.to_string()
                };
                out.push(GrepHit {
                    path: path.clone(),
                    line: (i + 1) as u32,
                    text,
                });
                if out.len() >= max_hits {
                    return out;
                }
            }
        }
    }
    out
}

/// Direct children of `dir` (store paths). `dir` may be `""` or `"/"` for root.
/// Returns relative names for children (files + implied subdirs), sorted.
pub fn list_dir_children(dir: &str, paths: &[String]) -> Vec<String> {
    let prefix = normalize_dir_prefix(dir);
    let mut kids: Vec<String> = Vec::new();
    for p in paths {
        let rest = if prefix.is_empty() {
            p.trim_start_matches('/')
        } else if let Some(r) = p.strip_prefix(&prefix) {
            r.trim_start_matches('/')
        } else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }
        let name = rest.split('/').next().unwrap_or(rest);
        if name.is_empty() {
            continue;
        }
        // Directory vs file: if more path remains, mark as dir with trailing /
        let entry = if rest.contains('/') {
            format!("{name}/")
        } else {
            name.to_string()
        };
        if !kids.iter().any(|k| k == &entry) {
            kids.push(entry);
        }
    }
    kids.sort();
    kids
}

fn normalize_dir_prefix(dir: &str) -> String {
    let d = dir.trim();
    if d.is_empty() || d == "/" {
        return String::new();
    }
    let d = d.trim_end_matches('/');
    if d.starts_with('/') {
        d.to_string()
    } else {
        format!("/{d}")
    }
}

/// Slice `content` to lines `[start_line, end_line]` inclusive (1-based).
/// `start_line == 0` means 1; `end_line == 0` means last line. Caps output at
/// `max_bytes` (appends a truncation marker when cut).
pub fn line_range(content: &str, start_line: u32, end_line: u32, max_bytes: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let start = if start_line == 0 { 1 } else { start_line as usize };
    let end = if end_line == 0 {
        lines.len()
    } else {
        end_line as usize
    };
    let start = start.clamp(1, lines.len());
    let end = end.clamp(start, lines.len());
    let mut out = String::new();
    for (idx, line) in lines[start - 1..end].iter().enumerate() {
        let n = start + idx;
        let row = alloc::format!("{n:>6}|{line}\n");
        if out.len() + row.len() > max_bytes {
            out.push_str(&alloc::format!("… truncated at {max_bytes} bytes\n"));
            break;
        }
        out.push_str(&row);
    }
    out
}

/// Safer edit: replace `old` with `new` in `content`.
/// - empty `old` → error
/// - 0 matches → error
/// - \>1 matches and `replace_all == false` → ambiguous error (require a unique match)
/// - else apply replacement(s)
pub fn safe_edit(content: &str, old: &str, new: &str, replace_all: bool) -> Result<String, &'static str> {
    if old.is_empty() {
        return Err("empty search string");
    }
    let count = content.matches(old).count();
    if count == 0 {
        return Err("substring not found");
    }
    if count > 1 && !replace_all {
        return Err("ambiguous: multiple matches (pass replace_all or a unique old string)");
    }
    if replace_all {
        Ok(content.replace(old, new))
    } else {
        Ok(content.replacen(old, new, 1))
    }
}

/// Cap a MEMORY.md body to `max_lines` and `max_bytes` (line cut first, then
/// byte cut at last newline). Appends a warning when truncated.
pub fn truncate_memory_md(raw: &str, max_lines: usize, max_bytes: usize) -> (String, bool) {
    let trimmed = raw.trim();
    let lines: Vec<&str> = trimmed.lines().collect();
    let mut truncated = false;
    let body = if lines.len() > max_lines {
        truncated = true;
        lines[..max_lines].join("\n")
    } else {
        trimmed.to_string()
    };
    let body = if body.len() > max_bytes {
        truncated = true;
        let cut = body[..max_bytes].rfind('\n').unwrap_or(max_bytes);
        body[..cut].to_string()
    } else {
        body
    };
    if truncated {
        (
            alloc::format!(
                "{body}\n\n> WARNING: MEMORY.md truncated (limit {max_lines} lines / {max_bytes} bytes)."
            ),
            true,
        )
    } else {
        (body, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    /// The kernel panicked here: `grep` cut its 200-byte preview with a raw slice, and a
    /// line whose `…` sat at bytes 198..201 split the character.
    ///
    /// `KERNEL PANIC: end byte index 200 is not a char boundary; it is inside '…'` — from
    /// `/search agent` over this repo's own files, so any document with non-ASCII text
    /// near a 200-byte offset was a crash waiting for a query to find it.
    #[test_case]
    fn grep_preview_cuts_a_long_line_on_a_char_boundary() {
        // Reproduce the panic's exact geometry: '…' at bytes 198..201, so the 200-byte
        // cut falls *inside* the character (not on either end of it).
        let mut line = String::from("agent");
        while line.len() < 198 {
            line.push('x');
        }
        line.push('…');
        while line.len() < 260 {
            line.push('y');
        }
        assert!(
            !line.is_char_boundary(200),
            "the fixture must put byte 200 inside a character, as the crash did"
        );
        let files = vec![(String::from("/doc.md"), line)];
        let hits = grep_files_ex("agent", &files, 8, false);
        assert_eq!(hits.len(), 1);
        // Truncated below the limit rather than panicking, and still valid UTF-8.
        assert!(hits[0].text.len() <= 201, "preview stays near the cap");
        assert!(hits[0].text.ends_with('…'), "an elided preview is marked");

        // A multi-byte character at every offset around the cut: each must be survivable.
        for pad in 195..=205 {
            let mut l = String::from("agent");
            while l.len() < pad {
                l.push('x');
            }
            l.push('é');
            while l.len() < 240 {
                l.push('z');
            }
            let f = vec![(String::from("/p"), l)];
            assert_eq!(grep_files_ex("agent", &f, 4, false).len(), 1, "pad {pad}");
        }
    }

    /// The shared helper: never past `max`, never inside a character, never lossy below.
    #[test_case]
    fn truncate_on_char_boundary_is_exact_and_safe() {
        assert_eq!(truncate_on_char_boundary("hello", 99), "hello", "short is untouched");
        assert_eq!(truncate_on_char_boundary("hello", 5), "hello", "exact fit");
        assert_eq!(truncate_on_char_boundary("hello", 2), "he");
        // '…' is 3 bytes: cutting at 1 or 2 must yield nothing, not a partial character.
        assert_eq!(truncate_on_char_boundary("…", 1), "");
        assert_eq!(truncate_on_char_boundary("…", 2), "");
        assert_eq!(truncate_on_char_boundary("…", 3), "…");
        assert_eq!(truncate_on_char_boundary("a…b", 2), "a");
        assert_eq!(truncate_on_char_boundary("a…b", 3), "a");
        assert_eq!(truncate_on_char_boundary("a…b", 4), "a…");
        // 4-byte characters too (the widest UTF-8 encodes).
        let emoji = "🙂🙂";
        for max in 0..=8 {
            let out = truncate_on_char_boundary(emoji, max);
            assert!(out.len() <= max, "max {max}: {} bytes", out.len());
            assert_eq!(out.len() % 4, 0, "max {max}: whole characters only");
        }
        assert_eq!(truncate_on_char_boundary("", 0), "");
    }

    #[test_case]
    fn glob_basics() {
        assert!(glob_match("*.md", "MEMORY.md"));
        assert!(glob_match("*.md", "/agent/1/MEMORY.md"));
        assert!(!glob_match("*.md", "/agent/1/note.txt"));
        assert!(glob_match("/agent/1/*", "/agent/1/MEMORY.md"));
        assert!(glob_match("**/memory/*", "/agent/7/memory/colour"));
        assert!(glob_match("?", "a"));
        assert!(!glob_match("?", "ab"));
        assert!(glob_match("a*c", "abc"));
        assert!(!glob_match("a*c", "a/b/c")); // * does not cross /
    }

    #[test_case]
    fn grep_and_line_range() {
        let files = vec![
            (String::from("/a"), String::from("hello\nworld\nhello again")),
            (String::from("/b"), String::from("nope")),
        ];
        let hits = grep_files("hello", &files, 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[1].line, 3);

        let sliced = line_range("one\ntwo\nthree\nfour", 2, 3, 4096);
        assert!(sliced.contains("two") && sliced.contains("three"));
        assert!(!sliced.contains("one"));
        assert!(sliced.contains("     2|"));
    }

    #[test_case]
    fn safe_edit_unique_and_ambiguous() {
        assert_eq!(safe_edit("a x a", "x", "y", false).unwrap(), "a y a");
        assert!(safe_edit("a x a", "a", "b", false).unwrap_err().contains("ambiguous"));
        assert_eq!(safe_edit("a x a", "a", "b", true).unwrap(), "b x b");
        assert!(safe_edit("abc", "", "x", false).is_err());
        assert!(safe_edit("abc", "z", "x", false).is_err());
    }

    #[test_case]
    fn memory_md_truncation() {
        let long = (0..300).map(|i| alloc::format!("line {i}")).collect::<Vec<_>>().join("\n");
        let (out, trunc) = truncate_memory_md(&long, 200, 25_000);
        assert!(trunc);
        assert!(out.contains("WARNING"));
        let (short, t2) = truncate_memory_md("just a note", 200, 25_000);
        assert!(!t2);
        assert_eq!(short, "just a note");
    }
}
