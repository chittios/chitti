//! Pure path/content helpers for agent tools (`glob`, `grep`, line-range
//! `read`, safer `edit`). No I/O — callers supply path lists and file bytes so
//! unit tests exercise the logic off-hardware, and the Router applies
//! capability scope filters before calling in.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

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

/// Search `files` (path, utf-8 content) for lines containing `query` (substring,
/// case-sensitive). Caps total hits at `max_hits`.
pub fn grep_files(query: &str, files: &[(String, String)], max_hits: usize) -> Vec<GrepHit> {
    let mut out = Vec::new();
    if query.is_empty() || max_hits == 0 {
        return out;
    }
    for (path, content) in files {
        for (i, line) in content.lines().enumerate() {
            if line.contains(query) {
                let text = if line.len() > 200 {
                    let mut t = line[..200].to_string();
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
