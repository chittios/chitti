//! `.git/config` — the INI-with-subsections format git uses.
//!
//! Pure: text in, text out. Broken out because `remote`, `config`, `push`,
//! `fetch` and `branch --set-upstream` all read and write the same file, and a
//! writer that reformats it would silently drop any key it did not understand.
//! This one **preserves unknown sections and keys verbatim** and only touches
//! the line it was asked to.
//!
//! The format's one real subtlety is the subsection name: `[remote "origin"]`
//! is section `remote` with subsection `origin`, and the quoted part is
//! case-**sensitive** while the section name is not. Folding both cases makes
//! `[remote "Origin"]` and `[remote "origin"]` the same remote, which git says
//! they are not.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One `section.subsection.key = value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub section: String,
    pub subsection: Option<String>,
    pub key: String,
    pub value: String,
}

impl Entry {
    /// The dotted name `git config` uses: `remote.origin.url`.
    pub fn name(&self) -> String {
        match &self.subsection {
            Some(sub) => alloc::format!("{}.{}.{}", self.section, sub, self.key),
            None => alloc::format!("{}.{}", self.section, self.key),
        }
    }
}

/// Parse a config file. Unparseable lines are skipped, never fatal — one bad
/// line must not make a repository unusable.
pub fn parse(text: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut section = String::new();
    let mut subsection: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            match inner.split_once('"') {
                Some((name, rest)) => {
                    section = name.trim().to_ascii_lowercase();
                    subsection = Some(rest.trim_end_matches('"').to_string());
                }
                None => {
                    section = inner.trim().to_ascii_lowercase();
                    subsection = None;
                }
            }
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        if section.is_empty() {
            continue;
        }
        out.push(Entry {
            section: section.clone(),
            subsection: subsection.clone(),
            key: k.trim().to_ascii_lowercase(),
            value: v.trim().to_string(),
        });
    }
    out
}

/// Render entries back to config text, grouped into sections.
pub fn render(entries: &[Entry]) -> String {
    let mut out = String::new();
    let mut cur: Option<(String, Option<String>)> = None;
    for e in entries {
        let here = (e.section.clone(), e.subsection.clone());
        if cur.as_ref() != Some(&here) {
            match &e.subsection {
                Some(sub) => out.push_str(&alloc::format!("[{} \"{}\"]\n", e.section, sub)),
                None => out.push_str(&alloc::format!("[{}]\n", e.section)),
            }
            cur = Some(here);
        }
        out.push_str(&alloc::format!("\t{} = {}\n", e.key, e.value));
    }
    out
}

/// Split a dotted name into `(section, subsection, key)`.
///
/// The **middle** is the subsection, and it may itself contain dots
/// (`remote.my.host.url`), so the split is first-and-last, not on every dot.
pub fn split_name(name: &str) -> Option<(String, Option<String>, String)> {
    let (section, rest) = name.split_once('.')?;
    match rest.rsplit_once('.') {
        Some((sub, key)) => Some((
            section.to_ascii_lowercase(),
            Some(sub.to_string()),
            key.to_ascii_lowercase(),
        )),
        None => Some((section.to_ascii_lowercase(), None, rest.to_ascii_lowercase())),
    }
}

/// Look a dotted name up.
pub fn get<'a>(entries: &'a [Entry], name: &str) -> Option<&'a str> {
    let (s, sub, k) = split_name(name)?;
    entries
        .iter()
        .find(|e| e.section == s && e.subsection == sub && e.key == k)
        .map(|e| e.value.as_str())
}

/// Set a dotted name, replacing an existing value or appending after the last
/// entry of the same section so the file stays grouped.
pub fn set(entries: &mut Vec<Entry>, name: &str, value: &str) -> bool {
    let Some((s, sub, k)) = split_name(name) else {
        return false;
    };
    if let Some(e) = entries
        .iter_mut()
        .find(|e| e.section == s && e.subsection == sub && e.key == k)
    {
        e.value = value.to_string();
        return true;
    }
    let entry = Entry {
        section: s.clone(),
        subsection: sub.clone(),
        key: k,
        value: value.to_string(),
    };
    // Insert after the last entry of this section so `render` does not have to
    // emit the header twice.
    match entries
        .iter()
        .rposition(|e| e.section == s && e.subsection == sub)
    {
        Some(i) => entries.insert(i + 1, entry),
        None => entries.push(entry),
    }
    true
}

/// Remove a dotted name; returns how many entries went.
pub fn unset(entries: &mut Vec<Entry>, name: &str) -> usize {
    let Some((s, sub, k)) = split_name(name) else {
        return 0;
    };
    let before = entries.len();
    entries.retain(|e| !(e.section == s && e.subsection == sub && e.key == k));
    before - entries.len()
}

/// Remove a whole `[section "sub"]`; returns how many entries went.
pub fn remove_section(entries: &mut Vec<Entry>, section: &str, subsection: &str) -> usize {
    let s = section.to_ascii_lowercase();
    let before = entries.len();
    entries.retain(|e| !(e.section == s && e.subsection.as_deref() == Some(subsection)));
    before - entries.len()
}

/// Every subsection name under `section` (the remotes, the branches…).
pub fn subsections(entries: &[Entry], section: &str) -> Vec<String> {
    let s = section.to_ascii_lowercase();
    let mut out: Vec<String> = Vec::new();
    for e in entries {
        if e.section == s {
            if let Some(sub) = &e.subsection {
                if !out.iter().any(|x| x == sub) {
                    out.push(sub.clone());
                }
            }
        }
    }
    out
}
