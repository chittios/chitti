//! tooljson
//!
//! Extracted from the former 16k-line `shell/mod.rs` monolith (see the
//! shell README / CLAUDE.md conventions). Every item below was moved
//! verbatim; `use super::*` makes the parent module's statics and items
//! visible here, and the parent re-exports this module's items with
//! `pub(crate) use tooljson::*`, so intra-shell callers are unchanged.

use super::*;

/// Extract a string field's value from a small JSON object. Tolerant of
/// whitespace; handles `\"`/`\n`/`\t` escapes. Returns `None` if the key is
/// absent or its value is not a string.
pub(super) fn json_str(obj: &str, key: &str) -> Option<String> {
    let pat = alloc::format!("\"{}\"", key);
    let i = obj.find(&pat)?;
    let rest = &obj[i + pat.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut esc = false;
    for c in rest.chars() {
        if esc {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                other => other,
            });
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// Flatten a tool call's `"arguments"` into the single argument line our
/// dispatchers take: a bare-string arguments value is used as-is; an object is
/// probed for the conventional keys the builtin schemas use.
///
/// Memory tools encode multi-field args as `key\\x1fvalue` (unit separator) so
/// both key and value survive the flatten step into `execute_chat_tool`.
pub(super) fn json_args(body: &str) -> String {
    if let Some(i) = body.find("\"arguments\"") {
        let rest = &body[i..];
        // `"arguments": "..."` (string form).
        if let Some(v) = json_str(rest, "arguments") {
            return v;
        }
        // memory_add / memory_get: preserve key (+ value) as a structured line.
        if let Some(k) = json_str(rest, "key") {
            if let Some(v) = json_str(rest, "value") {
                let mut s = k;
                s.push('\u{1f}');
                s.push_str(&v);
                return s;
            }
            return k;
        }
        // `"arguments": {...}` (object form): first conventional key present.
        for key in ["args", "task", "path", "host", "query", "text", "intent", "name"] {
            if let Some(v) = json_str(rest, key) {
                if !v.is_empty() {
                    return v;
                }
            }
        }
    }
    String::new()
}

/// Extract the `arguments` value from a tool-call body as a JSON object string
/// suitable for the Synapse Router. Object form is preserved; a bare string
/// form becomes `{"args":"…"}` so shell tools still work.
pub(super) fn extract_arguments_json(body: &str) -> String {
    let Some(i) = body.find("\"arguments\"") else {
        return String::from("{}");
    };
    let after_key = &body[i + "\"arguments\"".len()..];
    let Some(colon) = after_key.find(':') else {
        return String::from("{}");
    };
    let rest = after_key[colon + 1..].trim_start();
    if rest.starts_with('"') {
        // `"arguments": "flattened line"`
        if let Some(v) = json_str(&body[i..], "arguments") {
            return wrap_args_json(&v);
        }
        return String::from("{}");
    }
    if rest.starts_with('{') {
        return extract_balanced_json_object(rest).unwrap_or_else(|| String::from("{}"));
    }
    String::from("{}")
}

/// Slice a balanced `{…}` JSON object from the start of `s` (string-aware).
pub(super) fn extract_balanced_json_object(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

pub(super) fn wrap_args_json(args_line: &str) -> String {
    let mut out = String::from("{\"args\":\"");
    for c in args_line.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push_str("\"}");
    out
}

/// Normalize a chat-layer arg payload into Router JSON. Accepts a full object,
/// a flattened shell line, or memory `key\\x1fvalue`.
pub(super) fn normalize_tool_args_json(name: &str, args: &str) -> String {
    let t = args.trim();
    if t.is_empty() {
        return String::from("{}");
    }
    if t.starts_with('{') {
        return t.to_string();
    }
    // Memory: structured unit-separator form from the older flattener.
    if matches!(name, "memory_add" | "memory_get" | "memory_list" | "remember" | "recall") {
        if name == "memory_list" {
            return String::from("{}");
        }
        if let Some((k, v)) = t.split_once('\u{1f}') {
            let mut o = String::from("{\"key\":\"");
            json_escape_into(&mut o, k);
            o.push_str("\",\"value\":\"");
            json_escape_into(&mut o, v);
            o.push_str("\"}");
            return o;
        }
        let mut o = String::from("{\"key\":\"");
        json_escape_into(&mut o, t);
        o.push_str("\"}");
        return o;
    }
    // Single-arg synapse conveniences (small models often flatten).
    match name {
        "read" | "delete" => {
            let mut o = String::from("{\"path\":\"");
            json_escape_into(&mut o, t);
            o.push_str("\"}");
            o
        }
        "search" | "grep" | "memory_search" | "search_tools" => {
            let mut o = String::from("{\"query\":\"");
            json_escape_into(&mut o, t);
            o.push_str("\"}");
            o
        }
        "skill" | "load_skill" => {
            // name [asset]
            if let Some((n, a)) = t.split_once(char::is_whitespace) {
                let mut o = String::from("{\"name\":\"");
                json_escape_into(&mut o, n.trim());
                o.push_str("\",\"asset\":\"");
                json_escape_into(&mut o, a.trim());
                o.push_str("\"}");
                return o;
            }
            let mut o = String::from("{\"name\":\"");
            json_escape_into(&mut o, t);
            o.push_str("\"}");
            o
        }
        "glob" => {
            let mut o = String::from("{\"pattern\":\"");
            json_escape_into(&mut o, t);
            o.push_str("\"}");
            o
        }
        "console" => {
            let mut o = String::from("{\"text\":\"");
            json_escape_into(&mut o, t);
            o.push_str("\"}");
            o
        }
        "list" | "todo_write" => {
            if name == "todo_write" && !t.is_empty() {
                // Accept a bare todos array or object as the full args payload.
                if t.starts_with('{') || t.starts_with('[') {
                    if t.starts_with('[') {
                        return alloc::format!(r#"{{"todos":{t}}}"#);
                    }
                    return t.to_string();
                }
            }
            String::from("{}")
        }
        _ => wrap_args_json(t),
    }
}

pub(super) fn json_escape_into(out: &mut String, v: &str) {
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
}

/// Tag names a model reply may wrap a tool call in, in any decoration
/// (see [`undecorate_tool_tags`]).
const TOOL_TAGS: &[&str] = &[
    "tool_call",
    "tool_calls",
    "function_calls",
    "invoke",
    "parameter",
];

/// Longest decoration accepted between `<` (or `</`) and a tag name, in chars.
/// `｜DSML｜` is 6; the headroom is for namespace prefixes and mojibake of the
/// same token (a 3-byte `｜` that arrived re-encoded is several chars).
const MAX_TAG_DECOR: usize = 24;

/// True when `c` may appear in the decoration between `<` and a tag name.
/// Whitespace and `<`/`>` end the candidate: prose like `a < b, tool_call…`
/// must not be rewritten into a tag.
fn is_tag_decor(c: char) -> bool {
    !c.is_whitespace() && c != '<' && c != '>' && c != '"'
}

/// Strip vendor decoration out of tool-call tags so one parser serves every
/// model: DeepSeek-V4 emits DSML (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke
/// name="x">`), V3.2 emitted `<｜DSML｜function_calls>`, and others use a
/// namespace prefix — all of which mean the tag our prompt asked for.
///
/// Keyed on the tag **name**, never on the decorating bytes: the same token
/// reaches us differently depending on how the provider encoded it (and a
/// terminal renders every non-ASCII byte identically, so the decoration is not
/// even diagnosable from a transcript). Only a `<`/`</` followed by decoration
/// and then one of [`TOOL_TAGS`] at a word boundary is rewritten, so prose is
/// untouched — and an undecorated reply allocates nothing.
pub(super) fn undecorate_tool_tags(text: &str) -> alloc::borrow::Cow<'_, str> {
    use alloc::borrow::Cow;
    if !text.contains('<') {
        return Cow::Borrowed(text);
    }
    let mut out = String::new();
    // Everything before `copied` is already in `out` (once anything was
    // rewritten); `i` is the scan cursor.
    let mut copied = 0usize;
    let mut rewrote = false;
    let mut i = 0usize;
    while let Some(rel) = text[i..].find('<') {
        let lt = i + rel;
        let after_lt = lt + 1;
        let (name_from, closing) = match text[after_lt..].starts_with('/') {
            true => (after_lt + 1, true),
            false => (after_lt, false),
        };
        // Walk forward over decoration chars, testing for a tag name at each
        // step. `at == name_from` on the first step is the undecorated case.
        let mut at = name_from;
        let mut skipped = 0usize;
        let hit = loop {
            // The boundary test belongs *inside* the search, not after it:
            // `tool_call` is a prefix of `tool_calls`, so picking the first
            // name that matches and then rejecting it for having no boundary
            // abandons the position while the longer name was sitting right
            // there — which left every `</…tool_calls>` undecorated.
            let matched = TOOL_TAGS.iter().find(|t| {
                text[at..].starts_with(**t)
                    && text[at + t.len()..]
                        .chars()
                        .next()
                        .map(|n| n == '>' || n == '/' || n.is_whitespace())
                        .unwrap_or(false)
            });
            if let Some(tag) = matched {
                break Some((at, at + tag.len()));
            }
            let Some(c) = text[at..].chars().next() else { break None };
            if skipped >= MAX_TAG_DECOR || !is_tag_decor(c) {
                break None;
            }
            at += c.len_utf8();
            skipped += 1;
        };
        match hit {
            Some((at, end)) if at > name_from => {
                out.push_str(&text[copied..lt]);
                out.push('<');
                if closing {
                    out.push('/');
                }
                out.push_str(&text[at..end]);
                copied = end;
                rewrote = true;
                i = end;
            }
            // An undecorated tag: leave it exactly as it is.
            Some((_, end)) => i = end,
            None => i = after_lt,
        }
    }
    if rewrote {
        out.push_str(&text[copied..]);
        Cow::Owned(out)
    } else {
        Cow::Borrowed(text)
    }
}

/// Read a double-quoted attribute value out of a tag's attribute span.
fn tag_attr(attrs: &str, key: &str) -> Option<String> {
    let mut rest = attrs;
    loop {
        let i = rest.find(key)?;
        let after = &rest[i + key.len()..];
        let trimmed = after.trim_start();
        // `key` must be a whole attribute name (`name=`), not a suffix of one.
        let name_start_ok = rest[..i]
            .chars()
            .next_back()
            .map(|c| c.is_whitespace() || c == '<')
            .unwrap_or(true);
        if name_start_ok {
            if let Some(v) = trimmed.strip_prefix('=') {
                let v = v.trim_start();
                if let Some(v) = v.strip_prefix('"') {
                    let end = v.find('"')?;
                    return Some(String::from(&v[..end]));
                }
            }
        }
        rest = after;
    }
}

/// Parse DSML `invoke`/`parameter` tool calls (DeepSeek-V4's native format,
/// after [`undecorate_tool_tags`]) into `(name, args_json)` with the byte
/// offset each call started at, so a reply mixing formats keeps document order.
///
/// A parameter is a JSON **string** unless the tag says `string="false"`, in
/// which case its body is already JSON (number/bool/array/object) and is
/// inserted verbatim — that distinction is the whole point of the format, and
/// quoting a `false` would turn it into a true-ish string.
fn parse_dsml_invokes(text: &str) -> alloc::vec::Vec<(usize, String, String)> {
    let mut out = alloc::vec::Vec::new();
    let mut i = 0usize;
    while let Some(rel) = text[i..].find("<invoke") {
        let open = i + rel;
        let Some(gt_rel) = text[open..].find('>') else { break };
        let gt = open + gt_rel;
        let attrs = &text[open + "<invoke".len()..gt];
        let body_start = gt + 1;
        let close = text[body_start..]
            .find("</invoke")
            .map(|c| body_start + c)
            .unwrap_or(text.len());
        let body = &text[body_start..close];
        let name = tag_attr(attrs, "name").unwrap_or_default();
        let name = name.trim().trim_start_matches('/').to_string();
        if name.is_empty() {
            crate::ktrace::log_fmt(format_args!(
                "chat.toolcall: <invoke> without a name attribute: {:.120}",
                attrs.trim()
            ));
        } else {
            out.push((open, name, dsml_params_json(body)));
        }
        i = close + 1;
        if i >= text.len() {
            break;
        }
    }
    out
}

/// Collect a DSML invoke body's `<parameter name="k" [string="false"]>v</parameter>`
/// children into one JSON object.
fn dsml_params_json(body: &str) -> String {
    let mut obj = String::from("{");
    let mut n = 0usize;
    let mut i = 0usize;
    while let Some(rel) = body[i..].find("<parameter") {
        let open = i + rel;
        let Some(gt_rel) = body[open..].find('>') else { break };
        let gt = open + gt_rel;
        let attrs = &body[open + "<parameter".len()..gt];
        let vstart = gt + 1;
        let close = body[vstart..]
            .find("</parameter")
            .map(|c| vstart + c)
            .unwrap_or(body.len());
        let raw = &body[vstart..close];
        // A value written on its own lines carries the newlines that put it
        // there; they are layout, not content.
        let val = raw.strip_prefix('\n').unwrap_or(raw);
        let val = val.strip_suffix('\n').unwrap_or(val);
        if let Some(key) = tag_attr(attrs, "name") {
            if n > 0 {
                obj.push(',');
            }
            obj.push('"');
            json_escape_into(&mut obj, &key);
            obj.push_str("\":");
            let is_str = tag_attr(attrs, "string").map(|s| s != "false").unwrap_or(true);
            if is_str {
                obj.push('"');
                json_escape_into(&mut obj, val);
                obj.push('"');
            } else {
                let t = val.trim();
                if t.is_empty() {
                    obj.push_str("null");
                } else {
                    obj.push_str(t);
                }
            }
            n += 1;
        }
        i = close + 1;
        if i >= body.len() {
            break;
        }
    }
    obj.push('}');
    obj
}

/// Detect **all** tool calls in a model reply (multi-tool turn). Primary:
/// Qwen3.5 `<tool_call>{…}</tool_call>` blocks (each block's own JSON only —
/// never merge name from block 1 with arguments from block 2). Also accepted:
/// DSML `<invoke name="x"><parameter …>` blocks (DeepSeek-V4's native tool
/// format), which a hosted model emits regardless of what our prompt asked
/// for. Fallback: a single legacy `TOOL: /cmd args` line when no XML blocks
/// are present.
pub(crate) fn parse_tool_calls(
    text: &str,
) -> alloc::vec::Vec<(alloc::string::String, alloc::string::String)> {
    let undecorated = undecorate_tool_tags(text);
    parse_tool_calls_bare(undecorated.as_ref())
}

/// Opening container tags a JSON tool call may be wrapped in, longest first so
/// `tool_calls` is never mistaken for `tool_call` plus a stray `s`.
const CALL_CONTAINERS: &[&str] = &["<function_calls>", "<tool_calls>", "<tool_call>"];

/// Find the next JSON-tool-call container: its start offset and the offset just
/// past its `>`.
fn next_call_container(hay: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for tag in CALL_CONTAINERS {
        if let Some(i) = hay.find(*tag) {
            let cand = (i, i + tag.len());
            if best.map(|(b, _)| i < b).unwrap_or(true) {
                best = Some(cand);
            }
        }
    }
    best
}

/// Where this container's contents end: the first closing tag or the next
/// opening one, whichever comes first.
fn call_container_end(after: &str) -> usize {
    let mut end = after.len();
    for close in ["</function_calls>", "</tool_calls>", "</tool_call>"] {
        if let Some(i) = after.find(close) {
            end = end.min(i);
        }
    }
    if let Some((i, _)) = next_call_container(after) {
        end = end.min(i);
    }
    end
}

/// [`parse_tool_calls`] on already-undecorated text.
fn parse_tool_calls_bare(
    text: &str,
) -> alloc::vec::Vec<(alloc::string::String, alloc::string::String)> {
    use alloc::string::ToString;
    // (offset, name, args) so JSON and DSML blocks interleave in reply order —
    // a model can emit both in one turn, and running them out of order would
    // silently reorder side effects.
    let mut found: alloc::vec::Vec<(usize, String, String)> = alloc::vec::Vec::new();
    let mut out = alloc::vec::Vec::new();
    let mut cursor = 0usize;
    // The tag is treated as a **container**, not as a fixed `<tool_call>…
    // </tool_call>` pair, because a real reply violates that pairing in every
    // direction: DeepSeek-V4 opens `<｜DSML｜tool_calls>` (plural) and closes
    // `</｜DSML｜tool_call>` (singular), and puts one *or several* JSON objects
    // inside. Matching the literal singular tag found none of it, so a
    // well-formed call was printed to the user as prose.
    while let Some((start, body_at)) = next_call_container(&text[cursor..]) {
        let start = cursor + start;
        let body_at = cursor + body_at;
        let after = &text[body_at..];
        let inner_len = call_container_end(after);
        let block = &after[..inner_len];
        // Every top-level JSON object in the block is its own call: a container
        // may hold parallel calls. Nested objects are consumed by the balanced
        // scan, so `arguments` is never mistaken for a second call.
        let mut n_in_block = 0usize;
        let mut b = 0usize;
        while let Some(rel) = block[b..].find('{') {
            let at = b + rel;
            let Some(obj) = extract_balanced_json_object(&block[at..]) else {
                break;
            };
            let advance = obj.len().max(1);
            if let Some(name) = json_str(&obj, "name") {
                let name = name.trim().trim_start_matches('/').to_string();
                if !name.is_empty() {
                    found.push((start + at, name, extract_arguments_json(&obj)));
                    n_in_block += 1;
                }
            }
            b = at + advance;
        }
        if n_in_block == 0 && !block.contains("<invoke") {
            // The model *tried* and we could not read a tool name out of it.
            // Without this trace that is indistinguishable from the model never
            // calling a tool at all — which is exactly the wrong thing to be
            // guessing about on a heavily-quantized model, where malformed JSON
            // is the likely failure and "it ignores its tools" is the likely
            // wrong conclusion. Truncated: a block can be long.
            //
            // A DSML reply nests `<invoke>` in the block instead of JSON — that
            // is not unparsable, it is the other format, and the pass below
            // reads it. Tracing it here would cry wolf on every turn.
            crate::ktrace::log_fmt(format_args!(
                "chat.toolcall: unparsable tool-call block ({} bytes), no usable \"name\": {:.160}",
                block.len(),
                block.trim()
            ));
        }
        // Always advance past the container tag itself, so an empty or malformed
        // block cannot spin here.
        cursor = body_at + inner_len.max(0);
        if cursor <= start {
            break;
        }
    }
    // DSML `<invoke name="x"><parameter …>` blocks (DeepSeek-V4).
    found.extend(parse_dsml_invokes(text));
    if !found.is_empty() {
        found.sort_by_key(|(pos, _, _)| *pos);
        out.extend(found.into_iter().map(|(_, name, args)| (name, args)));
        return out;
    }
    // Legacy fallback → wrap the free-form line as shell `args`.
    for line in text.lines() {
        let l = line.trim();
        let rest = ["TOOL:", "TOOLS:", "Tool:", "tool:", "TOOL "]
            .iter()
            .find_map(|p| l.strip_prefix(p))
            .map(|r| r.trim().trim_start_matches('/').trim());
        if let Some(rest) = rest {
            if rest.is_empty() {
                continue;
            }
            let mut parts = rest.splitn(2, char::is_whitespace);
            let cmd = parts.next().unwrap_or("").to_string();
            let args = parts.next().unwrap_or("").trim().to_string();
            if !cmd.is_empty() {
                let json = normalize_tool_args_json(&cmd, &args);
                out.push((cmd, json));
                break;
            }
        }
    }
    out
}

/// Remove tool-call markup from a reply that is about to be **shown to a human**.
///
/// A model that emits a malformed or empty call (DeepSeek-V4 has produced a bare
/// `<｜DSML｜tool_calls>` with nothing in it) parses to no calls, so the reply
/// falls through to the "this is the answer" path and the raw tags land in the
/// chat — as mojibake, since the console has no glyph for `｜`. Stripping is for
/// presentation only: the raw text still goes into history and the audit trail,
/// and a block that *did* parse never reaches here.
pub(crate) fn strip_tool_markup(s: &str) -> alloc::string::String {
    use alloc::string::ToString;
    let un = undecorate_tool_tags(s);
    let mut out = alloc::string::String::new();
    let mut rest = un.as_ref();
    'outer: while !rest.is_empty() {
        let Some(lt) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        // Only drop a `<…>` whose name is one of ours; anything else is content.
        let head = &rest[lt..];
        let after_lt = head.strip_prefix("</").unwrap_or_else(|| &head[1..]);
        for tag in TOOL_TAGS {
            if after_lt.starts_with(*tag) {
                let boundary = after_lt[tag.len()..]
                    .chars()
                    .next()
                    .map(|c| c == '>' || c == '/' || c.is_whitespace())
                    .unwrap_or(false);
                if boundary {
                    if let Some(gt) = head.find('>') {
                        out.push_str(&rest[..lt]);
                        rest = &head[gt + 1..];
                        continue 'outer;
                    }
                }
            }
        }
        // Not one of ours: keep the `<` and carry on past it.
        out.push_str(&rest[..lt + 1]);
        rest = &head[1..];
    }
    out.trim().to_string()
}

/// First tool call only — compatibility wrapper (tests + oneshot paths).
pub(crate) fn parse_tool_call(text: &str) -> Option<(alloc::string::String, alloc::string::String)> {
    parse_tool_calls(text).into_iter().next()
}

/// A friendly verb + primary argument for an agent tool call, for the styled
/// chat header (`◆ Edit  src/api/checkout.ts`). Unknown tools title-case their
/// own name and show a compact arg summary.
pub(super) fn tool_header(cmd: &str, args: &str) -> (alloc::string::String, alloc::string::String) {
    use crate::session::todo::json_str;
    let pick = |keys: &[&str]| -> alloc::string::String {
        keys.iter().find_map(|k| json_str(args, k)).unwrap_or_default()
    };
    let (verb, arg): (&str, alloc::string::String) = match cmd {
        "read" | "cat" | "open" => ("Read", pick(&["path", "file", "args"])),
        "write" | "edit" => ("Edit", pick(&["path", "file"])),
        "list" | "ls" => ("List", {
            let a = pick(&["path", "dir", "args"]);
            if a.is_empty() { "/".into() } else { a }
        }),
        "glob" | "grep" | "search" => ("Search", pick(&["pattern", "query", "args"])),
        "search_tools" => ("Search tools", pick(&["query", "args"])),
        "http" => ("Fetch", pick(&["url", "args"])),
        "download" => ("Download", pick(&["url", "args"])),
        "mkdir" => ("Make dir", pick(&["path", "args"])),
        "touch" => ("Create", pick(&["path", "args"])),
        "rm" | "delete" => ("Delete", pick(&["path", "args"])),
        "cp" => ("Copy", pick(&["args"])),
        "mv" => ("Move", pick(&["args"])),
        "memory_add" => ("Remember", pick(&["key", "args"])),
        "memory_get" | "memory_search" | "memory_list" => ("Recall", pick(&["key", "query", "args"])),
        "skill" => ("Skill", pick(&["name", "args"])),
        "spawn_subagent" | "subagent" => ("Delegate", pick(&["task", "args"])),
        _ => return (cap_first(cmd), compact_args(args)),
    };
    (alloc::string::String::from(verb), arg)
}

/// Title-case a tool name for the chat header ("browse" → "Browse").
pub(super) fn cap_first(s: &str) -> alloc::string::String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<alloc::string::String>() + c.as_str(),
        None => alloc::string::String::new(),
    }
}

/// A compact one-line argument summary for a tool header; empty `{}` → "".
pub(super) fn compact_args(args: &str) -> alloc::string::String {
    let a = args.trim();
    if a.is_empty() || a == "{}" {
        return alloc::string::String::new();
    }
    let inner = a.trim_start_matches('{').trim_end_matches('}').trim();
    inner.chars().take(56).collect()
}

/// A truecolor SGR (`ESC[38;2;R;G;Bm`) for a theme palette key, so chat styling
/// follows the active theme instead of fixed ANSI colours. `def` is the fallback
/// when the key/theme is unavailable (the pane renders `38;2` truecolor).
pub(crate) fn theme_sgr(key: &str, def: (u8, u8, u8)) -> alloc::string::String {
    #[cfg(test)]
    {
        let _ = key;
        return alloc::format!("\x1b[38;2;{};{};{}m", def.0, def.1, def.2);
    }
    #[cfg(not(test))]
    {
        let cfg = crate::ui_config::current();
        let hex = cfg.theme.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()).unwrap_or_default();
        let (r, g, b) = crate::framebuffer::parse_hex(&hex, def);
        alloc::format!("\x1b[38;2;{};{};{}m", r, g, b)
    }
}

/// Drop a `<think>…</think>` reasoning block from a (remote) model reply for
/// display — the reasoning is summarized as a "Thought for Xs" line instead.
pub(crate) fn strip_think(s: &str) -> alloc::string::String {
    use alloc::string::ToString;
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].trim_start().to_string()
    } else if let Some(start) = s.find("<think>") {
        s[..start].trim().to_string() // unterminated: keep the prefix
    } else {
        s.to_string()
    }
}
