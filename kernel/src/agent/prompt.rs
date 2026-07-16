//! **Prompt assembly helpers** — structured sections for the shell
//! agent system prompt, post-compact short prompt, skill envelopes, tool-result
//! bounding, and system-reminder wrappers.
//!
//! All pure / no_std: unit-tested without hardware. Side-effecting inject
//! (MEMORY.md, live SOUL) stays in `shell::agent_system_prompt`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Hard cap on model-facing tool result text (bytes). Full body spills to the
/// session store when larger — heap-hostile multi-MB dumps kill the allocator.
pub const TOOL_RESULT_MAX_BYTES: usize = 12 * 1024;

/// Compact system prompt used after `/compact` (compact system prompt).
/// Callers still re-inject MEMORY / skill L0 / tools via the normal builders.
pub const COMPACT_SYSTEM_CORE: &str =
    "You are Chitti, an agentic OS shell agent. Complete the user's request. \
     Prefer tools for machine state; never invent tool-readable data. \
     When done, answer in clear prose (markdown OK for structure).";

/// Structured summary request for interactive `/compact` (structured full-replace
/// sections, shortened for small on-device models).
pub fn compaction_user_prompt() -> &'static str {
    "Summarize this conversation for continuation. Use these section headings exactly, \
     keep each section tight (bullet lists preferred), and stay under ~500 words total:\n\
     1. Primary Request and Intent\n\
     2. Key Technical Concepts\n\
     3. Tool Usage (names + paths only, no full dumps)\n\
     4. Files and Artifacts\n\
     5. Errors and Fixes\n\
     6. Open Tasks / Todos\n\
     7. All User Messages (verbatim or high-fidelity short quotes)\n\
     Reply with only the seven sections — no preamble."
}

/// Operating-rules + bordered tagged sections appended after SOUL/memory.
pub fn operating_rules_block() -> &'static str {
    "You are Chitti, an agentic OS shell agent on bare metal. For greetings and small \
     talk, just reply in prose — do NOT call a tool. Call a tool only when the task needs \
     machine state or an action, and never invent data a tool can read (current time, \
     network status, files, disks). Use read/write/edit/glob/grep for files, memory_* for \
     durable notes (prefer the exact keys listed under Stored facts; if unsure call \
     memory_list or memory_search before guessing a key), notes_list/notes_get/notes_set for \
     markdown notes, download to fetch HTTP(S) files into /downloads/, skill to load a \
     procedure, todo_write for multi-step work, spawn_subagent to delegate, use_tool for \
     deferred/MCP tools found via search_tools.\n\
     \n\
     <action_safety>\n\
     Weigh each action by how easily it can be undone. Local reads and small edits are fine. \
     Before hard-to-reverse, shared-state, or destructive actions (delete, overwrite large \
     trees, network posts, install), the OS will ask the human — do not try to bypass that. \
     Investigate unexpected state before deleting or overwriting.\n\
     </action_safety>\n\
     \n\
     <tool_calling>\n\
     - Prefer specialized tools over inventing shell-like sequences.\n\
     - You may emit MULTIPLE <tool_call> blocks in one turn for independent reads; \
     write/edit/delete tools should be sequential when they touch the same path.\n\
     - Never use a tool only to \"talk\" to the user — put communication in your final prose.\n\
     - After tools run you get <tool_response>…</tool_response>; then answer or call more tools.\n\
     </tool_calling>\n\
     \n\
     <formatting>\n\
     Final answers may use GitHub-flavored markdown: lists, **bold**, `code`, fenced blocks, \
     and short tables for enumerable facts. Keep answers proportional to task complexity. \
     Do not wrap tool calls in markdown fences.\n\
     </formatting>\n"
}

/// Sub-agent (worker) persona core — parallel tools + no further delegation.
pub fn subagent_rules_block() -> &'static str {
    "You are an isolated Chitti sub-agent completing one delegated task. Use tools to \
     gather facts; never repeat a tool call you already ran, and never delegate further. \
     You may emit multiple read-only <tool_call> blocks in one turn. When you have the \
     facts, reply in plain prose (markdown OK) with a concise factual report of EXACTLY \
     what the tool output showed — never invent details.\n\
     <tool_calling>\n\
     Parallelize independent reads. Prefer specialized tools. Never use tools to chatter.\n\
     </tool_calling>\n"
}

/// Cheap runtime user_info block (arch, agent id, mode label).
pub fn user_info_block(arch: &str, agent_id: u64, mode: &str, model: &str) -> String {
    format!(
        "<user_info>\nOS: ChittiOS\nArch: {arch}\nActive agent id: {agent_id}\n\
         Approval mode: {mode}\nModel: {model}\n</user_info>\n"
    )
}

/// L0 skill listing as XML `<agent_skill>` rows (budgeted).
pub fn format_skill_l0_listing(skills: &[(String, String)], max: usize) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "## Installed skills (L0 — invoke with skill {\"name\":\"…\"} or /skill <name>)\n",
    );
    for (name, desc) in skills.iter().take(max) {
        s.push_str("<agent_skill name=\"");
        s.push_str(name);
        s.push_str("\" description=\"");
        // Keep description one line for prompt budget.
        for c in desc.chars().take(160) {
            if c == '"' || c == '\n' || c == '\r' {
                s.push(' ');
            } else {
                s.push(c);
            }
        }
        s.push_str("\"/>\n");
    }
    s.push('\n');
    s
}

/// Wrap an L1 skill body in the bordered `<skill>` envelope for tool results.
pub fn skill_result_envelope(name: &str, path: &str, body: &str) -> String {
    format!("<skill name=\"{name}\" path=\"{path}\">\n{body}\n</skill>")
}

/// Wrap automated host text so models (and taint) can treat it as non-user.
pub fn system_reminder(body: &str) -> String {
    format!("<system-reminder>\n{body}\n</system-reminder>")
}

/// Slash/skill expansion preamble for zero-RTT skill inject into a user turn.
pub fn skill_information_envelope(name: &str, path: &str, body: &str, args: &str) -> String {
    let args_attr = if args.is_empty() {
        String::new()
    } else {
        format!(" args=\"{args}\"")
    };
    format!(
        "<skill_information>\n\
         <skills_referenced>\n\
         <skill name=\"{name}\" path=\"{path}\"/>\n\
         </skills_referenced>\n\
         <skill name=\"{name}\"{args_attr}>\n\
         {body}\n\
         </skill>\n\
         </skill_information>\n"
    )
}

/// Truncate model-facing tool output; optionally return a spill path hint.
/// Returns `(text_for_model, spilled)` where spilled means the full body was
/// written by the caller to `spill_path` (this fn only formats the hint).
pub fn bound_tool_result(result: &str, spill_path: Option<&str>) -> String {
    if result.len() <= TOOL_RESULT_MAX_BYTES {
        return result.to_string();
    }
    // Prefer a char-safe cut near the limit.
    let mut end = TOOL_RESULT_MAX_BYTES.min(result.len());
    while end > 0 && !result.is_char_boundary(end) {
        end -= 1;
    }
    let head = &result[..end];
    match spill_path {
        Some(p) => format!(
            "{head}\n… [truncated {} bytes → full output at {p}]",
            result.len()
        ),
        None => format!(
            "{head}\n… [truncated {} bytes; re-run with a narrower query]",
            result.len()
        ),
    }
}

/// Score a tool against a keyword query (term overlap on name + description).
/// Higher is better; 0 means no match.
pub fn tool_search_score(name: &str, description: &str, query: &str) -> u32 {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return 0;
    }
    let name_l = name.to_ascii_lowercase();
    let desc_l = description.to_ascii_lowercase();
    let mut score = 0u32;
    for term in q.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-') {
        if term.is_empty() {
            continue;
        }
        if name_l == term {
            score = score.saturating_add(100);
        } else if name_l.contains(term) {
            score = score.saturating_add(40);
        }
        if desc_l.contains(term) {
            score = score.saturating_add(10 + term.len() as u32);
        }
    }
    score
}

/// Deterministic short condensation of old messages for agent-layer auto-compact
/// (no model). Preserves role + short content for resume quality over 24-char
/// snippets.
pub fn deterministic_compact_summary(parts: &[(/*role*/ &str, /*content*/ &str)]) -> String {
    let mut out = String::from("[compacted]\n");
    for (role, content) in parts {
        let snip: String = content.chars().take(120).collect();
        let one_line: String = snip.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&one_line);
        out.push('\n');
    }
    out
}

/// Format multi-tool results for a single `<tool_response>` prefill.
pub fn format_multi_tool_response(results: &[(String, String)]) -> String {
    if results.len() == 1 {
        return format!("<tool_response>\n{}\n</tool_response>", results[0].1);
    }
    let mut s = String::from("<tool_response>\n");
    for (name, body) in results {
        s.push_str("[tool: ");
        s.push_str(name);
        s.push_str("]\n");
        s.push_str(body);
        s.push_str("\n\n");
    }
    s.push_str("</tool_response>");
    s
}

/// Path under the session store for a spilled tool result.
pub fn tool_spill_path(session_id: u64, call_id: u64) -> String {
    format!("/sessions/{session_id}/tool_out/{call_id}")
}

/// Path for the plan-mode plan file (session plan file).
pub fn plan_file_path(session_id: u64) -> String {
    format!("/sessions/{session_id}/plan.md")
}

/// Session summary index JSON body (minimal, human-readable fields).
pub fn session_summary_json(
    id: u64,
    title: &str,
    messages: usize,
    model: &str,
    parent: Option<u64>,
    updated_ticks: u64,
) -> String {
    let parent_field = match parent {
        Some(p) => format!(",\"parent\":{p}"),
        None => String::new(),
    };
    // Escape title for JSON string.
    let mut esc = String::new();
    for c in title.chars().take(80) {
        match c {
            '"' => esc.push_str("\\\""),
            '\\' => esc.push_str("\\\\"),
            '\n' | '\r' => esc.push(' '),
            c => esc.push(c),
        }
    }
    format!(
        "{{\"id\":{id},\"title\":\"{esc}\",\"messages\":{messages},\"model\":\"{model}\",\
         \"updated_ticks\":{updated_ticks}{parent_field}}}\n"
    )
}

/// First user-message line as a session title fallback.
pub fn title_from_messages(user_texts: &[&str]) -> String {
    for t in user_texts {
        let line = t.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
        if !line.is_empty() && !line.starts_with('<') {
            return line.chars().take(60).collect();
        }
    }
    String::from("(untitled)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn bound_tool_result_small_unchanged() {
        assert_eq!(bound_tool_result("hi", None), "hi");
    }

    #[test_case]
    fn bound_tool_result_truncates_with_spill() {
        let big = "x".repeat(TOOL_RESULT_MAX_BYTES + 50);
        let out = bound_tool_result(&big, Some("/sessions/1/tool_out/2"));
        assert!(out.contains("truncated"));
        assert!(out.contains("/sessions/1/tool_out/2"));
        // Suffix can make the string slightly longer than TOOL_RESULT_MAX, but
        // it must not contain the full original body.
        assert!(out.len() < big.len() + 128);
        assert!(!out.ends_with("xxxxx"), "must not keep the full tail");
        assert!(out.len() < TOOL_RESULT_MAX_BYTES + 200);
    }

    #[test_case]
    fn tool_search_score_prefers_name_hit() {
        let a = tool_search_score("memory_get", "read a durable fact", "memory");
        let b = tool_search_score("read", "read a file from the store", "memory");
        assert!(a > b, "name match should beat description-only: {a} vs {b}");
        assert_eq!(tool_search_score("read", "file", ""), 0);
    }

    #[test_case]
    fn skill_envelopes_roundtrip_shape() {
        let e = skill_result_envelope("commit", "/agent/1/skills/commit", "do it");
        assert!(e.contains("<skill name=\"commit\""));
        assert!(e.contains("do it"));
        let i = skill_information_envelope("commit", "/p", "body", "fix typo");
        assert!(i.contains("<skill_information>"));
        assert!(i.contains("args=\"fix typo\""));
    }

    #[test_case]
    fn format_skill_l0_uses_agent_skill_tags() {
        let skills = alloc::vec![("note".into(), "summarize notes".into())];
        let s = format_skill_l0_listing(&skills, 12);
        assert!(s.contains("<agent_skill name=\"note\""));
        assert!(s.contains("summarize notes"));
    }

    #[test_case]
    fn multi_tool_response_labels_tools() {
        let r = format_multi_tool_response(&[
            ("read".into(), "a".into()),
            ("grep".into(), "b".into()),
        ]);
        assert!(r.contains("[tool: read]"));
        assert!(r.contains("[tool: grep]"));
    }

    #[test_case]
    fn operating_rules_allow_markdown_and_multi_tool() {
        let r = operating_rules_block();
        assert!(r.contains("<action_safety>"));
        assert!(r.contains("<formatting>"));
        assert!(r.contains("MULTIPLE <tool_call>"));
        assert!(!r.contains("never markdown"));
    }

    #[test_case]
    fn deterministic_compact_summary_includes_roles() {
        let parts = [("user", "hello world"), ("assistant", "hi there")];
        let s = deterministic_compact_summary(&parts);
        assert!(s.contains("user: hello"));
        assert!(s.contains("assistant: hi"));
    }

    #[test_case]
    fn session_summary_json_escapes_title() {
        let j = session_summary_json(3, "say \"hi\"", 2, "local", Some(1), 99);
        assert!(j.contains("\\\"hi\\\""));
        assert!(j.contains("\"parent\":1"));
    }
}
