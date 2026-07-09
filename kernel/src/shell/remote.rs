//! **Remote model backend** — chat through a *hosted* model instead of the
//! embedded GGUF. Speaks the OpenAI-compatible `/v1/chat/completions` JSON
//! shape over [`crate::net::http`], which is what llama.cpp's server, Ollama,
//! vLLM, and LM Studio all serve — the self-hosted case the missing TLS
//! doesn't hurt (plain `http://` on the host/LAN).
//!
//! The remote chat runs the SAME agentic contract as the local one: the same
//! system prompt (SOUL.md + operating rules + CORE tools + `search_tools`),
//! the same `<tool_call>` convention parsed by `parse_tool_call`, the same
//! approval-gated `execute_chat_tool`. Only generation moves off-box; tool
//! *execution* stays on this machine, inside the capability system.
//!
//! Config persists at `/configs/core/model.json` (`/model` command), so an
//! installed system keeps its backend across reboots.

use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;

const MODEL_CFG_PATH: &str = "/configs/core/model.json";
/// A hosted model can legitimately think for a while; generous per-call cap.
const HTTP_TIMEOUT_MS: u64 = 120_000;

/// The persisted backend config.
#[derive(Clone)]
pub struct RemoteConfig {
    /// Base URL, e.g. `http://192.168.1.20:8080` (no trailing slash needed).
    pub url: String,
    /// Model name passed through to the server (Ollama/vLLM route on it;
    /// llama.cpp ignores it).
    pub model: String,
    /// Optional bearer token (`Authorization: Bearer …`).
    pub key: Option<String>,
}

/// Load the persisted config: `(remote_active, Option<RemoteConfig>)`.
pub fn load() -> (bool, Option<RemoteConfig>) {
    let Some(bytes) = crate::synapse::fs::read(MODEL_CFG_PATH) else {
        return (false, None);
    };
    let Some(j) = core::str::from_utf8(&bytes).ok().and_then(Json::parse) else {
        return (false, None);
    };
    let remote = j.get("mode").and_then(|v| v.as_str()) == Some("remote");
    let cfg = j.get("url").and_then(|v| v.as_str()).map(|url| RemoteConfig {
        url: url.trim_end_matches('/').to_string(),
        model: j.get("model").and_then(|v| v.as_str()).unwrap_or("default").to_string(),
        key: j.get("key").and_then(|v| v.as_str()).filter(|k| !k.is_empty()).map(|k| k.to_string()),
    });
    (remote && cfg.is_some(), cfg)
}

/// Persist the backend choice (+ config, when remote is configured).
pub fn save(remote_active: bool, cfg: Option<&RemoteConfig>) {
    let mut obj = vec![("mode".to_string(), Json::Str(if remote_active { "remote" } else { "local" }.to_string()))];
    if let Some(c) = cfg {
        obj.push(("url".to_string(), Json::Str(c.url.clone())));
        obj.push(("model".to_string(), Json::Str(c.model.clone())));
        obj.push(("key".to_string(), Json::Str(c.key.clone().unwrap_or_default())));
    }
    crate::synapse::fs::write(MODEL_CFG_PATH, Json::Obj(obj).to_pretty().as_bytes());
}

/// JSON-escape into a `Json::Str` via the existing writer.
fn jstr(s: &str) -> Json {
    Json::Str(s.to_string())
}

/// Build the OpenAI `/v1/chat/completions` request body for `model` +
/// `messages` (role/content pairs). Pure — unit-tested so the wire shape can't
/// silently drift.
pub fn build_chat_request(model: &str, messages: &[(String, String)]) -> String {
    let msgs: Vec<Json> = messages
        .iter()
        .map(|(role, content)| Json::Obj(vec![("role".to_string(), jstr(role)), ("content".to_string(), jstr(content))]))
        .collect();
    Json::Obj(vec![
        ("model".to_string(), jstr(model)),
        ("messages".to_string(), Json::Arr(msgs)),
        ("stream".to_string(), Json::Bool(false)),
    ])
    .to_pretty()
}

/// Extract `choices[0].message.content` from an OpenAI-style completion
/// response body. `None` if the JSON is malformed or the field is absent.
/// Pure — unit-tested (the response shape is what hosted backends must match).
pub fn parse_completion(body: &str) -> Option<String> {
    let j = Json::parse(body)?;
    j.get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .map(|s| s.to_string())
}

/// One OpenAI-style chat completion: POST the full message history, return
/// the assistant text.
pub fn chat_completion(cfg: &RemoteConfig, messages: &[(String, String)]) -> Result<String, String> {
    let body = build_chat_request(&cfg.model, messages);
    let url = format!("{}/v1/chat/completions", cfg.url);
    crate::ktrace::log_fmt(format_args!("remote: POST {url} ({} messages)", messages.len()));
    let resp = crate::net::http::post_json(&url, &body, cfg.key.as_deref(), HTTP_TIMEOUT_MS)?;
    if resp.status != 200 {
        let text = resp.text();
        let snip = &text[..text.len().min(300)];
        return Err(format!("HTTP {}: {}", resp.status, snip));
    }
    parse_completion(&resp.text()).ok_or_else(|| "no choices[0].message.content in response".to_string())
}

/// A remote chat session: the plain-text message history (system prompt
/// first), re-sent whole each turn — the server holds no state, so
/// save/clear/compact are all local list operations.
pub struct RemoteChat {
    cfg: RemoteConfig,
    messages: Vec<(String, String)>,
}

impl RemoteChat {
    pub fn new(cfg: RemoteConfig) -> Self {
        RemoteChat { cfg, messages: Vec::new() }
    }

    /// True when no messages have been exchanged yet (used to seed from a
    /// resumed orchestrator session).
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Seed the remote message list from a resumed orchestrator session so
    /// `/session resume` continues the conversation on the hosted backend too.
    pub fn hydrate_from_session(&mut self, session: &crate::agent::types::Session) {
        use crate::agent::types::Role;
        self.messages.clear();
        for m in &session.messages {
            match m.role {
                Role::System if !m.content.is_empty() => {
                    self.messages.push(("system".to_string(), m.content.clone()));
                }
                Role::User => self.messages.push(("user".to_string(), m.content.clone())),
                Role::Assistant if !m.content.is_empty() => {
                    self.messages.push(("assistant".to_string(), m.content.clone()));
                }
                Role::Tool => {
                    self.messages.push((
                        "user".to_string(),
                        format!("<tool_response>\n{}\n</tool_response>", m.content),
                    ));
                }
                _ => {}
            }
        }
        if self.messages.is_empty() {
            self.messages.push(("system".to_string(), super::agent_system_prompt()));
        }
    }

    /// One chat turn: the same bounded ReAct loop as the local
    /// `ChatSession::turn`, with generation done by the hosted model. Records
    /// into `session` so `/session save|resume` works for remote chat too.
    /// Returns the final assistant answer (the voice loop speaks it).
    pub fn turn(&mut self, msg: &str, session: &mut crate::agent::types::Session) -> String {
        use crate::agent::orchestrator::now;
        use crate::agent::types::{Provenance, Role, ToolCall};
        const MAX_TOOL_ITERS: usize = 4;
        if self.messages.is_empty() {
            self.messages.push(("system".to_string(), super::agent_system_prompt()));
        }
        self.messages.push(("user".to_string(), msg.to_string()));
        session.push_message(Role::User, msg.to_string(), Provenance::UserTyped, now());
        session.budget.turns_used = session.budget.turns_used.saturating_add(1);
        let mut last_call: Option<(String, String)> = None;
        let mut call_id = 1u64;
        for _ in 0..MAX_TOOL_ITERS {
            // The remote round-trip blocks in `net::http`; show a thinking spinner
            // (driven by `upkeep`, which the HTTP poll loop calls) while we wait.
            crate::shell::begin_thinking("thinking");
            let result = chat_completion(&self.cfg, &self.messages);
            crate::shell::end_thinking();
            let reply = match result {
                Ok(r) => r,
                Err(e) if e == "cancelled" => {
                    crate::serial_println!("\x1b[33m[stopped]\x1b[0m");
                    let _ = crate::session::save(session);
                    return String::new();
                }
                Err(e) => {
                    crate::serial_println!("\x1b[31mremote model error:\x1b[0m {} (see /model)", e);
                    let _ = crate::session::save(session);
                    return String::new();
                }
            };
            crate::serial_println!("\x1b[1;36mchitti[{}]:\x1b[0m {}", self.cfg.model, reply.trim());
            self.messages.push(("assistant".to_string(), reply.clone()));
            match super::parse_tool_call(&reply) {
                Some(pair) if last_call.as_ref() == Some(&pair) => {
                    crate::serial_println!("\x1b[33m[tool loop stopped: repeated call]\x1b[0m");
                    let _ = crate::session::save(session);
                    return String::new();
                }
                Some((cmd, args)) => {
                    last_call = Some((cmd.clone(), args.clone()));
                    crate::serial_println!("\x1b[33m\u{2192} running\x1b[0m /{} {}", cmd, args);
                    session.push_assistant_tool_calls(
                        String::new(),
                        alloc::vec![ToolCall { call_id, tool: cmd.clone(), args: args.clone() }],
                        now(),
                    );
                    session.budget.tool_calls_used = session.budget.tool_calls_used.saturating_add(1);
                    let obs = if cmd == "spawn_subagent" || cmd == "subagent" {
                        // Sub-agents stay on the local model; without one, say so.
                        "spawn_subagent is unavailable on the remote backend; do the task yourself with tools".to_string()
                    } else {
                        super::execute_chat_tool(&cmd, &args, session)
                    };
                    let prov = if obs.starts_with("error:") || obs.starts_with("Denied:") {
                        Provenance::SystemTrusted
                    } else {
                        Provenance::UntrustedIngested
                    };
                    session.push_tool_result(call_id, obs.clone(), prov, now());
                    call_id += 1;
                    self.messages.push(("user".to_string(), format!("<tool_response>\n{}\n</tool_response>", obs)));
                }
                None => {
                    session.push_message(Role::Assistant, reply.clone(), Provenance::SystemTrusted, now());
                    let _ = crate::session::save(session);
                    return reply;
                }
            }
        }
        crate::serial_println!("\x1b[33m[tool-call budget reached]\x1b[0m");
        let _ = crate::session::save(session);
        String::new()
    }

    /// `/compact` for the remote backend: the hosted model summarizes, then
    /// the history is rebuilt as system prompt + summary.
    pub fn compact(&mut self) {
        if self.messages.len() <= 1 {
            crate::serial_println!("(nothing to compact — empty context)");
            return;
        }
        let mut msgs = self.messages.clone();
        msgs.push((
            "user".to_string(),
            "Summarize this conversation so far in under 120 words: key facts, decisions, and open tasks. Reply with only the summary.".to_string(),
        ));
        match chat_completion(&self.cfg, &msgs) {
            Ok(summary) => {
                let before = self.messages.len();
                self.messages.clear();
                self.messages.push(("system".to_string(), super::agent_system_prompt()));
                self.messages.push(("system".to_string(), format!("Conversation so far (compacted): {}", summary.trim())));
                crate::serial_println!("(compacted: {} -> {} messages)", before, self.messages.len());
            }
            Err(e) => crate::serial_println!("compact> remote error: {}", e),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.cfg.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hosted-model request is the OpenAI `/v1/chat/completions` shape:
    /// model + role/content messages, non-streaming.
    #[test_case]
    fn builds_openai_chat_request() {
        let msgs = alloc::vec![
            ("system".to_string(), "You are Chitti.".to_string()),
            ("user".to_string(), "hi".to_string()),
        ];
        let body = build_chat_request("llama-3.1-8b", &msgs);
        assert!(body.contains("\"model\""));
        assert!(body.contains("llama-3.1-8b"));
        assert!(body.contains("\"role\""));
        assert!(body.contains("\"system\""));
        assert!(body.contains("\"user\""));
        assert!(body.contains("\"stream\""));
    }

    /// The assistant text is pulled from choices[0].message.content; malformed
    /// or empty responses yield None (the caller surfaces an error).
    #[test_case]
    fn parses_completion_content() {
        let ok = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"the answer is 42\"}}]}";
        assert_eq!(parse_completion(ok).as_deref(), Some("the answer is 42"));
        assert!(parse_completion("{\"choices\":[]}").is_none());
        assert!(parse_completion("not json").is_none());
        assert!(parse_completion("{\"error\":\"bad key\"}").is_none());
    }

    /// `/model` config round-trips through model.json (mode + endpoint + key).
    #[test_case]
    fn config_json_roundtrip() {
        // Persisted "remote" config parses back with mode active.
        let json = "{\"mode\":\"remote\",\"url\":\"http://192.168.1.20:8080\",\"model\":\"qwen3\",\"key\":\"sk-abc\"}";
        let j = crate::json::Json::parse(json).unwrap();
        assert_eq!(j.get("mode").and_then(|v| v.as_str()), Some("remote"));
        assert_eq!(j.get("url").and_then(|v| v.as_str()), Some("http://192.168.1.20:8080"));
    }
}
