//! Construction and bookkeeping for [`Session`]: create a fresh session from
//! an orchestrator manifest, append messages (updating token/turn accounting),
//! and estimate token counts. These are the primitives the agentic loop calls
//! as it drives a conversation.

use crate::agent::types::*;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Rough token estimate for budgeting/compaction: ~4 bytes/token. The kernel
/// has no tokenizer in the fast test build, and the loop only needs a monotone
/// proxy for "how full is the context", so a byte-based estimate is fine.
pub fn est_tokens(text: &str) -> u32 {
    (text.len() / 4 + 1) as u32
}

impl Session {
    /// A fresh session driven by `manifest`, seeded with `seed`. The manifest's
    /// `capabilities` are the MAX authority; `live` are the effective live caps
    /// already minted for this session's agent (⊆ manifest.capabilities).
    pub fn new(manifest: &AgentManifest, seed: u64, live: Vec<Capability>, now: Ticks) -> Session {
        let mut s = Session {
            schema_version: 1,
            id: next_session_id(),
            created_ticks: now,
            updated_ticks: now,
            agent: AgentRef { manifest_id: manifest.id, version: manifest.version.clone() },
            seed,
            messages: Vec::new(),
            context: ContextState {
                live_tokens: 0,
                window_limit: manifest.budgets.max_context_tokens,
                compactions: Vec::new(),
                recall_index: None,
            },
            todos: Vec::new(),
            env: Env { cwd: "/".to_string(), vars: alloc::collections::BTreeMap::new() },
            capabilities: live,
            skills_in_scope: Vec::new(),
            subagents: Vec::new(),
            budget: BudgetState::new(manifest.budgets),
            audit_cursor: 0,
        };
        // The system prompt is message 0, trusted, resident.
        s.push_message(Role::System, manifest.system_prompt.clone(), Provenance::SystemTrusted, now);
        s
    }

    /// Append a plain message (no tool calls). Returns its id.
    pub fn push_message(&mut self, role: Role, content: String, provenance: Provenance, now: Ticks) -> MsgId {
        let tokens = est_tokens(&content);
        let id = next_msg_id();
        self.context.live_tokens += tokens;
        self.messages.push(Message {
            id,
            role,
            content,
            provenance,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tokens,
            ticks: now,
            resident: true,
            store_ref: None,
        });
        self.updated_ticks = now;
        id
    }

    /// Append an assistant turn that emits tool calls. Returns its id.
    pub fn push_assistant_tool_calls(&mut self, content: String, calls: Vec<ToolCall>, now: Ticks) -> MsgId {
        let tokens = est_tokens(&content) + calls.iter().map(|c| est_tokens(&c.args)).sum::<u32>();
        let id = next_msg_id();
        self.context.live_tokens += tokens;
        self.messages.push(Message {
            id,
            role: Role::Assistant,
            content,
            provenance: Provenance::SystemTrusted,
            tool_calls: calls,
            tool_call_id: None,
            tokens,
            ticks: now,
            resident: true,
            store_ref: None,
        });
        self.updated_ticks = now;
        id
    }

    /// Append a tool-result turn tagged with the tool output's provenance
    /// (usually `UntrustedIngested` — the taint that Phase E gating reads).
    pub fn push_tool_result(&mut self, call_id: u64, content: String, provenance: Provenance, now: Ticks) -> MsgId {
        let tokens = est_tokens(&content);
        let id = next_msg_id();
        self.context.live_tokens += tokens;
        self.messages.push(Message {
            id,
            role: Role::Tool,
            content,
            provenance,
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id),
            tokens,
            ticks: now,
            resident: true,
            store_ref: None,
        });
        self.updated_ticks = now;
        id
    }

    /// The worst (least-trusted) provenance among the currently *resident*
    /// messages — the taint a tool call issued now would be justified by. This
    /// is what the Phase E gate folds over to defend against injection.
    pub fn resident_max_taint(&self) -> Provenance {
        self.messages
            .iter()
            .filter(|m| m.resident)
            .map(|m| m.provenance)
            .fold(Provenance::SystemTrusted, |acc, p| acc.join(p))
    }

    /// A compact transcript of resident messages, for a StepSource that needs
    /// to see the conversation (real Cortex renders this into tokens).
    pub fn transcript(&self) -> Vec<(Role, &str)> {
        self.messages.iter().filter(|m| m.resident).map(|m| (m.role, m.content.as_str())).collect()
    }

    /// Drop the conversation while keeping the same session id, agent ref,
    /// seed, and live capabilities. Keeps the system prompt (message 0) so a
    /// subsequent turn still has the orchestrator persona. Used by `/clear`
    /// so the shell chat and the persisted session stay in lock-step.
    pub fn clear_transcript(&mut self, now: Ticks) {
        let system = self.messages.first().filter(|m| m.role == Role::System).cloned();
        self.messages.clear();
        self.context.live_tokens = 0;
        self.context.compactions.clear();
        self.context.recall_index = None;
        self.todos.clear();
        self.subagents.clear();
        self.budget.turns_used = 0;
        self.budget.tool_calls_used = 0;
        self.budget.subagents_used = 0;
        self.budget.tokens_used = 0;
        self.budget.wall_ticks_used = 0;
        if let Some(sys) = system {
            self.context.live_tokens = sys.tokens;
            self.messages.push(sys);
        }
        self.updated_ticks = now;
    }
}
