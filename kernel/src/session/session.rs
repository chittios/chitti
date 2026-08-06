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
            schema_version: crate::session::store::SCHEMA_VERSION,
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
            origins: Vec::new(),
            trusted_origins: Vec::new(),
            remote_planner_used: false,
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
            origin: None,
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
            origin: None,
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
            origin: None,
        });
        self.updated_ticks = now;
        id
    }

    /// Append a tool-result turn, recording **where the content came from**.
    ///
    /// The sibling of `push_tool_result` rather than a new parameter on it:
    /// only the paths that actually have an origin in hand need to change, and
    /// the dozen callers that do not stay untouched and keep meaning "unknown
    /// source".
    pub fn push_tool_result_from(
        &mut self,
        call_id: u64,
        content: String,
        provenance: Provenance,
        origin: Option<&str>,
        now: Ticks,
    ) -> MsgId {
        let idx = origin.and_then(|o| self.intern_origin(o));
        let id = self.push_tool_result(call_id, content, provenance, now);
        if let Some(m) = self.messages.last_mut() {
            m.origin = idx;
        }
        id
    }

    /// Intern a source name, returning its index in `origins`.
    ///
    /// Append-only and capped. An index is **never** reused: live messages hold
    /// indices into this table, so recycling a slot would silently relabel an
    /// existing message's source. Past the cap we return `None` -- "unknown
    /// source" -- rather than reusing or evicting, which degrades to the old
    /// behaviour instead of to a confident wrong attribution.
    pub fn intern_origin(&mut self, name: &str) -> Option<u16> {
        const MAX_ORIGINS: usize = 256;
        const MAX_ORIGIN_LEN: usize = 120;
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        if let Some(i) = self.origins.iter().position(|o| o == name) {
            return Some(i as u16);
        }
        if self.origins.len() >= MAX_ORIGINS {
            crate::ktrace::log_fmt(format_args!(
                "session: origin table full ({MAX_ORIGINS}); '{name}' recorded as unknown source"
            ));
            return None;
        }
        let mut owned = String::from(name);
        if owned.len() > MAX_ORIGIN_LEN {
            let cut = owned.char_indices().map(|(i, _)| i).take_while(|i| *i <= MAX_ORIGIN_LEN).last().unwrap_or(0);
            owned.truncate(cut);
        }
        self.origins.push(owned);
        Some((self.origins.len() - 1) as u16)
    }

    /// The recorded source of a message, if it has one.
    ///
    /// Reads through `get`, never indexing: a hand-edited or truncated session
    /// blob must yield "unknown source", not a panic.
    pub fn origin_of(&self, m: &Message) -> Option<&str> {
        m.origin.and_then(|i| self.origins.get(i as usize)).map(|s| s.as_str())
    }

    /// This message's provenance *after* any human declassification of its
    /// source.
    ///
    /// Sticky trust is applied here, folded into the computation of taint,
    /// rather than as a special case in a gate. Everything downstream --
    /// `Router::justification`, `to_taint`, `blocks_destructive`, the executor's
    /// gate 3, the audit record -- is untouched, because the *input* to the fold
    /// changes and the predicate does not. A gate that had learned a second
    /// reason to permit would be a bypass path to keep in step forever.
    ///
    /// Two deliberate properties. The downgrade is to `UserTyped`, not
    /// `SystemTrusted`: a human vouched for it, the kernel did not author it.
    /// (Note this distinction is per message -- `resident_max_taint` folds from
    /// a `SystemTrusted` seed, and in this lattice `UserTyped` is rank 0, the
    /// most trusted, so the seed dominates the fold. Nothing reads the
    /// difference today; the tag is the honest one for when something does.)
    /// And a message with **no** origin can never be declassified, so the
    /// feature is exactly as safe as origin coverage is complete, and an
    /// ingestion path that forgets to name its source fails closed.
    fn effective_provenance(&self, m: &Message) -> Provenance {
        if m.provenance == Provenance::UntrustedIngested {
            if let Some(i) = m.origin {
                if self.trusted_origins.contains(&i) {
                    return Provenance::UserTyped;
                }
            }
        }
        m.provenance
    }

    /// Mark a source as declassified for the rest of this session.
    pub fn trust_origin(&mut self, idx: u16) {
        if !self.trusted_origins.contains(&idx) {
            self.trusted_origins.push(idx);
        }
    }

    /// Revoke a declassification. Returns whether anything was revoked.
    pub fn untrust_origin(&mut self, idx: u16) -> bool {
        let before = self.trusted_origins.len();
        self.trusted_origins.retain(|i| *i != idx);
        self.trusted_origins.len() != before
    }

    /// The sources the human has declassified, by name.
    pub fn trusted_origin_names(&self) -> Vec<&str> {
        self.trusted_origins.iter().filter_map(|i| self.origins.get(*i as usize)).map(|s| s.as_str()).collect()
    }

    /// The worst (least-trusted) provenance among the currently *resident*
    /// messages — the taint a tool call issued now would be justified by. This
    /// is what the Phase E gate folds over to defend against injection.
    pub fn resident_max_taint(&self) -> Provenance {
        self.messages
            .iter()
            .filter(|m| m.resident)
            .map(|m| self.effective_provenance(m))
            .fold(Provenance::SystemTrusted, |acc, p| acc.join(p))
    }

    /// The worst provenance among resident messages **ignoring** any human
    /// declassification.
    ///
    /// Sticky trust is a bounded grant, and the bound is the effect: vouching
    /// for a source so the agent may tidy a local file is a different act from
    /// vouching so it may send that file somewhere. Egress asks this question
    /// instead, so a declassified source still blocks exfiltration.
    pub fn resident_max_taint_strict(&self) -> Provenance {
        self.messages
            .iter()
            .filter(|m| m.resident)
            .map(|m| m.provenance)
            .fold(Provenance::SystemTrusted, |acc, p| acc.join(p))
    }

    /// The text of every resident message that arrived untrusted.
    ///
    /// This is what a value-granular policy needs and a whole-turn one does not:
    /// not "was anything untrusted here" but "*what* was, so we can ask whether
    /// this particular call derives from it".
    pub fn untrusted_excerpts(&self) -> Vec<&str> {
        self.untrusted_sources().into_iter().map(|(_, t)| t).collect()
    }

    /// Every resident untrusted message as `(source, text)`.
    ///
    /// What the approval dialogue needs: a human asked to approve a delete
    /// "justified by ingested content" can only answer by reading a payload,
    /// where one told the content came from `evil.example` is deciding about a
    /// source -- which is the decision the policy actually needs from them, and
    /// the only one they hold information the kernel lacks about.
    ///
    /// A message still declassified by [`Session::trust_origin`] is excluded:
    /// it no longer justifies anything, so showing it would be asking about a
    /// source the human already ruled on.
    pub fn untrusted_sources(&self) -> Vec<(Option<&str>, &str)> {
        self.messages
            .iter()
            .filter(|m| m.resident && self.effective_provenance(m) == Provenance::UntrustedIngested)
            .map(|m| (self.origin_of(m), m.content.as_str()))
            .collect()
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
        // The origin table indexes the messages we just dropped, and the trust
        // set indexes the origin table. Keeping either would grow unboundedly
        // across repeated `/clear` and -- worse -- would carry a human's
        // declassification into a conversation they did not make it in. The
        // system prompt is trusted and carries no origin, so this is safe.
        self.origins.clear();
        self.trusted_origins.clear();
        if let Some(sys) = system {
            self.context.live_tokens = sys.tokens;
            self.messages.push(sys);
        }
        self.updated_ticks = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::{AgentRef, BudgetState, Budgets, ContextState, Env, SessionId};

    fn sess() -> Session {
        Session {
            schema_version: crate::session::store::SCHEMA_VERSION,
            id: SessionId(4242),
            created_ticks: 0,
            updated_ticks: 0,
            agent: AgentRef { manifest_id: crate::agent::types::AgentId(1), version: "1".into() },
            seed: 1,
            messages: Vec::new(),
            context: ContextState { live_tokens: 0, window_limit: 4096, compactions: Vec::new(), recall_index: None },
            todos: Vec::new(),
            env: Env { cwd: "/".to_string(), vars: alloc::collections::BTreeMap::new() },
            capabilities: Vec::new(),
            skills_in_scope: Vec::new(),
            subagents: Vec::new(),
            budget: BudgetState::new(Budgets {
                max_turns: 8,
                max_context_tokens: 4096,
                compact_threshold: 3000,
                max_tool_calls: 256,
                max_subagents: 2,
                max_wall_ticks: 1_000_000,
                max_depth: 2,
            }),
            audit_cursor: 0,
            origins: Vec::new(),
            trusted_origins: Vec::new(),
            remote_planner_used: false,
        }
    }

    #[test_case]
    fn intern_origin_dedups_and_caps_without_reusing_an_index() {
        let mut s = sess();
        let a = s.intern_origin("host:evil.example").unwrap();
        let b = s.intern_origin("host:evil.example").unwrap();
        assert_eq!(a, b, "the same source must intern to the same index");
        assert_eq!(s.origins.len(), 1);
        assert!(s.intern_origin("   ").is_none(), "an empty name is not a source");

        // Past the cap we report "unknown source" rather than evicting: a live
        // message holds an index, so recycling one would relabel its source --
        // and under sticky trust would transfer a human's decision to content
        // they never saw.
        for i in 0..300 {
            let _ = s.intern_origin(&alloc::format!("host:h{i}.example"));
        }
        assert_eq!(s.origins.len(), 256);
        assert_eq!(s.origins[a as usize], "host:evil.example", "an existing index must never be reused");
        assert!(s.intern_origin("host:one-more.example").is_none());
    }

    #[test_case]
    fn an_out_of_range_origin_index_reads_as_unknown_not_a_panic() {
        let mut s = sess();
        s.push_message(Role::Tool, "x".into(), Provenance::UntrustedIngested, 0);
        if let Some(m) = s.messages.last_mut() {
            m.origin = Some(9999); // a truncated or hand-edited snapshot
        }
        let m = s.messages.last().unwrap().clone();
        assert_eq!(s.origin_of(&m), None);
    }

    /// Sticky declassification, and the property that makes it tolerable.
    #[test_case]
    fn trusting_one_source_leaves_the_others_gating() {
        let mut s = sess();
        s.push_tool_result_from(1, "poisoned".into(), Provenance::UntrustedIngested, Some("host:evil.example"), 0);
        s.push_tool_result_from(2, "also poisoned".into(), Provenance::UntrustedIngested, Some("host:other.example"), 0);
        assert_eq!(s.resident_max_taint(), Provenance::UntrustedIngested);
        assert_eq!(s.untrusted_sources().len(), 2);

        let evil = s.intern_origin("host:evil.example").unwrap();
        s.trust_origin(evil);
        assert_eq!(
            s.resident_max_taint(),
            Provenance::UntrustedIngested,
            "the other source still gates -- trust is per source, not per turn"
        );
        assert_eq!(s.untrusted_sources().len(), 1, "a declassified source stops being a reason to ask");

        let other = s.intern_origin("host:other.example").unwrap();
        s.trust_origin(other);
        // With every source declassified nothing blocks any more. Note the fold
        // reports `SystemTrusted` rather than `UserTyped`: this lattice ranks
        // UserTyped as the *most* trusted (rank 0) and the fold is seeded at
        // SystemTrusted, so the seed dominates. The per-message downgrade is
        // still UserTyped -- what matters here, and the only thing any gate
        // reads, is that the turn is no longer untrusted.
        assert!(!s.resident_max_taint().is_untrusted(), "a fully declassified turn must not block");
        assert!(s.untrusted_sources().is_empty());

        assert!(s.untrust_origin(evil));
        assert_eq!(s.resident_max_taint(), Provenance::UntrustedIngested, "revocation must actually revoke");
    }

    /// The fail-closed rule: an ingestion path that did not record a source can
    /// never be trusted away, so the feature is exactly as safe as origin
    /// coverage is complete.
    /// A declassified source is still refused egress. The grant is bounded by
    /// the effect, and this is the bound: vouching for a document so the agent
    /// can tidy a local file is not vouching for sending that document away.
    #[test_case]
    fn declassification_does_not_extend_to_egress() {
        let mut s = sess();
        s.push_tool_result_from(1, "secret".into(), Provenance::UntrustedIngested, Some("host:evil.example"), 0);
        let i = s.intern_origin("host:evil.example").unwrap();
        s.trust_origin(i);
        assert!(!s.resident_max_taint().is_untrusted(), "local work is permitted after the grant");
        assert!(
            s.resident_max_taint_strict().is_untrusted(),
            "egress must still see the untrusted source"
        );
    }

    #[test_case]
    fn content_with_no_recorded_source_is_never_declassified() {
        let mut s = sess();
        s.push_tool_result(1, "from somewhere".into(), Provenance::UntrustedIngested, 0);
        // Trust *every* index there could be.
        s.trusted_origins = (0u16..64).collect();
        assert_eq!(
            s.resident_max_taint(),
            Provenance::UntrustedIngested,
            "unknown source must not be declassifiable"
        );
    }

    #[test_case]
    fn clear_transcript_drops_the_origin_table_and_the_trust_set() {
        let mut s = sess();
        s.push_tool_result_from(1, "x".into(), Provenance::UntrustedIngested, Some("host:evil.example"), 0);
        let i = s.intern_origin("host:evil.example").unwrap();
        s.trust_origin(i);
        s.clear_transcript(1);
        assert!(s.origins.is_empty(), "stale indices would relabel future messages");
        assert!(s.trusted_origins.is_empty(), "a grant must not survive into a new conversation");
    }
}
