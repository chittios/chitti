//! The shared type contract for the agentic re-architecture
//! (`CHITTI_SCHEMAS.md`): the three interfaces every `agent`/`session`/`skills`
//! phase reads and writes — [`AgentManifest`], [`Session`], and
//! [`SkillManifest`] (+ [`InstallRecord`]) — plus the Part-0 primitive types
//! they share (identifiers, [`Provenance`], the capability model).
//!
//! These types are the pinned contract; every phase reads and writes the same
//! shapes. Storage uses `postcard` (compact, no_std); JSON is for debug only.
//!
//! **Deviation from the schema doc's serde attributes (logged in DECISIONS.md):**
//! the doc annotates `Provenance`/`Scope`/`Origin`/`InstallSource` with
//! `#[serde(tag=..., content=...)]` to show a tidy *JSON* shape. Those are
//! internally/adjacently-tagged representations, which serde can only
//! deserialize via `deserialize_any` — a facility `postcard` (our canonical,
//! non-self-describing store format) does not provide. Since postcard is the
//! contract that actually round-trips, these enums use serde's default
//! *externally-tagged* form instead. No field *meaning* changes, so
//! `schema_version` stays 1; only the illustrative JSON shape differs.

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

// --- Identifiers (newtypes over u64; names are for humans/audit) ---

macro_rules! id_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
        pub struct $name(pub u64);
    };
}
id_newtype!(AgentId);
id_newtype!(SessionId);
id_newtype!(SkillId);
id_newtype!(MsgId);
// CapId: a LIVE, unforgeable token id minted by the cap system. A display/
// audit handle; the runtime authority remains the kernel `cap::Right` held in
// the owning task's own table (see DECISIONS.md #13).
id_newtype!(CapId);

/// Monotonic kernel ticks. No wall clock assumed; RTC optional and separate.
pub type Ticks = u64;

/// A key into the persistent memory store (the "disk").
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct StoreKey(pub String);

impl StoreKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// --- Monotonic id minting (deterministic, from 1) ---

macro_rules! id_minter {
    ($fn_name:ident, $ty:ident, $counter:ident) => {
        static $counter: AtomicU64 = AtomicU64::new(1);
        /// Mint the next unique id of this kind (monotonic, process-global).
        pub fn $fn_name() -> $ty {
            $ty($counter.fetch_add(1, Ordering::Relaxed))
        }
    };
}
id_minter!(next_agent_id, AgentId, AGENT_CTR);
id_minter!(next_session_id, SessionId, SESSION_CTR);
id_minter!(next_skill_id, SkillId, SKILL_CTR);
id_minter!(next_msg_id, MsgId, MSG_CTR);
id_minter!(next_cap_id, CapId, CAP_CTR);

// --- Provenance: the ONE taint type read by Phase E gating and Phase G bounding ---

/// Where a piece of context/content came from — the single taint tag the
/// Phase E gate reads and Phase G skill-bounding respects. Ordered from most to
/// least trusted; `UntrustedIngested` never authorizes a destructive effect.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Provenance {
    UserTyped,               // typed by the human at the shell — highest trust
    SystemTrusted,           // kernel/orchestrator-authored
    SkillInstalled(SkillId), // carried by an installed skill — bounded by its grant
    UntrustedIngested,       // read from FS / tool output — never authorizes side effects
}

impl Provenance {
    /// Fold two provenances to the *least* trusted (the taint join). Used to
    /// compute the worst provenance a justification traces to.
    pub fn join(self, other: Provenance) -> Provenance {
        // Higher rank = less trusted.
        fn rank(p: Provenance) -> u8 {
            match p {
                Provenance::UserTyped => 0,
                Provenance::SystemTrusted => 1,
                Provenance::SkillInstalled(_) => 2,
                Provenance::UntrustedIngested => 3,
            }
        }
        if rank(other) > rank(self) {
            other
        } else {
            self
        }
    }

    /// True for content that must never, on its own, authorize a destructive
    /// effect (the injection-defense predicate).
    pub fn is_untrusted(self) -> bool {
        matches!(self, Provenance::UntrustedIngested)
    }
}

// --- Capability model: declarative REQUEST vs live TOKEN ---

/// The kernel resource domain a capability governs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapDomain {
    Fs,          // memory filesystem
    Console,     // serial/framebuffer I/O
    Spawn,       // dispatch sub-agents
    Todo,        // mutate the session todo list
    Inference,   // request Cortex forward passes
    Ipc,         // send/receive IPC
    SkillManage, // install/uninstall skills (privileged)
    Channel,     // create inter-agent byte/datagram channels; use granted ends
    Net,         // network: listen/accept + outbound http (host/port scoped)
    Ui,          // own a compositor surface and draw to it
}

bitflags::bitflags! {
    /// The rights a capability confers. `DELETE` is destructive and always hits
    /// the Phase E taint gate.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
    pub struct Rights: u8 {
        const READ   = 0b0000_0001;
        const WRITE  = 0b0000_0010;
        const EXEC   = 0b0000_0100;
        const DELETE = 0b0000_1000; // destructive — always hits the Phase E gate
        const LIST   = 0b0001_0000;
    }
}

/// What a capability applies to.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Scope {
    Any,
    Path(String),     // glob, e.g. "/skills/pdf/**"
    Resource(String), // named resource
    /// A network host + inclusive port range. Host is a glob (`*.example.com`
    /// or `*` for any host); a *granted* scope names a range, a *target* names a
    /// single point (`port_lo == port_hi`).
    Net { host: String, port_lo: u16, port_hi: u16 },
}

impl Scope {
    /// Whether `self` is at least as permissive as `other` (subset check for
    /// attenuation). `Any` covers everything; a path/resource covers only an
    /// equal or more-specific-under-glob target. Unknown pairings are `false`,
    /// so a broader grant can never be synthesized from a narrower one.
    pub fn covers(&self, other: &Scope) -> bool {
        match (self, other) {
            (Scope::Any, _) => true,
            (Scope::Path(a), Scope::Path(b)) => glob_covers(a, b),
            (Scope::Resource(a), Scope::Resource(b)) => a == b,
            (
                Scope::Net { host: ha, port_lo: la, port_hi: ua },
                Scope::Net { host: hb, port_lo: lb, port_hi: ub },
            ) => host_glob_covers(ha, hb) && la <= lb && ub <= ua,
            _ => false,
        }
    }
}

/// Host-glob cover: `a` covers `b` if `a` is `*`, an exact match, or a
/// `*.suffix` wildcard that `b` ends with (matching a label boundary).
fn host_glob_covers(a: &str, b: &str) -> bool {
    if a == "*" || a == b {
        return true;
    }
    if let Some(suffix) = a.strip_prefix("*.") {
        // "*.example.com" covers "api.example.com" and "example.com".
        return b == suffix || b.ends_with(&{
            let mut s = String::from(".");
            s.push_str(suffix);
            s
        });
    }
    false
}

/// Minimal glob cover: `a` covers `b` when `a` equals `b`, or `a` ends in `**`
/// and `b` starts with `a`'s prefix. Sufficient for the `/work/**`-style scopes
/// the schemas use; not a full glob engine.
fn glob_covers(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if let Some(prefix) = a.strip_suffix("**") {
        return b.starts_with(prefix);
    }
    if let Some(prefix) = a.strip_suffix('*') {
        return b.starts_with(prefix) && !b[prefix.len()..].contains('/');
    }
    false
}

/// Declarative: what a manifest/skill ASKS for. Portable, human-readable, shown
/// verbatim in the install permission prompt. Never grants authority by itself.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct CapabilityRequest {
    pub domain: CapDomain,
    pub rights: Rights,
    pub scope: Scope,
}

impl CapabilityRequest {
    pub fn new(domain: CapDomain, rights: Rights, scope: Scope) -> Self {
        Self { domain, rights, scope }
    }

    /// Whether `self` fully contains `other`: same domain, `other.rights ⊆
    /// self.rights`, and `self.scope` covers `other.scope`. This is the atom of
    /// the attenuation rule — a sub-agent/skill request is admissible only if
    /// some granting request contains it.
    pub fn contains(&self, other: &CapabilityRequest) -> bool {
        self.domain == other.domain && self.rights.contains(other.rights) && self.scope.covers(&other.scope)
    }
}

/// Attenuation: the effective authority is `intersection(requested, granting)`.
/// A `requested` entry survives only if some `granting` entry contains it, and
/// then only with the rights both allow. This single rule is used at spawn
/// (sub-agent ∩ parent) and at install (skill ∩ user-approved).
pub fn intersect_caps(requested: &[CapabilityRequest], granting: &[CapabilityRequest]) -> Vec<CapabilityRequest> {
    let mut out = Vec::new();
    for req in requested {
        for grant in granting {
            if grant.domain == req.domain && grant.scope.covers(&req.scope) {
                let rights = req.rights & grant.rights;
                if !rights.is_empty() {
                    out.push(CapabilityRequest::new(req.domain, rights, req.scope.clone()));
                    break;
                }
            }
        }
    }
    out
}

/// Runtime: a LIVE capability held by a session/agent. `id` is the unforgeable
/// token handle; `req` is kept alongside for display and audit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capability {
    pub id: CapId,
    pub req: CapabilityRequest,
}

// =====================================================================
// Part 1 — AgentManifest
// =====================================================================

/// Defines a role: the orchestrator, a sub-agent type, or an installed
/// skill-agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentManifest {
    pub schema_version: u16,
    pub id: AgentId,
    pub name: String,
    pub version: String, // semver
    pub kind: AgentKind,
    pub description: String, // WHEN to dispatch this role — read by the orchestrator
    pub system_prompt: String,
    pub toolset: Vec<String>, // allowed tool names ("*" = all granted); gated by caps
    pub capabilities: Vec<CapabilityRequest>, // MAX authority this role may request
    pub skills: Vec<SkillRef>, // skill deps made available to this role
    pub sampling: Sampling,
    pub budgets: Budgets,
    pub summary: SummaryPolicy, // how a sub-agent condenses its result on return
    pub origin: Origin,         // where this manifest came from (trust)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Orchestrator,
    Subagent,
    SkillAgent,
    /// A long-running daemon (Network/SSH/HTTP/Doc/…): runs as a real scheduled
    /// task with a native `serve()` loop, not a request/response reasoning agent.
    /// Started/supervised by `crate::service`.
    Service,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillRef {
    pub id: SkillId,
    pub name: String,
    pub min_version: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,              // fixed seed → reproducible; tests use temp 0
    pub max_output_tokens: u32, // per turn
}

impl Sampling {
    /// The deterministic test/replay policy: exact greedy, fixed seed.
    pub const fn deterministic(seed: u64) -> Self {
        Self { temperature: 0.0, top_p: 1.0, seed, max_output_tokens: 512 }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Budgets {
    pub max_turns: u32,
    pub max_context_tokens: u32, // the KV/context window budget
    pub compact_threshold: u32,  // trigger auto-compaction at this live-token count
    pub max_tool_calls: u32,
    pub max_subagents: u16,
    pub max_depth: u8, // delegation depth cap (start 1–2)
    pub max_wall_ticks: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SummaryPolicy {
    pub max_tokens: u32,
    pub style: SummaryStyle,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryStyle {
    Terse,
    Structured,
    Verbatim,
}

/// Where a manifest came from (trust origin).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Origin {
    Builtin,
    Installed { skill: SkillId }, // came from a skill-agent package
}

// =====================================================================
// Part 2 — Session
// =====================================================================

/// The persisted, resumable unit. Deliberately does NOT hold sub-agent
/// transcripts (isolation — only their summaries) nor the raw KV cache
/// (recomputed on resume from the seed + messages).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub schema_version: u16,
    pub id: SessionId,
    pub created_ticks: Ticks,
    pub updated_ticks: Ticks,
    pub agent: AgentRef, // the orchestrator manifest driving this session
    pub seed: u64,       // session-level determinism seed
    pub messages: Vec<Message>,
    pub context: ContextState,
    pub todos: Vec<Todo>,
    pub env: Env,
    pub capabilities: Vec<Capability>, // LIVE, effective caps for the main agent
    pub skills_in_scope: Vec<SkillScope>,
    pub subagents: Vec<SubagentRecord>,
    pub budget: BudgetState,
    pub audit_cursor: u64, // offset into the append-only audit log
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentRef {
    pub manifest_id: AgentId,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: MsgId,
    pub role: Role,
    pub content: String,
    pub provenance: Provenance, // taint tag — read by the Phase E gate
    pub tool_calls: Vec<ToolCall>, // present on assistant turns that call tools
    pub tool_call_id: Option<u64>, // present on tool-result turns
    pub tokens: u32,
    pub ticks: Ticks,
    pub resident: bool, // true = live in context; false = compacted out
    pub store_ref: Option<StoreKey>, // where the full text lives once compacted
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: u64,
    pub tool: String,
    pub args: String, // JSON, grammar-constrained at emission
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextState {
    pub live_tokens: u32,
    pub window_limit: u32,
    pub compactions: Vec<CompactionRecord>,
    pub recall_index: Option<StoreKey>, // RAG index handle for demand-paged history
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionRecord {
    pub covers: (MsgId, MsgId), // inclusive range summarized out
    pub summary: String,        // the summary kept live in context
    pub summary_ref: StoreKey,
    pub tokens: u32,
    pub at_ticks: Ticks,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Todo {
    pub id: u32,
    pub text: String,
    pub status: TodoStatus,
    pub created_ticks: Ticks,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Done,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Env {
    pub cwd: String,
    pub vars: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillScope {
    pub skill: SkillId,
    pub loaded: LoadTier,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadTier {
    Metadata, // L0
    Body,     // L1
    Full,     // L2 — progressive disclosure
}

/// A sub-agent's ledger entry. Holds the returned SUMMARY, never the transcript
/// — the type has no transcript field, so the isolation leak is unrepresentable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentRecord {
    pub id: AgentId,
    pub manifest_id: AgentId,
    pub dispatched_ticks: Ticks,
    pub status: SubagentStatus,
    pub summary: Option<String>, // the ONLY thing that crosses back
    pub effective_caps: Vec<CapabilityRequest>, // for audit/display
    pub core: Option<u8>,        // which core it ran on (SMP)
    pub audit_ref: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetState {
    pub limits: Budgets,
    pub turns_used: u32,
    pub tool_calls_used: u32,
    pub subagents_used: u16,
    pub tokens_used: u64,
    pub wall_ticks_used: u64,
}

impl BudgetState {
    pub fn new(limits: Budgets) -> Self {
        Self {
            limits,
            turns_used: 0,
            tool_calls_used: 0,
            subagents_used: 0,
            tokens_used: 0,
            wall_ticks_used: 0,
        }
    }
}

// =====================================================================
// Part 3 — SkillManifest + InstallRecord
// =====================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillManifest {
    pub schema_version: u16,
    pub id: SkillId,
    pub name: String,
    pub version: String,
    pub description: String, // L0 — always cheap in the skill index
    pub kind: SkillKind,
    pub requested_capabilities: Vec<CapabilityRequest>, // shown verbatim at install
    pub body_ref: StoreKey,  // L1 — instruction body, loaded on match
    pub bundled_tools: Vec<BundledTool>,
    pub assets: Vec<Asset>,             // L2 — demand-paged on use
    pub agent: Option<AgentManifest>,   // present iff kind == SkillAgent
    /// The agent's SOUL.md (persona) text ref — `Some` for a SkillAgent whose
    /// package ships a soul. Placed into `/agent/<id>/SOUL.md` on install.
    pub soul_ref: Option<StoreKey>,
    /// Extra markdown procedure docs placed into `/agent/<id>/skills/*.md`.
    pub skill_docs: Vec<SkillDoc>,
    pub signature: SignatureBlock,
}

/// One markdown procedure doc shipped in an agent package — the "programming in
/// markdown" surface beyond the single L1 body. Placed into the agent's home
/// `skills/` dir; `trigger` is an optional match hint (None = always available).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillDoc {
    pub name: String,
    pub store_ref: StoreKey,
    pub trigger: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind {
    Skill,
    SkillAgent,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundledTool {
    pub name: String,
    pub description: String,
    pub input_schema: String,      // JSON schema
    pub synapse_primitive: String, // the deterministic executor it binds to
    pub required_caps: Vec<CapabilityRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub store_ref: StoreKey,
    pub bytes: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureBlock {
    pub algo: SigAlgo,          // Ed25519 to start
    pub key_id: String,         // which trusted key
    pub content_hash: [u8; 32], // hash of package contents
    pub sig: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SigAlgo {
    Ed25519,
}

/// Produced at install; the authoritative record of what was APPROVED. Not
/// shipped in the package.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallRecord {
    pub skill: SkillId,
    pub installed_ticks: Ticks,
    pub granted_capabilities: Vec<CapabilityRequest>, // the approved SUBSET of requested
    pub approved_by: String,                          // the user who consented at the shell
    pub source: InstallSource,
    pub verified: bool, // signature/hash checked and matched
    pub key_id: String,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum InstallSource {
    BootModule { name: String },
    Store { key: StoreKey },
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample_session() -> Session {
        Session {
            schema_version: 1,
            id: SessionId(9001),
            created_ticks: 1000,
            updated_ticks: 1450,
            agent: AgentRef { manifest_id: AgentId(1), version: "1.0.0".into() },
            seed: 42,
            messages: vec![Message {
                id: MsgId(2),
                role: Role::User,
                content: "Summarize report.txt".into(),
                provenance: Provenance::UserTyped,
                tool_calls: vec![],
                tool_call_id: None,
                tokens: 22,
                ticks: 1005,
                resident: true,
                store_ref: None,
            }],
            context: ContextState { live_tokens: 22, window_limit: 8192, compactions: vec![], recall_index: None },
            todos: vec![Todo { id: 1, text: "Read report.txt".into(), status: TodoStatus::Done, created_ticks: 1008 }],
            env: Env { cwd: "/work".into(), vars: BTreeMap::new() },
            capabilities: vec![Capability {
                id: CapId(5001),
                req: CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE | Rights::LIST, Scope::Any),
            }],
            skills_in_scope: vec![],
            subagents: vec![],
            budget: BudgetState::new(Budgets {
                max_turns: 64,
                max_context_tokens: 8192,
                compact_threshold: 6500,
                max_tool_calls: 256,
                max_subagents: 8,
                max_depth: 2,
                max_wall_ticks: 0,
            }),
            audit_cursor: 91,
        }
    }

    #[test_case]
    fn session_postcard_roundtrips() {
        let s = sample_session();
        let bytes = postcard::to_allocvec(&s).expect("serialize");
        let back: Session = postcard::from_bytes(&bytes).expect("deserialize");
        // Spot-check across every enum/flag kind that must survive postcard.
        assert_eq!(back.id, s.id);
        assert_eq!(back.messages[0].role, Role::User);
        assert_eq!(back.messages[0].provenance, Provenance::UserTyped);
        assert_eq!(back.todos[0].status, TodoStatus::Done);
        assert!(back.capabilities[0].req.rights.contains(Rights::WRITE));
        assert_eq!(back.budget.limits.max_depth, 2);
    }

    #[test_case]
    fn provenance_and_scope_survive_postcard() {
        // The enums whose serde attributes were changed for postcard: prove
        // they round-trip, including the data-carrying variants.
        for p in [
            Provenance::UserTyped,
            Provenance::SystemTrusted,
            Provenance::SkillInstalled(SkillId(700)),
            Provenance::UntrustedIngested,
        ] {
            let b = postcard::to_allocvec(&p).unwrap();
            assert_eq!(postcard::from_bytes::<Provenance>(&b).unwrap(), p);
        }
        for sc in [Scope::Any, Scope::Path("/work/**".into()), Scope::Resource("cam0".into())] {
            let b = postcard::to_allocvec(&sc).unwrap();
            assert_eq!(postcard::from_bytes::<Scope>(&b).unwrap(), sc);
        }
    }

    #[test_case]
    fn taint_join_picks_least_trusted() {
        assert_eq!(Provenance::UserTyped.join(Provenance::UntrustedIngested), Provenance::UntrustedIngested);
        assert_eq!(Provenance::SystemTrusted.join(Provenance::UserTyped), Provenance::SystemTrusted);
        assert!(Provenance::UntrustedIngested.is_untrusted());
        assert!(!Provenance::UserTyped.is_untrusted());
    }

    #[test_case]
    fn cap_attenuation_narrows_never_widens() {
        let parent = vec![CapabilityRequest::new(
            CapDomain::Fs,
            Rights::READ | Rights::WRITE | Rights::LIST,
            Scope::Any,
        )];
        // Child asks for READ+DELETE on /work/**; parent lacks DELETE and grants Any.
        let child = vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::DELETE, Scope::Path("/work/**".into()))];
        let eff = intersect_caps(&child, &parent);
        assert_eq!(eff.len(), 1);
        assert!(eff[0].rights.contains(Rights::READ));
        assert!(!eff[0].rights.contains(Rights::DELETE)); // narrowed away — never widened
        // A domain the parent lacks entirely is dropped.
        let child2 = vec![CapabilityRequest::new(CapDomain::SkillManage, Rights::EXEC, Scope::Any)];
        assert!(intersect_caps(&child2, &parent).is_empty());
    }
}
