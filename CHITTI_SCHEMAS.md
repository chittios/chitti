# Chitti OS — Anchor Schemas

The three interfaces every phase in `CHITTI_AGENTIC_HANDOFF.md` reads and writes:

1. **`AgentManifest`** — defines an agent role (orchestrator, sub-agent, or installed skill-agent).
2. **`Session`** — the serialized, resumable unit of persistence.
3. **`SkillManifest`** (+ `InstallRecord`) — a portable, permissioned skill package.

Rust type definitions are the source of truth; JSON examples show the serialized shape. Structs assume `serde` (no_std + `alloc`) with `#[derive(Serialize, Deserialize)]`. Storage uses `postcard` (compact, no_std); JSON is for debug/inspection and the shapes shown here.

> **Single-model note:** there is one bundled model, so the manifest carries a `sampling` policy (temperature / top-p / seed), **not** model routing.

---

## Part 0 — Shared primitive types

These are referenced by all three schemas. Define once (e.g. `kernel/src/agent/types.rs`) and re-export.

```rust
use alloc::{string::String, vec::Vec, collections::BTreeMap};

// --- Identifiers (newtypes over u64; names are for humans/audit) ---
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentId(pub u64);
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId(pub u64);
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillId(pub u64);
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MsgId(pub u64);
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapId(pub u64);          // a LIVE, unforgeable token minted by the cap system

/// Monotonic kernel ticks. No wall clock assumed; RTC optional and separate.
pub type Ticks = u64;

/// A key into the persistent memory store (the "disk").
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreKey(pub String);

// --- Provenance: the ONE taint type read by Phase E gating and Phase G bounding ---
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "src")]
pub enum Provenance {
    UserTyped,                 // typed by the human at the shell — highest trust
    SystemTrusted,             // kernel/orchestrator-authored
    SkillInstalled(SkillId),   // carried by an installed skill — bounded by its grant
    UntrustedIngested,         // read from FS / tool output — never authorizes side effects
}

// --- Capability model: declarative REQUEST vs live TOKEN ---

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapDomain {
    Fs,          // memory filesystem
    Console,     // serial/framebuffer I/O
    Spawn,       // dispatch sub-agents
    Todo,        // mutate the session todo list
    Inference,   // request Cortex forward passes
    Ipc,         // send/receive IPC
    SkillManage, // install/uninstall skills (privileged)
}

bitflags::bitflags! {
    #[derive(Serialize, Deserialize)]
    pub struct Rights: u8 {
        const READ   = 0b0000_0001;
        const WRITE  = 0b0000_0010;
        const EXEC   = 0b0000_0100;
        const DELETE = 0b0000_1000; // destructive — always hits the Phase E gate
        const LIST   = 0b0001_0000;
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "t", content = "v")]
pub enum Scope {
    Any,
    Path(String),      // glob, e.g. "/skills/pdf/**"
    Resource(String),  // named resource
}

/// Declarative: what a manifest/skill ASKS for. Portable, human-readable,
/// shown verbatim in the install permission prompt. Never grants authority by itself.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    pub domain: CapDomain,
    pub rights: Rights,
    pub scope: Scope,
}

/// Runtime: a LIVE capability held by a session/agent. `id` is the unforgeable token;
/// `req` is kept alongside for display and audit.
#[derive(Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: CapId,
    pub req: CapabilityRequest,
}
```

Effective authority is always `intersection(requested, granting-context)` — computed at spawn (sub-agent ∩ parent) and at install (skill ∩ user-approved). This single rule is enforced in `cap/`.

---

## Part 1 — `AgentManifest`

Defines a role. The orchestrator is one; each sub-agent type is one; an installed skill-agent registers one. Human-authored manifests may be TOML, deserialized into this struct; the canonical form is below.

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentManifest {
    pub schema_version: u16,
    pub id: AgentId,
    pub name: String,
    pub version: String,               // semver
    pub kind: AgentKind,
    pub description: String,           // WHEN to dispatch this role — read by the orchestrator
    pub system_prompt: String,
    pub toolset: Vec<String>,          // allowed tool names ("*" = all granted); authority still gated by caps
    pub capabilities: Vec<CapabilityRequest>, // MAX authority this role may request
    pub skills: Vec<SkillRef>,         // skill deps made available to this role
    pub sampling: Sampling,
    pub budgets: Budgets,
    pub summary: SummaryPolicy,        // how a sub-agent condenses its result on return
    pub origin: Origin,                // where this manifest came from (trust)
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind { Orchestrator, Subagent, SkillAgent }

#[derive(Clone, Serialize, Deserialize)]
pub struct SkillRef { pub id: SkillId, pub name: String, pub min_version: String }

#[derive(Clone, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,                     // fixed seed → reproducible; tests use temp 0
    pub max_output_tokens: u32,        // per turn
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Budgets {
    pub max_turns: u32,
    pub max_context_tokens: u32,       // the KV/context window budget
    pub compact_threshold: u32,        // trigger auto-compaction at this live-token count
    pub max_tool_calls: u32,
    pub max_subagents: u16,
    pub max_depth: u8,                 // delegation depth cap (start 1–2)
    pub max_wall_ticks: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SummaryPolicy { pub max_tokens: u32, pub style: SummaryStyle }

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryStyle { Terse, Structured, Verbatim }

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Origin {
    Builtin,
    Installed { skill: SkillId },      // came from a skill-agent package
}
```

### Example — orchestrator manifest (JSON)

```json
{
  "schema_version": 1,
  "id": 1,
  "name": "orchestrator",
  "version": "1.0.0",
  "kind": "orchestrator",
  "description": "Interactive main agent behind the intent shell.",
  "system_prompt": "You are Chitti's main agent. Plan, use tools, delegate to sub-agents when a task is self-contained, and keep the user informed.",
  "toolset": ["read", "write", "edit", "list", "search", "run", "spawn_subagent", "todo_write", "load_skill", "emit_result"],
  "capabilities": [
    { "domain": "fs", "rights": ["READ","WRITE","LIST"], "scope": { "t": "any" } },
    { "domain": "console", "rights": ["READ","WRITE"], "scope": { "t": "any" } },
    { "domain": "spawn", "rights": ["EXEC"], "scope": { "t": "any" } },
    { "domain": "todo", "rights": ["READ","WRITE"], "scope": { "t": "any" } },
    { "domain": "inference", "rights": ["EXEC"], "scope": { "t": "any" } }
  ],
  "skills": [],
  "sampling": { "temperature": 0.2, "top_p": 0.9, "seed": 42, "max_output_tokens": 1024 },
  "budgets": {
    "max_turns": 64, "max_context_tokens": 8192, "compact_threshold": 6500,
    "max_tool_calls": 256, "max_subagents": 8, "max_depth": 2, "max_wall_ticks": 0
  },
  "summary": { "max_tokens": 512, "style": "structured" },
  "origin": { "kind": "builtin" }
}
```

A sub-agent role differs only by `kind: "subagent"`, a narrower `toolset`, a smaller `capabilities` set, and usually a tighter budget — its effective caps at dispatch are `min(these, parent's)`.

---

## Part 2 — `Session`

The persisted, resumable state. Note what it deliberately does **not** hold: sub-agent transcripts (isolation — only their summaries) and the raw KV cache (recomputed on resume from the seed + messages).

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub schema_version: u16,
    pub id: SessionId,
    pub created_ticks: Ticks,
    pub updated_ticks: Ticks,
    pub agent: AgentRef,               // the orchestrator manifest driving this session
    pub seed: u64,                     // session-level determinism seed
    pub messages: Vec<Message>,
    pub context: ContextState,
    pub todos: Vec<Todo>,
    pub env: Env,
    pub capabilities: Vec<Capability>, // LIVE, effective caps for the main agent
    pub skills_in_scope: Vec<SkillScope>,
    pub subagents: Vec<SubagentRecord>,
    pub budget: BudgetState,
    pub audit_cursor: u64,             // offset into the append-only audit log
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AgentRef { pub manifest_id: AgentId, pub version: String }

#[derive(Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: MsgId,
    pub role: Role,
    pub content: String,
    pub provenance: Provenance,        // taint tag — read by the Phase E gate
    pub tool_calls: Vec<ToolCall>,     // present on assistant turns that call tools
    pub tool_call_id: Option<u64>,     // present on tool-result turns
    pub tokens: u32,
    pub ticks: Ticks,
    pub resident: bool,                // true = live in context; false = compacted out
    pub store_ref: Option<StoreKey>,   // where the full text lives once compacted
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role { System, User, Assistant, Tool }

#[derive(Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: u64,
    pub tool: String,
    pub args: String,                  // JSON, grammar-constrained at emission
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ContextState {
    pub live_tokens: u32,
    pub window_limit: u32,
    pub compactions: Vec<CompactionRecord>,
    pub recall_index: Option<StoreKey>, // RAG index handle for demand-paged history
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CompactionRecord {
    pub covers: (MsgId, MsgId),        // inclusive range summarized out
    pub summary: String,               // the summary kept live in context
    pub summary_ref: StoreKey,
    pub tokens: u32,
    pub at_ticks: Ticks,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Todo {
    pub id: u32,
    pub text: String,
    pub status: TodoStatus,
    pub created_ticks: Ticks,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus { Pending, InProgress, Done, Cancelled }

#[derive(Clone, Serialize, Deserialize)]
pub struct Env { pub cwd: String, pub vars: BTreeMap<String, String> }

#[derive(Clone, Serialize, Deserialize)]
pub struct SkillScope { pub skill: SkillId, pub loaded: LoadTier }

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadTier { Metadata, Body, Full } // L0 / L1 / L2 — progressive disclosure

/// A sub-agent's ledger entry. Holds the returned SUMMARY, never the transcript.
#[derive(Clone, Serialize, Deserialize)]
pub struct SubagentRecord {
    pub id: AgentId,
    pub manifest_id: AgentId,
    pub dispatched_ticks: Ticks,
    pub status: SubagentStatus,
    pub summary: Option<String>,       // the ONLY thing that crosses back
    pub effective_caps: Vec<CapabilityRequest>, // for audit/display
    pub core: Option<u8>,              // which core it ran on (SMP)
    pub audit_ref: u64,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus { Running, Completed, Failed }

#[derive(Clone, Serialize, Deserialize)]
pub struct BudgetState {
    pub limits: Budgets,
    pub turns_used: u32,
    pub tool_calls_used: u32,
    pub subagents_used: u16,
    pub tokens_used: u64,
    pub wall_ticks_used: u64,
}
```

### Example — a session mid-task (JSON, trimmed)

```json
{
  "schema_version": 1,
  "id": 9001,
  "created_ticks": 1000,
  "updated_ticks": 1450,
  "agent": { "manifest_id": 1, "version": "1.0.0" },
  "seed": 42,
  "messages": [
    { "id": 1, "role": "system", "content": "You are Chitti's main agent...", "provenance": { "kind": "system_trusted" }, "tool_calls": [], "tool_call_id": null, "tokens": 180, "ticks": 1000, "resident": false, "store_ref": "sess/9001/msg/1" },
    { "id": 2, "role": "user", "content": "Summarize report.txt and note any risks in risks.md", "provenance": { "kind": "user_typed" }, "tool_calls": [], "tool_call_id": null, "tokens": 22, "ticks": 1005, "resident": true, "store_ref": null },
    { "id": 3, "role": "assistant", "content": "", "provenance": { "kind": "system_trusted" }, "tool_calls": [ { "call_id": 1, "tool": "read", "args": "{\"path\":\"/work/report.txt\"}" } ], "tool_call_id": null, "tokens": 14, "ticks": 1010, "resident": true, "store_ref": null },
    { "id": 4, "role": "tool", "content": "<file contents…>", "provenance": { "kind": "untrusted_ingested" }, "tool_calls": [], "tool_call_id": 1, "tokens": 900, "ticks": 1012, "resident": true, "store_ref": null }
  ],
  "context": {
    "live_tokens": 1136,
    "window_limit": 8192,
    "compactions": [
      { "covers": [1, 1], "summary": "System prompt (compacted).", "summary_ref": "sess/9001/cmp/1", "tokens": 20, "at_ticks": 1400 }
    ],
    "recall_index": "sess/9001/ragidx"
  },
  "todos": [
    { "id": 1, "text": "Read report.txt", "status": "done", "created_ticks": 1008 },
    { "id": 2, "text": "Write risks.md", "status": "in_progress", "created_ticks": 1008 }
  ],
  "env": { "cwd": "/work", "vars": {} },
  "capabilities": [
    { "id": 5001, "req": { "domain": "fs", "rights": ["READ","WRITE","LIST"], "scope": { "t": "any" } } },
    { "id": 5002, "req": { "domain": "inference", "rights": ["EXEC"], "scope": { "t": "any" } } }
  ],
  "skills_in_scope": [ { "skill": 700, "loaded": "metadata" } ],
  "subagents": [
    { "id": 12, "manifest_id": 3, "dispatched_ticks": 1300, "status": "completed", "summary": "report.txt is a Q3 vendor review; 3 risks flagged.", "effective_caps": [ { "domain": "fs", "rights": ["READ"], "scope": { "t": "path", "v": "/work/**" } } ], "core": 2, "audit_ref": 88 }
  ],
  "budget": {
    "limits": { "max_turns": 64, "max_context_tokens": 8192, "compact_threshold": 6500, "max_tool_calls": 256, "max_subagents": 8, "max_depth": 2, "max_wall_ticks": 0 },
    "turns_used": 3, "tool_calls_used": 2, "subagents_used": 1, "tokens_used": 1300, "wall_ticks_used": 450
  },
  "audit_cursor": 91
}
```

Note the taint flow in `messages`: the user turn is `user_typed`, but the file contents pulled by `read` are `untrusted_ingested`. If the model later tries a `DELETE` justified by message 4, the Phase E gate fires — exactly the injection defense.

---

## Part 3 — `SkillManifest` + `InstallRecord`

The package manifest **requests**; installation **grants** a subset. Keep them separate: the manifest ships inside the (signed) package; the `InstallRecord` is produced by the install flow and lives in the local skill registry.

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    pub schema_version: u16,
    pub id: SkillId,
    pub name: String,
    pub version: String,
    pub description: String,            // L0 — always cheap in the skill index
    pub kind: SkillKind,
    pub requested_capabilities: Vec<CapabilityRequest>, // shown verbatim at install
    pub body_ref: StoreKey,            // L1 — instruction body, loaded on match
    pub bundled_tools: Vec<BundledTool>,
    pub assets: Vec<Asset>,            // L2 — demand-paged on use
    pub agent: Option<AgentManifest>,  // present iff kind == SkillAgent
    pub signature: SignatureBlock,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind { Skill, SkillAgent }

#[derive(Clone, Serialize, Deserialize)]
pub struct BundledTool {
    pub name: String,
    pub description: String,
    pub input_schema: String,          // JSON schema
    pub synapse_primitive: String,     // the deterministic executor it binds to
    pub required_caps: Vec<CapabilityRequest>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Asset { pub name: String, pub store_ref: StoreKey, pub bytes: u32 }

#[derive(Clone, Serialize, Deserialize)]
pub struct SignatureBlock {
    pub algo: SigAlgo,                  // Ed25519 to start
    pub key_id: String,                // which trusted key
    pub content_hash: [u8; 32],        // hash of package contents
    pub sig: Vec<u8>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SigAlgo { Ed25519 }

/// Produced at install; the authoritative record of what was APPROVED. Not shipped in the package.
#[derive(Clone, Serialize, Deserialize)]
pub struct InstallRecord {
    pub skill: SkillId,
    pub installed_ticks: Ticks,
    pub granted_capabilities: Vec<CapabilityRequest>, // the approved SUBSET of requested
    pub approved_by: String,           // the user who consented at the shell
    pub source: InstallSource,
    pub verified: bool,                // signature/hash checked and matched
    pub key_id: String,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum InstallSource { BootModule { name: String }, Store { key: StoreKey } }
```

### Example — a signed skill-agent package (JSON, trimmed)

```json
{
  "schema_version": 1,
  "id": 700,
  "name": "pdf-filler",
  "version": "0.3.1",
  "description": "Fills PDF forms from a data file. Use when the task mentions filling or completing a PDF.",
  "kind": "skill_agent",
  "requested_capabilities": [
    { "domain": "fs", "rights": ["READ","WRITE"], "scope": { "t": "path", "v": "/work/**" } },
    { "domain": "inference", "rights": ["EXEC"], "scope": { "t": "any" } }
  ],
  "body_ref": "skills/700/body.md",
  "bundled_tools": [
    { "name": "pdf_fill", "description": "Fill a PDF form's fields from JSON.", "input_schema": "{\"type\":\"object\",\"properties\":{\"pdf\":{\"type\":\"string\"},\"data\":{\"type\":\"string\"}}}", "synapse_primitive": "prim.pdf.fill", "required_caps": [ { "domain": "fs", "rights": ["READ","WRITE"], "scope": { "t": "path", "v": "/work/**" } } ] }
  ],
  "assets": [ { "name": "field-map-ref", "store_ref": "skills/700/refs/fields.md", "bytes": 4096 } ],
  "agent": {
    "schema_version": 1, "id": 3001, "name": "pdf-filler-agent", "version": "0.3.1",
    "kind": "skill_agent", "description": "Delegate PDF form-filling here.",
    "system_prompt": "You fill PDF forms accurately from the provided data. Never invent field values.",
    "toolset": ["read", "write", "pdf_fill", "emit_result"],
    "capabilities": [ { "domain": "fs", "rights": ["READ","WRITE"], "scope": { "t": "path", "v": "/work/**" } } ],
    "skills": [], "sampling": { "temperature": 0.0, "top_p": 1.0, "seed": 7, "max_output_tokens": 512 },
    "budgets": { "max_turns": 12, "max_context_tokens": 4096, "compact_threshold": 3500, "max_tool_calls": 32, "max_subagents": 0, "max_depth": 0, "max_wall_ticks": 0 },
    "summary": { "max_tokens": 256, "style": "terse" },
    "origin": { "kind": "installed", "skill": 700 }
  },
  "signature": { "algo": "ed25519", "key_id": "chitti-registry-key-1", "content_hash": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "sig": [] }
}
```

### Example — the resulting `InstallRecord` (JSON)

```json
{
  "skill": 700,
  "installed_ticks": 20500,
  "granted_capabilities": [
    { "domain": "fs", "rights": ["READ","WRITE"], "scope": { "t": "path", "v": "/work/**" } }
  ],
  "approved_by": "vinoth",
  "source": { "kind": "boot_module", "name": "pdf-filler-0.3.1.skill" },
  "verified": true,
  "key_id": "chitti-registry-key-1"
}
```

Here the package also requested an `inference` capability, but the user approved only `fs` at the prompt — so `granted_capabilities` omits it, and the skill-agent's `pdf_fill` runs while any inference-requiring behavior it tries is denied at Synapse and audited. That gap between requested and granted is the whole point.

---

## Part 4 — How they interlock

- `Session.agent.manifest_id` → an `AgentManifest`. `Session.capabilities` (live) ⊆ that manifest's `capabilities` (requested).
- `SubagentRecord.manifest_id` → an `AgentManifest` (builtin, or an installed skill-agent whose `origin` names its `SkillId`). Its `effective_caps` = `min(manifest.capabilities, parent live caps, install grant if from a skill)`.
- `Session.skills_in_scope[*].skill` → a `SkillManifest` in the registry; `loaded` tracks progressive disclosure (L0 → L1 → L2).
- `Provenance` is shared by every `Message` and by skill-carried instructions (`SkillInstalled(id)`), so the Phase E gate and Phase G bounding read the same tag.
- `CapabilityRequest` is the common currency: manifests and skills declare it, the install prompt renders it, and the cap system mints matching `Capability` tokens from it.

### Design notes

- **Serialization:** `postcard` for the store (compact, no_std, fast); `serde_json` (needs `alloc`) for debug dumps and the shapes above. Keep `schema_version` on every top-level struct; bump it and provide a migration when a field's meaning changes.
- **Determinism / resume:** a session resumes from `seed` + `messages`; the KV cache is **recomputed**, not serialized (too large, cheaper to rebuild). Same seed + same messages ⇒ identical continuation, which is also how the resume test asserts correctness.
- **Compaction representation:** a compacted message keeps its metadata in `messages` with `resident: false` and a `store_ref`; the live summary lives in `ContextState.compactions`. Recall pulls the full text back from `store_ref` via `recall_index` on demand.
- **Isolation in the schema:** `SubagentRecord` has a `summary` but no transcript field — the type system makes the leak impossible to represent, which is the cheapest enforcement you can get.
- **Floats:** `f32` in `Sampling` serializes fine, but if you want byte-identical postcard across arches, consider fixed-point (e.g. temperature as `u16` milli-units). Flag for your x86_64/aarch64 parity tests.

*These three types are the contract. Pin them, version them, and every phase reads and writes the same shapes.*
