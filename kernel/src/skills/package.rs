//! The skill **package** format and placement. A package bundles the
//! `SkillManifest` (which describes it + requests capabilities + lists bundled
//! tools/assets) with the actual body text and asset bytes. Placing a package
//! writes the body + assets to the store, registers the bundled tools as normal
//! Synapse-backed, capability-gated tools (via `tools::provider`), and adds the
//! skill's L0 metadata to the index.
//!
//! Phase F places packages directly as **trusted** ([`place_trusted`]); Phase G
//! ([`install`](crate::skills::install)) verifies a signature and takes consent
//! first, then grants only the approved capability subset.

use crate::agent::types::*;
use crate::tools::provider;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// A deliverable skill package: manifest + body + asset payloads. This is what
/// a boot module carries (postcard-serialized) for Phase G install.
#[derive(Clone, Serialize, Deserialize)]
pub struct SkillPackage {
    pub manifest: SkillManifest,
    pub body: String,
    /// The agent's SOUL.md persona text (a SkillAgent package). Placed into
    /// `/agent/<id>/SOUL.md` on install so it seeds the agent's system prompt.
    pub soul: Option<String>,
    /// `(doc name, markdown)` — matched to `manifest.skill_docs` by name; placed
    /// into `/agent/<id>/skills/<name>`.
    pub skill_docs: Vec<(String, String)>,
    /// `(asset name, bytes)` — matched to `manifest.assets` by name on placement.
    /// ONNX models for an ML agent live here (demand-paged as L2 assets).
    pub assets: Vec<(String, Vec<u8>)>,
}

impl SkillPackage {
    /// Serialize for boot-module delivery.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }
    /// Parse a delivered package.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// The canonical bytes that are signed/verified: the manifest with its
    /// signature block zeroed, followed by the body and every asset payload.
    /// Signing over this ties the signature to the *entire* package, so any
    /// tampering (body, a tool, a requested cap) invalidates it.
    fn signing_message(&self) -> Vec<u8> {
        let mut m = self.manifest.clone();
        m.signature = SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: m.signature.key_id.clone(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        };
        let mut out = postcard::to_allocvec(&m).unwrap_or_default();
        out.extend_from_slice(self.body.as_bytes());
        // Cover the soul + skill docs so a tampered persona/procedure invalidates
        // the signature just like a tampered body or requested-cap would.
        if let Some(soul) = &self.soul {
            out.extend_from_slice(soul.as_bytes());
        }
        for (name, md) in &self.skill_docs {
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(md.as_bytes());
        }
        for (name, bytes) in &self.assets {
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(bytes);
        }
        out
    }

    /// Place the agent-home artifacts a SkillAgent package carries: its SOUL.md
    /// persona and any markdown procedure docs, under `/agent/<id>/`. Called by
    /// [`crate::skills::install`] when `manifest.agent` is present, *before* the
    /// agent's home is first `ensure`d, so a packaged soul is never clobbered by
    /// the default persona.
    pub fn place_agent_home(&self, agent_id: AgentId) {
        let home = crate::agent::home::path(agent_id.0);
        if let Some(soul) = &self.soul {
            crate::synapse::fs::write(&alloc::format!("{home}/SOUL.md"), soul.as_bytes());
        }
        for (name, md) in &self.skill_docs {
            crate::synapse::fs::write(&alloc::format!("{home}/skills/{name}"), md.as_bytes());
        }
        // Bundled assets (e.g. the Doc agent's HTML + logo) land in the agent's
        // own install folder, which it then reads at runtime with read-only
        // authority scoped to its home.
        for (name, bytes) in &self.assets {
            crate::synapse::fs::write(&alloc::format!("{home}/assets/{name}"), bytes);
        }
        // Seed the memory area so the agent's fs tools have a place to write.
        let keep = alloc::format!("{home}/memory/.keep");
        if !crate::synapse::fs::exists(&keep) {
            crate::synapse::fs::write(&keep, b"");
        }
    }

    /// Sign the package with the registry key (fills `signature`). Used to
    /// produce signed sample packages.
    pub fn sign(&mut self) {
        let msg = self.signing_message();
        self.manifest.signature = SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: crate::skills::crypto::hash32(&msg),
            sig: crate::skills::crypto::sign(&msg),
        };
    }

    /// Verify the package's signature + content hash against the trust store,
    /// dispatching on the signature algorithm: `Ed25519` = the self-contained
    /// keyed-MAC scheme (local dev/boot packages); `P256` = real ECDSA against
    /// the baked publisher trust store (public-registry packages). Returns false
    /// for an unsigned, tampered, or untrusted-key package.
    pub fn verify(&self) -> bool {
        let msg = self.signing_message();
        let sb = &self.manifest.signature;
        if sb.content_hash != crate::skills::crypto::hash32(&msg) {
            return false;
        }
        match sb.algo {
            SigAlgo::Ed25519 => crate::skills::crypto::verify(&sb.key_id, &msg, &sb.sig),
            SigAlgo::P256 => crate::skills::crypto::verify_p256(&sb.key_id, &msg, &sb.sig),
        }
    }

    /// Write the package's body + assets to the store and register its bundled
    /// tools + L0 metadata. Does NOT verify a signature or gate capabilities —
    /// only safe for **boot/bundled** paths and for [`crate::skills::install`]
    /// *after* `verify()`. Not public so agent-facing code cannot place an
    /// unsigned package by calling this directly.
    pub(crate) fn place_trusted(&self) -> Result<(), postcard::Error> {
        // L1 body.
        crate::synapse::fs::write(&self.manifest.body_ref.0, self.body.as_bytes());
        // L2 assets: match payloads to declared asset store_refs by name.
        for (name, bytes) in &self.assets {
            if let Some(a) = self.manifest.assets.iter().find(|a| &a.name == name) {
                crate::synapse::fs::write(&a.store_ref.0, bytes);
            }
        }
        // Register bundled tools as normal Synapse-backed, gated tools.
        register_bundled_tools(&self.manifest);
        // L0 metadata into the index.
        crate::skills::index::register(&self.manifest)?;
        crate::ktrace::log_fmt(format_args!(
            "skills.package: placed '{}' (trusted) — body + {} asset(s) + {} tool(s)",
            self.manifest.name,
            self.manifest.assets.len(),
            self.manifest.bundled_tools.len()
        ));
        Ok(())
    }
}

/// Register each of a skill's bundled tools with the tool registry. Identity
/// arg-mapping over the tool's declared required keys; every call still routes
/// through Synapse with a capability check.
pub fn register_bundled_tools(manifest: &SkillManifest) {
    for bt in &manifest.bundled_tools {
        let required = extract_required(&bt.input_schema);
        let arg_pairs: Vec<(&str, &str)> = required.iter().map(|k| (k.as_str(), k.as_str())).collect();
        let req_refs: Vec<&str> = required.iter().map(|s| s.as_str()).collect();
        let def = provider::synapse_tool(&bt.name, &bt.description, &bt.input_schema, &bt.synapse_primitive, &arg_pairs, &req_refs);
        crate::tools::registry::register(def);
    }
}

/// Extract the `"required":[ ... ]` string keys from a JSON schema (tolerant,
/// no full JSON parse needed for these controlled schemas).
fn extract_required(schema: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(i) = schema.find("\"required\"") {
        if let Some(open) = schema[i..].find('[') {
            let start = i + open + 1;
            if let Some(close) = schema[start..].find(']') {
                for part in schema[start..start + close].split(',') {
                    let k = part.trim().trim_matches('"').trim();
                    if !k.is_empty() {
                        out.push(k.to_string());
                    }
                }
            }
        }
    }
    out
}

/// The Phase-F/G sample **plain skill**: a note summarizer/searcher. Its bundled
/// `note_search` tool binds to the existing `mem_fs_search` primitive, so it
/// executes only through Synapse with a capability check. `content_hash`/`sig`
/// are filled by the signer for Phase G; here they are placeholders.
pub fn sample_note_summarizer(id: SkillId) -> SkillPackage {
    let body = "To summarize notes: (1) use note_search to find files mentioning the topic, \
                (2) read each match, (3) write a concise summary file. Keep it under 5 lines."
        .to_string();
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", id.0));
    let asset_ref = StoreKey(alloc::format!("skills/{}/refs/style.md", id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id,
        name: "note-summarizer".to_string(),
        version: "1.0.0".to_string(),
        description: "Summarize and search note files. Use when the task mentions notes or summaries.".to_string(),
        kind: SkillKind::Skill,
        requested_capabilities: vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE | Rights::LIST, Scope::Any)],
        body_ref,
        bundled_tools: vec![BundledTool {
            name: "note_search".to_string(),
            description: "Find note files whose contents contain a query.".to_string(),
            input_schema: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#.to_string(),
            synapse_primitive: "mem_fs_search".to_string(),
            required_caps: vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Any)],
        }],
        assets: vec![Asset { name: "style-guide".to_string(), store_ref: asset_ref, bytes: 48 }],
        agent: None,
        soul_ref: None,
        skill_docs: Vec::new(),
        signature: SignatureBlock { algo: SigAlgo::Ed25519, key_id: "chitti-registry-key-1".to_string(), content_hash: [0u8; 32], sig: Vec::new() },
    };
    SkillPackage {
        manifest,
        body,
        soul: None,
        skill_docs: Vec::new(),
        assets: vec![("style-guide".to_string(), b"Style: terse, factual, no fluff.".to_vec())],
    }
}

/// The Phase-G sample **skill-agent**: an installable agent role (the schema
/// Part 3 `pdf-filler` example, adapted to the memory FS). Its bundled tool
/// `note_write` binds to `mem_fs_write`. Requests Fs READ|WRITE; a real install
/// may approve only a subset. Signed via [`SkillPackage::sign`] by the caller.
pub fn sample_report_agent(skill_id: SkillId, agent_id: AgentId) -> SkillPackage {
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", skill_id.0));
    let agent = AgentManifest {
        schema_version: 1,
        id: agent_id,
        name: "report-writer-agent".to_string(),
        version: "1.0.0".to_string(),
        kind: AgentKind::SkillAgent,
        description: "Delegate report writing here. Use when the task mentions writing a report.".to_string(),
        system_prompt: "You write concise reports from provided facts. Never invent data.".to_string(),
        toolset: vec!["read".to_string(), "write".to_string(), "emit_result".to_string()],
        capabilities: vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE, Scope::Any)],
        skills: Vec::new(),
        sampling: Sampling::deterministic(7),
        budgets: Budgets {
            max_turns: 12,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 128,
            max_subagents: 0,
            max_depth: 0,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Installed { skill: skill_id },
        mcp_servers: alloc::vec::Vec::new(),
    };
    let soul_ref = StoreKey(alloc::format!("/agent/{}/SOUL.md", agent_id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id: skill_id,
        name: "report-writer".to_string(),
        version: "1.0.0".to_string(),
        description: "Write reports from facts. Use when the task mentions writing a report.".to_string(),
        kind: SkillKind::SkillAgent,
        requested_capabilities: vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE, Scope::Any)],
        body_ref,
        bundled_tools: Vec::new(),
        assets: Vec::new(),
        agent: Some(agent),
        soul_ref: Some(soul_ref),
        skill_docs: Vec::new(),
        signature: SignatureBlock { algo: SigAlgo::Ed25519, key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(), content_hash: [0u8; 32], sig: Vec::new() },
    };
    SkillPackage {
        manifest,
        body: "Write a clear, factual report from the given facts.".to_string(),
        soul: Some("You are the report-writer agent. You turn facts into concise, factual reports and never invent data.".to_string()),
        skill_docs: Vec::new(),
        assets: Vec::new(),
    }
}
