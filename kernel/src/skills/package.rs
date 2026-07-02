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
    /// `(asset name, bytes)` — matched to `manifest.assets` by name on placement.
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

    /// Write the package's body + assets to the store and register its bundled
    /// tools + L0 metadata. Does NOT verify a signature or gate capabilities —
    /// Phase F treats placed skills as trusted (Phase G adds verify + consent).
    pub fn place_trusted(&self) -> Result<(), postcard::Error> {
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
        signature: SignatureBlock { algo: SigAlgo::Ed25519, key_id: "chitti-registry-key-1".to_string(), content_hash: [0u8; 32], sig: Vec::new() },
    };
    SkillPackage {
        manifest,
        body,
        assets: vec![("style-guide".to_string(), b"Style: terse, factual, no fluff.".to_vec())],
    }
}
