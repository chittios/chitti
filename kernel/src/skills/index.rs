//! The skill **index**: L0 metadata (name + description) for every installed
//! skill, always cheap to keep in context, plus description-based matching of a
//! task to a skill. The full `SkillManifest` lives in the memory store (keyed
//! `skills/<id>/manifest`, postcard); only the L0 slice is held in memory.

use crate::agent::types::*;
use crate::mm::Locked;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// L0 metadata: what the orchestrator sees for every skill without paying to
/// load its body.
#[derive(Clone)]
pub struct SkillMeta {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub kind: SkillKind,
}

static INDEX: Locked<Vec<SkillMeta>> = Locked::new(Vec::new());

fn manifest_key(id: SkillId) -> String {
    format!("skills/{}/manifest", id.0)
}

/// Register a skill: persist its full manifest to the store and add its L0
/// metadata to the index. Idempotent by id.
pub fn register(manifest: &SkillManifest) -> Result<(), postcard::Error> {
    let bytes = postcard::to_allocvec(manifest)?;
    crate::synapse::fs::write(&manifest_key(manifest.id), &bytes);
    INDEX.with(|idx| {
        if !idx.iter().any(|m| m.id == manifest.id) {
            idx.push(SkillMeta {
                id: manifest.id,
                name: manifest.name.clone(),
                description: manifest.description.clone(),
                kind: manifest.kind,
            });
        }
    });
    crate::ktrace::log_fmt(format_args!(
        "skills.index: registered '{}' (id {}) — L0 metadata only",
        manifest.name, manifest.id.0
    ));
    Ok(())
}

/// Remove a skill from the index and drop its stored manifest (Phase G uninstall).
pub fn deregister(id: SkillId) {
    INDEX.with(|idx| idx.retain(|m| m.id != id));
    crate::synapse::fs::delete(&manifest_key(id));
    crate::ktrace::log_fmt(format_args!("skills.index: deregistered skill {}", id.0));
}

/// The L0 metadata for every installed skill.
pub fn metadata() -> Vec<SkillMeta> {
    INDEX.with(|idx| idx.clone())
}

/// Load a skill's full manifest from the store.
pub fn get(id: SkillId) -> Option<SkillManifest> {
    let bytes = crate::synapse::fs::read(&manifest_key(id))?;
    postcard::from_bytes(&bytes).ok()
}

/// Look up a skill by name.
pub fn by_name(name: &str) -> Option<SkillMeta> {
    INDEX.with(|idx| idx.iter().find(|m| m.name == name).cloned())
}

/// Match a task to the most relevant skill by description (L0 only — no body
/// load). A skill matches if the task text contains its name or any
/// "significant" (≥5-char) word from its description. Returns the first match;
/// `None` means no skill applies (so nothing is loaded — context stays lean).
pub fn match_task(task: &str) -> Option<SkillMeta> {
    let t = task.to_ascii_lowercase();
    INDEX.with(|idx| {
        for m in idx.iter() {
            if t.contains(&m.name.to_ascii_lowercase()) {
                return Some(m.clone());
            }
            for word in m.description.to_ascii_lowercase().split(|c: char| !c.is_alphanumeric()) {
                if word.len() >= 5 && t.contains(word) {
                    return Some(m.clone());
                }
            }
        }
        None
    })
}

/// Test-only: clear the index (manifests in the store are left; harmless).
#[cfg(test)]
pub fn reset() {
    INDEX.with(|idx| idx.clear());
}

/// Human-readable one-line L0 render for prompts/inspection.
pub fn render(meta: &SkillMeta) -> String {
    format!("{} — {}", meta.name, meta.description).to_string()
}
