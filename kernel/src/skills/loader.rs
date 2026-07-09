//! Progressive disclosure ([`CHITTI_AGENTIC_HANDOFF.md`] Phase F): load a
//! skill's body (L1) into context only when a task matches, and pull a
//! reference asset (L2) only when it is actually used. The tiers are tracked on
//! the session's `skills_in_scope` so the orchestrator knows what it has paid
//! to load.

use crate::agent::types::*;
use crate::skills::index;
use alloc::string::String;

/// Note a skill's L0 metadata as in-scope for this session (no body load).
pub fn ensure_metadata(session: &mut Session, id: SkillId) {
    if !session.skills_in_scope.iter().any(|s| s.skill == id) {
        session.skills_in_scope.push(SkillScope { skill: id, loaded: LoadTier::Metadata });
    }
}

/// The tier a skill is currently loaded at in this session (default L0 once
/// registered; `None` if not in scope at all).
pub fn tier(session: &Session, id: SkillId) -> Option<LoadTier> {
    session.skills_in_scope.iter().find(|s| s.skill == id).map(|s| s.loaded)
}

/// Load a skill's instruction body (L1) into the session context — only call
/// this on a task match. The body enters as a message tagged
/// `SkillInstalled(id)`: trusted to *steer* the agent, but bounded by the
/// install grant at the capability layer. Returns the body text, or `None` if
/// the skill / its body is missing. Idempotent (won't double-load).
pub fn load_body(session: &mut Session, id: SkillId, now: Ticks) -> Option<String> {
    if tier(session, id) == Some(LoadTier::Body) || tier(session, id) == Some(LoadTier::Full) {
        // Already loaded.
        let key = index::get(id)?.body_ref;
        return crate::synapse::fs::read(&key.0).map(|b| String::from_utf8_lossy(&b).into_owned());
    }
    let manifest = index::get(id)?;
    let bytes = crate::synapse::fs::read(&manifest.body_ref.0)?;
    let body = String::from_utf8_lossy(&bytes).into_owned();
    session.push_message(
        Role::System,
        alloc::format!("[skill:{}] {}", manifest.name, body),
        Provenance::SkillInstalled(id),
        now,
    );
    ensure_metadata(session, id);
    if let Some(s) = session.skills_in_scope.iter_mut().find(|s| s.skill == id) {
        s.loaded = LoadTier::Body;
    }
    crate::ktrace::log_fmt(format_args!(
        "skills.loader: L1 body of '{}' (id {}) loaded into session {} on match",
        manifest.name, id.0, session.id.0
    ));
    Some(body)
}

/// Demand-page a skill's L2 reference asset by name — pulled only when used,
/// never up front. Returns the asset bytes, or `None` if unknown. Marks the
/// skill Full in scope.
pub fn load_asset(session: &mut Session, id: SkillId, asset_name: &str) -> Option<alloc::vec::Vec<u8>> {
    let manifest = index::get(id)?;
    let asset = manifest.assets.iter().find(|a| a.name == asset_name)?;
    let bytes = crate::synapse::fs::read(&asset.store_ref.0)?;
    if let Some(s) = session.skills_in_scope.iter_mut().find(|s| s.skill == id) {
        s.loaded = LoadTier::Full;
    }
    crate::ktrace::log_fmt(format_args!(
        "skills.loader: L2 asset '{}' of skill {} demand-paged ({} bytes)",
        asset_name, id.0, bytes.len()
    ));
    Some(bytes)
}

/// **Skill invoke**: load L1 body by name (and optional L2 `asset`). Returns a
/// model-facing summary. Idempotent body load.
pub fn invoke(
    session: &mut Session,
    name: &str,
    asset: Option<&str>,
    now: Ticks,
) -> Result<alloc::string::String, alloc::string::String> {
    use alloc::format;
    use alloc::string::ToString;
    let meta = index::by_name(name).ok_or_else(|| format!("no installed skill named '{name}'"))?;
    let body = load_body(session, meta.id, now).ok_or_else(|| format!("skill '{name}' has no body"))?;
    let mut out = format!(
        "loaded skill '{name}' (L1, {} bytes). Instructions are now in context.\n\n{}",
        body.len(),
        body
    );
    if let Some(a) = asset {
        if a.is_empty() {
            return Ok(out);
        }
        match load_asset(session, meta.id, a) {
            Some(bytes) => {
                let text = alloc::string::String::from_utf8_lossy(&bytes);
                out.push_str(&format!("\n\n[L2 asset '{a}']\n{text}"));
            }
            None => {
                // List available asset names to help the model.
                let names: alloc::vec::Vec<_> = index::get(meta.id)
                    .map(|m| m.assets.iter().map(|x| x.name.clone()).collect())
                    .unwrap_or_default();
                return Err(format!(
                    "skill '{name}' has no L2 asset '{a}' (available: [{}])",
                    names.join(", ")
                ));
            }
        }
    }
    Ok(out)
}
