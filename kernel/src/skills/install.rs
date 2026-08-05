//! The permissioned **install** flow (`CHITTI_AGENTIC_HANDOFF.md` Phase G):
//! verify a package's Ed25519 signature against the trust store, present its
//! requested capabilities for consent, and on approval register it granting
//! only the approved subset. Produces an [`InstallRecord`] — the authoritative
//! record of what was approved. A skill-agent package additionally registers a
//! dispatchable role bounded by that grant.
//!
//! Invariant: **a skill is bounded by its install-time grant, forever.** The
//! grant is the intersection of what the user approved with what the package
//! requested; a skill's instructions (provenance `SkillInstalled`) can steer
//! the agent but never exceed the grant — every effect still hits the Synapse
//! capability gate.

use crate::agent::types::*;
use crate::mm::Locked;
use crate::skills::package::SkillPackage;
use crate::skills::{agent_skill, index};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    /// Signature/hash did not verify against a trusted key — unsigned, tampered,
    /// or signed by an untrusted key.
    VerificationFailed,
    /// An approved capability was never requested by the package (cannot grant
    /// authority the package didn't ask for).
    ApprovedCapNotRequested,
}

/// The install registry: what has been installed + its approved grant.
static RECORDS: Locked<Vec<InstallRecord>> = Locked::new(Vec::new());

fn record_key(id: SkillId) -> String {
    format!("skills/{}/install", id.0)
}

/// The approved capability grant for an installed skill, or `None`.
pub fn granted_caps(id: SkillId) -> Option<Vec<CapabilityRequest>> {
    RECORDS.with(|r| r.iter().find(|rec| rec.skill == id).map(|rec| rec.granted_capabilities.clone()))
}

/// Whether a skill is currently installed.
pub fn is_installed(id: SkillId) -> bool {
    RECORDS.with(|r| r.iter().any(|rec| rec.skill == id))
}

/// Read an install record back **from the store**.
///
/// [`RECORDS`] is only ever populated by [`install`], so after a reboot it is empty
/// even though `skills/<id>/install` is still on disk. Everything installed from
/// compiled-in data is re-installed at boot and so refills it; a package that lives
/// only in the store has no such path, and without this its grant would be lost —
/// which matters because **the recorded grant is the ceiling on a re-install**
/// (invariant 5: a skill is bounded by its install-time grant, forever). A manifest
/// that has been edited to ask for more must not get it.
pub fn load_record(id: SkillId) -> Option<InstallRecord> {
    let bytes = crate::synapse::fs::read(&record_key(id))?;
    postcard::from_bytes::<InstallRecord>(&bytes).ok()
}

/// Remember a record read from the store, so `granted_caps`/`is_installed` answer
/// for it without a second install.
pub fn adopt_record(record: InstallRecord) {
    RECORDS.with(|r| {
        r.retain(|rec| rec.skill != record.skill);
        r.push(record);
    });
}

/// Install a package: verify → consent (the caller passes the human-`approved`
/// subset) → register granting only `approved ∩ requested`. Refuses an
/// unsigned/tampered/untrusted package. `now`/`approved_by`/`source` describe
/// the install for the record + audit.
pub fn install(
    pkg: &SkillPackage,
    approved: &[CapabilityRequest],
    approved_by: &str,
    source: InstallSource,
    now: Ticks,
) -> Result<InstallRecord, InstallError> {
    // Gate 1: signature/hash verification against the trust store.
    if !pkg.verify() {
        crate::ktrace::log_fmt(format_args!(
            "skills.install: REFUSED '{}' — signature/hash verification failed",
            pkg.manifest.name
        ));
        return Err(InstallError::VerificationFailed);
    }
    // Gate 2: the approved set may not exceed what was requested. Clamp to the
    // intersection so consent can only ever narrow.
    for a in approved {
        if !pkg.manifest.requested_capabilities.iter().any(|r| r.contains(a)) {
            crate::ktrace::log_fmt(format_args!(
                "skills.install: REFUSED '{}' — approved a capability the package never requested",
                pkg.manifest.name
            ));
            return Err(InstallError::ApprovedCapNotRequested);
        }
    }
    let granted = intersect_caps(approved, &pkg.manifest.requested_capabilities);

    // Register: place body/assets + bundled tools + L0 metadata, then record
    // the grant. (Bundled tools are gated by caps at call time regardless.)
    let _ = pkg.place_trusted();

    // A skill-agent package registers a dispatchable role bounded by the grant,
    // and lands its SOUL.md + procedure docs into the agent's home *before* the
    // home is first ensured, so the packaged persona is never clobbered by the
    // default one.
    if let Some(role) = &pkg.manifest.agent {
        pkg.place_agent_home(role.id);
        // Baseline sandbox: every installed agent gets read/write/list/delete
        // over its OWN home (`/agent/<id>/**`) and nothing else, unless its
        // manifest explicitly requested (and the human approved) a broader Fs
        // scope. Home access is the floor, not a privilege — the equivalent of
        // a process's own working directory — so it is not part of the consent
        // prompt; anything wider IS a requested capability shown at install.
        let effective = with_home_sandbox(&granted, role.id, role.kind);
        agent_skill::register(role.clone(), effective);
    }

    let record = InstallRecord {
        skill: pkg.manifest.id,
        installed_ticks: now,
        granted_capabilities: granted.clone(),
        approved_by: approved_by.to_string(),
        source,
        verified: true,
        key_id: pkg.manifest.signature.key_id.clone(),
    };
    if let Ok(bytes) = postcard::to_allocvec(&record) {
        crate::synapse::fs::write(&record_key(pkg.manifest.id), &bytes);
    }
    RECORDS.with(|r| {
        r.retain(|rec| rec.skill != pkg.manifest.id);
        r.push(record.clone());
    });
    crate::ktrace::log_fmt(format_args!(
        "skills.install: '{}' verified + installed by {}; granted {} of {} requested caps",
        pkg.manifest.name,
        approved_by,
        granted.len(),
        pkg.manifest.requested_capabilities.len()
    ));
    Ok(record)
}

/// Uninstall a skill: drop its install record + grant, de-register its L0
/// metadata (so it no longer loads/matches), and remove any skill-agent role.
pub fn uninstall(id: SkillId) {
    RECORDS.with(|r| r.retain(|rec| rec.skill != id));
    crate::synapse::fs::delete(&record_key(id));
    agent_skill::deregister_for_skill(id);
    index::deregister(id);
    crate::ktrace::log_fmt(format_args!("skills.uninstall: skill {} revoked + de-registered", id.0));
}

/// Add the per-agent home sandbox (`Fs READ|WRITE|LIST|DELETE @ /agent/<id>/**`)
/// to a grant, unless the agent is the orchestrator (the root, full-FS).
///
/// The home floor is **always** injected for non-orchestrator agents, even when
/// the human also approved other Fs scopes. A broader approved grant (e.g.
/// `Fs @ Any`) still covers everything via scope OR-semantics; a narrow
/// non-home grant no longer *replaces* the home floor (which used to leave an
/// agent without its own working directory if the only approved Fs was elsewhere).
pub fn with_home_sandbox(granted: &[CapabilityRequest], id: AgentId, kind: AgentKind) -> Vec<CapabilityRequest> {
    let mut out = granted.to_vec();
    // The orchestrator (shell agent) is the root — never sandboxed.
    if kind == AgentKind::Orchestrator {
        return out;
    }
    let home = crate::agent::home::path(id.0);
    let home_scope = Scope::Path(alloc::format!("{home}/**"));
    let home_rights = Rights::READ | Rights::WRITE | Rights::LIST | Rights::DELETE;
    // Skip only when an existing grant already covers the full home floor
    // (rights + path). Scope::Any with full rights counts as covering.
    let already = out.iter().any(|c| {
        c.domain == CapDomain::Fs
            && c.rights.contains(home_rights)
            && c.scope.covers(&home_scope)
    });
    if !already {
        out.push(CapabilityRequest::new(CapDomain::Fs, home_rights, home_scope));
    }
    out
}

/// Render the consent prompt lines for a package's requested capabilities.
pub fn consent_prompt(pkg: &SkillPackage) -> Vec<String> {
    pkg.manifest
        .requested_capabilities
        .iter()
        .map(crate::agent::manifest::render_cap)
        .collect()
}

#[cfg(test)]
pub fn reset() {
    RECORDS.with(|r| r.clear());
}
