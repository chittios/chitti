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

    // A skill-agent package registers a dispatchable role bounded by the grant.
    if let Some(role) = &pkg.manifest.agent {
        agent_skill::register(role.clone(), granted.clone());
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
