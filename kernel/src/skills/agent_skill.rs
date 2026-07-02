//! Installable **skill-agents** (`CHITTI_AGENTIC_HANDOFF.md` Phase G): a skill
//! package whose manifest carries an `AgentManifest` registers a new
//! dispatchable sub-agent role. Its effective capabilities at dispatch are the
//! intersection `min(role.capabilities, install grant, parent caps)` — never
//! wider than any of the three. This is where "an installed skill is bounded by
//! its install-time grant, forever" meets sub-agent attenuation.

use crate::agent::agent_loop::{StepSource, ToolDispatch};
use crate::agent::subagent::{self, DispatchError, SubagentOutcome};
use crate::agent::types::*;
use crate::mm::Locked;
use alloc::vec::Vec;

/// A registered skill-agent role + the install grant that bounds it.
#[derive(Clone)]
struct Role {
    skill: SkillId,
    manifest: AgentManifest,
    install_grant: Vec<CapabilityRequest>,
}

static ROLES: Locked<Vec<Role>> = Locked::new(Vec::new());

/// Register a skill-agent role bounded by its install grant (called by the
/// install flow). Idempotent by agent id.
pub fn register(manifest: AgentManifest, install_grant: Vec<CapabilityRequest>) {
    let skill = match manifest.origin {
        Origin::Installed { skill } => skill,
        Origin::Builtin => SkillId(0),
    };
    ROLES.with(|roles| {
        roles.retain(|r| r.manifest.id != manifest.id);
        roles.push(Role { skill, manifest: manifest.clone(), install_grant: install_grant.clone() });
    });
    crate::ktrace::log_fmt(format_args!(
        "skills.agent: registered skill-agent role '{}' (bounded by {} granted caps)",
        manifest.name,
        install_grant.len()
    ));
}

/// Remove all roles belonging to a skill (uninstall).
pub fn deregister_for_skill(skill: SkillId) {
    ROLES.with(|roles| roles.retain(|r| r.skill != skill));
}

/// Look up a registered skill-agent role by name.
pub fn by_name(name: &str) -> Option<AgentManifest> {
    ROLES.with(|roles| roles.iter().find(|r| r.manifest.name == name).map(|r| r.manifest.clone()))
}

/// The effective caps a dispatch of `name` would run with, given `parent_caps`:
/// `min(role.capabilities, install grant, parent caps)`. `None` if no such role.
pub fn effective_caps(name: &str, parent_caps: &[CapabilityRequest]) -> Option<Vec<CapabilityRequest>> {
    ROLES.with(|roles| {
        let r = roles.iter().find(|r| r.manifest.name == name)?;
        let bounded = intersect_caps(&r.manifest.capabilities, &r.install_grant);
        Some(intersect_caps(&bounded, parent_caps))
    })
}

/// Dispatch a registered skill-agent: clamp its manifest caps to
/// `min(role, install grant)` first, then hand off to the normal sub-agent
/// dispatch (which further clamps to the parent). Net effective caps are the
/// three-way intersection, and delegation/isolation rules apply unchanged.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    name: &str,
    parent_caps: &[CapabilityRequest],
    parent_depth: u8,
    max_depth: u8,
    task: &str,
    steps: &mut dyn StepSource,
    tools: &mut dyn ToolDispatch,
    core: Option<u8>,
) -> Result<SubagentOutcome, DispatchError> {
    let (mut role, grant) = ROLES.with(|roles| {
        roles
            .iter()
            .find(|r| r.manifest.name == name)
            .map(|r| (r.manifest.clone(), r.install_grant.clone()))
    })
    .ok_or(DispatchError::CapabilityRefused(CapabilityRequest::new(CapDomain::SkillManage, Rights::EXEC, Scope::Any)))?;
    // Bound the role's requested caps by the install grant BEFORE dispatch, so
    // the sub-agent can never exceed what was approved at install.
    role.capabilities = intersect_caps(&role.capabilities, &grant);
    subagent::dispatch(parent_caps, parent_depth, max_depth, role, task, steps, tools, core)
}

#[cfg(test)]
pub fn reset() {
    ROLES.with(|roles| roles.clear());
}
