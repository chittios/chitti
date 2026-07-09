//! **Skills** — portable, permissioned packages of procedural knowledge
//! (`CHITTI_AGENTIC_HANDOFF.md` Phase F/G). A skill is a manifest (name,
//! description, requested caps, signature) + an instruction body + optional
//! bundled tools + optional reference assets, stored in the memory store and
//! surfaced by **progressive disclosure**:
//!
//! * **L0** metadata (name + description) — always cheap, kept in the index.
//! * **L1** instruction body — loaded into context only when a task matches.
//! * **L2** bundled refs/tools — demand-paged / executed on use.
//!
//! * [`index`] — the L0 registry + description-based task matching.
//! * [`loader`] — progressive disclosure (load body / asset on demand).
//! * [`package`] — the package format + placing a skill in the store; Phase F
//!   places skills directly as trusted, Phase G ([`install`]) verifies a
//!   signature and takes consent first.
//! * [`install`] — the permissioned install flow (Phase G).

pub mod agent_skill;
pub mod bundled;
pub mod crypto;
pub mod index;
pub mod install;
pub mod loader;
pub mod package;
pub mod registry_client;

#[cfg(test)]
mod tests {
    use crate::agent::agent_loop::ToolDispatch;
    use crate::agent::rule_steps::{args, tool};
    use crate::agent::types::*;
    use crate::agent::{manifest, orchestrator};
    use crate::skills::{index, loader, package};
    use crate::synapse::audit;
    use alloc::vec;

    fn place_sample() -> SkillId {
        let id = next_skill_id();
        package::sample_note_summarizer(id).place_trusted().expect("place");
        id
    }

    /// (a) Only L0 metadata is present until a matching task triggers the L1
    /// body load; (c) an unrelated task loads nothing.
    #[test_case]
    fn progressive_disclosure_loads_body_only_on_match() {
        index::reset(); // isolate: only this test's skill in the index
        let id = place_sample();
        let m = manifest::orchestrator_manifest();
        let mut session = Session::new(&m, 1, vec![], 0);
        loader::ensure_metadata(&mut session, id);
        // L0 only: no skill body message yet.
        assert_eq!(loader::tier(&session, id), Some(LoadTier::Metadata));
        assert!(!session.messages.iter().any(|m| matches!(m.provenance, Provenance::SkillInstalled(_))));

        // (c) An unrelated task matches nothing → no load.
        assert!(index::match_task("calculate orbital trajectory xyzzy").is_none());

        // A matching task triggers L1.
        let hit = index::match_task("please summarize my project notes").expect("match");
        assert_eq!(hit.id, id);
        loader::load_body(&mut session, id, 1).expect("body");
        assert_eq!(loader::tier(&session, id), Some(LoadTier::Body));
        // The body is now in context, tagged SkillInstalled (trusted to steer).
        let body_msg = session.messages.iter().find(|m| matches!(m.provenance, Provenance::SkillInstalled(_))).unwrap();
        assert!(body_msg.content.contains("note_search"));
    }

    /// (b) A skill's bundled tool executes only through Synapse with a
    /// capability check: an agent lacking the underlying cap is denied; one
    /// holding it executes.
    #[test_case]
    fn bundled_tool_is_capability_gated() {
        place_sample();
        // The bundled tool is now a registered tool.
        assert!(crate::tools::registry::get("note_search").is_some());
        crate::synapse::fs::write("f_note1", b"topic: rockets are great");

        // An agent with only Console cap cannot search (no MEM_FS_SEARCH).
        let mut console_only = manifest::orchestrator_manifest();
        console_only.capabilities = vec![CapabilityRequest::new(CapDomain::Console, Rights::WRITE, Scope::Any)];
        let mut o1 = orchestrator::Orchestrator::spawn(console_only, 2);
        let mut r1 = crate::tools::Router::new();
        let denied = r1.call(&mut o1.session, o1.caller, &tool("note_search", args(&[("query", "rockets")])));
        assert!(denied.is_error && denied.result.contains("denied"), "bundled tool must be cap-gated: {}", denied.result);

        // An agent with Fs READ (→ MEM_FS_SEARCH) executes it, through Synapse.
        let mut o2 = orchestrator::Orchestrator::spawn(manifest::orchestrator_manifest(), 3);
        let mut r2 = crate::tools::Router::new();
        let before = audit::len();
        let ok = r2.call(&mut o2.session, o2.caller, &tool("note_search", args(&[("query", "rockets")])));
        assert!(!ok.is_error, "search should execute: {}", ok.result);
        assert!(ok.result.contains("f_note1"), "search found the note: {}", ok.result);
        assert_eq!(audit::len(), before + 1, "bundled tool call was audited by Synapse");
    }

    /// (d) An L2 asset is pulled on demand, not up front.
    #[test_case]
    fn l2_asset_demand_paged() {
        let id = place_sample();
        let m = manifest::orchestrator_manifest();
        let mut session = Session::new(&m, 4, vec![], 0);
        loader::load_body(&mut session, id, 1).expect("body");
        // Body loaded (L1) but the asset is not yet pulled.
        assert_eq!(loader::tier(&session, id), Some(LoadTier::Body));
        // Demand-page the asset only now.
        let bytes = loader::load_asset(&mut session, id, "style-guide").expect("asset");
        assert!(alloc::string::String::from_utf8_lossy(&bytes).contains("terse"));
        assert_eq!(loader::tier(&session, id), Some(LoadTier::Full));
    }

    // ---- Phase G: permissioned install ------------------------------------
    use crate::skills::{agent_skill, install};

    /// (a) An unsigned or tampered package is refused at install; a properly
    /// signed one installs.
    #[test_case]
    fn unsigned_or_tampered_package_refused() {
        install::reset();
        let id = next_skill_id();
        // Unsigned (empty signature) → refused.
        let mut pkg = package::sample_note_summarizer(id);
        assert!(!pkg.verify(), "unsigned package must not verify");
        let src = InstallSource::BootModule { name: "note-summarizer.skill".into() };
        assert_eq!(
            install::install(&pkg, &pkg.manifest.requested_capabilities.clone(), "vinoth", src.clone(), 1).unwrap_err(),
            install::InstallError::VerificationFailed
        );
        // Sign, then tamper → refused.
        pkg.sign();
        assert!(pkg.verify(), "freshly signed package verifies");
        let mut tampered = pkg.clone();
        tampered.body.push_str(" <injected>");
        assert!(!tampered.verify(), "tampering invalidates the signature");
        assert_eq!(
            install::install(&tampered, &tampered.manifest.requested_capabilities.clone(), "vinoth", src.clone(), 1).unwrap_err(),
            install::InstallError::VerificationFailed
        );
        // Clean signed package installs.
        assert!(install::install(&pkg, &pkg.manifest.requested_capabilities.clone(), "vinoth", src, 1).is_ok());
    }

    /// (b) Install grants only the approved subset of the requested caps.
    #[test_case]
    fn install_grants_only_approved_subset() {
        install::reset();
        let id = next_skill_id();
        let mut pkg = package::sample_note_summarizer(id); // requests Fs READ|WRITE|LIST
        pkg.sign();
        assert!(install::consent_prompt(&pkg).len() >= 1, "consent prompt lists requested caps");
        // Approve only READ.
        let approved = alloc::vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Any)];
        let rec = install::install(&pkg, &approved, "vinoth", InstallSource::BootModule { name: "n.skill".into() }, 2).unwrap();
        assert_eq!(rec.granted_capabilities.len(), 1);
        assert!(rec.granted_capabilities[0].rights.contains(Rights::READ));
        assert!(!rec.granted_capabilities[0].rights.contains(Rights::WRITE), "WRITE was not approved → not granted");
        assert!(rec.verified);
    }

    /// (c) A skill body instructing a capability it was not granted is blocked
    /// at Synapse and audited — `SkillInstalled` provenance does not bypass the
    /// grant.
    #[test_case]
    fn skill_cannot_exceed_its_grant() {
        install::reset();
        let id = next_skill_id();
        let mut pkg = package::sample_note_summarizer(id);
        pkg.sign();
        // Approve only READ (the skill body might "want" to write, but can't).
        let approved = alloc::vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Any)];
        let rec = install::install(&pkg, &approved, "vinoth", InstallSource::BootModule { name: "n.skill".into() }, 3).unwrap();
        // An agent running under exactly the install grant tries to write.
        let mut m = manifest::orchestrator_manifest();
        m.capabilities = rec.granted_capabilities.clone(); // Fs READ only
        let mut orch = orchestrator::Orchestrator::spawn(m, 9);
        let mut router = crate::tools::Router::new();
        let before = audit::len();
        let out = router.call(&mut orch.session, orch.caller, &tool("write", args(&[("path", "g_x"), ("content", "y")])));
        assert!(out.is_error && out.result.contains("denied"), "write beyond grant must be denied: {}", out.result);
        assert_eq!(audit::snapshot().last().unwrap().outcome, audit::Outcome::DeniedNoCapability);
        assert_eq!(audit::len(), before + 1, "the denial was audited");
    }

    /// (d) A dispatched skill-agent's effective caps are the intersection with
    /// the parent's — never wider.
    #[test_case]
    fn skill_agent_effective_caps_never_widen() {
        install::reset();
        agent_skill::reset();
        let skill_id = next_skill_id();
        let agent_id = next_agent_id();
        let mut pkg = package::sample_report_agent(skill_id, agent_id); // requests Fs READ|WRITE
        pkg.sign();
        // Approve only READ at install.
        let approved = alloc::vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ, Scope::Any)];
        install::install(&pkg, &approved, "vinoth", InstallSource::BootModule { name: "r.skill".into() }, 4).unwrap();
        // Parent holds READ|WRITE, but the grant capped the skill-agent at READ.
        let parent = alloc::vec![CapabilityRequest::new(CapDomain::Fs, Rights::READ | Rights::WRITE, Scope::Any)];
        let eff = agent_skill::effective_caps("report-writer-agent", &parent).expect("role");
        assert_eq!(eff.len(), 1);
        assert!(eff[0].rights.contains(Rights::READ));
        assert!(!eff[0].rights.contains(Rights::WRITE), "install grant (READ) bounds the skill-agent below the parent");
    }

    /// (e) Uninstall revokes the grant and the skill no longer loads.
    #[test_case]
    fn uninstall_revokes_and_unloads() {
        install::reset();
        index::reset(); // isolate: only this test's skill in the index
        let id = next_skill_id();
        let mut pkg = package::sample_note_summarizer(id);
        pkg.sign();
        install::install(&pkg, &pkg.manifest.requested_capabilities.clone(), "vinoth", InstallSource::BootModule { name: "n.skill".into() }, 5).unwrap();
        assert!(install::is_installed(id));
        assert!(index::by_name("note-summarizer").is_some());
        install::uninstall(id);
        assert!(!install::is_installed(id), "grant revoked");
        assert!(index::by_name("note-summarizer").is_none(), "L0 metadata removed");
        let mut s = Session::new(&manifest::orchestrator_manifest(), 1, alloc::vec![], 0);
        assert!(loader::load_body(&mut s, id, 1).is_none(), "uninstalled skill no longer loads");
    }
}
