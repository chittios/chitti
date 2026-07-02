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

pub mod index;
pub mod install;
pub mod loader;
pub mod package;

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
}
