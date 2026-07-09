//! Built-in **skills** installed at boot as trusted packages (progressive
//! disclosure). Only L0 (name + description) sits in the index until the agent
//! invokes `skill` / `load_skill`, which loads the L1 body (and optional L2
//! assets on demand).

use crate::agent::types::*;
use crate::skills::package::SkillPackage;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

/// Install the bundled skills (idempotent by name: skips if already registered).
/// Called once from boot after the system agents land.
pub fn install_all() {
    for builder in [remember_skill, debug_net_skill, safe_files_skill] {
        let mut pkg = builder(next_skill_id());
        if crate::skills::index::by_name(&pkg.manifest.name).is_some() {
            continue;
        }
        pkg.sign();
        // Boot packages are pre-trusted (same as local MAC-signed samples).
        if let Err(_) = pkg.place_trusted() {
            crate::ktrace::log_fmt(format_args!(
                "skills.bundled: failed to place '{}'",
                pkg.manifest.name
            ));
        }
    }
    let n = crate::skills::index::metadata().len();
    crate::ktrace::log_fmt(format_args!("skills.bundled: {n} skill(s) in L0 index"));
}

fn remember_skill(id: SkillId) -> SkillPackage {
    let body = "\
# remember — durable notes\n\
\n\
When the user asks you to remember a fact, preference, or decision:\n\
1. Call `memory_add` with a short key (`[A-Za-z0-9._-]`) and the value.\n\
2. Optionally append a one-line note via writing MEMORY.md (path `/agent/<id>/MEMORY.md`).\n\
3. Confirm what you stored in one short sentence.\n\
\n\
To recall: `memory_get` / `memory_search` / `memory_list`.\n\
Do not invent keys the user did not imply.\n"
        .to_string();
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", id.0));
    let asset_ref = StoreKey(alloc::format!("skills/{}/refs/examples.md", id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id,
        name: "remember".to_string(),
        version: "1.0.0".to_string(),
        description: "Store and recall durable notes with memory_* tools. Use when the user says remember, recall, or note that.".to_string(),
        kind: SkillKind::Skill,
        requested_capabilities: vec![],
        body_ref,
        bundled_tools: vec![],
        assets: vec![Asset {
            name: "examples".to_string(),
            store_ref: asset_ref,
            bytes: 64,
        }],
        agent: None,
        soul_ref: None,
        skill_docs: Vec::new(),
        signature: SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        },
    };
    SkillPackage {
        manifest,
        body,
        soul: None,
        skill_docs: Vec::new(),
        assets: vec![(
            "examples".to_string(),
            b"Examples:\n- memory_add key=project value=chitti\n- memory_search query=chitti\n".to_vec(),
        )],
    }
}

fn debug_net_skill(id: SkillId) -> SkillPackage {
    let body = "\
# debug-net — network diagnosis\n\
\n\
When networking fails or the user asks about connectivity:\n\
1. `network` (no args) — show IP/gw/dns.\n\
2. `ping` with a host (e.g. 10.0.2.2 or 1.1.1.1).\n\
3. If needed, `http` with a simple GET to a known URL.\n\
4. Report what each tool returned; do not invent IPs.\n\
\n\
Wi-Fi: `wifi` scan/connect only when a wireless interface exists.\n"
        .to_string();
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id,
        name: "debug-net".to_string(),
        version: "1.0.0".to_string(),
        description: "Diagnose network connectivity (network, ping, http). Use when the task mentions network, ping, DNS, or offline.".to_string(),
        kind: SkillKind::Skill,
        requested_capabilities: vec![],
        body_ref,
        bundled_tools: vec![],
        assets: Vec::new(),
        agent: None,
        soul_ref: None,
        skill_docs: Vec::new(),
        signature: SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        },
    };
    SkillPackage {
        manifest,
        body,
        soul: None,
        skill_docs: Vec::new(),
        assets: Vec::new(),
    }
}

fn safe_files_skill(id: SkillId) -> SkillPackage {
    let body = "\
# safe-files — careful file edits\n\
\n\
When editing or searching files:\n\
1. `glob` to find paths (e.g. `*.md`, `/agent/1/**`).\n\
2. `grep` for a unique substring before `edit`.\n\
3. `read` with start_line/end_line for large files.\n\
4. `edit` only with a unique `old` string; use replace_all only when intentional.\n\
5. Prefer `write` for new files; never invent path contents.\n"
        .to_string();
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", id.0));
    let asset_ref = StoreKey(alloc::format!("skills/{}/refs/checklist.md", id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id,
        name: "safe-files".to_string(),
        version: "1.0.0".to_string(),
        description: "Safe file search and edit workflow (glob, grep, read, edit). Use when editing files or searching the store.".to_string(),
        kind: SkillKind::Skill,
        requested_capabilities: vec![CapabilityRequest::new(
            CapDomain::Fs,
            Rights::READ | Rights::WRITE | Rights::LIST,
            Scope::Any,
        )],
        body_ref,
        bundled_tools: vec![],
        assets: vec![Asset {
            name: "checklist".to_string(),
            store_ref: asset_ref,
            bytes: 80,
        }],
        agent: None,
        soul_ref: None,
        skill_docs: Vec::new(),
        signature: SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        },
    };
    SkillPackage {
        manifest,
        body,
        soul: None,
        skill_docs: Vec::new(),
        assets: vec![(
            "checklist".to_string(),
            b"Checklist: glob -> grep unique hit -> read range -> edit unique old -> verify read.\n".to_vec(),
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{index, loader};

    #[test_case]
    fn bundled_skills_install_and_progressive_load() {
        index::reset();
        install_all();
        let metas = index::metadata();
        assert!(metas.iter().any(|m| m.name == "remember"));
        assert!(metas.iter().any(|m| m.name == "debug-net"));
        assert!(metas.iter().any(|m| m.name == "safe-files"));

        let m = crate::agent::manifest::orchestrator_manifest();
        let mut session = Session::new(&m, 1, vec![], 0);
        let rem = index::by_name("remember").expect("remember L0");
        // L0 only until load.
        loader::ensure_metadata(&mut session, rem.id);
        assert_eq!(loader::tier(&session, rem.id), Some(LoadTier::Metadata));
        let body = loader::load_body(&mut session, rem.id, 1).expect("L1");
        assert!(body.contains("memory_add"), "body={body}");
        assert_eq!(loader::tier(&session, rem.id), Some(LoadTier::Body));
        // L2 asset on demand.
        let asset = loader::load_asset(&mut session, rem.id, "examples").expect("L2");
        assert!(!asset.is_empty());
        assert_eq!(loader::tier(&session, rem.id), Some(LoadTier::Full));
    }
}
