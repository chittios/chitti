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
    for builder in [remember_skill, debug_net_skill, safe_files_skill, build_agent_skill] {
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

/// L2 asset for `remember`.
const REMEMBER_EXAMPLES: &str =
    "Examples:\n- memory_add key=project value=chitti\n- memory_search query=chitti\n";

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
            // Computed, not written by hand: this said 64 while the payload was 77,
            // which `bundled_skill_assets_are_declared_and_present` now catches. A
            // declared size that lies is small but makes the metadata untrustworthy.
            bytes: REMEMBER_EXAMPLES.len() as u32,
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
        assets: vec![("examples".to_string(), REMEMBER_EXAMPLES.as_bytes().to_vec())],
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

/// L2 asset for `safe-files`.
const SAFE_FILES_CHECKLIST: &str =
    "Checklist: glob -> grep unique hit -> read range -> edit unique old -> verify read.\n";

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
            // Computed, as for `remember`: the hand-written 80 had drifted from 84.
            bytes: SAFE_FILES_CHECKLIST.len() as u32,
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
        assets: vec![("checklist".to_string(), SAFE_FILES_CHECKLIST.as_bytes().to_vec())],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{index, loader};

    /// Every declared asset must have a payload and vice versa.
    ///
    /// `SkillPackage::place_trusted` writes a payload only if the manifest *declares*
    /// an asset of that name, and silently drops anything else. So a mismatch is not
    /// an error anywhere — the file simply never reaches the store, and the failure
    /// arrives later as a missing asset with nothing pointing at the cause.
    #[test_case]
    fn bundled_skill_assets_are_declared_and_present() {
        for builder in [remember_skill, debug_net_skill, safe_files_skill, build_agent_skill] {
            let pkg = builder(SkillId(1));
            for (name, bytes) in &pkg.assets {
                let declared = pkg.manifest.assets.iter().find(|a| &a.name == name);
                let Some(d) = declared else {
                    panic!("'{}' ships an undeclared asset '{name}' -- place_trusted would drop it", pkg.manifest.name);
                };
                assert_eq!(
                    d.bytes as usize,
                    bytes.len(),
                    "'{}' declares {} bytes for '{name}' but ships {}",
                    pkg.manifest.name,
                    d.bytes,
                    bytes.len()
                );
            }
            for d in &pkg.manifest.assets {
                assert!(
                    pkg.assets.iter().any(|(n, _)| n == &d.name),
                    "'{}' declares asset '{}' with no payload",
                    pkg.manifest.name,
                    d.name
                );
            }
        }
    }

    /// Every authoring step the skill tells the agent to run must actually be runnable.
    ///
    /// The skill shipped saying "run these with `run_shell_command`" while `/agents` was
    /// not in `dispatch_system` at all, so `agents new test` answered "not available as a
    /// tool" and the agent stalled asking the human to type it. Asserting that the body
    /// *mentions* the commands did not catch that — mentioning and being callable are
    /// different claims, and only the second one is what the skill promises.
    #[test_case]
    fn the_build_agent_skills_authoring_steps_are_callable_as_tools() {
        // Called with no package name, so each one prints its usage and touches nothing.
        // What matters is that the dispatcher *reached* them: the bug was the generic
        // "not available as a tool" refusal, which is what this asserts is absent.
        for sub in ["new", "build", "validate"] {
            let out = crate::shell::run_tool_command("agents", sub);
            assert!(
                !out.contains("not available as a tool"),
                "the skill tells the agent to run `agents {sub}`, but run_shell_command \
                 refuses it: {out}"
            );
            assert!(
                out.contains("usage") || out.contains("agents>"),
                "`agents {sub}` should have reached its handler, got: {out}"
            );
        }
        // And the step that grants authority must NOT be silently runnable by an agent.
        // (`agents` as a whole is toolable; the subcommand gate is what refuses install.)
        let out = crate::shell::run_tool_command("agents", "install evil --path /x");
        assert!(
            out.contains("only the human"),
            "install must be refused with a reason, got: {out}"
        );
    }

    /// The `build-agent` skill has to name the commands that exist, in the order they
    /// are used, or it teaches a loop that does not work.
    #[test_case]
    fn the_build_agent_skill_describes_the_real_commands() {
        let pkg = build_agent_skill(SkillId(1));
        for step in [
            "agents new",
            "agents build",
            "agents validate",
            "agents install",
            "--path",
            "agents test",
            "agents reload",
        ] {
            assert!(pkg.body.contains(step), "the body must mention `{step}`");
        }
        // The three contract facts a script gets wrong without being told.
        assert!(pkg.body.contains("export function"), "must state the export shape");
        assert!(pkg.body.contains("no arguments"), "must state that tools take no arguments");
        assert!(pkg.body.contains("re-runs"), "must warn that top level re-runs per call");
        // And the escalation rule, since editing a manifest is the obvious wrong move.
        assert!(pkg.body.contains("cannot** widen") || pkg.body.contains("cannot widen"));
        // The L0 description is what decides relevance, so it must carry the triggers.
        let d = &pkg.manifest.description;
        for trigger in ["new agent", "new tool", "extend"] {
            assert!(d.contains(trigger), "the description should trigger on `{trigger}`");
        }
        // The example asset must be real code, not a placeholder.
        let (_, example) = pkg.assets.first().expect("an example asset");
        let text = core::str::from_utf8(example).expect("utf-8");
        assert!(text.contains("export function notes_add"));
        assert!(text.contains("Javy.IO.readSync"), "the example must show reading stdin");
        assert!(text.contains("Chitti.storageSet"), "the example must show durable state");
    }

    #[test_case]
    fn bundled_skills_install_and_progressive_load() {
        index::reset();
        install_all();
        let metas = index::metadata();
        assert!(metas.iter().any(|m| m.name == "remember"));
        assert!(metas.iter().any(|m| m.name == "debug-net"));
        assert!(metas.iter().any(|m| m.name == "safe-files"));
        assert!(metas.iter().any(|m| m.name == "build-agent"));

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

/// How to build an agent on this machine — the skill that makes the OS able to
/// explain its own extension mechanism.
///
/// Worth being a skill rather than prose in a prompt: it is long, it is only
/// relevant when someone asks for a new tool or agent, and progressive disclosure is
/// exactly the mechanism for that. The L2 asset is a complete working `tools.js`, so
/// the agent can produce a correct one without inventing the ABI.
fn build_agent_skill(id: SkillId) -> SkillPackage {
    let body = "\
# build-agent — create a new agent (or a new tool) on this machine\n\
\n\
ChittiOS compiles JavaScript to a real wasm module locally, so a new agent needs no\n\
host toolchain and no kernel rebuild.\n\
\n\
**Steps 1-4 you run yourself** with `run_shell_command` (they only write and check\n\
files). **Steps 5-7 the human types** at the prompt: `install` grants capabilities and\n\
needs their approval, and `test`/`reload` act on the live chat session. So do the\n\
authoring, then tell them exactly which line to type — do not stall waiting for\n\
permission to scaffold, and do not report the loop as blocked when four of its steps\n\
are yours.\n\
\n\
## The loop\n\
\n\
1. `agents new <name>` — scaffolds `~/agents/<name>/` with SOUL.md, manifest.json and\n\
   a working `tools.js` (two example tools). Names: lowercase, digits, `-`/`_`.\n\
2. Edit `~/agents/<name>/tools.js`. Use the `write` tool for the whole file, or\n\
   `search_replace` for a part of it. Each tool is one `export function`.\n\
3. `agents build <name>` — compiles `tools.js` to `assets/tools.wasm` (~100 ms).\n\
   Reports which tools it exported. **Editing the script changes nothing until you\n\
   build.**\n\
4. `agents validate <name>` — lints the manifest and the module. Fix every `error`\n\
   before installing; `warn` lines are advisory.\n\
5. **(human types)** `/agents install <name> --path ~/agents/<name>` — the consent\n\
   screen; they approve each capability, then it registers the agent and its tools.\n\
   You cannot run this: installing is what grants authority, so it is theirs.\n\
6. **(human types)** `/agents test <name> --tool <tool> --args {\"k\":1}` — runs one\n\
   tool under that agent's identity and prints the structured outcome.\n\
7. After a later edit: `agents build <name>` (yours), then the human types\n\
   `/agents reload <name>`. Both are needed — a build alone does not reload.\n\
\n\
## How a tool is written\n\
\n\
The engine can only call `export function` names, they take **no arguments**, and\n\
their **return value is dropped**. So arguments arrive as JSON on stdin and the result\n\
leaves as JSON on stdout — the scaffold's `readArgs()` and `reply()` do this. Load the\n\
`example` asset of this skill for a complete file.\n\
\n\
Rules that are not obvious:\n\
\n\
- **Prefix every tool with the agent name** (`notes_add`, not `add`). The tool\n\
  registry is global.\n\
- **The manifest's `toolset` and the script's exports must agree.** Only a name the\n\
  script exports becomes a callable tool; a name only the manifest lists is invisible,\n\
  and `agents validate` says so.\n\
- **Module top level re-runs on every call**, so a JS global does not persist between\n\
  calls. Durable state goes in `Chitti.storageSet`.\n\
- Do not write wasm by hand, and do not claim a tool the script does not export.\n\
\n\
## What a tool may do: the `Chitti` global\n\
\n\
The same capability-gated surface a Rust tool module gets:\n\
\n\
- `Chitti.storageGet(durable, key)` / `storageSet(durable, key, value)` /\n\
  `storageRemove` / `storageList` — `durable` true survives a reboot.\n\
- `Chitti.fsRead(path)` / `fsWrite(path, data)` / `fsList(path)` / `fsExists(path)` —\n\
  confined to the agent's own folder unless its manifest grants a wider `fs` scope.\n\
- `Chitti.http(requestJson)` — only if the manifest declares a `net` capability.\n\
- `Chitti.notify(severity, title, body?)` — tell the human something that outlives\n\
  this call. `severity` is `info` | `ok` | `warn` | `error`. Use it for what they\n\
  need to know **later** (a job finished, a fetch failed), never as a reply — a\n\
  tool's return value is the reply. Write-only: an agent cannot read notifications\n\
  back, and the source is stamped by the OS, so it always says which agent posted.\n\
- `Chitti.log(msg)`, `Chitti.sha1(data)`, `Chitti.home()`, `Chitti.nowMs()`.\n\
\n\
Anything the agent may not do **throws**; a value that simply is not there comes back\n\
as `null`. Catch and report the message — never retry a refusal in a loop.\n\
\n\
## When it does not work\n\
\n\
- `tools.wasm is missing or empty` → run `agents build <name>`.\n\
- `built against JS engine '…'` → the engine was rebuilt; run `agents build` again.\n\
- a tool refuses with `no such capability bound` → the manifest does not request it, or\n\
  the human did not approve it. Editing the manifest and reloading **cannot** widen a\n\
  grant; re-run `agents install <name> --path` so a human can approve the addition.\n\
- `matches no registered tool` from validate → the script does not export that name.\n"
        .to_string();
    let body_ref = StoreKey(alloc::format!("skills/{}/body.md", id.0));
    let asset_ref = StoreKey(alloc::format!("skills/{}/refs/example.md", id.0));
    let manifest = SkillManifest {
        schema_version: 1,
        id,
        name: "build-agent".to_string(),
        version: "1.0.0".to_string(),
        description: "Create a new agent or a new tool on this machine: scaffold, write it in JavaScript, compile to wasm, install under consent, test and reload. Use when the user asks for a new agent, a new tool, a custom command, or to extend what this OS can do.".to_string(),
        kind: SkillKind::Skill,
        requested_capabilities: vec![],
        body_ref,
        bundled_tools: vec![],
        assets: vec![Asset {
            name: "example".to_string(),
            store_ref: asset_ref,
            bytes: EXAMPLE_TOOLS_JS.len() as u32,
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
        assets: vec![("example".to_string(), EXAMPLE_TOOLS_JS.as_bytes().to_vec())],
    }
}

/// L2 asset for `build-agent`: a complete, correct `tools.js`.
///
/// Loaded on demand rather than living in the body, because the body is what decides
/// *whether* to do this and the example is what makes the doing correct. Written out
/// in full so an agent copies working code instead of reconstructing the ABI from
/// prose — the parts it would get wrong (reading stdin to EOF, one `export function`
/// per tool, state in storage rather than a global) are exactly the parts here.
const EXAMPLE_TOOLS_JS: &str = r#"A complete tools.js for an agent named `notes`.
Write this with the `write` tool, then: agents build notes

```javascript
// Arguments arrive as JSON on stdin; the result leaves as JSON on stdout, because
// exported functions take no parameters and their return value is dropped.
function readArgs() {
  const chunks = [];
  const buf = new Uint8Array(1024);
  let n;
  while ((n = Javy.IO.readSync(0, buf)) > 0) chunks.push(buf.slice(0, n));
  let total = 0;
  for (const c of chunks) total += c.length;
  const all = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) { all.set(c, at); at += c.length; }
  return JSON.parse(new TextDecoder().decode(all) || "{}");
}

function reply(value) {
  Javy.IO.writeSync(1, new TextEncoder().encode(JSON.stringify(value)));
}

// One `export function` per tool. Prefix with the agent name: the registry is global.
export function notes_add() {
  const args = readArgs();
  if (typeof args.text !== "string" || !args.text) {
    reply({ ok: false, error: "text is required" });
    return;
  }
  // Module top level re-runs on every call, so a JS global would not survive.
  // Durable state goes in storage.
  const existing = Chitti.storageGet(true, "notes");   // null when nothing is stored
  const notes = existing ? JSON.parse(existing) : [];
  notes.push({ text: args.text, at: Chitti.nowMs() });
  Chitti.storageSet(true, "notes", JSON.stringify(notes));
  reply({ ok: true, count: notes.length });
}

export function notes_list() {
  const existing = Chitti.storageGet(true, "notes");
  reply({ ok: true, notes: existing ? JSON.parse(existing) : [] });
}

export function notes_clear() {
  Chitti.storageRemove(true, "notes");
  reply({ ok: true });
}
```

The manifest's `toolset` must list exactly these names:

```json
"toolset": ["notes_add", "notes_list", "notes_clear"]
```

A capability the agent does not hold **throws**, so when a tool may be refused, say
what happened rather than failing silently:

```javascript
export function notes_fetch() {
  const args = readArgs();
  try {
    reply({ ok: true, response: JSON.parse(Chitti.http(JSON.stringify({ url: args.url }))) });
  } catch (e) {
    // e.g. no `net` capability in the manifest, or the human did not approve it
    reply({ ok: false, refused: String(e) });
  }
}
```
"#;
