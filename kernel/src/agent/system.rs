//! **System agents** — built-in packages under the repo's `agents/` folder:
//! services (`doc`, `ssh`), media (`media`, `pdf`), package-UI apps (`chess`,
//! `paint`, `slides`, games, plus default-OS UI like `calc`/`files`/`sheets`/…),
//! and chat skill-agents (`download`, `todo`, `browser`, `librarian`, …).
//! Each is a markdown `SOUL.md` + JSON manifest. Definitions are compiled in
//! via `include_str!`/`include_bytes!`; at boot [`install_all`] signs each into
//! a `SkillPackage` and installs it through the normal permissioned flow —
//! SOUL lands in `/agent/<id>/SOUL.md`, grants are recorded, roles registered
//! (pre-trusted, same shape as a registry package).
//!
//! Packages with `"autostart": true` (download, notes, todo) are activated by
//! [`autostart_agents`]: homes are ensured and their toolsets merge into the
//! shell orchestrator so chat can use them without `/agents start`. UI packages
//! still need `/agents start <name>` for a surface; web content uses the
//! generic pipeline. Protocol / codec logic stays native code below the
//! determinism boundary — markdown is persona + policy only.

use crate::agent::types::*;
use crate::json::Json;
use crate::skills::package::SkillPackage;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A built-in system agent: its markdown SOUL + JSON manifest (compiled in),
/// stable ids, and any bundled assets that land in its install folder. The
/// web agents (network/http/doc) run together as the [`super::super::service::pipeline`];
/// ssh runs standalone. Wiring is done by the `/agents start` command.
struct SystemAgentDef {
    name: &'static str,
    soul: &'static str,
    manifest_json: &'static str,
    skill_id: SkillId,
    agent_id: AgentId,
    /// `(filename, contents)` written to `/agent/<id>/assets/` on install (e.g.
    /// the Doc agent's HTML + logo). Text, compiled in via `include_str!`.
    assets: &'static [(&'static str, &'static str)],
    /// Binary assets (e.g. `tools.wasm`), via `include_bytes!`.
    binary_assets: &'static [(&'static str, &'static [u8])],
}

// Stable ids so `/agent/<id>/` is consistent across boots and doesn't collide
// with runtime-minted ids (which start at 1).
const SYSTEM_SKILL_BASE: u64 = 9000;
const SYSTEM_AGENT_BASE: u64 = 9000;

// Only agents that actually reason from a SOUL are installed agents. The
// `network` and `http` stages are pure mechanical plumbing (relay bytes, parse a
// protocol) — no judgment, no SOUL — so they live entirely in `crate::service`,
// not here. `doc` routes via `assets/tools.wasm` (deterministic); SOUL remains
// for model fallback. `ssh` carries a login/tunnel persona.
static SYSTEM_AGENTS: &[SystemAgentDef] = &[
    SystemAgentDef {
        name: "doc",
        soul: include_str!("../../../agents/doc/SOUL.md"),
        manifest_json: include_str!("../../../agents/doc/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 1),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 1),
        assets: &[
            ("index.html", include_str!("../../../agents/doc/assets/index.html")),
            ("docs.html", include_str!("../../../agents/doc/assets/docs.html")),
            ("logo.svg", include_str!("../../../agents/doc/assets/logo.svg")),
        ],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/doc/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "ssh",
        soul: include_str!("../../../agents/ssh/SOUL.md"),
        manifest_json: include_str!("../../../agents/ssh/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 2),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 2),
        assets: &[],
        binary_assets: &[],
    },
    // UI agent: paints a chess board via board_set/board_mark; rules live in
    // assets/tools.wasm (built by tools/chess-wasm/).
    SystemAgentDef {
        name: "chess",
        soul: include_str!("../../../agents/chess/SOUL.md"),
        manifest_json: include_str!("../../../agents/chess/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 3),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 3),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/chess/assets/tools.wasm"),
        )],
    },
    // Media agent: image / audio / video tools with full filesystem access.
    SystemAgentDef {
        name: "media",
        soul: include_str!("../../../agents/media/SOUL.md"),
        manifest_json: include_str!("../../../agents/media/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 4),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 4),
        assets: &[],
        binary_assets: &[],
    },
    // PDF agent: previews/answers over PDF documents; the parsing (xref,
    // FlateDecode, page tree, text extraction) is deterministic wasm
    // (tools/pdf-wasm). Full filesystem READ so any mounted PDF opens.
    SystemAgentDef {
        name: "pdf",
        soul: include_str!("../../../agents/pdf/SOUL.md"),
        manifest_json: include_str!("../../../agents/pdf/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 5),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 5),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/pdf/assets/tools.wasm"),
        )],
    },
    // App packages: logic only in assets/tools.wasm + SOUL.md (tools/apps-wasm).
    SystemAgentDef {
        name: "notes",
        soul: include_str!("../../../agents/notes/SOUL.md"),
        manifest_json: include_str!("../../../agents/notes/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 6),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 6),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/notes/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "paint",
        soul: include_str!("../../../agents/paint/SOUL.md"),
        manifest_json: include_str!("../../../agents/paint/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 7),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 7),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/paint/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "slides",
        soul: include_str!("../../../agents/slides/SOUL.md"),
        manifest_json: include_str!("../../../agents/slides/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 8),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 8),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/slides/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "minesweeper",
        soul: include_str!("../../../agents/minesweeper/SOUL.md"),
        manifest_json: include_str!("../../../agents/minesweeper/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 9),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 9),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/minesweeper/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "snake",
        soul: include_str!("../../../agents/snake/SOUL.md"),
        manifest_json: include_str!("../../../agents/snake/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 10),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 10),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/snake/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "synth",
        soul: include_str!("../../../agents/synth/SOUL.md"),
        manifest_json: include_str!("../../../agents/synth/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 11),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 11),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/synth/assets/tools.wasm"),
        )],
    },
    // Download agent: HTTP(S) fetch + save via host `download` tool + http/fs.
    SystemAgentDef {
        name: "download",
        soul: include_str!("../../../agents/download/SOUL.md"),
        manifest_json: include_str!("../../../agents/download/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 12),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 12),
        assets: &[],
        binary_assets: &[],
    },
    // Todo agent: session planning (host todo_write / plan-mode tools).
    SystemAgentDef {
        name: "todo",
        soul: include_str!("../../../agents/todo/SOUL.md"),
        manifest_json: include_str!("../../../agents/todo/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 13),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 13),
        assets: &[],
        binary_assets: &[],
    },
    // Browser agent: fetch + subset HTML layout/paint in the action pane.
    SystemAgentDef {
        name: "browser",
        soul: include_str!("../../../agents/browser/SOUL.md"),
        manifest_json: include_str!("../../../agents/browser/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 14),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 14),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "calc",
        soul: include_str!("../../../agents/calc/SOUL.md"),
        manifest_json: include_str!("../../../agents/calc/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 15),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 15),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/calc/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "clock",
        soul: include_str!("../../../agents/clock/SOUL.md"),
        manifest_json: include_str!("../../../agents/clock/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 16),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 16),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/clock/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "files",
        soul: include_str!("../../../agents/files/SOUL.md"),
        manifest_json: include_str!("../../../agents/files/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 17),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 17),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/files/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "gallery",
        soul: include_str!("../../../agents/gallery/SOUL.md"),
        manifest_json: include_str!("../../../agents/gallery/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 18),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 18),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/gallery/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "sheets",
        soul: include_str!("../../../agents/sheets/SOUL.md"),
        manifest_json: include_str!("../../../agents/sheets/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 19),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 19),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/sheets/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "calendar",
        soul: include_str!("../../../agents/calendar/SOUL.md"),
        manifest_json: include_str!("../../../agents/calendar/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 20),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 20),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/calendar/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "contacts",
        soul: include_str!("../../../agents/contacts/SOUL.md"),
        manifest_json: include_str!("../../../agents/contacts/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 21),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 21),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/contacts/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "writer",
        soul: include_str!("../../../agents/writer/SOUL.md"),
        manifest_json: include_str!("../../../agents/writer/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 22),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 22),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/writer/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "archive",
        soul: include_str!("../../../agents/archive/SOUL.md"),
        manifest_json: include_str!("../../../agents/archive/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 23),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 23),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/archive/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "hex",
        soul: include_str!("../../../agents/hex/SOUL.md"),
        manifest_json: include_str!("../../../agents/hex/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 24),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 24),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/hex/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "game2048",
        soul: include_str!("../../../agents/game2048/SOUL.md"),
        manifest_json: include_str!("../../../agents/game2048/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 25),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 25),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/game2048/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "activity",
        soul: include_str!("../../../agents/activity/SOUL.md"),
        manifest_json: include_str!("../../../agents/activity/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 26),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 26),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/activity/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "weather",
        soul: include_str!("../../../agents/weather/SOUL.md"),
        manifest_json: include_str!("../../../agents/weather/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 27),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 27),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/weather/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "settings",
        soul: include_str!("../../../agents/settings/SOUL.md"),
        manifest_json: include_str!("../../../agents/settings/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 28),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 28),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/settings/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "dict",
        soul: include_str!("../../../agents/dict/SOUL.md"),
        manifest_json: include_str!("../../../agents/dict/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 29),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 29),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/dict/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "diff",
        soul: include_str!("../../../agents/diff/SOUL.md"),
        manifest_json: include_str!("../../../agents/diff/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 30),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 30),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/diff/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "librarian",
        soul: include_str!("../../../agents/librarian/SOUL.md"),
        manifest_json: include_str!("../../../agents/librarian/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 31),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 31),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "researcher",
        soul: include_str!("../../../agents/researcher/SOUL.md"),
        manifest_json: include_str!("../../../agents/researcher/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 32),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 32),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "ops",
        soul: include_str!("../../../agents/ops/SOUL.md"),
        manifest_json: include_str!("../../../agents/ops/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 33),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 33),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "onboard",
        soul: include_str!("../../../agents/onboard/SOUL.md"),
        manifest_json: include_str!("../../../agents/onboard/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 34),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 34),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "store",
        soul: include_str!("../../../agents/store/SOUL.md"),
        manifest_json: include_str!("../../../agents/store/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 35),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 35),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "mail",
        soul: include_str!("../../../agents/mail/SOUL.md"),
        manifest_json: include_str!("../../../agents/mail/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 36),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 36),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "disk",
        soul: include_str!("../../../agents/disk/SOUL.md"),
        manifest_json: include_str!("../../../agents/disk/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 37),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 37),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "pass",
        soul: include_str!("../../../agents/pass/SOUL.md"),
        manifest_json: include_str!("../../../agents/pass/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 38),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 38),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "recorder",
        soul: include_str!("../../../agents/recorder/SOUL.md"),
        manifest_json: include_str!("../../../agents/recorder/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 39),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 39),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "reader",
        soul: include_str!("../../../agents/reader/SOUL.md"),
        manifest_json: include_str!("../../../agents/reader/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 40),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 40),
        assets: &[],
        binary_assets: &[],
    },
    SystemAgentDef {
        name: "breakout",
        soul: include_str!("../../../agents/breakout/SOUL.md"),
        manifest_json: include_str!("../../../agents/breakout/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 41),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 41),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/breakout/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "tetris",
        soul: include_str!("../../../agents/tetris/SOUL.md"),
        manifest_json: include_str!("../../../agents/tetris/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 42),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 42),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/tetris/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "console",
        soul: include_str!("../../../agents/console/SOUL.md"),
        manifest_json: include_str!("../../../agents/console/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 43),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 43),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/console/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "maps",
        soul: include_str!("../../../agents/maps/SOUL.md"),
        manifest_json: include_str!("../../../agents/maps/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 44),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 44),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/maps/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "radio",
        soul: include_str!("../../../agents/radio/SOUL.md"),
        manifest_json: include_str!("../../../agents/radio/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 45),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 45),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/radio/assets/tools.wasm"),
        )],
    },
    SystemAgentDef {
        name: "sandbox-lab",
        soul: include_str!("../../../agents/sandbox-lab/SOUL.md"),
        manifest_json: include_str!("../../../agents/sandbox-lab/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 46),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 46),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/sandbox-lab/assets/tools.wasm"),
        )],
    },
    // git: version control. All logic ships as tools.wasm (git_command); the
    // `/git` command hook routes the shell to it. Host imports give the wasm
    // FS (home-scoped), SHA-1, zlib, clock and HTTP (net cap) — see
    // `wasm_rt::register_host_imports`.
    SystemAgentDef {
        name: "git",
        soul: include_str!("../../../agents/git/SOUL.md"),
        manifest_json: include_str!("../../../agents/git/manifest.json"),
        skill_id: SkillId(SYSTEM_SKILL_BASE + 47),
        agent_id: AgentId(SYSTEM_AGENT_BASE + 47),
        assets: &[],
        binary_assets: &[(
            "tools.wasm",
            include_bytes!("../../../agents/git/assets/tools.wasm"),
        )],
    },
];

/// Whether system agent `id` declares a `net` capability in its manifest
/// (gates the wasm `host_http` host import — a wasm tool must not reach the
/// network unless the agent's granted capabilities include it).
pub fn has_net_cap(id: u64) -> bool {
    SYSTEM_AGENTS.iter().any(|d| {
        d.agent_id.0 == id
            && parse_manifest(d.manifest_json).is_some_and(|m| {
                m.capabilities
                    .iter()
                    .any(|c| c.domain == crate::agent::types::CapDomain::Net)
            })
    })
}

/// The manifest's `wasm.fuel` for system agent `id` (the wasm runtime's per-call
/// instruction budget). `None` when the manifest declares none.
pub fn manifest_fuel(id: u64) -> Option<u64> {
    let def = SYSTEM_AGENTS.iter().find(|d| d.agent_id.0 == id)?;
    let j = Json::parse(def.manifest_json)?;
    j.get("wasm")
        .and_then(|w| w.get("fuel"))
        .and_then(|f| f.as_i64())
        .map(|v| v.max(1) as u64)
}

/// The manifest's `wasm.memory_pages` for system agent `id` (the wasm runtime's
/// linear-memory ceiling, in 64 KiB pages). `None` when the manifest declares
/// none, which leaves [`crate::agent::wasm_rt::DEFAULT_MAX_MEMORY_PAGES`].
///
/// Fuel bounds how long a guest runs; this bounds how much it can hold, and the
/// two are not interchangeable. The default 2 MiB suits a tool that digests one
/// small argument, and is well under what a guest handling a whole *document*
/// needs — the git agent's clone holds a packfile plus every object it unpacks.
pub fn manifest_pages(id: u64) -> Option<u32> {
    let def = SYSTEM_AGENTS.iter().find(|d| d.agent_id.0 == id)?;
    let j = Json::parse(def.manifest_json)?;
    j.get("wasm")
        .and_then(|w| w.get("memory_pages"))
        .and_then(|f| f.as_i64())
        .map(|v| v.clamp(1, u32::MAX as i64) as u32)
}

/// Whether system agent `id`'s manifest grants an **unrestricted** (`Scope::Any`)
/// filesystem scope — gates the wasm `host_fs_write` import. The git agent
/// declares one so it can clone/checkout into `/home/…` and any user folder;
/// every other agent keeps the `home` scope (only its own `/agent/<id>/`).
pub fn fs_any_scope(id: u64) -> bool {
    SYSTEM_AGENTS.iter().any(|d| {
        d.agent_id.0 == id
            && parse_manifest(d.manifest_json).is_some_and(|m| {
                m.capabilities.iter().any(|c| {
                    c.domain == crate::agent::types::CapDomain::Fs
                        && matches!(c.scope, crate::agent::types::Scope::Any)
                })
            })
    })
}

/// The install-folder path (`/agent/<id>`) of a system agent by name.
pub fn home_for(name: &str) -> Option<String> {
    SYSTEM_AGENTS.iter().find(|d| d.name == name).map(|d| crate::agent::home::path(d.agent_id.0))
}

/// One extension → tool mapping under a [`CommandHook`].
#[derive(Clone, Debug)]
pub struct HookDispatch {
    /// Tool name to invoke (e.g. `audio_player`).
    pub tool: String,
    /// JSON arg key for the path (default `path`).
    pub path_arg: String,
}

/// A shell command interception declared by a package (e.g. media owns `/open`
/// for media extensions, or the git agent owns the bare command `/git`).
/// Parsed from manifest `command_hooks`.
#[derive(Clone, Debug)]
pub struct CommandHook {
    /// Slash command, e.g. `"/open"`.
    pub command: String,
    pub description: String,
    /// File extensions this hook matches (lowercase, with leading `.`).
    pub extensions: Vec<String>,
    /// Bare-command hook: the shell routes `command` to the agent's tool with
    /// the *rest of the line* as the argument, no path/extension matching
    /// (e.g. `/settings`, `/git …`). Declared with `"match": {"bare": true}`.
    pub bare: bool,
    /// Per-extension tool dispatch (key = extension, or `"default"` for a bare
    /// hook's single tool).
    pub dispatch: alloc::collections::BTreeMap<String, HookDispatch>,
}

/// Result of resolving a path against installed agents' command hooks.
#[derive(Clone, Debug)]
pub struct OpenHookMatch {
    pub agent_name: &'static str,
    pub agent_id: u64,
    pub tool: String,
    pub path_arg: String,
    pub extension: String,
}

/// The parsed, portable part of a system agent's `manifest.json`.
pub struct ParsedManifest {
    pub name: String,
    pub version: String,
    pub kind: AgentKind,
    pub description: String,
    pub toolset: Vec<String>,
    pub capabilities: Vec<CapabilityRequest>,
    pub default_port: u16,
    pub autostart: bool,
    pub mcp_servers: Vec<crate::agent::types::McpServerSpec>,
    /// Shell command hooks (`/open`, …) this package claims.
    pub command_hooks: Vec<CommandHook>,
    /// Guest wasm module path from `wasm.module` (e.g. `"assets/tools.wasm"`).
    /// Absent for chat/service agents that have no package-UI surface.
    pub wasm_module: Option<String>,
    /// Explicit `"package_ui": true` — the package is a canvas app started via
    /// `package_ui::start`. When `None`, classification falls back to
    /// wasm.module + Ui EXEC inference.
    pub package_ui: Option<bool>,
}

fn parse_domain(s: &str) -> Option<CapDomain> {
    Some(match s {
        "fs" => CapDomain::Fs,
        "console" => CapDomain::Console,
        "spawn" => CapDomain::Spawn,
        "todo" => CapDomain::Todo,
        "inference" => CapDomain::Inference,
        "ipc" => CapDomain::Ipc,
        "skill_manage" => CapDomain::SkillManage,
        "channel" => CapDomain::Channel,
        "net" => CapDomain::Net,
        "ui" => CapDomain::Ui,
        _ => return None,
    })
}

fn parse_rights(arr: &[Json]) -> Rights {
    let mut r = Rights::empty();
    for v in arr {
        match v.as_str() {
            Some("read") => r |= Rights::READ,
            Some("write") => r |= Rights::WRITE,
            Some("exec") => r |= Rights::EXEC,
            Some("delete") => r |= Rights::DELETE,
            Some("list") => r |= Rights::LIST,
            _ => {}
        }
    }
    r
}

fn parse_scope(s: &str) -> Scope {
    // "home" = read/write only within the agent's own install folder + memory.
    // Resolved to the concrete `/agent/<id>/**` path in `build_package` (which
    // knows the id) via the `$HOME` sentinel.
    if s == "home" {
        return Scope::Path("$HOME/**".to_string());
    }
    if let Some(rest) = s.strip_prefix("path:") {
        return Scope::Path(rest.to_string());
    }
    if let Some(rest) = s.strip_prefix("net:") {
        // net:host:lo-hi  or  net:host:port
        let mut it = rest.rsplitn(2, ':');
        let ports = it.next().unwrap_or("");
        let host = it.next().unwrap_or("*").to_string();
        let (lo, hi) = match ports.split_once('-') {
            Some((a, b)) => (a.parse().unwrap_or(0), b.parse().unwrap_or(0)),
            None => {
                let p = ports.parse().unwrap_or(0);
                (p, p)
            }
        };
        return Scope::Net { host, port_lo: lo, port_hi: hi };
    }
    Scope::Any
}

/// Parse a system agent's `manifest.json` into its portable manifest.
pub fn parse_manifest(json: &str) -> Option<ParsedManifest> {
    let j = Json::parse(json)?;
    let kind = match j.get("kind").and_then(|k| k.as_str()) {
        Some("service") => AgentKind::Service,
        Some("skill_agent") => AgentKind::SkillAgent,
        Some("subagent") => AgentKind::Subagent,
        _ => AgentKind::Service,
    };
    let toolset = j
        .get("toolset")
        .and_then(|t| t.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let mut caps = Vec::new();
    if let Some(arr) = j.get("capabilities").and_then(|c| c.as_array()) {
        for c in arr {
            if let Some(domain) = c.get("domain").and_then(|d| d.as_str()).and_then(parse_domain) {
                let rights = c.get("rights").and_then(|r| r.as_array()).map(parse_rights).unwrap_or_else(Rights::empty);
                let scope = c.get("scope").and_then(|s| s.as_str()).map(parse_scope).unwrap_or(Scope::Any);
                caps.push(CapabilityRequest::new(domain, rights, scope));
            }
        }
    }
    // Optional `mcp_servers`: [{ "name", "url", "bearer"? }, …].
    let mut mcp_servers = Vec::new();
    if let Some(arr) = j.get("mcp_servers").and_then(|m| m.as_array()) {
        for s in arr {
            if let (Some(name), Some(url)) =
                (s.get("name").and_then(|v| v.as_str()), s.get("url").and_then(|v| v.as_str()))
            {
                mcp_servers.push(crate::agent::types::McpServerSpec {
                    name: name.to_string(),
                    url: url.to_string(),
                    bearer: s.get("bearer").and_then(|v| v.as_str()).map(|b| b.to_string()),
                });
            }
        }
    }
    let command_hooks = parse_command_hooks(&j);
    let wasm_module = j
        .get("wasm")
        .and_then(|w| w.get("module"))
        .and_then(|m| m.as_str())
        .map(|s| s.to_string());
    let package_ui = j.get("package_ui").and_then(|v| v.as_bool());
    Some(ParsedManifest {
        name: j.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        version: j.get("version").and_then(|v| v.as_str()).unwrap_or("0.0.0").to_string(),
        kind,
        description: j.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        toolset,
        capabilities: caps,
        default_port: j.get("default_port").and_then(|v| v.as_i64()).unwrap_or(0) as u16,
        autostart: j.get("autostart").and_then(|v| v.as_bool()).unwrap_or(false),
        mcp_servers,
        command_hooks,
        wasm_module,
        package_ui,
    })
}

/// Parse `command_hooks` from a package root JSON object.
fn parse_command_hooks(j: &Json) -> Vec<CommandHook> {
    let Some(arr) = j.get("command_hooks").and_then(|c| c.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for h in arr {
        let command = h
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if command.is_empty() {
            continue;
        }
        let description = h
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        // `"match": {"bare": true}` = the shell routes the bare command (no
        // path/extension) to the agent's tool with the rest of the line as the
        // argument. Otherwise this is an extension hook (media owns /open).
        let bare = h
            .get("match")
            .and_then(|m| m.get("bare"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        let mut extensions = Vec::new();
        if let Some(exts) = h
            .get("match")
            .and_then(|m| m.get("extensions"))
            .and_then(|e| e.as_array())
        {
            for e in exts {
                if let Some(s) = e.as_str() {
                    let mut ext = s.to_ascii_lowercase();
                    if !ext.starts_with('.') {
                        ext.insert(0, '.');
                    }
                    extensions.push(ext);
                }
            }
        }
        let mut dispatch = alloc::collections::BTreeMap::new();
        if let Some(Json::Obj(pairs)) = h.get("dispatch") {
            for (ext_key, rule) in pairs {
                let mut ext = ext_key.to_ascii_lowercase();
                if !ext.starts_with('.') {
                    ext.insert(0, '.');
                }
                let tool = rule
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if tool.is_empty() {
                    continue;
                }
                let path_arg = rule
                    .get("path_arg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("path")
                    .to_string();
                dispatch.insert(ext, HookDispatch { tool, path_arg });
            }
        }
        // Fill dispatch from extensions if a single default tool is missing keys.
        out.push(CommandHook {
            command,
            description,
            extensions,
            bare,
            dispatch,
        });
    }
    out
}

/// File extension of `path` (lowercase, with `.`), or empty if none.
pub fn path_extension(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    if let Some(dot) = base.rfind('.') {
        if dot > 0 {
            return base[dot..].to_ascii_lowercase();
        }
    }
    String::new()
}

/// Resolve a shell `/open <path>` against system agents' `command_hooks`.
/// Returns the owning agent + tool when an extension matches a hook for
/// `command` (normalized to start with `/`).
pub fn resolve_open_hook(path: &str) -> Option<OpenHookMatch> {
    resolve_command_hook("/open", path)
}

/// Generic: resolve `command` + path against all system package hooks.
pub fn resolve_command_hook(command: &str, path: &str) -> Option<OpenHookMatch> {
    let cmd = if command.starts_with('/') {
        command.to_string()
    } else {
        alloc::format!("/{command}")
    };
    let ext = path_extension(path);
    if ext.is_empty() {
        return None;
    }
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        for hook in &m.command_hooks {
            if hook.command != cmd {
                continue;
            }
            let matches_ext = hook.extensions.iter().any(|e| e == &ext)
                || hook.dispatch.contains_key(&ext);
            if !matches_ext {
                continue;
            }
            let disp = hook.dispatch.get(&ext)?;
            return Some(OpenHookMatch {
                agent_name: def.name,
                agent_id: def.agent_id.0,
                tool: disp.tool.clone(),
                path_arg: disp.path_arg.clone(),
                extension: ext,
            });
        }
    }
    None
}

/// Resolve a **bare** command (`/git …`, `/settings`) against system agents'
/// `command_hooks` — the manifest-declared aliases that turn an agent into a
/// shell command. `args` is the rest of the line after the command name; the
/// matching agent's tool receives it under its `path_arg` key. `None` when no
/// agent claims the command.
pub fn resolve_command_hook_bare(command: &str) -> Option<OpenHookMatch> {
    // Reserved names a manifest may never claim. Today this is dead code — the
    // login commands are matched in the REPL above the fall-through that reaches
    // hooks, so one could never fire for them — but the reservation says the
    // intent out loud instead of depending on the order of two `match` arms in
    // another file.
    let bare = command.trim_start_matches('/');
    if crate::shell::catalog::is_human_only(bare) {
        return None;
    }
    let cmd = if command.starts_with('/') {
        command.to_string()
    } else {
        alloc::format!("/{command}")
    };
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        for hook in &m.command_hooks {
            if hook.bare && hook.command == cmd {
                // The single dispatch entry (`"default"` key) names the tool.
                let disp = hook
                    .dispatch
                    .get("default")
                    .or_else(|| hook.dispatch.values().next());
                if let Some(disp) = disp {
                    return Some(OpenHookMatch {
                        agent_name: def.name,
                        agent_id: def.agent_id.0,
                        tool: disp.tool.clone(),
                        path_arg: disp.path_arg.clone(),
                        extension: String::new(),
                    });
                }
            }
        }
    }
    None
}

/// All `/open` extensions claimed by any system agent (for help text).
pub fn open_hook_extensions() -> Vec<String> {
    let mut exts = Vec::new();
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        for hook in &m.command_hooks {
            if hook.command != "/open" {
                continue;
            }
            for e in &hook.extensions {
                if !exts.iter().any(|x| x == e) {
                    exts.push(e.clone());
                }
            }
        }
    }
    exts.sort();
    exts
}

/// The agent's declared capabilities with the `$HOME` sentinel resolved to its
/// concrete install folder (so a `scope: "home"` grant becomes a real path).
fn resolved_caps(def: &SystemAgentDef, m: &ParsedManifest) -> Vec<CapabilityRequest> {
    let home = crate::agent::home::path(def.agent_id.0);
    m.capabilities
        .iter()
        .map(|c| {
            let scope = match &c.scope {
                Scope::Path(p) if p.contains("$HOME") => Scope::Path(p.replace("$HOME", &home)),
                other => other.clone(),
            };
            CapabilityRequest::new(c.domain, c.rights, scope)
        })
        .collect()
}

/// Build the signed `SkillPackage` for a system agent from its parsed manifest,
/// SOUL, resolved capabilities, and bundled assets.
fn build_package(def: &SystemAgentDef, m: &ParsedManifest, caps: &[CapabilityRequest]) -> SkillPackage {
    let agent = AgentManifest {
        schema_version: 1,
        id: def.agent_id,
        name: m.name.clone(),
        version: m.version.clone(),
        kind: m.kind,
        description: m.description.clone(),
        system_prompt: def.soul.to_string(),
        toolset: m.toolset.clone(),
        capabilities: caps.to_vec(),
        skills: Vec::new(),
        sampling: Sampling::deterministic(1),
        budgets: Budgets {
            max_turns: 8,
            max_context_tokens: 4096,
            compact_threshold: 3500,
            max_tool_calls: 256,
            max_subagents: 2,
            max_depth: 1,
            max_wall_ticks: 0,
        },
        summary: SummaryPolicy { max_tokens: 256, style: SummaryStyle::Terse },
        origin: Origin::Installed { skill: def.skill_id },
        mcp_servers: m.mcp_servers.clone(),
    };
    // Bundled assets → manifest.assets (declared) + package payload (placed into
    // the agent's install folder by `place_agent_home`).
    let mut asset_meta: Vec<Asset> = def
        .assets
        .iter()
        .map(|(name, content)| Asset {
            name: (*name).to_string(),
            store_ref: StoreKey(alloc::format!("/agent/{}/assets/{name}", def.agent_id.0)),
            bytes: content.len() as u32,
        })
        .collect();
    for (name, content) in def.binary_assets {
        asset_meta.push(Asset {
            name: (*name).to_string(),
            store_ref: StoreKey(alloc::format!("/agent/{}/assets/{name}", def.agent_id.0)),
            bytes: content.len() as u32,
        });
    }
    let mut asset_payload: Vec<(String, Vec<u8>)> = def
        .assets
        .iter()
        .map(|(name, content)| ((*name).to_string(), content.as_bytes().to_vec()))
        .collect();
    for (name, content) in def.binary_assets {
        asset_payload.push(((*name).to_string(), content.to_vec()));
    }
    let manifest = SkillManifest {
        schema_version: 2,
        id: def.skill_id,
        name: m.name.clone(),
        version: m.version.clone(),
        description: m.description.clone(),
        kind: SkillKind::SkillAgent,
        requested_capabilities: caps.to_vec(),
        body_ref: StoreKey(alloc::format!("skills/{}/body.md", def.skill_id.0)),
        bundled_tools: Vec::new(),
        assets: asset_meta,
        agent: Some(agent),
        soul_ref: Some(StoreKey(alloc::format!("/agent/{}/SOUL.md", def.agent_id.0))),
        skill_docs: Vec::new(),
        signature: SignatureBlock {
            algo: SigAlgo::Ed25519,
            key_id: crate::skills::crypto::REGISTRY_KEY_ID.to_string(),
            content_hash: [0u8; 32],
            sig: Vec::new(),
        },
    };
    let mut pkg = SkillPackage {
        manifest,
        body: alloc::format!("System {} agent.", m.name),
        soul: Some(def.soul.to_string()),
        skill_docs: Vec::new(),
        assets: asset_payload,
    };
    pkg.sign();
    pkg
}

/// Install every built-in system agent (idempotent per boot): sign its package
/// and install it granting its full declared (home-resolved) capability set
/// (system agents are pre-trusted), landing its SOUL + assets in `/agent/<id>/`
/// and registering its role. Called once from `run_os` after the FS + net are up.
/// Agents with `autostart: true` are then activated via [`autostart_agents`].
pub fn install_all(now: Ticks) {
    // One on-disk flush for the whole roster, not one per file. The ext4 store's
    // sync re-formats the partition and rewrites every file, so installing ~40
    // agents (~120 files, several MB once the wasm assets are counted) file by file
    // is quadratic — it took *minutes* on a real disk, while a diskless guest kept
    // everything in memory and showed nothing.
    crate::synapse::fs::begin_batch();
    // The user home (`~`, `/home/chitti`) must exist as a directory before the
    // shell starts in it — on a diskless memfs boot and on a fresh install's
    // empty data partition alike. Idempotent; never touches user files.
    crate::agent::home::ensure_user_home();
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            crate::ktrace::log_fmt(format_args!("system-agent: '{}' manifest.json failed to parse", def.name));
            continue;
        };
        let caps = resolved_caps(def, &m);
        let pkg = build_package(def, &m, &caps);
        match crate::skills::install::install(&pkg, &caps, "system", InstallSource::BootModule { name: alloc::format!("system:{}", def.name) }, now) {
            Ok(rec) => crate::ktrace::log_fmt(format_args!(
                "system-agent: installed '{}' -> /agent/{} ({} caps, {} asset(s))",
                def.name,
                def.agent_id.0,
                rec.granted_capabilities.len(),
                def.assets.len() + def.binary_assets.len()
            )),
            Err(e) => crate::ktrace::log_fmt(format_args!("system-agent: install '{}' failed: {:?}", def.name, e)),
        }
    }
    // Flush before autostart: the service agents read their own files back, and a
    // crash after this point should leave the roster on disk.
    crate::synapse::fs::end_batch();
    crate::serial_println!(
        "Chitti: system agents installed (builtin suite + UI apps + chat agents) in /agent/"
    );
    autostart_agents();
}

/// Names of system packages whose manifest sets `"autostart": true`.
pub fn autostart_names() -> Vec<&'static str> {
    let mut out = Vec::new();
    for def in SYSTEM_AGENTS {
        if let Some(m) = parse_manifest(def.manifest_json) {
            if m.autostart {
                out.push(def.name);
            }
        }
    }
    out
}

/// Union of toolset entries from every autostart package (deduped, stable order).
/// Merged into the shell orchestrator toolset so chat can use download / notes /
/// todos without `/agents start`.
pub fn autostart_toolset() -> Vec<String> {
    let mut out = Vec::new();
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        if !m.autostart {
            continue;
        }
        for t in m.toolset {
            if !out.iter().any(|x| x == &t) {
                out.push(t);
            }
        }
    }
    out
}

/// Resolve which system package owns a tool name (for WASM export dispatch).
/// Prefers agents that ship `tools.wasm` so notes/paint/… tools hit the right
/// home even when the chat session is still the shell orchestrator.
pub fn owner_agent_for_tool(tool: &str) -> Option<u64> {
    let mut fallback = None;
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        if !m.toolset.iter().any(|t| t == tool) {
            continue;
        }
        let has_wasm = def.binary_assets.iter().any(|(n, _)| *n == "tools.wasm");
        if has_wasm {
            return Some(def.agent_id.0);
        }
        if fallback.is_none() {
            fallback = Some(def.agent_id.0);
        }
    }
    fallback
}

/// Activate packages with `autostart: true` after install: ensure homes exist
/// and log the set. Tools are merged into the shell via [`autostart_toolset`]
/// when the chat binds to the orchestrator — no service loop is spawned
/// (skill agents are request/response, not daemons).
pub fn autostart_agents() {
    let names = autostart_names();
    if names.is_empty() {
        return;
    }
    for def in SYSTEM_AGENTS {
        let Some(m) = parse_manifest(def.manifest_json) else {
            continue;
        };
        if !m.autostart {
            continue;
        }
        crate::agent::home::ensure(def.agent_id.0, def.name);
        crate::ktrace::log_fmt(format_args!(
            "system-agent: autostart '{}' (id {}, {} tools)",
            def.name,
            def.agent_id.0,
            m.toolset.len()
        ));
    }
    crate::serial_println!(
        "Chitti: autostart agents ready: {} (tools merged into shell)",
        names.join(", ")
    );
}

/// The installed system agents, for `/agents list`/display: (name, agent_id).
pub fn list() -> Vec<(&'static str, u64)> {
    SYSTEM_AGENTS.iter().map(|d| (d.name, d.agent_id.0)).collect()
}

/// How an installed package presents itself in the `/agents` browser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentUiClass {
    /// Canvas package UI (`tools.wasm` + `ui_surface_*` / Ui cap) — chess, paint, …
    UiCanvas,
    /// Chat/SOUL agent (notes, download, librarian, …) or content/service package.
    Shell,
}

/// True when a parsed package manifest describes a **package-UI** canvas app.
///
/// Preference order (no name hardcoding):
/// 1. Explicit `"package_ui": true|false` in the manifest (preferred).
/// 2. Fallback inference: `wasm.module` is tools.wasm **and** `Ui` has `EXEC`.
///
/// Chat agents that only *call* Ui tools (browser/media) omit `package_ui` and
/// lack `wasm.module`. Wasm chat tools (notes/pdf) and content services (doc)
/// omit `package_ui` and lack `Ui` EXEC. Pure — unit-tested.
pub fn is_package_ui_manifest(m: &ParsedManifest) -> bool {
    if let Some(flag) = m.package_ui {
        return flag;
    }
    let has_tools_wasm = m.wasm_module.as_ref().is_some_and(|p| {
        p.ends_with("tools.wasm") || p.contains("/tools.wasm")
    });
    let has_ui_surface = m.capabilities.iter().any(|c| {
        c.domain == CapDomain::Ui && c.rights.contains(Rights::EXEC)
    });
    has_tools_wasm && has_ui_surface
}

/// True when `name` is a **package-UI** app (`/agents start` → `package_ui::start`).
/// Derived from the package's `manifest.json` via [`is_package_ui_manifest`] —
/// not a hardcoded name list.
pub fn is_package_ui_app(name: &str) -> bool {
    let Some(def) = SYSTEM_AGENTS.iter().find(|d| d.name == name) else {
        return false;
    };
    parse_manifest(def.manifest_json)
        .map(|m| is_package_ui_manifest(&m))
        .unwrap_or(false)
}

/// Classify a system agent as **UI (canvas)** vs **shell** for the agents popup.
/// Only apps that `package_ui::start` can run are `UiCanvas` — so Enter always
/// does something that matches the badge.
pub fn ui_class(name: &str) -> AgentUiClass {
    if is_package_ui_app(name) {
        AgentUiClass::UiCanvas
    } else {
        AgentUiClass::Shell
    }
}

/// Short badge for the agents browser right column (ASCII — safe for all fonts).
pub fn ui_class_label(c: AgentUiClass) -> &'static str {
    match c {
        AgentUiClass::UiCanvas => "ui/canvas",
        AgentUiClass::Shell => "shell",
    }
}

/// One-line description from the package manifest (may be empty).
pub fn description_of(name: &str) -> String {
    let Some(def) = SYSTEM_AGENTS.iter().find(|d| d.name == name) else {
        return String::new();
    };
    parse_manifest(def.manifest_json)
        .map(|m| m.description)
        .unwrap_or_default()
}

/// Human-readable command-hook summary for agent `name` (e.g. `"/open media"`),
/// or empty if the package declares none.
pub fn command_hook_summary(name: &str) -> String {
    let Some(def) = SYSTEM_AGENTS.iter().find(|d| d.name == name) else {
        return String::new();
    };
    let Some(m) = parse_manifest(def.manifest_json) else {
        return String::new();
    };
    if m.command_hooks.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = m
        .command_hooks
        .iter()
        .map(|h| {
            if h.command == "/open" {
                alloc::format!("{} media", h.command)
            } else {
                h.command.clone()
            }
        })
        .collect();
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_the_bundled_doc_manifest() {
        let doc = SYSTEM_AGENTS.iter().find(|d| d.name == "doc").expect("doc agent");
        let m = parse_manifest(doc.manifest_json).expect("doc manifest parses");
        assert_eq!(m.name, "doc");
        assert_eq!(m.kind, AgentKind::Service);
        // The Doc agent reads files (channel + fs read), scoped to its home.
        assert!(m.capabilities.iter().any(|c| c.domain == CapDomain::Fs && c.rights.contains(Rights::READ)));
    }

    #[test_case]
    fn ui_class_splits_canvas_and_shell() {
        assert_eq!(ui_class("chess"), AgentUiClass::UiCanvas);
        assert_eq!(ui_class("paint"), AgentUiClass::UiCanvas);
        assert_eq!(ui_class("snake"), AgentUiClass::UiCanvas);
        assert_eq!(ui_class("synth"), AgentUiClass::UiCanvas);
        assert_eq!(ui_class("notes"), AgentUiClass::UiCanvas);
        assert_eq!(ui_class("download"), AgentUiClass::Shell);
        assert_eq!(ui_class("doc"), AgentUiClass::Shell);
        // Have Ui caps / tools but are chat agents, not package_ui canvases.
        assert_eq!(ui_class("browser"), AgentUiClass::Shell);
        assert_eq!(ui_class("media"), AgentUiClass::Shell);
        assert!(is_package_ui_app("chess"));
        assert!(!is_package_ui_app("browser"));
        // Every system package classified from its own manifest, not a name list.
        for def in SYSTEM_AGENTS {
            let m = parse_manifest(def.manifest_json).expect(def.name);
            assert_eq!(
                is_package_ui_app(def.name),
                is_package_ui_manifest(&m),
                "{} classification must match its manifest",
                def.name
            );
        }
    }

    #[test_case]
    fn package_ui_manifest_prefers_explicit_flag() {
        let mut m = parse_manifest(
            SYSTEM_AGENTS
                .iter()
                .find(|d| d.name == "chess")
                .unwrap()
                .manifest_json,
        )
        .unwrap();
        assert_eq!(m.package_ui, Some(true));
        assert!(is_package_ui_manifest(&m));
        // Explicit false wins over wasm + Ui.
        m.package_ui = Some(false);
        assert!(!is_package_ui_manifest(&m));
        // Explicit true without inferable fields still counts.
        m.package_ui = Some(true);
        m.wasm_module = None;
        m.capabilities.clear();
        assert!(is_package_ui_manifest(&m));
        // Fallback inference when flag absent: need wasm + Ui EXEC.
        m.package_ui = None;
        assert!(!is_package_ui_manifest(&m));
        m.wasm_module = Some(String::from("assets/tools.wasm"));
        m.capabilities.push(CapabilityRequest::new(
            CapDomain::Ui,
            Rights::EXEC | Rights::WRITE | Rights::DELETE,
            Scope::Any,
        ));
        assert!(is_package_ui_manifest(&m));
        // Notes: package UI list/reader + wasm tools (autostart chat tools too).
        let notes = parse_manifest(
            SYSTEM_AGENTS
                .iter()
                .find(|d| d.name == "notes")
                .unwrap()
                .manifest_json,
        )
        .unwrap();
        assert_eq!(notes.package_ui, Some(true));
        assert!(notes.wasm_module.is_some());
        assert!(is_package_ui_manifest(&notes));
        // Browser: Ui but no guest tools.wasm / no package_ui flag.
        let browser = parse_manifest(
            SYSTEM_AGENTS
                .iter()
                .find(|d| d.name == "browser")
                .unwrap()
                .manifest_json,
        )
        .unwrap();
        assert!(browser.package_ui.is_none());
        assert!(browser.wasm_module.is_none());
        assert!(!is_package_ui_manifest(&browser));
        // Every canvas system package declares the flag.
        for def in SYSTEM_AGENTS {
            let m = parse_manifest(def.manifest_json).expect(def.name);
            if is_package_ui_app(def.name) {
                assert_eq!(
                    m.package_ui,
                    Some(true),
                    "{} should declare package_ui: true",
                    def.name
                );
            }
        }
    }

    #[test_case]
    fn all_system_manifests_parse_and_declare_caps() {
        for def in SYSTEM_AGENTS {
            let m = parse_manifest(def.manifest_json).unwrap_or_else(|| panic!("{} manifest", def.name));
            assert_eq!(m.name, def.name);
            assert!(!m.capabilities.is_empty(), "{} declares capabilities", def.name);
        }
        // SOUL + package agents (no network/http plumbing as agents).
        // 14 original + 16 UI + 10 chat + 6 more (breakout/tetris/console/maps/radio/sandbox-lab) + git.
        assert_eq!(SYSTEM_AGENTS.len(), 47);
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "browser"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "chess"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "media"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "notes"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "snake"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "todo"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "download"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "calc"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "files"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "game2048"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "librarian"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "settings"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "breakout"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "tetris"));
        assert!(SYSTEM_AGENTS.iter().any(|d| d.name == "sandbox-lab"));
        // UI apps ship tools.wasm; chat-only librarian does not.
        let calc = SYSTEM_AGENTS.iter().find(|d| d.name == "calc").unwrap();
        assert!(calc.binary_assets.iter().any(|(n, _)| *n == "tools.wasm"));
        let lib = SYSTEM_AGENTS.iter().find(|d| d.name == "librarian").unwrap();
        assert!(lib.binary_assets.is_empty());
        assert!(!SYSTEM_AGENTS.iter().any(|d| d.name == "network" || d.name == "http"));
    }

    #[test_case]
    fn download_notes_todo_autostart() {
        for name in ["download", "notes", "todo"] {
            let def = SYSTEM_AGENTS.iter().find(|d| d.name == name).expect(name);
            let m = parse_manifest(def.manifest_json).expect("manifest");
            assert!(m.autostart, "{name} must autostart");
        }
        let names = autostart_names();
        assert!(names.contains(&"download"));
        assert!(names.contains(&"notes"));
        assert!(names.contains(&"todo"));
        let tools = autostart_toolset();
        assert!(tools.iter().any(|t| t == "download"));
        assert!(tools.iter().any(|t| t == "notes_list"));
        assert!(tools.iter().any(|t| t == "todo_write"));
    }

    #[test_case]
    fn owner_agent_for_notes_tools_is_notes_package() {
        let notes = SYSTEM_AGENTS.iter().find(|d| d.name == "notes").unwrap();
        assert_eq!(owner_agent_for_tool("notes_list"), Some(notes.agent_id.0));
        assert_eq!(owner_agent_for_tool("notes_set"), Some(notes.agent_id.0));
        assert!(owner_agent_for_tool("no_such_tool_xyz").is_none());
    }

    #[test_case]
    fn media_manifest_declares_open_command_hook() {
        let media = SYSTEM_AGENTS.iter().find(|d| d.name == "media").expect("media");
        let m = parse_manifest(media.manifest_json).expect("media manifest");
        assert!(
            !m.command_hooks.is_empty(),
            "media package must declare command_hooks"
        );
        let open = m
            .command_hooks
            .iter()
            .find(|h| h.command == "/open")
            .expect("/open hook");
        assert!(open.extensions.iter().any(|e| e == ".mp3"));
        assert_eq!(
            open.dispatch.get(".mp3").map(|d| d.tool.as_str()),
            Some("audio_player")
        );
        assert_eq!(
            open.dispatch.get(".png").map(|d| d.tool.as_str()),
            Some("draw_image")
        );
        assert_eq!(
            open.dispatch.get(".mp4").map(|d| d.tool.as_str()),
            Some("video_player")
        );
    }

    #[test_case]
    fn resolve_open_hook_routes_media_extensions() {
        let h = resolve_open_hook("/downloads/sample.mp3").expect("mp3");
        assert_eq!(h.agent_name, "media");
        assert_eq!(h.tool, "audio_player");
        assert_eq!(h.path_arg, "path");
        let h = resolve_open_hook("photo.JPEG").expect("jpeg");
        assert_eq!(h.tool, "draw_image");
        let h = resolve_open_hook("clip.webm").expect("webm");
        assert_eq!(h.tool, "video_player");
        assert!(resolve_open_hook("notes.txt").is_none());
        assert!(resolve_open_hook("noext").is_none());
    }

    #[test_case]
    fn path_extension_extracts_last_suffix() {
        assert_eq!(path_extension("/a/b/c.mp3"), ".mp3");
        assert_eq!(path_extension("Foo.PNG"), ".png");
        assert_eq!(path_extension("nope"), "");
        assert_eq!(path_extension(".hidden"), "");
    }

    #[test_case]
    fn doc_home_scope_resolves_to_the_agent_folder() {
        let doc = SYSTEM_AGENTS.iter().find(|d| d.name == "doc").unwrap();
        let m = parse_manifest(doc.manifest_json).unwrap();
        let caps = resolved_caps(doc, &m);
        // The doc agent's Fs READ scope is its own install folder, not "$HOME".
        let fs = caps.iter().find(|c| c.domain == CapDomain::Fs).expect("doc has an Fs cap");
        assert_eq!(fs.rights, Rights::READ);
        match &fs.scope {
            Scope::Path(p) => assert_eq!(p, &alloc::format!("/agent/{}/**", doc.agent_id.0)),
            other => panic!("expected a resolved path scope, got {other:?}"),
        }
    }

    #[test_case]
    fn parses_scope_variants() {
        assert_eq!(parse_scope("any"), Scope::Any);
        assert_eq!(parse_scope("path:/work/**"), Scope::Path("/work/**".into()));
        assert_eq!(parse_scope("net:*.example.com:80-443"), Scope::Net { host: "*.example.com".into(), port_lo: 80, port_hi: 443 });
    }
}
