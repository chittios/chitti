//! **Per-agent home directories** on the filesystem: every agent gets
//! `/agent/<id>/` holding its **SOUL.md** (persona — prepended to the agent's
//! system prompt), a **skills/** area (agent-local skill state), and a
//! **memory/** area (durable notes the agent reads/writes with its fs tools).
//! The store is flat (path → bytes), so the "directories" are path prefixes
//! seeded with a `.keep` marker; on an installed system these persist on ext4
//! like the rest of `synapse::fs`.

use crate::synapse::fs;
use alloc::format;
use alloc::string::String;

/// The home path for agent `id` (no trailing slash).
pub fn path(id: u64) -> String {
    format!("/agent/{}", id)
}

/// Ensure agent `id`'s home exists: `SOUL.md` (seeded with a default persona on
/// first boot), `skills/` and `memory/` markers. Idempotent; cheap when present.
/// A SOUL.md placed by an installed package (see `skills::package::place_agent_home`)
/// already exists here, so `ensure` never overwrites a packaged persona.
pub fn ensure(id: u64, name: &str) {
    let home = path(id);
    let soul = format!("{}/SOUL.md", home);
    if !fs::exists(&soul) {
        // A concise *persona* the model adopts — not documentation about the
        // file (a chatty meta description gets parroted back as an answer).
        let default = format!(
            "You are {name}, the shell agent of Chitti OS. You are concise, direct, and \
             practical. You operate the machine through tools and answer in plain prose."
        );
        fs::write(&soul, default.as_bytes());
        crate::ktrace::log_fmt(format_args!("agent.home: created {} (SOUL.md + skills/ + memory/)", home));
    }
    for marker in [format!("{}/skills/.keep", home), format!("{}/memory/.keep", home)] {
        if !fs::exists(&marker) {
            fs::write(&marker, b"");
        }
    }
}

/// Agent `id`'s SOUL.md contents, if present (lossy UTF-8).
pub fn soul(id: u64) -> Option<String> {
    fs::read(&format!("{}/SOUL.md", path(id))).map(|b| String::from_utf8_lossy(&b).into_owned())
}
