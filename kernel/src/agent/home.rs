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
pub fn ensure(id: u64, name: &str) {
    let home = path(id);
    let soul = format!("{}/SOUL.md", home);
    if !fs::exists(&soul) {
        let default = format!(
            "# SOUL — {name}\n\n\
             I am {name}, an agent of Chitti OS (agent id {id}).\n\n\
             This file is my persona: it is prepended to my system prompt every\n\
             session. Edit it to change how I behave — tone, priorities, standing\n\
             instructions. My skills live in {home}/skills/, my durable notes in\n\
             {home}/memory/ (I read and write them with my fs tools).\n"
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
