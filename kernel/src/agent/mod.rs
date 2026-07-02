//! **Agent** — the Claude-Code-style agent layer (`CHITTI_AGENTIC_HANDOFF.md`).
//! Replaces the flat `persona` model with an orchestrator that runs a tool-use
//! loop over a first-class [`tool`](crate::tools) layer, dispatches isolated
//! sub-agents, and persists to first-class [`session`](crate::session)s.
//!
//! This module owns the shared type contract ([`types`], from
//! `CHITTI_SCHEMAS.md`) — `AgentManifest`, `Session`, `SkillManifest`, and the
//! Part-0 primitives (identifiers, `Provenance`, the capability model). The
//! agentic machinery (loop, orchestrator, sub-agents) is layered on in Phase A+.
//!
//! Everything above the determinism boundary lives here; every *effect* still
//! flows down through Synapse (locked invariant #1).

pub mod types;

pub use types::*;
