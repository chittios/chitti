//! The permissioned **install** flow (`CHITTI_AGENTIC_HANDOFF.md` Phase G):
//! verify a package's signature/hash against the trust store, present its
//! requested capabilities to the human for consent, and on approval register it
//! granting only the approved subset. Phase F places skills directly as trusted
//! ([`package::place_trusted`](crate::skills::package)); this module adds the
//! verification + consent that a real install requires.
//!
//! Filled in during Phase G.

// (Phase G) — see the git history for the incremental build.
