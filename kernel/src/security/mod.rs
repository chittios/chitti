//! **Security** -- the Phase 6 differentiators' shared substrate
//! (`CHITTI_OS_HANDOFF.md` Phase 6).
//!
//! Today this is the provenance/taint vocabulary ([`taint`]) that the Synapse
//! taint gate and the Persona memory tagger both build on. It is a low-level,
//! dependency-free module on purpose: everything above the determinism
//! boundary tags content with a [`taint::Provenance`], and Synapse -- at the
//! boundary -- gates destructive primitives on the [`taint::Justification`]
//! that content adds up to.
//!
//! The other Phase 6 feature, self-compiling agents (compiled intents), lives
//! with the runtime it caches in `crate::persona::compiled`.

//! [`redteam`] is the evaluation half of the same story: an attack corpus, a
//! benign-task suite, and weaker baselines, run against the real tool router so
//! the policy is measured where it is actually enforced rather than where it is
//! defined.

//! [`rng`] is the third piece: one seeding path for every byte of key material
//! in the kernel, because the obvious two-line version silently degrades to all
//! zeros on any machine without `RDRAND`/`RNDR` -- which is every emulator we
//! develop and test on.

pub mod citation;
pub mod redteam;
pub mod rng;
pub mod taint;

pub use taint::{Justification, Provenance};
