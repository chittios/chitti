//! Provenance / taint tags (`CHITTI_OS_HANDOFF.md` Phase 6, feature 1).
//!
//! Every piece of content an agent reasons over carries a [`Provenance`]:
//! where it came from, and therefore how much it may be trusted to *justify*
//! a privileged action. This is the primitive the Synapse taint gate is built
//! on -- the OS-boundary defence against prompt-injection-as-privilege-
//! escalation: an instruction that arrived inside untrusted, ingested content
//! must not be able to talk the agent into a destructive primitive, no matter
//! how it is phrased.
//!
//! This module is deliberately dependency-free (just an enum + a struct) so
//! both the layer that *tags* content (`persona::memory`) and the layer that
//! *gates* on it (`synapse::executor`) can depend on it without a cycle.

/// Where a token / message / value originated, ordered by how dangerous it is
/// as the justification for a privileged action (least -> most dangerous).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// The system / persona prompt: fully trusted, kernel-authored.
    SystemTrusted,
    /// Typed by the human at the shell: trusted as an explicit instruction.
    UserTyped,
    /// Read from the FS or returned by a tool: **untrusted**. Anything an
    /// agent ingests from outside itself lands here, and stays here.
    UntrustedIngested,
}

impl Provenance {
    /// Ordinal used to combine provenances: higher = less trusted. Taint is
    /// contagious, so combining always takes the *worse* of two.
    fn rank(self) -> u8 {
        match self {
            Provenance::SystemTrusted => 0,
            Provenance::UserTyped => 1,
            Provenance::UntrustedIngested => 2,
        }
    }

    /// The less-trusted (more tainted) of two provenances. Used to fold a
    /// whole context down to the worst thing in it.
    pub fn join(self, other: Provenance) -> Provenance {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }

    /// Whether this provenance is untrusted ingested content -- the case the
    /// destructive-primitive gate refuses.
    pub fn is_tainted(self) -> bool {
        matches!(self, Provenance::UntrustedIngested)
    }
}

/// The justification an agent presents to Synapse for a call: the combined
/// provenance of the context that led to it, plus whether a human has
/// explicitly confirmed this specific action at the shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Justification {
    pub provenance: Provenance,
    /// Set only by an explicit human confirmation at the shell. It is the one
    /// thing that lets a tainted justification through the destructive gate.
    pub human_confirmed: bool,
}

impl Justification {
    /// A fully-trusted justification: what system/kernel-internal callers use,
    /// and the default for `synapse::execute` (preserving pre-Phase-6
    /// behaviour where every caller was implicitly trusted).
    pub const fn trusted() -> Self {
        Self { provenance: Provenance::SystemTrusted, human_confirmed: false }
    }

    /// Justify a call by the provenance of the context that produced it.
    pub const fn from_context(provenance: Provenance) -> Self {
        Self { provenance, human_confirmed: false }
    }

    /// Mark this justification as explicitly human-confirmed at the shell.
    pub const fn confirmed(mut self) -> Self {
        self.human_confirmed = true;
        self
    }

    /// Whether this justification must be refused for a *destructive*
    /// primitive: tainted content that no human has confirmed.
    pub fn blocks_destructive(&self) -> bool {
        self.provenance.is_tainted() && !self.human_confirmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn join_takes_the_worse_provenance() {
        use Provenance::*;
        assert_eq!(SystemTrusted.join(UserTyped), UserTyped);
        assert_eq!(UserTyped.join(UntrustedIngested), UntrustedIngested);
        assert_eq!(UntrustedIngested.join(SystemTrusted), UntrustedIngested);
        assert_eq!(SystemTrusted.join(SystemTrusted), SystemTrusted);
    }

    #[test_case]
    fn only_unconfirmed_tainted_blocks_destructive() {
        assert!(Justification::from_context(Provenance::UntrustedIngested).blocks_destructive());
        assert!(!Justification::from_context(Provenance::UserTyped).blocks_destructive());
        assert!(!Justification::from_context(Provenance::SystemTrusted).blocks_destructive());
        // Human confirmation overrides taint.
        assert!(!Justification::from_context(Provenance::UntrustedIngested).confirmed().blocks_destructive());
        assert!(!Justification::trusted().blocks_destructive());
    }
}
