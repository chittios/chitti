//! **Command exit status**, so `&&` and `||` have something real to branch on.
//!
//! `dispatch_system` returns `bool` meaning *"was this command name handled"*,
//! not *"did it succeed"*, and command handlers return `()`. Rather than change
//! ~150 signatures, a handler reports failure by calling [`fail`] on its error
//! paths and the runner reads [`get`] after the stage.
//!
//! ## The honest limitation, stated because it is silent otherwise
//!
//! A handler that has **not** been converted never calls [`fail`], so it always
//! reports success — and `/somecmd-that-failed && /ls` would run `/ls`. That is
//! plausible and wrong, which is the worst kind of bug, so the converted set is
//! **enumerated** in [`REPORTS_STATUS`] and pinned by a test rather than left as
//! folklore. `/help pipeline` prints it, and `&&`/`||` after an unconverted
//! command warn once rather than quietly implying a check happened.
//!
//! Grow the list deliberately: add the `fail()` calls, then add the name here.

use core::sync::atomic::{AtomicI32, Ordering};

/// Status of the most recently run stage. 0 is success.
static STATUS: AtomicI32 = AtomicI32::new(0);

/// No such command — `sh` uses 127 and so do we, so a missing command is
/// distinguishable from a command that ran and failed.
pub const NOT_FOUND: i32 = 127;
/// A command that ran but reported a problem, when it has nothing finer to say.
pub const FAILURE: i32 = 1;

/// Clear the status. The runner calls this before each stage, so a stale
/// failure cannot make the *next* command look broken.
pub fn reset() {
    STATUS.store(0, Ordering::Relaxed);
}

/// Report failure from a command handler.
///
/// Clamped away from 0: a handler calling `fail(0)` means "I failed" and must
/// never be readable as success.
pub fn fail(code: i32) {
    STATUS.store(if code == 0 { FAILURE } else { code }, Ordering::Relaxed);
}

/// Shorthand for the ordinary "this did not work" case.
pub fn fail1() {
    fail(FAILURE);
}

/// Status of the last stage; 0 if it did not report a failure.
pub fn get() -> i32 {
    STATUS.load(Ordering::Relaxed)
}

/// Whether the last stage succeeded.
pub fn ok() -> bool {
    get() == 0
}

/// Commands whose handlers call [`fail`] on their error paths, so `&&` and
/// `||` after them mean what they say.
///
/// Everything absent from this list always reports success. Keep it sorted and
/// keep it truthful — a name here whose handler never calls `fail` is worse
/// than an absent one, because it claims a check that is not happening.
pub const REPORTS_STATUS: &[&str] = &[
    "cat", "cd", "cp", "glob", "grep", "head", "ls", "mkdir", "mv", "pbcopy", "rm", "tail",
    "touch",
];

/// Whether `name`'s failures are visible to `&&` / `||`.
pub fn reports_status(name: &str) -> bool {
    REPORTS_STATUS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn failure_is_never_readable_as_success() {
        reset();
        assert!(ok());
        // A handler calling fail(0) means "I failed"; honouring the 0 literally
        // would make it report success, which is the one thing it must not do.
        fail(0);
        assert!(!ok());
        assert_eq!(get(), FAILURE);
        fail(NOT_FOUND);
        assert_eq!(get(), NOT_FOUND);
        reset();
        assert_eq!(get(), 0);
    }

    #[test_case]
    fn the_reporting_set_is_sorted_and_free_of_duplicates() {
        // It is read by a human deciding whether to trust `&&` after a command,
        // so it must be scannable and must not claim a name twice.
        let mut sorted = REPORTS_STATUS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), REPORTS_STATUS, "REPORTS_STATUS must be sorted and unique");
    }

    #[test_case]
    fn every_reporting_command_is_a_real_command() {
        // A name here that no longer exists would silently promise a status
        // check for a command nobody can run.
        for name in REPORTS_STATUS {
            assert!(
                crate::shell::catalog::is_command_name(name),
                "REPORTS_STATUS lists '{name}', which is not a command"
            );
        }
    }

    #[test_case]
    fn a_command_outside_the_set_is_reported_as_not_reporting() {
        // This is what lets the runner warn instead of implying a check that is
        // not happening.
        assert!(reports_status("cat"));
        assert!(!reports_status("mounts"));
        assert!(!reports_status("http"));
    }
}
