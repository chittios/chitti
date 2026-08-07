//! Persistence for the login credential — **the only code allowed to touch
//! [`super::PATH`]**.
//!
//! Every function here takes a [`crate::synapse::fs::CredentialAccess`] guard for
//! the duration of its store call. That guard is the sole hole in the facade
//! refusal, which is why this module is small and does nothing else: the smaller
//! the set of code that can reach the record, the smaller the review surface.
//!
//! Reads also refresh the cached policy ([`super::set_cached`]) so the idle poll
//! never has to hit the store.

use super::{Credential, DEFAULT_IDLE_LOCK_MINUTES, DEFAULT_ITERATIONS, HASH_LEN, KDF_PBKDF2_SHA256, PATH, SALT_LEN, VERSION};
use crate::synapse::fs;
use alloc::string::String;

/// Whether a record exists on the store.
///
/// `fs::exists` is deliberately *not* refused for this path — `/passwd status`
/// needs it, and whether a machine has a password is already observable from the
/// fact that it asks for one at boot.
pub fn exists() -> bool {
    fs::exists(PATH)
}

/// Read and parse the record. `None` when absent **or malformed** — see
/// [`Credential::from_json`] for why a malformed record must not become a
/// default one.
pub fn load() -> Option<Credential> {
    let rec = {
        let _access = fs::CredentialAccess::new();
        fs::read(PATH)
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|s| crate::json::Json::parse(&s))
            .as_ref()
            .and_then(Credential::from_json)
    };
    super::set_cached(rec.as_ref());
    rec
}

/// Refresh the cached policy from the store. Called once at boot, before
/// anything asks [`super::enrolled`].
///
/// A record that is present but **unparseable** is reported, loudly: it means
/// the machine will refuse to unlock (`verify` cannot succeed against a record
/// that does not exist in memory) and the human needs to know why rather than
/// discovering it at the prompt.
pub fn refresh() {
    if exists() && load().is_none() {
        crate::ktrace::log(
            "auth",
            "the login credential record exists but does not parse -- login is DISABLED; \
             boot another OS and delete /configs/core/auth.json to recover",
        );
    }
}

/// Write the record.
pub fn save(rec: &Credential) -> Result<(), &'static str> {
    let text = rec.to_json().to_pretty();
    {
        let _access = fs::CredentialAccess::new();
        fs::write(PATH, text.as_bytes());
    }
    // Read back rather than trusting the write: on a full or failing store
    // `fs::write` is infallible by signature, and silently enrolling a password
    // that was never stored would lock the human out of nothing while they
    // believe otherwise.
    let back = {
        let _access = fs::CredentialAccess::new();
        fs::read(PATH)
    };
    match back {
        Some(b) if b == text.as_bytes() => {
            super::set_cached(Some(rec));
            Ok(())
        }
        _ => Err("the credential record could not be written to the store"),
    }
}

/// Remove the record — disables the gate entirely.
pub fn clear() -> Result<(), &'static str> {
    let removed = {
        let _access = fs::CredentialAccess::new();
        fs::delete(PATH)
    };
    super::set_cached(None);
    if removed {
        Ok(())
    } else {
        Err("no password was set")
    }
}

/// Build a record for `password`, drawing a fresh salt.
///
/// The salt comes from [`crate::security::rng`], never `arch::hw_rand()` — see
/// that module for why the obvious version yields all zeros under QEMU.
pub fn enrol(password: &str, pump: &mut dyn FnMut()) -> Result<Credential, &'static str> {
    super::validate_new(password)?;
    let mut salt = [0u8; SALT_LEN];
    crate::security::rng::fill_random(&mut salt);
    if salt.iter().all(|&b| b == 0) {
        // Cannot happen with a working CSPRNG; refusing beats enrolling against
        // a zero salt if one ever does.
        return Err("could not draw a random salt");
    }
    let hash = super::derive_pumped(password, &salt, DEFAULT_ITERATIONS, pump);
    debug_assert_eq!(hash.len(), HASH_LEN);
    Ok(Credential {
        version: VERSION,
        user: crate::agent::home::USER_NAME.into(),
        kdf: KDF_PBKDF2_SHA256.into(),
        iterations: DEFAULT_ITERATIONS,
        salt,
        hash,
        created_ms: crate::arch::now_ms(),
        idle_lock_minutes: DEFAULT_IDLE_LOCK_MINUTES,
        lock_on_resume: true,
        failed_total: 0,
        last_failed_ms: 0,
    })
}

/// Record a failed attempt in the durable counter.
///
/// Best-effort: a store that will not take the write must not stop the gate
/// working, so the error is dropped. The *consecutive* count that drives the
/// backoff lives in RAM ([`super::note_failure`]) and is what actually defends
/// the prompt — this is only the lifetime tally `/passwd status` reports.
pub fn note_failed_attempt() {
    if let Some(mut rec) = load() {
        rec.failed_total = rec.failed_total.saturating_add(1);
        rec.last_failed_ms = crate::arch::now_ms();
        let _ = save(&rec);
    }
}
