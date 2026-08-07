//! **Single-user login** for the fixed user `chitti` — the console lock.
//!
//! ChittiOS is single-user by construction ([`crate::agent::home::USER_HOME`] is
//! a constant, not a per-login path), so this is one password, one record, and no
//! account model. It gates the shell REPL at boot, `/lock` on demand, an idle
//! timeout, and resume from suspend.
//!
//! This module is **pure**: the record, its JSON codec, the KDF, the comparison,
//! the backoff curve and the idle arithmetic, all `#[test_case]`-testable. The
//! blocking prompt that draws a screen and reads keystrokes lives in
//! [`prompt`], which is `#[cfg(not(test))]` — the standing split, because a test
//! written inside a module the test build excludes is silent dead code.
//!
//! # What this protects, and what it does not
//!
//! It is a **console lock, not confidentiality.** Nine things a reader would
//! otherwise assume are handled:
//!
//! 1. **On an unencrypted disk the gate is bypassable offline in minutes** —
//!    mount the ext4 partition elsewhere, delete the record, boot. Only
//!    `/encrypt` ([`crate::block::volcrypto`]) makes the data private, and this
//!    password is deliberately *independent* of that passphrase, so login without
//!    volume encryption is a lock on the door of a glass house.
//! 2. **On a memfs store nothing persists** (live ISO, dev boot). `/passwd` says
//!    so when it enrols one.
//! 3. **PBKDF2-HMAC-SHA256 is GPU-friendly.** No argon2/scrypt/bcrypt exists in
//!    this tree; [`Credential::kdf`] is a string label so a future one is a
//!    migration rather than a silent reinterpretation of the same bytes.
//! 4. **Passwords are printable ASCII only** — enforced at enrolment by
//!    [`validate_new`], because both input paths accept only `0x20..=0x7e` and a
//!    password that cannot be typed again is worse than no password.
//! 5. **`/lock` locks the console, not the machine.** Running services,
//!    listeners and in-flight tasks keep executing. It *does* stop new work — see
//!    the pump note in [`prompt`].
//! 6. **Serial is the same trust domain as the keyboard.** Anyone on the UART can
//!    type at the prompt, matching how the rest of the OS treats serial.
//! 7. **Ctrl+C does not dismiss the gate** — the one documented exception to the
//!    standing "Ctrl+C interrupts every command" rule. A gate you can Ctrl+C out
//!    of is not a gate.
//! 8. **A reboot resets the backoff** ([`backoff_ms`]). The consecutive-failure
//!    count is in RAM by necessity: persisting it means a store write per guess
//!    and lets an attacker brick the machine by guessing. Online guessing is
//!    *slowed*, not stopped.
//! 9. **There is no recovery path.** Forgetting the password on a durable store
//!    means booting another OS and deleting the file. Deliberate — a recovery key
//!    is a second credential to protect — but stated rather than discovered.
//!
//! # Why `/passwd` and `/lock` are not `dispatch_system` arms
//!
//! They are **interactive-only** commands, matched in the REPL above the
//! `dispatch_system` fall-through. That is not stylistic. `run_shell_command`
//! ([`crate::tools::shell_cmd`]) passes *any* `[A-Za-z0-9_-]+` token straight to
//! `dispatch_system` without requiring a registry entry, and it is in `CORE_TOOLS`
//! and the orchestrator manifest — so a `dispatch_system` arm would be callable by
//! every agent, with no capability and no manifest change. There is also no
//! `Right::Shell`: shell commands are not capability-gated at all, so "don't grant
//! the capability" was never available. `dispatch_system` has exactly one
//! agent-reachable call site (`run_tool_command`), and everything else — scheduled
//! fires, Telegram turns, package-UI apps, sub-agents — funnels through it. Keeping
//! the names out of it closes all of them at once.

pub mod prompt;
pub mod store;

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};

/// The one path the whole kernel agrees is the credential record.
///
/// Reachability is denied against this by the Synapse executor **and** by the
/// [`crate::synapse::fs`] facade — two layers, because `/cat`, `/rm`, `/cp`,
/// `/mv`, `/grep` and `/glob` are shell-bound tools that reach the store directly
/// and never enter the executor.
pub const PATH: &str = "/configs/core/auth.json";

pub const SALT_LEN: usize = 32;
pub const HASH_LEN: usize = 32;

/// Domain separation from the volume passphrase: prefixed to the salt so the two
/// KDFs cannot produce the same bytes for the same password even given the same
/// salt. Independence is then structural rather than a claim in a comment
/// (`the_login_kdf_is_domain_separated_from_the_volume_kdf`).
pub const CONTEXT: &[u8] = b"chittios-login-v1:";

/// PBKDF2 rounds for a newly enrolled password. Stored *in* the record, so this
/// can be raised later without invalidating an enrolled password — verification
/// always uses whatever the record says.
pub const DEFAULT_ITERATIONS: u32 = 200_000;

/// The KDF label written into the record. A different value is refused rather
/// than reinterpreted.
pub const KDF_PBKDF2_SHA256: &str = "pbkdf2-hmac-sha256";

pub const MIN_PASSWORD_LEN: usize = 8;
pub const MAX_PASSWORD_LEN: usize = 128;

/// Default idle timeout in minutes; 0 disables auto-lock.
pub const DEFAULT_IDLE_LOCK_MINUTES: u32 = 15;

/// Attempts that are free before the backoff starts.
const FREE_ATTEMPTS: u32 = 3;
/// Backoff ceiling. No permanent lockout: on a single-user OS with no recovery
/// account, a permanent lockout is a brick.
const MAX_BACKOFF_MS: u64 = 30_000;

/// Why the console is being locked. Carried into the prompt so the banner names
/// the cause — "my machine locked itself" otherwise has four indistinguishable
/// causes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Boot,
    Manual,
    Idle,
    Resume,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Boot => "boot",
            Reason::Manual => "manual",
            Reason::Idle => "idle",
            Reason::Resume => "resume",
        }
    }
    fn to_code(self) -> u8 {
        match self {
            Reason::Boot => 1,
            Reason::Manual => 2,
            Reason::Idle => 3,
            Reason::Resume => 4,
        }
    }
    fn from_code(c: u8) -> Option<Reason> {
        match c {
            1 => Some(Reason::Boot),
            2 => Some(Reason::Manual),
            3 => Some(Reason::Idle),
            4 => Some(Reason::Resume),
            _ => None,
        }
    }
}

// ── The record ───────────────────────────────────────────────────────────

/// The stored credential. **Absent file = no password enrolled = no gate
/// anywhere** — one state, tested with one `fs::exists`, and `/passwd clear`
/// removes the file. There is deliberately no "enrolled: false" record.
#[derive(Clone, Debug, PartialEq)]
pub struct Credential {
    pub version: u32,
    pub user: String,
    pub kdf: String,
    pub iterations: u32,
    pub salt: [u8; SALT_LEN],
    pub hash: [u8; HASH_LEN],
    pub created_ms: u64,
    /// 0 disables idle auto-lock.
    ///
    /// This and [`Self::lock_on_resume`] live **here rather than in `ui.json`**:
    /// `ui.json` is agent-writable, so an agent could set the auto-lock timeout
    /// to 0. Policy that protects the credential belongs inside the protected
    /// record.
    pub idle_lock_minutes: u32,
    pub lock_on_resume: bool,
    pub failed_total: u64,
    pub last_failed_ms: u64,
}

pub const VERSION: u32 = 1;

impl Credential {
    /// Parse a record. **`None` on any malformed field** — never a partially
    /// defaulted record: a `Default` fallback here would build a credential that
    /// nothing can match, and a caller might treat it as enrolled. A record that
    /// will not parse locks the console rather than opening it (see [`verify`]).
    pub fn from_json(j: &crate::json::Json) -> Option<Credential> {
        let version = j.get("version")?.as_i64()? as u32;
        if version != VERSION {
            return None;
        }
        let user = j.get("user")?.as_str()?.into();
        let kdf: String = j.get("kdf")?.as_str()?.into();
        if kdf != KDF_PBKDF2_SHA256 {
            return None; // an unknown KDF is refused, never reinterpreted
        }
        let iterations = j.get("iterations")?.as_i64()? as u32;
        if iterations == 0 {
            return None;
        }
        let salt_v = hex_decode(j.get("salt")?.as_str()?)?;
        let hash_v = hex_decode(j.get("hash")?.as_str()?)?;
        if salt_v.len() != SALT_LEN || hash_v.len() != HASH_LEN {
            return None;
        }
        let mut salt = [0u8; SALT_LEN];
        salt.copy_from_slice(&salt_v);
        let mut hash = [0u8; HASH_LEN];
        hash.copy_from_slice(&hash_v);
        // The remaining fields are policy/telemetry, not authentication material,
        // so a missing one takes its default rather than voiding the record.
        let created_ms = j.get("created_ms").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64;
        let idle_lock_minutes = j
            .get("idle_lock_minutes")
            .and_then(|v| v.as_i64())
            .unwrap_or(DEFAULT_IDLE_LOCK_MINUTES as i64)
            .clamp(0, u32::MAX as i64) as u32;
        let lock_on_resume = j.get("lock_on_resume").and_then(|v| v.as_bool()).unwrap_or(true);
        let failed_total = j.get("failed_total").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64;
        let last_failed_ms = j.get("last_failed_ms").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64;
        Some(Credential {
            version,
            user,
            kdf,
            iterations,
            salt,
            hash,
            created_ms,
            idle_lock_minutes,
            lock_on_resume,
            failed_total,
            last_failed_ms,
        })
    }

    pub fn to_json(&self) -> crate::json::Json {
        use crate::json::Json;
        Json::Obj(alloc::vec![
            ("version".into(), Json::Num(self.version as f64)),
            ("user".into(), Json::Str(self.user.clone())),
            ("kdf".into(), Json::Str(self.kdf.clone())),
            ("iterations".into(), Json::Num(self.iterations as f64)),
            ("salt".into(), Json::Str(hex_encode(&self.salt))),
            ("hash".into(), Json::Str(hex_encode(&self.hash))),
            ("created_ms".into(), Json::Num(self.created_ms as f64)),
            ("idle_lock_minutes".into(), Json::Num(self.idle_lock_minutes as f64)),
            ("lock_on_resume".into(), Json::Bool(self.lock_on_resume)),
            ("failed_total".into(), Json::Num(self.failed_total as f64)),
            ("last_failed_ms".into(), Json::Num(self.last_failed_ms as f64)),
        ])
    }
}

// ── Hex ──────────────────────────────────────────────────────────────────

pub fn hex_encode(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 0xf) as usize] as char);
    }
    s
}

/// Decode hex. `None` on odd length or a non-hex nibble — a lenient decoder here
/// would turn a corrupted record into a *different* valid-looking one.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    if b.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 2);
    for pair in b.chunks(2) {
        let hi = nibble(pair[0])?;
        let lo = nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// ── KDF and comparison ───────────────────────────────────────────────────

/// Derive the verifier for `password` under `salt`.
pub fn derive(password: &str, salt: &[u8; SALT_LEN], iterations: u32) -> [u8; HASH_LEN] {
    derive_pumped(password, salt, iterations, &mut || {})
}

/// [`derive`], pumping the UI every so often so a deliberately-slow KDF does not
/// freeze the clock, caret, mouse and net stack while it runs.
///
/// Byte-identical to [`derive`] — pinned by
/// `derive_and_derive_pumped_agree_and_the_pump_actually_fired`, which is what
/// stops the pumped variant drifting into deriving a different key (a bug whose
/// only symptom would be "the password stopped working").
pub fn derive_pumped(
    password: &str,
    salt: &[u8; SALT_LEN],
    iterations: u32,
    pump: &mut dyn FnMut(),
) -> [u8; HASH_LEN] {
    let mut salted = Vec::with_capacity(CONTEXT.len() + SALT_LEN);
    salted.extend_from_slice(CONTEXT);
    salted.extend_from_slice(salt);
    let mut out = [0u8; HASH_LEN];
    crate::block::volcrypto::pbkdf2_hmac_sha256_pumped(
        password.as_bytes(),
        &salted,
        iterations,
        &mut out,
        pump,
    );
    out
}

/// Compare two byte slices without an early exit.
///
/// The KDF dominates the observable timing here, so this is belt-and-braces
/// rather than the load-bearing defence — but there is no constant-time compare
/// anywhere in this tree, and a naive `==` on a verifier is the kind of thing
/// that gets cited later.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Check `password` against `rec`.
pub fn verify(rec: &Credential, password: &str, pump: &mut dyn FnMut()) -> bool {
    // A record whose KDF we do not implement must not authenticate anything.
    // `from_json` already refuses one, but `verify` is the security boundary and
    // should not depend on how its argument was built.
    if rec.kdf != KDF_PBKDF2_SHA256 || rec.iterations == 0 {
        return false;
    }
    let got = derive_pumped(password, &rec.salt, rec.iterations, pump);
    ct_eq(&got, &rec.hash)
}

/// What a new password must satisfy.
///
/// Printable ASCII only is **not** a style choice: `modal::input` accepts only
/// `0x20..=0x7e` and so does the login prompt's own reader, so anything else
/// could be enrolled and then never typed again — an unrecoverable machine.
pub fn validate_new(password: &str) -> Result<(), &'static str> {
    if password.is_empty() {
        return Err("empty");
    }
    if password.chars().count() < MIN_PASSWORD_LEN {
        return Err("too short (minimum 8 characters)");
    }
    if password.chars().count() > MAX_PASSWORD_LEN {
        return Err("too long (maximum 128 characters)");
    }
    if !password.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err("must be printable ASCII (the login prompt cannot type anything else)");
    }
    Ok(())
}

// ── Backoff and idle arithmetic ──────────────────────────────────────────

/// Delay before the next attempt is accepted, given consecutive failures.
///
/// `0,0,0` then `1s,2s,4s,8s,16s,30s,30s…`. The first few are free because a
/// typo is the overwhelmingly common case; the ramp is what makes online
/// guessing pointless. See limitation 8 — a reboot resets this.
pub fn backoff_ms(consecutive_failures: u32) -> u64 {
    if consecutive_failures <= FREE_ATTEMPTS {
        return 0;
    }
    let step = consecutive_failures - FREE_ATTEMPTS - 1; // 0-based past the free ones
    if step >= 63 {
        return MAX_BACKOFF_MS;
    }
    (1000u64 << step).min(MAX_BACKOFF_MS)
}

/// Whether the console should auto-lock.
///
/// Pure so the boundary is testable. `idle_minutes == 0` disables it, and a
/// `last_input_ms` in the *future* (a clock that moved backwards, which happens
/// here — `/ntp` can jump the wall clock) must never lock immediately.
pub fn should_idle_lock(now_ms: u64, last_input_ms: u64, idle_minutes: u32) -> bool {
    if idle_minutes == 0 {
        return false;
    }
    let timeout = (idle_minutes as u64).saturating_mul(60_000);
    now_ms.saturating_sub(last_input_ms) >= timeout
}

/// Does this path name the credential record? Normalised first, so
/// `//configs//core/auth.json` and `/configs/core/../core/auth.json` are the same
/// question. Bare (relative) names stay relative per
/// [`crate::synapse::vpath::normalize`] and are therefore a *different* store
/// key — not this one.
/// `normalize_cow`, not `normalize`: this sits on the Synapse gate chain **and**
/// on every `synapse::fs` call, and the plain form allocates a `String` every
/// time. The cow form borrows when the path is already canonical — which it
/// almost always is, since the executor has normalised it once already — so the
/// common case is a compare with no allocation at all. (Performance trap #3: the
/// kernel allocator punishes churn.)
pub fn is_credential_path(path: &str) -> bool {
    &*crate::synapse::vpath::normalize_cow(path) == PATH
}

// ── Live state ───────────────────────────────────────────────────────────
//
// Cached rather than re-read from the store, because `should_lock_for_idle` is
// polled from `read_line`'s idle arm — a store `exists` per poll would be a
// directory lookup per keystroke-wait on ext4.

static ENROLLED: AtomicBool = AtomicBool::new(false);
static IDLE_LOCK_MINUTES: AtomicU32 = AtomicU32::new(DEFAULT_IDLE_LOCK_MINUTES);
static LOCK_ON_RESUME: AtomicBool = AtomicBool::new(true);
static LOCKED: AtomicBool = AtomicBool::new(false);
static LOCK_REQUESTED: AtomicU8 = AtomicU8::new(0);
static LAST_UNLOCK_MS: AtomicU64 = AtomicU64::new(0);
static CONSECUTIVE_FAILURES: AtomicU32 = AtomicU32::new(0);

/// Refresh the cached policy from a record (or clear it when there is none).
/// Called by [`store`] on every load/save/clear — the only writer.
pub(crate) fn set_cached(rec: Option<&Credential>) {
    match rec {
        Some(r) => {
            ENROLLED.store(true, Ordering::Relaxed);
            IDLE_LOCK_MINUTES.store(r.idle_lock_minutes, Ordering::Relaxed);
            LOCK_ON_RESUME.store(r.lock_on_resume, Ordering::Relaxed);
        }
        None => {
            ENROLLED.store(false, Ordering::Relaxed);
            IDLE_LOCK_MINUTES.store(DEFAULT_IDLE_LOCK_MINUTES, Ordering::Relaxed);
            LOCK_ON_RESUME.store(true, Ordering::Relaxed);
        }
    }
}

/// Whether a password is enrolled. Everything else is inert when this is false.
pub fn enrolled() -> bool {
    ENROLLED.load(Ordering::Relaxed)
}

pub fn idle_lock_minutes() -> u32 {
    IDLE_LOCK_MINUTES.load(Ordering::Relaxed)
}

pub fn lock_on_resume() -> bool {
    LOCK_ON_RESUME.load(Ordering::Relaxed)
}

/// Whether the gate is currently on screen.
pub fn is_locked() -> bool {
    LOCKED.load(Ordering::Relaxed)
}

pub(crate) fn set_locked(v: bool) {
    LOCKED.store(v, Ordering::Relaxed);
    if !v {
        LAST_UNLOCK_MS.store(crate::arch::now_ms(), Ordering::Relaxed);
    }
}

pub fn last_unlock_ms() -> u64 {
    LAST_UNLOCK_MS.load(Ordering::Relaxed)
}

/// Ask for a lock. The request is *performed* by the shell's interactive loop,
/// never here: locking from inside `upkeep()` would re-enter the shell from under
/// whatever compute stack happened to be pumping, the same reason the power
/// button is acted on in the loop rather than in the driver poll.
pub fn request_lock(reason: Reason) {
    LOCK_REQUESTED.store(reason.to_code(), Ordering::Relaxed);
}

/// Whether a lock is pending, **without** consuming it — the reason has to
/// survive from the poll that notices it to the loop that performs it.
pub fn lock_pending() -> bool {
    LOCK_REQUESTED.load(Ordering::Relaxed) != 0
}

/// Consume a pending lock request.
pub fn take_lock_request() -> Option<Reason> {
    let c = LOCK_REQUESTED.swap(0, Ordering::Relaxed);
    Reason::from_code(c)
}

/// Consecutive failed attempts this boot.
pub fn failures() -> u32 {
    CONSECUTIVE_FAILURES.load(Ordering::Relaxed)
}

pub(crate) fn note_failure() -> u32 {
    CONSECUTIVE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn reset_failures() {
    CONSECUTIVE_FAILURES.store(0, Ordering::Relaxed);
}

/// When any input source was last seen — keyboard **or** pointer.
///
/// Both timestamps already exist and between them cover every source: the
/// console one is stamped in the *merged* reader ([`crate::console::read_byte`]),
/// so serial, PS/2, xHCI/HID, virtio-keyboard and PL011 are all included, and the
/// mouse keeps its own. Do not add a third — and never call `mouse::tick()` to
/// get this, which is a *consuming* poll that would steal the caller's clicks.
pub fn last_input_ms() -> u64 {
    crate::console::input_activity_ms().max(crate::mouse::activity_ms())
}

/// Whether the console should lock itself right now for idleness.
pub fn should_lock_for_idle() -> bool {
    if !enrolled() || is_locked() {
        return false;
    }
    // Max with the last unlock so a machine that has just been unlocked cannot
    // immediately re-lock on a stale input timestamp.
    let last = last_input_ms().max(last_unlock_ms());
    should_idle_lock(crate::arch::now_ms(), last, idle_lock_minutes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cheap iteration count: `cargo xtask test` runs x86 under QEMU TCG, where
    /// the real 200k rounds would dominate the whole suite.
    const T_ITERS: u32 = 64;

    fn a_salt(seed: u8) -> [u8; SALT_LEN] {
        let mut s = [0u8; SALT_LEN];
        for (i, b) in s.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8).wrapping_mul(31);
        }
        s
    }

    fn a_record(password: &str) -> Credential {
        let salt = a_salt(7);
        Credential {
            version: VERSION,
            user: "chitti".into(),
            kdf: KDF_PBKDF2_SHA256.into(),
            iterations: T_ITERS,
            salt,
            hash: derive(password, &salt, T_ITERS),
            created_ms: 1_700_000_000_000,
            idle_lock_minutes: 15,
            lock_on_resume: true,
            failed_total: 2,
            last_failed_ms: 1_700_000_001_000,
        }
    }

    #[test_case]
    fn hex_round_trips_and_rejects_odd_length_and_bad_nibbles() {
        let bytes = [0x00u8, 0x0f, 0xa5, 0xff, 0x10];
        let s = hex_encode(&bytes);
        assert_eq!(s, "000fa5ff10");
        assert_eq!(hex_decode(&s).unwrap(), bytes);
        // Uppercase decodes too (a human may have edited the file).
        assert_eq!(hex_decode("000FA5FF10").unwrap(), bytes);
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_decode("").unwrap(), alloc::vec::Vec::<u8>::new());
        // Malformed input is refused, never silently repaired.
        assert!(hex_decode("abc").is_none(), "odd length accepted");
        assert!(hex_decode("zz").is_none(), "non-hex nibble accepted");
        assert!(hex_decode("00 11").is_none(), "space accepted");
    }

    #[test_case]
    fn a_credential_survives_a_json_round_trip() {
        let rec = a_record("hunter2!!");
        let text = rec.to_json().to_pretty();
        let parsed = crate::json::Json::parse(&text).expect("emitted JSON did not parse");
        let back = Credential::from_json(&parsed).expect("round trip lost the record");
        assert_eq!(back, rec);
    }

    #[test_case]
    fn a_malformed_record_is_none_not_a_default() {
        let rec = a_record("hunter2!!");
        let good = rec.to_json().to_pretty();
        // Sanity: the unmodified record parses, so each failure below is caused
        // by the mutation and not by the fixture.
        assert!(Credential::from_json(&crate::json::Json::parse(&good).unwrap()).is_some());

        let bad_cases: &[(&str, alloc::string::String)] = &[
            ("truncated salt", good.replace(&hex_encode(&rec.salt), "00ff")),
            ("truncated hash", good.replace(&hex_encode(&rec.hash), "00ff")),
            ("non-hex salt", good.replace(&hex_encode(&rec.salt), &"zz".repeat(SALT_LEN))),
            ("zero iterations", good.replace("\"iterations\": 64", "\"iterations\": 0")),
            ("unknown kdf", good.replace(KDF_PBKDF2_SHA256, "argon2id")),
            ("wrong version", good.replace("\"version\": 1", "\"version\": 2")),
        ];
        for (what, text) in bad_cases {
            let j = crate::json::Json::parse(text).expect("fixture is not valid JSON");
            assert!(Credential::from_json(&j).is_none(), "{what} was accepted");
        }
        // A missing required field is refused too.
        for key in ["version", "user", "kdf", "iterations", "salt", "hash"] {
            let j = crate::json::Json::parse(&good).unwrap();
            let pruned = match j {
                crate::json::Json::Obj(pairs) => {
                    crate::json::Json::Obj(pairs.into_iter().filter(|(k, _)| k != key).collect())
                }
                other => other,
            };
            assert!(Credential::from_json(&pruned).is_none(), "missing '{key}' was accepted");
        }
    }

    #[test_case]
    fn the_right_password_verifies_and_a_neighbouring_one_does_not() {
        let pw = "correct horse";
        let rec = a_record(pw);
        let mut noop = || {};
        assert!(verify(&rec, pw, &mut noop), "the correct password was rejected");
        for wrong in ["", "correct hors", "correct horse ", " correct horse", "Correct horse", "correct horsf"] {
            assert!(!verify(&rec, wrong, &mut noop), "{wrong:?} was accepted");
        }
    }

    #[test_case]
    fn the_same_password_under_a_different_salt_gives_a_different_hash() {
        let pw = "same password";
        assert_ne!(derive(pw, &a_salt(1), T_ITERS), derive(pw, &a_salt(2), T_ITERS));
        // …and the same salt is reproducible, or nothing could ever verify.
        assert_eq!(derive(pw, &a_salt(1), T_ITERS), derive(pw, &a_salt(1), T_ITERS));
    }

    /// Pins [`CONTEXT`]. Without this, "the login password is independent of the
    /// volume passphrase" is a claim in a comment rather than a property.
    #[test_case]
    fn the_login_kdf_is_domain_separated_from_the_volume_kdf() {
        let pw = "shared passphrase";
        let salt = a_salt(3);
        let login = derive(pw, &salt, T_ITERS);
        let mut volume = [0u8; HASH_LEN];
        crate::block::volcrypto::pbkdf2_hmac_sha256(pw.as_bytes(), &salt, T_ITERS, &mut volume);
        assert_ne!(login, volume, "the same password and salt produced the same bytes in both KDFs");
    }

    #[test_case]
    fn derive_and_derive_pumped_agree_and_the_pump_actually_fired() {
        let pw = "a password";
        let salt = a_salt(5);
        // Above volcrypto's pump interval so the pump really runs.
        let iters = 2048;
        let plain = derive(pw, &salt, iters);
        let mut pumps = 0usize;
        let pumped = derive_pumped(pw, &salt, iters, &mut || pumps += 1);
        assert_eq!(plain, pumped, "the pumped derivation produced different bytes");
        assert!(pumps > 0, "the pump never fired");
    }

    #[test_case]
    fn ct_eq_matches_slice_equality_including_length_mismatch() {
        assert!(ct_eq(b"", b""));
        assert!(ct_eq(b"abcd", b"abcd"));
        assert!(!ct_eq(b"abcd", b"abce"));
        assert!(!ct_eq(b"abcd", b"abcde"), "length mismatch compared equal");
        assert!(!ct_eq(b"abcde", b"abcd"), "length mismatch compared equal");
        assert!(!ct_eq(b"", b"a"));
        // Differences in the first byte and the last byte are both caught.
        assert!(!ct_eq(&[0u8, 1, 2, 3], &[9, 1, 2, 3]));
        assert!(!ct_eq(&[0u8, 1, 2, 3], &[0, 1, 2, 9]));
    }

    #[test_case]
    fn backoff_is_zero_for_the_first_three_and_then_doubles_to_a_ceiling() {
        let want = [0u64, 0, 0, 0, 1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000];
        for (n, &ms) in want.iter().enumerate() {
            assert_eq!(backoff_ms(n as u32), ms, "backoff_ms({n})");
        }
        // Never overflows, however many failures accumulate.
        assert_eq!(backoff_ms(u32::MAX), 30000);
    }

    #[test_case]
    fn idle_lock_fires_only_after_the_configured_minutes_and_never_when_zero() {
        let min = 60_000u64;
        // 15 minutes configured.
        assert!(!should_idle_lock(14 * min, 0, 15), "locked early");
        assert!(should_idle_lock(15 * min, 0, 15), "did not lock at the timeout");
        assert!(should_idle_lock(100 * min, 0, 15));
        // Recent input holds it off.
        assert!(!should_idle_lock(100 * min, 99 * min, 15));
        // 0 disables it entirely, however long the machine has been idle.
        assert!(!should_idle_lock(u64::MAX, 0, 0), "auto-lock fired while disabled");
        // A clock that moved backwards must not lock instantly (`/ntp` can jump
        // the wall clock, and `last > now` is then normal, not corruption).
        assert!(!should_idle_lock(5 * min, 100 * min, 15), "a backwards clock locked the console");
        // No overflow at the extremes.
        assert!(should_idle_lock(u64::MAX, 0, u32::MAX) || !should_idle_lock(u64::MAX, 0, u32::MAX));
    }

    #[test_case]
    fn validate_new_rejects_short_empty_and_untypeable_passwords() {
        assert!(validate_new("").is_err());
        assert!(validate_new("short").is_err(), "a 5-character password was accepted");
        assert!(validate_new("1234567").is_err(), "a 7-character password was accepted");
        assert!(validate_new("12345678").is_ok(), "an 8-character password was rejected");
        assert!(validate_new("a longer pass phrase!").is_ok());
        // The whole printable-ASCII range is fine.
        assert!(validate_new("~!@#$%^&*()_+ 09azAZ").is_ok());
        // Anything the prompt cannot type is refused at enrolment.
        assert!(validate_new("pässwörd123").is_err(), "non-ASCII was accepted");
        assert!(validate_new("パスワードだよ12").is_err(), "CJK was accepted");
        assert!(validate_new("tab\there1").is_err(), "a control character was accepted");
        let too_long: alloc::string::String = core::iter::repeat('a').take(MAX_PASSWORD_LEN + 1).collect();
        assert!(validate_new(&too_long).is_err(), "an over-long password was accepted");
    }

    #[test_case]
    fn is_credential_path_matches_through_normalisation() {
        for p in [
            "/configs/core/auth.json",
            "//configs//core/auth.json",
            "/configs/core/../core/auth.json",
            "/configs/./core/auth.json",
            "/configs/core/auth.json/",
        ] {
            assert!(is_credential_path(p), "{p} was not recognised as the credential path");
        }
        for p in [
            "/configs/core/auth.json.bak",
            "/configs/core/ui.json",
            "/configs/core/auth.jsonx",
            // Bare names stay relative under `vpath::normalize`, so this is a
            // different store key entirely — not the record.
            "configs/core/auth.json",
            "/auth.json",
            "/configs/auth.json",
        ] {
            assert!(!is_credential_path(p), "{p} was wrongly treated as the credential path");
        }
    }

    #[test_case]
    fn a_lock_request_is_taken_exactly_once() {
        // Drain anything a previous test left behind.
        let _ = take_lock_request();
        assert_eq!(take_lock_request(), None);
        request_lock(Reason::Idle);
        assert_eq!(take_lock_request(), Some(Reason::Idle));
        assert_eq!(take_lock_request(), None, "a lock request was delivered twice");
        // Every reason round-trips through the atomic encoding.
        for r in [Reason::Boot, Reason::Manual, Reason::Idle, Reason::Resume] {
            request_lock(r);
            assert_eq!(take_lock_request(), Some(r), "{} did not round trip", r.as_str());
        }
    }
}
