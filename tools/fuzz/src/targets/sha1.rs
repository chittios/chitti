//! Fuzz target: the kernel's SHA-1 / HMAC / PBKDF2 (`kernel/src/net/sha1.rs`).
//!
//! SHA-1 exists in the kernel for one reason: IEEE 802.11i / WPA2 (it is
//! deliberately absent from the TLS path). The supplicant runs it on
//! attacker-controlled 802.11 frame material, and PBKDF2 is the long pole of a
//! `/wifi connect` — a panic here would take the kernel down mid-handshake.
//! Mostly a round-trip sanity target (SHA-1 is a hash, not a length-driven
//! parser), it pins the mount pattern for the pure crypto modules.

// `kernel/src/net/sha1.rs` is `no_std` and names `alloc` explicitly.

#[path = "../../../../kernel/src/net/sha1.rs"]
pub mod sha1;

pub fn run(data: &[u8]) {
    let _ = sha1::sha1(data);
    // PBKDF2 with a *bounded* iteration count: the point is exercising the
    // length/alignment edge cases of the block loop, not melting the host.
    if data.len() <= 256 {
        let mut out = [0u8; 20];
        sha1::pbkdf2_sha1(data, b"chitti-fuzz", 16, &mut out);
    }
}
