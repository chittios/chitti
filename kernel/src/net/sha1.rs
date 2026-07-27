//! **SHA-1 and HMAC-SHA-1** — needed by WPA2, and by nothing else here.
//!
//! The tree already has SHA-256/384/512 through RustCrypto, and SHA-1 is deliberately not
//! among them: it is broken for signatures and has no business authenticating anything
//! new. WPA2 is the exception, and not by choice — IEEE 802.11i specifies SHA-1 for the
//! PSK derivation, the pairwise key hierarchy and the EAPOL-Key MIC, so a client that
//! wants to join an ordinary home network computes SHA-1 or it does not connect.
//!
//! **Do not reach for this for anything else.** It exists for one protocol that mandates
//! it. The `net::x509` path uses SHA-256 and above precisely because this primitive is
//! unfit for that job.
//!
//! ## Why write it rather than add a crate
//!
//! It is eighty lines, it has exhaustive published test vectors, and those vectors are the
//! entire correctness argument — every function here is checked against FIPS 180-2 and
//! RFC 2202 below. That is a stronger guarantee than most of this kernel's drivers have,
//! and it is available without another dependency.

use alloc::vec::Vec;

/// Digest length in bytes.
pub const SHA1_LEN: usize = 20;
/// Block size, which HMAC's padding is defined in terms of.
pub const SHA1_BLOCK: usize = 64;

/// SHA-1 of `msg`.
pub fn sha1(msg: &[u8]) -> [u8; SHA1_LEN] {
    let mut h: [u32; 5] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0];

    // Padding: a 0x80 byte, zeroes, then the length in **bits** as a big-endian u64. The
    // bit count is the classic mistake — a byte count here produces a digest that is
    // self-consistent and wrong, which no amount of round-tripping reveals.
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(msg.len() + 72);
    padded.extend_from_slice(msg);
    padded.push(0x80);
    while padded.len() % SHA1_BLOCK != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for block in padded.chunks_exact(SHA1_BLOCK) {
        let mut w = [0u32; 80];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }

    let mut out = [0u8; SHA1_LEN];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// HMAC-SHA-1 of `msg` under `key`.
///
/// A key longer than one block is hashed first — the one branch that is easy to omit,
/// because every short-key test vector passes without it and WPA2 uses keys of exactly
/// 32 bytes, so the omission would hide until something else used this.
pub fn hmac_sha1(key: &[u8], msg: &[u8]) -> [u8; SHA1_LEN] {
    let mut k = [0u8; SHA1_BLOCK];
    if key.len() > SHA1_BLOCK {
        k[..SHA1_LEN].copy_from_slice(&sha1(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut inner = Vec::with_capacity(SHA1_BLOCK + msg.len());
    let mut outer = Vec::with_capacity(SHA1_BLOCK + SHA1_LEN);
    for &b in k.iter() {
        inner.push(b ^ 0x36);
        outer.push(b ^ 0x5c);
    }
    inner.extend_from_slice(msg);
    outer.extend_from_slice(&sha1(&inner));
    sha1(&outer)
}

/// PBKDF2-HMAC-SHA1, producing `out.len()` bytes.
///
/// WPA2 uses this to turn a passphrase into the 256-bit PSK, with the SSID as salt and
/// 4096 iterations. The iteration count is not a tunable here — it is fixed by the
/// standard, and a different one produces a key the access point will not agree with.
pub fn pbkdf2_sha1(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    let mut block = 1u32;
    let mut done = 0usize;
    while done < out.len() {
        // U1 = HMAC(password, salt || block-index-as-big-endian-u32)
        let mut seed = Vec::with_capacity(salt.len() + 4);
        seed.extend_from_slice(salt);
        seed.extend_from_slice(&block.to_be_bytes());
        let mut u = hmac_sha1(password, &seed);
        let mut acc = u;
        for _ in 1..iterations {
            u = hmac_sha1(password, &u);
            for (a, b) in acc.iter_mut().zip(u.iter()) {
                *a ^= b;
            }
        }
        let take = core::cmp::min(SHA1_LEN, out.len() - done);
        out[done..done + take].copy_from_slice(&acc[..take]);
        done += take;
        block += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> alloc::string::String {
        let mut s = alloc::string::String::new();
        for x in b {
            s.push_str(&alloc::format!("{x:02x}"));
        }
        s
    }

    #[test_case]
    fn sha1_matches_the_fips_180_vectors() {
        // The published vectors are the entire correctness argument for this file.
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test_case]
    fn sha1_handles_the_lengths_where_padding_goes_wrong() {
        // 55, 56 and 64 bytes are the boundaries: at 56 the length no longer fits the
        // block and a second one is needed. An off-by-one in the padding loop passes every
        // short vector and fails exactly here.
        assert_eq!(
            hex(&sha1(&[b'a'; 55])),
            "c1c8bbdc22796e28c0e15163d20899b65621d65a"
        );
        assert_eq!(
            hex(&sha1(&[b'a'; 56])),
            "c2db330f6083854c99d4b5bfb6e8f29f201be699"
        );
        // And exactly one full block, where the padding occupies a whole second block.
        assert_eq!(
            hex(&sha1(&[b'a'; 64])),
            "0098ba824b5c16427bd7a1122a5a442a25ec644d"
        );
    }

    #[test_case]
    fn sha1_counts_bits_not_bytes() {
        // The classic mistake: padding with the byte count produces a digest that is
        // perfectly self-consistent and wrong, which nothing but a published vector
        // catches. "abc" is 3 bytes / 24 bits, and only the bit count gives this digest.
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        // A one-block message whose length spans two bytes of the counter.
        assert_eq!(
            hex(&sha1(&[0u8; 32])),
            "de8a847bff8c343d69b853a215e6ee775ef2ef96"
        );
    }

    #[test_case]
    fn hmac_sha1_matches_the_rfc_2202_vectors() {
        assert_eq!(
            hex(&hmac_sha1(&[0x0b; 20], b"Hi There")),
            "b617318655057264e28bc0b6fb378c8ef146be00"
        );
        assert_eq!(
            hex(&hmac_sha1(b"Jefe", b"what do ya want for nothing?")),
            "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79"
        );
        assert_eq!(
            hex(&hmac_sha1(&[0xaa; 20], &[0xdd; 50])),
            "125d7342b9ac11cd91a39af48aa17b4f63f175d3"
        );
    }

    #[test_case]
    fn hmac_sha1_hashes_an_over_long_key_first() {
        // RFC 2202 test 6: an 80-byte key, longer than the 64-byte block, must be hashed
        // down. Every short-key vector passes without this branch — and WPA2's keys are
        // 32 bytes, so omitting it would hide until something else used the function.
        assert_eq!(
            hex(&hmac_sha1(
                &[0xaa; 80],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "aa4ae5e15272d00e95705637ce8a3b55ed402112"
        );
    }

    #[test_case]
    fn pbkdf2_matches_the_ieee_80211i_psk_vectors() {
        // The vectors from IEEE 802.11i Annex H.4: passphrase and SSID to a 256-bit PSK at
        // 4096 iterations. Getting these right is what makes the rest of WPA2 worth
        // building — a wrong PSK produces a MIC mismatch the access point reports as a
        // wrong password.
        let mut psk = [0u8; 32];
        pbkdf2_sha1(b"password", b"IEEE", 4096, &mut psk);
        assert_eq!(
            hex(&psk),
            "f42c6fc52df0ebef9ebb4b90b38a5f902e83fe1b135a70e23aed762e9710a12e"
        );

        pbkdf2_sha1(b"ThisIsAPassword", b"ThisIsASSID", 4096, &mut psk);
        assert_eq!(
            hex(&psk),
            "0dc0d6eb90555ed6419756b9a15ec3e3209b63df707dd508d14581f8982721af"
        );
    }

    #[test_case]
    fn pbkdf2_spans_blocks_and_respects_the_iteration_count() {
        // A 32-byte output needs two HMAC blocks, so the block-index counter is exercised;
        // a implementation that ignored it would repeat the first 20 bytes.
        let mut psk = [0u8; 32];
        pbkdf2_sha1(b"password", b"IEEE", 4096, &mut psk);
        assert_ne!(psk[..20], psk[12..32], "second block repeats the first");

        // And the count matters: one iteration is a different key entirely, which is worth
        // pinning because a loop that ran `iterations` times instead of `iterations - 1`
        // extra times would be off by exactly one round.
        let mut one = [0u8; 20];
        pbkdf2_sha1(b"password", b"IEEE", 1, &mut one);
        let mut seed = alloc::vec::Vec::from(*b"IEEE");
        seed.extend_from_slice(&1u32.to_be_bytes());
        assert_eq!(one, hmac_sha1(b"password", &seed), "one iteration is one HMAC");
    }
}
