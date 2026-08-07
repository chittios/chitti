//! Test-only helpers shared across the HEVC modules.

/// Arithmetic-encode a bypass bin string so tests can feed the decoder a known
/// sequence.
///
/// **A bypass bin is not a raw bit.** With `range` fixed at 510 the decoder's
/// state obeys `offset_k = S_k - 510 * N_k`, where `S_k` is the first `9 + k`
/// bits of the stream read as an integer and `N_k` is the bins so far read as
/// one. Since `offset_k` is confined to `0..510` by construction, that inverts
/// exactly: `N_k = S_k / 510`. Encoding is therefore picking any `S` in the
/// right interval — the middle of it, here, for maximum slack.
///
/// This only holds while `range` stays at 510, i.e. for a run of bypass bins
/// starting at initialisation. A context-coded decision renormalises `range`
/// and invalidates it, which is why the tests that use this drive only bypass
/// paths.
pub fn pack_bypass(bits: &[u8]) -> alloc::vec::Vec<u8> {
    assert!(bits.len() <= 40, "would overflow the u64 interval arithmetic");
    let mut n: u64 = 0;
    for &b in bits {
        n = (n << 1) | b as u64;
    }
    let s = 510 * n + 255;
    let total = 9 + bits.len();
    let mut out = alloc::vec::Vec::new();
    let mut acc = 0u8;
    let mut nb = 0u32;
    for i in (0..total).rev() {
        acc = (acc << 1) | ((s >> i) & 1) as u8;
        nb += 1;
        if nb == 8 {
            out.push(acc);
            acc = 0;
            nb = 0;
        }
    }
    if nb > 0 {
        out.push(acc << (8 - nb));
    }
    // The engine keeps a 32-bit reservoir and refills eagerly, so give it bytes
    // to read past the encoded string rather than relying on the past-EOF fill.
    out.extend_from_slice(&[0u8; 16]);
    out
}
