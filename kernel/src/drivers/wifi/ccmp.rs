//! **CCMP** — the cipher that actually carries Wi-Fi traffic, and the generic
//! **CCM** mode underneath it.
//!
//! [`super::wpa`] takes a passphrase to a PTK, and that is where the WPA2 story
//! used to stop: the RSN parser *named* CCMP as the only cipher worth
//! supporting and nothing implemented it, so a completed four-way handshake
//! could not move one byte. This is the missing half, and it lives above the
//! radio — Intel, Broadcom, Realtek and MediaTek all need exactly this code.
//!
//! Two layers, deliberately separated:
//!
//! * [`ccm_encrypt`] / [`ccm_decrypt`] are **generic CCM (RFC 3610)** — CTR for
//!   confidentiality, CBC-MAC for authenticity, sharing one key. Pinned to RFC
//!   3610's own published packet vectors, which is an oracle written by someone
//!   else; a self-consistency test here would prove nothing, since encrypt and
//!   decrypt share every table.
//! * [`encrypt`] / [`decrypt`] are the **802.11 framing** on top: the nonce, the
//!   additional authenticated data, the 8-byte CCMP header and the packet
//!   number.
//!
//! ## Where CCMP goes wrong silently
//!
//! Every mistake below produces a well-formed frame that the peer discards
//! without saying why — an association that succeeds and then passes no
//! traffic, which is indistinguishable from a dozen other faults.
//!
//! * **The packet number is split across the CCMP header out of order.** Bytes
//!   0 and 1 are PN0 and PN1, then the KeyID byte interrupts, and PN2..PN5
//!   follow at bytes 4..8. Writing the six bytes consecutively puts the KeyID
//!   inside the number and the number across the KeyID.
//! * **The PN is little-endian in the header and big-endian in the nonce.**
//!   Same six bytes, opposite order, eight bytes apart in the same function.
//! * **The AAD is the header with fields masked out**, not the header. Retry,
//!   power-management and more-data are masked because they change in flight;
//!   the sequence number is masked to its fragment nibble for the same reason;
//!   and the Protected bit is forced *on*. Authenticating them unmasked makes
//!   every retransmitted frame fail its MIC — so the link works until the first
//!   retry, which on a real network is immediately.
//! * **The nonce carries the TID**, so a QoS frame and a non-QoS frame with the
//!   same PN encrypt differently. Getting it wrong works on exactly the frames
//!   you test with by hand.

use super::wpa::aes128_encrypt_block;
use alloc::vec::Vec;

/// CCMP's MIC is 8 bytes (`M`).
pub const MIC_LEN: usize = 8;
/// The CCMP header that precedes the ciphertext.
pub const HDR_LEN: usize = 8;
/// The packet number is 48 bits.
pub const PN_LEN: usize = 6;
/// CCMP's length field is 2 bytes (`L`), which bounds a frame to 65535 bytes.
const L: usize = 2;

/// Bytes CCMP adds to a frame body: header + MIC.
pub const OVERHEAD: usize = HDR_LEN + MIC_LEN;

// --- generic CCM (RFC 3610) ----------------------------------------------

/// The `B_0` flags byte: `64*Adata + 8*M' + L'`.
///
/// A named function because all three fields are derived, not chosen, and a
/// wrong one changes the MIC without changing anything observable about the
/// frame.
fn b0_flags(has_aad: bool, mic_len: usize) -> u8 {
    let adata = if has_aad { 1u8 } else { 0 };
    let m_prime = ((mic_len - 2) / 2) as u8;
    let l_prime = (L - 1) as u8;
    (adata << 6) | (m_prime << 3) | l_prime
}

/// Counter block `A_i` — the CTR half. Its flags carry only `L'`: no Adata bit
/// and no MIC length, because the counter stream authenticates nothing.
fn ctr_block(nonce: &[u8], i: u16) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[0] = (L - 1) as u8;
    a[1..1 + nonce.len()].copy_from_slice(nonce);
    a[14..16].copy_from_slice(&i.to_be_bytes());
    a
}

/// CBC-MAC over `B_0`, the length-prefixed AAD, and the payload.
fn cbc_mac(key: &[u8; 16], nonce: &[u8], aad: &[u8], payload: &[u8], mic_len: usize) -> [u8; 16] {
    let mut x = [0u8; 16];
    // B_0 = flags || nonce || l(m), big-endian.
    x[0] = b0_flags(!aad.is_empty(), mic_len);
    x[1..1 + nonce.len()].copy_from_slice(nonce);
    x[14..16].copy_from_slice(&(payload.len() as u16).to_be_bytes());
    aes128_encrypt_block(key, &mut x);

    if !aad.is_empty() {
        // The AAD is prefixed with its own length — two bytes for anything
        // under 2^16 - 2^8, which every 802.11 header is — and then zero-padded
        // to a block. Omitting the prefix shifts the whole AAD by two bytes and
        // yields a MIC that is wrong in a way no length check catches.
        let mut block = Vec::with_capacity(2 + aad.len() + 15);
        block.extend_from_slice(&(aad.len() as u16).to_be_bytes());
        block.extend_from_slice(aad);
        while block.len() % 16 != 0 {
            block.push(0);
        }
        for chunk in block.chunks_exact(16) {
            for (xi, ci) in x.iter_mut().zip(chunk) {
                *xi ^= ci;
            }
            aes128_encrypt_block(key, &mut x);
        }
    }

    // Payload, zero-padded to a block.
    let mut i = 0;
    while i < payload.len() {
        let n = (payload.len() - i).min(16);
        for k in 0..n {
            x[k] ^= payload[i + k];
        }
        aes128_encrypt_block(key, &mut x);
        i += n;
    }
    x
}

/// Apply the CTR keystream to `data` in place, starting at counter 1.
fn ctr_xor(key: &[u8; 16], nonce: &[u8], data: &mut [u8]) {
    for (i, chunk) in data.chunks_mut(16).enumerate() {
        let mut s = ctr_block(nonce, (i + 1) as u16);
        aes128_encrypt_block(key, &mut s);
        for (d, k) in chunk.iter_mut().zip(s.iter()) {
            *d ^= k;
        }
    }
}

/// Generic CCM encryption: returns `ciphertext || MIC`.
///
/// `nonce` must be `15 - L` = 13 bytes.
pub fn ccm_encrypt(key: &[u8; 16], nonce: &[u8], aad: &[u8], plain: &[u8], mic_len: usize) -> Option<Vec<u8>> {
    if nonce.len() != 15 - L || !(4..=16).contains(&mic_len) || mic_len % 2 != 0 {
        return None;
    }
    let t = cbc_mac(key, nonce, aad, plain, mic_len);
    let mut out = plain.to_vec();
    ctr_xor(key, nonce, &mut out);
    // The MIC is encrypted with counter **zero**, which the payload never uses.
    let mut s0 = ctr_block(nonce, 0);
    aes128_encrypt_block(key, &mut s0);
    for k in 0..mic_len {
        out.push(t[k] ^ s0[k]);
    }
    Some(out)
}

/// Generic CCM decryption. `None` when the MIC does not verify — and the
/// plaintext is **not** returned in that case, because a caller handed
/// unauthenticated bytes will use them.
pub fn ccm_decrypt(key: &[u8; 16], nonce: &[u8], aad: &[u8], data: &[u8], mic_len: usize) -> Option<Vec<u8>> {
    if nonce.len() != 15 - L || data.len() < mic_len {
        return None;
    }
    let (ct, mic) = data.split_at(data.len() - mic_len);
    let mut plain = ct.to_vec();
    ctr_xor(key, nonce, &mut plain);
    let t = cbc_mac(key, nonce, aad, &plain, mic_len);
    let mut s0 = ctr_block(nonce, 0);
    aes128_encrypt_block(key, &mut s0);
    // Constant-time-ish compare: accumulate differences rather than returning
    // early, so the loop does not leak how many leading bytes matched.
    let mut diff = 0u8;
    for k in 0..mic_len {
        diff |= (t[k] ^ s0[k]) ^ mic[k];
    }
    if diff != 0 {
        return None;
    }
    Some(plain)
}

// --- 802.11 framing -------------------------------------------------------

const FC_RETRY: u16 = 0x0800;
const FC_PWR_MGT: u16 = 0x1000;
const FC_MORE_DATA: u16 = 0x2000;
const FC_PROTECTED: u16 = 0x4000;
const FC_SUBTYPE: u16 = 0x0070;
const FC_ORDER: u16 = 0x8000;

/// Frame type bits (2..3 of the frame control field).
fn fc_type(fc: u16) -> u16 {
    (fc >> 2) & 0x3
}
fn is_mgmt(fc: u16) -> bool {
    fc_type(fc) == 0
}
fn is_data(fc: u16) -> bool {
    fc_type(fc) == 2
}
/// A QoS data frame carries two extra header bytes.
fn is_qos(fc: u16) -> bool {
    is_data(fc) && (fc & 0x0080) != 0
}
/// Both ToDS and FromDS set means a four-address (WDS) frame.
fn has_a4(fc: u16) -> bool {
    (fc & 0x0300) == 0x0300
}

/// Length of the 802.11 MAC header this frame control implies.
pub fn header_len(fc: u16) -> usize {
    let mut n = 24;
    if has_a4(fc) {
        n += 6;
    }
    if is_qos(fc) {
        n += 2;
    }
    // HT Control, present when the Order bit is set on a QoS frame.
    if is_qos(fc) && (fc & FC_ORDER) != 0 {
        n += 4;
    }
    n
}

/// The CCMP nonce: flags, transmitter address, packet number.
///
/// `pn` is **big-endian** here (most significant byte first) — the opposite of
/// the order the same six bytes take in the CCMP header. The flags byte carries
/// the traffic identifier in its low nibble and a management-frame bit above
/// it, so the same PN under the same key produces a different keystream for a
/// QoS frame than for a best-effort one.
pub fn nonce(fc: u16, a2: &[u8; 6], pn: &[u8; PN_LEN], tid: u8) -> [u8; 13] {
    let mut n = [0u8; 13];
    n[0] = (tid & 0x0f) | (u8::from(is_mgmt(fc)) << 4);
    n[1..7].copy_from_slice(a2);
    n[7..13].copy_from_slice(pn);
    n
}

/// The additional authenticated data: the MAC header with the fields that
/// change in flight masked out.
///
/// `hdr` is the full 802.11 header. Returns `None` if it is shorter than the
/// frame control implies.
pub fn aad(hdr: &[u8]) -> Option<Vec<u8>> {
    if hdr.len() < 24 {
        return None;
    }
    let fc = u16::from_le_bytes([hdr[0], hdr[1]]);
    let hlen = header_len(fc);
    if hdr.len() < hlen {
        return None;
    }
    // Retry, power-management and more-data all change between the original
    // transmission and a retransmission of the same frame, so authenticating
    // them makes every retry fail its MIC. The Protected bit is forced on
    // because the receiver sees it set and the transmitter computes the AAD
    // before setting it.
    let mut mask = fc & !(FC_RETRY | FC_PWR_MGT | FC_MORE_DATA);
    if !is_mgmt(fc) {
        // Subtype is masked for data frames only — a management frame's subtype
        // is what distinguishes the frames CCMP protects.
        mask &= !FC_SUBTYPE;
    }
    mask |= FC_PROTECTED;

    let mut a = Vec::with_capacity(30);
    a.extend_from_slice(&mask.to_le_bytes());
    a.extend_from_slice(&hdr[4..22]); // A1, A2, A3
    // Sequence Control: only the fragment number survives. The sequence number
    // itself advances on retransmission.
    a.push(hdr[22] & 0x0f);
    a.push(0);
    if has_a4(fc) {
        a.extend_from_slice(&hdr[24..30]);
    }
    if is_qos(fc) {
        let at = if has_a4(fc) { 30 } else { 24 };
        // Only the TID survives; the rest of the QoS control field is
        // transmitter state.
        a.push(hdr[at] & 0x0f);
        a.push(0);
    }
    Some(a)
}

/// The traffic identifier a frame belongs to — its QoS priority, or 0.
pub fn tid_of(hdr: &[u8]) -> u8 {
    let fc = u16::from_le_bytes([hdr[0], hdr[1]]);
    if !is_qos(fc) {
        return 0;
    }
    let at = if has_a4(fc) { 30 } else { 24 };
    hdr.get(at).map_or(0, |b| b & 0x0f)
}

/// Write the 8-byte CCMP header for `pn` (big-endian) and `key_id`.
///
/// **The packet number is not contiguous.** PN0 and PN1 come first, the KeyID
/// byte sits at index 3, and PN2..PN5 follow — and within that the bytes run
/// least-significant first, the reverse of the nonce. Writing the six bytes in
/// order puts the KeyID inside the number.
pub fn write_header(pn: &[u8; PN_LEN], key_id: u8) -> [u8; HDR_LEN] {
    let mut h = [0u8; HDR_LEN];
    h[0] = pn[5];
    h[1] = pn[4];
    h[2] = 0; // reserved
    h[3] = 0x20 | ((key_id & 0x3) << 6); // ExtIV always set for CCMP
    h[4] = pn[3];
    h[5] = pn[2];
    h[6] = pn[1];
    h[7] = pn[0];
    h
}

/// Recover `(pn, key_id)` from a CCMP header, or `None` when the ExtIV bit is
/// clear — which means this is not a CCMP-protected frame at all (WEP sets it
/// to zero), and reading it as one yields a plausible packet number.
pub fn read_header(h: &[u8]) -> Option<([u8; PN_LEN], u8)> {
    let h = h.get(..HDR_LEN)?;
    if h[3] & 0x20 == 0 {
        return None;
    }
    let pn = [h[7], h[6], h[5], h[4], h[1], h[0]];
    Some((pn, (h[3] >> 6) & 0x3))
}

/// Encrypt a frame body. `hdr` is the plaintext 802.11 header, `body` the
/// payload after it. Returns `CCMP header || ciphertext || MIC`.
pub fn encrypt(tk: &[u8; 16], hdr: &[u8], pn: &[u8; PN_LEN], key_id: u8, body: &[u8]) -> Option<Vec<u8>> {
    let fc = u16::from_le_bytes([*hdr.first()?, *hdr.get(1)?]);
    let a2: [u8; 6] = hdr.get(10..16)?.try_into().ok()?;
    let aad = aad(hdr)?;
    let n = nonce(fc, &a2, pn, tid_of(hdr));
    let ct = ccm_encrypt(tk, &n, &aad, body, MIC_LEN)?;
    let mut out = Vec::with_capacity(HDR_LEN + ct.len());
    out.extend_from_slice(&write_header(pn, key_id));
    out.extend_from_slice(&ct);
    Some(out)
}

/// Decrypt a frame body. `data` is `CCMP header || ciphertext || MIC`.
/// Returns the plaintext and the packet number it carried.
///
/// `None` covers every failure — not CCMP, too short, MIC mismatch — because a
/// caller cannot do anything different with them and a partially-validated
/// frame must never escape.
pub fn decrypt(tk: &[u8; 16], hdr: &[u8], data: &[u8]) -> Option<(Vec<u8>, [u8; PN_LEN])> {
    let fc = u16::from_le_bytes([*hdr.first()?, *hdr.get(1)?]);
    let a2: [u8; 6] = hdr.get(10..16)?.try_into().ok()?;
    let (pn, _key_id) = read_header(data)?;
    if data.len() < HDR_LEN + MIC_LEN {
        return None;
    }
    let aad = aad(hdr)?;
    let n = nonce(fc, &a2, &pn, tid_of(hdr));
    let plain = ccm_decrypt(tk, &n, &aad, &data[HDR_LEN..], MIC_LEN)?;
    Some((plain, pn))
}

/// A transmit packet-number counter.
///
/// The PN must never repeat under one key: CCM is a counter mode, so a repeat
/// hands an observer the XOR of two plaintexts, and a receiver that tracks
/// replay drops the second frame anyway. Exhaustion is therefore a hard stop
/// rather than a wrap — 2^48 frames is unreachable in practice, and wrapping
/// silently would be a confidentiality failure rather than an outage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PnCounter(u64);

impl PnCounter {
    /// A fresh counter. The first frame sent uses PN 1: zero is reserved so a
    /// receiver can treat "no frame seen yet" as a distinct state.
    pub fn new() -> PnCounter {
        PnCounter(0)
    }

    /// The next packet number, or `None` once the 48-bit space is exhausted.
    pub fn next(&mut self) -> Option<[u8; PN_LEN]> {
        let n = self.0.checked_add(1)?;
        if n >= 1 << 48 {
            return None;
        }
        self.0 = n;
        Some(pn_bytes(n))
    }
}

impl Default for PnCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// A 48-bit packet number as its six big-endian bytes.
pub fn pn_bytes(v: u64) -> [u8; PN_LEN] {
    let b = v.to_be_bytes();
    [b[2], b[3], b[4], b[5], b[6], b[7]]
}

/// The integer value of a packet number.
pub fn pn_value(pn: &[u8; PN_LEN]) -> u64 {
    let mut v = 0u64;
    for &b in pn {
        v = (v << 8) | b as u64;
    }
    v
}

/// Receive-side replay window: a frame whose PN is not greater than the last
/// accepted one is a replay and must be dropped.
///
/// Strictly increasing rather than a sliding window, which is what 802.11
/// requires per TID without block-ack reordering. A window would accept frames
/// out of order; without one, the rule is simply that the number must advance.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayGuard {
    last: u64,
    seen: bool,
}

impl ReplayGuard {
    pub fn new() -> ReplayGuard {
        ReplayGuard { last: 0, seen: false }
    }

    /// Accept `pn` if it advances. Returns false for a replay or a stale frame.
    pub fn accept(&mut self, pn: &[u8; PN_LEN]) -> bool {
        let v = pn_value(pn);
        if self.seen && v <= self.last {
            return false;
        }
        self.last = v;
        self.seen = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **RFC 3610 Packet Vector #1** — an oracle written by someone else. A
    /// round-trip test here would prove nothing: encrypt and decrypt share every
    /// table, so a wrong S-box or a wrong flags byte round-trips perfectly and
    /// interoperates with nothing.
    #[test_case]
    fn ccm_matches_rfc3610_packet_vector_1() {
        let key: [u8; 16] = [
            0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD,
            0xCE, 0xCF,
        ];
        let nonce: [u8; 13] = [
            0x00, 0x00, 0x00, 0x03, 0x02, 0x01, 0x00, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
        ];
        let aad: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let plain: [u8; 23] = [
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
            0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        ];
        let want: [u8; 31] = [
            0x58, 0x8C, 0x97, 0x9A, 0x61, 0xC6, 0x63, 0xD2, 0xF0, 0x66, 0xD0, 0xC2, 0xC0, 0xF9,
            0x89, 0x80, 0x6D, 0x5F, 0x6B, 0x61, 0xDA, 0xC3, 0x84, // ciphertext (23)
            0x17, 0xE8, 0xD1, 0x2C, 0xFD, 0xF9, 0x26, 0xE0, // MIC (8)
        ];
        let got = ccm_encrypt(&key, &nonce, &aad, &plain, 8).expect("encrypts");
        assert_eq!(&got[..], &want[..], "RFC 3610 vector 1");

        // And it decrypts back.
        let back = ccm_decrypt(&key, &nonce, &aad, &want, 8).expect("verifies");
        assert_eq!(&back[..], &plain[..]);
    }

    /// A tampered MIC, a tampered ciphertext and a tampered AAD must all be
    /// rejected — and rejected without handing back plaintext, because a caller
    /// given unauthenticated bytes will use them.
    #[test_case]
    fn ccm_rejects_tampering_everywhere() {
        let key = [0x11u8; 16];
        let nonce = [0x22u8; 13];
        let aad = [0x33u8; 20];
        let plain = [0x44u8; 40];
        let ct = ccm_encrypt(&key, &nonce, &aad, &plain, 8).unwrap();
        assert_eq!(ccm_decrypt(&key, &nonce, &aad, &ct, 8).as_deref(), Some(&plain[..]));

        for flip in [0usize, 20, ct.len() - 1] {
            let mut bad = ct.clone();
            bad[flip] ^= 1;
            assert_eq!(ccm_decrypt(&key, &nonce, &aad, &bad, 8), None, "byte {flip}");
        }
        let mut bad_aad = aad;
        bad_aad[0] ^= 1;
        assert_eq!(ccm_decrypt(&key, &nonce, &bad_aad, &ct, 8), None, "AAD is authenticated");
        let mut bad_nonce = nonce;
        bad_nonce[0] ^= 1;
        assert_eq!(ccm_decrypt(&key, &bad_nonce, &aad, &ct, 8), None, "the nonce binds too");
        // A wrong key, obviously.
        assert_eq!(ccm_decrypt(&[0x99u8; 16], &nonce, &aad, &ct, 8), None);
    }

    /// The `B_0` flags byte is three derived fields, and a wrong one changes the
    /// MIC without changing anything visible about the frame.
    #[test_case]
    fn b0_flags_are_derived_not_chosen() {
        // CCMP: AAD present, M = 8, L = 2 → 64 + 24 + 1.
        assert_eq!(b0_flags(true, 8), 0x59);
        // No AAD drops the top bit; a 16-byte MIC raises M'.
        assert_eq!(b0_flags(false, 8), 0x19);
        assert_eq!(b0_flags(true, 16), 0x79);
    }

    /// **The packet number is split across the CCMP header, out of order, and
    /// least-significant first** — while the nonce carries the same six bytes
    /// most-significant first. Writing them contiguously puts the KeyID byte
    /// inside the number.
    #[test_case]
    fn the_ccmp_header_splits_the_packet_number_around_the_key_id() {
        let pn = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]; // big-endian: PN5..PN0
        let h = write_header(&pn, 2);
        assert_eq!(h[0], 0x66, "byte 0 is the least significant PN byte");
        assert_eq!(h[1], 0x55);
        assert_eq!(h[2], 0, "reserved");
        assert_eq!(h[3], 0x20 | (2 << 6), "ExtIV set, key id in the top bits");
        assert_eq!(h[4], 0x44);
        assert_eq!(h[5], 0x33);
        assert_eq!(h[6], 0x22);
        assert_eq!(h[7], 0x11, "byte 7 is the most significant");
        // The number is *not* contiguous anywhere in the header.
        assert_ne!(&h[0..6], &pn[..]);
        assert_ne!(&h[2..8], &pn[..]);

        let (back, kid) = read_header(&h).expect("round-trips");
        assert_eq!(back, pn);
        assert_eq!(kid, 2);

        // The same PN in the nonce runs the other way.
        let n = nonce(0x0800, &[0xaa; 6], &pn, 0);
        assert_eq!(&n[7..13], &pn[..], "the nonce is big-endian, the header is not");
    }

    /// ExtIV clear means this is not CCMP — WEP leaves it zero. Reading such a
    /// header as CCMP yields a perfectly plausible packet number.
    #[test_case]
    fn a_header_without_ext_iv_is_not_ccmp() {
        let mut h = write_header(&[1, 2, 3, 4, 5, 6], 0);
        h[3] &= !0x20;
        assert_eq!(read_header(&h), None);
        assert_eq!(read_header(&h[..7]), None, "and a short header is refused");
    }

    /// Build a minimal 24-byte data header: FC, duration, A1, A2, A3, seq.
    fn data_hdr(fc: u16, seq: u16) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(&fc.to_le_bytes());
        h.extend_from_slice(&[0, 0]); // duration
        h.extend_from_slice(&[0x01; 6]); // A1
        h.extend_from_slice(&[0x02; 6]); // A2
        h.extend_from_slice(&[0x03; 6]); // A3
        h.extend_from_slice(&seq.to_le_bytes());
        h
    }

    /// **The AAD masks the fields that change in flight.** Authenticating them
    /// unmasked makes every *retransmitted* frame fail its MIC — so the link
    /// works until the first retry, which on a real network is immediately.
    #[test_case]
    fn the_aad_masks_what_changes_between_retransmissions() {
        let base = data_hdr(0x0808, 0x1230); // data, ToDS
        let a = aad(&base).unwrap();

        // Retry, power-management and more-data must not affect the AAD.
        for bit in [FC_RETRY, FC_PWR_MGT, FC_MORE_DATA] {
            let h = data_hdr(0x0808 | bit, 0x1230);
            assert_eq!(aad(&h).unwrap(), a, "bit {bit:#06x} must be masked");
        }
        // Nor must the sequence *number* — only its fragment nibble counts.
        let h = data_hdr(0x0808, 0x9990);
        assert_eq!(aad(&h).unwrap(), a, "the sequence number advances on retry");
        // The fragment number, however, is authenticated.
        let h = data_hdr(0x0808, 0x1232);
        assert_ne!(aad(&h).unwrap(), a, "the fragment number is real");

        // The Protected bit is forced on, so a header with it already set gives
        // the same AAD as one without.
        let h = data_hdr(0x0808 | FC_PROTECTED, 0x1230);
        assert_eq!(aad(&h).unwrap(), a);
        assert_eq!(a[0] & 0x08, 0x08, "ToDS survives");
        assert_ne!(u16::from_le_bytes([a[0], a[1]]) & FC_PROTECTED, 0, "Protected forced on");
    }

    /// A data frame's subtype is masked; a management frame's is not, because
    /// the subtype is what distinguishes the frames CCMP protects.
    #[test_case]
    fn subtype_is_masked_for_data_but_not_for_management() {
        let d0 = aad(&data_hdr(0x0008, 0)).unwrap(); // data, subtype 0
        let d4 = aad(&data_hdr(0x0048, 0)).unwrap(); // data, subtype 4 (null)
        assert_eq!(d0, d4, "data subtype is masked");

        let m0 = aad(&data_hdr(0x0000, 0)).unwrap(); // mgmt, subtype 0
        let m4 = aad(&data_hdr(0x0040, 0)).unwrap(); // mgmt, subtype 4
        assert_ne!(m0, m4, "management subtype is authenticated");
    }

    /// A QoS frame is longer, its TID reaches the AAD *and* the nonce, and both
    /// matter: the same payload under the same key and PN encrypts differently
    /// per traffic class.
    #[test_case]
    fn qos_extends_the_header_the_aad_and_the_nonce() {
        assert_eq!(header_len(0x0808), 24, "plain data");
        assert_eq!(header_len(0x0888), 26, "QoS data");
        assert_eq!(header_len(0x0308), 30, "four-address");
        assert_eq!(header_len(0x0388), 32, "four-address QoS");

        let mut q = data_hdr(0x0888, 0); // QoS data
        q.extend_from_slice(&[0x05, 0x00]); // QoS control, TID 5
        let a = aad(&q).unwrap();
        assert_eq!(a.len(), 24, "22 + the QoS pair");
        assert_eq!(a[22], 5, "the TID is authenticated");
        assert_eq!(tid_of(&q), 5);

        // The nonce carries it too, so the keystream differs per TID.
        let pn = [0, 0, 0, 0, 0, 7];
        let n5 = nonce(0x0888, &[0xaa; 6], &pn, 5);
        let n0 = nonce(0x0888, &[0xaa; 6], &pn, 0);
        assert_ne!(n5, n0, "same PN, different traffic class, different nonce");
        assert_eq!(n5[0] & 0x0f, 5);
        // A management frame raises bit 4.
        assert_ne!(nonce(0x0000, &[0xaa; 6], &pn, 0)[0] & 0x10, 0);
        assert_eq!(nonce(0x0808, &[0xaa; 6], &pn, 0)[0] & 0x10, 0);
    }

    /// A truncated header is refused rather than read past — these bytes came
    /// off the air from an unauthenticated sender.
    #[test_case]
    fn a_truncated_header_is_refused() {
        assert_eq!(aad(&[]), None);
        assert_eq!(aad(&data_hdr(0x0808, 0)[..23]), None);
        // A QoS frame control with only a 24-byte header present.
        assert_eq!(aad(&data_hdr(0x0888, 0)), None, "QoS needs 26 bytes");
    }

    /// The whole path: encrypt a frame, decrypt it back, and confirm the
    /// ciphertext is not the plaintext.
    #[test_case]
    fn a_frame_round_trips_through_ccmp() {
        let tk = [0x5au8; 16];
        let hdr = data_hdr(0x0808, 0x0010);
        let body = b"the quick brown fox jumps over the lazy dog";
        let pn = [0, 0, 0, 0, 0x12, 0x34];
        let enc = encrypt(&tk, &hdr, &pn, 0, body).expect("encrypts");
        assert_eq!(enc.len(), HDR_LEN + body.len() + MIC_LEN);
        assert_ne!(&enc[HDR_LEN..HDR_LEN + body.len()], &body[..], "must be encrypted");

        let (plain, got_pn) = decrypt(&tk, &hdr, &enc).expect("decrypts");
        assert_eq!(&plain[..], &body[..]);
        assert_eq!(got_pn, pn);

        // A frame authenticated against a *different* header must not verify —
        // that is the whole point of the AAD.
        let other = data_hdr(0x0808, 0x0010);
        let mut other = other.clone();
        other[10] ^= 1; // change A2, which is in both the AAD and the nonce
        assert_eq!(decrypt(&tk, &other, &enc), None);
    }

    /// The packet number must never repeat under one key: CCM is a counter
    /// mode, so a repeat hands an observer the XOR of two plaintexts.
    #[test_case]
    fn the_packet_number_advances_and_never_wraps() {
        let mut c = PnCounter::new();
        assert_eq!(c.next(), Some([0, 0, 0, 0, 0, 1]), "PN 0 is reserved");
        assert_eq!(c.next(), Some([0, 0, 0, 0, 0, 2]));
        assert_eq!(pn_value(&[0, 0, 0, 0, 0, 2]), 2);
        assert_eq!(pn_bytes(0xffff_ffff_ffff), [0xff; 6]);
        assert_eq!(pn_value(&[0xff; 6]), 0xffff_ffff_ffff);

        // At the end of the space it stops rather than wrapping.
        let mut end = PnCounter(0xffff_ffff_fffe);
        assert_eq!(end.next(), Some([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]));
        assert_eq!(end.next(), None, "exhaustion is a hard stop, not a wrap");
    }

    /// A replayed frame is dropped. Without this an attacker can re-inject a
    /// captured frame indefinitely and it decrypts perfectly every time.
    #[test_case]
    fn a_replayed_packet_number_is_rejected() {
        let mut g = ReplayGuard::new();
        assert!(g.accept(&[0, 0, 0, 0, 0, 1]));
        assert!(g.accept(&[0, 0, 0, 0, 0, 2]));
        assert!(!g.accept(&[0, 0, 0, 0, 0, 2]), "exact replay");
        assert!(!g.accept(&[0, 0, 0, 0, 0, 1]), "older");
        assert!(g.accept(&[0, 0, 0, 0, 0, 9]), "a jump forward is fine");
        // PN 0 is only ever accepted as a first frame, and 0 never occurs
        // because the counter starts at 1 — but the guard must not treat its
        // own initial zero as "already seen 0".
        let mut fresh = ReplayGuard::new();
        assert!(fresh.accept(&[0, 0, 0, 0, 0, 0]), "no frame seen yet");
        assert!(!fresh.accept(&[0, 0, 0, 0, 0, 0]), "but only once");
    }
}
