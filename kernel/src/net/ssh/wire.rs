//! SSH wire primitives — the data types of RFC 4251 §5 and the binary packet
//! of RFC 4253 §6.
//!
//! Pure: bytes in, bytes out, no I/O and no crypto. Everything above this
//! module is expressed in terms of these readers/writers, so a wire-format
//! mistake shows up here as a failing test rather than as a connection that
//! desynchronises three messages later.
//!
//! Four encodings carry essentially all the risk, and each has a test:
//!
//! * **`string` is binary, length-prefixed and NOT NUL-terminated.** It carries
//!   key blobs and signatures as often as it carries text, so anything that
//!   treats it as UTF-8 corrupts a key.
//! * **`mpint` is two's-complement with a minimal encoding.** A positive number
//!   whose top bit is set needs a leading zero byte or the peer reads it as
//!   negative; a zero is the *empty* string, not one zero byte; and leading
//!   zeros are otherwise forbidden. Getting this wrong produces a shared secret
//!   that differs from the server's only in its first byte, so the exchange hash
//!   mismatches and the failure looks like a bad host key.
//! * **`name-list` is a `string` holding comma-separated ASCII**, with no
//!   spaces and no trailing comma — an empty list is an empty string, not a
//!   list containing one empty name.
//! * **The packet is padded to a block multiple, with at least four bytes**, and
//!   *which* bytes are counted differs by cipher mode ([`pad_len`]).

use alloc::string::String;
use alloc::vec::Vec;

/// Minimum padding on any SSH binary packet (RFC 4253 §6).
pub const MIN_PADDING: usize = 4;

/// A packet larger than this is refused rather than allocated.
///
/// RFC 4253 §6.1 requires support for 35000 bytes and the peer controls this
/// number, so an unchecked `packet_length` is an attacker-chosen allocation.
pub const MAX_PACKET: usize = 256 * 1024;

// --- writing ---------------------------------------------------------------

/// Builds an SSH payload. Every `put_*` appends in wire order.
#[derive(Default, Clone)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Start a payload whose first byte is a message number.
    pub fn msg(kind: u8) -> Self {
        let mut w = Self::new();
        w.put_u8(kind);
        w
    }

    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// RFC 4251 boolean: any non-zero reads as true, but we always emit 1.
    pub fn put_bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// A `string`: uint32 length then the raw bytes.
    pub fn put_string(&mut self, v: &[u8]) {
        self.put_u32(v.len() as u32);
        self.buf.extend_from_slice(v);
    }

    pub fn put_str(&mut self, v: &str) {
        self.put_string(v.as_bytes());
    }

    /// A `name-list`: the names joined with `,` inside a `string`.
    pub fn put_name_list(&mut self, names: &[&str]) {
        let mut joined = String::new();
        for (i, n) in names.iter().enumerate() {
            if i > 0 {
                joined.push(',');
            }
            joined.push_str(n);
        }
        self.put_str(&joined);
    }

    /// An `mpint`: two's-complement big-endian, minimal length.
    ///
    /// `v` is an unsigned big-endian magnitude (the only kind SSH kex produces).
    /// Leading zero bytes are stripped, and a leading `0x00` is *added* when the
    /// top bit would otherwise make the value negative.
    pub fn put_mpint(&mut self, v: &[u8]) {
        let start = v.iter().position(|&b| b != 0).unwrap_or(v.len());
        let mag = &v[start..];
        if mag.is_empty() {
            // Zero is the empty string, not a zero byte.
            self.put_u32(0);
            return;
        }
        if mag[0] & 0x80 != 0 {
            self.put_u32(mag.len() as u32 + 1);
            self.buf.push(0);
        } else {
            self.put_u32(mag.len() as u32);
        }
        self.buf.extend_from_slice(mag);
    }

    /// Append raw bytes with no length prefix.
    pub fn put_raw(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// --- reading ---------------------------------------------------------------

/// Reads an SSH payload. Every accessor is bounds-checked and returns `None`
/// rather than panicking — the input is attacker-controlled by definition.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// The bytes not yet consumed.
    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos.min(self.buf.len())..]
    }

    pub fn u8(&mut self) -> Option<u8> {
        let v = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(v)
    }

    pub fn bool(&mut self) -> Option<bool> {
        Some(self.u8()? != 0)
    }

    pub fn u32(&mut self) -> Option<u32> {
        let b = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Option<u64> {
        let b = self.buf.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Some(u64::from_be_bytes(a))
    }

    /// A length-prefixed `string`, borrowed from the input.
    pub fn string(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        // The length is the peer's claim about our buffer; a lying one must be
        // refused, never clamped — a truncated key blob that parses is worse
        // than one that does not.
        let s = self.buf.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }

    /// A `string` that must be UTF-8 (algorithm names, usernames).
    pub fn utf8(&mut self) -> Option<&'a str> {
        core::str::from_utf8(self.string()?).ok()
    }

    /// A `name-list`, split on commas. An empty list yields no names.
    pub fn name_list(&mut self) -> Option<Vec<&'a str>> {
        let s = self.utf8()?;
        if s.is_empty() {
            return Some(Vec::new());
        }
        Some(s.split(',').collect())
    }

    /// An `mpint`, returned as an unsigned big-endian magnitude.
    ///
    /// A negative value (top bit set with no leading zero) is refused: nothing
    /// in the SSH kex uses one, and silently reinterpreting it as unsigned would
    /// hide a malformed peer.
    pub fn mpint(&mut self) -> Option<&'a [u8]> {
        let s = self.string()?;
        match s.first() {
            None => Some(s),
            Some(&b) if b & 0x80 != 0 => None,
            Some(&0) => Some(&s[1..]),
            Some(_) => Some(s),
        }
    }

    /// Take `n` raw bytes.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
}

// --- binary packet ---------------------------------------------------------

/// How the length field is treated, which decides what the padding aligns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthMode {
    /// Classic encrypt-and-MAC: the length field is encrypted with everything
    /// else, so `4 + packet_length` is what must be a block multiple.
    Encrypted,
    /// AEAD (`aes*-gcm@openssh.com`, `chacha20-poly1305@openssh.com`) and
    /// encrypt-then-MAC: the length is outside the encrypted region, so
    /// `packet_length` alone is what must be a block multiple.
    Plain,
}

/// Padding length for a payload of `payload_len` at block size `block`.
///
/// RFC 4253 §6: at least [`MIN_PADDING`] bytes, and the aligned region must be a
/// multiple of `max(block, 8)`. Which region that is depends on `mode` — an
/// AEAD packet aligns without its length field, and using the wrong rule yields
/// a packet the peer rejects as badly padded on the very first encrypted
/// message.
pub fn pad_len(payload_len: usize, block: usize, mode: LengthMode) -> usize {
    let block = block.max(8);
    // padding_length byte + payload, plus the length field when it is inside.
    let unpadded = match mode {
        LengthMode::Encrypted => 4 + 1 + payload_len,
        LengthMode::Plain => 1 + payload_len,
    };
    let mut pad = block - (unpadded % block);
    if pad < MIN_PADDING {
        pad += block;
    }
    pad
}

/// Frame `payload` into an unencrypted binary packet (length, pad len, payload,
/// padding). `pad` supplies the padding bytes — random in production, fixed in
/// tests.
pub fn frame(payload: &[u8], block: usize, mode: LengthMode, pad: &mut dyn FnMut(&mut [u8])) -> Vec<u8> {
    let padding = pad_len(payload.len(), block, mode);
    let packet_len = 1 + payload.len() + padding;
    let mut out = Vec::with_capacity(4 + packet_len);
    out.extend_from_slice(&(packet_len as u32).to_be_bytes());
    out.push(padding as u8);
    out.extend_from_slice(payload);
    let at = out.len();
    out.resize(at + padding, 0);
    pad(&mut out[at..]);
    out
}

/// The payload inside a complete, decrypted binary packet.
///
/// `buf` starts at the length field. Returns `None` when the packet is
/// malformed — a padding length that overruns the packet is the classic way a
/// peer gets a parser to read past its buffer.
pub fn payload_of(buf: &[u8]) -> Option<&[u8]> {
    if buf.len() < 5 {
        return None;
    }
    let packet_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if packet_len < 2 || packet_len > MAX_PACKET || 4 + packet_len > buf.len() {
        return None;
    }
    let padding = buf[4] as usize;
    if padding < MIN_PADDING || padding + 1 > packet_len {
        return None;
    }
    let payload_len = packet_len - padding - 1;
    buf.get(5..5 + payload_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mpint` is where a kex silently produces the wrong shared secret.
    #[test_case]
    fn mpint_is_minimal_two_complement() {
        // Zero is the *empty* string.
        let mut w = Writer::new();
        w.put_mpint(&[0, 0, 0]);
        assert_eq!(w.as_slice(), &[0, 0, 0, 0]);

        // Top bit set → a leading zero byte, and the length grows to match.
        let mut w = Writer::new();
        w.put_mpint(&[0x80, 0x09]);
        assert_eq!(w.as_slice(), &[0, 0, 0, 3, 0x00, 0x80, 0x09]);

        // Top bit clear → no padding byte.
        let mut w = Writer::new();
        w.put_mpint(&[0x7f, 0xff]);
        assert_eq!(w.as_slice(), &[0, 0, 0, 2, 0x7f, 0xff]);

        // Leading zeros in the input are not leading zeros on the wire.
        let mut w = Writer::new();
        w.put_mpint(&[0x00, 0x00, 0x01, 0x02]);
        assert_eq!(w.as_slice(), &[0, 0, 0, 2, 0x01, 0x02]);

        // RFC 4251 §5's own example: 0x9a378f9b2e332a7 encodes with no pad byte.
        let mut w = Writer::new();
        w.put_mpint(&[0x09, 0xa3, 0x78, 0xf9, 0xb2, 0xe3, 0x32, 0xa7]);
        assert_eq!(
            w.as_slice(),
            &[0, 0, 0, 8, 0x09, 0xa3, 0x78, 0xf9, 0xb2, 0xe3, 0x32, 0xa7]
        );
    }

    /// And it round-trips back to the magnitude, with a negative refused.
    #[test_case]
    fn mpint_round_trips_and_refuses_negative() {
        for mag in [&[0x80u8, 0x09][..], &[0x7f, 0xff][..], &[][..], &[0x01][..]] {
            let mut w = Writer::new();
            w.put_mpint(mag);
            let v = w.into_vec();
            let mut r = Reader::new(&v);
            assert_eq!(r.mpint().expect("valid mpint"), mag, "round trip {mag:?}");
        }
        // A genuinely negative mpint (top bit set, no leading zero) is refused
        // rather than reinterpreted as a large positive.
        let neg = [0, 0, 0, 1, 0xff];
        assert!(Reader::new(&neg).mpint().is_none());
    }

    /// A `string` is binary and length-prefixed — not NUL-terminated, and not
    /// necessarily UTF-8 (it carries key blobs).
    #[test_case]
    fn strings_are_binary_and_length_prefixed() {
        let mut w = Writer::new();
        w.put_string(&[0x00, 0xff, b'a', 0x00]);
        assert_eq!(w.as_slice(), &[0, 0, 0, 4, 0x00, 0xff, b'a', 0x00]);
        let v = w.into_vec();
        let mut r = Reader::new(&v);
        assert_eq!(r.string().unwrap(), &[0x00, 0xff, b'a', 0x00]);
        assert!(r.is_empty());

        // A length that overruns the buffer is refused, not clamped.
        let lying = [0, 0, 0, 16, b'x'];
        assert!(Reader::new(&lying).string().is_none());
    }

    /// Name-lists are comma-joined with no spaces; empty means no names.
    #[test_case]
    fn name_lists_round_trip() {
        let mut w = Writer::new();
        w.put_name_list(&["curve25519-sha256", "ecdh-sha2-nistp256"]);
        let v = w.into_vec();
        assert_eq!(&v[4..], b"curve25519-sha256,ecdh-sha2-nistp256");
        let mut r = Reader::new(&v);
        assert_eq!(
            r.name_list().unwrap(),
            alloc::vec!["curve25519-sha256", "ecdh-sha2-nistp256"]
        );

        let mut w = Writer::new();
        w.put_name_list(&[]);
        let v = w.into_vec();
        assert_eq!(v, alloc::vec![0, 0, 0, 0]);
        // An empty list is zero names, not one empty name.
        assert!(Reader::new(&v).name_list().unwrap().is_empty());
    }

    /// Padding: at least 4 bytes, and the *right* region aligned for the mode.
    #[test_case]
    fn padding_aligns_the_region_the_mode_says() {
        // Classic: 4 + 1 + payload + pad is a multiple of the block size.
        for payload in 0..64usize {
            for block in [8usize, 16] {
                let p = pad_len(payload, block, LengthMode::Encrypted);
                assert!(p >= MIN_PADDING, "padding {p} below the minimum");
                assert_eq!((4 + 1 + payload + p) % block.max(8), 0, "payload {payload} block {block}");
            }
        }
        // AEAD: the length field is outside, so 1 + payload + pad aligns.
        for payload in 0..64usize {
            for block in [8usize, 16] {
                let p = pad_len(payload, block, LengthMode::Plain);
                assert!(p >= MIN_PADDING);
                assert_eq!((1 + payload + p) % block.max(8), 0, "payload {payload} block {block}");
            }
        }
        // The two modes really do differ — otherwise this test proves nothing.
        assert_ne!(
            pad_len(10, 16, LengthMode::Encrypted),
            pad_len(10, 16, LengthMode::Plain)
        );
    }

    /// A framed packet parses back to exactly its payload.
    #[test_case]
    fn frame_and_payload_round_trip() {
        for payload_len in [0usize, 1, 5, 20, 100, 255] {
            let payload: Vec<u8> = (0..payload_len).map(|i| i as u8).collect();
            for mode in [LengthMode::Encrypted, LengthMode::Plain] {
                let pkt = frame(&payload, 16, mode, &mut |p| p.fill(0xAB));
                assert_eq!(payload_of(&pkt).expect("well-formed"), &payload[..]);
            }
        }
    }

    /// A malformed packet is refused rather than read past.
    #[test_case]
    fn payload_of_refuses_malformed_packets() {
        assert!(payload_of(&[]).is_none());
        assert!(payload_of(&[0, 0, 0, 10]).is_none(), "truncated");
        // padding_length larger than the packet would underflow the payload len.
        assert!(payload_of(&[0, 0, 0, 6, 200, 1, 2, 3, 4, 5]).is_none());
        // Padding below the four-byte minimum is malformed.
        assert!(payload_of(&[0, 0, 0, 6, 1, 1, 2, 3, 4, 5]).is_none());
        // An absurd length is refused before any allocation.
        let huge = [0xff, 0xff, 0xff, 0xff, 4, 0, 0, 0, 0];
        assert!(payload_of(&huge).is_none());
    }
}
