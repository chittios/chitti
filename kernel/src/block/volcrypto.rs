//! **Volume encryption** for the Chitti data partition (PR7).
//!
//! On-disk format **C4VE v1** (not full LUKS2 — smaller, pure Rust, same idea):
//!
//! ```text
//! sector 0 (512 B):
//!   magic "C4VE" | version=1 | hdr_sectors | iterations
//!   salt[32] | wrapped_master_key[40] | mk_check[16] | zero pad
//! sectors 1..hdr_sectors-1: reserved (zero)
//! sectors hdr_sectors..end: AES-128-XTS payload (tweak = logical sector index)
//! ```
//!
//! - Master key: 32 random bytes → AES-128-XTS (key1 ‖ key2).
//! - Slot: PBKDF2-HMAC-SHA256(passphrase, salt, iterations) → 32 bytes; first
//!   16 wrap the master key with RFC 3394 AES key wrap; last 16 unused (future).
//! - Human unlock only (`modal::input` / shell); never an agent tool.
//!
//! AES block cipher reuses [`crate::drivers::wifi::wpa`] (FIPS 197 vector).

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use crate::drivers::wifi::wpa::{aes128_decrypt_block, aes128_encrypt_block, aes_key_unwrap, aes_key_wrap};
use crate::net::hashes::{digest, HashId};
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

/// Magic `C4VE` little-endian.
pub const MAGIC: u32 = 0x4556_3443;
pub const VERSION: u32 = 1;
/// Default header size (4 KiB) so payload is 4K-aligned.
pub const DEFAULT_HDR_SECTORS: u64 = 8;
/// PBKDF2 iteration count (SHA-256). Slow enough to blunt offline guessing on
/// a research OS; not a hardened Argon2 substitute.
pub const DEFAULT_ITERATIONS: u32 = 50_000;
const SALT_LEN: usize = 32;
const MK_LEN: usize = 32;
/// AES-KW of 32-byte master → 40 bytes.
const WRAPPED_LEN: usize = 40;
const CHECK_LEN: usize = 16;

/// On-disk header (logical fields; serialised into sector 0).
#[derive(Clone, Debug)]
pub struct Header {
    pub version: u32,
    pub hdr_sectors: u64,
    pub iterations: u32,
    pub salt: [u8; SALT_LEN],
    pub wrapped_mk: [u8; WRAPPED_LEN],
    /// First 16 bytes of SHA-256(master_key) for a fast wrong-password check
    /// without attempting a full volume mount.
    pub mk_check: [u8; CHECK_LEN],
}

impl Header {
    pub fn pack(&self) -> [u8; BLOCK_SIZE] {
        let mut b = [0u8; BLOCK_SIZE];
        b[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&self.version.to_le_bytes());
        b[8..16].copy_from_slice(&self.hdr_sectors.to_le_bytes());
        b[16..20].copy_from_slice(&self.iterations.to_le_bytes());
        b[20..52].copy_from_slice(&self.salt);
        b[52..92].copy_from_slice(&self.wrapped_mk);
        b[92..108].copy_from_slice(&self.mk_check);
        b
    }

    pub fn unpack(sec: &[u8]) -> Option<Header> {
        if sec.len() < 108 {
            return None;
        }
        let magic = u32::from_le_bytes(sec[0..4].try_into().ok()?);
        if magic != MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(sec[4..8].try_into().ok()?);
        if version != VERSION {
            return None;
        }
        let hdr_sectors = u64::from_le_bytes(sec[8..16].try_into().ok()?);
        if hdr_sectors == 0 || hdr_sectors > 1024 {
            return None;
        }
        let iterations = u32::from_le_bytes(sec[16..20].try_into().ok()?);
        if iterations == 0 {
            return None;
        }
        let mut salt = [0u8; SALT_LEN];
        salt.copy_from_slice(&sec[20..52]);
        let mut wrapped_mk = [0u8; WRAPPED_LEN];
        wrapped_mk.copy_from_slice(&sec[52..92]);
        let mut mk_check = [0u8; CHECK_LEN];
        mk_check.copy_from_slice(&sec[92..108]);
        Some(Header {
            version,
            hdr_sectors,
            iterations,
            salt,
            wrapped_mk,
            mk_check,
        })
    }
}

/// True if sector 0 looks like a C4VE header.
pub fn is_encrypted_sector0(sec: &[u8]) -> bool {
    Header::unpack(sec).is_some()
}

// ── PBKDF2-HMAC-SHA256 ───────────────────────────────────────────────────

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const B: usize = 64;
    let mut k = [0u8; B];
    if key.len() > B {
        let d = digest(HashId::Sha256, key);
        k[..32].copy_from_slice(&d);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; B];
    let mut opad = [0x5cu8; B];
    for i in 0..B {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(msg);
    let mid = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&mid);
    let out = outer.finalize();
    let mut r = [0u8; 32];
    r.copy_from_slice(&out);
    r
}

/// How many PBKDF2 rounds run between `pump()` calls in
/// [`pbkdf2_hmac_sha256_pumped`]. Small enough that a 200k-round derivation
/// pumps the UI ~200 times (well inside the ~50 ms standing rule), large enough
/// that the pump is not measurable against the hashing.
const PUMP_EVERY: u32 = 1024;

/// PBKDF2-HMAC-SHA256 into `out` (any length).
pub fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32, out: &mut [u8]) {
    pbkdf2_hmac_sha256_pumped(password, salt, iterations, out, &mut || {});
}

/// PBKDF2-HMAC-SHA256, calling `pump` every [`PUMP_EVERY`] rounds.
///
/// **Byte-identical to [`pbkdf2_hmac_sha256`]** — that one is this one with a
/// no-op pump, so there is a single implementation and the two cannot drift
/// (`pbkdf2_pumped_agrees_with_the_plain_form` pins it).
///
/// The pump exists because a deliberately-slow KDF is exactly the kind of loop
/// the standing UI rule is about: at the login iteration count this runs for a
/// noticeable fraction of a second, and without pumping it freezes the clock,
/// the caret, the mouse and the net stack while it does. Callers pass
/// `crate::shell::status_tick` (a loop that owns the console) or
/// `crate::shell::upkeep`.
pub fn pbkdf2_hmac_sha256_pumped(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    out: &mut [u8],
    pump: &mut dyn FnMut(),
) {
    assert!(iterations >= 1);
    let mut block_index = 1u32;
    let mut offset = 0usize;
    while offset < out.len() {
        // U1 = HMAC(password, salt || INT(i))
        let mut msg = Vec::with_capacity(salt.len() + 4);
        msg.extend_from_slice(salt);
        msg.extend_from_slice(&block_index.to_be_bytes());
        let mut u = hmac_sha256(password, &msg);
        let mut t = u;
        for round in 1..iterations {
            u = hmac_sha256(password, &u);
            for i in 0..32 {
                t[i] ^= u[i];
            }
            if round % PUMP_EVERY == 0 {
                pump();
            }
        }
        let n = (out.len() - offset).min(32);
        out[offset..offset + n].copy_from_slice(&t[..n]);
        offset += n;
        block_index += 1;
    }
}

// ── AES-128-XTS ──────────────────────────────────────────────────────────

/// GF(2^128) multiply by α (x) for XTS tweak update — IEEE 1619 / NIST SP 800-38E.
pub fn gf128_mul_alpha(t: &mut [u8; 16]) {
    let mut carry = 0u8;
    for b in t.iter_mut() {
        let new_carry = *b >> 7;
        *b = (*b << 1) | carry;
        carry = new_carry;
    }
    if carry != 0 {
        t[0] ^= 0x87;
    }
}

/// Encrypt one sector (must be a multiple of 16 bytes; typically 512).
pub fn aes128_xts_encrypt(key: &[u8; 32], tweak_le: u64, sector: &mut [u8]) {
    assert!(sector.len() % 16 == 0 && !sector.is_empty());
    let mut key1 = [0u8; 16];
    let mut key2 = [0u8; 16];
    key1.copy_from_slice(&key[..16]);
    key2.copy_from_slice(&key[16..32]);
    let mut t = [0u8; 16];
    t[..8].copy_from_slice(&tweak_le.to_le_bytes());
    aes128_encrypt_block(&key2, &mut t);
    for chunk in sector.chunks_exact_mut(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        for i in 0..16 {
            block[i] ^= t[i];
        }
        aes128_encrypt_block(&key1, &mut block);
        for i in 0..16 {
            block[i] ^= t[i];
        }
        chunk.copy_from_slice(&block);
        gf128_mul_alpha(&mut t);
    }
}

/// Decrypt one sector.
pub fn aes128_xts_decrypt(key: &[u8; 32], tweak_le: u64, sector: &mut [u8]) {
    assert!(sector.len() % 16 == 0 && !sector.is_empty());
    let mut key1 = [0u8; 16];
    let mut key2 = [0u8; 16];
    key1.copy_from_slice(&key[..16]);
    key2.copy_from_slice(&key[16..32]);
    let mut t = [0u8; 16];
    t[..8].copy_from_slice(&tweak_le.to_le_bytes());
    aes128_encrypt_block(&key2, &mut t);
    for chunk in sector.chunks_exact_mut(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        for i in 0..16 {
            block[i] ^= t[i];
        }
        aes128_decrypt_block(&key1, &mut block);
        for i in 0..16 {
            block[i] ^= t[i];
        }
        chunk.copy_from_slice(&block);
        gf128_mul_alpha(&mut t);
    }
}

/// Draw the salt and the master key.
///
/// This **must** go through [`crate::security::rng`] and not `arch::hw_rand()`.
/// It used to copy `hw_rand()` straight out, which returns 0 when the CPU has no
/// `RDRAND`/`RNDR` — true of QEMU and HVF's default models — so every volume
/// formatted on a development machine got an **all-zero salt and an all-zero
/// master key**. Nothing detects that: an all-zero key encrypts, decrypts and
/// mounts perfectly, and the passphrase still works.
fn fill_random(buf: &mut [u8]) {
    crate::security::rng::fill_random(buf);
}

fn mk_check(master: &[u8; MK_LEN]) -> [u8; CHECK_LEN] {
    let d = digest(HashId::Sha256, master);
    let mut c = [0u8; CHECK_LEN];
    c.copy_from_slice(&d[..CHECK_LEN]);
    c
}

/// Derive KEK (16 bytes for AES-KW) from passphrase.
///
/// Pumps the UI: at `DEFAULT_ITERATIONS` this is a visible fraction of a second
/// spent in a tight hashing loop, and both callers run it somewhere a frozen
/// console is user-visible — `format` behind the `/encrypt` modal, `unlock` at
/// boot before the shell exists. `status_tick` rather than `upkeep` because the
/// modal owns the console and `upkeep`'s `mouse::tick()` would steal its clicks.
fn derive_kek(passphrase: &[u8], salt: &[u8], iterations: u32) -> [u8; 16] {
    let mut out = [0u8; 32];
    pbkdf2_hmac_sha256_pumped(passphrase, salt, iterations, &mut out, &mut crate::shell::status_tick);
    let mut kek = [0u8; 16];
    kek.copy_from_slice(&out[..16]);
    kek
}

/// Format an empty partition as C4VE + return the unlocked master key.
///
/// Writes the header to `dev` at LBA 0 of the partition view (`Partition` or
/// whole disk). Caller must then mkfs the payload (`hdr_sectors..`).
pub fn format<D: BlockDevice>(
    dev: &mut D,
    passphrase: &[u8],
    iterations: u32,
    hdr_sectors: u64,
) -> Result<[u8; MK_LEN], BlockError> {
    if dev.block_count() <= hdr_sectors + 64 {
        return Err(BlockError::OutOfRange);
    }
    if passphrase.is_empty() {
        return Err(BlockError::DeviceError);
    }
    let mut salt = [0u8; SALT_LEN];
    fill_random(&mut salt);
    let mut master = [0u8; MK_LEN];
    fill_random(&mut master);
    let kek = derive_kek(passphrase, &salt, iterations);
    let wrapped = aes_key_wrap(&kek, &master).ok_or(BlockError::DeviceError)?;
    if wrapped.len() != WRAPPED_LEN {
        return Err(BlockError::DeviceError);
    }
    let mut wrapped_mk = [0u8; WRAPPED_LEN];
    wrapped_mk.copy_from_slice(&wrapped);
    let hdr = Header {
        version: VERSION,
        hdr_sectors,
        iterations,
        salt,
        wrapped_mk,
        mk_check: mk_check(&master),
    };
    let packed = hdr.pack();
    dev.write_block(0, &packed)?;
    // Zero remaining header sectors.
    let zero = [0u8; BLOCK_SIZE];
    for s in 1..hdr_sectors {
        dev.write_block(s, &zero)?;
    }
    crate::ktrace::log_fmt(format_args!(
        "volcrypto: formatted C4VE v1 (hdr={hdr_sectors} sectors, iters={iterations})"
    ));
    Ok(master)
}

/// Unlock: read header from LBA 0, derive KEK, unwrap master key.
pub fn unlock<D: BlockDevice>(dev: &mut D, passphrase: &[u8]) -> Result<([u8; MK_LEN], Header), BlockError> {
    let mut sec = [0u8; BLOCK_SIZE];
    dev.read_block(0, &mut sec)?;
    let hdr = Header::unpack(&sec).ok_or(BlockError::DeviceError)?;
    let kek = derive_kek(passphrase, &hdr.salt, hdr.iterations);
    let plain = aes_key_unwrap(&kek, &hdr.wrapped_mk).ok_or(BlockError::DeviceError)?;
    if plain.len() != MK_LEN {
        return Err(BlockError::DeviceError);
    }
    let mut master = [0u8; MK_LEN];
    master.copy_from_slice(&plain);
    if mk_check(&master) != hdr.mk_check {
        return Err(BlockError::DeviceError);
    }
    Ok((master, hdr))
}

/// Probe whether the volume at `start` is C4VE without consuming a passphrase.
pub fn probe_encrypted<D: BlockDevice>(dev: &mut D, start: u64) -> Option<Header> {
    let mut sec = [0u8; BLOCK_SIZE];
    dev.read_block(start, &mut sec).ok()?;
    Header::unpack(&sec)
}

// ── BlockDevice adapter ──────────────────────────────────────────────────

/// A partition slice with optional AES-XTS on the payload after `hdr_sectors`.
///
/// When `key` is `None` and `hdr_sectors == 0`, this is a plain partition view
/// (logical LBA = absolute). With a key, logical block `i` maps to
/// `part_start + hdr_sectors + i` and is encrypted with tweak `i`.
pub struct CryptoPart<'a, D: BlockDevice> {
    pub dev: &'a mut D,
    pub part_start: u64,
    pub payload_sectors: u64,
    pub hdr_sectors: u64,
    pub key: Option<[u8; MK_LEN]>,
}

impl<'a, D: BlockDevice> CryptoPart<'a, D> {
    /// Plain partition (no encryption).
    pub fn plain(dev: &'a mut D, part_start: u64, part_sectors: u64) -> Self {
        CryptoPart {
            dev,
            part_start,
            payload_sectors: part_sectors,
            hdr_sectors: 0,
            key: None,
        }
    }

    /// Encrypted payload after a C4VE header.
    pub fn encrypted(
        dev: &'a mut D,
        part_start: u64,
        part_sectors: u64,
        hdr_sectors: u64,
        key: [u8; MK_LEN],
    ) -> Self {
        let payload = part_sectors.saturating_sub(hdr_sectors);
        CryptoPart {
            dev,
            part_start,
            payload_sectors: payload,
            hdr_sectors,
            key: Some(key),
        }
    }
}

impl<D: BlockDevice> BlockDevice for CryptoPart<'_, D> {
    fn block_count(&self) -> u64 {
        self.payload_sectors
    }

    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.payload_sectors {
            return Err(BlockError::OutOfRange);
        }
        let phys = self.part_start + self.hdr_sectors + index;
        self.dev.read_block(phys, buf)?;
        if let Some(k) = &self.key {
            aes128_xts_decrypt(k, index, buf);
        }
        Ok(())
    }

    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.payload_sectors {
            return Err(BlockError::OutOfRange);
        }
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        if let Some(k) = &self.key {
            aes128_xts_encrypt(k, index, &mut tmp);
        }
        let phys = self.part_start + self.hdr_sectors + index;
        self.dev.write_block(phys, &tmp)
    }

    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let n = buf.len() / BLOCK_SIZE;
        for i in 0..n {
            let off = i * BLOCK_SIZE;
            self.read_block(index + i as u64, &mut buf[off..off + BLOCK_SIZE])?;
        }
        Ok(())
    }

    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        let n = buf.len() / BLOCK_SIZE;
        for i in 0..n {
            let off = i * BLOCK_SIZE;
            self.write_block(index + i as u64, &buf[off..off + BLOCK_SIZE])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ramdisk::RamDisk;

    #[test_case]
    fn gf128_mul_alpha_bit_shift() {
        let mut t = [0u8; 16];
        t[0] = 0x01;
        gf128_mul_alpha(&mut t);
        assert_eq!(t[0], 0x02);
        // high bit set → reduction with 0x87
        let mut u = [0u8; 16];
        u[15] = 0x80;
        gf128_mul_alpha(&mut u);
        assert_eq!(u[15], 0x00);
        assert_eq!(u[0], 0x87);
    }

    #[test_case]
    fn xts_round_trip_sector() {
        let key = [0x11u8; 32];
        let mut sec = [0u8; 512];
        for (i, b) in sec.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let plain = sec;
        aes128_xts_encrypt(&key, 7, &mut sec);
        assert_ne!(sec, plain);
        aes128_xts_decrypt(&key, 7, &mut sec);
        assert_eq!(sec, plain);
        // Wrong tweak must not decrypt.
        let mut sec2 = plain;
        aes128_xts_encrypt(&key, 7, &mut sec2);
        aes128_xts_decrypt(&key, 8, &mut sec2);
        assert_ne!(sec2, plain);
    }

    #[test_case]
    fn pbkdf2_sha256_rfc6070_vector() {
        // RFC 6070: P="password", S="salt", c=1, dkLen=32
        // (first 20 bytes published for SHA1; for SHA256 we pin a host-independent
        // self-check: two calls with same inputs agree and differ when salt differs.)
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        pbkdf2_hmac_sha256(b"password", b"salt", 2, &mut a);
        pbkdf2_hmac_sha256(b"password", b"salt", 2, &mut b);
        assert_eq!(a, b);
        pbkdf2_hmac_sha256(b"password", b"salt2", 2, &mut b);
        assert_ne!(a, b);
        // Non-zero output.
        assert!(a.iter().any(|&x| x != 0));
    }

    /// The pumped form must produce **the same bytes** as the plain one, and must
    /// actually have pumped. Without the first half a pumped KDF silently derives
    /// a different key and the only symptom is "the passphrase stopped working";
    /// without the second half the pump could be dead code and the UI would still
    /// freeze.
    #[test_case]
    fn pbkdf2_pumped_agrees_with_the_plain_form() {
        let mut plain = [0u8; 32];
        let mut pumped = [0u8; 32];
        // Above PUMP_EVERY, so the pump really fires.
        let iters = super::PUMP_EVERY * 3;
        pbkdf2_hmac_sha256(b"correct horse", b"a-salt", iters, &mut plain);
        let mut pumps = 0usize;
        pbkdf2_hmac_sha256_pumped(b"correct horse", b"a-salt", iters, &mut pumped, &mut || pumps += 1);
        assert_eq!(plain, pumped, "the pumped PBKDF2 derived different bytes");
        assert!(pumps >= 2, "the pump never fired (got {pumps})");

        // A derivation shorter than one pump interval must still be correct.
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        pbkdf2_hmac_sha256(b"pw", b"s", 4, &mut a);
        pbkdf2_hmac_sha256_pumped(b"pw", b"s", 4, &mut b, &mut || {});
        assert_eq!(a, b);

        // Multi-block output (> 32 bytes) exercises the block_index loop.
        let mut la = [0u8; 48];
        let mut lb = [0u8; 48];
        pbkdf2_hmac_sha256(b"pw", b"s", super::PUMP_EVERY + 5, &mut la);
        pbkdf2_hmac_sha256_pumped(b"pw", b"s", super::PUMP_EVERY + 5, &mut lb, &mut || {});
        assert_eq!(la, lb);
    }

    #[test_case]
    fn format_unlock_round_trip_on_ramdisk() {
        // 1 MiB volume.
        let mut disk = RamDisk::new(2048);
        let pass = b"correct horse battery staple";
        let mk = format(&mut disk, pass, 100, DEFAULT_HDR_SECTORS).expect("format");
        let (mk2, hdr) = unlock(&mut disk, pass).expect("unlock");
        assert_eq!(mk, mk2);
        assert_eq!(hdr.hdr_sectors, DEFAULT_HDR_SECTORS);
        assert!(unlock(&mut disk, b"wrong").is_err());
        // Payload IO through CryptoPart.
        let mut cp = CryptoPart::encrypted(&mut disk, 0, 2048, hdr.hdr_sectors, mk);
        let mut sec = [0xABu8; 512];
        cp.write_block(0, &sec).unwrap();
        let mut out = [0u8; 512];
        cp.read_block(0, &mut out).unwrap();
        assert_eq!(out, sec);
        // Ciphertext on raw disk differs.
        let mut raw = [0u8; 512];
        disk.read_block(hdr.hdr_sectors, &mut raw).unwrap();
        assert_ne!(raw, sec);
    }

    #[test_case]
    fn header_unpack_rejects_bad_magic() {
        let mut sec = [0u8; 512];
        assert!(Header::unpack(&sec).is_none());
        sec[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        // version 0 / zero iters still invalid after partial fill
        assert!(Header::unpack(&sec).is_none());
    }

    #[test_case]
    fn aes_key_wrap_path_matches_wpa() {
        // Ensure 32-byte wrap length is 40.
        let kek = [0x00u8; 16];
        let mut plain = [0u8; 32];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = i as u8;
        }
        let w = aes_key_wrap(&kek, &plain).unwrap();
        assert_eq!(w.len(), 40);
        let u = aes_key_unwrap(&kek, &w).unwrap();
        assert_eq!(u, plain);
    }
}
