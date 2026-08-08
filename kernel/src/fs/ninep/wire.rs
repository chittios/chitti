//! **9P2000.L wire codec** — pure encode/decode over byte slices, no device.
//!
//! Every message is `size[4] type[1] tag[2]` then a body of fixed-width
//! little-endian integers, length-prefixed strings and 13-byte qids. Message
//! numbers and field orders here come from Linux's `include/net/9p/9p.h` and
//! the 9P2000.L protocol document — **fetched, not recalled**, which is the
//! standing rule this tree learned from `iwlwifi` writing a plausible wrong
//! struct from memory.
//!
//! Encoding writes into a caller's buffer rather than allocating, because that
//! buffer is the DMA region the device reads out of: building a `Vec` and
//! copying would double the cost of every read on a path that is meant to move
//! whole files.
//!
//! Three properties of this protocol cause most of the bugs, and each has a
//! test below:
//!
//! * **`size` counts itself.** A four-byte message body is a `size` of 11, not
//!   7. Getting it wrong desynchronises the stream rather than erroring.
//! * **Strings are `length[2]`-prefixed and not NUL-terminated**, so a name is
//!   bounded at 65535 bytes and must be refused rather than truncated — a
//!   truncated name addresses a *different, existing* file.
//! * **A short `Rwalk` is a success reply, not an error.** Walking four
//!   elements and receiving three qids means the fourth does not exist; a
//!   client that only checks the message type reports the parent as the child.

use alloc::string::String;
use alloc::vec::Vec;

// --- message types (Linux include/net/9p/9p.h, p9_msg_t) ---
pub const T_LERROR: u8 = 6;
pub const R_LERROR: u8 = 7;
pub const T_STATFS: u8 = 8;
pub const R_STATFS: u8 = 9;
pub const T_LOPEN: u8 = 12;
pub const R_LOPEN: u8 = 13;
pub const T_LCREATE: u8 = 14;
pub const R_LCREATE: u8 = 15;
pub const T_GETATTR: u8 = 24;
pub const R_GETATTR: u8 = 25;
pub const T_SETATTR: u8 = 26;
pub const R_SETATTR: u8 = 27;
pub const T_READDIR: u8 = 40;
pub const R_READDIR: u8 = 41;
pub const T_FSYNC: u8 = 50;
pub const R_FSYNC: u8 = 51;
pub const T_MKDIR: u8 = 72;
pub const R_MKDIR: u8 = 73;
pub const T_RENAMEAT: u8 = 74;
pub const R_RENAMEAT: u8 = 75;
pub const T_UNLINKAT: u8 = 76;
pub const R_UNLINKAT: u8 = 77;
pub const T_VERSION: u8 = 100;
pub const R_VERSION: u8 = 101;
pub const T_ATTACH: u8 = 104;
pub const R_ATTACH: u8 = 105;
pub const T_WALK: u8 = 110;
pub const R_WALK: u8 = 111;
pub const T_READ: u8 = 116;
pub const R_READ: u8 = 117;
pub const T_WRITE: u8 = 118;
pub const R_WRITE: u8 = 119;
pub const T_CLUNK: u8 = 120;
pub const R_CLUNK: u8 = 121;

/// `size[4] type[1] tag[2]` — every message starts with these seven bytes.
pub const HDR: usize = 7;

/// A 9P qid is 13 bytes on the wire.
pub const QID_LEN: usize = 13;

/// Most walk elements one `Twalk` may carry (`P9_MAXWELEM`). A deeper path
/// needs several walks, which is a real constraint and not a tunable.
pub const MAX_WELEM: usize = 16;

/// The version string this client speaks. A server that answers anything else
/// (notably `9P2000` or `9P2000.u`) is refused rather than driven with L-only
/// messages it will not understand.
pub const VERSION_L: &str = "9P2000.L";

// --- getattr request masks ---
pub const GETATTR_MODE: u64 = 0x0000_0001;
pub const GETATTR_SIZE: u64 = 0x0000_0200;
/// Everything a `stat` needs, and what Linux's client asks for by default.
pub const GETATTR_BASIC: u64 = 0x0000_07ff;

// --- qid type bits ---
pub const QTDIR: u8 = 0x80;
pub const QTSYMLINK: u8 = 0x02;

// --- Linux open flags used by Tlopen/Tlcreate ---
pub const O_RDONLY: u32 = 0;
pub const O_WRONLY: u32 = 1;
pub const O_RDWR: u32 = 2;
pub const O_CREAT: u32 = 0o100;
pub const O_TRUNC: u32 = 0o1000;

/// `Tunlinkat` flag meaning "the target is a directory" (`AT_REMOVEDIR`).
pub const AT_REMOVEDIR: u32 = 0x200;

/// A server-side file identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qid {
    pub typ: u8,
    pub version: u32,
    pub path: u64,
}

impl Qid {
    pub fn is_dir(&self) -> bool {
        self.typ & QTDIR != 0
    }
    pub fn is_symlink(&self) -> bool {
        self.typ & QTSYMLINK != 0
    }
}

/// What went wrong with a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P9Error {
    /// The server answered `Rlerror` with this Linux errno.
    Server(u32),
    /// A reply was shorter than its own fields, or a field ran off the end.
    Malformed,
    /// The reply was well-formed but of an unexpected type.
    Unexpected(u8),
    /// The message did not fit the buffer / negotiated `msize`.
    TooLarge,
    /// The transport failed to exchange the message at all.
    Transport,
    /// The server does not speak 9P2000.L.
    Version,
}

/// Writes a message into a caller-provided buffer.
pub struct Enc<'a> {
    buf: &'a mut [u8],
    pos: usize,
    /// Sticky: once a write has overflowed, everything after it is skipped and
    /// [`Enc::finish`] fails. Checking once at the end beats checking at every
    /// field and silently producing a half message.
    ok: bool,
}

impl<'a> Enc<'a> {
    /// Begin a message of `typ` with `tag`, reserving the size field.
    pub fn start(buf: &'a mut [u8], typ: u8, tag: u16) -> Enc<'a> {
        let mut e = Enc { buf, pos: 0, ok: true };
        e.u32(0); // size, backfilled by `finish`
        e.u8(typ);
        e.u16(tag);
        e
    }

    fn put(&mut self, bytes: &[u8]) {
        if !self.ok || self.pos + bytes.len() > self.buf.len() {
            self.ok = false;
            return;
        }
        self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
    }

    pub fn u8(&mut self, v: u8) {
        self.put(&[v]);
    }
    pub fn u16(&mut self, v: u16) {
        self.put(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.put(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.put(&v.to_le_bytes());
    }

    /// A `length[2]`-prefixed string. A name that does not fit the 16-bit
    /// length is **refused**, never truncated: a truncated name is a valid name
    /// for a different file, so writing one would address the wrong thing.
    pub fn str(&mut self, s: &str) {
        if s.len() > u16::MAX as usize {
            self.ok = false;
            return;
        }
        self.u16(s.len() as u16);
        self.put(s.as_bytes());
    }

    /// Raw bytes with no length prefix (the payload of a `Twrite`).
    pub fn raw(&mut self, b: &[u8]) {
        self.put(b);
    }

    /// Bytes written so far, including the header.
    pub fn len(&self) -> usize {
        self.pos
    }

    /// Backfill `size` and return the total message length.
    pub fn finish(self) -> Option<usize> {
        if !self.ok {
            return None;
        }
        // `size` counts itself — the whole message, header included.
        let total = self.pos as u32;
        self.buf[0..4].copy_from_slice(&total.to_le_bytes());
        Some(self.pos)
    }
}

/// Reads fields out of a reply.
pub struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    pub fn new(buf: &'a [u8]) -> Dec<'a> {
        Dec { buf, pos: 0 }
    }

    /// Start reading a message body, returning `(type, tag)` after validating
    /// that `size` agrees with the bytes actually present.
    pub fn header(buf: &'a [u8]) -> Result<(u8, u16, Dec<'a>), P9Error> {
        if buf.len() < HDR {
            return Err(P9Error::Malformed);
        }
        let size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        // A `size` larger than what arrived means a truncated reply; smaller is
        // fine (the transport may hand us a whole buffer) but the message ends
        // where `size` says, so trailing bytes are never read as fields.
        if size < HDR || size > buf.len() {
            return Err(P9Error::Malformed);
        }
        let typ = buf[4];
        let tag = u16::from_le_bytes([buf[5], buf[6]]);
        Ok((typ, tag, Dec { buf: &buf[..size], pos: HDR }))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], P9Error> {
        if self.pos + n > self.buf.len() {
            return Err(P9Error::Malformed);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8, P9Error> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, P9Error> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    pub fn u32(&mut self) -> Result<u32, P9Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub fn u64(&mut self) -> Result<u64, P9Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// A `length[2]`-prefixed string. Invalid UTF-8 is replaced rather than
    /// rejected: a host filesystem may legitimately hold a name we cannot
    /// represent, and refusing to list the directory at all would be worse than
    /// showing one entry with a replacement character.
    pub fn str(&mut self) -> Result<String, P9Error> {
        let n = self.u16()? as usize;
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    pub fn qid(&mut self) -> Result<Qid, P9Error> {
        let typ = self.u8()?;
        let version = self.u32()?;
        let path = self.u64()?;
        Ok(Qid { typ, version, path })
    }

    /// The remaining bytes of the message.
    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    /// Bytes consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Classify a reply: an `Rlerror` becomes [`P9Error::Server`], a reply of the
/// wrong type becomes [`P9Error::Unexpected`], and the expected type yields its
/// body decoder.
pub fn expect<'a>(buf: &'a [u8], want: u8) -> Result<Dec<'a>, P9Error> {
    let (typ, _tag, mut d) = Dec::header(buf)?;
    if typ == R_LERROR {
        return Err(P9Error::Server(d.u32()?));
    }
    if typ != want {
        return Err(P9Error::Unexpected(typ));
    }
    Ok(d)
}

/// One entry of an `Rreaddir` payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry9 {
    pub qid: Qid,
    /// Cookie to pass as the next `Treaddir` offset — **this entry's**
    /// position, so resuming uses the last one seen.
    pub offset: u64,
    /// Linux `DT_*` file type.
    pub typ: u8,
    pub name: String,
}

/// Parse the packed entry stream inside an `Rreaddir` payload.
///
/// A truncated trailing entry is **dropped, not an error**: the server fills up
/// to `count` bytes and the client resumes from the last complete entry's
/// offset, so a partial tail is the normal end-of-buffer condition rather than
/// corruption.
pub fn parse_dirents(data: &[u8]) -> Vec<DirEntry9> {
    let mut out = Vec::new();
    let mut d = Dec::new(data);
    loop {
        let start = d.pos();
        let Ok(qid) = d.qid() else { break };
        let Ok(offset) = d.u64() else { break };
        let Ok(typ) = d.u8() else { break };
        let Ok(name) = d.str() else { break };
        // A zero-length step would loop forever on a malformed stream.
        if d.pos() == start {
            break;
        }
        out.push(DirEntry9 { qid, offset, typ, name });
    }
    out
}

/// Attributes from an `Rgetattr`, narrowed to the fields a `stat` needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Attr {
    pub qid: Qid,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u64,
    pub size: u64,
    pub atime: u64,
    pub mtime: u64,
    pub ctime: u64,
}

/// Decode `Rgetattr`. The field order is fixed and long; reading it as anything
/// other than the exact sequence below silently reports one attribute as
/// another (the `rdev`-for-`size` class of bug).
pub fn parse_getattr(buf: &[u8]) -> Result<Attr, P9Error> {
    let mut d = expect(buf, R_GETATTR)?;
    let _valid = d.u64()?;
    let qid = d.qid()?;
    let mode = d.u32()?;
    let uid = d.u32()?;
    let gid = d.u32()?;
    let nlink = d.u64()?;
    let _rdev = d.u64()?;
    let size = d.u64()?;
    let _blksize = d.u64()?;
    let _blocks = d.u64()?;
    let atime = d.u64()?;
    let _atime_ns = d.u64()?;
    let mtime = d.u64()?;
    let _mtime_ns = d.u64()?;
    let ctime = d.u64()?;
    let _ctime_ns = d.u64()?;
    Ok(Attr { qid, mode, uid, gid, nlink, size, atime, mtime, ctime })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test_case]
    fn size_counts_itself() {
        // Tclunk is header + one u32 fid = 11 bytes, and `size` says 11 — not
        // 4 (the body) and not 7 (the header). An off-by-four here does not
        // error, it desynchronises the stream.
        let mut buf = [0u8; 64];
        let mut e = Enc::start(&mut buf, T_CLUNK, 1);
        e.u32(7);
        let n = e.finish().unwrap();
        assert_eq!(n, HDR + 4);
        assert_eq!(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 11);
        assert_eq!(buf[4], T_CLUNK);
        assert_eq!(u16::from_le_bytes([buf[5], buf[6]]), 1);
        // And it round-trips: the decoder agrees on where the body starts.
        let (typ, tag, mut d) = Dec::header(&buf[..n]).unwrap();
        assert_eq!((typ, tag), (T_CLUNK, 1));
        assert_eq!(d.u32().unwrap(), 7);
    }

    #[test_case]
    fn a_string_is_length_prefixed_not_terminated() {
        let mut buf = [0u8; 64];
        let mut e = Enc::start(&mut buf, T_VERSION, 0xffff);
        e.u32(8192);
        e.str(VERSION_L);
        let n = e.finish().unwrap();
        // 4 msize + 2 length + 8 chars, no NUL.
        assert_eq!(n, HDR + 4 + 2 + VERSION_L.len());
        assert_eq!(buf[n - 1], b'L');
        let (typ, tag, mut d) = Dec::header(&buf[..n]).unwrap();
        assert_eq!((typ, tag), (T_VERSION, 0xffff));
        assert_eq!(d.u32().unwrap(), 8192);
        assert_eq!(d.str().unwrap(), VERSION_L);
    }

    #[test_case]
    fn an_overlong_name_is_refused_never_truncated() {
        // A truncated name is a valid name for a *different* file, so silently
        // shortening one would address the wrong thing.
        let mut buf = vec![0u8; 200_000];
        let long = alloc::string::String::from_utf8(vec![b'a'; 70_000]).unwrap();
        let mut e = Enc::start(&mut buf, T_WALK, 0);
        e.str(&long);
        assert_eq!(e.finish(), None);
    }

    #[test_case]
    fn a_full_buffer_fails_the_whole_message_not_half_of_it() {
        // The encoder is sticky: an overflow part-way must not produce a
        // well-formed prefix that the server would happily act on.
        let mut buf = [0u8; 12];
        let mut e = Enc::start(&mut buf, T_WRITE, 0);
        e.u32(1);
        e.u64(0); // overflows a 12-byte buffer
        assert_eq!(e.finish(), None);
    }

    #[test_case]
    fn rlerror_becomes_a_server_errno() {
        // ENOENT is 2 on Linux; a client that only checked the message type
        // would read the errno as a qid.
        let mut buf = [0u8; 32];
        let mut e = Enc::start(&mut buf, R_LERROR, 3);
        e.u32(2);
        let n = e.finish().unwrap();
        assert_eq!(expect(&buf[..n], R_LOPEN).err(), Some(P9Error::Server(2)));
        // A reply of some other type is distinguishable from an error reply.
        let mut buf2 = [0u8; 32];
        let e2 = Enc::start(&mut buf2, R_CLUNK, 3);
        let n2 = e2.finish().unwrap();
        assert_eq!(expect(&buf2[..n2], R_LOPEN).err(), Some(P9Error::Unexpected(R_CLUNK)));
    }

    #[test_case]
    fn a_truncated_reply_is_malformed_not_silently_short() {
        // `size` claims more bytes than arrived.
        let mut buf = [0u8; 32];
        let mut e = Enc::start(&mut buf, R_GETATTR, 0);
        e.u64(0);
        let n = e.finish().unwrap();
        assert_eq!(Dec::header(&buf[..n - 1]).map(|_| ()), Err(P9Error::Malformed));
        // A field running past the end is caught too.
        assert_eq!(parse_getattr(&buf[..n]), Err(P9Error::Malformed));
    }

    #[test_case]
    fn dirents_parse_as_a_packed_stream_and_drop_a_partial_tail() {
        let mut buf = [0u8; 256];
        let mut e = Enc::start(&mut buf, R_READDIR, 0);
        // Two whole entries, then a deliberately truncated third.
        for (i, name) in [".", "hello.txt"].iter().enumerate() {
            e.u8(if i == 0 { QTDIR } else { 0 });
            e.u32(0);
            e.u64(100 + i as u64);
            e.u64(i as u64 + 1); // offset cookie
            e.u8(if i == 0 { 4 } else { 8 }); // DT_DIR / DT_REG
            e.str(name);
        }
        let whole = e.len();
        e.u8(QTDIR);
        e.u32(0); // a qid cut off mid-field
        let n = e.finish().unwrap();

        let full = parse_dirents(&buf[HDR..whole]);
        assert_eq!(full.len(), 2);
        assert_eq!(full[0].name, ".");
        assert!(full[0].qid.is_dir());
        assert_eq!(full[1].name, "hello.txt");
        // The cookie to resume from is the LAST COMPLETE entry's own offset.
        assert_eq!(full[1].offset, 2);

        // With the partial tail included the parse yields the same two entries
        // — a truncated trailing entry is the normal end-of-buffer condition,
        // not corruption.
        let with_tail = parse_dirents(&buf[HDR..n]);
        assert_eq!(with_tail, full);
    }

    #[test_case]
    fn getattr_fields_are_read_in_their_exact_order() {
        // Reading this sequence wrong reports one attribute as another — the
        // failure mode is a plausible number, not an error.
        let mut buf = [0u8; 256];
        let mut e = Enc::start(&mut buf, R_GETATTR, 0);
        e.u64(GETATTR_BASIC); // valid
        e.u8(QTDIR);
        e.u32(7);
        e.u64(0xdead); // qid
        e.u32(0o40755); // mode
        e.u32(501); // uid
        e.u32(20); // gid
        e.u64(3); // nlink
        e.u64(0); // rdev
        e.u64(4096); // size
        e.u64(512); // blksize
        e.u64(8); // blocks
        e.u64(1111);
        e.u64(0); // atime
        e.u64(2222);
        e.u64(0); // mtime
        e.u64(3333);
        e.u64(0); // ctime
        let n = e.finish().unwrap();

        let a = parse_getattr(&buf[..n]).unwrap();
        assert_eq!(a.qid, Qid { typ: QTDIR, version: 7, path: 0xdead });
        assert!(a.qid.is_dir());
        assert_eq!(a.mode, 0o40755);
        assert_eq!((a.uid, a.gid, a.nlink), (501, 20, 3));
        // size must be `size`, not `rdev` or `blksize` on either side of it.
        assert_eq!(a.size, 4096);
        assert_eq!((a.atime, a.mtime, a.ctime), (1111, 2222, 3333));
    }

    #[test_case]
    fn max_walk_elements_is_sixteen() {
        // A deeper path needs several walks; treating this as a tunable and
        // sending 20 names gets the message rejected by the server.
        assert_eq!(MAX_WELEM, 16);
    }
}
