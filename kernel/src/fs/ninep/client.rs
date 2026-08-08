//! **9P2000.L client session** — fid lifetime, path walking and the chunked
//! read/write/readdir loops, over an abstract [`Rpc`] transport.
//!
//! The transport is a trait rather than the virtio device directly, for the
//! reason the whole tree keeps rediscovering: the device lives behind DMA and a
//! virtqueue, so a client welded to it could only ever be tested by booting
//! QEMU. Here the same client runs against an in-memory server in the unit
//! tests, which is where the fid and chunking bugs actually get caught.
//!
//! **Fids are the resource to get right.** A fid is a server-side handle with
//! no timeout: leak one per directory listing and a long session exhausts the
//! server's table, which surfaces much later as an unrelated operation failing.
//! Every helper here clunks what it opened, including on the error paths.

use super::wire::*;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Exchange one framed 9P message for its reply.
pub trait Rpc {
    /// Send `req` and place the reply in `reply`, returning its length.
    fn rpc(&mut self, req: &[u8], reply: &mut [u8]) -> Result<usize, P9Error>;
}

/// The fid the attach lands on. Every walk starts by cloning this one, so it
/// stays live for the session's lifetime and is never consumed.
const ROOT_FID: u32 = 0;

/// Message size to propose. The server may lower it, and everything downstream
/// derives its chunk size from what was actually agreed rather than from this.
pub const MSIZE_WANT: u32 = 64 * 1024;

/// `Twrite` header: `size[4] type[1] tag[2] fid[4] offset[8] count[4]`.
const TWRITE_HDR: usize = HDR + 4 + 8 + 4;
/// `Rread` header: `size[4] type[1] tag[2] count[4]`.
const RREAD_HDR: usize = HDR + 4;

/// A live attachment to one exported tree.
pub struct Session<T: Rpc> {
    t: T,
    msize: u32,
    tag: u16,
    next_fid: u32,
    /// Fids returned by `clunk`, reused before the counter advances so a long
    /// session does not walk the fid space upward forever.
    free_fids: Vec<u32>,
    req: Vec<u8>,
    rep: Vec<u8>,
}

impl<T: Rpc> Session<T> {
    /// Negotiate a version and attach to `aname` as `uname`.
    pub fn attach(t: T, uname: &str, aname: &str) -> Result<Session<T>, P9Error> {
        let mut s = Session {
            t,
            msize: MSIZE_WANT,
            tag: 0,
            next_fid: ROOT_FID + 1,
            free_fids: Vec::new(),
            req: vec![0; MSIZE_WANT as usize],
            rep: vec![0; MSIZE_WANT as usize],
        };

        // Tversion uses the NOTAG tag (0xffff) by definition.
        let n = s.build(T_VERSION, NOTAG, |e| {
            e.u32(MSIZE_WANT);
            e.str(VERSION_L);
        })?;
        let len = s.exchange(n)?;
        let mut d = expect(&s.rep[..len], R_VERSION)?;
        let agreed = d.u32()?;
        let version = d.str()?;
        if version != VERSION_L {
            // A server answering `9P2000` or `unknown` does not understand the
            // .L messages this client sends. Driving it anyway would produce
            // "unexpected reply" at every later step instead of one clear
            // reason here.
            return Err(P9Error::Version);
        }
        // The server may only ever *lower* msize. Honouring a larger value
        // would overrun the buffers already sized for the proposal.
        s.msize = agreed.min(MSIZE_WANT);
        if (s.msize as usize) < TWRITE_HDR + 1 {
            return Err(P9Error::TooLarge);
        }

        let n = s.build(T_ATTACH, 0, |e| {
            e.u32(ROOT_FID);
            e.u32(u32::MAX); // afid = NOFID: no auth
            e.str(uname);
            e.str(aname);
            e.u32(u32::MAX); // n_uname: let the server map us
        })?;
        let len = s.exchange(n)?;
        let mut d = expect(&s.rep[..len], R_ATTACH)?;
        let _root_qid = d.qid()?;
        Ok(s)
    }

    /// The negotiated maximum message size.
    pub fn msize(&self) -> u32 {
        self.msize
    }

    fn next_tag(&mut self) -> u16 {
        // NOTAG is reserved for Tversion, so the ordinary sequence skips it.
        self.tag = self.tag.wrapping_add(1);
        if self.tag == NOTAG {
            self.tag = 0;
        }
        self.tag
    }

    fn alloc_fid(&mut self) -> u32 {
        if let Some(f) = self.free_fids.pop() {
            return f;
        }
        let f = self.next_fid;
        self.next_fid = self.next_fid.wrapping_add(1);
        f
    }

    /// Frame a message into the request buffer.
    fn build(&mut self, typ: u8, tag: u16, body: impl FnOnce(&mut Enc)) -> Result<usize, P9Error> {
        let cap = self.msize as usize;
        let mut e = Enc::start(&mut self.req[..cap], typ, tag);
        body(&mut e);
        e.finish().ok_or(P9Error::TooLarge)
    }

    fn exchange(&mut self, n: usize) -> Result<usize, P9Error> {
        let cap = self.msize as usize;
        // Split the borrow: the request and reply buffers are separate fields,
        // so the transport can read one while writing the other.
        let (req, rep) = (&self.req[..n], &mut self.rep[..cap]);
        // SAFETY-free: both are plain slices of owned Vecs; the raw pointer
        // dance is only to satisfy the borrow checker across the &mut self.
        let req_ptr = req.as_ptr();
        let req_len = req.len();
        let rep_ptr = rep.as_mut_ptr();
        let rep_len = rep.len();
        // SAFETY: `req` and `rep` are disjoint fields of `self`, both live for
        // this call, and neither is reallocated by `rpc`.
        let (req, rep) = unsafe {
            (
                core::slice::from_raw_parts(req_ptr, req_len),
                core::slice::from_raw_parts_mut(rep_ptr, rep_len),
            )
        };
        self.t.rpc(req, rep)
    }

    /// Release a fid on the server and recycle it locally.
    pub fn clunk(&mut self, fid: u32) {
        if fid == ROOT_FID {
            return;
        }
        let tag = self.next_tag();
        if let Ok(n) = self.build(T_CLUNK, tag, |e| e.u32(fid)) {
            // A failed clunk is not actionable — the fid is gone either way as
            // far as this client is concerned — but it must not be recycled if
            // the server still holds it.
            if let Ok(len) = self.exchange(n) {
                if expect(&self.rep[..len], R_CLUNK).is_ok() {
                    self.free_fids.push(fid);
                    return;
                }
            }
        }
    }

    /// Walk `path` from the root into a fresh fid.
    ///
    /// A path deeper than [`MAX_WELEM`] takes several `Twalk`s — that limit is
    /// in the protocol, not a tunable.
    pub fn walk(&mut self, path: &str) -> Result<u32, P9Error> {
        let parts = split_path(path);
        let fid = self.alloc_fid();
        // The first walk clones ROOT_FID into `fid`; with no names that is
        // exactly how 9P duplicates a fid, which is why the root survives.
        let mut from = ROOT_FID;
        let mut done = 0usize;
        loop {
            let chunk: Vec<&str> = parts[done..].iter().take(MAX_WELEM).copied().collect();
            let tag = self.next_tag();
            let n = self.build(T_WALK, tag, |e| {
                e.u32(from);
                e.u32(fid);
                e.u16(chunk.len() as u16);
                for c in &chunk {
                    e.str(c);
                }
            })?;
            let len = match self.exchange(n) {
                Ok(l) => l,
                Err(err) => {
                    self.free_fids.push(fid);
                    return Err(err);
                }
            };
            let nwqid = match expect(&self.rep[..len], R_WALK).and_then(|mut d| d.u16()) {
                Ok(v) => v,
                Err(err) => {
                    // Nothing was cloned, so there is no fid to clunk — but the
                    // local id is free to reuse.
                    self.free_fids.push(fid);
                    return Err(err);
                }
            };
            // A SHORT WALK IS A SUCCESS REPLY. Fewer qids than names means the
            // walk stopped at a component that does not exist; a client that
            // checks only the message type ends up holding the parent and
            // reporting it as the child.
            if (nwqid as usize) < chunk.len() {
                if done > 0 || nwqid > 0 {
                    // Something was cloned into `fid`; release it.
                    self.clunk(fid);
                } else {
                    self.free_fids.push(fid);
                }
                return Err(P9Error::Server(ENOENT));
            }
            done += chunk.len();
            if done >= parts.len() {
                return Ok(fid);
            }
            // Subsequent chunks continue from the fid we are building.
            from = fid;
        }
    }

    /// `stat` a path.
    pub fn getattr(&mut self, path: &str) -> Result<Attr, P9Error> {
        let fid = self.walk(path)?;
        let tag = self.next_tag();
        let r = (|| {
            let n = self.build(T_GETATTR, tag, |e| {
                e.u32(fid);
                e.u64(GETATTR_BASIC);
            })?;
            let len = self.exchange(n)?;
            parse_getattr(&self.rep[..len])
        })();
        self.clunk(fid);
        r
    }

    /// Whether a path exists (and whether it is a directory).
    pub fn exists(&mut self, path: &str) -> Option<bool> {
        self.getattr(path).ok().map(|a| a.qid.is_dir())
    }

    fn lopen(&mut self, fid: u32, flags: u32) -> Result<(), P9Error> {
        let tag = self.next_tag();
        let n = self.build(T_LOPEN, tag, |e| {
            e.u32(fid);
            e.u32(flags);
        })?;
        let len = self.exchange(n)?;
        expect(&self.rep[..len], R_LOPEN)?;
        Ok(())
    }

    /// Read a whole file.
    ///
    /// `Rread` returning zero bytes is end-of-file; looping on a byte count
    /// instead would never terminate on a file the server reports as larger
    /// than it can deliver.
    pub fn read_file(&mut self, path: &str) -> Result<Vec<u8>, P9Error> {
        let fid = self.walk(path)?;
        let r = (|| {
            self.lopen(fid, O_RDONLY)?;
            let chunk = self.msize as usize - RREAD_HDR;
            let mut out = Vec::new();
            let mut off = 0u64;
            loop {
                let tag = self.next_tag();
                let n = self.build(T_READ, tag, |e| {
                    e.u32(fid);
                    e.u64(off);
                    e.u32(chunk as u32);
                })?;
                let len = self.exchange(n)?;
                let mut d = expect(&self.rep[..len], R_READ)?;
                let count = d.u32()? as usize;
                if count == 0 {
                    break;
                }
                let data = d.rest();
                if data.len() < count {
                    return Err(P9Error::Malformed);
                }
                out.extend_from_slice(&data[..count]);
                off += count as u64;
            }
            Ok(out)
        })();
        self.clunk(fid);
        r
    }

    /// Create (or truncate) a file and write `data` to it.
    pub fn write_file(&mut self, path: &str, data: &[u8]) -> Result<(), P9Error> {
        let (dir, name) = split_parent(path);
        if name.is_empty() {
            return Err(P9Error::Server(EINVAL));
        }
        let dfid = self.walk(dir)?;
        let r = (|| {
            // Tlcreate opens the new file *on the directory fid*, which then
            // refers to the file — the directory fid is consumed, not kept.
            let tag = self.next_tag();
            let n = self.build(T_LCREATE, tag, |e| {
                e.u32(dfid);
                e.str(name);
                e.u32(O_WRONLY | O_CREAT | O_TRUNC);
                e.u32(0o644);
                e.u32(u32::MAX); // gid: let the server decide
            })?;
            let len = self.exchange(n)?;
            expect(&self.rep[..len], R_LCREATE)?;
            self.write_all(dfid, data)
        })();
        self.clunk(dfid);
        r
    }

    fn write_all(&mut self, fid: u32, data: &[u8]) -> Result<(), P9Error> {
        let chunk = self.msize as usize - TWRITE_HDR;
        let mut off = 0usize;
        while off < data.len() {
            let take = (data.len() - off).min(chunk);
            let tag = self.next_tag();
            let n = self.build(T_WRITE, tag, |e| {
                e.u32(fid);
                e.u64(off as u64);
                e.u32(take as u32);
                e.raw(&data[off..off + take]);
            })?;
            let len = self.exchange(n)?;
            let mut d = expect(&self.rep[..len], R_WRITE)?;
            let wrote = d.u32()? as usize;
            // A short write is legal and must be honoured by advancing only
            // what was accepted; assuming `take` silently drops bytes.
            if wrote == 0 {
                return Err(P9Error::Transport);
            }
            off += wrote.min(take);
        }
        Ok(())
    }

    /// List a directory.
    ///
    /// Resumption uses the **last entry's own offset cookie**, not a running
    /// byte count: the cookie is opaque and need not be the byte position.
    pub fn readdir(&mut self, path: &str) -> Result<Vec<DirEntry9>, P9Error> {
        let fid = self.walk(path)?;
        let r = (|| {
            self.lopen(fid, O_RDONLY)?;
            let chunk = self.msize as usize - RREAD_HDR;
            let mut out: Vec<DirEntry9> = Vec::new();
            let mut cookie = 0u64;
            loop {
                let tag = self.next_tag();
                let n = self.build(T_READDIR, tag, |e| {
                    e.u32(fid);
                    e.u64(cookie);
                    e.u32(chunk as u32);
                })?;
                let len = self.exchange(n)?;
                let mut d = expect(&self.rep[..len], R_READDIR)?;
                let count = d.u32()? as usize;
                if count == 0 {
                    break;
                }
                let data = d.rest();
                if data.len() < count {
                    return Err(P9Error::Malformed);
                }
                let batch = parse_dirents(&data[..count]);
                let Some(last) = batch.last() else {
                    // Bytes arrived but no entry parsed out of them: the server
                    // is producing something this client cannot advance past,
                    // and looping would hang.
                    break;
                };
                cookie = last.offset;
                out.extend(batch);
            }
            Ok(out)
        })();
        self.clunk(fid);
        r
    }

    /// Create a directory.
    pub fn mkdir(&mut self, path: &str) -> Result<(), P9Error> {
        let (dir, name) = split_parent(path);
        if name.is_empty() {
            return Err(P9Error::Server(EINVAL));
        }
        let dfid = self.walk(dir)?;
        let tag = self.next_tag();
        let r = (|| {
            let n = self.build(T_MKDIR, tag, |e| {
                e.u32(dfid);
                e.str(name);
                e.u32(0o755);
                e.u32(u32::MAX);
            })?;
            let len = self.exchange(n)?;
            expect(&self.rep[..len], R_MKDIR)?;
            Ok(())
        })();
        self.clunk(dfid);
        r
    }

    /// Remove a file or an (empty) directory.
    pub fn unlink(&mut self, path: &str, is_dir: bool) -> Result<(), P9Error> {
        let (dir, name) = split_parent(path);
        if name.is_empty() {
            return Err(P9Error::Server(EINVAL));
        }
        let dfid = self.walk(dir)?;
        let tag = self.next_tag();
        let r = (|| {
            let n = self.build(T_UNLINKAT, tag, |e| {
                e.u32(dfid);
                e.str(name);
                e.u32(if is_dir { AT_REMOVEDIR } else { 0 });
            })?;
            let len = self.exchange(n)?;
            expect(&self.rep[..len], R_UNLINKAT)?;
            Ok(())
        })();
        self.clunk(dfid);
        r
    }
}

/// `NOTAG` — the tag `Tversion` must use.
pub const NOTAG: u16 = 0xffff;
/// Linux errno values this client raises itself.
pub const ENOENT: u32 = 2;
pub const EINVAL: u32 = 22;

/// Split a path into walk components, dropping empties and `.`.
pub fn split_path(path: &str) -> Vec<&str> {
    path.split('/').filter(|c| !c.is_empty() && *c != ".").collect()
}

/// Split into `(parent, name)`. The parent of a top-level name is the root.
pub fn split_parent(path: &str) -> (&str, &str) {
    let trimmed = path.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(i) => (&trimmed[..i], &trimmed[i + 1..]),
        None => ("", trimmed),
    }
}

/// Render a 9P error for a human. A bare errno is not a diagnosis.
pub fn describe(e: P9Error) -> String {
    use alloc::format;
    match e {
        P9Error::Server(2) => "no such file or directory".into(),
        P9Error::Server(13) => "permission denied".into(),
        P9Error::Server(17) => "already exists".into(),
        P9Error::Server(20) => "not a directory".into(),
        P9Error::Server(21) => "is a directory".into(),
        P9Error::Server(22) => "invalid argument".into(),
        P9Error::Server(28) => "no space left on the host".into(),
        P9Error::Server(39) => "directory not empty".into(),
        P9Error::Server(n) => format!("host error {n}"),
        P9Error::Malformed => "malformed 9P reply".into(),
        P9Error::Unexpected(t) => format!("unexpected 9P reply type {t}"),
        P9Error::TooLarge => "message exceeds the negotiated size".into(),
        P9Error::Transport => "9P transport failure".into(),
        P9Error::Version => "host does not speak 9P2000.L".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::string::ToString;

    // ---------------------------------------------------------------
    // An in-memory 9P2000.L server, so the client is exercised end to end
    // without a device. This is where the fid, chunking and short-walk bugs
    // actually surface — none of them are reachable from the codec alone.
    // ---------------------------------------------------------------

    #[derive(Clone)]
    struct Node {
        is_dir: bool,
        data: Vec<u8>,
        children: Vec<String>,
    }

    struct FakeServer {
        /// Absolute path → node. The root is "".
        nodes: BTreeMap<String, Node>,
        /// fid → absolute path.
        fids: BTreeMap<u32, String>,
        msize: u32,
        /// Largest number of fids live at once — the leak detector.
        peak_fids: usize,
        /// Force the server to answer reads in small pieces, so the client's
        /// chunking loop is exercised rather than one-shotting every file.
        read_cap: usize,
    }

    impl FakeServer {
        fn new(msize: u32) -> FakeServer {
            let mut nodes = BTreeMap::new();
            nodes.insert(
                "".to_string(),
                Node { is_dir: true, data: Vec::new(), children: Vec::new() },
            );
            FakeServer {
                nodes,
                fids: BTreeMap::new(),
                msize,
                peak_fids: 0,
                read_cap: usize::MAX,
            }
        }

        fn add(&mut self, path: &str, is_dir: bool, data: &[u8]) {
            let (parent, name) = split_parent(path);
            self.nodes.entry(parent.to_string()).and_modify(|n| {
                if !n.children.contains(&name.to_string()) {
                    n.children.push(name.to_string());
                }
            });
            self.nodes.insert(
                path.to_string(),
                Node { is_dir, data: data.to_vec(), children: Vec::new() },
            );
        }

        fn err(rep: &mut [u8], tag: u16, code: u32) -> usize {
            let mut e = Enc::start(rep, R_LERROR, tag);
            e.u32(code);
            e.finish().unwrap()
        }

        fn handle(&mut self, req: &[u8], rep: &mut [u8]) -> usize {
            let (typ, tag, mut d) = Dec::header(req).expect("client sent a malformed message");
            match typ {
                T_VERSION => {
                    let want = d.u32().unwrap();
                    let _v = d.str().unwrap();
                    self.msize = self.msize.min(want);
                    let mut e = Enc::start(rep, R_VERSION, tag);
                    e.u32(self.msize);
                    e.str(VERSION_L);
                    e.finish().unwrap()
                }
                T_ATTACH => {
                    let fid = d.u32().unwrap();
                    self.fids.insert(fid, String::new());
                    let mut e = Enc::start(rep, R_ATTACH, tag);
                    e.u8(QTDIR);
                    e.u32(0);
                    e.u64(0);
                    e.finish().unwrap()
                }
                T_WALK => {
                    let fid = d.u32().unwrap();
                    let newfid = d.u32().unwrap();
                    let n = d.u16().unwrap() as usize;
                    assert!(n <= MAX_WELEM, "client sent {n} walk elements, max is {MAX_WELEM}");
                    let Some(base) = self.fids.get(&fid).cloned() else {
                        return Self::err(rep, tag, ENOENT);
                    };
                    let mut cur = base;
                    let mut qids = Vec::new();
                    for _ in 0..n {
                        let name = d.str().unwrap();
                        let next = if cur.is_empty() {
                            name.clone()
                        } else {
                            alloc::format!("{cur}/{name}")
                        };
                        match self.nodes.get(&next) {
                            Some(node) => {
                                qids.push(if node.is_dir { QTDIR } else { 0 });
                                cur = next;
                            }
                            // Stop early — a SHORT Rwalk, not an error reply.
                            None => break,
                        }
                    }
                    if qids.len() == n {
                        self.fids.insert(newfid, cur);
                        self.peak_fids = self.peak_fids.max(self.fids.len());
                    } else if !qids.is_empty() {
                        // Partial walks still clone as far as they got.
                        self.fids.insert(newfid, cur);
                        self.peak_fids = self.peak_fids.max(self.fids.len());
                    }
                    let mut e = Enc::start(rep, R_WALK, tag);
                    e.u16(qids.len() as u16);
                    for q in qids {
                        e.u8(q);
                        e.u32(0);
                        e.u64(0);
                    }
                    e.finish().unwrap()
                }
                T_LOPEN => {
                    let fid = d.u32().unwrap();
                    if !self.fids.contains_key(&fid) {
                        return Self::err(rep, tag, ENOENT);
                    }
                    let mut e = Enc::start(rep, R_LOPEN, tag);
                    e.u8(0);
                    e.u32(0);
                    e.u64(0);
                    e.u32(0);
                    e.finish().unwrap()
                }
                T_LCREATE => {
                    let fid = d.u32().unwrap();
                    let name = d.str().unwrap();
                    let Some(dir) = self.fids.get(&fid).cloned() else {
                        return Self::err(rep, tag, ENOENT);
                    };
                    let path =
                        if dir.is_empty() { name.clone() } else { alloc::format!("{dir}/{name}") };
                    self.add(&path, false, &[]);
                    // The directory fid now refers to the new file.
                    self.fids.insert(fid, path);
                    let mut e = Enc::start(rep, R_LCREATE, tag);
                    e.u8(0);
                    e.u32(0);
                    e.u64(0);
                    e.u32(0);
                    e.finish().unwrap()
                }
                T_READ => {
                    let fid = d.u32().unwrap();
                    let off = d.u64().unwrap() as usize;
                    let count = d.u32().unwrap() as usize;
                    let Some(p) = self.fids.get(&fid) else {
                        return Self::err(rep, tag, ENOENT);
                    };
                    let data = &self.nodes[p].data;
                    let start = off.min(data.len());
                    let take = count.min(self.read_cap).min(data.len() - start);
                    let mut e = Enc::start(rep, R_READ, tag);
                    e.u32(take as u32);
                    e.raw(&data[start..start + take]);
                    e.finish().unwrap()
                }
                T_WRITE => {
                    let fid = d.u32().unwrap();
                    let off = d.u64().unwrap() as usize;
                    let count = d.u32().unwrap() as usize;
                    let payload = &d.rest()[..count];
                    let p = self.fids.get(&fid).cloned().unwrap();
                    let node = self.nodes.get_mut(&p).unwrap();
                    if node.data.len() < off + count {
                        node.data.resize(off + count, 0);
                    }
                    node.data[off..off + count].copy_from_slice(payload);
                    let mut e = Enc::start(rep, R_WRITE, tag);
                    e.u32(count as u32);
                    e.finish().unwrap()
                }
                T_READDIR => {
                    let fid = d.u32().unwrap();
                    let cookie = d.u64().unwrap();
                    let count = d.u32().unwrap() as usize;
                    let Some(p) = self.fids.get(&fid).cloned() else {
                        return Self::err(rep, tag, ENOENT);
                    };
                    let kids = self.nodes[&p].children.clone();
                    let mut body = Vec::new();
                    // The cookie is 1-based so that 0 means "from the start".
                    for (i, name) in kids.iter().enumerate().skip(cookie as usize) {
                        let full =
                            if p.is_empty() { name.clone() } else { alloc::format!("{p}/{name}") };
                        let is_dir = self.nodes[&full].is_dir;
                        let mut one = [0u8; 512];
                        let mut e = Enc::start(&mut one, 0, 0);
                        e.u8(if is_dir { QTDIR } else { 0 });
                        e.u32(0);
                        e.u64(0);
                        e.u64(i as u64 + 1);
                        e.u8(if is_dir { 4 } else { 8 });
                        e.str(name);
                        let n = e.finish().unwrap();
                        let ent = &one[HDR..n];
                        // Stop before overflowing what the client asked for —
                        // this is what forces a second Treaddir round.
                        if body.len() + ent.len() > count.min(self.read_cap) {
                            break;
                        }
                        body.extend_from_slice(ent);
                    }
                    let mut e = Enc::start(rep, R_READDIR, tag);
                    e.u32(body.len() as u32);
                    e.raw(&body);
                    e.finish().unwrap()
                }
                T_GETATTR => {
                    let fid = d.u32().unwrap();
                    let Some(p) = self.fids.get(&fid) else {
                        return Self::err(rep, tag, ENOENT);
                    };
                    let node = &self.nodes[p];
                    let mut e = Enc::start(rep, R_GETATTR, tag);
                    e.u64(GETATTR_BASIC);
                    e.u8(if node.is_dir { QTDIR } else { 0 });
                    e.u32(0);
                    e.u64(0);
                    e.u32(if node.is_dir { 0o40755 } else { 0o100644 });
                    e.u32(501);
                    e.u32(20);
                    e.u64(1);
                    e.u64(0);
                    e.u64(node.data.len() as u64);
                    e.u64(4096);
                    e.u64(0);
                    for _ in 0..6 {
                        e.u64(7);
                    }
                    e.finish().unwrap()
                }
                T_MKDIR => {
                    let fid = d.u32().unwrap();
                    let name = d.str().unwrap();
                    let dir = self.fids.get(&fid).cloned().unwrap();
                    let path =
                        if dir.is_empty() { name.clone() } else { alloc::format!("{dir}/{name}") };
                    self.add(&path, true, &[]);
                    let mut e = Enc::start(rep, R_MKDIR, tag);
                    e.u8(QTDIR);
                    e.u32(0);
                    e.u64(0);
                    e.finish().unwrap()
                }
                T_UNLINKAT => {
                    let fid = d.u32().unwrap();
                    let name = d.str().unwrap();
                    let dir = self.fids.get(&fid).cloned().unwrap();
                    let path =
                        if dir.is_empty() { name.clone() } else { alloc::format!("{dir}/{name}") };
                    self.nodes.remove(&path);
                    if let Some(p) = self.nodes.get_mut(&dir) {
                        p.children.retain(|c| c != &name);
                    }
                    Enc::start(rep, R_UNLINKAT, tag).finish().unwrap()
                }
                T_CLUNK => {
                    let fid = d.u32().unwrap();
                    self.fids.remove(&fid);
                    Enc::start(rep, R_CLUNK, tag).finish().unwrap()
                }
                other => panic!("fake server got unhandled message type {other}"),
            }
        }
    }

    /// The `Rpc` end: hands each request to the server and takes the reply.
    struct Loopback {
        srv: FakeServer,
    }

    impl Rpc for Loopback {
        fn rpc(&mut self, req: &[u8], reply: &mut [u8]) -> Result<usize, P9Error> {
            Ok(self.srv.handle(req, reply))
        }
    }

    fn session(build: impl FnOnce(&mut FakeServer)) -> Session<Loopback> {
        let mut srv = FakeServer::new(MSIZE_WANT);
        build(&mut srv);
        Session::attach(Loopback { srv }, "chitti", "").unwrap()
    }

    #[test_case]
    fn attach_negotiates_and_reads_a_file_back() {
        let mut s = session(|srv| {
            srv.add("hello.txt", false, b"hi from the host");
        });
        assert_eq!(s.msize(), MSIZE_WANT);
        assert_eq!(s.read_file("/hello.txt").unwrap(), b"hi from the host");
        // A leading slash, a bare name and a redundant `.` are the same path.
        assert_eq!(s.read_file("hello.txt").unwrap(), b"hi from the host");
        assert_eq!(s.read_file("./hello.txt").unwrap(), b"hi from the host");
    }

    #[test_case]
    fn a_short_walk_is_a_missing_file_not_the_parent() {
        // The trap: Rwalk with fewer qids than names is a SUCCESS reply. A
        // client that checks only the message type would hold the parent's fid
        // and report the parent's contents under the child's name.
        let mut s = session(|srv| {
            srv.add("dir", true, &[]);
            srv.add("dir/real.txt", false, b"x");
        });
        assert_eq!(s.getattr("/dir/missing.txt").unwrap_err(), P9Error::Server(ENOENT));
        assert_eq!(s.read_file("/dir/missing.txt").unwrap_err(), P9Error::Server(ENOENT));
        // The real one still resolves, so the failure above was not blanket.
        assert_eq!(s.read_file("/dir/real.txt").unwrap(), b"x");
    }

    #[test_case]
    fn a_large_file_round_trips_through_many_chunks() {
        // Force the server to dribble bytes out so the read loop runs many
        // times: a one-shot read would pass whatever the chunking did.
        let big: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
        let mut s = session(|srv| {
            srv.read_cap = 1000;
            srv.add("big.bin", false, &[]);
        });
        s.write_file("/big.bin", &big).unwrap();
        let back = s.read_file("/big.bin").unwrap();
        assert_eq!(back.len(), big.len());
        assert_eq!(back, big);
    }

    #[test_case]
    fn readdir_resumes_from_the_last_entry_cookie() {
        // With a small cap the listing needs several Treaddir rounds; resuming
        // from a byte count instead of the entry's own cookie would repeat or
        // skip entries.
        let mut s = session(|srv| {
            srv.read_cap = 64;
            for i in 0..25 {
                srv.add(&alloc::format!("f{i:02}"), false, b"x");
            }
            srv.add("sub", true, &[]);
        });
        let ents = s.readdir("/").unwrap();
        assert_eq!(ents.len(), 26, "every entry listed exactly once");
        let names: Vec<&str> = ents.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"f00") && names.contains(&"f24") && names.contains(&"sub"));
        // No duplicates — the classic symptom of resuming from the wrong cookie.
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        assert!(ents.iter().find(|e| e.name == "sub").unwrap().qid.is_dir());
    }

    #[test_case]
    fn a_path_deeper_than_one_walk_message_still_resolves() {
        // MAX_WELEM is 16, so a 20-deep path needs two Twalks. The fake server
        // asserts the client never sends more than 16 names in one message.
        let mut s = session(|srv| {
            let mut p = String::new();
            for i in 0..20 {
                if i > 0 {
                    p.push('/');
                }
                p.push_str(&alloc::format!("d{i}"));
                srv.add(&p, true, &[]);
            }
            srv.add(&alloc::format!("{p}/deep.txt"), false, b"found me");
        });
        let path: Vec<String> = (0..20).map(|i| alloc::format!("d{i}")).collect();
        let deep = alloc::format!("/{}/deep.txt", path.join("/"));
        assert_eq!(s.read_file(&deep).unwrap(), b"found me");
    }

    #[test_case]
    fn fids_do_not_leak_across_many_operations() {
        // A fid is a server-side handle with no timeout. Leaking one per
        // listing exhausts the server's table and surfaces much later as an
        // unrelated operation failing, so every helper must clunk what it
        // opened — including on its error paths.
        let mut s = session(|srv| {
            srv.add("a", true, &[]);
            srv.add("a/f.txt", false, b"data");
        });
        for _ in 0..50 {
            let _ = s.readdir("/a");
            let _ = s.read_file("/a/f.txt");
            let _ = s.getattr("/a");
            let _ = s.read_file("/a/nope.txt"); // error path must clunk too
            let _ = s.getattr("/a/nope.txt");
        }
        // Only the attach fid should ever be live between calls; the peak
        // allows for the one in flight.
        assert!(
            s.t.srv.peak_fids <= 2,
            "fid leak: peak {} live fids over 250 operations",
            s.t.srv.peak_fids
        );
    }

    #[test_case]
    fn mkdir_write_list_and_unlink_round_trip() {
        let mut s = session(|_| {});
        s.mkdir("/notes").unwrap();
        s.write_file("/notes/a.txt", b"alpha").unwrap();
        s.write_file("/notes/b.txt", b"beta").unwrap();
        let mut names: Vec<String> =
            s.readdir("/notes").unwrap().into_iter().map(|e| e.name).collect();
        names.sort();
        assert_eq!(names, alloc::vec!["a.txt".to_string(), "b.txt".to_string()]);
        assert_eq!(s.read_file("/notes/b.txt").unwrap(), b"beta");
        assert_eq!(s.getattr("/notes/a.txt").unwrap().size, 5);

        s.unlink("/notes/a.txt", false).unwrap();
        assert_eq!(s.readdir("/notes").unwrap().len(), 1);
        assert_eq!(s.exists("/notes/a.txt"), None);
        assert_eq!(s.exists("/notes"), Some(true));
        assert_eq!(s.exists("/notes/b.txt"), Some(false));
    }

    #[test_case]
    fn the_server_may_only_lower_msize() {
        // Honouring a larger msize than proposed would overrun buffers already
        // sized for the proposal.
        let mut srv = FakeServer::new(8192);
        srv.add("x", false, b"y");
        let s = Session::attach(Loopback { srv }, "chitti", "").unwrap();
        assert_eq!(s.msize(), 8192);

        struct Greedy;
        impl Rpc for Greedy {
            fn rpc(&mut self, req: &[u8], reply: &mut [u8]) -> Result<usize, P9Error> {
                let (t, tag, _d) = Dec::header(req).unwrap();
                Ok(match t {
                    T_VERSION => {
                        let mut e = Enc::start(reply, R_VERSION, tag);
                        e.u32(u32::MAX); // claims a bigger msize than proposed
                        e.str(VERSION_L);
                        e.finish().unwrap()
                    }
                    _ => {
                        let mut e = Enc::start(reply, R_ATTACH, tag);
                        e.u8(QTDIR);
                        e.u32(0);
                        e.u64(0);
                        e.finish().unwrap()
                    }
                })
            }
        }
        let s = Session::attach(Greedy, "chitti", "").unwrap();
        assert_eq!(s.msize(), MSIZE_WANT);
    }

    #[test_case]
    fn a_server_that_does_not_speak_l_is_refused_with_that_reason() {
        // Driving a 9P2000 server with .L messages gives "unexpected reply" at
        // every later step instead of one clear reason here.
        struct Legacy;
        impl Rpc for Legacy {
            fn rpc(&mut self, req: &[u8], reply: &mut [u8]) -> Result<usize, P9Error> {
                let (_t, tag, _d) = Dec::header(req).unwrap();
                let mut e = Enc::start(reply, R_VERSION, tag);
                e.u32(8192);
                e.str("9P2000");
                Ok(e.finish().unwrap())
            }
        }
        assert_eq!(Session::attach(Legacy, "chitti", "").err(), Some(P9Error::Version));
        assert_eq!(describe(P9Error::Version), "host does not speak 9P2000.L");
    }

    #[test_case]
    fn path_splitting_matches_the_shell_view() {
        assert_eq!(split_path("/a/b/c"), alloc::vec!["a", "b", "c"]);
        assert_eq!(split_path("a//b/./c/"), alloc::vec!["a", "b", "c"]);
        assert!(split_path("/").is_empty());
        assert_eq!(split_parent("/a/b/c.txt"), ("/a/b", "c.txt"));
        assert_eq!(split_parent("top.txt"), ("", "top.txt"));
        assert_eq!(split_parent("/only"), ("", "only"));
        // A trailing slash names the directory, not an empty child.
        assert_eq!(split_parent("/a/b/"), ("/a", "b"));
    }
}
