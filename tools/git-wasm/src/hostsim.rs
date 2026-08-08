//! **Host simulator** — the `chitti::host_*` imports, implemented for a native
//! build so the git logic can be tested off the OS.
//!
//! The kernel's own primitives are *mounted*, not reimplemented (the
//! `h264diff`/`pdf-wasm` pattern): SHA-1, zlib inflate and the stored-block
//! deflate here are byte-for-byte the ones `wasm_rt.rs` calls, so a packfile that
//! walks in a test walks on the machine. The FS is a flat map with directories
//! derived from key prefixes, which is the same shape `synapse::fs` presents.
//!
//! **The one behaviour these stubs must copy exactly is the buffer contract**:
//! write at most `cap` bytes and report the result's *full* length. Faking that
//! (answering `min(len, cap)`, as the kernel used to) would make the truncation
//! these tests exist to catch invisible here — a 182 KiB packfile came back as a
//! plausible complete 64 KiB one, and the clone died in the decompressor.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

// The kernel's real implementations. A non-`cfg(test)` library build compiles
// these without their `#[test_case]` modules, which is why the tests driving this
// live in `tests/` rather than inside the crate.
#[path = "../../../kernel/src/image/inflate.rs"]
pub mod inflate;
#[path = "../../../kernel/src/image/deflate.rs"]
pub mod deflate;
#[path = "../../../kernel/src/net/sha1.rs"]
pub mod sha1;

/// The simulated machine: a flat store plus scripted HTTP replies.
#[derive(Default)]
pub struct Sim {
    /// Absolute store path → contents.
    pub files: BTreeMap<String, Vec<u8>>,
    /// `(url substring, status, body)`, matched in order.
    pub http: Vec<(String, u16, Vec<u8>)>,
    /// Every request the guest made, for asserting how many downloads a clone took.
    pub requests: Vec<String>,
    pub home: String,
    pub user_home: String,
}

static mut SIM: Option<Sim> = None;
/// The simulated machine is process-wide (it stands in for imports, which are),
/// and `cargo test` runs tests on threads — so a test holds this for its whole
/// body. Without it two clones interleave into one store and fail in ways that
/// have nothing to do with git.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exclusive use of the simulated machine, released when the test ends.
pub struct Guard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

/// Install a fresh simulated machine and take it for this test.
pub fn reset(home: &str, user_home: &str) -> Guard {
    // A panicking test poisons the lock; the state is replaced wholesale on the
    // next line anyway, so recovering keeps one failure from cascading into
    // every later test reporting a lock error instead of its own result.
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe {
        SIM = Some(Sim {
            home: home.to_string(),
            user_home: user_home.to_string(),
            ..Default::default()
        });
    }
    crate::reset_state();
    Guard(g)
}

#[allow(static_mut_refs)]
pub fn sim() -> &'static mut Sim {
    unsafe { SIM.get_or_insert_with(Sim::default) }
}

impl Sim {
    /// Script a reply for any URL containing `pat`.
    pub fn reply(&mut self, pat: &str, status: u16, body: Vec<u8>) {
        self.http.push((pat.to_string(), status, body));
    }

    /// Children of `dir` derived from the key prefixes: `(name, is_dir, size)`.
    fn children(&self, dir: &str) -> Vec<(String, bool, u64)> {
        let prefix = if dir.ends_with('/') { dir.to_string() } else { alloc::format!("{dir}/") };
        let mut out: Vec<(String, bool, u64)> = Vec::new();
        for (k, v) in &self.files {
            let Some(rest) = k.strip_prefix(&prefix) else { continue };
            let (name, is_dir) = match rest.split_once('/') {
                Some((head, _)) => (head.to_string(), true),
                None => (rest.to_string(), false),
            };
            if out.iter().any(|(n, _, _)| *n == name) {
                continue;
            }
            let size = if is_dir { 0 } else { v.len() as u64 };
            out.push((name, is_dir, size));
        }
        out
    }
}

// --- the imports ------------------------------------------------------------

/// Copy at most `cap` bytes to `out` and answer `data.len()` — the kernel's
/// `fill_guest` contract. See the module note: getting this wrong here would hide
/// the exact bug these tests cover.
unsafe fn fill(out: *mut u8, cap: i32, data: &[u8]) -> i32 {
    let n = data.len().min(cap.max(0) as usize);
    if n > 0 {
        core::ptr::copy_nonoverlapping(data.as_ptr(), out, n);
    }
    data.len() as i32
}

unsafe fn guest_str(p: *const u8, len: i32) -> String {
    String::from_utf8_lossy(core::slice::from_raw_parts(p, len.max(0) as usize)).into_owned()
}

pub unsafe fn host_fs_read(path: *const u8, path_len: i32, out: *mut u8, out_cap: i32) -> i32 {
    let p = guest_str(path, path_len);
    match sim().files.get(&p) {
        Some(v) => fill(out, out_cap, &v.clone()),
        None => -2,
    }
}

pub unsafe fn host_fs_write(path: *const u8, path_len: i32, data: *const u8, data_len: i32) -> i32 {
    let p = guest_str(path, path_len);
    let d = core::slice::from_raw_parts(data, data_len.max(0) as usize).to_vec();
    sim().files.insert(p, d);
    0
}

pub unsafe fn host_fs_list(path: *const u8, path_len: i32, out: *mut u8, out_cap: i32) -> i32 {
    let p = guest_str(path, path_len);
    let mut s = String::new();
    for (name, is_dir, size) in sim().children(&p) {
        s.push_str(&alloc::format!("{name}\t{}\t{size}\n", if is_dir { 'd' } else { 'f' }));
    }
    fill(out, out_cap, s.as_bytes())
}

pub unsafe fn host_fs_exists(path: *const u8, path_len: i32) -> i32 {
    let p = guest_str(path, path_len);
    let s = sim();
    let dir = alloc::format!("{p}/");
    i32::from(s.files.contains_key(&p) || s.files.keys().any(|k| k.starts_with(&dir)))
}

pub unsafe fn host_home(out: *mut u8, out_cap: i32) -> i32 {
    fill(out, out_cap, sim().home.clone().as_bytes())
}

pub unsafe fn host_user_home(out: *mut u8, out_cap: i32) -> i32 {
    fill(out, out_cap, sim().user_home.clone().as_bytes())
}

pub unsafe fn host_now_unix() -> i64 {
    1_600_000_000
}

pub unsafe fn host_sha1(src: *const u8, src_len: i32, out: *mut u8) -> i32 {
    let d = sha1::sha1(core::slice::from_raw_parts(src, src_len.max(0) as usize));
    core::ptr::copy_nonoverlapping(d.as_ptr(), out, d.len());
    0
}

pub unsafe fn host_inflate(src: *const u8, src_len: i32, out: *mut u8, out_cap: i32) -> i64 {
    let s = core::slice::from_raw_parts(src, src_len.max(0) as usize);
    let Ok((dec, consumed)) = inflate::zlib_decompress_len(s) else { return -2 };
    ((consumed as i64) << 32) | fill(out, out_cap, &dec) as i64
}

pub unsafe fn host_deflate(src: *const u8, src_len: i32, out: *mut u8, out_cap: i32) -> i32 {
    let s = core::slice::from_raw_parts(src, src_len.max(0) as usize);
    fill(out, out_cap, &deflate::zlib_stored(s))
}

pub unsafe fn host_http(req: *const u8, req_len: i32, out: *mut u8, out_cap: i32) -> i64 {
    let raw = guest_str(req, req_len);
    // The guest builds `{"m":…,"u":…,…}` by hand, so pulling the URL back out the
    // same way is enough — this is a stub, not a JSON parser.
    let url = raw
        .split("\"u\":\"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .unwrap_or("")
        .to_string();
    let s = sim();
    s.requests.push(url.clone());
    let Some((_, status, body)) = s.http.iter().find(|(pat, _, _)| url.contains(pat.as_str())).cloned()
    else {
        return ((404i64) << 32) | 0;
    };
    ((status as i64) << 32) | fill(out, out_cap, &body) as i64
}

/// `host_ssh` for the host tests.
///
/// Deliberately a **refusal**, not a simulated SSH server: the transport's value
/// is that it speaks to a real sshd, and a simulator here would test this file
/// rather than the protocol. The clone-over-SSH path is covered by the
/// `git_clone_ssh` e2e scenario against real OpenSSH; what this stub pins is that
/// an SSH URL fails *cleanly* on a build with no SSH available.
///
/// # Safety
/// Matches the wasm import signature; `out` must be valid for `out_cap` bytes.
pub unsafe fn host_ssh(_req: *const u8, _req_len: i32, out: *mut u8, out_cap: i32) -> i64 {
    const MSG: &[u8] = b"no ssh transport in the host test harness";
    let n = MSG.len().min(out_cap.max(0) as usize);
    if n > 0 {
        core::ptr::copy_nonoverlapping(MSG.as_ptr(), out, n);
    }
    -5
}
