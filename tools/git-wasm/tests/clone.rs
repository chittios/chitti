//! Clone over the smart-HTTP protocol, driven against the host simulator.
//!
//! These live in `tests/` rather than in the crate on purpose: an integration
//! test compiles the library **without** `cfg(test)`, which is what lets
//! `hostsim` mount the kernel's real `image/inflate.rs` and `net/sha1.rs` — their
//! own suites use the kernel's `#[test_case]` framework and would not compile
//! under a host `cargo test` build of this crate.
//!
//! `fixture.pack` is a packfile produced by **real git** (`git pack-objects`)
//! from a three-file repository with a nested `src/net/` directory. Real bytes
//! matter here: every earlier version of the pack walk was checked against packs
//! this code could also have written, and the interesting failures are all in
//! what a real server sends — Huffman-compressed streams, exact stream lengths,
//! subdirectory trees.

use chitti_git_wasm::{git, hostsim};

/// The fixture repo's HEAD commit, its tree, and its files.
const HEAD: &str = "717e80764843313dbdaa42b43f3be97bb725c0c8";
const PACK: &[u8] = include_bytes!("fixture.pack");

/// A second real repository, two commits deep, whose history git **deltifies** —
/// packed twice by `git pack-objects`, once in each base-reference form
/// (`--delta-base-offset` or not). Both hold one delta among twelve objects.
const DELTA_HEAD: &str = "35528a725a8925975d241e31a82755548dced31f";
const REF_DELTA_PACK: &[u8] = include_bytes!("ref-delta.pack");
const OFS_DELTA_PACK: &[u8] = include_bytes!("ofs-delta.pack");
const FILES: &[(&str, &str)] = &[
    ("README.md", "hello chitti\n"),
    ("src/main.rs", "fn main() {}\n"),
    ("src/net/mod.rs", "pub fn poll() {}\n"),
];

/// Refs advertisement in pkt-line form, as `GET /info/refs` returns it.
fn advertisement(head: &str) -> Vec<u8> {
    let mut out = b"001e# service=git-upload-pack\n0000".to_vec();
    for (sha, name) in [(head, "HEAD"), (head, "refs/heads/main")] {
        let payload = format!("{sha} {name}\0multi_ack ofs-delta no-progress\n");
        out.extend_from_slice(format!("{:04x}{payload}", 4 + payload.len()).as_bytes());
    }
    out.extend_from_slice(b"0000");
    out
}

/// `POST /git-upload-pack` returns `NAK` then the raw pack.
fn upload_pack_body(pack: &[u8]) -> Vec<u8> {
    let mut out = b"0008NAK\n".to_vec();
    out.extend_from_slice(pack);
    out
}

/// Install the simulated machine and script the two endpoints a clone hits. The
/// returned guard must be held for the test's body — see `hostsim::reset`.
#[must_use]
fn boot_at(pack: &[u8], head: &str) -> hostsim::Guard {
    let g = hostsim::reset("/agent/9047", "/home/chitti");
    let sim = hostsim::sim();
    sim.reply("/info/refs", 200, advertisement(head));
    sim.reply("/git-upload-pack", 200, upload_pack_body(pack));
    g
}

/// [`boot_at`] for the checked-in fixture, whose HEAD is known.
#[must_use]
fn boot(pack: &[u8]) -> hostsim::Guard {
    boot_at(pack, HEAD)
}

fn read(path: &str) -> Option<String> {
    hostsim::sim().files.get(path).map(|v| String::from_utf8_lossy(v).into_owned())
}

/// **A real packfile clones, and every file lands where the tree says.**
///
/// The whole path in one assertion: pkt-line parsing, the want/done exchange, the
/// pack header, per-object zlib streams walked by their reported consumed length,
/// loose-object writes, checkout and the index.
#[test]
fn a_real_packfile_clones_and_checks_out() {
    let _g = boot(PACK);
    let out = git::command("clone https://example.invalid/r.git");
    assert!(out.starts_with("ok:"), "{out}");
    assert!(out.contains("7 objects"), "{out}");

    for (path, want) in FILES {
        assert_eq!(
            read(&format!("/home/chitti/r/{path}")).as_deref(),
            Some(*want),
            "{path} is missing or wrong after checkout"
        );
    }
}

/// **Nested files keep their directory.**
///
/// The checkout walked subtrees without carrying the path prefix, so every nested
/// blob was written at the repo root under its bare basename — a flat pile, with
/// same-named files in different directories overwriting each other. Pinned
/// separately from the clone test because the symptom is *files exist and have the
/// right contents*, which the assertion above would have accepted.
#[test]
fn a_nested_file_is_not_flattened_into_the_repo_root() {
    let _g = boot(PACK);
    assert!(git::command("clone https://example.invalid/r.git").starts_with("ok:"));
    assert!(
        read("/home/chitti/r/main.rs").is_none(),
        "src/main.rs was written at the repo root"
    );
    assert!(read("/home/chitti/r/src/main.rs").is_some());
    assert!(read("/home/chitti/r/src/net/mod.rs").is_some());
}

/// **A fresh clone has nothing to report.**
///
/// The index is built from the same walk as the checkout, so paths agree; when
/// they did not, every file under a subdirectory showed as untracked immediately
/// after cloning.
#[test]
fn status_after_a_clone_is_clean() {
    let _g = boot(PACK);
    assert!(git::command("clone https://example.invalid/r.git").starts_with("ok:"));
    let st = git::command("status");
    assert!(st.contains("nothing to commit"), "{st}");
    assert!(st.contains("on branch main"), "{st}");
}

/// **A packfile bigger than the guest's buffer arrives whole, by growing it.**
///
/// This is the bug. The buffer was a fixed 64 KiB and the host answered
/// `min(len, cap)`, so a real repository's pack — 182 KiB for the one that was
/// reported — came back truncated and *indistinguishable from a complete one*,
/// and the clone failed inside the decompressor with `object inflate failed`.
///
/// The pack here is deliberately over the guest's **starting** size, not merely
/// over the old 64 KiB: a test that fits in the first buffer passes whether or not
/// the host reports truncation, which is exactly the hole an earlier draft of this
/// test had. The blob is incompressible so the pack cannot shrink back under the
/// threshold, and the assertion is on the file's *contents* — a truncated pack
/// that happened to parse would still lose its tail.
#[test]
fn a_packfile_larger_than_the_buffer_is_not_truncated() {
    let big = noise(2 << 20);
    let (pack, head) = pack_of(&[("big.txt", big.as_str())]);

    let _g = boot_at(&pack, &head);
    let out = git::command("clone https://example.invalid/r.git");
    assert!(out.starts_with("ok:"), "{out}");
    assert_eq!(read("/home/chitti/r/big.txt").as_deref(), Some(big.as_str()));
    // info/refs, the first upload-pack, and the one retry at the reported size.
    let reqs = &hostsim::sim().requests;
    assert_eq!(reqs.len(), 3, "an oversized pack costs exactly one retry: {reqs:?}");
}

/// **A clone downloads the pack once when it fits.**
///
/// Growing the buffer costs a *second request* for HTTP, unlike a file read — so
/// the starting size is chosen to cover ordinary repositories, and this pins that
/// the common case does not silently pay for it twice.
#[test]
fn a_clone_that_fits_makes_one_request_per_endpoint() {
    let _g = boot(PACK);
    assert!(git::command("clone https://example.invalid/r.git").starts_with("ok:"));
    let reqs = &hostsim::sim().requests;
    assert_eq!(reqs.len(), 2, "{reqs:?}");
}

/// **A file bigger than the read buffer is read whole, not to its first 64 KiB.**
///
/// The same contract on the cheap side: a `git add` staged the truncated prefix of
/// any file over the buffer and hashed *that*, so the committed blob silently was
/// not the file. Unlike HTTP, growing here costs only a second local read.
///
/// Sized past the buffer the clone above already grew it to — a smaller file lands
/// in slack space and proves nothing.
#[test]
fn a_file_larger_than_the_read_buffer_is_staged_whole() {
    let _g = boot(PACK);
    assert!(git::command("clone https://example.invalid/r.git").starts_with("ok:"));
    let big = noise(2 << 20);
    hostsim::sim().files.insert("/home/chitti/r/big.txt".into(), big.clone().into_bytes());

    assert!(git::command("add .").starts_with("ok:"));
    assert!(git::command("commit -m big").starts_with("ok:"));
    // The blob the commit points at must be the whole file.
    let staged = git::command("status");
    assert!(staged.contains("nothing to commit"), "{staged}");
    let blob = hostsim::sim()
        .files
        .keys()
        .find(|k| k.starts_with("/home/chitti/r/.git/objects/"))
        .cloned()
        .expect("objects were written");
    let _ = blob;
    // Round-trip through checkout: the working file must come back byte-identical.
    hostsim::sim().files.remove("/home/chitti/r/big.txt");
    assert!(git::command("checkout main").starts_with("ok:"));
    assert_eq!(read("/home/chitti/r/big.txt").as_deref(), Some(big.as_str()));
}

/// Incompressible-enough filler: `zlib_stored` does not shrink it, so a fixture
/// built from it really is the size it looks.
fn noise(n: usize) -> String {
    (0..n as u32)
        .map(|i| char::from(b'!' + (i.wrapping_mul(2_654_435_761) >> 24) as u8 % 90))
        .collect()
}

/// **A REF_DELTA object resolves — its base sha is raw in the pack, not compressed.**
///
/// The second bug, and the one the truncation fix uncovered. A delta object's base
/// reference sits between the type/size header and the zlib stream: twenty raw
/// bytes of sha here, an offset varint for `a_pack_with_an_ofs_delta_clones`. The
/// walk read it out of the *decompressed* payload instead, so it inflated from
/// twenty bytes too early, handed the decoder a sha, and got `not deflate` back —
/// reported as `object inflate failed (object 5 of 172)`.
///
/// The old fixture had no deltas at all, which is why the first round of tests
/// passed while a real clone still failed at object 5. Any repository with more
/// than a couple of commits gets deltified by the server, so this is the case that
/// matters most and it was the one not covered.
#[test]
fn a_pack_with_a_ref_delta_clones() {
    let _g = boot_at(REF_DELTA_PACK, DELTA_HEAD);
    let out = git::command("clone https://example.invalid/r.git");
    assert!(out.starts_with("ok:"), "{out}");
    assert_deltified_checkout();
}

/// **An OFS_DELTA object resolves — its base is a byte count back to a header.**
///
/// The same pack from the same repository, written with `--delta-base-offset`.
/// Servers pick the form, so both have to work; the two differ only in how the
/// base is named, and getting the *offset* frame wrong (from the stream rather
/// than from the object header) yields a plausible earlier object rather than an
/// error.
#[test]
fn a_pack_with_an_ofs_delta_clones() {
    let _g = boot_at(OFS_DELTA_PACK, DELTA_HEAD);
    let out = git::command("clone https://example.invalid/r.git");
    assert!(out.starts_with("ok:"), "{out}");
    assert_deltified_checkout();
}

/// The deltified fixture's second commit: a 400-line file whose first version is
/// the delta base, edited at two places, plus a file added in the same commit.
///
/// Asserting the *reconstructed bytes* rather than "the clone succeeded" is the
/// point — a delta applied against the wrong base, or with the payload offset by
/// the twenty bytes of its own base reference, still produces a file.
fn assert_deltified_checkout() {
    let body = read("/home/chitti/r/src/net/mod.rs").expect("the deltified file");
    let lines: Vec<&str> = body.split('\n').collect();
    assert_eq!(lines.len(), 400, "wrong line count: {}", lines.len());
    assert_eq!(body.len(), 18_398, "wrong byte count");
    assert_eq!(lines[10], "// a small edit that git will store as a delta");
    assert_eq!(lines[200], "// and another one further down");
    assert_eq!(read("/home/chitti/r/src/main.rs").as_deref(), Some("fn main() {}\n"));
    assert_eq!(read("/home/chitti/r/README.md").as_deref(), Some("hello chitti\n"));
    let st = git::command("status");
    assert!(st.contains("nothing to commit"), "{st}");
}

/// **A truncated pack is an error, not a partial clone.**
#[test]
fn a_truncated_pack_is_refused() {
    let _g = boot(&PACK[..PACK.len() - 120]);
    let out = git::command("clone https://example.invalid/r.git");
    assert!(out.starts_with("error:"), "{out}");
}

// --- fixture construction ---------------------------------------------------

/// Build a packfile holding a blob/tree/commit for each named file, using the
/// same stored-block zlib the kernel's `host_deflate` emits (which real git also
/// accepts — it is `core.compression=0`).
///
/// Returns the pack and the commit sha to advertise as HEAD.
///
/// Only used for the size cases; correctness against *real* git bytes is what
/// `fixture.pack` is for.
fn pack_of(files: &[(&str, &str)]) -> (Vec<u8>, String) {
    fn obj(kind: &str, body: &[u8]) -> ([u8; 20], Vec<u8>) {
        let mut raw = format!("{kind} {}\0", body.len()).into_bytes();
        raw.extend_from_slice(body);
        (hostsim::sha1::sha1(&raw), body.to_vec())
    }
    let mut objs: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut tree = Vec::new();
    for (name, content) in files {
        let (sha, body) = obj("blob", content.as_bytes());
        objs.push((3, body));
        tree.extend_from_slice(format!("100644 {name}\0").as_bytes());
        tree.extend_from_slice(&sha);
    }
    let (tree_sha, tree_body) = obj("tree", &tree);
    objs.push((2, tree_body));
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let commit = format!(
        "tree {}\nauthor t <t@t> 0 +0000\ncommitter t <t@t> 0 +0000\n\nfixture\n",
        hex(&tree_sha)
    );
    let (commit_sha, commit_body) = obj("commit", commit.as_bytes());
    objs.push((1, commit_body));

    let mut pack = b"PACK".to_vec();
    pack.extend_from_slice(&2u32.to_be_bytes());
    pack.extend_from_slice(&(objs.len() as u32).to_be_bytes());
    for (typ, body) in &objs {
        let mut size = body.len() as u64;
        let mut first = ((*typ as u64) << 4) | (size & 0x0f);
        size >>= 4;
        if size > 0 {
            first |= 0x80;
        }
        pack.push(first as u8);
        while size > 0 {
            let mut b = (size & 0x7f) as u8;
            size >>= 7;
            if size > 0 {
                b |= 0x80;
            }
            pack.push(b);
        }
        pack.extend_from_slice(&hostsim::deflate::zlib_stored(body));
    }
    pack.extend_from_slice(&[0u8; 20]); // trailer, unchecked by the walk
    (pack, hex(&commit_sha))
}
