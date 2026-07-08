# Third-party licenses

Chitti OS is licensed under the GNU General Public License v3.0 (see
[LICENSE](LICENSE)). It bundles the following third-party source in-tree; each
retains its own license, reproduced below.

---

## smoltcp — `third_party/smoltcp/`

The [smoltcp](https://github.com/smoltcp-rs/smoltcp) `no_std` TCP/IP stack backs
the [`net`](kernel/src/net/) subsystem (DHCPv4, DNS, ICMP, TCP/UDP over the
virtio-net / e1000 drivers). It is **vendored** (upstream `main`) rather than
pulled from crates.io so the stack can be read, patched, and step-debugged
alongside the kernel. smoltcp's own dependencies (`managed`, `byteorder`,
`bitflags`, `cfg-if`, `heapless`) still resolve normally from crates.io.

Licensed under the **0-clause BSD (0BSD)** license
(`third_party/smoltcp/LICENSE-0BSD.txt`):

```
Copyright (C) smoltcp contributors

Permission to use, copy, modify, and/or distribute this software for
any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

---

## embedded-tls + RustCrypto — crates.io (TLS 1.3 client)

`https://` support ([`net/tls.rs`](kernel/src/net/tls.rs)) is built on
[`embedded-tls`](https://github.com/drogue-iot/embedded-tls) (a `no_std`,
allocator-optional TLS 1.3 client) driven in blocking mode over an
`embedded-io` adapter around the smoltcp TCP socket. Its cryptography comes
from the pure-Rust [RustCrypto](https://github.com/RustCrypto) crates it depends
on — `aes-gcm`, `sha2`, `hkdf`, `hmac`, `p256`, `digest` — plus `rand_core` /
`rand_chacha` for the handshake CSPRNG. All resolve from crates.io (no C or
assembly, so they build under the kernel's custom `build-std` targets).

- **embedded-tls**, **embedded-io**: Apache-2.0.
- **RustCrypto** crates (aes-gcm, sha2, hkdf, hmac, p256, digest, …),
  **rand_core**, **rand_chacha**: dual-licensed **MIT OR Apache-2.0**.

Full license texts ship in each crate's source under
`~/.cargo/registry/` and in the respective upstream repositories.

---

## minimp3 (Rust port) — `kernel/src/audio/mp3.rs`

The `/open <file>.mp3` player's MPEG Layer III decoder is a hand-written
`no_std` Rust **port** of [minimp3](https://github.com/lieff/minimp3)
(scalar path, Layer III only); its numeric tables are generated verbatim from
`minimp3.h` by [`tools/gen_mp3_tables.py`](tools/gen_mp3_tables.py) into
[`kernel/src/audio/mp3_tables.rs`](kernel/src/audio/mp3_tables.rs). The port
is validated sample-for-sample (±1 LSB) against minimp3's own scalar decode.

minimp3 is dedicated to the public domain under **CC0 1.0**:

```
To the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the
public domain worldwide. This software is distributed without any
warranty. See <http://creativecommons.org/publicdomain/zero/1.0/>.
```

---

## embedded-tls — `third_party/embedded-tls/`

The `https://` client ([`net/tls.rs`](kernel/src/net/tls.rs)) is built on
[embedded-tls](https://github.com/drogue-iot/embedded-tls) (a `no_std` TLS 1.3
client). It is **vendored in-tree** (upstream 0.17.0) rather than pulled from
crates.io so its handshake parser can be patched for real-world CDN
compatibility — modern servers (e.g. `upload.wikimedia.org`) advertise
post-quantum hybrid key-exchange groups (X25519MLKEM768) in their
EncryptedExtensions `supported_groups`, which the stock crate's `NamedGroup`
enum rejects, aborting the handshake over an informational field. The in-tree
copy parses that (and a few over-tight extension vectors) leniently. Its
dependencies (heapless, p256, aes-gcm, sha2, rand_core, …) still resolve from
crates.io.

Licensed under **Apache-2.0** (see `third_party/embedded-tls/LICENSE`).
