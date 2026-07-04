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
