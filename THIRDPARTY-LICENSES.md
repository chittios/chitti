# Third-party licenses

ChittiOS is licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and
[NOTICE](NOTICE)). It bundles the following third-party source in-tree; each
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

## wasmi — crates.io (WebAssembly interpreter)

Agent-authored tools (`assets/tools.wasm`) and the PDF rasterizer run on
[wasmi](https://github.com/wasmi-labs/wasmi) 1.1 with the `simd` feature — a
pure-Rust, `no_std` + alloc WebAssembly **interpreter** (not a JIT). `simd` is
what lets a guest compiled with `-Ctarget-feature=+simd128` execute vector
instructions rather than scalarized ones; `wasmi_core` is also a direct dependency
purely to name the `LimiterError` type wasmi does not re-export. Fuel metering and
`ResourceLimiter` bound instruction count and linear memory; guests may only
touch the world through capability-gated host imports registered by
`kernel/src/agent/wasm_rt.rs`. wasmi and its crates (`wasmi_core`,
`wasmi_ir`, `wasmi_collections`, `wasmparser`, `libm`, `spin`, …) resolve from
crates.io.

Licensed under **MIT OR Apache-2.0**. Full license texts ship in each crate's
source under `~/.cargo/registry/` and upstream.

---

## hayro + vello_cpu — crates.io (PDF page rasterizer, `assets/wasm/pdfrender.wasm`)

The PDF viewer renders real pages with [hayro](https://github.com/LaurenzV/hayro)
— a pure-Rust PDF interpreter — over
[vello_cpu](https://github.com/linebender/vello), a CPU 2D rasterizer. Neither is
vendored: `tools/pdfrender-wasm/` depends on them from crates.io (pinned by its
`Cargo.lock`) and is compiled to **wasm**, and it is that module
(`assets/wasm/pdfrender.wasm`, checked in) which ships in the kernel image and runs
under the `wasmi` sandbox above. The tree therefore redistributes the compiled
artifact but carries no copy of the sources.

hayro's own NOTICE records code adapted from **PDFBox** (Apache-2.0), **pdf.js**
(Apache-2.0) and the **png** crate (Apache-2.0). Its dependency set includes
`skrifa`/`read-fonts`, `kurbo`, `peniko`, `color`, `zune-jpeg`, `flate2`, `brotli`,
`pic-scale`, `moxcms`, `image` and `fearless_simd`, and hayro embeds substitute
fonts for the 14 PDF standard faces plus the 61 predefined CMaps.

hayro and vello_cpu are licensed under **MIT OR Apache-2.0**. Because the compiled
module is what this tree distributes, the data baked into it is worth naming
explicitly: the substitute standard fonts are the **Foxit/PDFium** Type 1 faces
(`FoxitSans`, `FoxitSerif`, `FoxitFixed`, `FoxitDingbats`, …), "Copyright 2014
PDFium Authors / original code copyright 2014 Foxit Software Inc.", under PDFium's
**BSD-3-Clause**; the bundled CMYK ICC profile is **CC0-1.0** (Compact ICC
Profiles). Full license texts ship in each crate's source under
`~/.cargo/registry/` and upstream.

---

## brotli-decompressor — crates.io (WOFF2 web fonts)

WOFF2 web fonts are Brotli-compressed. `kernel/src/font_woff2.rs` decompresses
the payload with [brotli-decompressor](https://github.com/dropbox/rust-brotli-decompressor)
4.x — a pure-Rust, `no_std` + `alloc` Brotli decoder (default features off; the
crate is written to avoid the Rust stdlib). It is driven with a small
`alloc`-backed `Allocator` so decompression uses the kernel heap, never the
large stack buffers the crate's bundled no_std wrapper would allocate. The
`glyf`/`loca` transform-reversal in `font_woff2.rs` is a faithful port of
fontTools' WOFF2 reconstruction (fontTools: **MIT**).

Licensed under **BSD-3-Clause OR MIT** (`alloc-no-stdlib` likewise). Full
license texts ship in the crate source under `~/.cargo/registry/`.

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

---

## Symphonia AAC-LC (Rust port) — `kernel/src/audio/aac/`

The in-kernel AAC-LC decoder used by `/open <file>.mp4` (audio track) is a
`no_std` port of the [Symphonia](https://github.com/pdeljanov/Symphonia)
`symphonia-codec-aac` path (which itself ported NihAV's AAC decoder with the
author's permission). Spectral Huffman tables, ICS/CPE/TNS/pulse decode, the
Kaiser-Bessel / sine windows, and the FFT-based IMDCT follow that implementation.

Copyright (c) 2019–2026 The Project Symphonia Developers.  
Previous Author: Kostya Shishkov \<kostya.shiskov@gmail.com\>

Licensed under the **Mozilla Public License 2.0 (MPL-2.0)**:

```
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

---

## oxideav-aac SBR/PS (Rust port) — `kernel/src/audio/aac/sbr/`

HE-AAC Spectral Band Replication and Parametric Stereo reconstruction is a
`no_std` port of [oxideav-aac](https://github.com/OxideAV/oxideav-aac)
(`sbr_*.rs`, `ps_*.rs`) by Karpelès Lab Inc. Bit-reader calls were retargeted
to the in-tree AAC `BitReader`; float math uses local pure-Rust helpers under
`sbr/math.rs`.

Copyright (c) 2026 Karpelès Lab Inc.

Licensed under the **MIT License**:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## rust_h264 — `third_party/rust_h264/`

Pure-Rust H.264/AVC decoder (Baseline/Main/High, CAVLC + CABAC) used as the
**primary** video decode backend for `/open`. Vendored from crates.io
`rust_h264` 0.4.0 and patched for `no_std` + `alloc` (BTreeMap, spin::Once,
etc.). Our hand-rolled CAVLC/CABAC path in `kernel/src/video/h264/` remains as
an automatic fallback if rust_h264 rejects a stream.

Licensed under **MIT OR Apache-2.0** — see `third_party/rust_h264/LICENSE-MIT`
and `LICENSE-APACHE`. Upstream: https://github.com/roticv/rust_h264

---

## fontdue — crates.io (TTF/OTF runtime rasterizer)

Browser text (and optional UI faces) use
[`fontdue`](https://crates.io/crates/fontdue) 0.9 — pure-Rust, **`no_std` +
`alloc`** TrueType/OpenType parser + rasterizer (`default-features = false`,
`hashbrown`). The default face is the in-tree
[`assets/GeistMono-Regular.ttf`](assets/GeistMono-Regular.ttf) (SIL OFL; see
`assets/GeistMono-OFL.txt`), loaded via `include_bytes!` in
`kernel/src/font_ttf.rs`. Additional `.ttf`/`.otf` files can be loaded at runtime
with `font_ttf::load_bytes`.

Licensed under **MIT OR Apache-2.0 OR Zlib**. Upstream:
https://github.com/mooman219/fontdue. Depends on **ttf-parser** (Apache-2.0 /
MIT) and **hashbrown**.

## Noto fonts — script fallback chain (Indic / emoji / CJK)

Non-Latin text (browser **and** console/UI) is covered by Google's **Noto**
family (the same faces Linux ships in `fonts-noto`), registered as a per-glyph
fallback chain in `kernel/src/font_ttf.rs` (see `NOTO_FALLBACKS` /
`register_bundled_fallbacks`) and consulted by both the browser paint path and
the UI-cell rasterizer. All faces are baked into the kernel via `include_bytes!`.

- **Noto Sans Devanagari/Bengali/Gurmukhi/Gujarati/Tamil/Telugu/Kannada/Malayalam**
  — `assets/fonts/Noto-*.ttf`, from https://github.com/notofonts/notofonts.github.io
- **Noto Emoji** (monochrome; fontdue has no colour-table support) —
  `assets/fonts/Noto-Emoji.ttf`, from https://github.com/google/fonts
- **Noto Sans CJK** — `assets/fonts/Noto-CJK.otf`, a **subset** of Noto Sans CJK
  SC (Latin + kana + CJK punctuation + ~3.5k common Han, ~8k glyphs / ~1.7 MB)
  produced with `fonttools pyftsubset`, from https://github.com/notofonts/noto-cjk.
  The full 65k-glyph face is *not* used: fontdue's per-glyph allocation churns
  the kernel's first-fit allocator ~O(glyphs²), so the full face parses for
  minutes; the subset parses in ~1-2 s. Covers Chinese + Japanese; Hangul
  (Korean) is omitted to keep the glyph count parse-able.

All are licensed under the **SIL Open Font License 1.1**. Indic pre-base matra
reordering (`kernel/src/font_shape.rs`) is an original minimal shaper, not a
port; full OpenType shaping (GSUB conjuncts, GPOS mark positioning) is a
documented follow-up.

---

## htmlparser — crates.io (browser HTML tokenizer)

The browser agent (`kernel/src/browser/`) uses
[`htmlparser`](https://crates.io/crates/htmlparser) 0.2 as a pull-based,
zero-allocation **tokenizer** (`default-features = false` → `no_std`). The DOM
tree, layout, and paint pipeline are first-party code; only token boundaries
come from the crate. Zero transitive dependencies.

Licensed under **MIT OR Apache-2.0**. Upstream:
https://github.com/jdrouet/htmlparser

### Ladybird / LibWeb (architecture reference, not vendored)

The browser pipeline stages (HTML → style → layout → paint, plus a sandboxed
JS interpreter) follow the split used by
[Ladybird](https://github.com/SerenityOS/serenity/tree/master/Ladybird) /
LibWeb + LibJS. **No Ladybird or Serenity C++ sources are compiled into
Chitti** — they are `std`/GUI multi-process engines unsuitable for this
`no_std` kernel. Ideas are reimplemented in pure Rust under
`kernel/src/browser/`. Serenity/Ladybird code is typically BSD-2-Clause;
we do not copy their sources.

### just-engine (applegrew/just) — ES6 JavaScript engine, vendored + no_std'd

The browser's ES6 JavaScript tier is the [`just`](https://github.com/applegrew/just)
engine (crate `just-engine`), **vendored in-tree at `third_party/just-ref`** and
adapted to `no_std` + `alloc` for the kernel (see `kernel/src/browser/js_just.rs`).
Only the **parser + tree-walking interpreter + standard-library built-ins** are
compiled into the kernel; the Cranelift JIT and the two bytecode VMs
(`src/runner/jit/`) are host-only and feature-gated off (`jit`). One source tree
serves both the kernel (`default-features = false`, no_std) and the host
webcompat harness (`tools/webcompat/just_runner`, default `std` + `jit`).
Kernel-side substitutions: `std::collections::HashMap` → `hashbrown`, `uuid` →
an atomic counter for anonymous symbols, `std::time`-seeded `Math.random` → an
atomic LCG, `lazy_static` → its `spin_no_std` feature, `console.*` → a drainable
global sink, `f64`/`f32` transcendentals → `num-traits`/`libm`.

Licensed under **MIT OR Apache-2.0**. Upstream: https://github.com/applegrew/just

### pest parser stack — vendored + no_std'd (for just-engine)

`just`'s parser uses [pest](https://github.com/pest-parser/pest) (a PEG parser)
via [`pest_consume`](https://github.com/Nadrieril/pest_consume). Neither
supports `no_std` at the pinned versions (pest 2.1.3, pest_consume 1.0.6), so the
stack is **vendored under `third_party/{pest,pest_consume,pest_generator,`
`pest_consume_macros,pest_derive,pest_meta}`** and patched:

* runtime `pest` + `pest_consume`: `#![no_std]` + `extern crate alloc`, the
  `std::` → `core::`/`alloc::` module moves, `std::error::Error` impl dropped,
  `prec_climber`'s `HashMap` → `BTreeMap`;
* the code-generators (`pest_generator`, `pest_consume_macros` — proc-macro
  crates that stay `std` on the host build) emit `::core::`/`::alloc::` paths and
  a fully-qualified `::alloc::boxed::Box` instead of `::std::…`/bare `Box`;
* `proc-macro-hack` (unmaintained) removed — `match_nodes!` is now a plain
  `#[proc_macro]` (function-like proc-macros work in expression position on
  modern Rust).

The kernel's `[patch.crates-io]` in `kernel/Cargo.toml` repoints these crates to
the vendored copies for its build only; the host `just_runner` keeps the
crates.io versions. All are **MIT OR Apache-2.0** (pest: MIT/Apache-2.0).
Transitive: `ucd-trie` (no_std), `hashbrown`, `num-traits`+`libm`, `spin`.

---

## m1n1 — `third_party/m1n1/` (git submodule, build-time only)

The [m1n1](https://github.com/AsahiLinux/m1n1) bootloader is the Asahi Linux
first stage for Apple Silicon. It is referenced as a **git submodule** (not
copied source) and is a **build-/boot-time dev dependency only** — it is *not*
linked into or distributed as part of the ChittiOS kernel image. It is used to
package ChittiOS as an arm64 `Image` and boot it over the m1n1 USB proxy on a
real Mac (`cargo xtask m1n1`). Its own build products (`build/m1n1.bin`,
`build/m1n1.macho`) and its nested `artwork` submodule stay within the submodule
tree.

Licensed under the **MIT License** (Copyright The Asahi Linux Contributors;
`third_party/m1n1/LICENSE`).

## AgentDojo (MIT)

`kernel/src/security/redteam.rs` embeds the **goal text of the 27 injection
tasks** from AgentDojo, quoted verbatim with the suites' own attacker constants
substituted, to run a third-party attack corpus against the Synapse boundary
(`/redteam`, paper §E2b). No AgentDojo code is used --- only the task text and
the tool each task drives, translated onto ChittiOS primitives.

Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic,
Luca Beurer-Kellner, Marc Fischer, and Florian Tramer.
Licensed under the MIT License. Source: https://github.com/ethz-spylab/agentdojo

## InjecAgent (MIT)

`kernel/src/security/redteam.rs` embeds the **attacker instructions of the 62
cases** from InjecAgent (30 direct-harm, 32 data-stealing), quoted verbatim, to
run a second third-party attack corpus against the Synapse boundary (`/redteam`,
paper §E2b). No InjecAgent code is used --- only the instruction text and the
attacker tool each case drives, translated onto ChittiOS primitives.

Copyright (c) 2024 Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang.
Licensed under the MIT License.
Source: https://github.com/uiuc-kang-lab/InjecAgent

---

## Font Awesome Free — `assets/fonts/FontAwesome7Free-Solid-900.otf`

[Font Awesome Free](https://fontawesome.com) **7.3.1** Solid face used for system
UI icons (status bar, cursors, close marks, tab labels, agents browser, todos,
settings chrome). Bundled as a single OTF and registered first in the kernel TTF
fallback chain (`font_ttf::register_bundled_fallbacks` / `FA_FALLBACK_NAME`).

- **Fonts**: SIL Open Font License 1.1  
- **Icons**: CC BY 4.0  
- **Code** (upstream JS, unused here): MIT  

Full license text: https://fontawesome.com/license/free  
Only the **Free Solid** face is vendored (no Brands pack). Codepoints are the
Font Awesome 7 Free Solid Private Use Area map (`kernel/src/icons.rs`).

---

## `/samples/` corpus — `assets/samples/` (fetched, **not** vendored)

The sample images / video / audio / documents that `make run` embeds into an image
and the OS seeds into `/samples/` at boot. **Nothing here is committed to this
repository** — `assets/samples/` is gitignored and the files are downloaded on
demand by `cargo xtask sample-files`, so the source tree redistributes none of
them (the same posture as the voice models in `assets/voice/` and the WiFi
firmware in `assets/wifi/`).

Each file keeps the licence of its upstream source. The authoritative,
per-file record — name, purpose, source URL — is generated by the fetch into
`assets/samples/README.md` and embedded as `/samples/README.md`, so an image
carries its own provenance. Sources currently used:

- **OpenCV** `samples/data` (`fruits.jpg`, `baboon.jpg`, `sudoku.png`) — Apache-2.0
- **libpng PNGSuite** (`transparency.png` = `basn6a08.png`, `grayscale.png` = `basn0g08.png`) — freely usable PNG conformance images
- **Big Buck Bunny** 360p clip via test-videos.co.uk (`sample.mp4`) — © Blender Foundation, CC BY 3.0
- **whisper.cpp** `samples/jfk.wav` (`jfk-speech.wav`) — JFK speech excerpt, US public domain
- **rafaelreis-hotmart/Audio-Sample-files** (`sample.wav`, `sample.mp3`, `sample.ogg`)
- **FFmpeg sample archive** (`sample.aac`)
- **py-pdf/sample-files** (`minimal.pdf`, `document.pdf`)
- **IETF RFC 1951** (`rfc1951-deflate.txt`) — IETF document, freely distributable
- **vega-datasets** (`cars.json`, `seattle-weather.csv`) — BSD-3-Clause
- **CERN** first web page (`first-web-page.html`)

Anyone redistributing a built image with `SAMPLES=1` is redistributing these
files and should honour those licences; `SAMPLES=` builds an image without them.
