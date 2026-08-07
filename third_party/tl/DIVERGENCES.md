# Divergences from upstream `tl` 0.7.8

Vendored from crates.io (MIT, y21) so ChittiOS can run it as an **alternative**
HTML parser to `kernel/src/browser/html.rs`, switchable at runtime with
`/html ours|tl`. Our own parser is not removed and remains the default.

The conversion is shallow on purpose — nothing algorithmic changed, so upstream
fixes still apply nearly cleanly.

## 1. `no_std` + `alloc`

`#![no_std]` and `extern crate alloc` in `lib.rs`, and every `std::` path
rewritten to its `core::` or `alloc::` equivalent. `no_std` has no prelude, so
each module gained explicit `use alloc::{string::String, vec::Vec, boxed::Box,
format}` as it needed them.

## 2. `HashMap` -> `hashbrown`

`std::collections::HashMap` has no `core`/`alloc` equivalent.
`hashbrown` with `default-hasher` is a drop-in and is **already in the kernel's
dependency graph** (fontdue pulls it), so this adds no new third-party code.

Sites: `parser/base.rs`, `inline/hashmap.rs`.

## 3. `std::error::Error` impls dropped

`errors.rs` implemented it for `ParseError` and `SetBytesError`. There is no
`core` equivalent (the trait is still unstable there). Nothing in the crate or
in the kernel's adapter uses `dyn Error` — both errors are matched on directly —
so this is invisible to every caller.

## 4. `src/tests.rs` removed

It uses `std::thread` and `std::panic::catch_unwind`, neither of which exists
here, and `cargo test` cannot run inside the kernel anyway. The kernel's own
`#[test_case]` coverage of the adapter lives in `browser/html_tl.rs`; the
host-side speed and correctness comparison is `tools/webbench`, which tests
upstream `tl` from crates.io directly.

## 5. `src/simd/nightly.rs` removed

It needs `core::simd` (`portable_simd`). The `simd` feature is off — the scalar
`stable.rs` path is what is measured and shipped. Removing the module rather
than leaving it feature-gated keeps `no_std` verifiable without a nightly
feature flag.

## 6. `#![doc = include_str!("../README.md")]` replaced

The README is not vendored; `#![deny(missing_docs)]` then needs a crate-level
doc comment, which `lib.rs` now carries inline.

## 7. `read_ident` no longer swallows a trailing `/` (upstream bug fix)

`util::is_ident` counts `/` as an identifier byte, so `<br/>` reads its tag name
as `br/`. That name is not in `VOID_TAGS`, and the self-closing check that
follows finds `>` rather than `/` — so the tag is pushed onto the open-element
stack and **every later element nests inside it**. `<br/><hr/><div>x</div>`
comes out as `br/ > hr/ > div` instead of three siblings. Attribute names have
the same problem (`<input disabled/>` reads `disabled/`).

`read_ident` now trims one trailing `/` and rewinds the stream a byte, so the
caller's `expect_and_skip_cond(b'/')` sees the slash it was always looking for.
Fixed there rather than in `is_ident`, whose `/` is load-bearing for the
query-selector tokenizer that shares it.

Found by the kernel's cross-engine tree-equality test
(`browser::html_tl::tests::both_engines_build_the_same_tree_for_our_pages`) on
markup `html::preprocess` emits — which is the whole argument for running two
parsers: neither could have found this alone.
