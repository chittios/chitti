# Assets

What lives in this folder (and what is fetched into it, gitignored).

## Fonts (committed)

`GeistMono-Regular.ttf` — [Geist Mono](https://vercel.com/font) by Vercel
(in collaboration with basement.studio), licensed under the SIL Open Font
License 1.1 (see `GeistMono-OFL.txt`).

**Two paths:**

1. **Console / compositor** — `tools/fonts/gen_geist.py` pre-renders printable
   ASCII (0x20..0x7e) into `kernel/src/font_geist.rs` for the fixed grid. Keep
   regenerating that for shell text.

```sh
python3 tools/fonts/gen_geist.py   # -> kernel/src/font_geist.rs
```

2. **Browser (and future proportional UI)** — the same TTF is **embedded and
   rasterized at runtime** via `fontdue` in `kernel/src/font_ttf.rs`
   (`include_bytes!` of this file). Proportional advances + AA coverage, not
   monospaced cell splitting. Other faces: `font_ttf::load_bytes(ttf_bytes, name)`.

## The chat model (fetched, gitignored)

`model.gguf` — the GGUF the kernel loads as a boot module for local inference.
Not committed (hundreds of MB to GB). Fetch one:

```sh
xtask/fetch-model.sh                 # default GGUF
xtask/fetch-model.sh qwen3.5-0.8b    # the compact model (Q8_0, ~812 MB)
```

`cargo xtask run`/`image` bundle whatever `assets/model.gguf` holds (or
`-model <preset|path>` selects another file); an absent model builds a
model-less image, and `/model load <file.gguf>` can load one off a mounted
volume at runtime instead.

## Voice models (fetched, gitignored)

`voice/` — the ONNX voice models the `/voice` pipeline uses: silero-vad
(embedded in the kernel), parakeet-ctc int8 STT (~131 MB) and KittenTTS
(~78 MB). Download with:

```sh
cargo xtask voice-assets             # -> assets/voice/
```

Images bundle them on the ESP so `find_on_disks` auto-loads them at first use.

## Note: agent app assets live elsewhere

Built-in agent packages (SOULs, manifests, and their `tools.wasm` modules)
live under [`agents/`](../agents/), with the wasm sources in
`tools/<name>-wasm/` — see the Apps bullet in [CLAUDE.md](../CLAUDE.md).
