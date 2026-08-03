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

## WiFi dongle firmware (extracted, gitignored)

`wifi/brcm/` — Apple FullMAC firmware for the on-board radio (j473 Mac mini M2
= BCM4388 / `apple,miyake`). Extracted from this Mac's
`/usr/share/firmware/wifi/C-4388__s-*/miyake.trx` into the Asahi naming layout:

```sh
make wifi-assets                 # or: cargo xtask wifi-assets
# -> assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.bin
# -> assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.txt   (NVRAM, antenna X3)
```

When present, the **kernel embeds** the `.bin` (so bare `make m1n1` boots can
`/wifi load` with no disk), and `image` / ESP builds also copy `brcm/` onto the
FAT volume for QEMU/VBox/`find_on_disks`. Not redistributable as a URL — comes
from the local macOS firmware tree (same source Asahi's fwextract uses).

## Note: agent app assets live elsewhere

Built-in agent packages (SOULs, manifests, and their `tools.wasm` modules)
live under [`agents/`](../agents/), with the wasm sources in
`tools/<name>-wasm/` — see the Apps bullet in [CLAUDE.md](../CLAUDE.md).
## Font Awesome Free Solid

`FontAwesome6Free-Solid-900.otf` (SIL OFL 1.1 / icons CC BY 4.0) — system UI
icons. Registered **first** in the kernel TTF fallback chain so status-bar,
agents browser, and package-UI chrome resolve Private-Use-Area codepoints
(`kernel/src/icons.rs`, mirrored in `tools/apps-wasm/src/fa.rs`) to real
glyphs. See [THIRDPARTY-LICENSES.md](../THIRDPARTY-LICENSES.md).
