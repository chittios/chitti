# Fonts

`GeistMono-Regular.ttf` — [Geist Mono](https://vercel.com/font) by Vercel
(in collaboration with basement.studio), licensed under the SIL Open Font
License 1.1 (see `GeistMono-OFL.txt`).

The kernel is `no_std` and cannot rasterize a TrueType font at runtime, so
`tools/fonts/gen_geist.py` pre-renders printable ASCII (0x20..0x7e) at a fixed
cell size into a grayscale coverage atlas emitted as `kernel/src/font_geist.rs`.
The framebuffer compositor (`kernel/src/framebuffer.rs`) alpha-blends those
coverage bytes to draw antialiased text. Regenerate with:

```
python3 tools/fonts/gen_geist.py   # -> kernel/src/font_geist.rs
```
