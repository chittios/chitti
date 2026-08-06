# `csslayout` — host harness for the kernel's CSS + layout engine

Lays a page out **natively, in milliseconds**, using the kernel's own modules —
`kernel/src/browser/{html,css,elements,flex,layout,canvas,form,url}.rs` are
mounted with `#[path]`, not copied. Same pattern as `tools/h264diff`,
`tools/pngbench` and `tools/cortexdiff`: one implementation, two build targets,
so a fix verified here is the fix that ships and there is no second engine to
drift.

```sh
cargo build --release --manifest-path tools/csslayout/Cargo.toml

./tools/csslayout/target/release/chitti-csslayout page.html
./tools/csslayout/target/release/chitti-csslayout page.html --rects        # boxes only
./tools/csslayout/target/release/chitti-csslayout page.html --runs         # text only
./tools/csslayout/target/release/chitti-csslayout page.html --width 1024 --height 768
```

It prints what layout produced: every background/border box with its position,
size, colour, radius and shadow blur; every text run with its colour and size;
and every form control. `<link rel=stylesheet>` is resolved against the page's
directory and read off disk, and sheets are applied in exact document order.

## Why it exists

"The card has no background", "the badge renders as bare text", "the row
collapsed" are all decided in layout, and finding them by booting QEMU costs two
minutes per attempt. Every CSS fix behind the React/Tailwind and shadcn/ui
sample pages was found with this in a few seconds each:

- `rgb(255, 255, 255)` shredded by a whitespace split, so no Tailwind colour
  applied
- `hsl(var(--card))` unsupported, so no shadcn colour applied
- `28rem` parsed as `em` (it ends in "em"), so every Tailwind size was dropped
- `0 0 #0000` read as opaque black, painting a solid rectangle over every card
- a block's background painted **over** its own children
- `display: inline-flex` sized as a block, making every button a full-width bar

## What it does not do

- **It does not paint.** Compositing, blur and glyph rasterisation are the
  kernel's; this reports the geometry that feeds them.
- **It does not run JavaScript.** For a page whose DOM is built by a script, run
  the bundle through `tools/webcompat/just_runner --browser` first and lay out
  the DOM it dumps.
- **Font metrics are approximated** (`main.rs`'s `font_ttf` stub). Anything that
  turns on exact glyph advances must be confirmed in the kernel — a unit test in
  `layout.rs` — not here.
