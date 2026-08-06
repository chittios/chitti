# CSS engine hardening (Fix 11)

Companion to the production CSS plan for `/browse`. Extends the cascade beyond
the earlier “parse popular props into `ComputedStyle`” layer.

## Shipped

1. **`@media` width queries** — `Stylesheet::parse_with_viewport(css, vw)` keeps
   `(min|max)-width` / `screen` / `all`; drops `print`. Unknown features fail-open.
2. **`calc()`** — `parse_px` / `parse_px_rel` for `Npx ± Mpx|em|%` (CB width for `%`).
3. **`var()`** — nested fallbacks + cycle → empty; `custom_props` inherit parent→child.
4. **Selectors** — real `>` child combinator; `:nth-child(odd|even|An+B)`;
   `[href]` / `[type=…]` / `[class~=…]`; `::before` / `::after` kept as rules.
5. **`clear`** — advances past float exclusions before block/float placement.
6. **Generated content** — `content: "…"` on `::before`/`::after` emits text runs.
7. **`object-position`** — wired into `blit_image_fit` with cover/contain alignment.
8. **Tables** — `rowspan` occupancy + `table-layout: fixed` equal columns.
9. **Background `url()`** — end-to-end path unchanged (session `bg_pixels` → paint);
   relative URLs resolve against the document base in the browse host.

## Design-system pass (Tailwind 3 / shadcn-ui)

Found with `tools/csslayout` (host harness — mounts the kernel's own
`css.rs`/`layout.rs`, milliseconds per page) while bringing up the React and
shadcn sample pages. Every one of these dropped a declaration *silently*:

10. **Functional colours survive shorthand tokenizing** — `css::value_tokens`
    splits a value on whitespace at paren depth 0, so `rgb(255, 255, 255)` and
    `rgb(248 250 252 / var(--tw-bg-opacity, 1))` stay one token. A plain
    `split_whitespace` found no colour in either, so *no* Tailwind background,
    border or text colour applied while `#ffffff` in the same slot worked.
11. **`hsl()` / `hsla()`** — shadcn/ui defines its whole palette as HSL triples
    in custom properties and reads them back as `hsl(var(--card))`. Without it a
    shadcn page has no colour at all.
12. **A fully transparent colour is `None`, not black** — `#0000`, `#RRGGBB00`
    and `rgba(…, 0)`. Tailwind's shadow chain is literally `0 0 #0000`, and
    reading it as opaque black painted a solid rectangle over every card.
13. **`box-shadow` is a list** — `parse_box_shadow` walks the comma-separated
    entries and takes the first with a visible colour. Tailwind puts the real
    shadow **last**, behind two `0 0 #0000` placeholders.
14. **`rem` before `em`** — `"28rem"` ends in "em", so the em arm consumed it and
    failed on the leftover `"28r"`. Every Tailwind size is in rem, so widths,
    padding and the type scale were all dropped. Also `vh`/`vw`/`vmin`/`vmax`
    (against the layout's viewport), `pt`, `ch`, `ex`.
15. **Inline backgrounds and borders** — an inline element now paints its own
    box (one per line box, with padding and radius) and advances the cursor by
    its horizontal padding. A badge — `<span class="rounded-full border
    bg-emerald-50 px-2">` — used to render as bare text.
16. **A block's background paints behind its children** — decoration is
    *inserted* at the fragment start rather than pushed after the children, since
    a block's height (and so its background) is only known once its children are
    laid out. A white card used to erase everything inside it.
17. **`display: inline-flex`** is its own `DisplayMode` and shrink-wraps to its
    content, both as a flex container and (the common case) as a text-only box
    with no element children. As block-level flex, every shadcn button and badge
    was a full-width bar.

Kernel unit coverage: `functional_colors_survive_shorthand_tokenizing`,
`hsl_colors_parse`, `a_fully_transparent_color_is_none_not_black`,
`rem_is_not_em_and_viewport_units_resolve`, `inline_flex_is_its_own_display_mode`,
`box_shadow_skips_transparent_placeholder_entries`,
`an_inline_element_paints_its_background_and_border`,
`a_block_background_paints_behind_its_children`,
`inline_flex_shrinks_to_its_content`,
`a_tailwind_card_gets_its_width_padding_and_colours`,
`a_shadcn_shaped_section_paints_its_components`.

## Explicit non-goals

- Timed animations / transitions timeline
- `:hover` hit-testing restyle
- CSS Grid Level 2 (`subgrid`, `grid-template-areas`)
- `position: sticky` compositing
- Full css3test.com scoreboard / Chrome parity

## Fixtures

- `tools/webcompat/fixtures/css/hn-like.html` — HN colours / table shape
- `tools/webcompat/fixtures/css/modern.html` — flex, grid, `@media`, `calc`, `var`, `::before`
- `/samples/html/react-tw.html` — a real Vite/React 18 + Tailwind 3 bundle
- `/samples/html/shadcn.html` — 16 shadcn/ui components (see `tools/react-shadcn`)

Iterate on any of them with the host harness rather than a QEMU boot:

```sh
cargo build --release --manifest-path tools/csslayout/Cargo.toml
./tools/csslayout/target/release/chitti-csslayout assets/samples/html/css-full.html --rects
```

Kernel unit coverage: `media_calc_var_and_selectors`, `clear_both_drops_below_float`,
`before_pseudo_emits_generated_content`, `hn_like_link_and_modern_css_fixture`.
