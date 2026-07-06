# Chitti OS — design & brand

The visual identity of Chitti OS: the mark, the palette, the type, and how the
console renders them. Everything here is driven by the framebuffer compositor
([`kernel/src/framebuffer.rs`](kernel/src/framebuffer.rs)) and is
**configurable at runtime** from `/configs/core/ui.json` (edit via `/ui` or the
`/open` editor) — no rebuild needed to retheme.

## The mark — "Synapse C"

An open **ring** (the capability) in terracotta with round end-caps, and a filled
**node** (the agent) at its centre in cream. It reads as a *C*, as a
neuron/synapse, and as a terminal caret — the agent as the driver. The console
draws it programmatically (an integer ring test plus one angular gap,
`Screen::draw_logo`), so the same geometry scales from the boot splash to the
status-bar glyph.

Reference SVG (ring `r 17`, stroke `6`, a ~91° opening via `dasharray 80 27`
rotated 35°, centre node `r 5.5`; stroke width ≈ `6/17·r` and node ≈ `0.32·r`
drive the integer form):

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" aria-label="Chitti OS">
  <rect width="64" height="64" rx="14" fill="#1c1917"/>
  <!-- Synapse-C mark: an open ring (capability) with a node (the agent) -->
  <circle cx="32" cy="32" r="17" fill="none" stroke="#cc785c" stroke-width="6" stroke-linecap="round" stroke-dasharray="80 27" transform="rotate(35 32 32)"/>
  <circle cx="32" cy="32" r="5.5" fill="#f5efe6"/>
</svg>
```

On the dark console both the splash and the status-bar mark draw the **ring in
terracotta** (`accent`) and the **node in cream** (`chat_fg`), over the warm-ink
surface (the SVG's `#1c1917` tile).

Where it appears:

- **Boot splash** — the mark, `Chitti OS`, and the tagline *an agentic operating
  system*, centred on the canvas, held ~1.3 s before the shell (`draw_splash`).
- **Status bar** — a small mark at the bottom-left, before the brand text.

## Palette

Terracotta primary `#cc785c`, paired with warm ink `#141413` and cream
`#faf9f5`. The console ships a **dark** theme (`Theme::BRAND_DARK`): terracotta
on warm-ink surfaces, cream text.

Full brand palette:

| Token | Hex | Use |
|---|---|---|
| `primary` | `#cc785c` | accent — active border, caret, brand, logo |
| `primary-active` | `#a9583e` | pressed / active primary |
| `primary-disabled` | `#e6dfd8` | disabled primary |
| `ink` | `#141413` | darkest text / light-bg mark |
| `body` | `#3d3d3a` | body text on light |
| `body-strong` | `#252523` | strong body |
| `muted` | `#6c6a64` | muted text |
| `muted-soft` | `#8e8b82` | softer muted |
| `hairline` | `#e6dfd8` | borders on light |
| `hairline-soft` | `#ebe6df` | softer borders |
| `canvas` | `#faf9f5` | light background |
| `surface-soft` | `#f5f0e8` | light surface |
| `surface-card` | `#efe9de` | card |
| `surface-cream-strong` | `#e8e0d2` | strong cream |
| `surface-dark` | `#181715` | dark background (console `screen_bg`) |
| `surface-dark-elevated` | `#252320` | dark elevated (console `status_bg`) |
| `surface-dark-soft` | `#1f1e1b` | dark soft (console `chat_bg`) |
| `on-primary` | `#ffffff` | text on primary |
| `on-dark` | `#faf9f5` | text on dark (console `chat_fg`) |
| `on-dark-soft` | `#a09d96` | muted text on dark (console `logs_fg`) |
| `accent-teal` | `#5db8a6` | secondary accent |
| `accent-amber` | `#e8a55a` | secondary accent |
| `success` | `#5db872` | success |
| `warning` | `#d4a017` | warning |
| `error` | `#c64545` | error |

## Typography

**Geist Mono** everywhere — the console renders text from a bundled Geist Mono
glyph atlas ([`kernel/src/font_geist.rs`](kernel/src/font_geist.rs),
alpha-blended per pixel). The atlas is ASCII (`0x20`–`0x7e`); the font scales by
integer factor with the panel resolution so type stays legible from 1080p to 4K.

Agent replies are formatted with **ANSI SGR** escape codes (colour + emphasis),
not markdown — the console parses `\x1b[…m` (see `Screen::apply_sgr`).

## Configuring it — `/configs/core/ui.json`

Every colour and the splash toggle live in the UI config, applied live on `/ui
reload`. The `theme` object maps console colour slots to `#rrggbb` strings;
omit any key to keep the brand default.

```json
{
  "chat_pct": 56,
  "font_scale": 0,
  "swap_panes": false,
  "chat_title": "chat",
  "logs_title": "ktrace",
  "status_left": "Chitti OS v${version}",
  "status_right": "${datetime}  ${tz}",
  "tz_offset": 0,
  "splash": true,
  "theme": {
    "accent": "#cc785c",
    "screen_bg": "#181715",
    "chat_bg": "#1f1e1b",
    "logs_bg": "#141311",
    "chat_fg": "#faf9f5",
    "logs_fg": "#a09d96",
    "border_dim": "#3a3733",
    "title_active": "#cc785c",
    "title_dim": "#6c6a64",
    "sep_dim": "#2a2825",
    "status_bg": "#252320",
    "status_fg": "#a09d96",
    "editor_bg": "#1f1e1b",
    "editor_fg": "#faf9f5",
    "editor_lineno": "#6c6a64",
    "editor_sel": "#5a3a2e"
  }
}
```

- **`theme.<slot>`** — any of the colour slots above; a light theme is just a
  different set (e.g. `screen_bg`/`chat_bg` = `#faf9f5`/`#f5f0e8`, `*_fg` = ink).
- **`splash`** — show the boot splash (default `true`).
- **`status_left` / `status_right`** — templates with `${version}`,
  `${datetime}`, `${tz}`, `${model}`, `${arch}`, `${uptime}`, `${brand}`.
- Layout (`chat_pct`, `font_scale`, `swap_panes`, titles) and shortcuts
  (`/configs/core/shortcuts.json`) are configured the same way.

Colour slots map 1:1 to `framebuffer::Theme` fields; `theme_from_pairs` applies
the config over `Theme::BRAND_DARK`, and malformed/unknown entries keep the
brand value.
