#!/usr/bin/env python3
"""Convert omarchy theme palettes into ChittiOS theme JSON.

    tools/import_omarchy_themes.py <path-to-omarchy-checkout> [--out assets/themes]

omarchy (https://github.com/basecamp/omarchy, MIT) ships each theme as a flat,
*semantic* `colors.toml` -- `background`, `foreground`, `accent`, `selection`,
plus the eight ANSI hues. That is exactly the vocabulary a terminal UI needs, so
the conversion is a rename rather than an interpretation, which is why this is a
table below and not a colour-science exercise.

Two things are deliberately NOT imported:

* **The wallpapers.** 108 MB across 22 themes, and a wallpaper's provenance is
  usually not the repo's own -- they are collected images with their own
  licences. Each theme instead gets `gradient:<background>,<darker_background>`,
  which our compositor renders natively and which is derived purely from the
  palette we did import.
* **The per-app configs** (`neovim.lua`, `vscode.json`, `icons.theme`,
  `keyboard.rgb`). They configure software this OS does not run.

Names are prefixed `omarchy-` so an import cannot silently overwrite a
hand-tuned theme of ours -- both projects ship a `nord`, and they are not the
same file.
"""

import argparse
import json
import pathlib
import re
import sys

# omarchy key -> our key, per section. A missing source key falls back to the
# next candidate in the tuple, then to the theme's own default: the palettes are
# consistent but not identical (only some define `orange`, `brown`, or the
# hyprland border colours), and a theme that omits one should still import.
PALETTE = {
    "accent":           ("accent",),
    "logo":             ("accent",),
    "logo_node":        ("bright_foreground", "foreground"),
    "screen_bg":        ("background",),
    "chat_bg":          ("lighter_background", "background"),
    "logs_bg":          ("darker_background", "dark_background", "background"),
    "chat_fg":          ("foreground",),
    "logs_fg":          ("light_foreground", "foreground"),
    "border_dim":       ("muted", "selection"),
    "title_active":     ("accent",),
    "title_dim":        ("dark_foreground", "muted"),
    "sep_dim":          ("lighter_background", "selection"),
    "status_bg":        ("lighter_background", "dark_background"),
    "status_fg":        ("foreground",),
    "editor_bg":        ("background",),
    "editor_fg":        ("foreground",),
    "editor_lineno":    ("muted", "dark_foreground"),
    "editor_sel":       ("selection", "lighter_background"),
    "composer_bg":      ("lighter_background", "background"),
    "composer_border":  ("muted", "selection"),
    "composer_hint":    ("dark_foreground", "muted"),
}

SYNTAX = {
    "text":    ("foreground",),
    "keyword": ("blue", "accent"),
    "string":  ("green",),
    "number":  ("magenta", "orange"),
    "comment": ("dark_foreground", "muted"),
    "punct":   ("cyan", "accent"),
    "heading": ("bright_cyan", "cyan", "accent"),
    "code":    ("green",),
}

# `key = "value"` and nothing else -- these files are flat, so a real TOML parser
# would be a dependency bought for no benefit.
LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*"([^"]*)"')


def read_colors(path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def pick(src, candidates):
    for c in candidates:
        v = src.get(c)
        if v and v.startswith("#"):
            return v.lower()
    return None


def convert(name, src):
    """One colors.toml -> one ChittiOS theme dict, or None if unusable."""
    if not src.get("background") or not src.get("foreground"):
        return None
    palette = {k: v for k, v in ((k, pick(src, c)) for k, c in PALETTE.items()) if v}
    syntax = {k: v for k, v in ((k, pick(src, c)) for k, c in SYNTAX.items()) if v}
    bg = pick(src, ("background",))
    deep = pick(src, ("darker_background", "dark_background", "background"))
    return {
        "name": f"omarchy-{name}",
        # Left as our default face and automatic scale: a theme should recolour
        # the desktop, not silently resize every glyph on it.
        "font": "Geist Mono",
        "font_scale": 0,
        "wallpaper": f"gradient:{bg},{deep}",
        "opacity": 232,
        "cursor": {
            "fill": bg,
            "outline": pick(src, ("bright_foreground", "foreground")),
        },
        "palette": palette,
        "syntax": syntax,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("omarchy", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("assets/themes"))
    args = ap.parse_args()

    themes_dir = args.omarchy / "themes"
    if not themes_dir.is_dir():
        sys.exit(f"no themes/ under {args.omarchy}")
    args.out.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for d in sorted(p for p in themes_dir.iterdir() if p.is_dir()):
        toml = d / "colors.toml"
        if not toml.is_file():
            skipped.append((d.name, "no colors.toml"))
            continue
        theme = convert(d.name, read_colors(toml))
        if theme is None:
            skipped.append((d.name, "no background/foreground"))
            continue
        dest = args.out / f"omarchy-{d.name}.json"
        dest.write_text(json.dumps(theme, indent=2) + "\n", encoding="utf-8")
        written.append((dest.name, len(theme["palette"]), len(theme["syntax"])))

    for n, p, s in written:
        print(f"  {n:<34} {p} palette, {s} syntax")
    for n, why in skipped:
        print(f"  SKIP {n}: {why}")
    print(f"{len(written)} theme(s) written to {args.out}, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
