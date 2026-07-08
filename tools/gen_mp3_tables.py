#!/usr/bin/env python3
"""Generate kernel/src/audio/mp3_tables.rs from minimp3.h (CC0 / public
domain, https://github.com/lieff/minimp3) — the numeric tables of the MPEG
Layer III decoder, extracted verbatim so the Rust port never retypes them.

Usage: python3 tools/gen_mp3_tables.py path/to/minimp3.h > kernel/src/audio/mp3_tables.rs
"""
import re
import sys


def grab(src, name):
    """The initializer list of `static const <ty> name[...] = { ... };`."""
    m = re.search(rf"static const \w+ {re.escape(name)}\s*\[[^=]*=\s*\{{(.*?)\}};", src, re.S)
    if not m:
        raise SystemExit(f"table {name} not found")
    body = m.group(1)
    # Drop comments and the DQ() macro layer is handled separately.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return body


def nums(body):
    """Flatten an initializer into a list of numeric literal strings."""
    toks = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?f?", body)
    return [t.rstrip("f") for t in toks]


def flat(name, rust_name, ty, src, expect=None):
    vals = nums(grab(src, name))
    if expect is not None and len(vals) != expect:
        raise SystemExit(f"{name}: expected {expect} values, got {len(vals)}")
    if ty == "f32":
        body = ", ".join(v if ("." in v or "e" in v or "E" in v) else f"{v}.0" for v in vals)
    else:
        body = ", ".join(vals)
    return f"pub const {rust_name}: [{ty}; {len(vals)}] = [{body}];\n"


def grid(name, rust_name, ty, rows, cols, src):
    vals = nums(grab(src, name))
    if len(vals) != rows * cols:
        raise SystemExit(f"{name}: expected {rows}x{cols}, got {len(vals)}")
    out = [f"pub const {rust_name}: [[{ty}; {cols}]; {rows}] = ["]
    for r in range(rows):
        out.append("    [" + ", ".join(vals[r * cols:(r + 1) * cols]) + "],")
    out.append("];\n")
    return "\n".join(out)


def fgrid(name, rust_name, rows, cols, src):
    vals = nums(grab(src, name))
    if len(vals) != rows * cols:
        raise SystemExit(f"{name}: expected {rows}x{cols}, got {len(vals)}")
    out = [f"pub const {rust_name}: [[f32; {cols}]; {rows}] = ["]
    for r in range(rows):
        row = vals[r * cols:(r + 1) * cols]
        row = [v if ("." in v or "e" in v) else f"{v}.0" for v in row]
        out.append("    [" + ", ".join(row) + "],")
    out.append("];\n")
    return "\n".join(out)


def main():
    src = open(sys.argv[1]).read()
    print("//! MPEG Layer III decoder tables, generated **verbatim** from")
    print("//! minimp3.h (CC0/public domain, https://github.com/lieff/minimp3)")
    print("//! by `tools/gen_mp3_tables.py`. Do not edit by hand — regenerate.")
    print("#![allow(clippy::excessive_precision)]")
    print()
    print(grid("g_scf_long", "SCF_LONG", "u8", 8, 23, src))
    print(grid("g_scf_short", "SCF_SHORT", "u8", 8, 40, src))
    # g_scf_mixed rows are ragged in C (37/40/37/... entries); pad with 0 to 40.
    body = grab(src, "g_scf_mixed")
    rows = [nums(r) for r in re.findall(r"\{([^{}]*)\}", body)]
    assert len(rows) == 8, len(rows)
    print("pub const SCF_MIXED: [[u8; 40]; 8] = [")
    for r in rows:
        r = r + ["0"] * (40 - len(r))
        print("    [" + ", ".join(r) + "],")
    print("];\n")
    print(grid("g_scf_partitions", "SCF_PARTITIONS", "u8", 3, 28, src))
    print(flat("g_scfc_decode", "SCFC_DECODE", "u8", src, 16))
    print(flat("g_mod", "SCF_MOD", "u8", src, 24))
    print(flat("g_preamp", "PREAMP", "u8", src, 10))
    print(flat("g_expfrac", "EXPFRAC", "f32", src, 4))
    print(flat("g_pow43", "POW43", "f32", src, 145))
    print(flat("tabs", "HUFF_TABS", "i16", src))
    print(flat("tab32", "HUFF_TAB32", "u8", src, 28))
    print(flat("tab33", "HUFF_TAB33", "u8", src, 16))
    print(flat("tabindex", "HUFF_TABINDEX", "i16", src, 32))
    print(flat("g_linbits", "HUFF_LINBITS", "u8", src, 32))
    print(flat("g_pan", "PAN", "f32", src, 14))
    print(fgrid("g_aa", "AA", 2, 8, src))
    print(flat("g_twid9", "TWID9", "f32", src, 18))
    print(flat("g_twid3", "TWID3", "f32", src, 6))
    print(fgrid("g_mdct_window", "MDCT_WINDOW", 2, 18, src))
    print(flat("g_sec", "DCT2_SEC", "f32", src, 24))
    print(flat("g_win", "SYNTH_WIN", "f32", src, 240))
    # Bitrate half-rate table: [2][3][15].
    vals = nums(grab(src, "halfrate"))
    assert len(vals) == 90, len(vals)
    print("pub const HALFRATE: [[[u8; 15]; 3]; 2] = [")
    for a in range(2):
        print("    [")
        for b in range(3):
            row = vals[(a * 3 + b) * 15:(a * 3 + b) * 15 + 15]
            print("        [" + ", ".join(row) + "],")
        print("    ],")
    print("];")


if __name__ == "__main__":
    main()
