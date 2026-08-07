#!/usr/bin/env python3
"""Transpile libvpx's 1-D inverse transform kernels into Rust.

`idct32_c` alone is ~380 lines of `step1[17] = WRAPLOW(step2[16] + step2[23]);`
— utterly regular and utterly unreviewable by eye. Hand-porting it is the same
class of mistake as hand-transcribing a probability table: one transposed index
gives a transform that is *nearly* right, so the picture is nearly right, and
the difference only shows as drift once inter frames start predicting from it.

So the kernels are translated mechanically. The C is a strict subset — scalar
declarations, array declarations, and assignments built from `+ - *`, array
indexing, casts, `WRAPLOW()` and `dct_const_round_shift()` — which makes a
targeted transpiler tractable and, unlike a hand port, *checkable*: the
generated Rust is diffed against the C statement-for-statement by construction,
and the arithmetic is then verified numerically against libvpx by
`tools/videodiff`.

Usage:
    python3 tools/gen_vp9_idct.py --src <libvpx dir>
"""

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "kernel", "src", "video", "vp9", "idct_kernels.rs")

# The 1-D kernels VP9 needs, with the length of their input/output vectors.
KERNELS = [
    ("idct4_c", 4),
    ("idct8_c", 8),
    ("idct16_c", 16),
    ("idct32_c", 32),
    ("iadst4_c", 4),
    ("iadst8_c", 8),
    ("iadst16_c", 16),
]


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def body_of(text, fn):
    m = re.search(r"\bvoid\s+%s\s*\([^)]*\)\s*\{" % re.escape(fn), text)
    if not m:
        raise KeyError("function %s not found" % fn)
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[m.end():i - 1]


def cospi_consts(txfm_common):
    out = {}
    for m in re.finditer(r"static const tran_coef_t (\w+)\s*=\s*(-?\d+);", txfm_common):
        out[m.group(1)] = int(m.group(2))
    return out


def translate(body, n, consts):
    """Translate one kernel body to Rust statements.

    Everything is computed in `i64` and narrowed only where the C narrows
    (`WRAPLOW`, which is a *deliberate* 16-bit wrap that real streams never
    reach but corrupt ones do — dropping it turns a corrupt frame into a
    panic or a wildly wrong picture instead of libvpx's defined behaviour).
    """
    lines = []
    decls = []
    # The `if (!(x0|x1|...)) { memset(); return; }` early-out spans two
    # `;`-separated pieces, so the `return` that follows the guard has already
    # been emitted inside the block. Emitting it again at top level would make
    # the kernel return unconditionally — which produces an all-zero transform,
    # i.e. a grey picture rather than a compile error.
    skip_next_return = False
    for stmt in [s.strip() for s in body.split(";")]:
        if not stmt:
            continue
        stmt = " ".join(stmt.split())
        # Splitting on `;` leaves a block's closing brace glued to the front of
        # the next statement (`} s0 = sinpi_1_9 * x0`). The brace is structural
        # and already emitted by the `if` handler, so drop it here.
        while stmt.startswith("}"):
            stmt = stmt[1:].strip()
        if not stmt:
            continue

        # `memset(output, 0, N * sizeof(*output)); return;` — the all-zero
        # early-out in the iadst kernels.
        if stmt.startswith("memset("):
            lines.append("for v in output.iter_mut() { *v = 0; }")
            continue
        if stmt == "return":
            if skip_next_return:
                skip_next_return = False
                continue
            lines.append("return;")
            continue

        # Declarations.
        m = re.fullmatch(r"(?:int16_t|tran_low_t|tran_high_t|int)\s+(.+)", stmt)
        if m and "=" not in m.group(1):
            for item in m.group(1).split(","):
                item = item.strip()
                a = re.fullmatch(r"(\w+)\[(\d+)\]", item)
                if a:
                    decls.append("let mut %s = [0i64; %s];" % (a.group(1), a.group(2)))
                elif re.fullmatch(r"\w+", item):
                    decls.append("let mut %s: i64 = 0;" % item)
                else:
                    raise ValueError("unhandled declaration %r" % item)
            continue
        # Declaration with an initialiser: `tran_high_t x0 = input[7]`.
        m = re.fullmatch(r"(?:int16_t|tran_low_t|tran_high_t|int)\s+(\w+)\s*=\s*(.+)", stmt)
        if m:
            decls.append("let mut %s: i64 = 0;" % m.group(1))
            lines.append("%s = %s;" % (m.group(1), expr(m.group(2), consts)))
            continue

        # `if (!(x0 | x1 | ...)) {` guards are handled by the caller emitting
        # the early-out; the transpiler refuses anything else with control flow
        # rather than guessing.
        if stmt.startswith("if ") or stmt.startswith("if("):
            m = re.fullmatch(r"if \(!\(([\w \|]+)\)\) \{ (.*)", stmt)
            if not m:
                raise ValueError("unhandled control flow %r" % stmt)
            vars_ = [v.strip() for v in m.group(1).split("|")]
            # Parenthesised: in Rust `==` binds tighter than `|`, so
            # `x0 | x1 == 0` is `x0 | (x1 == 0)` — a type error here, but the
            # same shape in a bitwise context would compile and be wrong.
            lines.append("if (%s) == 0 {" % " | ".join(vars_))
            rest = m.group(2).strip()
            if rest.startswith("memset("):
                lines.append("    for v in output.iter_mut() { *v = 0; }")
            lines.append("    return;")
            lines.append("}")
            skip_next_return = True
            continue
        if stmt == "}":
            continue

        # Plain assignment.
        m = re.fullmatch(r"([\w\[\]]+)\s*=\s*(.+)", stmt)
        if not m:
            raise ValueError("unhandled statement %r" % stmt)
        lines.append("%s = %s;" % (lhs(m.group(1)), expr(m.group(2), consts)))
    return decls, lines


def lhs(s):
    return s


def expr(e, consts):
    e = e.strip()
    # Casts are dropped: everything is i64 here, and the only narrowing that
    # matters is WRAPLOW's, which is explicit.
    e = re.sub(r"\((?:int16_t|tran_low_t|tran_high_t|int|int32_t)\)\s*", "", e)
    e = re.sub(r"\bWRAPLOW\b", "wraplow", e)
    e = re.sub(r"\bdct_const_round_shift\b", "dct_round", e)
    e = re.sub(r"\bcheck_range\b", "", e)
    # cospi_/sinpi_ constants → their Rust upper-case names.
    def const(m):
        name = m.group(0)
        if name not in consts:
            raise KeyError("unknown constant %r" % name)
        return name.upper()
    e = re.sub(r"\b(?:cospi|sinpi)_\d+_\d+\b", const, e)
    # `input[i]` reads are i64 already (the caller passes an i64 slice).
    if re.search(r"\b(?!wraplow|dct_round)[a-z_]\w*\s*\(", e):
        bad = re.findall(r"\b([a-z_]\w*)\s*\(", e)
        bad = [b for b in bad if b not in ("wraplow", "dct_round")]
        if bad:
            raise ValueError("unexpected call %r in %r" % (bad, e))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("-o", "--out", default=OUT)
    args = ap.parse_args()

    inv = strip_comments(open(os.path.join(args.src, "inv_txfm.c"), errors="ignore").read())
    consts = cospi_consts(open(os.path.join(args.src, "txfm_common.h"), errors="ignore").read())

    parts = [HEADER]
    for name, _ in sorted(consts.items()):
        pass
    for name in sorted(consts, key=lambda k: (k.split("_")[0], int(k.split("_")[1]))):
        parts.append("pub const %s: i64 = %d;\n" % (name.upper(), consts[name]))
    parts.append("\n")

    for fn, n in KERNELS:
        decls, lines = translate(body_of(inv, fn), n, consts)
        rname = fn[:-2]  # drop the `_c`
        parts.append("/// libvpx `%s` — %d-point, transpiled by `tools/gen_vp9_idct.py`.\n" % (fn, n))
        parts.append("#[allow(clippy::needless_range_loop)]\n")
        parts.append("pub fn %s(input: &[i64], output: &mut [i64]) {\n" % rname)
        for d in decls:
            parts.append("    %s\n" % d)
        for l in lines:
            parts.append("    %s\n" % l)
        parts.append("}\n\n")

    out = "".join(parts)
    with open(args.out, "w") as f:
        f.write(out)
    sys.stderr.write("wrote %s (%d bytes, %d kernels)\n" % (args.out, len(out), len(KERNELS)))


HEADER = '''//! VP9 one-dimensional inverse transform kernels — **generated, do not edit**.
//!
//! Transpiled from libvpx's `vpx_dsp/inv_txfm.c` by `tools/gen_vp9_idct.py`
//! (BSD; see THIRDPARTY-LICENSES.md). Regenerate with:
//!
//! ```sh
//! python3 tools/gen_vp9_idct.py --src target/libvpx-src
//! ```
//!
//! `idct32` is ~380 statements of `step1[17] = WRAPLOW(step2[16] + step2[23])`.
//! One transposed index there yields a transform that is *nearly* right, so the
//! picture is nearly right, and the error only becomes visible as drift once
//! inter frames predict from it — which is why these are machine-translated
//! rather than hand-ported, and then checked numerically against libvpx.
//!
//! Everything is computed in `i64`; the only narrowing is [`wraplow`], which is
//! libvpx's deliberate 16-bit wrap. A conforming stream never reaches it, but a
//! corrupt one does, and dropping it would turn defined behaviour into a wildly
//! wrong picture (or, with overflow checks on, a panic in the kernel).

#![allow(clippy::all)]

/// libvpx `WRAPLOW`: wrap to 16 bits. Not a clamp — the wrap is what the
/// reference does, and a clamp would diverge on corrupt input.
#[inline(always)]
pub fn wraplow(x: i64) -> i64 {
    ((x as i32) << 16 >> 16) as i64
}

/// libvpx `dct_const_round_shift`: round to nearest, shift by `DCT_CONST_BITS`.
#[inline(always)]
pub fn dct_round(x: i64) -> i64 {
    (x + (1 << (DCT_CONST_BITS - 1))) >> DCT_CONST_BITS
}

/// `ROUND_POWER_OF_TWO(value, n)`.
#[inline(always)]
pub fn round_pow2(v: i64, n: u32) -> i64 {
    (v + (1 << (n - 1))) >> n
}

pub const DCT_CONST_BITS: u32 = 14;
/// `UNIT_QUANT_SHIFT` — the lossless Walsh-Hadamard path's input shift.
pub const UNIT_QUANT_SHIFT: u32 = 2;

'''

if __name__ == "__main__":
    main()
