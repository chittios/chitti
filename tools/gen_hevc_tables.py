#!/usr/bin/env python3
"""Generate `kernel/src/video/hevc/cabac_tables.rs` from FFmpeg's HEVC decoder.

HEVC's CABAC has **199 contexts**, each with three initialisation bytes (one per
`init_type`), plus a per-syntax-element offset table derived from an X-macro
listing every element and how many contexts it owns. That is ~600 numbers and a
50-entry offset table where a single wrong offset silently decodes one syntax
element against another's probabilities — the decoder does not fail, it produces
a picture that is wrong in a way that looks like a different bug.

So they are parsed, the same rule `gen_vp9_tables.py` and `gen_cabac_tables.py`
follow. The **arithmetic engine itself is not generated**: HEVC shares H.264's
`rangeTabLPS` and state-transition tables, which are already in-tree as
`video::h264::cabac_tables::{RANGE_LPS, TRANS_MPS, TRANS_LPS}`.

Usage:
    python3 tools/gen_hevc_tables.py --fetch
    python3 tools/gen_hevc_tables.py --src <dir with FFmpeg's libavcodec/hevc>

Source: FFmpeg `libavcodec/hevc/cabac.c` (LGPL-2.1 — the *values* are the ITU-T
H.265 specification's own tables; see THIRDPARTY-LICENSES.md).
"""

import argparse
import os
import re
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/"
FILES = [
    "hevc/cabac.c",
    "hevc/hevcdec.h",
    "hevc/data.c",
    "hevc/dsp.c",
    "hevc/pred_template.c",
    "hevc/filter.c",
    "hevc/ps.c",
    "hevc/hevcdec.c",
]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "kernel", "src", "video", "hevc", "cabac_tables.rs")
OUT_TABLES = os.path.join(REPO, "kernel", "src", "video", "hevc", "tables.rs")


def fetch(dest):
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        out = os.path.join(dest, os.path.basename(f))
        if not os.path.exists(out):
            sys.stderr.write("fetch %s\n" % f)
            urllib.request.urlretrieve(BASE + f, out)


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def elements(src):
    """The `CABAC_ELEMS` X-macro: (name, context count) in declaration order."""
    m = re.search(r"#define CABAC_ELEMS\(ELEM\)(.*?)\n\n", src, re.S)
    if not m:
        raise KeyError("CABAC_ELEMS not found")
    out = []
    for name, n in re.findall(r"ELEM\((\w+),\s*(\d+)\)", m.group(1)):
        out.append((name, int(n)))
    if not out:
        raise KeyError("CABAC_ELEMS parsed empty")
    return out


def init_values(src, n_contexts):
    """`init_values[3][HEVC_CONTEXTS]`, with `CNU` resolved."""
    i = src.index("init_values[3][HEVC_CONTEXTS]")
    b = src.index("{", i)
    depth = 0
    for e in range(b, len(src)):
        if src[e] == "{":
            depth += 1
        elif src[e] == "}":
            depth -= 1
            if depth == 0:
                break
    body = src[b + 1 : e]
    cnu = int(re.search(r"#define CNU\s+(\d+)", src).group(1))
    vals = []
    for tok in re.finditer(r"\b(?:CNU|\d+)\b", body):
        t = tok.group(0)
        vals.append(cnu if t == "CNU" else int(t))
    # `HEVC_CONTEXTS` is the *allocated* size (199, sized for the range
    # extensions); the initialiser only fills the contexts the base profile
    # uses, and C zero-fills the rest. So the parsed count must be a whole
    # multiple of three and no larger than the allocation, and the remainder is
    # padded here exactly as the compiler would.
    if len(vals) % 3 != 0 or len(vals) > 3 * n_contexts:
        raise ValueError(
            "parsed %d init values, which is not 3 x (<= %d)" % (len(vals), n_contexts)
        )
    per = len(vals) // 3
    return per, [
        vals[k * per : (k + 1) * per] + [0] * (n_contexts - per) for k in range(3)
    ]


def rust_rows(v, per_line=16):
    rows = []
    for i in range(0, len(v), per_line):
        rows.append("    " + ", ".join(str(x) for x in v[i : i + per_line]) + ",")
    return "\n".join(rows)



def find_array(src, name):
    """The brace-delimited body of the declaration whose text contains `name`.

    Scans braces rather than matching a regex against the body: several of these
    are nested (`transform[32][32]`) and every one of them contains commas,
    comments and line breaks in positions a flat pattern gets wrong.
    """
    i = src.index(name)
    b = src.index("{", i)
    depth = 0
    for e in range(b, len(src)):
        if src[e] == "{":
            depth += 1
        elif src[e] == "}":
            depth -= 1
            if depth == 0:
                return src[b : e + 1]
    raise KeyError("unterminated array for %s" % name)


def flat(body, expect=None, pad=None):
    """Every integer in `body`, in order.

    `pad` fills a C partial initialiser (`{ 0 }` for a 4- or 16-tap filter row
    that is all zeroes) out to the declared length — dropping that row would
    shift every later filter phase, which is a wrong picture rather than an
    error.
    """
    v = [int(x) for x in re.findall(r"-?\d+", body)]
    if pad is not None and len(v) < pad:
        v += [0] * (pad - len(v))
    if expect is not None and len(v) != expect:
        raise ValueError("expected %d values, got %d" % (expect, len(v)))
    return v


def nest(body, rows, cols):
    """A 2-D initialiser as a list of rows, each padded to `cols`.

    Splits on the *inner* braces so a short row (`{ 0 }`) is padded in place
    rather than stealing the next row's values, which is exactly what a flat
    parse of `ff_hevc_qpel_filters` does.
    """
    inner = re.findall(r"\{([^{}]*)\}", body)
    if len(inner) != rows:
        raise ValueError("expected %d rows, got %d" % (rows, len(inner)))
    out = []
    for r in inner:
        v = [int(x) for x in re.findall(r"-?\d+", r)]
        if len(v) > cols:
            raise ValueError("row has %d values, more than %d" % (len(v), cols))
        out.append(v + [0] * (cols - len(v)))
    return out


def rust_1d(name, ty, vals, doc, per_line=16):
    body = rust_rows(vals, per_line)
    return "%spub const %s: [%s; %d] = [\n%s\n];\n\n" % (doc, name, ty, len(vals), body)


def rust_2d(name, ty, rows, doc):
    parts = ["%spub const %s: [[%s; %d]; %d] = [\n" % (doc, name, ty, len(rows[0]), len(rows))]
    for r in rows:
        parts.append("    [%s],\n" % ", ".join(str(x) for x in r))
    parts.append("];\n\n")
    return "".join(parts)


def build_tables(src_dir):
    def read(f):
        return open(os.path.join(src_dir, f), errors="ignore").read()

    data = strip_comments(read("data.c"))
    dsp = strip_comments(read("dsp.c"))
    pred = strip_comments(read("pred_template.c"))
    filt = strip_comments(read("filter.c"))
    ps = strip_comments(read("ps.c"))
    cab = strip_comments(read("cabac.c"))

    out = [TABLES_HEADER]

    for nm, n in (("4x4_x", 16), ("4x4_y", 16), ("8x8_x", 64), ("8x8_y", 64)):
        v = flat(find_array(data, "ff_hevc_diag_scan%s[" % nm), expect=n)
        out.append(
            rust_1d(
                "DIAG_SCAN%s" % nm.upper().replace("X4_", "X4_").replace("X8_", "X8_"),
                "u8",
                v,
                "/// Up-right diagonal scan (H.265 §6.5.3): %s offset of the n-th\n"
                "/// position in a %s block.\n" % (nm[-1], nm[:3]),
            )
        )

    out.append(
        rust_2d(
            "TRANSFORM",
            "i8",
            nest(find_array(dsp, "transform[32][32]"), 32, 32),
            "/// The HEVC integer DCT basis (H.265 §8.6.4.2). Row `k` of the\n"
            "/// leading `N` rows, sub-sampled by `32 / N`, is the size-`N`\n"
            "/// transform — one matrix serves 4/8/16/32, which is why the\n"
            "/// specification defines only this one.\n",
        )
    )
    out.append(
        rust_2d(
            "QPEL_FILTERS",
            "i8",
            [r[:8] for r in nest(find_array(dsp, "ff_hevc_qpel_filters)[4][16]"), 4, 16)],
            "/// 8-tap luma interpolation, indexed by the quarter-pel phase.\n"
            "/// Phase 0 is the integer position and has no filter; FFmpeg stores\n"
            "/// each row twice for its SIMD, so only the first 8 taps are kept.\n",
        )
    )
    out.append(
        rust_2d(
            "EPEL_FILTERS",
            "i8",
            nest(find_array(dsp, "ff_hevc_epel_filters)[8][4]"), 8, 4),
            "/// 4-tap chroma interpolation, indexed by the eighth-pel phase.\n",
        )
    )

    ang = flat(find_array(pred, "intra_pred_angle[]"), expect=33)
    out.append(
        rust_1d(
            "INTRA_PRED_ANGLE",
            "i16",
            ang,
            "/// Angular intra displacement per 32 rows, for the 33 angular\n"
            "/// modes 2..=34\n"
            "/// (index `mode - 2`). Positive runs one way along the reference,\n"
            "/// negative the other, and mode 10/26 (angle 0) is pure\n"
            "/// horizontal/vertical.\n",
        )
    )
    inv = flat(find_array(pred, "inv_angle[]"), expect=15)
    out.append(
        rust_1d(
            "INV_ANGLE",
            "i16",
            inv,
            "/// Reciprocal of a **negative** angle, used to project the far\n"
            "/// reference array into the extension (H.265 §8.4.4.2.6). Indexed\n"
            "/// by `mode - 11`, so it covers only the 15 modes that need it.\n",
        )
    )

    out.append(
        rust_1d(
            "TC_TABLE",
            "u8",
            flat(find_array(filt, "tctable[54]"), expect=54),
            "/// Deblocking `tc'` by QP. 54 entries, not 52: the index can\n"
            "/// exceed MAX_QP by the intra offset.\n",
        )
    )
    out.append(
        rust_1d(
            "BETA_TABLE",
            "u8",
            flat(find_array(filt, "betatable[52]"), expect=52),
            "/// Deblocking `beta'` by QP.\n",
        )
    )
    out.append(
        rust_1d(
            "QP_C",
            "u8",
            flat(find_array(cab, "qp_c[] = "), expect=14),
            "/// Chroma QP mapping for `qPi` in 30..=43 (H.265 table 8-10); below\n"
            "/// 30 the mapping is the identity and above 43 it is `qPi - 6`.\n",
        )
    )
    out.append(
        rust_1d(
            "LEVEL_SCALE",
            "u16",
            flat(find_array(cab, "level_scale[] = "), expect=6),
            "/// Dequantisation multiplier by `qp % 6`.\n",
        )
    )

    # --- residual-coding scans -------------------------------------------
    # These come in matched pairs (position -> coordinate, and its inverse
    # coordinate -> position). A mismatch between a scan and its inverse does
    # not fail: it decodes coefficients into transposed positions inside a
    # 4x4 group, which after the transform is a plausible block.
    for name, cname, n in (
        ("SCAN_1X1", "scan_1x1[1]", 1),
        ("HORIZ_SCAN2X2_X", "horiz_scan2x2_x[4]", 4),
        ("HORIZ_SCAN2X2_Y", "horiz_scan2x2_y[4]", 4),
        ("HORIZ_SCAN4X4_X", "horiz_scan4x4_x[16]", 16),
        ("HORIZ_SCAN4X4_Y", "horiz_scan4x4_y[16]", 16),
        ("DIAG_SCAN2X2_X", "diag_scan2x2_x[4]", 4),
        ("DIAG_SCAN2X2_Y", "diag_scan2x2_y[4]", 4),
    ):
        out.append(
            rust_1d(name, "u8", flat(find_array(cab, cname), expect=n), "/// Scan order.\n")
        )
    for name, cname, rows, cols in (
        ("HORIZ_SCAN8X8_INV", "horiz_scan8x8_inv[8][8]", 8, 8),
        ("DIAG_SCAN2X2_INV", "diag_scan2x2_inv[2][2]", 2, 2),
        ("DIAG_SCAN4X4_INV", "diag_scan4x4_inv[4][4]", 4, 4),
        ("DIAG_SCAN8X8_INV", "diag_scan8x8_inv[8][8]", 8, 8),
    ):
        out.append(
            rust_2d(
                name,
                "u8",
                nest(find_array(cab, cname), rows, cols),
                "/// Inverse scan: `[y][x] -> scan position`.\n",
            )
        )

    # The significance-flag context map: three scan orders x five rows of 16,
    # where row 0 is the 4x4 special case and rows 1..3 are selected by the
    # neighbouring-group pattern (row 4 is the transform-skip case).
    out.append(
        rust_2d(
            "SIG_CTX_IDX_MAP",
            "u8",
            [
                [int(v) for v in re.findall(r"-?\d+", blk)]
                for blk in re.findall(
                    r"\{([^{}]*)\}", find_array(cab, "ctx_idx_map[3][5 * 16]")
                )
            ],
            "/// `significant_coeff_flag` context increments, indexed\n"
            "/// `[scan_idx][row * 16 + scan position]` over five 16-entry rows:\n"
            "/// row 0 is the 4x4 special case, rows 1..=3 are chosen by which\n"
            "/// neighbouring coefficient groups are significant (`prev_sig + 1`),\n"
            "/// and row 4 is the transform-skip case.\n",
        )
    )

    dec = strip_comments(read("hevcdec.c"))
    out.append(
        rust_1d(
            "CHROMA_422_MODE_MAP",
            "u8",
            flat(find_array(dec, "tab_mode_idx[]"), expect=35),
            "/// 4:2:2 remaps the resolved chroma intra mode (H.265 table 8-3),\n"
            "/// because halving only the horizontal dimension changes what a\n"
            "/// given angle means. Recalling this table gives a plausible wrong\n"
            "/// answer -- it looks like a gentle monotone curve either way.\n",
        )
    )

    out.append(
        rust_1d(
            "DEFAULT_SCALING_LIST_INTRA",
            "u8",
            flat(find_array(ps, "default_scaling_list_intra[]"), expect=64),
            "/// Default 8x8 scaling list for intra (H.265 table 7-6), in the\n"
            "/// specification's diagonal scan order, **not** raster.\n",
        )
    )
    out.append(
        rust_1d(
            "DEFAULT_SCALING_LIST_INTER",
            "u8",
            flat(find_array(ps, "default_scaling_list_inter[]"), expect=64),
            "/// Default 8x8 scaling list for inter (H.265 table 7-6).\n",
        )
    )

    return "".join(out)


TABLES_HEADER = '''//! HEVC constant tables — **generated, do not edit**.
//!
//! Produced by `tools/gen_hevc_tables.py` from FFmpeg's `libavcodec/hevc/`
//! (the values are ITU-T H.265\'s own tables; see THIRDPARTY-LICENSES.md).
//!
//! Every one of these is a table where a transcription slip is silent: a wrong
//! interpolation tap blurs by a fraction of a pixel, a wrong scan position
//! moves one coefficient, a wrong `tc` deblocks slightly too hard. None of them
//! fails, and all of them drift a picture away from the reference over a GOP.
//! So they are parsed, the same rule `gen_vp9_tables.py` and
//! `gen_cabac_tables.py` follow.

#![allow(clippy::all)]

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("-o", "--out", default=OUT)
    args = ap.parse_args()
    src_dir = args.src or os.path.join(REPO, "target", "ffmpeg-hevc-src")
    if args.fetch or not os.path.isdir(src_dir):
        fetch(src_dir)

    cabac = strip_comments(open(os.path.join(src_dir, "cabac.c"), errors="ignore").read())
    hdr = open(os.path.join(src_dir, "hevcdec.h"), errors="ignore").read()
    n_ctx = int(re.search(r"#define HEVC_CONTEXTS\s+(\d+)", hdr).group(1))

    elems = elements(cabac)
    used, inits = init_values(cabac, n_ctx)

    # Offsets follow the X-macro's own rule: each element starts where the
    # previous ended, and an element with 0 bins still occupies one slot in the
    # enum (its OFFSET and END coincide).
    # A zero-bin element occupies **no** context slot: the X-macro sets
    # `NAME_END = NAME_OFFSET + NUM_BINS - 1`, so with 0 bins `END` is one
    # *below* `OFFSET` and the next element starts at the same index. Advancing
    # by one for those shifts every later element's contexts.
    offsets = {}
    pos = 0
    for name, n in elems:
        offsets[name] = pos
        pos += n
    if pos != used:
        raise ValueError(
            "element bins sum to %d but %d contexts were initialised" % (pos, used)
        )

    parts = [HEADER % (n_ctx, len(elems))]
    parts.append("pub const HEVC_CONTEXTS: usize = %d;\n" % n_ctx)
    parts.append(
        "/// How many of those the base profile actually initialises; the rest\n"
        "/// belong to the range extensions and are zero.\n"
    )
    parts.append("pub const HEVC_CONTEXTS_USED: usize = %d;\n\n" % used)
    parts.append("/// Per-syntax-element base context index (FFmpeg's `CABAC_ELEMS` order).\n")
    for name, n in elems:
        parts.append("/// %d context%s.\n" % (n, "" if n == 1 else "s"))
        parts.append("pub const %s: usize = %d;\n" % (name, offsets[name]))
    parts.append("\n")
    parts.append(
        "/// `init_values[init_type][ctx]` — the specification's initialisation\n"
        "/// bytes. `init_type` is `2 - slice_type`, flipped by `cabac_init_flag`\n"
        "/// on a non-I slice.\n"
    )
    parts.append("pub const INIT_VALUES: [[u8; %d]; 3] = [\n" % n_ctx)
    for row in inits:
        parts.append("[\n%s\n],\n" % rust_rows(row))
    parts.append("];\n")

    out = "".join(parts)
    with open(args.out, "w") as f:
        f.write(out)
    sys.stderr.write(
        "wrote %s (%d bytes, %d contexts, %d elements)\n"
        % (args.out, len(out), n_ctx, len(elems))
    )

    tables = build_tables(src_dir)
    with open(OUT_TABLES, "w") as f:
        f.write(tables)
    sys.stderr.write("wrote %s (%d bytes)\n" % (OUT_TABLES, len(tables)))


HEADER = '''//! HEVC CABAC context tables — **generated, do not edit**.
//!
//! Produced by `tools/gen_hevc_tables.py` from FFmpeg's `libavcodec/hevc/cabac.c`
//! (the values are ITU-T H.265's own tables; see THIRDPARTY-LICENSES.md).
//! Regenerate with:
//!
//! ```sh
//! python3 tools/gen_hevc_tables.py --fetch
//! ```
//!
//! %d contexts across %d syntax elements. The **offsets** matter as much as the
//! values: a wrong one decodes an element against another element's
//! probabilities, which does not fail — it produces a picture that is wrong in a
//! way that looks like a different bug entirely.
//!
//! The arithmetic engine is *not* here. HEVC shares H.264's `rangeTabLPS` and
//! state-transition tables, which are already in-tree as
//! [`crate::video::h264::cabac_tables`].

#![allow(clippy::all)]

'''

if __name__ == "__main__":
    main()
