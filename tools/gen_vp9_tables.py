#!/usr/bin/env python3
"""Generate `kernel/src/video/vp9/tables.rs` from the libvpx sources.

VP9's decoder is mostly tables: ~9000 probability, scan, neighbour, filter and
dequantiser values. Every one of them is a number that produces a *plausible*
picture when wrong — a mistyped coefficient probability does not fail, it
decodes a slightly different frame, and the error then propagates through every
inter frame that predicts from it. So none of them are hand-transcribed; this
script parses them out of libvpx and emits Rust, the same rule
`tools/gen_cabac_tables.py` and `tools/gen_iq_tables.py` follow for H.264 and
the i-quants.

Usage:
    python3 tools/gen_vp9_tables.py --fetch          # download libvpx sources
    python3 tools/gen_vp9_tables.py --src <dir>      # use an existing checkout

Sources (BSD-licensed, see THIRDPARTY-LICENSES.md):
    vp9/common/vp9_entropy.c      coefficient probabilities, Pareto model
    vp9/common/vp9_entropymode.c  mode/partition/reference probabilities, trees
    vp9/common/vp9_entropymv.c    motion-vector probabilities
    vp9/common/vp9_scan.c         scan orders and coefficient-context neighbours
    vp9/common/vp9_filter.c       sub-pel interpolation kernels
    vp9/common/vp9_quant_common.c dequantiser lookups
    vp9/common/vp9_common_data.c  block-geometry lookups
"""

import argparse
import os
import re
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/webmproject/libvpx/main/"
FILES = [
    "vp9/common/vp9_entropy.c",
    "vp9/common/vp9_entropy.h",
    "vp9/common/vp9_entropymode.c",
    "vp9/common/vp9_entropymode.h",
    "vp9/common/vp9_entropymv.c",
    "vp9/common/vp9_entropymv.h",
    "vp9/common/vp9_scan.c",
    "vp9/common/vp9_filter.c",
    "vp9/common/vp9_filter.h",
    "vp9/common/vp9_quant_common.c",
    "vp9/common/vp9_common_data.c",
    "vp9/common/vp9_enums.h",
    "vp9/common/vp9_blockd.h",
    "vp9/decoder/vp9_dsubexp.c",
    "vp9/common/vp9_mvref_common.h",
    "vp9/common/vp9_loopfilter.c",
]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "kernel", "src", "video", "vp9", "tables.rs")


def fetch(dest):
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        out = os.path.join(dest, os.path.basename(f))
        if os.path.exists(out):
            continue
        sys.stderr.write("fetch %s\n" % f)
        urllib.request.urlretrieve(BASE + f, out)


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


# --- symbol table -----------------------------------------------------------
#
# The arrays are indexed and sometimes *valued* by enum constants
# (`subsize_lookup` holds `BLOCK_8X8`, `max_txsize_lookup` holds `TX_16X16`), so
# a numeric-only parser silently drops them. Enums are read from the headers
# rather than assumed.

def symbols(src):
    text = ""
    for name in os.listdir(src):
        if name.endswith((".h", ".c")):
            text += strip_comments(open(os.path.join(src, name), errors="ignore").read())
    sym = {}
    for m in re.finditer(r"#define\s+([A-Za-z_]\w*)\s+\(?(-?\d+)\)?\s*$", text, re.M):
        sym[m.group(1)] = int(m.group(2))
    for body in re.findall(r"\benum\s*(?:\w+\s*)?\{(.*?)\}", text, re.S):
        nxt = 0
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            m = re.match(r"^([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", item, re.S)
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if val is not None:
                val = val.strip()
                if re.fullmatch(r"-?\d+", val):
                    nxt = int(val)
                elif val in sym:
                    nxt = sym[val]
                else:
                    try:
                        nxt = int(eval(val, {"__builtins__": {}}, dict(sym)))
                    except Exception:
                        continue
            sym[name] = nxt
            nxt += 1
    # Object-like defines whose body is an expression over other symbols
    # (`#define BLOCK_INVALID BLOCK_SIZES`). Iterated to a fixed point because
    # they chain and the headers do not declare them in dependency order.
    # `#define TX_4X4 ((TX_SIZE)0)` — libvpx spells several of these as a cast
    # to a typedef'd uint8_t rather than an enum, so the cast is stripped before
    # evaluating. Only `(Identifier)` immediately before a value is removed, so
    # ordinary parenthesised arithmetic survives.
    cast = re.compile(r"\(\s*(?:TX_SIZE|BLOCK_SIZE|PARTITION_TYPE|uint8_t|int|unsigned)\s*\)")
    pending = {}
    for m in re.finditer(r"#define\s+([A-Za-z_]\w*)\s+([^\n(][^\n]*|\([^\n]*)$", text, re.M):
        name, expr = m.group(1), cast.sub("", m.group(2)).strip()
        if name not in sym and "(" not in name:
            pending[name] = expr
    for _ in range(8):
        progress = False
        for name, expr in list(pending.items()):
            try:
                sym[name] = int(eval(expr, {"__builtins__": {}}, dict(sym)))
                del pending[name]
                progress = True
            except Exception:
                continue
        if not progress:
            break
    return sym


def find_array(text, name):
    """Return the brace body of `<name>[dims] = { ... };`.

    Handles libvpx's three declaration shapes — plain, `DECLARE_ALIGNED(n, type,
    name[dims])`, and a `static const struct` — by anchoring on the name and
    then scanning braces, rather than trying to match the declaration syntax.
    """
    for m in re.finditer(r"\b%s\b" % re.escape(name), text):
        i = m.end()
        # Skip any [dims] and whitespace/newlines, then require `= {`.
        j = i
        while j < len(text) and (text[j].isspace() or text[j] in "[]" or text[j].isalnum() or text[j] in "_-+*"):
            if text[j] == "=":
                break
            j += 1
        k = text.find("=", i)
        if k < 0:
            continue
        between = text[i:k]
        if not re.fullmatch(r"[\s\w\[\]\-\+\*/,()]*", between or ""):
            continue
        b = text.find("{", k)
        if b < 0 or text[k + 1:b].strip() not in ("", ")"):
            continue
        depth = 0
        for e in range(b, len(text)):
            if text[e] == "{":
                depth += 1
            elif text[e] == "}":
                depth -= 1
                if depth == 0:
                    return text[b + 1:e]
        break
    raise KeyError("array %r not found" % name)


def fn_macros(src):
    """Function-like `#define`s, e.g. `INTER_OFFSET(mode) ((mode) - NEARESTMV)`.

    The mode trees are written in terms of these, so a parser that only knows
    plain enums stops dead on `-INTER_OFFSET(ZEROMV)` — which is the good
    outcome; silently dropping the token would have produced a *shorter* tree
    that still looked like a tree.
    """
    out = {}
    for name in os.listdir(src):
        if not name.endswith((".h", ".c")):
            continue
        text = strip_comments(open(os.path.join(src, name), errors="ignore").read())
        for m in re.finditer(r"#define\s+([A-Za-z_]\w*)\(([^)]*)\)\s+([^\n]+)", text):
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            out[m.group(1)] = (params, m.group(3).strip())
    return out


def expand(body, sym, macros):
    """Expand function-like macro calls to their integer value."""
    changed = True
    while changed:
        changed = False
        for name, (params, tmpl) in macros.items():
            pat = re.compile(r"\b%s\s*\(" % re.escape(name))
            m = pat.search(body)
            if not m:
                continue
            # Find the matching close paren.
            depth, i = 1, m.end()
            while i < len(body) and depth:
                depth += (body[i] == "(") - (body[i] == ")")
                i += 1
            args = [a.strip() for a in body[m.end():i - 1].split(",")]
            if len(args) != len(params):
                continue
            expr = tmpl
            for p, a in zip(params, args):
                expr = re.sub(r"\b%s\b" % re.escape(p), "(%s)" % a, expr)
            try:
                val = int(eval(expr, {"__builtins__": {}}, dict(sym)))
            except Exception:
                continue
            body = body[:m.start()] + str(val) + body[i:]
            changed = True
    return body


def values(body, sym, macros=None):
    """Flatten a brace body to a list of ints, resolving enum names."""
    if macros:
        body = expand(body, sym, macros)
    out = []
    # A **negated identifier** must be matched as one token. The trees are
    # written `-PARTITION_NONE, 2, -PARTITION_HORZ, ...` where a negative entry
    # means "leaf" and a positive one means "index of the next node pair" — so
    # a pattern that matches only `-?\d+` or a bare identifier silently drops
    # every minus and turns each leaf into a node index. The result is a tree
    # with cycles: `read_tree` walked 1 -> 2 -> 1 forever and the decoder hung.
    for tok in re.finditer(r"-?\d+|-?[A-Za-z_]\w*", body):
        t = tok.group(0)
        if re.fullmatch(r"-?\d+", t):
            out.append(int(t))
            continue
        neg = t.startswith("-")
        name = t[1:] if neg else t
        if name not in sym:
            raise KeyError("unresolved symbol %r in table" % name)
        out.append(-sym[name] if neg else sym[name])
    return out


def nest(flat, dims):
    """Reshape a flat list into `dims`, checking the count exactly.

    The check is the point: a dimension mismatch is how a table that parsed
    "fine" turns out to be shifted by one, which is undetectable downstream.
    """
    total = 1
    for d in dims:
        total *= d
    if len(flat) != total:
        raise ValueError("expected %d values for dims %s, parsed %d" % (total, dims, len(flat)))
    for d in reversed(dims[1:]):
        flat = [flat[i:i + d] for i in range(0, len(flat), d)]
    return flat


def rust_type(dims, scalar):
    t = scalar
    for d in reversed(dims):
        t = "[%s; %d]" % (t, d)
    return t


def rust_lit(v, per_line=16):
    if not isinstance(v, list):
        return str(v)
    if not isinstance(v[0], list):
        if len(v) <= per_line:
            return "[" + ", ".join(str(x) for x in v) + "]"
        rows = []
        for i in range(0, len(v), per_line):
            rows.append("    " + ", ".join(str(x) for x in v[i:i + per_line]) + ",")
        return "[\n" + "\n".join(rows) + "\n]"
    inner = [rust_lit(x, per_line) for x in v]
    return "[\n" + "\n".join(indent(s) + "," for s in inner) + "\n]"


def indent(s):
    return "\n".join("    " + line for line in s.split("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="directory holding the libvpx sources")
    ap.add_argument("--fetch", action="store_true", help="download them first")
    ap.add_argument("-o", "--out", default=OUT)
    args = ap.parse_args()
    src = args.src or os.path.join(REPO, "target", "libvpx-src")
    if args.fetch or not os.path.isdir(src):
        fetch(src)

    sym = symbols(src)
    macros = fn_macros(src)
    text = {}
    for name in os.listdir(src):
        if name.endswith((".c", ".h")):
            text[name] = strip_comments(open(os.path.join(src, name), errors="ignore").read())
    allc = "\n".join(text[k] for k in sorted(text) if k.endswith(".c"))

    def table(name, dims, scalar="u8", rust=None, per_line=16, src_text=None, pad=False):
        """`pad` mirrors C's partial initialisation: a few tables (the coef
        token tree) declare more entries than they initialise and rely on the
        rest being zero. Padding is opt-in so that everywhere else a short
        parse is still an error rather than a silently zero-filled table."""
        body = find_array(src_text if src_text is not None else allc, name)
        flat = values(body, sym, macros)
        if pad:
            total = 1
            for d in dims:
                total *= d
            if len(flat) > total:
                raise ValueError("%s: %d values exceed declared %d" % (name, len(flat), total))
            flat = flat + [0] * (total - len(flat))
        v = nest(flat, dims)
        rn = rust or name.upper().lstrip("VP9_").replace("VP9_", "")
        return "pub const %s: %s = %s;\n" % (rn, rust_type(dims, scalar), rust_lit(v, per_line))

    parts = []
    parts.append(HEADER)

    # --- coefficient model ------------------------------------------------
    parts.append(table("vp9_pareto8_full", [255, 8], rust="PARETO8_FULL", per_line=8))
    # The coefficient tables are **ragged**: band 0 has only 3 contexts while
    # bands 1..5 have 6, and libvpx declares the array [6][6][3] anyway, leaving
    # band 0's contexts 3..5 zero-filled by C's partial initialisation. Parsing
    # it as a dense [6][6][3] therefore reads 396 values into 432 slots and
    # shifts every band after the first — so the raggedness is reconstructed
    # here rather than assumed away.
    coef = []
    for tx in ("4x4", "8x8", "16x16", "32x32"):
        flat = values(find_array(allc, "default_coef_probs_%s" % tx), sym, macros)
        want = 2 * 2 * (3 + 5 * 6) * 3
        if len(flat) != want:
            raise ValueError("coef probs %s: expected %d values, parsed %d" % (tx, want, len(flat)))
        it = iter(flat)
        planes = []
        for _ in range(2):          # PLANE_TYPES
            refs = []
            for _ in range(2):      # REF_TYPES
                bands = []
                for band in range(6):
                    ctxs = []
                    for c in range(6):
                        if band == 0 and c >= 3:
                            ctxs.append([0, 0, 0])
                        else:
                            ctxs.append([next(it), next(it), next(it)])
                    bands.append(ctxs)
                refs.append(bands)
            planes.append(refs)
        if next(it, None) is not None:
            raise ValueError("coef probs %s: values left over" % tx)
        coef.append(planes)
    parts.append("pub const DEFAULT_COEF_PROBS: %s = %s;\n"
                 % (rust_type([4, 2, 2, 6, 6, 3], "u8"), rust_lit(coef, 3)))
    parts.append(table("vp9_coefband_trans_8x8plus", [1024], rust="COEFBAND_TRANS_8X8PLUS", per_line=32))
    parts.append(table("vp9_coefband_trans_4x4", [16], rust="COEFBAND_TRANS_4X4"))
    parts.append(table("vp9_pt_energy_class", [12], rust="PT_ENERGY_CLASS"))
    for n, ln in (("cat1", 1), ("cat2", 2), ("cat3", 3), ("cat4", 4), ("cat5", 5), ("cat6", 14)):
        parts.append(table("vp9_%s_prob" % n, [ln], rust="%s_PROB" % n.upper()))

    # --- modes, partitions, references ------------------------------------
    parts.append(table("vp9_kf_y_mode_prob", [10, 10, 9], rust="KF_Y_MODE_PROB", per_line=9))
    parts.append(table("vp9_kf_uv_mode_prob", [10, 9], rust="KF_UV_MODE_PROB", per_line=9))
    parts.append(table("vp9_kf_partition_probs", [16, 3], rust="KF_PARTITION_PROBS", per_line=3))
    parts.append(table("default_if_y_probs", [4, 9], rust="DEFAULT_Y_MODE_PROBS", per_line=9))
    parts.append(table("default_if_uv_probs", [10, 9], rust="DEFAULT_UV_MODE_PROBS", per_line=9))
    parts.append(table("default_partition_probs", [16, 3], rust="DEFAULT_PARTITION_PROBS", per_line=3))
    parts.append(table("default_inter_mode_probs", [7, 3], rust="DEFAULT_INTER_MODE_PROBS", per_line=3))
    parts.append(table("default_switchable_interp_prob", [4, 2], rust="DEFAULT_INTERP_FILTER_PROBS", per_line=2))
    parts.append(table("default_intra_inter_p", [4], rust="DEFAULT_INTRA_INTER_PROBS"))
    parts.append(table("default_comp_inter_p", [5], rust="DEFAULT_COMP_INTER_PROBS"))
    parts.append(table("default_comp_ref_p", [5], rust="DEFAULT_COMP_REF_PROBS"))
    parts.append(table("default_single_ref_p", [5, 2], rust="DEFAULT_SINGLE_REF_PROBS", per_line=2))
    parts.append(table("default_skip_probs", [3], rust="DEFAULT_SKIP_PROBS"))
    # struct tx_probs is three differently-shaped members in one initialiser.
    txbody = find_array(allc, "default_tx_probs")
    txv = values(txbody, sym, macros)
    p32 = nest(txv[0:6], [2, 3])
    p16 = nest(txv[6:10], [2, 2])
    p8 = nest(txv[10:12], [2, 1])
    parts.append("pub const DEFAULT_TX_PROBS_32X32: [[u8; 3]; 2] = %s;\n" % rust_lit(p32, 3))
    parts.append("pub const DEFAULT_TX_PROBS_16X16: [[u8; 2]; 2] = %s;\n" % rust_lit(p16, 2))
    parts.append("pub const DEFAULT_TX_PROBS_8X8: [[u8; 1]; 2] = %s;\n" % rust_lit(p8, 1))

    # --- trees -------------------------------------------------------------
    for name, ln, rn in (
        ("vp9_intra_mode_tree", 18, "INTRA_MODE_TREE"),
        ("vp9_inter_mode_tree", 6, "INTER_MODE_TREE"),
        ("vp9_partition_tree", 6, "PARTITION_TREE"),
        ("vp9_switchable_interp_tree", 4, "INTERP_FILTER_TREE"),
        ("vp9_coef_con_tree", 22, "COEF_CON_TREE"),
        ("vp9_mv_joint_tree", 6, "MV_JOINT_TREE"),
        ("vp9_mv_class_tree", 20, "MV_CLASS_TREE"),
        ("vp9_mv_class0_tree", 2, "MV_CLASS0_TREE"),
        ("vp9_mv_fp_tree", 6, "MV_FP_TREE"),
    ):
        parts.append(table(name, [ln], scalar="i8", rust=rn, pad=True))

    # --- motion vectors ----------------------------------------------------
    mv = values(find_array(allc, "default_nmv_context"), sym, macros)
    joints, mv = mv[:3], mv[3:]
    parts.append("pub const DEFAULT_MV_JOINT_PROBS: [u8; 3] = %s;\n" % rust_lit(joints))
    comps = []
    for _ in range(2):
        sign, mv = mv[0], mv[1:]
        cls, mv = mv[:10], mv[10:]
        class0, mv = mv[:1], mv[1:]
        bits, mv = mv[:10], mv[10:]
        class0_fp, mv = nest(mv[:6], [2, 3]), mv[6:]
        fp, mv = mv[:3], mv[3:]
        class0_hp, mv = mv[0], mv[1:]
        hp, mv = mv[0], mv[1:]
        comps.append((sign, cls, class0, bits, class0_fp, fp, class0_hp, hp))
    assert not mv, "unconsumed nmv_context values: %r" % mv
    parts.append(MV_COMP_STRUCT)
    parts.append("pub const DEFAULT_MV_COMP_PROBS: [MvCompProbs; 2] = [\n")
    for (sign, cls, class0, bits, class0_fp, fp, class0_hp, hp) in comps:
        parts.append("    MvCompProbs {\n")
        parts.append("        sign: %d,\n" % sign)
        parts.append("        classes: %s,\n" % rust_lit(cls))
        parts.append("        class0: %s,\n" % rust_lit(class0))
        parts.append("        bits: %s,\n" % rust_lit(bits))
        parts.append("        class0_fp: %s,\n" % rust_lit(class0_fp, 3).replace("\n", "\n        "))
        parts.append("        fp: %s,\n" % rust_lit(fp))
        parts.append("        class0_hp: %d,\n" % class0_hp)
        parts.append("        hp: %d,\n" % hp)
        parts.append("    },\n")
    parts.append("];\n")

    # --- scans and neighbours ---------------------------------------------
    scanc = text["vp9_scan.c"]
    for size, n in (("4x4", 16), ("8x8", 64), ("16x16", 256), ("32x32", 1024)):
        kinds = ("default",) if size == "32x32" else ("default", "col", "row")
        for kind in kinds:
            parts.append(table("%s_scan_%s" % (kind, size), [n], scalar="i16",
                               rust="%s_SCAN_%s" % (kind.upper(), size.upper()),
                               per_line=16, src_text=scanc))
            parts.append(table("%s_scan_%s_neighbors" % (kind, size), [(n + 1) * 2], scalar="i16",
                               rust="%s_SCAN_%s_NEIGHBORS" % (kind.upper(), size.upper()),
                               per_line=16, src_text=scanc))

    # --- interpolation filters --------------------------------------------
    # `vp9_filter_kernels` is an array of *pointers*, so the order of the four
    # kernels is read from it rather than assumed: it is
    # EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP, BILINEAR — which is the enum
    # order, and deliberately NOT the order the bitstream's 2-bit literal uses.
    kernels = re.search(r"vp9_filter_kernels\s*\[\s*\d*\s*\]\s*=\s*\{(.*?)\}", allc, re.S).group(1)
    names = [n.strip() for n in kernels.split(",") if n.strip()][:4]
    filt = [nest(values(find_array(allc, n), sym, macros), [16, 8]) for n in names]
    parts.append("/// Sub-pel filters in `InterpFilter` enum order: %s.\n" % ", ".join(names))
    parts.append("pub const SUBPEL_FILTERS: [[[i16; 8]; 16]; 4] = %s;\n" % rust_lit(filt, 8))

    # --- motion-vector reference search -----------------------------------
    # `mv_ref_blocks` is the fixed 8-neighbour search order per block size, as
    # (row, col) offsets in 8x8 units; `mode_2_counter`/`counter_to_context`
    # turn the two nearest neighbours' modes into the inter-mode context.
    # These three live in a *header*, which `allc` (the .c files) does not
    # include, so the source text is passed explicitly.
    mvh = text["vp9_mvref_common.h"]
    parts.append(table("mv_ref_blocks", [13, 8, 2], scalar="i8", rust="MV_REF_BLOCKS", per_line=2, src_text=mvh))
    parts.append(table("mode_2_counter", [14], rust="MODE_2_COUNTER", src_text=mvh))
    # `log_in_base_2` maps a motion-vector magnitude to its class for counting.
    parts.append(table("log_in_base_2", [1025], rust="LOG_IN_BASE_2", per_line=32,
                       src_text=text["vp9_entropymv.c"]))
    # Which loop-filter *mode* delta a block's mode selects: 0 for every intra
    # mode and for ZEROMV, 1 for the three moving inter modes.
    parts.append(table("mode_lf_lut", [14], rust="MODE_LF_LUT", src_text=text["vp9_loopfilter.c"]))
    parts.append(table("counter_to_context", [19], rust="COUNTER_TO_CONTEXT", src_text=mvh))
    # Which sub-block of a sub-8x8 *neighbour* contributes its motion vector,
    # indexed `[block_idx][search_col == 0]`.
    parts.append(table("idx_n_column_to_subblock", [4, 2], rust="IDX_N_COLUMN_TO_SUBBLOCK", per_line=2, src_text=mvh))

    # --- probability remapping --------------------------------------------
    # `inv_map_table` inverts the encoder's probability remap. It is 254
    # near-arithmetic-progression entries with deliberate irregularities at both
    # ends (it starts 7, 20, 33 … and ends 253, 253) — exactly the shape that
    # looks transcribable and is not.
    parts.append(table("inv_map_table", [255], rust="INV_MAP_TABLE", per_line=15, pad=True))

    # --- dequantisers ------------------------------------------------------
    parts.append(table("dc_qlookup", [256], scalar="i16", rust="DC_QLOOKUP", per_line=16))
    parts.append(table("ac_qlookup", [256], scalar="i16", rust="AC_QLOOKUP", per_line=16))

    # --- block geometry ----------------------------------------------------
    parts.append(table("b_width_log2_lookup", [13], rust="B_WIDTH_LOG2"))
    parts.append(table("b_height_log2_lookup", [13], rust="B_HEIGHT_LOG2"))
    parts.append(table("num_4x4_blocks_wide_lookup", [13], rust="NUM_4X4_W"))
    parts.append(table("num_4x4_blocks_high_lookup", [13], rust="NUM_4X4_H"))
    parts.append(table("num_8x8_blocks_wide_lookup", [13], rust="NUM_8X8_W"))
    parts.append(table("num_8x8_blocks_high_lookup", [13], rust="NUM_8X8_H"))
    parts.append(table("mi_width_log2_lookup", [13], rust="MI_WIDTH_LOG2"))
    parts.append(table("num_pels_log2_lookup", [13], rust="NUM_PELS_LOG2"))
    parts.append(table("size_group_lookup", [13], rust="SIZE_GROUP"))
    parts.append(table("subsize_lookup", [4, 13], rust="SUBSIZE", per_line=13))
    parts.append(table("max_txsize_lookup", [13], rust="MAX_TXSIZE"))
    parts.append(table("ss_size_lookup", [13, 2, 2], rust="SS_SIZE", per_line=2))
    parts.append(table("uv_txsize_lookup", [13, 4, 2, 2], rust="UV_TXSIZE", per_line=2))
    parts.append(table("tx_mode_to_biggest_tx_size", [5], rust="TX_MODE_TO_BIGGEST_TX_SIZE"))
    parts.append(table("partition_lookup", [5, 13], rust="PARTITION_LOOKUP", per_line=13))
    # `partition_context_lookup` is an array of two-field structs; the flattener
    # sees it as [13][2] = (above, left) bitmasks.
    parts.append(table("partition_context_lookup", [13, 2], rust="PARTITION_CONTEXT_LOOKUP", per_line=2))

    out = "".join(parts)
    with open(args.out, "w") as f:
        f.write(out)
    sys.stderr.write("wrote %s (%d bytes)\n" % (args.out, len(out)))


HEADER = '''//! VP9 constant tables — **generated, do not edit**.
//!
//! Produced by `tools/gen_vp9_tables.py` from the libvpx sources (BSD; see
//! THIRDPARTY-LICENSES.md). Regenerate with:
//!
//! ```sh
//! python3 tools/gen_vp9_tables.py --fetch
//! ```
//!
//! Everything here is a number that fails *quietly* when wrong: a mistyped
//! coefficient probability does not error, it decodes a slightly different
//! picture, and every inter frame predicting from it inherits the difference.
//! Hand-transcribing ~9000 of them is not a thing that can be reviewed, which
//! is why they are parsed out of the reference implementation instead — the
//! same rule `gen_cabac_tables.py` follows for H.264.

#![allow(clippy::all)]

'''

MV_COMP_STRUCT = '''
/// One motion-vector component's probability set (`nmv_component` in libvpx).
#[derive(Clone, Copy)]
pub struct MvCompProbs {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: [u8; 1],
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

'''

if __name__ == "__main__":
    main()
