#!/usr/bin/env python3
"""Dump GGUF metadata + tensor directory (no tensor data needed).

Works on truncated files — only the header/kv/tensor-info section is parsed, so
a range-downloaded head of a multi-GB HF file is enough:

    curl -L -r 0-67108863 -o /tmp/head.gguf https://huggingface.co/<repo>/resolve/main/<file>.gguf
    python3 tools/ggufmeta.py /tmp/head.gguf

Prints every metadata key (arrays summarized), the tensor list with dims/quant
types, and a quant-type histogram. Used by the dynamic-GGUF work (Stage 0) to
pin each architecture's key set before kernel code is written.
"""

import struct
import sys

# GGUF metadata value types.
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

# ggml tensor types (ggml.h) — name by id for the tensor directory.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}


class R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise EOFError(f"need {n} bytes at {self.o}, have {len(self.b)}")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def s(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")


def read_val(r, t):
    if t == T_U8: return r.take(1)[0]
    if t == T_I8: return struct.unpack("<b", r.take(1))[0]
    if t == T_U16: return struct.unpack("<H", r.take(2))[0]
    if t == T_I16: return struct.unpack("<h", r.take(2))[0]
    if t == T_U32: return r.u32()
    if t == T_I32: return struct.unpack("<i", r.take(4))[0]
    if t == T_F32: return struct.unpack("<f", r.take(4))[0]
    if t == T_BOOL: return bool(r.take(1)[0])
    if t == T_STR: return r.s()
    if t == T_U64: return r.u64()
    if t == T_I64: return struct.unpack("<q", r.take(8))[0]
    if t == T_F64: return struct.unpack("<d", r.take(8))[0]
    if t == T_ARR:
        et, n = r.u32(), r.u64()
        return ("array", et, n, [read_val(r, et) for _ in range(n)])
    raise ValueError(f"bad value type {t}")


def main(path):
    b = open(path, "rb").read()
    r = R(b)
    magic = r.take(4)
    assert magic == b"GGUF", f"not GGUF: {magic!r}"
    ver, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    print(f"== {path}: GGUF v{ver}, {n_tensors} tensors, {n_kv} kv ==")

    for _ in range(n_kv):
        k = r.s()
        t = r.u32()
        v = read_val(r, t)
        if isinstance(v, tuple) and v[0] == "array":
            _, et, n, vals = v
            head = ", ".join(repr(x) for x in vals[:6])
            print(f"  {k} = array(etype={et}, n={n}) [{head}{', ...' if n > 6 else ''}]")
        elif isinstance(v, str) and len(v) > 200:
            print(f"  {k} = str({len(v)}) {v[:200]!r}...")
        else:
            print(f"  {k} = {v!r}")

    hist = {}
    print(f"  -- tensors ({n_tensors}) --")
    for _ in range(n_tensors):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty = r.u32()
        off = r.u64()
        tn = GGML_TYPES.get(ty, f"?{ty}")
        hist[tn] = hist.get(tn, 0) + 1
        print(f"  {name}  dims={dims} type={tn} off={off}")
    print(f"  -- quant histogram: {dict(sorted(hist.items()))} --")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
