#!/usr/bin/env python3
"""Diff the kernel-interpreter dump against the onnxruntime reference dump.

  python3 diff.py /tmp/kernel_dump.txt /tmp/ref_dump.txt [max_reports]

Walks the kernel dump in execution order and reports the first tensors whose
stats (n, maxabs, mean, head values) diverge beyond tolerance — the earliest
divergence is the bug; everything after is fallout.
"""
import re, sys

LINE = re.compile(r"^(NODE (\S+)|REF) '([^']+)' (?:dims=\[([^\]]*)\] )?n=(\d+)(?: maxabs=([\d.eE+-]+|NaN|inf) mean=([\d.eE+-]+|NaN|inf) v=\[([^\]]*)\])?")

def parse(path, kind):
    out = {}
    order = []
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        op = m.group(2) or "?"
        name = m.group(3)
        dims = m.group(4) or ""
        n = int(m.group(5))
        maxabs = float(m.group(6)) if m.group(6) not in (None, "NaN", "inf") else float("nan") if m.group(6) == "NaN" else float("inf") if m.group(6) == "inf" else 0.0
        mean = float(m.group(7)) if m.group(7) not in (None, "NaN", "inf") else 0.0
        head = [float(x) for x in m.group(8).split(",")] if m.group(8) else []
        if name not in out:  # first write wins (Loop bodies re-log)
            out[name] = (op, dims, n, maxabs, mean, head)
            order.append(name)
    return out, order

def close(a, b, atol=1e-3, rtol=0.02):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

def main():
    kpath, rpath = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    kern, korder = parse(kpath, "kernel")
    ref, _ = parse(rpath, "ref")
    shown = 0
    matched = 0
    for name in korder:
        kop, kdims, kn, kmax, kmean, khead = kern[name]
        if name not in ref:
            continue
        _, rdims, rn, rmax, rmean, rhead = ref[name]
        if rn == 0:
            continue
        bad = []
        if kn != rn:
            bad.append(f"n {kn} vs {rn}")
        if not close(kmax, rmax):
            bad.append(f"maxabs {kmax:.6g} vs {rmax:.6g}")
        if not close(kmean, rmean, atol=1e-3):
            bad.append(f"mean {kmean:.6g} vs {rmean:.6g}")
        for i, (a, b) in enumerate(zip(khead, rhead)):
            if not close(a, b):
                bad.append(f"v[{i}] {a:.6g} vs {b:.6g}")
                break
        if bad:
            print(f"DIVERGE [{kop}] '{name}' kdims=[{kdims}] rdims=[{rdims}]: {'; '.join(bad)}")
            shown += 1
            if shown >= limit:
                print(f"... (stopped after {limit}; {matched} matched before)")
                return
        else:
            matched += 1
    print(f"done: {shown} divergent, {matched} matched")

if __name__ == "__main__":
    main()
