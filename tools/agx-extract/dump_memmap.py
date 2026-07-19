#!/usr/bin/env python3
# Enumerate the full ctx-0 GPU VA memory map from the live page tables — the
# complete list of what ChittiOS must replicate for initdata to be accepted.
import os
os.environ.setdefault("AGX_FWVER", "V13_5")
os.environ.setdefault("AGX_GPU", "G14")
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party/m1n1/proxyclient"))
from m1n1.setup import *
from m1n1 import constructutils as cu
for g in ("G15", "G16"):
    if g not in cu.Ver.MATRIX["G"]:
        cu.Ver.MATRIX["G"].append(g)
from m1n1.hw.uat import UAT, PTE, Page_PTE
uat = UAT(u.iface, u)

# Walk ctx-0, TTBR0 (low) and TTBR1 (high), collecting mapped 16 KiB leaf pages,
# coalescing contiguous VA runs.
PAGE = 0x4000
ranges = []
def add(va, pa):
    if ranges and ranges[-1][0] + ranges[-1][2] == va and ranges[-1][1] + ranges[-1][2] == pa:
        ranges[-1][2] += PAGE
    else:
        ranges.append([va, pa, PAGE])

# iotranslate probes; instead walk via the UAT's own recurse. Fall back: scan the
# known kern_va window in 16 KiB steps and translate.
KBASE = 0xffffffa000000000
# TTBR0 window (rings/states/objects) + TTBR1 (fw shared) — scan a bounded span.
def scan(lo, hi, ctx=0):
    va = lo
    while va < hi:
        try:
            res = uat.iotranslate(ctx, va & 0xFFFFFFFFFF, PAGE)
            pa, sz = res[0]
        except Exception:
            pa = None
        if pa is not None:
            add(va, pa)
        va += PAGE

print("scanning kern_va low (rings/states/objects) 0..0x40000000 + states 0x40000000..0x41000000 ...")
scan(KBASE, KBASE + 0x40000000)          # rings + fw channels + objects
scan(KBASE + 0x40000000, KBASE + 0x41000000)  # state areas + more objects
out = pathlib.Path(__file__).resolve().parent / "regions" / "memmap.json"
data = [{"va": hex(v), "pa": hex(p), "size": hex(s)} for v, p, s in ranges]
out.write_text(json.dumps(data, indent=1))
total = sum(s for _, _, s in ranges)
print(f"{len(ranges)} contiguous mapped ranges, total {total:#x} bytes; wrote memmap.json")
for v, p, s in ranges[:40]:
    print(f"  va={v:#x} pa={p:#x} size={s:#x}")
