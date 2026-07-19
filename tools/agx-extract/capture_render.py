#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Capture a COMPLETE, known-good GPU render submission from the m1n1 proxy, to use
# as the reference for porting AGX command submission into ChittiOS (see
# kernel/src/agx/COMPUTE_PLAN.md, de-risking step 1).
#
# `experiments/agx_1tri.py` builds the WorkCommands/microsequence in code, then
# submits a triangle render; dumping the render context's whole VM afterward gives
# exact bytes for every GPU object the firmware consumes (WorkCommandTA/3D, the
# microsequences, the cmdqueue RunCmdQueueMsg, EventControl, TVB/tiler buffers, the
# context TTBR).
#
# !!! DEPENDENCY (confirmed 2026-07): agx_1tri is NOT self-contained. It does
#     ctx.load_blob(0x1100000000, ..., "gpudata/bunny/mem_*.bin") — Metal-captured
#     SHADER/PIPELINE bytes — which are NOT in the m1n1 tree. Without gpudata/bunny/
#     it FileNotFoundErrors before submitting. To use this script you must first
#     place the bunny capture at proxyclient/experiments/gpudata/bunny/ (a macOS
#     Metal capture, or the Asahi m1n1 render-asset set). See COMPUTE_PLAN.md
#     "Proxy-session findings": Layer 3 (shader bytes) is unavoidable even for a
#     render, so Layers 1-2 are better ported from m1n1 SOURCE (Path A) and this
#     capture is only for Path B (validate render plumbing with a sourced bunny).
#
# PREREQUISITES (same as the initdata capture):
#   - The M2 booted into m1n1 PROXY mode (NOT ChittiOS), proxy cable attached.
#   - pip install --break-system-packages pyserial construct
#   - env: AGX_FWVER=V13_5 AGX_GPU=G14   (and M1N1DEVICE=/dev/cu.usbmodem…)
#
# USAGE:
#   AGX_FWVER=V13_5 AGX_GPU=G14 M1N1DEVICE=/dev/cu.usbmodemXXXX \
#     python3 tools/agx-extract/capture_render.py
#
# OUTPUT (this dir): render_capture/ with
#   - ctx_memmap.json   : [{va, size}] every mapped page-run in the render ctx
#   - ctx_data.bin      : concatenated bytes of all data-bearing ranges
#   - ranges.json       : [{va, size, data_off|null}] indexing into ctx_data.bin
#   - meta.json         : ctx_id, ttbr0, the WorkCommand/microsequence/cmdqueue VAs
#
# NB: this dumps the RENDER context (ctx_id from agx_1tri, not ctx 0). We read the
# page tables for that context via uat.foreach_page(ctx_id, ...), the same walk the
# initdata full_memmap capture used for ctx 0.

import os, sys, json, pathlib, runpy

os.environ.setdefault("AGX_FWVER", "V13_5")
os.environ.setdefault("AGX_GPU", "G14")

HERE = pathlib.Path(__file__).resolve().parent
M1N1 = HERE.parents[1] / "third_party/m1n1/proxyclient"
sys.path.insert(0, str(M1N1))

from m1n1 import constructutils as cu
for g in ("G15", "G16"):
    if g not in cu.Ver.MATRIX["G"]:
        cu.Ver.MATRIX["G"].append(g)

OUT = HERE / "render_capture"
OUT.mkdir(exist_ok=True)

# ---- Run agx_1tri.py up through submission, capturing its locals -------------
# agx_1tri.py runs at module level inside a try/except that reboots on error and
# calls agx.stop() at the end. We neutralise the reboot/stop so we can dump after
# the render completes, and grab the module namespace for the object VAs.
tri_path = M1N1 / "experiments/agx_1tri.py"

# runpy executes the script as __main__; it leaves its globals in the returned
# dict. We patch Proxy.reboot to a no-op via the m1n1 setup first is hard here, so
# instead we rely on a successful run (the render completes on a healthy GPU) and
# read back the objects it registered with the AGX allocator.
ns = runpy.run_path(str(tri_path), run_name="__main__")

agx = ns["agx"]
ctx = ns["ctx"]
ctx_id = ctx.ctx

# ---- Dump the render context's whole VM via the UAT page walk ----------------
ranges = []  # [va, size]
def page_fn(start, end, i, pte, level, sparse=False):
    if pte is None or not pte.valid():
        return
    size = end - start
    if ranges and ranges[-1][0] + ranges[-1][1] == start:
        ranges[-1][1] += size
    else:
        ranges.append([start, size])

agx.uat.foreach_page(ctx_id, page_fn)
print(f"render ctx {ctx_id}: {len(ranges)} mapped range(s)")

blob = bytearray()
index = []
for va, size in ranges:
    data = agx.uat.ioread(ctx_id, va, size)
    if data is None or data == bytes(size):
        index.append({"va": hex(va), "size": hex(size), "data_off": None})
    else:
        index.append({"va": hex(va), "size": hex(size), "data_off": len(blob)})
        blob += data

(OUT / "ctx_data.bin").write_bytes(bytes(blob))
(OUT / "ranges.json").write_text(json.dumps(index, indent=1))
(OUT / "ctx_memmap.json").write_text(
    json.dumps([{"va": hex(v), "size": hex(s)} for v, s in ranges], indent=1))

# ---- Record the key struct VAs so the port knows what to submit -------------
def addr_of(name):
    obj = ns.get(name)
    return hex(obj._addr) if obj is not None and hasattr(obj, "_addr") else None

meta = {
    "ctx_id": ctx_id,
    "ttbr0": hex(getattr(ctx, "ttbr0_base", 0)),
    "pipeline_base": hex(getattr(ctx, "pipeline_base", 0)),
    "wc_3d": addr_of("wc_3d"),
    "wc_ta": addr_of("wc_ta"),
    "wc_initbm": addr_of("wc_initbm"),
    "event_control": addr_of("event_control"),
    "blob_bytes": len(blob),
    "ranges": len(ranges),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("wrote", OUT)
print(json.dumps(meta, indent=1))
