#!/usr/bin/env python3
# The GPU is already booted (prior run) with its page tables live in DRAM.
# Read the initdata regions read-only via a bare UAT (walks the live tables),
# no AGX boot / PPL handshake.
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
OUT = pathlib.Path(__file__).resolve().parent / "regions"
OUT.mkdir(exist_ok=True)

from m1n1.hw.uat import UAT
uat = UAT(u.iface, u)

REGIONS = [
    ("InitData",            0xffffffa0407c3f44, 0xbc),
    ("InitData_RegionA",    0xffffffa040804000, 0x4000),
    ("InitData_RegionB",    0xffffffa0003b9440, 0x6bc0),
    ("InitData_RegionC",    0xffffffa0408d1c6c, 0x12394),
    ("AGXHWDataA",          0xffffffa0004cfde4, 0x421c),
    ("AGXHWDataB",          0xffffffa00055a77c, 0x1884),
    ("AGXFaultInfo",        0xffffffa04084bf80, 0x80),
    ("InitData_BufferMgrCtl", 0xffffffa04088f810, 0x7f0),
    ("InitData_FWStatus",   0xffffffa00066bf80, 0x80),
    ("InitData_GPUGlobalStats3D", 0xffffffa0004478b8, 0x748),
    ("InitData_GPUGlobalStatsTA", 0xffffffa000403970, 0x690),
]
manifest = {"initdata_va": 0xffffffa0407c3f44, "kern_va_base": 0xffffffa000000000, "regions": []}
for name, va, size in REGIONS:
    try:
        raw = uat.ioread(0, va & 0xFFFFFFFFFF, size)
        (OUT / f"{name}.bin").write_bytes(raw)
        nz = sum(1 for b in raw if b)
        manifest["regions"].append({"name": name, "va": va, "size": size})
        print(f"  {name:28} va={va:#x} size={size:#x} nonzero={nz}/{len(raw)}")
    except Exception as e:
        print(f"  {name}: ERR {e}")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("wrote manifest.json + region bins")
