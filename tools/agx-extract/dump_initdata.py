#!/usr/bin/env python3
# Extract the exact AGX initdata the working proxyclient builds for THIS M2.
import os
os.environ.setdefault("AGX_FWVER", "V13_5")   # firmware 13.5.0
os.environ.setdefault("AGX_GPU", "G14")        # t8112 = M2 (G14 family)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party/m1n1/proxyclient"))
from m1n1.setup import *
from m1n1 import constructutils as cu
# The vendored uat.py references newer G tags absent from the Ver matrix.
for g in ("G15", "G16"):
    if g not in cu.Ver.MATRIX["G"]:
        cu.Ver.MATRIX["G"].append(g)
print("Ver:", cu.Ver._version)
OUT = pathlib.Path(__file__).resolve().parent

from m1n1.agx.initdata import CHIP_INFO
ci = CHIP_INFO[0x8112]
with open(OUT / "chip_info_8112.txt", "w") as f:
    for k in sorted(ci.keys()):
        if not k.startswith("_"):
            f.write(f"{k} = {ci[k]!r}\n")
print("wrote chip_info_8112.txt")

p.pmgr_adt_power_enable("/arm-io/gfx-asc")
p.pmgr_adt_power_enable("/arm-io/sgx")
from m1n1.agx import AGX
agx = AGX(u)
agx.verbose = 2
print("=== agx.start() (full working boot) ===")
try:
    agx.start()
    print("*** GPU START OK on this M2 ***")
except Exception as e:
    import traceback; traceback.print_exc()

idata = getattr(agx, "initdata", None)
if idata is not None:
    print("initdata GPU VA:", hex(idata._addr))
    (OUT / "initdata_struct.txt").write_text(str(idata))
    print("wrote initdata_struct.txt")
    try:
        raw = agx.uat.ioread(0, idata._addr & 0xFFFFFFFFFF, getattr(idata, "_size", 0x4000))
        (OUT / "initdata_bytes.bin").write_bytes(raw)
        print("wrote initdata_bytes.bin", len(raw))
    except Exception as e:
        print("raw read err:", e)
else:
    print("no initdata")
print("DONE")
