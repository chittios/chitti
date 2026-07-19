# AGX initdata extraction (live-M2 oracle)

The m1n1 proxy (`M1N1DEVICE=/dev/cu.usbmodem*`) lets the *working* proxyclient
boot the AGX GPU on the real M2 and dump the exact initdata it builds, so
ChittiOS can replicate it (same machine → same GPU VAs → embedded pointers stay
valid). Deps: `pip install --break-system-packages pyserial construct`.

- `dump_initdata.py` — full working boot (`AGX.start()`), dumps `initdata_struct.txt`
  (full field/offset/value tree), `chip_info_8112.txt`, `initdata_bytes.bin`.
- `dump_regions.py` — read-only dump of every initdata sub-region's raw bytes via
  a bare UAT walking the live page tables (no re-boot), + `manifest.json`.

Ground truth (t8112 / FW 13.5 / RTKit v12): kern_va_base=0xffffffa000000000,
RTKit buffers at +0x80000000, regions per manifest.json. Regenerate anytime the
proxy is live. Region .bin dumps live in scratchpad-agx/regions/ (gitignored).
