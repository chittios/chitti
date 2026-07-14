# Apple-Silicon device trees (for booting ChittiOS via m1n1)

Booting ChittiOS on a real Mac over the m1n1 USB proxy needs the machine's base
**Linux device tree** (`.dtb`). m1n1's `linux.py` uploads it and m1n1 patches it
at runtime — memory size, the `simple-framebuffer`, and MMIO tunables from the
Apple Device Tree — before handing the final FDT to ChittiOS (which reads it via
[`crate::fdt`](../../kernel/src/fdt.rs)).

These DTBs are **not vendored**: they are generated from the Asahi Linux kernel's
device-tree sources (GPL-2.0), so this directory only carries the build recipe.
The `.dtb` files and the sparse-clone cache (`.asahi-linux/`) are gitignored.

## Build

```sh
tools/build-apple-dtb.sh            # Mac mini M2  → third_party/dtb/t8112-j473.dtb
tools/build-apple-dtb.sh t6020-j474 # Mac mini M2 Pro
```

Then point the boot loop at it:

```sh
export CHITTI_DTB=third_party/dtb/t8112-j473.dtb
make m1n1 RELEASE=1                 # (also needs CHITTI_M1N1, M1N1DEVICE — see `make help`)
```

Requires `dtc`, `clang`, and `python3` (all standard on macOS with the Xcode CLT
+ `brew install dtc`). The Asahi dts uses floating-point cell values (GPU/CPU
power-tuning gains) that only Asahi's patched `dtc` accepts; the script
pre-converts each to its IEEE-754 f32 hex encoding — exactly what Asahi's `dtc`
emits — so mainline `dtc` compiles them and, crucially, m1n1's device-tree prep
still finds the properties to write the GPU tables into (stripping them makes
m1n1 fail with "DT prepare failed").
