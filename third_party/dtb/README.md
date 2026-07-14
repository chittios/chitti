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

Requires `dtc` and `clang` (both standard on macOS with the Xcode CLT +
`brew install dtc`). The script strips a handful of floating-point power-tuning
properties the Asahi dts uses, since only Asahi's patched `dtc` accepts them and
they are irrelevant to booting.
