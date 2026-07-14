# Apple-Silicon USB HID (keyboard/mouse) — status & hardware verification

Stage 2 of the [m1n1 Apple-Silicon port](../CLAUDE.md). ChittiOS already boots
and renders its UI on a real Mac mini M2 (t8112) via m1n1, single-core, with the
FDT `simple-framebuffer` and the s5l serial console. This documents the **USB
HID host** work: bringing up a USB keyboard/mouse so the machine is usable
without the serial tether.

## What is implemented (committed, builds, untested on hardware)

A grounded first draft, ported from m1n1's own C source
(`third_party/m1n1/src/{dart,usb,usb_dwc3}.c`) + the real t8112 device tree:

- **DART** (`arch/aarch64/dart.rs`) — the T8110 IOMMU. Exact PTE/TTBR encoding
  and IOVA split from `dart.c` (16 KiB pages), unit-tested. `set_bypass()` puts
  the USB stream in bypass (device addresses = physical); with ChittiOS's
  identity map that is the simplest correct DMA path, and m1n1 already leaves the
  USB DART in bypass at handoff. Full translation primitives are written for
  later.
- **`apple_usb.rs`** — discovers from the FDT: DWC3 core (`apple,t8112-dwc3`,
  `usb@382280000` reg[0]), ATC-PHY core + `pipehandler` (`apple,t8112-atcphy`,
  `phy@383000000` reg[0]/reg[4]), and the USB DART + stream id (via the dwc3's
  `iommus = <&dart SID>`). Then: replay m1n1's USB2 PHY + pipehandler writes,
  soft-reset the DWC3 core/PHYs into **HOST** mode (`GCTL.PRTCAPDIR = HOST` — m1n1
  runs it as a *device*), put the DART in bypass, and hand the xHCI window at
  `DWC3_base + 0x0` to the shared `xhci` core (`xhci::attach_at`).
- All gated on `is_apple()`; degrades gracefully (logs, returns false); QEMU is
  unchanged.

## What is NOT done — why a plugged-in keyboard won't enumerate yet

m1n1 only ever brings the PHY up as a **gadget** (it *is* the USB device; the
host cable supplies Vbus/CC). Driving a **downstream** device as a host needs
two things m1n1's source does not provide:

1. **Host-mode PHY / port mux.** `apple_usb::phy_bringup` currently replays
   m1n1's `PIPEHANDLER_MUX_CTRL_DUMMY` (dummy PHY). A host wanting a real USB2
   device on the port needs the native USB2 path / a different mux + lane
   configuration. There is no host-mode ATC-PHY sequence in m1n1 to port; it
   must come from the Asahi Linux `phy/apple/atc.c` driver or be reverse-engineered.
2. **Type-C orientation + port power** via the **TPS6598x PD controller over
   I²C** (`hpm@...` in the FDT). A host must tell the PD controller to source
   Vbus and set CC orientation. This needs an Apple I²C driver (`apple,i2c`) +
   the TPS6598x command set (m1n1 has `tps6598x_*`/`hpm_init` in `usb.c`, but
   only masks its IRQs for the gadget path).

Practical implication: the two **USB-A** ports on the Mac mini may be simpler
(no Type-C orientation), but they still sit behind the same dwc3/ATC path and
need host-mode PHY. Start there for testing.

## How to test on hardware (the catch-22)

**The m1n1 *hypervisor* debug path removes USB from the guest** — the hv boot
log shows `Removing ADT node /arm-io/dart-usb0`, `atc-phy0`, `usb-drd0`. So USB
is *only* available on a **bare** `linux.py` boot (no hv), which has **no serial
console** (m1n1 tears the USB proxy down at handoff). But on a bare boot the
**framebuffer works** (the FDT simple-framebuffer), so:

- **Debug over the display, not serial.** ChittiOS's `apple_usb`/`dart` steps
  all `ktrace`, which also render in the framebuffer console. On a bare boot,
  watch the Mac's monitor for the `apple_usb:` / `dart:` / `xhci(apple):` lines
  and where it stops.
- Bare boot (no `CHITTI_M1N1_HV`, no `-t`):
  ```sh
  CHITTI_M1N1=third_party/m1n1 CHITTI_DTB=third_party/dtb/t8112-j473.dtb \
  M1N1DEVICE=/dev/cu.usbmodemXXXXD1 \
  make m1n1 RELEASE=1
  ```
  Then read the Mac's screen. Plug a **USB-A keyboard** in before/at boot.
- Iterating on the PHY/PD work will need this loop plus, ideally, a physical
  UART cable to the Mac's debug UART for real serial logs (the s5l console is on
  hardware pins; the hv VUART is unavailable on a bare boot).

## If DART bypass is wrong

If the controller faults on DMA (a DART error, or the guest sees an unmapped
IPA around the xHCI DMA buffers), switch `apple_usb` from `dart.set_bypass()` to
the translation path: allocate a 16 KiB L1 table, `make_ttbr` → write
`TTBR(sid,0)`, set `TCR.TRANSLATE_ENABLE`, and map each xHCI DMA buffer's
`(iova, pa)` with the `dart.rs` PTE helpers, then flush the SID's TLB. The pure
encoders (`iova_split`, `make_pte`, `make_ttbr`) are unit-tested.

## References

- m1n1 source (vendored): `third_party/m1n1/src/{dart,usb,usb_dwc3,pmgr}.c`.
- The machine device tree: `third_party/dtb/t8112-j473.dtb`
  (`tools/build-apple-dtb.sh`); decompile with `dtc -I dtb -O dts` to read the
  `usb@`, `phy@`, `iommu@` nodes.
- Asahi Linux drivers for the missing pieces: `phy/apple/atc.c` (host PHY),
  `drivers/usb/typec/tipd/` + `drivers/i2c/busses/i2c-apple.c` (PD controller).
