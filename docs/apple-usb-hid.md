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

## How to test on hardware — with a host-captured serial log

A **wired USB keyboard on a USB-C port works today** (dual-DART translation +
non-coherent DMA maintenance, commit `6838562`). USB-A ports (internal hub) and
Type-C orientation (cd321x PD controller) are still pending, so for now use a
**USB-C** port.

**A bare `linux.py` boot has NO USB serial.** The USB serial device (both the
`_01` proxy channel and the `_03` UART bridge) is *m1n1's own USB gadget* — it
disappears the instant m1n1 hands off to our payload. The host log stops dead at
`Preparing to run next stage … / --- Exit TTY mode ---`; ChittiOS's own
`ktrace`/banner writes go to the physical s5l UART, which is not wired to any
host-visible device. So there are exactly two ways to see ChittiOS's serial:

- **The m1n1 hypervisor path — the one that works with no extra hardware.**
  m1n1 stays *resident* as a hypervisor, traps the guest's s5l UART writes, and
  forwards them over its still-live USB gadget. `CHITTI_SERIAL_LOG` then tees
  that to a logfile — every `ktrace`/panic line lands there, self-serve:

  ```sh
  CHITTI_M1N1_HV=1 CHITTI_SERIAL_LOG=target/serial.log make m1n1 RELEASE=1
  ```

  The catch: **the hv strips USB from the guest** (`Removing ADT node
  /arm-io/dart-usb0`, `atc-phy0`, `usb-drd0`), so this path serial-logs
  *everything except* USB/Bluetooth bring-up. Use it for PCIe, storage, AIC,
  SMP, MMU, model, etc.
- **A physical UART cable** to the Mac's debug-UART pins (the s5l TX/RX exposed
  on a specific USB-C port with the Apple-Silicon debug cable). This is the
  *only* way to get serial on a **bare** boot, and thus the only way to
  serial-log the **USB/Bluetooth** work (which the hv can't host). Then
  `CHITTI_M1N1_TTY=/dev/cu.usbmodem…` + `CHITTI_SERIAL_LOG=target/serial.log`
  capture it. (A software alternative — a ChittiOS DWC3 *device-mode* CDC-ACM
  gadget so the OS re-creates the USB serial console after handoff — is a real
  but bounded project reusing the host-mode DWC3 work.)

`ktrace` is synchronous (spins on TX-empty, no buffering), so on whichever
channel is live, the last line before a fault/hang is the true failure point —
add `ktrace::log_fmt` liberally and read them back from the logfile.

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
