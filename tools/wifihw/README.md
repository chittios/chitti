# WiFi hardware bring-up harness (m1n1 proxyclient)

Drives the BCM4387/4388 WiFi endpoint on a real M2 (t8112/j473) directly from
Python over the m1n1 proxyclient, so the dongle bring-up — and specifically
**what powers the SYS_MEM/RAM domain** (the BAR2/TCM read-abort blocker) — can be
found *empirically* instead of guessed, then ported to the ChittiOS driver
(`kernel/src/arch/aarch64/apple_pcie.rs`, `kernel/src/drivers/wifi/brcm/`).

## Prereqs

- The M2 booted into **m1n1 proxy mode** (plain m1n1, NOT chainloading ChittiOS),
  USB connected. Verify the device: `ls /dev/cu.usbmodem*`.
- The vendored proxyclient venv: `third_party/m1n1/.venv`.

## Run

```sh
M1N1DEVICE=/dev/cu.usbmodemW945XQL26D1 \
  third_party/m1n1/.venv/bin/python tools/wifihw/wifi.py <cmd>
```

Commands:
- `state` — read current port/PERST/link + gP0d + WiFi config (read-only).
- `up` — `apple_pcie_setup_link` order: CLKREQ pinmux (pins 162/163/164
  → periph1), APPCLK, assert PERST#, optional rail-off **while held**, pwren,
  refclk, 100 ms, deassert, 100 ms, LTSSM. Then map BARs and read chipcommon
  + **SYS_MEM coreinfo** + TCM. A real coreinfo (not `0xffffffff`/`0xabad1dea`)
  means the RAM domain is POWERED. Never drop the rail with PERST# deasserted.

`p.read32/write32` recover from external aborts (return `0xabad1dea`), so pokes
can't crash m1n1 — safe to experiment with power/PERST/refclk orderings until
SYS_MEM comes alive, then copy the winning sequence into the kernel.

## Known-good live facts (captured over the proxyclient)

- ECAM `0x690000000`; port0 regs `0x681000000`; port0 PHY `0x680084000`.
- In proxy mode m1n1 holds the WiFi in reset: `port0+0x814 PERST=0x0`, LTSSM=0.
- Bus-1 (WiFi) config aborts until the port is brought up.
