# ChittiOS — hardware support

What the OS actually drives, on both architectures, and how far each driver has
been proven. This is a status document, not a wish list: everything marked
**Working** has been exercised, and everything that has *not* been run against
the real part says so.

The project rule that shapes this table is in [CLAUDE.md](CLAUDE.md): drivers
target real, standards-based hardware — ACPI/PCIe ECAM, UEFI GOP, EDID, HID
report descriptors, PrimeCell IDs — never an emulator quirk. A feature that only
works under QEMU is not done.

## How to read this

**Status**

| | |
|---|---|
| **Working** | Implemented and exercised end to end. |
| **Partial** | Useful but incomplete; the gap is named in Notes. |
| **Identify only** | The device is recognised and reported, but deliberately not driven — see Notes for why. |
| **Absent** | Not implemented. |

**Verified on** — where the code has actually been run. This distinction matters
more than status: a driver can be complete and still never have met its device.

| | |
|---|---|
| **HW** | Run against the physical part. |
| **QEMU** / **VBox** / **UTM** | Run against that emulator or hypervisor only. |
| **Tests** | Pure logic covered by `cargo xtask test`; the hardware path is unexercised. |

Arch column: **x86** = x86_64, **arm** = aarch64, **both** = one code path serving
each behind the same API.

---

## Boot and platform

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| Limine boot protocol | x86 | Working | QEMU, VBox | Memory map, HHDM, framebuffer, boot modules. |
| UEFI stub (`stub/`) | arm | Working | QEMU (AAVMF), VBox, UTM | Own bootloader; GOP mode selection, EDID capture, boot-info handoff. |
| `-kernel` direct boot | arm | Working | QEMU | Dev loop; no PCI (ECAM comes from the stub's ACPI). |
| m1n1 chainload | arm | Working | HW (Apple M2 / t8112) | `cargo xtask m1n1`. |
| Device tree (FDT) | arm | Working | QEMU | `/memory`, PSCI, GICv3 discovery. |
| ACPI tables | both | Working | QEMU, VBox | RSDP/XSDT/FADT/MADT/MCFG. x86 maps each table explicitly — the tables sit outside the HHDM. |
| AML interpreter | both | Working | QEMU | Fail-closed subset: `_S5_`, `_CRS`, `_STA`, `_BST`, `_BIX`/`_BIF`, `_PSR`, `_DSM`. Unsupported opcode returns nothing rather than guessing. |
| fw_cfg | both | Working | QEMU | RAM size, ramfb. |
| Apple SMC | arm | Partial | HW (M2) | System endpoints; used for platform info. |

## CPU, memory, privilege

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| SMP bring-up | x86 | Working | QEMU, VBox | Limine/MADT; APs woken by IPI, parked in `hlt`. |
| SMP bring-up | arm | Working | QEMU, HW (M2) | PSCI `CPU_ON`; `WFE`-parked workers with a counter event-stream fallback. Degrades to single-core where a hypervisor traps `WFE` (VirtualBox-ARM). |
| MMU / paging | both | Working | QEMU, VBox, HW | 4-level PML4 (x86); L1/L2/L3 walker plus GiB/2 MiB blocks (arm). |
| Physical frame allocator | both | Working | Tests, QEMU | Shared bitmap allocator; constructor takes usable regions, so each arch feeds it from its own source. |
| Per-task address spaces | both | Working | QEMU | `mm/space.rs`, `mm/walk.rs`. |
| Ring 3 / EL0 | both | Working | QEMU | Tenants run unprivileged; a tenant fault is contained and reported, not fatal. |
| Heap growth + OOM policy | both | Working | Tests, QEMU | Grow → reclaim hooks → retry → OOM-kill the offending task. Bootstrap/shell still panics — nowhere safe to land. |
| Fault isolation | both | Working | QEMU | A faulting task is killed and reaped; the machine survives. Double fault stays fatal. |
| NEON / SIMD | arm | Working | HW (M2), QEMU | Hot loops use inline-asm loads — `+strict-align` scalarises intrinsics. |
| AVX2 / XSAVE | x86 | Working | QEMU | Per-task FPU save area. |

## Interrupts and timers

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| Local APIC timer | x86 | Working | QEMU, VBox | Calibrated against the HPET. |
| HPET | x86 | Working | QEMU | Reference clock, with a counter-liveness probe. |
| PIT / 8259 PIC | x86 | Working | QEMU | Fallback where a UEFI-only machine omits the APIC path. |
| GICv3 | arm | Working | QEMU, VBox, UTM | Base from FDT, else the ACPI MADT. Apple Silicon has neither — stays cooperative by design. |
| ARM generic timer | arm | Working | QEMU, HW | Virtual counter (`CNTVCT_EL0`). |
| RTC / wall clock | both | Working | QEMU, VBox | CMOS RTC, UEFI `GetTime`, or the virtual counter. |
| SNTP | both | Working | QEMU | Network time; IANA timezones with DST. |

## Buses

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| PCIe (ECAM) | both | Working | QEMU, VBox | From ACPI MCFG. 64-bit BARs above 512 GiB are mapped correctly. |
| Apple PCIe | arm | Partial | HW (M2) | Port bring-up. |
| DART IOMMU | arm | Partial | HW (M2) | Apple's IOMMU, for AGX. |
| USB — xHCI | both | Working | QEMU, VBox, HW | Control, bulk, isochronous; hot-plug and hot-unplug teardown. |
| I²C (DesignWare/LPSS) | x86 | Partial | Tests | Master implemented; **unverified on hardware** — QEMU has no LPSS controller. Identification only ever reads, since the same bus carries the EC. |
| virtio-mmio / virtio-PCI | both | Working | QEMU | Shared transport for blk/net/input/snd/gpu/9p/serial. |

## Storage

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| virtio-blk | both | Working | QEMU | |
| NVMe | both | Working | QEMU, VBox | Namespaces via IDENTIFY CNS=2 — NSIDs are sparse on VirtualBox. |
| AHCI / SATA | both | Working | QEMU | Every HBA × every populated port, not just the first. |
| USB mass storage | both | Working | QEMU | Bulk-Only Transport + SCSI, over the shared xHCI bulk pair. |
| SDHCI (SD / eMMC) | both | Partial | Tests | The storage on tablets, Chromebook-class laptops and SBCs. Pure layer tested; controller path unverified on hardware. |
| GPT / MBR | both | Working | QEMU, VBox | Free-extent planning for install-alongside. |
| Volume encryption (C4VE v1) | both | Working | Tests, QEMU | Chitti data partition. Not LUKS2 — same idea, smaller, pure Rust. |

## Filesystems

| Filesystem | Access | Verified on | Notes |
|---|---|---|---|
| ext4 | read + write | Tests, QEMU | Default filesystem; images are e2fsck-clean by construction. |
| FAT12/16/32 | read + write | Tests, QEMU | ESP; the install-alongside writer preserves the existing loader. |
| exFAT | read + write | Tests | |
| NTFS | read only | Tests | |
| 9P (virtio-9p) | read + write | QEMU | Host shared folder (`-virtfs`). |
| VFS + mount table | — | QEMU | `fs/vfs.rs`, `fs/mount.rs`. |

## Input

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| USB HID keyboard | both | Working | QEMU, VBox, HW | Report-descriptor driven; software typematic (boot keyboards report press edges only). |
| USB HID mouse | both | Working | QEMU, VBox, HW | Shared report decode with the I²C touchpad. |
| USB touchscreen digitizer | both | Working | QEMU | |
| PS/2 (i8042) | x86 | Working | QEMU, VBox | Keyboard + aux mouse. |
| PL050 | arm | Working | QEMU | |
| virtio-input | arm | Working | QEMU | |
| HID-over-I²C touchpad | x86 | Partial | Tests | The touchpad on laptops from ~2016. Address from `_CRS`, descriptor register from `_DSM`. **Unverified on hardware.** |
| Bluetooth HID | both | Partial | Tests | USB HCI transport, classic host, PIN pairing. Radio path unverified. |
| Keyboard layouts | — | Working | QEMU | `/keyboard`. |

## Display

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| Firmware framebuffer | both | Working | QEMU, VBox, UTM, HW | Limine (x86), GOP via the stub (arm), m1n1 (Apple). Geometry comes from the firmware — the kernel holds no resolution. |
| EDID | both | Working | QEMU, VBox | Base block parsed; drives loader mode choice and per-monitor settings. |
| ramfb | arm | Working | QEMU | Fallback. |
| KMS — virtio-gpu | both | Working | QEMU (screendump) | Scans out of our own DMA pages; damage-driven flush. |
| KMS — VMSVGA | x86 | Working | QEMU (screendump) | VirtualBox/`vmware-svga`. The **MMIO BAR0 transport is unverified and declines** rather than risk mis-programming a real display; only the I/O-port path is proven. |
| KMS — Bochs VBE | x86 | Working | QEMU | QEMU's default adapter (`1234:1111`), so the stock VM can mode-set. |
| Multi-output selection | arm | Working | QEMU | Console-out marker → EDID → output 0. |
| GPU acceleration | — | **Absent** | — | See AGX below. No i915/AMD/Intel drivers. |

## Network

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| virtio-net | both | Working | QEMU | mmio and PCI. |
| Intel e1000 / e1000e | both | Working | QEMU | Covers I217/I218/I219 — the NIC in most business laptops. Unknown Intel IDs fall back here, logged as a guess. |
| Intel igb / igc | both | Working | QEMU | 82575–I350, I225/I226. Advanced descriptors, different ring registers. |
| Realtek r8169/8168/8125 | both | Partial | — | **Unverified on hardware** — QEMU models no r8169-family part. Per-chip register map is unit-tested. |
| USB CDC-ECM | both | Working | QEMU | Matched by interface class, not an ID list. |
| USB RNDIS | both | Partial | Tests | Android tethering; pure wire layer done. |
| ASIX / Realtek USB NICs | both | Identify only | — | Recognised then refused: they need per-chip setup and packet headers, and treating framed packets as raw would hand the stack garbage. |
| TCP/IP (smoltcp) | both | Working | QEMU | DHCPv4, DNS, ICMP, TCP/UDP, loopback, **dual-stack IPv6**. |
| TLS client | both | Working | QEMU | TLS 1.3 with real certificate verification against an embedded Mozilla root store. No CRL/OCSP. |
| TLS server | — | **Absent** | — | Client only, so service agents serve plaintext. |

## Wireless

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| WPA2-PSK supplicant | both | Working | Tests | Pure and vector-pinned: PBKDF2→PMK, PRF-384→PTK, EAPOL MIC, RFC 3394 unwrap. `/wifi psk` is checkable against `wpa_passphrase`. |
| 802.11 frame parsing | both | Working | Tests | Beacons/RSN. Attacker-controlled, so a lying length is refused, never clamped. |
| CCMP | both | Working | Tests | |
| Broadcom (brcm) | arm | Partial | — | Firmware load and shared-ring location for scan/connect. Cannot associate yet. |
| Intel (iwl) | x86 | Partial | — | Firmware handover and the alive notification. **Cannot scan or associate** — the configuration commands are large per-API-version structures and no emulator provides the part; writing them from memory would send well-formed garbage to a real radio. |
| Realtek RTL8852, Qualcomm/Killer | — | **Absent** | — | |

## Audio

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| Intel HDA | both | Working | QEMU, VBox | Built-in codecs; the VirtualBox path. |
| virtio-snd | both | Working | QEMU | PCM in/out over mmio and PCI. |
| AC'97 | x86 | Working | QEMU | |
| Sound Blaster 16 | x86 | Working | QEMU | Uses bounded DMA allocation for the 8237's real constraints (<16 MiB, no 128 KiB straddle). |
| USB audio (UAC 1.0) | both | Partial | Tests | Headsets and DACs over the isochronous OUT path. Descriptor layer tested. |

## Camera and video

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| UVC camera | both | Partial | Tests | Descriptor parse, PROBE/COMMIT, bulk or isoc pick, frame reassembly. `/camera` grabs stills. Unverified against a physical camera. |
| H.264 decode | both | Working | Tests (bit-exact vs ffmpeg) | Baseline + Main/High, CABAC, B-frames, deblocking. |
| AAC / MP3 / WAV | both | Working | Tests | |
| PNG / JPEG decode | both | Working | QEMU | Runs in **ring 3** — a malformed file becomes a status word from a discarded tenant. |

## Power and thermal

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| ACPI S5 poweroff | x86 | Working | QEMU | Real `SLP_TYPa` from the DSDT's `\_S5_`, not just the debug-exit port. |
| ACPI S3 suspend/resume | x86 | Partial | QEMU | Real-mode resume trampoline. |
| PSCI suspend / poweroff | arm | Working | QEMU | |
| Fixed-feature power button | x86 | Working | QEMU (`system_powerdown`) | Polled; write-1-to-clear on only the button bit. |
| Control-method button (GPE) | x86 | Working | QEMU | `PNP0C0C` — what most laptops use. |
| Battery (ACPI `_BST`/`_BIX`) | both | Partial | Tests | Reports remaining/last-full, sums multiple packs, `_PSR` for AC. **Unverified on hardware** — no emulator models an ACPI EC or battery. |
| Embedded controller | both | Partial | Tests | `PNP0C09` via `_CRS`; bounded spins, unclaimed-port rejection. Unverified. |
| Energy policy (EPB) | x86 | Working | QEMU | `/power performance\|powersave\|auto`. |
| CPU frequency scaling | — | **Absent** | — | No P-states or governor. |
| Thermal management | — | **Absent** | — | No trip points, no fan control. |
| Backlight / brightness | — | **Absent** | — | |

## GPU compute

| Component | Arch | Status | Verified on | Notes |
|---|---|---|---|---|
| AGX coprocessor bring-up | arm | Working | **HW (Apple M2, t8112)** | PMGR power-on → GFXHandoff → RTKit HELLO/EPMAP/START_EP → RUNNING. UAT is real 16 KiB ARMv8 paging with the G14 bit-39 TTBR select. |
| AGX compute dispatch | arm | Partial | HW (M2) | A hello-world dispatch returns the expected magic. |
| GPU-accelerated inference | — | **Absent** | — | `cortex` is CPU-only on every machine; the matmul path is not wired to the GPU. This is the largest available performance win. |

## Virtual-machine guest integration

| Component | Status | Verified on | Notes |
|---|---|---|---|
| VirtualBox VMMDev / HGCM | Working | **VBox** | PCI `80ee:cafe`. |
| VirtualBox shared clipboard | Working | **VBox** | Over HGCM. |
| virtio-serial / vdagent clipboard | Working | QEMU | |
| OSC 52 clipboard bridge | Working | QEMU, VBox | Works over a serial console with no guest driver at all. |
| virtio-9p shared folder | Working | QEMU | |

---

## Not implemented

Grouped by what it would take, so the list is actionable rather than a lament.

**Needs a machine with the part in it** — the code cannot be written honestly
without one, because a plausible-looking implementation would send well-formed
garbage to real hardware and report success:

- WiFi scan/associate on Intel and Broadcom (and thus any wireless connectivity).
- Realtek r8169 verification; Broadcom `tg3`, Atheros/Killer `alx`, Aquantia NICs.
- HID-over-I²C touchpad, ACPI battery and EC — all unverified.
- VMSVGA's MMIO BAR0 transport.
- SDHCI, UVC and USB-audio hardware paths.

**Needs design, not just porting:**

- GPU-accelerated inference (AGX compute → `cortex` GEMM).
- Real GPU drivers: i915, AMD, Apple AGX graphics.
- CPU frequency scaling, thermal management, backlight.
- TLS server (the client exists; the server side does not).

**Absent outright:** printing, and any cross-device sync or sharing.

---

## Verifying a change

```sh
cargo xtask build -arch x86_64 && cargo xtask build -arch aarch64
cargo xtask test -arch x86_64  && cargo xtask test -arch aarch64
make e2e                                   # boots the real kernel, drives the shell
```

Exercising specific hardware paths under emulation:

```sh
CHITTI_DISK_IF=ahci|nvme|virtio-blk cargo xtask run -arch x86_64
CHITTI_NIC=e1000|e1000e|igb|rtl8139|virtio-net-pci cargo xtask run -arch x86_64
cargo xtask run -arch aarch64 --uefi        # the PCI/ACPI path (no PCI on plain -kernel)
cargo xtask m1n1                            # Apple Silicon, bare metal
```

In the booted OS: `/lspci`, `/disks`, `/mounts`, `/battery`, `/power`, `/display`,
`/network`, `/wifi info`, `/bluetooth`, `/camera`, `/touchscreen`, `/keyboard`,
`/vbox diag`, and `/top`. Each reports how far bring-up got rather than failing
silently — "the driver did not bind" and "the driver bound and the device is
quiet" are different diagnoses, and the ktrace says which.
