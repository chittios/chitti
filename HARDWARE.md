# Hardware support

What ChittiOS drives on a real machine, what is written but has never met the
silicon it targets, and what is simply absent.

**Read this before booting on hardware you care about.** ChittiOS is developed
and verified against QEMU, VirtualBox and UTM. Several drivers here are written
from a specification and logged verbosely but have **never run on a physical
device**, because nothing in this environment emulates the part — that is a
different claim from "works", and this file keeps the two apart.

## Legend

| Mark | Meaning |
|---|---|
| ✅ | Implemented and exercised — by the e2e suite, the unit suite, or a real boot. |
| ⚠️ | Implemented, but **unverified on the hardware it targets** (no emulator models the part), or verified with a stated limit. |
| ❌ | Not implemented. A machine that needs this does not get the feature. |

A ⚠️ driver is not a stub: it is complete code written from the vendor
specification or from Linux's own headers, with every wait bounded and every
refusal logged. It has just never been proven. Treat it the way we treat
`r8169` — plausible, and unproven.

---

## Summary — booting a typical laptop today

| Subsystem | On a typical laptop |
|---|---|
| Console + shell | ✅ works, at the firmware's resolution |
| Keyboard, trackpad | ✅ basic input works (no gestures, no media keys) |
| Internal disk | ✅ **unless** the firmware is in Intel VMD / "RST" mode |
| Wired Ethernet | ✅ Intel, ⚠️ Realtek |
| **WiFi** | ❌ **no machine can join a network** (USB tether works) |
| **Bluetooth peripherals** | ❌ won't pair (no SSP, no BLE) |
| Audio | ✅ stereo (WAV/MP3); ⚠️ AAC mono; ❌ no USB headsets, ❌ no jack detect |
| Screen brightness | ❌ |
| Suspend / lid close | ⚠️ unverified (devices do come back) / ❌ no lid switch |
| Battery + charger reporting | ⚠️ unverified |

The honest short version: on real hardware ChittiOS is a **wired-network console
OS**. In a VM it is the whole system.

---

## Input

### Keyboard — ✅

| Transport | Status |
|---|---|
| USB HID boot keyboard (xHCI) | ✅ |
| PS/2 set-1 (x86 i8042) | ✅ |
| PS/2 set-2 (PL050, ARM) | ✅ |
| virtio-input | ✅ |

Every transport funnels through [`keymap/`](kernel/src/keymap/) — one decoder,
one modifier state, one caps-lock rule. Nine layouts with four levels each
(Base / Shift / AltGr / Shift+AltGr), dead keys, Compose, software auto-repeat
with an accelerating streak amplifier, and a romaji→kana IME. `/keyboard test
<layout> <keys>` asserts the tables from a running kernel.

**Not supported:**

- ❌ **HID consumer page** — volume, brightness, media transport, airplane mode
  and most laptop Fn-layer keys are simply not decoded. Only the boot-keyboard
  usage page is read.
- ❌ Report-protocol-only keyboards (a few gaming boards expose no boot protocol).
- ❌ Caps-lock / num-lock LEDs, keyboard backlight.
- ❌ Hangul, pinyin and kanji IMEs. Hangul composition is implemented and
  **deliberately refused** because the bundled font has no Hangul glyphs — it
  would compose correctly and render tofu.

### Mouse / pointer — ✅

| Transport | Status |
|---|---|
| USB HID (report-descriptor driven) | ✅ |
| PS/2 aux port, x86 (incl. IntelliMouse wheel) | ✅ |
| PL050 aux, ARM | ✅ |
| virtio-pointer / tablet | ✅ |
| HID-over-I2C | ⚠️ |

Absolute and relative motion, buttons, scroll wheel. USB and I2C share one
report decoder ([`xhci::feed_pointer_report`]) so a touchpad and a mouse cannot
drift apart.

**Not supported:** ❌ pointer acceleration or sensitivity settings; ❌ Bluetooth
mice in practice (see [Bluetooth](#bluetooth--️-wont-pair-with-modern-devices)).

### Touchpad — ⚠️ works as a plain mouse

[`drivers/i2c_hid.rs`](kernel/src/drivers/i2c_hid.rs) is the correct driver for
laptops from ~2016 onward, which have no PS/2 aux port. It locates the device by
asking the ACPI namespace which one claims `PNP0C50` (an I2C device cannot be
probed for — its address comes from `_CRS`), reads the descriptor register from
`_DSM`, validates the HID descriptor, then powers the device on and resets it.

**⚠️ Unverified on hardware** — QEMU emulates no LPSS/DesignWare I2C controller,
so none of this has met a real touchpad. Identification only ever *reads*,
because the same bus commonly carries the embedded controller.

**Not supported:**

- ❌ **Multi-touch and gestures** — first contact only. No two-finger scroll, no
  pinch, no three-finger swipe.
- ❌ Tap-to-click configuration, palm rejection, pointer settings.
- ❌ Native Synaptics/ELAN PS/2 protocols. An older laptop's touchpad falls back
  to the generic 3-byte PS/2 mouse protocol — it moves the cursor and clicks,
  and that is all.

### Touchscreen / digitizer — ⚠️

HID digitizers decode through the same pointer path (Tip Switch → left click,
absolute X/Y scaled to the framebuffer). First contact only; no gestures.
`/touchscreen` reports the live state.

---

## Display

### What always works — ✅

The kernel holds **no resolution of its own**. Geometry arrives from the
firmware and every layout is a ratio of whatever the panel turned out to be:

| Path | Platform |
|---|---|
| Limine GOP framebuffer | x86_64 UEFI/BIOS |
| UEFI GOP via the `stub/` bootloader | aarch64 |
| QEMU ramfb | aarch64 `-kernel` dev loop |
| m1n1-prepared framebuffer | Apple Silicon bare metal |

Mode selection is **EDID-preferred → keep the firmware's current mode → largest
advertised mode**, with `\chitti-display.cfg` on the ESP outranking all of them
because it is the one size a human typed on purpose. On a multi-output machine
the stub picks the display carrying the firmware's console-out marker — the one
you were watching boot messages on.

Display settings are stored **per monitor**, keyed on the panel's own EDID
vendor/product/serial, the way `monitors.xml` does it.

### Kernel mode setting — partial

| Backend | Status |
|---|---|
| virtio-gpu | ✅ verified by device screendump |
| VMSVGA (VirtualBox, QEMU `vmware-svga`) | ✅ x86 I/O-port path; ⚠️ ARM MMIO path **declines by design** — its register layout is unverified and acting on the guess mis-programmed a real display |
| Intel i915 | ❌ |
| AMD amdgpu | ❌ |
| NVIDIA | ❌ |
| Apple AGX | ⚠️ coprocessor boots to RUNNING on a real M2; **drives no display** |

### What this means on a real machine — ⚠️

With no bound KMS backend the compositor keeps the loader's framebuffer. That
is exactly the position Linux is in with `efifb`/`simpledrm` under `nomodeset`:

- ❌ **No runtime mode change.** `/display set` letterboxes a smaller logical
  desktop inside the physical framebuffer (rendered 1:1, so text stays sharp) —
  it does not reprogram the panel. `/display boot` records a preference the
  loader would have to apply, and says so rather than implying a reboot fixes it.
- ❌ **No external-monitor hotplug.** Plugging in a second display does nothing.
- ❌ **No acceleration.** Everything is CPU compositing into a linear framebuffer.
- ❌ **No backlight or brightness control anywhere.** There is no ACPI `_BCM`
  path and no GPU backlight path. You cannot dim a laptop screen.

Console legibility on a high-resolution panel is handled by font size instead:
`/display scale <1-4>|auto`.

---

## Storage

### Controllers

| Controller | Status |
|---|---|
| AHCI / SATA | ✅ every HBA on the bus × every implemented, populated port |
| NVMe | ✅ namespaces via IDENTIFY CNS=2 active list (never "walk NSIDs until empty" — they are sparse on real machines) |
| virtio-blk | ✅ mmio and PCI |
| USB mass storage (BOT/SCSI) | ✅ read **and** write, hot-plug with mount prune |
| Apple ANS2 NVMe (via DART) | ⚠️ Apple Silicon bare metal |
| **SD / eMMC (SDHCI)** | ❌ |
| Intel VMD / RST | ❌ |

Exercise the real-hardware paths in QEMU with
`CHITTI_DISK_IF=ahci|nvme|virtio-blk cargo xtask run -arch x86_64`.

**⚠️ Intel VMD is the one that bites.** Many 2020+ Dell and Lenovo machines ship
with the firmware in "RAID"/"Intel RST" mode, which hides the NVMe behind a VMD
bridge on a separate PCI domain that is not visible in the main ECAM. ChittiOS
finds **no disk at all** on such a machine. The workaround is to switch the
firmware's SATA/NVMe mode to **AHCI**; there is no driver-side fix today.

**❌ No eMMC/SDHCI** means tablets, Chromebook-class machines and most SBCs have
no storage under ChittiOS whatsoever.

### USB external disks — ✅ with limits

- LUN 0 only, and `probe_nth(0)` — **one** mass-storage device at a time.
- 512-byte logical blocks only; a 4Kn drive is refused rather than mis-read.
- No bulk-stall recovery: a stalled endpoint fails closed on the CSW.
- ✅ **USB hubs are enumerated to the full 5 tiers USB allows**, so a drive
  behind a dock behind a monitor's built-in hub is reached. A position the xHCI
  route string cannot express is refused and logged rather than truncated into
  one naming a different device.

### Filesystems

| Item | Status |
|---|---|
| ext4 | ✅ read + write (the default filesystem) |
| FAT12/16/32 | ✅ read + write |
| NTFS | ⚠️ read only |
| GPT / MBR | ✅ |
| 9P (host shared folder) | ✅ |
| C4VE encrypted volumes | ✅ |
| btrfs, XFS, APFS, exFAT | ❌ |

`/install` writes a fresh GPT (whole-disk) or `/install alongside` adds a loader
to an existing ESP without touching the partition table. `/install plan` is
read-only and reports what it would do.

---

## Networking

### Wired Ethernet

| Driver | Parts | Status |
|---|---|---|
| `e1000` | Intel 82540–82547 | ✅ |
| `e1000e` | Intel 82571 … **I217/I218/I219** | ✅ — the NIC in most business laptops |
| `igb` | Intel 82575–I350, I210/I211 | ✅ |
| `igc` | Intel I225/I226 2.5GbE | ✅ |
| `virtio-net` | mmio + PCI | ✅ |
| `r8169` | Realtek RTL8168/8111/8125 | ⚠️ **unverified** — QEMU models no r8169-family part |
| `rtl8139` | Realtek RTL8139 | ❌ recognised, deliberately not implemented |
| Broadcom `tg3` | many Dell/HP desktops and older laptops | ❌ |
| Atheros / Killer `alx` | | ❌ |
| Aquantia AQC | 2.5/10GbE on newer boards | ❌ |
| Marvell / Yukon | | ❌ |

A NIC is claimed by **vendor + device ID**, never vendor alone — all Intel
Ethernet reports `8086`/class `02:00:00` while the families are
register-incompatible. Unknown Intel IDs fall back to `e1000e` (the only
open-ended family) with the ID logged as a guess. Test the dispatch against
every family QEMU can emulate with `CHITTI_NIC=e1000|e1000e|igb|rtl8139|
virtio-net-pci cargo xtask run -arch x86_64`.

❌ **No MSI or MSI-X anywhere.** Every driver in this OS polls. That is an
architectural choice, not an oversight — it costs latency and CPU, and it means
no driver depends on interrupt routing being correct.

### USB Ethernet / tethering

| Shape | Status |
|---|---|
| CDC-ECM | ⚠️ implemented, **unverified** — QEMU emulates only RNDIS |
| RNDIS (Android USB tethering, QEMU `usb-net`) | ✅ |
| ASIX AX88179 | ❌ recognised and refused (needs per-chip register setup) |
| Realtek RTL8152 | ❌ recognised and refused |

Tried **last** in `autodetect`, after virtio and PCI, so a built-in NIC always
wins. An iPhone tether presents CDC-ECM (so it *should* work, unproven);
**Android tethers over RNDIS**, which is implemented — bring-up over the control
pipe (`INITIALIZE` → MAC query → packet filter), a 44-byte per-packet header, and
several frames per transfer. RNDIS is identified by its *control* interface's
class triple, because its data interface is class `0x0A` exactly like CDC-ECM's.

### WiFi — ❌ no machine can join a network

This is the largest gap in the OS. The parts that exist are real and complete:

| Layer | Status |
|---|---|
| WPA2-PSK supplicant (PBKDF2 → PMK → PTK, EAPOL MIC, RFC 3394 key unwrap) | ✅ pure, pinned to the published 802.11i vectors |
| 802.11 frame + beacon + RSN element parsing | ✅ |
| `/wifi psk <ssid> <passphrase>` — derive and print a key | ✅ (checkable against `wpa_passphrase` on any Linux box) |

The radios are what is missing:

| Radio | Status |
|---|---|
| Intel `iwlwifi` (AX200 and later) | ⚠️ **bring-up only** — resets the device, hands over firmware, waits for the *alive* notification and checks its status word, reads the MAC, sends one **read-only** command. ❌ **Cannot scan. Cannot associate.** No 802.11 data path. |
| Broadcom FullMAC (Apple Silicon) | ⚠️ blocked — BAR2/TCM reads take an external abort |
| Realtek RTL8852 / RTL8821 | ❌ |
| MediaTek MT7921 / MT7922 | ❌ |
| Qualcomm Atheros ath10k/11k/12k | ❌ |

Bring-up is **command-driven** (`/wifi up`), never automatic at boot: an
untested driver should not touch a device just because the machine started. The
missing pieces for Intel are the receive path, the command round-trip, and then
802.11 association — each of which needs a machine with the part in it, because
no emulator provides one and code written from memory would send well-formed
garbage to a real radio and report success.

**Plan for a wireless machine: use Ethernet, or a USB Ethernet dongle
(CDC-ECM), or an iPhone tether.**

### Protocols — ✅

Full TCP/IP on vendored smoltcp: DHCPv4, static IP, DNS, ICMP, TCP/UDP,
loopback (a second interface with its own socket set). HTTP/1.1 client with
streaming, **HTTPS with real certificate verification** against an embedded
121-root Mozilla store (RSA PKCS#1 v1.5 + PSS on `crypto-bigint`, ECDSA P-256/
P-384), WebSockets, MCP, SNTP, SSH version exchange.

❌ Out of scope and documented as such: CRL/OCSP revocation. ❌ No IPv6.

---

## Bluetooth — ⚠️ won't pair with modern devices

| Layer | Status |
|---|---|
| USB HCI transport (class `E0/01/01`) | ✅ — dongles, and most laptop combo cards, which are internally USB |
| HCI command/event codec | ✅ pure, unit-tested |
| Classic BR/EDR inquiry (`/bluetooth scan`) | ✅ |
| L2CAP + HID profile (PSM 0x11/0x13) | ✅ |
| Durable bond store | ✅ |
| **Secure Simple Pairing (SSP)** | ❌ |
| **Bluetooth Low Energy (BLE)** | ❌ |
| **A2DP / AVDTP / SBC (audio)** | ❌ |
| UART / SDIO HCI transport | ❌ |

Only **legacy PIN pairing** is implemented. SSP has been mandatory since
Bluetooth 2.1 (2007), so in practice **a modern mouse, keyboard or headset will
not pair.** And most current peripherals are BLE-only, which is absent
entirely.

Bluetooth today is best understood as staged infrastructure — the transport,
codec and HID plumbing are real and the pairing model is a generation behind
the devices people own.

---

## Audio

### Controllers

| Driver | Platform | Status |
|---|---|---|
| Intel HDA | real Intel/ARM machines, VirtualBox (x86 *and* ARM), QEMU `intel-hda` | ✅ |
| virtio-snd | QEMU (mmio + PCI) | ✅ |
| AC'97 | x86 legacy | ✅ |
| Sound Blaster 16 | x86 legacy | ✅ |
| **USB Audio Class** | USB headsets, DACs | ❌ |
| Bluetooth A2DP | | ❌ |

The HDA driver does a genuine codec-graph walk rather than guessing: it ranks
the output pin complexes by `CONFIG_DEFAULT` (speaker, then headphone, then
line-out, refusing pins the board wired nowhere and any SPDIF/HDMI pin belonging
to the graphics device), then **searches the graph** from that pin back to a
DAC — because the common shape is `pin ← mixer ← dac` and pointing the pin
straight at "the first DAC we saw" leaves the codec mute. Every widget along the
path gets its input select pointed at the next hop, its amp unmuted and power
set to D0.

### The limits — ⚠️

- ✅ **Stereo plays** on HDA for WAV and MP3 — the channel count rides on
  `Audio` and the decoders keep their interleaving. Anything wider than stereo
  is folded to stereo rather than to its first two channels, so a 5.1 track
  keeps its centre.
- ⚠️ **AAC still folds to mono**, so a video's audio track is mono. Its
  downmix is a weighted BS.775 matrix rather than an average, so stereo means
  writing a second matrix into a decoder that is bit-exact-validated against
  Symphonia — deliberately not disturbed for a channel count.
- ⚠️ Drivers other than HDA (virtio-snd, AC'97, SB16) fold to mono via the
  default `SndDevice::play_ch`, so they are unchanged.
- ❌ **No jack detection** (no unsolicited responses / pin sense). Plugging
  headphones in does not switch output away from the speakers.
- ⚠️ **Volume is software-only** — a gain applied in `sound::play`, so every
  backend gets it without per-driver wiring (↑/↓ on the media tabs, plus mute).
  The codec's own amps are set once to a 0 dB offset and never touched again, so
  there is no hardware mixer and no per-stream levels.
- ❌ **No USB headsets or USB DACs** — the USB Audio Class is not implemented.

### Decoders — ✅

WAV (PCM 8/16/24/32-bit + float32, any channel count), MP3 (a no_std port of
minimp3, validated ±1 LSB against its own scalar decode), AAC-LC and HE-AAC
with SBR/PS. `/open <file>.wav|.mp3|.aac` plays with transport controls.

---

## Power and battery

Everything in this section that touches a laptop is **⚠️ unverified**: QEMU
emulates no ACPI embedded controller, no battery and no AC adapter.

### Working — ✅

| Capability | Status |
|---|---|
| ACPI S5 poweroff | ✅ real `SLP_TYPa` from the DSDT's `\_S5_`, not a QEMU debug port |
| ACPI fixed-feature power button | ✅ verified in a VM via QEMU's `system_powerdown` |
| CPU idle (`hlt` / `wfi`) | ✅ — the single biggest real-world power win here |
| Local APIC timer calibrated against HPET, PIT fallback | ✅ |

### Written, unverified — ⚠️

| Capability | Status |
|---|---|
| Battery percentage | ⚠️ ACPI `_BST` with `_BIX`/`_BIF` last-full capacity (never design capacity, or a worn pack reads permanently below 100%), evaluated through the AML interpreter over the embedded controller. Multiple packs summed; `_STA` bit 4 checks a bay is populated. |
| AC adapter / charging | ⚠️ `ACPI0003._PSR`. Reported as *unknown* rather than guessed when no adapter device exists — once a pack is full, `_BST` reports neither charging nor discharging, so `_PSR` is the only signal that distinguishes plugged-in from running down. |
| Embedded controller | ⚠️ `PNP0C09._CRS`-driven, bounded spins, `0xff` status rejected as an unclaimed port before any command is written |
| Suspend to RAM | ⚠️ x86 ACPI **S3** (real-mode resume trampoline via the FACS waking vector); aarch64 **PSCI `SYSTEM_SUSPEND`** |

`/suspend plan` is read-only and enumerates every precondition; the transition
**refuses** unless all of them hold, because a suspend that does not resume is
the worst failure this kernel can have.

### Missing — ❌

- ⚠️ **Resume re-probes the polled subsystems**: xHCI (so a USB keyboard comes
  back), the NIC, and the sound device, alongside the APIC/GIC + timer, the
  i8042 and the I2C-HID touchpad. Disks need nothing — they re-probe on first
  access. Two documented costs: each re-probe **leaks its previous instance's
  DMA pages** (bounded per resume; the frame allocator has no free path), and
  the NIC comes back **unaddressed**, so a pre-suspend DHCP lease is not
  re-asserted and `/network dhcp` is needed. `/suspend plan` says so before you
  commit. All of it is still **unverified on real hardware**.
- ❌ **Lid switch** (`PNP0C0D`) — closing a laptop lid does nothing.
- ❌ **Thermal zones** (`_TMP`), fan control, thermal throttling. Under
  sustained load the machine relies entirely on firmware and hardware thermal
  protection.
- ❌ **Control-method (GPE) power button** — reported when the FADT declares one
  (flags bit 4), but GPE dispatch is unimplemented. Many laptops use this rather
  than the fixed-feature button.
- ❌ **CPU frequency scaling.** x86 gets `IA32_ENERGY_PERF_BIAS` (MSR `0x1B0`)
  and three human modes; there is no HWP, no `_PSS`, no cpufreq governor.
  aarch64 records the policy and applies nothing.
- ❌ USB-C Power Delivery, charge thresholds, battery health limits.
- ❌ Screen backlight (see [Display](#display)).

---

## Other peripherals

| Item | Status |
|---|---|
| USB webcam (UVC) | ⚠️ descriptor parse, PROBE/COMMIT, MJPEG/YUY2 frame assembly, `/camera grab` → one still. Unverified against a physical camera. |
| Host shared folder (virtio-9p) | ✅ verified byte-exact both directions |
| Host clipboard — OSC 52 over serial | ✅ (needs the console attached to a terminal) |
| Host clipboard — SPICE vdagent over virtio-serial | ⚠️ link verified against QEMU's own trace; ❌ **does not reach the macOS pasteboard** (QEMU's `cocoa` display registers no clipboard peer — use OSC 52 or VirtualBox there) |
| VirtualBox guest integration (HGCM) | ❌ transport only; the clipboard and shared-folder services are not implemented |
| Printers, scanners, game controllers, MIDI, TPM, fingerprint readers, smartcards | ❌ |

---

## Firmware and discovery

Discovery follows what real firmware does, never an emulator quirk:
ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors, PrimeCell IDs, EDID,
device tree. ACPI tables are explicitly mapped on x86 (they live in
firmware-reserved regions outside Limine's HHDM, so both the physical address
and its translation are unmapped and touching either faults the boot).

The AML interpreter evaluates a **fail-closed subset** — an unsupported opcode
returns nothing rather than a guessed value, because an evaluator that invents
an integer is worse than the validated default it would replace.

Interrupt-controller bases come from the device tree where there is one and from
the ACPI MADT where there is not — which matters because those are *different
real platforms*: QEMU `virt` boots with an FDT, while VirtualBox-ARM, UTM and
real SBSA machines boot the UEFI stub with none.

---

## Reporting a hardware failure

The boot log is the diagnosis. Every driver here logs what it found, what it
refused and why, through `ktrace` — a refusal is always a named reason, never
silence. When filing an issue please include:

1. `/lspci`, `/disks`, `/mounts`, `/network`, `/battery`, `/power` output.
2. The serial boot log (`-serial mon:stdio` under QEMU; a USB-serial adapter or
   a photo of the screen on bare metal).
3. The exact machine — vendor, model, and for a NIC/WiFi/audio failure the PCI
   vendor:device ID from `/lspci`.

An unrecognised PCI ID is the single most useful thing you can send: several
families here (`e1000e` in particular) grow by ID and the log prints the ID it
guessed at.
