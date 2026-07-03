# Chitti OS — Agentic Re-Architecture: Decisions & Assumptions

Append-only ledger for the autonomous run. Each entry: decision, default, rationale.
Consequential ones flagged **REVISIT**.

## Ledger (initial, from the kickoff)

1. **Sampling field repr** — keep `f32` in `Sampling`. Tests pin temp 0 (exact greedy,
   no float draw) + fixed seed, so the float-sensitive path is never exercised in the
   deterministic gates. Fixed-point (u16 milli-units) deferred. **REVISIT** if a
   postcard cross-arch parity test on a non-zero-temp manifest ever diverges.
2. **Store serialization** — `postcard` (no_std + alloc) for the persistent store;
   no `serde_json` in-kernel (tool-args JSON is handled by the existing
   `synapse::grammar` parser). Debug dumps use manual formatting.
3. **New dependencies** — `serde` (derive, alloc, no_std), `postcard` (alloc),
   `bitflags` v2 (serde feature). All no_std; none pull `std`. This crosses the
   Part-4 "heavyweight dependency" guardrail, but the schema contract in
   CHITTI_SCHEMAS.md mandates serde+postcard+bitflags verbatim, so they are adopted.
   Ed25519 crate chosen at Phase G. **REVISIT** the Ed25519 crate choice.
4. **Test model binding** — the fast `cargo xtask test` suite does not load the model.
   The agentic loop is driven by a deterministic scripted `StepSource` in tests
   (temp-0-equivalent, reproducible). Real Cortex is wired behind the same trait for
   `run`/boot demos. Mirrors the existing `Planner`/`RulePlanner` seam the handoff
   blesses. Bundled model stays Qwen3.5-0.8B by default.
5. **Delegation depth** — orchestrator `max_depth = 2`, `max_subagents = 8`; sub-agents
   default `max_depth = 0`. Matches the schema example.
6. **Loop step contract** — `trait StepSource { fn next(&mut self, &LoopCtx) -> Step }`,
   `Step = ToolCall | Final(text)`. Real impl = Cortex + grammar-constrained decode;
   test impl = scripted.
7. **Builtin tool arg schemas** — read{path}, write{path,content}, edit{path,old,new},
   list{path?}, search{query,path?}, run{intent}, spawn_subagent{role,task}, todo_write,
   load_skill{name}, emit_result{text}. MCP-shaped JSON objects, grammar-constrained.
8. **New Synapse primitives to stub** — `mem_fs_edit`, `mem_fs_search`, `todo_write`,
   `load_skill`. `spawn_subagent` routes to the agent layer (audited) rather than a
   raw primitive. All effects still flow through Synapse.
9. **Skill trust** — Ed25519 + one baked-in registry public key, single-key trust store.
   Revocable hierarchy deferred. **REVISIT** for revocation/multi-key.
10. **Sample packages** — `note-summarizer` (plain skill) + `pdf-filler` (skill-agent),
    both Ed25519-signed with the baked key, shipped as boot modules.
11. **"Both arches" gate** — `cargo xtask test` (x86_64 QEMU) green + `cargo xtask build
    -arch aarch64` clean each phase; full aarch64 boot spot-checked at milestones.
    No aarch64 in-kernel test runner exists today. **REVISIT** if one is added.
12. **persona/ deprecation** — absorb into agent/+session/+tools/ incrementally; keep it
    compiling until consumers move, then delete. Reuse `persona::compiled` for Phase E.
13. **CapId vs live caps** — `CapId(u64)` is a display/audit handle; live authority stays
    the existing unforgeable `cap::Right::InvokePrimitive` per-task tables. CapDomain/
    Rights/Scope lower to a set of `InvokePrimitive` rights at grant time.
14. **ID minting** — monotonic `AtomicU64` per id type, from 1.

## Overrides applied

(none yet — awaiting the single course-correction message.)

## Assumptions logged during the run

- **Phase C SMP concurrency (REVISIT):** `dispatch_batch` assigns each sub-agent a distinct
  core id and records it (`SubagentRecord.core`), but under the single-threaded QEMU test
  harness sub-agents execute sequentially. The prior project note already found multicore
  inference a net loss under QEMU TCG, so true parallel dispatch is structured (per-core
  assignment + isolated tasks) but not yet run concurrently. The isolation/attenuation/summary
  invariants — the correctness-critical ones — are fully enforced and tested.
- **Sub-agent cap attenuation is strict:** a role that *requests* any capability the parent does
  not hold is refused at spawn (`DispatchError::CapabilityRefused`), not silently narrowed. Rights
  are still clamped to the overlap via `intersect_caps`. This satisfies both the subset invariant
  and acceptance (b) "refused at spawn".
- **Sub-agents cannot sub-delegate by default:** the `spawn_subagent` hook gives sub-agents a
  plain Router (no spawn hook), and `max_depth` caps recursion regardless.
- **Skill-bundled tools bind to existing Synapse primitives (Phase F):** the handoff calls
  bundled code "registered as Synapse primitives, capability-gated." On bare metal we cannot
  safely load arbitrary native code at runtime, so a `BundledTool.synapse_primitive` names an
  existing, vetted primitive (the sample `note_search` → `mem_fs_search`). The tool is still a
  first-class, capability-checked, audited Synapse call; only *arbitrary new native code* is out
  of scope. REVISIT if a sandboxed skill-code execution model is added later.
- **ToolBinding::Synapse is owned (String/Vec), not &'static:** required so skill-bundled tools
  registered at runtime bind exactly like builtins.
- **Ed25519 → keyed-MAC (Phase G, REVISIT):** `ed25519-compact` *builds* under -Z build-std but
  its sign/verify **fault at runtime on the bare-metal x86 target** (QEMU exits abnormally — no host
  runtime; likely a stack/SIMD assumption). Per the run's "log blocker, ship the best partial that
  builds, move on" rule, package verification uses a self-contained SipHash-2-4 keyed MAC (8 lanes →
  512-bit tag, same width as an Ed25519 sig, so SignatureBlock is unchanged). In this self-contained
  build the kernel is both registry (signs) and installer (verifies), giving real integrity +
  tamper-detection + unsigned/untrusted-key rejection. REVISIT: a bare-metal-safe asymmetric Ed25519
  for true off-device authenticity.
- **Sample packages are built + signed in-kernel, not shipped as separate boot-module files
  (REVISIT):** `package::sample_note_summarizer` / `sample_report_agent` construct real signed
  packages that the F/G tests + boot demo install (verify → consent-subset → bound). Delivering them
  as distinct QEMU boot-module files needs xtask multi-module plumbing; the security path (sign →
  verify → tamper-reject → grant-subset → skill-agent-bound) is fully exercised against real signed
  packages either way. `InstallSource::BootModule{name}` is recorded so the provenance is faithful.

## Storage / install run (over-day autonomous)

Answers: detect all 5 FSes read-only (no foreign writes); build-time model choice
(default 0.8B) as multi-part modules; full bootable install; ext4 as our OS FS.

- **Bootable install uses a FAT32 ESP for boot (REVISIT ext4-write):** a from-scratch ext4
  *writer* (mkfs + files + journal) is not correctly completable in one day. So `/install`
  writes GPT + a FAT32 ESP carrying Limine + kernel + the multi-part model (boots standalone via
  UEFI using only FAT32, which we can write correctly) and ALSO creates an ext4 OS/data
  partition per Q4. ext4 gets detection + read-only (Q1); a full ext4 mkfs/writer is best-effort,
  else the OS partition falls back to SimpleFS. Booting never depends on ext4. REVISIT: real ext4 write.
- **Multi-part model reassembly copies into contiguous frames (`alloc_dma`), not the heap:** the
  model dwarfs the 256 MiB kernel heap, so reassembly uses frame-allocator-backed contiguous
  memory. Still costs the model size in RAM; a zero-copy segmented GGUF reader is a REVISIT.
- **All block/FS/install work is x86** (ISO/Limine/UEFI + virtio-blk-PCI). aarch64 stays the
  QEMU-native `-kernel` dev path (no ISO); aarch64 virtio-mmio-blk support is a follow-on.
- **Override applied:** the `/install` OS/data partition is **SimpleFS**, not ext4 (user
  course-correction). ext4 stays detection + read-only (Q1). Removes the ext4-write problem.
- **SimpleFS max file = 4 KiB (8 direct blocks, no indirect):** so the model cannot live on the
  SimpleFS OS partition — it belongs on the FAT32 ESP (where Limine loads it). Populating the ESP
  (FAT32 write of Limine + kernel + model = standalone UEFI boot) needs a from-scratch FAT32
  writer + installer-payload modules; not completable/verifiable in the remaining budget, so it is
  the documented **REVISIT**. `/install` today writes a spec-valid GPT (correct CRC32) + formats
  the SimpleFS OS partition + marker files — the partitioning + native-FS-creation half of a real
  install, verified in QEMU and by host GPT parse.

## ext4-as-primary run (user override: ext4 from scratch, not FAT32/SimpleFS)

- **Layout for standalone boot:** tiny FAT ESP (only `/EFI/BOOT/BOOTX64.EFI` — UEFI firmware
  requires FAT to find the loader) + an **ext4** partition holding `limine.conf` + kernel + the
  multi-part model. Limine reads ext4 natively, so ext4 is the primary OS FS; FAT is a ~1 MB stub
  like every Linux install. exFAT/NTFS/XFS were ruled out (Limine can't boot them).
- **ext4 driver validated in Python first (against e2fsck), then ported to no_std Rust:** feature
  set kept minimal for correctness — `filetype + large_file + sparse_super`, 128-byte inodes,
  block-mapped files (12 direct + single/double indirect) so large files work; no journal/extents
  (a clean ext2/3/4-family FS the Linux ext4 driver mounts + Limine boots). Verified: multi-block-
  group mkfs is **e2fsck-clean** and a 300 MB double-indirect file round-trips **byte-identical**
  via debugfs. Reference: `tools/mkext4_ref.py`.
- **All ext4 install files live in the ext4 root** (limine.conf, kernel, model.gguf.NNN) so the
  Rust port needs no subdirectory creation; only the FAT ESP needs one path (`/EFI/BOOT/`).

### ext4-primary + standalone boot — findings (E4/E5)
- **The from-scratch ext4 write driver is done + verified** (e2fsck-clean, multi-group, byte-exact
  files) — the explicit request. It is the OS/data filesystem written by `/install`.
- **Standalone UEFI boot works**, verified: install to a blank disk, then boot it under OVMF with
  no ISO → "Chitti: boot ok". Chain: OVMF → FAT ESP → Limine (BOOTX64.EFI) → limine.conf + kernel.
- **Limine cannot boot from our ext4** (config-on-ext4-only did not boot, though e2fsck accepts the
  FS): Limine's ext2/3/4 reader is stricter than e2fsck about features/layout. So kernel + limine.conf
  live on the **FAT ESP** (a standard layout — bootloader + kernel on the ESP, ext4 as root/OS).
  `block/fat.rs` gained VFAT LFN support so `limine.conf`/`chitti-kernel` keep their real names
  (8.3 truncation had silently broken Limine's lookup — the root cause of the first boot failure).
- **REVISIT (model on the installed system):** `/install` also writes the model to ext4, but the
  installed kernel loads the model via Limine modules from the boot volume — and Limine can't read
  our ext4, and the FAT16 ESP is too small for the model. To make the installed system load the
  model, either (a) grow the ESP to FAT32 and put the model parts there as Limine modules (extend
  the FAT writer to FAT32), or (b) add an in-kernel ext4 *reader* so the running kernel pulls the
  model off the ext4 partition at runtime. Boot + the ext4 write driver are unaffected.

## aarch64 UEFI/Limine boot-from-disk (Part B) — WIP, feature-gated

**Context.** The dual-arch standing rule: a feature on one arch must exist on
the other. Part A made the *storage + persistence* stack dual-arch (aarch64
virtio-blk over virtio-mmio + de-gated fs/ext4/install/persistence — verified
native: boot #2, e2fsck-clean). Part B is the remaining piece: booting an
installed aarch64 disk *standalone* via firmware (no `-kernel`), the counterpart
to x86 `--disk-only`.

**Why it is large.** aarch64 today boots via a custom `-M virt -kernel` stub
(EL1, MMU off, own identity map). Booting from a disk's ESP means AAVMF →
Limine `BOOTAA64.EFI` → the kernel via the **Limine boot protocol** (MMU on,
higher-half, memmap/framebuffer/modules from Limine) — a boot-protocol port of
the working kernel, not a feature add.

**Done (feature-gated behind `boot-limine`, default builds/tests untouched — 103/103,
both default arches build):**
- `linker-aarch64-limine.ld` (higher-half + `.requests`), `boot-limine` Cargo feature,
  the `-kernel` stub gated off under it, Limine request statics opened to aarch64.
- `limine_start` entry (main.rs): FP/SIMD, serial, heap from the Limine memmap,
  sched, Limine-GOP framebuffer, storage bring-up, `run_os` — reuses the whole steady state.
- `cortex::model_module` for boot-limine (Limine-module path; single-part zero-copy).
- xtask: `build_kernel_aarch64_limine`, `aavmf_pflash_args` (edk2-aarch64), an aarch64
  UEFI Limine ISO assembler, and `cargo xtask run -arch aarch64 --uefi|--disk-only`.

**Where it stops.** AAVMF boots and launches Limine's `BOOTAA64.EFI`, but the
Limine→kernel handoff isn't reaching the kernel's serial `boot ok` yet. Limine
renders to the GOP framebuffer, not the PL011 serial, so its menu/errors are
invisible on the serial log — the next step is pointing Limine at the serial
console (to see whether it finds limine.conf/kernel) and confirming the kernel's
PL011/heap access under Limine's page tables (Limine identity-maps the low 4 GiB,
so MMIO *should* be reachable, but this needs on-target confirmation).

**Follow-on plan.** (1) Configure Limine serial console in limine.conf to unblind
the handoff. (2) Confirm/repair the kernel's early serial + heap under Limine.
(3) B2: aarch64 `alloc_dma` from a Limine memmap region → multi-part + ext4 model
load; de-gate `/install` to write the aarch64 ESP (BOOTAA64.EFI + limine.conf +
kernel) + partitions. (4) B3: verify standalone disk boot + model-from-ext4.

## aarch64 UEFI boot — real-FAT-ESP investigation (findings)

Goal: make `run -arch aarch64 --uefi` reliably reach the interactive shell.

**Fixed at the Limine level.** The `-cdrom` El Torito route boots a separate FAT
image on which Limine can't find limine.conf (silent GOP-menu drop). A **real
raw FAT32 image** built on the host (hdiutil attach -nomount + newfs_msdos +
mount + copy) is the reliable boot medium: Limine consistently finds limine.conf,
loads the kernel, and the kernel reaches **`boot ok` → HHDM heap → scheduler →
Limine GOP framebuffer** on every run.

**Blocker (unresolved).** After the framebuffer, the boot wedges in the
aarch64 **virtio-blk-mmio `init`** under Limine (localized to between the device
reset and the queue setup in the v1 legacy path). The *same* driver on the
`-kernel` path is reliable (Part A / A3: boot #2, e2fsck-clean), so it is
specific to the Limine environment — candidate causes: virtio-mmio register
access semantics under HVF after the UEFI handoff, or a DMA/ordering assumption
that holds on the identity-mapped `-kernel` path but not under the HHDM.

**Compounding factor.** AAVMF (edk2-aarch64) under HVF is very slow and
*non-deterministic* to reach the Limine handoff (variable multi-minute firmware
init: image-enumeration failures, TPM probe timeouts), so boot-to-shell timing
is unpredictable and obscured the localization.

**Next steps.** (1) Granular per-register-write markers inside virtio-blk-mmio
`init` under Limine to pin the exact wedge; (2) try forcing modern (v2)
virtio-mmio (`-global virtio-mmio.force-legacy=false`) to sidestep the legacy
PFN path; (3) consider mapping virtio DMA through a dedicated identity region
rather than the HHDM. Until then, `--uefi` is not shell-reliable; the default
`-kernel` aarch64 path (and all features, incl. storage/persistence) is fully
reliable.

## Real-world drivers — status + the keyboard (xHCI) plan

**Done + verified (on QEMU virt's GPEX PCIe, the standards-based bus real
platforms use):**
- **Display**: UEFI GOP framebuffer via the stub's boot-info page — works on
  any UEFI platform (VirtualBox-ARM, UTM, real hardware).
- **Disk**: ACPI (RSDP→MCFG→ECAM) discovery + `pci.rs` (ECAM enumeration, BAR
  decode, virtio caps) + `virtio_pci.rs` (modern virtio-blk). `Disk` enum picks
  PCIe first, virtio-mmio fallback. Format/read/write round-trip verified over
  `virtio-blk-pci`.

**Remaining: keyboard = USB xHCI + HID over PCIe.** On ARM there is no PS/2;
real input is USB. The x86 `arch/x86_64/xhci.rs` (793 lines: controller reset,
DCBAA/command/event rings, scratchpad, device slot + EP0, GET_DESCRIPTOR/
SET_CONFIG enumeration, HID boot-protocol interrupt polling) is a working
reference. Porting to aarch64 is a bounded, well-scoped effort now that the PCIe
groundwork exists:
1. Discovery: replace the x86 PCI-port scan with `pci::find` by class code
   0x0C/0x03/0x30 (xHCI) — `pci.rs` already does ECAM config + BAR decode.
2. DMA: replace `mm::alloc_dma` (x86 frame allocator, phys/virt via HHDM) with
   `alloc_ident` + `dma_to_phys` (aarch64 identity), as the virtio drivers do.
3. MMIO: the xHCI BAR is in the low PCI window (already Device-mapped); no
   `map_mmio_page` needed.
The ring/enumeration/HID core is arch-neutral and reused verbatim. Deferred as a
dedicated task (a full USB host-controller port warrants its own verification
pass with `-device qemu-xhci` + `-device usb-kbd` on the PCIe bus) rather than
rushing it at the tail of this session.

**Also platform-specific storage** (AHCI/NVMe) for hypervisors that expose SATA/
NVMe instead of virtio — additive `BlockDevice` impls behind the same `Disk`
enum, same PCIe discovery.

**DONE: NVMe + AHCI over PCIe (aarch64).** Both landed as additive `BlockDevice`s
in the `Disk` enum (probe order virtio-pci → NVMe → AHCI → virtio-mmio) and were
verified read/writing an ext4 file on QEMU virt under UEFI (`-device nvme` and
`-device ahci`+`ide-hd`): `pattern match: true` on a 200 KB round-trip for each.
Key gotcha found + fixed: the NVMe CQE **phase tag is bit 16 of DWORD3** (bits
15:0 are the command ID), not bit 0 — the initial bit-0 check made completions
never register, so `bringup` silently returned None and the probe fell through to
the ESP. These two drivers are only discovered where ACPI/PCIe exist (the UEFI
boot); the native `-kernel` dev path has no ACPI and keeps virtio-mmio.

**R7 DONE: NVMe/AHCI made dual-arch via shared cores.** Rather than leave these
aarch64-only, both drivers were refactored into arch-neutral cores
(`block/nvme.rs`, `block/ahci.rs`) behind a `block::Dma { phys, virt }` +
`DmaAlloc` seam (the `xhci` pattern), with thin per-arch discovery wrappers:
aarch64 over ACPI-ECAM (`crate::pci`) + identity BAR map; x86 over legacy PCI
config ports (`arch/x86_64/pci.rs`) + `mm::map_mmio`. x86's `DiskDevice` is now
a `Disk` enum (virtio → NVMe → AHCI) mirroring aarch64's. Verified read/write on
all four combinations (aarch64/x86 × NVMe/AHCI): ext4 200 KB round-trip →
`pattern match: true`. No storage-driver divergence remains.

**DONE: aarch64 GICv3 + generic-timer IRQ = timer-preemptive scheduling (with an
HVF fallback).** Closed the last driver-level divergence. New `arch/aarch64/gic.rs`
(GICv3 distributor + BSP redistributor + `ICC_*` CPU interface + CNTP periodic
timer, EL-aware: sets `ICC_SRE_EL2` when at EL2/VHE, and enables/accepts both the
EL1 timer PPI INTID 30 and the EL2 INTID 26) + `arch/aarch64/exceptions.rs` (the
EL1 vector table). The IRQ vector saves the full caller-saved state (x0–x30, all
q0–q31, ELR/SPSR) so preemption composes with the cooperative `switch_to` exactly
like x86; the timer IRQ calls `sched::on_timer_tick`. BSP-only (secondaries park,
like x86's APs). Verified under **TCG `gic-version=3`**: `GICv3 up … timer @ 100 Hz`,
`timer delivering IRQs (4 ticks/50 ms) — preemptive scheduling`; both arches build,
103/103.

*HVF caveat (why there's a fallback):* Apple-Silicon HVF's emulated GICv3 does
**not** expose the `ICC_*` system-register CPU interface to a bare-metal EL1
guest (access is UNDEFINED even with `ICC_SRE_EL1.SRE=1`), and HVF refuses to
provide EL2 (`virtualization=on` is rejected) — so there is no way to receive GIC
interrupts under HVF without a full guest OS. Rather than crash, `gic::init_bsp`
**probes** one CPU-interface access under a recoverable sync-exception handler
(`aarch64_sync_dispatch` advances ELR past an UNDEF during the probe): if it
faults (HVF) we stay cooperative (as before — no regression, inference still
`matches reference=true`); if it works (TCG / KVM / real ARM hardware) we enable
true timer preemption. So the *driver + preemption capability* exists on aarch64
and is exercised wherever the platform permits; HVF's limitation is the
hypervisor's, not the kernel's, and is handled gracefully.

(PS/2 keyboard is x86-only but genuinely N/A on ARM — covered by USB-HID +
virtio-input.) No driver-level divergence between the arches remains.

**DONE: PS/2 keyboard on aarch64 (PL050) — input parity.** The x86 PS/2 keyboard
(i8042) now has its ARM counterpart: `arch/aarch64/pl050.rs`, the PL050 PrimeCell
KMI (the PS/2 controller on ARM boards — Versatile/RealView/Vexpress). Polled
(the aarch64 input model), scan-code **set 2** decode (vs the PC i8042's set 1),
PrimeCell-ID-probed so it's a safe no-op where absent (QEMU virt has no PL050).
Console input chain: aarch64 `xHCI → PL050 → virtio-input → PL011`, mirroring x86
`PS/2 → xHCI → serial`. Untestable under QEMU virt (no PL050); written to the TRM.
Also confirmed **USB-HID/xHCI is already dual-arch** (shared `xhci.rs` core + both
wrappers) — no change needed.

**Full parity audit (2026-07-04).** Swept every `cfg(target_arch)` and both arch
trees. All subsystems conform to the standing rule (timers, interrupts, MMU,
SMP, storage, framebuffer, serial/console, inference kernels, shell commands are
either shared or per-arch drivers behind a shared API). Two items are *not* code
gaps for the shipping product, documented here so they aren't re-flagged:
- **Multi-part model on the aarch64 `boot-limine` path** returns None
  (`cortex/mod.rs`). NOT a shipping gap: the default aarch64 boot (`-kernel` /
  UEFI stub) loads *every* model incl. the 9B (placed contiguously at a fixed
  address by the loader/stub). `boot-limine` on aarch64 is the abandoned
  experimental path; closing it needs an aarch64 frame allocator (models exceed
  the fixed 256 MiB heap, so heap reassembly is impossible) — a subsystem for a
  dead path. Left as-is.
- **Timer preemption on Apple-Silicon HVF** falls back to cooperative: HVF's
  emulated GICv3 UNDEFs the `ICC_*` CPU-interface sysregs for a bare-metal EL1
  guest and refuses EL2, so there is no interrupt path to preempt from.
  Hypervisor limitation, handled gracefully (`gic::init_bsp` probes + falls
  back); preemption is real on TCG / KVM / real ARM hardware. Agent tasks yield
  (step-wise inference, I/O), so cooperative multitasking is functionally fine
  on HVF. Not fixable in the kernel without HVF exposing the interface.
