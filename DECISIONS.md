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
