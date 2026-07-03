# Chitti OS — Agentic Re-Architecture: Progress Log

Append one entry per milestone: phase, what landed, gate status per arch, next step.

---

## Run start

- Read all three handoff docs in full; emitted the Decisions ledger (see DECISIONS.md).
- Baseline before the re-arch: 69/69 in-kernel tests green; both arches build; 9B forward
  validated byte-exact vs llama.cpp on `/infer`. Carried-over uncommitted work: a top-k/top-p
  nucleus sampler (`cortex::sampler::sample_topk_topp`) + chat `pick()` using it — committing
  as a clean baseline before the re-architecture.
- Plan: Milestone 0 (deps + shared `agent/types.rs` contract), then Phases A→G, each gated by
  `cargo xtask test` (x86_64 QEMU) + aarch64 build. 9B chat fix deferred to the end.

<!-- milestone entries appended below -->

## Milestone 0 — shared type contract  ✅

- Added deps: serde (derive/alloc/no_std), postcard (alloc), bitflags v2 (serde) — all compile
  under `-Z build-std` on both arches, none pull std.
- `kernel/src/agent/types.rs`: the full CHITTI_SCHEMAS.md contract — id newtypes + monotonic
  minters, `Provenance` (+join/is_untrusted), `CapDomain`/`Rights`(bitflags)/`Scope`/
  `CapabilityRequest` (+`contains`, `intersect_caps` attenuation), `AgentManifest`, `Session`
  (+ all sub-structs), `SkillManifest`+`InstallRecord`.
- Deviation: dropped the schema's `#[serde(tag/content)]` on Provenance/Scope/Origin/
  InstallSource — internally/adjacently-tagged enums need `deserialize_any`, unsupported by
  postcard (non-self-describing). Externally-tagged instead; postcard is the canonical format,
  JSON is debug-only. schema_version stays 1 (no field meaning changed). Logged DECISIONS.md.
- Gate: x86_64 `cargo xtask test` = 73/73 (4 new: postcard roundtrip, provenance/scope survive,
  taint join, cap attenuation narrows-never-widens). aarch64 builds clean.
- Next: Phase A — Session object + agentic loop core with the StepSource seam.

## Phase A — Session + agentic loop core  ✅

- `kernel/src/session/`: Session construction + message/token bookkeeping (`session.rs`),
  persist/resume/fork over the memory store via postcard (`store.rs`), todo list + `todo_write`
  (`todo.rs` incl. a minimal JSON field extractor).
- `kernel/src/agent/agent_loop.rs`: the loop `model → tool_calls → Synapse → tool_results →
  repeat` with `max_turns`/`max_tool_calls` budgets + clean stop conditions, over two seams —
  `StepSource` (next Step) and `ToolDispatch` (execute one call, all effects via Synapse).
- `agent/manifest.rs`: builtin orchestrator + reader-subagent roles; lowers declarative
  CapabilityRequests to live `cap::Right::InvokePrimitive` grants (`grant_to_task`).
- `agent/orchestrator.rs`: the foreground main agent (`spawn`/`from_session`/`handle`/`kill`)
  + `SynapseTools` dispatcher (write/read/list/delete/console/emit_result → Synapse;
  todo_write session-local). Justification is trusted in Phase A; taint-aware flag ready for E.
- `agent/rule_steps.rs`: deterministic `StepSource` (ScriptedSteps + `for_intent`) — the
  model stand-in for tests/boot demo.
- Kernel change: `sched::spawn_parked` — cap-owning agent identity tasks are created (cap table
  live) but NOT enqueued, so they never steal a scheduler turn. This fixed a regression where the
  phase5 IPC test's cooperative yield loop scheduled leftover agent tasks. Reusable by Phase C.
- Gate: x86_64 `cargo xtask test` = 76/76 (3 new: loop-completes-with-tool, save/resume+continue,
  turn-budget-stops); aarch64 builds; **live x86_64 QEMU boot** shows the loop (stop=Final,
  turns=3, tool_calls=2) + save→resume(7 msgs)→continue(11 msgs).
- Next: Phase B — first-class MCP-shaped tool layer (registry + dispatch + builtin toolset).

## Phase B — first-class tool layer  ✅

- Two new Synapse primitives (`mem_fs_edit` id=8, `mem_fs_search` id=9) so `edit`/`search`
  route through the same capability/taint-gated, audited executor as every other effect.
- `kernel/src/tools/`: `registry.rs` (MCP-shaped ToolDef catalogue — the 11 builtin tools with
  JSON input-schemas + `ToolBinding`; provider registration; per-agent `for_agent` discovery +
  `describe` for prompts), `dispatch.rs` (`Router`: the real ToolDispatch — shape-validate →
  Synapse cap+taint gate → tool_result; agent-layer bindings (spawn_subagent/load_skill/run)
  delegate to hooks installed in later phases), `provider.rs` (in-kernel "MCP server" registration,
  used by Phase F skill-bundled tools).
- Router supersedes the Phase-A inline dispatcher (SynapseTools removed); orchestrator keeps only
  the shared `synapse_call` + `to_taint` helpers. Demo/tests now use `tools::Router`.
- Gate: x86_64 `cargo xtask test` = 81/81 (5 new: malformed-rejected-before-dispatch,
  ungranted-denied+audited, write/read roundtrip+audit, todo_write updates session, discovery
  intersects toolset); aarch64 builds; live x86_64 boot demo still completes via the Router.
- Next: Phase C — sub-agents (spawn_subagent, isolation, cap attenuation, parallel, depth cap).

## Phase C — sub-agents (isolated delegation)  ✅

- `kernel/src/agent/subagent.rs`: `dispatch` (depth check → `attenuate` subset enforcement →
  isolated Session on its own cap-owning parked task → run its loop → condensed summary),
  `integrate`/`record` (summary crosses back, transcript never merged), `dispatch_batch`
  (per-core assignment, SMP-ready), `attenuate` (strict subset; refuse on widen).
- `orchestrator.router()` wires the `spawn_subagent` tool hook (enforces parent caps + depth;
  sub-agents run a rule StepSource, get a plain Router so they can't sub-delegate).
- Gate: x86_64 `cargo xtask test` = 85/85 (4 new: context-isolated, widening-cap-refused,
  two-subagents-integrate-both, depth-limit); aarch64 builds; **live x86_64 boot** shows 2
  sub-agents on cores 0/1 with isolated 5-msg transcripts, parent left with only 3 messages
  (system + 2 summaries — no sub-transcripts).
- Caveat (DECISIONS.md): SMP true-concurrency deferred under QEMU TCG; per-core structure in place.
- Next: Phase D — context compaction + todo-driven planning + session fork.

## Phase D — context management + planning  ✅

- `kernel/src/agent/context.rs`: `maybe_compact` (when live_tokens ≥ compact_threshold, evict the
  oldest non-system, non-recent turns to the store keyed `sess/<id>/cmp/<msg>`, mark them
  resident=false + store_ref, keep a summary in `ContextState.compactions`, recompute live tokens);
  `recall` (demand-page a compacted message's full text back into context). Wired into the loop —
  compaction runs after each tool turn.
- Todo-driven planning reuses `session::todo::write` (idempotent whole-list replace, returns
  remaining count) — a 5-step plan is tracked and worked down.
- Session fork reuses `session::store::fork` (new id, deep clone, independent).
- Gate: x86_64 `cargo xtask test` = 88/88 (3 new: compaction-evicts+recall-pages-back,
  5-step-task-via-todos, fork-diverges-without-mutating-parent); aarch64 builds; **live boot**
  shows compaction (162→127 tokens), recall of a compacted fact verbatim, and an independent fork.
- Next: Phase E — permission+safety (taint+cap gating in dispatch, compiled-intent replay).

## Phase E — permission + safety integration  ✅

- `agent/compiled.rs`: agent-layer compiled intents — record a validated tool-call plan keyed by
  (intent signature, file-content preconditions); `lookup` replays deterministically with ZERO
  inference when preconditions hold, `compile` caches after a Final run, stale preconditions
  re-plan. Replays still flow each call through Router→Synapse (gated + audited).
- `orchestrator.handle_compiled` (cache-first, compile-on-success) + `safe_router` (taint-aware:
  justification derived from the session's worst resident provenance; `human_confirmed` flips the
  shell-approval bit). Destructive/tainted calls hit the existing Synapse taint gate.
- Gate: x86_64 `cargo xtask test` = 92/92 (4 new: injected-destructive-gated+audited,
  confirmed-destructive-proceeds, repeated-plan-replays-without-inference, stale-precondition-
  replans); aarch64 builds; **live boot** shows the injected delete blocked (secret survives) and
  a compiled intent replaying with +0 inference.
- Next: Phase F — skill subsystem (package/index/loader, progressive disclosure) + sample skill.

## Phase F — skill subsystem (progressive disclosure)  ✅

- `kernel/src/skills/`: `index.rs` (L0 metadata registry + description-based `match_task`; full
  manifest persisted to the store), `loader.rs` (progressive disclosure — `load_body` L1 on match
  tagged `SkillInstalled`, `load_asset` L2 demand-paged, tier tracking on `skills_in_scope`),
  `package.rs` (SkillPackage format + `place_trusted` + `sample_note_summarizer`; bundled tools
  register via `tools::provider` and bind to existing Synapse primitives), `install.rs` (stub → G).
- `orchestrator.router()` wires the `load_skill` tool hook. `ToolBinding::Synapse` made owned so
  runtime-registered bundled tools bind like builtins. Fs READ now also grants MEM_FS_SEARCH,
  WRITE grants MEM_FS_EDIT.
- Gate: x86_64 `cargo xtask test` = 95/95 (3 new: progressive-disclosure-loads-body-only-on-match
  [+ unrelated-loads-nothing], bundled-tool-capability-gated, L2-asset-demand-paged); aarch64
  builds; **live boot**: skill placed L0-only, unrelated task matched=false, matching task loads
  L1 body, bundled note_search runs through Synapse.
- Next: Phase G — permissioned install (Ed25519 verify + consent + skill-agent) + signed packages.

## Phase G — permissioned skill/agent installation  ✅

- `skills/crypto.rs`: package signing/verification. (Ed25519 crate faults at runtime on bare metal
  — see DECISIONS; swapped for a self-contained SipHash-2-4 keyed MAC, 512-bit tag.)
- `skills/package.rs`: `sign`/`verify` over a canonical package message (manifest sans-sig + body +
  assets), so any tampering invalidates the signature; `sample_report_agent` (signed skill-agent).
- `skills/install.rs`: the install flow — verify → consent (approved ⊆ requested) → grant only the
  intersection → register (body/assets/tools/index) + persist an InstallRecord; `uninstall` revokes.
- `skills/agent_skill.rs`: installable skill-agent roles; dispatch effective caps =
  min(role, install grant, parent) — never wider than any.
- Gate: x86_64 `cargo xtask test` = 101/101 (6 new: crypto roundtrip + unsigned/tampered-refused,
  approved-subset-only, skill-can't-exceed-grant [SkillInstalled doesn't bypass caps], skill-agent-
  caps-never-widen, uninstall-revokes-and-unloads); aarch64 builds; **live boot**: tampered package
  refused, skill installed READ-only, skill-agent bounded (WRITE present: false).

## A→G COMPLETE — all seven phase gates green (x86_64 tests + aarch64 build + live boot demos).

## 9B chat degeneration fix

- Root cause (from prior deep debugging, recorded in CLAUDE.md): the 9B forward is correct
  (/infer is byte-exact vs llama.cpp); terse chat prompts degenerated into repeated punctuation —
  a *decode-side* degeneration typical of thinking models under weak sampling.
- Fixes (committed):
  1. `cortex::sampler::sample_topk_topp` — Qwen's documented decoding (temp 0.7 / top_k 20 /
     top_p 0.8) + a light repetition penalty in the chat `pick()`. Unit-tested: over 2000 draws
     on a peaked+long-tail distribution it NEVER draws a tail token (the degeneration mechanism).
  2. Hard anti-degeneration decode guard in `ChatSession::turn`: stop generation if a token
     repeats 5× in a row, so chat can never emit an unbounded run regardless of the forward.
- Validation limitation (honest): the fast unit suite proves the sampler mechanism. Full *interactive*
  9B chat validation was blocked in this environment — driving the aarch64 chat non-interactively
  (expect/pipe → PL011 UART RX) did not deliver input reliably (banner captured, input not echoed;
  the user's manual interactive runs are unaffected). Each 9B boot is also ~10 min (5.4 GiB load +
  ~1 tok/s). The fixes are correct and unit-tested; interactive re-confirmation is left to a manual run.
- Note: observed a rare (~1/7) pre-existing flake in `ipc_round_trip_delivers_a_message` (a latent
  spawn→timer-preempt→grant race in that test, unrelated to the agent layer — agent tasks use the
  non-enqueued `spawn_parked`). 6/6 subsequent runs green at 103/103.

## FINAL STATUS

- Agentic re-architecture Phases A→G: COMPLETE. 103/103 x86_64 in-kernel tests; both arches
  (x86_64 + aarch64) build for both models (0.8B + 9B); every phase demonstrated live at boot
  (x86_64), and the full A→G demo suite also verified running on the aarch64 9B build under HVF.
- Three invariants upheld: all effects route through Synapse; delegation only narrows authority
  (strict subset, refuse-on-widen); an installed skill is bounded by its install grant forever
  (verify → consent-subset → min(role,grant,parent), SkillInstalled never bypasses the cap gate).
- Deviations (all in DECISIONS.md): externally-tagged enums for postcard; Ed25519 → SipHash keyed
  MAC (crate faults on bare metal); SMP sub-agent concurrency structured but sequential under TCG;
  sample packages signed in-kernel vs shipped as separate boot-module files.

## Storage/install run

### P1a — multi-part model in the ISO  ✅
- `xtask`: `split_model_into_parts` streams the GGUF into `model.gguf.000/.001/...` (<=3 GiB each,
  override CHITTI_MODEL_PART_MB), each declared as a Limine module; build-time model choice (default 0.8B).
- kernel `cortex::model_module` (x86): collect all `model.gguf*` modules, sort by path; 1 part =
  zero-copy slice; >1 part = reassemble into contiguous frames via `alloc_dma` (heap is only 256 MiB).
  `limine_protocol::File` gained `path_str`/`path_contains`.
- Verified: forced the 0.8B into 3 parts (300 MiB); boot reassembled 811,843,840 bytes and `/info`
  parsed a valid GGUF (dim 1024, layers 24, vocab 248320). Single-ISO distribution, no separate disk.

### P1b — no auto-format at boot  ✅
- `disk_demo` now mounts SimpleFS ONLY (never `mount_or_format`); a blank/foreign disk is reported
  and left untouched ("NOT auto-formatting; use /install or /mkfs"). Verified: booting a zeroed
  virtio-blk disk leaves it all-zero. 103/103 tests.

### P2 — FS detection + read-only mount + shell commands  ✅
- `fs/detect.rs`: parse GPT (LBA1 "EFI PART" + entries) and MBR (0x55AA + entries), or treat a
  bare device as a super-floppy; classify each volume as FAT32/exFAT/NTFS/ext2/3/4/XFS/SimpleFS by
  on-disk signature, pulling FAT/ext/XFS labels. No writes to foreign filesystems.
- `fs/roread.rs`: read-only FAT32 root-directory listing (8.3 names, cluster-chain walk); SimpleFS
  uses native listing. exFAT/NTFS/ext4/XFS are detected but listing is reported unimplemented ("where
  feasible" per Q1).
- Shell: `/disks` (list volumes + FS + label + size), `/ls <n>` (root dir of FAT32/SimpleFS volume),
  `/mkfs [yes]` (explicit destructive SimpleFS format). x86-only (virtio-blk over PCI); aarch64 stubs.
- Verified: a crafted GPT disk with 5 partitions → `/disks` detected FAT32/exFAT/NTFS/ext4/XFS with
  correct labels. `limine_protocol::File` + `fs::SIMPLEFS_MAGIC` exposed.

### P3 — /install: GPT partitioning + SimpleFS OS partition  ✅ (bootloader payload = REVISIT)
- `block/gpt.rs`: GPT writer (protective MBR + primary/backup header + 128-entry array, IEEE
  CRC32) with a 512 MiB ESP + a Chitti/Linux-data (SimpleFS) partition; `default_layout`/`standard_parts`.
- `block/mod.rs`: `Partition` block-device view (offset+len) so SimpleFS lives inside a partition.
- Shell `/install [yes]`: writes the GPT, formats the OS partition SimpleFS, writes markers
  (chitti-os, VERSION). Destructive, explicit (never at boot).
- Verified: `/install yes` on a 768 MiB disk -> `/disks` re-reads GPT as [EFI System 512 MiB] +
  [SimpleFS 255 MiB]; `/ls 1` lists chitti-os + VERSION across a reboot; host python confirms a
  valid GPT with correct header CRC32 and the two named partitions.
- REVISIT (DECISIONS.md): FAT32 ESP population (Limine + kernel + multi-part model) for standalone
  UEFI boot -- needs a FAT32 writer + installer-payload modules. The model can't go on SimpleFS
  (4 KiB file cap), so it targets the ESP. Booting today is still via the USB/ISO.

## Storage/install run status: P1, P2 done + verified; P3 partitioning + SimpleFS install done +
## verified; standalone-boot ESP population is the honest remaining piece (REVISIT). Both arches
## build; 103/103 tests throughout.

### ext4 write driver (E1/E2) — validated in Python, e2fsck-clean  ✅
- `tools/mkext4_ref.py`: from-scratch ext2/4-family mkfs + file writer — multi-block-group
  (superblock + sparse backups + GDT + per-group block/inode bitmaps + inode table + root dir),
  block-mapped files (direct + single/double indirect). Verified with e2fsprogs: e2fsck -fn is
  CLEAN on a 5-group 640 MiB image; a 300 MiB double-indirect file dumps byte-identical via debugfs;
  small file + root dir list correctly. This is the validated spec for the no_std Rust port.
- Remaining: E3 port to Rust (block/ext4.rs over BlockDevice), E4 minimal FAT ESP writer, E5 wire
  /install (GPT + FAT ESP[Limine] + ext4[limine.conf+kernel+model]) + verify e2fsck + OVMF boot.

### E3 — ext4 write driver ported to no_std Rust  ✅ (e2fsck-clean in QEMU)
- `block/ext4.rs`: `Ext4Writer::format(dev, files)` — the validated ext4 mkfs+writer, ported to
  no_std over `BlockDevice`. Streams file data straight to device blocks (the model dwarfs the
  heap); builds metadata (bitmaps, GDT, superblock+sparse backups) a block at a time. Multi-block-
  group, block-mapped files (direct + single/double indirect).
- New `/mkext4 [yes]` shell command (ext4 format + test files), parallel to `/mkfs`.
- Verified in QEMU + host e2fsprogs: the KERNEL-written disk is `e2fsck -fn` CLEAN at both 96 MiB
  (1 group) and 640 MiB (5 groups); `debugfs` reads hello.txt correctly and dumps the 200 KB
  indirect-block big.bin byte-identical. 103/103 tests; both arches build.
- Remaining: E4 minimal FAT ESP writer (BOOTX64.EFI) + E5 wire /install to GPT + FAT ESP(Limine) +
  ext4(limine.conf+kernel+model) and verify standalone OVMF boot.

### E4/E5 — FAT ESP writer + standalone bootable install  ✅ (boot verified)
- `block/fat.rs`: minimal FAT16 writer + VFAT LFN (so limine.conf/chitti-kernel keep full names).
- `/install`: GPT (block/gpt) -> FAT ESP (BOOTX64.EFI + limine.conf + kernel) -> ext4 OS partition
  (block/ext4: limine.conf + kernel + model parts). xtask bundles BOOTX64.EFI + kernel as payload modules.
- Verified: installed 256 MiB disk -> e2fsck-clean ext4 + fsck_msdos-clean FAT (LFN names correct);
  booted STANDALONE under OVMF (no ISO) to "Chitti: boot ok".
- Finding: Limine can't read our minimal ext4 (kernel therefore boots from the FAT ESP, standard);
  loading the model on the installed system is a documented REVISIT (FAT32 model partition, or an
  in-kernel ext4 reader).

## ext4-primary run status: ext4 WRITE driver done + e2fsck-verified; standalone UEFI boot done +
## verified; model-on-installed-system is the remaining follow-on. Both arches build; 103/103 tests.

### R1-R4 — in-kernel ext4 READER + model-from-ext4 at runtime  ✅
- `block/ext4_read.rs`: read-only ext4/ext2 reader — superblock, block-group descriptors, inodes,
  root directory, and file data via BOTH block maps (12 direct + single/double indirect — our
  writer's output) and extent trees (real mke2fs ext4). Streams a block at a time into a caller
  buffer (no second full-size copy).
- `cortex::model_module` fallback (`model_from_ext4`): when no Limine model module is present (an
  installed system booted from the FAT ESP, kernel-only), find the ext4 OS partition, read every
  `*.gguf*` root file in order into contiguous frames (alloc_dma), and return the blob. New
  `/ext4read` command for verification.
- Verified in QEMU: (R2) `/ext4read` reads back our writer's hello.txt + a 200 KB indirect-block
  big.bin byte-identical, AND reads a real `mke2fs` ext4 (extents) the same way; (R4) booting a
  model-LESS ISO + an ext4 disk holding the real 0.8B model → "loaded model from ext4 partition
  (811,843,840 bytes)" and the model parsed correctly (24 layers, dim 1024, vocab 248320).
- Net: the install loop is closed — `/install` writes the model to ext4 (e2fsck-clean), and the
  installed kernel reads it back off ext4 at runtime. Both arches build; 103/103 tests.

### W1-W5 — durable agent state: synapse::fs persisted to ext4 across reboots  ✅
- `block/ext4_store.rs`: an ext4-backed store for `synapse::fs` — an in-memory cache that
  **persists on every mutation by rewriting a dedicated ext4 data partition** with the verified
  `Ext4Writer` (mkfs + write-all) and reads it back with `Ext4Reader` on mount. Reuses the two
  verified drivers, so each persisted image is e2fsck-clean by construction. This is
  *rewrite-on-sync* (O(total) per write; fine for KB-scale agent state) — a true incremental RW
  ext4 driver (live bitmap alloc/free, in-place dir edits) is a documented follow-on.
- `synapse::fs` is now a pluggable backend: in-memory by default (tests + live ISO), swapped to the
  ext4 store by `mount_ext4` on an installed system. Migrates any pre-mount writes into ext4.
- `main::mount_persistent_store` auto-mounts at boot: picks an ext4 volume that holds no `*.gguf`
  (never the model/OS partition), points synapse::fs at it, and runs a boot counter *through
  synapse::fs* to prove the round-trip.
- `/install` now lays down THREE partitions: FAT ESP + ext4 OS/model + a 256 MiB ext4 **data**
  partition (formatted empty) for durable agent state.
- **Two writer bugs fixed:** (1) synapse keys contain `/` (`sess/5/cmp/26`), illegal in an ext4
  filename — the store percent-encodes `/`→`%2F` (and `%`→`%25`), reversibly, on write/mount.
  (2) `Ext4Writer` only wrote the inode-table blocks that held a live inode but set groups
  non-`INODE_UNINIT`, so e2fsck read stale bytes in the rest — now it zeroes the **whole** inode
  table (all blocks, all groups), making a format robust to any prior disk content (the earlier
  2-file tests passed only because the disk was `dd`-zeroed).
- **Verified:** boot twice with the same ext4 data disk → boot #2 recovered **34 real agent files**
  written at runtime (sessions, skills, memory, notes), the synapse.fs boot counter reached #2
  ("survived a reboot"), and **e2fsck is fully clean** (all 5 passes) after the runtime writes;
  on-disk names are `%2F`-encoded (legal) and their content round-trips; the empty (0-file)
  data-partition format is e2fsck-clean too. Both arches build; 103/103 tests.
- Known limit: a full model-bearing `/install` + standalone boot in one run is impractically slow
  under QEMU TCG (the 17 MB kernel FAT write is cluster-by-cluster over polled virtio-blk) — the
  3-partition GPT layout, empty data format, standalone boot, and synapse persistence are each
  verified; at device speed (real HW / HVF) the combined flow runs normally.

### U1-U4 — aarch64 UEFI boot SOLVED: Chitti stub bootloader + trampoline handoff  ✅
- **Root causes of the long stall found:** (1) the identity `-kernel` build and the
  `boot-limine` higher-half build share one artifact path, so test cycles kept
  loading the WRONG kernel (several "failures" were invalid experiments); (2) the
  Limine path forces an HHDM memory model our identity-map kernel was never built
  for; (3) handing off on UEFI's page tables (MMU-on) was fragile, and disabling
  the MMU *from stub code* faults the stub's own UEFI-mapped PC.
- **The proper solution — `stub/`, a from-scratch UEFI bootloader:** AAVMF launches
  `BOOTAA64.EFI` (our stub, `aarch64-unknown-uefi` + uefi-rs); it loads the normal
  identity `-kernel` ELF + `model.gguf.000` off a **real FAT32 ESP image** (hdiutil-
  built; no VVFAT cap, carries the 774 MiB model), reserves the kernel's fixed heap,
  exits boot services, cleans the D-cache for everything written, then jumps through
  a **trampoline copied into identity RAM** that disables the MMU + caches and
  branches to the kernel entry — handing the kernel the **exact QEMU `-kernel`
  state**. Zero kernel changes; the whole proven identity-map path (incl.
  virtio-blk-mmio) runs as-is.
- xtask: `run -arch aarch64 --uefi` now uses the stub path (identity kernel with an
  entry-address guard against artifact mixups; ESP attached before the data disk so
  `probe_disk` targets the data disk, never the boot ESP). The Limine run-path is
  retired (kernel `boot-limine` feature remains, documented as superseded).
- **Verified: 5 consecutive UEFI boots to the interactive shell** (3 harness + 2 via
  `cargo xtask run -arch aarch64 -model qwen3.5-0.8b --uefi --disk 2G`), with the
  model loaded by the stub and parsed correctly (`/info`: dim 1024, layers 24,
  vocab 248320) and the 2 GiB data disk correctly probed. 103/103; all builds green.
- Follow-on: aarch64 `/install` writes the stub+kernel+model ESP payload so
  `--disk-only` boots the installed disk standalone (B2/B3).

### V1-V5 — arch-parity audit + aarch64 `/install` (the last feature gap)  ✅
- **Audit:** every `target_arch` gate reviewed. Agent layers (agent/session/skills/
  tools/synapse/persona/fs) have ZERO gates. Remaining gates are legitimate
  arch-specific drivers behind shared APIs (tensor SSE2/AVX2|NEON, context switch,
  serial 16550|PL011, keyboard PS/2+xHCI|virtio-input, framebuffer GOP|ramfb,
  virtio-blk PCI|MMIO, mm, SMP) or boot machinery. `/exit` = PSCI SYSTEM_OFF ✓.
  The ONE feature gap: `/install` was x86-only.
- **aarch64 `/install` built:** new `block/fat_read.rs` (FAT16/32 reader: VFAT LFN,
  subdirectory walk, file read + `file_size`); `probe_disk_nth` on both drivers
  behind the same facade; `FatWriter` clusters now adaptive (2-32 KiB, so a
  model-carrying ~840 MiB ESP fits FAT16); `gpt::esp_data_parts` (ESP + ext4 data,
  no OS partition — the stub reads everything off the ESP). The installer payload
  (BOOTAA64.EFI + kernel) is read from the boot ESP; the model is sliced from RAM
  (`model_module()` + the FAT dir-entry size) — it can't fit the 256 MiB heap.
- **Verified:** aarch64 `--uefi` → `/install yes` completes end-to-end: payload
  read (stub 49 KB + kernel 1 MB + model 774 MB), GPT written (ESP lba 34..1718809
  + data), FAT ESP written incl. the model, ext4 data formatted → DONE. The
  `--disk-only` standalone boot of the installed disk uses the same stub/ESP
  mechanism as the verified `--uefi` boot (per the user, the long full-boot wait
  was skipped; the firmware streams 774 MB off the ESP, which takes minutes).
- Both arches build; 103/103 tests. Feature parity: every user-visible capability
  (chat/inference, storage, /install, persistence, UEFI boot, framebuffer,
  keyboard, SMP) now exists on both arches.

### Batched block IO — `/install` model write: minutes → 5 seconds  ✅
- The slowness was per-sector polled virtio: 1 request per 512 B meant ~1.6M
  round trips for the 774 MiB model, plus ~100K FAT read-modify-writes for the
  ~24K cluster allocations.
- `BlockDevice` gained `read_blocks`/`write_blocks` (default per-sector loop;
  `Partition` forwards). **Both** virtio drivers (x86 PCI + aarch64 MMIO, per the
  parity rule) now move up to 64 KiB per polled request through a DMA bounce
  buffer. Hot paths batched: FAT `write_clusters` writes whole contiguous runs;
  `alloc_chain` builds the FAT a sector at a time (256x fewer IOs); ext4
  `write_eblock`/`stream_file` write 4 KiB blocks / contiguous multi-block runs;
  `ext4_read` reads whole fs-blocks per request.
- **Verified:** the full aarch64 `--uefi` `/install yes` (774 MiB model) now
  completes in **~5 s** (was minutes); the installed FAT ESP is fsck_msdos-clean,
  the ext4 data partition e2fsck-clean, and the model file on the installed disk
  is **sha256-identical** to the source. 103/103; both arches build.

### aarch64 `--disk-only` verified end-to-end (after an xtask disk-wipe bug)  ✅
- The user's `--disk-only` run dropped to the EFI shell because xtask's aarch64
  option threading passed `fresh=true` when no `--disk` size was given — it
  **wiped the just-installed disk** and recreated it as an empty 4 MiB image.
  Fixed: `--disk-only` keeps the existing disk as-is (wipe only on an explicit
  `--fresh-disk`).
- **Full x86-parity cycle now verified on aarch64:** `--uefi` boot → `/install
  yes` (~5 s with the model) → `--disk-only` → the installed disk boots itself:
  the stub loads the kernel + 774 MiB model off the installed FAT ESP,
  `boot ok`, synapse persistence mounts the installed ext4 data partition
  (**boot #2, agent files recovered — durable across the reboot**), the shell
  comes up, and `/info` shows the model parsed (dim 1024, layers 24, vocab
  248320). Remaining wall-clock cost is AAVMF streaming the model off the ESP
  at boot (~1-2 min, inside edk2).
