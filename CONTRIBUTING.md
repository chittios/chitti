# Contributing to ChittiOS

Thanks for your interest! Chitti is an experimental, agent-native operating
system. It moves fast and is not stable — but contributions, bug reports, and
ideas are very welcome.

Please read [CLAUDE.md](CLAUDE.md) first: it states the invariants and the two
standing rules that every change must honour. [DEVELOPMENT.md](DEVELOPMENT.md)
has the full local setup.

## Ground rules (non-negotiable)

Every change must uphold:

1. **The determinism boundary** — model output is an untrusted plan; it never
   causes a side effect directly. All effects route through Synapse
   (grammar → capability → scope → taint gate → deterministic execution → audit).
   A **service agent**'s protocol/codec logic is deterministic native code
   *below* the boundary; the model plans policy over capabilities, it never
   implements a protocol.
2. **Dual-architecture parity** — the kernel builds and works on **both**
   `x86_64` and `aarch64` from one codebase. Never gate behaviour behind
   `target_arch` unless it is genuinely arch-specific, and then provide the
   equivalent for the other arch behind the same API. A feature on one arch must
   exist on the other in the **same** change.
3. **Real hardware, nothing hardcoded to an emulator** — discover hardware the
   way firmware does (ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors,
   PrimeCell IDs) and degrade gracefully. Don't hardcode addresses, resolutions,
   or device layouts to QEMU/VirtualBox. The same image must run on QEMU,
   VirtualBox, and real UEFI hardware.
4. **No ambient authority** — a resource an agent names (a Synapse primitive, a
   channel end, a listener, a UI surface) is reachable only through a capability
   it holds in its **own** table; never a global id anyone can guess. Handles the
   model emits are resolved against the caller's own capability space.
5. **Delegation only narrows authority**, and **a skill/agent is bounded by its
   install-time grant, forever.** Consent can only shrink a package's requested
   capabilities; a scope (fs path, host/port) granted narrowly is enforced at the
   executor, not just declared.

### Every feature/fix ships with tests

Two layers, and new work adds to **both** where they apply:

- **Unit tests** (`cargo xtask test`) for the pure logic — pull the fiddly logic
  into a pure function and test it with cases.
- **End-to-end scenarios** (`tests/e2e/`, `make e2e`) for anything that only
  exists on the running OS: a shell command, a networked/service/UI/model/voice
  flow. Adding one of those means adding an e2e scenario.

## Before you open a PR

Run all of these and make sure they pass:

```sh
cargo xtask build -arch x86_64 && cargo xtask test    # keep the unit suite green
cargo xtask build -arch aarch64
cargo xtask run   -arch aarch64                        # if the change is boot-visible, spot-check the boot
make e2e                                               # if the change is boot-visible or networked
```

If the change touches the framebuffer, drivers, or input, verify it with a QMP
screendump (see DEVELOPMENT.md) and, where relevant, on VirtualBox.

## Coding conventions

- **`no_std`, no ambient authority.** No new heavyweight dependencies, nothing
  that pulls `std`, nothing doing un-auditable allocation.
- **Every `unsafe` block gets an adjacent `// SAFETY:` comment** justifying each
  invariant it relies on.
- **Every public module has a doc comment** stating its responsibility and where
  it sits in the layer stack.
- **Deterministic by default** — tests use fixed seeds and temperature 0; any RNG
  is seeded and logged. `ktrace` every capability invocation and inference call.
- **Match the surrounding code** — its style, comment density, naming, and idioms.
- **Keep the tree building at every commit.** Commit per sub-milestone with a
  clear message. Never leave the tree broken across a commit.

## Commits & PRs

- Write focused commits with descriptive messages (what changed and why).
- Fill in the pull-request template; state how you verified on **both** arches.
- Small, reviewable PRs over large ones. Explain any deviation from the rules
  above and why it's justified.

## Reporting bugs / requesting features

Use the issue templates. For bugs, include the arch, how you booted (QEMU
`-arch …`, UEFI, VirtualBox, real hardware), and the relevant serial/`ktrace`
output — especially the boot `INPUT` line for input/clock issues.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE) (as in the Apache-2.0 §5 inbound=outbound
terms — no separate CLA).
