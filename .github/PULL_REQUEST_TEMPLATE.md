<!-- Thanks for contributing to ChittiOS! Please read CONTRIBUTING.md first. -->

## What & why

<!-- What does this change do, and why? Link any related issue (Fixes #123). -->

## How it works

<!-- Brief notes on the approach / any design decisions or trade-offs. -->

## Standing rules

- [ ] **Dual-arch parity** — builds and works on both `x86_64` and `aarch64`
      (no `target_arch` gate without a same-API equivalent for the other arch).
- [ ] **Real hardware** — no addresses/resolutions/layouts hardcoded to QEMU or
      VirtualBox; hardware is discovered (ACPI/GOP/fw_cfg/HID/…) and degrades
      gracefully.
- [ ] Effects route through Synapse; delegation only narrows authority; any skill
      stays bounded by its install grant. (N/A if unrelated.)

## Verification

<!-- Show what you ran. -->

- [ ] `cargo xtask build -arch x86_64 && cargo xtask test` → **104/104**
- [ ] `cargo xtask build -arch aarch64`
- [ ] `cargo xtask run -arch aarch64` (boot spot-checked, if boot-visible)
- [ ] Framebuffer/driver/input change verified via QMP screendump / VirtualBox
      (attach output if useful)

## Conventions

- [ ] Every `unsafe` has an adjacent `// SAFETY:` comment.
- [ ] Public modules have doc comments; code matches surrounding style.
- [ ] Tree builds at every commit; commit messages are clear.
