---
name: Bug report
about: Something crashed, misbehaved, or didn't boot
title: "[bug] "
labels: bug
---

<!-- Reminder: Chitti is under active development and not stable. Please check
     `main` still reproduces before filing. -->

## What happened

<!-- Clear description of the bug and what you expected instead. -->

## Environment

- **Architecture:** x86_64 / aarch64
- **Boot path:** `cargo xtask run -arch …` / UEFI (`--uefi`) / VirtualBox / real hardware
- **Model:** qwen3.5-0.8b / qwen3.5-9b / N/A
- **Host:** (e.g. macOS on Apple Silicon, QEMU version)
- **Commit:** `git rev-parse --short HEAD`

## Steps to reproduce

1.
2.
3.

## Logs / output

<!-- Paste the relevant serial / ktrace output. For input or clock problems,
     include the boot `INPUT` line. A framebuffer screenshot helps for UI bugs. -->

```
```
