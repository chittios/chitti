# Security Policy

## Status

Chitti OS is an experimental research operating system under **active
development**. It is **not stable and not intended for production or for
handling sensitive data.** Run it in a virtual machine. It is provided "as is",
with no warranty; the authors are not responsible for any damage or data loss.

That said, security *is* the point of the project — the determinism boundary,
unforgeable capabilities, provenance/taint gating, and skill sandboxing are core
design goals — so we take reports of holes in those mechanisms seriously.

## Supported versions

There are no released versions yet. Only the latest `main` is supported.

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability.

- Use **GitHub → Security → "Report a vulnerability"** (private advisory) on the
  [chittios/chitti](https://github.com/chittios/chitti) repository, or
- open a minimal private report and we'll coordinate from there.

Include: what breaks, which architecture and boot path (QEMU `-arch …`, UEFI,
VirtualBox, real hardware), a reproducer or the relevant serial/`ktrace` output,
and the impact.

We'll acknowledge as soon as we can and work with you on a fix and disclosure
timeline. There is no bug-bounty program.

## Especially interesting classes of bug

Because of what Chitti is, these are high-value:

- **Determinism-boundary escapes** — anything where model output causes a side
  effect without going through Synapse's grammar + capability + taint gate.
- **Capability forgery / privilege escalation** — obtaining authority not
  granted, or a sub-agent exceeding its parent's capabilities.
- **Prompt-injection escalation** — untrusted, ingested content driving a
  destructive/high-privilege primitive without the taint gate firing.
- **Skill grant escapes** — an installed skill acting beyond its install-time
  capability grant, or install accepting an unsigned/tampered/untrusted package.
- **Audit-log tampering** — making the append-only audit log lose or alter a
  prior entry.
