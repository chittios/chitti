# Security Policy

## Status

Chitti OS is an experimental research operating system under **active
development**. It is **not stable and not intended for production or for
handling sensitive data.** Run it in a virtual machine. It is provided "as is",
with no warranty; the authors are not responsible for any damage or data loss.

That said, security *is* the point of the project — the determinism boundary,
unforgeable capabilities (including channel/listener/surface handles),
per-target **scope enforcement**, provenance/taint gating, skill sandboxing, and
signed package install are core design goals — so we take reports of holes in
those mechanisms seriously.

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
  effect without going through Synapse's grammar + capability + scope + taint
  gate. This includes a *service agent* (native daemon) that is coerced by
  attacker-controlled wire data into an effect outside its granted capabilities.
- **Capability forgery / privilege escalation** — obtaining authority not
  granted, or a sub-agent exceeding its parent's capabilities.
- **Channel-handle forgery / ambient authority over channels** — reading or
  writing a channel (or a forwarded TCP connection) without holding its
  per-direction `ChannelRead`/`ChannelWrite` end, e.g. by guessing a handle
  integer that resolves outside the caller's own capability table, or a
  `channel_grant` that hands away an end the caller does not hold.
- **Scope-gate bypass** — a primitive touching a resource outside the granted
  scope: an fs write outside the granted path glob, or network egress to a host/
  port outside the granted `Net{host,port}` range, without a `DeniedScope`.
- **Prompt-injection escalation** — untrusted, ingested content (including bytes
  read off the network) driving a destructive/high-privilege primitive —
  `mem_fs_delete`, **`net_http_post`/net egress**, **`channel_grant`**, port
  binding — without the taint gate firing.
- **UI surface fencing bypass** — an agent drawing to, polling input from, or
  closing a surface it does not own.
- **Skill grant escapes** — an installed skill/agent acting beyond its
  install-time capability grant, or install accepting an unsigned/tampered/
  untrusted package (MAC or ECDSA-P256), or a registry package verifying against
  a key not in the baked trust store.
- **Audit-log tampering** — making the append-only audit log lose or alter a
  prior entry.
