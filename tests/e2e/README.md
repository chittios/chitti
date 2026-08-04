# ChittiOS — end-to-end tests

These boot the real kernel under QEMU and drive its shell over the serial
console, exercising the **networked core flows** against local host servers:

| scenario | what it checks |
|----------|----------------|
| `network` | DHCP brought up an IPv4 address (`/network`) |
| `http_get` | `/http -v <url>` — GET, verbose head, 200 + body |
| `http_post` | `/http -X POST -H … -d …` — method, header, body echoed back |
| `http_stream` | `/http --stream <sse>` — chunked/SSE body rendered live |
| `ws` | `/ws ws://…` — WebSocket handshake + echo round-trip |
| `wss` | `/ws wss://…` — WebSocket over TLS 1.3 |
| `model_remote_https` | `/model remote https://…` — hosted-model chat over TLS |
| `ping` | `/ping` — ICMP echo to the gateway |
| *os group* | every shell command: `/help /info /datetime /disks /lspci /mounts /ls /pwd /skills /shortcuts /mode /think /agents /ui /ktrace /close /top /clear /wifi`, plus `memory` (add/get/list), `fs_basic` (hierarchical `/ls`, `/mkdir` `/cp` `/mv` `/rm` `/touch` `/glob` `/cat`), `composer_path_complete` (Tab completes a filesystem command's path argument; Enter runs the completed command), `session` (save/clear/resume), `help_restart` (`/help` lists `/restart` + `/memory`) |
| `restart` (final) | `/restart` reboots (aarch64: in-place second boot; x86 `-no-reboot`: guest may exit) — always last |
| *model group* (`--slow`) | `/bench`, `/infer`, `/perf`, a local chat turn, `/compact` — needs `assets/model.gguf` |
| *voice group* (`--slow`) | `/voice models`, `/voice say` (TTS) — needs `assets/voice/` + a sound device |

The in-kernel unit suite (`cargo xtask test`, x86 under QEMU) covers the pure
logic — parsing, decoding, crypto vectors, frame codecs. These e2e tests cover
the parts that only exist end-to-end: the live TCP/TLS/WebSocket exchanges over
the real network stack. They need no bundled model — the hosted-model test uses
a fake OpenAI server, so generation is remote.

## Running

```sh
make e2e                              # os + net + agents groups (~13 min serially)
make e2e E2E_JOBS=3                    # same, across 3 concurrent guest boots (~1-3 min)
make e2e-full                         # + model + voice groups (slow; needs assets/model.gguf + voice)
# or directly, with a TLS-1.3-capable interpreter for the https/wss scenarios:
/opt/homebrew/bin/python3 tests/e2e/run.py
/opt/homebrew/bin/python3 tests/e2e/run.py --jobs 3 --smp 4   # parallel: 3 shards, 4 vCPUs each
/opt/homebrew/bin/python3 tests/e2e/run.py -v          # stream guest serial live
/opt/homebrew/bin/python3 tests/e2e/run.py -arch aarch64 -model qwen3.5-0.8b
```

- **Parallel (`--jobs N`)**: the scenarios are split round-robin across `N`
  concurrent guest boots, each on its own host-forwarded port triple (the host
  test servers are shared — they already handle a thread per connection). Every
  shard VM then requests fewer vCPUs (`--smp`, default `max(2, 8//N)`) so the
  host isn't oversubscribed. `/restart` still runs last, alone, after every
  shard finishes. Pick `N` from how much RAM the host has — each VM uses ~3 GB.
- **Dependency-free** — stdlib only. `run.py` starts the host servers
  (`servers.py`), boots the guest via `cargo xtask run` (`guest.py`), runs each
  scenario, and exits non-zero on any failure.
- **TLS scenarios auto-skip** (not fail) if the running Python lacks TLS 1.3
  (macOS system Python is LibreSSL — use Homebrew's `python3`). The ECDSA-P256
  cert is generated once into `certs/` (gitignored) via `openssl`.
- The guest reaches the host servers at `10.0.2.2` (QEMU user-net). Boots
  headless (`CHITTI_DISPLAY=none`, `CHITTI_AUDIO=off`).

## Notes / limits

- aarch64 uses HVF on Apple Silicon (fast); elsewhere QEMU falls back to TCG
  (slower, but the net flows still pass).
- TLS uses the same in-kernel stack as `https://`: servers must present an
  **ECDSA-P256** cert and TLS 1.3, and certificates are **not** verified
  (see `kernel/src/net/tls.rs`).
- Local inference (`/perf`, plain chat with the bundled model) is intentionally
  out of scope here — it needs the ~800 MB model and is slow; its numerics are
  covered by `cargo xtask ref-check` and the `cortex` unit tests.
