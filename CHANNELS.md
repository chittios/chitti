# Messaging channels

**External chat platforms as agent inboxes** — Telegram, Discord, and Slack
today; webhooks later. This is the OpenClaw-style *messaging* surface, not
the inter-agent byte pipes in `kernel/src/channel/` (those are the Linux
pipe/socket analog between tasks).

| Concept | Module | Purpose |
|---|---|---|
| **Messaging channel** | `msgchan/` | Telegram (etc.) ↔ shell agent |
| **IPC channel** | `channel/` | Cap-gated stream/datagram between agents |

Config lives on the Synapse store at **`/configs/core/channels.json`** and is
loaded at boot (`msgchan::load`). Running instances are polled cooperatively from
`shell::upkeep` via `msgchan::tick()`.

Shell command: **`/channel`** (alias `/channels`).

---

## Quick start — Discord

Token form: `BOT_TOKEN#CHANNEL_ID` (bot token, hash, channel snowflake to poll).

```text
/channel add discord home <BOT_TOKEN>#<CHANNEL_ID> pairing
/channel start home
```

Uses REST only (`GET /channels/{id}/messages`, `POST …/messages`) — no Gateway
WebSocket. Invite the bot to the channel with Read Message History + Send Messages.

## Quick start — Slack

Token form: `xoxb-…#C01234567` (bot token, hash, channel id).

```text
/channel add slack home <BOT_TOKEN>#<CHANNEL_ID> pairing
/channel start home
```

Uses `conversations.history` + `chat.postMessage`. The bot must be in the channel.

---

## Quick start — Telegram

### 1. Prerequisites

- Chitti booted with **network** up (`/network` or DHCP at boot).
- **HTTPS** works to the public internet (`api.telegram.org`).
- A local or remote **model** if you want auto-replies from the agent
  (`/model` / bundled GGUF). Without a model you can still `/channel send` and
  `/channel reply` from the console.

### 2. Create a bot

1. Open Telegram and chat with **[@BotFather](https://t.me/BotFather)**
   (confirm the handle is exactly `@BotFather`).
2. Run `/newbot`, follow the prompts, and **copy the bot token**
   (`123456789:AA…`).
3. Optional BotFather toggles:
   - `/setprivacy` — for groups, disable privacy if the bot should see all messages
   - `/setjoingroups` — allow/deny adding the bot to groups

### 3. Add and start the channel on Chitti

```text
/channel add telegram home <BOT_TOKEN> pairing
/channel start home
/channel status home
```

- **`home`** is an instance name you choose (any short id).
- **`pairing`** (default) — first DM from a new user must be approved on the
  console. Alternatives: `allowlist`, `open` (see [Access policies](#access-policies)).

On a successful start, Chitti probes Telegram `getMe` and logs the bot username.

### 4. Pair your Telegram user (default policy)

1. Open a **private chat** with your bot and send any message (e.g. `hi`).
2. The bot replies with a **pairing code** (four hex digits), e.g. `AB12`.
3. On the Chitti console:

```text
/channel pair home AB12
```

4. You are now on the allowlist. Further messages are accepted.

To allow yourself without pairing (or a second account):

```text
/channel allow home <YOUR_NUMERIC_TELEGRAM_USER_ID>
```

Find your user id: DM the bot, then `/channel status` / serial logs, or use a
Bot API `getUpdates` call / `@userinfobot` (third-party; less private).

### 5. Talk to the agent

With **`auto_agent` on** (default for new instances):

- Text you send in Telegram is queued as an inbound message.
- The shell loop drains the queue, runs a **shell-agent turn**, and sends the
  reply back on the same chat.
- Built-in remote commands (no model): `/ping`, `/whoami`, `/help`.

From the console you can also push text without waiting for inbound:

```text
/channel send home <chat_id> Hello from Chitti
/channel reply home Got it
```

`/channel reply` uses the **last inbound peer** for that instance.

### 6. Stop / remove

```text
/channel stop home
/channel remove home
```

---

## `/channel` command reference

| Command | Description |
|---|---|
| `/channel` / `/channel list` | List instances (name, kind, running, policy, …) |
| `/channel types` | Available backends (`telegram` today) |
| `/channel add <type> <name> <token> [policy]` | Create an instance |
| `/channel start <name>` | Start polling (`getUpdates` for Telegram) |
| `/channel stop <name>` | Stop polling |
| `/channel remove <name>` | Delete instance + drop from config |
| `/channel status [name]` | Detail: offset, allowlist, last peer, errors, pending pair |
| `/channel allow <name> <user_id\|*>` | Add a sender to the allowlist |
| `/channel pair <name> <CODE>` | Approve a pending DM pairing |
| `/channel send <name> <peer> <text…>` | Outbound text |
| `/channel reply <name> <text…>` | Reply to last inbound peer |
| `/channel help` | Usage summary |

Config file: **`/configs/core/channels.json`**.

---

## Access policies

Set at **add** time: `/channel add telegram <name> <token> <policy>`.

| Policy | Behaviour |
|---|---|
| **`pairing`** (default) | Unknown senders get a one-time code; approve with `/channel pair`. |
| **`allowlist`** | Only numeric ids listed via `/channel allow` (or `"*"`). |
| **`open`** | Any sender if `allow` includes `*` (or empty allow with open — use carefully). |

**Security notes**

- Prefer **pairing** or **allowlist** for personal bots.
- `open` + public bot username lets anyone who finds the bot drive the agent
  (and its tools under current mode). Use only for intentionally public demos
  with tight tool permissions (`/mode`, `/permissions`).
- Pairing grants **DM access only** for that user id; it does not widen filesystem
  or network capabilities by itself.

---

## How messages flow

```text
Telegram user
    │  HTTPS Bot API
    ▼
msgchan::telegram  (getUpdates)
    │  access policy
    ▼
inbound queue  ──►  shell loop  ──►  ChatSession::turn (agent)
    │                                      │
    │                                      ▼
    └──────── sendMessage ◄────────── reply text
```

- **Poll:** `msgchan::tick()` from `shell::upkeep` — short non-blocking HTTP so
  the UI/net stack stay cooperative (Ctrl+C and the clock still work).
- **Agent work:** heavy inference is **not** done in `tick`. The interactive
  shell drains the inbound queue (`drain_channel_inbound`) between prompts.
- **Origin context:** user text is wrapped as  
  `[via telegram/<name> from <display>]\n…` so the model knows the source.

---

## Config shape

`/configs/core/channels.json` is a JSON **array** of objects:

```json
[
  {
    "name": "home",
    "kind": "telegram",
    "token": "123456:AA…",
    "policy": "pairing",
    "allow_from": ["8734062810"],
    "running": true,
    "offset": 42,
    "auto_agent": true
  }
]
```

| Field | Meaning |
|---|---|
| `name` | Instance id used in `/channel …` |
| `kind` | Backend (`telegram`) |
| `token` | Bot credential (opaque string) |
| `policy` | `pairing` \| `allowlist` \| `open` |
| `allow_from` | Sender ids; `"*"` = wildcard |
| `running` | Whether to poll after boot |
| `offset` | Telegram `getUpdates` cursor |
| `auto_agent` | Queue inbound text for the shell agent |

Edit with `/open /configs/core/channels.json` if needed; restart polling with
`/channel stop` / `/channel start` after manual edits, or reboot so `load()`
runs again.

---

## Adding a new channel backend

Messaging is designed so **new platforms do not require a new shell grammar** —
only a kind + adapter.

### 1. Extend `Kind`

In `kernel/src/msgchan/mod.rs`:

```rust
pub enum Kind {
    Telegram,
    Discord,  // new
}

impl Kind {
    pub fn as_str(self) -> &'static str { /* … */ }
    pub fn parse(s: &str) -> Option<Kind> { /* "discord" | "dc" => … */ }
}
```

Update `types()` to advertise the new name.

### 2. Implement the adapter module

Add e.g. `kernel/src/msgchan/discord.rs` that can:

- **Probe** identity at start (optional but recommended).
- **`poll(&mut Instance) -> Result<Vec<…>, String>`** — non-blocking or short
  timeout; advance any cursor on `Instance` (reuse `offset` or add fields).
- **`send_message(token, peer, text) -> Result<(), String>`**.

Normalize inbound into something you can map to `Inbound` (or call
`handle_inbound`-style logic). Prefer HTTPS via `crate::net::http` so dual-arch
and TLS stay consistent.

### 3. Wire match arms

In `msgchan/mod.rs`:

- `start` — optional identity probe  
- `send` — dispatch on `inst.kind`  
- `tick` — `poll` on `inst.kind`  

Keep **access policy** (`DmPolicy`, `allow_from`, pairing) in the generic layer
so every backend shares the same security model.

### 4. Document and test

- Document setup steps in this file (new section under “Quick start”).
- Add unit tests for pure parse/chunk helpers (no QEMU).
- Add an e2e marker for `/channel list` (or a mock HTTP path if you can).

### 5. Do **not** overload `crate::channel`

IPC channels remain for agent-to-agent Synapse primitives. Messaging stays in
`msgchan` so capabilities, taint, and audits for “talk to the internet” stay
obvious at the OS boundary.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `start failed` / `getMe` error | Token wrong; network/DNS/HTTPS to `api.telegram.org`; `/network`, `/ping` |
| Bot silent after DM | Policy: complete **pairing**, or `/channel allow`; check `status` for `pending_pair` / `last_error` |
| No agent reply | Model loaded? (`/model`); `auto_agent` true?; watch serial for `channel[…] → agent` |
| `send failed` | Peer id (numeric chat id); bot blocked user; token revoked |
| Offset stuck / miss messages | Only one poller per bot token; stop other gateways using the same token |
| Config lost after reboot | Persistence needs ext4 data partition (`/install`); else store is in-memory |

Useful commands: `/channel status`, `/network`, `/http https://api.telegram.org/…` (careful with tokens in history).

---

## Related docs

- [README.md](README.md) — project overview and feature list  
- [DEVELOPMENT.md](DEVELOPMENT.md) — build / run / test  
- [DESIGN.md](DESIGN.md) — console brand and UI  
- [CLAUDE.md](CLAUDE.md) — invariants for agents working in-tree  

Kernel entry points: `kernel/src/msgchan/mod.rs`, `kernel/src/msgchan/telegram.rs`,
shell handler `/channel` in `kernel/src/shell/mod.rs`.
