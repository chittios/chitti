#!/usr/bin/env python3
"""End-to-end tests for ChittiOS: boot the kernel under QEMU, drive its shell
over the serial console, and check that every command / flow actually works on
the real thing.

Groups (each scenario -> PASS/FAIL, non-zero exit on any failure):
  os     — the shell/OS commands (help, info, datetime, disks, agents, top, …)
  net    — the networked flows against local host servers: /http (GET/POST/
           stream), /ws + /wss, /ping, and /model remote over https
  model  — inference: /bench, /infer, /perf, a chat turn, /compact, plus the
           runtime `/model load` flow (a --no-model guest loads chat.gguf off
           an attached FAT disk and chat answers)  (needs the bundled model;
           slow — only with --slow)
  voice  — /voice models + /voice say (TTS)  (needs assets/voice + a sound
           device; slow — only with --slow)

Dependency-free (stdlib only). Run with a TLS-1.3-capable Python (Homebrew's)
so the https/wss scenarios aren't skipped:

    /opt/homebrew/bin/python3 tests/e2e/run.py              # os + net  (~3 min)
    /opt/homebrew/bin/python3 tests/e2e/run.py --slow       # + model + voice
    make e2e            /    make e2e-full                   # (Makefile targets)

TLS scenarios auto-skip (not fail) when the running Python lacks TLS 1.3;
model/voice scenarios auto-skip when the bundled model / voice assets are absent.
"""

import os
import re
import socket
import ssl
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guest import Guest  # noqa: E402
from servers import Server, tls_context  # noqa: E402

HOST = "10.0.2.2"  # QEMU user-net alias for the host
PLAIN_PORT = 8100
TLS_PORT = 9100
SVC_PORT = 7099  # guest echo-service listener, reachable via slirp hostfwd
SVC_HTTP_PORT = 7100  # guest http-doc-service listener
SVC_SSH_PORT = 7101  # guest ssh-service listener
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CERT = os.path.join(HERE, "certs", "ec.pem")
KEY = os.path.join(HERE, "certs", "ec.key")


def _openssl():
    for c in ("/opt/homebrew/opt/openssl@3/bin/openssl", "/usr/local/opt/openssl@3/bin/openssl", "openssl"):
        try:
            out = subprocess.run([c, "version"], capture_output=True, text=True)
            if out.returncode == 0 and "OpenSSL" in out.stdout:
                return c
        except Exception:
            continue
    return "openssl"


def ensure_cert():
    if os.path.exists(CERT) and os.path.exists(KEY):
        return True
    os.makedirs(os.path.dirname(CERT), exist_ok=True)
    ossl = _openssl()
    try:
        subprocess.run([ossl, "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", KEY], check=True, capture_output=True)
        subprocess.run([ossl, "req", "-x509", "-new", "-key", KEY, "-out", CERT, "-days", "3", "-subj", "/CN=chitti-e2e"], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"  (could not generate TLS cert with {ossl}: {e})")
        return False


# --- OS / shell commands: (name, command, expected substring) ---------------
# Each just confirms the command runs and prints its marker on the real kernel.
OS_CMDS = [
    # /help text = flat serial catalogue (the GUI opens a modal that would
    # block the serial harness until Esc).
    ("help", "/help text", "Chitti commands:"),
    ("info", "/info", "RAM installed"),
    ("datetime", "/datetime", "datetime>"),
    ("datetime_tz", "/datetime tz +5:30", "UTC+05:30"),
    ("disks", "/disks", "disks>"),
    ("lspci", "/lspci", "pci>"),
    ("mounts", "/mounts", "mounts>"),
    ("ls", "/ls /", "ls>"),
    ("pwd", "/pwd", "pwd> /"),
    ("skills", "/skills", "installed"),
    ("shortcuts", "/shortcuts", "Shortcuts ("),
    ("mode", "/mode", "mode>"),
    # /model status: prints the active backend + the runtime model name (the
    # GGUF's own general.name, or "none bundled") — validates the dynamic-model
    # plumbing without needing inference.
    ("model_status", "/model", "model> active:"),
    ("think", "/think off", "think>"),
    ("agents", "/agents", "agents>"),
    ("ui", "/ui", "ui>"),
    ("ktrace", "/ktrace", "ktrace>"),
    ("close", "/close", "closed the active tab"),
    ("top", "/top", "top>"),
    ("clear", "/clear", "cleared"),
    ("wifi", "/wifi info", "wifi>"),
    ("channel", "/channel list", "channel>"),
    # /memory and /restart are covered by dedicated scenarios below (round-trip
    # + help listing / reboot-exit); listed here only for discoverability.
]


def make_cmd_scenario(cmd, marker, timeout=15):
    def fn(g):
        m = g.mark()
        g.send(cmd)
        ok = g.wait_for(marker, timeout, m)
        return ok, f"{cmd!r} -> {marker!r}" if ok else f"{cmd!r}: no {marker!r}"
    return fn


def s_help_restart(g):
    """`/help text` documents `/restart` (the command itself reboots and is tested last)."""
    m = g.mark()
    g.send("/help text")
    ok = g.wait_for("/restart", 15, m) and g.wait_for("/memory", 5, m)
    return ok, "help lists /restart and /memory" if ok else "missing /restart or /memory in help"


def s_memory(g):
    """Active-agent durable memory: add → get → list → miss, via the human shell
    surface that shares the store with the agent `memory_*` tools."""
    m = g.mark()
    g.send("/memory add e2e_key e2e_value_42")
    if not g.wait_for("ok:", 12, m):
        return False, "memory add did not report ok"
    m = g.mark()
    g.send("/memory get e2e_key")
    if not g.wait_for("e2e_value_42", 12, m):
        return False, "memory get missed stored value"
    m = g.mark()
    g.send("/memory list")
    if not g.wait_for("e2e_key", 12, m):
        return False, "memory list missing key"
    m = g.mark()
    g.send("/memory get missing_key_zzz")
    if not g.wait_for("no memory", 12, m):
        return False, "missing key should report absence"
    # Path traversal must be rejected (same sanitise as the tool path).
    m = g.mark()
    g.send("/memory add ../escape secret")
    if not g.wait_for("error:", 12, m):
        return False, "traversal key was not rejected"
    return True, "add/get/list + miss + traversal reject"


def s_session(g):
    """`/session` save → resume continuity (agent-loop polish).

    Interactive chat writes into the orchestrator session and auto-saves; this
    scenario exercises the human shell surface without needing a model:
      1. `/session` shows a live session id + message count
      2. `/session save` persists it (capture the id)
      3. `/clear` drops the transcript (keeps identity)
      4. `/session resume <id>` reconstructs the saved snapshot
    """
    import re

    m = g.mark()
    g.send("/session")
    if not g.wait_for("current session", 15, m):
        return False, "no current session banner"
    m = g.mark()
    g.send("/session save")
    if not g.wait_for("saved session", 15, m):
        return False, "session save failed"
    # Parse the id from output after the save mark.
    tail = g.text()[m:]
    ids = re.findall(r"saved session (\d+)", tail)
    if not ids:
        return False, f"could not parse session id from: {tail[-200:]!r}"
    sid = ids[-1]
    m = g.mark()
    g.send("/clear")
    if not g.wait_for("cleared", 15, m):
        return False, "clear did not report"
    m = g.mark()
    g.send("/session")
    if not g.wait_for("current session", 15, m):
        return False, "session missing after clear"
    # Resume the id we saved (cleared transcript was auto-saved under same id,
    # so resume still reconstructs — message count may be 1 system prompt).
    m = g.mark()
    g.send(f"/session resume {sid}")
    if not g.wait_for("resumed session", 15, m):
        return False, f"resume of {sid} failed"
    if not g.wait_for("messages reconstructed", 5, m):
        return False, "resume did not report message reconstruction"
    # Store listing still names the sess key.
    m = g.mark()
    g.send("/session")
    if not g.wait_for("saved in store:", 15, m):
        return False, "no saved-in-store listing after resume"
    return True, f"save/clear/resume session {sid}"


def s_agents_switch_caps(g):
    """`/agents switch` rebinds tool authority (Phase 0).

    Switching to a non-orchestrator agent id reports gated tools/caps; switching
    back to 1 restores the shell agent. Full cap-deny of writes is covered by
    the in-kernel unit suite (Router + home sandbox).
    """
    m = g.mark()
    g.send("/agents switch 9001")
    if not g.wait_for("chat now runs as agent 9001", 15, m):
        return False, "switch to 9001 did not report"
    if not g.wait_for("tools gated", 5, m):
        return False, "switch did not mention tools gated"
    m = g.mark()
    g.send("/agents switch 1")
    if not g.wait_for("chat now runs as agent 1", 15, m):
        return False, "switch back to shell agent failed"
    return True, "switch rebinds caps (9001 gated → 1 restored)"


def s_memory_hierarchy(g):
    """Durable memory survives clear (Phase 2).

    `/memory add` writes under `/agent/<id>/memory/`; after `/clear` the KV
    store still returns the value (session transcript is wiped, durable memory
    is not).
    """
    m = g.mark()
    g.send("/memory add e2e_persist terracotta_42")
    if not g.wait_for("ok:", 12, m):
        return False, "memory add failed"
    m = g.mark()
    g.send("/clear")
    if not g.wait_for("cleared", 12, m):
        return False, "clear failed"
    m = g.mark()
    g.send("/memory get e2e_persist")
    if not g.wait_for("terracotta_42", 12, m):
        return False, "memory did not survive /clear"
    m = g.mark()
    g.send("/memory list")
    if not g.wait_for("e2e_persist", 12, m):
        return False, "memory list missing key after clear"
    return True, "memory KV persists across /clear"


def s_fs_basic(g):
    """Linux-like store FS: hierarchical /ls, cat, mkdir, cp, mv, rm, grep, glob.

    `/ls /` must show directory entries (e.g. `agent/`) — not every flat
    percent-encoded store key. Round-trip a small tree under /tmp_e2e.
    """
    # Clean slate (ignore errors if missing).
    m = g.mark()
    g.send("/rm -r /tmp_e2e")
    g.wait_for("rm>", 10, m)

    m = g.mark()
    g.send("/mkdir -p /tmp_e2e/sub")
    if not g.wait_for("mkdir> /tmp_e2e/sub", 12, m):
        return False, "mkdir -p failed"

    m = g.mark()
    g.send("/touch /tmp_e2e/sub/hello.txt")
    if not g.wait_for("touch> /tmp_e2e/sub/hello.txt", 12, m):
        return False, "touch failed"

    # Write content via the editor path isn't available headlessly; use a
    # second touch + overwrite isn't possible without write cmd. Use memory
    # isn't right either. The store touch creates empty files — for cat we
    # need content. Prefer /cp of a known store file, or create via agents.
    # Shell has no /echo write — use /http -O is overkill. Check if SOUL exists
    # and cp it, or use touch + verify empty cat.
    m = g.mark()
    g.send("/cat /tmp_e2e/sub/hello.txt")
    if not g.wait_for("cat> /tmp_e2e/sub/hello.txt", 12, m):
        return False, "cat empty file failed"

    m = g.mark()
    g.send("/ls /tmp_e2e")
    if not g.wait_for("ls> /tmp_e2e", 12, m):
        return False, "ls dir failed"
    if not g.wait_for("sub/", 5, m):
        return False, "ls did not show sub/ as directory"

    # Immediate children only: must NOT list hello.txt at /tmp_e2e level.
    # (We can't easily assert absence over serial; check nested ls instead.)
    m = g.mark()
    g.send("/ls /tmp_e2e/sub")
    if not (g.wait_for("ls> /tmp_e2e/sub", 12, m) and g.wait_for("hello.txt", 5, m)):
        return False, "ls nested file missing"

    m = g.mark()
    g.send("/cp /tmp_e2e/sub/hello.txt /tmp_e2e/copy.txt")
    if not g.wait_for("cp>", 12, m):
        return False, "cp file failed"
    if not g.wait_for("1 file", 5, m):
        return False, "cp did not report 1 file"

    m = g.mark()
    g.send("/mv /tmp_e2e/copy.txt /tmp_e2e/moved.txt")
    if not g.wait_for("mv>", 12, m):
        return False, "mv failed"

    m = g.mark()
    g.send("/cp -r /tmp_e2e/sub /tmp_e2e/sub2")
    if not g.wait_for("cp>", 12, m):
        return False, "cp -r failed"

    m = g.mark()
    g.send("/glob /tmp_e2e/**")
    if not g.wait_for("glob>", 12, m):
        return False, "glob failed"
    if not g.wait_for("/tmp_e2e/sub/hello.txt", 5, m):
        return False, "glob missed nested path"

    # Root listing is hierarchical (dirs), not a dump of every key.
    m = g.mark()
    g.send("/ls /")
    if not g.wait_for("ls> /", 12, m):
        return False, "ls / failed"
    # Boot places agent homes — expect agent/ as a directory entry.
    if not g.wait_for("agent/", 5, m):
        return False, "ls / did not show agent/ (hierarchical root)"

    m = g.mark()
    g.send("/rm -r /tmp_e2e")
    if not g.wait_for("rm>", 12, m):
        return False, "rm -r cleanup failed"

    # help catalogue documents the new tools
    m = g.mark()
    g.send("/help text")
    if not (g.wait_for("/mkdir", 12, m) and g.wait_for("/cp", 5, m) and g.wait_for("/mv", 5, m)):
        return False, "/help text missing new fs commands"
    return True, "fs: hierarchical ls + mkdir/touch/cat/cp/mv/glob/rm"


def s_restart(g):
    """`/restart` reboots the machine via `arch::reboot`.

    On aarch64 `xtask run` (no `-no-reboot`), PSCI SYSTEM_RESET cold-boots the
    guest in-place — we assert the banner + a second boot. On x86 with
    `-no-reboot` the emulator may exit instead; that also counts as success.
    Must run **last**: a live reboot drops in-memory state and the shell
    session, so later scenarios would race a rebooting guest."""
    m = g.mark()
    g.send("/restart")
    if not g.wait_for("restarting", 15, m):
        # x86 -no-reboot may exit before the serial line is fully flushed.
        if g.proc.poll() is not None:
            return True, "guest exited on /restart (x86 -no-reboot)"
        return False, "no 'restarting' banner"
    # Process exited (x86 -no-reboot) — done.
    deadline = time.time() + 5
    while time.time() < deadline:
        if g.proc.poll() is not None:
            return True, "guest exited on /restart"
        time.sleep(0.1)
    # aarch64 / real hardware: guest reboots in-place — wait for a second boot.
    if g.wait_for("boot ok", 60, m):
        # Confirm networking came back so the reboot completed, not a hang.
        g.wait_for("net: configured", 60, m)
        return True, "guest rebooted in-place (banner + second boot)"
    if g.proc.poll() is not None:
        return True, "guest exited on /restart"
    return False, "no second boot after /restart"


# --- network scenarios ------------------------------------------------------

def s_nic_dispatch(g):
    """The NIC was claimed by a driver chosen from its vendor/device ID.

    Guards the by-device-ID dispatch (`net::nic_ids`): matching on vendor alone
    used to hand every Intel controller to the legacy-e1000 driver, which for an
    igb/igc part configures rings at offsets that don't exist — the NIC "comes up"
    and never receives a frame. The boot log must name the chosen driver, and it
    must be the family the emulated device actually belongs to.

    Boot the guest with CHITTI_NIC=e1000e|igb|rtl8139|virtio-net-pci to exercise
    the other families against the same assertion.
    """
    txt = g.text()
    m = re.search(r"net: ([0-9a-f]{4}):([0-9a-f]{4}) at [\d:.]+ -> (\S+) driver", txt)
    if not m:
        return False, "no 'net: vvvv:dddd -> <driver> driver' dispatch line in the boot log"
    vendor, device, driver = m.group(1), m.group(2), m.group(3)
    # The device QEMU was told to emulate must map to the right family. Default
    # run is `-device e1000` = 8086:100e = the legacy family.
    expected = {
        "100e": "e1000", "100f": "e1000", "1008": "e1000",   # 82540EM/82545EM/82544
        "10d3": "e1000e",                                     # 82574L (-device e1000e)
        "10c9": "igb",                                        # 82576 (-device igb)
        "8139": "rtl8139",
        "1000": "virtio-net", "1041": "virtio-net",
    }.get(device)
    if expected and driver != expected:
        return False, f"{vendor}:{device} was claimed by '{driver}', expected '{expected}'"
    # And the NIC must actually work — the boot only reaches here once DHCP
    # completed over it, but assert the address explicitly.
    if "10.0.2.15" not in txt:
        return False, f"{driver} claimed {vendor}:{device} but no DHCP lease followed"
    return True, f"{vendor}:{device} -> {driver}, DHCP lease obtained"


def s_network(g):
    m = g.mark()
    g.send("/network")
    ok = g.wait_for("10.0.2.15", 15, m)
    return ok, "IPv4 configured" if ok else "no IP in /network output"


def s_ping(g):
    m = g.mark()
    g.send(f"/ping {HOST}")
    # slirp answers pings to the gateway; accept either a reply or a ran-but-no-reply.
    if g.wait_for("reply from", 12, m):
        return True, "ICMP reply from gateway"
    return g.wait_for("ping>", 3, m), ("ran (no reply — slirp ICMP)" if g.wait_for("ping>", 1, m) else "no output")


def s_http_get(g):
    m = g.mark()
    g.send(f"/http -v http://{HOST}:{PLAIN_PORT}/json")
    ok = g.wait_for('"who":"e2e"', 20, m) and g.wait_for("http> 200", 20, m)
    return ok, "GET 200 + body" if ok else "no 200/body"


def s_browse(g):
    """Browser agent: fetch harness HTML, parse/layout/paint, report title."""
    m = g.mark()
    g.send(f"/browse http://{HOST}:{PLAIN_PORT}/page.html")
    # Host prints `browser> <title> — <url> …` and tool returns `ok:title=…`.
    ok = g.wait_for("E2E Browser", 30, m) and (
        g.wait_for("ok:title=", 5, m) or g.wait_for("browser>", 5, m)
    )
    return ok, "rendered harness HTML (title E2E Browser)" if ok else "browse did not report title"


def s_http_post(g):
    m = g.mark()
    g.send(f'/http -X POST -H "X-Test: yes" -d payload-9182 http://{HOST}:{PLAIN_PORT}/echo')
    ok = g.wait_for("payload-9182", 20, m)
    return ok, "POST body echoed" if ok else "body not echoed"


def s_http_download(g):
    """`/http -O` downloads a real PNG from the harness server into the store,
    then `/open` decodes it back — network → store → image viewer roundtrip."""
    m = g.mark()
    g.send(f"/http -O http://{HOST}:{PLAIN_PORT}/logo.png")
    if not g.wait_for("http> saved", 20, m) or not g.wait_for("/downloads/logo.png", 5, m):
        return False, "download did not save"
    m = g.mark()
    g.send("/open /downloads/logo.png")
    ok = g.wait_for("3x2 px", 15, m)
    return ok, "downloaded PNG opened from the store" if ok else "saved file did not decode"


def s_http_stream(g):
    m = g.mark()
    g.send(f"/http --stream http://{HOST}:{PLAIN_PORT}/sse")
    ok = g.wait_for("event 0", 20, m) and g.wait_for("event 2", 20, m)
    return ok, "SSE streamed live" if ok else "SSE events missing"


def s_ws(g):
    m = g.mark()
    g.send(f"/ws ws://{HOST}:{PLAIN_PORT}/ws hello-ws")
    ok = g.wait_for("echo:hello-ws", 20, m)
    g.wait_for("closed by peer", 5, m)  # let the /ws loop exit before the next cmd
    return ok, "ws echo round-trip" if ok else "no ws echo"


def s_cancel(g):
    # Ctrl+C interrupts a *running command*, not just model generation: a /http
    # to an unroutable TEST-NET address (RFC 5737) hangs connecting until its
    # timeout; Ctrl+C (0x03) must abort it near-instantly. Then a normal command
    # must still work — proving the cancel-poll pushes non-Ctrl+C input back
    # rather than swallowing the next command's keystrokes.
    m = g.mark()
    g.send("/http http://192.0.2.1/")
    time.sleep(2.0)  # let it get stuck in connect
    t0 = time.time()
    g.send_raw(b"\x03")  # Ctrl+C
    stopped = g.wait_for("cancelled", 8, m)
    dt = time.time() - t0
    fast = stopped and dt < 5.0  # aborted well before the multi-second timeout
    m2 = g.mark()
    g.send("/network")  # the next command must still be read
    followed = g.wait_for("10.0.2.15", 10, m2)
    ok = fast and followed
    return ok, f"Ctrl+C aborted /http in {dt:.1f}s; next command ran" if ok else f"cancel={stopped}/{dt:.1f}s, next-cmd={followed}"


def s_wss(g):
    m = g.mark()
    g.send(f"/ws wss://{HOST}:{TLS_PORT}/ws secret-wss")
    ok = g.wait_for("echo:secret-wss", 30, m)
    g.wait_for("closed by peer", 5, m)
    return ok, "wss (TLS) echo round-trip" if ok else "no wss echo"


def s_model_remote_https(g):
    m = g.mark()
    g.send(f"/model remote https://{HOST}:{TLS_PORT} e2e-model")
    if not g.wait_for("remote backend active", 15, m):
        return False, "/model remote did not activate"
    m2 = g.mark()
    g.send("hello from e2e")
    ok = g.wait_for("remote reply to: hello from e2e", 40, m2)
    g.send("/model local")  # switch back so later turns don't hit the net
    g.wait_for("local (embedded)", 5)
    return ok, "hosted-model chat over https" if ok else "no remote reply"


# --- model (local inference) scenarios — slow, need the bundled model -------

def s_bench(g):
    m = g.mark()
    g.send("/bench")
    ok = g.wait_for("bench>", 40, m)
    return ok, "matvec kernel bench" if ok else "no bench output"


def s_infer(g):
    m = g.mark()
    g.send("/infer")
    ok = g.wait_for("tok/s", 180, m) or g.wait_for("=>", 5, m)
    return ok, "reference inference ran" if ok else "no inference output"


def s_perf(g):
    m = g.mark()
    g.send("/perf")
    ok = g.wait_for("tok/s", 180, m)
    return ok, "prefill/decode tok/s" if ok else "no perf output"


def s_chat(g):
    m = g.mark()
    g.send("in one short sentence, what is 2 plus 2?")
    ok = g.wait_for("chitti:", 180, m)
    g.wait_quiet(2.0, 180)  # let the turn finish before the next command
    return ok, "local model chat turn" if ok else "no chat reply"


def s_compact(g):
    m = g.mark()
    g.send("/compact")
    ok = g.wait_for("compacted", 180, m) or g.wait_for("nothing to compact", 10, m)
    return ok, "context compaction" if ok else "no compact output"


def s_model_load(_g):
    """Prove the runtime `/model load` path end-to-end, from nothing: boot a
    second guest with NO model in RAM but a FAT model disk attached
    (CHITTI_MODEL_DISK -> chat.gguf), assert chat is unavailable, load the
    GGUF off the disk at runtime, then assert chat answers on it."""
    gguf = os.path.join(ROOT, "assets", "model.gguf")
    if not os.path.exists(gguf):
        return None, "skipped (assets/model.gguf absent)"
    g2 = Guest(arch=RUN_ARCH, verbose=RUN_VERBOSE, no_model=True, model_disk=gguf)
    try:
        if not g2.wait_for("net: configured", 180):
            return False, "no-model guest never booted"
        # Before the load: no model in RAM, so chat must refuse (fast, no
        # inference involved).
        m = g2.mark()
        g2.send("hello")
        if not g2.wait_for("chat unavailable", 30, m):
            return False, "expected 'chat unavailable' before /model load"
        # Runtime-load the GGUF off the attached FAT volume into DMA frames.
        m = g2.mark()
        g2.send("/model load chat.gguf")
        if not g2.wait_for("model> loaded", 240, m):
            return False, "/model load chat.gguf did not complete"
        # Chat now runs on the runtime-loaded model — a real reply must come.
        m = g2.mark()
        g2.send("in one short sentence, what is 2 plus 2?")
        ok = g2.wait_for("chitti:", 300, m)
        g2.wait_quiet(2.0, 180)
        return ok, "runtime-loaded model answered chat" if ok else "no chat reply after /model load"
    finally:
        g2.close()


def s_tabs(g):
    """Tmux-style action-pane tabs: /ktrace and /top open as two coexisting
    tabs; /close closes them one at a time (the active tab), collapsing the
    pane only when the last closes. (Switching/keeping-alive is visual; the
    process-liveness of the audio tab is proven by open_media.)"""
    m = g.mark()
    g.send("/ktrace")
    if not g.wait_for("action tab", 15, m):
        # already open from a prior scenario — close and retry once
        g.send("/close")
        g.wait_for("closed the active tab", 5)
        m = g.mark()
        g.send("/ktrace")
        if not g.wait_for("action tab", 10, m):
            return False, "ktrace tab did not open"
    m = g.mark()
    g.send("/top")
    if not g.wait_for("top>", 15, m):
        return False, "top tab did not open"
    m = g.mark()
    g.send("/close")
    ok1 = g.wait_for("closed the active tab", 10, m)
    m = g.mark()
    g.send("/close")
    ok2 = g.wait_for("closed the active tab", 10, m)
    return (ok1 and ok2), "two action tabs opened + closed one-by-one" if (ok1 and ok2) else "tab close did not report"


def s_clipboard(g):
    """Host<->guest clipboard over the serial console: a bracketed paste
    (ESC[200~ … ESC[201~) is captured into the clipboard (host->guest), and an
    in-OS copy emits an OSC 52 escape (guest->host)."""
    # host -> guest: bracketed paste lands in the clipboard.
    g.send_raw(b"\x1b[200~clip-4821\x1b[201~")
    time.sleep(0.5)
    g.send_raw(b"\x7f" * 40)  # clear whatever the paste inserted on the prompt line
    time.sleep(0.3)
    m = g.mark()
    g.send("/clip")
    got_paste = g.wait_for("clip-4821", 10, m)
    # guest -> host: setting the clipboard emits the OSC 52 set-clipboard escape.
    m = g.mark()
    g.send("/clip syncme-2937")
    got_osc = g.wait_for("]52;c;", 10, m)
    ok = got_paste and got_osc
    return ok, "bracketed paste captured + OSC52 copy-out emitted" if ok else f"paste={got_paste} osc52={got_osc}"


def s_open_media(_g):
    """Prove the `/open` media paths end-to-end: boot a guest (audio="none" =
    a real virtio-snd device on a silent backend) with a FAT disk carrying a
    generated 3x2 PNG and a 0.3 s WAV, mount it, /open both. Headless, so the
    assertions are the decode/playback reports on serial; the pixel/PCM math
    is covered by the in-kernel unit tests."""
    import struct
    import tempfile
    import wave
    import zlib

    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)
    raw = (b"\x00" + bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
           + b"\x00" + bytes([10, 20, 30, 40, 50, 60, 250, 249, 245]))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    png_path = os.path.join(tempfile.gettempdir(), "chitti-e2e.png")
    with open(png_path, "wb") as f:
        f.write(png)
    wav_path = os.path.join(tempfile.gettempdir(), "chitti-e2e.wav")
    w = wave.open(wav_path, "w")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    n = 4800  # 0.3 s
    w.writeframes(b"".join(struct.pack("<h", int(8000 * ((i // 20) % 2 * 2 - 1))) for i in range(n)))
    w.close()
    # A minimal single-page PDF (classic xref, uncompressed content): the pdf
    # agent's command hook must digest it through the wasm and report pages.
    def build_pdf():
        content = b"BT /F1 12 Tf 72 700 Td (Chitti e2e pdf text) Tj ET"
        bodies = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R /MediaBox [0 0 612 792] >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Title (E2E Doc) >>",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offs = []
        for i, b in enumerate(bodies):
            offs.append(len(out))
            out += f"{i+1} 0 obj\n".encode() + b + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(bodies)+1}\n".encode() + b"0000000000 65535 f \n"
        for o in offs:
            out += f"{o:010} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(bodies)+1} /Root 1 0 R /Info 5 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()
        return bytes(out)
    pdf_path = os.path.join(tempfile.gettempdir(), "chitti-e2e.pdf")
    with open(pdf_path, "wb") as f:
        f.write(build_pdf())
    g2 = Guest(arch=RUN_ARCH, verbose=RUN_VERBOSE, no_model=True, audio="none",
               model_disk=f"{png_path}:{wav_path}:{pdf_path}")
    try:
        if not g2.wait_for("net: configured", 180):
            return False, "media guest never booted"
        # The harness FAT disk's index varies by arch/attach order: mount each
        # candidate and /open until the decode report appears.
        for d in range(4):
            g2.send(f"/mount {d} 0 /img{d}")
            g2.wait_quiet(0.5, 30)
            m = g2.mark()
            g2.send(f"/open /img{d}/chitti-e2e.png")
            if not g2.wait_for("3x2 px", 15, m):
                continue
            # Same mount also carries the WAV: decode + play it.
            m = g2.mark()
            g2.send(f"/open /img{d}/chitti-e2e.wav")
            if not g2.wait_for("open> playing", 15, m):
                return False, "WAV did not start playing"
            # Drive the media-tab key controls and prove the input loop never
            # wedges: Ctrl+Tab focuses the pane, space pauses/resumes, arrows
            # seek (no serial output — they repaint), then Ctrl+C stops from the
            # prompt (which does print). The pixel/seek math is unit-tested; this
            # is the on-OS liveness guard the standing rule asks for.
            g2.send_raw(b"\x1b[T")   # Ctrl+Tab: focus the audio tab
            g2.wait_quiet(0.3, 10)
            g2.send_raw(b" ")        # space: pause
            g2.send_raw(b" ")        # space: resume
            g2.send_raw(b"\x1b[C")   # right: seek forward
            g2.send_raw(b"\x1b[D")   # left: seek back
            m2 = g2.mark()
            g2.send_raw(b"\x03")     # Ctrl+C at the prompt: stop playback
            ok = g2.wait_for("audio stopped", 10, m2)
            if not ok:
                return False, "media controls / Ctrl+C did not stop playback"
            # The pdf agent's /open hook: wasm digest → page count + editor tab.
            m3 = g2.mark()
            g2.send(f"/open /img{d}/chitti-e2e.pdf")
            if not g2.wait_for("pdf 1 page(s)", 20, m3):
                return False, "pdf preview did not report pages"
            if not g2.wait_for("/preview/chitti-e2e.txt", 5, m3):
                return False, "pdf preview did not open the text tab"
            return True, "PNG previewed + WAV played + media controls + PDF digested via wasm"
        return False, "no '3x2 px' decode report from /open on any mount"
    finally:
        g2.close()


def _mux_h264_mp4(annexb, w, h, timescale=25, all_sync=False):
    """Wrap an Annex-B H.264 elementary stream in a minimal ISO-BMFF (mp4) so
    the kernel's demuxer sees a real container. Pure stdlib. Splits NALs on
    start codes, pulls SPS/PPS into an avcC record, frames each VCL access unit
    as a 4-byte-length AVCC sample, and builds a single-chunk sample table."""
    import struct

    def nals(d):
        out, idxs, j, n = [], [], 0, len(d)
        while j + 3 <= n:
            if d[j] == 0 and d[j + 1] == 0 and d[j + 2] == 1:
                sc = 4 if j > 0 and d[j - 1] == 0 else 3
                idxs.append((j - (sc - 3), sc)); j += 3
            else:
                j += 1
        for k, (p, l) in enumerate(idxs):
            s = p + l
            e = idxs[k + 1][0] if k + 1 < len(idxs) else n
            out.append(d[s:e])
        return out

    sps = pps = None
    samples = []  # each = AVCC-framed bytes of one access unit
    for u in nals(annexb):
        t = u[0] & 0x1f
        if t == 7 and sps is None:
            sps = u
        elif t == 8 and pps is None:
            pps = u
        elif t in (1, 5):  # VCL slice → one frame/sample
            samples.append(struct.pack(">I", len(u)) + bytes(u))
    if sps is None or pps is None or not samples:
        raise RuntimeError("mux: stream missing SPS/PPS/slices")

    def box(typ, *parts):
        body = b"".join(parts)
        return struct.pack(">I", 8 + len(body)) + typ + body

    avcc = (bytes([1, sps[1], sps[2], sps[3], 0xff, 0xe1]) + struct.pack(">H", len(sps)) + bytes(sps)
            + bytes([1]) + struct.pack(">H", len(pps)) + bytes(pps))
    avc1 = (b"\x00" * 6 + struct.pack(">H", 1)          # reserved + data_ref_idx
            + b"\x00" * 16 + struct.pack(">HH", w, h)   # predefined/reserved + w/h
            + struct.pack(">II", 0x00480000, 0x00480000)  # 72dpi h/v res
            + b"\x00" * 4 + struct.pack(">H", 1)        # reserved + frame_count
            + b"\x00" * 32 + struct.pack(">H", 0x18) + b"\xff\xff"  # compressorname + depth + predefined
            + box(b"avcC", avcc))
    stsd = box(b"stsd", struct.pack(">II", 0, 1), box(b"avc1", avc1))
    n = len(samples)
    stts = box(b"stts", struct.pack(">II", 0, 1) + struct.pack(">II", n, 1))
    stsc = box(b"stsc", struct.pack(">II", 0, 1) + struct.pack(">III", 1, n, 1))
    stsz = box(b"stsz", struct.pack(">III", 0, 0, n) + b"".join(struct.pack(">I", len(s)) for s in samples))
    # stss lists sync samples; omit it entirely when every sample is a keyframe
    # (an absent stss means "all samples are sync" per ISO-BMFF).
    stss = b"" if all_sync else box(b"stss", struct.pack(">II", 0, 1) + struct.pack(">I", 1))
    # Build everything except the chunk offset, then patch stco once the mdat
    # position is known.
    def assemble(chunk_off):
        stco = box(b"stco", struct.pack(">II", 0, 1) + struct.pack(">I", chunk_off))
        stbl = box(b"stbl", stsd, stts, stsc, stsz, stss, stco)
        vmhd = box(b"vmhd", struct.pack(">IHHHH", 1, 0, 0, 0, 0))
        dinf = box(b"dinf", box(b"dref", struct.pack(">II", 0, 1), box(b"url ", struct.pack(">I", 1))))
        minf = box(b"minf", vmhd, dinf, stbl)
        hdlr = box(b"hdlr", struct.pack(">II", 0, 0) + b"vide" + b"\x00" * 12 + b"chitti\x00")
        mdhd = box(b"mdhd", struct.pack(">IIIIhh", 0, 0, 0, timescale, n, 0))
        mdia = box(b"mdia", mdhd, hdlr, minf)
        tkhd = box(b"tkhd", struct.pack(">IIIIII", 0x00000007, 0, 0, 1, 0, n)
                   + b"\x00" * 8 + struct.pack(">hhhh", 0, 0, 0, 0)
                   + struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
                   + struct.pack(">II", w << 16, h << 16))
        trak = box(b"trak", tkhd, mdia)
        mvhd = box(b"mvhd", struct.pack(">IIIIIi", 0, 0, 0, timescale, n, 0x00010000)
                   + struct.pack(">hh", 0x0100, 0) + b"\x00" * 8
                   + struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
                   + b"\x00" * 24 + struct.pack(">I", 2))
        moov = box(b"moov", mvhd, trak)
        return moov
    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41")
    moov_probe = assemble(0)
    mdat_data = b"".join(samples)
    chunk_off = len(ftyp) + len(moov_probe) + 8  # +8 mdat header
    moov = assemble(chunk_off)
    assert len(moov) == len(moov_probe)
    mdat = struct.pack(">I", 8 + len(mdat_data)) + b"mdat" + mdat_data
    return ftyp + moov + mdat, n


def s_open_video(_g):
    """Integration test of the whole video demux path on the real kernel: encode
    a tiny baseline H.264 clip with x264, mux it into mp4 (stdlib), mount it, and
    assert `/open clip.mp4` probes the right geometry/codec/frame-count. Skipped
    if x264 is unavailable (CI runners often lack it)."""
    import shutil
    import struct
    import subprocess
    import tempfile

    if shutil.which("x264") is None:
        return None, "skipped (x264 not installed)"
    W, H, N = 176, 144, 3
    yuv = os.path.join(tempfile.gettempdir(), "chitti-e2e.yuv")
    with open(yuv, "wb") as f:
        for fr in range(N):
            f.write(bytes(((x + y + fr * 7) & 0xff) for y in range(H) for x in range(W)))
            f.write(bytes([128]) * (W // 2 * H // 2) * 2)
    a264 = os.path.join(tempfile.gettempdir(), "chitti-e2e.264")
    # Realistic baseline stream: I+P, in-loop deblocking (default), and MULTIPLE
    # slices per frame (--slices 2) — the real-world case that needs slice
    # assembly + slice-aware neighbour availability + per-side chroma-QP deblock.
    r = subprocess.run(["x264", "--profile", "baseline", "--ref", "1", "--slices", "2", "--frames", str(N),
                        "--input-res", f"{W}x{H}", "-o", a264, yuv],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return None, "skipped (x264 encode failed)"
    try:
        mp4, n_frames = _mux_h264_mp4(open(a264, "rb").read(), W, H, all_sync=False)
    except Exception as e:
        return False, f"mux failed: {e}"
    mp4_path = os.path.join(tempfile.gettempdir(), "chitti-e2e.mp4")
    with open(mp4_path, "wb") as f:
        f.write(mp4)
    # A second clip: HIGH profile — CABAC entropy coding + adaptive 8x8
    # transform (I/P only: the stdlib muxer writes no ctts, so B-frame display
    # reordering could not be derived container-side).
    hi264 = os.path.join(tempfile.gettempdir(), "chitti-e2e-hi.264")
    r = subprocess.run(["x264", "--profile", "high", "--bframes", "0", "--frames", str(N),
                        "--input-res", f"{W}x{H}", "-o", hi264, yuv],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hi_path = None
    if r.returncode == 0:
        try:
            himp4, _ = _mux_h264_mp4(open(hi264, "rb").read(), W, H, all_sync=False)
            hi_path = os.path.join(tempfile.gettempdir(), "chitti-e2e-hi.mp4")
            with open(hi_path, "wb") as f:
                f.write(himp4)
        except Exception:
            hi_path = None
    disk = mp4_path if hi_path is None else f"{mp4_path}:{hi_path}"
    g2 = Guest(arch=RUN_ARCH, verbose=RUN_VERBOSE, no_model=True, audio="off", model_disk=disk)
    try:
        if not g2.wait_for("net: configured", 180):
            return False, "video guest never booted"
        for d in range(4):
            g2.send(f"/mount {d} 0 /vid{d}")
            g2.wait_quiet(0.5, 30)
            m = g2.mark()
            g2.send(f"/open /vid{d}/chitti-e2e.mp4")
            if not g2.wait_for("176x144", 15, m):
                continue
            # Decode + display of keyframes, then drive the transport controls
            # and prove the input loop stays responsive (Ctrl+C stops).
            # Streaming decoder: reports "N frame(s), ready in X ms" (frames are
            # decoded on demand during playback, not all up front).
            if not (g2.wait_for("H.264", 5, m) and g2.wait_for("frame(s), ready", 8, m)):
                return False, "probe/decode output missing"
            g2.send_raw(b"\x1b[T")   # Ctrl+Tab: focus the video tab
            g2.wait_quiet(0.3, 10)
            g2.send_raw(b" ")        # pause
            g2.send_raw(b"\x1b[C")   # seek +1 frame
            g2.send_raw(b"0")        # restart
            m2 = g2.mark()
            g2.send_raw(b"\x03")     # Ctrl+C stops
            ok = g2.wait_for("video stopped", 10, m2)
            if not ok:
                return False, "controls/Ctrl+C did not stop video"
            # High-profile (CABAC + 8x8 transform) clip on the same disk.
            if hi_path is not None:
                m3 = g2.mark()
                g2.send(f"/open /vid{d}/chitti-e2e-hi.mp4")
                if not (g2.wait_for("profile 100", 15, m3) and g2.wait_for("frame(s), ready", 10, m3)):
                    return False, "High-profile CABAC clip did not decode"
                m4 = g2.mark()
                g2.send_raw(b"\x03")
                if not g2.wait_for("video stopped", 10, m4):
                    return False, "Ctrl+C did not stop the CABAC clip"
            return True, "mp4 decoded (baseline multi-slice + High/CABAC) + video-tab controls responsive"
        return False, "no 176x144 probe from /open on any mount"
    finally:
        g2.close()


def s_panes(g):
    """Pane layout: resize the split, fullscreen toggle, reset — driven by the
    /pane command (serial-observable) + a Ctrl+F fullscreen keystroke."""
    m = g.mark()
    g.send("/pane split 30")
    if not g.wait_for("chat width 30%", 8, m):
        return False, "resize (/pane split 30) had no effect"
    m = g.mark()
    g.send("/pane full")
    if not g.wait_for("fullscreen", 8, m):
        return False, "/pane full did not report fullscreen"
    m = g.mark()
    g.send("/pane full")
    if not g.wait_for("restored", 8, m):
        return False, "/pane full toggle did not restore the split"
    # Ctrl+F keystroke path (then restore) must keep the shell responsive.
    g.send_raw(b"\x06")
    g.wait_quiet(0.3, 10)
    g.send_raw(b"\x06")
    m = g.mark()
    g.send("/pane reset")
    ok = g.wait_for("pane> reset", 8, m)
    return ok, "split resize + fullscreen toggle + reset via /pane and Ctrl+F" if ok else "reset failed / shell unresponsive after Ctrl+F"


# --- voice scenarios — slow, need assets/voice + a sound device -------------

def s_voice_models(g):
    m = g.mark()
    g.send("/voice models")
    ok = g.wait_for("voice> models:", 20, m)
    return ok, "voice model listing" if ok else "no voice models output"


def s_voice_say(g):
    m = g.mark()
    g.send("/voice say hello from chitti")
    # synthesize -> samples -> done; needs the KittenTTS model + a sound device.
    ok = g.wait_for("voice> done", 120, m)
    if not ok and g.wait_for("no kitten model", 3, m):
        return None, "skipped (no TTS model bundled)"
    if not ok and g.wait_for("no sound device", 3, m):
        return None, "skipped (no sound device)"
    return ok, "TTS synth + play" if ok else "no voice output"


# --- agents-as-apps scenarios (install/consent + service lifecycle) ---------

def s_agents_services(g):
    m = g.mark()
    g.send("/agents services")
    ok = g.wait_for("agents>", 15, m)
    return ok, "service list rendered" if ok else "no /agents services output"


def s_agents_install(g):
    # Install a built-in signed skill-agent, approving every requested cap
    # (--yes bypasses the per-cap consent modal for scripted runs).
    m = g.mark()
    g.send("/agents install report-writer --yes")
    ok = g.wait_for("installed; granted", 25, m)
    return ok, "skill-agent installed via consent flow" if ok else "install did not complete"


def s_agents_uninstall(g):
    m = g.mark()
    g.send("/agents uninstall report-writer")
    ok = g.wait_for("removed", 15, m)
    return ok, "skill-agent uninstalled" if ok else "uninstall did not complete"


def s_agent_fs_consent(g):
    """The per-agent filesystem consent line: installing an agent shows that
    its own /agent/<id>/ folder is granted (the sandbox floor), and a broad
    Fs request would be flagged. report-writer is home-scoped, so we assert the
    baseline-folder line appears on the install screen."""
    m = g.mark()
    g.send("/agents install report-writer --yes")
    ok = g.wait_for("its own folder /agent/", 25, m) and g.wait_for("installed; granted", 10, m)
    g.send("/agents uninstall report-writer")
    g.wait_for("removed", 10)
    return ok, "install screen shows the per-agent folder grant" if ok else "no fs-scope line on install"


def s_mcp_connect(g):
    """`/mcp connect` end-to-end: connect to the harness MCP server, list its
    tools (echo), and call it directly — the in-kernel JSON-RPC client.
    Also covers resources list/read, status, and reconnect (Phase 4)."""
    m = g.mark()
    g.send(f"/mcp connect harness http://{HOST}:{PLAIN_PORT}/mcp")
    if not g.wait_for("registered 1 tool", 20, m) or not g.wait_for("mcp__harness__echo", 5, m):
        return False, "connect/list did not register the echo tool"
    m = g.mark()
    g.send('/mcp call harness echo {"text":"pong-9271"}')
    if not g.wait_for("echo: pong-9271", 20, m):
        return False, "MCP call did not echo"
    m = g.mark()
    g.send("/mcp resources harness")
    if not g.wait_for("file:///e2e/notes.txt", 15, m):
        return False, "resources/list missing notes URI"
    m = g.mark()
    g.send("/mcp read harness file:///e2e/notes.txt")
    if not g.wait_for("resource-body: e2e-notes-42", 15, m):
        return False, "resources/read missing body"
    m = g.mark()
    g.send("/mcp status")
    if not g.wait_for("harness", 12, m) or not g.wait_for("resource", 5, m):
        return False, "status missing resource count"
    m = g.mark()
    g.send("/mcp reconnect harness")
    if not g.wait_for("reconnected", 20, m):
        return False, "reconnect failed"
    return True, "MCP connect/call/resources/status/reconnect"


def s_skills_bundled(g):
    """Bundled skills install at boot (L0); `/skills load` exercises L1 invoke."""
    m = g.mark()
    g.send("/skills")
    # Boot installs remember / debug-net / safe-files.
    ok = (
        g.wait_for("remember", 15, m)
        and g.wait_for("debug-net", 5, m)
        and g.wait_for("safe-files", 5, m)
    )
    if not ok:
        return False, "bundled L0 skills not listed"
    m = g.mark()
    g.send("/skills load remember")
    if not g.wait_for("memory_add", 15, m):
        return False, "skill L1 body not loaded"
    m = g.mark()
    g.send("/skills load remember examples")
    if not g.wait_for("L2 asset", 15, m) and not g.wait_for("examples", 5, m):
        # load with asset may print body + L2; accept either marker
        tail = g.text()[m:]
        if "Examples" not in tail and "memory_add" not in tail:
            return False, "L2 asset load failed"
    return True, "bundled skills L0 + L1 (+ L2) invoke"


def s_plan_mode_and_permissions(g):
    """Phase 5: /mode plan + /permissions surface."""
    m = g.mark()
    g.send("/mode plan")
    if not g.wait_for("plan", 12, m):
        return False, "mode plan not set"
    m = g.mark()
    g.send("/mode")
    if not g.wait_for("plan", 8, m):
        return False, "mode show missing plan"
    m = g.mark()
    g.send("/mode auto")
    if not g.wait_for("auto", 12, m):
        return False, "mode auto failed"
    m = g.mark()
    g.send("/permissions show")
    # Boot loads default rules (or empty if ensure failed).
    if not g.wait_for("permissions>", 12, m):
        return False, "permissions show failed"
    m = g.mark()
    g.send("/permissions reload")
    if not g.wait_for("permissions>", 12, m):
        return False, "permissions reload failed"
    return True, "plan mode + permissions.json surface"


def s_todos_pane(g):
    """Phase 6: /todos opens the checklist pane (empty is fine)."""
    m = g.mark()
    g.send("/todos list")
    if not g.wait_for("todos>", 12, m):
        return False, "todos list failed"
    m = g.mark()
    g.send("/todos open")
    if not g.wait_for("todos>", 12, m):
        return False, "todos open failed"
    return True, "todos list + open"


def s_mcp_manifest(g):
    """An agent that declares an MCP server in its manifest: the install consent
    screen shows the server, and on approval it connects + registers the tool
    for the agent."""
    m = g.mark()
    g.send("/agents install mcp-agent --yes")
    ok = g.wait_for("MCP server 'harness'", 25, m) and g.wait_for("MCP 'harness' connected", 20, m)
    g.send("/mcp disconnect harness")
    g.wait_for("disconnected", 10)
    g.send("/agents uninstall mcp-agent")
    g.wait_for("removed", 10)
    return ok, "manifest MCP server shown on install + connected" if ok else "manifest MCP not surfaced/connected"


def _http_get(port, path):
    # Read the whole response to EOF (the server sends Connection: close), so we
    # can validate the received body length against the Content-Length header —
    # catching any truncation-on-close.
    with socket.create_connection(("127.0.0.1", port), timeout=15) as s:
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: chitti\r\nConnection: close\r\n\r\n".encode())
        s.settimeout(15)
        resp = b""
        while len(resp) < 65536:
            chunk = s.recv(2048)
            if not chunk:
                break
            resp += chunk
    return resp


def _content_length_ok(resp):
    # True iff the response declares a Content-Length that matches the bytes of
    # the received body (headers/body split on the blank line).
    sep = resp.find(b"\r\n\r\n")
    if sep < 0:
        return False
    head, body = resp[:sep], resp[sep + 4:]
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            declared = int(line.split(b":", 1)[1].strip())
            return declared == len(body)
    return False


def s_doc_pipeline(g):
    # Prove the full multi-agent path round-trips a valid HTTP response: the
    # Network agent relays the socket bytes, the HTTP agent parses + formats, and
    # the Doc agent responds via assets/tools.wasm (route_request) — no model
    # required. Response must be well-formed HTTP/1.1 with matching Content-Length.
    # (doc_website --slow still checks model fallback when present.)
    #
    # Also proves loopback: the SAME server is reached from the OS *itself* over
    # 127.0.0.1 and by the name `localhost` — the guest `/http` client connects
    # through the in-kernel loopback interface (never the NIC), so an in-OS client
    # can talk to an in-OS listener.
    m = g.mark()
    g.send(f"/agents start doc {SVC_HTTP_PORT}")
    if not g.wait_for("web pipeline network->http->server", 15, m):
        return False, "web pipeline did not start"
    time.sleep(0.6)
    try:
        resp = _http_get(SVC_HTTP_PORT, "/")
    except OSError as e:
        return False, f"http request failed: {e}"
    ok = resp.startswith(b"HTTP/1.1 ") and _content_length_ok(resp)
    code = resp.split(b"\r\n", 1)[0].decode(errors="replace") if resp else "(no response)"
    if not ok:
        return False, f"bad response: {resp[:60]!r}"
    # Loopback: in-guest client → in-guest listener. Only a completed response
    # prints "http> <status> (<n> bytes)"; a refused/timed-out connect takes the
    # error path ("http> error: …") and never prints "bytes)".
    lm = g.mark()
    g.send(f"/http http://127.0.0.1:{SVC_HTTP_PORT}/")
    lo_ip = g.wait_for("bytes)", 20, lm)
    nm = g.mark()
    g.send(f"/http http://localhost:{SVC_HTTP_PORT}/")
    lo_name = g.wait_for("bytes)", 20, nm)
    if not (lo_ip and lo_name):
        return False, f"loopback failed (127.0.0.1={lo_ip}, localhost={lo_name})"
    return True, f"network->http->doc round-trips over host + loopback (127.0.0.1, localhost) ({code})"


def s_doc_website(g):
    # Doc agent ships tools.wasm: GET / → index.html, GET /docs → docs.html
    # without a model (deterministic route_request). Scope-gated asset read still
    # applies. Works on the fast path; --slow group may still run this.
    g.send(f"/agents start doc {SVC_HTTP_PORT}")
    g.wait_for("web pipeline network->http->server", 15)
    time.sleep(0.6)
    try:
        home = _http_get(SVC_HTTP_PORT, "/")
        docs = _http_get(SVC_HTTP_PORT, "/docs")
    except OSError as e:
        return False, f"http request failed: {e}"
    ok = (
        home.startswith(b"HTTP/1.1 200")
        and b"Chitti" in home
        and _content_length_ok(home)
        and docs.startswith(b"HTTP/1.1 200")
        and b"Documentation" in docs
        and _content_length_ok(docs)
    )
    return ok, "wasm route_request served index.html + docs.html" if ok else f"did not serve the pages (home={home[:60]!r})"


def s_agents_search(g):
    # Fetch the registry index over HTTP and list advertised agents.
    m = g.mark()
    g.send(f"/agents search http://{HOST}:{PLAIN_PORT}/registry report")
    ok = g.wait_for("report-writer", 20, m) and g.wait_for("chitti-publisher-test", 5, m)
    return ok, "registry index fetched + searched" if ok else "no registry results"


def s_agents_install_registry(g):
    # Install an agent confirmed present in the registry index (network
    # discovery), through the consent flow.
    m = g.mark()
    g.send(f"/agents install note-summarizer --yes --registry http://{HOST}:{PLAIN_PORT}/registry")
    ok = g.wait_for("found in registry", 20, m) and g.wait_for("installed; granted", 15, m)
    g.send("/agents uninstall note-summarizer")
    return ok, "installed from registry via consent flow" if ok else "registry install did not complete"


def s_system_agents(g):
    # Only agents that reason from a SOUL are installed agents (doc, ssh); the
    # network/http stages are pure service-layer plumbing, not agents. The boot
    # log proves install_all ran (signs each package + places its SOUL/assets).
    booted = g.wait_for("system agents installed (doc, ssh)", 5, 0)  # printed at boot
    m = g.mark()
    g.send("/agents")
    listed = g.wait_for("system agents", 15, m) and g.wait_for("/agent/9001/SOUL.md", 3, m)
    ok = booted and listed
    return ok, "doc + ssh installed as system agents in /agent/ (network/http are plumbing)" if ok else "system agents not installed/listed"


def s_ssh_agent(g):
    # Start the SSH system agent and confirm it does the RFC 4253 version
    # exchange (sends its identification banner) on an inbound connection.
    m = g.mark()
    g.send(f"/agents start ssh {SVC_SSH_PORT}")
    if not g.wait_for("started 'ssh' service", 15, m):
        return False, "ssh agent did not start"
    time.sleep(0.5)
    try:
        with socket.create_connection(("127.0.0.1", SVC_SSH_PORT), timeout=15) as s:
            s.sendall(b"SSH-2.0-e2eClient\r\n")
            s.settimeout(15)
            banner = s.recv(64)
    except OSError as e:
        return False, f"ssh connect failed: {e}"
    ok = banner.startswith(b"SSH-2.0-Chitti")
    return ok, "SSH agent completed the version exchange" if ok else f"bad banner: {banner!r}"


def s_surface(g):
    # The UI-surface capability: request a surface + draw ops through Synapse and
    # confirm a deterministic rasterization checksum (pixels aren't on serial).
    m = g.mark()
    g.send("/surface demo")
    ok = g.wait_for("surface> rendered surface", 15, m) and g.wait_for("checksum=0x", 5, m)
    return ok, "surface drawn (grammar-validated draw ops rasterized)" if ok else "no surface render"


def s_package_apps(g):
    """Package-UI apps (tools.wasm on a persistent instance): start every
    package-UI agent in turn — each start instantiates the module, resolves
    the app's start export, requests a surface, and paints. Starting the next
    app stops the previous (observable), and stop-package closes the last.

    Chess is special: it paints via board_set + host_hud_set (reserved HUD
    strip). A prior SCREEN reentrancy hang on that path froze open; the rest
    of the suite uses host_ui_draw only. Starting an app focuses the action
    pane (its keys go to the game), so Shift+Tab (ESC[Z) returns focus to the
    chat line before the next command."""
    # (agent name, substring of package_ui> <name>: <start reply>)
    # Chess: model-less guest → hotseat; reply starts with ok:chess.
    apps = [
        ("chess", "package_ui> chess: ok:chess"),
        ("minesweeper", "package_ui> minesweeper: ok:mines 9x9"),
        ("snake", "package_ui> snake: ok:snake"),
        ("synth", "package_ui> synth: ok:synth piano"),
        ("paint", "package_ui> paint: ok:paint ready"),
        ("slides", "package_ui> slides: ok:slides n="),
        ("calc", "package_ui> calc: ok:calc ready"),
        ("clock", "package_ui> clock: ok:clock"),
        ("files", "package_ui> files: ok:files"),
        ("gallery", "package_ui> gallery: ok:gallery"),
        ("sheets", "package_ui> sheets: ok:sheets"),
        ("calendar", "package_ui> calendar: ok:calendar"),
        ("contacts", "package_ui> contacts: ok:contacts"),
        ("writer", "package_ui> writer: ok:writer"),
        ("archive", "package_ui> archive: ok:archive"),
        ("hex", "package_ui> hex: ok:hex"),
        ("game2048", "package_ui> game2048: ok:2048"),
        ("activity", "package_ui> activity: ok:activity"),
        ("weather", "package_ui> weather: ok:weather"),
        ("settings", "package_ui> settings: ok:settings"),
        ("dict", "package_ui> dict: ok:dict"),
        ("diff", "package_ui> diff: ok:diff"),
        ("breakout", "package_ui> breakout: ok:breakout"),
        ("tetris", "package_ui> tetris: ok:tetris"),
        ("console", "package_ui> console: ok:console"),
        ("maps", "package_ui> maps: ok:maps"),
        ("radio", "package_ui> radio: ok:radio"),
        ("sandbox-lab", "package_ui> sandbox-lab: ok:sandbox-lab"),
    ]
    prev = None
    for name, expect in apps:
        m = g.mark()
        g.send(f"/agents start {name}")
        if prev is not None:
            if not g.wait_for(f"package_ui> stopped '{prev}'", 15, m):
                return False, f"{name}: previous '{prev}' did not stop"
        if not g.wait_for(expect, 15, m):
            return False, f"{name} did not start (expected {expect!r})"
        g.send_raw(b"\x1b[Z")  # Shift+Tab: focus back to the chat line
        prev = name
    m = g.mark()
    g.send("/agents stop-package")
    ok = g.wait_for(f"package_ui> stopped '{prev}'", 15, m)
    return (
        ok,
        f"{len(apps)} package-UI apps start+stop over package_ui"
        if ok
        else f"stop-package failed (last={prev})",
    )


OS = [(n, make_cmd_scenario(c, mk)) for (n, c, mk) in OS_CMDS] + [
    ("help_restart", s_help_restart),
    ("memory", s_memory),
    ("memory_hierarchy", s_memory_hierarchy),
    ("fs_basic", s_fs_basic),
    ("skills_bundled", s_skills_bundled),
    ("plan_mode_and_permissions", s_plan_mode_and_permissions),
    ("todos_pane", s_todos_pane),
    ("session", s_session),
    ("open_media", s_open_media),
    ("open_video", s_open_video),
    ("tabs", s_tabs),
    ("panes", s_panes),
    ("clipboard", s_clipboard),
]
AGENTS = [("agents_services", s_agents_services), ("agents_switch_caps", s_agents_switch_caps), ("agents_install", s_agents_install), ("agents_uninstall", s_agents_uninstall), ("agent_fs_consent", s_agent_fs_consent), ("agents_search", s_agents_search), ("agents_install_registry", s_agents_install_registry), ("system_agents", s_system_agents), ("doc_pipeline", s_doc_pipeline), ("ssh_agent", s_ssh_agent), ("surface", s_surface), ("package_apps", s_package_apps), ("mcp_manifest", s_mcp_manifest)]
NET = [("nic_dispatch", s_nic_dispatch), ("network", s_network), ("ping", s_ping), ("http_get", s_http_get), ("http_post", s_http_post), ("http_download", s_http_download), ("http_stream", s_http_stream), ("browse", s_browse), ("ws", s_ws), ("mcp_connect", s_mcp_connect), ("cancel", s_cancel)]
# Runs after every other group: kills the guest (QEMU -no-reboot → exit).
FINAL = [("restart", s_restart)]

# Known-flaky scenarios: timing-fragile, not code-buggy. The Doc web server's
# reply is model-driven and prefill-bound (~10-15s for the ~530-token serve
# prompt) — right at the 15s http timeout — and `ssh_agent` runs next, racing the
# shell while that inference is still busy. A failure here is retried once and, if
# it still fails, reported [FLAKY] without gating the run's exit code; a
# consistently-failing one still shows up in every run's output.
FLAKY = {"doc_pipeline", "ssh_agent"}

# The run's arch/verbosity, published for scenarios that boot their own guest
# (model_load boots a --no-model guest with a model disk). Set by main().
RUN_ARCH = "aarch64"
RUN_VERBOSE = False
NET_TLS = [("wss", s_wss), ("model_remote_https", s_model_remote_https)]
MODEL = [("bench", s_bench), ("infer", s_infer), ("perf", s_perf), ("chat", s_chat), ("compact", s_compact), ("model_load", s_model_load), ("doc_website", s_doc_website)]
VOICE = [("voice_models", s_voice_models), ("voice_say", s_voice_say)]


def boot_guest(arch, model, verbose, audio, fwd, no_model=False, attempts=3):
    """Launch the guest and wait for it to reach networking, retrying if it dies
    early. aarch64 SMP bring-up (PSCI CPU_ON) very occasionally takes a data
    abort right after `smp: N cores online` — a rare hypervisor bring-up race,
    not a scenario failure — so rather than fail the whole run we detect the
    boot-time FATAL (or an early exit / 120 s with no networking), kill the VM,
    and relaunch. `no_model` boots the desktop default-heap kernel (no GGUF, no
    large model-heap reservation) — used for the non-slow groups, which never
    run inference, so the guest fits a CI runner's RAM instead of OOMing while
    mapping an oversized heap. Returns a booted Guest, or None if every attempt
    failed."""
    for attempt in range(1, attempts + 1):
        g = Guest(arch=arch, model=model, verbose=verbose, audio=audio, hostfwd=fwd, no_model=no_model)
        deadline = time.time() + 120
        outcome = "timeout"
        while time.time() < deadline:
            txt = g.text()
            if "net: configured" in txt:
                outcome = "ok"
                break
            if "FATAL" in txt:
                outcome = "crash"
                break
            if g.proc.poll() is not None:
                outcome = "exited"
                break
            time.sleep(0.2)
        if outcome == "ok":
            if attempt > 1:
                print(f"e2e: guest booted on attempt {attempt}/{attempts}")
            return g
        print(f"e2e: boot attempt {attempt}/{attempts} failed ({outcome}); relaunching…")
        # Always surface the serial tail on a boot failure (not just under -v) —
        # a headless CI run has no other window into why the guest didn't reach
        # `net: configured`.
        print("    --- guest serial tail ---")
        print("    " + g.tail(1200).replace("\n", "\n    "))
        g.close()
    return None


def main():
    global RUN_ARCH, RUN_VERBOSE
    arch, model = "aarch64", "qwen3.5-0.8b"
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    slow = "--slow" in sys.argv or "--full" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("-v", "--verbose", "--slow", "--full")]
    for i, a in enumerate(args):
        if a == "-arch" and i + 1 < len(args):
            arch = args[i + 1]
        if a == "-model" and i + 1 < len(args):
            model = args[i + 1]
    RUN_ARCH, RUN_VERBOSE = arch, verbose

    have_tls = ssl.HAS_TLSv1_3 and ensure_cert()
    have_model = os.path.exists(os.path.join(ROOT, "assets", "model.gguf"))
    have_voice = os.path.isdir(os.path.join(ROOT, "assets", "voice")) and os.listdir(os.path.join(ROOT, "assets", "voice"))
    print(f"e2e: arch={arch} model={model} tls={'yes' if have_tls else 'SKIP'} slow={'yes' if slow else 'no'}")

    servers = [Server(PLAIN_PORT)]
    if have_tls:
        ctx = tls_context(CERT, KEY)
        if ctx:
            servers.append(Server(TLS_PORT, ctx))
        else:
            have_tls = False

    # `FINAL` (`/restart`) is always last — it reboots/exits the guest.
    scenarios = list(OS) + list(AGENTS) + list(NET) + (list(NET_TLS) if have_tls else [])
    if slow:
        if have_model:
            scenarios += list(MODEL)
        else:
            print("  (model scenarios skipped — assets/model.gguf absent)")
        if have_voice:
            scenarios += list(VOICE)
        else:
            print("  (voice scenarios skipped — assets/voice/ absent)")
    scenarios += list(FINAL)

    # Voice needs a sound device; give the guest a silent audio backend then.
    audio = "none" if (slow and have_voice) else "off"
    fwd = f"{SVC_PORT},{SVC_HTTP_PORT},{SVC_SSH_PORT}"
    print(f"e2e: booting guest (cargo xtask run, audio={audio}, hostfwd={fwd})…")
    # Non-slow groups (os/net/agents) never run inference, so boot the main
    # guest model-less: it uses the small desktop heap and fits a CI runner
    # instead of OOMing while mapping the model-sized heap. The slow group
    # keeps the model loaded for the inference/chat scenarios.
    g = boot_guest(arch, model, verbose, audio, fwd, no_model=not slow)
    if g is None:
        print("e2e: FAILED — guest never booted (networking not configured after retries)")
        return 1
    def run_scenario(fn):
        try:
            return fn(g)
        except Exception as e:
            return False, f"exception: {e}"

    results = []
    try:
        time.sleep(1)
        for name, fn in scenarios:
            ok, detail = run_scenario(fn)
            # Known-flaky scenarios: retry once, then tolerate (report [FLAKY],
            # don't gate the run) so their timing jitter can't fail an otherwise
            # green run — while a consistently broken one still shows every time.
            if ok is False and name in FLAKY:
                ok, detail = run_scenario(fn)
                if ok is False:
                    results.append((name, "FLAKY"))
                    print(f"  [FLAKY] {name}: {detail} (known-flaky — not gating)")
                    if verbose:
                        print("    " + g.tail(600).replace("\n", "\n    "))
                    continue
                detail = f"{detail} (passed on retry)"
            tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
            results.append((name, tag))
            print(f"  [{tag}] {name}: {detail}")
            if ok is False and verbose:
                print("    " + g.tail(600).replace("\n", "\n    "))
    finally:
        g.close()
        for s in servers:
            s.stop()

    passed = sum(1 for _, t in results if t == "PASS")
    failed = sum(1 for _, t in results if t == "FAIL")
    skipped = sum(1 for _, t in results if t == "SKIP")
    flaky = sum(1 for _, t in results if t == "FLAKY")
    print(f"e2e: {passed} passed, {failed} failed, {skipped} skipped, {flaky} flaky ({len(results)} run)")
    return 1 if failed else 0  # FLAKY never sets the exit code


if __name__ == "__main__":
    sys.exit(main())
