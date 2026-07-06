#!/usr/bin/env python3
"""End-to-end tests for Chitti OS: boot the kernel under QEMU, drive its shell
over the serial console, and check that every command / flow actually works on
the real thing.

Groups (each scenario -> PASS/FAIL, non-zero exit on any failure):
  os     — the shell/OS commands (help, info, datetime, disks, agents, top, …)
  net    — the networked flows against local host servers: /http (GET/POST/
           stream), /ws + /wss, /ping, and /model remote over https
  model  — inference: /bench, /infer, /perf, a chat turn, /compact  (needs the
           bundled model; slow — only with --slow)
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
SVC_PORT = 7099  # guest TCP listener, reachable from the host via slirp hostfwd
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
    ("help", "/help", "Chitti commands:"),
    ("info", "/info", "RAM installed"),
    ("datetime", "/datetime", "datetime>"),
    ("datetime_tz", "/datetime tz +5:30", "UTC+05:30"),
    ("disks", "/disks", "disks>"),
    ("lspci", "/lspci", "pci>"),
    ("mounts", "/mounts", "mounts>"),
    ("ls", "/ls /", "ls>"),
    ("skills", "/skills", "installed"),
    ("shortcuts", "/shortcuts", "Shortcuts ("),
    ("mode", "/mode", "mode>"),
    ("think", "/think off", "think>"),
    ("agents", "/agents", "agents>"),
    ("ui", "/ui", "ui>"),
    ("ktrace", "/ktrace", "ktrace>"),
    ("close", "/close", "action pane closed"),
    ("top", "/top", "top>"),
    ("clear", "/clear", "cleared"),
    ("wifi", "/wifi info", "wifi>"),
]


def make_cmd_scenario(cmd, marker, timeout=15):
    def fn(g):
        m = g.mark()
        g.send(cmd)
        ok = g.wait_for(marker, timeout, m)
        return ok, f"{cmd!r} -> {marker!r}" if ok else f"{cmd!r}: no {marker!r}"
    return fn


# --- network scenarios ------------------------------------------------------

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


def s_http_post(g):
    m = g.mark()
    g.send(f'/http -X POST -H "X-Test: yes" -d payload-9182 http://{HOST}:{PLAIN_PORT}/echo')
    ok = g.wait_for("payload-9182", 20, m)
    return ok, "POST body echoed" if ok else "body not echoed"


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


def s_net_service_echo(g):
    # Start the Network service agent listening on SVC_PORT, then connect to it
    # from the host (via the slirp hostfwd) and confirm it echoes bytes back.
    # Proves the full inbound path: net_listen -> try_accept -> channel::adopt_tcp
    # -> channel read/write over a live TCP connection.
    m = g.mark()
    g.send(f"/agents start-net {SVC_PORT}")
    if not g.wait_for("network-echo service listening", 15, m):
        return False, "service did not start"
    time.sleep(0.5)
    payload = b"svc-echo-ping-4242"
    try:
        with socket.create_connection(("127.0.0.1", SVC_PORT), timeout=15) as s:
            s.sendall(payload)
            s.settimeout(15)
            got = b""
            while len(got) < len(payload):
                chunk = s.recv(64)
                if not chunk:
                    break
                got += chunk
    except OSError as e:
        return False, f"connect/echo failed: {e}"
    ok = got == payload
    return ok, "inbound connection echoed via Tcp-backed channel" if ok else f"bad echo: {got!r}"


def s_surface(g):
    # The UI-surface capability: request a surface + draw ops through Synapse and
    # confirm a deterministic rasterization checksum (pixels aren't on serial).
    m = g.mark()
    g.send("/surface demo")
    ok = g.wait_for("surface> rendered surface", 15, m) and g.wait_for("checksum=0x", 5, m)
    return ok, "surface drawn (grammar-validated draw ops rasterized)" if ok else "no surface render"


OS = [(n, make_cmd_scenario(c, mk)) for (n, c, mk) in OS_CMDS]
AGENTS = [("agents_services", s_agents_services), ("agents_install", s_agents_install), ("agents_uninstall", s_agents_uninstall), ("net_service_echo", s_net_service_echo), ("surface", s_surface)]
NET = [("network", s_network), ("ping", s_ping), ("http_get", s_http_get), ("http_post", s_http_post), ("http_stream", s_http_stream), ("ws", s_ws)]
NET_TLS = [("wss", s_wss), ("model_remote_https", s_model_remote_https)]
MODEL = [("bench", s_bench), ("infer", s_infer), ("perf", s_perf), ("chat", s_chat), ("compact", s_compact)]
VOICE = [("voice_models", s_voice_models), ("voice_say", s_voice_say)]


def main():
    arch, model = "aarch64", "qwen3.5-0.8b"
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    slow = "--slow" in sys.argv or "--full" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("-v", "--verbose", "--slow", "--full")]
    for i, a in enumerate(args):
        if a == "-arch" and i + 1 < len(args):
            arch = args[i + 1]
        if a == "-model" and i + 1 < len(args):
            model = args[i + 1]

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

    # Voice needs a sound device; give the guest a silent audio backend then.
    audio = "none" if (slow and have_voice) else "off"
    print(f"e2e: booting guest (cargo xtask run, audio={audio}, hostfwd={SVC_PORT})…")
    g = Guest(arch=arch, model=model, verbose=verbose, audio=audio, hostfwd=SVC_PORT)
    results = []
    try:
        if not g.wait_for("net: configured", 120):
            print("e2e: FAILED — guest never configured networking (boot/DHCP)")
            print(g.tail(1500))
            return 1
        time.sleep(1)
        for name, fn in scenarios:
            try:
                ok, detail = fn(g)
            except Exception as e:
                ok, detail = False, f"exception: {e}"
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
    print(f"e2e: {passed} passed, {failed} failed, {skipped} skipped ({len(results)} run)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
