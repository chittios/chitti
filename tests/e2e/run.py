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
    # the Doc agent responds. WITHOUT a model the Doc agent can't plan a route and
    # returns a well-formed 503 (still exercising all three agents); WITH a model
    # it serves the page (see the --slow doc_website scenario). Either way the
    # response must be a well-formed HTTP/1.1 message with a matching Content-Length.
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
    # --slow (needs the model): with the model loaded, the Doc agent's model
    # PLANS the route from its SOUL — GET / → it chooses index.html, GET /docs →
    # docs.html — and the read runs through the scope-gated file tool call.
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
    return ok, "model-planned routing served index.html + docs.html" if ok else f"model did not serve the pages (home={home[:60]!r})"


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


OS = [(n, make_cmd_scenario(c, mk)) for (n, c, mk) in OS_CMDS]
AGENTS = [("agents_services", s_agents_services), ("agents_install", s_agents_install), ("agents_uninstall", s_agents_uninstall), ("agents_search", s_agents_search), ("agents_install_registry", s_agents_install_registry), ("system_agents", s_system_agents), ("doc_pipeline", s_doc_pipeline), ("ssh_agent", s_ssh_agent), ("surface", s_surface)]
NET = [("network", s_network), ("ping", s_ping), ("http_get", s_http_get), ("http_post", s_http_post), ("http_stream", s_http_stream), ("ws", s_ws)]
NET_TLS = [("wss", s_wss), ("model_remote_https", s_model_remote_https)]
MODEL = [("bench", s_bench), ("infer", s_infer), ("perf", s_perf), ("chat", s_chat), ("compact", s_compact), ("doc_website", s_doc_website)]
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
    fwd = f"{SVC_PORT},{SVC_HTTP_PORT},{SVC_SSH_PORT}"
    print(f"e2e: booting guest (cargo xtask run, audio={audio}, hostfwd={fwd})…")
    g = Guest(arch=arch, model=model, verbose=verbose, audio=audio, hostfwd=fwd)
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
