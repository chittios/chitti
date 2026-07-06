#!/usr/bin/env python3
"""End-to-end tests for Chitti OS: boot the kernel under QEMU, drive its shell
over the serial console, and exercise the networked core flows against local
host servers (HTTP methods + streaming, WebSocket ws:// and wss://, and a
hosted-model chat via /model remote over https).

Dependency-free (stdlib only). Run it with a TLS-1.3-capable Python (e.g.
Homebrew's) so the https/wss scenarios aren't skipped:

    /opt/homebrew/bin/python3 tests/e2e/run.py           # or: make e2e

Exits non-zero if any scenario fails. TLS scenarios auto-skip (not fail) when
the running Python lacks TLS 1.3.
"""

import os
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
HERE = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(HERE, "certs", "ec.pem")
KEY = os.path.join(HERE, "certs", "ec.key")


def _openssl():
    """A modern OpenSSL (3.x) if present — LibreSSL (macOS system openssl)
    produces certs the embedded-tls handshake rejects, so prefer Homebrew's."""
    for c in ("/opt/homebrew/opt/openssl@3/bin/openssl", "/usr/local/opt/openssl@3/bin/openssl", "openssl"):
        try:
            out = subprocess.run([c, "version"], capture_output=True, text=True)
            if out.returncode == 0 and "OpenSSL" in out.stdout:  # not "LibreSSL"
                return c
        except Exception:
            continue
    return "openssl"  # last resort


def ensure_cert():
    """Generate an ECDSA P-256 self-signed cert (what embedded-tls accepts)."""
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


# --- scenarios: each takes the guest, returns (ok, detail) ------------------

def s_network(g):
    m = g.mark()
    g.send("/network")
    ok = g.wait_for("10.0.2.15", 15, m)
    return ok, "IPv4 configured" if ok else "no IP in /network output"


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
    # Wait for the /ws loop to exit (it consumes input for its Ctrl+C check,
    # so the next command must not be sent until the shell prompt is back).
    g.wait_for("closed by peer", 5, m)
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
    # Switch back so a stray later turn doesn't hit the network.
    g.send("/model local")
    return ok, "hosted-model chat over https" if ok else "no remote reply"


PLAIN = [("network", s_network), ("http_get", s_http_get), ("http_post", s_http_post), ("http_stream", s_http_stream), ("ws", s_ws)]
TLS = [("wss", s_wss), ("model_remote_https", s_model_remote_https)]


def main():
    arch = "aarch64"
    model = "qwen3.5-0.8b"
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("-v", "--verbose")]
    for i, a in enumerate(args):
        if a == "-arch" and i + 1 < len(args):
            arch = args[i + 1]
        if a == "-model" and i + 1 < len(args):
            model = args[i + 1]

    have_tls = ssl.HAS_TLSv1_3 and ensure_cert()
    print(f"e2e: arch={arch} model={model} tls={'yes' if have_tls else 'SKIP (need TLS 1.3 python)'}")

    servers = [Server(PLAIN_PORT)]
    if have_tls:
        ctx = tls_context(CERT, KEY)
        if ctx:
            servers.append(Server(TLS_PORT, ctx))
        else:
            have_tls = False

    scenarios = list(PLAIN) + (list(TLS) if have_tls else [])
    results = []
    print("e2e: booting guest (cargo xtask run)…")
    g = Guest(arch=arch, model=model, verbose=verbose)
    try:
        # Wait for networking to come up before driving net commands.
        if not g.wait_for("net: configured", 120):
            print("e2e: FAILED — guest never configured networking (boot/DHCP)")
            print("---- last output ----")
            print(g.tail(1500))
            g.close()
            for s in servers:
                s.stop()
            return 1
        time.sleep(1)
        for name, fn in scenarios:
            try:
                ok, detail = fn(g)
            except Exception as e:
                ok, detail = False, f"exception: {e}"
            results.append((name, ok, detail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            if not ok and verbose:
                print("    ---- recent output ----")
                print("    " + g.tail(600).replace("\n", "\n    "))
    finally:
        g.close()
        for s in servers:
            s.stop()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    skipped = len(PLAIN) + len(TLS) - total
    print(f"e2e: {passed}/{total} passed" + (f", {skipped} skipped" if skipped else ""))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
