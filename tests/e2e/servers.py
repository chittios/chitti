"""Host-side test servers for the ChittiOS end-to-end tests.

One raw-socket protocol handler serves everything the guest exercises — a few
HTTP routes, chunked SSE streaming, a WebSocket echo, and an OpenAI-compatible
chat endpoint — and it is run twice: once plaintext (for http:// / ws://) and
once wrapped in TLS 1.3 (for https:// / wss://). The guest reaches these at
10.0.2.2 (QEMU user-net maps that to the host).
"""

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# The git smart-HTTP fixture: a two-commit repository packed by real `git
# pack-objects`, whose history git chose to **delta-compress**. Shared verbatim
# with the git agent's own host tests (tools/git-wasm/tests/) so the two cannot
# drift — and deltas are the case that matters, since a server deltifies any
# repository past a couple of commits and resolving one wrong is what made every
# real `/git clone` fail.
GIT_PACK = os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "git-wasm", "tests", "ref-delta.pack"
)
GIT_HEAD = "35528a725a8925975d241e31a82755548dced31f"


# Where the `ssh_client` scenario puts the keypair it generates for the real
# `sshd` it starts. Served over plain HTTP so the guest can fetch its own
# identity with `/http -O` — the same path `http_download` already proves. It is
# a throwaway key for a loopback sshd, generated per run and deleted after.
SSH_DIR = os.path.join(tempfile.gettempdir(), "chitti-e2e-ssh")


# Where the `open_hls` scenario writes the playlist + MPEG-TS segments it
# serves. HLS is a *network* format — the player fetches a playlist and then
# every segment it names — so the only honest test of it goes over HTTP rather
# than off a mounted disk.
HLS_DIR = os.path.join(tempfile.gettempdir(), "chitti-e2e-hls")


def _pktline(payload: bytes) -> bytes:
    return b"%04x%s" % (4 + len(payload), payload)


def _git_advertisement() -> bytes:
    """`GET /info/refs?service=git-upload-pack` — the v0 ref advertisement."""
    caps = b"multi_ack ofs-delta no-progress"
    sha = GIT_HEAD.encode()
    out = _pktline(b"# service=git-upload-pack\n") + b"0000"
    out += _pktline(b"%s HEAD\x00%s\n" % (sha, caps))
    out += _pktline(b"%s refs/heads/main\n" % sha)
    return out + b"0000"


def _openssl():
    for c in ("/opt/homebrew/opt/openssl@3/bin/openssl", "/usr/local/opt/openssl@3/bin/openssl", "openssl"):
        try:
            out = subprocess.run([c, "version"], capture_output=True, text=True)
            if out.returncode == 0 and "OpenSSL" in out.stdout:
                return c
        except Exception:
            continue
    return "openssl"


REGISTRY_ENTRIES = [
    {"name": "report-writer", "version": "1.0.0",
     "description": "Write reports from facts",
     "download": "http://10.0.2.2:8100/pkg/report-writer",
     "key_id": "chitti-publisher-test"},
    {"name": "note-summarizer", "version": "1.0.0",
     "description": "Summarize and search note files",
     "download": "http://10.0.2.2:8100/pkg/note-summarizer",
     "key_id": "chitti-publisher-test"},
]


def _registry_index():
    """The agent-registry index, **signed** with the e2e publisher key.

    The kernel refuses unsigned indexes (`registry_client::parse_index`), so
    this is signed exactly the way the kernel verifies: the message is one
    `name\\0version\\0download\\0key_id\\n` line per entry (description is
    display-only), hashed with SHA-256, signed with ECDSA/P-256, and shipped as
    base64 DER under the root `sig` field. The private key is the test
    publisher's (`chitti-publisher-test`, whose public point is baked into
    `kernel/src/skills/crypto.rs`); it lives under `tests/e2e/certs/` and is
    gitignored — it is a test key, never a real one."""
    doc = {"schema": 1, "entries": REGISTRY_ENTRIES}
    msg = b"".join(
        ("%s\0%s\0%s\0%s\n" % (e["name"], e["version"], e["download"], e["key_id"])).encode()
        for e in REGISTRY_ENTRIES
    )
    key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs", "registry-key.pem")
    if not os.path.exists(key):
        raise RuntimeError(f"missing registry signing key {key} — regenerate with "
                           "openssl ecparam -name prime256v1 -genkey and rebake the public key")
    p = subprocess.run([_openssl(), "dgst", "-sha256", "-sign", key], input=msg, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"openssl sign failed: {p.stderr.decode(errors='replace')}")
    doc["key_id"] = "chitti-publisher-test"
    doc["sig"] = base64.b64encode(p.stdout).decode()
    return json.dumps(doc).encode()


def _read_headers(conn):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            return None, b""
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode(errors="replace").split("\r\n")
    method, path = (lines[0].split(" ") + ["", ""])[:2]
    hdrs = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    return (method, path, hdrs), rest


def _ws_handshake(conn, hdrs):
    key = hdrs.get("sec-websocket-key", "")
    accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
    conn.sendall(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode()
    )


def _ws_read_frame(conn, rest):
    buf = bytearray(rest)
    while len(buf) < 2:
        buf += conn.recv(4096)
    ln = buf[1] & 0x7F
    off = 2
    if ln == 126:
        while len(buf) < 4:
            buf += conn.recv(4096)
        ln = struct.unpack(">H", buf[2:4])[0]
        off = 4
    while len(buf) < off + 4:
        buf += conn.recv(4096)
    mask = buf[off : off + 4]
    off += 4
    while len(buf) < off + ln:
        buf += conn.recv(4096)
    return bytes(buf[off + i] ^ mask[i & 3] for i in range(ln))


def _ws_send_text(conn, msg: bytes):
    conn.sendall(b"\x81" + bytes([len(msg)]) + msg)


def _handle(conn):
    try:
        req, rest = _read_headers(conn)
        if req is None:
            return
        method, path, hdrs = req
        # WebSocket upgrade → echo one frame back with an "echo:" prefix.
        if hdrs.get("upgrade", "").lower() == "websocket":
            _ws_handshake(conn, hdrs)
            payload = _ws_read_frame(conn, rest)
            _ws_send_text(conn, b"echo:" + payload)
            time.sleep(0.3)
            return
        if path == "/id_ed25519":
            # The guest fetches its own SSH identity here.
            try:
                with open(os.path.join(SSH_DIR, "client"), "rb") as f:
                    body = f.read()
            except OSError:
                conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                return
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
        elif path.startswith("/repo.git/info/refs"):
            body = _git_advertisement()
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/x-git-upload-pack-advertisement\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
        elif path.startswith("/repo.git/git-upload-pack"):
            # Drain the client's want/done request before replying — a server that
            # answers without reading leaves the body in the socket, and the guest's
            # next request reads it as a response.
            n = int(hdrs.get("content-length", "0"))
            raw = rest
            while len(raw) < n:
                raw += conn.recv(4096)
            with open(GIT_PACK, "rb") as f:
                body = b"0008NAK\n" + f.read()
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/x-git-upload-pack-result\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
        elif path.startswith("/v1/models"):
            body = b'{"object":"list","data":[{"id":"e2e-model"}]}'
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        elif path.startswith("/v1/chat/completions"):
            n = int(hdrs.get("content-length", "0"))
            raw = rest
            while len(raw) < n:
                raw += conn.recv(4096)
            try:
                last = json.loads(raw or b"{}").get("messages", [{}])[-1].get("content", "")
            except Exception:
                last = ""
            out = json.dumps({"choices": [{"message": {"role": "assistant", "content": f"remote reply to: {last[:40]}"}}]}).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(out), out))
        elif path == "/json":
            body = b'{"ok":true,"who":"e2e"}'
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        elif path == "/page.html":
            # Minimal HTML for the browser agent (`/browse`) — CSS + JS subset.
            body = (
                b"<!DOCTYPE html><html><head><title>E2E Browser</title>"
                b"<style>body{background:#f5f0e8}h1{color:#cc785c}</style>"
                b"<script>document.title='E2E Browser';console.log('e2e-js');</script>"
                b"</head><body><h1>Hello Chitti</h1>"
                b"<p>Who: e2e-browser</p>"
                b'<p><a href="/json">JSON link</a></p>'
                b"</body></html>"
            )
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
        elif path == "/runaway.html":
            # A page whose script never returns. The in-kernel JS engine is a
            # tree-walker with no yield points of its own, so this used to run
            # the shell thread until the machine was rebooted; the browser now
            # bounds it (and Ctrl+C reaches it) and renders the page without it.
            body = (
                b"<!DOCTYPE html><html><head><title>Runaway</title></head>"
                b"<body><h1>Runaway script</h1>"
                b"<p id=\"m\">static content</p>"
                b"<script>var n=0; while(true){ n=n+1; }</script>"
                b"</body></html>"
            )
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
        elif path == "/mcp":
            # A minimal MCP server (JSON-RPC 2.0 over Streamable HTTP): supports
            # initialize / notifications/initialized / tools/list / tools/call,
            # exposing one "echo" tool. Enough to prove the in-kernel client.
            n = int(hdrs.get("content-length", "0"))
            raw = rest
            while len(raw) < n:
                raw += conn.recv(4096)
            try:
                req_obj = json.loads(raw or b"{}")
            except Exception:
                req_obj = {}
            rpc_id = req_obj.get("id")
            rpc_method = req_obj.get("method", "")
            if rpc_id is None:
                # A notification (e.g. notifications/initialized): 202, no body.
                conn.sendall(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
            else:
                if rpc_method == "initialize":
                    result = {"protocolVersion": "2025-06-18",
                              "capabilities": {"tools": {}, "resources": {}},
                              "serverInfo": {"name": "e2e-mcp", "version": "1.0"}}
                elif rpc_method == "tools/list":
                    result = {"tools": [{"name": "echo",
                                         "description": "Echo the given text back",
                                         "inputSchema": {"type": "object",
                                                         "properties": {"text": {"type": "string"}},
                                                         "required": ["text"]}}]}
                elif rpc_method == "tools/call":
                    args = (req_obj.get("params") or {}).get("arguments") or {}
                    text = args.get("text", args.get("input", ""))
                    result = {"content": [{"type": "text", "text": "echo: " + str(text)}]}
                elif rpc_method == "resources/list":
                    result = {"resources": [{
                        "uri": "file:///e2e/notes.txt",
                        "name": "notes",
                        "description": "e2e demo resource",
                        "mimeType": "text/plain",
                    }]}
                elif rpc_method == "resources/read":
                    uri = (req_obj.get("params") or {}).get("uri", "")
                    result = {"contents": [{
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": "resource-body: e2e-notes-42",
                    }]}
                else:
                    result = {}
                out = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}).encode()
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: e2e-sess\r\nContent-Length: %d\r\n\r\n%s" % (len(out), out))
        elif path == "/logo.png":
            # A real 3x2 PNG for the `/http -O` download → `/open` roundtrip.
            import struct as _s
            import zlib as _z

            def _chunk(t, d):
                c = _s.pack(">I", len(d)) + t + d
                return c + _s.pack(">I", _z.crc32(t + d) & 0xFFFFFFFF)

            raw = (b"\x00" + bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
                   + b"\x00" + bytes([10, 20, 30, 40, 50, 60, 250, 249, 245]))
            body = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", _s.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
                    + _chunk(b"IDAT", _z.compress(raw, 9)) + _chunk(b"IEND", b""))
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        elif path.startswith("/hls/"):
            # The HLS fixture: a playlist and its MPEG-TS segments. The name is
            # whitelisted to a bare filename so a path in the request cannot
            # reach outside the fixture directory.
            name = path[len("/hls/"):].split("?")[0]
            body = None
            if name and "/" not in name and ".." not in name:
                try:
                    with open(os.path.join(HLS_DIR, name), "rb") as f:
                        body = f.read()
                except OSError:
                    body = None
            if body is None:
                conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            else:
                ctype = (b"application/vnd.apple.mpegurl" if name.endswith(".m3u8")
                         else b"video/mp2t")
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %d\r\n\r\n%s"
                    % (ctype, len(body), body)
                )
        elif path == "/registry":
            # A public agent-registry index (discovery over the network),
            # **signed** so the kernel's index verification accepts it.
            body = _registry_index()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        elif path == "/hang":
            # Accept the request, complete the (TLS) handshake, then send NOTHING.
            # This is the state a real hosted endpoint leaves us in while it is
            # thinking, and it is where Ctrl+C used to be misreported: the cancel
            # was detected inside the TLS read, embedded-tls dropped the reason,
            # and the HTTP layer called it "no response head (connection closed
            # early)" — a network fault the user never had.
            time.sleep(30)
        elif path == "/sse":
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
            for i in range(3):
                d = ("data: event %d\n\n" % i).encode()
                conn.sendall(b"%X\r\n%s\r\n" % (len(d), d))
                time.sleep(0.2)
            conn.sendall(b"0\r\n\r\n")
        elif method == "POST":
            n = int(hdrs.get("content-length", "0"))
            body = rest
            while len(body) < n:
                body += conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        else:
            conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class Server:
    """A background accept loop; TLS-wraps each connection when `ctx` is set."""

    def __init__(self, port, ctx=None):
        self.port = port
        self.ctx = ctx
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.listen(16)
        self.sock.settimeout(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                raw, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if self.ctx is not None:
                try:
                    raw = self.ctx.wrap_socket(raw, server_side=True)
                except Exception:
                    try:
                        raw.close()
                    except Exception:
                        pass
                    continue
            threading.Thread(target=_handle, args=(raw,), daemon=True).start()

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


def tls_context(cert, key):
    """A TLS 1.3 server context matching what embedded-tls negotiates
    (AES-128-GCM-SHA256 over P-256, ECDSA cert). None if TLS 1.3 is unavailable."""
    if not ssl.HAS_TLSv1_3:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    # Each in its own try: `set_ciphers` with a TLS-1.3 suite name raises on
    # some builds, and it must NOT prevent `set_ecdh_curve` from restricting the
    # group to P-256 — embedded-tls only offers secp256r1, and leaving the
    # server's default group preference (X25519-first) in place makes its
    # handshake response unparseable to the client (DecodeError).
    try:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    except Exception:
        pass
    try:
        ctx.set_ciphers("TLS_AES_128_GCM_SHA256")
    except Exception:
        pass
    try:
        ctx.set_ecdh_curve("prime256v1")
    except Exception:
        pass
    return ctx
