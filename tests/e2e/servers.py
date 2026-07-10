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
import socket
import ssl
import struct
import threading
import time

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


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
        if path.startswith("/v1/models"):
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
        elif path == "/registry":
            # A public agent-registry index (discovery over the network).
            body = json.dumps({
                "schema": 1,
                "entries": [
                    {"name": "report-writer", "version": "1.0.0",
                     "description": "Write reports from facts",
                     "download": "http://10.0.2.2:8100/pkg/report-writer",
                     "key_id": "chitti-publisher-test"},
                    {"name": "note-summarizer", "version": "1.0.0",
                     "description": "Summarize and search note files",
                     "download": "http://10.0.2.2:8100/pkg/note-summarizer",
                     "key_id": "chitti-publisher-test"},
                ],
            }).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
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
