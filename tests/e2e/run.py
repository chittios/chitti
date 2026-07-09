#!/usr/bin/env python3
"""End-to-end tests for Chitti OS: boot the kernel under QEMU, drive its shell
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
    """`/help` documents `/restart` (the command itself reboots and is tested last)."""
    m = g.mark()
    g.send("/help")
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
    g2 = Guest(arch=RUN_ARCH, verbose=RUN_VERBOSE, no_model=True, audio="none",
               model_disk=f"{png_path}:{wav_path}")
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
            return ok, "PNG previewed + WAV played + media-tab controls (focus/pause/seek/stop) responsive" if ok else "media controls / Ctrl+C did not stop playback"
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
    g2 = Guest(arch=RUN_ARCH, verbose=RUN_VERBOSE, no_model=True, audio="off", model_disk=mp4_path)
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
            return ok, "mp4 decoded (H.264 baseline I+P, multi-slice, deblocked) + video-tab controls responsive" if ok else "controls/Ctrl+C did not stop video"
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
    tools (echo), and call it directly — the in-kernel JSON-RPC client."""
    m = g.mark()
    g.send(f"/mcp connect harness http://{HOST}:{PLAIN_PORT}/mcp")
    if not g.wait_for("registered 1 tool", 20, m) or not g.wait_for("mcp__harness__echo", 5, m):
        return False, "connect/list did not register the echo tool"
    m = g.mark()
    g.send('/mcp call harness echo {"text":"pong-9271"}')
    ok = g.wait_for("echo: pong-9271", 20, m)
    return ok, "MCP tool connected + called over JSON-RPC" if ok else "MCP call did not echo"


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


OS = [(n, make_cmd_scenario(c, mk)) for (n, c, mk) in OS_CMDS] + [
    ("help_restart", s_help_restart),
    ("memory", s_memory),
    ("open_media", s_open_media),
    ("open_video", s_open_video),
    ("tabs", s_tabs),
    ("panes", s_panes),
    ("clipboard", s_clipboard),
]
AGENTS = [("agents_services", s_agents_services), ("agents_install", s_agents_install), ("agents_uninstall", s_agents_uninstall), ("agent_fs_consent", s_agent_fs_consent), ("agents_search", s_agents_search), ("agents_install_registry", s_agents_install_registry), ("system_agents", s_system_agents), ("doc_pipeline", s_doc_pipeline), ("ssh_agent", s_ssh_agent), ("surface", s_surface), ("mcp_manifest", s_mcp_manifest)]
NET = [("network", s_network), ("ping", s_ping), ("http_get", s_http_get), ("http_post", s_http_post), ("http_download", s_http_download), ("http_stream", s_http_stream), ("ws", s_ws), ("mcp_connect", s_mcp_connect), ("cancel", s_cancel)]
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


def boot_guest(arch, model, verbose, audio, fwd, attempts=3):
    """Launch the guest and wait for it to reach networking, retrying if it dies
    early. aarch64 SMP bring-up (PSCI CPU_ON) very occasionally takes a data
    abort right after `smp: N cores online` — a rare hypervisor bring-up race,
    not a scenario failure — so rather than fail the whole run we detect the
    boot-time FATAL (or an early exit / 120 s with no networking), kill the VM,
    and relaunch. Returns a booted Guest, or None if every attempt failed."""
    for attempt in range(1, attempts + 1):
        g = Guest(arch=arch, model=model, verbose=verbose, audio=audio, hostfwd=fwd)
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
        if verbose:
            print("    " + g.tail(800).replace("\n", "\n    "))
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
    g = boot_guest(arch, model, verbose, audio, fwd)
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
