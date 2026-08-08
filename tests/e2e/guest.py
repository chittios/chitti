"""Drive a ChittiOS guest booted under QEMU over its serial console.

Spawns `cargo xtask run` (which sets up QEMU with user-mode networking and a
serial `mon:stdio` console), reads the serial output on a background thread,
and lets a test feed shell commands and wait for expected output. Reuses
xtask's QEMU invocation so the harness doesn't have to reconstruct it.
"""

import codecs
import os
import re
import signal
import subprocess
import threading
import time

# CSI (`ESC [ … final`) **and** OSC (`ESC ] … BEL` or `ESC ] … ESC \`).
#
# OSC matters because the guest's clipboard pushes contents to the host
# terminal as OSC 52 (`ESC ] 52 ; c ; <base64> BEL`), which `/clip` and
# `/pbcopy` both emit. With only the CSI pattern, `text()` sees an escape it
# cannot match, treats it as unterminated, and holds back everything after it —
# so the command's own output never becomes visible and the scenario times out
# on a line the guest definitely printed.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Default forwarded port set (matches run.py's SVC_PORT/SVC_HTTP_PORT/SVC_SSH_PORT).
# A parallel run overrides these per shard.
SVC_PORT_DEFAULT = 7099
SVC_HTTP_PORT_DEFAULT = 7100
SVC_SSH_PORT_DEFAULT = 7101


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


class Guest:
    def __init__(self, arch="aarch64", model="qwen3.5-0.8b", verbose=False, audio="off", hostfwd=None,
                 no_model=False, model_disk=None, release=False, disk=None, smp=None, share=None):
        self.verbose = verbose
        self.buf = bytearray()
        self.lock = threading.Lock()
        # Incremental decode of `buf` into ANSI-stripped text.
        #
        # `text()` used to decode and regex-strip the WHOLE buffer on every
        # call, and `wait_for` calls it ten times a second. Over a full sweep
        # the buffer reaches several MiB, so each poll spent ~17 ms re-doing
        # work it had already done — and, worse, held `self.lock` for a
        # multi-MiB copy while the reader thread was trying to take it. The
        # reader then starved, guest output arrived late, and scenarios that
        # pass in 3 s standing alone timed out at 12 s in a sweep. Sixteen
        # scenarios failed that way, scattered through the run, looking exactly
        # like a kernel regression.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._decoded = ""
        self._consumed = 0
        self._pending = ""
        env = dict(os.environ)
        # Headless: no display window. `audio="none"` gives the guest a
        # virtio-snd device on a silent backend (so /voice has a sound device to
        # enumerate); "off" omits audio entirely (faster; net/OS tests).
        env["CHITTI_DISPLAY"] = "none"
        env["CHITTI_AUDIO"] = audio
        # Fewer vCPUs per guest for a *parallel* run (--jobs > 1): several VMs
        # each asking for `-smp 8` oversubscribes the host and slows every shard.
        if smp:
            env["CHITTI_SMP"] = str(smp)
        # Opt-in slirp host-forward so the host can reach a guest TCP listener
        # (the Network-service-agent e2e). xtask adds hostfwd for this port.
        if hostfwd:
            env["CHITTI_HOSTFWD"] = str(hostfwd)
        # The forwarded host-side ports, so scenarios reach THIS guest's
        # listeners: under --jobs each shard boots on its own ports (the port
        # number is both the guest listen port and the host forward, 1:1).
        self.svc_http_port = SVC_HTTP_PORT_DEFAULT
        self.svc_ssh_port = SVC_SSH_PORT_DEFAULT
        self.svc_port = SVC_PORT_DEFAULT
        if hostfwd:
            fw = [p for p in (str(hostfwd).split(",")) if p.strip().isdigit()]
            if len(fw) > 0:
                self.svc_port = int(fw[0])
            if len(fw) > 1:
                self.svc_http_port = int(fw[1])
            if len(fw) > 2:
                self.svc_ssh_port = int(fw[2])
        # Opt-in FAT disk carrying `model_disk` as chat.gguf, for the runtime
        # `/model load` scenario; `no_model` boots with no model in RAM so the
        # runtime-load path is proven from nothing. Such guests boot *next to*
        # the main e2e guest, which holds QEMU's write lock on the shared
        # voice-assets disk — skip it or the second QEMU fails to launch.
        # Opt-in host shared folder over virtio-9p, mounted by the guest at
        # /host. Off by default: attaching one adds a mount, which would change
        # what unrelated /mounts and /ls scenarios see.
        if share:
            env["CHITTI_SHARE"] = str(share)
        if model_disk:
            env["CHITTI_MODEL_DISK"] = str(model_disk)
        if no_model:
            env["CHITTI_VOICE_DISK"] = "off"
        cmd = ["cargo", "xtask", "run", "-arch", arch, "-model", model]
        if no_model:
            cmd.append("--no-model")
        # Optimized kernel — inference-speed scenarios need release timing.
        if release:
            cmd.append("--release")
        # A raw data disk (`--disk <SIZE>`), for scenarios that need a block
        # device: /disks, /mkext4, and the install paths. Off by default because
        # attaching one changes where synapse persistence lands, which would alter
        # unrelated scenarios' behaviour.
        if disk:
            cmd += ["--disk", str(disk)]
        # New session/process group so `close()` can kill the whole tree
        # (cargo → xtask → qemu) at once — SIGTERM to just `cargo` would orphan
        # the qemu grandchild, which would keep the HVF/hostfwd ports and block a
        # relaunch on boot retry.
        self.proc = subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        # `os.read` returns whatever is available rather than blocking for a
        # full buffer, so this stays responsive while taking the lock once per
        # chunk instead of once per byte. `stdout.read(1)` meant one lock
        # acquisition per character of guest output.
        fd = self.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            with self.lock:
                self.buf.extend(chunk)
            if self.verbose:
                try:
                    os.write(1, chunk)
                except OSError:
                    pass

    def text(self):
        """Everything the guest has printed, decoded and ANSI-stripped.

        Only the bytes that arrived since the last call are decoded; the rest
        is cached. A trailing partial escape sequence is held back rather than
        stripped, since `ANSI.sub` on half a sequence would emit the tail as
        literal text and the pattern the caller is waiting for might straddle
        it.
        """
        with self.lock:
            if len(self.buf) == self._consumed:
                return self._decoded
            fresh = bytes(self.buf[self._consumed:])
            self._consumed = len(self.buf)
        chunk = self._pending + self._decoder.decode(fresh)
        self._pending = ""
        # Hold back an unterminated escape so it can be stripped once its final
        # byte arrives: `ANSI.sub` on half a sequence emits the tail as literal
        # text, and the pattern a caller is waiting for might straddle it.
        cut = chunk.rfind("\x1b")
        if cut != -1 and not ANSI.match(chunk, cut):
            self._pending = chunk[cut:]
            chunk = chunk[:cut]
        self._decoded += ANSI.sub("", chunk)
        return self._decoded

    def mark(self):
        """A cursor into the output stream, so `wait_for` ignores prior text."""
        return len(self.text())

    def since(self, mark):
        """Everything the guest has printed since `mark`.

        For assertions that need to look at the *whole* reply rather than wait
        for one line — "the saved path is there AND there is no re-decode
        warning" is two facts about one block of output, and two `wait_for`s
        would be two timeouts."""
        return self.text()[mark:]

    def saw(self, pattern, since=0):
        """True if `pattern` has already appeared after offset `since`.

        Non-blocking, unlike `wait_for` — for asserting a message did **not**
        happen (e.g. that a cancel was not reported as a network error). Waiting
        for the absence of text can only ever be a timeout, so it needs its own
        helper rather than a `wait_for` someone reads as a positive assertion.
        """
        return pattern in self.text()[since:]

    def wait_for(self, pattern, timeout=20.0, since=0):
        """Block until `pattern` appears after offset `since`, or timeout.
        Returns True on match, False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                # Guest exited; one last check of what we captured.
                return pattern in self.text()[since:]
            if pattern in self.text()[since:]:
                return True
            time.sleep(0.1)
        return False

    def wait_quiet(self, quiet=1.5, timeout=180.0):
        """Wait until the serial output has been idle for `quiet` seconds — i.e.
        a generating turn (chat/infer) has finished and the prompt is back, so
        the next command isn't typed mid-generation (the decode loop's Ctrl+C
        check would eat it)."""
        deadline = time.time() + timeout
        last_len = len(self.text())
        last_change = time.time()
        while time.time() < deadline:
            n = len(self.text())
            if n != last_len:
                last_len = n
                last_change = time.time()
            elif time.time() - last_change >= quiet:
                return True
            time.sleep(0.2)
        return False

    def send(self, line):
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def send_raw(self, data: bytes):
        """Send raw bytes with no trailing newline — for control keys (Ctrl+C =
        b'\\x03', Ctrl+V = b'\\x16') and partial input lines."""
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def close(self):
        try:
            self.send("/exit")
            time.sleep(0.5)
        except Exception:
            pass
        # Kill the whole process group (cargo → xtask → qemu), not just `cargo`,
        # so no orphaned qemu lingers holding the HVF/hostfwd ports.
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def tail(self, n=800):
        return self.text()[-n:]
