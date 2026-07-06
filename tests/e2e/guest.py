"""Drive a Chitti OS guest booted under QEMU over its serial console.

Spawns `cargo xtask run` (which sets up QEMU with user-mode networking and a
serial `mon:stdio` console), reads the serial output on a background thread,
and lets a test feed shell commands and wait for expected output. Reuses
xtask's QEMU invocation so the harness doesn't have to reconstruct it.
"""

import os
import re
import subprocess
import threading
import time

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


class Guest:
    def __init__(self, arch="aarch64", model="qwen3.5-0.8b", verbose=False, audio="off"):
        self.verbose = verbose
        self.buf = bytearray()
        self.lock = threading.Lock()
        env = dict(os.environ)
        # Headless: no display window. `audio="none"` gives the guest a
        # virtio-snd device on a silent backend (so /voice has a sound device to
        # enumerate); "off" omits audio entirely (faster; net/OS tests).
        env["CHITTI_DISPLAY"] = "none"
        env["CHITTI_AUDIO"] = audio
        cmd = ["cargo", "xtask", "run", "-arch", arch, "-model", model]
        self.proc = subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while True:
            chunk = self.proc.stdout.read(1)
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
        with self.lock:
            raw = bytes(self.buf)
        return ANSI.sub("", raw.decode(errors="replace"))

    def mark(self):
        """A cursor into the output stream, so `wait_for` ignores prior text."""
        return len(self.text())

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

    def close(self):
        try:
            self.send("/exit")
            time.sleep(0.5)
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def tail(self, n=800):
        return self.text()[-n:]
