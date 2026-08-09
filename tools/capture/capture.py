#!/usr/bin/env python3
"""Capture ChittiOS screenshots and an animation from a real boot.

Boots the kernel under QEMU with a QMP socket and a serial socket, types a
scripted session into the shell, and screendumps the framebuffer at each step.
The output is real frames from the real OS — the same path
`DEVELOPMENT.md` documents for headless framebuffer verification, scripted so
the README and the website can be regenerated rather than hand-curated.

Why a script and not a screen recording: a recording captures whatever the host
compositor did, at whatever moment, and cannot be reproduced. This produces the
same frames every run from the same kernel, so a stale screenshot is a rebuild
away rather than a re-shoot.

    python3 tools/capture/capture.py --arch x86_64 --out docs/media

Requires: qemu (via `cargo xtask run`). The stills need nothing else; the
animation uses Pillow if it is installed, else ImageMagick.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Qmp:
    """Minimal QMP client — connect, negotiate, execute."""

    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(100):
            try:
                self.sock.connect(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.1)
        else:
            raise SystemExit(f"capture: QMP socket never appeared at {path}")
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self.f.readline()  # greeting
        self.execute("qmp_capabilities")

    def execute(self, cmd: str, **args: object) -> dict:
        msg = {"execute": cmd}
        if args:
            msg["arguments"] = args
        self.f.write(json.dumps(msg) + "\n")
        self.f.flush()
        # Skip asynchronous events; the reply is the first object with a
        # `return` or `error` key.
        while True:
            line = self.f.readline()
            if not line:
                raise SystemExit("capture: QMP closed unexpectedly")
            obj = json.loads(line)
            if "return" in obj or "error" in obj:
                if "error" in obj:
                    raise SystemExit(f"capture: QMP error: {obj['error']}")
                return obj

    def screendump(self, path: Path) -> None:
        self.execute("screendump", filename=str(path))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class Guest:
    """`cargo xtask run` as a child process, driven over its stdio.

    The same approach `tests/e2e/guest.py` takes, and for the same reason: xtask
    builds a long, carefully-ordered QEMU command (accel, machine, disks, net,
    share, display) and reconstructing it here to add one flag would mean
    reproducing all of it — and then drifting from it. `CHITTI_QEMU_EXTRA` adds
    the QMP socket; everything else is xtask's.
    """

    def __init__(self, arch: str, qmp_path: str, verbose: bool) -> None:
        env = dict(os.environ)
        env["CHITTI_QEMU_EXTRA"] = f"-qmp unix:{qmp_path},server,nowait"
        env["CHITTI_DISPLAY"] = "none"
        env.setdefault("CHITTI_SAMPLE_FILES", "1")
        cmd = ["cargo", "xtask", "run", "-arch", arch, "--release", "--no-model"]
        print("capture: " + " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(
            cmd, cwd=REPO, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        os.set_blocking(self.proc.stdout.fileno(), False)
        self.buf = bytearray()
        self.verbose = verbose

    def send(self, text: str) -> None:
        self.proc.stdin.write(text.encode())
        self.proc.stdin.flush()

    def drain(self) -> str:
        try:
            while True:
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                self.buf += chunk
                if self.verbose:
                    sys.stderr.write(chunk.decode("utf-8", "replace"))
        except (BlockingIOError, TypeError):
            pass
        return self.buf.decode("utf-8", "replace")

    def wait_for(self, needle: str, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.drain():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.3)
        return False

    def stop(self) -> None:
        # Kill the whole group: cargo -> xtask -> qemu.
        try:
            os.killpg(os.getpgid(self.proc.pid), 15)
            self.proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.proc.pid), 9)
            except ProcessLookupError:
                pass


# The session, as (label, keystrokes, settle seconds). `None` keystrokes means
# "just wait and shoot" — used for the boot splash.
#
# Chosen to show what the OS actually is rather than what looks busiest: the
# shell it boots to, that it explains itself, that it drives real hardware, and
# that the agent's authority is gated and audited.
SCRIPT: list[tuple[str, str | None, float]] = [
    ("01-boot", None, 2.0),
    ("02-help", "/help\n", 3.0),
    ("03-about", "\x1b/about\n", 2.5),
    ("04-disks", "/disks\n", 2.5),
    ("05-lspci", "/lspci\n", 2.5),
    ("06-network", "/network\n", 2.5),
    ("07-samples", "/ls /samples\n", 2.5),
    ("08-image", "/open /samples/images/fruits.jpg\n", 5.0),
    ("09-close", "/close\n", 1.5),
    ("10-audit", "/audit\n", 2.5),
    ("11-agents", "/agents\n", 3.0),
    ("12-top", "/top\n", 3.0),
]


def to_png(ppm: Path, png: Path) -> None:
    """PPM -> PNG, by whatever is installed."""
    if shutil.which("magick"):
        subprocess.run(["magick", str(ppm), str(png)], check=True)
    elif shutil.which("convert"):
        subprocess.run(["convert", str(ppm), str(png)], check=True)
    elif shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "png", str(ppm), "--out", str(png)],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            raise SystemExit("capture: need ImageMagick, sips or Pillow to convert PPM")
        Image.open(ppm).save(png)
    ppm.unlink(missing_ok=True)


def animate(frames: list[Path], out_dir: Path, hold_ms: int = 1800) -> None:
    """Build an animated WebP and a GIF from the stills.

    Pillow first, ImageMagick second. **Not ffmpeg**: the Homebrew build on this
    machine has an x265 ABI mismatch (`_x265_api_get_215` against
    `libx265.216`) and aborts on any encode — the same broken install the video
    work routes around by using PyAV. Pillow writes both formats natively, so
    the animation needs no external binary at all.

    A slideshow of a shell needs to be *read*, so each frame holds for most of
    two seconds rather than running at a video frame rate.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        Image = None  # type: ignore

    if Image is not None:
        # Downscale to a width that stays legible in a README without being
        # megabytes; the source is 1280 wide.
        target_w = 1024
        imgs = []
        for f in frames:
            im = Image.open(f).convert("RGB")
            w, h = im.size
            imgs.append(im.resize((target_w, round(h * target_w / w)), Image.LANCZOS))

        webp = out_dir / "demo.webp"
        imgs[0].save(webp, save_all=True, append_images=imgs[1:],
                     duration=hold_ms, loop=0, quality=72, method=4)
        print(f"capture: demo.webp ({webp.stat().st_size // 1024} KiB)")

        # GIF is the fallback for anywhere WebP is not rendered. Palette-quantise
        # each frame: a 256-colour GIF of a terminal is fine, and the default
        # conversion dithers text into mush.
        gif = out_dir / "demo.gif"
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
        pal[0].save(gif, save_all=True, append_images=pal[1:],
                    duration=hold_ms, loop=0, optimize=True)
        print(f"capture: demo.gif ({gif.stat().st_size // 1024} KiB)")
        return

    if shutil.which("magick"):
        subprocess.run(
            ["magick", "-delay", str(hold_ms // 10), "-loop", "0",
             *[str(f) for f in frames], "-resize", "1024x",
             str(out_dir / "demo.gif")], check=True)
        print("capture: demo.gif (ImageMagick)")
        return

    print("capture: no Pillow or ImageMagick — stills only", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="x86_64", choices=["aarch64", "x86_64"])
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="mirror the guest serial to stderr")
    args = ap.parse_args()

    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    qmp_path = "/tmp/chitti-cap-q.sock"
    Path(qmp_path).unlink(missing_ok=True)

    guest = Guest(args.arch, qmp_path, args.verbose)
    frames: list[Path] = []
    try:
        print("capture: waiting for the shell…", flush=True)
        if not guest.wait_for("~ >", args.boot_timeout):
            tail = guest.drain()[-2000:]
            raise SystemExit(f"capture: shell never came up.\n--- serial tail ---\n{tail}")
        print("capture: shell up", flush=True)
        qmp = Qmp(qmp_path)

        for label, keys, settle in SCRIPT:
            if keys is not None:
                guest.send(keys)
            time.sleep(settle)
            guest.drain()  # keep the pipe from filling and blocking the guest
            ppm = out / f"{label}.ppm"
            png = out / f"{label}.png"
            qmp.screendump(ppm)
            for _ in range(60):
                if ppm.exists() and ppm.stat().st_size > 0:
                    break
                time.sleep(0.1)
            if not ppm.exists():
                print(f"capture: no frame for {label}, skipping", file=sys.stderr)
                continue
            to_png(ppm, png)
            frames.append(png)
            print(f"capture: {png.name}", flush=True)

        qmp.close()
    finally:
        guest.stop()
        Path(qmp_path).unlink(missing_ok=True)

    if frames:
        animate(frames, out)
    print(f"capture: {len(frames)} still(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
