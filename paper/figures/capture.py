#!/usr/bin/env python3
"""Capture real screenshots of the booted OS for the paper.

Boots ChittiOS under QEMU with the e2e harness's own `Guest` (so the QEMU
invocation is xtask's, not a reconstruction), drives the shell over serial, and
dumps the actual framebuffer through the **QEMU monitor**.

The monitor is reachable because `cargo xtask run` uses `-serial mon:stdio`:
`Ctrl+A c` toggles the shared stdio between the guest's serial console and the
monitor, where `screendump <file>` writes the scanout as a PPM. That is the same
channel the `power_button` e2e scenario uses to press ACPI power, so no extra
plumbing is needed — and it means these are screenshots of the guest's real
framebuffer, not a mock-up or a terminal transcript.

Output: `paper/figures/*.png` (converted from PPM with ImageMagick).

Usage:  python3 paper/figures/capture.py [--keep-ppm]
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "e2e"))

from guest import Guest  # noqa: E402


def screendump(g, path):
    """Toggle to the QEMU monitor, dump the framebuffer, toggle back.

    Both toggles are `Ctrl+A c` on the multiplexed stdio. The sleeps are not
    decoration: the monitor needs a moment to take the escape and to finish
    writing the file before we switch away, and the guest console needs one to
    reattach before the next command is typed at it.
    """
    g.send_raw(b"\x01c")
    time.sleep(0.7)
    g.send(f"screendump {path}")
    time.sleep(2.5)
    g.send_raw(b"\x01c")
    time.sleep(0.7)
    return os.path.exists(path)


def to_png(ppm, png):
    """PPM -> PNG. The framebuffer is already at the panel's pixel size, so this
    only changes container format: no resampling, no recompression artefacts in
    the glyphs."""
    subprocess.run(["magick", ppm, "-strip", png], check=True)
    return png


# Order matters. `/top` is captured before `/redteam` so the compositor shot has a
# clean chat pane; `/clear` wipes the scrollback first for the same reason. The
# `/bench synapse` shot is deliberately absent: taken here it would run while the
# browser pane from the ambient-authority attack is still spinning, and report a
# contended figure that disagrees with the paper's idle-machine table — a
# screenshot that argues with its own text is worse than no screenshot.
PGUP = b"\x1b[5~"

SHOTS = [
    # (name, [commands], scroll-ups, what it is for)
    ("desktop", [], 0, "the console after boot: chat pane, composer, status bar, brand mark"),
    ("panes", ["/clear", "/top"], 0, "the split-pane compositor with a live /top action pane"),
    ("redteam", ["/clear", "/redteam"], 0, "E2/E3/E4 summary on the machine under test"),
    # At scale 2 the split chat pane is only ~47 columns, which wraps the per-attack
    # table into unreadable fragments. Closing the action band first gives the chat
    # the full width (~90 columns), which is what the table was formatted for — and
    # the scrollback survives, so this is the same run, just no longer cramped.
    ("redteam_table", ["/close", "/close"], 8,
     "the per-attack table full width: each refusal and the gate that produced it"),
]


def main():
    keep = "--keep-ppm" in sys.argv
    os.makedirs(HERE, exist_ok=True)
    # A model-less guest boots fast and is what the os-group e2e uses; nothing in
    # these shots depends on inference.
    g = Guest(arch="aarch64", verbose=False, no_model=True)
    made = []
    try:
        print("capture: booting…", flush=True)
        # Wait for a marker the *guest* prints, not one cargo does. Waiting for
        # "chitti" matched "chitti-kernel" in the build output, so when a release
        # rebuild ran first the script sailed past the wait and screendumped a
        # machine that had not booted — every dump silently empty. This string only
        # appears once the shell is interactive.
        if not g.wait_for("Commands start with", 420):
            print("capture: guest never reached the shell prompt", file=sys.stderr)
            print(g.text()[-800:], file=sys.stderr)
            return 1
        time.sleep(4)  # let the splash settle into the steady-state console
        # Captured at the console's NATIVE font scale (8x16 cells). Doubling the
        # scale first was tried and abandoned: it makes the type legible at print
        # size but the console stops looking like itself -- chunky glyphs, half the
        # columns, and every line of output wrapping in places it normally would
        # not. A screenshot's job here is to show what the system actually looks
        # like. Where the fine print matters, the figure earns legibility by giving
        # the output the full pane width (see the `/close` in the table shot) and by
        # the paper carrying the values in a table beside it — not by photographing
        # the OS at a font size nobody runs it at.

        for name, cmds, scrollups, why in SHOTS:
            for c in cmds:
                g.send(c)
                time.sleep(1.0)
            if cmds:
                # /redteam runs a dozen attacks through the router and the
                # baselines really execute (loopback egress), so wait for the
                # serial to go quiet rather than for a fixed sleep.
                g.wait_quiet(quiet=6, timeout=300)
            for _ in range(scrollups):
                g.send_raw(PGUP)  # PgUp: page the chat pane's scrollback
                time.sleep(0.25)
            time.sleep(2.5)
            ppm = os.path.join(HERE, f"{name}.ppm")
            png = os.path.join(HERE, f"{name}.png")
            if screendump(g, ppm):
                to_png(ppm, png)
                if not keep:
                    os.remove(ppm)
                made.append((name, png, why))
                print(f"capture: {name}.png — {why}", flush=True)
            else:
                print(f"capture: FAILED to dump {name} (monitor did not write the file)", file=sys.stderr)
    finally:
        g.close()

    for name, png, _ in made:
        sz = subprocess.run(["magick", "identify", "-format", "%wx%h", png], capture_output=True, text=True)
        print(f"  {os.path.basename(png)}: {sz.stdout.strip()}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
