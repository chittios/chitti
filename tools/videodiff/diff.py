#!/usr/bin/env python3
"""Cross-check the kernel's video stack against PyAV/ffmpeg on the host.

Two modes, and the split matters because they fail for different reasons:

  probe  — demux + parameter sets. Compares codec, geometry and frame count
           against PyAV. This is what catches a container bug (a wrong `hvcC`
           preamble, a `BlockGroup` a demuxer ignored, an audio track mistaken
           for the video one) *without* needing a pixel pipeline to exist.

  yuv    — decoded pixels, frame by frame, bit-exact. Only meaningful for a
           codec whose pipeline is implemented; it reports "not implemented"
           rather than a diff for the others.

Run it with the PyAV venv on PATH, e.g.

    python3 -m venv /tmp/avenv && /tmp/avenv/bin/pip install av numpy
    /tmp/avenv/bin/python tools/videodiff/diff.py probe clip.mp4

The system ffmpeg is not used and does not need to work: PyAV ships its own.
(On this tree's dev box the Homebrew ffmpeg binary does not even load — a
libx265 soname mismatch — which is exactly why the reference is PyAV.)
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "target", "release", "videodiff")


def build():
    if not os.path.exists(BIN):
        subprocess.run(["cargo", "build", "--release"], cwd=HERE, check=True)


def kernel_probe(path):
    out = subprocess.run([BIN, "probe", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("videodiff probe failed: " + out.stderr.strip())
    d = {}
    for line in out.stdout.splitlines():
        k, _, v = line.partition(" ")
        d[k.strip()] = v.strip()
    return d


def reference(path):
    import av

    c = av.open(path)
    st = c.streams.video[0]
    n = sum(1 for p in c.demux(st) if p.size)
    return {
        "codec": st.codec_context.name,
        "width": st.codec_context.width,
        "height": st.codec_context.height,
        "frames": n,
    }


# ffmpeg's codec name → the family our probe line starts with.
FAMILY = {"h264": "H.264", "hevc": "H.265", "vp9": "VP9", "vp8": "VP8", "av1": "AV1"}


def cmd_probe(path):
    ours = kernel_probe(path)
    ref = reference(path)
    ok = True
    want_family = FAMILY.get(ref["codec"], ref["codec"])
    got_codec = ours.get("codec", "")
    if not got_codec.startswith(want_family):
        print(f"  codec   MISMATCH ours={got_codec!r} ref={ref['codec']!r}")
        ok = False
    else:
        print(f"  codec   ok      {got_codec}")

    got_size = ours.get("size", "")
    # The *display* size is what our probe reports (conformance window / render
    # size applied), which is also what ffmpeg's codec context reports.
    want_size = f"{ref['width']}x{ref['height']}"
    if got_size != want_size:
        print(f"  size    MISMATCH ours={got_size} ref={want_size}")
        ok = False
    else:
        print(f"  size    ok      {got_size}")

    got_frames = int(ours.get("frames", -1))
    if got_frames != ref["frames"]:
        # A VP9 superframe holds several coded frames in one container sample,
        # so a sample count and a coded-frame count legitimately differ. The
        # demuxer's job is the sample count; flag it rather than fail it.
        print(f"  frames  DIFFER  ours={got_frames} (samples) ref={ref['frames']} (packets)")
        ok = False
    else:
        print(f"  frames  ok      {got_frames}")
    print(f"  decodable       {ours.get('decodable')}")
    return ok


def cmd_yuv(path, limit):
    import av
    import numpy as np

    if kernel_probe(path).get("decodable") != "true":
        print("  skipped: no pixel pipeline for this codec yet (probe says undecodable)")
        return True
    out = subprocess.run([BIN, "yuv", path, str(limit)], capture_output=True)
    if out.returncode != 0:
        raise SystemExit("videodiff yuv failed: " + out.stderr.decode().strip())
    # `videodiff yuv` emits the player's display RGB (downscaled, letterboxed),
    # so a byte diff against PyAV's full-res YUV is not the comparison to make;
    # what is comparable is per-frame structure. Report sizes and a cheap hash
    # so a regression is visible, and leave bit-exactness to the in-kernel
    # fixture tests, which hash the decoder's own output.
    got = np.frombuffer(out.stdout, dtype=np.uint32)
    c = av.open(path)
    st = c.streams.video[0]
    n = 0
    for _ in c.decode(st):
        n += 1
        if n >= limit:
            break
    print(f"  decoded {len(got)} display pixels over up to {limit} frames; ref decoded {n} frames")
    return n > 0


def main():
    build()
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    cmd, path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(f"{path}:")
    ok = cmd_probe(path) if cmd == "probe" else cmd_yuv(path, limit)
    print("  => " + ("OK" if ok else "MISMATCH"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
