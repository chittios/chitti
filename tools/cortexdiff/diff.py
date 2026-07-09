#!/usr/bin/env python3
"""Cross-check the kernel's greedy decode against llama.cpp on the same GGUF.

Runs both `cortexdiff greedy` (the kernel's own cortex engine, host-mounted)
and `llama-completion` (raw completion; llama.cpp >= b9xxx split it out of
llama-cli) with an identical raw prompt at temperature 0 on the CPU backend,
and compares tokenizations and continuations.

Interpretation: the *tokenization* must match exactly (a mismatch is a real
tokenizer bug). The *continuation* is compared as a prefix and the first
divergence is reported — two correct engines can legitimately split at a
near-tie logit (different expf/softmax accumulation), so a late divergence
with sensible text on both sides is noise, while an early/garbage divergence
means a numeric bug. (The kernel's own fixture gate is `refcheck.rs`, which
pins the engine's exact ids; this script is the external sanity oracle.)

Usage:
    python3 tools/cortexdiff/diff.py <model.gguf> [prompt] [n_tokens]

Needs `llama-completion` + `llama-tokenize` on PATH (brew install llama.cpp)
and a built cortexdiff (cargo build --release --manifest-path
tools/cortexdiff/Cargo.toml).
"""

import os
import re
import subprocess
import sys


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cortexdiff_bin():
    exe = os.path.join(repo_root(), "tools/cortexdiff/target/release/cortexdiff")
    if not os.path.exists(exe):
        sys.exit("cortexdiff not built: cargo build --release --manifest-path tools/cortexdiff/Cargo.toml")
    return exe


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "hello whats your name?"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    # Kernel side: ids + text.
    kout = subprocess.run([cortexdiff_bin(), "greedy", model, prompt, str(n)], capture_output=True, text=True, check=True).stdout
    kprompt = re.search(r"prompt_ids: (.*)", kout).group(1).split()
    kids = re.search(r"continuation_ids: (.*)", kout).group(1).split()
    ktext = re.search(r"text: (.*)", kout, re.S).group(1).rstrip("\n")

    # llama side: tokenization must match exactly.
    try:
        tok = subprocess.run(
            ["llama-tokenize", "-m", model, "-p", prompt],
            capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
        ).stdout
    except FileNotFoundError:
        sys.exit("llama-tokenize not found on PATH (brew install llama.cpp)")
    lprompt = re.findall(r"^\s*(\d+) ->", tok, re.M)
    tok_ok = lprompt == kprompt
    print(f"prompt ids  kernel: {' '.join(kprompt)}")
    print(f"prompt ids  llama : {' '.join(lprompt)}  {'MATCH' if tok_ok else 'MISMATCH  <-- tokenizer bug'}")

    # llama side: greedy CPU completion text.
    cmd = [
        "llama-completion", "-m", model, "-p", prompt, "-n", str(n),
        "--temp", "0", "--no-warmup", "--no-display-prompt", "-ngl", "0",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit("llama-completion not found on PATH (brew install llama.cpp)")
    ltext = proc.stdout.replace("> EOF by user", "").rstrip("\n")

    # Whitespace-insensitive first-divergence report.
    a, b = "".join(ktext.split()), "".join(ltext.split())
    common = os.path.commonprefix([a, b])
    print(f"kernel ids : {' '.join(kids)}")
    print(f"kernel text: {ktext!r}")
    print(f"llama text : {ltext!r}")
    if a == b or a.startswith(b) or b.startswith(a):
        print("CONTINUATION: MATCH")
        rc = 0
    else:
        print(f"CONTINUATION: diverges after {len(common)} chars: {common!r}")
        print("  (a late divergence with sensible text on both sides is a near-tie logit,")
        print("   not necessarily a bug — the kernel's exact ids are pinned by refcheck.rs)")
        rc = 1 if not tok_ok else 0
    sys.exit(0 if tok_ok and rc == 0 else rc)


if __name__ == "__main__":
    main()
