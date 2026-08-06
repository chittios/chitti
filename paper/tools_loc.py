#!/usr/bin/env python3
"""Count the mechanism's non-test lines — the paper's TCB figure.

Why this is a script and not a shell one-liner: the obvious recipe ("everything
before the first `#[cfg(test)]`") was correct when each file ended with a single
test module, and silently wrong the moment one did not. `synapse/executor.rs` now
carries a `#[cfg(test)] mod gate_contract_tests` in the *middle* of the file with
several hundred lines of real code after it, so the one-liner reported 244
non-test lines for a 763-line file. A wrong TCB figure flatters us, which is the
direction that matters.

So this walks braces: a `#[cfg(test)]` attribute followed by a `mod` opens a
region that ends when its brace depth returns to zero, and every line inside is
excluded no matter how many such regions a file has.

    python3 paper/tools_loc.py            # the figure, plus a per-file breakdown
    python3 paper/tools_loc.py --total    # just the number, for scripting
"""

import os
import sys

# The enforcement path, grouped exactly as the paper's E5 table reports it, so the
# table is generated rather than hand-maintained (it had drifted by ~35% before
# this script existed). Deliberately NOT the whole of synapse/ — `ui.rs` is a
# surface registry, `bench.rs` and `redteam.rs` are measurement harnesses, and
# `tenant.rs`/`chunked.rs`/`abi.rs` are the ring-3 crossing rather than the
# authorization decision. Adding a row here is a claim that it is load-bearing for
# the security argument.
COMPONENTS = [
    ("Grammar (gate 1)", ["kernel/src/synapse/grammar.rs"]),
    ("Capability (gate 2)", ["kernel/src/cap/mod.rs"]),
    ("Provenance (gate 3)", ["kernel/src/security/taint.rs", "kernel/src/security/mod.rs"]),
    ("Scope: path normalisation (gate 4)", ["kernel/src/synapse/vpath.rs"]),
    ("Effect policy (what is destructive)", ["kernel/src/synapse/policy.rs"]),
    ("Per-value citations + declassification",
     ["kernel/src/security/citation.rs", "kernel/src/synapse/citation.rs"]),
    ("Executor + primitive registry",
     ["kernel/src/synapse/executor.rs", "kernel/src/synapse/registry.rs"]),
    ("Audit log", ["kernel/src/synapse/audit.rs"]),
    ("Object store", ["kernel/src/synapse/fs.rs"]),
]
MECHANISM = [f for _, files in COMPONENTS for f in files]


def non_test_lines(path):
    """Lines outside every `#[cfg(test)] mod … { … }` region.

    `splitlines()`, not `split("\\n")`: a file ending in a newline makes the
    latter yield a phantom empty final element, which added one line per file and
    put this script 10 ahead of xtask's independent count. Two implementations
    disagreeing is how that was caught — keep them both.
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    total = 0
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("#[cfg(test)]"):
            # Skip the attribute, then the `mod …{` and everything it contains.
            depth = 0
            opened = False
            i += 1
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]:
                    opened = True
                i += 1
                if opened and depth <= 0:
                    break
            continue
        total += 1
        i += 1
    return total


def test_cases(path):
    """`#[test_case]` attributes in one file, ignoring commented-out ones."""
    return sum(
        1 for l in open(path, encoding="utf-8")
        if "#[test_case]" in l and not l.lstrip().startswith("//")
    )


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rows = []
    for name, files in COMPONENTS:
        paths = [os.path.join(root, f) for f in files]
        rows.append((name, sum(non_test_lines(p) for p in paths), sum(test_cases(p) for p in paths)))
    total_lines = sum(r[1] for r in rows)
    total_tests = sum(r[2] for r in rows)

    if "--total" in sys.argv:
        print(total_lines)
        return 0
    if "--latex" in sys.argv:
        # The E5 table body, ready to paste — so the numbers in the paper and the
        # numbers in the tree cannot disagree without someone editing one by hand.
        for name, lines, tests in rows:
            print(f"{name} & {lines:,} & {tests} \\\\".replace(",", "{,}"))
        print("\\midrule")
        print(f"\\textbf{{Total}} & \\textbf{{{total_lines:,}}} & \\textbf{{{total_tests}}} \\\\".replace(",", "{,}"))
        return 0

    for name, lines, tests in rows:
        print(f"  {lines:5}  {tests:3} tests  {name}")
    print(f"  {total_lines:5}  {total_tests:3} tests  TOTAL (mechanism, excluding its own tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
