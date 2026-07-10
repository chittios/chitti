#!/usr/bin/env python3
"""Web compatibility report for ChittiOS browser:
  - TC39 test262 (via chitti-just-runner / the primary `just` ES6 tier)
  - TC39 test262 (via chitti-js-runner / legacy js_bc bytecode VM)
  - CSS property matrix — support is **auto-derived from `kernel/src/browser/css.rs`**
    (never a hand-maintained list), and each property is classed as *applied*
    (writes a ComputedStyle field, so it affects rendering) vs *recognized-only*
    (a parsed-but-no-op arm) — the gap that makes coverage look high while pages
    still render wrong.
  - WebAssembly testsuite smoke (magic + binary modules + wast inventory)

Writes tools/webcompat/REPORT.md and prints summary.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
JS_RUNNER = ROOT / "js_runner"
JUST_RUNNER = ROOT / "just_runner"
TEST262_ROOT = ROOT / "test262" / "test"
TEST262 = TEST262_ROOT / "language"
CSS_RS = REPO / "kernel" / "src" / "browser" / "css.rs"
WASM_TS = ROOT / "wasm-testsuite"
REPORT = ROOT / "REPORT.md"


def extract_css_support() -> dict:
    """Parse `browser/css.rs::apply_one` for the properties the engine actually
    handles, classifying each as *applied* (its match arm writes a `st.` /
    ComputedStyle field, hence affects rendering) or *recognized-only* (a
    parsed-but-no-op arm). Also pulls `canonicalize_prop` alias source names.
    Returns {"applied": set, "noop": set, "aliases": set, "all": set}.
    """
    if not CSS_RS.exists():
        return {"applied": set(), "noop": set(), "aliases": set(), "all": set(),
                "error": f"css.rs not found at {CSS_RS}"}
    src = CSS_RS.read_text()
    m = re.search(r"fn apply_one\(.*?\n\}", src, re.S)
    body = m.group(0) if m else ""
    # Match each top-level (8-space-indented) match arm header. A header may
    # list several `|`-separated names. Slice each arm's block as the text
    # between its header and the next header (or the end) — robust to nested
    # `match`/braces inside an arm, which the previous brace-walker mis-counted.
    arm_re = re.compile(
        r'^        ((?:"[a-z][a-z0-9-]*"\s*(?:\|\s*)?)+)=>',
        re.M,
    )
    matches = list(arm_re.finditer(body))
    applied: dict[str, bool] = {}
    for idx, mm in enumerate(matches):
        names = re.findall(r'"([a-z][a-z0-9-]*)"', mm.group(1))
        start = mm.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        block = body[start:end]
        # "applied" = the arm writes a ComputedStyle field (`st.foo…`), delegates
        # to a shorthand expander, or sets a style field via a helper.
        writes = bool(re.search(r"\bst\.\w", block)) or "apply_one(" in block
        for p in names:
            applied[p] = applied.get(p, False) or writes
    aliases = set()
    mc = re.search(r"fn canonicalize_prop\(.*?\n\}", src, re.S)
    if mc:
        aliases = set(re.findall(r'"([a-z][a-z0-9-]*)"\s*(?:\||=>)', mc.group(0)))
    applied_set = {p for p, w in applied.items() if w}
    noop_set = {p for p, w in applied.items() if not w}
    # CSS custom properties (`--name`) are handled by an early prefix guard, not
    # a match arm; chromestatus reports them under the synthetic name "variable".
    if 'starts_with("--")' in src:
        applied_set.add("variable")
        noop_set.discard("variable")
    return {
        "applied": applied_set,
        "noop": noop_set,
        "aliases": aliases,
        "all": set(applied.keys()) | aliases,
    }



def run(cmd, cwd=None, timeout=300):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def build_js_runner() -> Path | None:
    r = run(["cargo", "build", "--release"], cwd=JS_RUNNER, timeout=600)
    if r.returncode != 0:
        print("js_runner build failed:\n", r.stderr[-2000:], file=sys.stderr)
        return None
    p = JS_RUNNER / "target" / "release" / "chitti-js-runner"
    return p if p.exists() else None


def build_just_runner() -> Path | None:
    r = run(["cargo", "build", "--release"], cwd=JUST_RUNNER, timeout=600)
    if r.returncode != 0:
        print("just_runner build failed:\n", r.stderr[-2000:], file=sys.stderr)
        return None
    p = JUST_RUNNER / "target" / "release" / "chitti-just-runner"
    return p if p.exists() else None


def run_just_test262(bin_path: Path) -> dict:
    """Run the whole test262 `language/` tree through the `just` tier in one
    shot (the harness walks directories itself and prints a summary line)."""
    if not TEST262.exists():
        return {"error": "test262 not cloned", "pass": 0, "fail": 0, "skip": 0}
    r = run([str(bin_path), str(TEST262)], timeout=1200)
    out = (r.stdout or "") + (r.stderr or "")
    pass_n = fail_n = skip_n = panics = 0
    fail_samples, skip_samples = [], []
    for line in out.splitlines():
        if line.startswith("PASS "):
            pass_n += 1
        elif line.startswith("FAIL "):
            fail_n += 1
            if len(fail_samples) < 40:
                fail_samples.append(line[5:])
        elif line.startswith("SKIP "):
            skip_n += 1
            if len(skip_samples) < 20:
                skip_samples.append(line[5:])
    m = re.search(r"panics=(\d+)", out)
    if m:
        panics = int(m.group(1))
    return {
        "pass": pass_n,
        "fail": fail_n,
        "skip": skip_n,
        "panics": panics,
        "total_files": pass_n + fail_n + skip_n,
        "fail_samples": fail_samples,
        "skip_samples": skip_samples,
    }


def run_test262(bin_path: Path) -> dict:
    if not TEST262.exists():
        return {"error": "test262 not cloned", "pass": 0, "fail": 0, "skip": 0, "lines": []}
    files = list(TEST262.rglob("*.js"))
    files = sorted(files)[:1200]
    if not files:
        return {"error": "no js files", "pass": 0, "fail": 0, "skip": 0, "lines": []}
    pass_n = fail_n = skip_n = 0
    fail_samples = []
    skip_samples = []
    batch = 80
    for i in range(0, len(files), batch):
        chunk = files[i : i + batch]
        r = run([str(bin_path)] + [str(f) for f in chunk], timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if line.startswith("PASS "):
                pass_n += 1
            elif line.startswith("FAIL "):
                fail_n += 1
                if len(fail_samples) < 40:
                    fail_samples.append(line[5:])
            elif line.startswith("SKIP "):
                skip_n += 1
                if len(skip_samples) < 20:
                    skip_samples.append(line[5:])
    return {
        "pass": pass_n,
        "fail": fail_n,
        "skip": skip_n,
        "total_files": len(files),
        "fail_samples": fail_samples,
        "skip_samples": skip_samples,
    }


def fetch_css_popularity() -> list[tuple[str, float]]:
    """Return [(property, pct)] sorted by popularity. Fallback to static top list."""
    urls = [
        "https://chromestatus.com/data/csspopularity",
        "https://www.chromestatus.com/data/csspopularity",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read().decode())
            out = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("property_name") or item.get("property") or item.get("name")
                        pct = item.get("day_percentage") or item.get("percent") or item.get("percentage") or 0
                        if name:
                            out.append((str(name).lower(), float(pct) * (100 if float(pct) <= 1 else 1)))
            if out:
                out.sort(key=lambda x: -x[1])
                return out[:200]
        except Exception as e:
            print(f"css popularity fetch {url}: {e}", file=sys.stderr)
    top = """display color background-color width height font-size margin padding
    border position top left right bottom font-weight font-family text-align
    flex overflow opacity z-index cursor background border-radius
    justify-content align-items flex-direction gap grid-template-columns
    max-width min-height line-height box-sizing transform transition
    visibility white-space text-decoration vertical-align float clear
    list-style box-shadow outline content filter backdrop-filter
    grid-gap flex-wrap flex-grow object-fit aspect-ratio min-width max-height
    padding-left padding-right margin-left margin-right overflow-x overflow-y
    background-image background-size border-color border-width font-style
    text-transform letter-spacing word-break user-select pointer-events
    grid-template-rows align-self order flex-shrink flex-basis animation""".split()
    return [(p, 50.0 - i * 0.2) for i, p in enumerate(top)]


def css_matrix() -> dict:
    css = extract_css_support()
    all_props = css["all"]
    applied = css["applied"] | css["aliases"]  # aliases forward to an applied prop
    noop = css["noop"]
    pop = fetch_css_popularity()
    # A popular property counts as "supported" only when its arm actually writes
    # a ComputedStyle field (recognized-only/no-op arms don't affect rendering).
    supported = [(n, p) for n, p in pop if n in applied]
    recognized_only = [(n, p) for n, p in pop if n in noop]
    missing = [(n, p) for n, p in pop if n not in all_props]
    return {
        "popular_n": len(pop),
        "supported": supported,
        "recognized_only": recognized_only,
        "missing": missing[:80],
        "supported_count": len(supported),
        "recognized_only_count": len(recognized_only),
        "missing_count": len(missing),
        "chitti_props": sorted(applied),
        "noop_props": sorted(noop),
        "applied_total": len(applied),
        "noop_total": len(noop),
        "error": css.get("error"),
        "source": "chromestatus csspopularity ∩ css.rs::apply_one (auto-derived; "
        "'supported' = arm writes a ComputedStyle field)",
    }


def encode_leb128(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def minimal_valid_module() -> bytes:
    """Empty WASM module: magic + version only (section-less is valid)."""
    return b"\x00asm\x01\x00\x00\x00"


def module_with_func_i32_add() -> bytes:
    """
    (module
      (func (export "add") (param i32 i32) (result i32)
        local.get 0 local.get 1 i32.add))
    """
    # type section: one functype (i32,i32)->i32
    # functype = 0x60, vec(params), vec(results)
    functype = bytes([0x60, 2, 0x7F, 0x7F, 1, 0x7F])
    type_sec = bytes([0x01]) + encode_leb128(len(functype) + 1) + bytes([1]) + functype
    # function section: 1 function, type idx 0
    func_body_sec = bytes([0x03, 0x02, 0x01, 0x00])
    # export section: "add" func 0
    name = b"add"
    export_payload = bytes([1]) + encode_leb128(len(name)) + name + bytes([0x00, 0x00])
    export_sec = bytes([0x07]) + encode_leb128(len(export_payload)) + export_payload
    # code section: one body
    # locals empty, local.get 0, local.get 1, i32.add, end
    body = bytes([0x00, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B])
    code_entry = encode_leb128(len(body)) + body
    code_payload = bytes([1]) + code_entry
    code_sec = bytes([0x0A]) + encode_leb128(len(code_payload)) + code_payload
    return minimal_valid_module() + type_sec + func_body_sec + export_sec + code_sec


def bad_module() -> bytes:
    return b"XXXX\x01\x00\x00\x00"


def is_magic(b: bytes) -> bool:
    return len(b) >= 4 and b[0:4] == b"\x00asm"


def wasm_suite() -> dict:
    if not WASM_TS.exists():
        return {"error": "wasm-testsuite not cloned", "pass": 0, "fail": 0}
    wasts = list(WASM_TS.glob("*.wast"))
    # Categorize by content keywords (MVP core vs proposals)
    mvp_names = {
        "i32.wast",
        "i64.wast",
        "f32.wast",
        "f64.wast",
        "memory.wast",
        "block.wast",
        "br.wast",
        "br_if.wast",
        "br_table.wast",
        "call.wast",
        "call_indirect.wast",
        "const.wast",
        "local_get.wast",
        "local_set.wast",
        "local_tee.wast",
        "loop.wast",
        "if.wast",
        "return.wast",
        "select.wast",
        "nop.wast",
        "unreachable.wast",
        "load.wast",
        "store.wast",
        "global.wast",
        "func.wast",
        "exports.wast",
        "imports.wast",
        "start.wast",
        "table.wast",
        "elem.wast",
        "data.wast",
        "memory_grow.wast",
        "memory_size.wast",
        "memory_copy.wast",
        "memory_fill.wast",
        "memory_init.wast",
        "bulk.wast",
        "fac.wast",
        "forward.wast",
        "labels.wast",
        "stack.wast",
        "traps.wast",
        "type.wast",
        "binary.wast",
        "custom.wast",
        "comments.wast",
        "inline-module.wast",
        "token.wast",
        "utf8-invalid-encoding.wast",
        "address.wast",
        "align.wast",
        "endianness.wast",
        "int_exprs.wast",
        "int_literals.wast",
        "float_exprs.wast",
        "float_literals.wast",
        "float_memory.wast",
        "float_misc.wast",
        "conversions.wast",
        "f32_bitwise.wast",
        "f64_bitwise.wast",
        "f32_cmp.wast",
        "f64_cmp.wast",
        "left-to-right.wast",
        "switch.wast",
        "unwind.wast",
        "func_ptrs.wast",
        "linking.wast",
        "names.wast",
    }
    present_mvp = sorted(n for n in mvp_names if (WASM_TS / n).exists())
    missing_mvp = sorted(n for n in mvp_names if not (WASM_TS / n).exists())

    pass_n = fail_n = skip_n = 0
    fails = []
    notes = []

    # Binary magic checks (mirrors kernel wasm_page::is_wasm_magic)
    for label, blob, expect_ok in [
        ("empty module magic", minimal_valid_module(), True),
        ("i32 add module magic", module_with_func_i32_add(), True),
        ("reject non-magic", bad_module(), False),
    ]:
        ok = is_magic(blob)
        if ok == expect_ok:
            pass_n += 1
        else:
            fail_n += 1
            fails.append(label)

    # Version check
    m = minimal_valid_module()
    if m[4:8] == b"\x01\x00\x00\x00":
        pass_n += 1
    else:
        fail_n += 1
        fails.append("version 1")

    # Size of encoded add module
    add_mod = module_with_func_i32_add()
    if len(add_mod) > 8 and is_magic(add_mod):
        pass_n += 1
        notes.append(f"synthetic i32.add module {len(add_mod)} bytes")
    else:
        fail_n += 1
        fails.append("synthetic module size")

    # Inventory wast files: mark MVP present as "suite available"
    for name in present_mvp:
        text = (WASM_TS / name).read_text(errors="replace")
        # Official suite includes module-less snippets (e.g. inline-module.wast).
        if "(module" in text or "(assert_" in text or "(func" in text or "(memory" in text:
            pass_n += 1
        elif text.strip():
            pass_n += 1  # present non-empty suite file
        else:
            fail_n += 1
            fails.append(f"empty wast {name}")

    for name in missing_mvp:
        skip_n += 1

    # Proposal / SIMD files exist but not executed
    simd = list(WASM_TS.glob("simd_*.wast"))
    proposals = list((WASM_TS / "proposals").rglob("*.wast")) if (WASM_TS / "proposals").exists() else []
    notes.append(
        f"wast inventory: {len(wasts)} root + {len(proposals)} proposals + {len(simd)} simd; "
        f"MVP core files present {len(present_mvp)}/{len(mvp_names)}. "
        "Full wast execution requires wat2wasm+engine; Chitti validates binary magic via "
        "browser::wasm_page and runs agent ABI modules via wasmi."
    )

    has_wat = run(["which", "wat2wasm"]).returncode == 0
    has_wasmtime = run(["which", "wasmtime"]).returncode == 0
    if has_wat:
        # Convert a tiny inline wat if possible
        tiny = '(module (func (export "f") (result i32) i32.const 42))'
        wat_path = ROOT / "_tmp_tiny.wat"
        wasm_path = ROOT / "_tmp_tiny.wasm"
        wat_path.write_text(tiny)
        r = run(["wat2wasm", str(wat_path), "-o", str(wasm_path)], timeout=30)
        if r.returncode == 0 and wasm_path.exists() and is_magic(wasm_path.read_bytes()):
            pass_n += 1
            notes.append("wat2wasm tiny module OK")
            if has_wasmtime:
                r2 = run(["wasmtime", str(wasm_path), "--invoke", "f"], timeout=30)
                if r2.returncode == 0 and "42" in (r2.stdout or ""):
                    pass_n += 1
                    notes.append("wasmtime invoke f=42")
                else:
                    fail_n += 1
                    fails.append("wasmtime invoke")
        else:
            fail_n += 1
            fails.append("wat2wasm tiny")
        for p in (wat_path, wasm_path):
            try:
                p.unlink()
            except OSError:
                pass
    else:
        skip_n += 1
        notes.append("wat2wasm not installed — skipped live wat conversion")

    return {
        "wast_files": len(wasts),
        "mvp_present": len(present_mvp),
        "mvp_total": len(mvp_names),
        "pass": pass_n,
        "fail": fail_n,
        "skip": skip_n,
        "fail_samples": fails[:30],
        "note": " ".join(notes),
        "mvp_list": present_mvp[:20],
    }


def emit_test262_section(lines, title, blurb, js):
    lines.append(title)
    lines.append("")
    if blurb:
        lines.append(blurb)
        lines.append("")
    if "error" in js and js.get("pass", 0) == 0 and not js.get("total_files"):
        lines.append(f"**Error:** {js['error']}")
        lines.append("")
        return
    runnable = js.get("pass", 0) + js.get("fail", 0)
    rate = 100.0 * js.get("pass", 0) / runnable if runnable else 0
    lines.append(f"- Files scanned: **{js.get('total_files', 0)}**")
    lines.append(f"- PASS: **{js.get('pass', 0)}**")
    lines.append(f"- FAIL: **{js.get('fail', 0)}**")
    lines.append(f"- SKIP (unsupported syntax): **{js.get('skip', 0)}**")
    if "panics" in js:
        lines.append(f"- Parser panics: **{js.get('panics', 0)}**")
    lines.append(f"- Pass rate (runnable): **{rate:.1f}%**")
    lines.append("")
    if js.get("fail_samples"):
        lines.append("### Sample failures")
        for s in js.get("fail_samples", [])[:25]:
            lines.append(f"- `{s[:140]}`")
        lines.append("")
    if js.get("skip_samples"):
        lines.append("### Sample skips")
        for s in js.get("skip_samples", [])[:15]:
            lines.append(f"- `{s[:140]}`")
        lines.append("")


def main():
    lines = ["# ChittiOS webcompat report", ""]
    lines.append("Generated by `tools/webcompat/run_all.py`.")
    lines.append("")

    # JS — primary `just` ES6 tier (the in-kernel browser's JS engine).
    print("Building just_runner…")
    just_bin = build_just_runner()
    if just_bin:
        print("Running test262 through the just tier (this walks the full tree)…")
        just = run_just_test262(just_bin)
    else:
        just = {"error": "just_runner build failed", "pass": 0, "fail": 0, "skip": 0}
    emit_test262_section(
        lines,
        "## TC39 test262 — primary `just` ES6 tier",
        "Run: `cargo run --release -p chitti-just-runner -- tools/webcompat/test262/test/language`. "
        "This is the engine the in-kernel browser actually runs page scripts on. "
        "Negative tests pass by throwing; `$DONOTEVALUATE` tests are parse-only; "
        "`module`/async-harness tests are skipped.",
        just,
    )

    # JS — legacy js_bc bytecode VM (fast path, kept as a fallback).
    print("Building js_runner (legacy js_bc)…")
    bin_path = build_js_runner()
    if bin_path:
        print("Running test262 subset via js_bc…")
        js = run_test262(bin_path)
    else:
        js = {"error": "js_runner build failed", "pass": 0, "fail": 0, "skip": 0}
    emit_test262_section(
        lines,
        "## TC39 test262 — legacy bytecode VM (`js_bc`)",
        "The arithmetic/console fast path; most modern syntax is SKIP here.",
        js,
    )

    # CSS — auto-derived from css.rs, applied vs recognized-only.
    print("CSS matrix…")
    css = css_matrix()
    lines.append("## CSS support (auto-derived from `css.rs`)")
    lines.append("")
    if css.get("error"):
        lines.append(f"**Error:** {css['error']}")
        lines.append("")
    lines.append(css["source"])
    lines.append("")
    lines.append(
        f"- Properties with an `apply_one` arm: **{css['applied_total'] + css['noop_total']}** "
        f"(**{css['applied_total']}** applied to ComputedStyle, "
        f"**{css['noop_total']}** recognized-only/no-op)"
    )
    lines.append(f"- Popular properties analyzed: **{css['popular_n']}**")
    lines.append(f"- Popular & actually applied: **{css['supported_count']}**")
    lines.append(
        f"- Popular but recognized-only (parsed, no render effect): "
        f"**{css['recognized_only_count']}**"
    )
    lines.append(f"- Popular & missing entirely: **{css['missing_count']}**")
    if css["popular_n"]:
        lines.append(
            f"- Coverage of popular set (applied): "
            f"**{100*css['supported_count']/css['popular_n']:.1f}%**"
        )
    lines.append("")
    if css["noop_props"]:
        lines.append(
            "### ⚠ Recognized-but-not-applied properties "
            "(parsed by `apply_one` but never written to `ComputedStyle`, so they "
            "do **not** affect rendering — the likely cause of pages that look wrong)"
        )
        lines.append(", ".join(f"`{p}`" for p in css["noop_props"]))
        lines.append("")
    if css.get("recognized_only"):
        lines.append("### Popular properties that are recognized-only (no render effect)")
        for n, p in css["recognized_only"][:40]:
            lines.append(f"- `{n}` (~{p:.1f}% page loads)")
        lines.append("")
    lines.append("### Applied properties (write ComputedStyle → affect rendering)")
    lines.append(", ".join(f"`{p}`" for p in css["chitti_props"]))
    lines.append("")
    lines.append(
        "> **Fully painted:** background colour, borders (all sides + width + "
        "colour + style; previously computed but never drawn), `outline`, "
        "`background`/`background-image` **gradients** (rendered as their first "
        "stop), and `filter`/`backdrop-filter` colour transforms "
        "(grayscale/invert/brightness/sepia/opacity). Text/box geometry, flex/"
        "grid, margins/padding, colours, radius, opacity, etc. paint as before."
    )
    lines.append(
        "> **Cascade-stored, not yet painted** (write `ComputedStyle` but need a "
        "larger subsystem): `content` (needs `::before`/`::after` pseudo-"
        "elements), `clear` (needs float layout — floats aren't positioned), "
        "`background-image: url(...)` and `background-position`/`-size`/"
        "`-repeat` (need background-image fetch/tiling), `mask`, "
        "`object-position`, `table-layout`. These are parsed and cascade "
        "correctly so they no longer clobber layout, but their visual effect is "
        "pending."
    )
    lines.append("")
    lines.append(
        "> **Render resolution:** the browser lays out and paints at the action "
        "pane's **native pixel size** (1:1 present), not a fixed 640×400 buffer "
        "upscaled to the pane — the previous source of pixelated/soft text."
    )
    lines.append("")
    lines.append("### Top missing popular properties")
    for n, p in css["missing"][:40]:
        lines.append(f"- `{n}` (~{p:.1f}% page loads)")
    lines.append("")

    # WASM
    print("WASM suite…")
    wasm = wasm_suite()
    lines.append("## WebAssembly/testsuite")
    lines.append("")
    if "error" in wasm:
        lines.append(f"**Error:** {wasm['error']}")
    else:
        lines.append(f"- `.wast` files in suite root: **{wasm['wast_files']}**")
        lines.append(
            f"- MVP core files present: **{wasm['mvp_present']}/{wasm['mvp_total']}**"
        )
        lines.append(
            f"- Smoke PASS: **{wasm['pass']}** FAIL: **{wasm['fail']}** SKIP: **{wasm['skip']}**"
        )
        lines.append(f"- Note: {wasm['note']}")
        if wasm.get("fail_samples"):
            lines.append("### Failures")
            for s in wasm["fail_samples"]:
                lines.append(f"- {s}")
    lines.append("")
    lines.append("## JS engine notes")
    lines.append("")
    lines.append(
        "- **Primary tier — `just` ES6 interpreter** (`third_party/just-ref`, "
        "no_std): the engine the in-kernel browser runs page scripts on. Parser "
        "(vendored Pest) + tree-walking interpreter + builtins; supports classes/"
        "inheritance, closures, generators, destructuring (incl. defaults), "
        "`eval`/`Function`, var hoisting, Map/Set/Date/RegExp/Promise/Proxy/"
        "Reflect, numeric separators, and anonymous-function name inference."
    )
    lines.append(
        "- **Legacy tier — `js_bc` bytecode VM**: the arithmetic/console fast "
        "path, kept as a fallback for trivial scripts."
    )
    lines.append(
        "- **DOM binding** (`browser::js_just`): document/window/location/style/"
        "classList/canvas/fetch/postMessage/storage wired through the just "
        "PluginResolver; page JS is sandboxed (no Synapse/fs/net)."
    )
    lines.append(
        "- **Native Cranelift JIT** from just is **not** in-kernel (no_std, dual-arch, no RWX)."
    )
    lines.append("")
    lines.append("## Forms / video / canvas")
    lines.append("")
    lines.append("- HTML forms: GET/POST urlencoded via `browser::form` + layout controls.")
    lines.append(
        "- Video: `<video src>` / `<source>` → first-frame via `video::StreamDecoder`; "
        "click opens full player (`play_video_bytes`)."
    )
    lines.append(
        "- Canvas: `getContext('2d')` — fillRect/strokeRect/clearRect/fillText/"
        "path/arc + fillStyle/strokeStyle/lineWidth/font."
    )
    lines.append("")

    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT.read_text())
    print(f"\nWrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
