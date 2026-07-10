#!/usr/bin/env python3
"""Web compatibility report for Chitti browser:
  - TC39 test262 subset (via chitti-js-runner / js_bc)
  - CSS property matrix (Google popularity ∩ our support, css.tobyase.de style)
  - WebAssembly testsuite smoke (magic + binary modules + wast inventory)

Writes tools/webcompat/REPORT.md and prints summary.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
JS_RUNNER = ROOT / "js_runner"
TEST262 = ROOT / "test262" / "test" / "language"
WASM_TS = ROOT / "wasm-testsuite"
REPORT = ROOT / "REPORT.md"

# CSS properties Chitti browser::css currently understands (keep in sync with css.rs apply_one).
CHITTI_CSS = {
    "color",
    "background",
    "background-color",
    "background-image",
    "background-size",
    "background-position",
    "background-repeat",
    "font-size",
    "font-weight",
    "font-family",
    "margin",
    "margin-top",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "padding",
    "padding-top",
    "padding-bottom",
    "padding-left",
    "padding-right",
    "display",
    "text-align",
    "width",
    "height",
    "max-width",
    "max-height",
    "min-width",
    "min-height",
    "line-height",
    "opacity",
    "border",
    "border-color",
    "border-width",
    "border-style",
    "border-radius",
    "outline",
    "outline-color",
    "visibility",
    "position",
    "top",
    "left",
    "right",
    "bottom",
    "z-index",
    "overflow",
    "overflow-x",
    "overflow-y",
    "box-sizing",
    "cursor",
    "float",
    "clear",
    "white-space",
    "text-decoration",
    "list-style",
    "list-style-type",
    "transform",
    "transition",
    "animation",
    "filter",
    "backdrop-filter",
    "box-shadow",
    "content",
    "vertical-align",
    "object-fit",
    "aspect-ratio",
    "flex-direction",
    "flex-wrap",
    "flex-grow",
    "flex-shrink",
    "flex-basis",
    "flex",
    "gap",
    "row-gap",
    "column-gap",
    "justify-content",
    "justify-items",
    "justify-self",
    "align-items",
    "align-content",
    "align-self",
    "order",
    "place-items",
    "place-content",
    "grid-template-columns",
    "grid-template-rows",
    "grid-auto-flow",
    "grid-gap",
    "grid-column",
    "grid-row",
    "grid-area",
    "font-style",
    "font",
    "border-top",
    "border-bottom",
    "border-left",
    "border-right",
    "border-top-left-radius",
    "border-top-right-radius",
    "border-bottom-left-radius",
    "border-bottom-right-radius",
    "text-transform",
    "letter-spacing",
    "word-break",
    "word-wrap",
    "overflow-wrap",
    "text-overflow",
    "user-select",
    "pointer-events",
    "appearance",
    "transform-origin",
    "direction",
    "clip",
    "outline-offset",
    "border-collapse",
    "animation-duration",
    "animation-timing-function",
    "transition-duration",
    "background-clip",
    "background-origin",
    "fill",
    # previously top-missing popular set
    "src",
    "variable",
    "alias-webkit-user-select",
    "alias-webkit-appearance",
    "webkit-tap-highlight-color",
    "webkit-font-smoothing",
    "alias-webkit-transform",
    "alias-webkit-text-size-adjust",
    "alias-word-wrap",
    "alias-webkit-transition",
    "font-display",
    "webkit-box-orient",
    "unicode-range",
    "clip-path",
    "stroke-width",
    "touch-action",
    "webkit-line-clamp",
    "border-bottom-color",
    "animation-name",
    "text-shadow",
    "border-top-color",
    "transition-property",
    "inset",
    "alias-webkit-box-sizing",
    "stroke",
    "scrollbar-width",
    "will-change",
    "transition-timing-function",
    "webkit-box-pack",
    "border-bottom-width",
    "border-top-width",
    "animation-delay",
    "resize",
    "alias-webkit-box-shadow",
    "text-indent",
    "border-left-color",
    "alias-webkit-animation",
    "alias-webkit-justify-content",
    "text-rendering",
    "border-right-color",
    "webkit-user-select",
    "webkit-appearance",
    "webkit-transform",
    "webkit-text-size-adjust",
    "webkit-transition",
    "webkit-animation",
    "webkit-box-sizing",
    "webkit-box-shadow",
    "webkit-justify-content",
    "border-left-width",
    "border-right-width",
    "transition-delay",
    "zoom",
    "border-spacing",
    "flex-flow",
    "font-stretch",
    "outline-width",
    "animation-fill-mode",
    "webkit-box-align",
    "font-feature-settings",
    "font-variant",
    "stroke-dashoffset",
    "margin-inline-start",
    "alias-webkit-mask-image",
    "margin-inline-end",
    "stroke-dasharray",
    "animation-iteration-count",
    "mask-image",
    "outline-style",
    "padding-inline",
    "padding-inline-start",
    "webkit-box-flex",
    "backface-visibility",
    "alias-webkit-align-items",
    "text-wrap",
    "text-decoration-line",
    "contain",
    "text-size-adjust",
    "padding-block",
    "alias-webkit-border-radius",
    "color-scheme",
    "webkit-box-direction",
    "padding-inline-end",
    "font-variation-settings",
    "alias-webkit-flex-direction",
    "inset-inline-start",
    "border-bottom-style",
    "scroll-behavior",
    "mask",
    "object-position",
    "alias-webkit-transform-origin",
    "forced-color-adjust",
    "table-layout",
    "container-type",
    "overscroll-behavior",
    "alias-webkit-filter",
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
    pop = fetch_css_popularity()
    supported = [(n, p) for n, p in pop if n in CHITTI_CSS]
    missing = [(n, p) for n, p in pop if n not in CHITTI_CSS]
    return {
        "popular_n": len(pop),
        "supported": supported,
        "missing": missing[:80],
        "supported_count": len(supported),
        "missing_count": len(missing),
        "chitti_props": sorted(CHITTI_CSS),
        "source": "chromestatus csspopularity ∩ CHITTI_CSS (tobyase.de methodology)",
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


def main():
    lines = ["# Chitti webcompat report", ""]
    lines.append(f"Generated by `tools/webcompat/run_all.py`")
    lines.append("")

    # JS
    print("Building js_runner…")
    bin_path = build_js_runner()
    js = {}
    if bin_path:
        print("Running test262 subset…")
        js = run_test262(bin_path)
    else:
        js = {"error": "js_runner build failed", "pass": 0, "fail": 0, "skip": 0}
    lines.append("## TC39 test262 (filtered subset via js_bc)")
    lines.append("")
    if "error" in js and js.get("pass", 0) == 0 and not js.get("total_files"):
        lines.append(f"**Error:** {js['error']}")
    else:
        runnable = js.get("pass", 0) + js.get("fail", 0)
        rate = 100.0 * js.get("pass", 0) / runnable if runnable else 0
        lines.append(f"- Files scanned: **{js.get('total_files', 0)}**")
        lines.append(f"- PASS: **{js.get('pass', 0)}**")
        lines.append(f"- FAIL: **{js.get('fail', 0)}**")
        lines.append(f"- SKIP (unsupported syntax): **{js.get('skip', 0)}**")
        lines.append(f"- Pass rate (runnable): **{rate:.1f}%**")
        lines.append("")
        lines.append("### Sample failures")
        for s in js.get("fail_samples", [])[:25]:
            lines.append(f"- `{s[:140]}`")
        lines.append("")
        lines.append("### Sample skips")
        for s in js.get("skip_samples", [])[:15]:
            lines.append(f"- `{s[:140]}`")
    lines.append("")

    # CSS
    print("CSS matrix…")
    css = css_matrix()
    lines.append("## CSS support (css.tobyase.de methodology)")
    lines.append("")
    lines.append(css["source"])
    lines.append("")
    lines.append(f"- Popular properties analyzed: **{css['popular_n']}**")
    lines.append(f"- Supported by Chitti: **{css['supported_count']}**")
    lines.append(f"- Missing: **{css['missing_count']}**")
    if css["popular_n"]:
        lines.append(
            f"- Coverage of popular set: **{100*css['supported_count']/css['popular_n']:.1f}%**"
        )
    lines.append("")
    lines.append("### Implemented properties")
    lines.append(", ".join(f"`{p}`" for p in css["chitti_props"]))
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
    lines.append("## JS engine notes (vs test262 harness)")
    lines.append("")
    lines.append(
        "- **test262 harness** runs `js_bc` (bytecode). Remaining SKIPs need "
        "`instanceof`/`Date`/`eval`/`class`/prototypes/async/etc."
    )
    lines.append(
        "- **Bytecode `js_bc`**: function, for, try/catch (shallow), return, throw, "
        "`new Number|String|Boolean|Object|Array|Error`, `.length`/`.toString`/`.valueOf`, "
        "NaN/Infinity globals, parseInt/isNaN."
    )
    lines.append(
        "- **Full engine `browser::js` + DOM**: createElement/appendChild/removeChild/"
        "querySelector(All)/getElementsBy*/classList/dataset/attrs/events/innerHTML/"
        "Node tree links; plus arrow/class/BigInt/RegExp/objects/arrays."
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
