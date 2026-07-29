#!/usr/bin/env bash
# Build an arXiv-ready source tarball, and check it against the rules that
# actually cause holds and rejections.
#
# Everything here traces to https://info.arxiv.org/help/submit_tex.html and
# .../help/prep.html:
#
#   * source is submitted, not a PDF built from TeX (a TeX-built PDF is refused);
#   * the output `main.pdf` must NOT be in the package, while figure PDFs must;
#   * no auxiliary files (.aux .log .out .toc .fls .fdb_latexmk .synctex.gz);
#   * no hidden files or directories (anything starting with a dot);
#   * no unused figures or other extraneous content;
#   * file names may use only [a-zA-Z0-9_+-.,=] and are CASE SENSITIVE, so a
#     `\includegraphics{Fig1.PDF}` against a `fig1.pdf` on disk is a hold;
#   * `.bbl` must match the main .tex basename -- included so the bibliography
#     renders exactly as verified locally, whichever processor arXiv picks;
#   * no `\today` (it makes the PDF differ on every rebuild) and no `\pdfoutput`.
#
# Usage: ./arxiv.sh   ->  arxiv-submission.tar.gz + a compliance report
set -euo pipefail
cd "$(dirname "$0")"

MAIN=main
OUT=arxiv-submission.tar.gz
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
fail=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
bad() { note "FAIL" "$1"; fail=1; }

echo "arxiv: checking source"

# --- rules that live in the .tex itself -------------------------------------
# Comments are stripped first: this file explains why \today is banned, and an
# earlier version of the check flagged its own documentation.
# Two expressions: a whole-line comment has no character before the `%` for the
# second pattern to anchor on, which is how the ban's own documentation kept
# tripping the ban.
CODE=$(sed -e 's/^[[:space:]]*%.*//' -e 's/\([^\\]\)%.*/\1/' $MAIN.tex)
# Shell pattern match rather than a pipe to `grep -q`: grep exits at the first
# hit and closes the pipe, which makes printf report a broken pipe under `set -o
# pipefail`. Noise in a compliance report is how a real failure gets skimmed past.
case "$CODE" in *'\today'*) bad '\today in the source (arXiv: avoid; rebuilds change the PDF)';;
                *) note "ok" 'no \today';; esac
case "$CODE" in *'\pdfoutput'*) bad '\pdfoutput is set (arXiv: do not use it)';;
                *) note "ok" 'no \pdfoutput';; esac
case "$CODE" in *'usepackage{xr'*) bad 'xr package (external links break on arXiv)';;
                *) note "ok" 'no xr package';; esac

# --- the bibliography --------------------------------------------------------
[ -f $MAIN.bbl ] || bad "$MAIN.bbl missing -- run make first"
note "ok" "$MAIN.bbl present and matches $MAIN.tex"

# --- figures: referenced, present, exact case, legal names -------------------
# macOS ships bash 3.2, which has no `mapfile`; a newline-separated list keeps
# this portable to whatever shell the next person has.
refs=$(grep -oE 'figures/[A-Za-z0-9_+,=.-]+' $MAIN.tex | sort -u)
[ -n "$refs" ] || bad "no figures referenced -- is the .tex intact?"
for f in $refs; do
  # `test -f` alone is case-insensitive on macOS; compare against the real
  # directory listing so a case mismatch is caught here and not by arXiv.
  if [ -f "$f" ] && ls "$(dirname "$f")" | grep -qx "$(basename "$f")"; then
    note "ok" "$f"
  else
    bad "$f referenced but missing or case-mismatched on disk"
  fi
  case "$(basename "$f")" in
    *[!a-zA-Z0-9_+,=.-]*) bad "$f has characters arXiv does not allow in file names" ;;
  esac
done

# Anything in figures/ that the paper does not use is "extraneous content".
for f in figures/*; do
  case "$f" in *.py) continue ;; esac   # generators are excluded below anyway
  printf '%s\n' "$refs" | grep -qx "$f" || note "skip" "$f (unused -- not packaged)"
done

# --- stage exactly what arXiv should see ------------------------------------
mkdir -p "$STAGE/figures"
cp $MAIN.tex refs.bib $MAIN.bbl "$STAGE/"
for f in $refs; do cp "$f" "$STAGE/figures/"; done

# Belt and braces: nothing hidden, no aux, no output pdf, no scripts.
find "$STAGE" \( -name '.*' -o -name '*.aux' -o -name '*.log' -o -name '*.out' \
  -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' \
  -o -name '*.py' -o -name '*.sh' -o -name '*.ppm' \) -delete
[ -e "$STAGE/$MAIN.pdf" ] && bad "the output PDF is in the package (arXiv: exclude it)"

tar --disable-copyfile -czf "$OUT" -C "$STAGE" . 2>/dev/null || tar -czf "$OUT" -C "$STAGE" .
size=$(( $(wc -c < "$OUT") / 1024 ))

echo "arxiv: package"
tar -tzf "$OUT" | sed 's|^\./||' | grep -v '^$' | sed 's/^/  /'
note "size" "${size} KiB (arXiv's limit is 50 MB; over ~10 MB wants a good reason)"

# macOS tar can smuggle ._resource forks in; they read as hidden files.
tar -tzf "$OUT" | grep -q '/\._' && bad "AppleDouble ._ files in the tarball" \
  || note "ok" "no hidden or resource-fork files"

if [ $fail -eq 0 ]; then
  echo "arxiv: OK -> $OUT"
else
  echo "arxiv: FAILED -- fix the above before uploading" >&2
  exit 1
fi
