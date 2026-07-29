# Build the paper. `make` -> main.pdf; `make clean` removes build artifacts.
#
# Requires a TeX distribution (MacTeX / TeX Live). latexmk is preferred because
# it reruns bibtex + latex until references settle; the fallback target spells
# the passes out for a minimal install.

PAPER := main

.PHONY: all clean watch manual wordcount arxiv arxiv-meta

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib
	latexmk -pdf -halt-on-error -interaction=nonstopmode $(PAPER).tex

# No latexmk available: the classic four passes.
manual:
	pdflatex -halt-on-error -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -halt-on-error -interaction=nonstopmode $(PAPER).tex
	pdflatex -halt-on-error -interaction=nonstopmode $(PAPER).tex

watch:
	latexmk -pdf -pvc $(PAPER).tex

# Rough body word count (strips TeX markup). Needs detex or texcount.
wordcount:
	@detex $(PAPER).tex 2>/dev/null | wc -w || texcount -1 $(PAPER).tex

# arXiv submission: build the source tarball and check it against the rules that
# actually cause holds (see arxiv.sh for the rule-by-rule provenance).
arxiv: $(PAPER).pdf
	./arxiv.sh

# Regenerate the ASCII title/abstract for the submission form from the paper
# itself, so the metadata cannot drift from what the PDF says.
arxiv-meta:
	python3 tools_meta.py

clean:
	latexmk -C 2>/dev/null || true
	rm -f $(PAPER).aux $(PAPER).blg $(PAPER).log $(PAPER).out \
	      $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz $(PAPER).toc \
	      arxiv-submission.tar.gz
	@echo "kept $(PAPER).bbl -- arXiv wants it in the source package"
