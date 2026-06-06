#!/bin/bash
# compile_paper.sh
# Builds paper/main.pdf from LaTeX source using a 4-pass pipeline.
# Citation engine: biblatex (authoryear/APA-7 style).
# Backend: biber (preferred) with automatic fallback to bibtex.

set -e

PAPER_DIR="$(dirname "$0")/../paper"
cd "$PAPER_DIR"

echo "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode main.tex

# Use biber if available (biblatex recommended backend for full APA 7).
# Falls back to bibtex when biber is not installed (e.g. basic TeX Live).
# With backend=bibtex, biblatex uses `bibtex main` (reads main-blx.bib).
if command -v biber &>/dev/null; then
    echo "=== Pass 2: biber (APA 7 recommended backend) ==="
    biber main
else
    echo "=== Pass 2: bibtex (biblatex bibtex backend fallback) ==="
    bibtex main
fi

echo "=== Pass 3: pdflatex (resolve citations) ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== Pass 4: pdflatex (finalise cross-references) ==="
pdflatex -interaction=nonstopmode main.tex

echo ""
if [ -f main.pdf ]; then
    SIZE=$(du -h main.pdf | cut -f1)
    echo "SUCCESS: paper/main.pdf generated (${SIZE})"
else
    echo "FAILURE: main.pdf was not produced."
    exit 1
fi
