#!/bin/bash
# compile_paper.sh
# Build paper/main.pdf from LaTeX source.
# Run from the project root: bash scripts/compile_paper.sh

set -e

PAPER_DIR="$(dirname "$0")/../paper"
cd "$PAPER_DIR"

echo "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== Pass 2: bibtex ==="
bibtex main

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
