# Academic Writing Guidelines: AI Infrastructure Bubble Research Paper

## Document Target

- **Submission venue:** arXiv preprint (q-fin.RM / q-fin.ST / cs.CE)
- **Target length:** 18–24 pages (including appendices and references)
- **LaTeX build:** `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- **Figure DPI:** All figures must be generated at 300 DPI (enforced by `set_paper_style(dpi=300)`)

---

## Tri-Tiered Accessibility Framework

The paper must simultaneously serve three audiences. Each section should be written
with a specific audience as the primary reader while remaining accessible to the others.

### Tier 1: Abstract and Introduction
**Primary audience:** Non-technical macro-economic observers, journalists, policy-makers, portfolio managers without a quantitative background.

**Writing standards:**
- Explain the "Compute Cliff" in plain language: "AI companies are spending more than twice what they earn in free cash flow on data centres. This is structurally identical to what telecom companies did in the late 1990s before the Dotcom crash."
- Avoid unexplained acronyms. When introducing VaR, write: "Value at Risk (VaR)—a standard measure of how much money could be lost on a bad day."
- All monetary figures must be expressed in billions of dollars with full context (e.g., "a $293 billion annual capital expenditure—equivalent to the annual GDP of Finland").
- Every bold quantitative claim in the abstract must be backed by a citation or by reference to a specific notebook output.
- Keep paragraphs short (3–4 sentences maximum). No footnotes in the introduction.

**Prohibited language:**
- Mathematical notation in abstract (exception: dollar signs for monetary values)
- "We find that..." without immediate quantitative qualification
- Unqualified superlatives ("largest ever") without citation

### Tier 2: Methodology and Results Sections
**Primary audience:** Quantitative analysts, risk managers, data scientists, academic economists.

**Writing standards:**
- All equations must be numbered and referenced in text (`Equation~\ref{eq:cf_var}`).
- The Cornish-Fisher expansion (Appendix B) must be derived from first principles:
  define the cumulant generating function, explain why the Gaussian approximation
  fails for fat-tailed distributions, then introduce the correction terms.
- TimeSeriesSplit logic: explicitly state that each fold's validation data ends strictly
  before the next fold's training data starts—no temporal overlap, no data snooping.
- The IS/OOS split boundary (December 31, 2021) must be justified: it was chosen to
  isolate the 2022 tech drawdown as a pristine verification set, preventing any
  post-shock data from contaminating the calibration.
- When reporting Kupiec or Christoffersen test results, always report: LR statistic,
  p-value, degrees of freedom, and whether H0 is rejected at the 5% level.

**Required mathematical content per section:**
- Section 1 (Catalyst): Capex CAGR formula; Monetisation Runway calculation; FCF definition
- Section 2 (Impact): HHI formula; VaR definitions (historical, CF); PFE definition;
  liquidity elasticity coefficient `β_liq`; Kupiec LR; Christoffersen LR
- Section 4 (Limitations): Quantified uncertainty ranges for key assumptions

### Tier 3: Discussion and Mitigation Section
**Primary audience:** Semi-technical industry professionals—traders, risk officers, sell-side analysts, institutional investors.

**Writing standards:**
- Focus on actionable implications: what should a CRO do differently because of this research?
- Express hedging strategies in terms of standard instruments (puts, VIX spreads, sector rotations)—not in terms of abstract portfolio optimisation theory.
- Avoid academic passive voice: "We recommend..." not "It would be beneficial if..."
- Include explicit thresholds and trigger levels for risk monitoring
  (e.g., "increase defensive positioning when NDX top-5 concentration exceeds 70%").
- The compute derivatives discussion should reference real markets where they could trade
  (CME, CBOE, OTC) rather than remaining purely theoretical.

---

## Citation Standards

**Rule:** Every empirical claim must be backed by a BibTeX citation from `references.bib`.

### Mandatory citation categories:

| Claim type | Required citation source |
|-----------|--------------------------|
| IPO valuations | Bloomberg, WSJ, FT (with URL and access date) |
| Hyperscaler Capex/FCF figures | SEC 10-K filing (by CIK) |
| Historical index prices | Yahoo Finance via yfinance (version pinned) |
| Macro indicators (Fed Funds, M2, STLFSI) | FRED series documentation |
| Statistical test methodology | Original academic paper (Kupiec 1995, Christoffersen 1998) |
| Index concentration methodology | S&P/Nasdaq published methodology documents |

### Citation enforcement:
- Before adding any numeric claim (valuation, percentage, dollar figure) to any `.tex` file,
  confirm the corresponding `\citep{}` key exists in `references.bib`.
- If the source is a computational result from this project (notebook output), cite the
  GitHub repository URL and the specific notebook filename in a footnote.

---

## Methodology Limitations Documentation

The limitations section (`04_limitations.tex`) must explicitly address:

1. **Private valuation proxies:**
   - Quantify the uncertainty range: ±50% on the $210B float drain estimate
   - Explain why preferred-share liquidation preferences make private round valuations
     structurally incomparable to public common equity
   - Reference the academic literature on IPO valuation correction (e.g., Bessembinder 2020)

2. **Flat FCF baseline:**
   - Provide both upside (AI revenue materialises) and downside (capex acceleration)
     deviations from the flat assumption
   - State explicitly: "We do not forecast hyperscaler revenue; we stress-test what
     happens if monetisation lags capital deployment by the historically observed 7.3-quarter runway."

3. **Correlation matrix stability:**
   - The 2022 drawdown calibration window may not capture the structural changes
     introduced by a functioning compute derivatives market
   - Specifically flag: "If hyperscalers begin delta-hedging GPU forward contracts at scale,
     the correlation between NDX realised volatility and compute spot prices would introduce
     an entirely unmodelled feedback loop into our scenario engine."

4. **Passive vehicle amplification:**
   - The index concentration amplifier `α_HHI` assumes that passive outflows occur at
     normal market depth. During a systemic stress event, market depth collapses;
     actual amplification could be 2–4× higher than the HHI-based estimate.

---

## Figure Requirements

All figures must pass these quality checks before being included in the paper:

- **Background:** Pure white (`facecolor='white'`, `savefig.facecolor='white'`)
- **DPI:** 300 (enforced by `set_paper_style(dpi=300)`)
- **Font:** DejaVu Sans (LaTeX-compatible, no missing glyph warnings)
- **Dimensions:** 6.5 inches wide (single-column) or 13 inches wide (double-column)
- **Caption:** Every figure must have a `\caption{}` that is self-contained—a reader
  who looks only at the figure and caption should understand the finding
- **File naming convention:** `fig{N}_{short_description}.png` in `paper/figures/`

---

## Build Instructions

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For a quick syntax check without bibliography resolution:
```bash
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "Error|Warning|Overfull"
```

Expected warnings (acceptable):
- `Overfull \hbox` for wide tables (use `\resizebox` if severe)
- Missing bibliography items during first pass (resolve with `bibtex`)

Errors that must be fixed before submission:
- Undefined references (`\ref{...}` or `\citep{...}`)
- Missing `\end{...}` for any environment
- Duplicate label warnings
