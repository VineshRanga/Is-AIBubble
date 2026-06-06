# Is-AIBubble

Quantitative research project analyzing systemic macro risk driven by AI infrastructure capital expenditures (Capex).

## Research Modules

| Module | Focus |
|--------|-------|
| **1 — The Catalyst** | Hyperscaler Capex vs. monetization gap; Dotcom telecom parallels; 2026 IPO liquidity vacuum |
| **2 — The Impact** | S&P 500 / Nasdaq 100 concentration; amplified drawdowns from macro shorts and momentum unwinds |
| **3 — The Mitigation** | OCPI, ICE GPU futures, and Kalshi prediction markets as balance-sheet pressure valves |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC EDGAR user-agent
pre-commit install
```

## Data Pipeline

- **Market data:** `yfinance`
- **SEC filings:** `sec-edgar-downloader`
- **Macro series:** `pandas-datareader` (FRED)

All ingestion is cached locally under `data/` (gitignored). Configuration lives in `config/`.

## Project Layout

```
config/          YAML pipeline settings
src/etl/         Modular ingestion + cache layer
src/features/    Feature engineering
src/models/      VaR/PFE, validation, backtesting
notebooks/       ETL → EDA → modeling by module
paper/           LaTeX research paper scaffold
tests/           Unit tests
```

## Security

- Never commit `.env` or `data/` contents.
- A pre-commit hook and runtime guard in `src/utils/logging.py` enforce this policy.
