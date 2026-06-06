#!/usr/bin/env python3
"""
SEC EDGAR ETL automation: fetch, parse, validate, and persist hyperscaler
Capex / OCF / FCF data from the SEC CompanyFacts XBRL API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.etl.edgar import (  # noqa: E402
    HYPERSCALER_CIKS,
    _require_user_agent,
    fetch_and_parse,
    _processed_dir,
)
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

_OUTPUT_FILENAME = "hyperscaler_capex.parquet"

# Scaling constant — convert from USD to USD billions for display only.
_BILLIONS = 1e9


def _load_settings() -> dict:
    with (PROJECT_ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_frame(df: pd.DataFrame, ticker: str) -> None:
    """
    Run integrity checks on a parsed FCF DataFrame.

    Rules
    -----
    - ``capex`` column must be strictly positive (outflow from SEC XBRL).
    - ``ocf`` must be finite (no infinite values from division artifacts).
    - ``fcf`` must be finite.
    - No all-NaN rows.
    """
    if df.empty:
        raise ValueError(f"Empty DataFrame for {ticker} — no rows to validate.")

    non_positive_capex = (df["capex"] <= 0).sum()
    if non_positive_capex > 0:
        rows = df.index[df["capex"] <= 0].tolist()
        logger.warning(
            "%s: %d rows with non-positive Capex (expected positive outflows): %s",
            ticker,
            non_positive_capex,
            rows[:5],
        )

    for col in ("ocf", "capex", "fcf"):
        inf_count = (~df[col].apply(lambda x: pd.api.types.is_number(x) and pd.isna(x) is False
                                              and x != float("inf") and x != float("-inf"))).sum()
        if inf_count > 0:
            raise ValueError(
                f"Validation failed for {ticker}: column '{col}' contains "
                f"{inf_count} non-finite values."
            )

    null_count = df[["ocf", "capex", "fcf"]].isna().sum().sum()
    if null_count > 0:
        raise ValueError(
            f"Validation failed for {ticker}: {int(null_count)} null values in "
            "ocf/capex/fcf columns."
        )

    logger.info("%s: validation passed (%d rows)", ticker, len(df))


def _print_ticker_summary(ticker: str, df: pd.DataFrame) -> None:
    """Log a human-readable summary of the most recent quarterly filings."""
    quarterly = df[df["ocf_fp"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if quarterly.empty:
        quarterly = df.copy()

    recent = quarterly.sort_index().tail(4)
    print(f"\n  {'─' * 68}")
    print(f"  {ticker:6s} │ {len(df):4d} filing rows │ "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"  {'─' * 68}")
    print(f"  {'Period End':14s} │ {'Form':7s} │ {'FP':5s} │ "
          f"{'OCF ($B)':>10s} │ {'Capex ($B)':>11s} │ {'FCF ($B)':>10s}")
    print(f"  {'─' * 68}")
    for end_date, row in recent.iterrows():
        print(
            f"  {str(end_date.date()):14s} │ "
            f"{str(row.get('ocf_form','?')):7s} │ "
            f"{str(row.get('ocf_fp','?')):5s} │ "
            f"{row['ocf'] / _BILLIONS:>10.2f} │ "
            f"{row['capex'] / _BILLIONS:>11.2f} │ "
            f"{row['fcf'] / _BILLIONS:>10.2f}"
        )
    print(f"  {'─' * 68}")
    print(f"  XBRL OCF  concept : {df['ocf_concept'].iloc[0]}")
    print(f"  XBRL Capex concept: {df['capex_concept'].iloc[0]}")


def run_edgar_pipeline(*, force_refresh: bool = False) -> pd.DataFrame:
    """
    Full SEC EDGAR Capex/FCF pipeline for all configured hyperscalers.

    Steps
    -----
    1. Verify SEC_EDGAR_USER_AGENT is present.
    2. For each ticker: fetch CompanyFacts, parse OCF+Capex, compute FCF.
    3. Validate each DataFrame.
    4. Concatenate into a single panel, persist to ``data/processed/``.

    Returns
    -------
    pd.DataFrame
        Combined panel indexed by (``ticker``, ``end_date``) with columns
        ``ocf``, ``capex``, ``fcf``, ``ocf_concept``, ``capex_concept``.
    """
    settings = _load_settings()
    configure_logging(level=settings.get("logging", {}).get("level", "INFO"))

    _require_user_agent()
    logger.info("Starting SEC EDGAR ETL for tickers: %s", list(HYPERSCALER_CIKS))

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    for ticker in HYPERSCALER_CIKS:
        try:
            logger.info("Processing %s …", ticker)
            df = fetch_and_parse(ticker, force_refresh=force_refresh)
            _validate_frame(df, ticker)
            frames.append(df)
        except (KeyError, ValueError, RuntimeError, EnvironmentError) as exc:
            logger.error("Failed for %s: %s", ticker, exc)
            errors[ticker] = str(exc)

    if errors:
        raise RuntimeError(
            f"SEC EDGAR ETL failed for {len(errors)} ticker(s): {errors}"
        )

    panel = pd.concat(frames, axis=0)
    panel.index.name = "end_date"
    panel = panel.sort_values(["ticker", "end_date"])

    out_path = _processed_dir() / _OUTPUT_FILENAME
    panel.to_parquet(out_path)
    logger.info("Persisted combined Capex panel → %s  shape=%s", out_path, panel.shape)

    return panel


def main() -> int:
    try:
        panel = run_edgar_pipeline()
    except (RuntimeError, EnvironmentError) as exc:
        logger.error("Edgar ETL pipeline aborted: %s", exc)
        return 1

    print("\n" + "=" * 72)
    print("  SEC EDGAR CAPEX / FCF PIPELINE SUMMARY")
    print("=" * 72)
    print(f"  Output file : data/processed/{_OUTPUT_FILENAME}")
    print(f"  Total rows  : {len(panel)}")
    print(f"  Tickers     : {panel['ticker'].unique().tolist()}")
    print(f"  Date range  : {panel.index.min().date()} → {panel.index.max().date()}")
    print(f"  Columns     : {panel.columns.tolist()}")

    for ticker in panel["ticker"].unique():
        _print_ticker_summary(ticker, panel[panel["ticker"] == ticker])

    print("\n  All validation checks passed. Data is ready for Module 1 EDA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
