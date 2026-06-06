#!/usr/bin/env python3
"""End-to-end smoke test for market and macro ETL pipelines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.macro import (  # noqa: E402
    align_macro_to_daily,
    fetch_fred_series,
    get_cache_file_path as macro_cache_path,
)
from src.etl.market import (  # noqa: E402
    ValidationError,
    fetch_equity_data,
    fetch_index_data,
    get_cache_file_path as market_cache_path,
)
from src.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Summary metrics for a single ETL retrieval."""

    name: str
    source: str
    cache_path: Path
    rows: int
    columns: int
    start_date: str
    end_date: str
    latest_observation: str
    null_count: int
    fetched_at: str


def _load_configs() -> tuple[dict, dict]:
    paths_cfg_path = PROJECT_ROOT / "config" / "paths.yaml"
    settings_cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    with paths_cfg_path.open(encoding="utf-8") as handle:
        paths_cfg = yaml.safe_load(handle)
    with settings_cfg_path.open(encoding="utf-8") as handle:
        settings_cfg = yaml.safe_load(handle)
    return paths_cfg, settings_cfg


def _summarize_frame(name: str, source: str, cache_path: Path, frame: pd.DataFrame) -> PipelineResult:
    null_count = int(frame.isna().sum().sum())
    if null_count > 0:
        raise ValidationError(
            f"Null check failed for '{name}': {null_count} missing values remain."
        )
    return PipelineResult(
        name=name,
        source=source,
        cache_path=cache_path,
        rows=frame.shape[0],
        columns=frame.shape[1],
        start_date=str(frame.index.min().date()),
        end_date=str(frame.index.max().date()),
        latest_observation=str(frame.index.max().date()),
        null_count=null_count,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def _summarize_series(name: str, source: str, cache_path: Path, series: pd.Series) -> PipelineResult:
    null_count = int(series.isna().sum())
    if null_count > 0:
        raise ValidationError(
            f"Null check failed for '{name}': {null_count} missing values remain."
        )
    return PipelineResult(
        name=name,
        source=source,
        cache_path=cache_path,
        rows=len(series),
        columns=1,
        start_date=str(series.index.min().date()),
        end_date=str(series.index.max().date()),
        latest_observation=str(series.index.max().date()),
        null_count=null_count,
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def _log_result(result: PipelineResult) -> None:
    logger.info(
        "%s | source=%s | path=%s | shape=%dx%d | dates=%s→%s | "
        "latest=%s | nulls=%d | fetched_at=%s",
        result.name,
        result.source,
        result.cache_path,
        result.rows,
        result.columns,
        result.start_date,
        result.end_date,
        result.latest_observation,
        result.null_count,
        result.fetched_at,
    )


def run_pipeline() -> list[PipelineResult]:
    """
    Execute cache-aside retrieval for all configured equities, indices, and FRED series.

    Returns
    -------
    list[PipelineResult]
        Summary records for each successfully ingested dataset.
    """
    paths_cfg, settings_cfg = _load_configs()
    log_level = settings_cfg.get("logging", {}).get("level", "INFO")
    configure_logging(level=log_level)

    logger.info("Starting market & macro ETL smoke test")
    logger.info(
        "Config loaded — hyperscalers=%s indices=%s fred=%s",
        paths_cfg["tickers"]["hyperscalers"],
        paths_cfg["tickers"]["indices"],
        list(paths_cfg["fred_series"].values()),
    )

    results: list[PipelineResult] = []
    equity_frames: dict[str, pd.DataFrame] = {}

    for ticker in paths_cfg["tickers"]["hyperscalers"]:
        logger.info("Fetching equity: %s", ticker)
        frame = fetch_equity_data(ticker)
        equity_frames[ticker] = frame
        result = _summarize_frame(
            name=f"equity:{ticker}",
            source="yfinance",
            cache_path=market_cache_path(ticker, "ai_cycle", equity=True),
            frame=frame,
        )
        results.append(result)
        _log_result(result)

    reference_index: pd.DatetimeIndex | None = None
    for ticker in paths_cfg["tickers"]["indices"]:
        logger.info("Fetching index: %s", ticker)
        frame = fetch_index_data(ticker)
        if reference_index is None:
            reference_index = frame.index
        result = _summarize_frame(
            name=f"index:{ticker}",
            source="yfinance",
            cache_path=market_cache_path(ticker, "full", equity=False),
            frame=frame,
        )
        results.append(result)
        _log_result(result)

    if reference_index is None:
        raise RuntimeError("No index data retrieved; cannot align macro series.")

    macro_frames: dict[str, pd.DataFrame] = {}
    for label, series_id in paths_cfg["fred_series"].items():
        logger.info("Fetching FRED: %s (%s)", label, series_id)
        series = fetch_fred_series(series_id)
        raw_result = _summarize_series(
            name=f"fred:{series_id}",
            source="fred",
            cache_path=macro_cache_path(series_id),
            series=series,
        )
        results.append(raw_result)
        _log_result(raw_result)

        aligned = align_macro_to_daily(series.to_frame(name=series_id), reference_index)
        leading_nulls = int(aligned.isna().sum().sum())
        if leading_nulls > 0:
            aligned = aligned.ffill().dropna()
        if aligned.isna().sum().sum() > 0:
            raise ValidationError(
                f"Aligned macro '{series_id}' still contains nulls after forward-fill."
            )
        macro_frames[label] = aligned
        align_result = _summarize_frame(
            name=f"fred_aligned:{series_id}",
            source="fred_aligned",
            cache_path=macro_cache_path(series_id),
            frame=aligned,
        )
        results.append(align_result)
        _log_result(align_result)

    logger.info(
        "ETL complete — %d datasets ingested, %d macro alignments produced",
        len(results) - len(macro_frames),
        len(macro_frames),
    )
    return results


def main() -> int:
    try:
        results = run_pipeline()
    except (ValidationError, ValueError, RuntimeError) as exc:
        logger.error("ETL pipeline failed: %s", exc)
        return 1

    print("\n=== Market & Macro ETL Summary ===")
    for result in results:
        print(
            f"  {result.name:30s} | {result.rows:5d} x {result.columns:<2d} | "
            f"{result.start_date} → {result.end_date} | nulls={result.null_count}"
        )
    print(f"\nTotal datasets: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
