"""Macroeconomic data ingestion via FRED (pandas-datareader)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas_datareader import data as pdr

from src.etl.cache import get_cache_path, get_rate_limit, read_or_fetch, with_rate_limit
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CONFIG = _PROJECT_ROOT / "config" / "paths.yaml"
_CACHE_SOURCE = "fred"


def _load_paths_config() -> dict[str, Any]:
    with _PATHS_CONFIG.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {_PATHS_CONFIG}")
    return loaded


def get_analysis_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the master analysis window from ``config/paths.yaml``."""
    cfg = _load_paths_config()
    analysis = cfg["analysis"]
    return pd.Timestamp(analysis["start_date"]), pd.Timestamp(analysis["end_date"])


def _cache_key(series_id: str) -> str:
    start, end = get_analysis_date_range()
    return f"{series_id}_{start.date()}_{end.date()}"


@with_rate_limit(get_rate_limit("fred"))
def _download_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Rate-limited FRED pull via pandas-datareader."""
    _logger.debug("Downloading FRED series %s from %s to %s", series_id, start.date(), end.date())
    frame = pdr.DataReader(series_id, "fred", start, end)
    if frame.empty:
        raise ValueError(
            f"No FRED data returned for '{series_id}' between {start.date()} and {end.date()}."
        )

    series = frame.iloc[:, 0].copy()
    series.name = series_id
    series.index = pd.to_datetime(series.index).normalize()
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def fetch_fred_series(
    series_id: str,
    *,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Pull a FRED macro indicator with cache-aside semantics.

    Parameters
    ----------
    series_id:
        FRED series code (e.g. ``FEDFUNDS``, ``M2SL``, ``STLFSI4``).
    force_refresh:
        Bypass local Parquet cache and re-fetch from FRED.

    Returns
    -------
    pd.Series
        Observation values indexed by release date. Frequency may be
        daily, weekly, or monthly depending on the series.
    """
    start, end = get_analysis_date_range()
    key = _cache_key(series_id)

    def _fetch() -> pd.Series:
        return _download_fred_series(series_id, start, end)

    result = read_or_fetch(_CACHE_SOURCE, key, "parquet", _fetch, force_refresh=force_refresh)
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            series = result.iloc[:, 0]
            series.name = series_id
            return series
        raise ValueError(f"Unexpected DataFrame shape for FRED series '{series_id}': {result.shape}")
    if isinstance(result, pd.Series):
        result.name = series_id
        return result
    raise TypeError(f"Expected Series from cache for {series_id}, got {type(result)}")


def align_macro_to_daily(
    df_macro: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align lower-frequency macro data to a daily equity trading calendar.

    Uses forward-fill only (no backward-fill) to prevent look-ahead bias.
    Handles weekly (STLFSI4) and monthly (M2SL, FEDFUNDS) release schedules.

    Parameters
    ----------
    df_macro:
        Macro features indexed by observation/release date.
    target_index:
        Daily DatetimeIndex from equity price data.

    Returns
    -------
    pd.DataFrame
        Macro columns reindexed to ``target_index`` with safe forward-fill.
    """
    if df_macro.empty:
        raise ValueError("Cannot align empty macro DataFrame.")
    if len(target_index) == 0:
        raise ValueError("Target index is empty.")

    aligned = df_macro.copy()
    aligned.index = pd.to_datetime(aligned.index).normalize()
    aligned = aligned.sort_index()

    daily_index = pd.DatetimeIndex(target_index).normalize().sort_values().unique()
    start = min(aligned.index.min(), daily_index.min())
    end = daily_index.max()

    # Extend macro index to cover the full daily window before forward-fill.
    union_index = aligned.index.union(daily_index)
    union_index = union_index[(union_index >= start) & (union_index <= end)]

    aligned = aligned.reindex(union_index)
    aligned = aligned.ffill()
    aligned = aligned.reindex(daily_index)

    remaining_nulls = aligned.isna().sum()
    if remaining_nulls.any():
        cols = remaining_nulls[remaining_nulls > 0].to_dict()
        _logger.warning(
            "Macro alignment left leading NaNs before first observation: %s", cols
        )

    _logger.debug(
        "Aligned macro data to %d daily rows (%s → %s)",
        len(aligned),
        aligned.index.min().date(),
        aligned.index.max().date(),
    )
    return aligned


def fetch_all_configured_fred_series(
    *,
    force_refresh: bool = False,
) -> dict[str, pd.Series]:
    """Fetch every FRED series declared in ``config/paths.yaml``."""
    cfg = _load_paths_config()
    fred_map: dict[str, str] = cfg["fred_series"]
    return {
        label: fetch_fred_series(series_id, force_refresh=force_refresh)
        for label, series_id in fred_map.items()
    }


def get_cache_file_path(series_id: str) -> Path:
    """Return the Parquet cache path for a FRED series."""
    return get_cache_path(_CACHE_SOURCE, _cache_key(series_id), "parquet")
