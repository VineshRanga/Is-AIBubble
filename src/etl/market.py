"""Market data ingestion via yfinance with cache-aside and gap validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
import yfinance as yf

from src.etl.cache import get_cache_path, get_rate_limit, read_or_fetch, with_rate_limit
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CONFIG = _PROJECT_ROOT / "config" / "paths.yaml"
_SETTINGS_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"

Era = Literal["dotcom", "ai_cycle", "full"]
_CACHE_SOURCE = "yfinance"
_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

# US equity market closures spanning more than three consecutive trading days.
_KNOWN_EXTENDED_CLOSURES: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (
    (pd.Timestamp("2001-09-11"), pd.Timestamp("2001-09-14")),  # 9/11 attacks
    (pd.Timestamp("2012-10-29"), pd.Timestamp("2012-10-30")),  # Hurricane Sandy
)


class ValidationError(Exception):
    """Raised when market data fails quality or continuity checks."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return loaded


def _load_paths_config() -> dict[str, Any]:
    return _load_yaml(_PATHS_CONFIG)


def _load_settings_config() -> dict[str, Any]:
    return _load_yaml(_SETTINGS_CONFIG)


def get_era_date_range(era: Era) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve analysis window boundaries for a historical era.

    Parameters
    ----------
    era:
        ``dotcom`` (1995–2002), ``ai_cycle`` (2020–2026), or ``full`` master range.
    """
    paths_cfg = _load_paths_config()
    analysis = paths_cfg["analysis"]

    if era == "dotcom":
        window = analysis["dotcom_comparison"]
        start, end = window["start"], window["end"]
    elif era == "ai_cycle":
        window = analysis["ai_cycle"]
        start, end = window["start"], window["end"]
    else:
        start, end = analysis["start_date"], analysis["end_date"]

    return pd.Timestamp(start), pd.Timestamp(end)


def _cache_key(ticker: str, era: Era) -> str:
    start, end = get_era_date_range(era)
    safe_ticker = ticker.replace("^", "idx_")
    return f"{safe_ticker}_{era}_{start.date()}_{end.date()}"


@with_rate_limit(get_rate_limit("yfinance"))
def _download_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rate-limited yfinance download for a single ticker."""
    _logger.debug("Downloading %s from %s to %s", ticker, start.date(), end.date())
    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise ValidationError(
            f"No market data returned for ticker '{ticker}' "
            f"between {start.date()} and {end.date()}."
        )
    return _normalize_price_frame(raw, ticker)


def _normalize_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance multi-index columns and enforce a DatetimeIndex."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame.index = pd.to_datetime(frame.index)
    frame.index = frame.index.normalize()
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    missing_cols = [col for col in _PRICE_COLUMNS if col not in frame.columns]
    if missing_cols:
        raise ValidationError(
            f"Ticker '{ticker}' is missing expected price columns: {missing_cols}"
        )

    return frame.loc[:, list(_PRICE_COLUMNS)]


def _missing_days_between(prev: pd.Timestamp, curr: pd.Timestamp) -> pd.DatetimeIndex:
    """Return US business days strictly between two consecutive observations."""
    return pd.bdate_range(
        start=prev + pd.Timedelta(days=1),
        end=curr - pd.Timedelta(days=1),
    )


def _is_known_closure_gap(missing_days: pd.DatetimeIndex) -> bool:
    """Return True when all missing days fall inside a registered market closure."""
    if len(missing_days) == 0:
        return True
    for start, end in _KNOWN_EXTENDED_CLOSURES:
        if missing_days.min() >= start and missing_days.max() <= end:
            return True
    return False


def _max_consecutive_missing_trading_days(index: pd.DatetimeIndex) -> int:
    """Count the largest run of unexpected missing US business days between observations."""
    if len(index) < 2:
        return 0

    sorted_idx = index.sort_values()
    max_gap = 0
    for i in range(1, len(sorted_idx)):
        prev, curr = sorted_idx[i - 1], sorted_idx[i]
        between = _missing_days_between(prev, curr)
        if _is_known_closure_gap(between):
            continue
        max_gap = max(max_gap, len(between))
    return max_gap


def _longest_unexpected_nan_run(frame: pd.DataFrame) -> int:
    """Return longest consecutive unexpected NaN streak after calendar reindex."""
    if frame.empty:
        return 0

    dense = frame.reindex(pd.bdate_range(frame.index.min(), frame.index.max()))
    mask = dense[list(_PRICE_COLUMNS)].isna().any(axis=1)
    if not mask.any():
        return 0

    max_unexpected = 0
    run_dates: list[pd.Timestamp] = []
    for date, is_nan in zip(dense.index, mask):
        if is_nan:
            run_dates.append(date)
            continue
        if run_dates:
            run_index = pd.DatetimeIndex(run_dates)
            if not _is_known_closure_gap(run_index):
                max_unexpected = max(max_unexpected, len(run_dates))
            run_dates = []

    if run_dates:
        run_index = pd.DatetimeIndex(run_dates)
        if not _is_known_closure_gap(run_index):
            max_unexpected = max(max_unexpected, len(run_dates))

    return max_unexpected


def validate_and_fill_gaps(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Enforce continuity rules on daily market data.

    Gaps of three or fewer consecutive trading days are forward-filled.
    Gaps exceeding three consecutive trading days raise ``ValidationError``.
    """
    if frame.empty:
        raise ValidationError(f"Cannot validate empty DataFrame for '{ticker}'.")

    inter_obs_gap = _max_consecutive_missing_trading_days(frame.index)
    if inter_obs_gap > 3:
        raise ValidationError(
            f"Data breach for '{ticker}': {inter_obs_gap} consecutive missing "
            f"trading days detected between observations (limit is 3)."
        )

    dense_index = pd.bdate_range(frame.index.min(), frame.index.max())
    filled = frame.reindex(dense_index)
    nan_run = _longest_unexpected_nan_run(filled)
    if nan_run > 3:
        raise ValidationError(
            f"Data breach for '{ticker}': {nan_run} consecutive missing "
            f"trading days after calendar reindex (limit is 3)."
        )

    price_cols = list(_PRICE_COLUMNS)
    filled[price_cols] = filled[price_cols].ffill()
    remaining_nulls = filled[price_cols].isna().sum().sum()
    if remaining_nulls:
        raise ValidationError(
            f"Data breach for '{ticker}': {int(remaining_nulls)} null values "
            "remain after forward-fill."
        )

    _logger.debug(
        "Validated %s: %d rows, max inter-obs gap %d",
        ticker,
        len(filled),
        inter_obs_gap,
    )
    return filled


def fetch_equity_data(
    ticker: str,
    *,
    era: Era = "ai_cycle",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull historical daily equity prices for a hyperscaler ticker.

    Parameters
    ----------
    ticker:
        Equity symbol (e.g. ``MSFT``, ``GOOGL``).
    era:
        Historical window — defaults to the AI Capex cycle (2020–2026).
    force_refresh:
        Bypass local Parquet cache and re-fetch from yfinance.

    Returns
    -------
    pd.DataFrame
        OHLCV + Adj Close indexed by trading date.
    """
    start, end = get_era_date_range(era)
    key = _cache_key(ticker, era)

    def _fetch() -> pd.DataFrame:
        raw = _download_yfinance(ticker, start, end)
        return validate_and_fill_gaps(raw, ticker)

    frame = read_or_fetch(_CACHE_SOURCE, key, "parquet", _fetch, force_refresh=force_refresh)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from cache for {ticker}, got {type(frame)}")
    return frame


def fetch_index_data(
    ticker: str,
    *,
    era: Era = "full",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull historical daily benchmark index prices.

    Parameters
    ----------
    ticker:
        Index symbol (e.g. ``^GSPC``, ``^NDX``).
    era:
        Historical window — defaults to the full analysis range (1995–2026).
    force_refresh:
        Bypass local Parquet cache and re-fetch from yfinance.

    Returns
    -------
    pd.DataFrame
        OHLCV + Adj Close indexed by trading date.
    """
    start, end = get_era_date_range(era)
    key = _cache_key(ticker, era)

    def _fetch() -> pd.DataFrame:
        raw = _download_yfinance(ticker, start, end)
        return validate_and_fill_gaps(raw, ticker)

    frame = read_or_fetch(_CACHE_SOURCE, key, "parquet", _fetch, force_refresh=force_refresh)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from cache for {ticker}, got {type(frame)}")
    return frame


def get_cache_file_path(ticker: str, era: Era, *, equity: bool = True) -> Path:
    """Return the Parquet cache path for a ticker/era combination."""
    key = _cache_key(ticker, era)
    return get_cache_path(_CACHE_SOURCE, key, "parquet")
