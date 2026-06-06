"""SEC EDGAR CompanyFacts XBRL ingestion, Capex/OCF parsing, and FCF calculation."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.etl.cache import get_rate_limit, read_or_fetch, with_rate_limit
from src.utils.logging import get_logger

_logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CONFIG = _PROJECT_ROOT / "config" / "paths.yaml"
_SEC_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# Static CIK map — zero-padded to 10 digits as required by the EDGAR API.
HYPERSCALER_CIKS: dict[str, str] = {
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
}

# XBRL concept resolution order (primary first, then fallbacks).
_OCF_CONCEPTS: tuple[str, ...] = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "NetCashProvidedByOperatingActivities",
)

_CAPEX_CONCEPTS: tuple[str, ...] = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "CapitalExpendituresIncurredButNotYetPaid",
)

_VALID_FORMS: frozenset[str] = frozenset({"10-K", "10-Q", "10-Q/A"})

# Period-length windows (days from `start` to `end`) for standalone filings.
# YTD 10-Q cumulations (H1 ≈180d, 9-mo ≈270d) are intentionally excluded.
_QUARTERLY_DAYS_RANGE: tuple[int, int] = (60, 110)   # standalone fiscal quarter
_ANNUAL_DAYS_RANGE: tuple[int, int] = (330, 380)     # full fiscal year (10-K)


# ---------------------------------------------------------------------------
# Environment & configuration helpers
# ---------------------------------------------------------------------------


def _require_user_agent() -> str:
    """
    Load and return the SEC_EDGAR_USER_AGENT from the environment.

    The SEC EDGAR Fair Access Policy requires every automated request to include
    a valid ``User-Agent`` header of the form ``Name email@domain.com``.

    Raises
    ------
    EnvironmentError
        When the variable is absent or blank in the environment.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    agent = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    if not agent:
        raise EnvironmentError(
            "SEC_EDGAR_USER_AGENT is not set. "
            "Copy .env.example to .env and supply a valid 'Name email@domain.com' string. "
            "The SEC will block requests that omit this header."
        )
    return agent


def _load_paths_config() -> dict[str, Any]:
    with _PATHS_CONFIG.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in {_PATHS_CONFIG}")
    return cfg


def _processed_dir() -> Path:
    cfg = _load_paths_config()
    path = _PROJECT_ROOT / cfg["data"]["processed"]
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# SEC API fetcher
# ---------------------------------------------------------------------------


@with_rate_limit(get_rate_limit("sec_edgar"))
def _http_get_json(url: str, user_agent: str) -> dict[str, Any]:
    """Perform a rate-limited HTTP GET, returning parsed JSON."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            import gzip as _gzip
            raw = resp.read()
            if resp.info().get("Content-Encoding") == "gzip":
                raw = _gzip.decompress(raw)
            import json as _json
            return _json.loads(raw)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"SEC EDGAR request failed (HTTP {exc.code}) for URL: {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"SEC EDGAR network error for URL: {url} — {exc.reason}"
        ) from exc


def fetch_company_facts(ticker: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """
    Retrieve the full CompanyFacts XBRL payload for a hyperscaler ticker.

    Uses the cache-aside pattern from ``cache.py``; the raw JSON is stored under
    ``data/cache/sec_edgar/<ticker>_companyfacts.json.gz``.

    Parameters
    ----------
    ticker:
        Hyperscaler symbol — must be a key in ``HYPERSCALER_CIKS``.
    force_refresh:
        Bypass the local cache and re-fetch from the SEC API.

    Returns
    -------
    dict
        Full CompanyFacts JSON payload as returned by the SEC EDGAR API.
    """
    ticker = ticker.upper()
    cik = HYPERSCALER_CIKS.get(ticker)
    if cik is None:
        raise ValueError(
            f"Ticker '{ticker}' not in HYPERSCALER_CIKS. "
            f"Known tickers: {list(HYPERSCALER_CIKS)}"
        )

    user_agent = _require_user_agent()
    url = f"{_SEC_BASE_URL}/CIK{cik}.json"
    cache_key = f"{ticker}_companyfacts"

    def _fetch() -> dict[str, Any]:
        _logger.info("Fetching SEC CompanyFacts for %s (CIK %s)", ticker, cik)
        return _http_get_json(url, user_agent)

    result = read_or_fetch("sec_edgar", cache_key, "json", _fetch, force_refresh=force_refresh)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict from cache for {ticker} facts, got {type(result)}")
    return result


# ---------------------------------------------------------------------------
# XBRL parsing & FCF calculation
# ---------------------------------------------------------------------------


def _is_standalone_period(entry: dict[str, Any]) -> bool:
    """
    Return True when an XBRL entry represents a standalone fiscal period.

    Rules (applied per filing form type):
    - **10-K**: accept only annual periods (~330–380 days).
    - **10-Q / 10-Q/A**: accept only standalone fiscal quarters (~60–110 days).
      - Reject YTD accumulations (H1 ≈180d, 9-month ≈272d).
      - Reject rolling trailing-twelve-month windows filed in 10-Qs (≈364d) —
        Amazon reports TTM values inside 10-Q filings, which would otherwise
        pass the annual-range check if form type were ignored.
    """
    start = entry.get("start")
    end = entry.get("end")
    form = entry.get("form", "")

    if not start or not end:
        return True  # no period info — keep and let de-dup resolve

    import datetime
    days = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days
    lo_q, hi_q = _QUARTERLY_DAYS_RANGE
    lo_a, hi_a = _ANNUAL_DAYS_RANGE

    if form == "10-K":
        return lo_a <= days <= hi_a
    # 10-Q and 10-Q/A: standalone quarters only
    return lo_q <= days <= hi_q


def _extract_concept_entries(
    usgaap: dict[str, Any],
    concept: str,
    ticker: str,
) -> list[dict[str, Any]]:
    """Return filtered rows from a single XBRL concept, or [] if absent/empty."""
    if concept not in usgaap:
        return []
    usd_entries = usgaap[concept].get("units", {}).get("USD", [])
    rows = []
    for entry in usd_entries:
        if entry.get("form", "") not in _VALID_FORMS:
            continue
        if not _is_standalone_period(entry):
            continue
        rows.append({
            "end_date": pd.Timestamp(entry["end"]),
            "val": entry["val"],
            "filed": pd.Timestamp(entry["filed"]),
            "form": entry["form"],
            "fp": entry.get("fp", ""),
            "concept_used": concept,
        })
    return rows


def _extract_concept_series(
    usgaap: dict[str, Any],
    concepts: tuple[str, ...],
    ticker: str,
    label: str,
    *,
    combine_all: bool = False,
) -> pd.DataFrame:
    """
    Extract USD values from XBRL concepts, optionally combining multiple concepts.

    When ``combine_all=True``, rows from every matching concept are merged and
    de-duplicated by ``end_date`` (most-recently-filed wins).  This is required
    for AMZN Capex, which spans two separate XBRL concepts across time.

    When ``combine_all=False`` (default), the first non-empty concept wins.

    Returns a DataFrame with columns ``end_date``, ``val``, ``filed``, ``form``,
    ``fp``, and ``concept_used``. Raises ``KeyError`` if no concept has data.
    """
    if combine_all:
        # Build per-concept DataFrames in priority order, then stitch:
        # for each end_date, the highest-priority concept that has an entry wins.
        # This prevents lower-priority supplemental concepts (e.g.
        # CapitalExpendituresIncurredButNotYetPaid, a non-cash accrual) from
        # overwriting the authoritative cash-payment concept.
        concept_frames: list[pd.DataFrame] = []
        concepts_used: list[str] = []
        for concept in concepts:
            rows = _extract_concept_entries(usgaap, concept, ticker)
            if rows:
                concept_frames.append(pd.DataFrame(rows))
                concepts_used.append(concept)

        if not concept_frames:
            raise KeyError(
                f"No matching XBRL concept found for {label} in {ticker}. "
                f"Tried (in order): {list(concepts)}"
            )

        # Start with the lowest-priority frame then overwrite with higher-priority
        # frames so the primary concept always wins where coverage overlaps.
        combined = concept_frames[-1].copy()
        for frame in reversed(concept_frames[:-1]):
            # Overwrite rows: index on end_date, higher-priority frame takes precedence
            combined = combined.set_index("end_date")
            frame_idx = frame.set_index("end_date")
            combined = frame_idx.combine_first(combined).reset_index()

        _logger.info("%s: combined concepts for %s: %s", ticker, label, concepts_used)
        return combined

    for concept in concepts:
        rows = _extract_concept_entries(usgaap, concept, ticker)
        if not rows:
            _logger.debug("%s: concept %s yielded no standalone rows, trying fallback", ticker, concept)
            continue
        _logger.info("%s: using XBRL concept '%s' for %s", ticker, concept, label)
        return pd.DataFrame(rows)

    raise KeyError(
        f"No matching XBRL concept found for {label} in {ticker}. "
        f"Tried (in order): {list(concepts)}"
    )


def _deduplicate_by_end_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve overlapping filings by keeping the most recently filed value per end date.

    The SEC often re-reports the same period in amended filings (10-Q/A) or when a
    company restates. We keep the latest ``filed`` date per ``end_date``.
    """
    return (
        df.sort_values("filed", ascending=True)
        .drop_duplicates(subset=["end_date"], keep="last")
        .set_index("end_date")
        .sort_index()
    )


def parse_capex_and_fcf(facts_json: dict[str, Any], ticker: str) -> pd.DataFrame:
    """
    Parse Capex, OCF, and FCF from a raw CompanyFacts payload.

    Workflow
    --------
    1. Navigate to ``facts["us-gaap"]``.
    2. Resolve OCF via ``_OCF_CONCEPTS`` priority order.
    3. Resolve Capex via ``_CAPEX_CONCEPTS`` priority order.
    4. Filter to ``10-K`` / ``10-Q`` forms only.
    5. De-duplicate by ``end_date`` (keep most recently filed).
    6. Merge OCF and Capex on ``end_date``.
    7. Compute ``FCF = OCF - Capex``.

    Parameters
    ----------
    facts_json:
        Raw dict returned by ``fetch_company_facts``.
    ticker:
        Ticker symbol for logging context.

    Returns
    -------
    pd.DataFrame
        Indexed by ``end_date`` (period-end). Columns: ``ocf``, ``capex``, ``fcf``,
        ``ocf_concept``, ``capex_concept``, ``ocf_form``, ``capex_form``.
    """
    ticker = ticker.upper()

    try:
        usgaap = facts_json["facts"]["us-gaap"]
    except KeyError as exc:
        raise KeyError(
            f"Unexpected CompanyFacts structure for {ticker}: missing key {exc}. "
            "The SEC may have changed the payload schema."
        ) from exc

    try:
        # OCF: standard priority-order fallback (no stitching needed)
        ocf_raw = _extract_concept_series(usgaap, _OCF_CONCEPTS, ticker, "OCF")
    except KeyError as exc:
        raise KeyError(f"OCF extraction failed for {ticker}: {exc}") from exc

    try:
        # Capex: combine all matching concepts chronologically — AMZN switches
        # from PaymentsToAcquirePropertyPlantAndEquipment (→2017) to
        # PaymentsToAcquireProductiveAssets (2016→) mid-history.
        capex_raw = _extract_concept_series(
            usgaap, _CAPEX_CONCEPTS, ticker, "Capex", combine_all=True
        )
    except KeyError as exc:
        raise KeyError(f"Capex extraction failed for {ticker}: {exc}") from exc

    ocf_dedup = _deduplicate_by_end_date(ocf_raw)
    capex_dedup = _deduplicate_by_end_date(capex_raw)

    # Retain metadata columns before merging.
    ocf_final = ocf_dedup[["val", "form", "fp", "concept_used"]].rename(
        columns={"val": "ocf", "form": "ocf_form", "fp": "ocf_fp", "concept_used": "ocf_concept"}
    )
    capex_final = capex_dedup[["val", "form", "fp", "concept_used"]].rename(
        columns={"val": "capex", "form": "capex_form", "fp": "capex_fp", "concept_used": "capex_concept"}
    )

    merged = ocf_final.join(capex_final, how="inner")

    if merged.empty:
        raise ValueError(
            f"No overlapping end_date entries for OCF and Capex in {ticker}. "
            "Cannot compute FCF."
        )

    # Capex in SEC XBRL is stored as a positive outflow. FCF = OCF - Capex.
    merged["fcf"] = merged["ocf"] - merged["capex"]
    merged["ticker"] = ticker

    _logger.info(
        "%s: parsed %d OCF rows, %d Capex rows, %d merged (FCF) rows",
        ticker,
        len(ocf_dedup),
        len(capex_dedup),
        len(merged),
    )
    return merged[["ticker", "ocf", "capex", "fcf", "ocf_concept", "capex_concept",
                   "ocf_form", "capex_form", "ocf_fp", "capex_fp"]]


# ---------------------------------------------------------------------------
# Convenience: fetch + parse in one call
# ---------------------------------------------------------------------------


def fetch_and_parse(ticker: str, *, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch raw CompanyFacts and return a parsed FCF DataFrame for a single ticker.

    Parameters
    ----------
    ticker:
        Hyperscaler symbol.
    force_refresh:
        Re-fetch from the SEC API, bypassing all local cache.
    """
    facts = fetch_company_facts(ticker, force_refresh=force_refresh)
    return parse_capex_and_fcf(facts, ticker)
