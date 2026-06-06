"""Local filesystem cache engine with API rate-limit compliance."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, overload

import pandas as pd
import yaml

from src.utils.logging import abort_if_unsafe_staging, get_logger

P = ParamSpec("P")
R = TypeVar("R")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PATHS_CONFIG = _PROJECT_ROOT / "config" / "paths.yaml"
_SETTINGS_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"

_VALID_EXTENSIONS = frozenset({"parquet", "json", "csv", "txt"})
_SAFE_KEY_PATTERN = re.compile(r"^[\w.\-]+$")
_SAFE_SOURCE_PATTERN = re.compile(r"^[\w\-]+$")

_logger = get_logger(__name__)

# Enforce no staged secrets/data before any cache I/O.
abort_if_unsafe_staging(_PROJECT_ROOT)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return loaded


def _resolve_path(relative: str) -> Path:
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def get_cache_root() -> Path:
    """Return the configured cache directory, creating it if needed."""
    cfg = _load_yaml(_PATHS_CONFIG)
    cache_dir = _resolve_path(str(cfg["data"]["cache"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sanitize_source(source: str) -> str:
    source = source.strip().lower()
    if not _SAFE_SOURCE_PATTERN.fullmatch(source):
        raise ValueError(
            f"Invalid cache source '{source}'. "
            "Use alphanumeric characters, underscores, or hyphens."
        )
    return source


def _sanitize_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise ValueError("Cache key cannot be empty.")
    if _SAFE_KEY_PATTERN.fullmatch(key):
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    _logger.debug("Hashing non-alphanumeric cache key to %s", digest)
    return digest


def _sanitize_extension(extension: str) -> str:
    extension = extension.lstrip(".").lower().strip()
    if extension not in _VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported cache extension '{extension}'. "
            f"Allowed: {sorted(_VALID_EXTENSIONS)}"
        )
    return extension


def get_cache_path(source: str, key: str, extension: str) -> Path:
    """
    Map a data query to a deterministic path under ``data/cache/<source>/``.

    Parameters
    ----------
    source:
        Logical data provider namespace (e.g. ``yfinance``, ``sec_edgar``).
    key:
        Stable identifier for the query. Non-alphanumeric keys are hashed.
    extension:
        File format suffix without dot (``parquet``, ``json``, ``csv``, ``txt``).
    """
    safe_source = _sanitize_source(source)
    safe_key = _sanitize_key(key)
    safe_ext = _sanitize_extension(extension)
    path = get_cache_root() / safe_source / f"{safe_key}.{safe_ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _RateLimiter:
    """Thread-safe minimum-interval gate for API calls."""

    def __init__(self, max_calls_per_sec: float) -> None:
        if max_calls_per_sec <= 0:
            raise ValueError("max_calls_per_sec must be positive.")
        self._min_interval = 1.0 / max_calls_per_sec
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_limiter_registry: dict[float, _RateLimiter] = {}
_registry_lock = threading.Lock()


def _get_limiter(max_calls_per_sec: float) -> _RateLimiter:
    with _registry_lock:
        limiter = _limiter_registry.get(max_calls_per_sec)
        if limiter is None:
            limiter = _RateLimiter(max_calls_per_sec)
            _limiter_registry[max_calls_per_sec] = limiter
        return limiter


def with_rate_limit(max_calls_per_sec: float) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that enforces a maximum call rate (calls per second).

    SEC EDGAR allows at most 10 requests/sec; use ``with_rate_limit(10)``.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        limiter = _get_limiter(max_calls_per_sec)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            limiter.wait()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_rate_limit(source: str) -> float:
    """Load per-source rate limit from ``config/settings.yaml``."""
    cfg = _load_yaml(_SETTINGS_CONFIG)
    limits = cfg.get("etl", {}).get("rate_limits", {})
    return float(limits.get(source, limits.get("default", 5.0)))


def _resolve_cache_path(path: Path) -> Path | None:
    """Return the on-disk cache path, checking gzip-compressed JSON siblings."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    gzip_json = path.with_suffix(path.suffix + ".gz")
    if gzip_json.is_file() and gzip_json.stat().st_size > 0:
        return gzip_json
    return None


def cache_exists(path: Path) -> bool:
    """Return True when a cache artifact exists and is non-empty."""
    return _resolve_cache_path(path) is not None


@overload
def save_to_cache(data: pd.DataFrame, path: Path, **kwargs: Any) -> None: ...


@overload
def save_to_cache(data: pd.Series, path: Path, **kwargs: Any) -> None: ...


@overload
def save_to_cache(data: dict[str, Any] | list[Any], path: Path, **kwargs: Any) -> None: ...


@overload
def save_to_cache(data: str, path: Path, **kwargs: Any) -> None: ...


def save_to_cache(data: Any, path: Path, **kwargs: Any) -> None:
    """
    Persist an object to disk using a format inferred from the file extension.

    - ``DataFrame`` / ``Series`` -> Parquet (pyarrow)
    - ``dict`` / ``list`` -> JSON (optionally gzip-compressed)
    - ``str`` -> plain text
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lstrip(".").lower()

    settings = _load_yaml(_SETTINGS_CONFIG)
    cache_cfg = settings.get("etl", {}).get("cache", {})
    json_compression = cache_cfg.get("json_compression")
    csv_compression = cache_cfg.get("csv_compression", "gzip")

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=kwargs.pop("index", True), **kwargs)
    elif isinstance(data, pd.Series):
        data.to_frame(name=data.name or "value").to_parquet(path, **kwargs)
    elif isinstance(data, (dict, list)):
        if extension != "json":
            raise TypeError("JSON-serializable objects must use a .json cache path.")
        payload = json.dumps(data, default=str, indent=2)
        write_path = path
        if json_compression == "gzip":
            import gzip

            write_path = path.with_suffix(path.suffix + ".gz")
            with gzip.open(write_path, "wt", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            path.write_text(payload, encoding="utf-8")
        _logger.info("Cache saved: %s", write_path)
        return
    elif isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        raise TypeError(f"Unsupported cache payload type: {type(data).__name__}")

    _logger.info("Cache saved: %s", path)


def load_from_cache(path: Path) -> Any:
    """
    Load a cache artifact based on file extension.

    Returns
    -------
    Any
        ``DataFrame``, ``dict``, ``list``, or ``str`` depending on format.
    """
    path = Path(path)
    resolved = _resolve_cache_path(path)
    if resolved is None:
        raise FileNotFoundError(f"Cache file not found: {path}")
    path = resolved

    extension = path.suffix.lstrip(".").lower()
    if path.name.endswith(".json.gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    if extension == "parquet":
        return pd.read_parquet(path)
    if extension == "json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if extension == "csv":
        return pd.read_csv(path, index_col=kwargs_index_col(path))
    if extension == "txt":
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Cannot load unsupported cache extension: {extension}")


def kwargs_index_col(path: Path) -> int | None:
    """Heuristic: treat first CSV column as index when it is unnamed."""
    preview = pd.read_csv(path, nrows=1)
    first = preview.columns[0]
    if first in ("", "Unnamed: 0", "index", "date", "Date"):
        return 0
    return None


def read_or_fetch(
    source: str,
    key: str,
    extension: str,
    fetcher: Callable[[], Any],
    *,
    force_refresh: bool = False,
) -> Any:
    """
    Return cached data when present; otherwise fetch, persist, and return.

    Parameters
    ----------
    fetcher:
        Zero-argument callable that retrieves fresh data from an API.
    """
    path = get_cache_path(source, key, extension)
    if not force_refresh and cache_exists(path):
        _logger.debug("Cache hit: %s", path)
        return load_from_cache(path)

    _logger.info("Cache miss — fetching: %s/%s", source, key)
    data = fetcher()
    save_to_cache(data, path)
    return data
