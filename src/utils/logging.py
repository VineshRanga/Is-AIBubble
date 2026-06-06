"""Logging setup and git staging safety guards."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_STAGED_PREFIXES: tuple[str, ...] = (
    ".env",
    "data/",
)

FORBIDDEN_STAGED_EXACT: frozenset[str] = frozenset({".env"})


class UnsafeStagingError(RuntimeError):
    """Raised when forbidden files are staged for commit."""


def _resolve_repo_root(repo_root: Path | None = None) -> Path:
    root = repo_root or _PROJECT_ROOT
    if not (root / ".git").exists():
        raise FileNotFoundError(f"Git repository not found at {root}")
    return root


def _staged_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to inspect staged files: {stderr or 'unknown git error'}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_forbidden_staged_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized == ".env.example":
        return False
    if normalized in FORBIDDEN_STAGED_EXACT:
        return True
    if normalized.startswith(".env."):
        return True
    return any(normalized.startswith(prefix) for prefix in FORBIDDEN_STAGED_PREFIXES)


def find_forbidden_staged_files(repo_root: Path | None = None) -> list[str]:
    """Return staged paths that violate the no-secrets/no-data policy."""
    root = _resolve_repo_root(repo_root)
    return [path for path in _staged_paths(root) if _is_forbidden_staged_path(path)]


def verify_git_staging_safety(repo_root: Path | None = None) -> None:
    """
    Abort execution if `.env` or any file under `data/` is staged for commit.

    Call at pipeline startup and from pre-commit hooks.
    """
    violations = find_forbidden_staged_files(repo_root)
    if not violations:
        return

    message = (
        "SECURITY ABORT: forbidden files are staged for commit:\n"
        + "\n".join(f"  - {path}" for path in violations)
        + "\n\nUnstage these files before continuing. "
        "Never commit `.env` or `data/` contents."
    )
    raise UnsafeStagingError(message)


def abort_if_unsafe_staging(repo_root: Path | None = None) -> None:
    """Log and exit the process when forbidden files are staged."""
    try:
        verify_git_staging_safety(repo_root)
    except UnsafeStagingError as exc:
        logging.getLogger(__name__).critical(str(exc))
        sys.exit(1)
    except (FileNotFoundError, RuntimeError) as exc:
        logging.getLogger(__name__).warning("Skipping staging safety check: %s", exc)


def configure_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """Configure root logger once with a consistent format."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, applying default configuration if needed."""
    configure_logging()
    return logging.getLogger(name)
