#!/usr/bin/env python3
"""Pre-commit hook: abort if .env or data/ files are staged."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import UnsafeStagingError, verify_git_staging_safety  # noqa: E402


def main() -> int:
    try:
        verify_git_staging_safety(PROJECT_ROOT)
    except UnsafeStagingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARNING: staging safety check skipped: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
