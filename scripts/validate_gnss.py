#!/usr/bin/env python3
"""Backward-compatible CLI shim for the platform GNSS validation service."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.platform.validation import gnss_service as _implementation  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
