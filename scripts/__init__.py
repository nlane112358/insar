"""Thin compatibility CLI wrappers forwarding to `insar_platform`."""

from __future__ import annotations

import sys
from pathlib import Path

_PY_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))
