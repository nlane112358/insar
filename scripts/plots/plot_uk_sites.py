#!/usr/bin/env python3
"""Forward-only CLI shim for `insar_platform.viz.uk_sites`."""

from __future__ import annotations

import sys
from pathlib import Path

_PY_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from insar_platform.viz.uk_sites import *  # noqa: E402,F401,F403
from insar_platform.viz.uk_sites import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
