"""Typed stack synchronization service backed by the existing domain algorithm."""

from __future__ import annotations

from typing import Any


def sync_stack(config: Any) -> Any:
    from scripts.stack_timeseries import sync_stack as sync_legacy_stack

    return sync_legacy_stack(config)
