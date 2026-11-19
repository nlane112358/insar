"""Health and metering routes."""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request

from insar_platform.api.deps import get_engine_bin, get_stack_bin, verify_api_key
from insar_platform.storage.database import get_db
from insar_platform.workflows.dispatch import celery_enabled

router = APIRouter()


@router.get("/health")
def health(request: Request):
    with get_db() as conn:
        conn.execute("SELECT 1").fetchone()
    remote_compute = celery_enabled()
    engine_bin = get_engine_bin()
    stack_bin = get_stack_bin()
    return {
        "status": "ok",
        "version": request.app.version,
        "database": "postgresql" if os.getenv("INSAR_DATABASE_URL", "").startswith(("postgres://", "postgresql://")) else "sqlite",
        "task_mode": os.getenv("INSAR_TASK_MODE", "inline"),
        "storage": os.getenv("INSAR_STORAGE_BACKEND", "filesystem"),
        "engine_bin": str(engine_bin),
        "compute_mode": "remote_worker" if remote_compute else "local",
        "engine_ready": None if remote_compute else engine_bin.is_file(),
        "stack_bin": str(stack_bin),
        "stack_ready": None if remote_compute else stack_bin.is_file(),
    }


@router.get("/v1/admin/usage")
def api_usage(date_from: Optional[str] = None, _: str = Depends(verify_api_key)):
    query = """SELECT api_key_hash, route, method, status_code, COUNT(*) AS request_count,
                      AVG(latency_ms) AS average_latency_ms, MAX(created_at) AS last_seen
               FROM api_usage"""
    params: tuple[Any, ...] = ()
    if date_from:
        query += " WHERE created_at >= ?"
        params = (date_from,)
    query += " GROUP BY api_key_hash, route, method, status_code ORDER BY last_seen DESC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"usage": [dict(row) for row in rows]}
