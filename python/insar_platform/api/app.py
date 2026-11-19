"""FastAPI application factory assembling middleware and route routers."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request

from insar_platform.api.deps import get_data_dir
from insar_platform.api.routes import assets, health, jobs, monitors, stac, stacks, webhooks
from insar_platform.storage.database import get_db, init_db


def create_app() -> FastAPI:
    """Create and configure the InSAR Platform Control API FastAPI application."""
    get_data_dir().mkdir(parents=True, exist_ok=True)
    init_db()

    application = FastAPI(title="InSAR Platform Control API", version="0.5.0")

    @application.middleware("http")
    async def record_api_usage(request: Request, call_next):
        """Record route-level request counts and latency without retaining API keys."""
        if request.url.path == "/health":
            return await call_next(request)
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        route = getattr(request.scope.get("route"), "path", request.url.path)
        supplied_key = request.headers.get("x-api-key", "public")
        key_hash = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO api_usage
                       (request_id, api_key_hash, route, method, status_code, latency_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex, key_hash, route, request.method, response.status_code, latency_ms, now),
                )
                conn.commit()
        except Exception:
            # A metering write must not make an otherwise healthy API request fail.
            pass
        return response

    for module in (health, monitors, jobs, stacks, stac, assets, webhooks):
        application.include_router(module.router)

    return application


app = create_app()
