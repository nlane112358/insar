"""Pairwise InSAR compute job and interactive preview routes."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse

from insar_platform.api.deps import get_base_url, get_engine_bin, validate_safe_id, verify_api_key
from insar_platform.domain.contracts import PairwiseJobTask
from insar_platform.domain.schemas import CreateJobRequest
from insar_platform.storage.database import get_db
from insar_platform.workflows.dispatch import add_outbox_message, celery_enabled
from insar_platform.workflows.state import get_steps

router = APIRouter()
_PREVIEW_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "job_preview.html"


def _render_job_preview(job_id: str, initial_status: str, monitor_name: str) -> str:
    """Build the preview shell; job outputs are fetched through the existing asset route."""
    template = _PREVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__JOB_ID_JSON__", json.dumps(job_id))
        .replace("__JOB_ID__", job_id)
        .replace("__INITIAL_STATUS_JSON__", json.dumps(initial_status))
        .replace("__INITIAL_STATUS__", initial_status)
        .replace("__MONITOR_NAME__", monitor_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    )


@router.post("/v1/jobs", status_code=202)
def create_job(
    req: CreateJobRequest,
    _: str = Depends(verify_api_key),
):
    if not req.slc1_path or not req.slc2_path:
        raise HTTPException(status_code=400, detail="slc1_path and slc2_path are required")
    engine_bin = get_engine_bin()
    if not celery_enabled() and not engine_bin.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {engine_bin}")

    base_url = get_base_url()
    idempotency_key = req.idempotency_key
    if not idempotency_key:
        identity = (
            {"monitor_name": req.monitor_name, "granule_id": req.granule_id, "engine_version": req.engine_version}
            if req.granule_id
            else req.model_dump(mode="json", exclude={"webhook_url", "idempotency_key"})
        )
        idempotency_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        existing = conn.execute(
            "SELECT job_id, status FROM jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing:
            job_id = existing["job_id"]
            return {
                "job_id": job_id,
                "status": existing["status"],
                "status_url": f"{base_url}/v1/jobs/{job_id}",
                "preview_url": f"{base_url}/preview/{job_id}",
                "stac_item_url": f"{base_url}/stac/items/{job_id}",
                "idempotent_replay": True,
            }
        cursor = conn.execute(
            """INSERT INTO jobs
               (job_id, monitor_name, mode, status, created_at, webhook_url, engine_version, idempotency_key)
               VALUES (?, ?, 'local', 'queued', ?, ?, ?, ?)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (job_id, req.monitor_name, now, req.webhook_url, req.engine_version, idempotency_key),
        )
        inserted = cursor.rowcount == 1
        if not inserted:
            existing = conn.execute(
                "SELECT job_id, status FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                conn.commit()
                job_id = existing["job_id"]
                return {
                    "job_id": job_id,
                    "status": existing["status"],
                    "status_url": f"{base_url}/v1/jobs/{job_id}",
                    "preview_url": f"{base_url}/preview/{job_id}",
                    "stac_item_url": f"{base_url}/stac/items/{job_id}",
                    "idempotent_replay": True,
                }
        if inserted and celery_enabled():
            add_outbox_message(
                conn,
                "insar_platform.worker.tasks.run_pairwise_job",
                PairwiseJobTask(job_id=job_id, request=req.model_dump(mode="json")).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if inserted and not celery_enabled():
        from insar_platform.worker.execution import execute_insar_job

        execute_insar_job(job_id, req)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"{base_url}/v1/jobs/{job_id}",
        "preview_url": f"{base_url}/preview/{job_id}",
        "stac_item_url": f"{base_url}/stac/items/{job_id}",
    }


@router.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _: str = Depends(verify_api_key)):
    validate_safe_id(job_id, "job_id")
    base_url = get_base_url()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    res = dict(row)
    if res["summary_json"]:
        res["summary"] = json.loads(res.pop("summary_json"))
    else:
        res.pop("summary_json", None)
    if res["status"] == "completed":
        res["stac_item_url"] = f"{base_url}/stac/items/{job_id}"
    res["preview_url"] = f"{base_url}/preview/{job_id}"
    res["workflow_steps"] = get_steps(job_id)
    return res


@router.get("/preview/{job_id}", response_class=HTMLResponse, include_in_schema=True)
def preview_job(job_id: str):
    """Serve a live, lightweight map preview for a queued or completed job."""
    validate_safe_id(job_id, "job_id")
    with get_db() as conn:
        row = conn.execute("SELECT status, monitor_name FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    html = _render_job_preview(job_id, row["status"], row["monitor_name"])
    return HTMLResponse(content=html)
