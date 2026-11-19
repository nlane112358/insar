"""Sequential multi-pass time-series stacking routes."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from insar_platform.api.deps import (
    get_base_url,
    get_data_dir,
    get_engine_bin,
    get_stack_bin,
    validate_safe_id,
    verify_api_key,
)
from insar_platform.domain.contracts import StackUpdateTask
from insar_platform.domain.schemas import StackUpdateRequest
from insar_platform.storage.database import get_db
from insar_platform.storage.object_store import enabled as s3_enabled
from insar_platform.workflows.dispatch import add_outbox_message, celery_enabled
from insar_platform.workflows.state import get_steps

router = APIRouter()


@router.post("/v1/stacks/{stack_id}/updates", status_code=202)
def update_stack(
    stack_id: str,
    req: StackUpdateRequest,
    _: str = Depends(verify_api_key),
):
    validate_safe_id(stack_id, "stack_id")
    base_url = get_base_url()
    monitor_name = req.monitor_name or stack_id
    identity = {
        "stack_id": stack_id,
        "monitor_name": monitor_name,
        "engine_version": req.engine_version,
        "request": req.model_dump(mode="json", exclude={"idempotency_key", "webhook_url"}),
    }
    idempotency_key = req.idempotency_key or hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with get_db() as conn:
        duplicate = conn.execute(
            "SELECT update_id, status FROM stack_updates WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    if duplicate:
        return {
            "stack_id": stack_id,
            "update_id": duplicate["update_id"],
            "status": duplicate["status"],
            "stack_url": f"{base_url}/v1/stacks/{stack_id}",
            "stac_item_url": f"{base_url}/stac/stacks/{stack_id}",
            "idempotent_replay": True,
        }

    stack_bin = get_stack_bin()
    engine_bin = get_engine_bin()
    if not celery_enabled() and not stack_bin.exists():
        raise HTTPException(status_code=500, detail=f"C++ stack binary not found at {stack_bin}")
    if req.pair_job is not None and not celery_enabled() and not engine_bin.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {engine_bin}")

    has_pair_input = bool(req.pair_job is not None or req.source_job_id or (req.disp_path and req.coh_path))
    if bool(req.disp_path) != bool(req.coh_path):
        raise HTTPException(status_code=400, detail="disp_path and coh_path must both be provided")
    if has_pair_input:
        has_t = req.t0_days is not None and req.t1_days is not None
        if not has_t and req.dt_days is None:
            raise HTTPException(status_code=400, detail="Provide dt_days or (t0_days, t1_days) when ingesting a pair")
        if has_t and req.t1_days <= req.t0_days:
            raise HTTPException(status_code=400, detail="t1_days must exceed t0_days")
    else:
        state_path = get_data_dir() / "stacks" / stack_id / "stack_state.tif"
        has_persisted_state = False
        if s3_enabled():
            with get_db() as conn:
                has_persisted_state = conn.execute(
                    "SELECT 1 FROM assets WHERE owner_type = 'stack' AND owner_id = ? AND asset_key = 'stack_state.tif'",
                    (stack_id,),
                ).fetchone() is not None
        if not state_path.exists() and not has_persisted_state:
            raise HTTPException(
                status_code=400,
                detail="Provide source_job_id, disp_path/coh_path, or pair_job to initialize a new stack",
            )

    req = req.model_copy(update={"idempotency_key": idempotency_key})
    update_id = f"upd_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        existing = conn.execute("SELECT monitor_name FROM stacks WHERE stack_id = ?", (stack_id,)).fetchone()
        if existing:
            monitor_name = req.monitor_name or existing["monitor_name"]
            conn.execute(
                """UPDATE stacks SET monitor_name = ?, status = 'queued', updated_at = ?, webhook_url = ?
                   WHERE stack_id = ?""",
                (monitor_name, now, req.webhook_url, stack_id),
            )
        else:
            conn.execute(
                """INSERT INTO stacks (stack_id, monitor_name, status, pair_count, created_at, updated_at, webhook_url)
                   VALUES (?, ?, 'queued', 0, ?, ?, ?)
                   ON CONFLICT (stack_id) DO UPDATE SET monitor_name = excluded.monitor_name,
                       status = 'queued', updated_at = excluded.updated_at, webhook_url = excluded.webhook_url""",
                (stack_id, monitor_name, now, now, req.webhook_url),
            )
        cursor = conn.execute(
            """INSERT INTO stack_updates
               (update_id, stack_id, source_job_id, status, created_at, engine_version, idempotency_key)
               VALUES (?, ?, ?, 'queued', ?, ?, ?)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (update_id, stack_id, req.source_job_id, now, req.engine_version, idempotency_key),
        )
        if cursor.rowcount != 1:
            duplicate = conn.execute(
                "SELECT update_id, status FROM stack_updates WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            conn.commit()
            return {
                "stack_id": stack_id,
                "update_id": duplicate["update_id"],
                "status": duplicate["status"],
                "stack_url": f"{base_url}/v1/stacks/{stack_id}",
                "stac_item_url": f"{base_url}/stac/stacks/{stack_id}",
                "idempotent_replay": True,
            }
        if celery_enabled():
            add_outbox_message(
                conn,
                "insar_platform.worker.tasks.run_stack_update",
                StackUpdateTask(
                    stack_id=stack_id,
                    update_id=update_id,
                    monitor_name=monitor_name,
                    request=req.model_dump(mode="json"),
                ).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if not celery_enabled():
        from insar_platform.worker.execution import execute_stack_update

        execute_stack_update(stack_id, update_id, monitor_name, req)
    return {
        "stack_id": stack_id,
        "update_id": update_id,
        "status": "queued",
        "stack_url": f"{base_url}/v1/stacks/{stack_id}",
        "stac_item_url": f"{base_url}/stac/stacks/{stack_id}",
    }


@router.get("/v1/stacks")
def list_stacks(_: str = Depends(verify_api_key)):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM stacks ORDER BY updated_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        if item["summary_json"]:
            item["summary"] = json.loads(item.pop("summary_json"))
        else:
            item.pop("summary_json", None)
        out.append(item)
    return {"stacks": out}


@router.get("/v1/stacks/{stack_id}")
def get_stack(stack_id: str, _: str = Depends(verify_api_key)):
    validate_safe_id(stack_id, "stack_id")
    base_url = get_base_url()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM stacks WHERE stack_id = ?", (stack_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Stack not found")
        updates = conn.execute(
            "SELECT update_id, source_job_id, engine_version, status, created_at, finished_at, error_message "
            "FROM stack_updates WHERE stack_id = ? ORDER BY created_at DESC LIMIT 25",
            (stack_id,),
        ).fetchall()

    res = dict(row)
    if res["summary_json"]:
        res["summary"] = json.loads(res.pop("summary_json"))
    else:
        res.pop("summary_json", None)
    if res["status"] == "completed":
        res["stac_item_url"] = f"{base_url}/stac/stacks/{stack_id}"
    res["recent_updates"] = [
        {**dict(u), "workflow_steps": get_steps(u["update_id"])}
        for u in updates
    ]
    return res
