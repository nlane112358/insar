"""Persisted workflow state and step transitions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from services.platform.storage.database import get_db


WORKFLOW_STEPS = (
    "DiscoverPass",
    "FetchAuxData",
    "RunCompute",
    "ValidateAndScore",
    "PublishSTAC",
    "DispatchWebhooks",
)
StepStatus = Literal["queued", "running", "completed", "failed", "skipped"]


def mark_step(
    workflow_id: str,
    step_name: str,
    status: StepStatus,
    *,
    detail: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if step_name not in WORKFLOW_STEPS:
        raise ValueError(f"Unknown workflow step: {step_name}")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started_at = now if status == "running" else None
    finished_at = now if status in {"completed", "failed", "skipped"} else None
    with get_db() as conn:
        conn.execute(
            """INSERT INTO workflow_steps
               (workflow_id, step_name, status, started_at, finished_at, detail_json, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (workflow_id, step_name) DO UPDATE SET
                   status = excluded.status,
                   started_at = CASE WHEN excluded.status = 'running' THEN excluded.started_at
                                     ELSE workflow_steps.started_at END,
                   finished_at = excluded.finished_at,
                   detail_json = excluded.detail_json,
                   error_message = excluded.error_message""",
            (workflow_id, step_name, status, started_at, finished_at, json.dumps(detail or {}), error),
        )
        conn.commit()


def get_steps(workflow_id: str) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT step_name, status, started_at, finished_at, detail_json, error_message "
            "FROM workflow_steps WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchall()
    results = [
        {**dict(row), "detail": json.loads(row["detail_json"])}
        for row in rows
    ]
    order = {step: index for index, step in enumerate(WORKFLOW_STEPS)}
    return sorted(results, key=lambda row: order[row["step_name"]])
