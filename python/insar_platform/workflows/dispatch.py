"""Transactional outbox for reliable API-to-worker handoff."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from insar_platform.storage.database import get_db


def celery_enabled() -> bool:
    return os.getenv("INSAR_TASK_MODE", "inline").lower() == "celery"


def add_outbox_message(conn: Any, task_name: str, payload: dict[str, Any], created_at: str) -> str:
    outbox_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO workflow_outbox (outbox_id, task_name, payload_json, created_at)
           VALUES (?, ?, ?, ?)""",
        (outbox_id, task_name, json.dumps(payload), created_at),
    )
    return outbox_id


def dispatch_pending(limit: int = 100) -> int:
    from insar_platform.workflows.celery_app import celery_app

    published = 0
    with get_db() as conn:
        lock_clause = " FOR UPDATE SKIP LOCKED" if conn.dialect == "postgresql" else ""
        rows = conn.execute(
            """SELECT outbox_id, task_name, payload_json FROM workflow_outbox
               WHERE published_at IS NULL ORDER BY created_at LIMIT ?""" + lock_clause,
            (limit,),
        ).fetchall()
        for row in rows:
            try:
                celery_app.send_task(
                    row["task_name"],
                    args=[json.loads(row["payload_json"])],
                    task_id=row["outbox_id"],
                )
                conn.execute(
                    """UPDATE workflow_outbox SET published_at = ?, attempts = attempts + 1,
                       last_error = NULL WHERE outbox_id = ?""",
                    (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), row["outbox_id"]),
                )
                published += 1
            except Exception as exc:
                conn.execute(
                    "UPDATE workflow_outbox SET attempts = attempts + 1, last_error = ? WHERE outbox_id = ?",
                    (str(exc)[:2000], row["outbox_id"]),
                )
        conn.commit()
    return published
