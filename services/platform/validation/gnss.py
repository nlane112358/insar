"""GNSS station discovery and validation service boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.platform.storage.database import get_db, init_db


def validate_stack(stack_dir: Path, **options: Any) -> Any:
    monitor_name = str(options.pop("monitor_name", stack_dir.name))
    job_id = options.pop("job_id", None)
    from scripts.validate_gnss import validate_stack_against_gnss

    report = validate_stack_against_gnss(stack_dir=stack_dir, **options)
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    if metrics:
        measured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metric_id = hashlib.sha256(f"gnss:{monitor_name}:{job_id or stack_dir}:{measured_at}".encode()).hexdigest()
        init_db()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO validation_metrics (metric_id, monitor_name, job_id, measured_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (metric_id) DO NOTHING""",
                (metric_id, monitor_name, job_id, measured_at, json.dumps(metrics)),
            )
            conn.commit()
    return report
