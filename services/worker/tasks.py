"""Celery tasks executed by the stateless compute worker pool."""

from __future__ import annotations

import os
from typing import Any

from celery import shared_task

from services.platform.storage.database import get_db
from services.platform.domain.contracts import IngestMonitorTask, PairwiseJobTask, StackUpdateTask
from services.platform.domain.schemas import CreateJobRequest, StackUpdateRequest
from services.platform.publishing.webhooks import deliver_delivery
from services.platform.workflows.dispatch import dispatch_pending
from services.worker.execution import execute_insar_job, execute_stack_update


def _status(table: str, key_column: str, key: str) -> str | None:
    with get_db() as conn:
        row = conn.execute(f"SELECT status FROM {table} WHERE {key_column} = ?", (key,)).fetchone()
    return row["status"] if row else None


@shared_task(bind=True, name="services.worker.tasks.run_pairwise_job", max_retries=5)
def run_pairwise_job(self, payload: dict[str, Any]) -> dict[str, Any]:
    message = PairwiseJobTask.model_validate(payload)
    job_id = message.job_id
    if _status("jobs", "job_id", job_id) == "completed":
        return {"job_id": job_id, "status": "completed", "duplicate": True}

    try:
        request = CreateJobRequest.model_validate(message.request)
        execute_insar_job(job_id, request)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=min(300, 2 ** (self.request.retries + 1)))

    status = _status("jobs", "job_id", job_id)
    if status != "completed":
        raise self.retry(
            exc=RuntimeError(f"Pairwise job {job_id} ended with status={status}"),
            countdown=min(300, 2 ** (self.request.retries + 1)),
        )
    return {"job_id": job_id, "status": status}


@shared_task(bind=True, name="services.worker.tasks.run_stack_update", max_retries=5)
def run_stack_update(self, payload: dict[str, Any]) -> dict[str, Any]:
    message = StackUpdateTask.model_validate(payload)
    stack_id = message.stack_id
    update_id = message.update_id
    if _status("stack_updates", "update_id", update_id) == "completed":
        return {"stack_id": stack_id, "update_id": update_id, "status": "completed", "duplicate": True}

    try:
        request = StackUpdateRequest.model_validate(message.request)
        execute_stack_update(stack_id, update_id, message.monitor_name, request)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=min(300, 2 ** (self.request.retries + 1)))

    status = _status("stack_updates", "update_id", update_id)
    if status != "completed":
        raise self.retry(
            exc=RuntimeError(f"Stack update {update_id} ended with status={status}"),
            countdown=min(300, 2 ** (self.request.retries + 1)),
        )
    return {"stack_id": stack_id, "update_id": update_id, "status": status}


@shared_task(bind=True, name="services.worker.tasks.run_monitor_ingestion", max_retries=5)
def run_monitor_ingestion(self, payload: dict[str, Any]) -> dict[str, Any]:
    message = IngestMonitorTask.model_validate(payload)
    ingestion_id = message.ingestion_id
    if _status("ingestion_runs", "ingestion_id", ingestion_id) == "completed":
        return {"ingestion_id": ingestion_id, "status": "completed", "duplicate": True}

    try:
        from services.worker.execution import execute_monitor_ingestion

        execute_monitor_ingestion(ingestion_id, message.monitor_name, message.request)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=min(300, 2 ** (self.request.retries + 1)))

    status = _status("ingestion_runs", "ingestion_id", ingestion_id)
    if status != "completed":
        raise self.retry(
            exc=RuntimeError(f"Ingestion {ingestion_id} ended with status={status}"),
            countdown=min(300, 2 ** (self.request.retries + 1)),
        )
    return {"ingestion_id": ingestion_id, "status": status}


@shared_task(name="services.worker.tasks.dispatch_outbox")
def dispatch_outbox() -> int:
    return dispatch_pending()


@shared_task(bind=True, name="services.worker.tasks.deliver_webhook", max_retries=8)
def deliver_webhook(self, payload: dict[str, Any]) -> str:
    try:
        return deliver_delivery(
            str(payload["delivery_id"]),
            os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret"),
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=min(900, 5 * (2 ** self.request.retries)))
