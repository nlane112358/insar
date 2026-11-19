"""Celery configuration for durable, restartable compute workflows."""

from __future__ import annotations

import os

from celery import Celery


broker_url = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")
visibility_timeout = int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "172800"))

celery_app = Celery(
    "insar_platform",
    broker=broker_url,
    backend=result_backend,
    include=["services.worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": visibility_timeout},
    result_backend_transport_options={"visibility_timeout": visibility_timeout},
    task_publish_retry=True,
    task_publish_retry_policy={"max_retries": 8, "interval_start": 0.2, "interval_step": 0.5, "interval_max": 5.0},
    result_expires=86400,
    timezone="UTC",
    beat_schedule={
        "dispatch-transactional-outbox": {
            "task": "services.worker.tasks.dispatch_outbox",
            "schedule": 3.0,
        }
    },
)
