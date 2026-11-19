"""Signed webhook delivery with an idempotency ledger and durable retry path."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from insar_platform.storage.database import get_db


def event_key(payload: dict[str, Any]) -> str:
    identity = payload.get("job_id") or f"{payload.get('stack_id', '')}:{payload.get('update_id', '')}"
    if not identity.strip(":"):
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{payload.get('event', 'unknown')}:{identity}"


def enqueue_delivery(
    webhook_url: str,
    payload: dict[str, Any],
    *,
    secret: str,
    outbox_writer,
) -> str:
    key = event_key(payload)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    delivery_id = hashlib.sha256(key.encode()).hexdigest()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT status FROM webhook_deliveries WHERE event_key = ?",
            (key,),
        ).fetchone()
        if existing and existing["status"] == "delivered":
            return "delivered (duplicate suppressed)"
        if existing and existing["status"] in {"queued", "sending"}:
            return "queued"
        conn.execute(
            """INSERT INTO webhook_deliveries
               (delivery_id, event_key, event_type, target_url, status, attempts, created_at, payload_json)
               VALUES (?, ?, ?, ?, 'queued', 0, ?, ?)
               ON CONFLICT (event_key) DO UPDATE SET target_url = excluded.target_url,
                   status = 'queued', error_message = NULL, payload_json = excluded.payload_json""",
            (delivery_id, key, str(payload.get("event", "unknown")), webhook_url, now, json.dumps(payload)),
        )
        outbox_writer(
            conn,
            "insar_platform.worker.tasks.deliver_webhook",
            {"delivery_id": delivery_id},
            now,
        )
        conn.commit()
    return "queued"


def deliver_delivery(delivery_id: str, secret: str) -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM webhook_deliveries WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Webhook delivery not found: {delivery_id}")
        if row["status"] == "delivered":
            return "delivered (duplicate suppressed)"
        conn.execute(
            "UPDATE webhook_deliveries SET status = 'sending', attempts = attempts + 1 WHERE delivery_id = ?",
            (delivery_id,),
        )
        conn.commit()

    payload = json.loads(row["payload_json"])
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-InSAR-Signature": f"sha256={signature}",
        "X-InSAR-Event": payload["event"],
        "Idempotency-Key": row["event_key"],
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(row["target_url"], content=body, headers=headers)
        if response.status_code >= 500:
            response.raise_for_status()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with get_db() as conn:
            conn.execute(
                """UPDATE webhook_deliveries SET status = ?, response_code = ?, delivered_at = ?, error_message = NULL
                   WHERE delivery_id = ?""",
                ("delivered" if response.is_success else "failed", response.status_code, now if response.is_success else None, delivery_id),
            )
            conn.commit()
        if not response.is_success:
            raise RuntimeError(f"Webhook returned HTTP {response.status_code}")
        return f"delivered ({response.status_code})"
    except Exception as exc:
        with get_db() as conn:
            conn.execute(
                "UPDATE webhook_deliveries SET status = 'failed', error_message = ? WHERE delivery_id = ?",
                (str(exc)[:2000], delivery_id),
            )
            conn.commit()
        raise


def send_inline(webhook_url: str, payload: dict[str, Any], secret: str) -> str:
    """Compatibility sender for local single-process mode."""
    key = event_key(payload)
    delivery_id = hashlib.sha256(key.encode()).hexdigest()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db() as conn:
        existing = conn.execute("SELECT status FROM webhook_deliveries WHERE event_key = ?", (key,)).fetchone()
        if existing and existing["status"] == "delivered":
            return "delivered (duplicate suppressed)"
        conn.execute(
            """INSERT INTO webhook_deliveries
               (delivery_id, event_key, event_type, target_url, status, attempts, created_at, payload_json)
               VALUES (?, ?, ?, ?, 'queued', 0, ?, ?)
               ON CONFLICT (event_key) DO UPDATE SET status = 'sending', attempts = webhook_deliveries.attempts + 1,
                   target_url = excluded.target_url, payload_json = excluded.payload_json""",
            (delivery_id, key, str(payload.get("event", "unknown")), webhook_url, now, json.dumps(payload)),
        )
        conn.commit()
    try:
        return deliver_delivery(delivery_id, secret)
    except Exception as exc:
        return f"failed ({exc})"


def dispatch_event(webhook_url: str, payload: dict[str, Any]) -> str:
    secret = os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret")
    if os.getenv("INSAR_TASK_MODE", "inline").lower() == "celery":
        from insar_platform.workflows.dispatch import add_outbox_message

        return enqueue_delivery(webhook_url, payload, secret=secret, outbox_writer=add_outbox_message)
    return send_inline(webhook_url, payload, secret)
