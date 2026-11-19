"""Development webhook signature verification endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json

from fastapi import APIRouter, Header, HTTPException, Request

from insar_platform.api.deps import get_webhook_secret

router = APIRouter()


@router.post("/v1/dev/webhook-sink")
async def dev_webhook_sink(
    request: Request,
    x_insar_signature: str = Header(..., alias="X-InSAR-Signature"),
):
    """Built-in local receiver to verify HMAC-SHA256 webhook signatures."""
    raw_body = await request.body()
    expected = "sha256=" + hmac.new(
        get_webhook_secret().encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(x_insar_signature, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = json.loads(raw_body)
    target = event.get("job_id") or event.get("stack_id")
    print(f"\n[WEBHOOK VERIFIED] {event['event']} | target={target} | stac={event.get('stac_item_url')}\n")
    return {"received": True, "signature_valid": True}
