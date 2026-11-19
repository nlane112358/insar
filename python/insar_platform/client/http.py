"""Thin synchronous HTTP client wrapping the InSAR Platform Control API."""

from __future__ import annotations

import time
from typing import Any

import httpx


class InsarClient:
    """Synchronous client for job submission, stack updates, and STAC lookups."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "insar_dev_secret_key",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def __enter__(self) -> InsarClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/v1/jobs", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()

    def wait_job(self, job_id: str, poll_s: float = 0.5) -> dict[str, Any]:
        while True:
            status = self.get_job(job_id)
            if status["status"] in ("completed", "failed"):
                return status
            time.sleep(poll_s)

    def get_stac_item(self, job_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/stac/items/{job_id}")
        resp.raise_for_status()
        return resp.json()

    def update_stack(self, stack_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(f"/v1/stacks/{stack_id}/updates", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_stack(self, stack_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/v1/stacks/{stack_id}")
        resp.raise_for_status()
        return resp.json()

    def wait_stack(self, stack_id: str, poll_s: float = 0.5) -> dict[str, Any]:
        while True:
            status = self.get_stack(stack_id)
            if status["status"] in ("completed", "failed"):
                return status
            time.sleep(poll_s)

    def get_stack_stac_item(self, stack_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/stac/stacks/{stack_id}")
        resp.raise_for_status()
        return resp.json()
