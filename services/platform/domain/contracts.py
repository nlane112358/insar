"""Typed messages that cross API, workflow, and worker boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PairwiseJobTask(BaseModel):
    task_version: Literal[1] = 1
    job_id: str
    request: dict[str, Any]


class StackUpdateTask(BaseModel):
    task_version: Literal[1] = 1
    stack_id: str
    update_id: str
    monitor_name: str
    request: dict[str, Any]


class IngestMonitorTask(BaseModel):
    task_version: Literal[1] = 1
    ingestion_id: str
    monitor_name: str
    request: dict[str, Any]


class WorkflowRecord(BaseModel):
    workflow_id: str
    monitor_name: str
    granule_id: str | None = None
    engine_version: str = "0.5.0"
    state: Literal["queued", "running", "completed", "failed"] = "queued"
    current_step: str = "DiscoverPass"
    metadata: dict[str, Any] = Field(default_factory=dict)
