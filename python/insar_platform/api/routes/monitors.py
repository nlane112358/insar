"""Monitor registration, scheduled ingestion sync, asset registration, and screening routes."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from insar_platform.api.deps import (
    coords_bounds,
    get_base_url,
    validate_geojson_geometry,
    validate_safe_id,
    verify_api_key,
)
from insar_platform.domain.contracts import IngestMonitorTask
from insar_platform.domain.schemas import SyncMonitorRequest
from insar_platform.storage.database import get_db
from insar_platform.workflows.dispatch import add_outbox_message, celery_enabled
from insar_platform.workflows.state import get_steps

router = APIRouter()


class RegisterMonitorRequest(BaseModel):
    monitor_name: str
    geometry: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)


class RegisterAssetRequest(BaseModel):
    asset_id: str
    name: str
    asset_type: str
    geometry: dict[str, Any]
    monitor_name: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


@router.post("/v1/monitors", status_code=201)
def register_monitor(req: RegisterMonitorRequest, _: str = Depends(verify_api_key)):
    monitor_name = validate_safe_id(req.monitor_name, "monitor_name")
    validate_geojson_geometry(req.geometry)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    geometry_json = json.dumps(req.geometry, separators=(",", ":"))
    with get_db() as conn:
        if conn.dialect == "postgresql":
            conn.execute(
                """INSERT INTO monitors (monitor_name, config_json, created_at, updated_at, footprint)
                   VALUES (?, ?, ?, ?, ST_SetSRID(ST_GeomFromGeoJSON(?), 4326))
                   ON CONFLICT (monitor_name) DO UPDATE SET config_json = EXCLUDED.config_json,
                       updated_at = EXCLUDED.updated_at, footprint = EXCLUDED.footprint""",
                (monitor_name, json.dumps(req.config), now, now, geometry_json),
            )
        else:
            conn.execute(
                """INSERT INTO monitors (monitor_name, config_json, created_at, updated_at, footprint)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (monitor_name) DO UPDATE SET config_json = excluded.config_json,
                       updated_at = excluded.updated_at, footprint = excluded.footprint""",
                (monitor_name, json.dumps(req.config), now, now, geometry_json),
            )
        conn.commit()
    return {"monitor_name": monitor_name, "geometry": req.geometry, "config": req.config}


@router.get("/v1/monitors")
def list_monitors(_: str = Depends(verify_api_key)):
    with get_db() as conn:
        if conn.dialect == "postgresql":
            rows = conn.execute(
                "SELECT monitor_name, config_json, created_at, updated_at, ST_AsGeoJSON(footprint) AS geometry FROM monitors ORDER BY monitor_name"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT monitor_name, config_json, created_at, updated_at, footprint AS geometry FROM monitors ORDER BY monitor_name"
            ).fetchall()
    return {
        "monitors": [
            {
                **dict(row),
                "config": json.loads(row["config_json"]),
                "geometry": json.loads(row["geometry"]) if row["geometry"] else None,
            }
            for row in rows
        ]
    }


@router.post("/v1/monitors/{monitor_name}/sync", status_code=202)
def sync_monitor(
    monitor_name: str,
    req: SyncMonitorRequest | None = None,
    _: str = Depends(verify_api_key),
):
    """Queue provider discovery and ingestion for a registered monitor."""
    monitor_name = validate_safe_id(monitor_name, "monitor_name")
    req = req or SyncMonitorRequest()
    base_url = get_base_url()
    with get_db() as conn:
        monitor = conn.execute(
            "SELECT config_json FROM monitors WHERE monitor_name = ?", (monitor_name,)
        ).fetchone()
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")

    site_config = json.loads(monitor["config_json"])
    identity = {
        "monitor_name": monitor_name,
        "engine_version": req.engine_version,
        "schedule_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "site_config": site_config,
    }
    idempotency_key = req.idempotency_key or hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ingestion_id = f"ing_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    task_request = {
        "site_config": site_config,
        "engine_version": req.engine_version,
        "webhook_url": req.webhook_url,
    }

    with get_db() as conn:
        existing = conn.execute(
            "SELECT ingestion_id, status FROM ingestion_runs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing and existing["status"] == "failed" and not req.idempotency_key:
            idempotency_key = hashlib.sha256(
                f"{idempotency_key}:retry:{uuid.uuid4().hex}".encode("utf-8")
            ).hexdigest()
            existing = None
        if existing:
            return {
                "ingestion_id": existing["ingestion_id"],
                "status": existing["status"],
                "status_url": f"{base_url}/v1/ingestions/{existing['ingestion_id']}",
                "idempotent_replay": True,
            }
        cursor = conn.execute(
            """INSERT INTO ingestion_runs
               (ingestion_id, monitor_name, status, request_json, engine_version, idempotency_key, created_at)
               VALUES (?, ?, 'queued', ?, ?, ?, ?)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (ingestion_id, monitor_name, json.dumps(task_request), req.engine_version, idempotency_key, now),
        )
        if cursor.rowcount != 1:
            existing = conn.execute(
                "SELECT ingestion_id, status FROM ingestion_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            conn.commit()
            return {
                "ingestion_id": existing["ingestion_id"],
                "status": existing["status"],
                "status_url": f"{base_url}/v1/ingestions/{existing['ingestion_id']}",
                "idempotent_replay": True,
            }
        if celery_enabled():
            add_outbox_message(
                conn,
                "insar_platform.worker.tasks.run_monitor_ingestion",
                IngestMonitorTask(
                    ingestion_id=ingestion_id,
                    monitor_name=monitor_name,
                    request=task_request,
                ).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if not celery_enabled():
        from insar_platform.worker.execution import execute_monitor_ingestion

        execute_monitor_ingestion(ingestion_id, monitor_name, task_request)
    return {
        "ingestion_id": ingestion_id,
        "status": "queued",
        "status_url": f"{base_url}/v1/ingestions/{ingestion_id}",
    }


@router.get("/v1/ingestions/{ingestion_id}")
def get_ingestion(ingestion_id: str, _: str = Depends(verify_api_key)):
    validate_safe_id(ingestion_id, "ingestion_id")
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_runs WHERE ingestion_id = ?", (ingestion_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    result = dict(row)
    result["request"] = json.loads(result.pop("request_json"))
    if result.get("summary_json"):
        result["summary"] = json.loads(result.pop("summary_json"))
    else:
        result.pop("summary_json", None)
    result["workflow_steps"] = get_steps(ingestion_id)
    return result


@router.post("/v1/assets", status_code=201)
def register_infrastructure_asset(req: RegisterAssetRequest, _: str = Depends(verify_api_key)):
    asset_id = validate_safe_id(req.asset_id, "asset_id")
    if req.monitor_name:
        validate_safe_id(req.monitor_name, "monitor_name")
    validate_geojson_geometry(req.geometry)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    geometry_json = json.dumps(req.geometry, separators=(",", ":"))
    with get_db() as conn:
        if conn.dialect == "postgresql":
            conn.execute(
                """INSERT INTO infrastructure_assets
                   (asset_id, monitor_name, name, asset_type, properties_json, created_at, geom)
                   VALUES (?, ?, ?, ?, ?, ?, ST_SetSRID(ST_GeomFromGeoJSON(?), 4326))
                   ON CONFLICT (asset_id) DO UPDATE SET monitor_name = EXCLUDED.monitor_name,
                       name = EXCLUDED.name, asset_type = EXCLUDED.asset_type,
                       properties_json = EXCLUDED.properties_json, geom = EXCLUDED.geom""",
                (asset_id, req.monitor_name, req.name, req.asset_type, json.dumps(req.properties), now, geometry_json),
            )
        else:
            conn.execute(
                """INSERT INTO infrastructure_assets
                   (asset_id, monitor_name, name, asset_type, properties_json, created_at, geom)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (asset_id) DO UPDATE SET monitor_name = excluded.monitor_name,
                       name = excluded.name, asset_type = excluded.asset_type,
                       properties_json = excluded.properties_json, geom = excluded.geom""",
                (asset_id, req.monitor_name, req.name, req.asset_type, json.dumps(req.properties), now, geometry_json),
            )
        conn.commit()
    return {"asset_id": asset_id, "name": req.name, "asset_type": req.asset_type}


@router.get("/v1/screen")
def screen(min_lon: float, min_lat: float, max_lon: float, max_lat: float, _: str = Depends(verify_api_key)):
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise HTTPException(status_code=422, detail="Invalid WGS84 bbox")
    bbox = (min_lon, min_lat, max_lon, max_lat)
    base_url = get_base_url()
    with get_db() as conn:
        if conn.dialect == "postgresql":
            assets = conn.execute(
                """SELECT asset_id, monitor_name, name, asset_type, properties_json,
                          ST_AsGeoJSON(geom) AS geometry
                   FROM infrastructure_assets
                   WHERE geom && ST_MakeEnvelope(?, ?, ?, ?, 4326)
                     AND ST_Intersects(geom, ST_MakeEnvelope(?, ?, ?, ?, 4326))""",
                bbox + bbox,
            ).fetchall()
            monitors = conn.execute(
                """SELECT monitor_name, config_json, ST_AsGeoJSON(footprint) AS geometry
                   FROM monitors
                   WHERE footprint && ST_MakeEnvelope(?, ?, ?, ?, 4326)
                     AND ST_Intersects(footprint, ST_MakeEnvelope(?, ?, ?, ?, 4326))""",
                bbox + bbox,
            ).fetchall()
            products = conn.execute(
                """SELECT a.owner_type, a.owner_id, a.asset_key, a.media_type,
                          j.monitor_name AS job_monitor, s.monitor_name AS stack_monitor,
                          ST_AsGeoJSON(a.footprint) AS geometry
                   FROM assets a
                   LEFT JOIN jobs j ON a.owner_type = 'job' AND j.job_id = a.owner_id
                   LEFT JOIN stacks s ON a.owner_type = 'stack' AND s.stack_id = a.owner_id
                   WHERE a.owner_type IN ('job', 'stack')
                     AND a.asset_key IN ('displacement_mm.tif', 'velocity_mm_yr.tif', 'cumulative_disp_mm.tif')
                     AND a.footprint && ST_MakeEnvelope(?, ?, ?, ?, 4326)
                     AND ST_Intersects(a.footprint, ST_MakeEnvelope(?, ?, ?, ?, 4326))""",
                bbox + bbox,
            ).fetchall()
            return {
                "bbox": list(bbox),
                "assets": [
                    {**dict(row), "properties": json.loads(row["properties_json"]), "geometry": json.loads(row["geometry"])}
                    for row in assets
                ],
                "monitors": [
                    {**dict(row), "config": json.loads(row["config_json"]), "geometry": json.loads(row["geometry"])}
                    for row in monitors
                ],
                "products": [
                    {
                        **dict(row),
                        "monitor_name": row["job_monitor"] or row["stack_monitor"],
                        "geometry": json.loads(row["geometry"]) if row["geometry"] else None,
                        "href": (
                            f"{base_url}/assets/{row['owner_id']}/{row['asset_key']}"
                            if row["owner_type"] == "job"
                            else f"{base_url}/stack-assets/{row['owner_id']}/{row['asset_key']}"
                        ),
                    }
                    for row in products
                ],
            }

        asset_rows = conn.execute("SELECT * FROM infrastructure_assets").fetchall()
        monitor_rows = conn.execute("SELECT * FROM monitors").fetchall()
        product_rows = conn.execute("SELECT * FROM assets").fetchall()

    def overlaps(geometry_json: str | None) -> bool:
        if not geometry_json:
            return False
        bounds = coords_bounds(json.loads(geometry_json).get("coordinates", []))
        return bool(bounds and bounds[0] <= max_lon and bounds[2] >= min_lon and bounds[1] <= max_lat and bounds[3] >= min_lat)

    return {
        "bbox": list(bbox),
        "assets": [
            {
                "asset_id": row["asset_id"],
                "monitor_name": row["monitor_name"],
                "name": row["name"],
                "asset_type": row["asset_type"],
                "properties": json.loads(row["properties_json"]),
                "geometry": json.loads(row["geom"]),
            }
            for row in asset_rows
            if overlaps(row["geom"])
        ],
        "monitors": [
            {
                "monitor_name": row["monitor_name"],
                "config": json.loads(row["config_json"]),
                "geometry": json.loads(row["footprint"]),
            }
            for row in monitor_rows
            if overlaps(row["footprint"])
        ],
        "products": [
            {
                "owner_type": row["owner_type"],
                "owner_id": row["owner_id"],
                "asset_key": row["asset_key"],
                "media_type": row["media_type"],
                "geometry": json.loads(row["footprint"]) if row["footprint"] else None,
                "href": (
                    f"{base_url}/assets/{row['owner_id']}/{row['asset_key']}"
                    if row["owner_type"] == "job"
                    else f"{base_url}/stack-assets/{row['owner_id']}/{row['asset_key']}"
                ),
            }
            for row in product_rows
            if row["owner_type"] in {"job", "stack"}
            and row["asset_key"] in {"displacement_mm.tif", "velocity_mm_yr.tif", "cumulative_disp_mm.tif"}
            and overlaps(row["footprint"])
        ],
    }
