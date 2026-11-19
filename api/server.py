import hashlib
import hmac
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from services.platform.storage.database import get_db, init_db
from services.platform.storage.object_store import enabled as s3_enabled
from services.platform.storage.object_store import (
    get_object as get_s3_object,
    head_object as head_s3_object,
    iter_body as iter_s3_body,
)
from services.platform.domain.contracts import IngestMonitorTask, PairwiseJobTask, StackUpdateTask
from services.platform.domain.schemas import AlertRules, CreateJobRequest, StackUpdateRequest, SyncMonitorRequest
from services.platform.workflows.state import get_steps
from services.platform.workflows.dispatch import add_outbox_message, celery_enabled

# ============================================================================
# Configuration
# ============================================================================

ENGINE_BIN = Path(os.getenv("INSAR_ENGINE_BIN", "./build/insar_proto"))
STACK_BIN = Path(os.getenv("INSAR_STACK_BIN", "./build/insar_stack"))
DATA_DIR = Path(os.getenv("INSAR_DATA_DIR", "./api_data"))
DB_PATH = DATA_DIR / "oem_jobs.db"
API_KEY = os.getenv("INSAR_API_KEY", "insar_dev_secret_key")
WEBHOOK_SECRET = os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret")
BASE_URL = os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

DATA_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="InSAR Platform Control API", version="0.5.0")


init_db()


@app.middleware("http")
async def record_api_usage(request: Request, call_next):
    """Record route-level request counts and latency without retaining API keys."""
    if request.url.path == "/health":
        return await call_next(request)
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    route = getattr(request.scope.get("route"), "path", request.url.path)
    supplied_key = request.headers.get("x-api-key", "public")
    key_hash = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO api_usage
                   (request_id, api_key_hash, route, method, status_code, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (uuid.uuid4().hex, key_hash, route, request.method, response.status_code, latency_ms, now),
            )
            conn.commit()
    except Exception:
        # A metering write must not make an otherwise healthy API request fail.
        pass
    return response


# ============================================================================
# Auth & Request Models
# ============================================================================

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return x_api_key


def _validate_safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value) or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return value


class RegisterMonitorRequest(BaseModel):
    monitor_name: str
    geometry: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)


class RegisterAssetRequest(BaseModel):
    asset_id: str
    name: str
    asset_type: str
    geometry: dict[str, Any]
    monitor_name: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


def _s3_asset_response(
    owner_type: str,
    owner_id: str,
    filename: str,
    media_type: str,
    request: Request,
) -> Response | None:
    if not s3_enabled():
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT s3_uri FROM assets WHERE owner_type = ? AND owner_id = ? AND asset_key = ?",
            (owner_type, owner_id, filename),
        ).fetchone()
    if not row:
        return None

    range_header = request.headers.get("range")
    try:
        if request.method == "HEAD":
            remote = head_s3_object(row["s3_uri"])
            headers = {"Content-Length": str(remote["ContentLength"]), "Accept-Ranges": "bytes"}
            if remote.get("ETag"):
                headers["ETag"] = remote["ETag"]
            return Response(status_code=200, media_type=media_type, headers=headers)

        remote = get_s3_object(row["s3_uri"], range_header)
        headers = {"Accept-Ranges": "bytes"}
        for source, target in (("ContentLength", "Content-Length"), ("ContentRange", "Content-Range"), ("ETag", "ETag")):
            if remote.get(source):
                headers[target] = str(remote[source])
        status = int(remote.get("ResponseMetadata", {}).get("HTTPStatusCode", 200))
        return StreamingResponse(iter_s3_body(remote["Body"]), status_code=status, media_type=media_type, headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to read stored asset: {exc}") from exc


def _read_s3_json_asset(owner_type: str, owner_id: str, filename: str) -> dict | None:
    if not s3_enabled():
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT s3_uri FROM assets WHERE owner_type = ? AND owner_id = ? AND asset_key = ?",
            (owner_type, owner_id, filename),
        ).fetchone()
    if not row:
        return None
    remote = get_s3_object(row["s3_uri"])
    try:
        return json.loads(remote["Body"].read())
    finally:
        remote["Body"].close()


# ============================================================================
# API Routes — Health & Single-Pair Jobs (`insar_proto`)
# ============================================================================

@app.get("/health")
def health():
    with get_db() as conn:
        conn.execute("SELECT 1").fetchone()
    remote_compute = celery_enabled()
    return {
        "status": "ok",
        "version": app.version,
        "database": "postgresql" if os.getenv("INSAR_DATABASE_URL", "").startswith(("postgres://", "postgresql://")) else "sqlite",
        "task_mode": os.getenv("INSAR_TASK_MODE", "inline"),
        "storage": os.getenv("INSAR_STORAGE_BACKEND", "filesystem"),
        "engine_bin": str(ENGINE_BIN),
        "compute_mode": "remote_worker" if remote_compute else "local",
        "engine_ready": None if remote_compute else ENGINE_BIN.is_file(),
        "stack_bin": str(STACK_BIN),
        "stack_ready": None if remote_compute else STACK_BIN.is_file(),
    }


@app.get("/v1/admin/usage")
def api_usage(date_from: Optional[str] = None, _: str = Depends(verify_api_key)):
    query = """SELECT api_key_hash, route, method, status_code, COUNT(*) AS request_count,
                      AVG(latency_ms) AS average_latency_ms, MAX(created_at) AS last_seen
               FROM api_usage"""
    params: tuple[Any, ...] = ()
    if date_from:
        query += " WHERE created_at >= ?"
        params = (date_from,)
    query += " GROUP BY api_key_hash, route, method, status_code ORDER BY last_seen DESC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"usage": [dict(row) for row in rows]}


def _register_geometry(geometry: dict[str, Any]) -> None:
    if geometry.get("type") not in {
        "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"
    }:
        raise HTTPException(status_code=422, detail="Unsupported GeoJSON geometry type")


@app.post("/v1/monitors", status_code=201)
def register_monitor(req: RegisterMonitorRequest, _: str = Depends(verify_api_key)):
    monitor_name = _validate_safe_id(req.monitor_name, "monitor_name")
    _register_geometry(req.geometry)
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


@app.get("/v1/monitors")
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


@app.post("/v1/monitors/{monitor_name}/sync", status_code=202)
def sync_monitor(
    monitor_name: str,
    req: SyncMonitorRequest | None = None,
    _: str = Depends(verify_api_key),
):
    """Queue provider discovery and ingestion for a registered monitor."""
    monitor_name = _validate_safe_id(monitor_name, "monitor_name")
    req = req or SyncMonitorRequest()
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
                "status_url": f"{BASE_URL}/v1/ingestions/{existing['ingestion_id']}",
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
                "status_url": f"{BASE_URL}/v1/ingestions/{existing['ingestion_id']}",
                "idempotent_replay": True,
            }
        if celery_enabled():
            add_outbox_message(
                conn,
                "services.worker.tasks.run_monitor_ingestion",
                IngestMonitorTask(
                    ingestion_id=ingestion_id,
                    monitor_name=monitor_name,
                    request=task_request,
                ).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if not celery_enabled():
        from services.worker.execution import execute_monitor_ingestion

        execute_monitor_ingestion(ingestion_id, monitor_name, task_request)
    return {
        "ingestion_id": ingestion_id,
        "status": "queued",
        "status_url": f"{BASE_URL}/v1/ingestions/{ingestion_id}",
    }


@app.get("/v1/ingestions/{ingestion_id}")
def get_ingestion(ingestion_id: str, _: str = Depends(verify_api_key)):
    _validate_safe_id(ingestion_id, "ingestion_id")
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


@app.post("/v1/assets", status_code=201)
def register_infrastructure_asset(req: RegisterAssetRequest, _: str = Depends(verify_api_key)):
    asset_id = _validate_safe_id(req.asset_id, "asset_id")
    if req.monitor_name:
        _validate_safe_id(req.monitor_name, "monitor_name")
    _register_geometry(req.geometry)
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


def _coords_bounds(value: Any) -> tuple[float, float, float, float] | None:
    points: list[tuple[float, float]] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and all(isinstance(v, (int, float)) for v in item[:2]):
            points.append((float(item[0]), float(item[1])))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    if not points:
        return None
    return min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)


@app.get("/v1/screen")
def screen(min_lon: float, min_lat: float, max_lon: float, max_lat: float, _: str = Depends(verify_api_key)):
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise HTTPException(status_code=422, detail="Invalid WGS84 bbox")
    bbox = (min_lon, min_lat, max_lon, max_lat)
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
                            f"{BASE_URL}/assets/{row['owner_id']}/{row['asset_key']}"
                            if row["owner_type"] == "job"
                            else f"{BASE_URL}/stack-assets/{row['owner_id']}/{row['asset_key']}"
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
        bounds = _coords_bounds(json.loads(geometry_json).get("coordinates", []))
        return bool(bounds and bounds[0] <= max_lon and bounds[2] >= min_lon and bounds[1] <= max_lat and bounds[3] >= min_lat)
    return {
        "bbox": list(bbox),
        "assets": [
            {"asset_id": row["asset_id"], "monitor_name": row["monitor_name"], "name": row["name"],
             "asset_type": row["asset_type"], "properties": json.loads(row["properties_json"]),
             "geometry": json.loads(row["geom"])}
            for row in asset_rows if overlaps(row["geom"])
        ],
        "monitors": [
            {"monitor_name": row["monitor_name"], "config": json.loads(row["config_json"]),
             "geometry": json.loads(row["footprint"])}
            for row in monitor_rows if overlaps(row["footprint"])
        ],
        "products": [
            {"owner_type": row["owner_type"], "owner_id": row["owner_id"], "asset_key": row["asset_key"],
             "media_type": row["media_type"], "geometry": json.loads(row["footprint"]) if row["footprint"] else None,
             "href": (f"{BASE_URL}/assets/{row['owner_id']}/{row['asset_key']}" if row["owner_type"] == "job"
                      else f"{BASE_URL}/stack-assets/{row['owner_id']}/{row['asset_key']}")}
            for row in product_rows
            if row["owner_type"] in {"job", "stack"}
            and row["asset_key"] in {"displacement_mm.tif", "velocity_mm_yr.tif", "cumulative_disp_mm.tif"}
            and overlaps(row["footprint"])
        ],
    }


@app.post("/v1/jobs", status_code=202)
def create_job(
    req: CreateJobRequest,
    _: str = Depends(verify_api_key),
):
    if not req.slc1_path or not req.slc2_path:
        raise HTTPException(status_code=400, detail="slc1_path and slc2_path are required")
    if not celery_enabled() and not ENGINE_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {ENGINE_BIN}")

    idempotency_key = req.idempotency_key
    if not idempotency_key:
        identity = (
            {"monitor_name": req.monitor_name, "granule_id": req.granule_id, "engine_version": req.engine_version}
            if req.granule_id
            else req.model_dump(mode="json", exclude={"webhook_url", "idempotency_key"})
        )
        idempotency_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        existing = conn.execute(
            "SELECT job_id, status FROM jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing:
            job_id = existing["job_id"]
            return {
                "job_id": job_id,
                "status": existing["status"],
                "status_url": f"{BASE_URL}/v1/jobs/{job_id}",
                "preview_url": f"{BASE_URL}/preview/{job_id}",
                "stac_item_url": f"{BASE_URL}/stac/items/{job_id}",
                "idempotent_replay": True,
            }
        cursor = conn.execute(
            """INSERT INTO jobs
               (job_id, monitor_name, mode, status, created_at, webhook_url, engine_version, idempotency_key)
               VALUES (?, ?, 'local', 'queued', ?, ?, ?, ?)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (job_id, req.monitor_name, now, req.webhook_url, req.engine_version, idempotency_key),
        )
        inserted = cursor.rowcount == 1
        if not inserted:
            existing = conn.execute(
                "SELECT job_id, status FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                conn.commit()
                job_id = existing["job_id"]
                return {
                    "job_id": job_id,
                    "status": existing["status"],
                    "status_url": f"{BASE_URL}/v1/jobs/{job_id}",
                    "preview_url": f"{BASE_URL}/preview/{job_id}",
                    "stac_item_url": f"{BASE_URL}/stac/items/{job_id}",
                    "idempotent_replay": True,
                }
        if inserted and celery_enabled():
            add_outbox_message(
                conn,
                "services.worker.tasks.run_pairwise_job",
                PairwiseJobTask(job_id=job_id, request=req.model_dump(mode="json")).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if inserted and not celery_enabled():
        from services.worker.execution import execute_insar_job

        execute_insar_job(job_id, req)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"{BASE_URL}/v1/jobs/{job_id}",
        "preview_url": f"{BASE_URL}/preview/{job_id}",
        "stac_item_url": f"{BASE_URL}/stac/items/{job_id}",
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _: str = Depends(verify_api_key)):
    _validate_safe_id(job_id, "job_id")
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    res = dict(row)
    if res["summary_json"]:
        res["summary"] = json.loads(res.pop("summary_json"))
    else:
        res.pop("summary_json", None)
    if res["status"] == "completed":
        res["stac_item_url"] = f"{BASE_URL}/stac/items/{job_id}"
    res["preview_url"] = f"{BASE_URL}/preview/{job_id}"
    res["workflow_steps"] = get_steps(job_id)
    return res


@app.get("/stac/items/{job_id}")
def get_stac_item(job_id: str):
    _validate_safe_id(job_id, "job_id")
    stac_path = DATA_DIR / "jobs" / job_id / "stac_item.json"
    if stac_path.exists():
        item = json.loads(stac_path.read_text())
    else:
        item = _read_s3_json_asset("job", job_id, "stac_item.json")
    if item is None:
        raise HTTPException(status_code=404, detail="STAC item not ready or job not found")
    return JSONResponse(
        content=item,
        media_type="application/geo+json",
    )


@app.get("/stac")
def stac_catalog():
    return {
        "type": "Catalog",
        "id": "insar-platform",
        "stac_version": "1.1.0",
        "description": "InSAR pair products and time-series stack results",
        "links": [
            {"rel": "self", "href": f"{BASE_URL}/stac", "type": "application/json"},
            {"rel": "data", "href": f"{BASE_URL}/stac/collections", "type": "application/json"},
            {"rel": "search", "href": f"{BASE_URL}/stac/search", "method": "POST", "type": "application/geo+json"},
        ],
    }


@app.get("/stac/ingestions/{ingestion_id}")
def get_ingestion_stac_item(ingestion_id: str):
    _validate_safe_id(ingestion_id, "ingestion_id")
    stac_path = DATA_DIR / "ingestions" / ingestion_id / "stac_item.json"
    if stac_path.is_file():
        item = json.loads(stac_path.read_text(encoding="utf-8"))
    else:
        item = _read_s3_json_asset("ingestion", ingestion_id, "stac_item.json")
    if item is None:
        raise HTTPException(status_code=404, detail="STAC item not ready or ingestion not found")
    return JSONResponse(content=item, media_type="application/geo+json")


@app.get("/stac/collections")
def stac_collections():
    return {
        "collections": [{
            "id": "insar-results",
            "type": "Collection",
            "stac_version": "1.1.0",
            "description": "Pairwise and time-series InSAR products",
            "license": "proprietary",
            "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}, "temporal": {"interval": [[None, None]]}},
            "links": [
                {"rel": "self", "href": f"{BASE_URL}/stac/collections/insar-results", "type": "application/json"},
                {"rel": "items", "href": f"{BASE_URL}/stac/collections/insar-results/items", "type": "application/geo+json"},
            ],
        }]
    }


@app.get("/stac/collections/insar-results")
def get_stac_collection():
    return stac_collections()["collections"][0]


def _stac_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    limit = max(1, min(int(payload.get("limit", 100)), 1000))
    requested_ids = set(payload.get("ids") or [])
    bbox = payload.get("bbox")
    candidates: list[tuple[str, str]] = []
    with get_db() as conn:
        if bbox and len(bbox) == 4 and conn.dialect == "postgresql":
            args = tuple(float(value) for value in bbox)
            matches = conn.execute(
                """SELECT DISTINCT owner_type, owner_id FROM assets
                   WHERE footprint && ST_MakeEnvelope(?, ?, ?, ?, 4326)
                     AND ST_Intersects(footprint, ST_MakeEnvelope(?, ?, ?, ?, 4326))
                   ORDER BY owner_type, owner_id LIMIT ?""",
                args + args + (limit,),
            ).fetchall()
            candidates.extend((row["owner_type"], row["owner_id"]) for row in matches)
        if not candidates:
            jobs = conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'completed' ORDER BY finished_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            stacks = conn.execute(
                "SELECT stack_id FROM stacks WHERE status = 'completed' ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            candidates.extend(("job", row["job_id"]) for row in jobs)
            candidates.extend(("stack", row["stack_id"]) for row in stacks)

    items: list[dict[str, Any]] = []
    for owner_type, owner_id in candidates:
        if requested_ids and owner_id not in requested_ids:
            continue
        local = DATA_DIR / ("jobs" if owner_type == "job" else "stacks") / owner_id / "stac_item.json"
        item = json.loads(local.read_text()) if local.is_file() else _read_s3_json_asset(owner_type, owner_id, "stac_item.json")
        if item is None:
            continue
        if bbox and item.get("bbox"):
            left, bottom, right, top = item["bbox"]
            if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
                continue
        items.append(item)
        if len(items) >= limit:
            break
    return items


@app.post("/stac/search")
def stac_search(payload: dict[str, Any]):
    items = _stac_search_items(payload)
    return {"type": "FeatureCollection", "stac_version": "1.1.0", "features": items, "links": []}


@app.get("/stac/collections/insar-results/items")
def stac_collection_items(limit: int = 100):
    return {"type": "FeatureCollection", "stac_version": "1.1.0", "features": _stac_search_items({"limit": limit}), "links": []}


ALLOWED_ASSETS = {
    "displacement_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "coherence.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "wrapped_phase.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "alerts.geojson": "application/geo+json",
    "summary.json": "application/json",
}


@app.api_route("/assets/{job_id}/{filename}", methods=["GET", "HEAD"])
def stream_asset(job_id: str, filename: str, request: Request):
    _validate_safe_id(job_id, "job_id")
    if filename not in ALLOWED_ASSETS:
        raise HTTPException(status_code=400, detail="Invalid asset request")
    response = _s3_asset_response("job", job_id, filename, ALLOWED_ASSETS[filename], request)
    if response is not None:
        return response
    file_path = DATA_DIR / "jobs" / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(file_path, media_type=ALLOWED_ASSETS[filename])


@app.get("/preview/{job_id}", response_class=HTMLResponse, include_in_schema=True)
def preview_job(job_id: str):
    """Serve a live, lightweight map preview for a queued or completed job."""
    _validate_safe_id(job_id, "job_id")
    with get_db() as conn:
        row = conn.execute("SELECT status, monitor_name FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    html = _render_job_preview(job_id, row["status"], row["monitor_name"])
    return HTMLResponse(content=html)


def _render_job_preview(job_id: str, initial_status: str, monitor_name: str) -> str:
    """Build the preview shell; job outputs are fetched through the existing asset route."""
    template = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>InSAR preview · __MONITOR_NAME__</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { color-scheme:dark; --bg:#10151c; --panel:#171e27; --line:#2a3542; --text:#edf2f7; --muted:#a9b6c5; }
    * { box-sizing:border-box; } html,body { height:100%; margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    .shell { height:100%; display:grid; grid-template-rows:auto auto minmax(0,1fr); }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:13px 18px; border-bottom:1px solid var(--line); background:#121820; }
    h1 { font-size:17px; margin:0; } .subtitle { margin-top:2px; color:var(--muted); font-size:11px; }
    #status { color:#f3bd68; font-size:12px; white-space:nowrap; } #status.ready { color:#69d3be; } #status.failed { color:#ff867f; }
    .toolbar { display:flex; align-items:center; gap:18px; flex-wrap:wrap; padding:8px 16px; background:var(--panel); border-bottom:1px solid var(--line); color:var(--muted); font-size:12px; }
    .toolbar label { display:flex; gap:8px; align-items:center; } input[type=range] { width:145px; accent-color:#69d3be; }
    #summary { color:var(--text); } #alert-count { margin-left:auto; }
    #map { min-height:0; width:100%; background:#111821; }
    .leaflet-popup-content-wrapper,.leaflet-popup-tip { background:#171e27; color:var(--text); }
    .leaflet-control-layers { background:#171e27; color:var(--text); border:1px solid var(--line); }
    .leaflet-control-attribution { background:rgba(17,24,32,.7)!important; color:#b9c3d0!important; }
    @media(max-width:650px) { header { align-items:flex-start; flex-direction:column; } #alert-count { margin-left:0; } }
  </style>
</head>
<body>
  <div class="shell">
    <header><div><h1>__MONITOR_NAME__</h1><div class="subtitle">Job __JOB_ID__ · displacement COG with alert overlays</div></div><div id="status">__INITIAL_STATUS__</div></header>
    <div class="toolbar">
      <label>Raster opacity <input id="opacity" type="range" min="10" max="100" value="82"><span id="opacity-value">82%</span></label>
      <span id="summary">Waiting for job summary…</span>
      <span id="alert-count">Alerts: —</span>
    </div>
    <div id="map" role="application" aria-label="Interactive job preview map"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/proj4@2.9.0/dist/proj4.js"></script>
  <script src="https://unpkg.com/georaster@1.6.0/dist/georaster.browser.bundle.min.js"></script>
  <script src="https://unpkg.com/georaster-layer-for-leaflet@3.10.0/dist/georaster-layer-for-leaflet.min.js"></script>
  <script>
    const JOB_ID = __JOB_ID_JSON__;
    const BASE = `/assets/${JOB_ID}`;
    for(let zone=1;zone<=60;zone++){
      proj4.defs(`EPSG:${32600+zone}`,`+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs +type=crs`);
      proj4.defs(`EPSG:${32700+zone}`,`+proj=utm +zone=${zone} +south +datum=WGS84 +units=m +no_defs +type=crs`);
    }
    const map = L.map('map', {preferCanvas:true}).setView([0,0],2);
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom:19, attribution:'Tiles © Esri — Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community'});
    const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {subdomains:'abcd',maxZoom:20,attribution:'© OpenStreetMap contributors © CARTO'});
    satellite.addTo(map);
    L.control.layers({'Satellite':satellite,'Dark':dark},null,{collapsed:false}).addTo(map);
    const initialStatus = __INITIAL_STATUS_JSON__;
    let rasterLayer = null, alertLayer = null, rasterLoading = false, finished = initialStatus === 'failed';
    const status = document.getElementById('status');
    const setStatus = (text, kind='') => {status.textContent=text;status.className=kind;};
    const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
    const opacity = document.getElementById('opacity');
    opacity.addEventListener('input',()=>{if(rasterLayer)rasterLayer.setOpacity(Number(opacity.value)/100);document.getElementById('opacity-value').textContent=`${opacity.value}%`;});
    const velocityStops = [[0,'#153e74'],[.24,'#64a8c7'],[.5,'#e9e8df'],[.76,'#e99c73'],[1,'#8b2338']];
    function rgb(hex){const n=parseInt(hex.slice(1),16);return[(n>>16)&255,(n>>8)&255,n&255];}
    function colorAt(x){x=Math.max(0,Math.min(1,x));let i=1;while(i<velocityStops.length&&x>velocityStops[i][0])i++;const a=velocityStops[i-1],b=velocityStops[Math.min(i,velocityStops.length-1)],t=(x-a[0])/Math.max(1e-9,b[0]-a[0]),ca=rgb(a[1]),cb=rgb(b[1]);return `rgb(${ca.map((v,j)=>Math.round(v+(cb[j]-v)*t)).join(',')})`;}
    async function loadRaster(){
      if(rasterLayer||rasterLoading)return !!rasterLayer;
      rasterLoading=true;
      try{
        const response=await fetch(`${BASE}/displacement_mm.tif`,{cache:'no-store'});
        if(!response.ok){rasterLoading=false;return false;}
        const georaster=await parseGeoraster(await response.arrayBuffer());
        const min=Number(georaster.mins?.[0]??georaster.min??-30),max=Number(georaster.maxs?.[0]??georaster.max??30),limit=Math.max(10,Math.abs(min),Math.abs(max));
        rasterLayer=new GeoRasterLayer({georaster,opacity:Number(opacity.value)/100,resolution:256,pixelValuesToColorFn:values=>{const value=Number(values?.[0]);if(!Number.isFinite(value)||(georaster.noDataValue!=null&&value===georaster.noDataValue))return'rgba(0,0,0,0)';return colorAt((value+limit)/(2*limit));}});
        rasterLayer.addTo(map);map.fitBounds(rasterLayer.getBounds().pad(.06));
        if(initialStatus !== 'failed')setStatus('COG loaded','ready');
        rasterLoading=false;return true;
      }catch(error){rasterLoading=false;setStatus(`Loading raster… ${error.message}`);return false;}
    }
    function popupFor(feature){
      const props=feature.properties||{}, box=document.createElement('div');
      const title=document.createElement('strong');title.textContent=props.name||props.alert_type||props.alert||'Displacement alert';box.appendChild(title);
      const rows=Object.entries(props).filter(([key,value])=>value!=null&&key!=='name').slice(0,8);
      for(const [key,value] of rows){const row=document.createElement('div');row.style.cssText='font-size:11px;color:#bdc8d5;margin-top:3px';row.textContent=`${key.replaceAll('_',' ')}: ${typeof value==='number'?fmt(value):String(value)}`;box.appendChild(row);}
      return box;
    }
    async function loadAlerts(){
      try{
        const response=await fetch(`${BASE}/alerts.geojson`,{cache:'no-store'});if(!response.ok)return;
        const data=await response.json();
        if(alertLayer)map.removeLayer(alertLayer);
        alertLayer=L.geoJSON(data,{pointToLayer:(feature,latlng)=>L.circleMarker(latlng,{radius:6,color:'#24191b',weight:1.5,fillColor:'#ff6b61',fillOpacity:.95}),style:feature=>({color:'#ff6b61',weight:3,opacity:.95}),onEachFeature:(feature,layer)=>layer.bindPopup(popupFor(feature))}).addTo(map);
        document.getElementById('alert-count').textContent=`Alerts: ${(data.features||[]).length}`;
      }catch(_error){}
    }
    async function loadSummary(){
      try{
        const response=await fetch(`${BASE}/summary.json`,{cache:'no-store'});if(!response.ok)return;
        const summary=await response.json();
        const parts=[];
        if(summary.valid_pixels!=null)parts.push(`${summary.valid_pixels} valid pixels`);
        if(summary.min_displacement_mm!=null||summary.max_displacement_mm!=null)parts.push(`${fmt(summary.min_displacement_mm)} to ${fmt(summary.max_displacement_mm)} mm`);
        if(summary.processing_time_ms!=null)parts.push(`${fmt(summary.processing_time_ms)} ms`);
        document.getElementById('summary').textContent=parts.join(' · ')||'Job summary loaded';
        finished=true;setStatus('Job complete','ready');
      }catch(_error){}
    }
    async function refresh(){
      await loadRaster();await loadSummary();await loadAlerts();
      if(!finished)setTimeout(refresh,5000);
    }
    setStatus(initialStatus === 'failed' ? 'Job failed' : initialStatus, initialStatus === 'failed' ? 'failed' : '');
    refresh();
  </script>
</body>
</html>'''
    return (
        template.replace("__JOB_ID_JSON__", json.dumps(job_id))
        .replace("__JOB_ID__", job_id)
        .replace("__INITIAL_STATUS_JSON__", json.dumps(initial_status))
        .replace("__INITIAL_STATUS__", initial_status)
        .replace("__MONITOR_NAME__", monitor_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    )


# ============================================================================
# API Routes — Sequential Multi-Pass Stacking (`insar_stack`)
# ============================================================================

@app.post("/v1/stacks/{stack_id}/updates", status_code=202)
def update_stack(
    stack_id: str,
    req: StackUpdateRequest,
    _: str = Depends(verify_api_key),
):
    _validate_safe_id(stack_id, "stack_id")
    monitor_name = req.monitor_name or stack_id
    identity = {
        "stack_id": stack_id,
        "monitor_name": monitor_name,
        "engine_version": req.engine_version,
        "request": req.model_dump(mode="json", exclude={"idempotency_key", "webhook_url"}),
    }
    idempotency_key = req.idempotency_key or hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with get_db() as conn:
        duplicate = conn.execute(
            "SELECT update_id, status FROM stack_updates WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    if duplicate:
        return {
            "stack_id": stack_id,
            "update_id": duplicate["update_id"],
            "status": duplicate["status"],
            "stack_url": f"{BASE_URL}/v1/stacks/{stack_id}",
            "stac_item_url": f"{BASE_URL}/stac/stacks/{stack_id}",
            "idempotent_replay": True,
        }
    if not celery_enabled() and not STACK_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ stack binary not found at {STACK_BIN}")
    if req.pair_job is not None and not celery_enabled() and not ENGINE_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {ENGINE_BIN}")

    has_pair_input = bool(req.pair_job is not None or req.source_job_id or (req.disp_path and req.coh_path))
    if bool(req.disp_path) != bool(req.coh_path):
        raise HTTPException(status_code=400, detail="disp_path and coh_path must both be provided")
    if has_pair_input:
        has_t = req.t0_days is not None and req.t1_days is not None
        if not has_t and req.dt_days is None:
            raise HTTPException(status_code=400, detail="Provide dt_days or (t0_days, t1_days) when ingesting a pair")
        if has_t and req.t1_days <= req.t0_days:
            raise HTTPException(status_code=400, detail="t1_days must exceed t0_days")
    else:
        state_path = DATA_DIR / "stacks" / stack_id / "stack_state.tif"
        has_persisted_state = False
        if s3_enabled():
            with get_db() as conn:
                has_persisted_state = conn.execute(
                    "SELECT 1 FROM assets WHERE owner_type = 'stack' AND owner_id = ? AND asset_key = 'stack_state.tif'",
                    (stack_id,),
                ).fetchone() is not None
        if not state_path.exists() and not has_persisted_state:
            raise HTTPException(
                status_code=400,
                detail="Provide source_job_id, disp_path/coh_path, or pair_job to initialize a new stack",
            )

    req = req.model_copy(update={"idempotency_key": idempotency_key})
    update_id = f"upd_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        existing = conn.execute("SELECT monitor_name FROM stacks WHERE stack_id = ?", (stack_id,)).fetchone()
        if existing:
            monitor_name = req.monitor_name or existing["monitor_name"]
            conn.execute(
                """UPDATE stacks SET monitor_name = ?, status = 'queued', updated_at = ?, webhook_url = ?
                   WHERE stack_id = ?""",
                (monitor_name, now, req.webhook_url, stack_id),
            )
        else:
            conn.execute(
                """INSERT INTO stacks (stack_id, monitor_name, status, pair_count, created_at, updated_at, webhook_url)
                   VALUES (?, ?, 'queued', 0, ?, ?, ?)
                   ON CONFLICT (stack_id) DO UPDATE SET monitor_name = excluded.monitor_name,
                       status = 'queued', updated_at = excluded.updated_at, webhook_url = excluded.webhook_url""",
                (stack_id, monitor_name, now, now, req.webhook_url),
            )
        cursor = conn.execute(
            """INSERT INTO stack_updates
               (update_id, stack_id, source_job_id, status, created_at, engine_version, idempotency_key)
               VALUES (?, ?, ?, 'queued', ?, ?, ?)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (update_id, stack_id, req.source_job_id, now, req.engine_version, idempotency_key),
        )
        if cursor.rowcount != 1:
            duplicate = conn.execute(
                "SELECT update_id, status FROM stack_updates WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            conn.commit()
            return {
                "stack_id": stack_id,
                "update_id": duplicate["update_id"],
                "status": duplicate["status"],
                "stack_url": f"{BASE_URL}/v1/stacks/{stack_id}",
                "stac_item_url": f"{BASE_URL}/stac/stacks/{stack_id}",
                "idempotent_replay": True,
            }
        if celery_enabled():
            add_outbox_message(
                conn,
                "services.worker.tasks.run_stack_update",
                StackUpdateTask(
                    stack_id=stack_id,
                    update_id=update_id,
                    monitor_name=monitor_name,
                    request=req.model_dump(mode="json"),
                ).model_dump(mode="json"),
                now,
            )
        conn.commit()

    if not celery_enabled():
        from services.worker.execution import execute_stack_update

        execute_stack_update(stack_id, update_id, monitor_name, req)
    return {
        "stack_id": stack_id,
        "update_id": update_id,
        "status": "queued",
        "stack_url": f"{BASE_URL}/v1/stacks/{stack_id}",
        "stac_item_url": f"{BASE_URL}/stac/stacks/{stack_id}",
    }


@app.get("/v1/stacks")
def list_stacks(_: str = Depends(verify_api_key)):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM stacks ORDER BY updated_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        if item["summary_json"]:
            item["summary"] = json.loads(item.pop("summary_json"))
        else:
            item.pop("summary_json", None)
        out.append(item)
    return {"stacks": out}


@app.get("/v1/stacks/{stack_id}")
def get_stack(stack_id: str, _: str = Depends(verify_api_key)):
    _validate_safe_id(stack_id, "stack_id")
    with get_db() as conn:
        row = conn.execute("SELECT * FROM stacks WHERE stack_id = ?", (stack_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Stack not found")
        updates = conn.execute(
            "SELECT update_id, source_job_id, engine_version, status, created_at, finished_at, error_message "
            "FROM stack_updates WHERE stack_id = ? ORDER BY created_at DESC LIMIT 25",
            (stack_id,),
        ).fetchall()

    res = dict(row)
    if res["summary_json"]:
        res["summary"] = json.loads(res.pop("summary_json"))
    else:
        res.pop("summary_json", None)
    if res["status"] == "completed":
        res["stac_item_url"] = f"{BASE_URL}/stac/stacks/{stack_id}"
    res["recent_updates"] = [
        {**dict(u), "workflow_steps": get_steps(u["update_id"])}
        for u in updates
    ]
    return res


@app.get("/stac/stacks/{stack_id}")
def get_stack_stac_item(stack_id: str):
    _validate_safe_id(stack_id, "stack_id")
    stac_path = DATA_DIR / "stacks" / stack_id / "stac_item.json"
    if stac_path.exists():
        item = json.loads(stac_path.read_text())
    else:
        item = _read_s3_json_asset("stack", stack_id, "stac_item.json")
    if item is None:
        raise HTTPException(status_code=404, detail="Stack STAC item not ready or stack not found")
    return JSONResponse(
        content=item,
        media_type="application/geo+json",
    )


ALLOWED_STACK_ASSETS = {
    "velocity_mm_yr.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "acceleration_mm_yr2.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "cumulative_disp_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "temporal_coherence.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "velocity_uncertainty_mm_yr.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "velocity_snr.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "ramp_removed_disp_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "closure_residual_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "corridor_report.geojson": "application/geo+json",
    "corridor_alerts.geojson": "application/geo+json",
    "stack_summary.json": "application/json",
}


@app.api_route("/stack-assets/{stack_id}/{filename}", methods=["GET", "HEAD"])
def stream_stack_asset(stack_id: str, filename: str, request: Request):
    _validate_safe_id(stack_id, "stack_id")
    if filename not in ALLOWED_STACK_ASSETS:
        raise HTTPException(status_code=400, detail="Invalid stack asset request")
    response = _s3_asset_response("stack", stack_id, filename, ALLOWED_STACK_ASSETS[filename], request)
    if response is not None:
        return response
    file_path = DATA_DIR / "stacks" / stack_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stack asset not found")
    return FileResponse(file_path, media_type=ALLOWED_STACK_ASSETS[filename])


@app.post("/v1/dev/webhook-sink")
async def dev_webhook_sink(
    request: Request,
    x_insar_signature: str = Header(..., alias="X-InSAR-Signature"),
):
    """Built-in local receiver to verify HMAC-SHA256 webhook signatures."""
    raw_body = await request.body()
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(x_insar_signature, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = json.loads(raw_body)
    target = event.get("job_id") or event.get("stack_id")
    print(f"\n[WEBHOOK VERIFIED] {event['event']} | target={target} | stac={event.get('stac_item_url')}\n")
    return {"received": True, "signature_valid": True}
