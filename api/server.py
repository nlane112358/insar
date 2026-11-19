import hashlib
import hmac
import json
import os
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

ENGINE_BIN = Path(os.getenv("INSAR_ENGINE_BIN", "./build/insar_proto"))
DATA_DIR = Path(os.getenv("INSAR_DATA_DIR", "./api_data"))
DB_PATH = DATA_DIR / "oem_jobs.db"
API_KEY = os.getenv("INSAR_API_KEY", "insar_dev_secret_key")
WEBHOOK_SECRET = os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret")
BASE_URL = os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

DATA_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="InSAR OEM Compute API", version="0.3.0")


# ============================================================================
# SQLite State Store
# ============================================================================

@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id         TEXT PRIMARY KEY,
                monitor_name   TEXT NOT NULL,
                mode           TEXT NOT NULL,
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                finished_at    TEXT,
                webhook_url    TEXT,
                webhook_status TEXT,
                summary_json   TEXT,
                error_message  TEXT
            )
        """)
        conn.commit()


init_db()


# ============================================================================
# Auth & Request Models
# ============================================================================

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return x_api_key


class AlertRules(BaseModel):
    threshold_mm: float = Field(25.0, gt=0, description="Flag high-coherence motion exceeding +/- mm")
    min_coherence: float = Field(0.65, ge=0.0, le=1.0)


class CreateJobRequest(BaseModel):
    monitor_name: str = Field(..., examples=["yorkshire_trunk_main_north"])
    mode: Literal["synthetic", "local"] = "synthetic"
    # Used when mode == "local" (points to mounted OPERA CSLC-S1 .h5 / NISAR GSLC / GeoTIFF)
    slc1_path: Optional[str] = None
    slc2_path: Optional[str] = None
    subdataset: str = "/data/VV"
    wavelength_mm: float = 55.4658
    stride_x: int = 1
    stride_y: int = 1
    # Used when mode == "synthetic"
    width: int = Field(900, ge=64, le=8192)
    height: int = Field(700, ge=64, le=8192)
    alert_rules: AlertRules = Field(default_factory=AlertRules)
    webhook_url: Optional[str] = None


# ============================================================================
# STAC 1.1.0 Item Builder & Signed Webhook Dispatcher
# ============================================================================

def build_stac_item(job_id: str, monitor_name: str, finished_at: str, summary: dict) -> dict:
    """Build a valid STAC 1.1.0 Feature Item pointing to streamable COGs."""
    bbox = summary.get("bbox_wgs84")
    if bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }
    else:
        geometry = None

    asset_base = f"{BASE_URL}/assets/{job_id}"
    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": job_id,
        "geometry": geometry,
        "properties": {
            "datetime": finished_at,
            "insar:monitor_name": monitor_name,
            "insar:processing_time_ms": summary.get("processing_time_ms"),
            "insar:valid_pixels": summary.get("valid_pixels"),
            "insar:min_displacement_mm": summary.get("min_displacement_mm"),
            "insar:max_displacement_mm": summary.get("max_displacement_mm"),
            "insar:anomalies_detected": summary.get("anomalies_detected"),
        },
        "links": [
            {
                "rel": "self",
                "href": f"{BASE_URL}/stac/items/{job_id}",
                "type": "application/geo+json",
            }
        ],
        "assets": {
            "displacement_mm": {
                "href": f"{asset_base}/displacement_mm.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": "Unwrapped Line-of-Sight Displacement (mm)",
                "roles": ["data"],
            },
            "coherence": {
                "href": f"{asset_base}/coherence.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": "Interferometric Coherence (0-1)",
                "roles": ["data", "quality"],
            },
            "wrapped_phase": {
                "href": f"{asset_base}/wrapped_phase.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "title": "Wrapped Interferometric Phase (rad)",
                "roles": ["data"],
            },
            "alerts": {
                "href": f"{asset_base}/alerts.geojson",
                "type": "application/geo+json",
                "title": "High-Coherence Displacement Anomaly Alerts",
                "roles": ["metadata"],
            },
        },
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def dispatch_webhook(webhook_url: str, payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-InSAR-Signature": f"sha256={sig}",
        "X-InSAR-Event": payload["event"],
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(webhook_url, content=body, headers=headers)
            return f"delivered ({resp.status_code})"
    except Exception as exc:
        return f"failed ({exc})"


# ============================================================================
# Background Worker Invoking Native C++ Engine
# ============================================================================

def execute_insar_job(job_id: str, req: CreateJobRequest) -> None:
    out_dir = DATA_DIR / "jobs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
        conn.commit()

    cmd = [
        str(ENGINE_BIN),
        "--out-dir", str(out_dir),
        "--wavelength-mm", str(req.wavelength_mm),
        "--alert-mm", str(req.alert_rules.threshold_mm),
        "--alert-coh", str(req.alert_rules.min_coherence),
    ]
    if req.mode == "synthetic":
        cmd += ["--width", str(req.width), "--height", str(req.height)]
    else:
        cmd += [
            "--slc1", req.slc1_path,
            "--slc2", req.slc2_path,
            "--subdataset", req.subdataset,
            "--stride-x", str(req.stride_x),
            "--stride-y", str(req.stride_y),
        ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary = json.loads((out_dir / "summary.json").read_text())
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        stac_item = build_stac_item(job_id, req.monitor_name, finished_at, summary)
        (out_dir / "stac_item.json").write_text(json.dumps(stac_item, indent=2))

        wh_status = None
        if req.webhook_url:
            event_payload = {
                "event": "monitor.pass_completed",
                "job_id": job_id,
                "monitor_name": req.monitor_name,
                "finished_at": finished_at,
                "processing_time_ms": summary["processing_time_ms"],
                "anomalies_detected": summary["anomalies_detected"],
                "stac_item_url": f"{BASE_URL}/stac/items/{job_id}",
                "assets": {k: v["href"] for k, v in stac_item["assets"].items()},
            }
            wh_status = dispatch_webhook(req.webhook_url, event_payload)

        with get_db() as conn:
            conn.execute(
                """UPDATE jobs
                   SET status = 'completed', finished_at = ?, summary_json = ?, webhook_status = ?
                   WHERE job_id = ?""",
                (finished_at, json.dumps(summary), wh_status, job_id),
            )
            conn.commit()

    except Exception as exc:
        err_msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            err_msg = f"Engine exit {exc.returncode}: {exc.stderr.strip()}"
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error_message = ? WHERE job_id = ?",
                (err_msg, job_id),
            )
            conn.commit()


# ============================================================================
# API Routes
# ============================================================================

@app.post("/v1/jobs", status_code=202)
def create_job(
    req: CreateJobRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    if req.mode == "local" and (not req.slc1_path or not req.slc2_path):
        raise HTTPException(status_code=400, detail="slc1_path and slc2_path are required for local mode")
    if not ENGINE_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {ENGINE_BIN}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, monitor_name, mode, status, created_at, webhook_url)
               VALUES (?, ?, ?, 'queued', ?, ?)""",
            (job_id, req.monitor_name, req.mode, now, req.webhook_url),
        )
        conn.commit()

    background_tasks.add_task(execute_insar_job, job_id, req)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"{BASE_URL}/v1/jobs/{job_id}",
        "stac_item_url": f"{BASE_URL}/stac/items/{job_id}",
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _: str = Depends(verify_api_key)):
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
    return res


@app.get("/stac/items/{job_id}")
def get_stac_item(job_id: str):
    if "/" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    stac_path = DATA_DIR / "jobs" / job_id / "stac_item.json"
    if not stac_path.exists():
        raise HTTPException(status_code=404, detail="STAC item not ready or job not found")
    return JSONResponse(
        content=json.loads(stac_path.read_text()),
        media_type="application/geo+json",
    )


ALLOWED_ASSETS = {
    "displacement_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "coherence.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "wrapped_phase.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "alerts.geojson": "application/geo+json",
    "summary.json": "application/json",
}


@app.api_route("/assets/{job_id}/{filename}", methods=["GET", "HEAD"])
def stream_asset(job_id: str, filename: str):
    if filename not in ALLOWED_ASSETS or "/" in job_id or ".." in job_id:
        raise HTTPException(status_code=400, detail="Invalid asset request")
    file_path = DATA_DIR / "jobs" / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    # FileResponse natively supports HTTP Range requests for COG streaming in QGIS
    return FileResponse(file_path, media_type=ALLOWED_ASSETS[filename])


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
    print(f"\n[WEBHOOK VERIFIED] {event['event']} | job={event['job_id']} "
          f"| anomalies={event['anomalies_detected']} | stac={event['stac_item_url']}\n")
    return {"received": True, "signature_valid": True}
