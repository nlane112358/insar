import hashlib
import hmac
import json
import os
import re
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
STACK_BIN = Path(os.getenv("INSAR_STACK_BIN", "./build/insar_stack"))
DATA_DIR = Path(os.getenv("INSAR_DATA_DIR", "./api_data"))
DB_PATH = DATA_DIR / "oem_jobs.db"
API_KEY = os.getenv("INSAR_API_KEY", "insar_dev_secret_key")
WEBHOOK_SECRET = os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret")
BASE_URL = os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

DATA_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="InSAR OEM Compute & Stacking API", version="0.4.0")


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stacks (
                stack_id       TEXT PRIMARY KEY,
                monitor_name   TEXT NOT NULL,
                status         TEXT NOT NULL,
                pair_count     INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                webhook_url    TEXT,
                webhook_status TEXT,
                summary_json   TEXT,
                error_message  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stack_updates (
                update_id      TEXT PRIMARY KEY,
                stack_id       TEXT NOT NULL,
                source_job_id  TEXT,
                status         TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                finished_at    TEXT,
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


def _validate_safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value) or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return value


class AlertRules(BaseModel):
    threshold_mm: float = Field(25.0, gt=0, description="Flag high-coherence motion exceeding +/- mm")
    min_coherence: float = Field(0.65, ge=0.0, le=1.0)


class CreateJobRequest(BaseModel):
    monitor_name: str = Field(..., examples=["yorkshire_trunk_main_north"])
    slc1_path: str = Field(..., description="Reference OPERA CSLC-S1 .h5, NISAR GSLC, or complex GeoTIFF")
    slc2_path: str = Field(..., description="Secondary OPERA CSLC-S1 .h5, NISAR GSLC, or complex GeoTIFF")
    subdataset: str = "/data/VV"
    wavelength_mm: float = 55.4658
    stride_x: int = 6
    stride_y: int = 3
    ref_lonlat: Optional[tuple[float, float]] = None
    ref_radius_m: float = Field(1000.0, gt=0)
    ref_px: Optional[tuple[int, int, int]] = None
    alert_rules: AlertRules = Field(default_factory=AlertRules)
    webhook_url: Optional[str] = None


class ClosureTripletRequest(BaseModel):
    disp_a_path: str
    disp_b_path: str
    mode: Literal["add_sub", "sum_sub"] = "add_sub"


class StackUpdateRequest(BaseModel):
    monitor_name: Optional[str] = None
    # Pair input options: existing job_id, local disp/coh GeoTIFFs, or inline insar_proto run
    source_job_id: Optional[str] = None
    disp_path: Optional[str] = None
    coh_path: Optional[str] = None
    pair_job: Optional[CreateJobRequest] = None
    # Timing (required when ingesting a pair)
    dt_days: Optional[float] = Field(default=None, gt=0)
    t0_days: Optional[float] = None
    t1_days: Optional[float] = None
    # Reference region, DEM tropospheric correction, and ramp masking
    ramp_order: int = Field(1, ge=-1, le=2)
    min_coh: float = Field(0.35, ge=0.0, le=1.0)
    dem_path: Optional[str] = None
    ref_lonlat: Optional[tuple[float, float]] = None
    ref_radius_m: float = Field(1000.0, gt=0)
    ref_px: Optional[tuple[int, int, int]] = None
    ref_region_path: Optional[str] = None
    deform_mask_path: Optional[str] = None
    prior_vel_path: Optional[str] = None
    ramp_deform_vel_mm_yr: float = Field(25.0, ge=0.0)
    ramp_mask_sigma: float = Field(2.5, ge=0.0)
    ramp_mask_dilate_px: int = Field(2, ge=0, le=25)
    # Triplet phase closure & redundant skip pairs
    closure_triplets: list[ClosureTripletRequest] = Field(default_factory=list)
    closure_thresh_mm: float = Field(10.0, gt=0)
    closure_max_bad_frac: float = Field(0.35, ge=0.0, le=1.0)
    closure_mode: Literal["consensus", "strict"] = "consensus"
    redundant_pair: bool = False
    pair_weight_scale: float = Field(1.0, gt=0)
    # Corridors & SNR/maturity gating
    corridors_path: Optional[str] = None
    buffer_m: float = Field(100.0, ge=0.0)
    vel_alert_mm_yr: float = Field(15.0, gt=0)
    accel_alert_mm_yr2: float = Field(25.0, gt=0)
    min_alert_coh: float = Field(0.55, ge=0.0, le=1.0)
    min_alert_snr: float = Field(2.5, ge=0.0)
    min_alert_span_days: float = Field(60.0, ge=0.0)
    webhook_url: Optional[str] = None


# ============================================================================
# STAC 1.1.0 Item Builders & Signed Webhook Dispatcher
# ============================================================================

def build_stac_item(job_id: str, monitor_name: str, finished_at: str, summary: dict) -> dict:
    """Build a valid STAC 1.1.0 Feature Item pointing to streamable pair COGs."""
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
            "insar:ref_region_pinned": summary.get("ref_region_pinned"),
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


def build_stack_stac_item(
    stack_id: str,
    monitor_name: str,
    updated_at: str,
    summary: dict,
    stack_dir: Path,
) -> dict:
    """Build a STAC 1.1.0 Feature Item pointing to all available derived stack COGs & corridor GeoJSONs."""
    asset_base = f"{BASE_URL}/stack-assets/{stack_id}"
    candidate_assets = [
        ("velocity_mm_yr", "velocity_mm_yr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "LOS Velocity (mm/yr)", ["data"]),
        ("acceleration_mm_yr2", "acceleration_mm_yr2.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "LOS Acceleration (mm/yr^2)", ["data"]),
        ("cumulative_disp_mm", "cumulative_disp_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Cumulative LOS Displacement (mm)", ["data"]),
        ("temporal_coherence", "temporal_coherence.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Mean Temporal Coherence (0-1)", ["data", "quality"]),
        ("velocity_uncertainty_mm_yr", "velocity_uncertainty_mm_yr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Velocity Standard Error (mm/yr)", ["data", "quality"]),
        ("velocity_snr", "velocity_snr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Velocity Signal-to-Noise Ratio (|v|/sigma_v)", ["data", "quality"]),
        ("ramp_removed_disp_mm", "ramp_removed_disp_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Latest Corrected Pair Displacement (mm)", ["data"]),
        ("closure_residual_mm", "closure_residual_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Triplet Phase Closure Residual (mm)", ["data", "quality"]),
        ("corridor_report", "corridor_report.geojson", "application/geo+json", "Infrastructure Corridor Risk Report", ["metadata"]),
        ("corridor_alerts", "corridor_alerts.geojson", "application/geo+json", "SNR/Maturity-Gated Corridor Alerts", ["metadata"]),
        ("stack_summary", "stack_summary.json", "application/json", "Stack Summary & Quality Metrics", ["metadata"]),
    ]
    assets: dict[str, dict] = {}
    for key, fname, mime, title, roles in candidate_assets:
        if (stack_dir / fname).is_file():
            assets[key] = {
                "href": f"{asset_base}/{fname}",
                "type": mime,
                "title": title,
                "roles": roles,
            }

    mat = summary.get("maturity_gating", {})
    corr = summary.get("corridors", {})
    ramp = summary.get("ramp_removal", {})
    ref = summary.get("reference_region", {})
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": stack_id,
        "geometry": None,
        "properties": {
            "datetime": updated_at,
            "insar:monitor_name": monitor_name,
            "insar:pair_count": summary.get("pair_count"),
            "insar:redundant_pair_count": summary.get("redundant_pair_count"),
            "insar:rejected_pair_count": summary.get("rejected_pair_count"),
            "insar:calendar_span_days": summary.get("calendar_span_days"),
            "insar:valid_stack_pixels": summary.get("valid_stack_pixels"),
            "insar:mean_temporal_coherence": summary.get("mean_temporal_coherence"),
            "insar:p95_abs_velocity_mm_yr": summary.get("p95_abs_velocity_mm_yr"),
            "insar:dem_tropo_applied": ramp.get("dem_applied"),
            "insar:ref_region_applied": ref.get("applied"),
            "insar:stack_mature": mat.get("stack_mature"),
            "insar:critical_corridors": corr.get("critical_count"),
            "insar:corridor_alerts": corr.get("corridor_alerts"),
        },
        "links": [
            {
                "rel": "self",
                "href": f"{BASE_URL}/stac/stacks/{stack_id}",
                "type": "application/geo+json",
            }
        ],
        "assets": assets,
    }


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
# Background Workers Invoking Native C++ Engines (`insar_proto` & `insar_stack`)
# ============================================================================

def _build_proto_cmd(out_dir: Path, req: CreateJobRequest) -> list[str]:
    cmd = [
        str(ENGINE_BIN),
        "--slc1", str(req.slc1_path),
        "--slc2", str(req.slc2_path),
        "--subdataset", req.subdataset,
        "--stride-x", str(req.stride_x),
        "--stride-y", str(req.stride_y),
        "--out-dir", str(out_dir),
        "--wavelength-mm", str(req.wavelength_mm),
        "--alert-mm", str(req.alert_rules.threshold_mm),
        "--alert-coh", str(req.alert_rules.min_coherence),
    ]
    if req.ref_lonlat is not None:
        cmd += [
            "--ref-lonlat", str(req.ref_lonlat[0]), str(req.ref_lonlat[1]),
            "--ref-radius-m", str(req.ref_radius_m),
        ]
    elif req.ref_px is not None:
        cmd += ["--ref-px", str(req.ref_px[0]), str(req.ref_px[1]), str(req.ref_px[2])]
    return cmd


def execute_insar_job(job_id: str, req: CreateJobRequest) -> None:
    out_dir = DATA_DIR / "jobs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
        conn.commit()

    cmd = _build_proto_cmd(out_dir, req)

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


def execute_stack_update(stack_id: str, update_id: str, monitor_name: str, req: StackUpdateRequest) -> None:
    stack_dir = DATA_DIR / "stacks" / stack_id
    stack_dir.mkdir(parents=True, exist_ok=True)
    state_path = stack_dir / "stack_state.tif"

    with get_db() as conn:
        conn.execute("UPDATE stacks SET status = 'running' WHERE stack_id = ?", (stack_id,))
        conn.execute("UPDATE stack_updates SET status = 'running' WHERE update_id = ?", (update_id,))
        conn.commit()

    try:
        disp_path = req.disp_path
        coh_path = req.coh_path
        source_job_id = req.source_job_id

        # Option A: run insar_proto inline first
        if req.pair_job is not None:
            pair_dir = stack_dir / "pairs" / update_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            proto_cmd = _build_proto_cmd(pair_dir, req.pair_job)
            subprocess.run(proto_cmd, capture_output=True, text=True, check=True)
            disp_path = str(pair_dir / "displacement_mm.tif")
            coh_path = str(pair_dir / "coherence.tif")
        # Option B: pull from an existing completed /v1/jobs/{job_id}
        elif source_job_id:
            job_dir = DATA_DIR / "jobs" / source_job_id
            disp_file = job_dir / "displacement_mm.tif"
            coh_file = job_dir / "coherence.tif"
            if not (disp_file.is_file() and coh_file.is_file()):
                raise FileNotFoundError(f"Job '{source_job_id}' rasters not found at {job_dir}")
            disp_path = str(disp_file)
            coh_path = str(coh_file)

        cmd = [
            str(STACK_BIN),
            "--state", str(state_path),
            "--out-dir", str(stack_dir),
            "--ramp-order", str(req.ramp_order),
            "--min-coh", str(req.min_coh),
            "--ramp-deform-vel-mm-yr", str(req.ramp_deform_vel_mm_yr),
            "--ramp-mask-sigma", str(req.ramp_mask_sigma),
            "--ramp-mask-dilate-px", str(req.ramp_mask_dilate_px),
            "--closure-thresh-mm", str(req.closure_thresh_mm),
            "--closure-max-bad-frac", str(req.closure_max_bad_frac),
            "--closure-mode", req.closure_mode,
            "--buffer-m", str(req.buffer_m),
            "--vel-alert-mm-yr", str(req.vel_alert_mm_yr),
            "--accel-alert-mm-yr2", str(req.accel_alert_mm_yr2),
            "--min-alert-coh", str(req.min_alert_coh),
            "--min-alert-snr", str(req.min_alert_snr),
            "--min-alert-span-days", str(req.min_alert_span_days),
        ]
        if disp_path and coh_path:
            cmd += ["--disp", disp_path, "--coh", coh_path]
            if req.t0_days is not None and req.t1_days is not None:
                cmd += ["--t0-days", str(req.t0_days), "--t1-days", str(req.t1_days)]
            elif req.dt_days is not None:
                cmd += ["--dt-days", str(req.dt_days)]
            if req.redundant_pair:
                cmd += ["--redundant-pair", "--pair-weight-scale", str(req.pair_weight_scale)]

        if req.dem_path:
            cmd += ["--dem", req.dem_path]
        if req.ref_lonlat is not None:
            cmd += [
                "--ref-lonlat", str(req.ref_lonlat[0]), str(req.ref_lonlat[1]),
                "--ref-radius-m", str(req.ref_radius_m),
            ]
        elif req.ref_px is not None:
            cmd += ["--ref-px", str(req.ref_px[0]), str(req.ref_px[1]), str(req.ref_px[2])]
        if req.ref_region_path:
            cmd += ["--ref-region", req.ref_region_path]
        if req.deform_mask_path:
            cmd += ["--deform-mask", req.deform_mask_path]
        if req.prior_vel_path:
            cmd += ["--prior-vel", req.prior_vel_path]
        if req.corridors_path:
            cmd += ["--corridors", req.corridors_path]
        for trip in req.closure_triplets:
            cmd += ["--closure-triplet", trip.disp_a_path, trip.disp_b_path, trip.mode]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary = json.loads((stack_dir / "stack_summary.json").read_text())
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        stac_item = build_stack_stac_item(stack_id, monitor_name, finished_at, summary, stack_dir)
        (stack_dir / "stac_item.json").write_text(json.dumps(stac_item, indent=2))

        wh_status = None
        if req.webhook_url:
            event_payload = {
                "event": "stack.updated",
                "stack_id": stack_id,
                "update_id": update_id,
                "monitor_name": monitor_name,
                "finished_at": finished_at,
                "pair_count": summary.get("pair_count"),
                "calendar_span_days": summary.get("calendar_span_days"),
                "stack_mature": summary.get("maturity_gating", {}).get("stack_mature"),
                "pair_rejected": summary.get("triplet_closure", {}).get("pair_rejected"),
                "corridor_alerts": summary.get("corridors", {}).get("corridor_alerts"),
                "stac_item_url": f"{BASE_URL}/stac/stacks/{stack_id}",
                "assets": {k: v["href"] for k, v in stac_item["assets"].items()},
            }
            wh_status = dispatch_webhook(req.webhook_url, event_payload)

        with get_db() as conn:
            conn.execute(
                """UPDATE stacks
                   SET status = 'completed', pair_count = ?, updated_at = ?,
                       summary_json = ?, webhook_status = ?, error_message = NULL
                   WHERE stack_id = ?""",
                (
                    int(summary.get("pair_count", 0)),
                    finished_at,
                    json.dumps(summary),
                    wh_status,
                    stack_id,
                ),
            )
            conn.execute(
                """UPDATE stack_updates
                   SET status = 'completed', finished_at = ?, summary_json = ?, error_message = NULL
                   WHERE update_id = ?""",
                (finished_at, json.dumps(summary), update_id),
            )
            conn.commit()
    except Exception as exc:
        err_msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            err_msg = f"Stack engine exit {exc.returncode}: {exc.stderr.strip()}"
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with get_db() as conn:
            conn.execute(
                "UPDATE stacks SET status = 'failed', updated_at = ?, error_message = ? WHERE stack_id = ?",
                (finished_at, err_msg, stack_id),
            )
            conn.execute(
                "UPDATE stack_updates SET status = 'failed', finished_at = ?, error_message = ? WHERE update_id = ?",
                (finished_at, err_msg, update_id),
            )
            conn.commit()


# ============================================================================
# API Routes — Health & Single-Pair Jobs (`insar_proto`)
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "engine_bin": str(ENGINE_BIN),
        "engine_ready": ENGINE_BIN.is_file(),
        "stack_bin": str(STACK_BIN),
        "stack_ready": STACK_BIN.is_file(),
    }


@app.post("/v1/jobs", status_code=202)
def create_job(
    req: CreateJobRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    if not req.slc1_path or not req.slc2_path:
        raise HTTPException(status_code=400, detail="slc1_path and slc2_path are required")
    if not ENGINE_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ engine binary not found at {ENGINE_BIN}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, monitor_name, mode, status, created_at, webhook_url)
               VALUES (?, ?, 'local', 'queued', ?, ?)""",
            (job_id, req.monitor_name, now, req.webhook_url),
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
    return res


@app.get("/stac/items/{job_id}")
def get_stac_item(job_id: str):
    _validate_safe_id(job_id, "job_id")
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
    _validate_safe_id(job_id, "job_id")
    if filename not in ALLOWED_ASSETS:
        raise HTTPException(status_code=400, detail="Invalid asset request")
    file_path = DATA_DIR / "jobs" / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(file_path, media_type=ALLOWED_ASSETS[filename])


# ============================================================================
# API Routes — Sequential Multi-Pass Stacking (`insar_stack`)
# ============================================================================

@app.post("/v1/stacks/{stack_id}/updates", status_code=202)
def update_stack(
    stack_id: str,
    req: StackUpdateRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
):
    _validate_safe_id(stack_id, "stack_id")
    if not STACK_BIN.exists():
        raise HTTPException(status_code=500, detail=f"C++ stack binary not found at {STACK_BIN}")
    if req.pair_job is not None and not ENGINE_BIN.exists():
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
        if not state_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Provide source_job_id, disp_path/coh_path, or pair_job to initialize a new stack",
            )

    update_id = f"upd_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    monitor_name = req.monitor_name or stack_id

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
                   VALUES (?, ?, 'queued', 0, ?, ?, ?)""",
                (stack_id, monitor_name, now, now, req.webhook_url),
            )
        conn.execute(
            """INSERT INTO stack_updates (update_id, stack_id, source_job_id, status, created_at)
               VALUES (?, ?, ?, 'queued', ?)""",
            (update_id, stack_id, req.source_job_id, now),
        )
        conn.commit()

    background_tasks.add_task(execute_stack_update, stack_id, update_id, monitor_name, req)
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
            "SELECT update_id, source_job_id, status, created_at, finished_at, error_message "
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
    res["recent_updates"] = [dict(u) for u in updates]
    return res


@app.get("/stac/stacks/{stack_id}")
def get_stack_stac_item(stack_id: str):
    _validate_safe_id(stack_id, "stack_id")
    stac_path = DATA_DIR / "stacks" / stack_id / "stac_item.json"
    if not stac_path.exists():
        raise HTTPException(status_code=404, detail="Stack STAC item not ready or stack not found")
    return JSONResponse(
        content=json.loads(stac_path.read_text()),
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
def stream_stack_asset(stack_id: str, filename: str):
    _validate_safe_id(stack_id, "stack_id")
    if filename not in ALLOWED_STACK_ASSETS:
        raise HTTPException(status_code=400, detail="Invalid stack asset request")
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
