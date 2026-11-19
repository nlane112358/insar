"""Worker-side orchestration for native pairwise and stack compute."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from services.platform.domain.schemas import CreateJobRequest, StackUpdateRequest
from services.platform.pipeline.native import run_native_command
from services.platform.publishing.stac import build_pair_item as build_stac_item
from services.platform.publishing.stac import build_stack_item as build_stack_stac_item
from services.platform.publishing.webhooks import dispatch_event as dispatch_webhook
from services.platform.storage.database import get_db
from services.platform.storage.object_store import enabled as s3_enabled
from services.platform.storage.object_store import register_artifacts, upload_directory
from services.platform.workflows.state import get_steps, mark_step

ENGINE_BIN = Path(os.getenv("INSAR_ENGINE_BIN", "./build/insar_proto"))
STACK_BIN = Path(os.getenv("INSAR_STACK_BIN", "./build/insar_stack"))
DATA_DIR = Path(os.getenv("INSAR_DATA_DIR", "./api_data"))
BASE_URL = os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _gdal_input(path: str | Path) -> str:
    value = str(path)
    parsed = urlparse(value)
    if parsed.scheme == "s3" and parsed.netloc and parsed.path:
        return f"/vsis3/{parsed.netloc}/{parsed.path.lstrip('/')}"
    return value


def _register_ingested_pairs(monitor_name: str, engine_version: str, products_dir: Path) -> list[str]:
    """Expose ingestion outputs through the same STAC/job index as direct pair jobs."""
    from services.platform.publishing.stac import build_pair_item

    pair_ids: list[str] = []
    if not products_dir.is_dir():
        return pair_ids
    for pair_dir in sorted(path for path in products_dir.iterdir() if path.is_dir()):
        summary_path = pair_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict):
            continue
        identity = f"{monitor_name}:{pair_dir.name}:{engine_version}"
        pair_id = f"pair_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        item = build_stac_item(pair_id, monitor_name, finished_at, summary)
        (pair_dir / "stac_item.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        with get_db() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (job_id, monitor_name, mode, status, created_at, finished_at, summary_json,
                    engine_version, idempotency_key)
                   VALUES (?, ?, 'ingestion', 'completed', ?, ?, ?, ?, ?)
                   ON CONFLICT (job_id) DO UPDATE SET status = 'completed', finished_at = excluded.finished_at,
                       summary_json = excluded.summary_json, engine_version = excluded.engine_version""",
                (
                    pair_id,
                    monitor_name,
                    finished_at,
                    finished_at,
                    json.dumps({**summary, "ingestion_pair_dir": pair_dir.name}),
                    engine_version,
                    hashlib.sha256(identity.encode()).hexdigest(),
                ),
            )
            conn.commit()

        if s3_enabled():
            _persist_output_assets("job", pair_id, pair_dir)
        else:
            jobs_dir = DATA_DIR / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            alias = jobs_dir / pair_id
            if not alias.exists():
                alias.symlink_to(pair_dir.resolve(), target_is_directory=True)
        pair_ids.append(pair_id)
    return pair_ids


def execute_monitor_ingestion(ingestion_id: str, monitor_name: str, request: dict) -> None:
    """Execute one provider sync in the worker and publish its pair products."""
    from services.platform.catalog.ingest_service import _build_site_from_mapping
    from services.platform.pipeline.ingestion import run_site

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data_root = DATA_DIR / "sites" / monitor_name
    site_payload = dict(request.get("site_config") or {})
    site_payload.update(
        {
            "site_id": monitor_name,
            "work_dir": str(data_root),
            "stack_bin": str(STACK_BIN),
        }
    )
    active_step = "DiscoverPass"
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM ingestion_runs WHERE ingestion_id = ?", (ingestion_id,)
        ).fetchone()
        if row and row["status"] == "completed":
            return
        conn.execute(
            "UPDATE ingestion_runs SET status = 'running', error_message = NULL WHERE ingestion_id = ?",
            (ingestion_id,),
        )
        conn.commit()

    try:
        products_dir = data_root / "products" / monitor_name
        _hydrate_prior_ingested_products(monitor_name, products_dir)
        _hydrate_s3_assets(
            "stack",
            monitor_name,
            data_root / "stacks" / monitor_name,
            {"stack_state.tif", "stack_ledger.json", "corridor_timeseries.json", "stack_summary.json"},
        )
        site = _build_site_from_mapping(site_payload, {})
        result = run_site(site, data_root, ENGINE_BIN, workflow_id=ingestion_id)
        active_step = "PublishSTAC"
        mark_step(ingestion_id, "PublishSTAC", "running")
        pair_ids = _register_ingested_pairs(
            monitor_name,
            str(request.get("engine_version", "0.5.0")),
            data_root / "products" / monitor_name,
        )
        stack_id = None
        stack_dir = data_root / "stacks" / monitor_name
        stack_summary_path = stack_dir / "stack_summary.json"
        if stack_summary_path.is_file():
            stack_id = monitor_name
            stack_summary = json.loads(stack_summary_path.read_text(encoding="utf-8"))
            stack_finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            stack_item = build_stack_stac_item(stack_id, monitor_name, stack_finished_at, stack_summary, stack_dir)
            (stack_dir / "stac_item.json").write_text(json.dumps(stack_item, indent=2), encoding="utf-8")
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO stacks
                       (stack_id, monitor_name, status, pair_count, created_at, updated_at, summary_json)
                       VALUES (?, ?, 'completed', ?, ?, ?, ?)
                       ON CONFLICT (stack_id) DO UPDATE SET monitor_name = excluded.monitor_name,
                           status = 'completed', pair_count = excluded.pair_count,
                           updated_at = excluded.updated_at, summary_json = excluded.summary_json,
                           error_message = NULL""",
                    (
                        stack_id,
                        monitor_name,
                        int(stack_summary.get("pair_count", 0)),
                        stack_finished_at,
                        stack_finished_at,
                        json.dumps(stack_summary),
                    ),
                )
                conn.commit()
            _persist_output_assets("stack", stack_id, stack_dir, version=ingestion_id)
            if not s3_enabled():
                public_stack_dir = DATA_DIR / "stacks"
                public_stack_dir.mkdir(parents=True, exist_ok=True)
                public_stack_path = public_stack_dir / stack_id
                if not public_stack_path.exists():
                    public_stack_path.symlink_to(stack_dir.resolve(), target_is_directory=True)
        pair_item_links = [
            {"rel": "child", "href": f"{BASE_URL}/stac/items/{pair_id}", "type": "application/geo+json"}
            for pair_id in pair_ids
        ]
        summary = dict(vars(result))
        ingestion_dir = DATA_DIR / "ingestions" / ingestion_id
        ingestion_dir.mkdir(parents=True, exist_ok=True)
        stac_item = {
            "type": "Feature",
            "stac_version": "1.1.0",
            "id": ingestion_id,
            "geometry": None,
            "properties": {
                "datetime": now,
                "insar:monitor_name": monitor_name,
                "insar:engine_version": str(request.get("engine_version", "0.5.0")),
                "insar:discovered_count": result.discovered_count,
                "insar:pairs_processed": result.pairs_processed,
                "insar:stack_id": stack_id,
            },
            "links": [
                {"rel": "self", "href": f"{BASE_URL}/stac/ingestions/{ingestion_id}", "type": "application/geo+json"},
                *pair_item_links,
            ],
            "assets": {},
        }
        (ingestion_dir / "stac_item.json").write_text(json.dumps(stac_item, indent=2), encoding="utf-8")
        if s3_enabled():
            objects = upload_directory(ingestion_dir, f"ingestion/{ingestion_id}", version=ingestion_id)
            register_artifacts("ingestion", ingestion_id, objects)
        mark_step(ingestion_id, "PublishSTAC", "completed", detail={"pair_items": len(pair_ids)})

        active_step = "DispatchWebhooks"
        mark_step(ingestion_id, active_step, "running")
        webhook_status = None
        webhook_url = request.get("webhook_url")
        if webhook_url:
            webhook_status = dispatch_webhook(
                webhook_url,
                {
                    "event": "monitor.ingestion_completed",
                    "ingestion_id": ingestion_id,
                    "monitor_name": monitor_name,
                    "finished_at": now,
                    "summary": summary,
                    "stac_item_url": f"{BASE_URL}/stac/ingestions/{ingestion_id}",
                },
            )
            mark_step(ingestion_id, active_step, "completed", detail={"delivery_status": webhook_status})
        else:
            mark_step(ingestion_id, active_step, "skipped", detail={"reason": "no webhook configured"})

        with get_db() as conn:
            conn.execute(
                """UPDATE ingestion_runs SET status = 'completed', finished_at = ?, summary_json = ?,
                   error_message = NULL WHERE ingestion_id = ?""",
                (now, json.dumps({**summary, "pair_job_ids": pair_ids, "stack_id": stack_id, "webhook_status": webhook_status}), ingestion_id),
            )
            conn.commit()
    except Exception as exc:
        try:
            running = [step for step in get_steps(ingestion_id) if step["status"] == "running"]
            active_step = running[-1]["step_name"] if running else active_step
            mark_step(ingestion_id, active_step, "failed", error=str(exc))
        except Exception:
            pass
        with get_db() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET status = 'failed', finished_at = ?, error_message = ? WHERE ingestion_id = ?",
                (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), str(exc)[:4000], ingestion_id),
            )
            conn.commit()

def _build_proto_cmd(out_dir: Path, req: CreateJobRequest) -> list[str]:
    cmd = [
        str(ENGINE_BIN),
        "--slc1", _gdal_input(req.slc1_path),
        "--slc2", _gdal_input(req.slc2_path),
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


def _run_cpp(cmd: list[str], engine: Literal["pairwise", "stack"]) -> None:
    run_native_command(cmd, engine)


def _persist_output_assets(owner_type: str, owner_id: str, output_dir: Path, *, version: str | None = None) -> None:
    if not s3_enabled():
        return
    objects = upload_directory(output_dir, f"{owner_type}/{owner_id}", version=version or owner_id)
    summary_name = "summary.json" if owner_type == "job" else "stack_summary.json"
    summary_path = output_dir / summary_name
    footprint = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        bbox = summary.get("bbox_wgs84")
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            footprint = {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
                    [min_lon, max_lat], [min_lon, min_lat],
                ]],
            }
    register_artifacts(owner_type, owner_id, objects, footprint=footprint)


def _hydrate_s3_assets(
    owner_type: str,
    owner_id: str,
    destination: Path,
    asset_keys: set[str] | None,
) -> None:
    """Materialize selected prior artifacts from object storage for a stateless worker run."""
    if not s3_enabled():
        return
    from services.platform.storage.object_store import get_object

    with get_db() as conn:
        rows = conn.execute(
            "SELECT asset_key, s3_uri FROM assets WHERE owner_type = ? AND owner_id = ?",
            (owner_type, owner_id),
        ).fetchall()
    for row in rows:
        key = row["asset_key"]
        if asset_keys is not None and key not in asset_keys:
            continue
        response = get_object(row["s3_uri"])
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as output:
                while chunk := response["Body"].read(1024 * 1024):
                    output.write(chunk)
        finally:
            response["Body"].close()


def _hydrate_prior_ingested_products(monitor_name: str, products_dir: Path) -> None:
    """Restore prior pair products so a fresh worker can resume stack discovery."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT job_id, summary_json FROM jobs WHERE monitor_name = ? AND mode = 'ingestion' AND status = 'completed'",
            (monitor_name,),
        ).fetchall()
    for row in rows:
        try:
            summary = json.loads(row["summary_json"] or "{}")
        except json.JSONDecodeError:
            continue
        pair_name = summary.get("ingestion_pair_dir")
        if not pair_name:
            continue
        target = products_dir / str(pair_name)
        if (target / "summary.json").is_file():
            continue
        with get_db() as conn:
            assets = conn.execute(
                "SELECT asset_key FROM assets WHERE owner_type = 'job' AND owner_id = ?",
                (row["job_id"],),
            ).fetchall()
        _hydrate_s3_assets("job", row["job_id"], target, {asset["asset_key"] for asset in assets})


def execute_insar_job(job_id: str, req: CreateJobRequest) -> None:
    out_dir = DATA_DIR / "jobs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    active_step = "DiscoverPass"

    with get_db() as conn:
        existing = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if existing and existing["status"] == "completed":
            return
        conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
        conn.commit()

    cmd = _build_proto_cmd(out_dir, req)

    try:
        mark_step(job_id, "DiscoverPass", "completed", detail={"granule_id": req.granule_id})
        active_step = "FetchAuxData"
        mark_step(job_id, active_step, "running", detail={"slc1": req.slc1_path, "slc2": req.slc2_path})
        mark_step(job_id, active_step, "completed")
        active_step = "RunCompute"
        mark_step(job_id, active_step, "running", detail={"engine_version": req.engine_version})
        _run_cpp(cmd, "pairwise")
        mark_step(job_id, active_step, "completed")

        active_step = "ValidateAndScore"
        summary = json.loads((out_dir / "summary.json").read_text())
        if not isinstance(summary, dict):
            raise ValueError("Pairwise engine summary must be a JSON object")
        validation_metrics = {
            key: summary.get(key)
            for key in ("valid_pixels", "processing_time_ms", "anomalies_detected", "min_displacement_mm", "max_displacement_mm")
        }
        metric_id = hashlib.sha256(f"{job_id}:{req.engine_version}".encode()).hexdigest()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO validation_metrics (metric_id, monitor_name, job_id, measured_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (metric_id) DO UPDATE SET
                       measured_at = excluded.measured_at, metrics_json = excluded.metrics_json""",
                (metric_id, req.monitor_name, job_id, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), json.dumps(validation_metrics)),
            )
            conn.commit()
        mark_step(job_id, active_step, "completed", detail=validation_metrics)

        active_step = "PublishSTAC"
        mark_step(job_id, active_step, "running")
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        stac_item = build_stac_item(job_id, req.monitor_name, finished_at, summary)
        (out_dir / "stac_item.json").write_text(json.dumps(stac_item, indent=2))
        _persist_output_assets("job", job_id, out_dir)
        mark_step(job_id, active_step, "completed", detail={"asset_count": len(stac_item["assets"])})

        wh_status = None
        active_step = "DispatchWebhooks"
        mark_step(job_id, active_step, "running")
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
            mark_step(job_id, active_step, "completed", detail={"delivery_status": wh_status})
        else:
            mark_step(job_id, active_step, "skipped", detail={"reason": "no webhook configured"})

        with get_db() as conn:
            conn.execute(
                """UPDATE jobs
                   SET status = 'completed', finished_at = ?, summary_json = ?, webhook_status = ?
                   WHERE job_id = ?""",
                (finished_at, json.dumps(summary), wh_status, job_id),
            )
            conn.commit()

    except Exception as exc:
        try:
            mark_step(job_id, active_step, "failed", error=str(exc))
        except Exception:
            pass
        err_msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            err_msg = f"Engine exit {exc.returncode}: {(exc.stderr or '').strip()}"
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error_message = ? WHERE job_id = ?",
                (err_msg, job_id),
            )
            conn.commit()


def execute_stack_update(stack_id: str, update_id: str, monitor_name: str, req: StackUpdateRequest) -> None:
    stack_dir = DATA_DIR / "stacks" / stack_id
    stack_dir.mkdir(parents=True, exist_ok=True)
    _hydrate_s3_assets("stack", stack_id, stack_dir, {"stack_state.tif"})
    state_path = stack_dir / "stack_state.tif"
    active_step = "DiscoverPass"

    with get_db() as conn:
        conn.execute("UPDATE stacks SET status = 'running' WHERE stack_id = ?", (stack_id,))
        conn.execute("UPDATE stack_updates SET status = 'running' WHERE update_id = ?", (update_id,))
        conn.commit()

    try:
        mark_step(update_id, "DiscoverPass", "completed", detail={"source_job_id": req.source_job_id})
        active_step = "FetchAuxData"
        mark_step(update_id, active_step, "running", detail={"dem_path": req.dem_path, "corridors_path": req.corridors_path})
        mark_step(update_id, active_step, "completed")
        active_step = "RunCompute"
        mark_step(update_id, active_step, "running")
        disp_path = req.disp_path
        coh_path = req.coh_path
        source_job_id = req.source_job_id

        # Option A: run insar_proto inline first
        if req.pair_job is not None:
            pair_dir = stack_dir / "pairs" / update_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            proto_cmd = _build_proto_cmd(pair_dir, req.pair_job)
            _run_cpp(proto_cmd, "pairwise")
            disp_path = str(pair_dir / "displacement_mm.tif")
            coh_path = str(pair_dir / "coherence.tif")
        # Option B: pull from an existing completed /v1/jobs/{job_id}
        elif source_job_id:
            job_dir = DATA_DIR / "jobs" / source_job_id
            _hydrate_s3_assets("job", source_job_id, job_dir, {"displacement_mm.tif", "coherence.tif"})
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
            cmd += ["--disp", _gdal_input(disp_path), "--coh", _gdal_input(coh_path)]
            if req.t0_days is not None and req.t1_days is not None:
                cmd += ["--t0-days", str(req.t0_days), "--t1-days", str(req.t1_days)]
            elif req.dt_days is not None:
                cmd += ["--dt-days", str(req.dt_days)]
            if req.redundant_pair:
                cmd += ["--redundant-pair", "--pair-weight-scale", str(req.pair_weight_scale)]

        if req.dem_path:
            cmd += ["--dem", _gdal_input(req.dem_path)]
        if req.ref_lonlat is not None:
            cmd += [
                "--ref-lonlat", str(req.ref_lonlat[0]), str(req.ref_lonlat[1]),
                "--ref-radius-m", str(req.ref_radius_m),
            ]
        elif req.ref_px is not None:
            cmd += ["--ref-px", str(req.ref_px[0]), str(req.ref_px[1]), str(req.ref_px[2])]
        if req.ref_region_path:
            cmd += ["--ref-region", _gdal_input(req.ref_region_path)]
        if req.deform_mask_path:
            cmd += ["--deform-mask", _gdal_input(req.deform_mask_path)]
        if req.prior_vel_path:
            cmd += ["--prior-vel", _gdal_input(req.prior_vel_path)]
        if req.corridors_path:
            cmd += ["--corridors", req.corridors_path]
        for trip in req.closure_triplets:
            cmd += ["--closure-triplet", _gdal_input(trip.disp_a_path), _gdal_input(trip.disp_b_path), trip.mode]

        _run_cpp(cmd, "stack")
        mark_step(update_id, active_step, "completed")
        active_step = "ValidateAndScore"
        summary = json.loads((stack_dir / "stack_summary.json").read_text())
        if not isinstance(summary, dict):
            raise ValueError("Stack engine summary must be a JSON object")
        mark_step(update_id, active_step, "completed", detail={
            "pair_count": summary.get("pair_count"),
            "stack_mature": summary.get("maturity_gating", {}).get("stack_mature"),
            "corridor_alerts": summary.get("corridors", {}).get("corridor_alerts"),
        })
        active_step = "PublishSTAC"
        mark_step(update_id, active_step, "running")
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        stac_item = build_stack_stac_item(stack_id, monitor_name, finished_at, summary, stack_dir)
        (stack_dir / "stac_item.json").write_text(json.dumps(stac_item, indent=2))
        _persist_output_assets("stack", stack_id, stack_dir, version=update_id)
        mark_step(update_id, active_step, "completed", detail={"asset_count": len(stac_item["assets"])})

        wh_status = None
        active_step = "DispatchWebhooks"
        mark_step(update_id, active_step, "running")
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
            mark_step(update_id, active_step, "completed", detail={"delivery_status": wh_status})
        else:
            mark_step(update_id, active_step, "skipped", detail={"reason": "no webhook configured"})

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
        try:
            mark_step(update_id, active_step, "failed", error=str(exc))
        except Exception:
            pass
        err_msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            err_msg = f"Stack engine exit {exc.returncode}: {(exc.stderr or '').strip()}"
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


