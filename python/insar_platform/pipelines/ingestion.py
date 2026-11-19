#!/usr/bin/env python3
"""
Automated InSAR ingestion and C++ processor orchestrator.

Supports:
  - NASA/ASF OPERA CSLC-S1 bursts (via asf_search)
  - NASA-ISRO NISAR L-band GSLC products (via earthaccess + dynamic HDF5 wavelength)
  - Single-site CLI runs or multi-site configuration files (sites.json / sites.yaml)

Tracks granules and processed interferometric pairs in PostGIS (or local SQLite),
streams downloaded source granules to S3-compatible storage, and runs pixel compute
through the in-process libinsar binding when it is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

from insar_platform.paths import resolve_runtime_path
from insar_platform.providers.asf_opera import (
    _OPERA_BURST_HYPHEN_RE,
    _atomic_asf_download,
    _create_asf_session,
    _normalize_opera_burst_id,
    _resolve_asf_stream_target,
    discover_burst_id,
    fetch_opera_candidates,
)
from insar_platform.providers.nasa_nisar import (
    _NISAR_UR_PATTERN,
    _atomic_earthaccess_download,
    _extract_nisar_metadata,
    _extract_scalar_float,
    _get_umm_attr,
    _inspect_pair_metadata,
    _open_h5_source,
    _open_remote_h5,
    _pick_h5_links,
    _read_center_frequency_hz,
    _read_center_frequency_hz_from_h5file,
    _read_grid_geometry_from_h5file,
    _resolve_earthaccess_stream_target,
    _stream_subprocess_env,
    compute_window_from_bbox,
    fetch_nisar_candidates,
    read_nisar_wavelength_mm,
)
from insar_platform.providers.sensors import (
    NISAR_DEFAULT_SHORT_NAME,
    NISAR_DEFAULT_SUBDATASET,
    NISAR_DEFAULT_WAVELENGTH_MM,
    OPERA_DEFAULT_SUBDATASET,
    OPERA_DEFAULT_WAVELENGTH_MM,
    SPEED_OF_LIGHT_M_S,
    GranuleCandidate,
    StreamTarget,
    _detect_aws_in_region,
    _extract_bearer_token,
    _format_gdal_hdf5_uri,
    _import_asf,
    _import_earthaccess,
    _import_fsspec,
    _import_h5py,
    _login_earthaccess,
    _normalize_direction,
    _normalize_timestamp,
    _resolve_https_redirect,
    _utc_now,
)
from insar_platform.storage.database import connect_db, database_url, init_db as init_platform_db


@dataclass(frozen=True)
class SiteConfig:
    """Configuration for a single monitored InSAR site."""

    site_id: str = ""
    sensor: str = "opera"
    burst_id: str | None = None
    lon: float = -99.1332
    lat: float = 19.4326
    bbox: tuple[float, float, float, float] | None = None
    short_name: str = NISAR_DEFAULT_SHORT_NAME
    track: int | None = None
    frame: int | None = None
    direction: str | None = None
    subdataset: str | None = None
    wavelength_mm: float | None = None
    lookback_days: int = 45
    max_granules: int = 6
    stride_x: int = 6
    stride_y: int = 3
    look_rx: int | None = None
    look_ry: int | None = None
    iters: int | None = None
    window: tuple[int, int, int, int] | None = None
    alert_mm: float | None = None
    alert_coh: float | None = None
    prune_raw: bool = False
    stream: bool = False
    stream_transport: str = "auto"
    work_dir: Path | None = None
    stack: bool = False
    stack_bin: Path = Path("./build/insar_stack")
    corridors: Path | None = None
    buffer_m: float = 100.0
    ramp_order: int = 1
    vel_alert_mm_yr: float = 15.0
    accel_alert_mm_yr2: float = 25.0
    min_alert_snr: float = 2.5
    min_alert_span_days: float = 60.0
    dem: Path | None = None
    ref_lonlat: tuple[float, float] | None = None
    ref_radius_m: float = 1000.0
    ref_region: Path | None = None
    deform_mask: Path | None = None
    two_pass_deform_mask: bool = False
    max_skip: int = 0
    closure_thresh_mm: float = 10.0
    closure_max_bad_frac: float = 0.35
    closure_mode: str = "consensus"
    include_redundant: bool = True

    def resolved_subdataset(self) -> str:
        """Return explicit subdataset or the sensor-appropriate default."""
        if self.subdataset:
            return self.subdataset
        return (
            NISAR_DEFAULT_SUBDATASET
            if self.sensor == "nisar"
            else OPERA_DEFAULT_SUBDATASET
        )


@dataclass(frozen=True)
class SiteRunSummary:
    """Execution summary for one sync_site pass."""

    site_id: str
    sensor: str
    discovered_count: int
    downloaded_count: int
    pairs_processed: int
    pairs_skipped: int
    pruned_count: int
    stack_pairs: int = 0
    stack_new_pairs: int = 0


def _compact_timestamp(iso_ts: str) -> str:
    """Format an ISO timestamp compactly for collision-free same-day pair tags."""
    return re.sub(r"[^0-9T]", "", iso_ts[:19])


def _build_pair_tag(stream_key: str, ref_t: str, sec_t: str) -> str:
    """Build deterministic pair directory name, disambiguating same-day passes."""
    if ref_t[:10] == sec_t[:10]:
        return f"{stream_key}_{_compact_timestamp(ref_t)}_{_compact_timestamp(sec_t)}"
    return f"{stream_key}_{ref_t[:10]}_{sec_t[:10]}"


def _sanitize_site_id(site_id: str) -> str:
    """Validate that a multi-site ID is safe to use as a subdirectory name."""
    cleaned = site_id.strip()
    if not cleaned:
        raise ValueError("site_id must be a non-empty string in multi-site config")
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Unsafe site_id '{site_id}': must not contain path separators or '..'")
    return cleaned


def _ensure_table_columns(
    conn: Any,
    table: str,
    columns: Sequence[tuple[str, str]],
) -> set[str]:
    """Add missing columns to an existing SQLite table idempotently."""
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    added: set[str] = set()
    for col_name, col_def in columns:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
            added.add(col_name)
    return added


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Use PostGIS in platform deployments and SQLite for standalone local runs."""
    db_str = str(db_path)
    if db_str != ":memory:" and database_url():
        init_platform_db()
        return connect_db()  # type: ignore[return-value]
    if db_str != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, timeout=30.0)
    conn.row_factory = sqlite3.Row
    if db_str != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS acquisitions (
            granule_id TEXT PRIMARY KEY,
            monitor_name TEXT NOT NULL,
            sensor TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            asset_uri TEXT,
            checksum TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS granules (
            granule_id  TEXT NOT NULL,
            burst_id    TEXT NOT NULL,
            acq_time    TEXT NOT NULL,
            local_path  TEXT NOT NULL,
            sensor      TEXT NOT NULL DEFAULT 'opera',
            site_id     TEXT NOT NULL DEFAULT '',
            stream_key  TEXT NOT NULL DEFAULT '',
            pruned      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (site_id, granule_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_pairs (
            ref_id        TEXT NOT NULL,
            sec_id        TEXT NOT NULL,
            out_dir       TEXT NOT NULL,
            finished_at   TEXT NOT NULL,
            site_id       TEXT NOT NULL DEFAULT '',
            stream_key    TEXT NOT NULL DEFAULT '',
            wavelength_mm REAL,
            PRIMARY KEY (site_id, ref_id, sec_id)
        )
    """)

    added_granule_cols = _ensure_table_columns(
        conn,
        "granules",
        [
            ("sensor", "TEXT NOT NULL DEFAULT 'opera'"),
            ("site_id", "TEXT NOT NULL DEFAULT ''"),
            ("stream_key", "TEXT NOT NULL DEFAULT ''"),
            ("pruned", "INTEGER NOT NULL DEFAULT 0"),
        ],
    )
    if "stream_key" in added_granule_cols:
        conn.execute("UPDATE granules SET stream_key = burst_id WHERE stream_key = ''")

    _ensure_table_columns(
        conn,
        "processed_pairs",
        [
            ("site_id", "TEXT NOT NULL DEFAULT ''"),
            ("stream_key", "TEXT NOT NULL DEFAULT ''"),
            ("wavelength_mm", "REAL"),
        ],
    )
    conn.commit()
    return conn


def _record_acquisition(
    conn: Any,
    site: SiteConfig,
    candidate: GranuleCandidate,
    local_ref: str,
) -> None:
    """Record provider provenance and archive downloaded SLCs as content-addressed objects."""
    asset_uri = candidate.stream_url or local_ref
    checksum = None
    if not local_ref.startswith("stream://"):
        local_path = Path(local_ref)
        if local_path.is_file():
            try:
                from insar_platform.storage.object_store import enabled as s3_enabled
            except ImportError:
                s3_enabled = lambda: False
            if s3_enabled():
                from insar_platform.storage.object_store import ensure_bucket, upload_file

                digest = hashlib.sha256()
                with local_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                checksum = digest.hexdigest()
                granule_key = hashlib.sha256(candidate.granule_id.encode("utf-8")).hexdigest()
                filename = Path(candidate.filename).name
                object_key = f"acquisitions/{granule_key}/{checksum}/{filename}"
                previous = conn.execute(
                    "SELECT asset_uri, checksum FROM acquisitions WHERE granule_id = ?",
                    (candidate.granule_id,),
                ).fetchone()
                if previous and previous["checksum"] == checksum and str(previous["asset_uri"] or "").startswith("s3://"):
                    asset_uri = previous["asset_uri"]
                else:
                    ensure_bucket()
                    stored = upload_file(local_path, object_key)
                    asset_uri = stored.uri
                    checksum = stored.checksum
            else:
                asset_uri = str(local_path)
    metadata = json.dumps(
        {
            "granule_id": candidate.granule_id,
            "stream_key": candidate.stream_key,
            "filename": candidate.filename,
            "provider_url": candidate.stream_url or None,
        },
        separators=(",", ":"),
    )
    conn.execute(
        """INSERT INTO acquisitions
           (granule_id, monitor_name, sensor, acquired_at, metadata_json, asset_uri, checksum)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (granule_id) DO UPDATE SET monitor_name = excluded.monitor_name,
               sensor = excluded.sensor, acquired_at = excluded.acquired_at,
               metadata_json = excluded.metadata_json, asset_uri = excluded.asset_uri,
               checksum = excluded.checksum""",
        (candidate.granule_id, site.site_id, site.sensor, candidate.acq_time, metadata, asset_uri, checksum),
    )


def build_engine_command(
    engine_bin: Path,
    ref_path: str,
    sec_path: str,
    out_dir: Path,
    site: SiteConfig,
    wavelength_mm: float | None,
) -> list[str]:
    """Assemble the CLI invocation for ./build/insar_proto."""
    cmd = [
        str(engine_bin),
        "--slc1", ref_path,
        "--slc2", sec_path,
        "--subdataset", site.resolved_subdataset(),
        "--stride-x", str(site.stride_x),
        "--stride-y", str(site.stride_y),
        "--out-dir", str(out_dir),
    ]
    if wavelength_mm is not None:
        cmd.extend(["--wavelength-mm", f"{wavelength_mm:.6f}"])
    if site.window is not None:
        cmd.extend(["--window", *(str(v) for v in site.window)])
    if site.look_rx is not None:
        cmd.extend(["--look-rx", str(site.look_rx)])
    if site.look_ry is not None:
        cmd.extend(["--look-ry", str(site.look_ry)])
    if site.iters is not None:
        cmd.extend(["--iters", str(site.iters)])
    if site.alert_mm is not None:
        cmd.extend(["--alert-mm", str(site.alert_mm)])
    if site.alert_coh is not None:
        cmd.extend(["--alert-coh", str(site.alert_coh)])
    if site.ref_lonlat is not None:
        cmd.extend([
            "--ref-lonlat",
            f"{site.ref_lonlat[0]:.8f}",
            f"{site.ref_lonlat[1]:.8f}",
            "--ref-radius-m",
            f"{site.ref_radius_m:.2f}",
        ])
    return cmd


def _validate_engine_binary(engine_bin: Path) -> None:
    """Verify the compiled insar_proto binary exists and is executable."""
    if os.getenv("INSAR_COMPUTE_BACKEND", "auto").lower() != "subprocess":
        from insar_platform.pipelines.native import native_available

        if native_available():
            return
        if os.getenv("INSAR_COMPUTE_BACKEND", "auto").lower() == "libinsar":
            raise RuntimeError("INSAR_COMPUTE_BACKEND=libinsar but _insar_core is unavailable")
    if "/" in str(engine_bin) or os.sep in str(engine_bin) or engine_bin.exists():
        if not engine_bin.is_file() or not os.access(engine_bin, os.X_OK):
            raise FileNotFoundError(
                f"InSAR engine binary not found or not executable: {engine_bin}. "
                "Build it with: cmake -B build && cmake --build build"
            )
    elif shutil.which(str(engine_bin)) is None:
        raise FileNotFoundError(
            f"InSAR engine binary '{engine_bin}' not found on PATH."
        )


def _resolve_pair_endpoint(
    granule_id: str,
    local_path: str,
    site: SiteConfig,
    cand_by_id: Mapping[str, GranuleCandidate],
    subdataset: str,
    *,
    dry_run: bool,
) -> tuple[str, Path | StreamTarget, StreamTarget | None] | None:
    """Resolve a pair input into either a local file path or a streaming GDAL target."""
    if not str(local_path).startswith("stream://") and Path(local_path).exists():
        p = Path(local_path)
        return (str(p), p, None)

    if not site.stream:
        if dry_run and not str(local_path).startswith("stream://"):
            p = Path(local_path)
            return (str(p), p, None)
        return None

    cand = cand_by_id.get(granule_id)
    if cand is None or cand.stream_fn is None:
        return None

    if dry_run:
        preview_base = cand.stream_url or cand.filename
        if site.stream_transport == "s3" and preview_base.startswith("s3://"):
            vsi = f"/vsis3/{preview_base[len('s3://'):]}"
        else:
            vsi = f"/vsicurl/{preview_base}"
        uri = _format_gdal_hdf5_uri(vsi, subdataset)
        return (uri, Path(local_path), None)

    target = cand.stream_fn(site.stream_transport)
    return (target.gdal_uri, target, target)


def _deduplicate_same_day_candidates(
    candidates: Sequence[GranuleCandidate],
) -> list[GranuleCandidate]:
    """Keep the newest granule per (stream_key, acquisition_date) so reprocessed products never self-pair."""
    latest_by_day: dict[tuple[str, str], GranuleCandidate] = {}
    for cand in sorted(candidates, key=lambda c: (c.acq_time, c.granule_id)):
        key = (cand.stream_key, cand.acq_time[:10])
        latest_by_day[key] = cand
    deduped = list(latest_by_day.values())
    deduped.sort(key=lambda c: (c.acq_time, c.granule_id))
    return deduped


def _ensure_site_dem(
    site: SiteConfig,
    effective_work_dir: Path,
    out_root: Path,
) -> Path | None:
    """Auto-fetch and align Copernicus GLO-30 DEM if site.dem is configured but missing on disk."""
    if site.dem is None:
        return None
    dem_path = resolve_runtime_path(site.dem)
    if dem_path.is_file():
        return dem_path

    ref_raster = next(
        (
            d / "displacement_mm.tif"
            for d in sorted(out_root.iterdir())
            if d.is_dir() and (d / "displacement_mm.tif").is_file()
        ),
        None,
    )
    if ref_raster is None:
        return dem_path

    from insar_platform.providers.cop30_dem import (
        build_aligned_dem,
        download_cop30_tiles,
        read_raster_grid_and_wgs84_bbox,
        tiles_for_bbox,
    )

    print(f"[{site.site_id}] Fetching & aligning Copernicus GLO-30 DEM -> {dem_path} ...")
    ref_grid = read_raster_grid_and_wgs84_bbox(ref_raster)
    tiles = tiles_for_bbox(*ref_grid["wgs84_bbox"])  # type: ignore[misc]
    sources = download_cop30_tiles(tiles, effective_work_dir / "dem" / "tiles")
    return build_aligned_dem(sources, dem_path, ref_grid=ref_grid)


def _run_site_stack(
    site: SiteConfig,
    effective_work_dir: Path,
    out_root: Path,
    engine_bin: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Run O(1) incremental stacking on a site's processed product directory."""
    stack_dir = (
        effective_work_dir / "stacks" / site.site_id
        if site.site_id
        else effective_work_dir / "stacks"
    )
    if dry_run:
        print(
            f"[DRY-RUN] Would update sequential stack in {stack_dir} "
            f"(ramp_order={site.ramp_order}, corridors={site.corridors}, buffer_m={site.buffer_m})"
        )
        return 0, 0

    from insar_platform.pipelines.stacking import StackConfig, print_corridor_leaderboard, sync_stack

    if not out_root.exists():
        return 0, 0
    has_pairs = any(
        d.is_dir() and (d / "displacement_mm.tif").is_file() and (d / "coherence.tif").is_file()
        for d in out_root.iterdir()
    )
    if not has_pairs:
        return 0, 0

    resolved_dem = _ensure_site_dem(site, effective_work_dir, out_root)
    stack_bin = (
        site.stack_bin
        if site.stack_bin != Path("./build/insar_stack")
        else (engine_bin.parent / "insar_stack")
    )
    cfg = StackConfig(
        products_dir=out_root,
        out_dir=stack_dir,
        stack_bin=stack_bin,
        ramp_order=site.ramp_order,
        dem=resolved_dem,
        ref_lonlat=site.ref_lonlat,
        ref_radius_m=site.ref_radius_m,
        ref_region=site.ref_region,
        deform_mask=site.deform_mask,
        two_pass_deform_mask=site.two_pass_deform_mask,
        closure_thresh_mm=site.closure_thresh_mm,
        closure_max_bad_frac=site.closure_max_bad_frac,
        closure_mode=site.closure_mode,
        include_redundant=site.include_redundant,
        corridors=site.corridors,
        buffer_m=site.buffer_m,
        vel_alert_mm_yr=site.vel_alert_mm_yr,
        accel_alert_mm_yr2=site.accel_alert_mm_yr2,
        min_alert_snr=site.min_alert_snr,
        min_alert_span_days=site.min_alert_span_days,
    )
    summary = sync_stack(cfg)
    if site.corridors is not None:
        print_corridor_leaderboard(stack_dir, top_n=8)
    return summary.total_pairs_in_chain, summary.newly_stacked


def sync_site(
    site: SiteConfig,
    work_dir: Path,
    engine_bin: Path,
    *,
    dry_run: bool = False,
    asf_mod: Any = None,
    ea_mod: Any = None,
    h5py_mod: Any = None,
    fsspec_mod: Any = None,
    _catalog_candidates: list[GranuleCandidate] | None = None,
    _workflow_id: str | None = None,
) -> SiteRunSummary:
    """Synchronize new granules and process consecutive InSAR pairs for one site."""
    effective_work_dir = site.work_dir or work_dir
    if not dry_run:
        _validate_engine_binary(engine_bin)

    raw_dir = effective_work_dir / "raw" / site.site_id if site.site_id else effective_work_dir / "raw"
    out_root = (
        effective_work_dir / "products" / site.site_id
        if site.site_id
        else effective_work_dir / "products"
    )
    if not dry_run:
        if not site.stream:
            raw_dir.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)

    if _catalog_candidates is not None:
        resolved_site, candidates = site, _catalog_candidates
    else:
        from insar_platform.providers.sensors import sensor_provider

        provider = sensor_provider(site.sensor, asf_module=asf_mod, earthaccess_module=ea_mod)
        resolved_site, candidates = provider.discover(site)

    candidates = _deduplicate_same_day_candidates(candidates)
    print(f"Found {len(candidates)} granules in window.")

    cand_by_id = {cand.granule_id: cand for cand in candidates}
    db_path = effective_work_dir / "state.db"
    if dry_run and not db_path.exists():
        conn = init_db(":memory:")
    else:
        conn = init_db(db_path)

    downloaded_count = 0
    pairs_processed = 0
    pairs_skipped = 0
    pruned_count = 0
    pending_dry_run_cands: list[GranuleCandidate] = []
    if _workflow_id:
        from insar_platform.workflows.state import mark_step

        mark_step(_workflow_id, "FetchAuxData", "running", detail={"granule_count": len(candidates)})

    try:
        # 1. Register streamed granules or download any unseen (and unpruned) granules
        for cand in candidates:
            row = conn.execute(
                "SELECT local_path, pruned FROM granules WHERE site_id = ? AND granule_id = ?",
                (resolved_site.site_id, cand.granule_id),
            ).fetchone()
            if row:
                if row["pruned"]:
                    continue
                existing_path = str(row["local_path"])
                if resolved_site.stream and existing_path.startswith("stream://"):
                    continue
                if Path(existing_path).exists():
                    _record_acquisition(conn, resolved_site, cand, existing_path)
                    conn.commit()
                    continue

            if dry_run:
                if resolved_site.stream:
                    print(
                        f"[DRY-RUN] Would stream ({resolved_site.stream_transport}) "
                        f"{cand.filename} ({cand.acq_time})"
                    )
                else:
                    print(f"[DRY-RUN] Would download {cand.filename} ({cand.acq_time}) -> {raw_dir}")
                pending_dry_run_cands.append(cand)
                continue

            if resolved_site.stream:
                local_ref = f"stream://{cand.granule_id}"
            else:
                archived = conn.execute(
                    "SELECT asset_uri FROM acquisitions WHERE granule_id = ?",
                    (cand.granule_id,),
                ).fetchone()
                stored_uri = str(archived["asset_uri"] or "") if archived else ""
                try:
                    from insar_platform.storage.object_store import enabled as s3_enabled
                except ImportError:
                    s3_enabled = lambda: False
                if s3_enabled() and stored_uri.startswith("s3://"):
                    from insar_platform.storage.object_store import download_file

                    target_path = raw_dir / Path(cand.filename).name
                    print(f"Restoring {cand.filename} from object storage...")
                    local_ref = str(download_file(stored_uri, target_path))
                else:
                    print(f"Downloading {cand.filename} ({cand.acq_time})...")
                    local_ref = str(cand.download_fn(raw_dir))
                downloaded_count += 1

            conn.execute(
                """
                INSERT INTO granules
                    (granule_id, burst_id, acq_time, local_path, sensor, site_id, stream_key, pruned)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT (site_id, granule_id) DO UPDATE SET
                    burst_id = excluded.burst_id, acq_time = excluded.acq_time,
                    local_path = excluded.local_path, sensor = excluded.sensor,
                    stream_key = excluded.stream_key, pruned = 0
                """,
                (
                    cand.granule_id,
                    cand.stream_key,
                    cand.acq_time,
                    local_ref,
                    resolved_site.sensor,
                    resolved_site.site_id,
                    cand.stream_key,
                ),
            )
            _record_acquisition(conn, resolved_site, cand, local_ref)
            conn.commit()

        if _workflow_id:
            mark_step(_workflow_id, "FetchAuxData", "completed", detail={"downloaded_count": downloaded_count})
            mark_step(_workflow_id, "RunCompute", "running")

        # 2. Fetch chronological granules grouped by compatible geometric stream
        if resolved_site.sensor == "opera" and not resolved_site.site_id and resolved_site.burst_id:
            rows = conn.execute(
                """
                SELECT granule_id, stream_key, acq_time, local_path, pruned
                FROM granules
                WHERE site_id = ? AND stream_key = ?
                ORDER BY acq_time ASC, granule_id ASC
                """,
                (resolved_site.site_id, resolved_site.burst_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT granule_id, stream_key, acq_time, local_path, pruned
                FROM granules
                WHERE site_id = ?
                ORDER BY stream_key ASC, acq_time ASC, granule_id ASC
                """,
                (resolved_site.site_id,),
            ).fetchall()

        streams: dict[str, list[dict[str, Any]]] = {}
        seen_ids: set[str] = set()
        for row in rows:
            entry = {
                "granule_id": row["granule_id"],
                "stream_key": row["stream_key"],
                "acq_time": row["acq_time"],
                "local_path": row["local_path"],
                "pruned": bool(row["pruned"]),
                "simulated": False,
            }
            streams.setdefault(row["stream_key"], []).append(entry)
            seen_ids.add(row["granule_id"])

        if dry_run:
            for cand in pending_dry_run_cands:
                if cand.granule_id in seen_ids:
                    continue
                dry_path = f"stream://{cand.granule_id}" if resolved_site.stream else str(raw_dir / cand.filename)
                streams.setdefault(cand.stream_key, []).append(
                    {
                        "granule_id": cand.granule_id,
                        "stream_key": cand.stream_key,
                        "acq_time": cand.acq_time,
                        "local_path": dry_path,
                        "pruned": False,
                        "simulated": True,
                    }
                )
            for stream_rows in streams.values():
                stream_rows.sort(key=lambda r: (r["acq_time"], r["granule_id"]))

        # 3. Form consecutive pairs (t_k -> t_{k+1}) plus optional redundant skip-1 / skip-2 pairs per stream
        subdataset = resolved_site.resolved_subdataset()
        max_step = max(1, 1 + max(0, int(resolved_site.max_skip)))
        for stream_key, stream_rows in streams.items():
            # Deduplicate historical rows by acquisition date per stream
            deduped_rows: dict[str, dict[str, Any]] = {}
            for r in stream_rows:
                deduped_rows[str(r["acq_time"])[:10]] = r
            ordered_rows = [deduped_rows[d] for d in sorted(deduped_rows.keys())]

            pair_schedule: list[tuple[dict[str, Any], dict[str, Any], int]] = []
            for step in range(1, max_step + 1):
                for idx in range(len(ordered_rows) - step):
                    pair_schedule.append((ordered_rows[idx], ordered_rows[idx + step], step - 1))

            for ref_row, sec_row, _skip_level in pair_schedule:
                ref_id = ref_row["granule_id"]
                sec_id = sec_row["granule_id"]
                ref_t = ref_row["acq_time"]
                sec_t = sec_row["acq_time"]
                ref_path = ref_row["local_path"]
                sec_path = sec_row["local_path"]

                done = conn.execute(
                    """
                    SELECT 1 FROM processed_pairs
                    WHERE site_id = ? AND ref_id = ? AND sec_id = ?
                    """,
                    (resolved_site.site_id, ref_id, sec_id),
                ).fetchone()
                if done:
                    pairs_skipped += 1
                    continue

                if dry_run and (ref_row["pruned"] or sec_row["pruned"]):
                    print(f"[DRY-RUN] Skipping {ref_id} -> {sec_id}: pruned raw granule.")
                    pairs_skipped += 1
                    continue

                ref_ep = _resolve_pair_endpoint(
                    ref_id, ref_path, resolved_site, cand_by_id, subdataset, dry_run=dry_run
                )
                sec_ep = _resolve_pair_endpoint(
                    sec_id, sec_path, resolved_site, cand_by_id, subdataset, dry_run=dry_run
                )
                if ref_ep is None or sec_ep is None:
                    reason = "stream granule outside search window" if resolved_site.stream else "missing raw file on disk"
                    print(f"Skipping {ref_id} -> {sec_id}: {reason}.")
                    pairs_skipped += 1
                    continue

                ref_arg, ref_source, ref_target = ref_ep
                sec_arg, sec_source, sec_target = sec_ep
                pair_tag = _build_pair_tag(stream_key, ref_t, sec_t)
                pair_out = out_root / pair_tag

                if dry_run:
                    wavelength_mm = resolved_site.wavelength_mm
                    if wavelength_mm is None and resolved_site.sensor == "nisar" and isinstance(ref_source, Path) and ref_source.exists():
                        wavelength_mm = read_nisar_wavelength_mm(
                            ref_source, subdataset, sec_path=sec_source if isinstance(sec_source, Path) else None, h5py_mod=h5py_mod
                        )
                    cmd = build_engine_command(
                        engine_bin=engine_bin,
                        ref_path=ref_arg,
                        sec_path=sec_arg,
                        out_dir=pair_out,
                        site=resolved_site,
                        wavelength_mm=wavelength_mm,
                    )
                    print(f"[DRY-RUN] Would run C++ engine: {pair_tag} ({' '.join(cmd)})")
                    continue

                wavelength_mm, effective_window = _inspect_pair_metadata(
                    ref_source,
                    sec_source,
                    resolved_site,
                    subdataset,
                    h5py_mod=h5py_mod,
                    fsspec_mod=fsspec_mod,
                )
                if resolved_site.stream and resolved_site.bbox is not None and effective_window is None:
                    print(f"Skipping {ref_id} -> {sec_id}: bbox does not overlap granule grid.")
                    pairs_skipped += 1
                    continue

                effective_site = (
                    resolved_site
                    if effective_window == resolved_site.window
                    else SiteConfig(**{**resolved_site.__dict__, "window": effective_window})
                )
                cmd = build_engine_command(
                    engine_bin=engine_bin,
                    ref_path=ref_arg,
                    sec_path=sec_arg,
                    out_dir=pair_out,
                    site=effective_site,
                    wavelength_mm=wavelength_mm,
                )

                pair_out.mkdir(parents=True, exist_ok=True)
                wl_info = f", lambda={wavelength_mm:.3f} mm" if wavelength_mm is not None else ""
                win_info = f", window={effective_window}" if effective_window is not None else ""
                mode_info = " [stream]" if (ref_target or sec_target) else ""
                print(f"Running C++ engine{mode_info}: {pair_tag} (subdataset={subdataset}{wl_info}{win_info})")

                from insar_platform.pipelines.native import run_native_command

                if ref_target is not None or sec_target is not None:
                    with _stream_subprocess_env(ref_target, sec_target) as stream_env:
                        run_native_command(cmd, "pairwise", environment=stream_env)
                else:
                    run_native_command(cmd, "pairwise")
                pairs_processed += 1

                conn.execute(
                    """
                    INSERT INTO processed_pairs
                        (ref_id, sec_id, out_dir, finished_at, site_id, stream_key, wavelength_mm)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (site_id, ref_id, sec_id) DO UPDATE SET
                        out_dir = excluded.out_dir, finished_at = excluded.finished_at,
                        stream_key = excluded.stream_key, wavelength_mm = excluded.wavelength_mm
                    """,
                    (
                        ref_id,
                        sec_id,
                        str(pair_out),
                        _utc_now().isoformat(),
                        resolved_site.site_id,
                        stream_key,
                        wavelength_mm,
                    ),
                )
                conn.commit()

            # 4. Optional disk hygiene: keep newest (max_skip + 1) raw .h5 per stream, prune only fully forward-paired local files
            keep_tail = max_step
            if resolved_site.prune_raw and not dry_run and len(ordered_rows) > keep_tail:
                for idx, old_row in enumerate(ordered_rows[:-keep_tail]):
                    old_id = old_row["granule_id"]
                    old_local = str(old_row["local_path"])
                    if old_local.startswith("stream://"):
                        continue
                    expected_secs = [
                        ordered_rows[idx + step]["granule_id"]
                        for step in range(1, max_step + 1)
                        if idx + step < len(ordered_rows)
                    ]
                    all_paired = True
                    for expected_sec_id in expected_secs:
                        paired_forward = conn.execute(
                            """
                            SELECT 1 FROM processed_pairs
                            WHERE site_id = ? AND ref_id = ? AND sec_id = ?
                            """,
                            (resolved_site.site_id, old_id, expected_sec_id),
                        ).fetchone()
                        if not paired_forward:
                            all_paired = False
                            break
                    if not all_paired:
                        continue

                    p = Path(old_local)
                    if p.exists():
                        print(f"Pruning processed raw granule: {p.name}")
                        p.unlink()
                        pruned_count += 1
                    conn.execute(
                        "UPDATE granules SET pruned = 1 WHERE site_id = ? AND granule_id = ?",
                        (resolved_site.site_id, old_id),
                    )
                conn.commit()
    finally:
        conn.close()

    stack_pairs = 0
    stack_new_pairs = 0
    if resolved_site.stack:
        stack_pairs, stack_new_pairs = _run_site_stack(
            resolved_site,
            effective_work_dir,
            out_root,
            engine_bin,
            dry_run=dry_run,
        )

    if _workflow_id:
        from insar_platform.workflows.state import mark_step

        mark_step(_workflow_id, "RunCompute", "completed", detail={"pairs_processed": pairs_processed})
        mark_step(_workflow_id, "ValidateAndScore", "completed", detail={"pairs_skipped": pairs_skipped})

    return SiteRunSummary(
        site_id=resolved_site.site_id or (resolved_site.burst_id or resolved_site.sensor),
        sensor=resolved_site.sensor,
        discovered_count=len(candidates),
        downloaded_count=downloaded_count,
        pairs_processed=pairs_processed,
        pairs_skipped=pairs_skipped,
        pruned_count=pruned_count,
        stack_pairs=stack_pairs,
        stack_new_pairs=stack_new_pairs,
    )


def sync_and_process(
    burst_id: str,
    work_dir: Path,
    engine_bin: Path,
    lookback_days: int,
    max_granules: int,
    stride_x: int,
    stride_y: int,
    prune_old_h5: bool,
    *,
    asf_mod: Any = None,
) -> SiteRunSummary:
    """Backwards-compatible single-burst OPERA CSLC-S1 entry point."""
    site = SiteConfig(
        sensor="opera",
        burst_id=burst_id,
        lookback_days=lookback_days,
        max_granules=max_granules,
        stride_x=stride_x,
        stride_y=stride_y,
        prune_raw=prune_old_h5,
    )
    return sync_site(
        site=site,
        work_dir=work_dir,
        engine_bin=engine_bin,
        asf_mod=asf_mod,
    )


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if len(raw) != 4:
        raise ValueError(f"bbox must have 4 numbers [min_lon, min_lat, max_lon, max_lat], got {raw}")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in raw)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox coordinates {raw}: min must be < max")
    return (min_lon, min_lat, max_lon, max_lat)


def _parse_window(raw: Any) -> tuple[int, int, int, int] | None:
    if raw is None:
        return None
    if len(raw) != 4:
        raise ValueError(f"window must have 4 integers [x, y, w, h], got {raw}")
    x, y, w, h = (int(v) for v in raw)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid window {raw}: x,y >= 0 and w,h > 0 required")
    return (x, y, w, h)


def _parse_ref_lonlat(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    if len(raw) != 2:
        raise ValueError(f"ref_lonlat must have 2 numbers [lon, lat], got {raw}")
    lon, lat = float(raw[0]), float(raw[1])
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid ref_lonlat coordinates {raw}")
    return (lon, lat)


def _build_site_from_mapping(
    entry: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> SiteConfig:
    merged = {**defaults, **entry}
    site_id = _sanitize_site_id(str(merged.get("site_id") or merged.get("id") or ""))
    sensor = str(merged.get("sensor", "opera")).strip().lower()
    if sensor not in {"opera", "nisar"}:
        raise ValueError(f"Site '{site_id}': unsupported sensor '{sensor}'")

    stride_x = int(merged.get("stride_x", 6))
    stride_y = int(merged.get("stride_y", 3))
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"Site '{site_id}': stride_x and stride_y must be positive")

    stream_transport = str(merged.get("stream_transport", "auto")).strip().lower()
    if stream_transport not in {"auto", "https", "s3"}:
        raise ValueError(
            f"Site '{site_id}': unsupported stream_transport '{stream_transport}'; "
            "expected 'auto', 'https', or 's3'"
        )

    max_skip = int(merged.get("max_skip", 0))
    if max_skip < 0 or max_skip > 2:
        raise ValueError(f"Site '{site_id}': max_skip must be 0, 1, or 2 (got {max_skip})")

    closure_mode = str(merged.get("closure_mode", "consensus")).strip().lower()
    if closure_mode not in {"consensus", "strict"}:
        raise ValueError(f"Site '{site_id}': closure_mode must be 'consensus' or 'strict'")

    work_dir_raw = merged.get("work_dir")
    corridors_raw = merged.get("corridors")
    stack_bin_raw = merged.get("stack_bin")
    dem_raw = merged.get("dem")
    ref_region_raw = merged.get("ref_region")
    deform_mask_raw = merged.get("deform_mask")
    return SiteConfig(
        site_id=site_id,
        sensor=sensor,
        burst_id=merged.get("burst_id"),
        lon=float(merged.get("lon", -99.1332)),
        lat=float(merged.get("lat", 19.4326)),
        bbox=_parse_bbox(merged.get("bbox")),
        short_name=str(merged.get("short_name", NISAR_DEFAULT_SHORT_NAME)),
        track=int(merged["track"]) if merged.get("track") is not None else None,
        frame=int(merged["frame"]) if merged.get("frame") is not None else None,
        direction=_normalize_direction(merged.get("direction")),
        subdataset=merged.get("subdataset"),
        wavelength_mm=(
            float(merged["wavelength_mm"])
            if merged.get("wavelength_mm") is not None
            else None
        ),
        lookback_days=int(merged.get("lookback_days", 45)),
        max_granules=int(merged.get("max_granules", 6)),
        stride_x=stride_x,
        stride_y=stride_y,
        look_rx=int(merged["look_rx"]) if merged.get("look_rx") is not None else None,
        look_ry=int(merged["look_ry"]) if merged.get("look_ry") is not None else None,
        iters=int(merged["iters"]) if merged.get("iters") is not None else None,
        window=_parse_window(merged.get("window")),
        alert_mm=float(merged["alert_mm"]) if merged.get("alert_mm") is not None else None,
        alert_coh=float(merged["alert_coh"]) if merged.get("alert_coh") is not None else None,
        prune_raw=bool(merged.get("prune_raw", False)),
        stream=bool(merged.get("stream", False)),
        stream_transport=stream_transport,
        work_dir=Path(work_dir_raw) if work_dir_raw else None,
        stack=bool(merged.get("stack", False)),
        stack_bin=Path(stack_bin_raw) if stack_bin_raw else Path("./build/insar_stack"),
        corridors=Path(corridors_raw) if corridors_raw else None,
        buffer_m=float(merged.get("buffer_m", 100.0)),
        ramp_order=int(merged.get("ramp_order", 1)),
        vel_alert_mm_yr=float(merged.get("vel_alert_mm_yr", 15.0)),
        accel_alert_mm_yr2=float(merged.get("accel_alert_mm_yr2", 25.0)),
        min_alert_snr=float(merged.get("min_alert_snr", 2.5)),
        min_alert_span_days=float(merged.get("min_alert_span_days", 60.0)),
        dem=Path(dem_raw) if dem_raw else None,
        ref_lonlat=_parse_ref_lonlat(merged.get("ref_lonlat")),
        ref_radius_m=float(merged.get("ref_radius_m", 1000.0)),
        ref_region=Path(ref_region_raw) if ref_region_raw else None,
        deform_mask=Path(deform_mask_raw) if deform_mask_raw else None,
        two_pass_deform_mask=bool(merged.get("two_pass_deform_mask", False)),
        max_skip=max_skip,
        closure_thresh_mm=float(merged.get("closure_thresh_mm", 10.0)),
        closure_max_bad_frac=float(merged.get("closure_max_bad_frac", 0.35)),
        closure_mode=closure_mode,
        include_redundant=bool(merged.get("include_redundant", True)),
    )


def load_sites_config(
    config_path: Path,
    cli_defaults: Mapping[str, Any] | None = None,
) -> list[SiteConfig]:
    """Load and validate a multi-site JSON or YAML configuration file."""
    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load .yaml/.yml site configs. "
                "Use a .json config or install pyyaml."
            ) from exc
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)

    base_defaults: dict[str, Any] = dict(cli_defaults or {})
    if isinstance(payload, list):
        raw_sites = payload
    elif isinstance(payload, dict):
        file_defaults = payload.get("defaults") or {}
        if not isinstance(file_defaults, dict):
            raise ValueError("'defaults' in sites config must be a mapping")
        base_defaults.update(file_defaults)
        raw_sites = payload.get("sites")
        if not isinstance(raw_sites, list):
            raise ValueError("Sites config dict must contain a 'sites' list")
    else:
        raise ValueError("Sites config must be a JSON/YAML list or an object with a 'sites' list")

    sites: list[SiteConfig] = []
    seen_ids: set[str] = set()
    for entry in raw_sites:
        if not isinstance(entry, Mapping):
            raise ValueError("Each entry in 'sites' must be a mapping/object")
        site = _build_site_from_mapping(entry, base_defaults)
        if site.site_id in seen_ids:
            raise ValueError(f"Duplicate site_id '{site.site_id}' in {config_path}")
        seen_ids.add(site.site_id)
        sites.append(site)

    if not sites:
        raise ValueError(f"No sites found in {config_path}")
    return sites


def run_site(site: Any, work_dir: Path, engine_bin: Path, **options: Any) -> SiteRunSummary:
    """Discover candidates via configured sensor provider and execute site sync."""
    from insar_platform.providers.sensors import sensor_provider

    workflow_id = options.pop("workflow_id", None)
    if workflow_id:
        from insar_platform.workflows.state import mark_step

        mark_step(workflow_id, "DiscoverPass", "running", detail={"sensor": site.sensor})
    provider = sensor_provider(
        site.sensor,
        asf_module=options.pop("asf_mod", None),
        earthaccess_module=options.pop("ea_mod", None),
    )
    resolved_site, candidates = provider.discover(site)
    if workflow_id:
        mark_step(workflow_id, "DiscoverPass", "completed", detail={"granules_found": len(candidates)})

    return sync_site(
        resolved_site,
        work_dir,
        engine_bin,
        _catalog_candidates=candidates,
        _workflow_id=workflow_id,
        **options,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Automated OPERA CSLC-S1 and NISAR GSLC InSAR ingestion orchestrator."
    )
    parser.add_argument(
        "--sensor",
        choices=["opera", "nisar"],
        default="opera",
        help="Satellite dataset family for single-site mode (default: opera)",
    )
    parser.add_argument("--site-id", default="", help="Optional logical site ID for isolated dirs/state")
    parser.add_argument("--sites-config", type=Path, help="Path to multi-site sites.json or sites.yaml")
    parser.add_argument("--burst-id", help="OPERA Burst ID (e.g. T078-165486-IW2)")
    parser.add_argument("--lon", type=float, default=-99.1332, help="AOI longitude (default: Mexico City)")
    parser.add_argument("--lat", type=float, default=19.4326, help="AOI latitude (default: Mexico City)")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box for spatial search (primarily NISAR)",
    )
    parser.add_argument(
        "--short-name",
        default=NISAR_DEFAULT_SHORT_NAME,
        help=f"NASA CMR short_name for NISAR (default: {NISAR_DEFAULT_SHORT_NAME})",
    )
    parser.add_argument("--track", type=int, help="Optional NISAR relative orbit / track filter")
    parser.add_argument("--frame", type=int, help="Optional NISAR frame number filter")
    parser.add_argument("--direction", help="Optional orbit direction filter (ASCENDING / DESCENDING)")
    parser.add_argument("--subdataset", help="HDF5 dataset path override")
    parser.add_argument("--wavelength-mm", type=float, help="Radar wavelength override in mm")
    parser.add_argument("--work-dir", type=Path, default=Path("./var/work"))
    parser.add_argument("--engine", type=Path, default=Path("./build/insar_proto"))
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--max-granules", type=int, default=6)
    parser.add_argument("--stride-x", type=int, default=6)
    parser.add_argument("--stride-y", type=int, default=3)
    parser.add_argument("--look-rx", type=int, help="Optional coherence half-window X override")
    parser.add_argument("--look-ry", type=int, help="Optional coherence half-window Y override")
    parser.add_argument("--iters", type=int, help="Optional PCG max iterations override")
    parser.add_argument(
        "--window",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Optional pixel crop window forwarded to insar_proto",
    )
    parser.add_argument("--alert-mm", type=float, help="Optional displacement alert threshold (mm)")
    parser.add_argument("--alert-coh", type=float, help="Optional minimum coherence for alerts")
    parser.add_argument("--prune-raw", action="store_true", help="Delete older .h5 files once paired")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream HDF5 windows directly via GDAL /vsicurl/ or /vsis3/ without full downloads",
    )
    parser.add_argument(
        "--stream-transport",
        choices=["auto", "https", "s3"],
        default="auto",
        help="Transport for --stream mode (auto picks /vsis3/ in us-west-2, /vsicurl/ elsewhere)",
    )
    parser.add_argument("--stack", action="store_true", help="Run sequential multi-pass stacking & ramp removal after pairing")
    parser.add_argument("--max-skip", type=int, default=0, choices=[0, 1, 2], help="Generate redundant skip-1 (1) and skip-2 (2) pairs for triplet phase closure (default: 0)")
    parser.add_argument("--dem", type=Path, help="DEM GeoTIFF (local/SRTM/Copernicus) for elevation-correlated tropospheric correction")
    parser.add_argument("--ref-lonlat", type=float, nargs=2, metavar=("LON", "LAT"), help="Pin zero displacement to known stable WGS84 reference region")
    parser.add_argument("--ref-radius-m", type=float, default=1000.0, help="Radius (m) around --ref-lonlat (default: 1000.0)")
    parser.add_argument("--ref-region", type=Path, help="Vector/raster polygon mask of stable reference bedrock")
    parser.add_argument("--deform-mask", type=Path, help="Vector/raster exclusion mask for high-deformation zones before ramp fit")
    parser.add_argument("--two-pass-deform-mask", action="store_true", help="Two-pass stack build to mask deforming areas before ramp fit from pass 0")
    parser.add_argument("--closure-thresh-mm", type=float, default=10.0, help="Triplet closure threshold (mm) to mask unwrapping errors")
    parser.add_argument("--closure-max-bad-frac", type=float, default=0.35, help="Reject pair if closure failure fraction exceeds this")
    parser.add_argument("--closure-mode", choices=["consensus", "strict"], default="consensus", help="Multi-triplet masking mode")
    parser.add_argument("--corridors", type=Path, help="Optional GeoJSON/GPKG infrastructure corridors to pin stack outputs")
    parser.add_argument("--buffer-m", type=float, default=100.0, help="Corridor buffer distance in meters (default: 100.0)")
    parser.add_argument("--ramp-order", type=int, default=1, choices=[-1, 0, 1, 2], help="Robust IRLS ramp removal order (default: 1)")
    parser.add_argument("--vel-alert-mm-yr", type=float, default=15.0, help="Corridor |velocity| alert threshold (mm/yr)")
    parser.add_argument("--accel-alert-mm-yr2", type=float, default=25.0, help="Corridor |acceleration| alert threshold (mm/yr^2)")
    parser.add_argument("--min-alert-snr", type=float, default=2.5, help="Minimum velocity SNR |v|/sigma_v for alerts (default: 2.5)")
    parser.add_argument("--min-alert-span-days", type=float, default=60.0, help="Minimum stack span in days before critical velocity alerts (default: 60.0)")
    parser.add_argument("--dry-run", action="store_true", help="Query catalog and print plan without executing")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In multi-site mode, continue remaining sites if one site fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)

    if args.sites_config is not None:
        cli_defaults = {
            "lookback_days": args.lookback_days,
            "max_granules": args.max_granules,
            "stride_x": args.stride_x,
            "stride_y": args.stride_y,
            "look_rx": args.look_rx,
            "look_ry": args.look_ry,
            "iters": args.iters,
            "alert_mm": args.alert_mm,
            "alert_coh": args.alert_coh,
            "prune_raw": args.prune_raw,
            "stream": args.stream,
            "stream_transport": args.stream_transport,
            "stack": args.stack,
            "max_skip": args.max_skip,
            "dem": str(args.dem) if args.dem else None,
            "ref_lonlat": list(args.ref_lonlat) if args.ref_lonlat else None,
            "ref_radius_m": args.ref_radius_m,
            "ref_region": str(args.ref_region) if args.ref_region else None,
            "deform_mask": str(args.deform_mask) if args.deform_mask else None,
            "two_pass_deform_mask": args.two_pass_deform_mask,
            "closure_thresh_mm": args.closure_thresh_mm,
            "closure_max_bad_frac": args.closure_max_bad_frac,
            "closure_mode": args.closure_mode,
            "corridors": str(args.corridors) if args.corridors else None,
            "buffer_m": args.buffer_m,
            "ramp_order": args.ramp_order,
            "vel_alert_mm_yr": args.vel_alert_mm_yr,
            "accel_alert_mm_yr2": args.accel_alert_mm_yr2,
            "min_alert_snr": args.min_alert_snr,
            "min_alert_span_days": args.min_alert_span_days,
        }
        sites = load_sites_config(args.sites_config, cli_defaults=cli_defaults)
        if args.site_id:
            wanted_ids = {_sanitize_site_id(s) for s in str(args.site_id).split(",") if s.strip()}
            sites = [s for s in sites if s.site_id in wanted_ids]
            missing = sorted(wanted_ids - {s.site_id for s in sites})
            if missing:
                raise SystemExit(f"Site ID(s) not found in {args.sites_config}: {missing}")
    else:
        site_id = _sanitize_site_id(args.site_id) if args.site_id else ""
        sites = [
            SiteConfig(
                site_id=site_id,
                sensor=args.sensor,
                burst_id=args.burst_id,
                lon=args.lon,
                lat=args.lat,
                bbox=_parse_bbox(args.bbox),
                short_name=args.short_name,
                track=args.track,
                frame=args.frame,
                direction=_normalize_direction(args.direction),
                subdataset=args.subdataset,
                wavelength_mm=args.wavelength_mm,
                lookback_days=args.lookback_days,
                max_granules=args.max_granules,
                stride_x=args.stride_x,
                stride_y=args.stride_y,
                look_rx=args.look_rx,
                look_ry=args.look_ry,
                iters=args.iters,
                window=_parse_window(args.window),
                alert_mm=args.alert_mm,
                alert_coh=args.alert_coh,
                prune_raw=args.prune_raw,
                stream=args.stream,
                stream_transport=args.stream_transport,
                stack=args.stack,
                max_skip=args.max_skip,
                dem=args.dem,
                ref_lonlat=_parse_ref_lonlat(args.ref_lonlat),
                ref_radius_m=args.ref_radius_m,
                ref_region=args.ref_region,
                deform_mask=args.deform_mask,
                two_pass_deform_mask=args.two_pass_deform_mask,
                closure_thresh_mm=args.closure_thresh_mm,
                closure_max_bad_frac=args.closure_max_bad_frac,
                closure_mode=args.closure_mode,
                corridors=args.corridors,
                buffer_m=args.buffer_m,
                ramp_order=args.ramp_order,
                vel_alert_mm_yr=args.vel_alert_mm_yr,
                accel_alert_mm_yr2=args.accel_alert_mm_yr2,
                min_alert_snr=args.min_alert_snr,
                min_alert_span_days=args.min_alert_span_days,
            )
        ]

    failures = 0
    for site in sites:
        label = site.site_id or site.burst_id or site.sensor
        try:
            summary = sync_site(
                site=site,
                work_dir=args.work_dir,
                engine_bin=args.engine,
                dry_run=args.dry_run,
            )
            stack_msg = (
                f", stack_pairs={summary.stack_pairs} (+{summary.stack_new_pairs})"
                if summary.stack_pairs > 0 or site.stack
                else ""
            )
            print(
                f"[{summary.site_id}] done: discovered={summary.discovered_count}, "
                f"downloaded={summary.downloaded_count}, pairs_processed={summary.pairs_processed}, "
                f"pairs_skipped={summary.pairs_skipped}, pruned={summary.pruned_count}{stack_msg}"
            )
        except Exception as exc:
            failures += 1
            print(f"[{label}] ERROR: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
