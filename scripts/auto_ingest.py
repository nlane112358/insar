#!/usr/bin/env python3
"""
Automated InSAR ingestion and C++ processor orchestrator.

Supports:
  - NASA/ASF OPERA CSLC-S1 bursts (via asf_search)
  - NASA-ISRO NISAR L-band GSLC products (via earthaccess + dynamic HDF5 wavelength)
  - Single-site CLI runs or multi-site configuration files (sites.json / sites.yaml)

Tracks downloaded granules and processed interferometric pairs in SQLite,
and invokes ./build/insar_proto as a subprocess strictly for pixel compute
and Cloud-Optimized GeoTIFF / JSON export.
"""

from __future__ import annotations

import argparse
import contextlib
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

SPEED_OF_LIGHT_M_S = 299_792_458.0
OPERA_DEFAULT_SUBDATASET = "/data/VV"
OPERA_DEFAULT_WAVELENGTH_MM = 55.4658
NISAR_DEFAULT_SUBDATASET = "/science/LSAR/GSLC/grids/frequencyA/HH"
NISAR_DEFAULT_SHORT_NAME = "NISAR_L2_GSLC_V1"
NISAR_DEFAULT_WAVELENGTH_MM = 238.4

_NISAR_UR_PATTERN = re.compile(r"_(?:GSLC|RSLC)_(\d{3})_(\d{3})_([AD])_(\d{3})_")
_OPERA_BURST_HYPHEN_RE = re.compile(r"^T\d{3}-\d{6}-IW\d$", re.IGNORECASE)


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
class StreamTarget:
    """Resolved GDAL VSI URI and auth metadata for cloud streaming."""

    gdal_uri: str
    fsspec_url: str
    transport: str
    bearer_token: str | None = None
    s3_creds: Mapping[str, str] | None = None


@dataclass(frozen=True)
class GranuleCandidate:
    """Normalized catalog granule returned by a sensor provider."""

    granule_id: str
    stream_key: str
    acq_time: str
    filename: str
    download_fn: Callable[[Path], Path]
    stream_url: str = ""
    stream_fn: Callable[[str], StreamTarget] | None = None


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


def _import_asf() -> Any:
    try:
        import asf_search as asf
    except ImportError as exc:
        raise RuntimeError(
            "asf_search is required for OPERA CSLC-S1 ingestion. "
            "Install dependencies with: pip install -r scripts/requirements-ingest.txt"
        ) from exc
    return asf


def _import_earthaccess() -> Any:
    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError(
            "earthaccess is required for NISAR ingestion. "
            "Install dependencies with: pip install -r scripts/requirements-ingest.txt"
        ) from exc
    return earthaccess


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read HDF5 radar centerFrequency metadata. "
            "Install dependencies with: pip install -r scripts/requirements-ingest.txt"
        ) from exc
    return h5py


def _import_fsspec() -> Any:
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "fsspec is required for remote HDF5 metadata streaming. "
            "Install dependencies with: pip install -r scripts/requirements-ingest.txt"
        ) from exc
    return fsspec


def _format_gdal_hdf5_uri(vsi_or_path: str, subdataset: str) -> str:
    """Format an explicit GDAL HDF5 subdataset URI (supports /vsicurl/ and /vsis3/)."""
    sub_norm = subdataset if subdataset.startswith("/") else f"/{subdataset}"
    return f'HDF5:"{vsi_or_path}":/{sub_norm}'


def _detect_aws_in_region(ea_mod: Any = None) -> bool:
    """Best-effort non-blocking check for running inside AWS us-west-2."""
    if os.environ.get("INSAR_AWS_IN_REGION", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if ea_mod is not None:
        store = getattr(ea_mod, "__store__", None)
        if getattr(store, "in_region", False):
            return True
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    in_aws_runtime = bool(
        os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("ECS_CONTAINER_METADATA_URI")
    )
    return in_aws_runtime and region == "us-west-2"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: Any) -> str:
    """Normalize a datetime or ISO-8601 string into UTC ISO-8601 format."""
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(value or "").strip()
    if not text:
        raise ValueError("Granule missing acquisition timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return text


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


def _normalize_direction(direction: str | None) -> str | None:
    """Normalize orbit direction values ('A'/'ASCENDING', 'D'/'DESCENDING')."""
    if direction is None:
        return None
    upper = str(direction).strip().upper()
    if not upper:
        return None
    if upper in {"A", "ASC", "ASCENDING"}:
        return "ASCENDING"
    if upper in {"D", "DESC", "DESCENDING"}:
        return "DESCENDING"
    raise ValueError(f"Unsupported orbit direction '{direction}'; expected ASCENDING or DESCENDING")


def _ensure_table_columns(
    conn: sqlite3.Connection,
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
    """Initialize (or migrate) the SQLite ingestion ledger."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, timeout=30.0)
    conn.row_factory = sqlite3.Row
    if db_str != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")

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


def _create_asf_session(asf_mod: Any) -> Any:
    """Create a headless ASFSession using env credentials or ~/.netrc."""
    session = asf_mod.ASFSession()
    token = os.environ.get("EARTHDATA_TOKEN")
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if token:
        session.auth_with_token(token)
    elif user and password:
        session.auth_with_creds(user, password)
    return session


def _login_earthaccess(ea_mod: Any) -> Any:
    """Authenticate earthaccess headlessly without ever blocking on stdin."""
    has_env_creds = bool(
        os.environ.get("EARTHDATA_TOKEN")
        or (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))
    )
    strategies = ["environment", "netrc"] if has_env_creds else ["netrc"]
    last_exc: Exception | None = None
    for strategy in strategies:
        try:
            auth = ea_mod.login(strategy=strategy)
            if getattr(auth, "authenticated", False):
                return auth
        except Exception as exc:
            last_exc = exc

    msg = (
        "Earthdata authentication failed in headless mode. Configure ~/.netrc "
        "(machine urs.earthdata.nasa.gov login ... password ...) or set EARTHDATA_TOKEN."
    )
    if last_exc is not None:
        raise RuntimeError(f"{msg} Cause: {last_exc}") from last_exc
    raise RuntimeError(msg)


def discover_burst_id(
    lon: float,
    lat: float,
    lookback_days: int = 60,
    asf_mod: Any = None,
) -> str:
    """Find an active OPERA CSLC-S1 burst covering a lon/lat point."""
    asf = asf_mod or _import_asf()
    start = (_utc_now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    results = asf.search(
        dataset=asf.DATASET.OPERA_S1,
        processingLevel="CSLC",
        intersectsWith=f"POINT({lon} {lat})",
        start=start,
        maxResults=50,
    )
    if not results:
        raise RuntimeError(f"No OPERA CSLC-S1 bursts found at POINT({lon} {lat})")

    # Group by operaBurstID so we prefer bursts with >= 2 passes and the latest acquisition
    burst_stats: dict[str, tuple[int, str]] = {}
    for prod in results:
        props = prod.properties
        bid = props.get("operaBurstID")
        if not bid:
            continue
        t = _normalize_timestamp(props.get("startTime", "")) if props.get("startTime") else ""
        prev_count, prev_latest = burst_stats.get(bid, (0, ""))
        burst_stats[bid] = (prev_count + 1, max(prev_latest, t))

    if not burst_stats:
        raise RuntimeError(f"OPERA CSLC-S1 search at POINT({lon} {lat}) returned no operaBurstID")

    burst_id = max(
        burst_stats.items(),
        key=lambda kv: (kv[1][0] >= 2, kv[1][1], kv[1][0]),
    )[0]
    print(f"Auto-selected OPERA Burst ID: {burst_id}")
    return burst_id


def _atomic_asf_download(prod: Any, session: Any, raw_dir: Path, fname: str) -> Path:
    """Download an ASF product via a staging folder so partial files never pollute raw_dir."""
    dest_file = raw_dir / fname
    with tempfile.TemporaryDirectory(dir=raw_dir, prefix=".tmp_dl_") as tmp:
        tmp_dir = Path(tmp)
        prod.download(path=str(tmp_dir), session=session)
        candidate = tmp_dir / fname
        if not candidate.exists():
            files = [p for p in tmp_dir.iterdir() if p.is_file()]
            if not files:
                raise FileNotFoundError(f"ASF download produced no file for {fname}")
            h5_files = [p for p in files if p.suffix.lower() in {".h5", ".hdf5"}]
            candidate = h5_files[0] if h5_files else files[0]
            dest_file = raw_dir / candidate.name
        if candidate.stat().st_size == 0:
            raise IOError(f"Downloaded empty file from ASF: {candidate.name}")
        os.replace(candidate, dest_file)
    return dest_file


def _normalize_opera_burst_id(burst_id: str | None) -> str | None:
    """Convert filename-style hyphenated OPERA burst IDs to CMR underscore format."""
    if burst_id is None:
        return None
    cleaned = burst_id.strip()
    if _OPERA_BURST_HYPHEN_RE.fullmatch(cleaned):
        normalized = cleaned.upper().replace("-", "_")
        print(f"Normalized OPERA Burst ID '{cleaned}' -> '{normalized}' for ASF CMR query.")
        return normalized
    return cleaned


def _resolve_https_redirect(session: Any, url: str) -> str:
    """Resolve Earthdata/ASF OAuth redirects to the final GET-signed HTTPS URL without reading body."""
    if session is None or not hasattr(session, "get"):
        return url
    with session.get(url, stream=True, allow_redirects=True, timeout=30) as resp:
        resp.raise_for_status()
        return str(getattr(resp, "url", "") or url)


def _extract_bearer_token(session: Any, fallback_token: str | None = None) -> str | None:
    if session is not None:
        auth_hdr = str((getattr(session, "headers", None) or {}).get("Authorization", ""))
        if auth_hdr.lower().startswith("bearer "):
            return auth_hdr.split(None, 1)[1].strip()
    return fallback_token or os.environ.get("EARTHDATA_TOKEN")


def _resolve_asf_stream_target(
    prod: Any,
    session: Any,
    subdataset: str,
    transport: str,
    ea_mod: Any = None,
) -> StreamTarget:
    """Resolve an ASFProduct into a GDAL /vsis3/ or /vsicurl/ HDF5 StreamTarget."""
    props = prod.properties
    https_url = str(props.get("url") or "")
    s3_urls = [str(u) for u in (props.get("s3Urls") or []) if str(u).startswith("s3://")]

    chosen = transport
    if chosen == "auto":
        chosen = "s3" if (s3_urls and _detect_aws_in_region(ea_mod)) else "https"

    if chosen == "s3":
        if not s3_urls:
            raise RuntimeError(
                f"ASF product '{props.get('sceneName')}' has no s3Urls for /vsis3/ streaming"
            )
        s3_url = s3_urls[0]
        ea = ea_mod or _import_earthaccess()
        auth = _login_earthaccess(ea)
        creds = auth.get_s3_credentials(daac="ASF")
        vsi_path = f"/vsis3/{s3_url[len('s3://'):]}"
        return StreamTarget(
            gdal_uri=_format_gdal_hdf5_uri(vsi_path, subdataset),
            fsspec_url=s3_url,
            transport="s3",
            s3_creds={
                "AWS_ACCESS_KEY_ID": creds["accessKeyId"],
                "AWS_SECRET_ACCESS_KEY": creds["secretAccessKey"],
                "AWS_SESSION_TOKEN": creds["sessionToken"],
                "AWS_REGION": "us-west-2",
            },
        )

    if not https_url:
        raise RuntimeError(f"ASF product '{props.get('sceneName')}' has no HTTPS URL")
    resolved_url = _resolve_https_redirect(session, https_url)
    # Only pass Bearer token when no cross-origin redirect to a pre-signed CloudFront/S3 URL happened
    bearer = _extract_bearer_token(session) if resolved_url == https_url else None
    return StreamTarget(
        gdal_uri=_format_gdal_hdf5_uri(f"/vsicurl/{resolved_url}", subdataset),
        fsspec_url=resolved_url,
        transport="https",
        bearer_token=bearer,
    )


def fetch_opera_candidates(
    site: SiteConfig,
    asf_mod: Any = None,
) -> tuple[SiteConfig, list[GranuleCandidate]]:
    """Query ASF for OPERA CSLC-S1 granules for a site."""
    asf = asf_mod or _import_asf()
    raw_burst_id = _normalize_opera_burst_id(site.burst_id)
    burst_id = raw_burst_id or discover_burst_id(
        site.lon, site.lat, site.lookback_days, asf_mod=asf
    )
    resolved_site = SiteConfig(**{**site.__dict__, "burst_id": burst_id})
    session = _create_asf_session(asf)

    start_date = (_utc_now() - timedelta(days=site.lookback_days)).strftime("%Y-%m-%d")
    print(f"[{_utc_now().isoformat()}] Querying ASF OPERA CSLC-S1 for {burst_id} since {start_date}...")

    results = asf.search(
        dataset=asf.DATASET.OPERA_S1,
        processingLevel="CSLC",
        operaBurstID=burst_id,
        start=start_date,
        maxResults=site.max_granules if site.max_granules > 0 else None,
    )

    subdataset = resolved_site.resolved_subdataset()
    candidates: list[GranuleCandidate] = []
    for prod in results:
        props = prod.properties
        gid = props.get("sceneName") or props.get("fileID")
        if not gid:
            continue
        acq_time = _normalize_timestamp(props["startTime"])
        url = str(props.get("url", ""))
        fname = props.get("fileName")
        if not fname:
            fname = Path(urlparse(url).path).name or f"{gid}.h5"

        def _make_downloader(p: Any = prod, fn: str = fname) -> Callable[[Path], Path]:
            return lambda dest_dir: _atomic_asf_download(p, session, dest_dir, fn)

        def _make_streamer(p: Any = prod) -> Callable[[str], StreamTarget]:
            return lambda transport: _resolve_asf_stream_target(p, session, subdataset, transport)

        candidates.append(
            GranuleCandidate(
                granule_id=gid,
                stream_key=burst_id,
                acq_time=acq_time,
                filename=fname,
                download_fn=_make_downloader(),
                stream_url=url,
                stream_fn=_make_streamer(),
            )
        )
    return resolved_site, candidates


def _get_umm_attr(umm: Mapping[str, Any], attr_name: str) -> str | None:
    for entry in umm.get("AdditionalAttributes", []):
        if entry.get("Name") == attr_name:
            vals = entry.get("Values") or []
            if vals:
                return str(vals[0])
    return None


def _extract_nisar_metadata(granule: Mapping[str, Any]) -> dict[str, Any]:
    """Extract normalized ID, time, filename, and orbital stream metadata from CMR DataGranule."""
    umm = granule.get("umm", {})
    meta = granule.get("meta", {})
    gid = umm.get("GranuleUR") or meta.get("concept-id")
    if not gid:
        raise ValueError("NISAR DataGranule missing both GranuleUR and concept-id")

    temporal = umm.get("TemporalExtent", {})
    acq_raw = (
        temporal.get("RangeDateTime", {}).get("BeginningDateTime")
        or temporal.get("SingleDateTime")
    )
    acq_time = _normalize_timestamp(acq_raw)

    track_str = _get_umm_attr(umm, "TRACK_NUMBER")
    frame_str = _get_umm_attr(umm, "FRAME_NUMBER")
    dir_raw = _get_umm_attr(umm, "ASCENDING_DESCENDING")

    track = int(track_str) if track_str and track_str.isdigit() else None
    frame = int(frame_str) if frame_str and frame_str.isdigit() else None
    direction = _normalize_direction(dir_raw) if dir_raw else None

    match = _NISAR_UR_PATTERN.search(str(gid))
    if match:
        if track is None:
            track = int(match.group(2))
        if direction is None:
            direction = _normalize_direction(match.group(3))
        if frame is None:
            frame = int(match.group(4))

    parts = ["NISAR"]
    if track is not None:
        parts.append(f"T{track:03d}")
    if frame is not None:
        parts.append(f"F{frame:03d}")
    if direction is not None:
        parts.append(direction[0])
    stream_key = "_".join(parts) if len(parts) > 1 else "NISAR_DEFAULT"

    fname = None
    data_links = granule.data_links() if hasattr(granule, "data_links") else []
    for url in data_links:
        name = Path(urlparse(url).path).name
        if name.lower().endswith((".h5", ".hdf5")):
            fname = name
            break
    if not fname:
        fname = str(gid) if str(gid).lower().endswith((".h5", ".hdf5")) else f"{gid}.h5"

    return {
        "granule_id": str(gid),
        "acq_time": acq_time,
        "track": track,
        "frame": frame,
        "direction": direction,
        "stream_key": stream_key,
        "filename": fname,
    }


def _atomic_earthaccess_download(
    ea_mod: Any,
    granule: Any,
    raw_dir: Path,
    expected_name: str,
) -> Path:
    """Download a NISAR DataGranule via earthaccess into a staging directory and keep its .h5 file."""
    with tempfile.TemporaryDirectory(dir=raw_dir, prefix=".tmp_dl_") as tmp:
        tmp_dir = Path(tmp)
        paths = ea_mod.download([granule], local_path=str(tmp_dir), show_progress=False)
        downloaded = [Path(p) for p in paths if Path(p).exists()]
        if not downloaded:
            downloaded = [p for p in tmp_dir.rglob("*") if p.is_file()]
        if not downloaded:
            raise FileNotFoundError(f"earthaccess.download returned no files for {expected_name}")

        h5_files = [p for p in downloaded if p.suffix.lower() in {".h5", ".hdf5"}]
        chosen = h5_files[0] if h5_files else downloaded[0]
        if chosen.stat().st_size == 0:
            raise IOError(f"Downloaded empty file from earthaccess: {chosen.name}")

        dest_file = raw_dir / chosen.name
        os.replace(chosen, dest_file)
    return dest_file


def _pick_h5_links(granule: Any, access: str | None) -> list[str]:
    if not hasattr(granule, "data_links"):
        return []
    try:
        links = granule.data_links(access=access) if access is not None else granule.data_links()
    except TypeError:
        links = granule.data_links()
    h5_links = [str(u) for u in links if str(u).lower().split("?")[0].endswith((".h5", ".hdf5"))]
    return h5_links or [str(u) for u in links]


def _resolve_earthaccess_stream_target(
    granule: Any,
    ea_mod: Any,
    auth_provider: Callable[[], Any],
    subdataset: str,
    transport: str,
) -> StreamTarget:
    """Resolve an earthaccess DataGranule into a GDAL /vsis3/ or /vsicurl/ HDF5 StreamTarget."""
    auth = auth_provider()
    s3_links = _pick_h5_links(granule, "direct")
    https_links = _pick_h5_links(granule, "external")

    chosen = transport
    if chosen == "auto":
        chosen = "s3" if (s3_links and _detect_aws_in_region(ea_mod)) else "https"

    if chosen == "s3":
        if not s3_links:
            gid = granule.get("umm", {}).get("GranuleUR", "granule")
            raise RuntimeError(f"Granule '{gid}' has no S3 links for /vsis3/ streaming")
        s3_url = s3_links[0]
        endpoint = granule.get_s3_credentials_endpoint() if hasattr(granule, "get_s3_credentials_endpoint") else None
        provider = granule.get("meta", {}).get("provider-id")
        if endpoint:
            creds = auth.get_s3_credentials(endpoint=endpoint)
        elif provider:
            creds = auth.get_s3_credentials(provider=provider)
        else:
            creds = auth.get_s3_credentials(daac="ASF")
        vsi_path = f"/vsis3/{s3_url[len('s3://'):]}"
        return StreamTarget(
            gdal_uri=_format_gdal_hdf5_uri(vsi_path, subdataset),
            fsspec_url=s3_url,
            transport="s3",
            s3_creds={
                "AWS_ACCESS_KEY_ID": creds["accessKeyId"],
                "AWS_SECRET_ACCESS_KEY": creds["secretAccessKey"],
                "AWS_SESSION_TOKEN": creds["sessionToken"],
                "AWS_REGION": "us-west-2",
            },
        )

    if not https_links:
        gid = granule.get("umm", {}).get("GranuleUR", "granule")
        raise RuntimeError(f"Granule '{gid}' has no HTTPS links for /vsicurl/ streaming")
    https_url = https_links[0]
    sess = auth.get_session() if hasattr(auth, "get_session") else None
    resolved_url = _resolve_https_redirect(sess, https_url)
    token = (getattr(auth, "token", None) or {}).get("access_token") if resolved_url == https_url else None
    return StreamTarget(
        gdal_uri=_format_gdal_hdf5_uri(f"/vsicurl/{resolved_url}", subdataset),
        fsspec_url=resolved_url,
        transport="https",
        bearer_token=token,
    )


def fetch_nisar_candidates(
    site: SiteConfig,
    ea_mod: Any = None,
) -> tuple[SiteConfig, list[GranuleCandidate]]:
    """Query NASA CMR via earthaccess for NISAR L2 GSLC granules."""
    ea = ea_mod or _import_earthaccess()
    auth_state: dict[str, Any] = {"auth": None}

    def _ensure_authenticated() -> Any:
        if auth_state["auth"] is None:
            auth_state["auth"] = _login_earthaccess(ea)
        return auth_state["auth"]

    now = _utc_now()
    start_date = (now - timedelta(days=site.lookback_days)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    search_kwargs: dict[str, Any] = {
        "short_name": site.short_name,
        "temporal": (start_date, end_date),
        "count": -1,
    }
    if site.bbox is not None:
        search_kwargs["bounding_box"] = tuple(site.bbox)
        aoi_desc = f"bbox={site.bbox}"
    else:
        search_kwargs["point"] = (site.lon, site.lat)
        aoi_desc = f"point=({site.lon}, {site.lat})"

    print(
        f"[{now.isoformat()}] Querying earthaccess {site.short_name} ({aoi_desc}) "
        f"for {start_date}..{end_date}..."
    )
    results = ea.search_data(**search_kwargs)

    site_dir = _normalize_direction(site.direction)
    subdataset = site.resolved_subdataset()
    per_stream: dict[str, list[GranuleCandidate]] = {}

    for granule in results:
        info = _extract_nisar_metadata(granule)
        if site.track is not None and info["track"] != site.track:
            continue
        if site.frame is not None and info["frame"] != site.frame:
            continue
        if site_dir is not None and info["direction"] != site_dir:
            continue

        https_links = _pick_h5_links(granule, "external")
        preview_url = https_links[0] if https_links else info["filename"]

        def _make_downloader(g: Any = granule, fn: str = info["filename"]) -> Callable[[Path], Path]:
            def _download(dest_dir: Path) -> Path:
                _ensure_authenticated()
                return _atomic_earthaccess_download(ea, g, dest_dir, fn)
            return _download

        def _make_streamer(g: Any = granule) -> Callable[[str], StreamTarget]:
            return lambda transport: _resolve_earthaccess_stream_target(
                g, ea, _ensure_authenticated, subdataset, transport
            )

        cand = GranuleCandidate(
            granule_id=info["granule_id"],
            stream_key=info["stream_key"],
            acq_time=info["acq_time"],
            filename=info["filename"],
            download_fn=_make_downloader(),
            stream_url=preview_url,
            stream_fn=_make_streamer(),
        )
        per_stream.setdefault(cand.stream_key, []).append(cand)

    candidates: list[GranuleCandidate] = []
    for stream_cands in per_stream.values():
        stream_cands.sort(key=lambda c: (c.acq_time, c.granule_id))
        if site.max_granules > 0:
            stream_cands = stream_cands[-site.max_granules :]
        candidates.extend(stream_cands)

    return site, candidates


def _extract_scalar_float(raw: Any) -> float | None:
    """Extract a scalar float from an HDF5 0-D scalar, NumPy array, or nested list."""
    if raw is None:
        return None
    if hasattr(raw, "flat"):
        flat = raw.flat
        if len(flat) == 0:
            return None
        return float(flat[0])
    while isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    return float(raw)


def _read_center_frequency_hz_from_h5file(h5_file: Any, subdataset: str) -> float | None:
    """Read centerFrequency (in Hz) from an already-open h5py.File."""
    sub_norm = subdataset if subdataset.startswith("/") else f"/{subdataset}"
    parent = sub_norm.rsplit("/", 1)[0] if "/" in sub_norm[1:] else ""
    candidates: list[str] = []
    if parent:
        candidates.append(f"{parent}/centerFrequency")
    for fallback in (
        "/science/LSAR/GSLC/grids/frequencyA/centerFrequency",
        "/science/LSAR/GSLC/grids/frequencyB/centerFrequency",
        "/science/SSAR/GSLC/grids/frequencyA/centerFrequency",
    ):
        if fallback not in candidates:
            candidates.append(fallback)

    for ds_path in candidates:
        if ds_path not in h5_file:
            continue
        ds = h5_file[ds_path]
        freq_val = _extract_scalar_float(ds[()])
        if freq_val is None:
            continue
        units_raw = ds.attrs.get("units", "Hz")
        if isinstance(units_raw, bytes):
            units_raw = units_raw.decode("utf-8", errors="ignore")
        units = str(units_raw).strip().lower()
        if units == "mhz":
            freq_val *= 1e6
        elif units == "ghz":
            freq_val *= 1e9
        elif units == "khz":
            freq_val *= 1e3
        if freq_val > 0.0:
            return freq_val
    return None


def _read_center_frequency_hz(h5_path: Path, subdataset: str, h5py_mod: Any) -> float | None:
    """Read centerFrequency (in Hz) from a local NISAR HDF5 file."""
    with h5py_mod.File(h5_path, "r") as f:
        return _read_center_frequency_hz_from_h5file(f, subdataset)


@contextlib.contextmanager
def _open_remote_h5(
    target: StreamTarget,
    h5py_mod: Any = None,
    fsspec_mod: Any = None,
) -> Iterator[Any]:
    """Open a remote (/vsicurl/ HTTPS or /vsis3/ S3) HDF5 file via fsspec byte-range requests."""
    h5py = h5py_mod or _import_h5py()
    fsspec = fsspec_mod or _import_fsspec()

    if target.transport == "s3":
        s3_creds = target.s3_creds or {}
        fs = fsspec.filesystem(
            "s3",
            key=s3_creds.get("AWS_ACCESS_KEY_ID"),
            secret=s3_creds.get("AWS_SECRET_ACCESS_KEY"),
            token=s3_creds.get("AWS_SESSION_TOKEN"),
        )
        with fs.open(target.fsspec_url, "rb", block_size=65536) as fobj:
            with h5py.File(fobj, "r") as hf:
                yield hf
        return

    client_kwargs: dict[str, Any] = {}
    if target.bearer_token:
        client_kwargs = {
            "headers": {"Authorization": f"Bearer {target.bearer_token}"},
            "trust_env": False,
        }
    fs = fsspec.filesystem("http", client_kwargs=client_kwargs) if client_kwargs else fsspec.filesystem("http")
    with fs.open(target.fsspec_url, "rb", block_size=65536) as fobj:
        with h5py.File(fobj, "r") as hf:
            yield hf


def _read_grid_geometry_from_h5file(
    h5_file: Any,
    subdataset: str,
) -> tuple[int, int, float, float, float, float, str] | None:
    """Extract grid dimensions, top-left origin, pixel spacing, and CRS from HDF5 metadata."""
    sub_norm = subdataset if subdataset.startswith("/") else f"/{subdataset}"
    parent = sub_norm.rsplit("/", 1)[0] if "/" in sub_norm[1:] else ""
    if not parent or parent not in h5_file:
        return None
    grp = h5_file[parent]

    x_ds = grp.get("xCoordinates") or grp.get("x_coordinates")
    y_ds = grp.get("yCoordinates") or grp.get("y_coordinates")
    if x_ds is None or y_ds is None or len(x_ds) < 2 or len(y_ds) < 2:
        return None

    nx = len(x_ds)
    ny = len(y_ds)
    x0, x1 = float(x_ds[0]), float(x_ds[1])
    y0, y1 = float(y_ds[0]), float(y_ds[1])
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 or dy == 0.0:
        return None

    gt_x0 = x0 - 0.5 * dx
    gt_y0 = y0 - 0.5 * dy

    target_srs: str | None = None
    proj_ds = grp.get("projection")
    if proj_ds is not None:
        epsg_attr = proj_ds.attrs.get("epsg_code")
        if epsg_attr is not None:
            epsg_val = int(_extract_scalar_float(epsg_attr) or 0)
            if epsg_val > 0:
                target_srs = f"EPSG:{epsg_val}"
        if target_srs is None:
            spatial_ref = proj_ds.attrs.get("spatial_ref")
            if isinstance(spatial_ref, bytes):
                spatial_ref = spatial_ref.decode("utf-8", errors="ignore")
            if spatial_ref:
                target_srs = str(spatial_ref)
        if target_srs is None:
            proj_scalar = _extract_scalar_float(proj_ds[()])
            if proj_scalar and proj_scalar > 0:
                target_srs = f"EPSG:{int(proj_scalar)}"

    if not target_srs:
        return None
    return (nx, ny, gt_x0, gt_y0, dx, dy, target_srs)


def compute_window_from_bbox(
    bbox: tuple[float, float, float, float],
    grid: tuple[int, int, float, float, float, float, str],
    max_dims: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Convert a WGS84 bounding box into a clamped pixel crop window (x, y, w, h)."""
    nx, ny, gt_x0, gt_y0, dx, dy, target_srs = grid
    if max_dims is not None:
        nx = min(nx, max_dims[0])
        ny = min(ny, max_dims[1])

    min_lon, min_lat, max_lon, max_lat = bbox
    corners = (
        f"{min_lon} {min_lat}\n"
        f"{max_lon} {min_lat}\n"
        f"{max_lon} {max_lat}\n"
        f"{min_lon} {max_lat}\n"
    )
    tr = subprocess.run(
        ["gdaltransform", "-s_srs", "EPSG:4326", "-t_srs", target_srs],
        input=corners,
        capture_output=True,
        text=True,
        check=True,
    )
    pts: list[tuple[float, float]] = []
    for line in tr.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 4:
        raise RuntimeError(f"gdaltransform returned unexpected output for {target_srs}: {tr.stdout!r}")

    pxs = [(px - gt_x0) / dx for px, _ in pts]
    pys = [(py - gt_y0) / dy for _, py in pts]

    wx0 = max(0, int(min(pxs)))
    wy0 = max(0, int(min(pys)))
    wx1 = min(nx, int(max(pxs) + 0.999))
    wy1 = min(ny, int(max(pys) + 0.999))
    ww = wx1 - wx0
    wh = wy1 - wy0
    if ww <= 0 or wh <= 0:
        return None
    return (wx0, wy0, ww, wh)


def _inspect_stream_pair(
    ref_target: StreamTarget,
    sec_target: StreamTarget,
    site: SiteConfig,
    subdataset: str,
    h5py_mod: Any = None,
    fsspec_mod: Any = None,
) -> tuple[float | None, tuple[int, int, int, int] | None]:
    """Read remote HDF5 header once for NISAR wavelength and optional bbox auto-window."""
    need_wavelength = site.wavelength_mm is None and site.sensor == "nisar"
    need_window = site.window is None and site.bbox is not None
    if not need_wavelength and not need_window:
        return site.wavelength_mm, site.window

    wavelength_mm = site.wavelength_mm
    window = site.window

    with _open_remote_h5(ref_target, h5py_mod=h5py_mod, fsspec_mod=fsspec_mod) as ref_hf:
        if need_wavelength:
            freq_hz = _read_center_frequency_hz_from_h5file(ref_hf, subdataset)
            if freq_hz is None:
                print(
                    f"Warning: centerFrequency not found in remote header; "
                    f"falling back to {NISAR_DEFAULT_WAVELENGTH_MM:.1f} mm"
                )
                wavelength_mm = NISAR_DEFAULT_WAVELENGTH_MM
            else:
                wavelength_mm = (SPEED_OF_LIGHT_M_S / freq_hz) * 1000.0

        if need_window and site.bbox is not None:
            grid = _read_grid_geometry_from_h5file(ref_hf, subdataset)
            if grid is None:
                raise RuntimeError(
                    "Could not read HDF5 coordinate/projection metadata for auto-window; "
                    "pass --window x y w h explicitly."
                )
            sec_dims: tuple[int, int] | None = None
            with _open_remote_h5(sec_target, h5py_mod=h5py_mod, fsspec_mod=fsspec_mod) as sec_hf:
                if subdataset in sec_hf:
                    sh = sec_hf[subdataset].shape
                    sec_dims = (int(sh[1]), int(sh[0]))
                if need_wavelength and wavelength_mm is not None:
                    sec_freq_hz = _read_center_frequency_hz_from_h5file(sec_hf, subdataset)
                    if sec_freq_hz is not None:
                        sec_wl = (SPEED_OF_LIGHT_M_S / sec_freq_hz) * 1000.0
                        if abs(wavelength_mm - sec_wl) > 0.5:
                            print(
                                f"Warning: radar wavelength mismatch ({wavelength_mm:.3f} mm vs "
                                f"{sec_wl:.3f} mm); using reference wavelength."
                            )
            window = compute_window_from_bbox(site.bbox, grid, max_dims=sec_dims)

    return wavelength_mm, window


@contextlib.contextmanager
def _stream_subprocess_env(
    ref_target: StreamTarget | None,
    sec_target: StreamTarget | None,
) -> Iterator[dict[str, str]]:
    """Build GDAL environment variables for low-bandwidth /vsicurl/ or /vsis3/ streaming."""
    env = dict(os.environ)
    env.update(
        {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "VSI_CACHE": "FALSE",
            "GDAL_PAM_ENABLED": "NO",
            "CPL_VSIL_CURL_CHUNK_SIZE": "65536",
            "CPL_VSIL_CURL_USE_HEAD": "NO",
            "GDAL_SKIP": "NETCDF",
        }
    )

    for target in (ref_target, sec_target):
        if target and target.s3_creds:
            env.update(target.s3_creds)
            env["AWS_NO_SIGN_REQUEST"] = "NO"
            break

    bearer_token = None
    for target in (ref_target, sec_target):
        if target and target.bearer_token:
            bearer_token = target.bearer_token
            break

    header_path: Path | None = None
    try:
        if bearer_token:
            fd, tmp_name = tempfile.mkstemp(prefix="gdal_edl_hdr_", suffix=".txt")
            header_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                os.chmod(header_path, 0o600)
                fh.write(f"Authorization: Bearer {bearer_token}\r\n")
            env["GDAL_HTTP_HEADER_FILE"] = str(header_path)
        yield env
    finally:
        if header_path is not None and header_path.exists():
            header_path.unlink()


def read_nisar_wavelength_mm(
    ref_path: Path,
    subdataset: str,
    sec_path: Path | None = None,
    h5py_mod: Any = None,
) -> float:
    """Compute exact NISAR radar wavelength in mm from HDF5 centerFrequency."""
    h5py = h5py_mod or _import_h5py()
    freq_hz = _read_center_frequency_hz(ref_path, subdataset, h5py)
    if freq_hz is None:
        print(
            f"Warning: centerFrequency not found in {ref_path.name}; "
            f"falling back to {NISAR_DEFAULT_WAVELENGTH_MM:.1f} mm"
        )
        return NISAR_DEFAULT_WAVELENGTH_MM

    wavelength_mm = (SPEED_OF_LIGHT_M_S / freq_hz) * 1000.0
    if sec_path is not None and sec_path.exists():
        sec_freq_hz = _read_center_frequency_hz(sec_path, subdataset, h5py)
        if sec_freq_hz is not None:
            sec_wavelength_mm = (SPEED_OF_LIGHT_M_S / sec_freq_hz) * 1000.0
            if abs(wavelength_mm - sec_wavelength_mm) > 0.5:
                print(
                    f"Warning: radar wavelength mismatch between {ref_path.name} "
                    f"({wavelength_mm:.3f} mm) and {sec_path.name} ({sec_wavelength_mm:.3f} mm); "
                    f"using reference wavelength."
                )
    return wavelength_mm


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
    return cmd


def _validate_engine_binary(engine_bin: Path) -> None:
    """Verify the compiled insar_proto binary exists and is executable."""
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


def sync_site(
    site: SiteConfig,
    work_dir: Path,
    engine_bin: Path,
    *,
    dry_run: bool = False,
    asf_mod: Any = None,
    ea_mod: Any = None,
    h5py_mod: Any = None,
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
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_root.mkdir(parents=True, exist_ok=True)

    if site.sensor == "opera":
        resolved_site, candidates = fetch_opera_candidates(site, asf_mod=asf_mod)
    elif site.sensor == "nisar":
        resolved_site, candidates = fetch_nisar_candidates(site, ea_mod=ea_mod)
    else:
        raise ValueError(f"Unsupported sensor '{site.sensor}'; expected 'opera' or 'nisar'")

    candidates.sort(key=lambda c: (c.acq_time, c.granule_id))
    print(f"Found {len(candidates)} granules in window.")

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

    try:
        # 1. Download any unseen (and unpruned) granules
        for cand in candidates:
            row = conn.execute(
                "SELECT local_path, pruned FROM granules WHERE site_id = ? AND granule_id = ?",
                (resolved_site.site_id, cand.granule_id),
            ).fetchone()
            if row:
                if row["pruned"]:
                    continue
                if Path(row["local_path"]).exists():
                    continue

            if dry_run:
                print(f"[DRY-RUN] Would download {cand.filename} ({cand.acq_time}) -> {raw_dir}")
                pending_dry_run_cands.append(cand)
                continue

            print(f"Downloading {cand.filename} ({cand.acq_time})...")
            local_file = cand.download_fn(raw_dir)
            downloaded_count += 1

            conn.execute(
                """
                INSERT OR REPLACE INTO granules
                    (granule_id, burst_id, acq_time, local_path, sensor, site_id, stream_key, pruned)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    cand.granule_id,
                    cand.stream_key,
                    cand.acq_time,
                    str(local_file),
                    resolved_site.sensor,
                    resolved_site.site_id,
                    cand.stream_key,
                ),
            )
            conn.commit()

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
                streams.setdefault(cand.stream_key, []).append(
                    {
                        "granule_id": cand.granule_id,
                        "stream_key": cand.stream_key,
                        "acq_time": cand.acq_time,
                        "local_path": str(raw_dir / cand.filename),
                        "pruned": False,
                        "simulated": True,
                    }
                )
            for stream_rows in streams.values():
                stream_rows.sort(key=lambda r: (r["acq_time"], r["granule_id"]))

        # 3. Form consecutive pairs (t_k -> t_{k+1}) per stream and invoke C++ engine
        subdataset = resolved_site.resolved_subdataset()
        for stream_key, stream_rows in streams.items():
            for ref_row, sec_row in zip(stream_rows, stream_rows[1:]):
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

                if not dry_run and (not Path(ref_path).exists() or not Path(sec_path).exists()):
                    print(f"Skipping {ref_id} -> {sec_id}: missing raw file on disk.")
                    pairs_skipped += 1
                    continue

                pair_tag = _build_pair_tag(stream_key, ref_t, sec_t)
                pair_out = out_root / pair_tag

                wavelength_mm = resolved_site.wavelength_mm
                if wavelength_mm is None and resolved_site.sensor == "nisar":
                    if Path(ref_path).exists():
                        wavelength_mm = read_nisar_wavelength_mm(
                            Path(ref_path), subdataset, sec_path=Path(sec_path), h5py_mod=h5py_mod
                        )

                cmd = build_engine_command(
                    engine_bin=engine_bin,
                    ref_path=ref_path,
                    sec_path=sec_path,
                    out_dir=pair_out,
                    site=resolved_site,
                    wavelength_mm=wavelength_mm,
                )

                if dry_run:
                    print(f"[DRY-RUN] Would run C++ engine: {pair_tag} ({' '.join(cmd)})")
                    continue

                pair_out.mkdir(parents=True, exist_ok=True)
                wl_info = f", lambda={wavelength_mm:.3f} mm" if wavelength_mm is not None else ""
                print(f"Running C++ engine: {pair_tag} (subdataset={subdataset}{wl_info})")
                subprocess.run(cmd, check=True)
                pairs_processed += 1

                conn.execute(
                    """
                    INSERT OR REPLACE INTO processed_pairs
                        (ref_id, sec_id, out_dir, finished_at, site_id, stream_key, wavelength_mm)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
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

            # 4. Optional disk hygiene: keep latest raw .h5 per stream, prune only forward-paired files
            if resolved_site.prune_raw and not dry_run and len(stream_rows) > 1:
                for old_row in stream_rows[:-1]:
                    old_id = old_row["granule_id"]
                    paired_forward = conn.execute(
                        """
                        SELECT 1 FROM processed_pairs
                        WHERE site_id = ? AND ref_id = ?
                        """,
                        (resolved_site.site_id, old_id),
                    ).fetchone()
                    if not paired_forward:
                        continue

                    p = Path(old_row["local_path"])
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

    return SiteRunSummary(
        site_id=resolved_site.site_id or (resolved_site.burst_id or resolved_site.sensor),
        sensor=resolved_site.sensor,
        discovered_count=len(candidates),
        downloaded_count=downloaded_count,
        pairs_processed=pairs_processed,
        pairs_skipped=pairs_skipped,
        pruned_count=pruned_count,
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

    work_dir_raw = merged.get("work_dir")
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
        work_dir=Path(work_dir_raw) if work_dir_raw else None,
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
    parser.add_argument("--work-dir", type=Path, default=Path("./insar_work"))
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
        }
        sites = load_sites_config(args.sites_config, cli_defaults=cli_defaults)
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
            print(
                f"[{summary.site_id}] done: discovered={summary.discovered_count}, "
                f"downloaded={summary.downloaded_count}, pairs_processed={summary.pairs_processed}, "
                f"pairs_skipped={summary.pairs_skipped}, pruned={summary.pruned_count}"
            )
        except Exception as exc:
            failures += 1
            print(f"[{label}] ERROR: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
