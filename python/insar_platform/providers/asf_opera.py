"""ASF OPERA CSLC-S1 discovery, streaming target resolution, and atomic download."""

from __future__ import annotations

import dataclasses
import os
import re
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from insar_platform.providers.sensors import (
    GranuleCandidate,
    OPERA_DEFAULT_SUBDATASET,
    OPERA_DEFAULT_WAVELENGTH_MM,
    StreamTarget,
    _detect_aws_in_region,
    _extract_bearer_token,
    _format_gdal_hdf5_uri,
    _import_asf,
    _import_earthaccess,
    _login_earthaccess,
    _normalize_timestamp,
    _resolve_https_redirect,
    _s3_creds_env,
    _utc_now,
)

_OPERA_BURST_HYPHEN_RE = re.compile(r"^T\d{3}-\d{6}-IW\d$", re.IGNORECASE)


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
            s3_creds=_s3_creds_env(creds),
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
    site: Any,
    asf_mod: Any = None,
) -> tuple[Any, list[GranuleCandidate]]:
    """Query ASF for OPERA CSLC-S1 granules for a site."""
    asf = asf_mod or _import_asf()
    raw_burst_id = _normalize_opera_burst_id(site.burst_id)
    burst_id = raw_burst_id or discover_burst_id(
        site.lon, site.lat, site.lookback_days, asf_mod=asf
    )
    resolved_site = (
        dataclasses.replace(site, burst_id=burst_id)
        if dataclasses.is_dataclass(site)
        else type(site)(**{**site.__dict__, "burst_id": burst_id})
    )
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
