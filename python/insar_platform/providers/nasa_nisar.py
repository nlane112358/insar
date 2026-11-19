"""NASA Earthdata / NISAR GSLC discovery, HDF5 metadata inspection, and streaming."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlparse

from insar_platform.providers.sensors import (
    GranuleCandidate,
    NISAR_DEFAULT_SHORT_NAME,
    NISAR_DEFAULT_SUBDATASET,
    NISAR_DEFAULT_WAVELENGTH_MM,
    SPEED_OF_LIGHT_M_S,
    StreamTarget,
    _detect_aws_in_region,
    _format_gdal_hdf5_uri,
    _import_earthaccess,
    _import_fsspec,
    _import_h5py,
    _login_earthaccess,
    _normalize_direction,
    _normalize_timestamp,
    _resolve_https_redirect,
    _s3_creds_env,
    _utc_now,
)

_NISAR_UR_PATTERN = re.compile(r"_(?:GSLC|RSLC)_(\d{3})_(\d{3})_([AD])_(\d{3})_")


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
            s3_creds=_s3_creds_env(creds),
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
    site: Any,
    ea_mod: Any = None,
) -> tuple[Any, list[GranuleCandidate]]:
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


@contextlib.contextmanager
def _open_h5_source(
    source: Path | StreamTarget,
    h5py_mod: Any = None,
    fsspec_mod: Any = None,
) -> Iterator[Any]:
    """Open either a local HDF5 Path or a remote StreamTarget via h5py."""
    h5py = h5py_mod or _import_h5py()
    if isinstance(source, Path):
        with h5py.File(source, "r") as hf:
            yield hf
    else:
        with _open_remote_h5(source, h5py_mod=h5py, fsspec_mod=fsspec_mod) as hf:
            yield hf


def _inspect_pair_metadata(
    ref_source: Path | StreamTarget,
    sec_source: Path | StreamTarget,
    site: Any,
    subdataset: str,
    h5py_mod: Any = None,
    fsspec_mod: Any = None,
) -> tuple[float | None, tuple[int, int, int, int] | None]:
    """Read HDF5 header metadata once for NISAR wavelength and optional bbox auto-window."""
    need_wavelength = site.wavelength_mm is None and site.sensor == "nisar"
    need_window = site.stream and site.window is None and site.bbox is not None
    if not need_wavelength and not need_window:
        return site.wavelength_mm, site.window

    wavelength_mm = site.wavelength_mm
    window = site.window

    with _open_h5_source(ref_source, h5py_mod=h5py_mod, fsspec_mod=fsspec_mod) as ref_hf:
        if need_wavelength:
            freq_hz = _read_center_frequency_hz_from_h5file(ref_hf, subdataset)
            if freq_hz is None:
                print(
                    f"Warning: centerFrequency not found in HDF5 header; "
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
            with _open_h5_source(sec_source, h5py_mod=h5py_mod, fsspec_mod=fsspec_mod) as sec_hf:
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
