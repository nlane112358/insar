"""Shared sensor provider interfaces, stream descriptors, and Earthdata auth helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

SPEED_OF_LIGHT_M_S = 299_792_458.0
OPERA_DEFAULT_SUBDATASET = "/data/VV"
OPERA_DEFAULT_WAVELENGTH_MM = 55.4658
NISAR_DEFAULT_SUBDATASET = "/science/LSAR/GSLC/grids/frequencyA/HH"
NISAR_DEFAULT_SHORT_NAME = "NISAR_L2_GSLC_V1"
NISAR_DEFAULT_WAVELENGTH_MM = 238.4


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
    download_fn: Callable[[Any], Any]
    stream_url: str = ""
    stream_fn: Callable[[str], StreamTarget] | None = None


def _import_asf() -> Any:
    try:
        import asf_search as asf
    except ImportError as exc:
        raise RuntimeError(
            "asf_search is required for OPERA CSLC-S1 ingestion. "
            "Install dependencies with: pip install -r deploy/requirements-ingest.txt"
        ) from exc
    return asf


def _import_earthaccess() -> Any:
    try:
        import earthaccess
    except ImportError as exc:
        raise RuntimeError(
            "earthaccess is required for NISAR ingestion. "
            "Install dependencies with: pip install -r deploy/requirements-ingest.txt"
        ) from exc
    return earthaccess


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read HDF5 radar centerFrequency metadata. "
            "Install dependencies with: pip install -r deploy/requirements-ingest.txt"
        ) from exc
    return h5py


def _import_fsspec() -> Any:
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "fsspec is required for remote HDF5 metadata streaming. "
            "Install dependencies with: pip install -r deploy/requirements-ingest.txt"
        ) from exc
    return fsspec


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


def _s3_creds_env(creds: Mapping[str, Any]) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": str(creds["accessKeyId"]),
        "AWS_SECRET_ACCESS_KEY": str(creds["secretAccessKey"]),
        "AWS_SESSION_TOKEN": str(creds["sessionToken"]),
        "AWS_REGION": "us-west-2",
    }


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


@runtime_checkable
class SensorProvider(Protocol):
    sensor_name: str

    def discover(self, site: Any) -> tuple[Any, list[Any]]: ...


@dataclass
class OperaProvider:
    """ASF OPERA adapter; authentication and download logic stays injectable."""

    asf_module: Any = None
    sensor_name: str = "opera"

    def discover(self, site: Any) -> tuple[Any, list[GranuleCandidate]]:
        from insar_platform.providers.asf_opera import fetch_opera_candidates

        return fetch_opera_candidates(site, asf_mod=self.asf_module)


@dataclass
class NisarProvider:
    """NASA Earthdata/NISAR adapter with injectable Earthaccess client."""

    earthaccess_module: Any = None
    sensor_name: str = "nisar"

    def discover(self, site: Any) -> tuple[Any, list[GranuleCandidate]]:
        from insar_platform.providers.nasa_nisar import fetch_nisar_candidates

        return fetch_nisar_candidates(site, ea_mod=self.earthaccess_module)


def sensor_provider(sensor: str, *, asf_module: Any = None, earthaccess_module: Any = None) -> SensorProvider:
    normalized = sensor.strip().lower()
    if normalized == "opera":
        return OperaProvider(asf_module)
    if normalized == "nisar":
        return NisarProvider(earthaccess_module)
    raise ValueError(f"Unsupported sensor provider: {sensor}")
