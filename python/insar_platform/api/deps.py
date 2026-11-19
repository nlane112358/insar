"""Shared FastAPI configuration helpers, authentication, and S3/GeoJSON utilities."""

from __future__ import annotations

import hmac
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from insar_platform.storage.database import get_db
from insar_platform.storage.object_store import enabled as s3_enabled
from insar_platform.storage.object_store import (
    get_object as get_s3_object,
    head_object as head_s3_object,
    iter_body as iter_s3_body,
)

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SUPPORTED_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }
)


def get_engine_bin() -> Path:
    return Path(os.getenv("INSAR_ENGINE_BIN", "./build/insar_proto"))


def get_stack_bin() -> Path:
    return Path(os.getenv("INSAR_STACK_BIN", "./build/insar_stack"))


def get_data_dir() -> Path:
    return Path(os.getenv("INSAR_DATA_DIR", "./var/api_data"))


def get_api_key() -> str:
    return os.getenv("INSAR_API_KEY", "insar_dev_secret_key")


def get_webhook_secret() -> str:
    return os.getenv("INSAR_WEBHOOK_SECRET", "whsec_local_dev_signing_secret")


def get_base_url() -> str:
    return os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if not hmac.compare_digest(x_api_key, get_api_key()):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return x_api_key


def validate_safe_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value) or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return value


def validate_geojson_geometry(geometry: dict[str, Any]) -> None:
    if geometry.get("type") not in SUPPORTED_GEOMETRY_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported GeoJSON geometry type")


def coords_bounds(value: Any) -> tuple[float, float, float, float] | None:
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


def s3_asset_response(
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


def read_s3_json_asset(owner_type: str, owner_id: str, filename: str) -> dict | None:
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
