"""S3-compatible immutable artifact storage used by workers and the API."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import urlparse


@dataclass(frozen=True)
class StoredObject:
    key: str
    uri: str
    media_type: str
    checksum: str
    size: int
    asset_key: str | None = None


def enabled() -> bool:
    return os.getenv("INSAR_STORAGE_BACKEND", "filesystem").lower() == "s3"


def _settings() -> tuple[str, str, str, str]:
    return (
        os.getenv("INSAR_S3_BUCKET", "insar-artifacts"),
        os.getenv("INSAR_S3_ENDPOINT_URL", "http://127.0.0.1:9000"),
        os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )


def _client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3 storage requires boto3") from exc
    from botocore.config import Config

    _, endpoint, access_key, secret_key = _settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    bucket, _, _, _ = _settings()
    client = _client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def upload_file(path: Path, key: str, *, asset_key: str | None = None) -> StoredObject:
    bucket, _, _, _ = _settings()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    _client().upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": media_type, "Metadata": {"sha256": checksum}},
    )
    return StoredObject(key, f"s3://{bucket}/{key}", media_type, checksum, path.stat().st_size, asset_key)


def upload_directory(root: Path, prefix: str, *, version: str | None = None) -> list[StoredObject]:
    if not enabled():
        return []
    ensure_bucket()
    uploaded: list[StoredObject] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            version_prefix = f"/{version.strip('/')}" if version else ""
            uploaded.append(upload_file(path, f"{prefix.rstrip('/')}{version_prefix}/{rel}", asset_key=rel))
    return uploaded


def register_artifacts(
    owner_type: str,
    owner_id: str,
    objects: list[StoredObject],
    footprint: dict | None = None,
) -> None:
    if not objects:
        return
    from .database import get_db

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prefix = f"{owner_type}/{owner_id}/"
    footprint_json = json.dumps(footprint, separators=(",", ":")) if footprint else None
    with get_db() as conn:
        for item in objects:
            common = (
                hashlib.sha256(f"{owner_type}:{owner_id}:{item.key}".encode()).hexdigest(),
                owner_type,
                owner_id,
                item.asset_key or item.key.removeprefix(prefix),
                item.media_type,
                item.uri,
                item.checksum,
                now,
            )
            if conn.dialect == "postgresql":
                conn.execute(
                    """INSERT INTO assets
                       (asset_id, owner_type, owner_id, asset_key, media_type, s3_uri, checksum, created_at, footprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ST_SetSRID(ST_GeomFromGeoJSON(?::text), 4326))
                       ON CONFLICT (owner_type, owner_id, asset_key) DO UPDATE SET
                           media_type = excluded.media_type, s3_uri = excluded.s3_uri,
                           checksum = excluded.checksum, created_at = excluded.created_at,
                           footprint = excluded.footprint""",
                    common + (footprint_json,),
                )
            else:
                conn.execute(
                    """INSERT INTO assets
                       (asset_id, owner_type, owner_id, asset_key, media_type, s3_uri, checksum, created_at, footprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (owner_type, owner_id, asset_key) DO UPDATE SET
                           media_type = excluded.media_type, s3_uri = excluded.s3_uri,
                           checksum = excluded.checksum, created_at = excluded.created_at,
                           footprint = excluded.footprint""",
                    common + (footprint_json,),
                )
        conn.commit()


def split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_object(uri: str, byte_range: str | None = None):
    bucket, key = split_uri(uri)
    params = {"Bucket": bucket, "Key": key}
    if byte_range:
        params["Range"] = byte_range
    return _client().get_object(**params)


def download_file(uri: str, destination: Path) -> Path:
    """Stream one stored object to a local scratch path without buffering it in memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    response = get_object(uri)
    try:
        with temporary.open("wb") as output:
            while chunk := response["Body"].read(1024 * 1024):
                output.write(chunk)
    finally:
        response["Body"].close()
    os.replace(temporary, destination)
    return destination


def head_object(uri: str):
    bucket, key = split_uri(uri)
    return _client().head_object(Bucket=bucket, Key=key)


def iter_body(body, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    try:
        while chunk := body.read(chunk_size):
            yield chunk
    finally:
        body.close()
