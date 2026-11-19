"""STAC 1.1.0 catalog and search endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from insar_platform.api.deps import get_base_url, get_data_dir, read_s3_json_asset, validate_safe_id
from insar_platform.storage.database import get_db

router = APIRouter()


def _load_stac_item(owner_type: str, subdir: str, obj_id: str, id_label: str, not_found_detail: str) -> JSONResponse:
    validate_safe_id(obj_id, id_label)
    stac_path = get_data_dir() / subdir / obj_id / "stac_item.json"
    if stac_path.is_file():
        item = json.loads(stac_path.read_text(encoding="utf-8"))
    else:
        item = read_s3_json_asset(owner_type, obj_id, "stac_item.json")
    if item is None:
        raise HTTPException(status_code=404, detail=not_found_detail)
    return JSONResponse(content=item, media_type="application/geo+json")


@router.get("/stac/items/{job_id}")
def get_stac_item(job_id: str):
    return _load_stac_item("job", "jobs", job_id, "job_id", "STAC item not ready or job not found")


@router.get("/stac")
def stac_catalog():
    base_url = get_base_url()
    return {
        "type": "Catalog",
        "id": "insar-platform",
        "stac_version": "1.1.0",
        "description": "InSAR pair products and time-series stack results",
        "links": [
            {"rel": "self", "href": f"{base_url}/stac", "type": "application/json"},
            {"rel": "data", "href": f"{base_url}/stac/collections", "type": "application/json"},
            {"rel": "search", "href": f"{base_url}/stac/search", "method": "POST", "type": "application/geo+json"},
        ],
    }


@router.get("/stac/ingestions/{ingestion_id}")
def get_ingestion_stac_item(ingestion_id: str):
    return _load_stac_item(
        "ingestion", "ingestions", ingestion_id, "ingestion_id", "STAC item not ready or ingestion not found"
    )


@router.get("/stac/collections")
def stac_collections():
    base_url = get_base_url()
    return {
        "collections": [{
            "id": "insar-results",
            "type": "Collection",
            "stac_version": "1.1.0",
            "description": "Pairwise and time-series InSAR products",
            "license": "proprietary",
            "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}, "temporal": {"interval": [[None, None]]}},
            "links": [
                {"rel": "self", "href": f"{base_url}/stac/collections/insar-results", "type": "application/json"},
                {"rel": "items", "href": f"{base_url}/stac/collections/insar-results/items", "type": "application/geo+json"},
            ],
        }]
    }


@router.get("/stac/collections/insar-results")
def get_stac_collection():
    return stac_collections()["collections"][0]


def _stac_search_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    limit = max(1, min(int(payload.get("limit", 100)), 1000))
    requested_ids = set(payload.get("ids") or [])
    bbox = payload.get("bbox")
    data_dir = get_data_dir()
    candidates: list[tuple[str, str]] = []
    with get_db() as conn:
        if bbox and len(bbox) == 4 and conn.dialect == "postgresql":
            args = tuple(float(value) for value in bbox)
            matches = conn.execute(
                """SELECT DISTINCT owner_type, owner_id FROM assets
                   WHERE footprint && ST_MakeEnvelope(?, ?, ?, ?, 4326)
                     AND ST_Intersects(footprint, ST_MakeEnvelope(?, ?, ?, ?, 4326))
                   ORDER BY owner_type, owner_id LIMIT ?""",
                args + args + (limit,),
            ).fetchall()
            candidates.extend((row["owner_type"], row["owner_id"]) for row in matches)
        if not candidates:
            jobs = conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'completed' ORDER BY finished_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            stacks = conn.execute(
                "SELECT stack_id FROM stacks WHERE status = 'completed' ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            candidates.extend(("job", row["job_id"]) for row in jobs)
            candidates.extend(("stack", row["stack_id"]) for row in stacks)

    items: list[dict[str, Any]] = []
    for owner_type, owner_id in candidates:
        if requested_ids and owner_id not in requested_ids:
            continue
        local = data_dir / ("jobs" if owner_type == "job" else "stacks") / owner_id / "stac_item.json"
        item = json.loads(local.read_text()) if local.is_file() else read_s3_json_asset(owner_type, owner_id, "stac_item.json")
        if item is None:
            continue
        if bbox and item.get("bbox"):
            left, bottom, right, top = item["bbox"]
            if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
                continue
        items.append(item)
        if len(items) >= limit:
            break
    return items


@router.post("/stac/search")
def stac_search(payload: dict[str, Any]):
    items = _stac_search_items(payload)
    return {"type": "FeatureCollection", "stac_version": "1.1.0", "features": items, "links": []}


@router.get("/stac/collections/insar-results/items")
def stac_collection_items(limit: int = 100):
    return {"type": "FeatureCollection", "stac_version": "1.1.0", "features": _stac_search_items({"limit": limit}), "links": []}


@router.get("/stac/stacks/{stack_id}")
def get_stack_stac_item(stack_id: str):
    return _load_stac_item("stack", "stacks", stack_id, "stack_id", "Stack STAC item not ready or stack not found")
