"""Asset streaming routes for job and stack outputs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from insar_platform.api.deps import get_data_dir, s3_asset_response, validate_safe_id

router = APIRouter()

ALLOWED_ASSETS = {
    "displacement_mm.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "coherence.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "wrapped_phase.tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    "alerts.geojson": "application/geo+json",
    "summary.json": "application/json",
}

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


def _serve_asset(
    *,
    owner_type: str,
    owner_id: str,
    id_label: str,
    subdir: str,
    filename: str,
    allowed_map: dict[str, str],
    invalid_detail: str,
    not_found_detail: str,
    request: Request,
):
    validate_safe_id(owner_id, id_label)
    if filename not in allowed_map:
        raise HTTPException(status_code=400, detail=invalid_detail)
    media_type = allowed_map[filename]
    response = s3_asset_response(owner_type, owner_id, filename, media_type, request)
    if response is not None:
        return response
    file_path = get_data_dir() / subdir / owner_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=not_found_detail)
    return FileResponse(file_path, media_type=media_type)


@router.api_route("/assets/{job_id}/{filename}", methods=["GET", "HEAD"])
def stream_asset(job_id: str, filename: str, request: Request):
    return _serve_asset(
        owner_type="job",
        owner_id=job_id,
        id_label="job_id",
        subdir="jobs",
        filename=filename,
        allowed_map=ALLOWED_ASSETS,
        invalid_detail="Invalid asset request",
        not_found_detail="Asset not found",
        request=request,
    )


@router.api_route("/stack-assets/{stack_id}/{filename}", methods=["GET", "HEAD"])
def stream_stack_asset(stack_id: str, filename: str, request: Request):
    return _serve_asset(
        owner_type="stack",
        owner_id=stack_id,
        id_label="stack_id",
        subdir="stacks",
        filename=filename,
        allowed_map=ALLOWED_STACK_ASSETS,
        invalid_detail="Invalid stack asset request",
        not_found_detail="Stack asset not found",
        request=request,
    )
