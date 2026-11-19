"""STAC 1.1 item builders shared by workers and the public catalog API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def build_pair_item(job_id: str, monitor_name: str, finished_at: str, summary: dict[str, Any], base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
    bbox = summary.get("bbox_wgs84")
    geometry = None
    if bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
        geometry = {"type": "Polygon", "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat],
        ]]}
    asset_base = f"{base}/assets/{job_id}"
    item: dict[str, Any] = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": job_id,
        "geometry": geometry,
        "properties": {
            "datetime": finished_at,
            "insar:monitor_name": monitor_name,
            "insar:processing_time_ms": summary.get("processing_time_ms"),
            "insar:valid_pixels": summary.get("valid_pixels"),
            "insar:min_displacement_mm": summary.get("min_displacement_mm"),
            "insar:max_displacement_mm": summary.get("max_displacement_mm"),
            "insar:anomalies_detected": summary.get("anomalies_detected"),
            "insar:ref_region_pinned": summary.get("ref_region_pinned"),
        },
        "links": [{"rel": "self", "href": f"{base}/stac/items/{job_id}", "type": "application/geo+json"}],
        "assets": {
            "displacement_mm": {"href": f"{asset_base}/displacement_mm.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "title": "Unwrapped Line-of-Sight Displacement (mm)", "roles": ["data"]},
            "coherence": {"href": f"{asset_base}/coherence.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "title": "Interferometric Coherence (0-1)", "roles": ["data", "quality"]},
            "wrapped_phase": {"href": f"{asset_base}/wrapped_phase.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "title": "Wrapped Interferometric Phase (rad)", "roles": ["data"]},
            "alerts": {"href": f"{asset_base}/alerts.geojson", "type": "application/geo+json", "title": "High-Coherence Displacement Anomaly Alerts", "roles": ["metadata"]},
        },
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def build_stack_item(
    stack_id: str,
    monitor_name: str,
    updated_at: str,
    summary: dict[str, Any],
    stack_dir: Path,
    base_url: str | None = None,
) -> dict[str, Any]:
    base = (base_url or os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
    asset_base = f"{base}/stack-assets/{stack_id}"
    candidates = [
        ("velocity_mm_yr", "velocity_mm_yr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "LOS Velocity (mm/yr)", ["data"]),
        ("acceleration_mm_yr2", "acceleration_mm_yr2.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "LOS Acceleration (mm/yr^2)", ["data"]),
        ("cumulative_disp_mm", "cumulative_disp_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Cumulative LOS Displacement (mm)", ["data"]),
        ("temporal_coherence", "temporal_coherence.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Mean Temporal Coherence (0-1)", ["data", "quality"]),
        ("velocity_uncertainty_mm_yr", "velocity_uncertainty_mm_yr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Velocity Standard Error (mm/yr)", ["data", "quality"]),
        ("velocity_snr", "velocity_snr.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Velocity Signal-to-Noise Ratio (|v|/sigma_v)", ["data", "quality"]),
        ("ramp_removed_disp_mm", "ramp_removed_disp_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Latest Corrected Pair Displacement (mm)", ["data"]),
        ("closure_residual_mm", "closure_residual_mm.tif", "image/tiff; application=geotiff; profile=cloud-optimized", "Triplet Phase Closure Residual (mm)", ["data", "quality"]),
        ("corridor_report", "corridor_report.geojson", "application/geo+json", "Infrastructure Corridor Risk Report", ["metadata"]),
        ("corridor_alerts", "corridor_alerts.geojson", "application/geo+json", "SNR/Maturity-Gated Corridor Alerts", ["metadata"]),
        ("stack_summary", "stack_summary.json", "application/json", "Stack Summary & Quality Metrics", ["metadata"]),
    ]
    assets: dict[str, dict[str, Any]] = {}
    for key, filename, media_type, title, roles in candidates:
        if (stack_dir / filename).is_file():
            assets[key] = {"href": f"{asset_base}/{filename}", "type": media_type, "title": title, "roles": roles}
    maturity = summary.get("maturity_gating", {})
    corridors = summary.get("corridors", {})
    ramp = summary.get("ramp_removal", {})
    reference = summary.get("reference_region", {})
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": stack_id,
        "geometry": None,
        "properties": {
            "datetime": updated_at,
            "insar:monitor_name": monitor_name,
            "insar:pair_count": summary.get("pair_count"),
            "insar:redundant_pair_count": summary.get("redundant_pair_count"),
            "insar:rejected_pair_count": summary.get("rejected_pair_count"),
            "insar:calendar_span_days": summary.get("calendar_span_days"),
            "insar:valid_stack_pixels": summary.get("valid_stack_pixels"),
            "insar:mean_temporal_coherence": summary.get("mean_temporal_coherence"),
            "insar:p95_abs_velocity_mm_yr": summary.get("p95_abs_velocity_mm_yr"),
            "insar:dem_tropo_applied": ramp.get("dem_applied"),
            "insar:ref_region_applied": reference.get("applied"),
            "insar:stack_mature": maturity.get("stack_mature"),
            "insar:critical_corridors": corridors.get("critical_count"),
            "insar:corridor_alerts": corridors.get("corridor_alerts"),
        },
        "links": [{"rel": "self", "href": f"{base}/stac/stacks/{stack_id}", "type": "application/geo+json"}],
        "assets": assets,
    }
