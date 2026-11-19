#!/usr/bin/env python3
"""
UK sector & hazard analytics for NISAR L-band InSAR stacks.

Rolls up stack COGs and corridor analytics into UK stakeholder domains:
  - UK Rail (Network Rail mainlines, London Underground, DLR, Croydon & NET trams)
  - Water Utilities / AMP8 (reservoirs, earth dams, WTWs/STWs, aqueducts, canals, rivers)
  - Home Insurance & Geological Exposure (borough/district shrink-swell & coalfield metrics)

Explicitly preserves SNR and stack-maturity gating so immature 12-24 day stacks are
reported as baseline/provisional observations rather than false-positive long-term rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


SECTOR_GROUPS: dict[str, set[str]] = {
    "uk_rail": {"rail", "metro", "light_rail"},
    "water_amp8": {"reservoir", "dam", "water_works", "canal", "river"},
    "strategic_transport": {"motorway", "trunk", "airport_runway"},
}

SECTOR_LABELS: dict[str, str] = {
    "uk_rail": "UK Rail (Mainlines, Metro & Light Rail)",
    "water_amp8": "Water Utilities (AMP8 Reservoirs, Dams, WTWs & Waterways)",
    "strategic_transport": "Strategic Transport (Motorways, Trunk Roads & Runways)",
}

UK_SITE_PROFILES: dict[str, dict[str, str]] = {
    "london_thames": {
        "display_name": "Greater London & Thames Valley, UK",
        "sensor": "NISAR L-band GSLC",
        "phenomenology": "London Clay shrink-swell seasonal cycles & urban ground motion",
        "reference_geology": "Cretaceous Chalk Group bedrock (Chilterns / North Downs)",
        "target_sectors": "UK rail, water utilities (AMP8), home insurance",
    },
    "east_midlands_coalfields": {
        "display_name": "East Midlands Coalfields, UK",
        "sensor": "NISAR L-band GSLC",
        "phenomenology": "Historic coal-mine uplift (mine-water rebound) & legacy settlement",
        "reference_geology": "Triassic Mercia Mudstone (BGS Keyworth) / Peak District Gritstone",
        "target_sectors": "UK rail, water utilities (AMP8), home insurance",
    },
}


from insar_platform.paths import ensure_runtime_python, resolve_runtime_path


def _ensure_gdal_runtime(script_file: str, argv: Sequence[str] | None = None) -> None:
    ensure_runtime_python(script_file, ("numpy", "osgeo"), argv, set_mpl_cache=False)
    try:
        import numpy  # noqa: F401
        from osgeo import gdal, ogr, osr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("uk_sectors requires numpy and GDAL (osgeo).") from exc


def _load_raster_band(path: Path) -> tuple[Any, tuple[float, float, float, float, float, float], str]:
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = tuple(float(v) for v in ds.GetGeoTransform())
    wkt = ds.GetProjection() or ""
    ds = None
    return arr, gt, wkt


def _finite_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def summarize_corridor_sectors(corridor_features: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up corridor_report.geojson metrics into UK Rail, AMP8 Water, and Transport sectors."""
    import numpy as np

    sectors_out: dict[str, Any] = {}
    for sector_key, classes in SECTOR_GROUPS.items():
        matched = [
            f.get("properties", {})
            for f in corridor_features
            if str(f.get("properties", {}).get("asset_class", "")) in classes
        ]
        covered = [p for p in matched if int(p.get("valid_pixels") or 0) > 0]
        total_px = sum(int(p.get("pixel_count") or 0) for p in matched)
        valid_px = sum(int(p.get("valid_pixels") or 0) for p in covered)
        cohs = [float(p["mean_coherence"]) for p in covered if _finite_float(p.get("mean_coherence")) is not None]
        weights = [int(p.get("valid_pixels") or 0) for p in covered if _finite_float(p.get("mean_coherence")) is not None]
        p95_vels = [float(p["p95_abs_velocity_mm_yr"]) for p in covered if _finite_float(p.get("p95_abs_velocity_mm_yr")) is not None]
        mean_vels = [float(p["mean_velocity_mm_yr"]) for p in covered if _finite_float(p.get("mean_velocity_mm_yr")) is not None]
        max_disps = [float(p["max_abs_cumulative_disp_mm"]) for p in covered if _finite_float(p.get("max_abs_cumulative_disp_mm")) is not None]
        mean_disps = [float(p["mean_cumulative_disp_mm"]) for p in covered if _finite_float(p.get("mean_cumulative_disp_mm")) is not None]

        risk_counts: dict[str, int] = {}
        for p in matched:
            rl = str(p.get("risk_level") or "unknown")
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

        top_assets = sorted(
            covered,
            key=lambda p: (
                -(_finite_float(p.get("p95_abs_velocity_mm_yr")) or 0.0),
                -(_finite_float(p.get("max_abs_cumulative_disp_mm")) or 0.0),
            ),
        )[:8]

        weighted_coh = float(np.average(cohs, weights=weights)) if (cohs and sum(weights) > 0) else None
        sectors_out[sector_key] = {
            "label": SECTOR_LABELS[sector_key],
            "asset_classes": sorted(classes),
            "total_assets": len(matched),
            "covered_assets": len(covered),
            "total_pixels": total_px,
            "valid_pixels": valid_px,
            "coherent_coverage_pct": round(100.0 * valid_px / total_px, 2) if total_px > 0 else 0.0,
            "weighted_mean_coherence": round(weighted_coh, 4) if weighted_coh is not None else None,
            "median_p95_abs_vel_mm_yr": round(float(np.median(p95_vels)), 2) if p95_vels else None,
            "max_p95_abs_vel_mm_yr": round(float(np.max(p95_vels)), 2) if p95_vels else None,
            "median_abs_mean_vel_mm_yr": round(float(np.median(np.abs(mean_vels))), 2) if mean_vels else None,
            "median_abs_mean_disp_mm": round(float(np.median(np.abs(mean_disps))), 2) if mean_disps else None,
            "max_abs_disp_mm": round(float(np.max(max_disps)), 2) if max_disps else None,
            "maturity_gated_assets": sum(1 for p in covered if p.get("maturity_gated")),
            "risk_breakdown": risk_counts,
            "top_assets": [
                {
                    "asset_id": p.get("asset_id"),
                    "name": p.get("name"),
                    "asset_class": p.get("asset_class"),
                    "valid_pixels": int(p.get("valid_pixels") or 0),
                    "coverage_ratio": _finite_float(p.get("coverage_ratio")),
                    "mean_coherence": _finite_float(p.get("mean_coherence")),
                    "mean_velocity_mm_yr": _finite_float(p.get("mean_velocity_mm_yr")),
                    "p95_abs_velocity_mm_yr": _finite_float(p.get("p95_abs_velocity_mm_yr")),
                    "mean_cumulative_disp_mm": _finite_float(p.get("mean_cumulative_disp_mm")),
                    "max_abs_cumulative_disp_mm": _finite_float(p.get("max_abs_cumulative_disp_mm")),
                    "risk_level": p.get("risk_level"),
                    "maturity_gated": bool(p.get("maturity_gated")),
                }
                for p in top_assets
            ],
        }
    return sectors_out


def summarize_insurance_zones(
    zones_path: Path,
    *,
    vel: Any,
    cum: Any,
    coh: Any,
    snr: Any | None,
    gt: tuple[float, float, float, float, float, float],
    proj_wkt: str,
    stack_mature: bool,
    min_coh: float = 0.35,
) -> list[dict[str, Any]]:
    """Compute residential / borough exposure metrics from a polygon GeoJSON for home insurance."""
    import numpy as np
    from osgeo import gdal, ogr, osr

    if not zones_path.is_file():
        raise FileNotFoundError(f"Zones GeoJSON not found: {zones_path}")
    if not proj_wkt:
        raise RuntimeError("Stack raster lacks CRS WKT; cannot project zone polygons.")

    h, w = vel.shape
    pix_area_km2 = abs(gt[1] * gt[5]) / 1e6

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(proj_wkt)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    vds = ogr.Open(str(zones_path))
    if vds is None:
        raise RuntimeError(f"Failed to open zones vector file: {zones_path}")
    layer = vds.GetLayer(0)
    src_srs = layer.GetSpatialRef()
    if src_srs is None:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
    else:
        src_srs = src_srs.Clone()
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src_srs, dst_srs)

    mem_drv = gdal.GetDriverByName("MEM")
    mask_ds = mem_drv.Create("", w, h, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt)
    mask_ds.SetProjection(proj_wkt)

    mem_vec_drv = ogr.GetDriverByName("Memory")

    results: list[dict[str, Any]] = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        name = feat.GetField("name") or feat.GetField("asset_id") or "Zone"
        zone_id = feat.GetField("asset_id") or name.lower().replace(" ", "_")

        g_proj = geom.Clone()
        if g_proj.Transform(ct) != 0:
            continue

        # Fast bounding check
        minx, maxx, miny, maxy = g_proj.GetEnvelope()
        r_xmin = min(gt[0], gt[0] + w * gt[1])
        r_xmax = max(gt[0], gt[0] + w * gt[1])
        r_ymin = min(gt[3], gt[3] + h * gt[5])
        r_ymax = max(gt[3], gt[3] + h * gt[5])
        if maxx < r_xmin or minx > r_xmax or maxy < r_ymin or miny > r_ymax:
            continue

        tmp_vds = mem_vec_drv.CreateDataSource("mem_zone")
        tmp_lyr = tmp_vds.CreateLayer("poly", srs=dst_srs, geom_type=ogr.wkbUnknown)
        f_out = ogr.Feature(tmp_lyr.GetLayerDefn())
        f_out.SetGeometry(g_proj)
        tmp_lyr.CreateFeature(f_out)

        mask_ds.GetRasterBand(1).Fill(0)
        gdal.RasterizeLayer(mask_ds, [1], tmp_lyr, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        mask = mask_ds.GetRasterBand(1).ReadAsArray() == 1
        tmp_vds = None

        total_zone_px = int(mask.sum())
        if total_zone_px < 25:
            continue

        valid = mask & np.isfinite(cum) & np.isfinite(coh) & (coh >= min_coh)
        valid_px = int(valid.sum())
        if valid_px < 20:
            continue

        d_vals = cum[valid]
        c_vals = coh[valid]
        v_vals = vel[valid & np.isfinite(vel)]
        high_coh_px = int((c_vals >= 0.55).sum())

        p5, p25, p50, p75, p95 = [float(x) for x in np.percentile(d_vals, [5, 25, 50, 75, 95])]
        p95_abs_disp = float(np.percentile(np.abs(d_vals), 95.0))
        p95_abs_vel = float(np.percentile(np.abs(v_vals), 95.0)) if v_vals.size else 0.0
        mean_vel = float(np.mean(v_vals)) if v_vals.size else 0.0

        # Sign convention: insar_proto / insar_stack outputs range change (+ = away / settlement, - = toward / uplift)
        uplift_frac = float((d_vals <= -2.0).sum()) / float(valid_px)
        settlement_frac = float((d_vals >= 2.0).sum()) / float(valid_px)
        spread_p95_p5 = p95 - p5
        iqr_disp = p75 - p25

        sig_frac = 0.0
        if snr is not None:
            s_vals = snr[valid & np.isfinite(snr)]
            if s_vals.size:
                sig_frac = float((s_vals >= 2.5).sum()) / float(s_vals.size)

        if not stack_mature:
            tier = "provisional_immature"
        elif spread_p95_p5 >= 15.0 or p95_abs_vel >= 15.0:
            tier = "elevated"
        elif spread_p95_p5 >= 8.0 or p95_abs_vel >= 8.0:
            tier = "moderate"
        else:
            tier = "baseline"

        results.append(
            {
                "zone_id": zone_id,
                "name": name,
                "total_pixels": total_zone_px,
                "valid_pixels": valid_px,
                "coherent_coverage_pct": round(100.0 * valid_px / total_zone_px, 2),
                "coherent_area_km2": round(valid_px * pix_area_km2, 2),
                "mean_coherence": round(float(np.mean(c_vals)), 4),
                "high_coh_pct": round(100.0 * high_coh_px / valid_px, 2),
                "median_disp_mm": round(p50, 2),
                "p95_abs_disp_mm": round(p95_abs_disp, 2),
                "iqr_disp_mm": round(iqr_disp, 2),
                "p95_p5_spread_mm": round(spread_p95_p5, 2),
                "mean_vel_mm_yr": round(mean_vel, 2),
                "p95_abs_vel_mm_yr": round(p95_abs_vel, 2),
                "uplift_pct": round(100.0 * uplift_frac, 2),
                "settlement_pct": round(100.0 * settlement_frac, 2),
                "snr_ge_2_5_pct": round(100.0 * sig_frac, 2),
                "exposure_tier": tier,
            }
        )

    mask_ds = None
    vds = None
    results.sort(key=lambda r: (-r["p95_p5_spread_mm"], -r["valid_pixels"]))
    return results


def analyze_uk_stack(
    stack_dir: Path,
    *,
    profile: str | None = None,
    zones_path: Path | None = None,
    min_coh: float = 0.35,
    out_json: Path | None = None,
    out_zones_csv: Path | None = None,
    out_sectors_csv: Path | None = None,
) -> dict[str, Any]:
    """Generate comprehensive UK Rail, Water AMP8, and Home Insurance report for a stack."""
    import numpy as np

    stack_dir = resolve_runtime_path(stack_dir)
    if zones_path is not None:
        zones_path = resolve_runtime_path(zones_path)

    summary_p = stack_dir / "stack_summary.json"
    ledger_p = stack_dir / "stack_ledger.json"
    corridor_p = stack_dir / "corridor_report.geojson"

    summary = json.loads(summary_p.read_text(encoding="utf-8")) if summary_p.is_file() else {}
    ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.is_file() else {}
    corridor_feats = (
        json.loads(corridor_p.read_text(encoding="utf-8")).get("features", [])
        if corridor_p.is_file()
        else []
    )

    vel, gt, proj_wkt = _load_raster_band(stack_dir / "velocity_mm_yr.tif")
    cum, _, _ = _load_raster_band(stack_dir / "cumulative_disp_mm.tif")
    coh, _, _ = _load_raster_band(stack_dir / "temporal_coherence.tif")
    snr_p = stack_dir / "velocity_snr.tif"
    snr = _load_raster_band(snr_p)[0] if snr_p.is_file() else None
    acc_p = stack_dir / "acceleration_mm_yr2.tif"
    acc = _load_raster_band(acc_p)[0] if acc_p.is_file() else None

    valid = np.isfinite(cum) & np.isfinite(coh) & (coh >= min_coh)
    total_footprint = int(np.isfinite(coh).sum())
    valid_px = int(valid.sum())
    pix_area_km2 = abs(gt[1] * gt[5]) / 1e6

    coh_vals = coh[np.isfinite(coh)]
    disp_vals = cum[valid] if valid_px > 0 else np.array([], dtype=np.float64)
    vel_vals = vel[valid & np.isfinite(vel)] if valid_px > 0 else np.array([], dtype=np.float64)
    acc_vals = acc[valid & np.isfinite(acc)] if (acc is not None and valid_px > 0) else np.array([], dtype=np.float64)

    mat_cfg = summary.get("maturity_gating", {})
    stack_mature = bool(mat_cfg.get("stack_mature", False))

    scene_metrics = {
        "footprint_pixels": total_footprint,
        "valid_coherent_pixels": valid_px,
        "coherent_area_km2": round(valid_px * pix_area_km2, 2),
        "coherent_fraction_pct": round(100.0 * valid_px / total_footprint, 2) if total_footprint > 0 else 0.0,
        "mean_coherence": round(float(np.mean(coh_vals)), 4) if coh_vals.size else None,
        "median_coherence": round(float(np.median(coh_vals)), 4) if coh_vals.size else None,
        "high_coh_050_pct": round(100.0 * float((coh_vals >= 0.50).sum()) / coh_vals.size, 2) if coh_vals.size else 0.0,
        "high_coh_065_pct": round(100.0 * float((coh_vals >= 0.65).sum()) / coh_vals.size, 2) if coh_vals.size else 0.0,
        "median_disp_mm": round(float(np.median(disp_vals)), 2) if disp_vals.size else None,
        "p95_abs_disp_mm": round(float(np.percentile(np.abs(disp_vals), 95.0)), 2) if disp_vals.size else None,
        "p95_p5_disp_spread_mm": round(float(np.percentile(disp_vals, 95.0) - np.percentile(disp_vals, 5.0)), 2) if disp_vals.size else None,
        "p95_abs_vel_mm_yr": round(float(np.percentile(np.abs(vel_vals), 95.0)), 2) if vel_vals.size else None,
        "uplift_area_pct": round(100.0 * float((disp_vals <= -2.0).sum()) / disp_vals.size, 2) if disp_vals.size else 0.0,
        "settlement_area_pct": round(100.0 * float((disp_vals >= 2.0).sum()) / disp_vals.size, 2) if disp_vals.size else 0.0,
        "acceleration_available": bool(acc_vals.size > 0),
        "stack_mature": stack_mature,
        "actual_span_days": mat_cfg.get("actual_span_days", summary.get("calendar_span_days")),
        "min_alert_span_days": mat_cfg.get("min_alert_span_days"),
        "pair_count": summary.get("pair_count"),
        "redundant_pair_count": summary.get("redundant_pair_count", 0),
        "rejected_pair_count": summary.get("rejected_pair_count", 0),
    }

    sectors = summarize_corridor_sectors(corridor_feats)
    zones = (
        summarize_insurance_zones(
            zones_path,
            vel=vel,
            cum=cum,
            coh=coh,
            snr=snr,
            gt=gt,
            proj_wkt=proj_wkt,
            stack_mature=stack_mature,
            min_coh=min_coh,
        )
        if zones_path is not None
        else []
    )

    report = {
        "site_profile": UK_SITE_PROFILES.get(profile or "", {"display_name": stack_dir.name}),
        "stack_dir": str(stack_dir),
        "stream_key": ledger.get("stream_key"),
        "pair_ids": ledger.get("pair_ids", []),
        "redundant_pair_ids": ledger.get("redundant_pair_ids", []),
        "reference_region": summary.get("reference_region", {}),
        "ramp_removal": summary.get("ramp_removal", {}),
        "triplet_closure": summary.get("triplet_closure", {}),
        "scene_lband_metrics": scene_metrics,
        "sectors": sectors,
        "home_insurance_zones": zones,
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote UK sector report JSON -> {out_json}")

    if out_zones_csv is not None and zones:
        out_zones_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_zones_csv, "w", encoding="utf-8", newline="") as fh:
            w_csv = csv.DictWriter(fh, fieldnames=list(zones[0].keys()))
            w_csv.writeheader()
            w_csv.writerows(zones)
        print(f"Wrote UK home insurance zones CSV -> {out_zones_csv}")

    if out_sectors_csv is not None:
        out_sectors_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "sector": sk,
                "label": sdata["label"],
                "total_assets": sdata["total_assets"],
                "covered_assets": sdata["covered_assets"],
                "valid_pixels": sdata["valid_pixels"],
                "coherent_coverage_pct": sdata["coherent_coverage_pct"],
                "weighted_mean_coherence": sdata["weighted_mean_coherence"],
                "median_p95_abs_vel_mm_yr": sdata["median_p95_abs_vel_mm_yr"],
                "max_abs_disp_mm": sdata["max_abs_disp_mm"],
                "maturity_gated_assets": sdata["maturity_gated_assets"],
            }
            for sk, sdata in sectors.items()
        ]
        with open(out_sectors_csv, "w", encoding="utf-8", newline="") as fh:
            w_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w_csv.writeheader()
            w_csv.writerows(rows)
        print(f"Wrote UK sector summary CSV -> {out_sectors_csv}")

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate UK Rail, Water AMP8, and Home Insurance analytics from an InSAR stack.")
    p.add_argument("--stack-dir", required=True, type=Path, help="Stack directory with velocity/coherence COGs and corridor_report.geojson")
    p.add_argument("--profile", choices=sorted(UK_SITE_PROFILES.keys()), default=None, help="UK site profile metadata")
    p.add_argument("--zones", type=Path, default=None, help="Optional GeoJSON of borough/district polygons for home insurance exposure")
    p.add_argument("--min-coh", type=float, default=0.35, help="Minimum temporal coherence (default: 0.35)")
    p.add_argument("--out-json", type=Path, default=None, help="Output JSON report path")
    p.add_argument("--out-zones-csv", type=Path, default=None, help="Output home insurance zones CSV path")
    p.add_argument("--out-sectors-csv", type=Path, default=None, help="Output sector rollup CSV path")
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_gdal_runtime(__file__, argv)
    args = build_arg_parser().parse_args(argv)
    report = analyze_uk_stack(
        args.stack_dir,
        profile=args.profile,
        zones_path=args.zones,
        min_coh=args.min_coh,
        out_json=args.out_json,
        out_zones_csv=args.out_zones_csv,
        out_sectors_csv=args.out_sectors_csv,
    )
    sc = report["scene_lband_metrics"]
    print(
        f"\nUK Stack Summary ({report['site_profile'].get('display_name', args.stack_dir.name)} | "
        f"stream={report.get('stream_key')} | mature={sc['stack_mature']}):"
    )
    print(
        f"  Coherent area: {sc['coherent_area_km2']} km^2 ({sc['coherent_fraction_pct']}%) | "
        f"mean_coh={sc['mean_coherence']} | high_coh(>=0.50)={sc['high_coh_050_pct']}%"
    )
    for sk, sdata in report["sectors"].items():
        print(
            f"  [{sk:19s}] covered={sdata['covered_assets']}/{sdata['total_assets']} | "
            f"mean_coh={sdata['weighted_mean_coherence']} | p95|v|={sdata['median_p95_abs_vel_mm_yr']} mm/yr"
        )
    if report["home_insurance_zones"]:
        print(f"  Home insurance zones evaluated: {len(report['home_insurance_zones'])}")
        for z in report["home_insurance_zones"][:5]:
            print(
                f"    - {z['name']:28s} coh={z['mean_coherence']:.2f} | "
                f"spread(p95-p5)={z['p95_p5_spread_mm']:.1f} mm | tier={z['exposure_tier']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
