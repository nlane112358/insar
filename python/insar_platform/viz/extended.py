#!/usr/bin/env python3
"""Render the extended corridor, confidence, transect, GNSS, and NISAR comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "insar_mpl_cache"))

from insar_platform.paths import REPO_ROOT as _ROOT, resolve_runtime_path
from insar_platform.viz._common import (
    asset_color,
    asset_label,
    build_wgs84_to_raster_ct,
    configure_matplotlib,
    load_raster,
    make_diverging_cmap,
    reproject_geojson_geometry,
    save_figure,
)


RISK_COLORS = {
    "critical": "#cf3341",
    "warning": "#ed8b27",
    "watch": "#d6ae31",
    "nominal": "#43866f",
    "no_coverage": "#8c9299",
    "baseline": "#697386",
}


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_validation_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No station rows in validation CSV: {path}")
    return rows


def plot_corridor_heatmap(timeseries_path: Path, out_path: Path, *, top_n: int = 40) -> Path:
    """Plot corridor p95 absolute velocity by acquisition date, with risk-coded labels."""
    import numpy as np
    from matplotlib.colors import PowerNorm

    data = json.loads(timeseries_path.read_text(encoding="utf-8"))
    epochs = data.get("epochs", [])
    if len(epochs) < 2:
        raise ValueError(f"Need at least two epochs in {timeseries_path}")

    records: list[dict[str, Any]] = []
    for corridor in data.get("corridors", {}).values():
        by_date = {p.get("date"): p for p in corridor.get("points", []) if p.get("date")}
        vals = [
            _finite_float(by_date.get(day, {}).get("p95_abs_vel_mm_yr"))
            for day in epochs
        ]
        valid_count = sum(v is not None for v in vals)
        if valid_count < 3:
            continue
        latest = by_date.get(epochs[-1], {})
        records.append(
            {
                "name": corridor.get("name") or corridor.get("asset_id", "Corridor"),
                "risk": latest.get("risk_level") or corridor.get("latest_risk_level") or "no_coverage",
                "values": vals,
                "latest": _finite_float(latest.get("p95_abs_vel_mm_yr")),
                "coverage": valid_count,
                "asset_class": corridor.get("asset_class", "infrastructure"),
            }
        )
    if not records:
        raise ValueError("No corridors have three or more measured epochs")
    records.sort(key=lambda row: (row["latest"] is not None, row["latest"] or -1.0), reverse=True)
    records = records[: max(1, top_n)]
    matrix = np.array(
        [[np.nan if v is None else v for v in row["values"]] for row in records],
        dtype=float,
    )
    finite = matrix[np.isfinite(matrix)]
    vmax = max(1.0, float(np.nanpercentile(finite, 97.0)))

    plt = configure_matplotlib()
    fig_h = max(7.5, 0.27 * len(records) + 2.2)
    fig, ax = plt.subplots(figsize=(15.5, fig_h))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=PowerNorm(gamma=0.42, vmin=0.0, vmax=vmax, clip=True),
    )
    ax.set_xticks(np.arange(len(epochs)))
    step = max(1, int(math.ceil(len(epochs) / 9)))
    ax.set_xticklabels([d if i % step == 0 or i == len(epochs) - 1 else "" for i, d in enumerate(epochs)], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([r["name"] for r in records], fontsize=8.1)
    for label, row in zip(ax.get_yticklabels(), records):
        label.set_color(RISK_COLORS.get(row["risk"], "#384252"))
    ax.set_xlabel("Acquisition date")
    ax.set_ylabel("Corridor (label color shows latest risk class)")
    fig.suptitle("Mexico City corridor motion through the Sentinel-1 stack", x=0.06, y=0.985, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.06,
        0.955,
        f"p95 absolute LOS velocity · mm/yr · {len(records)} corridors with at least three measured epochs",
        fontsize=9,
        color="#596579",
        va="top",
    )
    ax.set_xticks(np.arange(-0.5, len(epochs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(records), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("p95 |LOS velocity| (mm/yr)")
    fig.subplots_adjust(left=0.25, right=0.94, bottom=0.12, top=0.92)
    save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_velocity_confidence(stack_dir: Path, out_path: Path, *, min_coherence: float = 0.45) -> Path:
    """Show velocity alongside its SNR and standard error rasters."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    vel = load_raster(stack_dir / "velocity_mm_yr.tif")
    coh = load_raster(stack_dir / "temporal_coherence.tif")
    snr = load_raster(stack_dir / "velocity_snr.tif")
    unc = load_raster(stack_dir / "velocity_uncertainty_mm_yr.tif")
    valid = np.isfinite(vel.data) & np.isfinite(coh.data) & (coh.data >= min_coherence)
    if int(valid.sum()) == 0:
        raise ValueError(f"No valid velocity pixels at coherence >= {min_coherence:.2f}")
    significant = valid & np.isfinite(snr.data) & (snr.data >= 2.5)
    velocity_mask = significant if int(significant.sum()) >= 100 else valid
    vmin, vmax = (float(v) for v in np.nanpercentile(vel.data[velocity_mask], [2.5, 97.5]))
    vmin, vmax = min(vmin, -10.0), max(vmax, 10.0)
    ulim = max(5.0, float(np.nanpercentile(unc.data[valid & np.isfinite(unc.data)], 90.0)))
    snr_vmax = max(3.0, min(10.0, float(np.nanpercentile(snr.data[valid & np.isfinite(snr.data)], 97.0))))
    extent = vel.extent

    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.2), constrained_layout=True)
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    velocity = np.where(velocity_mask, vel.data, np.nan)
    div_cmap = make_diverging_cmap().copy()
    div_cmap.set_under(div_cmap(0.0))
    div_cmap.set_over(div_cmap(1.0))
    vel_norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im0 = axes[0].imshow(velocity, extent=extent, origin="upper", cmap=div_cmap, norm=vel_norm, interpolation="nearest")
    axes[0].set_title(f"LOS velocity\ncoherence ≥ {min_coherence:.2f} and SNR ≥ 2.5")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.78, pad=0.02, extend="both")
    cb0.set_label("LOS velocity (mm/yr)")

    snr_plot = np.where(valid & np.isfinite(snr.data), np.clip(snr.data, 0.0, snr_vmax), np.nan)
    im1 = axes[1].imshow(snr_plot, extent=extent, origin="upper", cmap="viridis", vmin=0.0, vmax=snr_vmax, interpolation="nearest")
    axes[1].set_title("Velocity signal-to-noise\noutlined at SNR = 2.5")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.78, pad=0.02)
    cb1.set_label("|velocity| / uncertainty")
    gt = vel.geotransform
    h, w = vel.data.shape
    xs = gt[0] + (np.arange(w) + 0.5) * gt[1]
    ys = gt[3] + (np.arange(h) + 0.5) * gt[5]
    if snr.data.shape == vel.data.shape:
        axes[1].contour(xs, ys, np.where(valid, snr.data, np.nan), levels=[2.5], colors=["white"], linewidths=0.65, alpha=0.9)

    unc_plot = np.where(valid & np.isfinite(unc.data), np.clip(unc.data, 0.0, ulim), np.nan)
    from matplotlib.colors import PowerNorm
    im2 = axes[2].imshow(unc_plot, extent=extent, origin="upper", cmap="inferno", norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=ulim, clip=True), interpolation="nearest")
    axes[2].set_title("Velocity standard error")
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.78, pad=0.02)
    cb2.set_label("Standard error (mm/yr)")
    cb2.set_ticks(np.linspace(0.0, ulim, 5))
    fig.suptitle(
        f"Mexico City velocity with confidence context  ·  coherence ≥ {min_coherence:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.018, f"Velocity clipped at p2.5–p97.5 ({vmin:.0f} to {vmax:.0f} mm/yr) · uncertainty uses a p90 stretch", ha="center", fontsize=8.5, color="#596579")
    save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _velocity_gradient_per_100m(velocity: Any, geotransform: tuple[float, float, float, float, float, float], valid: Any) -> Any:
    """Calculate a locally smoothed |∇v| in mm/yr per 100 m."""
    import numpy as np

    keep = np.asarray(valid, dtype=bool) & np.isfinite(velocity)
    values = np.where(keep, velocity, 0.0).astype(np.float32, copy=False)
    weights = keep.astype(np.float32)
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float32)
    radius = len(kernel) // 2
    h, w = values.shape
    values_pad = np.pad(values, ((0, 0), (radius, radius)), mode="constant")
    weights_pad = np.pad(weights, ((0, 0), (radius, radius)), mode="constant")
    horizontal_sum = np.zeros_like(values)
    horizontal_weight = np.zeros_like(weights)
    for idx, weight in enumerate(kernel):
        window = (slice(0, h), slice(idx, idx + w))
        horizontal_sum += weight * values_pad[window]
        horizontal_weight += weight * weights_pad[window]
    vertical_sum_pad = np.pad(horizontal_sum, ((radius, radius), (0, 0)), mode="constant")
    vertical_weight_pad = np.pad(horizontal_weight, ((radius, radius), (0, 0)), mode="constant")
    total = np.zeros_like(values)
    count = np.zeros_like(weights)
    for idx, weight in enumerate(kernel):
        window = (slice(idx, idx + h), slice(0, w))
        total += weight * vertical_sum_pad[window]
        count += weight * vertical_weight_pad[window]
    smooth = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=smooth, where=count > 0.0)
    field = np.where(keep, smooth, np.nan)
    if min(field.shape) < 2:
        raise ValueError("Velocity raster is too small to calculate a spatial gradient")
    d_col = np.gradient(field, axis=1)
    d_row = np.gradient(field, axis=0)
    gt = geotransform
    pixel_to_world = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=float)
    world_from_pixel = np.linalg.inv(pixel_to_world.T)
    d_dx = world_from_pixel[0, 0] * d_col + world_from_pixel[0, 1] * d_row
    d_dy = world_from_pixel[1, 0] * d_col + world_from_pixel[1, 1] * d_row
    return np.hypot(d_dx, d_dy) * 100.0


def plot_velocity_gradient(
    stack_dir: Path,
    corridors_path: Path,
    out_path: Path,
    *,
    min_coherence: float = 0.45,
) -> Path:
    """Render the engineering view: velocity and differential movement along corridors."""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D

    velocity = load_raster(stack_dir / "velocity_mm_yr.tif")
    coherence = load_raster(stack_dir / "temporal_coherence.tif")
    finite = np.isfinite(velocity.data)
    confident = finite & np.isfinite(coherence.data) & (coherence.data >= min_coherence)
    if int(confident.sum()) < 10:
        raise ValueError(f"Too few velocity pixels at coherence >= {min_coherence:.2f}")
    gradient = _velocity_gradient_per_100m(velocity.data, velocity.geotransform, confident)
    vals = velocity.data[confident]
    vmin, vmax = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
    vmin, vmax = min(vmin, -10.0), max(vmax, 10.0)
    grad_vals = gradient[np.isfinite(gradient)]
    if grad_vals.size == 0:
        raise ValueError("No valid velocity gradients could be calculated")
    grad_max = max(0.1, float(np.nanpercentile(grad_vals, 95.0)))
    grad_min = float(np.nanmin(grad_vals))
    grad_high = float(np.nanpercentile(grad_vals, 90.0))
    show_gradient_contour = grad_min < grad_high < grad_max

    if not corridors_path.is_file():
        raise FileNotFoundError(f"Corridor GeoJSON not found: {corridors_path}")
    payload = json.loads(corridors_path.read_text(encoding="utf-8"))
    transform = build_wgs84_to_raster_ct(velocity.projection_wkt)
    corridor_paths: list[tuple[str, str, list[tuple[float, float]]]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        cls = str(props.get("asset_class", "infrastructure"))
        for path in reproject_geojson_geometry(feature.get("geometry", {}), transform):
            if len(path) >= 2:
                corridor_paths.append((cls, asset_label(cls), path))

    div_cmap = make_diverging_cmap().copy()
    div_cmap.set_under(div_cmap(0.0))
    div_cmap.set_over(div_cmap(1.0))
    velocity_rgba = div_cmap(TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)(np.nan_to_num(velocity.data, nan=0.0)))
    coherence_ramp = np.clip((np.nan_to_num(coherence.data, nan=0.0) - 0.25) / 0.35, 0.0, 1.0)
    velocity_rgba[..., 3] = np.where(finite, 0.08 + 0.92 * coherence_ramp ** 1.4, 0.0)
    gradient_plot = np.where(confident, gradient, np.nan)
    gradient_cmap = colormaps["magma"].copy()
    gradient_cmap.set_over(gradient_cmap(1.0))

    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.1), constrained_layout=True)
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    axes[0].imshow(velocity_rgba, extent=velocity.extent, origin="upper", interpolation="nearest")
    image_gradient = axes[1].imshow(
        gradient_plot,
        extent=velocity.extent,
        origin="upper",
        cmap=gradient_cmap,
        norm=Normalize(vmin=0.0, vmax=grad_max, clip=False),
        interpolation="nearest",
    )
    axes[0].set_title("LOS velocity", loc="left")
    axes[1].set_title("Differential movement  |∇v|", loc="left")
    if show_gradient_contour:
        gt = velocity.geotransform
        h, w = velocity.data.shape
        xs = gt[0] + (np.arange(w) + 0.5) * gt[1]
        ys = gt[3] + (np.arange(h) + 0.5) * gt[5]
        axes[1].contour(
            xs,
            ys,
            np.where(confident, gradient, np.nan),
            levels=[grad_high],
            colors=["#17141b"],
            linewidths=0.55,
            alpha=0.8,
            zorder=3,
        )
    for ax in axes:
        for cls, _label, pts in corridor_paths:
            xs, ys = zip(*pts)
            color = asset_color(cls)
            ax.plot(xs, ys, color="white", linewidth=2.0, alpha=0.80, solid_capstyle="round", zorder=4)
            ax.plot(xs, ys, color=color, linewidth=1.05, alpha=0.96, solid_capstyle="round", zorder=5)
        ax.set_xlim(velocity.extent[0], velocity.extent[1])
        ax.set_ylim(velocity.extent[2], velocity.extent[3])
    velocity_mappable = plt.cm.ScalarMappable(
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), cmap=div_cmap
    )
    velocity_mappable.set_array([])
    cb_velocity = fig.colorbar(velocity_mappable, ax=axes[0], shrink=0.78, pad=0.02, extend="both")
    cb_velocity.set_label("LOS velocity (mm/yr)")
    cb_gradient = fig.colorbar(image_gradient, ax=axes[1], shrink=0.78, pad=0.02, extend="max")
    cb_gradient.set_label("mm/yr per 100 m")
    classes = sorted({cls for cls, _label, _pts in corridor_paths})
    handles = [Line2D([0], [0], color=asset_color(cls), lw=2.0, label=asset_label(cls)) for cls in classes[:6]]
    if show_gradient_contour:
        handles.append(Line2D([0], [0], color="#17141b", lw=1.0, linestyle="--", label="Gradient p90"))
    if handles:
        axes[1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)
    fig.suptitle(
        "Mexico City engineering view · velocity gradients identify differential ground movement",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        f"Gradient from the 210 m weighted velocity smooth · coherence ≥ {min_coherence:.2f} · color clipped at the 95th percentile",
        ha="center",
        fontsize=8.5,
        color="#596579",
    )
    save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def _pixel_indices(layer: Any, xs: Any, ys: Any) -> tuple[Any, Any]:
    import numpy as np

    gt = layer.geotransform
    matrix = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=float)
    inv = np.linalg.inv(matrix)
    coords = inv @ np.vstack([np.asarray(xs) - gt[0], np.asarray(ys) - gt[3]])
    cols = np.floor(coords[0]).astype(int)
    rows = np.floor(coords[1]).astype(int)
    return rows, cols


def _sample_layer_in_stack_crs(layer: Any, xs: Any, ys: Any, stack_wkt: str) -> Any:
    import numpy as np
    from osgeo import osr

    qx = np.asarray(xs, dtype=float)
    qy = np.asarray(ys, dtype=float)
    if layer.projection_wkt and stack_wkt and layer.projection_wkt != stack_wkt:
        src = osr.SpatialReference()
        src.ImportFromWkt(stack_wkt)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromWkt(layer.projection_wkt)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, dst)
        pts = [ct.TransformPoint(float(x), float(y), 0.0) for x, y in zip(qx, qy)]
        qx = np.array([p[0] for p in pts])
        qy = np.array([p[1] for p in pts])
    rows, cols = _pixel_indices(layer, qx, qy)
    out = np.full(rows.shape, np.nan, dtype=float)
    inside = (rows >= 0) & (cols >= 0) & (rows < layer.data.shape[0]) & (cols < layer.data.shape[1])
    out[inside] = layer.data[rows[inside], cols[inside]]
    return out


def _rolling_median(values: Any, window: int = 9) -> Any:
    import numpy as np

    half = max(1, int(window) // 2)
    out = np.full(np.asarray(values).shape, np.nan, dtype=float)
    for i in range(len(out)):
        local = np.asarray(values[max(0, i - half) : min(len(out), i + half + 1)], dtype=float)
        local = local[np.isfinite(local)]
        if local.size:
            out[i] = float(np.median(local))
    return out


def plot_basin_transect(stack_dir: Path, dem_path: Path, validation_csv: Path, out_path: Path, *, n_samples: int = 325) -> Path:
    """Sample LOS rates, cumulative motion, uncertainty, coherence, and elevation across the basin."""
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    vel = load_raster(stack_dir / "velocity_mm_yr.tif")
    cumulative = load_raster(stack_dir / "cumulative_disp_mm.tif")
    uncertainty = load_raster(stack_dir / "velocity_uncertainty_mm_yr.tif")
    coherence = load_raster(stack_dir / "temporal_coherence.tif")
    dem = load_raster(dem_path)
    summary = json.loads((stack_dir / "stack_summary.json").read_text(encoding="utf-8"))
    ref = summary.get("reference_region", {})
    if not ref.get("applied") or ref.get("lon") is None or ref.get("lat") is None:
        raise ValueError("Stack summary has no applied reference point for the transect start")
    stations = _read_validation_csv(validation_csv)
    airport = next((r for r in stations if r.get("station_id") == "MMX1"), None)
    if airport is None:
        raise ValueError(f"MMX1 endpoint was not found in {validation_csv}")

    transform = build_wgs84_to_raster_ct(vel.projection_wkt)
    x0, y0, _ = transform.TransformPoint(float(ref["lon"]), float(ref["lat"]), 0.0)
    x1, y1, _ = transform.TransformPoint(float(airport["lon"]), float(airport["lat"]), 0.0)
    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)
    distance_km = np.hypot(xs - x0, ys - y0) / 1000.0
    rate = _sample_layer_in_stack_crs(vel, xs, ys, vel.projection_wkt)
    cum = _sample_layer_in_stack_crs(cumulative, xs, ys, vel.projection_wkt)
    unc = _sample_layer_in_stack_crs(uncertainty, xs, ys, vel.projection_wkt)
    coh = _sample_layer_in_stack_crs(coherence, xs, ys, vel.projection_wkt)
    elev = _sample_layer_in_stack_crs(dem, xs, ys, vel.projection_wkt)
    keep = np.isfinite(coh) & (coh >= 0.35)
    rate = np.where(keep, rate, np.nan)
    cum = np.where(keep, cum, np.nan)
    unc = np.where(keep, unc, np.nan)
    rate = _rolling_median(rate)
    cum = _rolling_median(cum)
    unc = _rolling_median(unc)
    elev = _rolling_median(elev)
    coh = _rolling_median(coh)

    plt = configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1.1, 0.9, 0.8]})
    axes[0].plot(distance_km, rate, color="#a53543", linewidth=1.5, label="LOS velocity")
    axes[0].fill_between(distance_km, rate - unc, rate + unc, color="#d9959a", alpha=0.28, linewidth=0, label="±1σ velocity error")
    axes[0].axhline(0.0, color="#667085", linewidth=0.75, linestyle="--")
    axes[0].set_ylabel("Velocity (mm/yr)")
    axes[0].set_title("Reference bedrock to Mexico City Airport transect", loc="left", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(axis="y", color="#d8dde4", linewidth=0.6, linestyle=":")

    axes[1].plot(distance_km, cum, color="#315f85", linewidth=1.5)
    axes[1].axhline(0.0, color="#667085", linewidth=0.75, linestyle="--")
    axes[1].set_ylabel("Cumulative LOS\ndisplacement (mm)")
    axes[1].grid(axis="y", color="#d8dde4", linewidth=0.6, linestyle=":")

    axes[2].plot(distance_km, elev, color="#56634a", linewidth=1.3, label="Elevation")
    axes[2].set_ylabel("Elevation (m)")
    axes[2].set_xlabel("Distance from reference point (km)")
    axes[2].grid(axis="y", color="#d8dde4", linewidth=0.6, linestyle=":")
    ax_coh = axes[2].twinx()
    ax_coh.plot(distance_km, coh, color="#317b78", linewidth=1.0, alpha=0.85, label="Temporal coherence")
    ax_coh.set_ylabel("Temporal coherence")
    ax_coh.set_ylim(0.0, 1.0)
    handles_a, labels_a = axes[2].get_legend_handles_labels()
    handles_b, labels_b = ax_coh.get_legend_handles_labels()
    axes[2].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", fontsize=8.2)
    axes[2].xaxis.set_major_locator(MultipleLocator(2.0))
    axes[2].set_xlim(0.0, float(distance_km[-1]))
    fig.suptitle("Basin deformation profile  ·  500 m median-smoothed along-track samples", fontsize=14, fontweight="bold")
    save_figure(fig, out_path, dpi=250)
    plt.close(fig)
    return out_path


def plot_gnss_residual_diagnostics(stack_dir: Path, validation_csv: Path, out_path: Path, *, min_coherence: float = 0.25) -> Path:
    """Map InSAR-minus-GNSS residuals and relate them to coherence and spatial spread."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    rows = _read_validation_csv(validation_csv)
    usable = [
        row
        for row in rows
        if all(_finite_float(row.get(key)) is not None for key in ("lat", "lon", "residual_mm_yr", "insar_coherence", "insar_spatial_std_mm_yr"))
    ]
    if not usable:
        raise ValueError(f"No complete GNSS residual records in {validation_csv}")
    coh = load_raster(stack_dir / "temporal_coherence.tif")
    valid = np.isfinite(coh.data) & (coh.data >= min_coherence)
    residuals = np.array([float(r["residual_mm_yr"]) for r in usable])
    rlim = max(10.0, math.ceil(float(np.max(np.abs(residuals))) / 10.0) * 10.0)
    ct = build_wgs84_to_raster_ct(coh.projection_wkt)
    pts = [ct.TransformPoint(float(r["lon"]), float(r["lat"]), 0.0) for r in usable]
    norm = TwoSlopeNorm(vmin=-rlim, vcenter=0.0, vmax=rlim)
    residual_cmap = make_diverging_cmap()

    plt = configure_matplotlib()
    fig = plt.figure(figsize=(14.0, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_coh = fig.add_subplot(gs[0, 1])
    ax_spread = fig.add_subplot(gs[1, 1])
    ax_map.imshow(np.where(valid, coh.data, np.nan), extent=coh.extent, origin="upper", cmap="Greys", vmin=min_coherence, vmax=0.9, interpolation="nearest")
    ax_map.set_title("GNSS stations and velocity residuals", loc="left", fontsize=11.5, fontweight="bold")
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    map_offsets = {
        "MXAZ": (7, 7), "UNVA": (7, 8), "ICMX": (-24, 8),
        "UTAC": (-30, -15), "MXMX": (8, 8), "MMX1": (8, -14),
    }
    for row, (x, y, _z) in zip(usable, pts):
        res = float(row["residual_mm_yr"])
        ax_map.scatter([x], [y], s=92, c=[res], cmap=residual_cmap, norm=norm, edgecolors="#17212e", linewidths=0.85, zorder=5)
        offset = map_offsets.get(row["station_id"], (6, 6))
        ax_map.annotate(row["station_id"], (x, y), xytext=offset, textcoords="offset points", fontsize=8, color="#182230", zorder=6, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 1.1})
    ax_map.set_aspect("equal", adjustable="box")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=residual_cmap)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=ax_map, pad=0.025, fraction=0.04, extend="both")
    cb1.set_label("InSAR - GNSS residual (mm/yr)")

    coherence_values = np.array([float(r["insar_coherence"]) for r in usable])
    spread_values = np.array([float(r["insar_spatial_std_mm_yr"]) for r in usable])
    names = [r["station_id"] for r in usable]
    for ax, xvals, xlabel in (
        (ax_coh, coherence_values, "Station InSAR coherence"),
        (ax_spread, spread_values, "InSAR within-station spatial spread (mm/yr)"),
    ):
        ax.axhline(0.0, color="#667085", linewidth=0.8, linestyle="--")
        ax.scatter(xvals, residuals, c=residuals, cmap=residual_cmap, norm=norm, s=68, edgecolors="#17212e", linewidths=0.55, zorder=3)
        diag_offsets = {
            "MXAZ": (6, 6), "ICMX": (6, 7), "MMX1": (6, -13),
            "UNVA": (6, 7), "UTAC": (-34, -5), "MXMX": (6, -24),
        }
        for xx, yy, label in zip(xvals, residuals, names):
            ax.annotate(label, (xx, yy), xytext=diag_offsets.get(label, (5, 4)), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("InSAR − GNSS (mm/yr)")
        ax.grid(color="#d8dde4", linewidth=0.6, linestyle=":")
    ax_coh.set_xlim(0.0, 1.0)
    ax_coh.axvline(min_coherence, color="#8993a4", linewidth=0.8, linestyle=":")
    fig.suptitle("Mexico City GNSS residual diagnostics", fontsize=14, fontweight="bold")
    save_figure(fig, out_path, dpi=250)
    plt.close(fig)
    return out_path


def _pair_date_label(pair_dir: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$", pair_dir.name)
    return f"{match.group(1)} to {match.group(2)}" if match else pair_dir.name


def plot_nisar_track_comparison(track_a_dir: Path, track_b_dir: Path, out_path: Path) -> Path:
    """Compare pair displacement and coherence for two overlapping NISAR tracks."""
    import numpy as np
    from matplotlib.colors import Normalize, TwoSlopeNorm

    for pair_dir in (track_a_dir, track_b_dir):
        if not (pair_dir / "displacement_mm.tif").is_file() or not (pair_dir / "coherence.tif").is_file():
            raise FileNotFoundError(f"Pair displacement/coherence rasters are missing in {pair_dir}")
    a_disp = load_raster(track_a_dir / "displacement_mm.tif")
    a_coh = load_raster(track_a_dir / "coherence.tif")
    b_disp = load_raster(track_b_dir / "displacement_mm.tif")
    b_coh = load_raster(track_b_dir / "coherence.tif")
    vals = np.concatenate(
        [
            a_disp.data[np.isfinite(a_disp.data) & (a_coh.data >= 0.3)],
            b_disp.data[np.isfinite(b_disp.data) & (b_coh.data >= 0.3)],
        ]
    )
    if vals.size:
        dmin, dmax = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
        dmin, dmax = min(dmin, -5.0), max(dmax, 5.0)
    else:
        dmin, dmax = -10.0, 10.0
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 15.2), constrained_layout=True)
    images = {}
    cases = [
        (axes[0, 0], a_disp, a_coh, "Track T052 · pair displacement", _pair_date_label(track_a_dir), True),
        (axes[1, 0], b_disp, b_coh, "Track T029 · pair displacement", _pair_date_label(track_b_dir), True),
        (axes[0, 1], a_coh, a_coh, "Track T052 · spatial coherence", _pair_date_label(track_a_dir), False),
        (axes[1, 1], b_coh, b_coh, "Track T029 · spatial coherence", _pair_date_label(track_b_dir), False),
    ]
    for ax, layer, coherence_layer, title, dates, diverging in cases:
        arr = layer.data
        if diverging:
            arr = np.where(np.isfinite(arr) & (coherence_layer.data >= 0.3), arr, np.nan)
            dnorm = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)
            im = ax.imshow(arr, extent=layer.extent, origin="upper", cmap=make_diverging_cmap(), norm=dnorm, interpolation="nearest")
            images["displacement"] = im
        else:
            im = ax.imshow(arr, extent=layer.extent, origin="upper", cmap="viridis", norm=Normalize(0.0, 1.0), interpolation="nearest")
            images["coherence"] = im
        ax.set_title(f"{title}\n{dates}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal", adjustable="box")
    cb_disp = fig.colorbar(images["displacement"], ax=axes[:, 0], shrink=0.84, pad=0.035, extend="both")
    cb_disp.set_label("LOS displacement (mm)")
    cb_coh = fig.colorbar(images["coherence"], ax=axes[:, 1], shrink=0.84, pad=0.035)
    cb_coh.set_label("Coherence")
    fig.suptitle(
        "West Yorkshire NISAR pair comparison\n"
        "LOS displacement is shown in each track geometry; compare spatial patterns and coherence",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT, help="Project root")
    parser.add_argument("--stack-dir", type=Path, default=None, help="Mexico City stack with final COGs and corridor history")
    parser.add_argument("--validation-csv", type=Path, default=None, help="GNSS station validation CSV")
    parser.add_argument("--corridors", type=Path, default=None, help="Source infrastructure GeoJSON in WGS84")
    parser.add_argument("--dem", type=Path, default=None, help="DEM aligned to or transformable into the stack grid")
    parser.add_argument("--track-a-pair-dir", type=Path, default=None, help="NISAR T052 pair product directory")
    parser.add_argument("--track-b-pair-dir", type=Path, default=None, help="NISAR T029 pair product directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for generated PNG figures")
    parser.add_argument("--top-corridors", type=int, default=40, help="Maximum corridors in the heatmap")
    return parser


def main(argv: list[str] | None = None) -> int:
    from insar_platform.viz._common import ensure_plot_runtime

    ensure_plot_runtime(__file__, argv)
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    stack_dir = resolve_runtime_path(args.stack_dir or root / "var/work/stacks/mexico_city_iw1", root).resolve()
    validation_csv = (args.validation_csv or root / "data/validation/mexico_city_gnss_validation.csv").resolve()
    corridors = (args.corridors or root / "data/corridors/mexico_city_corridors.geojson").resolve()
    dem = resolve_runtime_path(args.dem or root / "var/work/dem/mexico_city_iw1_cop30.tif", root).resolve()
    track_a = resolve_runtime_path(args.track_a_pair_dir or root / "var/work/products/NISAR_T052_F061_D_2026-10-27_2026-11-08", root).resolve()
    track_b = resolve_runtime_path(args.track_b_pair_dir or root / "var/work/products/west_yorkshire_nisar_t029/NISAR_T029_F028_A_2026-10-26_2026-11-07", root).resolve()
    out_dir = (args.out_dir or root / "docs/figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = [
        ("corridor heatmap", lambda: plot_corridor_heatmap(stack_dir / "corridor_timeseries.json", out_dir / "mexico_city_corridor_heatmap.png", top_n=args.top_corridors)),
        ("velocity confidence", lambda: plot_velocity_confidence(stack_dir, out_dir / "mexico_city_velocity_confidence.png")),
        ("velocity gradient engineering view", lambda: plot_velocity_gradient(stack_dir, corridors, out_dir / "mexico_city_velocity_gradient.png")),
        ("basin transect", lambda: plot_basin_transect(stack_dir, dem, validation_csv, out_dir / "mexico_city_basin_transect.png")),
        ("GNSS residual diagnostics", lambda: plot_gnss_residual_diagnostics(stack_dir, validation_csv, out_dir / "mexico_city_gnss_residuals.png")),
        ("NISAR track comparison", lambda: plot_nisar_track_comparison(track_a, track_b, out_dir / "west_yorkshire_nisar_track_comparison.png")),
    ]
    for label, render in requests:
        print(f"Rendering {label}...")
        print(f"  {render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
