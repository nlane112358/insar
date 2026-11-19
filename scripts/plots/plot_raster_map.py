#!/usr/bin/env python3
"""Render standalone cartographic maps for derived InSAR stack GeoTIFF layers.

Velocity maps use DEM hillshade, confidence-weighted opacity, asymmetric scientific
color clipping, crisp infrastructure corridor overlays, and a km scalebar.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

# Allow standalone execution (`python scripts/plots/plot_raster_map.py`)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import (  # noqa: E402
    INK,
    LIGHT_MUTED,
    MUTED,
    NODATA_BG,
    add_editorial_header,
    add_footer,
    add_scalebar,
    asset_color,
    asset_label,
    build_wgs84_to_raster_ct,
    compute_valid_extent,
    configure_matplotlib,
    ensure_plot_runtime,
    format_date_range,
    format_site_label,
    load_corridor_alerts,
    load_corridor_report,
    load_raster,
    load_stack_ledger,
    load_stack_summary,
    make_coherence_cmap,
    make_diverging_cmap,
    make_uncertainty_cmap,
    reproject_geojson_geometry,
    save_figure,
    shorten_name,
)


LAYER_SPECS: dict[str, dict[str, Any]] = {
    "velocity": {
        "file": "velocity_mm_yr.tif",
        "title": "Line-of-Sight Surface Velocity",
        "unit": "mm / year",
        "diverging": True,
        "percentile": 97.5,
        "min_abs": 15.0,
        "coh_fade": True,
    },
    "cumulative": {
        "file": "cumulative_disp_mm.tif",
        "title": "Cumulative Line-of-Sight Displacement",
        "unit": "mm",
        "diverging": True,
        "percentile": 97.5,
        "min_abs": 5.0,
        "coh_fade": True,
    },
    "coherence": {
        "file": "temporal_coherence.tif",
        "title": "Mean Temporal Coherence",
        "unit": "Coherence (γ)",
        "diverging": False,
        "vmin": 0.0,
        "vmax": 1.0,
        "coh_fade": False,
    },
    "uncertainty": {
        "file": "velocity_uncertainty_mm_yr.tif",
        "title": "Velocity Standard Error (Uncertainty)",
        "unit": "mm / year",
        "diverging": False,
        "percentile": 96.0,
        "coh_fade": False,
    },
    "acceleration": {
        "file": "acceleration_mm_yr2.tif",
        "title": "Line-of-Sight Surface Acceleration",
        "unit": "mm / year²",
        "diverging": True,
        "percentile": 95.0,
        "min_abs": 50.0,
        "coh_fade": True,
    },
}


def _compute_urban_focus_extent(
    coh_arr: Any,
    gt: tuple[float, float, float, float, float, float],
    coh_threshold: float = 0.62,
) -> tuple[float, float, float, float]:
    """Focus on the dense high-coherence core using 5th-99th percentile spatial bounds."""
    import numpy as np

    mask = np.isfinite(coh_arr) & (coh_arr >= coh_threshold)
    ys, xs = np.where(mask)
    if ys.size < 500:
        return compute_valid_extent(coh_arr, gt)
    # Trim isolated scattered coherent outliers outside main urban core
    x_lo, x_hi = np.percentile(xs, [8.0, 99.5])
    y_lo, y_hi = np.percentile(ys, [1.0, 99.0])
    ux0 = gt[0] + float(x_lo) * gt[1]
    ux1 = gt[0] + float(x_hi) * gt[1]
    uy0 = gt[3] + float(y_lo) * gt[5]
    uy1 = gt[3] + float(y_hi) * gt[5]
    xmin, xmax = min(ux0, ux1), max(ux0, ux1)
    ymin, ymax = min(uy0, uy1), max(uy0, uy1)
    pad_x = 0.04 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def _recorded_dem_path(stack_dir: Path, ledger: dict[str, Any]) -> Path | None:
    """Resolve the DEM path recorded by stack_timeseries, stripping its fingerprint."""
    raw = (ledger.get("stack_params", {}) or {}).get("dem")
    if not isinstance(raw, str) or not raw:
        return None
    path_part, separator, fingerprint = raw.rpartition(":")
    if separator and len(fingerprint) >= 8 and all(ch in "0123456789abcdefABCDEF" for ch in fingerprint):
        raw = path_part
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    # Stack ledgers may have been copied from another checkout. Use the conventional
    # site DEM beside insar_work/stacks when the recorded absolute path has gone stale.
    for parent in (stack_dir.parent.parent, stack_dir.parent.parent.parent):
        fallback = parent / "dem" / f"{stack_dir.name}_cop30.tif"
        if fallback.is_file():
            return fallback
    return None


def _load_dem_on_grid(dem_path: Path, target: Any) -> Any:
    """Read a DEM aligned to the target raster, reprojecting in memory if needed."""
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()
    src = gdal.Open(str(dem_path))
    if src is None:
        raise FileNotFoundError(f"Could not open DEM: {dem_path}")
    src_band = src.GetRasterBand(1)
    src_nodata = src_band.GetNoDataValue()
    target_gt = tuple(target.geotransform)
    source_gt = tuple(float(v) for v in src.GetGeoTransform())
    source_wkt = src.GetProjection() or ""
    h, w = target.data.shape
    same_crs = source_wkt == target.projection_wkt or (not source_wkt and not target.projection_wkt)
    same_grid = (
        src.RasterXSize == w
        and src.RasterYSize == h
        and np.allclose(source_gt, target_gt, rtol=0.0, atol=1e-8)
        and same_crs
    )
    if same_grid:
        data = src_band.ReadAsArray().astype(np.float64)
    else:
        if not source_wkt or not target.projection_wkt:
            src = None
            raise ValueError(f"Cannot align DEM without source and target CRS: {dem_path}")
        driver = gdal.GetDriverByName("MEM")
        dst = driver.Create("", w, h, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(target_gt)
        dst.SetProjection(target.projection_wkt)
        dst_band = dst.GetRasterBand(1)
        dst_band.SetNoDataValue(-9999.0)
        dst_band.Fill(-9999.0)
        gdal.ReprojectImage(src, dst, source_wkt, target.projection_wkt, gdal.GRA_Bilinear)
        data = dst_band.ReadAsArray().astype(np.float64)
        data[data == -9999.0] = np.nan
        dst = None
    if src_nodata is not None and np.isfinite(src_nodata):
        data[data == src_nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    src = None
    return data


def _multidirectional_hillshade(dem: Any, geotransform: tuple[float, float, float, float, float, float]) -> Any:
    """Build a softly normalized four-direction hillshade from a projected DEM."""
    import numpy as np
    from matplotlib.colors import LightSource

    valid = np.isfinite(dem)
    if not valid.any():
        return None
    filled = np.where(valid, dem, float(np.nanmedian(dem[valid])))
    dx = max(1.0, math.hypot(geotransform[1], geotransform[4]))
    dy = max(1.0, math.hypot(geotransform[2], geotransform[5]))
    shades = [
        LightSource(azdeg=azimuth, altdeg=42).hillshade(filled, vert_exag=1.0, dx=dx, dy=dy)
        for azimuth in (0, 90, 180, 270)
    ]
    shade = np.mean(shades, axis=0)
    lo, hi = np.nanpercentile(shade[valid], [3.0, 97.0])
    if hi <= lo:
        shade[:] = 0.5
    else:
        shade = np.clip((shade - lo) / (hi - lo), 0.0, 1.0)
    shade[~valid] = np.nan
    return shade


def _shade_rgba_luminance(rgba: Any, hillshade: Any, strength: float) -> Any:
    """Modulate velocity color luminance while preserving its diverging hue."""
    import numpy as np
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

    shaded = rgba.copy()
    hsv = rgb_to_hsv(shaded[..., :3])
    strength = float(np.clip(strength, 0.0, 1.0))
    modulation = 1.0 + strength * (2.0 * np.nan_to_num(hillshade, nan=0.5) - 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * modulation, 0.0, 1.0)
    shaded[..., :3] = hsv_to_rgb(hsv)
    return shaded


def _clip_rgba_with_coherence(
    data: Any,
    coh: Any | None,
    uncertainty: Any | None,
    cmap: Any,
    norm: Any,
    fade_low: float,
    fade_high: float,
    use_fade: bool,
    uncertainty_scale: float | None,
) -> Any:
    import numpy as np

    finite = np.isfinite(data)
    safe = np.where(finite, data, 0.0)
    rgba = cmap(norm(safe))
    rgba[~finite, 3] = 0.0

    if use_fade and coh is not None:
        c = np.where(np.isfinite(coh), coh, 0.0)
        ramp = np.clip((c - fade_low) / max(fade_high - fade_low, 1e-4), 0.0, 1.0)
        alpha = 0.08 + 0.92 * ramp * ramp * (3.0 - 2.0 * ramp)
        rgba[..., 3] *= np.where(finite, alpha, 0.0)

    if uncertainty is not None and uncertainty_scale is not None and uncertainty_scale > 0.0:
        unc_valid = np.isfinite(uncertainty) & (uncertainty >= 0.0)
        scaled = np.where(unc_valid, uncertainty / uncertainty_scale, 0.0)
        confidence = 1.0 / (1.0 + scaled * scaled)
        rgba[..., 3] *= np.where(finite, confidence, 0.0)
    return rgba


def _clip_path_inside_extent(
    pts: list[tuple[float, float]],
    extent: tuple[float, float, float, float],
) -> bool:
    xmin, xmax, ymin, ymax = extent
    return any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in pts)


def _anchor_inside_extent(
    paths: list[list[tuple[float, float]]],
    extent: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    xmin, xmax, ymin, ymax = extent
    mx0 = xmin + 0.06 * (xmax - xmin)
    mx1 = xmax - 0.06 * (xmax - xmin)
    my0 = ymin + 0.08 * (ymax - ymin)
    my1 = ymax - 0.08 * (ymax - ymin)
    in_pts = [p for path in paths for p in path if mx0 <= p[0] <= mx1 and my0 <= p[1] <= my1]
    if not in_pts:
        return None
    return in_pts[len(in_pts) // 2]


def render_raster_map(
    stack_dir: Path,
    out_path: Path,
    *,
    layer: str = "velocity",
    show_corridors: bool = True,
    show_alerts: bool = False,
    coh_fade: bool = True,
    fade_low: float = 0.30,
    fade_high: float = 0.58,
    focus_urban: bool = False,
    focus_coh: float = 0.62,
    label_top: int = 5,
    min_label_coh: float = 0.60,
    vlim: float | None = None,
    dem_path: Path | None = None,
    show_hillshade: bool = True,
    hillshade_strength: float = 0.13,
    dpi: int = 220,
) -> Path:
    import numpy as np
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt = configure_matplotlib()
    spec = LAYER_SPECS[layer]
    raster_path = stack_dir / spec["file"]
    if not raster_path.is_file():
        raise FileNotFoundError(f"Layer raster not found: {raster_path}")

    rlayer = load_raster(raster_path)
    coh_path = stack_dir / "temporal_coherence.tif"
    coh_layer = load_raster(coh_path) if coh_path.is_file() else None
    summary = load_stack_summary(stack_dir)
    ledger = load_stack_ledger(stack_dir)

    resolved_dem = dem_path or _recorded_dem_path(stack_dir, ledger)
    hillshade = None
    if layer == "velocity" and show_hillshade and hillshade_strength > 0.0 and resolved_dem is not None:
        if not resolved_dem.is_file():
            if dem_path is not None:
                raise FileNotFoundError(f"DEM not found: {resolved_dem}")
        else:
            dem_data = _load_dem_on_grid(resolved_dem, rlayer)
            hillshade = _multidirectional_hillshade(dem_data, rlayer.geotransform)
            if hillshade is not None:
                print(f"Using DEM hillshade -> {resolved_dem}")

    if focus_urban and coh_layer is not None:
        view_extent = _compute_urban_focus_extent(coh_layer.data, rlayer.geotransform, coh_threshold=focus_coh)
    else:
        view_extent = compute_valid_extent(rlayer.data, rlayer.geotransform, pad_frac=0.03)

    xmin, xmax, ymin, ymax = view_extent
    # Determine color scale from visible finite pixels (preferring coherent pixels for clean contrast)
    gt = rlayer.geotransform
    h, w = rlayer.data.shape
    ix0 = max(0, min(w - 1, int(round((xmin - gt[0]) / gt[1]))))
    ix1 = max(0, min(w, int(round((xmax - gt[0]) / gt[1]))))
    iy_a = int(round((ymax - gt[3]) / gt[5]))
    iy_b = int(round((ymin - gt[3]) / gt[5]))
    iy0, iy1 = max(0, min(iy_a, iy_b)), min(h, max(iy_a, iy_b))
    sub = rlayer.data[iy0:iy1, ix0:ix1]
    sub_coh = coh_layer.data[iy0:iy1, ix0:ix1] if coh_layer is not None else None
    uncertainty_path = stack_dir / "velocity_uncertainty_mm_yr.tif"
    uncertainty_layer = load_raster(uncertainty_path) if layer == "velocity" and uncertainty_path.is_file() else None
    sub_uncertainty = uncertainty_layer.data[iy0:iy1, ix0:ix1] if uncertainty_layer is not None else None

    if spec["diverging"]:
        cmap = make_diverging_cmap().copy()
        cmap.set_under(cmap(0.0))
        cmap.set_over(cmap(1.0))
        if vlim is not None:
            lim = abs(float(vlim))
            if lim <= 0.0:
                raise ValueError("--vlim must be greater than zero")
            vmin, vmax = -lim, lim
        else:
            mask = np.isfinite(sub)
            if sub_coh is not None and (mask & (sub_coh >= 0.50)).sum() > 200:
                mask = mask & (sub_coh >= 0.50)
            vals = sub[mask]
            if vals.size:
                vmin, vmax = (float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
            else:
                vmin, vmax = -spec["min_abs"], spec["min_abs"]
            vmin = min(vmin, -float(spec["min_abs"]))
            vmax = max(vmax, float(spec["min_abs"]))
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    elif layer == "coherence":
        cmap = make_coherence_cmap()
        vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    else:
        cmap = make_uncertainty_cmap()
        vals = sub[np.isfinite(sub)]
        if vals.size == 0:
            raise RuntimeError(f"No finite values in {raster_path} (uncertainty requires >= 2 stacked pairs).")
        vmax = float(vlim) if vlim is not None else max(5.0, float(np.nanpercentile(vals, spec["percentile"])))
        vmin = 0.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

    uncertainty_scale = None
    if sub_uncertainty is not None:
        uncertainty_valid = np.isfinite(sub_uncertainty) & (sub_uncertainty >= 0.0)
        if sub_coh is not None:
            coherent_uncertainty = uncertainty_valid & np.isfinite(sub_coh) & (sub_coh >= fade_low)
            if coherent_uncertainty.any():
                uncertainty_valid = coherent_uncertainty
        if uncertainty_valid.any():
            uncertainty_scale = max(1.0, float(np.nanpercentile(sub_uncertainty[uncertainty_valid], 75.0)))

    # Dynamic balanced figure dimensions based on physical map aspect
    aspect = (xmax - xmin) / max(ymax - ymin, 1.0)
    clamped = max(0.68, min(2.25, aspect))
    if clamped >= 1.2:
        map_w = 10.6
        map_h = map_w / clamped
    else:
        map_h = 7.4
        map_w = map_h * clamped

    top_pad_in = 1.05
    bot_pad_in = 1.85
    fig_w = max(7.8, map_w + 1.5)
    fig_h = map_h + top_pad_in + bot_pad_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    left_m = 0.095 if fig_w < 8.5 else 0.075
    right_m = 0.965
    top_m = 1.0 - (top_pad_in / fig_h)
    bot_m = bot_pad_in / fig_h
    ax = fig.add_axes([left_m, bot_m, right_m - left_m, top_m - bot_m])
    ax.set_facecolor(NODATA_BG)

    if hillshade is not None:
        ax.imshow(
            np.ma.masked_invalid(hillshade),
            extent=rlayer.extent,
            origin="upper",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            alpha=0.25 * min(1.0, max(0.0, hillshade_strength) / 0.13),
            zorder=1,
        )

    rgba = _clip_rgba_with_coherence(
        rlayer.data,
        coh_layer.data if coh_layer is not None else None,
        uncertainty_layer.data if uncertainty_layer is not None else None,
        cmap,
        norm,
        fade_low=fade_low,
        fade_high=fade_high,
        use_fade=(coh_fade and spec["coh_fade"]),
        uncertainty_scale=uncertainty_scale,
    )
    if hillshade is not None:
        rgba = _shade_rgba_luminance(rgba, hillshade, hillshade_strength)
    ax.imshow(rgba, extent=rlayer.extent, origin="upper", interpolation="nearest", zorder=2)

    # Overlay infrastructure corridors cleanly
    drawn_classes: set[str] = set()
    labeled_anchors: list[tuple[float, float, str, str]] = []
    if show_corridors and rlayer.projection_wkt:
        corridors = load_corridor_report(stack_dir)
        if corridors:
            ct = build_wgs84_to_raster_ct(rlayer.projection_wkt)
            ranked_candidates: list[tuple[float, dict[str, Any], list[list[tuple[float, float]]]]] = []
            for feat in corridors:
                props = feat.get("properties", {})
                if (props.get("valid_pixels") or 0) <= 0:
                    continue
                paths = reproject_geojson_geometry(feat.get("geometry", {}), ct)
                vis_paths = [p for p in paths if len(p) >= 2 and _clip_path_inside_extent(p, view_extent)]
                if not vis_paths:
                    continue
                cls = str(props.get("asset_class", "infrastructure"))
                col = asset_color(cls)
                drawn_classes.add(cls)
                is_priority = (props.get("mean_coherence") or 0.0) >= min_label_coh and (props.get("valid_pixels") or 0) >= 80
                lw = 1.55 if is_priority else 0.95
                for pts in vis_paths:
                    xs, ys = zip(*pts)
                    # Crisp subtle white halo + rich centerline
                    ax.plot(xs, ys, color="#FFFFFF", linewidth=lw + 1.5, alpha=0.72, solid_capstyle="round", zorder=5)
                    ax.plot(xs, ys, color=col, linewidth=lw, alpha=0.92, solid_capstyle="round", zorder=6)
                if is_priority:
                    score = abs(float(props.get("p95_abs_velocity_mm_yr") or 0.0))
                    ranked_candidates.append((score, props, vis_paths))

            # Place non-overlapping direct labels for top coherent corridors
            ranked_candidates.sort(key=lambda t: -t[0])
            min_dist_x = 0.16 * (xmax - xmin)
            min_dist_y = 0.085 * (ymax - ymin)
            for _, props, vis_paths in ranked_candidates:
                if len(labeled_anchors) >= label_top:
                    break
                anchor = _anchor_inside_extent(vis_paths, view_extent)
                if anchor is None:
                    continue
                ax_x, ax_y = anchor
                # Avoid upper-left legend zone and upper-right north arrow
                in_legend = drawn_classes and ax_x < xmin + 0.48 * (xmax - xmin) and ax_y > ymax - 0.16 * (ymax - ymin)
                in_north = ax_x > xmax - 0.12 * (xmax - xmin) and ax_y > ymax - 0.18 * (ymax - ymin)
                if in_legend or in_north:
                    continue
                if any(abs(ax_x - lx) < min_dist_x and abs(ax_y - ly) < min_dist_y for lx, ly, _, _ in labeled_anchors):
                    continue
                cls = str(props.get("asset_class", "infrastructure"))
                mv = props.get("mean_velocity_mm_yr")
                tag = f"{shorten_name(str(props.get('name', '')), 24)}"
                if mv is not None and layer in {"velocity", "cumulative"}:
                    tag += f"\n{mv:+.0f} mm/yr"
                labeled_anchors.append((ax_x, ax_y, tag, asset_color(cls)))

            for ax_x, ax_y, tag, col in labeled_anchors:
                ax.scatter([ax_x], [ax_y], s=22, color=col, edgecolor="#FFFFFF", linewidth=1.1, zorder=11)
                near_right = ax_x > xmin + 0.68 * (xmax - xmin)
                near_top = ax_y > ymax - 0.15 * (ymax - ymin)
                dx = (-0.012 if near_right else 0.012) * (xmax - xmin)
                dy = (-0.012 if near_top else 0.012) * (ymax - ymin)
                ax.text(
                    ax_x + dx,
                    ax_y + dy,
                    tag,
                    fontsize=7.6,
                    fontweight="bold",
                    color=INK,
                    ha="right" if near_right else "left",
                    va="top" if near_top else "bottom",
                    zorder=12,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground="#FFFFFF", alpha=0.92)],
                )

    if show_alerts and rlayer.projection_wkt and layer == "velocity":
        alerts = load_corridor_alerts(stack_dir)
        if alerts:
            ct = build_wgs84_to_raster_ct(rlayer.projection_wkt)
            ax_pts, ay_pts = [], []
            for feat in alerts:
                props = feat.get("properties", {})
                if (props.get("temporal_coherence") or 0.0) < min_label_coh:
                    continue
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    ux, uy, _ = ct.TransformPoint(float(coords[0]), float(coords[1]), 0.0)
                    if xmin <= ux <= xmax and ymin <= uy <= ymax:
                        ax_pts.append(ux)
                        ay_pts.append(uy)
            if ax_pts:
                ax.scatter(
                    ax_pts[:80],
                    ay_pts[:80],
                    s=26,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.1,
                    alpha=0.75,
                    zorder=9,
                )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # Crisp UTM kilometer formatting (eliminates awkward 1e6 corner offsets)
    km_fmt = FuncFormatter(lambda v, _: f"{v / 1000.0:,.0f}")
    ax.xaxis.set_major_formatter(km_fmt)
    ax.yaxis.set_major_formatter(km_fmt)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("UTM Easting (km)", labelpad=5)
    ax.set_ylabel("UTM Northing (km)", labelpad=6)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#B8C2CC")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.5, width=0.75)

    # Cartographic scalebar & minimal north indicator
    add_scalebar(ax, xmin, xmax, ymin, ymax)
    ax.annotate(
        "N",
        xy=(0.965, 0.945),
        xytext=(0.965, 0.875),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2, "mutation_scale": 10},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.82},
        zorder=15,
    )

    # Compact corridor class legend inside top-left if corridors drawn
    if drawn_classes:
        priority = ["rail", "metro", "light_rail", "water_works", "reservoir", "dam", "canal", "river", "motorway", "trunk", "airport_runway", "infrastructure"]
        ordered_cls = [c for c in priority if c in drawn_classes] + [c for c in sorted(drawn_classes) if c not in priority]
        handles = [
            Line2D([0], [0], color=asset_color(c), lw=2.0, label=asset_label(c))
            for c in ordered_cls[:9]
        ]
        leg = ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            facecolor="#FFFFFF",
            framealpha=0.86,
            edgecolor="#D5DCE4",
            fontsize=8.0,
            ncol=min(3, len(handles)),
            borderpad=0.45,
            handlelength=1.4,
            columnspacing=1.0,
        )
        leg.set_zorder(16)

    # Horizontal editorial colorbar anchored cleanly beneath the map
    cax_y = 0.86 / fig_h
    cax_h = 0.17 / fig_h
    cax = fig.add_axes([left_m + 0.08, cax_y, (right_m - left_m) - 0.16, cax_h])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb_extend = "both" if spec["diverging"] else ("max" if layer == "uncertainty" and np.any(rlayer.data > vmax) else "neither")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=cb_extend)
    cb.outline.set_edgecolor("#B8C2CC")
    cb.outline.set_linewidth(0.75)
    cb.ax.tick_params(labelsize=8.5, length=3, color=MUTED, labelcolor=MUTED)
    cb.set_label(spec["unit"], fontsize=9.2, color=INK, labelpad=3)
    if spec["diverging"]:
        cax.text(
            0.0,
            1.38,
            r"$\leftarrow$ Range decrease (toward / uplift)",
            transform=cax.transAxes,
            fontsize=8.0,
            color="#1D5B8F",
            ha="left",
            va="bottom",
            fontweight="medium",
        )
        cax.text(
            1.0,
            1.38,
            r"Range increase (away / subsidence) $\rightarrow$",
            transform=cax.transAxes,
            fontsize=8.0,
            color="#9C2B23",
            ha="right",
            va="bottom",
            fontweight="medium",
        )

    avail_w_in = (right_m - left_m) * fig_w
    scope = "Urban High-Coherence Focus" if focus_urban else "Full Stack Footprint"
    title = f"{spec['title']} — {format_site_label(stack_dir, ledger)}"
    subtitle = f"{format_date_range(ledger, summary)}  ·  {scope}"
    title_fs = min(14.5, max(10.5, avail_w_in * 72.0 / max(len(title) * 0.57, 1.0)))
    add_editorial_header(
        fig,
        title,
        subtitle,
        left=left_m,
        top=1.0 - (0.28 / fig_h),
        title_fontsize=title_fs,
        subtitle_gap=0.34 / fig_h,
    )

    if coh_fade and spec["coh_fade"]:
        fade_note = f"Smooth confidence opacity (γ {fade_low:.2f}–{fade_high:.2f} + velocity error)"
    elif layer == "velocity" and uncertainty_scale is not None:
        fade_note = "Opacity faded by velocity error"
    else:
        fade_note = "Full-opacity raster display"
    buf_m = summary.get("corridors", {}).get("buffer_m", 100.0)
    ramp_cfg = summary.get("ramp_removal", {})
    ramp_ord = ramp_cfg.get("order", 1)
    extras: list[str] = [f"Ramp: order-{ramp_ord} IRLS"]
    if ramp_cfg.get("dem_applied"):
        extras.append("DEM tropo: ON")
    if summary.get("reference_region", {}).get("applied"):
        extras.append("Ref pin: ON")
    extras.append(f"Corridor buffer: {buf_m:.0f} m")
    if spec["diverging"]:
        extras.append("Clip: manual" if vlim is not None else "Clip: p2.5–p97.5")
    if layer == "velocity":
        extras.append("DEM hillshade: ON" if hillshade is not None else "DEM hillshade: unavailable")
    extras.append(fade_note)
    footer_note = "  ·  ".join(extras)
    footer_fs = min(8.0, max(6.6, avail_w_in * 72.0 / max(len(footer_note) * 0.49, 1.0)))
    add_footer(
        fig,
        footer_note,
        left=left_m,
        bottom=0.14 / fig_h,
        fontsize=footer_fs,
    )

    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved map [{layer}] -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_plot_runtime(__file__, argv)
    p = argparse.ArgumentParser(description="Render a single standalone InSAR stack cartographic map.")
    p.add_argument("--stack-dir", required=True, type=Path, help="Stack directory containing COGs and corridor_report.geojson")
    p.add_argument("--layer", choices=sorted(LAYER_SPECS.keys()), default="velocity")
    p.add_argument("--out", type=Path, default=None, help="Output image path (default: <stack-dir>/figures/<layer>_map.png)")
    p.add_argument("--focus-urban", action="store_true", help="Zoom cleanly onto the high-coherence urban core")
    p.add_argument("--focus-coh", type=float, default=0.62, help="Coherence threshold for --focus-urban")
    p.add_argument("--no-corridors", action="store_true", help="Disable corridor centerline overlay")
    p.add_argument("--show-alerts", action="store_true", help="Overlay high-coherence corridor alert points")
    p.add_argument("--no-coh-fade", action="store_true", help="Disable coherence opacity fade")
    p.add_argument("--label-top", type=int, default=5, help="Number of top coherent corridors to label directly")
    p.add_argument("--min-label-coh", type=float, default=0.60, help="Minimum coherence to label a corridor")
    p.add_argument("--vlim", type=float, default=None, help="Override symmetric diverging limit (or vmax for sequential layers)")
    p.add_argument("--dem", type=Path, default=None, help="DEM for velocity hillshade (defaults to the path recorded in stack_ledger.json)")
    p.add_argument("--no-hillshade", action="store_true", help="Disable DEM relief under and within the velocity layer")
    p.add_argument("--hillshade-strength", type=float, default=0.13, help="Velocity luminance modulation from 0 (off) to 1 (strong)")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args(argv)

    suffix = f"{args.layer}_urban_map.png" if args.focus_urban else f"{args.layer}_map.png"
    out = args.out or (args.stack_dir / "figures" / suffix)
    render_raster_map(
        args.stack_dir,
        out,
        layer=args.layer,
        show_corridors=not args.no_corridors,
        show_alerts=args.show_alerts,
        coh_fade=not args.no_coh_fade,
        focus_urban=args.focus_urban,
        focus_coh=args.focus_coh,
        label_top=args.label_top,
        min_label_coh=args.min_label_coh,
        vlim=args.vlim,
        dem_path=args.dem,
        show_hillshade=not args.no_hillshade,
        hillshade_strength=args.hillshade_strength,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
