#!/usr/bin/env python3
"""
Render a single standalone cartographic map for any derived InSAR stack GeoTIFF layer
(velocity, cumulative displacement, temporal coherence, uncertainty, or acceleration)
with coherence-modulated shading, crisp infrastructure corridor overlays, km scalebar,
and dynamically proportioned layout.
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


def _clip_rgba_with_coherence(
    data: Any,
    coh: Any | None,
    cmap: Any,
    vmin: float,
    vmax: float,
    fade_low: float,
    fade_high: float,
    use_fade: bool,
) -> Any:
    import numpy as np
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    finite = np.isfinite(data)
    safe = np.where(finite, data, 0.0)
    rgba = cmap(norm(safe))
    rgba[~finite, 3] = 0.0

    if use_fade and coh is not None:
        c = np.where(np.isfinite(coh), coh, 0.0)
        alpha = np.clip((c - fade_low) / max(fade_high - fade_low, 1e-4), 0.14, 1.0)
        rgba[..., 3] = np.where(finite, alpha, 0.0)
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
    dpi: int = 220,
) -> Path:
    import numpy as np
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
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

    if spec["diverging"]:
        cmap = make_diverging_cmap()
        if vlim is not None:
            lim = abs(float(vlim))
        else:
            mask = np.isfinite(sub)
            if sub_coh is not None and (mask & (sub_coh >= 0.50)).sum() > 200:
                mask = mask & (sub_coh >= 0.50)
            vals = np.abs(sub[mask])
            lim = float(np.nanpercentile(vals, spec["percentile"])) if vals.size else spec["min_abs"]
            lim = max(lim, spec["min_abs"])
            # Snap cleanly to tidy tick bound
            step = 50.0 if lim > 150 else (20.0 if lim > 50 else (10.0 if lim > 20 else 5.0))
            lim = math.ceil(lim / step) * step
        vmin, vmax = -lim, lim
    elif layer == "coherence":
        cmap = make_coherence_cmap()
        vmin, vmax = 0.0, 1.0
    else:
        cmap = make_uncertainty_cmap()
        vals = sub[np.isfinite(sub)]
        if vals.size == 0:
            raise RuntimeError(f"No finite values in {raster_path} (uncertainty requires >= 2 stacked pairs).")
        vmax = float(vlim) if vlim is not None else max(5.0, float(np.nanpercentile(vals, spec["percentile"])))
        vmin = 0.0

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

    rgba = _clip_rgba_with_coherence(
        rlayer.data,
        coh_layer.data if coh_layer is not None else None,
        cmap,
        vmin,
        vmax,
        fade_low=fade_low,
        fade_high=fade_high,
        use_fade=(coh_fade and spec["coh_fade"]),
    )
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
        ordered_cls = sorted(drawn_classes, key=lambda c: list(LAYER_SPECS.keys()) if False else c)
        handles = [
            Line2D([0], [0], color=asset_color(c), lw=2.0, label=asset_label(c))
            for c in ordered_cls[:6]
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
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
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

    fade_note = f"Opacity faded below γ={fade_high:.2f} to suppress incoherent terrain" if (coh_fade and spec["coh_fade"]) else "Full-opacity raster display"
    buf_m = summary.get("corridors", {}).get("buffer_m", 100.0)
    ramp_ord = summary.get("ramp_removal", {}).get("order", 1)
    footer_note = f"Ramp removal: order-{ramp_ord} robust Huber/Tukey IRLS  ·  Corridor buffer: {buf_m:.0f} m  ·  {fade_note}"
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
    p.add_argument("--vlim", type=float, default=None, help="Override symmetric color limit (or vmax for sequential layers)")
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
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
