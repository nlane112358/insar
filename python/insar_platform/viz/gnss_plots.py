"""Publication-grade GNSS vs. InSAR validation figure renderers."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))

if TYPE_CHECKING:
    from insar_platform.providers.unr_gnss import GnssStation, StationMatch


def _configure_scientific_matplotlib() -> Any:
    """Apply clean, standard scientific matplotlib rcParams without editorial styling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.labelweight": "normal",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.85,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "text.color": "black",
            "legend.frameon": True,
            "legend.fancybox": False,
        }
    )
    return plt


def _utm_axis_labels(projection_wkt: str) -> tuple[str, str]:
    """Derive UTM zone Easting/Northing axis labels from raster WKT when available."""
    if projection_wkt:
        from osgeo import osr  # type: ignore

        srs = osr.SpatialReference()
        if srs.ImportFromWkt(projection_wkt) == 0:
            zone = srs.GetUTMZone()
            if zone > 0:
                return f"UTM {zone}N Easting (km)", f"UTM {zone}N Northing (km)"
            if zone < 0:
                return f"UTM {-zone}S Easting (km)", f"UTM {-zone}S Northing (km)"
    return "Easting (km)", "Northing (km)"


def _draw_cartographic_scalebar(ax: Any, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Draw a minimal alternating black-and-white kilometer scale bar."""
    from matplotlib.patches import Rectangle

    width_m = xmax - xmin
    height_m = ymax - ymin
    target = width_m * 0.18
    nice_steps = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    bar_m = min(nice_steps, key=lambda s: abs(s - target))
    label = f"{bar_m // 1000} km" if bar_m >= 1000 else f"{bar_m} m"

    x0 = xmin + 0.05 * width_m
    y0 = ymin + 0.06 * height_m
    bar_h = 0.012 * height_m
    ax.add_patch(Rectangle((x0, y0), bar_m / 2.0, bar_h, facecolor="black", edgecolor="black", lw=0.7, zorder=8))
    ax.add_patch(Rectangle((x0 + bar_m / 2.0, y0), bar_m / 2.0, bar_h, facecolor="white", edgecolor="black", lw=0.7, zorder=8))
    ax.text(x0, y0 + bar_h + 0.01 * height_m, "0", fontsize=8.5, ha="center", va="bottom", color="black", zorder=8)
    ax.text(x0 + bar_m, y0 + bar_h + 0.01 * height_m, label, fontsize=8.5, ha="center", va="bottom", color="black", zorder=8)


def _compute_map_view_and_norm(
    vel_disp: np.ndarray,
    vel_extent: tuple[float, float, float, float],
    ct: Any,
    matches: Sequence[StationMatch],
    ref_lonlat: tuple[float, float] | None,
) -> tuple[tuple[float, float, float, float], float]:
    pts_utm = [ct.TransformPoint(m.lon, m.lat, 0.0)[:2] for m in matches]
    if ref_lonlat is not None:
        pts_utm.append(ct.TransformPoint(ref_lonlat[0], ref_lonlat[1], 0.0)[:2])
    xs_u = [p[0] for p in pts_utm]
    ys_u = [p[1] for p in pts_utm]
    pad_x = max(7000.0, 0.35 * (max(xs_u) - min(xs_u)))
    pad_y = max(6000.0, 0.35 * (max(ys_u) - min(ys_u)))
    xmin = max(vel_extent[0], min(xs_u) - pad_x)
    xmax = min(vel_extent[1], max(xs_u) + pad_x)
    ymin = max(vel_extent[2], min(ys_u) - pad_y)
    ymax = min(vel_extent[3], max(ys_u) + pad_y)

    max_mag = max(
        [abs(m.gnss_los_mm_yr) for m in matches]
        + [abs(m.insar_los_mm_yr) for m in matches]
        + [25.0]
    )
    step = 10.0 if max_mag <= 60.0 else 50.0
    vlim = math.ceil((max_mag * 1.15) / step) * step
    return (xmin, xmax, ymin, ymax), vlim


def render_gnss_velocity_map(
    stack_dir: Path,
    matches: list[StationMatch],
    out_path: Path,
    *,
    ref_lonlat: tuple[float, float] | None = None,
    insar_sign: float = -1.0,
    min_plot_coh: float = 0.25,
    dpi: int = 300,
) -> Path:
    """Render standalone InSAR LOS velocity map with GNSS stations and reference point."""
    from insar_platform.viz._common import build_wgs84_to_raster_ct, load_raster, save_figure
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt = _configure_scientific_matplotlib()
    vel_layer = load_raster(stack_dir / "velocity_mm_yr.tif")
    coh_layer = load_raster(stack_dir / "temporal_coherence.tif")
    vel_disp = insar_sign * vel_layer.data
    ct = build_wgs84_to_raster_ct(vel_layer.projection_wkt)

    (xmin, xmax, ymin, ymax), vlim = _compute_map_view_and_norm(
        vel_disp, vel_layer.extent, ct, matches, ref_lonlat
    )
    norm = Normalize(vmin=-vlim, vmax=vlim, clip=True)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#EAEAEA")

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.set_facecolor("#EAEAEA")
    masked_vel = np.where(np.isfinite(vel_disp) & (coh_layer.data >= min_plot_coh), vel_disp, np.nan)
    ax.imshow(
        masked_vel,
        extent=vel_layer.extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )

    label_offsets: dict[str, tuple[float, float, str, str]] = {
        "MXAZ": (900.0, 450.0, "left", "bottom"),
        "UNVA": (900.0, 400.0, "left", "bottom"),
        "MMX1": (0.0, -1450.0, "center", "top"),
        "MXMX": (900.0, 500.0, "left", "bottom"),
        "UTAC": (-900.0, -700.0, "right", "top"),
        "ICMX": (900.0, -700.0, "left", "top"),
    }
    for m in matches:
        ux, uy, _ = ct.TransformPoint(m.lon, m.lat, 0.0)
        col = cmap(norm(m.gnss_los_mm_yr))
        ax.scatter([ux], [uy], s=95, color=col, edgecolors="black", linewidths=1.2, zorder=6)
        dx, dy, ha, va = label_offsets.get(m.station_id, (850.0, 450.0, "left", "bottom"))
        ax.text(
            ux + dx,
            uy + dy,
            m.station_id,
            fontsize=8.5,
            color="black",
            ha=ha,
            va=va,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.88)],
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.1, markersize=7.0, label="GNSS station",
        )
    ]
    if ref_lonlat is not None:
        rx, ry, _ = ct.TransformPoint(ref_lonlat[0], ref_lonlat[1], 0.0)
        ax.scatter([rx], [ry], s=140, marker="*", color="#FFD700", edgecolors="black", linewidths=0.9, zorder=7)
        legend_handles.append(
            Line2D(
                [0], [0], marker="*", color="none", markerfacecolor="#FFD700",
                markeredgecolor="black", markeredgewidth=0.8, markersize=9.5, label="Reference point",
            )
        )

    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=8.5,
        frameon=True, fancybox=False, edgecolor="0.65", facecolor="white", framealpha=0.95,
    )
    _draw_cartographic_scalebar(ax, xmin, xmax, ymin, ymax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    km_fmt = FuncFormatter(lambda v, _: f"{v / 1000.0:.0f}")
    ax.xaxis.set_major_formatter(km_fmt)
    ax.yaxis.set_major_formatter(km_fmt)
    tick_step = 5000.0 if (xmax - xmin) <= 45000.0 else 10000.0
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(direction="out", length=4.0, width=0.8, top=False, right=False)
    xlabel, ylabel = _utm_axis_labels(vel_layer.projection_wkt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, extend="both")
    cb.set_label("LOS velocity (mm/yr)", labelpad=6)
    cb.ax.tick_params(labelsize=8.5, length=3.5, width=0.8)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved GNSS velocity map -> {out_path}")
    return out_path


def render_gnss_scatter_plot(
    matches: list[StationMatch],
    out_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    """Render standalone 1:1 GNSS vs. InSAR LOS velocity scatter figure."""
    from insar_platform.viz._common import save_figure
    from matplotlib.ticker import MultipleLocator

    plt = _configure_scientific_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))

    gx = np.array([m.gnss_los_mm_yr for m in matches])
    iy = np.array([m.insar_los_mm_yr for m in matches])
    g_err = np.array([m.gnss_los_sigma_mm_yr for m in matches])
    i_err = np.array([m.insar_spatial_std_mm_yr for m in matches])

    bx = [m.gnss_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None]
    by = [m.baseline_insar_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None]

    all_lo = [float(np.min(gx - g_err)), float(np.min(iy - i_err)), -40.0] + ([float(min(by))] if by else [])
    all_hi = [float(np.max(gx + g_err)), float(np.max(iy + i_err)), 25.0] + ([float(max(by))] if by else [])
    lo_lim = min(all_lo) - 18.0
    hi_lim = max(all_hi) + 14.0
    line_x = np.linspace(lo_lim, hi_lim, 200)

    ax.fill_between(line_x, line_x - 20.0, line_x + 20.0, color="0.91", alpha=0.8, zorder=1, label="±20 mm/yr")
    ax.fill_between(line_x, line_x - 10.0, line_x + 10.0, color="0.81", alpha=0.8, zorder=2, label="±10 mm/yr")
    ax.plot(line_x, line_x, "k--", linewidth=1.0, zorder=3, label="1:1 line")

    if bx and by:
        ax.scatter(bx, by, marker="x", s=40, color="0.45", linewidths=1.2, zorder=4, label="Baseline (unmasked)")

    ax.errorbar(
        gx, iy, xerr=g_err, yerr=i_err, fmt="o", markersize=6.0,
        color="#c0392b", ecolor="0.25", elinewidth=1.0, capsize=2.5,
        markeredgecolor="black", markeredgewidth=0.6, zorder=6,
        label="InSAR stack (±1σ)",
    )

    scatter_label_offsets: dict[str, tuple[float, float, str, str, bool]] = {
        "MMX1": (14.0, -14.0, "left", "top", False),
        "MXAZ": (-8.0, 4.0, "right", "bottom", False),
        "ICMX": (-10.0, 6.0, "right", "bottom", False),
        "UNVA": (-14.0, -34.0, "right", "top", True),
        "MXMX": (12.0, -28.0, "left", "top", True),
        "UTAC": (16.0, -4.0, "left", "center", True),
    }
    for m in matches:
        default_off = (10.0, -10.0, "left", "top", False) if m.gnss_los_mm_yr < -100 else (-8.0, 6.0, "right", "bottom", False)
        ox, oy, ha, va, arr = scatter_label_offsets.get(m.station_id, default_off)
        ax.annotate(
            m.station_id,
            (m.gnss_los_mm_yr, m.insar_los_mm_yr),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8.2,
            color="black",
            ha=ha,
            va=va,
            zorder=7,
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.7} if arr else None,
        )

    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.6, color="0.80")
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
    ax.set_xlabel("UNR GNSS LOS velocity (mm/yr)")
    ax.set_ylabel("InSAR LOS velocity (mm/yr)")
    ax.legend(
        loc="lower right", fontsize=8.3,
        frameon=True, fancybox=False, edgecolor="0.65", facecolor="white", framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved GNSS scatter plot -> {out_path}")
    return out_path


def render_gnss_timeseries_plot(
    matches: list[StationMatch],
    out_path: Path,
    *,
    ts_station: GnssStation | None = None,
    ts_gnss_dates: list[date] | None = None,
    ts_gnss_los_mm: list[float] | None = None,
    ts_insar_dates: list[date] | None = None,
    ts_insar_los_mm: list[float] | None = None,
    sensor_label: str = "Sentinel-1 InSAR",
    dpi: int = 300,
) -> Path:
    """Render standalone cumulative LOS displacement time-series overlay (or residual bars)."""
    from insar_platform.viz._common import save_figure
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator

    plt = _configure_scientific_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.1))

    if (
        ts_station is not None
        and ts_gnss_dates
        and ts_gnss_los_mm
        and ts_insar_dates
        and ts_insar_los_mm
    ):
        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8, zorder=1)
        ax.scatter(
            ts_gnss_dates, ts_gnss_los_mm, s=15, color="#1f77b4", alpha=0.65,
            edgecolors="none", zorder=3, label=f"UNR daily GNSS ({ts_station.station_id})",
        )
        ax.plot(
            ts_insar_dates, ts_insar_los_mm, "-o", color="#d62728", linewidth=1.6, markersize=4.3,
            markeredgecolor="black", markeredgewidth=0.45, zorder=5,
            label=f"{sensor_label} ({ts_station.station_id})",
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[15]))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.80")
        ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.2, width=0.6)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative LOS displacement (mm)")
        ax.legend(
            loc="lower left", fontsize=8.5,
            frameon=True, fancybox=False, edgecolor="0.65", facecolor="white", framealpha=0.95,
        )
    else:
        ids = [m.station_id for m in matches]
        x_idx = np.arange(len(ids))
        res_up = [m.residual_mm_yr for m in matches]
        if any(m.baseline_residual_mm_yr is not None for m in matches):
            res_base = [m.baseline_residual_mm_yr or 0.0 for m in matches]
            ax.bar(x_idx - 0.18, res_base, width=0.35, color="0.75", edgecolor="0.4", label="Baseline residual")
            ax.bar(x_idx + 0.18, res_up, width=0.35, color="#c0392b", edgecolor="black", linewidth=0.6, label="InSAR stack residual")
        else:
            ax.bar(x_idx, res_up, width=0.5, color="#1f77b4", edgecolor="black", linewidth=0.6, label="InSAR − GNSS")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(ids)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.80")
        ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
        ax.set_ylabel("Velocity residual (mm/yr)")
        ax.legend(loc="best", fontsize=8.5, frameon=True, fancybox=False, edgecolor="0.65", facecolor="white")

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved GNSS time-series plot -> {out_path}")
    return out_path


def render_gnss_validation_figures(
    *,
    stack_dir: Path,
    matches: list[StationMatch],
    ref_lonlat: tuple[float, float] | None = None,
    insar_sign: float = -1.0,
    ts_station: GnssStation | None = None,
    ts_gnss_dates: list[date] | None = None,
    ts_gnss_los_mm: list[float] | None = None,
    ts_insar_dates: list[date] | None = None,
    ts_insar_los_mm: list[float] | None = None,
    sensor_label: str = "Sentinel-1 InSAR",
    map_out: Path | None = None,
    scatter_out: Path | None = None,
    timeseries_out: Path | None = None,
    plots_dir: Path | None = None,
    plot_prefix: str = "gnss_validation",
    plot_out: Path | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Render requested standalone GNSS validation figures (map, scatter, time series)."""
    if plots_dir is not None:
        map_out = map_out or (plots_dir / f"{plot_prefix}_map.png")
        scatter_out = scatter_out or (plots_dir / f"{plot_prefix}_scatter.png")
        timeseries_out = timeseries_out or (plots_dir / f"{plot_prefix}_timeseries.png")
    elif plot_out is not None and not any([map_out, scatter_out, timeseries_out]):
        stem = plot_out.stem
        parent = plot_out.parent
        suffix = plot_out.suffix or ".png"
        map_out = parent / f"{stem}_map{suffix}"
        scatter_out = parent / f"{stem}_scatter{suffix}"
        timeseries_out = parent / f"{stem}_timeseries{suffix}"

    written: list[Path] = []
    if map_out is not None:
        written.append(
            render_gnss_velocity_map(
                stack_dir=stack_dir,
                matches=matches,
                out_path=map_out,
                ref_lonlat=ref_lonlat,
                insar_sign=insar_sign,
                dpi=dpi,
            )
        )
    if scatter_out is not None:
        written.append(
            render_gnss_scatter_plot(
                matches=matches,
                out_path=scatter_out,
                dpi=dpi,
            )
        )
    if timeseries_out is not None:
        written.append(
            render_gnss_timeseries_plot(
                matches=matches,
                out_path=timeseries_out,
                ts_station=ts_station,
                ts_gnss_dates=ts_gnss_dates,
                ts_gnss_los_mm=ts_gnss_los_mm,
                ts_insar_dates=ts_insar_dates,
                ts_insar_los_mm=ts_insar_los_mm,
                sensor_label=sensor_label,
                dpi=dpi,
            )
        )
    return written
