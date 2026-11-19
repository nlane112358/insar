"""Shared styling, GDAL I/O, cartographic helpers, and layout utilities for InSAR plot scripts."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence


# Ensure Matplotlib writes its cache to a writable temp folder before import
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))


def ensure_plot_runtime(script_file: str, argv: Sequence[str] | None = None) -> None:
    """Re-execute under Homebrew Python if current interpreter lacks matplotlib or osgeo."""
    from insar_platform.paths import ensure_runtime_python

    ensure_runtime_python(script_file, ("matplotlib", "numpy", "osgeo"), argv, set_mpl_cache=True)
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        from osgeo import gdal, osr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Plotting scripts require numpy, matplotlib, and GDAL (osgeo). "
            "Run with /opt/homebrew/bin/python3 or install them into your environment."
        ) from exc


# Editorial palette tokens
INK = "#1B222C"
MUTED = "#596574"
LIGHT_MUTED = "#8A95A5"
GRID_COLOR = "#E4E7EC"
SPINE_COLOR = "#C8D0DA"
PAPER_BG = "#FFFFFF"
SOFT_PANEL = "#F8F7F4"
NODATA_BG = "#ECEAE4"

ASSET_CLASS_COLORS: dict[str, str] = {
    "metro": "#B8332A",
    "rail": "#1D5B8F",
    "light_rail": "#2980B9",
    "motorway": "#C97A16",
    "trunk": "#B86B28",
    "canal": "#13856E",
    "river": "#2B7BBA",
    "reservoir": "#633974",
    "dam": "#4A235A",
    "water_works": "#0E7490",
    "airport_runway": "#2C3E50",
    "infrastructure": "#627282",
}

ASSET_CLASS_LABELS: dict[str, str] = {
    "metro": "Metro",
    "rail": "Rail",
    "light_rail": "Light Rail",
    "motorway": "Motorway",
    "trunk": "Trunk Road",
    "canal": "Canal",
    "river": "River",
    "reservoir": "Reservoir",
    "dam": "Dam",
    "water_works": "Water Works (AMP8)",
    "airport_runway": "Runway",
    "infrastructure": "Infrastructure",
}

# Okabe-Ito inspired editorial series palette for multi-line plots
SERIES_PALETTE: list[str] = [
    "#B8332A",  # crimson
    "#1D5B8F",  # deep cobalt
    "#C97A16",  # warm ochre
    "#13856E",  # pine teal
    "#6B3FA0",  # royal plum
    "#2E86AB",  # steel cyan
    "#A65628",  # terracotta brown
    "#556B2F",  # olive
    "#C2598C",  # muted rose
    "#4B5563",  # charcoal slate
]


def configure_matplotlib() -> Any:
    """Apply clean publication-grade rcParams and return pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PAPER_BG,
            "savefig.facecolor": PAPER_BG,
            "savefig.edgecolor": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Inter", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 10.0,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "text.color": INK,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def make_diverging_cmap() -> Any:
    """Return Crameri's perceptually uniform vik map, with a built-in fallback."""
    try:
        from cmcrameri import cm as cmc

        cmap = cmc.vik.copy()
        cmap.name = "insar_vik"
        return cmap
    except ImportError:
        pass

    from matplotlib.colors import LinearSegmentedColormap

    stops = [
        (0.00, "#143D59"),  # deep navy (range decrease / uplift)
        (0.22, "#2E7EB0"),  # slate blue
        (0.40, "#8FC1DC"),  # soft sky
        (0.50, "#F6F5F0"),  # quiet warm ivory at zero
        (0.60, "#F0B98D"),  # soft apricot
        (0.78, "#D45D35"),  # terracotta
        (1.00, "#7E1717"),  # deep crimson (range increase / subsidence)
    ]
    return LinearSegmentedColormap.from_list("insar_diverging", stops, N=256)


def make_coherence_cmap() -> Any:
    """Sequential muted slate-to-cream colormap for interferometric coherence [0..1]."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = [
        (0.00, "#18202B"),
        (0.30, "#3C5366"),
        (0.60, "#6F949C"),
        (0.85, "#B7C9BD"),
        (1.00, "#F5F2E6"),
    ]
    return LinearSegmentedColormap.from_list("insar_coherence", stops, N=256)


def make_uncertainty_cmap() -> Any:
    """Sequential ivory-to-amber-plum colormap for velocity uncertainty (mm/yr)."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = [
        (0.00, "#F6F5F0"),
        (0.35, "#E9C46A"),
        (0.70, "#D96B38"),
        (1.00, "#5E1938"),
    ]
    return LinearSegmentedColormap.from_list("insar_uncertainty", stops, N=256)


@dataclass(frozen=True)
class RasterLayer:
    data: Any  # np.ndarray float64 (H, W)
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def load_raster(path: Path) -> RasterLayer:
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = tuple(float(v) for v in ds.GetGeoTransform())
    proj = ds.GetProjection() or ""
    h, w = arr.shape
    ds = None
    extent = (gt[0], gt[0] + w * gt[1], gt[3] + h * gt[5], gt[3])
    return RasterLayer(data=arr, geotransform=gt, projection_wkt=proj, extent=extent)


def compute_valid_extent(
    arr: Any,
    gt: tuple[float, float, float, float, float, float],
    mask: Any | None = None,
    pad_frac: float = 0.035,
) -> tuple[float, float, float, float]:
    """Return tight (xmin, xmax, ymin, ymax) around valid raster pixels."""
    import numpy as np

    valid = np.isfinite(arr) if mask is None else (np.isfinite(arr) & mask)
    ys, xs = np.where(valid)
    h, w = arr.shape
    if ys.size == 0:
        return (gt[0], gt[0] + w * gt[1], gt[3] + h * gt[5], gt[3])

    x0 = gt[0] + float(xs.min()) * gt[1]
    x1 = gt[0] + float(xs.max() + 1) * gt[1]
    y_top = gt[3] + float(ys.min()) * gt[5]
    y_bot = gt[3] + float(ys.max() + 1) * gt[5]
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y_top, y_bot), max(y_top, y_bot)
    px = (xmax - xmin) * pad_frac
    py = (ymax - ymin) * pad_frac
    return (xmin - px, xmax + px, ymin - py, ymax + py)


def load_stack_summary(stack_dir: Path) -> dict[str, Any]:
    p = stack_dir / "stack_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_stack_ledger(stack_dir: Path) -> dict[str, Any]:
    p = stack_dir / "stack_ledger.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_corridor_report(stack_dir: Path) -> list[dict[str, Any]]:
    p = stack_dir / "corridor_report.geojson"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("features", [])


def load_corridor_timeseries(stack_dir: Path) -> dict[str, Any]:
    p = stack_dir / "corridor_timeseries.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_corridor_alerts(stack_dir: Path) -> list[dict[str, Any]]:
    p = stack_dir / "corridor_alerts.geojson"
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("features", [])


def build_wgs84_to_raster_ct(projection_wkt: str) -> Any:
    from osgeo import osr

    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(projection_wkt)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src, dst)


def reproject_geojson_geometry(geom: dict[str, Any], ct: Any) -> list[list[tuple[float, float]]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    paths: list[list[tuple[float, float]]] = []

    def _xf(pts: list[list[float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in pts:
            x, y, _ = ct.TransformPoint(float(p[0]), float(p[1]), 0.0)
            out.append((x, y))
        return out

    if gtype == "LineString":
        paths.append(_xf(coords))
    elif gtype == "MultiLineString":
        for line in coords:
            paths.append(_xf(line))
    elif gtype == "Polygon" and coords:
        paths.append(_xf(coords[0]))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                paths.append(_xf(poly[0]))
    return paths


def parse_iso_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def format_site_label(stack_dir: Path, ledger: dict[str, Any]) -> str:
    raw = stack_dir.name.replace("_", " ").strip()
    upper_tokens = {"iw1", "iw2", "iw3", "nisar", "uk", "amp8", "t023", "t029", "t052", "t101", "t173"}
    pretty = " ".join(w.upper() if w.lower() in upper_tokens else w.title() for w in raw.split())
    stream = ledger.get("stream_key")
    return f"{pretty} ({stream})" if stream else pretty


def format_date_range(ledger: dict[str, Any], summary: dict[str, Any]) -> str:
    pair_ids: list[str] = ledger.get("pair_ids", [])
    pairs = summary.get("pair_count", len(pair_ids))
    span_d = summary.get("total_span_days")
    if pair_ids:
        start = pair_ids[0].split("_")[0]
        end = pair_ids[-1].split("_")[1]
        d0 = parse_iso_date(start).strftime("%d %b %Y")
        d1 = parse_iso_date(end).strftime("%d %b %Y")
        span_txt = f"{span_d:.0f} stacked days" if span_d is not None else ""
        return f"{d0} – {d1}  ·  {pairs} {'pair' if pairs == 1 else 'pairs'}" + (f" ({span_txt})" if span_txt else "")
    return f"{pairs} pairs"


def add_editorial_header(
    fig: Any,
    title: str,
    subtitle: str,
    *,
    left: float = 0.06,
    top: float = 0.955,
    title_fontsize: float = 14.5,
    subtitle_gap: float = 0.036,
) -> None:
    fig.text(left, top, title, fontsize=title_fontsize, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(left, top - subtitle_gap, subtitle, fontsize=9.8, color=MUTED, ha="left", va="top")


def add_footer(
    fig: Any,
    note: str,
    *,
    left: float = 0.06,
    bottom: float = 0.018,
    fontsize: float = 8.0,
) -> None:
    fig.text(left, bottom, note, fontsize=fontsize, color=LIGHT_MUTED, ha="left", va="bottom")


def add_scalebar(ax: Any, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Draw a minimal cartographic kilometer scale bar inside the lower-left map corner."""
    width_m = xmax - xmin
    height_m = ymax - ymin
    target = width_m * 0.18
    # Snap to clean 1-2-5 kilometer step
    nice_steps = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    bar_m = min(nice_steps, key=lambda s: abs(s - target))
    label = f"{bar_m // 1000} km" if bar_m >= 1000 else f"{bar_m} m"

    x0 = xmin + 0.04 * width_m
    y0 = ymin + 0.055 * height_m
    bar_h = 0.011 * height_m

    import matplotlib.patches as patches

    # Crisp white halo backing pill
    pad_x = 0.014 * width_m
    pad_y = 0.014 * height_m
    bg = patches.FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y),
        bar_m + 2 * pad_x,
        bar_h + 0.038 * height_m + pad_y,
        boxstyle="round,pad=0.0,rounding_size=" + str(0.008 * width_m),
        facecolor="#FFFFFF",
        edgecolor="none",
        alpha=0.82,
        zorder=14,
    )
    ax.add_patch(bg)
    ax.add_patch(patches.Rectangle((x0, y0), bar_m / 2.0, bar_h, facecolor=INK, edgecolor=INK, linewidth=0.6, zorder=15))
    ax.add_patch(patches.Rectangle((x0 + bar_m / 2.0, y0), bar_m / 2.0, bar_h, facecolor="#FFFFFF", edgecolor=INK, linewidth=0.6, zorder=15))
    ax.text(x0, y0 + bar_h + 0.009 * height_m, "0", fontsize=7.8, color=INK, ha="left", va="bottom", zorder=16)
    ax.text(x0 + bar_m, y0 + bar_h + 0.009 * height_m, label, fontsize=7.8, fontweight="bold", color=INK, ha="right", va="bottom", zorder=16)


def repel_y_positions(targets: list[float], min_gap: float, y_min: float, y_max: float) -> list[float]:
    """Iteratively separate 1D vertical label anchors so direct right-margin labels never overlap."""
    if not targets:
        return []
    n = len(targets)
    order = sorted(range(n), key=lambda i: targets[i])
    pos = [targets[i] for i in order]

    for _ in range(120):
        moved = False
        for k in range(1, n):
            gap = pos[k] - pos[k - 1]
            if gap < min_gap:
                shift = 0.5 * (min_gap - gap)
                pos[k - 1] -= shift
                pos[k] += shift
                moved = True
        # Soft clamp inside bounds
        for k in range(n):
            if pos[k] < y_min:
                pos[k] = y_min
            elif pos[k] > y_max:
                pos[k] = y_max
        if not moved:
            break

    # Final forward/backward sweep if bounds caused compression
    for k in range(1, n):
        if pos[k] < pos[k - 1] + min_gap:
            pos[k] = pos[k - 1] + min_gap
    if pos[-1] > y_max:
        overflow = pos[-1] - y_max
        pos = [p - overflow for p in pos]
        for k in range(n - 2, -1, -1):
            if pos[k] > pos[k + 1] - min_gap:
                pos[k] = pos[k + 1] - min_gap

    out = [0.0] * n
    for rank, orig_idx in enumerate(order):
        out[orig_idx] = pos[rank]
    return out


def asset_color(asset_class: str) -> str:
    return ASSET_CLASS_COLORS.get(asset_class, ASSET_CLASS_COLORS["infrastructure"])


def asset_label(asset_class: str) -> str:
    return ASSET_CLASS_LABELS.get(asset_class, asset_class.replace("_", " ").title())


def shorten_name(name: str, max_len: int = 34) -> str:
    clean = name.replace("(Cuota)", "").strip()
    clean = clean.replace("Bicentenario", "").strip()
    clean = " ".join(clean.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def save_figure(fig: Any, out_path: Path, dpi: int = 220) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    return out_path
