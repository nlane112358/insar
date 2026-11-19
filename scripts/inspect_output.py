#!/usr/bin/env python3
"""
Inspect and visualize InSAR outputs (COG GeoTIFF directories or legacy .f32 binaries).

Examples:
    python3 scripts/inspect_output.py insar_work/products/NISAR_T153_F061_D_2026-10-22_2026-11-03
    python3 scripts/inspect_output.py insar_work/products/mexico_city_iw1/T041_086877_IW1_2026-11-11_2026-11-17 --show
    python3 scripts/inspect_output.py                # Legacy .f32 mode in current directory (or ./out)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def _import_matplotlib(show: bool):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to run inspect_output.py. "
            "Run with system Python where matplotlib & GDAL are installed: "
            "python3 scripts/inspect_output.py ..."
        ) from exc


def _import_gdal():
    try:
        from osgeo import gdal, osr
        gdal.UseExceptions()
        return gdal, osr
    except ImportError as exc:
        raise RuntimeError(
            "GDAL Python bindings (osgeo.gdal) are required to inspect GeoTIFF outputs. "
            "Run with: python3 scripts/inspect_output.py <product_dir>"
        ) from exc


def _read_cog(path: Path, gdal) -> tuple[np.ndarray, tuple[float, ...] | None, str]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Could not open GeoTIFF: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == np.float32(nodata), np.nan, arr)
    gt = ds.GetGeoTransform(can_return_null=True)
    proj = ds.GetProjectionRef() or ""
    ds = None
    return arr, gt, proj


def _load_alert_coords(
    alerts_path: Path,
    proj_wkt: str,
    use_map_coords: bool,
    gt: tuple[float, ...] | None,
    osr,
) -> tuple[list[float], list[float]]:
    if not alerts_path.is_file():
        return [], []
    data = json.loads(alerts_path.read_text(encoding="utf-8"))
    features = data.get("features") or []
    if not features:
        return [], []

    ct = None
    if proj_wkt:
        src = osr.SpatialReference()
        src.SetWellKnownGeogCS("WGS84")
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        if dst.ImportFromWkt(proj_wkt) == 0:
            dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ct = osr.CoordinateTransformation(src, dst)

    xs: list[float] = []
    ys: list[float] = []
    for feat in features:
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        if ct is not None:
            tx, ty, _ = ct.TransformPoint(lon, lat)
            if use_map_coords:
                xs.append(tx)
                ys.append(ty)
            elif gt is not None and gt[1] != 0 and gt[5] != 0:
                xs.append((tx - gt[0]) / gt[1])
                ys.append((ty - gt[3]) / gt[5])
    return xs, ys


def inspect_cog_dir(
    product_dir: Path,
    out_png: Path | None,
    vmin: float | None,
    vmax: float | None,
    show_alerts: bool,
    show: bool,
) -> Path:
    plt = _import_matplotlib(show)
    gdal, osr = _import_gdal()

    wrapped_path = product_dir / "wrapped_phase.tif"
    coh_path = product_dir / "coherence.tif"
    disp_path = product_dir / "displacement_mm.tif"
    for p in (wrapped_path, coh_path, disp_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing expected COG file: {p}")

    wrapped, gt, proj = _read_cog(wrapped_path, gdal)
    coh, _, _ = _read_cog(coh_path, gdal)
    disp, _, _ = _read_cog(disp_path, gdal)

    h, w = disp.shape
    valid = np.isfinite(disp) & np.isfinite(coh)
    valid_count = int(np.count_nonzero(valid))

    summary_path = product_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    print(f"Product directory: {product_dir}")
    print(f"Dimensions:        {w} x {h} ({valid_count:,} / {w * h:,} valid pixels, {100.0 * valid_count / (w * h):.1f}%)")
    if valid_count > 0:
        d_vals = disp[valid]
        c_vals = coh[valid]
        p5, med, p95 = np.percentile(d_vals, [5, 50, 95])
        print(f"Coherence:         mean={c_vals.mean():.3f}, median={np.median(c_vals):.3f}")
        print(
            f"Displacement (mm): min={d_vals.min():.2f}, p5={p5:.2f}, "
            f"median={med:.2f}, p95={p95:.2f}, max={d_vals.max():.2f}"
        )
    if summary:
        print(
            f"Compute stats:     total={summary.get('processing_time_ms', 0):.1f} ms, "
            f"iters={summary.get('iterations')}, anomalies={summary.get('anomalies_detected')}"
        )

    use_map_coords = bool(
        gt is not None and proj and abs(gt[1]) > 0 and abs(gt[5]) > 0 and not (gt[0] == 0.0 and gt[3] == 0.0)
    )
    if use_map_coords and gt is not None:
        x0, x1 = gt[0], gt[0] + w * gt[1]
        y1, y0 = gt[3], gt[3] + h * gt[5]  # gt[5] is negative for north-up rasters
        extent = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
        aspect = "equal"
        xlabel, ylabel = "Easting (m)", "Northing (m)"
    else:
        extent = None
        aspect = "equal"
        xlabel, ylabel = "X (px)", "Y (px)"

    if vmin is None or vmax is None:
        if valid_count > 0:
            sym = max(10.0, float(np.percentile(np.abs(disp[valid]), 99)))
            sym = math.ceil(sym / 5.0) * 5.0
        else:
            sym = 50.0
        disp_vmin = -sym if vmin is None else vmin
        disp_vmax = sym if vmax is None else vmax
    else:
        disp_vmin, disp_vmax = vmin, vmax

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    cmap_phase = plt.get_cmap("twilight").copy()
    cmap_phase.set_bad(color="#f0f0f0")
    cmap_coh = plt.get_cmap("gray").copy()
    cmap_coh.set_bad(color="#f0f0f0")
    cmap_disp = plt.get_cmap("RdBu_r").copy()
    cmap_disp.set_bad(color="#f0f0f0")

    im0 = axes[0].imshow(
        np.ma.masked_invalid(wrapped), cmap=cmap_phase, vmin=-np.pi, vmax=np.pi, extent=extent, aspect=aspect
    )
    axes[0].set_title("Wrapped Phase (rad)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        np.ma.masked_invalid(coh), cmap=cmap_coh, vmin=0.0, vmax=1.0, extent=extent, aspect=aspect
    )
    axes[1].set_title("Coherence")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        np.ma.masked_invalid(disp), cmap=cmap_disp, vmin=disp_vmin, vmax=disp_vmax, extent=extent, aspect=aspect
    )
    axes[2].set_title("LOS Displacement (mm)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    if show_alerts:
        ax_pts_x, ax_pts_y = _load_alert_coords(
            product_dir / "alerts.geojson", proj, use_map_coords, gt, osr
        )
        if ax_pts_x:
            axes[2].scatter(
                ax_pts_x, ax_pts_y, s=36, facecolors="none", edgecolors="lime", linewidths=1.3,
                label=f"Alerts ({len(ax_pts_x)})",
            )
            axes[2].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if use_map_coords:
            ax.ticklabel_format(style="plain", useOffset=False)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(product_dir.name, fontsize=12)
    dest = out_png or (product_dir / "preview.png")
    fig.savefig(dest, dpi=150)
    print(f"Saved {dest}")
    if show:
        plt.show()
    plt.close(fig)
    return dest


def inspect_legacy_f32(width: int | None, height: int | None, out_png: Path | None, show: bool) -> Path:
    plt = _import_matplotlib(show)

    RAD_TO_MM = 55.4658 / (4.0 * np.pi)
    if width is not None and height is not None:
        w, h = width, height
    elif width is not None:
        w = h = width
    else:
        default_dim = 1024
        if os.path.exists("wrapped_phase.f32"):
            n_floats = os.path.getsize("wrapped_phase.f32") // 4
            side = math.isqrt(n_floats)
            if side * side == n_floats and side > 0:
                default_dim = side
        w = h = default_dim

    wrapped = np.fromfile("wrapped_phase.f32", dtype=np.float32).reshape(h, w)
    coh = np.fromfile("coherence.f32", dtype=np.float32).reshape(h, w)
    unw = np.fromfile("unwrapped_phase.f32", dtype=np.float32).reshape(h, w) * RAD_TO_MM
    truth = np.fromfile("true_phase.f32", dtype=np.float32).reshape(h, w) * RAD_TO_MM

    unw -= np.median(unw[coh > 0.6] - truth[coh > 0.6])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Wrapped Phase (rad)")
    axes[1].imshow(coh, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Coherence")
    im = axes[2].imshow(unw, cmap="RdBu_r", vmin=-100, vmax=20)
    axes[2].set_title("CPU Unwrapped LOS (mm)")
    axes[3].imshow(unw - truth, cmap="coolwarm", vmin=-5, vmax=5)
    axes[3].set_title("Residual Error (mm)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()

    dest = out_png or Path("insar_proto_check.png")
    plt.savefig(dest, dpi=150)
    print(f"Saved {dest}")
    if show:
        plt.show()
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize InSAR GeoTIFF or legacy .f32 outputs.")
    parser.add_argument(
        "target",
        nargs="?",
        help="Product directory containing *.tif COGs, or legacy width (int)",
    )
    parser.add_argument("height", nargs="?", type=int, help="Optional legacy height when target is width")
    parser.add_argument("--dir", dest="product_dir", type=Path, help="Product directory containing *.tif COGs")
    parser.add_argument("--out", type=Path, help="Output preview PNG path")
    parser.add_argument("--vmin", type=float, help="Minimum displacement colorbar value (mm)")
    parser.add_argument("--vmax", type=float, help="Maximum displacement colorbar value (mm)")
    parser.add_argument("--no-alerts", action="store_true", help="Do not overlay alerts.geojson markers")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    args = parser.parse_args(argv)

    product_dir = args.product_dir
    legacy_w: int | None = None
    legacy_h: int | None = args.height

    if product_dir is None and args.target is not None:
        if args.target.isdigit():
            legacy_w = int(args.target)
        else:
            product_dir = Path(args.target)

    if product_dir is None and args.target is None:
        if Path("wrapped_phase.f32").is_file():
            inspect_legacy_f32(None, None, args.out, args.show)
            return 0
        if Path("wrapped_phase.tif").is_file():
            product_dir = Path(".")
        elif Path("out/wrapped_phase.tif").is_file():
            product_dir = Path("out")

    if product_dir is not None:
        inspect_cog_dir(
            product_dir=product_dir,
            out_png=args.out,
            vmin=args.vmin,
            vmax=args.vmax,
            show_alerts=not args.no_alerts,
            show=args.show,
        )
        return 0

    inspect_legacy_f32(legacy_w, legacy_h, args.out, args.show)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
