#!/usr/bin/env python3
"""Replay a stack into static panels and a synchronized InSAR/GNSS animation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "insar_mpl_cache"))

from insar_platform.paths import REPO_ROOT as _ROOT, resolve_runtime_path
from insar_platform.pipelines.stacking import StackConfig, sync_stack
from insar_platform.viz._common import (
    build_wgs84_to_raster_ct,
    configure_matplotlib,
    load_raster,
    make_diverging_cmap,
    save_figure,
)


def _unfingerprint(value: str | None) -> Path | None:
    if not value:
        return None
    raw = value.rsplit(":", 1)[0] if ":" in value else value
    path = resolve_runtime_path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"Stack configuration asset recorded in ledger is missing: {path}")
    return path


def _load_replay_config(
    source_stack_dir: Path,
    products_dir: Path,
    work_dir: Path,
    snapshot_dir: Path,
    stack_bin: Path,
) -> StackConfig:
    ledger_path = source_stack_dir / "stack_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    params = ledger.get("stack_params", {})
    exports = ledger.get("export_params", {})
    lonlat = params.get("ref_lonlat")
    ref_px = params.get("ref_px")
    return StackConfig(
        products_dir=products_dir,
        out_dir=work_dir,
        stack_bin=stack_bin,
        stream_key=ledger.get("stream_key"),
        ramp_order=int(params.get("ramp_order", 1)),
        min_coh=float(params.get("min_coh", 0.35)),
        dem=_unfingerprint(params.get("dem")),
        ref_lonlat=(float(lonlat[0]), float(lonlat[1])) if lonlat else None,
        ref_radius_m=float(params.get("ref_radius_m") or 1000.0),
        ref_px=tuple(int(v) for v in ref_px) if ref_px else None,
        ref_region=_unfingerprint(params.get("ref_region")),
        deform_mask=_unfingerprint(params.get("deform_mask")),
        prior_vel=_unfingerprint(params.get("prior_vel")),
        two_pass_deform_mask=bool(params.get("two_pass_deform_mask", False)),
        ramp_deform_vel_mm_yr=float(params.get("ramp_deform_vel_mm_yr", 25.0)),
        ramp_mask_sigma=float(params.get("ramp_mask_sigma", 2.5)),
        ramp_mask_dilate_px=int(params.get("ramp_mask_dilate_px", 2)),
        enable_closure=bool(params.get("enable_closure", True)),
        closure_thresh_mm=float(params.get("closure_thresh_mm", 10.0)),
        closure_max_bad_frac=float(params.get("closure_max_bad_frac", 0.35)),
        closure_mode=str(params.get("closure_mode", "consensus")),
        include_redundant=bool(params.get("include_redundant", True)),
        redundant_weight_scale=float(params.get("redundant_weight_scale", 0.5)),
        corridors=_unfingerprint(exports.get("corridors")),
        buffer_m=float(exports.get("buffer_m", 100.0)),
        vel_alert_mm_yr=float(exports.get("vel_alert_mm_yr", 15.0)),
        accel_alert_mm_yr2=float(exports.get("accel_alert_mm_yr2", 25.0)),
        min_alert_coh=float(exports.get("min_alert_coh", 0.55)),
        min_alert_snr=float(exports.get("min_alert_snr", 2.5)),
        min_alert_span_days=float(exports.get("min_alert_span_days", 60.0)),
        snapshots_dir=snapshot_dir,
        reset=True,
    )


def _read_frames(snapshot_dir: Path) -> list[tuple[str, Path, Path, tuple[float, float, float, float]]]:
    frames = []
    for cumulative_path in sorted(snapshot_dir.glob("*_cumulative_disp_mm.tif")):
        date = cumulative_path.name.removesuffix("_cumulative_disp_mm.tif")
        coherence_path = snapshot_dir / f"{date}_temporal_coherence.tif"
        if not coherence_path.is_file():
            continue
        cumulative = load_raster(cumulative_path)
        frames.append((date, cumulative_path, coherence_path, cumulative.extent))
    if not frames:
        raise RuntimeError(f"No accepted primary-pair snapshots were written under {snapshot_dir}")
    return frames


def _frame_arrays(frame: tuple[str, Any, Any, tuple[float, float, float, float]]) -> tuple[Any, Any]:
    import numpy as np

    displacement_ref, coherence_ref = frame[1], frame[2]
    if isinstance(displacement_ref, (str, Path)):
        displacement = load_raster(Path(displacement_ref)).data.astype(np.float32, copy=False)
        coherence = load_raster(Path(coherence_ref)).data
        coherent_mask = np.isfinite(coherence) & (coherence >= 0.4)
        return displacement, coherent_mask
    displacement = np.asarray(displacement_ref, dtype=np.float32)
    coherence = np.asarray(coherence_ref)
    coherent_mask = coherence if coherence.dtype == bool else (np.isfinite(coherence) & (coherence >= 0.4))
    return displacement, coherent_mask


def render_evolution_panels(frames: list[tuple[str, Any, Any, tuple[float, float, float, float]]], out_path: Path, *, dpi: int = 240) -> Path:
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    count = min(9, len(frames))
    indices = np.unique(np.rint(np.linspace(0, len(frames) - 1, count)).astype(int))
    chosen = [frames[i] for i in indices]
    extent = chosen[0][3]
    masked_values = []
    for frame in chosen:
        displacement, coherence_valid = _frame_arrays(frame)
        valid = np.isfinite(displacement) & coherence_valid
        sample = np.abs(displacement[::8, ::8])
        sample_valid = valid[::8, ::8]
        masked_values.append(sample[sample_valid])
        del displacement, coherence_valid
    pooled = np.concatenate([a for a in masked_values if a.size]) if any(a.size for a in masked_values) else np.array([10.0])
    limit = max(5.0, float(np.nanpercentile(pooled, 90.0)))

    plt = configure_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 10.4), constrained_layout=True)
    div_cmap = make_diverging_cmap().copy()
    div_cmap.set_under(div_cmap(0.0))
    div_cmap.set_over(div_cmap(1.0))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(chosen):
            ax.set_axis_off()
            continue
        date, _disp, _coh, frame_extent = chosen[idx]
        displacement, coherence_valid = _frame_arrays(chosen[idx])
        valid = np.isfinite(displacement) & coherence_valid
        shown = np.where(valid, displacement, np.nan)
        ax.imshow(shown, extent=frame_extent, origin="upper", cmap=div_cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(date, fontsize=10.5, fontweight="bold")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d4d9e0")
            spine.set_linewidth(0.65)

    mappable = ScalarMappable(norm=Normalize(-limit, limit), cmap=div_cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=list(axes.flat), shrink=0.78, pad=0.018, fraction=0.025, extend="both")
    colorbar.set_label("Cumulative LOS displacement (mm)")
    fig.suptitle("Mexico City basin deformation through the stack", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.008, f"{len(frames)} accepted pair updates · shared p90 color limits ±{limit:.0f} mm · values outside the color range are clipped · temporal coherence ≥ 0.40", ha="center", fontsize=8.5, color="#596579")
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _read_station_record(validation_csv: Path, station_id: str) -> dict[str, float]:
    with validation_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("station_id") != station_id:
                continue
            required = ("lat", "lon", "los_e", "los_n", "los_u")
            try:
                return {key: float(row[key]) for key in required}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Incomplete LOS metadata for station {station_id} in {validation_csv}") from exc
    raise ValueError(f"Station {station_id} was not found in {validation_csv}")


def _read_gnss_los_series(
    tenv3_path: Path,
    station: dict[str, float],
    start_date: date,
    end_date: date,
) -> tuple[list[date], list[float]]:
    """Read UNR daily ENU positions and project them into the station's LOS direction."""
    import numpy as np
    from datetime import date as date_type

    records: list[tuple[date, float, float, float]] = []
    for line in tenv3_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].lower() == "site" or len(parts) < 23:
            continue
        try:
            day = date_type(1858, 11, 17) + timedelta(days=int(parts[3]))
            if start_date <= day <= end_date:
                east = float(parts[7]) + float(parts[8])
                north = float(parts[9]) + float(parts[10])
                up = float(parts[11]) + float(parts[12])
                records.append((day, east, north, up))
        except (TypeError, ValueError, OverflowError):
            continue
    if len(records) < 2:
        raise ValueError(f"No daily GNSS samples for {start_date} to {end_date} in {tenv3_path}")
    e0, n0, u0 = records[0][1:]
    values = [
        1000.0
        * (
            station["los_e"] * (east - e0)
            + station["los_n"] * (north - n0)
            + station["los_u"] * (up - u0)
        )
        for _day, east, north, up in records
    ]
    zero = float(np.median(values[: min(7, len(values))]))
    dates = [record[0] for record in records]
    return dates, [value - zero for value in values]


def _snapshot_station_value(frame: tuple[str, Any, Any, tuple[float, float, float, float]], x: float, y: float, radius_m: float, insar_sign: float) -> float:
    import numpy as np

    displacement, coherent = _frame_arrays(frame)
    xmin, xmax, ymin, ymax = frame[3]
    h, w = displacement.shape
    dx, dy = (xmax - xmin) / w, (ymax - ymin) / h
    col_center = (x - xmin) / dx - 0.5
    row_center = (ymax - y) / dy - 0.5
    radius_cols = max(1, int(math.ceil(radius_m / max(dx, 1e-9))))
    radius_rows = max(1, int(math.ceil(radius_m / max(dy, 1e-9))))
    c0, r0 = int(round(col_center)), int(round(row_center))
    rows = np.arange(max(0, r0 - radius_rows), min(h, r0 + radius_rows + 1))
    cols = np.arange(max(0, c0 - radius_cols), min(w, c0 + radius_cols + 1))
    if rows.size == 0 or cols.size == 0:
        return float("nan")
    xx = xmin + (cols[None, :] + 0.5) * dx
    yy = ymax - (rows[:, None] + 0.5) * dy
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius_m * radius_m
    local_valid = coherent[np.ix_(rows, cols)] & np.isfinite(displacement[np.ix_(rows, cols)]) & inside
    values = displacement[np.ix_(rows, cols)][local_valid]
    if values.size == 0:
        return float("nan")
    # Validation uses insar_sign=-1 for range-increase products; default to that convention.
    return float(insar_sign * np.nanmedian(values))


def render_evolution_animation(
    frames: list[tuple[str, Any, Any, tuple[float, float, float, float]]],
    out_path: Path,
    *,
    gnss_cache_dir: Path,
    validation_csv: Path,
    station_id: str = "MMX1",
    gnss_start_date: date | None = None,
    insar_sign: float = -1.0,
    sample_radius_m: float = 200.0,
    fps: int = 2,
    dpi: int = 110,
) -> Path:
    """Animate cumulative deformation beside its synchronized GNSS validation curve."""
    import numpy as np
    import matplotlib.dates as mdates
    from matplotlib import animation

    if not frames:
        raise ValueError("Cannot animate an empty stack history")
    station = _read_station_record(validation_csv, station_id)
    tenv3_path = gnss_cache_dir / f"{station_id}.NA.tenv3"
    if not tenv3_path.is_file():
        raise FileNotFoundError(f"GNSS daily positions not found: {tenv3_path}; pass --gnss-cache-dir or --no-animation")

    frame_dates = [date.fromisoformat(frame[0]) for frame in frames]
    gnss_dates, gnss_values = _read_gnss_los_series(
        tenv3_path,
        station,
        gnss_start_date or frame_dates[0],
        frame_dates[-1],
    )
    # The station coordinates are transformed using the replayed raster CRS.
    first_raster = load_raster(Path(frames[0][1]))
    lonlat_to_stack = build_wgs84_to_raster_ct(first_raster.projection_wkt)
    station_x, station_y, _ = lonlat_to_stack.TransformPoint(station["lon"], station["lat"], 0.0)
    insar_values = [
        _snapshot_station_value(frame, station_x, station_y, sample_radius_m, insar_sign)
        for frame in frames
    ]
    del first_raster

    sampled_ranges: list[Any] = []
    for frame in frames:
        displacement, coherent = _frame_arrays(frame)
        sampled = np.abs(displacement[::8, ::8])
        sampled_valid = np.isfinite(displacement[::8, ::8]) & coherent[::8, ::8]
        if sampled_valid.any():
            sampled_ranges.append(sampled[sampled_valid])
        del displacement, coherent
    pool = np.concatenate(sampled_ranges) if sampled_ranges else np.array([10.0])
    limit = max(5.0, float(np.nanpercentile(pool, 90.0)))
    cmap = make_diverging_cmap().copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    first_displacement, first_coherent = _frame_arrays(frames[0])
    first_shown = np.where(
        np.isfinite(first_displacement) & first_coherent,
        insar_sign * first_displacement,
        np.nan,
    )

    plt = configure_matplotlib()
    fig = plt.figure(figsize=(11.8, 5.5), constrained_layout=True)
    gridspec = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_map = fig.add_subplot(gridspec[0, 0])
    ax_ts = fig.add_subplot(gridspec[0, 1])
    extent = frames[0][3]
    image = ax_map.imshow(
        first_shown,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax_map.set_title(f"Cumulative LOS displacement · {frame_dates[0].isoformat()}", loc="left", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.scatter([station_x], [station_y], marker="*", s=72, color="#1D5B8F", edgecolor="white", linewidth=0.9, zorder=6)
    ax_map.annotate(station_id, (station_x, station_y), xytext=(7, 6), textcoords="offset points", fontsize=8, fontweight="bold", color="#182230", zorder=7)
    cbar = fig.colorbar(image, ax=ax_map, shrink=0.80, pad=0.025, extend="both")
    sign_label = "toward satellite" if insar_sign < 0 else "range increase"
    cbar.set_label(f"Cumulative LOS displacement, {sign_label} (mm)")

    ax_ts.plot(gnss_dates, gnss_values, color="#1D5B8F", linewidth=1.1, alpha=0.85, label=f"GNSS {station_id}")
    insar_line, = ax_ts.plot([], [], "-o", color="#B8332A", linewidth=1.35, markersize=3.5, label="InSAR stack")
    insar_marker, = ax_ts.plot([], [], "o", color="#B8332A", markeredgecolor="white", markeredgewidth=0.65, markersize=7.0, zorder=6)
    cursor = ax_ts.axvline(frame_dates[0], color="#182230", linewidth=1.0, linestyle="--", alpha=0.75)
    ax_ts.set_title(f"GNSS validation at {station_id}", loc="left", fontsize=11, fontweight="bold")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Cumulative LOS displacement (mm)")
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_ts.tick_params(axis="x", rotation=30)
    ax_ts.grid(True, linestyle=":", linewidth=0.65, color="#D0D5DD")
    ax_ts.legend(loc="best", fontsize=8.5, framealpha=0.92)
    all_values = np.asarray(gnss_values + [v for v in insar_values if np.isfinite(v)], dtype=float)
    y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    y_pad = max(5.0, 0.08 * max(1.0, y_max - y_min))
    ax_ts.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_ts.set_xlim(gnss_dates[0], gnss_dates[-1])

    def update(index: int) -> tuple[Any, ...]:
        displacement, coherent = _frame_arrays(frames[index])
        shown = np.where(np.isfinite(displacement) & coherent, insar_sign * displacement, np.nan)
        image.set_data(shown)
        day = frame_dates[index]
        ax_map.set_title(f"Cumulative LOS displacement · {day.isoformat()}", loc="left", fontsize=11, fontweight="bold")
        cursor.set_xdata([day, day])
        visible_dates = [d for d, val in zip(frame_dates[: index + 1], insar_values[: index + 1]) if np.isfinite(val)]
        visible_values = [val for val in insar_values[: index + 1] if np.isfinite(val)]
        insar_line.set_data(visible_dates, visible_values)
        if np.isfinite(insar_values[index]):
            insar_marker.set_data([day], [insar_values[index]])
        else:
            insar_marker.set_data([], [])
        return image, cursor, insar_line, insar_marker

    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(1, fps), blit=False, repeat=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(1, fps), metadata={"title": "InSAR stack evolution with GNSS validation"})
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=max(1, fps), bitrate=1200, metadata={"title": "InSAR stack evolution with GNSS validation"})
    else:
        raise ValueError("Animation output must end in .gif or .mp4")
    movie.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", type=Path, default=_ROOT / "var/work/stacks/mexico_city_iw1", help="Existing stack with stack_ledger.json")
    parser.add_argument("--products-dir", type=Path, default=_ROOT / "var/work/products/mexico_city_iw1", help="Source pair products used to build the stack")
    parser.add_argument("--stack-bin", type=Path, default=_ROOT / "build/insar_stack", help="C++ insar_stack executable")
    parser.add_argument("--out", type=Path, default=_ROOT / "docs/figures/mexico_city_stack_evolution.png", help="Output date-by-date panel figure")
    parser.add_argument("--animation-out", type=Path, default=_ROOT / "docs/figures/mexico_city_stack_evolution.gif", help="Output synchronized animation (.gif or .mp4)")
    parser.add_argument("--no-animation", action="store_true", help="Render only the static snapshot panel")
    parser.add_argument("--gnss-cache-dir", type=Path, default=_ROOT / "var/work/gnss/unr", help="Directory with UNR station .tenv3 files")
    parser.add_argument("--validation-csv", type=Path, default=_ROOT / "data/validation/mexico_city_gnss_validation.csv", help="GNSS validation CSV containing station LOS vectors")
    parser.add_argument("--gnss-station", default="MMX1", help="GNSS station to synchronize with the stack")
    parser.add_argument("--insar-sign", type=float, default=-1.0, help="Sign used to align range-increase InSAR displacement to GNSS LOS")
    parser.add_argument("--animation-fps", type=int, default=2, help="Animation playback rate")
    parser.add_argument("--animation-dpi", type=int, default=110, help="Animation raster resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    from insar_platform.viz._common import ensure_plot_runtime

    ensure_plot_runtime(__file__, argv)
    args = build_parser().parse_args(argv)
    stack_dir = resolve_runtime_path(args.stack_dir).resolve()
    products_dir = resolve_runtime_path(args.products_dir).resolve()
    stack_bin = args.stack_bin.resolve()
    if not stack_dir.is_dir() or not (stack_dir / "stack_ledger.json").is_file():
        raise FileNotFoundError(f"Source stack ledger not found: {stack_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="insar_evolution_") as temp_name:
        temp = Path(temp_name)
        snapshots_dir = temp / "snapshots"
        cfg = _load_replay_config(stack_dir, products_dir, temp / "replay_stack", snapshots_dir, stack_bin)
        print("Replaying the stack with its recorded settings to capture corrected per-epoch COGs...")
        sync_stack(cfg)
        frames = _read_frames(snapshots_dir)
        render_evolution_panels(frames, args.out)
        if not args.no_animation:
            ledger = json.loads((stack_dir / "stack_ledger.json").read_text(encoding="utf-8"))
            pair_ids = ledger.get("pair_ids", [])
            start_date = date.fromisoformat(pair_ids[0].split("_")[0]) if pair_ids else None
            render_evolution_animation(
                frames,
                args.animation_out,
                gnss_cache_dir=args.gnss_cache_dir.resolve(),
                validation_csv=args.validation_csv.resolve(),
                station_id=args.gnss_station,
                gnss_start_date=start_date,
                insar_sign=args.insar_sign,
                fps=args.animation_fps,
                dpi=args.animation_dpi,
            )
    print(f"Saved stack evolution panels -> {args.out}")
    if not args.no_animation:
        print(f"Saved synchronized stack/GNSS animation -> {args.animation_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
