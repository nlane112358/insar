#!/usr/bin/env python3
"""Replay a stack and render evenly spaced cumulative-displacement map panels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "insar_mpl_cache"))

_PLOT_DIR = Path(__file__).resolve().parent
_ROOT = _PLOT_DIR.parents[1]
_SCRIPTS_DIR = _ROOT / "scripts"
for _path in (_PLOT_DIR, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import configure_matplotlib, load_raster, save_figure  # noqa: E402
from stack_timeseries import StackConfig, sync_stack  # noqa: E402


def _unfingerprint(value: str | None) -> Path | None:
    if not value:
        return None
    raw = value.rsplit(":", 1)[0] if ":" in value else value
    path = Path(raw)
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


def _read_frames(snapshot_dir: Path) -> list[tuple[str, Any, Any, tuple[float, float, float, float]]]:
    frames = []
    for cumulative_path in sorted(snapshot_dir.glob("*_cumulative_disp_mm.tif")):
        date = cumulative_path.name.removesuffix("_cumulative_disp_mm.tif")
        coherence_path = snapshot_dir / f"{date}_temporal_coherence.tif"
        if not coherence_path.is_file():
            continue
        cumulative = load_raster(cumulative_path)
        coherence = load_raster(coherence_path)
        frames.append((date, cumulative.data, coherence.data, cumulative.extent))
    if not frames:
        raise RuntimeError(f"No accepted primary-pair snapshots were written under {snapshot_dir}")
    return frames


def render_evolution_panels(frames: list[tuple[str, Any, Any, tuple[float, float, float, float]]], out_path: Path, *, dpi: int = 240) -> Path:
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    count = min(9, len(frames))
    indices = np.unique(np.rint(np.linspace(0, len(frames) - 1, count)).astype(int))
    chosen = [frames[i] for i in indices]
    extent = chosen[0][3]
    masked_values = []
    for _date, displacement, coherence, _extent in chosen:
        valid = np.isfinite(displacement) & np.isfinite(coherence) & (coherence >= 0.4)
        masked_values.append(np.abs(displacement[valid]))
    pooled = np.concatenate([a for a in masked_values if a.size]) if any(a.size for a in masked_values) else np.array([10.0])
    limit = max(5.0, float(np.nanpercentile(pooled, 98.0)))

    plt = configure_matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 10.4), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(chosen):
            ax.set_axis_off()
            continue
        date, displacement, coherence, frame_extent = chosen[idx]
        valid = np.isfinite(displacement) & np.isfinite(coherence) & (coherence >= 0.4)
        shown = np.where(valid, displacement, np.nan)
        ax.imshow(shown, extent=frame_extent, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(date, fontsize=10.5, fontweight="bold")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d4d9e0")
            spine.set_linewidth(0.65)

    mappable = ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r")
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=list(axes.flat), shrink=0.78, pad=0.018, fraction=0.025)
    colorbar.set_label("Cumulative LOS displacement (mm)")
    fig.suptitle("Mexico City basin deformation through the stack", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.008, f"{len(frames)} accepted pair updates · snapshots replay the stack’s recorded correction and closure settings · pixels shown at temporal coherence ≥ 0.40", ha="center", fontsize=8.5, color="#596579")
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", type=Path, default=_ROOT / "insar_work/stacks/mexico_city_iw1", help="Existing stack with stack_ledger.json")
    parser.add_argument("--products-dir", type=Path, default=_ROOT / "insar_work/products/mexico_city_iw1", help="Source pair products used to build the stack")
    parser.add_argument("--stack-bin", type=Path, default=_ROOT / "build/insar_stack", help="C++ insar_stack executable")
    parser.add_argument("--out", type=Path, default=_ROOT / "docs/figures/mexico_city_stack_evolution.png", help="Output date-by-date panel figure")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stack_dir = args.stack_dir.resolve()
    products_dir = args.products_dir.resolve()
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
    print(f"Saved stack evolution panels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
