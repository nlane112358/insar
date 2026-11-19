#!/usr/bin/env python3
"""Publication figures for UK NISAR L-band test sites (Greater London/Thames Valley & East Midlands Coalfields)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PLOT_DIR = Path(__file__).resolve().parent
_ROOT = _PLOT_DIR.parents[1]
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from _common import (  # noqa: E402
    INK,
    MUTED,
    NODATA_BG,
    add_editorial_header,
    add_footer,
    asset_color,
    asset_label,
    configure_matplotlib,
    ensure_plot_runtime,
    load_raster,
    make_coherence_cmap,
    make_diverging_cmap,
    save_figure,
    shorten_name,
)


def plot_uk_sector_dashboard(report_json: Path, out_path: Path, *, dpi: int = 230) -> Path:
    """Render a 3-panel UK Rail, Water AMP8, and Home Insurance exposure dashboard from uk_sector_analytics JSON."""
    import numpy as np
    from matplotlib.patches import Patch

    rep = json.loads(report_json.read_text(encoding="utf-8"))
    profile = rep.get("site_profile", {})
    scene = rep.get("scene_lband_metrics", {})
    sectors = rep.get("sectors", {})
    zones = rep.get("home_insurance_zones", [])
    stack_dir = Path(rep["stack_dir"])
    coh_layer = load_raster(stack_dir / "temporal_coherence.tif")
    coh_vals = coh_layer.data[np.isfinite(coh_layer.data)]

    plt = configure_matplotlib()
    fig = plt.figure(figsize=(16.2, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.25], left=0.055, right=0.975, top=0.80, bottom=0.14, wspace=0.38)
    ax_coh = fig.add_subplot(gs[0, 0])
    ax_sec = fig.add_subplot(gs[0, 1])
    ax_ins = fig.add_subplot(gs[0, 2])

    # Panel A: L-band temporal coherence distribution + sector mean markers
    ax_coh.hist(coh_vals, bins=36, range=(0.0, 1.0), density=True, color="#2E7EB0", alpha=0.30, edgecolor="#1D5B8F", linewidth=0.7)
    ax_coh.axvline(0.35, color="#B8332A", linestyle="--", linewidth=1.1, label="Stack gate (γ=0.35)")
    ax_coh.axvline(0.55, color="#13856E", linestyle=":", linewidth=1.2, label="Alert gate (γ=0.55)")
    if scene.get("mean_coherence") is not None:
        ax_coh.axvline(float(scene["mean_coherence"]), color=INK, linestyle="-", linewidth=1.3, label=f"Scene mean ({scene['mean_coherence']:.2f})")

    y_top = ax_coh.get_ylim()[1]
    ax_coh.set_ylim(0.0, y_top * 1.24)
    sector_markers = [
        ("uk_rail", "#1D5B8F", "UK Rail"),
        ("water_amp8", "#0E7490", "Water AMP8"),
    ]
    for idx, (skey, scol, slabel) in enumerate(sector_markers):
        wcoh = sectors.get(skey, {}).get("weighted_mean_coherence")
        if wcoh is not None:
            ymark = y_top * (0.98 - 0.13 * idx)
            ax_coh.scatter([wcoh], [ymark], s=64, color=scol, edgecolor="white", linewidth=0.9, zorder=5)
            ax_coh.annotate(
                f"{slabel}: {wcoh:.2f}",
                xy=(wcoh, ymark),
                xytext=(8 if wcoh < 0.68 else -8, 0),
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
                color=scol,
                ha="left" if wcoh < 0.68 else "right",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            )
    ax_coh.set_xlim(0.0, 1.0)
    ax_coh.set_xlabel("L-Band Temporal Coherence (γ)")
    ax_coh.set_ylabel("Pixel Density")
    ax_coh.set_title("A. L-Band Coherence Retention", loc="left", fontsize=11.0, fontweight="bold")
    ax_coh.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.88, edgecolor="#E0E4EA")
    ax_coh.grid(axis="y", alpha=0.45)

    # Panel B: Top UK Rail & Water AMP8 assets (preferring coherent engineered assets >= 30 px)
    combined_assets: list[dict[str, Any]] = []
    for skey in ("uk_rail", "water_amp8"):
        for item in sectors.get(skey, {}).get("top_assets", []):
            if (item.get("valid_pixels") or 0) >= 30 and (item.get("mean_coherence") or 0.0) >= 0.45:
                combined_assets.append({**item, "sector": skey})
    if not combined_assets:
        for skey in ("uk_rail", "water_amp8"):
            combined_assets.extend({**item, "sector": skey} for item in sectors.get(skey, {}).get("top_assets", [])[:5])
    combined_assets.sort(key=lambda r: -(r.get("p95_abs_velocity_mm_yr") or 0.0))
    shown_assets = combined_assets[:10][::-1]

    if shown_assets:
        y_pos = np.arange(len(shown_assets))
        vals = [float(r.get("p95_abs_velocity_mm_yr") or 0.0) for r in shown_assets]
        cols = [asset_color(str(r.get("asset_class", "infrastructure"))) for r in shown_assets]
        ax_sec.barh(y_pos, vals, color=cols, height=0.64, alpha=0.90)
        ax_sec.set_yticks(y_pos)
        ax_sec.set_yticklabels([shorten_name(str(r.get("name", "")), 24) for r in shown_assets], fontsize=8.0)
        xmax_b = max(vals) * 1.42 if vals else 50.0
        ax_sec.set_xlim(0.0, max(20.0, xmax_b))
        for yi, r, v in zip(y_pos, shown_assets, vals):
            coh_txt = f"γ={r['mean_coherence']:.2f}" if r.get("mean_coherence") is not None else ""
            cls_txt = asset_label(str(r.get("asset_class", "")))
            ax_sec.text(v + xmax_b * 0.015, yi, f"{v:.0f} mm/yr · {cls_txt} ({coh_txt})", va="center", ha="left", fontsize=7.2, color=INK)
    ax_sec.set_xlabel("p95 |LOS Velocity| (mm/yr, annualized)")
    ax_sec.set_title("B. UK Rail & AMP8 Water Utility Assets", loc="left", fontsize=11.0, fontweight="bold")
    ax_sec.grid(axis="x", alpha=0.45)

    # Panel C: Home Insurance ONS District Differential Motion & Uplift/Settlement Split
    substantive_zones = [z for z in zones if int(z.get("valid_pixels") or 0) >= 300] or zones
    top_zones = substantive_zones[:10][::-1]
    if top_zones:
        y_z = np.arange(len(top_zones))
        spreads = [float(z["p95_p5_spread_mm"]) for z in top_zones]
        uplift = [float(z["uplift_pct"]) for z in top_zones]
        settle = [float(z["settlement_pct"]) for z in top_zones]
        u_bars = [sp * (u / max(u + s, 1e-6)) for sp, u, s in zip(spreads, uplift, settle)]
        s_bars = [sp * (s / max(u + s, 1e-6)) for sp, u, s in zip(spreads, uplift, settle)]
        ax_ins.barh(y_z, u_bars, color="#2E7EB0", height=0.62, label="Uplift / heave (range -)")
        ax_ins.barh(y_z, s_bars, left=u_bars, color="#D45D35", height=0.62, label="Settlement (range +)")
        ax_ins.set_yticks(y_z)
        ax_ins.set_yticklabels([shorten_name(str(z["name"]), 22) for z in top_zones], fontsize=8.0)
        xmax_c = max(spreads) * 1.35 if spreads else 10.0
        ax_ins.set_xlim(0.0, max(5.0, xmax_c))
        for yi, z, sp in zip(y_z, top_zones, spreads):
            ax_ins.text(sp + xmax_c * 0.015, yi, f"{sp:.1f} mm · γ={z['mean_coherence']:.2f}", va="center", ha="left", fontsize=7.2, color=INK)
        ax_ins.legend(loc="lower right", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.88, edgecolor="#E0E4EA")
    ax_ins.set_xlabel("Intra-District Differential Spread (p95 − p5, mm)")
    ax_ins.set_title("C. Home Insurance District Exposure", loc="left", fontsize=11.0, fontweight="bold")
    ax_ins.grid(axis="x", alpha=0.45)

    mature_txt = "MATURE STACK" if scene.get("stack_mature") else f"BASELINE / MATURITY-GATED ({scene.get('actual_span_days', 0):.0f}d < {scene.get('min_alert_span_days', 60):.0f}d)"
    title = f"{profile.get('display_name', 'UK NISAR Site')} — Sector & Insurance Exposure"
    subtitle = f"{profile.get('phenomenology', 'NISAR L-band')}  ·  {rep.get('stream_key', '')}  ·  {mature_txt}"
    add_editorial_header(fig, title, subtitle, left=0.055, top=0.955, title_fontsize=14.0, subtitle_gap=0.045)

    ref_note = profile.get("reference_geology", "Bedrock reference pinned")
    footer = (
        f"Reference datum: {ref_note}  ·  "
        f"Coherent footprint (γ≥0.35): {scene.get('coherent_area_km2', 0):.1f} km² ({scene.get('coherent_fraction_pct', 0):.1f}%)  ·  "
        f"Boundaries: ONS UK LAD (May 2024 BGC) & OpenStreetMap"
    )
    add_footer(fig, footer, left=0.055, bottom=0.022, fontsize=7.9)

    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved UK sector dashboard -> {out_path}")
    return out_path


def plot_uk_triplet_closure(
    stack_dir: Path,
    products_dir: Path,
    out_path: Path,
    *,
    dpi: int = 230,
) -> Path:
    """Render 3-pass NISAR triplet displacement panels + triplet phase closure residual map."""
    import numpy as np

    ledger = json.loads((stack_dir / "stack_ledger.json").read_text(encoding="utf-8"))
    summary = json.loads((stack_dir / "stack_summary.json").read_text(encoding="utf-8"))
    stream_key = str(ledger.get("stream_key", ""))
    all_pids = list(ledger.get("pair_ids", [])) + list(ledger.get("redundant_pair_ids", []))
    if len(all_pids) < 3:
        raise ValueError(f"Expected >= 3 pairs in {stack_dir} for triplet closure plot, got {len(all_pids)}")

    pair_layers = []
    for pid in all_pids[:3]:
        pdir = products_dir / f"{stream_key}_{pid}"
        disp = load_raster(pdir / "displacement_mm.tif")
        coh = load_raster(pdir / "coherence.tif")
        pair_layers.append((pid.replace("_", " to "), disp, coh))

    closure_p = stack_dir / "closure_residual_mm.tif"
    if closure_p.is_file():
        clos_layer = load_raster(closure_p)
        clos_arr = clos_layer.data
    else:
        clos_arr = pair_layers[0][1].data + pair_layers[1][1].data - pair_layers[2][1].data

    ref_coh = pair_layers[0][2].data
    valid_c = np.isfinite(clos_arr) & np.isfinite(ref_coh) & (ref_coh >= 0.35)
    vals = np.concatenate([
        np.abs(d.data[np.isfinite(d.data) & (c.data >= 0.40)])
        for _, d, c in pair_layers
    ])
    dlim = max(10.0, float(np.nanpercentile(vals, 96.0))) if vals.size else 20.0
    clim = max(10.0, float(np.nanpercentile(np.abs(clos_arr[valid_c]), 97.0))) if int(valid_c.sum()) else 15.0

    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.4), gridspec_kw={"left": 0.045, "right": 0.96, "top": 0.80, "bottom": 0.17, "wspace": 0.16})
    cmap_div = make_diverging_cmap()

    for ax, (label, disp, coh) in zip(axes[:3], pair_layers):
        arr = np.where(np.isfinite(disp.data) & (coh.data >= 0.35), disp.data, np.nan)
        ax.set_facecolor(NODATA_BG)
        im_d = ax.imshow(arr, extent=disp.extent, origin="upper", cmap=cmap_div, vmin=-dlim, vmax=dlim, interpolation="nearest")
        ax.set_title(label, fontsize=9.8, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_c = axes[3]
    ax_c.set_facecolor(NODATA_BG)
    clos_plot = np.where(valid_c, clos_arr, np.nan)
    im_c = ax_c.imshow(clos_plot, extent=pair_layers[0][1].extent, origin="upper", cmap="RdBu_r", vmin=-clim, vmax=clim, interpolation="nearest")
    rms_mm = summary.get("triplet_closure", {}).get("rms_mm")
    rms_txt = f"RMS = {rms_mm:.2f} mm" if rms_mm is not None else f"RMS = {float(np.sqrt(np.mean(clos_arr[valid_c]**2))):.2f} mm"
    ax_c.set_title(f"Closure: AB + BC − AC\n({rms_txt})", fontsize=9.8, fontweight="bold")
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    cb1 = fig.colorbar(im_d, ax=list(axes[:3]), orientation="horizontal", fraction=0.055, pad=0.08, aspect=42)
    cb1.set_label("Pair LOS Displacement (mm) · ±p96 stretch")
    cb2 = fig.colorbar(im_c, ax=axes[3], orientation="horizontal", fraction=0.055, pad=0.08, aspect=16)
    cb2.set_label("Closure Residual (mm)")

    tc = summary.get("triplet_closure", {})
    add_editorial_header(
        fig,
        f"East London & Thames Estuary — 3-Pass NISAR L-Band Triplet Closure ({stream_key})",
        f"Closed 12-day + 12-day vs 24-day loop (24 Oct – 17 Nov 2026)  ·  Bad fraction = {tc.get('bad_fraction', 0.0):.3f}  ·  Rejected = {tc.get('pair_rejected', False)}",
        left=0.045,
        top=0.955,
        title_fontsize=13.5,
        subtitle_gap=0.048,
    )
    thresh_mm = tc.get("thresh_mm", 25.0)
    add_footer(
        fig,
        f"Zero-referenced to Dartford Chalk bedrock (0.215°E, 51.442°N)  ·  Pixels exceeding ±{thresh_mm:.0f} mm closure masked prior to WLS stack update",
        left=0.045,
        bottom=0.025,
    )
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved UK triplet closure figure -> {out_path}")
    return out_path


def plot_uk_track_comparison(
    stack_a_dir: Path,
    stack_b_dir: Path,
    out_path: Path,
    *,
    label_a: str,
    label_b: str,
    suptitle: str,
    dpi: int = 230,
) -> Path:
    """Compare displacement and coherence across two overlapping UK NISAR tracks (Asc vs Desc)."""
    import numpy as np
    from matplotlib.colors import Normalize

    a_disp = load_raster(stack_a_dir / "cumulative_disp_mm.tif")
    a_coh = load_raster(stack_a_dir / "temporal_coherence.tif")
    b_disp = load_raster(stack_b_dir / "cumulative_disp_mm.tif")
    b_coh = load_raster(stack_b_dir / "temporal_coherence.tif")

    vals = np.concatenate([
        np.abs(a_disp.data[np.isfinite(a_disp.data) & (a_coh.data >= 0.35)]),
        np.abs(b_disp.data[np.isfinite(b_disp.data) & (b_coh.data >= 0.35)]),
    ])
    dlim = max(10.0, float(np.nanpercentile(vals, 96.0))) if vals.size else 20.0

    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), constrained_layout=True)
    cmap_div = make_diverging_cmap()
    cmap_coh = make_coherence_cmap()

    cases = [
        (axes[0, 0], a_disp, a_coh, f"{label_a} · LOS displacement", True),
        (axes[0, 1], a_coh, a_coh, f"{label_a} · L-band coherence", False),
        (axes[1, 0], b_disp, b_coh, f"{label_b} · LOS displacement", True),
        (axes[1, 1], b_coh, b_coh, f"{label_b} · L-band coherence", False),
    ]
    im_d = im_c = None
    for ax, layer, coh_l, title, is_disp in cases:
        ax.set_facecolor(NODATA_BG)
        if is_disp:
            arr = np.where(np.isfinite(layer.data) & (coh_l.data >= 0.35), layer.data, np.nan)
            im_d = ax.imshow(arr, extent=layer.extent, origin="upper", cmap=cmap_div, vmin=-dlim, vmax=dlim, interpolation="nearest")
        else:
            im_c = ax.imshow(layer.data, extent=layer.extent, origin="upper", cmap=cmap_coh, norm=Normalize(0.0, 1.0), interpolation="nearest")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("UTM 30N Easting (m)")
        ax.set_ylabel("UTM 30N Northing (m)")
        ax.set_aspect("equal", adjustable="box")

    if im_d is not None:
        cb_d = fig.colorbar(im_d, ax=axes[:, 0], shrink=0.84, pad=0.03)
        cb_d.set_label("LOS displacement (mm)")
    if im_c is not None:
        cb_c = fig.colorbar(im_c, ax=axes[:, 1], shrink=0.84, pad=0.03)
        cb_c.set_label("Temporal coherence (γ)")
    fig.suptitle(suptitle, fontsize=13.5, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved UK track comparison -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ensure_plot_runtime(__file__, argv)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dash = sub.add_parser("dashboard", help="Render 3-panel UK sector & home insurance dashboard")
    p_dash.add_argument("--report-json", required=True, type=Path)
    p_dash.add_argument("--out", required=True, type=Path)

    p_trip = sub.add_parser("triplet", help="Render 3-pass NISAR triplet closure figure")
    p_trip.add_argument("--stack-dir", required=True, type=Path)
    p_trip.add_argument("--products-dir", required=True, type=Path)
    p_trip.add_argument("--out", required=True, type=Path)

    p_cmp = sub.add_parser("compare", help="Render Asc/Desc two-track UK comparison figure")
    p_cmp.add_argument("--stack-a", required=True, type=Path)
    p_cmp.add_argument("--stack-b", required=True, type=Path)
    p_cmp.add_argument("--label-a", required=True)
    p_cmp.add_argument("--label-b", required=True)
    p_cmp.add_argument("--suptitle", required=True)
    p_cmp.add_argument("--out", required=True, type=Path)

    args = p.parse_args(argv)
    if args.cmd == "dashboard":
        plot_uk_sector_dashboard(args.report_json, args.out)
    elif args.cmd == "triplet":
        plot_uk_triplet_closure(args.stack_dir, args.products_dir, args.out)
    elif args.cmd == "compare":
        plot_uk_track_comparison(
            args.stack_a,
            args.stack_b,
            args.out,
            label_a=args.label_a,
            label_b=args.label_b,
            suptitle=args.suptitle,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
