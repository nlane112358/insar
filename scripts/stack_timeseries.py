#!/usr/bin/env python3
"""
Sequential O(1) InSAR stack orchestrator.

Discovers chronological consecutive interferogram pair directories emitted by
`insar_proto` / `auto_ingest.py`, invokes `./build/insar_stack` strictly on
newly added pairs to update persistent sufficient statistics (`stack_state.tif`),
maintains `stack_ledger.json` and `corridor_timeseries.json`, and prints
prioritized infrastructure corridor risk summaries.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PAIR_DIR_RE = re.compile(r"^(?P<stream>.+)_(?P<ref>\d{4}-\d{2}-\d{2})_(?P<sec>\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class PairEntry:
    stream_key: str
    ref_date: str
    sec_date: str
    pair_dir: Path
    disp_path: Path
    coh_path: Path

    @property
    def pair_id(self) -> str:
        return f"{self.ref_date}_{self.sec_date}"


@dataclass(frozen=True)
class StackConfig:
    products_dir: Path
    out_dir: Path
    stack_bin: Path = Path("./build/insar_stack")
    stream_key: str | None = None
    ramp_order: int = 1
    min_coh: float = 0.35
    corridors: Path | None = None
    buffer_m: float = 100.0
    vel_alert_mm_yr: float = 15.0
    accel_alert_mm_yr2: float = 25.0
    min_alert_coh: float = 0.55
    reset: bool = False


@dataclass(frozen=True)
class StackRunSummary:
    stream_key: str
    total_pairs_in_chain: int
    newly_stacked: int
    skipped_existing: int
    corridors_evaluated: int
    critical_corridors: int
    warning_corridors: int
    corridor_alerts: int
    rebuilt: bool


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def discover_pair_chain(
    products_dir: Path,
    stream_key: str | None = None,
) -> tuple[str, list[PairEntry]]:
    """Scan products_dir and construct a non-overlapping chronological pair chain."""
    if not products_dir.exists():
        raise FileNotFoundError(f"Products directory does not exist: {products_dir}")

    candidates_by_stream: dict[str, list[PairEntry]] = {}
    for entry in sorted(products_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = PAIR_DIR_RE.match(entry.name)
        if not m:
            continue
        disp = entry / "displacement_mm.tif"
        coh = entry / "coherence.tif"
        if not (disp.is_file() and coh.is_file()):
            continue
        sk = m.group("stream")
        if stream_key is not None and sk != stream_key:
            continue
        ref_d, sec_d = m.group("ref"), m.group("sec")
        if _parse_date(sec_d) <= _parse_date(ref_d):
            continue
        candidates_by_stream.setdefault(sk, []).append(
            PairEntry(
                stream_key=sk,
                ref_date=ref_d,
                sec_date=sec_d,
                pair_dir=entry,
                disp_path=disp,
                coh_path=coh,
            )
        )

    if not candidates_by_stream:
        hint = f" for stream '{stream_key}'" if stream_key else ""
        raise RuntimeError(f"No valid displacement_mm.tif/coherence.tif pairs found in {products_dir}{hint}")

    if stream_key is None:
        if len(candidates_by_stream) > 1:
            # Prefer stream with most pairs, deterministic tie-break by name
            selected_stream = sorted(
                candidates_by_stream.keys(),
                key=lambda k: (-len(candidates_by_stream[k]), k),
            )[0]
            print(
                f"Multiple streams found in {products_dir} ({sorted(candidates_by_stream.keys())}); "
                f"selecting '{selected_stream}'. Pass --stream-key to override."
            )
        else:
            selected_stream = next(iter(candidates_by_stream.keys()))
    else:
        selected_stream = stream_key

    raw_pairs = sorted(
        candidates_by_stream[selected_stream],
        key=lambda p: (p.ref_date, p.sec_date),
    )

    chain: list[PairEntry] = []
    cursor_end: str | None = None
    # Group by ref_date and greedily advance earliest non-overlapping chain
    i = 0
    while i < len(raw_pairs):
        p = raw_pairs[i]
        if cursor_end is not None and p.ref_date < cursor_end:
            i += 1
            continue
        # Among pairs starting at same ref_date, prefer shortest step (sec_date smallest)
        same_ref = [x for x in raw_pairs[i:] if x.ref_date == p.ref_date]
        chosen = sorted(same_ref, key=lambda x: x.sec_date)[0]
        chain.append(chosen)
        cursor_end = chosen.sec_date
        while i < len(raw_pairs) and raw_pairs[i].ref_date == chosen.ref_date:
            i += 1

    return selected_stream, chain


def _file_fingerprint(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    if resolved.is_file():
        h = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
        return f"{resolved}:{h}"
    return str(resolved)


def _stack_params_key(cfg: StackConfig) -> dict[str, Any]:
    return {
        "ramp_order": cfg.ramp_order,
        "min_coh": round(cfg.min_coh, 6),
    }


def _export_params_key(cfg: StackConfig) -> dict[str, Any]:
    return {
        "corridors": _file_fingerprint(cfg.corridors),
        "buffer_m": round(cfg.buffer_m, 3),
        "vel_alert_mm_yr": round(cfg.vel_alert_mm_yr, 4),
        "accel_alert_mm_yr2": round(cfg.accel_alert_mm_yr2, 4),
        "min_alert_coh": round(cfg.min_alert_coh, 4),
    }


def _run_insar_stack(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"insar_stack failed: {err}")


def _update_corridor_timeseries(
    ts_data: dict[str, Any],
    origin_date: date,
    pair: PairEntry,
    corridor_report_path: Path,
) -> None:
    if not corridor_report_path.is_file():
        return
    report = json.loads(corridor_report_path.read_text(encoding="utf-8"))
    ref_days = float((_parse_date(pair.ref_date) - origin_date).days)
    sec_days = float((_parse_date(pair.sec_date) - origin_date).days)

    epochs: list[str] = ts_data.setdefault("epochs", [pair.ref_date])
    if pair.sec_date not in epochs:
        epochs.append(pair.sec_date)

    corridors_map: dict[str, Any] = ts_data.setdefault("corridors", {})
    for feat in report.get("features", []):
        props = feat.get("properties", {})
        aid = props.get("asset_id")
        if not aid:
            continue
        entry = corridors_map.setdefault(
            aid,
            {
                "asset_id": aid,
                "name": props.get("name", aid),
                "asset_class": props.get("asset_class", "infrastructure"),
                "points": [
                    {
                        "date": origin_date.isoformat(),
                        "days": 0.0,
                        "cumulative_disp_mm": 0.0,
                        "interval_disp_mm": 0.0,
                        "coh": None,
                        "mean_vel_mm_yr": None,
                        "p95_abs_vel_mm_yr": None,
                        "risk_level": "baseline",
                    }
                ]
                if ref_days == 0.0
                else [],
            },
        )
        entry["latest_risk_level"] = props.get("risk_level")
        entry["latest_mean_velocity_mm_yr"] = props.get("mean_velocity_mm_yr")
        entry["latest_p95_abs_velocity_mm_yr"] = props.get("p95_abs_velocity_mm_yr")
        entry["latest_mean_coherence"] = props.get("mean_coherence")
        entry["points"].append(
            {
                "date": pair.sec_date,
                "days": sec_days,
                "cumulative_disp_mm": props.get("mean_cumulative_disp_mm"),
                "interval_disp_mm": props.get("latest_interval_disp_mm"),
                "coh": props.get("mean_coherence"),
                "mean_vel_mm_yr": props.get("mean_velocity_mm_yr"),
                "p95_abs_vel_mm_yr": props.get("p95_abs_velocity_mm_yr"),
                "risk_level": props.get("risk_level"),
            }
        )


def sync_stack(cfg: StackConfig) -> StackRunSummary:
    """Incrementally update stack_state.tif and corridor analytics for newly discovered pairs."""
    if not cfg.stack_bin.is_file():
        raise FileNotFoundError(
            f"C++ stack binary not found at {cfg.stack_bin}. Build with `cmake --build build -j`."
        )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.out_dir / "stack_state.tif"
    ledger_path = cfg.out_dir / "stack_ledger.json"
    ts_path = cfg.out_dir / "corridor_timeseries.json"
    corridor_report_path = cfg.out_dir / "corridor_report.geojson"
    summary_path = cfg.out_dir / "stack_summary.json"

    stream_key, chain = discover_pair_chain(cfg.products_dir, cfg.stream_key)
    chain_ids = [p.pair_id for p in chain]

    rebuilt = False
    ledger: dict[str, Any] = {}
    if ledger_path.is_file() and state_path.is_file() and not cfg.reset:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    old_pair_ids: list[str] = ledger.get("pair_ids", [])
    stack_params = _stack_params_key(cfg)
    export_params = _export_params_key(cfg)

    is_clean_prefix = (
        bool(old_pair_ids)
        and ledger.get("stream_key") == stream_key
        and ledger.get("stack_params") == stack_params
        and len(old_pair_ids) <= len(chain_ids)
        and chain_ids[: len(old_pair_ids)] == old_pair_ids
    )

    if not is_clean_prefix:
        rebuilt = bool(old_pair_ids) or cfg.reset
        for p_clean in (state_path, ledger_path, ts_path):
            if p_clean.exists():
                p_clean.unlink()
        old_pair_ids = []
        origin_str = chain[0].ref_date
        ts_data: dict[str, Any] = {
            "stream_key": stream_key,
            "origin_date": origin_str,
            "epochs": [origin_str],
            "corridors": {},
        }
    else:
        origin_str = ledger.get("origin_date", chain[0].ref_date)
        if ts_path.is_file():
            ts_data = json.loads(ts_path.read_text(encoding="utf-8"))
        else:
            ts_data = {
                "stream_key": stream_key,
                "origin_date": origin_str,
                "epochs": [origin_str],
                "corridors": {},
            }

    origin_date = _parse_date(origin_str)
    pending = chain[len(old_pair_ids) :]

    common_args: list[str] = [
        str(cfg.stack_bin),
        "--state",
        str(state_path),
        "--out-dir",
        str(cfg.out_dir),
        "--ramp-order",
        str(cfg.ramp_order),
        "--min-coh",
        f"{cfg.min_coh:.4f}",
        "--buffer-m",
        f"{cfg.buffer_m:.2f}",
        "--vel-alert-mm-yr",
        f"{cfg.vel_alert_mm_yr:.3f}",
        "--accel-alert-mm-yr2",
        f"{cfg.accel_alert_mm_yr2:.3f}",
        "--min-alert-coh",
        f"{cfg.min_alert_coh:.4f}",
    ]
    if cfg.corridors is not None:
        common_args.extend(["--corridors", str(cfg.corridors)])

    for idx, pair in enumerate(pending):
        t0_days = float((_parse_date(pair.ref_date) - origin_date).days)
        t1_days = float((_parse_date(pair.sec_date) - origin_date).days)
        is_last = idx == len(pending) - 1

        cmd = list(common_args) + [
            "--disp",
            str(pair.disp_path),
            "--coh",
            str(pair.coh_path),
            "--t0-days",
            f"{t0_days:.6f}",
            "--t1-days",
            f"{t1_days:.6f}",
        ]
        if not is_last:
            if cfg.corridors is not None:
                cmd.append("--skip-cog")
            else:
                cmd.append("--no-export")

        print(f"[{stream_key}] Stacking pair {pair.pair_id} (t={t0_days:.0f}..{t1_days:.0f}d)...")
        _run_insar_stack(cmd)
        if cfg.corridors is not None:
            _update_corridor_timeseries(ts_data, origin_date, pair, corridor_report_path)

    # Refresh export cleanly if no new pairs arrived but corridor/alert params changed
    export_changed = ledger.get("export_params") != export_params
    if not pending and export_changed and state_path.is_file():
        print(f"[{stream_key}] Re-exporting stack & corridors (export configuration updated)...")
        _run_insar_stack(list(common_args))

    if cfg.corridors is not None and (pending or not ts_path.is_file()):
        ts_path.write_text(json.dumps(ts_data, indent=2) + "\n", encoding="utf-8")

    updated_ledger = {
        "stream_key": stream_key,
        "origin_date": origin_str,
        "pair_ids": chain_ids,
        "stack_params": stack_params,
        "export_params": export_params,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ledger_path.write_text(json.dumps(updated_ledger, indent=2) + "\n", encoding="utf-8")

    summary_json = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    corr_meta = summary_json.get("corridors", {})

    return StackRunSummary(
        stream_key=stream_key,
        total_pairs_in_chain=len(chain),
        newly_stacked=len(pending),
        skipped_existing=len(old_pair_ids),
        corridors_evaluated=int(corr_meta.get("total_features", 0)),
        critical_corridors=int(corr_meta.get("critical_count", 0)),
        warning_corridors=int(corr_meta.get("warning_count", 0)),
        corridor_alerts=int(corr_meta.get("corridor_alerts", 0)),
        rebuilt=rebuilt,
    )


def print_corridor_leaderboard(out_dir: Path, top_n: int = 10) -> None:
    report_path = out_dir / "corridor_report.geojson"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    feats = [f["properties"] for f in report.get("features", [])]
    covered = [p for p in feats if (p.get("valid_pixels") or 0) > 0]
    covered.sort(
        key=lambda p: (
            -(p.get("p95_abs_velocity_mm_yr") or 0.0),
            -(p.get("alert_pixel_count") or 0),
        )
    )
    if not covered:
        print("No corridors overlapped valid stack pixels.")
        return
    print(f"\nTop {min(top_n, len(covered))} infrastructure corridors (sorted by p95 |velocity|):")
    for p in covered[:top_n]:
        p95 = p.get("p95_abs_velocity_mm_yr")
        mean_v = p.get("mean_velocity_mm_yr")
        coh = p.get("mean_coherence")
        acc = p.get("mean_acceleration_mm_yr2")
        acc_str = f", acc={acc:+.1f} mm/yr^2" if acc is not None else ""
        print(
            f"  [{p.get('risk_level', 'unknown').upper():8s}] {p.get('name')} ({p.get('asset_class')}): "
            f"mean={mean_v:+.1f} mm/yr, p95={p95:.1f} mm/yr{acc_str}, "
            f"coh={coh:.2f}, px={p.get('valid_pixels')}, alerts={p.get('alert_pixel_count')}"
        )


def _load_plot_libs() -> tuple[Any, Any, Any, Any, Any] | None:
    try:
        from osgeo import gdal, ogr, osr  # type: ignore
        import numpy as np  # type: ignore
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        return gdal, ogr, osr, np, plt
    except ImportError:
        return None


def _reproject_geom_rings(geom_dict: dict[str, Any], ct: Any) -> list[list[tuple[float, float]]]:
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates", [])
    rings: list[list[tuple[float, float]]] = []

    def _xform_pts(pts: list[list[float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in pts:
            x, y, _ = ct.TransformPoint(float(p[0]), float(p[1]), 0.0)
            out.append((x, y))
        return out

    if gtype == "LineString":
        rings.append(_xform_pts(coords))
    elif gtype == "MultiLineString":
        for line in coords:
            rings.append(_xform_pts(line))
    elif gtype == "Polygon" and coords:
        rings.append(_xform_pts(coords[0]))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append(_xform_pts(poly[0]))
    return rings


def generate_stack_plots(out_dir: Path, plot_path: Path | None = None) -> Path:
    """Render 2x2 stack preview dashboard (coherence, cum disp, velocity+corridors, time series)."""
    target = plot_path or (out_dir / "stack_preview.png")
    libs = _load_plot_libs()
    if libs is None:
        import os
        import tempfile
        hb_py = Path("/opt/homebrew/bin/python3")
        alt_py = hb_py if hb_py.is_file() else Path(shutil.which("python3") or "")
        if alt_py.is_file() and os.path.abspath(sys.executable) != os.path.abspath(str(alt_py)):
            clean_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            clean_env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))
            subprocess.run(
                [str(alt_py), str(Path(__file__).resolve()), "--plot-only", "--out-dir", str(out_dir), "--plot-out", str(target)],
                env=clean_env,
                check=True,
            )
            return target
        raise RuntimeError(
            "Plotting requires GDAL (osgeo) + matplotlib; run with /opt/homebrew/bin/python3"
        )

    gdal, _ogr, osr, np, plt = libs
    gdal.UseExceptions()

    def _read_tif(name: str) -> tuple[Any, list[float], str]:
        ds = gdal.Open(str(out_dir / name))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        gt = list(ds.GetGeoTransform())
        proj = ds.GetProjection() or ""
        ds = None
        return arr, gt, proj

    coh, gt, proj = _read_tif("temporal_coherence.tif")
    cum, _, _ = _read_tif("cumulative_disp_mm.tif")
    vel, _, _ = _read_tif("velocity_mm_yr.tif")
    h, w = vel.shape
    extent = [gt[0], gt[0] + w * gt[1], gt[3] + h * gt[5], gt[3]]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    summary_p = out_dir / "stack_summary.json"
    summary = json.loads(summary_p.read_text(encoding="utf-8")) if summary_p.is_file() else {}
    pairs = summary.get("pair_count", "?")
    span_d = summary.get("total_span_days", "?")
    fig.suptitle(
        f"Sequential InSAR Stack Summary ({pairs} pairs, {span_d} days) — {out_dir.name}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Temporal coherence
    ax0 = axes[0, 0]
    im0 = ax0.imshow(coh, cmap="gray", vmin=0.0, vmax=1.0, extent=extent, origin="upper", interpolation="nearest")
    ax0.set_title("Mean Temporal Coherence")
    ax0.set_xlabel("Easting (m)")
    ax0.set_ylabel("Northing (m)")
    fig.colorbar(im0, ax=ax0, shrink=0.82)

    # Panel 2: Cumulative LOS displacement
    ax1 = axes[0, 1]
    fin_cum = cum[np.isfinite(cum)]
    c_lim = float(np.nanpercentile(np.abs(fin_cum), 98)) if fin_cum.size else 10.0
    c_lim = max(c_lim, 1.0)
    im1 = ax1.imshow(cum, cmap="RdBu_r", vmin=-c_lim, vmax=c_lim, extent=extent, origin="upper", interpolation="nearest")
    ax1.set_title("Cumulative LOS Displacement (mm)")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    fig.colorbar(im1, ax=ax1, shrink=0.82)

    # Panel 3: Velocity + Infrastructure corridors + corridor alerts
    ax2 = axes[1, 0]
    fin_vel = vel[np.isfinite(vel)]
    v_lim = float(np.nanpercentile(np.abs(fin_vel), 98)) if fin_vel.size else 20.0
    v_lim = max(v_lim, 5.0)
    im2 = ax2.imshow(vel, cmap="RdBu_r", vmin=-v_lim, vmax=v_lim, extent=extent, origin="upper", interpolation="nearest")
    ax2.set_title("LOS Velocity (mm/yr) + Pinned Infrastructure Corridors")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    fig.colorbar(im2, ax=ax2, shrink=0.82)

    ct_wgs_to_utm = None
    if proj:
        src = osr.SpatialReference()
        src.ImportFromEPSG(4326)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromWkt(proj)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct_wgs_to_utm = osr.CoordinateTransformation(src, dst)

    risk_colors = {"critical": "#ff006e", "warning": "#ff7b00", "watch": "#ffd166", "nominal": "#06d6a0"}
    rep_p = out_dir / "corridor_report.geojson"
    if rep_p.is_file() and ct_wgs_to_utm is not None:
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        for feat in rep.get("features", []):
            props = feat.get("properties", {})
            if (props.get("valid_pixels") or 0) <= 0:
                continue
            col = risk_colors.get(props.get("risk_level", ""), "#aaaaaa")
            lw = 1.2 if props.get("risk_level") in ("critical", "warning") else 0.7
            for ring in _reproject_geom_rings(feat.get("geometry", {}), ct_wgs_to_utm):
                if len(ring) >= 2:
                    xs, ys = zip(*ring)
                    ax2.plot(xs, ys, color=col, linewidth=lw, alpha=0.8)

    alerts_p = out_dir / "corridor_alerts.geojson"
    if alerts_p.is_file() and ct_wgs_to_utm is not None:
        af = json.loads(alerts_p.read_text(encoding="utf-8")).get("features", [])
        ax_pts, ay_pts = [], []
        for feat in af[:60]:
            c = feat.get("geometry", {}).get("coordinates", [])
            if len(c) >= 2:
                ux, uy, _ = ct_wgs_to_utm.TransformPoint(float(c[0]), float(c[1]), 0.0)
                ax_pts.append(ux)
                ay_pts.append(uy)
        if ax_pts:
            ax2.scatter(ax_pts, ay_pts, s=36, facecolors="none", edgecolors="lime", linewidths=1.3, label=f"Corridor Alerts ({len(af)})")
            ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])

    # Panel 4: Top Corridor Time Series or Bar Comparison
    ax3 = axes[1, 1]
    ts_p = out_dir / "corridor_timeseries.json"
    plotted_ts = False
    if ts_p.is_file():
        ts_data = json.loads(ts_p.read_text(encoding="utf-8"))
        corr_list = [
            c for c in ts_data.get("corridors", {}).values()
            if c.get("latest_p95_abs_velocity_mm_yr") is not None and len(c.get("points", [])) >= 2
        ]
        corr_list.sort(key=lambda c: -(c.get("latest_p95_abs_velocity_mm_yr") or 0.0))
        if len(ts_data.get("epochs", [])) >= 3 and corr_list:
            for c in corr_list[:6]:
                days = [pt["days"] for pt in c["points"] if pt.get("cumulative_disp_mm") is not None]
                vals = [pt["cumulative_disp_mm"] for pt in c["points"] if pt.get("cumulative_disp_mm") is not None]
                if len(days) >= 2:
                    lbl = f"{c['name'][:26]} ({c['latest_mean_velocity_mm_yr']:+.0f} mm/yr)"
                    ax3.plot(days, vals, marker="o", markersize=3.5, linewidth=1.6, label=lbl)
            ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax3.set_title("Top Corridor Cumulative Displacement Curves")
            ax3.set_xlabel("Days Since First Acquisition")
            ax3.set_ylabel("Mean Corridor LOS Displacement (mm)")
            ax3.grid(alpha=0.3)
            ax3.legend(fontsize=8, loc="best")
            plotted_ts = True
        elif corr_list:
            top = corr_list[:8][::-1]
            names = [f"{c['name'][:24]} ({c['latest_risk_level']})" for c in top]
            p95s = [float(c["latest_p95_abs_velocity_mm_yr"]) for c in top]
            cols = [risk_colors.get(c.get("latest_risk_level", ""), "#6c757d") for c in top]
            ax3.barh(names, p95s, color=cols)
            ax3.set_title("Top Covered Corridors — p95 |LOS Velocity| (mm/yr)")
            ax3.set_xlabel("p95 |Velocity| (mm/yr)")
            ax3.grid(axis="x", alpha=0.3)
            plotted_ts = True
    if not plotted_ts:
        ax3.text(0.5, 0.5, "No corridor time-series data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(f"Saved stack preview plot to {target}")
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequential O(1) InSAR stack and corridor analytics orchestrator.")
    p.add_argument("--products-dir", type=Path, default=None, help="Directory with pair subdirs (*_YYYY-MM-DD_YYYY-MM-DD)")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for stack_state.tif, COGs, corridor JSONs")
    p.add_argument("--stack-bin", type=Path, default=Path("./build/insar_stack"), help="Path to C++ insar_stack binary")
    p.add_argument("--stream-key", default=None, help="Filter specific stream_key prefix")
    p.add_argument("--ramp-order", type=int, default=1, choices=(-1, 0, 1, 2), help="Ramp removal polynomial order")
    p.add_argument("--min-coh", type=float, default=0.35, help="Minimum coherence for ramp fit & stack inclusion")
    p.add_argument("--corridors", type=Path, default=None, help="Vector GeoJSON/GPKG of infrastructure corridors")
    p.add_argument("--buffer-m", type=float, default=100.0, help="Metric corridor buffer width (m)")
    p.add_argument("--vel-alert-mm-yr", type=float, default=15.0, help="|Velocity| alert threshold (mm/yr)")
    p.add_argument("--accel-alert-mm-yr2", type=float, default=25.0, help="|Acceleration| alert threshold (mm/yr^2)")
    p.add_argument("--min-alert-coh", type=float, default=0.55, help="Minimum temporal coherence for alerts")
    p.add_argument("--reset", action="store_true", help="Reset state/ledger and rebuild full stack cleanly")
    p.add_argument("--plot", action="store_true", help="Render stack_preview.png dashboard after updating")
    p.add_argument("--plot-only", action="store_true", help="Render plot from an existing --out-dir stack")
    p.add_argument("--plot-out", type=Path, default=None, help="Custom output PNG path for --plot / --plot-only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.plot_only:
        generate_stack_plots(args.out_dir, args.plot_out)
        return 0
    if args.products_dir is None:
        raise SystemExit("--products-dir is required unless --plot-only is set")
    cfg = StackConfig(
        products_dir=args.products_dir,
        out_dir=args.out_dir,
        stack_bin=args.stack_bin,
        stream_key=args.stream_key,
        ramp_order=args.ramp_order,
        min_coh=args.min_coh,
        corridors=args.corridors,
        buffer_m=args.buffer_m,
        vel_alert_mm_yr=args.vel_alert_mm_yr,
        accel_alert_mm_yr2=args.accel_alert_mm_yr2,
        min_alert_coh=args.min_alert_coh,
        reset=args.reset,
    )
    summary = sync_stack(cfg)
    print(f"\nStack complete: {dataclasses.asdict(summary)}")
    print_corridor_leaderboard(cfg.out_dir)
    if args.plot:
        generate_stack_plots(cfg.out_dir, args.plot_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
