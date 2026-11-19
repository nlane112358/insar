#!/usr/bin/env python3
"""
Validate InSAR stack LOS velocity and cumulative displacement against UNR / Nevada Geodetic
Laboratory (NGL) GNSS ground-truth stations and render a publication-grade benchmark plot.

Supports:
  - Exact zero-Doppler radar LOS unit vector extraction from OPERA CSLC-S1 HDF5 orbit state vectors
    (or manual --incidence-deg / --heading-deg fallback)
  - UNR DataHoldings.txt + MIDAS velocity tables (midas.NA.txt / midas.IGS14.txt) + .tenv3 daily series
  - Direct & de-biased RMSE, MAE, Bias, R^2, and Pearson r in mm/yr
  - Optional before-vs-after baseline comparison (--baseline-stack-dir)
  - Optional overlapping epoch-by-epoch cumulative displacement validation against daily GNSS (.tenv3)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from insar_platform.paths import ensure_runtime_python, resolve_runtime_path
from insar_platform.providers.unr_gnss import (
    UNR_DATA_HOLDINGS_URL,
    UNR_MIDAS_NA_URL,
    UNR_MIDAS_TEMPLATE,
    UNR_TENV3_NA_TEMPLATE,
    UNR_TENV3_TEMPLATE,
    GnssStation,
    StationMatch,
    _mjd_to_date,
    estimate_tenv3_velocity,
    load_stations_from_csv,
    load_unr_stations_in_bbox,
    parse_tenv3,
    resolve_gnss_frame,
)
from insar_platform.viz.gnss_plots import (
    _compute_map_view_and_norm,
    _configure_scientific_matplotlib,
    _draw_cartographic_scalebar,
    _utm_axis_labels,
    render_gnss_scatter_plot,
    render_gnss_timeseries_plot,
    render_gnss_validation_figures,
    render_gnss_velocity_map,
)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _ensure_gdal_plot_runtime(script_file: str, argv: Sequence[str] | None = None) -> None:
    ensure_runtime_python(script_file, ("osgeo", "matplotlib"), argv, set_mpl_cache=True)
    try:
        import matplotlib  # noqa: F401
        from osgeo import gdal, osr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "gnss_validation requires GDAL (osgeo) and matplotlib for raster sampling and plotting."
        ) from exc


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float = 0.0) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n_val = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n_val + height_m) * cos_lat * math.cos(lon)
    y = (n_val + height_m) * cos_lat * math.sin(lon)
    z = (n_val * (1.0 - WGS84_E2) + height_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e_hat = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64)
    n_hat = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=np.float64,
    )
    u_hat = np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=np.float64,
    )
    return e_hat, n_hat, u_hat


def load_orbit_from_h5(h5_path: Path) -> dict[str, np.ndarray]:
    """Read ECEF satellite position & velocity state vectors from OPERA CSLC-S1 or NISAR GSLC HDF5."""
    def _extract_from_file(f: Any) -> dict[str, np.ndarray]:
        if "metadata/orbit" in f:
            grp = f["metadata/orbit"]
            return {
                "t": np.asarray(grp["time"][()], dtype=np.float64),
                "px": np.asarray(grp["position_x"][()], dtype=np.float64),
                "py": np.asarray(grp["position_y"][()], dtype=np.float64),
                "pz": np.asarray(grp["position_z"][()], dtype=np.float64),
                "vx": np.asarray(grp["velocity_x"][()], dtype=np.float64),
                "vy": np.asarray(grp["velocity_y"][()], dtype=np.float64),
                "vz": np.asarray(grp["velocity_z"][()], dtype=np.float64),
            }
        for nisar_orb in ("/science/LSAR/GSLC/metadata/orbit", "/science/SSAR/GSLC/metadata/orbit"):
            if nisar_orb in f:
                grp = f[nisar_orb]
                pos = np.asarray(grp["position"][()], dtype=np.float64)
                vel = np.asarray(grp["velocity"][()], dtype=np.float64)
                return {
                    "t": np.asarray(grp["time"][()], dtype=np.float64),
                    "px": pos[:, 0],
                    "py": pos[:, 1],
                    "pz": pos[:, 2],
                    "vx": vel[:, 0],
                    "vy": vel[:, 1],
                    "vz": vel[:, 2],
                }
        raise KeyError(f"No OPERA or NISAR orbit metadata group found in {h5_path}")

    try:
        import h5py  # type: ignore

        with h5py.File(h5_path, "r") as f:
            return _extract_from_file(f)
    except ImportError:
        # Fallback to reading via .venv python if system python lacks h5py
        venv_py = _project_root() / ".venv" / "bin" / "python"
        if venv_py.is_file():
            code = (
                "import json, h5py, numpy as np; "
                f"f = h5py.File({str(h5_path)!r}, 'r'); "
                "if 'metadata/orbit' in f:\n"
                "  g = f['metadata/orbit']\n"
                "  out = {k: g[v][()].tolist() for k, v in [('t','time'),('px','position_x'),('py','position_y'),('pz','position_z'),('vx','velocity_x'),('vy','velocity_y'),('vz','velocity_z')]}\n"
                "else:\n"
                "  g = f['/science/LSAR/GSLC/metadata/orbit']\n"
                "  p = np.asarray(g['position'][()]); v = np.asarray(g['velocity'][()])\n"
                "  out = {'t': g['time'][()].tolist(), 'px': p[:,0].tolist(), 'py': p[:,1].tolist(), 'pz': p[:,2].tolist(), 'vx': v[:,0].tolist(), 'vy': v[:,1].tolist(), 'vz': v[:,2].tolist()}\n"
                "print(json.dumps(out))"
            )
            proc = subprocess.run([str(venv_py), "-c", code], capture_output=True, text=True, check=True)
            raw = json.loads(proc.stdout)
            return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
        raise


def load_orbit_from_opera_h5(h5_path: Path) -> dict[str, np.ndarray]:
    """Backwards-compatible alias for load_orbit_from_h5."""
    return load_orbit_from_h5(h5_path)


def compute_los_vector_from_orbit(
    orbit: Mapping[str, np.ndarray],
    lat_deg: float,
    lon_deg: float,
    height_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Solve zero-Doppler closest approach along the satellite orbit to ground target (lat, lon, h)
    and return unit ground->satellite ENU look vector (u_E, u_N, u_U) plus incidence angle (deg).
    """
    p_gnd = geodetic_to_ecef(lat_deg, lon_deg, height_m)
    t = orbit["t"]
    t_mid = 0.5 * (t[0] + t[-1])
    dt = t - t_mid
    deg = min(4, len(t) - 1)
    c_px = np.polyfit(dt, orbit["px"], deg)
    c_py = np.polyfit(dt, orbit["py"], deg)
    c_pz = np.polyfit(dt, orbit["pz"], deg)
    c_vx = np.polyfit(dt, orbit["vx"], deg)
    c_vy = np.polyfit(dt, orbit["vy"], deg)
    c_vz = np.polyfit(dt, orbit["vz"], deg)

    def _sat_pv(tau: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([np.polyval(c_px, tau), np.polyval(c_py, tau), np.polyval(c_pz, tau)])
        v = np.array([np.polyval(c_vx, tau), np.polyval(c_vy, tau), np.polyval(c_vz, tau)])
        return p, v

    # Newton-Raphson on zero-Doppler condition f(tau) = (p_sat(tau) - p_gnd) . v_sat(tau) = 0
    tau = 0.0
    for _ in range(12):
        p_s, v_s = _sat_pv(tau)
        f_val = float(np.dot(p_s - p_gnd, v_s))
        # Numerical derivative df/dtau
        eps = 1e-3
        p_s2, v_s2 = _sat_pv(tau + eps)
        df = (float(np.dot(p_s2 - p_gnd, v_s2)) - f_val) / eps
        if abs(df) < 1e-9:
            break
        step = f_val / df
        tau -= step
        if abs(step) < 1e-6:
            break

    p_sat, _ = _sat_pv(tau)
    los_ecef = p_sat - p_gnd
    los_ecef /= np.linalg.norm(los_ecef)

    e_hat, n_hat, u_hat = enu_basis(lat_deg, lon_deg)
    u_e = float(np.dot(los_ecef, e_hat))
    u_n = float(np.dot(los_ecef, n_hat))
    u_u = float(np.dot(los_ecef, u_hat))
    inc_deg = math.degrees(math.acos(max(-1.0, min(1.0, u_u))))
    return u_e, u_n, u_u, inc_deg


def compute_los_vector_from_angles(incidence_deg: float, heading_deg: float, right_looking: bool = True) -> tuple[float, float, float, float]:
    """
    Compute ground->satellite ENU unit vector from radar incidence angle and satellite flight heading
    (degrees clockwise from North, e.g. ~-169 deg for Sentinel-1 descending, ~-11 deg for ascending).
    """
    inc = math.radians(incidence_deg)
    hdg = math.radians(heading_deg)
    # Ground->satellite horizontal azimuth is perpendicular to flight heading (left of track for right-looking)
    sign = -1.0 if right_looking else 1.0
    u_e = sign * math.sin(inc) * math.cos(hdg)
    u_n = -sign * math.sin(inc) * math.sin(hdg)
    u_u = math.cos(inc)
    return u_e, u_n, u_u, incidence_deg


def compute_validation_metrics(gnss_vals: np.ndarray, insar_vals: np.ndarray) -> dict[str, float]:
    """Compute direct and de-biased RMSE, MAE, bias, R^2, and Pearson r (in mm/yr)."""
    n = int(len(gnss_vals))
    if n == 0:
        return {}
    diff = insar_vals - gnss_vals
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    med_bias = float(np.median(diff))
    diff_deb = diff - med_bias
    rmse_debiased = float(np.sqrt(np.mean(diff_deb * diff_deb)))
    mae_debiased = float(np.mean(np.abs(diff_deb)))

    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((gnss_vals - np.mean(gnss_vals)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")
    if n >= 2 and np.std(gnss_vals) > 1e-9 and np.std(insar_vals) > 1e-9:
        pearson_r = float(np.corrcoef(gnss_vals, insar_vals)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "n_stations": n,
        "rmse_mm_yr": rmse,
        "mae_mm_yr": mae,
        "bias_mm_yr": bias,
        "median_bias_mm_yr": med_bias,
        "rmse_debiased_mm_yr": rmse_debiased,
        "mae_debiased_mm_yr": mae_debiased,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def _sample_stack_at_station(
    vel: np.ndarray,
    coh: np.ndarray,
    unc: np.ndarray | None,
    gt: Sequence[float],
    ux: float,
    uy: float,
    radius_m: float,
    min_coh: float,
) -> tuple[float, float, float, float, int] | None:
    """Return (median_vel, median_unc, spatial_std, median_coh, n_px) inside circular radius_m."""
    h, w = vel.shape
    cx = (ux - gt[0]) / gt[1]
    cy = (uy - gt[3]) / gt[5]
    pix_m = 0.5 * (abs(gt[1]) + abs(gt[5]))
    r_px = max(1.5, radius_m / max(pix_m, 1e-6))

    x0 = max(0, int(math.floor(cx - r_px)))
    x1 = min(w - 1, int(math.ceil(cx + r_px)))
    y0 = max(0, int(math.floor(cy - r_px)))
    y1 = min(h - 1, int(math.ceil(cy + r_px)))
    if x0 > x1 or y0 > y1:
        return None

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    in_disk = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) <= (r_px * r_px)
    sub_v = vel[y0 : y1 + 1, x0 : x1 + 1]
    sub_c = coh[y0 : y1 + 1, x0 : x1 + 1]
    valid = in_disk & np.isfinite(sub_v) & np.isfinite(sub_c) & (sub_c >= min_coh)
    if int(valid.sum()) < 3:
        valid = in_disk & np.isfinite(sub_v) & np.isfinite(sub_c) & (sub_c >= max(0.25, min_coh * 0.6))
    if int(valid.sum()) == 0:
        return None

    vals = sub_v[valid]
    cohs = sub_c[valid]
    med_v = float(np.median(vals))
    std_v = float(np.std(vals))
    med_c = float(np.median(cohs))
    if unc is not None:
        sub_u = unc[y0 : y1 + 1, x0 : x1 + 1]
        u_valid = valid & np.isfinite(sub_u)
        med_u = float(np.median(sub_u[u_valid])) if int(u_valid.sum()) > 0 else std_v
    else:
        med_u = std_v
    return med_v, med_u, std_v, med_c, int(valid.sum())


def _extract_pair_timeseries_at_station(
    products_dir: Path,
    stack_ledger: Mapping[str, Any],
    station: GnssStation,
    ref_lonlat: tuple[float, float] | None,
    ref_radius_m: float,
    sample_radius_m: float,
    dem_path: Path | None,
    stack_dir: Path,
    insar_sign: float,
) -> tuple[list[date], list[float]] | None:
    """
    Reconstruct cumulative LOS displacement curve at `station` across the stacked pair chain,
    differenced against the stable reference region (and with DEM slope applied when available).
    """
    from osgeo import gdal, osr  # type: ignore

    pair_ids: list[str] = stack_ledger.get("pair_ids", [])
    stream_key = stack_ledger.get("stream_key", "")
    if not pair_ids or not stream_key:
        return None

    ref_ds = gdal.Open(str(stack_dir / "velocity_mm_yr.tif"))
    vel_prior = ref_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    h_r, w_r = vel_prior.shape
    ref_ds = None

    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(proj)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)

    sx, sy, _ = ct.TransformPoint(station.lon, station.lat, 0.0)
    rx, ry = (None, None)
    if ref_lonlat is not None:
        rx, ry, _ = ct.TransformPoint(ref_lonlat[0], ref_lonlat[1], 0.0)

    # Restrict lightweight per-pair DEM tropo fit to stable non-deforming pixels inside
    # the local basin (within 25 km of midpoint between station and reference pin),
    # mirroring insar_stack's connected-component + high-deformation masking + ridge damping.
    dem_arr = None
    local_stable_mask = None
    if dem_path is not None and dem_path.is_file():
        dds = gdal.Open(str(dem_path))
        dem_arr = dds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        dds = None
        if dem_arr.shape == vel_prior.shape:
            cx_m = 0.5 * (sx + rx) if rx is not None else sx
            cy_m = 0.5 * (sy + ry) if ry is not None else sy
            xs_u = gt[0] + (np.arange(w_r) + 0.5) * gt[1]
            ys_u = gt[3] + (np.arange(h_r) + 0.5) * gt[5]
            in_x = np.abs(xs_u - cx_m) <= 22000.0
            in_y = np.abs(ys_u - cy_m) <= 22000.0
            in_basin = np.outer(in_y, in_x)
            local_stable_mask = in_basin & np.isfinite(dem_arr) & np.isfinite(vel_prior) & (np.abs(vel_prior) < 35.0)

    epochs: list[date] = [datetime.strptime(pair_ids[0].split("_")[0], "%Y-%m-%d").date()]
    cum_mm: list[float] = [0.0]
    running = 0.0

    for pid in pair_ids:
        d0_s, d1_s = pid.split("_")
        d0 = datetime.strptime(d0_s, "%Y-%m-%d").date()
        d1 = datetime.strptime(d1_s, "%Y-%m-%d").date()
        pdir = products_dir / f"{stream_key}_{pid}"
        disp_p = pdir / "displacement_mm.tif"
        coh_p = pdir / "coherence.tif"
        if not (disp_p.is_file() and coh_p.is_file()):
            continue

        dds = gdal.Open(str(disp_p))
        d_arr = dds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        dds = None
        cds = gdal.Open(str(coh_p))
        c_arr = cds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        cds = None

        # Ridge-damped DEM linear tropo removal on local stable coherent background
        if dem_arr is not None and local_stable_mask is not None and dem_arr.shape == d_arr.shape:
            m_ok = local_stable_mask & np.isfinite(d_arr) & np.isfinite(c_arr) & (c_arr >= 0.45)
            if int(m_ok.sum()) > 500:
                z_s = dem_arr[m_ok][::10]
                d_s = d_arr[m_ok][::10]
                z0 = float(np.mean(z_s))
                dz = (z_s - z0) / 1000.0
                d0_m = float(np.mean(d_s))
                s_zz = float(np.sum(dz * dz)) + len(dz) * 0.0025
                s_zd = float(np.sum(dz * (d_s - d0_m)))
                slope = s_zd / s_zz
                d_arr = d_arr - (d0_m + slope * ((dem_arr - z0) / 1000.0))

        s_samp = _sample_stack_at_station(d_arr, c_arr, None, gt, sx, sy, sample_radius_m, 0.35)
        r_val = 0.0
        if rx is not None and ry is not None:
            r_samp = _sample_stack_at_station(d_arr, c_arr, None, gt, rx, ry, ref_radius_m, 0.35)
            if r_samp is not None:
                r_val = r_samp[0]

        if s_samp is not None:
            step_mm = insar_sign * (s_samp[0] - r_val)
            # Bridge any calendar gap between previous epoch and d0 using average rate so far
            if d0 > epochs[-1] and len(epochs) > 1:
                elapsed = max(1, (epochs[-1] - epochs[0]).days)
                gap_d = (d0 - epochs[-1]).days
                running += (running / elapsed) * gap_d
            running += step_mm
            epochs.append(d1)
            cum_mm.append(running)

    return (epochs, cum_mm) if len(epochs) >= 3 else None


def _load_baseline_from_csv(csv_path: Path) -> dict[str, float]:
    """Load precomputed baseline InSAR LOS velocities keyed by station_id from a validation CSV."""
    baseline_by_id: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("station_id") or "").strip()
            raw_val = (row.get("baseline_insar_los_mm_yr") or "").strip()
            if sid and raw_val:
                baseline_by_id[sid] = float(raw_val)
    return baseline_by_id


def validate_stack_against_gnss(
    *,
    stack_dir: Path,
    gnss_cache_dir: Path | None = None,
    gnss_csv: Path | None = None,
    gnss_frame: str = "auto",
    slc_h5: Path | None = None,
    incidence_deg: float = 34.0,
    heading_deg: float = -169.0,
    right_looking: bool | None = None,
    sample_radius_m: float = 200.0,
    min_coh: float = 0.35,
    insar_sign: float = -1.0,
    fetch_missing: bool = False,
    baseline_stack_dir: Path | None = None,
    baseline_csv: Path | None = None,
    products_dir: Path | None = None,
    dem_path: Path | None = None,
    out_json: Path | None = None,
    out_csv: Path | None = None,
    plot_out: Path | None = None,
    map_out: Path | None = None,
    scatter_out: Path | None = None,
    timeseries_out: Path | None = None,
    plots_dir: Path | None = None,
    plot_prefix: str = "gnss_validation",
    dpi: int = 300,
) -> dict[str, Any]:
    """Run full InSAR-vs-GNSS validation and optionally render standalone matplotlib figures."""
    from osgeo import gdal, osr  # type: ignore

    gdal.UseExceptions()
    stack_dir = resolve_runtime_path(stack_dir)
    if baseline_stack_dir is not None:
        baseline_stack_dir = resolve_runtime_path(baseline_stack_dir)
    if products_dir is not None:
        products_dir = resolve_runtime_path(products_dir)
    if dem_path is not None:
        dem_path = resolve_runtime_path(dem_path)
    if slc_h5 is not None:
        slc_h5 = resolve_runtime_path(slc_h5)
    vel_ds = gdal.Open(str(stack_dir / "velocity_mm_yr.tif"))
    if vel_ds is None:
        raise FileNotFoundError(f"Missing velocity_mm_yr.tif in {stack_dir}")
    vel = vel_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = list(vel_ds.GetGeoTransform())
    proj = vel_ds.GetProjection()
    h, w = vel.shape
    vel_ds = None

    coh_ds = gdal.Open(str(stack_dir / "temporal_coherence.tif"))
    coh = coh_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    coh_ds = None

    unc_p = stack_dir / "velocity_uncertainty_mm_yr.tif"
    unc = None
    if unc_p.is_file():
        uds = gdal.Open(str(unc_p))
        unc = uds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        uds = None

    base_vel = base_coh = None
    if baseline_stack_dir is not None and (baseline_stack_dir / "velocity_mm_yr.tif").is_file():
        bds = gdal.Open(str(baseline_stack_dir / "velocity_mm_yr.tif"))
        base_vel = bds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        bds = None
        bcds = gdal.Open(str(baseline_stack_dir / "temporal_coherence.tif"))
        base_coh = bcds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        bcds = None
    baseline_by_id = _load_baseline_from_csv(baseline_csv) if (base_vel is None and baseline_csv is not None and baseline_csv.is_file()) else {}

    src_wgs = osr.SpatialReference()
    src_wgs.ImportFromEPSG(4326)
    src_wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_ras = osr.SpatialReference()
    dst_ras.ImportFromWkt(proj)
    dst_ras.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct_to_ras = osr.CoordinateTransformation(src_wgs, dst_ras)
    ct_to_wgs = osr.CoordinateTransformation(dst_ras, src_wgs)

    corners = [
        (gt[0], gt[3]),
        (gt[0] + w * gt[1], gt[3]),
        (gt[0] + w * gt[1], gt[3] + h * gt[5]),
        (gt[0], gt[3] + h * gt[5]),
    ]
    ll = [ct_to_wgs.TransformPoint(x, y, 0.0)[:2] for x, y in corners]
    bbox_wgs84 = (
        min(p[0] for p in ll),
        min(p[1] for p in ll),
        max(p[0] for p in ll),
        max(p[1] for p in ll),
    )

    ledger_p = stack_dir / "stack_ledger.json"
    ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.is_file() else {}
    window_dates: tuple[date, date] | None = None
    if ledger.get("pair_ids"):
        d_start_w = datetime.strptime(ledger["pair_ids"][0].split("_")[0], "%Y-%m-%d").date()
        d_end_w = datetime.strptime(ledger["pair_ids"][-1].split("_")[1], "%Y-%m-%d").date()
        window_dates = (d_start_w, d_end_w)

    resolved_slc_h5: Path | None = None
    if slc_h5 is not None:
        if slc_h5.is_file():
            resolved_slc_h5 = slc_h5
        elif slc_h5.is_dir():
            cands = sorted(slc_h5.glob("*.h5"))
            if cands:
                resolved_slc_h5 = cands[-1]
    else:
        auto_raw_dir = stack_dir.parent.parent / "raw" / stack_dir.name
        if auto_raw_dir.is_dir():
            cands = sorted(auto_raw_dir.glob("*.h5"))
            if cands:
                resolved_slc_h5 = cands[-1]

    resolved_frame = resolve_gnss_frame(gnss_frame, bbox_wgs84)
    stream_key = str(ledger.get("stream_key") or "")
    effective_right_looking = (
        not stream_key.upper().startswith("NISAR_")
        if right_looking is None
        else bool(right_looking)
    )
    sensor_label = "NISAR L-band InSAR" if stream_key.upper().startswith("NISAR_") else "Sentinel-1 InSAR"

    orbit = load_orbit_from_h5(resolved_slc_h5) if (resolved_slc_h5 is not None and resolved_slc_h5.is_file()) else None
    if gnss_csv is not None:
        stations = load_stations_from_csv(gnss_csv)
    else:
        stations = load_unr_stations_in_bbox(
            resolve_runtime_path(gnss_cache_dir) if gnss_cache_dir else resolve_runtime_path(Path("var/work/gnss/unr")),
            bbox_wgs84,
            fetch_missing=fetch_missing,
            window_dates=window_dates,
            gnss_frame=resolved_frame,
        )

    matches: list[StationMatch] = []
    station_by_id: dict[str, GnssStation] = {s.station_id: s for s in stations}

    for st in stations:
        ux, uy, _ = ct_to_ras.TransformPoint(st.lon, st.lat, 0.0)
        samp = _sample_stack_at_station(vel, coh, unc, gt, ux, uy, sample_radius_m, min_coh)
        if samp is None:
            continue
        med_v, med_u, std_v, med_c, n_px = samp
        if orbit is not None:
            u_e, u_n, u_u, inc = compute_los_vector_from_orbit(orbit, st.lat, st.lon, st.height_m)
        else:
            u_e, u_n, u_u, inc = compute_los_vector_from_angles(
                incidence_deg, heading_deg, right_looking=effective_right_looking
            )

        # Positive toward satellite (negative = subsidence) when insar_sign == -1.0
        # Because OPERA s1*conj(s2) gives range increase (+) for subsidence, multiplying by insar_sign=-1
        # converts InSAR to toward-satellite (+), matching +(u_E*v_E + u_N*v_N + u_U*v_U)!
        gnss_sign = -insar_sign  # +1 when insar_sign == -1
        gnss_los = gnss_sign * (u_e * st.ve_mm_yr + u_n * st.vn_mm_yr + u_u * st.vu_mm_yr)
        gnss_sig = math.sqrt((u_e * st.se_mm_yr) ** 2 + (u_n * st.sn_mm_yr) ** 2 + (u_u * st.su_mm_yr) ** 2)
        insar_los = insar_sign * med_v

        base_los = None
        base_res = None
        if base_vel is not None and base_coh is not None:
            bs = _sample_stack_at_station(base_vel, base_coh, None, gt, ux, uy, sample_radius_m, min_coh)
            if bs is not None:
                base_los = insar_sign * bs[0]
                base_res = base_los - gnss_los
        elif st.station_id in baseline_by_id:
            base_los = baseline_by_id[st.station_id]
            base_res = base_los - gnss_los

        matches.append(
            StationMatch(
                station_id=st.station_id,
                lat=st.lat,
                lon=st.lon,
                height_m=st.height_m,
                start_date=st.start_date,
                end_date=st.end_date,
                velocity_source=st.source,
                incidence_deg=inc,
                los_e=u_e,
                los_n=u_n,
                los_u=u_u,
                gnss_ve_mm_yr=st.ve_mm_yr,
                gnss_vn_mm_yr=st.vn_mm_yr,
                gnss_vu_mm_yr=st.vu_mm_yr,
                gnss_los_mm_yr=gnss_los,
                gnss_los_sigma_mm_yr=gnss_sig,
                insar_los_mm_yr=insar_los,
                insar_sigma_mm_yr=med_u,
                insar_spatial_std_mm_yr=std_v,
                insar_coherence=med_c,
                insar_sample_pixels=n_px,
                residual_mm_yr=insar_los - gnss_los,
                baseline_insar_los_mm_yr=base_los,
                baseline_residual_mm_yr=base_res,
            )
        )

    if not matches:
        raise RuntimeError(f"No GNSS stations overlapped valid pixels in {stack_dir}")

    gnss_arr = np.array([m.gnss_los_mm_yr for m in matches])
    insar_arr = np.array([m.insar_los_mm_yr for m in matches])
    metrics = compute_validation_metrics(gnss_arr, insar_arr)

    baseline_metrics = None
    if any(m.baseline_insar_los_mm_yr is not None for m in matches):
        bg = np.array([m.gnss_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None])
        bi = np.array([m.baseline_insar_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None])
        baseline_metrics = compute_validation_metrics(bg, bi)

    # Check if an active GNSS station overlaps the exact stack time window for Panel C time-series overlay
    summary_p = stack_dir / "stack_summary.json"
    summary = json.loads(summary_p.read_text(encoding="utf-8")) if summary_p.is_file() else {}
    ref_cfg = summary.get("reference_region", {})
    ref_lonlat = (
        (float(ref_cfg["lon"]), float(ref_cfg["lat"]))
        if ref_cfg.get("applied") and ref_cfg.get("lon") is not None and ref_cfg.get("lat") is not None
        else None
    )
    ref_radius_m = float(ref_cfg.get("radius_m") or 1000.0)

    ts_station = None
    ts_gnss_dates = ts_gnss_los_mm = ts_insar_dates = ts_insar_los_mm = None
    ts_rmse_mm = None
    if products_dir is not None and products_dir.is_dir() and gnss_cache_dir is not None and ledger.get("pair_ids"):
        d_start = datetime.strptime(ledger["pair_ids"][0].split("_")[0], "%Y-%m-%d").date()
        d_end = datetime.strptime(ledger["pair_ids"][-1].split("_")[1], "%Y-%m-%d").date()
        # Pick highest-magnitude active station with daily .tenv3 overlapping stack dates
        active_matches = sorted(matches, key=lambda m: -abs(m.gnss_los_mm_yr))
        for m in active_matches:
            tenv_p = gnss_cache_dir / f"{m.station_id}.{resolved_frame}.tenv3"
            if not tenv_p.is_file():
                continue
            ts_raw = parse_tenv3(tenv_p)
            dates_all = [_mjd_to_date(int(mjd)) for mjd in ts_raw["mjd"]]
            overlap_idx = [i for i, d in enumerate(dates_all) if d_start <= d <= d_end]
            if len(overlap_idx) < 30:
                continue
            st_obj = station_by_id[m.station_id]
            insar_curve = _extract_pair_timeseries_at_station(
                products_dir, ledger, st_obj, ref_lonlat, ref_radius_m, sample_radius_m, dem_path, stack_dir, insar_sign
            )
            if insar_curve is None:
                continue
            i_dates, i_vals = insar_curve
            gnss_sign = -insar_sign
            e0 = ts_raw["east_m"][overlap_idx[0]]
            n0 = ts_raw["north_m"][overlap_idx[0]]
            u0 = ts_raw["up_m"][overlap_idx[0]]
            g_dates = [dates_all[i] for i in overlap_idx]
            g_vals = [
                float(
                    gnss_sign
                    * 1000.0
                    * (
                        m.los_e * (ts_raw["east_m"][i] - e0)
                        + m.los_n * (ts_raw["north_m"][i] - n0)
                        + m.los_u * (ts_raw["up_m"][i] - u0)
                    )
                )
                for i in overlap_idx
            ]
            # Align zero datum to median of first 7 days of GNSS
            g_zero = float(np.median(g_vals[: min(7, len(g_vals))]))
            g_vals = [v - g_zero for v in g_vals]

            # Compute epoch-to-epoch cumulative RMSE against overlapping daily GNSS
            g_by_ord = {d.toordinal(): v for d, v in zip(g_dates, g_vals)}
            errs = []
            for d_in, v_in in zip(i_dates, i_vals):
                o = d_in.toordinal()
                near = [g_by_ord[k] for k in range(o - 3, o + 4) if k in g_by_ord]
                if near:
                    errs.append(v_in - float(np.median(near)))
            if errs:
                ts_rmse_mm = float(np.sqrt(np.mean(np.square(errs))))
            ts_station = st_obj
            ts_gnss_dates, ts_gnss_los_mm = g_dates, g_vals
            ts_insar_dates, ts_insar_los_mm = i_dates, i_vals
            break

    report = {
        "stack_dir": str(stack_dir),
        "convention": "LOS velocity toward satellite (mm/yr; negative = subsidence)" if insar_sign < 0 else "LOS range-increase velocity (mm/yr; positive = subsidence)",
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "timeseries_overlay": {
            "station_id": ts_station.station_id if ts_station else None,
            "cumulative_rmse_mm": ts_rmse_mm,
            "n_insar_epochs": len(ts_insar_dates) if ts_insar_dates else 0,
            "n_gnss_days": len(ts_gnss_dates) if ts_gnss_dates else 0,
        },
        "stations": [asdict(m) for m in matches],
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote GNSS validation JSON -> {out_json}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(matches[0]).keys())
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            w_csv = csv.DictWriter(fh, fieldnames=fieldnames)
            w_csv.writeheader()
            for m in matches:
                w_csv.writerow(asdict(m))
        print(f"Wrote GNSS validation CSV -> {out_csv}")

    if any(p is not None for p in (map_out, scatter_out, timeseries_out, plots_dir, plot_out)):
        render_gnss_validation_figures(
            stack_dir=stack_dir,
            matches=matches,
            ref_lonlat=ref_lonlat,
            insar_sign=insar_sign,
            ts_station=ts_station,
            ts_gnss_dates=ts_gnss_dates,
            ts_gnss_los_mm=ts_gnss_los_mm,
            ts_insar_dates=ts_insar_dates,
            ts_insar_los_mm=ts_insar_los_mm,
            sensor_label=sensor_label,
            map_out=map_out,
            scatter_out=scatter_out,
            timeseries_out=timeseries_out,
            plots_dir=plots_dir,
            plot_prefix=plot_prefix,
            plot_out=plot_out,
            dpi=dpi,
        )

    return report


def validate_stack(stack_dir: Path, **options: Any) -> Any:
    """Validate a stack against GNSS and persist validation summary metrics when available."""
    monitor_name = str(options.pop("monitor_name", stack_dir.name))
    job_id = options.pop("job_id", None)
    report = validate_stack_against_gnss(stack_dir=stack_dir, **options)
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    if metrics:
        from datetime import timezone
        from insar_platform.storage.database import get_db, init_db

        measured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metric_id = hashlib.sha256(f"gnss:{monitor_name}:{job_id or stack_dir}:{measured_at}".encode()).hexdigest()
        init_db()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO validation_metrics (metric_id, monitor_name, job_id, measured_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (metric_id) DO NOTHING""",
                (metric_id, monitor_name, job_id, measured_at, json.dumps(metrics)),
            )
            conn.commit()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate InSAR stack velocity against UNR/NGL GNSS ground truth.")
    p.add_argument("--stack-dir", required=True, type=Path, help="Stack directory with velocity_mm_yr.tif and temporal_coherence.tif")
    p.add_argument("--baseline-stack-dir", type=Path, default=None, help="Optional baseline stack dir for before/after comparison")
    p.add_argument("--baseline-csv", type=Path, default=None, help="Optional validation CSV with baseline_insar_los_mm_yr column")
    p.add_argument("--products-dir", type=Path, default=None, help="Optional pair products dir for 2026 cumulative time-series overlay")
    p.add_argument("--dem", type=Path, default=None, help="Optional aligned DEM GeoTIFF used during time-series overlay")
    p.add_argument("--gnss-cache-dir", type=Path, default=Path("var/work/gnss/unr"), help="Directory with UNR DataHoldings.txt, midas.<frame>.txt, *.<frame>.tenv3")
    p.add_argument("--gnss-csv", type=Path, default=None, help="Optional local GNSS station CSV instead of UNR cache")
    p.add_argument("--gnss-frame", default="auto", help="UNR plate reference frame (auto, NA, EU, IGS14, etc.; auto selects EU in Europe)")
    p.add_argument("--slc-h5", type=Path, default=None, help="OPERA CSLC-S1 or NISAR GSLC .h5 file or raw dir for exact zero-Doppler orbit LOS vector")
    p.add_argument("--incidence-deg", type=float, default=34.0, help="Fallback radar incidence angle (deg) if --slc-h5 omitted")
    p.add_argument("--heading-deg", type=float, default=-169.0, help="Fallback satellite heading (deg CW from N) if --slc-h5 omitted")
    p.add_argument("--look-direction", choices=["auto", "right", "left"], default="auto", help="Radar look side when --slc-h5 is omitted (auto selects left for NISAR, right otherwise)")
    p.add_argument("--sample-radius-m", type=float, default=200.0, help="Spatial median radius around each GNSS station (m)")
    p.add_argument("--min-coh", type=float, default=0.35, help="Minimum temporal coherence when sampling InSAR velocity")
    p.add_argument("--fetch-missing", action="store_true", help="Download missing UNR files if not cached locally")
    p.add_argument("--out-json", type=Path, default=None, help="Output validation metrics JSON path")
    p.add_argument("--out-csv", type=Path, default=None, help="Output per-station comparison CSV path")
    p.add_argument("--plot-out", type=Path, default=None, help="Base output path to emit separate *_map, *_scatter, *_timeseries PNGs")
    p.add_argument("--plots-dir", type=Path, default=None, help="Directory to write separate map, scatter, and timeseries PNGs")
    p.add_argument("--plot-prefix", type=str, default="gnss_validation", help="Filename prefix when using --plots-dir")
    p.add_argument("--map-out", type=Path, default=None, help="Output path for standalone velocity map figure")
    p.add_argument("--scatter-out", type=Path, default=None, help="Output path for standalone 1:1 scatter figure")
    p.add_argument("--timeseries-out", type=Path, default=None, help="Output path for standalone time-series figure")
    p.add_argument("--dpi", type=int, default=300, help="Figure output resolution in DPI (default: 300)")
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_gdal_plot_runtime(__file__, argv)
    args = build_arg_parser().parse_args(argv)
    right_looking = None if args.look_direction == "auto" else (args.look_direction == "right")

    report = validate_stack(
        stack_dir=args.stack_dir,
        monitor_name=args.stack_dir.name,
        gnss_cache_dir=args.gnss_cache_dir,
        gnss_csv=args.gnss_csv,
        gnss_frame=args.gnss_frame,
        slc_h5=args.slc_h5,
        incidence_deg=args.incidence_deg,
        heading_deg=args.heading_deg,
        right_looking=right_looking,
        sample_radius_m=args.sample_radius_m,
        min_coh=args.min_coh,
        fetch_missing=args.fetch_missing,
        baseline_stack_dir=args.baseline_stack_dir,
        baseline_csv=args.baseline_csv,
        products_dir=args.products_dir,
        dem_path=args.dem,
        out_json=args.out_json,
        out_csv=args.out_csv,
        plot_out=args.plot_out,
        map_out=args.map_out,
        scatter_out=args.scatter_out,
        timeseries_out=args.timeseries_out,
        plots_dir=args.plots_dir,
        plot_prefix=args.plot_prefix,
        dpi=args.dpi,
    )
    print("\nGNSS Validation Summary:")
    print(json.dumps(report["metrics"], indent=2))
    if report["baseline_metrics"]:
        print("Baseline Metrics:")
        print(json.dumps(report["baseline_metrics"], indent=2))
    print("\nPer-station comparison (LOS toward satellite, mm/yr):")
    for s in report["stations"]:
        b_str = f" | base={s['baseline_insar_los_mm_yr']:+7.1f}" if s["baseline_insar_los_mm_yr"] is not None else ""
        print(
            f"  {s['station_id']:5s} (inc={s['incidence_deg']:.1f}°, coh={s['insar_coherence']:.2f}, {s['velocity_source']:12s}): "
            f"GNSS={s['gnss_los_mm_yr']:+7.1f}±{s['gnss_los_sigma_mm_yr']:4.1f} | "
            f"InSAR={s['insar_los_mm_yr']:+7.1f}±{s['insar_sigma_mm_yr']:4.1f} | "
            f"res={s['residual_mm_yr']:+6.1f} mm/yr{b_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
