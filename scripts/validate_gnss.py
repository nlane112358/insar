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
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))


UNR_DATA_HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
UNR_MIDAS_NA_URL = "https://geodesy.unr.edu/velocities/midas.NA.txt"
UNR_TENV3_NA_TEMPLATE = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/NA/{station}.NA.tenv3"

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class GnssStation:
    station_id: str
    lat: float
    lon: float
    height_m: float
    start_date: str
    end_date: str
    num_solutions: int
    ve_mm_yr: float
    vn_mm_yr: float
    vu_mm_yr: float
    se_mm_yr: float
    sn_mm_yr: float
    su_mm_yr: float
    source: str  # "midas_na", "tenv3_recent", "tenv3_full", "csv"


@dataclass(frozen=True)
class StationMatch:
    station_id: str
    lat: float
    lon: float
    height_m: float
    start_date: str
    end_date: str
    velocity_source: str
    incidence_deg: float
    los_e: float
    los_n: float
    los_u: float
    gnss_ve_mm_yr: float
    gnss_vn_mm_yr: float
    gnss_vu_mm_yr: float
    gnss_los_mm_yr: float
    gnss_los_sigma_mm_yr: float
    insar_los_mm_yr: float
    insar_sigma_mm_yr: float
    insar_spatial_std_mm_yr: float
    insar_coherence: float
    insar_sample_pixels: int
    residual_mm_yr: float
    baseline_insar_los_mm_yr: float | None = None
    baseline_residual_mm_yr: float | None = None


def _ensure_gdal_plot_runtime(script_file: str, argv: Sequence[str] | None = None) -> None:
    try:
        import matplotlib  # noqa: F401
        from osgeo import gdal, osr  # noqa: F401
        return
    except ImportError:
        pass
    hb_py = Path("/opt/homebrew/bin/python3")
    alt_py = hb_py if hb_py.is_file() else Path(shutil.which("python3") or "")
    if alt_py.is_file() and os.path.abspath(sys.executable) != os.path.abspath(str(alt_py)):
        clean_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        clean_env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))
        args = list(sys.argv[1:] if argv is None else argv)
        res = subprocess.run([str(alt_py), str(Path(script_file).resolve()), *args], env=clean_env)
        raise SystemExit(res.returncode)
    raise RuntimeError("validate_gnss.py requires GDAL (osgeo) and matplotlib for raster sampling and plotting.")


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


def load_orbit_from_opera_h5(h5_path: Path) -> dict[str, np.ndarray]:
    """Read ECEF satellite position & velocity state vectors from an OPERA CSLC-S1 .h5 file."""
    try:
        import h5py  # type: ignore

        with h5py.File(h5_path, "r") as f:
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
    except ImportError:
        # Fallback to reading via h5dump or .venv python if system python lacks h5py
        venv_py = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
        if venv_py.is_file():
            code = (
                "import json, h5py; "
                f"f = h5py.File({str(h5_path)!r}, 'r'); g = f['metadata/orbit']; "
                "print(json.dumps({k: g[v][()].tolist() for k, v in "
                "[('t','time'),('px','position_x'),('py','position_y'),('pz','position_z'),"
                "('vx','velocity_x'),('vy','velocity_y'),('vz','velocity_z')]}))"
            )
            proc = subprocess.run([str(venv_py), "-c", code], capture_output=True, text=True, check=True)
            raw = json.loads(proc.stdout)
            return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
        raise


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


def parse_tenv3(path: Path) -> dict[str, np.ndarray]:
    """Parse UNR .tenv3 daily position file into decimal years, dates, ENU meters, and sigmas."""
    decyrs: list[float] = []
    mjds: list[int] = []
    east_m: list[float] = []
    north_m: list[float] = []
    up_m: list[float] = []
    sig_e: list[float] = []
    sig_n: list[float] = []
    sig_u: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    hgts: list[float] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].lower() == "site" or len(parts) < 23:
            continue
        decyrs.append(float(parts[2]))
        mjds.append(int(parts[3]))
        east_m.append(float(parts[7]) + float(parts[8]))
        north_m.append(float(parts[9]) + float(parts[10]))
        up_m.append(float(parts[11]) + float(parts[12]))
        sig_e.append(float(parts[14]))
        sig_n.append(float(parts[15]))
        sig_u.append(float(parts[16]))
        lats.append(float(parts[20]))
        lon_v = float(parts[21])
        if lon_v > 180.0:
            lon_v -= 360.0
        lons.append(lon_v)
        hgts.append(float(parts[22]))

    return {
        "decyr": np.asarray(decyrs, dtype=np.float64),
        "mjd": np.asarray(mjds, dtype=np.int64),
        "east_m": np.asarray(east_m, dtype=np.float64),
        "north_m": np.asarray(north_m, dtype=np.float64),
        "up_m": np.asarray(up_m, dtype=np.float64),
        "sig_e_m": np.asarray(sig_e, dtype=np.float64),
        "sig_n_m": np.asarray(sig_n, dtype=np.float64),
        "sig_u_m": np.asarray(sig_u, dtype=np.float64),
        "lat": np.asarray(lats, dtype=np.float64),
        "lon": np.asarray(lons, dtype=np.float64),
        "height_m": np.asarray(hgts, dtype=np.float64),
    }


def _mjd_to_date(mjd: int) -> date:
    return date(1858, 11, 17) + timedelta(days=int(mjd))


def estimate_tenv3_velocity(
    ts: Mapping[str, np.ndarray],
    *,
    recent_years: float = 3.0,
    window_dates: tuple[date, date] | None = None,
) -> tuple[float, float, float, float, float, float, str]:
    """
    Estimate robust ENU velocity (mm/yr) and uncertainty (mm/yr) from daily .tenv3 series.
    Prefers exact `window_dates` overlap (>= 60 daily epochs) when available, else most recent
    `recent_years` window (when span exceeds `recent_years + 0.5`), else full span.
    Applies iterative Huber reweighting to guard against antenna offsets/outliers.
    """
    t = ts["decyr"]
    if t.size < 30:
        raise ValueError("Too few daily GNSS epochs (<30) to estimate velocity")

    sel = None
    label = "tenv3_full"
    if window_dates is not None and "mjd" in ts:
        d0, d1 = window_dates
        mjd0 = (d0 - date(1858, 11, 17)).days
        mjd1 = (d1 - date(1858, 11, 17)).days
        w_mask = (ts["mjd"] >= mjd0) & (ts["mjd"] <= mjd1)
        if int(w_mask.sum()) >= 60:
            sel = w_mask
            label = f"tenv3_{d0.year}_overlap"

    t_end = float(t[-1])
    if sel is None and (t_end - float(t[0])) > recent_years + 0.5:
        mask = t >= (t_end - recent_years)
        if int(mask.sum()) >= 120:
            sel = mask
            label = f"tenv3_last_{recent_years:.0f}yr"
    if sel is None:
        sel = np.ones_like(t, dtype=bool)
        label = "tenv3_full"

    t_sel = t[sel] - float(t[sel][0])
    span_yr = float(t_sel[-1] - t_sel[0])
    if span_yr < 0.25:
        raise ValueError(f"GNSS time span too short ({span_yr:.2f} yr)")

    def _fit_component(y_m: np.ndarray, sig_m: np.ndarray) -> tuple[float, float]:
        y_mm = (y_m[sel] - y_m[sel][0]) * 1000.0
        w_base = 1.0 / np.maximum(sig_m[sel] * 1000.0, 0.5) ** 2
        w = w_base.copy()
        a_mat = np.column_stack([np.ones_like(t_sel), t_sel])
        beta = np.zeros(2)
        for _ in range(5):
            sw = np.sqrt(w)
            aw = a_mat * sw[:, None]
            yw = y_mm * sw
            beta, *_ = np.linalg.lstsq(aw, yw, rcond=None)
            res = y_mm - (a_mat @ beta)
            mad = max(1.0, 1.4826 * float(np.median(np.abs(res))))
            delta = 1.345 * mad
            huber = np.where(np.abs(res) <= delta, 1.0, delta / np.maximum(np.abs(res), 1e-9))
            w = w_base * huber
        res = y_mm - (a_mat @ beta)
        # Account for temporal correlation (colored noise inflation factor ~sqrt(30) for daily GNSS)
        n = len(t_sel)
        wssr = float(np.sum(w * res * res))
        cov = np.linalg.inv(a_mat.T @ (a_mat * w[:, None]))
        se_white = math.sqrt(max(0.0, (wssr / max(1, n - 2)) * cov[1, 1]))
        se_colored = max(0.5, se_white * math.sqrt(30.0))
        return float(beta[1]), float(se_colored)

    ve, se = _fit_component(ts["east_m"], ts["sig_e_m"])
    vn, sn = _fit_component(ts["north_m"], ts["sig_n_m"])
    vu, su = _fit_component(ts["up_m"], ts["sig_u_m"])
    return ve, vn, vu, se, sn, su, label


def load_unr_stations_in_bbox(
    cache_dir: Path,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    fetch_missing: bool = False,
    prefer_recent_tenv3: bool = True,
    recent_years: float = 3.0,
    window_dates: tuple[date, date] | None = None,
) -> list[GnssStation]:
    """Load UNR GNSS stations overlapping bbox from cached (or freshly fetched) UNR files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    holdings_p = cache_dir / "DataHoldings.txt"
    midas_p = cache_dir / "midas.NA.txt"

    def _download(url: str, dest: Path) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "insar-cpu-proto/1.0"})
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as exc:
            print(f"Warning: failed fetching {url}: {exc}")
            return False

    if not holdings_p.is_file() and fetch_missing:
        _download(UNR_DATA_HOLDINGS_URL, holdings_p)
    if not midas_p.is_file() and fetch_missing:
        _download(UNR_MIDAS_NA_URL, midas_p)
    if not holdings_p.is_file():
        raise FileNotFoundError(f"Missing UNR DataHoldings.txt in {cache_dir}")

    midas_by_sta: dict[str, list[str]] = {}
    if midas_p.is_file():
        for line in midas_p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 27:
                midas_by_sta[parts[0]] = parts

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    stations: list[GnssStation] = []
    for line in holdings_p.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 11:
            continue
        sta = parts[0]
        lat = float(parts[1])
        lon = float(parts[2])
        if lon > 180.0:
            lon -= 360.0
        hgt = float(parts[3])
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            continue
        dtbeg, dtend, nsol = parts[7], parts[8], int(parts[10])
        if nsol < 100:
            continue

        tenv3_p = cache_dir / f"{sta}.NA.tenv3"
        if not tenv3_p.is_file() and fetch_missing:
            _download(UNR_TENV3_NA_TEMPLATE.format(station=sta), tenv3_p)

        m = midas_by_sta.get(sta)
        vel_tuple = None
        # Prefer recent / overlapping .tenv3 velocity if station was active recently (or has no MIDAS entry)
        if tenv3_p.is_file() and (prefer_recent_tenv3 or m is None):
            try:
                ts = parse_tenv3(tenv3_p)
                if m is None or dtend >= "2022-01-01":
                    vel_tuple = estimate_tenv3_velocity(
                        ts, recent_years=recent_years, window_dates=window_dates
                    )
            except Exception:
                vel_tuple = None

        if vel_tuple is None and m is not None:
            vel_tuple = (
                float(m[8]) * 1000.0,
                float(m[9]) * 1000.0,
                float(m[10]) * 1000.0,
                float(m[11]) * 1000.0,
                float(m[12]) * 1000.0,
                float(m[13]) * 1000.0,
                "midas_na",
            )

        if vel_tuple is None:
            continue

        ve, vn, vu, se, sn, su, src_lbl = vel_tuple
        stations.append(
            GnssStation(
                station_id=sta,
                lat=lat,
                lon=lon,
                height_m=hgt,
                start_date=dtbeg,
                end_date=dtend,
                num_solutions=nsol,
                ve_mm_yr=ve,
                vn_mm_yr=vn,
                vu_mm_yr=vu,
                se_mm_yr=se,
                sn_mm_yr=sn,
                su_mm_yr=su,
                source=src_lbl,
            )
        )
    stations.sort(key=lambda s: s.station_id)
    return stations


def load_stations_from_csv(csv_path: Path) -> list[GnssStation]:
    """Load GNSS station velocities from a simple CSV (offline / custom ground truth)."""
    out: list[GnssStation] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                GnssStation(
                    station_id=row["station_id"].strip(),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    height_m=float(row.get("height_m") or 0.0),
                    start_date=row.get("start_date", ""),
                    end_date=row.get("end_date", ""),
                    num_solutions=int(row.get("num_solutions") or 0),
                    ve_mm_yr=float(row["ve_mm_yr"]),
                    vn_mm_yr=float(row["vn_mm_yr"]),
                    vu_mm_yr=float(row["vu_mm_yr"]),
                    se_mm_yr=float(row.get("se_mm_yr") or 1.0),
                    sn_mm_yr=float(row.get("sn_mm_yr") or 1.0),
                    su_mm_yr=float(row.get("su_mm_yr") or 2.0),
                    source=row.get("source", "csv"),
                )
            )
    return out


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


def render_gnss_benchmark_plot(
    stack_dir: Path,
    matches: list[StationMatch],
    metrics: dict[str, float],
    out_path: Path,
    *,
    baseline_metrics: dict[str, float] | None = None,
    ref_lonlat: tuple[float, float] | None = None,
    insar_sign: float = -1.0,
    ts_station: GnssStation | None = None,
    ts_gnss_dates: list[date] | None = None,
    ts_gnss_los_mm: list[float] | None = None,
    ts_insar_dates: list[date] | None = None,
    ts_insar_los_mm: list[float] | None = None,
    ts_rmse_mm: float | None = None,
    dpi: int = 220,
) -> Path:
    """Render 3-panel publication-grade GNSS ground-truth validation benchmark figure."""
    plots_dir = Path(__file__).resolve().parent / "plots"
    if str(plots_dir) not in sys.path:
        sys.path.insert(0, str(plots_dir))
    from _common import (  # noqa: E402
        INK,
        MUTED,
        NODATA_BG,
        add_editorial_header,
        add_footer,
        add_scalebar,
        build_wgs84_to_raster_ct,
        configure_matplotlib,
        format_date_range,
        format_site_label,
        load_raster,
        load_stack_ledger,
        load_stack_summary,
        make_diverging_cmap,
        save_figure,
    )
    import matplotlib.dates as mdates  # noqa: E402
    import matplotlib.gridspec as gridspec  # noqa: E402
    import matplotlib.patheffects as pe  # noqa: E402
    from matplotlib.cm import ScalarMappable  # noqa: E402
    from matplotlib.colors import Normalize  # noqa: E402
    from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

    plt = configure_matplotlib()
    vel_layer = load_raster(stack_dir / "velocity_mm_yr.tif")
    coh_layer = load_raster(stack_dir / "temporal_coherence.tif")
    summary = load_stack_summary(stack_dir)
    ledger = load_stack_ledger(stack_dir)

    # Display velocity in intuitive geodetic convention (negative = subsidence away from satellite)
    vel_disp = insar_sign * vel_layer.data
    ct = build_wgs84_to_raster_ct(vel_layer.projection_wkt)

    # Focus map tightly on the urban basin containing all GNSS stations + reference region
    pts_utm = [ct.TransformPoint(m.lon, m.lat, 0.0)[:2] for m in matches]
    if ref_lonlat is not None:
        pts_utm.append(ct.TransformPoint(ref_lonlat[0], ref_lonlat[1], 0.0)[:2])
    xs_u = [p[0] for p in pts_utm]
    ys_u = [p[1] for p in pts_utm]
    pad_x = max(7000.0, 0.35 * (max(xs_u) - min(xs_u)))
    pad_y = max(6000.0, 0.35 * (max(ys_u) - min(ys_u)))
    xmin = max(vel_layer.extent[0], min(xs_u) - pad_x)
    xmax = min(vel_layer.extent[1], max(xs_u) + pad_x)
    ymin = max(vel_layer.extent[2], min(ys_u) - pad_y)
    ymax = min(vel_layer.extent[3], max(ys_u) + pad_y)

    # Use inverted diverging map so red/crimson = negative (subsidence), blue = positive (uplift)
    cmap = make_diverging_cmap().reversed()
    max_mag = max(
        [abs(m.gnss_los_mm_yr) for m in matches]
        + [abs(m.insar_los_mm_yr) for m in matches]
        + [100.0]
    )
    vlim = math.ceil((max_mag * 1.15) / 50.0) * 50.0
    norm = Normalize(vmin=-vlim, vmax=vlim, clip=True)

    fig = plt.figure(figsize=(15.4, 9.4))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.22, 1.0],
        height_ratios=[1.12, 0.92],
        left=0.055, right=0.965, top=0.875, bottom=0.165,
        wspace=0.25, hspace=0.34,
    )

    # ---------------- Panel A: Spatial Map + GNSS Stations ----------------
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(NODATA_BG)

    finite = np.isfinite(vel_disp)
    rgba = cmap(norm(np.where(finite, vel_disp, 0.0)))
    c_safe = np.where(np.isfinite(coh_layer.data), coh_layer.data, 0.0)
    alpha = np.clip((c_safe - 0.28) / 0.28, 0.18, 1.0)
    rgba[..., 3] = np.where(finite, alpha, 0.0)
    ax_map.imshow(rgba, extent=vel_layer.extent, origin="upper", interpolation="nearest", zorder=2)

    # Plot UNR GNSS stations as circles filled with GNSS LOS velocity on the IDENTICAL colormap
    map_label_offsets: dict[str, tuple[float, float, str]] = {
        "MXAZ": (950.0, 500.0, "left"),
        "UNVA": (950.0, 450.0, "left"),
        "MMX1": (0.0, -1550.0, "center"),
        "MXMX": (950.0, 750.0, "left"),
        "UTAC": (-950.0, -1050.0, "right"),
        "ICMX": (950.0, -1100.0, "left"),
    }
    for m in matches:
        ux, uy, _ = ct.TransformPoint(m.lon, m.lat, 0.0)
        col = cmap(norm(m.gnss_los_mm_yr))
        ax_map.scatter(
            [ux], [uy], s=145, color=col, edgecolors="#111111", linewidths=1.8, zorder=10
        )
        dx_off, dy_off, ha = map_label_offsets.get(m.station_id, (950.0, 700.0, "left"))
        lbl = f"{m.station_id}\nGNSS {m.gnss_los_mm_yr:+.0f} | InSAR {m.insar_los_mm_yr:+.0f}"
        ax_map.text(
            ux + dx_off,
            uy + dy_off,
            lbl,
            fontsize=7.8,
            fontweight="bold",
            color=INK,
            ha=ha,
            va="center",
            zorder=12,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="#FFFFFF", alpha=0.95)],
        )

    if ref_lonlat is not None:
        rx, ry, _ = ct.TransformPoint(ref_lonlat[0], ref_lonlat[1], 0.0)
        ax_map.scatter(
            [rx], [ry], s=180, marker="*", color="#FFD166", edgecolors="#111111", linewidths=1.3,
            zorder=11, label="Stable Bedrock Ref Pin (0 mm/yr)",
        )
        ax_map.text(
            rx - 900.0, ry + 550.0, "REF PIN\n(Bedrock)",
            fontsize=7.6, fontweight="bold", color=INK, ha="right", va="center", zorder=12,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="#FFFFFF", alpha=0.92)],
        )

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.set_aspect("equal", adjustable="box")
    km_fmt = FuncFormatter(lambda v, _: f"{v / 1000.0:,.0f}")
    ax_map.xaxis.set_major_formatter(km_fmt)
    ax_map.yaxis.set_major_formatter(km_fmt)
    ax_map.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_map.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_map.set_xlabel("UTM Zone 14N Easting (km)")
    ax_map.set_ylabel("UTM Zone 14N Northing (km)")
    ax_map.set_title("A. InSAR LOS Velocity Map & Independent UNR GNSS Stations", loc="left", pad=8)
    for sp in ("top", "right", "bottom", "left"):
        ax_map.spines[sp].set_visible(True)
        ax_map.spines[sp].set_color("#B8C2CC")
    add_scalebar(ax_map, xmin, xmax, ymin, ymax)

    # Colorbar cleanly beneath Panel A with generous clearance above footer
    pos_a = ax_map.get_position()
    cax = fig.add_axes([pos_a.x0 + 0.02, 0.082, pos_a.width - 0.04, 0.017])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("LOS Velocity (mm/yr · negative = subsidence away from satellite)", fontsize=8.8, labelpad=3)
    cb.ax.tick_params(labelsize=8.2)

    # ---------------- Panel B: 1:1 Scatter Plot ----------------
    ax_sc = fig.add_subplot(gs[0, 1])
    gx = np.array([m.gnss_los_mm_yr for m in matches])
    iy = np.array([m.insar_los_mm_yr for m in matches])
    g_err = np.array([m.gnss_los_sigma_mm_yr for m in matches])
    i_err = np.array([m.insar_spatial_std_mm_yr for m in matches])

    lo_lim = min(float(np.min(gx)), float(np.min(iy)), -40.0) - 25.0
    hi_lim = max(float(np.max(gx)), float(np.max(iy)), 25.0) + 32.0
    line_x = np.linspace(lo_lim, hi_lim, 100)
    ax_sc.fill_between(line_x, line_x - 20.0, line_x + 20.0, color="#E4E7EC", alpha=0.45, label="±20 mm/yr band")
    ax_sc.fill_between(line_x, line_x - 10.0, line_x + 10.0, color="#CBD5E1", alpha=0.55, label="±10 mm/yr band")
    ax_sc.plot(line_x, line_x, "--", color="#1B222C", linewidth=1.3, label="1:1 Identity")

    # Show baseline points faded in background if available
    if baseline_metrics is not None and any(m.baseline_insar_los_mm_yr is not None for m in matches):
        bx = [m.gnss_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None]
        by = [m.baseline_insar_los_mm_yr for m in matches if m.baseline_insar_los_mm_yr is not None]
        ax_sc.scatter(
            bx, by, s=46, marker="x", color="#9CA3AF", linewidths=1.3, zorder=4,
            label=f"Baseline unmasked (RMSE {baseline_metrics['rmse_mm_yr']:.1f} mm/yr)",
        )

    ax_sc.errorbar(
        gx, iy, xerr=g_err, yerr=i_err, fmt="o", markersize=7.5,
        color="#B8332A", ecolor="#596574", elinewidth=1.1, capsize=3,
        markeredgecolor="#FFFFFF", markeredgewidth=0.9, zorder=8,
        label=f"Upgraded stack ±1σ spatial (RMSE {metrics['rmse_mm_yr']:.1f} mm/yr)",
    )
    scatter_label_offsets: dict[str, tuple[float, float, str]] = {
        "MMX1": (10.0, 10.0, "left"),
        "ICMX": (-12.0, 9.0, "right"),
        "UNVA": (-26.0, -18.0, "right"),
        "MXMX": (18.0, -24.0, "left"),
        "UTAC": (16.0, -5.0, "left"),
        "MXAZ": (-12.0, 9.0, "right"),
    }
    for m in matches:
        off_x, off_y, ha = scatter_label_offsets.get(
            m.station_id,
            (8.0 if m.gnss_los_mm_yr < -100 else -8.0, 8.0, "left" if m.gnss_los_mm_yr < -100 else "right"),
        )
        use_arrow = m.station_id in {"UNVA", "MXMX", "UTAC"}
        ax_sc.annotate(
            m.station_id,
            (m.gnss_los_mm_yr, m.insar_los_mm_yr),
            xytext=(off_x, off_y),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
            ha=ha,
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#8C98A6", "lw": 0.8} if use_arrow else None,
        )

    kpi_lines = [
        f"N = {metrics['n_stations']} UNR GNSS stations",
        f"RMSE = {metrics['rmse_mm_yr']:.1f} mm/yr (de-biased: {metrics['rmse_debiased_mm_yr']:.1f})",
        f"MAE  = {metrics['mae_mm_yr']:.1f} mm/yr  ·  Bias = {metrics['bias_mm_yr']:+.1f}",
        f"R²   = {metrics['r2']:.3f}  ·  r = {metrics['pearson_r']:.3f}",
    ]
    if baseline_metrics is not None:
        kpi_lines.append(f"Baseline RMSE: {baseline_metrics['rmse_mm_yr']:.1f} → {metrics['rmse_mm_yr']:.1f} mm/yr")
    ax_sc.text(
        0.03, 0.96, "\n".join(kpi_lines),
        transform=ax_sc.transAxes,
        fontsize=8.3, family="monospace", color=INK, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F8F7F4", "edgecolor": "#C8D0DA", "alpha": 0.95},
        zorder=10,
    )
    ax_sc.set_xlim(lo_lim, hi_lim)
    ax_sc.set_ylim(lo_lim, hi_lim)
    ax_sc.grid(True, alpha=0.55)
    ax_sc.set_xlabel("UNR GNSS Projected LOS Velocity (mm/yr)")
    ax_sc.set_ylabel("InSAR Stack LOS Velocity (mm/yr)")
    ax_sc.set_title("B. Ground-Truth Velocity Agreement (InSAR vs. UNR GNSS)", loc="left", pad=8)
    ax_sc.legend(loc="lower right", fontsize=7.6, frameon=True, facecolor="#FFFFFF", framealpha=0.9, edgecolor="#D5DCE4")

    # ---------------- Panel C: Overlapping 2026 Time Series OR Residual Comparison ----------------
    ax_c = fig.add_subplot(gs[1, 1])
    if (
        ts_station is not None
        and ts_gnss_dates
        and ts_gnss_los_mm
        and ts_insar_dates
        and ts_insar_los_mm
    ):
        ax_c.scatter(
            ts_gnss_dates, ts_gnss_los_mm, s=13, color="#1D5B8F", alpha=0.55,
            label=f"UNR Daily GNSS {ts_station.station_id} (projected LOS)", zorder=4,
        )
        ax_c.plot(
            ts_insar_dates, ts_insar_los_mm, "-o", color="#B8332A", linewidth=2.0, markersize=4.2,
            markeredgecolor="#FFFFFF", markeredgewidth=0.6,
            label="Sentinel-1 InSAR Cumulative LOS (6-day cadence)", zorder=6,
        )
        ax_c.axhline(0.0, color="#9CA3AF", linestyle=":", linewidth=0.9)
        ax_c.xaxis.set_major_locator(mdates.MonthLocator())
        ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_c.grid(True, alpha=0.55)
        ax_c.set_ylabel("Cumulative LOS Displacement (mm)")
        ax_c.set_xlabel("2026 Observation Date")
        ax_c.set_title(
            f"C. Live 2026 Time-Series Overlay at {ts_station.station_id} (Mexico City Airport)",
            loc="left", pad=8,
        )
        if ts_rmse_mm is not None:
            ax_c.text(
                0.97, 0.94,
                f"Epoch RMSE = {ts_rmse_mm:.1f} mm\n{len(ts_insar_dates)} SAR epochs · {len(ts_gnss_dates)} GNSS days",
                transform=ax_c.transAxes,
                fontsize=8.0, family="monospace", color=INK, va="top", ha="right",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8F7F4", "edgecolor": "#C8D0DA", "alpha": 0.95},
                zorder=10,
            )
        ax_c.legend(loc="lower left", fontsize=8.0, frameon=True, facecolor="#FFFFFF", framealpha=0.9, edgecolor="#D5DCE4")
    else:
        ids = [m.station_id for m in matches]
        x_idx = np.arange(len(ids))
        res_up = [m.residual_mm_yr for m in matches]
        if baseline_metrics is not None and any(m.baseline_residual_mm_yr is not None for m in matches):
            res_base = [m.baseline_residual_mm_yr or 0.0 for m in matches]
            ax_c.bar(x_idx - 0.18, res_base, width=0.35, color="#CBD5E1", edgecolor="#64748B", label="Baseline residual")
            ax_c.bar(x_idx + 0.18, res_up, width=0.35, color="#B8332A", label="Upgraded residual")
        else:
            ax_c.bar(x_idx, res_up, width=0.5, color="#1D5B8F", label="InSAR − GNSS residual")
        ax_c.axhline(0.0, color=INK, linewidth=0.9)
        ax_c.set_xticks(x_idx)
        ax_c.set_xticklabels(ids, fontweight="bold")
        ax_c.set_ylabel("Velocity Residual (mm/yr)")
        ax_c.set_title("C. Per-Station Velocity Residuals (InSAR − UNR GNSS)", loc="left", pad=8)
        ax_c.grid(axis="y", alpha=0.55)
        ax_c.legend(loc="best", fontsize=8.0)

    title = f"GNSS Ground-Truth Benchmark — {format_site_label(stack_dir, ledger)}"
    dem_used = summary.get("ramp_removal", {}).get("dem_applied", False)
    ref_used = summary.get("reference_region", {}).get("applied", False)
    subtitle = (
        f"{format_date_range(ledger, summary)}  ·  "
        f"Copernicus GLO-30 DEM Tropo: {'ON' if dem_used else 'OFF'}  ·  "
        f"Bedrock Reference Pin: {'ON' if ref_used else 'OFF'}"
    )
    add_editorial_header(fig, title, subtitle, left=0.055, top=0.962)
    add_footer(
        fig,
        "Ground truth: Nevada Geodetic Laboratory (UNR) North-America-fixed (NA) GNSS velocities & daily IGS20 .tenv3 positions projected onto exact Sentinel-1 zero-Doppler LOS.",
        left=0.055,
        bottom=0.012,
    )

    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved GNSS validation benchmark plot -> {out_path}")
    return out_path


def validate_stack_against_gnss(
    *,
    stack_dir: Path,
    gnss_cache_dir: Path | None = None,
    gnss_csv: Path | None = None,
    slc_h5: Path | None = None,
    incidence_deg: float = 34.0,
    heading_deg: float = -169.0,
    right_looking: bool = True,
    sample_radius_m: float = 200.0,
    min_coh: float = 0.35,
    insar_sign: float = -1.0,
    fetch_missing: bool = False,
    baseline_stack_dir: Path | None = None,
    products_dir: Path | None = None,
    dem_path: Path | None = None,
    out_json: Path | None = None,
    out_csv: Path | None = None,
    plot_out: Path | None = None,
) -> dict[str, Any]:
    """Run full InSAR-vs-GNSS validation and optionally render the benchmark figure."""
    from osgeo import gdal, osr  # type: ignore

    gdal.UseExceptions()
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

    orbit = load_orbit_from_opera_h5(resolved_slc_h5) if (resolved_slc_h5 is not None and resolved_slc_h5.is_file()) else None
    if gnss_csv is not None:
        stations = load_stations_from_csv(gnss_csv)
    else:
        stations = load_unr_stations_in_bbox(
            gnss_cache_dir or Path("insar_work/gnss/unr"),
            bbox_wgs84,
            fetch_missing=fetch_missing,
            window_dates=window_dates,
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
            u_e, u_n, u_u, inc = compute_los_vector_from_angles(incidence_deg, heading_deg, right_looking=right_looking)

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
            tenv_p = gnss_cache_dir / f"{m.station_id}.NA.tenv3"
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

    if plot_out is not None:
        render_gnss_benchmark_plot(
            stack_dir=stack_dir,
            matches=matches,
            metrics=metrics,
            out_path=plot_out,
            baseline_metrics=baseline_metrics,
            ref_lonlat=ref_lonlat,
            insar_sign=insar_sign,
            ts_station=ts_station,
            ts_gnss_dates=ts_gnss_dates,
            ts_gnss_los_mm=ts_gnss_los_mm,
            ts_insar_dates=ts_insar_dates,
            ts_insar_los_mm=ts_insar_los_mm,
            ts_rmse_mm=ts_rmse_mm,
        )

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate InSAR stack velocity against UNR/NGL GNSS ground truth.")
    p.add_argument("--stack-dir", required=True, type=Path, help="Stack directory with velocity_mm_yr.tif and temporal_coherence.tif")
    p.add_argument("--baseline-stack-dir", type=Path, default=None, help="Optional baseline stack dir for before/after comparison")
    p.add_argument("--products-dir", type=Path, default=None, help="Optional pair products dir for 2026 cumulative time-series overlay")
    p.add_argument("--dem", type=Path, default=None, help="Optional aligned DEM GeoTIFF used during time-series overlay")
    p.add_argument("--gnss-cache-dir", type=Path, default=Path("insar_work/gnss/unr"), help="Directory with UNR DataHoldings.txt, midas.NA.txt, *.NA.tenv3")
    p.add_argument("--gnss-csv", type=Path, default=None, help="Optional local GNSS station CSV instead of UNR cache")
    p.add_argument("--slc-h5", type=Path, default=None, help="OPERA CSLC-S1 .h5 file or raw directory containing .h5 granules for exact zero-Doppler orbit LOS vector")
    p.add_argument("--incidence-deg", type=float, default=34.0, help="Fallback radar incidence angle (deg) if --slc-h5 omitted")
    p.add_argument("--heading-deg", type=float, default=-169.0, help="Fallback satellite heading (deg CW from N) if --slc-h5 omitted")
    p.add_argument("--sample-radius-m", type=float, default=200.0, help="Spatial median radius around each GNSS station (m)")
    p.add_argument("--min-coh", type=float, default=0.35, help="Minimum temporal coherence when sampling InSAR velocity")
    p.add_argument("--fetch-missing", action="store_true", help="Download missing UNR files if not cached locally")
    p.add_argument("--out-json", type=Path, default=None, help="Output validation metrics JSON path")
    p.add_argument("--out-csv", type=Path, default=None, help="Output per-station comparison CSV path")
    p.add_argument("--plot-out", type=Path, default=None, help="Output publication-grade PNG benchmark plot path")
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_gdal_plot_runtime(__file__, argv)
    args = build_arg_parser().parse_args(argv)
    report = validate_stack_against_gnss(
        stack_dir=args.stack_dir,
        gnss_cache_dir=args.gnss_cache_dir,
        gnss_csv=args.gnss_csv,
        slc_h5=args.slc_h5,
        incidence_deg=args.incidence_deg,
        heading_deg=args.heading_deg,
        sample_radius_m=args.sample_radius_m,
        min_coh=args.min_coh,
        fetch_missing=args.fetch_missing,
        baseline_stack_dir=args.baseline_stack_dir,
        products_dir=args.products_dir,
        dem_path=args.dem,
        out_json=args.out_json,
        out_csv=args.out_csv,
        plot_out=args.plot_out,
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
