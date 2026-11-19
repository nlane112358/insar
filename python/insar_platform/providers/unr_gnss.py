"""UNR / Nevada Geodetic Laboratory GNSS station fetchers, .tenv3 parsers, and CSV loader."""

from __future__ import annotations

import csv
import math
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np

UNR_DATA_HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
UNR_MIDAS_NA_URL = "https://geodesy.unr.edu/velocities/midas.NA.txt"
UNR_TENV3_NA_TEMPLATE = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/NA/{station}.NA.tenv3"
UNR_MIDAS_TEMPLATE = "https://geodesy.unr.edu/velocities/midas.{frame}.txt"
UNR_TENV3_TEMPLATE = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/{frame}/{station}.{frame}.tenv3"


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


def resolve_gnss_frame(gnss_frame: str, bbox_wgs84: tuple[float, float, float, float]) -> str:
    """Resolve 'auto' to 'EU' for European AOIs and 'NA' otherwise, or return explicit plate code."""
    cleaned = (gnss_frame or "auto").strip().upper()
    if cleaned and cleaned != "AUTO":
        return cleaned
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    lon_c = 0.5 * (min_lon + max_lon)
    lat_c = 0.5 * (min_lat + max_lat)
    if -25.0 <= lon_c <= 45.0 and 34.0 <= lat_c <= 72.0:
        return "EU"
    return "NA"


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
    gnss_frame: str = "auto",
) -> list[GnssStation]:
    """Load UNR GNSS stations overlapping bbox from cached (or freshly fetched) UNR files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = resolve_gnss_frame(gnss_frame, bbox_wgs84)
    holdings_p = cache_dir / "DataHoldings.txt"
    midas_p = cache_dir / f"midas.{frame}.txt"

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
        _download(UNR_MIDAS_TEMPLATE.format(frame=frame), midas_p)
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

        tenv3_p = cache_dir / f"{sta}.{frame}.tenv3"
        if not tenv3_p.is_file() and fetch_missing:
            _download(UNR_TENV3_TEMPLATE.format(frame=frame, station=sta), tenv3_p)

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
                f"midas_{frame.lower()}",
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
