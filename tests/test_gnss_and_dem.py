#!/usr/bin/env python3
"""Unit and integration tests for scripts/fetch_dem.py and scripts/validate_gnss.py."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.fetch_dem import (
    build_aligned_dem,
    cop30_tile_url,
    read_raster_grid_and_wgs84_bbox,
    tile_name_for_latlon,
    tiles_for_bbox,
)
from scripts.validate_gnss import (
    compute_los_vector_from_angles,
    compute_validation_metrics,
    estimate_tenv3_velocity,
    parse_tenv3,
)


def _write_geotiff(path: Path, arr: np.ndarray, *, epsg: int, origin_x: float, origin_y: float, px_x: float, px_y: float) -> None:
    h, w = arr.shape
    raw_bin = path.with_suffix(".envi.bin")
    raw_hdr = path.with_suffix(".envi.hdr")
    arr.astype(np.float32).tofile(raw_bin)
    if epsg == 32614:
        map_info = f"map info = {{UTM, 1.0, 1.0, {origin_x}, {origin_y}, {px_x}, {abs(px_y)}, 14, North, WGS-84, units=Meters}}"
    else:
        map_info = f"map info = {{Geographic Lat/Lon, 1.0, 1.0, {origin_x}, {origin_y}, {px_x}, {abs(px_y)}, WGS-84, units=Degrees}}"
    raw_hdr.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {w}",
                f"lines = {h}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 0",
                map_info,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["gdal_translate", "-q", "-of", "GTiff", "-a_srs", f"EPSG:{epsg}", "-a_nodata", "nan", str(raw_bin), str(path)],
        check=True,
    )
    raw_bin.unlink(missing_ok=True)
    raw_hdr.unlink(missing_ok=True)


class TestFetchDemAndGnssValidation(unittest.TestCase):
    def test_cop30_tile_resolution(self) -> None:
        self.assertEqual(tile_name_for_latlon(19, -100), "Copernicus_DSM_COG_10_N19_00_W100_00_DEM")
        self.assertEqual(tile_name_for_latlon(-23, 15), "Copernicus_DSM_COG_10_S23_00_E015_00_DEM")
        self.assertIn("N19_00_W099_00", cop30_tile_url(19, -99))
        tiles = tiles_for_bbox(-99.84, 19.31, -98.90, 19.75)
        self.assertEqual(tiles, [(19, -100), (19, -99)])

    def test_build_aligned_dem_reprojects_wgs84_to_utm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            ref_path = tp / "ref_utm.tif"
            dem_wgs = tp / "dem_wgs84.tif"
            dem_aligned = tp / "dem_utm.tif"

            _write_geotiff(ref_path, np.zeros((40, 50), dtype=np.float32), epsg=32614, origin_x=480000.0, origin_y=2160000.0, px_x=30.0, px_y=-30.0)
            # Synthetic WGS84 DEM covering Mexico City (lon -99.30..-99.00, lat 19.45..19.65)
            yy, xx = np.mgrid[0:60, 0:60]
            elev = 2200.0 + 10.0 * xx + 5.0 * yy
            _write_geotiff(dem_wgs, elev, epsg=4326, origin_x=-99.35, origin_y=19.65, px_x=0.005, px_y=-0.005)

            grid = read_raster_grid_and_wgs84_bbox(ref_path)
            build_aligned_dem([dem_wgs], dem_aligned, ref_grid=grid)

            aligned_grid = read_raster_grid_and_wgs84_bbox(dem_aligned)
            self.assertEqual(aligned_grid["width"], 50)
            self.assertEqual(aligned_grid["height"], 40)
            self.assertAlmostEqual(aligned_grid["geotransform"][0], 480000.0, places=3)
            self.assertAlmostEqual(aligned_grid["geotransform"][3], 2160000.0, places=3)

    def test_tenv3_parser_and_robust_velocity_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tenv_p = Path(tmp) / "TEST.NA.tenv3"
            lines = [
                "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)"
            ]
            d0 = date(2024, 1, 1)
            mjd0 = (d0 - date(1858, 11, 17)).days
            rng = np.random.default_rng(7)
            for k in range(365):
                t_yr = k / 365.25
                e = 0.005 * t_yr + rng.normal(0.0, 0.0008)      # +5 mm/yr
                n = -0.012 * t_yr + rng.normal(0.0, 0.0008)     # -12 mm/yr
                u = -0.220 * t_yr + rng.normal(0.0, 0.0025)     # -220 mm/yr
                if k == 100:
                    u += 0.08  # Inject 80 mm outlier spike
                decyr = 2024.0 + t_yr
                lines.append(
                    f"TEST 24JAN01 {decyr:.4f} {mjd0 + k} 2295 1 -99.1 100 {e:.6f} 2150000 {n:.6f} 2230 {u:.6f} "
                    f"0.0000 0.0008 0.0008 0.0025 0.0 0.0 0.0 19.4300 -99.1000 {2230.0 + u:.4f}"
                )
            tenv_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

            ts = parse_tenv3(tenv_p)
            self.assertEqual(len(ts["decyr"]), 365)
            ve, vn, vu, se, sn, su, lbl = estimate_tenv3_velocity(ts, recent_years=3.0)
            self.assertEqual(lbl, "tenv3_full")
            self.assertAlmostEqual(ve, 5.0, delta=1.5)
            self.assertAlmostEqual(vn, -12.0, delta=1.5)
            self.assertAlmostEqual(vu, -220.0, delta=3.0)
            self.assertGreater(su, 0.4)

    def test_los_projection_and_validation_metrics(self) -> None:
        u_e, u_n, u_u, inc = compute_los_vector_from_angles(34.0, -169.0, right_looking=True)
        self.assertAlmostEqual(math.hypot(u_e, u_n, u_u), 1.0, places=6)
        self.assertGreater(u_e, 0.4)  # Descending right-looking looks West -> ground->sat has +East component
        self.assertAlmostEqual(u_u, math.cos(math.radians(34.0)), places=6)
        self.assertAlmostEqual(inc, 34.0, places=6)

        g = np.array([0.0, -25.0, -100.0, -220.0])
        i = np.array([1.0, -23.0, -102.0, -217.0])
        m = compute_validation_metrics(g, i)
        self.assertEqual(m["n_stations"], 4)
        self.assertLess(m["rmse_mm_yr"], 2.5)
        self.assertGreater(m["r2"], 0.999)
        self.assertGreater(m["pearson_r"], 0.999)

    def test_validate_gnss_cli_end_to_end_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            stack_dir = tp / "stack"
            stack_dir.mkdir()
            h, w = 80, 80
            # Synthetic linear gradient velocity in range-increase convention (+subsidence)
            yy, xx = np.mgrid[0:h, 0:w]
            vel_range_inc = 3.0 * xx  # 0 to +240 mm/yr range increase => 0 to -240 mm/yr toward satellite
            coh = np.full((h, w), 0.85, dtype=np.float32)
            unc = np.full((h, w), 3.0, dtype=np.float32)

            _write_geotiff(stack_dir / "velocity_mm_yr.tif", vel_range_inc, epsg=32614, origin_x=480000.0, origin_y=2155000.0, px_x=30.0, px_y=-30.0)
            _write_geotiff(stack_dir / "temporal_coherence.tif", coh, epsg=32614, origin_x=480000.0, origin_y=2155000.0, px_x=30.0, px_y=-30.0)
            _write_geotiff(stack_dir / "velocity_uncertainty_mm_yr.tif", unc, epsg=32614, origin_x=480000.0, origin_y=2155000.0, px_x=30.0, px_y=-30.0)

            # Build 4 synthetic stations inside grid with exact projected LOS rates matching vel_range_inc
            _, _, u_u, _ = compute_los_vector_from_angles(34.0, -169.0, right_looking=True)
            csv_p = tp / "gnss.csv"
            rows = []
            for idx, px in enumerate((12, 30, 50, 68)):
                ux = 480000.0 + (px + 0.5) * 30.0
                uy = 2155000.0 - 40.5 * 30.0
                tr = subprocess.run(
                    ["gdaltransform", "-s_srs", "EPSG:32614", "-t_srs", "EPSG:4326"],
                    input=f"{ux} {uy}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                )
                lon_s, lat_s, *_ = tr.stdout.strip().split()
                expected_los_toward = -3.0 * px
                vu = expected_los_toward / u_u
                rows.append({
                    "station_id": f"ST0{idx}",
                    "lat": lat_s,
                    "lon": lon_s,
                    "height_m": "2250",
                    "ve_mm_yr": "0.0",
                    "vn_mm_yr": "0.0",
                    "vu_mm_yr": f"{vu:.3f}",
                    "se_mm_yr": "0.5",
                    "sn_mm_yr": "0.5",
                    "su_mm_yr": "1.5",
                })
            with open(csv_p, "w", encoding="utf-8", newline="") as fh:
                w_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w_csv.writeheader()
                w_csv.writerows(rows)

            out_json = tp / "val.json"
            out_csv = tp / "val.csv"
            map_out = tp / "figs" / "val_map.png"
            scatter_out = tp / "figs" / "val_scatter.png"
            ts_out = tp / "figs" / "val_ts.png"
            baseline_csv = tp / "baseline.csv"
            with open(baseline_csv, "w", encoding="utf-8", newline="") as fh:
                w_base = csv.DictWriter(fh, fieldnames=["station_id", "baseline_insar_los_mm_yr"])
                w_base.writeheader()
                for idx, px in enumerate((12, 30, 50, 68)):
                    w_base.writerow({"station_id": f"ST0{idx}", "baseline_insar_los_mm_yr": f"{-2.5 * px:.2f}"})

            subprocess.run(
                [
                    "/opt/homebrew/bin/python3",
                    str(Path(__file__).resolve().parent.parent / "scripts" / "validate_gnss.py"),
                    "--stack-dir", str(stack_dir),
                    "--gnss-csv", str(csv_p),
                    "--baseline-csv", str(baseline_csv),
                    "--incidence-deg", "34.0",
                    "--heading-deg", "-169.0",
                    "--sample-radius-m", "90.0",
                    "--out-json", str(out_json),
                    "--out-csv", str(out_csv),
                    "--map-out", str(map_out),
                    "--scatter-out", str(scatter_out),
                    "--timeseries-out", str(ts_out),
                    "--dpi", "120",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            res = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(res["metrics"]["n_stations"], 4)
            self.assertIsNotNone(res["baseline_metrics"])
            self.assertLess(res["metrics"]["rmse_mm_yr"], 2.0)
            self.assertGreater(res["metrics"]["r2"], 0.999)
            self.assertTrue(out_csv.is_file())
            for fig_p in (map_out, scatter_out, ts_out):
                self.assertTrue(fig_p.is_file())
                self.assertGreater(fig_p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
