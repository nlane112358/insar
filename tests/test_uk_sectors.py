#!/usr/bin/env python3
"""Unit and integration tests for UK corridor presets, UK sector analytics, and NISAR GNSS validation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.fetch_corridors import PRESETS, classify_way, elements_to_grouped_geojson
from scripts.uk_sector_analytics import summarize_corridor_sectors
from scripts.validate_gnss import (
    compute_los_vector_from_orbit,
    load_orbit_from_h5,
    resolve_gnss_frame,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_utm30_f32(
    path: Path,
    arr: np.ndarray,
    *,
    origin_x: float = 690000.0,
    origin_y: float = 5715000.0,
    px: float = 40.0,
) -> None:
    h, w = arr.shape
    raw_bin = path.with_suffix(".envi.bin")
    raw_hdr = path.with_suffix(".envi.hdr")
    arr.astype(np.float32).tofile(raw_bin)
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
                f"map info = {{UTM, 1.0, 1.0, {origin_x}, {origin_y}, {px}, {px}, 30, North, WGS-84, units=Meters}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["gdal_translate", "-q", "-of", "GTiff", "-a_srs", "EPSG:32630", "-a_nodata", "nan", str(raw_bin), str(path)],
        check=True,
    )
    raw_bin.unlink(missing_ok=True)
    raw_hdr.unlink(missing_ok=True)


def _utm30_to_wgs84(ux: float, uy: float) -> list[float]:
    proc = subprocess.run(
        ["gdaltransform", "-s_srs", "EPSG:32630", "-t_srs", "EPSG:4326"],
        input=f"{ux} {uy}\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lon_s, lat_s, *_ = proc.stdout.strip().split()
    return [round(float(lon_s), 6), round(float(lat_s), 6)]


class TestUkCorridorsAndSectors(unittest.TestCase):
    def test_uk_presets_and_checked_in_assets_valid(self) -> None:
        self.assertIn("london_thames", PRESETS)
        self.assertIn("east_midlands_coalfields", PRESETS)
        for rel in (
            "data/corridors/london_thames_corridors.geojson",
            "data/corridors/east_midlands_coalfields_corridors.geojson",
            "data/zones/london_thames_zones.geojson",
            "data/zones/east_midlands_coalfields_zones.geojson",
        ):
            p = REPO_ROOT / rel
            self.assertTrue(p.is_file(), msg=f"Missing UK GeoJSON asset: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data.get("type"), "FeatureCollection")
            self.assertGreater(len(data.get("features", [])), 10)

    def test_classify_way_amp8_water_and_tram_and_relation_stitching(self) -> None:
        self.assertEqual(classify_way({"railway": "tram", "name": "NET Line 1"}), ("light_rail", "NET Line 1"))
        self.assertEqual(
            classify_way({"man_made": "water_works", "name": "Hampton Water Works"}),
            ("water_works", "Hampton Water Works"),
        )
        self.assertEqual(
            classify_way({"man_made": "pipeline", "substance": "water", "name": "Derwent Aqueduct"}),
            ("water_works", "Derwent Aqueduct"),
        )

        # Synthetic Overpass payload with split relation multipolygon + balanced classes
        osm_data = {
            "elements": [
                {"type": "node", "id": 1, "lon": -1.62, "lat": 53.06},
                {"type": "node", "id": 2, "lon": -1.60, "lat": 53.06},
                {"type": "node", "id": 3, "lon": -1.60, "lat": 53.08},
                {"type": "node", "id": 4, "lon": -1.62, "lat": 53.08},
                {"type": "way", "id": 10, "nodes": [1, 2, 3]},
                {"type": "way", "id": 11, "nodes": [3, 4, 1]},
                {
                    "type": "relation",
                    "id": 100,
                    "tags": {"natural": "water", "water": "reservoir", "name": "Carsington Water"},
                    "members": [
                        {"type": "way", "ref": 10, "role": "outer"},
                        {"type": "way", "ref": 11, "role": "outer"},
                    ],
                },
                {
                    "type": "way",
                    "id": 20,
                    "nodes": [1, 2],
                    "tags": {"railway": "rail", "name": "Robin Hood Line"},
                },
                {
                    "type": "way",
                    "id": 21,
                    "nodes": [2, 3],
                    "tags": {"man_made": "water_works", "name": "Ogston WTW"},
                },
            ]
        }
        fc = elements_to_grouped_geojson(osm_data, (-1.65, 52.86, -0.96, 53.36), max_features=3, balance_classes=True)
        classes = {f["properties"]["asset_class"]: f for f in fc["features"]}
        self.assertEqual(set(classes.keys()), {"rail", "reservoir", "water_works"})
        self.assertEqual(classes["reservoir"]["geometry"]["type"], "Polygon")

    def test_uk_sector_analytics_rollup_and_insurance_zones(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            stack_dir = tp / "stack"
            stack_dir.mkdir()
            h, w = 60, 60
            ox, oy, px = 690000.0, 5715000.0, 40.0

            yy, xx = np.mgrid[0:h, 0:w]
            # Left half uplifting (-6 mm), right half settling (+8 mm)
            cum = np.where(xx < 30, -6.0, 8.0).astype(np.float32)
            vel = (cum * (365.25 / 24.0)).astype(np.float32)
            coh = np.full((h, w), 0.78, dtype=np.float32)
            snr = np.full((h, w), 3.1, dtype=np.float32)

            _write_utm30_f32(stack_dir / "velocity_mm_yr.tif", vel, origin_x=ox, origin_y=oy, px=px)
            _write_utm30_f32(stack_dir / "cumulative_disp_mm.tif", cum, origin_x=ox, origin_y=oy, px=px)
            _write_utm30_f32(stack_dir / "temporal_coherence.tif", coh, origin_x=ox, origin_y=oy, px=px)
            _write_utm30_f32(stack_dir / "velocity_snr.tif", snr, origin_x=ox, origin_y=oy, px=px)

            (stack_dir / "stack_summary.json").write_text(
                json.dumps(
                    {
                        "pair_count": 2,
                        "calendar_span_days": 24.0,
                        "maturity_gating": {"stack_mature": False, "actual_span_days": 24.0, "min_alert_span_days": 60.0},
                    }
                ),
                encoding="utf-8",
            )
            (stack_dir / "stack_ledger.json").write_text(
                json.dumps({"stream_key": "NISAR_T173_F028_A", "pair_ids": ["2026-10-24_2026-11-05", "2026-11-05_2026-11-17"]}),
                encoding="utf-8",
            )
            (stack_dir / "corridor_report.geojson").write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "asset_id": "rail_elizabeth_line",
                                    "name": "Elizabeth Line",
                                    "asset_class": "rail",
                                    "valid_pixels": 400,
                                    "pixel_count": 420,
                                    "coverage_ratio": 0.95,
                                    "mean_coherence": 0.81,
                                    "mean_velocity_mm_yr": -15.0,
                                    "p95_abs_velocity_mm_yr": 45.0,
                                    "mean_cumulative_disp_mm": -1.0,
                                    "max_abs_cumulative_disp_mm": 3.0,
                                    "risk_level": "warning",
                                    "maturity_gated": True,
                                },
                                "geometry": None,
                            },
                            {
                                "type": "Feature",
                                "properties": {
                                    "asset_id": "water_works_beckton_stw",
                                    "name": "Beckton STW",
                                    "asset_class": "water_works",
                                    "valid_pixels": 250,
                                    "pixel_count": 260,
                                    "coverage_ratio": 0.96,
                                    "mean_coherence": 0.76,
                                    "mean_velocity_mm_yr": 20.0,
                                    "p95_abs_velocity_mm_yr": 60.0,
                                    "mean_cumulative_disp_mm": 1.3,
                                    "max_abs_cumulative_disp_mm": 4.0,
                                    "risk_level": "warning",
                                    "maturity_gated": True,
                                },
                                "geometry": None,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            # Build a district polygon spanning both uplift and settlement zones
            p0 = _utm30_to_wgs84(ox + 10 * px, oy - 10 * px)
            p1 = _utm30_to_wgs84(ox + 50 * px, oy - 10 * px)
            p2 = _utm30_to_wgs84(ox + 50 * px, oy - 50 * px)
            p3 = _utm30_to_wgs84(ox + 10 * px, oy - 50 * px)
            zones_p = tp / "zones.geojson"
            zones_p.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"asset_id": "E09000025", "name": "Newham", "asset_class": "local_authority_district"},
                                "geometry": {"type": "Polygon", "coordinates": [[p0, p1, p2, p3, p0]]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_json = tp / "uk_report.json"
            out_zones = tp / "uk_zones.csv"
            out_sectors = tp / "uk_sectors.csv"
            # Test in-process sector rollup + end-to-end CLI (auto-resolving GDAL runtime)
            corridor_feats = json.loads((stack_dir / "corridor_report.geojson").read_text(encoding="utf-8"))["features"]
            sec_rollup = summarize_corridor_sectors(corridor_feats)
            self.assertEqual(sec_rollup["uk_rail"]["covered_assets"], 1)
            self.assertEqual(sec_rollup["water_amp8"]["covered_assets"], 1)

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "uk_sector_analytics.py"),
                    "--stack-dir", str(stack_dir),
                    "--profile", "london_thames",
                    "--zones", str(zones_p),
                    "--out-json", str(out_json),
                    "--out-zones-csv", str(out_zones),
                    "--out-sectors-csv", str(out_sectors),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rep = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertFalse(rep["scene_lband_metrics"]["stack_mature"])
            self.assertEqual(rep["sectors"]["uk_rail"]["covered_assets"], 1)
            self.assertEqual(rep["sectors"]["water_amp8"]["covered_assets"], 1)
            self.assertEqual(len(rep["home_insurance_zones"]), 1)
            z = rep["home_insurance_zones"][0]
            self.assertEqual(z["name"], "Newham")
            self.assertEqual(z["exposure_tier"], "provisional_immature")
            self.assertAlmostEqual(z["p95_p5_spread_mm"], 14.0, delta=0.5)
            self.assertGreater(z["uplift_pct"], 40.0)
            self.assertGreater(z["settlement_pct"], 40.0)
            self.assertTrue(out_json.is_file() and out_zones.is_file() and out_sectors.is_file())

    def test_nisar_orbit_los_and_uk_eu_frame_resolution(self) -> None:
        self.assertEqual(resolve_gnss_frame("auto", (-0.62, 51.31, 0.16, 51.69)), "EU")
        self.assertEqual(resolve_gnss_frame("auto", (-99.4, 19.2, -98.9, 19.7)), "NA")
        self.assertEqual(resolve_gnss_frame("NA", (-0.62, 51.31, 0.16, 51.69)), "NA")

        # Synthetic NISAR GSLC orbit state vector (~747 km altitude ascending, left-looking at UK)
        with tempfile.TemporaryDirectory() as td:
            h5p = Path(td) / "nisar_orbit.h5"
            t = np.linspace(-30.0, 30.0, 11)
            # Target near London (51.5N, 0.0E), satellite slightly to the EAST (left-looking from ascending track flying NNW)
            r_sat = 6371000.0 + 747000.0
            lat_s = np.radians(51.5 + 0.06 * t)
            lon_s = np.radians(4.5 - 0.015 * t)
            px = r_sat * np.cos(lat_s) * np.cos(lon_s)
            py = r_sat * np.cos(lat_s) * np.sin(lon_s)
            pz = r_sat * np.sin(lat_s)
            vx = np.gradient(px, t)
            vy = np.gradient(py, t)
            vz = np.gradient(pz, t)
            with h5py.File(h5p, "w") as f:
                g = f.require_group("/science/LSAR/GSLC/metadata/orbit")
                g.create_dataset("time", data=t)
                g.create_dataset("position", data=np.column_stack([px, py, pz]))
                g.create_dataset("velocity", data=np.column_stack([vx, vy, vz]))

            orb = load_orbit_from_h5(h5p)
            u_e, u_n, u_u, inc = compute_los_vector_from_orbit(orb, 51.5, 0.0, 20.0)
            self.assertAlmostEqual(math.hypot(u_e, u_n, u_u), 1.0, places=6)
            self.assertGreater(u_e, 0.3)  # Satellite is east of target -> ground->sat points +East
            self.assertGreater(inc, 15.0)
            self.assertLess(inc, 50.0)


if __name__ == "__main__":
    unittest.main()
