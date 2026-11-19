import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from insar_platform.pipelines.ingestion import load_sites_config, sync_site
from insar_platform.pipelines.stacking import StackConfig, sync_stack


REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_BIN = REPO_ROOT / "build" / "insar_stack"

# Synthetic UTM Zone 14N grid (Mexico City vicinity: ~99.0W, 19.44N)
WIDTH = 120
HEIGHT = 100
PIXEL_M = 20.0
ORIGIN_X = 500000.0
ORIGIN_Y = 2150000.0


def _write_geotiff_f32(path: Path, arr: np.ndarray) -> None:
    """Write a float32 EPSG:32614 GeoTIFF via ENVI header + gdal_translate."""
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
                f"map info = {{UTM, 1.0, 1.0, {ORIGIN_X}, {ORIGIN_Y}, {PIXEL_M}, {PIXEL_M}, 14, North, WGS-84, units=Meters}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["gdal_translate", "-q", "-of", "GTiff", "-a_srs", "EPSG:32614", "-a_nodata", "nan", str(raw_bin), str(path)],
        check=True,
    )
    raw_bin.unlink(missing_ok=True)
    raw_hdr.unlink(missing_ok=True)


def _read_cog_f32(path: Path) -> np.ndarray:
    """Decode any compressed GDAL Float32 COG into a float64 NumPy array via ENVI."""
    tmp_bin = path.with_suffix(".tmpread.bin")
    subprocess.run(
        ["gdal_translate", "-q", "-of", "ENVI", str(path), str(tmp_bin)],
        check=True,
    )
    data = np.fromfile(tmp_bin, dtype=np.float32).astype(np.float64)
    tmp_bin.unlink(missing_ok=True)
    tmp_bin.with_suffix(".hdr").unlink(missing_ok=True)
    return data.reshape((HEIGHT, WIDTH))


def _utm_to_wgs84(px: float, py: float) -> list[float]:
    utm_x = ORIGIN_X + (px + 0.5) * PIXEL_M
    utm_y = ORIGIN_Y - (py + 0.5) * PIXEL_M
    proc = subprocess.run(
        ["gdaltransform", "-s_srs", "EPSG:32614", "-t_srs", "EPSG:4326"],
        input=f"{utm_x} {utm_y}\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lon_s, lat_s, *_ = proc.stdout.strip().split()
    return [float(lon_s), float(lat_s)]


@unittest.skipUnless(STACK_BIN.is_file(), f"Missing built binary: {STACK_BIN}")
class StackTimeseriesTest(unittest.TestCase):
    def _make_pair(
        self,
        products_dir: Path,
        stream_key: str,
        ref_date: str,
        sec_date: str,
        disp: np.ndarray,
        coh: np.ndarray,
    ) -> Path:
        pdir = products_dir / f"{stream_key}_{ref_date}_{sec_date}"
        pdir.mkdir(parents=True, exist_ok=True)
        _write_geotiff_f32(pdir / "displacement_mm.tif", disp)
        _write_geotiff_f32(pdir / "coherence.tif", coh)
        return pdir

    def test_ramp_removal_preserves_subsidence_bowl_and_ignores_low_coh_spike(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            u = 2.0 * xx / (WIDTH - 1) - 1.0
            v = 2.0 * yy / (HEIGHT - 1) - 1.0

            ramp = 25.0 + 40.0 * u - 30.0 * v
            bowl = -30.0 * np.exp(-(((xx - 60.0) ** 2 + (yy - 50.0) ** 2) / (2.0 * 7.0**2)))
            disp = ramp + bowl
            coh = np.full((HEIGHT, WIDTH), 0.9, dtype=np.float32)

            # Inject unwrapping spike masked by low coherence
            disp[10:18, 10:18] += 200.0
            coh[10:18, 10:18] = 0.15

            disp_p = tmp_p / "disp.tif"
            coh_p = tmp_p / "coh.tif"
            state_p = tmp_p / "state.tif"
            out_dir = tmp_p / "out"
            _write_geotiff_f32(disp_p, disp)
            _write_geotiff_f32(coh_p, coh)

            subprocess.run(
                [
                    str(STACK_BIN),
                    "--state",
                    str(state_p),
                    "--disp",
                    str(disp_p),
                    "--coh",
                    str(coh_p),
                    "--t0-days",
                    "0",
                    "--t1-days",
                    "6",
                    "--ramp-order",
                    "1",
                    "--min-coh",
                    "0.35",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            cleaned = _read_cog_f32(out_dir / "ramp_removed_disp_mm.tif")
            dist = np.hypot(xx - 60.0, yy - 50.0)
            bg_mask = (dist > 30.0) & (coh >= 0.35)
            self.assertLess(np.max(np.abs(cleaned[bg_mask])), 1.0)
            self.assertAlmostEqual(float(cleaned[50, 60]), -30.0, delta=1.5)

    def test_incremental_equivalence_accel_dropout_and_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prod_dir = tmp_p / "products"
            out_inc = tmp_p / "stack_inc"
            out_batch = tmp_p / "stack_batch"

            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            u = 2.0 * xx / (WIDTH - 1) - 1.0
            v = 2.0 * yy / (HEIGHT - 1) - 1.0

            lin_mask = (yy >= 20) & (yy < 45) & (xx >= 20) & (xx < 55)
            acc_mask = (yy >= 60) & (yy < 85) & (xx >= 65) & (xx < 95)

            dates = [
                ("2026-10-06", "2026-10-12", 0.0, 6.0),
                ("2026-10-12", "2026-10-18", 6.0, 12.0),
                ("2026-10-18", "2026-10-24", 12.0, 18.0),
                ("2026-10-24", "2026-10-30", 18.0, 24.0),
                ("2026-10-30", "2026-11-05", 24.0, 30.0),
            ]
            # Build pairs 2..5 initially (then 4..5, then backfill pair 1) or 1..5 incrementally
            true_v_lin = -120.0
            true_v0_acc = -40.0
            true_a_acc = -1200.0
            rng = np.random.default_rng(42)

            for idx, (d0, d1, t0_d, t1_d) in enumerate(dates):
                tau_yr = (t1_d - t0_d) / 365.25
                m_yr = 0.5 * (t0_d + t1_d) / 365.25
                ramp = rng.uniform(-15, 15) + rng.uniform(-25, 25) * u + rng.uniform(-20, 20) * v
                disp = ramp.copy()
                disp[lin_mask] += true_v_lin * tau_yr
                disp[acc_mask] += (true_v0_acc + true_a_acc * m_yr) * tau_yr

                coh = np.full((HEIGHT, WIDTH), 0.85, dtype=np.float32)
                # Mask pass index 2 on a sub-patch inside lin_mask to verify dropout resilience
                if idx == 2:
                    dropout = (yy >= 25) & (yy < 35) & (xx >= 25) & (xx < 35)
                    coh[dropout] = 0.12
                    disp[dropout] += 400.0
                self._make_pair(prod_dir, "T041_086877_IW1", d0, d1, disp, coh)

            # Move pairs 4, 5 aside first so we can test genuine O(1) incremental catch-up
            hidden_dir = tmp_p / "hidden"
            hidden_dir.mkdir()
            for d0, d1, *_ in dates[3:]:
                (prod_dir / f"T041_086877_IW1_{d0}_{d1}").rename(hidden_dir / f"T041_086877_IW1_{d0}_{d1}")

            s1 = sync_stack(StackConfig(products_dir=prod_dir, out_dir=out_inc, stack_bin=STACK_BIN))
            self.assertEqual(s1.newly_stacked, 3)
            self.assertEqual(s1.skipped_existing, 0)

            # Reveal pair 4 & 5
            for d0, d1, *_ in dates[3:]:
                (hidden_dir / f"T041_086877_IW1_{d0}_{d1}").rename(prod_dir / f"T041_086877_IW1_{d0}_{d1}")

            s2 = sync_stack(StackConfig(products_dir=prod_dir, out_dir=out_inc, stack_bin=STACK_BIN))
            self.assertEqual(s2.newly_stacked, 2)
            self.assertEqual(s2.skipped_existing, 3)
            self.assertFalse(s2.rebuilt)

            # Idempotent no-op re-run
            s3 = sync_stack(StackConfig(products_dir=prod_dir, out_dir=out_inc, stack_bin=STACK_BIN))
            self.assertEqual(s3.newly_stacked, 0)
            self.assertEqual(s3.skipped_existing, 5)

            # Fresh batch build across all 5 pairs
            s_batch = sync_stack(StackConfig(products_dir=prod_dir, out_dir=out_batch, stack_bin=STACK_BIN))
            self.assertEqual(s_batch.newly_stacked, 5)

            for fname in (
                "velocity_mm_yr.tif",
                "acceleration_mm_yr2.tif",
                "cumulative_disp_mm.tif",
                "temporal_coherence.tif",
                "velocity_uncertainty_mm_yr.tif",
            ):
                arr_inc = _read_cog_f32(out_inc / fname)
                arr_bat = _read_cog_f32(out_batch / fname)
                finite = np.isfinite(arr_inc) & np.isfinite(arr_bat)
                self.assertGreater(int(finite.sum()), 1000, msg=fname)
                np.testing.assert_allclose(np.isfinite(arr_inc), np.isfinite(arr_bat))
                np.testing.assert_allclose(arr_inc[finite], arr_bat[finite], atol=1e-3, err_msg=fname)

            vel = _read_cog_f32(out_inc / "velocity_mm_yr.tif")
            acc = _read_cog_f32(out_inc / "acceleration_mm_yr2.tif")
            cum = _read_cog_f32(out_inc / "cumulative_disp_mm.tif")
            # Linear zone recovered despite 1-pass dropout inside [25:35, 25:35]
            self.assertAlmostEqual(float(np.nanmedian(vel[lin_mask])), true_v_lin, delta=3.5)
            self.assertAlmostEqual(float(vel[30, 30]), true_v_lin, delta=4.0)
            self.assertAlmostEqual(float(np.nanmedian(cum[lin_mask])), true_v_lin * (30.0 / 365.25), delta=0.5)
            # Accelerating zone recovered
            self.assertAlmostEqual(float(np.nanmedian(acc[acc_mask])), true_a_acc, delta=40.0)

            # Prepending an earlier backfill pair triggers automatic clean rebuild
            early_disp = np.full((HEIGHT, WIDTH), 0.0, dtype=np.float32)
            early_disp[lin_mask] = true_v_lin * (6.0 / 365.25)
            early_coh = np.full((HEIGHT, WIDTH), 0.85, dtype=np.float32)
            self._make_pair(prod_dir, "T041_086877_IW1", "2026-09-30", "2026-10-06", early_disp, early_coh)

            s_backfill = sync_stack(StackConfig(products_dir=prod_dir, out_dir=out_inc, stack_bin=STACK_BIN))
            self.assertTrue(s_backfill.rebuilt)
            self.assertEqual(s_backfill.total_pairs_in_chain, 6)
            self.assertEqual(s_backfill.newly_stacked, 6)

    def test_corridor_pinning_filters_alerts_and_records_timeseries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prod_dir = tmp_p / "products"
            out_dir = tmp_p / "stack_corridors"

            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            subs_band = (yy >= 42) & (yy <= 58) & (xx >= 20) & (xx <= 90)
            outside_anomaly = (yy >= 85) & (yy <= 92) & (xx >= 5) & (xx <= 12)

            dates = [
                ("2026-10-01", "2026-10-07"),
                ("2026-10-07", "2026-10-13"),
                ("2026-10-13", "2026-10-19"),
                ("2026-10-19", "2026-10-25"),
            ]
            for d0, d1 in dates:
                disp = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                disp[subs_band] = -180.0 * (6.0 / 365.25)
                disp[outside_anomaly] = -300.0 * (6.0 / 365.25)
                coh = np.full((HEIGHT, WIDTH), 0.9, dtype=np.float32)
                self._make_pair(prod_dir, "NISAR_T052_F061_D", d0, d1, disp, coh)

            corridors_path = tmp_p / "corridors.geojson"
            corridors_fc = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"asset_id": "rail_active", "name": "Active Subsidence Rail", "asset_class": "rail"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [_utm_to_wgs84(25, 50), _utm_to_wgs84(85, 50)],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"asset_id": "highway_stable", "name": "Stable Highway", "asset_class": "motorway"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [_utm_to_wgs84(25, 15), _utm_to_wgs84(85, 15)],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"asset_id": "remote_line", "name": "Outside Footprint Line", "asset_class": "rail"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [_utm_to_wgs84(-400, -400), _utm_to_wgs84(-300, -400)],
                        },
                    },
                ],
            }
            corridors_path.write_text(json.dumps(corridors_fc), encoding="utf-8")

            summary_immature = sync_stack(
                StackConfig(
                    products_dir=prod_dir,
                    out_dir=out_dir,
                    stack_bin=STACK_BIN,
                    corridors=corridors_path,
                    buffer_m=60.0,
                    vel_alert_mm_yr=25.0,
                    min_alert_coh=0.6,
                    min_alert_snr=2.5,
                    min_alert_span_days=60.0,
                )
            )
            # 24-day span is below the 60-day maturity threshold: critical alerts must be gated!
            self.assertEqual(summary_immature.corridors_evaluated, 3)
            self.assertEqual(summary_immature.critical_corridors, 0)
            self.assertFalse(summary_immature.stack_mature)
            rep_imm = json.loads((out_dir / "corridor_report.geojson").read_text(encoding="utf-8"))
            by_id_imm = {f["properties"]["asset_id"]: f["properties"] for f in rep_imm["features"]}
            self.assertEqual(by_id_imm["rail_active"]["risk_level"], "warning")
            self.assertTrue(by_id_imm["rail_active"]["maturity_gated"])

            # Now test with maturity threshold met (e.g. re-exporting with 20-day threshold for this 24-day fixture)
            summary = sync_stack(
                StackConfig(
                    products_dir=prod_dir,
                    out_dir=out_dir,
                    stack_bin=STACK_BIN,
                    corridors=corridors_path,
                    buffer_m=60.0,
                    vel_alert_mm_yr=25.0,
                    min_alert_coh=0.6,
                    min_alert_snr=2.5,
                    min_alert_span_days=20.0,
                )
            )
            self.assertEqual(summary.corridors_evaluated, 3)
            self.assertTrue(summary.stack_mature)
            self.assertGreaterEqual(summary.critical_corridors, 1)
            self.assertGreater(summary.corridor_alerts, 0)

            report = json.loads((out_dir / "corridor_report.geojson").read_text(encoding="utf-8"))
            by_id = {f["properties"]["asset_id"]: f["properties"] for f in report["features"]}

            self.assertIn(by_id["rail_active"]["risk_level"], ("critical", "warning"))
            self.assertLess(by_id["rail_active"]["mean_velocity_mm_yr"], -100.0)
            self.assertEqual(by_id["highway_stable"]["risk_level"], "nominal")
            self.assertAlmostEqual(by_id["highway_stable"]["mean_velocity_mm_yr"], 0.0, delta=15.0)
            self.assertEqual(by_id["remote_line"]["risk_level"], "no_coverage")
            self.assertEqual(by_id["remote_line"]["valid_pixels"], 0)

            # Verify corridor_alerts.geojson strictly excludes outside_anomaly pixels
            alerts = json.loads((out_dir / "corridor_alerts.geojson").read_text(encoding="utf-8"))
            alert_corridors = {f["properties"]["corridor_asset_id"] for f in alerts["features"]}
            self.assertEqual(alert_corridors, {"rail_active"})
            for f in alerts["features"]:
                py = f["properties"]["pixel_y"]
                self.assertLess(py, 70)

            # Verify multi-epoch corridor_timeseries.json
            ts = json.loads((out_dir / "corridor_timeseries.json").read_text(encoding="utf-8"))
            self.assertEqual(ts["epochs"], ["2026-10-01", "2026-10-07", "2026-10-13", "2026-10-19", "2026-10-25"])
            rail_pts = ts["corridors"]["rail_active"]["points"]
            self.assertEqual(len(rail_pts), 5)
            self.assertEqual(rail_pts[0]["cumulative_disp_mm"], 0.0)
            self.assertLess(rail_pts[-1]["cumulative_disp_mm"], rail_pts[1]["cumulative_disp_mm"])

    def test_auto_ingest_example_sites_and_stack_wiring(self) -> None:
        sites = load_sites_config(REPO_ROOT / "configs" / "sites.example.json")
        by_id = {s.site_id: s for s in sites}
        self.assertTrue(by_id["mexico_city_iw1"].stack)
        self.assertEqual(
            by_id["mexico_city_iw1"].corridors,
            Path("data/corridors/mexico_city_corridors.geojson"),
        )
        self.assertTrue(by_id["west_yorkshire_nisar"].stack)
        self.assertEqual(by_id["west_yorkshire_nisar"].track, 52)
        self.assertEqual(
            by_id["west_yorkshire_nisar"].corridors,
            Path("data/corridors/west_yorkshire_corridors.geojson"),
        )

        # Verify dry-run with stack=True cleanly invokes dry-run stack hook
        with tempfile.TemporaryDirectory() as tmp:
            class _FakeResult:
                def __init__(self, properties):
                    self.properties = properties

            class _FakeAsf:
                class DATASET:
                    OPERA_S1 = "OPERA-S1"

                class ASFSession:
                    pass

                @staticmethod
                def search(**kwargs):
                    return [
                        _FakeResult(
                            {
                                "sceneName": "OPERA_T041_20261001",
                                "fileName": "OPERA_T041_20261001.h5",
                                "startTime": "2026-10-01T12:00:00Z",
                                "url": "https://example.invalid/o1.h5",
                            }
                        ),
                        _FakeResult(
                            {
                                "sceneName": "OPERA_T041_20261007",
                                "fileName": "OPERA_T041_20261007.h5",
                                "startTime": "2026-10-07T12:00:00Z",
                                "url": "https://example.invalid/o2.h5",
                            }
                        ),
                    ]

            summary = sync_site(
                by_id["mexico_city_iw1"],
                work_dir=Path(tmp),
                engine_bin=REPO_ROOT / "build" / "insar_proto",
                dry_run=True,
                asf_mod=_FakeAsf,
            )
            self.assertEqual(summary.discovered_count, 2)
            self.assertEqual(summary.stack_pairs, 0)

    def test_snr_gating_suppresses_noisy_corridor_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prod_dir = tmp_p / "products"
            out_dir = tmp_p / "stack_snr"

            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            steady_band = (yy >= 20) & (yy <= 34) & (xx >= 20) & (xx <= 90)
            noisy_band = (yy >= 64) & (yy <= 78) & (xx >= 20) & (xx <= 90)

            # 6 pairs * 12 days = 72 days (>= 60 days mature!)
            dates = [
                ("2026-07-01", "2026-07-13"),
                ("2026-07-13", "2026-07-25"),
                ("2026-07-25", "2026-08-06"),
                ("2026-08-06", "2026-08-18"),
                ("2026-08-18", "2026-08-30"),
                ("2026-08-30", "2026-09-11"),
            ]
            # Alternating +/- atmospheric jitter giving mean ~-25 mm/yr but sigma_v ~140 mm/yr (SNR << 2.5)
            jitter = [-8.0, +6.5, -9.0, +7.0, -8.5, +7.0]
            for idx, (d0, d1) in enumerate(dates):
                disp = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                disp[steady_band] = -120.0 * (12.0 / 365.25)
                disp[noisy_band] = jitter[idx]
                coh = np.full((HEIGHT, WIDTH), 0.88, dtype=np.float32)
                self._make_pair(prod_dir, "T041_086877_IW1", d0, d1, disp, coh)

            corridors_path = tmp_p / "corridors.geojson"
            corridors_fc = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"asset_id": "steady_rail", "name": "Steady High-SNR Rail", "asset_class": "rail"},
                        "geometry": {"type": "LineString", "coordinates": [_utm_to_wgs84(25, 27), _utm_to_wgs84(85, 27)]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"asset_id": "noisy_road", "name": "Noisy Low-SNR Road", "asset_class": "motorway"},
                        "geometry": {"type": "LineString", "coordinates": [_utm_to_wgs84(25, 71), _utm_to_wgs84(85, 71)]},
                    },
                ],
            }
            corridors_path.write_text(json.dumps(corridors_fc), encoding="utf-8")

            summary = sync_stack(
                StackConfig(
                    products_dir=prod_dir,
                    out_dir=out_dir,
                    stack_bin=STACK_BIN,
                    ramp_order=0,
                    ref_px=(10, 10, 4),
                    corridors=corridors_path,
                    buffer_m=60.0,
                    vel_alert_mm_yr=20.0,
                    min_alert_coh=0.6,
                    min_alert_snr=2.5,
                    min_alert_span_days=60.0,
                )
            )
            self.assertTrue(summary.stack_mature)
            report = json.loads((out_dir / "corridor_report.geojson").read_text(encoding="utf-8"))
            by_id = {f["properties"]["asset_id"]: f["properties"] for f in report["features"]}
            self.assertEqual(by_id["steady_rail"]["risk_level"], "critical")
            self.assertGreater(by_id["steady_rail"]["alert_pixel_count"], 10)
            self.assertEqual(by_id["noisy_road"]["alert_pixel_count"], 0)
            self.assertGreater(by_id["noisy_road"]["snr_gated_pixels"], 0)
            self.assertNotEqual(by_id["noisy_road"]["risk_level"], "critical")

    def test_ref_pin_deform_mask_dem_tropo_and_disconnected_island(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prod_dir = tmp_p / "products"
            out_dir = tmp_p / "stack_dem"

            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            # Create synthetic DEM: mountains on west (x < 35, up to 3500 m), foothills (35..70, 2300..2800 m),
            # flat subsiding basin on east (x >= 70, 2230 m)
            dem = 2230.0 + np.clip(70.0 - xx, 0.0, None) * 18.0 + 4.0 * np.abs(yy - 50.0)
            dem_path = tmp_p / "synth_dem.tif"
            _write_geotiff_f32(dem_path, dem)

            # Decorrelation barrier splitting west mountain island (x < 24) from main basin (x >= 32)
            barrier = (xx >= 24) & (xx < 32)
            west_island = xx < 24
            # Large eastern subsidence bowl (~35% of main basin)
            bowl = (xx >= 74) & (yy >= 15) & (yy < 85)
            # Explicit vector/raster exclusion mask for deformation bowl
            deform_mask_path = tmp_p / "deform_mask.tif"
            _write_geotiff_f32(deform_mask_path, bowl.astype(np.float32))

            dates = [
                ("2026-07-01", "2026-07-13"),
                ("2026-07-13", "2026-07-25"),
                ("2026-07-25", "2026-08-06"),
                ("2026-08-06", "2026-08-18"),
            ]
            true_v_bowl = -180.0
            dem_slopes = [24.0, -30.0, 18.0, -22.0]  # strong stratified troposphere (mm/km)
            island_jumps = [55.46, -83.19, 27.73, -55.46]  # integer 2pi jumps on disconnected west island
            for idx, (d0, d1) in enumerate(dates):
                tau_yr = 12.0 / 365.25
                disp = np.full((HEIGHT, WIDTH), 15.0, dtype=np.float64)  # arbitrary datum offset
                disp += dem_slopes[idx] * ((dem - 2400.0) / 1000.0)
                disp[bowl] += true_v_bowl * tau_yr
                disp[west_island] += island_jumps[idx]

                coh = np.full((HEIGHT, WIDTH), 0.85, dtype=np.float32)
                coh[barrier] = 0.15
                self._make_pair(prod_dir, "T041_086877_IW1", d0, d1, disp, coh)

            # Pin to stable foothills bedrock at (48, 50)
            sync_stack(
                StackConfig(
                    products_dir=prod_dir,
                    out_dir=out_dir,
                    stack_bin=STACK_BIN,
                    ramp_order=0,
                    dem=dem_path,
                    ref_px=(48, 50, 5),
                    deform_mask=deform_mask_path,
                )
            )
            vel = _read_cog_f32(out_dir / "velocity_mm_yr.tif")
            summary = json.loads((out_dir / "stack_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["ramp_removal"]["dem_applied"])
            self.assertTrue(summary["reference_region"]["applied"])
            self.assertAlmostEqual(summary["ramp_removal"]["dem_slope_mm_per_km"], dem_slopes[-1], delta=2.0)

            # Bedrock near reference & higher foothills must stay ~0 mm/yr despite +-30 mm/km tropo and west island jump
            stable_foothills = (xx >= 38) & (xx <= 66) & (yy >= 20) & (yy <= 80)
            self.assertAlmostEqual(float(np.nanmedian(vel[stable_foothills])), 0.0, delta=3.0)
            # Subsidence bowl recovered accurately
            self.assertAlmostEqual(float(np.nanmedian(vel[bowl])), true_v_bowl, delta=5.0)

    def test_triplet_phase_closure_masks_local_jump_and_rejects_corrupted_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prod_dir = tmp_p / "products"
            out_dir = tmp_p / "stack_closure"

            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            subs_zone = (yy >= 30) & (yy < 70) & (xx >= 40) & (xx < 80)
            true_v = -150.0  # mm/yr

            # 5 epochs separated by 12 days: E0..E4
            epochs = ["2026-07-01", "2026-07-13", "2026-07-25", "2026-08-06", "2026-08-18"]
            coh = np.full((HEIGHT, WIDTH), 0.90, dtype=np.float32)

            def _clean_disp(step_cnt: int) -> np.ndarray:
                d = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                d[subs_zone] = true_v * ((12.0 * step_cnt) / 365.25)
                return d

            # Consecutive pairs (0,1), (1,2), (2,3), (3,4)
            self._make_pair(prod_dir, "T041_086877_IW1", epochs[0], epochs[1], _clean_disp(1), coh)
            # Inject localized +27.73 mm (2*pi) unwrapping jump inside subs_zone on pair (1,2)
            d12 = _clean_disp(1)
            local_err = (yy >= 40) & (yy < 56) & (xx >= 50) & (xx < 66)
            d12[local_err] += 27.73
            self._make_pair(prod_dir, "T041_086877_IW1", epochs[1], epochs[2], d12, coh)

            # Inject catastrophic +55.46 mm (4*pi) unwrapping failure across 60% of pair (2,3)
            d23 = _clean_disp(1)
            d23[:60, :] += 55.46
            self._make_pair(prod_dir, "T041_086877_IW1", epochs[2], epochs[3], d23, coh)
            self._make_pair(prod_dir, "T041_086877_IW1", epochs[3], epochs[4], _clean_disp(1), coh)

            # Redundant skip-1 pairs (0,2), (1,3), (2,4) and skip-2 pairs (0,3), (1,4) — all clean!
            for i in range(3):
                self._make_pair(prod_dir, "T041_086877_IW1", epochs[i], epochs[i + 2], _clean_disp(2), coh)
            for i in range(2):
                self._make_pair(prod_dir, "T041_086877_IW1", epochs[i], epochs[i + 3], _clean_disp(3), coh)

            summary = sync_stack(
                StackConfig(
                    products_dir=prod_dir,
                    out_dir=out_dir,
                    stack_bin=STACK_BIN,
                    ramp_order=0,
                    ref_px=(10, 10, 4),
                    enable_closure=True,
                    closure_thresh_mm=10.0,
                    closure_max_bad_frac=0.35,
                    closure_mode="consensus",
                    include_redundant=True,
                )
            )
            self.assertEqual(summary.total_pairs_in_chain, 4)
            self.assertEqual(summary.redundant_pairs_found, 5)
            self.assertEqual(summary.rejected_pairs, 1)  # Catastrophic pair (2,3) rejected!
            self.assertGreater(summary.triplets_evaluated, 6)

            ledger = json.loads((out_dir / "stack_ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["rejected_pair_ids"], ["2026-07-25_2026-08-06"])

            vel = _read_cog_f32(out_dir / "velocity_mm_yr.tif")
            # Both the locally unwrapping-corrupted patch AND the full subsidence zone must match true_v!
            self.assertAlmostEqual(float(np.nanmedian(vel[local_err])), true_v, delta=3.0)
            self.assertAlmostEqual(float(np.nanmedian(vel[subs_zone])), true_v, delta=3.0)


if __name__ == "__main__":
    unittest.main()
