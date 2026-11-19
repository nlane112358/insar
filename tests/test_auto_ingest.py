#!/usr/bin/env python3
"""Unit and end-to-end tests for scripts/auto_ingest.py."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import auto_ingest as ai  # noqa: E402

ENGINE_BIN = REPO_ROOT / "build" / "insar_proto"


def _write_complex_h5(
    path: Path,
    dataset_path: str,
    width: int = 48,
    height: int = 48,
    phase_rad: float = 0.4,
    center_freq_hz: float | None = None,
    freq_units: str | None = None,
    scalar_freq: bool = True,
) -> Path:
    """Create a real complex64 HDF5 raster readable by GDAL / insar_proto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    carrier = np.exp(1j * (0.08 * xx + 0.05 * yy + phase_rad)).astype(np.complex64)
    group_path, ds_name = dataset_path.rsplit("/", 1)
    with h5py.File(path, "w") as f:
        grp = f.require_group(group_path)
        grp.create_dataset(ds_name, data=carrier)
        if center_freq_hz is not None:
            data = np.float64(center_freq_hz) if scalar_freq else np.array([center_freq_hz], dtype=np.float64)
            ds = grp.create_dataset("centerFrequency", data=data)
            if freq_units is not None:
                ds.attrs["units"] = freq_units
    return path


class _FakeASFProduct:
    """Minimal mock mimicking asf_search.ASFProduct."""

    def __init__(
        self,
        *,
        scene_name: str,
        burst_id: str,
        start_time: str,
        source_h5: Path,
    ) -> None:
        self.properties = {
            "sceneName": scene_name,
            "fileID": scene_name,
            "operaBurstID": burst_id,
            "startTime": start_time,
            "fileName": source_h5.name,
            "url": f"https://datapool.asf.alaska.edu/OPERA/{source_h5.name}",
        }
        self._source_h5 = source_h5

    def download(self, path: str, session: object = None) -> None:
        dest = Path(path) / self.properties["fileName"]
        shutil.copy2(self._source_h5, dest)


class _FakeASFSession:
    def __init__(self) -> None:
        self.token = None
        self.creds = None

    def auth_with_token(self, token: str) -> None:
        self.token = token

    def auth_with_creds(self, user: str, password: str) -> None:
        self.creds = (user, password)


class _FakeASFModule:
    DATASET = SimpleNamespace(OPERA_S1="OPERA-S1")
    ASFSession = _FakeASFSession

    def __init__(self, products: list[_FakeASFProduct]) -> None:
        self._products = products
        self.search_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        burst_id = kwargs.get("operaBurstID")
        if burst_id is not None:
            return [p for p in self._products if p.properties.get("operaBurstID") == burst_id]
        return list(self._products)


class _FakeEarthaccessGranule(dict):
    """Minimal mock mimicking earthaccess.results.DataGranule."""

    def __init__(
        self,
        *,
        granule_ur: str,
        start_time: str,
        source_h5: Path,
        track: int | None = None,
        frame: int | None = None,
        direction: str | None = None,
    ) -> None:
        attrs = []
        if track is not None:
            attrs.append({"Name": "TRACK_NUMBER", "Values": [str(track)]})
        if frame is not None:
            attrs.append({"Name": "FRAME_NUMBER", "Values": [str(frame)]})
        if direction is not None:
            attrs.append({"Name": "ASCENDING_DESCENDING", "Values": [direction]})

        super().__init__(
            {
                "meta": {"concept-id": f"G-{granule_ur}"},
                "umm": {
                    "GranuleUR": granule_ur,
                    "TemporalExtent": {
                        "RangeDateTime": {"BeginningDateTime": start_time}
                    },
                    "AdditionalAttributes": attrs,
                },
            }
        )
        self._source_h5 = source_h5

    def data_links(self) -> list[str]:
        return [
            f"https://nisar.asf.earthdatacloud.nasa.gov/{self._source_h5.name}.iso.xml",
            f"https://nisar.asf.earthdatacloud.nasa.gov/{self._source_h5.name}",
        ]


class _FakeEarthaccessModule:
    def __init__(self, granules: list[_FakeEarthaccessGranule]) -> None:
        self._granules = granules
        self.login_strategies: list[str] = []
        self.search_calls: list[dict] = []
        self.download_calls: list[str] = []

    def login(self, strategy: str = "all"):
        self.login_strategies.append(strategy)
        return SimpleNamespace(authenticated=True)

    def search_data(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self._granules)

    def download(self, granules, local_path: str, show_progress: bool = False):
        out_paths: list[Path] = []
        for g in granules:
            self.download_calls.append(g["umm"]["GranuleUR"])
            # Emit a sidecar .xml alongside .h5 to ensure .h5 selection works
            xml_path = Path(local_path) / f"{g._source_h5.name}.iso.xml"
            xml_path.write_text("<metadata/>", encoding="utf-8")
            h5_path = Path(local_path) / g._source_h5.name
            shutil.copy2(g._source_h5, h5_path)
            out_paths.extend([xml_path, h5_path])
        return out_paths


class TestAutoIngestUnit(unittest.TestCase):
    def test_legacy_db_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "state.db"
            legacy_conn = sqlite3.connect(db_path)
            legacy_conn.execute("""
                CREATE TABLE granules (
                    granule_id TEXT PRIMARY KEY,
                    burst_id   TEXT NOT NULL,
                    acq_time   TEXT NOT NULL,
                    local_path TEXT NOT NULL
                )
            """)
            legacy_conn.execute("""
                CREATE TABLE processed_pairs (
                    ref_id      TEXT NOT NULL,
                    sec_id      TEXT NOT NULL,
                    out_dir     TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    PRIMARY KEY (ref_id, sec_id)
                )
            """)
            legacy_conn.execute(
                "INSERT INTO granules VALUES (?, ?, ?, ?)",
                ("G1", "T078-165486-IW2", "2026-10-01T00:00:00Z", "/tmp/g1.h5"),
            )
            legacy_conn.commit()
            legacy_conn.close()

            conn = ai.init_db(db_path)
            row = conn.execute(
                "SELECT granule_id, stream_key, sensor, site_id, pruned FROM granules WHERE granule_id = 'G1'"
            ).fetchone()
            self.assertEqual(row["stream_key"], "T078-165486-IW2")
            self.assertEqual(row["sensor"], "opera")
            self.assertEqual(row["site_id"], "")
            self.assertEqual(row["pruned"], 0)
            conn.close()

    def test_discover_burst_id_prefers_multipass_burst(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dummy_h5 = Path(td) / "dummy.h5"
            dummy_h5.write_bytes(b"stub")
            products = [
                _FakeASFProduct(
                    scene_name="ORPHAN_NEW",
                    burst_id="T001-000001-IW1",
                    start_time="2026-11-18T00:00:00Z",
                    source_h5=dummy_h5,
                ),
                _FakeASFProduct(
                    scene_name="ACTIVE_1",
                    burst_id="T078-165486-IW2",
                    start_time="2026-11-01T00:00:00Z",
                    source_h5=dummy_h5,
                ),
                _FakeASFProduct(
                    scene_name="ACTIVE_2",
                    burst_id="T078-165486-IW2",
                    start_time="2026-11-13T00:00:00Z",
                    source_h5=dummy_h5,
                ),
            ]
            fake_asf = _FakeASFModule(products)
            selected = ai.discover_burst_id(-99.1332, 19.4326, lookback_days=45, asf_mod=fake_asf)
            self.assertEqual(selected, "T078-165486-IW2")

    def test_nisar_wavelength_extraction_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            h5_low = _write_complex_h5(
                td_path / "low.h5",
                "/science/LSAR/GSLC/grids/frequencyA/HH",
                width=16,
                height=16,
                center_freq_hz=1.2215e9,
                scalar_freq=True,
            )
            h5_high_mhz = _write_complex_h5(
                td_path / "high.h5",
                "/science/LSAR/GSLC/grids/frequencyB/VV",
                width=16,
                height=16,
                center_freq_hz=1293.5,
                freq_units="MHz",
                scalar_freq=False,
            )
            h5_missing = _write_complex_h5(
                td_path / "missing.h5",
                "/science/LSAR/GSLC/grids/frequencyA/HH",
                width=16,
                height=16,
                center_freq_hz=None,
            )

            wl_low = ai.read_nisar_wavelength_mm(
                h5_low, "/science/LSAR/GSLC/grids/frequencyA/HH"
            )
            self.assertAlmostEqual(wl_low, (ai.SPEED_OF_LIGHT_M_S / 1.2215e9) * 1000.0, places=4)

            wl_high = ai.read_nisar_wavelength_mm(
                h5_high_mhz, "/science/LSAR/GSLC/grids/frequencyB/VV"
            )
            self.assertAlmostEqual(wl_high, (ai.SPEED_OF_LIGHT_M_S / 1.2935e9) * 1000.0, places=4)

            wl_fallback = ai.read_nisar_wavelength_mm(
                h5_missing, "/science/LSAR/GSLC/grids/frequencyA/HH"
            )
            self.assertAlmostEqual(wl_fallback, ai.NISAR_DEFAULT_WAVELENGTH_MM, places=3)

    def test_nisar_granule_ur_stream_parsing(self) -> None:
        g = _FakeEarthaccessGranule(
            granule_ur="NISAR_L2_PR_GSLC_003_045_A_012_2005_DHDH_A_20260801T060000_20260801T060030_D00400_P_F_J_001",
            start_time="2026-08-01T06:00:00Z",
            source_h5=Path("/tmp/NISAR_L2_PR_GSLC_003_045_A_012.h5"),
        )
        info = ai._extract_nisar_metadata(g)
        self.assertEqual(info["track"], 45)
        self.assertEqual(info["frame"], 12)
        self.assertEqual(info["direction"], "ASCENDING")
        self.assertEqual(info["stream_key"], "NISAR_T045_F012_A")

    def test_sites_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "sites.json"
            cfg.write_text(
                json.dumps(
                    {
                        "defaults": {"lookback_days": 30, "prune_raw": True},
                        "sites": [
                            {"site_id": "site_a", "sensor": "opera", "burst_id": "T078-165486-IW2"},
                            {
                                "site_id": "site_b",
                                "sensor": "nisar",
                                "bbox": [-1.85, 53.55, -1.70, 53.68],
                                "window": [0, 0, 128, 128],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sites = ai.load_sites_config(cfg)
            self.assertEqual(len(sites), 2)
            self.assertEqual(sites[0].lookback_days, 30)
            self.assertTrue(sites[0].prune_raw)
            self.assertEqual(sites[1].window, (0, 0, 128, 128))
            self.assertEqual(sites[1].resolved_subdataset(), ai.NISAR_DEFAULT_SUBDATASET)

            bad_cfg = Path(td) / "bad_sites.json"
            bad_cfg.write_text(
                json.dumps([{"site_id": "../escape", "sensor": "opera"}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ai.load_sites_config(bad_cfg)

    def test_example_sites_config_loads(self) -> None:
        example_cfg = SCRIPTS_DIR / "sites.example.json"
        sites = ai.load_sites_config(example_cfg)
        self.assertEqual([s.site_id for s in sites], ["mexico_city_iw1", "west_yorkshire_nisar"])
        self.assertEqual(sites[0].sensor, "opera")
        self.assertEqual(sites[1].sensor, "nisar")
        self.assertEqual(sites[1].bbox, (-1.85, 53.55, -1.70, 53.68))
        self.assertEqual(sites[1].track, 153)
        self.assertEqual(sites[1].frame, 61)
        self.assertTrue(sites[1].stream)

    def test_opera_hyphen_burst_id_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            f1 = _write_complex_h5(Path(td) / "o1.h5", "/data/VV")
            prod = _FakeASFProduct(
                scene_name="O1",
                burst_id="T078_165486_IW2",
                start_time="2026-11-14T00:47:14Z",
                source_h5=f1,
            )
            fake_asf = _FakeASFModule([prod])
            site = ai.SiteConfig(sensor="opera", burst_id="T078-165486-IW2")
            resolved, cands = ai.fetch_opera_candidates(site, asf_mod=fake_asf)
            self.assertEqual(resolved.burst_id, "T078_165486_IW2")
            self.assertEqual(len(cands), 1)
            self.assertEqual(fake_asf.search_calls[0]["operaBurstID"], "T078_165486_IW2")

    def test_nisar_track_filter_and_env_token_login(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
            raw_dir = td_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            h_keep = _write_complex_h5(src_dir / "keep.h5", "/science/LSAR/GSLC/grids/frequencyA/HH", center_freq_hz=1.2575e9)
            h_drop = _write_complex_h5(src_dir / "drop.h5", "/science/LSAR/GSLC/grids/frequencyA/HH", center_freq_hz=1.2575e9)
            g_keep = _FakeEarthaccessGranule(
                granule_ur="NISAR_L2_PR_GSLC_001_045_A_012_20260801",
                start_time="2026-08-01T00:00:00Z",
                source_h5=h_keep,
                track=45,
                frame=12,
                direction="ASCENDING",
            )
            g_drop = _FakeEarthaccessGranule(
                granule_ur="NISAR_L2_PR_GSLC_001_089_D_033_20260802",
                start_time="2026-08-02T00:00:00Z",
                source_h5=h_drop,
                track=89,
                frame=33,
                direction="DESCENDING",
            )
            fake_ea = _FakeEarthaccessModule([g_keep, g_drop])
            site = ai.SiteConfig(
                site_id="filtered_nisar",
                sensor="nisar",
                bbox=(-1.85, 53.55, -1.70, 53.68),
                track=45,
                frame=12,
                direction="A",
            )
            with patch.dict(os.environ, {"EARTHDATA_TOKEN": "dummy_token"}):
                _, cands = ai.fetch_nisar_candidates(site, ea_mod=fake_ea)
                # Public CMR search must not require auth until a download runs
                self.assertEqual(fake_ea.login_strategies, [])
                self.assertEqual(len(cands), 1)
                self.assertEqual(cands[0].stream_key, "NISAR_T045_F012_A")
                downloaded_path = cands[0].download_fn(raw_dir)
                self.assertTrue(downloaded_path.is_file())
                self.assertEqual(fake_ea.login_strategies, ["environment"])


@unittest.skipUnless(ENGINE_BIN.is_file() and os.access(ENGINE_BIN, os.X_OK), "build/insar_proto not compiled")
class TestAutoIngestEndToEnd(unittest.TestCase):
    def test_opera_end_to_end_idempotency_and_safe_prune(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "upstream"
            work_dir = td_path / "work"

            f1 = _write_complex_h5(src_dir / "OPERA_T078_20261001.h5", "/data/VV", phase_rad=0.1)
            f2 = _write_complex_h5(src_dir / "OPERA_T078_20261013.h5", "/data/VV", phase_rad=0.5)
            f3 = _write_complex_h5(src_dir / "OPERA_T078_20261025.h5", "/data/VV", phase_rad=0.9)
            f4 = _write_complex_h5(src_dir / "OPERA_T078_20261106.h5", "/data/VV", phase_rad=1.2)

            p1 = _FakeASFProduct(scene_name="OPERA_G1", burst_id="T078_165486_IW2", start_time="2026-10-01T12:00:00Z", source_h5=f1)
            p2 = _FakeASFProduct(scene_name="OPERA_G2", burst_id="T078_165486_IW2", start_time="2026-10-13T12:00:00Z", source_h5=f2)
            p3 = _FakeASFProduct(scene_name="OPERA_G3", burst_id="T078_165486_IW2", start_time="2026-10-25T12:00:00Z", source_h5=f3)
            p4 = _FakeASFProduct(scene_name="OPERA_G4", burst_id="T078_165486_IW2", start_time="2026-11-06T12:00:00Z", source_h5=f4)

            # Run 1: 3 granules -> 2 pairs -> prune first 2, keep 3rd
            fake_asf_1 = _FakeASFModule([p1, p3, p2])  # deliberately unsorted
            sum1 = ai.sync_and_process(
                burst_id="T078_165486_IW2",
                work_dir=work_dir,
                engine_bin=ENGINE_BIN,
                lookback_days=45,
                max_granules=6,
                stride_x=2,
                stride_y=2,
                prune_old_h5=True,
                asf_mod=fake_asf_1,
            )
            self.assertEqual(sum1.downloaded_count, 3)
            self.assertEqual(sum1.pairs_processed, 2)
            self.assertEqual(sum1.pruned_count, 2)

            pair1_dir = work_dir / "products" / "T078_165486_IW2_2026-10-01_2026-10-13"
            pair2_dir = work_dir / "products" / "T078_165486_IW2_2026-10-13_2026-10-25"
            for pdir in (pair1_dir, pair2_dir):
                self.assertTrue((pdir / "wrapped_phase.tif").is_file())
                self.assertTrue((pdir / "coherence.tif").is_file())
                self.assertTrue((pdir / "displacement_mm.tif").is_file())
                self.assertTrue((pdir / "summary.json").is_file())
                self.assertTrue((pdir / "alerts.geojson").is_file())

            raw_dir = work_dir / "raw"
            self.assertFalse((raw_dir / f1.name).exists())
            self.assertFalse((raw_dir / f2.name).exists())
            self.assertTrue((raw_dir / f3.name).exists())

            # Run 2: same 3 granules -> MUST NOT re-download pruned f1/f2 or rerun engine
            sum2 = ai.sync_and_process(
                burst_id="T078_165486_IW2",
                work_dir=work_dir,
                engine_bin=ENGINE_BIN,
                lookback_days=45,
                max_granules=6,
                stride_x=2,
                stride_y=2,
                prune_old_h5=True,
                asf_mod=fake_asf_1,
            )
            self.assertEqual(sum2.downloaded_count, 0)
            self.assertEqual(sum2.pairs_processed, 0)
            self.assertEqual(sum2.pairs_skipped, 2)
            self.assertEqual(sum2.pruned_count, 0)

            # Run 3: 4th pass arrives -> download only f4, pair f3 -> f4, prune f3
            fake_asf_2 = _FakeASFModule([p1, p2, p3, p4])
            sum3 = ai.sync_and_process(
                burst_id="T078_165486_IW2",
                work_dir=work_dir,
                engine_bin=ENGINE_BIN,
                lookback_days=45,
                max_granules=6,
                stride_x=2,
                stride_y=2,
                prune_old_h5=True,
                asf_mod=fake_asf_2,
            )
            self.assertEqual(sum3.downloaded_count, 1)
            self.assertEqual(sum3.pairs_processed, 1)
            self.assertEqual(sum3.pruned_count, 1)
            self.assertFalse((raw_dir / f3.name).exists())
            self.assertTrue((raw_dir / f4.name).exists())
            self.assertTrue((work_dir / "products" / "T078_165486_IW2_2026-10-25_2026-11-06" / "displacement_mm.tif").is_file())

    def test_nisar_end_to_end_streams_and_dynamic_wavelength(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "upstream_nisar"
            work_dir = td_path / "work_nisar"

            # Stream 1 (Track 45, Frame 12, ASCENDING): 1221.5 MHz
            n1 = _write_complex_h5(
                src_dir / "NISAR_L2_PR_GSLC_001_045_A_012_20260801.h5",
                "/science/LSAR/GSLC/grids/frequencyA/HH",
                phase_rad=0.1,
                center_freq_hz=1.2215e9,
            )
            n2 = _write_complex_h5(
                src_dir / "NISAR_L2_PR_GSLC_002_045_A_012_20260813.h5",
                "/science/LSAR/GSLC/grids/frequencyA/HH",
                phase_rad=0.6,
                center_freq_hz=1.2215e9,
            )
            # Stream 2 (Track 89, Frame 33, DESCENDING): 1293.5 MHz -> single granule, must NOT pair with Track 45!
            n_other = _write_complex_h5(
                src_dir / "NISAR_L2_PR_GSLC_001_089_D_033_20260805.h5",
                "/science/LSAR/GSLC/grids/frequencyA/HH",
                phase_rad=0.9,
                center_freq_hz=1.2935e9,
            )

            g1 = _FakeEarthaccessGranule(
                granule_ur="NISAR_L2_PR_GSLC_001_045_A_012_20260801",
                start_time="2026-08-01T06:00:00Z",
                source_h5=n1,
                track=45,
                frame=12,
                direction="ASCENDING",
            )
            g2 = _FakeEarthaccessGranule(
                granule_ur="NISAR_L2_PR_GSLC_002_045_A_012_20260813",
                start_time="2026-08-13T06:00:00Z",
                source_h5=n2,
                track=45,
                frame=12,
                direction="ASCENDING",
            )
            g_other = _FakeEarthaccessGranule(
                granule_ur="NISAR_L2_PR_GSLC_001_089_D_033_20260805",
                start_time="2026-08-05T18:00:00Z",
                source_h5=n_other,
                track=89,
                frame=33,
                direction="DESCENDING",
            )

            fake_ea = _FakeEarthaccessModule([g1, g_other, g2])
            site = ai.SiteConfig(
                site_id="yorkshire_nisar",
                sensor="nisar",
                bbox=(-1.85, 53.55, -1.70, 53.68),
                stride_x=2,
                stride_y=2,
                window=(0, 0, 32, 32),
            )
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("EARTHDATA_TOKEN", None)
                os.environ.pop("EARTHDATA_USERNAME", None)
                os.environ.pop("EARTHDATA_PASSWORD", None)
                summary = ai.sync_site(site, work_dir, ENGINE_BIN, ea_mod=fake_ea)

            # Never use interactive login in headless mode
            self.assertEqual(fake_ea.login_strategies, ["netrc"])
            # 3 granules downloaded across 2 streams, but ONLY 1 same-stream pair processed!
            self.assertEqual(summary.downloaded_count, 3)
            self.assertEqual(summary.pairs_processed, 1)

            pair_out = work_dir / "products" / "yorkshire_nisar" / "NISAR_T045_F012_A_2026-08-01_2026-08-13"
            self.assertTrue((pair_out / "displacement_mm.tif").is_file())
            summary_json = json.loads((pair_out / "summary.json").read_text(encoding="utf-8"))
            # 32x32 window divided by stride 2x2 => 16x16
            self.assertEqual(summary_json["width"], 16)
            self.assertEqual(summary_json["height"], 16)

            conn = sqlite3.connect(work_dir / "state.db")
            wl_row = conn.execute(
                "SELECT wavelength_mm FROM processed_pairs WHERE site_id = 'yorkshire_nisar'"
            ).fetchone()
            conn.close()
            self.assertIsNotNone(wl_row)
            expected_wl = (ai.SPEED_OF_LIGHT_M_S / 1.2215e9) * 1000.0
            self.assertAlmostEqual(wl_row[0], expected_wl, places=3)

    def test_dry_run_and_multi_site_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
            work_dir = td_path / "work_multi"

            f1 = _write_complex_h5(src_dir / "o1.h5", "/data/VV", phase_rad=0.1)
            f2 = _write_complex_h5(src_dir / "o2.h5", "/data/VV", phase_rad=0.4)
            fake_asf = _FakeASFModule(
                [
                    _FakeASFProduct(scene_name="O1", burst_id="T078_165486_IW2", start_time="2026-10-01T00:00:00Z", source_h5=f1),
                    _FakeASFProduct(scene_name="O2", burst_id="T078_165486_IW2", start_time="2026-10-13T00:00:00Z", source_h5=f2),
                ]
            )

            # Dry run must not create raw/ or state.db (also exercises hyphenated burst normalization)
            site = ai.SiteConfig(site_id="s1", sensor="opera", burst_id="T078-165486-IW2", stride_x=2, stride_y=2)
            dry_sum = ai.sync_site(site, work_dir, ENGINE_BIN, dry_run=True, asf_mod=fake_asf)
            self.assertEqual(dry_sum.discovered_count, 2)
            self.assertEqual(dry_sum.downloaded_count, 0)
            self.assertEqual(dry_sum.pairs_processed, 0)
            self.assertFalse((work_dir / "state.db").exists())

            # Multi-site CLI with --continue-on-error: site_good succeeds even if site_bad fails
            cfg = td_path / "sites.json"
            cfg.write_text(
                json.dumps(
                    {
                        "sites": [
                            {"site_id": "site_bad", "sensor": "opera", "burst_id": "MISSING-BURST"},
                            {"site_id": "site_good", "sensor": "opera", "burst_id": "T078-165486-IW2", "stride_x": 2, "stride_y": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_sync(site, work_dir, engine_bin, dry_run=False):
                if site.site_id == "site_bad":
                    raise RuntimeError("Simulated ASF failure")
                return ai.SiteRunSummary(site.site_id, site.sensor, 2, 2, 1, 0, 0)

            with patch.object(ai, "sync_site", side_effect=fake_sync) as mock_sync:
                rc = ai.main(
                    [
                        "--sites-config", str(cfg),
                        "--work-dir", str(work_dir),
                        "--engine", str(ENGINE_BIN),
                        "--continue-on-error",
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertEqual(mock_sync.call_count, 2)

    def test_failed_pair_does_not_prune_reference_granule(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "upstream"
            work_dir = td_path / "work_fail"

            # Mismatched dimensions (48x48 vs 32x32) causes insar_proto to exit non-zero
            f1 = _write_complex_h5(src_dir / "bad_1.h5", "/data/VV", width=48, height=48)
            f2 = _write_complex_h5(src_dir / "bad_2.h5", "/data/VV", width=32, height=32)
            p1 = _FakeASFProduct(scene_name="BAD_1", burst_id="T078_165486_IW2", start_time="2026-10-01T00:00:00Z", source_h5=f1)
            p2 = _FakeASFProduct(scene_name="BAD_2", burst_id="T078_165486_IW2", start_time="2026-10-13T00:00:00Z", source_h5=f2)
            fake_asf = _FakeASFModule([p1, p2])

            with self.assertRaises(Exception):
                ai.sync_and_process(
                    burst_id="T078-165486-IW2",
                    work_dir=work_dir,
                    engine_bin=ENGINE_BIN,
                    lookback_days=45,
                    max_granules=6,
                    stride_x=2,
                    stride_y=2,
                    prune_old_h5=True,
                    asf_mod=fake_asf,
                )

            # Both raw files must remain unpruned on disk for inspection / retry
            self.assertTrue((work_dir / "raw" / f1.name).exists())
            self.assertTrue((work_dir / "raw" / f2.name).exists())

    def test_stream_target_resolution_and_subprocess_env(self) -> None:
        class _FakeResp:
            def __init__(self, url: str) -> None:
                self.url = url
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def raise_for_status(self) -> None:
                pass

        class _FakeSession:
            def __init__(self, final_url: str) -> None:
                self._final_url = final_url
            def get(self, url: str, **kwargs):
                return _FakeResp(self._final_url)

        class _FakeGranule(dict):
            def data_links(self, access: str = "external"):
                if access == "direct":
                    return ["s3://nisar-prod-bucket/path/frame.h5"]
                return ["https://nisar.asf.earthdatacloud.nasa.gov/frame.h5"]
            def get_s3_credentials_endpoint(self):
                return "https://nisar.asf.earthdatacloud.nasa.gov/s3credentials"

        g = _FakeGranule({"umm": {"GranuleUR": "G_STREAM"}, "meta": {"provider-id": "ASF"}})

        # 1. Redirected CloudFront/S3 pre-signed URL must NOT receive Bearer header
        auth_redirect = SimpleNamespace(
            authenticated=True,
            token={"access_token": "edl_tok_123"},
            get_session=lambda: _FakeSession("https://d1mv8.cloudfront.net/s3-abc/frame.h5?Signature=xyz"),
        )
        t_redir = ai._resolve_earthaccess_stream_target(
            g, SimpleNamespace(), lambda: auth_redirect, ai.NISAR_DEFAULT_SUBDATASET, "https"
        )
        self.assertEqual(t_redir.transport, "https")
        self.assertIsNone(t_redir.bearer_token)
        self.assertTrue(t_redir.gdal_uri.startswith('HDF5:"/vsicurl/https://d1mv8.cloudfront.net/'))
        self.assertTrue(t_redir.gdal_uri.endswith('://science/LSAR/GSLC/grids/frequencyA/HH'))

        # 2. Unredirected URL keeps bearer token and writes temporary GDAL_HTTP_HEADER_FILE
        auth_direct = SimpleNamespace(
            authenticated=True,
            token={"access_token": "edl_tok_123"},
            get_session=lambda: _FakeSession("https://nisar.asf.earthdatacloud.nasa.gov/frame.h5"),
        )
        t_direct = ai._resolve_earthaccess_stream_target(
            g, SimpleNamespace(), lambda: auth_direct, ai.NISAR_DEFAULT_SUBDATASET, "https"
        )
        self.assertEqual(t_direct.bearer_token, "edl_tok_123")
        with ai._stream_subprocess_env(t_direct, t_direct) as env:
            hdr_file = Path(env["GDAL_HTTP_HEADER_FILE"])
            self.assertTrue(hdr_file.is_file())
            self.assertIn("Authorization: Bearer edl_tok_123", hdr_file.read_text(encoding="utf-8"))
            self.assertEqual(env["VSI_CACHE"], "FALSE")
            self.assertEqual(env["CPL_VSIL_CURL_USE_HEAD"], "NO")
        self.assertFalse(hdr_file.exists())

        # 3. In-region S3 auto-selection produces /vsis3/ URI and temporary AWS_* credentials
        auth_s3 = SimpleNamespace(
            authenticated=True,
            get_s3_credentials=lambda **kw: {
                "accessKeyId": "AKIA_TEST",
                "secretAccessKey": "SECRET_TEST",
                "sessionToken": "SESS_TEST",
            },
        )
        ea_in_region = SimpleNamespace(__store__=SimpleNamespace(in_region=True))
        t_s3 = ai._resolve_earthaccess_stream_target(
            g, ea_in_region, lambda: auth_s3, ai.NISAR_DEFAULT_SUBDATASET, "auto"
        )
        self.assertEqual(t_s3.transport, "s3")
        self.assertEqual(
            t_s3.gdal_uri,
            'HDF5:"/vsis3/nisar-prod-bucket/path/frame.h5"://science/LSAR/GSLC/grids/frequencyA/HH',
        )
        with ai._stream_subprocess_env(t_s3, t_s3) as env_s3:
            self.assertEqual(env_s3["AWS_ACCESS_KEY_ID"], "AKIA_TEST")
            self.assertEqual(env_s3["AWS_SESSION_TOKEN"], "SESS_TEST")

    def test_bbox_to_window_projection_and_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            h5_path = Path(td) / "nisar_grid.h5"
            nx, ny = 2000, 4000
            with h5py.File(h5_path, "w") as f:
                g = f.require_group("/science/LSAR/GSLC/grids/frequencyA")
                # UTM Zone 30N covering part of West Yorkshire
                g.create_dataset("xCoordinates", data=570005.0 + np.arange(nx) * 10.0)
                g.create_dataset("yCoordinates", data=5955997.5 - np.arange(ny) * 5.0)
                proj = g.create_dataset("projection", data=np.int64(32630))
                proj.attrs["epsg_code"] = np.int64(32630)
                g.create_dataset("centerFrequency", data=np.float64(1.229e9))

            with h5py.File(h5_path, "r") as hf:
                grid = ai._read_grid_geometry_from_h5file(hf, "/science/LSAR/GSLC/grids/frequencyA/HH")
            self.assertIsNotNone(grid)
            win = ai.compute_window_from_bbox((-1.85, 53.55, -1.70, 53.68), grid)
            self.assertIsNotNone(win)
            x, y, w, h = win
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertLessEqual(x + w, nx)
            self.assertLessEqual(y + h, ny)

            # Non-overlapping bbox must return None
            disjoint = ai.compute_window_from_bbox((10.0, 10.0, 11.0, 11.0), grid)
            self.assertIsNone(disjoint)

    def test_streaming_mode_sync_site_zero_download_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "remote_h5"
            work_dir = td_path / "work_stream"
            nx, ny = 64, 64

            def _make_nisar_h5(p: Path, phase: float) -> Path:
                _write_complex_h5(
                    p,
                    "/science/LSAR/GSLC/grids/frequencyA/HH",
                    width=nx,
                    height=ny,
                    phase_rad=phase,
                    center_freq_hz=1.229e9,
                )
                with h5py.File(p, "a") as f:
                    g = f["/science/LSAR/GSLC/grids/frequencyA"]
                    g["HH"].attrs["grid_mapping"] = b"projection"
                    xc = g.create_dataset("xCoordinates", data=578005.0 + np.arange(nx) * 10.0)
                    xc.attrs["standard_name"] = b"projection_x_coordinate"
                    xc.attrs["units"] = b"m"
                    yc = g.create_dataset("yCoordinates", data=5946997.5 - np.arange(ny) * 5.0)
                    yc.attrs["standard_name"] = b"projection_y_coordinate"
                    yc.attrs["units"] = b"m"
                    proj = g.create_dataset("projection", data=np.int64(32630))
                    proj.attrs["epsg_code"] = np.int64(32630)
                    proj.attrs["spatial_ref"] = b"EPSG:32630"
                return p

            h1 = _make_nisar_h5(src_dir / "n1.h5", 0.1)
            h2 = _make_nisar_h5(src_dir / "n2.h5", 0.5)
            url_map = {
                "https://remote.example/n1.h5": h1,
                "https://remote.example/n2.h5": h2,
            }

            class _FakeStreamGranule(dict):
                def __init__(self, ur: str, start: str, url: str) -> None:
                    super().__init__(
                        {
                            "meta": {"concept-id": ur, "provider-id": "ASF"},
                            "umm": {
                                "GranuleUR": ur,
                                "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": start}},
                                "AdditionalAttributes": [
                                    {"Name": "TRACK_NUMBER", "Values": ["153"]},
                                    {"Name": "FRAME_NUMBER", "Values": ["61"]},
                                    {"Name": "ASCENDING_DESCENDING", "Values": ["DESCENDING"]},
                                ],
                            },
                        }
                    )
                    self._url = url
                def data_links(self, access: str = "external"):
                    return [self._url] if access == "external" else []

            class _FakeStreamSession:
                def get(self, url: str, **kwargs):
                    return contextlib.nullcontext(SimpleNamespace(url=url, raise_for_status=lambda: None))

            class _FakeStreamEA:
                def login(self, strategy: str = "all"):
                    return SimpleNamespace(
                        authenticated=True,
                        token={"access_token": "tok"},
                        get_session=lambda: _FakeStreamSession(),
                    )
                def search_data(self, **kwargs):
                    return [
                        _FakeStreamGranule("NISAR_L2_PR_GSLC_033_153_D_061_20261022", "2026-10-22T19:38:41Z", "https://remote.example/n1.h5"),
                        _FakeStreamGranule("NISAR_L2_PR_GSLC_034_153_D_061_20261103", "2026-11-03T19:38:42Z", "https://remote.example/n2.h5"),
                    ]
                def download(self, *args, **kwargs):
                    raise AssertionError("download() must not be called in stream=True mode")

            fake_fsspec = SimpleNamespace(
                filesystem=lambda proto, **kw: SimpleNamespace(
                    open=lambda url, mode="rb", **okw: open(url_map[url], mode)
                )
            )
            site = ai.SiteConfig(
                site_id="yorkshire_stream",
                sensor="nisar",
                bbox=(-1.85, 53.55, -1.70, 53.68),
                stride_x=2,
                stride_y=2,
                stream=True,
                stream_transport="https",
            )

            captured_cmds: list[tuple[list[str], dict[str, str]]] = []
            real_run = ai.subprocess.run

            def _intercept_run(cmd, *args, **kwargs):
                if str(cmd[0]) == str(ENGINE_BIN):
                    captured_cmds.append((list(cmd), dict(kwargs.get("env") or {})))
                    # Substitute local HDF5 files into real insar_proto to avoid TCP socket inside sandbox
                    local_cmd = list(cmd)
                    local_cmd[local_cmd.index("--slc1") + 1] = f'HDF5:"{h1}"://science/LSAR/GSLC/grids/frequencyA/HH'
                    local_cmd[local_cmd.index("--slc2") + 1] = f'HDF5:"{h2}"://science/LSAR/GSLC/grids/frequencyA/HH'
                    return real_run(local_cmd, *args, **{k: v for k, v in kwargs.items() if k != "env"})
                return real_run(cmd, *args, **kwargs)

            with patch.object(ai.subprocess, "run", side_effect=_intercept_run):
                sum1 = ai.sync_site(
                    site, work_dir, ENGINE_BIN, ea_mod=_FakeStreamEA(), fsspec_mod=fake_fsspec
                )
                sum2 = ai.sync_site(
                    site, work_dir, ENGINE_BIN, ea_mod=_FakeStreamEA(), fsspec_mod=fake_fsspec
                )

            self.assertEqual(sum1.downloaded_count, 0)
            self.assertEqual(sum1.pairs_processed, 1)
            self.assertEqual(sum2.pairs_processed, 0)
            self.assertEqual(sum2.pairs_skipped, 1)
            self.assertFalse((work_dir / "raw" / "yorkshire_stream").exists())
            self.assertEqual(len(captured_cmds), 1)
            cmd, env = captured_cmds[0]
            self.assertIn('HDF5:"/vsicurl/https://remote.example/n1.h5"://science/LSAR/GSLC/grids/frequencyA/HH', cmd)
            self.assertIn("--window", cmd)
            self.assertEqual(env["VSI_CACHE"], "FALSE")
            self.assertEqual(env["GDAL_SKIP"], "NETCDF")

            pair_dir = work_dir / "products" / "yorkshire_stream" / "NISAR_T153_F061_D_2026-10-22_2026-11-03"
            self.assertTrue((pair_dir / "displacement_mm.tif").is_file())
            summary_json = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["width"], 32)
            self.assertEqual(summary_json["height"], 32)


if __name__ == "__main__":
    unittest.main()
