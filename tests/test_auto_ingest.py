#!/usr/bin/env python3
"""Unit and end-to-end tests for scripts/auto_ingest.py."""

from __future__ import annotations

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
        self.assertEqual([s.site_id for s in sites], ["mexico_city_iw2", "west_yorkshire_nisar"])
        self.assertEqual(sites[0].sensor, "opera")
        self.assertEqual(sites[1].sensor, "nisar")
        self.assertEqual(sites[1].bbox, (-1.85, 53.55, -1.70, 53.68))

    def test_nisar_track_filter_and_env_token_login(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src_dir = td_path / "src"
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

            self.assertEqual(fake_ea.login_strategies, ["environment"])
            self.assertEqual(len(cands), 1)
            self.assertEqual(cands[0].stream_key, "NISAR_T045_F012_A")


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

            p1 = _FakeASFProduct(scene_name="OPERA_G1", burst_id="T078-165486-IW2", start_time="2026-10-01T12:00:00Z", source_h5=f1)
            p2 = _FakeASFProduct(scene_name="OPERA_G2", burst_id="T078-165486-IW2", start_time="2026-10-13T12:00:00Z", source_h5=f2)
            p3 = _FakeASFProduct(scene_name="OPERA_G3", burst_id="T078-165486-IW2", start_time="2026-10-25T12:00:00Z", source_h5=f3)
            p4 = _FakeASFProduct(scene_name="OPERA_G4", burst_id="T078-165486-IW2", start_time="2026-11-06T12:00:00Z", source_h5=f4)

            # Run 1: 3 granules -> 2 pairs -> prune first 2, keep 3rd
            fake_asf_1 = _FakeASFModule([p1, p3, p2])  # deliberately unsorted
            sum1 = ai.sync_and_process(
                burst_id="T078-165486-IW2",
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

            pair1_dir = work_dir / "products" / "T078-165486-IW2_2026-10-01_2026-10-13"
            pair2_dir = work_dir / "products" / "T078-165486-IW2_2026-10-13_2026-10-25"
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
                burst_id="T078-165486-IW2",
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
                burst_id="T078-165486-IW2",
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
            self.assertTrue((work_dir / "products" / "T078-165486-IW2_2026-10-25_2026-11-06" / "displacement_mm.tif").is_file())

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
                    _FakeASFProduct(scene_name="O1", burst_id="T078-165486-IW2", start_time="2026-10-01T00:00:00Z", source_h5=f1),
                    _FakeASFProduct(scene_name="O2", burst_id="T078-165486-IW2", start_time="2026-10-13T00:00:00Z", source_h5=f2),
                ]
            )

            # Dry run must not create raw/ or state.db
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
            p1 = _FakeASFProduct(scene_name="BAD_1", burst_id="T078-165486-IW2", start_time="2026-10-01T00:00:00Z", source_h5=f1)
            p2 = _FakeASFProduct(scene_name="BAD_2", burst_id="T078-165486-IW2", start_time="2026-10-13T00:00:00Z", source_h5=f2)
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


if __name__ == "__main__":
    unittest.main()
