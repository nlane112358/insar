#!/usr/bin/env python3
"""Integration tests for unified api/server.py (`insar_proto` + `insar_stack`)."""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_BIN = REPO_ROOT / "build" / "insar_proto"
STACK_BIN = REPO_ROOT / "build" / "insar_stack"


def _write_complex_slc_tif(path: Path, phase_rad: float, width: int = 64, height: int = 64) -> Path:
    yy, xx = np.mgrid[0:height, 0:width]
    bowl = -0.8 * np.exp(-(((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * 10.0**2)))
    data = np.exp(1j * (0.05 * xx + 0.03 * yy + phase_rad * bowl)).astype(np.complex64)
    raw_bin = path.with_suffix(".envi.bin")
    raw_hdr = path.with_suffix(".envi.hdr")
    data.tofile(raw_bin)
    raw_hdr.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {width}",
                f"lines = {height}",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 6",
                "interleave = bsq",
                "byte order = 0",
                "map info = {UTM, 1.0, 1.0, 500000.0, 2150000.0, 15.0, 15.0, 14, North, WGS-84, units=Meters}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["gdal_translate", "-q", "-of", "GTiff", "-a_srs", "EPSG:32614", str(raw_bin), str(path)],
        check=True,
    )
    raw_bin.unlink(missing_ok=True)
    raw_hdr.unlink(missing_ok=True)
    return path


def _write_utm_f32(path: Path, arr: np.ndarray) -> None:
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
                "map info = {UTM, 1.0, 1.0, 500000.0, 2150000.0, 30.0, 30.0, 14, North, WGS-84, units=Meters}",
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


@unittest.skipUnless(
    ENGINE_BIN.is_file() and STACK_BIN.is_file(),
    "Requires built insar_proto and insar_stack binaries",
)
class TestUnifiedApiServer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self._old_env = {
            k: os.environ.get(k)
            for k in ("INSAR_ENGINE_BIN", "INSAR_STACK_BIN", "INSAR_DATA_DIR", "INSAR_API_KEY")
        }
        os.environ["INSAR_ENGINE_BIN"] = str(ENGINE_BIN)
        os.environ["INSAR_STACK_BIN"] = str(STACK_BIN)
        os.environ["INSAR_DATA_DIR"] = str(self.tmp_dir / "api_data")
        os.environ["INSAR_API_KEY"] = "test_api_secret"

        import api.server as srv

        self.srv = importlib.reload(srv)
        self.client = TestClient(self.srv.app)
        self.headers = {"X-API-Key": "test_api_secret"}

    def tearDown(self) -> None:
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_health_and_auth(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["engine_ready"])
        self.assertTrue(body["stack_ready"])

        bad = self.client.get("/v1/stacks", headers={"X-API-Key": "wrong"})
        self.assertEqual(bad.status_code, 401)

    def test_pair_job_with_ref_pin_and_unified_stack_lifecycle(self) -> None:
        slc1 = _write_complex_slc_tif(self.tmp_dir / "slc1.tif", phase_rad=0.0)
        slc2 = _write_complex_slc_tif(self.tmp_dir / "slc2.tif", phase_rad=1.0)
        slc3 = _write_complex_slc_tif(self.tmp_dir / "slc3.tif", phase_rad=2.0)

        # 1. Create single-pair job with explicit pixel reference region pin
        r_job = self.client.post(
            "/v1/jobs",
            headers=self.headers,
            json={
                "monitor_name": "slc_ref_test",
                "slc1_path": str(slc1),
                "slc2_path": str(slc2),
                "stride_x": 2,
                "stride_y": 2,
                "ref_px": [6, 6, 4],
            },
        )
        self.assertEqual(r_job.status_code, 202)
        job_id = r_job.json()["job_id"]

        r_status = self.client.get(f"/v1/jobs/{job_id}", headers=self.headers)
        self.assertEqual(r_status.status_code, 200)
        job_data = r_status.json()
        self.assertEqual(job_data["status"], "completed", msg=job_data.get("error_message"))
        self.assertTrue(job_data["summary"]["ref_region_pinned"])
        self.assertGreater(job_data["summary"]["ref_pixels_used"], 10)

        # 2. Initialize a persistent stack using source_job_id (immature 30-day span)
        stack_id = "mexico_api_test"
        r_u1 = self.client.post(
            f"/v1/stacks/{stack_id}/updates",
            headers=self.headers,
            json={
                "monitor_name": "Mexico API Corridor",
                "source_job_id": job_id,
                "t0_days": 0.0,
                "t1_days": 30.0,
                "ref_px": [6, 6, 4],
                "min_alert_snr": 2.5,
                "min_alert_span_days": 60.0,
            },
        )
        self.assertEqual(r_u1.status_code, 202)

        s1 = self.client.get(f"/v1/stacks/{stack_id}", headers=self.headers).json()
        self.assertEqual(s1["status"], "completed", msg=s1.get("error_message"))
        self.assertEqual(s1["pair_count"], 1)
        self.assertFalse(s1["summary"]["maturity_gating"]["stack_mature"])
        self.assertTrue(s1["summary"]["reference_region"]["applied"])

        # 3. Append second pass inline via pair_job reaching 65 days -> mature stack!
        r_u2 = self.client.post(
            f"/v1/stacks/{stack_id}/updates",
            headers=self.headers,
            json={
                "pair_job": {
                    "monitor_name": "inline_pass_2",
                    "slc1_path": str(slc2),
                    "slc2_path": str(slc3),
                    "stride_x": 2,
                    "stride_y": 2,
                    "ref_px": [6, 6, 4],
                },
                "t0_days": 30.0,
                "t1_days": 65.0,
                "ref_px": [6, 6, 4],
                "min_alert_snr": 2.5,
                "min_alert_span_days": 60.0,
            },
        )
        self.assertEqual(r_u2.status_code, 202)

        s2 = self.client.get(f"/v1/stacks/{stack_id}", headers=self.headers).json()
        self.assertEqual(s2["status"], "completed", msg=s2.get("error_message"))
        self.assertEqual(s2["pair_count"], 2)
        self.assertTrue(s2["summary"]["maturity_gating"]["stack_mature"])
        self.assertEqual(len(s2["recent_updates"]), 2)

        # Verify STAC 1.1.0 item & streamable stack assets
        stac = self.client.get(f"/stac/stacks/{stack_id}").json()
        self.assertEqual(stac["stac_version"], "1.1.0")
        self.assertIn("velocity_mm_yr", stac["assets"])
        self.assertIn("velocity_snr", stac["assets"])
        self.assertTrue(stac["properties"]["insar:stack_mature"])

        head_vel = self.client.head(f"/stack-assets/{stack_id}/velocity_mm_yr.tif")
        self.assertEqual(head_vel.status_code, 200)

        # Verify list endpoint
        listed = self.client.get("/v1/stacks", headers=self.headers).json()
        self.assertEqual([s["stack_id"] for s in listed["stacks"]], [stack_id])

    def test_stack_api_rejects_corrupted_pair_via_triplet_closure(self) -> None:
        h, w = 48, 48
        coh = np.full((h, w), 0.9, dtype=np.float32)
        d_ab = np.full((h, w), -2.0, dtype=np.float32)
        d_bc = np.full((h, w), -2.0, dtype=np.float32)
        d_ac = np.full((h, w), -4.0, dtype=np.float32)
        # Corrupt 60% of d_ab_bad with a +27.7 mm (2*pi) unwrapping jump
        d_ab_bad = d_ab.copy()
        d_ab_bad[:30, :] += 27.73

        p_ab = self.tmp_dir / "ab.tif"
        p_ab_bad = self.tmp_dir / "ab_bad.tif"
        p_bc = self.tmp_dir / "bc.tif"
        p_ac = self.tmp_dir / "ac.tif"
        p_coh = self.tmp_dir / "coh.tif"
        for p, arr in ((p_ab, d_ab), (p_ab_bad, d_ab_bad), (p_bc, d_bc), (p_ac, d_ac), (p_coh, coh)):
            _write_utm_f32(p, arr)

        # First stack clean BC pair
        r1 = self.client.post(
            "/v1/stacks/closure_api_test/updates",
            headers=self.headers,
            json={
                "disp_path": str(p_bc),
                "coh_path": str(p_coh),
                "t0_days": 6.0,
                "t1_days": 12.0,
                "ramp_order": 0,
            },
        )
        self.assertEqual(r1.status_code, 202)
        s1 = self.client.get("/v1/stacks/closure_api_test", headers=self.headers).json()
        self.assertEqual(s1["pair_count"], 1)

        # Now attempt to stack corrupted AB pair with triplet closure against BC & AC
        r2 = self.client.post(
            "/v1/stacks/closure_api_test/updates",
            headers=self.headers,
            json={
                "disp_path": str(p_ab_bad),
                "coh_path": str(p_coh),
                "t0_days": 0.0,
                "t1_days": 6.0,
                "ramp_order": 0,
                "closure_triplets": [
                    {"disp_a_path": str(p_bc), "disp_b_path": str(p_ac), "mode": "add_sub"}
                ],
                "closure_thresh_mm": 10.0,
                "closure_max_bad_frac": 0.35,
            },
        )
        self.assertEqual(r2.status_code, 202)
        s2 = self.client.get("/v1/stacks/closure_api_test", headers=self.headers).json()
        self.assertEqual(s2["status"], "completed")
        self.assertEqual(s2["pair_count"], 1)  # Rejected pair did NOT increment pair_count!
        self.assertTrue(s2["summary"]["triplet_closure"]["pair_rejected"])
        self.assertEqual(s2["summary"]["rejected_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
