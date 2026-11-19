#!/usr/bin/env python3
"""End-to-end SDK example: submit a pair job, await completion, and update a time-series stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PY_ROOT = _REPO_ROOT / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from insar_platform.client import InsarClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo client for the unified InSAR OEM Compute & Stacking API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="insar_dev_secret_key")
    parser.add_argument("--slc1", required=True, help="Path to reference OPERA CSLC-S1 .h5 / NISAR GSLC / complex GeoTIFF")
    parser.add_argument("--slc2", required=True, help="Path to secondary OPERA CSLC-S1 .h5 / NISAR GSLC / complex GeoTIFF")
    parser.add_argument("--subdataset", default="/data/VV")
    parser.add_argument("--stack-id", default="mexico_city_demo_stack")
    parser.add_argument("--dt-days", type=float, default=12.0)
    args = parser.parse_args()

    with InsarClient(base_url=args.base_url, api_key=args.api_key) as client:
        health = client.health()
        print("API Health:", json.dumps(health, indent=2))

        job_res = client.create_job(
            {
                "monitor_name": "mexico_city_demo_corridor",
                "slc1_path": args.slc1,
                "slc2_path": args.slc2,
                "subdataset": args.subdataset,
                "stride_x": 6,
                "stride_y": 3,
                "ref_lonlat": [-99.215, 19.420],
                "ref_radius_m": 1000.0,
                "alert_rules": {"threshold_mm": 35.0, "min_coherence": 0.65},
                "webhook_url": f"{args.base_url}/v1/dev/webhook-sink",
            }
        )
        job_id = job_res["job_id"]
        print("Submitted pair job:", job_id)

        status_res = client.wait_job(job_id)
        print("Job status:", status_res["status"])

        if status_res["status"] == "completed":
            stac = client.get_stac_item(job_id)
            print("Pair STAC 1.1.0 displacement COG:", stac["assets"]["displacement_mm"]["href"])

            upd = client.update_stack(
                args.stack_id,
                {
                    "monitor_name": "Mexico City Demo Stack",
                    "source_job_id": job_id,
                    "dt_days": args.dt_days,
                    "ref_lonlat": [-99.215, 19.420],
                    "ref_radius_m": 1000.0,
                    "min_alert_snr": 2.5,
                    "min_alert_span_days": 60.0,
                    "webhook_url": f"{args.base_url}/v1/dev/webhook-sink",
                },
            )
            print("Submitted stack update:", upd["update_id"])

            stack_res = client.wait_stack(args.stack_id)
            print("Stack status:", stack_res["status"])
            print("\nStack Summary:")
            print(json.dumps(stack_res.get("summary", {}), indent=2))
            if stack_res["status"] == "completed":
                stack_stac = client.get_stack_stac_item(args.stack_id)
                print("Stack Velocity COG:", stack_stac["assets"]["velocity_mm_yr"]["href"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
