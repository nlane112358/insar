import argparse
import json
import time
import httpx

parser = argparse.ArgumentParser(description="Demo client for the unified InSAR OEM Compute & Stacking API.")
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--api-key", default="insar_dev_secret_key")
parser.add_argument("--slc1", required=True, help="Path to reference OPERA CSLC-S1 .h5 / NISAR GSLC / complex GeoTIFF")
parser.add_argument("--slc2", required=True, help="Path to secondary OPERA CSLC-S1 .h5 / NISAR GSLC / complex GeoTIFF")
parser.add_argument("--subdataset", default="/data/VV")
parser.add_argument("--stack-id", default="mexico_city_demo_stack")
parser.add_argument("--dt-days", type=float, default=12.0)
args = parser.parse_args()

HEADERS = {"X-API-Key": args.api_key}

with httpx.Client(base_url=args.base_url, headers=HEADERS, timeout=60.0) as client:
    health = client.get("/health").json()
    print("API Health:", json.dumps(health, indent=2))

    # 1. Submit a single-pair job pinned to a stable bedrock reference region + signed webhook
    resp = client.post("/v1/jobs", json={
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
    })
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print("Submitted pair job:", job_id)

    while True:
        status_res = client.get(f"/v1/jobs/{job_id}").json()
        print("Job status:", status_res["status"])
        if status_res["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    if status_res["status"] == "completed":
        stac = client.get(f"/stac/items/{job_id}").json()
        print("Pair STAC 1.1.0 displacement COG:", stac["assets"]["displacement_mm"]["href"])

        # 2. Append the completed pair into a persistent O(1) multi-pass stack
        upd = client.post(f"/v1/stacks/{args.stack_id}/updates", json={
            "monitor_name": "Mexico City Demo Stack",
            "source_job_id": job_id,
            "dt_days": args.dt_days,
            "ref_lonlat": [-99.215, 19.420],
            "ref_radius_m": 1000.0,
            "min_alert_snr": 2.5,
            "min_alert_span_days": 60.0,
            "webhook_url": f"{args.base_url}/v1/dev/webhook-sink",
        })
        upd.raise_for_status()
        print("Submitted stack update:", upd.json()["update_id"])

        while True:
            stack_res = client.get(f"/v1/stacks/{args.stack_id}").json()
            print("Stack status:", stack_res["status"])
            if stack_res["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)

        print("\nStack Summary:")
        print(json.dumps(stack_res.get("summary", {}), indent=2))
        if stack_res["status"] == "completed":
            stack_stac = client.get(f"/stac/stacks/{args.stack_id}").json()
            print("Stack Velocity COG:", stack_stac["assets"]["velocity_mm_yr"]["href"])
