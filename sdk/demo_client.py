import json
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "insar_dev_secret_key"}

with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=30.0) as client:
    # 1. Submit a job with alert rules + webhook pointing to our local sink
    resp = client.post("/v1/jobs", json={
        "monitor_name": "mexico_city_demo_corridor",
        "mode": "synthetic",
        "width": 900,
        "height": 700,
        "alert_rules": {"threshold_mm": 35.0, "min_coherence": 0.65},
        "webhook_url": f"{BASE_URL}/v1/dev/webhook-sink",
    })
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print("Submitted:", job_id)

    # 2. Poll until completed
    while True:
        status_res = client.get(f"/v1/jobs/{job_id}").json()
        print("Status:", status_res["status"])
        if status_res["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    print("\nJob Summary:")
    print(json.dumps(status_res, indent=2))

    # 3. Fetch the STAC 1.1.0 Item
    if status_res["status"] == "completed":
        stac = client.get(f"/stac/items/{job_id}").json()
        cog_url = stac["assets"]["displacement_mm"]["href"]
        print("\nSTAC 1.1.0 Item ready!")
        print(f"Stream directly in QGIS (Layer -> Add Raster -> Protocol HTTP): {cog_url}")
