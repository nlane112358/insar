# insar-cpu-proto

Native C++20 + OpenMP InSAR processing and sequential $O(1)$ time-series stacking engine for Sentinel-1 (OPERA CSLC-S1) and NISAR (L-band GSLC) radar imagery. Converts coregistered complex SLC pairs into calibrated millimeter-scale line-of-sight (LOS) deformation rasters, Cloud-Optimized GeoTIFFs, STAC 1.1.0 items, and SNR/maturity-gated linear infrastructure corridor risk alerts.

---

## Ground-Truth GNSS Validation — Mexico City (`T041_086877_IW1`)

<p align="center">
  <img src="docs/figures/mexico_city_gnss_map.png" alt="GNSS and InSAR LOS velocity map — Mexico City IW1" width="32%" />
  <img src="docs/figures/mexico_city_gnss_scatter.png" alt="GNSS versus InSAR LOS velocity scatter plot" width="32%" />
  <img src="docs/figures/mexico_city_gnss_timeseries.png" alt="GNSS and InSAR cumulative displacement time series at MMX1" width="32%" />
</p>

*Independent validation of a 24-pair Sentinel-1 OPERA CSLC-S1 stack (19 Jun – 17 Nov 2026, 145 stacked days) against 6 Nevada Geodetic Laboratory (UNR) North-America-fixed GNSS stations projected onto exact Sentinel-1 zero-Doppler LOS vectors (from OPERA ECEF orbit state vectors). Full per-station tables: [`data/validation/mexico_city_gnss_validation.json`](data/validation/mexico_city_gnss_validation.json) · [`mexico_city_gnss_validation.csv`](data/validation/mexico_city_gnss_validation.csv).*

| Pipeline Configuration | RMSE (mm/yr) | MAE (mm/yr) | Bias (mm/yr) | $R^2$ | Pearson $r$ | `MMX1` (2026 overlap, −215.0 mm/yr GNSS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (whole-scene median, unmasked order-1 ramp, no DEM) | 64.0 | 42.5 | +37.8 | 0.321 | 0.883 | −70.3 mm/yr (err +144.7) |
| + Bedrock reference pin (`-99.215, 19.420`, $r = 1\text{ km}$, order-0) | 34.6 | 31.2 | −12.2 | 0.802 | 0.932 | −234.5 mm/yr (err −19.5) |
| **+ Copernicus GLO-30 DEM tropo + 2-pass deformation masking** | **18.3** | **15.1** | **−3.0** | **0.945** | **0.973** | **−213.3 mm/yr (err +1.7)** |

- **71.5% RMSE reduction** (64.0 → **18.3 mm/yr**) and **$R^2 = 0.945$** across all 6 independent stations (the bedrock reference pin in Lomas de Chapultepec is distinct from all validation stations).
- **Live 2026 cumulative overlay at `MMX1` (Mexico City Airport):** **7.7 mm** epoch-to-epoch RMSE across 25 Sentinel-1 passes against 142 daily UNR IGS20 GNSS observations tracking −85 mm of active clay-basin subsidence.

---

## Outputs

### Mexico City — Sentinel-1 C-Band Sequential Stack (`T041_086877_IW1`)

![Line-of-Sight Surface Velocity — Mexico City IW1](docs/figures/mexico_city_iw1_velocity_map.png)

*24 consecutive OPERA CSLC-S1 pairs (19 Jun – 17 Nov 2026, 145 stacked days), connected-component bedrock reference pinning, two-pass adaptive high-deformation masking, Copernicus GLO-30 DEM tropospheric regression, DEM hillshade blended into velocity luminance, confidence-weighted opacity, and 100 m buffered Metro, Rail, and Motorway corridor overlays with $|v|/\sigma_v \ge 2.5$ SNR gating (suppressing false-positive critical alerts across incoherent mountain unwrapping boundaries while preserving true clay-basin alerts on Línea VK, Línea A, Línea B, and Circuito Interior).*

### West Yorkshire — NISAR L-Band (`NISAR_T052_F061_D`)

<p align="center">
  <img src="docs/figures/west_yorkshire_t052_velocity_map.png" alt="Line-of-Sight Surface Velocity — West Yorkshire T052" width="640" />
</p>

*12-day L-band NISAR GSLC pair (27 Oct – 08 Nov 2026) over Huddersfield and the Holme Valley with buffered Canal, Rail, Motorway, River, Dam, and Reservoir corridors.*

### Single-Pair Diagnostics (`insar_proto`)

![NISAR InSAR Output Preview](nisar_preview.png)

*Pair-level outputs for `NISAR_T052_F061_D_2026-10-27_2026-11-08`: wrapped interferometric phase $\psi \in [-\pi, \pi]$, spatial coherence $\gamma \in [0, 1]$, and unwrapped LOS displacement (mm) with high-coherence anomaly alerts.*

### Extended Stack Visualizations

Additional figures are generated from the checked-in stack, corridor, GNSS, DEM, and NISAR pair products:

| Figure | Contents |
| :--- | :--- |
| [`mexico_city_corridor_heatmap.png`](docs/figures/mexico_city_corridor_heatmap.png) | Corridor p95 absolute LOS velocity by acquisition date; label color shows the latest risk class. |
| [`mexico_city_stack_evolution.png`](docs/figures/mexico_city_stack_evolution.png) · [`mexico_city_stack_evolution.gif`](docs/figures/mexico_city_stack_evolution.gif) | Nine static snapshots plus a pass-by-pass animation synchronized to the selected GNSS station's LOS displacement series. |
| [`mexico_city_velocity_confidence.png`](docs/figures/mexico_city_velocity_confidence.png) | Significant LOS velocity, velocity SNR, and standard error. |
| [`mexico_city_velocity_gradient.png`](docs/figures/mexico_city_velocity_gradient.png) | Engineering view of LOS velocity gradients in mm/yr per 100 m with rail, metro, road, and water corridors. |
| [`mexico_city_velocity_hillshade.png`](docs/figures/mexico_city_velocity_hillshade.png) | Velocity with DEM relief blended into the luminance channel and confidence-weighted opacity. |
| [`mexico_city_basin_transect.png`](docs/figures/mexico_city_basin_transect.png) | Smoothed velocity, cumulative displacement, elevation, and coherence from the bedrock reference to MMX1. |
| [`mexico_city_gnss_residuals.png`](docs/figures/mexico_city_gnss_residuals.png) | Station residual map plus residual relationships with coherence and InSAR spatial spread. |
| [`west_yorkshire_nisar_track_comparison.png`](docs/figures/west_yorkshire_nisar_track_comparison.png) | T052 and T029 pair displacement and coherence in their respective LOS geometries. |

![Mexico City stack evolution synchronized with MMX1 GNSS validation](docs/figures/mexico_city_stack_evolution.gif)

Regenerate the extended figures that use existing exports with:

```bash
python3 scripts/plots/plot_extended_visualizations.py
```

The stack evolution plot replays the stack into a temporary directory to capture corrected per-epoch rasters:

```bash
python3 scripts/plots/plot_stack_evolution.py
```

The replay reads settings from `stack_ledger.json`, writes the static panel and synchronized GIF by default, and leaves the existing stack outputs untouched. It uses the selected station's `.tenv3` file and LOS vector from the validation CSV; pass `--no-animation` to render only the static panel.

Install the optional scientific colormaps and GIF writer with:

```bash
python3 -m pip install -r scripts/requirements-plot.txt
```

The diverging maps fall back to the built-in palette if `cmcrameri` is unavailable. GIF export requires Pillow; use an `.mp4` output path when FFmpeg is installed.

### Interactive site reports

- [Mexico City · Sentinel-1 C-band](docs/interactive/mexico_city.html)
- [West Yorkshire · NISAR L-band](docs/interactive/west_yorkshire.html)

Each HTML embeds its stack COGs and feature data for building-scale zoom and corridor/station popups. The map libraries and basemap tiles are loaded from public CDNs.

---

## Architecture

| Component | Path | Role |
| :--- | :--- | :--- |
| **Shared C++ Core** | [`include/insar/`](include/insar) · [`src/`](src) · [`bindings/python_module.cpp`](bindings/python_module.cpp) | `libinsar` shared library with pairwise and stack APIs; nanobind runs the full applications in-process and exposes a zero-copy CPU NumPy/DLPack interferogram kernel |
| **Online Stacker** | [`src/stack/`](src/stack) · `insar_stack` | $O(1)$ per-pass updates to a 13-band Float64 sufficient-statistics raster (`stack_state.tif`), connected-component reference pinning, explicit/two-pass high-deformation masking, joint robust polynomial + DEM tropospheric regression (`--dem`), skip-1/skip-2 triplet phase closure masking & corrupted-pair rejection, velocity/uncertainty/SNR/acceleration COGs, and SNR + stack-maturity gated corridor risk analytics |
| **Platform Domain Services** | [`services/platform/`](services/platform) | Provider adapters, ingestion and stack services, GNSS validation, STAC builders, object storage, database, and durable workflow boundaries; legacy `scripts/` commands remain compatibility shims |
| **Control Plane** | [`api/server.py`](api/server.py) | FastAPI monitor/job/stack APIs, PostGIS spatial screening, STAC catalog, S3 range reads, API key checks, and transactional outbox enqueueing |
| **Compute Plane** | [`services/worker/`](services/worker) · [`services/platform/workflows/`](services/platform/workflows) | Celery workers execute typed ingestion, pairwise, and stack tasks; Redis-backed scheduling retries work after transient failures |
| **State & Artifacts** | [`services/platform/storage/`](services/platform/storage) · [`docker-compose.yml`](docker-compose.yml) | PostGIS is the system of record; MinIO/S3 stores artifact objects and checksums; SQLite/filesystem mode remains available for a standalone developer run |
| **DEM & GNSS Validation** | [`services/platform/catalog/dem_service.py`](services/platform/catalog/dem_service.py) · [`services/platform/validation/`](services/platform/validation) | Copernicus GLO-30 COG tile fetch/alignment and UNR/NGL GNSS LOS ground-truth validation with exact zero-Doppler orbit projection and standalone map, scatter, and time-series figures |
| **Cartography** | [`scripts/plots/plot_raster_map.py`](scripts/plots/plot_raster_map.py) · [`scripts/plots/plot_extended_visualizations.py`](scripts/plots/plot_extended_visualizations.py) · [`scripts/plots/plot_stack_evolution.py`](scripts/plots/plot_stack_evolution.py) | Standalone raster maps, corridor heatmaps, confidence diagnostics, GNSS residuals, transects, track comparisons, and replayed stack evolution panels |
| **Interactive Maps** | [`scripts/plots/export_interactive_map.py`](scripts/plots/export_interactive_map.py) · [`api/server.py`](api/server.py) | Embedded-COG reports with basemap toggle, opacity and swipe controls, clickable corridor/GNSS charts, and live job previews at `/preview/{job_id}` |
| **Operator CLI** | [`cli/insarctl.py`](cli/insarctl.py) | Thin commands for Compose lifecycle, health and screening queries, monitor sync, and legacy local scientific commands |

---

## Quickstart

**Prerequisites:** CMake $\ge$ 3.18, a C++20 compiler (OpenMP optional), GDAL, and Python 3 (`fastapi`, `uvicorn`, `httpx`, `pydantic` for the API; `numpy`, `matplotlib`, and `osgeo` for plotting & GNSS validation).

### Local production-style stack

The Compose stack starts PostGIS, Redis, MinIO, the API, a C++ compute worker, and the outbox scheduler. Copy `.env.example` to `.env`, set local API and webhook secrets (and Earthdata credentials when required), then run:

```bash
cp .env.example .env
set -a; source .env; set +a
docker compose up -d --build
python3 -m pip install -e .
insarctl health
```

The `ingest` Python extra installs the ASF/Earthaccess clients for a local worker (`python3 -m pip install -e '.[ingest]'`). Raster-backed GNSS and DEM operations also need system GDAL Python bindings.

Register a monitor with its GeoJSON footprint and provider settings in `config`, then queue its daily idempotent discovery pass:

```bash
curl -X POST http://127.0.0.1:8000/v1/monitors \
  -H "X-API-Key: $INSAR_API_KEY" -H 'Content-Type: application/json' \
  -d '{"monitor_name":"site-a","geometry":{"type":"Point","coordinates":[-99.13,19.43]},"config":{"sensor":"opera","burst_id":"T041_086877_IW1","lookback_days":45,"max_granules":6,"stack":true}}'
insarctl sync-monitor site-a
```

The API stores the request and transactional outbox record in PostGIS; Celery dispatches it to the worker. Track progress at the returned `/v1/ingestions/{ingestion_id}` URL. `GET /v1/screen` data comes from PostGIS indexes, and result assets are served from MinIO with HTTP range support.

To import a legacy API database, keep its SQLite file available locally and run `insarctl migrate-sqlite /path/to/oem_jobs.db` with `INSAR_DATABASE_URL` pointing at the Compose PostGIS database. The import skips rows already present in Postgres.

API and worker containers keep only temporary working files. Put direct pair-job inputs in the configured S3/MinIO bucket and pass `s3://bucket/key` paths; the C++ worker reads those rasters through GDAL's S3 virtual filesystem. Ingestion workers stream provider inputs or download them into disposable scratch space, then publish pair and stack assets back to object storage.

To use the C++ binding in a local build, install `python3 -m pip install -e '.[native]'` in the selected Python environment and configure CMake with `-DINSAR_BUILD_PYTHON=ON`; the Compose worker image builds and installs this binding automatically. The ndarray interface accepts contiguous CPU arrays through DLPack (including NumPy and CPU PyTorch tensors); CUDA arrays are not supported by this CPU engine.

```bash
# Build C++ binaries (`insar_proto`, `insar_stack`) and run C++ + Python test suites
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -v
```

```bash
# 1. Run an interferometric pair with stable bedrock reference pinning -> COGs + alerts.geojson
./build/insar_proto \
  --slc1 ref.h5 --slc2 sec.h5 \
  --ref-lonlat -99.215 19.420 --ref-radius-m 1000 \
  --out-dir out_pair

# 2. Fetch and align Copernicus GLO-30 DEM to the InSAR stack grid
python3 scripts/fetch_dem.py \
  --target-raster out_pair/displacement_mm.tif \
  --out insar_work/dem/site_cop30.tif

# 3. Incrementally update a multi-pass stack (DEM tropo + reference pin + 2-pass deform mask
#    + automatic triplet closure when skip-1/skip-2 pairs exist + SNR/maturity corridor gating)
python3 scripts/stack_timeseries.py \
  --products-dir insar_work/products/mexico_city_iw1 \
  --out-dir insar_work/stacks/mexico_city_iw1 \
  --dem insar_work/dem/mexico_city_iw1_cop30.tif \
  --ref-lonlat -99.215 19.420 --ref-radius-m 1000 \
  --ramp-order 0 --two-pass-deform-mask \
  --corridors data/corridors/mexico_city_corridors.geojson \
  --min-alert-snr 2.5 --min-alert-span-days 60

# 4. Validate against UNR/NGL GNSS stations and render standalone benchmark figures
python3 scripts/validate_gnss.py \
  --stack-dir insar_work/stacks/mexico_city_iw1 \
  --products-dir insar_work/products/mexico_city_iw1 \
  --dem insar_work/dem/mexico_city_iw1_cop30.tif \
  --slc-h5 insar_work/raw/mexico_city_iw1 \
  --out-json data/validation/mexico_city_gnss_validation.json \
  --map-out docs/figures/mexico_city_gnss_map.png \
  --scatter-out docs/figures/mexico_city_gnss_scatter.png \
  --timeseries-out docs/figures/mexico_city_gnss_timeseries.png

# 5. Render a standalone cartographic velocity map
python3 scripts/plots/plot_raster_map.py \
  --stack-dir insar_work/stacks/mexico_city_iw1 \
  --layer velocity --focus-urban

# 6. Export standalone interactive maps (raster and vector payloads are embedded in each HTML)
python3 scripts/plots/export_interactive_map.py \
  --site-name "Mexico City · Sentinel-1 C-band" \
  --stack-dir insar_work/stacks/mexico_city_iw1 \
  --corridors data/corridors/mexico_city_corridors.geojson \
  --gnss-json data/validation/mexico_city_gnss_validation.json \
  --output docs/interactive/mexico_city.html
python3 scripts/plots/export_interactive_map.py \
  --site-name "West Yorkshire · NISAR L-band" \
  --stack-dir insar_work/stacks/west_yorkshire_t052 \
  --corridors data/corridors/west_yorkshire_corridors.geojson \
  --output docs/interactive/west_yorkshire.html
```

The reports provide satellite/dark basemap selection, raster opacity, a velocity/coherence swipe, clickable corridor displacement histories, and GNSS station charts where validation data are available. The HTML embeds the raster and feature payloads; Leaflet, Chart.js, and basemap tiles load from public CDNs.

### Unified OEM HTTP API & Docker

```bash
# Run locally (serves both /v1/jobs via insar_proto and /v1/stacks/{stack_id}/updates via insar_stack)
uvicorn api.server:app --host 0.0.0.0 --port 8000

# The API image is intended to run with the worker and durable services via Compose.
docker compose up -d --build
```

Key endpoints: `GET /health`, `GET /v1/admin/usage`, `POST /v1/monitors`, `POST /v1/monitors/{monitor_name}/sync`, `GET /v1/ingestions/{ingestion_id}`, `POST /v1/jobs`, `GET /v1/jobs/{job_id}`, `GET /preview/{job_id}`, `GET /stac/items/{job_id}`, `POST /v1/stacks/{stack_id}/updates`, `GET /v1/stacks`, `GET /v1/stacks/{stack_id}`, `GET /stac/stacks/{stack_id}`, `GET /assets/{job_id}/{file}`, and `GET /stack-assets/{stack_id}/{file}`. The `POST /v1/jobs` response includes a `preview_url` immediately, including while the job is queued.

---

<details>
<summary><strong>Method & Mathematical Summary</strong></summary>

1. **Interferogram & Coherence:** For coregistered SLCs $s_1, s_2$ over a $(2r_x+1)\times(2r_y+1)$ window $\mathcal{N}$, $C = \sum_{\mathcal{N}} s_1 s_2^*$, wrapped phase $\psi = \operatorname{atan2}(\operatorname{Im} C, \operatorname{Re} C)$, and coherence $\gamma = |C| / \big(\sqrt{\sum_{\mathcal{N}} |s_1|^2 \sum_{\mathcal{N}} |s_2|^2} + \varepsilon\big)$.
2. **Coherence-Weighted Unwrapping:** Minimizes $\frac{1}{2}\sum_e w_e (\nabla_e \phi - \mathcal{W}(\nabla_e \psi))^2$ with cubic edge weights $w_e = \max(10^{-4}, \min(\gamma_u, \gamma_v)^3)$ via Preconditioned Conjugate Gradient (PCG), preconditioned by an exact half-sample mirrored 2D radix-2 FFT Neumann Poisson solver.
3. **LOS Displacement & Reference Pinning:** $d_{\text{LOS}} = \frac{\lambda}{4\pi}\phi$ ($\lambda = 55.4658\text{ mm}$ for Sentinel-1 C-band; dynamic HDF5 wavelength $\approx 238.4\text{ mm}$ for NISAR L-band), zero-referenced to the coherence-weighted median of a user-specified stable reference disk (`--ref-lonlat` / `--ref-px`) or GeoJSON/raster mask (`--ref-region`) rather than the whole-scene median.
4. **Connected-Component Isolation, High-Deformation Masking & DEM Tropospheric Correction:** Before fitting orbital/tropospheric nuisance terms, `insar_stack` restricts regression to the 4-connected coherent component anchored at the reference region (preventing cross-barrier $2\pi$ unwrapping jumps from leaking into regression) and masks active deformation zones via explicit polygons/rasters (`--deform-mask`), prior velocity (`--prior-vel` / `--two-pass-deform-mask`), and adaptive quantile capping. It jointly solves for a polynomial ramp plus ridge-damped elevation-correlated tropospheric delay $d_{\text{nuisance}}(x,y,z) = p_k(x,y) + \alpha\,\Delta z_{\text{km}}$ via Huber/Tukey IRLS, then re-pins the reference region to $0\text{ mm}$.
5. **Redundant Skip-1/Skip-2 Triplet Phase Closure:** When skip-1/skip-2 interferograms are available (`--max-skip 1|2`), `stack_timeseries.py` automatically supplies triangle loops to `insar_stack` to evaluate closure residuals $c = d_{AB} + d_{BC} - d_{AC}$ (datum-centered). Pixels exceeding `--closure-thresh-mm` (default $10\text{ mm}$) across a majority of triplets are masked before state updates, and pairs exceeding `--closure-max-bad-frac` (default $35\%$) are rejected cleanly before mutating `stack_state.tif`.
6. **Sequential $O(1)$ Stacking & SNR/Maturity-Gated Corridor Alerts:** Each accepted interferogram updates a persistent 13-band Float64 normal-equation state (`stack_state.tif`) in constant time per pass, producing exact WLS velocity $v$, standard error $\sigma_v$, signal-to-noise ratio $\text{SNR} = |v|/\sigma_v$ (`velocity_snr.tif`), acceleration, cumulative displacement, and temporal coherence. Critical corridor alerts require both sufficient velocity SNR ($|v|/\sigma_v \ge 2.5$) and minimum stack temporal maturity ($\ge 60\text{–}90\text{ days}$); immature stacks downgrade critical alerts to provisional warnings.

</details>
