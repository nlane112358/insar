#!/usr/bin/env python3
"""
Download and align Copernicus GLO-30 (30 m) or local/SRTM DEM GeoTIFFs to an InSAR raster grid.

Copernicus GLO-30 tiles are hosted publicly without authentication on AWS Open Data:
  https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_<NS><LAT>_00_<EW><LON>_00_DEM/...tif
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


COP30_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name_for_latlon(lat_deg: int, lon_deg: int) -> str:
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def cop30_tile_url(lat_deg: int, lon_deg: int) -> str:
    name = tile_name_for_latlon(lat_deg, lon_deg)
    return f"{COP30_BASE_URL}/{name}/{name}.tif"


def tiles_for_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    pad_deg: float = 0.02,
) -> list[tuple[int, int]]:
    """Return sorted list of integer (lat_floor, lon_floor) 1x1 deg DEM tiles overlapping bbox."""
    lo_lat = math.floor(min_lat - pad_deg)
    hi_lat = math.floor(max_lat + pad_deg)
    lo_lon = math.floor(min_lon - pad_deg)
    hi_lon = math.floor(max_lon + pad_deg)
    out: list[tuple[int, int]] = []
    for lat in range(lo_lat, hi_lat + 1):
        for lon in range(lo_lon, hi_lon + 1):
            out.append((lat, lon))
    return out


def read_raster_grid_and_wgs84_bbox(raster_path: Path) -> dict[str, object]:
    """Query raster width, height, geotransform, projection WKT, and WGS84 bbox via gdalinfo -json."""
    proc = subprocess.run(
        ["gdalinfo", "-json", str(raster_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    info = json.loads(proc.stdout)
    w, h = info["size"]
    gt = info["geoTransform"]
    proj_wkt = info.get("coordinateSystem", {}).get("wkt", "")
    wgs_extent = info.get("wgs84Extent", {})
    coords = []
    if wgs_extent.get("type") == "Polygon" and wgs_extent.get("coordinates"):
        coords = wgs_extent["coordinates"][0]
    if not coords:
        # Fallback via cornerCoordinates + gdaltransform
        cc = info.get("cornerCoordinates", {})
        pts = [cc[k] for k in ("upperLeft", "upperRight", "lowerRight", "lowerLeft") if k in cc]
        tr = subprocess.run(
            ["gdaltransform", "-s_srs", proj_wkt, "-t_srs", "EPSG:4326"],
            input="\n".join(f"{p[0]} {p[1]}" for p in pts) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        for line in tr.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])

    lons = [float(p[0]) for p in coords]
    lats = [float(p[1]) for p in coords]
    xmin = min(gt[0], gt[0] + w * gt[1])
    xmax = max(gt[0], gt[0] + w * gt[1])
    ymin = min(gt[3], gt[3] + h * gt[5])
    ymax = max(gt[3], gt[3] + h * gt[5])
    return {
        "width": int(w),
        "height": int(h),
        "geotransform": [float(v) for v in gt],
        "wkt": str(proj_wkt),
        "native_bbox": (xmin, ymin, xmax, ymax),
        "wgs84_bbox": (min(lons), min(lats), max(lons), max(lats)),
    }


def download_cop30_tiles(
    tiles: Iterable[tuple[int, int]],
    cache_dir: Path,
    timeout_s: float = 60.0,
) -> list[Path]:
    """Download Copernicus GLO-30 1x1 deg COG tiles atomically into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for lat, lon in tiles:
        name = tile_name_for_latlon(lat, lon)
        dest = cache_dir / f"{name}.tif"
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[DEM cache hit] {dest}")
            local_paths.append(dest)
            continue

        url = cop30_tile_url(lat, lon)
        print(f"Downloading Copernicus GLO-30 tile {name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "insar-cpu-proto/1.0"})
        tmp_path = dest.with_suffix(".tif.tmp")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                with open(tmp_path, "wb") as out_f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out_f.write(chunk)
            if tmp_path.stat().st_size == 0:
                tmp_path.unlink(missing_ok=True)
                raise IOError(f"Empty response for {url}")
            os.replace(tmp_path, dest)
            local_paths.append(dest)
        except urllib.error.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.code == 404:
                print(f"  Tile {name} not present (ocean/no-land 404), skipping.")
                continue
            raise
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return local_paths


def build_aligned_dem(
    source_rasters: list[Path],
    out_path: Path,
    ref_grid: dict[str, object] | None = None,
) -> Path:
    """Mosaic source DEM tiles and optionally warp/align to ref_grid as a Float32 GeoTIFF."""
    if not source_rasters:
        raise RuntimeError("No DEM tiles available to build output DEM")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="insar_dem_vrt_") as tmp:
        vrt_path = Path(tmp) / "mosaic.vrt"
        subprocess.run(
            ["gdalbuildvrt", "-q", str(vrt_path), *(str(p) for p in source_rasters)],
            check=True,
        )
        cmd = [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-r",
            "bilinear",
            "-ot",
            "Float32",
            "-dstnodata",
            "nan",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "PREDICTOR=3",
        ]
        if ref_grid is not None:
            xmin, ymin, xmax, ymax = ref_grid["native_bbox"]  # type: ignore[misc]
            wkt = str(ref_grid["wkt"])
            w = int(ref_grid["width"])  # type: ignore[arg-type]
            h = int(ref_grid["height"])  # type: ignore[arg-type]
            if wkt:
                wkt_file = Path(tmp) / "target.wkt"
                wkt_file.write_text(wkt, encoding="utf-8")
                cmd.extend(["-t_srs", str(wkt_file)])
            cmd.extend([
                "-te", f"{xmin:.6f}", f"{ymin:.6f}", f"{xmax:.6f}", f"{ymax:.6f}",
                "-ts", str(w), str(h),
            ])
        cmd.extend([str(vrt_path), str(out_path)])
        subprocess.run(cmd, check=True)
    print(f"Wrote aligned DEM -> {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch Copernicus GLO-30 DEM tiles (or align local SRTM/DEM) to an InSAR raster grid."
    )
    p.add_argument("--ref-raster", type=Path, help="Reference InSAR GeoTIFF whose CRS/extent/size to match")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Explicit WGS84 bounding box (used when --ref-raster is omitted)",
    )
    p.add_argument(
        "--local-dem",
        type=Path,
        nargs="+",
        help="Use existing local DEM GeoTIFF(s) (e.g. SRTM/Copernicus) instead of downloading",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("insar_work/dem/tiles"),
        help="Cache directory for downloaded Copernicus GLO-30 1x1 deg tiles",
    )
    p.add_argument("--out", required=True, type=Path, help="Output aligned DEM GeoTIFF path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.ref_raster is None and args.bbox is None and not args.local_dem:
        raise SystemExit("Provide --ref-raster or --bbox (or --local-dem)")

    ref_grid = read_raster_grid_and_wgs84_bbox(args.ref_raster) if args.ref_raster else None
    if args.local_dem:
        sources = [p for p in args.local_dem if p.is_file()]
        if not sources:
            raise FileNotFoundError(f"None of --local-dem paths exist: {args.local_dem}")
    else:
        bbox = tuple(args.bbox) if args.bbox else ref_grid["wgs84_bbox"]  # type: ignore[index]
        tiles = tiles_for_bbox(*bbox)
        print(f"Resolved {len(tiles)} Copernicus GLO-30 tile(s) for WGS84 bbox {bbox}: {tiles}")
        sources = download_cop30_tiles(tiles, args.cache_dir)

    build_aligned_dem(sources, args.out, ref_grid=ref_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
