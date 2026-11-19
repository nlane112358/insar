"""DEM provider contract and Copernicus GLO-30 adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class DemProvider(Protocol):
    def align_to_raster(self, raster_path: Path, output_path: Path) -> Path: ...


class CopernicusGlo30Provider:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def align_to_raster(self, raster_path: Path, output_path: Path) -> Path:
        from scripts.fetch_dem import (
            build_aligned_dem,
            download_cop30_tiles,
            read_raster_grid_and_wgs84_bbox,
            tiles_for_bbox,
        )

        grid = read_raster_grid_and_wgs84_bbox(raster_path)
        tiles = tiles_for_bbox(*grid["wgs84_bbox"])
        sources = download_cop30_tiles(tiles, self.cache_dir)
        return build_aligned_dem(sources, output_path, ref_grid=grid)
