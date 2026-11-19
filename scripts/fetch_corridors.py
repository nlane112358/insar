#!/usr/bin/env python3
"""
Fetch and curate real linear/polygon infrastructure corridors from OpenStreetMap Overpass API.

Produces grouped GeoJSON FeatureCollections (one Feature per named corridor/asset)
ready for C++ `insar_stack --corridors ...` corridor pinning and alerting.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


PRESETS: dict[str, dict[str, Any]] = {
    "mexico_city": {
        "bbox": (-99.841, 19.308, -98.895, 19.750),  # min_lon, min_lat, max_lon, max_lat
        "queries": [
            'way["railway"~"^(rail|subway|light_rail)$"]["name"]',
            'way["aeroway"="runway"]',
            'way["highway"="motorway"]["name"]',
            'way["waterway"~"^(canal|drain)$"]["name"]',
        ],
    },
    "west_yorkshire": {
        "bbox": (-1.854, 53.548, -1.696, 53.682),
        "queries": [
            'way["railway"~"^(rail|light_rail)$"]',
            'way["highway"~"^(motorway|trunk)$"]',
            'way["waterway"~"^(canal|river)$"]["name"]',
            'way["natural"="water"]["water"="reservoir"]["name"]',
            'way["waterway"="dam"]',
        ],
    },
}


def build_overpass_query(bbox: tuple[float, float, float, float], filters: list[str], timeout_s: int = 60) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    lines = [f"[out:json][timeout:{timeout_s}];", "("]
    for flt in filters:
        lines.append(f"  {flt}({bbox_str});")
    lines.append(");")
    lines.append("out body;")
    lines.append(">;")
    lines.append("out skel qt;")
    return "\n".join(lines)


def fetch_overpass_json(query: str, urls: tuple[str, ...] = DEFAULT_OVERPASS_URLS, timeout_s: int = 75) -> dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err: Exception | None = None
    for url in urls:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"User-Agent": "insar-corridor-fetcher/1.0", "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
    raise RuntimeError(f"Overpass query failed across {urls}: {last_err}")


def classify_way(tags: dict[str, str]) -> tuple[str, str]:
    """Return (asset_class, display_name) for an OSM way."""
    name = tags.get("name") or tags.get("ref") or ""
    if "aeroway" in tags:
        rway = tags.get("ref") or tags.get("name") or f"runway_{tags.get('surface', 'paved')}"
        return "airport_runway", name or f"Runway {rway}"
    if "railway" in tags:
        rtype = tags["railway"]
        label = {"subway": "metro", "light_rail": "light_rail"}.get(rtype, "rail")
        return label, name or tags.get("operator") or f"Unnamed {label.title()}"
    if "highway" in tags:
        return tags["highway"], name or tags.get("ref") or f"Unnamed {tags['highway'].title()}"
    if tags.get("waterway") == "dam":
        return "dam", name or "Unnamed Dam"
    if tags.get("water") == "reservoir" or tags.get("natural") == "water":
        return "reservoir", name or "Unnamed Reservoir"
    if "waterway" in tags:
        return tags["waterway"], name or "Unnamed Waterway"
    return "infrastructure", name or "Unnamed Asset"


def _clean_slug(text: str) -> str:
    out = []
    prev_us = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out).strip("_") or "asset"


def _clip_line_to_bbox(coords: list[list[float]], bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return any(min_lon <= x <= max_lon and min_lat <= y <= max_lat for x, y in coords)


def elements_to_grouped_geojson(
    osm_data: dict[str, Any],
    bbox: tuple[float, float, float, float],
    max_features: int = 80,
) -> dict[str, Any]:
    elements = osm_data.get("elements", [])
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict[str, Any]] = []
    for el in elements:
        if el.get("type") == "node" and "lon" in el and "lat" in el:
            nodes[int(el["id"])] = (float(el["lon"]), float(el["lat"]))
        elif el.get("type") == "way" and el.get("nodes"):
            ways.append(el)

    grouped_lines: dict[tuple[str, str], list[list[list[float]]]] = defaultdict(list)
    grouped_polys: dict[tuple[str, str], list[list[list[list[float]]]]] = defaultdict(list)
    grouped_refs: dict[tuple[str, str], set[str]] = defaultdict(set)

    for way in ways:
        tags = way.get("tags", {})
        if not tags:
            continue
        asset_class, name = classify_way(tags)
        if name.startswith("Unnamed ") and asset_class not in {"airport_runway", "dam", "rail"}:
            continue
        coords = [[round(nodes[nid][0], 6), round(nodes[nid][1], 6)] for nid in way["nodes"] if int(nid) in nodes]
        if len(coords) < 2 or not _clip_line_to_bbox(coords, bbox):
            continue

        key = (asset_class, name)
        if tags.get("ref"):
            grouped_refs[key].add(tags["ref"])

        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        if is_closed and asset_class in {"reservoir", "dam"}:
            grouped_polys[key].append([coords])
        else:
            grouped_lines[key].append(coords)

    features: list[dict[str, Any]] = []
    all_keys = sorted(
        set(grouped_lines.keys()) | set(grouped_polys.keys()),
        key=lambda k: (
            -(sum(len(seg) for seg in grouped_lines.get(k, [])) + sum(len(p[0]) for p in grouped_polys.get(k, []))),
            k[0],
            k[1],
        ),
    )

    for idx, key in enumerate(all_keys[:max_features], start=1):
        asset_class, name = key
        refs = sorted(grouped_refs.get(key, set()))
        asset_id = f"{asset_class}_{_clean_slug(name)}"
        if asset_id in {f["properties"]["asset_id"] for f in features}:
            asset_id = f"{asset_id}_{idx}"
        props = {
            "asset_id": asset_id,
            "name": name,
            "asset_class": asset_class,
            "ref": ",".join(refs) if refs else None,
            "source": "OpenStreetMap Overpass",
        }
        if key in grouped_polys:
            polys = grouped_polys[key]
            geom = (
                {"type": "Polygon", "coordinates": polys[0]}
                if len(polys) == 1
                else {"type": "MultiPolygon", "coordinates": polys}
            )
        else:
            lines = grouped_lines[key]
            geom = (
                {"type": "LineString", "coordinates": lines[0]}
                if len(lines) == 1
                else {"type": "MultiLineString", "coordinates": lines}
            )
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    return {
        "type": "FeatureCollection",
        "name": "infrastructure_corridors",
        "features": features,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real OSM infrastructure corridors as grouped GeoJSON.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument("--out", required=True, help="Output GeoJSON path")
    parser.add_argument("--max-features", type=int, default=60)
    args = parser.parse_args(argv)

    spec = PRESETS[args.preset]
    query = build_overpass_query(spec["bbox"], spec["queries"])
    print(f"Querying Overpass API for preset '{args.preset}' bbox={spec['bbox']}...")
    osm_data = fetch_overpass_json(query)
    fc = elements_to_grouped_geojson(osm_data, spec["bbox"], max_features=args.max_features)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(fc['features'])} grouped corridor features to {out_path}")
    for feat in fc["features"][:15]:
        p = feat["properties"]
        print(f"  - [{p['asset_class']}] {p['name']} ({feat['geometry']['type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
