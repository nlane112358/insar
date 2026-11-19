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
    "london_thames": {
        "bbox": (-0.62, 51.31, 0.16, 51.69),
        "max_features": 80,
        "balance_classes": True,
        "queries": [
            'way["railway"~"^(rail|subway|light_rail|tram)$"]["name"]',
            'way["aeroway"="runway"]',
            'way["highway"="motorway"]',
            'way["highway"="trunk"]["name"]',
            'way["waterway"~"^(canal|river)$"]["name"]',
            'way["natural"="water"]["water"="reservoir"]["name"]',
            'relation["natural"="water"]["water"="reservoir"]["name"]',
            'way["waterway"="dam"]',
            'way["man_made"~"^(water_works|wastewater_plant|reservoir_covered|pumping_station)$"]["name"]',
            'relation["man_made"~"^(water_works|wastewater_plant|reservoir_covered)$"]["name"]',
            'way["man_made"="pipeline"]["substance"="water"]["name"]',
        ],
    },
    "east_midlands_coalfields": {
        "bbox": (-1.65, 52.86, -0.96, 53.36),
        "max_features": 80,
        "balance_classes": True,
        "queries": [
            'way["railway"~"^(rail|light_rail|tram)$"]["name"]',
            'way["highway"="motorway"]',
            'way["highway"="trunk"]["name"]',
            'way["waterway"~"^(canal|river)$"]["name"]',
            'way["natural"="water"]["water"="reservoir"]["name"]',
            'relation["natural"="water"]["water"="reservoir"]["name"]',
            'way["waterway"="dam"]',
            'way["man_made"~"^(water_works|wastewater_plant|reservoir_covered|pumping_station)$"]["name"]',
            'relation["man_made"~"^(water_works|wastewater_plant|reservoir_covered)$"]["name"]',
            'way["man_made"="pipeline"]["substance"="water"]["name"]',
        ],
    },
}


ONS_LAD_QUERY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_May_2024_Boundaries_UK_BGC/FeatureServer/0/query"
)

ZONE_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "london_thames": (-0.62, 51.31, 0.16, 51.69),
    "east_midlands_coalfields": (-1.65, 52.86, -0.96, 53.36),
}


def fetch_ons_uk_zones(
    bbox: tuple[float, float, float, float],
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Fetch official ONS UK Local Authority District boundaries (BGC) overlapping bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    params = urllib.parse.urlencode(
        {
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "LAD24CD,LAD24NM",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    req = urllib.request.Request(
        f"{ONS_LAD_QUERY_URL}?{params}",
        headers={"User-Agent": "insar-corridor-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    features: list[dict[str, Any]] = []
    for feat in raw.get("features", []):
        props = feat.get("properties") or {}
        name = str(props.get("LAD24NM") or "Unknown District")
        code = str(props.get("LAD24CD") or _clean_slug(name))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "asset_id": code,
                    "name": name,
                    "asset_class": "local_authority_district",
                    "lad_code": code,
                    "source": "ONS Open Geography Portal (May 2024 UK BGC)",
                },
                "geometry": feat.get("geometry"),
            }
        )
    features.sort(key=lambda f: f["properties"]["name"])
    return {
        "type": "FeatureCollection",
        "name": "uk_insurance_exposure_zones",
        "features": features,
    }


def build_overpass_query(bbox: tuple[float, float, float, float], filters: list[str], timeout_s: int = 120) -> str:
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


def fetch_overpass_json(query: str, urls: tuple[str, ...] = DEFAULT_OVERPASS_URLS, timeout_s: int = 135) -> dict[str, Any]:
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


_WATER_WORKS_LABELS: dict[str, str] = {
    "water_works": "Water Treatment Works",
    "wastewater_plant": "Wastewater Treatment Works",
    "reservoir_covered": "Service Reservoir",
    "pumping_station": "Pumping Station",
    "water_tower": "Water Tower",
    "pipeline": "Water Pipeline",
}

_BALANCED_CLASS_ORDER: tuple[str, ...] = (
    "rail",
    "metro",
    "light_rail",
    "dam",
    "reservoir",
    "water_works",
    "canal",
    "river",
    "motorway",
    "airport_runway",
    "trunk",
    "infrastructure",
)


def classify_way(tags: dict[str, str]) -> tuple[str, str]:
    """Return (asset_class, display_name) for an OSM way or relation."""
    name = tags.get("name") or tags.get("ref") or ""
    if "aeroway" in tags:
        rway = tags.get("ref") or tags.get("name") or f"runway_{tags.get('surface', 'paved')}"
        return "airport_runway", name or f"Runway {rway}"
    if "railway" in tags:
        rtype = tags["railway"]
        label = {"subway": "metro", "light_rail": "light_rail", "tram": "light_rail"}.get(rtype, "rail")
        return label, name or tags.get("operator") or f"Unnamed {label.title()}"
    if "highway" in tags:
        return tags["highway"], name or tags.get("ref") or f"Unnamed {tags['highway'].title()}"
    if tags.get("waterway") == "dam":
        return "dam", name or "Unnamed Dam"
    if tags.get("man_made") in _WATER_WORKS_LABELS:
        mm = tags["man_made"]
        return "water_works", name or tags.get("operator") or f"Unnamed {_WATER_WORKS_LABELS[mm]}"
    if tags.get("water") == "reservoir" or tags.get("natural") == "water":
        return "reservoir", name or "Unnamed Reservoir"
    if "waterway" in tags:
        return tags["waterway"], name or "Unnamed Waterway"
    if tags.get("boundary") == "administrative" or "place" in tags:
        return "residential_zone", name or "Unnamed District"
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


def _stitch_ring_ways(segments: list[list[list[float]]]) -> list[list[list[float]]]:
    """Assemble closed rings from possibly split OSM relation outer way segments."""
    rings: list[list[list[float]]] = []
    open_segs: list[list[list[float]]] = []
    for seg in segments:
        if len(seg) >= 4 and seg[0] == seg[-1]:
            rings.append(seg)
        elif len(seg) >= 2:
            open_segs.append(list(seg))

    while open_segs:
        cur = open_segs.pop(0)
        changed = True
        while changed and (cur[0] != cur[-1]):
            changed = False
            for idx, seg in enumerate(open_segs):
                if cur[-1] == seg[0]:
                    cur.extend(seg[1:])
                    open_segs.pop(idx)
                    changed = True
                    break
                if cur[-1] == seg[-1]:
                    cur.extend(list(reversed(seg[:-1])))
                    open_segs.pop(idx)
                    changed = True
                    break
                if cur[0] == seg[-1]:
                    cur = list(seg[:-1]) + cur
                    open_segs.pop(idx)
                    changed = True
                    break
                if cur[0] == seg[0]:
                    cur = list(reversed(seg[1:])) + cur
                    open_segs.pop(idx)
                    changed = True
                    break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def _select_balanced_keys(
    all_keys: list[tuple[str, str]],
    max_features: int,
) -> list[tuple[str, str]]:
    """Select up to max_features while ensuring diversity across infrastructure classes."""
    by_cls: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in all_keys:
        by_cls[k[0]].append(k)

    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    ordered_classes = [c for c in _BALANCED_CLASS_ORDER if c in by_cls] + [
        c for c in sorted(by_cls.keys()) if c not in _BALANCED_CLASS_ORDER
    ]
    pos = {c: 0 for c in ordered_classes}
    while len(selected) < max_features:
        progressed = False
        for c in ordered_classes:
            idx = pos[c]
            if idx < len(by_cls[c]):
                key = by_cls[c][idx]
                pos[c] = idx + 1
                if key not in seen:
                    selected.append(key)
                    seen.add(key)
                progressed = True
                if len(selected) >= max_features:
                    break
        if not progressed:
            break
    return selected


def elements_to_grouped_geojson(
    osm_data: dict[str, Any],
    bbox: tuple[float, float, float, float],
    max_features: int = 80,
    balance_classes: bool = False,
) -> dict[str, Any]:
    elements = osm_data.get("elements", [])
    nodes: dict[int, tuple[float, float]] = {}
    ways_by_id: dict[int, dict[str, Any]] = {}
    ways: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for el in elements:
        etype = el.get("type")
        if etype == "node" and "lon" in el and "lat" in el:
            nodes[int(el["id"])] = (float(el["lon"]), float(el["lat"]))
        elif etype == "way" and el.get("nodes"):
            ways_by_id[int(el["id"])] = el
            ways.append(el)
        elif etype == "relation" and el.get("members"):
            relations.append(el)

    grouped_lines: dict[tuple[str, str], list[list[list[float]]]] = defaultdict(list)
    grouped_polys: dict[tuple[str, str], list[list[list[list[float]]]]] = defaultdict(list)
    grouped_refs: dict[tuple[str, str], set[str]] = defaultdict(set)

    poly_classes = {"reservoir", "dam", "water_works", "residential_zone"}

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
        if is_closed and asset_class in poly_classes:
            grouped_polys[key].append([coords])
        else:
            grouped_lines[key].append(coords)

    for rel in relations:
        tags = rel.get("tags", {})
        if not tags:
            continue
        asset_class, name = classify_way(tags)
        if name.startswith("Unnamed ") and asset_class not in {"airport_runway", "dam", "water_works"}:
            continue
        outer_segs: list[list[list[float]]] = []
        for mem in rel.get("members", []):
            if mem.get("type") != "way":
                continue
            if mem.get("role", "") not in {"outer", ""}:
                continue
            w = ways_by_id.get(int(mem["ref"]))
            if not w:
                continue
            coords = [[round(nodes[nid][0], 6), round(nodes[nid][1], 6)] for nid in w["nodes"] if int(nid) in nodes]
            if len(coords) >= 2:
                outer_segs.append(coords)
        rings = _stitch_ring_ways(outer_segs)
        clipped_rings = [r for r in rings if _clip_line_to_bbox(r, bbox)]
        if not clipped_rings:
            continue
        key = (asset_class, name)
        if tags.get("ref"):
            grouped_refs[key].add(tags["ref"])
        if asset_class in poly_classes:
            for ring in clipped_rings:
                grouped_polys[key].append([ring])
        else:
            for ring in clipped_rings:
                grouped_lines[key].append(ring)

    features: list[dict[str, Any]] = []
    all_keys = sorted(
        set(grouped_lines.keys()) | set(grouped_polys.keys()),
        key=lambda k: (
            -(sum(len(seg) for seg in grouped_lines.get(k, [])) + sum(len(p[0]) for p in grouped_polys.get(k, []))),
            k[0],
            k[1],
        ),
    )
    selected_keys = _select_balanced_keys(all_keys, max_features) if balance_classes else all_keys[:max_features]

    for idx, key in enumerate(selected_keys, start=1):
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
    parser = argparse.ArgumentParser(description="Fetch real OSM infrastructure corridors or ONS UK zones as GeoJSON.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--preset", choices=sorted(PRESETS.keys()))
    grp.add_argument("--zones-preset", choices=sorted(ZONE_PRESETS.keys()))
    parser.add_argument("--out", required=True, help="Output GeoJSON path")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--balance-classes", action="store_true", help="Ensure balanced coverage across asset classes")
    args = parser.parse_args(argv)

    if args.zones_preset is not None:
        bbox = ZONE_PRESETS[args.zones_preset]
        print(f"Querying ONS Open Geography Portal for '{args.zones_preset}' bbox={bbox}...")
        fc = fetch_ons_uk_zones(bbox)
    else:
        spec = PRESETS[args.preset]
        max_features = args.max_features if args.max_features is not None else int(spec.get("max_features", 60))
        balance = args.balance_classes or bool(spec.get("balance_classes", False))
        query = build_overpass_query(spec["bbox"], spec["queries"])
        print(f"Querying Overpass API for preset '{args.preset}' bbox={spec['bbox']}...")
        osm_data = fetch_overpass_json(query)
        fc = elements_to_grouped_geojson(
            osm_data,
            spec["bbox"],
            max_features=max_features,
            balance_classes=balance,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(fc['features'])} grouped features to {out_path}")
    for feat in fc["features"][:15]:
        p = feat["properties"]
        print(f"  - [{p['asset_class']}] {p['name']} ({feat['geometry']['type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
