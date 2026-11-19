"""Corridor provider contract and OpenStreetMap/ONS adapters."""

from __future__ import annotations

from typing import Any, Protocol


class CorridorProvider(Protocol):
    def fetch(self, bbox: tuple[float, float, float, float], **options: Any) -> dict[str, Any]: ...


class OverpassCorridorProvider:
    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        *,
        filters: list[str],
        max_features: int = 80,
        balance_classes: bool = False,
    ) -> dict[str, Any]:
        from scripts.fetch_corridors import build_overpass_query, elements_to_grouped_geojson, fetch_overpass_json

        query = build_overpass_query(bbox, filters)
        raw = fetch_overpass_json(query)
        return elements_to_grouped_geojson(raw, bbox, max_features=max_features, balance_classes=balance_classes)


class OnsZoneProvider:
    def fetch(self, bbox: tuple[float, float, float, float], **options: Any) -> dict[str, Any]:
        from scripts.fetch_corridors import fetch_ons_uk_zones

        return fetch_ons_uk_zones(bbox, timeout_s=int(options.get("timeout_s", 30)))
