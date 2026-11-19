"""Provider interfaces and adapters for existing sensor catalog clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SensorProvider(Protocol):
    sensor_name: str

    def discover(self, site: Any) -> tuple[Any, list[Any]]: ...


@dataclass
class OperaProvider:
    """ASF OPERA adapter; authentication and download logic stays injectable."""

    asf_module: Any = None
    sensor_name: str = "opera"

    def discover(self, site: Any) -> tuple[Any, list[Any]]:
        from scripts.auto_ingest import fetch_opera_candidates

        return fetch_opera_candidates(site, asf_mod=self.asf_module)


@dataclass
class NisarProvider:
    """NASA Earthdata/NISAR adapter with injectable Earthaccess client."""

    earthaccess_module: Any = None
    sensor_name: str = "nisar"

    def discover(self, site: Any) -> tuple[Any, list[Any]]:
        from scripts.auto_ingest import fetch_nisar_candidates

        return fetch_nisar_candidates(site, ea_mod=self.earthaccess_module)


def sensor_provider(sensor: str, *, asf_module: Any = None, earthaccess_module: Any = None) -> SensorProvider:
    normalized = sensor.strip().lower()
    if normalized == "opera":
        return OperaProvider(asf_module)
    if normalized == "nisar":
        return NisarProvider(earthaccess_module)
    raise ValueError(f"Unsupported sensor provider: {sensor}")
