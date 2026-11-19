"""Application service for running a configured sensor ingestion pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.platform.catalog.providers import sensor_provider


def run_site(site: Any, work_dir: Path, engine_bin: Path, **options: Any) -> Any:
    """Run legacy-compatible ingestion with a provider selected at the boundary."""
    workflow_id = options.pop("workflow_id", None)
    if workflow_id:
        from services.platform.workflows.state import mark_step

        mark_step(workflow_id, "DiscoverPass", "running", detail={"sensor": site.sensor})
    provider = sensor_provider(
        site.sensor,
        asf_module=options.pop("asf_mod", None),
        earthaccess_module=options.pop("ea_mod", None),
    )
    resolved_site, candidates = provider.discover(site)
    if workflow_id:
        mark_step(workflow_id, "DiscoverPass", "completed", detail={"granules_found": len(candidates)})

    # Keep sync_site's mature ledger, streaming, pair graph, and pruning policy
    # intact until those stateful steps are migrated to PostGIS workflows.
    from scripts import auto_ingest

    return auto_ingest.sync_site(
        resolved_site,
        work_dir,
        engine_bin,
        _catalog_candidates=candidates,
        _workflow_id=workflow_id,
        **options,
    )
