"""One-time import of the legacy API SQLite database into PostGIS."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from services.platform.storage.database import database_url, init_db
from services.platform.storage.database import get_db


TABLES = (
    "monitors",
    "acquisitions",
    "granules",
    "processed_pairs",
    "jobs",
    "stacks",
    "stack_updates",
    "ingestion_runs",
    "validation_metrics",
    "infrastructure_assets",
    "assets",
    "webhook_deliveries",
    "workflow_outbox",
    "workflow_steps",
    "api_usage",
)
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "monitors": ("monitor_name",),
    "acquisitions": ("granule_id",),
    "granules": ("site_id", "granule_id"),
    "processed_pairs": ("site_id", "ref_id", "sec_id"),
    "jobs": ("job_id",),
    "stacks": ("stack_id",),
    "stack_updates": ("update_id",),
    "ingestion_runs": ("ingestion_id",),
    "validation_metrics": ("metric_id",),
    "infrastructure_assets": ("asset_id",),
    "assets": ("owner_type", "owner_id", "asset_key"),
    "webhook_deliveries": ("delivery_id",),
    "workflow_outbox": ("outbox_id",),
    "workflow_steps": ("workflow_id", "step_name"),
    "api_usage": ("request_id",),
}
GEOMETRY_COLUMNS = {"monitors": "footprint", "infrastructure_assets": "geom", "assets": "footprint"}


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}


def _postgres_columns(connection: Any, table: str) -> set[str]:
    rows = connection.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_schema = current_schema() AND table_name = ?""",
        (table,),
    ).fetchall()
    return {row["column_name"] for row in rows}


def _copy_table(source: sqlite3.Connection, destination: Any, table: str) -> tuple[int, int]:
    source_columns = _sqlite_columns(source, table)
    destination_columns = _postgres_columns(destination, table)
    geometry_column = GEOMETRY_COLUMNS.get(table)
    columns = sorted((source_columns & destination_columns) - ({geometry_column} if geometry_column else set()))
    if not columns:
        return 0, 0

    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'
    rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
    copied = 0
    for source_row in rows:
        row = dict(source_row)
        values = tuple(row.get(column) for column in columns)
        cursor = destination.execute(sql, values)
        inserted = cursor.rowcount > 0
        copied += int(inserted)

        if inserted and geometry_column and geometry_column in source_columns and geometry_column in destination_columns:
            geometry = row.get(geometry_column)
            if geometry:
                keys = PRIMARY_KEYS[table]
                assignments = f'"{geometry_column}" = ST_SetSRID(ST_GeomFromGeoJSON(?::text), 4326)'
                predicates = " AND ".join(f'"{key}" = ?' for key in keys)
                destination.execute(
                    f'UPDATE "{table}" SET {assignments} WHERE {predicates}',
                    (geometry, *(row.get(key) for key in keys)),
                )
    return len(rows), copied


def migrate_sqlite_to_postgres(source_path: Path | None = None) -> dict[str, dict[str, int]]:
    """Copy known legacy tables without overwriting rows already present in Postgres."""
    url = database_url()
    if not url.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("Set INSAR_DATABASE_URL to a Postgres/PostGIS connection before migration")
    data_dir = Path(os.getenv("INSAR_DATA_DIR", "./api_data"))
    source_path = source_path or Path(os.getenv("INSAR_SQLITE_PATH", str(data_dir / "oem_jobs.db")))
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy SQLite database not found: {source_path}")

    init_db()
    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    try:
        source_tables = _sqlite_tables(source)
        report: dict[str, dict[str, int]] = {}
        with get_db() as destination:
            if destination.dialect != "postgresql":
                raise RuntimeError("Migration target changed from Postgres to SQLite")
            for table in TABLES:
                if table not in source_tables:
                    continue
                seen, inserted = _copy_table(source, destination, table)
                report[table] = {"source_rows": seen, "inserted_rows": inserted}
            destination.commit()
        return report
    finally:
        source.close()
