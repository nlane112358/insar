"""Small DB-API adapter for local SQLite and production PostGIS."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class DatabaseConnection:
    def __init__(self, connection: Any, dialect: str):
        self._connection = connection
        self.dialect = dialect

    def execute(self, statement: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
        if self.dialect == "postgresql":
            statement = statement.replace("?", "%s")
        return self._connection.execute(statement, params)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def database_url() -> str:
    return os.getenv("INSAR_DATABASE_URL", "").strip()


def dialect() -> str:
    return "postgresql" if database_url().startswith(("postgres://", "postgresql://")) else "sqlite"


def _connect() -> DatabaseConnection:
    url = database_url()
    if url.startswith(("postgres://", "postgresql://")):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg[binary]") from exc
        connection = psycopg.connect(url, row_factory=dict_row, connect_timeout=10)
        return DatabaseConnection(connection, "postgresql")

    data_dir = Path(os.getenv("INSAR_DATA_DIR", "./var/api_data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(os.getenv("INSAR_SQLITE_PATH", str(data_dir / "oem_jobs.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return DatabaseConnection(connection, "sqlite")


def connect_db() -> DatabaseConnection:
    """Open one connection using the configured primary state backend."""
    return _connect()


@contextmanager
def get_db() -> Iterator[DatabaseConnection]:
    connection = _connect()
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with get_db() as conn:
        if conn.dialect == "postgresql":
            conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        schemas = (
            """CREATE TABLE IF NOT EXISTS monitors (
                monitor_name TEXT PRIMARY KEY, config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS acquisitions (
                granule_id TEXT PRIMARY KEY, monitor_name TEXT NOT NULL, sensor TEXT NOT NULL,
                acquired_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                asset_uri TEXT, checksum TEXT)""",
            """CREATE TABLE IF NOT EXISTS granules (
                granule_id TEXT NOT NULL, burst_id TEXT NOT NULL, acq_time TEXT NOT NULL,
                local_path TEXT NOT NULL, sensor TEXT NOT NULL DEFAULT 'opera',
                site_id TEXT NOT NULL DEFAULT '', stream_key TEXT NOT NULL DEFAULT '',
                pruned INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (site_id, granule_id))""",
            """CREATE TABLE IF NOT EXISTS processed_pairs (
                ref_id TEXT NOT NULL, sec_id TEXT NOT NULL, out_dir TEXT NOT NULL,
                finished_at TEXT NOT NULL, site_id TEXT NOT NULL DEFAULT '',
                stream_key TEXT NOT NULL DEFAULT '', wavelength_mm REAL,
                PRIMARY KEY (site_id, ref_id, sec_id))""",
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, monitor_name TEXT NOT NULL, mode TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
                webhook_url TEXT, webhook_status TEXT, summary_json TEXT, error_message TEXT,
                engine_version TEXT NOT NULL DEFAULT '0.4.0', idempotency_key TEXT)""",
            """CREATE TABLE IF NOT EXISTS stacks (
                stack_id TEXT PRIMARY KEY, monitor_name TEXT NOT NULL, status TEXT NOT NULL,
                pair_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                webhook_url TEXT, webhook_status TEXT, summary_json TEXT, error_message TEXT)""",
            """CREATE TABLE IF NOT EXISTS stack_updates (
                update_id TEXT PRIMARY KEY, stack_id TEXT NOT NULL, source_job_id TEXT,
                status TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT,
                summary_json TEXT, error_message TEXT, engine_version TEXT NOT NULL DEFAULT '0.5.0',
                idempotency_key TEXT)""",
            """CREATE TABLE IF NOT EXISTS ingestion_runs (
                ingestion_id TEXT PRIMARY KEY, monitor_name TEXT NOT NULL,
                status TEXT NOT NULL, request_json TEXT NOT NULL, engine_version TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                finished_at TEXT, summary_json TEXT, error_message TEXT)""",
            """CREATE TABLE IF NOT EXISTS validation_metrics (
                metric_id TEXT PRIMARY KEY, monitor_name TEXT NOT NULL, job_id TEXT,
                measured_at TEXT NOT NULL, metrics_json TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS infrastructure_assets (
                asset_id TEXT PRIMARY KEY, monitor_name TEXT, name TEXT NOT NULL,
                asset_type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS webhook_deliveries (
                delivery_id TEXT PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, event_type TEXT NOT NULL,
                target_url TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                response_code INTEGER, created_at TEXT NOT NULL, delivered_at TEXT,
                error_message TEXT, payload_json TEXT NOT NULL DEFAULT '{}')""",
            """CREATE TABLE IF NOT EXISTS workflow_outbox (
                outbox_id TEXT PRIMARY KEY, task_name TEXT NOT NULL, payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL, published_at TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT)""",
            """CREATE TABLE IF NOT EXISTS workflow_steps (
                workflow_id TEXT NOT NULL, step_name TEXT NOT NULL, status TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, detail_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT, PRIMARY KEY (workflow_id, step_name))""",
            """CREATE TABLE IF NOT EXISTS api_usage (
                request_id TEXT PRIMARY KEY, api_key_hash TEXT NOT NULL, route TEXT NOT NULL,
                method TEXT NOT NULL, status_code INTEGER NOT NULL, latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL)""",
        )
        for statement in schemas:
            conn.execute(statement)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_outbox_pending ON workflow_outbox (published_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_created ON api_usage (created_at)")

        if conn.dialect == "postgresql":
            conn.execute("ALTER TABLE webhook_deliveries ADD COLUMN IF NOT EXISTS payload_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS engine_version TEXT NOT NULL DEFAULT '0.4.0'")
            conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs (idempotency_key)")
            conn.execute("ALTER TABLE stack_updates ADD COLUMN IF NOT EXISTS engine_version TEXT NOT NULL DEFAULT '0.5.0'")
            conn.execute("ALTER TABLE stack_updates ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_stack_updates_idempotency ON stack_updates (idempotency_key)")
            conn.execute("ALTER TABLE monitors ADD COLUMN IF NOT EXISTS footprint geometry(Geometry, 4326)")
            conn.execute("ALTER TABLE infrastructure_assets ADD COLUMN IF NOT EXISTS geom geometry(Geometry, 4326)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitors_footprint ON monitors USING GIST (footprint)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_infrastructure_assets_geom ON infrastructure_assets USING GIST (geom)")
            conn.execute("""CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY, owner_type TEXT NOT NULL, owner_id TEXT NOT NULL,
                asset_key TEXT NOT NULL, media_type TEXT NOT NULL, s3_uri TEXT NOT NULL,
                checksum TEXT, footprint geometry(Geometry, 4326), created_at TEXT NOT NULL,
                UNIQUE (owner_type, owner_id, asset_key))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_footprint ON assets USING GIST (footprint)")
        else:
            webhook_columns = {row["name"] for row in conn.execute("PRAGMA table_info(webhook_deliveries)").fetchall()}
            if "payload_json" not in webhook_columns:
                conn.execute("ALTER TABLE webhook_deliveries ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
            job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "engine_version" not in job_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN engine_version TEXT NOT NULL DEFAULT '0.4.0'")
            if "idempotency_key" not in job_columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN idempotency_key TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs (idempotency_key)")
            stack_columns = {row[1] for row in conn.execute("PRAGMA table_info(stack_updates)").fetchall()}
            if "engine_version" not in stack_columns:
                conn.execute("ALTER TABLE stack_updates ADD COLUMN engine_version TEXT NOT NULL DEFAULT '0.5.0'")
            if "idempotency_key" not in stack_columns:
                conn.execute("ALTER TABLE stack_updates ADD COLUMN idempotency_key TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_stack_updates_idempotency ON stack_updates (idempotency_key)")
            monitor_columns = {row["name"] for row in conn.execute("PRAGMA table_info(monitors)").fetchall()}
            if "footprint" not in monitor_columns:
                conn.execute("ALTER TABLE monitors ADD COLUMN footprint TEXT")
            asset_columns = {row["name"] for row in conn.execute("PRAGMA table_info(infrastructure_assets)").fetchall()}
            if "geom" not in asset_columns:
                conn.execute("ALTER TABLE infrastructure_assets ADD COLUMN geom TEXT")
            conn.execute("""CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY, owner_type TEXT NOT NULL, owner_id TEXT NOT NULL,
                asset_key TEXT NOT NULL, media_type TEXT NOT NULL, s3_uri TEXT NOT NULL,
                checksum TEXT, footprint TEXT, created_at TEXT NOT NULL,
                UNIQUE (owner_type, owner_id, asset_key))""")
        conn.commit()
