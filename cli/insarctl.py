"""Thin operator CLI for local platform services and legacy scientific commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx


def _forward(module_name: str, argv: list[str]) -> int:
    module = __import__(module_name, fromlist=["main"])
    return int(module.main(argv))


def _api_request(args: argparse.Namespace, path: str):
    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.api_url.rstrip("/"), timeout=30.0) as client:
        response = client.get(path, headers=headers)
        response.raise_for_status()
        return response.json()


def _api_post(args: argparse.Namespace, path: str, payload: dict):
    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.api_url.rstrip("/"), timeout=30.0) as client:
        response = client.post(path, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insarctl", description="Operate the InSAR processing platform")
    parser.add_argument("--api-url", default=os.getenv("INSAR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("INSAR_API_KEY", "insar_dev_secret_key"))
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("up", help="Build and start the local Compose stack")
    commands.add_parser("down", help="Stop the local Compose stack and retain its volumes")
    commands.add_parser("health", help="Show API readiness and storage/queue mode")

    screen = commands.add_parser("screen", help="Query monitors and infrastructure assets by WGS84 bbox")
    screen.add_argument("min_lon", type=float)
    screen.add_argument("min_lat", type=float)
    screen.add_argument("max_lon", type=float)
    screen.add_argument("max_lat", type=float)

    sync = commands.add_parser("sync-monitor", help="Queue a catalog discovery and processing pass")
    sync.add_argument("monitor_name")
    sync.add_argument("--engine-version", default="0.5.0")
    sync.add_argument("--idempotency-key")
    sync.add_argument("--webhook-url")

    migrate = commands.add_parser("migrate-sqlite", help="Import legacy API SQLite rows into configured PostGIS")
    migrate.add_argument("source_path", nargs="?", type=Path)

    ingest = commands.add_parser("ingest", help="Run the sensor ingestion command")
    ingest.add_argument("args", nargs=argparse.REMAINDER)
    stack = commands.add_parser("stack", help="Run the time-series stack synchronization command")
    stack.add_argument("args", nargs=argparse.REMAINDER)
    validate = commands.add_parser("validate-gnss", help="Run GNSS validation")
    validate.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"up", "down"}:
        project_dir = Path(__file__).resolve().parents[1]
        cmd = ["docker", "compose", "-f", str(project_dir / "docker-compose.yml")]
        cmd += ["up", "-d", "--build"] if args.command == "up" else ["down"]
        return subprocess.run(cmd, cwd=project_dir, check=False).returncode
    if args.command == "health":
        print(json.dumps(_api_request(args, "/health"), indent=2))
        return 0
    if args.command == "screen":
        query = (
            f"/v1/screen?min_lon={args.min_lon}&min_lat={args.min_lat}"
            f"&max_lon={args.max_lon}&max_lat={args.max_lat}"
        )
        print(json.dumps(_api_request(args, query), indent=2))
        return 0
    if args.command == "sync-monitor":
        payload = {
            key: value
            for key, value in {
                "engine_version": args.engine_version,
                "idempotency_key": args.idempotency_key,
                "webhook_url": args.webhook_url,
            }.items()
            if value is not None
        }
        result = _api_post(args, f"/v1/monitors/{args.monitor_name}/sync", payload)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "migrate-sqlite":
        from services.platform.storage.migrate import migrate_sqlite_to_postgres

        print(json.dumps(migrate_sqlite_to_postgres(args.source_path), indent=2))
        return 0
    if args.command == "ingest":
        return _forward("scripts.auto_ingest", args.args)
    if args.command == "stack":
        return _forward("scripts.stack_timeseries", args.args)
    if args.command == "validate-gnss":
        return _forward("scripts.validate_gnss", args.args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
