"""Compatibility module exporting `app` and runtime configuration attributes."""

from __future__ import annotations

from insar_platform.api.app import create_app
from insar_platform.api.deps import (
    get_api_key,
    get_base_url,
    get_data_dir,
    get_engine_bin,
    get_stack_bin,
    get_webhook_secret,
    verify_api_key,
)
from insar_platform.api.routes.assets import ALLOWED_ASSETS, ALLOWED_STACK_ASSETS
from insar_platform.storage.database import get_db, init_db

ENGINE_BIN = get_engine_bin()
STACK_BIN = get_stack_bin()
DATA_DIR = get_data_dir()
DB_PATH = DATA_DIR / "oem_jobs.db"
API_KEY = get_api_key()
WEBHOOK_SECRET = get_webhook_secret()
BASE_URL = get_base_url()

app = create_app()

__all__ = [
    "ALLOWED_ASSETS",
    "ALLOWED_STACK_ASSETS",
    "API_KEY",
    "BASE_URL",
    "DATA_DIR",
    "DB_PATH",
    "ENGINE_BIN",
    "STACK_BIN",
    "WEBHOOK_SECRET",
    "app",
    "create_app",
    "get_db",
    "init_db",
    "verify_api_key",
]
