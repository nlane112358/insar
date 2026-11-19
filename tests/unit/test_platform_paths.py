"""Unit tests for canonical repository path helpers and legacy `var/` remapping."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from insar_platform.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_WORK_DIR,
    REPO_ROOT,
    alt_python_env,
    default_data_dir,
    resolve_runtime_path,
)


class TestPlatformPaths(unittest.TestCase):
    def test_canonical_defaults_and_config_exist(self) -> None:
        self.assertEqual(DEFAULT_WORK_DIR, Path("var/work"))
        self.assertEqual(DEFAULT_DATA_DIR, Path("./var/api_data"))
        self.assertTrue((REPO_ROOT / DEFAULT_CONFIG_PATH).is_file())
        self.assertTrue((REPO_ROOT / "deploy" / "docker-compose.yml").is_file())
        self.assertTrue((REPO_ROOT / "deploy" / "Dockerfile").is_file())

    def test_legacy_path_remapping_into_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "var" / "work" / "stacks" / "demo"
            target.mkdir(parents=True)
            resolved = resolve_runtime_path(Path("insar_work/stacks/demo"), base_dir=root)
            self.assertEqual(resolved, target)

            cfg = root / "configs" / "sites.example.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_runtime_path("scripts/sites.example.json", base_dir=root), cfg)

    def test_alt_python_env_and_data_dir_override(self) -> None:
        with patch.dict(os.environ, {"INSAR_DATA_DIR": "/tmp/custom_insar_data", "PYTHONPATH": ""}, clear=False):
            self.assertEqual(default_data_dir(), Path("/tmp/custom_insar_data"))
            env = alt_python_env(set_mpl_cache=True)
            self.assertIn(str(REPO_ROOT / "python"), env["PYTHONPATH"].split(os.pathsep))
            self.assertNotIn("VIRTUAL_ENV", env)
            self.assertIn("MPLCONFIGDIR", env)


if __name__ == "__main__":
    unittest.main()
