"""Repository root detection, runtime paths, and legacy path mapping helpers."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def repo_root() -> Path:
    """Return the repository root directory."""
    env_root = os.getenv("INSAR_REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() or (candidate / "cpp").is_dir():
        return candidate
    return Path.cwd().resolve()


REPO_ROOT = repo_root()
PYTHON_ROOT = REPO_ROOT / "python"
VAR_ROOT = REPO_ROOT / "var"
DEFAULT_WORK_DIR = Path("var/work")
DEFAULT_DATA_DIR = Path("./var/api_data")
DEFAULT_DEM_CACHE_DIR = Path("var/work/dem/tiles")
DEFAULT_GNSS_CACHE_DIR = Path("var/work/gnss/unr")
DEFAULT_CONFIG_PATH = Path("configs/sites.example.json")

_LEGACY_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("insar_work/", "var/work/"),
    ("nisar_run/", "var/nisar_run/"),
    ("api_data/", "var/api_data/"),
    ("out/", "var/out/pair/"),
    ("out_synth/", "var/out/synth/"),
    ("scripts/sites.example.json", "configs/sites.example.json"),
)


def default_data_dir() -> Path:
    """Return the API/worker state directory from INSAR_DATA_DIR or `./var/api_data`."""
    return Path(os.getenv("INSAR_DATA_DIR", str(DEFAULT_DATA_DIR)))


def resolve_runtime_path(raw_path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path while transparently remapping legacy root directories under `var/`."""
    p = Path(raw_path)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
        try:
            rel_posix = p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel_posix = ""
    else:
        base = base_dir if base_dir is not None else REPO_ROOT
        candidates.append(p)
        if base / p != p:
            candidates.append(base / p)
        rel_posix = p.as_posix().lstrip("./")

    for cand in candidates:
        if cand.exists():
            return cand

    if rel_posix:
        for old_prefix, new_prefix in _LEGACY_PREFIX_MAP:
            if rel_posix == old_prefix.rstrip("/") or rel_posix.startswith(old_prefix):
                mapped_rel = (
                    new_prefix + rel_posix[len(old_prefix) :]
                    if rel_posix.startswith(old_prefix)
                    else new_prefix.rstrip("/")
                )
                mapped = (
                    (base_dir / mapped_rel)
                    if (base_dir is not None and not p.is_absolute())
                    else (REPO_ROOT / mapped_rel)
                )
                if mapped.exists() or Path(mapped_rel).exists():
                    return mapped if mapped.exists() else Path(mapped_rel)
                return mapped

    return candidates[0]


def alt_python_env(*, set_mpl_cache: bool = True) -> dict[str, str]:
    """Build subprocess env with `python/` on `PYTHONPATH` and without `.venv` shadowing."""
    clean_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    py_path = str(PYTHON_ROOT)
    existing_pp = clean_env.get("PYTHONPATH", "")
    parts = [p for p in existing_pp.split(os.pathsep) if p]
    if py_path not in parts:
        clean_env["PYTHONPATH"] = f"{py_path}{os.pathsep}{existing_pp}" if existing_pp else py_path
    if set_mpl_cache:
        clean_env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "insar_mpl_cache"))
    return clean_env


def find_alt_python(required_modules: Sequence[str] = ("osgeo",)) -> Path | None:
    """Return `/opt/homebrew/bin/python3` (or `python3`) when scientific bindings are missing here."""
    missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    env_py = os.getenv("INSAR_ALT_PYTHON", "").strip()
    raw_candidates = [env_py] if env_py else ["/opt/homebrew/bin/python3", shutil.which("python3") or ""]
    for cand in raw_candidates:
        if not cand:
            continue
        alt_py = Path(cand)
        if alt_py.is_file() and os.path.abspath(sys.executable) != os.path.abspath(str(alt_py)):
            return alt_py
    return None


def ensure_runtime_python(
    entry_file: str,
    required_modules: Sequence[str] = ("osgeo",),
    argv: Sequence[str] | None = None,
    set_mpl_cache: bool = True,
) -> None:
    """Re-exec under system/Homebrew python if required scientific bindings are absent in `.venv`."""
    alt_py = find_alt_python(required_modules)
    if alt_py is None:
        return
    clean_env = alt_python_env(set_mpl_cache=set_mpl_cache)
    args = list(sys.argv[1:] if argv is None else argv)
    res = subprocess.run([str(alt_py), str(Path(entry_file).resolve()), *args], env=clean_env)
    raise SystemExit(res.returncode)
