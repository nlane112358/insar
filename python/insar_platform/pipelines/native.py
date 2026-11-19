"""In-process libinsar adapter with an explicit local CLI compatibility mode."""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any, Literal

_NATIVE_ENV_LOCK = threading.RLock()


def _import_insar_core() -> Any:
    try:
        from insar_platform import _insar_core

        return _insar_core
    except ImportError:
        import _insar_core

        return _insar_core


def native_available() -> bool:
    try:
        _import_insar_core()
    except ImportError:
        return False
    return True


def estimate_interferogram(
    reference: Any,
    secondary: Any,
    *,
    stride_x: int = 1,
    stride_y: int = 1,
    look_rx: int = 3,
    look_ry: int = 3,
) -> tuple[Any, Any, Any, int]:
    """Run the zero-copy CPU ndarray kernel on contiguous complex64 DLPack inputs."""
    try:
        core = _import_insar_core()
    except ImportError as exc:
        raise RuntimeError("The _insar_core nanobind module is unavailable") from exc
    return core.estimate_interferogram(
        reference,
        secondary,
        stride_x,
        stride_y,
        look_rx,
        look_ry,
    )


def run_native_command(
    command: list[str],
    engine: Literal["pairwise", "stack"],
    *,
    environment: dict[str, str] | None = None,
) -> str:
    backend = os.getenv("INSAR_COMPUTE_BACKEND", "auto").lower()
    if backend != "subprocess":
        try:
            core = _import_insar_core()
        except ImportError:
            if backend == "libinsar":
                raise RuntimeError("INSAR_COMPUTE_BACKEND=libinsar but _insar_core is unavailable")
        else:
            run = core.run_pairwise if engine == "pairwise" else core.run_stack
            if environment is None:
                code = int(run([str(value) for value in command[1:]]))
            else:
                # GDAL reads credentials and VSI tuning from process environment.
                # Serialize updates while the native call is active, then restore
                # the worker environment exactly as it was.
                with _NATIVE_ENV_LOCK:
                    previous = dict(os.environ)
                    try:
                        os.environ.clear()
                        os.environ.update(environment)
                        code = int(run([str(value) for value in command[1:]]))
                    finally:
                        os.environ.clear()
                        os.environ.update(previous)
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
            return ""

    proc = subprocess.run(command, capture_output=True, text=True, check=True, env=environment)
    return proc.stdout or ""
