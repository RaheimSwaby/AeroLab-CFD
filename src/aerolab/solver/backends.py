"""OpenFOAM backend detection and local command execution.

Handles discovering which solver backend is available (native shell, WSL2, or
Docker) and building the command line that runs a case through each one.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

OPENFOAM_BOOTSTRAP = r"""
if ! command -v foamRun >/dev/null 2>&1; then
  if [ -f /opt/openfoam13/etc/bashrc ]; then
    . /opt/openfoam13/etc/bashrc
  elif [ -f /usr/lib/openfoam/openfoam13/etc/bashrc ]; then
    . /usr/lib/openfoam/openfoam13/etc/bashrc
  elif [ -f "$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc" ]; then
    . "$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc"
  fi
fi
"""


def solver_status(timeout_seconds: int = 40) -> dict[str, object]:
    native_foam_run = shutil.which("foamRun")
    native_block_mesh = shutil.which("blockMesh")
    native_snappy = shutil.which("snappyHexMesh")
    native_features = shutil.which("surfaceFeatures")
    wsl_path = shutil.which("wsl")
    docker_path = shutil.which("docker")

    status: dict[str, object] = {
        "ok": True,
        "preferredBackend": None,
        "backends": {
            "native": {
                "available": bool(native_foam_run and native_block_mesh and native_snappy and native_features),
                "foamRun": native_foam_run,
                "blockMesh": native_block_mesh,
                "snappyHexMesh": native_snappy,
                "surfaceFeatures": native_features,
                "targetVersion": "OpenFOAM Foundation v13",
            },
            "wsl": {
                "available": False,
                "wsl": wsl_path,
                "openfoam": False,
            },
            "docker": {
                "available": False,
                "docker": docker_path,
                "image": os.environ.get("AEROLAB_OPENFOAM_IMAGE"),
            },
        },
    }

    if native_foam_run and native_block_mesh and native_snappy and native_features:
        status["preferredBackend"] = "native"

    if wsl_path:
        wsl_probe = _run_quick(["wsl", "--status"], timeout_seconds)
        wsl_available = False
        wsl_version: str | None = None
        wsl_message = _trim(wsl_probe.stderr or wsl_probe.stdout)
        if wsl_probe.returncode == 0:
            wsl_check = _run_quick(
                [
                    "wsl",
                    "bash",
                    "-lc",
                    f"{OPENFOAM_BOOTSTRAP}\n"
                    "command -v foamRun >/dev/null 2>&1 && "
                    "command -v blockMesh >/dev/null 2>&1 && "
                    "command -v snappyHexMesh >/dev/null 2>&1 && "
                    "command -v surfaceFeatures >/dev/null 2>&1 && "
                    "printf 'OPENFOAM_VERSION=%s' \"${WM_PROJECT_VERSION:-13}\"",
                ],
                timeout_seconds,
            )
            wsl_available = wsl_check.returncode == 0
            wsl_message = _trim(wsl_check.stderr or wsl_check.stdout)
            wsl_version = _openfoam_version(wsl_check.stdout)
        status["backends"]["wsl"]["available"] = wsl_available  # type: ignore[index]
        status["backends"]["wsl"]["openfoam"] = wsl_available  # type: ignore[index]
        status["backends"]["wsl"]["message"] = wsl_message  # type: ignore[index]
        status["backends"]["wsl"]["version"] = wsl_version  # type: ignore[index]
        status["backends"]["wsl"]["targetVersion"] = "OpenFOAM Foundation v13"  # type: ignore[index]
        if wsl_available and status["preferredBackend"] is None:
            status["preferredBackend"] = "wsl"

    docker_image = os.environ.get("AEROLAB_OPENFOAM_IMAGE")
    if docker_path and docker_image:
        docker_check = _run_quick(["docker", "image", "inspect", docker_image], timeout_seconds)
        docker_available = docker_check.returncode == 0
        status["backends"]["docker"]["available"] = docker_available  # type: ignore[index]
        status["backends"]["docker"]["message"] = _trim(docker_check.stderr or docker_check.stdout)  # type: ignore[index]
        if docker_available and status["preferredBackend"] is None:
            status["preferredBackend"] = "docker"

    return status


def _select_backend(status: dict[str, object], backend: str) -> str:
    if backend != "auto":
        backends = status.get("backends", {})
        if isinstance(backends, dict) and isinstance(backends.get(backend), dict):
            if backends[backend].get("available"):  # type: ignore[index]
                return backend
        raise RuntimeError(f"OpenFOAM backend is not available: {backend}")

    preferred = status.get("preferredBackend")
    if isinstance(preferred, str) and preferred:
        return preferred
    raise RuntimeError("No local OpenFOAM backend is available. Install OpenFOAM in WSL2, native shell, or set AEROLAB_OPENFOAM_IMAGE for Docker.")


def _run_command(
    case_path: Path,
    backend: str,
    timeout_seconds: int = 3600,
    script_name: str = "Allrun",
) -> list[str]:
    if script_name not in {"Allrun", "Allmesh", "Allsolve"}:
        raise ValueError(f"Unsupported case script: {script_name}")
    if backend == "native":
        return ["bash", "-lc", f"chmod +x {script_name} && ./{script_name}"]
    if backend == "wsl":
        wsl_case_path = _windows_path_to_wsl(case_path)
        stage_id = hashlib.sha256(str(case_path.resolve()).encode("utf-8")).hexdigest()[:16]
        run_timeout = max(1, int(timeout_seconds))
        wsl_script = (
            f"{OPENFOAM_BOOTSTRAP}\n"
            "set -eu\n"
            f"SOURCE_CASE={shlex.quote(wsl_case_path)}\n"
            'STAGE_ROOT="${AEROLAB_WSL_STAGE_ROOT:-$HOME/.cache/aerolab-cfd/runs}"\n'
            'mkdir -p -- "$STAGE_ROOT"\n'
            'STAGE_ROOT=$(cd "$STAGE_ROOT" && pwd -P)\n'
            f'STAGE_CASE="$STAGE_ROOT/case-{stage_id}"\n'
            f'STAGE_MARKER="$STAGE_ROOT/.case-{stage_id}.aerolab-stage"\n'
            'case "$STAGE_CASE" in "$STAGE_ROOT"/case-[0-9a-f]*) ;; '
            '*) printf "Unsafe AeroLab WSL staging path: %s\\n" "$STAGE_CASE" >&2; exit 90 ;; esac\n'
            'if [ -e "$STAGE_CASE" ]; then\n'
            '  if [ ! -f "$STAGE_MARKER" ]; then\n'
            '    printf "Refusing to replace unmarked WSL staging path: %s\\n" "$STAGE_CASE" >&2\n'
            '    exit 90\n'
            '  fi\n'
            '  printf "=== AEROLAB WSL: recovering previous staged case ===\\n"\n'
            '  rm -f -- "$STAGE_CASE/aerolab-run.log" "$STAGE_CASE/aerolab-run.json"\n'
            '  cp -a -- "$STAGE_CASE/." "$SOURCE_CASE/"\n'
            '  rm -rf -- "$STAGE_CASE"\n'
            '  rm -f -- "$STAGE_MARKER"\n'
            'fi\n'
            'mkdir -p -- "$STAGE_CASE"\n'
            ': > "$STAGE_MARKER"\n'
            'printf "=== AEROLAB WSL: staging case on Linux filesystem ===\\n"\n'
            'if ! cp -a -- "$SOURCE_CASE/." "$STAGE_CASE/"; then\n'
            '  rm -rf -- "$STAGE_CASE"\n'
            '  rm -f -- "$STAGE_MARKER"\n'
            '  exit 92\n'
            'fi\n'
            'copy_back() {\n'
            '  run_status=$?\n'
            '  trap - EXIT\n'
            '  printf "=== AEROLAB WSL: copying results back to Windows ===\\n"\n'
            '  rm -f -- "$STAGE_CASE/aerolab-run.log" "$STAGE_CASE/aerolab-run.json"\n'
            '  if cp -a -- "$STAGE_CASE/." "$SOURCE_CASE/"; then\n'
            '    rm -rf -- "$STAGE_CASE"\n'
            '    rm -f -- "$STAGE_MARKER"\n'
            '  else\n'
            '    printf "AeroLab copy-back failed; Linux results remain at %s\\n" "$STAGE_CASE" >&2\n'
            '    run_status=91\n'
            '  fi\n'
            '  exit "$run_status"\n'
            '}\n'
            'trap copy_back EXIT\n'
            'cd "$STAGE_CASE"\n'
            f"sed -i 's/\\r$//' {script_name}\n"
            "find 0 constant system -type f ! -name '*.stl' -exec sed -i 's/\\r$//' {} +\n"
            f"chmod +x {script_name}\n"
            'run_status=0\n'
            f'timeout --foreground --signal=TERM --kill-after=30s {run_timeout}s ./{script_name} '
            '|| run_status=$?\n'
            'exit "$run_status"'
        )
        encoded_script = base64.b64encode(wsl_script.encode("utf-8")).decode("ascii")
        return [
            "wsl",
            "bash",
            "-lc",
            f"printf %s {shlex.quote(encoded_script)} | base64 -d | bash",
        ]
    if backend == "docker":
        image = os.environ.get("AEROLAB_OPENFOAM_IMAGE")
        if not image:
            raise RuntimeError("Set AEROLAB_OPENFOAM_IMAGE to use the Docker backend.")
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{case_path}:/case",
            "-w",
            "/case",
            image,
            "bash",
            "-lc",
            f"chmod +x {script_name} && ./{script_name}",
        ]
    raise RuntimeError(f"Unsupported backend: {backend}")


def _windows_path_to_wsl(path: Path) -> str:
    text = str(path.resolve())
    drive = path.drive.rstrip(":").lower()
    if drive:
        rest = text[len(path.drive) :].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def _run_quick(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 1, "", str(exc))


def _trim(value: str, limit: int = 500) -> str:
    value = value.replace("\x00", "").strip()
    return value[:limit]


def _openfoam_version(value: str) -> str | None:
    match = re.search(r"OPENFOAM_VERSION=([^\s]+)", value)
    return match.group(1) if match else None
