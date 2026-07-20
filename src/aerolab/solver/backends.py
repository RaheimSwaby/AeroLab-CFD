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


OPENFOAM_EXECUTABLES = (
    "foamRun",
    "surfaceFeatures",
    "blockMesh",
    "snappyHexMesh",
    "checkMesh",
    "foamToVTK",
    "potentialFoam",
    "foamPostProcess",
)

OPENFOAM_PARALLEL_EXECUTABLES = (
    "mpirun",
    "decomposePar",
    "reconstructParMesh",
    "reconstructPar",
)


def _resource_probe_script() -> str:
    lines = [
        "set +e\n",
        "if command -v nproc >/dev/null 2>&1; then\n",
        "  AEROLAB_LOGICAL_CPUS=$(nproc 2>/dev/null)\n",
        "else\n",
        "  AEROLAB_LOGICAL_CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')\n",
        "fi\n",
        "printf 'AEROLAB_RESOURCE_LOGICAL_CPUS=%s\\n' \"$AEROLAB_LOGICAL_CPUS\"\n",
        "if [ -r /proc/meminfo ]; then\n",
        "  awk '/^MemAvailable:/ {printf \"AEROLAB_RESOURCE_MEM_AVAILABLE_KIB=%s\\n\", $2}' /proc/meminfo\n",
        "fi\n",
        "for AEROLAB_CPU_MAX in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us; do\n",
        "  if [ -r \"$AEROLAB_CPU_MAX\" ]; then printf 'AEROLAB_RESOURCE_CPU_LIMIT=%s\\n' \"$(tr '\\n' ' ' < \"$AEROLAB_CPU_MAX\")\"; break; fi\n",
        "done\n",
        "if [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then\n",
        "  printf 'AEROLAB_RESOURCE_CPU_PERIOD=%s\\n' \"$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)\"\n",
        "fi\n",
        "for AEROLAB_CPUSET in /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset/cpuset.cpus; do\n",
        "  if [ -r \"$AEROLAB_CPUSET\" ]; then printf 'AEROLAB_RESOURCE_CPUSET=%s\\n' \"$(cat \"$AEROLAB_CPUSET\")\"; break; fi\n",
        "done\n",
        "for AEROLAB_MEMORY_MAX in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do\n",
        "  if [ -r \"$AEROLAB_MEMORY_MAX\" ]; then printf 'AEROLAB_RESOURCE_MEMORY_MAX=%s\\n' \"$(cat \"$AEROLAB_MEMORY_MAX\")\"; break; fi\n",
        "done\n",
        "for AEROLAB_MEMORY_CURRENT in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do\n",
        "  if [ -r \"$AEROLAB_MEMORY_CURRENT\" ]; then printf 'AEROLAB_RESOURCE_MEMORY_CURRENT=%s\\n' \"$(cat \"$AEROLAB_MEMORY_CURRENT\")\"; break; fi\n",
        "done\n",
    ]
    for executable in OPENFOAM_PARALLEL_EXECUTABLES:
        lines.extend(
            (
                f"AEROLAB_TOOL_PATH=$(command -v {shlex.quote(executable)} 2>/dev/null || true)\n",
                f"printf 'AEROLAB_PARALLEL_TOOL_{executable}=%s\\n' \"$AEROLAB_TOOL_PATH\"\n",
            )
        )
    return "".join(lines)


def solver_status(timeout_seconds: int = 40) -> dict[str, object]:
    native_tools = {
        executable: shutil.which(executable)
        for executable in OPENFOAM_EXECUTABLES
    }
    wsl_path = shutil.which("wsl")
    docker_path = shutil.which("docker")

    status: dict[str, object] = {
        "ok": True,
        "preferredBackend": None,
        "backends": {
            "native": {
                "available": all(native_tools.values()),
                "foamRun": native_tools["foamRun"],
                "blockMesh": native_tools["blockMesh"],
                "snappyHexMesh": native_tools["snappyHexMesh"],
                "surfaceFeatures": native_tools["surfaceFeatures"],
                "toolchain": native_tools,
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

    if all(native_tools.values()):
        status["preferredBackend"] = "native"

    if wsl_path:
        wsl_probe = _run_quick(["wsl", "--status"], timeout_seconds)
        wsl_available = False
        wsl_version: str | None = None
        wsl_message = _trim(wsl_probe.stderr or wsl_probe.stdout)
        wsl_infrastructure_error: dict[str, object] | None = None
        if wsl_probe.returncode in {124, 127}:
            wsl_infrastructure_error = {
                "stage": "statusProbe",
                "returncode": wsl_probe.returncode,
                "message": wsl_message or "WSL status probe failed.",
            }
        elif wsl_probe.returncode == 0:
            tool_checks = " && ".join(
                f"command -v {shlex.quote(executable)} >/dev/null 2>&1"
                for executable in OPENFOAM_EXECUTABLES
            )
            wsl_check = _run_quick(
                [
                    "wsl",
                    "bash",
                    "-lc",
                    f"{OPENFOAM_BOOTSTRAP}\n{tool_checks} && "
                    "printf 'OPENFOAM_VERSION=%s' \"${WM_PROJECT_VERSION:-}\"",
                ],
                timeout_seconds,
            )
            wsl_available = wsl_check.returncode == 0
            wsl_message = _trim(wsl_check.stderr or wsl_check.stdout)
            wsl_version = _openfoam_version(wsl_check.stdout)
            if wsl_check.returncode in {124, 127}:
                wsl_infrastructure_error = {
                    "stage": "toolchainProbe",
                    "returncode": wsl_check.returncode,
                    "message": wsl_message or "WSL OpenFOAM toolchain probe failed.",
                }
        status["backends"]["wsl"]["available"] = wsl_available  # type: ignore[index]
        status["backends"]["wsl"]["openfoam"] = wsl_available  # type: ignore[index]
        status["backends"]["wsl"]["message"] = wsl_message  # type: ignore[index]
        status["backends"]["wsl"]["version"] = wsl_version  # type: ignore[index]
        status["backends"]["wsl"]["targetVersion"] = "OpenFOAM Foundation v13"  # type: ignore[index]
        if wsl_infrastructure_error:
            status["backends"]["wsl"]["infrastructureError"] = (  # type: ignore[index]
                wsl_infrastructure_error
            )
        if wsl_available and status["preferredBackend"] is None:
            status["preferredBackend"] = "wsl"

    docker_image = os.environ.get("AEROLAB_OPENFOAM_IMAGE")
    if docker_path and docker_image:
        daemon_check = _run_quick(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            timeout_seconds,
        )
        if daemon_check.returncode != 0:
            message = _trim(daemon_check.stderr or daemon_check.stdout)
            status["backends"]["docker"]["message"] = message  # type: ignore[index]
            status["backends"]["docker"]["infrastructureError"] = {  # type: ignore[index]
                "stage": "daemonProbe",
                "returncode": daemon_check.returncode,
                "message": message or "Docker daemon probe failed.",
            }
        else:
            docker_check = _run_quick(
                ["docker", "image", "inspect", docker_image],
                timeout_seconds,
            )
            docker_message = _trim(docker_check.stderr or docker_check.stdout)
            docker_available = docker_check.returncode == 0
            status["backends"]["docker"]["available"] = docker_available  # type: ignore[index]
            status["backends"]["docker"]["message"] = docker_message  # type: ignore[index]
            if docker_check.returncode in {124, 127}:
                status["backends"]["docker"]["infrastructureError"] = {  # type: ignore[index]
                    "stage": "imageProbe",
                    "returncode": docker_check.returncode,
                    "message": docker_message or "Docker image probe failed.",
                }
            elif docker_check.returncode != 0:
                daemon_recheck = _run_quick(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    timeout_seconds,
                )
                if daemon_recheck.returncode != 0:
                    recheck_message = _trim(
                        daemon_recheck.stderr or daemon_recheck.stdout
                    )
                    status["backends"]["docker"]["infrastructureError"] = {  # type: ignore[index]
                        "stage": "imageProbeDaemonRecheck",
                        "returncode": daemon_recheck.returncode,
                        "message": recheck_message or "Docker daemon recheck failed.",
                    }
            if docker_available and status["preferredBackend"] is None:
                status["preferredBackend"] = "docker"

    return status


def probe_backend_resources(
    backend: str,
    timeout_seconds: int = 40,
    solver_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Measure resources and MPI tools inside the exact execution backend."""
    script = _resource_probe_script()
    if backend == "native":
        command = ["bash", "-lc", script]
    elif backend == "wsl":
        if not shutil.which("wsl"):
            raise RuntimeError("The WSL executable is not available for resource probing.")
        command = ["wsl", "bash", "-lc", f"{OPENFOAM_BOOTSTRAP}\n{script}"]
    elif backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError("The Docker executable is not available for resource probing.")
        image: object = os.environ.get("AEROLAB_OPENFOAM_IMAGE")
        if solver_identity is not None and solver_identity.get("imageId"):
            image = solver_identity["imageId"]
        if not isinstance(image, str) or not image:
            raise RuntimeError("AEROLAB_OPENFOAM_IMAGE is not configured.")
        command = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            script,
        ]
    else:
        raise RuntimeError(f"Unsupported resource-probe backend: {backend}")

    probe = _run_quick(command, timeout_seconds, raise_errors=True)
    if probe.returncode != 0:
        detail = _trim(probe.stderr or probe.stdout)
        raise RuntimeError(
            f"The {backend} resource probe returned {probe.returncode}"
            + (f": {detail}" if detail else "")
        )
    return _parse_resource_probe(probe.stdout, backend)


def _parse_resource_probe(output: str, backend: str) -> dict[str, object]:
    logical_cpus = _positive_int(_probe_marker(output, "AEROLAB_RESOURCE_LOGICAL_CPUS")) or 1
    cpu_candidates = [logical_cpus]
    cpuset = _probe_marker(output, "AEROLAB_RESOURCE_CPUSET")
    cpuset_cpus = _cpuset_count(cpuset)
    if cpuset_cpus:
        cpu_candidates.append(cpuset_cpus)

    cpu_limit = _probe_marker(output, "AEROLAB_RESOURCE_CPU_LIMIT")
    cpu_period = _probe_marker(output, "AEROLAB_RESOURCE_CPU_PERIOD")
    quota_cpus = _cpu_quota_count(cpu_limit, cpu_period)
    if quota_cpus:
        cpu_candidates.append(quota_cpus)
    effective_cpus = max(1, min(cpu_candidates))

    mem_available_kib = _positive_int(
        _probe_marker(output, "AEROLAB_RESOURCE_MEM_AVAILABLE_KIB")
    )
    memory_candidates: list[int] = []
    if mem_available_kib:
        memory_candidates.append(mem_available_kib * 1024)
    memory_max = _finite_cgroup_limit(
        _probe_marker(output, "AEROLAB_RESOURCE_MEMORY_MAX")
    )
    memory_current = _positive_int(
        _probe_marker(output, "AEROLAB_RESOURCE_MEMORY_CURRENT")
    ) or 0
    if memory_max is not None:
        memory_candidates.append(max(0, memory_max - memory_current))
    memory_available = min(memory_candidates) if memory_candidates else None

    parallel_tools = {
        executable: _probe_marker(output, f"AEROLAB_PARALLEL_TOOL_{executable}")
        for executable in OPENFOAM_PARALLEL_EXECUTABLES
    }
    missing_tools = [name for name, path in parallel_tools.items() if not path]
    return {
        "backend": backend,
        "logicalCpus": logical_cpus,
        "cpuset": cpuset,
        "cpusetCpus": cpuset_cpus,
        "quotaCpus": quota_cpus,
        "effectiveCpus": effective_cpus,
        "memoryAvailableBytes": memory_available,
        "memoryLimitBytes": memory_max,
        "memoryCurrentBytes": memory_current if memory_max is not None else None,
        "parallelAvailable": not missing_tools,
        "parallelTools": parallel_tools,
        "missingParallelTools": missing_tools,
    }


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        number = int(value.strip())
    except ValueError:
        return None
    return number if number > 0 else None


def _cpuset_count(value: str | None) -> int | None:
    if not value:
        return None
    cpus: set[int] = set()
    try:
        for token in value.split(","):
            bounds = token.strip().split("-", 1)
            start = int(bounds[0])
            end = int(bounds[-1])
            if start < 0 or end < start:
                return None
            cpus.update(range(start, end + 1))
    except ValueError:
        return None
    return len(cpus) or None


def _cpu_quota_count(limit: str | None, period: str | None) -> int | None:
    if not limit:
        return None
    parts = limit.split()
    if parts[0] == "max":
        return None
    try:
        quota = int(parts[0])
        period_value = int(parts[1]) if len(parts) > 1 else int(period or "0")
    except ValueError:
        return None
    if quota <= 0 or period_value <= 0:
        return None
    return max(1, quota // period_value)


def _finite_cgroup_limit(value: str | None) -> int | None:
    if not value or value.strip() == "max":
        return None
    try:
        limit = int(value.strip())
    except ValueError:
        return None
    if limit <= 0 or limit >= 1 << 60:
        return None
    return limit


def openfoam_identity(backend: str, timeout_seconds: int = 40) -> dict[str, object]:
    """Probe the exact execution environment and return its solver identity."""
    probe_script = _identity_probe_script()
    image: str | None = None
    image_id: str | None = None
    if backend == "native":
        command = ["bash", "-lc", probe_script]
    elif backend == "wsl":
        if not shutil.which("wsl"):
            raise RuntimeError("The WSL executable disappeared before identity probing.")
        command = ["wsl", "bash", "-lc", f"{OPENFOAM_BOOTSTRAP}\n{probe_script}"]
    elif backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError("The Docker executable disappeared before identity probing.")
        image = os.environ.get("AEROLAB_OPENFOAM_IMAGE")
        if not image:
            raise RuntimeError("AEROLAB_OPENFOAM_IMAGE is not configured.")
        image_probe = _run_quick(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            timeout_seconds,
            raise_errors=True,
        )
        if image_probe.returncode != 0:
            detail = _trim(image_probe.stderr or image_probe.stdout)
            raise RuntimeError(
                "Docker image inspection failed"
                + (f": {detail}" if detail else "")
            )
        image_id = _trim(image_probe.stdout)
        if not image_id:
            raise RuntimeError("Docker image inspection returned no image ID.")
        command = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            image_id,
            "-lc",
            probe_script,
        ]
    else:
        raise RuntimeError(f"Unsupported OpenFOAM identity backend: {backend}")

    probe = _run_quick(command, timeout_seconds, raise_errors=True)
    output = f"{probe.stdout}\n{probe.stderr}".strip()
    if probe.returncode != 0:
        raise RuntimeError(
            f"OpenFOAM toolchain identity probe returned {probe.returncode}"
            + (f": {_trim(output)}" if output else "")
        )
    environment_version = _normalized_version_token(
        _probe_marker(output, "OPENFOAM_VERSION")
    )
    banner_version = _openfoam_banner_version(output)
    versions_agree = bool(
        banner_version
        and (environment_version is None or environment_version == banner_version)
    )
    toolchain = {
        executable: {
            "path": _probe_marker(
                output,
                f"OPENFOAM_TOOL_{executable}_PATH",
            ),
            "sha256": _probe_marker(
                output,
                f"OPENFOAM_TOOL_{executable}_SHA256",
            ),
        }
        for executable in OPENFOAM_EXECUTABLES
    }
    foam_run = toolchain["foamRun"]
    version = banner_version if versions_agree else None
    return {
        "backend": backend,
        "version": version,
        "environmentVersion": environment_version,
        "versionsAgree": versions_agree,
        "distribution": _openfoam_distribution(output),
        "executable": foam_run["path"],
        "executableSha256": foam_run["sha256"],
        "toolchain": toolchain,
        "probeReturncode": probe.returncode,
        "probeOutput": _trim(output, limit=4000),
        "probeOutputSha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "image": image,
        "imageId": image_id,
    }


def openfoam_version(backend: str, timeout_seconds: int = 40) -> str | None:
    """Return the major version verified in the exact backend environment."""
    version = openfoam_identity(backend, timeout_seconds).get("version")
    return str(version) if version is not None else None


def _identity_probe_script() -> str:
    script = [
        "set -eu\n",
        "aerolab_sha256() {\n",
        "  if command -v sha256sum >/dev/null 2>&1; then\n",
        "    sha256sum \"$1\" | awk '{print $1}'\n",
        "  elif command -v shasum >/dev/null 2>&1; then\n",
        "    shasum -a 256 \"$1\" | awk '{print $1}'\n",
        "  else\n",
        "    printf 'No SHA-256 utility is available for OpenFOAM attestation.\\n' >&2\n",
        "    return 87\n",
        "  fi\n",
        "}\n",
    ]
    for executable in OPENFOAM_EXECUTABLES:
        script.extend(
            (
                f"TOOL_PATH=$(command -v {shlex.quote(executable)})\n",
                "TOOL_SHA256=$(aerolab_sha256 \"$TOOL_PATH\")\n",
                f"printf 'OPENFOAM_TOOL_{executable}_PATH=%s\\n' \"$TOOL_PATH\"\n",
                f"printf 'OPENFOAM_TOOL_{executable}_SHA256=%s\\n' \"$TOOL_SHA256\"\n",
            )
        )
    script.extend(
        (
            "printf 'OPENFOAM_VERSION=%s\\n' \"${WM_PROJECT_VERSION:-}\"\n",
            "foamRun -help 2>&1",
        )
    )
    return "".join(script)


def _probe_marker(value: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\n){re.escape(name)}=([^\r\n]+)", value)
    return match.group(1).strip() if match else None


def _normalized_version_token(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"v?(\d+)(?:\.\d+)*", value.strip())
    return match.group(1) if match else None


def _openfoam_banner_version(value: str) -> str | None:
    patterns = (
        r"(?im)^\s*Using:\s*OpenFOAM-(\d+)(?:\.\d+)*\b",
        r"(?i)\bOpenFOAM\s+Foundation[-\s:]+v?(\d+)\b",
        r"(?i)\bOpenFOAM[-\s:]+v?(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def _openfoam_distribution(value: str) -> str | None:
    if re.search(r"(?i)\bOpenFOAM\s+Foundation\b", value) or re.search(
        r"(?i)https?://(?:www\.)?openfoam\.org\b",
        value,
    ):
        return "foundation"
    if re.search(r"(?i)https?://(?:www\.)?openfoam\.com\b", value):
        return "openfoam-com"
    return None


def _normalized_openfoam_version(value: str) -> str | None:
    marker_version = _normalized_version_token(_openfoam_version(value))
    if marker_version:
        return marker_version
    banner_version = _openfoam_banner_version(value)
    if banner_version:
        return banner_version
    match = re.search(r"(?i)\bVersion\s*[:=]\s*v?(\d+)\b", value)
    return match.group(1) if match else None


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


def _execution_identity_guard(
    solver_identity: dict[str, object] | None,
    backend: str,
) -> str:
    if solver_identity is None:
        return ""
    if solver_identity.get("backend") != backend:
        raise RuntimeError("The attested OpenFOAM backend does not match the execution backend.")
    toolchain = solver_identity.get("toolchain")
    if not isinstance(toolchain, dict):
        raise RuntimeError("The attested OpenFOAM toolchain is missing.")

    script = [
        "set -eu\n",
        "aerolab_sha256() {\n",
        "  if command -v sha256sum >/dev/null 2>&1; then\n",
        "    sha256sum \"$1\" | awk '{print $1}'\n",
        "  elif command -v shasum >/dev/null 2>&1; then\n",
        "    shasum -a 256 \"$1\" | awk '{print $1}'\n",
        "  else\n",
        "    printf 'No SHA-256 utility is available for OpenFOAM execution verification.\\n' >&2\n",
        "    return 97\n",
        "  fi\n",
        "}\n",
    ]
    for index, executable in enumerate(OPENFOAM_EXECUTABLES):
        entry = toolchain.get(executable)
        if not isinstance(entry, dict):
            raise RuntimeError(f"The attested {executable} identity is missing.")
        expected_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_path, str) or not expected_path:
            raise RuntimeError(f"The attested {executable} path is missing.")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}",
            expected_hash,
        ):
            raise RuntimeError(f"The attested {executable} SHA-256 is missing or invalid.")
        script.extend(
            (
                f"EXPECTED_TOOL_PATH_{index}={shlex.quote(expected_path)}\n",
                f"EXPECTED_TOOL_SHA256_{index}={shlex.quote(expected_hash.lower())}\n",
                f"if ! ACTUAL_TOOL_PATH_{index}=$(command -v {shlex.quote(executable)}); then\n",
                f"  printf 'Attested {executable} disappeared before execution.\\n' >&2\n",
                "  exit 96\n",
                "fi\n",
                f"if [ \"$ACTUAL_TOOL_PATH_{index}\" != \"$EXPECTED_TOOL_PATH_{index}\" ]; then\n",
                f"  printf 'Attested {executable} path changed before execution.\\n' >&2\n",
                "  exit 96\n",
                "fi\n",
                f"if ! ACTUAL_TOOL_SHA256_{index}=$(aerolab_sha256 \"$ACTUAL_TOOL_PATH_{index}\"); then\n",
                f"  printf 'Could not hash attested {executable} before execution.\\n' >&2\n",
                "  exit 97\n",
                "fi\n",
                f"if [ \"$ACTUAL_TOOL_SHA256_{index}\" != \"$EXPECTED_TOOL_SHA256_{index}\" ]; then\n",
                f"  printf 'Attested {executable} content changed before execution.\\n' >&2\n",
                "  exit 97\n",
                "fi\n",
            )
        )
    foam_run = toolchain.get("foamRun")
    if not isinstance(foam_run, dict) or (
        solver_identity.get("executable") != foam_run.get("path")
        or solver_identity.get("executableSha256") != foam_run.get("sha256")
    ):
        raise RuntimeError("The foamRun alias does not match the attested toolchain.")
    return "".join(script)


def _run_command(
    case_path: Path,
    backend: str,
    timeout_seconds: int = 3600,
    script_name: str = "Allrun",
    execution_id: str | None = None,
    solver_identity: dict[str, object] | None = None,
    processes: int = 1,
    file_handler: str = "auto",
    resume: bool = False,
    feature_cache_key: str = "disabled",
    block_cache_key: str = "disabled",
) -> list[str]:
    if script_name not in {"Allrun", "Allmesh", "Allsolve"}:
        raise ValueError(f"Unsupported case script: {script_name}")
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise ValueError("OpenFOAM processes must be a positive integer.")
    if file_handler not in {"auto", "uncollated", "collated", "masterUncollated"}:
        raise ValueError(f"Unsupported OpenFOAM file handler: {file_handler}")
    if not isinstance(resume, bool):
        raise ValueError("Resume must be a boolean.")
    for cache_key in (feature_cache_key, block_cache_key):
        if cache_key != "disabled" and not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise ValueError("Stage-cache keys must be lowercase SHA-256 values.")
    identity_guard = _execution_identity_guard(solver_identity, backend)
    process_export = f"export AEROLAB_PROCESSES={processes}\n"
    file_handler_export = f"export AEROLAB_FILE_HANDLER={shlex.quote(file_handler)}\n"
    cache_export = (
        f"export AEROLAB_RESUME={1 if resume else 0}\n"
        f"export AEROLAB_FEATURE_CACHE_KEY={shlex.quote(feature_cache_key)}\n"
        f"export AEROLAB_BLOCK_CACHE_KEY={shlex.quote(block_cache_key)}\n"
    )
    if backend == "native":
        return [
            "bash",
            "-lc",
            f"{identity_guard}{process_export}{file_handler_export}{cache_export}chmod +x {script_name} && ./{script_name}",
        ]
    if backend == "wsl":
        wsl_case_path = _windows_path_to_wsl(case_path)
        stage_id = hashlib.sha256(str(case_path.resolve()).encode("utf-8")).hexdigest()[:16]
        run_timeout = max(1, int(timeout_seconds))
        wsl_script = (
            f"{OPENFOAM_BOOTSTRAP}\n"
            "set -eu\n"
            f"{identity_guard}"
            f"{process_export}"
            f"{file_handler_export}"
            f"{cache_export}"
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
        execution_image = image
        if solver_identity is not None:
            image_id = solver_identity.get("imageId")
            if not isinstance(image_id, str) or not image_id:
                raise RuntimeError("The attested Docker image ID is missing.")
            execution_image = image_id
        container_name = _docker_container_name(case_path, execution_id)
        docker_script = (
            "set -eu\n"
            f"{identity_guard}"
            'SOURCE_CASE=/source\n'
            'STAGE_CASE=/work\n'
            'printf "=== AEROLAB DOCKER: staging case on Linux volume ===\\n"\n'
            'cp -a -- "$SOURCE_CASE/." "$STAGE_CASE/"\n'
            'copy_back() {\n'
            '  run_status=$?\n'
            '  trap - EXIT\n'
            '  printf "=== AEROLAB DOCKER: copying results back to host ===\\n"\n'
            '  rm -f -- "$STAGE_CASE/aerolab-run.log" "$STAGE_CASE/aerolab-run.json"\n'
            '  if ! cp -a -- "$STAGE_CASE/." "$SOURCE_CASE/"; then\n'
            '    printf "AeroLab Docker copy-back failed.\\n" >&2\n'
            '    run_status=91\n'
            '  fi\n'
            '  exit "$run_status"\n'
            '}\n'
            'trap copy_back EXIT\n'
            'cd "$STAGE_CASE"\n'
            f"chmod +x {script_name} && ./{script_name}"
        )
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--stop-timeout",
            "30",
            "--env",
            f"AEROLAB_PROCESSES={processes}",
            "--env",
            f"AEROLAB_FILE_HANDLER={file_handler}",
            "--env",
            f"AEROLAB_RESUME={1 if resume else 0}",
            "--env",
            f"AEROLAB_FEATURE_CACHE_KEY={feature_cache_key}",
            "--env",
            f"AEROLAB_BLOCK_CACHE_KEY={block_cache_key}",
            "-v",
            f"{case_path}:/source",
            "--mount",
            "type=volume,target=/work",
            "-w",
            "/work",
            "--entrypoint",
            "bash",
            execution_image,
            "-lc",
            docker_script,
        ]
    raise RuntimeError(f"Unsupported backend: {backend}")


def _docker_container_name(case_path: Path, execution_id: str | None) -> str:
    identity = f"{case_path.resolve()}\0{execution_id or 'manual'}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"aerolab-{digest}"


def _backend_cleanup_command(
    case_path: Path,
    backend: str,
    execution_id: str,
) -> list[str] | None:
    if backend == "docker":
        return [
            "docker",
            "rm",
            "--force",
            "--volumes",
            _docker_container_name(case_path, execution_id),
        ]
    return None


def _backend_cleanup_verification_command(
    case_path: Path,
    backend: str,
    execution_id: str,
) -> list[str] | None:
    if backend == "docker":
        container_name = _docker_container_name(case_path, execution_id)
        return [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.ID}}",
        ]
    return None


def _windows_path_to_wsl(path: Path) -> str:
    text = str(path.resolve())
    drive = path.drive.rstrip(":").lower()
    if drive:
        rest = text[len(path.drive) :].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def _run_quick(
    command: list[str],
    timeout_seconds: int,
    *,
    raise_errors: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        if raise_errors:
            raise
        return subprocess.CompletedProcess(command, 124, "", str(exc))
    except OSError as exc:
        if raise_errors:
            raise
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _trim(value: str, limit: int = 500) -> str:
    value = value.replace("\x00", "").strip()
    return value[:limit]


def _openfoam_version(value: str) -> str | None:
    match = re.search(r"OPENFOAM_VERSION=([^\s]+)", value)
    return match.group(1) if match else None
