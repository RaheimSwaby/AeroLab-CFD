from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from ..openfoam import configure_decomposition, ensure_case_postprocessing
from .analysis import _latest_body_surface_vtk, assess_meshed_surface_fidelity, case_report
from .backends import (
    _backend_cleanup_command,
    _backend_cleanup_verification_command,
    _run_command,
    _select_backend,
    probe_backend_resources,
    solver_status,
)
from .util import _read_json_object

MESH_INPUT_FILES = (
    "constant/geometry/body.stl",
    "system/blockMeshDict",
    "system/snappyHexMeshDict",
    "system/surfaceFeaturesDict",
)
RUN_MODES = {"full", "mesh"}
AUTO_MAX_PROCESSES = 8
AUTO_BYTES_PER_PROCESS = 2 * 1024**3
MANUAL_MIN_BYTES_PER_PROCESS = 1024**3
AUTO_CELLS_PER_PROCESS = 250_000
QUALITY_ESTIMATED_BYTES_PER_CELL = 2048
QUALITY_FIXED_MEMORY_BYTES = 2 * 1024**3


def normalize_process_request(value: object) -> str | int:
    """Normalize an OpenFOAM process request to ``auto`` or a positive integer."""
    if isinstance(value, bool):
        raise ValueError("Processes must be 'auto' or a positive integer.")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("Processes must be at least 1.")
        return value
    text = str(value if value is not None else "auto").strip().lower()
    if text == "auto":
        return "auto"
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError("Processes must be 'auto' or a positive integer.") from exc
    if number < 1:
        raise ValueError("Processes must be at least 1.")
    return number


def normalize_file_handler(value: object) -> str:
    handler = str(value if value is not None else "auto").strip()
    aliases = {
        "auto": "auto",
        "uncollated": "uncollated",
        "collated": "collated",
        "masteruncollated": "masterUncollated",
    }
    normalized = aliases.get(handler.lower())
    if normalized is None:
        raise ValueError(
            "File handler must be auto, uncollated, collated, or masterUncollated."
        )
    return normalized


def _case_cell_budget(case_path: Path) -> int | None:
    payload = _read_json_object(case_path / "case.json")
    mesh = payload.get("mesh_resolution")
    if not isinstance(mesh, dict):
        return None
    value = mesh.get("configured_max_global_cells")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = int(value)
    return number if number > 0 else None


def _stage_fingerprint(case_path: Path, relative_paths: list[str]) -> str | None:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        path = case_path / relative_path
        if not path.is_file():
            return None
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _stage_cache_plan(case_path: Path) -> dict[str, object]:
    geometry_files = [
        path.relative_to(case_path).as_posix()
        for path in sorted((case_path / "constant" / "geometry").glob("*.stl"))
    ]
    feature_key = _stage_fingerprint(
        case_path,
        [*geometry_files, "system/surfaceFeaturesDict"],
    )
    block_key = _stage_fingerprint(case_path, ["system/blockMeshDict"])
    cache_root = case_path / ".aerolab-cache"
    feature_path = cache_root / "surfaceFeatures" / str(feature_key)
    block_path = cache_root / "blockMesh" / str(block_key)
    return {
        "featureKey": feature_key or "disabled",
        "blockMeshKey": block_key or "disabled",
        "featureHit": bool(feature_key and (feature_path / ".ready").is_file()),
        "blockMeshHit": bool(
            block_key
            and (block_path / ".ready").is_file()
            and (block_path / "polyMesh" / "points").is_file()
        ),
    }


def _quality_recommendation(
    case_path: Path,
    resources: dict[str, object],
) -> dict[str, object]:
    payload = _read_json_object(case_path / "case.json")
    quality = payload.get("cfd_quality")
    quality_name = (
        str(quality.get("name") or "unknown")
        if isinstance(quality, dict)
        else "unknown"
    )
    cell_budget = _case_cell_budget(case_path)
    available_value = resources.get("memoryAvailableBytes")
    available = (
        int(available_value)
        if isinstance(available_value, int | float) and available_value >= 0
        else None
    )
    if cell_budget is None or available is None:
        return {
            "status": "unknown",
            "quality": quality_name,
            "detail": "A memory recommendation requires both a case cell budget and backend memory data.",
        }
    estimated = QUALITY_FIXED_MEMORY_BYTES + cell_budget * QUALITY_ESTIMATED_BYTES_PER_CELL
    headroom = available / estimated if estimated > 0 else 0.0
    if headroom >= 1.5:
        status = "comfortable"
        detail = f"The {quality_name} case has conservative memory headroom on this backend."
    elif headroom >= 1.0:
        status = "tight"
        detail = (
            f"The {quality_name} case is close to the conservative memory estimate; use Draft for iteration "
            "or increase Docker/WSL memory before a long run."
        )
    else:
        status = "insufficient"
        detail = (
            f"The {quality_name} case exceeds the conservative memory estimate; regenerate at Draft quality "
            "or increase Docker/WSL memory."
        )
    return {
        "status": status,
        "quality": quality_name,
        "configuredMaxCells": cell_budget,
        "estimatedMemoryBytes": estimated,
        "availableMemoryBytes": available,
        "headroomRatio": round(headroom, 3),
        "heuristic": "2 GiB fixed plus 2 KiB per configured maximum cell",
        "detail": detail,
    }


def _resolve_processes(
    case_path: Path,
    requested: str | int,
    backend: str,
    *,
    parallel_script: bool,
    solver_identity: dict[str, object] | None,
) -> tuple[int, dict[str, object]]:
    selection: dict[str, object] = {
        "requested": requested,
        "resolved": 1,
        "selectionReason": "Explicit serial execution.",
        "estimatedCellBudget": _case_cell_budget(case_path),
    }
    if requested == 1:
        return 1, selection
    if not parallel_script:
        if requested == "auto":
            selection["selectionReason"] = (
                "This legacy case predates parallel-capable scripts; using serial execution."
            )
            return 1, selection
        raise ValueError(
            "This case predates parallel-capable scripts. Create a new AeroLab case or use --processes 1."
        )

    try:
        resources = probe_backend_resources(
            backend,
            solver_identity=solver_identity,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if requested == "auto":
            selection["selectionReason"] = (
                f"Backend resource probing failed; using serial execution: {exc}"
            )
            return 1, selection
        raise RuntimeError(f"Could not validate {requested} OpenFOAM processes: {exc}") from exc

    selection["detectedResources"] = resources
    selection["qualityRecommendation"] = _quality_recommendation(case_path, resources)
    if not resources.get("parallelAvailable"):
        missing = ", ".join(str(item) for item in resources.get("missingParallelTools", []))
        if requested == "auto":
            selection["selectionReason"] = (
                "MPI tools are incomplete; using serial execution"
                + (f" (missing: {missing})." if missing else ".")
            )
            return 1, selection
        raise RuntimeError(
            "Parallel OpenFOAM execution is unavailable"
            + (f"; missing: {missing}." if missing else ".")
        )

    effective_cpus = max(1, int(resources.get("effectiveCpus") or 1))
    memory_available_value = resources.get("memoryAvailableBytes")
    memory_available = (
        int(memory_available_value)
        if isinstance(memory_available_value, int | float) and memory_available_value >= 0
        else None
    )
    cell_budget_value = selection["estimatedCellBudget"]
    cell_budget = int(cell_budget_value) if isinstance(cell_budget_value, int) else None

    if isinstance(requested, int):
        if requested > effective_cpus:
            raise ValueError(
                f"Requested {requested} processes, but the {backend} backend exposes {effective_cpus} CPUs."
            )
        if memory_available is not None and memory_available < requested * MANUAL_MIN_BYTES_PER_PROCESS:
            available_gib = memory_available / 1024**3
            raise ValueError(
                f"Requested {requested} processes, but only {available_gib:.1f} GiB is available in the {backend} backend."
            )
        selection.update(
            {
                "resolved": requested,
                "selectionReason": "Validated manual process count against backend CPU and memory limits.",
            }
        )
        return requested, selection

    cpu_cap = max(1, effective_cpus - 1)
    caps = {"cpu": cpu_cap, "safety": AUTO_MAX_PROCESSES}
    if memory_available is not None:
        reserved_memory = max(1024**3, memory_available // 4)
        usable_memory = max(0, memory_available - reserved_memory)
        caps["memory"] = max(1, usable_memory // AUTO_BYTES_PER_PROCESS)
    if cell_budget is not None:
        caps["cellBudget"] = max(1, cell_budget // AUTO_CELLS_PER_PROCESS)
    resolved = max(1, min(caps.values()))
    selection.update(
        {
            "resolved": resolved,
            "autoCaps": caps,
            "selectionReason": (
                "Selected a hardware- and case-aware MPI process count."
                if resolved > 1
                else "The available CPU, memory, or cell budget does not justify MPI; using serial execution."
            ),
        }
    )
    return resolved, selection


@dataclass(frozen=True)
class SolverRunResult:
    backend: str
    returncode: int
    log_path: Path
    started_at: str
    finished_at: str
    report: dict[str, object]
    run_mode: str = "full"
    reused_mesh: bool = False
    requested_processes: str | int = 1
    processes: int = 1
    file_handler: str = "auto"
    resumed: bool = False
    resume_from_time: float | None = None
    solver_input_fingerprint: str | None = None
    convergence_policy: dict[str, object] = field(default_factory=dict)
    process_selection: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def trusted(self) -> bool:
        assessment_name = "meshAssessment" if self.run_mode == "mesh" else "qualityAssessment"
        assessment = self.report.get(assessment_name)
        return bool(isinstance(assessment, dict) and assessment.get("trusted"))

    @property
    def numerically_qualified(self) -> bool:
        return self.trusted

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "trusted": self.trusted,
            "numericallyQualified": self.numerically_qualified,
            "mode": self.run_mode,
            "reusedMesh": self.reused_mesh,
            "backend": self.backend,
            "requestedProcesses": self.requested_processes,
            "processes": self.processes,
            "fileHandler": self.file_handler,
            "resumed": self.resumed,
            "resumeFromTime": self.resume_from_time,
            "solverInputFingerprint": self.solver_input_fingerprint,
            "convergencePolicy": self.convergence_policy,
            "processSelection": self.process_selection,
            "returncode": self.returncode,
            "logPath": str(self.log_path),
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "report": self.report,
        }


def _run_record(result: SolverRunResult) -> dict[str, object]:
    return {
        "status": "complete" if result.ok else "failed",
        "ok": result.ok,
        "trusted": result.trusted,
        "numericallyQualified": result.numerically_qualified,
        "mode": result.run_mode,
        "reusedMesh": result.reused_mesh,
        "backend": result.backend,
        "requestedProcesses": result.requested_processes,
        "processes": result.processes,
        "fileHandler": result.file_handler,
        "resumed": result.resumed,
        "resumeFromTime": result.resume_from_time,
        "solverInputFingerprint": result.solver_input_fingerprint,
        "convergencePolicy": result.convergence_policy,
        "processSelection": result.process_selection,
        "returncode": result.returncode,
        "logPath": str(result.log_path),
        "startedAt": result.started_at,
        "finishedAt": result.finished_at,
        "qualityAssessment": result.report.get("qualityAssessment"),
        "meshAssessment": result.report.get("meshAssessment"),
    }


def run_case(
    case_path: Path,
    backend: str = "auto",
    timeout_seconds: int = 3600,
    run_mode: str = "full",
    reuse_mesh: bool = True,
    solver_identity: dict[str, object] | None = None,
    processes: str | int = 1,
    file_handler: str = "auto",
    resume: bool = False,
) -> SolverRunResult:
    case_path = case_path.resolve()
    requested_processes = normalize_process_request(processes)
    selected_file_handler = normalize_file_handler(file_handler)
    if not isinstance(resume, bool):
        raise ValueError("Resume must be a boolean.")
    run_mode = str(run_mode or "full").lower()
    if run_mode not in RUN_MODES:
        raise ValueError(f"Unsupported run mode: {run_mode}")
    if resume and run_mode != "full":
        raise ValueError("Resume is only available for full solver runs.")
    if resume and not reuse_mesh:
        raise ValueError("Resume requires mesh reuse; remove the no-reuse-mesh option.")
    if not case_path.exists():
        raise FileNotFoundError(case_path)
    if not (case_path / "Allrun").exists():
        raise FileNotFoundError(case_path / "Allrun")

    ensure_case_postprocessing(case_path)
    solver_input_fingerprint = _solver_input_fingerprint(case_path)
    convergence_policy = _convergence_policy(case_path)
    resume_state = (
        _resume_compatibility(case_path, solver_input_fingerprint)
        if resume
        else None
    )
    resume_from_time = (
        float(resume_state["latestTime"])
        if isinstance(resume_state, dict)
        else None
    )
    status = solver_status()
    selected = _select_backend(status, backend)
    reused_mesh = bool(
        resume
        or (
            run_mode == "full"
            and reuse_mesh
            and (case_path / "Allsolve").is_file()
            and _mesh_record_reusable(case_path)
        )
    )
    script_name = "Allmesh" if run_mode == "mesh" else "Allsolve" if reused_mesh else "Allrun"
    if not (case_path / script_name).is_file():
        if run_mode == "mesh":
            raise ValueError("This older case has no Allmesh script; create a new case before validating its mesh.")
        raise FileNotFoundError(case_path / script_name)
    parallel_script = "AEROLAB_PROCESSES" in (case_path / script_name).read_text(
        encoding="utf-8",
        errors="ignore",
    )
    resolved_processes, process_selection = _resolve_processes(
        case_path,
        requested_processes,
        selected,
        parallel_script=parallel_script,
        solver_identity=solver_identity,
    )
    stage_cache = _stage_cache_plan(case_path)
    process_selection["stageCache"] = stage_cache
    configure_decomposition(case_path, resolved_processes)
    started_at = datetime.now(timezone.utc).isoformat()
    log_path = case_path / "aerolab-run.log"
    run_path = case_path / "aerolab-run.json"

    _clear_previous_solver_outputs(
        case_path,
        preserve_mesh=reused_mesh,
        preserve_solver_state=resume,
    )
    execution_id = uuid.uuid4().hex
    command = _run_command(
        case_path,
        selected,
        timeout_seconds=timeout_seconds,
        script_name=script_name,
        execution_id=execution_id,
        solver_identity=solver_identity,
        processes=resolved_processes,
        file_handler=selected_file_handler,
        resume=resume,
        feature_cache_key=str(stage_cache["featureKey"]),
        block_cache_key=str(stage_cache["blockMeshKey"]),
    )
    process_timeout = timeout_seconds + 900 if selected == "wsl" else timeout_seconds
    _update_case_status(case_path, "mesh_running" if run_mode == "mesh" else "solver_running")
    run_path.write_text(
        json.dumps(
            {
                "status": "running",
                "ok": None,
                "trusted": False,
                "numericallyQualified": False,
                "mode": run_mode,
                "reusedMesh": reused_mesh,
                "backend": selected,
                "requestedProcesses": requested_processes,
                "processes": resolved_processes,
                "fileHandler": selected_file_handler,
                "resumed": resume,
                "resumeFromTime": resume_from_time,
                "solverInputFingerprint": solver_input_fingerprint,
                "convergencePolicy": convergence_policy,
                "processSelection": process_selection,
                "returncode": None,
                "logPath": str(log_path),
                "startedAt": started_at,
                "finishedAt": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"AeroLab backend: {selected}\n")
        log_file.write(
            f"AeroLab processes: requested={requested_processes}, resolved={resolved_processes}\n"
        )
        log_file.write(f"AeroLab file handler: {selected_file_handler}\n")
        log_file.write(
            f"AeroLab resume: {'from time ' + format(resume_from_time, '.9g') if resume_from_time is not None else 'no'}\n"
        )
        log_file.write(
            f"AeroLab convergence policy: {convergence_policy.get('controller', 'unknown')}\n"
        )
        log_file.flush()
        returncode, outer_timeout = _run_solver_process(
            command,
            case_path=case_path,
            backend=selected,
            execution_id=execution_id,
            process_timeout=process_timeout,
            log_file=log_file,
        )
        if outer_timeout:
            log_file.write(
                f"\nAeroLab stopped the complete solver process tree after "
                f"its {process_timeout}-second outer limit.\n"
            )
        elif returncode == 124 and selected == "wsl":
            log_file.write(
                f"\nAeroLab stopped OpenFOAM after {timeout_seconds} seconds; "
                "staged partial results were copied back.\n"
            )
    finished_at = datetime.now(timezone.utc).isoformat()
    mesh_fidelity_path = case_path / "mesh-surface-fidelity.json"
    mesh_outputs_present = _mesh_outputs_present(case_path)
    if mesh_outputs_present:
        try:
            mesh_fidelity = assess_meshed_surface_fidelity(case_path)
        except Exception as exc:  # The failed audit must remain visible without hiding solver output.
            mesh_fidelity = {
                "status": "error",
                "verified": False,
                "detail": f"Meshed body fidelity audit failed: {exc}",
            }
        mesh_fidelity_path.write_text(json.dumps(mesh_fidelity, indent=2) + "\n", encoding="utf-8")
    mesh_returncode = 0 if mesh_outputs_present else returncode if run_mode == "mesh" else None
    report = case_report(
        case_path,
        solver_returncode=returncode if run_mode == "full" else None,
        mesh_returncode=mesh_returncode,
    )

    result = SolverRunResult(
        backend=selected,
        returncode=returncode,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        report=report,
        run_mode=run_mode,
        reused_mesh=reused_mesh,
        requested_processes=requested_processes,
        processes=resolved_processes,
        file_handler=selected_file_handler,
        resumed=resume,
        resume_from_time=resume_from_time,
        solver_input_fingerprint=solver_input_fingerprint,
        convergence_policy=convergence_policy,
        process_selection=process_selection,
    )
    run_path.write_text(json.dumps(_run_record(result), indent=2) + "\n", encoding="utf-8")
    if mesh_returncode is not None and not reused_mesh:
        _write_mesh_record(
            case_path,
            backend=selected,
            returncode=mesh_returncode,
            started_at=started_at,
            finished_at=finished_at,
            report=report,
            log_path=log_path,
            preserve_log=run_mode == "mesh",
            processes=resolved_processes,
            file_handler=selected_file_handler,
            process_selection=process_selection,
        )
    if run_mode == "mesh":
        if result.trusted:
            case_status = "mesh_validated"
        elif result.ok:
            case_status = "mesh_unverified"
        else:
            case_status = "mesh_failed"
    elif result.trusted:
        case_status = "solver_verified"
    elif result.ok:
        case_status = "solver_unverified"
    else:
        case_status = "solver_failed"
    _update_case_status(case_path, case_status)
    final_report = case_report(
        case_path,
        solver_returncode=returncode if run_mode == "full" else None,
        mesh_returncode=mesh_returncode,
    )
    return SolverRunResult(
        backend=selected,
        returncode=returncode,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        report=final_report,
        run_mode=run_mode,
        reused_mesh=reused_mesh,
        requested_processes=requested_processes,
        processes=resolved_processes,
        file_handler=selected_file_handler,
        resumed=resume,
        resume_from_time=resume_from_time,
        solver_input_fingerprint=solver_input_fingerprint,
        convergence_policy=convergence_policy,
        process_selection=process_selection,
    )


def _run_solver_process(
    command: list[str],
    *,
    case_path: Path,
    backend: str,
    execution_id: str,
    process_timeout: int,
    log_file: TextIO,
    termination_grace_seconds: float = 30.0,
) -> tuple[int, bool]:
    process_options: dict[str, object] = {}
    if os.name == "posix":
        process_options["start_new_session"] = True
    elif os.name == "nt":
        process_options["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    process = subprocess.Popen(
        command,
        cwd=str(case_path),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        **process_options,
    )
    try:
        return process.wait(timeout=process_timeout), False
    except subprocess.TimeoutExpired:
        cleanup_errors: list[str] = []
        try:
            _terminate_process_tree(
                process,
                grace_seconds=termination_grace_seconds,
            )
        except Exception as exc:
            cleanup_errors.append(f"process-tree termination failed: {exc}")
        try:
            _confirm_backend_cleanup(case_path, backend, execution_id)
        except Exception as exc:
            cleanup_errors.append(f"backend cleanup failed: {exc}")
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            log_file.write(f"\nAeroLab could not confirm timeout cleanup: {detail}\n")
            raise RuntimeError(
                f"Solver timeout cleanup could not be confirmed: {detail}"
            )
        return 124, True


def _confirm_backend_cleanup(
    case_path: Path,
    backend: str,
    execution_id: str,
) -> None:
    cleanup_command = _backend_cleanup_command(case_path, backend, execution_id)
    verification_command = _backend_cleanup_verification_command(
        case_path,
        backend,
        execution_id,
    )
    if cleanup_command is None and verification_command is None:
        return
    if cleanup_command is None or verification_command is None:
        raise RuntimeError("Backend cleanup commands are incomplete.")

    cleanup_detail = ""
    try:
        cleanup = subprocess.run(
            cleanup_command,
            capture_output=True,
            text=True,
            timeout=45,
        )
        if cleanup.returncode != 0:
            cleanup_detail = (cleanup.stderr or cleanup.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        cleanup_detail = str(exc)

    try:
        verification = subprocess.run(
            verification_command,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"container-absence verification failed: {exc}") from exc
    if verification.returncode != 0:
        detail = (verification.stderr or verification.stdout).strip()
        raise RuntimeError(
            "container-absence verification failed"
            + (f": {detail}" if detail else "")
            + (f"; cleanup reported: {cleanup_detail}" if cleanup_detail else "")
        )
    remaining_container_ids = verification.stdout.strip()
    if remaining_container_ids:
        raise RuntimeError(
            f"container still exists after forced removal: {remaining_container_ids}"
            + (f"; cleanup reported: {cleanup_detail}" if cleanup_detail else "")
        )


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = 30.0,
) -> None:
    if os.name == "posix":
        process_group_id = process.pid
        if process_group_id == os.getpgrp():
            raise RuntimeError("Refusing to signal AeroLab's own process group.")
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            process.poll()
            return
        if _wait_for_process_group_exit(process, process_group_id, grace_seconds):
            return
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            process.poll()
            return
        if not _wait_for_process_group_exit(process, process_group_id, 5.0):
            raise RuntimeError(
                f"process group {process_group_id} still exists after SIGKILL"
            )
        return

    if os.name == "nt":
        try:
            terminated = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=max(5.0, grace_seconds),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Windows process-tree termination failed: {exc}") from exc
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("Windows process tree did not exit after taskkill.") from exc
        if terminated.returncode != 0:
            detail = (terminated.stderr or terminated.stdout).strip()
            raise RuntimeError(
                "Windows taskkill could not confirm recursive termination"
                + (f": {detail}" if detail else "")
            )
        return

    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_for_process_group_exit(
    process: subprocess.Popen[str],
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clear_previous_solver_outputs(
    case_path: Path,
    preserve_mesh: bool = False,
    preserve_solver_state: bool = False,
) -> None:
    """Remove generated products unless an explicitly validated resume needs them."""
    if preserve_solver_state and not preserve_mesh:
        raise ValueError("Preserving solver state also requires preserving the mesh.")
    case_path = case_path.resolve()
    directory_targets: list[Path] = []
    post_processing = case_path / "postProcessing"
    if not preserve_solver_state:
        if preserve_mesh and post_processing.is_dir():
            directory_targets.extend(
                child for child in post_processing.iterdir() if child.name != "meshSurface"
            )
        else:
            directory_targets.append(post_processing)
    if not preserve_mesh:
        directory_targets.append(case_path / "constant" / "polyMesh")
    for child in case_path.iterdir():
        if not child.is_dir() or child.name == "0":
            continue
        processor_suffix = child.name.removeprefix("processor")
        if child.name.startswith("processor") and processor_suffix.isdigit():
            directory_targets.append(child)
            continue
        if preserve_solver_state:
            continue
        try:
            numeric_time = float(child.name)
        except ValueError:
            continue
        if math.isfinite(numeric_time):
            directory_targets.append(child)

    for target in directory_targets:
        resolved = target.resolve()
        if resolved != case_path and case_path in resolved.parents and resolved.exists():
            shutil.rmtree(resolved)

    if not preserve_mesh:
        for filename in (
            "mesh-surface-fidelity.json",
            "aerolab-mesh.json",
            "aerolab-mesh.log",
        ):
            target = case_path / filename
            if target.exists():
                target.unlink()


def _mesh_outputs_present(case_path: Path) -> bool:
    return bool(
        (case_path / "constant" / "polyMesh" / "points").is_file()
        and _latest_body_surface_vtk(case_path) is not None
    )


def _mesh_input_fingerprint(case_path: Path) -> str | None:
    digest = hashlib.sha256()
    geometry_files = [
        path.relative_to(case_path).as_posix()
        for path in sorted((case_path / "constant" / "geometry").glob("*.stl"))
    ]
    relative_paths = sorted(set((*MESH_INPUT_FILES, *geometry_files)))
    for relative_path in relative_paths:
        path = case_path / relative_path
        if not path.is_file():
            return None
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _mesh_record_reusable(case_path: Path) -> bool:
    record = _read_json_object(case_path / "aerolab-mesh.json")
    return bool(
        record.get("reusable")
        and record.get("inputFingerprint") == _mesh_input_fingerprint(case_path)
        and _mesh_outputs_present(case_path)
    )


def _solver_input_fingerprint(case_path: Path) -> str | None:
    """Fingerprint physical/numerical solver inputs while ignoring decomposition."""
    case_path = case_path.resolve()
    script_path = case_path / "Allsolve"
    if not script_path.is_file():
        return None

    input_files = [script_path]
    for root_name in ("0", "constant", "system"):
        root = case_path / root_name
        if not root.is_dir():
            return None
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(case_path)
            parts = relative_path.parts
            if root_name == "constant" and len(parts) > 1 and parts[1] in {
                "geometry",
                "polyMesh",
            }:
                continue
            if relative_path.as_posix() in {
                "system/blockMeshDict",
                "system/decomposeParDict",
                "system/snappyHexMeshDict",
                "system/surfaceFeaturesDict",
            }:
                continue
            input_files.append(path)

    digest = hashlib.sha256()
    for path in sorted(input_files, key=lambda value: value.relative_to(case_path).as_posix()):
        relative_path = path.relative_to(case_path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _latest_numeric_time(case_path: Path) -> float | None:
    latest: float | None = None
    for child in case_path.iterdir():
        if not child.is_dir() or child.name == "0":
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        if not (child / "U").is_file() or not (child / "p").is_file():
            continue
        if latest is None or value > latest:
            latest = value
    return latest


def _resume_compatibility(
    case_path: Path,
    solver_input_fingerprint: str | None,
) -> dict[str, object]:
    if solver_input_fingerprint is None:
        raise ValueError(
            "Resume requires a generated Allsolve script and fingerprintable solver inputs."
        )
    solve_script = (case_path / "Allsolve").read_text(encoding="utf-8", errors="ignore")
    if "AEROLAB_RESUME" not in solve_script or "foamRun -latestTime" not in solve_script:
        raise ValueError(
            "This case predates compatible resume scripts; regenerate it before using resume."
        )
    if not _mesh_record_reusable(case_path):
        raise ValueError(
            "Resume requires a reusable mesh whose input fingerprint still matches this case."
        )
    previous_run = _read_json_object(case_path / "aerolab-run.json")
    if previous_run.get("status") != "failed" or previous_run.get("mode") != "full":
        raise ValueError("Resume requires a previously failed full solver run.")
    previous_fingerprint = previous_run.get("solverInputFingerprint")
    if not isinstance(previous_fingerprint, str):
        raise ValueError(
            "The failed run predates solver-input fingerprints and cannot be resumed safely."
        )
    if previous_fingerprint != solver_input_fingerprint:
        raise ValueError(
            "Solver inputs changed after the failed run; start a clean run instead of resuming."
        )
    latest_time = _latest_numeric_time(case_path)
    if latest_time is None:
        raise ValueError(
            "No reconstructed numeric time state containing U and p is available to resume."
        )
    return {
        "latestTime": latest_time,
        "previousStartedAt": previous_run.get("startedAt"),
        "previousFinishedAt": previous_run.get("finishedAt"),
        "previousProcesses": previous_run.get("processes"),
    }


def _convergence_policy(case_path: Path) -> dict[str, object]:
    case = _read_json_object(case_path / "case.json")
    quality = case.get("cfd_quality")
    values = quality if isinstance(quality, dict) else {}
    simulation_mode = str(values.get("simulation_mode") or "").lower()
    if simulation_mode not in {"steady", "transient"}:
        simulation_type = str(case.get("simulation_type") or "").lower()
        simulation_mode = "transient" if simulation_type.startswith("transient") else "steady"

    if simulation_mode == "steady":
        return {
            "mode": "steady",
            "controller": "foundationResidualControl",
            "source": "system/fvSolution",
            "qualityPreset": values.get("name"),
            "maximumIterations": values.get("end_time"),
            "residualThresholds": {
                "p": values.get("pressure_residual_control"),
                "U": values.get("velocity_residual_control"),
                "turbulence": values.get("turbulence_residual_control"),
            },
            "qualificationGatesUnchanged": True,
        }
    return {
        "mode": "transient",
        "controller": "fixedPhysicalTimeWindow",
        "source": "system/controlDict",
        "qualityPreset": values.get("name"),
        "endTimeSeconds": values.get("end_time"),
        "warmupSeconds": values.get("warmup_time_s"),
        "averagingWindowSeconds": values.get("averaging_window_s"),
        "minimumForceSamples": values.get("minimum_force_samples"),
        "maximumCourantNumber": values.get("maximum_courant_number"),
        "qualificationGatesUnchanged": True,
    }


def _write_mesh_record(
    case_path: Path,
    *,
    backend: str,
    returncode: int,
    started_at: str,
    finished_at: str,
    report: dict[str, object],
    log_path: Path,
    preserve_log: bool,
    processes: int,
    file_handler: str,
    process_selection: dict[str, object],
) -> None:
    assessment = report.get("meshAssessment")
    reusable = bool(
        returncode == 0
        and isinstance(assessment, dict)
        and assessment.get("reusable")
        and _mesh_outputs_present(case_path)
    )
    trusted = bool(isinstance(assessment, dict) and assessment.get("trusted"))
    mesh_log_path = case_path / "aerolab-mesh.log"
    if preserve_log and log_path.is_file():
        shutil.copyfile(log_path, mesh_log_path)
    record = {
        "status": "verified" if trusted else "review" if reusable else "failed",
        "ok": returncode == 0,
        "trusted": trusted,
        "reusable": reusable,
        "backend": backend,
        "processes": processes,
        "fileHandler": file_handler,
        "processSelection": process_selection,
        "returncode": returncode,
        "inputFingerprint": _mesh_input_fingerprint(case_path),
        "logPath": str(mesh_log_path if preserve_log else log_path),
        "startedAt": started_at,
        "finishedAt": finished_at,
        "meshAssessment": assessment,
    }
    (case_path / "aerolab-mesh.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_case_status(case_path: Path, status: str) -> None:
    case_json_path = case_path / "case.json"
    if not case_json_path.exists():
        return
    payload = json.loads(case_json_path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    case_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
