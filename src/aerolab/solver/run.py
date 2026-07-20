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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from ..openfoam import ensure_case_postprocessing
from .analysis import _latest_body_surface_vtk, assess_meshed_surface_fidelity, case_report
from .backends import (
    _backend_cleanup_command,
    _backend_cleanup_verification_command,
    _run_command,
    _select_backend,
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
) -> SolverRunResult:
    case_path = case_path.resolve()
    run_mode = str(run_mode or "full").lower()
    if run_mode not in RUN_MODES:
        raise ValueError(f"Unsupported run mode: {run_mode}")
    if not case_path.exists():
        raise FileNotFoundError(case_path)
    if not (case_path / "Allrun").exists():
        raise FileNotFoundError(case_path / "Allrun")

    ensure_case_postprocessing(case_path)
    status = solver_status()
    selected = _select_backend(status, backend)
    reused_mesh = bool(
        run_mode == "full"
        and reuse_mesh
        and (case_path / "Allsolve").is_file()
        and _mesh_record_reusable(case_path)
    )
    script_name = "Allmesh" if run_mode == "mesh" else "Allsolve" if reused_mesh else "Allrun"
    if not (case_path / script_name).is_file():
        if run_mode == "mesh":
            raise ValueError("This older case has no Allmesh script; create a new case before validating its mesh.")
        raise FileNotFoundError(case_path / script_name)
    started_at = datetime.now(timezone.utc).isoformat()
    log_path = case_path / "aerolab-run.log"
    run_path = case_path / "aerolab-run.json"

    _clear_previous_solver_outputs(case_path, preserve_mesh=reused_mesh)
    execution_id = uuid.uuid4().hex
    command = _run_command(
        case_path,
        selected,
        timeout_seconds=timeout_seconds,
        script_name=script_name,
        execution_id=execution_id,
        solver_identity=solver_identity,
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


def _clear_previous_solver_outputs(case_path: Path, preserve_mesh: bool = False) -> None:
    """Remove generated run products so a rerun cannot reuse stale CFD evidence."""
    case_path = case_path.resolve()
    directory_targets: list[Path] = []
    post_processing = case_path / "postProcessing"
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
    for relative_path in MESH_INPUT_FILES:
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
