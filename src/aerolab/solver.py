from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .openfoam import ensure_case_postprocessing
from .repair import repair_fidelity_for_model, surface_deviation_metrics
from .stl import (
    Triangle,
    mesh_preview,
    read_stl_triangles,
    silhouette_projected_area_for_axis,
    silhouette_projected_areas_for_triangles,
)


CASE_PREVIEW_TRIANGLE_LIMIT = 30_000
MESH_INPUT_FILES = (
    "constant/geometry/body.stl",
    "system/blockMeshDict",
    "system/snappyHexMeshDict",
    "system/surfaceFeaturesDict",
)
RUN_MODES = {"full", "mesh"}

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

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "trusted": self.trusted,
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


def run_case(
    case_path: Path,
    backend: str = "auto",
    timeout_seconds: int = 3600,
    run_mode: str = "full",
    reuse_mesh: bool = True,
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
    command = _run_command(
        case_path,
        selected,
        timeout_seconds=timeout_seconds,
        script_name=script_name,
    )
    process_timeout = timeout_seconds + 900 if selected == "wsl" else timeout_seconds
    _update_case_status(case_path, "mesh_running" if run_mode == "mesh" else "solver_running")
    run_path.write_text(
        json.dumps(
            {
                "status": "running",
                "ok": None,
                "trusted": False,
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
        try:
            completed = subprocess.run(
                command,
                cwd=str(case_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=process_timeout,
            )
            returncode = completed.returncode
            if returncode == 124 and selected == "wsl":
                log_file.write(
                    f"\nAeroLab stopped OpenFOAM after {timeout_seconds} seconds; "
                    "staged partial results were copied back.\n"
                )
        except subprocess.TimeoutExpired:
            returncode = 124
            log_file.write(
                f"\nAeroLab stopped the run after its {process_timeout}-second outer limit.\n"
            )
        except OSError as exc:
            returncode = 127
            log_file.write(f"\nAeroLab could not start the solver process: {exc}\n")
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


def case_run_progress(case_path: Path) -> dict[str, object]:
    """Return a cheap, pollable run state derived from case metadata and the live log."""
    case_path = case_path.resolve()
    case_payload = _read_json_object(case_path / "case.json")
    run_payload = _read_json_object(case_path / "aerolab-run.json")
    case_status = str(case_payload.get("status") or "unknown")
    run_status = str(run_payload.get("status") or "")
    run_mode = str(run_payload.get("mode") or "full")
    log_text = _tail_text(case_path / "aerolab-run.log")
    end_time = 1.0
    quality = case_payload.get("cfd_quality")
    if isinstance(quality, dict):
        try:
            end_time = max(float(quality.get("end_time") or 1.0), 1.0)
        except (TypeError, ValueError):
            end_time = 1.0
    phase, percent, solver_time = _progress_from_log(log_text, end_time, run_mode=run_mode)

    returncode = run_payload.get("returncode")
    trusted = bool(run_payload.get("trusted"))
    running = case_status in {"solver_running", "mesh_running"} or run_status == "running"
    completed = run_mode != "mesh" and (
        case_status in {"solver_verified", "solver_unverified"}
        or (isinstance(returncode, int) and returncode == 0 and not running)
    )
    mesh_completed = run_mode == "mesh" and (
        case_status in {"mesh_validated", "mesh_unverified"}
        or (isinstance(returncode, int) and returncode == 0 and not running)
    )
    failed = case_status in {"solver_failed", "mesh_failed"} or (
        isinstance(returncode, int) and returncode != 0 and not running
    )

    if running:
        state = "running"
        tone = "running"
        label = f"{phase} - {percent}%"
        detail = "Live OpenFOAM progress from the current run log."
    elif completed:
        state = "complete"
        tone = "verified" if trusted or case_status == "solver_verified" else "review"
        percent = 100
        phase = "Complete"
        label = "Complete - verified" if tone == "verified" else "Complete - review checks"
        detail = (
            "Solver finished and all verification gates passed."
            if tone == "verified"
            else "Solver finished, but one or more accuracy checks still need attention."
        )
    elif mesh_completed:
        mesh_record = _read_json_object(case_path / "aerolab-mesh.json")
        mesh_trusted = bool(mesh_record.get("trusted"))
        mesh_reusable = bool(mesh_record.get("reusable"))
        state = "mesh_complete"
        tone = "verified" if mesh_trusted else "review" if mesh_reusable else "failed"
        percent = 100
        phase = "Mesh complete"
        if mesh_trusted:
            label = "Mesh complete - verified"
            detail = "The mesh passed every geometry, resolution, and wall-layer gate and can be reused."
        elif mesh_reusable:
            label = "Mesh complete - review checks"
            detail = "The mesh is reusable, but one or more accuracy checks still need attention before solving."
        else:
            label = "Mesh complete - failed checks"
            detail = "The generated mesh failed a required geometry-fidelity or mesh-quality gate and will not be reused."
    elif failed:
        state = "failed"
        tone = "failed"
        label = f"Failed during {phase.lower()}"
        detail = _run_failure_detail(case_payload, run_payload, log_text, phase)
    elif "=== AEROLAB COMPLETE ===" in log_text:
        state = "complete"
        tone = "review"
        percent = 100
        phase = "Complete"
        label = "Complete - review checks"
        detail = "OpenFOAM completed, but this older run has no recorded verification result."
    else:
        state = "ready"
        tone = "ready"
        phase = "Ready"
        percent = 0
        solver_time = None
        label = "Ready to run"
        detail = "Case files are generated; the 3D airflow is still a preview until OpenFOAM runs."

    return {
        "state": state,
        "tone": tone,
        "phase": phase,
        "percent": int(max(0, min(100, percent))),
        "label": label,
        "detail": detail,
        "isRunning": state == "running",
        "isComplete": state == "complete",
        "isMeshComplete": state == "mesh_complete",
        "runMode": run_mode,
        "solverTime": solver_time,
        "solverEndTime": end_time if solver_time is not None else None,
        "updatedAt": case_payload.get("updated_at") or run_payload.get("finishedAt"),
    }


def _tail_text(path: Path, maximum_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - maximum_bytes))
            return stream.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _run_failure_detail(
    case_payload: dict[str, object],
    run_payload: dict[str, object],
    log_text: str,
    phase: str,
) -> str:
    returncode = run_payload.get("returncode")
    cell_matches = re.findall(r"After refinement[^\n]*cells:\s*(\d+)", log_text, re.IGNORECASE)
    level_matches = re.findall(r"(?m)^\s+(\d+)\s+\d+\s*$", log_text)
    cells = int(cell_matches[-1]) if cell_matches else None
    maximum_level = max((int(value) for value in level_matches), default=None)
    mesh_resolution = case_payload.get("mesh_resolution")
    feature_m = (
        _finite_number(mesh_resolution.get("smallest_aero_feature_m"))
        if isinstance(mesh_resolution, dict)
        else None
    )
    if cells is not None or "Feature refinement iteration" in log_text:
        reason = "timed out" if returncode == 124 else "was interrupted"
        evidence = []
        if cells is not None:
            evidence.append(f"{cells:,} cells")
        if maximum_level is not None:
            evidence.append(f"refinement level {maximum_level}")
        target = f" for the {feature_m * 1000.0:.3g} mm feature target" if feature_m else ""
        reached = f" after reaching {' and '.join(evidence)}" if evidence else ""
        feature_phase = "feature" in phase.lower()
        if feature_phase:
            return (
                f"Feature meshing {reason}{reached}{target}. Recreate the case with a larger smallest-feature "
                "value or simplify/localize tiny edges before rerunning; no usable mesh was produced."
            )
        return (
            f"Mesh refinement {reason}{reached}{target}. OpenFOAM did not report a geometry failure before "
            "the process ended. During near-body refinement this usually indicates local WSL memory pressure; "
            "regenerate the case with the current workstation-safe mesh budget, then validate it again."
        )
    fatal = re.findall(r"FOAM FATAL (?:ERROR|IO ERROR):?\s*([^\n]+)", log_text, re.IGNORECASE)
    if fatal:
        return f"OpenFOAM stopped: {fatal[-1].strip()}"
    if returncode == 124:
        return "The local solver reached its runtime limit before producing a usable result."
    if returncode == 15:
        return "The local solver was interrupted before producing a usable result."
    return "The run stopped before a usable result completed; open the run log for the final solver message."


def _progress_from_log(
    log_text: str,
    solver_end_time: float,
    run_mode: str = "full",
) -> tuple[str, int, float | None]:
    stages = (
        ("=== AEROLAB COMPLETE ===", "Complete", 100),
        ("=== AEROLAB MESH COMPLETE ===", "Mesh complete", 95 if run_mode == "mesh" else 56),
        ("=== AEROLAB STEP: meshSurface ===", "Auditing body surface", 54),
        ("=== AEROLAB STEP: yPlus ===", "Checking wall resolution", 99),
        ("=== AEROLAB STEP: bodyPressure ===", "Mapping body pressure", 98),
        ("=== AEROLAB STEP: wallShearStress ===", "Mapping skin friction", 96),
        ("=== AEROLAB STEP: streamlines ===", "Creating streamlines", 94),
        ("=== AEROLAB STEP: foamRun ===", "Solving airflow", 58),
        ("=== AEROLAB STEP: potentialFoam ===", "Initializing flow", 55),
        ("=== AEROLAB STEP: checkMeshDiagnostics ===", "Auditing mesh", 52),
        ("=== AEROLAB STEP: checkMesh ===", "Checking mesh", 50),
        ("=== AEROLAB STEP: snappyHexMesh ===", "Meshing body", 12),
        ("=== AEROLAB STEP: blockMesh ===", "Building tunnel mesh", 8),
        ("=== AEROLAB STEP: surfaceFeatures ===", "Preparing geometry", 4),
        ("=== AEROLAB WSL: staging case on Linux filesystem ===", "Staging case", 2),
    )
    located = [
        (log_text.rfind(marker), marker, phase, percent)
        for marker, phase, percent in stages
        if marker in log_text
    ]
    if not located:
        return "Starting", 1, None
    position, marker, phase, percent = max(located, key=lambda item: item[0])
    segment = log_text[position + len(marker) :]

    if marker.endswith("foamRun ==="):
        times = re.findall(
            r"^Time\s*=\s*([\deE+.-]+)s?\s*$",
            segment,
            flags=re.MULTILINE,
        )
        solver_time = None
        if times:
            try:
                solver_time = float(times[-1])
            except ValueError:
                solver_time = None
        if solver_time is not None:
            fraction = max(0.0, min(1.0, solver_time / max(solver_end_time, 1e-12)))
            percent = 58 + round(fraction * 34)
        return phase, percent, solver_time

    if marker.endswith("snappyHexMesh ==="):
        mesh_stages = (
            ("Layer addition iteration", "Meshing boundary layers", 44),
            ("Snapping to features", "Snapping body surface", 38),
            ("Shell refinement iteration", "Refining near-body mesh", 30),
            ("Surface refinement iteration", "Refining body surface", 23),
            ("Feature refinement iteration", "Resolving sharp features", 16),
        )
        located_mesh = [
            (segment.rfind(token), mesh_phase, mesh_percent)
            for token, mesh_phase, mesh_percent in mesh_stages
            if token in segment
        ]
        if located_mesh:
            _, phase, percent = max(located_mesh, key=lambda item: item[0])
    return phase, percent, None


def case_report(
    case_path: Path,
    solver_returncode: int | None = None,
    mesh_returncode: int | None = None,
    include_visualization: bool = False,
    include_validation: bool = True,
) -> dict[str, object]:
    case_path = case_path.resolve()
    case_json_path = case_path / "case.json"
    case_payload: dict[str, object] = {}
    if case_json_path.exists():
        try:
            case_payload = json.loads(case_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            case_payload = {}

    force_coeffs = parse_force_coeffs(case_path)
    aerodynamic_forces = _aerodynamic_force_summary(case_payload, force_coeffs)
    run_json_path = case_path / "aerolab-run.json"
    run_payload: dict[str, object] | None = None
    if run_json_path.exists():
        try:
            stored_run = json.loads(run_json_path.read_text(encoding="utf-8"))
            run_payload = {key: stored_run.get(key) for key in (
                "status", "ok", "trusted", "mode", "reusedMesh", "backend", "returncode",
                "logPath", "startedAt", "finishedAt"
            )}
        except json.JSONDecodeError:
            run_payload = None

    if solver_returncode is None and run_payload and run_payload.get("mode") != "mesh":
        stored_returncode = run_payload.get("returncode")
        if isinstance(stored_returncode, int):
            solver_returncode = stored_returncode
    mesh_record = _read_json_object(case_path / "aerolab-mesh.json")
    if mesh_returncode is None:
        stored_mesh_returncode = mesh_record.get("returncode")
        if isinstance(stored_mesh_returncode, int):
            mesh_returncode = stored_mesh_returncode
        elif run_payload and run_payload.get("mode") == "mesh":
            stored_returncode = run_payload.get("returncode")
            if isinstance(stored_returncode, int):
                mesh_returncode = stored_returncode

    mesh_quality = parse_check_mesh(case_path)
    layer_coverage = parse_layer_coverage(case_path, case_payload.get("wall_resolution"))
    residuals = parse_residuals(case_path, case_payload.get("cfd_quality"))
    transient_state = parse_transient_state(case_path, case_payload.get("cfd_quality"))
    y_plus = parse_y_plus(case_path, case_payload.get("wall_resolution"))
    geometry_fidelity = case_payload.get("geometry_fidelity")
    source_model = case_payload.get("model")
    if not isinstance(geometry_fidelity, dict) and isinstance(source_model, str):
        geometry_fidelity = repair_fidelity_for_model(Path(source_model))
        if geometry_fidelity is None:
            geometry_fidelity = {
                "status": "original",
                "verified": True,
                "detail": "Solver geometry is the original STL, not an automatic repair.",
            }
    geometry_validation = case_payload.get("geometry_validation")
    mesh_surface_fidelity = _read_json_object(case_path / "mesh-surface-fidelity.json") or None
    assessment = _quality_assessment(
        solver_returncode,
        mesh_quality,
        layer_coverage,
        residuals,
        transient_state,
        case_payload.get("cfd_quality"),
        force_coeffs,
        y_plus,
        case_payload.get("wall_resolution"),
        geometry_fidelity,
        mesh_surface_fidelity,
        geometry_validation,
        case_payload.get("mesh_resolution"),
        case_payload.get("ground"),
        case_payload.get("placement"),
    )
    mesh_assessment = _mesh_quality_assessment(
        mesh_returncode,
        mesh_quality,
        layer_coverage,
        case_payload.get("wall_resolution"),
        geometry_fidelity,
        mesh_surface_fidelity,
        geometry_validation,
        case_payload.get("mesh_resolution"),
        case_payload.get("ground"),
        case_payload.get("placement"),
    )
    validation = grid_convergence_report(case_path) if include_validation else None

    visualization = _case_visualization(case_path, case_payload) if include_visualization else {}
    return {
        "casePath": str(case_path),
        "caseName": case_payload.get("name", case_path.name),
        "status": case_payload.get("status", "unknown"),
        "sourceModelPath": case_payload.get("model"),
        "geometryReport": case_payload.get("geometry_report"),
        "geometryFidelity": geometry_fidelity,
        "meshSurfaceFidelity": mesh_surface_fidelity,
        "geometryValidation": geometry_validation,
        "scaledGeometryReport": case_payload.get("scaled_geometry_report"),
        "caseSetup": {
            "units": case_payload.get("units"),
            "orientation": case_payload.get("orientation"),
            "flow": case_payload.get("flow"),
            "ground": case_payload.get("ground"),
            "placement": case_payload.get("placement"),
            "quality": case_payload.get("cfd_quality"),
            "simulationType": case_payload.get("simulation_type"),
        },
        "aerodynamicReference": case_payload.get("aerodynamic_reference"),
        "wallResolution": case_payload.get("wall_resolution"),
        "meshResolution": case_payload.get("mesh_resolution"),
        "surfacePressureSetup": _surface_pressure_setup(case_path),
        "forceCoeffs": force_coeffs,
        "aerodynamicForces": aerodynamic_forces,
        "meshQuality": mesh_quality,
        "layerCoverage": layer_coverage,
        "residuals": residuals,
        "transientState": transient_state,
        "yPlus": y_plus,
        "qualityAssessment": assessment,
        "meshAssessment": mesh_assessment,
        "meshRecord": mesh_record or None,
        "gridConvergence": validation,
        "lastRun": run_payload,
        "runProgress": case_run_progress(case_path),
        **visualization,
    }


def assess_meshed_surface_fidelity(
    case_path: Path,
    sample_count: int = 2000,
) -> dict[str, object]:
    case_path = case_path.resolve()
    case_payload = _read_json_object(case_path / "case.json")
    source_path = case_path / "constant" / "geometry" / "body.stl"
    if not source_path.is_file():
        return {
            "status": "missing",
            "verified": False,
            "detail": "The transformed solver STL is missing from constant/geometry/body.stl.",
        }
    vtk_path = _latest_body_surface_vtk(case_path)
    if vtk_path is None:
        return {
            "status": "missing",
            "verified": False,
            "detail": "The meshed OpenFOAM body-patch surface is missing; validate the mesh again.",
        }
    if vtk_path.stat().st_size > 256 * 1024 * 1024:
        return {
            "status": "error",
            "verified": False,
            "detail": "The meshed body-patch VTK exceeds the 256 MB fidelity-audit limit.",
            "meshSurfaceFile": str(vtk_path),
        }

    source_triangles, _ = read_stl_triangles(source_path)
    mesh_triangles = _read_vtk_surface_triangles(vtk_path)
    measured_samples = max(100, int(sample_count))
    deviations = surface_deviation_metrics(
        source_triangles,
        mesh_triangles,
        sample_count=measured_samples,
    )
    flow = case_payload.get("flow")
    flow_axis = str(flow.get("axis") if isinstance(flow, dict) else "x").lower()
    flow_index = {"x": 0, "y": 1, "z": 2}.get(flow_axis, 0)
    source_geometry = _surface_geometry_metrics(source_triangles, silhouette_axis=flow_index)
    mesh_geometry = _surface_geometry_metrics(mesh_triangles, silhouette_axis=flow_index)
    longest = max(source_geometry["dimensions"], default=0.0)
    if longest <= 0:
        raise ValueError("The solver STL has no usable physical dimensions.")

    mesh_resolution = case_payload.get("mesh_resolution")
    if not isinstance(mesh_resolution, dict):
        mesh_resolution = {}
    surface_cell = _finite_number(mesh_resolution.get("estimated_surface_cell_m"))
    smallest_feature = _finite_number(mesh_resolution.get("smallest_aero_feature_m"))
    cells_across = _finite_number(mesh_resolution.get("estimated_cells_across_feature"))
    max_p95 = max((surface_cell or longest * 0.01) * 0.5, longest * 1e-5)
    max_p99 = max(surface_cell or longest * 0.01, longest * 2e-5)
    if smallest_feature is not None and smallest_feature > 0:
        max_p95 = min(max_p95, smallest_feature / 8.0)
        max_p99 = min(max_p99, smallest_feature / 4.0)

    source_dimensions = source_geometry["dimensions"]
    mesh_dimensions = mesh_geometry["dimensions"]
    dimension_change = max(
        abs(mesh_dimensions[index] - source_dimensions[index])
        / max(abs(source_dimensions[index]), longest * 1e-6)
        for index in range(3)
    )
    source_projected = source_geometry["silhouetteAreas"][flow_index]
    mesh_projected = mesh_geometry["silhouetteAreas"][flow_index]
    projected_area_change = abs(mesh_projected - source_projected) / max(source_projected, longest**2 * 1e-9)
    source_normal_projected = source_geometry["projectedAreas"][flow_index]
    mesh_normal_projected = mesh_geometry["projectedAreas"][flow_index]
    normal_projected_area_change = abs(mesh_normal_projected - source_normal_projected) / max(
        source_normal_projected,
        longest**2 * 1e-9,
    )
    feature_resolution_pass = not (
        smallest_feature is not None
        and smallest_feature > 0
        and (cells_across is None or cells_across < 4.0)
    )
    verified = bool(
        deviations["symmetricP95"] <= max_p95
        and deviations["symmetricP99"] <= max_p99
        and dimension_change <= 0.01
        and projected_area_change <= 0.02
        and feature_resolution_pass
    )
    return {
        "status": "verified" if verified else "failed",
        "verified": verified,
        "sourceFile": str(source_path),
        "meshSurfaceFile": str(vtk_path),
        "sourceTriangleCount": len(source_triangles),
        "meshTriangleCount": len(mesh_triangles),
        "sampleCountPerSurface": measured_samples,
        "sourceToMeshP95M": deviations["sourceP95"],
        "sourceToMeshP99M": deviations["sourceP99"],
        "meshToSourceP95M": deviations["outputP95"],
        "meshToSourceP99M": deviations["outputP99"],
        "symmetricP95M": deviations["symmetricP95"],
        "symmetricP99M": deviations["symmetricP99"],
        "maximumP95M": max_p95,
        "maximumP99M": max_p99,
        "dimensionChangePercent": dimension_change * 100.0,
        "projectedAreaChangePercent": projected_area_change * 100.0,
        "normalProjectedAreaChangePercent": normal_projected_area_change * 100.0,
        "estimatedSurfaceCellM": surface_cell,
        "smallestAeroFeatureM": smallest_feature,
        "estimatedCellsAcrossFeature": cells_across,
        "featureResolutionVerified": feature_resolution_pass,
        "integrationMethod": "exact BVH point-to-triangle sampling in both directions",
        "detail": (
            "The actual OpenFOAM body patch preserved the transformed solver STL within the mesh-fidelity limits."
            if verified
            else "The actual OpenFOAM body patch lost geometry or does not resolve the requested smallest aero feature."
        ),
    }


def _latest_body_surface_vtk(case_path: Path) -> Path | None:
    post_dirs = (
        case_path / "postProcessing" / "bodyPressure",
        case_path / "postProcessing" / "meshSurface",
    )
    candidates = sorted(
        (
            path
            for post_dir in post_dirs
            if post_dir.is_dir()
            for path in post_dir.rglob("*.vtk")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_vtk_surface_triangles(path: Path) -> list[Triangle]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not any("ASCII" in line.upper() for line in lines[:10]):
        raise ValueError("Only ASCII legacy VTK body surfaces can be audited.")
    points_header = _vtk_header(lines, "POINTS")
    polygons_header = _vtk_header(lines, "POLYGONS")
    if not points_header or not polygons_header:
        raise ValueError("The body-patch VTK has no points or polygons.")
    point_line, point_parts = points_header
    polygon_line, polygon_parts = polygons_header
    point_count = int(point_parts[1])
    polygon_count = int(polygon_parts[1])
    connectivity_count = int(polygon_parts[2])
    point_values = _vtk_values(lines, point_line + 1, point_count * 3, float)
    connectivity = _vtk_values(lines, polygon_line + 1, connectivity_count, int)
    if len(point_values) != point_count * 3 or len(connectivity) != connectivity_count:
        raise ValueError("The body-patch VTK geometry arrays are incomplete.")
    points = [tuple(point_values[index : index + 3]) for index in range(0, len(point_values), 3)]
    triangles: list[Triangle] = []
    cursor = 0
    for _ in range(polygon_count):
        count = int(connectivity[cursor])
        indices = [int(value) for value in connectivity[cursor + 1 : cursor + 1 + count]]
        cursor += count + 1
        if len(indices) < 3 or not all(0 <= index < point_count for index in indices):
            continue
        for offset in range(1, len(indices) - 1):
            triangles.append((points[indices[0]], points[indices[offset]], points[indices[offset + 1]]))
    if not triangles:
        raise ValueError("The body-patch VTK has no usable triangles.")
    return triangles


def _surface_geometry_metrics(
    triangles: list[Triangle],
    silhouette_axis: int | None = None,
) -> dict[str, tuple[float, float, float]]:
    vertices = [point for triangle in triangles for point in triangle]
    minimum = tuple(min(point[axis] for point in vertices) for axis in range(3))
    maximum = tuple(max(point[axis] for point in vertices) for axis in range(3))
    projected = [0.0, 0.0, 0.0]
    for a, b, c in triangles:
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        for axis in range(3):
            projected[axis] += abs(cross[axis]) * 0.25
    if silhouette_axis is None:
        silhouette = silhouette_projected_areas_for_triangles(triangles)
        silhouette_values = (silhouette.x, silhouette.y, silhouette.z)
    else:
        silhouette_values_list = [0.0, 0.0, 0.0]
        silhouette_values_list[silhouette_axis] = silhouette_projected_area_for_axis(
            triangles,
            silhouette_axis,
        )
        silhouette_values = tuple(silhouette_values_list)
    return {
        "dimensions": tuple(maximum[axis] - minimum[axis] for axis in range(3)),
        "projectedAreas": tuple(projected),
        "silhouetteAreas": silhouette_values,
    }


def _surface_pressure_setup(case_path: Path) -> dict[str, object]:
    config_path = case_path / "system" / "bodyPressure"
    wall_shear_path = case_path / "system" / "wallShearStress"
    allrun_path = case_path / "Allrun"
    allrun_hook = False
    wall_shear_hook = False
    if allrun_path.is_file():
        allrun_text = allrun_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        allrun_hook = "foamPostProcess -func bodyPressure" in allrun_text
        wall_shear_hook = (
            "foamPostProcess -solver incompressibleFluid -func wallShearStress" in allrun_text
        )
    post_dir = case_path / "postProcessing" / "bodyPressure"
    output_available = post_dir.is_dir() and any(post_dir.rglob("*.vtk"))
    return {
        "configured": config_path.is_file() and allrun_hook and wall_shear_path.is_file() and wall_shear_hook,
        "configFile": str(config_path),
        "configExists": config_path.is_file(),
        "allrunHook": allrun_hook,
        "wallShearConfigFile": str(wall_shear_path),
        "wallShearConfigured": wall_shear_path.is_file() and wall_shear_hook,
        "outputAvailable": output_available,
    }


def grid_convergence_report(case_path: Path) -> dict[str, object] | None:
    case_path = case_path.resolve()
    selected_payload = _read_json_object(case_path / "case.json")
    selected_study = selected_payload.get("validation_study")
    if not isinstance(selected_study, dict) or not selected_study.get("id"):
        return None

    study_id = str(selected_study["id"])
    levels = ("draft", "standard", "fine")
    study_cases: dict[str, list[tuple[Path, dict[str, object]]]] = {level: [] for level in levels}
    for case_json_path in case_path.parent.glob("*/case.json"):
        payload = _read_json_object(case_json_path)
        study = payload.get("validation_study")
        if not isinstance(study, dict) or str(study.get("id") or "") != study_id:
            continue
        level = str(study.get("level") or "").lower()
        if level in study_cases:
            study_cases[level].append((case_json_path.parent, payload))

    records: list[dict[str, object]] = []
    signatures: list[str] = []
    duplicate_levels: list[str] = []
    for level in levels:
        matches = study_cases[level]
        if len(matches) > 1:
            duplicate_levels.append(level)
        if not matches:
            records.append(
                {
                    "level": level,
                    "casePath": None,
                    "status": "missing",
                    "trusted": False,
                    "cells": None,
                    "meanCd": None,
                    "meanCl": None,
                }
            )
            continue
        level_path, payload = matches[0]
        signatures.append(_study_setup_signature(payload))
        report = case_report(level_path, include_validation=False)
        assessment = report.get("qualityAssessment")
        mesh = report.get("meshQuality")
        forces = report.get("forceCoeffs")
        last_run = report.get("lastRun")
        trusted = bool(isinstance(assessment, dict) and assessment.get("trusted"))
        records.append(
            {
                "level": level,
                "casePath": str(level_path),
                "status": "verified" if trusted else "unverified" if last_run else "pending",
                "trusted": trusted,
                "cells": mesh.get("cells") if isinstance(mesh, dict) else None,
                "meanCd": _coefficient_value(forces, "meanCd", "Cd"),
                "meanCl": _coefficient_value(forces, "meanCl", "Cl"),
            }
        )

    present = all(record["casePath"] for record in records) and not duplicate_levels
    matching_setup = len(set(signatures)) <= 1 and not duplicate_levels
    failed_runs = [record for record in records if record["status"] == "unverified"]
    all_trusted = all(bool(record["trusted"]) for record in records)
    checks = [
        _study_check(
            "Three mesh levels",
            True if present else None if not duplicate_levels else False,
            "Draft, standard, and fine cases are present exactly once.",
            "Create one draft, standard, and fine case for this study.",
        ),
        _study_check(
            "Matching setup",
            matching_setup if present else None,
            "All cases use the same model, scale, flow, ground, and aerodynamic references.",
            "The study cases do not share an identical physical setup.",
        ),
        _study_check(
            "Verified runs",
            True if all_trusted else False if failed_runs else None,
            "All three OpenFOAM runs passed mesh, residual, force, and y+ gates.",
            "Run and verify all three cases before evaluating grid sensitivity.",
        ),
    ]

    cell_counts = [_finite_number(record.get("cells")) for record in records]
    mesh_growth: bool | None = None
    if all(value is not None for value in cell_counts):
        draft_cells, standard_cells, fine_cells = (float(value) for value in cell_counts)
        mesh_growth = bool(
            draft_cells < standard_cells < fine_cells
            and standard_cells / draft_cells >= 1.15
            and fine_cells / standard_cells >= 1.15
        )
    checks.append(
        _study_check(
            "Mesh growth",
            mesh_growth,
            "Actual cell counts increase meaningfully at each quality level.",
            "Cell counts must increase by at least 15% from draft to standard to fine.",
        )
    )

    cd_values = [_finite_number(record.get("meanCd")) for record in records]
    cl_values = [_finite_number(record.get("meanCl")) for record in records]
    cd_metrics: dict[str, object] | None = None
    cd_converged: bool | None = None
    if all_trusted and mesh_growth and all(value is not None for value in cd_values):
        draft_cd, standard_cd, fine_cd = (float(value) for value in cd_values)
        coarse_change = abs(standard_cd - draft_cd)
        fine_change = abs(fine_cd - standard_cd)
        fine_change_percent = fine_change / max(abs(fine_cd), 0.05) * 100.0
        monotonic = (standard_cd - draft_cd) * (fine_cd - standard_cd) >= 0.0
        decreasing = fine_change <= coarse_change
        cd_converged = monotonic and decreasing and fine_change_percent <= 2.0
        cd_metrics = {
            "draftToStandardAbsolute": coarse_change,
            "standardToFineAbsolute": fine_change,
            "standardToFinePercent": fine_change_percent,
            "monotonic": monotonic,
            "decreasingChange": decreasing,
        }
    checks.append(
        _study_check(
            "Drag grid sensitivity",
            cd_converged,
            "Fine-versus-standard mean Cd differs by no more than 2% with a converging trend.",
            "Mean Cd is missing, changes by more than 2%, or does not converge monotonically.",
        )
    )

    cl_metrics: dict[str, object] | None = None
    cl_converged: bool | None = None
    if all_trusted and mesh_growth and all(value is not None for value in cl_values):
        standard_cl = float(cl_values[1])
        fine_cl = float(cl_values[2])
        fine_change = abs(fine_cl - standard_cl)
        cl_converged = fine_change <= 0.02
        cl_metrics = {"standardToFineAbsolute": fine_change}
    checks.append(
        _study_check(
            "Lift grid sensitivity",
            cl_converged,
            "Fine-versus-standard mean Cl differs by no more than 0.02.",
            "Mean Cl is missing or changes by more than 0.02 on the fine mesh.",
        )
    )

    validated = all(check["status"] == "pass" for check in checks)
    failed = any(check["status"] == "fail" for check in checks)
    fine_record = records[2]
    return {
        "studyId": study_id,
        "status": "validated" if validated else "failed" if failed else "incomplete",
        "validated": validated,
        "levels": records,
        "checks": checks,
        "dragMetrics": cd_metrics,
        "liftMetrics": cl_metrics,
        "recommendedCd": fine_record.get("meanCd") if validated else None,
        "recommendedCl": fine_record.get("meanCl") if validated else None,
    }


def _study_check(label: str, passed: bool | None, success: str, failure: str) -> dict[str, str]:
    if passed is None:
        return {"label": label, "status": "pending", "detail": failure}
    return {"label": label, "status": "pass" if passed else "fail", "detail": success if passed else failure}


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _study_setup_signature(payload: dict[str, object]) -> str:
    keys = (
        "model",
        "units",
        "orientation",
        "simulation_type",
        "solver_module",
        "flow",
        "ground",
        "aerodynamic_reference",
        "wall_resolution",
    )
    return json.dumps({key: payload.get(key) for key in keys}, sort_keys=True, separators=(",", ":"))


def _coefficient_value(payload: object, preferred: str, fallback: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(preferred) if payload.get(preferred) is not None else payload.get(fallback)
    return _finite_number(value)


def _aerodynamic_force_summary(
    case_payload: dict[str, object],
    force_coeffs: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(force_coeffs, dict):
        return None
    mean_cd = _coefficient_value(force_coeffs, "meanCd", "Cd")
    mean_cl = _coefficient_value(force_coeffs, "meanCl", "Cl")
    flow = case_payload.get("flow")
    reference = case_payload.get("aerodynamic_reference")
    if not isinstance(flow, dict) or not isinstance(reference, dict):
        return None
    speed_mps = _finite_number(flow.get("speed_mps"))
    speed_mph = _finite_number(flow.get("speed_mph"))
    density = _finite_number(flow.get("air_density_kg_m3")) or 1.225
    area_m2 = _finite_number(reference.get("area_m2"))
    if speed_mps is None or area_m2 is None or speed_mps < 0 or area_m2 <= 0:
        return None
    dynamic_pressure = 0.5 * density * speed_mps * speed_mps
    lift_n = mean_cl * dynamic_pressure * area_m2 if mean_cl is not None else None
    drag_n = mean_cd * dynamic_pressure * area_m2 if mean_cd is not None else None
    newtons_per_lbf = 4.4482216152605
    vertical_type = None
    vertical_n = None
    if lift_n is not None:
        vertical_type = "downforce" if lift_n < 0 else "lift"
        vertical_n = abs(lift_n)
    return {
        "speedMps": speed_mps,
        "speedMph": speed_mph,
        "airDensityKgM3": density,
        "referenceAreaM2": area_m2,
        "dynamicPressurePa": dynamic_pressure,
        "dragN": drag_n,
        "dragLbf": drag_n / newtons_per_lbf if drag_n is not None else None,
        "signedLiftN": lift_n,
        "verticalForceType": vertical_type,
        "verticalForceN": vertical_n,
        "verticalForceLbf": vertical_n / newtons_per_lbf if vertical_n is not None else None,
    }


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_force_coeffs(case_path: Path) -> dict[str, object] | None:
    post_dir = case_path / "postProcessing" / "forceCoeffs"
    if not post_dir.exists():
        return None

    candidates = sorted(
        [
            path
            for path in post_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".dat", ".csv"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    quality = _read_json_object(case_path / "case.json").get("cfd_quality")
    for path in candidates:
        parsed = _parse_coeff_file(path, quality)
        if parsed:
            return parsed
    return None


def _parse_coeff_file(path: Path, quality: object = None) -> dict[str, object] | None:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            clean = line.lstrip("#").strip()
            if clean.lower().startswith("time"):
                header = clean.split()
            continue
        parts = line.split()
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue
        if not values:
            continue
        active_header = header if header and len(header) == len(values) else _fallback_coeff_header(len(values))
        rows.append(dict(zip(active_header, values)))

    if not rows:
        return None
    latest = rows[-1]
    transient = isinstance(quality, dict) and quality.get("simulation_mode") == "transient"
    averaging_rows = rows
    averaging_window = None
    if transient:
        averaging_window = _finite_number(quality.get("averaging_window_s"))
        latest_time = latest.get("Time")
        if averaging_window is not None and latest_time is not None:
            window_start = float(latest_time) - averaging_window
            averaging_rows = [row for row in rows if row.get("Time", -math.inf) >= window_start]
    window_count = (
        len(averaging_rows)
        if transient
        else min(len(rows), max(20, math.ceil(len(rows) * 0.2)))
    )
    cd_stats = _series_statistics(averaging_rows, "Cd", window_count)
    cl_stats = _series_statistics(averaging_rows, "Cl", window_count)
    reasons: list[str] = []
    minimum_samples = 30
    if transient and isinstance(quality, dict):
        minimum_samples = int(quality.get("minimum_force_samples") or 100)
    available_samples = window_count if transient else len(rows)
    stable = available_samples >= minimum_samples
    if available_samples < minimum_samples:
        reasons.append(
            f"At least {minimum_samples} force samples are required"
            + (" in the averaging window." if transient else ".")
        )
    if not cd_stats:
        stable = False
        reasons.append("Cd history is missing.")
    else:
        if not transient and cd_stats["relativeRange"] > 0.03:
            stable = False
            reasons.append("Cd varies by more than 3% in the final window.")
        if cd_stats["relativeDrift"] > 0.01:
            stable = False
            reasons.append("Mean Cd is still drifting by more than 1%.")
    if cl_stats and (
        cl_stats["absoluteDrift"] > 0.02
        or (not transient and cl_stats["range"] > 0.05)
    ):
        stable = False
        reasons.append("Mean Cl has not settled in the final window.")

    return {
        "file": str(path),
        "latest": latest,
        "time": latest.get("Time"),
        "Cd": latest.get("Cd"),
        "Cl": latest.get("Cl"),
        "Cs": latest.get("Cs"),
        "CmPitch": latest.get("CmPitch"),
        "meanCd": cd_stats.get("mean") if cd_stats else None,
        "meanCl": cl_stats.get("mean") if cl_stats else None,
        "averagingMode": "time-window" if transient else "final-sample-window",
        "windowStartTime": averaging_rows[0].get("Time") if averaging_rows else None,
        "windowEndTime": averaging_rows[-1].get("Time") if averaging_rows else None,
        "stable": stable,
        "statistics": {
            "sampleCount": len(rows),
            "windowSampleCount": window_count,
            "averagingWindowSeconds": averaging_window,
            "Cd": cd_stats,
            "Cl": cl_stats,
            "stable": stable,
            "reasons": reasons,
        },
    }


def _series_statistics(rows: list[dict[str, float]], key: str, window_count: int) -> dict[str, float] | None:
    values = [row[key] for row in rows[-window_count:] if key in row]
    if not values:
        return None
    mean = statistics.fmean(values)
    midpoint = max(1, len(values) // 2)
    first_mean = statistics.fmean(values[:midpoint])
    second_mean = statistics.fmean(values[midpoint:]) if values[midpoint:] else first_mean
    value_range = max(values) - min(values)
    denominator = max(abs(mean), 1e-9)
    return {
        "mean": mean,
        "standardDeviation": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": value_range,
        "relativeRange": value_range / denominator,
        "absoluteDrift": abs(second_mean - first_mean),
        "relativeDrift": abs(second_mean - first_mean) / denominator,
    }


def _fallback_coeff_header(count: int) -> list[str]:
    names = ["Time", "Cd", "Cs", "Cl", "CmRoll", "CmPitch", "CmYaw", "CdFront", "CdRear", "ClFront", "ClRear"]
    if count <= len(names):
        return names[:count]
    return names + [f"value{index}" for index in range(len(names), count)]


def parse_check_mesh(case_path: Path) -> dict[str, object] | None:
    log_path = case_path / "aerolab-run.log"
    if not log_path.exists():
        return None
    full_text = log_path.read_text(encoding="utf-8", errors="ignore")
    text = full_text
    marker = "=== AEROLAB STEP: checkMesh ==="
    if marker in text:
        text = text.rsplit(marker, 1)[1].split("=== AEROLAB STEP:", 1)[0]
    elif "Mesh OK." not in text and "mesh checks" not in text:
        return None

    failed_match = re.search(r"Failed\s+(\d+)\s+mesh checks?", text, re.IGNORECASE)
    cells = _last_number(text, r"\bcells:\s+(\d+)", int)
    max_aspect_ratio = _last_number(text, r"Max aspect ratio\s*=\s*([\deE+.-]+)")
    max_non_orthogonality = _last_number(text, r"Mesh non-orthogonality Max:\s*([\deE+.-]+)")
    max_skewness = _last_number(text, r"Max skewness\s*=\s*([\deE+.-]+)")
    passed = "Mesh OK." in text and not failed_match
    failed_checks = int(failed_match.group(1)) if failed_match else 0
    diagnostics_marker = "=== AEROLAB STEP: checkMeshDiagnostics ==="
    diagnostics = ""
    if diagnostics_marker in full_text:
        diagnostics = full_text.rsplit(diagnostics_marker, 1)[1].split("=== AEROLAB STEP:", 1)[0]
    warnings: list[str] = []
    concave_cells = _last_number(diagnostics, r"Concave cells.*?number of cells:\s*(\d+)", int)
    diagnostic_failed = _last_number(diagnostics, r"Failed\s+(\d+)\s+mesh checks?", int)
    if concave_cells:
        warnings.append(
            f"The exhaustive geometry scan found {concave_cells} concave cut cells; "
            "review them before treating a final vehicle run as validated."
        )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "failedChecks": failed_checks,
        "diagnosticFailedChecks": diagnostic_failed or 0,
        "warnings": warnings,
        "cells": cells,
        "maxAspectRatio": max_aspect_ratio,
        "maxNonOrthogonality": max_non_orthogonality,
        "maxSkewness": max_skewness,
    }


def parse_layer_coverage(
    case_path: Path,
    wall_resolution: object = None,
) -> dict[str, object] | None:
    requested_layers = 0
    if isinstance(wall_resolution, dict):
        try:
            requested_layers = int(wall_resolution.get("surface_layers") or 0)
        except (TypeError, ValueError):
            requested_layers = 0
    if requested_layers <= 0:
        return {
            "status": "not_applicable",
            "passed": True,
            "requestedLayers": 0,
            "detail": "This setup does not request prism boundary layers.",
        }

    log_path = case_path / "aerolab-run.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    marker = "=== AEROLAB STEP: snappyHexMesh ==="
    if marker in text:
        text = text.rsplit(marker, 1)[1].split("=== AEROLAB STEP:", 1)[0]

    final_summaries = re.findall(
        r"patch\s+faces\s+layers\s+overall thickness.*?\n"
        r"(?:.*\n){0,5}?body\s+(\d+)\s+([\deE+.-]+)\s+",
        text,
        re.IGNORECASE,
    )
    snapped_cells = _last_number(text, r"Snapped mesh\s*:\s*cells:(\d+)", int)
    layer_mesh_cells = _last_number(text, r"Layer mesh\s*:\s*cells:(\d+)", int)
    if final_summaries:
        target_faces = int(final_summaries[-1][0])
        average_layers = max(0.0, float(final_summaries[-1][1]))
        target_cells = target_faces * requested_layers
        added_layer_cells = max(0, int(layer_mesh_cells or 0) - int(snapped_cells or 0))
        added_layer_cells = min(added_layer_cells, target_cells)
        cell_percent = added_layer_cells / target_cells * 100.0 if target_cells else 0.0
        # Average layers gives a conservative upper bound on complete-stack face coverage.
        face_percent = min(100.0, average_layers / requested_layers * 100.0)
        full_layer_faces = int(math.floor(target_faces * face_percent / 100.0))
        coverage_method = "final snappyHexMesh layer summary and actual cell-count delta"
    else:
        face_pairs = [
            (int(added), int(target))
            for added, target in re.findall(r"Extruding\s+(\d+)\s+out of\s+(\d+)\s+faces", text)
        ]
        cell_pairs = [
            (int(added), int(target))
            for added, target in re.findall(r"Added\s+(\d+)\s+out of\s+(\d+)\s+cells", text)
        ]
        if not face_pairs and not cell_pairs:
            return None
        target_faces = max((target for _, target in face_pairs), default=0)
        full_layer_faces = min(sum(added for added, _ in face_pairs), target_faces) if target_faces else 0
        target_cells = max((target for _, target in cell_pairs), default=0)
        added_layer_cells = min(sum(added for added, _ in cell_pairs), target_cells) if target_cells else 0
        face_percent = full_layer_faces / target_faces * 100.0 if target_faces else 0.0
        cell_percent = added_layer_cells / target_cells * 100.0 if target_cells else 0.0
        average_layers = cell_percent / 100.0 * requested_layers
        coverage_method = "legacy cumulative extrusion log"
    minimum_average_layers = min(3.0, float(requested_layers))
    passed = (
        face_percent >= 70.0
        and cell_percent >= 70.0
        and average_layers >= minimum_average_layers
    )
    reasons: list[str] = []
    if face_percent < 70.0:
        reasons.append("Fewer than 70% of body faces received the complete requested layer stack.")
    if cell_percent < 70.0:
        reasons.append("Fewer than 70% of requested boundary-layer cells were added.")
    if average_layers < minimum_average_layers:
        reasons.append(f"The body averages fewer than {minimum_average_layers:g} prism layers.")
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "requestedLayers": requested_layers,
        "averageLayers": average_layers,
        "targetFaces": target_faces,
        "fullLayerFaces": full_layer_faces,
        "fullLayerFaceCoveragePercent": face_percent,
        "targetLayerCells": target_cells,
        "addedLayerCells": added_layer_cells,
        "layerCellCoveragePercent": cell_percent,
        "coverageMethod": coverage_method,
        "minimumFullLayerFaceCoveragePercent": 70.0,
        "minimumLayerCellCoveragePercent": 70.0,
        "minimumAverageLayers": minimum_average_layers,
        "reasons": reasons,
    }


def parse_residuals(case_path: Path, quality: object = None) -> dict[str, object] | None:
    log_path = case_path / "aerolab-run.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "=== AEROLAB STEP: foamRun ===" not in text:
        return None
    text = text.split("=== AEROLAB STEP: foamRun ===", 1)[1]
    pattern = re.compile(
        r"Solving for\s+([^,]+),\s+Initial residual\s*=\s*([\deE+.-]+),\s+"
        r"Final residual\s*=\s*([\deE+.-]+)",
        re.IGNORECASE,
    )
    histories: dict[str, list[float]] = {}
    for match in pattern.finditer(text):
        try:
            histories.setdefault(match.group(1).strip(), []).append(float(match.group(2)))
        except ValueError:
            continue
    if not histories:
        return None

    fields: dict[str, dict[str, object]] = {}
    stable = True
    reasons: list[str] = []
    transient = isinstance(quality, dict) and quality.get("simulation_mode") == "transient"
    transient_ceiling = (
        _finite_number(quality.get("transient_residual_ceiling"))
        if isinstance(quality, dict)
        else None
    ) or 0.2
    for name, values in histories.items():
        window = values[-min(50 if transient else 20, len(values)) :]
        threshold = _residual_threshold(name, quality)
        enough_samples = len(values) >= 20
        if transient:
            field_stable = enough_samples and max(window) <= transient_ceiling
        else:
            field_stable = enough_samples and values[-1] <= threshold and statistics.fmean(window) <= threshold * 3
        fields[name] = {
            "sampleCount": len(values),
            "latest": values[-1],
            "windowMean": statistics.fmean(window),
            "windowMax": max(window),
            "threshold": transient_ceiling if transient else threshold,
            "stable": field_stable,
        }
        if not field_stable:
            stable = False
            reasons.append(
                f"{name} residual exceeds the transient divergence ceiling."
                if transient
                else f"{name} residual has not reached its convergence gate."
            )

    required_groups = {
        "velocity": any(name.lower().startswith("u") for name in histories),
        "pressure": any(name.lower() in {"p", "p_rgh"} for name in histories),
        "turbulence": any(name.lower() in {"k", "omega"} for name in histories),
    }
    missing = [name for name, present in required_groups.items() if not present]
    if missing:
        stable = False
        reasons.append(f"Missing residual history for {', '.join(missing)} fields.")

    return {
        "status": "pass" if stable else "fail",
        "stable": stable,
        "mode": "transient-divergence" if transient else "steady-convergence",
        "fields": fields,
        "reasons": reasons,
    }


def parse_transient_state(case_path: Path, quality: object = None) -> dict[str, object] | None:
    if not isinstance(quality, dict) or quality.get("simulation_mode") != "transient":
        return None
    log_path = case_path / "aerolab-run.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "=== AEROLAB STEP: foamRun ===" in text:
        text = text.split("=== AEROLAB STEP: foamRun ===", 1)[1]
    times = [float(value) for value in re.findall(r"(?m)^Time\s*=\s*([\deE+.-]+)s?\s*$", text)]
    courant = [
        (float(mean), float(maximum))
        for mean, maximum in re.findall(
            r"Courant Number mean:\s*([\deE+.-]+)\s+max:\s*([\deE+.-]+)",
            text,
            re.IGNORECASE,
        )
    ]
    end_time = _finite_number(quality.get("end_time")) or 0.0
    latest_time = max(times, default=0.0)
    max_co_limit = _finite_number(quality.get("maximum_courant_number")) or 1.5
    max_co = max((item[1] for item in courant), default=None)
    completed = end_time > 0 and latest_time >= end_time * 0.995
    mean_fields = _latest_transient_mean_fields(case_path)
    averaged = {"UMean", "pMean"}.issubset(mean_fields)
    courant_controlled = max_co is not None and max_co <= max_co_limit * 1.05
    return {
        "status": "pass" if completed and averaged and courant_controlled else "fail",
        "completed": completed,
        "latestTime": latest_time,
        "targetEndTime": end_time,
        "courantControlled": courant_controlled,
        "maximumCourant": max_co,
        "maximumCourantLimit": max_co_limit,
        "timeAveraged": averaged,
        "meanFields": sorted(mean_fields),
    }


def _latest_transient_mean_fields(case_path: Path) -> set[str]:
    time_directories: list[tuple[float, Path]] = []
    for path in case_path.iterdir():
        if not path.is_dir():
            continue
        try:
            time_directories.append((float(path.name), path))
        except ValueError:
            continue
    if not time_directories:
        return set()
    latest = max(time_directories, key=lambda item: item[0])[1]
    return {name for name in ("UMean", "pMean", "kMean", "wallShearStressMean") if (latest / name).is_file()}


def parse_y_plus(case_path: Path, wall_resolution: object = None) -> dict[str, object] | None:
    log_path = case_path / "aerolab-run.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    marker = "=== AEROLAB STEP: yPlus ==="
    if marker in text:
        text = text.rsplit(marker, 1)[1].split("=== AEROLAB STEP:", 1)[0]
    pattern = re.compile(
        r"patch\s+(\S+)\s+y\+\s*:\s*min\s*=\s*([\deE+.-]+),\s*"
        r"max\s*=\s*([\deE+.-]+),\s*average\s*=\s*([\deE+.-]+)",
        re.IGNORECASE,
    )
    patches: dict[str, dict[str, float]] = {}
    for patch, minimum, maximum, average in pattern.findall(text):
        patches[patch] = {
            "min": float(minimum),
            "max": float(maximum),
            "average": float(average),
        }
    if not patches:
        return None

    body = patches.get("body") or patches.get("bodyGroup")
    if body is None:
        body = next((values for name, values in patches.items() if name.lower() != "ground"), None)
    distribution = _latest_y_plus_distribution(case_path, "body")
    if body is not None and distribution:
        body.update(distribution)
    target = 80.0
    if isinstance(wall_resolution, dict):
        try:
            target = float(
                wall_resolution.get("target_y_plus")
                or wall_resolution.get("estimated_y_plus")
                or target
            )
        except (TypeError, ValueError):
            pass
    upper_tail = body.get("p95", body["max"]) if body else math.inf
    passed = bool(
        body
        and target * 0.35 <= body["average"] <= target * 2.0
        and upper_tail <= target * 4.0
    )
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "target": target,
        "body": body,
        "patches": patches,
    }


def _latest_y_plus_distribution(case_path: Path, patch_name: str) -> dict[str, float] | None:
    candidates: list[tuple[float, Path]] = []
    for field_path in case_path.glob("*/yPlus"):
        try:
            candidates.append((float(field_path.parent.name), field_path))
        except ValueError:
            continue
    if not candidates:
        return None
    field_path = max(candidates, key=lambda item: item[0])[1]
    text = field_path.read_text(encoding="utf-8", errors="ignore")
    patch_match = re.search(rf"\b{re.escape(patch_name)}\s*\{{(.*?)\n\s*\}}", text, re.DOTALL)
    if not patch_match:
        return None
    values_match = re.search(
        r"value\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)",
        patch_match.group(1),
        re.DOTALL,
    )
    if not values_match:
        return None
    try:
        values = [float(value) for value in values_match.group(1).split()]
    except ValueError:
        return None
    if not values:
        return None
    return {
        "average": statistics.fmean(values),
        "p05": _percentile(values, 0.05),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def _case_visualization(case_path: Path, case_payload: dict[str, object]) -> dict[str, object]:
    body_path = case_path / "constant" / "geometry" / "body.stl"
    if not body_path.exists():
        body_path = case_path / "constant" / "triSurface" / "body.stl"
    if not body_path.exists():
        return {}

    flow = case_payload.get("flow")
    orientation = case_payload.get("orientation")
    flow_axis = "x"
    if isinstance(flow, dict) and str(flow.get("axis") or "").lower() in {"x", "y", "z"}:
        flow_axis = str(flow["axis"]).lower()
    elif isinstance(orientation, dict):
        candidate = str(orientation.get("target_flow_axis") or "x").lower()
        if candidate in {"x", "y", "z"}:
            flow_axis = candidate
    source_up = "+z" if flow_axis != "z" else "+y"
    preview = _cached_case_mesh_preview(case_path, body_path, flow_axis, source_up)
    streamlines = parse_streamlines(case_path, preview, flow_axis)
    speed_mps = float(flow.get("speed_mps") or 0.0) if isinstance(flow, dict) else 0.0
    density = float(flow.get("air_density_kg_m3") or 1.225) if isinstance(flow, dict) else 1.225
    reference = case_payload.get("aerodynamic_reference")
    reference_area = None
    if isinstance(reference, dict):
        try:
            reference_area = float(reference.get("area_m2") or 0.0) or None
        except (TypeError, ValueError):
            reference_area = None
    surface_pressure = parse_surface_pressure(
        case_path,
        preview,
        flow_axis,
        speed_mps,
        density,
        reference_area_m2=reference_area,
    )
    return {
        "geometryModelPath": str(body_path),
        "geometryPreview": preview,
        "solverStreamlines": streamlines,
        "surfacePressure": surface_pressure,
    }


def _cached_case_mesh_preview(
    case_path: Path,
    body_path: Path,
    flow_axis: str,
    source_up: str,
) -> dict[str, object]:
    stat = body_path.stat()
    signature = {
        "bodySize": stat.st_size,
        "bodyMtimeNs": stat.st_mtime_ns,
        "flowAxis": flow_axis,
        "triangleLimit": CASE_PREVIEW_TRIANGLE_LIMIT,
    }
    cache_path = case_path / "geometry-preview.json"
    cached = _read_json_object(cache_path)
    if cached.get("signature") == signature and isinstance(cached.get("preview"), dict):
        return cached["preview"]  # type: ignore[return-value]

    if flow_axis == "x":
        preview = mesh_preview(body_path, max_triangles=CASE_PREVIEW_TRIANGLE_LIMIT)
    else:
        preview = mesh_preview(
            body_path,
            max_triangles=CASE_PREVIEW_TRIANGLE_LIMIT,
            source_flow_direction=f"+{flow_axis}",
            source_up_direction=source_up,
            target_flow_axis="x",
        )
    try:
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps({"signature": signature, "preview": preview}, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    except OSError:
        pass
    return preview


def parse_streamlines(
    case_path: Path,
    geometry_preview: dict[str, object],
    flow_axis: str,
    max_lines: int = 220,
    max_points_per_line: int = 500,
) -> dict[str, object] | None:
    post_dir = case_path / "postProcessing" / "streamlines"
    if not post_dir.exists():
        return None
    candidates = sorted(
        (path for path in post_dir.rglob("*.vtk") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    path = candidates[0]
    if path.stat().st_size > 128 * 1024 * 1024:
        return {"file": str(path), "error": "Streamline VTK exceeds the 128 MB browser limit."}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not any("ASCII" in line.upper() for line in lines[:10]):
        return {"file": str(path), "error": "Only ASCII legacy VTK streamlines are supported."}

    points_header = _vtk_header(lines, "POINTS")
    line_header = _vtk_header(lines, "LINES")
    if not points_header or not line_header:
        return {"file": str(path), "error": "VTK points or line connectivity is missing."}
    point_line, point_parts = points_header
    line_line, line_parts = line_header
    point_count = int(point_parts[1])
    line_count = int(line_parts[1])
    connectivity_count = int(line_parts[2])
    point_values = _vtk_values(lines, point_line + 1, point_count * 3, float)
    connectivity = _vtk_values(lines, line_line + 1, connectivity_count, int)
    if len(point_values) != point_count * 3 or len(connectivity) != connectivity_count:
        return {"file": str(path), "error": "VTK streamline arrays are incomplete."}

    raw_points = [tuple(point_values[index : index + 3]) for index in range(0, len(point_values), 3)]
    velocity = _vtk_field(lines, "UMean", 3, point_count) or _vtk_field(lines, "U", 3, point_count)
    pressure = _vtk_field(lines, "pMean", 1, point_count) or _vtk_field(lines, "p", 1, point_count)
    time_averaged = _vtk_field(lines, "UMean", 3, point_count) is not None
    center_obj = geometry_preview.get("normalizedCenter")
    scale_obj = geometry_preview.get("normalizedScale")
    center = center_obj if isinstance(center_obj, list) and len(center_obj) == 3 else [0.0, 0.0, 0.0]
    scale = float(scale_obj) if isinstance(scale_obj, (int, float)) else 1.0

    canonical_points = []
    for point in raw_points:
        canonical = _canonical_solver_point(point, flow_axis)
        canonical_points.append(tuple((canonical[index] - float(center[index])) * scale for index in range(3)))

    speeds = None
    if velocity and len(velocity) == point_count * 3:
        speeds = [
            math.sqrt(sum(value * value for value in velocity[index : index + 3]))
            for index in range(0, len(velocity), 3)
        ]
    pressures = pressure if pressure and len(pressure) == point_count else None

    paths: list[list[list[float]]] = []
    cursor = 0
    for _ in range(line_count):
        if cursor >= len(connectivity):
            break
        count = int(connectivity[cursor])
        indices = [int(value) for value in connectivity[cursor + 1 : cursor + 1 + count]]
        cursor += count + 1
        if len(indices) < 2:
            continue
        step = max(1, math.ceil(len(indices) / max_points_per_line))
        sampled_indices = indices[::step]
        if sampled_indices[-1] != indices[-1]:
            sampled_indices.append(indices[-1])
        path_points = []
        for index in sampled_indices:
            if index < 0 or index >= len(canonical_points):
                continue
            point = canonical_points[index]
            path_points.append(
                [
                    round(point[0], 6),
                    round(point[1], 6),
                    round(point[2], 6),
                    round(speeds[index], 6) if speeds else 0.0,
                    round(pressures[index], 6) if pressures else 0.0,
                ]
            )
        if len(path_points) >= 2:
            paths.append(path_points)
        if len(paths) >= max_lines:
            break

    speed_values = [point[3] for path in paths for point in path]
    pressure_values = [point[4] for path in paths for point in path] if pressures else []
    return {
        "file": str(path),
        "lineCount": len(paths),
        "pointCount": sum(len(path_points) for path_points in paths),
        "hasPressure": pressures is not None,
        "timeAveraged": time_averaged,
        "speedRange": [min(speed_values), max(speed_values)] if speed_values else None,
        "pressureRange": [min(pressure_values), max(pressure_values)] if pressure_values else None,
        "lines": paths,
    }


def parse_surface_pressure(
    case_path: Path,
    geometry_preview: dict[str, object],
    flow_axis: str,
    speed_mps: float,
    density_kg_m3: float = 1.225,
    max_triangles: int = 180_000,
    reference_area_m2: float | None = None,
) -> dict[str, object] | None:
    post_dir = case_path / "postProcessing" / "bodyPressure"
    if not post_dir.exists():
        return None
    candidates = sorted(
        (path for path in post_dir.rglob("*.vtk") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    path = candidates[0]
    if path.stat().st_size > 256 * 1024 * 1024:
        return {"file": str(path), "error": "Body-pressure VTK exceeds the 256 MB parser limit."}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not any("ASCII" in line.upper() for line in lines[:10]):
        return {"file": str(path), "error": "Only ASCII legacy VTK body-pressure surfaces are supported."}

    points_header = _vtk_header(lines, "POINTS")
    polygons_header = _vtk_header(lines, "POLYGONS")
    if not points_header or not polygons_header:
        return {"file": str(path), "error": "VTK body-pressure points or polygons are missing."}
    point_line, point_parts = points_header
    polygon_line, polygon_parts = polygons_header
    try:
        point_count = int(point_parts[1])
        polygon_count = int(polygon_parts[1])
        connectivity_count = int(polygon_parts[2])
    except (IndexError, ValueError):
        return {"file": str(path), "error": "VTK body-pressure headers are invalid."}
    point_values = _vtk_values(lines, point_line + 1, point_count * 3, float)
    connectivity = _vtk_values(lines, polygon_line + 1, connectivity_count, int)
    if len(point_values) != point_count * 3 or len(connectivity) != connectivity_count:
        return {"file": str(path), "error": "VTK body-pressure arrays are incomplete."}

    raw_points = [tuple(point_values[index : index + 3]) for index in range(0, len(point_values), 3)]
    polygons: list[list[int]] = []
    polygon_source_indices: list[int] = []
    cursor = 0
    for polygon_source_index in range(polygon_count):
        if cursor >= len(connectivity):
            break
        count = int(connectivity[cursor])
        indices = [int(value) for value in connectivity[cursor + 1 : cursor + 1 + count]]
        cursor += count + 1
        if len(indices) >= 3 and all(0 <= index < point_count for index in indices):
            polygons.append(indices)
            polygon_source_indices.append(polygon_source_index)
    if not polygons:
        return {"file": str(path), "error": "VTK body-pressure surface has no usable polygons."}

    point_pressure = _vtk_field(lines, "pMean", 1, point_count) or _vtk_field(lines, "p", 1, point_count)
    cell_pressure = _vtk_field(lines, "pMean", 1, polygon_count) or _vtk_field(lines, "p", 1, polygon_count)
    time_averaged = _vtk_field(lines, "pMean", 1, point_count) is not None or _vtk_field(
        lines, "pMean", 1, polygon_count
    ) is not None
    if point_pressure is None and cell_pressure is None:
        return {"file": str(path), "error": "VTK body-pressure field is missing."}
    if point_pressure is not None:
        pressure = point_pressure
        pressure_location = "point"
    else:
        sums = [0.0] * point_count
        counts = [0] * point_count
        assert cell_pressure is not None
        for polygon_index, indices in enumerate(polygons):
            value = float(cell_pressure[polygon_source_indices[polygon_index]])
            for index in indices:
                sums[index] += value
                counts[index] += 1
        pressure = [sums[index] / counts[index] if counts[index] else 0.0 for index in range(point_count)]
        pressure_location = "cell-averaged-to-point"

    point_shear = _vtk_field(lines, "wallShearStressMean", 3, point_count) or _vtk_field(
        lines, "wallShearStress", 3, point_count
    )
    cell_shear = _vtk_field(lines, "wallShearStressMean", 3, polygon_count) or _vtk_field(
        lines, "wallShearStress", 3, polygon_count
    )
    wall_shear: list[tuple[float, float, float]] | None = None
    wall_shear_location: str | None = None
    if point_shear is not None and len(point_shear) == point_count * 3:
        wall_shear = [
            tuple(float(value) for value in point_shear[index : index + 3])
            for index in range(0, len(point_shear), 3)
        ]  # type: ignore[list-item]
        wall_shear_location = "point"
    elif cell_shear is not None and len(cell_shear) == polygon_count * 3:
        sums = [[0.0, 0.0, 0.0] for _ in range(point_count)]
        counts = [0] * point_count
        for polygon_index, indices in enumerate(polygons):
            source_index = polygon_source_indices[polygon_index]
            vector = cell_shear[source_index * 3 : source_index * 3 + 3]
            for index in indices:
                for component in range(3):
                    sums[index][component] += float(vector[component])
                counts[index] += 1
        wall_shear = [
            tuple(value / counts[index] if counts[index] else 0.0 for value in sums[index])
            for index in range(point_count)
        ]  # type: ignore[list-item]
        wall_shear_location = "cell-averaged-to-point"

    center_obj = geometry_preview.get("normalizedCenter")
    scale_obj = geometry_preview.get("normalizedScale")
    center = center_obj if isinstance(center_obj, list) and len(center_obj) == 3 else [0.0, 0.0, 0.0]
    scale = float(scale_obj) if isinstance(scale_obj, (int, float)) else 1.0
    canonical_points = [_canonical_solver_point(point, flow_axis) for point in raw_points]
    points = []
    for canonical in canonical_points:
        points.append(tuple((canonical[index] - float(center[index])) * scale for index in range(3)))

    denominator = 0.5 * max(float(speed_mps), 1e-6) ** 2
    cp_values = [float(value) / denominator for value in pressure]
    skin_drag_values = None
    if wall_shear is not None:
        skin_drag_values = [
            _canonical_solver_point(vector, flow_axis)[0] / denominator
            for vector in wall_shear
        ]
    triangles: list[tuple[int, int, int]] = []
    triangle_polygon_indices: list[int] = []
    for polygon_index, indices in enumerate(polygons):
        for offset in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[offset], indices[offset + 1]))
            triangle_polygon_indices.append(polygon_index)
    source_triangle_count = len(triangles)
    raw_drag_summary, raw_triangle_pressure_drag, raw_triangle_skin_drag = _integrated_triangle_drag(
        canonical_points,
        triangles,
        triangle_polygon_indices,
        polygon_source_indices,
        point_pressure,
        cell_pressure,
        point_shear,
        cell_shear,
        denominator,
        flow_axis,
        reference_area_m2,
    )
    decimated = False
    if len(triangles) > max_triangles:
        value_fields = [cp_values]
        if skin_drag_values is not None:
            value_fields.append(skin_drag_values)
        points, value_fields, triangles = _cluster_pressure_surface(
            points,
            value_fields,
            triangles,
            max_triangles,
        )
        cp_values = value_fields[0]
        skin_drag_values = value_fields[1] if skin_drag_values is not None else None
        decimated = True
    if not triangles:
        return {"file": str(path), "error": "Body-pressure visualization decimation removed every triangle."}

    cp_min = min(cp_values)
    cp_max = max(cp_values)
    robust_limit = max(
        abs(_percentile(cp_values, 0.02)),
        abs(_percentile(cp_values, 0.98)),
        0.25,
    )
    drag_values, displayed_triangle_pressure_drag, drag_summary = _pressure_drag_map(
        points,
        cp_values,
        triangles,
        normalization_scale=scale,
        reference_area_m2=reference_area_m2,
    )
    drag_summary.update(raw_drag_summary)
    has_wall_shear = skin_drag_values is not None
    if skin_drag_values is None:
        skin_drag_values = [0.0] * len(points)
    total_drag_values = [
        drag_values[index] + skin_drag_values[index]
        for index in range(len(points))
    ]
    if decimated:
        triangle_pressure_drag_values = displayed_triangle_pressure_drag
        triangle_skin_drag_values = [
            sum(float(skin_drag_values[index]) for index in triangle) / 3.0
            for triangle in triangles
        ]
    else:
        triangle_pressure_drag_values = raw_triangle_pressure_drag
        triangle_skin_drag_values = raw_triangle_skin_drag
    triangle_total_drag_values = [
        triangle_pressure_drag_values[index] + triangle_skin_drag_values[index]
        for index in range(len(triangles))
    ]
    skin_drag_coefficient = raw_drag_summary.get("skinFrictionDragCoefficient") if has_wall_shear else None
    pressure_drag_coefficient = drag_summary.get("pressureDragCoefficient")
    total_drag_coefficient = raw_drag_summary.get("totalDragCoefficient") if has_wall_shear else None
    total_drag_min = min(triangle_total_drag_values, default=0.0)
    total_drag_max = max(triangle_total_drag_values, default=0.0)
    total_drag_limit = max(
        abs(_percentile(triangle_total_drag_values, 0.02)),
        abs(_percentile(triangle_total_drag_values, 0.98)),
        0.05,
    )
    pressure_pa = [float(value) * density_kg_m3 for value in pressure]
    return {
        "file": str(path),
        "hasPressure": True,
        "timeAveraged": time_averaged,
        "pressureLocation": pressure_location,
        "pointCount": len(points),
        "triangleCount": len(triangles),
        "sourceTriangleCount": source_triangle_count,
        "decimatedForBrowser": decimated,
        "dynamicPressurePa": round(0.5 * density_kg_m3 * float(speed_mps) ** 2, 6),
        "pressurePaRange": [round(min(pressure_pa), 6), round(max(pressure_pa), 6)],
        "cpRange": [round(cp_min, 6), round(cp_max, 6)],
        "cpDisplayRange": [round(-robust_limit, 6), round(robust_limit, 6)],
        "hasPressureDrag": True,
        "hasWallShear": has_wall_shear,
        "wallShearLocation": wall_shear_location,
        "skinFrictionDragDensityRange": [
            round(min(skin_drag_values, default=0.0), 6),
            round(max(skin_drag_values, default=0.0), 6),
        ] if has_wall_shear else None,
        "skinFrictionDragCoefficient": (
            round(skin_drag_coefficient, 6) if skin_drag_coefficient is not None else None
        ),
        "totalDragDensityRange": [round(total_drag_min, 6), round(total_drag_max, 6)],
        "totalDragDisplayRange": [round(-total_drag_limit, 6), round(total_drag_limit, 6)],
        "totalDragCoefficient": (
            round(total_drag_coefficient, 6) if total_drag_coefficient is not None else None
        ),
        "wallShearDefinition": "Flow-direction wallShearStress divided by dynamic pressure; positive adds viscous drag.",
        **drag_summary,
        "points": [
            [
                round(point[0], 6),
                round(point[1], 6),
                round(point[2], 6),
                round(cp_values[index], 6),
                round(drag_values[index], 6),
                round(skin_drag_values[index], 6),
                round(total_drag_values[index], 6),
            ]
            for index, point in enumerate(points)
        ],
        "triangles": [list(triangle) for triangle in triangles],
        "trianglePressureDragValues": [round(value, 6) for value in triangle_pressure_drag_values],
        "triangleTotalDragValues": [round(value, 6) for value in triangle_total_drag_values],
    }


def _pressure_drag_map(
    points: list[tuple[float, float, float]],
    cp_values: list[float],
    triangles: list[tuple[int, int, int]],
    normalization_scale: float,
    reference_area_m2: float | None,
) -> tuple[list[float], list[float], dict[str, object]]:
    weighted_drag = [0.0] * len(points)
    vertex_area = [0.0] * len(points)
    triangle_drag_values: list[float] = []
    pressure_drag_area = 0.0
    positive_drag_area = 0.0
    offset_drag_area = 0.0
    physical_area_factor = 1.0 / max(float(normalization_scale) ** 2, 1e-12)

    for triangle in triangles:
        a, b, c = (points[index] for index in triangle)
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        cross_length = math.sqrt(sum(value * value for value in cross))
        if cross_length <= 1e-12:
            triangle_drag_values.append(0.0)
            continue
        display_area = 0.5 * cross_length
        flow_normal = cross[0] / cross_length
        mean_cp = sum(float(cp_values[index]) for index in triangle) / 3.0
        drag_density = mean_cp * flow_normal
        triangle_drag_values.append(drag_density)
        physical_contribution = drag_density * display_area * physical_area_factor
        pressure_drag_area += physical_contribution
        if physical_contribution >= 0:
            positive_drag_area += physical_contribution
        else:
            offset_drag_area += physical_contribution
        for index in triangle:
            weighted_drag[index] += drag_density * display_area
            vertex_area[index] += display_area

    drag_values = [
        weighted_drag[index] / vertex_area[index] if vertex_area[index] > 0 else 0.0
        for index in range(len(points))
    ]
    drag_min = min(triangle_drag_values, default=0.0)
    drag_max = max(triangle_drag_values, default=0.0)
    robust_limit = max(
        abs(_percentile(triangle_drag_values, 0.02)),
        abs(_percentile(triangle_drag_values, 0.98)),
        0.05,
    )
    reference_area = float(reference_area_m2 or 0.0)
    coefficient = pressure_drag_area / reference_area if reference_area > 0 else None
    positive_coefficient = positive_drag_area / reference_area if reference_area > 0 else None
    offset_coefficient = offset_drag_area / reference_area if reference_area > 0 else None
    return drag_values, triangle_drag_values, {
        "pressureDragDensityRange": [round(drag_min, 6), round(drag_max, 6)],
        "pressureDragDisplayRange": [round(-robust_limit, 6), round(robust_limit, 6)],
        "pressureDragCoefficient": round(coefficient, 6) if coefficient is not None else None,
        "positivePressureDragCoefficient": (
            round(positive_coefficient, 6) if positive_coefficient is not None else None
        ),
        "offsetPressureDragCoefficient": (
            round(offset_coefficient, 6) if offset_coefficient is not None else None
        ),
        "pressureDragReferenceAreaM2": round(reference_area, 6) if reference_area > 0 else None,
        "pressureDragDefinition": "Cp times inward patch-normal component along the flow axis; positive adds drag.",
    }


def _integrated_triangle_drag(
    points: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    triangle_polygon_indices: list[int],
    polygon_source_indices: list[int],
    point_pressure: list[float] | None,
    cell_pressure: list[float] | None,
    point_shear: list[float] | None,
    cell_shear: list[float] | None,
    dynamic_pressure_kinematic: float,
    flow_axis: str,
    reference_area_m2: float | None,
) -> tuple[dict[str, object], list[float], list[float]]:
    reference_area = float(reference_area_m2 or 0.0)
    body_min_x = min((point[0] for point in points), default=0.0)
    body_max_x = max((point[0] for point in points), default=body_min_x)
    body_length = max(body_max_x - body_min_x, 1e-12)
    pressure_integral = 0.0
    positive_pressure_integral = 0.0
    offset_pressure_integral = 0.0
    skin_integral = 0.0
    positive_total_integral = 0.0
    offset_total_integral = 0.0
    region_pressure_integrals = [0.0, 0.0, 0.0]
    region_skin_integrals = [0.0, 0.0, 0.0]
    region_positive_total_integrals = [0.0, 0.0, 0.0]
    has_shear = point_shear is not None or cell_shear is not None
    triangle_pressure_drag_values: list[float] = []
    triangle_skin_drag_values: list[float] = []
    for triangle_index, triangle in enumerate(triangles):
        a, b, c = (points[index] for index in triangle)
        ab = tuple(b[axis] - a[axis] for axis in range(3))
        ac = tuple(c[axis] - a[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        cross_length = math.sqrt(sum(value * value for value in cross))
        if cross_length <= 1e-12:
            triangle_pressure_drag_values.append(0.0)
            triangle_skin_drag_values.append(0.0)
            continue
        area = 0.5 * cross_length
        flow_normal = cross[0] / cross_length
        centroid_x = sum(points[index][0] for index in triangle) / 3.0
        body_fraction = max(0.0, min(1.0, (centroid_x - body_min_x) / body_length))
        region_index = min(2, int(body_fraction * 3.0))
        polygon_index = triangle_polygon_indices[triangle_index]
        source_polygon_index = polygon_source_indices[polygon_index]
        if point_pressure is not None:
            cp = sum(float(point_pressure[index]) for index in triangle) / (
                3.0 * dynamic_pressure_kinematic
            )
        else:
            assert cell_pressure is not None
            cp = float(cell_pressure[source_polygon_index]) / dynamic_pressure_kinematic
        pressure_contribution = cp * flow_normal * area
        triangle_pressure_drag_values.append(cp * flow_normal)
        pressure_integral += pressure_contribution
        region_pressure_integrals[region_index] += pressure_contribution
        if pressure_contribution >= 0:
            positive_pressure_integral += pressure_contribution
        else:
            offset_pressure_integral += pressure_contribution

        if point_shear is not None:
            shear = sum(
                _canonical_solver_point(
                    tuple(float(value) for value in point_shear[index * 3 : index * 3 + 3]),
                    flow_axis,
                )[0]
                for index in triangle
            ) / (3.0 * dynamic_pressure_kinematic)
            skin_integral += shear * area
        elif cell_shear is not None:
            vector = tuple(
                float(value)
                for value in cell_shear[source_polygon_index * 3 : source_polygon_index * 3 + 3]
            )
            shear = _canonical_solver_point(vector, flow_axis)[0] / dynamic_pressure_kinematic
            skin_integral += shear * area
        else:
            shear = 0.0
        skin_contribution = shear * area
        region_skin_integrals[region_index] += skin_contribution
        total_contribution = pressure_contribution + skin_contribution
        if total_contribution >= 0:
            positive_total_integral += total_contribution
            region_positive_total_integrals[region_index] += total_contribution
        else:
            offset_total_integral += total_contribution
        triangle_skin_drag_values.append(shear)

    if reference_area <= 0:
        return {}, triangle_pressure_drag_values, triangle_skin_drag_values

    pressure_coefficient = pressure_integral / reference_area
    result: dict[str, object] = {
        "pressureDragCoefficient": round(pressure_coefficient, 6),
        "positivePressureDragCoefficient": round(positive_pressure_integral / reference_area, 6),
        "offsetPressureDragCoefficient": round(offset_pressure_integral / reference_area, 6),
        "pressureDragReferenceAreaM2": round(reference_area, 6),
        "dragIntegrationSource": "original face values before browser decimation",
    }
    if has_shear:
        skin_coefficient = skin_integral / reference_area
        result.update(
            {
                "skinFrictionDragCoefficient": round(skin_coefficient, 6),
                "totalDragCoefficient": round(pressure_coefficient + skin_coefficient, 6),
            }
        )
    region_ids = ("front", "middle", "rear")
    region_labels = ("Front third", "Middle third", "Rear third")
    positive_total_coefficient = positive_total_integral / reference_area
    result.update(
        {
            "positiveTotalDragCoefficient": round(positive_total_coefficient, 6),
            "offsetTotalDragCoefficient": round(offset_total_integral / reference_area, 6),
            "dragRegions": [
                {
                    "id": region_ids[index],
                    "label": region_labels[index],
                    "pressureDragCoefficient": round(
                        region_pressure_integrals[index] / reference_area, 6
                    ),
                    "skinFrictionDragCoefficient": (
                        round(region_skin_integrals[index] / reference_area, 6)
                        if has_shear
                        else None
                    ),
                    "totalDragCoefficient": round(
                        (
                            region_pressure_integrals[index]
                            + region_skin_integrals[index]
                        )
                        / reference_area,
                        6,
                    ),
                    "positiveTotalDragCoefficient": round(
                        region_positive_total_integrals[index] / reference_area, 6
                    ),
                    "positiveDragSharePercent": round(
                        region_positive_total_integrals[index]
                        / max(positive_total_integral, 1e-12)
                        * 100.0,
                        2,
                    ),
                }
                for index in range(3)
            ],
            "dragHotspotRegion": (
                region_ids[
                    max(range(3), key=lambda index: region_positive_total_integrals[index])
                ]
                if positive_total_integral > 1e-12
                else None
            ),
            "dragRegionDefinition": (
                "Front, middle, and rear thirds along the wind axis; shares use positive "
                "local pressure plus skin-friction drag from original solver faces."
            ),
        }
    )
    return result, triangle_pressure_drag_values, triangle_skin_drag_values


def _cluster_pressure_surface(
    points: list[tuple[float, float, float]],
    value_fields: list[list[float]],
    triangles: list[tuple[int, int, int]],
    target_triangles: int,
) -> tuple[list[tuple[float, float, float]], list[list[float]], list[tuple[int, int, int]]]:
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    diagonal = math.sqrt(sum((maximum[axis] - minimum[axis]) ** 2 for axis in range(3)))
    spacing = max(diagonal / math.sqrt(max(target_triangles, 1)) * 0.35, 1e-9)
    best = (points, value_fields, triangles)
    for _ in range(8):
        clusters: dict[tuple[int, int, int], int] = {}
        sums: list[list[float]] = []
        counts: list[int] = []
        remap: list[int] = []
        for point_index, point in enumerate(points):
            key = tuple(math.floor((point[axis] - minimum[axis]) / spacing) for axis in range(3))
            cluster = clusters.get(key)
            if cluster is None:
                cluster = len(sums)
                clusters[key] = cluster
                sums.append([0.0] * (3 + len(value_fields)))
                counts.append(0)
            sums[cluster][0] += point[0]
            sums[cluster][1] += point[1]
            sums[cluster][2] += point[2]
            for field_index, values in enumerate(value_fields):
                sums[cluster][3 + field_index] += values[point_index]
            counts[cluster] += 1
            remap.append(cluster)
        clustered_points = [
            (total[0] / counts[index], total[1] / counts[index], total[2] / counts[index])
            for index, total in enumerate(sums)
        ]
        clustered_fields = [
            [total[3 + field_index] / counts[index] for index, total in enumerate(sums)]
            for field_index in range(len(value_fields))
        ]
        clustered_triangles: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for triangle in triangles:
            mapped = (remap[triangle[0]], remap[triangle[1]], remap[triangle[2]])
            if len(set(mapped)) < 3:
                continue
            duplicate_key = tuple(sorted(mapped))
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            clustered_triangles.append(mapped)
        if clustered_triangles:
            best = (clustered_points, clustered_fields, clustered_triangles)
        if 0 < len(clustered_triangles) <= target_triangles:
            return best
        spacing *= max(1.2, math.sqrt(len(clustered_triangles) / max(target_triangles, 1)) * 1.05)
    return best


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = min(max(quantile, 0.0), 1.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    amount = position - lower
    return ordered[lower] * (1.0 - amount) + ordered[upper] * amount


def _vtk_header(lines: list[str], keyword: str) -> tuple[int, list[str]] | None:
    for index, line in enumerate(lines):
        parts = line.strip().split()
        if parts and parts[0].upper() == keyword:
            return index, parts
    return None


def _vtk_values(lines: list[str], start: int, count: int, cast: type) -> list:
    values = []
    for line in lines[start:]:
        for token in line.strip().split():
            try:
                values.append(cast(token))
            except ValueError:
                return values
            if len(values) == count:
                return values
    return values


def _vtk_field(lines: list[str], name: str, components: int, count: int) -> list[float] | None:
    for index, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == name:
            try:
                if int(parts[1]) == components and int(parts[2]) == count:
                    return _vtk_values(lines, index + 1, components * count, float)
            except ValueError:
                continue
        if len(parts) >= 3 and parts[0].upper() == "VECTORS" and parts[1] == name and components == 3:
            return _vtk_values(lines, index + 1, components * count, float)
        if len(parts) >= 3 and parts[0].upper() == "SCALARS" and parts[1] == name and components == 1:
            start = index + 1
            if start < len(lines) and lines[start].strip().upper().startswith("LOOKUP_TABLE"):
                start += 1
            return _vtk_values(lines, start, count, float)
    return None


def _canonical_solver_point(point: tuple[float, ...], flow_axis: str) -> tuple[float, float, float]:
    x, y, z = point
    if flow_axis == "y":
        return (y, -x, z)
    if flow_axis == "z":
        return (z, x, y)
    return (x, y, z)


def _mesh_quality_assessment(
    mesh_returncode: int | None,
    mesh_quality: dict[str, object] | None,
    layer_coverage: dict[str, object] | None,
    wall_resolution: object,
    geometry_fidelity: object,
    mesh_surface_fidelity: object,
    geometry_validation: object,
    mesh_resolution: object,
    ground_setup: object,
    placement: object,
) -> dict[str, object]:
    completed = mesh_returncode is not None
    wall_layers = 0
    if isinstance(wall_resolution, dict):
        try:
            wall_layers = int(wall_resolution.get("surface_layers") or 0)
        except (TypeError, ValueError):
            wall_layers = 0
    fidelity_verified = isinstance(geometry_fidelity, dict) and bool(geometry_fidelity.get("verified"))
    body_fidelity_verified = (
        isinstance(mesh_surface_fidelity, dict) and bool(mesh_surface_fidelity.get("verified"))
    )
    dimensions_verified = isinstance(geometry_validation, dict) and bool(geometry_validation.get("verified"))
    mesh_quality_name = str(mesh_resolution.get("quality") or "") if isinstance(mesh_resolution, dict) else ""
    smallest_feature = (
        _finite_number(mesh_resolution.get("smallest_aero_feature_m"))
        if isinstance(mesh_resolution, dict)
        else None
    )
    cells_across_feature = (
        _finite_number(mesh_resolution.get("estimated_cells_across_feature"))
        if isinstance(mesh_resolution, dict)
        else None
    )
    feature_target_verified = bool(
        smallest_feature is not None
        and smallest_feature > 0
        and cells_across_feature is not None
        and cells_across_feature >= 4.0
        and bool(mesh_resolution.get("supported"))
    ) if isinstance(mesh_resolution, dict) else False
    checks = [
        _quality_check(
            "Geometry fidelity",
            fidelity_verified,
            "The solver STL is original geometry or has a matching accepted repair record.",
            "The solver STL repair fidelity is missing or unverified.",
        ),
        _quality_check(
            "Meshed body fidelity",
            _completed_gate(
                mesh_surface_fidelity if isinstance(mesh_surface_fidelity, dict) else None,
                "verified",
                completed,
            ),
            "The snapped OpenFOAM body patch preserves the transformed solver STL.",
            "The snapped body patch is missing, under-resolved, or does not preserve the solver STL.",
        ),
        _quality_check(
            "Measured dimensions",
            dimensions_verified,
            "Scaled length, width, and height agree with physical measurements within 2%.",
            "Measured length, width, and height are missing or do not agree with the scaled STL.",
        ),
        _quality_check(
            "Aero feature resolution",
            True if mesh_quality_name == "draft" else feature_target_verified,
            (
                "Draft setup does not claim pressure resolution of small aero features."
                if mesh_quality_name == "draft"
                else "The declared smallest aerodynamic feature has at least four estimated surface cells."
            ),
            "Set the smallest aerodynamic feature that matters and resolve it with at least four surface cells.",
        ),
        _quality_check(
            "Mesh process",
            None if mesh_returncode is None else mesh_returncode == 0,
            "The OpenFOAM mesh pipeline exited successfully.",
            "The mesh pipeline has not completed successfully.",
        ),
        _quality_check(
            "Mesh quality",
            _completed_gate(mesh_quality, "passed", completed),
            "The baseline OpenFOAM mesh checks passed; extended diagnostics remain available separately.",
            "The baseline OpenFOAM mesh checks did not pass.",
        ),
    ]
    ground_enabled = isinstance(ground_setup, dict) and bool(ground_setup.get("enabled"))
    if ground_enabled:
        clearance = _finite_number(ground_setup.get("clearance_m"))
        road_elevation = _finite_number(ground_setup.get("road_elevation_m"))
        lowest_model = _finite_number(ground_setup.get("lowest_model_z_m"))
        placement_recorded = isinstance(placement, dict) and bool(placement.get("verified"))
        placement_verified = bool(
            placement_recorded
            and clearance is not None
            and clearance >= 0
            and road_elevation is not None
            and lowest_model is not None
            and abs(lowest_model - road_elevation - clearance) <= 1e-6
        )
        checks.insert(
            2,
            _quality_check(
                "Road placement",
                placement_verified,
                "The solver STL lowest point matches the recorded road clearance.",
                "Road clearance and the transformed solver STL placement are missing or inconsistent.",
            ),
        )
    if wall_layers > 0:
        checks.append(
            _quality_check(
                "Boundary-layer coverage",
                _completed_gate(layer_coverage, "passed", completed),
                "The body received sufficient prism-layer coverage.",
                "Boundary layers are missing or cover too little of the body surface.",
            )
        )
    reusable = bool(
        mesh_returncode == 0
        and fidelity_verified
        and body_fidelity_verified
        and isinstance(mesh_quality, dict)
        and mesh_quality.get("passed")
    )
    trusted = all(check["status"] == "pass" for check in checks)
    failed = any(check["status"] == "fail" for check in checks)
    return {
        "status": "verified" if trusted else "review" if reusable else "failed" if failed else "incomplete",
        "trusted": trusted,
        "reusable": reusable,
        "checks": checks,
    }


def _quality_assessment(
    solver_returncode: int | None,
    mesh_quality: dict[str, object] | None,
    layer_coverage: dict[str, object] | None,
    residuals: dict[str, object] | None,
    transient_state: dict[str, object] | None,
    simulation_quality: object,
    force_coeffs: dict[str, object] | None,
    y_plus: dict[str, object] | None,
    wall_resolution: object,
    geometry_fidelity: object,
    mesh_surface_fidelity: object,
    geometry_validation: object,
    mesh_resolution: object,
    ground_setup: object,
    placement: object,
) -> dict[str, object]:
    completed = solver_returncode is not None
    transient = (
        isinstance(simulation_quality, dict)
        and simulation_quality.get("simulation_mode") == "transient"
    ) or isinstance(transient_state, dict) or (
        isinstance(residuals, dict) and residuals.get("mode") == "transient-divergence"
    )
    wall_layers = 0
    if isinstance(wall_resolution, dict):
        try:
            wall_layers = int(wall_resolution.get("surface_layers") or 0)
        except (TypeError, ValueError):
            wall_layers = 0
    fidelity_verified = isinstance(geometry_fidelity, dict) and bool(geometry_fidelity.get("verified"))
    dimensions_verified = isinstance(geometry_validation, dict) and bool(geometry_validation.get("verified"))
    mesh_quality_name = str(mesh_resolution.get("quality") or "") if isinstance(mesh_resolution, dict) else ""
    smallest_feature = (
        _finite_number(mesh_resolution.get("smallest_aero_feature_m"))
        if isinstance(mesh_resolution, dict)
        else None
    )
    cells_across_feature = (
        _finite_number(mesh_resolution.get("estimated_cells_across_feature"))
        if isinstance(mesh_resolution, dict)
        else None
    )
    feature_target_verified = bool(
        smallest_feature is not None
        and smallest_feature > 0
        and cells_across_feature is not None
        and cells_across_feature >= 4.0
        and bool(mesh_resolution.get("supported"))
    ) if isinstance(mesh_resolution, dict) else False
    checks = [
        _quality_check(
            "Geometry fidelity",
            fidelity_verified,
            "The solver STL is original geometry or has a matching accepted repair record.",
            "The solver STL repair fidelity is missing or unverified.",
        ),
        _quality_check(
            "Meshed body fidelity",
            _completed_gate(
                mesh_surface_fidelity if isinstance(mesh_surface_fidelity, dict) else None,
                "verified",
                completed,
            ),
            "The solved OpenFOAM body patch preserves the transformed solver STL.",
            "The solved body patch is missing, under-resolved, or does not preserve the solver STL.",
        ),
        _quality_check(
            "Measured dimensions",
            dimensions_verified,
            "Scaled length, width, and height agree with physical measurements within 2%.",
            "Measured length, width, and height are missing or do not agree with the scaled STL.",
        ),
        _quality_check(
            "Aero feature resolution",
            True if mesh_quality_name == "draft" else feature_target_verified,
            (
                "Draft setup does not claim pressure resolution of small aero features."
                if mesh_quality_name == "draft"
                else "The declared smallest aerodynamic feature has at least four estimated surface cells."
            ),
            "Set the smallest aerodynamic feature that matters and resolve it with at least four surface cells.",
        ),
        _quality_check(
            "Solver process",
            None if solver_returncode is None else solver_returncode == 0,
            "The complete OpenFOAM pipeline exited successfully.",
            "The solver has not completed successfully.",
        ),
        _quality_check(
            "Mesh quality",
            _completed_gate(mesh_quality, "passed", completed),
            "The baseline OpenFOAM mesh checks passed; extended diagnostics remain available separately.",
            "The baseline OpenFOAM mesh checks did not pass.",
        ),
        _quality_check(
            "Residual stability" if transient else "Residual convergence",
            _completed_gate(residuals, "stable", completed),
            (
                "Velocity, pressure, and turbulence residuals stayed below the divergence ceiling."
                if transient
                else "Velocity, pressure, and turbulence residuals reached their gates."
            ),
            (
                "Residuals are missing or exceed the transient divergence ceiling."
                if transient
                else "Residuals are missing or have not converged."
            ),
        ),
        _quality_check(
            "Force stability",
            _completed_gate(force_coeffs, "stable", completed),
            (
                "Time-averaged Cd and Cl are stable over the averaging window."
                if transient
                else "Cd and Cl are stable in the final sample window."
            ),
            "Force coefficients are missing or their mean is still changing.",
        ),
    ]
    if transient:
        checks.extend((
            _quality_check(
                "Courant control",
                _completed_gate(transient_state, "courantControlled", completed),
                "Adaptive time stepping kept the Courant number within its gate.",
                "The transient Courant history is missing or exceeds its gate.",
            ),
            _quality_check(
                "Time averaging",
                _completed_gate(transient_state, "timeAveraged", completed),
                "Mean velocity and pressure fields were written after the averaging period.",
                "The transient run has not produced complete mean velocity and pressure fields.",
            ),
        ))
    ground_enabled = isinstance(ground_setup, dict) and bool(ground_setup.get("enabled"))
    if ground_enabled:
        clearance = _finite_number(ground_setup.get("clearance_m"))
        road_elevation = _finite_number(ground_setup.get("road_elevation_m"))
        lowest_model = _finite_number(ground_setup.get("lowest_model_z_m"))
        placement_recorded = isinstance(placement, dict) and bool(placement.get("verified"))
        placement_verified = bool(
            placement_recorded
            and clearance is not None
            and clearance >= 0
            and road_elevation is not None
            and lowest_model is not None
            and abs(lowest_model - road_elevation - clearance) <= 1e-6
        )
        checks.insert(
            2,
            _quality_check(
                "Road placement",
                placement_verified,
                "The solver STL lowest point matches the recorded road clearance.",
                "Road clearance and the transformed solver STL placement are missing or inconsistent.",
            ),
        )
    if wall_layers > 0:
        checks.extend((
            _quality_check(
                "Boundary-layer coverage",
                _completed_gate(layer_coverage, "passed", completed),
                "The body received sufficient prism-layer coverage.",
                "Boundary layers are missing or cover too little of the body surface.",
            ),
            _quality_check(
                "Wall resolution",
                _completed_gate(y_plus, "passed", completed),
                "Solved body y+ is consistent with the wall-function mesh target.",
                "Body y+ is missing or outside the accepted wall-function range.",
            ),
        ))
    trusted = all(check["status"] == "pass" for check in checks)
    failed = any(check["status"] == "fail" for check in checks)
    return {
        "status": "verified" if trusted else "failed" if failed else "incomplete",
        "trusted": trusted,
        "checks": checks,
    }


def _completed_gate(payload: dict[str, object] | None, key: str, completed: bool) -> bool | None:
    if payload is None:
        return False if completed else None
    return bool(payload.get(key))


def _quality_check(label: str, passed: bool | None, success: str, failure: str) -> dict[str, str]:
    if passed is None:
        return {"label": label, "status": "pending", "detail": failure}
    return {"label": label, "status": "pass" if passed else "fail", "detail": success if passed else failure}


def _residual_threshold(field: str, quality: object = None) -> float:
    name = field.lower()
    if isinstance(quality, dict):
        key = (
            "pressure_residual_control"
            if name in {"p", "p_rgh"}
            else "turbulence_residual_control"
            if name in {"k", "omega"}
            else "velocity_residual_control"
        )
        value = _finite_number(quality.get(key))
        if value is not None and value > 0:
            return value
    if name in {"p", "p_rgh"}:
        return 1e-2
    return 1e-3


def _last_number(text: str, pattern: str, cast: type = float) -> float | int | None:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    try:
        return cast(matches[-1])
    except (TypeError, ValueError):
        return None


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


def _update_case_status(case_path: Path, status: str) -> None:
    case_json_path = case_path / "case.json"
    if not case_json_path.exists():
        return
    payload = json.loads(case_json_path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    case_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
