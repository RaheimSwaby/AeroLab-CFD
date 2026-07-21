from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TextIO

from ..openfoam import QUALITY_PRESETS, configure_decomposition, ensure_case_postprocessing
from .analysis import (
    _latest_body_surface_vtk,
    _tail_text,
    assess_meshed_surface_fidelity,
    case_report,
)
from .backends import (
    _backend_cleanup_command,
    _backend_cleanup_verification_command,
    _run_command,
    _select_backend,
    probe_backend_resources,
    solver_status,
)
from .util import _finite_number, _read_json_object

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
SAFE_CELL_BUDGET_ROUNDING = 50_000


class SolverRunCancelled(RuntimeError):
    """Raised after an explicitly requested run cancellation is made safe."""


@dataclass(frozen=True)
class _RegisteredSolverProcess:
    process: subprocess.Popen[str]
    case_path: Path
    backend: str
    execution_id: str


class SolverRunController:
    """Coordinate cancellation and ownership for one accepted run attempt."""

    def __init__(self, attempt_id: str | None = None) -> None:
        self.attempt_id = attempt_id or uuid.uuid4().hex
        self._cancelled = threading.Event()
        self._stop_complete = threading.Event()
        self._stop_complete.set()
        self._lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._transition_lock = threading.Lock()
        self._terminal = False
        self._processes: dict[subprocess.Popen[str], _RegisteredSolverProcess] = {}
        self._owned_case_paths: frozenset[Path] = frozenset()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancelled.is_set()

    def set_owned_case_paths(self, case_paths: list[Path]) -> None:
        """Record case locks acquired by the accepting server thread."""
        resolved = frozenset(path.resolve() for path in case_paths)
        with self._lock:
            if self._owned_case_paths and self._owned_case_paths != resolved:
                raise RuntimeError("Run ownership paths cannot change after acceptance.")
            self._owned_case_paths = resolved

    def owns_case_paths(self, case_paths: list[Path]) -> bool:
        expected = frozenset(path.resolve() for path in case_paths)
        with self._lock:
            return bool(expected) and expected.issubset(self._owned_case_paths)

    @property
    def owns_multiple_cases(self) -> bool:
        with self._lock:
            return len(self._owned_case_paths) > 1

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested:
            raise SolverRunCancelled("OpenFOAM run stopped by user request.")

    def wait_for_stop_completion(self) -> None:
        """Keep case ownership until synchronous termination verification finishes."""
        if self.cancellation_requested:
            self._stop_complete.wait()

    def register_process(
        self,
        process: subprocess.Popen[str],
        *,
        case_path: Path,
        backend: str,
        execution_id: str,
    ) -> None:
        registration = _RegisteredSolverProcess(
            process=process,
            case_path=case_path.resolve(),
            backend=backend,
            execution_id=execution_id,
        )
        with self._lock:
            self._processes[process] = registration
            cancelled = self._cancelled.is_set()
        if cancelled:
            self.request_stop()
            self.raise_if_cancelled()

    def unregister_process(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.pop(process, None)

    @contextmanager
    def terminal_commit(self, *, mark_terminal: bool = True) -> Iterator[None]:
        """Make completion and cancellation mutually exclusive terminal transitions."""
        with self._transition_lock:
            self.raise_if_cancelled()
            if mark_terminal and self._terminal:
                raise RuntimeError("This run already reached a terminal state.")
            yield
            if mark_terminal:
                self._terminal = True

    def request_stop(self) -> int:
        """Request cancellation and synchronously clean up every owned process tree."""
        with self._transition_lock:
            if self._terminal:
                raise ValueError("The OpenFOAM run already finished.")
            self._stop_complete.clear()
            self._cancelled.set()
        try:
            return self._terminate_registered_processes()
        finally:
            self._stop_complete.set()

    def _terminate_registered_processes(self) -> int:
        with self._termination_lock:
            with self._lock:
                registrations = list(self._processes.values())
                self._processes.clear()
            errors: list[str] = []
            for registration in registrations:
                try:
                    _terminate_process_tree(registration.process)
                except Exception as exc:
                    errors.append(
                        f"{registration.case_path.name} process-tree termination failed: {exc}"
                    )
            for registration in registrations:
                try:
                    _confirm_backend_cleanup(
                        registration.case_path,
                        registration.backend,
                        registration.execution_id,
                    )
                except Exception as exc:
                    errors.append(
                        f"{registration.case_path.name} backend cleanup failed: {exc}"
                    )
            if errors:
                raise RuntimeError("; ".join(errors))
            return len(registrations)


class _RunControllerContext(threading.local):
    controller: SolverRunController | None

    def __init__(self) -> None:
        self.controller = None


_RUN_CONTROLLER_CONTEXT = _RunControllerContext()


@contextmanager
def run_cancellation_context(
    controller: SolverRunController,
) -> Iterator[SolverRunController]:
    """Bind a controller to the current worker without changing public call sites."""
    previous = _RUN_CONTROLLER_CONTEXT.controller
    _RUN_CONTROLLER_CONTEXT.controller = controller
    try:
        yield controller
    finally:
        _RUN_CONTROLLER_CONTEXT.controller = previous


def _current_run_controller() -> SolverRunController | None:
    return _RUN_CONTROLLER_CONTEXT.controller


@contextmanager
def _case_execution_lock(case_path: Path) -> Iterator[None]:
    """Hold a cross-process lock while a case is running or being cleaned."""
    case_path = case_path.resolve()
    if not case_path.is_dir():
        raise FileNotFoundError(case_path)
    lock_path = case_path / ".aerolab-run.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"Could not open the case run lock: {exc}") from exc
    handle: BinaryIO | None = None
    acquired = False
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RuntimeError("The case run lock must be a regular file.")
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = -1
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":
                import msvcrt

                if lock_stat.st_size == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                raise RuntimeError(
                    f"Case run locking is unsupported on operating system {os.name!r}."
                )
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                raise RuntimeError(
                    "This case is active in another AeroLab process; stop that run before deleting or starting it again."
                ) from exc
            raise
        acquired = True
        yield
    finally:
        if handle is not None:
            if acquired:
                try:
                    if os.name == "posix":
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    elif os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)


def _case_execution_locks(case_paths: list[Path]) -> ExitStack:
    stack = ExitStack()
    try:
        for case_path in sorted(
            {path.resolve() for path in case_paths},
            key=lambda path: str(path),
        ):
            stack.enter_context(_case_execution_lock(case_path))
    except Exception:
        stack.close()
        raise
    return stack


def _safe_cell_budget(resources: dict[str, object]) -> int | None:
    """Estimate a conservative generated-cell allowance from backend memory."""
    available_value = resources.get("memoryAvailableBytes")
    if (
        isinstance(available_value, bool)
        or not isinstance(available_value, int | float)
        or available_value < 0
    ):
        return None
    available = int(available_value)
    reserve = max(1024**3, available // 4)
    cell_memory = max(0, available - reserve - QUALITY_FIXED_MEMORY_BYTES)
    cells = cell_memory // QUALITY_ESTIMATED_BYTES_PER_CELL
    return int(cells // SAFE_CELL_BUDGET_ROUNDING * SAFE_CELL_BUDGET_ROUNDING)


def _suggested_quality(safe_cell_budget: int | None) -> str | None:
    if safe_cell_budget is None:
        return None
    ranked = sorted(
        (
            (
                max(
                    int(preset["max_global_cells"]),
                    int(preset.get("adaptive_max_global_cells", 0)),
                ),
                name,
            )
            for name, preset in QUALITY_PRESETS.items()
        ),
        reverse=True,
    )
    return next(
        (name for cell_budget, name in ranked if safe_cell_budget >= cell_budget),
        None,
    )


def _failure_budget_recommendation(
    case_path: Path,
    *,
    returncode: int,
    log_text: str,
    requested_processes: str | int,
    processes: int,
    process_selection: dict[str, object],
) -> dict[str, object] | None:
    """Diagnose resource-related failures without changing case fidelity."""
    resources_value = process_selection.get("detectedResources")
    resources = resources_value if isinstance(resources_value, dict) else {}
    safe_cells = _safe_cell_budget(resources)
    configured_cells = _case_cell_budget(case_path)
    lower_quality = (
        _suggested_quality(safe_cells)
        if safe_cells is not None
        and configured_cells is not None
        and configured_cells > safe_cells
        else None
    )
    recommended_processes = max(1, math.ceil(max(1, processes) / 2))
    normalized_log = log_text.lower()

    def payload(
        *,
        category: str,
        confidence: str,
        title: str,
        detail: str,
        evidence: list[str],
        retry_allowed: bool = False,
        auto_retry_safe: bool = False,
        process_recommendation: int | None = None,
        preserves_fidelity: bool = True,
    ) -> dict[str, object]:
        return {
            "category": category,
            "confidence": confidence,
            "title": title,
            "detail": detail,
            "evidence": evidence,
            "retryAllowed": retry_allowed,
            "autoRetrySafe": auto_retry_safe,
            "recommendedProcesses": process_recommendation,
            "recommendedProcessBudget": None,
            "safeCellBudget": safe_cells,
            "configuredCellBudget": configured_cells,
            "suggestedQuality": lower_quality,
            "preservesCaseFidelity": preserves_fidelity,
        }

    storage_evidence = [
        phrase
        for phrase in ("no space left on device", "disk quota exceeded")
        if phrase in normalized_log
    ]
    if storage_evidence:
        return payload(
            category="storage_exhaustion",
            confidence="high",
            title="Solver storage is exhausted",
            detail=(
                "Free space in the solver backend or increase its storage allocation, then "
                "retry the unchanged case. Reducing MPI ranks does not repair a full filesystem."
            ),
            evidence=storage_evidence,
        )

    slot_evidence = [
        phrase
        for phrase in (
            "not enough slots",
            "not enough processors",
            "unable to allocate the requested resources",
        )
        if phrase in normalized_log
    ]
    if slot_evidence:
        can_retry = processes > 1
        return payload(
            category="cpu_mpi_oversubscription",
            confidence="high",
            title="MPI requested more slots than the backend can provide",
            detail=(
                f"Retry the unchanged case with {recommended_processes} process"
                f"{'es' if recommended_processes != 1 else ''}. Geometry, mesh quality, "
                "physics, and verification gates remain unchanged."
                if can_retry
                else "The run already used one process; inspect the backend MPI host/slot configuration."
            ),
            evidence=slot_evidence,
            retry_allowed=can_retry,
            auto_retry_safe=can_retry and requested_processes == "auto",
            process_recommendation=recommended_processes if can_retry else None,
        )

    memory_evidence: list[str] = []
    if returncode in {-9, 137}:
        memory_evidence.append(f"process return code {returncode}")
    for phrase in (
        "std::bad_alloc",
        "cannot allocate memory",
        "out of memory",
        "oom-kill",
        "oom killer",
    ):
        if phrase in normalized_log:
            memory_evidence.append(phrase)
    if memory_evidence:
        cell_budget_fits = (
            safe_cells is None
            or configured_cells is None
            or configured_cells <= safe_cells
        )
        can_retry = processes > 1 and cell_budget_fits
        if can_retry:
            detail = (
                f"Retry the unchanged case with {recommended_processes} process"
                f"{'es' if recommended_processes != 1 else ''} to reduce MPI memory overhead. "
                "Geometry, mesh quality, physics, and verification gates remain unchanged."
            )
        elif safe_cells is not None and configured_cells is not None and not cell_budget_fits:
            quality_detail = (
                f" The largest built-in preset within that allowance is {lower_quality.title()}."
                if lower_quality
                else " No built-in quality preset fits that allowance."
            )
            detail = (
                f"The backend's conservative allowance is about {safe_cells:,} cells, below "
                f"this case's {configured_cells:,}-cell cap.{quality_detail} Add backend memory "
                "to preserve this case, or explicitly regenerate with a lower mesh budget and "
                "repeat mesh validation; AeroLab will not downgrade it automatically."
            )
        else:
            detail = (
                "The run already used one process, so no safer same-fidelity rank reduction "
                "remains. Add backend memory or explicitly regenerate with a lower mesh budget "
                "and repeat mesh validation."
            )
        return payload(
            category="memory_oom",
            confidence="high",
            title="OpenFOAM exceeded the backend memory budget",
            detail=detail,
            evidence=memory_evidence,
            retry_allowed=can_retry,
            auto_retry_safe=can_retry and requested_processes == "auto",
            process_recommendation=recommended_processes if can_retry else None,
            preserves_fidelity=can_retry or lower_quality is None,
        )

    cell_matches = re.findall(
        r"After refinement[^\n]*cells:\s*(\d+)",
        log_text,
        re.IGNORECASE,
    )
    refinement_pressure = bool(cell_matches) or any(
        marker in normalized_log
        for marker in (
            "feature refinement iteration",
            "surface refinement iteration",
            "shell refinement iteration",
        )
    )
    if refinement_pressure:
        evidence = ["run ended during mesh refinement"]
        if cell_matches:
            evidence.append(f"last reported mesh size {int(cell_matches[-1]):,} cells")
        if safe_cells is None:
            detail = (
                "Mesh refinement ended before a usable mesh was produced, but the backend did "
                "not report enough memory data to calculate a safe cell allowance. Inspect the "
                "run log and backend memory before retrying."
            )
        elif configured_cells is not None and configured_cells > safe_cells:
            quality_detail = (
                f" {lower_quality.title()} is the largest built-in preset within that allowance."
                if lower_quality
                else " No built-in preset fits that allowance."
            )
            detail = (
                f"This backend's conservative allowance is about {safe_cells:,} cells versus "
                f"the case's {configured_cells:,}-cell cap.{quality_detail} Add memory to keep "
                "the case unchanged, or explicitly regenerate at a lower mesh budget and "
                "validate it again."
            )
        else:
            detail = (
                f"The configured mesh cap is within the backend's conservative {safe_cells:,}-cell "
                "allowance, so this log alone does not prove an out-of-memory failure. Inspect the "
                "final meshing messages or increase the runtime limit before retrying."
            )
        return payload(
            category="mesh_cell_budget",
            confidence="medium",
            title="Mesh refinement reached workstation budget pressure",
            detail=detail,
            evidence=evidence,
            preserves_fidelity=lower_quality is None,
        )

    if returncode == 124:
        return payload(
            category="runtime_timeout",
            confidence="high",
            title="The solver reached its runtime limit",
            detail=(
                "No direct memory, MPI-slot, or storage failure was found. Increase the runtime "
                "limit or inspect the final solver phase; AeroLab will not guess at a compute "
                "adjustment from timeout evidence alone."
            ),
            evidence=["process return code 124"],
        )
    return None


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
        missing_value = resources.get("missingParallelTools")
        missing_tools = missing_value if isinstance(missing_value, list) else []
        missing = ", ".join(str(item) for item in missing_tools)
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
    budget_recommendation: dict[str, object] | None = None

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
            "budgetRecommendation": self.budget_recommendation,
            "returncode": self.returncode,
            "logPath": str(self.log_path),
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "report": self.report,
        }


def _run_record(
    result: SolverRunResult,
    *,
    attempt_id: str,
    execution_id: str,
) -> dict[str, object]:
    return {
        "status": "complete" if result.ok else "failed",
        "attemptId": attempt_id,
        "executionId": execution_id,
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
        "budgetRecommendation": result.budget_recommendation,
        "returncode": result.returncode,
        "logPath": str(result.log_path),
        "startedAt": result.started_at,
        "finishedAt": result.finished_at,
        "qualityAssessment": result.report.get("qualityAssessment"),
        "meshAssessment": result.report.get("meshAssessment"),
    }


def _record_stopped_run(
    case_path: Path,
    run_mode: str,
    backend: str,
    reason: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    run_path = case_path / "aerolab-run.json"
    existing = _read_json_object(run_path)
    was_stopped = existing.get("status") == "stopped"
    started_at = (
        existing.get("startedAt")
        if existing.get("status") in {"running", "stopping", "stopped"}
        else now
    )
    record = {
        **existing,
        "status": "stopped",
        "ok": False,
        "cancelled": True,
        "trusted": False,
        "numericallyQualified": False,
        "mode": run_mode,
        "backend": existing.get("backend") or backend,
        "returncode": None,
        "budgetRecommendation": None,
        "startedAt": started_at,
        "finishedAt": now,
        "stopReason": reason,
    }
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _update_case_status(
        case_path,
        "mesh_stopped" if str(run_mode).lower() == "mesh" else "solver_stopped",
    )
    if not was_stopped:
        try:
            with (case_path / "aerolab-run.log").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(f"\nAeroLab stopped this run: {reason}\n")
        except OSError:
            pass


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
    *,
    cancellation_controller: SolverRunController | None = None,
    _case_lock_held: bool = False,
) -> SolverRunResult:
    """Run one case while holding its cross-process ownership lock."""
    resolved_path = case_path.resolve()
    controller = cancellation_controller or _current_run_controller()
    try:
        if controller is not None:
            controller.raise_if_cancelled()
        lock_already_owned = bool(
            _case_lock_held
            or (controller is not None and controller.owns_case_paths([resolved_path]))
        )
        if lock_already_owned:
            return _run_case_locked(
                resolved_path,
                backend=backend,
                timeout_seconds=timeout_seconds,
                run_mode=run_mode,
                reuse_mesh=reuse_mesh,
                solver_identity=solver_identity,
                processes=processes,
                file_handler=file_handler,
                resume=resume,
                cancellation_controller=controller,
            )
        with _case_execution_lock(resolved_path):
            return _run_case_locked(
                resolved_path,
                backend=backend,
                timeout_seconds=timeout_seconds,
                run_mode=run_mode,
                reuse_mesh=reuse_mesh,
                solver_identity=solver_identity,
                processes=processes,
                file_handler=file_handler,
                resume=resume,
                cancellation_controller=controller,
            )
    except SolverRunCancelled as exc:
        if controller is not None:
            controller.wait_for_stop_completion()
        _record_stopped_run(resolved_path, run_mode, backend, str(exc))
        raise


def _run_case_locked(
    case_path: Path,
    backend: str = "auto",
    timeout_seconds: int = 3600,
    run_mode: str = "full",
    reuse_mesh: bool = True,
    solver_identity: dict[str, object] | None = None,
    processes: str | int = 1,
    file_handler: str = "auto",
    resume: bool = False,
    *,
    cancellation_controller: SolverRunController | None = None,
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
    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()

    ensure_case_postprocessing(case_path)
    solver_input_fingerprint = _solver_input_fingerprint(case_path)
    convergence_policy = _convergence_policy(case_path)
    resume_state = (
        _resume_compatibility(case_path, solver_input_fingerprint)
        if resume
        else None
    )
    resume_from_time = None
    if isinstance(resume_state, dict):
        resume_from_time = _finite_number(resume_state.get("latestTime"))
        if resume_from_time is None:
            raise RuntimeError("Resume compatibility returned an invalid latest time.")
    status = solver_status()
    selected = _select_backend(status, backend)
    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()
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
    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()
    started_at = datetime.now(timezone.utc).isoformat()
    log_path = case_path / "aerolab-run.log"
    run_path = case_path / "aerolab-run.json"

    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()
    _clear_previous_solver_outputs(
        case_path,
        preserve_mesh=reused_mesh,
        preserve_solver_state=resume,
    )
    attempt_id = (
        cancellation_controller.attempt_id
        if cancellation_controller is not None
        else uuid.uuid4().hex
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
                "attemptId": attempt_id,
                "executionId": execution_id,
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
                "budgetRecommendation": None,
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
        if cancellation_controller is not None:
            cancellation_controller.raise_if_cancelled()
        returncode, outer_timeout = _run_solver_process(
            command,
            case_path=case_path,
            backend=selected,
            execution_id=execution_id,
            process_timeout=process_timeout,
            log_file=log_file,
            cancellation_controller=cancellation_controller,
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
    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()
    finished_at = datetime.now(timezone.utc).isoformat()
    budget_recommendation = (
        _failure_budget_recommendation(
            case_path,
            returncode=returncode,
            log_text=_tail_text(log_path, maximum_bytes=512 * 1024),
            requested_processes=requested_processes,
            processes=resolved_processes,
            process_selection=process_selection,
        )
        if returncode != 0
        else None
    )
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
        budget_recommendation=budget_recommendation,
    )
    with ExitStack() as terminal_stack:
        if cancellation_controller is not None:
            terminal_stack.enter_context(
                cancellation_controller.terminal_commit(
                    mark_terminal=not cancellation_controller.owns_multiple_cases
                )
            )
        run_path.write_text(
            json.dumps(
                _run_record(
                    result,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
        budget_recommendation=budget_recommendation,
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
    cancellation_controller: SolverRunController | None = None,
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
        if cancellation_controller is not None:
            cancellation_controller.register_process(
                process,
                case_path=case_path,
                backend=backend,
                execution_id=execution_id,
            )
        try:
            returncode = process.wait(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            if (
                cancellation_controller is not None
                and cancellation_controller.cancellation_requested
            ):
                cancellation_controller.request_stop()
                cancellation_controller.raise_if_cancelled()
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
        if cancellation_controller is not None:
            cancellation_controller.raise_if_cancelled()
        return returncode, False
    finally:
        if cancellation_controller is not None:
            cancellation_controller.unregister_process(process)


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

    verification_label = (
        "container-absence verification"
        if backend == "docker"
        else f"{backend} cleanup verification"
    )
    resource_label = "container" if backend == "docker" else f"{backend} run resource"
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
        raise RuntimeError(f"{verification_label} failed: {exc}") from exc
    if verification.returncode != 0:
        detail = (verification.stderr or verification.stdout).strip()
        raise RuntimeError(
            f"{verification_label} failed"
            + (f": {detail}" if detail else "")
            + (f"; cleanup reported: {cleanup_detail}" if cleanup_detail else "")
        )
    remaining_resources = verification.stdout.strip()
    if remaining_resources:
        raise RuntimeError(
            f"{resource_label} still exists after forced removal: {remaining_resources}"
            + (f"; cleanup reported: {cleanup_detail}" if cleanup_detail else "")
        )


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = 30.0,
) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            process_group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            process.poll()
            return
        if process.poll() is not None:
            return
        if process_group_id != process.pid:
            raise RuntimeError(
                "Refusing to signal a process that no longer leads its owned session."
            )
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


def _remove_case_generated_path(case_path: Path, target: Path) -> None:
    """Remove one case-local generated path without following any symlink."""
    case_path = case_path.resolve()
    try:
        relative = target.relative_to(case_path)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to remove a path outside the case: {target}") from exc
    parent = case_path
    for part in relative.parts[:-1]:
        parent = parent / part
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(parent_stat.st_mode):
            return
        if not stat.S_ISDIR(parent_stat.st_mode):
            return
    try:
        target_stat = target.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(target_stat.st_mode):
        target.unlink()
    elif stat.S_ISDIR(target_stat.st_mode):
        shutil.rmtree(target)


def _clear_previous_solver_outputs(
    case_path: Path,
    preserve_mesh: bool = False,
    preserve_solver_state: bool = False,
) -> None:
    """Remove generated products without following case-local symlinks."""
    if preserve_solver_state and not preserve_mesh:
        raise ValueError("Preserving solver state also requires preserving the mesh.")
    case_path = case_path.resolve()
    directory_targets: list[Path] = []
    post_processing = case_path / "postProcessing"
    try:
        post_processing_stat = post_processing.lstat()
    except FileNotFoundError:
        post_processing_stat = None
    if not preserve_solver_state:
        if (
            preserve_mesh
            and post_processing_stat is not None
            and stat.S_ISDIR(post_processing_stat.st_mode)
            and not stat.S_ISLNK(post_processing_stat.st_mode)
        ):
            directory_targets.extend(
                child for child in post_processing.iterdir() if child.name != "meshSurface"
            )
        else:
            directory_targets.append(post_processing)
    if not preserve_mesh:
        directory_targets.append(case_path / "constant" / "polyMesh")
    processor_pattern = re.compile(r"(?:processor\d+|processors\d+(?:_\d+-\d+)?)\Z")
    for child in case_path.iterdir():
        if child.name == "0":
            continue
        try:
            child_stat = child.lstat()
        except FileNotFoundError:
            continue
        removable_kind = stat.S_ISDIR(child_stat.st_mode) or stat.S_ISLNK(
            child_stat.st_mode
        )
        if not removable_kind:
            continue
        if processor_pattern.fullmatch(child.name):
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

    for target in dict.fromkeys(directory_targets):
        _remove_case_generated_path(case_path, target)

    if not preserve_mesh:
        for filename in (
            "mesh-surface-fidelity.json",
            "aerolab-mesh.json",
            "aerolab-mesh.log",
        ):
            target = case_path / filename
            if target.exists() or target.is_symlink():
                target.unlink()


def clear_case_run_outputs(
    case_path: Path,
    *,
    related_case_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Delete generated run products while retaining setup, geometry, and mesh."""
    case_path = case_path.resolve()
    related_paths = [path.resolve() for path in (related_case_paths or [])]
    locked_paths = list({case_path, *related_paths})
    for path in locked_paths:
        if not path.is_dir() or not (path / "case.json").is_file():
            raise ValueError(f"{path} is not an AeroLab case.")

    deleted: list[str] = []
    cleared_study_records: list[str] = []
    with _case_execution_locks(locked_paths):
        preserve_mesh = _mesh_record_reusable(case_path)
        mesh_record = _read_json_object(case_path / "aerolab-mesh.json")
        _clear_previous_solver_outputs(case_path, preserve_mesh=preserve_mesh)
        for filename in ("aerolab-run.json", "aerolab-run.log"):
            target = case_path / filename
            if target.exists() or target.is_symlink():
                target.unlink()
                deleted.append(filename)
        for related_path in locked_paths:
            removed = False
            for filename in (
                "aerolab-study-run.json",
                ".aerolab-study-run.json.tmp",
            ):
                target = related_path / filename
                if target.exists() or target.is_symlink():
                    target.unlink()
                    removed = True
            if removed:
                cleared_study_records.append(str(related_path))

        if preserve_mesh:
            reset_status = (
                "mesh_validated" if mesh_record.get("trusted") else "mesh_unverified"
            )
        else:
            reset_status = "openfoam_case_generated"
        _update_case_status(case_path, reset_status)

    return {
        "casePath": str(case_path),
        "deletedFiles": deleted,
        "preservedMesh": preserve_mesh,
        "resetStatus": reset_status,
        "clearedStudyRecordCasePaths": cleared_study_records,
    }


def _case_local_regular_file(case_path: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(case_path)
    except ValueError:
        return False
    current = case_path
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError:
            return False
        if stat.S_ISLNK(current_stat.st_mode):
            return False
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(current_stat.st_mode):
            return False
    return stat.S_ISREG(current_stat.st_mode)


def reconcile_interrupted_case_run(case_path: Path) -> str | None:
    """Safely reconcile a durable running record left by a previous server."""
    case_path = case_path.resolve()
    run_path = case_path / "aerolab-run.json"
    existing = _read_json_object(run_path)
    if existing.get("status") not in {"running", "stopping"}:
        return None
    with _case_execution_lock(case_path):
        existing = _read_json_object(run_path)
        if existing.get("status") not in {"running", "stopping"}:
            return None
        backend = str(existing.get("backend") or "")
        execution_id = existing.get("executionId")
        mode = str(existing.get("mode") or "full")
        if backend in {"docker", "wsl"} and isinstance(execution_id, str):
            _confirm_backend_cleanup(case_path, backend, execution_id)
            _record_stopped_run(
                case_path,
                mode,
                backend,
                "AeroLab safely cleaned up this interrupted backend run during server startup.",
            )
            return "stopped"
        now = datetime.now(timezone.utc).isoformat()
        run_path.write_text(
            json.dumps(
                {
                    **existing,
                    "status": "orphaned",
                    "ok": False,
                    "trusted": False,
                    "numericallyQualified": False,
                    "finishedAt": now,
                    "error": (
                        "The previous AeroLab server ended without a durable backend "
                        "cleanup identity. Verify that no native OpenFOAM process remains "
                        "before removing this record."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _update_case_status(
            case_path,
            "mesh_failed" if mode == "mesh" else "solver_failed",
        )
        return "orphaned"


def _mesh_outputs_present(case_path: Path) -> bool:
    points_path = case_path / "constant" / "polyMesh" / "points"
    surface_path = _latest_body_surface_vtk(case_path)
    return bool(
        surface_path is not None
        and _case_local_regular_file(case_path, points_path)
        and _case_local_regular_file(case_path, surface_path)
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
        relative_name = path.relative_to(case_path).as_posix()
        digest.update(relative_name.encode("utf-8"))
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
