"""One-factor sensitivity-study creation and statistical evidence assembly."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from ..case import create_case
from .backends import _select_backend, probe_backend_resources, solver_status
from .run import (
    AUTO_BYTES_PER_PROCESS,
    AUTO_CELLS_PER_PROCESS,
    AUTO_MAX_PROCESSES,
    MANUAL_MIN_BYTES_PER_PROCESS,
    QUALITY_ESTIMATED_BYTES_PER_CELL,
    QUALITY_FIXED_MEMORY_BYTES,
    SolverRunCancelled,
    SolverRunController,
    _case_cell_budget,
    _case_execution_locks,
    _current_run_controller,
    normalize_file_handler,
    normalize_process_request,
    run_case,
)
from .util import _finite_number, _read_json_object

SENSITIVITY_PARAMETERS: dict[str, dict[str, object]] = {
    "speed_mph": {"label": "Freestream speed", "unit": "mph", "minimum": 0.0},
    "yaw_degrees": {
        "label": "Yaw angle",
        "unit": "deg",
        "minimum": -90.0,
        "maximum": 90.0,
    },
    "crosswind_mps": {"label": "Crosswind component", "unit": "m/s"},
    "roughness_height_m": {
        "label": "Equivalent roughness height",
        "unit": "m",
        "minimum_inclusive": 0.0,
    },
    "ground_clearance_m": {
        "label": "Ground clearance",
        "unit": "m",
        "minimum_inclusive": 0.0,
    },
    "turbulence_intensity_percent": {
        "label": "Turbulence intensity",
        "unit": "%",
        "minimum": 0.0,
        "maximum_inclusive": 50.0,
    },
}

_COEFFICIENT_CHANNELS = ("Cd", "Cl", "Cs", "CmRoll", "CmPitch", "CmYaw")


def normalize_study_process_budget(value: object) -> str | int:
    """Normalize the aggregate rank budget shared by all concurrent members."""
    try:
        return normalize_process_request(value)
    except ValueError as exc:
        raise ValueError(
            "Study process budget must be 'auto' or a positive integer."
        ) from exc


def study_members(case_path: Path) -> dict[str, object]:
    """Return an ordered, validated set of sibling cases in one AeroLab study."""
    descriptor, member_paths = _discover_study(case_path.resolve())
    return {
        **descriptor,
        "casePaths": [str(path) for path in member_paths],
        "selectedCasePath": str(case_path.resolve()),
    }


def plan_study_schedule(
    case_path: Path,
    *,
    backend: str = "auto",
    processes: str | int = "auto",
    process_budget: str | int = "auto",
    solver_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one aggregate CPU/memory allocation for all independent members."""
    descriptor, member_paths = _discover_study(case_path.resolve())
    for member_path in member_paths:
        if not (member_path / "Allrun").is_file():
            raise ValueError(
                f"Study member {member_path.name} has no generated Allrun script."
            )
    status = solver_status()
    selected_backend = _select_backend(status, backend)
    resources = probe_backend_resources(
        selected_backend,
        solver_identity=solver_identity,
    )
    allocation = _study_resource_allocation(
        member_paths,
        resources,
        processes=processes,
        process_budget=process_budget,
    )
    return {
        **descriptor,
        "backend": selected_backend,
        "casePaths": [str(path) for path in member_paths],
        "resources": resources,
        **allocation,
    }


def run_study(
    case_path: Path,
    *,
    backend: str = "auto",
    timeout_seconds: int = 3600,
    run_mode: str = "full",
    reuse_mesh: bool = True,
    processes: str | int = "auto",
    process_budget: str | int = "auto",
    file_handler: str = "auto",
    solver_identity: dict[str, object] | None = None,
    cancellation_controller: SolverRunController | None = None,
) -> dict[str, object]:
    """Run study members under one shared cancellation and ownership scope."""
    case_path = case_path.resolve()
    controller = cancellation_controller or _current_run_controller()
    if controller is not None:
        controller.raise_if_cancelled()
    plan = plan_study_schedule(
        case_path,
        backend=backend,
        processes=processes,
        process_budget=process_budget,
        solver_identity=solver_identity,
    )
    locked_members = [
        Path(str(value)).resolve() for value in plan["casePaths"]
    ]
    with ExitStack() as lock_stack:
        if controller is None or not controller.owns_case_paths(locked_members):
            lock_stack.enter_context(_case_execution_locks(locked_members))
        return _run_study_locked(
            plan=plan,
            timeout_seconds=timeout_seconds,
            run_mode=run_mode,
            reuse_mesh=reuse_mesh,
            file_handler=file_handler,
            solver_identity=solver_identity,
            cancellation_controller=controller,
            locked_members=locked_members,
        )


def _run_study_locked(
    *,
    plan: dict[str, object],
    timeout_seconds: int,
    run_mode: str,
    reuse_mesh: bool,
    file_handler: str,
    solver_identity: dict[str, object] | None,
    cancellation_controller: SolverRunController | None,
    locked_members: list[Path],
) -> dict[str, object]:
    selected_file_handler = normalize_file_handler(file_handler)
    member_paths = [Path(str(value)).resolve() for value in plan["casePaths"]]
    if set(member_paths) != set(locked_members):
        raise RuntimeError("Study membership changed while acquiring run ownership.")
    if cancellation_controller is not None:
        cancellation_controller.raise_if_cancelled()
    ranks_per_case = int(plan["processesPerCase"])
    max_workers = int(plan["maxConcurrentCases"])
    started_at = datetime.now(timezone.utc).isoformat()
    attempt_id = (
        cancellation_controller.attempt_id
        if cancellation_controller is not None
        else hashlib.sha256(
            f"{plan['studyId']}\0{started_at}".encode()
        ).hexdigest()
    )
    running_record = {
        "status": "running",
        "attemptId": attempt_id,
        "ok": None,
        "studyId": plan["studyId"],
        "kind": plan["kind"],
        "startedAt": started_at,
        "finishedAt": None,
        "completedCases": 0,
        "totalCases": len(member_paths),
        "percent": 0,
        "plan": plan,
        "results": [],
        "budgetRecommendation": None,
    }
    _write_study_run_record(member_paths, running_record)

    indexed_results: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="aerolab-study",
    ) as executor:
        futures = {
            executor.submit(
                run_case,
                member_path,
                backend=str(plan["backend"]),
                timeout_seconds=timeout_seconds,
                run_mode=run_mode,
                reuse_mesh=reuse_mesh,
                solver_identity=solver_identity,
                processes=ranks_per_case,
                file_handler=selected_file_handler,
                cancellation_controller=cancellation_controller,
                _case_lock_held=True,
            ): (index, member_path)
            for index, member_path in enumerate(member_paths)
        }
        for future in as_completed(futures):
            index, member_path = futures[future]
            try:
                result = future.result()
            except SolverRunCancelled as exc:
                indexed_results[index] = {
                    "casePath": str(member_path),
                    "status": "stopped",
                    "ok": False,
                    "trusted": False,
                    "cancelled": True,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            except Exception as exc:
                indexed_results[index] = {
                    "casePath": str(member_path),
                    "status": "failed",
                    "ok": False,
                    "trusted": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            else:
                indexed_results[index] = {
                    "casePath": str(member_path),
                    "status": "complete" if result.ok else "failed",
                    "ok": result.ok,
                    "trusted": result.trusted,
                    "returncode": result.returncode,
                    "processes": result.processes,
                    "fileHandler": result.file_handler,
                    "reusedMesh": result.reused_mesh,
                    "logPath": str(result.log_path),
                    "qualityRecommendation": result.process_selection.get(
                        "qualityRecommendation"
                    ),
                    "budgetRecommendation": getattr(
                        result,
                        "budget_recommendation",
                        None,
                    ),
                    "stageCache": result.process_selection.get("stageCache"),
                }
            partial_results = [
                indexed_results[result_index]
                for result_index in sorted(indexed_results)
            ]
            completed_cases = sum(
                result.get("status") != "stopped" for result in partial_results
            )
            stopping = bool(
                cancellation_controller is not None
                and cancellation_controller.cancellation_requested
            )
            _write_study_run_record(
                member_paths,
                {
                    **running_record,
                    "status": "stopping" if stopping else "running",
                    "completedCases": completed_cases,
                    "percent": round(len(partial_results) / len(member_paths) * 100),
                    "results": partial_results,
                    "budgetRecommendation": (
                        None
                        if stopping
                        else _study_budget_recommendation(plan, partial_results)
                    ),
                },
            )

    results = [indexed_results[index] for index in range(len(member_paths))]
    stopped = bool(
        (
            cancellation_controller is not None
            and cancellation_controller.cancellation_requested
        )
        or any(result.get("status") == "stopped" for result in results)
    )
    ok = bool(not stopped and results and all(result.get("ok") for result in results))
    trusted = bool(
        not stopped and results and all(result.get("trusted") for result in results)
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    completed_cases = sum(result.get("status") != "stopped" for result in results)
    final_record = {
        "status": "stopped" if stopped else "complete" if ok else "partial_failure",
        "attemptId": attempt_id,
        "ok": ok,
        "trusted": trusted,
        "cancelled": stopped,
        "studyId": plan["studyId"],
        "kind": plan["kind"],
        "startedAt": started_at,
        "finishedAt": finished_at,
        "completedCases": completed_cases,
        "totalCases": len(member_paths),
        "percent": 100,
        "plan": plan,
        "results": results,
        "budgetRecommendation": (
            None if stopped else _study_budget_recommendation(plan, results)
        ),
    }
    if cancellation_controller is not None and stopped:
        cancellation_controller.wait_for_stop_completion()
        _write_study_run_record(member_paths, final_record)
    elif cancellation_controller is not None:
        with cancellation_controller.terminal_commit():
            _write_study_run_record(member_paths, final_record)
    else:
        _write_study_run_record(member_paths, final_record)
    return final_record


def _study_budget_recommendation(
    plan: dict[str, object],
    results: list[dict[str, object]],
) -> dict[str, object] | None:
    failed_results = [result for result in results if result.get("ok") is False]
    recommendations = [
        recommendation
        for result in failed_results
        if isinstance(
            recommendation := result.get("budgetRecommendation"),
            dict,
        )
    ]
    if not recommendations:
        return None

    retryable = bool(failed_results) and len(recommendations) == len(failed_results)
    retryable = retryable and all(
        recommendation.get("retryAllowed") is True
        and isinstance(recommendation.get("recommendedProcesses"), int)
        for recommendation in recommendations
    )
    planned_processes = max(1, int(plan.get("processesPerCase") or 1))
    recommended_processes = (
        min(int(recommendation["recommendedProcesses"]) for recommendation in recommendations)
        if retryable
        else max(1, math.ceil(planned_processes / 2))
    )
    safe_cell_values = [
        int(value)
        for recommendation in recommendations
        if isinstance(value := recommendation.get("safeCellBudget"), int)
    ]
    configured_cell_values = [
        int(value)
        for recommendation in recommendations
        if isinstance(value := recommendation.get("configuredCellBudget"), int)
    ]
    suggested_quality = next(
        (
            str(value)
            for recommendation in recommendations
            if (value := recommendation.get("suggestedQuality"))
        ),
        None,
    )
    confidence = (
        "high"
        if all(recommendation.get("confidence") == "high" for recommendation in recommendations)
        else "medium"
    )
    evidence = [
        f"{len(failed_results)} of {len(results)} completed study members failed",
        *[
            f"{Path(str(result.get('casePath') or 'member')).name}: "
            f"{recommendation.get('category') or 'resource pressure'}"
            for result in failed_results
            if isinstance(
                recommendation := result.get("budgetRecommendation"),
                dict,
            )
        ],
    ]
    if retryable:
        detail = (
            f"Retry explicitly with {recommended_processes} process"
            f"{'es' if recommended_processes != 1 else ''} per case and an aggregate budget "
            f"of {recommended_processes}. That runs one member at a time while preserving every "
            "case's geometry, mesh, physics, and verification gates. Successful members are not "
            "silently reused, so confirm the explicit whole-study retry."
        )
    else:
        detail = (
            "Review each failed member's resource evidence before rerunning the study. At least "
            "one failure has no safe same-fidelity compute adjustment, so AeroLab will not offer "
            "an automatic or one-click whole-study retry."
        )
    return {
        "category": "study_concurrency_pressure",
        "confidence": confidence,
        "title": "Study members exceeded the shared workstation budget",
        "detail": detail,
        "evidence": evidence,
        "retryAllowed": retryable,
        "autoRetrySafe": False,
        "recommendedProcesses": recommended_processes if retryable else None,
        "recommendedProcessBudget": recommended_processes if retryable else None,
        "safeCellBudget": min(safe_cell_values) if safe_cell_values else None,
        "configuredCellBudget": (
            max(configured_cell_values) if configured_cell_values else None
        ),
        "suggestedQuality": suggested_quality,
        "preservesCaseFidelity": retryable and all(
            recommendation.get("preservesCaseFidelity") is True
            for recommendation in recommendations
        ),
    }


def _discover_study(case_path: Path) -> tuple[dict[str, object], list[Path]]:
    selected_payload = _read_json_object(case_path / "case.json")
    if not selected_payload:
        raise ValueError(f"{case_path} is not an AeroLab case with readable metadata.")
    sensitivity = selected_payload.get("sensitivity_study")
    if isinstance(sensitivity, dict) and sensitivity.get("id"):
        return _discover_sensitivity_members(case_path, sensitivity)
    validation = selected_payload.get("validation_study")
    if isinstance(validation, dict) and validation.get("id"):
        return _discover_accuracy_members(case_path, validation)
    raise ValueError("The selected case is not part of an AeroLab study.")


def _discover_sensitivity_members(
    case_path: Path,
    selected_study: dict[str, object],
) -> tuple[dict[str, object], list[Path]]:
    study_id = str(selected_study["id"])
    parameter = str(selected_study.get("parameter") or "")
    specification = _parameter_specification(parameter)
    raw_values = selected_study.get("values")
    if not isinstance(raw_values, list):
        raise ValueError("Sensitivity-study metadata has no valid value list.")
    values = _sensitivity_values(raw_values, specification)
    expected_count = int(selected_study.get("count") or 0)
    baseline_index = int(selected_study.get("baseline_index") or 0)
    plan_lock = str(selected_study.get("plan_lock_hash") or "")
    if expected_count != len(values) or not plan_lock:
        raise ValueError("Sensitivity-study metadata is incomplete or inconsistent.")

    members: dict[int, tuple[Path, dict[str, object], dict[str, object]]] = {}
    for case_json_path in case_path.parent.glob("*/case.json"):
        payload = _read_json_object(case_json_path)
        study = payload.get("sensitivity_study")
        if not isinstance(study, dict) or str(study.get("id") or "") != study_id:
            continue
        try:
            index = int(study.get("index"))
        except (TypeError, ValueError):
            raise ValueError("A sensitivity-study member has an invalid index.") from None
        if index in members:
            raise ValueError(f"Sensitivity-study index {index} appears more than once.")
        members[index] = (case_json_path.parent.resolve(), payload, study)

    if sorted(members) != list(range(expected_count)):
        raise ValueError("Sensitivity-study members are missing or have non-contiguous indices.")
    ordered: list[Path] = []
    for index in range(expected_count):
        member_path, payload, study = members[index]
        if (
            str(study.get("parameter") or "") != parameter
            or study.get("values") != raw_values
            or int(study.get("count") or 0) != expected_count
            or int(study.get("baseline_index") or 0) != baseline_index
            or str(study.get("plan_lock_hash") or "") != plan_lock
        ):
            raise ValueError("Sensitivity-study member metadata does not match the selected plan.")
        if _member_plan_lock(payload, member_path, parameter, values) != plan_lock:
            raise ValueError("Sensitivity-study plan lock no longer matches a member case.")
        ordered.append(member_path)
    return (
        {
            "studyId": study_id,
            "kind": "one_factor_sensitivity",
            "parameter": parameter,
            "values": values,
            "baselineIndex": baseline_index,
            "planLockHash": plan_lock,
        },
        ordered,
    )


def _discover_accuracy_members(
    case_path: Path,
    selected_study: dict[str, object],
) -> tuple[dict[str, object], list[Path]]:
    study_id = str(selected_study["id"])
    raw_levels = selected_study.get("levels")
    levels = (
        [str(level).strip().lower() for level in raw_levels]
        if isinstance(raw_levels, list)
        else ["draft", "standard", "fine"]
    )
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Accuracy-study level metadata is invalid.")
    members: dict[str, Path] = {}
    for case_json_path in case_path.parent.glob("*/case.json"):
        payload = _read_json_object(case_json_path)
        study = payload.get("validation_study")
        if not isinstance(study, dict) or str(study.get("id") or "") != study_id:
            continue
        level = str(study.get("level") or "").lower()
        if level not in levels:
            continue
        if level in members:
            raise ValueError(f"Accuracy-study level {level} appears more than once.")
        member_levels = study.get("levels")
        if isinstance(member_levels, list) and [
            str(value).strip().lower() for value in member_levels
        ] != levels:
            raise ValueError("Accuracy-study member level metadata is inconsistent.")
        members[level] = case_json_path.parent.resolve()
    if set(members) != set(levels):
        raise ValueError("Accuracy-study members are missing one or more mesh levels.")
    return (
        {
            "studyId": study_id,
            "kind": "grid_convergence",
            "levels": levels,
        },
        [members[level] for level in levels],
    )


def _study_resource_allocation(
    member_paths: list[Path],
    resources: dict[str, object],
    *,
    processes: str | int,
    process_budget: str | int,
) -> dict[str, object]:
    requested_processes = normalize_process_request(processes)
    requested_budget = normalize_study_process_budget(process_budget)
    effective_cpus = max(1, int(resources.get("effectiveCpus") or 1))
    memory_value = resources.get("memoryAvailableBytes")
    available_memory = (
        int(memory_value)
        if isinstance(memory_value, int | float)
        and not isinstance(memory_value, bool)
        and memory_value >= 0
        else None
    )
    reserved_memory = (
        max(1024**3, available_memory // 4)
        if available_memory is not None
        else None
    )
    usable_memory = (
        max(0, available_memory - reserved_memory)
        if available_memory is not None and reserved_memory is not None
        else None
    )

    budget_caps: dict[str, int] = {}
    if requested_budget == "auto":
        budget_caps["cpu"] = max(1, effective_cpus - 1)
        budget_caps["safety"] = AUTO_MAX_PROCESSES
        if usable_memory is not None:
            budget_caps["memory"] = max(1, usable_memory // AUTO_BYTES_PER_PROCESS)
        resolved_budget = max(1, min(budget_caps.values()))
    else:
        resolved_budget = requested_budget
        if resolved_budget > effective_cpus:
            raise ValueError(
                f"Study process budget {resolved_budget} exceeds {effective_cpus} backend CPUs."
            )
        if (
            available_memory is not None
            and available_memory
            < resolved_budget * MANUAL_MIN_BYTES_PER_PROCESS
        ):
            raise ValueError(
                "Study process budget exceeds the backend's minimum aggregate memory allowance."
            )
        budget_caps["manual"] = resolved_budget

    parallel_available = bool(resources.get("parallelAvailable"))
    if isinstance(requested_processes, int) and requested_processes > 1:
        if not parallel_available:
            raise RuntimeError("The selected backend does not provide complete MPI tools.")
        if requested_processes > resolved_budget:
            raise ValueError(
                "Per-case processes cannot exceed the aggregate study process budget."
            )

    memory_estimates: list[dict[str, object]] = []
    for member_path in member_paths:
        cell_budget = _case_cell_budget(member_path)
        estimated_memory = QUALITY_FIXED_MEMORY_BYTES + (
            (cell_budget or 0) * QUALITY_ESTIMATED_BYTES_PER_CELL
        )
        memory_estimates.append(
            {
                "casePath": str(member_path),
                "configuredMaxCells": cell_budget,
                "estimatedMemoryBytes": estimated_memory,
            }
        )
    largest_estimate = max(
        int(item["estimatedMemoryBytes"]) for item in memory_estimates
    )
    memory_worker_cap = len(member_paths)
    if usable_memory is not None:
        memory_worker_cap = max(1, usable_memory // max(1, largest_estimate))

    if isinstance(requested_processes, int):
        ranks_per_case = requested_processes
        rank_worker_cap = max(1, resolved_budget // ranks_per_case)
        max_workers = min(len(member_paths), rank_worker_cap, memory_worker_cap)
        selection_reason = (
            "Packed validated manual per-case ranks into one aggregate CPU/memory budget."
        )
    elif not parallel_available:
        ranks_per_case = 1
        max_workers = min(len(member_paths), resolved_budget, memory_worker_cap)
        selection_reason = (
            "MPI is unavailable; scheduled independent serial cases within the shared budget."
        )
    else:
        initial_workers = (
            1
            if len(member_paths) == 1
            else min(len(member_paths), max(1, resolved_budget // 2))
        )
        max_workers = max(1, min(initial_workers, memory_worker_cap))
        ranks_per_case = max(1, resolved_budget // max_workers)
        cell_rank_caps = [
            max(1, cell_budget // AUTO_CELLS_PER_PROCESS)
            for cell_budget in (_case_cell_budget(path) for path in member_paths)
            if cell_budget is not None
        ]
        if cell_rank_caps:
            ranks_per_case = min(ranks_per_case, min(cell_rank_caps))
        ranks_per_case = min(ranks_per_case, AUTO_MAX_PROCESSES)
        max_workers = max(
            1,
            min(
                len(member_paths),
                resolved_budget // ranks_per_case,
                memory_worker_cap,
            ),
        )
        selection_reason = (
            "Balanced concurrent study throughput against per-case MPI ranks, CPU, memory, "
            "and cell budgets."
        )

    aggregate_allocation = max_workers * ranks_per_case
    memory_warning = None
    if available_memory is not None and largest_estimate > available_memory:
        memory_warning = (
            "At least one case exceeds the conservative local-memory estimate; the scheduler "
            "will run only one such case at a time, but more backend memory or lower quality "
            "may still be required."
        )
    return {
        "requestedProcessesPerCase": requested_processes,
        "processesPerCase": ranks_per_case,
        "requestedProcessBudget": requested_budget,
        "processBudget": resolved_budget,
        "budgetCaps": budget_caps,
        "maxConcurrentCases": max_workers,
        "maximumAllocatedProcesses": aggregate_allocation,
        "estimatedWaves": math.ceil(len(member_paths) / max_workers),
        "caseMemoryEstimates": memory_estimates,
        "memoryWorkerCap": memory_worker_cap,
        "memoryWarning": memory_warning,
        "selectionReason": selection_reason,
        "allocations": [
            {"casePath": str(path), "processes": ranks_per_case}
            for path in member_paths
        ],
    }


def _write_study_run_record(
    member_paths: list[Path],
    record: dict[str, object],
) -> None:
    content = json.dumps(record, indent=2) + "\n"
    for member_path in member_paths:
        destination = member_path / "aerolab-study-run.json"
        temporary = member_path / ".aerolab-study-run.json.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)


def create_sensitivity_study(
    *,
    base_options: dict[str, object],
    base_name: str,
    cases_dir: Path,
    parameter: str,
    values: list[float],
    generate_openfoam: bool = True,
    baseline_index: int | None = None,
) -> dict[str, object]:
    """Create a bounded one-factor-at-a-time case family.

    Every member is created through :func:`create_case`; the orchestrator only
    changes the declared parameter and records a shared plan lock. Numerical and
    statistical qualification remain result-time concerns.
    """

    specification = _parameter_specification(parameter)
    normalized_values = _sensitivity_values(values, specification)
    if baseline_index is None:
        baseline_index = _default_baseline_index(
            normalized_values,
            _finite_number(base_options.get(parameter)),
        )
    if baseline_index < 0 or baseline_index >= len(normalized_values):
        raise ValueError("Sensitivity-study baseline index is outside the value list.")

    cases_dir = cases_dir.resolve()
    study_id = f"sensitivity-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    paths: list[Path] = []
    for index, value in enumerate(normalized_values):
        options = dict(base_options)
        options[parameter] = value
        if parameter == "yaw_degrees":
            options["crosswind_mps"] = None
        elif parameter == "crosswind_mps":
            options["yaw_degrees"] = None
        metadata = {
            "schema_version": 1,
            "id": study_id,
            "kind": "one_factor_sensitivity",
            "parameter": parameter,
            "parameter_label": specification["label"],
            "unit": specification["unit"],
            "value": value,
            "index": index,
            "count": len(normalized_values),
            "values": normalized_values,
            "baseline_index": baseline_index,
            "is_baseline": index == baseline_index,
            "plan_lock_algorithm": "sha256",
            "plan_lock_hash": None,
            "plan_lock_scope": "reconstructed_non_varied_case_options",
        }
        case_path = create_case(
            case_name=f"{base_name}-{study_id}-{_value_token(value)}",
            cases_dir=cases_dir,
            generate_openfoam=generate_openfoam,
            sensitivity_study=metadata,
            **options,
        )
        paths.append(case_path)

    plan_lock = _persist_study_plan_lock(paths, parameter, normalized_values)
    return {
        "schemaVersion": 1,
        "studyId": study_id,
        "kind": "one_factor_sensitivity",
        "parameter": parameter,
        "parameterLabel": specification["label"],
        "unit": specification["unit"],
        "values": normalized_values,
        "baselineIndex": baseline_index,
        "planLockHash": plan_lock,
        "casePaths": [str(path) for path in paths],
        "selectedCasePath": str(paths[baseline_index]),
    }


def create_sensitivity_study_from_case(
    *,
    base_case_path: Path,
    parameter: str,
    values: list[float],
    cases_dir: Path | None = None,
    base_name: str | None = None,
    generate_openfoam: bool = True,
    baseline_index: int | None = None,
) -> dict[str, object]:
    """Create a one-factor family while preserving an existing case's stored setup."""

    base_case_path = base_case_path.resolve()
    payload = _read_json_object(base_case_path / "case.json")
    if not payload:
        raise ValueError(
            f"{base_case_path} is not an AeroLab case with readable case.json metadata."
        )
    specification = _parameter_specification(parameter)
    normalized_values = _sensitivity_values(values, specification)
    if baseline_index is None:
        baseline_index = _default_baseline_index(
            normalized_values,
            _stored_parameter_value(payload, parameter),
        )
    return create_sensitivity_study(
        base_options=_case_options_from_payload(payload, base_case_path),
        base_name=str(base_name or payload.get("name") or base_case_path.name),
        cases_dir=(cases_dir or base_case_path.parent),
        parameter=parameter,
        values=normalized_values,
        generate_openfoam=generate_openfoam,
        baseline_index=baseline_index,
    )


def sensitivity_study_report(case_path: Path) -> dict[str, object] | None:
    """Collect one study's numerical and statistical evidence."""

    from .analysis import case_report

    case_path = case_path.resolve()
    selected_payload = _read_json_object(case_path / "case.json")
    selected_study = selected_payload.get("sensitivity_study")
    if not isinstance(selected_study, dict) or not selected_study.get("id"):
        return None
    study_id = str(selected_study["id"])
    parameter = str(selected_study.get("parameter") or "")
    expected_count = int(selected_study.get("count") or 0)
    expected_values = selected_study.get("values")
    baseline_index = int(selected_study.get("baseline_index") or 0)
    try:
        normalized_expected_values = (
            _sensitivity_values(expected_values, _parameter_specification(parameter))
            if isinstance(expected_values, list)
            else None
        )
    except (TypeError, ValueError):
        normalized_expected_values = None

    matches: list[
        tuple[
            int,
            Path,
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ]
    ] = []
    duplicate_indices: set[int] = set()
    seen_indices: set[int] = set()
    for case_json_path in case_path.parent.glob("*/case.json"):
        payload = _read_json_object(case_json_path)
        study = payload.get("sensitivity_study")
        if not isinstance(study, dict) or str(study.get("id") or "") != study_id:
            continue
        try:
            index = int(study.get("index"))
        except (TypeError, ValueError):
            continue
        if index in seen_indices:
            duplicate_indices.add(index)
        seen_indices.add(index)
        report = case_report(case_json_path.parent, include_validation=False)
        matches.append((index, case_json_path.parent, study, report, payload))
    matches.sort(key=lambda item: item[0])

    records = [
        _sensitivity_record(index, member_path, study, report, payload, parameter)
        for index, member_path, study, report, payload in matches
    ]
    complete = bool(
        expected_count >= 2
        and len(records) == expected_count
        and not duplicate_indices
        and [record["index"] for record in records] == list(range(expected_count))
    )
    metadata_verified = bool(
        matches
        and all(
            str(study.get("parameter") or "") == parameter
            and study.get("values") == expected_values
            and study.get("count") == expected_count
            and study.get("baseline_index") == baseline_index
            for _, _, study, _, _ in matches
        )
    )
    plan_hashes = {
        str(study.get("plan_lock_hash") or "") for _, _, study, _, _ in matches
    }
    computed_plan_hashes: set[str] = set()
    plan_lock_error = normalized_expected_values is None
    if normalized_expected_values is not None:
        for _, member_path, _, _, payload in matches:
            try:
                computed_plan_hashes.add(
                    _member_plan_lock(
                        payload,
                        member_path,
                        parameter,
                        normalized_expected_values,
                    )
                )
            except (TypeError, ValueError):
                plan_lock_error = True
    plan_locked = bool(
        not plan_lock_error
        and len(plan_hashes) == 1
        and "" not in plan_hashes
        and computed_plan_hashes == plan_hashes
    )
    values_match = bool(
        normalized_expected_values is not None
        and [record["value"] for record in records] == normalized_expected_values
        and all(record["valueMatchesCase"] for record in records)
    )
    all_numerically_qualified = bool(
        records and all(record["numericallyQualified"] for record in records)
    )
    all_statistically_ready = bool(
        records and all(record["statisticallyReady"] for record in records)
    )

    baseline = next(
        (record for record in records if record["index"] == baseline_index),
        None,
    )
    comparisons = [
        _statistical_difference(baseline, record)
        for record in records
        if baseline is not None
    ]
    controlled = complete and metadata_verified and plan_locked and values_match
    decision_safe = controlled and all_numerically_qualified and all_statistically_ready
    if not complete:
        status = "incomplete"
    elif not controlled:
        status = "plan_lock_mismatch"
    elif not all_numerically_qualified:
        status = "numerical_qualification_required"
    elif not all_statistically_ready:
        status = "statistical_evidence_required"
    else:
        status = "ready"

    return {
        "schemaVersion": 1,
        "studyId": study_id,
        "kind": "one_factor_sensitivity",
        "status": status,
        "decisionSafeSensitivity": decision_safe,
        "parameterControlled": controlled,
        "studyMetadataVerified": metadata_verified,
        "planLockVerified": plan_locked,
        "parameterValuesVerified": values_match,
        "parameter": parameter,
        "parameterLabel": selected_study.get("parameter_label"),
        "unit": selected_study.get("unit"),
        "values": expected_values,
        "baselineIndex": baseline_index,
        "planLockHash": next(iter(plan_hashes), None) if plan_locked else None,
        "complete": complete,
        "allNumericallyQualified": all_numerically_qualified,
        "allStatisticallyReady": all_statistically_ready,
        "records": records,
        "comparisonsToBaseline": comparisons,
        "interpretation": (
            "Numerical qualification and time-series statistical evidence are separate gates. "
            "A resolved confidence interval supports a difference for this controlled case family; "
            "it does not establish a universal physical effect outside the tested range."
        ),
    }


def _sensitivity_record(
    index: int,
    case_path: Path,
    study: dict[str, object],
    report: dict[str, object],
    payload: dict[str, object],
    parameter: str,
) -> dict[str, object]:
    assessment = report.get("qualityAssessment")
    numerically_qualified = bool(
        isinstance(assessment, dict)
        and assessment.get("numericallyQualified", assessment.get("trusted"))
    )
    statistics = report.get("transientStatistics")
    overall = statistics.get("overall_evidence") if isinstance(statistics, dict) else None
    statistically_ready = bool(
        isinstance(overall, dict)
        and overall.get("stationarity_supported") is True
        and overall.get("minimum_effective_samples_30") is True
        and overall.get("meaningful_peak_has_at_least_10_cycles") is not False
    )
    force_coeffs = report.get("forceCoeffs")
    channels = statistics.get("channels") if isinstance(statistics, dict) else None
    coefficient_evidence = {
        channel: _channel_evidence(channel, channels, force_coeffs)
        for channel in _COEFFICIENT_CHANNELS
    }
    balance = _channel_evidence("frontAeroBalancePercent", channels, None)
    recorded_value = _finite_number(study.get("value"))
    actual_value = _stored_parameter_value(payload, parameter)
    value_matches_case = bool(
        recorded_value is not None
        and actual_value is not None
        and math.isclose(recorded_value, actual_value, rel_tol=1e-9, abs_tol=1e-12)
    )
    return {
        "index": index,
        "value": recorded_value,
        "actualValue": actual_value,
        "valueMatchesCase": value_matches_case,
        "isBaseline": bool(study.get("is_baseline")),
        "casePath": str(case_path),
        "caseName": report.get("caseName", case_path.name),
        "numericallyQualified": numerically_qualified,
        "qualificationStatus": report.get("qualificationStatus"),
        "statisticallyReady": statistically_ready,
        "statisticalEvidence": overall,
        "coefficientEvidence": coefficient_evidence,
        "aeroBalanceEvidence": balance,
    }


def _channel_evidence(
    channel: str,
    channels: object,
    force_coeffs: object,
) -> dict[str, object]:
    payload = channels.get(channel) if isinstance(channels, dict) else None
    if isinstance(payload, dict):
        interval = payload.get("confidence_interval")
        stationarity = payload.get("stationarity_evidence")
        spectrum = payload.get("spectrum")
        return {
            "mean": payload.get("mean"),
            "standardError": payload.get("standard_error"),
            "effectiveSampleCount": payload.get("effective_sample_count"),
            "confidenceLower": interval.get("lower") if isinstance(interval, dict) else None,
            "confidenceUpper": interval.get("upper") if isinstance(interval, dict) else None,
            "stationarityStatus": (
                stationarity.get("status") if isinstance(stationarity, dict) else None
            ),
            "stationaritySupported": (
                stationarity.get("supports_stationarity")
                if isinstance(stationarity, dict)
                else None
            ),
            "dominantFrequencyHz": (
                spectrum.get("dominant_frequency_hz") if isinstance(spectrum, dict) else None
            ),
            "cycleCoverage": spectrum.get("cycle_coverage") if isinstance(spectrum, dict) else None,
            "strouhalNumber": spectrum.get("strouhal_number") if isinstance(spectrum, dict) else None,
        }
    mean = None
    if isinstance(force_coeffs, dict):
        mean = force_coeffs.get(f"mean{channel}", force_coeffs.get(channel))
    return {
        "mean": mean,
        "standardError": None,
        "effectiveSampleCount": None,
        "confidenceLower": None,
        "confidenceUpper": None,
        "stationarityStatus": None,
        "stationaritySupported": None,
        "dominantFrequencyHz": None,
        "cycleCoverage": None,
        "strouhalNumber": None,
    }


def _statistical_difference(
    baseline: dict[str, object] | None,
    variant: dict[str, object],
) -> dict[str, object]:
    coefficient_differences: dict[str, object] = {}
    if baseline is not None:
        baseline_channels = baseline.get("coefficientEvidence")
        variant_channels = variant.get("coefficientEvidence")
        for channel in _COEFFICIENT_CHANNELS:
            baseline_channel = (
                baseline_channels.get(channel) if isinstance(baseline_channels, dict) else None
            )
            variant_channel = (
                variant_channels.get(channel) if isinstance(variant_channels, dict) else None
            )
            coefficient_differences[channel] = _difference_evidence(
                baseline_channel,
                variant_channel,
            )
    return {
        "index": variant.get("index"),
        "value": variant.get("value"),
        "isBaseline": variant.get("isBaseline"),
        "coefficientDifferences": coefficient_differences,
        "aeroBalanceDifference": _difference_evidence(
            baseline.get("aeroBalanceEvidence") if isinstance(baseline, dict) else None,
            variant.get("aeroBalanceEvidence"),
        ),
    }


def _difference_evidence(baseline: object, variant: object) -> dict[str, object]:
    baseline_mean = _mapping_number(baseline, "mean")
    variant_mean = _mapping_number(variant, "mean")
    baseline_error = _mapping_number(baseline, "standardError")
    variant_error = _mapping_number(variant, "standardError")
    if baseline_mean is None or variant_mean is None:
        return {
            "delta": None,
            "standardError": None,
            "confidenceLower": None,
            "confidenceUpper": None,
            "statisticallyResolved": None,
        }
    delta = variant_mean - baseline_mean
    combined_error = (
        math.hypot(baseline_error, variant_error)
        if baseline_error is not None and variant_error is not None
        else None
    )
    margin = NormalDist().inv_cdf(0.975) * combined_error if combined_error is not None else None
    lower = delta - margin if margin is not None else None
    upper = delta + margin if margin is not None else None
    resolved = bool(lower > 0 or upper < 0) if lower is not None and upper is not None else None
    return {
        "delta": delta,
        "standardError": combined_error,
        "confidenceLower": lower,
        "confidenceUpper": upper,
        "statisticallyResolved": resolved,
    }


def _mapping_number(payload: object, key: str) -> float | None:
    return _finite_number(payload.get(key)) if isinstance(payload, dict) else None


def _case_options_from_payload(
    payload: dict[str, object],
    base_case_path: Path,
) -> dict[str, object]:
    flow = _mapping(payload.get("flow"))
    units = _mapping(payload.get("units"))
    orientation = _mapping(payload.get("orientation"))
    rotation = _mapping(orientation.get("rotation_degrees"))
    ground = _mapping(payload.get("ground"))
    reference = _mapping(payload.get("aerodynamic_reference"))
    validation = _mapping(payload.get("geometry_validation"))
    measured = _mapping(validation.get("measured_dimensions_m"))
    mesh = _mapping(payload.get("mesh_resolution"))
    quality = _mapping(payload.get("cfd_quality"))
    physical = _mapping(payload.get("physical_model"))
    fluid = _mapping(physical.get("fluid"))
    inflow = _mapping(physical.get("inflow"))
    surface = _mapping(physical.get("surface"))
    domain = _mapping(physical.get("domain"))
    outlet = _mapping(physical.get("outlet"))
    road_and_wheels = _mapping(physical.get("road_and_wheels"))
    transient = _mapping(physical.get("transient"))
    turbulence = _mapping(physical.get("turbulence"))
    volume_zones = _mapping(physical.get("volume_zones"))
    datums = _mapping(payload.get("vehicle_datums"))

    model_value = payload.get("model")
    if not isinstance(model_value, str) or not model_value:
        raise ValueError("The base case does not record its source model path.")
    model_path = Path(model_value).expanduser()
    if not model_path.is_absolute():
        model_path = base_case_path / model_path

    property_source = str(fluid.get("property_source") or "")
    use_weather = property_source != "legacy_standard_air_reference"
    use_manual_air = property_source == "manual_override"
    intensity_source = str(inflow.get("turbulence_intensity_source") or "")
    length_source = str(inflow.get("turbulence_length_scale_source") or "")
    yaw_source = str(inflow.get("yaw_crosswind_source") or "")
    yaw = _finite_number(inflow.get("yaw_degrees"))
    crosswind = _finite_number(inflow.get("crosswind_mps"))
    if yaw_source == "crosswind":
        yaw = None
    elif yaw_source == "validated_yaw_and_crosswind":
        pass
    elif yaw_source == "yaw" or (not yaw_source and yaw not in (None, 0.0)):
        crosswind = None
    else:
        yaw = None
        crosswind = None

    simulation_mode = str(
        transient.get("mode")
        or quality.get("simulation_mode")
        or str(payload.get("simulation_type") or "steady").split("_", 1)[0]
    )
    stored_closed_tunnel = (
        domain.get("closed_tunnel") if domain.get("mode") == "closed_tunnel" else None
    )
    closed_tunnel = (
        {
            key: _finite_number(stored_closed_tunnel.get(key))
            for key in ("width_m", "height_m", "upstream_m", "downstream_m")
        }
        if isinstance(stored_closed_tunnel, dict)
        else None
    )
    return {
        "model_path": model_path.resolve(),
        "speed_mph": _required_number(flow.get("speed_mph"), "base-case speed"),
        "flow_axis": str(flow.get("axis") or orientation.get("target_flow_axis") or "x"),
        "include_ground": bool(ground.get("enabled")),
        "moving_ground": bool(ground.get("moving")),
        "ground_clearance_m": _finite_number(ground.get("clearance_m")) or 0.0,
        "unit_scale": _finite_number(units.get("scale_to_meters")) or 1.0,
        "unit_label": str(units.get("input_units") or "meters"),
        "reference_area_m2": (
            _finite_number(reference.get("area_m2"))
            if reference.get("area_source") == "manual"
            else None
        ),
        "reference_length_m": (
            _finite_number(reference.get("length_m"))
            if reference.get("length_source") == "manual"
            else None
        ),
        "measured_length_m": _finite_number(measured.get("length_m")),
        "measured_width_m": _finite_number(measured.get("width_m")),
        "measured_height_m": _finite_number(measured.get("height_m")),
        "smallest_aero_feature_m": _finite_number(mesh.get("smallest_aero_feature_m")),
        "quality": str(quality.get("name") or mesh.get("quality") or "standard"),
        "source_flow_direction": str(orientation.get("source_flow_direction") or "+x"),
        "source_up_direction": str(orientation.get("source_up_direction") or "+z"),
        "model_rotation_degrees": (
            _finite_number(rotation.get("x")) or 0.0,
            _finite_number(rotation.get("y")) or 0.0,
            _finite_number(rotation.get("z")) or 0.0,
        ),
        "simulation_mode": simulation_mode,
        "air_temperature_c": _finite_number(fluid.get("temperature_c")) if use_weather else None,
        "air_pressure_pa": _finite_number(fluid.get("pressure_pa")) if use_weather else None,
        "air_density_kg_m3": (
            _finite_number(fluid.get("density_kg_m3")) if use_manual_air else None
        ),
        "kinematic_viscosity_m2_s": (
            _finite_number(fluid.get("kinematic_viscosity_m2_s")) if use_manual_air else None
        ),
        "turbulence_intensity_percent": (
            _finite_number(inflow.get("turbulence_intensity_percent"))
            if intensity_source == "manual"
            else None
        ),
        "turbulence_length_scale_m": (
            _finite_number(inflow.get("turbulence_length_scale_m"))
            if length_source == "manual"
            else None
        ),
        "center_of_gravity_m": _optional_vector(datums.get("center_of_gravity_m")),
        "front_axle_station_m": _finite_number(datums.get("front_axle_station_m")),
        "rear_axle_station_m": _finite_number(datums.get("rear_axle_station_m")),
        "yaw_degrees": yaw,
        "crosswind_mps": crosswind,
        "roughness_height_m": _finite_number(surface.get("roughness_height_m")) or 0.0,
        "roughness_constant": _finite_number(surface.get("roughness_constant")) or 0.5,
        "closed_tunnel": dict(closed_tunnel) if isinstance(closed_tunnel, dict) else None,
        "backflow_safe_outlet": bool(outlet.get("backflow_safe")),
        "wheel_setup": _wheel_options(road_and_wheels.get("wheels")),
        "second_order_transient": bool(transient.get("second_order_temporal")),
        "fluid_profile": str(
            fluid.get("profile")
            or (
                "compressible_thermal"
                if payload.get("solver_module") == "fluid"
                else "incompressible"
            )
        ),
        "turbulence_model": str(turbulence.get("model") or "kOmegaSST"),
        "porous_zones": _stored_zone_options(volume_zones.get("porous_zones")),
        "fan_zones": _stored_zone_options(volume_zones.get("fan_zones")),
        "heat_zones": _stored_zone_options(volume_zones.get("heat_zones")),
    }


def _stored_parameter_value(payload: dict[str, object], parameter: str) -> float | None:
    flow = _mapping(payload.get("flow"))
    ground = _mapping(payload.get("ground"))
    physical = _mapping(payload.get("physical_model"))
    inflow = _mapping(physical.get("inflow"))
    surface = _mapping(physical.get("surface"))
    values = {
        "speed_mph": flow.get("speed_mph"),
        "yaw_degrees": inflow.get("yaw_degrees", flow.get("yaw_degrees")),
        "crosswind_mps": inflow.get("crosswind_mps", flow.get("crosswind_mps")),
        "roughness_height_m": surface.get("roughness_height_m"),
        "ground_clearance_m": ground.get("clearance_m"),
        "turbulence_intensity_percent": inflow.get("turbulence_intensity_percent"),
    }
    return _finite_number(values.get(parameter))


def _stored_zone_options(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[dict[str, object]] = []
    for index, zone_value in enumerate(value, start=1):
        if not isinstance(zone_value, dict):
            raise ValueError(f"Base-case volume zone {index} metadata is incomplete.")
        result.append(dict(zone_value))
    return result


def _wheel_options(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[dict[str, object]] = []
    for index, wheel_value in enumerate(value, start=1):
        wheel = _mapping(wheel_value)
        model_path = wheel.get("model_path")
        center = _optional_vector(wheel.get("source_center"))
        axis = _optional_vector(wheel.get("source_axis"))
        radius = _finite_number(wheel.get("source_radius"))
        if not model_path or center is None or axis is None or radius is None:
            raise ValueError(f"Base-case wheel {index} metadata is incomplete.")
        result.append(
            {
                "name": str(wheel.get("name") or f"wheel{index}"),
                "model_path": str(model_path),
                "center_source": center,
                "axis_source": axis,
                "radius_source": radius,
                "surface_speed_mps": _finite_number(wheel.get("surface_speed_mps")),
            }
        )
    return result


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_vector(value: object) -> tuple[float, float, float] | None:
    mapping = _mapping(value)
    components = tuple(_finite_number(mapping.get(axis)) for axis in ("x", "y", "z"))
    if any(component is None for component in components):
        return None
    return components  # type: ignore[return-value]


def _required_number(value: object, label: str) -> float:
    number = _finite_number(value)
    if number is None:
        raise ValueError(f"The base case does not record a finite {label}.")
    return number


def _parameter_specification(parameter: str) -> dict[str, object]:
    specification = SENSITIVITY_PARAMETERS.get(str(parameter))
    if specification is None:
        supported = ", ".join(sorted(SENSITIVITY_PARAMETERS))
        raise ValueError(f"Unsupported sensitivity parameter {parameter!r}; choose one of: {supported}.")
    return specification


def _sensitivity_values(
    values: list[float],
    specification: dict[str, object],
) -> list[float]:
    if not isinstance(values, list) or not 2 <= len(values) <= 12:
        raise ValueError("A sensitivity study requires between 2 and 12 parameter values.")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Sensitivity values must be finite numbers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Sensitivity values must be unique.")
    minimum = _finite_number(specification.get("minimum"))
    maximum = _finite_number(specification.get("maximum"))
    minimum_inclusive = _finite_number(specification.get("minimum_inclusive"))
    maximum_inclusive = _finite_number(specification.get("maximum_inclusive"))
    for value in normalized:
        if minimum is not None and value <= minimum:
            raise ValueError(f"Sensitivity values must be greater than {minimum}.")
        if maximum is not None and value >= maximum:
            raise ValueError(f"Sensitivity values must be lower than {maximum}.")
        if minimum_inclusive is not None and value < minimum_inclusive:
            raise ValueError(f"Sensitivity values must be at least {minimum_inclusive}.")
        if maximum_inclusive is not None and value > maximum_inclusive:
            raise ValueError(f"Sensitivity values must be no more than {maximum_inclusive}.")
    return normalized


def _default_baseline_index(values: list[float], base_value: float | None) -> int:
    if base_value is None:
        return len(values) // 2
    return min(range(len(values)), key=lambda index: abs(values[index] - base_value))


def _study_plan_lock(
    base_options: dict[str, object],
    parameter: str,
    values: list[float],
) -> str:
    canonical_options = {
        key: _json_safe(value)
        for key, value in base_options.items()
        if key not in {parameter, "validation_study", "sensitivity_study"}
    }
    payload = {
        "schema_version": 1,
        "kind": "one_factor_sensitivity",
        "parameter": parameter,
        "values": values,
        "base_options": canonical_options,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _member_plan_lock(
    payload: dict[str, object],
    case_path: Path,
    parameter: str,
    values: list[float],
) -> str:
    options = _case_options_from_payload(payload, case_path)
    return _study_plan_lock(options, parameter, values)


def _persist_study_plan_lock(
    case_paths: list[Path],
    parameter: str,
    values: list[float],
) -> str:
    members: list[tuple[Path, dict[str, object], dict[str, object]]] = []
    locks: set[str] = set()
    for case_path in case_paths:
        payload = _read_json_object(case_path / "case.json")
        study = payload.get("sensitivity_study")
        if not payload or not isinstance(study, dict):
            raise RuntimeError(f"Generated sensitivity member {case_path} has incomplete metadata.")
        locks.add(_member_plan_lock(payload, case_path, parameter, values))
        members.append((case_path, payload, study))
    if len(locks) != 1:
        raise RuntimeError(
            "Generated sensitivity members do not preserve one common non-varied setup."
        )

    plan_lock = next(iter(locks))
    for case_path, payload, study in members:
        payload["sensitivity_study"] = {
            **study,
            "plan_lock_hash": plan_lock,
            "plan_lock_scope": "reconstructed_non_varied_case_options",
        }
        (case_path / "case.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return plan_lock


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Sensitivity-study options must be finite.")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def _value_token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p").replace("+", "")
