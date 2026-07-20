from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from .. import __version__
from ..case import create_case
from ..solver import run_case, solver_status
from ..solver.backends import OPENFOAM_EXECUTABLES, openfoam_identity

BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_RESULT_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_ID = "cube-symmetry-v1"
BENCHMARK_UNAVAILABLE_EXIT_CODE = 3
_BENCHMARK_IDS = (DEFAULT_BENCHMARK_ID,)
_INFRASTRUCTURE_RETURN_CODES = {90, 91, 92, 96, 97, 125, 126, 127}


def available_benchmarks() -> tuple[str, ...]:
    return _BENCHMARK_IDS


def load_benchmark_manifest(benchmark_id: str) -> dict[str, object]:
    manifest, _, _, _, _ = _load_definition(benchmark_id)
    return manifest


def run_benchmark(
    benchmark_id: str = DEFAULT_BENCHMARK_ID,
    *,
    output_dir: Path = Path("outputs/benchmarks"),
    backend: str = "auto",
    timeout_seconds: int = 3600,
    prepare_only: bool = False,
) -> dict[str, object]:
    if backend not in {"auto", "native", "wsl", "docker"}:
        raise ValueError(f"Unsupported benchmark backend: {backend}")
    if timeout_seconds <= 0:
        raise ValueError("Benchmark timeout must be positive.")

    manifest, manifest_hash, manifest_bytes, fixture_bytes, fixture_hash = _load_definition(
        benchmark_id
    )
    started_at = datetime.now(timezone.utc)
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_root = output_dir.expanduser().resolve() / benchmark_id / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "benchmark-result.json"
    checks: list[dict[str, object]] = []
    result = _base_result(
        manifest=manifest,
        benchmark_id=benchmark_id,
        run_id=run_id,
        run_root=run_root,
        case_path=None,
        result_path=result_path,
        manifest_hash=manifest_hash,
        fixture_hash=fixture_hash,
        started_at=started_at,
        checks=checks,
    )

    input_dir = run_root / "input"
    fixture_path = input_dir / "body.stl"
    try:
        (run_root / "manifest.json").write_bytes(manifest_bytes)
        input_dir.mkdir()
        fixture_path.write_bytes(fixture_bytes)
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="Attempt setup",
            stage="attemptSetup",
            exc=exc,
        )
    checks.append(
        _check(
            "Fixture integrity",
            True,
            f"Packaged fixture SHA-256 matches {fixture_hash}.",
        )
    )

    try:
        case_config = _mapping(manifest, "case")
        cases_dir = run_root / "cases"
        case_path = create_case(
            model_path=fixture_path,
            case_name=str(case_config["name"]),
            speed_mph=float(case_config["speedMph"]),
            flow_axis=str(case_config["flowAxis"]),
            cases_dir=cases_dir,
            include_ground=bool(case_config.get("includeGround", False)),
            moving_ground=bool(case_config.get("movingGround", False)),
            quality=str(case_config["quality"]),
            simulation_mode=str(case_config["simulationMode"]),
            unit_scale=float(case_config["unitScale"]),
            unit_label=str(case_config["unitLabel"]),
            reference_area_m2=float(case_config["referenceAreaM2"]),
            reference_length_m=float(case_config["referenceLengthM"]),
            measured_length_m=float(case_config["measuredLengthM"]),
            measured_width_m=float(case_config["measuredWidthM"]),
            measured_height_m=float(case_config["measuredHeightM"]),
            air_temperature_c=float(case_config["airTemperatureC"]),
            air_pressure_pa=float(case_config["airPressurePa"]),
            air_density_kg_m3=float(case_config["airDensityKgM3"]),
            kinematic_viscosity_m2_s=float(case_config["kinematicViscosityM2S"]),
            turbulence_intensity_percent=float(case_config["turbulenceIntensityPercent"]),
            turbulence_length_scale_m=float(case_config["turbulenceLengthScaleM"]),
            source_flow_direction=str(case_config["sourceFlowDirection"]),
            source_up_direction=str(case_config["sourceUpDirection"]),
        )
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="Case generation",
            stage="caseGeneration",
            exc=exc,
        )

    _record_case_artifacts(result, case_path)
    checks.append(
        _check(
            "Case generation",
            (case_path / "case.json").is_file() and (case_path / "Allrun").is_file(),
            "A fresh benchmark case was generated from the packaged fixture.",
        )
    )

    if prepare_only:
        checks.append(
            _pending_check(
                "Real OpenFOAM execution",
                "The deterministic case is prepared; run the benchmark on a Foundation v13 backend.",
            )
        )
        result.update(
            {
                "status": "prepared",
                "passed": False,
                "solverAvailable": None,
                "nextAction": _next_action(benchmark_id, output_dir, backend, timeout_seconds),
            }
        )
        return _write_result(result, result_path)

    try:
        status = solver_status()
        selected_backend = _selected_backend(status, backend)
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="OpenFOAM backend discovery",
            stage="backendDiscovery",
            exc=exc,
        )
    if selected_backend is None:
        infrastructure_error = _backend_infrastructure_error(status, backend)
        if infrastructure_error:
            return _write_error_result(
                result,
                result_path,
                checks,
                label="OpenFOAM backend discovery",
                stage="backendDiscovery",
                exc=RuntimeError(infrastructure_error),
            )
        checks.append(
            _check(
                "OpenFOAM backend",
                False,
                "No local OpenFOAM backend is available; install Foundation v13 or configure an existing local image.",
            )
        )
        result.update(
            {
                "status": "unavailable",
                "passed": False,
                "solverAvailable": False,
                "requestedBackend": backend,
            }
        )
        return _write_result(result, result_path)

    try:
        solver_config = _mapping(manifest, "solver")
        expected_version = str(solver_config["majorVersion"])
        identity = openfoam_identity(selected_backend)
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="OpenFOAM identity",
            stage="identityProbe",
            exc=exc,
        )
    actual_version_value = identity.get("version")
    actual_version = str(actual_version_value) if actual_version_value is not None else None
    executable_hash = identity.get("executableSha256")
    toolchain_verified = _toolchain_identity_complete(identity)
    probe_succeeded = bool(
        identity.get("probeReturncode") == 0
        and identity.get("executable")
        and isinstance(executable_hash, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", executable_hash)
        and toolchain_verified
    )
    foundation_verified = identity.get("distribution") == "foundation"
    versions_agree = identity.get("versionsAgree") is True
    docker_digest_verified = bool(
        selected_backend != "docker" or identity.get("imageId")
    )
    image_detail = (
        f" and immutable image ID {identity.get('imageId')}"
        if identity.get("imageId")
        else ""
    )
    identity_matches = bool(
        probe_succeeded
        and foundation_verified
        and versions_agree
        and actual_version == expected_version
        and docker_digest_verified
    )
    checks.extend(
        (
            _check(
                "OpenFOAM backend",
                True,
                f"Using the available {selected_backend} backend.",
            ),
            _check(
                "OpenFOAM identity",
                identity_matches,
                (
                    f"Attested Foundation v{actual_version} across {len(OPENFOAM_EXECUTABLES)} OpenFOAM executables "
                    f"including {identity.get('executable')} at SHA-256 {executable_hash}{image_detail}; "
                    "execution is pinned to the complete recorded toolchain."
                    if identity_matches
                    else f"Expected an execution-pinned Foundation v{expected_version} environment with complete provenance; "
                    f"detected distribution {identity.get('distribution') or 'unknown'}, version "
                    f"{actual_version or 'unknown'}, environment version "
                    f"{identity.get('environmentVersion') or 'unset'}, complete toolchain "
                    f"{'yes' if toolchain_verified else 'no'}, executable "
                    f"{identity.get('executable') or 'unknown'}, executable SHA-256 "
                    f"{executable_hash or 'unknown'}, and image ID "
                    f"{identity.get('imageId') or 'not applicable/unavailable'}.",
                ),
            ),
        )
    )
    result.update(
        {
            "solverAvailable": True,
            "backend": selected_backend,
            "openfoamVersion": actual_version,
            "expectedOpenfoamVersion": expected_version,
            "openfoamIdentity": identity,
        }
    )
    if not identity_matches:
        result.update({"status": "failed", "passed": False})
        return _write_result(result, result_path)

    try:
        solver_result = run_case(
            case_path,
            backend=selected_backend,
            timeout_seconds=timeout_seconds,
            run_mode="full",
            reuse_mesh=False,
            solver_identity=identity,
        )
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="Real OpenFOAM execution",
            stage="solverExecution",
            exc=exc,
        )
    if solver_result.returncode in _INFRASTRUCTURE_RETURN_CODES:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="Real OpenFOAM execution",
            stage="solverInfrastructure",
            exc=RuntimeError(
                f"The {selected_backend} execution environment returned infrastructure code "
                f"{solver_result.returncode}; see {solver_result.log_path}."
            ),
        )

    try:
        evaluation = evaluate_benchmark(manifest, solver_result.to_dict())
    except Exception as exc:
        return _write_error_result(
            result,
            result_path,
            checks,
            label="Benchmark evaluation",
            stage="evaluation",
            exc=exc,
        )
    checks.extend(evaluation["checks"])
    passed = all(check.get("status") == "pass" for check in checks)
    result.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metrics": evaluation["metrics"],
            "qualification": evaluation["qualification"],
            "run": {
                "ok": solver_result.ok,
                "returncode": solver_result.returncode,
                "backend": solver_result.backend,
                "reusedMesh": solver_result.reused_mesh,
                "startedAt": solver_result.started_at,
                "finishedAt": solver_result.finished_at,
                "logPath": str(solver_result.log_path),
            },
        }
    )
    return _write_result(result, result_path)


def evaluate_benchmark(
    manifest: dict[str, object],
    run_payload: dict[str, object],
) -> dict[str, object]:
    report = run_payload.get("report")
    if not isinstance(report, dict):
        report = {}
    acceptance = _mapping(manifest, "acceptance")
    checks: list[dict[str, object]] = []
    checks.append(
        _check(
            "Real OpenFOAM execution",
            bool(run_payload.get("ok")) and run_payload.get("returncode") == 0,
            (
                "OpenFOAM completed the generated mesh, solve, and post-processing pipeline."
                if run_payload.get("ok") and run_payload.get("returncode") == 0
                else f"OpenFOAM returned {run_payload.get('returncode')}."
            ),
        )
    )

    assessment = report.get("qualityAssessment")
    assessment = assessment if isinstance(assessment, dict) else {}
    numerically_qualified = bool(
        assessment.get("numericallyQualified", assessment.get("trusted"))
    )
    checks.append(
        _check(
            "Numerical qualification",
            numerically_qualified,
            (
                "The run passed AeroLab's complete numerical qualification."
                if numerically_qualified
                else "The run did not pass AeroLab's complete numerical qualification."
            ),
        )
    )

    assessment_checks = assessment.get("checks")
    indexed_checks = {
        str(item.get("label")): item
        for item in assessment_checks
        if isinstance(item, dict) and item.get("label")
    } if isinstance(assessment_checks, list) else {}
    required_checks = acceptance.get("requiredQualificationChecks")
    if not isinstance(required_checks, list):
        raise ValueError("Benchmark acceptance requires a requiredQualificationChecks list.")
    for label_value in required_checks:
        label = str(label_value)
        source = indexed_checks.get(label)
        passed = bool(source and source.get("status") == "pass")
        detail = (
            str(source.get("detail"))
            if source
            else f"Required AeroLab qualification check {label!r} is missing."
        )
        checks.append(_check(f"Qualification: {label}", passed, detail))

    metrics: dict[str, float | None] = {}
    metric_specs = acceptance.get("metrics")
    if not isinstance(metric_specs, list):
        raise ValueError("Benchmark acceptance requires a metrics list.")
    for metric_spec in metric_specs:
        if not isinstance(metric_spec, dict):
            raise ValueError("Each benchmark metric must be an object.")
        path = str(metric_spec["path"])
        label = str(metric_spec.get("label") or path)
        value = _finite_number(_nested_value(report, path))
        metrics[path] = value
        passed, expectation = _metric_passes(value, metric_spec)
        actual = "missing" if value is None else f"{value:.9g}"
        checks.append(
            _check(
                f"Metric: {label}",
                passed,
                f"Observed {actual}; required {expectation}.",
            )
        )

    return {
        "checks": checks,
        "metrics": metrics,
        "qualification": {
            "numericallyQualified": numerically_qualified,
            "qualificationStatus": assessment.get("qualificationStatus"),
            "checks": assessment_checks if isinstance(assessment_checks, list) else [],
        },
    }


def _load_definition(
    benchmark_id: str,
) -> tuple[dict[str, object], str, bytes, bytes, str]:
    if benchmark_id not in _BENCHMARK_IDS:
        raise ValueError(
            f"Unknown benchmark {benchmark_id!r}; choose one of {', '.join(_BENCHMARK_IDS)}."
        )
    root = resources.files("aerolab.benchmarks").joinpath(benchmark_id)
    manifest_bytes = root.joinpath("manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Benchmark manifest must contain a JSON object.")
    if manifest.get("schemaVersion") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark manifest schema version.")
    if manifest.get("id") != benchmark_id:
        raise ValueError("Benchmark manifest ID does not match its package path.")
    fixture = _mapping(manifest, "fixture")
    fixture_name = str(fixture["path"])
    if Path(fixture_name).name != fixture_name:
        raise ValueError("Benchmark fixture must be a case-local filename.")
    fixture_bytes = root.joinpath(fixture_name).read_bytes()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    expected_hash = str(fixture["sha256"])
    if fixture_hash != expected_hash:
        raise ValueError(
            f"Benchmark fixture integrity failed: expected {expected_hash}, found {fixture_hash}."
        )
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, manifest_hash, manifest_bytes, fixture_bytes, fixture_hash


def _base_result(
    *,
    manifest: dict[str, object],
    benchmark_id: str,
    run_id: str,
    run_root: Path,
    case_path: Path | None,
    result_path: Path,
    manifest_hash: str,
    fixture_hash: str,
    started_at: datetime,
    checks: list[dict[str, object]],
) -> dict[str, object]:
    case_path_text = str(case_path) if case_path is not None else None
    return {
        "schemaVersion": BENCHMARK_RESULT_SCHEMA_VERSION,
        "benchmarkId": benchmark_id,
        "benchmarkName": manifest.get("name"),
        "benchmarkKind": manifest.get("kind"),
        "validationScope": manifest.get("validationScope"),
        "absoluteAccuracyValidated": bool(manifest.get("absoluteAccuracyValidated")),
        "runId": run_id,
        "applicationVersion": __version__,
        "buildRevision": os.environ.get("AEROLAB_BUILD_REVISION"),
        "manifestSha256": manifest_hash,
        "fixtureSha256": fixture_hash,
        "startedAt": started_at.isoformat(),
        "runRoot": str(run_root),
        "casePath": case_path_text,
        "resultPath": str(result_path),
        "checks": checks,
        "artifacts": {
            "manifest": str(run_root / "manifest.json"),
            "case": case_path_text,
            "caseMetadata": str(case_path / "case.json") if case_path else None,
            "runLog": str(case_path / "aerolab-run.log") if case_path else None,
            "runRecord": str(case_path / "aerolab-run.json") if case_path else None,
            "resultSeal": str(result_path.with_suffix(".sha256")),
        },
    }


def _record_case_artifacts(result: dict[str, object], case_path: Path) -> None:
    result["casePath"] = str(case_path)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Benchmark result artifacts must be an object.")
    artifacts.update(
        {
            "case": str(case_path),
            "caseMetadata": str(case_path / "case.json"),
            "runLog": str(case_path / "aerolab-run.log"),
            "runRecord": str(case_path / "aerolab-run.json"),
        }
    )


def _write_error_result(
    result: dict[str, object],
    result_path: Path,
    checks: list[dict[str, object]],
    *,
    label: str,
    stage: str,
    exc: Exception,
) -> dict[str, object]:
    checks.append(
        _check(
            label,
            False,
            f"{label} raised {type(exc).__name__}: {exc}",
        )
    )
    result.update(
        {
            "status": "error",
            "passed": False,
            "error": {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )
    return _write_result(result, result_path)


def _write_result(result: dict[str, object], result_path: Path) -> dict[str, object]:
    result["finishedAt"] = datetime.now(timezone.utc).isoformat()
    temporary_path = result_path.with_suffix(".json.tmp")
    seal_path = result_path.with_suffix(".sha256")
    run_root = Path(str(result["runRoot"]))
    evidence_hashes = _hash_tree(
        run_root,
        excluded={result_path.name, temporary_path.name, seal_path.name},
    )
    result["evidenceSha256"] = evidence_hashes
    result["evidenceFileCount"] = len(evidence_hashes)
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)
    seal_path.write_text(
        f"{_sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    return result


def _hash_tree(root: Path, *, excluded: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if (
            path.name in excluded
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _backend_infrastructure_error(
    status: dict[str, object],
    requested: str,
) -> str | None:
    backends = status.get("backends")
    if not isinstance(backends, dict):
        return None
    candidates = (requested,) if requested != "auto" else ("native", "wsl", "docker")
    for backend in candidates:
        backend_status = backends.get(backend)
        if not isinstance(backend_status, dict):
            continue
        error = backend_status.get("infrastructureError")
        if not isinstance(error, dict):
            continue
        stage = error.get("stage") or "probe"
        message = error.get("message") or "unknown infrastructure failure"
        return f"{backend} {stage} failed: {message}"
    return None


def _toolchain_identity_complete(identity: dict[str, object]) -> bool:
    toolchain = identity.get("toolchain")
    if not isinstance(toolchain, dict):
        return False
    for executable in OPENFOAM_EXECUTABLES:
        entry = toolchain.get(executable)
        if not isinstance(entry, dict):
            return False
        path = entry.get("path")
        executable_hash = entry.get("sha256")
        if not isinstance(path, str) or not path:
            return False
        if not isinstance(executable_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}",
            executable_hash,
        ):
            return False
    foam_run = toolchain.get("foamRun")
    return bool(
        isinstance(foam_run, dict)
        and identity.get("executable") == foam_run.get("path")
        and identity.get("executableSha256") == foam_run.get("sha256")
    )


def _selected_backend(status: dict[str, object], requested: str) -> str | None:
    if requested == "auto":
        preferred = status.get("preferredBackend")
        return preferred if isinstance(preferred, str) and preferred else None
    backends = status.get("backends")
    if not isinstance(backends, dict):
        return None
    selected = backends.get(requested)
    return requested if isinstance(selected, dict) and selected.get("available") else None


def _mapping(payload: dict[str, object], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark manifest requires an object at {key!r}.")
    return value


def _nested_value(payload: dict[str, object], path: str) -> object:
    value: object = payload
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_passes(
    value: float | None,
    specification: dict[str, object],
) -> tuple[bool, str]:
    expectations: list[str] = []
    passed = value is not None
    if "minimum" in specification:
        minimum = float(specification["minimum"])
        expectations.append(f">= {minimum:.9g}")
        passed = passed and value is not None and value >= minimum
    if "maximum" in specification:
        maximum = float(specification["maximum"])
        expectations.append(f"<= {maximum:.9g}")
        passed = passed and value is not None and value <= maximum
    if "absoluteMaximum" in specification:
        maximum = float(specification["absoluteMaximum"])
        expectations.append(f"absolute value <= {maximum:.9g}")
        passed = passed and value is not None and abs(value) <= maximum
    if not expectations:
        raise ValueError("Benchmark metric must define minimum, maximum, or absoluteMaximum.")
    return passed, " and ".join(expectations)


def _check(label: str, passed: bool, detail: str) -> dict[str, object]:
    return {
        "label": label,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _pending_check(label: str, detail: str) -> dict[str, object]:
    return {"label": label, "status": "pending", "detail": detail}


def _next_action(
    benchmark_id: str,
    output_dir: Path,
    backend: str,
    timeout_seconds: int,
) -> str:
    return (
        f"aerolab benchmark {benchmark_id} --output-dir {str(output_dir)!r} "
        f"--backend {backend} --timeout-seconds {timeout_seconds}"
    )
