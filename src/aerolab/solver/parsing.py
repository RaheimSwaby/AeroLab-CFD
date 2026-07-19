"""Parsers for OpenFOAM solver output (force coeffs, residuals, y+, checkMesh, VTK)."""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from .util import _finite_number, _percentile, _read_json_object


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
