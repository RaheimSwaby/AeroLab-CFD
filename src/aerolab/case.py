from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .openfoam import (
    estimate_reference_area,
    estimate_reference_length,
    generate_openfoam_case,
    mesh_resolution_metadata,
    simulation_quality_metadata,
    wall_layer_metadata,
)
from .repair import is_prepared_model_path, repair_fidelity_for_model
from .stl import inspect_stl, transformed_report, translated_report


GEOMETRY_DIMENSION_TOLERANCE = 0.02
STANDARD_AIR_DENSITY_KG_M3 = 1.225
STANDARD_KINEMATIC_VISCOSITY_M2_S = 1.5e-5
STANDARD_SPEED_OF_SOUND_MPS = 343.0
MAXIMUM_INCOMPRESSIBLE_MACH = 0.3


def create_case(
    model_path: Path,
    case_name: str,
    speed_mph: float,
    flow_axis: str,
    cases_dir: Path,
    include_ground: bool = False,
    moving_ground: bool = False,
    ground_clearance_m: float = 0.0,
    generate_openfoam: bool = True,
    unit_scale: float = 1.0,
    unit_label: str = "meters",
    reference_area_m2: float | None = None,
    reference_length_m: float | None = None,
    measured_length_m: float | None = None,
    measured_width_m: float | None = None,
    measured_height_m: float | None = None,
    smallest_aero_feature_m: float | None = None,
    quality: str = "standard",
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    model_rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    validation_study: dict[str, object] | None = None,
    simulation_mode: str = "steady",
) -> Path:
    speed_mph = float(speed_mph)
    if not math.isfinite(speed_mph) or speed_mph <= 0:
        raise ValueError("Air speed must be a finite positive value in mph.")
    speed_mps = speed_mph * 0.44704
    mach_number = speed_mps / STANDARD_SPEED_OF_SOUND_MPS
    if mach_number >= MAXIMUM_INCOMPRESSIBLE_MACH:
        maximum_mph = MAXIMUM_INCOMPRESSIBLE_MACH * STANDARD_SPEED_OF_SOUND_MPS / 0.44704
        raise ValueError(
            f"{speed_mph:.3g} mph is Mach {mach_number:.3f}; AeroLab's incompressible solver "
            f"is limited to below Mach {MAXIMUM_INCOMPRESSIBLE_MACH:.1f} "
            f"(about {maximum_mph:.0f} mph in standard air). Use a compressible CFD solver above this limit."
        )
    if include_ground and flow_axis.lower() == "z":
        raise ValueError("Ground runs require X or Y flow; Z flow uses the road plane as the inlet.")
    if moving_ground and not include_ground:
        raise ValueError("Moving ground requires the ground patch to be enabled.")
    ground_clearance_m = float(ground_clearance_m)
    if not math.isfinite(ground_clearance_m) or ground_clearance_m < 0:
        raise ValueError("Ground clearance must be a non-negative distance in meters.")
    if ground_clearance_m > 0 and not include_ground:
        raise ValueError("Ground clearance requires the ground patch to be enabled.")
    model_path = model_path.resolve()
    cases_dir = cases_dir.resolve()
    case_path, case_name = _available_case_path(cases_dir, case_name)

    raw_report = inspect_stl(model_path)
    if not raw_report.is_cfd_candidate:
        raise ValueError(
            "CFD case geometry must be watertight, manifold, and free of degenerate triangles. "
            "Use Prepare Scan to repair this model first."
        )
    repair_fidelity = repair_fidelity_for_model(model_path)
    if is_prepared_model_path(model_path) and not (repair_fidelity or {}).get("verified"):
        raise ValueError(
            "Prepared STL is missing a matching accepted repair-fidelity record; prepare the source scan again."
        )
    oriented_report = transformed_report(
        model_path,
        scale=unit_scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=flow_axis,
        rotation_degrees=model_rotation_degrees,
    )
    model_translation_m = (0.0, 0.0, 0.0)
    if include_ground:
        model_translation_m = (
            0.0,
            0.0,
            ground_clearance_m - oriented_report.bounds.minimum[2],
        )
    report = translated_report(oriented_report, model_translation_m)
    geometry_validation = validate_geometry_dimensions(
        report,
        flow_axis,
        measured_length_m=measured_length_m,
        measured_width_m=measured_width_m,
        measured_height_m=measured_height_m,
    )
    failed_dimension = next(
        (check for check in geometry_validation["checks"] if check["status"] == "fail"),
        None,
    )
    if failed_dimension:
        raise ValueError(str(failed_dimension["detail"]))
    if validation_study and not geometry_validation["verified"]:
        raise ValueError(
            "Accuracy studies require measured vehicle length, width, and height within 2% of the scaled STL."
        )
    if validation_study and (smallest_aero_feature_m is None or smallest_aero_feature_m <= 0):
        raise ValueError(
            "Accuracy studies require the smallest aerodynamic feature whose pressure effect matters."
        )
    auto_reference_area = estimate_reference_area(report, flow_axis)
    auto_reference_length = estimate_reference_length(report.bounds, flow_axis)
    effective_reference_area = reference_area_m2 or auto_reference_area
    effective_reference_length = reference_length_m or auto_reference_length
    wall_quality = "standard" if validation_study else quality
    mesh_resolution = mesh_resolution_metadata(
        report.bounds,
        flow_axis,
        include_ground,
        quality,
        smallest_aero_feature_m,
    )
    wall_resolution = wall_layer_metadata(speed_mps, effective_reference_length, wall_quality)
    quality_metadata = simulation_quality_metadata(
        quality,
        simulation_mode,
        speed_mps,
        effective_reference_length,
        float(mesh_resolution["estimated_surface_cell_m"]),
    )
    if smallest_aero_feature_m is not None and not mesh_resolution.get("supported"):
        required = mesh_resolution.get("required_surface_level")
        maximum = mesh_resolution.get("maximum_supported_surface_level")
        minimum_feature_mm = (
            float(mesh_resolution["estimated_surface_cell_m"])
            * float(mesh_resolution["minimum_cells_across_feature"])
            * 1000.0
        )
        raise ValueError(
            f"The requested {smallest_aero_feature_m * 1000.0:.3g} mm aero feature requires "
            f"surface level {required}, above AeroLab's local-device limit {maximum}. "
            f"Use a feature target of at least {minimum_feature_mm:.3g} mm or simplify/localize the geometry."
        )
    case_path.mkdir(parents=True, exist_ok=True)

    case = {
        "name": case_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "units": {
            "input_units": unit_label,
            "scale_to_meters": unit_scale,
        },
        "orientation": {
            "source_flow_direction": source_flow_direction,
            "source_up_direction": source_up_direction,
            "target_flow_axis": flow_axis,
            "rotation_degrees": {
                "x": model_rotation_degrees[0],
                "y": model_rotation_degrees[1],
                "z": model_rotation_degrees[2],
            },
        },
        "solver_target": "openfoam-foundation-v13",
        "cfd_quality": quality_metadata,
        "simulation_type": f"{simulation_mode}_external_incompressible_airflow",
        "solver_module": "incompressibleFluid",
        "flow": {
            "axis": flow_axis,
            "speed_mph": speed_mph,
            "speed_mps": speed_mps,
            "mach_number": mach_number,
            "maximum_supported_mach": MAXIMUM_INCOMPRESSIBLE_MACH,
            "speed_of_sound_mps": STANDARD_SPEED_OF_SOUND_MPS,
            "air_density_kg_m3": STANDARD_AIR_DENSITY_KG_M3,
            "kinematic_viscosity_m2_s": STANDARD_KINEMATIC_VISCOSITY_M2_S,
            "dynamic_pressure_pa": 0.5 * STANDARD_AIR_DENSITY_KG_M3 * speed_mps * speed_mps,
            "reynolds_number": speed_mps * effective_reference_length / STANDARD_KINEMATIC_VISCOSITY_M2_S,
        },
        "ground": {
            "enabled": include_ground,
            "moving": moving_ground,
            "clearance_m": ground_clearance_m if include_ground else 0.0,
            "road_elevation_m": 0.0 if include_ground else None,
            "lowest_model_z_m": report.bounds.minimum[2] if include_ground else None,
        },
        "placement": {
            "method": "lowest_point_to_road_clearance" if include_ground else "source_coordinates",
            "verified": True,
            "translation_m": {
                "x": model_translation_m[0],
                "y": model_translation_m[1],
                "z": model_translation_m[2],
            },
            "ground_clearance_m": ground_clearance_m if include_ground else None,
            "road_elevation_m": 0.0 if include_ground else None,
        },
        "aerodynamic_reference": {
            "area_m2": effective_reference_area,
            "length_m": effective_reference_length,
            "area_source": "manual" if reference_area_m2 else "auto_triangle_union_silhouette",
            "length_source": "manual" if reference_length_m else "auto_flow_axis",
            "auto_area_m2": auto_reference_area,
            "auto_length_m": auto_reference_length,
        },
        "wall_resolution": wall_resolution,
        "mesh_resolution": mesh_resolution,
        "geometry_report": raw_report.to_dict(),
        "geometry_fidelity": repair_fidelity
        or {
            "status": "original",
            "verified": True,
            "detail": "Solver geometry is the original checked STL, not an automatic repair.",
        },
        "geometry_validation": geometry_validation,
        "scaled_geometry_report": report.to_dict(),
        "status": "metadata_created",
        "next_steps": [
            "Run solver locally through WSL2 or Docker.",
            "Post-process drag, lift, pressure, and wake visualizations.",
        ],
    }
    if validation_study:
        case["validation_study"] = validation_study

    if generate_openfoam:
        files = generate_openfoam_case(
            case_path=case_path,
            model_path=model_path,
            report=report,
            speed_mps=speed_mps,
            flow_axis=flow_axis,
            include_ground=include_ground,
            moving_ground=moving_ground,
            model_scale=unit_scale,
            reference_area_m2=effective_reference_area,
            reference_length_m=effective_reference_length,
            quality=str(quality_metadata["name"]),
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            model_rotation_degrees=model_rotation_degrees,
            model_translation_m=model_translation_m,
            smallest_aero_feature_m=smallest_aero_feature_m,
            wall_quality=wall_quality,
            simulation_mode=simulation_mode,
        )
        case["status"] = "openfoam_case_generated"
        case["openfoam_files"] = [str(path.relative_to(case_path)) for path in files]

    with (case_path / "case.json").open("w", encoding="utf-8") as f:
        json.dump(case, f, indent=2)
        f.write("\n")

    return case_path


def _available_case_path(cases_dir: Path, requested_name: str) -> tuple[Path, str]:
    """Choose a new case directory without replacing prior solver evidence."""
    candidate = cases_dir / requested_name
    if not candidate.exists():
        return candidate, requested_name
    suffix = 2
    while True:
        candidate_name = f"{requested_name}-{suffix}"
        candidate = cases_dir / candidate_name
        if not candidate.exists():
            return candidate, candidate_name
        suffix += 1


def validate_geometry_dimensions(
    report: object,
    flow_axis: str,
    measured_length_m: float | None = None,
    measured_width_m: float | None = None,
    measured_height_m: float | None = None,
    tolerance: float = GEOMETRY_DIMENSION_TOLERANCE,
) -> dict[str, object]:
    dimensions = report.bounds.dimensions  # type: ignore[attr-defined]
    flow_index = {"x": 0, "y": 1, "z": 2}[flow_axis.lower()]
    up_index = 2 if flow_index != 2 else 1
    side_index = next(index for index in range(3) if index not in {flow_index, up_index})
    actual = {
        "length_m": float(dimensions[flow_index]),
        "width_m": float(dimensions[side_index]),
        "height_m": float(dimensions[up_index]),
    }
    measured = {
        "length_m": measured_length_m,
        "width_m": measured_width_m,
        "height_m": measured_height_m,
    }
    labels = {
        "length_m": "Vehicle length",
        "width_m": "Vehicle width",
        "height_m": "Vehicle height",
    }
    checks: list[dict[str, object]] = []
    for key in ("length_m", "width_m", "height_m"):
        expected = measured[key]
        if expected is None:
            checks.append(
                {
                    "label": labels[key],
                    "status": "pending",
                    "actual_m": actual[key],
                    "measured_m": None,
                    "error_percent": None,
                    "detail": f"Enter measured {labels[key].lower()} before final CFD validation.",
                }
            )
            continue
        if expected <= 0:
            checks.append(
                {
                    "label": labels[key],
                    "status": "fail",
                    "actual_m": actual[key],
                    "measured_m": expected,
                    "error_percent": None,
                    "detail": f"{labels[key]} must be greater than zero meters.",
                }
            )
            continue
        error = abs(actual[key] - expected) / expected
        passed = error <= tolerance
        checks.append(
            {
                "label": labels[key],
                "status": "pass" if passed else "fail",
                "actual_m": actual[key],
                "measured_m": expected,
                "error_percent": error * 100.0,
                "detail": (
                    f"Scaled STL {labels[key].lower()} {actual[key]:.4g} m versus measured {expected:.4g} m "
                    f"({error * 100.0:.2f}% error; limit {tolerance * 100.0:.1f}%)."
                ),
            }
        )
    failed = any(check["status"] == "fail" for check in checks)
    verified = all(check["status"] == "pass" for check in checks)
    return {
        "status": "verified" if verified else "failed" if failed else "incomplete",
        "verified": verified,
        "tolerance_percent": tolerance * 100.0,
        "actual_dimensions_m": actual,
        "measured_dimensions_m": measured,
        "checks": checks,
    }
