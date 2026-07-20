from __future__ import annotations

import json
import math
from pathlib import Path

from .stl import Bounds, StlReport, Vector, write_transformed_binary_stl

AXES = {"x": 0, "y": 1, "z": 2}
SIMULATION_MODES = {"steady", "transient"}

QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "draft": {
        "name": "draft",
        "description": "Fast setup for orientation, scale, and case smoke tests.",
        "tunnel_divisions": 28,
        "max_block_cells": 64,
        "max_local_cells": 75_000,
        "max_global_cells": 1_000_000,
        "adaptive_max_local_cells": 600_000,
        "adaptive_max_global_cells": 2_000_000,
        "max_supported_surface_level": 8,
        "n_cells_between_levels": 2,
        "feature_level": 2,
        "surface_min_level": 1,
        "surface_max_level": 3,
        "body_region_level": 2,
        "wake_region_level": 1,
        "surface_layers": 0,
        "target_y_plus": 100,
        "layer_expansion_ratio": 1.2,
        "snap_solve_iter": 20,
        "feature_snap_iter": 6,
        "end_time": 300,
        "write_interval": 75,
        "force_write_interval": 10,
        "pressure_residual_control": 1e-2,
        "velocity_residual_control": 1e-3,
        "turbulence_residual_control": 1e-3,
        "velocity_relaxation": 0.7,
    },
    "standard": {
        "name": "standard",
        "description": "Balanced mesh and iteration budget for normal local car checks.",
        "tunnel_divisions": 40,
        "max_block_cells": 80,
        "max_local_cells": 1_500_000,
        "max_global_cells": 2_800_000,
        "adaptive_max_local_cells": 2_000_000,
        "adaptive_max_global_cells": 2_800_000,
        "max_supported_surface_level": 8,
        "n_cells_between_levels": 3,
        "feature_level": 6,
        "surface_min_level": 4,
        "surface_max_level": 6,
        "body_region_level": 4,
        "wake_region_level": 3,
        "surface_layers": 5,
        "target_y_plus": 60,
        "layer_expansion_ratio": 1.2,
        "snap_solve_iter": 30,
        "feature_snap_iter": 10,
        "end_time": 1200,
        "write_interval": 200,
        "force_write_interval": 10,
        "pressure_residual_control": 3e-3,
        "velocity_residual_control": 5e-4,
        "turbulence_residual_control": 5e-4,
        "velocity_relaxation": 0.5,
    },
    "fine": {
        "name": "fine",
        "description": "Higher refinement and longer steady solve for more trustworthy comparisons.",
        "tunnel_divisions": 56,
        "max_block_cells": 120,
        "max_local_cells": 4_000_000,
        "max_global_cells": 32_000_000,
        "adaptive_max_local_cells": 6_000_000,
        "adaptive_max_global_cells": 32_000_000,
        "max_supported_surface_level": 9,
        "n_cells_between_levels": 4,
        "feature_level": 7,
        "surface_min_level": 5,
        "surface_max_level": 7,
        "body_region_level": 5,
        "wake_region_level": 4,
        "surface_layers": 8,
        "target_y_plus": 50,
        "layer_expansion_ratio": 1.18,
        "snap_solve_iter": 50,
        "feature_snap_iter": 15,
        "end_time": 2000,
        "write_interval": 250,
        "force_write_interval": 10,
        "pressure_residual_control": 1e-3,
        "velocity_residual_control": 2e-4,
        "turbulence_residual_control": 2e-4,
        "velocity_relaxation": 0.65,
    },
}


def quality_preset_metadata(quality: str = "standard") -> dict[str, object]:
    preset = _quality_preset(quality)
    return dict(preset)


def simulation_quality_metadata(
    quality: str,
    simulation_mode: str,
    speed_mps: float,
    reference_length_m: float,
    surface_cell_m: float,
) -> dict[str, object]:
    mode = str(simulation_mode or "steady").lower()
    if mode not in SIMULATION_MODES:
        raise ValueError(f"Unsupported simulation mode: {simulation_mode}")
    metadata = quality_preset_metadata(quality)
    metadata["simulation_mode"] = mode
    if mode == "steady":
        return metadata

    warmup_lengths, averaging_lengths = {
        "draft": (3.0, 3.0),
        "standard": (6.0, 8.0),
        "fine": (8.0, 12.0),
    }[str(metadata["name"])]
    speed = max(abs(float(speed_mps)), 0.1)
    length = max(float(reference_length_m), 1e-6)
    surface_cell = max(float(surface_cell_m), length * 1e-6)
    flow_through_time = length / speed
    warmup_time = warmup_lengths * flow_through_time
    averaging_window = averaging_lengths * flow_through_time
    end_time = warmup_time + averaging_window
    initial_delta_t = min(0.5 * surface_cell / speed, flow_through_time / 200.0)
    maximum_delta_t = min(2.0 * surface_cell / speed, flow_through_time / 50.0)
    metadata.update(
        {
            "end_time": end_time,
            "flow_through_time_s": flow_through_time,
            "warmup_flow_lengths": warmup_lengths,
            "averaging_flow_lengths": averaging_lengths,
            "warmup_time_s": warmup_time,
            "averaging_window_s": averaging_window,
            "initial_delta_t_s": initial_delta_t,
            "maximum_delta_t_s": maximum_delta_t,
            "maximum_courant_number": 1.5,
            "write_interval_s": max(end_time / 4.0, initial_delta_t),
            "force_write_interval": 1,
            "minimum_force_samples": 100,
            "transient_residual_ceiling": 0.2,
            "pimple_outer_correctors": 2,
            "pimple_pressure_correctors": 2,
            "estimated_initial_time_steps": math.ceil(end_time / initial_delta_t),
        }
    )
    return metadata


def mesh_resolution_metadata(
    bounds: Bounds,
    flow_axis: str,
    include_ground: bool,
    quality: str = "standard",
    smallest_aero_feature_m: float | None = None,
    domain: dict[str, object] | None = None,
) -> dict[str, object]:
    preset = _quality_preset(quality)
    tunnel = _wind_tunnel(bounds, flow_axis.lower(), include_ground, preset, domain)
    minimum = tunnel["min"]
    maximum = tunnel["max"]
    cells = tunnel["cells"]
    assert isinstance(minimum, tuple)
    assert isinstance(maximum, tuple)
    assert isinstance(cells, tuple)
    base_cell_dimensions = tuple(
        (maximum[index] - minimum[index]) / max(int(cells[index]), 1)
        for index in range(3)
    )
    conservative_base_cell = max(base_cell_dimensions)
    baseline_surface_level = int(preset["surface_max_level"])
    maximum_supported_level = int(preset["max_supported_surface_level"])
    target_cells_across = 4.0
    requested_surface_cell = (
        float(smallest_aero_feature_m) / target_cells_across
        if smallest_aero_feature_m is not None and smallest_aero_feature_m > 0
        else None
    )
    required_surface_level = baseline_surface_level
    if requested_surface_cell is not None:
        required_surface_level = max(
            baseline_surface_level,
            int(math.ceil(math.log2(conservative_base_cell / max(requested_surface_cell, 1e-12)) - 1e-12)),
        )
    supported = required_surface_level <= maximum_supported_level
    configured_surface_level = min(required_surface_level, maximum_supported_level)
    configured_feature_level = max(int(preset["feature_level"]), configured_surface_level)
    extra_levels = max(0, configured_surface_level - baseline_surface_level)
    configured_body_region_level = max(
        int(preset["wake_region_level"]),
        int(preset["body_region_level"]) - (1 if extra_levels >= 2 else 0),
    )
    configured_transition_levels = max(
        1,
        int(preset["n_cells_between_levels"]) - (1 if extra_levels >= 2 else 0),
    )
    budget_multiplier = 2 ** min(extra_levels, 3)
    configured_max_local_cells = min(
        int(preset["max_local_cells"]) * budget_multiplier,
        int(preset["adaptive_max_local_cells"]),
    )
    configured_max_global_cells = min(
        int(preset["max_global_cells"]) * budget_multiplier,
        int(preset["adaptive_max_global_cells"]),
    )
    surface_cell = conservative_base_cell / (2**configured_surface_level)
    broad_surface_cell = conservative_base_cell / (2 ** int(preset["surface_min_level"]))
    points_across = (
        float(smallest_aero_feature_m) / surface_cell
        if smallest_aero_feature_m is not None and smallest_aero_feature_m > 0
        else None
    )
    if points_across is None:
        status = "unset"
    elif not supported or points_across < target_cells_across - 1e-9:
        status = "fail"
    else:
        status = "pass"
    return {
        "quality": str(preset["name"]),
        "base_cell_dimensions_m": base_cell_dimensions,
        "baseline_surface_max_level": baseline_surface_level,
        "required_surface_level": required_surface_level if requested_surface_cell is not None else None,
        "configured_surface_min_level": int(preset["surface_min_level"]),
        "configured_surface_max_level": configured_surface_level,
        "configured_feature_level": configured_feature_level,
        "configured_body_region_level": configured_body_region_level,
        "configured_wake_region_level": int(preset["wake_region_level"]),
        "configured_n_cells_between_levels": configured_transition_levels,
        "maximum_supported_surface_level": maximum_supported_level,
        "configured_max_local_cells": configured_max_local_cells,
        "configured_max_global_cells": configured_max_global_cells,
        "cell_budget_multiplier": budget_multiplier,
        "adaptive_refinement": extra_levels > 0,
        "estimated_surface_cell_m": surface_cell,
        "estimated_broad_surface_cell_m": broad_surface_cell,
        "requested_surface_cell_m": requested_surface_cell,
        "smallest_aero_feature_m": smallest_aero_feature_m,
        "estimated_cells_across_feature": points_across,
        "minimum_cells_across_feature": target_cells_across,
        "supported": supported,
        "status": status,
        "estimate_only": True,
    }


def wall_layer_metadata(
    speed_mps: float,
    reference_length_m: float,
    quality: str = "standard",
    kinematic_viscosity_m2_s: float = 1.5e-5,
) -> dict[str, object]:
    preset = _quality_preset(quality)
    length = max(reference_length_m, 1e-6)
    speed = max(abs(speed_mps), 0.1)
    reynolds = speed * length / kinematic_viscosity_m2_s
    skin_friction = 0.026 / max(reynolds, 1.0) ** (1.0 / 7.0)
    friction_velocity = speed * math.sqrt(max(skin_friction, 1e-9) / 2.0)
    target_y_plus = float(preset["target_y_plus"])

    # firstLayerThickness spans the whole cell; y+ is evaluated near its centre.
    flat_plate_first_layer = 2.0 * target_y_plus * kinematic_viscosity_m2_s / max(friction_velocity, 1e-9)
    calibration_factor = 1.0
    first_layer = flat_plate_first_layer * calibration_factor
    first_layer = min(max(first_layer, length * 2e-6), length * 1e-3)
    flat_plate_estimated_y_plus = first_layer * 0.5 * friction_velocity / kinematic_viscosity_m2_s
    layer_count = int(preset["surface_layers"])
    expansion_ratio = float(preset["layer_expansion_ratio"])
    if layer_count > 0:
        total_thickness = first_layer * (expansion_ratio**layer_count - 1.0) / (expansion_ratio - 1.0)
    else:
        total_thickness = 0.0

    return {
        "mode": "high-y-plus-wall-functions",
        "target_y_plus": target_y_plus,
        "estimated_y_plus": target_y_plus,
        "flat_plate_estimated_y_plus": flat_plate_estimated_y_plus,
        "reynolds_number": reynolds,
        "estimated_skin_friction_coefficient": skin_friction,
        "estimated_friction_velocity_mps": friction_velocity,
        "flat_plate_first_layer_thickness_m": flat_plate_first_layer,
        "first_layer_calibration_factor": calibration_factor,
        "first_layer_thickness_m": first_layer,
        "total_layer_thickness_m": total_thickness,
        "surface_layers": layer_count,
        "expansion_ratio": expansion_ratio,
    }


def generate_openfoam_case(
    case_path: Path,
    model_path: Path,
    report: StlReport,
    speed_mps: float,
    flow_axis: str,
    include_ground: bool,
    moving_ground: bool,
    model_scale: float = 1.0,
    reference_area_m2: float | None = None,
    reference_length_m: float | None = None,
    quality: str = "standard",
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    model_rotation_degrees: Vector = (0.0, 0.0, 0.0),
    model_translation_m: Vector = (0.0, 0.0, 0.0),
    smallest_aero_feature_m: float | None = None,
    wall_quality: str | None = None,
    simulation_mode: str = "steady",
    air_density_kg_m3: float = 1.225,
    kinematic_viscosity_m2_s: float = 1.5e-5,
    turbulence_intensity: float = 0.01,
    turbulence_length_scale_m: float | None = None,
    force_reference_m: Vector | None = None,
    physical_model: dict[str, object] | None = None,
    body_rotation_center_source: Vector | None = None,
) -> list[Path]:
    flow_axis = flow_axis.lower()
    if flow_axis not in AXES:
        raise ValueError(f"Unsupported flow axis: {flow_axis}")
    if include_ground and flow_axis == "z":
        raise ValueError("Ground runs require X or Y flow; Z flow uses the road plane as the inlet.")
    preset = _quality_preset(quality)
    simulation_mode = str(simulation_mode or "steady").lower()
    if simulation_mode not in SIMULATION_MODES:
        raise ValueError(f"Unsupported simulation mode: {simulation_mode}")

    model_settings = physical_model or {}
    inflow_settings = _mapping_section(model_settings, "inflow")
    surface_settings = _mapping_section(model_settings, "surface")
    domain_settings = _mapping_section(model_settings, "domain")
    outlet_settings = _mapping_section(model_settings, "outlet")
    transient_settings = _mapping_section(model_settings, "transient")
    road_settings = _mapping_section(model_settings, "road_and_wheels")
    fluid_settings = _mapping_section(model_settings, "fluid")
    turbulence_settings = _mapping_section(model_settings, "turbulence")
    volume_zone_settings = _mapping_section(model_settings, "volume_zones")
    wheels_value = road_settings.get("wheels")
    wheels = [wheel for wheel in wheels_value if isinstance(wheel, dict)] if isinstance(wheels_value, list) else []
    porous_value = volume_zone_settings.get("porous_zones")
    porous_zones = (
        [zone for zone in porous_value if isinstance(zone, dict)]
        if isinstance(porous_value, list)
        else []
    )
    fan_value = volume_zone_settings.get("fan_zones")
    fan_zones = (
        [zone for zone in fan_value if isinstance(zone, dict)]
        if isinstance(fan_value, list)
        else []
    )
    heat_value = volume_zone_settings.get("heat_zones")
    heat_zones = (
        [zone for zone in heat_value if isinstance(zone, dict)]
        if isinstance(heat_value, list)
        else []
    )
    volume_zones = [*porous_zones, *fan_zones, *heat_zones]
    fluid_profile = str(fluid_settings.get("profile") or "incompressible")
    solver_module = "fluid" if fluid_profile == "compressible_thermal" else "incompressibleFluid"
    if heat_zones and solver_module != "fluid":
        raise ValueError("Heat-load zones require the compressible_thermal fluid profile.")
    turbulence_model = str(turbulence_settings.get("model") or "kOmegaSST")

    dirs = [
        case_path / "0",
        case_path / "constant",
        case_path / "constant" / "geometry",
        case_path / "system",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    for legacy_path in (
        case_path / "constant" / "transportProperties",
        case_path / "constant" / "turbulenceProperties",
        case_path / "system" / "surfaceFeatureExtractDict",
        case_path / "constant" / "triSurface" / "body.stl",
    ):
        if legacy_path.exists():
            legacy_path.unlink()

    body_stl = case_path / "constant" / "geometry" / "body.stl"
    write_transformed_binary_stl(
        model_path,
        body_stl,
        scale=model_scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=flow_axis,
        rotation_degrees=model_rotation_degrees,
        translation=model_translation_m,
    )
    wheel_stls: list[Path] = []
    if wheels and body_rotation_center_source is None:
        raise ValueError("Wheel geometry requires the shared body rotation center.")
    for wheel in wheels:
        patch = str(wheel["patch"])
        wheel_stl = case_path / "constant" / "geometry" / f"{patch}.stl"
        write_transformed_binary_stl(
            Path(str(wheel["model_path"])),
            wheel_stl,
            scale=model_scale,
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            target_flow_axis=flow_axis,
            rotation_degrees=model_rotation_degrees,
            translation=model_translation_m,
            rotation_center=body_rotation_center_source,
        )
        wheel_stls.append(wheel_stl)

    tunnel = _wind_tunnel(report.bounds, flow_axis, include_ground, preset, domain_settings)
    mesh_resolution = mesh_resolution_metadata(
        report.bounds,
        flow_axis,
        include_ground,
        quality,
        smallest_aero_feature_m,
        domain=domain_settings,
    )
    flow_vector = (
        _vector_value(inflow_settings["flow_vector_mps"])
        if "flow_vector_mps" in inflow_settings
        else _axis_vector(flow_axis, speed_mps)
    )
    drag_dir = _normalize_vector(flow_vector)
    lift_dir = (0.0, 0.0, 1.0) if flow_axis != "z" else (0.0, 1.0, 0.0)
    pitch_axis = _cross(lift_dir, drag_dir)
    reference_area = reference_area_m2 or estimate_reference_area(report, flow_axis)
    reference_length = reference_length_m or estimate_reference_length(report.bounds, flow_axis)
    wall_resolution = wall_layer_metadata(speed_mps, reference_length, wall_quality or quality)
    simulation_quality = simulation_quality_metadata(
        quality,
        simulation_mode,
        speed_mps,
        reference_length,
        float(mesh_resolution["estimated_surface_cell_m"]),
    )
    center = force_reference_m or _center(report.bounds)
    turbulent_length = turbulence_length_scale_m or 0.07 * reference_length
    patch_names = ["body", *(str(wheel["patch"]) for wheel in wheels)]

    values = {
        "speed_mps": speed_mps,
        "flow_vector": flow_vector,
        "drag_dir": drag_dir,
        "lift_dir": lift_dir,
        "pitch_axis": pitch_axis,
        "reference_area": reference_area,
        "reference_length": reference_length,
        "center": center,
        "air_density_kg_m3": air_density_kg_m3,
        "kinematic_viscosity_m2_s": kinematic_viscosity_m2_s,
        "turbulence_intensity": turbulence_intensity,
        "turbulence_length_scale_m": turbulent_length,
        "include_ground": include_ground,
        "moving_ground": moving_ground,
        "ground_velocity": flow_vector if moving_ground else (0.0, 0.0, 0.0),
        "k": _turbulent_kinetic_energy(speed_mps, turbulence_intensity),
        "omega": _specific_dissipation_rate(
            speed_mps,
            turbulence_intensity,
            turbulent_length,
        ),
        "location_in_mesh": _location_in_mesh(tunnel),
        "refinement_boxes": _refinement_boxes(report.bounds, flow_axis),
        "wall_resolution": wall_resolution,
        "mesh_resolution": mesh_resolution,
        "quality": simulation_quality,
        "simulation_mode": simulation_mode,
        "domain": domain_settings,
        "backflow_safe_outlet": bool(outlet_settings.get("backflow_safe", False)),
        "roughness_height_m": float(surface_settings.get("roughness_height_m", 0.0)),
        "roughness_constant": float(surface_settings.get("roughness_constant", 0.5)),
        "wheels": wheels,
        "patch_names": patch_names,
        "yawed_inflow": abs(float(inflow_settings.get("yaw_degrees", 0.0))) > 1e-12,
        "second_order_temporal": bool(transient_settings.get("second_order_temporal", False)),
        "fluid_profile": fluid_profile,
        "solver_module": solver_module,
        "turbulence_model": turbulence_model,
        "air_temperature_k": float(fluid_settings.get("temperature_k", 288.15)),
        "air_pressure_pa": float(fluid_settings.get("pressure_pa", 101325.0)),
        "dynamic_viscosity_pa_s": float(
            fluid_settings.get(
                "dynamic_viscosity_pa_s",
                air_density_kg_m3 * kinematic_viscosity_m2_s,
            )
        ),
        "porous_zones": porous_zones,
        "fan_zones": fan_zones,
        "heat_zones": heat_zones,
        "volume_zones": volume_zones,
        "nu_tilda": 3.0 * kinematic_viscosity_m2_s,
    }

    files = {
        "system/blockMeshDict": _block_mesh_dict(
            tunnel,
            flow_axis,
            include_ground,
            domain_settings,
        ),
        "system/snappyHexMeshDict": _snappy_hex_mesh_dict(
            values["location_in_mesh"],
            values["refinement_boxes"],
            values["wall_resolution"],
            values["mesh_resolution"],
            preset,
            wheels,
            volume_zones,
        ),
        "system/streamlines": _streamlines_dict(
            tunnel,
            report.bounds,
            flow_axis,
            include_ground,
            simulation_mode,
            "nuTilda" if turbulence_model != "kOmegaSST" else "k",
        ),
        "system/wallShearStress": _wall_shear_stress_dict(patch_names),
        "system/bodyPressure": _body_pressure_dict(
            simulation_mode,
            patch_names,
            include_temperature=solver_module == "fluid",
        ),
        "system/yPlus": _y_plus_dict(),
        "system/surfaceFeaturesDict": _surface_features_dict(wheels),
        "system/controlDict": _control_dict(values),
        "system/fvSchemes": _fv_schemes(
            simulation_mode,
            bool(values["second_order_temporal"]),
            solver_module,
            turbulence_model,
        ),
        "system/fvSolution": _fv_solution(
            simulation_quality,
            simulation_mode,
            solver_module,
            turbulence_model,
        ),
        "system/decomposeParDict": _decompose_par_dict(),
        "constant/physicalProperties": (
            _compressible_physical_properties(values)
            if solver_module == "fluid"
            else _physical_properties(kinematic_viscosity_m2_s)
        ),
        "constant/momentumTransport": _momentum_transport(turbulence_model),
        "0/U": _field_u(values),
        "0/p": (
            _field_p_compressible(include_ground, values)
            if solver_module == "fluid"
            else _field_p(include_ground, values)
        ),
        "0/k": _field_scalar("k", "0 2 -2 0 0 0 0", values["k"], include_ground, values),
        "0/omega": _field_scalar(
            "omega",
            "0 0 -1 0 0 0 0",
            values["omega"],
            include_ground,
            values,
        ),
        "0/nut": _field_nut(include_ground, values),
        "Allmesh": _allmesh(),
        "Allsolve": _allsolve(
            include_y_plus=int(wall_resolution["surface_layers"]) > 0,
            solver_module=solver_module,
        ),
        "Allrun": _allrun(
            include_y_plus=int(wall_resolution["surface_layers"]) > 0,
            solver_module=solver_module,
        ),
        "run-wsl.ps1": _run_wsl_ps1(),
        "README.case.md": _case_readme(include_ground, moving_ground, values),
    }
    if turbulence_model != "kOmegaSST":
        files.pop("0/k")
        files.pop("0/omega")
        files["0/nuTilda"] = _field_nu_tilda(include_ground, values)
        files["0/nut"] = _field_nut_spalart_allmaras(include_ground, values)
    if solver_module == "fluid":
        files["constant/thermophysicalTransport"] = _thermophysical_transport(
            turbulence_model
        )
        files["0/T"] = _field_temperature(include_ground, values)
        files["0/alphat"] = _field_alphat(include_ground, values)
    if volume_zones:
        files["constant/fvModels"] = _fv_models(porous_zones, fan_zones, heat_zones)

    written: list[Path] = [body_stl, *wheel_stls]
    for relative_path, content in files.items():
        path = case_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        written.append(path)

    return written


def ensure_case_postprocessing(case_path: Path) -> dict[str, bool]:
    """Add current AeroLab post-processing to an older generated case."""
    case_path = case_path.resolve()
    allrun_path = case_path / "Allrun"
    if not allrun_path.is_file():
        raise FileNotFoundError(allrun_path)

    solver_module = "incompressibleFluid"
    simulation_mode = "steady"
    case_json_path = case_path / "case.json"
    if case_json_path.is_file():
        try:
            case_payload = json.loads(case_json_path.read_text(encoding="utf-8"))
            solver_module = str(case_payload.get("solver_module") or solver_module)
            quality = case_payload.get("cfd_quality")
            if isinstance(quality, dict):
                simulation_mode = str(quality.get("simulation_mode") or simulation_mode)
            elif str(case_payload.get("simulation_type") or "").startswith("transient"):
                simulation_mode = "transient"
        except (OSError, ValueError, TypeError):
            pass
    include_temperature = solver_module == "fluid"
    temperature_field = "TMean" if simulation_mode == "transient" else "T"

    body_pressure_path = case_path / "system" / "bodyPressure"
    body_pressure_created = not body_pressure_path.exists()
    body_pressure_updated = body_pressure_created
    if body_pressure_path.is_file():
        body_pressure_text = body_pressure_path.read_text(encoding="utf-8", errors="ignore")
        body_pressure_updated = (
            "wallShearStress" not in body_pressure_text
            or "interpolate no;" not in body_pressure_text
            or (include_temperature and f" {temperature_field})" not in body_pressure_text)
        )
    if body_pressure_updated:
        body_pressure_path.parent.mkdir(parents=True, exist_ok=True)
        with body_pressure_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                _body_pressure_dict(
                    simulation_mode,
                    include_temperature=include_temperature,
                )
            )

    wall_shear_path = case_path / "system" / "wallShearStress"
    wall_shear_created = not wall_shear_path.exists()
    if wall_shear_created:
        with wall_shear_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_wall_shear_stress_dict())

    allrun = allrun_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    body_command = "foamPostProcess -func bodyPressure -latestTime\n"
    wall_command = (
        f"foamPostProcess -solver {solver_module} -func wallShearStress -latestTime\n"
    )
    allrun_updated = body_command not in allrun or wall_command not in allrun
    if body_command not in allrun:
        body_step = (
            'echo "=== AEROLAB STEP: bodyPressure ==="\n'
            + body_command
        )
        streamlines_command = "foamPostProcess -func streamlines -latestTime\n"
        if streamlines_command in allrun:
            allrun = allrun.replace(
                streamlines_command,
                streamlines_command + body_step,
                1,
            )
        else:
            complete_marker = 'echo "=== AEROLAB COMPLETE ==="'
            if complete_marker in allrun:
                allrun = allrun.replace(complete_marker, body_step + complete_marker, 1)
            else:
                allrun = allrun.rstrip("\n") + "\n" + body_step
    if wall_command not in allrun:
        wall_step = (
            'echo "=== AEROLAB STEP: wallShearStress ==="\n'
            + wall_command
        )
        body_marker = 'echo "=== AEROLAB STEP: bodyPressure ==="\n'
        if body_marker in allrun:
            allrun = allrun.replace(body_marker, wall_step + body_marker, 1)
        elif body_command in allrun:
            allrun = allrun.replace(body_command, wall_step + body_command, 1)
        else:
            allrun = allrun.rstrip("\n") + "\n" + wall_step
    if allrun_updated:
        with allrun_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(allrun)

    return {
        "upgraded": body_pressure_updated or wall_shear_created or allrun_updated,
        "bodyPressureCreated": body_pressure_created,
        "bodyPressureUpdated": body_pressure_updated,
        "wallShearStressCreated": wall_shear_created,
        "allrunUpdated": allrun_updated,
    }


def _quality_preset(quality: str) -> dict[str, object]:
    key = (quality or "standard").lower()
    if key not in QUALITY_PRESETS:
        raise ValueError(f"Unsupported CFD quality preset: {quality}")
    return QUALITY_PRESETS[key]


def _wind_tunnel(
    bounds: Bounds,
    flow_axis: str,
    include_ground: bool,
    preset: dict[str, object],
    domain: dict[str, object] | None = None,
) -> dict[str, object]:
    domain = domain or {}
    closed = domain.get("closed_tunnel")
    if domain.get("mode") == "closed_tunnel" and isinstance(closed, dict):
        minimum = list(_vector_value(closed["minimum_m"]))
        maximum = list(_vector_value(closed["maximum_m"]))
    else:
        axis = AXES[flow_axis]
        dims = bounds.dimensions
        flow_dim = max(dims[axis], 1.0)
        cross_dim = max((dims[i] for i in range(3) if i != axis), default=1.0)
        cross_dim = max(cross_dim, 1.0)

        minimum = list(bounds.minimum)
        maximum = list(bounds.maximum)
        for i in range(3):
            if i == axis:
                minimum[i] -= 3.0 * flow_dim
                maximum[i] += 7.0 * flow_dim
            else:
                minimum[i] -= 3.0 * cross_dim
                maximum[i] += 3.0 * cross_dim

        if include_ground:
            minimum[2] = 0.0

    size = [maximum[i] - minimum[i] for i in range(3)]
    target_divisions = float(preset["tunnel_divisions"])
    max_cells = int(preset["max_block_cells"])
    cells = tuple(max(12, min(max_cells, math.ceil(length / max(max(size) / target_divisions, 1e-6)))) for length in size)

    return {
        "min": tuple(minimum),
        "max": tuple(maximum),
        "cells": cells,
    }


def _refinement_boxes(bounds: Bounds, flow_axis: str) -> dict[str, dict[str, Vector]]:
    axis = AXES[flow_axis]
    dimensions = [max(value, 1e-6) for value in bounds.dimensions]
    flow_length = max(dimensions[axis], 1.0)
    body_min = list(bounds.minimum)
    body_max = list(bounds.maximum)
    wake_min = list(bounds.minimum)
    wake_max = list(bounds.maximum)

    for index in range(3):
        if index == axis:
            body_min[index] -= 0.6 * flow_length
            body_max[index] += 1.0 * flow_length
            wake_min[index] -= 0.1 * flow_length
            wake_max[index] += 5.0 * flow_length
        else:
            body_margin = max(0.65 * dimensions[index], 0.25 * flow_length)
            wake_margin = max(0.45 * dimensions[index], 0.18 * flow_length)
            body_min[index] -= body_margin
            body_max[index] += body_margin
            wake_min[index] -= wake_margin
            wake_max[index] += wake_margin

    return {
        "bodyRefinement": {"min": tuple(body_min), "max": tuple(body_max)},
        "wakeRefinement": {"min": tuple(wake_min), "max": tuple(wake_max)},
    }


def _block_mesh_dict(
    tunnel: dict[str, object],
    flow_axis: str,
    include_ground: bool,
    domain: dict[str, object] | None = None,
) -> str:
    minimum = tunnel["min"]
    maximum = tunnel["max"]
    cells = tunnel["cells"]
    assert isinstance(minimum, tuple)
    assert isinstance(maximum, tuple)
    assert isinstance(cells, tuple)

    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    nx, ny, nz = cells
    faces = {
        "x_min": "(0 3 7 4)",
        "x_max": "(1 5 6 2)",
        "y_min": "(0 4 5 1)",
        "y_max": "(3 2 6 7)",
        "z_min": "(0 1 2 3)",
        "z_max": "(4 7 6 5)",
    }
    inlet_face = {"x": "x_min", "y": "y_min", "z": "z_min"}[flow_axis]
    outlet_face = {"x": "x_max", "y": "y_max", "z": "z_max"}[flow_axis]
    if (domain or {}).get("mode") == "closed_tunnel":
        return _closed_tunnel_block_mesh_dict(
            minimum,
            maximum,
            cells,
            faces,
            inlet_face,
            outlet_face,
            flow_axis,
        )
    farfield_faces = [name for name in faces if name not in {inlet_face, outlet_face}]

    ground_section = ""
    if include_ground and "z_min" in farfield_faces:
        farfield_faces.remove("z_min")
        ground_section = f"""
    ground
    {{
        type wall;
        faces ({faces["z_min"]});
    }}
"""

    farfield = "\n            ".join(faces[name] for name in farfield_faces)

    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({x0:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y1:.9g} {z1:.9g})
    ({x0:.9g} {y1:.9g} {z1:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ({faces[inlet_face]});
    }}

    outlet
    {{
        type patch;
        faces ({faces[outlet_face]});
    }}

    farfield
    {{
        type patch;
        faces
        (
            {farfield}
        );
    }}
{ground_section}
);

mergePatchPairs
(
);
"""


def _closed_tunnel_block_mesh_dict(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    cells: tuple[int, int, int],
    faces: dict[str, str],
    inlet_face: str,
    outlet_face: str,
    flow_axis: str,
) -> str:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    nx, ny, nz = cells
    side_faces = ("y_min", "y_max") if flow_axis == "x" else ("x_min", "x_max")
    side_face_lines = "\n            ".join(faces[name] for name in side_faces)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({x0:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y0:.9g} {z0:.9g})
    ({x1:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y1:.9g} {z0:.9g})
    ({x0:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y0:.9g} {z1:.9g})
    ({x1:.9g} {y1:.9g} {z1:.9g})
    ({x0:.9g} {y1:.9g} {z1:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ({faces[inlet_face]});
    }}

    outlet
    {{
        type patch;
        faces ({faces[outlet_face]});
    }}

    sideWalls
    {{
        type wall;
        faces
        (
            {side_face_lines}
        );
    }}

    ceiling
    {{
        type wall;
        faces ({faces["z_max"]});
    }}

    ground
    {{
        type wall;
        faces ({faces["z_min"]});
    }}
);

mergePatchPairs
(
);
"""


def _snappy_hex_mesh_dict(
    location_in_mesh: object,
    refinement_boxes: object,
    wall_resolution: object,
    mesh_resolution: object,
    preset: dict[str, object],
    wheels: list[dict[str, object]] | None = None,
    volume_zones: list[dict[str, object]] | None = None,
) -> str:
    assert isinstance(refinement_boxes, dict)
    body_box = refinement_boxes["bodyRefinement"]
    wake_box = refinement_boxes["wakeRefinement"]
    assert isinstance(body_box, dict)
    assert isinstance(wake_box, dict)
    assert isinstance(wall_resolution, dict)
    assert isinstance(mesh_resolution, dict)
    surface_min_level = int(mesh_resolution["configured_surface_min_level"])
    surface_max_level = int(mesh_resolution["configured_surface_max_level"])
    feature_level = int(mesh_resolution["configured_feature_level"])
    max_local_cells = int(mesh_resolution["configured_max_local_cells"])
    max_global_cells = int(mesh_resolution["configured_max_global_cells"])
    body_region_level = int(mesh_resolution["configured_body_region_level"])
    wake_region_level = int(mesh_resolution["configured_wake_region_level"])
    transition_levels = int(mesh_resolution["configured_n_cells_between_levels"])
    surface_layers = int(wall_resolution["surface_layers"])
    wheel_list = wheels or []
    wheel_geometry = "".join(
        f'''\n\n    {wheel["patch"]}
    {{
        type triSurface;
        file "{wheel["patch"]}.stl";
    }}'''
        for wheel in wheel_list
    )
    wheel_feature_entries = "".join(
        f'''\n        {{
            file "{wheel["patch"]}.eMesh";
            level {feature_level};
        }}'''
        for wheel in wheel_list
    )
    wheel_refinement_surfaces = "".join(
        f'''\n\n        {wheel["patch"]}
        {{
            level ({surface_min_level} {surface_max_level});
            patchInfo
            {{
                type wall;
                inGroups (bodyGroup);
            }}
        }}'''
        for wheel in wheel_list
    )
    wheel_layers = "".join(
        f'''\n\n        {wheel["patch"]}
        {{
            nSurfaceLayers {surface_layers};
        }}'''
        for wheel in wheel_list
    )
    zone_list = volume_zones or []
    zone_geometry = "".join(
        f'''\n\n    {zone["name"]}
    {{
        type box;
        min {_foam_vec(_vector_value(zone["minimum_m"]))};
        max {_foam_vec(_vector_value(zone["maximum_m"]))};
    }}'''
        for zone in zone_list
    )
    zone_refinement_surfaces = "".join(
        f'''\n\n        {zone["name"]}
        {{
            level ({body_region_level} {body_region_level});
            faceZone {zone["face_zone"]};
            cellZone {zone["cell_zone"]};
            mode inside;
            faceType internal;
        }}'''
        for zone in zone_list
    )
    zone_refinement_regions = "".join(
        f'''\n\n        {zone["name"]}
        {{
            mode inside;
            level {body_region_level};
        }}'''
        for zone in zone_list
    )
    add_layers = "true" if surface_layers > 0 else "false"
    layers = (
        f"""layers
    {{
        body
        {{
            nSurfaceLayers {surface_layers};
        }}{wheel_layers}
    }}"""
        if surface_layers > 0
        else "layers {}"
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}}

castellatedMesh true;
snap true;
addLayers {add_layers};

geometry
{{
    body
    {{
        type triSurface;
        file "body.stl";
    }}{wheel_geometry}

    bodyRefinement
    {{
        type box;
        min {_foam_vec(body_box["min"])};
        max {_foam_vec(body_box["max"])};
    }}

    wakeRefinement
    {{
        type box;
        min {_foam_vec(wake_box["min"])};
        max {_foam_vec(wake_box["max"])};
    }}{zone_geometry}
}}

castellatedMeshControls
{{
    maxLocalCells {max_local_cells};
    maxGlobalCells {max_global_cells};
    minRefinementCells 100;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {transition_levels};

    features
    (
        {{
            file "body.eMesh";
            level {feature_level};
        }}{wheel_feature_entries}
    );

    refinementSurfaces
    {{
        body
        {{
            level ({surface_min_level} {surface_max_level});
            patchInfo
            {{
                type wall;
                inGroups (bodyGroup);
            }}
        }}{wheel_refinement_surfaces}{zone_refinement_surfaces}
    }}

    resolveFeatureAngle 30;
    refinementRegions
    {{
        bodyRefinement
        {{
            mode inside;
            level {body_region_level};
        }}

        wakeRefinement
        {{
            mode inside;
            level {wake_region_level};
        }}{zone_refinement_regions}
    }}

    insidePoint {_foam_vec(location_in_mesh)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter {preset["snap_solve_iter"]};
    nRelaxIter 5;
    nFeatureSnapIter {preset["feature_snap_iter"]};
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    {layers}
    expansionRatio {wall_resolution["expansion_ratio"]:.9g};
    firstLayerThickness {wall_resolution["first_layer_thickness_m"]:.9g};
    minThickness {max(float(wall_resolution["total_layer_thickness_m"]) * 0.05, 1e-7):.9g};
    nGrow 0;
    featureAngle 100;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 15;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 75;
}}

meshQualityControls
{{
    maxNonOrtho 70;
    maxBoundarySkewness 4;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality -1e30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""


def _surface_features_dict(wheels: list[dict[str, object]] | None = None) -> str:
    surfaces = " ".join(
        ["\"body.stl\"", *(f'\"{wheel["patch"]}.stl\"' for wheel in (wheels or []))]
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object surfaceFeaturesDict;
}}

surfaces ({surfaces});
includedAngle 150;

subsetFeatures
{{
    nonManifoldEdges no;
    openEdges yes;
}}
"""


def _streamlines_dict(
    tunnel: dict[str, object],
    bounds: Bounds,
    flow_axis: str,
    include_ground: bool,
    simulation_mode: str = "steady",
    turbulence_field_name: str = "k",
) -> str:
    seed_points = _streamline_seed_points(tunnel, bounds, flow_axis, include_ground)
    point_lines = "\n".join(f"        {_foam_vec(point)}" for point in seed_points)
    velocity_field = "UMean" if simulation_mode == "transient" else "U"
    pressure_field = "pMean" if simulation_mode == "transient" else "p"
    turbulence_field = (
        f"{turbulence_field_name}Mean"
        if simulation_mode == "transient"
        else turbulence_field_name
    )
    return f"""type streamlines;
libs ("libfieldFunctionObjects.so");
writeControl writeTime;
setFormat vtk;
U {velocity_field};
direction forward;
fields
(
    {velocity_field}
    {pressure_field}
    {turbulence_field}
);
lifeTime 20000;
nSubCycle 5;
cloudName aerolabStreamlines;
seedSampleSet
{{
    type points;
    points
    (
{point_lines}
    );
    ordered no;
    axis xyz;
}}
"""


def _streamline_seed_points(
    tunnel: dict[str, object],
    bounds: Bounds,
    flow_axis: str,
    include_ground: bool,
) -> list[Vector]:
    flow_index = AXES[flow_axis]
    up_index = 2 if flow_axis != "z" else 1
    side_index = next(index for index in range(3) if index not in {flow_index, up_index})
    dimensions = [max(value, 1e-6) for value in bounds.dimensions]
    flow_length = max(dimensions[flow_index], 1.0)
    seed_flow = bounds.minimum[flow_index] - 2.0 * flow_length

    side_center = (bounds.minimum[side_index] + bounds.maximum[side_index]) * 0.5
    side_factors = (
        -0.85,
        -0.70,
        -0.58,
        -0.49,
        -0.41,
        -0.33,
        -0.24,
        -0.12,
        0.0,
        0.12,
        0.24,
        0.33,
        0.41,
        0.49,
        0.58,
        0.70,
        0.85,
    )
    if include_ground:
        up_factors = (0.03, 0.09, 0.17, 0.27, 0.39, 0.53, 0.69, 0.87, 1.04, 1.20, 1.35)
    else:
        up_factors = (-0.25, -0.12, 0.0, 0.10, 0.22, 0.36, 0.52, 0.70, 0.90, 1.12, 1.35)

    tunnel_min = tunnel["min"]
    tunnel_max = tunnel["max"]
    assert isinstance(tunnel_min, tuple)
    assert isinstance(tunnel_max, tuple)
    seed_flow = min(max(seed_flow, tunnel_min[flow_index] + 1e-4), tunnel_max[flow_index] - 1e-4)

    points: list[Vector] = []
    for side_factor in side_factors:
        side = side_center + side_factor * dimensions[side_index]
        side = min(max(side, tunnel_min[side_index] + 1e-4), tunnel_max[side_index] - 1e-4)
        for up_factor in up_factors:
            up = bounds.minimum[up_index] + up_factor * dimensions[up_index]
            up = min(max(up, tunnel_min[up_index] + 1e-4), tunnel_max[up_index] - 1e-4)
            point = [0.0, 0.0, 0.0]
            point[flow_index] = seed_flow
            point[side_index] = side
            point[up_index] = up
            points.append(tuple(point))  # type: ignore[arg-type]
    return points


def _y_plus_dict() -> str:
    return """type yPlus;
libs ("libfieldFunctionObjects.so");
writeControl writeTime;
log yes;
"""


def _wall_shear_stress_dict(patch_names: list[str] | None = None) -> str:
    patches = " ".join(patch_names or ["body"])
    return f"""type wallShearStress;
libs ("libfieldFunctionObjects.so");
writeControl writeTime;
patches ({patches});
log yes;
"""


def _body_pressure_dict(
    simulation_mode: str = "steady",
    patch_names: list[str] | None = None,
    include_temperature: bool = False,
) -> str:
    fields = [
        "pMean" if simulation_mode == "transient" else "p",
        "wallShearStressMean" if simulation_mode == "transient" else "wallShearStress",
    ]
    if include_temperature:
        fields.append("TMean" if simulation_mode == "transient" else "T")
    patches = " ".join(patch_names or ["body"])
    return f"""type surfaces;
libs ("libsampling.so");
writeControl writeTime;
fields ({" ".join(fields)});
surfaceFormat vtk;
interpolationScheme cellPoint;
writeFormat ascii;
writeCompression off;
surfaces
(
    body
    {{
        type patch;
        patches ({patches});
        triangulate yes;
        interpolate no;
    }}
);
"""


def _control_dict(values: dict[str, object]) -> str:
    solver_module = str(values.get("solver_module") or "incompressibleFluid")
    turbulence_model = str(values.get("turbulence_model") or "kOmegaSST")
    if solver_module != "incompressibleFluid" or turbulence_model != "kOmegaSST":
        return _control_dict_advanced(values, solver_module, turbulence_model)
    quality = values["quality"]
    assert isinstance(quality, dict)
    simulation_mode = str(values.get("simulation_mode") or "steady")
    patches = " ".join(str(patch) for patch in values.get("patch_names", ["body"]))
    if simulation_mode == "transient":
        time_controls = f"""endTime {float(quality["end_time"]):.9g};
deltaT {float(quality["initial_delta_t_s"]):.9g};
writeControl adjustableRunTime;
writeInterval {float(quality["write_interval_s"]):.9g};
adjustTimeStep yes;
maxCo {float(quality["maximum_courant_number"]):.9g};
maxDeltaT {float(quality["maximum_delta_t_s"]):.9g};"""
        average_functions = f"""

    wallShearStressAverageInput
    {{
        type wallShearStress;
        libs ("libfieldFunctionObjects.so");
        patches ({patches});
        writeControl writeTime;
        log no;
    }}

    fieldAverage
    {{
        type fieldAverage;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
        restartOnRestart false;
        restartOnOutput false;
        periodicRestart false;
        base time;
        window {float(quality["averaging_window_s"]):.9g};
        mean yes;
        prime2Mean yes;
        fields (U p k wallShearStress);
    }}"""
    else:
        time_controls = f"""endTime {quality["end_time"]};
deltaT 1;
writeControl timeStep;
writeInterval {quality["write_interval"]};"""
        average_functions = ""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}

solver incompressibleFluid;

startFrom startTime;
startTime 0;
stopAt endTime;
{time_controls}
purgeWrite 2;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type forceCoeffs;
        libs ("libforces.so");
        patches ({patches});
        rho rhoInf;
        rhoInf {values["air_density_kg_m3"]:.9g};
        CofR {_foam_vec(values["center"])};
        liftDir {_foam_vec(values["lift_dir"])};
        dragDir {_foam_vec(values["drag_dir"])};
        pitchAxis {_foam_vec(values["pitch_axis"])};
        magUInf {values["speed_mps"]:.9g};
        lRef {values["reference_length"]:.9g};
        Aref {values["reference_area"]:.9g};
        writeControl timeStep;
        timeInterval {quality["force_write_interval"]};
        log yes;
    }}{average_functions}
}}
"""


def _control_dict_advanced(
    values: dict[str, object],
    solver_module: str,
    turbulence_model: str,
) -> str:
    quality = values["quality"]
    assert isinstance(quality, dict)
    simulation_mode = str(values.get("simulation_mode") or "steady")
    patches = " ".join(str(patch) for patch in values.get("patch_names", ["body"]))
    if simulation_mode == "transient":
        time_controls = f"""endTime {float(quality["end_time"]):.9g};
deltaT {float(quality["initial_delta_t_s"]):.9g};
writeControl adjustableRunTime;
writeInterval {float(quality["write_interval_s"]):.9g};
adjustTimeStep yes;
maxCo {float(quality["maximum_courant_number"]):.9g};
maxDeltaT {float(quality["maximum_delta_t_s"]):.9g};"""
        turbulence_field = "nuTilda" if turbulence_model != "kOmegaSST" else "k"
        average_fields = ["U", "p"]
        if solver_module == "fluid":
            average_fields.append("T")
        average_fields.extend((turbulence_field, "wallShearStress"))
        average_functions = f"""

    wallShearStressAverageInput
    {{
        type wallShearStress;
        libs ("libfieldFunctionObjects.so");
        patches ({patches});
        writeControl writeTime;
        log no;
    }}

    fieldAverage
    {{
        type fieldAverage;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
        restartOnRestart false;
        restartOnOutput false;
        periodicRestart false;
        base time;
        window {float(quality["averaging_window_s"]):.9g};
        mean yes;
        prime2Mean yes;
        fields ({" ".join(average_fields)});
    }}"""
    else:
        time_controls = f"""endTime {quality["end_time"]};
deltaT 1;
writeControl timeStep;
writeInterval {quality["write_interval"]};"""
        average_functions = ""
    density_setup = (
        "rho rho;"
        if solver_module == "fluid"
        else f'rho rhoInf;\n        rhoInf {values["air_density_kg_m3"]:.9g};'
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}

solver {solver_module};

startFrom startTime;
startTime 0;
stopAt endTime;
{time_controls}
purgeWrite 2;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type forceCoeffs;
        libs ("libforces.so");
        patches ({patches});
        {density_setup}
        CofR {_foam_vec(values["center"])};
        liftDir {_foam_vec(values["lift_dir"])};
        dragDir {_foam_vec(values["drag_dir"])};
        pitchAxis {_foam_vec(values["pitch_axis"])};
        magUInf {values["speed_mps"]:.9g};
        lRef {values["reference_length"]:.9g};
        Aref {values["reference_area"]:.9g};
        writeControl timeStep;
        timeInterval {quality["force_write_interval"]};
        log yes;
    }}{average_functions}
}}
"""


def _fv_schemes(
    simulation_mode: str = "steady",
    second_order_temporal: bool = False,
    solver_module: str = "incompressibleFluid",
    turbulence_model: str = "kOmegaSST",
) -> str:
    if solver_module != "incompressibleFluid" or turbulence_model != "kOmegaSST":
        return _fv_schemes_advanced(
            simulation_mode,
            second_order_temporal,
            solver_module,
            turbulence_model,
        )
    ddt_scheme = (
        "backward"
        if simulation_mode == "transient" and second_order_temporal
        else "Euler"
        if simulation_mode == "transient"
        else "steadyState"
    )
    bounded = "" if simulation_mode == "transient" else "bounded "
    template = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}

ddtSchemes
{
    default __DDT_SCHEME__;
}

gradSchemes
{
    default Gauss linear;
    grad(U) cellLimited Gauss linear 1;
}

divSchemes
{
    default none;
    div(phi,U) __BOUNDED__Gauss linearUpwindV grad(U);
    div(phi,k) __BOUNDED__Gauss upwind;
    div(phi,omega) __BOUNDED__Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default Gauss linear corrected;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default corrected;
}

wallDist
{
    method meshWave;
}
"""
    return template.replace("__DDT_SCHEME__", ddt_scheme).replace("__BOUNDED__", bounded)


def _fv_schemes_advanced(
    simulation_mode: str,
    second_order_temporal: bool,
    solver_module: str,
    turbulence_model: str,
) -> str:
    ddt_scheme = (
        "backward"
        if simulation_mode == "transient" and second_order_temporal
        else "Euler"
        if simulation_mode == "transient"
        else "steadyState"
    )
    bounded = "" if simulation_mode == "transient" else "bounded "
    turbulence_field = "nuTilda" if turbulence_model != "kOmegaSST" else "k"
    turbulence_divergence = (
        f"    div(phi,k) {bounded}Gauss upwind;\n"
        f"    div(phi,omega) {bounded}Gauss upwind;"
        if turbulence_model == "kOmegaSST"
        else f"    div(phi,nuTilda) {bounded}Gauss limitedLinear 1;"
    )
    stress_divergence = (
        "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;"
        if solver_module == "fluid"
        else "    div((nuEff*dev2(T(grad(U))))) Gauss linear;"
    )
    energy_divergence = ""
    if solver_module == "fluid":
        energy_divergence = f"""
    div(phi,h) {bounded}Gauss linearUpwind grad(h);
    div(phi,e) {bounded}Gauss linearUpwind grad(e);
    div(phi,K) {bounded}Gauss linearUpwind grad(K);
    div(phi,Ekp) {bounded}Gauss linearUpwind grad(Ekp);
    div(phi,(p|rho)) {bounded}Gauss upwind;
    div((phi|interpolate(rho)),p) Gauss upwind;"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}}

ddtSchemes
{{
    default {ddt_scheme};
}}

gradSchemes
{{
    default Gauss linear;
    grad(U) cellLimited Gauss linear 1;
    grad({turbulence_field}) cellLimited Gauss linear 1;
}}

divSchemes
{{
    default none;
    div(phi,U) {bounded}Gauss linearUpwind grad(U);
{turbulence_divergence}{energy_divergence}
{stress_divergence}
}}

laplacianSchemes
{{
    default Gauss linear corrected;
}}

interpolationSchemes
{{
    default linear;
}}

snGradSchemes
{{
    default corrected;
}}

wallDist
{{
    method meshWave;
}}
"""


def _fv_solution(
    quality: dict[str, object],
    simulation_mode: str = "steady",
    solver_module: str = "incompressibleFluid",
    turbulence_model: str = "kOmegaSST",
) -> str:
    if solver_module != "incompressibleFluid" or turbulence_model != "kOmegaSST":
        return _fv_solution_advanced(
            quality,
            simulation_mode,
            solver_module,
            turbulence_model,
        )
    if simulation_mode == "transient":
        return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}}

solvers
{{
    p
    {{
        solver GAMG;
        tolerance 1e-7;
        relTol 0.01;
        smoother GaussSeidel;
    }}

    pFinal
    {{
        $p;
        relTol 0;
    }}

    Phi
    {{
        $p;
    }}

    U
    {{
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }}

    UFinal
    {{
        $U;
        relTol 0;
    }}

    k
    {{
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }}

    kFinal
    {{
        $k;
        relTol 0;
    }}

    omega
    {{
        $k;
    }}

    omegaFinal
    {{
        $omega;
        relTol 0;
    }}
}}

PIMPLE
{{
    nOuterCorrectors {int(quality["pimple_outer_correctors"])};
    nCorrectors {int(quality["pimple_pressure_correctors"])};
    nNonOrthogonalCorrectors 0;
    momentumPredictor yes;
    consistent yes;
}}

potentialFlow
{{
    nNonOrthogonalCorrectors 10;
}}

relaxationFactors
{{
    equations
    {{
        ".*" 1;
    }}
}}

cache
{{
    grad(U);
}}
"""
    template = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}

solvers
{
    p
    {
        solver GAMG;
        tolerance 1e-7;
        relTol 0.01;
        smoother GaussSeidel;
    }

    Phi
    {
        $p;
    }

    U
    {
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }

    k
    {
        solver smoothSolver;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
        nSweeps 1;
    }

    omega
    {
        $k;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl
    {
        p __PRESSURE_RESIDUAL__;
        U __VELOCITY_RESIDUAL__;
        "(k|omega)" __TURBULENCE_RESIDUAL__;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 10;
}

relaxationFactors
{
    equations
    {
        U __VELOCITY_RELAXATION__;
        k 0.5;
        omega 0.5;
    }
}

cache
{
    grad(U);
}
"""
    return (
        template.replace("__PRESSURE_RESIDUAL__", f'{float(quality["pressure_residual_control"]):.9g}')
        .replace("__VELOCITY_RESIDUAL__", f'{float(quality["velocity_residual_control"]):.9g}')
        .replace("__TURBULENCE_RESIDUAL__", f'{float(quality["turbulence_residual_control"]):.9g}')
        .replace("__VELOCITY_RELAXATION__", f'{float(quality["velocity_relaxation"]):.9g}')
    )


def _fv_solution_advanced(
    quality: dict[str, object],
    simulation_mode: str,
    solver_module: str,
    turbulence_model: str,
) -> str:
    turbulence_pattern = "k|omega" if turbulence_model == "kOmegaSST" else "nuTilda"
    energy_pattern = "|h|e|rho" if solver_module == "fluid" else ""
    field_pattern = f"U|{turbulence_pattern}{energy_pattern}"
    outer_correctors = int(quality["pimple_outer_correctors"]) if simulation_mode == "transient" else 1
    pressure_correctors = (
        int(quality["pimple_pressure_correctors"])
        if simulation_mode == "transient"
        else 2
    )
    residual_energy = f'        "(h|e)" {float(quality["turbulence_residual_control"]):.9g};\n' if solver_module == "fluid" else ""
    relaxation_fields = (
        """    fields
    {
        p 0.3;
        rho 0.01;
    }

"""
        if solver_module == "fluid" and simulation_mode == "steady"
        else ""
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}}

solvers
{{
    p
    {{
        solver GAMG;
        tolerance 1e-7;
        relTol 0.01;
        smoother GaussSeidel;
    }}

    pFinal
    {{
        $p;
        relTol 0;
    }}

    "({field_pattern})"
    {{
        solver PBiCGStab;
        preconditioner DILU;
        tolerance 1e-8;
        relTol 0.1;
    }}

    "({field_pattern})Final"
    {{
        $U;
        relTol 0;
    }}
}}

PIMPLE
{{
    nOuterCorrectors {outer_correctors};
    nCorrectors {pressure_correctors};
    nNonOrthogonalCorrectors 0;
    momentumPredictor yes;
    consistent yes;
    residualControl
    {{
        p {float(quality["pressure_residual_control"]):.9g};
        U {float(quality["velocity_residual_control"]):.9g};
        "({turbulence_pattern})" {float(quality["turbulence_residual_control"]):.9g};
{residual_energy}    }}
}}

relaxationFactors
{{
{relaxation_fields}    equations
    {{
        ".*" {1 if simulation_mode == "transient" else float(quality["velocity_relaxation"]):.9g};
    }}
}}

cache
{{
    grad(U);
}}
"""


def _decompose_par_dict(processes: int = 1) -> str:
    if isinstance(processes, bool) or processes < 1:
        raise ValueError("OpenFOAM processes must be a positive integer.")
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object decomposeParDict;
}}

numberOfSubdomains {processes};
method scotch;
"""


def configure_decomposition(case_path: Path, processes: int) -> Path:
    """Atomically set the process count used by decomposePar for one case."""
    if isinstance(processes, bool) or not isinstance(processes, int) or processes < 1:
        raise ValueError("OpenFOAM processes must be a positive integer.")
    path = case_path.resolve() / "system" / "decomposeParDict"
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_decompose_par_dict(processes), encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path


def _physical_properties(kinematic_viscosity_m2_s: float = 1.5e-5) -> str:
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object physicalProperties;
}}

viscosityModel constant;
nu [0 2 -1 0 0 0 0] {kinematic_viscosity_m2_s:.9g};
"""


def _compressible_physical_properties(values: dict[str, object]) -> str:
    energy = (
        "sensibleInternalEnergy"
        if values.get("simulation_mode") == "transient"
        else "sensibleEnthalpy"
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object physicalProperties;
}}

thermoType
{{
    type hePsiThermo;
    mixture pureMixture;
    transport const;
    thermo hConst;
    equationOfState perfectGas;
    specie specie;
    energy {energy};
}}

mixture
{{
    specie
    {{
        molWeight 28.965;
    }}
    thermodynamics
    {{
        Cp 1005;
        hf 0;
    }}
    transport
    {{
        mu {float(values["dynamic_viscosity_pa_s"]):.9g};
        Pr 0.71;
    }}
}}
"""


def _thermophysical_transport(turbulence_model: str) -> str:
    simulation_type = "RAS" if turbulence_model == "kOmegaSST" else "LES"
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object thermophysicalTransport;
}}

{simulation_type}
{{
    model eddyDiffusivity;
    Prt 0.85;
}}
"""


def _momentum_transport(turbulence_model: str = "kOmegaSST") -> str:
    if turbulence_model != "kOmegaSST":
        delta = "IDDESDelta" if turbulence_model == "SpalartAllmarasIDDES" else "cubeRootVol"
        delta_coefficients = (
            """
    IDDESDeltaCoeffs
    {
        Cw 0.15;
    }
"""
            if delta == "IDDESDelta"
            else """
    cubeRootVolCoeffs
    {
        deltaCoeff 1;
    }
"""
        )
        return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object momentumTransport;
}}

simulationType LES;

LES
{{
    model {turbulence_model};
    delta {delta};
    turbulence on;
{delta_coefficients}}}
"""
    return """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object momentumTransport;
}

simulationType RAS;

RAS
{
    model kOmegaSST;
    turbulence on;
}
"""


def _field_u(values: dict[str, object]) -> str:
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    advanced = bool(
        values.get("backflow_safe_outlet")
        or values.get("yawed_inflow")
        or domain.get("mode") == "closed_tunnel"
        or wheels
    )
    if advanced:
        return _field_u_advanced(values, domain, wheels)

    ground_patch = ""
    if values["include_ground"]:
        ground_patch = f"""
    ground
    {{
        type fixedValue;
        value uniform {_foam_vec(values["ground_velocity"])};
    }}
"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform {_foam_vec(values["flow_vector"])};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {_foam_vec(values["flow_vector"])};
    }}

    outlet
    {{
        type zeroGradient;
    }}

    farfield
    {{
        type slip;
    }}
{ground_patch}
    body
    {{
        type noSlip;
    }}
}}
"""


def _field_u_advanced(
    values: dict[str, object],
    domain: dict[str, object],
    wheels: list[object],
) -> str:
    flow = _foam_vec(values["flow_vector"])
    if values.get("backflow_safe_outlet"):
        outlet_patch = f"""    outlet
    {{
        type pressureInletOutletVelocity;
        value uniform {flow};
    }}"""
    else:
        outlet_patch = """    outlet
    {
        type zeroGradient;
    }"""
    if domain.get("mode") == "closed_tunnel":
        domain_patch = """    sideWalls
    {
        type noSlip;
    }

    ceiling
    {
        type noSlip;
    }"""
    elif values.get("yawed_inflow"):
        domain_patch = f"""    farfield
    {{
        type inletOutlet;
        inletValue uniform {flow};
        value uniform {flow};
    }}"""
    else:
        domain_patch = """    farfield
    {
        type slip;
    }"""
    ground_patch = ""
    if values["include_ground"]:
        ground_patch = f"""

    ground
    {{
        type fixedValue;
        value uniform {_foam_vec(values["ground_velocity"])};
    }}"""
    wheel_patches = ""
    for wheel_value in wheels:
        if not isinstance(wheel_value, dict):
            continue
        wheel_patches += f"""

    {wheel_value["patch"]}
    {{
        type rotatingWallVelocity;
        origin {_foam_vec(_vector_value(wheel_value["center_m"]))};
        axis {_foam_vec(_vector_value(wheel_value["axis"]))};
        omega {float(wheel_value["omega_rad_s"]):.9g};
    }}"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform {flow};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {flow};
    }}

{outlet_patch}

{domain_patch}{ground_patch}

    body
    {{
        type noSlip;
    }}{wheel_patches}
}}
"""


def _field_p_compressible(
    include_ground: bool,
    values: dict[str, object],
) -> str:
    pressure = float(values["air_pressure_pa"])
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    if values.get("simulation_mode") == "transient":
        outlet_patch = f"""    outlet
    {{
        type waveTransmissive;
        field p;
        gamma 1.4;
        fieldInf {pressure:.9g};
        lInf {float(values["reference_length"]):.9g};
        value uniform {pressure:.9g};
    }}"""
    else:
        outlet_patch = f"""    outlet
    {{
        type fixedValue;
        value uniform {pressure:.9g};
    }}"""
    if domain.get("mode") == "closed_tunnel":
        domain_patches = """    sideWalls
    {
        type zeroGradient;
    }

    ceiling
    {
        type zeroGradient;
    }"""
    elif values.get("yawed_inflow"):
        domain_patches = f"""    farfield
    {{
        type freestreamPressure;
        freestreamValue uniform {pressure:.9g};
        value uniform {pressure:.9g};
    }}"""
    else:
        domain_patches = """    farfield
    {
        type zeroGradient;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = """

    ground
    {
        type zeroGradient;
    }"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        type zeroGradient;
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}

dimensions [1 -1 -2 0 0 0 0];
internalField uniform {pressure:.9g};

boundaryField
{{
    inlet
    {{
        type zeroGradient;
    }}

{outlet_patch}

{domain_patches}{ground_patch}

    body
    {{
        type zeroGradient;
    }}{wheel_patches}
}}
"""


def _field_temperature(
    include_ground: bool,
    values: dict[str, object],
) -> str:
    temperature = float(values["air_temperature_k"])
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    outlet_patch = (
        f"""    outlet
    {{
        type inletOutlet;
        inletValue uniform {temperature:.9g};
        value uniform {temperature:.9g};
    }}"""
        if values.get("backflow_safe_outlet")
        else """    outlet
    {
        type zeroGradient;
    }"""
    )
    if domain.get("mode") == "closed_tunnel":
        domain_patches = """    sideWalls
    {
        type zeroGradient;
    }

    ceiling
    {
        type zeroGradient;
    }"""
    elif values.get("yawed_inflow"):
        domain_patches = f"""    farfield
    {{
        type inletOutlet;
        inletValue uniform {temperature:.9g};
        value uniform {temperature:.9g};
    }}"""
    else:
        domain_patches = """    farfield
    {
        type zeroGradient;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = """

    ground
    {
        type zeroGradient;
    }"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        type zeroGradient;
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object T;
}}

dimensions [0 0 0 1 0 0 0];
internalField uniform {temperature:.9g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {temperature:.9g};
    }}

{outlet_patch}

{domain_patches}{ground_patch}

    body
    {{
        type zeroGradient;
    }}{wheel_patches}
}}
"""


def _field_alphat(
    include_ground: bool,
    values: dict[str, object],
) -> str:
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    wall = """type compressible::alphatWallFunction;
        value uniform 0;"""
    if domain.get("mode") == "closed_tunnel":
        domain_patches = f"""    sideWalls
    {{
        {wall}
    }}

    ceiling
    {{
        {wall}
    }}"""
    else:
        domain_patches = """    farfield
    {
        type calculated;
        value uniform 0;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = f"""

    ground
    {{
        {wall}
    }}"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        {wall}
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object alphat;
}}

dimensions [1 -1 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type calculated;
        value uniform 0;
    }}

    outlet
    {{
        type calculated;
        value uniform 0;
    }}

{domain_patches}{ground_patch}

    body
    {{
        {wall}
    }}{wheel_patches}
}}
"""


def _field_p(
    include_ground: bool,
    values: dict[str, object] | None = None,
) -> str:
    values = values or {}
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    if domain.get("mode") == "closed_tunnel" or wheels:
        wall_names = ["sideWalls", "ceiling"] if domain.get("mode") == "closed_tunnel" else ["farfield"]
        domain_patches = "".join(
            f"""
    {name}
    {{
        type zeroGradient;
    }}
"""
            for name in wall_names
        )
        ground_patch = ""
        if include_ground:
            ground_patch = """
    ground
    {
        type zeroGradient;
    }
"""
        wheel_patches = "".join(
            f"""
    {wheel["patch"]}
    {{
        type zeroGradient;
    }}
"""
            for wheel in wheels
            if isinstance(wheel, dict)
        )
        return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type zeroGradient;
    }}

    outlet
    {{
        type fixedValue;
        value uniform 0;
    }}
{domain_patches}{ground_patch}
    body
    {{
        type zeroGradient;
    }}
{wheel_patches}}}
"""

    ground_patch = ""
    if include_ground:
        ground_patch = """
    ground
    {
        type zeroGradient;
    }
"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type zeroGradient;
    }}

    outlet
    {{
        type fixedValue;
        value uniform 0;
    }}

    farfield
    {{
        type zeroGradient;
    }}
{ground_patch}
    body
    {{
        type zeroGradient;
    }}
}}
"""


def _field_scalar(
    name: str,
    dimensions: str,
    value: float,
    include_ground: bool,
    values: dict[str, object] | None = None,
) -> str:
    settings = values or {}
    domain = settings.get("domain") if isinstance(settings.get("domain"), dict) else {}
    wheels = settings.get("wheels") if isinstance(settings.get("wheels"), list) else []
    advanced = bool(
        settings.get("backflow_safe_outlet")
        or settings.get("yawed_inflow")
        or domain.get("mode") == "closed_tunnel"
        or wheels
    )
    if advanced:
        return _field_scalar_advanced(name, dimensions, value, include_ground, settings, domain, wheels)

    ground_patch = ""
    if include_ground:
        ground_patch = """
    ground
    {
        type kqRWallFunction;
        value uniform 0;
    }
""" if name == "k" else f"""
    ground
    {{
        type omegaWallFunction;
        value uniform {value:.9g};
    }}
"""
    body_patch = """
    body
    {
        type kqRWallFunction;
        value uniform 0;
    }
""" if name == "k" else f"""
    body
    {{
        type omegaWallFunction;
        value uniform {value:.9g};
    }}
"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {name};
}}

dimensions [{dimensions}];
internalField uniform {value:.9g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {value:.9g};
    }}

    outlet
    {{
        type zeroGradient;
    }}

    farfield
    {{
        type zeroGradient;
    }}
{ground_patch}{body_patch}}}
"""


def _field_scalar_advanced(
    name: str,
    dimensions: str,
    value: float,
    include_ground: bool,
    settings: dict[str, object],
    domain: dict[str, object],
    wheels: list[object],
) -> str:
    wall_type = "kqRWallFunction" if name == "k" else "omegaWallFunction"
    wall_value = 0.0 if name == "k" else value
    outlet = (
        f"""    outlet
    {{
        type inletOutlet;
        inletValue uniform {value:.9g};
        value uniform {value:.9g};
    }}"""
        if settings.get("backflow_safe_outlet")
        else """    outlet
    {
        type zeroGradient;
    }"""
    )
    if domain.get("mode") == "closed_tunnel":
        domain_patches = f"""    sideWalls
    {{
        type {wall_type};
        value uniform {wall_value:.9g};
    }}

    ceiling
    {{
        type {wall_type};
        value uniform {wall_value:.9g};
    }}"""
    elif settings.get("yawed_inflow"):
        domain_patches = f"""    farfield
    {{
        type inletOutlet;
        inletValue uniform {value:.9g};
        value uniform {value:.9g};
    }}"""
    else:
        domain_patches = """    farfield
    {
        type zeroGradient;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = f"""

    ground
    {{
        type {wall_type};
        value uniform {wall_value:.9g};
    }}"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        type {wall_type};
        value uniform {wall_value:.9g};
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {name};
}}

dimensions [{dimensions}];
internalField uniform {value:.9g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {value:.9g};
    }}

{outlet}

{domain_patches}{ground_patch}

    body
    {{
        type {wall_type};
        value uniform {wall_value:.9g};
    }}{wheel_patches}
}}
"""


def _field_nu_tilda(
    include_ground: bool,
    values: dict[str, object],
) -> str:
    value = float(values["nu_tilda"])
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    outlet_patch = (
        f"""    outlet
    {{
        type inletOutlet;
        inletValue uniform {value:.9g};
        value uniform {value:.9g};
    }}"""
        if values.get("backflow_safe_outlet")
        else """    outlet
    {
        type zeroGradient;
    }"""
    )
    wall = """type fixedValue;
        value uniform 0;"""
    if domain.get("mode") == "closed_tunnel":
        domain_patches = f"""    sideWalls
    {{
        {wall}
    }}

    ceiling
    {{
        {wall}
    }}"""
    elif values.get("yawed_inflow"):
        domain_patches = f"""    farfield
    {{
        type inletOutlet;
        inletValue uniform {value:.9g};
        value uniform {value:.9g};
    }}"""
    else:
        domain_patches = """    farfield
    {
        type zeroGradient;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = f"""

    ground
    {{
        {wall}
    }}"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        {wall}
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object nuTilda;
}}

dimensions [0 2 -1 0 0 0 0];
internalField uniform {value:.9g};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {value:.9g};
    }}

{outlet_patch}

{domain_patches}{ground_patch}

    body
    {{
        {wall}
    }}{wheel_patches}
}}
"""


def _field_nut_spalart_allmaras(
    include_ground: bool,
    values: dict[str, object],
) -> str:
    domain = values.get("domain") if isinstance(values.get("domain"), dict) else {}
    wheels = values.get("wheels") if isinstance(values.get("wheels"), list) else []
    wall = """type nutUSpaldingWallFunction;
        value uniform 0;"""
    if domain.get("mode") == "closed_tunnel":
        domain_patches = f"""    sideWalls
    {{
        {wall}
    }}

    ceiling
    {{
        {wall}
    }}"""
    else:
        domain_patches = """    farfield
    {
        type calculated;
        value uniform 0;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = f"""

    ground
    {{
        {wall}
    }}"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        {wall}
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object nut;
}}

dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type calculated;
        value uniform 0;
    }}

    outlet
    {{
        type calculated;
        value uniform 0;
    }}

{domain_patches}{ground_patch}

    body
    {{
        {wall}
    }}{wheel_patches}
}}
"""


def _field_nut(
    include_ground: bool,
    values: dict[str, object] | None = None,
) -> str:
    settings = values or {}
    domain = settings.get("domain") if isinstance(settings.get("domain"), dict) else {}
    wheels = settings.get("wheels") if isinstance(settings.get("wheels"), list) else []
    roughness = float(settings.get("roughness_height_m", 0.0))
    if domain.get("mode") == "closed_tunnel" or wheels or roughness > 0:
        return _field_nut_advanced(include_ground, settings, domain, wheels)

    ground_patch = ""
    if include_ground:
        ground_patch = """
    ground
    {
        type nutkWallFunction;
        value uniform 0;
    }
"""
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object nut;
}}

dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type calculated;
        value uniform 0;
    }}

    outlet
    {{
        type calculated;
        value uniform 0;
    }}

    farfield
    {{
        type calculated;
        value uniform 0;
    }}
{ground_patch}
    body
    {{
        type nutkWallFunction;
        value uniform 0;
    }}
}}
"""


def _field_nut_advanced(
    include_ground: bool,
    settings: dict[str, object],
    domain: dict[str, object],
    wheels: list[object],
) -> str:
    roughness = float(settings.get("roughness_height_m", 0.0))
    roughness_constant = float(settings.get("roughness_constant", 0.5))
    if roughness > 0:
        body_bc = f"""type nutkRoughWallFunction;
        Ks {roughness:.9g};
        Cs {roughness_constant:.9g};
        value uniform 0;"""
    else:
        body_bc = """type nutkWallFunction;
        value uniform 0;"""
    if domain.get("mode") == "closed_tunnel":
        domain_patches = """    sideWalls
    {
        type nutkWallFunction;
        value uniform 0;
    }

    ceiling
    {
        type nutkWallFunction;
        value uniform 0;
    }"""
    else:
        domain_patches = """    farfield
    {
        type calculated;
        value uniform 0;
    }"""
    ground_patch = ""
    if include_ground:
        ground_patch = """

    ground
    {
        type nutkWallFunction;
        value uniform 0;
    }"""
    wheel_patches = "".join(
        f"""

    {wheel["patch"]}
    {{
        {body_bc}
    }}"""
        for wheel in wheels
        if isinstance(wheel, dict)
    )
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object nut;
}}

dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet
    {{
        type calculated;
        value uniform 0;
    }}

    outlet
    {{
        type calculated;
        value uniform 0;
    }}

{domain_patches}{ground_patch}

    body
    {{
        {body_bc}
    }}{wheel_patches}
}}
"""


def _fv_models(
    porous_zones: list[dict[str, object]],
    fan_zones: list[dict[str, object]],
    heat_zones: list[dict[str, object]],
) -> str:
    entries: list[str] = []
    for zone in porous_zones:
        entries.append(
            f"""{zone["name"]}
{{
    type porosityForce;

    porosityForceCoeffs
    {{
        cellZone {zone["cell_zone"]};
        type DarcyForchheimer;
        d {_foam_vec(_vector_value(zone["darcy_d_per_m2"]))};
        f {_foam_vec(_vector_value(zone["forchheimer_f_per_m"]))};
        coordinateSystem
        {{
            type cartesian;
            origin (0 0 0);
            coordinateRotation
            {{
                type axesRotation;
                e1 (1 0 0);
                e2 (0 1 0);
            }}
        }}
    }}
}}"""
        )
    for zone in fan_zones:
        entries.append(
            f"""{zone["name"]}
{{
    type actuationDisk;
    cellZone {zone["cell_zone"]};
    diskDir {_foam_vec(_vector_value(zone["disk_direction"]))};
    Cp {float(zone["power_coefficient"]):.9g};
    Ct {float(zone["thrust_coefficient"]):.9g};
    diskArea {float(zone["disk_area_m2"]):.9g};
    upstreamPoint {_foam_vec(_vector_value(zone["upstream_point_m"]))};
}}"""
        )
    for zone in heat_zones:
        entries.append(
            f"""{zone["name"]}
{{
    type heatSource;
    cellZone {zone["cell_zone"]};
    Q {float(zone["power_w"]):.9g};
}}"""
        )
    body = "\n\n".join(entries)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object fvModels;
}}

{body}
"""


def _parallel_script_prelude() -> str:
    return """AEROLAB_PROCESSES=${AEROLAB_PROCESSES:-1}
AEROLAB_FILE_HANDLER=${AEROLAB_FILE_HANDLER:-auto}
AEROLAB_RESUME=${AEROLAB_RESUME:-0}
AEROLAB_FEATURE_CACHE_KEY=${AEROLAB_FEATURE_CACHE_KEY:-disabled}
AEROLAB_BLOCK_CACHE_KEY=${AEROLAB_BLOCK_CACHE_KEY:-disabled}
case "$AEROLAB_PROCESSES" in
    ''|*[!0-9]*)
        echo "AeroLab stopped: AEROLAB_PROCESSES must be a positive integer." >&2
        exit 64
        ;;
esac
if [ "$AEROLAB_PROCESSES" -lt 1 ]; then
    echo "AeroLab stopped: AEROLAB_PROCESSES must be at least 1." >&2
    exit 64
fi
case "$AEROLAB_FILE_HANDLER" in
    auto) ;;
    uncollated|collated|masterUncollated)
        export FOAM_FILEHANDLER="$AEROLAB_FILE_HANDLER"
        ;;
    *)
        echo "AeroLab stopped: unsupported OpenFOAM file handler '$AEROLAB_FILE_HANDLER'." >&2
        exit 64
        ;;
esac
case "$AEROLAB_RESUME" in
    0|1) ;;
    *)
        echo "AeroLab stopped: AEROLAB_RESUME must be 0 or 1." >&2
        exit 64
        ;;
esac
for cache_key in "$AEROLAB_FEATURE_CACHE_KEY" "$AEROLAB_BLOCK_CACHE_KEY"; do
    case "$cache_key" in
        disabled) ;;
        ''|*[!0-9a-f]*)
            echo "AeroLab stopped: invalid stage-cache key." >&2
            exit 64
            ;;
    esac
done

cleanup_processor_dirs() {
    for processor_dir in processor[0-9]*; do
        case "$processor_dir" in
            processor*[!0-9]*) continue ;;
        esac
        if [ -d "$processor_dir" ]; then
            rm -rf -- "$processor_dir"
        fi
    done
}

"""


def _mesh_steps() -> str:
    return """mkdir -p .aerolab-cache/surfaceFeatures .aerolab-cache/blockMesh
echo "=== AEROLAB STEP: surfaceFeatures ==="
FEATURE_CACHE=".aerolab-cache/surfaceFeatures/$AEROLAB_FEATURE_CACHE_KEY"
if [ "$AEROLAB_FEATURE_CACHE_KEY" != disabled ] && [ -f "$FEATURE_CACHE/.ready" ]; then
    echo "AeroLab stage cache hit: surfaceFeatures"
    find constant/geometry -maxdepth 1 -type f -name '*.eMesh' -delete
    cp -a -- "$FEATURE_CACHE/files/." constant/geometry/
else
    find constant/geometry -maxdepth 1 -type f -name '*.eMesh' -delete
    surfaceFeatures
    if [ "$AEROLAB_FEATURE_CACHE_KEY" != disabled ]; then
        FEATURE_CACHE_TMP="$FEATURE_CACHE.tmp-$$"
        rm -rf -- "$FEATURE_CACHE_TMP" "$FEATURE_CACHE"
        mkdir -p "$FEATURE_CACHE_TMP/files"
        find constant/geometry -maxdepth 1 -type f -name '*.eMesh' -exec cp -a -- {} "$FEATURE_CACHE_TMP/files/" \\;
        : > "$FEATURE_CACHE_TMP/.ready"
        mv -- "$FEATURE_CACHE_TMP" "$FEATURE_CACHE"
    fi
fi
echo "=== AEROLAB STEP: blockMesh ==="
BLOCK_CACHE=".aerolab-cache/blockMesh/$AEROLAB_BLOCK_CACHE_KEY"
if [ "$AEROLAB_BLOCK_CACHE_KEY" != disabled ] && [ -f "$BLOCK_CACHE/.ready" ] && [ -f "$BLOCK_CACHE/polyMesh/points" ]; then
    echo "AeroLab stage cache hit: blockMesh"
    rm -rf constant/polyMesh
    cp -a -- "$BLOCK_CACHE/polyMesh" constant/polyMesh
else
    blockMesh
    if [ "$AEROLAB_BLOCK_CACHE_KEY" != disabled ]; then
        BLOCK_CACHE_TMP="$BLOCK_CACHE.tmp-$$"
        rm -rf -- "$BLOCK_CACHE_TMP" "$BLOCK_CACHE"
        mkdir -p "$BLOCK_CACHE_TMP"
        cp -a -- constant/polyMesh "$BLOCK_CACHE_TMP/polyMesh"
        : > "$BLOCK_CACHE_TMP/.ready"
        mv -- "$BLOCK_CACHE_TMP" "$BLOCK_CACHE"
    fi
fi
if [ "$AEROLAB_PROCESSES" -gt 1 ]; then
    echo "=== AEROLAB STEP: decomposeMesh ==="
    decomposePar -force
    echo "=== AEROLAB STEP: snappyHexMesh ==="
    mpirun -np "$AEROLAB_PROCESSES" snappyHexMesh -parallel -overwrite
    echo "=== AEROLAB STEP: reconstructMesh ==="
    reconstructParMesh -constant
    cleanup_processor_dirs
else
    echo "=== AEROLAB STEP: snappyHexMesh ==="
    snappyHexMesh -overwrite
fi
echo "=== AEROLAB STEP: checkMesh ==="
checkMesh | tee log.checkMesh
if ! grep -q "Mesh OK." log.checkMesh; then
    echo "AeroLab stopped: baseline OpenFOAM mesh checks failed." >&2
    exit 3
fi
echo "=== AEROLAB STEP: checkMeshDiagnostics ==="
checkMesh -allGeometry -allTopology
echo "=== AEROLAB STEP: meshSurface ==="
rm -rf VTK
foamToVTK -ascii -noInternal -time 0 -fields '(U)'
mkdir -p postProcessing/meshSurface/0
cp VTK/body/body_0.vtk postProcessing/meshSurface/0/body.vtk
rm -rf VTK
echo "=== AEROLAB MESH COMPLETE ==="
"""


def _solve_steps(
    include_y_plus: bool,
    validate_mesh: bool,
    solver_module: str = "incompressibleFluid",
) -> str:
    y_plus_step = f"""echo "=== AEROLAB STEP: yPlus ==="
foamPostProcess -solver {solver_module} -func yPlus -latestTime
""" if include_y_plus else ""
    mesh_check = """echo "=== AEROLAB STEP: checkMesh ==="
checkMesh | tee log.checkMesh
if ! grep -q "Mesh OK." log.checkMesh; then
    echo "AeroLab stopped: baseline OpenFOAM mesh checks failed." >&2
    exit 3
fi
echo "=== AEROLAB STEP: checkMeshDiagnostics ==="
checkMesh -allGeometry -allTopology
""" if validate_mesh else ""
    serial_initialization = (
        """        echo "=== AEROLAB STEP: potentialFoam ==="
        potentialFoam
"""
        if solver_module == "incompressibleFluid"
        else ""
    )
    parallel_initialization = (
        """        echo "=== AEROLAB STEP: potentialFoam ==="
        mpirun -np "$AEROLAB_PROCESSES" potentialFoam -parallel
"""
        if solver_module == "incompressibleFluid"
        else ""
    )
    return f"""{mesh_check}if [ "$AEROLAB_PROCESSES" -gt 1 ]; then
    echo "=== AEROLAB STEP: decomposeSolve ==="
    if [ "$AEROLAB_RESUME" -eq 1 ]; then
        decomposePar -force -latestTime
    else
        decomposePar -force
{parallel_initialization}    fi
    echo "=== AEROLAB STEP: foamRun ==="
    if [ "$AEROLAB_RESUME" -eq 1 ]; then
        mpirun -np "$AEROLAB_PROCESSES" foamRun -parallel -latestTime
    else
        mpirun -np "$AEROLAB_PROCESSES" foamRun -parallel
    fi
    echo "=== AEROLAB STEP: reconstructSolve ==="
    reconstructPar -latestTime
    cleanup_processor_dirs
else
    if [ "$AEROLAB_RESUME" -eq 1 ]; then
        echo "=== AEROLAB STEP: foamRun ==="
        foamRun -latestTime
    else
{serial_initialization}        echo "=== AEROLAB STEP: foamRun ==="
        foamRun
    fi
fi
echo "=== AEROLAB STEP: streamlines ==="
foamPostProcess -func streamlines -latestTime
echo "=== AEROLAB STEP: wallShearStress ==="
foamPostProcess -solver {solver_module} -func wallShearStress -latestTime
echo "=== AEROLAB STEP: bodyPressure ==="
foamPostProcess -func bodyPressure -latestTime
{y_plus_step}echo "=== AEROLAB COMPLETE ==="
"""


def _allmesh() -> str:
    return f"""#!/bin/sh
set -eu

{_parallel_script_prelude()}{_mesh_steps()}"""


def _allsolve(
    include_y_plus: bool,
    solver_module: str = "incompressibleFluid",
) -> str:
    return f"""#!/bin/sh
set -eu

{_parallel_script_prelude()}{_solve_steps(include_y_plus, validate_mesh=True, solver_module=solver_module)}"""


def _allrun(
    include_y_plus: bool,
    solver_module: str = "incompressibleFluid",
) -> str:
    return f"""#!/bin/sh
set -eu

{_parallel_script_prelude()}{_mesh_steps()}{_solve_steps(include_y_plus, validate_mesh=False, solver_module=solver_module)}"""


def _run_wsl_ps1() -> str:
    return """$ErrorActionPreference = "Stop"
wsl bash -lc "cd \"$(wslpath -a '$PWD')\" && chmod +x Allrun && ./Allrun"
"""


def _case_readme(
    include_ground: bool,
    moving_ground: bool,
    values: dict[str, object] | None = None,
) -> str:
    ground = "enabled" if include_ground else "disabled"
    moving = "enabled" if moving_ground else "disabled"
    settings = values or {}
    solver_module = str(settings.get("solver_module") or "incompressibleFluid")
    base = f"""# AeroLab OpenFOAM Case

Ground patch: {ground}
Moving ground: {moving}

## WSL2 Run

```powershell
.\\run-wsl.ps1
```

## Manual Run Inside OpenFOAM

```bash
chmod +x Allrun
./Allrun
```

This case targets OpenFOAM Foundation v13 and uses `foamRun` with the
`{solver_module}` solver module. Results appear under
`postProcessing/forceCoeffs` only after mesh, initialization, and solver steps
finish successfully. AeroLab will keep the result unverified until mesh quality,
residual, and force-coefficient stability checks pass.
"""
    domain = settings.get("domain") if isinstance(settings.get("domain"), dict) else {}
    wheels = settings.get("wheels") if isinstance(settings.get("wheels"), list) else []
    volume_zones = (
        settings.get("volume_zones")
        if isinstance(settings.get("volume_zones"), list)
        else []
    )
    advanced = bool(
        settings.get("yawed_inflow")
        or settings.get("backflow_safe_outlet")
        or settings.get("roughness_height_m")
        or settings.get("second_order_temporal")
        or settings.get("fluid_profile") == "compressible_thermal"
        or str(settings.get("turbulence_model") or "kOmegaSST") != "kOmegaSST"
        or volume_zones
        or domain.get("mode") == "closed_tunnel"
        or wheels
    )
    if not advanced:
        return base
    return base + f"""
## Applied advanced setup

- Solver profile: {settings.get("fluid_profile", "incompressible")} / {settings.get("turbulence_model", "kOmegaSST")}
- Domain: {domain.get("mode", "open_field")}
- Backflow-safe outlet: {"enabled" if settings.get("backflow_safe_outlet") else "disabled"}
- Surface roughness height: {float(settings.get("roughness_height_m", 0.0)):.9g} m
- Rotating wheel patches: {len(wheels)}
- Porous/fan cell zones: {len(volume_zones)}
- Temporal scheme: {"backward" if settings.get("second_order_temporal") else "legacy/default"}
"""


def estimate_reference_area(report: StlReport, flow_axis: str) -> float:
    projected_area = (
        report.silhouette_projected_areas.for_axis(flow_axis)
        if report.silhouette_projected_areas is not None
        else report.projected_areas.for_axis(flow_axis)
    )
    if projected_area > 1e-9:
        return projected_area
    bounds = report.bounds
    axis = AXES[flow_axis]
    dims = bounds.dimensions
    area_dims = [max(dims[i], 1e-6) for i in range(3) if i != axis]
    return max(area_dims[0] * area_dims[1], 1e-6)


def estimate_reference_length(bounds: Bounds, flow_axis: str) -> float:
    return max(bounds.dimensions[AXES[flow_axis]], 1e-6)


def _center(bounds: Bounds) -> Vector:
    return tuple((bounds.minimum[i] + bounds.maximum[i]) / 2.0 for i in range(3))  # type: ignore[return-value]


def _location_in_mesh(tunnel: dict[str, object]) -> Vector:
    minimum = tunnel["min"]
    maximum = tunnel["max"]
    assert isinstance(minimum, tuple)
    assert isinstance(maximum, tuple)
    return tuple(minimum[i] + 0.85 * (maximum[i] - minimum[i]) for i in range(3))  # type: ignore[return-value]


def _axis_vector(axis: str, magnitude: float) -> Vector:
    values = [0.0, 0.0, 0.0]
    values[AXES[axis]] = magnitude
    return tuple(values)  # type: ignore[return-value]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _turbulent_kinetic_energy(speed_mps: float, intensity: float = 0.01) -> float:
    return 1.5 * (speed_mps * intensity) ** 2


def _specific_dissipation_rate(
    speed_mps: float,
    intensity: float = 0.01,
    turbulent_length_scale_m: float = 0.07,
) -> float:
    k = _turbulent_kinetic_energy(speed_mps, intensity)
    c_mu = 0.09
    turbulent_length = max(turbulent_length_scale_m, 1e-6)
    return math.sqrt(k) / ((c_mu**0.25) * turbulent_length)


def _mapping_section(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _vector_value(value: object) -> Vector:
    if isinstance(value, dict):
        try:
            vector = (float(value["x"]), float(value["y"]), float(value["z"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Expected a finite X, Y, Z vector.") from exc
    else:
        try:
            components = tuple(float(component) for component in value)  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected a finite X, Y, Z vector.") from exc
        if len(components) != 3:
            raise ValueError("Expected a finite X, Y, Z vector.")
        vector = components  # type: ignore[assignment]
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("Expected a finite X, Y, Z vector.")
    return vector  # type: ignore[return-value]


def _normalize_vector(vector: Vector) -> Vector:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= 1e-12:
        raise ValueError("Flow direction must have non-zero length.")
    return tuple(component / magnitude for component in vector)  # type: ignore[return-value]


def _foam_vec(vector: object) -> str:
    x, y, z = vector  # type: ignore[misc]
    return f"({x:.9g} {y:.9g} {z:.9g})"
