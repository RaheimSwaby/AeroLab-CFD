from __future__ import annotations

import hashlib
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
from .stl import (
    Bounds,
    inspect_stl,
    transform_direction,
    transform_point,
    transformed_report,
    translated_report,
)

GEOMETRY_DIMENSION_TOLERANCE = 0.02
CASE_SCHEMA_VERSION = 3
PHYSICAL_MODEL_SCHEMA_VERSION = 2
COMPARISON_LOCK_SCHEMA_VERSION = 2
STANDARD_AIR_TEMPERATURE_C = 15.0
STANDARD_AIR_PRESSURE_PA = 101_325.0
STANDARD_AIR_DENSITY_KG_M3 = 1.225
STANDARD_KINEMATIC_VISCOSITY_M2_S = 1.5e-5
STANDARD_SPEED_OF_SOUND_MPS = 343.0
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
DRY_AIR_HEAT_CAPACITY_RATIO = 1.4
SUTHERLAND_REFERENCE_TEMPERATURE_K = 273.15
SUTHERLAND_REFERENCE_VISCOSITY_PA_S = 1.716e-5
SUTHERLAND_CONSTANT_K = 110.4
MAXIMUM_INCOMPRESSIBLE_MACH = 0.3
FLUID_PROFILES = {"incompressible", "compressible_thermal"}
TURBULENCE_MODELS = {
    "kOmegaSST",
    "SpalartAllmarasDES",
    "SpalartAllmarasIDDES",
}


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
    sensitivity_study: dict[str, object] | None = None,
    simulation_mode: str = "steady",
    air_temperature_c: float | None = None,
    air_pressure_pa: float | None = None,
    air_density_kg_m3: float | None = None,
    kinematic_viscosity_m2_s: float | None = None,
    turbulence_intensity_percent: float | None = None,
    turbulence_length_scale_m: float | None = None,
    center_of_gravity_m: tuple[float, float, float] | None = None,
    front_axle_station_m: float | None = None,
    rear_axle_station_m: float | None = None,
    yaw_degrees: float | None = None,
    crosswind_mps: float | None = None,
    roughness_height_m: float = 0.0,
    roughness_constant: float = 0.5,
    closed_tunnel: dict[str, object] | None = None,
    backflow_safe_outlet: bool = False,
    wheel_setup: list[dict[str, object]] | None = None,
    second_order_transient: bool = False,
    fluid_profile: str = "incompressible",
    turbulence_model: str = "kOmegaSST",
    porous_zones: list[dict[str, object]] | None = None,
    fan_zones: list[dict[str, object]] | None = None,
    heat_zones: list[dict[str, object]] | None = None,
) -> Path:
    speed_mph = float(speed_mph)
    if not math.isfinite(speed_mph) or speed_mph <= 0:
        raise ValueError("Air speed must be a finite positive value in mph.")
    flow_axis = str(flow_axis).lower()
    if flow_axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported flow axis: {flow_axis}")
    fluid_profile = _normalize_fluid_profile(fluid_profile)
    solver_module = "fluid" if fluid_profile == "compressible_thermal" else "incompressibleFluid"
    air = _air_properties(
        temperature_c=air_temperature_c,
        pressure_pa=air_pressure_pa,
        density_kg_m3=air_density_kg_m3,
        kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
    )
    speed_mps = speed_mph * 0.44704
    speed_of_sound_mps = float(air["speed_of_sound_mps"])
    mach_number = speed_mps / speed_of_sound_mps
    if fluid_profile == "incompressible" and mach_number >= MAXIMUM_INCOMPRESSIBLE_MACH:
        maximum_mph = MAXIMUM_INCOMPRESSIBLE_MACH * speed_of_sound_mps / 0.44704
        raise ValueError(
            f"{speed_mph:.3g} mph is Mach {mach_number:.3f}; AeroLab's incompressible solver "
            f"is limited to below Mach {MAXIMUM_INCOMPRESSIBLE_MACH:.1f} "
            f"(about {maximum_mph:.0f} mph for the configured air state). "
            "Select a compressible CFD solver by choosing the compressible thermal fluid "
            "profile above this limit."
        )
    if include_ground and flow_axis == "z":
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
    effective_reference_area = _positive_value(
        reference_area_m2,
        auto_reference_area,
        "Aerodynamic reference area",
    )
    effective_reference_length = _positive_value(
        reference_length_m,
        auto_reference_length,
        "Aerodynamic reference length",
    )
    vehicle_datums = normalize_vehicle_datums(
        flow_axis=flow_axis,
        bounds_center=tuple(
            (minimum + maximum) / 2.0
            for minimum, maximum in zip(report.bounds.minimum, report.bounds.maximum)
        ),
        center_of_gravity_m=center_of_gravity_m,
        front_axle_station_m=front_axle_station_m,
        rear_axle_station_m=rear_axle_station_m,
    )
    body_rotation_center_source = tuple(
        (minimum + maximum) / 2.0
        for minimum, maximum in zip(raw_report.bounds.minimum, raw_report.bounds.maximum)
    )
    physical_model = _physical_model_metadata(
        air=air,
        reference_length_m=effective_reference_length,
        reference_area_m2=auto_reference_area,
        bounds=report.bounds,
        speed_mps=speed_mps,
        flow_axis=flow_axis,
        include_ground=include_ground,
        moving_ground=moving_ground,
        simulation_mode=simulation_mode,
        turbulence_intensity_percent=turbulence_intensity_percent,
        turbulence_length_scale_m=turbulence_length_scale_m,
        yaw_degrees=yaw_degrees,
        crosswind_mps=crosswind_mps,
        roughness_height_m=roughness_height_m,
        roughness_constant=roughness_constant,
        closed_tunnel=closed_tunnel,
        backflow_safe_outlet=backflow_safe_outlet,
        wheel_setup=wheel_setup,
        second_order_transient=second_order_transient,
        fluid_profile=fluid_profile,
        turbulence_model=turbulence_model,
        porous_zones=porous_zones,
        fan_zones=fan_zones,
        heat_zones=heat_zones,
        model_path=model_path,
        model_scale=unit_scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        model_rotation_degrees=model_rotation_degrees,
        model_translation_m=model_translation_m,
        body_rotation_center_source=body_rotation_center_source,
    )
    wall_quality = "standard" if validation_study else quality
    mesh_resolution = mesh_resolution_metadata(
        report.bounds,
        flow_axis,
        include_ground,
        quality,
        smallest_aero_feature_m,
        domain=physical_model["domain"],
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
        "schema_version": CASE_SCHEMA_VERSION,
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
        "simulation_type": (
            f"{simulation_mode}_external_compressible_thermal_airflow"
            if fluid_profile == "compressible_thermal"
            else f"{simulation_mode}_external_incompressible_airflow"
        ),
        "solver_module": solver_module,
        "flow": {
            "axis": flow_axis,
            "speed_mph": speed_mph,
            "speed_mps": speed_mps,
            "yaw_degrees": physical_model["inflow"]["yaw_degrees"],
            "crosswind_mps": physical_model["inflow"]["crosswind_mps"],
            "flow_vector_mps": physical_model["inflow"]["flow_vector_mps"],
            "mach_number": mach_number,
            "maximum_supported_mach": (
                MAXIMUM_INCOMPRESSIBLE_MACH if fluid_profile == "incompressible" else None
            ),
            "speed_of_sound_mps": speed_of_sound_mps,
            "air_temperature_c": air["temperature_c"],
            "air_temperature_k": air["temperature_k"],
            "air_pressure_pa": air["pressure_pa"],
            "air_density_kg_m3": air["density_kg_m3"],
            "dynamic_viscosity_pa_s": air["dynamic_viscosity_pa_s"],
            "kinematic_viscosity_m2_s": air["kinematic_viscosity_m2_s"],
            "dynamic_pressure_pa": 0.5 * float(air["density_kg_m3"]) * speed_mps * speed_mps,
            "reynolds_number": (
                speed_mps
                * effective_reference_length
                / float(air["kinematic_viscosity_m2_s"])
            ),
        },
        "physical_model": physical_model,
        "vehicle_datums": vehicle_datums,
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
    if sensitivity_study:
        case["sensitivity_study"] = sensitivity_study
    case["comparison_lock"] = comparison_lock_metadata(case)

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
            air_density_kg_m3=float(air["density_kg_m3"]),
            kinematic_viscosity_m2_s=float(air["kinematic_viscosity_m2_s"]),
            turbulence_intensity=float(physical_model["inflow"]["turbulence_intensity"]),
            turbulence_length_scale_m=float(
                physical_model["inflow"]["turbulence_length_scale_m"]
            ),
            force_reference_m=_vector_from_mapping(vehicle_datums["moment_reference_m"]),
            physical_model=physical_model,
            body_rotation_center_source=body_rotation_center_source,
        )
        case["status"] = "openfoam_case_generated"
        # POSIX separators keep case.json portable: cases are staged into WSL/Linux,
        # and a Windows-generated manifest must not carry backslash paths.
        case["openfoam_files"] = [path.relative_to(case_path).as_posix() for path in files]

    with (case_path / "case.json").open("w", encoding="utf-8") as f:
        json.dump(case, f, indent=2)
        f.write("\n")

    return case_path


def _positive_value(value: float | None, default: float, label: str) -> float:
    number = default if value is None else float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a finite positive value.")
    return number


def _air_properties(
    *,
    temperature_c: float | None,
    pressure_pa: float | None,
    density_kg_m3: float | None,
    kinematic_viscosity_m2_s: float | None,
) -> dict[str, object]:
    temperature_was_set = temperature_c is not None
    pressure_was_set = pressure_pa is not None
    temperature_c = STANDARD_AIR_TEMPERATURE_C if temperature_c is None else float(temperature_c)
    pressure_pa = STANDARD_AIR_PRESSURE_PA if pressure_pa is None else float(pressure_pa)
    temperature_k = temperature_c + 273.15
    if not math.isfinite(temperature_k) or not 150.0 <= temperature_k <= 400.0:
        raise ValueError("Air temperature must be between -123.15 and 126.85 degrees Celsius.")
    if not math.isfinite(pressure_pa) or pressure_pa <= 0:
        raise ValueError("Air pressure must be a finite positive value in pascals.")

    weather_was_set = temperature_was_set or pressure_was_set
    if density_kg_m3 is None:
        density = (
            pressure_pa / (DRY_AIR_GAS_CONSTANT_J_KG_K * temperature_k)
            if weather_was_set
            else STANDARD_AIR_DENSITY_KG_M3
        )
    else:
        density = float(density_kg_m3)
    if not math.isfinite(density) or density <= 0:
        raise ValueError("Air density must be a finite positive value in kilograms per cubic meter.")

    sutherland_viscosity = (
        SUTHERLAND_REFERENCE_VISCOSITY_PA_S
        * (temperature_k / SUTHERLAND_REFERENCE_TEMPERATURE_K) ** 1.5
        * (SUTHERLAND_REFERENCE_TEMPERATURE_K + SUTHERLAND_CONSTANT_K)
        / (temperature_k + SUTHERLAND_CONSTANT_K)
    )
    if kinematic_viscosity_m2_s is None:
        kinematic_viscosity = (
            sutherland_viscosity / density
            if weather_was_set or density_kg_m3 is not None
            else STANDARD_KINEMATIC_VISCOSITY_M2_S
        )
    else:
        kinematic_viscosity = float(kinematic_viscosity_m2_s)
    if not math.isfinite(kinematic_viscosity) or kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be a finite positive value in square meters per second.")

    if density_kg_m3 is not None or kinematic_viscosity_m2_s is not None:
        source = "manual_override"
    elif weather_was_set:
        source = "ideal_gas_and_sutherland"
    else:
        source = "legacy_standard_air_reference"
    speed_of_sound = (
        math.sqrt(DRY_AIR_HEAT_CAPACITY_RATIO * DRY_AIR_GAS_CONSTANT_J_KG_K * temperature_k)
        if weather_was_set
        else STANDARD_SPEED_OF_SOUND_MPS
    )
    return {
        "model": "dry_air",
        "property_source": source,
        "temperature_c": temperature_c,
        "temperature_k": temperature_k,
        "pressure_pa": pressure_pa,
        "density_kg_m3": density,
        "dynamic_viscosity_pa_s": density * kinematic_viscosity,
        "kinematic_viscosity_m2_s": kinematic_viscosity,
        "speed_of_sound_mps": speed_of_sound,
    }


def _physical_model_metadata(
    *,
    air: dict[str, object],
    reference_length_m: float,
    reference_area_m2: float,
    bounds: Bounds,
    speed_mps: float,
    flow_axis: str,
    include_ground: bool,
    moving_ground: bool,
    simulation_mode: str,
    turbulence_intensity_percent: float | None,
    turbulence_length_scale_m: float | None,
    yaw_degrees: float | None,
    crosswind_mps: float | None,
    roughness_height_m: float,
    roughness_constant: float,
    closed_tunnel: dict[str, object] | None,
    backflow_safe_outlet: bool,
    wheel_setup: list[dict[str, object]] | None,
    second_order_transient: bool,
    fluid_profile: str,
    turbulence_model: str,
    porous_zones: list[dict[str, object]] | None,
    fan_zones: list[dict[str, object]] | None,
    heat_zones: list[dict[str, object]] | None,
    model_path: Path,
    model_scale: float,
    source_flow_direction: str,
    source_up_direction: str,
    model_rotation_degrees: tuple[float, float, float],
    model_translation_m: tuple[float, float, float],
    body_rotation_center_source: tuple[float, float, float],
) -> dict[str, object]:
    intensity_percent = 1.0 if turbulence_intensity_percent is None else float(turbulence_intensity_percent)
    if not math.isfinite(intensity_percent) or not 0 < intensity_percent <= 50:
        raise ValueError("Turbulence intensity must be greater than 0 and no more than 50 percent.")
    length_scale = (
        0.07 * reference_length_m
        if turbulence_length_scale_m is None
        else float(turbulence_length_scale_m)
    )
    if not math.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("Turbulence length scale must be a finite positive distance in meters.")

    simulation_mode = str(simulation_mode or "steady").lower()
    if simulation_mode not in {"steady", "transient"}:
        raise ValueError(f"Unsupported simulation mode: {simulation_mode}")
    if second_order_transient and simulation_mode != "transient":
        raise ValueError("Second-order temporal integration requires transient simulation mode.")
    fluid_profile = _normalize_fluid_profile(fluid_profile)
    turbulence_model = _normalize_turbulence_model(turbulence_model)
    if turbulence_model != "kOmegaSST" and simulation_mode != "transient":
        raise ValueError(f"{turbulence_model} requires transient simulation mode.")

    inflow = _normalize_inflow(speed_mps, flow_axis, yaw_degrees, crosswind_mps)
    roughness_height = float(roughness_height_m)
    roughness_cs = float(roughness_constant)
    if not math.isfinite(roughness_height) or roughness_height < 0:
        raise ValueError("Surface roughness height must be a finite non-negative distance in meters.")
    if not math.isfinite(roughness_cs) or not 0.5 <= roughness_cs <= 1.0:
        raise ValueError("OpenFOAM rough-wall constant Cs must be between 0.5 and 1.0.")
    if turbulence_model != "kOmegaSST" and roughness_height > 0:
        raise ValueError(
            "Surface roughness is not supported with Spalart-Allmaras DES/IDDES; "
            "use smooth walls or kOmegaSST."
        )

    domain = _normalize_domain(
        bounds=bounds,
        flow_axis=flow_axis,
        include_ground=include_ground,
        reference_area_m2=reference_area_m2,
        closed_tunnel=closed_tunnel,
        yawed=abs(float(inflow["yaw_degrees"])) > 1e-12,
    )
    wheels = _normalize_wheel_setup(
        wheel_setup=wheel_setup,
        include_ground=include_ground,
        moving_ground=moving_ground,
        model_path=model_path,
        speed_mps=speed_mps,
        model_scale=model_scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        flow_axis=flow_axis,
        model_rotation_degrees=model_rotation_degrees,
        model_translation_m=model_translation_m,
        body_rotation_center_source=body_rotation_center_source,
    )
    volume_zones = _normalize_volume_zones(
        porous_zones=porous_zones,
        fan_zones=fan_zones,
        heat_zones=heat_zones,
        reserved_names={
            "body",
            "bodyRefinement",
            "wakeRefinement",
            "inlet",
            "outlet",
            "farfield",
            "ground",
            "sideWalls",
            "ceiling",
            *(str(name) for name in wheels["patch_names"]),
        },
    )
    heat_zone_values = volume_zones.get("heat_zones")
    normalized_heat_zones = (
        [zone for zone in heat_zone_values if isinstance(zone, dict)]
        if isinstance(heat_zone_values, list)
        else []
    )
    if normalized_heat_zones and fluid_profile != "compressible_thermal":
        raise ValueError(
            "Heat-load zones require the compressible_thermal fluid profile so OpenFOAM solves the energy equation."
        )

    fluid_metadata = {**air, "applied_to_solver": True}
    if fluid_profile == "compressible_thermal":
        fluid_metadata.update(
            {
                "profile": fluid_profile,
                "solver_module": "fluid",
                "pressure_form": "absolute",
                "equation_of_state": "perfectGas",
                "energy_equation": (
                    "sensibleInternalEnergy"
                    if simulation_mode == "transient"
                    else "sensibleEnthalpy"
                ),
                "temperature_field": "T",
                "turbulent_thermal_diffusivity_field": "alphat",
            }
        )

    result: dict[str, object] = {
        "schema_version": PHYSICAL_MODEL_SCHEMA_VERSION,
        "fluid": fluid_metadata,
        "inflow": {
            **inflow,
            "turbulence_intensity": intensity_percent / 100.0,
            "turbulence_intensity_percent": intensity_percent,
            "turbulence_intensity_source": (
                "manual" if turbulence_intensity_percent is not None else "default_1_percent"
            ),
            "turbulence_length_scale_m": length_scale,
            "turbulence_length_scale_source": (
                "manual" if turbulence_length_scale_m is not None else "default_7_percent_reference_length"
            ),
        },
        "surface": {
            "roughness_model": "nutkRoughWallFunction" if roughness_height > 0 else "smooth",
            "roughness_height_m": roughness_height,
            "roughness_constant": roughness_cs,
            "roughness_patches": ["body", *wheels["patch_names"]],
            "applied_to_solver": True,
        },
        "domain": domain,
        "outlet": {
            "velocity_boundary": (
                "pressureInletOutletVelocity" if backflow_safe_outlet else "zeroGradient"
            ),
            "turbulence_boundary": "inletOutlet" if backflow_safe_outlet else "zeroGradient",
            "backflow_safe": bool(backflow_safe_outlet),
            "applied_to_solver": True,
        },
        "road_and_wheels": wheels,
        "transient": {
            "mode": simulation_mode,
            "time_integration": (
                "backward"
                if second_order_transient
                else "Euler"
                if simulation_mode == "transient"
                else "steadyState"
            ),
            "second_order_temporal": bool(second_order_transient),
            "advanced_statistics_status": (
                "enabled_pending_solver_history"
                if simulation_mode == "transient"
                else "not_applicable_steady"
            ),
            "applied_to_solver": True,
        },
    }
    if fluid_profile != "incompressible" or turbulence_model != "kOmegaSST":
        result["turbulence"] = {
            "model": turbulence_model,
            "simulation_type": "RAS" if turbulence_model == "kOmegaSST" else "LES",
            "transport_fields": (
                ["k", "omega", "nut"]
                if turbulence_model == "kOmegaSST"
                else ["nuTilda", "nut"]
            ),
            "transient_only": turbulence_model != "kOmegaSST",
            "applied_to_solver": True,
        }
    if volume_zones["enabled"]:
        result["volume_zones"] = volume_zones
    if normalized_heat_zones:
        result["thermal"] = {
            "model": "direct_air_volumetric_heat_source",
            "source_type": "heatSource",
            "power_mode": "total",
            "total_power_w": sum(float(zone["power_w"]) for zone in normalized_heat_zones),
            "temperature_field": "T",
            "component_temperatures_solved": False,
            "applied_to_solver": True,
        }
    return result


def _normalize_fluid_profile(value: object) -> str:
    token = str(value or "incompressible").strip().lower().replace("-", "_")
    aliases = {
        "incompressible": "incompressible",
        "incompressiblefluid": "incompressible",
        "compressible": "compressible_thermal",
        "compressible_thermal": "compressible_thermal",
        "fluid": "compressible_thermal",
    }
    normalized = aliases.get(token)
    if normalized is None or normalized not in FLUID_PROFILES:
        supported = ", ".join(sorted(FLUID_PROFILES))
        raise ValueError(f"Unsupported fluid profile {value!r}; choose one of: {supported}.")
    return normalized


def _normalize_turbulence_model(value: object) -> str:
    token = str(value or "kOmegaSST").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "komegasst": "kOmegaSST",
        "spalartallmarasdes": "SpalartAllmarasDES",
        "spalartallmarasiddes": "SpalartAllmarasIDDES",
    }
    normalized = aliases.get(token)
    if normalized is None or normalized not in TURBULENCE_MODELS:
        supported = ", ".join(sorted(TURBULENCE_MODELS))
        raise ValueError(f"Unsupported turbulence model {value!r}; choose one of: {supported}.")
    return normalized


def _normalize_volume_zones(
    *,
    porous_zones: list[dict[str, object]] | None,
    fan_zones: list[dict[str, object]] | None,
    heat_zones: list[dict[str, object]] | None,
    reserved_names: set[str],
) -> dict[str, object]:
    normalized_porous: list[dict[str, object]] = []
    normalized_fans: list[dict[str, object]] = []
    normalized_heat: list[dict[str, object]] = []
    names = set(reserved_names)

    for index, value in enumerate(_zone_list(porous_zones, "Porous zones"), start=1):
        name = _zone_name(value.get("name"), f"porousZone{index}", names, "Porous zone")
        minimum, maximum = _zone_box(value, f"Porous zone {name!r}")
        darcy = _resistance_vector(
            value.get("darcy_d_per_m2", value.get("darcy")),
            f"Porous zone {name!r} Darcy coefficient",
            required=True,
        )
        forchheimer = _resistance_vector(
            value.get("forchheimer_f_per_m", value.get("forchheimer", 0.0)),
            f"Porous zone {name!r} Forchheimer coefficient",
            required=False,
        )
        if not any(component > 0 for component in (*darcy, *forchheimer)):
            raise ValueError(f"Porous zone {name!r} requires non-zero Darcy or Forchheimer resistance.")
        normalized_porous.append(
            {
                "name": name,
                "cell_zone": name,
                "face_zone": f"{name}Faces",
                "minimum_m": _vector_mapping(minimum),
                "maximum_m": _vector_mapping(maximum),
                "darcy_d_per_m2": _vector_mapping(darcy),
                "forchheimer_f_per_m": _vector_mapping(forchheimer),
                "coordinate_system": "solver_cartesian",
            }
        )

    for index, value in enumerate(_zone_list(fan_zones, "Fan zones"), start=1):
        name = _zone_name(value.get("name"), f"fanZone{index}", names, "Fan zone")
        minimum, maximum = _zone_box(value, f"Fan zone {name!r}")
        direction = _unit_vector_input(
            value.get("disk_direction", value.get("disk_dir")),
            f"Fan zone {name!r} disk direction",
        )
        disk_area = _positive_number(
            value.get("disk_area_m2", value.get("disk_area")),
            f"Fan zone {name!r} disk area",
        )
        cp = _finite_zone_number(
            value.get("power_coefficient", value.get("cp")),
            f"Fan zone {name!r} power coefficient Cp",
        )
        ct = _finite_zone_number(
            value.get("thrust_coefficient", value.get("ct")),
            f"Fan zone {name!r} thrust coefficient Ct",
        )
        if cp < 0 or ct <= 0 or cp >= ct:
            raise ValueError(
                f"Fan zone {name!r} requires 0 <= Cp < Ct with a positive thrust coefficient."
            )
        upstream = _vector_input(
            value.get("upstream_point_m", value.get("upstream_point")),
            f"Fan zone {name!r} upstream sample point",
        )
        if all(minimum[axis] <= upstream[axis] <= maximum[axis] for axis in range(3)):
            raise ValueError(f"Fan zone {name!r} upstream sample point must be outside its box.")
        normalized_fans.append(
            {
                "name": name,
                "cell_zone": name,
                "face_zone": f"{name}Faces",
                "minimum_m": _vector_mapping(minimum),
                "maximum_m": _vector_mapping(maximum),
                "disk_direction": _vector_mapping(direction),
                "power_coefficient": cp,
                "thrust_coefficient": ct,
                "disk_area_m2": disk_area,
                "upstream_point_m": _vector_mapping(upstream),
            }
        )

    for index, value in enumerate(_zone_list(heat_zones, "Heat-load zones"), start=1):
        name = _zone_name(value.get("name"), f"heatZone{index}", names, "Heat-load zone")
        shape = str(value.get("shape") or "box").strip().lower().replace("-", "_")
        if shape != "box":
            raise ValueError(
                f"Heat-load zone {name!r} uses unsupported shape {shape!r}; currently only 'box' is supported."
            )
        minimum, maximum = _zone_box(value, f"Heat-load zone {name!r}")
        component_value = value.get("component")
        if not isinstance(component_value, str) or not component_value.strip():
            raise ValueError(f"Heat-load zone {name!r} requires a non-empty component label.")
        has_watts = value.get("power_w") is not None
        has_kilowatts = value.get("power_kw") is not None
        if has_watts == has_kilowatts:
            raise ValueError(
                f"Heat-load zone {name!r} requires exactly one of power_w or power_kw."
            )
        power_w = (
            _positive_number(value.get("power_w"), f"Heat-load zone {name!r} power")
            if has_watts
            else 1000.0
            * _positive_number(value.get("power_kw"), f"Heat-load zone {name!r} power")
        )
        normalized_heat.append(
            {
                "name": name,
                "shape": "box",
                "component": component_value.strip(),
                "cell_zone": name,
                "face_zone": f"{name}Faces",
                "minimum_m": _vector_mapping(minimum),
                "maximum_m": _vector_mapping(maximum),
                "power_w": power_w,
                "source_model": "heatSource",
                "power_mode": "total",
                "coordinate_system": "solver_cartesian",
            }
        )

    zone_names = [
        *(str(zone["name"]) for zone in normalized_porous),
        *(str(zone["name"]) for zone in normalized_fans),
        *(str(zone["name"]) for zone in normalized_heat),
    ]
    result: dict[str, object] = {
        "enabled": bool(zone_names),
        "coordinate_frame": "explicit_solver_coordinates_m",
        "porous_zones": normalized_porous,
        "fan_zones": normalized_fans,
        "zone_names": zone_names,
        "applied_to_solver": True,
    }
    if normalized_heat:
        result["heat_zones"] = normalized_heat
    return result


def _zone_list(
    value: list[dict[str, object]] | None,
    label: str,
) -> list[dict[str, object]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of explicit box definitions.")
    result: list[dict[str, object]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label} entry {index} must be a mapping.")
        result.append(item)
    return result


def _zone_name(
    value: object,
    default: str,
    names: set[str],
    label: str,
) -> str:
    name = str(value or default).strip()
    valid_start = bool(name) and ("A" <= name[0] <= "Z" or "a" <= name[0] <= "z")
    valid_characters = all(
        character == "_" or "A" <= character <= "Z" or "a" <= character <= "z" or "0" <= character <= "9"
        for character in name
    )
    if not valid_start or not valid_characters:
        raise ValueError(
            f"{label} name {name!r} must begin with an ASCII letter and contain only letters, numbers, or underscores."
        )
    if name in names or f"{name}Faces" in names:
        raise ValueError(f"{label} name {name!r} conflicts with another mesh region or patch.")
    names.update({name, f"{name}Faces"})
    return name


def _zone_box(
    value: dict[str, object],
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minimum = _vector_input(value.get("minimum_m", value.get("min")), f"{label} minimum")
    maximum = _vector_input(value.get("maximum_m", value.get("max")), f"{label} maximum")
    if any(minimum[axis] >= maximum[axis] for axis in range(3)):
        raise ValueError(f"{label} maximum must exceed its minimum on X, Y, and Z.")
    return minimum, maximum


def _resistance_vector(
    value: object,
    label: str,
    *,
    required: bool,
) -> tuple[float, float, float]:
    if value is None:
        if required:
            raise ValueError(f"{label} is required.")
        return (0.0, 0.0, 0.0)
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = _finite_zone_number(value, label)
        vector = (number, number, number)
    else:
        vector = _vector_input(value, label)
    if any(component < 0 for component in vector):
        raise ValueError(f"{label} components must be non-negative.")
    return vector


def _finite_zone_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _normalize_inflow(
    speed_mps: float,
    flow_axis: str,
    yaw_degrees: float | None,
    crosswind_mps: float | None,
) -> dict[str, object]:
    speed = float(speed_mps)
    requested_yaw = None if yaw_degrees is None else float(yaw_degrees)
    requested_crosswind = None if crosswind_mps is None else float(crosswind_mps)
    if requested_yaw is not None and not math.isfinite(requested_yaw):
        raise ValueError("Yaw angle must be a finite number of degrees.")
    if requested_crosswind is not None and not math.isfinite(requested_crosswind):
        raise ValueError("Crosswind speed must be a finite number in meters per second.")
    if requested_crosswind is not None and abs(requested_crosswind) >= speed:
        raise ValueError("Crosswind magnitude must be lower than the total freestream speed.")

    if requested_crosswind is not None:
        derived_yaw = math.degrees(math.asin(requested_crosswind / speed))
        if requested_yaw is not None and not math.isclose(
            requested_yaw,
            derived_yaw,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "Yaw and crosswind are inconsistent: speed_mps is total freestream magnitude, "
                "so crosswind must equal speed_mps * sin(yaw)."
            )
        yaw = derived_yaw
        source = "crosswind" if requested_yaw is None else "validated_yaw_and_crosswind"
    else:
        yaw = requested_yaw or 0.0
        source = "yaw" if requested_yaw is not None else "default_zero_yaw"
    if abs(yaw) >= 90.0:
        raise ValueError("Yaw magnitude must be lower than 90 degrees so flow enters the primary inlet.")

    primary = _axis_unit_vector(flow_axis)
    lift = (0.0, 0.0, 1.0) if flow_axis != "z" else (0.0, 1.0, 0.0)
    side = _cross_vector(lift, primary)
    radians = math.radians(yaw)
    flow_vector = tuple(
        speed * (math.cos(radians) * primary[index] + math.sin(radians) * side[index])
        for index in range(3)
    )
    crosswind = speed * math.sin(radians)
    return {
        "yaw_degrees": yaw,
        "crosswind_mps": crosswind,
        "primary_speed_mps": speed * math.cos(radians),
        "flow_vector_mps": _vector_mapping(flow_vector),
        "primary_direction": _vector_mapping(primary),
        "crosswind_direction": _vector_mapping(side),
        "yaw_convention": "positive_rotates_primary_toward_cross_lift_primary",
        "speed_definition": "total_freestream_magnitude",
        "yaw_crosswind_source": source,
        "applied_to_solver": True,
    }


def _normalize_domain(
    *,
    bounds: Bounds,
    flow_axis: str,
    include_ground: bool,
    reference_area_m2: float,
    closed_tunnel: dict[str, object] | None,
    yawed: bool,
) -> dict[str, object]:
    if closed_tunnel is None:
        return {
            "mode": "open_road" if include_ground else "open_field",
            "farfield_boundary": "inletOutlet" if yawed else "slip",
            "tunnel_wall_mode": "open_farfield",
            "blockage_ratio": None,
            "closed_tunnel": None,
            "applied_to_solver": True,
        }
    if not isinstance(closed_tunnel, dict):
        raise ValueError("Closed-tunnel setup must be a mapping of explicit dimensions.")
    if not include_ground:
        raise ValueError("A closed tunnel requires the ground patch to be enabled.")
    if flow_axis == "z":
        raise ValueError("Closed tunnels require X or Y primary flow.")

    width = _positive_mapping_value(closed_tunnel, "width_m", "Tunnel width")
    height = _positive_mapping_value(closed_tunnel, "height_m", "Tunnel height")
    upstream = _positive_mapping_value(closed_tunnel, "upstream_m", "Tunnel upstream distance")
    downstream = _positive_mapping_value(closed_tunnel, "downstream_m", "Tunnel downstream distance")
    flow_index = {"x": 0, "y": 1}[flow_axis]
    side_index = 1 if flow_index == 0 else 0
    side_span = bounds.dimensions[side_index]
    if width <= side_span:
        raise ValueError(
            f"Tunnel width {width:.6g} m must exceed the {side_span:.6g} m solver-geometry width."
        )
    if bounds.minimum[2] < -1e-9:
        raise ValueError("Closed-tunnel geometry must not extend below the road plane at Z=0.")
    if height <= bounds.maximum[2]:
        raise ValueError(
            f"Tunnel height {height:.6g} m must exceed the {bounds.maximum[2]:.6g} m geometry top."
        )

    minimum = list(bounds.minimum)
    maximum = list(bounds.maximum)
    minimum[flow_index] = bounds.minimum[flow_index] - upstream
    maximum[flow_index] = bounds.maximum[flow_index] + downstream
    side_center = (bounds.minimum[side_index] + bounds.maximum[side_index]) * 0.5
    minimum[side_index] = side_center - width * 0.5
    maximum[side_index] = side_center + width * 0.5
    minimum[2] = 0.0
    maximum[2] = height
    blockage = float(reference_area_m2) / (width * height)
    if not 0 <= blockage < 1:
        raise ValueError("Closed-tunnel blockage ratio must be below 100 percent.")
    return {
        "mode": "closed_tunnel",
        "farfield_boundary": None,
        "tunnel_wall_mode": "noSlip",
        "blockage_ratio": blockage,
        "blockage_percent": blockage * 100.0,
        "blockage_assessment": "high" if blockage > 0.10 else "moderate" if blockage > 0.05 else "low",
        "closed_tunnel": {
            "width_m": width,
            "height_m": height,
            "upstream_m": upstream,
            "downstream_m": downstream,
            "minimum_m": _vector_mapping(tuple(minimum)),
            "maximum_m": _vector_mapping(tuple(maximum)),
            "patches": ["sideWalls", "ceiling", "ground"],
        },
        "applied_to_solver": True,
    }


def _normalize_wheel_setup(
    *,
    wheel_setup: list[dict[str, object]] | None,
    include_ground: bool,
    moving_ground: bool,
    model_path: Path,
    speed_mps: float,
    model_scale: float,
    source_flow_direction: str,
    source_up_direction: str,
    flow_axis: str,
    model_rotation_degrees: tuple[float, float, float],
    model_translation_m: tuple[float, float, float],
    body_rotation_center_source: tuple[float, float, float],
) -> dict[str, object]:
    if not wheel_setup:
        return {
            "enabled": False,
            "moving_ground": bool(moving_ground),
            "coordinate_contract": "shared_body_source_frame",
            "body_rotation_center_source": _vector_mapping(body_rotation_center_source),
            "patch_names": [],
            "wheels": [],
            "applied_to_solver": True,
        }
    if not isinstance(wheel_setup, list):
        raise ValueError("Wheel setup must be a list of explicit wheel definitions.")
    if not include_ground or not moving_ground:
        raise ValueError("Rotating wheels require both the ground patch and moving ground to be enabled.")

    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    for index, wheel in enumerate(wheel_setup, start=1):
        if not isinstance(wheel, dict):
            raise ValueError(f"Wheel {index} must be a mapping.")
        name = str(wheel.get("name") or f"wheel{index}")
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(
                f"Wheel name {name!r} must begin with a letter and contain only letters, numbers, or underscores."
            )
        if name in names:
            raise ValueError(f"Wheel name {name!r} is duplicated.")
        names.add(name)

        path_value = wheel.get("model_path") or wheel.get("geometry")
        if not path_value:
            raise ValueError(f"Wheel {name!r} requires model_path.")
        source_path = Path(str(path_value)).expanduser()
        if not source_path.is_absolute():
            source_path = model_path.parent / source_path
        source_path = source_path.resolve()
        wheel_report = inspect_stl(source_path)
        if not wheel_report.is_cfd_candidate:
            raise ValueError(
                f"Wheel {name!r} geometry must be watertight, manifold, and free of degenerate triangles."
            )

        center_source = _vector_input(
            wheel.get("center_source", wheel.get("center")),
            f"Wheel {name!r} source center",
        )
        axis_source = _unit_vector_input(
            wheel.get("axis_source", wheel.get("axis")),
            f"Wheel {name!r} directed source axis",
        )
        if wheel.get("radius_source") is not None:
            radius_source = _positive_number(wheel["radius_source"], f"Wheel {name!r} source radius")
            radius_m = radius_source * model_scale
        elif wheel.get("radius_m") is not None:
            radius_m = _positive_number(wheel["radius_m"], f"Wheel {name!r} radius")
            radius_source = radius_m / model_scale
        else:
            raise ValueError(f"Wheel {name!r} requires radius_source or radius_m.")
        surface_speed = _positive_number(
            wheel.get("surface_speed_mps", speed_mps),
            f"Wheel {name!r} surface speed",
        )
        center_m = transform_point(
            center_source,
            scale=model_scale,
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            target_flow_axis=flow_axis,
            rotation_degrees=model_rotation_degrees,
            translation=model_translation_m,
            rotation_center=body_rotation_center_source,
        )
        axis = transform_direction(
            axis_source,
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            target_flow_axis=flow_axis,
            rotation_degrees=model_rotation_degrees,
        )
        normalized.append(
            {
                "name": name,
                "patch": name,
                "model_path": str(source_path),
                "source_center": _vector_mapping(center_source),
                "source_axis": _vector_mapping(axis_source),
                "source_radius": radius_source,
                "center_m": _vector_mapping(center_m),
                "axis": _vector_mapping(axis),
                "radius_m": radius_m,
                "surface_speed_mps": surface_speed,
                "omega_rad_s": surface_speed / radius_m,
                "rotation_direction_source": "directed_axis",
            }
        )
    return {
        "enabled": True,
        "moving_ground": True,
        "coordinate_contract": "wheel_geometry_centers_and_axes_share_body_source_frame_and_transform",
        "body_rotation_center_source": _vector_mapping(body_rotation_center_source),
        "patch_names": [wheel["patch"] for wheel in normalized],
        "wheels": normalized,
        "applied_to_solver": True,
    }


def _positive_mapping_value(mapping: dict[str, object], key: str, label: str) -> float:
    if key not in mapping:
        raise ValueError(f"{label} is required for a closed tunnel.")
    return _positive_number(mapping[key], label)


def _positive_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive value.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a finite positive value.")
    return number


def _vector_input(value: object, label: str) -> tuple[float, float, float]:
    if isinstance(value, dict):
        try:
            vector = (float(value["x"]), float(value["y"]), float(value["z"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} requires finite X, Y, and Z values.") from exc
    else:
        try:
            vector = tuple(float(component) for component in value)  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} requires finite X, Y, and Z values.") from exc
        if len(vector) != 3:
            raise ValueError(f"{label} requires finite X, Y, and Z values.")
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{label} requires finite X, Y, and Z values.")
    return vector  # type: ignore[return-value]


def _unit_vector_input(value: object, label: str) -> tuple[float, float, float]:
    vector = _vector_input(value, label)
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= 1e-12:
        raise ValueError(f"{label} must have non-zero length.")
    return tuple(component / magnitude for component in vector)  # type: ignore[return-value]


def _axis_unit_vector(axis: str) -> tuple[float, float, float]:
    vector = [0.0, 0.0, 0.0]
    vector[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    return tuple(vector)  # type: ignore[return-value]


def _cross_vector(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize_vehicle_datums(
    *,
    flow_axis: str,
    bounds_center: tuple[float, float, float],
    center_of_gravity_m: tuple[float, float, float] | None,
    front_axle_station_m: float | None,
    rear_axle_station_m: float | None,
) -> dict[str, object]:
    if center_of_gravity_m is None:
        cg = None
    else:
        if len(center_of_gravity_m) != 3:
            raise ValueError("The center of gravity requires solver-coordinate X, Y, and Z values.")
        values = tuple(float(value) for value in center_of_gravity_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Center-of-gravity coordinates must be finite numbers in meters.")
        cg = values

    axle_count = sum(value is not None for value in (front_axle_station_m, rear_axle_station_m))
    if axle_count == 1:
        raise ValueError("Front and rear axle stations must be provided together.")
    if axle_count and cg is None:
        raise ValueError("Axle stations require a complete center-of-gravity vector.")

    front = float(front_axle_station_m) if front_axle_station_m is not None else None
    rear = float(rear_axle_station_m) if rear_axle_station_m is not None else None
    if front is not None and rear is not None:
        if not math.isfinite(front) or not math.isfinite(rear):
            raise ValueError("Axle stations must be finite solver-coordinate values in meters.")
        if front >= rear:
            raise ValueError(
                f"Front axle station must be lower than rear axle station along solver +{flow_axis.upper()}."
            )
        cg_station = cg[{"x": 0, "y": 1, "z": 2}[flow_axis]]  # type: ignore[index]
        if not front < cg_station < rear:
            raise ValueError("The center of gravity must lie between the front and rear axle stations.")

    reference = cg or tuple(float(value) for value in bounds_center)
    balance_qualified = bool(cg is not None and front is not None and rear is not None)
    return {
        "schema_version": 1,
        "coordinate_system": "solver_meters",
        "flow_axis": flow_axis,
        "center_of_gravity_m": _vector_mapping(cg) if cg is not None else None,
        "moment_reference_m": _vector_mapping(reference),
        "moment_reference_source": "vehicle_center_of_gravity" if cg is not None else "geometry_bounds_center",
        "front_axle_station_m": front,
        "rear_axle_station_m": rear,
        "axle_station_axis": flow_axis,
        "wheelbase_m": rear - front if front is not None and rear is not None else None,
        "balance_qualified": balance_qualified,
        "qualification_detail": (
            "CG and front/rear axle stations define a qualified aero-balance datum."
            if balance_qualified
            else "Provide a solver-coordinate CG and both axle stations to qualify aero balance."
        ),
    }


def _vector_mapping(vector: tuple[float, float, float]) -> dict[str, float]:
    return {"x": float(vector[0]), "y": float(vector[1]), "z": float(vector[2])}


def _vector_from_mapping(value: object) -> tuple[float, float, float]:
    if not isinstance(value, dict):
        raise ValueError("A complete solver-coordinate vector is required.")
    return (float(value["x"]), float(value["y"]), float(value["z"]))


def comparison_lock_metadata(case_payload: dict[str, object]) -> dict[str, object]:
    flow = case_payload.get("flow") if isinstance(case_payload.get("flow"), dict) else {}
    ground = case_payload.get("ground") if isinstance(case_payload.get("ground"), dict) else {}
    placement = case_payload.get("placement") if isinstance(case_payload.get("placement"), dict) else {}
    reference = (
        case_payload.get("aerodynamic_reference")
        if isinstance(case_payload.get("aerodynamic_reference"), dict)
        else {}
    )
    mesh = (
        case_payload.get("mesh_resolution")
        if isinstance(case_payload.get("mesh_resolution"), dict)
        else {}
    )
    setup = {
        "schema_version": COMPARISON_LOCK_SCHEMA_VERSION,
        "solver_target": case_payload.get("solver_target"),
        "solver_module": case_payload.get("solver_module"),
        "simulation_type": case_payload.get("simulation_type"),
        "flow": {
            key: flow.get(key)
            for key in (
                "axis",
                "speed_mps",
                "yaw_degrees",
                "crosswind_mps",
                "flow_vector_mps",
                "mach_number",
                "air_temperature_k",
                "air_pressure_pa",
                "air_density_kg_m3",
                "dynamic_viscosity_pa_s",
                "kinematic_viscosity_m2_s",
            )
        },
        "physical_model": case_payload.get("physical_model"),
        "ground": {
            key: ground.get(key)
            for key in ("enabled", "moving", "clearance_m", "road_elevation_m")
        },
        "placement": {
            key: placement.get(key)
            for key in ("method", "ground_clearance_m", "road_elevation_m")
        },
        "aerodynamic_reference": {
            key: reference.get(key) for key in ("area_m2", "length_m")
        },
        "solver_quality": case_payload.get("cfd_quality"),
        "wall_resolution": case_payload.get("wall_resolution"),
        "mesh_controls": {
            key: mesh.get(key)
            for key in (
                "quality",
                "smallest_aero_feature_m",
                "minimum_cells_across_feature",
                "configured_surface_min_level",
                "configured_surface_max_level",
                "configured_body_region_level",
                "configured_wake_region_level",
                "configured_n_cells_between_levels",
            )
        },
        "vehicle_datums": case_payload.get("vehicle_datums"),
    }
    canonical = json.dumps(setup, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": COMPARISON_LOCK_SCHEMA_VERSION,
        "algorithm": "sha256",
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "geometry_excluded": True,
        "setup": setup,
    }


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
