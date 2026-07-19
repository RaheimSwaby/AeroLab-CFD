"""Browser-facing visualization data: streamlines, surface pressure, and mesh preview."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..stl import mesh_preview
from .parsing import _vtk_field, _vtk_header, _vtk_values
from .util import _percentile, _read_json_object

CASE_PREVIEW_TRIANGLE_LIMIT = 30_000


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


def _canonical_solver_point(point: tuple[float, ...], flow_axis: str) -> tuple[float, float, float]:
    x, y, z = point
    if flow_axis == "y":
        return (y, -x, z)
    if flow_axis == "z":
        return (z, x, y)
    return (x, y, z)
