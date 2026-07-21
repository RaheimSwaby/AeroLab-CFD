from __future__ import annotations

import math
import struct
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, TypedDict

Vector = tuple[float, float, float]
Triangle = tuple[Vector, Vector, Vector]


class _AeroFeatureCandidate(TypedDict):
    type: str
    label: str
    confidence: str
    angle_degrees: float
    length_m: float
    width_m: float
    rise_m: float
    length_fraction_percent: float
    width_fraction_percent: float
    surface_area_m2: float
    triangle_count: int
    bounds_m: dict[str, list[float]]


class _ReadinessResult(TypedDict):
    score: int
    status: str
    failed: int
    warnings: int
    items: list[dict[str, str]]

SIGNED_AXES: dict[str, Vector] = {
    "x": (1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class Bounds:
    minimum: Vector
    maximum: Vector

    @property
    def dimensions(self) -> Vector:
        return (
            self.maximum[0] - self.minimum[0],
            self.maximum[1] - self.minimum[1],
            self.maximum[2] - self.minimum[2],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True)
class ProjectedAreas:
    x: float
    y: float
    z: float

    def for_axis(self, axis: str) -> float:
        if axis == "x":
            return self.x
        if axis == "y":
            return self.y
        if axis == "z":
            return self.z
        raise ValueError(f"Unsupported axis: {axis}")

    def scaled(self, scale: float) -> ProjectedAreas:
        factor = scale * scale
        return ProjectedAreas(
            x=self.x * factor,
            y=self.y * factor,
            z=self.z * factor,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class StlReport:
    path: str
    format: str
    triangle_count: int
    unique_vertex_count: int
    bounds: Bounds
    surface_area: float
    projected_areas: ProjectedAreas
    volume: float
    open_edge_count: int
    non_manifold_edge_count: int
    degenerate_triangle_count: int
    warnings: tuple[str, ...]
    alignment_suggestion: dict[str, object] | None = None
    silhouette_projected_areas: ProjectedAreas | None = None

    @property
    def is_watertight(self) -> bool:
        return self.open_edge_count == 0 and self.non_manifold_edge_count == 0

    @property
    def is_cfd_candidate(self) -> bool:
        return (
            self.triangle_count > 0
            and self.degenerate_triangle_count == 0
            and self.open_edge_count == 0
            and self.non_manifold_edge_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "path": self.path,
            "format": self.format,
            "triangle_count": self.triangle_count,
            "unique_vertex_count": self.unique_vertex_count,
            "bounds": self.bounds.to_dict(),
            "surface_area": self.surface_area,
            "projected_areas": self.projected_areas.to_dict(),
            "volume": self.volume,
            "open_edge_count": self.open_edge_count,
            "non_manifold_edge_count": self.non_manifold_edge_count,
            "degenerate_triangle_count": self.degenerate_triangle_count,
            "is_watertight": self.is_watertight,
            "is_cfd_candidate": self.is_cfd_candidate,
            "readiness": _readiness(
                self.bounds,
                self.triangle_count,
                self.open_edge_count,
                self.non_manifold_edge_count,
                self.degenerate_triangle_count,
            ),
            "warnings": list(self.warnings),
        }
        if self.alignment_suggestion is not None:
            payload["alignment_suggestion"] = self.alignment_suggestion
        if self.silhouette_projected_areas is not None:
            payload["silhouette_projected_areas"] = self.silhouette_projected_areas.to_dict()
            payload["silhouette_method"] = "projected_triangle_union_scanline"
        return payload

    def to_text(self) -> str:
        dims = self.bounds.dimensions
        status = "PASS" if self.is_cfd_candidate else "NEEDS CLEANUP"
        lines = [
            f"AeroLab mesh check: {status}",
            f"Model: {self.path}",
            f"Format: {self.format}",
            f"Triangles: {self.triangle_count}",
            f"Unique vertices: {self.unique_vertex_count}",
            f"Dimensions: x={dims[0]:.6g}, y={dims[1]:.6g}, z={dims[2]:.6g}",
            f"Surface area: {self.surface_area:.6g}",
            "Projected areas: "
            f"x={self.projected_areas.x:.6g}, "
            f"y={self.projected_areas.y:.6g}, "
            f"z={self.projected_areas.z:.6g}",
            f"Volume: {self.volume:.6g}",
            f"Watertight: {'yes' if self.is_watertight else 'no'}",
            f"Open edges: {self.open_edge_count}",
            f"Non-manifold edges: {self.non_manifold_edge_count}",
            f"Degenerate triangles: {self.degenerate_triangle_count}",
            "Readiness score: "
            f"{_readiness(self.bounds, self.triangle_count, self.open_edge_count, self.non_manifold_edge_count, self.degenerate_triangle_count)['score']}/100",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def read_stl_triangles(path: Path) -> tuple[list[Triangle], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {path}")
    return _load_stl(path.read_bytes())


def inspect_stl(path: Path) -> StlReport:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {path}")

    triangles, stl_format = read_stl_triangles(path)
    if not triangles:
        raise ValueError(f"No triangles found in STL: {path}")
    return report_for_triangles(path, triangles, stl_format)


def detect_aero_features(
    path: Path,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    """Find geometry that resembles a rear underbody diffuser in a canonical frame."""
    triangles, _ = read_stl_triangles(path)
    canonical = transform_triangles(
        triangles,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis="x",
        rotation_degrees=rotation_degrees,
    )
    return detect_aero_features_for_triangles(canonical)


def detect_aero_features_for_triangles(triangles: list[Triangle]) -> dict[str, object]:
    """Return conservative diffuser candidates; absence is not proof of no diffuser."""
    if not triangles:
        return _aero_feature_result([])
    bounds = _bounds(triangles)
    length, width, height = bounds.dimensions
    if min(length, width, height) <= 1e-9:
        return _aero_feature_result([])

    candidate_indices: list[int] = []
    candidate_values: dict[int, tuple[float, float]] = {}
    rear_start = bounds.minimum[0] + 0.52 * length
    lower_limit = bounds.minimum[2] + 0.42 * height
    for index, triangle in enumerate(triangles):
        area = _triangle_area(triangle)
        if area <= 1e-12:
            continue
        centroid = tuple(sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3))
        if centroid[0] < rear_start or centroid[2] > lower_limit:
            continue
        normal = _normal(triangle)
        if abs(normal[2]) < 0.35:
            continue
        slope = -normal[0] / normal[2]
        angle = math.degrees(math.atan(slope))
        if not 2.0 <= angle <= 30.0:
            continue
        candidate_indices.append(index)
        candidate_values[index] = (area, angle)

    if not candidate_indices:
        return _aero_feature_result([])

    vertex_to_triangles: dict[Vector, list[int]] = {}
    for index in candidate_indices:
        for vertex in triangles[index]:
            vertex_to_triangles.setdefault(_quantize_vertex(vertex), []).append(index)

    adjacency: dict[int, set[int]] = {index: set() for index in candidate_indices}
    for connected in vertex_to_triangles.values():
        if len(connected) < 2:
            continue
        for index in connected:
            adjacency[index].update(other for other in connected if other != index)

    components: list[list[int]] = []
    remaining = set(candidate_indices)
    while remaining:
        root = remaining.pop()
        component = [root]
        stack = [root]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & remaining
            remaining.difference_update(neighbors)
            component.extend(neighbors)
            stack.extend(neighbors)
        components.append(component)

    candidates: list[_AeroFeatureCandidate] = []
    for component in components:
        component_bounds = _bounds([triangles[index] for index in component])
        x_span, y_span, z_span = component_bounds.dimensions
        area = sum(candidate_values[index][0] for index in component)
        weighted_angle = sum(
            candidate_values[index][0] * candidate_values[index][1]
            for index in component
        ) / max(area, 1e-12)
        x_fraction = x_span / length
        y_fraction = y_span / width
        outlet_fraction = (component_bounds.maximum[0] - bounds.minimum[0]) / length
        area_fraction = area / max(length * width, 1e-12)
        if (
            x_fraction < 0.06
            or y_fraction < 0.18
            or outlet_fraction < 0.72
            or z_span < 0.006 * height
            or area_fraction < 0.0025
        ):
            continue
        confidence = (
            "high"
            if x_fraction >= 0.12 and y_fraction >= 0.42 and area_fraction >= 0.012
            else "medium"
        )
        candidates.append(
            {
                "type": "diffuser_candidate",
                "label": "Rear underbody ramp",
                "confidence": confidence,
                "angle_degrees": round(weighted_angle, 3),
                "length_m": round(x_span, 6),
                "width_m": round(y_span, 6),
                "rise_m": round(z_span, 6),
                "length_fraction_percent": round(x_fraction * 100.0, 2),
                "width_fraction_percent": round(y_fraction * 100.0, 2),
                "surface_area_m2": round(area, 6),
                "triangle_count": len(component),
                "bounds_m": {
                    "min": [round(value, 6) for value in component_bounds.minimum],
                    "max": [round(value, 6) for value in component_bounds.maximum],
                },
            }
        )

    candidates.sort(
        key=lambda item: (
            item["confidence"] == "high",
            item["surface_area_m2"],
        ),
        reverse=True,
    )
    return _aero_feature_result(candidates[:4])


def _aero_feature_result(candidates: list[_AeroFeatureCandidate]) -> dict[str, object]:
    return {
        "method": "rear-lower-surface-ramp-heuristic-v1",
        "canonical_frame": "+X airflow, +Z up",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "verified": False,
        "detail": (
            "Geometry-based candidates require visual confirmation; STL triangles do not encode part names or airflow connectivity."
        ),
    }


def report_for_triangles(path: Path, triangles: list[Triangle], stl_format: str) -> StlReport:
    bounds = _bounds(triangles)
    unique_vertices = {_quantize_vertex(v) for triangle in triangles for v in triangle}
    areas = [_triangle_area(triangle) for triangle in triangles]
    degenerate_count = sum(1 for area in areas if area <= 1e-12)
    edge_counts = _edge_counts(triangles)
    open_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    surface_area = sum(areas)
    projected_areas = _projected_areas(triangles, areas)
    silhouette_projected_areas = silhouette_projected_areas_for_triangles(triangles)
    volume = abs(sum(_signed_tetra_volume(triangle) for triangle in triangles))
    warnings = _warnings(bounds, open_edges, non_manifold_edges, degenerate_count)
    alignment_suggestion = _principal_axis_alignment(triangles, areas, bounds)

    return StlReport(
        path=str(path),
        format=stl_format,
        triangle_count=len(triangles),
        unique_vertex_count=len(unique_vertices),
        bounds=bounds,
        surface_area=surface_area,
        projected_areas=projected_areas,
        volume=volume,
        open_edge_count=open_edges,
        non_manifold_edge_count=non_manifold_edges,
        degenerate_triangle_count=degenerate_count,
        warnings=tuple(warnings),
        alignment_suggestion=alignment_suggestion,
        silhouette_projected_areas=silhouette_projected_areas,
    )


def _principal_axis_alignment(
    triangles: list[Triangle],
    areas: list[float],
    bounds: Bounds,
) -> dict[str, object] | None:
    """Fit a vehicle-like length/width/height frame to the STL surface."""
    total_area = sum(area for area in areas if area > 1e-12)
    if total_area <= 1e-12:
        return None

    first = [0.0, 0.0, 0.0]
    second = [[0.0, 0.0, 0.0] for _ in range(3)]
    for triangle, area in zip(triangles, areas):
        if area <= 1e-12:
            continue
        vertex_sum = [sum(vertex[axis] for vertex in triangle) for axis in range(3)]
        for row in range(3):
            first[row] += area * vertex_sum[row] / 3.0
            for column in range(3):
                diagonal_sum = sum(vertex[row] * vertex[column] for vertex in triangle)
                second[row][column] += area * (
                    vertex_sum[row] * vertex_sum[column] + diagonal_sum
                ) / 12.0

    center: Vector = (
        first[0] / total_area,
        first[1] / total_area,
        first[2] / total_area,
    )
    covariance = [
        [
            second[row][column] / total_area - center[row] * center[column]
            for column in range(3)
        ]
        for row in range(3)
    ]
    eigenpairs = _symmetric_eigenpairs(covariance)
    if not eigenpairs or eigenpairs[0][0] <= 1e-16:
        return None

    length_axis = eigenpairs[0][1]
    up_axis = eigenpairs[2][1]
    if _dot(length_axis, (1.0, 0.0, 0.0)) < 0.0:
        length_axis = (-length_axis[0], -length_axis[1], -length_axis[2])
    if _dot(up_axis, (0.0, 0.0, 1.0)) < 0.0:
        up_axis = (-up_axis[0], -up_axis[1], -up_axis[2])
    width_axis = _normalize(_cross(up_axis, length_axis))
    up_axis = _normalize(_cross(length_axis, width_axis))

    rotation = (length_axis, width_axis, up_axis)
    rotation_degrees = _rotation_matrix_to_euler_degrees(rotation)
    aligned_min = [math.inf, math.inf, math.inf]
    aligned_max = [-math.inf, -math.inf, -math.inf]
    for triangle in triangles:
        for vertex in triangle:
            relative: Vector = (
                vertex[0] - center[0],
                vertex[1] - center[1],
                vertex[2] - center[2],
            )
            for axis, basis_axis in enumerate(rotation):
                coordinate = _dot(relative, basis_axis)
                aligned_min[axis] = min(aligned_min[axis], coordinate)
                aligned_max[axis] = max(aligned_max[axis], coordinate)
    aligned_dimensions: Vector = (
        aligned_max[0] - aligned_min[0],
        aligned_max[1] - aligned_min[1],
        aligned_max[2] - aligned_min[2],
    )

    eigenvalues = [max(pair[0], 1e-16) for pair in eigenpairs]
    length_to_width = math.sqrt(eigenvalues[0] / eigenvalues[1])
    width_to_height = math.sqrt(eigenvalues[1] / eigenvalues[2])
    recommended = length_to_width >= 1.2 and width_to_height >= 1.08
    confidence = "high" if length_to_width >= 1.6 and width_to_height >= 1.2 else "medium"
    if not recommended:
        confidence = "low"

    return {
        "recommended": recommended,
        "confidence": confidence,
        "method": "area-weighted principal axes",
        "rotation_degrees": {
            "x": round(rotation_degrees[0], 3),
            "y": round(rotation_degrees[1], 3),
            "z": round(rotation_degrees[2], 3),
        },
        "original_dimensions": tuple(round(value, 6) for value in bounds.dimensions),
        "aligned_dimensions": tuple(round(value, 6) for value in aligned_dimensions),
        "length_to_width_ratio": round(length_to_width, 4),
        "width_to_height_ratio": round(width_to_height, 4),
        "detail": (
            "The longest surface axis maps to +X and the thinnest axis maps to +Z. "
            "Confirm the model points forward and sits upright before solving."
        ),
    }


def _symmetric_eigenpairs(matrix: list[list[float]]) -> list[tuple[float, Vector]]:
    values = [row[:] for row in matrix]
    vectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(32):
        row, column = max(
            ((0, 1), (0, 2), (1, 2)),
            key=lambda pair: abs(values[pair[0]][pair[1]]),
        )
        off_diagonal = values[row][column]
        scale = max(abs(values[index][index]) for index in range(3))
        if abs(off_diagonal) <= max(scale * 1e-12, 1e-16):
            break
        angle = 0.5 * math.atan2(
            2.0 * off_diagonal,
            values[column][column] - values[row][row],
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        row_value = values[row][row]
        column_value = values[column][column]
        values[row][row] = (
            cosine * cosine * row_value
            - 2.0 * sine * cosine * off_diagonal
            + sine * sine * column_value
        )
        values[column][column] = (
            sine * sine * row_value
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * column_value
        )
        values[row][column] = values[column][row] = 0.0
        for index in range(3):
            if index in (row, column):
                continue
            row_entry = cosine * values[index][row] - sine * values[index][column]
            column_entry = sine * values[index][row] + cosine * values[index][column]
            values[index][row] = values[row][index] = row_entry
            values[index][column] = values[column][index] = column_entry
        for index in range(3):
            row_entry = cosine * vectors[index][row] - sine * vectors[index][column]
            column_entry = sine * vectors[index][row] + cosine * vectors[index][column]
            vectors[index][row] = row_entry
            vectors[index][column] = column_entry

    pairs = [
        (
            values[index][index],
            _normalize(
                (
                    vectors[0][index],
                    vectors[1][index],
                    vectors[2][index],
                )
            ),
        )
        for index in range(3)
    ]
    return sorted(pairs, key=lambda pair: pair[0], reverse=True)


def _rotation_matrix_to_euler_degrees(rotation: tuple[Vector, Vector, Vector]) -> Vector:
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0][1], rotation[1][1])
    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def _normalize(vector: Vector) -> Vector:
    magnitude = math.sqrt(_dot(vector, vector))
    if magnitude <= 1e-16:
        return (0.0, 0.0, 0.0)
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )


def scaled_report(report: StlReport, scale: float) -> StlReport:
    bounds = Bounds(
        minimum=tuple(value * scale for value in report.bounds.minimum),  # type: ignore[arg-type]
        maximum=tuple(value * scale for value in report.bounds.maximum),  # type: ignore[arg-type]
    )
    return StlReport(
        path=report.path,
        format=report.format,
        triangle_count=report.triangle_count,
        unique_vertex_count=report.unique_vertex_count,
        bounds=bounds,
        surface_area=report.surface_area * scale * scale,
        projected_areas=report.projected_areas.scaled(scale),
        volume=report.volume * scale * scale * scale,
        open_edge_count=report.open_edge_count,
        non_manifold_edge_count=report.non_manifold_edge_count,
        degenerate_triangle_count=report.degenerate_triangle_count,
        warnings=report.warnings,
        alignment_suggestion=report.alignment_suggestion,
        silhouette_projected_areas=(
            report.silhouette_projected_areas.scaled(scale)
            if report.silhouette_projected_areas is not None
            else None
        ),
    )


def transformed_report(
    source_path: Path,
    scale: float,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector | None = None,
) -> StlReport:
    triangles, stl_format = read_stl_triangles(source_path)
    transformed = transform_triangles(
        triangles,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    return report_for_triangles(source_path, transformed, stl_format)


def write_scaled_ascii_stl(source_path: Path, target_path: Path, scale: float) -> None:
    write_transformed_ascii_stl(source_path, target_path, scale=scale)


def write_transformed_binary_stl(
    source_path: Path,
    target_path: Path,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector | None = None,
) -> None:
    triangles, _ = read_stl_triangles(source_path)
    transformed = transform_triangles(
        triangles,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    write_binary_stl_triangles(target_path, transformed, header="AeroLab transformed body in solver meters")


def write_binary_stl_triangles(
    target_path: Path,
    triangles: Iterable[Triangle],
    header: str = "AeroLab binary STL",
) -> None:
    triangle_list = list(triangles)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = header.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    with target_path.open("wb") as stream:
        stream.write(header_bytes)
        stream.write(struct.pack("<I", len(triangle_list)))
        for triangle in triangle_list:
            normal = _normal(triangle)
            values = normal + triangle[0] + triangle[1] + triangle[2]
            stream.write(struct.pack("<12fH", *values, 0))


def write_transformed_ascii_stl(
    source_path: Path,
    target_path: Path,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector | None = None,
) -> None:
    triangles, _ = read_stl_triangles(source_path)
    transformed = transform_triangles(
        triangles,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        f.write("solid body_meters\n")
        for triangle in transformed:
            normal = _normal(triangle)
            f.write(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
            f.write("    outer loop\n")
            for vertex in triangle:
                f.write(
                    "      vertex "
                    f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n"
                )
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid body_meters\n")


def transform_triangles(
    triangles: list[Triangle],
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector | None = None,
) -> list[Triangle]:
    center = _triangle_center(triangles) if rotation_center is None else _finite_vector(
        rotation_center,
        "Rotation center",
    )
    return [
        tuple(
            transform_point(
                vertex,
                scale=scale,
                source_flow_direction=source_flow_direction,
                source_up_direction=source_up_direction,
                target_flow_axis=target_flow_axis,
                rotation_degrees=rotation_degrees,
                translation=translation,
                rotation_center=center,
            )
            for vertex in triangle
        )  # type: ignore[misc]
        for triangle in triangles
    ]


def transform_point(
    point: Vector,
    *,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector = (0.0, 0.0, 0.0),
) -> Vector:
    """Transform a source-frame point with the same contract used for solver STL geometry."""
    source_point = _finite_vector(point, "Point")
    center = _finite_vector(rotation_center, "Rotation center")
    offset = _finite_vector(translation, "Translation")
    factor = float(scale)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Geometry scale must be a finite positive value.")
    rotation = _finite_vector(rotation_degrees, "Rotation")
    rotation_radians: Vector = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )
    basis = _orientation_basis(source_flow_direction, source_up_direction, target_flow_axis)
    transformed = _transform_vertex(
        _rotate_vertex(source_point, center, rotation_radians),
        basis,
        factor,
    )
    return (
        transformed[0] + offset[0],
        transformed[1] + offset[1],
        transformed[2] + offset[2],
    )


def transform_direction(
    direction: Vector,
    *,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
) -> Vector:
    """Rotate and orient a source-frame direction without translating or scaling it."""
    source_direction = _normalize(_finite_vector(direction, "Direction"))
    rotation = _finite_vector(rotation_degrees, "Rotation")
    rotation_radians: Vector = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )
    rotated = _rotate_vertex(source_direction, (0.0, 0.0, 0.0), rotation_radians)
    basis = _orientation_basis(source_flow_direction, source_up_direction, target_flow_axis)
    return _normalize(_transform_vertex(rotated, basis, 1.0))


def _coerce_float(value: object) -> float:
    if isinstance(value, (str, bytes, bytearray, memoryview, SupportsFloat, SupportsIndex)):
        return float(value)
    raise TypeError


def _finite_vector(value: object, label: str) -> Vector:
    if isinstance(value, dict):
        try:
            components = (value["x"], value["y"], value["z"])
        except KeyError as exc:
            raise ValueError(f"{label} requires finite X, Y, and Z values.") from exc
    else:
        if not isinstance(value, Iterable):
            raise ValueError(f"{label} requires finite X, Y, and Z values.")
        try:
            sequence = tuple(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} requires finite X, Y, and Z values.") from exc
        if len(sequence) != 3:
            raise ValueError(f"{label} requires finite X, Y, and Z values.")
        components = (sequence[0], sequence[1], sequence[2])
    try:
        values: Vector = (
            _coerce_float(components[0]),
            _coerce_float(components[1]),
            _coerce_float(components[2]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} requires finite X, Y, and Z values.") from exc
    if not all(math.isfinite(component) for component in values):
        raise ValueError(f"{label} requires finite X, Y, and Z values.")
    return values


def translated_report(report: StlReport, translation: Vector) -> StlReport:
    bounds = Bounds(
        minimum=tuple(report.bounds.minimum[axis] + float(translation[axis]) for axis in range(3)),  # type: ignore[arg-type]
        maximum=tuple(report.bounds.maximum[axis] + float(translation[axis]) for axis in range(3)),  # type: ignore[arg-type]
    )
    return StlReport(
        path=report.path,
        format=report.format,
        triangle_count=report.triangle_count,
        unique_vertex_count=report.unique_vertex_count,
        bounds=bounds,
        surface_area=report.surface_area,
        projected_areas=report.projected_areas,
        volume=report.volume,
        open_edge_count=report.open_edge_count,
        non_manifold_edge_count=report.non_manifold_edge_count,
        degenerate_triangle_count=report.degenerate_triangle_count,
        warnings=report.warnings,
        alignment_suggestion=report.alignment_suggestion,
        silhouette_projected_areas=report.silhouette_projected_areas,
    )


def _triangle_center(triangles: list[Triangle]) -> Vector:
    bounds = _bounds(triangles)
    return tuple((bounds.minimum[index] + bounds.maximum[index]) / 2.0 for index in range(3))  # type: ignore[return-value]


def _rotate_vertex(vertex: Vector, center: Vector, rotation_radians: Vector) -> Vector:
    x = vertex[0] - center[0]
    y = vertex[1] - center[1]
    z = vertex[2] - center[2]
    rx, ry, rz = rotation_radians

    cos_x, sin_x = math.cos(rx), math.sin(rx)
    y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
    cos_z, sin_z = math.cos(rz), math.sin(rz)
    x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z
    return (x + center[0], y + center[1], z + center[2])


def _orientation_basis(
    source_flow_direction: str,
    source_up_direction: str,
    target_flow_axis: str,
) -> tuple[Vector, Vector, Vector, Vector, Vector, Vector]:
    source_flow = _signed_axis(source_flow_direction)
    source_up = _signed_axis(source_up_direction)
    if abs(_dot(source_flow, source_up)) > 1e-9:
        raise ValueError("Source flow direction and source up direction must be perpendicular.")

    target_flow = _signed_axis(target_flow_axis)
    target_up: Vector = (0.0, 0.0, 1.0) if abs(target_flow[2]) < 1e-9 else (0.0, 1.0, 0.0)
    source_side = _cross(source_up, source_flow)
    target_side = _cross(target_up, target_flow)
    return (source_flow, source_side, source_up, target_flow, target_side, target_up)


def _transform_vertex(
    vertex: Vector,
    basis: tuple[Vector, Vector, Vector, Vector, Vector, Vector],
    scale: float,
) -> Vector:
    source_flow, source_side, source_up, target_flow, target_side, target_up = basis
    flow_component = _dot(vertex, source_flow) * scale
    side_component = _dot(vertex, source_side) * scale
    up_component = _dot(vertex, source_up) * scale
    return (
        flow_component * target_flow[0] + side_component * target_side[0] + up_component * target_up[0],
        flow_component * target_flow[1] + side_component * target_side[1] + up_component * target_up[1],
        flow_component * target_flow[2] + side_component * target_side[2] + up_component * target_up[2],
    )


def _signed_axis(axis: str) -> Vector:
    key = axis.strip().lower()
    if key in SIGNED_AXES:
        return SIGNED_AXES[key]
    raise ValueError(f"Unsupported signed axis: {axis}")


DEFAULT_PREVIEW_TRIANGLE_LIMIT = 120_000


def mesh_preview(
    path: Path,
    max_triangles: int = DEFAULT_PREVIEW_TRIANGLE_LIMIT,
    source_flow_direction: str | None = None,
    source_up_direction: str | None = None,
    target_flow_axis: str = "x",
) -> dict[str, object]:
    triangles, stl_format = read_stl_triangles(path)
    if source_flow_direction or source_up_direction:
        if not source_flow_direction or not source_up_direction:
            raise ValueError("Both source flow and source up are required for an oriented preview.")
        triangles = transform_triangles(
            triangles,
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            target_flow_axis=target_flow_axis,
        )
    bounds = _bounds(triangles)
    center = (
        (bounds.minimum[0] + bounds.maximum[0]) / 2.0,
        (bounds.minimum[1] + bounds.maximum[1]) / 2.0,
        (bounds.minimum[2] + bounds.maximum[2]) / 2.0,
    )
    max_dim = max(bounds.dimensions) or 1.0
    preview_scale = 3.8 / max_dim
    if len(triangles) <= max_triangles:
        sample_step = 1.0
        sampled = triangles
    else:
        sample_step = len(triangles) / max_triangles
        sampled = [
            triangles[min(len(triangles) - 1, math.floor(index * sample_step))]
            for index in range(max_triangles)
        ]
    normalized = []

    for triangle in sampled:
        normal = _normal(triangle)
        normalized.append(
            {
                "v": [
                    round((vertex[0] - center[0]) * preview_scale, 6)
                    if axis == 0
                    else round((vertex[1] - center[1]) * preview_scale, 6)
                    if axis == 1
                    else round((vertex[2] - center[2]) * preview_scale, 6)
                    for vertex in triangle
                    for axis in range(3)
                ],
                "n": [round(value, 6) for value in normal],
            }
        )

    return {
        "format": stl_format,
        "triangleCount": len(triangles),
        "sampledTriangleCount": len(normalized),
        "sampleStep": sample_step,
        "previewTriangleLimit": max_triangles,
        "isComplete": len(normalized) == len(triangles),
        "bounds": bounds.to_dict(),
        "normalizedCenter": [round(value, 9) for value in center],
        "normalizedScale": preview_scale,
        "triangles": normalized,
    }


def _load_stl(data: bytes) -> tuple[list[Triangle], str]:
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data), "binary"
    return _load_ascii_stl(data), "ascii"


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50
    return expected_size == len(data)


def _load_binary_stl(data: bytes) -> list[Triangle]:
    triangle_count = struct.unpack("<I", data[80:84])[0]
    triangles: list[Triangle] = []
    offset = 84
    for _ in range(triangle_count):
        chunk = data[offset : offset + 50]
        values = struct.unpack("<12fH", chunk)
        vertices = values[3:12]
        triangles.append(
            (
                (vertices[0], vertices[1], vertices[2]),
                (vertices[3], vertices[4], vertices[5]),
                (vertices[6], vertices[7], vertices[8]),
            )
        )
        offset += 50
    return triangles


def _load_ascii_stl(data: bytes) -> list[Triangle]:
    text = data.decode("utf-8", errors="ignore")
    vertices: list[Vector] = []
    triangles: list[Triangle] = []

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
            vertices.append(vertex)
            if len(vertices) == 3:
                triangles.append((vertices[0], vertices[1], vertices[2]))
                vertices = []

    return triangles


def _bounds(triangles: Iterable[Triangle]) -> Bounds:
    vertices = [vertex for triangle in triangles for vertex in triangle]
    return Bounds(
        minimum=(
            min(v[0] for v in vertices),
            min(v[1] for v in vertices),
            min(v[2] for v in vertices),
        ),
        maximum=(
            max(v[0] for v in vertices),
            max(v[1] for v in vertices),
            max(v[2] for v in vertices),
        ),
    )


def _edge_counts(triangles: Iterable[Triangle]) -> Counter[tuple[Vector, Vector]]:
    counts: Counter[tuple[Vector, Vector]] = Counter()
    for triangle in triangles:
        vertices = [_quantize_vertex(v) for v in triangle]
        edges = (
            _edge_key(vertices[0], vertices[1]),
            _edge_key(vertices[1], vertices[2]),
            _edge_key(vertices[2], vertices[0]),
        )
        counts.update(edges)
    return counts


def _edge_key(a: Vector, b: Vector) -> tuple[Vector, Vector]:
    return (a, b) if a <= b else (b, a)


def _quantize_vertex(vertex: Vector) -> Vector:
    return (round(vertex[0], 9), round(vertex[1], 9), round(vertex[2], 9))


def _triangle_area(triangle: Triangle) -> float:
    a, b, c = triangle
    ab = _sub(b, a)
    ac = _sub(c, a)
    return 0.5 * _length(_cross(ab, ac))


def _projected_areas(triangles: list[Triangle], areas: list[float]) -> ProjectedAreas:
    projected = [0.0, 0.0, 0.0]
    for triangle, area in zip(triangles, areas):
        normal = _normal(triangle)
        for axis in range(3):
            projected[axis] += abs(normal[axis]) * area * 0.5
    return ProjectedAreas(x=projected[0], y=projected[1], z=projected[2])


def silhouette_projected_areas_for_triangles(
    triangles: Iterable[Triangle],
    scanline_count: int | None = None,
) -> ProjectedAreas:
    """Estimate the visible projected union without filling concave model gaps."""
    triangle_list = list(triangles)
    if not triangle_list:
        return ProjectedAreas(x=0.0, y=0.0, z=0.0)
    if scanline_count is None:
        scanline_count = _silhouette_scanline_count(len(triangle_list))
    scanline_count = max(128, int(scanline_count))
    planes = ((1, 2), (0, 2), (0, 1))
    areas = [
        _projected_triangle_union_area(triangle_list, first, second, scanline_count)
        for first, second in planes
    ]
    return ProjectedAreas(x=areas[0], y=areas[1], z=areas[2])


def silhouette_projected_area_for_axis(
    triangles: Iterable[Triangle],
    axis: str | int,
    scanline_count: int | None = None,
) -> float:
    triangle_list = list(triangles)
    if not triangle_list:
        return 0.0
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, axis) if isinstance(axis, str) else axis
    if axis_index not in (0, 1, 2):
        raise ValueError(f"Unsupported silhouette axis: {axis}")
    if scanline_count is None:
        scanline_count = _silhouette_scanline_count(len(triangle_list))
    planes = ((1, 2), (0, 2), (0, 1))
    first, second = planes[int(axis_index)]
    return _projected_triangle_union_area(
        triangle_list,
        first,
        second,
        max(128, int(scanline_count)),
    )


def _silhouette_scanline_count(triangle_count: int) -> int:
    if triangle_count <= 100_000:
        return 512
    if triangle_count <= 500_000:
        return 384
    return 256


def _projected_triangle_union_area(
    triangles: list[Triangle],
    first_axis: int,
    second_axis: int,
    scanline_count: int,
) -> float:
    projected = [
        tuple((vertex[first_axis], vertex[second_axis]) for vertex in triangle)
        for triangle in triangles
    ]
    second_min = min(point[1] for triangle in projected for point in triangle)
    second_max = max(point[1] for triangle in projected for point in triangle)
    span = second_max - second_min
    if span <= 1e-15:
        return 0.0
    step = span / float(scanline_count)
    rows: list[list[tuple[float, float]]] = [[] for _ in range(scanline_count)]
    epsilon = max(span, 1.0) * 1e-13

    for triangle in projected:
        area_twice = (
            (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
            - (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[0][0])
        )
        if abs(area_twice) <= epsilon * epsilon:
            continue
        low = min(point[1] for point in triangle)
        high = max(point[1] for point in triangle)
        first_row = max(0, int(math.ceil((low - second_min) / step - 0.5 - 1e-12)))
        stop_row = min(
            scanline_count,
            int(math.ceil((high - second_min) / step - 0.5 - 1e-12)),
        )
        for row_index in range(first_row, stop_row):
            coordinate = second_min + (row_index + 0.5) * step
            crossings: list[float] = []
            for edge_index in range(3):
                edge_start = triangle[edge_index]
                edge_end = triangle[(edge_index + 1) % 3]
                edge_low = min(edge_start[1], edge_end[1])
                edge_high = max(edge_start[1], edge_end[1])
                if edge_high - edge_low <= epsilon or not (edge_low <= coordinate < edge_high):
                    continue
                fraction = (coordinate - edge_start[1]) / (edge_end[1] - edge_start[1])
                crossings.append(edge_start[0] + fraction * (edge_end[0] - edge_start[0]))
            if len(crossings) >= 2:
                interval = (min(crossings), max(crossings))
                if interval[1] - interval[0] > epsilon:
                    rows[row_index].append(interval)

    integrated_width = 0.0
    for intervals in rows:
        if not intervals:
            continue
        intervals.sort()
        current_start, current_end = intervals[0]
        for interval_start, interval_end in intervals[1:]:
            if interval_start <= current_end + epsilon:
                current_end = max(current_end, interval_end)
            else:
                integrated_width += current_end - current_start
                current_start, current_end = interval_start, interval_end
        integrated_width += current_end - current_start
    return integrated_width * step


def _normal(triangle: Triangle) -> Vector:
    a, b, c = triangle
    normal = _cross(_sub(b, a), _sub(c, a))
    length = _length(normal)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def _signed_tetra_volume(triangle: Triangle) -> float:
    a, b, c = triangle
    return _dot(a, _cross(b, c)) / 6.0


def _readiness(
    bounds: Bounds,
    triangle_count: int,
    open_edges: int,
    non_manifold_edges: int,
    degenerate_count: int,
) -> _ReadinessResult:
    items: list[dict[str, str]] = []
    score = 100

    def add(label: str, status: str, detail: str, penalty: int = 0) -> None:
        nonlocal score
        items.append({"label": label, "status": status, "detail": detail})
        score -= penalty

    if open_edges:
        add("Watertight mesh", "fail", f"{open_edges} open edges need hole filling.", 28)
    else:
        add("Watertight mesh", "pass", "No open edges detected.")

    if non_manifold_edges:
        add("Manifold surface", "fail", f"{non_manifold_edges} non-manifold edges need cleanup.", 26)
    else:
        add("Manifold surface", "pass", "No non-manifold edges detected.")

    if degenerate_count:
        add("Triangle quality", "fail", f"{degenerate_count} zero-area triangles should be removed.", 18)
    else:
        add("Triangle quality", "pass", "No zero-area triangles detected.")

    dims = bounds.dimensions
    max_dim = max(dims)
    min_dim = min(dims)
    if min_dim <= 0:
        add("3D dimensions", "fail", "At least one axis has zero thickness.", 22)
    else:
        ratio = max_dim / min_dim
        if ratio > 18:
            add("3D dimensions", "warn", f"Very thin bounding box ratio ({ratio:.1f}:1). Confirm orientation.", 8)
        else:
            add("3D dimensions", "pass", "All axes have usable thickness.")

    if triangle_count < 250:
        add("Surface detail", "warn", "Very low triangle count; useful for testing, not final vehicle CFD.", 8)
    elif triangle_count > 250_000:
        add(
            "Surface detail",
            "pass",
            "High-detail STL retained; the exact surface will increase volume-meshing cost.",
        )
    else:
        add("Surface detail", "pass", "Triangle count is reasonable for setup.")

    if max_dim > 100 or (0 < max_dim < 0.05):
        add("Scale sanity", "warn", "Raw dimensions look unusual; confirm units or set real length.", 10)
    else:
        add("Scale sanity", "pass", "Raw dimensions are within a normal working range.")

    score = max(0, min(100, score))
    failed = sum(1 for item in items if item["status"] == "fail")
    warned = sum(1 for item in items if item["status"] == "warn")
    if failed:
        status = "cleanup_required"
    elif warned:
        status = "setup_check"
    else:
        status = "ready"

    return {
        "score": score,
        "status": status,
        "failed": failed,
        "warnings": warned,
        "items": items,
    }


def _warnings(
    bounds: Bounds,
    open_edges: int,
    non_manifold_edges: int,
    degenerate_count: int,
) -> list[str]:
    warnings: list[str] = []
    dims = bounds.dimensions
    max_dim = max(dims)
    min_dim = min(dims)

    if open_edges:
        warnings.append("Mesh has open edges; scanned models often need hole filling before CFD.")
    if non_manifold_edges:
        warnings.append("Mesh has non-manifold edges; clean or remesh before generating a CFD volume mesh.")
    if degenerate_count:
        warnings.append("Mesh has zero-area or near-zero-area triangles.")
    if min_dim <= 0:
        warnings.append("One model dimension is zero; check orientation and export settings.")
    if max_dim > 100:
        warnings.append("Model may be in millimeters or centimeters; confirm scale before simulation.")
    if 0 < max_dim < 0.05:
        warnings.append("Model is very small; confirm units before simulation.")

    return warnings


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a: Vector) -> float:
    return math.sqrt(_dot(a, a))
