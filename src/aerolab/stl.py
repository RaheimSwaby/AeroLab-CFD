from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, TypedDict

import numpy as np

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


# ---------------------------------------------------------------------------
# Vectorized triangle-array core
#
# Triangles are handled internally as one float64 array of shape (n, 3, 3):
# n triangles, three vertices, xyz. Parsed meshes are cached by path, size
# and mtime so that inspect, transform, preview and feature detection over
# the same file never re-parse or re-allocate the triangle soup.
# ---------------------------------------------------------------------------


def _as_array(triangles: Iterable[Triangle]) -> np.ndarray:
    """Convert tuple triangles to the internal (n, 3, 3) float64 array."""
    if isinstance(triangles, np.ndarray):
        array = np.asarray(triangles, dtype=np.float64)
        if array.ndim == 3 and array.shape[1:] == (3, 3):
            return array
        raise ValueError("triangle array must have shape (n, 3, 3)")
    triangle_list = list(triangles)
    if not triangle_list:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(triangle_list, dtype=np.float64).reshape(-1, 3, 3)


def _as_tuples(array: np.ndarray) -> list[Triangle]:
    """Convert the internal array back to the public tuple representation."""
    return [
        (
            (float(triangle[0, 0]), float(triangle[0, 1]), float(triangle[0, 2])),
            (float(triangle[1, 0]), float(triangle[1, 1]), float(triangle[1, 2])),
            (float(triangle[2, 0]), float(triangle[2, 1]), float(triangle[2, 2])),
        )
        for triangle in array
    ]


def _cross_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def _triangle_areas(array: np.ndarray) -> np.ndarray:
    crosses = _cross_arrays(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
    return 0.5 * np.linalg.norm(crosses, axis=1)


def _triangle_normals(array: np.ndarray) -> np.ndarray:
    crosses = _cross_arrays(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    normals = np.zeros_like(crosses)
    valid = lengths > 1e-12
    normals[valid] = crosses[valid] / lengths[valid, None]
    return normals


def _bounds_array(array: np.ndarray) -> Bounds:
    vertices = array.reshape(-1, 3)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return Bounds(
        minimum=(float(minimum[0]), float(minimum[1]), float(minimum[2])),
        maximum=(float(maximum[0]), float(maximum[1]), float(maximum[2])),
    )


def _quantize_array(vertices: np.ndarray) -> np.ndarray:
    """Quantize vertices to the same 9-decimal grid used for topology keys."""
    return np.round(vertices * 1e9).astype(np.int64)


def _unique_vertex_count(array: np.ndarray) -> int:
    keys = _quantize_array(array.reshape(-1, 3))
    return int(np.unique(keys, axis=0).shape[0])


def _edge_open_non_manifold(array: np.ndarray) -> tuple[int, int]:
    triangles = _quantize_array(array)  # (n, 3, 3)
    first = triangles
    second = np.roll(triangles, -1, axis=1)
    # Lexicographic canonical order for each edge's endpoint pair.
    less = (
        (first[..., 0] < second[..., 0])
        | ((first[..., 0] == second[..., 0]) & (first[..., 1] < second[..., 1]))
        | (
            (first[..., 0] == second[..., 0])
            & (first[..., 1] == second[..., 1])
            & (first[..., 2] <= second[..., 2])
        )
    )
    low = np.where(less[..., None], first, second)
    high = np.where(less[..., None], second, first)
    edges = np.concatenate((low, high), axis=2).reshape(-1, 6)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2))


def _projected_areas_array(array: np.ndarray, areas: np.ndarray) -> ProjectedAreas:
    normals = _triangle_normals(array)
    projected = (np.abs(normals) * (areas * 0.5)[:, None]).sum(axis=0)
    return ProjectedAreas(x=float(projected[0]), y=float(projected[1]), z=float(projected[2]))


def _volume_array(array: np.ndarray) -> float:
    signed = np.einsum(
        "ij,ij->i",
        array[:, 0],
        _cross_arrays(array[:, 1], array[:, 2]),
    )
    return float(abs(signed.sum()) / 6.0)


def _silhouette_scanline_count(triangle_count: int) -> int:
    if triangle_count <= 100_000:
        return 512
    if triangle_count <= 500_000:
        return 384
    return 256


def _projected_triangle_union_area_array(
    array: np.ndarray,
    first_axis: int,
    second_axis: int,
    scanline_count: int,
) -> float:
    projected = array[:, :, (first_axis, second_axis)]
    x_values = projected[..., 0]
    y_values = projected[..., 1]
    second_min = float(y_values.min())
    second_max = float(y_values.max())
    span = second_max - second_min
    if span <= 1e-15:
        return 0.0
    step = span / float(scanline_count)
    epsilon = max(span, 1.0) * 1e-13

    # Skip triangles that project to a line, matching the scalar reference.
    area_twice = (
        (x_values[:, 1] - x_values[:, 0]) * (y_values[:, 2] - y_values[:, 0])
        - (y_values[:, 1] - y_values[:, 0]) * (x_values[:, 2] - x_values[:, 0])
    )
    usable = np.abs(area_twice) > epsilon * epsilon

    x0 = x_values
    x1 = np.roll(x_values, -1, axis=1)
    y0 = y_values
    y1 = np.roll(y_values, -1, axis=1)
    low = np.minimum(y0, y1)
    high = np.maximum(y0, y1)
    tall = (high - low) > epsilon
    active = usable[:, None] & tall

    # Flat, fully vectorized scanline rasterisation: expand every edge into
    # the scanlines it crosses, compute all crossing coordinates at once,
    # then reduce per (scanline, triangle) to that triangle's interval and
    # merge intervals per scanline. This replaces a per-row mask loop whose
    # temporary allocations dominated the whole inspect path.
    row_lo = np.ceil((low - second_min) / step - 0.5 - 1e-12).astype(np.int64)
    row_hi = np.ceil((high - second_min) / step - 0.5 - 1e-12).astype(np.int64)
    row_lo = np.clip(row_lo, 0, scanline_count)
    row_hi = np.clip(row_hi, 0, scanline_count)
    counts = np.maximum(row_hi - row_lo, 0)
    counts[~active] = 0
    flat_counts = counts.ravel()
    total_crossings = int(flat_counts.sum())
    if total_crossings == 0:
        return 0.0

    triangle_per_edge = np.broadcast_to(
        np.arange(len(array), dtype=np.int64)[:, None], counts.shape
    ).ravel()
    group_end = np.cumsum(flat_counts)
    within = np.arange(total_crossings, dtype=np.int64) - np.repeat(
        group_end - flat_counts, flat_counts
    )
    rows = np.repeat(row_lo.ravel(), flat_counts) + within
    edge_x0 = np.repeat(x0.ravel(), flat_counts)
    edge_x1 = np.repeat(x1.ravel(), flat_counts)
    edge_y0 = np.repeat(y0.ravel(), flat_counts)
    edge_y1 = np.repeat(y1.ravel(), flat_counts)
    triangles = np.repeat(triangle_per_edge, flat_counts)

    coordinates = second_min + (rows + 0.5) * step
    edge_low = np.repeat(low.ravel(), flat_counts)
    edge_high = np.repeat(high.ravel(), flat_counts)
    valid = (coordinates >= edge_low) & (coordinates < edge_high)
    rows = rows[valid]
    triangles = triangles[valid]
    fraction = (coordinates[valid] - edge_y0[valid]) / (edge_y1[valid] - edge_y0[valid])
    crossings = edge_x0[valid] + fraction * (edge_x1[valid] - edge_x0[valid])

    order = np.lexsort((crossings, triangles, rows))
    sorted_rows = rows[order]
    sorted_triangles = triangles[order]
    sorted_crossings = crossings[order]
    boundary = np.r_[
        True,
        (np.diff(sorted_rows) != 0) | (np.diff(sorted_triangles) != 0),
    ]
    group_starts = np.nonzero(boundary)[0]
    group_ends = np.r_[group_starts[1:], len(sorted_crossings)]
    group_sizes = group_ends - group_starts
    keep = group_sizes >= 2
    interval_rows = sorted_rows[group_starts[keep]]
    interval_starts = sorted_crossings[group_starts[keep]]
    interval_ends = sorted_crossings[group_ends[keep] - 1]
    wide = (interval_ends - interval_starts) > epsilon
    interval_rows = interval_rows[wide]
    interval_starts = interval_starts[wide]
    interval_ends = interval_ends[wide]
    if len(interval_rows) == 0:
        return 0.0

    order_intervals = np.lexsort((interval_starts, interval_rows))
    interval_rows = interval_rows[order_intervals]
    interval_starts = interval_starts[order_intervals]
    interval_ends = interval_ends[order_intervals]

    total_width = 0.0
    row_boundaries = np.r_[0, np.nonzero(np.diff(interval_rows) != 0)[0] + 1, len(interval_rows)]
    for start, end in zip(row_boundaries[:-1], row_boundaries[1:], strict=True):
        starts = interval_starts[start:end]
        ends = interval_ends[start:end]
        running_max = np.maximum.accumulate(ends)
        new_segment = starts > np.roll(running_max, 1) + epsilon
        new_segment[0] = True
        segment_start_positions = np.nonzero(new_segment)[0]
        segment_end_positions = np.r_[segment_start_positions[1:], len(starts)] - 1
        total_width += float(
            (running_max[segment_end_positions] - starts[segment_start_positions]).sum()
        )
    return total_width * step


def silhouette_projected_areas_for_triangles(
    triangles: Iterable[Triangle],
    scanline_count: int | None = None,
) -> ProjectedAreas:
    """Estimate the visible projected union without filling concave model gaps."""
    array = _as_array(triangles)
    if len(array) == 0:
        return ProjectedAreas(x=0.0, y=0.0, z=0.0)
    if scanline_count is None:
        scanline_count = _silhouette_scanline_count(len(array))
    scanline_count = max(128, int(scanline_count))
    planes = ((1, 2), (0, 2), (0, 1))
    areas = [
        _projected_triangle_union_area_array(array, first, second, scanline_count)
        for first, second in planes
    ]
    return ProjectedAreas(x=areas[0], y=areas[1], z=areas[2])


def silhouette_projected_area_for_axis(
    triangles: Iterable[Triangle],
    axis: str | int,
    scanline_count: int | None = None,
) -> float:
    array = _as_array(triangles)
    if len(array) == 0:
        return 0.0
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, axis) if isinstance(axis, str) else axis
    if axis_index not in (0, 1, 2):
        raise ValueError(f"Unsupported silhouette axis: {axis}")
    if scanline_count is None:
        scanline_count = _silhouette_scanline_count(len(array))
    planes = ((1, 2), (0, 2), (0, 1))
    first, second = planes[int(axis_index)]
    return _projected_triangle_union_area_array(
        array,
        first,
        second,
        max(128, int(scanline_count)),
    )


# ---------------------------------------------------------------------------
# STL parsing with a parsed-mesh cache
# ---------------------------------------------------------------------------

_BINARY_STL_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50
    return expected_size == len(data)


def _load_binary_stl_array(data: bytes) -> np.ndarray:
    triangle_count = struct.unpack("<I", data[80:84])[0]
    records = np.frombuffer(data, dtype=_BINARY_STL_DTYPE, count=triangle_count, offset=84)
    return np.asarray(records["vertices"], dtype=np.float64)


def _load_ascii_stl_array(data: bytes) -> np.ndarray:
    text = data.decode("utf-8", errors="ignore")
    rows: list[list[str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            rows.append(parts[1:4])
    if not rows:
        return np.empty((0, 3, 3), dtype=np.float64)
    vertices = np.asarray(rows, dtype=np.float64)
    usable = (len(vertices) // 3) * 3
    return vertices[:usable].reshape(-1, 3, 3)


def _load_stl_array(data: bytes) -> tuple[np.ndarray, str]:
    if _looks_like_binary_stl(data):
        return _load_binary_stl_array(data), "binary"
    return _load_ascii_stl_array(data), "ascii"


@lru_cache(maxsize=4)
def _read_stl_cached(path_str: str, size: int, mtime_ns: int) -> tuple[np.ndarray, str]:
    data = Path(path_str).read_bytes()
    return _load_stl_array(data)


def _read_stl_array(path: Path) -> tuple[np.ndarray, str]:
    stat = path.stat()
    return _read_stl_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def clear_stl_cache() -> None:
    """Drop cached parsed meshes, e.g. after a file is rewritten in place."""
    _read_stl_cached.cache_clear()


def read_stl_triangles(path: Path) -> tuple[list[Triangle], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {path}")
    array, stl_format = _read_stl_array(path)
    return _as_tuples(array), stl_format


def inspect_stl(path: Path) -> StlReport:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {path}")

    array, stl_format = _read_stl_array(path)
    if len(array) == 0:
        raise ValueError(f"No triangles found in STL: {path}")
    return _report_for_array(path, array, stl_format)


def detect_aero_features(
    path: Path,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    """Find geometry that resembles a rear underbody diffuser in a canonical frame."""
    array, _ = _read_stl_array(path)
    canonical = _transform_array(
        array,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis="x",
        rotation_degrees=rotation_degrees,
    )
    return _detect_aero_features_for_array(canonical)


def detect_aero_features_for_triangles(triangles: list[Triangle]) -> dict[str, object]:
    """Return conservative diffuser candidates; absence is not proof of no diffuser."""
    return _detect_aero_features_for_array(_as_array(triangles))


def _detect_aero_features_for_array(array: np.ndarray) -> dict[str, object]:
    if len(array) == 0:
        return _aero_feature_result([])
    bounds = _bounds_array(array)
    length, width, height = bounds.dimensions
    if min(length, width, height) <= 1e-9:
        return _aero_feature_result([])

    areas = _triangle_areas(array)
    centroids = array.mean(axis=1)
    normals = _triangle_normals(array)

    rear_start = bounds.minimum[0] + 0.52 * length
    lower_limit = bounds.minimum[2] + 0.42 * height
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(
            np.abs(normals[:, 2]) > 1e-12,
            -normals[:, 0] / np.where(np.abs(normals[:, 2]) > 1e-12, normals[:, 2], 1.0),
            0.0,
        )
    angles = np.degrees(np.arctan(slopes))
    mask = (
        (areas > 1e-12)
        & (centroids[:, 0] >= rear_start)
        & (centroids[:, 2] <= lower_limit)
        & (np.abs(normals[:, 2]) >= 0.35)
        & (angles >= 2.0)
        & (angles <= 30.0)
    )
    candidate_indices = np.nonzero(mask)[0].tolist()
    if not candidate_indices:
        return _aero_feature_result([])

    vertex_to_triangles: dict[tuple[int, int, int], list[int]] = {}
    quantized = _quantize_array(array)
    for index in candidate_indices:
        for vertex in quantized[index]:
            key = (int(vertex[0]), int(vertex[1]), int(vertex[2]))
            vertex_to_triangles.setdefault(key, []).append(index)

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
        component_array = array[np.asarray(component, dtype=np.int64)]
        component_bounds = _bounds_array(component_array)
        x_span, y_span, z_span = component_bounds.dimensions
        component_areas = areas[np.asarray(component, dtype=np.int64)]
        area = float(component_areas.sum())
        weighted_angle = float((component_areas * angles[np.asarray(component, dtype=np.int64)]).sum()) / max(area, 1e-12)
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
    return _report_for_array(path, _as_array(triangles), stl_format)


def _report_for_array(path: Path, array: np.ndarray, stl_format: str) -> StlReport:
    bounds = _bounds_array(array)
    unique_vertices = _unique_vertex_count(array)
    areas = _triangle_areas(array)
    degenerate_count = int(np.count_nonzero(areas <= 1e-12))
    open_edges, non_manifold_edges = _edge_open_non_manifold(array)
    surface_area = float(areas.sum())
    projected_areas = _projected_areas_array(array, areas)
    silhouette_projected_areas = silhouette_projected_areas_for_triangles(array)
    volume = _volume_array(array)
    warnings = _warnings(bounds, open_edges, non_manifold_edges, degenerate_count)
    alignment_suggestion = _principal_axis_alignment(array, areas, bounds)

    return StlReport(
        path=str(path),
        format=stl_format,
        triangle_count=len(array),
        unique_vertex_count=unique_vertices,
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
    array: np.ndarray,
    areas: np.ndarray,
    bounds: Bounds,
) -> dict[str, object] | None:
    """Fit a vehicle-like length/width/height frame to the STL surface."""
    valid = areas > 1e-12
    total_area = float(areas[valid].sum())
    if total_area <= 1e-12:
        return None

    triangles = array[valid]
    weights = areas[valid]
    vertex_sum = triangles.sum(axis=1)  # (n, 3)
    first = (weights[:, None] * vertex_sum / 3.0).sum(axis=0)
    diagonal_sum = np.einsum("nij,nik->njk", triangles, triangles)  # (n, 3, 3)
    second = (
        weights[:, None, None]
        * (
            vertex_sum[:, :, None] * vertex_sum[:, None, :]
            + diagonal_sum
        )
        / 12.0
    ).sum(axis=0)

    center = first / total_area
    covariance = second / total_area - np.outer(center, center)
    eigenpairs = _symmetric_eigenpairs(covariance.tolist())
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
    rotation_array = np.asarray(rotation, dtype=np.float64)
    relative = array.reshape(-1, 3) - center
    aligned = relative @ rotation_array.T
    aligned_min = aligned.min(axis=0)
    aligned_max = aligned.max(axis=0)
    aligned_dimensions: Vector = (
        float(aligned_max[0] - aligned_min[0]),
        float(aligned_max[1] - aligned_min[1]),
        float(aligned_max[2] - aligned_min[2]),
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
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {source_path}")
    array, stl_format = _read_stl_array(source_path)
    transformed = _transform_array(
        array,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    return _report_for_array(source_path, transformed, stl_format)


def write_scaled_ascii_stl(source_path: Path, target_path: Path, scale: float) -> None:
    write_transformed_ascii_stl(source_path, target_path, scale=scale)


def _euler_matrix(rotation_radians: Vector) -> np.ndarray:
    rx, ry, rz = rotation_radians
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)
    rotate_x = np.array(
        ((1.0, 0.0, 0.0), (0.0, cos_x, -sin_x), (0.0, sin_x, cos_x)),
        dtype=np.float64,
    )
    rotate_y = np.array(
        ((cos_y, 0.0, sin_y), (0.0, 1.0, 0.0), (-sin_y, 0.0, cos_y)),
        dtype=np.float64,
    )
    rotate_z = np.array(
        ((cos_z, -sin_z, 0.0), (sin_z, cos_z, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotate_z @ rotate_y @ rotate_x


def _basis_matrix(
    source_flow_direction: str,
    source_up_direction: str,
    target_flow_axis: str,
) -> np.ndarray:
    source_flow, source_side, source_up, target_flow, target_side, target_up = _orientation_basis(
        source_flow_direction, source_up_direction, target_flow_axis
    )
    return (
        np.outer(target_flow, source_flow)
        + np.outer(target_side, source_side)
        + np.outer(target_up, source_up)
    )


def _transform_affine(
    *,
    scale: float,
    source_flow_direction: str,
    source_up_direction: str,
    target_flow_axis: str,
    rotation_degrees: Vector,
    translation: Vector,
    rotation_center: Vector,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (matrix, offset) pair for the shared transform contract."""
    factor = float(scale)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Geometry scale must be a finite positive value.")
    center = _finite_vector(rotation_center, "Rotation center")
    offset = _finite_vector(translation, "Translation")
    rotation = _finite_vector(rotation_degrees, "Rotation")
    rotation_radians: Vector = (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )
    euler = _euler_matrix(rotation_radians)
    basis = _basis_matrix(source_flow_direction, source_up_direction, target_flow_axis)
    matrix = factor * (basis @ euler)
    center_array = np.asarray(center, dtype=np.float64)
    shift = factor * (basis @ center_array) + np.asarray(offset, dtype=np.float64)
    return matrix, shift - matrix @ center_array


def _apply_affine(vertices: np.ndarray, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return vertices @ matrix.T + offset


def _transform_array(
    array: np.ndarray,
    scale: float = 1.0,
    source_flow_direction: str = "+x",
    source_up_direction: str = "+z",
    target_flow_axis: str = "x",
    rotation_degrees: Vector = (0.0, 0.0, 0.0),
    translation: Vector = (0.0, 0.0, 0.0),
    rotation_center: Vector | None = None,
) -> np.ndarray:
    center = (
        _array_center(array)
        if rotation_center is None
        else _finite_vector(rotation_center, "Rotation center")
    )
    matrix, offset = _transform_affine(
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=center,
    )
    shape = array.shape
    transformed = _apply_affine(array.reshape(-1, 3), matrix, offset)
    return transformed.reshape(shape)


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
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {source_path}")
    array, _ = _read_stl_array(source_path)
    transformed = _transform_array(
        array,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    _write_binary_stl_array(target_path, transformed, header="AeroLab transformed body in solver meters")


def write_binary_stl_triangles(
    target_path: Path,
    triangles: Iterable[Triangle],
    header: str = "AeroLab binary STL",
) -> None:
    _write_binary_stl_array(target_path, _as_array(triangles), header=header)


def _write_binary_stl_array(
    target_path: Path,
    array: np.ndarray,
    header: str = "AeroLab binary STL",
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = header.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    records = np.zeros(len(array), dtype=_BINARY_STL_DTYPE)
    normals = _triangle_normals(array) if len(array) else np.zeros((0, 3))
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = array.astype(np.float32)
    records["attribute"] = 0
    with target_path.open("wb") as stream:
        stream.write(header_bytes)
        stream.write(struct.pack("<I", len(array)))
        records.tofile(stream)


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
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".stl":
        raise ValueError(f"Only STL files are supported right now: {source_path}")
    array, _ = _read_stl_array(source_path)
    transformed = _transform_array(
        array,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normals = _triangle_normals(transformed)
    lines = ["solid body_meters"]
    append = lines.append
    for triangle, normal in zip(transformed, normals, strict=False):
        append(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
        append("    outer loop")
        for vertex in triangle:
            append(f"      vertex {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
        append("    endloop")
        append("  endfacet")
    append("endsolid body_meters")
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    array = _as_array(triangles)
    transformed = _transform_array(
        array,
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=rotation_center,
    )
    return _as_tuples(transformed)


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
    matrix, offset = _transform_affine(
        scale=scale,
        source_flow_direction=source_flow_direction,
        source_up_direction=source_up_direction,
        target_flow_axis=target_flow_axis,
        rotation_degrees=rotation_degrees,
        translation=translation,
        rotation_center=center,
    )
    result = _apply_affine(np.asarray([source_point], dtype=np.float64), matrix, offset)[0]
    return (float(result[0]), float(result[1]), float(result[2]))


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
    rotated = _euler_matrix(rotation_radians) @ np.asarray(source_direction, dtype=np.float64)
    basis = _basis_matrix(source_flow_direction, source_up_direction, target_flow_axis)
    result = basis @ rotated
    return _normalize((float(result[0]), float(result[1]), float(result[2])))


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


def _array_center(array: np.ndarray) -> Vector:
    bounds = _bounds_array(array)
    return (
        (bounds.minimum[0] + bounds.maximum[0]) / 2.0,
        (bounds.minimum[1] + bounds.maximum[1]) / 2.0,
        (bounds.minimum[2] + bounds.maximum[2]) / 2.0,
    )


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
    array, stl_format = _read_stl_array(path)
    if source_flow_direction or source_up_direction:
        if not source_flow_direction or not source_up_direction:
            raise ValueError("Both source flow and source up are required for an oriented preview.")
        array = _transform_array(
            array,
            source_flow_direction=source_flow_direction,
            source_up_direction=source_up_direction,
            target_flow_axis=target_flow_axis,
        )
    bounds = _bounds_array(array)
    center = np.asarray(
        (
            (bounds.minimum[0] + bounds.maximum[0]) / 2.0,
            (bounds.minimum[1] + bounds.maximum[1]) / 2.0,
            (bounds.minimum[2] + bounds.maximum[2]) / 2.0,
        ),
        dtype=np.float64,
    )
    max_dim = max(bounds.dimensions) or 1.0
    preview_scale = 3.8 / max_dim
    triangle_total = len(array)
    if triangle_total <= max_triangles:
        sample_step = 1.0
        sampled = array
    else:
        sample_step = triangle_total / max_triangles
        indices = np.minimum(
            (np.arange(max_triangles) * sample_step).astype(np.int64),
            triangle_total - 1,
        )
        sampled = array[indices]

    normals = _triangle_normals(sampled)
    normalized = []
    scaled = (sampled - center) * preview_scale
    for triangle, normal in zip(scaled, normals, strict=False):
        normalized.append(
            {
                "v": [round(float(value), 6) for value in triangle.reshape(-1)],
                "n": [round(float(value), 6) for value in normal],
            }
        )

    return {
        "format": stl_format,
        "triangleCount": triangle_total,
        "sampledTriangleCount": len(normalized),
        "sampleStep": sample_step,
        "previewTriangleLimit": max_triangles,
        "isComplete": len(normalized) == triangle_total,
        "bounds": bounds.to_dict(),
        "normalizedCenter": [round(float(value), 9) for value in center],
        "normalizedScale": preview_scale,
        "triangles": normalized,
    }


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


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
