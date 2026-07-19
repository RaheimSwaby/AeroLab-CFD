from __future__ import annotations

import json
import heapq
import math
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .stl import (
    StlReport,
    Triangle,
    Vector,
    inspect_stl,
    read_stl_triangles,
    report_for_triangles,
    write_binary_stl_triangles,
)


DEFAULT_REPAIR_RESOLUTION = 384
MAX_REPAIR_DIMENSION_CHANGE = 0.02
MAX_REPAIR_PROJECTED_AREA_CHANGE = 0.02
MAX_REPAIR_SURFACE_DEVIATION_P95 = 0.003
MAX_REPAIR_SURFACE_DEVIATION_P99 = 0.006
MAX_REPAIR_DETAIL_RESOLUTION = 0.003
MAX_REPAIR_ADDED_SURFACE_DEVIATION_P95 = 0.004
MAX_REPAIR_ADDED_SURFACE_FAR_FRACTION = 0.025
REPAIR_ADDED_SURFACE_FAR_DISTANCE = 0.01


@dataclass(frozen=True)
class RepairResult:
    source_path: Path
    output_path: Path
    method: str
    accepted: bool
    grid_resolution: int
    voxel_size: float
    detail_resolution_percent: float
    max_detail_resolution_percent: float
    dimension_change_percent: float
    projected_area_change_percent: float
    surface_area_change_percent: float
    surface_deviation_p95: float
    surface_deviation_p95_percent: float
    surface_deviation_p99: float
    surface_deviation_p99_percent: float
    source_surface_deviation_p95_percent: float
    source_surface_deviation_p99_percent: float
    added_surface_deviation_p95_percent: float
    added_surface_far_fraction_percent: float
    added_surface_far_distance_percent: float
    surface_deviation_sample_count: int
    max_surface_deviation_p95_percent: float
    max_surface_deviation_p99_percent: float
    rejection_reasons: tuple[str, ...]
    source_report: StlReport
    output_report: StlReport
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        if self.method == "degenerate_face_retriangulation":
            notes = [
                "The original STL was not modified.",
                "Zero-area stitch faces were replaced by triangles split at existing collinear vertices.",
                "No source vertex positions or visible surface coordinates were changed.",
                "The resulting STL was rechecked for watertight, manifold CFD topology.",
            ]
        elif self.method == "small_hole_fill":
            notes = [
                "The original STL was not modified.",
                "Small open boundary loops were patched with local fan triangles.",
                "Every source triangle is preserved exactly; only patch triangles were added.",
                "The resulting STL was rechecked for watertight, manifold CFD topology.",
            ]
        elif self.method == "local_non_manifold_overlay_repair":
            notes = [
                "The original STL was not modified.",
                "Tiny redundant overlay components were removed from the dominant source surface.",
                "Only local loops around the removed non-manifold faces were capped; source body lines were not smoothed.",
                "The resulting STL was rechecked for watertight, manifold CFD topology and two-way surface fidelity.",
            ]
        else:
            notes = [
                "The original STL was not modified.",
                "The prepared STL seals small scan gaps and overlapping surface defects with a voxel shell.",
                "Local shape fidelity is checked in both directions with sampled point-to-triangle distances.",
                "Inspect the prepared shape before using it for final CFD because small openings are intentionally sealed.",
            ]
        return {
            "sourcePath": str(self.source_path),
            "outputPath": str(self.output_path),
            "accepted": self.accepted,
            "method": self.method,
            "gridResolution": self.grid_resolution,
            "voxelSize": self.voxel_size,
            "detailResolutionPercent": self.detail_resolution_percent,
            "maxDetailResolutionPercent": self.max_detail_resolution_percent,
            "dimensionChangePercent": self.dimension_change_percent,
            "projectedAreaChangePercent": self.projected_area_change_percent,
            "surfaceAreaChangePercent": self.surface_area_change_percent,
            "surfaceDeviationP95": self.surface_deviation_p95,
            "surfaceDeviationP95Percent": self.surface_deviation_p95_percent,
            "surfaceDeviationP99": self.surface_deviation_p99,
            "surfaceDeviationP99Percent": self.surface_deviation_p99_percent,
            "sourceSurfaceDeviationP95Percent": self.source_surface_deviation_p95_percent,
            "sourceSurfaceDeviationP99Percent": self.source_surface_deviation_p99_percent,
            "addedSurfaceDeviationP95Percent": self.added_surface_deviation_p95_percent,
            "addedSurfaceFarFractionPercent": self.added_surface_far_fraction_percent,
            "addedSurfaceFarDistancePercent": self.added_surface_far_distance_percent,
            "surfaceDeviationSampleCount": self.surface_deviation_sample_count,
            "maxSurfaceDeviationP95Percent": self.max_surface_deviation_p95_percent,
            "maxSurfaceDeviationP99Percent": self.max_surface_deviation_p99_percent,
            "rejectionReasons": list(self.rejection_reasons),
            "warnings": list(self.warnings),
            "sourceReport": self.source_report.to_dict(),
            "outputReport": self.output_report.to_dict(),
            "notes": notes,
        }


def repair_stl(
    source_path: Path,
    output_path: Path,
    resolution: int = DEFAULT_REPAIR_RESOLUTION,
    max_dimension_change: float = MAX_REPAIR_DIMENSION_CHANGE,
    max_projected_area_change: float = MAX_REPAIR_PROJECTED_AREA_CHANGE,
    max_surface_deviation_p95: float = MAX_REPAIR_SURFACE_DEVIATION_P95,
    max_surface_deviation_p99: float = MAX_REPAIR_SURFACE_DEVIATION_P99,
    max_detail_resolution: float = MAX_REPAIR_DETAIL_RESOLUTION,
    max_added_surface_deviation_p95: float = MAX_REPAIR_ADDED_SURFACE_DEVIATION_P95,
    max_added_surface_far_fraction: float = MAX_REPAIR_ADDED_SURFACE_FAR_FRACTION,
    surface_deviation_sample_count: int = 2000,
    smallest_feature_m: float | None = None,
    smallest_feature_source_units: float | None = None,
) -> RepairResult:
    if resolution < 96 or resolution > 512:
        raise ValueError("Repair resolution must be between 96 and 512.")

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    source_report = inspect_stl(source_path)
    triangles, _ = read_stl_triangles(source_path)
    bounds = source_report.bounds
    dimensions = np.asarray(bounds.dimensions, dtype=np.float64)
    longest = float(np.max(dimensions))
    if longest <= 0:
        raise ValueError("The model must have non-zero dimensions before it can be prepared.")

    if source_report.is_watertight and source_report.degenerate_triangle_count:
        retriangulated = _retriangulate_degenerate_faces(triangles)
        if retriangulated is not None:
            write_binary_stl_triangles(
                output_path,
                retriangulated,
                header="AeroLab exact degenerate-face repair",
            )
            surgical_result = assess_repaired_stl(
                source_path,
                output_path,
                resolution=0,
                max_dimension_change=max_dimension_change,
                max_projected_area_change=max_projected_area_change,
                max_surface_deviation_p95=max_surface_deviation_p95,
                max_surface_deviation_p99=max_surface_deviation_p99,
                max_detail_resolution=max_detail_resolution,
                max_added_surface_deviation_p95=max_added_surface_deviation_p95,
                max_added_surface_far_fraction=max_added_surface_far_fraction,
                surface_deviation_sample_count=surface_deviation_sample_count,
                method="degenerate_face_retriangulation",
                detail_resolution_override=0.0,
            )
            if surgical_result.accepted:
                return surgical_result

    if source_report.non_manifold_edge_count == 0 and source_report.open_edge_count:
        patched = _fill_small_boundary_loops(triangles, longest)
        if patched is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_binary_stl_triangles(
                output_path,
                patched,
                header="AeroLab small-hole fill repair",
            )
            hole_result = assess_repaired_stl(
                source_path,
                output_path,
                resolution=0,
                max_dimension_change=max_dimension_change,
                max_projected_area_change=max_projected_area_change,
                max_surface_deviation_p95=max_surface_deviation_p95,
                max_surface_deviation_p99=max_surface_deviation_p99,
                max_detail_resolution=max_detail_resolution,
                max_added_surface_deviation_p95=max_added_surface_deviation_p95,
                max_added_surface_far_fraction=max_added_surface_far_fraction,
                surface_deviation_sample_count=surface_deviation_sample_count,
                method="small_hole_fill",
                detail_resolution_override=0.0,
            )
            if hole_result.accepted:
                return hole_result

    if source_report.non_manifold_edge_count:
        surgical = _repair_local_non_manifold_overlays(triangles, longest)
        if surgical is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_binary_stl_triangles(
                output_path,
                surgical,
                header="AeroLab local non-manifold overlay repair",
            )
            surgical_result = assess_repaired_stl(
                source_path,
                output_path,
                resolution=0,
                max_dimension_change=max_dimension_change,
                max_projected_area_change=max_projected_area_change,
                max_surface_deviation_p95=max_surface_deviation_p95,
                max_surface_deviation_p99=max_surface_deviation_p99,
                max_detail_resolution=max_detail_resolution,
                max_added_surface_deviation_p95=max_added_surface_deviation_p95,
                max_added_surface_far_fraction=max_added_surface_far_fraction,
                surface_deviation_sample_count=surface_deviation_sample_count,
                method="local_non_manifold_overlay_repair",
                detail_resolution_override=0.0,
            )
            if surgical_result.accepted:
                return surgical_result

    repair_warnings: list[str] = []
    feature_target = smallest_feature_source_units or smallest_feature_m
    if feature_target and feature_target > 0:
        # Keep a repair voxel at most one third of the smallest feature while
        # respecting a local-memory cell budget.
        needed_resolution = int(math.ceil(3.0 * longest / feature_target))
        resolution = max(96, min(512, needed_resolution))
        while resolution > 96:
            candidate_voxel = longest / float(resolution)
            cell_count = math.prod(
                max(8, int(math.ceil(value / candidate_voxel)) + 6) for value in dimensions
            )
            if cell_count <= 24_000_000:
                break
            resolution -= 32
        if resolution < needed_resolution:
            achievable_size = longest / float(resolution)
            repair_warnings.append(
                f"The requested smallest feature needs a finer repair grid than this model allows "
                f"(best achievable voxel is {achievable_size:.6g} source units). Louvres, slots, "
                "thin blades, or ducts near that size may be sealed or erased. Scan or export the "
                "part on its own for a finer repair."
            )

    voxel_size = longest / float(resolution)
    detail_resolution = voxel_size / longest
    margin = 3
    shape = tuple(max(8, int(math.ceil(value / voxel_size)) + margin * 2) for value in dimensions)
    origin = np.asarray(bounds.minimum, dtype=np.float64) - voxel_size * margin
    surface = np.zeros(shape, dtype=np.bool_)

    for triangle in triangles:
        _rasterize_triangle(surface, triangle, origin, voxel_size)

    surface = _dilate(surface)
    outside = _flood_outside(surface)
    solid = ~outside
    solid = _regularize_solid(solid)
    solid = _erode(solid)
    solid = _remove_edge_contacts(solid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_voxel_surface(output_path, solid, origin, voxel_size)
    return assess_repaired_stl(
        source_path,
        output_path,
        resolution=resolution,
        max_dimension_change=max_dimension_change,
        max_projected_area_change=max_projected_area_change,
        max_surface_deviation_p95=max_surface_deviation_p95,
        max_surface_deviation_p99=max_surface_deviation_p99,
        max_detail_resolution=max_detail_resolution,
        max_added_surface_deviation_p95=max_added_surface_deviation_p95,
        max_added_surface_far_fraction=max_added_surface_far_fraction,
        surface_deviation_sample_count=surface_deviation_sample_count,
        warnings=tuple(repair_warnings),
    )


def assess_repaired_stl(
    source_path: Path,
    output_path: Path,
    resolution: int,
    max_dimension_change: float = MAX_REPAIR_DIMENSION_CHANGE,
    max_projected_area_change: float = MAX_REPAIR_PROJECTED_AREA_CHANGE,
    max_surface_deviation_p95: float = MAX_REPAIR_SURFACE_DEVIATION_P95,
    max_surface_deviation_p99: float = MAX_REPAIR_SURFACE_DEVIATION_P99,
    max_detail_resolution: float = MAX_REPAIR_DETAIL_RESOLUTION,
    max_added_surface_deviation_p95: float = MAX_REPAIR_ADDED_SURFACE_DEVIATION_P95,
    max_added_surface_far_fraction: float = MAX_REPAIR_ADDED_SURFACE_FAR_FRACTION,
    surface_deviation_sample_count: int = 2000,
    method: str = "sealed_voxel_shell",
    detail_resolution_override: float | None = None,
    warnings: tuple[str, ...] = (),
) -> RepairResult:
    if method == "sealed_voxel_shell" and (resolution < 96 or resolution > 512):
        raise ValueError("Repair resolution must be between 96 and 512.")
    if method != "sealed_voxel_shell" and resolution < 0:
        raise ValueError("Repair resolution cannot be negative.")
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    source_report = inspect_stl(source_path)
    triangles, _ = read_stl_triangles(source_path)
    longest = max(source_report.bounds.dimensions)
    if longest <= 0:
        raise ValueError("The model must have non-zero dimensions before it can be assessed.")
    voxel_size = longest / float(resolution) if resolution else 0.0
    detail_resolution = (
        detail_resolution_override
        if detail_resolution_override is not None
        else voxel_size / longest
    )
    output_report = inspect_stl(output_path)
    output_triangles, _ = read_stl_triangles(output_path)
    dimension_change = _dimension_change(source_report, output_report)
    projected_area_change = _projected_area_change(source_report, output_report)
    surface_area_change = _relative_change(source_report.surface_area, output_report.surface_area)
    exact_retriangulation = (
        method == "degenerate_face_retriangulation"
        and _exact_retriangulation_invariants_hold(
            source_report,
            output_report,
            triangles,
            output_triangles,
        )
    )
    if exact_retriangulation:
        deviation = {
            "symmetricP95": 0.0,
            "symmetricP99": 0.0,
            "sourceP95": 0.0,
            "sourceP99": 0.0,
            "outputP95": 0.0,
            "outputP99": 0.0,
            "outputFarFraction": 0.0,
        }
        measured_sample_count = 0
    else:
        deviation = _surface_deviation_metrics(
            triangles,
            output_triangles,
            sample_count=surface_deviation_sample_count,
            far_distance=longest * REPAIR_ADDED_SURFACE_FAR_DISTANCE,
        )
        measured_sample_count = surface_deviation_sample_count * 2
    deviation_p95 = deviation["symmetricP95"]
    deviation_p99 = deviation["symmetricP99"]
    deviation_p95_relative = deviation_p95 / longest
    deviation_p99_relative = deviation_p99 / longest
    source_deviation_p95_relative = deviation["sourceP95"] / longest
    source_deviation_p99_relative = deviation["sourceP99"] / longest
    added_deviation_p95_relative = deviation["outputP95"] / longest
    added_far_fraction = deviation["outputFarFraction"]
    rejection_reasons = _repair_rejection_reasons(
        output_report,
        dimension_change,
        projected_area_change,
        source_deviation_p95_relative,
        source_deviation_p99_relative,
        added_deviation_p95_relative,
        added_far_fraction,
        detail_resolution,
        max_dimension_change,
        max_projected_area_change,
        max_surface_deviation_p95,
        max_surface_deviation_p99,
        max_detail_resolution,
        max_added_surface_deviation_p95,
        max_added_surface_far_fraction,
    )
    if method == "degenerate_face_retriangulation" and not exact_retriangulation:
        rejection_reasons = (
            "Exact retriangulation invariants failed; source geometry preservation could not be verified.",
            *rejection_reasons,
        )
    accepted = not rejection_reasons
    warning_list = list(warnings)
    if method == "sealed_voxel_shell" and abs(surface_area_change) > 0.04:
        warning_list.append(
            f"Surface area changed by {surface_area_change * 100.0:+.1f}% during the reseal; "
            "narrow passages such as louvre slots, duct inlets, or wing-element gaps may have "
            "been sealed. Inspect the prepared model around those features before running CFD."
        )

    result = RepairResult(
        source_path=source_path,
        output_path=output_path,
        method=method,
        accepted=accepted,
        grid_resolution=resolution,
        voxel_size=voxel_size,
        detail_resolution_percent=detail_resolution * 100.0,
        max_detail_resolution_percent=max_detail_resolution * 100.0,
        dimension_change_percent=dimension_change * 100.0,
        projected_area_change_percent=projected_area_change * 100.0,
        surface_area_change_percent=surface_area_change * 100.0,
        surface_deviation_p95=deviation_p95,
        surface_deviation_p95_percent=deviation_p95_relative * 100.0,
        surface_deviation_p99=deviation_p99,
        surface_deviation_p99_percent=deviation_p99_relative * 100.0,
        source_surface_deviation_p95_percent=source_deviation_p95_relative * 100.0,
        source_surface_deviation_p99_percent=source_deviation_p99_relative * 100.0,
        added_surface_deviation_p95_percent=added_deviation_p95_relative * 100.0,
        added_surface_far_fraction_percent=added_far_fraction * 100.0,
        added_surface_far_distance_percent=REPAIR_ADDED_SURFACE_FAR_DISTANCE * 100.0,
        surface_deviation_sample_count=measured_sample_count,
        max_surface_deviation_p95_percent=max_surface_deviation_p95 * 100.0,
        max_surface_deviation_p99_percent=max_surface_deviation_p99 * 100.0,
        rejection_reasons=rejection_reasons,
        source_report=source_report,
        output_report=output_report,
        warnings=tuple(warning_list),
    )
    metadata_path = repair_metadata_path(output_path)
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def _exact_retriangulation_invariants_hold(
    source_report: StlReport,
    output_report: StlReport,
    source_triangles: list[Triangle],
    output_triangles: list[Triangle],
) -> bool:
    if not output_report.is_cfd_candidate:
        return False
    source_vertices = {
        _repair_vertex_key(vertex)
        for triangle in source_triangles
        for vertex in triangle
    }
    output_vertices = {
        _repair_vertex_key(vertex)
        for triangle in output_triangles
        for vertex in triangle
    }
    if output_vertices != source_vertices:
        return False

    source_scalars = (
        *source_report.bounds.minimum,
        *source_report.bounds.maximum,
        source_report.surface_area,
        source_report.projected_areas.x,
        source_report.projected_areas.y,
        source_report.projected_areas.z,
        source_report.volume,
    )
    output_scalars = (
        *output_report.bounds.minimum,
        *output_report.bounds.maximum,
        output_report.surface_area,
        output_report.projected_areas.x,
        output_report.projected_areas.y,
        output_report.projected_areas.z,
        output_report.volume,
    )
    return all(
        math.isclose(source, output, rel_tol=1e-8, abs_tol=1e-10)
        for source, output in zip(source_scalars, output_scalars)
    )


def _retriangulate_degenerate_faces(triangles: list[Triangle]) -> list[Triangle] | None:
    degenerate = {
        index
        for index, triangle in enumerate(triangles)
        if _triangle_area(triangle) <= 1e-12
    }
    if not degenerate:
        return list(triangles)

    edge_owners: dict[tuple[Vector, Vector], list[int]] = defaultdict(list)
    vertex_to_degenerate: dict[Vector, set[int]] = defaultdict(set)
    for index, triangle in enumerate(triangles):
        vertices = tuple(_repair_vertex_key(vertex) for vertex in triangle)
        for start, end in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
            edge_owners[_repair_edge_key(start, end)].append(index)
        if index in degenerate:
            for vertex in vertices:
                vertex_to_degenerate[vertex].add(index)

    components: list[set[int]] = []
    remaining = set(degenerate)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for vertex in triangles[current]:
                for neighbor in vertex_to_degenerate[_repair_vertex_key(vertex)]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
        components.append(component)

    replacements: dict[int, list[Triangle]] = {}
    for component in components:
        component_vertices = {
            _repair_vertex_key(vertex): vertex
            for index in component
            for vertex in triangles[index]
        }
        boundary_edges: set[tuple[tuple[Vector, Vector], int]] = set()
        for index in component:
            triangle = triangles[index]
            vertices = tuple(_repair_vertex_key(vertex) for vertex in triangle)
            for start, end in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
                key = _repair_edge_key(start, end)
                nondegenerate_owners = [owner for owner in edge_owners[key] if owner not in degenerate]
                if len(nondegenerate_owners) == 1:
                    boundary_edges.add((key, nondegenerate_owners[0]))
                elif len(nondegenerate_owners) > 1:
                    return None

        for key, owner in boundary_edges:
            triangle = triangles[owner]
            edge_index = next(
                (
                    index
                    for index in range(3)
                    if _repair_edge_key(
                        _repair_vertex_key(triangle[index]),
                        _repair_vertex_key(triangle[(index + 1) % 3]),
                    )
                    == key
                ),
                None,
            )
            if edge_index is None:
                return None
            start = triangle[edge_index]
            end = triangle[(edge_index + 1) % 3]
            interior = [
                vertex
                for vertex in component_vertices.values()
                if _point_strictly_on_segment(vertex, start, end)
            ]
            if not interior:
                continue
            if owner in replacements:
                return None
            direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
            denominator = float(np.dot(direction, direction))
            interior.sort(
                key=lambda point: float(
                    np.dot(np.asarray(point, dtype=np.float64) - np.asarray(start, dtype=np.float64), direction)
                    / denominator
                )
            )
            points = [start, *interior, end]
            opposite = triangle[(edge_index + 2) % 3]
            replacements[owner] = [
                (points[index], points[index + 1], opposite)
                for index in range(len(points) - 1)
            ]

    if not replacements:
        return None
    output: list[Triangle] = []
    for index, triangle in enumerate(triangles):
        if index in degenerate:
            continue
        output.extend(replacements.get(index, [triangle]))
    report = report_for_triangles(Path("retriangulated.stl"), output, "binary")
    return output if report.is_cfd_candidate else None


def _repair_local_non_manifold_overlays(
    triangles: list[Triangle],
    longest: float,
    minimum_dominant_fraction: float = 0.995,
    maximum_discarded_faces: int = 256,
    maximum_problem_faces: int = 12,
) -> list[Triangle] | None:
    """Remove tiny duplicate overlays, then cap only their local boundary loops."""
    edge_owners = _repair_edge_owners(triangles)
    adjacency: list[list[int]] = [[] for _ in triangles]
    for owners in edge_owners.values():
        if len(owners) == 2:
            first, second = owners
            adjacency[first].append(second)
            adjacency[second].append(first)

    components: list[list[int]] = []
    visited: set[int] = set()
    for start in range(len(triangles)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            face = stack.pop()
            component.append(face)
            for neighbour in adjacency[face]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(component)

    dominant = max(components, key=len)
    discarded = len(triangles) - len(dominant)
    if (
        len(dominant) / max(len(triangles), 1) < minimum_dominant_fraction
        or discarded > maximum_discarded_faces
    ):
        return None
    candidate = [triangles[index] for index in dominant]
    report = report_for_triangles(Path("dominant-component.stl"), candidate, "binary")
    if report.non_manifold_edge_count == 0:
        patched = _fill_small_boundary_loops(candidate, longest)
        if patched is None:
            return candidate if report.is_cfd_candidate else None
        return (
            patched
            if report_for_triangles(Path("patched-component.stl"), patched, "binary").is_cfd_candidate
            else None
        )

    edge_owners = _repair_edge_owners(candidate)
    problem_faces = sorted(
        {face for owners in edge_owners.values() if len(owners) > 2 for face in owners}
    )
    if not problem_faces or len(problem_faces) > maximum_problem_faces:
        return None

    for removal_count in range(1, len(problem_faces) + 1):
        for removal in combinations(problem_faces, removal_count):
            removed = set(removal)
            if any(
                len(owners) - sum(face in removed for face in owners) > 2
                for owners in edge_owners.values()
            ):
                continue
            pruned = [triangle for index, triangle in enumerate(candidate) if index not in removed]
            patched = _fill_small_boundary_loops(pruned, longest)
            if patched is None:
                continue
            patched_report = report_for_triangles(Path("surgical-repair.stl"), patched, "binary")
            if patched_report.is_cfd_candidate:
                return patched
    return None


def _repair_edge_owners(triangles: list[Triangle]) -> dict[tuple[Vector, Vector], list[int]]:
    owners: dict[tuple[Vector, Vector], list[int]] = defaultdict(list)
    for face, triangle in enumerate(triangles):
        keyed = tuple(_repair_vertex_key(vertex) for vertex in triangle)
        for index in range(3):
            owners[_repair_edge_key(keyed[index], keyed[(index + 1) % 3])].append(face)
    return owners


def _fill_small_boundary_loops(
    triangles: list[Triangle],
    longest: float,
    max_loop_edges: int = 64,
    max_loop_diameter_fraction: float = 0.03,
) -> list[Triangle] | None:
    """Patch small open loops while preserving every source triangle exactly."""
    edge_counts: dict[tuple[Vector, Vector], int] = defaultdict(int)
    keyed_triangles: list[tuple[Vector, Vector, Vector]] = []
    for triangle in triangles:
        keyed = tuple(_repair_vertex_key(vertex) for vertex in triangle)
        keyed_triangles.append(keyed)  # type: ignore[arg-type]
        for index in range(3):
            edge_counts[_repair_edge_key(keyed[index], keyed[(index + 1) % 3])] += 1

    boundary_next: dict[Vector, Vector] = {}
    for keyed in keyed_triangles:
        for index in range(3):
            start, end = keyed[index], keyed[(index + 1) % 3]
            if edge_counts[_repair_edge_key(start, end)] == 1:
                if start in boundary_next:
                    return None  # Branching boundaries are not simple pinholes.
                boundary_next[start] = end
    if not boundary_next:
        return None

    patches: list[Triangle] = []
    remaining = dict(boundary_next)
    while remaining:
        first = next(iter(remaining))
        loop = [first]
        current = remaining.pop(first)
        while current != first:
            loop.append(current)
            if current not in remaining or len(loop) > max_loop_edges:
                return None
            current = remaining.pop(current)
        if len(loop) < 3:
            return None
        points = np.asarray(loop, dtype=np.float64)
        diameter = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if diameter > longest * max_loop_diameter_fraction:
            return None
        centroid = tuple(float(value) for value in points.mean(axis=0))
        for index in range(len(loop)):
            start = loop[index]
            end = loop[(index + 1) % len(loop)]
            # Reverse the boundary edge so the fan keeps outward winding.
            patches.append((centroid, end, start))  # type: ignore[arg-type]
    return [*triangles, *patches]


def _triangle_area(triangle: Triangle) -> float:
    points = np.asarray(triangle, dtype=np.float64)
    return float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) * 0.5)


def _repair_vertex_key(vertex: Vector) -> Vector:
    return (round(vertex[0], 9), round(vertex[1], 9), round(vertex[2], 9))


def _repair_edge_key(start: Vector, end: Vector) -> tuple[Vector, Vector]:
    return (start, end) if start <= end else (end, start)


def _point_strictly_on_segment(point: Vector, start: Vector, end: Vector) -> bool:
    point_array = np.asarray(point, dtype=np.float64)
    start_array = np.asarray(start, dtype=np.float64)
    direction = np.asarray(end, dtype=np.float64) - start_array
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-24:
        return False
    parameter = float(np.dot(point_array - start_array, direction) / denominator)
    if parameter <= 1e-10 or parameter >= 1.0 - 1e-10:
        return False
    projected = start_array + direction * parameter
    tolerance = max(math.sqrt(denominator) * 1e-9, 1e-12)
    return float(np.linalg.norm(point_array - projected)) <= tolerance


def repair_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".repair.json")


def is_prepared_model_path(model_path: Path) -> bool:
    return model_path.parent.name.lower() == "prepared" or model_path.stem.lower().endswith("-prepared")


def repair_fidelity_for_model(model_path: Path) -> dict[str, object] | None:
    model_path = model_path.resolve()
    if not is_prepared_model_path(model_path):
        return None
    metadata_path = repair_metadata_path(model_path)
    if not metadata_path.exists():
        return {
            "status": "missing",
            "verified": False,
            "metadataPath": str(metadata_path),
            "detail": "Prepared STL has no repair-fidelity record; prepare it again before CFD.",
        }
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "verified": False,
            "metadataPath": str(metadata_path),
            "detail": f"Repair-fidelity record is invalid: {exc}",
        }
    try:
        recorded_output = Path(str(payload.get("outputPath") or "")).resolve()
    except (OSError, RuntimeError):
        recorded_output = Path()
    accepted = bool(payload.get("accepted"))
    output_matches = recorded_output == model_path
    if not output_matches:
        # Keep records valid when a project moves between Windows and WSL.
        recorded_name = (
            str(payload.get("outputPath") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        )
        output_matches = bool(recorded_name) and recorded_name.lower() == model_path.name.lower()
    limits_verified = all(
        (
            _metadata_percent_at_most(payload, "dimensionChangePercent", MAX_REPAIR_DIMENSION_CHANGE),
            _metadata_percent_at_most(payload, "projectedAreaChangePercent", MAX_REPAIR_PROJECTED_AREA_CHANGE),
            _metadata_percent_at_most(payload, "sourceSurfaceDeviationP95Percent", MAX_REPAIR_SURFACE_DEVIATION_P95),
            _metadata_percent_at_most(payload, "sourceSurfaceDeviationP99Percent", MAX_REPAIR_SURFACE_DEVIATION_P99),
            _metadata_percent_at_most(payload, "addedSurfaceDeviationP95Percent", MAX_REPAIR_ADDED_SURFACE_DEVIATION_P95),
            _metadata_percent_at_most(payload, "addedSurfaceFarFractionPercent", MAX_REPAIR_ADDED_SURFACE_FAR_FRACTION),
            _metadata_percent_at_most(payload, "detailResolutionPercent", MAX_REPAIR_DETAIL_RESOLUTION),
        )
    )
    verified = accepted and output_matches and limits_verified
    return {
        "status": "verified" if verified else "rejected",
        "verified": verified,
        "accepted": accepted,
        "outputMatches": output_matches,
        "limitsVerified": limits_verified,
        "metadataPath": str(metadata_path),
        "sourcePath": payload.get("sourcePath"),
        "method": payload.get("method"),
        "gridResolution": payload.get("gridResolution"),
        "voxelSize": payload.get("voxelSize"),
        "detailResolutionPercent": payload.get("detailResolutionPercent"),
        "surfaceDeviationP95Percent": payload.get("surfaceDeviationP95Percent"),
        "surfaceDeviationP99Percent": payload.get("surfaceDeviationP99Percent"),
        "sourceSurfaceDeviationP95Percent": payload.get("sourceSurfaceDeviationP95Percent"),
        "sourceSurfaceDeviationP99Percent": payload.get("sourceSurfaceDeviationP99Percent"),
        "addedSurfaceDeviationP95Percent": payload.get("addedSurfaceDeviationP95Percent"),
        "addedSurfaceFarFractionPercent": payload.get("addedSurfaceFarFractionPercent"),
        "projectedAreaChangePercent": payload.get("projectedAreaChangePercent"),
        "rejectionReasons": payload.get("rejectionReasons") or [],
        "detail": (
            "Prepared STL passed the recorded body-line fidelity limits."
            if verified
            else "Prepared STL did not pass or does not match its repair-fidelity record."
        ),
    }


def _metadata_percent_at_most(payload: dict[str, object], key: str, relative_limit: float) -> bool:
    try:
        return float(payload[key]) <= relative_limit * 100.0 + 1e-12
    except (KeyError, TypeError, ValueError):
        return False


def _symmetric_surface_deviation(
    source_triangles: list[Triangle],
    output_triangles: list[Triangle],
    sample_count: int = 2000,
) -> tuple[float, float]:
    metrics = _surface_deviation_metrics(source_triangles, output_triangles, sample_count=sample_count)
    return metrics["symmetricP95"], metrics["symmetricP99"]


def surface_deviation_metrics(
    source_triangles: list[Triangle],
    output_triangles: list[Triangle],
    sample_count: int = 2000,
    far_distance: float | None = None,
) -> dict[str, float]:
    return _surface_deviation_metrics(
        source_triangles,
        output_triangles,
        sample_count=sample_count,
        far_distance=far_distance,
    )


def _surface_deviation_metrics(
    source_triangles: list[Triangle],
    output_triangles: list[Triangle],
    sample_count: int = 2000,
    far_distance: float | None = None,
) -> dict[str, float]:
    if sample_count < 100:
        raise ValueError("Surface deviation requires at least 100 samples per mesh.")

    source = np.asarray(source_triangles, dtype=np.float64)
    output = np.asarray(output_triangles, dtype=np.float64)
    source_samples = _sample_triangle_surface(source, sample_count, np.random.default_rng(9341))
    output_samples = _sample_triangle_surface(output, sample_count, np.random.default_rng(2718))
    source_distances = _nearest_triangle_distances(source_samples, output)
    output_distances = _nearest_triangle_distances(output_samples, source)
    distances = np.concatenate((source_distances, output_distances))
    return {
        "symmetricP95": float(np.percentile(distances, 95)),
        "symmetricP99": float(np.percentile(distances, 99)),
        "sourceP95": float(np.percentile(source_distances, 95)),
        "sourceP99": float(np.percentile(source_distances, 99)),
        "outputP95": float(np.percentile(output_distances, 95)),
        "outputP99": float(np.percentile(output_distances, 99)),
        "outputFarFraction": (
            float(np.mean(output_distances > far_distance))
            if far_distance is not None
            else 0.0
        ),
    }


def _sample_triangle_surface(
    triangles: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("Surface deviation requires non-empty triangle meshes.")
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    usable = areas > 1e-12
    if not np.any(usable):
        raise ValueError("Surface deviation requires triangles with non-zero area.")
    triangles = triangles[usable]
    areas = areas[usable]
    cumulative = np.cumsum(areas)
    indices = np.searchsorted(cumulative, rng.random(sample_count) * cumulative[-1], side="right")
    selected = triangles[indices]
    root_u = np.sqrt(rng.random(sample_count))
    v = rng.random(sample_count)
    return (
        (1.0 - root_u)[:, None] * selected[:, 0]
        + (root_u * (1.0 - v))[:, None] * selected[:, 1]
        + (root_u * v)[:, None] * selected[:, 2]
    )


def _nearest_triangle_distances(
    points: np.ndarray,
    triangles: np.ndarray,
    candidate_count: int = 32,
    chunk_size: int = 24,
) -> np.ndarray:
    del candidate_count, chunk_size
    nodes, root = _build_triangle_bvh(triangles)
    result = np.empty(len(points), dtype=np.float64)
    for point_index, point in enumerate(points):
        best = math.inf
        queue: list[tuple[float, int]] = [(_point_box_distance_squared(point, nodes[root][0], nodes[root][1]), root)]
        while queue:
            lower_bound, node_index = heapq.heappop(queue)
            if lower_bound >= best * best:
                continue
            minimum, maximum, left, right, indices = nodes[node_index]
            if indices is not None:
                candidates = triangles[indices][None, :, :, :]
                distances = _point_to_triangle_distances(point[None, :], candidates)[0]
                if len(distances):
                    best = min(best, float(np.min(distances)))
                continue
            assert left is not None and right is not None
            for child in (left, right):
                child_minimum, child_maximum = nodes[child][0], nodes[child][1]
                child_bound = _point_box_distance_squared(point, child_minimum, child_maximum)
                if child_bound < best * best:
                    heapq.heappush(queue, (child_bound, child))
        result[point_index] = best
    return result


def _build_triangle_bvh(
    triangles: np.ndarray,
    leaf_size: int = 32,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int | None, int | None, np.ndarray | None]], int]:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("Triangle distance search requires non-empty triangle meshes.")
    triangle_minimum = np.min(triangles, axis=1)
    triangle_maximum = np.max(triangles, axis=1)
    centroids = (triangle_minimum + triangle_maximum) * 0.5
    nodes: list[tuple[np.ndarray, np.ndarray, int | None, int | None, np.ndarray | None] | None] = []

    def build(indices: np.ndarray) -> int:
        node_index = len(nodes)
        nodes.append(None)
        minimum = np.min(triangle_minimum[indices], axis=0)
        maximum = np.max(triangle_maximum[indices], axis=0)
        if len(indices) <= leaf_size:
            nodes[node_index] = (minimum, maximum, None, None, indices)
            return node_index

        centroid_span = np.ptp(centroids[indices], axis=0)
        axis = int(np.argmax(centroid_span))
        midpoint = len(indices) // 2
        order = np.argpartition(centroids[indices, axis], midpoint)
        partitioned = indices[order]
        left = build(partitioned[:midpoint])
        right = build(partitioned[midpoint:])
        nodes[node_index] = (minimum, maximum, left, right, None)
        return node_index

    root = build(np.arange(len(triangles), dtype=np.int64))
    return [node for node in nodes if node is not None], root


def _point_box_distance_squared(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    delta = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
    return float(np.dot(delta, delta))


def _point_to_triangle_distances(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    samples = points[:, None, :]
    a = triangles[:, :, 0]
    b = triangles[:, :, 1]
    c = triangles[:, :, 2]
    ab = b - a
    ac = c - a
    normals = np.cross(ab, ac)
    normal_squared = np.einsum("ijk,ijk->ij", normals, normals)
    safe_normal_squared = np.maximum(normal_squared, 1e-24)
    plane_numerator = np.einsum("ijk,ijk->ij", samples - a, normals)
    projection = samples - (plane_numerator / safe_normal_squared)[:, :, None] * normals

    projected_offset = projection - a
    d00 = np.einsum("ijk,ijk->ij", ab, ab)
    d01 = np.einsum("ijk,ijk->ij", ab, ac)
    d11 = np.einsum("ijk,ijk->ij", ac, ac)
    d20 = np.einsum("ijk,ijk->ij", projected_offset, ab)
    d21 = np.einsum("ijk,ijk->ij", projected_offset, ac)
    denominator = d00 * d11 - d01 * d01
    safe_denominator = np.where(np.abs(denominator) > 1e-24, denominator, 1.0)
    weight_b = (d11 * d20 - d01 * d21) / safe_denominator
    weight_c = (d00 * d21 - d01 * d20) / safe_denominator
    weight_a = 1.0 - weight_b - weight_c
    inside = (
        (normal_squared > 1e-24)
        & (weight_a >= -1e-9)
        & (weight_b >= -1e-9)
        & (weight_c >= -1e-9)
    )
    plane_distance = np.abs(plane_numerator) / np.sqrt(safe_normal_squared)
    edge_distance = np.minimum.reduce(
        (
            _point_to_segment_distances(samples, a, b),
            _point_to_segment_distances(samples, b, c),
            _point_to_segment_distances(samples, c, a),
        )
    )
    return np.where(inside, plane_distance, edge_distance)


def _point_to_segment_distances(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    segments = ends - starts
    length_squared = np.einsum("ijk,ijk->ij", segments, segments)
    projection = np.einsum("ijk,ijk->ij", points - starts, segments) / np.maximum(
        length_squared,
        1e-24,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, :, None] * segments
    delta = points - closest
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))


def _repair_rejection_reasons(
    report: StlReport,
    dimension_change: float,
    projected_area_change: float,
    source_deviation_p95: float,
    source_deviation_p99: float,
    added_deviation_p95: float,
    added_far_fraction: float,
    detail_resolution: float,
    max_dimension_change: float,
    max_projected_area_change: float,
    max_deviation_p95: float,
    max_deviation_p99: float,
    max_detail_resolution: float,
    max_added_deviation_p95: float,
    max_added_far_fraction: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not report.is_cfd_candidate:
        reasons.append(
            "Prepared mesh still has "
            f"{report.open_edge_count} open, {report.non_manifold_edge_count} non-manifold, "
            f"and {report.degenerate_triangle_count} degenerate elements."
        )
    checks = (
        (dimension_change, max_dimension_change, "Bounding-size change"),
        (projected_area_change, max_projected_area_change, "Projected-area change"),
        (source_deviation_p95, max_deviation_p95, "Source-surface deviation p95"),
        (source_deviation_p99, max_deviation_p99, "Source-surface deviation p99"),
        (added_deviation_p95, max_added_deviation_p95, "Added-surface deviation p95"),
        (
            added_far_fraction,
            max_added_far_fraction,
            f"Added surface farther than {REPAIR_ADDED_SURFACE_FAR_DISTANCE * 100:.1f}% of model length",
        ),
        (detail_resolution, max_detail_resolution, "Repair detail cell size"),
    )
    for value, limit, label in checks:
        if value > limit:
            reasons.append(f"{label} is {value * 100.0:.2f}%; limit is {limit * 100.0:.2f}%.")
    return tuple(reasons)


def _rasterize_triangle(
    surface: np.ndarray,
    triangle: Triangle,
    origin: np.ndarray,
    voxel_size: float,
) -> None:
    points = (np.asarray(triangle, dtype=np.float64) - origin) / voxel_size
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[0]
    normal = np.cross(edge_a, edge_b)
    dominant = int(np.argmax(np.abs(normal)))
    axes = [axis for axis in range(3) if axis != dominant]
    projected = points[:, axes]

    _mark_point(surface, np.mean(points, axis=0))
    for point in points:
        _mark_point(surface, point)
    for start, end in ((points[0], points[1]), (points[1], points[2]), (points[2], points[0])):
        steps = max(1, int(math.ceil(float(np.linalg.norm(end - start)) * 2.25)))
        for index in range(steps + 1):
            _mark_point(surface, start + (end - start) * (index / steps))

    denominator = (
        (projected[1, 1] - projected[2, 1]) * (projected[0, 0] - projected[2, 0])
        + (projected[2, 0] - projected[1, 0]) * (projected[0, 1] - projected[2, 1])
    )
    if abs(float(denominator)) <= 1e-12:
        return

    low = np.floor(np.min(projected, axis=0) - 0.5).astype(int)
    high = np.ceil(np.max(projected, axis=0) + 0.5).astype(int)
    for first in range(int(low[0]), int(high[0]) + 1):
        for second in range(int(low[1]), int(high[1]) + 1):
            sample = np.asarray((first + 0.5, second + 0.5), dtype=np.float64)
            weight_a = (
                (projected[1, 1] - projected[2, 1]) * (sample[0] - projected[2, 0])
                + (projected[2, 0] - projected[1, 0]) * (sample[1] - projected[2, 1])
            ) / denominator
            weight_b = (
                (projected[2, 1] - projected[0, 1]) * (sample[0] - projected[2, 0])
                + (projected[0, 0] - projected[2, 0]) * (sample[1] - projected[2, 1])
            ) / denominator
            weight_c = 1.0 - weight_a - weight_b
            if min(weight_a, weight_b, weight_c) < -0.03:
                continue
            point = points[0] * weight_a + points[1] * weight_b + points[2] * weight_c
            _mark_point(surface, point)


def _mark_point(surface: np.ndarray, point: np.ndarray) -> None:
    index = np.rint(point).astype(int)
    index = np.minimum(np.maximum(index, 0), np.asarray(surface.shape) - 1)
    surface[tuple(index)] = True


def _dilate(surface: np.ndarray) -> np.ndarray:
    padded = np.pad(surface, 1, mode="constant", constant_values=False)
    expanded = np.zeros_like(surface)
    for x_shift in range(3):
        for y_shift in range(3):
            for z_shift in range(3):
                expanded |= padded[
                    x_shift : x_shift + surface.shape[0],
                    y_shift : y_shift + surface.shape[1],
                    z_shift : z_shift + surface.shape[2],
                ]
    return expanded


def _flood_outside(surface: np.ndarray) -> np.ndarray:
    shape = surface.shape
    stride_x = shape[1] * shape[2]
    stride_y = shape[2]
    total = shape[0] * shape[1] * shape[2]
    blocked = surface.reshape(-1)
    visited = bytearray(total)
    queue: deque[int] = deque([0])
    visited[0] = 1

    while queue:
        index = queue.popleft()
        x = index // stride_x
        remainder = index - x * stride_x
        y = remainder // stride_y
        z = remainder - y * stride_y
        neighbors = []
        if x > 0:
            neighbors.append(index - stride_x)
        if x + 1 < shape[0]:
            neighbors.append(index + stride_x)
        if y > 0:
            neighbors.append(index - stride_y)
        if y + 1 < shape[1]:
            neighbors.append(index + stride_y)
        if z > 0:
            neighbors.append(index - 1)
        if z + 1 < shape[2]:
            neighbors.append(index + 1)
        for neighbor in neighbors:
            if not visited[neighbor] and not blocked[neighbor]:
                visited[neighbor] = 1
                queue.append(neighbor)

    return np.frombuffer(visited, dtype=np.uint8).reshape(shape).astype(bool)


def _regularize_solid(solid: np.ndarray) -> np.ndarray:
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    face_neighbors = np.zeros(solid.shape, dtype=np.uint8)
    for axis in range(3):
        lower = [slice(1, -1), slice(1, -1), slice(1, -1)]
        upper = [slice(1, -1), slice(1, -1), slice(1, -1)]
        lower[axis] = slice(0, -2)
        upper[axis] = slice(2, None)
        face_neighbors += padded[tuple(lower)]
        face_neighbors += padded[tuple(upper)]
    return solid | (face_neighbors >= 5)


def _erode(solid: np.ndarray) -> np.ndarray:
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    eroded = solid.copy()
    for x_shift in range(3):
        for y_shift in range(3):
            for z_shift in range(3):
                eroded &= padded[
                    x_shift : x_shift + solid.shape[0],
                    y_shift : y_shift + solid.shape[1],
                    z_shift : z_shift + solid.shape[2],
                ]
    return eroded


def _remove_edge_contacts(solid: np.ndarray) -> np.ndarray:
    result = solid.copy()
    for _ in range(4):
        changed = False
        planes = (
            (
                result[:, :-1, :-1],
                result[:, 1:, :-1],
                result[:, :-1, 1:],
                result[:, 1:, 1:],
            ),
            (
                result[:-1, :, :-1],
                result[1:, :, :-1],
                result[:-1, :, 1:],
                result[1:, :, 1:],
            ),
            (
                result[:-1, :-1, :],
                result[1:, :-1, :],
                result[:-1, 1:, :],
                result[1:, 1:, :],
            ),
        )
        for a, b, c, d in planes:
            first_diagonal = a & d & ~b & ~c
            second_diagonal = b & c & ~a & ~d
            if np.any(first_diagonal):
                b[first_diagonal] = True
                c[first_diagonal] = True
                changed = True
            if np.any(second_diagonal):
                a[second_diagonal] = True
                d[second_diagonal] = True
                changed = True
        if not changed:
            break
    return result


def _write_voxel_surface(path: Path, solid: np.ndarray, origin: np.ndarray, voxel_size: float) -> None:
    face_specs = (
        (0, -1, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        (0, 1, ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        (1, -1, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        (1, 1, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        (2, -1, ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        (2, 1, ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    exposed_faces: list[tuple[np.ndarray, tuple[tuple[int, int, int], ...]]] = []
    for axis, direction, corners in face_specs:
        neighbor = np.zeros_like(solid)
        current_slice = [slice(None), slice(None), slice(None)]
        neighbor_slice = [slice(None), slice(None), slice(None)]
        if direction < 0:
            current_slice[axis] = slice(1, None)
            neighbor_slice[axis] = slice(None, -1)
        else:
            current_slice[axis] = slice(None, -1)
            neighbor_slice[axis] = slice(1, None)
        neighbor[tuple(current_slice)] = solid[tuple(neighbor_slice)]
        indices = np.argwhere(solid & ~neighbor)
        exposed_faces.append((indices, corners))

    vertex_ids: dict[tuple[int, int, int], int] = {}
    grid_vertices: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    for indices, corners in exposed_faces:
        for index in indices:
            quad: list[int] = []
            for corner in corners:
                key = tuple(int(index[axis]) + corner[axis] for axis in range(3))
                vertex_id = vertex_ids.get(key)
                if vertex_id is None:
                    vertex_id = len(grid_vertices)
                    vertex_ids[key] = vertex_id
                    grid_vertices.append(key)
                quad.append(vertex_id)
            faces.append((quad[0], quad[1], quad[2]))
            faces.append((quad[0], quad[2], quad[3]))

    vertices = origin + np.asarray(grid_vertices, dtype=np.float64) * voxel_size
    vertices = _taubin_smooth(vertices, faces, iterations=8)

    with path.open("wb") as stream:
        header = b"AeroLab sealed voxel shell"[:80].ljust(80, b" ")
        stream.write(header)
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            triangle = tuple(tuple(float(value) for value in vertices[index]) for index in face)
            _write_binary_triangle(stream, triangle)  # type: ignore[arg-type]


def _taubin_smooth(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    iterations: int,
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for start, end in ((a, b), (b, c), (c, a)):
            edges.add((start, end) if start < end else (end, start))
    edge_array = np.asarray(tuple(edges), dtype=np.int64)
    first = edge_array[:, 0]
    second = edge_array[:, 1]
    degree = np.bincount(edge_array.reshape(-1), minlength=len(vertices)).astype(np.float64)
    points = vertices.copy()
    for _ in range(iterations):
        for coefficient in (0.42, -0.435):
            totals = np.zeros_like(points)
            np.add.at(totals, first, points[second])
            np.add.at(totals, second, points[first])
            averages = totals / np.maximum(1.0, degree)[:, None]
            points = points + coefficient * (averages - points)
    return points


def _write_binary_triangle(stream: object, triangle: Triangle) -> None:
    a, b, c = triangle
    normal = _normal(a, b, c)
    values = (*normal, *a, *b, *c, 0)
    stream.write(struct.pack("<12fH", *values))  # type: ignore[attr-defined]


def _normal(a: Vector, b: Vector, c: Vector) -> Vector:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in cross))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (cross[0] / length, cross[1] / length, cross[2] / length)


def _dimension_change(source: StlReport, output: StlReport) -> float:
    changes = []
    for before, after in zip(source.bounds.dimensions, output.bounds.dimensions):
        if before > 1e-12:
            changes.append(abs(after - before) / before)
    return max(changes, default=0.0)


def _projected_area_change(source: StlReport, output: StlReport) -> float:
    source_areas = source.silhouette_projected_areas or source.projected_areas
    output_areas = output.silhouette_projected_areas or output.projected_areas
    return max(
        _relative_change(source_areas.x, output_areas.x),
        _relative_change(source_areas.y, output_areas.y),
        _relative_change(source_areas.z, output_areas.z),
    )


def _relative_change(before: float, after: float) -> float:
    if abs(before) <= 1e-12:
        return 0.0 if abs(after) <= 1e-12 else math.inf
    return abs(after - before) / abs(before)
