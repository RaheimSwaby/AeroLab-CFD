from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from aerolab.repair import (
    _fill_small_boundary_loops,
    _nearest_triangle_distances,
    _projected_area_change,
    _repair_local_non_manifold_overlays,
    _repair_rejection_reasons,
    _retriangulate_degenerate_faces,
    _surface_deviation_metrics,
    _symmetric_surface_deviation,
    repair_fidelity_for_model,
    repair_stl,
)
from aerolab.stl import (
    Bounds,
    ProjectedAreas,
    StlReport,
    report_for_triangles,
    write_binary_stl_triangles,
)


class RepairFidelityTests(unittest.TestCase):
    def test_projected_fidelity_uses_visible_silhouette_instead_of_face_sum(self) -> None:
        source = StlReport(
            path="source.stl",
            format="binary",
            triangle_count=4,
            unique_vertex_count=8,
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            surface_area=6.0,
            projected_areas=ProjectedAreas(10.0, 10.0, 10.0),
            volume=1.0,
            open_edge_count=0,
            non_manifold_edge_count=0,
            degenerate_triangle_count=0,
            warnings=(),
            silhouette_projected_areas=ProjectedAreas(1.0, 1.0, 1.0),
        )
        output = replace(
            source,
            projected_areas=ProjectedAreas(20.0, 20.0, 20.0),
            silhouette_projected_areas=ProjectedAreas(1.01, 1.0, 1.0),
        )

        self.assertAlmostEqual(_projected_area_change(source, output), 0.01)

    def test_local_non_manifold_overlay_repair_preserves_dominant_surface(self) -> None:
        p000, p100, p110, p010 = (
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        )
        p001, p101, p111, p011 = (
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        )
        a, b, c, d = (
            (0.495, 0.495, 1.0), (0.505, 0.495, 1.0),
            (0.505, 0.505, 1.0), (0.495, 0.505, 1.0),
        )
        open_cube = [
            (p000, p110, p100), (p000, p010, p110),
            (p000, p100, p101), (p000, p101, p001),
            (p100, p110, p111), (p100, p111, p101),
            (p110, p010, p011), (p110, p011, p111),
            (p010, p000, p001), (p010, p001, p011),
            (p001, p101, b), (p001, b, a),
            (p101, p111, c), (p101, c, b),
            (p111, p011, d), (p111, d, c),
            (p011, p001, a), (p011, a, d),
        ]
        triangles = _fill_small_boundary_loops(open_cube, longest=3 ** 0.5)
        self.assertIsNotNone(triangles)
        contaminated = [*triangles, tuple(reversed(triangles[-1]))]
        self.assertGreater(
            report_for_triangles(Path("contaminated.stl"), contaminated, "binary").non_manifold_edge_count,
            0,
        )

        repaired = _repair_local_non_manifold_overlays(
            contaminated,
            longest=3 ** 0.5,
            minimum_dominant_fraction=0.8,
            maximum_discarded_faces=2,
        )

        self.assertIsNotNone(repaired)
        self.assertTrue(report_for_triangles(Path("repaired.stl"), repaired, "binary").is_cfd_candidate)

    def test_small_boundary_loop_is_filled_without_moving_source_triangles(self) -> None:
        p000, p100, p110, p010 = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        p001, p101, p111, p011 = (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
        a, b, c, d = (
            (0.495, 0.495, 1.0),
            (0.505, 0.495, 1.0),
            (0.505, 0.505, 1.0),
            (0.495, 0.505, 1.0),
        )
        triangles = [
            (p000, p110, p100), (p000, p010, p110),
            (p000, p100, p101), (p000, p101, p001),
            (p100, p110, p111), (p100, p111, p101),
            (p110, p010, p011), (p110, p011, p111),
            (p010, p000, p001), (p010, p001, p011),
            (p001, p101, b), (p001, b, a),
            (p101, p111, c), (p101, c, b),
            (p111, p011, d), (p111, d, c),
            (p011, p001, a), (p011, a, d),
        ]

        patched = _fill_small_boundary_loops(triangles, longest=3 ** 0.5)

        self.assertIsNotNone(patched)
        self.assertEqual(patched[: len(triangles)], triangles)
        self.assertEqual(len(patched), len(triangles) + 4)
        self.assertTrue(report_for_triangles(Path("patched.stl"), patched, "binary").is_cfd_candidate)

    def test_nearest_triangle_search_is_exact_when_centroids_are_misleading(self) -> None:
        large = np.asarray([[[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]]])
        distractors = []
        for index in range(40):
            x = (index % 8) * 0.002
            y = (index // 8) * 0.002
            distractors.append([[x, y, 0.1], [x + 0.0005, y, 0.1], [x, y + 0.0005, 0.1]])
        triangles = np.concatenate((large, np.asarray(distractors)), axis=0)

        distances = _nearest_triangle_distances(np.asarray([[0.0, 0.0, 0.0]]), triangles)

        self.assertAlmostEqual(float(distances[0]), 0.0, places=12)

    def test_retriangulates_collinear_stitches_without_moving_vertices(self) -> None:
        a = (0.0, 0.0, 0.0)
        b = (3.0, 0.0, 0.0)
        c = (0.0, 1.0, 0.0)
        d = (0.0, 0.0, 1.0)
        p = (1.0, 0.0, 0.0)
        q = (2.0, 0.0, 0.0)
        triangles = [
            (a, c, b),
            (a, p, d),
            (p, q, d),
            (q, b, d),
            (a, d, c),
            (b, c, d),
            (a, b, q),
            (a, q, p),
        ]
        source = report_for_triangles(Path("source.stl"), triangles, "binary")

        repaired = _retriangulate_degenerate_faces(triangles)

        self.assertEqual(source.open_edge_count, 0)
        self.assertEqual(source.degenerate_triangle_count, 2)
        self.assertIsNotNone(repaired)
        output = report_for_triangles(Path("output.stl"), repaired, "binary")
        self.assertTrue(output.is_cfd_candidate)
        self.assertEqual(len(repaired), len(triangles))
        self.assertEqual(
            {vertex for triangle in repaired for vertex in triangle},
            {vertex for triangle in triangles for vertex in triangle},
        )
        self.assertAlmostEqual(output.surface_area, source.surface_area, places=12)
        self.assertAlmostEqual(output.volume, source.volume, places=12)

    def test_repair_accepts_exact_degenerate_face_retriangulation(self) -> None:
        a = (0.0, 0.0, 0.0)
        b = (3.0, 0.0, 0.0)
        c = (0.0, 1.0, 0.0)
        d = (0.0, 0.0, 1.0)
        p = (1.0, 0.0, 0.0)
        q = (2.0, 0.0, 0.0)
        triangles = [
            (a, c, b),
            (a, p, d),
            (p, q, d),
            (q, b, d),
            (a, d, c),
            (b, c, d),
            (a, b, q),
            (a, q, p),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.stl"
            output_path = Path(temp_dir) / "prepared" / "source-prepared.stl"
            write_binary_stl_triangles(source_path, triangles)

            result = repair_stl(
                source_path,
                output_path,
                surface_deviation_sample_count=100,
            )

            self.assertTrue(result.accepted)
            self.assertEqual(result.method, "degenerate_face_retriangulation")
            self.assertEqual(result.surface_deviation_sample_count, 0)
            self.assertEqual(result.output_report.degenerate_triangle_count, 0)
            self.assertEqual(result.output_report.open_edge_count, 0)
            self.assertEqual(result.output_report.non_manifold_edge_count, 0)

    def test_coarse_voxel_cell_fails_body_line_fidelity(self) -> None:
        report = StlReport(
            path="prepared.stl",
            format="binary",
            triangle_count=1000,
            unique_vertex_count=502,
            bounds=Bounds((0.0, 0.0, 0.0), (4.5, 2.0, 1.5)),
            surface_area=20.0,
            projected_areas=ProjectedAreas(3.0, 6.0, 8.0),
            volume=5.0,
            open_edge_count=0,
            non_manifold_edge_count=0,
            degenerate_triangle_count=0,
            warnings=(),
        )

        reasons = _repair_rejection_reasons(
            report,
            dimension_change=0.0,
            projected_area_change=0.0,
            source_deviation_p95=0.0,
            source_deviation_p99=0.0,
            added_deviation_p95=0.0,
            added_far_fraction=0.0,
            detail_resolution=1.0 / 160.0,
            max_dimension_change=0.02,
            max_projected_area_change=0.02,
            max_deviation_p95=0.003,
            max_deviation_p99=0.006,
            max_detail_resolution=0.003,
            max_added_deviation_p95=0.004,
            max_added_far_fraction=0.025,
        )

        self.assertTrue(any("Repair detail cell size" in reason for reason in reasons))

    def test_prepared_mesh_requires_matching_fidelity_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = Path(temp_dir) / "prepared"
            prepared.mkdir()
            model_path = prepared / "car-prepared.stl"
            model_path.write_text("solid car\nendsolid car\n", encoding="utf-8")
            self.assertFalse(repair_fidelity_for_model(model_path)["verified"])

            metadata = {
                "accepted": True,
                "outputPath": str(model_path),
                "sourcePath": str(Path(temp_dir) / "car.stl"),
                "method": "sealed_voxel_shell",
                "gridResolution": 384,
                "voxelSize": 0.01,
                "detailResolutionPercent": 0.2604,
                "dimensionChangePercent": 0.5,
                "surfaceDeviationP95Percent": 0.2,
                "surfaceDeviationP99Percent": 0.4,
                "sourceSurfaceDeviationP95Percent": 0.2,
                "sourceSurfaceDeviationP99Percent": 0.4,
                "addedSurfaceDeviationP95Percent": 0.3,
                "addedSurfaceFarFractionPercent": 2.0,
                "projectedAreaChangePercent": 0.5,
                "rejectionReasons": [],
            }
            model_path.with_suffix(".repair.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            fidelity = repair_fidelity_for_model(model_path)
            self.assertTrue(fidelity["verified"])
            self.assertEqual(fidelity["gridResolution"], 384)

            metadata["outputPath"] = r"C:\moved-project\prepared\car-prepared.stl"
            model_path.with_suffix(".repair.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertTrue(repair_fidelity_for_model(model_path)["verified"])

            metadata["outputPath"] = r"C:\moved-project\prepared\another-model.stl"
            model_path.with_suffix(".repair.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self.assertFalse(repair_fidelity_for_model(model_path)["verified"])

    def test_directional_metrics_separate_preserved_and_added_surface(self) -> None:
        source = [((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))]
        output = source + [((0.0, 0.0, 0.5), (2.0, 0.0, 0.5), (0.0, 2.0, 0.5))]

        metrics = _surface_deviation_metrics(
            source,
            output,
            sample_count=400,
            far_distance=0.25,
        )

        self.assertAlmostEqual(metrics["sourceP99"], 0.0, places=9)
        self.assertGreater(metrics["outputP95"], 0.0)
        self.assertGreater(metrics["outputFarFraction"], 0.0)
    def test_identical_surface_has_zero_deviation(self) -> None:
        triangles = [
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
            ((2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
        ]

        p95, p99 = _symmetric_surface_deviation(triangles, triangles, sample_count=200)

        self.assertAlmostEqual(p95, 0.0, places=9)
        self.assertAlmostEqual(p99, 0.0, places=9)

    def test_parallel_surface_shift_is_measured(self) -> None:
        source = [((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))]
        shifted = [((0.0, 0.0, 0.25), (2.0, 0.0, 0.25), (0.0, 2.0, 0.25))]

        p95, p99 = _symmetric_surface_deviation(source, shifted, sample_count=200)

        self.assertAlmostEqual(p95, 0.25, places=9)
        self.assertAlmostEqual(p99, 0.25, places=9)


if __name__ == "__main__":
    unittest.main()
