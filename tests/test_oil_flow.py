"""Known-answer tests for the surface oil-flow tracer.

The tracer walks streamlines across a triangulated surface in barycentric
coordinates. On a flat plate under a uniform wall-shear field the answer is
analytic: streamlines are straight, the skin-friction coefficient is
``|tau| / q`` everywhere, and the alignment with the wind is +/-1. That makes it
a clean check of the barycentric walk, triangle adjacency, tangent projection,
and the Cf/alignment math without needing a solved OpenFOAM case.
"""

import unittest

from aerolab.solver.visualization import _surface_oil_flow

Vector = tuple[float, float, float]


def _flat_plate(nx: int, ny: int) -> tuple[list[Vector], list[tuple[int, int, int]]]:
    """A unit-cell flat plate in the z=0 plane, consistently wound (normal +z)."""
    points: list[Vector] = [
        (float(i), float(j), 0.0) for j in range(ny + 1) for i in range(nx + 1)
    ]

    def index(i: int, j: int) -> int:
        return j * (nx + 1) + i

    triangles: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            p00, p10 = index(i, j), index(i + 1, j)
            p11, p01 = index(i + 1, j + 1), index(i, j + 1)
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))
    return points, triangles


class SurfaceOilFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points, self.triangles = _flat_plate(6, 2)

    def _uniform(self, shear: Vector) -> list[Vector]:
        return [shear for _ in self.points]

    def test_uniform_shear_gives_straight_constant_cf_streamlines(self) -> None:
        tau = 5.0
        q = 100.0
        result = _surface_oil_flow(
            self.points,
            self.triangles,
            self._uniform((tau, 0.0, 0.0)),
            q,
            "x",
            wind_vector_mps=(10.0, 0.0, 0.0),
            time_averaged=False,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["lineCount"], 1)
        self.assertEqual(result["windDirection"], [1.0, 0.0, 0.0])

        expected_cf = tau / q  # 0.05
        low, high = result["cfMagnitudeRange"]
        self.assertAlmostEqual(low, expected_cf, places=6)
        self.assertAlmostEqual(high, expected_cf, places=6)

        for path in result["lines"]:
            xs = [sample[0] for sample in path]
            ys = [sample[1] for sample in path]
            cfs = [sample[3] for sample in path]
            aligns = [sample[4] for sample in path]
            # Straight in +x: y stays constant, x is strictly increasing.
            self.assertLess(max(ys) - min(ys), 1e-3)
            self.assertTrue(all(xs[k] < xs[k + 1] for k in range(len(xs) - 1)))
            # Cf is uniform and alignment with the +x wind is +1 everywhere.
            for cf in cfs:
                self.assertAlmostEqual(cf, expected_cf, places=6)
            for alignment in aligns:
                self.assertAlmostEqual(alignment, 1.0, places=6)

    def test_reversed_shear_reads_as_negative_alignment(self) -> None:
        # Shear opposes the wind: alignment must be -1 (separated / reversed flow).
        result = _surface_oil_flow(
            self.points,
            self.triangles,
            self._uniform((-5.0, 0.0, 0.0)),
            100.0,
            "x",
            wind_vector_mps=(10.0, 0.0, 0.0),
            time_averaged=False,
        )
        self.assertIsNotNone(result)
        for path in result["lines"]:
            for sample in path:
                self.assertAlmostEqual(sample[4], -1.0, places=6)

    def test_crossflow_shear_is_orthogonal_to_wind(self) -> None:
        # Shear points +y while the wind points +x: alignment ~ 0.
        result = _surface_oil_flow(
            self.points,
            self.triangles,
            self._uniform((0.0, 5.0, 0.0)),
            100.0,
            "x",
            wind_vector_mps=(10.0, 0.0, 0.0),
            time_averaged=False,
        )
        self.assertIsNotNone(result)
        for path in result["lines"]:
            for sample in path:
                self.assertAlmostEqual(sample[4], 0.0, places=6)

    def test_zero_shear_returns_none(self) -> None:
        self.assertIsNone(
            _surface_oil_flow(
                self.points,
                self.triangles,
                self._uniform((0.0, 0.0, 0.0)),
                100.0,
                "x",
                wind_vector_mps=(10.0, 0.0, 0.0),
                time_averaged=False,
            )
        )

    def test_mismatched_vector_count_returns_none(self) -> None:
        self.assertIsNone(
            _surface_oil_flow(
                self.points,
                self.triangles,
                [(5.0, 0.0, 0.0)],  # wrong length
                100.0,
                "x",
                wind_vector_mps=(10.0, 0.0, 0.0),
                time_averaged=False,
            )
        )

    def test_empty_surface_returns_none(self) -> None:
        self.assertIsNone(
            _surface_oil_flow(
                [], [], [], 100.0, "x", wind_vector_mps=(10.0, 0.0, 0.0), time_averaged=False
            )
        )


if __name__ == "__main__":
    unittest.main()
