"""Tests for controlled A/B case comparison.

The module's purpose is refusing comparisons that are not decision-safe, so
these tests focus on the setup lock and the gating that blocks a claim.
"""

import json
import tempfile
import unittest
from pathlib import Path

from aerolab.solver.comparison import compare_cases


def _payload(name: str, speed_mps: float = 31.3, area_m2: float = 3.9) -> dict[str, object]:
    return {
        "name": name,
        "status": "openfoam_case_generated",
        "solver_target": "openfoam-foundation-v13",
        "solver_module": "incompressibleFluid",
        "simulation_type": "steady_external_incompressible_airflow",
        "flow": {
            "axis": "x",
            "speed_mps": speed_mps,
            "air_density_kg_m3": 1.225,
            "kinematic_viscosity_m2_s": 1.5e-5,
        },
        "ground": {"enabled": True, "moving": True, "clearance_m": 0.0},
        "placement": {"method": "lowest_point_to_road_clearance"},
        "aerodynamic_reference": {"area_m2": area_m2, "length_m": 4.5},
        "cfd_quality": {"name": "standard"},
    }


def _write_case(root: Path, name: str, **kwargs) -> Path:
    case_path = root / name
    case_path.mkdir(parents=True)
    (case_path / "case.json").write_text(
        json.dumps(_payload(name, **kwargs)), encoding="utf-8"
    )
    return case_path


class CompareCasesTests(unittest.TestCase):
    def test_rejects_comparing_a_case_with_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _write_case(Path(tmp), "only")
            with self.assertRaises(ValueError):
                compare_cases(case, case)

    def test_identical_setup_locks_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline"), _write_case(root, "variant")
            )
            self.assertTrue(result["locksMatch"])
            self.assertEqual(result["setupDifferences"], [])
            self.assertNotEqual(result["status"], "setup_mismatch")

    def test_different_speed_breaks_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline", speed_mps=31.3),
                _write_case(root, "variant", speed_mps=40.0),
            )
            self.assertFalse(result["locksMatch"])
            self.assertEqual(result["status"], "setup_mismatch")
            self.assertTrue(result["setupDifferences"])
            self.assertFalse(result["decisionSafe"])

    def test_different_reference_area_breaks_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline", area_m2=3.9),
                _write_case(root, "variant", area_m2=4.4),
            )
            self.assertFalse(result["locksMatch"])
            self.assertEqual(result["status"], "setup_mismatch")

    def test_unsolved_cases_are_never_decision_safe(self):
        # Matching setup but no solver output: loads are incomplete.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline"), _write_case(root, "variant")
            )
            self.assertEqual(result["status"], "incomplete_loads")
            self.assertFalse(result["decisionSafe"])
            self.assertFalse(result["statisticalDecisionSafe"])
            self.assertEqual(
                result["statisticalStatus"], "controlled_comparison_required"
            )

    def test_geometry_is_excluded_from_the_lock(self):
        # A/B testing requires differing geometry with an otherwise identical setup.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline"), _write_case(root, "variant")
            )
            self.assertTrue(result["geometryExcludedFromLock"])

    def test_reports_an_interpretation_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare_cases(
                _write_case(root, "baseline"), _write_case(root, "variant")
            )
            self.assertEqual(result["schemaVersion"], 2)
            self.assertIsInstance(result["interpretation"], str)
            self.assertTrue(result["interpretation"])


if __name__ == "__main__":
    unittest.main()
