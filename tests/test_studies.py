"""Tests for sensitivity-study setup.

A sensitivity sweep drives physical case generation, so the parameter and value
guard rails are the part worth pinning down: a silently accepted out-of-range
value would produce a physically meaningless case family.
"""

import json
import tempfile
import unittest
from pathlib import Path

from aerolab.solver.studies import (
    SENSITIVITY_PARAMETERS,
    _default_baseline_index,
    _parameter_specification,
    _sensitivity_values,
    sensitivity_study_report,
)


def _spec(name: str) -> dict[str, object]:
    return _parameter_specification(name)


class ParameterSpecificationTests(unittest.TestCase):
    def test_known_parameter_returns_specification(self):
        specification = _parameter_specification("speed_mph")
        self.assertEqual(specification["unit"], "mph")
        self.assertIn("label", specification)

    def test_unknown_parameter_lists_supported_options(self):
        with self.assertRaises(ValueError) as context:
            _parameter_specification("wing_angle")
        message = str(context.exception)
        self.assertIn("wing_angle", message)
        self.assertIn("speed_mph", message)  # names the supported set

    def test_every_parameter_declares_label_and_unit(self):
        for name, specification in SENSITIVITY_PARAMETERS.items():
            with self.subTest(parameter=name):
                self.assertTrue(specification.get("label"))
                self.assertIn("unit", specification)


class SensitivityValueValidationTests(unittest.TestCase):
    def test_accepts_a_normal_sweep(self):
        self.assertEqual(
            _sensitivity_values([50.0, 70.0], _spec("speed_mph")), [50.0, 70.0]
        )

    def test_requires_at_least_two_values(self):
        with self.assertRaises(ValueError):
            _sensitivity_values([50.0], _spec("speed_mph"))

    def test_rejects_more_than_twelve_values(self):
        with self.assertRaises(ValueError):
            _sensitivity_values([float(v) for v in range(1, 15)], _spec("speed_mph"))

    def test_rejects_non_finite_values(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                _sensitivity_values([50.0, bad], _spec("speed_mph"))

    def test_rejects_duplicate_values(self):
        with self.assertRaises(ValueError):
            _sensitivity_values([50.0, 50.0], _spec("speed_mph"))

    def test_exclusive_minimum_rejects_the_bound(self):
        # speed_mph declares minimum 0.0 exclusively.
        with self.assertRaises(ValueError):
            _sensitivity_values([0.0, 70.0], _spec("speed_mph"))

    def test_exclusive_maximum_rejects_the_bound(self):
        # yaw_degrees is bounded to (-90, 90) exclusively.
        with self.assertRaises(ValueError):
            _sensitivity_values([0.0, 90.0], _spec("yaw_degrees"))
        self.assertEqual(
            _sensitivity_values([-10.0, 10.0], _spec("yaw_degrees")), [-10.0, 10.0]
        )

    def test_inclusive_minimum_accepts_the_bound(self):
        specification = _spec("ground_clearance_m")
        self.assertEqual(_sensitivity_values([0.0, 0.05], specification), [0.0, 0.05])
        with self.assertRaises(ValueError):
            _sensitivity_values([-0.1, 0.05], specification)

    def test_inclusive_maximum_accepts_the_bound(self):
        specification = _spec("turbulence_intensity_percent")
        self.assertEqual(
            _sensitivity_values([10.0, 50.0], specification), [10.0, 50.0]
        )
        with self.assertRaises(ValueError):
            _sensitivity_values([10.0, 50.1], specification)

    def test_unbounded_parameter_accepts_negative_values(self):
        # A crosswind component is signed and has no declared bounds.
        self.assertEqual(
            _sensitivity_values([-5.0, 5.0], _spec("crosswind_mps")), [-5.0, 5.0]
        )


class DefaultBaselineIndexTests(unittest.TestCase):
    def test_without_base_value_picks_the_middle(self):
        self.assertEqual(_default_baseline_index([10.0, 20.0, 30.0], None), 1)

    def test_with_base_value_picks_the_nearest(self):
        self.assertEqual(_default_baseline_index([10.0, 20.0, 30.0], 29.0), 2)
        self.assertEqual(_default_baseline_index([10.0, 20.0, 30.0], 11.0), 0)


class SensitivityStudyReportTests(unittest.TestCase):
    @staticmethod
    def _case(root: Path, payload: dict[str, object]) -> Path:
        case_path = root / "case"
        case_path.mkdir(parents=True)
        (case_path / "case.json").write_text(json.dumps(payload), encoding="utf-8")
        return case_path

    def test_returns_none_without_a_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case(Path(tmp), {"name": "plain", "status": "created"})
            self.assertIsNone(sensitivity_study_report(case))

    def test_returns_none_when_the_study_has_no_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case(
                Path(tmp), {"name": "plain", "sensitivity_study": {"parameter": "speed_mph"}}
            )
            self.assertIsNone(sensitivity_study_report(case))

    def test_returns_none_for_a_missing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sensitivity_study_report(Path(tmp) / "nope"))


class StudySchedulerTests(unittest.TestCase):
    @staticmethod
    def _member(root: Path, name: str, cells: int = 1_500_000) -> Path:
        case_path = root / name
        case_path.mkdir()
        case_path.joinpath("case.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "mesh_resolution": {
                        "configured_max_global_cells": cells,
                    },
                }
            ),
            encoding="utf-8",
        )
        return case_path

    def test_auto_allocation_never_exceeds_the_aggregate_rank_budget(self) -> None:
        from aerolab.solver.studies import _study_resource_allocation

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            members = [self._member(root, f"case-{index}") for index in range(3)]
            allocation = _study_resource_allocation(
                members,
                {
                    "effectiveCpus": 8,
                    "memoryAvailableBytes": 64 * 1024**3,
                    "parallelAvailable": True,
                },
                processes="auto",
                process_budget=7,
            )

        self.assertEqual(allocation["processesPerCase"], 2)
        self.assertEqual(allocation["maxConcurrentCases"], 3)
        self.assertEqual(allocation["maximumAllocatedProcesses"], 6)
        self.assertLessEqual(
            allocation["maximumAllocatedProcesses"],
            allocation["processBudget"],
        )

    def test_memory_cap_limits_concurrent_cases(self) -> None:
        from aerolab.solver.studies import _study_resource_allocation

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            members = [
                self._member(root, f"memory-{index}", cells=2_000_000)
                for index in range(3)
            ]
            allocation = _study_resource_allocation(
                members,
                {
                    "effectiveCpus": 8,
                    "memoryAvailableBytes": 10 * 1024**3,
                    "parallelAvailable": True,
                },
                processes=1,
                process_budget=7,
            )

        self.assertEqual(allocation["memoryWorkerCap"], 1)
        self.assertEqual(allocation["maxConcurrentCases"], 1)
        self.assertEqual(allocation["maximumAllocatedProcesses"], 1)

    def test_mpi_unavailable_schedules_independent_serial_cases(self) -> None:
        from aerolab.solver.studies import _study_resource_allocation

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            members = [self._member(root, f"serial-{index}") for index in range(3)]
            allocation = _study_resource_allocation(
                members,
                {
                    "effectiveCpus": 6,
                    "memoryAvailableBytes": 64 * 1024**3,
                    "parallelAvailable": False,
                },
                processes="auto",
                process_budget=5,
            )

        self.assertEqual(allocation["processesPerCase"], 1)
        self.assertEqual(allocation["maxConcurrentCases"], 3)
        self.assertEqual(allocation["maximumAllocatedProcesses"], 3)
        self.assertIn("MPI is unavailable", allocation["selectionReason"])

    def test_duplicate_accuracy_study_members_are_rejected(self) -> None:
        from aerolab.solver.studies import study_members

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected: Path | None = None
            for name, level in (
                ("draft-a", "draft"),
                ("draft-b", "draft"),
                ("standard", "standard"),
            ):
                case_path = root / name
                case_path.mkdir()
                case_path.joinpath("case.json").write_text(
                    json.dumps(
                        {
                            "name": name,
                            "validation_study": {
                                "id": "grid-duplicate",
                                "level": level,
                                "levels": ["draft", "standard"],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                if name == "draft-a":
                    selected = case_path

            assert selected is not None
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                study_members(selected)

    def test_run_study_persists_partial_and_final_records(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from aerolab.solver.studies import _write_study_run_record, run_study

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            members = [self._member(root, f"run-{index}") for index in range(2)]
            plan = {
                "studyId": "scheduler-progress",
                "kind": "grid_convergence",
                "backend": "docker",
                "casePaths": [str(path) for path in members],
                "processesPerCase": 2,
                "processBudget": 4,
                "maxConcurrentCases": 2,
            }
            result = SimpleNamespace(
                ok=True,
                trusted=True,
                returncode=0,
                processes=2,
                file_handler="collated",
                reused_mesh=False,
                log_path=members[0] / "aerolab-run.log",
                process_selection={
                    "qualityRecommendation": {"status": "comfortable"},
                    "stageCache": {"featureHit": True},
                },
            )
            snapshots: list[dict[str, object]] = []

            def capture(paths: list[Path], record: dict[str, object]) -> None:
                snapshots.append(json.loads(json.dumps(record)))
                _write_study_run_record(paths, record)

            with (
                mock.patch(
                    "aerolab.solver.studies.plan_study_schedule",
                    return_value=plan,
                ),
                mock.patch(
                    "aerolab.solver.studies.run_case",
                    return_value=result,
                ) as mocked_run,
                mock.patch(
                    "aerolab.solver.studies._write_study_run_record",
                    side_effect=capture,
                ),
            ):
                final = run_study(
                    members[0],
                    processes="auto",
                    process_budget=4,
                    file_handler="collated",
                )

            self.assertEqual(mocked_run.call_count, 2)
            for call in mocked_run.call_args_list:
                self.assertEqual(call.kwargs["processes"], 2)
                self.assertEqual(call.kwargs["file_handler"], "collated")
            self.assertTrue(
                any(
                    snapshot["status"] == "running"
                    and snapshot["completedCases"] == 1
                    and len(snapshot["results"]) == 1
                    for snapshot in snapshots
                )
            )
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["percent"], 100)
            for member in members:
                stored = json.loads(
                    member.joinpath("aerolab-study-run.json").read_text(encoding="utf-8")
                )
                self.assertEqual(stored, final)


if __name__ == "__main__":
    unittest.main()
