"""Safety tests for run-output deletion and interrupted-run reconciliation.

Deleting run outputs is irreversible, so the contract that matters is: it removes
generated solver products but never touches the geometry, case setup, or the
initial-conditions directory. Reconciliation must only act on a genuinely
interrupted record and must never mark a finished run as failed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from aerolab.solver import clear_case_run_outputs, reconcile_interrupted_case_run


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_solved_case(root: Path, status: str = "complete") -> Path:
    """A case dir carrying both preserved setup and deletable solver output."""
    case = root / "case"
    # --- setup / inputs that must survive a delete ---
    _write(case / "case.json", json.dumps({"name": "case", "status": status}))
    _write(case / "constant" / "geometry" / "body.stl", "solid body")
    _write(case / "system" / "controlDict", "controlDict")
    _write(case / "0" / "U", "initial U")  # initial conditions: preserved
    _write(case / "Allrun", "#!/bin/sh")
    # --- generated solver output that must be removed ---
    _write(case / "500" / "U", "solved U")  # numeric time-step
    _write(case / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat", "cd cl")
    _write(case / "constant" / "polyMesh" / "points", "points")
    _write(case / "processor0" / "0" / "U", "decomposed")
    _write(case / "aerolab-run.json", json.dumps({"status": status, "ok": True}))
    _write(case / "aerolab-run.log", "solver log")
    return case


class ClearCaseRunOutputsTests(unittest.TestCase):
    def test_preserves_setup_and_removes_solver_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _make_solved_case(Path(tmp))
            result = clear_case_run_outputs(case)

            # Setup, geometry, and initial conditions must survive.
            for survivor in (
                "case.json",
                "constant/geometry/body.stl",
                "system/controlDict",
                "0/U",
                "Allrun",
            ):
                with self.subTest(survivor=survivor):
                    self.assertTrue((case / survivor).exists(), f"{survivor} was deleted")

            # Generated solver products must be gone.
            for removed in (
                "500",
                "postProcessing",
                "processor0",
                "aerolab-run.json",
                "aerolab-run.log",
            ):
                with self.subTest(removed=removed):
                    self.assertFalse((case / removed).exists(), f"{removed} survived")

            self.assertIn("aerolab-run.json", result["deletedFiles"])
            self.assertIn("aerolab-run.log", result["deletedFiles"])
            # No reusable mesh record was present, so the mesh is not preserved.
            self.assertFalse(result["preservedMesh"])
            self.assertEqual(result["resetStatus"], "openfoam_case_generated")

    def test_rejects_a_non_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "not-a-case"
            plain.mkdir()
            with self.assertRaises(ValueError):
                clear_case_run_outputs(plain)

    def test_resets_case_status_after_clearing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = _make_solved_case(Path(tmp), status="complete")
            clear_case_run_outputs(case)
            payload = json.loads((case / "case.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "openfoam_case_generated")


class ReconcileInterruptedRunTests(unittest.TestCase):
    @staticmethod
    def _case(root: Path, run_record: dict[str, object] | None) -> Path:
        case = root / "case"
        case.mkdir(parents=True)
        (case / "case.json").write_text(json.dumps({"name": "c", "status": "x"}), encoding="utf-8")
        if run_record is not None:
            (case / "aerolab-run.json").write_text(json.dumps(run_record), encoding="utf-8")
        return case

    def test_no_run_record_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(reconcile_interrupted_case_run(self._case(Path(tmp), None)))

    def test_finished_run_is_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case(Path(tmp), {"status": "complete", "ok": True})
            self.assertIsNone(reconcile_interrupted_case_run(case))
            record = json.loads((case / "aerolab-run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")  # not downgraded

    def test_interrupted_native_run_is_orphaned(self) -> None:
        # A "running" record with no recoverable backend identity cannot be
        # verified, so it must be marked orphaned rather than left as running.
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case(Path(tmp), {"status": "running", "backend": "native", "mode": "full"})
            reconcile_interrupted_case_run(case)
            record = json.loads((case / "aerolab-run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "orphaned")
            self.assertFalse(record["ok"])


if __name__ == "__main__":
    unittest.main()
