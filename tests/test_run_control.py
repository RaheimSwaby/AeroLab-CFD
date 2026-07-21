"""Safety tests for run-output deletion and interrupted-run reconciliation.

Deleting run outputs is irreversible, so the contract that matters is: it removes
generated solver products but never touches the geometry, case setup, or the
initial-conditions directory. Reconciliation must only act on a genuinely
interrupted record and must never mark a finished run as failed.
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from aerolab.solver import (
    SolverRunCancelled,
    SolverRunController,
    clear_case_run_outputs,
    reconcile_interrupted_case_run,
)
from aerolab.webapp import AeroLabServer


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


class DestructiveEndpointConfinementTests(unittest.TestCase):
    """The stop and delete endpoints must refuse any path outside cases/."""

    @staticmethod
    def _post(server: AeroLabServer, path: str, case_path: Path):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps({"casePath": str(case_path)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urlopen(request, timeout=10)

    def _assert_rejected_outside_cases(self, endpoint: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cases").mkdir()
            # A case-shaped directory that lives OUTSIDE cases/.
            outside = root / "outside"
            outside.mkdir()
            (outside / "case.json").write_text(
                json.dumps({"name": "x", "status": "complete"}), encoding="utf-8"
            )
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with self.assertRaises(HTTPError) as context:
                    self._post(server, endpoint, outside)
                self.assertEqual(context.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            # A rejected request must not have touched the out-of-bounds case.
            self.assertTrue((outside / "case.json").exists())

    def test_delete_endpoint_rejects_path_outside_cases(self) -> None:
        self._assert_rejected_outside_cases("/api/delete-run-outputs")

    def test_stop_endpoint_rejects_path_outside_cases(self) -> None:
        self._assert_rejected_outside_cases("/api/stop-run")


class SolverRunControllerTests(unittest.TestCase):
    """The cancellation state machine and its ownership authorization."""

    def test_fresh_controller_is_not_cancelled(self) -> None:
        controller = SolverRunController()
        self.assertFalse(controller.cancellation_requested)
        controller.raise_if_cancelled()  # must not raise
        self.assertTrue(controller.attempt_id)

    def test_attempt_ids_are_unique(self) -> None:
        self.assertNotEqual(
            SolverRunController().attempt_id, SolverRunController().attempt_id
        )

    def test_request_stop_cancels_and_blocks_further_progress(self) -> None:
        controller = SolverRunController()
        # No processes were registered, so nothing is terminated.
        self.assertEqual(controller.request_stop(), 0)
        self.assertTrue(controller.cancellation_requested)
        with self.assertRaises(SolverRunCancelled):
            controller.raise_if_cancelled()

    def test_ownership_cannot_change_after_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_a = Path(tmp) / "a"
            case_b = Path(tmp) / "b"
            case_a.mkdir()
            case_b.mkdir()
            controller = SolverRunController()
            controller.set_owned_case_paths([case_a])
            controller.set_owned_case_paths([case_a])  # same set is idempotent
            self.assertTrue(controller.owns_case_paths([case_a]))
            self.assertFalse(controller.owns_case_paths([case_b]))
            self.assertFalse(controller.owns_case_paths([]))  # empty is never owned
            with self.assertRaises(RuntimeError):
                controller.set_owned_case_paths([case_b])  # reassignment is refused


if __name__ == "__main__":
    unittest.main()
