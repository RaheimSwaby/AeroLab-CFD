from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from aerolab.stl import (
    detect_aero_features_for_triangles,
    inspect_stl,
    read_stl_triangles,
    silhouette_projected_areas_for_triangles,
    transform_triangles,
    transformed_report,
    write_binary_stl_triangles,
)
from aerolab.webapp import AeroLabServer


class AccuracyStudyApiTests(unittest.TestCase):
    def test_diffuser_detector_flags_rear_ramp_but_not_flat_floor(self) -> None:
        flat_floor = [
            ((0.0, -1.0, 0.0), (4.0, -1.0, 0.0), (4.0, 1.0, 0.0)),
            ((0.0, -1.0, 0.0), (4.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, -1.0, 1.5), (4.0, 1.0, 1.5), (4.0, -1.0, 1.5)),
        ]
        diffuser = flat_floor + [
            ((3.0, -0.7, 0.2), (4.0, -0.7, 0.4), (4.0, 0.7, 0.4)),
            ((3.0, -0.7, 0.2), (4.0, 0.7, 0.4), (3.0, 0.7, 0.2)),
        ]

        flat_result = detect_aero_features_for_triangles(flat_floor)
        diffuser_result = detect_aero_features_for_triangles(diffuser)

        self.assertEqual(flat_result["candidate_count"], 0)
        self.assertEqual(diffuser_result["candidate_count"], 1)
        candidate = diffuser_result["candidates"][0]
        self.assertEqual(candidate["type"], "diffuser_candidate")
        self.assertEqual(candidate["confidence"], "high")
        self.assertAlmostEqual(candidate["angle_degrees"], 11.31, places=2)
        self.assertAlmostEqual(candidate["width_fraction_percent"], 70.0, places=2)

    def test_silhouette_area_preserves_gap_between_projected_panels(self) -> None:
        triangles = [
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
            ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
            ((0.0, 2.0, 0.0), (0.0, 3.0, 0.0), (0.0, 3.0, 1.0)),
            ((0.0, 2.0, 0.0), (0.0, 3.0, 1.0), (0.0, 2.0, 1.0)),
        ]

        silhouette = silhouette_projected_areas_for_triangles(triangles, scanline_count=512)

        self.assertAlmostEqual(silhouette.x, 2.0, places=9)

    def test_principal_axis_alignment_recovers_a_rotated_vehicle_shape(self) -> None:
        project = Path(__file__).resolve().parents[1]
        cube_report = inspect_stl(project / "models" / "sample_box.stl")
        self.assertEqual(cube_report.silhouette_projected_areas.to_dict(), {"x": 1.0, "y": 1.0, "z": 1.0})
        triangles, _ = read_stl_triangles(project / "models" / "sample_box.stl")
        elongated = [
            tuple((vertex[0] * 4.0, vertex[1] * 2.0, vertex[2]) for vertex in triangle)
            for triangle in triangles
        ]
        rotated = transform_triangles(elongated, rotation_degrees=(7.0, -5.0, -31.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "rotated-car-shape.stl"
            write_binary_stl_triangles(model_path, rotated)
            report = inspect_stl(model_path)
            alignment = report.alignment_suggestion
            self.assertIsNotNone(alignment)
            self.assertTrue(alignment["recommended"])
            rotation = alignment["rotation_degrees"]
            aligned = transformed_report(
                model_path,
                scale=1.0,
                rotation_degrees=(rotation["x"], rotation["y"], rotation["z"]),
            )

        for actual, expected in zip(aligned.bounds.dimensions, (4.0, 2.0, 1.0)):
            self.assertAlmostEqual(actual, expected, delta=0.02)

    def test_inverted_orbit_control_is_wired_and_persistent(self) -> None:
        project = Path(__file__).resolve().parents[1]
        index = (project / "src" / "aerolab" / "web" / "index.html").read_text(encoding="utf-8")
        app = (project / "src" / "aerolab" / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="invertOrbit"', index)
        self.assertIn('id="dragModeButton"', index)
        self.assertIn('id="temperatureModeButton"', index)
        self.assertIn('id="heatZonesJson"', index)
        self.assertIn('id="groundClearanceMm"', index)
        self.assertIn("els.invertOrbit.checked ? -1 : 1", app)
        self.assertIn('"aerolab-invert-orbit"', app)
        self.assertIn('setSurfaceMode("drag")', app)
        self.assertIn('setSurfaceMode("temperature")', app)
        self.assertIn("heatZones: engineering", app)
        self.assertIn("temperatureKValues", app)
        self.assertIn("hasSurfaceTemperature", app)
        self.assertIn("pressureDragDisplayRange", app)
        self.assertIn("totalDragDisplayRange", app)
        self.assertIn("triangleTotalDragValues", app)
        self.assertIn("Meshed body fidelity", app)
        self.assertIn("adaptiveMaxGlobalCells", app)
        self.assertIn('els.sidebar.inert = true', app)
        self.assertIn('id="runProgressBar"', index)
        self.assertIn('id="runLogDetails" class="advanced-disclosure run-log-disclosure"', index)
        self.assertIn('id="runLogOutput"', index)
        self.assertIn('id="meshCaseButton"', index)
        self.assertIn('runActiveCase("mesh")', app)
        self.assertIn("reuseMesh: true", app)
        self.assertIn("meshOnly ? 14400 : 21600", app)
        self.assertIn("startRunProgressPolling", app)
        self.assertIn("/api/case-progress?casePath=", app)
        self.assertIn("startRunLogPolling", app)
        self.assertIn("stopRunLogPolling", app)
        self.assertIn("/api/case-log?casePath=", app)
        self.assertIn("els.runLogDetails.open", app)
        self.assertIn("els.runLogOutput.textContent", app)
        self.assertIn("current 3D view is preview", app)
        self.assertIn("smallestFeatureM:", app)
        self.assertIn("unitScale: effectiveUnitScale()", app)
        self.assertIn("state.repair.warnings", app)
        self.assertIn("Boundary-layer coverage", app)
        self.assertIn("residual-controlled convergence", app)
        self.assertIn('id="autoAlignButton"', index)
        self.assertIn("alignment_suggestion", app)
        self.assertIn("/api/analyze-features", app)
        self.assertIn("state.caseReport?.geometryModelPath", app)
        self.assertIn('analyzingCaseGeometry ? "+x"', app)
        self.assertIn("Diffuser candidate", app)
        self.assertIn("forces.verticalForceType", app)
        self.assertIn("verticalForceLbf", app)
        self.assertIn("STL coordinates calculate", app)
        self.assertIn("Measured length m", index)

    def test_particle_trace_controls_and_rendering_are_wired(self) -> None:
        project = Path(__file__).resolve().parents[1]
        index = (project / "src" / "aerolab" / "web" / "index.html").read_text(encoding="utf-8")
        app = (project / "src" / "aerolab" / "web" / "app.js").read_text(encoding="utf-8")

        for control_id in (
            "solverFlowStatus",
            "solverLinesButton",
            "solverParticlesButton",
            "solverBothButton",
            "solverParticleSettings",
            "solverParticleMotionButton",
            "solverParticleRateSelect",
        ):
            with self.subTest(control_id=control_id):
                self.assertIn(f'id="{control_id}"', index)

        for function_name in (
            "prepareSolverParticles",
            "updateSolverParticles",
            "resetSolverParticles",
            "setSolverFlowMode",
            "ensureSolverParticlePoints",
            "syncSolverFlowControls",
            "restoreSolverParticleSettings",
            "saveSolverParticleSettings",
            "setSolverParticlePaused",
            "setSolverParticleRate",
        ):
            with self.subTest(function_name=function_name):
                self.assertIn(f"function {function_name}(", app)

        for label in ("Mean-flow speed", "Final-field speed"):
            with self.subTest(label=label):
                self.assertIn(label, app)

        for button_name, mode in (
            ("solverLinesButton", "lines"),
            ("solverParticlesButton", "particles"),
            ("solverBothButton", "both"),
        ):
            with self.subTest(mode=mode):
                self.assertIn(f'[els.{button_name}, "{mode}"]', app)
        self.assertIn('button.setAttribute("aria-pressed", String(active))', app)
        self.assertIn('"aerolab-particle-settings-v1"', app)
        self.assertIn('"(prefers-reduced-motion: reduce)"', app)
        self.assertIn("SOLVER_PARTICLE_ANIMATION_RATE * settings.rateMultiplier", app)
        self.assertIn("particleAvailable && state.viewer.solverFlowMode !== \"lines\"", app)
        for rate_value in ("0.5", "1", "2"):
            with self.subTest(rate_value=rate_value):
                self.assertIn(f'<option value="{rate_value}"', index)

    def test_creates_thermal_case_with_heat_zone_through_api(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            model_path = models / "sample_box.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/cases",
                    data=json.dumps(
                        {
                            "modelPath": str(model_path),
                            "name": "api-thermal-case",
                            "speedMph": 70,
                            "flowAxis": "x",
                            "fluidProfile": "compressible_thermal",
                            "heatZones": [
                                {
                                    "name": "radiatorReject",
                                    "shape": "box",
                                    "component": "radiator coolant",
                                    "minimum_m": [0.2, 0.2, 0.2],
                                    "maximum_m": [0.6, 0.6, 0.6],
                                    "power_kw": 5,
                                }
                            ],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            case_path = Path(result["casePath"])
            heat_zones = result["case"]["physical_model"]["volume_zones"]["heat_zones"]
            thermal = result["case"]["physical_model"]["thermal"]
            fv_models = (case_path / "constant" / "fvModels").read_text(encoding="utf-8")
            body_pressure = (case_path / "system" / "bodyPressure").read_text(encoding="utf-8")

            self.assertTrue(result["ok"])
            self.assertEqual(heat_zones[0]["power_w"], 5_000.0)
            self.assertEqual(thermal["total_power_w"], 5_000.0)
            self.assertEqual(thermal["model"], "direct_air_volumetric_heat_source")
            self.assertIn("constant/fvModels", result["files"])
            self.assertIn("type heatSource;", fv_models)
            self.assertIn("cellZone radiatorReject;", fv_models)
            self.assertIn("Q 5000;", fv_models)
            self.assertIn("fields (p wallShearStress T);", body_pressure)

    def test_local_optimization_controls_and_study_wiring_are_present(self) -> None:
        project = Path(__file__).resolve().parents[1]
        index = (project / "src" / "aerolab" / "web" / "index.html").read_text(
            encoding="utf-8"
        )
        app = (project / "src" / "aerolab" / "web" / "app.js").read_text(
            encoding="utf-8"
        )

        for control_id in (
            "solverProcesses",
            "solverFileHandler",
            "studyProcessBudget",
            "resumeSolver",
            "runStudyButton",
            "retryBudgetButton",
            "optimizationStatus",
        ):
            with self.subTest(control_id=control_id):
                self.assertIn(f'id="{control_id}"', index)
        self.assertIn('src="/assets/app.js?v=72"', index)
        self.assertIn("options.processes ?? els.solverProcesses.value", app)
        self.assertIn('fileHandler: els.solverFileHandler.value', app)
        self.assertIn('resume: attempt === 0 ? originalResume : false', app)
        self.assertIn("options.processBudget ?? els.studyProcessBudget.value", app)
        self.assertIn('fetchJson("/api/run-study"', app)
        self.assertIn('/api/study-progress?casePath=', app)
        self.assertIn('function renderOptimizationStatus()', app)
        self.assertIn('function activeBudgetRecommendation()', app)
        self.assertIn('function retryRecommendedBudget()', app)
        self.assertIn('function runActiveCase(mode, options = {})', app)
        self.assertIn('function runActiveStudy(options = {})', app)
        self.assertIn('while (attempt < 2)', app)
        self.assertIn('recommendation?.autoRetrySafe === true', app)
        self.assertIn('minimumUpdatedAt = 0', app)
        self.assertIn('const staleTerminal =', app)
        self.assertIn('recommendedProcessBudget', app)
        self.assertIn('Retry with safer budget', index)

    def test_run_case_api_normalizes_and_propagates_optimization_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_path = root / "cases" / "api-optimization"
            case_path.mkdir(parents=True)
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.object(server, "start_case_run") as start_case_run:
                    request = Request(
                        f"http://127.0.0.1:{server.server_port}/api/run-case",
                        data=json.dumps(
                            {
                                "casePath": str(case_path),
                                "backend": "docker",
                                "mode": "full",
                                "timeoutSeconds": 1234,
                                "reuseMesh": False,
                                "processes": "4",
                                "fileHandler": "masteruncollated",
                                "resume": True,
                            }
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        self.assertEqual(response.status, 202)
                        payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            start_case_run.assert_called_once_with(
                case_path.resolve(),
                "docker",
                1234,
                "full",
                False,
                4,
                "masterUncollated",
                True,
            )
            self.assertEqual(payload["processes"], 4)
            self.assertEqual(payload["fileHandler"], "masterUncollated")
            self.assertTrue(payload["resume"])

    def test_study_progress_endpoint_returns_the_shared_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_path = root / "cases" / "study-progress"
            case_path.mkdir(parents=True)
            case_path.joinpath("case.json").write_text(
                json.dumps({"name": "study-progress"}),
                encoding="utf-8",
            )
            expected = {
                "status": "running",
                "completedCases": 1,
                "totalCases": 3,
                "percent": 33,
            }
            case_path.joinpath("aerolab-study-run.json").write_text(
                json.dumps(expected),
                encoding="utf-8",
            )
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = (
                    f"http://127.0.0.1:{server.server_port}/api/study-progress"
                    f"?casePath={quote(str(case_path))}"
                )
                with urlopen(url, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["exists"])
            self.assertEqual(payload["studyRun"], expected)

    def test_study_run_reserves_every_member_against_duplicate_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "cases" / "study-first"
            second = root / "cases" / "study-second"
            first.mkdir(parents=True)
            second.mkdir()
            descriptor = {
                "studyId": "reserved-study",
                "kind": "grid_convergence",
                "casePaths": [str(first), str(second)],
            }
            server = AeroLabServer(("127.0.0.1", 0), root)
            started = threading.Event()
            release = threading.Event()

            def fake_run_study(*args: object, **kwargs: object) -> None:
                started.set()
                release.wait(timeout=5)

            try:
                with (
                    patch("aerolab.webapp.study_members", return_value=descriptor),
                    patch(
                        "aerolab.webapp.run_study",
                        side_effect=fake_run_study,
                    ) as mocked_run,
                ):
                    result = server.start_study_run(
                        first,
                        "docker",
                        3600,
                        "full",
                        True,
                        "auto",
                        6,
                        "auto",
                    )
                    self.assertEqual(result, descriptor)
                    self.assertTrue(started.wait(timeout=2))
                    with server.active_runs_lock:
                        worker = server.active_runs[first.resolve()]
                        self.assertIs(worker, server.active_runs[second.resolve()])
                    with self.assertRaisesRegex(ValueError, "already has an active"):
                        server.start_case_run(
                            second,
                            "docker",
                            3600,
                            "full",
                            True,
                            "auto",
                            "auto",
                            False,
                        )
                    release.set()
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive())
                    mocked_run.assert_called_once_with(
                        first.resolve(),
                        backend="docker",
                        timeout_seconds=3600,
                        run_mode="full",
                        reuse_mesh=True,
                        processes="auto",
                        process_budget=6,
                        file_handler="auto",
                    )
                    with server.active_runs_lock:
                        self.assertNotIn(first.resolve(), server.active_runs)
                        self.assertNotIn(second.resolve(), server.active_runs)
            finally:
                release.set()
                server.server_close()

    def test_case_runs_start_in_background_and_reject_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_path = root / "cases" / "background-case"
            case_path.mkdir(parents=True)
            server = AeroLabServer(("127.0.0.1", 0), root)
            started = threading.Event()
            release = threading.Event()

            def fake_run_case(*args: object, **kwargs: object) -> None:
                started.set()
                release.wait(timeout=5)

            try:
                with patch("aerolab.webapp.run_case", side_effect=fake_run_case) as mocked_run:
                    server.start_case_run(
                        case_path,
                        "wsl",
                        14400,
                        "mesh",
                        True,
                        "auto",
                        "auto",
                        False,
                    )
                    self.assertTrue(started.wait(timeout=2))
                    with self.assertRaisesRegex(ValueError, "already has an active"):
                        server.start_case_run(
                            case_path,
                            "wsl",
                            14400,
                            "mesh",
                            True,
                            "auto",
                            "auto",
                            False,
                        )
                    with server.active_runs_lock:
                        worker = server.active_runs[case_path.resolve()]
                    release.set()
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive())
                    mocked_run.assert_called_once_with(
                        case_path.resolve(),
                        backend="wsl",
                        timeout_seconds=14400,
                        run_mode="mesh",
                        reuse_mesh=True,
                        processes="auto",
                        file_handler="auto",
                        resume=False,
                    )
            finally:
                release.set()
                server.server_close()

    def test_repair_feature_size_is_converted_to_source_stl_units(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            model_path = models / "sample_box.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            fake_result = SimpleNamespace(
                accepted=False,
                output_report=inspect_stl(model_path),
                to_dict=lambda: {"accepted": False},
            )
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/repair-model",
                    data=json.dumps(
                        {
                            "modelPath": str(model_path),
                            "resolution": 384,
                            "smallestFeatureM": 0.01,
                            "unitScale": 0.001,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with patch("aerolab.webapp.repair_stl", return_value=fake_result) as repair:
                    with urlopen(request, timeout=10) as response:
                        payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(payload["ok"])
            self.assertFalse(payload["accepted"])
            self.assertAlmostEqual(repair.call_args.kwargs["smallest_feature_source_units"], 10.0)

    def test_case_progress_endpoint_is_lightweight_and_pollable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_path = root / "cases" / "running-case"
            case_path.mkdir(parents=True)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "name": "running-case",
                        "status": "solver_running",
                        "cfd_quality": {"end_time": 100},
                    }
                ),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps({"status": "running", "returncode": None}),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.log").write_text(
                "=== AEROLAB STEP: foamRun ===\nTime = 50s\n",
                encoding="utf-8",
            )
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = (
                    f"http://127.0.0.1:{server.server_port}/api/case-progress?"
                    f"casePath={quote(str(case_path), safe='')}"
                )
                with urlopen(url, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["progress"]["phase"], "Solving airflow")
            self.assertEqual(payload["progress"]["percent"], 75)

    def test_case_log_endpoint_returns_only_a_bounded_case_local_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_dir = root / "cases"
            case_path = cases_dir / "running-case"
            case_path.mkdir(parents=True)
            case_path.joinpath("case.json").write_text(
                json.dumps({"name": "running-case", "status": "mesh_running"}),
                encoding="utf-8",
            )
            log_text = "".join(
                f"old-line-{index:05d} {'x' * 24}\n"
                for index in range(3_000)
            ) + "Shell refinement iteration 9\nAfter refinement cells: 1234567\n"
            log_path = case_path / "aerolab-run.log"
            # newline="" disables text-mode newline translation so the on-disk byte
            # count matches log_text on Windows as well as POSIX.
            log_path.write_text(log_text, encoding="utf-8", newline="")

            missing_case = cases_dir / "not-run"
            missing_case.mkdir()
            missing_case.joinpath("case.json").write_text(
                json.dumps({"name": "not-run", "status": "created"}),
                encoding="utf-8",
            )
            non_case_root = root / "other" / "fake-case"
            non_case_root.mkdir(parents=True)
            non_case_root.joinpath("case.json").write_text("{}", encoding="utf-8")
            non_case_root.joinpath("aerolab-run.log").write_text(
                "must-not-be-readable",
                encoding="utf-8",
            )

            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}/api/case-log?casePath="
                with urlopen(base_url + quote(str(case_path), safe=""), timeout=10) as response:
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    payload = json.loads(response.read().decode("utf-8"))

                with urlopen(base_url + quote(str(missing_case), safe=""), timeout=10) as response:
                    missing_payload = json.loads(response.read().decode("utf-8"))

                with self.assertRaises(HTTPError) as context:
                    urlopen(base_url + quote(str(non_case_root), safe=""), timeout=10)
                self.assertEqual(context.exception.code, 400)
                self.assertNotIn("must-not-be-readable", context.exception.read().decode("utf-8"))

                secret_path = root / "private.log"
                secret_path.write_text("top-secret-log-content", encoding="utf-8")
                log_path.unlink()
                try:
                    log_path.symlink_to(secret_path)
                except OSError:
                    pass
                else:
                    with self.assertRaises(HTTPError) as symlink_context:
                        urlopen(base_url + quote(str(case_path), safe=""), timeout=10)
                    self.assertEqual(symlink_context.exception.code, 400)
                    self.assertNotIn(
                        "top-secret-log-content",
                        symlink_context.exception.read().decode("utf-8"),
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(payload["ok"])
            self.assertTrue(payload["exists"])
            self.assertTrue(payload["truncated"])
            self.assertLessEqual(payload["shownBytes"], 64 * 1024)
            self.assertEqual(payload["sizeBytes"], len(log_text.encode("utf-8")))
            self.assertNotIn("old-line-00000", payload["text"])
            self.assertIn("Shell refinement iteration 9", payload["text"])
            self.assertIn("After refinement cells: 1234567", payload["text"])
            self.assertIsNotNone(payload["modifiedAt"])
            self.assertTrue(missing_payload["ok"])
            self.assertFalse(missing_payload["exists"])
            self.assertEqual(missing_payload["text"], "")

    def test_serves_only_stl_files_inside_the_project(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            model_path = models / "sample_box.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            outside_path = project / "models" / "sample_box.stl"

            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                model_url = (
                    f"http://127.0.0.1:{server.server_port}/api/model-file?"
                    f"path={quote(str(model_path), safe='')}"
                )
                with urlopen(model_url, timeout=10) as response:
                    self.assertEqual(response.headers.get_content_type(), "model/stl")
                    self.assertEqual(response.read(), model_path.read_bytes())

                outside_url = (
                    f"http://127.0.0.1:{server.server_port}/api/model-file?"
                    f"path={quote(str(outside_path), safe='')}"
                )
                with self.assertRaises(HTTPError) as context:
                    urlopen(outside_url, timeout=10)
                self.assertEqual(context.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_creates_grouped_draft_standard_and_fine_cases(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            model_path = models / "sample_box.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            rotation = (2.0, -3.0, 4.0)
            transformed = transformed_report(model_path, scale=1.0, rotation_degrees=rotation)
            length, width, height = transformed.bounds.dimensions

            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                payload = {
                    "modelPath": str(model_path),
                    "name": "box-study",
                    "speedMph": 70,
                    "flowAxis": "x",
                    "includeGround": True,
                    "movingGround": True,
                    "groundClearanceM": 0.075,
                    "unitScale": 1.0,
                    "unitLabel": "m",
                    "sourceFlowDirection": "+x",
                    "sourceUpDirection": "+z",
                    "modelRotationDegrees": {"x": rotation[0], "y": rotation[1], "z": rotation[2]},
                    "measuredLengthM": length,
                    "measuredWidthM": width,
                    "measuredHeightM": height,
                    "smallestAeroFeatureM": 0.25,
                }
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/accuracy-study",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertTrue(result["ok"])
            self.assertEqual(len(result["casePaths"]), 3)
            self.assertEqual(result["report"]["gridConvergence"]["status"], "incomplete")
            self.assertTrue(result["report"]["geometryReport"]["is_cfd_candidate"])
            self.assertEqual(result["report"]["caseSetup"]["flow"]["speed_mph"], 70.0)
            self.assertEqual(result["report"]["caseSetup"]["ground"]["clearance_m"], 0.075)
            self.assertTrue(result["report"]["caseSetup"]["placement"]["verified"])
            self.assertEqual(
                result["report"]["caseSetup"]["orientation"]["rotation_degrees"],
                {"x": 2.0, "y": -3.0, "z": 4.0},
            )
            self.assertEqual(
                [item["studyLevel"] for item in reversed(result["state"]["cases"])],
                ["draft", "standard", "fine"],
            )
            study_ids = {
                json.loads((Path(path) / "case.json").read_text(encoding="utf-8"))["validation_study"]["id"]
                for path in result["casePaths"]
            }
            self.assertEqual(study_ids, {result["studyId"]})
            wall_setups = []
            for path in result["casePaths"]:
                case_payload = json.loads((Path(path) / "case.json").read_text(encoding="utf-8"))
                wall = case_payload["wall_resolution"]
                wall_setups.append((wall["target_y_plus"], wall["surface_layers"], wall["expansion_ratio"]))
                self.assertIn("nSurfaceLayers 5;", (Path(path) / "system" / "snappyHexMeshDict").read_text(encoding="utf-8"))
                self.assertIn("foamPostProcess -solver incompressibleFluid -func yPlus", (Path(path) / "Allrun").read_text(encoding="utf-8"))
            self.assertEqual(set(wall_setups), {(60.0, 5, 1.2)})

    def test_accuracy_study_requires_all_measured_dimensions(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            model_path = models / "sample_box.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            server = AeroLabServer(("127.0.0.1", 0), root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/accuracy-study",
                    data=json.dumps({"modelPath": str(model_path), "name": "missing-measurements"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=10)
                self.assertEqual(context.exception.code, 400)
                self.assertIn("measured vehicle length", context.exception.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertFalse((root / "cases").exists())


if __name__ == "__main__":
    unittest.main()
