from __future__ import annotations

import base64
import json
import math
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path

from aerolab.solver import (
    CASE_PREVIEW_TRIANGLE_LIMIT,
    OPENFOAM_BOOTSTRAP,
    _clear_previous_solver_outputs,
    _mesh_input_fingerprint,
    _mesh_record_reusable,
    _run_command,
    assess_meshed_surface_fidelity,
    case_report,
    case_run_progress,
    parse_check_mesh,
    parse_force_coeffs,
    parse_layer_coverage,
    parse_residuals,
    parse_streamlines,
    parse_surface_pressure,
    parse_temperature_results,
    parse_transient_state,
    parse_y_plus,
)
from aerolab.stl import write_binary_stl_triangles


class SolverQualityTests(unittest.TestCase):
    def test_backend_resource_probe_applies_cpu_memory_and_mpi_limits(self) -> None:
        from aerolab.solver.backends import _parse_resource_probe

        gib = 1024**3
        output = "\n".join(
            (
                "AEROLAB_RESOURCE_LOGICAL_CPUS=12",
                "AEROLAB_RESOURCE_CPUSET=0-5",
                "AEROLAB_RESOURCE_CPU_LIMIT=400000 100000",
                "AEROLAB_RESOURCE_CPU_PERIOD=100000",
                f"AEROLAB_RESOURCE_MEM_AVAILABLE_KIB={32 * 1024**2}",
                f"AEROLAB_RESOURCE_MEMORY_MAX={16 * gib}",
                f"AEROLAB_RESOURCE_MEMORY_CURRENT={4 * gib}",
                "AEROLAB_PARALLEL_TOOL_mpirun=/usr/bin/mpirun",
                "AEROLAB_PARALLEL_TOOL_decomposePar=/opt/openfoam13/bin/decomposePar",
                "AEROLAB_PARALLEL_TOOL_reconstructParMesh=/opt/openfoam13/bin/reconstructParMesh",
                "AEROLAB_PARALLEL_TOOL_reconstructPar=/opt/openfoam13/bin/reconstructPar",
            )
        )

        resources = _parse_resource_probe(output, "docker")

        self.assertEqual(resources["logicalCpus"], 12)
        self.assertEqual(resources["cpusetCpus"], 6)
        self.assertEqual(resources["quotaCpus"], 4)
        self.assertEqual(resources["effectiveCpus"], 4)
        self.assertEqual(resources["memoryAvailableBytes"], 12 * gib)
        self.assertTrue(resources["parallelAvailable"])
        self.assertEqual(resources["missingParallelTools"], [])

    def test_auto_process_selection_respects_case_and_backend_caps(self) -> None:
        from unittest import mock

        from aerolab.solver.run import _resolve_processes

        gib = 1024**3
        resources = {
            "effectiveCpus": 12,
            "memoryAvailableBytes": 32 * gib,
            "parallelAvailable": True,
            "missingParallelTools": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "cfd_quality": {"name": "standard"},
                        "mesh_resolution": {
                            "configured_max_global_cells": 1_500_000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "aerolab.solver.run.probe_backend_resources",
                return_value=resources,
            ):
                resolved, selection = _resolve_processes(
                    case_path,
                    "auto",
                    "docker",
                    parallel_script=True,
                    solver_identity=None,
                )
                with self.assertRaisesRegex(ValueError, "exposes 12 CPUs"):
                    _resolve_processes(
                        case_path,
                        13,
                        "docker",
                        parallel_script=True,
                        solver_identity=None,
                    )

        self.assertEqual(resolved, 6)
        self.assertEqual(selection["autoCaps"]["cellBudget"], 6)
        self.assertEqual(selection["qualityRecommendation"]["status"], "comfortable")

    def test_oom_recommendation_allows_one_same_fidelity_auto_retry(self) -> None:
        from aerolab.solver.run import (
            _failure_budget_recommendation,
            _safe_cell_budget,
            _suggested_quality,
        )

        gib = 1024**3
        resources = {
            "effectiveCpus": 8,
            "memoryAvailableBytes": 8 * gib,
            "parallelAvailable": True,
        }
        self.assertEqual(_safe_cell_budget(resources), 2_050_000)
        self.assertIsNone(_suggested_quality(1_500_000))
        self.assertEqual(_suggested_quality(2_050_000), "draft")
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "cfd_quality": {"name": "standard"},
                        "mesh_resolution": {
                            "configured_max_global_cells": 1_500_000,
                        },
                    }
                ),
                encoding="utf-8",
            )

            recommendation = _failure_budget_recommendation(
                case_path,
                returncode=137,
                log_text="OpenFOAM process killed by the backend",
                requested_processes="auto",
                processes=4,
                process_selection={"detectedResources": resources},
            )

        assert recommendation is not None
        self.assertEqual(recommendation["category"], "memory_oom")
        self.assertEqual(recommendation["confidence"], "high")
        self.assertEqual(recommendation["recommendedProcesses"], 2)
        self.assertTrue(recommendation["retryAllowed"])
        self.assertTrue(recommendation["autoRetrySafe"])
        self.assertTrue(recommendation["preservesCaseFidelity"])
        self.assertIsNone(recommendation["suggestedQuality"])

    def test_resource_failure_categories_do_not_guess_at_unsafe_retries(self) -> None:
        from aerolab.solver.run import _failure_budget_recommendation

        gib = 1024**3
        resources = {
            "effectiveCpus": 8,
            "memoryAvailableBytes": 4 * gib,
            "parallelAvailable": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "cfd_quality": {"name": "standard"},
                        "mesh_resolution": {
                            "configured_max_global_cells": 2_800_000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            selection = {"detectedResources": resources}
            single_rank = _failure_budget_recommendation(
                case_path,
                returncode=1,
                log_text="terminate called after throwing std::bad_alloc",
                requested_processes="auto",
                processes=1,
                process_selection=selection,
            )
            slots = _failure_budget_recommendation(
                case_path,
                returncode=1,
                log_text="There are not enough slots available in the system",
                requested_processes="auto",
                processes=8,
                process_selection=selection,
            )
            storage = _failure_budget_recommendation(
                case_path,
                returncode=1,
                log_text="write failed: No space left on device",
                requested_processes="auto",
                processes=4,
                process_selection=selection,
            )
            mesh = _failure_budget_recommendation(
                case_path,
                returncode=124,
                log_text=(
                    "Shell refinement iteration 2\n"
                    "After refinement shell refinement iteration 1 : "
                    "cells:2100000 faces:1 points:1"
                ),
                requested_processes="auto",
                processes=4,
                process_selection=selection,
            )
            timeout = _failure_budget_recommendation(
                case_path,
                returncode=124,
                log_text="=== AEROLAB STEP: foamRun ===\nTime = 25s",
                requested_processes="auto",
                processes=4,
                process_selection=selection,
            )

        assert single_rank is not None
        assert slots is not None
        assert storage is not None
        assert mesh is not None
        assert timeout is not None
        self.assertEqual(single_rank["category"], "memory_oom")
        self.assertFalse(single_rank["retryAllowed"])
        self.assertFalse(single_rank["autoRetrySafe"])
        self.assertEqual(single_rank["safeCellBudget"], 500_000)
        self.assertEqual(slots["category"], "cpu_mpi_oversubscription")
        self.assertEqual(slots["recommendedProcesses"], 4)
        self.assertTrue(slots["retryAllowed"])
        self.assertEqual(storage["category"], "storage_exhaustion")
        self.assertFalse(storage["retryAllowed"])
        self.assertEqual(mesh["category"], "mesh_cell_budget")
        self.assertEqual(mesh["safeCellBudget"], 500_000)
        self.assertEqual(mesh["configuredCellBudget"], 2_800_000)
        self.assertFalse(mesh["retryAllowed"])
        self.assertEqual(timeout["category"], "runtime_timeout")
        self.assertFalse(timeout["retryAllowed"])

    def test_wsl_and_docker_commands_propagate_optimization_settings(self) -> None:
        import os
        from unittest import mock

        feature_key = "a" * 64
        block_key = "b" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir) / "case with spaces"
            case_path.mkdir()
            wsl_command = _run_command(
                case_path,
                "wsl",
                processes=3,
                file_handler="masterUncollated",
                resume=True,
                feature_cache_key=feature_key,
                block_cache_key=block_key,
            )
            wsl_script = base64.b64decode(
                shlex.split(wsl_command[-1])[2]
            ).decode("utf-8")
            self.assertIn("export AEROLAB_PROCESSES=3", wsl_script)
            self.assertIn("export AEROLAB_FILE_HANDLER=masterUncollated", wsl_script)
            self.assertIn("export AEROLAB_RESUME=1", wsl_script)
            self.assertIn(f"export AEROLAB_FEATURE_CACHE_KEY={feature_key}", wsl_script)
            self.assertIn(f"export AEROLAB_BLOCK_CACHE_KEY={block_key}", wsl_script)

            with mock.patch.dict(
                os.environ,
                {"AEROLAB_OPENFOAM_IMAGE": "local/openfoam:13"},
            ):
                docker_command = _run_command(
                    case_path,
                    "docker",
                    processes=4,
                    file_handler="collated",
                    resume=True,
                    feature_cache_key=feature_key,
                    block_cache_key=block_key,
                )

            for setting in (
                "AEROLAB_PROCESSES=4",
                "AEROLAB_FILE_HANDLER=collated",
                "AEROLAB_RESUME=1",
                f"AEROLAB_FEATURE_CACHE_KEY={feature_key}",
                f"AEROLAB_BLOCK_CACHE_KEY={block_key}",
            ):
                self.assertIn(setting, docker_command)
            self.assertIn(f"{case_path}:/source", docker_command)
            self.assertEqual(
                docker_command[docker_command.index("--mount") + 1],
                "type=volume,target=/work",
            )
            self.assertIn('cp -a -- "$SOURCE_CASE/." "$STAGE_CASE/"', docker_command[-1])
            self.assertIn("trap copy_back EXIT", docker_command[-1])

    def test_resume_requires_matching_inputs_and_preserves_valid_state(self) -> None:
        from aerolab.case import create_case
        from aerolab.solver.run import (
            _latest_numeric_time,
            _resume_compatibility,
            _solver_input_fingerprint,
        )

        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="resume-contract",
                speed_mph=70,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                quality="draft",
            )
            poly_mesh = case_path / "constant" / "polyMesh"
            poly_mesh.mkdir(parents=True)
            poly_mesh.joinpath("points").write_text("points", encoding="utf-8")
            mesh_surface = case_path / "postProcessing" / "meshSurface" / "0"
            mesh_surface.mkdir(parents=True)
            mesh_surface.joinpath("body.vtk").write_text("surface", encoding="utf-8")
            mesh_fingerprint = _mesh_input_fingerprint(case_path)
            case_path.joinpath("aerolab-mesh.json").write_text(
                json.dumps(
                    {
                        "reusable": True,
                        "inputFingerprint": mesh_fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            solver_fingerprint = _solver_input_fingerprint(case_path)
            self.assertIsNotNone(solver_fingerprint)
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "mode": "full",
                        "solverInputFingerprint": solver_fingerprint,
                        "processes": 4,
                    }
                ),
                encoding="utf-8",
            )
            for time_name, fields in (("25", ("U", "p")), ("50", ("U",))):
                time_path = case_path / time_name
                time_path.mkdir()
                for field in fields:
                    time_path.joinpath(field).write_text(field, encoding="utf-8")

            self.assertEqual(_latest_numeric_time(case_path), 25.0)
            state = _resume_compatibility(case_path, solver_fingerprint)
            self.assertEqual(state["latestTime"], 25.0)

            case_path.joinpath("processor0").mkdir()
            case_path.joinpath("postProcessing", "forceCoeffs").mkdir()
            _clear_previous_solver_outputs(
                case_path,
                preserve_mesh=True,
                preserve_solver_state=True,
            )
            self.assertTrue(case_path.joinpath("25", "U").is_file())
            self.assertTrue(case_path.joinpath("postProcessing", "forceCoeffs").is_dir())
            self.assertFalse(case_path.joinpath("processor0").exists())

            decomposition = case_path / "system" / "decomposeParDict"
            decomposition.write_text(
                decomposition.read_text(encoding="utf-8") + "\n// changed ranks\n",
                encoding="utf-8",
            )
            self.assertEqual(_solver_input_fingerprint(case_path), solver_fingerprint)
            fv_solution = case_path / "system" / "fvSolution"
            fv_solution.write_text(
                fv_solution.read_text(encoding="utf-8") + "\n// changed numerics\n",
                encoding="utf-8",
            )
            changed_fingerprint = _solver_input_fingerprint(case_path)
            with self.assertRaisesRegex(ValueError, "Solver inputs changed"):
                _resume_compatibility(case_path, changed_fingerprint)

    def test_progress_exposes_optimization_and_study_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "status": "solver_running",
                        "cfd_quality": {"end_time": 100},
                    }
                ),
                encoding="utf-8",
            )
            process_selection = {
                "stageCache": {"featureHit": True, "blockMeshHit": False},
                "qualityRecommendation": {"status": "comfortable"},
            }
            convergence = {"controller": "foundationResidualControl"}
            budget_recommendation = {
                "category": "memory_oom",
                "detail": "Retry unchanged with fewer processes.",
                "retryAllowed": True,
                "recommendedProcesses": 3,
            }
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "mode": "full",
                        "requestedProcesses": "auto",
                        "processes": 6,
                        "fileHandler": "auto",
                        "resumed": False,
                        "resumeFromTime": None,
                        "convergencePolicy": convergence,
                        "processSelection": process_selection,
                        "budgetRecommendation": budget_recommendation,
                    }
                ),
                encoding="utf-8",
            )
            study_run = {"status": "running", "completedCases": 1, "totalCases": 3}
            case_path.joinpath("aerolab-study-run.json").write_text(
                json.dumps(study_run),
                encoding="utf-8",
            )

            progress = case_run_progress(case_path)

            self.assertEqual(progress["optimization"]["requestedProcesses"], "auto")
            self.assertEqual(progress["optimization"]["processes"], 6)
            self.assertEqual(
                progress["optimization"]["convergencePolicy"],
                convergence,
            )
            self.assertEqual(
                progress["optimization"]["processSelection"],
                process_selection,
            )
            self.assertEqual(progress["budgetRecommendation"], budget_recommendation)
            self.assertEqual(
                progress["optimization"]["budgetRecommendation"],
                budget_recommendation,
            )
            self.assertEqual(progress["studyRun"], study_run)

    def test_transient_force_and_residual_gates_accept_stable_oscillation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            quality = {
                "simulation_mode": "transient",
                "end_time": 10.0,
                "averaging_window_s": 2.0,
                "minimum_force_samples": 100,
                "maximum_courant_number": 1.5,
                "transient_residual_ceiling": 0.2,
            }
            case_path.joinpath("case.json").write_text(
                json.dumps({"cfd_quality": quality}), encoding="utf-8"
            )
            coeff_dir = case_path / "postProcessing" / "forceCoeffs" / "0"
            coeff_dir.mkdir(parents=True)
            rows = ["# Time Cd Cs Cl CmRoll CmPitch CmYaw"]
            for index in range(1001):
                time_value = index / 100.0
                phase = 2.0 * math.pi * 10.0 * time_value
                rows.append(
                    f"{time_value:.2f} {0.3 + 0.03 * math.sin(phase):.8f} 0 "
                    f"{0.04 * math.cos(phase):.8f} 0 0 0"
                )
            coeff_dir.joinpath("coefficient.dat").write_text("\n".join(rows), encoding="utf-8")

            log_lines = ["=== AEROLAB STEP: foamRun ==="]
            for index in range(25):
                log_lines.extend(
                    (
                        f"Time = {9.76 + index * 0.01:.2f}s",
                        "Courant Number mean: 0.3 max: 1.1",
                        "Solving for Ux, Initial residual = 0.05, Final residual = 0.001",
                        "Solving for p, Initial residual = 0.04, Final residual = 0.001",
                        "Solving for k, Initial residual = 0.03, Final residual = 0.001",
                        "Solving for omega, Initial residual = 0.02, Final residual = 0.001",
                    )
                )
            case_path.joinpath("aerolab-run.log").write_text("\n".join(log_lines), encoding="utf-8")
            final_time = case_path / "10"
            final_time.mkdir()
            final_time.joinpath("UMean").write_text("mean velocity", encoding="utf-8")
            final_time.joinpath("pMean").write_text("mean pressure", encoding="utf-8")

            force = parse_force_coeffs(case_path)
            residuals = parse_residuals(case_path, quality)
            transient = parse_transient_state(case_path, quality)

            self.assertTrue(force["stable"])
            self.assertEqual(force["averagingMode"], "time-window")
            self.assertGreater(force["statistics"]["Cd"]["relativeRange"], 0.1)
            self.assertTrue(residuals["stable"])
            self.assertEqual(residuals["mode"], "transient-divergence")
            self.assertEqual(transient["status"], "pass")

    def test_rerun_clears_only_generated_solver_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("0").mkdir()
            case_path.joinpath("0", "U").write_text("initial", encoding="utf-8")
            case_path.joinpath("500").mkdir()
            case_path.joinpath("500", "U").write_text("stale", encoding="utf-8")
            case_path.joinpath("postProcessing", "forceCoeffs").mkdir(parents=True)
            case_path.joinpath("constant", "polyMesh").mkdir(parents=True)
            case_path.joinpath("constant", "geometry").mkdir(parents=True)
            case_path.joinpath("mesh-surface-fidelity.json").write_text("{}", encoding="utf-8")

            _clear_previous_solver_outputs(case_path)

            self.assertTrue(case_path.joinpath("0", "U").is_file())
            self.assertTrue(case_path.joinpath("constant", "geometry").is_dir())
            self.assertFalse(case_path.joinpath("500").exists())
            self.assertFalse(case_path.joinpath("postProcessing").exists())
            self.assertFalse(case_path.joinpath("constant", "polyMesh").exists())
            self.assertFalse(case_path.joinpath("mesh-surface-fidelity.json").exists())

    def test_solver_only_run_preserves_validated_mesh_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("0").mkdir()
            case_path.joinpath("500").mkdir()
            case_path.joinpath("postProcessing", "meshSurface", "0").mkdir(parents=True)
            case_path.joinpath("postProcessing", "meshSurface", "0", "body.vtk").write_text(
                "mesh",
                encoding="utf-8",
            )
            case_path.joinpath("postProcessing", "forceCoeffs").mkdir(parents=True)
            case_path.joinpath("constant", "polyMesh").mkdir(parents=True)
            case_path.joinpath("constant", "polyMesh", "points").write_text("points", encoding="utf-8")
            case_path.joinpath("aerolab-mesh.json").write_text("{}", encoding="utf-8")
            case_path.joinpath("mesh-surface-fidelity.json").write_text("{}", encoding="utf-8")

            _clear_previous_solver_outputs(case_path, preserve_mesh=True)

            self.assertTrue(case_path.joinpath("constant", "polyMesh", "points").is_file())
            self.assertTrue(case_path.joinpath("postProcessing", "meshSurface", "0", "body.vtk").is_file())
            self.assertTrue(case_path.joinpath("aerolab-mesh.json").is_file())
            self.assertTrue(case_path.joinpath("mesh-surface-fidelity.json").is_file())
            self.assertFalse(case_path.joinpath("postProcessing", "forceCoeffs").exists())
            self.assertFalse(case_path.joinpath("500").exists())

    def test_mesh_reuse_is_bound_to_vehicle_and_mesh_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            for relative_path in (
                "constant/geometry/body.stl",
                "system/blockMeshDict",
                "system/snappyHexMeshDict",
                "system/surfaceFeaturesDict",
            ):
                path = case_path / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path, encoding="utf-8")
            case_path.joinpath("constant", "polyMesh").mkdir(parents=True)
            case_path.joinpath("constant", "polyMesh", "points").write_text("points", encoding="utf-8")
            mesh_surface = case_path / "postProcessing" / "meshSurface" / "0"
            mesh_surface.mkdir(parents=True)
            mesh_surface.joinpath("body.vtk").write_text("surface", encoding="utf-8")
            fingerprint = _mesh_input_fingerprint(case_path)
            case_path.joinpath("aerolab-mesh.json").write_text(
                json.dumps({"reusable": True, "inputFingerprint": fingerprint}),
                encoding="utf-8",
            )

            self.assertTrue(_mesh_record_reusable(case_path))
            case_path.joinpath("system", "snappyHexMeshDict").write_text("changed", encoding="utf-8")
            self.assertFalse(_mesh_record_reusable(case_path))

    def test_y_plus_uses_body_distribution_instead_of_a_single_corner_spike(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("aerolab-run.log").write_text(
                "=== AEROLAB STEP: yPlus ===\n"
                "patch body y+ : min = 80, max = 1000, average = 89.1\n",
                encoding="utf-8",
            )
            time_path = case_path / "1200"
            time_path.mkdir()
            values = "\n".join(["80"] * 100 + ["1000"])
            time_path.joinpath("yPlus").write_text(
                "boundaryField\n{\n"
                "    body\n    {\n        type calculated;\n"
                "        value nonuniform List<scalar>\n101\n(\n"
                f"{values}\n)\n;\n    }}\n}}\n",
                encoding="utf-8",
            )

            result = parse_y_plus(
                case_path,
                {"target_y_plus": 80, "estimated_y_plus": 12},
            )

            self.assertIsNotNone(result)
            self.assertTrue(result["passed"])
            self.assertEqual(result["target"], 80)
            self.assertEqual(result["body"]["p95"], 80)

    def test_case_geometry_preview_is_cached_for_large_scan_navigation(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            geometry_dir = case_path / "constant" / "geometry"
            geometry_dir.mkdir(parents=True)
            shutil.copyfile(project / "models" / "sample_box.stl", geometry_dir / "body.stl")
            case_path.joinpath("case.json").write_text(
                json.dumps({"flow": {"axis": "x", "speed_mps": 20.0}}),
                encoding="utf-8",
            )

            first = case_report(case_path, include_visualization=True)
            cache_path = case_path / "geometry-preview.json"
            first_mtime = cache_path.stat().st_mtime_ns
            second = case_report(case_path, include_visualization=True)

            self.assertTrue(cache_path.is_file())
            self.assertEqual(cache_path.stat().st_mtime_ns, first_mtime)
            self.assertLessEqual(first["geometryPreview"]["sampledTriangleCount"], CASE_PREVIEW_TRIANGLE_LIMIT)
            self.assertEqual(
                first["geometryPreview"]["sampledTriangleCount"],
                second["geometryPreview"]["sampledTriangleCount"],
            )
            self.assertEqual(first["geometryPreview"]["triangles"], second["geometryPreview"]["triangles"])

    def test_meshed_body_fidelity_compares_actual_patch_to_solver_stl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            geometry_dir = case_path / "constant" / "geometry"
            geometry_dir.mkdir(parents=True)
            triangles = [
                ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ]
            write_binary_stl_triangles(geometry_dir / "body.stl", triangles)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "flow": {"axis": "x"},
                        "mesh_resolution": {
                            "estimated_surface_cell_m": 0.1,
                            "smallest_aero_feature_m": 0.5,
                            "estimated_cells_across_feature": 5.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            vtk_dir = case_path / "postProcessing" / "bodyPressure" / "100"
            vtk_dir.mkdir(parents=True)

            def write_vtk(x_offset: float) -> None:
                vtk_dir.joinpath("body.vtk").write_text(
                    f"""# vtk DataFile Version 2.0
body patch
ASCII
DATASET POLYDATA
POINTS 4 float
{x_offset} 0 0  {1 + x_offset} 0 0  {x_offset} 1 0  {x_offset} 0 1
POLYGONS 4 16
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
""",
                    encoding="utf-8",
                )

            write_vtk(0.0)
            matching = assess_meshed_surface_fidelity(case_path, sample_count=300)
            write_vtk(0.2)
            shifted = assess_meshed_surface_fidelity(case_path, sample_count=300)

            self.assertTrue(matching["verified"])
            self.assertAlmostEqual(matching["symmetricP99M"], 0.0, places=10)
            self.assertFalse(shifted["verified"])
            self.assertGreater(shifted["symmetricP95M"], shifted["maximumP95M"])

    def test_legacy_prepared_case_is_marked_fidelity_unverified(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = root / "models" / "prepared"
            prepared.mkdir(parents=True)
            model_path = prepared / "legacy-prepared.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)
            case_path = root / "case"
            case_path.mkdir()
            case_path.joinpath("case.json").write_text(
                json.dumps({"name": "legacy", "model": str(model_path)}),
                encoding="utf-8",
            )

            report = case_report(case_path)

            self.assertEqual(report["geometryFidelity"]["status"], "missing")
            self.assertFalse(report["geometryFidelity"]["verified"])

    def test_extended_concavity_is_a_warning_after_baseline_mesh_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            (case_path / "aerolab-run.log").write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: checkMesh ===",
                        "cells: 29254",
                        "Max aspect ratio = 2.9 OK.",
                        "Mesh non-orthogonality Max: 37 average: 8",
                        "Max skewness = 0.53 OK.",
                        "Mesh OK.",
                        "=== AEROLAB STEP: checkMeshDiagnostics ===",
                        "***Concave cells (using face planes) found, number of cells: 1939",
                        "Failed 1 mesh checks.",
                        "=== AEROLAB STEP: potentialFoam ===",
                    ]
                ),
                encoding="utf-8",
            )

            mesh = parse_check_mesh(case_path)

            self.assertIsNotNone(mesh)
            self.assertTrue(mesh["passed"])
            self.assertEqual(mesh["diagnosticFailedChecks"], 1)
            self.assertIn("1939 concave cut cells", mesh["warnings"][0])

    def test_boundary_layer_coverage_requires_complete_face_stacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            log_path = case_path / "aerolab-run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "Extruding 605 out of 3616 faces (16.7%).",
                        "Added 14309 out of 18080 cells (79.1%).",
                        "Extruding 363 out of 3616 faces (10.0%).",
                        "Added 612 out of 18080 cells (3.4%).",
                        "=== AEROLAB STEP: checkMesh ===",
                    ]
                ),
                encoding="utf-8",
            )

            partial = parse_layer_coverage(case_path, {"surface_layers": 5})

            self.assertIsNotNone(partial)
            self.assertFalse(partial["passed"])
            self.assertAlmostEqual(partial["fullLayerFaceCoveragePercent"], 26.7699, places=3)
            self.assertAlmostEqual(partial["layerCellCoveragePercent"], 82.5277, places=3)

            log_path.write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "Extruding 50002 out of 50002 faces (100%).",
                        "Added 250010 out of 250010 cells (100%).",
                        "=== AEROLAB STEP: checkMesh ===",
                    ]
                ),
                encoding="utf-8",
            )
            complete = parse_layer_coverage(case_path, {"surface_layers": 5})
            self.assertTrue(complete["passed"])
            self.assertEqual(complete["fullLayerFaceCoveragePercent"], 100.0)
            self.assertEqual(complete["layerCellCoveragePercent"], 100.0)

            log_path.write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "Extruding 10 out of 10 faces (100%).",
                        "Added 50 out of 50 cells (100%).",
                        "Snapped mesh : cells:100 faces:300 points:200",
                        "patch faces    layers   overall thickness",
                        "                       [m]       [%]",
                        "----- -----    ------   ---       ---",
                        "body 10       0.3      0.001     10",
                        "Layer mesh : cells:103 faces:309 points:203",
                        "=== AEROLAB STEP: checkMesh ===",
                    ]
                ),
                encoding="utf-8",
            )
            final_summary = parse_layer_coverage(case_path, {"surface_layers": 5})
            self.assertFalse(final_summary["passed"])
            self.assertEqual(final_summary["averageLayers"], 0.3)
            self.assertEqual(final_summary["layerCellCoveragePercent"], 6.0)
            self.assertEqual(final_summary["coverageMethod"], "final snappyHexMesh layer summary and actual cell-count delta")

            log_path.write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "Snapped mesh : cells:100 faces:300 points:200",
                        "patch faces    layers   overall thickness",
                        "                       [m]       [%]",
                        "----- -----    ------   ---       ---",
                        "body 10       3.8      0.005     66",
                        "Layer mesh : cells:138 faces:414 points:250",
                        "=== AEROLAB STEP: checkMesh ===",
                    ]
                ),
                encoding="utf-8",
            )
            usable_partial_stack = parse_layer_coverage(case_path, {"surface_layers": 5})
            self.assertTrue(usable_partial_stack["passed"])
            self.assertEqual(usable_partial_stack["averageLayers"], 3.8)
            self.assertEqual(usable_partial_stack["minimumAverageLayers"], 3.0)
            self.assertAlmostEqual(usable_partial_stack["layerCellCoveragePercent"], 76.0)

    def test_bootstrap_targets_foundation_v13_without_ambiguous_glob_loop(self) -> None:
        self.assertIn("/opt/openfoam13/etc/bashrc", OPENFOAM_BOOTSTRAP)
        self.assertNotIn("for f in", OPENFOAM_BOOTSTRAP)

    def test_wsl_runs_stage_on_linux_filesystem_and_copy_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir) / "case with spaces"
            case_path.mkdir()

            command = _run_command(case_path, "wsl", timeout_seconds=123)
            encoded_script = shlex.split(command[-1])[2]
            script = base64.b64decode(encoded_script).decode("utf-8")

            self.assertEqual(command[:3], ["wsl", "bash", "-lc"])
            self.assertIn("base64 -d | bash", command[-1])
            self.assertIn("$HOME/.cache/aerolab-cfd/runs", script)
            self.assertIn(".aerolab-stage", script)
            self.assertIn('cp -a -- "$SOURCE_CASE/." "$STAGE_CASE/"', script)
            self.assertIn('cp -a -- "$STAGE_CASE/." "$SOURCE_CASE/"', script)
            self.assertEqual(script.count('rm -f -- "$STAGE_CASE/aerolab-run.log" "$STAGE_CASE/aerolab-run.json"'), 2)
            self.assertIn("trap copy_back EXIT", script)
            self.assertIn("timeout --foreground --signal=TERM --kill-after=30s 123s ./Allrun", script)
            self.assertIn("Refusing to replace unmarked WSL staging path", script)
            self.assertNotIn("cd /mnt/", script)

            mesh_command = _run_command(case_path, "wsl", timeout_seconds=123, script_name="Allmesh")
            mesh_script = base64.b64decode(shlex.split(mesh_command[-1])[2]).decode("utf-8")
            self.assertIn("timeout --foreground --signal=TERM --kill-after=30s 123s ./Allmesh", mesh_script)
            self.assertNotIn("./Allrun", mesh_script)

    def test_mesh_only_progress_is_distinct_from_solver_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps({"status": "mesh_unverified", "cfd_quality": {"end_time": 500}}),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps({"status": "complete", "mode": "mesh", "returncode": 0}),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-mesh.json").write_text(
                json.dumps({"reusable": True, "trusted": False}),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.log").write_text(
                "=== AEROLAB MESH COMPLETE ===\n",
                encoding="utf-8",
            )

            progress = case_run_progress(case_path)

            self.assertEqual(progress["state"], "mesh_complete")
            self.assertTrue(progress["isMeshComplete"])
            self.assertFalse(progress["isComplete"])
            self.assertEqual(progress["runMode"], "mesh")

    def test_failed_feature_mesh_reports_cell_count_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "status": "mesh_failed",
                        "cfd_quality": {"end_time": 1200},
                        "mesh_resolution": {"smallest_aero_feature_m": 0.004},
                    }
                ),
                encoding="utf-8",
            )
            budget_recommendation = {
                "category": "mesh_cell_budget",
                "title": "Mesh refinement reached workstation budget pressure",
                "detail": "Use a conservative 2,050,000-cell allowance.",
                "safeCellBudget": 2_050_000,
                "retryAllowed": False,
            }
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "mode": "mesh",
                        "returncode": 15,
                        "budgetRecommendation": budget_recommendation,
                    }
                ),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.log").write_text(
                "\n".join(
                    (
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "Feature refinement iteration 9",
                        "After refinement feature refinement iteration 8 : cells:3714808 faces:1 points:1",
                        "Cells per refinement level:",
                        "    8 1312179",
                        "    9 1460200",
                    )
                ),
                encoding="utf-8",
            )

            progress = case_run_progress(case_path)

            self.assertEqual(progress["state"], "failed")
            self.assertIn("3,714,808 cells", progress["detail"])
            self.assertIn("refinement level 9", progress["detail"])
            self.assertIn("4 mm feature target", progress["detail"])
            self.assertIn("2,050,000-cell allowance", progress["detail"])
            self.assertEqual(progress["budgetRecommendation"], budget_recommendation)

    def test_interrupted_shell_refinement_is_not_mislabeled_as_feature_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "status": "mesh_failed",
                        "cfd_quality": {"end_time": 1200},
                        "mesh_resolution": {"smallest_aero_feature_m": 0.0161},
                    }
                ),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.json").write_text(
                json.dumps({"status": "failed", "mode": "mesh", "returncode": 15}),
                encoding="utf-8",
            )
            case_path.joinpath("aerolab-run.log").write_text(
                "\n".join(
                    (
                        "=== AEROLAB STEP: snappyHexMesh ===",
                        "After refinement shell refinement iteration 1 : cells:3299095 faces:1 points:1",
                        "Shell refinement iteration 2",
                    )
                ),
                encoding="utf-8",
            )

            progress = case_run_progress(case_path)

            self.assertEqual(progress["phase"], "Refining near-body mesh")
            self.assertIn("Mesh refinement was interrupted", progress["detail"])
            self.assertIn("WSL memory pressure", progress["detail"])
            self.assertNotIn("Feature meshing", progress["detail"])

    def test_live_run_progress_uses_openfoam_solver_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_json_path = case_path / "case.json"
            case_json_path.write_text(
                json.dumps(
                    {
                        "status": "solver_running",
                        "cfd_quality": {"end_time": 500},
                    }
                ),
                encoding="utf-8",
            )
            (case_path / "aerolab-run.json").write_text(
                json.dumps({"status": "running", "returncode": None}),
                encoding="utf-8",
            )
            (case_path / "aerolab-run.log").write_text(
                "\n".join(
                    [
                        "=== AEROLAB STEP: foamRun ===",
                        "Time = 249s",
                        "Time = 250s",
                    ]
                ),
                encoding="utf-8",
            )

            running = case_run_progress(case_path)

            self.assertEqual(running["state"], "running")
            self.assertEqual(running["phase"], "Solving airflow")
            self.assertEqual(running["percent"], 75)
            self.assertEqual(running["solverTime"], 250.0)
            self.assertEqual(running["solverEndTime"], 500.0)

            case_json_path.write_text(
                json.dumps({"status": "solver_unverified", "cfd_quality": {"end_time": 500}}),
                encoding="utf-8",
            )
            (case_path / "aerolab-run.json").write_text(
                json.dumps({"status": "complete", "returncode": 0, "trusted": False}),
                encoding="utf-8",
            )

            complete = case_run_progress(case_path)

            self.assertEqual(complete["state"], "complete")
            self.assertEqual(complete["tone"], "review")
            self.assertEqual(complete["percent"], 100)

    def test_settled_run_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            self._write_case(case_path, drifting=False)
            report = case_report(case_path, solver_returncode=0)

            self.assertTrue(report["meshQuality"]["passed"])
            self.assertTrue(report["residuals"]["stable"])
            self.assertTrue(report["forceCoeffs"]["stable"])
            forces = report["aerodynamicForces"]
            dynamic_pressure = 0.5 * 1.225 * 31.2928**2
            self.assertEqual(forces["verticalForceType"], "downforce")
            self.assertAlmostEqual(forces["verticalForceN"], 0.08 * dynamic_pressure * 2.2, places=4)
            self.assertAlmostEqual(forces["dragN"], 0.28 * dynamic_pressure * 2.2, places=4)
            self.assertTrue(report["qualityAssessment"]["trusted"])

    def test_settled_standard_run_without_feature_target_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            self._write_case(case_path, drifting=False)
            metadata_path = case_path / "case.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["mesh_resolution"]["smallest_aero_feature_m"] = None
            metadata["mesh_resolution"]["estimated_cells_across_feature"] = None
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            report = case_report(case_path, solver_returncode=0)
            feature_check = next(
                check
                for check in report["qualityAssessment"]["checks"]
                if check["label"] == "Aero feature resolution"
            )

            self.assertEqual(feature_check["status"], "fail")
            self.assertFalse(report["qualityAssessment"]["trusted"])

    def test_drifting_coefficients_are_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            self._write_case(case_path, drifting=True)
            report = case_report(case_path, solver_returncode=0)

            self.assertFalse(report["forceCoeffs"]["stable"])
            self.assertFalse(report["qualityAssessment"]["trusted"])

    def test_legacy_vtk_streamlines_are_normalized_for_browser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            vtk_dir = case_path / "postProcessing" / "streamlines" / "500"
            vtk_dir.mkdir(parents=True)
            vtk_dir.joinpath("aerolabStreamlines.vtk").write_text(
                """# vtk DataFile Version 2.0
streamlines
ASCII
DATASET POLYDATA
POINTS 4 float
-2 -1 0  -1 -1 0  -2 1 0  -1 1 0
LINES 2 6
2 0 1
2 2 3
POINT_DATA 4
FIELD attributes 2
U 3 4 float
10 0 0  12 0 0  8 0 0  9 0 0
p 1 4 float
1 2 3 4
""",
                encoding="utf-8",
            )
            parsed = parse_streamlines(
                case_path,
                {"normalizedCenter": [0, 0, 0], "normalizedScale": 1.0},
                "x",
            )
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["lineCount"], 2)
            self.assertEqual(parsed["pointCount"], 4)
            self.assertTrue(parsed["hasPressure"])
            self.assertEqual(parsed["speedRange"], [8.0, 12.0])
            self.assertEqual(parsed["pressureRange"], [1.0, 4.0])

    def test_temperature_results_parse_latest_internal_tmean_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "name": "thermal-results",
                        "solver_module": "fluid",
                        "flow": {"air_temperature_k": 288.15},
                    }
                ),
                encoding="utf-8",
            )
            initial = case_path / "0"
            initial.mkdir()
            initial.joinpath("T").write_text(
                "FoamFile { format ascii; }\ninternalField uniform 288.15;\n",
                encoding="utf-8",
            )
            final = case_path / "100"
            final.mkdir()
            final.joinpath("T").write_text(
                "FoamFile { format ascii; }\ninternalField uniform 999;\n",
                encoding="utf-8",
            )
            final.joinpath("TMean").write_text(
                """FoamFile
{
    format ascii;
}
internalField nonuniform List<scalar>
4
(
288.15
293.15
303.15
313.15
)
;
boundaryField {}
""",
                encoding="utf-8",
            )

            parsed = parse_temperature_results(case_path)
            report = case_report(case_path)

            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["field"], "TMean")
            self.assertTrue(parsed["timeAveraged"])
            self.assertEqual(parsed["sampleCount"], 4)
            self.assertAlmostEqual(parsed["minimumC"], 15.0)
            self.assertAlmostEqual(parsed["meanC"], 26.25)
            self.assertAlmostEqual(parsed["maximumC"], 40.0)
            self.assertAlmostEqual(parsed["maximumRiseK"], 25.0)
            self.assertEqual(report["temperatureResults"]["field"], "TMean")

    def test_body_pressure_vtk_is_converted_to_cp_for_browser(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            geometry_dir = case_path / "constant" / "geometry"
            geometry_dir.mkdir(parents=True)
            shutil.copyfile(project / "models" / "sample_box.stl", geometry_dir / "body.stl")
            case_path.joinpath("case.json").write_text(
                json.dumps(
                    {
                        "name": "cp-test",
                        "model": str(project / "models" / "sample_box.stl"),
                        "flow": {
                            "axis": "x",
                            "speed_mps": 10.0,
                            "air_density_kg_m3": 1.225,
                        },
                    }
                ),
                encoding="utf-8",
            )
            vtk_dir = case_path / "postProcessing" / "bodyPressure" / "500"
            vtk_dir.mkdir(parents=True)
            vtk_dir.joinpath("body.vtk").write_text(
                """# vtk DataFile Version 2.0
body pressure
ASCII
DATASET POLYDATA
POINTS 4 float
0 0 0  1 0 0  1 1 0  0 1 0
POLYGONS 2 8
3 0 1 2
3 0 2 3
POINT_DATA 4
FIELD attributes 2
p 1 4 float
50 0 -50 25
T 1 4 float
288.15 293.15 303.15 313.15
""",
                encoding="utf-8",
            )

            parsed = parse_surface_pressure(
                case_path,
                {"normalizedCenter": [0, 0, 0], "normalizedScale": 1.0},
                "x",
                speed_mps=10.0,
            )

            self.assertIsNotNone(parsed)
            self.assertTrue(parsed["hasPressure"])
            self.assertEqual(parsed["triangleCount"], 2)
            self.assertEqual(parsed["pointCount"], 4)
            self.assertEqual([point[3] for point in parsed["points"]], [1.0, 0.0, -1.0, 0.5])
            self.assertEqual(parsed["cpRange"], [-1.0, 1.0])
            self.assertAlmostEqual(parsed["dynamicPressurePa"], 61.25)
            self.assertEqual(parsed["pressurePaRange"], [-61.25, 61.25])
            self.assertTrue(parsed["hasPressureDrag"])
            self.assertEqual([point[4] for point in parsed["points"]], [0.0, 0.0, 0.0, 0.0])
            self.assertTrue(parsed["hasTemperature"])
            self.assertEqual(parsed["temperatureKRange"], [288.15, 313.15])
            self.assertEqual(parsed["temperatureCRange"], [15.0, 40.0])
            self.assertEqual(parsed["temperatureKValues"], [288.15, 293.15, 303.15, 313.15])
            self.assertIn("not a solid-component", parsed["temperatureDefinition"])

            report = case_report(case_path, include_visualization=True)
            self.assertEqual(report["surfacePressure"]["triangleCount"], 2)

    def test_surface_pressure_maps_area_weighted_local_drag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            vtk_dir = case_path / "postProcessing" / "bodyPressure" / "100"
            vtk_dir.mkdir(parents=True)
            vtk_dir.joinpath("body.vtk").write_text(
                """# vtk DataFile Version 2.0
pressure drag faces
ASCII
DATASET POLYDATA
POINTS 6 float
0 0 0  0 1 0  0 0 1  1 0 0  1 0 1  1 1 0
POLYGONS 2 8
3 0 1 2
3 3 4 5
POINT_DATA 6
FIELD attributes 2
p 1 6 float
50 50 50 -25 -25 -25
wallShearStress 3 6 float
5 0 0  5 0 0  5 0 0  5 0 0  5 0 0  5 0 0
""",
                encoding="utf-8",
            )

            parsed = parse_surface_pressure(
                case_path,
                {"normalizedCenter": [0, 0, 0], "normalizedScale": 1.0},
                "x",
                speed_mps=10.0,
                reference_area_m2=1.0,
            )

            self.assertIsNotNone(parsed)
            self.assertEqual([point[4] for point in parsed["points"]], [1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            self.assertEqual(parsed["pressureDragDensityRange"], [0.5, 1.0])
            self.assertAlmostEqual(parsed["pressureDragCoefficient"], 0.75)
            self.assertAlmostEqual(parsed["positivePressureDragCoefficient"], 0.75)
            self.assertAlmostEqual(parsed["offsetPressureDragCoefficient"], 0.0)
            self.assertTrue(parsed["hasWallShear"])
            self.assertEqual([point[5] for point in parsed["points"]], [0.1] * 6)
            self.assertEqual([point[6] for point in parsed["points"]], [1.1, 1.1, 1.1, 0.6, 0.6, 0.6])
            self.assertAlmostEqual(parsed["skinFrictionDragCoefficient"], 0.1)
            self.assertAlmostEqual(parsed["totalDragCoefficient"], 0.85)
            self.assertEqual(parsed["dragHotspotRegion"], "front")
            self.assertEqual(
                [region["id"] for region in parsed["dragRegions"]],
                ["front", "middle", "rear"],
            )
            self.assertAlmostEqual(
                sum(region["positiveDragSharePercent"] for region in parsed["dragRegions"]),
                100.0,
                places=1,
            )
            self.assertGreater(
                parsed["dragRegions"][0]["positiveDragSharePercent"],
                parsed["dragRegions"][2]["positiveDragSharePercent"],
            )

    def test_surface_drag_integrates_face_centered_solver_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            vtk_dir = case_path / "postProcessing" / "bodyPressure" / "100"
            vtk_dir.mkdir(parents=True)
            vtk_dir.joinpath("body.vtk").write_text(
                """# vtk DataFile Version 2.0
face-centered drag
ASCII
DATASET POLYDATA
POINTS 6 float
0 0 0  0 1 0  0 0 1  1 0 0  1 0 1  1 1 0
POLYGONS 2 8
3 0 1 2
3 3 4 5
CELL_DATA 2
FIELD attributes 2
p 1 2 float
50 -25
wallShearStress 3 2 float
5 0 0  5 0 0
""",
                encoding="utf-8",
            )

            parsed = parse_surface_pressure(
                case_path,
                {"normalizedCenter": [0, 0, 0], "normalizedScale": 1.0},
                "x",
                speed_mps=10.0,
                reference_area_m2=1.0,
            )

            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["pressureLocation"], "cell-averaged-to-point")
            self.assertEqual(parsed["wallShearLocation"], "cell-averaged-to-point")
            self.assertEqual(parsed["dragIntegrationSource"], "original face values before browser decimation")
            self.assertEqual(parsed["trianglePressureDragValues"], [1.0, 0.5])
            self.assertEqual(parsed["triangleTotalDragValues"], [1.1, 0.6])
            self.assertAlmostEqual(parsed["pressureDragCoefficient"], 0.75)
            self.assertAlmostEqual(parsed["skinFrictionDragCoefficient"], 0.1)
            self.assertAlmostEqual(parsed["totalDragCoefficient"], 0.85)

    def test_three_level_grid_study_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            values = {
                "draft": (100_000, 0.300, -0.070),
                "standard": (220_000, 0.286, -0.078),
                "fine": (480_000, 0.282, -0.081),
            }
            for level, (cells, cd, cl) in values.items():
                self._write_case(
                    root / level,
                    drifting=False,
                    cells=cells,
                    mean_cd=cd,
                    mean_cl=cl,
                    study={"id": "grid-test", "level": level},
                )

            study = case_report(root / "fine")["gridConvergence"]

            self.assertTrue(study["validated"])
            self.assertEqual(study["status"], "validated")
            self.assertLess(study["dragMetrics"]["standardToFinePercent"], 2.0)
            self.assertAlmostEqual(study["recommendedCd"], 0.282, places=3)

    def test_mesh_sensitive_drag_fails_grid_study(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            values = {
                "draft": (100_000, 0.320),
                "standard": (220_000, 0.290),
                "fine": (480_000, 0.260),
            }
            for level, (cells, cd) in values.items():
                self._write_case(
                    root / level,
                    drifting=False,
                    cells=cells,
                    mean_cd=cd,
                    study={"id": "grid-sensitive", "level": level},
                )

            study = case_report(root / "fine")["gridConvergence"]

            self.assertFalse(study["validated"])
            self.assertEqual(study["status"], "failed")
            drag_check = next(check for check in study["checks"] if check["label"] == "Drag grid sensitivity")
            self.assertEqual(drag_check["status"], "fail")

    def _write_case(
        self,
        case_path: Path,
        drifting: bool,
        cells: int = 250_000,
        mean_cd: float = 0.28,
        mean_cl: float = -0.08,
        study: dict[str, object] | None = None,
    ) -> None:
        case_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "quality-test",
            "model": "shared-model.stl",
            "status": "openfoam_case_generated",
            "units": {"scale_to_meters": 1.0},
            "orientation": {"target_flow_axis": "x"},
            "simulation_type": "steady_external_incompressible_airflow",
            "solver_module": "incompressibleFluid",
            "flow": {"axis": "x", "speed_mph": 70, "speed_mps": 31.2928},
            "ground": {
                "enabled": True,
                "moving": True,
                "clearance_m": 0.0,
                "road_elevation_m": 0.0,
                "lowest_model_z_m": 0.0,
            },
            "placement": {
                "method": "lowest_point_to_road_clearance",
                "verified": True,
                "ground_clearance_m": 0.0,
            },
            "aerodynamic_reference": {"area_m2": 2.2, "length_m": 4.5},
            "wall_resolution": {"surface_layers": 5, "target_y_plus": 80},
            "mesh_resolution": {
                "quality": "standard",
                "smallest_aero_feature_m": 0.04,
                "estimated_cells_across_feature": 4.5,
                "supported": True,
            },
            "geometry_fidelity": {"status": "original", "verified": True},
            "geometry_validation": {
                "status": "verified",
                "verified": True,
                "measured_dimensions_m": {"length_m": 4.5, "width_m": 1.8, "height_m": 1.4},
            },
        }
        if study:
            payload["validation_study"] = study
        (case_path / "case.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        (case_path / "aerolab-run.json").write_text(
            json.dumps({"ok": True, "trusted": True, "backend": "test", "returncode": 0}),
            encoding="utf-8",
        )
        (case_path / "mesh-surface-fidelity.json").write_text(
            json.dumps(
                {
                    "status": "verified",
                    "verified": True,
                    "symmetricP95M": 0.001,
                    "symmetricP99M": 0.002,
                    "dimensionChangePercent": 0.1,
                    "projectedAreaChangePercent": 0.2,
                }
            ),
            encoding="utf-8",
        )
        residual_lines = []
        for _ in range(40):
            for field in ("Ux", "Uy", "Uz", "p", "k", "omega"):
                residual_lines.append(
                    f"smoothSolver: Solving for {field}, Initial residual = 1e-5, "
                    "Final residual = 1e-7, No Iterations 2"
                )
        log = "\n".join(
            [
                "=== AEROLAB STEP: snappyHexMesh ===",
                "Extruding 50000 out of 50000 faces (100%).",
                "Added 250000 out of 250000 cells (100%).",
                "=== AEROLAB STEP: checkMesh ===",
                f"cells: {cells}",
                "Max aspect ratio = 12.5 OK.",
                "Mesh non-orthogonality Max: 48 average: 8",
                "Max skewness = 2.1 OK.",
                "Mesh OK.",
                "=== AEROLAB STEP: foamRun ===",
                *residual_lines,
                "=== AEROLAB STEP: yPlus ===",
                "patch body y+ : min = 18, max = 150, average = 76",
                "=== AEROLAB COMPLETE ===",
            ]
        )
        (case_path / "aerolab-run.log").write_text(log, encoding="utf-8")

        coeff_dir = case_path / "postProcessing" / "forceCoeffs" / "0"
        coeff_dir.mkdir(parents=True)
        rows = ["# Time Cd Cs Cl CmRoll CmPitch CmYaw"]
        for index in range(40):
            cd = mean_cd + (0.004 * index if drifting else 0.0002 * ((index % 4) - 1.5))
            cl = mean_cl + 0.0002 * ((index % 5) - 2)
            rows.append(f"{index * 10} {cd:.6f} 0 {cl:.6f} 0 0 0")
        (coeff_dir / "coefficient.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
