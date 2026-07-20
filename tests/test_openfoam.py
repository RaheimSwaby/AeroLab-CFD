from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aerolab.case import create_case
from aerolab.cli import main as cli_main
from aerolab.openfoam import ensure_case_postprocessing
from aerolab.solver import case_report
from aerolab.stl import inspect_stl, read_stl_triangles, write_binary_stl_triangles


class OpenFoamCaseTests(unittest.TestCase):
    def test_generates_transient_time_averaged_vehicle_case(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="transient-vehicle",
                speed_mph=70,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                include_ground=True,
                moving_ground=True,
                quality="draft",
                simulation_mode="transient",
            )

            metadata = json.loads(case_path.joinpath("case.json").read_text(encoding="utf-8"))
            control = case_path.joinpath("system", "controlDict").read_text(encoding="utf-8")
            schemes = case_path.joinpath("system", "fvSchemes").read_text(encoding="utf-8")
            solution = case_path.joinpath("system", "fvSolution").read_text(encoding="utf-8")
            streamlines = case_path.joinpath("system", "streamlines").read_text(encoding="utf-8")
            body_pressure = case_path.joinpath("system", "bodyPressure").read_text(encoding="utf-8")

            self.assertEqual(metadata["simulation_type"], "transient_external_incompressible_airflow")
            self.assertEqual(metadata["cfd_quality"]["simulation_mode"], "transient")
            self.assertGreater(metadata["cfd_quality"]["averaging_window_s"], 0)
            self.assertIn("adjustTimeStep yes;", control)
            self.assertIn("maxCo 1.5;", control)
            self.assertIn("type fieldAverage;", control)
            self.assertIn("fields (U p k wallShearStress);", control)
            self.assertIn("default Euler;", schemes)
            self.assertNotIn("steadyState", schemes)
            self.assertIn("PIMPLE", solution)
            self.assertNotIn("SIMPLE\n", solution)
            self.assertIn("U UMean;", streamlines)
            self.assertIn("pMean", streamlines)
            self.assertIn("fields (pMean wallShearStressMean);", body_pressure)
            self.assertNotIn(" TMean", body_pressure)
            self.assertNotIn("thermal", metadata["physical_model"])
            self.assertFalse((case_path / "constant" / "fvModels").exists())

    def test_incompressible_case_rejects_mach_point_three_and_records_flow_state(self) -> None:
        project = Path(__file__).resolve().parents[1]
        model_path = project / "models" / "sample_box.stl"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "compressible CFD solver"):
                create_case(
                    model_path=model_path,
                    case_name="too-fast",
                    speed_mph=250,
                    flow_axis="x",
                    cases_dir=Path(temp_dir),
                    generate_openfoam=False,
                )

            case_path = create_case(
                model_path=model_path,
                case_name="flow-state",
                speed_mph=70,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                generate_openfoam=False,
            )
            flow = json.loads(case_path.joinpath("case.json").read_text(encoding="utf-8"))["flow"]

            self.assertAlmostEqual(flow["mach_number"], 31.2928 / 343.0)
            self.assertAlmostEqual(flow["dynamic_pressure_pa"], 0.5 * 1.225 * 31.2928**2)
            self.assertAlmostEqual(flow["reynolds_number"], 31.2928 / 1.5e-5)

    def test_reusing_case_name_preserves_existing_case(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            cases_dir = Path(temp_dir)
            first = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="same-name",
                speed_mph=70,
                flow_axis="x",
                cases_dir=cases_dir,
                generate_openfoam=False,
            )
            first.joinpath("prior-result.txt").write_text("keep", encoding="utf-8")

            second = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="same-name",
                speed_mph=70,
                flow_axis="x",
                cases_dir=cases_dir,
                generate_openfoam=False,
            )

            self.assertEqual(first.name, "same-name")
            self.assertEqual(second.name, "same-name-2")
            self.assertEqual(first.joinpath("prior-result.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual(json.loads(second.joinpath("case.json").read_text(encoding="utf-8"))["name"], "same-name-2")

    def test_ground_clearance_translates_body_above_zero_road_plane(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="ground-clearance",
                speed_mph=70,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                include_ground=True,
                moving_ground=True,
                ground_clearance_m=0.125,
                quality="draft",
            )

            metadata = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
            body_report = inspect_stl(case_path / "constant" / "geometry" / "body.stl")
            block_mesh = (case_path / "system" / "blockMeshDict").read_text(encoding="utf-8")
            vertices_text = block_mesh.split("vertices\n(", 1)[1].split(");", 1)[0]
            road_z_values = [
                float(line.strip().strip("()").split()[2])
                for line in vertices_text.splitlines()
                if line.strip().startswith("(")
            ]

            self.assertAlmostEqual(body_report.bounds.minimum[2], 0.125, places=6)
            self.assertAlmostEqual(min(road_z_values), 0.0, places=9)
            self.assertEqual(metadata["ground"]["clearance_m"], 0.125)
            self.assertAlmostEqual(metadata["ground"]["lowest_model_z_m"], 0.125, places=9)
            self.assertTrue(metadata["placement"]["verified"])
            self.assertAlmostEqual(metadata["placement"]["translation_m"]["z"], 0.125, places=9)
            self.assertEqual(case_report(case_path)["caseSetup"]["placement"]["ground_clearance_m"], 0.125)

            with self.assertRaisesRegex(ValueError, "requires the ground patch"):
                create_case(
                    model_path=project / "models" / "sample_box.stl",
                    case_name="floating-open-tunnel",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=Path(temp_dir),
                    ground_clearance_m=0.125,
                    generate_openfoam=False,
                )

    def test_measured_dimensions_validate_scale_and_gate_accuracy_studies(self) -> None:
        project = Path(__file__).resolve().parents[1]
        model_path = project / "models" / "sample_box.stl"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            verified = create_case(
                model_path=model_path,
                case_name="dimension-verified",
                speed_mph=70,
                flow_axis="x",
                cases_dir=root,
                measured_length_m=1.0,
                measured_width_m=1.0,
                measured_height_m=1.0,
                generate_openfoam=False,
            )
            metadata = json.loads((verified / "case.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["geometry_validation"]["verified"])

            with self.assertRaisesRegex(ValueError, "vehicle width"):
                create_case(
                    model_path=model_path,
                    case_name="dimension-mismatch",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=root,
                    measured_length_m=1.0,
                    measured_width_m=2.0,
                    measured_height_m=1.0,
                    generate_openfoam=False,
                )
            self.assertFalse((root / "dimension-mismatch").exists())

            with self.assertRaisesRegex(ValueError, "Accuracy studies require"):
                create_case(
                    model_path=model_path,
                    case_name="study-missing-measurements",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=root,
                    generate_openfoam=False,
                    validation_study={"id": "grid-test", "level": "draft"},
                )
            self.assertFalse((root / "study-missing-measurements").exists())

            with self.assertRaisesRegex(ValueError, "smallest aerodynamic feature"):
                create_case(
                    model_path=model_path,
                    case_name="study-missing-feature-target",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=root,
                    measured_length_m=1.0,
                    measured_width_m=1.0,
                    measured_height_m=1.0,
                    generate_openfoam=False,
                    validation_study={"id": "grid-test", "level": "draft"},
                )
            self.assertFalse((root / "study-missing-feature-target").exists())

    def test_rejects_prepared_mesh_without_fidelity_record(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared = root / "models" / "prepared"
            prepared.mkdir(parents=True)
            model_path = prepared / "sample-prepared.stl"
            shutil.copyfile(project / "models" / "sample_box.stl", model_path)

            with self.assertRaisesRegex(ValueError, "repair-fidelity record"):
                create_case(
                    model_path=model_path,
                    case_name="unverified-repair",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=root / "cases",
                )
            self.assertFalse((root / "cases" / "unverified-repair").exists())

    def test_generates_foundation_v13_external_aero_case(self) -> None:
        project = Path(__file__).resolve().parents[1]
        model_path = project / "models" / "sample_box.stl"
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=model_path,
                case_name="v13-case",
                speed_mph=70,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                include_ground=True,
                moving_ground=True,
                quality="standard",
                smallest_aero_feature_m=0.005,
            )

            control = (case_path / "system" / "controlDict").read_text(encoding="utf-8")
            snappy = (case_path / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
            allrun = (case_path / "Allrun").read_text(encoding="utf-8")
            allmesh = (case_path / "Allmesh").read_text(encoding="utf-8")
            allsolve = (case_path / "Allsolve").read_text(encoding="utf-8")
            self.assertNotIn(b"\r\n", (case_path / "Allrun").read_bytes())
            self.assertNotIn(b"\r\n", (case_path / "Allmesh").read_bytes())
            self.assertNotIn(b"\r\n", (case_path / "Allsolve").read_bytes())
            self.assertNotIn(b"\r\n", (case_path / "system" / "controlDict").read_bytes())

            self.assertIn("solver incompressibleFluid;", control)
            self.assertIn("timeInterval 10;", control)
            self.assertTrue((case_path / "constant" / "physicalProperties").exists())
            self.assertTrue((case_path / "constant" / "momentumTransport").exists())
            self.assertTrue((case_path / "constant" / "geometry" / "body.stl").exists())
            body_stl = case_path / "constant" / "geometry" / "body.stl"
            body_report = inspect_stl(body_stl)
            self.assertEqual(body_report.format, "binary")
            self.assertEqual(body_stl.stat().st_size, 84 + body_report.triangle_count * 50)
            self.assertTrue((case_path / "system" / "surfaceFeaturesDict").exists())
            self.assertIn("bodyRefinement", snappy)
            self.assertIn("wakeRefinement", snappy)
            self.assertIn("nSurfaceLayers 5;", snappy)
            self.assertIn("relativeSizes false;", snappy)
            self.assertIn("firstLayerThickness", snappy)
            self.assertIn("maxBoundarySkewness 4;", snappy)
            self.assertIn("minTetQuality -1e30;", snappy)
            self.assertIn("insidePoint", snappy)
            self.assertTrue((case_path / "system" / "streamlines").exists())
            streamlines = (case_path / "system" / "streamlines").read_text(encoding="utf-8")
            seed_lines = [line for line in streamlines.splitlines() if line.startswith("        (")]
            self.assertEqual(len(seed_lines), 187)
            self.assertTrue((case_path / "system" / "wallShearStress").exists())
            wall_shear = (case_path / "system" / "wallShearStress").read_text(encoding="utf-8")
            self.assertIn("type wallShearStress;", wall_shear)
            self.assertIn("patches (body);", wall_shear)
            self.assertTrue((case_path / "system" / "bodyPressure").exists())
            body_pressure = (case_path / "system" / "bodyPressure").read_text(encoding="utf-8")
            self.assertIn("type surfaces;", body_pressure)
            self.assertIn("patches (body);", body_pressure)
            self.assertIn("fields (p wallShearStress);", body_pressure)
            self.assertIn("interpolate no;", body_pressure)
            self.assertIn("writeFormat ascii;", body_pressure)
            self.assertTrue((case_path / "system" / "yPlus").exists())
            self.assertIn("checkMesh | tee log.checkMesh", allrun)
            self.assertIn('grep -q "Mesh OK." log.checkMesh', allrun)
            self.assertIn("checkMesh -allGeometry -allTopology", allrun)
            self.assertLess(allrun.index("checkMesh | tee"), allrun.index("checkMesh -allGeometry"))
            self.assertIn("potentialFoam", allrun)
            self.assertIn("foamRun", allrun)
            self.assertIn("foamPostProcess -solver incompressibleFluid -func yPlus", allrun)
            self.assertIn("foamPostProcess -func streamlines", allrun)
            self.assertIn("foamPostProcess -solver incompressibleFluid -func wallShearStress", allrun)
            self.assertIn("foamPostProcess -func bodyPressure", allrun)
            self.assertLess(allrun.index("foamPostProcess -func streamlines"), allrun.index("foamPostProcess -solver"))
            wall_shear_command = "foamPostProcess -solver incompressibleFluid -func wallShearStress"
            self.assertLess(allrun.index("foamPostProcess -func streamlines"), allrun.index(wall_shear_command))
            self.assertLess(allrun.index(wall_shear_command), allrun.index("foamPostProcess -func bodyPressure"))
            self.assertNotIn("simpleFoam", allrun)
            self.assertIn("snappyHexMesh -overwrite", allmesh)
            self.assertIn("postProcessing/meshSurface/0/body.vtk", allmesh)
            self.assertIn("=== AEROLAB MESH COMPLETE ===", allmesh)
            self.assertNotIn("foamRun", allmesh)
            self.assertIn("checkMesh | tee log.checkMesh", allsolve)
            self.assertIn("foamRun", allsolve)
            self.assertNotIn("snappyHexMesh", allsolve)
            fv_solution = (case_path / "system" / "fvSolution").read_text(encoding="utf-8")
            control_dict = (case_path / "system" / "controlDict").read_text(encoding="utf-8")
            self.assertIn("residualControl", fv_solution)
            self.assertIn('"(k|omega)" 0.0005;', fv_solution)
            self.assertIn("U 0.5;", fv_solution)
            self.assertIn("endTime 1200;", control_dict)

            case_metadata = (case_path / "case.json").read_text(encoding="utf-8")
            self.assertIn('"target_y_plus": 60.0', case_metadata)
            self.assertIn('"surface_layers": 5', case_metadata)
            metadata = json.loads(case_metadata)
            self.assertEqual(metadata["mesh_resolution"]["status"], "pass")
            self.assertTrue(metadata["mesh_resolution"]["adaptive_refinement"])
            self.assertEqual(metadata["mesh_resolution"]["configured_surface_max_level"], 8)
            self.assertEqual(metadata["mesh_resolution"]["configured_body_region_level"], 3)
            self.assertEqual(metadata["mesh_resolution"]["configured_n_cells_between_levels"], 2)
            self.assertGreaterEqual(metadata["mesh_resolution"]["estimated_cells_across_feature"], 4.0)
            self.assertIn("maxLocalCells 2000000;", snappy)
            self.assertIn("maxGlobalCells 2800000;", snappy)
            self.assertIn("nCellsBetweenLevels 2;", snappy)
            self.assertIn("bodyRefinement\n        {\n            mode inside;\n            level 3;", snappy)
            self.assertIn("minRefinementCells 100;", snappy)
            self.assertIn("level 8;", snappy)
            self.assertIn("level (4 8);", snappy)
            self.assertIn("level 3;", snappy)

            vtk_dir = case_path / "postProcessing" / "streamlines" / "500"
            vtk_dir.mkdir(parents=True)
            vtk_dir.joinpath("tracks.vtk").write_text(
                """# vtk DataFile Version 2.0
tracks
ASCII
DATASET POLYDATA
POINTS 2 float
-2 0 0  2 0 0
LINES 1 3
2 0 1
POINT_DATA 2
FIELD attributes 1
U 3 2 float
30 0 0  25 0 0
""",
                encoding="utf-8",
            )
            browser_report = case_report(case_path, include_visualization=True)
            self.assertGreater(browser_report["geometryPreview"]["sampledTriangleCount"], 0)
            self.assertEqual(browser_report["solverStreamlines"]["lineCount"], 1)
            self.assertEqual(browser_report["meshResolution"]["smallest_aero_feature_m"], 0.005)
            self.assertTrue(browser_report["surfacePressureSetup"]["configured"])
            self.assertTrue(browser_report["surfacePressureSetup"]["wallShearConfigured"])

    def test_generates_total_power_heat_sources_and_temperature_export(self) -> None:
        project = Path(__file__).resolve().parents[1]
        model_path = project / "models" / "sample_box.stl"
        heat_zones = [
            {
                "name": "engineHeat",
                "shape": "box",
                "component": "engine",
                "minimum_m": [0.2, 0.2, 0.2],
                "maximum_m": [0.6, 0.6, 0.6],
                "power_kw": 75,
            },
            {
                "name": "exhaustHeat",
                "shape": "box",
                "component": "exhaust",
                "minimum_m": [0.65, 0.2, 0.2],
                "maximum_m": [0.85, 0.4, 0.4],
                "power_w": 12_500,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cases_dir = Path(temp_dir)
            case_path = create_case(
                model_path=model_path,
                case_name="thermal-zones",
                speed_mph=70,
                flow_axis="x",
                cases_dir=cases_dir,
                quality="draft",
                simulation_mode="transient",
                fluid_profile="compressible_thermal",
                heat_zones=heat_zones,
            )

            metadata = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
            normalized = metadata["physical_model"]["volume_zones"]["heat_zones"]
            fv_models = (case_path / "constant" / "fvModels").read_text(encoding="utf-8")
            snappy = (case_path / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
            body_pressure = (case_path / "system" / "bodyPressure").read_text(encoding="utf-8")
            control = (case_path / "system" / "controlDict").read_text(encoding="utf-8")
            allrun = (case_path / "Allrun").read_text(encoding="utf-8")

            self.assertEqual(metadata["solver_module"], "fluid")
            self.assertEqual([zone["shape"] for zone in normalized], ["box", "box"])
            self.assertEqual([zone["power_w"] for zone in normalized], [75_000.0, 12_500.0])
            self.assertEqual(metadata["physical_model"]["thermal"]["total_power_w"], 87_500.0)
            self.assertEqual(metadata["physical_model"]["thermal"]["model"], "direct_air_volumetric_heat_source")
            self.assertIn("engineHeat\n{\n    type heatSource;", fv_models)
            self.assertIn("cellZone engineHeat;\n    Q 75000;", fv_models)
            self.assertIn("cellZone exhaustHeat;\n    Q 12500;", fv_models)
            self.assertNotIn("q 75000;", fv_models)
            self.assertIn("faceZone engineHeatFaces;", snappy)
            self.assertIn("cellZone engineHeat;", snappy)
            self.assertIn("fields (pMean wallShearStressMean TMean);", body_pressure)
            self.assertIn("fields (U p T k wallShearStress);", control)
            self.assertIn("solver fluid;", control)
            self.assertIn("foamPostProcess -solver fluid -func wallShearStress", allrun)
            self.assertTrue((case_path / "0" / "T").is_file())

            with self.assertRaisesRegex(ValueError, "compressible_thermal"):
                create_case(
                    model_path=model_path,
                    case_name="incompressible-heat",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=cases_dir,
                    generate_openfoam=False,
                    heat_zones=heat_zones[:1],
                )
            with self.assertRaisesRegex(ValueError, "unsupported shape"):
                create_case(
                    model_path=model_path,
                    case_name="unsupported-heat-shape",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=cases_dir,
                    generate_openfoam=False,
                    fluid_profile="compressible_thermal",
                    heat_zones=[{**heat_zones[0], "shape": "cylinder"}],
                )
            with self.assertRaisesRegex(ValueError, "exactly one of power_w or power_kw"):
                create_case(
                    model_path=model_path,
                    case_name="ambiguous-heat-power",
                    speed_mph=70,
                    flow_axis="x",
                    cases_dir=cases_dir,
                    generate_openfoam=False,
                    fluid_profile="compressible_thermal",
                    heat_zones=[{**heat_zones[0], "power_w": 75_000}],
                )

    def test_cli_reads_heat_zone_configuration(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_dir = root / "cases"
            heat_config = root / "heat-zones.json"
            heat_config.write_text(
                json.dumps(
                    [
                        {
                            "name": "brakeHeat",
                            "component": "front brakes",
                            "minimum_m": [0.1, 0.1, 0.1],
                            "maximum_m": [0.3, 0.3, 0.3],
                            "power_w": 2_500,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = cli_main(
                [
                    "init-case",
                    str(project / "models" / "sample_box.stl"),
                    "--name",
                    "cli-thermal-case",
                    "--cases-dir",
                    str(cases_dir),
                    "--quality",
                    "draft",
                    "--fluid-profile",
                    "compressible_thermal",
                    "--heat-zones-config",
                    str(heat_config),
                ]
            )

            case_path = cases_dir / "cli-thermal-case"
            metadata = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
            fv_models = (case_path / "constant" / "fvModels").read_text(encoding="utf-8")
            self.assertEqual(result, 0)
            self.assertEqual(
                metadata["physical_model"]["volume_zones"]["heat_zones"][0]["power_w"],
                2_500.0,
            )
            self.assertIn("cellZone brakeHeat;", fv_models)
            self.assertIn("Q 2500;", fv_models)

    def test_legacy_case_postprocessing_upgrade_is_idempotent(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="legacy-pressure-case",
                speed_mph=40,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                quality="standard",
            )
            pressure_path = case_path / "system" / "bodyPressure"
            pressure_path.unlink()
            wall_shear_path = case_path / "system" / "wallShearStress"
            wall_shear_path.unlink()
            allrun_path = case_path / "Allrun"
            old_allrun = allrun_path.read_text(encoding="utf-8")
            old_allrun = old_allrun.replace(
                'echo "=== AEROLAB STEP: bodyPressure ==="\n'
                "foamPostProcess -func bodyPressure -latestTime\n",
                "",
            )
            old_allrun = old_allrun.replace(
                'echo "=== AEROLAB STEP: wallShearStress ==="\n'
                "foamPostProcess -solver incompressibleFluid -func wallShearStress -latestTime\n",
                "",
            )
            allrun_path.write_text(old_allrun, encoding="utf-8", newline="\n")

            before = case_report(case_path)
            first = ensure_case_postprocessing(case_path)
            after_first = allrun_path.read_text(encoding="utf-8")
            second = ensure_case_postprocessing(case_path)
            after_second = allrun_path.read_text(encoding="utf-8")

            self.assertFalse(before["surfacePressureSetup"]["configured"])
            self.assertTrue(first["upgraded"])
            self.assertTrue(first["bodyPressureCreated"])
            self.assertTrue(first["wallShearStressCreated"])
            self.assertTrue(first["allrunUpdated"])
            self.assertTrue(pressure_path.is_file())
            self.assertTrue(wall_shear_path.is_file())
            self.assertIn("fields (p wallShearStress);", pressure_path.read_text(encoding="utf-8"))
            self.assertIn("interpolate no;", pressure_path.read_text(encoding="utf-8"))
            self.assertEqual(after_first.count("foamPostProcess -func bodyPressure"), 1)
            self.assertEqual(after_first.count("foamPostProcess -solver incompressibleFluid -func wallShearStress"), 1)
            self.assertLess(
                after_first.index("foamPostProcess -func streamlines"),
                after_first.index("foamPostProcess -solver incompressibleFluid -func wallShearStress"),
            )
            self.assertLess(
                after_first.index("foamPostProcess -solver incompressibleFluid -func wallShearStress"),
                after_first.index("foamPostProcess -func bodyPressure"),
            )
            self.assertFalse(second["upgraded"])
            self.assertEqual(after_first, after_second)
            self.assertTrue(case_report(case_path)["surfacePressureSetup"]["configured"])

    def test_rejects_ground_with_vertical_flow(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Ground runs require X or Y flow"):
                create_case(
                    model_path=project / "models" / "sample_box.stl",
                    case_name="bad-axis",
                    speed_mph=70,
                    flow_axis="z",
                    cases_dir=Path(temp_dir),
                    include_ground=True,
                )

    def test_rejects_feature_target_beyond_local_mesh_limit(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            cases_dir = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "local-device limit"):
                create_case(
                    model_path=project / "models" / "sample_box.stl",
                    case_name="unsupported-feature",
                    speed_mph=40,
                    flow_axis="x",
                    cases_dir=cases_dir,
                    quality="standard",
                    smallest_aero_feature_m=1e-6,
                )
            self.assertFalse((cases_dir / "unsupported-feature").exists())

    def test_draft_case_skips_y_plus_and_exports_streamlines(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="draft-case",
                speed_mph=30,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                quality="draft",
            )

            allrun = (case_path / "Allrun").read_text(encoding="utf-8")
            self.assertNotIn("foamPostProcess -func yPlus", allrun)
            self.assertIn("foamPostProcess -func streamlines", allrun)
            self.assertLess(allrun.index("foamRun"), allrun.index("streamlines"))

    def test_model_rotation_is_baked_into_case_geometry(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = create_case(
                model_path=project / "models" / "sample_box.stl",
                case_name="rotated-case",
                speed_mph=30,
                flow_axis="x",
                cases_dir=Path(temp_dir),
                quality="draft",
                model_rotation_degrees=(0.0, 0.0, 45.0),
            )

            metadata = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
            dimensions = metadata["scaled_geometry_report"]["bounds"]["dimensions"]
            self.assertAlmostEqual(dimensions[0], 2**0.5, places=5)
            self.assertAlmostEqual(dimensions[1], 2**0.5, places=5)
            self.assertEqual(metadata["orientation"]["rotation_degrees"]["z"], 45.0)

    def test_case_generation_adapts_to_different_vehicle_proportions(self) -> None:
        project = Path(__file__).resolve().parents[1]
        source_triangles, _ = read_stl_triangles(project / "models" / "sample_box.stl")
        vehicle_dimensions = {
            "compact-car": (4.2, 1.75, 1.45),
            "delivery-van": (5.8, 2.10, 2.55),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = {}
            for name, dimensions in vehicle_dimensions.items():
                model_path = root / f"{name}.stl"
                triangles = [
                    tuple(
                        (
                            vertex[0] * dimensions[0],
                            vertex[1] * dimensions[1],
                            vertex[2] * dimensions[2],
                        )
                        for vertex in triangle
                    )
                    for triangle in source_triangles
                ]
                write_binary_stl_triangles(model_path, triangles)
                case_path = create_case(
                    model_path=model_path,
                    case_name=name,
                    speed_mph=65,
                    flow_axis="x",
                    cases_dir=root / "cases",
                    include_ground=True,
                    moving_ground=True,
                    ground_clearance_m=0.12,
                    measured_length_m=dimensions[0],
                    measured_width_m=dimensions[1],
                    measured_height_m=dimensions[2],
                    smallest_aero_feature_m=0.15,
                    quality="standard",
                )
                metadata = json.loads(case_path.joinpath("case.json").read_text(encoding="utf-8"))
                generated[name] = metadata
                self.assertTrue(metadata["geometry_validation"]["verified"])
                self.assertAlmostEqual(metadata["aerodynamic_reference"]["length_m"], dimensions[0], places=5)
                self.assertEqual(metadata["model"], str(model_path.resolve()))
                self.assertTrue(case_path.joinpath("Allmesh").is_file())

            self.assertNotEqual(
                generated["compact-car"]["mesh_resolution"]["base_cell_dimensions_m"],
                generated["delivery-van"]["mesh_resolution"]["base_cell_dimensions_m"],
            )
            self.assertNotEqual(
                generated["compact-car"]["aerodynamic_reference"]["area_m2"],
                generated["delivery-van"]["aerodynamic_reference"]["area_m2"],
            )


if __name__ == "__main__":
    unittest.main()
