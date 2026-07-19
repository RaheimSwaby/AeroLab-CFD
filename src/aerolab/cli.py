from __future__ import annotations

import argparse
import json
from pathlib import Path

from .case import create_case
from .solver import case_report, run_case, solver_status
from .stl import inspect_stl
from .webapp import run_app

UNIT_SCALES = {
    "m": 1.0,
    "mm": 0.001,
    "cm": 0.01,
    "in": 0.0254,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aerolab",
        description="Local-first CFD workflow tools for external aerodynamics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Inspect an STL mesh and report whether it is a good CFD candidate.",
    )
    check_parser.add_argument("model", type=Path, help="Path to an STL model.")
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )

    case_parser = subparsers.add_parser(
        "init-case",
        help="Create local case metadata for a future CFD run.",
    )
    case_parser.add_argument("model", type=Path, help="Path to an STL model.")
    case_parser.add_argument("--name", required=True, help="Case folder name.")
    case_parser.add_argument(
        "--speed-mph",
        type=float,
        default=70.0,
        help="Free-stream air speed in miles per hour.",
    )
    case_parser.add_argument(
        "--flow-axis",
        choices=["x", "y", "z"],
        default="x",
        help="Axis aligned with incoming air flow.",
    )
    case_parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("cases"),
        help="Directory where local cases are created.",
    )
    case_parser.add_argument(
        "--ground",
        action="store_true",
        help="Add a ground patch to the virtual wind tunnel.",
    )
    case_parser.add_argument(
        "--moving-ground",
        action="store_true",
        help="Set the ground patch velocity to match the incoming air.",
    )
    case_parser.add_argument(
        "--ground-clearance-mm",
        type=float,
        default=0.0,
        help="Raise the STL's lowest point above the road by this many millimeters.",
    )
    case_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Create only case.json without OpenFOAM files.",
    )
    case_parser.add_argument(
        "--units",
        choices=sorted(UNIT_SCALES),
        default="m",
        help="Input STL units. Case geometry is converted to meters for CFD.",
    )
    case_parser.add_argument(
        "--reference-area-m2",
        type=float,
        help="Manual aerodynamic reference area in square meters for force coefficients.",
    )
    case_parser.add_argument(
        "--reference-length-m",
        type=float,
        help="Manual aerodynamic reference length in meters for force coefficients.",
    )
    case_parser.add_argument(
        "--measured-length-m",
        type=float,
        help="Physical model length in meters for geometry validation.",
    )
    case_parser.add_argument(
        "--measured-width-m",
        type=float,
        help="Physical model width in meters for geometry validation.",
    )
    case_parser.add_argument(
        "--measured-height-m",
        type=float,
        help="Physical model height in meters for geometry validation.",
    )
    case_parser.add_argument(
        "--smallest-aero-feature-mm",
        type=float,
        help="Smallest aerodynamic feature to resolve, in millimeters.",
    )
    case_parser.add_argument(
        "--quality",
        choices=["draft", "standard", "fine"],
        default="standard",
        help="CFD mesh/solve quality preset for generated OpenFOAM files.",
    )
    case_parser.add_argument(
        "--simulation-mode",
        choices=["steady", "transient"],
        default="steady",
        help="Steady RANS or transient PIMPLE with time-averaged output.",
    )
    case_parser.add_argument(
        "--source-flow-direction",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="+x",
        help="Direction that incoming air should travel across the source STL before orientation.",
    )
    case_parser.add_argument(
        "--source-up-direction",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="+z",
        help="Up direction in the source STL before orientation.",
    )
    for axis in ("x", "y", "z"):
        case_parser.add_argument(
            f"--rotate-{axis}-deg",
            type=float,
            default=0.0,
            help=f"Rotate the source model around its {axis.upper()} axis before tunnel alignment.",
        )

    app_parser = subparsers.add_parser(
        "app",
        help="Start the local AeroLab browser app.",
    )
    app_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local app.",
    )
    app_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local app.",
    )
    app_parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing models, cases, and outputs.",
    )

    status_parser = subparsers.add_parser(
        "solver-status",
        help="Check whether a local OpenFOAM backend is available.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print solver status as JSON.",
    )

    run_parser = subparsers.add_parser(
        "run-case",
        help="Run a generated OpenFOAM case locally.",
    )
    run_parser.add_argument("case", type=Path, help="Path to a generated case folder.")
    run_parser.add_argument(
        "--backend",
        choices=["auto", "native", "wsl", "docker"],
        default="auto",
        help="OpenFOAM backend to use.",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum solver runtime.",
    )
    run_parser.add_argument(
        "--mode",
        choices=["full", "mesh"],
        default="full",
        help="Validate only the mesh, or run the complete solver workflow.",
    )
    run_parser.add_argument(
        "--no-reuse-mesh",
        action="store_true",
        help="Rebuild the mesh before a full solver run.",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print run result as JSON.",
    )

    report_parser = subparsers.add_parser(
        "report-case",
        help="Read drag/lift result data from a generated case.",
    )
    report_parser.add_argument("case", type=Path, help="Path to a generated case folder.")
    report_parser.add_argument(
        "--json",
        action="store_true",
        help="Print case report as JSON.",
    )

    args = parser.parse_args(argv)

    if args.command == "check":
        report = inspect_stl(args.model)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.to_text())
        return 0 if report.is_cfd_candidate else 2

    if args.command == "init-case":
        case_path = create_case(
            model_path=args.model,
            case_name=args.name,
            speed_mph=args.speed_mph,
            flow_axis=args.flow_axis,
            cases_dir=args.cases_dir,
            include_ground=args.ground,
            moving_ground=args.moving_ground,
            ground_clearance_m=args.ground_clearance_mm / 1000.0,
            generate_openfoam=not args.metadata_only,
            unit_scale=UNIT_SCALES[args.units],
            unit_label=args.units,
            reference_area_m2=args.reference_area_m2,
            reference_length_m=args.reference_length_m,
            measured_length_m=args.measured_length_m,
            measured_width_m=args.measured_width_m,
            measured_height_m=args.measured_height_m,
            smallest_aero_feature_m=(
                args.smallest_aero_feature_mm / 1000.0
                if args.smallest_aero_feature_mm is not None
                else None
            ),
            quality=args.quality,
            simulation_mode=args.simulation_mode,
            source_flow_direction=args.source_flow_direction,
            source_up_direction=args.source_up_direction,
            model_rotation_degrees=(args.rotate_x_deg, args.rotate_y_deg, args.rotate_z_deg),
        )
        print(f"Created case: {case_path}")
        if args.metadata_only:
            print("Next step: generate solver files from this case metadata.")
        else:
            print("OpenFOAM case files generated.")
            print("Next step: run the case through OpenFOAM in WSL2 or Docker.")
        return 0

    if args.command == "app":
        run_app(host=args.host, port=args.port, root=args.root)
        return 0

    if args.command == "solver-status":
        status = solver_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Preferred backend: {status.get('preferredBackend') or 'none'}")
            for name, backend in status["backends"].items():
                available = "yes" if backend.get("available") else "no"
                print(f"{name}: {available}")
        return 0 if status.get("preferredBackend") else 2

    if args.command == "run-case":
        result = run_case(
            args.case,
            backend=args.backend,
            timeout_seconds=args.timeout_seconds,
            run_mode=args.mode,
            reuse_mesh=not args.no_reuse_mesh,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Backend: {result.backend}")
            print(f"Mode: {result.run_mode}")
            print(f"Reused mesh: {'yes' if result.reused_mesh else 'no'}")
            print(f"Return code: {result.returncode}")
            print(f"Verified: {'yes' if result.trusted else 'no'}")
            print(f"Log: {result.log_path}")
            force_coeffs = result.report.get("forceCoeffs")
            if force_coeffs:
                print(f"Mean Cd: {force_coeffs.get('meanCd')}")
                print(f"Mean Cl: {force_coeffs.get('meanCl')}")
            else:
                print("No force coefficient output found yet.")
        return 0 if result.ok else 1

    if args.command == "report-case":
        report = case_report(args.case)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Case: {report.get('caseName')}")
            print(f"Status: {report.get('status')}")
            assessment = report.get("qualityAssessment") or {}
            print(f"Verified: {'yes' if assessment.get('trusted') else 'no'}")
            force_coeffs = report.get("forceCoeffs")
            if force_coeffs:
                print(f"Mean Cd: {force_coeffs.get('meanCd')}")
                print(f"Mean Cl: {force_coeffs.get('meanCl')}")
                print(f"Source: {force_coeffs.get('file')}")
            else:
                print("No force coefficient output found.")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
