from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from .case import create_case
from .report import (
    render_comparison_html,
    render_comparison_markdown,
    render_html,
    render_markdown,
    render_sensitivity_html,
    render_sensitivity_markdown,
)
from .solver import (
    SENSITIVITY_PARAMETERS,
    case_report,
    compare_cases,
    create_sensitivity_study_from_case,
    normalize_file_handler,
    normalize_process_request,
    normalize_study_process_budget,
    run_case,
    run_study,
    sensitivity_study_report,
    solver_status,
)
from .stl import inspect_stl
from .webapp import run_app

UNIT_SCALES = {
    "m": 1.0,
    "mm": 0.001,
    "cm": 0.01,
    "in": 0.0254,
}


def _text_number(value: object, digits: int = 6) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _print_budget_recommendation(value: object) -> None:
    if not isinstance(value, dict):
        return
    print(f"Budget guidance: {value.get('title') or 'Safer workstation budget'}")
    if value.get("detail"):
        print(f"  {value['detail']}")
    if isinstance(value.get("recommendedProcesses"), int):
        print(f"  Recommended processes: {value['recommendedProcesses']}")
    if isinstance(value.get("recommendedProcessBudget"), int):
        print(f"  Recommended study process budget: {value['recommendedProcessBudget']}")
    if isinstance(value.get("safeCellBudget"), int):
        print(f"  Conservative cell allowance: {value['safeCellBudget']:,}")
    if isinstance(value.get("configuredCellBudget"), int):
        print(f"  Configured case cell cap: {value['configuredCellBudget']:,}")
    if value.get("suggestedQuality"):
        print(
            "  Explicit mesh alternative (changes fidelity): "
            f"{str(value['suggestedQuality']).title()}"
        )


def _report_text(report: dict[str, object]) -> str:
    lines = [
        f"Case: {report.get('caseName')}",
        f"Status: {report.get('status')}",
    ]
    assessment = report.get("qualityAssessment") or {}
    qualified = isinstance(assessment, dict) and assessment.get(
        "numericallyQualified",
        assessment.get("trusted"),
    )
    lines.append(f"Numerically qualified: {'yes' if qualified else 'no'}")
    force_coeffs = report.get("forceCoeffs")
    if isinstance(force_coeffs, dict):
        for key in ("Cd", "Cl", "Cs", "CmRoll", "CmPitch", "CmYaw"):
            lines.append(f"Mean {key}: {force_coeffs.get(f'mean{key}', force_coeffs.get(key))}")
        lines.append(f"Source: {force_coeffs.get('file')}")
    else:
        lines.append("No force or moment coefficient output found.")

    temperature = report.get("temperatureResults")
    if isinstance(temperature, dict) and temperature.get("meanC") is not None:
        lines.extend(
            (
                "Internal-air temperature:",
                f"- minimum / mean / maximum: {_text_number(temperature.get('minimumC'))} / "
                f"{_text_number(temperature.get('meanC'))} / "
                f"{_text_number(temperature.get('maximumC'))} C",
                f"- maximum rise above inlet: {_text_number(temperature.get('maximumRiseK'))} K",
                f"- field: {temperature.get('field')} at time {temperature.get('time')}",
            )
        )

    statistics = report.get("transientStatistics")
    if isinstance(statistics, dict):
        overall = statistics.get("overall_evidence")
        counts = statistics.get("sample_counts")
        lines.append("Transient statistical evidence:")
        if isinstance(counts, dict):
            lines.append(f"- retained samples: {counts.get('retained')}")
        if isinstance(overall, dict):
            ready = bool(
                overall.get("stationarity_supported") is True
                and overall.get("minimum_effective_samples_30") is True
                and overall.get("meaningful_peak_has_at_least_10_cycles") is not False
            )
            lines.extend(
                (
                    f"- evidence ready: {'yes' if ready else 'no'}",
                    f"- stationarity supported: {overall.get('stationarity_supported')}",
                    f"- every channel has at least 30 effective samples: {overall.get('minimum_effective_samples_30')}",
                    f"- meaningful peak has at least 10 cycles: {overall.get('meaningful_peak_has_at_least_10_cycles')}",
                )
            )
        channels = statistics.get("channels")
        if isinstance(channels, dict):
            for key, channel in channels.items():
                if not isinstance(channel, dict):
                    continue
                interval = channel.get("confidence_interval")
                stationarity = channel.get("stationarity_evidence")
                spectrum = channel.get("spectrum")
                lines.append(
                    f"- {key}: mean {_text_number(channel.get('mean'))}; "
                    f"95% CI [{_text_number(interval.get('lower') if isinstance(interval, dict) else None)}, "
                    f"{_text_number(interval.get('upper') if isinstance(interval, dict) else None)}]; "
                    f"effective samples {_text_number(channel.get('effective_sample_count'))}; "
                    f"stationarity {stationarity.get('status') if isinstance(stationarity, dict) else 'n/a'}; "
                    f"dominant {_text_number(spectrum.get('dominant_frequency_hz') if isinstance(spectrum, dict) else None)} Hz; "
                    f"cycles {_text_number(spectrum.get('cycle_coverage') if isinstance(spectrum, dict) else None)}"
                )

    study = report.get("sensitivityStudy")
    if isinstance(study, dict):
        lines.extend(
            (
                "Sensitivity study:",
                f"- parameter: {study.get('parameterLabel') or study.get('parameter')} ({study.get('unit') or ''})",
                f"- status: {study.get('status')}",
                f"- complete: {'yes' if study.get('complete') else 'no'}",
                f"- numerically qualified family: {'yes' if study.get('allNumericallyQualified') else 'no'}",
                f"- statistically ready family: {'yes' if study.get('allStatisticallyReady') else 'no'}",
                f"- decision-safe sensitivity: {'yes' if study.get('decisionSafeSensitivity') else 'no'}",
            )
        )
    return "\n".join(lines)


def _center_of_gravity(
    x: float | None,
    y: float | None,
    z: float | None,
) -> tuple[float, float, float] | None:
    provided = sum(value is not None for value in (x, y, z))
    if provided == 0:
        return None
    if provided != 3:
        raise ValueError("Vehicle CG requires --cg-x-m, --cg-y-m, and --cg-z-m together.")
    return (float(x), float(y), float(z))  # type: ignore[arg-type]


def _closed_tunnel_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    values = {
        "width_m": args.tunnel_width_m,
        "height_m": args.tunnel_height_m,
        "upstream_m": args.tunnel_upstream_m,
        "downstream_m": args.tunnel_downstream_m,
    }
    provided = sum(value is not None for value in values.values())
    if provided == 0:
        return None
    if provided != len(values):
        raise ValueError(
            "Closed tunnel requires --tunnel-width-m, --tunnel-height-m, "
            "--tunnel-upstream-m, and --tunnel-downstream-m together."
        )
    closed_tunnel: dict[str, object] = {
        key: float(value) for key, value in values.items()
    }
    return closed_tunnel


def _wheel_setup_from_file(path: Path | None) -> list[dict[str, object]] | None:
    if path is None:
        return None
    config_path = path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    wheels = payload.get("wheels") if isinstance(payload, dict) else payload
    if not isinstance(wheels, list):
        raise ValueError("Wheel configuration must be a JSON list or an object with a wheels list.")
    normalized: list[dict[str, object]] = []
    for index, value in enumerate(wheels, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"Wheel configuration entry {index} must be an object.")
        wheel = dict(value)
        model_value = wheel.get("model_path") or wheel.get("geometry")
        if model_value:
            model_path = Path(str(model_value)).expanduser()
            if not model_path.is_absolute():
                model_path = config_path.parent / model_path
            wheel["model_path"] = str(model_path.resolve())
        normalized.append(wheel)
    return normalized


def _volume_zones_from_file(
    path: Path | None,
    key: str,
    label: str,
) -> list[dict[str, object]] | None:
    if path is None:
        return None
    config_path = path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    zones = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(zones, list):
        raise ValueError(f"{label} configuration must be a JSON list or an object with a {key} list.")
    normalized: list[dict[str, object]] = []
    for index, value in enumerate(zones, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"{label} configuration entry {index} must be an object.")
        normalized.append(dict(value))
    return normalized


def _comparison_text(comparison: dict[str, object]) -> str:
    baseline_value = comparison.get("baseline")
    baseline = baseline_value if isinstance(baseline_value, dict) else {}
    variant_value = comparison.get("variant")
    variant = variant_value if isinstance(variant_value, dict) else {}
    lines = [
        f"Status: {comparison.get('statusLabel')}",
        f"Decision-safe numerical comparison: {'yes' if comparison.get('decisionSafe') else 'no'}",
        f"Decision-safe statistical comparison: {'yes' if comparison.get('statisticalDecisionSafe') else 'no'}",
        f"Statistical status: {comparison.get('statisticalStatusLabel') or comparison.get('statisticalStatus')}",
        f"Baseline: {baseline.get('name')}",
        f"Variant: {variant.get('name')}",
        f"Setup locks match: {'yes' if comparison.get('locksMatch') else 'no'}",
    ]
    deltas = comparison.get("coefficientDeltas")
    if isinstance(deltas, dict):
        for key, payload in deltas.items():
            if isinstance(payload, dict):
                lines.append(
                    f"Delta {key}: {payload.get('delta')} ({payload.get('percentDelta')}%)"
                )
    statistical = comparison.get("statisticalDeltas")
    if isinstance(statistical, dict):
        lines.append("Autocorrelation-adjusted difference evidence:")
        for key, payload in statistical.items():
            if not isinstance(payload, dict) or payload.get("delta") is None:
                continue
            lines.append(
                f"- {key}: delta {_text_number(payload.get('delta'))}; "
                f"95% CI [{_text_number(payload.get('confidenceLower'))}, "
                f"{_text_number(payload.get('confidenceUpper'))}]; "
                f"interval excludes zero {payload.get('statisticallyResolved')}; "
                f"effective samples {_text_number(payload.get('baselineEffectiveSamples'))} / "
                f"{_text_number(payload.get('variantEffectiveSamples'))}"
            )
    differences = comparison.get("setupDifferences")
    if isinstance(differences, list) and differences:
        lines.append("Setup mismatches:")
        for difference in differences:
            if isinstance(difference, dict):
                lines.append(
                    f"- {difference.get('field')}: {difference.get('baseline')} -> "
                    f"{difference.get('variant')}"
                )
    lines.append(str(comparison.get("interpretation") or ""))
    return "\n".join(lines)


def _sensitivity_text(study: dict[str, object]) -> str:
    lines = [
        f"Study: {study.get('studyId')}",
        f"Parameter: {study.get('parameterLabel') or study.get('parameter')} ({study.get('unit') or ''})",
        f"Values: {study.get('values')}",
        f"Status: {study.get('status')}",
        f"One parameter controlled: {'yes' if study.get('parameterControlled') else 'no'}",
        f"Study metadata verified: {'yes' if study.get('studyMetadataVerified') else 'no'}",
        f"Member setup lock verified: {'yes' if study.get('planLockVerified') else 'no'}",
        f"Recorded values match cases: {'yes' if study.get('parameterValuesVerified') else 'no'}",
        f"Family complete: {'yes' if study.get('complete') else 'no'}",
        f"All numerically qualified: {'yes' if study.get('allNumericallyQualified') else 'no'}",
        f"All statistically ready: {'yes' if study.get('allStatisticallyReady') else 'no'}",
        f"Decision-safe sensitivity: {'yes' if study.get('decisionSafeSensitivity') else 'no'}",
    ]
    records = study.get("records")
    if isinstance(records, list):
        lines.append("Members:")
        for record in records:
            if isinstance(record, dict):
                lines.append(
                    f"- {record.get('value')}: {record.get('caseName')}"
                    f"{' [baseline]' if record.get('isBaseline') else ''}; "
                    f"numerical={'yes' if record.get('numericallyQualified') else 'no'}, "
                    f"statistical={'yes' if record.get('statisticallyReady') else 'no'}"
                )
    lines.append(str(study.get("interpretation") or ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Reports contain Unicode (em dash, superscripts, status glyphs); keep console
    # output from crashing on legacy code pages such as Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

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
        "--air-temperature-c",
        type=float,
        help="Dry-air temperature in degrees Celsius; with pressure, derives density and viscosity.",
    )
    case_parser.add_argument(
        "--air-pressure-pa",
        type=float,
        help="Absolute dry-air pressure in pascals.",
    )
    case_parser.add_argument(
        "--air-density-kg-m3",
        type=float,
        help="Manual air-density override in kilograms per cubic meter.",
    )
    case_parser.add_argument(
        "--kinematic-viscosity-m2-s",
        type=float,
        help="Manual kinematic-viscosity override in square meters per second.",
    )
    case_parser.add_argument(
        "--turbulence-intensity-percent",
        type=float,
        help="Inlet turbulence intensity in percent (default: 1).",
    )
    case_parser.add_argument(
        "--turbulence-length-scale-m",
        type=float,
        help="Inlet turbulence length scale in meters (default: 7%% of reference length).",
    )
    case_parser.add_argument(
        "--yaw-deg",
        type=float,
        help="Yaw angle in degrees; positive rotates primary flow toward cross(lift, primary).",
    )
    case_parser.add_argument(
        "--crosswind-mps",
        type=float,
        help="Signed crosswind component in m/s; derives yaw from the total freestream speed.",
    )
    case_parser.add_argument(
        "--roughness-height-mm",
        type=float,
        default=0.0,
        help="Equivalent body and wheel roughness height Ks in millimeters.",
    )
    case_parser.add_argument(
        "--roughness-constant",
        type=float,
        default=0.5,
        help="Foundation nutkRoughWallFunction constant Cs (0.5 to 1.0).",
    )
    case_parser.add_argument(
        "--backflow-safe-outlet",
        action="store_true",
        help="Use pressure-inlet/outlet velocity and inletOutlet turbulence at the outlet.",
    )
    case_parser.add_argument(
        "--second-order-transient",
        action="store_true",
        help="Use the second-order backward time scheme; requires --simulation-mode transient.",
    )
    case_parser.add_argument(
        "--fluid-profile",
        choices=["incompressible", "compressible_thermal"],
        default="incompressible",
        help="Use the byte-compatible incompressible profile or Foundation v13 fluid with thermal/compressible fields.",
    )
    case_parser.add_argument(
        "--turbulence-model",
        choices=["kOmegaSST", "SpalartAllmarasDES", "SpalartAllmarasIDDES"],
        default="kOmegaSST",
        help="Momentum-transport model; DES and IDDES require transient mode and smooth walls.",
    )
    case_parser.add_argument(
        "--porous-zones-config",
        type=Path,
        help="JSON list of explicit solver-coordinate porous box zones.",
    )
    case_parser.add_argument(
        "--fan-zones-config",
        type=Path,
        help="JSON list of explicit solver-coordinate actuation-disk box zones.",
    )
    case_parser.add_argument(
        "--heat-zones-config",
        type=Path,
        help=(
            "JSON list of solver-coordinate box heat loads; each entry uses "
            "component and exactly one of power_w or power_kw."
        ),
    )
    case_parser.add_argument(
        "--tunnel-width-m",
        type=float,
        help="Closed-tunnel internal width in solver meters.",
    )
    case_parser.add_argument(
        "--tunnel-height-m",
        type=float,
        help="Closed-tunnel internal height above the road in solver meters.",
    )
    case_parser.add_argument(
        "--tunnel-upstream-m",
        type=float,
        help="Closed-tunnel distance upstream of the transformed model bounds.",
    )
    case_parser.add_argument(
        "--tunnel-downstream-m",
        type=float,
        help="Closed-tunnel distance downstream of the transformed model bounds.",
    )
    case_parser.add_argument(
        "--wheel-config",
        type=Path,
        help="JSON file defining separate wheel STLs, source-frame centers/axes, and radii.",
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
        "--cg-x-m",
        type=float,
        help="Vehicle CG X coordinate in transformed solver meters.",
    )
    case_parser.add_argument(
        "--cg-y-m",
        type=float,
        help="Vehicle CG Y coordinate in transformed solver meters.",
    )
    case_parser.add_argument(
        "--cg-z-m",
        type=float,
        help="Vehicle CG Z coordinate in transformed solver meters.",
    )
    case_parser.add_argument(
        "--front-axle-station-m",
        type=float,
        help="Front axle station along the positive solver flow axis in meters.",
    )
    case_parser.add_argument(
        "--rear-axle-station-m",
        type=float,
        help="Rear axle station along the positive solver flow axis in meters.",
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
        "--processes",
        type=normalize_process_request,
        default="auto",
        metavar="auto|N",
        help="MPI processes: auto-select for backend hardware, or use a positive integer (1 is serial).",
    )
    run_parser.add_argument(
        "--file-handler",
        type=normalize_file_handler,
        default="auto",
        metavar="auto|uncollated|collated|masterUncollated",
        help="OpenFOAM parallel file handler; auto preserves the backend default.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible failed full run from its latest reconstructed time.",
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

    study_run_parser = subparsers.add_parser(
        "run-study",
        help="Run every member of an accuracy or sensitivity study within one resource budget.",
    )
    study_run_parser.add_argument(
        "case",
        type=Path,
        help="Path to any member of the study.",
    )
    study_run_parser.add_argument(
        "--backend",
        choices=["auto", "native", "wsl", "docker"],
        default="auto",
        help="OpenFOAM backend shared by all study members.",
    )
    study_run_parser.add_argument(
        "--processes",
        type=normalize_process_request,
        default="auto",
        metavar="auto|N",
        help="MPI processes per active case; auto balances ranks against concurrency.",
    )
    study_run_parser.add_argument(
        "--process-budget",
        type=normalize_study_process_budget,
        default="auto",
        metavar="auto|N",
        help="Maximum aggregate processes across concurrently running study cases.",
    )
    study_run_parser.add_argument(
        "--file-handler",
        type=normalize_file_handler,
        default="auto",
        metavar="auto|uncollated|collated|masterUncollated",
        help="OpenFOAM parallel file handler for every member.",
    )
    study_run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum runtime for each study member.",
    )
    study_run_parser.add_argument(
        "--mode",
        choices=["full", "mesh"],
        default="full",
        help="Validate each mesh or run each complete solver workflow.",
    )
    study_run_parser.add_argument(
        "--no-reuse-mesh",
        action="store_true",
        help="Rebuild meshes instead of reusing compatible validated meshes.",
    )
    study_run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the aggregate study result as JSON.",
    )

    report_parser = subparsers.add_parser(
        "report-case",
        help="Read drag/lift result data from a generated case.",
    )
    report_parser.add_argument("case", type=Path, help="Path to a generated case folder.")
    report_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format for the case report.",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this file instead of standard output.",
    )
    report_parser.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --format json.",
    )

    compare_parser = subparsers.add_parser(
        "compare-cases",
        help="Compare a qualified variant against a setup-locked qualified baseline.",
    )
    compare_parser.add_argument("baseline", type=Path, help="Path to the baseline case folder.")
    compare_parser.add_argument("variant", type=Path, help="Path to the variant case folder.")
    compare_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format for the controlled comparison.",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        help="Write the comparison to this file instead of standard output.",
    )

    sensitivity_parser = subparsers.add_parser(
        "create-sensitivity-study",
        help="Create a setup-preserving one-factor family from an existing AeroLab case.",
    )
    sensitivity_parser.add_argument(
        "base_case",
        type=Path,
        help="Existing case whose complete physical and numerical setup will be preserved.",
    )
    sensitivity_parser.add_argument(
        "--parameter",
        required=True,
        choices=sorted(SENSITIVITY_PARAMETERS),
        help="Single input parameter varied across the family.",
    )
    sensitivity_parser.add_argument(
        "--values",
        required=True,
        nargs="+",
        type=float,
        help="Two to twelve unique finite values for the selected parameter.",
    )
    sensitivity_parser.add_argument(
        "--baseline-index",
        type=int,
        help="Zero-based baseline position; defaults to the value nearest the base case.",
    )
    sensitivity_parser.add_argument(
        "--name",
        help="Base name for generated members; defaults to the existing case name.",
    )
    sensitivity_parser.add_argument(
        "--cases-dir",
        type=Path,
        help="Destination directory; defaults to the existing case's parent directory.",
    )
    sensitivity_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Create case metadata without OpenFOAM dictionaries.",
    )
    sensitivity_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the created study descriptor as JSON.",
    )

    sensitivity_report_parser = subparsers.add_parser(
        "report-sensitivity-study",
        help="Collect qualification and confidence-interval evidence for a sensitivity family.",
    )
    sensitivity_report_parser.add_argument(
        "case",
        type=Path,
        help="Any member case in the sensitivity family.",
    )
    sensitivity_report_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format for the sensitivity report.",
    )
    sensitivity_report_parser.add_argument(
        "--output",
        type=Path,
        help="Write the sensitivity report to this file instead of standard output.",
    )

    from .benchmarks import DEFAULT_BENCHMARK_ID, available_benchmarks

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a packaged real-OpenFOAM regression benchmark.",
    )
    benchmark_parser.add_argument(
        "benchmark_id",
        nargs="?",
        choices=available_benchmarks(),
        default=DEFAULT_BENCHMARK_ID,
        help="Packaged benchmark identifier.",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Directory where separate benchmark attempts are archived.",
    )
    benchmark_parser.add_argument(
        "--backend",
        choices=["auto", "native", "wsl", "docker"],
        default="auto",
        help="OpenFOAM backend to verify and use.",
    )
    benchmark_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum real-solver runtime.",
    )
    benchmark_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify the packaged input and generate the deterministic case without running OpenFOAM.",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the benchmark result as JSON.",
    )

    parse_argv = list(argv) if argv is not None else sys.argv[1:]
    benchmark_json_mode = bool(
        parse_argv
        and parse_argv[0] == "benchmark"
        and "--json" in parse_argv
    )
    if benchmark_json_mode:
        parser_error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(parser_error_output):
                args = parser.parse_args(parse_argv)
        except SystemExit as exc:
            if exc.code == 0:
                raise
            error_lines = [
                line.strip()
                for line in parser_error_output.getvalue().splitlines()
                if line.strip()
            ]
            message = error_lines[-1] if error_lines else "Invalid benchmark arguments."
            if ": error: " in message:
                message = message.split(": error: ", 1)[1]
            print(
                json.dumps(
                    {
                        "benchmarkId": None,
                        "status": "error",
                        "passed": False,
                        "error": {
                            "type": "ArgumentError",
                            "message": message,
                        },
                    },
                    indent=2,
                )
            )
            return 2
    else:
        args = parser.parse_args(parse_argv)

    if args.command == "check":
        stl_report = inspect_stl(args.model)
        if args.json:
            print(json.dumps(stl_report.to_dict(), indent=2))
        else:
            print(stl_report.to_text())
        return 0 if stl_report.is_cfd_candidate else 2

    if args.command == "init-case":
        case_path = create_case(
            model_path=args.model,
            case_name=args.name,
            speed_mph=args.speed_mph,
            air_temperature_c=args.air_temperature_c,
            air_pressure_pa=args.air_pressure_pa,
            air_density_kg_m3=args.air_density_kg_m3,
            kinematic_viscosity_m2_s=args.kinematic_viscosity_m2_s,
            turbulence_intensity_percent=args.turbulence_intensity_percent,
            turbulence_length_scale_m=args.turbulence_length_scale_m,
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
            center_of_gravity_m=_center_of_gravity(args.cg_x_m, args.cg_y_m, args.cg_z_m),
            front_axle_station_m=args.front_axle_station_m,
            rear_axle_station_m=args.rear_axle_station_m,
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
            yaw_degrees=args.yaw_deg,
            crosswind_mps=args.crosswind_mps,
            roughness_height_m=args.roughness_height_mm / 1000.0,
            roughness_constant=args.roughness_constant,
            closed_tunnel=_closed_tunnel_from_args(args),
            backflow_safe_outlet=args.backflow_safe_outlet,
            wheel_setup=_wheel_setup_from_file(args.wheel_config),
            second_order_transient=args.second_order_transient,
            fluid_profile=args.fluid_profile,
            turbulence_model=args.turbulence_model,
            porous_zones=_volume_zones_from_file(
                args.porous_zones_config,
                "porous_zones",
                "Porous-zone",
            ),
            fan_zones=_volume_zones_from_file(
                args.fan_zones_config,
                "fan_zones",
                "Fan-zone",
            ),
            heat_zones=_volume_zones_from_file(
                args.heat_zones_config,
                "heat_zones",
                "Heat-load-zone",
            ),
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
        solver_status_payload = solver_status()
        if args.json:
            print(json.dumps(solver_status_payload, indent=2))
        else:
            print(
                "Preferred backend: "
                f"{solver_status_payload.get('preferredBackend') or 'none'}"
            )
            backends = solver_status_payload.get("backends")
            if isinstance(backends, dict):
                for name, backend in backends.items():
                    if not isinstance(backend, dict):
                        continue
                    available = "yes" if backend.get("available") else "no"
                    print(f"{name}: {available}")
        return 0 if solver_status_payload.get("preferredBackend") else 2

    if args.command == "run-case":
        case_run_result = run_case(
            args.case,
            backend=args.backend,
            timeout_seconds=args.timeout_seconds,
            run_mode=args.mode,
            reuse_mesh=not args.no_reuse_mesh,
            processes=args.processes,
            file_handler=args.file_handler,
            resume=args.resume,
        )
        if args.json:
            print(json.dumps(case_run_result.to_dict(), indent=2))
        else:
            print(f"Backend: {case_run_result.backend}")
            print(
                f"Processes: {case_run_result.processes} "
                f"(requested {case_run_result.requested_processes})"
            )
            print(f"File handler: {case_run_result.file_handler}")
            print(f"Mode: {case_run_result.run_mode}")
            print(f"Reused mesh: {'yes' if case_run_result.reused_mesh else 'no'}")
            print(
                "Resumed: "
                f"{'yes, from time ' + _text_number(case_run_result.resume_from_time) if case_run_result.resumed else 'no'}"
            )
            print(f"Return code: {case_run_result.returncode}")
            print(
                "Numerically qualified: "
                f"{'yes' if case_run_result.trusted else 'no'}"
            )
            print(f"Log: {case_run_result.log_path}")
            _print_budget_recommendation(case_run_result.budget_recommendation)
            force_coeffs = case_run_result.report.get("forceCoeffs")
            if isinstance(force_coeffs, dict) and force_coeffs:
                print(f"Mean Cd: {force_coeffs.get('meanCd')}")
                print(f"Mean Cl: {force_coeffs.get('meanCl')}")
            else:
                print("No force coefficient output found yet.")
        return 0 if case_run_result.ok else 1

    if args.command == "run-study":
        study_run_result = run_study(
            args.case,
            backend=args.backend,
            timeout_seconds=args.timeout_seconds,
            run_mode=args.mode,
            reuse_mesh=not args.no_reuse_mesh,
            processes=args.processes,
            process_budget=args.process_budget,
            file_handler=args.file_handler,
        )
        if args.json:
            print(json.dumps(study_run_result, indent=2))
        else:
            study_plan_value = study_run_result.get("plan")
            study_plan = study_plan_value if isinstance(study_plan_value, dict) else {}
            print(f"Study: {study_run_result.get('studyId')}")
            print(f"Status: {study_run_result.get('status')}")
            print(f"Backend: {study_plan.get('backend')}")
            print(
                f"Allocation: {study_plan.get('maxConcurrentCases')} concurrent cases x "
                f"{study_plan.get('processesPerCase')} processes "
                f"(budget {study_plan.get('processBudget')})"
            )
            if study_plan.get("memoryWarning"):
                print(f"Memory warning: {study_plan.get('memoryWarning')}")
            _print_budget_recommendation(study_run_result.get("budgetRecommendation"))
            study_members = study_run_result.get("results")
            if isinstance(study_members, list):
                for study_member in study_members:
                    if isinstance(study_member, dict):
                        member_status = "ok" if study_member.get("ok") else "failed"
                        member_detail = study_member.get("error")
                        print(
                            f"- {study_member.get('casePath')}: {member_status}"
                            + (f" ({member_detail})" if member_detail else "")
                        )
        return 0 if study_run_result.get("ok") else 1

    if args.command == "report-case":
        case_report_payload = case_report(args.case)
        report_format = "json" if args.json else args.format
        if report_format == "json":
            report_content = json.dumps(case_report_payload, indent=2)
        elif report_format == "markdown":
            report_content = render_markdown(case_report_payload)
        elif report_format == "html":
            report_content = render_html(case_report_payload)
        else:
            report_content = _report_text(case_report_payload)
        if args.output:
            args.output.write_text(report_content + "\n", encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(report_content)
        return 0

    if args.command == "compare-cases":
        comparison_payload = compare_cases(args.baseline, args.variant)
        if args.format == "json":
            comparison_content = json.dumps(comparison_payload, indent=2)
        elif args.format == "markdown":
            comparison_content = render_comparison_markdown(comparison_payload)
        elif args.format == "html":
            comparison_content = render_comparison_html(comparison_payload)
        else:
            comparison_content = _comparison_text(comparison_payload)
        if args.output:
            args.output.write_text(comparison_content + "\n", encoding="utf-8")
            print(f"Comparison written to {args.output}")
        else:
            print(comparison_content)
        return 0 if comparison_payload.get("decisionSafe") else 2

    if args.command == "create-sensitivity-study":
        created_study = create_sensitivity_study_from_case(
            base_case_path=args.base_case,
            parameter=args.parameter,
            values=args.values,
            cases_dir=args.cases_dir,
            base_name=args.name,
            generate_openfoam=not args.metadata_only,
            baseline_index=args.baseline_index,
        )
        if args.json:
            print(json.dumps(created_study, indent=2))
        else:
            print(f"Created sensitivity study: {created_study.get('studyId')}")
            print(
                f"Parameter: {created_study.get('parameterLabel')} "
                f"({created_study.get('unit')}) = {created_study.get('values')}"
            )
            created_case_paths = created_study.get("casePaths")
            if isinstance(created_case_paths, list):
                for created_case_path in created_case_paths:
                    print(f"- {created_case_path}")
            print(
                "Selected baseline member: "
                f"{created_study.get('selectedCasePath')}"
            )
        return 0

    if args.command == "report-sensitivity-study":
        sensitivity_report_payload = sensitivity_study_report(args.case)
        if sensitivity_report_payload is None:
            print("The selected case is not part of a sensitivity study.", file=sys.stderr)
            return 2
        if args.format == "json":
            sensitivity_content = json.dumps(sensitivity_report_payload, indent=2)
        elif args.format == "markdown":
            sensitivity_content = render_sensitivity_markdown(sensitivity_report_payload)
        elif args.format == "html":
            sensitivity_content = render_sensitivity_html(sensitivity_report_payload)
        else:
            sensitivity_content = _sensitivity_text(sensitivity_report_payload)
        if args.output:
            args.output.write_text(sensitivity_content + "\n", encoding="utf-8")
            print(f"Sensitivity report written to {args.output}")
        else:
            print(sensitivity_content)
        return 0

    if args.command == "benchmark":
        from .benchmarks import BENCHMARK_UNAVAILABLE_EXIT_CODE, run_benchmark

        try:
            benchmark_result = run_benchmark(
                args.benchmark_id,
                output_dir=args.output_dir,
                backend=args.backend,
                timeout_seconds=args.timeout_seconds,
                prepare_only=args.prepare_only,
            )
        except Exception as exc:
            benchmark_result = {
                "benchmarkId": args.benchmark_id,
                "status": "error",
                "passed": False,
                "requestedBackend": args.backend,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            if args.json:
                print(json.dumps(benchmark_result, indent=2))
            else:
                print(f"Benchmark: {args.benchmark_id}", file=sys.stderr)
                print("Status: error", file=sys.stderr)
                print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(benchmark_result, indent=2))
        else:
            print(f"Benchmark: {benchmark_result.get('benchmarkName')}")
            print(f"Status: {benchmark_result.get('status')}")
            print(
                "Real solver passed: "
                f"{'yes' if benchmark_result.get('passed') else 'no'}"
            )
            print(
                "Absolute aerodynamic accuracy reference: "
                f"{'yes' if benchmark_result.get('absoluteAccuracyValidated') else 'no'}"
            )
            print(f"Case: {benchmark_result.get('casePath')}")
            print(f"Result: {benchmark_result.get('resultPath')}")
            benchmark_checks = benchmark_result.get("checks")
            if isinstance(benchmark_checks, list):
                for benchmark_check in benchmark_checks:
                    if isinstance(benchmark_check, dict):
                        print(
                            f"- {benchmark_check.get('status')}: "
                            f"{benchmark_check.get('label')} - "
                            f"{benchmark_check.get('detail')}"
                        )
        if benchmark_result.get("status") == "prepared":
            return 0
        if benchmark_result.get("status") == "unavailable":
            return BENCHMARK_UNAVAILABLE_EXIT_CODE
        if benchmark_result.get("status") == "error":
            return 2
        return 0 if benchmark_result.get("passed") else 1

    parser.error(f"Unknown command: {args.command}")
    return 2
