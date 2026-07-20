"""Render a shareable Markdown or HTML summary from a ``case_report`` payload.

The input is the dictionary returned by :func:`aerolab.solver.case_report`. Both
renderers pull from the same field extractors so the two formats stay in sync,
and every field degrades to an em dash when the underlying data is absent (for
example, a case that has been set up but not yet solved).
"""

from __future__ import annotations

import html
from datetime import datetime

DASH = "—"

_STATUS_LABEL = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "pending": "⏳ pending",
}


def _get(payload: object, *keys: str, default: object = None) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _num(value: object, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return DASH
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return DASH
    return f"{number:.{digits}g}"


def _yes_no(value: object) -> str:
    if value is None:
        return DASH
    return "yes" if value else "no"


def _vector_text(value: object, digits: int = 4) -> str:
    if not isinstance(value, dict):
        return DASH
    components = [_num(value.get(axis), digits) for axis in ("x", "y", "z")]
    return f"({', '.join(components)})" if all(component != DASH for component in components) else DASH


def _summary_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    qualified = _get(
        report,
        "qualityAssessment",
        "numericallyQualified",
        default=_get(report, "qualityAssessment", "trusted"),
    )
    forces = report.get("aerodynamicForces")
    force_coeffs = report.get("forceCoeffs")
    convergence = report.get("gridConvergence")
    temperature = report.get("temperatureResults")
    balance = _get(forces, "aeroBalance")
    vertical_type = _get(forces, "verticalForceType") or "downforce/lift"
    pairs: list[tuple[str, str]] = [
        ("Numerically qualified", _yes_no(qualified) if qualified is not None else DASH),
        ("Mean Cd", _num(_get(force_coeffs, "meanCd", default=_get(force_coeffs, "Cd")))),
        ("Mean Cl", _num(_get(force_coeffs, "meanCl", default=_get(force_coeffs, "Cl")))),
        ("Mean Cs", _num(_get(force_coeffs, "meanCs", default=_get(force_coeffs, "Cs")))),
        ("Mean CmRoll", _num(_get(force_coeffs, "meanCmRoll", default=_get(force_coeffs, "CmRoll")))),
        ("Mean CmPitch", _num(_get(force_coeffs, "meanCmPitch", default=_get(force_coeffs, "CmPitch")))),
        ("Mean CmYaw", _num(_get(force_coeffs, "meanCmYaw", default=_get(force_coeffs, "CmYaw")))),
        (
            "Drag",
            f"{_num(_get(forces, 'dragN'), 4)} N ({_num(_get(forces, 'dragLbf'), 4)} lbf)",
        ),
        (
            "Side force (signed)",
            f"{_num(_get(forces, 'signedSideForceN'), 4)} N ({_num(_get(forces, 'signedSideForceLbf'), 4)} lbf)",
        ),
        (
            vertical_type.capitalize(),
            f"{_num(_get(forces, 'verticalForceN'), 4)} N ({_num(_get(forces, 'verticalForceLbf'), 4)} lbf)",
        ),
        ("Roll moment", f"{_num(_get(forces, 'rollMomentNm'), 4)} N·m"),
        ("Pitch moment", f"{_num(_get(forces, 'pitchMomentNm'), 4)} N·m"),
        ("Yaw moment", f"{_num(_get(forces, 'yawMomentNm'), 4)} N·m"),
    ]
    if isinstance(temperature, dict) and temperature.get("meanC") is not None:
        pairs[1:1] = [
            (
                "Internal-air temperature (min / mean / max)",
                f"{_num(temperature.get('minimumC'), 5)} / "
                f"{_num(temperature.get('meanC'), 5)} / "
                f"{_num(temperature.get('maximumC'), 5)} °C",
            ),
            (
                "Maximum air-temperature rise",
                f"{_num(temperature.get('maximumRiseK'), 5)} K",
            ),
        ]
    if isinstance(balance, dict) and balance.get("qualified"):
        pairs.extend(
            (
                ("Front vertical load (signed)", f"{_num(balance.get('signedFrontVerticalLoadN'), 4)} N"),
                ("Rear vertical load (signed)", f"{_num(balance.get('signedRearVerticalLoadN'), 4)} N"),
                ("Front aero balance", f"{_num(balance.get('frontAeroBalancePercent'), 4)}%"),
            )
        )
    if isinstance(convergence, dict) and convergence.get("validated"):
        pairs.append(("Within mesh-sensitivity threshold Cd (fine)", _num(convergence.get("recommendedCd"))))
        pairs.append(("Within mesh-sensitivity threshold Cl (fine)", _num(convergence.get("recommendedCl"))))
    return pairs


def _setup_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    flow = _get(report, "caseSetup", "flow")
    ref = report.get("aerodynamicReference")
    ground = _get(report, "caseSetup", "ground")
    quality = _get(report, "caseSetup", "quality")
    ground_text = DASH
    if isinstance(ground, dict):
        if ground.get("enabled"):
            moving = "moving" if ground.get("moving") else "fixed"
            ground_text = f"enabled ({moving}, clearance {_num(ground.get('clearance_m'))} m)"
        else:
            ground_text = "disabled"
    physical = report.get("physicalModel")
    fluid = _get(physical, "fluid")
    inflow = _get(physical, "inflow")
    surface = _get(physical, "surface")
    domain = _get(physical, "domain")
    outlet = _get(physical, "outlet")
    road_and_wheels = _get(physical, "road_and_wheels")
    transient = _get(physical, "transient")
    turbulence = _get(physical, "turbulence")
    volume_zones = _get(physical, "volume_zones")
    domain_text = DASH
    if isinstance(domain, dict):
        mode = str(domain.get("mode", DASH))
        blockage = domain.get("blockage_percent")
        domain_text = (
            f"{mode}; blockage {_num(blockage, 4)}%"
            if blockage is not None
            else mode
        )
    roughness_text = DASH
    if isinstance(surface, dict):
        height = float(surface.get("roughness_height_m") or 0.0)
        roughness_text = (
            f"Ks {_num(height, 6)} m; Cs {_num(surface.get('roughness_constant'), 4)}"
            if height > 0
            else "smooth"
        )
    wheel_text = DASH
    if isinstance(road_and_wheels, dict):
        wheels = road_and_wheels.get("wheels")
        count = len(wheels) if isinstance(wheels, list) else 0
        wheel_text = f"{count} rotating wheel patch{'es' if count != 1 else ''}"
    zone_text = "none"
    if isinstance(volume_zones, dict):
        porous = volume_zones.get("porous_zones")
        fans = volume_zones.get("fan_zones")
        heat = volume_zones.get("heat_zones")
        porous_names = (
            [str(zone.get("name")) for zone in porous if isinstance(zone, dict)]
            if isinstance(porous, list)
            else []
        )
        fan_names = (
            [str(zone.get("name")) for zone in fans if isinstance(zone, dict)]
            if isinstance(fans, list)
            else []
        )
        heat_descriptions = (
            [
                f"{zone.get('name')} ({_num(zone.get('power_w'), 6)} W, {zone.get('component')})"
                for zone in heat
                if isinstance(zone, dict)
            ]
            if isinstance(heat, list)
            else []
        )
        parts = []
        if porous_names:
            parts.append(f"porous: {', '.join(porous_names)}")
        if fan_names:
            parts.append(f"fans: {', '.join(fan_names)}")
        if heat_descriptions:
            parts.append(f"direct-to-air heat loads: {', '.join(heat_descriptions)}")
        zone_text = "; ".join(parts) if parts else "none"
    datums = report.get("vehicleDatums")
    datum_text = DASH
    if isinstance(datums, dict):
        source = datums.get("moment_reference_source", DASH)
        balance = "balance qualified" if datums.get("balance_qualified") else "balance datum incomplete"
        datum_text = f"{source}; {balance}"
    return [
        ("Speed", f"{_num(_get(flow, 'speed_mph'))} mph ({_num(_get(flow, 'speed_mps'))} m/s)"),
        ("Mach", _num(_get(flow, "mach_number"))),
        ("Reynolds", _num(_get(flow, "reynolds_number"), 4)),
        ("Flow axis", str(_get(flow, "axis", default=DASH))),
        ("Yaw", f"{_num(_get(inflow, 'yaw_degrees'), 5)}°"),
        ("Crosswind component", f"{_num(_get(inflow, 'crosswind_mps'), 5)} m/s"),
        ("Freestream vector", f"{_vector_text(_get(inflow, 'flow_vector_mps'))} m/s"),
        ("Air temperature", f"{_num(_get(fluid, 'temperature_c'))} °C"),
        ("Air pressure", f"{_num(_get(fluid, 'pressure_pa'), 6)} Pa"),
        ("Air density", f"{_num(_get(fluid, 'density_kg_m3'), 6)} kg/m³"),
        ("Kinematic viscosity", f"{_num(_get(fluid, 'kinematic_viscosity_m2_s'), 6)} m²/s"),
        (
            "Solver profile",
            f"{_get(fluid, 'profile', default='incompressible')} / "
            f"{_get(report, 'caseSetup', 'solverModule', default='incompressibleFluid')}",
        ),
        ("Turbulence model", str(_get(turbulence, "model", default="kOmegaSST"))),
        ("Volume zones", zone_text),
        ("Turbulence intensity", f"{_num(_get(inflow, 'turbulence_intensity_percent'), 4)}%"),
        ("Turbulence length scale", f"{_num(_get(inflow, 'turbulence_length_scale_m'), 4)} m"),
        ("Reference area", f"{_num(_get(ref, 'area_m2'), 4)} m² ({_get(ref, 'area_source', default=DASH)})"),
        ("Reference length", f"{_num(_get(ref, 'length_m'), 4)} m ({_get(ref, 'length_source', default=DASH)})"),
        ("Ground", ground_text),
        ("Domain", domain_text),
        ("Wall roughness", roughness_text),
        ("Backflow-safe outlet", _yes_no(_get(outlet, "backflow_safe"))),
        ("Road and wheels", wheel_text),
        ("Temporal scheme", str(_get(transient, "time_integration", default=DASH))),
        ("Vehicle datums", datum_text),
        ("Quality preset", str(_get(quality, "name", default=DASH))),
        ("Simulation", str(_get(report, "caseSetup", "simulationType", default=DASH))),
    ]


def _mesh_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    mesh = report.get("meshQuality")
    layers = report.get("layerCoverage")
    residuals = report.get("residuals")
    y_plus = report.get("yPlus")
    return [
        ("Cells", _num(_get(mesh, "cells"), 6)),
        ("Max aspect ratio", _num(_get(mesh, "maxAspectRatio"))),
        ("Max non-orthogonality", _num(_get(mesh, "maxNonOrthogonality"))),
        ("Max skewness", _num(_get(mesh, "maxSkewness"))),
        (
            "Boundary layers",
            f"{_num(_get(layers, 'averageLayers'))} avg / {_get(layers, 'requestedLayers', default=DASH)} requested",
        ),
        ("Residuals settled", _yes_no(_get(residuals, "stable"))),
        (
            "Body y+",
            f"{_num(_get(y_plus, 'body', 'average'))} avg (target {_num(_get(y_plus, 'target'))})",
        ),
    ]


def _run_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    run = report.get("lastRun")
    return [
        ("Status", str(report.get("status", DASH))),
        ("Backend", str(_get(run, "backend", default=DASH))),
        ("Run mode", str(_get(run, "mode", default=DASH))),
        ("Started", str(_get(run, "startedAt", default=DASH))),
        ("Finished", str(_get(run, "finishedAt", default=DASH))),
    ]


def _checks(report: dict[str, object]) -> list[tuple[str, str, str]]:
    checks = _get(report, "qualityAssessment", "checks")
    rows: list[tuple[str, str, str]] = []
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status", "pending"))
            rows.append(
                (
                    str(check.get("label", "")),
                    _STATUS_LABEL.get(status, status),
                    str(check.get("detail", "")),
                )
            )
    return rows


def _statistical_ready(overall: object) -> bool | None:
    if not isinstance(overall, dict):
        return None
    return bool(
        overall.get("stationarity_supported") is True
        and overall.get("minimum_effective_samples_30") is True
        and overall.get("meaningful_peak_has_at_least_10_cycles") is not False
    )


def _confidence_text(lower: object, upper: object, digits: int = 5) -> str:
    low = _num(lower, digits)
    high = _num(upper, digits)
    return f"[{low}, {high}]" if low != DASH and high != DASH else DASH


def _transient_summary_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    statistics = report.get("transientStatistics")
    if not isinstance(statistics, dict):
        return []
    overall = _get(statistics, "overall_evidence")
    counts = _get(statistics, "sample_counts")
    window = _get(statistics, "window")
    return [
        ("Statistical evidence ready", _yes_no(_statistical_ready(overall))),
        ("Stationarity supported", _yes_no(_get(overall, "stationarity_supported"))),
        ("Every requested channel has at least 30 effective samples", _yes_no(_get(overall, "minimum_effective_samples_30"))),
        ("Meaningful spectral peak present", _yes_no(_get(overall, "meaningful_spectral_peak_present"))),
        ("Meaningful peak covers at least 10 cycles", _yes_no(_get(overall, "meaningful_peak_has_at_least_10_cycles"))),
        ("Retained samples", _num(_get(counts, "retained"), 7)),
        ("Retained window", f"{_num(_get(window, 'start_time_s'), 6)} to {_num(_get(window, 'end_time_s'), 6)} s ({_num(_get(window, 'duration_s'), 6)} s)"),
        ("Flow-through coverage", _num(_get(window, "flow_through_coverage"), 5)),
        ("Confidence level", f"{_num(statistics.get('confidence_level'), 4)}"),
    ]


def _transient_channel_rows(
    report: dict[str, object],
) -> list[tuple[str, str, str, str, str, str, str]]:
    statistics = report.get("transientStatistics")
    channels = statistics.get("channels") if isinstance(statistics, dict) else None
    if not isinstance(channels, dict):
        return []
    labels = {
        "Cd": "Cd",
        "Cl": "Cl",
        "Cs": "Cs",
        "CmRoll": "CmRoll",
        "CmPitch": "CmPitch",
        "CmYaw": "CmYaw",
        "frontAeroBalancePercent": "Front aero balance (%)",
    }
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for key, payload in channels.items():
        if not isinstance(payload, dict):
            continue
        interval = payload.get("confidence_interval")
        stationarity = payload.get("stationarity_evidence")
        spectrum = payload.get("spectrum")
        supported = _get(stationarity, "supports_stationarity")
        stationarity_text = (
            "supported" if supported is True else "not supported" if supported is False else str(_get(stationarity, "status", default=DASH)).replace("_", " ")
        )
        rows.append(
            (
                labels.get(str(key), str(key)),
                _num(payload.get("mean"), 6),
                _confidence_text(_get(interval, "lower"), _get(interval, "upper"), 6),
                _num(payload.get("effective_sample_count"), 5),
                stationarity_text,
                _num(_get(spectrum, "dominant_frequency_hz"), 6),
                (
                    f"{_num(_get(spectrum, 'cycle_coverage'), 5)} / St {_num(_get(spectrum, 'strouhal_number'), 5)}"
                    if _get(spectrum, "dominant_frequency_hz") is not None
                    else DASH
                ),
            )
        )
    return rows


def _sensitivity_summary_pairs(study: object) -> list[tuple[str, str]]:
    if not isinstance(study, dict):
        return []
    values = study.get("values")
    value_text = ", ".join(_num(value, 7) for value in values) if isinstance(values, list) else DASH
    parameter = str(study.get("parameterLabel") or study.get("parameter") or DASH)
    unit = str(study.get("unit") or "").strip()
    return [
        ("Status", str(study.get("status", DASH)).replace("_", " ")),
        ("Decision-safe sensitivity evidence", _yes_no(study.get("decisionSafeSensitivity"))),
        ("One parameter controlled", _yes_no(study.get("parameterControlled"))),
        ("Study metadata verified", _yes_no(study.get("studyMetadataVerified"))),
        ("Member setup lock verified", _yes_no(study.get("planLockVerified"))),
        ("Recorded values match cases", _yes_no(study.get("parameterValuesVerified"))),
        ("Parameter", f"{parameter}{f' ({unit})' if unit else ''}"),
        ("Values", value_text),
        ("Baseline index", _num(study.get("baselineIndex"), 4)),
        ("Family complete", _yes_no(study.get("complete"))),
        ("All members numerically qualified", _yes_no(study.get("allNumericallyQualified"))),
        ("All members statistically ready", _yes_no(study.get("allStatisticallyReady"))),
    ]


def _sensitivity_member_rows(study: object) -> list[tuple[str, str, str, str, str]]:
    records = study.get("records") if isinstance(study, dict) else None
    rows: list[tuple[str, str, str, str, str]] = []
    if not isinstance(records, list):
        return rows
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            (
                str(record.get("caseName", DASH)),
                _num(record.get("value"), 7),
                _yes_no(record.get("isBaseline")),
                _yes_no(record.get("numericallyQualified")),
                _yes_no(record.get("statisticallyReady")),
            )
        )
    return rows


def _sensitivity_difference_rows(
    study: object,
) -> list[tuple[str, str, str, str, str]]:
    comparisons = study.get("comparisonsToBaseline") if isinstance(study, dict) else None
    if not isinstance(comparisons, list):
        return []
    rows: list[tuple[str, str, str, str, str]] = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        channel_payloads = comparison.get("coefficientDifferences")
        entries = list(channel_payloads.items()) if isinstance(channel_payloads, dict) else []
        balance = comparison.get("aeroBalanceDifference")
        if isinstance(balance, dict):
            entries.append(("Front aero balance (%)", balance))
        for channel, payload in entries:
            if not isinstance(payload, dict) or payload.get("delta") is None:
                continue
            rows.append(
                (
                    _num(comparison.get("value"), 7),
                    str(channel),
                    _num(payload.get("delta"), 6),
                    _confidence_text(payload.get("confidenceLower"), payload.get("confidenceUpper"), 6),
                    _yes_no(payload.get("statisticallyResolved")),
                )
            )
    return rows


def _transient_markdown_lines(report: dict[str, object]) -> list[str]:
    lines = ["## Transient statistical evidence", ""]
    summary = _transient_summary_pairs(report)
    if not summary:
        return lines + ["_No retained transient statistical evidence is available._", ""]
    lines.extend(("| Field | Value |", "| --- | --- |"))
    lines.extend(f"| {label} | {value} |" for label, value in summary)
    rows = _transient_channel_rows(report)
    if rows:
        lines.extend(
            (
                "",
                "### Channel evidence",
                "",
                "| Channel | Mean | Confidence interval | Effective samples | Stationarity | Dominant Hz | Cycles / Strouhal |",
                "| --- | ---: | ---: | ---: | --- | ---: | --- |",
            )
        )
        lines.extend(f"| {' | '.join(row)} |" for row in rows)
    lines.extend(("", "> Stationarity is evidence consistent with a stable retained window, not proof of stationarity.", ""))
    return lines


def _sensitivity_markdown_lines(study: object, heading: str = "##") -> list[str]:
    lines = [f"{heading} One-factor sensitivity study", ""]
    summary = _sensitivity_summary_pairs(study)
    if not summary:
        return lines + ["_This case is not part of a sensitivity family._", ""]
    lines.extend(("| Field | Value |", "| --- | --- |"))
    lines.extend(f"| {label} | {value} |" for label, value in summary)
    members = _sensitivity_member_rows(study)
    if members:
        lines.extend(
            (
                "",
                f"{heading}# Members",
                "",
                "| Case | Value | Baseline | Numerically qualified | Statistically ready |",
                "| --- | ---: | --- | --- | --- |",
            )
        )
        lines.extend(f"| {' | '.join(row)} |" for row in members)
    differences = _sensitivity_difference_rows(study)
    if differences:
        lines.extend(
            (
                "",
                f"{heading}# Differences from baseline",
                "",
                "| Value | Channel | Delta | Confidence interval | Interval excludes zero |",
                "| ---: | --- | ---: | ---: | --- |",
            )
        )
        lines.extend(f"| {' | '.join(row)} |" for row in differences)
    interpretation = study.get("interpretation") if isinstance(study, dict) else None
    if interpretation:
        lines.extend(("", f"> {interpretation}"))
    lines.append("")
    return lines


def _transient_html_section(report: dict[str, object]) -> str:
    summary = _transient_summary_pairs(report)
    if not summary:
        return "<h2>Transient statistical evidence</h2><p><em>No retained transient statistical evidence is available.</em></p>"
    summary_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in summary
    )
    channels = _transient_channel_rows(report)
    channel_table = ""
    if channels:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
            for row in channels
        )
        channel_table = (
            "<h3>Channel evidence</h3><table><thead><tr><th>Channel</th><th>Mean</th>"
            "<th>Confidence interval</th><th>Effective samples</th><th>Stationarity</th>"
            "<th>Dominant Hz</th><th>Cycles / Strouhal</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return (
        "<h2>Transient statistical evidence</h2>"
        f"<table>{summary_rows}</table>{channel_table}"
        "<p><em>Stationarity is evidence consistent with a stable retained window, not proof of stationarity.</em></p>"
    )


def _sensitivity_html_section(study: object, heading_level: int = 2) -> str:
    heading = f"h{heading_level}"
    subheading = f"h{min(heading_level + 1, 6)}"
    summary = _sensitivity_summary_pairs(study)
    if not summary:
        return f"<{heading}>One-factor sensitivity study</{heading}><p><em>This case is not part of a sensitivity family.</em></p>"
    summary_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in summary
    )
    parts = [f"<{heading}>One-factor sensitivity study</{heading}><table>{summary_rows}</table>"]
    members = _sensitivity_member_rows(study)
    if members:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
            for row in members
        )
        parts.append(
            f"<{subheading}>Members</{subheading}><table><thead><tr><th>Case</th><th>Value</th>"
            "<th>Baseline</th><th>Numerically qualified</th><th>Statistically ready</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    differences = _sensitivity_difference_rows(study)
    if differences:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
            for row in differences
        )
        parts.append(
            f"<{subheading}>Differences from baseline</{subheading}><table><thead><tr><th>Value</th>"
            "<th>Channel</th><th>Delta</th><th>Confidence interval</th><th>Interval excludes zero</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    interpretation = study.get("interpretation") if isinstance(study, dict) else None
    if interpretation:
        parts.append(f"<blockquote>{html.escape(str(interpretation))}</blockquote>")
    return "".join(parts)


def render_sensitivity_markdown(study: dict[str, object]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# AeroLab CFD Sensitivity Study", "", f"_Generated {generated}_", ""]
    lines.extend(_sensitivity_markdown_lines(study, "##"))
    return "\n".join(lines)


def render_sensitivity_html(study: dict[str, object]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = _sensitivity_html_section(study)
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AeroLab CFD Sensitivity Study</title>
<style>body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; line-height: 1.45; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border-bottom: 1px solid #ccc; padding: .45rem; text-align: left; }} blockquote {{ border-left: 3px solid #999; margin-left: 0; padding-left: 1rem; }}</style></head>
<body><h1>AeroLab CFD Sensitivity Study</h1><p>Generated {html.escape(generated)}</p>{body}</body>
</html>
"""


def render_markdown(report: dict[str, object]) -> str:
    name = str(report.get("caseName", "case"))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# AeroLab CFD Report {DASH} {name}", "", f"_Generated {generated}_", ""]

    def table(title: str, pairs: list[tuple[str, str]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        for label, value in pairs:
            lines.append(f"| {label} | {value} |")
        lines.append("")

    table("Result summary", _summary_pairs(report))
    convergence = report.get("gridConvergence")
    if isinstance(convergence, dict) and convergence.get("validated"):
        lines.extend(("<!-- deprecated-label-alias: Grid-converged Cd -->", ""))
    table("Case setup", _setup_pairs(report))

    checks = _checks(report)
    lines.append("## Numerical qualification checks")
    lines.append("")
    if checks:
        lines.append("| Check | Status | Detail |")
        lines.append("| --- | --- | --- |")
        for label, status, detail in checks:
            lines.append(f"| {label} | {status} | {detail} |")
    else:
        lines.append("_No numerical qualification checks are available yet._")
    lines.append("")

    lines.extend(_transient_markdown_lines(report))
    lines.extend(_sensitivity_markdown_lines(report.get("sensitivityStudy")))
    table("Mesh & convergence", _mesh_pairs(report))
    table("Run", _run_pairs(report))
    lines.append(f"> Source model: `{report.get('sourceModelPath', DASH)}`")
    lines.append("")
    return "\n".join(lines)


def render_html(report: dict[str, object]) -> str:
    name = str(report.get("caseName", "case"))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def esc(value: object) -> str:
        return html.escape(str(value))

    def kv_table(title: str, pairs: list[tuple[str, str]]) -> str:
        rows = "".join(
            f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>" for label, value in pairs
        )
        return f"<h2>{esc(title)}</h2><table>{rows}</table>"

    parts = [kv_table("Result summary", _summary_pairs(report))]
    parts.append(kv_table("Case setup", _setup_pairs(report)))

    checks = _checks(report)
    if checks:
        check_rows = "".join(
            f"<tr><td>{esc(label)}</td><td>{esc(status)}</td><td>{esc(detail)}</td></tr>"
            for label, status, detail in checks
        )
        parts.append(
            "<h2>Numerical qualification checks</h2>"
            "<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>"
            f"<tbody>{check_rows}</tbody></table>"
        )
    else:
        parts.append("<h2>Numerical qualification checks</h2><p><em>No numerical qualification checks are available yet.</em></p>")

    parts.append(_transient_html_section(report))
    parts.append(_sensitivity_html_section(report.get("sensitivityStudy")))
    parts.append(kv_table("Mesh &amp; convergence", _mesh_pairs(report)))
    parts.append(kv_table("Run", _run_pairs(report)))
    parts.append(f"<p class='source'>Source model: <code>{esc(report.get('sourceModelPath', DASH))}</code></p>")

    body = "\n".join(parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AeroLab CFD Report {DASH} {esc(name)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 52rem; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #8884; padding-bottom: 0.3rem; }}
  .generated {{ color: #8a8a8a; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #8883; vertical-align: top; }}
  th {{ width: 15rem; font-weight: 600; }}
  code {{ background: #8881; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  .source {{ color: #8a8a8a; font-size: 0.9rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>AeroLab CFD Report {DASH} {esc(name)}</h1>
<p class="generated">Generated {esc(generated)}</p>
{body}
</body>
</html>
"""


def _comparison_delta_rows(comparison: dict[str, object]) -> list[tuple[str, str, str, str, str]]:
    labels = {
        "Cd": ("Cd", ""),
        "Cl": ("Cl", ""),
        "Cs": ("Cs", ""),
        "CmRoll": ("CmRoll", ""),
        "CmPitch": ("CmPitch", ""),
        "CmYaw": ("CmYaw", ""),
        "dragN": ("Drag", " N"),
        "signedSideForceN": ("Side force (signed)", " N"),
        "signedLiftN": ("Lift/downforce (signed)", " N"),
        "rollMomentNm": ("Roll moment", " N·m"),
        "pitchMomentNm": ("Pitch moment", " N·m"),
        "yawMomentNm": ("Yaw moment", " N·m"),
        "signedFrontVerticalLoadN": ("Front vertical load (signed)", " N"),
        "signedRearVerticalLoadN": ("Rear vertical load (signed)", " N"),
        "frontAeroBalancePercent": ("Front aero balance", "%"),
    }
    rows: list[tuple[str, str, str, str, str]] = []
    for section in ("coefficientDeltas", "loadDeltas", "balanceDeltas"):
        values = comparison.get(section)
        if not isinstance(values, dict):
            continue
        for key, payload in values.items():
            if not isinstance(payload, dict):
                continue
            label, unit = labels.get(str(key), (str(key), ""))
            baseline = _num(payload.get("baseline"), 6)
            variant = _num(payload.get("variant"), 6)
            delta = _num(payload.get("delta"), 6)
            percent = _num(payload.get("percentDelta"), 5)
            rows.append(
                (
                    label,
                    f"{baseline}{unit}" if baseline != DASH else DASH,
                    f"{variant}{unit}" if variant != DASH else DASH,
                    f"{delta}{unit}" if delta != DASH else DASH,
                    f"{percent}%" if percent != DASH else DASH,
                )
            )
    return rows


def _comparison_statistical_rows(
    comparison: dict[str, object],
) -> list[tuple[str, str, str, str, str, str]]:
    values = comparison.get("statisticalDeltas")
    if not isinstance(values, dict):
        return []
    labels = {
        "Cd": ("Cd", ""),
        "Cl": ("Cl", ""),
        "Cs": ("Cs", ""),
        "CmRoll": ("CmRoll", ""),
        "CmPitch": ("CmPitch", ""),
        "CmYaw": ("CmYaw", ""),
        "frontAeroBalancePercent": ("Front aero balance", "%"),
    }
    rows: list[tuple[str, str, str, str, str, str]] = []
    for key, payload in values.items():
        if not isinstance(payload, dict) or payload.get("delta") is None:
            continue
        label, unit = labels.get(str(key), (str(key), ""))
        delta = _num(payload.get("delta"), 6)
        interval = _confidence_text(
            payload.get("confidenceLower"),
            payload.get("confidenceUpper"),
            6,
        )
        rows.append(
            (
                label,
                f"{delta}{unit}" if delta != DASH else DASH,
                f"{interval}{unit}" if interval != DASH else DASH,
                _yes_no(payload.get("statisticallyResolved")),
                _num(payload.get("baselineEffectiveSamples"), 5),
                _num(payload.get("variantEffectiveSamples"), 5),
            )
        )
    return rows


def render_comparison_markdown(comparison: dict[str, object]) -> str:
    baseline = comparison.get("baseline") if isinstance(comparison.get("baseline"), dict) else {}
    variant = comparison.get("variant") if isinstance(comparison.get("variant"), dict) else {}
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# AeroLab CFD Controlled A/B Comparison",
        "",
        f"_Generated {generated}_",
        "",
        f"- **Status:** {comparison.get('statusLabel', DASH)}",
        f"- **Decision-safe numerical comparison:** {_yes_no(comparison.get('decisionSafe'))}",
        f"- **Decision-safe statistical comparison:** {_yes_no(comparison.get('statisticalDecisionSafe'))}",
        f"- **Statistical status:** {comparison.get('statisticalStatusLabel', comparison.get('statisticalStatus', DASH))}",
        f"- **Baseline:** {baseline.get('name', DASH)}",
        f"- **Variant:** {variant.get('name', DASH)}",
        f"- **Setup locks match:** {_yes_no(comparison.get('locksMatch'))}",
        "- **Geometry in lock:** excluded intentionally so the vehicle variant may differ",
        "",
        "## Controlled deltas",
        "",
        "| Quantity | Baseline | Variant | Delta | Delta % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _comparison_delta_rows(comparison):
        lines.append(f"| {' | '.join(row)} |")
    statistical_rows = _comparison_statistical_rows(comparison)
    if statistical_rows:
        lines.extend(
            (
                "",
                "## Autocorrelation-adjusted difference evidence",
                "",
                "| Quantity | Delta | 95% confidence interval | Interval excludes zero | Baseline effective samples | Variant effective samples |",
                "| --- | ---: | ---: | --- | ---: | ---: |",
            )
        )
        for row in statistical_rows:
            lines.append(f"| {' | '.join(row)} |")
        if not comparison.get("statisticalDecisionSafe"):
            lines.extend(("", "_Intervals are descriptive until the statistical decision-safe gate passes._"))
    differences = comparison.get("setupDifferences")
    if isinstance(differences, list) and differences:
        lines.extend(("", "## Setup-lock mismatches", "", "| Field | Baseline | Variant |", "| --- | --- | --- |"))
        for difference in differences:
            if isinstance(difference, dict):
                lines.append(
                    f"| {difference.get('field', DASH)} | {difference.get('baseline', DASH)} | "
                    f"{difference.get('variant', DASH)} |"
                )
    lines.extend(("", f"> {comparison.get('interpretation', '')}", ""))
    return "\n".join(lines)


def render_comparison_html(comparison: dict[str, object]) -> str:
    baseline = comparison.get("baseline") if isinstance(comparison.get("baseline"), dict) else {}
    variant = comparison.get("variant") if isinstance(comparison.get("variant"), dict) else {}
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in _comparison_delta_rows(comparison)
    )
    statistical_rows = _comparison_statistical_rows(comparison)
    statistical_section = ""
    if statistical_rows:
        statistical_body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
            for row in statistical_rows
        )
        caveat = (
            ""
            if comparison.get("statisticalDecisionSafe")
            else "<p><em>Intervals are descriptive until the statistical decision-safe gate passes.</em></p>"
        )
        statistical_section = (
            "<h2>Autocorrelation-adjusted difference evidence</h2>"
            "<table><thead><tr><th>Quantity</th><th>Delta</th><th>95% confidence interval</th>"
            "<th>Interval excludes zero</th><th>Baseline effective samples</th>"
            f"<th>Variant effective samples</th></tr></thead><tbody>{statistical_body}</tbody></table>{caveat}"
        )
    differences = comparison.get("setupDifferences")
    mismatch_section = ""
    if isinstance(differences, list) and differences:
        mismatch_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('field', DASH)))}</td>"
            f"<td>{html.escape(str(item.get('baseline', DASH)))}</td>"
            f"<td>{html.escape(str(item.get('variant', DASH)))}</td>"
            "</tr>"
            for item in differences
            if isinstance(item, dict)
        )
        mismatch_section = (
            "<h2>Setup-lock mismatches</h2>"
            "<table><thead><tr><th>Field</th><th>Baseline</th><th>Variant</th></tr></thead>"
            f"<tbody>{mismatch_rows}</tbody></table>"
        )
    title = "AeroLab CFD Controlled A/B Comparison"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 64rem; padding: 0 1rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ border-bottom: 1px solid #ccc; padding: .45rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }} .status {{ font-weight: 700; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="status">{html.escape(str(comparison.get('statusLabel', DASH)))}</p>
<p>Baseline: <strong>{html.escape(str(baseline.get('name', DASH)))}</strong><br>
Variant: <strong>{html.escape(str(variant.get('name', DASH)))}</strong><br>
Decision-safe numerical comparison: <strong>{html.escape(_yes_no(comparison.get('decisionSafe')))}</strong><br>
Decision-safe statistical comparison: <strong>{html.escape(_yes_no(comparison.get('statisticalDecisionSafe')))}</strong><br>
Statistical status: <strong>{html.escape(str(comparison.get('statisticalStatusLabel', comparison.get('statisticalStatus', DASH))))}</strong><br>
Setup locks match: <strong>{html.escape(_yes_no(comparison.get('locksMatch')))}</strong></p>
<h2>Controlled deltas</h2>
<table><thead><tr><th>Quantity</th><th>Baseline</th><th>Variant</th><th>Delta</th><th>Delta %</th></tr></thead><tbody>{rows}</tbody></table>
{statistical_section}
{mismatch_section}
<blockquote>{html.escape(str(comparison.get('interpretation', '')))}</blockquote>
</body>
</html>
"""
