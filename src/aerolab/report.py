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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DASH
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return DASH
    return f"{number:.{digits}g}"


def _yes_no(value: object) -> str:
    if value is None:
        return DASH
    return "yes" if value else "no"


def _summary_pairs(report: dict[str, object]) -> list[tuple[str, str]]:
    trusted = _get(report, "qualityAssessment", "trusted")
    forces = report.get("aerodynamicForces")
    force_coeffs = report.get("forceCoeffs")
    convergence = report.get("gridConvergence")
    vertical_type = _get(forces, "verticalForceType") or "downforce/lift"
    pairs: list[tuple[str, str]] = [
        ("Verified", _yes_no(trusted) if trusted is not None else DASH),
        ("Mean Cd", _num(_get(force_coeffs, "meanCd", default=_get(force_coeffs, "Cd")))),
        ("Mean Cl", _num(_get(force_coeffs, "meanCl", default=_get(force_coeffs, "Cl")))),
        (
            "Drag",
            f"{_num(_get(forces, 'dragN'), 4)} N ({_num(_get(forces, 'dragLbf'), 4)} lbf)",
        ),
        (
            vertical_type.capitalize(),
            f"{_num(_get(forces, 'verticalForceN'), 4)} N ({_num(_get(forces, 'verticalForceLbf'), 4)} lbf)",
        ),
    ]
    if isinstance(convergence, dict) and convergence.get("validated"):
        pairs.append(("Grid-converged Cd (fine)", _num(convergence.get("recommendedCd"))))
        pairs.append(("Grid-converged Cl (fine)", _num(convergence.get("recommendedCl"))))
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
    return [
        ("Speed", f"{_num(_get(flow, 'speed_mph'))} mph ({_num(_get(flow, 'speed_mps'))} m/s)"),
        ("Mach", _num(_get(flow, "mach_number"))),
        ("Reynolds", _num(_get(flow, "reynolds_number"), 4)),
        ("Flow axis", str(_get(flow, "axis", default=DASH))),
        ("Reference area", f"{_num(_get(ref, 'area_m2'), 4)} m² ({_get(ref, 'area_source', default=DASH)})"),
        ("Reference length", f"{_num(_get(ref, 'length_m'), 4)} m ({_get(ref, 'length_source', default=DASH)})"),
        ("Ground", ground_text),
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
    table("Case setup", _setup_pairs(report))

    checks = _checks(report)
    lines.append("## Verification checks")
    lines.append("")
    if checks:
        lines.append("| Check | Status | Detail |")
        lines.append("| --- | --- | --- |")
        for label, status, detail in checks:
            lines.append(f"| {label} | {status} | {detail} |")
    else:
        lines.append("_No verification checks are available yet._")
    lines.append("")

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
            "<h2>Verification checks</h2>"
            "<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>"
            f"<tbody>{check_rows}</tbody></table>"
        )
    else:
        parts.append("<h2>Verification checks</h2><p><em>No verification checks are available yet.</em></p>")

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
