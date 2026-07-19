import tempfile
import unittest
from pathlib import Path

from aerolab.cli import main
from aerolab.report import render_html, render_markdown


def _sample_report() -> dict[str, object]:
    return {
        "caseName": "demo-70mph",
        "status": "openfoam_case_generated",
        "sourceModelPath": "/models/demo.stl",
        "caseSetup": {
            "flow": {
                "axis": "x",
                "speed_mph": 70.0,
                "speed_mps": 31.3,
                "mach_number": 0.091,
                "reynolds_number": 9.4e6,
            },
            "ground": {"enabled": True, "moving": True, "clearance_m": 0.02},
            "quality": {"name": "standard"},
            "simulationType": "steady_external_incompressible_airflow",
        },
        "aerodynamicReference": {
            "area_m2": 3.9,
            "length_m": 4.5,
            "area_source": "manual",
            "length_source": "manual",
        },
        "forceCoeffs": {"meanCd": 0.394, "meanCl": -0.12, "file": "/x/forceCoeffs.dat"},
        "aerodynamicForces": {
            "dragN": 938.7,
            "dragLbf": 211.0,
            "verticalForceType": "downforce",
            "verticalForceN": 285.0,
            "verticalForceLbf": 64.0,
        },
        "meshQuality": {
            "cells": 1200000,
            "maxAspectRatio": 12.3,
            "maxNonOrthogonality": 55.0,
            "maxSkewness": 2.1,
        },
        "layerCoverage": {"averageLayers": 4.2, "requestedLayers": 5},
        "residuals": {"stable": True},
        "yPlus": {"body": {"average": 60.0}, "target": 80.0},
        "qualityAssessment": {
            "trusted": False,
            "checks": [
                {"label": "checkMesh", "status": "pass", "detail": "Mesh OK."},
                {"label": "Residuals", "status": "fail", "detail": "p not converged."},
            ],
        },
        "gridConvergence": {"validated": True, "recommendedCd": 0.39, "recommendedCl": -0.11},
        "lastRun": {
            "backend": "wsl",
            "mode": "full",
            "startedAt": "2026-07-19T00:00:00Z",
            "finishedAt": "2026-07-19T00:10:00Z",
        },
    }


class RenderMarkdownTests(unittest.TestCase):
    def test_contains_headline_metrics(self):
        md = render_markdown(_sample_report())
        self.assertIn("# AeroLab CFD Report", md)
        self.assertIn("demo-70mph", md)
        self.assertIn("0.394", md)  # mean Cd
        self.assertIn("938.7 N", md)  # drag in newtons
        self.assertIn("211 lbf", md)  # drag in pounds-force
        self.assertIn("Downforce", md)  # vertical force labelled by type

    def test_surfaces_grid_convergence_when_validated(self):
        md = render_markdown(_sample_report())
        self.assertIn("Grid-converged Cd", md)

    def test_checks_table(self):
        md = render_markdown(_sample_report())
        self.assertIn("checkMesh", md)
        self.assertIn("Residuals", md)
        self.assertIn("pass", md)
        self.assertIn("fail", md)

    def test_graceful_on_empty_report(self):
        md = render_markdown({})  # must not raise
        self.assertIn("AeroLab CFD Report", md)
        self.assertIn("—", md)  # missing values become em dashes

    def test_hides_convergence_when_not_validated(self):
        report = _sample_report()
        report["gridConvergence"] = {"validated": False}
        self.assertNotIn("Grid-converged Cd", render_markdown(report))


class RenderHtmlTests(unittest.TestCase):
    def test_valid_document(self):
        html_out = render_html(_sample_report())
        self.assertTrue(html_out.lstrip().startswith("<!doctype html>"))
        self.assertIn("<title>", html_out)
        self.assertIn("demo-70mph", html_out)
        self.assertIn("938.7", html_out)

    def test_escapes_untrusted_case_name(self):
        report = _sample_report()
        report["caseName"] = "a<b>&c"
        html_out = render_html(report)
        self.assertIn("a&lt;b&gt;&amp;c", html_out)
        self.assertNotIn("a<b>", html_out)


class CliReportTests(unittest.TestCase):
    def test_markdown_output_written_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            (case / "case.json").write_text(
                '{"name": "cli-demo", "status": "metadata_created"}', encoding="utf-8"
            )
            out = Path(tmp) / "report.md"
            code = main(["report-case", str(case), "--format", "markdown", "--output", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())
            self.assertIn("cli-demo", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
