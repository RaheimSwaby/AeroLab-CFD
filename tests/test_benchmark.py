from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from aerolab.benchmarks import (
    evaluate_benchmark,
    load_benchmark_manifest,
    run_benchmark,
)
from aerolab.cli import main as cli_main
from aerolab.solver.backends import (
    OPENFOAM_EXECUTABLES,
    _backend_cleanup_command,
    _backend_cleanup_verification_command,
    _docker_container_name,
    _normalized_openfoam_version,
    _run_command,
    openfoam_identity,
    solver_status,
)
from aerolab.solver.run import _confirm_backend_cleanup, _run_solver_process

_EXECUTABLE_SHA256 = "a" * 64


def _toolchain(prefix: str = "/opt/openfoam13/bin") -> dict[str, dict[str, str]]:
    return {
        executable: {
            "path": f"{prefix}/{executable}",
            "sha256": _EXECUTABLE_SHA256,
        }
        for executable in OPENFOAM_EXECUTABLES
    }


def _identity(
    backend: str = "native",
    *,
    prefix: str = "/opt/openfoam13/bin",
    image_id: str | None = None,
) -> dict[str, object]:
    toolchain = _toolchain(prefix)
    return {
        "backend": backend,
        "version": "13",
        "environmentVersion": "13",
        "versionsAgree": True,
        "distribution": "foundation",
        "executable": toolchain["foamRun"]["path"],
        "executableSha256": toolchain["foamRun"]["sha256"],
        "toolchain": toolchain,
        "probeReturncode": 0,
        "imageId": image_id,
    }


def _probe_output(
    *,
    environment_version: str = "13",
    banner_version: str = "13",
    prefix: str = "/opt/openfoam13/bin",
) -> str:
    lines: list[str] = []
    for executable, entry in _toolchain(prefix).items():
        lines.extend(
            (
                f"OPENFOAM_TOOL_{executable}_PATH={entry['path']}",
                f"OPENFOAM_TOOL_{executable}_SHA256={entry['sha256']}",
            )
        )
    lines.extend(
        (
            f"OPENFOAM_VERSION={environment_version}",
            f"Using: OpenFOAM-{banner_version} (see https://openfoam.org)",
        )
    )
    return "\n".join(lines) + "\n"


class BenchmarkTests(unittest.TestCase):
    def test_prepare_only_archives_verified_package_fixture_and_case(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.dict(os.environ, {"AEROLAB_BUILD_REVISION": "test-revision"}),
        ):
            result = run_benchmark(
                output_dir=Path(temp_dir),
                prepare_only=True,
            )

            case_path = Path(str(result["casePath"]))
            result_path = Path(str(result["resultPath"]))
            run_root = Path(str(result["runRoot"]))
            manifest_path = run_root / "manifest.json"
            fixture_path = run_root / "input" / "body.stl"
            seal_path = result_path.with_suffix(".sha256")
            metadata = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
            stored_result = json.loads(result_path.read_text(encoding="utf-8"))
            result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()

            self.assertEqual(result["status"], "prepared")
            self.assertFalse(result["passed"])
            self.assertFalse(result["absoluteAccuracyValidated"])
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(result_path.is_file())
            self.assertTrue(seal_path.is_file())
            self.assertEqual(stored_result["runId"], result["runId"])
            self.assertEqual(stored_result["buildRevision"], "test-revision")
            self.assertEqual(
                result["fixtureSha256"],
                "13b9bef191ac63d1b63deafbbc4d634b71f8875562e048b371e281aeeaa7415f",
            )
            self.assertEqual(
                stored_result["manifestSha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                stored_result["evidenceSha256"]["manifest.json"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                stored_result["evidenceSha256"]["input/body.stl"],
                hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            )
            self.assertNotIn("benchmark-result.json", stored_result["evidenceSha256"])
            self.assertNotIn("benchmark-result.sha256", stored_result["evidenceSha256"])
            self.assertEqual(
                stored_result["evidenceFileCount"],
                len(stored_result["evidenceSha256"]),
            )
            self.assertEqual(
                stored_result["artifacts"]["resultSeal"],
                str(seal_path),
            )
            self.assertEqual(
                seal_path.read_text(encoding="utf-8"),
                f"{result_digest}  benchmark-result.json\n",
            )
            self.assertEqual(metadata["solver_target"], "openfoam-foundation-v13")
            self.assertTrue(metadata["geometry_validation"]["verified"])
            self.assertEqual(metadata["cfd_quality"]["name"], "draft")
            self.assertTrue((case_path / "Allrun").is_file())

    def test_case_generation_failure_is_archived_and_checksum_recorded(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "aerolab.benchmarks.runner.create_case",
                side_effect=RuntimeError("case setup failed"),
            ),
        ):
            result = run_benchmark(
                output_dir=Path(temp_dir),
                prepare_only=True,
            )

            result_path = Path(str(result["resultPath"]))
            seal_path = result_path.with_suffix(".sha256")
            stored_result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "error")
            self.assertFalse(result["passed"])
            self.assertIsNone(result["casePath"])
            self.assertEqual(result["error"]["stage"], "caseGeneration")
            self.assertTrue(result_path.is_file())
            self.assertTrue(seal_path.is_file())
            self.assertIn("manifest.json", stored_result["evidenceSha256"])
            self.assertIn("input/body.stl", stored_result["evidenceSha256"])
            self.assertEqual(
                seal_path.read_text(encoding="utf-8"),
                f"{hashlib.sha256(result_path.read_bytes()).hexdigest()}  benchmark-result.json\n",
            )

    def test_acceptance_requires_qualification_and_symmetric_loads(self) -> None:
        manifest = load_benchmark_manifest("cube-symmetry-v1")
        required = manifest["acceptance"]["requiredQualificationChecks"]
        report = {
            "meshQuality": {"cells": 100_000},
            "forceCoeffs": {
                "meanCd": 1.05,
                "meanCl": 0.01,
                "meanCs": -0.01,
                "meanCmRoll": 0.005,
                "meanCmPitch": -0.005,
                "meanCmYaw": 0.003,
            },
            "qualityAssessment": {
                "numericallyQualified": True,
                "qualificationStatus": "numerically_qualified",
                "checks": [
                    {"label": label, "status": "pass", "detail": "passed"}
                    for label in required
                ],
            },
        }

        accepted = evaluate_benchmark(
            manifest,
            {"ok": True, "returncode": 0, "report": report},
        )
        self.assertTrue(all(check["status"] == "pass" for check in accepted["checks"]))

        report["forceCoeffs"]["meanCl"] = 0.2
        rejected = evaluate_benchmark(
            manifest,
            {"ok": True, "returncode": 0, "report": report},
        )
        lift_check = next(
            check
            for check in rejected["checks"]
            if check["label"] == "Metric: vertical symmetry"
        )
        self.assertEqual(lift_check["status"], "fail")

    def test_cli_can_prepare_benchmark_as_json_without_solver(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                returncode = cli_main(
                    [
                        "benchmark",
                        "cube-symmetry-v1",
                        "--prepare-only",
                        "--output-dir",
                        temp_dir,
                        "--json",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(returncode, 0)
            self.assertEqual(payload["status"], "prepared")
            self.assertTrue(Path(payload["resultPath"]).is_file())

    def test_cli_preserves_json_contract_for_setup_errors(self) -> None:
        output = io.StringIO()
        with (
            mock.patch(
                "aerolab.benchmarks.run_benchmark",
                side_effect=RuntimeError("benchmark setup failed"),
            ),
            contextlib.redirect_stdout(output),
        ):
            returncode = cli_main(["benchmark", "--prepare-only", "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(returncode, 2)
        self.assertEqual(payload["status"], "error")
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["error"]["type"], "RuntimeError")
        self.assertEqual(payload["error"]["message"], "benchmark setup failed")

    def test_cli_preserves_json_contract_for_argument_errors(self) -> None:
        output = io.StringIO()
        error_output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error_output),
        ):
            returncode = cli_main(
                ["benchmark", "--backend", "not-a-backend", "--json"]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(returncode, 2)
        self.assertEqual(error_output.getvalue(), "")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["type"], "ArgumentError")
        self.assertIn("invalid choice", payload["error"]["message"])

    def test_cli_returns_error_exit_code_for_structured_runner_error(self) -> None:
        output = io.StringIO()
        with (
            mock.patch(
                "aerolab.benchmarks.run_benchmark",
                return_value={
                    "benchmarkId": "cube-symmetry-v1",
                    "status": "error",
                    "passed": False,
                    "checks": [],
                },
            ),
            contextlib.redirect_stdout(output),
        ):
            returncode = cli_main(["benchmark", "--json"])

        self.assertEqual(returncode, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_wsl_probe_timeout_is_marked_as_infrastructure_error(self) -> None:
        def which(executable: str) -> str | None:
            return "/usr/bin/wsl" if executable == "wsl" else None

        timeout = subprocess.CompletedProcess(
            ["wsl", "--status"],
            124,
            "",
            "WSL status probe timed out",
        )
        with (
            mock.patch("aerolab.solver.backends.shutil.which", side_effect=which),
            mock.patch(
                "aerolab.solver.backends._run_quick",
                return_value=timeout,
            ),
        ):
            status = solver_status()

        error = status["backends"]["wsl"]["infrastructureError"]
        self.assertEqual(error["stage"], "statusProbe")
        self.assertEqual(error["returncode"], 124)

    def test_docker_image_probe_daemon_failure_is_infrastructure_error(self) -> None:
        def which(executable: str) -> str | None:
            return "/usr/local/bin/docker" if executable == "docker" else None

        daemon_ok = subprocess.CompletedProcess(
            ["docker", "info"],
            0,
            "27.0",
            "",
        )
        image_failed = subprocess.CompletedProcess(
            ["docker", "image", "inspect"],
            1,
            "",
            "image inspection failed",
        )
        daemon_failed = subprocess.CompletedProcess(
            ["docker", "info"],
            1,
            "",
            "Cannot connect to the Docker daemon",
        )
        with (
            mock.patch.dict(
                os.environ,
                {"AEROLAB_OPENFOAM_IMAGE": "local/openfoam:13"},
            ),
            mock.patch("aerolab.solver.backends.shutil.which", side_effect=which),
            mock.patch(
                "aerolab.solver.backends._run_quick",
                side_effect=[daemon_ok, image_failed, daemon_failed],
            ),
        ):
            status = solver_status()

        error = status["backends"]["docker"]["infrastructureError"]
        self.assertEqual(error["stage"], "imageProbeDaemonRecheck")
        self.assertIn("Docker daemon", error["message"])

    def test_backend_infrastructure_failure_is_error_not_unavailable(self) -> None:
        status = {
            "preferredBackend": None,
            "backends": {
                "docker": {
                    "available": False,
                    "infrastructureError": {
                        "stage": "daemonProbe",
                        "returncode": 1,
                        "message": "Cannot connect to the Docker daemon",
                    },
                }
            },
        }
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "aerolab.benchmarks.runner.solver_status",
                return_value=status,
            ),
        ):
            result = run_benchmark(
                output_dir=Path(temp_dir),
                backend="docker",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["stage"], "backendDiscovery")
        self.assertIn("Docker daemon", result["error"]["message"])

    def test_solver_infrastructure_return_codes_are_errors_not_failures(self) -> None:
        status = {
            "preferredBackend": "native",
            "backends": {"native": {"available": True}},
        }
        for infrastructure_code in (96, 127):
            with self.subTest(returncode=infrastructure_code):
                solver_result = mock.Mock(
                    returncode=infrastructure_code,
                    log_path=Path("/tmp/aerolab-test.log"),
                )
                with (
                    tempfile.TemporaryDirectory() as temp_dir,
                    mock.patch(
                        "aerolab.benchmarks.runner.solver_status",
                        return_value=status,
                    ),
                    mock.patch(
                        "aerolab.benchmarks.runner.openfoam_identity",
                        return_value=_identity(),
                    ),
                    mock.patch(
                        "aerolab.benchmarks.runner.run_case",
                        return_value=solver_result,
                    ),
                ):
                    result = run_benchmark(
                        output_dir=Path(temp_dir),
                        backend="native",
                    )

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["stage"], "solverInfrastructure")
                self.assertIn(str(infrastructure_code), result["error"]["message"])

    def test_openfoam_major_version_normalization_requires_explicit_evidence(self) -> None:
        self.assertIsNone(_normalized_openfoam_version("13"))
        self.assertEqual(_normalized_openfoam_version("OPENFOAM_VERSION=13"), "13")
        self.assertIsNone(
            _normalized_openfoam_version("/opt/openfoam13/bin/foamRun")
        )
        self.assertEqual(_normalized_openfoam_version("OpenFOAM Foundation v13"), "13")
        self.assertEqual(_normalized_openfoam_version("OpenFOAM-v2312"), "2312")
        self.assertIsNone(_normalized_openfoam_version("unknown"))

    def test_native_identity_probe_attests_and_pins_complete_toolchain(self) -> None:
        probe = subprocess.CompletedProcess(
            ["bash"],
            0,
            _probe_output(),
            "",
        )
        with mock.patch(
            "aerolab.solver.backends._run_quick",
            return_value=probe,
        ) as run_quick:
            identity = openfoam_identity("native")

        command = run_quick.call_args.args[0]
        self.assertEqual(command[:2], ["bash", "-lc"])
        for executable in OPENFOAM_EXECUTABLES:
            self.assertIn(f"command -v {executable}", command[-1])
        self.assertIn("foamRun -help", command[-1])
        self.assertEqual(identity["version"], "13")
        self.assertEqual(identity["environmentVersion"], "13")
        self.assertTrue(identity["versionsAgree"])
        self.assertEqual(identity["distribution"], "foundation")
        self.assertEqual(identity["executable"], "/opt/openfoam13/bin/foamRun")
        self.assertEqual(identity["executableSha256"], _EXECUTABLE_SHA256)
        self.assertEqual(set(identity["toolchain"]), set(OPENFOAM_EXECUTABLES))
        self.assertEqual(identity["probeReturncode"], 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_command = _run_command(
                Path(temp_dir),
                "native",
                solver_identity=identity,
            )
        for index, executable in enumerate(OPENFOAM_EXECUTABLES):
            self.assertIn(
                f"EXPECTED_TOOL_PATH_{index}=/opt/openfoam13/bin/{executable}",
                run_command[-1],
            )
        self.assertIn(
            "if ! ACTUAL_TOOL_PATH_0=$(command -v foamRun); then",
            run_command[-1],
        )
        self.assertIn("exit 96", run_command[-1])
        self.assertLess(
            run_command[-1].index("EXPECTED_TOOL_PATH_0"),
            run_command[-1].index("./Allrun"),
        )

    def test_execution_guard_rejects_incomplete_attested_toolchain(self) -> None:
        identity = _identity()
        toolchain = identity["toolchain"]
        assert isinstance(toolchain, dict)
        del toolchain["foamPostProcess"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                RuntimeError,
                "foamPostProcess identity is missing",
            ):
                _run_command(
                    Path(temp_dir),
                    "native",
                    solver_identity=identity,
                )

    def test_identity_rejects_environment_and_executable_banner_disagreement(self) -> None:
        probe = subprocess.CompletedProcess(
            ["bash"],
            0,
            _probe_output(environment_version="13", banner_version="12"),
            "",
        )
        with mock.patch(
            "aerolab.solver.backends._run_quick",
            return_value=probe,
        ):
            identity = openfoam_identity("native")

        self.assertIsNone(identity["version"])
        self.assertFalse(identity["versionsAgree"])
        self.assertEqual(identity["environmentVersion"], "13")
        self.assertEqual(identity["distribution"], "foundation")

    def test_docker_identity_records_image_digest_from_exact_environment(self) -> None:
        image = "local/openfoam-foundation:13"
        image_probe = subprocess.CompletedProcess(
            ["docker"],
            0,
            "sha256:verified-image\n",
            "",
        )
        solver_probe = subprocess.CompletedProcess(
            ["docker"],
            0,
            _probe_output(prefix="/opt/openfoam13/platforms/bin"),
            "",
        )
        with (
            mock.patch.dict(os.environ, {"AEROLAB_OPENFOAM_IMAGE": image}),
            mock.patch(
                "aerolab.solver.backends.shutil.which",
                return_value="/usr/local/bin/docker",
            ),
            mock.patch(
                "aerolab.solver.backends._run_quick",
                side_effect=[image_probe, solver_probe],
            ) as run_quick,
        ):
            identity = openfoam_identity("docker")

        commands = [entry.args[0] for entry in run_quick.call_args_list]
        self.assertEqual(
            commands[0],
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        )
        self.assertEqual(
            commands[1][:6],
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "bash",
                "sha256:verified-image",
            ],
        )
        self.assertIn("foamRun -help", commands[1][-1])
        self.assertEqual(identity["version"], "13")
        self.assertEqual(identity["distribution"], "foundation")
        self.assertEqual(identity["image"], image)
        self.assertEqual(identity["imageId"], "sha256:verified-image")
        self.assertEqual(identity["executableSha256"], _EXECUTABLE_SHA256)

    def test_docker_execution_is_identity_pinned_and_has_verified_cleanup(self) -> None:
        image = "local/openfoam-foundation:13"
        identity = _identity(
            "docker",
            image_id="sha256:verified-image",
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.dict(os.environ, {"AEROLAB_OPENFOAM_IMAGE": image}),
        ):
            case_path = Path(temp_dir)
            command = _run_command(
                case_path,
                "docker",
                script_name="Allrun",
                execution_id="test-execution",
                solver_identity=identity,
            )
            expected_name = _docker_container_name(case_path, "test-execution")
            cleanup = _backend_cleanup_command(
                case_path,
                "docker",
                "test-execution",
            )
            verification = _backend_cleanup_verification_command(
                case_path,
                "docker",
                "test-execution",
            )

        self.assertEqual(command[command.index("--name") + 1], expected_name)
        self.assertEqual(command[command.index("--stop-timeout") + 1], "30")
        self.assertEqual(command[command.index("--entrypoint") + 1], "bash")
        self.assertEqual(command[command.index("bash") + 1], "sha256:verified-image")
        self.assertNotIn(image, command)
        self.assertIn(f"{case_path}:/source", command)
        self.assertEqual(command[command.index("--mount") + 1], "type=volume,target=/work")
        self.assertIn("AEROLAB_PROCESSES=1", command)
        self.assertIn("AEROLAB_FILE_HANDLER=auto", command)
        self.assertIn("AEROLAB_RESUME=0", command)
        self.assertIn("SOURCE_CASE=/source", command[-1])
        self.assertIn("STAGE_CASE=/work", command[-1])
        self.assertIn('trap copy_back EXIT', command[-1])
        self.assertIn("EXPECTED_TOOL_PATH_0=/opt/openfoam13/bin/foamRun", command[-1])
        self.assertEqual(
            cleanup,
            ["docker", "rm", "--force", "--volumes", expected_name],
        )
        self.assertEqual(
            verification,
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"name=^/{expected_name}$",
                "--format",
                "{{.ID}}",
            ],
        )

    def test_docker_cleanup_succeeds_only_after_absence_is_confirmed(self) -> None:
        cleanup_failure = subprocess.CompletedProcess(
            ["docker", "rm"],
            1,
            "",
            "No such container",
        )
        absent = subprocess.CompletedProcess(["docker", "ps"], 0, "", "")
        daemon_failure = subprocess.CompletedProcess(
            ["docker", "ps"],
            1,
            "",
            "Cannot connect to the Docker daemon",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            with mock.patch(
                "aerolab.solver.run.subprocess.run",
                side_effect=[cleanup_failure, absent],
            ):
                _confirm_backend_cleanup(case_path, "docker", "verified-absent")

            with mock.patch(
                "aerolab.solver.run.subprocess.run",
                side_effect=[cleanup_failure, daemon_failure],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "container-absence verification failed",
                ):
                    _confirm_backend_cleanup(case_path, "docker", "daemon-failed")

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup requires POSIX.")
    def test_solver_timeout_kills_signal_ignoring_descendants_before_mutation(self) -> None:
        child_script = "\n".join(
            (
                "import signal",
                "import time",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "Path('child-started').write_text('started', encoding='utf-8')",
                "time.sleep(1.5)",
                "Path('late-mutation').write_text('mutated', encoding='utf-8')",
            )
        )
        parent_script = "\n".join(
            (
                "import subprocess",
                "import sys",
                "import time",
                "from pathlib import Path",
                f"subprocess.Popen([sys.executable, '-c', {child_script!r}])",
                "deadline = time.monotonic() + 5",
                "while not Path('child-started').exists() and time.monotonic() < deadline:",
                "    time.sleep(0.01)",
                "time.sleep(10)",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            case_path = Path(temp_dir)
            log_path = case_path / "solver.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                returncode, timed_out = _run_solver_process(
                    [sys.executable, "-c", parent_script],
                    case_path=case_path,
                    backend="native",
                    execution_id="timeout-test",
                    process_timeout=1,
                    log_file=log_file,
                    termination_grace_seconds=0.1,
                )

            self.assertEqual(returncode, 124)
            self.assertTrue(timed_out)
            self.assertTrue((case_path / "child-started").is_file())
            time.sleep(0.8)
            self.assertFalse((case_path / "late-mutation").exists())

    @unittest.skipUnless(
        os.environ.get("AEROLAB_RUN_OPENFOAM_BENCHMARK") == "1",
        "Set AEROLAB_RUN_OPENFOAM_BENCHMARK=1 on a Foundation v13 host.",
    )
    def test_real_foundation_v13_cube_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_benchmark(
                output_dir=Path(temp_dir),
                backend=os.environ.get("AEROLAB_BENCHMARK_BACKEND", "auto"),
                timeout_seconds=int(os.environ.get("AEROLAB_BENCHMARK_TIMEOUT", "3600")),
            )

            self.assertTrue(result["passed"], json.dumps(result, indent=2))
            self.assertEqual(result["openfoamVersion"], "13")
            self.assertEqual(result["openfoamIdentity"]["distribution"], "foundation")
            self.assertTrue(result["openfoamIdentity"]["executable"])
            self.assertRegex(
                result["openfoamIdentity"]["executableSha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                set(result["openfoamIdentity"]["toolchain"]),
                set(OPENFOAM_EXECUTABLES),
            )
            if result["backend"] == "docker":
                self.assertTrue(result["openfoamIdentity"]["imageId"])


if __name__ == "__main__":
    unittest.main()
