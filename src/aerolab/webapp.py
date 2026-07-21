from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import stat
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .case import create_case
from .repair import repair_fidelity_for_model, repair_stl
from .solver import (
    SENSITIVITY_PARAMETERS,
    case_report,
    case_run_progress,
    compare_cases,
    create_sensitivity_study,
    normalize_file_handler,
    normalize_process_request,
    normalize_study_process_budget,
    run_case,
    run_study,
    solver_status,
    study_members,
)
from .stl import detect_aero_features, inspect_stl, mesh_preview


class AeroLabServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], root: Path):
        super().__init__(server_address, AeroLabHandler)
        self.root = root.resolve()
        self.web_root = Path(__file__).parent / "web"
        self.active_runs: dict[Path, threading.Thread] = {}
        self.active_runs_lock = threading.Lock()

    def start_case_run(
        self,
        case_path: Path,
        backend: str,
        timeout_seconds: int,
        run_mode: str,
        reuse_mesh: bool,
        processes: str | int,
        file_handler: str,
        resume: bool,
    ) -> None:
        case_path = case_path.resolve()
        with self.active_runs_lock:
            existing = self.active_runs.get(case_path)
            if existing is not None and existing.is_alive():
                raise ValueError("This case already has an active OpenFOAM run.")
            worker = threading.Thread(
                target=self._run_case_worker,
                args=(
                    case_path,
                    backend,
                    timeout_seconds,
                    run_mode,
                    reuse_mesh,
                    processes,
                    file_handler,
                    resume,
                ),
                name=f"aerolab-{run_mode}-{case_path.name}",
                daemon=True,
            )
            self.active_runs[case_path] = worker
            worker.start()

    def _run_case_worker(
        self,
        case_path: Path,
        backend: str,
        timeout_seconds: int,
        run_mode: str,
        reuse_mesh: bool,
        processes: str | int,
        file_handler: str,
        resume: bool,
    ) -> None:
        try:
            run_case(
                case_path,
                backend=backend,
                timeout_seconds=timeout_seconds,
                run_mode=run_mode,
                reuse_mesh=reuse_mesh,
                processes=processes,
                file_handler=file_handler,
                resume=resume,
            )
        except Exception as exc:
            _record_background_run_failure(case_path, run_mode, backend, exc)
        finally:
            with self.active_runs_lock:
                if self.active_runs.get(case_path) is threading.current_thread():
                    self.active_runs.pop(case_path, None)

    def start_study_run(
        self,
        case_path: Path,
        backend: str,
        timeout_seconds: int,
        run_mode: str,
        reuse_mesh: bool,
        processes: str | int,
        process_budget: str | int,
        file_handler: str,
    ) -> dict[str, object]:
        descriptor = study_members(case_path)
        member_paths = [Path(str(value)).resolve() for value in descriptor["casePaths"]]
        with self.active_runs_lock:
            conflicts = [
                path.name
                for path in member_paths
                if (worker := self.active_runs.get(path)) is not None and worker.is_alive()
            ]
            if conflicts:
                raise ValueError(
                    "Study members already running: " + ", ".join(conflicts)
                )
            worker = threading.Thread(
                target=self._run_study_worker,
                args=(
                    case_path.resolve(),
                    member_paths,
                    backend,
                    timeout_seconds,
                    run_mode,
                    reuse_mesh,
                    processes,
                    process_budget,
                    file_handler,
                ),
                name=f"aerolab-study-{descriptor['studyId']}",
                daemon=True,
            )
            for path in member_paths:
                self.active_runs[path] = worker
            worker.start()
        return descriptor

    def _run_study_worker(
        self,
        case_path: Path,
        member_paths: list[Path],
        backend: str,
        timeout_seconds: int,
        run_mode: str,
        reuse_mesh: bool,
        processes: str | int,
        process_budget: str | int,
        file_handler: str,
    ) -> None:
        try:
            run_study(
                case_path,
                backend=backend,
                timeout_seconds=timeout_seconds,
                run_mode=run_mode,
                reuse_mesh=reuse_mesh,
                processes=processes,
                process_budget=process_budget,
                file_handler=file_handler,
            )
        except Exception as exc:
            _record_background_study_failure(member_paths, backend, exc)
        finally:
            with self.active_runs_lock:
                current = threading.current_thread()
                for path in member_paths:
                    if self.active_runs.get(path) is current:
                        self.active_runs.pop(path, None)


class AeroLabHandler(BaseHTTPRequestHandler):
    server: AeroLabServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(self.server.web_root / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/assets/"):
            asset = self.server.web_root / parsed.path.removeprefix("/assets/")
            self._send_file(asset)
            return
        if parsed.path == "/api/state":
            self._send_json(self._state())
            return
        if parsed.path == "/api/solver":
            self._send_json(solver_status())
            return
        if parsed.path == "/api/case-log":
            try:
                self._handle_case_log(parsed.query)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        if parsed.path == "/api/case-progress":
            try:
                self._handle_case_progress(parsed.query)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        if parsed.path == "/api/study-progress":
            try:
                self._handle_study_progress(parsed.query)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        if parsed.path == "/api/model-file":
            try:
                self._handle_model_file(parsed.query)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/check":
                self._handle_upload_check(parsed.query)
                return
            if parsed.path == "/api/check-model":
                self._handle_check_model()
                return
            if parsed.path == "/api/repair-model":
                self._handle_repair_model()
                return
            if parsed.path == "/api/analyze-features":
                self._handle_analyze_features()
                return
            if parsed.path == "/api/cases":
                self._handle_create_case()
                return
            if parsed.path == "/api/accuracy-study":
                self._handle_create_accuracy_study()
                return
            if parsed.path == "/api/sensitivity-study":
                self._handle_create_sensitivity_study()
                return
            if parsed.path == "/api/run-case":
                self._handle_run_case()
                return
            if parsed.path == "/api/run-study":
                self._handle_run_study()
                return
            if parsed.path == "/api/compare-cases":
                self._handle_compare_cases()
                return
            if parsed.path == "/api/case-report":
                self._handle_case_report()
                return
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_upload_check(self, query: str) -> None:
        params = parse_qs(query)
        filename = params.get("filename", ["model.stl"])[0]
        safe_name = _safe_filename(unquote(filename))
        data = self.rfile.read(_content_length(self.headers.get("Content-Length")))
        if not data:
            raise ValueError("No model file was received.")
        if not safe_name.lower().endswith(".stl"):
            raise ValueError("Only STL files are supported right now.")

        upload_dir = self.server.root / "models" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = upload_dir / f"{stamp}-{safe_name}"
        model_path.write_bytes(data)

        report = inspect_stl(model_path)
        self._send_json(
            {
                "ok": True,
                "modelPath": str(model_path),
                "report": _report_payload(model_path, report),
                "preview": mesh_preview(model_path),
            }
        )

    def _handle_check_model(self) -> None:
        payload = self._json_body()
        model_path = self._resolve_project_path(str(payload["modelPath"]))
        report = inspect_stl(model_path)
        self._send_json(
            {
                "ok": True,
                "modelPath": str(model_path),
                "report": _report_payload(model_path, report),
                "preview": mesh_preview(model_path),
            }
        )

    def _handle_model_file(self, query: str) -> None:
        params = parse_qs(query)
        value = params.get("path", [None])[0]
        if not value:
            raise ValueError("A model path is required.")
        model_path = self._resolve_project_path(unquote(value))
        if model_path.suffix.lower() != ".stl" or not model_path.is_file():
            raise ValueError("Only project STL files can be loaded by the viewer.")
        data = model_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "model/stl")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_case_log(self, query: str) -> None:
        params = parse_qs(query)
        value = params.get("casePath", [None])[0]
        if not value:
            raise ValueError("A case path is required.")
        case_path = self._resolve_project_path(unquote(value))
        cases_root = (self.server.root / "cases").resolve()
        if not case_path.is_dir() or not case_path.is_relative_to(cases_root):
            raise ValueError("The selected folder must be an AeroLab case inside the cases directory.")
        if not (case_path / "case.json").is_file():
            raise ValueError("The selected folder is not an AeroLab case.")

        log_path = case_path / "aerolab-run.log"
        maximum_bytes = 64 * 1024
        open_flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(log_path, open_flags)
        except FileNotFoundError:
            self._send_json(
                {
                    "ok": True,
                    "exists": False,
                    "text": "",
                    "sizeBytes": 0,
                    "shownBytes": 0,
                    "truncated": False,
                    "modifiedAt": None,
                }
            )
            return
        except OSError as exc:
            raise ValueError("The case run log must be a regular case-local file.") from exc

        try:
            log_stat = os.fstat(descriptor)
            if not stat.S_ISREG(log_stat.st_mode):
                raise ValueError("The case run log must be a regular case-local file.")
            try:
                path_stat = log_path.lstat()
                resolved_log = log_path.resolve(strict=True)
                resolved_stat = resolved_log.stat()
            except OSError as exc:
                raise ValueError("The case run log changed while it was being opened.") from exc
            opened_identity = (log_stat.st_dev, log_stat.st_ino)
            path_identity = (resolved_stat.st_dev, resolved_stat.st_ino)
            if (
                stat.S_ISLNK(path_stat.st_mode)
                or resolved_log.parent != case_path
                or opened_identity != path_identity
            ):
                raise ValueError("The case run log must be a regular case-local file.")

            start = max(0, log_stat.st_size - maximum_bytes)
            os.lseek(descriptor, start, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = maximum_bytes
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(descriptor)
        if start > 0:
            newline = data.find(b"\n")
            if newline >= 0:
                data = data[newline + 1 :]
        self._send_json(
            {
                "ok": True,
                "exists": True,
                "text": data.decode("utf-8", errors="replace"),
                "sizeBytes": log_stat.st_size,
                "shownBytes": len(data),
                "truncated": start > 0,
                "modifiedAt": datetime.fromtimestamp(
                    log_stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )

    def _handle_case_progress(self, query: str) -> None:
        params = parse_qs(query)
        value = params.get("casePath", [None])[0]
        if not value:
            raise ValueError("A case path is required.")
        case_path = self._resolve_project_path(unquote(value))
        if not (case_path / "case.json").is_file():
            raise ValueError("The selected folder is not an AeroLab case.")
        self._send_json({"ok": True, "progress": case_run_progress(case_path)})

    def _handle_study_progress(self, query: str) -> None:
        params = parse_qs(query)
        value = params.get("casePath", [None])[0]
        if not value:
            raise ValueError("A study member path is required.")
        case_path = self._resolve_project_path(unquote(value))
        if not (case_path / "case.json").is_file():
            raise ValueError("The selected folder is not an AeroLab case.")
        record_path = case_path / "aerolab-study-run.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            record = None
        except json.JSONDecodeError as exc:
            raise ValueError("The study run record is not valid JSON.") from exc
        self._send_json({"ok": True, "exists": record is not None, "studyRun": record})

    def _handle_repair_model(self) -> None:
        payload = self._json_body()
        model_path = self._resolve_project_path(str(payload["modelPath"]))
        resolution = int(payload.get("resolution") or 384)
        smallest_feature_m = _optional_float(payload.get("smallestFeatureM"))
        unit_scale = _optional_float(payload.get("unitScale"))
        smallest_feature_source_units = (
            smallest_feature_m / unit_scale
            if smallest_feature_m and unit_scale
            else None
        )
        prepared_dir = self.server.root / "models" / "prepared"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_path = prepared_dir / f"{stamp}-{_safe_case_name(model_path.stem)}-prepared.stl"
        result = repair_stl(
            model_path,
            target_path,
            resolution=resolution,
            smallest_feature_source_units=smallest_feature_source_units,
        )
        self._send_json(
            {
                "ok": True,
                "accepted": result.accepted,
                "modelPath": str(target_path if result.accepted else model_path),
                "report": _report_payload(target_path, result.output_report),
                "preview": mesh_preview(target_path) if result.accepted else None,
                "repair": result.to_dict(),
            }
        )

    def _handle_analyze_features(self) -> None:
        payload = self._json_body()
        model_path = self._resolve_project_path(str(payload["modelPath"]))
        rotation = _model_rotation(payload.get("modelRotationDegrees"))
        result = detect_aero_features(
            model_path,
            scale=float(payload.get("unitScale") or 1.0),
            source_flow_direction=str(payload.get("sourceFlowDirection") or "+x"),
            source_up_direction=str(payload.get("sourceUpDirection") or "+z"),
            rotation_degrees=rotation,
        )
        self._send_json({"ok": True, "features": result})

    def _handle_create_case(self) -> None:
        payload = self._json_body()
        options = self._case_options(payload)
        model_path = options["model_path"]
        case_name = _safe_case_name(str(payload.get("name") or model_path.stem))

        case_path = create_case(
            case_name=case_name,
            cases_dir=self.server.root / "cases",
            generate_openfoam=True,
            **options,
        )
        case_json = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
        self._send_json(
            {
                "ok": True,
                "casePath": str(case_path),
                "case": case_json,
                "files": _case_files(case_path),
                "state": self._state(),
            }
        )

    def _handle_create_accuracy_study(self) -> None:
        payload = self._json_body()
        options = self._case_options(payload)
        model_path = options["model_path"]
        base_name = _safe_case_name(str(payload.get("name") or model_path.stem))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        study_id = f"grid-{stamp}"
        case_paths: list[Path] = []
        for level in ("draft", "standard", "fine"):
            level_options = dict(options)
            level_options["quality"] = level
            level_options["validation_study"] = {
                "id": study_id,
                "level": level,
                "levels": ["draft", "standard", "fine"],
                "drag_change_limit_percent": 2.0,
                "lift_change_limit": 0.02,
            }
            case_paths.append(
                create_case(
                    case_name=f"{base_name}-{study_id}-{level}",
                    cases_dir=self.server.root / "cases",
                    generate_openfoam=True,
                    **level_options,
                )
            )
        selected = case_paths[-1]
        self._send_json(
            {
                "ok": True,
                "studyId": study_id,
                "casePaths": [str(path) for path in case_paths],
                "selectedCasePath": str(selected),
                "report": case_report(selected, include_visualization=True),
                "state": self._state(),
            }
        )

    def _handle_create_sensitivity_study(self) -> None:
        payload = self._json_body()
        options = self._case_options(payload)
        model_path = options["model_path"]
        base_name = _safe_case_name(str(payload.get("name") or model_path.stem))
        parameter = str(payload.get("sensitivityParameter") or "")
        if parameter not in SENSITIVITY_PARAMETERS:
            supported = ", ".join(sorted(SENSITIVITY_PARAMETERS))
            raise ValueError(f"Choose one supported sensitivity parameter: {supported}.")
        raw_values = payload.get("sensitivityValues")
        if not isinstance(raw_values, list):
            raise ValueError("Sensitivity values must be a JSON list of two to twelve numbers.")
        values = [float(value) for value in raw_values]
        raw_baseline = payload.get("sensitivityBaselineIndex")
        baseline_index = None if raw_baseline in (None, "") else int(raw_baseline)
        study = create_sensitivity_study(
            base_options=options,
            base_name=base_name,
            cases_dir=self.server.root / "cases",
            parameter=parameter,
            values=values,
            generate_openfoam=True,
            baseline_index=baseline_index,
        )
        selected = Path(str(study["selectedCasePath"]))
        self._send_json(
            {
                "ok": True,
                "study": study,
                "selectedCasePath": str(selected),
                "report": case_report(selected, include_visualization=True),
                "state": self._state(),
            }
        )

    def _wheel_setup_options(self, value: object) -> list[dict[str, object]] | None:
        if value in (None, "", []):
            return None
        if not isinstance(value, list):
            raise ValueError("Wheel setup must be a JSON list.")
        wheels: list[dict[str, object]] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Wheel setup entry {index} must be an object.")
            wheel = dict(item)
            model_value = wheel.get("model_path") or wheel.get("modelPath") or wheel.get("geometry")
            if not model_value:
                raise ValueError(f"Wheel setup entry {index} requires model_path.")
            wheel["model_path"] = str(self._resolve_project_path(str(model_value)))
            wheels.append(wheel)
        return wheels

    def _volume_zone_options(
        self,
        value: object,
        label: str,
    ) -> list[dict[str, object]] | None:
        if value in (None, "", []):
            return None
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a JSON list.")
        zones: list[dict[str, object]] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{label} entry {index} must be an object.")
            zones.append(dict(item))
        return zones

    def _case_options(self, payload: dict[str, object]) -> dict[str, object]:
        return {
            "model_path": self._resolve_project_path(str(payload["modelPath"])),
            "speed_mph": float(payload.get("speedMph") or 70),
            "air_temperature_c": _optional_finite_float(
                payload.get("airTemperatureC"),
                "Air temperature",
            ),
            "air_pressure_pa": _optional_positive_float(
                payload.get("airPressurePa"),
                "Air pressure",
            ),
            "air_density_kg_m3": _optional_positive_float(
                payload.get("airDensityKgM3"),
                "Air density",
            ),
            "kinematic_viscosity_m2_s": _optional_positive_float(
                payload.get("kinematicViscosityM2S"),
                "Kinematic viscosity",
            ),
            "turbulence_intensity_percent": _optional_positive_float(
                payload.get("turbulenceIntensityPercent"),
                "Turbulence intensity",
            ),
            "turbulence_length_scale_m": _optional_positive_float(
                payload.get("turbulenceLengthScaleM"),
                "Turbulence length scale",
            ),
            "yaw_degrees": _optional_finite_float(payload.get("yawDegrees"), "Yaw angle"),
            "crosswind_mps": _optional_finite_float(
                payload.get("crosswindMps"),
                "Crosswind speed",
            ),
            "roughness_height_m": _nonnegative_float(
                payload.get("roughnessHeightM"),
                "Surface roughness height",
            ),
            "roughness_constant": (
                _optional_finite_float(payload.get("roughnessConstant"), "Roughness constant")
                or 0.5
            ),
            "closed_tunnel": payload.get("closedTunnel"),
            "backflow_safe_outlet": bool(payload.get("backflowSafeOutlet")),
            "wheel_setup": self._wheel_setup_options(payload.get("wheelSetup")),
            "second_order_transient": bool(payload.get("secondOrderTransient")),
            "fluid_profile": str(payload.get("fluidProfile") or "incompressible"),
            "turbulence_model": str(payload.get("turbulenceModel") or "kOmegaSST"),
            "porous_zones": self._volume_zone_options(
                payload.get("porousZones"),
                "Porous zones",
            ),
            "fan_zones": self._volume_zone_options(
                payload.get("fanZones"),
                "Fan zones",
            ),
            "heat_zones": self._volume_zone_options(
                payload.get("heatZones"),
                "Heat-load zones",
            ),
            "flow_axis": str(payload.get("flowAxis") or "x"),
            "include_ground": bool(payload.get("includeGround")),
            "moving_ground": bool(payload.get("movingGround")),
            "ground_clearance_m": _nonnegative_float(
                payload.get("groundClearanceM"),
                "Ground clearance",
            ),
            "unit_scale": float(payload.get("unitScale") or 1.0),
            "unit_label": str(payload.get("unitLabel") or "m"),
            "reference_area_m2": _optional_float(payload.get("referenceAreaM2")),
            "reference_length_m": _optional_float(payload.get("referenceLengthM")),
            "center_of_gravity_m": _optional_vector(payload.get("centerOfGravityM"), "Vehicle CG"),
            "front_axle_station_m": _optional_finite_float(
                payload.get("frontAxleStationM"),
                "Front axle station",
            ),
            "rear_axle_station_m": _optional_finite_float(
                payload.get("rearAxleStationM"),
                "Rear axle station",
            ),
            "measured_length_m": _optional_float(payload.get("measuredLengthM")),
            "measured_width_m": _optional_float(payload.get("measuredWidthM")),
            "measured_height_m": _optional_float(payload.get("measuredHeightM")),
            "smallest_aero_feature_m": _optional_float(payload.get("smallestAeroFeatureM")),
            "quality": str(payload.get("quality") or "standard"),
            "simulation_mode": str(payload.get("simulationMode") or "steady"),
            "source_flow_direction": str(payload.get("sourceFlowDirection") or "+x"),
            "source_up_direction": str(payload.get("sourceUpDirection") or "+z"),
            "model_rotation_degrees": _model_rotation(payload.get("modelRotationDegrees")),
        }

    def _handle_run_case(self) -> None:
        payload = self._json_body()
        case_path = self._resolve_project_path(str(payload["casePath"]))
        backend = str(payload.get("backend") or "auto")
        run_mode = str(payload.get("mode") or "full")
        default_timeout = 14400 if run_mode == "mesh" else 21600
        timeout_seconds = int(payload.get("timeoutSeconds") or default_timeout)
        reuse_mesh = payload.get("reuseMesh") is not False
        processes = normalize_process_request(payload.get("processes", "auto"))
        file_handler = normalize_file_handler(payload.get("fileHandler", "auto"))
        resume_value = payload.get("resume", False)
        if not isinstance(resume_value, bool):
            raise ValueError("Resume must be true or false.")
        self.server.start_case_run(
            case_path,
            backend,
            timeout_seconds,
            run_mode,
            reuse_mesh,
            processes,
            file_handler,
            resume_value,
        )
        self._send_json(
            {
                "ok": True,
                "accepted": True,
                "casePath": str(case_path),
                "mode": run_mode,
                "timeoutSeconds": timeout_seconds,
                "processes": processes,
                "fileHandler": file_handler,
                "resume": resume_value,
            },
            status=202,
        )

    def _handle_run_study(self) -> None:
        payload = self._json_body()
        case_path = self._resolve_project_path(str(payload["casePath"]))
        backend = str(payload.get("backend") or "auto")
        run_mode = str(payload.get("mode") or "full")
        default_timeout = 14400 if run_mode == "mesh" else 21600
        timeout_seconds = int(payload.get("timeoutSeconds") or default_timeout)
        reuse_mesh = payload.get("reuseMesh") is not False
        processes = normalize_process_request(payload.get("processes", "auto"))
        process_budget = normalize_study_process_budget(
            payload.get("processBudget", "auto")
        )
        file_handler = normalize_file_handler(payload.get("fileHandler", "auto"))
        descriptor = self.server.start_study_run(
            case_path,
            backend,
            timeout_seconds,
            run_mode,
            reuse_mesh,
            processes,
            process_budget,
            file_handler,
        )
        self._send_json(
            {
                "ok": True,
                "accepted": True,
                "casePath": str(case_path),
                "study": descriptor,
                "mode": run_mode,
                "timeoutSeconds": timeout_seconds,
                "processes": processes,
                "processBudget": process_budget,
                "fileHandler": file_handler,
            },
            status=202,
        )

    def _handle_compare_cases(self) -> None:
        payload = self._json_body()
        baseline_path = self._resolve_project_path(str(payload["baselineCasePath"]))
        variant_path = self._resolve_project_path(str(payload["variantCasePath"]))
        self._send_json(
            {
                "ok": True,
                "comparison": compare_cases(baseline_path, variant_path),
            }
        )

    def _handle_case_report(self) -> None:
        payload = self._json_body()
        case_path = self._resolve_project_path(str(payload["casePath"]))
        self._send_json({"ok": True, "report": case_report(case_path, include_visualization=True)})

    def _state(self) -> dict[str, object]:
        sample = self.server.root / "models" / "sample_box.stl"
        return {
            "ok": True,
            "root": str(self.server.root),
            "sampleModel": str(sample) if sample.exists() else None,
            "cases": _list_cases(self.server.root / "cases"),
            "sensitivityParameters": SENSITIVITY_PARAMETERS,
        }

    def _json_body(self) -> dict[str, object]:
        data = self.rfile.read(_content_length(self.headers.get("Content-Length")))
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.server.root / path
        path = path.resolve()
        if not path.is_relative_to(self.server.root):
            raise ValueError("Path must be inside the AeroLab project.")
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        path = path.resolve()
        if not path.exists() or not path.is_file() or not path.is_relative_to(self.server.web_root):
            self.send_error(404)
            return
        data = path.read_bytes()
        if content_type is None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _record_background_run_failure(
    case_path: Path,
    run_mode: str,
    backend: str,
    error: Exception,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    run_path = case_path / "aerolab-run.json"
    existing: dict[str, object] = {}
    try:
        existing = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    record = {
        **existing,
        "status": "failed",
        "ok": False,
        "trusted": False,
        "numericallyQualified": False,
        "mode": run_mode,
        "backend": backend,
        "returncode": 127,
        "budgetRecommendation": None,
        "startedAt": existing.get("startedAt") or now,
        "finishedAt": now,
        "error": str(error),
    }
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    case_json_path = case_path / "case.json"
    try:
        case_payload = json.loads(case_json_path.read_text(encoding="utf-8"))
        case_payload["status"] = "mesh_failed" if run_mode == "mesh" else "solver_failed"
        case_json_path.write_text(json.dumps(case_payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass
    try:
        with (case_path / "aerolab-run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\nAeroLab background run failed: {error}\n")
    except OSError:
        pass


def _record_background_study_failure(
    member_paths: list[Path],
    backend: str,
    error: Exception,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for member_path in member_paths:
        record_path = member_path / "aerolab-study-run.json"
        existing: dict[str, object] = {}
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        record = {
            **existing,
            "status": "failed",
            "ok": False,
            "backend": backend,
            "budgetRecommendation": None,
            "startedAt": existing.get("startedAt") or now,
            "finishedAt": now,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_app(host: str, port: int, root: Path) -> None:
    root = root.resolve()
    (root / "models" / "uploads").mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    server = AeroLabServer((host, port), root)
    print(f"AeroLab app running at http://{host}:{port}")
    print(f"Project root: {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAeroLab app stopped.")


def _content_length(value: str | None) -> int:
    if value is None:
        return 0
    return int(value)


def _safe_filename(value: str) -> str:
    name = Path(value).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return name or "model.stl"


def _safe_case_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return name or "aerolab-case"


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("Reference values must be positive.")
    return number


def _optional_positive_float(value: object, label: str) -> float | None:
    number = _optional_finite_float(value, label)
    if number is not None and number <= 0:
        raise ValueError(f"{label} must be a positive number.")
    return number


def _optional_finite_float(value: object, label: str) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _optional_vector(value: object, label: str) -> tuple[float, float, float] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} requires X, Y, and Z coordinates.")
    raw = [value.get(axis) for axis in ("x", "y", "z")]
    provided = sum(component not in (None, "") for component in raw)
    if provided == 0:
        return None
    if provided != 3:
        raise ValueError(f"{label} requires X, Y, and Z coordinates together.")
    return tuple(
        float(_optional_finite_float(component, f"{label} {axis.upper()}"))
        for axis, component in zip(("x", "y", "z"), raw)
    )  # type: ignore[return-value]


def _nonnegative_float(value: object, label: str) -> float:
    if value in (None, ""):
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a non-negative number.")
    return number


def _model_rotation(value: object) -> tuple[float, float, float]:
    if not isinstance(value, dict):
        return (0.0, 0.0, 0.0)
    result = []
    for axis in ("x", "y", "z"):
        try:
            angle = float(value.get(axis) or 0.0)
        except (TypeError, ValueError):
            raise ValueError(f"Model rotation {axis.upper()} must be a number.") from None
        if not math.isfinite(angle) or abs(angle) > 3600:
            raise ValueError(f"Model rotation {axis.upper()} must be between -3600 and 3600 degrees.")
        result.append(angle)
    return tuple(result)  # type: ignore[return-value]


def _report_payload(model_path: Path, report: object) -> dict[str, object]:
    payload = report.to_dict()  # type: ignore[attr-defined]
    fidelity = repair_fidelity_for_model(model_path)
    if fidelity is None:
        return payload
    payload["repair_fidelity"] = fidelity
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict):
        readiness = {"score": 100, "status": "ready", "failed": 0, "warnings": 0, "items": []}
        payload["readiness"] = readiness
    items = readiness.get("items")
    if not isinstance(items, list):
        items = []
        readiness["items"] = items
    if fidelity.get("verified"):
        items.append(
            {
                "label": "Repair fidelity",
                "status": "pass",
                "detail": (
                    f"Recorded cell detail {float(fidelity.get('detailResolutionPercent') or 0):.3g}%; "
                    f"source p99 {float(fidelity.get('sourceSurfaceDeviationP99Percent') or 0):.3g}%; "
                    f"far sealing area {float(fidelity.get('addedSurfaceFarFractionPercent') or 0):.3g}%."
                ),
            }
        )
    else:
        payload["is_cfd_candidate"] = False
        readiness["score"] = min(int(readiness.get("score") or 0), 60)
        readiness["status"] = "needs_cleanup"
        readiness["failed"] = int(readiness.get("failed") or 0) + 1
        items.append(
            {
                "label": "Repair fidelity",
                "status": "fail",
                "detail": str(fidelity.get("detail") or "Prepared geometry fidelity is not verified."),
            }
        )
    return payload


def _list_cases(cases_dir: Path) -> list[dict[str, object]]:
    if not cases_dir.exists():
        return []
    cases: list[dict[str, object]] = []
    for case_json in sorted(cases_dir.glob("*/case.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(case_json.read_text(encoding="utf-8"))
            cases.append(
                {
                    "name": payload.get("name", case_json.parent.name),
                    "path": str(case_json.parent),
                    "status": payload.get("status", "unknown"),
                    "speedMph": payload.get("flow", {}).get("speed_mph"),
                    "quality": payload.get("cfd_quality", {}).get("name"),
                    "studyId": payload.get("validation_study", {}).get("id"),
                    "studyLevel": payload.get("validation_study", {}).get("level"),
                    "sensitivityStudyId": payload.get("sensitivity_study", {}).get("id"),
                    "sensitivityParameter": payload.get("sensitivity_study", {}).get("parameter"),
                    "sensitivityParameterLabel": payload.get("sensitivity_study", {}).get("parameter_label"),
                    "sensitivityUnit": payload.get("sensitivity_study", {}).get("unit"),
                    "sensitivityValue": payload.get("sensitivity_study", {}).get("value"),
                    "sensitivityIndex": payload.get("sensitivity_study", {}).get("index"),
                    "sensitivityCount": payload.get("sensitivity_study", {}).get("count"),
                    "sensitivityBaseline": payload.get("sensitivity_study", {}).get("is_baseline"),
                    "reference": payload.get("aerodynamic_reference"),
                    "comparisonLockHash": payload.get("comparison_lock", {}).get("hash"),
                    "balanceDatumQualified": payload.get("vehicle_datums", {}).get("balance_qualified"),
                    "createdAt": payload.get("created_at"),
                    "progress": case_run_progress(case_json.parent),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return cases


def _case_files(case_path: Path) -> list[str]:
    # POSIX separators keep the JSON API response consistent across platforms.
    return [
        path.relative_to(case_path).as_posix()
        for path in sorted(case_path.rglob("*"))
        if path.is_file()
    ]
