# Local Setup

## Minimum Setup

Install Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
aerolab check .\models\sample_box.stl
```

If `python` is not recognized on Windows, use:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
aerolab check .\models\sample_box.stl
```

## Start The App

```powershell
aerolab app
```

Open `http://127.0.0.1:8765/` in your browser. The app runs locally on your machine.

## Solver Check

```powershell
aerolab solver-status
```

If no backend is available, install WSL2 and OpenFOAM. The app can generate cases without OpenFOAM, but real drag/lift results require the solver.

## Run A Case

```powershell
aerolab run-case .\cases\your-case-name
aerolab report-case .\cases\your-case-name
```

## CFD Solver Setup Later

For full CFD runs on Windows, the cleanest options are:

- WSL2 with OpenFOAM installed inside Linux
- Docker Desktop with an OpenFOAM image

The Python CLI will remain the local controller. The solver can run in WSL2/Docker so your Windows project files stay easy to manage.

## Recommended Local Tools

- Blender: scan cleanup and mesh repair
- MeshLab: mesh inspection and decimation
- ParaView: CFD visualization
- OpenFOAM: CFD solver
- WSL2 or Docker: local solver environment

## Reproducible OpenFOAM benchmark

AeroLab includes an offline `cube-symmetry-v1` real-solver smoke benchmark. It verifies the packaged STL checksum, generates a fresh deterministic case, requires OpenFOAM Foundation v13, applies the normal numerical-qualification gates, and checks cube symmetry and broad drag sanity limits. It does **not** claim absolute aerodynamic accuracy; a reference-geometry benchmark is still required for that claim.

Prepare and inspect the benchmark without a solver:

```bash
aerolab benchmark cube-symmetry-v1 --prepare-only --output-dir outputs/benchmarks --json
```

Run it on a machine with Foundation v13 available through the native, WSL, or configured local Docker backend:

```bash
aerolab benchmark cube-symmetry-v1 --backend auto --timeout-seconds 3600 --json
```

Every attempt is stored separately under `outputs/benchmarks/cube-symmetry-v1/<run-id>/` with the raw packaged manifest, generated case, and `benchmark-result.json`. Evidence files are SHA-256 indexed in the result, and `benchmark-result.sha256` records a checksum of the result itself. These adjacent checksums detect accidental or unreconciled changes, but they are not an external signature and do not prevent someone with write access from deliberately recomputing them. Solver execution is pinned to the probed paths and executable hashes for the complete OpenFOAM toolchain used by AeroLab; Docker probes and runs are additionally pinned to the inspected image ID. Set `AEROLAB_BUILD_REVISION` to a release or source revision when you need that identifier recorded with the solver provenance. The command exits `0` for a passed or successfully prepared benchmark, `1` for a scientific or numerical benchmark failure, `2` for a setup, infrastructure, or evaluation error, and `3` when no requested OpenFOAM backend is available. Normal benchmark execution never downloads a solver image.

To enable the environment-gated real-solver integration test on a pre-provisioned Foundation v13 host:

```bash
AEROLAB_RUN_OPENFOAM_BENCHMARK=1 \
AEROLAB_BENCHMARK_BACKEND=auto \
python -m unittest discover -s tests -p 'test_benchmark.py' -v
```
