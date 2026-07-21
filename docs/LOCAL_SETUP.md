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

## Solver Backends

Check the machine before attempting a run:

```powershell
aerolab solver-status --json
```

Case generation and reports work without OpenFOAM. Meshing, solving, resume, and real performance measurements require OpenFOAM Foundation v13 in one of these environments:

### Windows: WSL2 (recommended)

1. Install a WSL2 Linux distribution and make the distribution containing Foundation v13 the WSL default.
2. Install OpenFOAM Foundation v13 inside that distribution.
3. From PowerShell, verify the distribution and required tools:

```powershell
wsl --status
wsl --list --verbose
wsl bash -lc "command -v foamRun blockMesh snappyHexMesh mpirun decomposePar reconstructParMesh reconstructPar"
aerolab solver-status --json
```

AeroLab copies each case from the Windows filesystem to `$HOME/.cache/aerolab-cfd/runs` inside WSL, runs it on the Linux filesystem, and copies results back even after normal solver failure. This avoids running OpenFOAM's many small-file operations directly under `/mnt/c`. A marked interrupted stage is recovered on the next run; unmarked paths are never replaced.

### Linux: native Foundation v13

Install Foundation v13 so its environment and required commands are available to `bash -lc`, then run `aerolab solver-status --json`. Native execution does not need WSL or Docker.

### Docker: optional, not required

Docker is not required and is not installed or configured by AeroLab. If you intentionally choose it later:

1. install and start a Docker-compatible local engine;
2. complete any organization-required sign-in or policy setup;
3. build or pull a vetted image containing OpenFOAM Foundation v13 and all commands reported by `solver-status`;
4. configure the image explicitly; and
5. rerun the status check.

PowerShell:

```powershell
$env:AEROLAB_OPENFOAM_IMAGE = "your-vetted-foundation-v13-image"
aerolab solver-status --json
```

Bash:

```bash
export AEROLAB_OPENFOAM_IMAGE=your-vetted-foundation-v13-image
aerolab solver-status --json
```

AeroLab never pulls an image automatically. Docker runs bind-mount the durable case at `/source`, copy it to an anonymous Linux volume at `/work`, run there, copy results back, and remove the container with attached volumes. This avoids direct macOS/Windows bind-mount execution for OpenFOAM's small-file workload. Do not claim Docker support for an image until `solver-status` verifies the Foundation v13 identity and complete toolchain on the target architecture.

## Run And Accelerate A Case

The CLI and browser default to backend-aware automatic process selection:

```powershell
# Validate and fingerprint the mesh first.
aerolab run-case .\cases\your-case-name --mode mesh --processes auto

# Reuse the matching mesh for the full solve.
aerolab run-case .\cases\your-case-name --processes auto

# Request an exact rank count; invalid CPU/memory requests fail clearly.
aerolab run-case .\cases\your-case-name --processes 4

# Resume only a compatible failed full run.
aerolab run-case .\cases\your-case-name --processes auto --resume

aerolab report-case .\cases\your-case-name
```

For an accuracy or sensitivity study, select any member:

```powershell
aerolab run-study .\cases\one-study-member --processes auto --process-budget auto
```

The process budget is shared across every concurrent member. AeroLab estimates the largest member's memory need, bounds worker count, and falls back to concurrent serial cases when MPI is unavailable.

### Post-failure workstation budget

After an unsuccessful solver process, AeroLab inspects the return code, bounded log tail, configured cell cap, resolved ranks, and backend-visible memory. `aerolab-run.json`, JSON CLI output, progress APIs, and the browser can report a `budgetRecommendation` for supported evidence: out-of-memory termination, MPI slot oversubscription, study concurrency pressure, exhausted storage, mesh/cell pressure, or a runtime limit.

The conservative cell allowance reserves the larger of 25% or 1 GiB of backend-visible memory, subtracts a 2 GiB fixed allowance, budgets 2 KiB per configured maximum cell, and rounds down to 50,000 cells. It is diagnostic guidance, not proof that a particular mesh will fit.

A single case receives at most one automatic retry, and only when the original process request was `auto`, memory/MPI evidence is high-confidence, reducing ranks is possible, and the configured cell cap does not exceed the calculated allowance. The retry changes only MPI rank count; geometry, mesh dictionaries, quality preset, physical setup, convergence controls, and verification gates remain unchanged. The browser waits briefly for worker cleanup and retries once. If it still fails, the failure remains visible.

**Retry with safer budget** performs the same explicit rank-only retry when available. For studies, AeroLab never retries automatically; the button uses the recommended ranks as both the per-case and aggregate budget so one unchanged member runs at a time. A whole-study retry reruns its members, including members that previously succeeded.

Single-rank memory failures, storage exhaustion, ambiguous timeouts, and cases whose configured cell cap exceeds the allowance are not auto-adjusted. Add backend memory/storage, increase an evidenced runtime limit, or explicitly regenerate at a lower mesh budget. Regeneration changes numerical fidelity and requires mesh validation and comparison evidence again; AeroLab never selects a lower quality preset on the user's behalf.

## Performance And Safety Contract

Automatic rank selection uses resources visible **inside** the selected backend, not browser CPU counts or host-only values. It:

- reserves one effective CPU;
- reserves 25% of available memory, with a 1 GiB minimum reserve;
- budgets 2 GiB per automatic MPI rank;
- considers the configured maximum cell count at roughly 250,000 cells per useful rank;
- caps one case at eight ranks; and
- selects one rank if CPU, memory, case size, or MPI availability does not justify parallel execution.

An explicit positive rank count is validated against backend CPU and memory data and is never silently clamped. `--file-handler auto` preserves the OpenFOAM default and is the safe choice until another handler has been verified with that MPI build.

Generated cases cache only fingerprint-matched `surfaceFeatures` and `blockMesh` stages. Final mesh reuse additionally requires the validated mesh record, matching geometry/mesh inputs, `constant/polyMesh/points`, and the audited body surface. Any relevant input change causes rebuilding instead of stale reuse.

`--resume` is intentionally strict. It requires a previously failed full run, unchanged solver inputs, a reusable mesh, and the latest reconstructed numeric time with both `U` and `p`. It skips initialization and uses `-latestTime`; it is rejected for mesh-only runs, successful runs, changed numerics, missing reconstructed fields, and disabled mesh reuse.

Steady cases retain Foundation v13 `residualControl`; transient cases retain their fixed physical warmup and averaging windows. Resume and acceleration do not weaken any mesh, convergence, force-stability, Courant, field-completeness, or fidelity gate.

### Performance expectations

These are engineering expectations, not measured claims:

- MPI can reduce expensive mesh and solve stages on sufficiently large cases, but scaling is not linear.
- Small cases may be slower with MPI because decomposition, communication, reconstruction, and cleanup add overhead.
- Stage-cache hits and validated mesh reuse can save more time than extra ranks on repeated unchanged work.
- WSL/Docker Linux-volume staging targets filesystem overhead; it does not accelerate solver mathematics.
- Concurrent study scheduling improves total throughput while respecting one aggregate resource budget.
- Standard Foundation v13 workflows use CPU execution; AeroLab does not currently provide GPU acceleration.

Benchmark serial and automatic modes on the actual target computer before quoting a speedup. Preserve the generated run records with the comparison because they contain backend resources, requested/resolved ranks, cache state, and solver identity.

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
