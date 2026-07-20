# AeroLab CFD

Local-first CFD workflow for checking the aerodynamics of 3D models.

The goal is to turn this into a practical pipeline:

```text
3D model -> geometry check -> virtual wind tunnel -> CFD case -> solver run -> aero report
```

The current version loads and repairs STL scans, creates OpenFOAM Foundation v13 external-aerodynamics cases, runs a configured local solver, and audits mesh and convergence quality before marking results verified.

## Current Features

- Local Python CLI
- Local browser app
- Animated 3D-style airflow preview
- Exact STL rendering in the airflow viewer, streamed independently of the sampled airflow helper mesh
- Input unit scaling for scanned STL files
- Area-weighted principal-axis auto-alignment with editable rotation values
- ASCII and binary STL inspection
- Mesh quality checks:
  - triangle count
  - bounding box
  - approximate surface area
  - approximate enclosed volume
  - open edges
  - non-manifold edges
  - degenerate triangles
- Non-destructive scan preparation that writes a separate sealed STL and verifies the result
- Projected triangle-union reference-area estimates with manual reference overrides
- Case setup metadata for external airflow runs
- OpenFOAM Foundation v13 cases using `foamRun` and `incompressibleFluid`
- Near-body and wake refinement zones plus Reynolds-based absolute boundary layers on standard/fine meshes
- Mesh, residual, and force-coefficient convergence gates
- Actual body `y+` verification after a completed run
- Two-way fidelity audit between the transformed STL and the actual solved OpenFOAM body patch
- Solver-derived, speed-colored VTK streamlines and animated particle traces in the browser when OpenFOAM output is present
- Shareable Markdown or self-contained HTML result reports from `report-case`
- Sample STL model for smoke testing

## Quick Start

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
aerolab check .\models\sample_box.stl
```

On Windows, if `python` is not on PATH, try the Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
aerolab check .\models\sample_box.stl
```

If you do not want to install it yet, you can run it directly:

```powershell
$env:PYTHONPATH = ".\src"
python -m aerolab check .\models\sample_box.stl
```

Or with the Windows launcher:

```powershell
$env:PYTHONPATH = ".\src"
py -m aerolab check .\models\sample_box.stl
```

## Create A Local CFD Case

```powershell
aerolab init-case .\models\sample_box.stl --name sample-70mph --speed-mph 70
```

This creates a complete OpenFOAM case folder under `cases/` with a `case.json` file.

## Start The Local App

```powershell
aerolab app
```

Then open:

```text
http://127.0.0.1:8765/
```

For the complete scan-to-results workflow and trust checklist, read [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

The app currently lets you load/check an STL, view the actual STL mesh inside an animated airflow preview, prepare small scan defects into a separate verified STL, adjust speed/ground/unit settings, check CFD readiness, and generate an OpenFOAM case folder.

Creating a case only prepares the files; it does not run CFD. Use **Validate Mesh** first to build and audit the volume mesh without spending time on a flow solve. A mesh that passes the reusable gates is fingerprinted to that exact vehicle STL and mesh setup, then **Run Solver** reuses it. Any geometry or mesh-dictionary change forces a rebuild. Each action shows live OpenFOAM progress; green means verified, amber needs review, and red failed a required gate.

Choose **Steady RANS** for faster setup and mean-flow checks. Choose **Transient + averaged** when separated wakes and speed-dependent unsteadiness matter: AeroLab runs Foundation OpenFOAM's transient PIMPLE algorithm with Courant-controlled time steps, develops the wake for multiple vehicle flow-through lengths, and exports time-averaged velocity, pressure, wall shear, and force histories. Transient output is accepted only when the mean forces stop drifting, residuals remain bounded, Courant control passes, and mean fields are present.

The workflow is vehicle-agnostic. It does not identify or tune for a Civic model: tunnel dimensions, reference estimates, refinement boxes, placement, and fidelity limits are derived from each transformed STL. Cars, trucks, vans, open-wheel cars, motorcycles, and other vehicle bodies use the same validation path. AeroLab also scans the transformed rear underbody for diffuser-like rising surfaces and reports ranked geometry candidates. These candidates help inspection, but an STL has no part names and cannot prove that a ramp is a functional diffuser. Accuracy still depends on a complete exterior, correct units and orientation, measured dimensions, a declared smallest relevant aero feature, and a suitable physical setup for that vehicle.

Use **Create Accuracy Study** for final force numbers. Enter the model's measured length, width, and height first; each scaled STL dimension must agree within 2%. Also enter the smallest aerodynamic feature whose pressure effect matters. The study creates matched draft, standard, and fine cases. AeroLab only marks the study grid-independent after all three runs are individually verified, actual cell counts increase at each level, mean Cd changes by no more than 2% from standard to fine with a converging trend, and mean Cl changes by no more than 0.02.

The animated smoke lines are a geometry-aware setup preview. They wrap around cross-sections measured from the STL, but they are not solver results. After a completed run, AeroLab reads OpenFOAM's VTK tracks and automatically replaces the setup smoke with speed-colored solver streamlines and animated particles over the exact transformed case geometry. Use the viewer's `Lines | Particles | Both` control to choose the solved-flow display; transient runs are labeled as mean-flow visualization because they use `UMean`. New cases also export solved body pressure and `wallShearStress` as an ASCII VTK surface; AeroLab converts incompressible kinematic pressure to `Cp` and enables solved `Cp` and `Drag areas` surface modes. The drag map combines local pressure drag (`Cp * n_flow`) with flow-direction wall shear, while its pressure, skin-friction, and total `Cd` values are integrated from the original solver faces before any browser decimation. The drag readout also divides the body into front, middle, and rear thirds along the wind axis and reports each section's share of positive drag. The result summary converts mean `Cl` and `Cd` into total downforce or lift and drag in newtons and pounds-force at the case's solved speed. Older completed cases remain pressure-only until they are upgraded and rerun.

If a scan reports open or non-manifold edges, use **Prepare Scan**. AeroLab creates a separate STL under `models/prepared`, preserves the original upload, and records a matching `.repair.json` fidelity file. Simple pinholes up to 3% of the model length are capped locally without moving source triangles. When nearly all triangles form one valid body and a tiny overlapping defect is isolated, AeroLab can remove only that local overlay and cap its small remaining loops. It accepts this surgical route only after exact two-way surface-deviation and silhouette checks. Broader defects fall back to a sealed voxel-shell repair. Entering **Smallest aero feature mm** raises voxel resolution when practical, with an in-app warning when the local cell budget cannot preserve that scale. Voxel repairs are accepted only when watertight, manifold, free of degenerate triangles, dimensionally faithful, and within strict source-surface and added-seal deviation limits. Prepared meshes without a matching accepted fidelity record are blocked from new CFD cases.

The shaded viewer fetches the actual STL bytes and is not limited by the smaller geometry payload used to animate setup airflow. **Display tris** reports the exact rendered triangle count; **Edges** exposes the original triangle network. More triangles preserve detail that exists in the scan, but subdividing a coarse or damaged surface does not recreate missing body lines. The viewer renders every triangle in the STL; OpenFOAM then creates a separate volume mesh around that exact solver surface.

For scanned cars, choose the correct STL units before creating a case. Phone/scan/mesh tools often export in millimeters.

For road-car cases, enable **Ground** and set **Lowest-point road gap mm**. Use `0` only when the STL includes tire contact at the road. For a body-only model, enter the measured distance from its lowest modeled point to the road. AeroLab translates the solver STL so that clearance is exact, keeps the tunnel road at `z = 0`, records the translation in `case.json`, and includes road placement in solver-quality validation. The CLI equivalent is `--ground-clearance-mm`.

If the scan or downloaded model is not full scale, enter the real vehicle length in meters. AeroLab uses that flow-axis length to scale the STL before generating the OpenFOAM case. Also enter the measured width and height: a mismatch over 2% blocks case generation, and all three measurements are required for a verified accuracy study.

If the scan uses a different coordinate system, use **Auto-align** as a starting point or set source flow, source up, and rotation manually before creating the case. Auto-align uses area-weighted principal axes and is offered only for vehicle-like proportions with adequate confidence. Confirm that the nose points into the indicated wind direction and the roof is up; the numeric rotation fields allow exact corrections. AeroLab bakes that transform into the exported `body.stl`.

For meaningful drag/lift coefficients, set the aerodynamic reference area and length when you know them. If left blank, AeroLab estimates `Aref` from a high-resolution scanline union of the scaled model's projected triangles and estimates `lRef` from the model length along the flow axis. This preserves visible concavities and disconnected gaps that a convex outline would fill. Use a measured manufacturer or CAD frontal area for published coefficient comparisons.

Use the CFD quality preset based on intent: `draft` for quick setup checks, `standard` for normal local checks with surface levels 4-6, five requested wall layers, a workstation-safe 2.8-million-cell threshold, and up to 1200 steady iterations, and `fine` for comparisons with surface levels 5-7, eight requested wall layers, and up to 2000 iterations. When a Standard case raises sharp-feature refinement to level 8, AeroLab retains that surface target while limiting the surrounding body volume to level 3 and using two transition layers to control peak WSL memory. Standard and fine cases estimate absolute layer thickness from Reynolds number and target high-wall-function `y+`. AeroLab reads the final snappyHexMesh layer summary and solved body `y+` distribution; candidate extrusion messages do not count as successful layer coverage. The run stops before solving when baseline `checkMesh` does not report `Mesh OK.`

Enter the width of the smallest aerodynamic feature whose pressure effect matters, such as a diffuser strake or vortex generator. AeroLab targets at least four sharp-feature cells across that width by raising the `surfaceFeatures` and maximum body-surface refinement levels, with quality-dependent local/global cell caps. The broad body surface keeps a lower level so a small feature does not force the entire tunnel to the same cell size. Draft and Standard cases currently stop at refinement level 8; Fine stops at level 9 with a larger cell budget. A smaller unsupported target is rejected before meshing instead of silently generating an under-resolved case or exhausting a local machine. The post-run body-patch fidelity audit remains the final proof that OpenFOAM actually retained the feature.

## Solver Status And Runs

Check whether OpenFOAM is available locally:

```powershell
aerolab solver-status
```

Create a case with manual coefficient references:

```powershell
aerolab init-case .\models\sample_box.stl --name sample-ref --speed-mph 70 --reference-area-m2 2.2 --reference-length-m 4.5
```

Run a generated case:

```powershell
aerolab run-case .\cases\sample-ref
```

Validate and preserve its mesh before solving:

```powershell
aerolab run-case .\cases\sample-ref --mode mesh
```

Read available force coefficient results:

```powershell
aerolab report-case .\cases\sample-ref
```

Export a shareable summary as Markdown or self-contained HTML:

```powershell
aerolab report-case .\cases\sample-ref --format markdown --output report.md
aerolab report-case .\cases\sample-ref --format html --output report.html
```

The report gathers the verified/unverified status, mean `Cd`/`Cl`, drag and
downforce in newtons and pounds-force, the case setup, every verification
check, and mesh/convergence metrics. Fields that are not available yet (for a
case that has been set up but not solved) render as an em dash. Without
`--output`, the report prints to standard output; `--format json` and the
default human-readable text output are unchanged.

The generated cases target OpenFOAM Foundation v13. After a completed run, AeroLab compares sampled points on the transformed STL and the actual OpenFOAM body patch in both directions using exact point-to-triangle distance queries. It checks p95/p99 deviation, dimensions, frontal silhouette, requested feature resolution, final boundary-layer coverage, and the body `y+` distribution. AeroLab only labels a result verified when that body-fidelity audit passes, the solver exits successfully, baseline `checkMesh` passes, velocity/pressure/turbulence residuals meet the selected preset, at least 30 force samples stabilize, and wall treatment is compatible with the generated mesh. A completed process can therefore remain unverified, which is intentional. Final aerodynamic claims still require the three-level accuracy study and real-world validation against measured dimensions or test data.

AeroLab currently generates incompressible external-flow cases. It records Mach number, dynamic pressure, and Reynolds number in every new case and rejects speeds at or above Mach 0.3, about 230 mph in the standard-air model. Higher-speed work requires a compressible solver setup rather than extending the current case beyond its physical range.

The preferred local route is OpenFOAM Foundation v13 inside WSL2. AeroLab automatically stages each WSL case under `$HOME/.cache/aerolab-cfd/runs` on the native Linux filesystem, runs OpenFOAM there, and copies the generated mesh and results back into the Windows case folder. This avoids the severe small-file I/O penalty of meshing directly under `/mnt/c` while keeping every durable case in the standalone project. If `solver-status` reports no backend, install/configure WSL2 and OpenFOAM before running cases.

## Folder Layout

```text
aerolab-cfd/
  src/aerolab/             Python CLI and core logic
  models/                  Input STL/OBJ/STEP models later
  cases/                   Generated local CFD cases
  outputs/                 Reports and solver outputs
  templates/               Solver case templates
  docs/                    Notes and roadmap
```

## Local Solver Direction

The recommended local path on Windows is:

1. Python CLI/browser app on Windows
2. OpenFOAM inside WSL2 or Docker
3. ParaView for visualization
4. Later: imported solver result overlays in the browser app

We will use proven CFD solvers underneath instead of writing fluid dynamics math from scratch.
