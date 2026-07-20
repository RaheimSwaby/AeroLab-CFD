# AeroLab CFD User Guide

This guide describes the workflow for turning an STL scan into a local OpenFOAM wind-tunnel case and deciding whether its results are trustworthy.

## Start AeroLab

Open PowerShell and run:

```powershell
cd C:\Users\Rahei\Downloads\aerolab-cfd

& .\.venv\Scripts\python.exe .\scripts\aerolab_app.py
```

Keep that PowerShell window open. Open `http://127.0.0.1:8765/` in a browser. Press `Ctrl+C` in PowerShell to stop the app.

WSL and OpenFOAM do not need to be started separately. AeroLab starts the configured solver backend when **Validate Mesh** or **Run Solver** is pressed.

## Supported Vehicles

AeroLab does not recognize or tune for one car model. It derives the tunnel, orientation mapping, refinement regions, reference estimates, and fidelity checks from the transformed STL selected for each case. The same workflow accepts passenger cars, trucks, vans, open-wheel vehicles, motorcycles, and custom bodies when their geometry passes inspection.

Vehicle type does not remove the need for a physically correct setup. Supply the real dimensions, orientation, road clearance, and smallest relevant aero feature for that specific STL. A single imported STL is currently treated as one stationary body patch; separately rotating wheels, porous radiators, fans, and moving aero require additional boundary-condition support.

## Trust Checklist

Do not treat a result as accurate until all of these are true:

| Requirement | Evidence in AeroLab |
| --- | --- |
| The scan represents the intended exterior | Inspect the solid shaded STL and its edges from every side. |
| The surface is closed and valid | Watertight, manifold, and triangle-quality checks pass. |
| Repair preserved the body | Repair fidelity passes and the prepared model still has the expected openings and body lines. |
| Orientation is correct | The nose faces into the wind arrow, the roof is up, and the model is level. |
| Physical scale is known | Measured length, width, and height agree within 2%. |
| Road placement is correct | Ground is enabled for a road car and the lowest-point road gap matches the real geometry. |
| Relevant details are resolved | The smallest aerodynamic feature is entered and has at least four estimated cells across it. |
| The CFD mesh is usable | Baseline `checkMesh`, meshed-body fidelity, and boundary-layer coverage pass. |
| The solver settled | Residual, force-stability, and body `y+` checks pass. |
| Results are mesh-independent | Draft, standard, and fine cases complete a passing Accuracy Study. |
| The operating point is valid | Speed stays below Mach 0.3 and matches the comparison being made. |

A green individual run proves that run passed its numerical gates. A passing three-grid Accuracy Study is still required before using final `Cd` or `Cl` values.

## Prepare The Scan

1. Scan or model the complete exterior that touches the airflow.
2. Include the underbody, wheel faces, wings, splitters, diffusers, and ducts that matter to the test.
3. Remove interior detail that is completely sealed away from the external airflow.
4. Avoid duplicated shells, self-intersections, floating fragments, and paper-thin overlapping panels.
5. Export an STL without reducing the triangle count so far that body lines become faceted.
6. Record real length, width, height, and the lowest modeled point's road clearance.

An STL is a triangle surface, not a solid volume file. It becomes a valid CFD obstacle when the triangles form one watertight, manifold boundary with consistent usable thickness.

## Load And Inspect

1. Select the STL in AeroLab.
2. Choose the source units. Scanners commonly export millimeters even when the file has no unit metadata.
3. Wait for geometry inspection to finish.
4. Rotate and zoom around the shaded model.
5. Enable **Edges** to inspect the original triangle network.
6. Check **Display tris** to confirm the complete STL is being rendered.

The displayed body should be solid and retain the exact body lines present in the STL. Adding triangles after scanning does not recreate detail that was never captured.

### Automatic Diffuser Scan

After loading or rotating a model, AeroLab scans the rear underbody for connected surfaces that rise toward the rear like a diffuser. The readiness panel reports the strongest candidate's ramp angle, length, width, and confidence. Confirm the highlighted geometry visually: STL files contain triangles only, not part names, vehicle semantics, or proof that a ramp was designed to generate downforce. A missed candidate does not prove that no diffuser exists, especially on a damaged, coarsely scanned, or unusually oriented model.

## Repair Geometry

Use **Prepare Scan** only when watertight or manifold checks fail.

AeroLab preserves the original upload and writes a separate file under `models/prepared`. Small holes and isolated overlapping defects can be repaired locally. More extensive damage may require a voxel-shell repair, which can seal vents, grille openings, ducts, or small wing gaps.

After repair:

1. Inspect the prepared body from every side.
2. Confirm openings that should carry airflow are still open.
3. Confirm splitters, diffusers, wing gaps, and panel edges were not rounded away.
4. Require the repair-fidelity check to pass.

Reject the repair and clean the source in Blender, MeshLab, or a CAD tool when the prepared model changes aerodynamic geometry.

## Align And Scale

1. Press **Auto-align** for a principal-axis starting point.
2. Confirm the nose points into the wind-direction indicator.
3. Confirm the roof is up and the road-contact plane is level.
4. Correct X, Y, and Z rotation with the sliders or numeric fields.
5. Enter the real vehicle length.
6. Enter independently measured width and height.

AeroLab always calculates geometric length, width, and height from the transformed STL coordinates. STL files do not store a required unit or certify the scan's real-world scale, so those calculated values cannot replace physical evidence. Do not make the measured dimensions match by copying values from AeroLab; enter tape, manufacturer, or CAD measurements as an independent check against scan distortion and incorrect units.

## Configure The Wind Tunnel

### Speed

AeroLab uses standard-air properties and records speed, Mach number, dynamic pressure, and Reynolds number in every new case. The current solver is incompressible and blocks Mach 0.3 or higher, approximately 230 mph in standard air.

Every comparison must use the same speed unless speed is the variable being studied. A new speed requires a new solver run; changing the preview control does not alter an existing OpenFOAM result.

### Ground

Enable **Ground** and **Moving ground** for a road car. Enter the measured **Lowest-point road gap mm**:

- Use `0` when the STL includes tires or geometry that truly contacts the road.
- Use a measured positive gap for a body-only model.
- Do not use `0` merely to make a floating body look grounded.

The current single-body workflow does not rotate wheel patches. Moving ground is still more representative than a stationary road, but wheel-rotation effects require separately identified wheel surfaces and a dedicated rotating-wall setup.

### Reference Values

Use a measured or CAD frontal area for final published `Cd` comparisons. If left blank, AeroLab estimates area from the union of projected STL triangles, preserving visible gaps and concavities. Reference length defaults to the flow-axis model length.

Keep reference area and length identical across cases being compared.

### Smallest Aero Feature

Enter the physical width of the smallest feature whose pressure or vortex effect matters, not the smallest triangle in the STL. Examples include:

- diffuser strake thickness or spacing;
- vortex-generator height or chord;
- splitter edge thickness;
- wing-element gap;
- duct, grille, or vent opening;
- wheel-to-body gap.

AeroLab requires at least four estimated sharp-feature cells across this value. Four cells are a minimum screening threshold, not proof of detailed vortex accuracy. Use **Fine** and a passing grid study for small-device comparisons. Draft and Standard stop at refinement level 8; Fine stops at level 9 with a larger cell budget. If the requested feature exceeds that local limit, AeroLab rejects the case before starting an excessively large mesh. Use a physically meaningful larger target, simplify or isolate the region, or move the case to a larger compute system.

## Choose CFD Quality

| Preset | Use |
| --- | --- |
| Draft | Orientation, scale, tunnel, and early mesh checks. Do not use for final aero comparisons. |
| Standard | Normal local solver checks after geometry setup is correct. |
| Fine | Final member of an accuracy study and higher-confidence design comparisons. |

Standard and fine request prism layers and calculate absolute layer thickness from Reynolds number. Requested layers are not assumed to exist: AeroLab reads the final snappyHexMesh summary and requires at least 70% stack completion, at least three prism layers on average, a passing baseline mesh, and compatible solved body `y+`. Local sharp corners may shed layers to preserve valid cells; broad loss still fails the gate.

### Choose The Flow Mode

Use **Steady RANS** for orientation checks, mesh development, and faster estimates of a settled mean flow. Use **Transient + averaged** for bluff bodies, open-wheel vehicles, hatchback or truck wakes, and comparisons where changing separation matters. Transient mode uses adaptive PIMPLE time stepping, develops the wake for several vehicle flow-through lengths, then averages velocity, pressure, turbulence, wall shear, `Cd`, and `Cl` over a physical-time window. It costs more runtime and does not turn a coarse mesh into an accurate one.

## Create And Run A Case

1. Resolve every red CFD Readiness item.
2. Review warnings and decide whether each one is acceptable for the intended test.
3. Press **Create OpenFOAM Case**.
4. Select the created case in the Files list.
5. Press **Validate Mesh** and review every mesh gate.
6. Correct the STL, feature size, placement, or mesh setup when validation is red.
7. Press **Run Solver** after the mesh is reusable.
8. Leave AeroLab, PowerShell, and the computer running until completion.

Creating a case only writes solver files. The STL remains in preview until the OpenFOAM pipeline finishes and solver-generated visualization files are available.

Progress colors:

| Color | Meaning |
| --- | --- |
| Blue | Case is prepared but has not run. |
| Cyan | OpenFOAM is running. |
| Green | Run completed and passed individual-run verification. |
| Amber | Process completed but one or more trust gates require review. |
| Red | Solver or required validation failed. |

## Read The Results

### Preview Airflow

The initial animated smoke is geometry-guided setup visualization. It is useful for checking orientation and gross obstruction, but it is not Navier-Stokes solver output and must not be used to judge a diffuser, vortex generator, wake size, `Cd`, or `Cl`.

### Solved Flow Traces

After a successful OpenFOAM run, solver-derived streamlines and animated particles replace preview smoke. Use **Lines**, **Particles**, or **Both** to choose the display. Both use the solved local-speed color legend, while `Cp`, temperature, and drag keep separate surface legends. Transient cases are labeled **Mean-flow speed** because their paths use the time-averaged velocity field; the particle motion is visual and must not be interpreted as instantaneous transient structure or elapsed solver time. Trace density is a visualization choice, not physical concentration or mass-flow density.

### Cp Surface

Use **Cp** to inspect solved surface pressure coefficient. Read values from the displayed legend. Compare matching locations and identical operating conditions between cases.

### Drag Areas

Use **Drag areas** to find local pressure-drag and skin-friction contributions in the flow direction. A bright local region can identify an improvement target, but local positive drag alone does not equal the complete vehicle drag. Confirm the integrated force coefficient and its stability.

### Cd And Cl

- `Cd` is drag normalized by dynamic pressure and reference area.
- `Cl` is lift normalized by the same dynamic pressure and area.
- Negative `Cl` normally represents downforce under the current axis convention.
- AeroLab reports total downforce or lift and drag in newtons and pounds-force at the case's solved speed.

The force conversion is `F = coefficient x 0.5 x air density x speed^2 x reference area`. It uses the run's mean coefficient, not the animated preview. Do not change the speed control and scale an old result as though it were a new solution: coefficient and flow separation can change with Reynolds number, ride height effects, and wake behavior. Create and solve a separate case for each speed you want to compare.

Never compare coefficients calculated with different reference areas, orientations, road setups, or unresolved meshes.

## Run An Accuracy Study

Use **Create Accuracy Study** after the setup case behaves correctly.

Required inputs:

1. measured length;
2. measured width;
3. measured height;
4. smallest relevant aerodynamic feature;
5. consistent ground, speed, orientation, reference area, and reference length.

AeroLab creates matched draft, standard, and fine cases. Run all three. Grid independence passes only when:

- every case passes its individual trust gates;
- actual cell counts increase at each level;
- `Cd` converges and standard-to-fine change is no more than 2%;
- standard-to-fine `Cl` change is no more than 0.02.

If the study fails, do not average the results. Investigate mesh quality, feature resolution, force drift, layers, `y+`, domain setup, and geometry fidelity, then create a new study.

## Understand The Verification Gates

| Gate | What failure means |
| --- | --- |
| Geometry fidelity | The repair record is absent, rejected, or does not match the prepared STL. |
| Measured dimensions | Scale is unverified or differs from physical measurements by more than 2%. |
| Aero feature resolution | No meaningful feature target was entered or fewer than four cells span it. |
| Mesh quality | Baseline OpenFOAM mesh checks did not pass. The solver is stopped before force calculation. |
| Meshed body fidelity | OpenFOAM's actual body patch lost or distorted the transformed STL. |
| Boundary-layer coverage | Too little of the body received complete prism-layer stacks. |
| Wall resolution | Solved body `y+` is inconsistent with the selected wall-function setup. |
| Residual convergence or stability | Steady equations did not settle, or transient residuals exceeded the divergence ceiling. |
| Force stability | Steady final samples or transient time-window means are still drifting. Physical transient oscillation is allowed when its mean is stable. |
| Courant control | Transient time steps exceeded the accepted Courant limit. |
| Time averaging | A transient run did not reach its averaging window or write complete mean fields. |
| Grid independence | Force coefficients still depend materially on mesh density. |

## Current Physical Limits

AeroLab currently targets steady RANS and transient time-averaged URANS incompressible external flow with OpenFOAM Foundation v13. Treat these as real limits:

- no compressible flow at Mach 0.3 or above;
- no transient vortex-shedding history from a steady run; choose transient mode when wake motion matters;
- no LES/DES resolution of highly unsteady small vortices;
- no automatic wheel-patch separation or wheel rotation from a single STL;
- no automatic fan, radiator, porous grille, heat-transfer, or internal-flow model;
- automatic repair may intentionally seal narrow openings;
- a visually smooth STL does not prove scan dimensional accuracy;
- a verified numerical result still benefits from comparison with wind-tunnel, coast-down, track, or published reference data.

Use AeroLab for controlled relative comparisons inside this model envelope. Escalate to LES/DES, compressible, rotating-wheel, or thermal CFD when the design question depends on those effects.

## Troubleshooting

### The app does not open

Confirm the startup PowerShell window is still open, then visit `http://127.0.0.1:8765/`. Review:

```text
outputs/aerolab-app.out.log
outputs/aerolab-app.err.log
```

### The model is upside down or angled

Use **Auto-align**, confirm the wind arrow, then correct rotation numerically. Recreate the case after changing orientation.

### The model looks transparent or faceted

Use the shaded surface mode and inspect **Display tris**. Edges intentionally reveal tessellation. Faceting in the solid view comes from the STL geometry; CFD refinement cannot recover missing scan shape.

### Airflow still looks like preview lines

Creating or validating a mesh is not solving airflow. Select the case and press **Run Solver**. Solved flow appears only after OpenFOAM exports its post-processing data.

### The run completed but is amber

Read every failed item in CFD Readiness. Completion only means the process stopped; it does not mean the result is trustworthy.

### WSL or OpenFOAM is unavailable

Press **Check Solver**. The recommended backend is OpenFOAM Foundation v13 inside the `AeroLab-Ubuntu` WSL2 distribution.

## Project Files

```text
models/uploads/     Original imported STL files
models/prepared/    Non-destructively repaired STL files and fidelity records
cases/              Generated OpenFOAM cases and solver results
outputs/            App logs and exported reports
docs/               Project and usage documentation
```

Keep the original scan, prepared STL, `case.json`, solver logs, and accuracy-study cases together when documenting a result.
