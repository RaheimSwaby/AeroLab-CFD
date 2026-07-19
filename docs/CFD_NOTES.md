# CFD Notes

## First Simulation Target

Start with steady, incompressible external airflow.

Good default assumptions:

- Solver family: OpenFOAM
- Turbulence model: k-omega SST
- Air density: 1.225 kg/m^3
- Kinematic viscosity: 1.5e-5 m^2/s
- Initial speed: 70 mph

## Model Requirements

The model should be:

- Watertight
- Correctly scaled
- Oriented consistently
- Free of internal surfaces
- Clean enough to mesh
- Not overloaded with tiny scan noise

## Scan Unit Handling

STL files do not reliably carry units. A model that visually looks correct can still be numerically wrong for CFD if the scan is exported in millimeters, centimeters, or inches and the solver assumes meters.

AeroLab stores both:

- Raw geometry report
- Meter-scaled geometry report

Generated OpenFOAM `body.stl` files are written in meters according to the selected input units.

## Reference Area And Length

Drag and lift coefficients depend on reference values:

- `Aref`: reference/frontal area in m2
- `lRef`: reference length in m

For a car, the most useful `Aref` is usually frontal area. AeroLab can estimate this from the bounding box, but a measured or known value is better.

The current automatic `Aref` estimate uses the STL's projected mesh area along the selected flow axis. For clean closed scans, this is closer to a real frontal-area estimate than the bounding-box face. For open, noisy, or non-manifold scans, treat it as a rough estimate and enter a measured value when available.

## Result Loop

Generated cases include OpenFOAM `forceCoeffs` output. After a solver run, AeroLab reads the latest coefficient file and reports:

- `Cd`
- `Cl`
- `Cs`
- `CmPitch`

## Vehicle-Specific Requirements Later

For realistic car aero, add:

- Moving ground plane
- Rotating wheels
- Correct ride height
- Sealed or modeled engine bay openings
- Underbody detail
- Separate regions for aero parts
