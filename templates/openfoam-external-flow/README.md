# OpenFOAM External Flow Template

This folder will hold the generated OpenFOAM setup for external airflow simulations.

Planned contents:

- `0/` initial and boundary conditions
- `constant/` turbulence and transport properties
- `constant/triSurface/` imported model STL
- `system/` meshing and solver dictionaries

The first generated solver target should use:

- `blockMesh`
- `surfaceFeatureExtract`
- `snappyHexMesh`
- `simpleFoam`
- force coefficient post-processing
