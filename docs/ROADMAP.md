# AeroLab Roadmap

## Phase 1: Local Geometry Foundation

- STL mesh inspection
- Basic geometry metrics
- CFD-readiness report
- Case metadata creation

## Phase 2: OpenFOAM Case Generation

- Create external-flow wind tunnel dimensions from model bounds
- Generate OpenFOAM dictionaries
- Add `blockMesh` and `snappyHexMesh` setup
- Copy model into `constant/triSurface`
- Produce repeatable local run commands

## Phase 3: Local Solver Runner

- Detect WSL2 or Docker
- Run OpenFOAM commands locally
- Track solver logs
- Fail clearly when meshing or solving breaks

## Phase 4: Results

- Extract drag and lift force coefficients
- Export pressure and velocity fields
- Generate ParaView state files
- Create HTML/Markdown report summaries

## Phase 5: Vehicle Mode

- Moving ground
- Rotating wheel regions
- Front/rear lift balance
- Underbody and diffuser-focused metrics
- Compare baseline versus modified aero parts

## Phase 6: Interface

- Local browser UI
- Drag-and-drop model upload
- Animated airflow preview
- Real STL preview rendering
- Input unit scaling for scan exports
- Manual reference area and length for force coefficients
- Projected mesh area for automatic `Aref`
- Solver status, run, and report commands
- Case presets
- Result comparison dashboard

## Phase 7: Real 3D And Solver Data

- Bundle Three.js locally instead of depending on a CDN
- Add particle traces from OpenFOAM velocity fields
- Color model surfaces using pressure coefficient data
- Add full real drag/lift result cards after solver run
