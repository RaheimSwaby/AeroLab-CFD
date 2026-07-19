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
