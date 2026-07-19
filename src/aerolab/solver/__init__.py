"""Local OpenFOAM solver orchestration, output parsing, and result assessment.

This package is split into focused modules, but the public surface is
re-exported here so callers keep importing everything from ``aerolab.solver``:

- backends:      backend detection (native/WSL/Docker) and command execution
- run:           run orchestration, mesh reuse, and the run result type
- parsing:       parsers for OpenFOAM output (force coeffs, residuals, y+, checkMesh)
- visualization: streamlines, surface pressure, and mesh preview data
- analysis:      case reporting plus fidelity, convergence, and quality assessment
- util:          shared numeric/JSON primitives
"""

from __future__ import annotations

from .backends import (  # noqa: F401
    OPENFOAM_BOOTSTRAP,
    solver_status,
    _run_command,
)
from .parsing import (  # noqa: F401
    parse_check_mesh,
    parse_force_coeffs,
    parse_layer_coverage,
    parse_residuals,
    parse_transient_state,
    parse_y_plus,
)
from .visualization import (  # noqa: F401
    CASE_PREVIEW_TRIANGLE_LIMIT,
    parse_streamlines,
    parse_surface_pressure,
)
from .analysis import (  # noqa: F401
    assess_meshed_surface_fidelity,
    case_report,
    case_run_progress,
)
from .run import (  # noqa: F401
    SolverRunResult,
    run_case,
    _clear_previous_solver_outputs,
    _mesh_input_fingerprint,
    _mesh_record_reusable,
)
