"""Local OpenFOAM solver orchestration, output parsing, and result assessment.

This package is split into focused modules, but the public surface is
re-exported here so callers keep importing everything from ``aerolab.solver``.
"""

from __future__ import annotations

from .core import *  # noqa: F401,F403  (public API re-export)

# Public names that live in focused submodules.
from .backends import OPENFOAM_BOOTSTRAP, solver_status  # noqa: F401
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

# Underscored helpers that are part of the tested surface.
from .core import (  # noqa: F401
    _clear_previous_solver_outputs,
    _mesh_input_fingerprint,
    _mesh_record_reusable,
)
from .backends import _run_command  # noqa: F401
