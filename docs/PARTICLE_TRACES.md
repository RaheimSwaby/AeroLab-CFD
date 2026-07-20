# Animated Particle Traces — Implementation Plan

Status: **implementation-ready.** This defines the browser-only Phase 7 feature
"Add particle traces from OpenFOAM velocity fields."

## Summary

Particle traces can be implemented entirely in the existing browser viewer. No
OpenFOAM dictionary changes, backend parsing, API endpoint, or case-generation
changes are required. OpenFOAM has already integrated the paths, and
`parse_streamlines` already returns the geometry and local speed needed to
animate particles along them.

The first version will:

- animate particles along solved streamline paths;
- offer `Lines | Particles | Both`, defaulting to `Both`;
- color both solved lines and particles by local speed;
- label transient results as mean-flow visualization;
- retain pressure, temperature, and drag as separate surface fields; and
- fall back to solved lines if WebGL particle rendering is unavailable.

## Existing data and viewer architecture

AeroLab writes an OpenFOAM `streamlines` function object into every case
(`system/streamlines`) with `direction forward`. It samples `U`/`UMean`,
`p`/`pMean`, and a turbulence field. After a completed run,
`aerolab.solver.visualization.parse_streamlines` returns a bounded payload:

```jsonc
{
  "lineCount": 220,
  "pointCount": 41000,
  "timeAveraged": false,
  "speedRange": [0.0, 46.2],
  "pressureRange": [-1200.0, 900.0],
  "lines": [
    [ [x, y, z, speed, pressure], ... ]
  ]
}
```

Important properties:

1. Points are already transformed into the canonical flow frame, centered, and
   scaled with the geometry preview's `normalizedCenter` and `normalizedScale`.
2. Per-point physical speed is at index 3, with a global `speedRange`.
3. `timeAveraged` distinguishes transient `UMean` tracks from final-field `U`
   tracks.
4. The parser limits output to 220 lines and about 500 points per line.
5. The viewer uses two stacked canvases: Canvas2D renders the tunnel and solved
   lines, while Three.js renders the model. Particles belong in the existing
   Three.js scene so they can be depth-tested against the model.

The points still require the same runtime `meshGroundOffset()` applied to the
model and Canvas2D solved lines.

## Design decisions

### Flow-field semantics

The current implementation colors solved ribbons by pressure even though the
product documentation describes them as speed-colored. Particle traces require
a speed scale, so the first version will make the solved-flow semantics
consistent:

- solved ribbons use the speed ramp;
- particles use the same speed ramp;
- a dedicated speed legend is shown whenever solved flow is visible; and
- surface `Cp`, temperature, and drag retain their existing independent legend.

When both a solved-flow legend and a surface-field legend are visible, they are
stacked rather than sharing a scale.

### Visual time, not physical time

Streamline coordinates are normalized viewer units while speed values are
physical metres per second. They must not be combined directly as
`distance += speed * dt`.

For each usable segment, compute its viewer-space length `ds` and a dimensionless
speed ratio:

```text
speedRatio = max(meanSegmentSpeed / globalMaximumSpeed, speedFloor)
segmentTravelTime = ds / speedRatio
```

For a degenerate speed range, use one constant visual speed ratio. A particle
stores its position in this cumulative **visual travel-time** coordinate:

```text
travel += dt * animationRate
travel %= totalTravelTime
```

The segment's cumulative travel-time interval determines the interpolation
amount between its endpoints. This preserves relative local-speed differences
without pretending that the animation clock is solver time.

Particles are initially spaced uniformly in visual travel time. They therefore
appear closer together in slow regions immediately instead of waiting for an
initially uniform arc-length distribution to distort.

## Data preparation

Prepare data once for each new `solverStreamlines` object:

1. Reject lines with fewer than two finite points.
2. Collapse consecutive duplicate positions.
3. Store coordinates and speeds in typed arrays.
4. Precompute cumulative visual travel time for every line.
5. Allocate particles across lines in proportion to line travel time, targeting
   roughly 24 particles per source line and capping the total near 5,200.
6. Store each particle's line index, travel position, and current segment index.
7. Preallocate position and color `Float32Array` buffers.

Keeping a segment cursor makes normal per-frame interpolation `O(N)`. A particle
that crosses multiple short segments advances the cursor until it reaches the
correct interval; wrapping resets the cursor to zero.

## Animation and rendering

### Shared animation loop

Reuse the existing `requestAnimationFrame` loop. `tickViewer` already computes a
clamped `dt`; pass it into the render path instead of discarding it.

The shared loop remains active for the viewer. Particle buffer updates are
skipped when:

- the mode is `Lines`;
- solved streamlines are unavailable;
- the document is hidden; or
- WebGL initialization failed.

A direct redraw caused by orbit, zoom, or a control change uses `dt = 0`, which
updates projected state without advancing particles.

### Three.js layer

Use one `THREE.Points` object attached to the existing viewer group:

- one `BufferGeometry`;
- one preallocated position attribute;
- one preallocated color attribute;
- a small radial sprite texture for round particles;
- additive blending;
- depth testing enabled and depth writes disabled; and
- the same camera, group translation, and `meshGroundOffset()` as the model.

Both position and color attributes are updated in place. Color must change as a
particle enters a segment with a different speed.

### Canvas2D solved lines

Keep solved ribbons on Canvas2D, but cache their projected geometry and local
speed instead of pressure. Draw the underlay with the shared local-speed ramp
and use a thin neutral moving dash as the direction cue so a path-average color
does not obscure local values. The line animation remains based on absolute
frame time; the particle animation uses `dt`.

## Controls and labels

Add a second viewer control group:

```text
Solved flow   Lines | Particles | Both
```

Behavior:

- default to `Both` when solved streamlines are available;
- keep the selected mode while switching cases in the current session;
- disable all three buttons when solved streamlines are absent or contain an
  `error`;
- show the parser error, or explain that a solver run is required;
- disable particle modes and select `Lines` if WebGL is unavailable;
- label transient results `Mean-flow speed`; and
- label steady/final results `Final-field speed`.

The mean-flow label is required because animating `UMean` paths must not imply
that instantaneous transient structures are being displayed.

## Lifecycle and disposal

Particle state is keyed by the `solverStreamlines` object returned for the
active report. On model load, case switch, report replacement, or missing flow:

1. remove the old `THREE.Points` object from the group;
2. dispose its geometry, material, and sprite texture;
3. clear typed-array and prepared-line references; and
4. reset the cached Canvas2D solved-flow layer.

Changing surface mode may rebuild model geometry but must not dispose the
independent particle object.

## Edge cases

| Case | Behavior |
| --- | --- |
| `solverStreamlines.error` present | Disable flow controls and surface the message |
| Fewer than two finite distinct points | Skip the line |
| Degenerate `speedRange` | Uniform color and constant visual advection |
| Zero or near-zero local speed | Clamp to a small visual floor so particles recycle |
| Particle-only mode with WebGL failure | Fall back to `Lines`, use the neutral Canvas model, disable solved surface-field modes, and explain why |
| Case switched or model loaded | Dispose stale particle state before redraw |
| Browser tab hidden | Keep rendering lifecycle intact but skip particle advancement |

## Performance budget

Target about 24 particles per usable line, capped near 5,200. With at most 220
source lines, this remains one draw call and linear typed-array work per active
frame. No arrays, vectors, colors, or materials are allocated inside the
particle loop.

## Known limitations

1. **Transient cases show mean-flow paths, not instantaneous flow.** The label
   must say `Mean-flow speed` when `timeAveraged` is true.
2. **Particles remain confined to seeded streamlines.** Arbitrary free-space
   particles would require volumetric velocity output and browser interpolation,
   which is a different feature.
3. **Streamlines are browser-decimated.** Particle density is a visualization of
   relative residence time along the sampled tracks, not physical concentration
   or mass-flow density.
4. **Animation time is intentionally visual.** It preserves local speed ratios
   but does not represent elapsed solver seconds.

## Delivery stages

### Version 1 — this implementation

- typed-array path preparation and visual-time advection;
- one Three.js particle draw call;
- speed-colored solved lines and particles;
- `Lines | Particles | Both` controls;
- separate speed and surface-field legends;
- mean/final-field labeling; and
- lifecycle cleanup and WebGL fallback.

### Future polish

- particle density control;
- particle size or opacity controls;
- pause or animation-rate control; and
- executable JavaScript unit tests if the project adopts a JS test runner.

## Validation

The repository currently has no JavaScript test runner. Validate the feature by:

1. running Ruff and the complete Python unit-test suite;
2. loading a report with final-field streamlines and checking all three modes;
3. loading a transient report and verifying the `Mean-flow speed` label;
4. orbiting and zooming to verify particle/model alignment and depth testing;
5. switching cases to verify stale particles disappear; and
6. confirming speed and surface-field legends remain distinct in `Both` mode.

## Post–Version 1 game plan

Version 1 (commit `70ae6ce`) landed the advection, rendering, the
`Lines | Particles | Both` control, dual legends, mean/final-field labeling, and
WebGL fallback — i.e. it absorbed what an earlier draft split across two stages.
What remains is lock-in, polish, and one strategic decision.

Everything tunable is currently a hardcoded constant
(`SOLVER_PARTICLE_TARGET_PER_LINE = 24`, `SOLVER_PARTICLE_MAX_COUNT = 5_200`,
`SOLVER_PARTICLE_ANIMATION_RATE = 0.9`; material `size: 0.052`,
`opacity: 0.92`). The polish work is therefore a repeatable pattern: expose a
constant as a control and persist it to `localStorage`, following the existing
`"aerolab-invert-orbit"` convention.

Recommended order: **A → C (decide) → B → D**. Lock the foundation, make the
strategic call so polish is not spent on an architecture that might be replaced,
then polish, then decide on JS tooling.

### Stage A — Lock in what exists (do first)

Not new features; makes Version 1 durable before more is built on it.

- **Wiring tests** in `tests/test_webapp.py`, following the
  `test_inverted_orbit_control_is_wired_and_persistent` pattern: assert the
  `solverLinesButton` / `solverBothButton` controls exist in `index.html`; that
  `prepareSolverParticles`, `updateSolverParticles`, `resetSolverParticles`, and
  `setSolverFlowMode` are referenced in `app.js`; and that both `Mean-flow speed`
  and `Final-field speed` label strings are present.
- **Manual visual pass** — the Validation checklist above, run in a browser on a
  machine that can reach the local app.
- Effort: small. Risk: none. Rationale: ~800 lines shipped with no automated
  guard, so a later refactor could silently unwire the controls or labels.

### Stage B — Interaction polish

The "Future polish" items, ordered cheapest to dearest by how deeply they touch
the render path:

1. **Pause + animation-rate control** — scale `dt` by a rate factor; pause is
   rate 0. Per-frame only, no buffer rebuild. Trivial.
2. **Particle size / opacity controls** — write `material.size` /
   `material.opacity` and trigger a `dt = 0` redraw. No buffer rebuild. Easy.
3. **Particle density control** — changing particles-per-line reallocates the
   typed arrays, so it must re-run `prepareSolverParticles` rather than tweak a
   uniform. Debounce the input so a drag does not rebuild every frame. Moderate.

Effort: small–moderate. Risk: low. Each item is expose-constant → control →
persist.

### Stage C — Strategic decision (not code)

The open question from this document: **are seeded-streamline particles
sufficient, or is particle seeding anywhere in the domain a real requirement?**

- Version 1 visualizes seeded flow topology. Free-space particles require
  volumetric velocity output (new OpenFOAM sampling), a payload well beyond the
  current 128 MB streamline guard, and browser-side field interpolation — the
  volumetric approach this document explicitly rejected. It is a different
  feature, not an increment.
- Decide this before investing in Stage B, because a free-space requirement would
  replace the architecture some polish work targets.

### Stage D — JavaScript test tooling (optional, infrastructure)

Stage A asserts control *presence*, not *behavior*. Unit-testing the advection
and arc-length math would mean adopting a JS runner (for example Vitest or Node's
built-in test runner); the pure functions (`prepareSolverParticles`,
`writeSolverParticleColor`) are already extractable. This is a project-level
decision about adding a Node toolchain to a Python-first repository and is worth
treating on its own.
