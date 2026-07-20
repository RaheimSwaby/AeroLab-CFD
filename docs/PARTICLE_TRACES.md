# Animated Particle Traces — Implementation and Roadmap

Status: **Version 1 implemented.** Commit `70ae6ce` shipped the browser-only
Phase 7 feature "Add particle traces from OpenFOAM velocity fields." The
post-Version 1 roadmap below is proposed and implementation-ready.

## Summary

Particle traces are implemented entirely in the existing browser viewer. No
OpenFOAM dictionary changes, backend parsing, API endpoint, or case-generation
changes are required. OpenFOAM has already integrated the paths, and
`parse_streamlines` returns the geometry and local speed needed to animate
particles along them.

Version 1:

- animates particles along solved streamline paths;
- offers `Lines | Particles | Both`, defaulting to `Both`;
- colors both solved lines and particles by local speed;
- labels transient results as mean-flow visualization;
- retains pressure, temperature, and drag as separate surface fields; and
- falls back to solved lines if WebGL particle rendering is unavailable.

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

Before Version 1, the viewer colored solved ribbons by pressure even though the
product documentation described them as speed-colored. Particle traces require
a speed scale, so Version 1 made the solved-flow semantics consistent:

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

The viewer includes a second control group:

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

## Delivery status

### Version 1 — implemented in `70ae6ce`

- typed-array path preparation and visual-time advection;
- one Three.js particle draw call;
- speed-colored solved lines and particles;
- `Lines | Particles | Both` controls;
- separate speed and surface-field legends;
- mean/final-field labeling; and
- lifecycle cleanup and WebGL fallback.

### Post-Version 1 — proposed order

Proceed in the order **A → C → B1 → B2 → B3 → D**:

1. lock the shipped behavior with a focused static test and a manual baseline;
2. record the seeded-versus-volumetric product decision;
3. add motion, appearance, and density polish as separate reversible changes;
4. add JavaScript unit-test infrastructure only if its maintenance cost is
   accepted.

## Version 1 baseline validation

The repository currently has no JavaScript test runner. The shipped feature's
baseline is:

1. run Ruff and the complete Python unit-test suite;
2. load a report with final-field streamlines and check all three flow modes;
3. load a transient report and verify the `Mean-flow speed` label;
4. orbit and zoom to verify particle/model alignment and depth testing;
5. switch cases to verify stale particles disappear; and
6. confirm speed and surface-field legends remain distinct in `Both` mode.

## Post-Version 1 implementation roadmap

### Current implementation constraints

All tunable values are hardcoded in `src/aerolab/web/app.js`:

| Setting | Current value |
| --- | ---: |
| Target particles per line | `24` |
| Maximum total particles | `5_200` |
| Base animation rate | `0.9` |
| Point size | `0.052` |
| Point opacity | `0.92` |

`prepareSolverParticles` is not a pure function: it mutates viewer state and
calls `resetSolverParticles`. Density changes therefore require an explicit
rebuild lifecycle, not an assignment to an existing material. The project also
has no `package.json`, JavaScript test runner, Node setup in CI, or importable
side-effect-free particle module.

The browser currently stores independent preferences under keys such as
`aerolab-invert-orbit` and `aerolab-view-mode`. Particle polish should instead
use one versioned structured record so validation, migration, and restoration
remain atomic.

### Shared contract for Stages B1–B3

Persist the following object under `aerolab-particle-settings-v1`:

```json
{
  "paused": false,
  "rateMultiplier": 1.0,
  "pointSize": 0.052,
  "opacity": 0.92,
  "particlesPerLine": 24
}
```

Rules:

1. Load once during viewer initialization inside `try/catch`. A missing key,
   malformed JSON, array, `null`, or non-object value restores all defaults.
2. Validate each field independently. Missing or invalid fields use their
   defaults; finite numeric fields are clamped to the safety bounds below and
   snapped to the nearest supported preset.
3. Persist the complete normalized object after each user change. Never write
   partial settings or use one storage key per control.
4. A storage write failure must not prevent the in-memory setting from working.
5. Pause is independent from rate. Pausing must not overwrite
   `rateMultiplier`; resuming uses the previously selected rate.
6. Keep `SOLVER_PARTICLE_MAX_COUNT = 5_200` as a non-configurable safety cap.
7. Settings survive report and case changes, while prepared particle buffers do
   not. `resetSolverParticles` must never clear the preference object.
8. If `prefers-reduced-motion: reduce` matches and no valid saved settings exist,
   initialize with `paused: true`. An explicit user Play/Pause choice is then
   persisted and takes precedence on later loads.

Recommended initial presets and safety bounds:

| Setting | Presets | Default | Safety bounds |
| --- | --- | ---: | ---: |
| Rate multiplier | `0.5×`, `1×`, `2×` | `1×` | `0.25–2` |
| Point size | Small `0.040`, Standard `0.052`, Large `0.070` | `0.052` | `0.030–0.090` |
| Opacity | Soft `0.55`, Medium `0.75`, Strong `0.92` | `0.92` | `0.25–1` |
| Particles per line | Low `12`, Standard `24`, High `36` | `24` | `1–48`, then total cap |

Use buttons and selects rather than unrestricted sliders in the first polish
release. This avoids continuous density rebuilds, gives keyboard users discrete
choices, and limits control-panel pressure on narrow screens.

Place the settings directly below `Lines | Particles | Both` in the existing
solved-flow control group. They must follow the same Basic-view visibility rules
as that group. Disable particle-specific controls when flow data is unavailable,
contains an error, WebGL is unavailable, or `Lines` is selected; preserve the
chosen values while disabled. Every control requires a visible label, keyboard
operation, a programmatic accessible name, and an exposed selected/pressed
state. On narrow layouts, rows wrap without horizontal page overflow.

Whenever `app.js` or `style.css` changes, increment its existing query-string
asset version in `index.html` (`app.js?v=65` and `style.css?v=32` at the time of
this plan). Use the values present when implementing rather than assuming these
numbers remain current.

### Stage A — lock in Version 1

#### A1. Add a focused static wiring test

**Files**

- `tests/test_webapp.py`

Add a new test named for particle wiring, separate from
`test_inverted_orbit_control_is_wired_and_persistent`. It should verify:

- `solverLinesButton`, `solverParticlesButton`, `solverBothButton`, and
  `solverFlowStatus` exist in `index.html`;
- `prepareSolverParticles`, `updateSolverParticles`, `resetSolverParticles`,
  `setSolverFlowMode`, and `ensureSolverParticlePoints` remain wired in
  `app.js`;
- both `Mean-flow speed` and `Final-field speed` labels remain present; and
- the particle and line modes are both represented in control synchronization.

Keep this a narrow source-wiring guard. It must not assert implementation line
order or copy the full function bodies.

**Acceptance criteria**

- Removing a flow button, lifecycle call, or required label makes this test fail.
- Existing webapp tests remain unchanged and pass.
- No production file changes are required.

**Validation**

```bash
ruff check tests/test_webapp.py
python -m unittest discover -s tests -p 'test_webapp.py' -v
```

**Risk and rollback:** Very low risk. The test cannot prove numerical or visual
correctness; it only catches accidental unwiring. Roll back by removing this one
test.

#### A2. Record the manual browser baseline

Run the matrix below before interaction controls are added. Record browser,
operating system, report type, and pass/fail evidence in the change description
or linked task.

| Area | Checks |
| --- | --- |
| Flow modes | `Lines`, `Particles`, and `Both`; correct legends in each mode |
| Result semantics | Final-field and transient reports; exact mean/final label |
| Camera | Orbit and zoom close to and far from the model; no unexpected clipping or alignment drift |
| Lifecycle | Switch reports/cases repeatedly; stale particles and legends disappear |
| Missing data | No-flow report and parser-error report disable controls with useful status |
| WebGL | Initialization failure and, where browser tooling permits, context loss fall back to lines without an uncaught error |
| Visibility | Hide/show the tab; hidden time is not replayed as a large particle jump |
| Accessibility | Keyboard traversal, pressed states, focus visibility, and OS reduced-motion setting |
| Responsive layout | Desktop plus a narrow mobile-sized viewport; no clipped controls or horizontal overflow |

The current lack of reduced-motion behavior is an expected B1 gap, not a reason
to misreport the A2 baseline as passing.

**Acceptance criteria**

- No correctness, lifecycle, fallback, or layout blocker remains unexplained.
- Any non-blocking discrepancy is captured against a later stage.
- A reproducible WebGL fallback problem blocks Stage B until fixed.

**Risk and rollback:** No code risk. Manual evidence is not repeatable automation,
which is why A1 and optional Stage D remain separate.

### Stage C — record the product boundary

**Recommended decision:** keep seeded-streamline particles for the current
product. Treat arbitrary free-space seeding as a separate epic only after a
concrete user workflow demonstrates that the existing seeded topology is
insufficient.

This is a product and architecture gate, not an implementation task. Free-space
particles require all of the following outside the current browser-only design:

- volumetric velocity sampling or a new OpenFOAM output format;
- backend parsing, payload limits, and likely compression or spatial tiling;
- browser interpolation and out-of-domain behavior;
- a seeding model, density semantics, and physical-time decision;
- new performance budgets for memory, transfer, and frame time; and
- independent validation against the solver field.

The current 128 MB streamline guard and bounded `lines` payload are not a basis
for shipping volumetric fields.

**Decision record acceptance criteria**

- State either `seeded streamlines are sufficient` or link a separately scoped
  volumetric epic with a named user need.
- If the volumetric epic is chosen, do not extend Stages B2/B3 automatically;
  reassess whether their controls and storage schema still fit the replacement
  renderer.
- Do not describe free-space advection as a continuation of Version 1 polish.

**Risk and rollback:** No code or rollback. The risk is spending polish effort on
a renderer that a confirmed volumetric requirement would replace.

### Stage B1 — pause and animation-rate presets

**Files and touch points**

- `src/aerolab/web/index.html`: add Play/Pause and rate controls; bump asset
  versions for changed browser files.
- `src/aerolab/web/style.css`: add compact, wrapping settings-row styles.
- `src/aerolab/web/app.js`: add normalized settings load/save/sync helpers;
  integrate pause and rate with `updateSolverParticles`,
  `syncSolverFlowControls`, and the existing direct-redraw path.
- `tests/test_webapp.py`: extend the focused particle wiring test with the new
  IDs, storage key, and pause/rate wiring.
- `docs/USER_GUIDE.md`: document controls and reduced-motion startup behavior.

Use an explicit boolean pause state:

```text
effectiveAnimationRate = paused
  ? 0
  : SOLVER_PARTICLE_ANIMATION_RATE * rateMultiplier
```

A paused frame may redraw after orbit, zoom, resize, or a style change, but it
must not advance particle travel. `document.hidden` is a temporary lifecycle
condition and must not mutate or persist `paused`.

**Acceptance criteria**

- Play/Pause works in `Particles` and `Both`; resume retains the selected rate.
- `0.5×`, `1×`, and `2×` visibly change advection without changing path,
  particle count, or solved-line animation semantics.
- A `dt = 0` redraw never advances particles.
- Invalid/corrupt storage restores defaults without an exception and is replaced
  with normalized data after the next user change.
- Reduced-motion users start paused only when they have no valid saved choice.
- Controls disable and re-enable correctly across `Lines`, missing flow, parser
  errors, WebGL fallback, and case switches.
- The narrow layout wraps cleanly and all controls work by keyboard.

**Validation**

```bash
ruff check src tests scripts
python -m unittest discover -s tests -v
node --check src/aerolab/web/app.js
```

Repeat the A2 motion, lifecycle, accessibility, responsive, and fallback rows.

**Risk and rollback:** Low risk because no particle buffers change. Revert B1 as
one slice to restore the current constant rate; no stored setting is required by
older code, so the versioned key can remain harmlessly unused.

### Stage B2 — size and opacity presets

**Files and touch points**

- `src/aerolab/web/index.html`: add size and opacity selects.
- `src/aerolab/web/style.css`: reuse the B1 settings-row layout.
- `src/aerolab/web/app.js`: apply normalized values to the existing
  `THREE.PointsMaterial`, synchronize controls, and persist the full settings
  object.
- `tests/test_webapp.py`: add focused wiring and preset assertions.
- `docs/USER_GUIDE.md`: document the appearance presets.

Changing appearance updates `material.size` and `material.opacity`, then uses the
existing `dt = 0` redraw path. It must not call `prepareSolverParticles`, replace
position/color attributes, or reset particle travel.

**Acceptance criteria**

- All size and opacity combinations apply immediately in `Particles` and
  `Both`.
- The defaults are visually identical to Version 1 (`0.052`, `0.92`).
- Appearance survives reload and case switches through the shared settings key.
- Switching appearance while paused does not move particles.
- No geometry, material, texture, or typed array is allocated per frame.
- Near/far camera checks remain legible without particles dominating the model.

**Validation**

Run the B1 commands and repeat the A2 camera, accessibility, responsive, and
lifecycle rows.

**Risk and rollback:** Low risk. Excessive size/opacity can obscure the model,
which is why values are presets with hard bounds. Revert B2 independently while
retaining the B1 fields in the same versioned record.

### Stage B3 — density presets and safe rebuilds

**Files and touch points**

- `src/aerolab/web/index.html`: add the density select.
- `src/aerolab/web/app.js`: make particles-per-line an input to preparation; add
  a trailing 150 ms rebuild scheduler; integrate cancellation into
  `resetSolverParticles`; retain the fixed total cap.
- `tests/test_webapp.py`: assert density control, safety-cap, debounce, and reset
  wiring.
- `docs/USER_GUIDE.md`: explain that density is visual sampling, not physical
  concentration.

Density is the only polish setting that rebuilds buffers. The lifecycle is:

1. normalize and persist the selected preset;
2. schedule one trailing rebuild after 150 ms;
3. cancel a pending rebuild on report replacement, case switch, model load,
   missing flow, or WebGL fallback;
4. dispose/remove stale particle geometry through the existing reset path;
5. prepare new typed arrays from the active `solverStreamlines` object using
   the selected target and `min(calculatedCount, 5_200)`; and
6. restore the current flow mode, pause state, appearance, and rate without
   clearing preferences.

The High preset may converge on the same 5,200 total as Standard for reports
with many usable lines. That is expected safety-cap behavior, not a reason to
raise the cap.

**Acceptance criteria**

- Low, Standard, and High produce bounded deterministic counts for the same
  report, never exceeding 5,200.
- One user selection produces at most one rebuild after the debounce window.
- Repeated selections, rapid case switches, and reset during a pending timer do
  not resurrect stale particles or leak GPU resources.
- Changing density while paused leaves the rebuilt set paused.
- Reports with zero usable lines remain stable and show the existing status.
- Frame updates remain allocation-free after preparation completes.

**Validation**

Run the B1 commands, then repeat the full A2 matrix while switching density and
cases rapidly. Use browser performance/memory tooling for a short repeated
rebuild check; the retained `THREE.Points`, geometry, material, and texture
counts must return to their steady-state values.

**Risk and rollback:** Moderate risk because this phase changes allocation and
disposal. Ship it separately from B1/B2. Reverting B3 restores the fixed target
of 24 without changing the stored motion or appearance settings.

### Stage D — optional pure module and Node tests

Do this only if the team accepts a pinned Node runtime in a Python-first
repository. Use Node's built-in test runner; do not add a third-party test
framework solely for this feature.

**Files and touch points**

- Add `src/aerolab/web/particle_math.mjs` for side-effect-free preparation,
  allocation, advancement, interpolation, and color math.
- Add `tests/js/particle_math.test.mjs` using `node:test` and
  `node:assert/strict`.
- Update `src/aerolab/web/app.js` to remain the DOM/Three.js lifecycle adapter.
- Update `src/aerolab/web/index.html` to load the browser entry point as a module
  if static imports are used, and bump the script asset version.
- Add `.node-version` containing the exact supported Node 24.x patch selected at
  implementation time.
- Update `.github/workflows/ci.yml` to read that version file and run
  `node --test`.
- Verify the package-data rules in `pyproject.toml`; change them only if the
  built wheel does not contain `particle_math.mjs`.

Do not export the current stateful `prepareSolverParticles` unchanged and call
it unit-testable. Extract pure lower-level functions and leave reset, disposal,
DOM access, Three.js objects, and global viewer state in `app.js`.

Minimum test matrix:

- non-finite points and consecutive duplicate positions;
- degenerate and zero speed ranges;
- speed-floor travel-time calculation;
- deterministic proportional allocation and the 5,200 cap;
- initial spacing in visual travel time;
- segment crossing, multi-segment advancement, and wrapping;
- interpolation at segment boundaries; and
- color normalization and clamping.

**Acceptance criteria**

- Browser behavior and the Version 1/B-stage manual matrix are unchanged.
- `app.js` consumes the exported math rather than retaining duplicate logic.
- Tests are deterministic and require no DOM, WebGL, network, or test fixture
  report.
- CI and local development use the same exact Node version.
- The built wheel contains the `.mjs` module and a clean installation can load
  it from the packaged web app.

**Validation**

```bash
node --input-type=module --check < src/aerolab/web/app.js
node --check src/aerolab/web/particle_math.mjs
node --test tests/js/*.test.mjs
ruff check src tests scripts
python -m unittest discover -s tests -v
uv build --wheel
```

Inspect the wheel contents for `aerolab/web/particle_math.mjs`, then run the A2
flow-mode, lifecycle, and fallback rows against the built package.

**Risk and rollback:** Moderate infrastructure risk: module loading, packaging,
and CI now depend on Node even though runtime rendering remains browser-only.
Keep Stage D in its own change so it can be reverted without removing product
controls.

## Cross-stage non-goals

Unless Stage C creates a separate epic, Stages A–D do not include:

- arbitrary free-space or volumetric particle seeding;
- new OpenFOAM output, backend parsing, or API payload fields;
- physical-time or mass-flow-accurate particle motion;
- changes to streamline decimation, line rendering, or legend semantics;
- per-case particle preferences;
- unrestricted sliders or raising the 5,200 safety cap; or
- a third-party JavaScript test dependency.

Each stage should be reviewed, validated, and revertible on its own. Do not
combine density lifecycle work or Node infrastructure with the low-risk motion
and appearance controls.
