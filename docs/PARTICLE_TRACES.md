# Animated Particle Traces — Design Plan

Status: **proposed, not implemented.** This covers the Phase 7 roadmap item
"Add particle traces from OpenFOAM velocity fields."

## Summary

Particle traces can be built as a **browser-only feature**. No OpenFOAM
dictionary changes, no new backend parsing, no new API endpoint, and no change
to case generation. Everything required is already in the payload that
`parse_streamlines` sends today.

This was the main finding while scoping the work, and it changes the size of the
job considerably.

## What already exists

AeroLab writes an OpenFOAM `streamlines` function object into every case
(`system/streamlines`), seeded from explicit points, sampling `U`/`UMean`,
`p`/`pMean`, and a turbulence field. After a completed run,
`aerolab.solver.visualization.parse_streamlines` reads the resulting legacy
ASCII VTK track file and returns:

```jsonc
{
  "lineCount": 220,
  "pointCount": 41000,
  "timeAveraged": false,
  "speedRange": [0.0, 46.2],
  "pressureRange": [-1200.0, 900.0],
  "lines": [
    [ [x, y, z, speed, pressure], ... ],   // one polyline per streamline
    ...
  ]
}
```

Three properties of this payload matter for the design:

1. **Geometry is already in viewer space.** Points are transformed to the
   canonical flow frame, then centred and scaled with the same
   `normalizedCenter`/`normalizedScale` used by the geometry preview. They align
   with the rendered STL with no extra transform.
2. **Per-point speed is already present** (index 3 of each point), along with a
   global `speedRange` for normalisation.
3. **It is already decimated** for the browser (`max_lines=220`,
   `max_points_per_line=500`), so the payload size is bounded.

OpenFOAM has, in effect, already performed the path integration. The browser
does not need to integrate a velocity field itself.

## Rejected alternative: volumetric advection

The conventional way to build particle traces is to sample the full 3-D velocity
field onto a regular grid, ship it to the browser, and advect particles through
it (often on the GPU). That was considered and rejected:

- It requires **new OpenFOAM output** (volume field sampling or a full field
  export) and therefore changes case generation.
- The payload is **large**. A standard-quality case targets 2.8 M cells; even a
  coarse resampling is orders of magnitude above the existing 128 MB streamline
  guard.
- It requires **runtime interpolation** in the browser, with its own
  correctness and performance risks.

The only thing it buys is particles in arbitrary regions of the domain rather
than along seeded paths. That trade is not worth it for the first version — see
*Known limitations* below.

## Design

### 1. Data preparation

When a `solverStreamlines` payload arrives, precompute once per line:

- cumulative arc length `s[i]` along the polyline
- total length `L`
- per-vertex speed `v[i]`

Lines with fewer than two points are skipped.

### 2. Advection model

Each particle stores a single scalar: its arc position `u` along its assigned
line. Per animation frame:

```
u += v(u) * dt * timeScale     // v(u) linearly interpolated from v[i] at s[i]
u %= L                          // recycle at the end of the line
```

The particle's world position is the linear interpolation of the polyline
vertices at arc length `u`.

Advancing at a rate proportional to **local** speed is the point of the feature:
particles bunch up in slow and separated regions and streak through fast ones,
which is what makes wakes, stagnation, and reattachment legible. A constant-rate
animation would look plausible and communicate nothing.

### 3. Rendering

- A single `THREE.Points` object with one `BufferGeometry`.
- One preallocated `Float32Array` for positions, updated in place each frame with
  `needsUpdate = true`. No allocation inside the animation loop.
- Per-particle colour from speed, normalised by the existing `speedRange` so the
  particles and the streamline ribbons share one legend and one colour ramp.
- Additive blending with a small round sprite for a smoke/spark appearance.

This is one draw call and `O(N)` scalar work per frame.

### 4. Budget

Target **3,000–6,000 particles** — roughly 15–30 per line across 220 lines. At
that count a single `Points` object with a typed-array update is comfortably
60 fps. Particle count should scale down when `lineCount` is small.

### 5. Integration with the existing viewer

Particle traces are a **display option on top of solver streamlines**, not a
separate surface mode:

- A control offering `lines | particles | both`.
- Reuse the existing speed legend and colour ramp.
- Reuse the animation loop that already drives the airflow preview. Do not add a
  second `requestAnimationFrame` loop.
- Disable the control, with the reason shown, when `solverStreamlines` is absent
  or carries an `error` key.

### 6. Edge cases

| Case | Behaviour |
| --- | --- |
| `solverStreamlines.error` present | Control disabled, message surfaced |
| Line with fewer than 2 points | Skipped during preparation |
| `speedRange` degenerate (min == max) | Uniform colour, constant advection rate |
| Zero-speed segment | Clamp advection to a small floor so particles cannot freeze permanently |
| Case switched, or panel hidden | Stop animating — same discipline as the run-log polling |

## Known limitations

These are intrinsic to the approach and should be stated in the UI rather than
papered over.

1. **Transient cases show mean-flow paths, not instantaneous ones.** When the
   case is transient, the streamlines function object samples `UMean`, so the
   tracks are pathlines of the *time-averaged* field. Animating them can imply
   unsteady structure that is not being shown. The payload already carries
   `timeAveraged`; the label must reflect it, in the same spirit as the
   "Mean field" / "Final field" distinction used for surface temperature.

2. **Particles are confined to the seeded streamline set.** They show the
   topology of the seeded flow, not arbitrary regions of the tunnel. Particles
   anywhere in the domain would require the rejected volumetric approach and is a
   genuinely different design — worth deciding deliberately rather than drifting
   into it.

3. **Streamlines are decimated for the browser.** The rendered paths are a
   sampled subset of the solver's tracks. This is existing behaviour, but it
   means particle density is not a physical quantity and must not be read as one.

## Staging

1. **Stage 1** — arc-length precompute, advection, `Points` rendering. Rendered
   alongside the existing lines; no new control yet.
2. **Stage 2** — the `lines | particles | both` control, legend integration, and
   the transient/mean-flow labelling.
3. **Stage 3** — polish: particle density control, size or opacity varying with
   speed.

Stage 1 is self-contained and independently reviewable.

## Testing

No backend changes means no new Python unit tests for the feature itself.

The repository has no JavaScript test runner, and the existing convention is to
assert that front-end behaviour is wired by checking the contents of
`index.html` and `app.js` — see
`tests/test_webapp.py::test_inverted_orbit_control_is_wired_and_persistent`.
New tests should follow that convention:

- assert the mode control id exists in `index.html`
- assert the advection and particle-setup functions are referenced in `app.js`
- assert the transient/mean-flow label text is present

If stronger guarantees are wanted later, the arc-length and interpolation
helpers are pure functions and could be extracted for a real JS test runner.
That is a separate decision about adding front-end tooling.

## Open questions

- Should particles default to on when solver streamlines are available, or stay
  opt-in?
- Is the seeded-set limitation acceptable long term, or is free-space particle
  seeding a real requirement? That answer determines whether the volumetric
  approach ever needs revisiting.
