import * as THREE from "./vendor/three.module.min.js";

const INVERT_ORBIT_STORAGE_KEY = "aerolab-invert-orbit";
const VIEW_MODE_STORAGE_KEY = "aerolab-view-mode";
const VIEWER_GROUND_Z = -0.58;
const VIEWER_CAMERA_NEAR = 0.1;
const VIEWER_CAMERA_FAR = 100;
const SOLVER_PARTICLE_TARGET_PER_LINE = 24;
const SOLVER_PARTICLE_MAX_COUNT = 5_200;
const SOLVER_PARTICLE_SPEED_FLOOR = 0.04;
const SOLVER_PARTICLE_ANIMATION_RATE = 0.9;
const SOLVER_PARTICLE_LINE_AVAILABILITY = new WeakMap();
const SPEED_COLOR_STOPS = [
  [38, 105, 208],
  [54, 205, 216],
  [246, 207, 75],
  [235, 77, 59],
];
const SPEED_COLOR_STOPS_LINEAR = SPEED_COLOR_STOPS.map((stop) => stop.map((channel) => {
  const value = channel / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}));

const state = {
  busy: false,
  viewMode: "basic",
  root: "",
  sampleModel: null,
  modelPath: null,
  report: null,
  mesh: null,
  cases: [],
  sensitivityParameters: {},
  activeCasePath: null,
  solver: null,
  caseReport: null,
  comparison: null,
  activeRunProgress: null,
  runProgressTimer: null,
  runProgressToken: 0,
  runLogTimer: null,
  runLogToken: 0,
  repair: null,
  aeroFeatures: null,
  aeroFeatureScanStatus: "idle",
  aeroFeatureScanTimer: null,
  aeroFeatureScanToken: 0,
  viewer: {
    yaw: -0.58,
    pitch: 0.46,
    zoom: 1,
    smokeTrails: [],
    flowEnvelope: null,
    meshBounds: null,
    orientedMesh: null,
    smoothVertexLight: null,
    flowLayer: null,
    meshSource: "raw",
    surfaceMode: "material",
    solverFlowMode: "both",
    solverParticles: null,
    exactMesh: null,
    exactMeshLoading: false,
    exactMeshRequest: 0,
    alignmentLoading: false,
    modelLayer: null,
    webgl: {
      renderer: null,
      scene: null,
      camera: null,
      group: null,
      mesh: null,
      wireframe: null,
      geometryKey: null,
      failed: false,
      width: 0,
      height: 0,
      pixelRatio: 0,
      contextLostHandler: null,
    },
    lastTime: 0,
    running: false,
    dragging: false,
    lastPointer: null,
  },
};

const els = {
  sidebar: document.querySelector(".sidebar"),
  basicModeButton: document.querySelector("#basicModeButton"),
  engineeringModeButton: document.querySelector("#engineeringModeButton"),
  viewModeDescription: document.querySelector("#viewModeDescription"),
  basicAirflowSummary: document.querySelector("#basicAirflowSummary"),
  rootPath: document.querySelector("#rootPath"),
  modelFile: document.querySelector("#modelFile"),
  fileLabel: document.querySelector("#fileLabel"),
  sampleButton: document.querySelector("#sampleButton"),
  prepareScanButton: document.querySelector("#prepareScanButton"),
  repairStatus: document.querySelector("#repairStatus"),
  unitScale: document.querySelector("#unitScale"),
  targetLength: document.querySelector("#targetLength"),
  targetWidth: document.querySelector("#targetWidth"),
  targetHeight: document.querySelector("#targetHeight"),
  sourceFlowDirection: document.querySelector("#sourceFlowDirection"),
  sourceUpDirection: document.querySelector("#sourceUpDirection"),
  rotateX: document.querySelector("#rotateX"),
  rotateY: document.querySelector("#rotateY"),
  rotateZ: document.querySelector("#rotateZ"),
  rotateXValue: document.querySelector("#rotateXValue"),
  rotateYValue: document.querySelector("#rotateYValue"),
  rotateZValue: document.querySelector("#rotateZValue"),
  autoAlignButton: document.querySelector("#autoAlignButton"),
  resetRotationButton: document.querySelector("#resetRotationButton"),
  invertOrbit: document.querySelector("#invertOrbit"),
  speedMph: document.querySelector("#speedMph"),
  airTemperatureC: document.querySelector("#airTemperatureC"),
  airPressurePa: document.querySelector("#airPressurePa"),
  airDensity: document.querySelector("#airDensity"),
  kinematicViscosity: document.querySelector("#kinematicViscosity"),
  turbulenceIntensity: document.querySelector("#turbulenceIntensity"),
  turbulenceLengthScale: document.querySelector("#turbulenceLengthScale"),
  flowAxis: document.querySelector("#flowAxis"),
  includeGround: document.querySelector("#includeGround"),
  movingGround: document.querySelector("#movingGround"),
  groundClearanceMm: document.querySelector("#groundClearanceMm"),
  yawDegrees: document.querySelector("#yawDegrees"),
  crosswindMps: document.querySelector("#crosswindMps"),
  roughnessHeightMm: document.querySelector("#roughnessHeightMm"),
  roughnessConstant: document.querySelector("#roughnessConstant"),
  backflowSafeOutlet: document.querySelector("#backflowSafeOutlet"),
  secondOrderTransient: document.querySelector("#secondOrderTransient"),
  fluidProfile: document.querySelector("#fluidProfile"),
  turbulenceModel: document.querySelector("#turbulenceModel"),
  closedTunnel: document.querySelector("#closedTunnel"),
  tunnelWidthM: document.querySelector("#tunnelWidthM"),
  tunnelHeightM: document.querySelector("#tunnelHeightM"),
  tunnelUpstreamM: document.querySelector("#tunnelUpstreamM"),
  tunnelDownstreamM: document.querySelector("#tunnelDownstreamM"),
  wheelSetupJson: document.querySelector("#wheelSetupJson"),
  porousZonesJson: document.querySelector("#porousZonesJson"),
  fanZonesJson: document.querySelector("#fanZonesJson"),
  heatZonesJson: document.querySelector("#heatZonesJson"),
  caseName: document.querySelector("#caseName"),
  referenceArea: document.querySelector("#referenceArea"),
  referenceLength: document.querySelector("#referenceLength"),
  cgX: document.querySelector("#cgX"),
  cgY: document.querySelector("#cgY"),
  cgZ: document.querySelector("#cgZ"),
  frontAxleStation: document.querySelector("#frontAxleStation"),
  rearAxleStation: document.querySelector("#rearAxleStation"),
  qualityPreset: document.querySelector("#qualityPreset"),
  simulationMode: document.querySelector("#simulationMode"),
  sensitivityParameter: document.querySelector("#sensitivityParameter"),
  sensitivityValues: document.querySelector("#sensitivityValues"),
  sensitivityBaselineIndex: document.querySelector("#sensitivityBaselineIndex"),
  createSensitivityButton: document.querySelector("#createSensitivityButton"),
  sensitivityStatus: document.querySelector("#sensitivityStatus"),
  smallestFeatureMm: document.querySelector("#smallestFeatureMm"),
  createCaseButton: document.querySelector("#createCaseButton"),
  createStudyButton: document.querySelector("#createStudyButton"),
  checkSolverButton: document.querySelector("#checkSolverButton"),
  meshCaseButton: document.querySelector("#meshCaseButton"),
  runCaseButton: document.querySelector("#runCaseButton"),
  runProgress: document.querySelector("#runProgress"),
  runProgressLabel: document.querySelector("#runProgressLabel"),
  runProgressPercent: document.querySelector("#runProgressPercent"),
  runProgressBar: document.querySelector("#runProgressBar"),
  runProgressDetail: document.querySelector("#runProgressDetail"),
  runLogDetails: document.querySelector("#runLogDetails"),
  runLogStatus: document.querySelector("#runLogStatus"),
  runLogOutput: document.querySelector("#runLogOutput"),
  solverStatus: document.querySelector("#solverStatus"),
  resultSummary: document.querySelector("#resultSummary"),
  modelName: document.querySelector("#modelName"),
  modelStatus: document.querySelector("#modelStatus"),
  candidateBadge: document.querySelector("#candidateBadge"),
  metrics: document.querySelector("#metrics"),
  readiness: document.querySelector("#readiness"),
  warnings: document.querySelector("#warnings"),
  caseStatus: document.querySelector("#caseStatus"),
  caseList: document.querySelector("#caseList"),
  comparisonBaseline: document.querySelector("#comparisonBaseline"),
  comparisonVariant: document.querySelector("#comparisonVariant"),
  compareCasesButton: document.querySelector("#compareCasesButton"),
  comparisonSummary: document.querySelector("#comparisonSummary"),
  canvas: document.querySelector("#flowCanvas"),
  modelCanvas: document.querySelector("#modelCanvas"),
  dragSummary: document.querySelector("#dragSummary"),
  solverFlowControl: document.querySelector("#solverFlowControl"),
  solverFlowStatus: document.querySelector("#solverFlowStatus"),
  solverLinesButton: document.querySelector("#solverLinesButton"),
  solverParticlesButton: document.querySelector("#solverParticlesButton"),
  solverBothButton: document.querySelector("#solverBothButton"),
  showEdges: document.querySelector("#showEdges"),
  surfaceModeButton: document.querySelector("#surfaceModeButton"),
  pressureModeButton: document.querySelector("#pressureModeButton"),
  temperatureModeButton: document.querySelector("#temperatureModeButton"),
  dragModeButton: document.querySelector("#dragModeButton"),
};

async function boot() {
  restoreViewMode();
  restoreViewerPreferences();
  const payload = await apiGet("/api/state");
  state.root = payload.root;
  state.sampleModel = payload.sampleModel;
  state.cases = payload.cases || [];
  state.sensitivityParameters = payload.sensitivityParameters || {};
  state.activeCasePath = state.cases[0]?.path || null;
  state.activeRunProgress = state.cases[0]?.progress || null;
  syncRunLogForActiveCase();
  els.rootPath.textContent = state.root;
  renderCases();
  await refreshSolverStatus();
  if (state.activeCasePath) await refreshCaseReport(state.activeCasePath);
  syncGroundControls();
  syncAdvancedFlowControls();
  syncRotationOutputs();
  initFlowVisualization();
  syncSolverFlowControls();
  startViewer();
}

els.basicModeButton.addEventListener("click", () => setViewMode("basic"));
els.engineeringModeButton.addEventListener("click", () => setViewMode("engineering"));

function restoreViewMode() {
  let mode = "basic";
  try {
    if (window.localStorage.getItem(VIEW_MODE_STORAGE_KEY) === "engineering") mode = "engineering";
  } catch (_error) {
    // Basic remains the first-run view when storage is unavailable.
  }
  setViewMode(mode, false);
}

function setViewMode(mode, persist = true) {
  const normalized = mode === "engineering" ? "engineering" : "basic";
  state.viewMode = normalized;
  document.body.dataset.viewMode = normalized;
  const basic = normalized === "basic";
  els.basicModeButton.classList.toggle("active", basic);
  els.engineeringModeButton.classList.toggle("active", !basic);
  els.basicModeButton.setAttribute("aria-pressed", String(basic));
  els.engineeringModeButton.setAttribute("aria-pressed", String(!basic));
  els.viewModeDescription.textContent = basic
    ? "Guided airflow workflow with safe defaults"
    : "Full setup, qualification, loads, and comparisons";
  els.createCaseButton.textContent = basic ? "Prepare Airflow Case" : "Create OpenFOAM Case";
  els.runCaseButton.textContent = basic ? "Calculate Airflow" : "Run Solver";
  updateActionAvailability();
  renderResultSummary();
  if (state.solver) renderSolverStatus();
  window.requestAnimationFrame(drawFlow);
  if (!persist) return;
  try {
    window.localStorage.setItem(VIEW_MODE_STORAGE_KEY, normalized);
  } catch (_error) {
    // The selected view still works for this session when storage is unavailable.
  }
}

els.modelFile.addEventListener("change", async () => {
  const file = els.modelFile.files[0];
  if (!file) return;
  setBusy(true, "Checking");
  try {
    const data = await file.arrayBuffer();
    const payload = await fetchJson(`/api/check?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: data,
    });
    loadReport(payload.modelPath, payload.report, payload.preview, data);
    els.fileLabel.textContent = file.name;
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
});

els.sampleButton.addEventListener("click", async () => {
  if (!state.sampleModel) return;
  setBusy(true, "Loading");
  try {
    const payload = await fetchJson("/api/check-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelPath: state.sampleModel }),
    });
    loadReport(payload.modelPath, payload.report, payload.preview);
    els.fileLabel.textContent = "sample_box.stl";
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
});

async function prepareCurrentModel() {
  els.repairStatus.textContent = "Sealing at body-line detail. This can take several minutes.";
  const smallestFeatureMm = optionalNumber(els.smallestFeatureMm.value);
  const payload = await fetchJson("/api/repair-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      modelPath: state.modelPath,
      resolution: 384,
      smallestFeatureM: smallestFeatureMm != null ? smallestFeatureMm / 1000 : null,
      unitScale: effectiveUnitScale(),
    }),
  });
  state.repair = payload.repair;
  if (!payload.accepted) {
    const reason = payload.repair?.rejectionReasons?.[0] || "The prepared copy did not meet the geometry acceptance limits.";
    els.repairStatus.textContent = `Repair needs review: ${reason}`;
    renderWarnings();
    throw new Error(`Could not prepare this model for CFD: ${reason}`);
  }
  loadReport(payload.modelPath, payload.report, payload.preview);
  state.repair = payload.repair;
  els.fileLabel.textContent = basename(payload.modelPath);
  els.repairStatus.textContent = `Prepared copy accepted: source p95 deviation ${fmt(state.repair.sourceSurfaceDeviationP95Percent)}%.`;
  renderReadiness();
  renderWarnings();
  return payload.modelPath;
}

els.prepareScanButton.addEventListener("click", async () => {
  if (!state.modelPath || !state.report || state.report.is_cfd_candidate) return;
  setBusy(true, "Preparing scan");
  try {
    await prepareCurrentModel();
  } catch (error) {
    showError(error);
    els.repairStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

els.createCaseButton.addEventListener("click", async () => {
  if (!state.modelPath) return;
  setBusy(true, "Creating");
  try {
    if (state.report && !state.report.is_cfd_candidate) {
      setBusy(true, "Preparing scan");
      await prepareCurrentModel();
      setBusy(true, "Creating case");
    }
    const payload = await fetchJson("/api/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentCasePayload()),
    });
    state.cases = payload.state.cases || [];
    state.comparison = null;
    state.activeCasePath = payload.casePath;
    syncRunLogForActiveCase();
    els.caseStatus.textContent = `Created ${payload.case.name}`;
    renderCases();
    await refreshCaseReport(payload.casePath);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
});

els.createStudyButton.addEventListener("click", async () => {
  if (!state.modelPath || geometryDimensionCheck().status !== "pass") return;
  setBusy(true, "Creating study");
  try {
    if (state.report && !state.report.is_cfd_candidate) {
      setBusy(true, "Preparing scan");
      await prepareCurrentModel();
      setBusy(true, "Creating study");
    }
    const payload = await fetchJson("/api/accuracy-study", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentCasePayload()),
    });
    state.cases = payload.state.cases || [];
    state.comparison = null;
    state.activeCasePath = payload.selectedCasePath;
    syncRunLogForActiveCase();
    state.caseReport = payload.report;
    els.caseStatus.textContent = "Created draft, standard, and fine accuracy cases";
    renderCases();
    await refreshCaseReport(payload.selectedCasePath);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
});

els.createSensitivityButton.addEventListener("click", async () => {
  if (!state.modelPath) return;
  setBusy(true, "Creating sensitivity study");
  try {
    if (state.report && !state.report.is_cfd_candidate) {
      setBusy(true, "Preparing scan");
      await prepareCurrentModel();
      setBusy(true, "Creating sensitivity study");
    }
    const values = sensitivityValuesPayload();
    const baselineIndex = optionalNumber(els.sensitivityBaselineIndex.value);
    if (baselineIndex != null && (!Number.isInteger(baselineIndex) || baselineIndex < 0 || baselineIndex >= values.length)) {
      throw new Error(`Baseline index must be a whole number from 0 to ${values.length - 1}.`);
    }
    const payload = await fetchJson("/api/sensitivity-study", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...currentCasePayload(),
        sensitivityParameter: els.sensitivityParameter.value,
        sensitivityValues: values,
        sensitivityBaselineIndex: baselineIndex,
      }),
    });
    state.cases = payload.state.cases || [];
    state.comparison = null;
    state.activeCasePath = payload.selectedCasePath;
    syncRunLogForActiveCase();
    state.caseReport = payload.report;
    const study = payload.study || {};
    els.sensitivityStatus.textContent = `Created ${study.casePaths?.length || values.length} ${study.parameterLabel || study.parameter} cases; baseline index ${study.baselineIndex}.`;
    els.caseStatus.textContent = `Created sensitivity family ${study.studyId || ""}`.trim();
    renderCases();
    await refreshCaseReport(payload.selectedCasePath);
  } catch (error) {
    showError(error);
    els.sensitivityStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

function syncBasicSourceUpDirection() {
  if (state.viewMode !== "basic") return;
  const flowAxis = signedAxisName(els.sourceFlowDirection.value);
  if (flowAxis === signedAxisName(els.sourceUpDirection.value)) {
    els.sourceUpDirection.value = flowAxis === "z" ? "+y" : "+z";
  }
}

function sensitivityValuesPayload() {
  const tokens = String(els.sensitivityValues.value || "")
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean);
  if (tokens.length < 2 || tokens.length > 12) {
    throw new Error("Enter between 2 and 12 sensitivity values.");
  }
  const values = tokens.map(Number);
  if (!values.every(Number.isFinite)) {
    throw new Error("Sensitivity values must all be finite numbers.");
  }
  if (new Set(values).size !== values.length) {
    throw new Error("Sensitivity values must be unique.");
  }
  return values;
}

function wheelSetupPayload() {
  const text = String(els.wheelSetupJson.value || "").trim();
  if (!text) return null;
  const payload = JSON.parse(text);
  if (!Array.isArray(payload)) throw new Error("Wheel setup JSON must be an array.");
  return payload;
}

function volumeZonesPayload(input, label) {
  const text = String(input.value || "").trim();
  if (!text) return null;
  const payload = JSON.parse(text);
  if (!Array.isArray(payload)) throw new Error(`${label} JSON must be an array.`);
  return payload;
}

function currentCasePayload() {
  syncBasicSourceUpDirection();
  const smallestFeatureMm = optionalNumber(els.smallestFeatureMm.value);
  const includeGround = els.includeGround.checked;
  const engineering = state.viewMode === "engineering";
  const closedTunnel = engineering && els.closedTunnel.checked
    ? {
        width_m: optionalNumber(els.tunnelWidthM.value),
        height_m: optionalNumber(els.tunnelHeightM.value),
        upstream_m: optionalNumber(els.tunnelUpstreamM.value),
        downstream_m: optionalNumber(els.tunnelDownstreamM.value),
      }
    : null;
  return {
    modelPath: state.modelPath,
    name: els.caseName.value,
    speedMph: Number(els.speedMph.value || 70),
    airTemperatureC: optionalFiniteNumber(els.airTemperatureC.value),
    airPressurePa: optionalNumber(els.airPressurePa.value),
    airDensityKgM3: optionalNumber(els.airDensity.value),
    kinematicViscosityM2S: optionalNumber(els.kinematicViscosity.value),
    turbulenceIntensityPercent: optionalNumber(els.turbulenceIntensity.value),
    turbulenceLengthScaleM: optionalNumber(els.turbulenceLengthScale.value),
    yawDegrees: engineering ? optionalFiniteNumber(els.yawDegrees.value) : null,
    crosswindMps: engineering ? optionalFiniteNumber(els.crosswindMps.value) : null,
    roughnessHeightM: engineering
      ? Math.max(0, Number(els.roughnessHeightMm.value || 0)) / 1000
      : 0,
    roughnessConstant: engineering ? Number(els.roughnessConstant.value || 0.5) : 0.5,
    closedTunnel,
    backflowSafeOutlet: engineering && els.backflowSafeOutlet.checked,
    wheelSetup: engineering ? wheelSetupPayload() : null,
    secondOrderTransient: engineering && els.secondOrderTransient.checked,
    fluidProfile: engineering ? els.fluidProfile.value : "incompressible",
    turbulenceModel: engineering ? els.turbulenceModel.value : "kOmegaSST",
    porousZones: engineering
      ? volumeZonesPayload(els.porousZonesJson, "Porous zones")
      : null,
    fanZones: engineering
      ? volumeZonesPayload(els.fanZonesJson, "Fan zones")
      : null,
    heatZones: engineering
      ? volumeZonesPayload(els.heatZonesJson, "Heat-load zones")
      : null,
    flowAxis: els.flowAxis.value,
    includeGround,
    movingGround: includeGround && els.movingGround.checked,
    groundClearanceM: includeGround
      ? Math.max(0, Number(els.groundClearanceMm.value || 0)) / 1000
      : 0,
    unitScale: effectiveUnitScale(),
    unitLabel: scaleLabel(),
    measuredLengthM: optionalNumber(els.targetLength.value),
    measuredWidthM: optionalNumber(els.targetWidth.value),
    measuredHeightM: optionalNumber(els.targetHeight.value),
    referenceAreaM2: optionalNumber(els.referenceArea.value),
    referenceLengthM: optionalNumber(els.referenceLength.value),
    centerOfGravityM: {
      x: optionalFiniteNumber(els.cgX.value),
      y: optionalFiniteNumber(els.cgY.value),
      z: optionalFiniteNumber(els.cgZ.value),
    },
    frontAxleStationM: optionalFiniteNumber(els.frontAxleStation.value),
    rearAxleStationM: optionalFiniteNumber(els.rearAxleStation.value),
    quality: els.qualityPreset.value,
    simulationMode: els.simulationMode.value,
    smallestAeroFeatureM: smallestFeatureMm != null
      ? smallestFeatureMm / 1000
      : null,
    sourceFlowDirection: els.sourceFlowDirection.value,
    sourceUpDirection: els.sourceUpDirection.value,
    modelRotationDegrees: modelRotationDegrees(),
  };
}

els.checkSolverButton.addEventListener("click", async () => {
  setBusy(true, "Checking solver");
  try {
    await refreshSolverStatus();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
});

els.meshCaseButton.addEventListener("click", () => runActiveCase("mesh"));
els.runCaseButton.addEventListener("click", () => runActiveCase("full"));
els.runLogDetails.addEventListener("toggle", () => {
  if (els.runLogDetails.open) startRunLogPolling();
  else stopRunLogPolling();
});

function syncRunLogForActiveCase() {
  stopRunLogPolling();
  if (!state.activeCasePath) {
    els.runLogStatus.textContent = "Select a case to view its OpenFOAM output";
    els.runLogOutput.textContent = "No active case selected.";
    return;
  }
  els.runLogStatus.textContent = "Open to view the latest OpenFOAM output";
  els.runLogOutput.textContent = "Open this panel to load the run log.";
  if (els.runLogDetails.open) startRunLogPolling();
}

function startRunLogPolling() {
  stopRunLogPolling();
  const casePath = state.activeCasePath;
  if (!els.runLogDetails.open || !casePath) {
    syncRunLogForActiveCase();
    return;
  }
  const token = state.runLogToken + 1;
  state.runLogToken = token;
  els.runLogStatus.textContent = "Loading latest OpenFOAM output...";

  const poll = async () => {
    if (
      state.runLogToken !== token
      || !els.runLogDetails.open
      || state.activeCasePath !== casePath
    ) return;
    try {
      const payload = await apiGet(`/api/case-log?casePath=${encodeURIComponent(casePath)}`);
      if (
        state.runLogToken !== token
        || !els.runLogDetails.open
        || state.activeCasePath !== casePath
      ) return;
      renderRunLogPayload(payload);
    } catch (error) {
      if (
        state.runLogToken !== token
        || !els.runLogDetails.open
        || state.activeCasePath !== casePath
      ) return;
      els.runLogStatus.textContent = "Run log unavailable";
      els.runLogOutput.textContent = `Could not read the run log: ${error.message}`;
    }
    if (
      state.runLogToken === token
      && els.runLogDetails.open
      && state.activeCasePath === casePath
    ) {
      state.runLogTimer = window.setTimeout(poll, 1200);
    }
  };
  void poll();
}

function renderRunLogPayload(payload) {
  const stickToBottom = (
    els.runLogOutput.scrollHeight
    - els.runLogOutput.scrollTop
    - els.runLogOutput.clientHeight
  ) < 40;
  if (!payload.exists) {
    els.runLogStatus.textContent = "Waiting for OpenFOAM output";
    els.runLogOutput.textContent = "No run log yet. Start mesh validation or a solver run.";
  } else {
    const totalBytes = formatInt(payload.sizeBytes || 0);
    const shownBytes = formatInt(payload.shownBytes || 0);
    els.runLogStatus.textContent = payload.truncated
      ? `Showing latest ${shownBytes} of ${totalBytes} bytes`
      : `${totalBytes} bytes · live`;
    els.runLogOutput.textContent = payload.text || "The run log is currently empty.";
  }
  if (stickToBottom) {
    window.requestAnimationFrame(() => {
      els.runLogOutput.scrollTop = els.runLogOutput.scrollHeight;
    });
  }
}

function stopRunLogPolling() {
  state.runLogToken += 1;
  if (state.runLogTimer != null) {
    window.clearTimeout(state.runLogTimer);
    state.runLogTimer = null;
  }
}

async function runActiveCase(mode) {
  if (!state.activeCasePath) return;
  const runningCasePath = state.activeCasePath;
  const meshOnly = mode === "mesh";
  applyRunProgress(runningCasePath, {
    state: "running",
    tone: "running",
    phase: meshOnly ? "Starting mesh validation" : "Starting solver",
    percent: 1,
    label: `${meshOnly ? "Starting mesh validation" : "Starting solver"} - 1%`,
    detail: meshOnly
      ? "Opening OpenFOAM to build and audit this vehicle's mesh."
      : "Opening OpenFOAM and staging the selected case.",
    isRunning: true,
    isComplete: false,
    isMeshComplete: false,
    runMode: mode,
  });
  setBusy(true, meshOnly ? "Validating mesh" : "Running solver");
  const timeoutSeconds = meshOnly ? 14400 : 21600;
  try {
    await fetchJson("/api/run-case", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        casePath: runningCasePath,
        backend: "auto",
        timeoutSeconds,
        mode,
        reuseMesh: true,
      }),
    });
    const progress = await startRunProgressPolling(
      runningCasePath,
      (timeoutSeconds + 1800) * 1000,
    );
    if (progress?.state === "failed") throw new Error(progress.detail);
  } catch (error) {
    showError(error);
  } finally {
    stopRunProgressPolling();
    try {
      const latestState = await apiGet("/api/state");
      state.cases = latestState.cases || state.cases;
      if (state.activeCasePath === runningCasePath) {
        await refreshCaseReport(runningCasePath);
      }
    } catch (refreshError) {
      showError(refreshError);
    }
    renderCases();
    setBusy(false);
  }
}

els.caseList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-case-path]");
  if (!button) return;
  const requestedCasePath = button.dataset.casePath;
  state.activeCasePath = requestedCasePath;
  state.caseReport = null;
  resetSolverParticles();
  state.viewer.flowLayer = null;
  syncSolverFlowControls();
  state.activeRunProgress = state.cases.find((item) => item.path === requestedCasePath)?.progress || null;
  syncRunLogForActiveCase();
  state.report = null;
  state.mesh = null;
  els.modelName.textContent = "Loading case";
  els.modelStatus.textContent = "Reading full CFD geometry";
  els.sidebar.inert = true;
  els.caseList.inert = true;
  els.sidebar.setAttribute("aria-busy", "true");
  setBusy(true, "Loading case");
  renderCases();
  renderMetrics();
  renderReadiness();
  renderRunProgress();
  let loaded = false;
  try {
    await refreshCaseReport(requestedCasePath);
    loaded = true;
  } catch (error) {
    showError(error);
  } finally {
    els.sidebar.inert = false;
    els.caseList.inert = false;
    els.sidebar.removeAttribute("aria-busy");
    setBusy(false);
    if (loaded) els.caseStatus.textContent = "";
    renderCases();
  }
});

for (const select of [els.comparisonBaseline, els.comparisonVariant]) {
  select.addEventListener("change", () => {
    state.comparison = null;
    renderComparisonControls();
  });
}

els.compareCasesButton.addEventListener("click", async () => {
  const baselineCasePath = els.comparisonBaseline.value;
  const variantCasePath = els.comparisonVariant.value;
  if (!baselineCasePath || !variantCasePath || baselineCasePath === variantCasePath) return;
  els.compareCasesButton.disabled = true;
  els.comparisonSummary.innerHTML = `<div class="comparison-status">Checking locked setup and qualified loads...</div>`;
  try {
    const payload = await fetchJson("/api/compare-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baselineCasePath, variantCasePath }),
    });
    state.comparison = payload.comparison;
    renderComparisonControls();
  } catch (error) {
    state.comparison = {
      decisionSafe: false,
      statusLabel: error.message,
      coefficientDeltas: {},
      balanceDeltas: {},
      setupDifferences: [],
      interpretation: "The comparison request did not complete.",
    };
  } finally {
    renderComparisonControls();
  }
});

for (const input of [
  els.speedMph,
  els.airTemperatureC,
  els.airPressurePa,
  els.airDensity,
  els.kinematicViscosity,
  els.turbulenceIntensity,
  els.turbulenceLengthScale,
  els.flowAxis,
  els.includeGround,
  els.movingGround,
  els.groundClearanceMm,
  els.yawDegrees,
  els.crosswindMps,
  els.roughnessHeightMm,
  els.roughnessConstant,
  els.backflowSafeOutlet,
  els.secondOrderTransient,
  els.fluidProfile,
  els.turbulenceModel,
  els.porousZonesJson,
  els.fanZonesJson,
  els.heatZonesJson,
  els.closedTunnel,
  els.tunnelWidthM,
  els.tunnelHeightM,
  els.tunnelUpstreamM,
  els.tunnelDownstreamM,
  els.wheelSetupJson,
  els.unitScale,
  els.targetLength,
  els.targetWidth,
  els.targetHeight,
  els.referenceArea,
  els.referenceLength,
  els.cgX,
  els.cgY,
  els.cgZ,
  els.frontAxleStation,
  els.rearAxleStation,
  els.qualityPreset,
  els.simulationMode,
  els.sensitivityParameter,
  els.sensitivityValues,
  els.sensitivityBaselineIndex,
  els.smallestFeatureMm,
]) {
  input.addEventListener("input", () => {
    updateCaseName();
    renderMetrics();
    renderReadiness();
    drawFlow();
    if (input === els.unitScale) scheduleAeroFeatureScan();
  });
}

els.includeGround.addEventListener("change", syncGroundControls);
els.closedTunnel.addEventListener("change", () => {
  if (els.closedTunnel.checked) els.includeGround.checked = true;
  syncGroundControls();
  syncAdvancedFlowControls();
});
els.simulationMode.addEventListener("change", () => {
  syncAdvancedFlowControls();
  renderReadiness();
});
els.fluidProfile.addEventListener("change", () => {
  syncAdvancedFlowControls();
  renderReadiness();
});
els.turbulenceModel.addEventListener("change", () => {
  syncAdvancedFlowControls();
  renderReadiness();
});
els.sensitivityParameter.addEventListener("change", () => {
  const defaults = {
    speed_mph: "50, 70, 90",
    yaw_degrees: "-5, 0, 5",
    crosswind_mps: "-3, 0, 3",
    roughness_height_m: "0, 0.0005, 0.001",
    ground_clearance_m: "0.05, 0.075, 0.1",
    turbulence_intensity_percent: "0.5, 1, 2",
  };
  els.sensitivityValues.value = defaults[els.sensitivityParameter.value] || "";
  els.sensitivityBaselineIndex.value = "";
  els.sensitivityStatus.textContent = "";
  renderReadiness();
});

function syncGroundControls() {
  const enabled = els.includeGround.checked;
  els.movingGround.disabled = !enabled;
  els.groundClearanceMm.disabled = !enabled;
}

function syncAdvancedFlowControls() {
  const tunnelEnabled = els.closedTunnel.checked;
  for (const input of [
    els.tunnelWidthM,
    els.tunnelHeightM,
    els.tunnelUpstreamM,
    els.tunnelDownstreamM,
  ]) {
    input.disabled = !tunnelEnabled;
  }
  const hybridModel = els.turbulenceModel.value !== "kOmegaSST";
  if (hybridModel) els.simulationMode.value = "transient";
  const transient = els.simulationMode.value === "transient";
  els.secondOrderTransient.disabled = !transient;
  if (!transient) els.secondOrderTransient.checked = false;
  els.roughnessHeightMm.disabled = hybridModel;
  els.roughnessConstant.disabled = hybridModel;
  if (hybridModel) {
    els.roughnessHeightMm.value = "0";
    els.roughnessConstant.value = "0.5";
  }
  els.speedMph.max = els.fluidProfile.value === "compressible_thermal" ? "1000" : "230";
}

for (const input of [els.sourceFlowDirection, els.sourceUpDirection]) {
  input.addEventListener("input", async () => {
    if (input === els.sourceFlowDirection) syncBasicSourceUpDirection();
    await applyRotationChange();
  });
}

for (const slider of [els.rotateX, els.rotateY, els.rotateZ]) {
  slider.addEventListener("input", async () => {
    syncRotationOutputs();
    await applyRotationChange();
  });
}

for (const [numberInput, slider] of [
  [els.rotateXValue, els.rotateX],
  [els.rotateYValue, els.rotateY],
  [els.rotateZValue, els.rotateZ],
]) {
  numberInput.addEventListener("input", async () => {
    if (!Number.isFinite(numberInput.valueAsNumber)) return;
    const angle = clamp(numberInput.valueAsNumber, -180, 180);
    numberInput.value = fmtAngle(angle);
    slider.value = String(angle);
    await applyRotationChange();
  });
}

async function applyRotationChange() {
  await beginSourceAlignment();
  invalidateModelOrientation();
  updateCaseName();
  renderMetrics();
  renderReadiness();
  initFlowVisualization();
  drawFlow();
  scheduleAeroFeatureScan();
}

els.autoAlignButton.addEventListener("click", async () => {
  const suggestion = state.report?.alignment_suggestion;
  if (!suggestion?.recommended || !suggestion.rotation_degrees) return;
  setModelRotation(suggestion.rotation_degrees);
  els.sourceFlowDirection.value = "+x";
  els.sourceUpDirection.value = "+z";
  await beginSourceAlignment();
  invalidateModelOrientation();
  updateCaseName();
  renderMetrics();
  renderReadiness();
  initFlowVisualization();
  drawFlow();
  scheduleAeroFeatureScan();
  const dimensions = suggestion.aligned_dimensions || [];
  els.modelStatus.textContent = dimensions.length === 3
    ? `Auto-aligned: ${dimensions.map((value) => fmt(value)).join(" × ")}`
    : "Auto-aligned to principal axes";
});

els.resetRotationButton.addEventListener("click", async () => {
  setModelRotation({ x: 0, y: 0, z: 0 });
  await beginSourceAlignment();
  invalidateModelOrientation();
  renderMetrics();
  renderReadiness();
  initFlowVisualization();
  drawFlow();
  scheduleAeroFeatureScan();
});

els.invertOrbit.addEventListener("change", saveViewerPreferences);

els.showEdges.addEventListener("change", drawFlow);
els.surfaceModeButton.addEventListener("click", () => setSurfaceMode("material"));
els.pressureModeButton.addEventListener("click", () => setSurfaceMode("cp"));
els.temperatureModeButton.addEventListener("click", () => setSurfaceMode("temperature"));
els.dragModeButton.addEventListener("click", () => setSurfaceMode("drag"));
els.solverLinesButton.addEventListener("click", () => setSolverFlowMode("lines"));
els.solverParticlesButton.addEventListener("click", () => setSolverFlowMode("particles"));
els.solverBothButton.addEventListener("click", () => setSolverFlowMode("both"));

function hasSolverStreamlines() {
  const flow = state.caseReport?.solverStreamlines;
  return Boolean(!flow?.error && flow?.lines?.length);
}

function hasRenderableSolverParticleLines(flow = state.caseReport?.solverStreamlines) {
  if (!flow || typeof flow !== "object") return false;
  if (SOLVER_PARTICLE_LINE_AVAILABILITY.has(flow)) return SOLVER_PARTICLE_LINE_AVAILABILITY.get(flow);
  let renderable = false;
  for (const path of flow.lines || []) {
    let previousX = 0;
    let previousY = 0;
    let previousZ = 0;
    let hasPrevious = false;
    for (const sample of path || []) {
      const x = Number(sample?.[0]);
      const y = Number(sample?.[1]);
      const z = Number(sample?.[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      if (hasPrevious && Math.hypot(x - previousX, y - previousY, z - previousZ) > 1e-8) {
        renderable = true;
        break;
      }
      previousX = x;
      previousY = y;
      previousZ = z;
      hasPrevious = true;
    }
    if (renderable) break;
  }
  SOLVER_PARTICLE_LINE_AVAILABILITY.set(flow, renderable);
  return renderable;
}

function solverParticleUnavailableReason() {
  if (state.viewer.webgl.failed) return "WebGL could not start";
  if (!hasRenderableSolverParticleLines()) return "Solved tracks contain no usable particle segments";
  const hasSurface = state.mesh?.triangles?.length || state.caseReport?.geometryPreview?.triangles?.length;
  if (!hasSurface) return "The active report has no renderable model surface";
  return null;
}

function shouldDrawSolverLines() {
  if (!hasSolverStreamlines()) return false;
  return Boolean(solverParticleUnavailableReason()) || state.viewer.solverFlowMode !== "particles";
}

function shouldDrawSolverParticles() {
  return hasSolverStreamlines()
    && !solverParticleUnavailableReason()
    && state.viewer.solverFlowMode !== "lines";
}

function setSolverFlowMode(mode) {
  const requested = ["lines", "particles", "both"].includes(mode) ? mode : "both";
  const particlesUnavailable = hasSolverStreamlines() && solverParticleUnavailableReason();
  state.viewer.solverFlowMode = particlesUnavailable && requested !== "lines" ? "lines" : requested;
  syncSolverFlowControls();
  if (state.viewer.solverParticles?.points) {
    state.viewer.solverParticles.points.visible = shouldDrawSolverParticles();
  }
  drawFlow();
}

function syncSolverFlowControls() {
  const flow = state.caseReport?.solverStreamlines;
  const available = hasSolverStreamlines();
  const particleReason = available ? solverParticleUnavailableReason() : null;
  const particleAvailable = available && !particleReason;
  if (available && !particleAvailable && state.viewer.solverFlowMode !== "lines") {
    state.viewer.solverFlowMode = "lines";
  }

  els.solverLinesButton.disabled = !available;
  els.solverParticlesButton.disabled = !particleAvailable;
  els.solverBothButton.disabled = !particleAvailable;
  for (const [button, mode] of [
    [els.solverLinesButton, "lines"],
    [els.solverParticlesButton, "particles"],
    [els.solverBothButton, "both"],
  ]) {
    const active = available && state.viewer.solverFlowMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }

  let status = "Run the solver to generate solved flow";
  let detail = status;
  if (flow?.error) {
    status = String(flow.error);
    detail = `Solved flow unavailable: ${status}`;
  } else if (available) {
    const fieldLabel = flow.timeAveraged ? "Mean-flow speed" : "Final-field speed";
    status = particleReason ? `${fieldLabel} · ${particleReason}` : fieldLabel;
    detail = particleReason
      ? `${fieldLabel}; particles unavailable: ${particleReason}`
      : `${fieldLabel}; particle time is visual, not solver time`;
  }
  els.solverFlowStatus.textContent = status;
  els.solverFlowControl.title = detail;
  els.solverFlowControl.classList.toggle("unavailable", !available);
}

function hasSurfacePressure() {
  const pressure = state.caseReport?.surfacePressure;
  return Boolean(pressure?.hasPressure && pressure.points?.length && pressure.triangles?.length);
}

function hasSurfaceTemperature() {
  const surface = state.caseReport?.surfacePressure;
  return Boolean(
    hasSurfacePressure()
    && surface?.hasTemperature
    && surface.temperatureKValues?.length === surface.points?.length,
  );
}

function hasSurfaceDrag() {
  const pressure = state.caseReport?.surfacePressure;
  return Boolean(hasSurfacePressure() && pressure?.hasPressureDrag && pressure.points?.some((point) => point.length >= 5));
}

function hasSurfaceWallShear() {
  const pressure = state.caseReport?.surfacePressure;
  return Boolean(hasSurfaceDrag() && pressure?.hasWallShear && pressure.points?.some((point) => point.length >= 7));
}

function setSurfaceMode(mode) {
  const webglAvailable = !state.viewer.webgl.failed;
  if (webglAvailable && mode === "cp" && hasSurfacePressure()) state.viewer.surfaceMode = "cp";
  else if (webglAvailable && mode === "temperature" && hasSurfaceTemperature()) state.viewer.surfaceMode = "temperature";
  else if (webglAvailable && mode === "drag" && hasSurfaceDrag()) state.viewer.surfaceMode = "drag";
  else state.viewer.surfaceMode = "material";
  syncSurfaceModeControls();
  invalidateThreeGeometry();
  renderReadiness();
  drawFlow();
}

function syncSurfaceModeControls() {
  const hasPressure = hasSurfacePressure();
  const hasTemperature = hasSurfaceTemperature();
  const hasDrag = hasSurfaceDrag();
  const webglAvailable = !state.viewer.webgl.failed;
  if (!webglAvailable) state.viewer.surfaceMode = "material";
  if (!hasPressure && state.viewer.surfaceMode === "cp") state.viewer.surfaceMode = "material";
  if (!hasTemperature && state.viewer.surfaceMode === "temperature") state.viewer.surfaceMode = "material";
  if (!hasDrag && state.viewer.surfaceMode === "drag") state.viewer.surfaceMode = "material";
  const showPressure = state.viewer.surfaceMode === "cp";
  const showTemperature = state.viewer.surfaceMode === "temperature";
  const showDrag = state.viewer.surfaceMode === "drag";
  const showMaterial = !showPressure && !showTemperature && !showDrag;
  els.surfaceModeButton.classList.toggle("active", showMaterial);
  els.pressureModeButton.classList.toggle("active", showPressure);
  els.temperatureModeButton.classList.toggle("active", showTemperature);
  els.dragModeButton.classList.toggle("active", showDrag);
  els.surfaceModeButton.setAttribute("aria-pressed", String(showMaterial));
  els.pressureModeButton.setAttribute("aria-pressed", String(showPressure));
  els.temperatureModeButton.setAttribute("aria-pressed", String(showTemperature));
  els.dragModeButton.setAttribute("aria-pressed", String(showDrag));
  els.pressureModeButton.disabled = !webglAvailable || !hasPressure;
  els.temperatureModeButton.disabled = !webglAvailable || !hasTemperature;
  els.dragModeButton.disabled = !webglAvailable || !hasDrag;
  const pressureSetup = state.caseReport?.surfacePressureSetup;
  const pressureLabel = !webglAvailable
    ? "Solved surface coloring is unavailable because WebGL could not start"
    : hasPressure
      ? "Show solved pressure coefficient Cp"
      : pressureSetup?.configured
        ? "Pressure coefficient Cp is pending a completed solver run"
        : "Pressure coefficient Cp will be configured when this case is run";
  els.pressureModeButton.setAttribute("aria-label", pressureLabel);
  els.pressureModeButton.title = pressureLabel;
  const temperatureLabel = !webglAvailable
    ? "Solved surface coloring is unavailable because WebGL could not start"
    : hasTemperature
      ? "Show solved adjacent-air temperature on the body surface"
      : pressureSetup?.temperatureConfigured
        ? "Adjacent-air temperature is pending a completed solver run"
        : "Air temperature requires a compressible + thermal case";
  els.temperatureModeButton.setAttribute("aria-label", temperatureLabel);
  els.temperatureModeButton.title = temperatureLabel;
  const dragLabel = !webglAvailable
    ? "Solved surface coloring is unavailable because WebGL could not start"
    : hasDrag
      ? hasSurfaceWallShear()
        ? "Show solved local total drag from pressure and wall shear"
        : "Show solved local pressure-drag contribution"
      : pressureSetup?.configured
        ? "Pressure-drag areas are pending a completed solver run"
        : "Pressure-drag areas will be configured when this case is run";
  els.dragModeButton.setAttribute("aria-label", dragLabel);
  els.dragModeButton.title = dragLabel;
  renderDragSummary();
}

function renderDragSummary() {
  const surface = state.caseReport?.surfacePressure;
  const regions = Array.isArray(surface?.dragRegions) ? surface.dragRegions : [];
  const showTemperature = state.viewer.surfaceMode === "temperature" && hasSurfaceTemperature();
  const showDrag = state.viewer.surfaceMode === "drag" && hasSurfaceDrag();
  els.dragSummary.hidden = !showTemperature && !showDrag;
  if (!showTemperature && !showDrag) {
    els.dragSummary.innerHTML = "";
    return;
  }
  if (showTemperature) {
    const range = surface.temperatureCRange || [];
    const summary = state.caseReport?.temperatureResults;
    els.dragSummary.innerHTML = `
      <div class="drag-summary-heading">
        <strong>Adjacent-air temperature</strong>
        <span>${surface.temperatureTimeAveraged ? "Mean field" : "Final field"}</span>
      </div>
      <p>${fmt(range[0])} to ${fmt(range[1])} °C on the solved surface</p>
      ${summary?.maximumRiseK != null ? `<small>Internal-air maximum rise: ${fmt(summary.maximumRiseK)} K above inlet</small>` : ""}
      <small>${escapeHtml(surface.temperatureDefinition || "This is air temperature next to the surface, not solid-component temperature.")}</small>
    `;
    return;
  }

  const totalCd = Number(surface.totalDragCoefficient ?? surface.pressureDragCoefficient);
  const pressureCd = Number(surface.pressureDragCoefficient);
  const skinCd = surface.hasWallShear
    ? Number(surface.skinFrictionDragCoefficient)
    : Number.NaN;
  const hotspot = regions.find((region) => region.id === surface.dragHotspotRegion)
    || regions.reduce(
      (largest, region) => Number(region.positiveDragSharePercent || 0) > Number(largest?.positiveDragSharePercent || -1)
        ? region
        : largest,
      null,
    );
  const componentText = Number.isFinite(skinCd)
    ? `Pressure Cd ${fmt(pressureCd)} | skin Cd ${fmt(skinCd)}`
    : `Pressure Cd ${fmt(pressureCd)}`;
  const regionMarkup = regions.length
    ? regions.map((region) => {
      const share = clamp(Number(region.positiveDragSharePercent || 0), 0, 100);
      const shortLabel = region.id === "middle" ? "Middle" : region.id === "rear" ? "Rear" : "Front";
      const isHotspot = hotspot?.id === region.id;
      return `
        <div class="drag-region${isHotspot ? " hotspot" : ""}" style="--drag-share:${share}" aria-label="${escapeAttr(region.label)} ${fmt(share)} percent of positive drag">
          <i aria-hidden="true"></i>
          <span>${escapeHtml(shortLabel)}</span>
          <strong>${fmt(share)}%</strong>
        </div>
      `;
    }).join("")
    : `<p class="drag-summary-empty">Solved face colors show local drag contribution.</p>`;

  els.dragSummary.innerHTML = `
    <div class="drag-summary-heading">
      <strong>Drag areas</strong>
      <span>${Number.isFinite(totalCd) ? `Cd ${fmt(totalCd)}` : "Solved"}</span>
    </div>
    <p>${componentText}</p>
    ${hotspot ? `<small>Largest positive zone: ${escapeHtml(hotspot.label)}</small>` : ""}
    <div class="drag-regions" aria-label="Positive drag contribution by body section">
      ${regionMarkup}
    </div>
  `;
}

els.canvas.addEventListener("pointerdown", (event) => {
  state.viewer.dragging = true;
  state.viewer.lastPointer = { x: event.clientX, y: event.clientY };
  els.canvas.setPointerCapture(event.pointerId);
});

els.canvas.addEventListener("pointermove", (event) => {
  if (!state.viewer.dragging || !state.viewer.lastPointer) return;
  const dx = event.clientX - state.viewer.lastPointer.x;
  const dy = event.clientY - state.viewer.lastPointer.y;
  const orbitDirection = els.invertOrbit.checked ? -1 : 1;
  state.viewer.yaw += dx * 0.006 * orbitDirection;
  state.viewer.pitch = clamp(state.viewer.pitch + dy * 0.004 * orbitDirection, -0.15, 1.05);
  state.viewer.lastPointer = { x: event.clientX, y: event.clientY };
  drawFlow();
});

els.canvas.addEventListener("pointerup", () => {
  state.viewer.dragging = false;
  state.viewer.lastPointer = null;
});

els.canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    state.viewer.zoom = clamp(state.viewer.zoom + (event.deltaY > 0 ? -0.08 : 0.08), 0.72, 1.55);
    drawFlow();
  },
  { passive: false },
);

function restoreViewerPreferences() {
  try {
    els.invertOrbit.checked = window.localStorage.getItem(INVERT_ORBIT_STORAGE_KEY) === "true";
  } catch (_error) {
    els.invertOrbit.checked = false;
  }
}

function saveViewerPreferences() {
  try {
    window.localStorage.setItem(INVERT_ORBIT_STORAGE_KEY, String(els.invertOrbit.checked));
  } catch (_error) {
    // The control still works for this session when storage is unavailable.
  }
}

function loadReport(modelPath, report, preview = null, exactData = null) {
  state.modelPath = modelPath;
  state.report = report;
  state.mesh = preview;
  state.repair = null;
  state.aeroFeatures = null;
  state.aeroFeatureScanStatus = "idle";
  state.caseReport = null;
  resetSolverParticles();
  state.viewer.flowLayer = null;
  syncSolverFlowControls();
  state.activeCasePath = null;
  state.activeRunProgress = null;
  syncRunLogForActiveCase();
  state.viewer.meshSource = "raw";
  state.viewer.surfaceMode = "material";
  syncSurfaceModeControls();
  els.repairStatus.textContent = "";
  state.viewer.modelLayer = null;
  state.viewer.flowEnvelope = null;
  state.viewer.meshBounds = null;
  state.viewer.orientedMesh = null;
  state.viewer.smoothVertexLight = null;
  state.viewer.flowLayer = null;
  updateActionAvailability();
  els.modelName.textContent = basename(modelPath);
  els.modelStatus.textContent = `${report.format.toUpperCase()} STL`;
  els.candidateBadge.textContent = report.is_cfd_candidate ? "Mesh Ready" : "Cleanup";
  els.candidateBadge.className = report.is_cfd_candidate ? "badge" : "badge warn";
  updateCaseName();
  renderMetrics();
  renderReadiness();
  renderWarnings();
  renderCases();
  renderResultSummary();
  renderRunProgress();
  initFlowVisualization();
  drawFlow();
  void loadExactStl(modelPath, exactData);
  scheduleAeroFeatureScan(0);
}

function scheduleAeroFeatureScan(delay = 250) {
  if (state.aeroFeatureScanTimer) window.clearTimeout(state.aeroFeatureScanTimer);
  const token = state.aeroFeatureScanToken + 1;
  state.aeroFeatureScanToken = token;
  if (!state.modelPath) return;
  state.aeroFeatures = null;
  state.aeroFeatureScanStatus = "running";
  renderMetrics();
  state.aeroFeatureScanTimer = window.setTimeout(() => refreshAeroFeatures(token), delay);
}

async function refreshAeroFeatures(token) {
  const caseGeometryPath = state.viewer.meshSource === "case"
    ? state.caseReport?.geometryModelPath
    : null;
  const modelPath = caseGeometryPath || state.modelPath;
  const analyzingCaseGeometry = Boolean(caseGeometryPath);
  try {
    const payload = await fetchJson("/api/analyze-features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modelPath,
        unitScale: analyzingCaseGeometry ? 1 : effectiveUnitScale(),
        sourceFlowDirection: analyzingCaseGeometry ? "+x" : els.sourceFlowDirection.value,
        sourceUpDirection: analyzingCaseGeometry ? "+z" : els.sourceUpDirection.value,
        modelRotationDegrees: analyzingCaseGeometry ? { x: 0, y: 0, z: 0 } : modelRotationDegrees(),
      }),
    });
    if (token !== state.aeroFeatureScanToken) return;
    state.aeroFeatures = payload.features;
    state.aeroFeatureScanStatus = "complete";
  } catch (error) {
    if (token !== state.aeroFeatureScanToken) return;
    state.aeroFeatures = { candidate_count: 0, candidates: [], detail: error.message };
    state.aeroFeatureScanStatus = "failed";
  } finally {
    if (token === state.aeroFeatureScanToken) {
      state.aeroFeatureScanTimer = null;
      renderMetrics();
      renderReadiness();
    }
  }
}

async function loadExactStl(modelPath, suppliedData = null) {
  const requestId = state.viewer.exactMeshRequest + 1;
  state.viewer.exactMeshRequest = requestId;
  state.viewer.exactMesh = null;
  state.viewer.exactMeshLoading = true;
  invalidateThreeGeometry();
  renderMetrics();
  try {
    const data = suppliedData || await fetchArrayBuffer(
      `/api/model-file?path=${encodeURIComponent(modelPath)}`,
    );
    if (state.viewer.exactMeshRequest !== requestId) return;
    const parsed = parseStlBuffer(data);
    state.viewer.exactMesh = {
      ...parsed,
      path: modelPath,
      revision: requestId,
    };
    invalidateThreeGeometry();
  } catch (error) {
    if (state.viewer.exactMeshRequest === requestId) {
      console.warn("Exact STL display unavailable; using sampled preview geometry.", error);
    }
  } finally {
    if (state.viewer.exactMeshRequest === requestId) {
      state.viewer.exactMeshLoading = false;
      renderMetrics();
      drawFlow();
    }
  }
}

function parseStlBuffer(data) {
  const view = new DataView(data);
  if (data.byteLength >= 84) {
    const triangleCount = view.getUint32(80, true);
    if (84 + triangleCount * 50 === data.byteLength) {
      const positions = new Float32Array(triangleCount * 9);
      for (let triangle = 0; triangle < triangleCount; triangle += 1) {
        const sourceOffset = 84 + triangle * 50 + 12;
        const targetOffset = triangle * 9;
        for (let value = 0; value < 9; value += 1) {
          positions[targetOffset + value] = view.getFloat32(sourceOffset + value * 4, true);
        }
      }
      return { format: "binary", triangleCount, positions };
    }
  }

  const text = new TextDecoder().decode(data);
  const number = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?";
  const vertexPattern = new RegExp(`^\\s*vertex\\s+(${number})\\s+(${number})\\s+(${number})`, "gim");
  const values = [];
  let match = vertexPattern.exec(text);
  while (match) {
    values.push(Number(match[1]), Number(match[2]), Number(match[3]));
    match = vertexPattern.exec(text);
  }
  if (!values.length || values.length % 9 !== 0) {
    throw new Error("The exact STL does not contain complete triangle vertices.");
  }
  return {
    format: "ascii",
    triangleCount: values.length / 9,
    positions: new Float32Array(values),
  };
}

async function beginSourceAlignment() {
  if (state.viewer.meshSource !== "case" || state.viewer.alignmentLoading) return;
  const sourcePath = state.caseReport?.sourceModelPath || state.modelPath;
  if (!sourcePath) return;
  state.viewer.alignmentLoading = true;
  try {
    const payload = await fetchJson("/api/check-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelPath: sourcePath }),
    });
    state.modelPath = payload.modelPath;
    state.report = payload.report;
    state.mesh = payload.preview;
    state.caseReport = null;
    resetSolverParticles();
    state.viewer.flowLayer = null;
    syncSolverFlowControls();
    state.activeCasePath = null;
    state.activeRunProgress = null;
    syncRunLogForActiveCase();
    state.viewer.meshSource = "raw";
    state.viewer.surfaceMode = "material";
    syncSurfaceModeControls();
    state.viewer.modelLayer = null;
    state.viewer.flowEnvelope = null;
    state.viewer.meshBounds = null;
    state.viewer.orientedMesh = null;
    state.viewer.smoothVertexLight = null;
    state.viewer.flowLayer = null;
    void loadExactStl(payload.modelPath);
    els.modelName.textContent = basename(payload.modelPath);
    els.modelStatus.textContent = "Model alignment changed";
    els.candidateBadge.textContent = payload.report.is_cfd_candidate ? "Mesh Ready" : "Cleanup";
    els.candidateBadge.className = payload.report.is_cfd_candidate ? "badge" : "badge warn";
    updateActionAvailability();
    renderCases();
    renderResultSummary();
    renderRunProgress();
  } catch (error) {
    showError(error);
  } finally {
    state.viewer.alignmentLoading = false;
  }
}

function modelRotationDegrees() {
  return {
    x: Number(els.rotateX.value || 0),
    y: Number(els.rotateY.value || 0),
    z: Number(els.rotateZ.value || 0),
  };
}

function setModelRotation(rotation = {}) {
  els.rotateX.value = String(Number(rotation.x || 0));
  els.rotateY.value = String(Number(rotation.y || 0));
  els.rotateZ.value = String(Number(rotation.z || 0));
  syncRotationOutputs();
}

function syncRotationOutputs() {
  els.rotateXValue.value = fmtAngle(els.rotateX.value);
  els.rotateYValue.value = fmtAngle(els.rotateY.value);
  els.rotateZValue.value = fmtAngle(els.rotateZ.value);
}

function fmtAngle(value) {
  const angle = Number(value || 0);
  return Number.isInteger(angle) ? String(angle) : angle.toFixed(1);
}

function invalidateModelOrientation() {
  state.viewer.modelLayer = null;
  state.viewer.flowEnvelope = null;
  state.viewer.meshBounds = null;
  state.viewer.orientedMesh = null;
  state.viewer.smoothVertexLight = null;
  state.viewer.flowLayer = null;
  state.viewer.smokeTrails = [];
  invalidateThreeGeometry();
}

function updateCaseName() {
  if (!state.modelPath) return;
  const stem = basename(state.modelPath).replace(/\.stl$/i, "");
  const speed = Math.round(Number(els.speedMph.value || 70));
  els.caseName.value = `${slug(stem)}-${speed}mph`;
}

function renderMetrics() {
  const report = state.report;
  if (!report) {
    els.metrics.innerHTML = "";
    return;
  }
  const dims = report.bounds.dimensions;
  const unitScale = effectiveUnitScale();
  const transformedBounds = transformedDisplayBounds(report, unitScale);
  const scaledDims = transformedBounds?.dimensions || dims.map((dim) => dim * unitScale);
  const projectedAreas = report.projected_areas || {};
  const silhouetteAreas = report.silhouette_projected_areas || projectedAreas;
  const sourceFlowAxis = signedAxisName(els.sourceFlowDirection.value);
  const autoAref = Number(silhouetteAreas[sourceFlowAxis] || 0) * unitScale * unitScale;
  const rows = [
    ["Triangles", formatInt(report.triangle_count)],
    ["Vertices", formatInt(report.unique_vertex_count)],
    ["Raw size", `${fmt(dims[0])} x ${fmt(dims[1])} x ${fmt(dims[2])}`],
    ["CFD meters", `${fmt(scaledDims[0])} x ${fmt(scaledDims[1])} x ${fmt(scaledDims[2])}`],
    ["Scale", scaleLabel()],
    ["Auto Aref", `${fmt(autoAref)} m2`],
    [
      "Silhouette",
      `x ${fmt(silhouetteAreas.x || 0)}, y ${fmt(silhouetteAreas.y || 0)}, z ${fmt(silhouetteAreas.z || 0)}`,
    ],
    ["Surface", fmt(report.surface_area)],
    ["Volume", fmt(report.volume)],
    ["Open edges", formatInt(report.open_edge_count)],
    ["Non-manifold", formatInt(report.non_manifold_edge_count)],
    ["Display tris", previewTriangleLabel()],
    [
      "Diffuser scan",
      state.aeroFeatureScanStatus === "running"
        ? "Analyzing"
        : state.aeroFeatureScanStatus === "complete"
          ? `${Number(state.aeroFeatures?.candidate_count || 0)} candidate${Number(state.aeroFeatures?.candidate_count || 0) === 1 ? "" : "s"}`
          : "Not analyzed",
    ],
  ];
  els.metrics.innerHTML = rows
    .map(([label, value]) => `<div class="metric-row"><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderReadiness() {
  if (!els.readiness) return;
  const report = state.report;
  if (!report) {
    els.readiness.innerHTML = `<div class="readiness-empty">Load an STL to check scan and solver readiness.</div>`;
    updateActionAvailability();
    return;
  }

  const mesh = report.readiness || { score: report.is_cfd_candidate ? 100 : 55, items: [] };
  const setupItems = setupReadinessItems(report);
  const setupScore = setupItems.reduce((score, item) => {
    if (item.status === "fail") return score - 24;
    if (item.status === "warn") return score - 9;
    return score;
  }, 100);
  const allItems = [...(mesh.items || []), ...setupItems];
  const meshFailed = (mesh.items || []).some((item) => item.status === "fail");
  const setupFailed = setupItems.some((item) => item.status === "fail");
  const failed = meshFailed || setupFailed;
  const warned = allItems.some((item) => item.status === "warn");
  const score = clamp(Math.round((Number(mesh.score || 0) * 0.68 + Math.max(0, setupScore) * 0.32)), 0, 100);
  const status = meshFailed ? "Mesh cleanup" : setupFailed ? "Run blocked" : warned ? "Setup check" : "Ready";
  const statusClass = failed ? "fail" : warned ? "warn" : "pass";
  const meshScore = clamp(Math.round(Number(mesh.score || 0)), 0, 100);
  const runScore = clamp(Math.round(setupScore), 0, 100);

  els.readiness.innerHTML = `
    <div class="readiness-summary ${statusClass}">
      <div class="readiness-scores">
        <div class="readiness-score"><strong>${score}/100</strong><small>Overall</small></div>
        <div class="readiness-breakdown"><span>Mesh ${meshScore}/100</span><span>Run setup ${runScore}/100</span></div>
      </div>
      <span>${status}</span>
    </div>
    <div class="readiness-list">
      ${allItems.map(readinessItemHtml).join("")}
    </div>
  `;
  updateActionAvailability();
}

function setupReadinessItems(report) {
  const items = [];
  const scale = effectiveUnitScale();
  const transformedBounds = transformedDisplayBounds(report, scale);
  const dims = transformedBounds?.dimensions || report.bounds.dimensions.map((dim) => dim * scale);
  const flowLength = dims[flowAxisIndex()] || 0;
  const maxDim = Math.max(...dims, 0);
  const targetLength = optionalNumber(els.targetLength.value);
  const referenceArea = optionalNumber(els.referenceArea.value);
  const referenceLength = optionalNumber(els.referenceLength.value);
  const sourceFlow = signedAxisVector(els.sourceFlowDirection.value);
  const sourceUp = signedAxisVector(els.sourceUpDirection.value);
  const speedMph = Number(els.speedMph.value);
  const speedMps = speedMph * 0.44704;
  const machNumber = speedMps / 343;
  const engineering = state.viewMode === "engineering";
  const fluidProfile = engineering ? els.fluidProfile.value : "incompressible";
  const turbulenceModel = engineering ? els.turbulenceModel.value : "kOmegaSST";
  const transientMode = els.simulationMode.value === "transient";
  const hybridModel = turbulenceModel !== "kOmegaSST";
  const roughnessHeightM = engineering
    ? Math.max(0, Number(els.roughnessHeightMm.value || 0)) / 1000
    : 0;

  if (!Number.isFinite(speedMph) || speedMph <= 0) {
    items.push({
      label: "Flow regime",
      status: "fail",
      detail: "Enter a positive finite tunnel speed.",
    });
  } else if (fluidProfile === "incompressible" && machNumber >= 0.3) {
    items.push({
      label: "Flow regime",
      status: "fail",
      detail: `${fmt(speedMph)} mph is approximately Mach ${fmt(machNumber)}; select the compressible + thermal profile at Mach 0.3 or above.`,
    });
  } else if (fluidProfile === "compressible_thermal") {
    items.push({
      label: "Flow regime",
      status: "pass",
      detail: `${fmt(speedMph)} mph is approximately Mach ${fmt(machNumber)}; the fluid profile solves absolute pressure and temperature.`,
    });
  } else {
    items.push({
      label: "Flow regime",
      status: "pass",
      detail: `${fmt(speedMph)} mph is approximately Mach ${fmt(machNumber)} in standard air; incompressible treatment is applicable.`,
    });
  }

  const profileLabel = fluidProfile === "compressible_thermal"
    ? "Compressible + thermal fluid"
    : "Incompressible fluid";
  let modelStatus = "pass";
  let modelDetail = `${profileLabel}; ${turbulenceModel}`;
  if (hybridModel && !transientMode) {
    modelStatus = "fail";
    modelDetail = `${turbulenceModel} requires a transient case.`;
  } else if (hybridModel && roughnessHeightM > 0) {
    modelStatus = "fail";
    modelDetail = `${turbulenceModel} does not support the configured rough wall treatment.`;
  } else if (hybridModel) {
    modelDetail += "; transient mode enforced";
  }
  items.push({
    label: "Solver model",
    status: modelStatus,
    detail: `${modelDetail}.`,
  });

  if (engineering) {
    try {
      const porousZones = volumeZonesPayload(els.porousZonesJson, "Porous zones") || [];
      const fanZones = volumeZonesPayload(els.fanZonesJson, "Fan zones") || [];
      const heatZones = volumeZonesPayload(els.heatZonesJson, "Heat-load zones") || [];
      const zoneCount = porousZones.length + fanZones.length + heatZones.length;
      const heatProfileMismatch = heatZones.length > 0 && fluidProfile !== "compressible_thermal";
      items.push({
        label: "Volume zones",
        status: heatProfileMismatch ? "fail" : "pass",
        detail: heatProfileMismatch
          ? "Heat-load zones require the Compressible + thermal fluid profile."
          : zoneCount
            ? `${porousZones.length} porous, ${fanZones.length} fan, and ${heatZones.length} heat-load zone${zoneCount === 1 ? "" : "s"} will be generated from explicit solver-coordinate boxes.`
            : "No porous, fan, or heat-load volume zones are configured.",
      });
    } catch (error) {
      items.push({
        label: "Volume zones",
        status: "fail",
        detail: error instanceof Error ? error.message : "Volume-zone JSON is invalid.",
      });
    }
  }

  const airTemperatureC = optionalFiniteNumber(els.airTemperatureC.value);
  const airPressurePa = optionalNumber(els.airPressurePa.value);
  const airDensity = optionalNumber(els.airDensity.value);
  const viscosity = optionalNumber(els.kinematicViscosity.value);
  const turbulenceIntensity = optionalNumber(els.turbulenceIntensity.value);
  const turbulenceLength = optionalNumber(els.turbulenceLengthScale.value);
  const atmosphereValid = airTemperatureC != null
    && airTemperatureC >= -123.15
    && airTemperatureC <= 126.85
    && airPressurePa > 0
    && (airDensity == null || airDensity > 0)
    && (viscosity == null || viscosity > 0);
  items.push({
    label: "Air properties",
    status: atmosphereValid ? "pass" : "fail",
    detail: atmosphereValid
      ? `${fmt(airTemperatureC)} C and ${formatInt(airPressurePa)} Pa; density and viscosity are ${airDensity || viscosity ? "manually overridden where entered" : "derived and written to OpenFOAM"}.`
      : "Enter a supported dry-air temperature, positive absolute pressure, and positive optional density/viscosity overrides.",
  });
  const turbulenceValid = turbulenceIntensity > 0
    && turbulenceIntensity <= 50
    && (turbulenceLength == null || turbulenceLength > 0);
  items.push({
    label: "Inlet turbulence",
    status: turbulenceValid ? "pass" : "fail",
    detail: turbulenceValid
      ? `${fmt(turbulenceIntensity)}% intensity; ${turbulenceLength ? `${fmt(turbulenceLength)} m` : "7% of reference length"} length scale.`
      : "Set turbulence intensity above 0% and at most 50%, with a positive optional length scale.",
  });

  if (Math.abs(dot3(sourceFlow, sourceUp)) > 1e-9) {
    items.push({
      label: "Orientation",
      status: "fail",
      detail: "Source flow and source up cannot use the same axis.",
    });
  } else {
    const rotation = modelRotationDegrees();
    items.push({
      label: "Orientation",
      status: "pass",
      detail: `${els.sourceFlowDirection.value.toUpperCase()} maps to ${els.flowAxis.value.toUpperCase()}; rotation X ${fmtAngle(rotation.x)}, Y ${fmtAngle(rotation.y)}, Z ${fmtAngle(rotation.z)} deg.`,
    });
  }

  const alignment = report.alignment_suggestion;
  if (alignment?.recommended && alignment.rotation_degrees) {
    const current = modelRotationDegrees();
    const suggested = alignment.rotation_degrees;
    const delta = Math.max(
      angularDistance(current.x, suggested.x),
      angularDistance(current.y, suggested.y),
      angularDistance(current.z, suggested.z),
    );
    const applied = delta <= 1.5
      && els.sourceFlowDirection.value === "+x"
      && els.sourceUpDirection.value === "+z";
    items.push({
      label: "Scan alignment",
      status: applied ? "pass" : "warn",
      detail: applied
        ? `${alignment.confidence || "Geometry"} confidence principal-axis fit is applied.`
        : `Suggested X ${fmtAngle(suggested.x)}, Y ${fmtAngle(suggested.y)}, Z ${fmtAngle(suggested.z)} deg; use Auto-align, then verify front and top.`,
    });
  }

  if (targetLength && targetLength > 0) {
    items.push({
      label: "Geometry scale",
      status: "pass",
      detail: `Scaled to ${fmt(flowLength)} m along ${els.flowAxis.value.toUpperCase()}.`,
    });
  } else if (maxDim > 8 || (maxDim > 0 && maxDim < 1)) {
    items.push({
      label: "Geometry scale",
      status: "warn",
      detail: "Scaled size is unusual for a full vehicle; set real length if this scan is not full scale.",
    });
  } else {
    items.push({
      label: "Geometry scale",
      status: "pass",
      detail: `Scaled size is ${fmt(dims[0])} x ${fmt(dims[1])} x ${fmt(dims[2])} m.`,
    });
  }

  if (els.includeGround.checked && flowLength > 0) {
    const sideAxis = flowAxisIndex() === 0 ? 1 : 0;
    const sideWidth = Number(dims[sideAxis] || 0);
    const height = Number(dims[2] || 0);
    const lengthToWidth = sideWidth > 0 ? flowLength / sideWidth : 0;
    const heightToLength = height / flowLength;
    const plausible = lengthToWidth >= 1.45
      && lengthToWidth <= 4.5
      && heightToLength >= 0.15
      && heightToLength <= 0.65;
    items.push({
      label: "Vehicle proportions",
      status: plausible ? "pass" : "warn",
      detail: plausible
        ? `Length/width ${fmt(lengthToWidth)} and height/length ${fmt(heightToLength)} are plausible for a road vehicle.`
        : `Scaled body is ${fmt(flowLength)} m long, ${fmt(sideWidth)} m wide, and ${fmt(height)} m high; confirm scan proportions before trusting Cd/Cl.`,
    });
  }

  const dimensionCheck = geometryDimensionCheck(report);
  if (dimensionCheck.status === "pass") {
    items.push({
      label: "Measured dimensions",
      status: "pass",
      detail: `Length, width, and height agree with measurements; maximum error ${fmt(dimensionCheck.maxErrorPercent)}%.`,
    });
  } else if (dimensionCheck.status === "fail") {
    items.push({
      label: "Measured dimensions",
      status: "fail",
      detail: `${dimensionCheck.failedLabel} differs by ${fmt(dimensionCheck.maxErrorPercent)}%; limit is 2%.`,
    });
  } else {
    const actual = dimensionCheck.actualDimensions || {};
    items.push({
      label: "Measured dimensions",
      status: "warn",
      detail: `STL coordinates calculate ${fmt(actual.length)} x ${fmt(actual.width)} x ${fmt(actual.height)} m. Enter independent tape/CAD ${dimensionCheck.missingLabels.join(", ").toLowerCase()} values before an accuracy study can be numerically qualified.`,
    });
  }

  const diffuserCandidates = Array.isArray(state.aeroFeatures?.candidates)
    ? state.aeroFeatures.candidates
    : [];
  if (state.aeroFeatureScanStatus === "failed") {
    items.push({
      label: "Aero feature scan",
      status: "warn",
      detail: "Automatic rear-underbody geometry analysis did not complete.",
    });
  } else if (diffuserCandidates.length) {
    const diffuser = diffuserCandidates[0];
    items.push({
      label: "Diffuser candidate",
      status: diffuser.confidence === "high" ? "pass" : "warn",
      detail: `${diffuser.confidence} confidence rear underbody ramp: ${fmt(diffuser.angle_degrees)} deg, ${fmt(diffuser.length_m)} m long, ${fmt(diffuser.width_m)} m wide. Confirm visually; STL triangles do not encode part names or duct connectivity.`,
    });
  } else if (state.aeroFeatureScanStatus === "complete") {
    items.push({
      label: "Diffuser candidate",
      status: "pass",
      detail: "No qualifying rear underbody ramp was found in the current orientation; this does not prove a diffuser is absent.",
    });
  }

  if (referenceArea && referenceLength) {
    items.push({
      label: "Aero references",
      status: "pass",
      detail: "Manual reference area and length are set.",
    });
  } else if (referenceArea || referenceLength) {
    items.push({
      label: "Aero references",
      status: "warn",
      detail: "Set both reference area and length for cleaner Cd/Cl comparisons.",
    });
  } else {
    items.push({
      label: "Aero references",
      status: "warn",
      detail: "Using automatic triangle-silhouette area and flow-axis length.",
    });
  }

  const cgValues = [els.cgX, els.cgY, els.cgZ].map((input) => optionalFiniteNumber(input.value));
  const cgCount = cgValues.filter((value) => value != null).length;
  const axleValues = [
    optionalFiniteNumber(els.frontAxleStation.value),
    optionalFiniteNumber(els.rearAxleStation.value),
  ];
  const axleCount = axleValues.filter((value) => value != null).length;
  if (cgCount === 0 && axleCount === 0) {
    items.push({
      label: "Vehicle datums",
      status: "warn",
      detail: "Moments will use the geometry center; enter solver-coordinate CG and both axle stations to qualify aero balance.",
    });
  } else if (cgCount !== 3 || axleCount === 1 || (axleCount && cgCount !== 3)) {
    items.push({
      label: "Vehicle datums",
      status: "fail",
      detail: "Enter all three CG coordinates together and provide front/rear axle stations as a pair.",
    });
  } else if (axleCount === 0) {
    items.push({
      label: "Vehicle datums",
      status: "warn",
      detail: "CG qualifies the moment reference, but front/rear axle stations are still required for aero balance.",
    });
  } else {
    const cgStation = cgValues[flowAxisIndex()];
    const [frontAxle, rearAxle] = axleValues;
    const validOrder = frontAxle < cgStation && cgStation < rearAxle;
    items.push({
      label: "Vehicle datums",
      status: validOrder ? "pass" : "fail",
      detail: validOrder
        ? `CG lies between axle stations along +${els.flowAxis.value.toUpperCase()}; wheelbase ${fmt(rearAxle - frontAxle)} m.`
        : `Require front axle < CG < rear axle along solver +${els.flowAxis.value.toUpperCase()}.`,
    });
  }

  if (state.repair?.accepted) {
    items.push({
      label: "Repair fidelity",
      status: "pass",
      detail: `Source surface p95 ${fmt(state.repair.sourceSurfaceDeviationP95Percent)}%, p99 ${fmt(state.repair.sourceSurfaceDeviationP99Percent)}%; ${fmt(state.repair.addedSurfaceFarFractionPercent)}% of sealing surface lies beyond ${fmt(state.repair.addedSurfaceFarDistancePercent)}% of model length.`,
    });
  } else {
    const fidelity = report.repair_fidelity || state.caseReport?.geometryFidelity;
    if (fidelity) {
      items.push({
        label: "Geometry fidelity",
        status: fidelity.verified ? "pass" : "fail",
        detail: fidelity.verified
          ? fidelity.detail || "Prepared geometry has a matching accepted fidelity record."
          : fidelity.detail || "Prepared geometry fidelity is not verified.",
      });
    }
  }

  const wallResolution = state.caseReport?.wallResolution;
  if (wallResolution?.surface_layers > 0) {
    items.push({
      label: "Wall layers",
      status: "pass",
      detail: `${wallResolution.surface_layers} layers, ${(Number(wallResolution.first_layer_thickness_m) * 1000).toFixed(2)} mm first cell, target y+ ${fmt(wallResolution.target_y_plus)}.`,
    });
    const layerCoverage = state.caseReport?.layerCoverage;
    const completedRun = state.caseReport?.lastRun?.returncode === 0;
    items.push({
      label: "Boundary-layer coverage",
      status: layerCoverage ? (layerCoverage.passed ? "pass" : "fail") : completedRun ? "fail" : "warn",
      detail: layerCoverage
        ? `${fmt(layerCoverage.fullLayerFaceCoveragePercent)}% of body faces received the complete stack; ${fmt(layerCoverage.layerCellCoveragePercent)}% of requested prism cells were added.`
        : completedRun
          ? "The run log has no measurable prism-layer coverage; recreate and rerun this case."
          : "OpenFOAM will verify complete prism-layer coverage after meshing.",
    });
  }

  if (els.includeGround.checked && els.movingGround.checked) {
    items.push({
      label: "Road condition",
      status: "pass",
      detail: "Ground and moving road are enabled.",
    });
  } else if (els.includeGround.checked) {
    items.push({
      label: "Road condition",
      status: "warn",
      detail: "Ground is enabled, but moving road is off.",
    });
  } else {
    items.push({
      label: "Road condition",
      status: "warn",
      detail: "Open tunnel mode; enable ground for road-car runs.",
    });
  }

  if (els.includeGround.checked) {
    const gapMm = Math.max(0, Number(els.groundClearanceMm.value || 0));
    const storedPlacement = state.caseReport?.caseSetup?.placement;
    const legacyPlacement = state.viewer.meshSource === "case" && !storedPlacement?.verified;
    const unusuallyHigh = gapMm > 500;
    items.push({
      label: "Road placement",
      status: legacyPlacement || unusuallyHigh ? "warn" : "pass",
      detail: legacyPlacement
        ? "This case predates explicit road placement; recreate it before trusting underbody or diffuser results."
        : unusuallyHigh
          ? `The STL's lowest point is ${fmt(gapMm)} mm above the road; confirm this unusually large gap.`
          : gapMm > 0
            ? `The STL's lowest point will be placed ${fmt(gapMm)} mm above the road.`
            : "The STL's lowest point touches the road; use zero only when tire contact is included in the model.",
    });
  }

  if (els.includeGround.checked && els.flowAxis.value === "z") {
    items.push({
      label: "Tunnel axis",
      status: "fail",
      detail: "Ground runs require X or Y flow; Z flow would put the inlet on the road plane.",
    });
  }

  const quality = els.qualityPreset.value;
  items.push({
    label: "Flow solution",
    status: transientMode ? "pass" : "warn",
    detail: transientMode
      ? "Transient PIMPLE with adaptive time stepping and time-averaged wake, pressure, and force output selected."
      : "Steady RANS is efficient for setup and mean-flow checks, but it cannot resolve a changing separated wake.",
  });
  if (quality === "fine") {
    items.push({
      label: "CFD quality",
      status: "pass",
      detail: transientMode
        ? "Fine mesh with the longest wake-development and averaging windows selected."
        : "Fine mesh with residual-controlled convergence and up to 2,000 steady iterations selected.",
    });
  } else if (quality === "standard") {
    items.push({
      label: "CFD quality",
      status: "pass",
      detail: transientMode
        ? "Standard mesh with extended wake-development and time-averaging windows selected."
        : "Standard mesh with residual-controlled convergence and up to 1,200 steady iterations selected.",
    });
  } else {
    items.push({
      label: "CFD quality",
      status: "warn",
      detail: "Draft is best for setup checks, not final aerodynamic comparison.",
    });
  }

  const smallestFeatureMm = optionalNumber(els.smallestFeatureMm.value);
  if (smallestFeatureMm && smallestFeatureMm > 0) {
    const meshResolution = estimatePreviewMeshResolution(
      dims,
      quality,
      els.includeGround.checked,
      smallestFeatureMm,
    );
    const cellsAcross = smallestFeatureMm / meshResolution.surfaceCellMm;
    items.push({
      label: "Aero feature resolution",
      status: meshResolution.supported ? "pass" : "fail",
      detail: meshResolution.supported
        ? `Adaptive level ${meshResolution.surfaceLevel} targets ${fmt(meshResolution.surfaceCellMm)} mm sharp-feature cells (${fmt(cellsAcross)} across); broad surface about ${fmt(meshResolution.broadSurfaceCellMm)} mm with a ${formatInt(meshResolution.maxGlobalCells)} cell cap.`
        : `${fmt(smallestFeatureMm)} mm requires level ${meshResolution.requiredLevel}, above the local-device limit ${meshResolution.maximumLevel}; the smallest supported target here is ${fmt(meshResolution.minimumSupportedFeatureMm)} mm.`,
    });
  } else {
    items.push({
      label: "Aero feature resolution",
      status: quality === "draft" ? "warn" : "fail",
      detail: quality === "draft"
        ? "Set the smallest relevant aero feature before moving beyond setup checks."
        : "Enter the smallest diffuser strake, vortex generator, gap, or body detail whose pressure effect must be resolved.",
    });
  }

  const meshSurfaceFidelity = state.caseReport?.meshSurfaceFidelity;
  if (meshSurfaceFidelity) {
    const p95Mm = Number(meshSurfaceFidelity.symmetricP95M || 0) * 1000;
    const p99Mm = Number(meshSurfaceFidelity.symmetricP99M || 0) * 1000;
    const limitP99Mm = Number(meshSurfaceFidelity.maximumP99M || 0) * 1000;
    const meshTriangles = formatInt(meshSurfaceFidelity.meshTriangleCount || 0);
    items.push({
      label: "Meshed body fidelity",
      status: meshSurfaceFidelity.verified ? "pass" : "fail",
      detail: meshSurfaceFidelity.verified
        ? `${meshTriangles} OpenFOAM body faces preserve the solver STL; two-way deviation p95 ${fmtDistanceMm(p95Mm)}, p99 ${fmtDistanceMm(p99Mm)}.`
        : `${meshSurfaceFidelity.detail || "The OpenFOAM body patch did not preserve the solver STL."} Two-way p99 ${fmtDistanceMm(p99Mm)} versus ${fmtDistanceMm(limitP99Mm)} limit.`,
    });
  } else if (state.activeCasePath) {
    const completedRun = state.caseReport?.lastRun?.returncode === 0;
    items.push({
      label: "Meshed body fidelity",
      status: completedRun ? "fail" : "warn",
      detail: completedRun
        ? "This older run has no body-patch fidelity audit; rerun it before trusting small aero features."
        : "Run the case to verify that OpenFOAM preserved body lines and small aero features in the actual CFD boundary.",
    });
  }

  const surfacePressure = state.caseReport?.surfacePressure;
  const pressureSetup = state.caseReport?.surfacePressureSetup;
  const solvedPressure = state.caseReport?.solverStreamlines?.hasPressure;
  if (surfacePressure?.hasPressure) {
    items.push({
      label: "Pressure field",
      status: "pass",
      detail: `Solved body Cp covers ${formatInt(surfacePressure.triangleCount)} displayed triangles${surfacePressure.decimatedForBrowser ? ` from ${formatInt(surfacePressure.sourceTriangleCount)}` : ""} at q ${formatInt(Math.round(surfacePressure.dynamicPressurePa || 0))} Pa.`,
    });
    if (surfacePressure.hasPressureDrag) {
      const pressureCd = Number(surfacePressure.pressureDragCoefficient);
      const skinCd = Number(surfacePressure.skinFrictionDragCoefficient);
      const mappedTotalCd = Number(surfacePressure.totalDragCoefficient);
      const forceCd = Number(state.caseReport?.forceCoeffs?.meanCd ?? state.caseReport?.forceCoeffs?.Cd);
      const hasWallShear = surfacePressure.hasWallShear && Number.isFinite(skinCd);
      const comparison = hasWallShear && Number.isFinite(mappedTotalCd)
        ? `Mapped pressure Cd ${fmt(pressureCd)} + skin Cd ${fmt(skinCd)} = total Cd ${fmt(mappedTotalCd)}${Number.isFinite(forceCd) ? ` versus force Cd ${fmt(forceCd)}` : ""}. `
        : Number.isFinite(pressureCd)
          ? `Mapped pressure Cd ${fmt(pressureCd)}${Number.isFinite(forceCd) ? ` versus total solved Cd ${fmt(forceCd)}` : ""}. `
          : "";
      items.push({
        label: hasWallShear ? "Total drag map" : "Pressure drag map",
        status: "pass",
        detail: hasWallShear
          ? `${comparison}Red adds local total drag and blue offsets it; integration uses original solver faces before browser decimation.`
          : `${comparison}Red adds pressure drag and blue offsets it. Rerun this case to add wall-shear zones.`,
      });
    }
  } else if (solvedPressure) {
    items.push({
      label: "Pressure field",
      status: "warn",
      detail: "Streamlines carry pressure, but the body Cp output is missing. Rerun this generated case to map panel loading.",
    });
  } else if (state.activeCasePath && pressureSetup?.configured) {
    items.push({
      label: "Body Cp output",
      status: "warn",
      detail: "Body-pressure export is configured and will become available after a completed solver run.",
    });
  } else if (state.activeCasePath) {
    items.push({
      label: "Body Cp output",
      status: "warn",
      detail: "This legacy case predates body-pressure export; AeroLab will upgrade it automatically before the next run.",
    });
  }

  if (state.solver?.preferredBackend) {
    items.push({
      label: "Solver backend",
      status: "pass",
      detail: `${state.solver.preferredBackend} is available.`,
    });
  } else {
    const wslMessage = state.solver?.backends?.wsl?.message || "";
    const missingWsl = wslMessage.toLowerCase().includes("not installed");
    items.push({
      label: "Solver backend",
      status: "fail",
      detail: missingWsl ? "WSL2 is not installed, so OpenFOAM cannot run yet." : "No OpenFOAM backend detected yet.",
    });
  }

  const assessment = state.caseReport?.qualityAssessment;
  const assessedRun = (assessment?.checks || []).some(
    (check) => check.label === "Solver process" && check.status !== "pending",
  );
  if (assessment?.trusted) {
    items.push({
      label: "Solver result",
      status: "pass",
      detail: "Selected case passed body-fidelity, mesh, residual, wall-resolution, and force-stability checks.",
    });
  } else if (state.caseReport?.lastRun || assessedRun) {
    const failedChecks = (assessment?.checks || [])
      .filter((check) => check.status === "fail")
      .map((check) => check.label)
      .join(", ");
    items.push({
      label: "Solver result",
      status: "fail",
      detail: failedChecks ? `Qualification failed: ${failedChecks}.` : "The selected run is not numerically qualified.",
    });
  } else if (state.activeCasePath) {
    items.push({
      label: "Solver result",
      status: "warn",
      detail: "The selected case has not completed a numerically qualified solver run.",
    });
  }

  const grid = state.caseReport?.gridConvergence;
  if (grid?.validated) {
    items.push({
      label: "Mesh sensitivity",
      status: "pass",
      detail: `Within mesh-sensitivity threshold; fine-grid Cd ${fmt(grid.recommendedCd)} and Cl ${fmt(grid.recommendedCl)}.`,
    });
  } else if (grid?.status === "failed") {
    const reason = grid.checks?.find((check) => check.status === "fail")?.detail || "The mesh-sensitivity study failed.";
    items.push({ label: "Mesh sensitivity", status: "fail", detail: reason });
  } else if (grid) {
    const completed = grid.levels?.filter((level) => level.trusted).length || 0;
    items.push({
      label: "Mesh sensitivity",
      status: "warn",
      detail: `${completed}/3 study runs numerically qualified; complete draft, standard, and fine cases.`,
    });
  } else {
    items.push({
      label: "Mesh sensitivity",
      status: "warn",
      detail: "Create an accuracy study before treating force coefficients as within the mesh-sensitivity threshold.",
    });
  }

  const transientStatistics = state.caseReport?.transientStatistics;
  if (transientStatistics?.overall_evidence) {
    const overall = transientStatistics.overall_evidence;
    const channels = transientStatistics.channels || {};
    const effectiveCounts = Object.values(channels)
      .map((channel) => Number(channel?.effective_sample_count))
      .filter(Number.isFinite);
    const minimumEffective = effectiveCounts.length ? Math.min(...effectiveCounts) : null;
    items.push({
      label: "Statistical stationarity",
      status: overall.stationarity_supported === true
        ? "pass"
        : overall.stationarity_supported === false
          ? "fail"
          : "warn",
      detail: overall.stationarity_supported === true
        ? "No requested channel has statistically resolved half-window or linear drift at the configured confidence level; this supports, but does not prove, stationarity."
        : overall.stationarity_supported === false
          ? "At least one requested channel has statistically resolved drift; extend or revise the retained averaging window."
          : "The retained history is too short or too correlated to assess stationarity reliably.",
    });
    items.push({
      label: "Effective samples",
      status: overall.minimum_effective_samples_30 === true
        ? "pass"
        : overall.minimum_effective_samples_30 === false
          ? "fail"
          : "warn",
      detail: minimumEffective != null
        ? `Minimum autocorrelation-adjusted effective sample count is ${fmt(minimumEffective)}; at least 30 are required in every requested channel.`
        : "Effective sample counts are unavailable for the retained transient history.",
    });
    const confidenceSummaries = ["Cd", "Cl", "Cs", "CmPitch", "frontAeroBalancePercent"]
      .map((name) => [name, channels[name]])
      .filter(([, channel]) => channel?.confidence_interval?.lower != null && channel?.confidence_interval?.upper != null)
      .slice(0, 4)
      .map(([name, channel]) => `${name} ${fmt(channel.mean)} [${fmt(channel.confidence_interval.lower)}, ${fmt(channel.confidence_interval.upper)}]`);
    items.push({
      label: "Mean confidence intervals",
      status: confidenceSummaries.length ? "pass" : "warn",
      detail: confidenceSummaries.length
        ? `Autocorrelation-adjusted 95% evidence: ${confidenceSummaries.join("; ")}.`
        : "No autocorrelation-adjusted confidence intervals are available yet.",
    });
    const meaningfulPeaks = Object.entries(channels)
      .filter(([, channel]) => channel?.spectrum?.meaningful_peak)
      .map(([name, channel]) => ({ name, ...channel.spectrum }));
    const peakCoverage = overall.meaningful_peak_has_at_least_10_cycles;
    items.push({
      label: "Spectral cycle coverage",
      status: meaningfulPeaks.length ? (peakCoverage === true ? "pass" : "fail") : "pass",
      detail: meaningfulPeaks.length
        ? meaningfulPeaks.slice(0, 4).map((peak) => `${peak.name} ${fmt(peak.dominant_frequency_hz)} Hz, ${fmt(peak.cycle_coverage)} cycles${peak.strouhal_number == null ? "" : `, St ${fmt(peak.strouhal_number)}`}`).join("; ")
        : "No meaningful coherent spectral peak was detected, so the 10-cycle peak-coverage gate is not applicable.",
    });
  } else if (state.activeCasePath && els.simulationMode.value === "transient") {
    items.push({
      label: "Transient statistics",
      status: "warn",
      detail: "Run the transient case to evaluate washout, stationarity, effective samples, confidence intervals, and spectral cycle coverage.",
    });
  }

  const sensitivityStudy = state.caseReport?.sensitivityStudy;
  if (sensitivityStudy) {
    const familyStatus = String(sensitivityStudy.status || "pending").replaceAll("_", " ");
    items.push({
      label: "Sensitivity family",
      status: sensitivityStudy.parameterControlled
        ? sensitivityStudy.complete ? "pass" : "warn"
        : "fail",
      detail: `${sensitivityStudy.parameterLabel || sensitivityStudy.parameter} varies across ${sensitivityStudy.records?.length || 0}/${sensitivityStudy.values?.length || 0} members; setup lock ${sensitivityStudy.planLockVerified ? "verified" : "unverified"}; recorded values ${sensitivityStudy.parameterValuesVerified ? "verified" : "do not match cases"}; status ${familyStatus}.`,
    });
    items.push({
      label: "Sensitivity decision evidence",
      status: sensitivityStudy.decisionSafeSensitivity
        ? "pass"
        : sensitivityStudy.status === "plan_lock_mismatch"
          ? "fail"
          : "warn",
      detail: sensitivityStudy.decisionSafeSensitivity
        ? "Every member is numerically qualified and statistically ready; baseline difference intervals may be interpreted within this tested family."
        : `Numerical qualification: ${sensitivityStudy.allNumericallyQualified ? "complete" : "incomplete"}; statistical evidence: ${sensitivityStudy.allStatisticallyReady ? "complete" : "incomplete"}. These remain separate gates.`,
    });
  }

  return items;
}

function angularDistance(first, second) {
  const difference = ((Number(first || 0) - Number(second || 0) + 180) % 360 + 360) % 360 - 180;
  return Math.abs(difference);
}

function estimatePreviewMeshResolution(dimensions, quality, includeGround, smallestFeatureMm = null) {
  const presets = {
    draft: { divisions: 28, maxBlockCells: 64, surfaceMinLevel: 1, surfaceMaxLevel: 3, maximumLevel: 8, maxGlobalCells: 1_000_000, adaptiveMaxGlobalCells: 2_000_000 },
    standard: { divisions: 40, maxBlockCells: 80, surfaceMinLevel: 4, surfaceMaxLevel: 6, maximumLevel: 8, maxGlobalCells: 12_000_000, adaptiveMaxGlobalCells: 12_000_000 },
    fine: { divisions: 56, maxBlockCells: 120, surfaceMinLevel: 5, surfaceMaxLevel: 7, maximumLevel: 9, maxGlobalCells: 32_000_000, adaptiveMaxGlobalCells: 32_000_000 },
  };
  const preset = presets[quality] || presets.standard;
  const flowIndex = flowAxisIndex();
  const dims = dimensions.map((value) => Math.max(Number(value) || 0, 1e-6));
  const flowDimension = Math.max(dims[flowIndex], 1);
  const crossDimension = Math.max(
    1,
    ...dims.filter((_, index) => index !== flowIndex),
  );
  const tunnelSizes = dims.map((dimension, index) => {
    if (index === flowIndex) return flowDimension * 11;
    if (includeGround && index === 2) return dimension + crossDimension * 3;
    return dimension + crossDimension * 6;
  });
  const targetCell = Math.max(...tunnelSizes) / preset.divisions;
  const blockCellDimensions = tunnelSizes.map((size) => {
    const cells = clamp(Math.ceil(size / Math.max(targetCell, 1e-6)), 12, preset.maxBlockCells);
    return size / cells;
  });
  const conservativeBaseCellM = Math.max(...blockCellDimensions);
  const requestedCellM = Number(smallestFeatureMm) > 0 ? Number(smallestFeatureMm) / 4000 : null;
  const requiredLevel = requestedCellM
    ? Math.max(preset.surfaceMaxLevel, Math.ceil(Math.log2(conservativeBaseCellM / requestedCellM) - 1e-12))
    : preset.surfaceMaxLevel;
  const maximumLevel = preset.maximumLevel;
  const surfaceLevel = Math.min(requiredLevel, maximumLevel);
  const extraLevels = Math.max(0, surfaceLevel - preset.surfaceMaxLevel);
  const budgetMultiplier = 2 ** Math.min(extraLevels, 3);
  return {
    surfaceLevel,
    requiredLevel,
    maximumLevel,
    supported: requiredLevel <= maximumLevel,
    surfaceCellMm: conservativeBaseCellM * 1000 / (2 ** surfaceLevel),
    broadSurfaceCellMm: conservativeBaseCellM * 1000 / (2 ** preset.surfaceMinLevel),
    minimumSupportedFeatureMm: conservativeBaseCellM * 1000 * 4 / (2 ** maximumLevel),
    maxGlobalCells: Math.min(preset.maxGlobalCells * budgetMultiplier, preset.adaptiveMaxGlobalCells),
  };
}

function readinessItemHtml(item) {
  return `
    <div class="readiness-item ${escapeAttr(item.status)}">
      <span>${escapeHtml(item.status || "")}</span>
      <div>
        <strong>${escapeHtml(item.label || "")}</strong>
        <p>${escapeHtml(item.detail || "")}</p>
      </div>
    </div>
  `;
}

function previewTriangleLabel() {
  if (!state.mesh) return "0";
  if (state.viewer.exactMesh?.triangleCount) {
    return `${formatInt(state.viewer.exactMesh.triangleCount)} exact`;
  }
  const sampled = formatInt(state.mesh.sampledTriangleCount || 0);
  const total = formatInt(state.mesh.triangleCount || 0);
  if (state.viewer.exactMeshLoading) return `${sampled} loading exact`;
  return state.mesh.isComplete ? sampled : `${sampled} / ${total} preview`;
}

function effectiveUnitScale() {
  const targetLength = optionalNumber(els.targetLength.value);
  const orientedBounds = state.report ? transformedDisplayBounds(state.report, 1) : null;
  const axisLength = orientedBounds?.dimensions?.[flowAxisIndex()] || 0;
  if (targetLength && targetLength > 0 && axisLength > 0) {
    return targetLength / axisLength;
  }
  return Number(els.unitScale.value || 1);
}

function scaleLabel() {
  const targetLength = optionalNumber(els.targetLength.value);
  if (targetLength && targetLength > 0 && state.report?.bounds?.dimensions?.[sourceFlowAxisIndex()] > 0) {
    return `real length ${fmt(targetLength)} m`;
  }
  return selectedUnitLabel();
}

function geometryDimensionCheck(report = state.report) {
  if (!report?.bounds) {
    return { status: "incomplete", maxErrorPercent: 0, failedLabel: "", missingLabels: ["Length", "width", "height"], actualDimensions: {} };
  }
  const scale = effectiveUnitScale();
  const transformed = transformedDisplayBounds(report, scale);
  const dims = transformed?.dimensions || report.bounds.dimensions.map((value) => value * scale);
  const flowIndex = flowAxisIndex();
  const upIndex = flowIndex === 2 ? 1 : 2;
  const sideIndex = [0, 1, 2].find((index) => index !== flowIndex && index !== upIndex);
  const values = [
    { label: "Length", actual: Number(dims[flowIndex] || 0), measured: optionalNumber(els.targetLength.value) },
    { label: "Width", actual: Number(dims[sideIndex] || 0), measured: optionalNumber(els.targetWidth.value) },
    { label: "Height", actual: Number(dims[upIndex] || 0), measured: optionalNumber(els.targetHeight.value) },
  ];
  const actualDimensions = {
    length: values[0].actual,
    width: values[1].actual,
    height: values[2].actual,
  };
  const missingLabels = values.filter((item) => !item.measured).map((item) => item.label);
  const errors = values.filter((item) => item.measured).map((item) => ({
    ...item,
    errorPercent: Math.abs(item.actual - item.measured) / item.measured * 100,
  }));
  const failed = errors.filter((item) => item.errorPercent > 2);
  if (failed.length) {
    const worst = failed.reduce((current, item) => item.errorPercent > current.errorPercent ? item : current);
    return {
      status: "fail",
      maxErrorPercent: worst.errorPercent,
      failedLabel: worst.label,
      missingLabels,
      actualDimensions,
    };
  }
  if (missingLabels.length) {
    return { status: "incomplete", maxErrorPercent: 0, failedLabel: "", missingLabels, actualDimensions };
  }
  const worst = errors.reduce((current, item) => item.errorPercent > current.errorPercent ? item : current);
  return {
    status: worst.errorPercent <= 2 ? "pass" : "fail",
    maxErrorPercent: worst.errorPercent,
    failedLabel: worst.label,
    missingLabels: [],
    actualDimensions,
  };
}

function flowAxisIndex() {
  const axis = els.flowAxis.value;
  if (axis === "y") return 1;
  if (axis === "z") return 2;
  return 0;
}

function sourceFlowAxisIndex() {
  return signedAxisIndex(els.sourceFlowDirection.value);
}

function signedAxisName(axis) {
  return axis.replace("+", "").replace("-", "").toLowerCase();
}

function signedAxisIndex(axis) {
  const name = signedAxisName(axis);
  if (name === "y") return 1;
  if (name === "z") return 2;
  return 0;
}

function signedAxisVector(axis) {
  const sign = axis.trim().startsWith("-") ? -1 : 1;
  const name = signedAxisName(axis);
  if (name === "y") return [0, sign, 0];
  if (name === "z") return [0, 0, sign];
  return [sign, 0, 0];
}

function transformedDisplayBounds(report, scale) {
  const bounds = report?.bounds;
  if (!bounds) return null;
  const sourceFlow = signedAxisVector(els.sourceFlowDirection.value);
  const sourceUp = signedAxisVector(els.sourceUpDirection.value);
  if (Math.abs(dot3(sourceFlow, sourceUp)) > 1e-9) return null;
  const targetFlow = signedAxisVector(`+${els.flowAxis.value}`);
  const targetUp = targetFlow[2] === 0 ? [0, 0, 1] : [0, 1, 0];
  const sourceSide = cross3(sourceUp, sourceFlow);
  const targetSide = cross3(targetUp, targetFlow);
  const rotation = modelRotationDegrees();
  const center = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ];
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const rotated = rotatePointDegrees([x, y, z], rotation, center);
        const transformed = transformPoint(rotated, sourceFlow, sourceSide, sourceUp, targetFlow, targetSide, targetUp, scale);
        for (let axis = 0; axis < 3; axis += 1) {
          min[axis] = Math.min(min[axis], transformed[axis]);
          max[axis] = Math.max(max[axis], transformed[axis]);
        }
      }
    }
  }
  return {
    min,
    max,
    dimensions: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
  };
}

function transformPoint(point, sourceFlow, sourceSide, sourceUp, targetFlow, targetSide, targetUp, scale) {
  const flowComponent = dot3(point, sourceFlow) * scale;
  const sideComponent = dot3(point, sourceSide) * scale;
  const upComponent = dot3(point, sourceUp) * scale;
  return [
    flowComponent * targetFlow[0] + sideComponent * targetSide[0] + upComponent * targetUp[0],
    flowComponent * targetFlow[1] + sideComponent * targetSide[1] + upComponent * targetUp[1],
    flowComponent * targetFlow[2] + sideComponent * targetSide[2] + upComponent * targetUp[2],
  ];
}

function rotatePointDegrees(point, rotation, center = [0, 0, 0]) {
  let x = point[0] - center[0];
  let y = point[1] - center[1];
  let z = point[2] - center[2];
  const rx = Number(rotation.x || 0) * Math.PI / 180;
  const ry = Number(rotation.y || 0) * Math.PI / 180;
  const rz = Number(rotation.z || 0) * Math.PI / 180;

  [y, z] = [y * Math.cos(rx) - z * Math.sin(rx), y * Math.sin(rx) + z * Math.cos(rx)];
  [x, z] = [x * Math.cos(ry) + z * Math.sin(ry), -x * Math.sin(ry) + z * Math.cos(ry)];
  [x, y] = [x * Math.cos(rz) - y * Math.sin(rz), x * Math.sin(rz) + y * Math.cos(rz)];
  return [x + center[0], y + center[1], z + center[2]];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function renderWarnings() {
  const report = state.report;
  const warnings = report?.warnings?.length ? [...report.warnings] : ["No mesh warnings."];
  if (state.repair?.accepted) {
    warnings.unshift(
      `Prepared copy sealed the scan at ${fmt(state.repair.voxelSize)} source units per cell; source-surface p95 ${fmt(state.repair.sourceSurfaceDeviationP95Percent)}%, p99 ${fmt(state.repair.sourceSurfaceDeviationP99Percent)}%, size change ${fmt(state.repair.dimensionChangePercent)}%.`,
    );
    if (state.repair.warnings?.length) warnings.unshift(...state.repair.warnings);
  } else if (state.repair?.rejectionReasons?.length) {
    warnings.unshift(...state.repair.rejectionReasons);
  }
  els.warnings.innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
}

function renderCases() {
  if (!state.cases.length) {
    els.caseList.innerHTML = `<div class="case-item"><div><strong>No cases yet</strong><span>${state.root}</span></div><span></span></div>`;
    renderComparisonControls();
    return;
  }
  els.caseList.innerHTML = state.cases
    .slice(0, 8)
    .map((item) => {
      const progress = item.progress || fallbackCaseProgress(item.status);
      const percent = clamp(Number(progress.percent || 0), 0, 100);
      const percentageLabel = progress.state === "ready" ? "Not run" : `${Math.round(percent)}%`;
      const studyDetail = item.studyLevel
        ? `${item.studyLevel} accuracy study | ${item.path}`
        : item.sensitivityParameter
          ? `${item.sensitivityParameterLabel || item.sensitivityParameter} ${fmt(item.sensitivityValue)}${item.sensitivityUnit ? ` ${item.sensitivityUnit}` : ""}${item.sensitivityBaseline ? " | baseline" : ""} | ${item.path}`
          : item.path;
      return `
        <div class="case-item tone-${escapeAttr(progress.tone || "ready")}">
          <div>
            <strong>${escapeHtml(item.name)}</strong>
            <span>${escapeHtml(studyDetail)}</span>
          </div>
          <div class="case-state-block">
            <div class="case-state-line">
              <span class="case-state tone-${escapeAttr(progress.tone || "ready")}"><i aria-hidden="true"></i>${escapeHtml(progress.label || "Ready to run")}</span>
              <strong>${escapeHtml(percentageLabel)}</strong>
            </div>
            <progress class="case-progress-bar tone-${escapeAttr(progress.tone || "ready")}" max="100" value="${percent}" aria-label="${escapeAttr(item.name)} ${escapeAttr(progress.label || "Ready to run")}">${percent}%</progress>
          </div>
          <button class="small-action" type="button" data-case-path="${escapeAttr(item.path)}">
            ${item.path === state.activeCasePath ? "Selected" : "Use"}
          </button>
        </div>
      `;
    })
    .join("");
  renderComparisonControls();
}

function renderComparisonControls() {
  const paths = new Set(state.cases.map((item) => item.path));
  const previousBaseline = els.comparisonBaseline.value;
  const previousVariant = els.comparisonVariant.value;
  const options = state.cases
    .map((item) => `<option value="${escapeAttr(item.path)}">${escapeHtml(item.name)}</option>`)
    .join("");
  els.comparisonBaseline.innerHTML = options;
  els.comparisonVariant.innerHTML = options;
  if (paths.has(previousBaseline)) els.comparisonBaseline.value = previousBaseline;
  else if (state.cases[1]) els.comparisonBaseline.value = state.cases[1].path;
  if (paths.has(previousVariant)) els.comparisonVariant.value = previousVariant;
  else if (state.cases[0]) els.comparisonVariant.value = state.cases[0].path;
  const comparable = state.cases.length >= 2
    && els.comparisonBaseline.value
    && els.comparisonVariant.value
    && els.comparisonBaseline.value !== els.comparisonVariant.value;
  els.compareCasesButton.disabled = !comparable;

  const comparison = state.comparison;
  if (!comparison) {
    els.comparisonSummary.innerHTML = "";
    return;
  }
  const deltas = comparison.coefficientDeltas || {};
  const balance = comparison.balanceDeltas || {};
  const rows = [
    ["Cd", deltas.Cd],
    ["Cl", deltas.Cl],
    ["Cs", deltas.Cs],
    ["CmPitch", deltas.CmPitch],
    ["Front balance", balance.frontAeroBalancePercent, "%"],
  ].filter(([, value]) => value?.delta != null);
  const statistical = comparison.statisticalDeltas || {};
  const statisticalRows = [
    ["Cd", statistical.Cd],
    ["Cl", statistical.Cl],
    ["Cs", statistical.Cs],
    ["CmPitch", statistical.CmPitch],
    ["Front balance", statistical.frontAeroBalancePercent, "%"],
  ].filter(([, value]) => value?.delta != null);
  const mismatch = Array.isArray(comparison.setupDifferences)
    ? comparison.setupDifferences.slice(0, 4)
    : [];
  els.comparisonSummary.innerHTML = `
    <div class="comparison-status ${comparison.decisionSafe ? "pass" : "fail"}">
      ${escapeHtml(comparison.statusLabel || comparison.status || "Comparison")}
    </div>
    <div class="comparison-status ${comparison.statisticalDecisionSafe ? "pass" : "fail"}">
      ${escapeHtml(comparison.statisticalStatusLabel || comparison.statisticalStatus || "Statistical evidence pending")}
    </div>
    ${rows.map(([label, value, unit = ""]) => `
      <div class="comparison-delta">
        <span>${escapeHtml(label)}</span>
        <strong>${fmt(value.delta)}${unit} (${value.percentDelta == null ? "n/a" : `${fmt(value.percentDelta)}%`})</strong>
      </div>
    `).join("")}
    ${statisticalRows.map(([label, value, unit = ""]) => {
      const interval = value.confidenceLower != null && value.confidenceUpper != null
        ? `[${fmt(value.confidenceLower)}, ${fmt(value.confidenceUpper)}]${unit}`
        : "CI unavailable";
      const resolution = value.statisticallyResolved == null
        ? "resolution unavailable"
        : value.statisticallyResolved
          ? "interval excludes zero"
          : "interval includes zero";
      const gate = comparison.statisticalDecisionSafe ? "" : " | evidence gate pending";
      return `
        <div class="comparison-delta statistical-delta">
          <span>${escapeHtml(label)} 95% difference</span>
          <strong>${fmt(value.delta)}${unit} | ${escapeHtml(interval)} | ${escapeHtml(resolution + gate)} | N_eff ${fmt(value.baselineEffectiveSamples)} / ${fmt(value.variantEffectiveSamples)}</strong>
        </div>
      `;
    }).join("")}
    ${mismatch.length ? `<div class="comparison-mismatch"><strong>Lock mismatches</strong><br>${mismatch.map((item) => escapeHtml(item.field)).join("<br>")}</div>` : ""}
    <div class="comparison-mismatch">${escapeHtml(comparison.interpretation || "")}</div>
  `;
}

function fallbackCaseProgress(status) {
  if (status === "solver_verified") {
    return { state: "complete", tone: "verified", percent: 100, label: "Complete - qualified" };
  }
  if (status === "solver_unverified") {
    return { state: "complete", tone: "review", percent: 100, label: "Complete - review checks" };
  }
  if (status === "solver_failed") {
    return { state: "failed", tone: "failed", percent: 0, label: "Failed" };
  }
  if (status === "solver_running") {
    return { state: "running", tone: "running", percent: 1, label: "Starting - 1%" };
  }
  return { state: "ready", tone: "ready", percent: 0, label: "Ready to run" };
}

function renderRunProgress() {
  const progress = state.activeRunProgress;
  const visible = Boolean(state.activeCasePath && progress);
  els.runProgress.hidden = !visible;
  if (!visible) return;
  const percent = clamp(Number(progress.percent || 0), 0, 100);
  const tone = progress.tone || "ready";
  els.runProgress.className = `run-progress tone-${tone}`;
  els.runProgressLabel.textContent = progress.label || progress.phase || "Ready to run";
  els.runProgressPercent.textContent = progress.state === "ready" ? "Not run" : `${Math.round(percent)}%`;
  els.runProgressBar.value = percent;
  els.runProgressBar.textContent = `${Math.round(percent)}%`;
  els.runProgressDetail.textContent = progress.detail || "";
}

function applyRunProgress(casePath, progress) {
  const item = state.cases.find((candidate) => candidate.path === casePath);
  if (item) item.progress = progress;
  if (state.activeCasePath === casePath) {
    state.activeRunProgress = progress;
    renderRunProgress();
    if (progress.isRunning) {
      els.modelStatus.textContent = `${progress.phase} ${Math.round(progress.percent || 0)}% - current 3D view is preview`;
      els.candidateBadge.textContent = `${Math.round(progress.percent || 0)}%`;
      els.candidateBadge.className = "badge running";
    } else if (progress.isMeshComplete) {
      els.modelStatus.textContent = "Mesh validation complete - solver has not run";
    } else if (progress.isComplete) {
      els.modelStatus.textContent = "CFD complete - loading solved airflow";
    }
  }
  renderResultSummary();
  renderCases();
}

function startRunProgressPolling(casePath, maximumWaitMs) {
  stopRunProgressPolling();
  const token = state.runProgressToken + 1;
  state.runProgressToken = token;
  const deadline = Date.now() + maximumWaitMs;
  return new Promise((resolve, reject) => {
    const poll = async () => {
      if (state.runProgressToken !== token) return;
      try {
        const payload = await apiGet(`/api/case-progress?casePath=${encodeURIComponent(casePath)}`);
        if (state.runProgressToken !== token) return;
        const progress = payload.progress;
        applyRunProgress(casePath, progress);
        if (["complete", "mesh_complete", "failed"].includes(progress.state)) {
          state.runProgressTimer = null;
          resolve(progress);
          return;
        }
      } catch (_error) {
        // A transient polling miss should not interrupt a healthy local solver.
      }
      if (Date.now() >= deadline) {
        state.runProgressTimer = null;
        reject(new Error("AeroLab stopped waiting for run status; OpenFOAM may still be active in the background."));
        return;
      }
      if (state.runProgressToken === token) {
        state.runProgressTimer = window.setTimeout(poll, 1200);
      }
    };
    state.runProgressTimer = window.setTimeout(poll, 400);
  });
}

function stopRunProgressPolling() {
  state.runProgressToken += 1;
  if (state.runProgressTimer != null) {
    window.clearTimeout(state.runProgressTimer);
    state.runProgressTimer = null;
  }
}

async function refreshSolverStatus() {
  state.solver = await apiGet("/api/solver");
  renderSolverStatus();
}

async function refreshCaseReport(casePath) {
  if (!casePath) return;
  const requestedCasePath = casePath;
  const payload = await fetchJson("/api/case-report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ casePath }),
  });
  if (state.activeCasePath !== requestedCasePath) return;
  state.caseReport = payload.report;
  resetSolverParticles();
  state.viewer.flowLayer = null;
  state.activeRunProgress = payload.report.runProgress || null;
  state.viewer.surfaceMode = hasSurfacePressure() ? "cp" : "material";
  syncSurfaceModeControls();
  syncSolverFlowControls();
  restoreCaseContext(payload.report);
  if (payload.report.geometryPreview?.triangles?.length) {
    state.mesh = payload.report.geometryPreview;
    state.viewer.meshSource = "case";
    state.viewer.modelLayer = null;
    state.viewer.flowEnvelope = null;
    state.viewer.meshBounds = null;
    state.viewer.smoothVertexLight = null;
    state.viewer.flowLayer = null;
    els.modelName.textContent = payload.report.caseName || "OpenFOAM case";
    const progress = payload.report.runProgress;
    els.modelStatus.textContent = payload.report.solverStreamlines?.lines?.length
      ? "Solved OpenFOAM flow"
      : progress?.isRunning
        ? `${progress.phase} ${Math.round(progress.percent || 0)}% - current 3D view is preview`
        : progress?.state === "failed"
          ? "OpenFOAM run failed - preview only"
        : progress?.isMeshComplete
          ? "Mesh validated - preview flow until solver runs"
        : progress?.isComplete
            ? "OpenFOAM finished - solved visualization unavailable"
            : "Preview only - case created, solver not run";
    initFlowVisualization();
    renderMetrics();
    void loadExactStl(payload.report.geometryModelPath || payload.report.sourceModelPath);
    scheduleAeroFeatureScan(0);
  }
  renderResultSummary();
  renderRunProgress();
  renderReadiness();
  drawFlow();
}

function restoreCaseContext(report) {
  const geometry = report?.geometryReport;
  if (!geometry?.bounds?.dimensions) return;
  const setup = report.caseSetup || {};
  const orientation = setup.orientation || {};
  const flow = setup.flow || {};
  const ground = setup.ground || {};
  const quality = setup.quality || {};
  const units = setup.units || {};
  const reference = report.aerodynamicReference || {};
  const physical = report.physicalModel || setup.physicalModel || {};
  const fluid = physical.fluid || {};
  const inflow = physical.inflow || {};
  const surface = physical.surface || {};
  const domain = physical.domain || {};
  const outlet = physical.outlet || {};
  const roadAndWheels = physical.road_and_wheels || {};
  const transient = physical.transient || {};
  const turbulence = physical.turbulence || {};
  const volumeZones = physical.volume_zones || {};
  const datums = report.vehicleDatums || setup.vehicleDatums || {};

  state.report = geometry;
  state.repair = null;
  state.modelPath = report.sourceModelPath || state.modelPath;
  els.sourceFlowDirection.value = orientation.source_flow_direction || "+x";
  els.sourceUpDirection.value = orientation.source_up_direction || "+z";
  setModelRotation(orientation.rotation_degrees || {});
  els.flowAxis.value = orientation.target_flow_axis || flow.axis || "x";
  els.speedMph.value = flow.speed_mph ?? 70;
  els.airTemperatureC.value = fluid.temperature_c ?? flow.air_temperature_c ?? 15;
  els.airPressurePa.value = fluid.pressure_pa ?? flow.air_pressure_pa ?? 101325;
  els.airDensity.value = fluid.property_source === "manual_override"
    ? String(fluid.density_kg_m3 ?? "")
    : "";
  els.kinematicViscosity.value = fluid.property_source === "manual_override"
    ? String(fluid.kinematic_viscosity_m2_s ?? "")
    : "";
  els.turbulenceIntensity.value = inflow.turbulence_intensity_percent ?? 1;
  els.turbulenceLengthScale.value = inflow.turbulence_length_scale_source === "manual"
    ? String(inflow.turbulence_length_scale_m ?? "")
    : "";
  els.yawDegrees.value = String(inflow.yaw_degrees ?? "");
  els.crosswindMps.value = "";
  els.roughnessHeightMm.value = String(Math.max(0, Number(surface.roughness_height_m || 0)) * 1000);
  els.roughnessConstant.value = String(surface.roughness_constant ?? 0.5);
  els.backflowSafeOutlet.checked = Boolean(outlet.backflow_safe);
  els.secondOrderTransient.checked = Boolean(transient.second_order_temporal);
  els.fluidProfile.value = fluid.profile
    || (setup.solverModule === "fluid" ? "compressible_thermal" : "incompressible");
  els.turbulenceModel.value = turbulence.model || "kOmegaSST";
  const closedTunnel = domain.mode === "closed_tunnel" ? domain.closed_tunnel || {} : null;
  els.closedTunnel.checked = Boolean(closedTunnel);
  els.tunnelWidthM.value = closedTunnel?.width_m != null ? String(closedTunnel.width_m) : "";
  els.tunnelHeightM.value = closedTunnel?.height_m != null ? String(closedTunnel.height_m) : "";
  els.tunnelUpstreamM.value = closedTunnel?.upstream_m != null ? String(closedTunnel.upstream_m) : "";
  els.tunnelDownstreamM.value = closedTunnel?.downstream_m != null ? String(closedTunnel.downstream_m) : "";
  const wheels = Array.isArray(roadAndWheels.wheels)
    ? roadAndWheels.wheels.map((wheel) => ({
        name: wheel.name,
        model_path: wheel.model_path,
        center_source: wheel.source_center,
        axis_source: wheel.source_axis,
        radius_source: wheel.source_radius,
        surface_speed_mps: wheel.surface_speed_mps,
      }))
    : [];
  els.wheelSetupJson.value = wheels.length ? JSON.stringify(wheels, null, 2) : "";
  const porousZones = Array.isArray(volumeZones.porous_zones)
    ? volumeZones.porous_zones.map((zone) => ({
        name: zone.name,
        minimum_m: zone.minimum_m,
        maximum_m: zone.maximum_m,
        darcy_d_per_m2: zone.darcy_d_per_m2,
        forchheimer_f_per_m: zone.forchheimer_f_per_m,
      }))
    : [];
  const fanZones = Array.isArray(volumeZones.fan_zones)
    ? volumeZones.fan_zones.map((zone) => ({
        name: zone.name,
        minimum_m: zone.minimum_m,
        maximum_m: zone.maximum_m,
        disk_direction: zone.disk_direction,
        power_coefficient: zone.power_coefficient,
        thrust_coefficient: zone.thrust_coefficient,
        disk_area_m2: zone.disk_area_m2,
        upstream_point_m: zone.upstream_point_m,
      }))
    : [];
  const heatZones = Array.isArray(volumeZones.heat_zones)
    ? volumeZones.heat_zones.map((zone) => ({
        name: zone.name,
        shape: zone.shape || "box",
        component: zone.component,
        minimum_m: zone.minimum_m,
        maximum_m: zone.maximum_m,
        power_w: zone.power_w,
      }))
    : [];
  els.porousZonesJson.value = porousZones.length ? JSON.stringify(porousZones, null, 2) : "";
  els.fanZonesJson.value = fanZones.length ? JSON.stringify(fanZones, null, 2) : "";
  els.heatZonesJson.value = heatZones.length ? JSON.stringify(heatZones, null, 2) : "";
  els.includeGround.checked = Boolean(ground.enabled);
  els.movingGround.checked = Boolean(ground.moving);
  els.groundClearanceMm.value = String(Math.max(0, Number(ground.clearance_m || 0)) * 1000);
  syncGroundControls();
  if (quality.name) els.qualityPreset.value = quality.name;
  const simulationMode = String(quality.simulation_mode || setup.simulationType || "steady");
  els.simulationMode.value = simulationMode.startsWith("transient") ? "transient" : "steady";
  syncAdvancedFlowControls();
  const sensitivity = report.sensitivityStudy;
  if (sensitivity?.parameter) {
    els.sensitivityParameter.value = sensitivity.parameter;
    if (Array.isArray(sensitivity.values)) {
      els.sensitivityValues.value = sensitivity.values.join(", ");
    }
    els.sensitivityBaselineIndex.value = sensitivity.baselineIndex != null
      ? String(sensitivity.baselineIndex)
      : "";
    els.sensitivityStatus.textContent = `${sensitivity.parameterLabel || sensitivity.parameter}: ${String(sensitivity.status || "pending").replaceAll("_", " ")}`;
  } else {
    els.sensitivityStatus.textContent = "";
  }

  const unitScale = Number(units.scale_to_meters || 1);
  const solvedBounds = report.scaledGeometryReport?.bounds?.dimensions;
  const rawFlowLength = Number(geometry.bounds.dimensions[signedAxisIndex(els.sourceFlowDirection.value)] || 0);
  const physicalFlowLength = Number(solvedBounds?.[flowAxisIndex()] || rawFlowLength * unitScale);
  const measured = report.geometryValidation?.measured_dimensions_m || {};
  els.targetLength.value = measured.length_m != null
    ? String(measured.length_m)
    : String(units.input_units || "").startsWith("real length") && physicalFlowLength > 0
      ? String(physicalFlowLength)
      : "";
  els.targetWidth.value = measured.width_m != null ? String(measured.width_m) : "";
  els.targetHeight.value = measured.height_m != null ? String(measured.height_m) : "";
  els.referenceArea.value = reference.area_source === "manual" ? String(reference.area_m2 || "") : "";
  els.referenceLength.value = reference.length_source === "manual" ? String(reference.length_m || "") : "";
  const cg = datums.center_of_gravity_m || {};
  els.cgX.value = cg.x != null ? String(cg.x) : "";
  els.cgY.value = cg.y != null ? String(cg.y) : "";
  els.cgZ.value = cg.z != null ? String(cg.z) : "";
  els.frontAxleStation.value = datums.front_axle_station_m != null
    ? String(datums.front_axle_station_m)
    : "";
  els.rearAxleStation.value = datums.rear_axle_station_m != null
    ? String(datums.rear_axle_station_m)
    : "";
  const storedFeature = Number(report.meshResolution?.smallest_aero_feature_m || 0);
  els.smallestFeatureMm.value = storedFeature > 0 ? String(storedFeature * 1000) : "";
  els.caseName.value = report.caseName || basename(state.modelPath || "case");
  els.fileLabel.textContent = basename(state.modelPath || "case.stl");
  const fidelityBlocked = report.geometryFidelity?.verified === false;
  const geometryCandidate = geometry.is_cfd_candidate && !fidelityBlocked;
  const qualifiedCfd = Boolean(
    report.qualityAssessment?.numericallyQualified ?? report.qualityAssessment?.trusted,
  ) && geometryCandidate;
  const runProgress = report.runProgress;
  els.candidateBadge.textContent = runProgress?.isRunning
    ? `${Math.round(runProgress.percent || 0)}%`
    : qualifiedCfd
      ? "Qualified CFD"
      : runProgress?.isComplete
        ? "CFD Review"
        : fidelityBlocked
          ? "Repair Fidelity Missing"
          : geometryCandidate
            ? "Preview"
            : "Cleanup";
  els.candidateBadge.className = runProgress?.isRunning
    ? "badge running"
    : qualifiedCfd
      ? "badge"
      : runProgress?.isComplete || !geometryCandidate
        ? "badge warn"
        : "badge muted";
  els.createCaseButton.disabled = fidelityBlocked;
  els.createStudyButton.disabled = fidelityBlocked;
  els.createSensitivityButton.disabled = fidelityBlocked;
  els.prepareScanButton.disabled = true;
  renderMetrics();
  renderWarnings();
}

function renderSolverStatus() {
  const preferred = state.solver?.preferredBackend || "none";
  const version = preferred === "none" ? null : state.solver?.backends?.[preferred]?.version;
  const wslMessage = state.solver?.backends?.wsl?.message || "";
  const unavailable = preferred === "none" && wslMessage.toLowerCase().includes("not installed")
    ? "unavailable (WSL2 not installed)"
    : preferred;
  els.solverStatus.textContent = state.viewMode === "basic"
    ? preferred === "none"
      ? "Airflow solver unavailable. Open Engineering for setup details."
      : "Airflow solver ready."
    : `Solver: ${unavailable}${version ? ` (${version})` : ""}`;
  renderReadiness();
}

function updateActionAvailability() {
  const report = state.report;
  const fidelityBlocked = state.caseReport?.geometryFidelity?.verified === false
    || report?.repair_fidelity?.verified === false;
  const geometryCandidate = Boolean(report?.is_cfd_candidate) && !fidelityBlocked;
  // A loaded model can be prepared automatically before case creation.
  const usableModel = Boolean(report) && !fidelityBlocked;
  const dimensionStatus = geometryDimensionCheck(report).status;
  const caseGeometrySelected = state.viewer.meshSource === "case";
  els.sampleButton.disabled = state.busy;
  els.modelFile.disabled = state.busy;
  els.autoAlignButton.disabled = state.busy || !report?.alignment_suggestion?.recommended;
  els.autoAlignButton.title = report?.alignment_suggestion?.recommended
    ? `Apply ${report.alignment_suggestion.confidence || "geometry"} confidence principal-axis alignment`
    : "Automatic alignment needs a clearly vehicle-shaped model";
  els.createCaseButton.disabled = state.busy || !usableModel || dimensionStatus === "fail";
  els.createStudyButton.disabled = state.busy || !usableModel || dimensionStatus !== "pass";
  els.createSensitivityButton.disabled = state.busy || !usableModel || dimensionStatus === "fail";
  els.createCaseButton.textContent = state.viewMode === "basic"
    ? usableModel && !geometryCandidate
      ? "Prepare + Set Up Airflow"
      : "Set Up Airflow"
    : usableModel && !geometryCandidate
      ? "Prepare + Create Case"
      : "Create OpenFOAM Case";
  els.prepareScanButton.disabled = state.busy
    || !report
    || report.is_cfd_candidate
    || Boolean(report.repair_fidelity)
    || caseGeometrySelected;
  els.checkSolverButton.disabled = state.busy;
  els.meshCaseButton.disabled = state.busy || !state.activeCasePath || state.solver?.preferredBackend == null;
  els.runCaseButton.disabled = state.busy || !state.activeCasePath || state.solver?.preferredBackend == null;
}

function basicAirflowState() {
  const progress = state.activeRunProgress || state.caseReport?.runProgress;
  const solvedTracks = Number(state.caseReport?.solverStreamlines?.lineCount || 0);
  const solvedSurface = Boolean(state.caseReport?.surfacePressure?.hasPressure);
  const coefficients = state.caseReport?.forceCoeffs;
  const assessment = state.caseReport?.qualityAssessment;
  const qualified = Boolean(assessment?.numericallyQualified ?? assessment?.trusted);

  if (solvedTracks > 0) {
    return {
      tone: qualified ? "pass" : "warn",
      title: "Calculated airflow ready",
      short: "Calculated airflow is displayed.",
      detail: qualified
        ? "The moving paths come from the completed OpenFOAM result, and the run passed its numerical checks."
        : "The moving paths come from the completed OpenFOAM result. Review numerical checks in Engineering before using it for design decisions.",
    };
  }
  if (progress?.isRunning) {
    return {
      tone: "running",
      title: "Calculating airflow",
      short: "Airflow calculation is in progress.",
      detail: "The moving paths remain a visual preview until solved airflow tracks are available.",
    };
  }
  if (progress?.state === "failed") {
    return {
      tone: "warn",
      title: "Calculation did not finish",
      short: "Airflow calculation needs attention.",
      detail: "The viewer is showing preview paths, not solved airflow. Open Engineering for the run details.",
    };
  }
  if (progress?.isMeshComplete) {
    return {
      tone: "running",
      title: "Model preparation complete",
      short: "The model is ready for airflow calculation.",
      detail: "Choose Calculate Airflow to replace the visual preview with solver results.",
    };
  }
  if (progress?.isComplete || coefficients) {
    return {
      tone: "warn",
      title: solvedSurface ? "Surface result ready" : "Calculation finished",
      short: solvedSurface ? "Solved surface airflow is displayed." : "Solved airflow paths are unavailable.",
      detail: solvedSurface
        ? "The surface coloring is calculated, but the moving paths are still a preview because solved tracks were not exported."
        : "The current moving paths are a preview. Open Engineering to inspect the completed run and missing output.",
    };
  }
  if (state.activeCasePath) {
    return {
      tone: "running",
      title: "Ready to calculate",
      short: "Airflow setup is ready.",
      detail: "The moving paths are a visual preview. Choose Calculate Airflow for solver-generated paths.",
    };
  }
  if (state.report) {
    return {
      tone: "running",
      title: "Preview airflow",
      short: "Preview airflow is displayed.",
      detail: "These moving paths are a visual guide around the model, not CFD results. Set up and calculate airflow for solved tracks.",
    };
  }
  return {
    tone: "",
    title: "Load a model to begin",
    short: "",
    detail: "Choose an STL file or load the sample to see an airflow preview.",
  };
}

function renderBasicAirflowSummary(summary = basicAirflowState()) {
  els.basicAirflowSummary.className = `basic-airflow-summary${summary.tone ? ` ${summary.tone}` : ""}`;
  els.basicAirflowSummary.innerHTML = `
    <strong>${summary.title}</strong>
    <p>${summary.detail}</p>
  `;
}

function renderResultSummary() {
  const basicSummary = basicAirflowState();
  renderBasicAirflowSummary(basicSummary);
  if (state.viewMode === "basic") {
    els.resultSummary.textContent = basicSummary.short;
    return;
  }

  const coeffs = state.caseReport?.forceCoeffs;
  const forces = state.caseReport?.aerodynamicForces;
  const assessment = state.caseReport?.qualityAssessment;
  const temperature = state.caseReport?.temperatureResults;
  const temperatureSummary = temperature?.meanC != null
    ? ` | air T ${fmt(temperature.meanC)} °C mean, ${fmt(temperature.maximumC)} °C max${temperature.maximumRiseK == null ? "" : `, +${fmt(temperature.maximumRiseK)} K max`}`
    : "";
  if (!coeffs) {
    const progress = state.activeRunProgress || state.caseReport?.runProgress;
    const label = progress?.isRunning
      ? progress.label
      : progress?.state === "failed"
        ? "run failed - preview only"
        : progress?.isMeshComplete
          ? "mesh ready - solver not run"
        : progress?.isComplete
          ? "solver finished - coefficients missing"
          : "not run - preview only";
    els.resultSummary.textContent = state.activeCasePath ? `Results: ${label}${temperatureSummary}` : "";
    return;
  }
  const cd = coeffs.meanCd ?? coeffs.Cd ?? "n/a";
  const cl = coeffs.meanCl ?? coeffs.Cl ?? "n/a";
  const samples = coeffs.statistics?.sampleCount || 0;
  const qualified = assessment?.numericallyQualified ?? assessment?.trusted;
  const status = qualified ? "numerically qualified" : "qualification incomplete";
  const yPlus = state.caseReport?.yPlus?.body?.average;
  const tracks = state.caseReport?.solverStreamlines?.lineCount || 0;
  const grid = state.caseReport?.gridConvergence;
  const transientStatistics = state.caseReport?.transientStatistics;
  const statisticalOverall = transientStatistics?.overall_evidence;
  const statisticalChannels = transientStatistics?.channels || {};
  const cdStatistics = statisticalChannels.Cd;
  const meaningfulPeak = Object.entries(statisticalChannels)
    .map(([name, channel]) => ({ name, spectrum: channel?.spectrum }))
    .filter((item) => item.spectrum?.meaningful_peak)
    .sort((first, second) => Number(second.spectrum?.peak_power_fraction || 0) - Number(first.spectrum?.peak_power_fraction || 0))[0];
  const statisticalReady = statisticalOverall
    && statisticalOverall.stationarity_supported === true
    && statisticalOverall.minimum_effective_samples_30 === true
    && statisticalOverall.meaningful_peak_has_at_least_10_cycles !== false;
  const sensitivityStudy = state.caseReport?.sensitivityStudy;
  const verticalForce = forces?.verticalForceType && forces?.verticalForceN != null
    ? `${forces.verticalForceType} ${fmt(forces.verticalForceN)} N (${fmt(forces.verticalForceLbf)} lbf) @ ${fmt(forces.speedMph)} mph`
    : null;
  const balance = forces?.aeroBalance;
  const details = [
    `mean Cd ${fmt(cd)}`,
    `mean Cl ${fmt(cl)}`,
    coeffs.meanCs != null ? `mean Cs ${fmt(coeffs.meanCs)}` : null,
    verticalForce,
    forces?.dragN != null ? `drag ${fmt(forces.dragN)} N (${fmt(forces.dragLbf)} lbf)` : null,
    forces?.signedSideForceN != null ? `side ${fmt(forces.signedSideForceN)} N` : null,
    forces?.pitchMomentNm != null ? `pitch ${fmt(forces.pitchMomentNm)} N m` : null,
    balance?.qualified ? `front balance ${fmt(balance.frontAeroBalancePercent)}%` : "balance datum incomplete",
    temperature?.meanC != null
      ? `air T ${fmt(temperature.minimumC)} / ${fmt(temperature.meanC)} / ${fmt(temperature.maximumC)} °C min/mean/max${temperature.maximumRiseK == null ? "" : `; max rise ${fmt(temperature.maximumRiseK)} K`}`
      : null,
    `${samples} samples`,
    cdStatistics?.confidence_interval?.lower != null
      ? `Cd 95% CI [${fmt(cdStatistics.confidence_interval.lower)}, ${fmt(cdStatistics.confidence_interval.upper)}]`
      : null,
    cdStatistics?.effective_sample_count != null
      ? `Cd N_eff ${fmt(cdStatistics.effective_sample_count)}`
      : null,
    statisticalOverall
      ? `statistics ${statisticalReady ? "ready" : "pending"}; stationarity ${statisticalOverall.stationarity_supported === true ? "supported" : statisticalOverall.stationarity_supported === false ? "not supported" : "unresolved"}`
      : null,
    meaningfulPeak
      ? `${meaningfulPeak.name} peak ${fmt(meaningfulPeak.spectrum.dominant_frequency_hz)} Hz, ${fmt(meaningfulPeak.spectrum.cycle_coverage)} cycles${meaningfulPeak.spectrum.strouhal_number == null ? "" : `, St ${fmt(meaningfulPeak.spectrum.strouhal_number)}`}`
      : transientStatistics ? "no meaningful spectral peak" : null,
    sensitivityStudy
      ? `sensitivity ${String(sensitivityStudy.status || "pending").replaceAll("_", " ")}${sensitivityStudy.decisionSafeSensitivity ? " (decision-safe)" : ""}`
      : null,
    yPlus != null ? `avg y+ ${fmt(yPlus)}` : null,
    tracks ? `${tracks} solved tracks` : null,
    grid ? (grid.qualificationLabel || `mesh sensitivity ${grid.status}`) : "mesh-sensitivity study missing",
  ].filter(Boolean);
  els.resultSummary.textContent = `Results: ${status} | ${details.join(" | ")}`;
}

function drawFlow() {
  renderFlowScene(performance.now(), 0);
}

function startViewer() {
  if (state.viewer.running) return;
  state.viewer.running = true;
  state.viewer.lastTime = performance.now();
  requestAnimationFrame(tickViewer);
}

function tickViewer(time) {
  const dt = Math.min(0.04, Math.max(0.001, (time - state.viewer.lastTime) / 1000));
  state.viewer.lastTime = time;
  try {
    if (!document.hidden) renderFlowScene(time, dt);
  } catch (error) {
    disableThreeViewer(error);
  } finally {
    requestAnimationFrame(tickViewer);
  }
}

function initFlowVisualization() {
  const env = bodyEnvelope();
  initSmokeTrails(env);
}

function initSmokeTrails(env = bodyEnvelope()) {
  const lanes = [];
  const columns = [-0.58, -0.39, -0.2, 0, 0.2, 0.39, 0.58];
  const heights = [0.08, 0.28, 0.5, 0.72, 0.92];
  for (const yFactor of columns) {
    for (let heightIndex = 0; heightIndex < heights.length; heightIndex += 1) {
      const zFactor = heights[heightIndex];
      const laneMagnitude = Math.abs(yFactor);
      lanes.push({
        kind: "free",
        y: yFactor * env.width * 1.08,
        z: -0.56 + zFactor * Math.max(env.height + 0.72, 1.55),
        phase: heightIndex * 0.83 + laneMagnitude * 2.4,
        wobble: 0.006 + ((heightIndex + Math.round(laneMagnitude * 10)) % 3) * 0.002,
        alpha: 0.15 + (heightIndex % 2) * 0.025,
        width: 1.05 + ((heightIndex + Math.round(laneMagnitude * 10)) % 3) * 0.12,
      });
    }
  }

  const clearances = [1.03, 1.08, 1.16, 1.28];
  for (let angleStep = 0; angleStep < 24; angleStep += 1) {
    const angle = (angleStep / 24) * Math.PI * 2;
    for (let clearanceIndex = 0; clearanceIndex < clearances.length; clearanceIndex += 1) {
      const clearance = clearances[clearanceIndex];
      const phase = Math.sin(angle) * 0.9 + clearanceIndex * 1.17;
      const side = Math.cos(angle);
      lanes.push({
        kind: "surface",
        angle,
        clearance,
        vortexSign: side > 0.15 ? 1 : side < -0.15 ? -1 : 0,
        phase,
        wobble: 0.0015 + clearanceIndex * 0.0008,
        alpha: 0.17 + (1.32 - clearance) * 0.24,
        width: 0.72 + (1.32 - clearance) * 0.82,
      });
    }
  }
  state.viewer.smokeTrails = lanes;
}

function renderFlowScene(time, dt = 0) {
  const canvas = els.canvas;
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#0f1822");
  sky.addColorStop(0.58, "#162533");
  sky.addColorStop(1, "#101822");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  const camera = makeCamera(w, h);
  drawTunnelGrid(ctx, camera);
  if (hasSolverStreamlines()) {
    if (shouldDrawSolverLines()) drawSolverStreamlines(ctx, camera, time);
  } else {
    drawSmokeRibbons(ctx, camera, time);
  }
  drawModel(ctx, camera, dt);
  drawViewerHud(ctx, w, h);
  const surfaceLegendDrawn = drawPressureLegend(ctx, w, h);
  drawSpeedLegend(ctx, w, h, surfaceLegendDrawn ? 66 : 0);
  drawWindDirectionIndicator(ctx, camera, w, h);
}

function makeCamera(width, height) {
  return {
    width,
    height,
    centerX: width * 0.5,
    centerY: height * 0.55,
    unit: Math.min(width, height) * 0.145 * state.viewer.zoom,
    yaw: state.viewer.yaw,
    pitch: state.viewer.pitch,
    focal: 7.5,
  };
}

function project(point, camera) {
  const yawCos = Math.cos(camera.yaw);
  const yawSin = Math.sin(camera.yaw);
  const pitchCos = Math.cos(camera.pitch);
  const pitchSin = Math.sin(camera.pitch);
  const x1 = point.x * yawCos - point.y * yawSin;
  const y1 = point.x * yawSin + point.y * yawCos;
  const vertical = point.z * pitchCos - y1 * pitchSin;
  const z2 = y1 * pitchCos + point.z * pitchSin;
  const depth = z2 + 8;
  const perspective = camera.focal / Math.max(2.2, camera.focal + z2);
  return {
    x: camera.centerX + x1 * camera.unit * perspective,
    y: camera.centerY - vertical * camera.unit * perspective,
    scale: perspective,
    depth,
  };
}

function drawTunnelGrid(ctx, camera) {
  const env = bodyEnvelope();
  const halfLength = env.tunnelLength / 2;
  const halfWidth = env.tunnelWidth / 2;
  ctx.lineWidth = 1;
  for (let y = -halfWidth; y <= halfWidth; y += 1) {
    const start = project({ x: -halfLength, y, z: VIEWER_GROUND_Z }, camera);
    const end = project({ x: halfLength, y, z: VIEWER_GROUND_Z }, camera);
    ctx.strokeStyle = y === 0 ? "rgba(255,255,255,0.20)" : "rgba(140,174,190,0.15)";
    line(ctx, start, end);
  }

  for (let x = -halfLength; x <= halfLength; x += 1) {
    const start = project({ x, y: -halfWidth, z: VIEWER_GROUND_Z }, camera);
    const end = project({ x, y: halfWidth, z: VIEWER_GROUND_Z }, camera);
    ctx.strokeStyle = "rgba(140,174,190,0.12)";
    line(ctx, start, end);
  }

  if (els.includeGround.checked) {
    const corners = [
      project({ x: -halfLength, y: -halfWidth, z: VIEWER_GROUND_Z - 0.01 }, camera),
      project({ x: halfLength, y: -halfWidth, z: VIEWER_GROUND_Z - 0.01 }, camera),
      project({ x: halfLength, y: halfWidth, z: VIEWER_GROUND_Z - 0.01 }, camera),
      project({ x: -halfLength, y: halfWidth, z: VIEWER_GROUND_Z - 0.01 }, camera),
    ];
    ctx.fillStyle = "rgba(35, 48, 57, 0.45)";
    polygon(ctx, corners);
  }
}

function drawSmokeRibbons(ctx, camera, time) {
  if (!state.viewer.smokeTrails.length) initSmokeTrails();
  const env = bodyEnvelope();
  const speedMph = Number(els.speedMph.value || 70);
  const speedScale = clamp(speedMph / 70, 0.35, 2.4);
  const speedResponse = previewSpeedResponse();
  const halfLength = env.tunnelLength / 2;
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = `rgba(84,195,207,${0.2 + speedResponse.amount * 0.18})`;
  ctx.shadowBlur = 4 + speedResponse.amount * 6;

  for (const trail of state.viewer.smokeTrails) {
    ctx.beginPath();
    const steps = 132;
    let previousPoint = null;
    let drawing = false;
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const x = -halfLength + t * env.tunnelLength;
      const pulse = time * 0.00045 * speedScale + trail.phase + x * 0.38;
      const point = trail.kind === "surface"
        ? surfaceFlowPosition(trail, x, time)
        : flowPosition(
            {
              x,
              y: trail.y + Math.sin(pulse) * trail.wobble * env.width,
              z: trail.z + Math.cos(pulse * 0.7) * trail.wobble * env.height,
              phase: trail.phase,
            },
            time,
          );
      const projected = project(point, camera);
      const blocked =
        pointInsideObstacle(point, 0.045) ||
        (previousPoint && segmentIntersectsObstacle(previousPoint, point, 0.035));
      if (blocked || !drawing) {
        ctx.moveTo(projected.x, projected.y);
        drawing = !blocked;
      } else {
        ctx.lineTo(projected.x, projected.y);
      }
      previousPoint = point;
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = `rgba(88,180,192,${trail.alpha * (trail.kind === "surface" ? 0.45 : 0.3)})`;
    ctx.lineWidth = trail.width * (trail.kind === "surface" ? 3.4 : 4.2) * speedResponse.glow;
    ctx.stroke();

    ctx.strokeStyle = `rgba(86,203,214,${Math.min(0.58, trail.alpha * 1.35)})`;
    ctx.lineWidth = trail.width * (trail.kind === "surface" ? 1.05 : 0.9);
    ctx.stroke();

    ctx.setLineDash(trail.kind === "surface" ? [12, 25] : [16, 35]);
    ctx.lineDashOffset = -(time * 0.032 * speedScale + trail.phase * 18);
    ctx.strokeStyle = `rgba(75,226,235,${Math.min(0.74, trail.alpha * 1.85)})`;
    ctx.lineWidth = Math.max(1, trail.width * 0.75);
    ctx.stroke();
  }

  ctx.setLineDash([]);
  ctx.restore();
}

function drawSolverStreamlines(ctx, camera, time) {
  const flow = state.caseReport?.solverStreamlines;
  if (!flow?.lines?.length) return;
  const layer = solverFlowLayer(camera, flow);
  ctx.drawImage(layer.canvas, 0, 0);

  const speedMin = Number(flow.speedRange?.[0] ?? 0);
  const speedMax = Number(flow.speedRange?.[1] ?? speedMin);
  const speedSpan = Math.max(speedMax - speedMin, 1e-6);

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let pathIndex = 0; pathIndex < layer.projectedPaths.length; pathIndex += 1) {
    const projected = layer.projectedPaths[pathIndex];
    const meanSpeed = projected.reduce((sum, sample) => sum + sample.speed, 0) / Math.max(1, projected.length);
    const speedAmount = clamp((meanSpeed - speedMin) / speedSpan, 0, 1);
    ctx.beginPath();
    projected.forEach((sample, index) => {
      if (index === 0) ctx.moveTo(sample.point.x, sample.point.y);
      else ctx.lineTo(sample.point.x, sample.point.y);
    });
    ctx.setLineDash([5 + speedAmount * 4, 29 - speedAmount * 10]);
    ctx.lineDashOffset = -(time * (0.021 + speedAmount * 0.027) + pathIndex * 7);
    ctx.strokeStyle = "rgba(232,252,255,0.62)";
    ctx.lineWidth = 1.05 + speedAmount * 0.65;
    ctx.shadowColor = "rgba(112,222,232,0.38)";
    ctx.shadowBlur = 4 + speedAmount * 5;
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.shadowBlur = 0;
  ctx.restore();
}

function solverFlowCameraSample(point, camera, zOffset) {
  const x = Number(point?.[0]);
  const y = Number(point?.[1]);
  const z = Number(point?.[2]) + zOffset;
  const speed = Number(point?.[3] ?? 0);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  const yawCos = Math.cos(camera.yaw);
  const yawSin = Math.sin(camera.yaw);
  const pitchCos = Math.cos(camera.pitch);
  const pitchSin = Math.sin(camera.pitch);
  const cameraX = x * yawCos - y * yawSin;
  const yawDepth = x * yawSin + y * yawCos;
  const cameraY = z * pitchCos - yawDepth * pitchSin;
  const cameraDepth = camera.focal + yawDepth * pitchCos + z * pitchSin;
  return { x: cameraX, y: cameraY, depth: cameraDepth, speed: Number.isFinite(speed) ? speed : 0 };
}

function interpolateSolverFlowCameraSample(start, end, amount) {
  return {
    x: lerp(start.x, end.x, amount),
    y: lerp(start.y, end.y, amount),
    depth: lerp(start.depth, end.depth, amount),
    speed: lerp(start.speed, end.speed, amount),
  };
}

function clipSolverFlowSegment(start, end) {
  const depthDelta = end.depth - start.depth;
  let firstAmount = 0;
  let lastAmount = 1;
  if (Math.abs(depthDelta) <= 1e-12) {
    if (start.depth < VIEWER_CAMERA_NEAR || start.depth > VIEWER_CAMERA_FAR) return null;
  } else {
    const nearAmount = (VIEWER_CAMERA_NEAR - start.depth) / depthDelta;
    const farAmount = (VIEWER_CAMERA_FAR - start.depth) / depthDelta;
    firstAmount = Math.max(0, Math.min(nearAmount, farAmount));
    lastAmount = Math.min(1, Math.max(nearAmount, farAmount));
    if (firstAmount >= lastAmount) return null;
  }
  return [
    interpolateSolverFlowCameraSample(start, end, firstAmount),
    interpolateSolverFlowCameraSample(start, end, lastAmount),
  ];
}

function projectSolverFlowCameraSample(sample, camera) {
  const perspective = camera.focal / sample.depth;
  return {
    point: {
      x: camera.centerX + sample.x * camera.unit * perspective,
      y: camera.centerY - sample.y * camera.unit * perspective,
      scale: perspective,
      depth: sample.depth,
    },
    speed: sample.speed,
  };
}

function projectSolverFlowPaths(lines, camera, zOffset) {
  const projectedPaths = [];
  for (const path of lines || []) {
    let projected = [];
    let previousEnd = null;
    for (let index = 1; index < path.length; index += 1) {
      const start = solverFlowCameraSample(path[index - 1], camera, zOffset);
      const end = solverFlowCameraSample(path[index], camera, zOffset);
      const clipped = start && end ? clipSolverFlowSegment(start, end) : null;
      if (!clipped) {
        if (projected.length >= 2) projectedPaths.push(projected);
        projected = [];
        previousEnd = null;
        continue;
      }
      const [clippedStart, clippedEnd] = clipped;
      const continuous = previousEnd
        && Math.abs(clippedStart.x - previousEnd.x) <= 1e-7
        && Math.abs(clippedStart.y - previousEnd.y) <= 1e-7
        && Math.abs(clippedStart.depth - previousEnd.depth) <= 1e-7;
      if (!continuous) {
        if (projected.length >= 2) projectedPaths.push(projected);
        projected = [projectSolverFlowCameraSample(clippedStart, camera)];
      }
      projected.push(projectSolverFlowCameraSample(clippedEnd, camera));
      previousEnd = clippedEnd;
    }
    if (projected.length >= 2) projectedPaths.push(projected);
  }
  return projectedPaths;
}

function solverFlowLayer(camera, flow) {
  const key = [
    flow.file || "solved-flow",
    flow.pointCount || 0,
    flow.speedRange?.join(":") || "no-speed",
    Math.round(camera.width),
    Math.round(camera.height),
    camera.yaw.toFixed(4),
    camera.pitch.toFixed(4),
    camera.unit.toFixed(2),
    meshGroundOffset().toFixed(6),
  ].join("|");
  if (state.viewer.flowLayer?.key === key) return state.viewer.flowLayer;

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.ceil(camera.width));
  canvas.height = Math.max(1, Math.ceil(camera.height));
  const layerCtx = canvas.getContext("2d");
  const speedMin = Number(flow.speedRange?.[0] ?? 0);
  const speedMax = Number(flow.speedRange?.[1] ?? speedMin);
  const speedSpan = Math.max(speedMax - speedMin, 1e-6);
  const zOffset = meshGroundOffset();
  const projectedPaths = projectSolverFlowPaths(flow.lines, camera, zOffset);

  layerCtx.save();
  layerCtx.globalCompositeOperation = "screen";
  layerCtx.lineCap = "round";
  layerCtx.lineJoin = "round";
  for (const projected of projectedPaths) {
    for (let index = 1; index < projected.length; index += 1) {
      const previous = projected[index - 1];
      const current = projected[index];
      const speedAmount = clamp((current.speed - speedMin) / speedSpan, 0, 1);
      layerCtx.strokeStyle = speedColor(current.speed, speedMin, speedMax, 0.12 + speedAmount * 0.08);
      layerCtx.lineWidth = Math.max(3.2, (5 + speedAmount * 3.8) * current.point.scale);
      line(layerCtx, previous.point, current.point);
      layerCtx.strokeStyle = speedColor(current.speed, speedMin, speedMax, 0.66);
      layerCtx.lineWidth = Math.max(0.9, (1.25 + speedAmount * 1.3) * current.point.scale);
      line(layerCtx, previous.point, current.point);
    }
  }
  layerCtx.restore();
  state.viewer.flowLayer = { key, canvas, projectedPaths };
  return state.viewer.flowLayer;
}

function pressureColorChannels(value, minimum, maximum) {
  const limit = Math.max(Math.abs(minimum), Math.abs(maximum), 1e-6);
  const amount = clamp(value / (limit * 2) + 0.5, 0, 1);
  const low = [43, 120, 218];
  const middle = [91, 184, 194];
  const high = [226, 72, 56];
  const from = amount < 0.5 ? low : middle;
  const to = amount < 0.5 ? middle : high;
  const blend = amount < 0.5 ? amount * 2 : (amount - 0.5) * 2;
  return from.map((channel, index) => Math.round(lerp(channel, to[index], blend)));
}

function pressureColor(value, minimum, maximum, alpha = 1) {
  const color = pressureColorChannels(value, minimum, maximum);
  return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
}

function speedColorChannels(value, minimum, maximum) {
  const span = maximum - minimum;
  const amount = span > 1e-9 ? clamp((value - minimum) / span, 0, 1) : 0.5;
  const scaled = amount * (SPEED_COLOR_STOPS.length - 1);
  const index = Math.min(SPEED_COLOR_STOPS.length - 2, Math.floor(scaled));
  const blend = scaled - index;
  return SPEED_COLOR_STOPS[index].map(
    (channel, channelIndex) => Math.round(lerp(channel, SPEED_COLOR_STOPS[index + 1][channelIndex], blend)),
  );
}

function speedColor(value, minimum, maximum, alpha = 1) {
  const color = speedColorChannels(value, minimum, maximum);
  return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
}

function temperatureColorChannels(value, minimum, maximum) {
  const span = Math.max(maximum - minimum, 1e-9);
  const amount = clamp((value - minimum) / span, 0, 1);
  const stops = [
    [43, 76, 181],
    [43, 190, 196],
    [242, 204, 66],
    [220, 58, 47],
  ];
  const scaled = amount * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const blend = scaled - index;
  return stops[index].map(
    (channel, channelIndex) => Math.round(lerp(channel, stops[index + 1][channelIndex], blend)),
  );
}

function dragColorChannels(value, minimum, maximum) {
  const negativeLimit = Math.max(Math.abs(Math.min(minimum, 0)), 1e-6);
  const positiveLimit = Math.max(Math.max(maximum, 0), 1e-6);
  const low = [42, 111, 205];
  const neutral = [119, 134, 141];
  const high = [232, 69, 48];
  if (value < 0) {
    const amount = clamp(Math.abs(value) / negativeLimit, 0, 1);
    return neutral.map((channel, index) => Math.round(lerp(channel, low[index], amount)));
  }
  const amount = clamp(value / positiveLimit, 0, 1);
  return neutral.map((channel, index) => Math.round(lerp(channel, high[index], amount)));
}

function drawPressureLegend(ctx, width, height) {
  const basic = state.viewMode === "basic";
  const showTemperature = state.viewer.surfaceMode === "temperature" && hasSurfaceTemperature();
  const showDrag = state.viewer.surfaceMode === "drag" && hasSurfaceDrag();
  const surfacePressure = (
    state.viewer.surfaceMode === "cp" || showTemperature || showDrag
  ) && hasSurfacePressure()
    ? state.caseReport.surfacePressure
    : null;
  if (!surfacePressure) return false;
  const range = showTemperature
    ? surfacePressure.temperatureDisplayRangeK || surfacePressure.temperatureKRange
    : showDrag
      ? hasSurfaceWallShear()
        ? surfacePressure.totalDragDisplayRange || surfacePressure.totalDragDensityRange
        : surfacePressure.pressureDragDisplayRange || surfacePressure.pressureDragDensityRange
      : surfacePressure.cpDisplayRange || surfacePressure.cpRange;
  const rangeMin = Number(range?.[0] ?? 0);
  const rangeMax = Number(range?.[1] ?? 0);
  const limit = Math.max(Math.abs(rangeMin), Math.abs(rangeMax), 1e-6);
  const panelWidth = Math.min(250, width - 28);
  const x = 14;
  const y = Math.max(104, height - 72);
  const barX = x + 14;
  const barY = y + 24;
  const barWidth = panelWidth - 28;

  ctx.save();
  ctx.fillStyle = "rgba(15,24,34,0.78)";
  roundRectPath(ctx, x, y, panelWidth, 58, 8);
  ctx.fill();
  for (let pixel = 0; pixel < barWidth; pixel += 1) {
    const amount = pixel / Math.max(1, barWidth - 1);
    const value = showTemperature
      ? rangeMin + amount * (rangeMax - rangeMin)
      : -limit + amount * limit * 2;
    const color = showTemperature
      ? temperatureColorChannels(value, rangeMin, rangeMax)
      : showDrag
        ? dragColorChannels(value, -limit, limit)
        : pressureColorChannels(value, -limit, limit);
    ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
    ctx.fillRect(barX + pixel, barY, 1.2, 8);
  }
  ctx.fillStyle = "#dfeaf1";
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(
    showTemperature
      ? "Adjacent-air temperature"
      : basic
        ? showDrag ? "Relative drag areas" : "Relative pressure"
        : showDrag
          ? hasSurfaceWallShear() ? "Local total drag areas" : "Local pressure-drag areas"
          : "Pressure coefficient Cp",
    x + panelWidth / 2,
    y + 16,
  );
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(
    showTemperature ? `${fmt(rangeMin - 273.15)} °C` : basic ? "Low" : showDrag ? "Offset" : fmt(-limit),
    barX,
    y + 47,
  );
  ctx.textAlign = "center";
  ctx.fillText(
    showTemperature ? `${fmt((rangeMin + rangeMax) / 2 - 273.15)} °C` : basic ? "Neutral" : showDrag ? "Neutral" : "0",
    x + panelWidth / 2,
    y + 47,
  );
  ctx.textAlign = "right";
  ctx.fillText(
    showTemperature ? `${fmt(rangeMax - 273.15)} °C` : basic ? "High" : showDrag ? "High drag" : fmt(limit),
    x + panelWidth - 14,
    y + 47,
  );
  ctx.restore();
  return true;
}

function drawSpeedLegend(ctx, width, height, verticalOffset = 0) {
  if (!hasSolverStreamlines()) return false;
  const flow = state.caseReport.solverStreamlines;
  const rangeMin = Number(flow.speedRange?.[0] ?? 0);
  const rangeMax = Number(flow.speedRange?.[1] ?? rangeMin);
  const panelWidth = Math.min(250, width - 28);
  const x = 14;
  const y = Math.max(104, height - 72 - verticalOffset);
  const barX = x + 14;
  const barY = y + 24;
  const barWidth = panelWidth - 28;

  ctx.save();
  ctx.fillStyle = "rgba(15,24,34,0.78)";
  roundRectPath(ctx, x, y, panelWidth, 58, 8);
  ctx.fill();
  for (let pixel = 0; pixel < barWidth; pixel += 1) {
    const amount = pixel / Math.max(1, barWidth - 1);
    const value = rangeMin + amount * (rangeMax - rangeMin);
    const color = speedColorChannels(value, rangeMin, rangeMax);
    ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
    ctx.fillRect(barX + pixel, barY, 1.2, 8);
  }
  ctx.fillStyle = "#dfeaf1";
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(flow.timeAveraged ? "Mean-flow speed" : "Final-field speed", x + panelWidth / 2, y + 16);
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${fmt(rangeMin)} m/s`, barX, y + 47);
  ctx.textAlign = "center";
  ctx.fillText(`${fmt((rangeMin + rangeMax) / 2)} m/s`, x + panelWidth / 2, y + 47);
  ctx.textAlign = "right";
  ctx.fillText(`${fmt(rangeMax)} m/s`, x + panelWidth - 14, y + 47);
  ctx.restore();
  return true;
}

function drawWindDirectionIndicator(ctx, camera, width, height) {
  const projectedOrigin = project({ x: 0, y: 0, z: 0 }, camera);
  const projectedTip = project({ x: 1, y: 0, z: 0 }, camera);
  let dx = projectedTip.x - projectedOrigin.x;
  let dy = projectedTip.y - projectedOrigin.y;
  const magnitude = Math.hypot(dx, dy);
  if (magnitude < 1e-6) {
    dx = 1;
    dy = 0;
  } else {
    dx /= magnitude;
    dy /= magnitude;
  }

  const axis = state.caseReport?.caseSetup?.flow?.axis || els.flowAxis.value || "x";
  const centerX = width - 62;
  const centerY = height - 35;
  const halfLength = 25;
  const startX = centerX - dx * halfLength;
  const startY = centerY - dy * halfLength;
  const endX = centerX + dx * halfLength;
  const endY = centerY + dy * halfLength;
  const angle = Math.atan2(dy, dx);

  ctx.save();
  ctx.strokeStyle = "rgba(128,229,237,0.96)";
  ctx.fillStyle = "rgba(128,229,237,0.96)";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.shadowColor = "rgba(71,199,211,0.65)";
  ctx.shadowBlur = 7;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX - Math.cos(angle - 0.55) * 11, endY - Math.sin(angle - 0.55) * 11);
  ctx.lineTo(endX - Math.cos(angle + 0.55) * 11, endY - Math.sin(angle + 0.55) * 11);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#dfeaf1";
  ctx.font = "700 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`WIND +${String(axis).toUpperCase()}`, centerX, centerY - 20);
  ctx.restore();
}

function flowPosition(particle, time) {
  const x = particle.x;
  let y = particle.y;
  let z = particle.z;
  ({ y, z } = deflectAroundObstacle({ x, y, z, phase: particle.phase || 0 }, time));
  const envelope = modelFlowEnvelope();
  if (x > envelope.maxX) {
    const speedResponse = previewSpeedResponse();
    const distance = x - envelope.maxX;
    const wake = Math.exp(-distance / Math.max(1.2, envelope.length * 0.72));
    const laneInfluence = Math.exp(
      -Math.pow((particle.y - envelope.centerY) / Math.max(0.25, envelope.maxRadiusY * 1.25), 2)
      -Math.pow((particle.z - envelope.centerZ) / Math.max(0.25, envelope.maxRadiusZ * 1.25), 2),
    );
    const pulse = time * 0.0016 * speedResponse.frequency + (particle.phase || 0) + distance * 1.15;
    const sideSign = Math.sign(particle.y - envelope.centerY);
    y += sideSign * Math.sin(pulse) * wake * laneInfluence * envelope.maxRadiusY
      * (0.035 + speedResponse.amount * 0.075);
    z += Math.cos(pulse * 0.82) * wake * laneInfluence * envelope.maxRadiusZ
      * (0.045 + speedResponse.amount * 0.075);
  }
  return { x, y, z };
}

function surfaceFlowPosition(trail, x, time) {
  const envelope = modelFlowEnvelope();
  const speedResponse = previewSpeedResponse();
  const angle = trail.angle || 0;
  const side = Math.cos(angle);
  const lateralSign = Math.abs(side) < 0.01 ? 0 : Math.sign(side);
  const clearance = trail.clearance || 1.18;
  const approachLength = Math.max(0.9, envelope.length * 0.82);
  const nose = surfaceGuideSection(envelope, envelope.minX);
  const shoulder = sampleEnvelopeSection(envelope, envelope.minX + envelope.length * 0.1) || envelope.front;

  if (x < envelope.minX) {
    const influence = smoothstep((x - (envelope.minX - approachLength)) / approachLength);
    const farY = shoulder.centerY + Math.cos(angle) * shoulder.radiusY * clearance;
    const farZ = shoulder.centerZ + Math.sin(angle) * shoulder.radiusZ * clearance;
    const nearY = nose.centerY + Math.cos(angle) * nose.radiusY * clearance;
    const nearZ = nose.centerZ + Math.sin(angle) * nose.radiusZ * clearance;
    return {
      x,
      y: lerp(farY, nearY, influence),
      z: lerp(farZ, nearZ, influence),
    };
  }

  if (x <= envelope.maxX) {
    const section = surfaceGuideSection(envelope, x);
    const pulse = time * 0.00055 * speedResponse.frequency + trail.phase + x * 0.42;
    return {
      x,
      y: section.centerY + side * section.radiusY * clearance + lateralSign * Math.sin(pulse) * trail.wobble,
      z: section.centerZ + Math.sin(angle) * section.radiusZ * clearance + Math.cos(pulse * 0.78) * trail.wobble,
    };
  }

  const tail = surfaceGuideSection(envelope, envelope.maxX);
  const distance = x - envelope.maxX;
  const normalizedDistance = distance / Math.max(envelope.length, 0.25);
  const recovery = 1 - Math.exp(-normalizedDistance / 1.2);
  const rotation = (trail.vortexSign || 0) * recovery * Math.PI
    * (0.22 + speedResponse.amount * 0.16);
  const wakeAngle = angle + rotation;
  const wakeScale = clearance * (1 + recovery * (0.12 + speedResponse.amount * 0.2));
  const meander = Math.sin(
    time * 0.00055 * speedResponse.frequency + trail.phase + normalizedDistance * 2.1,
  )
    * Math.exp(-normalizedDistance / 1.8);
  const wakeFlutter = Math.sin(
    time * 0.00125 * speedResponse.frequency + trail.phase * 1.9 + normalizedDistance * 5.2,
  ) * Math.exp(-normalizedDistance / 1.3);
  return {
    x,
    y: tail.centerY + Math.cos(wakeAngle) * tail.radiusY * wakeScale
      + lateralSign * meander * envelope.maxRadiusY * (0.008 + speedResponse.amount * 0.026)
      + lateralSign * wakeFlutter * envelope.maxRadiusY * speedResponse.amount * 0.012,
    z: tail.centerZ + Math.sin(wakeAngle) * tail.radiusZ * wakeScale
      + meander * envelope.maxRadiusZ * (0.007 + speedResponse.amount * 0.023)
      + wakeFlutter * envelope.maxRadiusZ * speedResponse.amount * 0.01,
  };
}

function previewSpeedResponse() {
  const speedMph = clamp(Number(els.speedMph.value || 70), 10, 160);
  const amount = smoothstep(clamp((speedMph - 20) / 120, 0, 1));
  return {
    amount,
    frequency: 0.62 + amount * 1.18,
    glow: 0.88 + amount * 0.34,
  };
}

function surfaceGuideSection(envelope, x) {
  const section = sampleEnvelopeSection(envelope, clamp(x, envelope.minX, envelope.maxX)) || envelope.front;
  const wakeBlend = smoothstep(
    (x - (envelope.maxX - envelope.length * 0.12)) / Math.max(envelope.length * 0.12, 0.01),
  );
  if (wakeBlend <= 0) return section;
  return {
    centerY: lerp(section.centerY, envelope.tail.centerY, wakeBlend),
    centerZ: lerp(section.centerZ, envelope.tail.centerZ, wakeBlend),
    radiusY: lerp(section.radiusY, envelope.tail.radiusY, wakeBlend),
    radiusZ: lerp(section.radiusZ, envelope.tail.radiusZ, wakeBlend),
  };
}

function deflectAroundObstacle(point, time) {
  const section = flowCrossSectionAt(point.x);
  if (!section || section.influence <= 0) return { y: point.y, z: point.z };

  const radiusY = Math.max(0.08, section.radiusY);
  const radiusZ = Math.max(0.08, section.radiusZ);
  const offsetY = point.y - section.centerY;
  const offsetZ = point.z - section.centerZ;
  const normalizedY = offsetY / radiusY;
  const normalizedZ = offsetZ / radiusZ;
  const radial = Math.sqrt(normalizedY * normalizedY + normalizedZ * normalizedZ);
  const guard = 1.18;
  if (radial >= guard) return { y: point.y, z: point.z };

  const phase = point.phase || 0;
  const sideSign = offsetY === 0 ? (Math.sin(phase) >= 0 ? 1 : -1) : Math.sign(offsetY);
  const verticalSign = offsetZ >= 0 || els.includeGround.checked ? 1 : -1;
  const sideBias = Math.abs(normalizedY);
  const verticalBias = Math.abs(normalizedZ) * 0.88;
  const routeOver = verticalBias >= sideBias || Math.abs(normalizedY) < 0.24;
  const clearance = 1.12 + section.influence * 0.08;
  const penetration = smoothstep(clamp((guard - radial) / guard, 0, 1));
  const blend = clamp(section.influence * (0.72 + penetration * 0.34), 0, 1);

  if (routeOver) {
    const targetZ = section.centerZ + verticalSign * radiusZ * clearance;
    return {
      y: lerp(point.y, point.y + sideSign * radiusY * 0.06, blend * 0.35),
      z: lerp(point.z, targetZ + Math.sin(time * 0.0007 + phase) * 0.008, blend),
    };
  }

  const targetY = section.centerY + sideSign * radiusY * clearance;
  return {
    y: lerp(point.y, targetY, blend),
    z: lerp(point.z, point.z + Math.max(0, radiusZ * 0.035), blend * 0.28),
  };
}

function drawModel(ctx, camera, dt = 0) {
  if (state.mesh?.triangles?.length) {
    if (renderThreeStlModel(camera, dt)) return;
    drawCachedStlMesh(ctx, camera);
    return;
  }
  clearThreeViewer();
  state.viewer.modelLayer = null;
  drawPreviewVehicle(ctx, camera);
}

function prepareSolverParticles(flow) {
  resetSolverParticles();
  const speedMin = Number(flow.speedRange?.[0] ?? 0);
  const speedMax = Number(flow.speedRange?.[1] ?? speedMin);
  const uniformSpeed = !Number.isFinite(speedMin)
    || !Number.isFinite(speedMax)
    || speedMax - speedMin <= 1e-9
    || speedMax <= 1e-9;
  const speedReference = Math.max(Number.isFinite(speedMax) ? speedMax : 0, 1e-9);
  const lines = [];

  for (const rawLine of flow.lines || []) {
    const coordinates = [];
    const speeds = [];
    for (const sample of rawLine || []) {
      const x = Number(sample?.[0]);
      const y = Number(sample?.[1]);
      const z = Number(sample?.[2]);
      const speed = Number(sample?.[3]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      if (coordinates.length) {
        const previous = coordinates.length - 3;
        if (Math.hypot(x - coordinates[previous], y - coordinates[previous + 1], z - coordinates[previous + 2]) <= 1e-8) {
          speeds[speeds.length - 1] = Number.isFinite(speed) ? Math.max(0, speed) : 0;
          continue;
        }
      }
      coordinates.push(x, y, z);
      speeds.push(Number.isFinite(speed) ? Math.max(0, speed) : 0);
    }
    if (speeds.length < 2) continue;

    const travelTimes = new Float32Array(speeds.length);
    let totalTravelTime = 0;
    for (let index = 1; index < speeds.length; index += 1) {
      const previousOffset = (index - 1) * 3;
      const currentOffset = index * 3;
      const distance = Math.hypot(
        coordinates[currentOffset] - coordinates[previousOffset],
        coordinates[currentOffset + 1] - coordinates[previousOffset + 1],
        coordinates[currentOffset + 2] - coordinates[previousOffset + 2],
      );
      const meanSpeed = (speeds[index - 1] + speeds[index]) * 0.5;
      const speedRatio = uniformSpeed
        ? 0.55
        : clamp(meanSpeed / speedReference, SOLVER_PARTICLE_SPEED_FLOOR, 1);
      totalTravelTime += distance / speedRatio;
      travelTimes[index] = totalTravelTime;
    }
    if (!Number.isFinite(totalTravelTime) || totalTravelTime <= 1e-8) continue;
    lines.push({
      coordinates: new Float32Array(coordinates),
      speeds: new Float32Array(speeds),
      travelTimes,
      totalTravelTime,
    });
  }

  if (!lines.length) {
    state.viewer.solverParticles = { source: flow, lines: [], points: null };
    return;
  }
  const targetCount = Math.min(SOLVER_PARTICLE_MAX_COUNT, lines.length * SOLVER_PARTICLE_TARGET_PER_LINE);
  const totalTravelTime = lines.reduce((sum, line) => sum + line.totalTravelTime, 0);
  const counts = [];
  let allocated = 0;
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const remainingLines = lines.length - lineIndex - 1;
    const remainingCapacity = targetCount - allocated - remainingLines * 2;
    const weightedCount = Math.round(targetCount * lines[lineIndex].totalTravelTime / totalTravelTime);
    const count = lineIndex === lines.length - 1
      ? targetCount - allocated
      : clamp(weightedCount, 2, remainingCapacity);
    counts.push(count);
    allocated += count;
  }

  const lineIndices = new Uint16Array(targetCount);
  const travelPositions = new Float32Array(targetCount);
  const segmentIndices = new Uint16Array(targetCount);
  let particleIndex = 0;
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const count = counts[lineIndex];
    const stagger = (lineIndex * 0.38196601125) % 1;
    for (let localIndex = 0; localIndex < count; localIndex += 1) {
      const phase = ((localIndex + 0.5) / count + stagger) % 1;
      const travel = phase * line.totalTravelTime;
      let segment = 0;
      while (segment < line.travelTimes.length - 2 && travel >= line.travelTimes[segment + 1]) segment += 1;
      lineIndices[particleIndex] = lineIndex;
      travelPositions[particleIndex] = travel;
      segmentIndices[particleIndex] = segment;
      particleIndex += 1;
    }
  }

  state.viewer.solverParticles = {
    source: flow,
    lines,
    lineIndices,
    travelPositions,
    segmentIndices,
    positions: new Float32Array(targetCount * 3),
    colors: new Float32Array(targetCount * 3),
    speedMin: Number.isFinite(speedMin) ? speedMin : 0,
    speedMax: Number.isFinite(speedMax) ? speedMax : 0,
    points: null,
  };
}

function createSolverParticleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  const glow = ctx.createRadialGradient(32, 32, 0, 32, 32, 31);
  glow.addColorStop(0, "rgba(255,255,255,1)");
  glow.addColorStop(0.32, "rgba(255,255,255,0.96)");
  glow.addColorStop(0.72, "rgba(255,255,255,0.42)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function ensureSolverParticlePoints(view, particles) {
  if (particles.points) return;
  const geometry = new THREE.BufferGeometry();
  const positions = new THREE.BufferAttribute(particles.positions, 3);
  const colors = new THREE.BufferAttribute(particles.colors, 3);
  positions.setUsage(THREE.DynamicDrawUsage);
  colors.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("position", positions);
  geometry.setAttribute("color", colors);
  const material = new THREE.PointsMaterial({
    alphaTest: 0.02,
    blending: THREE.AdditiveBlending,
    depthTest: true,
    depthWrite: false,
    map: createSolverParticleTexture(),
    opacity: 0.92,
    size: 0.052,
    sizeAttenuation: true,
    toneMapped: false,
    transparent: true,
    vertexColors: true,
  });
  particles.points = new THREE.Points(geometry, material);
  particles.points.frustumCulled = false;
  particles.points.renderOrder = 2;
  view.group.add(particles.points);
}

function writeSolverParticleColor(colors, offset, value, minimum, maximum) {
  const span = maximum - minimum;
  const amount = span > 1e-9 ? clamp((value - minimum) / span, 0, 1) : 0.5;
  const scaled = amount * (SPEED_COLOR_STOPS_LINEAR.length - 1);
  const index = Math.min(SPEED_COLOR_STOPS_LINEAR.length - 2, Math.floor(scaled));
  const blend = scaled - index;
  const from = SPEED_COLOR_STOPS_LINEAR[index];
  const to = SPEED_COLOR_STOPS_LINEAR[index + 1];
  colors[offset] = lerp(from[0], to[0], blend);
  colors[offset + 1] = lerp(from[1], to[1], blend);
  colors[offset + 2] = lerp(from[2], to[2], blend);
}

function updateSolverParticles(view, dt) {
  const flow = state.caseReport?.solverStreamlines;
  if (!shouldDrawSolverParticles()) {
    if (state.viewer.solverParticles?.points) state.viewer.solverParticles.points.visible = false;
    return;
  }
  if (state.viewer.solverParticles?.source !== flow) prepareSolverParticles(flow);
  const particles = state.viewer.solverParticles;
  if (!particles?.lines?.length) return;
  ensureSolverParticlePoints(view, particles);
  particles.points.visible = true;

  const advance = Math.max(0, Number(dt) || 0) * SOLVER_PARTICLE_ANIMATION_RATE;
  const zOffset = meshGroundOffset();
  for (let particleIndex = 0; particleIndex < particles.lineIndices.length; particleIndex += 1) {
    const line = particles.lines[particles.lineIndices[particleIndex]];
    let travel = particles.travelPositions[particleIndex] + advance;
    let segment = particles.segmentIndices[particleIndex];
    if (travel >= line.totalTravelTime) {
      travel %= line.totalTravelTime;
      segment = 0;
    }
    while (segment < line.travelTimes.length - 2 && travel >= line.travelTimes[segment + 1]) segment += 1;
    while (segment > 0 && travel < line.travelTimes[segment]) segment -= 1;
    particles.travelPositions[particleIndex] = travel;
    particles.segmentIndices[particleIndex] = segment;

    const segmentStart = line.travelTimes[segment];
    const segmentEnd = line.travelTimes[segment + 1];
    const amount = clamp((travel - segmentStart) / Math.max(segmentEnd - segmentStart, 1e-9), 0, 1);
    const fromOffset = segment * 3;
    const toOffset = fromOffset + 3;
    const outputOffset = particleIndex * 3;
    particles.positions[outputOffset] = lerp(line.coordinates[fromOffset], line.coordinates[toOffset], amount);
    particles.positions[outputOffset + 1] = lerp(line.coordinates[fromOffset + 1], line.coordinates[toOffset + 1], amount);
    particles.positions[outputOffset + 2] = lerp(
      line.coordinates[fromOffset + 2],
      line.coordinates[toOffset + 2],
      amount,
    ) + zOffset;
    const speed = lerp(line.speeds[segment], line.speeds[segment + 1], amount);
    writeSolverParticleColor(particles.colors, outputOffset, speed, particles.speedMin, particles.speedMax);
  }
  particles.points.geometry.getAttribute("position").needsUpdate = true;
  particles.points.geometry.getAttribute("color").needsUpdate = true;
}

function resetSolverParticles() {
  const particles = state.viewer.solverParticles;
  if (particles?.points) {
    particles.points.parent?.remove(particles.points);
    particles.points.geometry.dispose();
    particles.points.material.map?.dispose();
    particles.points.material.dispose();
  }
  state.viewer.solverParticles = null;
}

function renderThreeStlModel(cameraState, dt = 0) {
  const view = ensureThreeViewer();
  if (!view) return false;
  try {
    const geometryKey = [
      state.modelPath || "model",
      state.mesh?.triangleCount || 0,
      state.mesh?.sampledTriangleCount || 0,
      modelOrientationKey(),
      state.viewer.surfaceMode,
      state.caseReport?.surfacePressure?.file || "no-cp",
      state.caseReport?.surfacePressure?.triangleCount || 0,
    ].join("|");
    if (view.geometryKey !== geometryKey) rebuildThreeGeometry(view, geometryKey);
    if (!view.mesh) return false;

    const width = Math.max(1, Math.round(cameraState.width));
    const height = Math.max(1, Math.round(cameraState.height));
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    if (view.width !== width || view.height !== height || view.pixelRatio !== pixelRatio) {
      view.renderer.setPixelRatio(pixelRatio);
      view.renderer.setSize(width, height, false);
      view.width = width;
      view.height = height;
      view.pixelRatio = pixelRatio;
    }

    updateThreeCamera(view, cameraState);
    view.wireframe.visible = els.showEdges.checked;
    updateSolverParticles(view, dt);
    view.renderer.render(view.scene, view.camera);
    return true;
  } catch (error) {
    disableThreeViewer(error);
    return false;
  }
}

function ensureThreeViewer() {
  const view = state.viewer.webgl;
  if (view.failed) return null;
  if (view.renderer) return view;
  try {
    view.renderer = new THREE.WebGLRenderer({
      canvas: els.modelCanvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    view.contextLostHandler = (event) => {
      event.preventDefault();
      disableThreeViewer(new Error("WebGL context was lost."));
      drawFlow();
    };
    els.modelCanvas.addEventListener("webglcontextlost", view.contextLostHandler, { once: true });
    view.renderer.setClearColor(0x000000, 0);
    view.renderer.outputColorSpace = THREE.SRGBColorSpace;
    view.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    view.renderer.toneMappingExposure = 1.1;
    view.scene = new THREE.Scene();
    view.camera = new THREE.PerspectiveCamera(45, 1, VIEWER_CAMERA_NEAR, VIEWER_CAMERA_FAR);
    view.group = new THREE.Group();
    view.scene.add(view.group);
    view.scene.add(new THREE.HemisphereLight(0xdcebf0, 0x17232c, 2.25));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.1);
    keyLight.position.set(-4, -6, 8);
    view.scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x8fcbd5, 1.2);
    rimLight.position.set(6, 4, 3);
    view.scene.add(rimLight);
    return view;
  } catch (error) {
    disableThreeViewer(error);
    return null;
  }
}

function disableThreeViewer(error) {
  const view = state.viewer.webgl;
  if (!view.failed) console.warn("Three.js viewer failed; using Canvas2D solved lines.", error);
  resetSolverParticles();
  try {
    if (view.group) disposeThreeGeometry(view);
  } catch (_disposeError) {
    // Continue clearing the failed viewer even when a partial resource cannot be disposed.
  }
  if (view.contextLostHandler) {
    els.modelCanvas.removeEventListener("webglcontextlost", view.contextLostHandler);
    view.contextLostHandler = null;
  }
  try {
    view.renderer?.clear();
    view.renderer?.dispose();
  } catch (_rendererError) {
    // Resetting the canvas below removes any stale frame left by a failed renderer.
  }
  view.failed = true;
  view.renderer = null;
  view.scene = null;
  view.camera = null;
  view.group = null;
  view.mesh = null;
  view.wireframe = null;
  view.geometryKey = null;
  view.width = 0;
  view.height = 0;
  view.pixelRatio = 0;
  els.modelCanvas.width = Math.max(1, els.modelCanvas.width);
  state.viewer.modelLayer = null;
  state.viewer.surfaceMode = "material";
  state.viewer.solverFlowMode = "lines";
  syncSurfaceModeControls();
  syncSolverFlowControls();
}

function rebuildThreeGeometry(view, geometryKey) {
  disposeThreeGeometry(view);
  const solvedSurfaceMode = ["cp", "temperature", "drag"].includes(state.viewer.surfaceMode);
  const surfacePressure = solvedSurfaceMode && hasSurfacePressure()
    ? state.caseReport.surfacePressure
    : null;
  if (surfacePressure) {
    rebuildThreeSolvedSurfaceGeometry(view, geometryKey, surfacePressure);
    return;
  }
  const zOffset = meshGroundOffset();
  const vertexIds = new Map();
  const positions = [];
  const indices = [];
  const appendVertex = (x, y, z) => {
    const key = `${x}|${y}|${z}`;
    let vertexId = vertexIds.get(key);
    if (vertexId == null) {
      vertexId = positions.length / 3;
      vertexIds.set(key, vertexId);
      positions.push(x, y, z);
    }
    indices.push(vertexId);
  };
  const exact = state.viewer.exactMesh;
  if (exact?.positions?.length) {
    for (let index = 0; index < exact.positions.length; index += 3) {
      const point = exactDisplayPoint(
        exact.positions[index],
        exact.positions[index + 1],
        exact.positions[index + 2],
      );
      appendVertex(point[0], point[1], point[2] + zOffset);
    }
  } else {
    const triangles = orientedPreviewTriangles();
    if (!triangles.length) return;
    for (const triangle of triangles) {
      for (let index = 0; index < triangle.v.length; index += 3) {
        appendVertex(
          triangle.v[index],
          triangle.v[index + 1],
          triangle.v[index + 2] + zOffset,
        );
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshStandardMaterial({
    color: 0x66747d,
    roughness: 0.72,
    metalness: 0.04,
    side: THREE.DoubleSide,
  });
  const wireMaterial = new THREE.MeshBasicMaterial({
    color: 0xa2d2db,
    wireframe: true,
    transparent: true,
    opacity: 0.24,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  view.mesh = new THREE.Mesh(geometry, material);
  view.wireframe = new THREE.Mesh(geometry, wireMaterial);
  view.wireframe.visible = els.showEdges.checked;
  view.group.add(view.mesh, view.wireframe);
  view.geometryKey = geometryKey;
}

function rebuildThreeSolvedSurfaceGeometry(view, geometryKey, surfacePressure) {
  const zOffset = meshGroundOffset();
  const positions = [];
  const colors = [];
  const indices = [];
  const showTemperature = state.viewer.surfaceMode === "temperature";
  const showDrag = state.viewer.surfaceMode === "drag";
  const range = showTemperature
    ? surfacePressure.temperatureDisplayRangeK || surfacePressure.temperatureKRange || [273.15, 373.15]
    : showDrag
      ? hasSurfaceWallShear()
        ? surfacePressure.totalDragDisplayRange || surfacePressure.totalDragDensityRange || [-1, 1]
        : surfacePressure.pressureDragDisplayRange || surfacePressure.pressureDragDensityRange || [-1, 1]
      : surfacePressure.cpDisplayRange || surfacePressure.cpRange || [-1, 1];
  const minimum = Number(range[0] ?? -1);
  const maximum = Number(range[1] ?? 1);

  const triangleDragValues = hasSurfaceWallShear()
    ? surfacePressure.triangleTotalDragValues
    : surfacePressure.trianglePressureDragValues;
  const useFaceDrag = showDrag && triangleDragValues?.length === surfacePressure.triangles.length;
  if (useFaceDrag) {
    surfacePressure.triangles.forEach((triangle, triangleIndex) => {
      const value = Number(triangleDragValues[triangleIndex] || 0);
      const color = dragColorChannels(value, minimum, maximum);
      for (const pointIndex of triangle) {
        const point = surfacePressure.points[Number(pointIndex)];
        positions.push(Number(point[0]), Number(point[1]), Number(point[2]) + zOffset);
        colors.push(color[0] / 255, color[1] / 255, color[2] / 255);
      }
    });
  } else {
    surfacePressure.points.forEach((point, pointIndex) => {
      positions.push(Number(point[0]), Number(point[1]), Number(point[2]) + zOffset);
      const dragValueIndex = hasSurfaceWallShear() ? 6 : 4;
      const value = showTemperature
        ? Number(surfacePressure.temperatureKValues?.[pointIndex] ?? minimum)
        : Number(point[showDrag ? dragValueIndex : 3] || 0);
      const color = showTemperature
        ? temperatureColorChannels(value, minimum, maximum)
        : showDrag
          ? dragColorChannels(value, minimum, maximum)
          : pressureColorChannels(value, minimum, maximum);
      colors.push(color[0] / 255, color[1] / 255, color[2] / 255);
    });
    for (const triangle of surfacePressure.triangles) {
      indices.push(Number(triangle[0]), Number(triangle[1]), Number(triangle[2]));
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  if (indices.length) geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.58,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  const wireMaterial = new THREE.MeshBasicMaterial({
    color: 0xa2d2db,
    wireframe: true,
    transparent: true,
    opacity: 0.24,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  view.mesh = new THREE.Mesh(geometry, material);
  view.wireframe = new THREE.Mesh(geometry, wireMaterial);
  view.wireframe.visible = els.showEdges.checked;
  view.group.add(view.mesh, view.wireframe);
  view.geometryKey = geometryKey;
}

function exactDisplayPoint(x, y, z) {
  let point = [x, y, z];
  const center = state.mesh?.normalizedCenter || [0, 0, 0];
  const scale = Number(state.mesh?.normalizedScale || 1);
  if (state.viewer.meshSource === "case") {
    const flowAxis = state.caseReport?.caseSetup?.flow?.axis
      || state.caseReport?.caseSetup?.orientation?.target_flow_axis
      || "x";
    const sourceFlow = signedAxisVector(`+${flowAxis}`);
    const sourceUp = signedAxisVector(flowAxis === "z" ? "+y" : "+z");
    const sourceSide = cross3(sourceUp, sourceFlow);
    const targetFlow = [1, 0, 0];
    const targetUp = [0, 0, 1];
    const targetSide = cross3(targetUp, targetFlow);
    point = transformPoint(point, sourceFlow, sourceSide, sourceUp, targetFlow, targetSide, targetUp, 1);
    return point.map((value, axis) => (value - Number(center[axis] || 0)) * scale);
  }

  point = point.map((value, axis) => (value - Number(center[axis] || 0)) * scale);
  point = rotatePointDegrees(point, modelRotationDegrees());
  const sourceFlow = signedAxisVector(els.sourceFlowDirection.value);
  const sourceUp = signedAxisVector(els.sourceUpDirection.value);
  if (Math.abs(dot3(sourceFlow, sourceUp)) > 1e-9) return point;
  const sourceSide = cross3(sourceUp, sourceFlow);
  const targetFlow = [1, 0, 0];
  const targetUp = [0, 0, 1];
  const targetSide = cross3(targetUp, targetFlow);
  return transformPoint(point, sourceFlow, sourceSide, sourceUp, targetFlow, targetSide, targetUp, 1);
}

function updateThreeCamera(view, cameraState) {
  const yaw = cameraState.yaw;
  const pitch = cameraState.pitch;
  const sinYaw = Math.sin(yaw);
  const cosYaw = Math.cos(yaw);
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const up = new THREE.Vector3(-sinPitch * sinYaw, -sinPitch * cosYaw, cosPitch);
  const depth = new THREE.Vector3(cosPitch * sinYaw, cosPitch * cosYaw, sinPitch);
  const focal = cameraState.focal;
  view.camera.fov = THREE.MathUtils.radToDeg(
    2 * Math.atan(cameraState.height / Math.max(1, 2 * cameraState.unit * focal)),
  );
  view.camera.aspect = cameraState.width / Math.max(1, cameraState.height);
  view.camera.position.copy(depth).multiplyScalar(-focal);
  view.camera.up.copy(up);
  view.camera.lookAt(0, 0, 0);
  view.camera.setViewOffset(
    cameraState.width,
    cameraState.height,
    0,
    -cameraState.height * 0.05,
    cameraState.width,
    cameraState.height,
  );
  view.camera.updateProjectionMatrix();
  view.group.position.set(0, 0, 0);
}

function invalidateThreeGeometry() {
  const view = state.viewer.webgl;
  if (!view.renderer) return;
  disposeThreeGeometry(view);
  view.renderer.clear();
}

function disposeThreeGeometry(view) {
  if (view.mesh) {
    view.group.remove(view.mesh);
    view.mesh.geometry.dispose();
    view.mesh.material.dispose();
  }
  if (view.wireframe) {
    view.group.remove(view.wireframe);
    view.wireframe.material.dispose();
  }
  view.mesh = null;
  view.wireframe = null;
  view.geometryKey = null;
}

function clearThreeViewer() {
  const view = state.viewer.webgl;
  if (view.renderer) view.renderer.clear();
}

function drawCachedStlMesh(ctx, camera) {
  const key = stlLayerKey(camera);
  if (!state.viewer.modelLayer || state.viewer.modelLayer.key !== key) {
    const layerCanvas = document.createElement("canvas");
    layerCanvas.width = Math.max(1, Math.ceil(camera.width));
    layerCanvas.height = Math.max(1, Math.ceil(camera.height));
    const layerCtx = layerCanvas.getContext("2d");
    drawStlMesh(layerCtx, camera);
    state.viewer.modelLayer = { key, canvas: layerCanvas };
  }
  ctx.drawImage(state.viewer.modelLayer.canvas, 0, 0);
}

function stlLayerKey(camera) {
  return [
    state.modelPath || "model",
    state.mesh?.triangleCount || 0,
    state.mesh?.sampledTriangleCount || 0,
    Math.round(camera.width),
    Math.round(camera.height),
    camera.yaw.toFixed(4),
    camera.pitch.toFixed(4),
    camera.unit.toFixed(2),
    modelOrientationKey(),
  ].join("|");
}

function drawStlMesh(ctx, camera) {
  const zOffset = meshGroundOffset();
  const oriented = orientedPreviewTriangles();
  const useSmoothMaterial = Number(state.mesh?.triangleCount || 0) > 2_500;
  const vertexLights = useSmoothMaterial ? smoothedPreviewVertexLight(oriented) : null;
  const triangles = oriented
    .map((triangle, triangleIndex) => {
      const vertices = [];
      for (let index = 0; index < triangle.v.length; index += 3) {
        vertices.push({ x: triangle.v[index], y: triangle.v[index + 1], z: triangle.v[index + 2] + zOffset });
      }
      const projected = vertices.map((vertex) => project(vertex, camera));
      return {
        points: projected,
        normal: triangle.n,
        vertexLights: vertexLights?.[triangleIndex] || null,
        depth: projected.reduce((sum, point) => sum + point.depth, 0) / 3,
      };
    })
    .sort((a, b) => b.depth - a.depth);

  const silhouette = convexHull2D(triangles.flatMap((triangle) => triangle.points));
  const needsBacking = state.mesh?.isComplete === false;
  if (needsBacking && silhouette.length >= 3) {
    ctx.fillStyle = "rgb(52, 63, 72)";
    polygon(ctx, silhouette);
  }

  for (const triangle of triangles) {
    ctx.fillStyle = triangle.vertexLights
      ? surfaceColorFromLight(
          triangle.vertexLights.reduce((sum, value) => sum + value, 0) / triangle.vertexLights.length,
        )
      : surfaceColor(triangle.normal);
    polygon(ctx, triangle.points);
  }

  if (needsBacking && silhouette.length >= 3) {
    ctx.strokeStyle = "rgba(191, 211, 220, 0.34)";
    ctx.lineWidth = 1.1;
    strokePolygon(ctx, silhouette);
  }
}

function smoothedPreviewVertexLight(triangles) {
  const key = `${state.modelPath || "model"}|${state.mesh?.sampledTriangleCount || 0}|${modelOrientationKey()}`;
  if (state.viewer.smoothVertexLight?.key === key) return state.viewer.smoothVertexLight.lights;

  const sums = new Map();
  for (const triangle of triangles) {
    const normal = triangle.n || [0, 0, 1];
    for (let index = 0; index < triangle.v.length; index += 3) {
      const vertexKey = `${triangle.v[index]}|${triangle.v[index + 1]}|${triangle.v[index + 2]}`;
      const sum = sums.get(vertexKey) || [0, 0, 0];
      sum[0] += normal[0] || 0;
      sum[1] += normal[1] || 0;
      sum[2] += normal[2] || 0;
      sums.set(vertexKey, sum);
    }
  }

  const lights = triangles.map((triangle) => {
    const fallback = triangle.n || [0, 0, 1];
    const values = [];
    for (let index = 0; index < triangle.v.length; index += 3) {
      const vertexKey = `${triangle.v[index]}|${triangle.v[index + 1]}|${triangle.v[index + 2]}`;
      const sum = sums.get(vertexKey) || fallback;
      const length = Math.hypot(sum[0], sum[1], sum[2]);
      const normal = length > 1e-8 ? sum.map((value) => value / length) : fallback;
      values.push(surfaceLight(normal, true));
    }
    return values;
  });
  state.viewer.smoothVertexLight = { key, lights };
  return lights;
}

function convexHull2D(points) {
  if (points.length < 4) return points;
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const unique = sorted.filter((point, index) => {
    const previous = sorted[index - 1];
    return !previous || point.x !== previous.x || point.y !== previous.y;
  });
  if (unique.length < 4) return unique;

  const turn = (origin, first, second) =>
    (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (second.x - origin.x);
  const lower = [];
  for (const point of unique) {
    while (lower.length >= 2 && turn(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index];
    while (upper.length >= 2 && turn(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function modelOrientationKey() {
  const rotation = modelRotationDegrees();
  const orientation = state.viewer.meshSource === "case"
    ? "case-geometry"
    : `${els.sourceFlowDirection.value}|${els.sourceUpDirection.value}|${rotation.x}|${rotation.y}|${rotation.z}`;
  const roadGapMm = els.includeGround.checked ? Number(els.groundClearanceMm.value || 0) : 0;
  return `${orientation}|road-gap:${roadGapMm}`;
}

function orientedPreviewTriangles() {
  if (!state.mesh?.triangles?.length) return [];
  const key = `${state.modelPath || "model"}|${state.mesh.sampledTriangleCount || 0}|${modelOrientationKey()}`;
  if (state.viewer.orientedMesh?.key === key) return state.viewer.orientedMesh.triangles;
  if (state.viewer.meshSource === "case") {
    state.viewer.orientedMesh = { key, triangles: state.mesh.triangles };
    return state.mesh.triangles;
  }

  const sourceFlow = signedAxisVector(els.sourceFlowDirection.value);
  const sourceUp = signedAxisVector(els.sourceUpDirection.value);
  const sourceSide = cross3(sourceUp, sourceFlow);
  const targetFlow = [1, 0, 0];
  const targetUp = [0, 0, 1];
  const targetSide = cross3(targetUp, targetFlow);
  const canOrient = Math.abs(dot3(sourceFlow, sourceUp)) <= 1e-9;
  const rotation = modelRotationDegrees();
  const triangles = state.mesh.triangles.map((triangle) => {
    const vertices = [];
    for (let index = 0; index < triangle.v.length; index += 3) {
      const rotated = rotatePointDegrees(
        [triangle.v[index], triangle.v[index + 1], triangle.v[index + 2]],
        rotation,
      );
      vertices.push(
        canOrient
          ? transformPoint(rotated, sourceFlow, sourceSide, sourceUp, targetFlow, targetSide, targetUp, 1)
          : rotated,
      );
    }
    return {
      v: vertices.flat(),
      n: triangleNormal(vertices),
    };
  });
  state.viewer.orientedMesh = { key, triangles };
  return triangles;
}

function triangleNormal(vertices) {
  const edgeA = vertices[1].map((value, index) => value - vertices[0][index]);
  const edgeB = vertices[2].map((value, index) => value - vertices[0][index]);
  const normal = cross3(edgeA, edgeB);
  const length = Math.hypot(...normal) || 1;
  return normal.map((value) => value / length);
}

function surfaceColor(normal) {
  return surfaceColorFromLight(surfaceLight(normal, false));
}

function surfaceLight(normal, smooth) {
  const ny = normal?.[1] || 0;
  const nz = normal?.[2] || 0;
  return smooth
    ? clamp(0.55 + nz * 0.14 - ny * 0.05, 0.38, 0.72)
    : clamp(0.5 + nz * 0.25 - ny * 0.08, 0.3, 0.88);
}

function surfaceColorFromLight(light) {
  return `rgb(${Math.round(98 * light + 36)}, ${Math.round(119 * light + 34)}, ${Math.round(132 * light + 34)})`;
}

function drawPreviewVehicle(ctx, camera) {
  const dims = normalizedModelDimensions();
  const length = dims.length;
  const width = dims.width;
  const height = dims.height;
  const baseZ = -0.58;
  const body = boxFaces(-length / 2, length / 2, -width / 2, width / 2, baseZ, baseZ + height * 0.48);
  const cabin = [
    face(
      [
        { x: -length * 0.23, y: -width * 0.28, z: baseZ + height * 0.48 },
        { x: length * 0.24, y: -width * 0.25, z: baseZ + height * 0.48 },
        { x: length * 0.16, y: -width * 0.18, z: baseZ + height * 0.92 },
        { x: -length * 0.13, y: -width * 0.20, z: baseZ + height * 0.96 },
      ],
      "#263746",
      "#8fc7d2",
    ),
    face(
      [
        { x: -length * 0.23, y: width * 0.28, z: baseZ + height * 0.48 },
        { x: length * 0.24, y: width * 0.25, z: baseZ + height * 0.48 },
        { x: length * 0.16, y: width * 0.18, z: baseZ + height * 0.92 },
        { x: -length * 0.13, y: width * 0.20, z: baseZ + height * 0.96 },
      ],
      "#202f3d",
      "#8fc7d2",
    ),
    face(
      [
        { x: -length * 0.13, y: -width * 0.20, z: baseZ + height * 0.96 },
        { x: length * 0.16, y: -width * 0.18, z: baseZ + height * 0.92 },
        { x: length * 0.16, y: width * 0.18, z: baseZ + height * 0.92 },
        { x: -length * 0.13, y: width * 0.20, z: baseZ + height * 0.96 },
      ],
      "#334756",
      "#a3d5dc",
    ),
  ];
  const faces = [...body, ...cabin].sort((a, b) => averageDepth(b.points, camera) - averageDepth(a.points, camera));

  for (const item of faces) {
    const points = item.points.map((point) => project(point, camera));
    ctx.fillStyle = item.fill;
    polygon(ctx, points);
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 1;
    strokePolygon(ctx, points);
    if (item.highlight) {
      ctx.fillStyle = item.highlight;
      ctx.globalAlpha = 0.17;
      polygon(ctx, points);
      ctx.globalAlpha = 1;
    }
  }

  drawPressureFace(ctx, camera, length, width, height, baseZ);
  drawWheels(ctx, camera, length, width, baseZ);
}

function drawPressureFace(ctx, camera, length, width, height, baseZ) {
  const front = [
    { x: -length / 2 - 0.012, y: -width / 2, z: baseZ },
    { x: -length / 2 - 0.012, y: width / 2, z: baseZ },
    { x: -length / 2 - 0.012, y: width / 2, z: baseZ + height * 0.48 },
    { x: -length / 2 - 0.012, y: -width / 2, z: baseZ + height * 0.48 },
  ].map((point) => project(point, camera));
  const low = [
    { x: length * 0.12, y: -width / 2 - 0.012, z: baseZ + height * 0.48 },
    { x: length / 2, y: -width / 2 - 0.012, z: baseZ + height * 0.38 },
    { x: length / 2, y: width / 2 + 0.012, z: baseZ + height * 0.38 },
    { x: length * 0.12, y: width / 2 + 0.012, z: baseZ + height * 0.48 },
  ].map((point) => project(point, camera));
  ctx.fillStyle = "rgba(229,73,58,0.34)";
  polygon(ctx, front);
  ctx.fillStyle = "rgba(75,138,255,0.24)";
  polygon(ctx, low);
}

function drawWheels(ctx, camera, length, width, baseZ) {
  const wheelX = [-length * 0.34, length * 0.34];
  const wheelY = [-width * 0.56, width * 0.56];
  for (const x of wheelX) {
    for (const y of wheelY) {
      const point = project({ x, y, z: baseZ - 0.02 }, camera);
      ctx.fillStyle = "#0a0d11";
      ctx.beginPath();
      ctx.ellipse(point.x, point.y, 10 * point.scale, 15 * point.scale, 0.15, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.26)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

function drawViewerHud(ctx, width, height) {
  const report = state.report;
  const solvedFlow = state.caseReport?.solverStreamlines?.lines?.length;
  const caseSpeedMph = Number(state.caseReport?.caseSetup?.flow?.speed_mph);
  const speedMph = solvedFlow && Number.isFinite(caseSpeedMph)
    ? caseSpeedMph
    : Number(els.speedMph.value || 70);
  const speedMps = speedMph * 0.44704;
  const dynamicPressurePa = 0.5 * 1.225 * speedMps * speedMps;
  const machNumber = speedMps / 343;
  const gapMm = Math.max(0, Number(els.groundClearanceMm.value || 0));
  const lines = state.viewMode === "basic"
    ? [
      `${Math.round(speedMph)} mph airflow`,
      solvedFlow ? "Calculated OpenFOAM paths" : "Visual preview paths",
      report ? "Drag to orbit | scroll to zoom" : "Load a model to begin",
    ]
    : [
      `${Math.round(speedMph)} mph | M ${fmt(machNumber)} | q ${formatInt(Math.round(dynamicPressurePa))} Pa`,
      els.includeGround.checked
        ? `${els.movingGround.checked ? "moving ground" : "ground"} | gap ${fmt(gapMm)} mm`
        : "open tunnel",
      state.viewer.surfaceMode === "drag" && hasSurfaceDrag()
        ? hasSurfaceWallShear() ? "OpenFOAM total drag" : "OpenFOAM pressure drag"
        : state.viewer.surfaceMode === "cp" && hasSurfacePressure()
          ? "OpenFOAM body Cp"
        : solvedFlow
          ? state.caseReport.solverStreamlines.timeAveraged
            ? "OpenFOAM mean-flow speed"
            : "OpenFOAM final-field speed"
          : report ? "surface-guided air preview" : "sample airflow preview",
    ];
  ctx.fillStyle = "rgba(15,24,34,0.68)";
  roundRectPath(ctx, 14, 14, 238, 78, 8);
  ctx.fill();
  ctx.fillStyle = "#dfeaf1";
  ctx.font = "700 13px Inter, system-ui, sans-serif";
  lines.forEach((lineText, index) => {
    ctx.fillText(lineText, 28, 38 + index * 18);
  });
}

function normalizedModelDimensions() {
  const meshBounds = meshPreviewBounds();
  if (meshBounds) {
    return {
      length: clamp(meshBounds.max[0] - meshBounds.min[0], 1.0, 4.2),
      width: clamp(meshBounds.max[1] - meshBounds.min[1], 0.7, 3.0),
      height: clamp(meshBounds.max[2] - meshBounds.min[2], 0.45, 2.4),
    };
  }
  const dims = state.report?.bounds?.dimensions || [4.2, 1.8, 1.35];
  const maxDim = Math.max(...dims, 1);
  const normalized = dims.map((dim) => Math.max(0.2, (dim / maxDim) * 3.8));
  const length = clamp(normalized[0], 2.5, 4.2);
  const width = clamp(Math.max(normalized[1], 1.2), 1.2, 2.2);
  const height = clamp(Math.max(normalized[2], 0.75), 0.75, 1.55);
  return { length, width, height };
}

function meshPreviewBounds() {
  const bounds = rawMeshPreviewBounds();
  if (!bounds) return null;
  const zOffset = VIEWER_GROUND_Z + groundGapDisplay() - bounds.min[2];
  return {
    min: [bounds.min[0], bounds.min[1], bounds.min[2] + zOffset],
    max: [bounds.max[0], bounds.max[1], bounds.max[2] + zOffset],
  };
}

function rawMeshPreviewBounds() {
  const oriented = orientedPreviewTriangles();
  if (!oriented.length) return null;
  if (state.viewer.meshBounds) return state.viewer.meshBounds;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const triangle of oriented) {
    for (let index = 0; index < triangle.v.length; index += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        const value = triangle.v[index + axis];
        min[axis] = Math.min(min[axis], value);
        max[axis] = Math.max(max[axis], value);
      }
    }
  }
  state.viewer.meshBounds = { min, max };
  return state.viewer.meshBounds;
}

function meshGroundOffset() {
  const bounds = rawMeshPreviewBounds();
  if (!bounds) return 0;
  return VIEWER_GROUND_Z + groundGapDisplay() - bounds.min[2];
}

function groundGapDisplay() {
  if (!els.includeGround.checked) return 0;
  const gapM = Math.max(0, Number(els.groundClearanceMm.value || 0)) / 1000;
  const previewScale = Number(state.mesh?.normalizedScale || 1);
  if (state.viewer.meshSource === "case") return gapM * previewScale;
  return (gapM / Math.max(effectiveUnitScale(), 1e-12)) * previewScale;
}

function modelFlowEnvelope() {
  const meshKey = `${state.modelPath || "preview"}|${state.mesh?.sampledTriangleCount || 0}|${modelOrientationKey()}`;
  if (state.viewer.flowEnvelope?.key === meshKey) return state.viewer.flowEnvelope;

  const bounds = meshPreviewBounds();
  const binCount = 96;
  if (!bounds || !state.mesh?.triangles?.length) {
    const dims = normalizedModelDimensions();
    const minX = -dims.length / 2;
    const maxX = dims.length / 2;
    const centerZ = -0.58 + dims.height * 0.47;
    const bins = Array.from({ length: binCount }, (_, index) => {
      const t = index / (binCount - 1);
      const nose = smoothstep(clamp(t / 0.16, 0, 1));
      const tail = smoothstep(clamp((1 - t) / 0.2, 0, 1));
      const profile = Math.max(0.08, Math.min(nose, tail));
      return {
        centerY: 0,
        centerZ,
        radiusY: dims.width * 0.5 * (0.7 + profile * 0.3),
        radiusZ: dims.height * 0.5 * (0.56 + profile * 0.44),
      };
    });
    state.viewer.flowEnvelope = finishFlowEnvelope(meshKey, minX, maxX, bins);
    return state.viewer.flowEnvelope;
  }

  const minX = bounds.min[0];
  const maxX = bounds.max[0];
  const length = Math.max(0.01, maxX - minX);
  const binWidth = length / (binCount - 1);
  const slices = Array.from({ length: binCount }, () => ({
    yMin: Infinity,
    yMax: -Infinity,
    zMin: Infinity,
    zMax: -Infinity,
  }));
  const zOffset = meshGroundOffset();

  for (const triangle of orientedPreviewTriangles()) {
    const vertices = [];
    for (let index = 0; index < triangle.v.length; index += 3) {
      vertices.push({ x: triangle.v[index], y: triangle.v[index + 1], z: triangle.v[index + 2] + zOffset });
    }
    const triMinX = Math.min(...vertices.map((vertex) => vertex.x));
    const triMaxX = Math.max(...vertices.map((vertex) => vertex.x));
    const firstBin = clamp(Math.floor((triMinX - minX) / binWidth) - 1, 0, binCount - 1);
    const lastBin = clamp(Math.ceil((triMaxX - minX) / binWidth) + 1, 0, binCount - 1);
    for (let binIndex = firstBin; binIndex <= lastBin; binIndex += 1) {
      const x = minX + binIndex * binWidth;
      const points = triangleSlicePoints(vertices, x, binWidth * 0.55);
      for (const point of points) {
        const slice = slices[binIndex];
        slice.yMin = Math.min(slice.yMin, point.y);
        slice.yMax = Math.max(slice.yMax, point.y);
        slice.zMin = Math.min(slice.zMin, point.z);
        slice.zMax = Math.max(slice.zMax, point.z);
      }
    }
  }

  const bins = slices.map((slice, index) => {
    if (Number.isFinite(slice.yMin)) return sliceToSection(slice);
    for (let distance = 1; distance < binCount; distance += 1) {
      for (const candidate of [index - distance, index + distance]) {
        if (candidate >= 0 && candidate < binCount && Number.isFinite(slices[candidate].yMin)) {
          return sliceToSection(slices[candidate]);
        }
      }
    }
    return { centerY: 0, centerZ: -0.1, radiusY: 0.1, radiusZ: 0.1 };
  });

  for (let pass = 0; pass < 2; pass += 1) {
    const previous = bins.map((section) => ({ ...section }));
    for (let index = 1; index < bins.length - 1; index += 1) {
      for (const key of ["centerY", "centerZ", "radiusY", "radiusZ"]) {
        bins[index][key] = previous[index - 1][key] * 0.2 + previous[index][key] * 0.6 + previous[index + 1][key] * 0.2;
      }
    }
  }

  state.viewer.flowEnvelope = finishFlowEnvelope(meshKey, minX, maxX, bins);
  return state.viewer.flowEnvelope;
}

function triangleSlicePoints(vertices, x, tolerance) {
  const points = [];
  const edges = [
    [vertices[0], vertices[1]],
    [vertices[1], vertices[2]],
    [vertices[2], vertices[0]],
  ];
  for (const [start, end] of edges) {
    const delta = end.x - start.x;
    if (Math.abs(delta) < 1e-8) {
      if (Math.abs(start.x - x) <= tolerance) {
        points.push(start, end);
      }
      continue;
    }
    const t = (x - start.x) / delta;
    if (t >= -0.02 && t <= 1.02) {
      points.push({
        x,
        y: lerp(start.y, end.y, clamp(t, 0, 1)),
        z: lerp(start.z, end.z, clamp(t, 0, 1)),
      });
    }
  }
  if (!points.length) {
    for (const vertex of vertices) {
      if (Math.abs(vertex.x - x) <= tolerance) points.push(vertex);
    }
  }
  return points;
}

function sliceToSection(slice) {
  return {
    centerY: (slice.yMin + slice.yMax) / 2,
    centerZ: (slice.zMin + slice.zMax) / 2,
    radiusY: Math.max(0.04, (slice.yMax - slice.yMin) / 2 + 0.035),
    radiusZ: Math.max(0.04, (slice.zMax - slice.zMin) / 2 + 0.035),
  };
}

function finishFlowEnvelope(key, minX, maxX, bins) {
  const length = Math.max(0.01, maxX - minX);
  const frontCount = Math.max(3, Math.floor(bins.length * 0.18));
  const tailCount = Math.max(3, Math.floor(bins.length * 0.18));
  const front = bins.slice(0, frontCount).reduce((best, section) => {
    return section.radiusY * section.radiusZ > best.radiusY * best.radiusZ ? section : best;
  }, bins[0]);
  const tailBins = bins.slice(-tailCount);
  const tail = tailBins.reduce((best, section) => {
    return section.radiusY * section.radiusZ > best.radiusY * best.radiusZ ? section : best;
  }, tailBins[0]);
  const centerY = bins.reduce((sum, section) => sum + section.centerY, 0) / bins.length;
  const centerZ = bins.reduce((sum, section) => sum + section.centerZ, 0) / bins.length;
  return {
    key,
    minX,
    maxX,
    length,
    bins,
    front: { ...front },
    tail: { ...tail },
    centerY,
    centerZ,
    maxRadiusY: Math.max(...bins.map((section) => section.radiusY)),
    maxRadiusZ: Math.max(...bins.map((section) => section.radiusZ)),
  };
}

function sampleEnvelopeSection(envelope, x) {
  if (x < envelope.minX || x > envelope.maxX) return null;
  const position = ((x - envelope.minX) / envelope.length) * (envelope.bins.length - 1);
  const lowerIndex = clamp(Math.floor(position), 0, envelope.bins.length - 1);
  const upperIndex = clamp(lowerIndex + 1, 0, envelope.bins.length - 1);
  const amount = position - lowerIndex;
  const lower = envelope.bins[lowerIndex];
  const upper = envelope.bins[upperIndex];
  return {
    centerY: lerp(lower.centerY, upper.centerY, amount),
    centerZ: lerp(lower.centerZ, upper.centerZ, amount),
    radiusY: lerp(lower.radiusY, upper.radiusY, amount),
    radiusZ: lerp(lower.radiusZ, upper.radiusZ, amount),
  };
}

function flowCrossSectionAt(x) {
  const envelope = modelFlowEnvelope();
  const local = sampleEnvelopeSection(envelope, x);
  if (local) return { ...local, influence: 1, region: "body" };

  if (x < envelope.minX) {
    const distance = envelope.minX - x;
    const approach = Math.max(0.9, envelope.length * 0.62);
    if (distance >= approach) return null;
    const influence = smoothstep(1 - distance / approach);
    return {
      ...envelope.front,
      radiusY: envelope.front.radiusY * (0.62 + influence * 0.38),
      radiusZ: envelope.front.radiusZ * (0.62 + influence * 0.38),
      influence,
      region: "approach",
    };
  }

  const distance = x - envelope.maxX;
  const influence = Math.exp(-distance / Math.max(0.8, envelope.length * 0.48));
  if (influence < 0.025) return null;
  const tail = envelope.bins[envelope.bins.length - 1];
  return {
    ...tail,
    radiusY: Math.max(tail.radiusY, envelope.maxRadiusY * 0.72) * influence,
    radiusZ: Math.max(tail.radiusZ, envelope.maxRadiusZ * 0.72) * influence,
    influence: influence * 0.78,
    region: "wake",
  };
}

function modelObstacleBounds(padding = 0) {
  const meshBounds = meshPreviewBounds();
  if (meshBounds) {
    return {
      min: {
        x: meshBounds.min[0] - padding,
        y: meshBounds.min[1] - padding,
        z: meshBounds.min[2] - padding,
      },
      max: {
        x: meshBounds.max[0] + padding,
        y: meshBounds.max[1] + padding,
        z: meshBounds.max[2] + padding,
      },
    };
  }

  const dims = normalizedModelDimensions();
  const baseZ = -0.58;
  return {
    min: {
      x: -dims.length / 2 - padding,
      y: -dims.width / 2 - padding,
      z: baseZ - padding,
    },
    max: {
      x: dims.length / 2 + padding,
      y: dims.width / 2 + padding,
      z: baseZ + dims.height * 0.96 + padding,
    },
  };
}

function pointInsideObstacle(point, padding = 0) {
  const section = sampleEnvelopeSection(modelFlowEnvelope(), point.x);
  if (!section) return false;
  const radiusY = Math.max(0.02, section.radiusY + padding);
  const radiusZ = Math.max(0.02, section.radiusZ + padding);
  const y = (point.y - section.centerY) / radiusY;
  const z = (point.z - section.centerZ) / radiusZ;
  return y * y + z * z <= 1;
}

function segmentIntersectsObstacle(start, end, padding = 0) {
  for (let index = 0; index <= 8; index += 1) {
    const amount = index / 8;
    if (
      pointInsideObstacle(
        {
          x: lerp(start.x, end.x, amount),
          y: lerp(start.y, end.y, amount),
          z: lerp(start.z, end.z, amount),
        },
        padding,
      )
    ) {
      return true;
    }
  }
  return false;
}

function bodyEnvelope() {
  const dims = normalizedModelDimensions();
  return {
    ...dims,
    tunnelLength: Math.max(12, dims.length * 4),
    tunnelWidth: Math.max(6.4, dims.width * 3.4),
  };
}

function boxFaces(x0, x1, y0, y1, z0, z1) {
  const points = {
    a: { x: x0, y: y0, z: z0 },
    b: { x: x1, y: y0, z: z0 },
    c: { x: x1, y: y1, z: z0 },
    d: { x: x0, y: y1, z: z0 },
    e: { x: x0, y: y0, z: z1 },
    f: { x: x1, y: y0, z: z1 },
    g: { x: x1, y: y1, z: z1 },
    h: { x: x0, y: y1, z: z1 },
  };
  return [
    face([points.a, points.b, points.c, points.d], "#1a2631"),
    face([points.e, points.h, points.g, points.f], "#314354", "#5ed7e5"),
    face([points.a, points.e, points.f, points.b], "#223241"),
    face([points.b, points.f, points.g, points.c], "#1d2b37"),
    face([points.c, points.g, points.h, points.d], "#263747"),
    face([points.d, points.h, points.e, points.a], "#344555"),
  ];
}

function face(points, fill, highlight = null) {
  return { points, fill, highlight };
}

function polygon(ctx, points) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.fill();
}

function strokePolygon(ctx, points) {
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.closePath();
  ctx.stroke();
}

function line(ctx, start, end) {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
}

function averageDepth(points, camera) {
  return points.reduce((sum, point) => sum + project(point, camera).depth, 0) / points.length;
}

function roundRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function rgba(hex, alpha) {
  const value = hex.replace("#", "");
  const red = parseInt(value.slice(0, 2), 16);
  const green = parseInt(value.slice(2, 4), 16);
  const blue = parseInt(value.slice(4, 6), 16);
  return `rgba(${red},${green},${blue},${alpha})`;
}

function setBusy(isBusy, label = "") {
  state.busy = isBusy;
  updateActionAvailability();
  if (label) {
    els.caseStatus.textContent = label;
    els.caseStatus.classList.remove("error");
  }
}

async function apiGet(url) {
  return fetchJson(url, { method: "GET" });
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch (_error) {
      // The model endpoint normally returns JSON errors, but keep a useful HTTP fallback.
    }
    throw new Error(message);
  }
  return response.arrayBuffer();
}

function showError(error) {
  els.caseStatus.textContent = error.message;
  els.caseStatus.classList.add("error");
}

function basename(path) {
  return String(path).split(/[\\/]/).pop();
}

function slug(value) {
  return value.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "aerolab-case";
}

function selectedUnitLabel() {
  const option = els.unitScale.selectedOptions?.[0];
  return option?.dataset?.label || "m";
}

function optionalFiniteNumber(value) {
  if (value === "" || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function optionalNumber(value) {
  if (value === "" || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function fmt(value) {
  return Number(value).toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

function fmtDistanceMm(value) {
  const millimeters = Math.abs(Number(value) || 0);
  return millimeters > 0 && millimeters < 0.001 ? "<0.001 mm" : `${fmt(millimeters)} mm`;
}

function formatInt(value) {
  return Number(value).toLocaleString();
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, amount) {
  return a + (b - a) * clamp(amount, 0, 1);
}

function smoothstep(value) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

window.addEventListener("resize", drawFlow);
boot().catch(showError);
