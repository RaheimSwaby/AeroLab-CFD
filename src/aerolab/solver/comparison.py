"""Locked baseline-versus-variant aerodynamic comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

from ..case import comparison_lock_metadata
from .analysis import case_report
from .util import _finite_number, _read_json_object

COEFFICIENT_CHANNELS = ("Cd", "Cl", "Cs", "CmRoll", "CmPitch", "CmYaw")
LOAD_CHANNELS = (
    "dragN",
    "signedSideForceN",
    "signedLiftN",
    "rollMomentNm",
    "pitchMomentNm",
    "yawMomentNm",
)
BALANCE_CHANNELS = (
    "signedFrontVerticalLoadN",
    "signedRearVerticalLoadN",
    "frontAeroBalancePercent",
)


def compare_cases(baseline_path: Path, variant_path: Path) -> dict[str, object]:
    """Compare two solved cases only after proving their decision-critical setup matches."""
    baseline_path = baseline_path.resolve()
    variant_path = variant_path.resolve()
    if baseline_path == variant_path:
        raise ValueError("Baseline and variant must be different AeroLab cases.")

    baseline_payload = _case_payload(baseline_path)
    variant_payload = _case_payload(variant_path)
    baseline_lock = comparison_lock_metadata(baseline_payload)
    variant_lock = comparison_lock_metadata(variant_payload)
    locks_match = baseline_lock["hash"] == variant_lock["hash"]
    setup_differences = (
        []
        if locks_match
        else _mapping_differences(baseline_lock["setup"], variant_lock["setup"])
    )

    baseline_report = case_report(baseline_path, include_validation=False)
    variant_report = case_report(variant_path, include_validation=False)
    baseline_qualified = _numerically_qualified(baseline_report)
    variant_qualified = _numerically_qualified(variant_report)
    coefficient_deltas = _coefficient_deltas(baseline_report, variant_report)
    load_deltas = _load_deltas(baseline_report, variant_report)
    balance_deltas = _balance_deltas(baseline_report, variant_report)
    statistical_deltas = _statistical_deltas(baseline_report, variant_report)
    baseline_statistically_ready = _statistically_ready(baseline_report)
    variant_statistically_ready = _statistically_ready(variant_report)
    six_axis_complete = all(
        isinstance(delta, dict)
        and delta.get("baseline") is not None
        and delta.get("variant") is not None
        for delta in coefficient_deltas.values()
    )
    balance_complete = all(
        isinstance(delta, dict)
        and delta.get("baseline") is not None
        and delta.get("variant") is not None
        for delta in balance_deltas.values()
    )
    decision_safe = bool(
        locks_match and baseline_qualified and variant_qualified and six_axis_complete
    )
    statistical_decision_safe = bool(
        decision_safe and baseline_statistically_ready and variant_statistically_ready
    )

    if not locks_match:
        status = "setup_mismatch"
        label = "Comparison blocked: setup lock mismatch"
    elif not six_axis_complete:
        status = "incomplete_loads"
        label = "Comparison blocked: six-axis loads incomplete"
    elif not baseline_qualified or not variant_qualified:
        status = "qualification_required"
        label = "Comparison blocked: numerical qualification required"
    else:
        status = "ready"
        label = "Controlled A/B difference"

    if not decision_safe:
        statistical_status = "controlled_comparison_required"
        statistical_label = "Statistical claim blocked: controlled numerical comparison required"
    elif not baseline_statistically_ready or not variant_statistically_ready:
        statistical_status = "statistical_evidence_required"
        statistical_label = "Statistical claim blocked: time-history evidence required"
    else:
        statistical_status = "ready"
        statistical_label = "Autocorrelation-adjusted difference evidence ready"

    if statistical_decision_safe:
        interpretation = (
            "The setup-locked delta is numerically controlled and both retained transient histories "
            "meet the statistical evidence gates. A statistically resolved interval excludes zero; "
            "it does not establish a universal effect outside these cases."
        )
    elif decision_safe:
        interpretation = (
            "The setup-locked delta is numerically controlled, but a statistical significance claim "
            "remains blocked until both transient histories support stationarity, at least 30 effective "
            "samples per requested channel, and at least 10 cycles for any meaningful spectral peak."
        )
    else:
        interpretation = (
            "The point differences are descriptive only. A design decision remains blocked until the "
            "setup lock, six-axis loads, and numerical qualification all pass; transient statistical "
            "evidence is a separate subsequent gate."
        )

    return {
        "schemaVersion": 2,
        "status": status,
        "statusLabel": label,
        "decisionSafe": decision_safe,
        "statisticalDecisionSafe": statistical_decision_safe,
        "statisticalStatus": statistical_status,
        "statisticalStatusLabel": statistical_label,
        "geometryExcludedFromLock": True,
        "locksMatch": locks_match,
        "setupDifferences": setup_differences,
        "baseline": _case_summary(
            baseline_path,
            baseline_report,
            baseline_payload,
            baseline_lock,
            baseline_qualified,
        ),
        "variant": _case_summary(
            variant_path,
            variant_report,
            variant_payload,
            variant_lock,
            variant_qualified,
        ),
        "coefficientDeltas": coefficient_deltas,
        "loadDeltas": load_deltas,
        "balanceDeltas": balance_deltas,
        "statisticalDeltas": statistical_deltas,
        "balanceComparisonQualified": balance_complete,
        "interpretation": interpretation,
    }


def _case_payload(case_path: Path) -> dict[str, object]:
    payload = _read_json_object(case_path / "case.json")
    if not payload:
        raise ValueError(f"{case_path} is not an AeroLab case with readable case.json metadata.")
    return payload


def _numerically_qualified(report: dict[str, object]) -> bool:
    assessment = report.get("qualityAssessment")
    return bool(
        isinstance(assessment, dict)
        and assessment.get("numericallyQualified", assessment.get("trusted"))
    )


def _case_summary(
    case_path: Path,
    report: dict[str, object],
    payload: dict[str, object],
    lock: dict[str, object],
    numerically_qualified: bool,
) -> dict[str, object]:
    stored_lock = payload.get("comparison_lock")
    stored_hash = stored_lock.get("hash") if isinstance(stored_lock, dict) else None
    return {
        "name": report.get("caseName", case_path.name),
        "path": str(case_path),
        "numericallyQualified": numerically_qualified,
        "qualificationStatus": report.get("qualificationStatus"),
        "statisticallyReady": _statistically_ready(report),
        "statisticalEvidence": _overall_statistical_evidence(report),
        "lockHash": lock["hash"],
        "storedLockHash": stored_hash,
        "storedLockCurrent": stored_hash == lock["hash"] if stored_hash is not None else None,
    }


def _coefficient_deltas(
    baseline_report: dict[str, object],
    variant_report: dict[str, object],
) -> dict[str, dict[str, float | None]]:
    baseline = baseline_report.get("forceCoeffs")
    variant = variant_report.get("forceCoeffs")
    return {
        key: _delta(
            _coefficient_value(baseline, key),
            _coefficient_value(variant, key),
        )
        for key in COEFFICIENT_CHANNELS
    }


def _coefficient_value(payload: object, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    preferred = payload.get(f"mean{key}")
    return _finite_number(preferred if preferred is not None else payload.get(key))


def _load_deltas(
    baseline_report: dict[str, object],
    variant_report: dict[str, object],
) -> dict[str, dict[str, float | None]]:
    baseline = baseline_report.get("aerodynamicForces")
    variant = variant_report.get("aerodynamicForces")
    return {
        key: _delta(_mapping_number(baseline, key), _mapping_number(variant, key))
        for key in LOAD_CHANNELS
    }


def _balance_deltas(
    baseline_report: dict[str, object],
    variant_report: dict[str, object],
) -> dict[str, dict[str, float | None]]:
    baseline_forces = baseline_report.get("aerodynamicForces")
    variant_forces = variant_report.get("aerodynamicForces")
    baseline = baseline_forces.get("aeroBalance") if isinstance(baseline_forces, dict) else None
    variant = variant_forces.get("aeroBalance") if isinstance(variant_forces, dict) else None
    return {
        key: _delta(_mapping_number(baseline, key), _mapping_number(variant, key))
        for key in BALANCE_CHANNELS
    }


def _overall_statistical_evidence(report: dict[str, object]) -> dict[str, object] | None:
    statistics = report.get("transientStatistics")
    overall = statistics.get("overall_evidence") if isinstance(statistics, dict) else None
    return overall if isinstance(overall, dict) else None


def _statistically_ready(report: dict[str, object]) -> bool:
    overall = _overall_statistical_evidence(report)
    return bool(
        isinstance(overall, dict)
        and overall.get("stationarity_supported") is True
        and overall.get("minimum_effective_samples_30") is True
        and overall.get("meaningful_peak_has_at_least_10_cycles") is not False
    )


def _statistical_deltas(
    baseline_report: dict[str, object],
    variant_report: dict[str, object],
) -> dict[str, dict[str, object]]:
    channels = (*COEFFICIENT_CHANNELS, "frontAeroBalancePercent")
    return {
        channel: _statistical_delta(
            _statistical_channel(baseline_report, channel),
            _statistical_channel(variant_report, channel),
        )
        for channel in channels
    }


def _statistical_channel(report: dict[str, object], channel: str) -> dict[str, object] | None:
    statistics = report.get("transientStatistics")
    channels = statistics.get("channels") if isinstance(statistics, dict) else None
    value = channels.get(channel) if isinstance(channels, dict) else None
    return value if isinstance(value, dict) else None


def _statistical_delta(
    baseline: dict[str, object] | None,
    variant: dict[str, object] | None,
) -> dict[str, object]:
    baseline_mean = _mapping_number(baseline, "mean")
    variant_mean = _mapping_number(variant, "mean")
    baseline_error = _mapping_number(baseline, "standard_error")
    variant_error = _mapping_number(variant, "standard_error")
    if baseline_mean is None or variant_mean is None:
        return {
            "baseline": baseline_mean,
            "variant": variant_mean,
            "delta": None,
            "standardError": None,
            "confidenceLower": None,
            "confidenceUpper": None,
            "statisticallyResolved": None,
            "baselineEffectiveSamples": _mapping_number(baseline, "effective_sample_count"),
            "variantEffectiveSamples": _mapping_number(variant, "effective_sample_count"),
        }
    difference = variant_mean - baseline_mean
    combined_error = (
        math.hypot(baseline_error, variant_error)
        if baseline_error is not None and variant_error is not None
        else None
    )
    margin = NormalDist().inv_cdf(0.975) * combined_error if combined_error is not None else None
    lower = difference - margin if margin is not None else None
    upper = difference + margin if margin is not None else None
    return {
        "baseline": baseline_mean,
        "variant": variant_mean,
        "delta": difference,
        "standardError": combined_error,
        "confidenceLower": lower,
        "confidenceUpper": upper,
        "statisticallyResolved": (
            bool(lower > 0 or upper < 0) if lower is not None and upper is not None else None
        ),
        "baselineEffectiveSamples": _mapping_number(baseline, "effective_sample_count"),
        "variantEffectiveSamples": _mapping_number(variant, "effective_sample_count"),
    }


def _mapping_number(payload: object, key: str) -> float | None:
    return _finite_number(payload.get(key)) if isinstance(payload, dict) else None


def _delta(baseline: float | None, variant: float | None) -> dict[str, float | None]:
    if baseline is None or variant is None:
        return {"baseline": baseline, "variant": variant, "delta": None, "percentDelta": None}
    difference = variant - baseline
    percent = difference / abs(baseline) * 100.0 if abs(baseline) > 1e-12 else None
    return {
        "baseline": baseline,
        "variant": variant,
        "delta": difference,
        "percentDelta": percent,
    }


def _mapping_differences(baseline: object, variant: object, prefix: str = "") -> list[dict[str, object]]:
    if isinstance(baseline, dict) and isinstance(variant, dict):
        differences: list[dict[str, object]] = []
        for key in sorted(set(baseline) | set(variant)):
            path = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(
                _mapping_differences(baseline.get(key), variant.get(key), path)
            )
        return differences
    if baseline == variant:
        return []
    return [{"field": prefix, "baseline": baseline, "variant": variant}]
