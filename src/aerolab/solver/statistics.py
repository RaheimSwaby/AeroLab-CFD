"""Statistical diagnostics for transient solver histories.

The public function in this module returns only JSON-safe scalars and mappings.
Evidence labels are deliberately conservative: failure to resolve drift is not a
proof that a finite history is stationary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist
from typing import SupportsFloat, SupportsIndex, TypeGuard

import numpy as np

_MIN_EFFECTIVE_SAMPLES = 30.0
_MIN_STATIONARITY_SAMPLES = 6
_MIN_STATIONARITY_EFFECTIVE_SAMPLES = 4.0
_MIN_SPECTRUM_SAMPLES = 8
_MIN_MEANINGFUL_PEAK_CYCLES = 3.0
_REQUIRED_PEAK_CYCLES = 10.0
_MIN_PEAK_POWER_FRACTION = 0.10
_MIN_PEAK_TO_MEDIAN_POWER = 5.0


def analyze_transient_history(
    rows: list[dict[str, float]],
    *,
    channels: tuple[str, ...],
    warmup_time_s: float | None = None,
    averaging_window_s: float | None = None,
    flow_through_time_s: float | None = None,
    reference_length_m: float | None = None,
    speed_mps: float | None = None,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Analyze the retained portion of a transient time history.

    ``warmup_time_s`` is an absolute simulation-time washout cutoff. The final
    averaging window is measured back from the latest finite, deduplicated
    ``Time`` sample. Duplicate times are merged, with the last finite value for
    each requested channel taking precedence.
    """

    requested_channels = _normalize_channels(channels)
    input_rows = _normalize_rows(rows)
    clean_rows, finite_time_count = _clean_rows(input_rows, requested_channels)

    confidence = _confidence_level(confidence_level)
    washout_cutoff = _finite_float(warmup_time_s)
    averaging_window = _positive_float(averaging_window_s)
    latest_clean_time = clean_rows[-1][0] if clean_rows else None
    averaging_cutoff = (
        _safe_difference(latest_clean_time, averaging_window)
        if latest_clean_time is not None and averaging_window is not None
        else None
    )
    active_cutoffs = [
        cutoff
        for cutoff in (washout_cutoff, averaging_cutoff)
        if cutoff is not None
    ]
    applied_cutoff = max(active_cutoffs) if active_cutoffs else None
    retained_rows = [
        item
        for item in clean_rows
        if applied_cutoff is None or item[0] >= applied_cutoff
    ]

    total_count = len(input_rows)
    unique_time_count = len(clean_rows)
    retained_count = len(retained_rows)
    duplicate_count = finite_time_count - unique_time_count
    invalid_time_count = total_count - finite_time_count
    cutoff_discarded_count = unique_time_count - retained_count

    start_time = retained_rows[0][0] if retained_rows else None
    end_time = retained_rows[-1][0] if retained_rows else None
    duration = (
        _safe_difference(end_time, start_time)
        if start_time is not None and end_time is not None
        else None
    )

    reference_length = _positive_float(reference_length_m)
    speed = _positive_float(speed_mps)
    provided_flow_time = _positive_float(flow_through_time_s)
    derived_flow_time = (
        _safe_ratio(reference_length, speed)
        if reference_length is not None and speed is not None
        else None
    )
    flow_time = provided_flow_time or derived_flow_time
    flow_time_source = (
        "provided"
        if provided_flow_time is not None
        else "reference_length_m / speed_mps"
        if derived_flow_time is not None
        else None
    )
    flow_coverage = (
        _safe_ratio(duration, flow_time)
        if duration is not None and flow_time is not None
        else None
    )

    channel_results = {
        channel: _analyze_channel(
            retained_rows,
            channel,
            confidence,
            duration,
            reference_length,
            speed,
        )
        for channel in requested_channels
    }
    overall_evidence = _overall_evidence(channel_results, requested_channels)

    return {
        "confidence_level": confidence,
        "sample_counts": {
            "total": total_count,
            "finite_time": finite_time_count,
            "unique_time": unique_time_count,
            "retained": retained_count,
            "discarded": total_count - retained_count,
            "invalid_time": invalid_time_count,
            "duplicate_time": duplicate_count,
            "discarded_by_cutoff": cutoff_discarded_count,
        },
        "window": {
            "washout_cutoff_s": washout_cutoff,
            "averaging_window_s": averaging_window,
            "averaging_window_cutoff_s": averaging_cutoff,
            "applied_cutoff_s": applied_cutoff,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": duration,
            "flow_through_time_s": flow_time,
            "flow_through_time_source": flow_time_source,
            "flow_through_coverage": flow_coverage,
        },
        "channels": channel_results,
        "overall_evidence": overall_evidence,
    }


def _is_object_iterable(value: object) -> TypeGuard[Iterable[object]]:
    return isinstance(value, Iterable)


def _normalize_rows(rows: object) -> list[object]:
    if not _is_object_iterable(rows):
        return []
    try:
        return list(rows)
    except TypeError:
        return []


def _normalize_channels(channels: object) -> tuple[str, ...]:
    candidates: tuple[object, ...]
    if isinstance(channels, str):
        candidates = (channels,)
    elif _is_object_iterable(channels):
        try:
            candidates = tuple(channels)
        except TypeError:
            candidates = ()
    else:
        candidates = ()
    normalized: list[str] = []
    seen: set[str] = set()
    for channel in candidates:
        if isinstance(channel, str) and channel not in seen:
            normalized.append(channel)
            seen.add(channel)
    return tuple(normalized)


def _clean_rows(
    rows: list[object],
    channels: tuple[str, ...],
) -> tuple[list[tuple[float, dict[str, float]]], int]:
    merged: dict[float, dict[str, float]] = {}
    finite_time_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        time = _finite_float(row.get("Time"))
        if time is None:
            continue
        finite_time_count += 1
        clean_row = merged.setdefault(time, {"Time": time})
        for channel in channels:
            value = time if channel == "Time" else _finite_float(row.get(channel))
            if value is not None:
                clean_row[channel] = value
    return sorted(merged.items()), finite_time_count


def _analyze_channel(
    rows: list[tuple[float, dict[str, float]]],
    channel: str,
    confidence: float,
    full_window_duration_s: float | None,
    reference_length_m: float | None,
    speed_mps: float | None,
) -> dict[str, object]:
    samples = [
        (time, row[channel])
        for time, row in rows
        if channel in row
    ]
    times = np.asarray([sample[0] for sample in samples], dtype=float)
    values = np.asarray([sample[1] for sample in samples], dtype=float)
    count = int(values.size)
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    mean = _stable_mean(values)
    sample_std = _sample_standard_deviation(values)
    autocorrelation_time = _integrated_autocorrelation_time(values)
    effective_count = (
        _safe_ratio(float(count), autocorrelation_time)
        if count and autocorrelation_time is not None
        else None
    )
    standard_error = _mean_standard_error(sample_std, effective_count)
    confidence_interval = _confidence_interval(mean, standard_error, z_score, confidence)

    median_interval = _median_interval(times)
    autocorrelation_time_s = (
        _safe_product(autocorrelation_time, median_interval)
        if autocorrelation_time is not None and median_interval is not None
        else None
    )
    halves = _half_diagnostics(values)
    drift = _safe_difference(halves["second_mean"], halves["first_mean"])
    drift_standard_error = _root_sum_squares(
        halves["first_standard_error"],
        halves["second_standard_error"],
    )
    drift_interval = _confidence_interval(
        drift,
        drift_standard_error,
        z_score,
        confidence,
    )

    trend = _linear_trend(
        times,
        values,
        autocorrelation_time,
        full_window_duration_s,
        confidence,
        z_score,
    )
    stationarity = _stationarity_evidence(
        count=count,
        effective_count=effective_count,
        mean=mean,
        sample_std=sample_std,
        drift=drift,
        drift_standard_error=drift_standard_error,
        full_window_trend=trend["full_window_trend"],
        full_window_trend_standard_error=trend["full_window_trend_standard_error"],
        confidence=confidence,
        z_score=z_score,
    )
    spectrum = _spectrum_diagnostics(
        times,
        values,
        reference_length_m,
        speed_mps,
    )

    return {
        "sample_count": count,
        "missing_sample_count": len(rows) - count,
        "observed_start_time_s": _safe_float(times[0]) if count else None,
        "observed_end_time_s": _safe_float(times[-1]) if count else None,
        "observed_duration_s": (
            _safe_difference(times[-1], times[0]) if count > 1 else 0.0 if count else None
        ),
        "mean": mean,
        "sample_std": sample_std,
        "integrated_autocorrelation_time_samples": autocorrelation_time,
        "integrated_autocorrelation_time_s": autocorrelation_time_s,
        "autocorrelation_method": "paired initial-positive sequence with initial-monotone adjustment",
        "effective_sample_count": effective_count,
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "first_half": {
            "sample_count": halves["first_count"],
            "mean": halves["first_mean"],
            "effective_sample_count": halves["first_effective_count"],
        },
        "second_half": {
            "sample_count": halves["second_count"],
            "mean": halves["second_mean"],
            "effective_sample_count": halves["second_effective_count"],
        },
        "half_mean_drift": drift,
        "half_mean_drift_standard_error": drift_standard_error,
        "half_mean_drift_confidence_interval": drift_interval,
        **trend,
        "stationarity_evidence": stationarity,
        "spectrum": spectrum,
    }


def _half_diagnostics(values: np.ndarray) -> dict[str, float | int | None]:
    count = int(values.size)
    if count == 0:
        first = values
        second = values
    elif count == 1:
        first = values
        second = values[:0]
    else:
        midpoint = count // 2
        first = values[:midpoint]
        second = values[midpoint:]

    first_mean = _stable_mean(first)
    second_mean = _stable_mean(second)
    first_std = _sample_standard_deviation(first)
    second_std = _sample_standard_deviation(second)
    first_tau = _integrated_autocorrelation_time(first)
    second_tau = _integrated_autocorrelation_time(second)
    first_effective = (
        _safe_ratio(float(first.size), first_tau)
        if first.size and first_tau is not None
        else None
    )
    second_effective = (
        _safe_ratio(float(second.size), second_tau)
        if second.size and second_tau is not None
        else None
    )
    return {
        "first_count": int(first.size),
        "second_count": int(second.size),
        "first_mean": first_mean,
        "second_mean": second_mean,
        "first_effective_count": first_effective,
        "second_effective_count": second_effective,
        "first_standard_error": _mean_standard_error(first_std, first_effective),
        "second_standard_error": _mean_standard_error(second_std, second_effective),
    }


def _linear_trend(
    times: np.ndarray,
    values: np.ndarray,
    autocorrelation_time: float | None,
    full_window_duration_s: float | None,
    confidence: float,
    z_score: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "linear_slope_per_s": None,
        "linear_slope_standard_error_per_s": None,
        "linear_slope_confidence_interval_per_s": _confidence_interval(
            None,
            None,
            z_score,
            confidence,
        ),
        "full_window_trend": None,
        "full_window_trend_standard_error": None,
        "full_window_trend_confidence_interval": _confidence_interval(
            None,
            None,
            z_score,
            confidence,
        ),
    }
    count = int(values.size)
    if count < 2:
        return result

    time_scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    value_scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    normalized_times = times / time_scale
    normalized_values = values / value_scale
    centered_times = normalized_times - float(np.mean(normalized_times))
    centered_values = normalized_values - float(np.mean(normalized_values))
    denominator = float(np.dot(centered_times, centered_times))
    if not math.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return result

    normalized_slope = float(np.dot(centered_times, centered_values) / denominator)
    slope = _safe_product(normalized_slope, _safe_ratio(value_scale, time_scale))
    duration = full_window_duration_s
    full_trend = (
        _safe_product(slope, duration)
        if slope is not None and duration is not None
        else None
    )

    slope_standard_error = None
    trend_standard_error = None
    if count > 2:
        fitted = float(np.mean(normalized_values)) + normalized_slope * centered_times
        residual = normalized_values - fitted
        residual_variance = float(np.dot(residual, residual) / (count - 2))
        if residual_variance >= 0.0 and math.isfinite(residual_variance):
            normalized_error = math.sqrt(residual_variance / denominator)
            if autocorrelation_time is not None:
                normalized_error *= math.sqrt(max(autocorrelation_time, 1.0))
            slope_standard_error = _safe_product(
                normalized_error,
                _safe_ratio(value_scale, time_scale),
            )
            if slope_standard_error is not None and duration is not None:
                trend_standard_error = _safe_product(slope_standard_error, duration)

    result.update(
        {
            "linear_slope_per_s": slope,
            "linear_slope_standard_error_per_s": slope_standard_error,
            "linear_slope_confidence_interval_per_s": _confidence_interval(
                slope,
                slope_standard_error,
                z_score,
                confidence,
            ),
            "full_window_trend": full_trend,
            "full_window_trend_standard_error": trend_standard_error,
            "full_window_trend_confidence_interval": _confidence_interval(
                full_trend,
                trend_standard_error,
                z_score,
                confidence,
            ),
        }
    )
    return result


def _stationarity_evidence(
    *,
    count: int,
    effective_count: float | None,
    mean: float | None,
    sample_std: float | None,
    drift: float | None,
    drift_standard_error: float | None,
    full_window_trend: object,
    full_window_trend_standard_error: object,
    confidence: float,
    z_score: float,
) -> dict[str, object]:
    scale = max(
        abs(mean) if mean is not None else 0.0,
        sample_std if sample_std is not None else 0.0,
        np.finfo(float).tiny,
    )
    tolerance = 64.0 * np.finfo(float).eps * scale
    half_drift_resolved = _statistically_resolved(
        drift,
        drift_standard_error,
        z_score,
        tolerance,
    )
    linear_trend_resolved = _statistically_resolved(
        _finite_float(full_window_trend),
        _finite_float(full_window_trend_standard_error),
        z_score,
        tolerance,
    )
    tests = [
        result
        for result in (half_drift_resolved, linear_trend_resolved)
        if result is not None
    ]
    resolved = any(tests) if tests else None
    enough_information = bool(
        count >= _MIN_STATIONARITY_SAMPLES
        and effective_count is not None
        and effective_count >= _MIN_STATIONARITY_EFFECTIVE_SAMPLES
        and tests
    )

    if resolved:
        status = "statistically_resolved_drift"
        supports_stationarity = False
        interpretation = (
            f"At least one drift diagnostic is resolved at the {confidence:.1%} confidence level; "
            "the retained window does not support stationarity."
        )
    elif enough_information:
        status = "no_statistically_resolved_drift_detected"
        supports_stationarity = True
        interpretation = (
            f"No drift diagnostic is resolved at the {confidence:.1%} confidence level. "
            "This is evidence consistent with stationarity, not proof of stationarity."
        )
    else:
        status = "insufficient_evidence"
        supports_stationarity = None
        interpretation = (
            "The retained history is too short or too correlated to assess drift reliably; "
            "stationarity is not established."
        )

    return {
        "status": status,
        "supports_stationarity": supports_stationarity,
        "statistically_resolved_drift": resolved,
        "half_mean_drift_resolved": half_drift_resolved,
        "linear_trend_resolved": linear_trend_resolved,
        "confidence_level": confidence,
        "interpretation": interpretation,
    }


def _spectrum_diagnostics(
    times: np.ndarray,
    values: np.ndarray,
    reference_length_m: float | None,
    speed_mps: float | None,
) -> dict[str, object]:
    count = int(values.size)
    result: dict[str, object] = {
        "status": "insufficient_data",
        "method": "linear interpolation to a uniform grid, linear detrending, and Hann-window FFT",
        "sample_count": count,
        "uniform_sample_count": count,
        "nonuniform_input": None,
        "sampling_interval_s": None,
        "frequency_resolution_hz": None,
        "nyquist_frequency_hz": None,
        "dominant_frequency_hz": None,
        "peak_power_fraction": None,
        "peak_to_median_power_ratio": None,
        "cycle_coverage": None,
        "strouhal_number": None,
        "meaningful_peak": False,
        "at_least_10_cycles": None,
        "meaningful_peak_thresholds": {
            "minimum_samples": _MIN_SPECTRUM_SAMPLES,
            "minimum_cycles": _MIN_MEANINGFUL_PEAK_CYCLES,
            "minimum_peak_power_fraction": _MIN_PEAK_POWER_FRACTION,
            "minimum_peak_to_median_power_ratio": _MIN_PEAK_TO_MEDIAN_POWER,
        },
    }
    if count < 2:
        return result

    duration = _safe_difference(times[-1], times[0])
    if duration is None or duration <= 0.0:
        return result
    sampling_interval = _safe_ratio(duration, float(count - 1))
    if sampling_interval is None or sampling_interval <= 0.0:
        return result

    differences = np.diff(times)
    difference_mean = float(np.mean(differences))
    nonuniform = not bool(
        np.allclose(
            differences,
            difference_mean,
            rtol=1e-3,
            atol=max(abs(difference_mean) * 1e-6, np.finfo(float).tiny),
        )
    )
    nyquist = _safe_ratio(0.5, sampling_interval)
    frequency_resolution = _safe_ratio(1.0, _safe_product(count, sampling_interval))
    result.update(
        {
            "nonuniform_input": nonuniform,
            "sampling_interval_s": sampling_interval,
            "frequency_resolution_hz": frequency_resolution,
            "nyquist_frequency_hz": nyquist,
        }
    )
    if count < _MIN_SPECTRUM_SAMPLES:
        return result

    uniform_times = np.linspace(float(times[0]), float(times[-1]), count)
    uniform_values = np.interp(uniform_times, times, values)
    value_scale = max(float(np.max(np.abs(uniform_values))), np.finfo(float).tiny)
    normalized_values = uniform_values / value_scale
    coordinates = np.arange(count, dtype=float)
    centered_coordinates = coordinates - float(np.mean(coordinates))
    coordinate_energy = float(np.dot(centered_coordinates, centered_coordinates))
    normalized_slope = (
        float(
            np.dot(
                centered_coordinates,
                normalized_values - float(np.mean(normalized_values)),
            )
            / coordinate_energy
        )
        if coordinate_energy > 0.0
        else 0.0
    )
    fitted = float(np.mean(normalized_values)) + normalized_slope * centered_coordinates
    detrended = normalized_values - fitted
    residual_amplitude = float(np.max(np.abs(detrended)))
    numerical_floor = 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(normalized_values))),
        np.finfo(float).tiny,
    )
    if not math.isfinite(residual_amplitude) or residual_amplitude <= numerical_floor:
        result["status"] = "no_resolved_spectral_energy"
        return result

    windowed = detrended / residual_amplitude * np.hanning(count)
    transform = np.fft.rfft(windowed)
    power = np.square(np.abs(transform))
    frequencies = np.fft.rfftfreq(count, d=sampling_interval)
    positive_power = power[1:]
    if positive_power.size == 0:
        return result
    total_power = float(np.sum(positive_power))
    if not math.isfinite(total_power) or total_power <= np.finfo(float).tiny:
        result["status"] = "no_resolved_spectral_energy"
        return result

    peak_offset = int(np.argmax(positive_power))
    peak_index = peak_offset + 1
    peak_power = float(power[peak_index])
    dominant_frequency = _refined_peak_frequency(frequencies, power, peak_index)
    peak_fraction = _safe_ratio(peak_power, total_power)
    median_power = float(np.median(positive_power))
    peak_to_median = (
        _safe_ratio(peak_power, median_power)
        if median_power > np.finfo(float).tiny
        else None
    )
    cycle_coverage = _safe_product(dominant_frequency, duration)
    strouhal = (
        _safe_ratio(
            _safe_product(dominant_frequency, reference_length_m),
            speed_mps,
        )
        if dominant_frequency is not None
        and reference_length_m is not None
        and speed_mps is not None
        else None
    )
    above_noise_floor = (
        peak_to_median is None or peak_to_median >= _MIN_PEAK_TO_MEDIAN_POWER
    )
    meaningful_peak = bool(
        peak_fraction is not None
        and peak_fraction >= _MIN_PEAK_POWER_FRACTION
        and cycle_coverage is not None
        and cycle_coverage >= _MIN_MEANINGFUL_PEAK_CYCLES
        and above_noise_floor
    )
    ten_cycles = (
        bool(cycle_coverage is not None and cycle_coverage >= _REQUIRED_PEAK_CYCLES)
        if meaningful_peak
        else None
    )
    result.update(
        {
            "status": "meaningful_peak" if meaningful_peak else "no_meaningful_peak",
            "dominant_frequency_hz": dominant_frequency,
            "peak_power_fraction": peak_fraction,
            "peak_to_median_power_ratio": peak_to_median,
            "cycle_coverage": cycle_coverage,
            "strouhal_number": strouhal,
            "meaningful_peak": meaningful_peak,
            "at_least_10_cycles": ten_cycles,
        }
    )
    return result


def _refined_peak_frequency(
    frequencies: np.ndarray,
    power: np.ndarray,
    peak_index: int,
) -> float | None:
    frequency = float(frequencies[peak_index])
    if 0 < peak_index < power.size - 1:
        neighborhood = power[peak_index - 1 : peak_index + 2]
        if bool(np.all(neighborhood > 0.0)):
            logarithms = np.log(neighborhood)
            denominator = float(logarithms[0] - 2.0 * logarithms[1] + logarithms[2])
            if math.isfinite(denominator) and abs(denominator) > np.finfo(float).eps:
                offset = 0.5 * float(logarithms[0] - logarithms[2]) / denominator
                offset = min(max(offset, -0.5), 0.5)
                bin_width = float(frequencies[1] - frequencies[0])
                frequency += offset * bin_width
    return _safe_float(frequency)


def _overall_evidence(
    channel_results: dict[str, dict[str, object]],
    requested_channels: tuple[str, ...],
) -> dict[str, object]:
    if not requested_channels:
        stationarity = None
        minimum_effective_samples = None
    else:
        stationarity_flags = [
            result["stationarity_evidence"]["supports_stationarity"]  # type: ignore[index]
            for result in channel_results.values()
        ]
        if any(flag is False for flag in stationarity_flags):
            stationarity = False
        elif stationarity_flags and all(flag is True for flag in stationarity_flags):
            stationarity = True
        else:
            stationarity = None

        effective_counts = [
            _finite_float(result.get("effective_sample_count"))
            for result in channel_results.values()
        ]
        if any(
            count is not None and count < _MIN_EFFECTIVE_SAMPLES
            for count in effective_counts
        ):
            minimum_effective_samples = False
        elif effective_counts and all(count is not None for count in effective_counts):
            minimum_effective_samples = True
        else:
            minimum_effective_samples = None

    meaningful_spectra = [
        result["spectrum"]
        for result in channel_results.values()
        if result["spectrum"]["meaningful_peak"]  # type: ignore[index]
    ]
    spectral_cycle_coverage = (
        all(spectrum["at_least_10_cycles"] is True for spectrum in meaningful_spectra)  # type: ignore[index]
        if meaningful_spectra
        else None
    )
    return {
        "stationarity_supported": stationarity,
        "minimum_effective_samples_30": minimum_effective_samples,
        "meaningful_spectral_peak_present": bool(meaningful_spectra),
        "meaningful_peak_has_at_least_10_cycles": spectral_cycle_coverage,
        "interpretation": (
            "True stationarity means every requested channel has no statistically resolved drift; "
            "it is evidence, not proof. Spectral cycle coverage is null when no meaningful peak exists."
        ),
    }


def _integrated_autocorrelation_time(values: np.ndarray) -> float | None:
    count = int(values.size)
    if count == 0:
        return None
    if count < 2:
        return 1.0

    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    normalized = values / scale
    centered = normalized - float(np.mean(normalized))
    energy = float(np.dot(centered, centered))
    numerical_energy_floor = 64.0 * np.finfo(float).eps**2 * count
    if not math.isfinite(energy) or energy <= numerical_energy_floor:
        return 1.0

    fft_size = 1 << (2 * count - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(transform * np.conjugate(transform), n=fft_size)[:count]
    if autocovariance[0] <= 0.0 or not np.isfinite(autocovariance[0]):
        return 1.0
    autocorrelation = autocovariance / autocovariance[0]

    previous_pair = math.inf
    pair_sum = 0.0
    for lag in range(0, count - 1, 2):
        pair = float(autocorrelation[lag] + autocorrelation[lag + 1])
        if not math.isfinite(pair) or pair <= 0.0:
            break
        monotone_pair = min(pair, previous_pair)
        pair_sum += monotone_pair
        previous_pair = monotone_pair

    estimate = -1.0 + 2.0 * pair_sum
    return float(min(float(count), max(1.0, estimate)))


def _stable_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    return _safe_product(float(np.mean(values / scale)), scale)


def _sample_standard_deviation(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    return _safe_product(float(np.std(values / scale, ddof=1)), scale)


def _mean_standard_error(
    sample_std: float | None,
    effective_count: float | None,
) -> float | None:
    if sample_std is None or effective_count is None or effective_count <= 0.0:
        return None
    return _safe_ratio(sample_std, math.sqrt(effective_count))


def _confidence_interval(
    estimate: float | None,
    standard_error: float | None,
    z_score: float,
    confidence: float,
) -> dict[str, object]:
    margin = (
        _safe_product(z_score, standard_error)
        if standard_error is not None
        else None
    )
    return {
        "confidence_level": confidence,
        "sidedness": "two-sided",
        "method": "normal approximation with autocorrelation-adjusted standard error",
        "lower": (
            _safe_difference(estimate, margin)
            if estimate is not None and margin is not None
            else None
        ),
        "upper": (
            _safe_sum(estimate, margin)
            if estimate is not None and margin is not None
            else None
        ),
    }


def _statistically_resolved(
    estimate: float | None,
    standard_error: float | None,
    z_score: float,
    tolerance: float,
) -> bool | None:
    if estimate is None or standard_error is None:
        return None
    if abs(estimate) <= tolerance:
        return False
    margin = _safe_product(z_score, standard_error)
    if margin is None:
        return None
    return bool(abs(estimate) > max(margin, tolerance))


def _median_interval(times: np.ndarray) -> float | None:
    if times.size < 2:
        return None
    differences = np.diff(times)
    positive = differences[differences > 0.0]
    if positive.size == 0:
        return None
    return _safe_float(np.median(positive))


def _root_sum_squares(first: object, second: object) -> float | None:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is None or second_number is None:
        return None
    return _safe_float(math.hypot(first_number, second_number))


def _confidence_level(value: object) -> float:
    candidate = _finite_float(value)
    return candidate if candidate is not None and 0.0 < candidate < 1.0 else 0.95


def _positive_float(value: object) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number > 0.0 else None


def _is_float_compatible(
    value: object,
) -> TypeGuard[str | bytes | bytearray | memoryview | SupportsFloat | SupportsIndex]:
    return isinstance(
        value,
        (str, bytes, bytearray, memoryview, SupportsFloat, SupportsIndex),
    )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not _is_float_compatible(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_float(value: object) -> float | None:
    return _finite_float(value)


def _safe_product(first: object, second: object) -> float | None:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is None or second_number is None:
        return None
    try:
        return _safe_float(first_number * second_number)
    except OverflowError:
        return None


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    numerator_number = _finite_float(numerator)
    denominator_number = _finite_float(denominator)
    if (
        numerator_number is None
        or denominator_number is None
        or denominator_number == 0.0
    ):
        return None
    try:
        return _safe_float(numerator_number / denominator_number)
    except (OverflowError, ZeroDivisionError):
        return None


def _safe_difference(first: object, second: object) -> float | None:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is None or second_number is None:
        return None
    try:
        return _safe_float(first_number - second_number)
    except OverflowError:
        return None


def _safe_sum(first: object, second: object) -> float | None:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is None or second_number is None:
        return None
    try:
        return _safe_float(first_number + second_number)
    except OverflowError:
        return None
