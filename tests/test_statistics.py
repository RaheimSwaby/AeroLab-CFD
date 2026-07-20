"""Tests for the transient statistical diagnostics.

These back the "decision-safe" verdicts, so the autocorrelation estimator is
checked against signals whose integrated autocorrelation time is known
analytically rather than against a recorded snapshot.
"""

import unittest

import numpy as np

from aerolab.solver.statistics import (
    _confidence_interval,
    _integrated_autocorrelation_time,
    _mean_standard_error,
    _sample_standard_deviation,
    _stable_mean,
    analyze_transient_history,
)


def _ar1(phi: float, count: int, seed: int) -> np.ndarray:
    """First-order autoregressive series: rho_k = phi**k."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count)
    values = np.empty(count)
    values[0] = noise[0]
    for index in range(1, count):
        values[index] = phi * values[index - 1] + noise[index]
    return values


class IntegratedAutocorrelationTimeTests(unittest.TestCase):
    def test_white_noise_is_near_one(self):
        rng = np.random.default_rng(20260720)
        tau = _integrated_autocorrelation_time(rng.standard_normal(5000))
        self.assertIsNotNone(tau)
        self.assertGreaterEqual(tau, 1.0)
        self.assertLess(tau, 3.0)

    def test_ar1_recovers_theoretical_time(self):
        # For AR(1), tau_int = (1 + phi) / (1 - phi).
        phi = 0.8
        expected = (1.0 + phi) / (1.0 - phi)  # 9.0
        tau = _integrated_autocorrelation_time(_ar1(phi, 20000, seed=7))
        self.assertIsNotNone(tau)
        # The estimator has genuine sampling variance; assert the right scale.
        self.assertGreater(tau, expected * 0.5)
        self.assertLess(tau, expected * 1.8)

    def test_stronger_correlation_gives_larger_time(self):
        weak = _integrated_autocorrelation_time(_ar1(0.3, 20000, seed=11))
        strong = _integrated_autocorrelation_time(_ar1(0.9, 20000, seed=11))
        self.assertLess(weak, strong)

    def test_constant_series_is_one(self):
        self.assertEqual(_integrated_autocorrelation_time(np.full(500, 2.5)), 1.0)

    def test_degenerate_inputs(self):
        self.assertIsNone(_integrated_autocorrelation_time(np.array([])))
        self.assertEqual(_integrated_autocorrelation_time(np.array([3.0])), 1.0)

    def test_never_exceeds_sample_count(self):
        rng = np.random.default_rng(3)
        walk = np.cumsum(rng.standard_normal(200))  # strongly correlated
        tau = _integrated_autocorrelation_time(walk)
        self.assertGreaterEqual(tau, 1.0)
        self.assertLessEqual(tau, 200.0)


class BasicStatisticsTests(unittest.TestCase):
    def test_stable_mean(self):
        self.assertAlmostEqual(_stable_mean(np.array([1.0, 2.0, 3.0, 4.0])), 2.5, places=12)

    def test_stable_mean_survives_large_magnitudes(self):
        value = _stable_mean(np.full(100, 1e150))
        self.assertAlmostEqual(value / 1e150, 1.0, places=9)

    def test_sample_std_uses_ddof_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(
            _sample_standard_deviation(values), float(np.std(values, ddof=1)), places=12
        )

    def test_sample_std_requires_two_samples(self):
        self.assertIsNone(_sample_standard_deviation(np.array([1.0])))

    def test_standard_error_divides_by_sqrt_effective_count(self):
        self.assertAlmostEqual(_mean_standard_error(2.0, 4.0), 1.0, places=12)

    def test_standard_error_guards(self):
        self.assertIsNone(_mean_standard_error(None, 4.0))
        self.assertIsNone(_mean_standard_error(2.0, 0.0))

    def test_confidence_interval_brackets_estimate(self):
        interval = _confidence_interval(10.0, 2.0, z_score=1.96, confidence=0.95)
        self.assertAlmostEqual(interval["lower"], 10.0 - 1.96 * 2.0, places=12)
        self.assertAlmostEqual(interval["upper"], 10.0 + 1.96 * 2.0, places=12)
        self.assertEqual(interval["confidence_level"], 0.95)

    def test_confidence_interval_without_standard_error(self):
        interval = _confidence_interval(10.0, None, z_score=1.96, confidence=0.95)
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])


class AnalyzeTransientHistoryTests(unittest.TestCase):
    @staticmethod
    def _rows(count: int, value: float = 0.5) -> list[dict[str, float]]:
        return [{"Time": float(index), "Cd": value} for index in range(count)]

    def test_constant_channel_reports_its_value(self):
        result = analyze_transient_history(self._rows(100), channels=("Cd",))
        channel = result["channels"]["Cd"]
        self.assertAlmostEqual(channel["mean"], 0.5, places=12)
        self.assertEqual(result["sample_counts"]["total"], 100)
        self.assertEqual(result["sample_counts"]["retained"], 100)

    def test_averaging_window_trims_to_final_window(self):
        # Latest time is 99; a 10 s window keeps t >= 89.
        result = analyze_transient_history(
            self._rows(100), channels=("Cd",), averaging_window_s=10.0
        )
        self.assertAlmostEqual(result["window"]["applied_cutoff_s"], 89.0, places=12)
        self.assertEqual(result["sample_counts"]["retained"], 11)

    def test_warmup_cutoff_discards_early_samples(self):
        result = analyze_transient_history(
            self._rows(100), channels=("Cd",), warmup_time_s=50.0
        )
        self.assertAlmostEqual(result["window"]["applied_cutoff_s"], 50.0, places=12)
        self.assertEqual(result["sample_counts"]["retained"], 50)

    def test_most_restrictive_cutoff_wins(self):
        result = analyze_transient_history(
            self._rows(100), channels=("Cd",), warmup_time_s=50.0, averaging_window_s=10.0
        )
        self.assertAlmostEqual(result["window"]["applied_cutoff_s"], 89.0, places=12)
        self.assertEqual(result["sample_counts"]["retained"], 11)

    def test_duplicate_times_are_merged(self):
        rows = self._rows(10) + [{"Time": 9.0, "Cd": 0.7}]
        result = analyze_transient_history(rows, channels=("Cd",))
        counts = result["sample_counts"]
        self.assertEqual(counts["total"], 11)
        self.assertEqual(counts["unique_time"], 10)
        self.assertGreaterEqual(counts["duplicate_time"], 1)

    def test_non_finite_times_are_rejected(self):
        rows = self._rows(10) + [{"Time": float("nan"), "Cd": 0.5}]
        result = analyze_transient_history(rows, channels=("Cd",))
        self.assertEqual(result["sample_counts"]["invalid_time"], 1)

    def test_empty_history_does_not_raise(self):
        result = analyze_transient_history([], channels=("Cd",))
        self.assertEqual(result["sample_counts"]["retained"], 0)

    def test_correlated_channel_has_fewer_effective_samples(self):
        values = _ar1(0.9, 2000, seed=5)
        rows = [{"Time": float(i), "Cd": float(v)} for i, v in enumerate(values)]
        channel = analyze_transient_history(rows, channels=("Cd",))["channels"]["Cd"]
        self.assertEqual(channel["sample_count"], 2000)
        # autocorrelation must shrink the independent-information count
        self.assertLess(channel["effective_sample_count"], 2000)
        self.assertGreater(channel["integrated_autocorrelation_time_samples"], 1.0)


if __name__ == "__main__":
    unittest.main()
