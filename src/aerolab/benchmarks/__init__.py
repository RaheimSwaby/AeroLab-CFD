"""Reproducible real-solver benchmarks for AeroLab."""

from .runner import (
    BENCHMARK_UNAVAILABLE_EXIT_CODE,
    DEFAULT_BENCHMARK_ID,
    available_benchmarks,
    evaluate_benchmark,
    load_benchmark_manifest,
    run_benchmark,
)

__all__ = [
    "BENCHMARK_UNAVAILABLE_EXIT_CODE",
    "DEFAULT_BENCHMARK_ID",
    "available_benchmarks",
    "evaluate_benchmark",
    "load_benchmark_manifest",
    "run_benchmark",
]
