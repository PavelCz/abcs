"""
Tests for joint-coverage adaptive sampler ensuring max normalized neighbor gaps
on both axes are below a desired fraction.
"""

import os
import sys
from typing import Callable, Tuple

import numpy as np

# Ensure local src/ is importable before any installed package named abcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from abcs import JointCoverageSampler
from tests.visualization_utils_v2 import (
    initialize_test_run,
    save_joint_artifacts,
    print_artifact_summary,
)


def make_eval_callables(add_noise: bool = True, full_range: bool = False) -> Tuple[
    Callable[[float], Tuple[float, float]],
    Callable[[], Tuple[float, float]],
    Callable[[], Tuple[float, float]],
]:
    """Factory producing the three required evaluation callables.

    Underlying mapping uses a monotone threshold→AFHP function and a
    performance function that increases with AFHP. Percentiles map linearly
    to thresholds for testing.
    """

    rng = np.random.RandomState(42)

    def threshold_to_afhp(threshold: float) -> float:
        if full_range:
            afhp = threshold
        else:
            z = (threshold - 50.0) / 8.0
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            afhp = sigmoid * 95.0 + 2.5
        if add_noise:
            afhp = float(np.clip(afhp + rng.randn() * 0.5, 0.0, 100.0))
        else:
            afhp = float(np.clip(afhp, 0.0, 100.0))
        return afhp

    def afhp_to_performance(afhp: float) -> float:
        base_return = 25.0
        max_return = 90.0
        if afhp <= 0.0:
            value = base_return
        else:
            scaled = afhp / 100.0
            k = 100.0
            log_factor = np.log(1.0 + k * scaled) / np.log(1.0 + k)
            transformed = log_factor ** 0.3
            value = base_return + (max_return - base_return) * transformed
        if add_noise:
            value = float(value + rng.randn() * 0.5)
        return float(np.clip(value, base_return - 5.0, max_return + 5.0))

    def eval_at_percentile(p: float) -> Tuple[float, float]:
        threshold = float(np.clip(p, 0.0, 1.0) * 100.0)
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(afhp)
        return afhp, perf

    def eval_at_lower_extreme() -> Tuple[float, float]:
        threshold = 0.0
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(afhp)
        return afhp, perf

    def eval_at_upper_extreme() -> Tuple[float, float]:
        threshold = 100.0
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(afhp)
        return afhp, perf

    return eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme


def test_joint_coverage_meets_fraction_noise():
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=True, full_range=False)
    sampler = JointCoverageSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        coverage_fraction=0.10,
        max_total_evals=200,
    )
    initialize_test_run()
    result = sampler.run()
    artifacts = save_joint_artifacts(result.points, result, test_name="noise_fraction_0_10")
    print_artifact_summary(artifacts)
    assert result.coverage_x_max_gap <= 0.10 + 1e-9
    assert result.coverage_y_max_gap <= 0.10 + 1e-9


def test_joint_coverage_linear_full_range_tight_fraction():
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=False, full_range=True)
    sampler = JointCoverageSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        coverage_fraction=0.05,
        max_total_evals=400,
    )
    initialize_test_run()
    result = sampler.run()
    artifacts = save_joint_artifacts(result.points, result, test_name="linear_fraction_0_05")
    print_artifact_summary(artifacts)
    assert result.coverage_x_max_gap <= 0.05 + 1e-9
    assert result.coverage_y_max_gap <= 0.05 + 1e-9


def test_joint_coverage_pathological_converges():
    # Pathological AFHP mapping: slow then very steep
    rng = np.random.RandomState(7)

    def pathological_threshold_to_afhp(threshold: float) -> float:
        if threshold < 50.0:
            afhp = threshold * 0.1
        else:
            afhp = 5.0 + (threshold - 50.0) * 1.9
        return float(np.clip(afhp + rng.randn() * 0.2, 0.0, 100.0))

    def performance_from_afhp(afhp: float) -> float:
        return 20.0 + 60.0 * (afhp / 100.0) ** 3

    def eval_at_percentile(p: float) -> Tuple[float, float]:
        threshold = float(np.clip(p, 0.0, 1.0) * 100.0)
        afhp = pathological_threshold_to_afhp(threshold)
        perf = performance_from_afhp(afhp)
        return afhp, perf

    def eval_at_lower_extreme() -> Tuple[float, float]:
        afhp = pathological_threshold_to_afhp(0.0)
        return afhp, performance_from_afhp(afhp)

    def eval_at_upper_extreme() -> Tuple[float, float]:
        afhp = pathological_threshold_to_afhp(100.0)
        return afhp, performance_from_afhp(afhp)

    sampler = JointCoverageSampler(
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        coverage_fraction=0.10,
        max_total_evals=400,
    )
    initialize_test_run()
    result = sampler.run()
    artifacts = save_joint_artifacts(result.points, result, test_name="pathological_fraction_0_10")
    print_artifact_summary(artifacts)
    assert result.early_stop_reason is None
    assert result.coverage_x_max_gap <= 0.10 + 1e-9
    assert result.coverage_y_max_gap <= 0.10 + 1e-9
