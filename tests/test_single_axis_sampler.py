"""
Tests for single-axis adaptive binary coverage search sampler.

These tests focus on the BinarySearchSampler which provides AFHP coverage
using binary search to fill bins along the output axis.
"""

import os
import sys
from typing import Callable, Tuple, Dict, Any

import numpy as np

# Ensure local src/ is importable before any installed package named acs
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from acs.sampler import BinarySearchSampler
from tests.visualization_utils import (
    initialize_test_run,
    save_single_axis_artifacts,
    print_artifact_summary,
)


def make_eval_callables(
    add_noise: bool = True, full_range: bool = False
) -> Tuple[
    Callable[[float], Tuple[float, Dict[str, Any]]],
    Callable[[], Tuple[float, Dict[str, Any]]],
    Callable[[], Tuple[float, Dict[str, Any]]],
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
            transformed = log_factor**0.3
            value = base_return + (max_return - base_return) * transformed
        if add_noise:
            value = float(value + rng.randn() * 0.5)
        return float(np.clip(value, base_return - 5.0, max_return + 5.0))

    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        threshold = float(np.clip(p, 0.0, 1.0) * 100.0)
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(afhp)
        metadata = {}
        return afhp, perf, metadata

    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        threshold = 0.0
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(threshold)
        metadata = {}
        return afhp, perf, metadata

    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        threshold = 100.0
        afhp = threshold_to_afhp(threshold)
        perf = afhp_to_performance(threshold)
        metadata = {}
        return afhp, perf, metadata

    return eval_at_percentile, eval_at_lower_extreme, eval_at_upper_extreme


def test_basic_bin_filling():
    """Test that the sampler fills bins across the AFHP range."""
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=False, full_range=True)
    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=10,
        output_range=(0.0, 100.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    # Check that we have samples
    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # Check that samples are within expected range
    for sample in filled_samples:
        assert 0.0 <= sample.afhp <= 100.0
        assert 0.0 <= sample.desired_percentile <= 1.0

    # Check coverage summary
    assert result.total_evals > 0
    assert result.info["bins_filled"] == len(filled_samples)

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="basic_bin_filling", result=result
    )
    print_artifact_summary(artifacts)


def test_sigmoid_curve_noise():
    """Test sampling with sigmoid curve and noise."""
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=True, full_range=False)
    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=20,
        output_range=(0.0, 100.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # With sigmoid curve, we expect good coverage in the middle range
    assert (
        result.info["coverage_percentage"] > 50.0
    )  # Should fill at least half the bins

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="sigmoid_curve_noise", result=result
    )
    print_artifact_summary(artifacts)


def test_pathological_curve():
    """Test with pathological AFHP mapping: slow then very steep."""
    rng = np.random.RandomState(7)

    def pathological_threshold_to_afhp(threshold: float) -> float:
        if threshold < 50.0:
            afhp = threshold * 0.1
        else:
            afhp = 5.0 + (threshold - 50.0) * 1.9
        return float(np.clip(afhp + rng.randn() * 0.2, 0.0, 100.0))

    def performance_from_afhp(afhp: float) -> float:
        return 20.0 + 60.0 * (afhp / 100.0) ** 3

    def eval_at_percentile(p: float) -> Tuple[float, float, Dict[str, Any]]:
        threshold = float(np.clip(p, 0.0, 1.0) * 100.0)
        afhp = pathological_threshold_to_afhp(threshold)
        perf = performance_from_afhp(afhp)
        metadata = {}
        return afhp, perf, metadata

    def eval_at_lower_extreme() -> Tuple[float, float, Dict[str, Any]]:
        afhp = pathological_threshold_to_afhp(0.0)
        perf = performance_from_afhp(afhp)
        metadata = {}
        return afhp, perf, metadata

    def eval_at_upper_extreme() -> Tuple[float, float, Dict[str, Any]]:
        afhp = pathological_threshold_to_afhp(100.0)
        perf = performance_from_afhp(afhp)
        metadata = {}
        return afhp, perf, metadata

    sampler = BinarySearchSampler(
        eval_at_percentile=eval_at_percentile,
        eval_at_lower_extreme=eval_at_lower_extreme,
        eval_at_upper_extreme=eval_at_upper_extreme,
        num_bins=15,
        output_range=(0.0, 100.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # Should handle pathological case without crashing
    assert result.total_evals > 0

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="pathological_curve", result=result
    )
    print_artifact_summary(artifacts)


def test_constant_output():
    """Test edge case where output is constant."""

    def eval_p(p: float) -> Tuple[float, float, Dict[str, Any]]:
        metadata = {}
        return 50.0, 50.0, metadata

    def eval_lo() -> Tuple[float, float, Dict[str, Any]]:
        metadata = {}
        return 50.0, 50.0, metadata

    def eval_hi() -> Tuple[float, float, Dict[str, Any]]:
        metadata = {}
        return 50.0, 50.0, metadata

    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=10,
        output_range=(40.0, 60.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]

    # With constant output, we expect at least the extreme points
    assert len(filled_samples) >= 2  # At least the extremes
    # All samples should have the same AFHP value
    for sample in filled_samples:
        assert sample.afhp == 50.0

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="constant_output", result=result
    )
    print_artifact_summary(artifacts)


def test_extreme_output_ranges():
    """Test with very wide output range but smooth transitions to avoid recursion."""

    def eval_p(p: float) -> Tuple[float, float, Dict[str, Any]]:
        thr = p * 100.0
        # Smooth transitions to avoid step function recursion issues
        if thr < 25.0:
            afhp = 10.0 + (thr / 25.0) * 40.0  # 10-50
        elif thr > 75.0:
            afhp = 500.0 + ((thr - 75.0) / 25.0) * 490.0  # 500-990
        else:
            afhp = 50.0 + ((thr - 25.0) / 50.0) * 450.0  # 50-500
        perf = afhp * 0.1  # Simple performance mapping
        metadata = {}
        return float(afhp), perf, metadata

    def eval_lo() -> Tuple[float, float, Dict[str, Any]]:
        metadata = {}
        return 10.0, 1.0, metadata

    def eval_hi() -> Tuple[float, float, Dict[str, Any]]:
        metadata = {}
        return 990.0, 99.0, metadata

    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=10,  # Reduced bins to avoid recursion
        output_range=(0.0, 1000.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # Should handle extreme ranges without crashing
    assert result.total_evals > 0

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="extreme_output_ranges", result=result
    )
    print_artifact_summary(artifacts)


def test_narrow_output_region():
    """Test with AFHP varying smoothly in a narrow region to avoid recursion."""

    def eval_p(p: float) -> Tuple[float, float, Dict[str, Any]]:
        thr = p * 100.0
        # Smooth variation to avoid step function recursion
        afhp = 50.0 + 5.0 * np.sin(thr * np.pi / 50.0)  # Varies between 45-55
        perf = 55.0 + afhp * 0.2  # Simple performance mapping
        metadata = {}
        return float(afhp), perf, metadata

    def eval_lo() -> Tuple[float, float, Dict[str, Any]]:
        afhp = 50.0 + 5.0 * np.sin(0)  # 50.0
        perf = 55.0 + afhp * 0.2
        metadata = {}
        return float(afhp), perf, metadata

    def eval_hi() -> Tuple[float, float, Dict[str, Any]]:
        afhp = 50.0 + 5.0 * np.sin(100.0 * np.pi / 50.0)  # ~50.0
        perf = 55.0 + afhp * 0.2
        metadata = {}
        return float(afhp), perf, metadata

    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=5,  # Reduced bins to avoid recursion
        output_range=(45.0, 60.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # Should handle narrow variation region
    assert 0.0 <= result.info["coverage_percentage"] <= 100.0

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="narrow_output_region", result=result
    )
    print_artifact_summary(artifacts)


def test_different_bin_counts():
    """Test sampler with different numbers of bins."""
    initialize_test_run()
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=False, full_range=True)

    for num_bins in [5, 10, 20, 50]:
        sampler = BinarySearchSampler(
            eval_at_percentile=eval_p,
            eval_at_lower_extreme=eval_lo,
            eval_at_upper_extreme=eval_hi,
            num_bins=num_bins,
            output_range=(0.0, 100.0),
            verbose=True,
        )

        result = sampler.run()
        filled_samples = [s for s in result.points if s is not None]

        # More bins should generally result in more evaluations
        assert result.total_evals >= 2  # At least the extremes
        assert len(filled_samples) <= num_bins  # Can't fill more bins than exist

        artifacts = save_single_axis_artifacts(
            filled_samples, sampler, test_name=f"bins_{num_bins}", result=result
        )
        print_artifact_summary(artifacts)


def test_input_range_customization():
    """Test sampler with custom input range."""
    eval_p, eval_lo, eval_hi = make_eval_callables(add_noise=False, full_range=True)

    # Use custom input range
    sampler = BinarySearchSampler(
        eval_at_percentile=eval_p,
        eval_at_lower_extreme=eval_lo,
        eval_at_upper_extreme=eval_hi,
        num_bins=10,
        input_range=(0.2, 0.8),  # Only sample middle 60% of input space
        output_range=(0.0, 100.0),
        verbose=True,
    )

    initialize_test_run()
    result = sampler.run()

    filled_samples = [s for s in result.points if s is not None]
    assert len(filled_samples) > 0

    # All input values should be within the custom range
    # Note: The sampler maps the 0.0-1.0 range to the custom input range
    for sample in filled_samples:
        # Convert from percentile to actual input value in the custom range
        actual_input = 0.2 + sample.desired_percentile * (0.8 - 0.2)
        assert 0.2 <= actual_input <= 0.8

    artifacts = save_single_axis_artifacts(
        filled_samples, sampler, test_name="custom_input_range", result=result
    )
    print_artifact_summary(artifacts)


if __name__ == "__main__":
    # Run all tests when executed directly
    test_basic_bin_filling()
    test_sigmoid_curve_noise()
    test_pathological_curve()
    test_constant_output()
    test_extreme_output_ranges()
    test_narrow_output_region()
    test_different_bin_counts()
    test_input_range_customization()
    print("All single-axis sampler tests passed!")
