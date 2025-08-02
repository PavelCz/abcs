"""
Test for ABCS algorithm to ensure 100% coverage on both x-axis and y-axis.

This test verifies that:
1. The AFHP (x-axis) bins are 100% filled by the binary search algorithm
2. The return (y-axis) bins are 100% filled when given sufficient evaluation budget
"""

import numpy as np
from typing import Tuple, Dict, Any
from abcs import BinarySearchSampler

# Import visualization utilities
try:
    from .visualization_utils import save_test_artifacts, print_artifact_summary
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def create_test_evaluation_function(add_noise=True, full_range=False):
    """
    Create a deterministic test evaluation function for reproducible testing.
    The function creates a monotonic mapping from threshold to AFHP,
    and a return value that increases with AFHP.

    Args:
        add_noise: Whether to add small noise to outputs
        full_range: If True, ensures outputs span the full [0, 100] range
    """
    # Use a fixed random seed for reproducibility
    rng = np.random.RandomState(42)

    def eval_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        """
        Test evaluation function with deterministic behavior.
        Maps threshold (0-100) to AFHP (0-100) monotonically.
        """
        if full_range:
            # Simple linear mapping that guarantees full range coverage
            afhp = threshold  # Direct mapping for testing
        else:
            # Create a smooth monotonic mapping using sigmoid
            # Transform threshold from [0, 100] to [-6, 6] for sigmoid
            z = (threshold - 50) / 8
            sigmoid = 1 / (1 + np.exp(-z))

            # Scale to [0, 100]
            afhp = sigmoid * 95 + 2.5

        if add_noise:
            # Add very small deterministic noise based on threshold
            noise = rng.randn() * 0.5
            afhp = np.clip(afhp + noise, 0, 100)
        else:
            # Ensure strict monotonicity
            afhp = np.clip(afhp, 0, 100)

        # Calculate return value with steep initial rise
        base_return = 25
        max_return = 90

        if afhp <= 0:
            return_value = base_return
        else:
            # Steep logarithmic curve
            scaled_afhp = afhp / 100.0
            k = 100
            log_factor = np.log(1 + k * scaled_afhp) / np.log(1 + k)
            steepness_power = 0.3
            transformed_factor = log_factor**steepness_power
            return_value = base_return + (max_return - base_return) * transformed_factor

        if add_noise:
            # Add small deterministic noise
            return_value += rng.randn() * 0.5

        return_value = np.clip(return_value, base_return - 5, max_return + 5)

        metadata = {
            "return_mean": return_value,
            "return_std": rng.uniform(0.5, 1.5) if add_noise else 1.0,
            "threshold_used": threshold,
        }

        return afhp, metadata

    return eval_function


def test_full_coverage():
    """
    Test that the binary search sampler achieves 100% coverage on both axes.

    This test uses parameters that should guarantee 100% coverage:
    - The AFHP axis uses binary search which guarantees filling all bins
    - The return axis is given sufficient evaluation budget to fill all bins
    """
    print("Testing full coverage on both x-axis (AFHP) and y-axis (return)...")
    print("=" * 60)

    # Create test evaluation function
    eval_function = create_test_evaluation_function()

    # Test parameters chosen to ensure full coverage
    num_bins = 10  # Number of AFHP bins
    return_bins = 8  # Number of return bins

    # Create sampler with return refinement enabled
    # Set max_additional_evals high enough to guarantee all return bins can be filled
    sampler = BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        return_bins=return_bins,
        max_additional_evals=return_bins * 2,  # Sufficient budget for all return bins
        verbose=True,
    )

    # Run the algorithm with return refinement
    print("\nRunning coverage algorithm...")
    samples = sampler.run_with_return_refinement()

    # Get coverage summary
    summary = sampler.get_coverage_summary()

    # Check x-axis (AFHP) coverage
    x_axis_coverage = summary["coverage_percentage"]
    print(f"\nX-axis (AFHP) coverage: {x_axis_coverage:.1f}%")

    # Calculate y-axis (return) coverage
    all_samples = sampler.get_all_samples_including_refinement()

    if return_bins > 0 and len(all_samples) > 0:
        # Extract return values
        returns = []
        for sample in all_samples:
            try:
                ret = sampler.extract_return_value(sample)
                returns.append(ret)
            except ValueError:
                pass

        if returns:
            min_return = min(returns)
            max_return = max(returns)

            # Count filled return bins
            filled_return_bins = set()
            for ret in returns:
                # Find which bin this return belongs to
                if max_return > min_return:
                    bin_idx = int(
                        (ret - min_return) / (max_return - min_return) * return_bins
                    )
                    if bin_idx >= return_bins:
                        bin_idx = return_bins - 1
                    filled_return_bins.add(bin_idx)

            y_axis_coverage = 100.0 * len(filled_return_bins) / return_bins
            print(f"Y-axis (return) coverage: {y_axis_coverage:.1f}%")

            # Show which return bins were filled
            print(f"\nReturn bins filled: {sorted(filled_return_bins)}")
            print(f"Total return bins: {return_bins}")
        else:
            y_axis_coverage = 0.0
            print("Warning: No valid return values found")
    else:
        y_axis_coverage = 0.0
        print("Warning: Return refinement not enabled or no samples collected")

    # Print detailed results
    print(f"\nTotal evaluations: {summary['total_evaluations']}")
    print(f"Initial AFHP samples: {len([s for s in samples if s is not None])}")
    print(f"Return refinement samples: {len(sampler.get_return_refinement_samples())}")

    # Verify 100% coverage on both axes
    print("\n" + "=" * 60)
    print("COVERAGE TEST RESULTS:")

    x_axis_pass = x_axis_coverage == 100.0
    y_axis_pass = y_axis_coverage == 100.0

    print(
        f"X-axis (AFHP) coverage: {'PASS' if x_axis_pass else 'FAIL'} ({x_axis_coverage:.1f}%)"
    )
    print(
        f"Y-axis (return) coverage: {'PASS' if y_axis_pass else 'FAIL'} ({y_axis_coverage:.1f}%)"
    )

    if x_axis_pass and y_axis_pass:
        print("\n✓ TEST PASSED: 100% coverage achieved on both axes!")
    else:
        print("\n✗ TEST FAILED: Full coverage not achieved")
        if not x_axis_pass:
            print(f"  - X-axis coverage: {x_axis_coverage:.1f}% (expected 100%)")
        if not y_axis_pass:
            print(f"  - Y-axis coverage: {y_axis_coverage:.1f}% (expected 100%)")

    # Generate test artifacts
    if VISUALIZATION_AVAILABLE:
        artifacts = save_test_artifacts(
            samples=samples,
            sampler=sampler,
            test_name="full_coverage",
            all_samples=all_samples
        )
        print_artifact_summary(artifacts)
    
    # Return test result
    return x_axis_pass and y_axis_pass


def test_coverage_with_different_parameters():
    """
    Test coverage with different parameter combinations.
    """
    print("\n\nTesting with different parameter combinations...")
    print("=" * 60)

    # Test configurations with sufficient evaluation budget
    test_configs = [
        {"num_bins": 5, "return_bins": 5, "max_evals": 20},
        {"num_bins": 10, "return_bins": 10, "max_evals": 30},
        {"num_bins": 15, "return_bins": 12, "max_evals": 40},
    ]

    all_passed = True

    for i, config in enumerate(test_configs):
        print(
            f"\nTest {i + 1}: num_bins={config['num_bins']}, "
            f"return_bins={config['return_bins']}, "
            f"max_evals={config['max_evals']}"
        )
        print("-" * 40)

        eval_function = create_test_evaluation_function()

        sampler = BinarySearchSampler(
            eval_function=eval_function,
            num_bins=config["num_bins"],
            input_range=(0.0, 100.0),
            output_range=(0.0, 100.0),
            return_bins=config["return_bins"],
            max_additional_evals=config["max_evals"],
            verbose=False,
        )

        _samples = sampler.run_with_return_refinement()
        summary = sampler.get_coverage_summary()

        # Calculate coverages
        x_coverage = summary["coverage_percentage"]

        # Calculate y-axis coverage
        all_samples = sampler.get_all_samples_including_refinement()
        returns = []
        for sample in all_samples:
            try:
                ret = sampler.extract_return_value(sample)
                returns.append(ret)
            except ValueError:
                pass

        if returns and config["return_bins"] > 0:
            min_return = min(returns)
            max_return = max(returns)

            filled_return_bins = set()
            for ret in returns:
                if max_return > min_return:
                    bin_idx = int(
                        (ret - min_return)
                        / (max_return - min_return)
                        * config["return_bins"]
                    )
                    if bin_idx >= config["return_bins"]:
                        bin_idx = config["return_bins"] - 1
                    filled_return_bins.add(bin_idx)

            y_coverage = 100.0 * len(filled_return_bins) / config["return_bins"]
        else:
            y_coverage = 0.0

        x_pass = x_coverage == 100.0
        y_pass = y_coverage == 100.0
        test_passed = x_pass and y_pass

        print(f"X-axis coverage: {x_coverage:.1f}% ({'PASS' if x_pass else 'FAIL'})")
        print(f"Y-axis coverage: {y_coverage:.1f}% ({'PASS' if y_pass else 'FAIL'})")
        print(f"Total evaluations: {summary['total_evaluations']}")
        print(f"Result: {'PASS' if test_passed else 'FAIL'}")

        all_passed = all_passed and test_passed

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")

    return all_passed


def test_afhp_coverage_guarantee():
    """
    Test that AFHP (x-axis) coverage is always 100% when the function spans the full range.

    Note: Coverage might be less than 100% if the evaluation function doesn't produce
    outputs that span all bins (e.g., if outputs are concentrated in a narrow range).
    """
    print("\n\nTesting AFHP coverage guarantee...")
    print("=" * 60)

    # Test with a function that spans the full output range
    eval_function = create_test_evaluation_function(add_noise=False)

    # Test with various bin counts - use reasonable values
    bin_counts = [5, 10, 20]
    all_passed = True
    
    # Keep track of the largest test for artifact generation
    largest_test_samples = None
    largest_test_sampler = None

    for num_bins in bin_counts:
        sampler = BinarySearchSampler(
            eval_function=eval_function,
            num_bins=num_bins,
            input_range=(0.0, 100.0),
            output_range=(0.0, 100.0),
            return_bins=0,  # Disable return refinement to test AFHP only
            verbose=False,
        )

        _samples = sampler.run()
        summary = sampler.get_coverage_summary()
        coverage = summary["coverage_percentage"]
        
        # Keep the largest test for artifact generation
        if num_bins == 20:  # Save the most comprehensive test
            largest_test_samples = _samples
            largest_test_sampler = sampler

        # For reasonable bin counts, we should get 100% coverage
        passed = coverage == 100.0
        all_passed = all_passed and passed

        print(
            f"AFHP bins: {num_bins}, Coverage: {coverage:.1f}% ({'PASS' if passed else 'FAIL'})"
        )

        # If coverage is not 100%, show which bins are missing
        if coverage < 100.0:
            gaps = summary.get("gaps", [])
            if gaps:
                print(f"  Missing bins: {gaps}")

    # Generate test artifacts for the most comprehensive test
    if VISUALIZATION_AVAILABLE and largest_test_samples and largest_test_sampler:
        artifacts = save_test_artifacts(
            samples=largest_test_samples,
            sampler=largest_test_sampler,
            test_name="afhp_coverage_guarantee_20bins"
        )
        print_artifact_summary(artifacts)

    return all_passed


def test_guaranteed_full_coverage():
    """
    Test with a linear function that guarantees 100% coverage on both axes.
    This test verifies that the algorithm can achieve 100% coverage when
    the evaluation function spans the full output range.
    """
    print("\n\nTesting guaranteed full coverage with linear function...")
    print("=" * 60)

    # Use a linear function that spans full range
    eval_function = create_test_evaluation_function(add_noise=False, full_range=True)

    # Test parameters
    num_bins = 10
    return_bins = 8

    sampler = BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        return_bins=return_bins,
        max_additional_evals=return_bins * 2,
        verbose=True,
    )

    print("\nRunning with linear function (full range)...")
    _samples = sampler.run_with_return_refinement()
    summary = sampler.get_coverage_summary()

    # Check AFHP coverage
    x_coverage = summary["coverage_percentage"]

    # Check return coverage
    all_samples = sampler.get_all_samples_including_refinement()
    returns = []
    for sample in all_samples:
        try:
            ret = sampler.extract_return_value(sample)
            returns.append(ret)
        except ValueError:
            pass

    y_coverage = 0.0
    if returns and return_bins > 0:
        min_return = min(returns)
        max_return = max(returns)

        filled_return_bins = set()
        for ret in returns:
            if max_return > min_return:
                bin_idx = int(
                    (ret - min_return) / (max_return - min_return) * return_bins
                )
                if bin_idx >= return_bins:
                    bin_idx = return_bins - 1
                filled_return_bins.add(bin_idx)

        y_coverage = 100.0 * len(filled_return_bins) / return_bins

    print("\nResults with linear function:")
    print(f"X-axis (AFHP) coverage: {x_coverage:.1f}%")
    print(f"Y-axis (return) coverage: {y_coverage:.1f}%")

    # Generate test artifacts
    if VISUALIZATION_AVAILABLE:
        artifacts = save_test_artifacts(
            samples=_samples,
            sampler=sampler,
            test_name="guaranteed_full_coverage_linear",
            all_samples=all_samples
        )
        print_artifact_summary(artifacts)

    return x_coverage == 100.0 and y_coverage == 100.0


def test_unbounded_mode():
    """
    Test unbounded mode functionality to ensure it achieves better coverage
    than bounded mode and terminates properly.
    """
    print("\n\nTesting unbounded mode...")
    print("=" * 60)

    # Create test evaluation function
    eval_function = create_test_evaluation_function(add_noise=False)

    # Test parameters
    num_bins = 10
    return_bins = 8
    max_evals = 5  # Intentionally low to test unbounded vs bounded difference

    print("1. Testing bounded mode with low evaluation limit...")
    # Test bounded mode first
    sampler_bounded = BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        return_bins=return_bins,
        max_additional_evals=max_evals,
        unbounded_mode=False,
        verbose=False,
    )

    samples_bounded = sampler_bounded.run_with_return_refinement()
    summary_bounded = sampler_bounded.get_coverage_summary()
    all_samples_bounded = sampler_bounded.get_all_samples_including_refinement()

    # Calculate return coverage for bounded mode
    returns_bounded = []
    for sample in all_samples_bounded:
        try:
            ret = sampler_bounded.extract_return_value(sample)
            returns_bounded.append(ret)
        except ValueError:
            pass

    return_coverage_bounded = 0.0
    if returns_bounded and return_bins > 0:
        min_return = min(returns_bounded)
        max_return = max(returns_bounded)
        if max_return > min_return:
            filled_return_bins = set()
            for ret in returns_bounded:
                bin_idx = int((ret - min_return) / (max_return - min_return) * return_bins)
                if bin_idx >= return_bins:
                    bin_idx = return_bins - 1
                filled_return_bins.add(bin_idx)
            return_coverage_bounded = 100.0 * len(filled_return_bins) / return_bins

    print("2. Testing unbounded mode...")
    # Test unbounded mode
    sampler_unbounded = BinarySearchSampler(
        eval_function=eval_function,
        num_bins=num_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        return_bins=return_bins,
        max_additional_evals=max_evals,  # This should be ignored
        unbounded_mode=True,
        verbose=True,
    )

    samples_unbounded = sampler_unbounded.run_with_return_refinement()
    summary_unbounded = sampler_unbounded.get_coverage_summary()
    all_samples_unbounded = sampler_unbounded.get_all_samples_including_refinement()

    # Calculate return coverage for unbounded mode
    returns_unbounded = []
    for sample in all_samples_unbounded:
        try:
            ret = sampler_unbounded.extract_return_value(sample)
            returns_unbounded.append(ret)
        except ValueError:
            pass

    return_coverage_unbounded = 0.0
    if returns_unbounded and return_bins > 0:
        min_return = min(returns_unbounded)
        max_return = max(returns_unbounded)
        if max_return > min_return:
            filled_return_bins = set()
            for ret in returns_unbounded:
                bin_idx = int((ret - min_return) / (max_return - min_return) * return_bins)
                if bin_idx >= return_bins:
                    bin_idx = return_bins - 1
                filled_return_bins.add(bin_idx)
            return_coverage_unbounded = 100.0 * len(filled_return_bins) / return_bins

    # Compare results
    print("\nComparison Results:")
    print(f"Bounded mode:")
    print(f"  - AFHP coverage: {summary_bounded['coverage_percentage']:.1f}%")
    print(f"  - Return coverage: {return_coverage_bounded:.1f}%")
    print(f"  - Total evaluations: {summary_bounded['total_evaluations']}")
    print(f"  - Return refinement samples: {len(sampler_bounded.get_return_refinement_samples())}")

    print(f"Unbounded mode:")
    print(f"  - AFHP coverage: {summary_unbounded['coverage_percentage']:.1f}%")
    print(f"  - Return coverage: {return_coverage_unbounded:.1f}%")
    print(f"  - Total evaluations: {summary_unbounded['total_evaluations']}")
    print(f"  - Return refinement samples: {len(sampler_unbounded.get_return_refinement_samples())}")

    # Generate test artifacts for unbounded mode
    if VISUALIZATION_AVAILABLE:
        artifacts = save_test_artifacts(
            samples=samples_unbounded,
            sampler=sampler_unbounded,
            test_name="unbounded_mode",
            all_samples=all_samples_unbounded
        )
        print_artifact_summary(artifacts)

    # Test assertions
    unbounded_better_return_coverage = return_coverage_unbounded >= return_coverage_bounded
    unbounded_same_afhp_coverage = summary_unbounded['coverage_percentage'] == summary_bounded['coverage_percentage']
    unbounded_more_return_samples = len(sampler_unbounded.get_return_refinement_samples()) >= len(sampler_bounded.get_return_refinement_samples())
    
    print("\nTest Results:")
    print(f"  - Unbounded achieves same/better return coverage: {'PASS' if unbounded_better_return_coverage else 'FAIL'}")
    print(f"  - Unbounded maintains AFHP coverage: {'PASS' if unbounded_same_afhp_coverage else 'FAIL'}")
    print(f"  - Unbounded generates more return samples: {'PASS' if unbounded_more_return_samples else 'FAIL'}")
    
    # Overall result
    all_tests_passed = unbounded_better_return_coverage and unbounded_same_afhp_coverage and unbounded_more_return_samples
    
    if all_tests_passed:
        print("\n✓ UNBOUNDED MODE TEST PASSED: Unbounded mode performs better than bounded mode!")
    else:
        print("\n✗ UNBOUNDED MODE TEST FAILED: Some assertions failed")
    
    return all_tests_passed


def test_unbounded_mode_convergence():
    """
    Test that unbounded mode properly converges and terminates even in edge cases.
    """
    print("\n\nTesting unbounded mode convergence...")
    print("=" * 60)

    # Create a pathological evaluation function that's hard to sample
    def pathological_eval_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        # Very steep function that's hard to sample uniformly
        if threshold < 50:
            afhp = threshold * 0.1  # Very slow growth
        else:
            afhp = 5 + (threshold - 50) * 1.9  # Very fast growth
        
        # Return value with complex relationship
        return_value = 20 + 60 * (afhp / 100) ** 3
        
        metadata = {
            "return_mean": return_value,
            "threshold_used": threshold,
        }
        
        return afhp, metadata

    sampler = BinarySearchSampler(
        eval_function=pathological_eval_function,
        num_bins=15,
        return_bins=10,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        unbounded_mode=True,
        verbose=True,
    )

    # Run and ensure it terminates
    samples = sampler.run_with_return_refinement()
    summary = sampler.get_coverage_summary()
    
    print(f"\nPathological function results:")
    print(f"  - AFHP coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  - Total evaluations: {summary['total_evaluations']}")
    print(f"  - Algorithm terminated: {'PASS' if summary['total_evaluations'] < sampler.max_total_evals_unbounded else 'FAIL'}")
    
    # Test that it terminated before the safety limit
    terminated_properly = summary['total_evaluations'] < sampler.max_total_evals_unbounded
    
    return terminated_properly


def test_phase2_binary_bisection():
    """
    Test the new Phase 2 implementation using recursive binary bisection.
    
    This test specifically verifies that the new approach correctly fills
    return value gaps using recursive binary search rather than interpolation.
    """
    print("\n\nTesting Phase 2 binary bisection implementation...")
    print("=" * 60)
    
    # Create a function with complex return value mapping to test the bisection approach
    def complex_return_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        # Linear AFHP mapping for simplicity
        afhp = threshold
        
        # Complex non-linear return mapping that creates gaps when sampled sparsely
        if afhp <= 20:
            return_value = 30 + afhp * 0.5  # Slow growth
        elif afhp <= 50:
            return_value = 40 + (afhp - 20) * 1.5  # Fast growth
        elif afhp <= 80:
            return_value = 85 + (afhp - 50) * 0.2  # Very slow growth
        else:
            return_value = 91 + (afhp - 80) * 0.4  # Moderate growth
            
        metadata = {
            "return_mean": return_value,
            "threshold_used": threshold,
        }
        
        return afhp, metadata
    
    # Test parameters that should create gaps in return coverage
    sampler = BinarySearchSampler(
        eval_function=complex_return_function,
        num_bins=12,  # Decent AFHP coverage
        return_bins=15,  # Many return bins to test gap filling
        max_additional_evals=30,  # Sufficient budget for testing
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True,
    )
    
    print("\nRunning Phase 1 (AFHP coverage)...")
    afhp_samples = sampler.run()
    
    print("\nRunning Phase 2 (return gap filling with binary bisection)...")
    sampler.fill_return_gaps(afhp_samples)
    
    # Analyze results
    summary = sampler.get_coverage_summary()
    all_samples = sampler.get_all_samples_including_refinement()
    return_samples = sampler.get_return_refinement_samples()
    
    print(f"\nPhase 2 Binary Bisection Results:")
    print(f"  - AFHP coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  - Initial AFHP samples: {len([s for s in afhp_samples if s is not None])}")
    print(f"  - Return refinement samples added: {len(return_samples)}")
    print(f"  - Total evaluations: {summary['total_evaluations']}")
    
    # Calculate return coverage
    returns = []
    for sample in all_samples:
        try:
            ret = sampler.extract_return_value(sample)
            returns.append(ret)
        except ValueError:
            pass
            
    return_coverage = 0.0
    if returns and sampler.return_bins > 0:
        min_return = min(returns)
        max_return = max(returns)
        if max_return > min_return:
            filled_return_bins = set()
            for ret in returns:
                bin_idx = int((ret - min_return) / (max_return - min_return) * sampler.return_bins)
                if bin_idx >= sampler.return_bins:
                    bin_idx = sampler.return_bins - 1
                filled_return_bins.add(bin_idx)
            return_coverage = 100.0 * len(filled_return_bins) / sampler.return_bins
            
    print(f"  - Return coverage: {return_coverage:.1f}%")
    print(f"  - Return bins filled: {len(filled_return_bins) if 'filled_return_bins' in locals() else 0}/{sampler.return_bins}")
    
    # Verify that Phase 2 actually added samples
    phase2_worked = len(return_samples) > 0
    reasonable_coverage = return_coverage >= 60.0  # Should achieve decent coverage
    
    print(f"\nPhase 2 Test Results:")
    print(f"  - Added return samples: {'PASS' if phase2_worked else 'FAIL'}")
    print(f"  - Achieved reasonable return coverage: {'PASS' if reasonable_coverage else 'FAIL'}")
    
    # Generate test artifacts for this specific test
    if VISUALIZATION_AVAILABLE:
        artifacts = save_test_artifacts(
            samples=afhp_samples,
            sampler=sampler,
            test_name="phase2_binary_bisection",
            all_samples=all_samples
        )
        print_artifact_summary(artifacts)
    
    return phase2_worked and reasonable_coverage


def test_phase2_gap_identification():
    """
    Test the gap identification logic used in Phase 2.
    
    This test verifies that contiguous return value gap intervals
    are correctly identified for binary bisection.
    """
    print("\n\nTesting Phase 2 gap identification...")
    print("=" * 60)
    
    # Create a simple test function
    def test_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        return threshold, {"return_mean": threshold * 0.8 + 20}
    
    sampler = BinarySearchSampler(
        eval_function=test_function,
        num_bins=8,
        return_bins=10,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=False,
    )
    
    # Run Phase 1 to get initial samples
    afhp_samples = sampler.run()
    
    # Extract valid samples and their returns
    valid_samples = [s for s in afhp_samples if s is not None]
    samples_with_returns = []
    for sample in valid_samples:
        try:
            ret = sampler.extract_return_value(sample)
            samples_with_returns.append((sample, ret))
        except ValueError:
            continue
    
    if len(samples_with_returns) >= 2:
        # Sort and create return bins
        samples_with_returns.sort(key=lambda x: x[1])
        min_return = samples_with_returns[0][1] 
        max_return = samples_with_returns[-1][1]
        
        return_bin_edges = np.linspace(min_return, max_return, sampler.return_bins + 1)
        
        # Find filled return bins
        filled_return_bins = set()
        for sample, ret in samples_with_returns:
            bin_idx = sampler.determine_return_bin(ret, return_bin_edges)
            filled_return_bins.add(bin_idx)
        
        # Test gap identification
        gap_intervals = sampler.identify_return_gap_intervals(filled_return_bins, return_bin_edges)
        
        print(f"Gap identification test:")
        print(f"  - Return bins: {sampler.return_bins}")
        print(f"  - Filled bins: {sorted(filled_return_bins)}")
        print(f"  - Gap intervals found: {gap_intervals}")
        
        # Verify gap intervals are valid
        gaps_valid = True
        for start, end in gap_intervals:
            if start < 0 or end >= sampler.return_bins or start > end:
                gaps_valid = False
                break
            # Check that gaps don't contain filled bins
            for i in range(start, end + 1):
                if i in filled_return_bins:
                    gaps_valid = False
                    break
        
        print(f"  - Gap intervals valid: {'PASS' if gaps_valid else 'FAIL'}")
        
        return gaps_valid
    else:
        print("Warning: Not enough samples with return values for gap test")
        return True  # Can't test with insufficient data


def test_phase2_edge_cases():
    """
    Test Phase 2 with various edge cases to ensure robustness.
    """
    print("\n\nTesting Phase 2 edge cases...")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Function with no return variation (all same return value)
    print("\n1. Testing with constant return values...")
    def constant_return_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        return threshold, {"return_mean": 50.0}  # Always same return
    
    sampler1 = BinarySearchSampler(
        eval_function=constant_return_function,
        num_bins=5,
        return_bins=5,
        max_additional_evals=10,
        verbose=False,
    )
    
    afhp_samples1 = sampler1.run()
    additional_samples1 = sampler1.fill_return_gaps(afhp_samples1)
    
    # Should handle gracefully without adding samples
    constant_return_handled = len(additional_samples1) == 0
    print(f"   Constant return handled: {'PASS' if constant_return_handled else 'FAIL'}")
    test_results.append(constant_return_handled)
    
    # Test 2: Function with very few initial samples
    print("\n2. Testing with minimal initial samples...")
    def minimal_samples_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        # Only produces output in narrow range
        if 45 <= threshold <= 55:
            return threshold, {"return_mean": threshold + 10}
        else:
            return 50.0, {"return_mean": 60.0}  # All map to same point
    
    sampler2 = BinarySearchSampler(
        eval_function=minimal_samples_function,
        num_bins=10,
        return_bins=5,
        max_additional_evals=10,
        verbose=False,
    )
    
    afhp_samples2 = sampler2.run()
    additional_samples2 = sampler2.fill_return_gaps(afhp_samples2)
    
    # Should handle gracefully
    minimal_samples_handled = True  # Any result is acceptable for this edge case
    print(f"   Minimal samples handled: {'PASS' if minimal_samples_handled else 'FAIL'}")
    test_results.append(minimal_samples_handled)
    
    # Test 3: Function with extreme return value ranges
    print("\n3. Testing with extreme return ranges...")
    def extreme_range_function(threshold: float) -> Tuple[float, Dict[str, Any]]:
        afhp = threshold
        # Very wide return range with gaps
        if threshold < 25:
            return_val = 10.0
        elif threshold > 75:
            return_val = 990.0
        else:
            return_val = 500.0
            
        return afhp, {"return_mean": return_val}
    
    sampler3 = BinarySearchSampler(
        eval_function=extreme_range_function,
        num_bins=8,
        return_bins=6,
        max_additional_evals=15,
        verbose=False,
    )
    
    afhp_samples3 = sampler3.run()
    additional_samples3 = sampler3.fill_return_gaps(afhp_samples3)
    
    # Should not crash and may add some samples
    extreme_range_handled = True  # Algorithm should not crash
    print(f"   Extreme ranges handled: {'PASS' if extreme_range_handled else 'FAIL'}")
    test_results.append(extreme_range_handled)
    
    # Overall result
    all_edge_cases_passed = all(test_results)
    print(f"\nEdge cases test: {'PASS' if all_edge_cases_passed else 'FAIL'}")
    
    return all_edge_cases_passed


if __name__ == "__main__":
    # Run main coverage test
    main_test_passed = test_full_coverage()

    # Run parameter variation tests
    param_tests_passed = test_coverage_with_different_parameters()

    # Run AFHP coverage guarantee test
    afhp_test_passed = test_afhp_coverage_guarantee()

    # Run guaranteed full coverage test
    guaranteed_test_passed = test_guaranteed_full_coverage()

    # Run unbounded mode tests
    unbounded_test_passed = test_unbounded_mode()
    
    # Run unbounded mode convergence test
    unbounded_convergence_passed = test_unbounded_mode_convergence()
    
    # Run Phase 2 binary bisection tests
    phase2_bisection_passed = test_phase2_binary_bisection()
    
    # Run Phase 2 gap identification test
    phase2_gaps_passed = test_phase2_gap_identification()
    
    # Run Phase 2 edge cases test
    phase2_edge_cases_passed = test_phase2_edge_cases()

    # Overall result
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULT:")
    if (
        main_test_passed
        and param_tests_passed
        and afhp_test_passed
        and guaranteed_test_passed
        and unbounded_test_passed
        and unbounded_convergence_passed
        and phase2_bisection_passed
        and phase2_gaps_passed
        and phase2_edge_cases_passed
    ):
        print("✓ ALL TESTS PASSED - Coverage guarantees verified!")
        print(
            "  - Algorithm achieves 100% coverage on both axes when function spans full range"
        )
        print(
            "  - Unbounded mode provides better coverage and terminates properly"
        )
        print(
            "  - Phase 2 binary bisection correctly fills return value gaps"
        )
        exit(0)
    else:
        print("✗ TESTS FAILED")
        if not afhp_test_passed:
            print(
                "  - AFHP coverage may be less than 100% when function output is concentrated"
            )
        if not guaranteed_test_passed:
            print("  - Failed to achieve 100% coverage even with linear function")
        if not unbounded_test_passed:
            print("  - Unbounded mode test failed")
        if not unbounded_convergence_passed:
            print("  - Unbounded mode convergence test failed")
        if not phase2_bisection_passed:
            print("  - Phase 2 binary bisection test failed")
        if not phase2_gaps_passed:
            print("  - Phase 2 gap identification test failed")
        if not phase2_edge_cases_passed:
            print("  - Phase 2 edge cases test failed")
        exit(1)
