"""
Test for ABCS algorithm to ensure 100% coverage on both x-axis and y-axis.

This test verifies that:
1. The AFHP (x-axis) bins are 100% filled by the binary search algorithm
2. The return (y-axis) bins are 100% filled when given sufficient evaluation budget
"""

import numpy as np
from typing import Tuple, Dict, Any
from abcs import BinarySearchSampler


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
            transformed_factor = log_factor ** steepness_power
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
    valid_samples = [s for s in samples if s is not None]
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
            
            # Create return bin edges
            return_bin_edges = np.linspace(min_return, max_return, return_bins + 1)
            
            # Count filled return bins
            filled_return_bins = set()
            for ret in returns:
                # Find which bin this return belongs to
                if max_return > min_return:
                    bin_idx = int((ret - min_return) / (max_return - min_return) * return_bins)
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
    
    print(f"X-axis (AFHP) coverage: {'PASS' if x_axis_pass else 'FAIL'} ({x_axis_coverage:.1f}%)")
    print(f"Y-axis (return) coverage: {'PASS' if y_axis_pass else 'FAIL'} ({y_axis_coverage:.1f}%)")
    
    if x_axis_pass and y_axis_pass:
        print("\n✓ TEST PASSED: 100% coverage achieved on both axes!")
    else:
        print("\n✗ TEST FAILED: Full coverage not achieved")
        if not x_axis_pass:
            print(f"  - X-axis coverage: {x_axis_coverage:.1f}% (expected 100%)")
        if not y_axis_pass:
            print(f"  - Y-axis coverage: {y_axis_coverage:.1f}% (expected 100%)")
    
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
        print(f"\nTest {i+1}: num_bins={config['num_bins']}, "
              f"return_bins={config['return_bins']}, "
              f"max_evals={config['max_evals']}")
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
        
        samples = sampler.run_with_return_refinement()
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
                    bin_idx = int((ret - min_return) / (max_return - min_return) * config["return_bins"])
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
    
    for num_bins in bin_counts:
        sampler = BinarySearchSampler(
            eval_function=eval_function,
            num_bins=num_bins,
            input_range=(0.0, 100.0),
            output_range=(0.0, 100.0),
            return_bins=0,  # Disable return refinement to test AFHP only
            verbose=False,
        )
        
        samples = sampler.run()
        summary = sampler.get_coverage_summary()
        coverage = summary["coverage_percentage"]
        
        # For reasonable bin counts, we should get 100% coverage
        passed = coverage == 100.0
        all_passed = all_passed and passed
        
        print(f"AFHP bins: {num_bins}, Coverage: {coverage:.1f}% ({'PASS' if passed else 'FAIL'})")
        
        # If coverage is not 100%, show which bins are missing
        if coverage < 100.0:
            gaps = summary.get("gaps", [])
            if gaps:
                print(f"  Missing bins: {gaps}")
    
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
    samples = sampler.run_with_return_refinement()
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
                bin_idx = int((ret - min_return) / (max_return - min_return) * return_bins)
                if bin_idx >= return_bins:
                    bin_idx = return_bins - 1
                filled_return_bins.add(bin_idx)
        
        y_coverage = 100.0 * len(filled_return_bins) / return_bins
    
    print(f"\nResults with linear function:")
    print(f"X-axis (AFHP) coverage: {x_coverage:.1f}%")
    print(f"Y-axis (return) coverage: {y_coverage:.1f}%")
    
    return x_coverage == 100.0 and y_coverage == 100.0


if __name__ == "__main__":
    # Run main coverage test
    main_test_passed = test_full_coverage()
    
    # Run parameter variation tests
    param_tests_passed = test_coverage_with_different_parameters()
    
    # Run AFHP coverage guarantee test
    afhp_test_passed = test_afhp_coverage_guarantee()
    
    # Run guaranteed full coverage test
    guaranteed_test_passed = test_guaranteed_full_coverage()
    
    # Overall result
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULT:")
    if main_test_passed and param_tests_passed and afhp_test_passed and guaranteed_test_passed:
        print("✓ ALL TESTS PASSED - Coverage guarantees verified!")
        print("  - Algorithm achieves 100% coverage on both axes when function spans full range")
        exit(0)
    else:
        print("✗ TESTS FAILED")
        if not afhp_test_passed:
            print("  - AFHP coverage may be less than 100% when function output is concentrated")
        if not guaranteed_test_passed:
            print("  - Failed to achieve 100% coverage even with linear function")
        exit(1)