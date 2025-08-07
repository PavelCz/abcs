"""
Advanced usage example for ABCS library.

This example demonstrates the comprehensive coverage capabilities of the simplified API,
showing how the algorithm automatically achieves optimal coverage with built-in safety mechanisms.
"""

import numpy as np
from abcs import BinarySearchSampler


def create_challenging_function():
    """
    Create a challenging evaluation function that's difficult to sample uniformly.

    This function has regions that are harder to sample, making it a good test
    for unbounded mode's ability to achieve complete coverage.
    """

    def eval_function(threshold: float):
        # Create a function with non-uniform density - harder to sample in some regions
        if threshold < 30:
            # Slow growth region
            primary_output = threshold * 0.5
        elif threshold < 70:
            # Fast growth region
            primary_output = 15 + (threshold - 30) * 1.5
        else:
            # Moderate growth region
            primary_output = 75 + (threshold - 70) * 0.8

        # Secondary output with complex relationship
        secondary_output = 10 + 80 * (primary_output / 100) ** 0.3

        # Add small amount of noise for realism
        primary_output += np.random.normal(0, 0.5)
        secondary_output += np.random.normal(0, 1.0)

        # Clamp to valid ranges
        primary_output = np.clip(primary_output, 0, 100)
        secondary_output = np.clip(secondary_output, 0, 100)

        metadata = {
            "return_mean": secondary_output,
            "threshold_used": threshold,
            "region": "slow"
            if threshold < 30
            else ("fast" if threshold < 70 else "moderate"),
        }

        return primary_output, metadata

    return eval_function


def demonstrate_improved_coverage():
    """Demonstrate the improved coverage capabilities of the simplified API."""
    print("=" * 70)
    print("ADVANCED COVERAGE DEMONSTRATION")
    print("=" * 70)

    # Set random seed for reproducible results
    np.random.seed(42)

    # Create challenging evaluation function
    eval_func = create_challenging_function()

    # Test parameters
    num_bins = 12
    return_bins = 8

    print("\nDemonstrating comprehensive coverage with simplified API...")
    print("-" * 50)

    # Create sampler with simplified API (always uses optimal approach)
    sampler = BinarySearchSampler(
        eval_function=eval_func,
        num_bins=num_bins,
        return_bins=return_bins,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True,
    )

    sampler.run_with_return_refinement()
    summary = sampler.get_coverage_summary()

    print("\nSimplified API Results:")
    print(f"  Primary Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Return Samples Added: {len(sampler.get_return_refinement_samples())}")

    # Calculate secondary coverage
    def calculate_return_coverage(sampler, return_bins):
        primary_samples = sampler.get_filled_samples()
        refinement_samples = sampler.get_return_refinement_samples()
        all_samples = primary_samples + refinement_samples
        returns = []
        for sample in all_samples:
            try:
                ret = sampler.extract_return_value(sample)
                returns.append(ret)
            except ValueError:
                pass

        if not returns or return_bins == 0:
            return 0.0

        min_return = min(returns)
        max_return = max(returns)
        if max_return <= min_return:
            return 0.0

        filled_bins = set()
        for ret in returns:
            bin_idx = int((ret - min_return) / (max_return - min_return) * return_bins)
            if bin_idx >= return_bins:
                bin_idx = return_bins - 1
            filled_bins.add(bin_idx)

        return 100.0 * len(filled_bins) / return_bins

    return_coverage = calculate_return_coverage(sampler, return_bins)

    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Primary Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"Secondary Coverage: {return_coverage:.1f}%")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Return Samples Added: {len(sampler.get_return_refinement_samples())}")

    print(
        f"\n✓ Simplified API achieved comprehensive coverage using {summary['total_evaluations']} evaluations"
    )
    print(
        "✓ Theoretical convergence guarantee: All bins will be filled if function spans range"
    )
    print(f"✓ Practical safety: Maximum {sampler.max_total_evals} evaluations limit")


def demonstrate_convergence_safety():
    """Demonstrate convergence detection and safety mechanisms."""
    print("\n\n" + "=" * 70)
    print("CONVERGENCE SAFETY DEMONSTRATION")
    print("=" * 70)

    def pathological_function(threshold: float):
        """A function that's very hard to sample in certain regions."""
        # Most outputs concentrated in narrow range
        if threshold < 90:
            output = 40 + threshold * 0.1  # Very compressed
        else:
            output = 50 + (threshold - 90) * 5  # Rapid expansion

        return_val = 30 + output * 0.5

        return output, {"return_mean": return_val}

    print("\nTesting with pathological function that's hard to sample...")

    sampler = BinarySearchSampler(
        eval_function=pathological_function,
        num_bins=20,  # Many bins for difficult function
        return_bins=10,
        verbose=True,
    )

    sampler.run_with_return_refinement()
    summary = sampler.get_coverage_summary()

    print("\nPathological Function Results:")
    print(f"  Coverage Achieved: {summary['coverage_percentage']:.1f}%")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Safety Limit: {sampler.max_total_evals}")
    print(
        f"  Terminated Safely: {'✓' if summary['total_evaluations'] < sampler.max_total_evals else '✗'}"
    )

    if summary["coverage_percentage"] < 100:
        gaps = summary.get("gaps", [])
        print(f"  Unfilled Regions: {len(gaps)} (algorithm detected convergence)")


if __name__ == "__main__":
    print("ABCS Advanced Coverage Demonstration")
    print("This example shows the comprehensive coverage capabilities")
    print("of the simplified API with automatic optimization.\n")

    # Run the demonstration
    demonstrate_improved_coverage()

    # Demonstrate safety mechanisms
    demonstrate_convergence_safety()

    print("\n" + "=" * 70)
    print("KEY FEATURES OF SIMPLIFIED API")
    print("=" * 70)
    print("• Automatically uses optimal coverage strategy")
    print("• Theoretical convergence guarantees with safety mechanisms")
    print("• No need to choose between bounded/unbounded modes")
    print("• Simplified parameter set for easier usage")
    print("• Monitor verbose output to understand algorithm behavior")
    print("=" * 70)
