"""
Basic usage example for ABCS library.

This example demonstrates how to use ABCS to sample a simple monotonic function
with both primary and secondary output coverage.
"""

import numpy as np
from abcs import BinarySearchSampler


def create_example_function():
    """
    Create an example monotonic function for demonstration.

    This function simulates a threshold-based system where:
    - Input: threshold percentile (0-100)
    - Primary output: activation percentage (0-100)
    - Secondary output: performance score (0-100)
    """

    def eval_function(percentile: float):
        # Convert percentile to a monotonic primary output using sigmoid
        z = (percentile - 50) / 15
        sigmoid = 1 / (1 + np.exp(-z))
        primary_output = sigmoid * 90 + 5  # Scale to [5, 95]

        # Secondary output increases with primary output but with diminishing returns
        secondary_output = 20 + 70 * np.log(1 + primary_output / 20)

        # Add some realistic noise
        primary_output += np.random.normal(0, 1)
        secondary_output += np.random.normal(0, 2)

        # Ensure outputs stay in reasonable ranges
        primary_output = np.clip(primary_output, 0, 100)
        secondary_output = np.clip(secondary_output, 0, 100)

        metadata = {
            "return_mean": secondary_output,
            "percentile_used": percentile,
            "noise_level": "low",
        }

        return primary_output, metadata

    return eval_function


def example_basic_coverage():
    """Example 1: Basic primary coverage only."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Primary Coverage")
    print("=" * 60)

    # Create evaluation function
    eval_func = create_example_function()

    # Create sampler for primary coverage only
    sampler = BinarySearchSampler(
        eval_function=eval_func,
        num_bins=10,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True,
    )

    # Run Phase 1 only
    samples = sampler.run()

    # Print results
    summary = sampler.get_coverage_summary()
    print("\nResults:")
    print(f"- Bins filled: {summary['bins_filled']}/{sampler.num_bins}")
    print(f"- Coverage: {summary['coverage_percentage']}%")
    print(f"- Total evaluations: {summary['total_evaluations']}")
    print(f"- Output range covered: {summary['output_range_covered']}")

    return samples


def example_two_phase_coverage():
    """Example 2: Two-phase coverage (primary + secondary)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Two-Phase Coverage")
    print("=" * 60)

    # Create evaluation function
    eval_func = create_example_function()

    # Create sampler with secondary coverage enabled
    sampler = BinarySearchSampler(
        eval_function=eval_func,
        num_bins=12,
        return_bins=8,  # Enable secondary axis coverage
        max_additional_evals=20,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True,
    )

    # Run both phases
    samples = sampler.run_with_return_refinement()

    # Get all samples including refinement
    all_samples = sampler.get_all_samples_including_refinement()
    refinement_samples = sampler.get_return_refinement_samples()

    # Print results
    summary = sampler.get_coverage_summary()
    print("\nResults:")
    print(f"- Primary bins filled: {summary['bins_filled']}/{sampler.num_bins}")
    print(f"- Primary coverage: {summary['coverage_percentage']}%")
    print(f"- Total evaluations: {summary['total_evaluations']}")
    print(f"- Phase 1 samples: {len([s for s in samples if s is not None])}")
    print(f"- Phase 2 samples: {len(refinement_samples)}")
    print(f"- Total samples: {len(all_samples)}")

    # Calculate secondary coverage
    if refinement_samples:
        secondary_values = []
        for sample in all_samples:
            try:
                ret = sampler.extract_return_value(sample)
                secondary_values.append(ret)
            except ValueError:
                pass

        if secondary_values:
            print(
                f"- Secondary output range: [{min(secondary_values):.1f}, {max(secondary_values):.1f}]"
            )

    return all_samples


def example_custom_transformation():
    """Example 3: Using custom input transformation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Input Transformation")
    print("=" * 60)

    # Create evaluation function that expects actual thresholds
    def threshold_eval_function(threshold: float):
        # Simulate evaluation with actual threshold values
        if threshold == float("inf"):
            activation_rate = 0.0  # Never activate
        elif threshold == float("-inf"):
            activation_rate = 100.0  # Always activate
        else:
            # Sigmoid activation based on threshold
            activation_rate = 100 / (1 + np.exp(threshold))

        # Performance increases with activation but with costs
        performance = min(95, activation_rate * 0.8 + 10)

        metadata = {"return_mean": performance, "threshold_used": threshold}

        return activation_rate, metadata

    # Custom transformation from percentile to threshold
    def percentile_to_threshold(percentile: float):
        if percentile <= 0:
            return float("inf")  # Never activate
        elif percentile >= 100:
            return float("-inf")  # Always activate
        else:
            # Convert percentile to threshold in reasonable range
            return 3.0 - (percentile / 100.0) * 6.0  # Range: [3, -3]

    # Create sampler with custom transformation
    sampler = BinarySearchSampler(
        eval_function=threshold_eval_function,
        num_bins=8,
        return_bins=6,
        input_range=(0.0, 100.0),  # Still use percentiles as input
        output_range=(0.0, 100.0),
        input_to_threshold=percentile_to_threshold,
        verbose=True,
    )

    # Run the algorithm
    _samples = sampler.run_with_return_refinement()

    # Print results
    summary = sampler.get_coverage_summary()
    print("\nResults:")
    print(f"- Coverage: {summary['coverage_percentage']}%")
    print(f"- Total evaluations: {summary['total_evaluations']}")

    # Show some sample transformations
    print("\nSample input transformations:")
    for percentile in [0, 25, 50, 75, 100]:
        threshold = percentile_to_threshold(percentile)
        print(f"  {percentile}% percentile → threshold {threshold}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)

    print("ABCS Library - Basic Usage Examples")
    print("This demonstrates the Adaptive Binary Coverage Search algorithm")

    # Run examples
    example_basic_coverage()
    example_two_phase_coverage()
    example_custom_transformation()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("Try modifying the parameters to see how they affect coverage.")
    print("=" * 60)
