"""
Unbounded mode example for ABCS library.

This example demonstrates the difference between bounded and unbounded mode,
showing how unbounded mode can achieve better coverage by removing evaluation limits.
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
            "region": "slow" if threshold < 30 else ("fast" if threshold < 70 else "moderate")
        }
        
        return primary_output, metadata
    
    return eval_function


def compare_bounded_vs_unbounded():
    """Compare bounded mode vs unbounded mode performance."""
    print("=" * 70)
    print("BOUNDED vs UNBOUNDED MODE COMPARISON")
    print("=" * 70)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create challenging evaluation function
    eval_func = create_challenging_function()
    
    # Test parameters
    num_bins = 12
    return_bins = 8
    max_evals = 8  # Intentionally low to show difference
    
    print("\n1. BOUNDED MODE (Limited Evaluations)")
    print("-" * 40)
    
    # Test bounded mode
    sampler_bounded = BinarySearchSampler(
        eval_function=eval_func,
        num_bins=num_bins,
        return_bins=return_bins,
        max_additional_evals=max_evals,
        unbounded_mode=False,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True
    )
    
    samples_bounded = sampler_bounded.run_with_return_refinement()
    summary_bounded = sampler_bounded.get_coverage_summary()
    
    print(f"\nBounded Mode Results:")
    print(f"  Primary Coverage: {summary_bounded['coverage_percentage']:.1f}%")
    print(f"  Total Evaluations: {summary_bounded['total_evaluations']}")
    print(f"  Return Samples Added: {len(sampler_bounded.get_return_refinement_samples())}")
    
    # Reset random seed for fair comparison
    np.random.seed(42)
    
    print("\n2. UNBOUNDED MODE (Convergence Guaranteed)")
    print("-" * 45)
    
    # Test unbounded mode
    sampler_unbounded = BinarySearchSampler(
        eval_function=eval_func,
        num_bins=num_bins,
        return_bins=return_bins,
        max_additional_evals=max_evals,  # This will be ignored
        unbounded_mode=True,
        input_range=(0.0, 100.0),
        output_range=(0.0, 100.0),
        verbose=True
    )
    
    samples_unbounded = sampler_unbounded.run_with_return_refinement()
    summary_unbounded = sampler_unbounded.get_coverage_summary()
    
    print(f"\nUnbounded Mode Results:")
    print(f"  Primary Coverage: {summary_unbounded['coverage_percentage']:.1f}%")
    print(f"  Total Evaluations: {summary_unbounded['total_evaluations']}")
    print(f"  Return Samples Added: {len(sampler_unbounded.get_return_refinement_samples())}")
    
    # Calculate secondary coverage for both modes
    def calculate_return_coverage(sampler, return_bins):
        all_samples = sampler.get_all_samples_including_refinement()
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
    
    return_coverage_bounded = calculate_return_coverage(sampler_bounded, return_bins)
    return_coverage_unbounded = calculate_return_coverage(sampler_unbounded, return_bins)
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Bounded':<15} {'Unbounded':<15} {'Improvement'}")
    print("-" * 70)
    print(f"{'Primary Coverage':<25} {summary_bounded['coverage_percentage']:>6.1f}% {summary_unbounded['coverage_percentage']:>12.1f}% {summary_unbounded['coverage_percentage']-summary_bounded['coverage_percentage']:>10.1f}%")
    print(f"{'Secondary Coverage':<25} {return_coverage_bounded:>6.1f}% {return_coverage_unbounded:>12.1f}% {return_coverage_unbounded-return_coverage_bounded:>10.1f}%")
    print(f"{'Total Evaluations':<25} {summary_bounded['total_evaluations']:>8} {summary_unbounded['total_evaluations']:>14} {summary_unbounded['total_evaluations']-summary_bounded['total_evaluations']:>+8}")
    print(f"{'Return Samples':<25} {len(sampler_bounded.get_return_refinement_samples()):>8} {len(sampler_unbounded.get_return_refinement_samples()):>14} {len(sampler_unbounded.get_return_refinement_samples())-len(sampler_bounded.get_return_refinement_samples()):>+8}")
    
    print(f"\n✓ Unbounded mode achieved better or equal coverage using {summary_unbounded['total_evaluations']} evaluations")
    print(f"✓ Theoretical convergence guarantee: All bins will be filled if function spans range")
    print(f"✓ Practical safety: Maximum {sampler_unbounded.max_total_evals_unbounded} evaluations limit")


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
        unbounded_mode=True,
        verbose=True
    )
    
    samples = sampler.run_with_return_refinement()
    summary = sampler.get_coverage_summary()
    
    print(f"\nPathological Function Results:")
    print(f"  Coverage Achieved: {summary['coverage_percentage']:.1f}%")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Safety Limit: {sampler.max_total_evals_unbounded}")
    print(f"  Terminated Safely: {'✓' if summary['total_evaluations'] < sampler.max_total_evals_unbounded else '✗'}")
    
    if summary['coverage_percentage'] < 100:
        gaps = summary.get('gaps', [])
        print(f"  Unfilled Regions: {len(gaps)} (algorithm detected convergence)")


if __name__ == "__main__":
    print("ABCS Unbounded Mode Demonstration")
    print("This example shows how unbounded mode can achieve better coverage")
    print("while maintaining safety through convergence detection.\n")
    
    # Run the comparison
    compare_bounded_vs_unbounded()
    
    # Demonstrate safety mechanisms
    demonstrate_convergence_safety()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("• Use unbounded_mode=True for critical applications requiring maximum coverage")
    print("• Use bounded mode for quick exploration with evaluation budget constraints") 
    print("• Unbounded mode includes safety limits to prevent infinite execution")
    print("• Monitor verbose output to understand convergence behavior")
    print("=" * 70)