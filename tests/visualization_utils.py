"""
Visualization utilities for test artifacts.

This module provides functions to generate and save visualizations
of test results, including threshold-to-AFHP mappings and AFHP-to-return
mappings. These artifacts help with debugging and understanding algorithm behavior.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # Use non-interactive backend for headless environments
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from abcs.types import SamplePoint


def create_test_artifacts_dir(timestamp: Optional[str] = None) -> Path:
    """
    Create a timestamped test_artifacts subdirectory if it doesn't exist.
    
    Args:
        timestamp: Optional timestamp string (if None, current time is used)
    
    Returns:
        Path to the timestamped test_artifacts subdirectory
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main test_artifacts directory
    main_dir = Path("test_artifacts")
    main_dir.mkdir(exist_ok=True)
    
    # Create timestamped subdirectory
    artifacts_dir = main_dir / timestamp
    artifacts_dir.mkdir(exist_ok=True)
    
    return artifacts_dir


def plot_threshold_to_afhp_mapping(
    samples: List[SamplePoint],
    test_name: str = "test",
    timestamp: Optional[str] = None
) -> Optional[Path]:
    """
    Generate and save a plot showing the threshold to AFHP mapping.
    
    Args:
        samples: List of sample points from the algorithm
        test_name: Name of the test for the filename
        timestamp: Optional timestamp string (if None, current time is used)
        
    Returns:
        Path to the saved plot, or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping threshold-to-AFHP plot")
        return None
    
    if not samples:
        print("Warning: No samples provided, skipping threshold-to-AFHP plot")
        return None
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract threshold and AFHP values
    thresholds = []
    afhp_values = []
    
    for sample in samples:
        if sample is not None:
            # Get threshold from input
            threshold = sample.input_value
            thresholds.append(threshold)
            
            # Get AFHP from output
            afhp = sample.output_value
            afhp_values.append(afhp)
    
    if not thresholds:
        print("Warning: No valid samples found, skipping threshold-to-AFHP plot")
        return None
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, afhp_values, alpha=0.7, s=50)
    plt.xlabel('Threshold')
    plt.ylabel('AFHP')
    plt.title(f'Threshold to AFHP Mapping - {test_name}')
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    min_threshold, max_threshold = min(thresholds), max(thresholds)
    min_afhp, max_afhp = min(afhp_values), max(afhp_values)
    
    plt.text(0.02, 0.98, 
             f'Samples: {len(thresholds)}\n'
             f'Threshold range: [{min_threshold:.1f}, {max_threshold:.1f}]\n'
             f'AFHP range: [{min_afhp:.1f}, {max_afhp:.1f}]',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    artifacts_dir = create_test_artifacts_dir(timestamp)
    filename = f"threshold_to_afhp_{test_name}.png"
    filepath = artifacts_dir / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved threshold-to-AFHP plot: {filepath}")
    return filepath


def plot_afhp_to_return_mapping(
    samples: List[SamplePoint],
    sampler,  # BinarySearchSampler instance
    test_name: str = "test",
    timestamp: Optional[str] = None
) -> Optional[Path]:
    """
    Generate and save a plot showing the AFHP to return value mapping.
    
    Args:
        samples: List of sample points from the algorithm
        sampler: BinarySearchSampler instance (needed to extract return values)
        test_name: Name of the test for the filename
        timestamp: Optional timestamp string (if None, current time is used)
        
    Returns:
        Path to the saved plot, or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping AFHP-to-return plot")
        return None
    
    if not samples:
        print("Warning: No samples provided, skipping AFHP-to-return plot")
        return None
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract AFHP and return values
    afhp_values = []
    return_values = []
    
    for sample in samples:
        if sample is not None:
            try:
                # Get AFHP from output
                afhp = sample.output_value
                
                # Get return value using the sampler's method
                return_val = sampler.extract_return_value(sample)
                
                afhp_values.append(afhp)
                return_values.append(return_val)
            except (ValueError, AttributeError):
                # Skip samples where return value cannot be extracted
                continue
    
    if not afhp_values:
        print("Warning: No valid AFHP-return pairs found, skipping AFHP-to-return plot")
        return None
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(afhp_values, return_values, alpha=0.7, s=50, c='orange')
    plt.xlabel('AFHP')
    plt.ylabel('Return Value')
    plt.title(f'AFHP to Return Value Mapping - {test_name}')
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    min_afhp, max_afhp = min(afhp_values), max(afhp_values)
    min_return, max_return = min(return_values), max(return_values)
    
    plt.text(0.02, 0.98, 
             f'Samples: {len(afhp_values)}\n'
             f'AFHP range: [{min_afhp:.1f}, {max_afhp:.1f}]\n'
             f'Return range: [{min_return:.1f}, {max_return:.1f}]',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save the plot
    artifacts_dir = create_test_artifacts_dir(timestamp)
    filename = f"afhp_to_return_{test_name}.png"
    filepath = artifacts_dir / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved AFHP-to-return plot: {filepath}")
    return filepath


def save_test_artifacts(
    samples: List[SamplePoint],
    sampler,  # BinarySearchSampler instance
    test_name: str = "test",
    all_samples: Optional[List[SamplePoint]] = None
) -> Dict[str, Optional[Path]]:
    """
    Generate and save all test artifacts for a test run.
    
    Args:
        samples: Primary samples from Phase 1
        sampler: BinarySearchSampler instance
        test_name: Name of the test for filenames
        all_samples: Optional list including refinement samples
        
    Returns:
        Dictionary mapping artifact type to file path
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping all test artifacts")
        return {"threshold_to_afhp": None, "afhp_to_return": None, "directory": None}
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use all_samples if provided, otherwise use primary samples
    plot_samples = all_samples if all_samples is not None else samples
    
    # Generate threshold-to-AFHP plot
    threshold_plot = plot_threshold_to_afhp_mapping(
        plot_samples, test_name, timestamp
    )
    
    # Generate AFHP-to-return plot
    return_plot = plot_afhp_to_return_mapping(
        plot_samples, sampler, test_name, timestamp
    )
    
    # Create a summary file in the timestamped directory
    artifacts_dir = create_test_artifacts_dir(timestamp)
    summary_file = artifacts_dir / "test_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Test Name: {test_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of samples: {len(plot_samples) if plot_samples else 0}\n")
        
        # Get coverage summary
        summary = sampler.get_coverage_summary()
        f.write(f"AFHP Coverage: {summary['coverage_percentage']:.1f}%\n")
        f.write(f"Total Evaluations: {summary['total_evaluations']}\n")
        f.write(f"Bins Filled: {summary['bins_filled']}/{sampler.num_bins}\n")
        
        if sampler.return_bins > 0:
            # Calculate return coverage if applicable
            returns = []
            for sample in (all_samples or samples):
                if sample is not None:
                    try:
                        ret = sampler.extract_return_value(sample)
                        returns.append(ret)
                    except ValueError:
                        pass
            
            if returns:
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
                    f.write(f"Return Coverage: {return_coverage:.1f}%\n")
                    f.write(f"Return Bins Filled: {len(filled_return_bins)}/{sampler.return_bins}\n")
    
    print(f"Test artifacts saved in: {artifacts_dir}")
    
    return {
        "threshold_to_afhp": threshold_plot,
        "afhp_to_return": return_plot,
        "directory": artifacts_dir
    }


def print_artifact_summary(artifacts: Dict[str, Optional[Path]]) -> None:
    """
    Print a summary of generated artifacts.
    
    Args:
        artifacts: Dictionary from save_test_artifacts
    """
    if artifacts.get("directory"):
        print(f"\nTest artifacts saved in timestamped directory: {artifacts['directory']}")
    else:
        print("\nTest artifacts generated:")
        for artifact_type, filepath in artifacts.items():
            if artifact_type != "directory":  # Skip the directory entry
                if filepath:
                    print(f"  - {artifact_type}: {filepath}")
                else:
                    print(f"  - {artifact_type}: Not generated (matplotlib unavailable)")