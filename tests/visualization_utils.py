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


def create_test_artifacts_dir() -> Path:
    """
    Create the test_artifacts directory if it doesn't exist.
    
    Returns:
        Path to the test_artifacts directory
    """
    artifacts_dir = Path("test_artifacts")
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
    artifacts_dir = create_test_artifacts_dir()
    filename = f"threshold_to_afhp_{test_name}_{timestamp}.png"
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
    artifacts_dir = create_test_artifacts_dir()
    filename = f"afhp_to_return_{test_name}_{timestamp}.png"
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
        return {"threshold_to_afhp": None, "afhp_to_return": None}
    
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
    
    return {
        "threshold_to_afhp": threshold_plot,
        "afhp_to_return": return_plot
    }


def print_artifact_summary(artifacts: Dict[str, Optional[Path]]) -> None:
    """
    Print a summary of generated artifacts.
    
    Args:
        artifacts: Dictionary from save_test_artifacts
    """
    print("\nTest artifacts generated:")
    for artifact_type, filepath in artifacts.items():
        if filepath:
            print(f"  - {artifact_type}: {filepath}")
        else:
            print(f"  - {artifact_type}: Not generated (matplotlib unavailable)")