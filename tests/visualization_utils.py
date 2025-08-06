"""
Visualization utilities for test artifacts.

This module provides functions to generate and save visualizations
of test results, including threshold-to-AFHP mappings and AFHP-to-return
mappings. These artifacts help with debugging and understanding algorithm behavior.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import datetime
import os

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

# Global variable to track the current test run timestamp
_CURRENT_TEST_RUN_TIMESTAMP = None


def _cleanup_old_artifact_folders(max_folders: int = 5) -> None:
    """
    Remove old timestamped artifact folders, keeping only the most recent ones.
    
    Args:
        max_folders: Maximum number of timestamped folders to keep (default: 5)
    """
    artifacts_root = Path("test_artifacts")
    if not artifacts_root.exists():
        return
    
    # Get all timestamped directories (matching YYYYMMDD_HHMMSS pattern)
    timestamped_dirs = []
    for item in artifacts_root.iterdir():
        if item.is_dir() and len(item.name) == 15 and '_' in item.name:
            # Check if it matches the timestamp pattern
            try:
                datetime.datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                timestamped_dirs.append(item)
            except ValueError:
                # Skip directories that don't match the timestamp pattern
                continue
    
    # Sort by name (which sorts chronologically due to timestamp format)
    timestamped_dirs.sort(key=lambda x: x.name)
    
    # Remove old directories if we have more than max_folders
    if len(timestamped_dirs) >= max_folders:
        folders_to_remove = timestamped_dirs[:-max_folders + 1]  # Keep space for the new one
        for old_folder in folders_to_remove:
            try:
                import shutil
                shutil.rmtree(old_folder)
                print(f"🗑️ Removed old test artifacts: {old_folder.name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not remove old folder {old_folder.name}: {e}")


def initialize_test_run() -> str:
    """
    Initialize a new test run with a timestamp.
    Automatically cleans up old test run folders to keep only the 5 most recent.
    
    Returns:
        The timestamp string for the test run
    """
    global _CURRENT_TEST_RUN_TIMESTAMP
    
    # Clean up old artifact folders before creating a new one
    _cleanup_old_artifact_folders(max_folders=5)
    
    _CURRENT_TEST_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the main test run directory
    test_run_dir = Path("test_artifacts") / _CURRENT_TEST_RUN_TIMESTAMP
    test_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test run summary file
    summary_file = test_run_dir / "test_run_info.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Test Run Started: {_CURRENT_TEST_RUN_TIMESTAMP}\n")
        f.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Run Directory: {test_run_dir}\n\n")
        f.write("Individual Test Results:\n")
        f.write("-" * 50 + "\n")
    
    print(f"📁 Test run artifacts directory: {test_run_dir}")
    return _CURRENT_TEST_RUN_TIMESTAMP


def get_current_test_run_timestamp() -> Optional[str]:
    """
    Get the current test run timestamp.
    
    Returns:
        Current test run timestamp or None if not initialized
    """
    return _CURRENT_TEST_RUN_TIMESTAMP


def create_test_artifacts_dir(test_name: str) -> Path:
    """
    Create a test-specific subdirectory within the current test run.
    
    Args:
        test_name: Name of the test for the subdirectory
    
    Returns:
        Path to the test-specific artifacts subdirectory
    """
    global _CURRENT_TEST_RUN_TIMESTAMP
    
    # If no test run is initialized, initialize one
    if _CURRENT_TEST_RUN_TIMESTAMP is None:
        initialize_test_run()
    
    # Create test-specific subdirectory
    test_run_dir = Path("test_artifacts") / _CURRENT_TEST_RUN_TIMESTAMP
    test_dir = test_run_dir / test_name
    test_dir.mkdir(exist_ok=True)
    
    return test_dir


def plot_threshold_to_afhp_mapping(
    samples: List[SamplePoint],
    test_name: str = "test"
) -> Optional[Path]:
    """
    Generate and save a plot showing the threshold to AFHP mapping.
    
    Args:
        samples: List of sample points from the algorithm
        test_name: Name of the test for the filename
        
    Returns:
        Path to the saved plot, or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping threshold-to-AFHP plot")
        return None
    
    if not samples:
        print("Warning: No samples provided, skipping threshold-to-AFHP plot")
        return None
    
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
    artifacts_dir = create_test_artifacts_dir(test_name)
    filename = f"threshold_to_afhp.png"
    filepath = artifacts_dir / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved threshold-to-AFHP plot: {filepath}")
    return filepath


def plot_afhp_to_return_mapping(
    samples: List[SamplePoint],
    sampler,  # BinarySearchSampler instance
    test_name: str = "test"
) -> Optional[Path]:
    """
    Generate and save a plot showing the AFHP to return value mapping.
    Uses different colors for Phase 1 vs Phase 2 samples.
    
    Args:
        samples: List of sample points from the algorithm
        sampler: BinarySearchSampler instance (needed to extract return values)
        test_name: Name of the test for the filename
        
    Returns:
        Path to the saved plot, or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping AFHP-to-return plot")
        return None
    
    if not samples:
        print("Warning: No samples provided, skipping AFHP-to-return plot")
        return None
    
    # Get Phase 2 (return refinement) samples to distinguish them
    phase2_samples = sampler.get_return_refinement_samples()
    phase2_sample_ids = {id(sample) for sample in phase2_samples if sample is not None}
    
    # Separate Phase 1 and Phase 2 samples
    phase1_afhp = []
    phase1_return = []
    phase2_afhp = []
    phase2_return = []
    
    for sample in samples:
        if sample is not None:
            try:
                # Get AFHP from output
                afhp = sample.output_value
                
                # Get return value using the sampler's method
                return_val = sampler.extract_return_value(sample)
                
                # Determine if this is a Phase 1 or Phase 2 sample
                if id(sample) in phase2_sample_ids:
                    phase2_afhp.append(afhp)
                    phase2_return.append(return_val)
                else:
                    phase1_afhp.append(afhp)
                    phase1_return.append(return_val)
                    
            except (ValueError, AttributeError):
                # Skip samples where return value cannot be extracted
                continue
    
    total_samples = len(phase1_afhp) + len(phase2_afhp)
    if total_samples == 0:
        print("Warning: No valid AFHP-return pairs found, skipping AFHP-to-return plot")
        return None
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot Phase 1 samples (primary coverage)
    if phase1_afhp:
        plt.scatter(phase1_afhp, phase1_return, alpha=0.7, s=50, c='orange', label=f'Phase 1 ({len(phase1_afhp)} samples)')
    
    # Plot Phase 2 samples (return gap filling) 
    if phase2_afhp:
        plt.scatter(phase2_afhp, phase2_return, alpha=0.7, s=50, c='red', label=f'Phase 2 ({len(phase2_afhp)} samples)')
    
    plt.xlabel('AFHP')
    plt.ylabel('Return Value')
    plt.title(f'AFHP to Return Value Mapping - {test_name}')
    plt.grid(True, alpha=0.3)
    
    # Calculate statistics from all samples
    all_afhp = phase1_afhp + phase2_afhp
    all_return = phase1_return + phase2_return
    min_afhp, max_afhp = min(all_afhp), max(all_afhp)
    min_return, max_return = min(all_return), max(all_return)
    
    # Add legend in bottom right if we have both types of samples
    if phase1_afhp and phase2_afhp:
        plt.legend(loc='lower right')
    
    # Add statistics text box in top left (now without legend overlap)
    plt.text(0.02, 0.98, 
             f'Total: {total_samples} samples\n'
             f'Phase 1: {len(phase1_afhp)} | Phase 2: {len(phase2_afhp)}\n'
             f'AFHP range: [{min_afhp:.1f}, {max_afhp:.1f}]\n'
             f'Return range: [{min_return:.1f}, {max_return:.1f}]',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save the plot
    artifacts_dir = create_test_artifacts_dir(test_name)
    filename = f"afhp_to_return.png"
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
        # This should not be called when matplotlib is unavailable - the calling code should handle this
        raise ImportError("matplotlib is required for test visualizations but is not installed. Install with: pip install matplotlib")
    
    # Use all_samples if provided, otherwise use primary samples
    plot_samples = all_samples if all_samples is not None else samples
    
    # Generate threshold-to-AFHP plot
    threshold_plot = plot_threshold_to_afhp_mapping(
        plot_samples, test_name
    )
    
    # Generate AFHP-to-return plot
    return_plot = plot_afhp_to_return_mapping(
        plot_samples, sampler, test_name
    )
    
    # Create a summary file in the test-specific directory
    artifacts_dir = create_test_artifacts_dir(test_name)
    summary_file = artifacts_dir / "test_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Test Name: {test_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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