"""
Visualization utilities for joint-coverage sampler test artifacts (v2).

Generates simple plots for:
- Percentile -> AFHP (x) mapping
- AFHP (x) -> Performance (y) mapping
and stores a small textual summary per test.

Works with `CurvePoint` and `SamplingResult` from the new joint sampler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import datetime
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

# Type imports for annotations
from typing import Any


_CURRENT_TEST_RUN_TIMESTAMP: Optional[str] = None


def _cleanup_old_artifact_folders(max_folders: int = 5) -> None:
    artifacts_root = Path("test_artifacts")
    if not artifacts_root.exists():
        return
    timestamped_dirs = []
    for item in artifacts_root.iterdir():
        if item.is_dir() and len(item.name) == 15 and "_" in item.name:
            datetime.datetime.strptime(item.name, "%Y%m%d_%H%M%S")
            timestamped_dirs.append(item)
    timestamped_dirs.sort(key=lambda x: x.name)
    if len(timestamped_dirs) >= max_folders:
        import shutil

        for old_folder in timestamped_dirs[: -max_folders + 1]:
            shutil.rmtree(old_folder)
            print(f"🗑️ Removed old test artifacts: {old_folder.name}")


def initialize_test_run() -> str:
    global _CURRENT_TEST_RUN_TIMESTAMP
    # If already initialized in this process, reuse the same timestamp
    if _CURRENT_TEST_RUN_TIMESTAMP is not None:
        return _CURRENT_TEST_RUN_TIMESTAMP

    # Allow forcing a shared timestamp across processes via env var
    forced_ts = os.environ.get("ACS_TEST_RUN_TIMESTAMP") or os.environ.get(
        "ABCS_TEST_RUN_TIMESTAMP"
    )
    if forced_ts:
        _CURRENT_TEST_RUN_TIMESTAMP = forced_ts
    else:
        _cleanup_old_artifact_folders(max_folders=5)
        _CURRENT_TEST_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_dir = Path("test_artifacts") / _CURRENT_TEST_RUN_TIMESTAMP
    test_run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = test_run_dir / "test_run_info.txt"
    with open(summary_file, "w") as f:
        f.write(f"Test Run Started: {_CURRENT_TEST_RUN_TIMESTAMP}\n")
        f.write(
            f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Test Run Directory: {test_run_dir}\n\n")
        f.write("Individual Test Results:\n")
        f.write("-" * 50 + "\n")
    print(f"📁 Test run artifacts directory: {test_run_dir}")
    return _CURRENT_TEST_RUN_TIMESTAMP


def create_test_artifacts_dir(test_name: str) -> Path:
    global _CURRENT_TEST_RUN_TIMESTAMP
    if _CURRENT_TEST_RUN_TIMESTAMP is None:
        initialize_test_run()
    test_run_dir = Path("test_artifacts") / _CURRENT_TEST_RUN_TIMESTAMP  # type: ignore[arg-type]
    test_dir = test_run_dir / test_name
    test_dir.mkdir(exist_ok=True)
    return test_dir


def plot_percentile_to_afhp(points: List[Any], test_name: str) -> Optional[Path]:
    if not points:
        return None
    artifacts_dir = create_test_artifacts_dir(test_name)

    x = [p.desired_percentile for p in points]
    y = [p.afhp for p in points]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=40, alpha=0.8)
    # Label with sampling order
    for p in points:
        plt.annotate(
            str(p.order),
            (p.desired_percentile, p.afhp),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("Percentile (input)")
    plt.ylabel("AFHP (x)")
    plt.title(f"Percentile to AFHP - {test_name}")
    plt.grid(True, alpha=0.3)
    path = artifacts_dir / "percentile_to_afhp.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_afhp_to_performance(points: List[Any], test_name: str) -> Optional[Path]:
    if not points:
        return None
    artifacts_dir = create_test_artifacts_dir(test_name)
    x = [p.afhp for p in points]
    y = [p.performance for p in points]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=40, alpha=0.8)
    # Label with sampling order
    for p in points:
        plt.annotate(
            str(p.order),
            (p.afhp, p.performance),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("AFHP (x)")
    plt.ylabel("Performance (y)")
    plt.title(f"AFHP to Performance - {test_name}")
    plt.grid(True, alpha=0.3)
    path = artifacts_dir / "afhp_to_performance.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_joint_artifacts(
    points: List[Any], result: Any, test_name: str
) -> Dict[str, Optional[Path]]:
    artifacts_dir = create_test_artifacts_dir(test_name)
    # Plots
    p_plot = plot_percentile_to_afhp(points, test_name)
    xy_plot = plot_afhp_to_performance(points, test_name)
    # Summary
    summary_file = artifacts_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Test: {test_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total points: {len(points)}\n")
        f.write(f"Total evals: {result.total_evals}\n")
        f.write(f"coverage_x_max_gap: {result.coverage_x_max_gap:.4f}\n")
        f.write(f"coverage_y_max_gap: {result.coverage_y_max_gap:.4f}\n")
        f.write(f"early_stop_reason: {result.early_stop_reason}\n")
        f.write(
            f"monotonicity_violations_remaining: {result.monotonicity_violations_remaining}\n"
        )
    # Dump points
    points_file = artifacts_dir / "points.tsv"
    with open(points_file, "w") as f:
        f.write("percentile\tafhp\tperformance\trepeats\n")
        for p in points:
            f.write(
                f"{p.desired_percentile:.6f}\t{p.afhp:.6f}\t{p.performance:.6f}\t{p.repeats_used}\n"
            )
    return {
        "percentile_to_afhp": p_plot,
        "afhp_to_performance": xy_plot,
        "directory": artifacts_dir,
    }


def print_artifact_summary(artifacts: Dict[str, Optional[Path]]) -> None:
    if artifacts.get("directory"):
        print(f"Artifacts saved in: {artifacts['directory']}")
    for k, v in artifacts.items():
        if k != "directory" and v is not None:
            print(f"  - {k}: {v}")


# ============================================================================
# Single-axis sampler visualization functions
# ============================================================================


def plot_input_to_output(samples: List[Any], test_name: str) -> Optional[Path]:
    """Plot input values (percentiles) to output values (AFHP)."""
    if not samples:
        return None
    artifacts_dir = create_test_artifacts_dir(test_name)

    # Extract data from CurvePoint objects
    x = [s.desired_percentile for s in samples]
    y = [s.afhp for s in samples]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=40, alpha=0.8, c="blue")

    # Sort by input for line plot
    sorted_pairs = sorted(zip(x, y))
    sorted_x, sorted_y = zip(*sorted_pairs)
    plt.plot(sorted_x, sorted_y, alpha=0.5, color="blue", linewidth=1)

    plt.xlabel("Input Value (percentile)")
    plt.ylabel("Output Value (AFHP)")
    plt.title(f"Input to Output Mapping - {test_name}")
    plt.grid(True, alpha=0.3)
    path = artifacts_dir / "input_to_output.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_output_to_performance(samples: List[Any], test_name: str) -> Optional[Path]:
    """Plot output values (AFHP) to performance values."""
    if not samples:
        return None

    # Extract performance values from CurvePoint objects
    performance_values = [s.performance for s in samples]
    output_values = [s.afhp for s in samples]

    artifacts_dir = create_test_artifacts_dir(test_name)

    plt.figure(figsize=(8, 5))
    plt.scatter(output_values, performance_values, s=40, alpha=0.8, c="red")

    # Sort by output for line plot
    sorted_pairs = sorted(zip(output_values, performance_values))
    sorted_x, sorted_y = zip(*sorted_pairs)
    plt.plot(sorted_x, sorted_y, alpha=0.5, color="red", linewidth=1)

    plt.xlabel("Output Value (AFHP)")
    plt.ylabel("Performance")
    plt.title(f"Output to Performance Mapping - {test_name}")
    plt.grid(True, alpha=0.3)
    path = artifacts_dir / "output_to_performance.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_bin_coverage(
    samples: List[Any], sampler: Any, test_name: str, result: Any
) -> Optional[Path]:
    """Plot bin coverage showing which bins were filled."""
    if not samples:
        return None

    artifacts_dir = create_test_artifacts_dir(test_name)

    # Create histogram of output values with bin edges
    output_values = [s.afhp for s in samples]

    plt.figure(figsize=(10, 6))

    # Plot histogram
    plt.hist(output_values, bins=sampler.bin_edges, alpha=0.7, edgecolor="black")

    # Mark bin edges
    for edge in sampler.bin_edges:
        plt.axvline(x=edge, color="red", linestyle="--", alpha=0.5)

    plt.xlabel("Output Value (AFHP)")
    plt.ylabel("Number of Samples per Bin")
    plt.title(f"Bin Coverage - {test_name}")
    plt.grid(True, alpha=0.3)

    # Add text showing coverage stats
    bins_filled = result.info["bins_filled"]
    total_bins = result.info["total_bins"]
    coverage_percentage = result.info["coverage_percentage"]
    total_evals = result.total_evals

    plt.text(
        0.02,
        0.98,
        f"Bins filled: {bins_filled}/{total_bins}\n"
        f"Coverage: {coverage_percentage:.1f}%\n"
        f"Total evals: {total_evals}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    path = artifacts_dir / "bin_coverage.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_single_axis_artifacts(
    samples: List[Any], sampler: Any, test_name: str, result: Any
) -> Dict[str, Optional[Path]]:
    """Save all artifacts for single-axis sampler test results."""
    artifacts_dir = create_test_artifacts_dir(test_name)

    # Generate plots
    input_output_plot = plot_input_to_output(samples, test_name)
    output_perf_plot = plot_output_to_performance(samples, test_name)
    bin_coverage_plot = plot_bin_coverage(samples, sampler, test_name, result)

    # Get coverage summary from result
    bins_filled = result.info["bins_filled"]
    total_bins = result.info["total_bins"]
    coverage_percentage = result.info["coverage_percentage"]
    total_evals = result.total_evals
    output_range_covered = (
        (min(s.afhp for s in samples), max(s.afhp for s in samples))
        if samples
        else (None, None)
    )
    gaps = result.info.get("uncovered_bins", [])

    # Save summary
    summary_file = artifacts_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Test: {test_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Total evaluations: {total_evals}\n")
        f.write(f"Bins filled: {bins_filled}/{total_bins}\n")
        f.write(f"Coverage percentage: {coverage_percentage:.2f}%\n")
        f.write(f"Output range covered: {output_range_covered}\n")
        f.write(f"Number of gaps: {len(gaps)}\n")
        if gaps:
            f.write("Gap ranges:\n")
            for i, (gap_start, gap_end) in enumerate(gaps):
                f.write(f"  Gap {i + 1}: [{gap_start:.2f}, {gap_end:.2f}]\n")

    # Save sample points
    points_file = artifacts_dir / "points.tsv"
    with open(points_file, "w") as f:
        f.write("input_value\toutput_value\tthreshold\tperformance\n")
        for sample in samples:
            input_value = sample.desired_percentile
            output_value = sample.afhp
            threshold = (
                sample.desired_percentile * 100.0
            )  # Convert percentile to threshold
            performance = sample.performance
            f.write(
                f"{input_value:.6f}\t{output_value:.6f}\t{threshold}\t{performance}\n"
            )

    return {
        "input_to_output": input_output_plot,
        "output_to_performance": output_perf_plot,
        "bin_coverage": bin_coverage_plot,
        "directory": artifacts_dir,
    }
