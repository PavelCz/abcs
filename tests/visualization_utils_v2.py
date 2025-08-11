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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

# Lazily import types to avoid circular issues during test discovery
try:
    from acs import CurvePoint, SamplingResult
except Exception:  # pragma: no cover
    CurvePoint = object  # type: ignore
    SamplingResult = object  # type: ignore


_CURRENT_TEST_RUN_TIMESTAMP: Optional[str] = None


def _cleanup_old_artifact_folders(max_folders: int = 5) -> None:
    artifacts_root = Path("test_artifacts")
    if not artifacts_root.exists():
        return
    timestamped_dirs = []
    for item in artifacts_root.iterdir():
        if item.is_dir() and len(item.name) == 15 and "_" in item.name:
            try:
                datetime.datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                timestamped_dirs.append(item)
            except ValueError:
                continue
    timestamped_dirs.sort(key=lambda x: x.name)
    if len(timestamped_dirs) >= max_folders:
        for old_folder in timestamped_dirs[: -max_folders + 1]:
            try:
                import shutil

                shutil.rmtree(old_folder)
                print(f"🗑️ Removed old test artifacts: {old_folder.name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not remove old folder {old_folder.name}: {e}")


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


def plot_percentile_to_afhp(points: List[CurvePoint], test_name: str) -> Optional[Path]:
    if plt is None or not points:
        return None
    artifacts_dir = create_test_artifacts_dir(test_name)
    # Support both 'percentile' and 'desired_percentile' field names
    x = [
        getattr(p, "percentile", getattr(p, "desired_percentile", None)) for p in points
    ]
    y = [p.afhp for p in points]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=40, alpha=0.8)
    # Label with sampling order
    for p in points:
        try:
            px = getattr(p, "percentile", getattr(p, "desired_percentile", None))
            plt.annotate(
                str(p.order),
                (px, p.afhp),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        except Exception:
            pass
    plt.xlabel("Percentile (input)")
    plt.ylabel("AFHP (x)")
    plt.title(f"Percentile to AFHP - {test_name}")
    plt.grid(True, alpha=0.3)
    path = artifacts_dir / "percentile_to_afhp.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_afhp_to_performance(
    points: List[CurvePoint], test_name: str
) -> Optional[Path]:
    if plt is None or not points:
        return None
    artifacts_dir = create_test_artifacts_dir(test_name)
    x = [p.afhp for p in points]
    y = [p.performance for p in points]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=40, alpha=0.8)
    # Label with sampling order
    for p in points:
        try:
            plt.annotate(
                str(p.order),
                (p.afhp, p.performance),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        except Exception:
            pass
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
    points: List[CurvePoint], result: SamplingResult, test_name: str
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
            px = getattr(p, "percentile", getattr(p, "desired_percentile", 0.0))
            f.write(f"{px:.6f}\t{p.afhp:.6f}\t{p.performance:.6f}\t{p.repeats_used}\n")
    return {
        "percentile_to_afhp": p_plot,
        "afhp_to_performance": xy_plot,
        "directory": artifacts_dir,
    }


def print_artifact_summary(artifacts: Dict[str, Optional[Path]]) -> None:
    if artifacts.get("directory"):
        print(f"Artifacts saved in: {artifacts['directory']}")
    for k, v in artifacts.items():
        if k != "directory":
            print(f"  - {k}: {v}")
