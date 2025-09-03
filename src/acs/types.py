"""
Type definitions for ABCS library.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# -----------------------------
# Public result data structures
# -----------------------------


@dataclass
class CurvePoint:
    """A single point on the trade-off curve.

    Fields represent aggregated (mean) values across repeats for the same input.
    """

    desired_percentile: float
    afhp: float
    performance: float
    repeats_used: int
    order: int
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SamplingResult:
    """Result of the joint-coverage sampling run."""

    points: List[CurvePoint]
    coverage_x_max_gap: float
    coverage_y_max_gap: float
    total_evals: int
    early_stop_reason: Optional[str]
    monotonicity_violations_remaining: bool
    info: Optional[Dict[str, Any]] = None  # Optional info dict for additional data
