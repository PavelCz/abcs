"""
Joint-coverage adaptive sampler.

Implements a single-phase algorithm that adaptively evaluates points until the
maximum normalized neighbor gap on both axes (AFHP and performance) is below a
user-specified coverage fraction. Monotonicity (performance non-decreasing with
AFHP) is assumed in expectation and enforced under noise by re-running
violating points (bounded by the overall evaluation budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


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


@dataclass
class SamplingResult:
    """Result of the joint-coverage sampling run."""

    points: List[CurvePoint]
    coverage_x_max_gap: float
    coverage_y_max_gap: float
    total_evals: int
    early_stop_reason: Optional[str]
    monotonicity_violations_remaining: bool


# --------------------
# Internal data models
# --------------------


@dataclass
class _PointState:
    """Internal state for a sampled input with aggregation buffers."""

    percentile: float
    afhp_samples: List[float]
    performance_samples: List[float]
    order_index: int

    def add_observation(self, afhp: float, performance: float) -> None:
        self.afhp_samples.append(afhp)
        self.performance_samples.append(performance)

    @property
    def repeats_used(self) -> int:
        return len(self.afhp_samples)

    @property
    def afhp_mean(self) -> float:
        return sum(self.afhp_samples) / len(self.afhp_samples)

    @property
    def performance_mean(self) -> float:
        return sum(self.performance_samples) / len(self.performance_samples)


# -------------
# Sampler class
# -------------


class JointCoverageSampler:
    """Adaptive sampler achieving joint coverage on AFHP and performance.

    The sampler accepts three evaluation callables:
      - eval_at_percentile(p): Evaluate at desired percentile p and return (afhp, performance)
      - eval_at_lower_extreme(): Evaluate at the lower extreme and return (afhp, performance)
      - eval_at_upper_extreme(): Evaluate at the upper extreme and return (afhp, performance)

    The algorithm starts by evaluating extremes (assigned synthetic percentiles 0.0 and 1.0),
    then iteratively splits the worst normalized neighbor gap on either axis until both axes
    meet the requested coverage_fraction or the max_total_evals budget is exhausted.
    """

    def __init__(
        self,
        *,
        eval_at_percentile: Callable[[float], Tuple[float, float]],
        eval_at_lower_extreme: Callable[[], Tuple[float, float]],
        eval_at_upper_extreme: Callable[[], Tuple[float, float]],
        coverage_fraction: float,
        max_total_evals: int,
    ) -> None:
        if not (0.0 < coverage_fraction <= 1.0):
            raise ValueError("coverage_fraction must be in (0, 1]")
        if max_total_evals < 2:
            raise ValueError("max_total_evals must be at least 2 to include extremes")

        self._eval_at_percentile = eval_at_percentile
        self._eval_lower = eval_at_lower_extreme
        self._eval_upper = eval_at_upper_extreme
        self._coverage_fraction = coverage_fraction
        self._max_total_evals = max_total_evals

        self._points: List[_PointState] = []
        self._total_evals: int = 0
        self._early_stop_reason: Optional[str] = None

    # -----------------
    # Public entrypoint
    # -----------------

    def run(self) -> SamplingResult:
        """Execute the adaptive sampling loop until coverage or budget exhaustion."""
        self._seed_extremes()

        while not self._coverage_satisfied():
            if self._total_evals >= self._max_total_evals:
                self._early_stop_reason = "max_total_evals"
                break

            # Resolve monotonicity violations by re-running the offending points
            self._resolve_monotonicity_with_reruns()
            if self._total_evals >= self._max_total_evals:
                self._early_stop_reason = "max_total_evals"
                break

            if self._coverage_satisfied():
                break

            # Choose which adjacent pair to split based on the worst normalized gap
            axis, left_index, right_index = self._select_worst_gap_pair()

            # Determine the new percentile as the midpoint between the adjacent pair percentiles
            p_left = self._points[left_index].percentile
            p_right = self._points[right_index].percentile
            p_new = 0.5 * (p_left + p_right)

            # Guard against duplicates due to finite precision
            epsilon = 1e-9
            if abs(p_new - p_left) < epsilon:
                p_new = p_left + epsilon
            if abs(p_right - p_new) < epsilon:
                p_new = p_right - epsilon

            # Clamp to [0.0, 1.0]
            p_new = min(max(p_new, 0.0), 1.0)

            # Evaluate new point
            afhp, performance = self._safe_eval_at_percentile(p_new)
            self._insert_point(p_new, afhp, performance)

        x_gap, y_gap = self._compute_current_gaps()
        result = SamplingResult(
            points=[
                CurvePoint(
                    desired_percentile=pt.percentile,
                    afhp=pt.afhp_mean,
                    performance=pt.performance_mean,
                    repeats_used=pt.repeats_used,
                    order=getattr(pt, "order_index", -1),
                )
                for pt in self._points
            ],
            coverage_x_max_gap=x_gap,
            coverage_y_max_gap=y_gap,
            total_evals=self._total_evals,
            early_stop_reason=self._early_stop_reason,
            monotonicity_violations_remaining=self._has_monotonicity_violation(),
        )
        return result

    # -----------------
    # Initialization
    # -----------------

    def _seed_extremes(self) -> None:
        if self._points:
            return
        # Assign synthetic percentiles 0.0 and 1.0 for extremes
        afhp_low, perf_low = self._safe_eval_lower()
        self._points.append(
            _PointState(
                percentile=0.0,
                afhp_samples=[afhp_low],
                performance_samples=[perf_low],
                order_index=1,
            )
        )

        afhp_high, perf_high = self._safe_eval_upper()
        self._points.append(
            _PointState(
                percentile=1.0,
                afhp_samples=[afhp_high],
                performance_samples=[perf_high],
                order_index=2,
            )
        )

        # Keep points ordered by percentile
        self._points.sort(key=lambda s: s.percentile)

    # -----------------
    # Evaluations
    # -----------------

    def _safe_eval_at_percentile(self, p: float) -> Tuple[float, float]:
        afhp, performance = self._eval_at_percentile(p)
        self._validate_outputs(afhp, performance)
        self._total_evals += 1
        return afhp, performance

    def _safe_eval_lower(self) -> Tuple[float, float]:
        afhp, performance = self._eval_lower()
        self._validate_outputs(afhp, performance)
        self._total_evals += 1
        return afhp, performance

    def _safe_eval_upper(self) -> Tuple[float, float]:
        afhp, performance = self._eval_upper()
        self._validate_outputs(afhp, performance)
        self._total_evals += 1
        return afhp, performance

    @staticmethod
    def _validate_outputs(afhp: float, performance: float) -> None:
        for name, value in ("afhp", afhp), ("performance", performance):
            if value != value:  # NaN check
                raise ValueError(f"{name} is NaN from evaluation")
            if value == float("inf") or value == float("-inf"):
                raise ValueError(f"{name} is infinite from evaluation")

    # -----------------
    # Point management
    # -----------------

    def _insert_point(self, percentile: float, afhp: float, performance: float) -> None:
        # If a point with identical percentile exists, treat as a re-run
        for pt in self._points:
            if abs(pt.percentile - percentile) < 1e-12:
                pt.add_observation(afhp, performance)
                return
        self._points.append(
            _PointState(
                percentile=percentile,
                afhp_samples=[afhp],
                performance_samples=[performance],
                order_index=len(self._points) + 1,
            )
        )
        self._points.sort(key=lambda s: s.percentile)

    # -----------------
    # Coverage logic
    # -----------------

    def _coverage_satisfied(self) -> bool:
        x_gap, y_gap = self._compute_current_gaps()
        # Treat degenerate ranges as covered by definition (gap == 0.0)
        return x_gap <= self._coverage_fraction and y_gap <= self._coverage_fraction

    def _compute_current_gaps(self) -> Tuple[float, float]:
        # Sort by AFHP mean for x-axis gaps
        by_x = sorted(self._points, key=lambda s: s.afhp_mean)
        x_min = by_x[0].afhp_mean
        x_max = by_x[-1].afhp_mean
        x_gap = 0.0
        if x_max > x_min:
            for i in range(len(by_x) - 1):
                gap = (by_x[i + 1].afhp_mean - by_x[i].afhp_mean) / (x_max - x_min)
                if gap > x_gap:
                    x_gap = gap
        # Sort by performance mean for y-axis gaps
        by_y = sorted(self._points, key=lambda s: s.performance_mean)
        y_min = by_y[0].performance_mean
        y_max = by_y[-1].performance_mean
        y_gap = 0.0
        if y_max > y_min:
            for i in range(len(by_y) - 1):
                gap = (by_y[i + 1].performance_mean - by_y[i].performance_mean) / (
                    y_max - y_min
                )
                if gap > y_gap:
                    y_gap = gap
        return x_gap, y_gap

    def _select_worst_gap_pair(self) -> Tuple[str, int, int]:
        # Compute x-axis gaps in AFHP order (on means)
        x_order = sorted(
            range(len(self._points)), key=lambda idx: self._points[idx].afhp_mean
        )
        x_min = self._points[x_order[0]].afhp_mean
        x_max = self._points[x_order[-1]].afhp_mean
        worst_x_gap = -1.0
        worst_x_pair = (x_order[0], x_order[1]) if len(x_order) >= 2 else (0, 0)
        if x_max > x_min:
            for i in range(len(x_order) - 1):
                a = self._points[x_order[i]].afhp_mean
                b = self._points[x_order[i + 1]].afhp_mean
                gap = (b - a) / (x_max - x_min)
                if gap > worst_x_gap:
                    worst_x_gap = gap
                    worst_x_pair = (x_order[i], x_order[i + 1])
        else:
            worst_x_gap = 0.0

        # Compute y-axis gaps in performance order (on means)
        y_order = sorted(
            range(len(self._points)), key=lambda idx: self._points[idx].performance_mean
        )
        y_min = self._points[y_order[0]].performance_mean
        y_max = self._points[y_order[-1]].performance_mean
        worst_y_gap = -1.0
        worst_y_pair = (y_order[0], y_order[1]) if len(y_order) >= 2 else (0, 0)
        if y_max > y_min:
            for i in range(len(y_order) - 1):
                a = self._points[y_order[i]].performance_mean
                b = self._points[y_order[i + 1]].performance_mean
                gap = (b - a) / (y_max - y_min)
                if gap > worst_y_gap:
                    worst_y_gap = gap
                    worst_y_pair = (y_order[i], y_order[i + 1])
        else:
            worst_y_gap = 0.0

        # Choose the axis with the larger gap (tie-break by x)
        if worst_x_gap >= worst_y_gap:
            left_idx, right_idx = worst_x_pair
            return "x", left_idx, right_idx
        else:
            left_idx, right_idx = worst_y_pair
            return "y", left_idx, right_idx

    # -----------------------------
    # Monotonicity under noise
    # -----------------------------

    def _has_monotonicity_violation(self) -> bool:
        # Sort by AFHP mean and check performance means are non-decreasing
        ordered = sorted(self._points, key=lambda s: s.afhp_mean)
        for i in range(len(ordered) - 1):
            if ordered[i].performance_mean > ordered[i + 1].performance_mean:
                return True
        return False

    def _resolve_monotonicity_with_reruns(self) -> None:
        # Keep re-running violating adjacent pairs while budget allows
        while self._has_monotonicity_violation():
            if self._total_evals + 2 > self._max_total_evals:
                # Not enough budget for a full re-run of both points
                return
            # Identify first violating adjacent pair
            ordered_indices = sorted(
                range(len(self._points)), key=lambda idx: self._points[idx].afhp_mean
            )
            violation_pair: Optional[Tuple[int, int]] = None
            for i in range(len(ordered_indices) - 1):
                a_idx = ordered_indices[i]
                b_idx = ordered_indices[i + 1]
                if (
                    self._points[a_idx].performance_mean
                    > self._points[b_idx].performance_mean
                ):
                    violation_pair = (a_idx, b_idx)
                    break
            if violation_pair is None:
                return

            a_idx, b_idx = violation_pair
            # Re-run both points once
            a_pt = self._points[a_idx]
            b_pt = self._points[b_idx]

            # If either is an extreme, re-evaluate via extremes; otherwise via percentile
            if abs(a_pt.percentile - 0.0) < 1e-12:
                afhp_a, perf_a = self._safe_eval_lower()
            elif abs(a_pt.percentile - 1.0) < 1e-12:
                afhp_a, perf_a = self._safe_eval_upper()
            else:
                afhp_a, perf_a = self._safe_eval_at_percentile(a_pt.percentile)
            a_pt.add_observation(afhp_a, perf_a)

            if abs(b_pt.percentile - 0.0) < 1e-12:
                afhp_b, perf_b = self._safe_eval_lower()
            elif abs(b_pt.percentile - 1.0) < 1e-12:
                afhp_b, perf_b = self._safe_eval_upper()
            else:
                afhp_b, perf_b = self._safe_eval_at_percentile(b_pt.percentile)
            b_pt.add_observation(afhp_b, perf_b)

            # Loop again to re-check after updated means
            if self._total_evals >= self._max_total_evals:
                return
