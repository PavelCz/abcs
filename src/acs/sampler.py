"""
Legacy binary-search sampler exposing the same public interface as JointCoverageSampler.

This module provides a compatibility implementation that matches the constructor and
the return types of the joint sampler. Internally it focuses on covering the AFHP
axis using binary search in percentile space and reports coverage metrics for both
axes using the collected points.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from acs.joint_sampler import CurvePoint, SamplingResult


@dataclass
class _Obs:
    percentile: float
    afhp: float
    performance: float
    order: int


class BinarySearchSampler:
    """
    Adaptive sampler with the same interface as JointCoverageSampler.

    Constructor parameters and the run() return value mirror those of
    `JointCoverageSampler` so the two samplers can be swapped by users.
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

        self._eval_p = eval_at_percentile
        self._eval_lo = eval_at_lower_extreme
        self._eval_hi = eval_at_upper_extreme
        self._coverage_fraction = coverage_fraction
        self._max_total_evals = max_total_evals

        # Derived AFHP-bin count chosen to satisfy the requested fraction on x-axis
        self._num_bins = max(2, int(np.ceil(1.0 / coverage_fraction)))

        self._observations: List[_Obs] = []
        self._total_evals: int = 0
        self._early_stop_reason: Optional[str] = None

        # Computed after seeding extremes
        self._afhp_min: Optional[float] = None
        self._afhp_max: Optional[float] = None
        self._bin_edges: Optional[np.ndarray] = None
        self._bin_repr: List[Optional[_Obs]] = [None] * self._num_bins

    # -----------------
    # Public entrypoint
    # -----------------

    def run(self) -> SamplingResult:
        self._seed_extremes()
        if self._coverage_satisfied():
            return self._build_result()

        # Binary-search fill in percentile space to populate AFHP bins
        left_p = 0.0
        right_p = 1.0
        left_bin = self._determine_bin(self._bin_repr_non_none(0).afhp)
        right_bin = self._determine_bin(self._bin_repr_non_none(-1).afhp)
        if left_bin > right_bin:
            left_bin, right_bin = right_bin, left_bin

        self._binary_fill(left_p, right_p, left_bin, right_bin)

        return self._build_result()

    # -----------------
    # Initialization
    # -----------------

    def _seed_extremes(self) -> None:
        if self._observations:
            return
        afhp_lo, perf_lo = self._safe_eval_lo()
        self._add_observation(percentile=0.0, afhp=afhp_lo, performance=perf_lo)

        afhp_hi, perf_hi = self._safe_eval_hi()
        self._add_observation(percentile=1.0, afhp=afhp_hi, performance=perf_hi)

        # Establish AFHP range and bin edges
        self._afhp_min = min(afhp_lo, afhp_hi)
        self._afhp_max = max(afhp_lo, afhp_hi)
        self._bin_edges = np.linspace(self._afhp_min, self._afhp_max, self._num_bins + 1)

        # Assign extremes to bins
        self._assign_to_bin(self._observations[0])
        self._assign_to_bin(self._observations[1])

    # -------------
    # Evaluations
    # -------------

    def _safe_eval_p(self, p: float) -> Tuple[float, float]:
        afhp, perf = self._eval_p(float(np.clip(p, 0.0, 1.0)))
        self._validate_outputs(afhp, perf)
        self._total_evals += 1
        # Update AFHP range if needed and edges accordingly
        if self._afhp_min is None or afhp < self._afhp_min:
            self._afhp_min = afhp
        if self._afhp_max is None or afhp > self._afhp_max:
            self._afhp_max = afhp
        if self._afhp_min is not None and self._afhp_max is not None and self._afhp_max > self._afhp_min:
            self._bin_edges = np.linspace(self._afhp_min, self._afhp_max, self._num_bins + 1)
        return afhp, perf

    def _safe_eval_lo(self) -> Tuple[float, float]:
        afhp, perf = self._eval_lo()
        self._validate_outputs(afhp, perf)
        self._total_evals += 1
        return afhp, perf

    def _safe_eval_hi(self) -> Tuple[float, float]:
        afhp, perf = self._eval_hi()
        self._validate_outputs(afhp, perf)
        self._total_evals += 1
        return afhp, perf

    @staticmethod
    def _validate_outputs(afhp: float, performance: float) -> None:
        for name, value in ("afhp", afhp), ("performance", performance):
            if value != value:
                raise ValueError(f"{name} is NaN from evaluation")
            if value == float("inf") or value == float("-inf"):
                raise ValueError(f"{name} is infinite from evaluation")

    # -----------------
    # Bin utilities
    # -----------------

    def _determine_bin(self, afhp: float) -> int:
        assert self._bin_edges is not None
        if afhp >= self._bin_edges[-1]:
            return self._num_bins - 1
        for i in range(self._num_bins):
            if self._bin_edges[i] <= afhp < self._bin_edges[i + 1]:
                return i
        return min(max(int(self._num_bins / 2), 0), self._num_bins - 1)

    def _assign_to_bin(self, obs: _Obs) -> None:
        idx = self._determine_bin(obs.afhp)
        if self._bin_repr[idx] is None:
            self._bin_repr[idx] = obs

    def _bin_repr_non_none(self, index: int) -> _Obs:
        candidates = [b for b in self._bin_repr if b is not None]
        if not candidates:
            raise RuntimeError("No observations assigned to bins")
        return candidates[index]

    def _bins_remaining(self, left_bin: int, right_bin: int) -> bool:
        for i in range(left_bin + 1, right_bin):
            if 0 <= i < self._num_bins and self._bin_repr[i] is None:
                return True
        return False

    # -----------------
    # Core search
    # -----------------

    def _binary_fill(self, left_p: float, right_p: float, left_bin: int, right_bin: int) -> None:
        if self._total_evals >= self._max_total_evals:
            self._early_stop_reason = "max_total_evals"
            return
        if right_bin - left_bin <= 1:
            return

        mid_p = 0.5 * (left_p + right_p)
        afhp, perf = self._safe_eval_p(mid_p)
        obs = self._add_observation(mid_p, afhp, perf)
        self._assign_to_bin(obs)

        mid_bin = self._determine_bin(afhp)

        if self._bins_remaining(left_bin, mid_bin):
            self._binary_fill(left_p, mid_p, left_bin, mid_bin)
            if self._early_stop_reason is not None:
                return
        if self._bins_remaining(mid_bin, right_bin):
            self._binary_fill(mid_p, right_p, mid_bin, right_bin)
            if self._early_stop_reason is not None:
                return

    # -----------------
    # Point management
    # -----------------

    def _add_observation(self, percentile: float, afhp: float, performance: float) -> _Obs:
        order = len(self._observations) + 1
        obs = _Obs(percentile=percentile, afhp=afhp, performance=performance, order=order)
        self._observations.append(obs)
        return obs

    # -----------------
    # Coverage & result
    # -----------------

    def _coverage_satisfied(self) -> bool:
        x_gap, y_gap = self._compute_gaps()
        return x_gap <= self._coverage_fraction and y_gap <= self._coverage_fraction

    def _compute_gaps(self) -> Tuple[float, float]:
        if not self._observations:
            return 1.0, 1.0
        by_x = sorted(self._observations, key=lambda o: o.afhp)
        x_min = by_x[0].afhp
        x_max = by_x[-1].afhp
        x_gap = 0.0
        if x_max > x_min:
            for i in range(len(by_x) - 1):
                gap = (by_x[i + 1].afhp - by_x[i].afhp) / (x_max - x_min)
                if gap > x_gap:
                    x_gap = gap
        by_y = sorted(self._observations, key=lambda o: o.performance)
        y_min = by_y[0].performance
        y_max = by_y[-1].performance
        y_gap = 0.0
        if y_max > y_min:
            for i in range(len(by_y) - 1):
                gap = (by_y[i + 1].performance - by_y[i].performance) / (y_max - y_min)
                if gap > y_gap:
                    y_gap = gap
        return x_gap, y_gap

    def _build_result(self) -> SamplingResult:
        x_gap, y_gap = self._compute_gaps()
        points = [
            CurvePoint(
                desired_percentile=obs.percentile,
                afhp=obs.afhp,
                performance=obs.performance,
                repeats_used=1,
                order=obs.order,
            )
            for obs in self._observations
        ]
        return SamplingResult(
            points=points,
            coverage_x_max_gap=x_gap,
            coverage_y_max_gap=y_gap,
            total_evals=self._total_evals,
            early_stop_reason=self._early_stop_reason,
            monotonicity_violations_remaining=False,
        )


