# Requirements: Joint-coverage adaptive sampling (ABCS v2)

## Scope & goal
- Produce a set of evaluated points for the trade-off curve with x = AFHP (ask-for-help percentage) and y = performance.
- Guarantee user-specified coverage on both axes: no gaps larger than a normalized fraction when points are sorted by each axis separately.
- Assume performance is monotone non-decreasing in AFHP in expectation, but tolerate evaluation noise via re-runs where monotonicity appears violated.

## Inputs
- `eval_at_percentile(p: float) -> (afhp: float, performance: float)`
  - Interprets `p` as a desired training-data percentile; maps to a threshold and evaluates on the target eval set.
- `eval_at_lower_extreme() -> (afhp, performance)`
  - Threshold at +infinity (minimal AFHP); returns the lower bound point.
- `eval_at_upper_extreme() -> (afhp, performance)`
  - Threshold at -infinity (maximal AFHP); returns the upper bound point.
- `coverage_fraction: float in (0, 1]`
  - Maximum allowed normalized gap on both axes (e.g., 0.10).
- Optional configuration:
  - `max_total_evals`

## Outputs

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CurvePoint:
    percentile: float
    afhp: float
    performance: float
    repeats_used: int

@dataclass
class SamplingResult:
    points: List[CurvePoint]  # evaluated points (ordering unspecified)
    coverage_x_max_gap: float  # normalized max AFHP gap
    coverage_y_max_gap: float  # normalized max performance gap
    total_evals: int           # total evaluation calls including re-runs
    early_stop_reason: Optional[str]  # None if coverage achieved
    monotonicity_violations_remaining: bool
```

## Definitions
- Normalized gap on x-axis (AFHP): sort all points by `afhp` ascending; define gaps as `Δx_i = (afhp[i+1] - afhp[i]) / (x_max - x_min)` where `x_min, x_max` are the AFHPs from the lower/upper extremes; coverage on x is `max_i Δx_i`.
- Normalized gap on y-axis (performance): sort all points by `performance` ascending; define gaps `Δy_i = (performance[i+1] - performance[i]) / (y_max - y_min)` using extreme performance values; coverage on y is `max_i Δy_i`.
- Coverage criterion satisfied iff both `max_i Δx_i ≤ coverage_fraction` and `max_i Δy_i ≤ coverage_fraction`.

## Required behavior
- Always include and use the two extreme evaluations as endpoints for normalization and as initial curve endpoints.
- Iteratively add new evaluations until both axis-wise coverage criteria are satisfied (or a configured safety cap is hit).
- Selection of the next evaluation must be driven by the current worst coverage violation (largest normalized gap on either axis), such that added points reduce that violation over time.
- Direction-agnostic w.r.t. the underlying percentile→threshold mapping; algorithm reasons entirely in the output space (AFHP/performance).

## Monotonicity handling under noise
- Expected monotonicity: performance should be non-decreasing with AFHP.
- Detection: if, after adding or reusing points, there exists any adjacent pair in AFHP order with `performance[i] > performance[i+1]` (strict violation), the algorithm must attempt to resolve via re-runs at the involved inputs.
- Re-run policy:
  - Re-evaluate the offending inputs up to `max_reruns_per_point` each, aggregating results per point (e.g., by mean) for coverage and monotonicity checks.
  - Monotonicity checks and coverage calculations must use the aggregated estimates (not single noisy draws).
  - If violations persist after allowed re-runs, the algorithm may accept a local non-monotonicity but must still proceed to satisfy the coverage criteria; it must flag this condition in the summary.

## Coverage accounting and aggregation
- Use the aggregated AFHP and performance values per input when sorting and measuring gaps on both axes.
- Do not discard raw repeated measurements; retain them internally and expose counts in the summary for transparency.
- Enforce `min_separation_x`/`min_separation_y` when proposing or accepting new points to avoid near-duplicates that do not materially improve coverage.

## Stopping conditions
- Success: both axis-wise normalized max gaps ≤ `coverage_fraction`.
- Early termination (must be reported): hitting `max_total_evals`, or inability to make progress due to `min_separation_*` constraints or repeated non-monotonicity.

## Robustness & validation
- Handle degenerate ranges gracefully:
  - If `x_max == x_min` or `y_max == y_min`, report zero range on that axis and consider coverage satisfied for that axis by definition; continue to fill the other axis.
- Validate outputs:
  - Reject or retry evaluations that return NaN/Inf; report such events.
  - Clamp AFHP to [0, 1] only for reporting if required, but use raw values for internal detection unless configured otherwise.
- Ensure determinism given a fixed `random_seed` for any tie-breaking or stochastic steps.

## Reporting & telemetry
- Provide a machine-readable summary including:
  - Final counts: total unique inputs evaluated, total eval calls (including re-runs), per-point repeat counts.
  - Final normalized max gaps on both axes and which adjacent pairs determine them.
  - Any monotonicity violations remaining at stop time.
- Provide easy ways to access `points` sorted by AFHP and by performance for plotting (sorting may be done by the caller).

## Non-requirements (for clarity)
- No requirement to estimate or expose the percentile→threshold mapping details.
- No requirement to smooth or fit a curve; noise is handled via targeted re-runs only.
- No requirement to guarantee optimality of where to sample next, only that the procedure converges to the specified coverage thresholds under the stated assumptions.
