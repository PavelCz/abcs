# API Reference: JointCoverageSampler

## Overview
Adaptive single-phase sampler that ensures joint coverage on AFHP (x) and performance (y) by iteratively splitting the largest normalized neighbor gap on either axis. Handles noisy evaluations by re-running offending adjacent pairs to restore monotonicity in expectation.

## Data classes

### CurvePoint
Represents an evaluated point with aggregated statistics.
- percentile (float): Input percentile used for evaluation (0.0–1.0)
- afhp (float): Mean AFHP across repeats for this percentile
- performance (float): Mean performance across repeats for this percentile
- repeats_used (int): Number of times this percentile was evaluated
- order (int): Sampling order index (1-based) useful for visualization

### SamplingResult
Structured result of a sampling run.
- points (List[CurvePoint]): Collected points (ordering unspecified)
- coverage_x_max_gap (float): Max normalized neighbor gap on AFHP axis
- coverage_y_max_gap (float): Max normalized neighbor gap on performance axis
- total_evals (int): Total evaluation calls including re-runs
- early_stop_reason (Optional[str]): None if coverage achieved; otherwise a short reason (e.g., "max_total_evals")
- monotonicity_violations_remaining (bool): True if any violations remain at stop

## Class: JointCoverageSampler

### Constructor
```python
JointCoverageSampler(
    *,
    eval_at_percentile: Callable[[float], Tuple[float, float]],
    eval_at_lower_extreme: Callable[[], Tuple[float, float]],
    eval_at_upper_extreme: Callable[[], Tuple[float, float]],
    coverage_fraction: float,
    max_total_evals: int,
)
```
Parameters:
- eval_at_percentile: Evaluate at desired percentile p∈[0,1] and return (afhp, performance)
- eval_at_lower_extreme: Evaluate lower extreme (e.g., threshold = +∞) and return (afhp, performance)
- eval_at_upper_extreme: Evaluate upper extreme (e.g., threshold = −∞) and return (afhp, performance)
- coverage_fraction: Maximum allowed normalized neighbor gap on both axes (e.g., 0.10)
- max_total_evals: Global evaluation budget (includes re-runs)

### Method
```python
run() -> SamplingResult
```
Executes the adaptive loop until both axes meet coverage_fraction or the budget is exhausted.

Returns: SamplingResult

### Example
```python
from abcs import JointCoverageSampler

def eval_at_percentile(p: float):
    threshold = p * 100.0
    afhp = threshold  # example linear mapping
    performance = 25.0 + 0.6 * (afhp / 100.0) * 100.0
    return afhp, performance

def eval_at_lower_extreme():
    return 0.0, 25.0

def eval_at_upper_extreme():
    return 100.0, 85.0

sampler = JointCoverageSampler(
    eval_at_percentile=eval_at_percentile,
    eval_at_lower_extreme=eval_at_lower_extreme,
    eval_at_upper_extreme=eval_at_upper_extreme,
    coverage_fraction=0.10,
    max_total_evals=200,
)
result = sampler.run()
print(result.coverage_x_max_gap, result.coverage_y_max_gap)
```
