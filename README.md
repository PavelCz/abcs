# ACS - Adaptive Coverage Sampling for Noisy Monotonic Functions

A Python library for efficient sampling of monotonic curves with coverage guarantees. Includes a new joint-coverage sampler that adaptively achieves uniform coverage on both AFHP (x) and performance (y).

## Overview

ACS provides samplers to efficiently sample points along monotonic curves while ensuring comprehensive coverage across both input and output dimensions. It's particularly useful for:

- Threshold evaluation in machine learning
- Performance curve characterization
- Trade-off analysis between competing metrics
- Any scenario requiring uniform sampling of monotonic relationships

## Features

- **Joint coverage**: JointCoverageSampler enforces max normalized neighbor gaps on both axes
- **Adaptive**: Gap-driven refinement focuses evaluations where needed most
- **Noise-tolerant**: Resolves local non-monotonicity via targeted re-runs
- **Minimal dependencies**: Only requires NumPy

## Installation

```bash
pip install acs
```

Or install from source:

```bash
git clone https://github.com/yourusername/acs.git
cd acs
pip install -e .
```

## Quick Start (Joint Coverage)

```python
from acs import JointCoverageSampler

def eval_at_percentile(p: float):
  threshold = p * 100.0
  afhp = threshold
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

## Documentation

For detailed documentation, see the [docs](docs/) directory:

- [Joint Sampler API](docs/api_reference.md)

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.
