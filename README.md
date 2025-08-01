# ABCS - Adaptive Binary Coverage Search

A Python library for efficient sampling of monotonic curves using adaptive binary search with coverage guarantees.

## Overview

ABCS (Adaptive Binary Coverage Search) is a two-stage algorithm designed to efficiently sample points along monotonic curves while ensuring comprehensive coverage across both input and output dimensions. It's particularly useful for:

- Threshold evaluation in machine learning
- Performance curve characterization
- Trade-off analysis between competing metrics
- Any scenario requiring uniform sampling of monotonic relationships

## Features

- **Efficient**: Uses binary search to minimize evaluations
- **Adaptive**: Automatically identifies and fills coverage gaps
- **Two-stage approach**: 
  - Stage 1: Ensures complete coverage along the primary axis
  - Stage 2: Refines coverage along the secondary axis (optional)
- **Guaranteed coverage**: Achieves 100% bin coverage when function spans full range
- **Minimal dependencies**: Only requires NumPy

## Installation

```bash
pip install abcs
```

Or install from source:

```bash
git clone https://github.com/yourusername/abcs.git
cd abcs
pip install -e .
```

## Quick Start

```python
from abcs import BinarySearchSampler

# Define your evaluation function
def evaluate(x):
    # Your evaluation logic here
    output = some_computation(x)
    metadata = {"additional_info": value}
    return output, metadata

# Create sampler
sampler = BinarySearchSampler(
    eval_function=evaluate,
    num_bins=10,  # Number of bins for primary axis
    return_bins=8,  # Number of bins for secondary axis (0 to disable)
    input_range=(0.0, 100.0),
    output_range=(0.0, 100.0),
    verbose=True
)

# Run the algorithm
samples = sampler.run_with_return_refinement()

# Check coverage
summary = sampler.get_coverage_summary()
print(f"Coverage: {summary['coverage_percentage']}%")
```

## Documentation

For detailed documentation, see the [docs](docs/) directory:
- [Algorithm Explanation](docs/algorithm_explanation.md)
- [API Reference](docs/api_reference.md)

## Examples

See the [examples](examples/) directory for complete examples:
- [Basic Usage](examples/basic_usage.py)
- [Visualization](examples/visualization.py)

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use ABCS in your research, please cite:

```bibtex
@software{abcs,
  title = {ABCS: Adaptive Binary Coverage Search},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/abcs}
}
```