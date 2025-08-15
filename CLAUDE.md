# CLAUDE.md

This file provides guidance to Claude Code when working with the ACS (Adaptive Coverage Sampling) library.

## Project Overview

ACS is a Python library for sampling monotonic curves with coverage guarantees. It includes a joint-coverage sampler that adaptively ensures coverage on both axes.

**Python Version**: Requires Python 3.8 or higher.

### Core Algorithm

The algorithm operates in two phases:
1. **Phase 1: Primary Coverage** - Uses recursive binary search to fill bins along the primary output axis (e.g., AFHP)
2. **Phase 2: Return Gap Filling** - Uses recursive binary bisection to systematically fill gaps along secondary output axis (e.g., return values)

## Key Commands

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with example dependencies (for matplotlib)
pip install -e ".[examples]"
```

### Code Quality
```bash
# Format code
ruff format src/ tests/ examples/

# Lint code
ruff check src/ tests/ examples/

# Type check (suppress verbose output like CI does)
pytype src/acs/*.py --verbosity=0 2>/dev/null

# Run all quality checks at once
ci/format_and_check.sh

# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=acs --cov-report=html
```

### Testing
```bash
# Run the main test suite
python tests/test_coverage.py

# Run specific test functions
python -m pytest tests/test_coverage.py::test_full_coverage -v

# Run examples to verify they work
python examples/basic_usage.py
```

### Building and Distribution
```bash
# Build the package
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI (when ready)
twine upload dist/*
```

## Architecture Overview

### Core Components

1. **`src/acs/joint_sampler.py`**: Joint coverage sampler
2. **`src/acs/__init__.py`**: Public API for ACS

### Key Design Patterns

1. **Generic Interface**: Algorithm is domain-agnostic, works with any monotonic function
2. **Configurable Coverage**: Both primary and secondary axis coverage can be tuned
3. **Metadata Preservation**: All evaluation results and metadata are preserved
4. **Progress Tracking**: Optional verbose output for monitoring algorithm progress

## Important Implementation Details

### Algorithm Guarantees
- **Primary Coverage**: Guarantees 100% bin coverage when evaluation function spans the full output range
- **Secondary Coverage**: Uses recursive binary bisection for systematic gap filling
- **Monotonicity**: Assumes monotonic relationship between input and primary output

### Phase 2 Binary Bisection Algorithm
The improved Phase 2 implementation uses a systematic approach:

1. **Gap Identification**: Identifies contiguous intervals of empty return bins
2. **Bracket Finding**: Locates samples that bracket each gap interval
3. **Recursive Bisection**: Applies binary search within each gap interval
4. **Convergence Detection**: Stops when precision thresholds are reached or evaluation limits hit
5. **Safety Mechanisms**: Includes iteration limits and precision-based termination

This approach ensures more reliable and comprehensive return value coverage compared to the previous interpolation-based method.

### Performance Characteristics
- **Time Complexity**: O(n log n) for n primary bins + O(m) for m secondary bins
- **Space Complexity**: O(n + m + total_evaluations)
- **Evaluation Efficiency**: Minimizes function evaluations while maximizing coverage

### Configuration Options
- `num_bins`: Number of primary axis bins (affects coverage granularity)
- `return_bins`: Number of secondary axis bins (0 = disable Phase 2)
- `max_additional_evals`: Budget for secondary axis refinement (ignored in unbounded mode)
- `unbounded_mode`: Removes evaluation limits for theoretical convergence guarantees
- `input_range`/`output_range`: Expected value ranges for proper binning
- `input_to_threshold`: Optional transformation function for input values

### Unbounded Mode
When `unbounded_mode=True`, the algorithm removes iteration limits and continues until all bins are filled or no progress can be made:

- **Truly Unbounded**: No artificial iteration limits - only precision-based convergence detection
- **Theoretical Guarantees**: Provides convergence guarantees for monotonic functions
- **Practical Safety**: Includes safety limit (max 10,000 total evaluations) to prevent infinite execution
- **Enhanced Coverage**: Achieves better coverage than bounded mode in most cases
- **Smart Termination**: Stops when precision threshold reached or consecutive failures detected
- **Use Cases**: Recommended for critical applications where maximum coverage is required

**Performance Trade-off**: Unbounded mode may use more evaluations but guarantees theoretical convergence.

## Testing Strategy

### Test Coverage
The test suite verifies:
1. **Coverage Guarantees**: 100% coverage achieved under proper conditions
2. **Parameter Robustness**: Algorithm works across different parameter combinations
3. **Edge Cases**: Handles boundary conditions and degenerate cases
4. **Integration**: API works correctly for typical use cases

### Test Functions
- `test_full_coverage()`: Main coverage verification
- `test_coverage_with_different_parameters()`: Parameter robustness
- `test_afhp_coverage_guarantee()`: Primary axis guarantee verification
- `test_guaranteed_full_coverage()`: Linear function coverage test
- `test_unbounded_mode()`: Unbounded vs bounded mode comparison
- `test_unbounded_mode_convergence()`: Convergence with pathological functions

## Usage Patterns

### Basic Usage (Primary Coverage Only)
```python
from acs import JointCoverageSampler

def eval_func(x):
    return monotonic_function(x), {"metadata": "value"}

sampler = JointCoverageSampler(
    eval_at_percentile=lambda p: (p*100.0, 25.0 + 60.0*p),
    eval_at_lower_extreme=lambda: (0.0, 25.0),
    eval_at_upper_extreme=lambda: (100.0, 85.0),
    coverage_fraction=0.1,
    max_total_evals=200,
)
samples = sampler.run()
```

### Two-Phase Usage (Primary + Secondary Coverage)
```python
sampler = BinarySearchSampler(
    eval_func, 
    num_bins=15, 
    return_bins=10,
    max_additional_evals=25
)
result = sampler.run()
all_samples = result.points
```

### Custom Input Transformation
```python
def percentile_to_threshold(percentile):
    # Custom transformation logic
    return some_threshold_function(percentile)

sampler = BinarySearchSampler(
    eval_func,
    num_bins=20,
    input_to_threshold=percentile_to_threshold
)
```

### Unbounded Mode Usage (Maximum Coverage)
```python
sampler = BinarySearchSampler(
    eval_func,
    num_bins=15,
    return_bins=10,
    unbounded_mode=True,  # Removes evaluation limits for convergence
    verbose=True
)
result = sampler.run()
# Achieves theoretical convergence guarantees with safety mechanisms
```

## Common Development Tasks

### Adding New Features
1. Add functionality to `sampler.py`
2. Update tests in `test_coverage.py`
3. Add examples if user-facing
4. Update documentation in `docs/`

### Performance Optimization
- Profile with `cProfile` on large evaluation budgets
- Monitor memory usage for large sample collections
- Consider evaluation function caching for expensive computations

### API Changes
- Maintain backward compatibility in public interface
- Update version in `pyproject.toml` and `__init__.py`
- Add deprecation warnings for removed features
- Update documentation and examples

## Debugging and Troubleshooting

### Common Issues
1. **Low Coverage**: Check that evaluation function spans expected output range
2. **Slow Performance**: Reduce bin counts or evaluation budget
3. **Memory Issues**: Process samples in batches for large-scale applications
4. **Convergence Problems**: Verify monotonicity assumption holds

### Debugging Tools
- Set `verbose=True` for algorithm progress output
- Use `result.info` to inspect coverage statistics
- Use `result.points` to examine samples
- Plot results to visualize coverage patterns

## Dependencies

### Runtime Dependencies
- `numpy>=1.20.0`: Core numerical operations and array handling

### Development Dependencies
- `pytest>=7.0`: Testing framework
- `pytest-cov>=4.0`: Coverage reporting
- `ruff>=0.1.0`: Fast Python linter and formatter (replaces black + flake8)
- `pytype>=2023.04.11`: Google's static type checker

### Example Dependencies
- `matplotlib>=3.5.0`: For visualization examples

## Contributing Guidelines

### Code Style
- Use ruff for formatting and linting (replaces black and flake8)
- Follow PEP 8 naming conventions
- Add type hints for all public functions (checked with pytype)
- Include docstrings for all public APIs

### Testing Requirements
- All new features must include tests
- Maintain 100% test coverage for core algorithm
- Include both unit tests and integration tests
- Test edge cases and error conditions

### Documentation Requirements
- Update API reference for public interface changes
- Add examples for new features
- Update algorithm explanation for algorithmic changes
- Keep README current with installation and usage

## Performance Benchmarks

Expected performance characteristics:
- **10 bins, basic coverage**: ~10-12 evaluations
- **20 bins, basic coverage**: ~20-25 evaluations  
- **10 bins + 8 return bins**: ~15-20 evaluations
- **Memory usage**: ~1KB per sample point

## Best Practices

1. **Evaluation Function Design**: Keep evaluation functions pure and deterministic when possible
2. **Parameter Selection**: Start with reasonable bin counts (5-20) and increase as needed
3. **Budget Management**: Set `max_additional_evals` based on computational constraints
4. **Range Specification**: Set input/output ranges to match your problem domain
5. **Metadata Usage**: Store all relevant information in metadata for post-analysis

## Integration Notes

This library is designed to be:
- **Framework-agnostic**: Works with any Python evaluation function
- **Minimal dependencies**: Only requires NumPy for core functionality
- **Extensible**: Easy to add domain-specific wrapper functions
- **Testable**: Comprehensive test suite ensures reliability

## Development Best Practices

- Always commit changes in small atomic commits