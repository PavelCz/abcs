# CLAUDE.md

This file provides guidance to Claude Code when working with the ABCS (Adaptive Binary Coverage Search) library.

## Project Overview

ABCS is a Python library that implements the Adaptive Binary Coverage Search algorithm for efficient sampling of monotonic curves. The algorithm uses a two-phase approach to ensure comprehensive coverage across both primary and secondary output dimensions with minimal evaluations.

**Python Version**: Requires Python 3.8 or higher.

### Core Algorithm

The algorithm operates in two phases:
1. **Phase 1: Primary Coverage** - Uses binary search to fill bins along the primary output axis (e.g., AFHP)
2. **Phase 2: Secondary Refinement** - Optionally fills gaps along secondary output axis (e.g., return values)

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
pytype src/abcs/*.py --verbosity=0 2>/dev/null

# Run all quality checks at once
ci/format_and_check.sh

# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=abcs --cov-report=html
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

1. **`src/abcs/types.py`**: Core data structures
   - `SamplePoint`: Represents a single evaluation point with input, output, and metadata

2. **`src/abcs/sampler.py`**: Main algorithm implementation
   - `BinarySearchSampler`: The core ABCS algorithm class
   - Handles both single-phase and two-phase coverage

3. **`src/abcs/__init__.py`**: Public API
   - Exports `BinarySearchSampler` and `SamplePoint`
   - Defines version and public interface

### Key Design Patterns

1. **Generic Interface**: Algorithm is domain-agnostic, works with any monotonic function
2. **Configurable Coverage**: Both primary and secondary axis coverage can be tuned
3. **Metadata Preservation**: All evaluation results and metadata are preserved
4. **Progress Tracking**: Optional verbose output for monitoring algorithm progress

## Important Implementation Details

### Algorithm Guarantees
- **Primary Coverage**: Guarantees 100% bin coverage when evaluation function spans the full output range
- **Secondary Coverage**: Best-effort coverage based on available evaluation budget
- **Monotonicity**: Assumes monotonic relationship between input and primary output

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
from abcs import BinarySearchSampler

def eval_func(x):
    return monotonic_function(x), {"metadata": "value"}

sampler = BinarySearchSampler(eval_func, num_bins=10)
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
samples = sampler.run_with_return_refinement()
all_samples = sampler.get_all_samples_including_refinement()
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
samples = sampler.run_with_return_refinement()
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
- Use `get_coverage_summary()` to inspect coverage statistics
- Check `get_all_samples()` to examine evaluation history
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