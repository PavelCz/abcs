# ABCS API Reference

## Classes

### `SamplePoint`

A dataclass representing a single evaluation point.

#### Attributes

- `input_value` (float): The input parameter value (e.g., threshold percentile)
- `output_value` (float): The measured primary output (e.g., AFHP)
- `metadata` (Dict[str, Any]): Additional data including secondary outputs

#### Example

```python
from abcs import SamplePoint

sample = SamplePoint(
    input_value=50.0,
    output_value=45.2,
    metadata={"return_mean": 78.5, "threshold_used": 0.234}
)
```

### `BinarySearchSampler`

The main ABCS algorithm implementation.

#### Constructor

```python
BinarySearchSampler(
    eval_function: Callable[[float], Tuple[float, Dict[str, Any]]],
    num_bins: int,
    input_range: Tuple[float, float] = (0.0, 100.0),
    output_range: Tuple[float, float] = (0.0, 100.0),
    input_to_threshold: Optional[Callable[[float], float]] = None,
    verbose: bool = True,
    return_bins: int = 0,
    max_additional_evals: int = 20,
)
```

##### Parameters

- **eval_function**: Function that takes an input value and returns `(output_value, metadata_dict)`
- **num_bins**: Number of bins to divide the primary output space into
- **input_range**: Range of valid input values `(min, max)`
- **output_range**: Range of expected primary output values `(min, max)`
- **input_to_threshold**: Optional function to convert input values to actual thresholds for evaluation
- **verbose**: Whether to print progress messages
- **return_bins**: Number of secondary output bins for refinement (0 = disabled)
- **max_additional_evals**: Maximum additional evaluations for secondary refinement

#### Methods

##### `run() -> List[Optional[SamplePoint]]`

Run Phase 1 only (primary axis coverage via binary search).

**Returns**: List of samples, one per bin (where possible)

**Example**:
```python
samples = sampler.run()
filled_samples = [s for s in samples if s is not None]
```

##### `run_with_return_refinement() -> List[Optional[SamplePoint]]`

Run the complete two-phase ABCS algorithm.

**Returns**: List of samples from Phase 1 (Phase 2 samples stored separately)

**Example**:
```python
samples = sampler.run_with_return_refinement()
all_samples = sampler.get_all_samples_including_refinement()
```

##### `get_filled_samples() -> List[SamplePoint]`

Get only the non-None samples from primary bins.

**Returns**: List of valid samples from primary bins

##### `get_all_samples() -> List[SamplePoint]`

Get all samples from Phase 1 in evaluation order.

**Returns**: List of all Phase 1 samples

##### `get_all_samples_including_refinement() -> List[SamplePoint]`

Get all samples including those from Phase 2 secondary refinement.

**Returns**: Combined list of all samples from both phases

##### `get_return_refinement_samples() -> List[SamplePoint]`

Get only the samples added during Phase 2 secondary refinement.

**Returns**: List of Phase 2 samples

##### `get_coverage_summary() -> Dict[str, Any]`

Get summary statistics about the primary axis sampling coverage.

**Returns**: Dictionary with coverage information:
- `bins_filled`: Number of primary bins filled
- `coverage_percentage`: Percentage of primary bins filled
- `output_range_covered`: Tuple of (min, max) primary output values covered
- `gaps`: List of uncovered primary output ranges
- `total_evaluations`: Total number of evaluations performed

**Example**:
```python
summary = sampler.get_coverage_summary()
print(f"Coverage: {summary['coverage_percentage']}%")
print(f"Evaluations: {summary['total_evaluations']}")
```

##### `determine_bin(output_value: float) -> int`

Determine which primary bin an output value falls into.

**Parameters**:
- **output_value**: Primary output value to categorize

**Returns**: Bin index (0-based)

**Raises**: `ValueError` if output_value is outside the expected range

##### `extract_return_value(sample: SamplePoint) -> float`

Extract secondary output value from sample metadata.

**Parameters**:
- **sample**: Sample point to extract from

**Returns**: Secondary output value

**Raises**: `ValueError` if secondary output cannot be found in metadata

The method tries multiple keys in this order:
1. `summary[split]["return_mean"]` for splits in ["test", "val", "eval"]
2. `"return_mean"` directly in metadata
3. `"return"` directly in metadata

#### Properties

- `num_bins`: Number of primary output bins
- `return_bins`: Number of secondary output bins
- `input_range`: Input value range
- `output_range`: Primary output value range
- `total_evals`: Total number of evaluations performed
- `bin_edges`: Primary bin edge values
- `verbose`: Whether progress messages are printed

## Usage Patterns

### Basic Usage (Primary Coverage Only)

```python
from abcs import BinarySearchSampler

def my_eval_function(x):
    y = some_computation(x)
    return y, {"metadata": "value"}

sampler = BinarySearchSampler(
    eval_function=my_eval_function,
    num_bins=10,
    verbose=True
)

samples = sampler.run()
coverage = sampler.get_coverage_summary()
```

### Two-Phase Usage (Primary + Secondary Coverage)

```python
from abcs import BinarySearchSampler

def my_eval_function(x):
    primary_output = compute_primary(x)
    secondary_output = compute_secondary(x)
    return primary_output, {"return_mean": secondary_output}

sampler = BinarySearchSampler(
    eval_function=my_eval_function,
    num_bins=15,
    return_bins=10,
    max_additional_evals=25,
    verbose=True
)

# Run both phases
samples = sampler.run_with_return_refinement()

# Get all samples
all_samples = sampler.get_all_samples_including_refinement()
refinement_samples = sampler.get_return_refinement_samples()

print(f"Phase 1 samples: {len(samples)}")
print(f"Phase 2 samples: {len(refinement_samples)}")
```

### Custom Input Transformation

```python
def percentile_to_threshold(percentile):
    if percentile == 0:
        return float("inf")
    elif percentile == 100:
        return float("-inf")
    else:
        return policy.train_percentile(100 - percentile)

sampler = BinarySearchSampler(
    eval_function=my_eval_function,
    num_bins=20,
    input_range=(0.0, 100.0),
    input_to_threshold=percentile_to_threshold,
    verbose=True
)
```

## Error Handling

### Common Exceptions

- **ValueError**: Raised when output values are outside expected ranges or required metadata is missing
- **RuntimeError**: May be raised during evaluation if the provided function fails

### Best Practices

1. **Evaluation Function**: Ensure your evaluation function handles edge cases gracefully
2. **Metadata Format**: Use consistent keys for secondary outputs in metadata
3. **Range Specification**: Set `input_range` and `output_range` to match your problem domain
4. **Budget Planning**: Set `max_additional_evals` based on available computational budget

## Performance Considerations

- **Primary Coverage**: Requires O(n log n) evaluations for n bins
- **Secondary Coverage**: Requires up to `max_additional_evals` additional evaluations
- **Memory Usage**: Stores all samples in memory; consider this for large-scale applications
- **Evaluation Cost**: The algorithm's efficiency depends on your evaluation function's computational cost