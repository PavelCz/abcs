# Detailed Algorithm Explanation: Adaptive Binary Coverage Search (ABCS)

## Problem Statement

When evaluating threshold-based coordination policies or any monotonic functions, we need to understand the relationship between:
- **Input**: Input parameter (e.g., threshold percentile 0-100%)
- **Primary Output**: Primary metric (e.g., Ask-for-help percentage (AFHP) 0-100%)
- **Secondary Output**: Performance metric (e.g., Return/reward achieved)

This creates a monotonic curve where the input has a predictable relationship with the primary output.

## The ABCS Solution

ABCS uses a two-phase approach to efficiently sample monotonic curves with coverage guarantees:

### Phase 1: Primary Axis Coverage via Binary Search

The algorithm uses binary search to fill bins along the primary output axis. Here's how it works:

#### 1. **Monotonicity Exploitation**

Since the input-to-primary-output mapping is monotonic:
- Lower input → Higher threshold → Less help → Lower primary output
- Higher input → Lower threshold → More help → Higher primary output

This monotonicity enables binary search.

#### 2. **Bin-Based Coverage**

The algorithm divides the primary output range into equal bins (e.g., [0-10%], [10-20%], ..., [90-100%]). The goal is to have at least one sample in each bin, ensuring uniform coverage along the primary axis.

#### 3. **Recursive Binary Search**

```
Initial state: Evaluate at 0% and 100% input values
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  ✓  │     │     │     │     │     │     │     │     │  ✓  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  0-10  10-20 20-30 30-40 40-50 50-60 60-70 70-80 80-90 90-100

Step 1: Evaluate at 50% input (middle)
        Suppose it gives primary output = 45%
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  ✓  │     │     │     │  ✓  │     │     │     │     │  ✓  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Step 2: Recursively search [0%, 50%] and [50%, 100%]
        Continue until all bins are filled
```

### Phase 2: Secondary Axis Refinement (Optional)

After Phase 1, ABCS can optionally refine coverage along the secondary output axis (e.g., return values):

#### 1. **Gap Identification**

Analyze secondary output values from Phase 1 samples to identify coverage gaps.

#### 2. **Return Binning**

Create bins along the secondary output axis based on the observed range.

#### 3. **Targeted Sampling**

Use binary search to find input values that produce secondary outputs in empty bins.

## Algorithm Walkthrough

### Phase 1: Primary Coverage

```python
def binary_search_fill(left_input, right_input, left_bin, right_bin):
    # 1. Calculate middle input value
    middle_input = (left_input + right_input) / 2
    
    # 2. Evaluate at middle point
    primary_output, metadata = eval_function(middle_input)
    
    # 3. Determine which bin this primary output falls into
    bin_idx = determine_bin(primary_output)
    
    # 4. Fill the bin if empty
    if bin_is_empty(bin_idx):
        fill_bin(bin_idx, sample)
    
    # 5. Recursively search left and right
    if bins_remain(left_bin, bin_idx):
        binary_search_fill(left_input, middle_input, left_bin, bin_idx)
    
    if bins_remain(bin_idx, right_bin):
        binary_search_fill(middle_input, right_input, bin_idx, right_bin)
```

### Phase 2: Secondary Coverage

```python
def fill_return_gaps(initial_samples):
    # 1. Extract secondary outputs from initial samples
    secondary_outputs = [extract_secondary(s) for s in initial_samples]
    
    # 2. Create bins along secondary axis
    secondary_bins = create_bins(min(secondary_outputs), max(secondary_outputs))
    
    # 3. Identify empty bins
    empty_bins = find_empty_bins(secondary_bins, secondary_outputs)
    
    # 4. For each empty bin, use binary search to find samples
    for target_range in empty_bins:
        sample = search_for_secondary_range(initial_samples, target_range)
        if sample:
            additional_samples.append(sample)
```

## Characteristics of ABCS

### Advantages

1. **Efficiency**: O(n log n) evaluations for n bins
2. **Coverage Guarantee**: Ensures 100% primary axis coverage when function spans full range
3. **Adaptive**: Automatically identifies and fills coverage gaps
4. **Dual-Axis**: Covers both primary and secondary output dimensions
5. **Robust**: Handles non-linear input-to-output mappings

### Complexity Analysis

- **Phase 1**: O(n log n) evaluations for n primary bins
- **Phase 2**: O(m) additional evaluations for m secondary bins
- **Total**: O(n log n + m) evaluations

## Limitations and Improvements

### Current Limitations

1. **Fixed Resolution**: Number of bins is predetermined
2. **Uniform Bins**: All bins have equal width
3. **Monotonicity Assumption**: Requires monotonic input-output relationship
4. **Evaluation Budget**: Secondary coverage depends on available evaluations

### Potential Modifications

1. **Adaptive Bin Sizing**: Variable bin widths based on local properties
2. **Multiple Samples per Bin**: Multiple evaluations within each bin
3. **Confidence Intervals**: Statistical treatment of evaluation noise
4. **Interpolation**: Curve fitting between sampled points
5. **Multi-objective**: Consider multiple secondary outputs simultaneously

## Example Usage

```python
from abcs import BinarySearchSampler

# Define your evaluation function
def evaluate(input_value):
    # Your domain-specific evaluation logic
    primary_output = your_computation(input_value)
    secondary_output = another_computation(input_value)
    metadata = {"secondary_output": secondary_output}
    return primary_output, metadata

# Create sampler with both primary and secondary coverage
sampler = BinarySearchSampler(
    eval_function=evaluate,
    num_bins=20,          # Primary axis bins
    return_bins=15,       # Secondary axis bins
    input_range=(0.0, 100.0),
    output_range=(0.0, 100.0),
    max_additional_evals=30,
    verbose=True
)

# Run the two-phase algorithm
samples = sampler.run_with_return_refinement()

# Get coverage summary
summary = sampler.get_coverage_summary()
print(f"Primary coverage: {summary['coverage_percentage']}%")
print(f"Total evaluations: {summary['total_evaluations']}")

# Access all samples including secondary refinement
all_samples = sampler.get_all_samples_including_refinement()
```

## Summary

ABCS is a two-phase algorithm that efficiently samples monotonic curves by:

1. **Phase 1**: Using binary search to guarantee coverage along the primary output axis
2. **Phase 2**: Optionally refining coverage along secondary output axes

The algorithm leverages monotonicity to minimize evaluations while ensuring comprehensive coverage. It's particularly useful for characterizing trade-offs between competing metrics in machine learning, optimization, and decision-making applications.