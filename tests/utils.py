from typing import Tuple


def _calculate_return_coverage(
    samples, sampler, return_bins: int = None
) -> Tuple[float, int, int]:
    """
    Calculate return value coverage for a list of samples.

    Args:
        samples: List of sample points
        sampler: BinarySearchSampler instance (for extract_return_value method)
        return_bins: Number of return bins to use (if None, uses sampler.return_bins)

    Returns:
        Tuple of (coverage_percentage, filled_bins_count, total_bins)
    """
    # Extract return values from samples
    returns = []
    for sample in samples:
        if sample is not None:
            try:
                ret = sampler.extract_return_value(sample)
                returns.append(ret)
            except ValueError:
                pass

    # Use provided return_bins or fall back to sampler's setting
    num_bins = return_bins if return_bins is not None else sampler.return_bins

    if not returns or num_bins <= 0:
        return 0.0, 0, num_bins

    # Calculate coverage
    min_return = min(returns)
    max_return = max(returns)

    if max_return <= min_return:
        # All return values are the same - only fills one bin
        return 100.0 / num_bins, 1, num_bins

    filled_return_bins = set()
    for ret in returns:
        bin_idx = int((ret - min_return) / (max_return - min_return) * num_bins)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        filled_return_bins.add(bin_idx)

    coverage_percentage = 100.0 * len(filled_return_bins) / num_bins
    return coverage_percentage, len(filled_return_bins), num_bins
