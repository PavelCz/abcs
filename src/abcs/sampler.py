"""
Adaptive Binary Coverage Search (ABCS) algorithm for efficient monotonic curve sampling.

This module provides a generic implementation of the ABCS algorithm used for
efficiently sampling points along monotonic curves with coverage guarantees.
"""

from typing import List, Tuple, Callable, Optional, Any, Dict
import numpy as np
from numpy.typing import NDArray

from abcs.types import SamplePoint


class BinarySearchSampler:
    """
    Adaptive Binary Coverage Search (ABCS) sampler for monotonic curves.

    This sampler efficiently fills bins along the output axis by using
    binary search in the input space. It operates in two phases:
    1. Phase 1: AFHP coverage via recursive binary search
    2. Phase 2: Return value gap filling via recursive binary bisection (optional)

    Phase 2 uses a systematic approach to fill gaps in return value coverage:
    - Identifies contiguous intervals of empty return bins
    - Applies recursive binary search within each gap interval
    - Ensures comprehensive coverage across both primary and secondary dimensions
    """

    def __init__(
        self,
        eval_function: Callable[[float], Tuple[float, Dict[str, Any]]],
        num_bins: int,
        input_range: Tuple[float, float] = (0.0, 100.0),
        output_range: Tuple[float, float] = (0.0, 100.0),
        input_to_threshold: Optional[Callable[[float], float]] = None,
        verbose: bool = True,
        return_bins: int = 0,
        return_value_function: Optional[Callable[[Dict[str, Any]], float]] = None,
    ):
        """
        Initialize the ABCS sampler.

        The sampler always operates in unbounded mode with theoretical convergence
        guarantees. It continues sampling until all bins are filled or no progress
        can be made, with safety mechanisms to prevent infinite loops.

        Args:
            eval_function: Function that takes an input value and returns
                          (output_value, metadata_dict)
            num_bins: Number of bins to divide the output space into
            input_range: Range of valid input values (min, max)
            output_range: Range of expected output values (min, max)
            input_to_threshold: Optional function to convert input values
                               to actual thresholds for evaluation
            verbose: Whether to print progress messages
            return_bins: Number of return bins for curve smoothing (0 = disabled)
            return_value_function: Function to extract return value from sample
                metadata. Only used if return_bins > 0.
        """
        self.eval_function = eval_function
        self.num_bins = num_bins
        self.input_range = input_range
        self.output_range = output_range
        self.input_to_threshold = input_to_threshold or (lambda x: x)
        self.verbose = verbose
        self.return_bins = return_bins
        self.return_value_function = return_value_function
        # If no return_value_function is provided, `extract_return_value` will
        # attempt to read standard fields from metadata and raise if unavailable.

        # Safety limit to prevent infinite loops
        self.max_total_evals = 200

        # Initialize bins
        self.bin_edges: NDArray[np.float64] = np.linspace(
            output_range[0], output_range[1], num_bins + 1
        )
        self.bin_samples: List[Optional[SamplePoint]] = [None] * num_bins
        self.return_refinement_samples: List[SamplePoint] = []
        self.total_evals: int = 0

    def run_with_return_refinement(self) -> List[Optional[SamplePoint]]:
        """
        Run the enhanced sampling algorithm with return gap filling.

        This is the main entry point for the two-phase ABCS algorithm:
        1. Phase 1: Standard AFHP-based binary search
        2. Phase 2: Return gap filling (if return_bins > 0)

        Returns:
            Combined list of samples from both phases
        """
        # Phase 1: Standard AFHP coverage
        if self.verbose:
            print("Phase 1: AFHP coverage using binary search...")

        afhp_samples = self.run()

        # Phase 2: Return gap filling
        if self.return_bins > 0:
            if self.verbose:
                print(
                    f"Phase 2: Return gap filling with {self.return_bins} return "
                    "bins..."
                )

            self.fill_return_gaps(afhp_samples)

            # Return combined samples, but maintain the original bin structure
            # The additional samples are stored separately in return_refinement_samples
            return afhp_samples
        else:
            return afhp_samples

    def run(self) -> List[Optional[SamplePoint]]:
        """
        Run the adaptive sampling algorithm (Phase 1 only).

        Returns a list of samples, one per bin (where possible).
        """
        # Evaluate at extremes
        left_sample = self.evaluate_at_input(self.input_range[0])
        right_sample = self.evaluate_at_input(self.input_range[1])

        # Place extreme samples in appropriate bins
        left_bin = self.determine_bin(left_sample.output_value)
        right_bin = self.determine_bin(right_sample.output_value)

        self.bin_samples[left_bin] = left_sample
        self.bin_samples[right_bin] = right_sample

        # Fill remaining bins using binary search
        if left_bin > right_bin:
            raise ValueError(
                "Monotonicity assumption violated: left endpoint maps to a higher bin than right endpoint"
            )
        if left_bin < right_bin:
            self.binary_search_fill(
                self.input_range[0], self.input_range[1], left_bin, right_bin
            )

        if self.verbose:
            print(f"Total evaluations: {self.total_evals}")
            print(
                f"Bins filled: {sum(1 for s in self.bin_samples if s is not None)}"
                f"/{self.num_bins}"
            )

        return self.bin_samples

    def fill_return_gaps(
        self, initial_samples: List[Optional[SamplePoint]]
    ) -> List[SamplePoint]:
        """
        Fill gaps in return values using recursive binary search (Phase 2).

        This method implements the improved Phase 2 algorithm that uses recursive
        binary bisection to systematically fill gaps in return value coverage.
        Unlike the previous interpolation-based approach, this method:

        1. Identifies contiguous intervals of empty return bins
        2. Finds samples that bracket each gap interval
        3. Applies recursive binary search within each interval
        4. Continues until all gaps are filled or convergence is reached

        Args:
            initial_samples: Samples from the initial AFHP-based binary search

        Returns:
            List of additional samples to fill return gaps
        """
        if self.return_bins == 0:
            raise ValueError("Called fill_return_gaps with return_bins=0")

        # Extract valid samples and their returns
        valid_samples = [s for s in initial_samples if s is not None]
        if len(valid_samples) < 2:
            # Not enough samples to refine
            return []

        # Build list of samples with their return values
        samples_with_returns = []
        for sample in valid_samples:
            try:
                ret = self.extract_return_value(sample)
            except ValueError:
                continue
            samples_with_returns.append((sample, ret))

        if len(samples_with_returns) < 2:
            # Not enough usable return values; nothing to refine
            return []

        # Sort samples by return value
        samples_with_returns.sort(key=lambda x: x[1])
        min_return = samples_with_returns[0][1]
        max_return = samples_with_returns[-1][1]

        if max_return <= min_return:
            # No variation to fill
            return []

        # Create return bins
        return_bin_edges = np.linspace(min_return, max_return, self.return_bins + 1)

        # Find which return bins are already filled
        filled_return_bins = set()
        for sample, ret in samples_with_returns:
            bin_idx = self.determine_return_bin(ret, return_bin_edges)
            filled_return_bins.add(bin_idx)

        additional_samples = []

        # Continue until all return bins are filled
        while len(filled_return_bins) < self.return_bins:
            self._check_safety()
            prev_filled_count = len(filled_return_bins)

            # Identify contiguous gap intervals for current state
            gap_intervals = self.identify_return_gap_intervals(
                filled_return_bins, return_bin_edges
            )

            if self.verbose and gap_intervals:
                mode_text = f"safety limit {self.max_total_evals} evals"
                print(
                    f"Found {len(gap_intervals)} return gap intervals to fill ({mode_text})"
                )
                for start, end in gap_intervals:
                    print(
                        f"  Gap interval: bins {start}-{end} (return values {return_bin_edges[start]:.2f}-{return_bin_edges[end + 1]:.2f})"
                    )

            made_progress = False
            prev_additional_len = len(additional_samples)

            # Fill gaps using recursive binary search
            for gap_start, gap_end in gap_intervals:
                # Find samples that bracket this gap interval
                left_sample = None
                right_sample = None

                # Find the best bracketing samples for this gap
                for i, (sample, ret) in enumerate(samples_with_returns):
                    sample_bin = self.determine_return_bin(ret, return_bin_edges)

                    # Update left bracket if this sample is to the left of the gap
                    if sample_bin < gap_start:
                        left_sample = sample

                    # Set right bracket if this sample is to the right of the gap
                    if sample_bin > gap_end and right_sample is None:
                        right_sample = sample
                        break

                # If we can't bracket the gap, skip it for now
                if left_sample is None or right_sample is None:
                    if self.verbose:
                        print(
                            f"  Cannot bracket gap interval {gap_start}-{gap_end}, skipping for now"
                        )
                    continue

                # Apply recursive binary search to fill this gap interval
                if self.verbose:
                    print(
                        f"  Filling gap interval {gap_start}-{gap_end} using binary search"
                    )

                before_fill_count = len(filled_return_bins)
                
                # Try multiple times if dealing with noise
                max_attempts = 3  # Try up to 3 times to fill difficult gaps
                for attempt in range(max_attempts):
                    if gap_start in filled_return_bins and gap_end in filled_return_bins:
                        # Gap is filled, stop trying
                        break
                        
                    evals = self.binary_search_return_gaps(
                        left_sample,
                        right_sample,
                        gap_start,
                        gap_end,
                        filled_return_bins,
                        return_bin_edges,
                        additional_samples,
                    )
                    
                    # Check if we made progress
                    new_fills = len(filled_return_bins) - before_fill_count
                    if new_fills > 0:
                        made_progress = True
                        # Update samples list with new samples for better bracketing
                        for new_sample in additional_samples[-(new_fills):]:
                            try:
                                new_ret = self.extract_return_value(new_sample)
                                samples_with_returns.append((new_sample, new_ret))
                            except ValueError:
                                continue
                        samples_with_returns.sort(key=lambda x: x[1])
                        
                        # If we partially filled the gap, update the brackets and continue
                        if attempt < max_attempts - 1:
                            # Re-find brackets with updated samples
                            for sample, ret in samples_with_returns:
                                sample_bin = self.determine_return_bin(ret, return_bin_edges)
                                if sample_bin < gap_start and (left_sample is None or ret > self.extract_return_value(left_sample)):
                                    left_sample = sample
                                if sample_bin > gap_end and (right_sample is None or ret < self.extract_return_value(right_sample)):
                                    right_sample = sample
                                    break
                    
                    # If gap is filled or significantly reduced, stop retrying
                    remaining_in_gap = sum(1 for b in range(gap_start, gap_end + 1) if b not in filled_return_bins)
                    if remaining_in_gap == 0:
                        break

                # Append newly created samples to samples_with_returns for better bracketing
                for new_sample in additional_samples[prev_additional_len:]:
                    try:
                        new_ret = self.extract_return_value(new_sample)
                        samples_with_returns.append((new_sample, new_ret))
                    except ValueError:
                        continue
                samples_with_returns.sort(key=lambda x: x[1])
                prev_additional_len = len(additional_samples)

            if not made_progress:
                # Cannot fill remaining bins - function doesn't produce values in those ranges
                if self.verbose:
                    unfilled = self.return_bins - len(filled_return_bins)
                    print(
                        f"  Could not fill {unfilled} return bins - function may not produce values in those ranges"
                    )
                break  # Exit gracefully

        if self.verbose and additional_samples:
            print(f"Added {len(additional_samples)} samples for return gap filling")

        return additional_samples

    def evaluate_at_input(self, input_value: float) -> SamplePoint:
        """Evaluate the function at the given input value."""
        threshold = self.input_to_threshold(input_value)
        output_value, metadata = self.eval_function(threshold)
        self.total_evals += 1

        sample = SamplePoint(
            input_value=input_value, output_value=output_value, metadata=metadata
        )
        return sample

    def determine_bin(self, output_value: float) -> int:
        """Determine which bin an output value falls into."""
        if output_value < self.output_range[0] or output_value > self.output_range[1]:
            raise ValueError(
                f"Output value {output_value} outside range {self.output_range}"
            )
        return self._determine_bin_generic(output_value, self.bin_edges, self.num_bins)

    def binary_search_fill(
        self,
        left_input: float,
        right_input: float,
        left_bin_idx: int,
        right_bin_idx: int,
    ) -> int:
        """
        Recursively fill bins using binary search.

        Returns the number of evaluations performed.
        """
        # Input validation to avoid silent failures
        if not (left_input < right_input):
            raise ValueError(
                f"binary_search_fill called with invalid inputs: left_input={left_input} right_input={right_input}"
            )
        if not (left_bin_idx < right_bin_idx):
            raise ValueError(
                f"binary_search_fill called with invalid bin indices: left_bin_idx={left_bin_idx} right_bin_idx={right_bin_idx}"
            )

        def get_bin(sample: SamplePoint) -> int:
            return self.determine_bin(sample.output_value)

        def has_work_between(a: int, b: int) -> bool:
            return self.bins_remaining(a, b)

        def record_sample(sample: SamplePoint, bin_idx: int) -> None:
            if self.bin_samples[bin_idx] is None:
                self.bin_samples[bin_idx] = sample

        return self._bisect_over_input(
            left_input,
            right_input,
            left_bin_idx,
            right_bin_idx,
            get_bin,
            has_work_between,
            record_sample,
        )

    def extract_return_value(self, sample: SamplePoint) -> float:
        """Extract return value from sample metadata."""
        # If return_value_function is provided, use it
        if self.return_value_function:
            return self.return_value_function(sample.metadata)

        # Otherwise use fallback logic for backward compatibility
        metadata = sample.metadata

        if "summary" in metadata:
            summary = metadata["summary"]
            if isinstance(summary, dict):
                # Try different split names
                for split in ["test", "val", "eval"]:
                    if split in summary and "return_mean" in summary[split]:
                        return summary[split]["return_mean"]

        # Fallback: look for return_mean directly in metadata
        if "return_mean" in metadata:
            return metadata["return_mean"]

        # Final fallback: assume it's stored as 'return'
        if "return" in metadata:
            return metadata["return"]

        raise ValueError(
            f"Could not extract return value from sample metadata: {metadata}"
        )

    def bins_remaining(self, left_idx: int, right_idx: int) -> bool:
        """Check if there are empty bins in the given range."""
        return self._bins_remaining_generic(
            left_idx, right_idx, self.bin_samples, exclusive=True
        )

    def get_filled_samples(self) -> List[SamplePoint]:
        """Return only the non-None samples from bins."""
        return [s for s in self.bin_samples if s is not None]

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the sampling coverage."""
        filled_samples = self.get_filled_samples()

        if not filled_samples:
            return {
                "bins_filled": 0,
                "coverage_percentage": 0.0,
                "output_range_covered": (None, None),
                "gaps": [],
                "total_evaluations": self.total_evals,
            }

        # Find gaps in coverage
        gaps = []
        for i in range(self.num_bins):
            if self.bin_samples[i] is None:
                gaps.append((self.bin_edges[i], self.bin_edges[i + 1]))

        # Get actual output range covered
        output_values = [s.output_value for s in filled_samples]

        return {
            "bins_filled": len(filled_samples),
            "coverage_percentage": 100.0 * len(filled_samples) / self.num_bins,
            "output_range_covered": (min(output_values), max(output_values)),
            "gaps": gaps,
            "total_evaluations": self.total_evals,
        }

    def identify_return_gap_intervals(
        self, filled_return_bins: set, return_bin_edges: NDArray[np.float64]
    ) -> List[Tuple[int, int]]:
        """
        Identify contiguous intervals of empty return bins.

        Args:
            filled_return_bins: Set of bin indices that are already filled
            return_bin_edges: Array of bin edge values

        Returns:
            List of (start_bin, end_bin) tuples representing gap intervals
        """
        gap_intervals = []
        current_gap_start = None

        for i in range(self.return_bins):
            if i not in filled_return_bins:
                # Start of a new gap
                if current_gap_start is None:
                    current_gap_start = i
            else:
                # End of a gap
                if current_gap_start is not None:
                    gap_intervals.append((current_gap_start, i - 1))
                    current_gap_start = None

        # Handle gap that extends to the end
        if current_gap_start is not None:
            gap_intervals.append((current_gap_start, self.return_bins - 1))

        return gap_intervals

    def determine_return_bin(
        self, return_value: float, return_bin_edges: NDArray[np.float64]
    ) -> int:
        """
        Determine which return bin a value falls into.

        Args:
            return_value: The return value to bin
            return_bin_edges: Array of bin edge values

        Returns:
            Bin index (0 to return_bins-1)
        """
        return self._determine_bin_generic(
            return_value, return_bin_edges, self.return_bins
        )

    def return_bins_remaining(
        self, left_bin_idx: int, right_bin_idx: int, filled_return_bins: set
    ) -> bool:
        """
        Check if there are empty return bins in the given range.

        Args:
            left_bin_idx: Left boundary bin index (inclusive)
            right_bin_idx: Right boundary bin index (inclusive)
            filled_return_bins: Set of already filled bin indices

        Returns:
            True if there are empty bins in the range
        """
        return self._bins_remaining_generic(
            left_bin_idx, right_bin_idx, filled_return_bins, exclusive=False
        )

    def binary_search_return_gaps(
        self,
        left_sample: SamplePoint,
        right_sample: SamplePoint,
        left_return_bin: int,
        right_return_bin: int,
        filled_return_bins: set,
        return_bin_edges: NDArray[np.float64],
        additional_samples: List[SamplePoint],
    ) -> int:
        """
        Recursively fill return bins using binary search.

        Args:
            left_sample: Sample with lower return value
            right_sample: Sample with higher return value
            left_return_bin: Left boundary return bin
            right_return_bin: Right boundary return bin
            filled_return_bins: Set of already filled bin indices
            return_bin_edges: Array of return bin edges
            additional_samples: List to append new samples to

        Returns:
            Number of evaluations performed
        """
        # Input validation to avoid silent failures
        if not (left_sample.input_value < right_sample.input_value):
            raise ValueError(
                "binary_search_return_gaps requires left_sample.input_value < right_sample.input_value"
            )
        if not (left_return_bin <= right_return_bin):
            raise ValueError(
                f"binary_search_return_gaps called with invalid return bin indices: {left_return_bin}..{right_return_bin}"
            )
        
        # If gap is a single bin or no gap, try to fill it directly
        if left_return_bin == right_return_bin:
            # Single bin to fill - try multiple points due to noise
            total_evals = 0
            target_bin = left_return_bin
            
            # Try the midpoint first
            middle_input = (left_sample.input_value + right_sample.input_value) / 2
            middle_sample = self.evaluate_at_input(middle_input)
            total_evals += 1
            
            try:
                middle_return = self.extract_return_value(middle_sample)
                middle_bin = self.determine_return_bin(middle_return, return_bin_edges)
                
                if middle_bin == target_bin and target_bin not in filled_return_bins:
                    filled_return_bins.add(target_bin)
                    additional_samples.append(middle_sample)
                    self.return_refinement_samples.append(middle_sample)
                    return total_evals
                    
                # If we didn't hit the target, try points slightly offset from midpoint
                # This helps when noise prevents hitting the exact bin
                if target_bin not in filled_return_bins:
                    offsets = [0.25, 0.75, 0.4, 0.6]  # Try different positions
                    for offset in offsets:
                        if total_evals >= 5:  # Limit attempts
                            break
                        trial_input = left_sample.input_value + (right_sample.input_value - left_sample.input_value) * offset
                        trial_sample = self.evaluate_at_input(trial_input)
                        total_evals += 1
                        
                        try:
                            trial_return = self.extract_return_value(trial_sample)
                            trial_bin = self.determine_return_bin(trial_return, return_bin_edges)
                            
                            if trial_bin == target_bin:
                                filled_return_bins.add(target_bin)
                                additional_samples.append(trial_sample)
                                self.return_refinement_samples.append(trial_sample)
                                return total_evals
                        except ValueError:
                            continue
                            
            except ValueError:
                pass  # Can't extract return value
            
            return total_evals

        def get_bin(sample: SamplePoint) -> int:
            try:
                middle_return = self.extract_return_value(sample)
            except ValueError as exc:
                raise ValueError(
                    "Failed to extract return value during return-gap search"
                ) from exc
            return self.determine_return_bin(middle_return, return_bin_edges)

        def has_work_between(a: int, b: int) -> bool:
            return self.return_bins_remaining(a, b, filled_return_bins)

        def record_sample(sample: SamplePoint, bin_idx: int) -> None:
            if bin_idx not in filled_return_bins:
                filled_return_bins.add(bin_idx)
                additional_samples.append(sample)
                self.return_refinement_samples.append(sample)

        return self._bisect_over_input(
            left_sample.input_value,
            right_sample.input_value,
            left_return_bin,
            right_return_bin,
            get_bin,
            has_work_between,
            record_sample,
        )

    def _bisect_over_input(
        self,
        left_input: float,
        right_input: float,
        left_bin_idx: int,
        right_bin_idx: int,
        get_bin: Callable[[SamplePoint], int],
        has_work_between: Callable[[int, int], bool],
        record_sample: Callable[[SamplePoint, int], None],
    ) -> int:
        """
        Generic recursive bisection over the input domain.

        - Evaluates the midpoint between `left_input` and `right_input`.
        - Maps the resulting sample to a bin via `get_bin`.
        - Records coverage via `record_sample`.
        - Recurse left/right while `has_work_between` indicates uncovered bins.

        Returns the number of evaluations performed.
        """
        # Validate inputs
        if not (left_input < right_input):
            raise ValueError(
                f"_bisect_over_input called with invalid inputs: left_input={left_input} right_input={right_input}"
            )
        if not (left_bin_idx <= right_bin_idx):
            raise ValueError(
                f"_bisect_over_input called with invalid bin indices: {left_bin_idx}..{right_bin_idx}"
            )
        
        # Base case: no bins to fill
        if left_bin_idx == right_bin_idx:
            return 0

        # Safety check
        self._check_safety()

        # Evaluate at midpoint
        middle_input = (left_input + right_input) / 2
        middle_sample = self.evaluate_at_input(middle_input)
        evals = 1

        # Determine bin and record
        middle_bin = get_bin(middle_sample)
        record_sample(middle_sample, middle_bin)
        
        # With noise, the middle_bin might not be between left and right bins
        # But we should still explore both sides as they might contain unfilled bins

        # Recurse if bins remain and there's space to search
        if left_bin_idx < middle_bin and has_work_between(left_bin_idx, middle_bin):
            evals += self._bisect_over_input(
                left_input,
                middle_input,
                left_bin_idx,
                middle_bin,
                get_bin,
                has_work_between,
                record_sample,
            )

        if middle_bin < right_bin_idx and has_work_between(middle_bin, right_bin_idx):
            evals += self._bisect_over_input(
                middle_input,
                right_input,
                middle_bin,
                right_bin_idx,
                get_bin,
                has_work_between,
                record_sample,
            )

        return evals

    def search_for_return_range(
        self,
        existing_samples: List[SamplePoint],
        target_return: float,
        return_min: float,
        return_max: float,
        max_iterations: Optional[int] = None,
    ) -> Optional[SamplePoint]:
        """
        Search for a sample within a specific return value range.

        Uses binary search between existing samples to find a new sample
        whose return value falls within [return_min, return_max].

        Args:
            existing_samples: List of existing samples to use for bracketing
            target_return: Target return value to search for
            return_min: Minimum acceptable return value
            return_max: Maximum acceptable return value
            max_iterations: Maximum binary search iterations (None = use defaults)

        Returns:
            SamplePoint if found, None if not found within max_iterations
        """
        # Set max_iterations - using unbounded approach with safety limit
        if max_iterations is None:
            max_iterations = float("inf")  # Only limited by safety checks

        # Sort existing samples by return value to use for interpolation
        samples_with_returns = []
        for sample in existing_samples:
            try:
                ret = self.extract_return_value(sample)
                samples_with_returns.append((sample, ret))
            except ValueError:
                continue

        if len(samples_with_returns) < 2:
            return None

        # Sort by return value
        samples_with_returns.sort(key=lambda x: x[1])

        # Find bracketing samples for the target return
        lower_sample = None
        upper_sample = None

        for sample, ret in samples_with_returns:
            if ret <= target_return:
                lower_sample = sample
                lower_return = ret
            if ret >= target_return and upper_sample is None:
                upper_sample = sample
                upper_return = ret
                break

        if lower_sample is None or upper_sample is None:
            return None

        # Check if target is outside the bracketing range
        if target_return < lower_return or target_return > upper_return:
            return None

        # Binary search between the bracketing input values
        left_input = lower_sample.input_value
        right_input = upper_sample.input_value

        iteration_count = 0
        while True:
            self._check_safety()
            # Calculate middle input value
            middle_input = (left_input + right_input) / 2

            # Evaluate at middle point
            sample = self.evaluate_at_input(middle_input)

            try:
                sample_return = self.extract_return_value(sample)
            except ValueError:
                # Can't extract return value; shrink interval conservatively
                right_input = middle_input
                iteration_count += 1
                continue

            # Check if this sample is within the desired range
            if return_min <= sample_return <= return_max:
                return sample

            # Update search bounds based on monotonicity assumption
            if sample_return < target_return:
                left_input = middle_input
            else:
                right_input = middle_input

            iteration_count += 1

        return None

    def _determine_bin_generic(
        self, value: float, bin_edges: NDArray[np.float64], num_bins: int
    ) -> int:
        """
        Generic method to determine which bin a value falls into.

        Args:
            value: The value to bin
            bin_edges: Array of bin edge values
            num_bins: Total number of bins

        Returns:
            Bin index (0 to num_bins-1)
        """
        min_val = bin_edges[0]
        max_val = bin_edges[-1]

        # Handle edge cases
        if value <= min_val:
            return 0
        if value >= max_val:
            return num_bins - 1

        # Find the appropriate bin
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                return i

        # If we reached here, the value did not fit any bin interval
        raise ValueError(
            "Value did not fall within any provided bin interval; check bin edges"
        )

    def _bins_remaining_generic(
        self, left_idx: int, right_idx: int, filled_bins, exclusive: bool = True
    ) -> bool:
        """
        Generic method to check if there are empty bins in the given range.

        Args:
            left_idx: Left boundary index
            right_idx: Right boundary index
            filled_bins: Set of filled bin indices (or None-check for Phase 1)
            exclusive: If True, exclude boundaries (Phase 1 style); if False, include (Phase 2 style)

        Returns:
            True if there are empty bins in the range
        """
        start = left_idx + 1 if exclusive else left_idx
        end = right_idx if exclusive else right_idx + 1

        for i in range(start, end):
            if isinstance(filled_bins, set):
                # Phase 2 style: check set membership
                if i not in filled_bins:
                    return True
            else:
                # Phase 1 style: check for None in list
                if i < len(filled_bins) and filled_bins[i] is None:
                    return True
        return False

    def _check_safety(self) -> None:
        """
        Check if binary search should terminate based on convergence criteria.

        Uses unbounded convergence logic with safety mechanisms to prevent
        infinite loops while providing theoretical convergence guarantees.

        Args:
            left_value: Left boundary value
            right_value: Right boundary value
            iteration_count: Current iteration count (unused in unbounded mode)

        Returns:
            True if search should terminate
        """
        # Check safety limit to prevent infinite loops
        if self.total_evals >= self.max_total_evals:
            raise RuntimeError(
                f"Reached safety limit of {self.max_total_evals} total evaluations"
            )

    def get_return_refinement_samples(self) -> List[SamplePoint]:
        """Return only the samples added during return refinement."""
        return self.return_refinement_samples
