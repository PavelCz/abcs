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
        if self.return_value_function is None and self.return_bins > 0:
            raise ValueError(
                "return_value_function must be provided if return_bins > 0"
            )

        # Safety limit to prevent infinite loops
        self.max_total_evals = 10000

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
            raise ValueError(
                "Return refinement requires at least two initial samples; received fewer than two"
            )

        # Build list of samples with their return values
        samples_with_returns = []
        for sample in valid_samples:
            ret = self.return_value_function(sample.metadata)

            samples_with_returns.append((sample, ret))

        if len(samples_with_returns) < 2:
            raise ValueError(
                "Could not extract enough return values for gap filling; need at least two"
            )

        # Sort samples by return value
        samples_with_returns.sort(key=lambda x: x[1])
        min_return = samples_with_returns[0][1]
        max_return = samples_with_returns[-1][1]

        if max_return <= min_return:
            raise ValueError(
                "All return values are equal or non-increasing; cannot perform return gap filling"
            )

        # Create return bins
        return_bin_edges = np.linspace(min_return, max_return, self.return_bins + 1)

        # Find which return bins are already filled
        filled_return_bins = set()
        for sample, ret in samples_with_returns:
            bin_idx = self.determine_return_bin(ret, return_bin_edges)
            filled_return_bins.add(bin_idx)

        additional_samples = []
        total_evals = 0

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
                evals = self.binary_search_return_gaps(
                    left_sample,
                    right_sample,
                    gap_start,
                    gap_end,
                    filled_return_bins,
                    return_bin_edges,
                    additional_samples,
                    total_evals,
                )
                total_evals += evals

                if len(filled_return_bins) > before_fill_count:
                    made_progress = True

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
                raise RuntimeError(
                    "Return gap filling made no progress; cannot fill remaining return bins"
                )

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
        # Calculate middle input value
        middle_input = (left_input + right_input) / 2

        # Evaluate at middle point
        sample = self.evaluate_at_input(middle_input)

        # Determine which bin this sample falls into
        bin_idx = self.determine_bin(sample.output_value)

        # Only add to bin if it's empty
        if self.bin_samples[bin_idx] is None:
            self.bin_samples[bin_idx] = sample

        # Recursively search left and right if bins remain
        evals = 1

        if self.bins_remaining(left_bin_idx, bin_idx):
            evals += self.binary_search_fill(
                left_input, middle_input, left_bin_idx, bin_idx
            )

        if self.bins_remaining(bin_idx, right_bin_idx):
            evals += self.binary_search_fill(
                middle_input, right_input, bin_idx, right_bin_idx
            )

        return evals

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
        iteration_count: int = 0,
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
            iteration_count: Current iteration count for safety checks

        Returns:
            Number of evaluations performed
        """
        # Check convergence criteria using the generic method
        self._check_safety()

        # Calculate middle input value
        middle_input = (left_sample.input_value + right_sample.input_value) / 2

        # Evaluate at middle point
        middle_sample = self.evaluate_at_input(middle_input)
        evals = 1

        try:
            middle_return = self.extract_return_value(middle_sample)
        except ValueError as exc:
            raise ValueError(
                "Failed to extract return value during return-gap search"
            ) from exc

        # Determine which bin this return falls into
        middle_bin = self.determine_return_bin(middle_return, return_bin_edges)

        # Add sample to the filled bins set
        if middle_bin not in filled_return_bins:
            filled_return_bins.add(middle_bin)
            additional_samples.append(middle_sample)
            self.return_refinement_samples.append(middle_sample)

        # Recursively search left and right if bins remain
        if self.return_bins_remaining(left_return_bin, middle_bin, filled_return_bins):
            evals += self.binary_search_return_gaps(
                left_sample,
                middle_sample,
                left_return_bin,
                middle_bin,
                filled_return_bins,
                return_bin_edges,
                additional_samples,
                iteration_count + evals,
            )

        if self.return_bins_remaining(middle_bin, right_return_bin, filled_return_bins):
            evals += self.binary_search_return_gaps(
                middle_sample,
                right_sample,
                middle_bin,
                right_return_bin,
                filled_return_bins,
                return_bin_edges,
                additional_samples,
                iteration_count + evals,
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
