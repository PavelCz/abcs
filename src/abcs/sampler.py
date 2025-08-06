"""
Adaptive Binary Coverage Search (ABCS) algorithm for efficient monotonic curve sampling.

This module provides a generic implementation of the ABCS algorithm used for
efficiently sampling points along monotonic curves with coverage guarantees.
"""

from typing import List, Tuple, Callable, Optional, Any, Dict
import numpy as np
from numpy.typing import NDArray

from .types import SamplePoint


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
        max_additional_evals: int = 20,
        unbounded_mode: bool = False,
    ):
        """
        Initialize the ABCS sampler.

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
            max_additional_evals: Maximum additional evaluations for return refinement
                                 (ignored when unbounded_mode=True)
            unbounded_mode: If True, removes evaluation limits and continues until
                           all bins are filled or no progress can be made. Provides
                           theoretical convergence guarantees for monotonic functions.
        """
        self.eval_function = eval_function
        self.num_bins = num_bins
        self.input_range = input_range
        self.output_range = output_range
        self.input_to_threshold = input_to_threshold or (lambda x: x)
        self.verbose = verbose
        self.return_bins = return_bins
        self.max_additional_evals = max_additional_evals
        self.unbounded_mode = unbounded_mode

        # Safety limit for unbounded mode (prevent infinite loops)
        self.max_total_evals_unbounded = 10000

        # Initialize bins
        self.bin_edges: NDArray[np.float64] = np.linspace(
            output_range[0], output_range[1], num_bins + 1
        )
        self.bin_samples: List[Optional[SamplePoint]] = [None] * num_bins
        self.all_samples: List[SamplePoint] = []
        self.return_refinement_samples: List[SamplePoint] = []
        self.total_evals: int = 0

    def determine_bin(self, output_value: float) -> int:
        """Determine which bin an output value falls into."""
        if output_value < self.output_range[0] or output_value > self.output_range[1]:
            raise ValueError(
                f"Output value {output_value} outside range {self.output_range}"
            )

        # Handle edge case where output equals max value
        if output_value == self.output_range[1]:
            return self.num_bins - 1

        # Find the bin
        for i in range(len(self.bin_edges) - 1):
            if self.bin_edges[i] <= output_value < self.bin_edges[i + 1]:
                return i

        # This should not happen given the checks above
        raise ValueError(f"Could not find bin for output value {output_value}")

    def bins_remaining(self, left_idx: int, right_idx: int) -> bool:
        """Check if there are empty bins in the given range."""
        for i in range(left_idx + 1, right_idx):
            if self.bin_samples[i] is None:
                return True
        return False

    def evaluate_at_input(self, input_value: float) -> SamplePoint:
        """Evaluate the function at the given input value."""
        threshold = self.input_to_threshold(input_value)
        output_value, metadata = self.eval_function(threshold)
        self.total_evals += 1

        sample = SamplePoint(
            input_value=input_value, output_value=output_value, metadata=metadata
        )
        self.all_samples.append(sample)
        return sample

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
        if left_bin < right_bin:
            self.binary_search_fill(
                self.input_range[0], self.input_range[1], left_bin, right_bin
            )

        if self.verbose:
            print(f"Total evaluations: {self.total_evals}")
            print(
                f"Bins filled: {sum(1 for s in self.bin_samples if s is not None)}/{self.num_bins}"
            )

        return self.bin_samples

    def get_filled_samples(self) -> List[SamplePoint]:
        """Return only the non-None samples from bins."""
        return [s for s in self.bin_samples if s is not None]

    def get_all_samples(self) -> List[SamplePoint]:
        """Return all samples in evaluation order."""
        return self.all_samples

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

    def extract_return_value(self, sample: SamplePoint) -> float:
        """Extract return value from sample metadata."""
        # Try different possible keys for return value
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
        min_return = return_bin_edges[0]
        max_return = return_bin_edges[-1]

        # Handle edge cases
        if return_value <= min_return:
            return 0
        if return_value >= max_return:
            return self.return_bins - 1

        # Find the appropriate bin
        for i in range(len(return_bin_edges) - 1):
            if return_bin_edges[i] <= return_value < return_bin_edges[i + 1]:
                return i

        # This should not happen, but default to last bin
        return self.return_bins - 1

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
        for i in range(left_bin_idx, right_bin_idx + 1):
            if i not in filled_return_bins:
                return True
        return False

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
        # Safety checks
        if self.unbounded_mode and self.total_evals >= self.max_total_evals_unbounded:
            if self.verbose:
                print(
                    f"Reached safety limit of {self.max_total_evals_unbounded} total evaluations"
                )
            return 0

        if not self.unbounded_mode and iteration_count >= self.max_additional_evals:
            return 0

        # Check precision threshold
        precision_threshold = 1e-8 if self.unbounded_mode else 1e-6
        if (
            abs(right_sample.input_value - left_sample.input_value)
            < precision_threshold
        ):
            return 0

        # Calculate middle input value
        middle_input = (left_sample.input_value + right_sample.input_value) / 2

        # Evaluate at middle point
        middle_sample = self.evaluate_at_input(middle_input)
        evals = 1

        try:
            middle_return = self.extract_return_value(middle_sample)
        except ValueError:
            # Can't extract return value, skip this sample
            return evals

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
            return []

        # Extract valid samples and their returns
        valid_samples = [s for s in initial_samples if s is not None]
        if len(valid_samples) < 2:
            return []

        # Build list of samples with their return values
        samples_with_returns = []
        for sample in valid_samples:
            try:
                ret = self.extract_return_value(sample)
                samples_with_returns.append((sample, ret))
            except ValueError:
                continue

        if len(samples_with_returns) < 2:
            if self.verbose:
                print("Warning: Could not extract enough return values for gap filling")
            return []

        # Sort samples by return value
        samples_with_returns.sort(key=lambda x: x[1])
        min_return = samples_with_returns[0][1]
        max_return = samples_with_returns[-1][1]

        if max_return <= min_return:
            if self.verbose:
                print("Warning: All return values are equal, cannot fill gaps")
            return []

        # Create return bins
        return_bin_edges = np.linspace(min_return, max_return, self.return_bins + 1)

        # Find which return bins are already filled
        filled_return_bins = set()
        for sample, ret in samples_with_returns:
            bin_idx = self.determine_return_bin(ret, return_bin_edges)
            filled_return_bins.add(bin_idx)

        # Identify contiguous gap intervals
        gap_intervals = self.identify_return_gap_intervals(
            filled_return_bins, return_bin_edges
        )

        if self.verbose and gap_intervals:
            mode_text = (
                "unbounded mode"
                if self.unbounded_mode
                else f"max {self.max_additional_evals} evals"
            )
            print(
                f"Found {len(gap_intervals)} return gap intervals to fill ({mode_text})"
            )
            for start, end in gap_intervals:
                print(
                    f"  Gap interval: bins {start}-{end} (return values {return_bin_edges[start]:.2f}-{return_bin_edges[end + 1]:.2f})"
                )

        # Fill gaps using recursive binary search
        additional_samples = []
        total_evals = 0

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

            # If we can't bracket the gap, skip it
            if left_sample is None or right_sample is None:
                if self.verbose:
                    print(
                        f"  Cannot bracket gap interval {gap_start}-{gap_end}, skipping"
                    )
                continue

            # Apply recursive binary search to fill this gap interval
            if self.verbose:
                print(
                    f"  Filling gap interval {gap_start}-{gap_end} using binary search"
                )

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

            # Update samples list with new samples for better bracketing in next intervals
            for new_sample in additional_samples[len(samples_with_returns) :]:
                try:
                    new_ret = self.extract_return_value(new_sample)
                    samples_with_returns.append((new_sample, new_ret))
                except ValueError:
                    continue
            samples_with_returns.sort(key=lambda x: x[1])

        if self.verbose and additional_samples:
            print(f"Added {len(additional_samples)} samples for return gap filling")

        return additional_samples

    def search_for_return_range(
        self,
        existing_samples: List[SamplePoint],
        target_return: float,
        return_min: float,
        return_max: float,
        max_iterations: Optional[int] = None,
    ) -> Optional[SamplePoint]:
        """
        Binary search for an input value that produces a return in the specified range.

        Args:
            existing_samples: Existing samples to guide the search
            target_return: Target return value (midpoint of range)
            return_min: Minimum acceptable return value
            return_max: Maximum acceptable return value
            max_iterations: Maximum binary search iterations (None = use defaults)

        Returns:
            SamplePoint if found, None if not found within max_iterations
        """
        # Set max_iterations based on mode
        if max_iterations is None:
            if self.unbounded_mode:
                max_iterations = float(
                    "inf"
                )  # Truly unbounded - only safety checks limit
            else:
                max_iterations = 10  # Default for bounded mode

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

        samples_with_returns.sort(key=lambda x: x[1])  # Sort by return value

        # Find the two samples that bracket the target return
        lower_sample, lower_return = samples_with_returns[0]
        upper_sample, upper_return = samples_with_returns[-1]

        # Find the best bracketing samples
        for i in range(len(samples_with_returns) - 1):
            sample1, return1 = samples_with_returns[i]
            sample2, return2 = samples_with_returns[i + 1]

            if return1 <= target_return <= return2:
                lower_sample, lower_return = sample1, return1
                upper_sample, upper_return = sample2, return2
                break

        # If target is outside the range of existing samples, we can't find it
        if target_return < lower_return or target_return > upper_return:
            return None

        # Binary search between the bracketing input values
        left_input = lower_sample.input_value
        right_input = upper_sample.input_value

        iteration_count = 0
        while True:
            # Check iteration limit for bounded mode
            if not self.unbounded_mode and iteration_count >= max_iterations:
                break

            # Safety check for unbounded mode
            if (
                self.unbounded_mode
                and self.total_evals >= self.max_total_evals_unbounded
            ):
                if self.verbose:
                    print(
                        f"Reached safety limit of {self.max_total_evals_unbounded} total evaluations"
                    )
                break

            # Calculate middle input value
            middle_input = (left_input + right_input) / 2

            # Evaluate at middle point
            sample = self.evaluate_at_input(middle_input)

            try:
                sample_return = self.extract_return_value(sample)
            except ValueError:
                # If we can't extract return, skip this sample
                return None

            # Check if we found a return in the target range
            if return_min <= sample_return <= return_max:
                return sample

            # Update search bounds based on monotonicity assumption
            if sample_return < target_return:
                left_input = middle_input
            else:
                right_input = middle_input

            # Stop if search space is too small (convergence based on precision)
            precision_threshold = 1e-8 if self.unbounded_mode else 1e-6
            if abs(right_input - left_input) < precision_threshold:
                break

            iteration_count += 1

        return None

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
                    f"Phase 2: Return gap filling with {self.return_bins} return bins..."
                )

            self.fill_return_gaps(afhp_samples)

            # Return combined samples, but maintain the original bin structure
            # The additional samples are stored separately in return_refinement_samples
            return afhp_samples
        else:
            return afhp_samples

    def get_all_samples_including_refinement(self) -> List[SamplePoint]:
        """Return all samples including those from return refinement."""
        return self.all_samples + self.return_refinement_samples

    def get_return_refinement_samples(self) -> List[SamplePoint]:
        """Return only the samples added during return refinement."""
        return self.return_refinement_samples
