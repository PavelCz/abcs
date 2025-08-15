"""
Adaptive Binary Coverage Search (ABCS) algorithm for efficient monotonic curve sampling.

This module provides a generic implementation of the ABCS algorithm used for
efficiently sampling points along monotonic curves with coverage guarantees.
"""

from typing import List, Tuple, Callable, Optional, Any, Dict
import numpy as np
from numpy.typing import NDArray

from acs.types import SamplePoint


class BinarySearchSampler:
    """
    Adaptive Binary Coverage Search (ABCS) sampler for monotonic curves.

    This sampler efficiently fills bins along the output axis by using
    binary search in the input space to achieve AFHP coverage.
    """

    def __init__(
        self,
        *,
        eval_at_percentile: Callable[[float], Tuple[float, Dict[str, Any]]],
        eval_at_lower_extreme: Callable[[], Tuple[float, Dict[str, Any]]],
        eval_at_upper_extreme: Callable[[], Tuple[float, Dict[str, Any]]],
        num_bins: int,
        input_range: Tuple[float, float] = (0.0, 1.0),
        output_range: Tuple[float, float] = (0.0, 100.0),
        verbose: bool = True,
    ):
        """
        Initialize the ABCS sampler.

        Args:
            eval_at_percentile: Function that takes a percentile (0-1) and returns
                               (output_value, metadata_dict)
            eval_at_lower_extreme: Function that evaluates at the lower extreme and returns
                                  (output_value, metadata_dict)
            eval_at_upper_extreme: Function that evaluates at the upper extreme and returns
                                  (output_value, metadata_dict)
            num_bins: Number of bins to divide the output space into
            input_range: Range of valid input percentile values (min, max), default (0.0, 1.0)
            output_range: Range of expected output values (min, max)
            verbose: Whether to print progress messages
        """
        self.eval_at_percentile = eval_at_percentile
        self.eval_at_lower_extreme = eval_at_lower_extreme
        self.eval_at_upper_extreme = eval_at_upper_extreme
        self.num_bins = num_bins
        self.input_range = input_range
        self.output_range = output_range
        self.verbose = verbose

        # Initialize bins
        self.bin_edges: NDArray[np.float64] = np.linspace(
            output_range[0], output_range[1], num_bins + 1
        )
        self.bin_samples: List[Optional[SamplePoint]] = [None] * num_bins
        self.all_samples: List[SamplePoint] = []
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
        """Evaluate the function at the given input value (percentile)."""
        # Handle extremes specially
        if abs(input_value - self.input_range[0]) < 1e-9:
            output_value, metadata = self.eval_at_lower_extreme()
        elif abs(input_value - self.input_range[1]) < 1e-9:
            output_value, metadata = self.eval_at_upper_extreme()
        else:
            # Convert input range to 0-1 percentile for eval_at_percentile
            percentile = (input_value - self.input_range[0]) / (
                self.input_range[1] - self.input_range[0]
            )
            output_value, metadata = self.eval_at_percentile(percentile)

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

    def run_with_return_refinement(self) -> List[Optional[SamplePoint]]:
        """
        Run the sampling algorithm.

        Returns:
            List of samples from the binary search
        """
        return self.run()
