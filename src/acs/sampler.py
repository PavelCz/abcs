"""
Adaptive Binary Coverage Search (ABCS) algorithm for efficient monotonic curve sampling.

This module provides a generic implementation of the ABCS algorithm used for
efficiently sampling points along monotonic curves with coverage guarantees.
"""

from typing import List, Tuple, Callable, Optional, Any, Dict
import numpy as np
from numpy.typing import NDArray

from acs.types import SamplePoint
from acs.joint_sampler import CurvePoint, SamplingResult


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

    def run(self) -> SamplingResult:
        """
        Run the adaptive sampling algorithm (Phase 1 only).

        Returns a SamplingResult with curve points and coverage information.
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

        # Convert to SamplingResult format
        return self._create_sampling_result()

    def _create_sampling_result(self) -> SamplingResult:
        """Create a SamplingResult from the current state."""
        # Convert SamplePoints to CurvePoints
        curve_points = []
        for i, sample in enumerate(self.all_samples):
            # Map input_value to desired_percentile (normalized to 0-1)
            percentile = (sample.input_value - self.input_range[0]) / (
                self.input_range[1] - self.input_range[0]
            )

            # Extract performance from metadata if available, otherwise use output_value
            performance = sample.metadata.get("performance", sample.output_value)

            curve_point = CurvePoint(
                desired_percentile=percentile,
                afhp=sample.output_value,  # output_value is treated as AFHP
                performance=performance,
                repeats_used=1,  # Single-axis sampler doesn't do repeats
                order=i + 1,  # Order of evaluation
            )
            curve_points.append(curve_point)

        # Calculate coverage gap for the output axis
        filled_samples = [s for s in self.bin_samples if s is not None]
        output_gap = 0.0

        if len(filled_samples) > 1:
            sorted_samples = sorted(filled_samples, key=lambda s: s.output_value)
            output_min = sorted_samples[0].output_value
            output_max = sorted_samples[-1].output_value

            if output_max > output_min:
                # Find maximum normalized gap between consecutive samples
                for i in range(len(sorted_samples) - 1):
                    gap = (
                        sorted_samples[i + 1].output_value
                        - sorted_samples[i].output_value
                    ) / (output_max - output_min)
                    output_gap = max(output_gap, gap)

        # Create info dict with single-axis specific information
        filled_samples = [s for s in self.bin_samples if s is not None]
        info: Dict[str, Any] = {
            "bins_filled": len(filled_samples),
            "total_bins": self.num_bins,
            "coverage_percentage": 100.0 * len(filled_samples) / self.num_bins,
            "bin_edges": self.bin_edges.tolist(),
            "output_range": self.output_range,
            "input_range": self.input_range,
        }

        # Find gaps in bin coverage
        gaps = []
        for i in range(self.num_bins):
            if self.bin_samples[i] is None:
                gaps.append((self.bin_edges[i], self.bin_edges[i + 1]))
        if gaps:
            info["uncovered_bins"] = gaps

        return SamplingResult(
            points=curve_points,
            coverage_x_max_gap=output_gap,  # For single-axis, x-axis represents the output
            coverage_y_max_gap=0.0,  # Not applicable for single-axis sampling
            total_evals=self.total_evals,
            early_stop_reason=None,  # Single-axis doesn't have early stopping
            monotonicity_violations_remaining=False,  # Not applicable for single-axis
            info=info,  # Add the info dict with single-axis specific data
        )
