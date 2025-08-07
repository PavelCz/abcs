from typing import Tuple
import abc
import numpy as np

class SamplePoint:
    def __init__(self, desired_percentile: float, observed_percentile: float, performance: float):
        self.desired_percentile = desired_percentile
        self.observed_percentile = observed_percentile
        self.performance = performance


class eval_object(abc.ABC):
    @abc.abstractmethod
    def eval_for_percentile(self, percentile: float) -> SamplePoint:
        """Takes the _desired_ percentile and returns the _actual_ percentile and the
        performance value at that percentile."""
        pass

    @abc.abstractmethod
    def eval_lower_extreme(self) -> SamplePoint:
        """Returns the performance value at the lower extreme of the curve."""
        pass

    @abc.abstractmethod
    def eval_upper_extreme(self) -> SamplePoint:
        """Returns the performance value at the upper extreme of the curve."""
        pass


def perform_abcs(
    eval_object: eval_object,
    num_bins: int,
):
    """Performs ABCS on the given eval_object. Bins both the percentile and the
    performance into num_bins bins. Stops when all bins are filled.

    Args:
        eval_object: The eval_object to perform ABCS on.
        num_bins: The number of bins to use for ABCS.
    """
    # First, we need to find the lower and upper extremes of the curve.
    lower_extreme = eval_object.eval_lower_extreme()
    upper_extreme = eval_object.eval_upper_extreme()

    # Initialize the bins
    percentile_bin_edges = np.linspace(0, 100, num_bins + 1)
    performance_bin_edges = np.linspace(
        lower_extreme.performance, upper_extreme.performance, num_bins + 1
    )

    percentile_sample_bins = [None] * num_bins
    performance_sample_bins = [None] * num_bins



    # Fill the bins
    for percentile in percentile_bin_edges:
        sample = eval_object.eval_for_percentile(percentile)
        print(sample)