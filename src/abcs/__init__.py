"""
ABCS - Adaptive Binary Coverage Search

A Python library for efficient sampling of monotonic curves using adaptive
binary search with coverage guarantees.
"""

from .joint_sampler import JointCoverageSampler, CurvePoint, SamplingResult

__version__ = "0.1.0"
__all__ = [
    "JointCoverageSampler",
    "CurvePoint",
    "SamplingResult",
]
