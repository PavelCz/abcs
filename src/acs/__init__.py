"""
ACS - Adaptive Coverage Sampling

Python library for sampling monotonic curves with coverage guarantees (joint and legacy options).
"""

from .types import CurvePoint, SamplingResult
from .joint_sampler import (
    JointCoverageSampler,
)

from .sampler import BinarySearchSampler

__version__ = "0.1.0"
__all__ = [
    "JointCoverageSampler",
    "CurvePoint",
    "BinarySearchSampler",
    "SamplingResult",
]
