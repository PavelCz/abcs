"""
ACS - Adaptive Coverage Sampling

Python library for sampling monotonic curves with coverage guarantees (joint and legacy options).
"""

from .joint_sampler import (
    JointCoverageSampler,
    CurvePoint,
    SamplingResult,
)

from .sampler import BinarySearchSampler

__version__ = "0.1.0"
__all__ = [
    "JointCoverageSampler",
    "CurvePoint",
    "BinarySearchSampler",
    "SamplingResult",
]
