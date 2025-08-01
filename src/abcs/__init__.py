"""
ABCS - Adaptive Binary Coverage Search

A Python library for efficient sampling of monotonic curves using adaptive 
binary search with coverage guarantees.
"""

from .sampler import BinarySearchSampler
from .types import SamplePoint

__version__ = "0.1.0"
__all__ = ["BinarySearchSampler", "SamplePoint"]