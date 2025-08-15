"""
Type definitions for ABCS library.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SamplePoint:
    """Represents a single evaluation point."""

    input_value: float  # The input parameter (e.g., threshold percentile)
    output_value: float  # The measured output (e.g., AFHP)
    metadata: Dict[str, Any]  # Additional data (e.g., return, std, full results)