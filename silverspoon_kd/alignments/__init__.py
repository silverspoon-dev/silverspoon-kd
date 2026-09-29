"""
Alignment classes and utilities for knowledge distillation.

This module provides the Alignment class and factory functions for knowledge distillation:
- Alignment: A single teacher-module <-> student-module pair
- create_alignments: Primary factory for creating alignments from regex patterns
"""

from .alignment import Alignment
from .output_selector import OutputSelector
from .utils import (
    create_alignments,
    load_student_weights_from_checkpoint,
    load_student_with_projectors_from_checkpoint,
)

__all__ = [
    "Alignment",
    "OutputSelector",
    "create_alignments",
    "load_student_weights_from_checkpoint",
    "load_student_with_projectors_from_checkpoint",
]
