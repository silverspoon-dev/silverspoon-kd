"""Composite optimizer and scheduler for multi-student distillation."""

from .composite_optimizer import CompositeOptimizer
from .composite_scheduler import CompositeScheduler

__all__ = ["CompositeOptimizer", "CompositeScheduler"]
