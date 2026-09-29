"""Distributed teacher placement for knowledge distillation."""

from .split_gpu import setup_split_gpu
from .teacher_placement import TeacherPlacement

__all__ = ["TeacherPlacement", "setup_split_gpu"]
