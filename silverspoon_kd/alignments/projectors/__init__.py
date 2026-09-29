"""Projector modules for shape-matching teacher and student activations."""

from .fusion import fuse_projectors_into_module
from .generic_conv2d_projector import GenericConv2dProjector
from .generic_linear_projector import GenericLinearProjector

__all__ = [
    "GenericConv2dProjector",
    "GenericLinearProjector",
    "fuse_projectors_into_module",
]
