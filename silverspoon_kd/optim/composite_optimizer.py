"""Composite optimizer that wraps multiple per-student optimizers as one."""

from collections.abc import Callable

import torch

from ._composite_base import _CompositeBase


class CompositeOptimizer(_CompositeBase, torch.optim.Optimizer):
    """Wraps N per-student optimizers as a single ``torch.optim.Optimizer``.

    Inheriting from ``Optimizer`` ensures ``isinstance`` checks pass and
    ``accelerator.prepare()`` wraps it correctly via ``AcceleratedOptimizer``.

    Each child optimizer's ``param_groups`` are tagged with metadata so they
    can be disaggregated for ``state_dict`` / ``load_state_dict``.

    Args:
        children: Mapping from student name to its optimizer.
    """

    _CHILD_KIND = "optimizer"

    # Skip ``Optimizer.__init__`` — it would re-process ``param_groups``
    # (already validated by the children) and overwrite ``self.state``.
    def __init__(self, children: dict[str, torch.optim.Optimizer]):
        self._init_composite(children)

        # Build a flat list of param_groups from all children.
        # Tag each group so we can disaggregate later.
        all_groups = []
        for name in self._child_order:
            child = children[name]
            for group_idx, group in enumerate(child.param_groups):
                group["_composite_child"] = name
                group["_composite_group_idx"] = group_idx
                all_groups.append(group)

        # Optimizer.__init__ expects defaults dict; children own their defaults.
        self.defaults = {}
        self.state: dict = {}  # pyright: ignore[reportIncompatibleVariableOverride]
        self.param_groups: list = all_groups

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Delegate ``step()`` to active children only."""
        loss = None
        for _name, child in self._iter_active():
            child_loss = child.step(closure)
            if child_loss is not None:
                loss = child_loss if loss is None else loss + child_loss
        return loss

    def zero_grad(self, set_to_none=True):
        """Delegate ``zero_grad()`` to **all** children (not just active)."""
        for child in self._children.values():
            child.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Properties expected by AcceleratedOptimizer / Trainer
    # ------------------------------------------------------------------

    @property
    def is_overflow(self):
        """Return False — no overflow detection in composite optimizer."""
        return False
