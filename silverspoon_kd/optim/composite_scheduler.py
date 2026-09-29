"""Composite scheduler that wraps multiple per-student schedulers as one."""

from typing import Any

from ._composite_base import _CompositeBase


class CompositeScheduler(_CompositeBase):
    """Wraps N per-student LR schedulers as a single scheduler.

    Does not need to inherit from a base class — the Trainer only calls
    ``.step()``, ``.state_dict()``, and ``.load_state_dict()``.

    Args:
        children: Mapping from student name to its LR scheduler.
    """

    _CHILD_KIND = "scheduler"

    def __init__(self, children: dict[str, Any]):
        self._init_composite(children)

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def step(self, *args, **kwargs):
        """Delegate ``step()`` to active children only."""
        for _name, child in self._iter_active():
            child.step(*args, **kwargs)

    def get_last_lr(self) -> list:
        """Concatenate ``get_last_lr()`` from active children."""
        result = []
        for _name, child in self._iter_active():
            if hasattr(child, "get_last_lr"):
                result.extend(child.get_last_lr())
        return result
