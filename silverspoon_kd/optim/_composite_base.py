"""Shared state-management helpers for composite optimizer/scheduler.

Both :class:`CompositeOptimizer` and :class:`CompositeScheduler` wrap a
dict of named children and need identical:

* validation of the children dict,
* an "active subset" filter used by ``step()``,
* versioned state-dict aggregation, and
* a ``__repr__`` that lists the children.

This module factors that logic into :class:`_CompositeBase` so both
wrappers can inherit it without duplicating ~30 lines of plumbing.
"""

import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


class _CompositeBase:
    """Shared child-container / state plumbing for composite wrappers.

    Subclasses must call :meth:`_init_composite` in their ``__init__`` and
    can override :attr:`_CHILD_KIND` to get more specific error messages.
    """

    # Format version written into every state dict, so a state dict from a
    # different layout can be recognised on load.  Version 1 wraps the
    # children in a ``{"_composite_version", "children"}`` dict.
    _STATE_VERSION: int = 1

    # Human-readable label for error messages ("optimizer", "scheduler", ...).
    _CHILD_KIND: str = "child"

    # These are populated by ``_init_composite``; declaring them at class
    # level keeps static analysers happy without requiring an ``__init__``
    # in the mixin (which would fight with the real base classes).
    _children: dict[str, Any]
    _active_names: set[str] | None
    _child_order: list

    def _init_composite(self, children: dict[str, Any]) -> None:
        """Validate ``children`` and record the initial state.

        Args:
            children: Mapping from child name to the wrapped object.

        Raises:
            ValueError: If ``children`` is empty.
        """
        if not children:
            raise ValueError(
                f"{type(self).__name__} requires at least one child {self._CHILD_KIND}."
            )
        self._children = children
        self._active_names = None  # None = all active
        self._child_order = list(children.keys())

    # ------------------------------------------------------------------
    # Active child filtering
    # ------------------------------------------------------------------

    def set_active(self, names: set[str] | None = None) -> None:
        """Set which children are active for ``step()``.

        Args:
            names: Set of child names to activate. ``None`` means all active.

        Raises:
            ValueError: If any name in ``names`` is not a known child.
        """
        if names is not None:
            unknown = names - set(self._child_order)
            if unknown:
                raise ValueError(f"Unknown child {self._CHILD_KIND} names: {unknown}")
        self._active_names = names

    def _iter_active(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, child)`` pairs for the currently active children."""
        for name in self._child_order:
            if self._active_names is None or name in self._active_names:
                yield name, self._children[name]

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Aggregate children's state dicts with a version tag."""
        children_state = {}
        for name in self._child_order:
            children_state[name] = self._children[name].state_dict()
        return {
            "_composite_version": self._STATE_VERSION,
            "children": children_state,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load aggregated state produced by :meth:`state_dict`."""
        if "_composite_version" not in state_dict:
            raise ValueError(
                f"{type(self).__name__}.load_state_dict expects a state dict produced by "
                f"{type(self).__name__}.state_dict (missing '_composite_version')"
            )
        children_state = state_dict["children"]

        for name in self._child_order:
            if name in children_state:
                self._children[name].load_state_dict(children_state[name])
            else:
                logger.warning(
                    "%s: no saved state for child %s %r — it will start from "
                    "scratch (not resumed from checkpoint).",
                    type(self).__name__,
                    self._CHILD_KIND,
                    name,
                )
        extra = set(children_state) - set(self._child_order)
        if extra:
            logger.warning(
                "%s: checkpoint contains state for unknown children %s "
                "which will be discarded. This usually means the model "
                "architecture changed between save and load.",
                type(self).__name__,
                sorted(extra),
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}(active={self._active_names})"]
        for name, child in self._children.items():
            lines.append(f"  {name}: {type(child).__name__}")
        return "\n".join(lines)
