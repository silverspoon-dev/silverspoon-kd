"""Distiller factory function for configuration-driven dispatch."""

from typing import Any

from .base_distiller import BaseDistiller
from .blockwise_distiller import BlockwiseDistiller
from .holistic_distiller import HolisticDistiller
from .response_based_distiller import ResponseBasedDistiller

_DISTILLER_REGISTRY: dict[str, type[BaseDistiller]] = {
    "blockwise": BlockwiseDistiller,
    "bkd": BlockwiseDistiller,
    "holistic": HolisticDistiller,
    "hkd": HolisticDistiller,
    "response_based": ResponseBasedDistiller,
    "reskd": ResponseBasedDistiller,
}


def Distiller(distiller_type: str, **kwargs: Any) -> BaseDistiller:
    """Factory that creates a distiller by type name.

    A convenience wrapper that maps a string to the concrete distiller
    class. Useful for configuration-driven workflows where the distiller
    type comes from a config file or CLI argument.

    Distiller types:

    - ``"blockwise"`` / ``"bkd"`` →
      [BlockwiseDistiller][silverspoon_kd.BlockwiseDistiller]
    - ``"holistic"`` / ``"hkd"`` →
      [HolisticDistiller][silverspoon_kd.HolisticDistiller]
    - ``"response_based"`` / ``"reskd"`` →
      [ResponseBasedDistiller][silverspoon_kd.ResponseBasedDistiller]

    Args:
        distiller_type: One of the type names listed above.
        **kwargs: Arguments forwarded to the concrete distiller constructor.
            See the individual class docstrings for accepted parameters.

    Returns:
        An instance of the selected distiller.

    Raises:
        ValueError: If ``distiller_type`` is not recognized.

    Example::

        distiller = Distiller(
            distiller_type="blockwise",
            teacher_model=teacher,
            alignments=alignments,
            args=TrainingArguments(output_dir="./out", backward_per_block=True),
            train_dataset=dataset,
        )
        distiller.train()
    """
    key = distiller_type.lower().replace("-", "_")
    if key not in _DISTILLER_REGISTRY:
        available = sorted(set(_DISTILLER_REGISTRY.values()), key=lambda c: c.__name__)
        names = ", ".join(
            f"'{alias}'" for alias, cls in sorted(_DISTILLER_REGISTRY.items()) if cls in available
        )
        raise ValueError(f"Unknown distiller_type '{distiller_type}'. Available: {names}")
    return _DISTILLER_REGISTRY[key](**kwargs)
