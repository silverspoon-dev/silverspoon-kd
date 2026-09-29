"""TeacherPlacement dataclass for configuring teacher model distribution."""

from dataclasses import dataclass, field

_VALID_STRATEGIES = ("pp", "tp", "sharded")
_VALID_STRING_PLACEMENTS = ("replicated", "sharded")


@dataclass
class TeacherPlacement:
    """Configuration for dedicated-GPU teacher placement.

    Specifies which physical GPUs are reserved for the teacher model and
    which distribution strategy to use across those GPUs.

    Args:
        teacher_only_devices: Absolute physical GPU IDs (matching ``nvidia-smi``
            output) dedicated to the teacher. These GPUs will be hidden from
            the student's distributed setup.
        strategy: Distribution strategy for the teacher across its dedicated
            GPUs. One of ``"pp"`` (pipeline parallel), ``"tp"`` (tensor
            parallel), or ``"sharded"`` (FSDP full_shard).
        device_type: Accelerator type. Auto-detected as ``"cuda"`` if not set.
        wrap_cls: Module class name(s) for FSDP wrapping granularity
            (``strategy="sharded"`` only). If ``None``, uses the model's
            ``_no_split_modules`` or falls back to size-based policy.
    """

    teacher_only_devices: list[int] = field(default_factory=list)
    strategy: str = "pp"
    device_type: str | None = None
    wrap_cls: str | list[str] | None = None

    def __post_init__(self):
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy {self.strategy!r}. Must be one of {_VALID_STRATEGIES}."
            )
        if not self.teacher_only_devices:
            raise ValueError(
                "teacher_only_devices is required — specifies which GPUs are "
                "dedicated to the teacher. Use string 'replicated' or 'sharded' "
                "for all-ranks strategies without dedicated devices."
            )
        if self.wrap_cls is not None and self.strategy != "sharded":
            raise ValueError("wrap_cls is only valid with strategy='sharded'.")


def normalize_teacher_placement(
    value: None | str | dict | TeacherPlacement,
) -> str | TeacherPlacement:
    """Normalize a teacher_placement argument to str or TeacherPlacement.

    Accepts:
      - ``None`` → ``"replicated"``
      - ``str`` → validated string (``"replicated"`` or ``"sharded"``)
      - ``dict`` → ``TeacherPlacement(**dict)``
      - ``TeacherPlacement`` → passed through
    """
    if value is None:
        return "replicated"
    if isinstance(value, str):
        if value not in _VALID_STRING_PLACEMENTS:
            raise ValueError(
                f"Invalid teacher_placement string {value!r}. "
                f"Must be one of {_VALID_STRING_PLACEMENTS} or a "
                f"TeacherPlacement instance."
            )
        return value
    if isinstance(value, dict):
        return TeacherPlacement(**value)
    if isinstance(value, TeacherPlacement):
        return value
    raise TypeError(
        f"teacher_placement must be str, dict, or TeacherPlacement, got {type(value).__name__}."
    )
