"""GPU splitting utilities for dedicated teacher/student GPU assignment.

Call [setup_split_gpu][silverspoon_kd.setup_split_gpu] **before**
``import torch`` to reorder ``CUDA_VISIBLE_DEVICES`` so that student
GPUs come first (low CUDA indices) and teacher GPUs follow (high CUDA
indices).  HuggingFace Trainer/DDP/FSDP use ``LOCAL_RANK`` for device
assignment, so they naturally land on the low-index (student) GPUs.
"""

import logging
import os

from .teacher_placement import TeacherPlacement

logger = logging.getLogger(__name__)


def compute_student_gpus(
    teacher_gpus: list[int],
    total_gpus: int | None = None,
) -> list[int]:
    """Return sorted physical GPU IDs *not* in *teacher_gpus*.

    Args:
        teacher_gpus: Absolute physical GPU IDs reserved for the teacher.
        total_gpus: Total number of GPUs on the machine. If ``None``,
            inferred from ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi``.

    Returns:
        Sorted list of student GPU physical IDs.
    """
    if total_gpus is None:
        total_gpus = _detect_total_gpus()
    all_gpus = set(range(total_gpus))
    teacher_set = set(teacher_gpus)
    unknown = teacher_set - all_gpus
    if unknown:
        raise ValueError(
            f"teacher_only_devices {sorted(unknown)} exceed available GPUs (0..{total_gpus - 1})."
        )
    return sorted(all_gpus - teacher_set)


def get_remapped_teacher_devices(
    teacher_gpus: list[int],
    student_gpus: list[int],
) -> list[int]:
    """Return CUDA indices of teacher GPUs **after** reordering.

    After ``setup_split_gpu``, ``CUDA_VISIBLE_DEVICES`` is
    ``student_gpus + teacher_gpus``.  The teacher GPUs therefore start
    at index ``len(student_gpus)``.

    Args:
        teacher_gpus: Original physical teacher GPU IDs.
        student_gpus: Original physical student GPU IDs.

    Returns:
        List of remapped CUDA indices for the teacher GPUs.
    """
    offset = len(student_gpus)
    return list(range(offset, offset + len(teacher_gpus)))


def setup_split_gpu(placement: TeacherPlacement) -> tuple[list[int], list[int]]:
    """Reorder ``CUDA_VISIBLE_DEVICES`` for split-GPU teacher placement.

    **Must be called before** ``import torch`` (or at least before any
    CUDA context is created).

    The environment variable is set so that student GPUs occupy the
    lowest CUDA indices and teacher GPUs follow:

    .. code-block:: text

        teacher_only_devices=[0, 1]  on a 4-GPU system
        → student physical GPUs: [2, 3]
        → CUDA_VISIBLE_DEVICES=2,3,0,1
            cuda:0 → physical 2  (student)
            cuda:1 → physical 3  (student)
            cuda:2 → physical 0  (teacher)
            cuda:3 → physical 1  (teacher)

    Args:
        placement: ``TeacherPlacement`` with ``teacher_only_devices`` set.

    Returns:
        Tuple of ``(student_physical_gpus, remapped_teacher_cuda_indices)``.
    """
    teacher_gpus = list(placement.teacher_only_devices)
    student_gpus = compute_student_gpus(teacher_gpus)

    if not student_gpus:
        raise ValueError("All GPUs assigned to teacher — no GPUs left for the student.")

    # Build the reordered list: students first, teachers after
    reordered = student_gpus + teacher_gpus
    new_cvd = ",".join(str(g) for g in reordered)

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    logger.info(
        "Split-GPU: setting CUDA_VISIBLE_DEVICES=%s (student physical=%s, teacher physical=%s)",
        new_cvd,
        student_gpus,
        teacher_gpus,
    )
    if prev_cvd is not None and prev_cvd != new_cvd:
        logger.warning(
            "Overwriting existing CUDA_VISIBLE_DEVICES=%s. "
            "This change is permanent for this process — subsequent calls "
            "to setup_split_gpu or manual CUDA_VISIBLE_DEVICES changes may "
            "not take effect if a CUDA context has already been initialized.",
            prev_cvd,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = new_cvd

    remapped_teacher = get_remapped_teacher_devices(teacher_gpus, student_gpus)
    return student_gpus, remapped_teacher


def _detect_total_gpus() -> int:
    """Detect total GPU count from env or nvidia-smi."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return len(cvd.split(","))
    # Fallback: try nvidia-smi.
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot detect GPU count. Set CUDA_VISIBLE_DEVICES or ensure nvidia-smi is available."
        ) from exc
