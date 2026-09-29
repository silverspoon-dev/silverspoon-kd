"""Tests for the top-level public API surface.

Locks in the curated ``__all__`` so that:
1. Every name listed in ``__all__`` actually resolves on the package
   (catches typos and accidental removals).
2. Names outside ``__all__`` that form the secondary surface (e.g.
   ``ModuleCaptureEngine``, ``BaseDistiller``) resolve as top-level
   attributes for advanced use.
"""

import silverspoon_kd

# Names that must appear in ``__all__`` — the canonical user-facing surface.
PRIMARY_API = {
    "Alignment",
    "BlockwiseDistiller",
    "Distiller",
    "HolisticDistiller",
    "ResponseBasedDistiller",
    "TeacherPlacement",
    "TrainingArguments",
    "create_alignments",
    "freeze_parameters",
    "get_loss_function",
    "jsd_loss",
    "kl_divergence_loss",
    "load_student_weights_from_checkpoint",
    "load_student_with_projectors_from_checkpoint",
    "prune_model",
    "reconfig_model",
    "setup_split_gpu",
}

# Names intentionally hidden from the primary surface but kept importable
# from the top-level for power-user or advanced use.
SECONDARY_IMPORTABLE = {
    "BaseDistiller",
    "ContrastiveDistillationLoss",
    "GenericConv2dProjector",
    "GenericLinearProjector",
    "ModuleCaptureEngine",
    "OutputSelector",
    "SilentProgressCallback",
    "fuse_projectors_into_module",
    "partial_summarize_layer_names",
    "summarize_layer_names",
}


class TestPublicAPI:
    def test_all_matches_primary_api(self):
        """``__all__`` is exactly the curated primary surface."""
        assert set(silverspoon_kd.__all__) == PRIMARY_API

    def test_primary_api_is_resolvable(self):
        """Every name in __all__ resolves on the package."""
        unresolved = [n for n in silverspoon_kd.__all__ if not hasattr(silverspoon_kd, n)]
        assert not unresolved, f"__all__ lists unresolvable names: {unresolved}"

    def test_secondary_names_remain_importable(self):
        """Secondary names resolve at the top level without being in ``__all__``."""
        missing = [n for n in SECONDARY_IMPORTABLE if not hasattr(silverspoon_kd, n)]
        assert not missing, (
            f"Names {missing} do not resolve at the top level. If that is "
            f"intentional, update SECONDARY_IMPORTABLE in this test."
        )

    def test_no_secondary_in_primary(self):
        """A name cannot be both primary and secondary."""
        overlap = PRIMARY_API & SECONDARY_IMPORTABLE
        assert not overlap, f"Name in both PRIMARY_API and SECONDARY_IMPORTABLE: {overlap}"
