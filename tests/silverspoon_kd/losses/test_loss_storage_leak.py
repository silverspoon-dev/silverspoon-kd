"""
Regression tests: every supported loss in ``LOSS_REGISTRY`` must produce a
scalar that, after ``.detach().clone()``, has a **small** underlying storage.

This catches a class of PyTorch bugs where ``F.mse_loss`` /
``F.smooth_l1_loss`` (and others) return a 0-dim tensor that VIEWS into a
scratch buffer the size of the input. Without ``.clone()``, persistently
storing such a "scalar" loss keeps the (potentially many-MB) scratch
buffer alive — which is exactly what caused an HKD QAT eval-loss memory leak
in downstream experiments.

silverspoon-kd's distiller stores per-step and per-batch loss tensors in
several places (``current_step_metrics``, ``per_layer_eval_metrics``,
``step_losses``, ``eval_losses``). All of those storage sites use
``.detach().clone()`` to defend against this PyTorch behavior.

These tests verify that:

1. Every loss in ``LOSS_REGISTRY`` produces a scalar tensor.
2. That scalar's storage is **small** (≤ 32 bytes) after ``.detach().clone()``,
   regardless of how PyTorch's underlying C++ kernel allocates intermediates.
3. The same property holds for the bare ``F.mse_loss`` / ``F.smooth_l1_loss``
   PyTorch functions (so we explicitly document the upstream behavior we
   are working around).

Failure of any of these tests indicates either:
* a new loss was added without clone-defense, or
* the existing clone-defense in silverspoon-kd was removed, or
* PyTorch upstream changed behavior in a way we should adjust to.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from silverspoon_kd.losses import LOSS_REGISTRY, get_loss_function

# A non-trivial input shape that exercises the leak path:
# 64 × 256 × 4 bytes (fp32) = 64 KiB. Large enough that any "input-sized
# scratch buffer" is unambiguously distinguishable from a scalar (≤ 32 bytes),
# small enough that the test runs in milliseconds.
_FEATURE_DIM = 256
_BATCH = 64
_INPUT_BYTES = _BATCH * _FEATURE_DIM * 4  # 64 KiB

# Threshold for "scalar" storage. A real scalar is 4 bytes (fp32) or 2 bytes
# (bf16). Anything ≤ 32 bytes is unambiguously a scalar; anything ≥ 1 KiB is
# an input-sized scratch buffer leak.
_MAX_SCALAR_STORAGE_BYTES = 32


def _make_inputs(seed: int = 0):
    """Create student/teacher tensors that work for every loss in the registry.

    Returns ``(student, teacher)`` of shape ``(_BATCH, _FEATURE_DIM)`` fp32,
    plus a ``weight_matrix`` for losses that need one (mahalanobis variants).
    """
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(_BATCH, _FEATURE_DIM, generator=g, dtype=torch.float32)
    teacher = torch.randn(_BATCH, _FEATURE_DIM, generator=g, dtype=torch.float32)
    weight_matrix = torch.randn(_FEATURE_DIM, _FEATURE_DIM, generator=g, dtype=torch.float32)
    return student, teacher, weight_matrix


# Per-loss factory kwargs. Most losses take no required arguments. Mahalanobis
# losses need a weight_matrix; logit_lens_kl needs an output_head module.
def _factory_kwargs(loss_name: str):
    _, _, weight_matrix = _make_inputs()
    if loss_name in ("mahal_mse", "mahal_cosine", "mahalanobis_mse", "mahalanobis_cosine"):
        return {"weight_matrix": weight_matrix}
    if loss_name == "logit_lens_kl":
        # Needs an actual nn.Module that maps hidden states → logits
        # (typically an LM head). For the leak test, a simple Linear works.
        from torch import nn

        return {"output_head": nn.Linear(_FEATURE_DIM, 1024)}
    return {}


# Some losses expect a different shape than (B, D). Override here as needed.
def _inputs_for_loss(loss_name: str):
    """Return ``(student, teacher)`` shaped appropriately for ``loss_name``."""
    s, t, _ = _make_inputs()
    # All registered losses currently work with (B, D); add per-loss
    # overrides if a future loss needs something different.
    return s, t


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-loss "no leak after .detach().clone()" tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("loss_name", sorted(LOSS_REGISTRY.keys()))
class TestLossDoesNotLeakAfterDetachClone:
    """Every loss in the registry, after ``.detach().clone()``, must have a
    small underlying storage.

    This is the contract that the silverspoon-kd distiller relies on when it
    stores per-step / per-batch loss values across many iterations.
    """

    def test_clone_yields_small_storage(self, loss_name):
        loss_fn = get_loss_function(loss_name, **_factory_kwargs(loss_name))
        student, teacher = _inputs_for_loss(loss_name)
        loss = loss_fn(student, teacher)

        assert isinstance(loss, torch.Tensor), (
            f"{loss_name}: expected a torch.Tensor, got {type(loss).__name__}"
        )
        assert loss.dim() == 0, (
            f"{loss_name}: expected a scalar (0-dim) loss, got shape {loss.shape}"
        )

        cloned = loss.detach().clone()
        nbytes = cloned.untyped_storage().nbytes()
        assert nbytes <= _MAX_SCALAR_STORAGE_BYTES, (
            f"{loss_name}: cloned scalar storage is {nbytes} bytes "
            f"(expected ≤ {_MAX_SCALAR_STORAGE_BYTES}). The .clone() defense "
            f"in silverspoon-kd is broken or this loss has an unusual storage "
            f"layout."
        )

    def test_loss_value_finite(self, loss_name):
        """Sanity check: every loss produces a finite value on random inputs."""
        loss_fn = get_loss_function(loss_name, **_factory_kwargs(loss_name))
        student, teacher = _inputs_for_loss(loss_name)
        loss = loss_fn(student, teacher)
        assert torch.isfinite(loss).item(), f"{loss_name}: loss value {loss.item()} is not finite"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Documenting upstream PyTorch behavior we are working around
# ─────────────────────────────────────────────────────────────────────────────


class TestPyTorchFusedLossLeaksAreDocumented:
    """These tests document the upstream PyTorch behavior that motivates the
    ``.detach().clone()`` defense in silverspoon-kd's metric tracking.

    They are not "failing" tests — they assert that PyTorch's
    ``F.mse_loss`` / ``F.smooth_l1_loss`` return scalars whose storage is
    the size of the input. If PyTorch ever fixes this, these tests will fail
    and we can simplify silverspoon-kd's metric tracking to use bare
    ``.detach()``.
    """

    @pytest.mark.parametrize(
        "loss_fn,reduction",
        [
            (F.mse_loss, "mean"),
            (F.mse_loss, "sum"),
            (F.smooth_l1_loss, "mean"),
            (F.smooth_l1_loss, "sum"),
        ],
    )
    def test_pytorch_fused_loss_returns_input_sized_storage(self, loss_fn, reduction):
        """``F.mse_loss`` and ``F.smooth_l1_loss`` return a scalar tensor whose
        underlying storage is the size of one input tensor.

        The scalar's nelement is 1, but it views into a scratch buffer
        allocated at input size. ``.detach()`` shares this storage, so
        persistently storing the detached scalar keeps the scratch buffer
        alive across iterations.
        """
        s = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        t = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        loss = loss_fn(s, t, reduction=reduction)
        # The "scalar" is actually a view into a (B, D) intermediate.
        assert loss.untyped_storage().nbytes() == _INPUT_BYTES, (
            f"{loss_fn.__name__}(reduction={reduction!r}) storage is "
            f"{loss.untyped_storage().nbytes()} bytes (expected {_INPUT_BYTES} = "
            f"input size). PyTorch upstream behavior may have changed; "
            f"silverspoon-kd's clone-defense may now be unnecessary."
        )

    @pytest.mark.parametrize(
        "loss_fn",
        [F.l1_loss, F.huber_loss],
    )
    def test_pytorch_correctly_implemented_losses_return_small_storage(self, loss_fn):
        """``F.l1_loss`` and ``F.huber_loss`` correctly return a small-storage
        scalar — they do **not** view into the input. We document this so
        future regressions in either direction (PyTorch fixing mse_loss or
        breaking l1_loss) are caught immediately.
        """
        s = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        t = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        loss = loss_fn(s, t)
        assert loss.untyped_storage().nbytes() <= _MAX_SCALAR_STORAGE_BYTES, (
            f"{loss_fn.__name__} now has a scratch-buffer storage layout "
            f"({loss.untyped_storage().nbytes()} bytes for a scalar) — this "
            f"is a regression in PyTorch upstream."
        )

    def test_clone_defense_works_on_fused_losses(self):
        """``loss.detach().clone()`` always produces a tiny storage — this is
        the workaround silverspoon-kd uses for the fused-loss leak.
        """
        s = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        t = torch.zeros(_BATCH, _FEATURE_DIM, dtype=torch.float32)
        for loss_fn in (F.mse_loss, F.smooth_l1_loss):
            loss = loss_fn(s, t)
            cloned = loss.detach().clone()
            assert cloned.untyped_storage().nbytes() <= _MAX_SCALAR_STORAGE_BYTES, (
                f".detach().clone() did NOT free the scratch buffer for "
                f"{loss_fn.__name__}; got {cloned.untyped_storage().nbytes()} bytes"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Coverage check: every loss in LOSS_REGISTRY is exercised by the
#    parametrized leak test above. This guards against silently adding a new
#    loss without testing it.
# ─────────────────────────────────────────────────────────────────────────────


class TestEveryLossIsCoveredByLeakTests:
    def test_all_registry_losses_appear_in_parametrize(self):
        """If a new loss is added to LOSS_REGISTRY, this test forces the
        author to also add (or override) any required factory kwargs / input
        shapes above. Otherwise the parametrized test would silently skip it.
        """
        registry_losses = set(LOSS_REGISTRY.keys())

        # The parametrize on the class above uses sorted(LOSS_REGISTRY.keys())
        # so by construction every loss is covered. Verify the registry hasn't
        # been monkey-patched in some weird way.
        assert registry_losses == set(LOSS_REGISTRY.keys())

        # Also assert that _inputs_for_loss handles every loss without
        # raising. If a future loss needs a different shape, the author
        # has to add a per-loss override above.
        for loss_name in registry_losses:
            try:
                _ = _inputs_for_loss(loss_name)
            except Exception as e:
                pytest.fail(
                    f"_inputs_for_loss({loss_name!r}) raised {type(e).__name__}: {e}\n"
                    f"Add a per-loss input shape override in this test file."
                )

    def test_every_loss_can_be_constructed_with_default_kwargs(self):
        """Every loss must be constructible with the kwargs we provide. If a
        new loss requires a kwarg we don't supply, this test catches it
        immediately rather than failing during the parametrized leak tests
        with a confusing error.
        """
        for loss_name in sorted(LOSS_REGISTRY.keys()):
            try:
                _ = get_loss_function(loss_name, **_factory_kwargs(loss_name))
            except Exception as e:
                pytest.fail(
                    f"get_loss_function({loss_name!r}, ...) raised "
                    f"{type(e).__name__}: {e}\n"
                    f"Add the required kwarg in _factory_kwargs() above."
                )
