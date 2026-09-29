"""Speed benchmarks for all loss functions in the loss registry.

Measures forward pass time for each loss function across multiple tensor sizes.
"""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.losses.contrastive import ContrastiveDistillationLoss
from silverspoon_kd.losses.registry import get_loss_function

TENSOR_SIZES = [
    pytest.param((2, 16, 64), id="small-2x16x64"),
    pytest.param((4, 64, 256), id="medium-4x64x256"),
    pytest.param((8, 128, 512), id="large-8x128x512"),
]


def _make_tensors(shape, device):
    """Create student and teacher tensors for loss computation."""
    student = torch.randn(*shape, device=device, requires_grad=True)
    teacher = torch.randn(*shape, device=device)
    return student, teacher


@pytest.mark.benchmark(group="losses")
class TestMSELoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("mse")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestCosineLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("cosine")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestSmoothL1Loss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("smooth_l1")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestKLDivergenceLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("kl_div")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestContrastiveLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = ContrastiveDistillationLoss(temperature=0.07)
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestNormalizedMSELoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("normalized_mse")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestAngularMagnitudeLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("angular_magnitude")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestLogitLensKLLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        lm_head = nn.Linear(shape[-1], 1000).to(device)
        loss_fn = get_loss_function("logit_lens_kl", output_head=lm_head)
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestMahalMSELoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        W = torch.randn(1000, shape[-1], device=device)
        loss_fn = get_loss_function("mahal_mse", weight_matrix=W)
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestMahalCosineLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        W = torch.randn(1000, shape[-1], device=device)
        loss_fn = get_loss_function("mahal_cosine", weight_matrix=W)
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestRelKDDistanceLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("relkd_distance")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestRelKDAngleLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("relkd_angle")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestRelKDDALoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("relkd_da")
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses")
class TestJSDLoss:
    @pytest.mark.parametrize("shape", TENSOR_SIZES)
    def test_forward(self, benchmark, device, shape):
        loss_fn = get_loss_function("jsd", temperature=4.0, beta=0.5)
        student, teacher = _make_tensors(shape, device)
        benchmark(loss_fn, student, teacher)


@pytest.mark.benchmark(group="losses-backward")
class TestLossBackward:
    """Measure forward + backward pass cost for each loss function."""

    @pytest.mark.parametrize(
        "loss_type",
        [
            "mse",
            "cosine",
            "smooth_l1",
            "kl_div",
            "jsd",
            "normalized_mse",
            "angular_magnitude",
            "relkd_distance",
            "relkd_angle",
            "relkd_da",
        ],
    )
    def test_forward_backward(self, benchmark, device, loss_type):
        loss_fn = get_loss_function(loss_type)
        shape = (4, 64, 256)

        def run():
            student = torch.randn(*shape, device=device, requires_grad=True)
            teacher = torch.randn(*shape, device=device)
            loss = loss_fn(student, teacher)
            loss.backward()

        benchmark(run)

    def test_contrastive_forward_backward(self, benchmark, device):
        loss_fn = ContrastiveDistillationLoss(temperature=0.07)
        shape = (4, 64, 256)

        def run():
            student = torch.randn(*shape, device=device, requires_grad=True)
            teacher = torch.randn(*shape, device=device)
            loss = loss_fn(student, teacher)
            loss.backward()

        benchmark(run)

    def test_logit_lens_kl_forward_backward(self, benchmark, device):
        lm_head = nn.Linear(256, 1000).to(device)
        loss_fn = get_loss_function("logit_lens_kl", output_head=lm_head)

        def run():
            student = torch.randn(4, 64, 256, device=device, requires_grad=True)
            teacher = torch.randn(4, 64, 256, device=device)
            loss = loss_fn(student, teacher)
            loss.backward()

        benchmark(run)

    @pytest.mark.parametrize("loss_type", ["mahal_mse", "mahal_cosine"])
    def test_mahal_forward_backward(self, benchmark, device, loss_type):
        W = torch.randn(1000, 256, device=device)
        loss_fn = get_loss_function(loss_type, weight_matrix=W)

        def run():
            student = torch.randn(4, 64, 256, device=device, requires_grad=True)
            teacher = torch.randn(4, 64, 256, device=device)
            loss = loss_fn(student, teacher)
            loss.backward()

        benchmark(run)

    @pytest.mark.parametrize("loss_type", ["mahal_mse", "mahal_cosine"])
    def test_mahal_with_pre_norm_forward_backward(self, benchmark, device, loss_type):
        W = torch.randn(1000, 256, device=device)
        pre_norm = nn.LayerNorm(256).to(device)
        loss_fn = get_loss_function(loss_type, weight_matrix=W, pre_norm=pre_norm)

        def run():
            student = torch.randn(4, 64, 256, device=device, requires_grad=True)
            teacher = torch.randn(4, 64, 256, device=device)
            loss = loss_fn(student, teacher)
            loss.backward()

        benchmark(run)
