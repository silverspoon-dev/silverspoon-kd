"""Tests for split_gpu module — GPU reordering and remapping logic."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.distributed.split_gpu import (
    compute_student_gpus,
    get_remapped_teacher_devices,
    setup_split_gpu,
)
from silverspoon_kd.distributed.strategies import (
    _apply_device_map,
    _cuda_device_ctx,
    _make_device_hook,
)
from silverspoon_kd.distributed.teacher_placement import TeacherPlacement


class TestComputeStudentGpus:
    def test_basic_complement(self):
        assert compute_student_gpus([0, 1], total_gpus=4) == [2, 3]

    def test_single_teacher(self):
        assert compute_student_gpus([2], total_gpus=4) == [0, 1, 3]

    def test_non_contiguous_teacher(self):
        assert compute_student_gpus([0, 3], total_gpus=4) == [1, 2]

    def test_all_but_one_teacher(self):
        assert compute_student_gpus([0, 1, 2], total_gpus=4) == [3]

    def test_invalid_gpu_id(self):
        with pytest.raises(ValueError, match="exceed available GPUs"):
            compute_student_gpus([5], total_gpus=4)

    def test_two_gpus(self):
        assert compute_student_gpus([0], total_gpus=2) == [1]


class TestRemappedTeacherDevices:
    def test_basic(self):
        # 4 GPUs, teacher=[0,1], student=[2,3]
        # After reorder: student first → offset = 2
        result = get_remapped_teacher_devices(teacher_gpus=[0, 1], student_gpus=[2, 3])
        assert result == [2, 3]

    def test_single_teacher(self):
        result = get_remapped_teacher_devices(teacher_gpus=[2], student_gpus=[0, 1, 3])
        assert result == [3]

    def test_non_contiguous(self):
        result = get_remapped_teacher_devices(teacher_gpus=[0, 3], student_gpus=[1, 2])
        assert result == [2, 3]


class TestSetupSplitGpu:
    def test_sets_cuda_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        placement = TeacherPlacement(teacher_only_devices=[0, 1], strategy="pp")
        student_physical, remapped_teacher = setup_split_gpu(placement)

        assert student_physical == [2, 3]
        assert remapped_teacher == [2, 3]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3,0,1"

    def test_all_teacher_raises(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        placement = TeacherPlacement(teacher_only_devices=[0, 1], strategy="pp")
        with pytest.raises(ValueError, match="no GPUs left"):
            setup_split_gpu(placement)

    def test_non_contiguous_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        placement = TeacherPlacement(teacher_only_devices=[1, 3], strategy="tp")
        student_physical, remapped_teacher = setup_split_gpu(placement)

        assert student_physical == [0, 2]
        assert remapped_teacher == [2, 3]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,2,1,3"

    def test_warns_when_overwriting_cuda_visible_devices(self, monkeypatch, caplog):
        """Overwriting an existing CUDA_VISIBLE_DEVICES logs a warning."""
        import logging

        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        placement = TeacherPlacement(teacher_only_devices=[0], strategy="pp")
        with caplog.at_level(logging.WARNING):
            setup_split_gpu(placement)
        assert any("Overwriting" in msg for msg in caplog.messages)


class TestMakeDeviceHook:
    """Regression tests for _make_device_hook — must move both args AND kwargs."""

    def test_moves_positional_tensor_args(self):
        """Hook must move positional tensor arguments to target device."""
        device = torch.device("cpu")
        hook = _make_device_hook(device)

        t = torch.randn(2, 3)
        new_args, _new_kwargs = hook(None, (t,), {})
        assert new_args[0].device == device

    def test_moves_kwargs_tensors(self):
        """Hook must move keyword argument tensors to target device (regression for kwargs bug)."""
        device = torch.device("cpu")
        hook = _make_device_hook(device)

        t = torch.randn(2, 3)
        _new_args, new_kwargs = hook(None, (), {"hidden_states": t, "mask": t})
        assert new_kwargs["hidden_states"].device == device
        assert new_kwargs["mask"].device == device

    def test_preserves_non_tensor_kwargs(self):
        """Hook must pass through non-tensor kwargs unchanged."""
        device = torch.device("cpu")
        hook = _make_device_hook(device)

        _new_args, new_kwargs = hook(None, (), {"use_cache": True, "output_attentions": False})
        assert new_kwargs["use_cache"] is True
        assert new_kwargs["output_attentions"] is False

    def test_mixed_args_and_kwargs(self):
        """Hook must handle a mix of tensor/non-tensor in both args and kwargs."""
        device = torch.device("cpu")
        hook = _make_device_hook(device)

        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        new_args, new_kwargs = hook(
            None,
            (t1, 42, "str_arg"),
            {"key": t2, "flag": True},
        )
        assert isinstance(new_args[0], torch.Tensor)
        assert new_args[1] == 42
        assert new_args[2] == "str_arg"
        assert isinstance(new_kwargs["key"], torch.Tensor)
        assert new_kwargs["flag"] is True

    def test_returns_tuple_and_dict(self):
        """Hook must return (tuple, dict) for PyTorch pre-hook with_kwargs protocol."""
        device = torch.device("cpu")
        hook = _make_device_hook(device)

        result = hook(None, (torch.randn(1),), {"k": torch.randn(1)})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        assert isinstance(result[1], dict)


class TestApplyDeviceMap:
    """Regression tests for _apply_device_map hook registration."""

    def test_hooks_registered_with_kwargs_support(self):
        """_apply_device_map hooks must use with_kwargs=True so kwargs are moved."""

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(4, 4)

            def forward(self, x):
                return self.child(x)

        model = Parent()
        device = torch.device("cpu")
        _apply_device_map(model, {"child": device})

        # Verify a pre-forward hook was registered
        hooks = list(model.child._forward_pre_hooks.values())
        assert len(hooks) >= 1

    def test_modulelist_children_get_hooks(self):
        """Hooks must be on ModuleList children, not the container itself."""

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Parent()
        device = torch.device("cpu")
        _apply_device_map(model, {"layers": device})

        # Each child should have a hook, not the ModuleList container
        for child in model.layers:
            assert len(child._forward_pre_hooks) >= 1

    def test_repeated_calls_do_not_accumulate_hooks(self):
        """Calling _apply_device_map twice must not double-register hooks."""

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(4, 4)

            def forward(self, x):
                return self.child(x)

        model = Parent()
        device = torch.device("cpu")
        _apply_device_map(model, {"child": device})
        hooks_after_first = len(model.child._forward_pre_hooks)

        _apply_device_map(model, {"child": device})
        hooks_after_second = len(model.child._forward_pre_hooks)

        assert hooks_after_second == hooks_after_first

    def test_repeated_calls_modulelist_no_accumulation(self):
        """Calling _apply_device_map twice on ModuleList children must not accumulate."""

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Parent()
        device = torch.device("cpu")
        _apply_device_map(model, {"layers": device})
        counts_first = [len(c._forward_pre_hooks) for c in model.layers]

        _apply_device_map(model, {"layers": device})
        counts_second = [len(c._forward_pre_hooks) for c in model.layers]

        assert counts_second == counts_first


class TestCudaDeviceCtx:
    """Tests for _cuda_device_ctx — must restore device even on exception."""

    def test_restores_device_on_normal_exit(self):
        """Device is restored after the with-block completes normally."""
        device_log = []

        with (
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.set_device", side_effect=device_log.append),
        ):
            with _cuda_device_ctx(3):
                pass

        # set_device(3) on entry, set_device(0) on exit
        assert device_log == [3, 0]

    def test_restores_device_on_exception(self):
        """Device is restored even when the body raises."""
        device_log = []

        with (
            patch("torch.cuda.current_device", return_value=1),
            patch("torch.cuda.set_device", side_effect=device_log.append),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                with _cuda_device_ctx(5):
                    raise RuntimeError("boom")

        # set_device(5) on entry, set_device(1) in finally
        assert device_log == [5, 1]


# ── Pipeline placement (device map construction) ─────────────────────────


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x, position_embeddings=None):
        if position_embeddings is not None:
            cos, sin = position_embeddings
            x = x + cos + sin
        return self.mlp(self.attn(x))


class _CausalLM(nn.Module):
    """Mimics the HF layout: ``model.{embed_tokens,layers,norm}`` + ``lm_head``."""

    def __init__(self, dim=8, vocab=16, num_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, dim)
        self.model.layers = nn.ModuleList([_Block(dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        cos = torch.zeros_like(x)
        sin = torch.zeros_like(x)
        for layer in self.model.layers:
            x = layer(x, position_embeddings=(cos, sin))
        return self.lm_head(self.model.norm(x))


class TestBalancedDeviceMap:
    """_build_balanced_device_map splits the layer stack, not the top level."""

    def test_layer_stack_is_split_contiguously(self):
        from silverspoon_kd.distributed.strategies import _build_balanced_device_map

        model = _CausalLM(num_layers=4)
        device_map = _build_balanced_device_map(model, [0, 1], "cpu")

        # Embeddings on the first device, head and final norm on the last.
        assert device_map["model.embed_tokens"] == torch.device("cpu", 0)
        assert device_map["model.norm"] == torch.device("cpu", 1)
        assert device_map["lm_head"] == torch.device("cpu", 1)
        # The stack itself is assigned per layer, in contiguous halves.
        assert "model.layers" not in device_map
        layers = [device_map[f"model.layers.{i}"].index for i in range(4)]
        assert layers == [0, 0, 1, 1]

    def test_uneven_split_stays_contiguous_and_uses_every_device(self):
        from silverspoon_kd.distributed.strategies import _build_balanced_device_map

        model = _CausalLM(num_layers=5)
        device_map = _build_balanced_device_map(model, [2, 5, 7], "cpu")
        layers = [device_map[f"model.layers.{i}"].index for i in range(5)]
        assert layers == sorted(layers)
        assert set(layers) == {2, 5, 7}

    def test_model_without_layer_stack_splits_top_level(self):
        from silverspoon_kd.distributed.strategies import _build_balanced_device_map

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)
                self.c = nn.Linear(8, 8)

        device_map = _build_balanced_device_map(Flat(), [0, 1], "cpu")
        assert [device_map[n].index for n in ("a", "b", "c")] == sorted(
            device_map[n].index for n in ("a", "b", "c")
        )
        assert {d.index for d in device_map.values()} == {0, 1}

    def test_sequential_root_is_the_stack(self):
        from silverspoon_kd.distributed.strategies import _build_balanced_device_map

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4))
        device_map = _build_balanced_device_map(model, [0, 1], "cpu")
        assert [device_map[str(i)].index for i in range(4)] == [0, 0, 1, 1]


class TestPlaceTeacherPP:
    """place_teacher_pp keeps the forward pass working across the split."""

    def test_forward_matches_unplaced_model(self):
        from silverspoon_kd.distributed.strategies import place_teacher_pp

        torch.manual_seed(0)
        model = _CausalLM(num_layers=4)
        input_ids = torch.randint(0, 16, (2, 5))
        with torch.no_grad():
            expected = model(input_ids).clone()

        placed = place_teacher_pp(model, [0, 1], device_type="cpu")
        assert placed is model
        with torch.no_grad():
            actual = placed(input_ids)
        assert torch.allclose(actual, expected)

    def test_every_layer_has_a_device_hook(self):
        from silverspoon_kd.distributed.strategies import place_teacher_pp

        model = place_teacher_pp(_CausalLM(num_layers=4), [0, 1], device_type="cpu")
        for layer in model.model.layers:
            hooks = [
                h
                for h in layer._forward_pre_hooks.values()
                if getattr(h, "_skd_device_hook", False)
            ]
            assert len(hooks) == 1
        assert not any(
            getattr(h, "_skd_device_hook", False)
            for h in model.model.layers._forward_pre_hooks.values()
        )

    def test_single_device_moves_whole_model(self):
        from silverspoon_kd.distributed.strategies import place_teacher_pp

        model = place_teacher_pp(_CausalLM(), [3], device_type="cpu")
        assert all(p.device.type == "cpu" for p in model.parameters())


class TestDeviceHookNestedInputs:
    """The pre-forward hook moves tensors nested inside tuples, lists and dicts."""

    def test_nested_structures_are_traversed(self):
        device = torch.device("cpu")
        hook = _make_device_hook(device)
        cos, sin = torch.zeros(2), torch.ones(2)
        args, kwargs = hook(
            nn.Identity(),
            (torch.zeros(1), [torch.zeros(1)]),
            {"position_embeddings": (cos, sin), "cache": {"k": torch.zeros(1)}, "flag": True},
        )
        assert isinstance(args[1], list) and args[1][0].device == device
        assert kwargs["position_embeddings"][1].device == device
        assert torch.equal(kwargs["position_embeddings"][1], sin)
        assert kwargs["cache"]["k"].device == device
        assert kwargs["flag"] is True
