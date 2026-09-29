"""Tests for TrainingArguments mixed-precision warnings and native dtype behavior."""

import logging

import pytest
import torch

from silverspoon_kd.alignments import Alignment
from silverspoon_kd.distillers.blockwise_distiller import BlockwiseDistiller
from silverspoon_kd.distillers.holistic_distiller import HolisticDistiller
from silverspoon_kd.distillers.response_based_distiller import ResponseBasedDistiller
from silverspoon_kd.training_arguments import (
    TrainingArguments,
)
from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel

_NUM_LAYERS = 2
_DATASET = DummyDataset(num_samples=20, seq_len=16)


def _make_args(tmp_path, args_cls=TrainingArguments, **extra):
    extra.setdefault("save_strategy", "no")
    return args_cls(
        output_dir=str(tmp_path),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=1,
        max_steps=3,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=True,
        **extra,
    )


# ── Warning tests ────────────────────────────────────────────────────────


class TestMixedPrecisionWarning:
    """Verify that fp16/bf16 flags emit a warning."""

    def test_fp16_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, fp16=True)
        assert any("fp16=True" in msg for msg in caplog.messages)

    def test_bf16_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, bf16=True)
        assert any("bf16=True" in msg for msg in caplog.messages)

    def test_fp16_full_eval_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, fp16_full_eval=True)
        assert any("fp16_full_eval=True" in msg for msg in caplog.messages)

    def test_bf16_full_eval_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, bf16_full_eval=True)
        assert any("bf16_full_eval=True" in msg for msg in caplog.messages)

    def test_multiple_flags_warns_all(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, bf16=True, bf16_full_eval=True)
        assert any("bf16=True" in msg and "bf16_full_eval=True" in msg for msg in caplog.messages)

    def test_no_warning_when_neither_set(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path)
        assert not any("was set" in msg or "bypass" in msg for msg in caplog.messages)

    def test_warning_propagates_through_subclass(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, args_cls=TrainingArguments, bf16=True)
        assert any("bf16=True" in msg for msg in caplog.messages)

    def test_warning_propagates_through_response_based(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="silverspoon_kd.training_arguments"):
            _make_args(tmp_path, args_cls=TrainingArguments, fp16=True)
        assert any("fp16=True" in msg for msg in caplog.messages)


# ── Native dtype tests ───────────────────────────────────────────────────


def _make_alignments(teacher, student, **kwargs):
    alignments = []
    for i in range(_NUM_LAYERS):
        alignments.append(
            Alignment(
                teacher_block=teacher.get_layer(i),
                student_block=student.get_layer(i),
                teacher_model_name="teacher",
                student_model_name="student",
                teacher_module_name=f"layers.{i}",
                student_module_name=f"layers.{i}",
                **kwargs,
            )
        )
    return alignments


class TestNativeDtypeBehavior:
    """Verify models train in their native parameter dtype, not autocast."""

    def _check_model_dtype(self, model, expected_dtype):
        for name, param in model.named_parameters():
            assert param.dtype == expected_dtype, (
                f"Parameter {name} has dtype {param.dtype}, expected {expected_dtype}"
            )

    def test_blockwise_preserves_fp32(self, tmp_path):
        """fp32 models stay fp32 during and after training."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS)
        student = SimpleModel(64, 128, _NUM_LAYERS)
        aligns = _make_alignments(teacher, student)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = 1
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.float32)
        self._check_model_dtype(student, torch.float32)

    def test_blockwise_preserves_bf16(self, tmp_path):
        """bf16 models stay bf16 even without bf16=True in args."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)
        aligns = _make_alignments(teacher, student)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = 1
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.bfloat16)
        self._check_model_dtype(student, torch.bfloat16)

    def test_blockwise_mixed_dtype_with_auto_match(self, tmp_path):
        """fp32 teacher + bf16 student works with auto_dtype_match."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)
        aligns = _make_alignments(teacher, student, auto_dtype_match=True)

        args = _make_args(tmp_path, args_cls=TrainingArguments)
        args._n_gpu = 1
        d = BlockwiseDistiller(
            teacher_model=teacher,
            alignments=aligns,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.float32)
        self._check_model_dtype(student, torch.bfloat16)

    def test_response_based_preserves_fp32(self, tmp_path):
        """fp32 models stay fp32 during and after training."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS)
        student = SimpleModel(64, 128, _NUM_LAYERS)

        args = _make_args(tmp_path, args_cls=TrainingArguments, learning_rate=1e-3)
        args._n_gpu = 1
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.float32)
        self._check_model_dtype(student, torch.float32)

    def test_response_based_preserves_bf16(self, tmp_path):
        """bf16 models stay bf16 even without bf16=True in args."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)

        args = _make_args(
            tmp_path,
            args_cls=TrainingArguments,
            learning_rate=1e-3,
        )
        args._n_gpu = 1
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.bfloat16)
        self._check_model_dtype(student, torch.bfloat16)

    def test_response_based_mixed_dtype_with_auto_match(self, tmp_path):
        """fp32 teacher + bf16 student works with auto_dtype_match."""
        teacher = SimpleModel(64, 128, _NUM_LAYERS)
        student = SimpleModel(64, 128, _NUM_LAYERS).to(dtype=torch.bfloat16)

        args = _make_args(
            tmp_path,
            args_cls=TrainingArguments,
            learning_rate=1e-3,
            auto_dtype_match=True,
        )
        args._n_gpu = 1
        d = ResponseBasedDistiller(
            student_model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=_DATASET,
        )
        d.train()

        self._check_model_dtype(teacher, torch.float32)
        self._check_model_dtype(student, torch.bfloat16)


# ── Magnitude-aware weighting parameter tests ─────────────────────────────


class TestMagnitudeAwareWeightingParam:
    """Verify magnitude_aware_weighting lives on TrainingArguments."""

    def test_accepts_magnitude_aware(self, tmp_path):
        """TrainingArguments accepts magnitude_aware_weighting."""
        args = _make_args(tmp_path, magnitude_aware_weighting=True)
        assert args.magnitude_aware_weighting is True

    def test_defaults_to_false(self, tmp_path):
        """TrainingArguments defaults magnitude_aware_weighting to False."""
        args = _make_args(tmp_path)
        assert args.magnitude_aware_weighting is False


class TestAlphaValidation:
    """Regression tests: alpha must be validated at init time."""

    @pytest.mark.parametrize("alpha", [-0.1, -1.0, 1.1, 2.0])
    def test_invalid_alpha_raises(self, tmp_path, alpha):
        """alpha outside [0, 1] must raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            _make_args(tmp_path, args_cls=TrainingArguments, alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_valid_alpha_succeeds(self, tmp_path, alpha):
        """alpha in [0, 1] must be accepted."""
        args = _make_args(tmp_path, args_cls=TrainingArguments, alpha=alpha)
        assert args.alpha == alpha


class TestIrrelevantArgsWarning:
    """Tests that each distiller warns on every irrelevant non-default arg
    and stays silent on every arg it actually uses.

    The mapping of params to distillers:
        BlockwiseDistiller:      backward_per_block, deepcopy_captured_args_and_kwargs
        HolisticDistiller:       alpha, deepcopy_captured_args_and_kwargs, magnitude_aware_weighting
        ResponseBasedDistiller:  alpha, auto_device_match, auto_dtype_match, magnitude_aware_weighting
    """

    _PARAM_NON_DEFAULTS: dict = {
        "alpha": 0.5,
        "auto_device_match": True,
        "auto_dtype_match": True,
        "backward_per_block": True,
        "deepcopy_captured_args_and_kwargs": True,
        "magnitude_aware_weighting": True,
    }

    _USED_BY: dict = {
        "BlockwiseDistiller": {"backward_per_block", "deepcopy_captured_args_and_kwargs"},
        "HolisticDistiller": {
            "alpha",
            "deepcopy_captured_args_and_kwargs",
            "magnitude_aware_weighting",
        },
        "ResponseBasedDistiller": {
            "alpha",
            "auto_device_match",
            "auto_dtype_match",
            "magnitude_aware_weighting",
        },
    }

    def _make_distiller(self, cls_name, tmp_path, **extra_args):
        teacher = SimpleModel(64, 128, _NUM_LAYERS)
        student = SimpleModel(64, 128, _NUM_LAYERS)
        args = _make_args(tmp_path, **extra_args)
        if cls_name == "BlockwiseDistiller":
            aligns = _make_alignments(teacher, student)
            return BlockwiseDistiller(
                teacher_model=teacher,
                alignments=aligns,
                args=args,
                train_dataset=_DATASET,
            )
        elif cls_name == "HolisticDistiller":
            aligns = _make_alignments(teacher, student)
            return HolisticDistiller(
                student_model=student,
                teacher_model=teacher,
                alignments=aligns,
                args=args,
                train_dataset=_DATASET,
            )
        else:
            return ResponseBasedDistiller(
                student_model=student,
                teacher_model=teacher,
                args=args,
                train_dataset=_DATASET,
            )

    @pytest.mark.parametrize(
        "cls_name", ["BlockwiseDistiller", "HolisticDistiller", "ResponseBasedDistiller"]
    )
    def test_no_warning_on_defaults(self, cls_name, tmp_path, caplog):
        """No irrelevant-args warning when all params are at defaults."""
        with caplog.at_level(logging.WARNING):
            self._make_distiller(cls_name, tmp_path)
        assert not any("does not use" in msg for msg in caplog.messages)

    @pytest.mark.parametrize(
        "cls_name", ["BlockwiseDistiller", "HolisticDistiller", "ResponseBasedDistiller"]
    )
    def test_warns_on_every_irrelevant_param(self, cls_name, tmp_path, caplog):
        """Each param NOT used by this distiller triggers a warning."""
        used = self._USED_BY[cls_name]
        for param, value in self._PARAM_NON_DEFAULTS.items():
            if param in used:
                continue
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                self._make_distiller(cls_name, tmp_path, **{param: value})
            assert any(param in msg and cls_name in msg for msg in caplog.messages), (
                f"{cls_name} should warn about {param}={value!r} but didn't"
            )

    @pytest.mark.parametrize(
        "cls_name", ["BlockwiseDistiller", "HolisticDistiller", "ResponseBasedDistiller"]
    )
    def test_no_warning_on_used_params(self, cls_name, tmp_path, caplog):
        """Each param used by this distiller does NOT trigger a warning."""
        used = self._USED_BY[cls_name]
        for param in used:
            value = self._PARAM_NON_DEFAULTS[param]
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                self._make_distiller(cls_name, tmp_path, **{param: value})
            assert not any(param in msg and "does not use" in msg for msg in caplog.messages), (
                f"{cls_name} should NOT warn about {param}={value!r} but did"
            )


class TestFSDPVersionDefault:
    """``fsdp`` selects FSDP1 unless the caller picks a version explicitly."""

    def test_fsdp_defaults_to_version_1(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path / "out"), fsdp="full_shard")
        assert args.fsdp_config["version"] == 1

    def test_fsdp_config_entries_are_kept(self, tmp_path):
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            fsdp="full_shard",
            fsdp_config={"limit_all_gathers": True},
        )
        assert args.fsdp_config["version"] == 1
        assert args.fsdp_config["limit_all_gathers"] is True

    def test_explicit_version_is_respected(self, tmp_path):
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            fsdp="full_shard",
            fsdp_config={"version": 2},
        )
        assert args.fsdp_config["version"] == 2

    def test_no_fsdp_leaves_config_alone(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path / "out"))
        assert not args.fsdp
        assert not (args.fsdp_config or {}).get("version")
