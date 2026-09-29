"""Tests for TeacherPlacement dataclass and normalization."""

import pytest

from silverspoon_kd.distributed.teacher_placement import (
    TeacherPlacement,
    normalize_teacher_placement,
)
from silverspoon_kd.training_arguments import TrainingArguments


class TestTeacherPlacement:
    """TeacherPlacement validation."""

    def test_valid_pp(self):
        p = TeacherPlacement(teacher_only_devices=[0, 1], strategy="pp")
        assert p.strategy == "pp"
        assert p.teacher_only_devices == [0, 1]

    def test_valid_tp(self):
        p = TeacherPlacement(teacher_only_devices=[2, 3], strategy="tp")
        assert p.strategy == "tp"

    def test_valid_sharded(self):
        p = TeacherPlacement(teacher_only_devices=[0], strategy="sharded")
        assert p.strategy == "sharded"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            TeacherPlacement(teacher_only_devices=[0], strategy="ddp")

    def test_empty_devices_raises(self):
        with pytest.raises(ValueError, match="teacher_only_devices is required"):
            TeacherPlacement(teacher_only_devices=[], strategy="pp")

    def test_wrap_cls_only_sharded(self):
        # Valid: wrap_cls with sharded
        p = TeacherPlacement(
            teacher_only_devices=[0], strategy="sharded", wrap_cls="LlamaDecoderLayer"
        )
        assert p.wrap_cls == "LlamaDecoderLayer"

    def test_wrap_cls_with_non_sharded_raises(self):
        with pytest.raises(ValueError, match="wrap_cls is only valid"):
            TeacherPlacement(teacher_only_devices=[0], strategy="pp", wrap_cls="Block")

    def test_default_strategy_is_pp(self):
        p = TeacherPlacement(teacher_only_devices=[0])
        assert p.strategy == "pp"

    def test_device_type_default_none(self):
        p = TeacherPlacement(teacher_only_devices=[0])
        assert p.device_type is None


class TestNormalize:
    """normalize_teacher_placement tests."""

    def test_none_defaults_to_replicated(self):
        assert normalize_teacher_placement(None) == "replicated"

    def test_string_replicated(self):
        assert normalize_teacher_placement("replicated") == "replicated"

    def test_string_sharded(self):
        assert normalize_teacher_placement("sharded") == "sharded"

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid teacher_placement string"):
            normalize_teacher_placement("ddp")

    def test_dict_conversion(self):
        result = normalize_teacher_placement({"teacher_only_devices": [0, 1], "strategy": "tp"})
        assert isinstance(result, TeacherPlacement)
        assert result.strategy == "tp"
        assert result.teacher_only_devices == [0, 1]

    def test_passthrough_teacher_placement(self):
        p = TeacherPlacement(teacher_only_devices=[0], strategy="pp")
        assert normalize_teacher_placement(p) is p

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            normalize_teacher_placement(42)


class TestTrainingArgs:
    """TrainingArguments integration with teacher_placement."""

    def test_default_replicated(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path))
        assert args.teacher_placement == "replicated"

    def test_string_sharded(self, tmp_path):
        args = TrainingArguments(output_dir=str(tmp_path), teacher_placement="sharded")
        assert args.teacher_placement == "sharded"

    def test_dict_accepted(self, tmp_path):
        args = TrainingArguments(
            output_dir=str(tmp_path),
            teacher_placement={"teacher_only_devices": [0, 1], "strategy": "pp"},
        )
        assert isinstance(args.teacher_placement, TeacherPlacement)
        assert args.teacher_placement.strategy == "pp"

    def test_teacher_placement_instance(self, tmp_path):
        p = TeacherPlacement(teacher_only_devices=[2, 3], strategy="tp")
        args = TrainingArguments(output_dir=str(tmp_path), teacher_placement=p)
        assert args.teacher_placement is p
