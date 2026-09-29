"""Tests for the Distiller factory function."""

import pytest

from silverspoon_kd import Distiller, TrainingArguments
from silverspoon_kd.distillers import (
    BlockwiseDistiller,
    HolisticDistiller,
    ResponseBasedDistiller,
)

from ..conftest import DummyDataset, SimpleModel, create_alignment


def _make_args(tmp_path, **extra):
    extra.setdefault("save_strategy", "no")
    return TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        dataloader_num_workers=0,
        report_to=[],
        use_cpu=True,
        **extra,
    )


_DATASET = DummyDataset(num_samples=10, seq_len=16)


def _make_teacher_student_alignments():
    teacher = SimpleModel(64, 128, 2)
    student = SimpleModel(64, 128, 2)
    alignments = [
        create_alignment(
            teacher.get_layer(i),
            student.get_layer(i),
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            teacher_hidden_dim=128,
        )
        for i in range(2)
    ]
    return teacher, student, alignments


class TestDistillerFactory:
    """Tests for Distiller() factory function."""

    @pytest.mark.parametrize("alias", ["blockwise", "bkd"])
    def test_creates_blockwise(self, alias, tmp_path):
        """Blockwise aliases produce a BlockwiseDistiller."""
        teacher, _, alignments = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type=alias,
            teacher_model=teacher,
            alignments=alignments,
            args=_make_args(tmp_path),
            train_dataset=_DATASET,
        )
        assert isinstance(distiller, BlockwiseDistiller)

    @pytest.mark.parametrize("alias", ["holistic", "hkd"])
    def test_creates_holistic(self, alias, tmp_path):
        """Holistic aliases produce a HolisticDistiller."""
        teacher, student, alignments = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type=alias,
            student_model=student,
            teacher_model=teacher,
            alignments=alignments,
            args=_make_args(tmp_path),
            train_dataset=_DATASET,
        )
        assert isinstance(distiller, HolisticDistiller)

    @pytest.mark.parametrize("alias", ["response_based", "reskd"])
    def test_creates_response_based(self, alias, tmp_path):
        """Response-based aliases produce a ResponseBasedDistiller."""
        teacher, student, _ = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type=alias,
            student_model=student,
            teacher_model=teacher,
            args=_make_args(tmp_path),
            train_dataset=_DATASET,
        )
        assert isinstance(distiller, ResponseBasedDistiller)

    def test_case_insensitive(self, tmp_path):
        """distiller_type is case-insensitive."""
        teacher, student, _ = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type="ResKD",
            student_model=student,
            teacher_model=teacher,
            args=_make_args(tmp_path),
            train_dataset=_DATASET,
        )
        assert isinstance(distiller, ResponseBasedDistiller)

    def test_hyphen_alias(self, tmp_path):
        """Hyphens are accepted (normalized to underscores)."""
        teacher, student, _ = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type="response-based",
            student_model=student,
            teacher_model=teacher,
            args=_make_args(tmp_path),
            train_dataset=_DATASET,
        )
        assert isinstance(distiller, ResponseBasedDistiller)

    def test_unknown_type_raises(self):
        """Unknown distiller_type raises ValueError with available options."""
        with pytest.raises(ValueError, match="Unknown distiller_type"):
            Distiller(distiller_type="nonexistent")

    def test_error_message_lists_available(self):
        """Error message includes all valid aliases."""
        with pytest.raises(
            ValueError, match="bkd.*blockwise.*hkd.*holistic.*reskd.*response_based"
        ):
            Distiller(distiller_type="bad")

    def test_kwargs_forwarded_to_concrete_class(self, tmp_path):
        """Constructor kwargs are forwarded to the concrete distiller."""
        teacher, student, alignments = _make_teacher_student_alignments()
        distiller = Distiller(
            distiller_type="blockwise",
            teacher_model=teacher,
            alignments=alignments,
            args=_make_args(tmp_path, backward_per_block=True),
            train_dataset=_DATASET,
        )
        assert distiller.args.backward_per_block is True

    def test_importable_from_top_level(self):
        """Distiller is importable from silverspoon_kd."""
        from silverspoon_kd import Distiller as top_level

        assert top_level is Distiller
