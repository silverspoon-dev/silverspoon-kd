"""
Tests for create_alignments and alignment configuration.
"""

import pytest
import torch
import torch.nn as nn

from silverspoon_kd.alignments import (
    Alignment,
    create_alignments,
)


class SimpleModel(nn.Module):
    """Simple model for testing alignments."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.name_or_path = "test_model"
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(128, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class NestedModel(nn.Module):
    """Model with nested module structure for testing."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.name_or_path = "nested_model"

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.self_attn = nn.Linear(dim, dim)
                self.ffn = nn.Linear(dim, dim)

            def forward(self, x):
                return self.ffn(self.self_attn(x))

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.fixture
def teacher_model():
    """Create a teacher model."""
    return SimpleModel(hidden_dim=128, num_layers=3)


@pytest.fixture
def student_model():
    """Create a smaller student model."""
    return SimpleModel(hidden_dim=64, num_layers=3)


@pytest.fixture
def same_dim_student_model():
    """Create a student model with same dimensions as teacher."""
    return SimpleModel(hidden_dim=128, num_layers=3)


@pytest.fixture
def nested_teacher_model():
    """Create a nested teacher model."""
    return NestedModel(hidden_dim=128, num_layers=3)


@pytest.fixture
def nested_student_model():
    """Create a nested student model."""
    return NestedModel(hidden_dim=64, num_layers=3)


class TestCreateAlignmentsWithString:
    """Tests for create_alignments with a single string regex."""

    def test_basic_alignment_creation(self, teacher_model, same_dim_student_model):
        """Test basic alignment creation with a single regex string."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
        )

        assert len(alignments) == 3
        for alignment in alignments:
            assert isinstance(alignment, Alignment)

    def test_single_layer_match(self, teacher_model, same_dim_student_model):
        """Test matching a single layer by exact name."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules="layers.0",
        )

        assert len(alignments) == 1
        assert isinstance(alignments[0], Alignment)
        assert alignments[0].teacher_module_name == "layers.0"
        assert alignments[0].student_module_name == "layers.0"

    def test_model_names_set(self, teacher_model, same_dim_student_model):
        """Test that teacher and student model names are set correctly."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
        )

        for alignment in alignments:
            assert alignment.teacher_model_name == "test_model"
            assert alignment.student_model_name == "test_model"

    def test_auto_projector_enabled_by_default(self, teacher_model, same_dim_student_model):
        """Test that auto_projector is enabled by default."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
        )

        for alignment in alignments:
            assert alignment.auto_projector is True

    def test_auto_projector_disabled(self, teacher_model, same_dim_student_model):
        """Test that auto_projector can be disabled."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            auto_projector=False,
        )

        for alignment in alignments:
            assert alignment.auto_projector is False


class TestCreateAlignmentsWithList:
    """Tests for create_alignments with a list of regex patterns."""

    def test_list_of_exact_names(self, teacher_model, same_dim_student_model):
        """Test alignment creation with list of exact module names."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=["layers.0", "layers.1", "layers.2"],
        )

        assert len(alignments) == 3
        for alignment in alignments:
            assert isinstance(alignment, Alignment)

    def test_list_with_regex(self, teacher_model, same_dim_student_model):
        """Test alignment creation with list containing regex patterns."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=[r"layers\.\d+$"],
        )

        assert len(alignments) == 3

    def test_student_not_found_raises(self, teacher_model):
        """Test that list mapping with non-matching student module raises ValueError."""

        class DifferentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.name_or_path = "different_model"
                self.blocks = nn.ModuleList([nn.Linear(128, 128)])

            def forward(self, x):
                return self.blocks[0](x)

        student = DifferentModel()
        with pytest.raises(ValueError, match="Could not find student module"):
            create_alignments(
                teacher_model=teacher_model,
                student_model=student,
                modules=["layers.0"],
            )


class TestCreateAlignmentsWithDict:
    """Tests for create_alignments with dict-based teacher->student mapping."""

    def test_dict_mapping_basic(self, teacher_model, same_dim_student_model):
        """Test dict-based teacher-student module mapping."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules={
                r"layers\.0$": r"layers\.0$",
                r"layers\.1$": r"layers\.1$",
            },
        )
        assert len(alignments) == 2
        for alignment in alignments:
            assert isinstance(alignment, Alignment)

    def test_dict_mapping_with_backreferences(self):
        """Test dict mapping with regex backreferences."""
        teacher = NestedModel(hidden_dim=128, num_layers=3)
        student = NestedModel(hidden_dim=128, num_layers=3)

        alignments = create_alignments(
            teacher_model=teacher,
            student_model=student,
            modules={
                r"model\.layers\.(\d+)$": r"model.layers.\1",
            },
        )
        assert len(alignments) == 3
        for alignment in alignments:
            assert isinstance(alignment, Alignment)

    def test_dict_mapping_student_not_found_raises(self, teacher_model, student_model):
        """Test that non-existent student module raises ValueError."""
        with pytest.raises(ValueError, match="Could not find student module"):
            create_alignments(
                teacher_model=teacher_model,
                student_model=student_model,
                modules={
                    r"layers\.0$": r"nonexistent_module",
                },
            )


class TestCreateAlignmentsNoMatch:
    """Tests for create_alignments error when no modules match."""

    def test_no_modules_match_raises(self, teacher_model, same_dim_student_model):
        """Test that no matching modules raises ValueError."""
        with pytest.raises(ValueError, match="No modules matched"):
            create_alignments(
                teacher_model=teacher_model,
                student_model=same_dim_student_model,
                modules=r"nonexistent_module_pattern",
            )

    def test_no_modules_match_dict_raises(self, teacher_model, same_dim_student_model):
        """Test that no matching modules with dict raises ValueError."""
        with pytest.raises(ValueError, match="No modules matched"):
            create_alignments(
                teacher_model=teacher_model,
                student_model=same_dim_student_model,
                modules={r"nonexistent_pattern": r"replacement"},
            )


class TestCreateAlignmentsOutputSelector:
    """Tests for output_selector_index parameter in create_alignments."""

    def test_output_selector_index_sets_selectors(self, teacher_model, same_dim_student_model):
        """Test that output_selector_index sets OutputSelector on all alignments."""
        from silverspoon_kd.alignments import OutputSelector

        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            output_selector_index=1,
        )

        for alignment in alignments:
            assert isinstance(alignment.teacher_output_selector, OutputSelector)
            assert alignment.teacher_output_selector.index == 1
            assert isinstance(alignment.student_output_selector, OutputSelector)
            assert alignment.student_output_selector.index == 1


class TestCreateAlignmentsWithLossFunction:
    """Tests for loss function parameter in create_alignments."""

    def test_custom_loss_function(self, teacher_model, same_dim_student_model):
        """Test alignment creation with custom callable loss function."""
        custom_loss = nn.L1Loss()

        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function=custom_loss,
        )

        for alignment in alignments:
            assert alignment.loss_function is custom_loss

    def test_string_loss_function(self, teacher_model, same_dim_student_model):
        """Test that a string loss name flows through to alignments."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="cosine",
        )

        assert len(alignments) > 0
        for alignment in alignments:
            # The loss function should be callable and not an MSELoss
            pred = torch.randn(2, 128)
            target = torch.randn(2, 128)
            loss = alignment.loss_function(pred, target)
            assert loss.dim() == 0


class TestCreateAlignmentsMaxGradNorm:
    """Tests for max_grad_norm parameter in create_alignments."""

    def test_max_grad_norm_set(self, teacher_model, same_dim_student_model):
        """Test that max_grad_norm is set on alignments."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            max_grad_norm=1.0,
        )

        for alignment in alignments:
            assert alignment.max_grad_norm == 1.0


class TestCreateAlignmentsModelNames:
    """Tests for model name handling in create_alignments."""

    def test_model_name_with_special_chars(self, teacher_model, same_dim_student_model):
        """Test that special characters in model names are handled."""
        teacher_model.name_or_path = "org/model-name_v1.2"
        same_dim_student_model.name_or_path = "org/student-model_v0.1"

        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
        )

        for alignment in alignments:
            assert alignment.teacher_model_name == "org/model-name_v1.2"
            assert alignment.student_model_name == "org/student-model_v0.1"


class TestCreateAlignmentsWithLossFunctionKwargs:
    """Tests for loss_function_kwargs in create_alignments."""

    def test_loss_kwargs_forwarded(self, teacher_model, same_dim_student_model):
        """loss_function_kwargs are forwarded to the loss factory."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="kl_div",
            loss_function_kwargs={"temperature": 4.0},
        )
        # Should not raise; loss is callable
        assert len(alignments) > 0
        pred = torch.randn(2, 128)
        target = torch.randn(2, 128)
        loss = alignments[0].loss_function(pred, target)
        assert loss.dim() == 0

    def test_mahal_mse_with_direct_attr(self, teacher_model, same_dim_student_model):
        """Mahalanobis MSE with direct attribute access for weight matrix."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="mahal_mse",
            loss_function_kwargs={
                "weight_matrix": teacher_model.lm_head.weight,
            },
        )
        assert len(alignments) > 0
        pred = torch.randn(2, 128)
        target = torch.randn(2, 128)
        loss = alignments[0].loss_function(pred, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_mahal_cosine_with_direct_attr(self, teacher_model, same_dim_student_model):
        """Mahalanobis cosine with direct attribute access for weight matrix."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="mahal_cosine",
            loss_function_kwargs={
                "weight_matrix": teacher_model.lm_head.weight,
            },
        )
        assert len(alignments) > 0
        pred = torch.randn(2, 128)
        target = torch.randn(2, 128)
        loss = alignments[0].loss_function(pred, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_kwargs_none_by_default(self, teacher_model, same_dim_student_model):
        """No loss_function_kwargs → works as before."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="cosine",
        )
        assert len(alignments) > 0

    def test_magic_strings_are_not_resolved(self, teacher_model, same_dim_student_model):
        """Strings such as 'teacher:...' are passed through as-is, never resolved.

        ``"teacher:lm_head.weight"`` is not resolved to a tensor: it stays a
        string, so the loss factory receives a string where it expects a
        tensor and raises TypeError.
        """
        with pytest.raises((TypeError, AttributeError)):
            create_alignments(
                teacher_model=teacher_model,
                student_model=same_dim_student_model,
                modules=r"layers\.\d+$",
                loss_function="mahal_mse",
                loss_function_kwargs={"weight_matrix": "teacher:lm_head.weight"},
            )

    def test_direct_attribute_access_works(self, teacher_model, same_dim_student_model):
        """Passing a model attribute directly (no helper) works fine."""
        alignments = create_alignments(
            teacher_model=teacher_model,
            student_model=same_dim_student_model,
            modules=r"layers\.\d+$",
            loss_function="mahal_mse",
            loss_function_kwargs={
                "weight_matrix": teacher_model.lm_head.weight,
            },
        )
        pred = torch.randn(2, 128)
        target = torch.randn(2, 128)
        loss = alignments[0].loss_function(pred, target)
        assert torch.isfinite(loss)
