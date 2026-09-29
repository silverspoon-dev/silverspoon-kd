"""Unit tests for loss registry."""

import pytest
import torch

from silverspoon_kd.losses.registry import (
    LOSS_REGISTRY,
    get_loss_function,
)


class TestLossRegistry:
    """Tests for loss registry and get_loss_function."""

    # Canonical names — the primary keys users should use.
    CANONICAL_NAMES = {
        "mse",
        "normalized_mse",
        "cosine",
        "smooth_l1",
        "kl_divergence",
        "jsd",
        "logit_lens_kl",
        "contrastive",
        "angular_magnitude",
        "mahalanobis_mse",
        "mahalanobis_cosine",
        "relkd_distance",
        "relkd_angle",
        "relkd_distance_angle",
    }

    # Aliases — kept for convenience, map to the same factories.
    ALIASES = {
        "kl_div": "kl_divergence",
        "mahal_mse": "mahalanobis_mse",
        "mahal_cosine": "mahalanobis_cosine",
        "relkd_da": "relkd_distance_angle",
    }

    def test_loss_registry_contains_all_canonical(self):
        """Registry contains every canonical name."""
        assert set(LOSS_REGISTRY.keys()) >= self.CANONICAL_NAMES

    def test_loss_registry_contains_all_aliases(self):
        """Registry contains every alias."""
        assert set(self.ALIASES.keys()) <= set(LOSS_REGISTRY.keys())

    def test_aliases_resolve_to_same_factory(self):
        """Each alias maps to the same factory as its canonical name."""
        for alias, canonical in self.ALIASES.items():
            assert LOSS_REGISTRY[alias] is LOSS_REGISTRY[canonical], (
                f"Alias {alias!r} does not map to the same factory as {canonical!r}"
            )

    def test_no_unexpected_keys(self):
        """Registry contains only canonical names and known aliases."""
        expected = self.CANONICAL_NAMES | set(self.ALIASES.keys())
        assert set(LOSS_REGISTRY.keys()) == expected

    def test_get_loss_function_mse(self):
        """Test get_loss_function with mse."""
        loss_fn = get_loss_function("mse")
        assert callable(loss_fn)
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        assert loss_fn(student, teacher).shape == ()

    def test_get_loss_function_cosine(self):
        """Test get_loss_function with cosine."""
        loss_fn = get_loss_function("cosine")
        student = torch.randn(4, 64)
        teacher = torch.randn(4, 64)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_get_loss_function_kl_divergence_with_kwargs(self):
        """Test get_loss_function passes kwargs through."""
        loss_fn = get_loss_function("kl_divergence", temperature=3.0)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        result = loss_fn(student, teacher)
        assert result.shape == ()

    def test_get_loss_function_unknown_raises(self):
        """Test that unknown loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function("nonexistent_loss")

    def test_get_loss_function_error_shows_available(self):
        """Test that error message includes available loss types."""
        with pytest.raises(ValueError, match="Available:"):
            get_loss_function("bad_name")

    def test_get_loss_function_importable_from_top_level(self):
        """Test that get_loss_function can be imported from silverspoon_kd."""
        from silverspoon_kd import get_loss_function as top_level_get_loss

        assert top_level_get_loss is get_loss_function
        loss_fn = top_level_get_loss("mse")
        assert callable(loss_fn)
