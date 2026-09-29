"""Tests for CompositeOptimizer."""

import logging

import pytest
import torch
from torch import nn

from silverspoon_kd.optim import CompositeOptimizer


def _make_child(in_features=4, out_features=4, lr=0.01):
    """Create a simple model + optimizer pair."""
    model = nn.Linear(in_features, out_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


class TestCompositeOptimizer:
    def test_requires_at_least_one_child(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeOptimizer({})

    def test_param_groups_aggregated(self):
        _, opt_a = _make_child(lr=0.01)
        _, opt_b = _make_child(lr=0.02)
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        assert len(composite.param_groups) == len(opt_a.param_groups) + len(opt_b.param_groups)
        assert composite.param_groups[0]["_composite_child"] == "a"
        assert composite.param_groups[-1]["_composite_child"] == "b"

    def test_step_delegates_to_all_active(self):
        model_a, opt_a = _make_child()
        model_b, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        # Create gradients
        loss_a = model_a(torch.randn(2, 4)).sum()
        loss_b = model_b(torch.randn(2, 4)).sum()
        loss_a.backward()
        loss_b.backward()

        params_a_before = model_a.weight.data.clone()
        params_b_before = model_b.weight.data.clone()

        composite.step()

        # Both should have updated
        assert not torch.equal(model_a.weight.data, params_a_before)
        assert not torch.equal(model_b.weight.data, params_b_before)

    def test_step_delegates_to_active_only(self):
        model_a, opt_a = _make_child()
        model_b, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        # Create gradients for both
        loss_a = model_a(torch.randn(2, 4)).sum()
        loss_b = model_b(torch.randn(2, 4)).sum()
        loss_a.backward()
        loss_b.backward()

        params_a_before = model_a.weight.data.clone()
        params_b_before = model_b.weight.data.clone()

        # Only activate "a"
        composite.set_active({"a"})
        composite.step()

        assert not torch.equal(model_a.weight.data, params_a_before)
        assert torch.equal(model_b.weight.data, params_b_before)

    def test_zero_grad_delegates_to_all(self):
        model_a, opt_a = _make_child()
        model_b, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        # Create gradients
        model_a(torch.randn(2, 4)).sum().backward()
        model_b(torch.randn(2, 4)).sum().backward()

        assert model_a.weight.grad is not None
        assert model_b.weight.grad is not None

        # Even with only "a" active, zero_grad clears all
        composite.set_active({"a"})
        composite.zero_grad()

        assert model_a.weight.grad is None or model_a.weight.grad.abs().sum() == 0
        assert model_b.weight.grad is None or model_b.weight.grad.abs().sum() == 0

    def test_state_dict_round_trip(self):
        model_a, opt_a = _make_child()
        model_b, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        # Do a step to populate optimizer state
        model_a(torch.randn(2, 4)).sum().backward()
        model_b(torch.randn(2, 4)).sum().backward()
        composite.step()

        state = composite.state_dict()
        assert "_composite_version" in state
        assert "children" in state
        assert "a" in state["children"]
        assert "b" in state["children"]

        # Create new composite and load state
        _, opt_a2 = _make_child()
        _, opt_b2 = _make_child()
        composite2 = CompositeOptimizer({"a": opt_a2, "b": opt_b2})
        composite2.load_state_dict(state)

        # Verify state was loaded
        state2 = composite2.state_dict()
        for name in ["a", "b"]:
            assert state["children"][name].keys() == state2["children"][name].keys()

    def test_set_active_unknown_name_raises(self):
        _, opt_a = _make_child()
        composite = CompositeOptimizer({"a": opt_a})

        with pytest.raises(ValueError, match="Unknown"):
            composite.set_active({"nonexistent"})

    def test_set_active_none_activates_all(self):
        _model_a, opt_a = _make_child()
        _model_b, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        # Restrict, then unrestrict
        composite.set_active({"a"})
        composite.set_active(None)

        # Both should be active
        active = list(composite._iter_active())
        assert len(active) == 2

    def test_repr(self):
        _, opt_a = _make_child()
        composite = CompositeOptimizer({"a": opt_a})
        r = repr(composite)
        assert "CompositeOptimizer" in r
        assert "SGD" in r

    def test_is_overflow_property(self):
        _, opt_a = _make_child()
        composite = CompositeOptimizer({"a": opt_a})
        assert composite.is_overflow is False

    def test_load_state_dict_rejects_unversioned_dict(self):
        """A state dict without the composite version marker is rejected."""
        _, opt_a = _make_child()
        composite = CompositeOptimizer({"a": opt_a})

        with pytest.raises(ValueError, match="_composite_version"):
            composite.load_state_dict({"a": opt_a.state_dict()})


class TestCompositeOptimizerBehavioral:
    """Behavioral tests verifying numerical correctness of CompositeOptimizer."""

    def test_sgd_update_numerically_correct(self):
        """SGD lr=0.1, no momentum: w_new == w_old - 0.1 * grad."""
        model, opt = _make_child(lr=0.1)
        composite = CompositeOptimizer({"a": opt})

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        w_old = model.weight.data.clone()
        grad = model.weight.grad.clone()
        composite.step()

        expected = w_old - 0.1 * grad
        assert torch.allclose(model.weight.data, expected, atol=1e-7)

    def test_state_dict_value_roundtrip(self):
        """SGD with momentum: save/load state dict preserves actual tensor values."""
        model, _ = _make_child()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        composite = CompositeOptimizer({"a": opt})

        # Step to populate momentum state
        model(torch.randn(2, 4)).sum().backward()
        composite.step()

        state = composite.state_dict()

        # Create new composite and load state
        model2, _ = _make_child()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
        composite2 = CompositeOptimizer({"a": opt2})
        composite2.load_state_dict(state)

        # Compare actual tensor values in optimizer state
        orig_state = opt.state_dict()["state"]
        loaded_state = opt2.state_dict()["state"]
        for param_idx in orig_state:
            for key, val in orig_state[param_idx].items():
                if isinstance(val, torch.Tensor):
                    assert torch.allclose(val, loaded_state[param_idx][key], atol=1e-7)

    def test_zero_grad_set_to_none_propagates(self):
        """set_to_none=True -> grad is None; set_to_none=False -> grad is zero."""
        model, opt = _make_child()
        composite = CompositeOptimizer({"a": opt})

        # Create gradients
        model(torch.randn(2, 4)).sum().backward()
        assert model.weight.grad is not None

        # set_to_none=True
        composite.zero_grad(set_to_none=True)
        assert model.weight.grad is None

        # Create gradients again
        model(torch.randn(2, 4)).sum().backward()
        assert model.weight.grad is not None

        # set_to_none=False
        composite.zero_grad(set_to_none=False)
        assert model.weight.grad is not None
        assert model.weight.grad.abs().sum() == 0


class TestCompositeLoadStateDictWarnings:
    """Regression tests: load_state_dict must warn about mismatched children."""

    def test_warns_on_extra_checkpoint_children(self, caplog):
        """Checkpoint with children not in composite logs a warning."""
        _, opt = _make_child()
        composite = CompositeOptimizer({"a": opt})

        state = {
            "_composite_version": 1,
            "children": {
                "a": opt.state_dict(),
                "b_removed": opt.state_dict(),
            },
        }
        with caplog.at_level(logging.WARNING):
            composite.load_state_dict(state)
        assert any("unknown children" in msg.lower() for msg in caplog.messages)

    def test_warns_on_missing_checkpoint_children(self, caplog):
        """Composite with children missing from checkpoint logs a warning."""
        _, opt_a = _make_child()
        _, opt_b = _make_child()
        composite = CompositeOptimizer({"a": opt_a, "b": opt_b})

        state = {
            "_composite_version": 1,
            "children": {
                "a": opt_a.state_dict(),
                # "b" missing
            },
        }
        with caplog.at_level(logging.WARNING):
            composite.load_state_dict(state)
        assert any("no saved state" in msg.lower() for msg in caplog.messages)

    def test_no_warning_on_matching_children(self, caplog):
        """Matching children produce no warnings."""
        _, opt = _make_child()
        composite = CompositeOptimizer({"a": opt})
        state = composite.state_dict()

        with caplog.at_level(logging.WARNING):
            composite.load_state_dict(state)
        assert not any(
            "unknown children" in msg.lower() or "no saved state" in msg.lower()
            for msg in caplog.messages
        )
