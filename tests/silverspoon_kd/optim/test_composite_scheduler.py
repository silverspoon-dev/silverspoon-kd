"""Tests for CompositeScheduler."""

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from silverspoon_kd.optim import CompositeScheduler


def _make_child(lr=0.1, step_size=2, gamma=0.5):
    """Create a simple model + optimizer + scheduler triple."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Do a dummy optimizer step so scheduler.step() won't warn about ordering
    optimizer.zero_grad()
    optimizer.step()
    return model, optimizer, scheduler


class TestCompositeScheduler:
    def test_requires_at_least_one_child(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeScheduler({})

    def test_step_delegates_to_all_active(self):
        _, _opt_a, sched_a = _make_child(lr=0.1)
        _, _opt_b, sched_b = _make_child(lr=0.2)
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        # Step composite (steps both children)
        composite.step()

        # Both should have advanced
        assert sched_a.last_epoch == 1
        assert sched_b.last_epoch == 1

    def test_step_delegates_to_active_only(self):
        _, _opt_a, sched_a = _make_child()
        _, _opt_b, sched_b = _make_child()
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        composite.set_active({"a"})
        composite.step()

        assert sched_a.last_epoch == 1
        assert sched_b.last_epoch == 0

    def test_get_last_lr(self):
        _, _opt_a, sched_a = _make_child(lr=0.1)
        _, _opt_b, sched_b = _make_child(lr=0.2)
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        composite.step()
        lrs = composite.get_last_lr()
        assert len(lrs) == 2
        assert lrs[0] == pytest.approx(0.1)
        assert lrs[1] == pytest.approx(0.2)

    def test_get_last_lr_active_only(self):
        _, _opt_a, sched_a = _make_child(lr=0.1)
        _, _opt_b, sched_b = _make_child(lr=0.2)
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        composite.step()  # Step both to populate get_last_lr
        composite.set_active({"a"})
        lrs = composite.get_last_lr()
        assert len(lrs) == 1
        assert lrs[0] == pytest.approx(0.1)

    def test_state_dict_round_trip(self):
        _, _, sched_a = _make_child()
        _, _, sched_b = _make_child()
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        # Advance a couple steps
        composite.step()
        composite.step()

        state = composite.state_dict()
        assert "_composite_version" in state
        assert "children" in state

        # Create new composite and load state
        _, _, sched_a2 = _make_child()
        _, _, sched_b2 = _make_child()
        composite2 = CompositeScheduler({"a": sched_a2, "b": sched_b2})
        composite2.load_state_dict(state)

        # Verify state was loaded
        assert sched_a2.last_epoch == sched_a.last_epoch
        assert sched_b2.last_epoch == sched_b.last_epoch

    def test_set_active_unknown_name_raises(self):
        _, _, sched_a = _make_child()
        composite = CompositeScheduler({"a": sched_a})

        with pytest.raises(ValueError, match="Unknown"):
            composite.set_active({"nonexistent"})

    def test_set_active_none_activates_all(self):
        _, _, sched_a = _make_child()
        _, _, sched_b = _make_child()
        composite = CompositeScheduler({"a": sched_a, "b": sched_b})

        composite.set_active({"a"})
        composite.set_active(None)

        active = list(composite._iter_active())
        assert len(active) == 2

    def test_repr(self):
        _, _, sched_a = _make_child()
        composite = CompositeScheduler({"a": sched_a})
        r = repr(composite)
        assert "CompositeScheduler" in r
        assert "StepLR" in r

    def test_load_state_dict_rejects_unversioned_dict(self):
        """A state dict without the composite version marker is rejected."""
        _, _, sched_a = _make_child()
        composite = CompositeScheduler({"a": sched_a})

        with pytest.raises(ValueError, match="_composite_version"):
            composite.load_state_dict({"a": sched_a.state_dict()})


class TestCompositeSchedulerBehavioral:
    """Behavioral tests verifying numerical correctness of CompositeScheduler."""

    def test_lr_decreases_over_steps(self):
        """LinearLR from 1.0 to 0.1: each step should decrease the LR."""
        from torch.optim.lr_scheduler import LinearLR

        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        optimizer.step()
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
        composite = CompositeScheduler({"a": scheduler})

        prev_lr = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            optimizer.step()
            composite.step()
            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr < prev_lr, f"LR did not decrease: {current_lr} >= {prev_lr}"
            prev_lr = current_lr

    def test_state_dict_preserves_position(self):
        """StepLR: save after 3 steps, load into fresh, verify last_epoch and _last_lr match."""
        _, _opt_a, sched_a = _make_child(lr=0.1, step_size=2, gamma=0.5)
        composite = CompositeScheduler({"a": sched_a})

        # Step 3 times
        composite.step()
        composite.step()
        composite.step()

        state = composite.state_dict()
        orig_sched_state = sched_a.state_dict()

        # Create fresh and load
        _, _opt_a2, sched_a2 = _make_child(lr=0.1, step_size=2, gamma=0.5)
        composite2 = CompositeScheduler({"a": sched_a2})
        composite2.load_state_dict(state)

        loaded_sched_state = sched_a2.state_dict()
        assert loaded_sched_state["last_epoch"] == orig_sched_state["last_epoch"]
        assert loaded_sched_state["_last_lr"] == pytest.approx(orig_sched_state["_last_lr"])
