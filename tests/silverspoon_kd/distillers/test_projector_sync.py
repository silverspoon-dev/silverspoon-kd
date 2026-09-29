"""Projectors are kept identical across data-parallel ranks.

Runs two CPU processes on the gloo backend; the distillers' synchronisation
helpers are exercised directly.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from silverspoon_kd import HolisticDistiller, TrainingArguments
from silverspoon_kd.alignments import Alignment
from silverspoon_kd.alignments.projectors import GenericLinearProjector
from tests.silverspoon_kd.conftest import DummyDataset, SimpleModel


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _make_distiller(rank, tmpdir):
    torch.manual_seed(100 + rank)  # rank-specific initialisation on purpose
    teacher = SimpleModel(input_dim=64, hidden_dim=128, num_layers=2)
    student = SimpleModel(input_dim=64, hidden_dim=64, num_layers=2)
    explicit = GenericLinearProjector(64, 128, mode="output")
    alignments = [
        Alignment(
            teacher_block=teacher.get_layer(0),
            student_block=student.get_layer(0),
            output_projector=explicit,
        ),
        Alignment(
            teacher_block=teacher.get_layer(1),
            student_block=student.get_layer(1),
            auto_projector=True,
        ),
    ]
    args = TrainingArguments(
        output_dir=os.path.join(tmpdir, f"rank{rank}"),
        max_steps=1,
        per_device_train_batch_size=2,
        report_to=[],
        logging_steps=999,
        save_strategy="no",
        use_cpu=True,
        local_rank=rank,
    )
    return HolisticDistiller(
        student_model=student,
        teacher_model=teacher,
        alignments=alignments,
        args=args,
        train_dataset=DummyDataset(num_samples=8, seq_len=16),
    )


def _worker(rank, world_size, port, tmpdir):
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    completed = False
    try:
        distiller = _make_distiller(rank, tmpdir)
        explicit = distiller.alignments[0].output_projector
        assert explicit is not None

        # Explicit projectors start rank-specific and are broadcast from rank 0.
        before = explicit.weight.detach().clone()
        distiller._sync_projectors_from_rank0()
        rank0_weight = before.clone()
        dist.broadcast(rank0_weight, src=0)
        assert torch.equal(explicit.weight, rank0_weight)
        if rank == 1:
            assert not torch.equal(before, rank0_weight)

        # Gradients are averaged across ranks.
        explicit.weight.grad = torch.full_like(explicit.weight, float(rank + 1))
        distiller._all_reduce_projector_gradients()
        assert torch.allclose(explicit.weight.grad, torch.full_like(explicit.weight, 1.5))

        # A training step materialises the auto-projector and leaves every
        # projector identical on both ranks.
        distiller._register_capture()
        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
        distiller.training_step(distiller.model, batch)
        auto = distiller.alignments[1].output_projector
        assert auto is not None
        for proj in (explicit, auto):
            for tensor in proj.parameters():
                reference = tensor.detach().clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(tensor.detach(), reference)
                assert tensor.grad is not None
                grad_reference = tensor.grad.clone()
                dist.broadcast(grad_reference, src=0)
                assert torch.allclose(tensor.grad, grad_reference)
        completed = True
    finally:
        if completed:
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not available")
def test_projectors_are_synchronised_across_ranks(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
