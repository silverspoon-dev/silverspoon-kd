"""Shared helpers for distributed training tests."""

import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port():
    """Find a free TCP port by briefly binding to port 0.

    Sets SO_REUSEADDR so the OS allows immediate re-bind of the port by the
    mp.spawn children, reducing EADDRINUSE races when many distributed tests
    run in parallel under pytest-xdist.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


_requires_4_gpus = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 4 if torch.cuda.is_available() else True,
        reason="At least 4 GPUs required for distributed training tests",
    ),
]


def _try_import_deepspeed():
    try:
        import deepspeed  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _init_process(rank, world_size, port, fn, *args):
    """Initialize distributed process group and run fn."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # A rank that raises must exit without destroy_process_group(): that call
    # blocks on the other rank, which is then killed only by the NCCL timeout,
    # and the real traceback is lost. Exiting promptly lets mp.spawn stop the
    # other rank and report this one's error.
    completed = False
    try:
        fn(rank, world_size, *args)
        completed = True
    finally:
        if completed:
            dist.destroy_process_group()


def _init_process_ds(rank, world_size, port, fn, *args):
    """Initialize distributed for DeepSpeed workers.

    Sets ACCELERATE_USE_DEEPSPEED so the HF Trainer's accelerate integration
    recognizes the DeepSpeed backend. nccl is init'd first so that teacher
    FSDP (which uses torch.distributed) can be set up before the Trainer.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # A rank that raises must exit without destroy_process_group(): that call
    # blocks on the other rank, which is then killed only by the NCCL timeout,
    # and the real traceback is lost. Exiting promptly lets mp.spawn stop the
    # other rank and report this one's error.
    completed = False
    try:
        fn(rank, world_size, *args)
        completed = True
    finally:
        if completed:
            dist.destroy_process_group()
        os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)


def _make_fsdp_teacher(rank, world_size, teacher, wrap_cls="SimpleBlock"):
    """FSDP-shard teacher on dedicated GPUs (rank+2) and return."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardingStrategy,
    )

    from silverspoon_kd.distributed.strategies import _build_wrap_policy

    teacher_gpu = rank + 2
    torch.cuda.set_device(teacher_gpu)
    teacher.to(f"cuda:{teacher_gpu}")
    teacher_group = dist.new_group(ranks=list(range(world_size)))
    teacher = FSDP(
        teacher,
        process_group=teacher_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(f"cuda:{teacher_gpu}"),
        auto_wrap_policy=_build_wrap_policy(teacher, wrap_cls=wrap_cls),
    )
    torch.cuda.set_device(rank)
    return teacher


def _snapshot_params(model):
    """Clone all trainable params."""
    return {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}


def _infer_paradigm(distiller):
    """Infer paradigm id from the distiller's class name."""
    cls_name = type(distiller).__name__
    if "Holistic" in cls_name:
        return "holistic"
    if "Blockwise" in cls_name:
        return "blockwise"
    if "ResponseBased" in cls_name:
        return "response_based"
    return None


def _verify_training(
    rank,
    distiller,
    student,
    student_params_before,
    expected_steps=10,
    expected_teacher_device=None,
    check_cross_rank=False,
    world_size=2,
    paradigm=None,
):
    """Standard post-training assertions — called by every worker.

    Paradigm-specific checks are applied automatically based on the
    distiller's class (or via the explicit ``paradigm`` argument).  These
    catch paradigm-specific regressions — e.g. HKD must update embeddings,
    BKD must keep non-aligned params frozen — that a loose "any param
    changed" check would miss.
    """
    if paradigm is None:
        paradigm = _infer_paradigm(distiller)
    # 1. Step count
    assert distiller.state.global_step == expected_steps, (
        f"Rank {rank}: expected {expected_steps} steps, got {distiller.state.global_step}"
    )

    # 2. Extract losses
    losses = [e["loss"] for e in distiller.state.log_history if "loss" in e]
    assert len(losses) > 0, f"Rank {rank}: no losses logged"

    # 3. All losses finite and positive
    for i, v in enumerate(losses):
        assert torch.isfinite(torch.tensor(v)), f"Rank {rank}: step {i} loss={v} not finite"
        assert v > 0, f"Rank {rank}: step {i} loss={v} not positive"

    # 4. Loss decreasing (first half avg > second half avg)
    if len(losses) >= 4:
        mid = len(losses) // 2
        early_avg = sum(losses[:mid]) / mid
        late_avg = sum(losses[mid:]) / (len(losses) - mid)
        assert late_avg < early_avg, (
            f"Rank {rank}: loss not decreasing. "
            f"Early avg={early_avg:.6f}, late avg={late_avg:.6f}. "
            f"Losses: {[f'{v:.6f}' for v in losses]}"
        )

    # 5. Student params changed — paradigm-specific checks where applicable.
    #    The generic "any param changed" check is too weak to catch the class
    #    of failures where some params silently never update (e.g. embedding
    #    parameters that receive gradients but are never stepped).  We
    #    therefore also assert the paradigm's expected update pattern for
    #    non-aligned parameters.
    changed_names = {
        n
        for n, p in student.named_parameters()
        if p.requires_grad
        and n in student_params_before
        and not torch.equal(p.data, student_params_before[n])
    }
    assert changed_names, f"Rank {rank}: student params unchanged after training"

    if paradigm == "holistic":
        # HKD uses a single whole-student optimizer.  Parameters upstream of
        # the alignments (the embeddings feeding the encoder) must receive
        # updates.  If this fails, the optimizer is scoped to the aligned
        # blocks alone and leaves non-aligned params un-updated even though
        # they receive gradients.
        embedding_updated = any("embedding" in n for n in changed_names)
        assert embedding_updated, (
            f"Rank {rank}: HKD did not update any embedding parameter. "
            f"Changed params: {sorted(changed_names)[:5]}... "
            "Embeddings and other non-aligned params receive gradients but "
            "are never stepped by any optimizer."
        )
    elif paradigm == "blockwise":
        # BKD runs each block in isolation on teacher-captured inputs via
        # StudentBlocksContainer.  The full student model's embeddings and
        # lm_head are outside the container and must NOT be updated.
        for n in changed_names:
            assert "embedding" not in n and "lm_head" not in n, (
                f"Rank {rank}: BKD unexpectedly updated {n}.  BKD should "
                "train only the aligned student blocks, not the embeddings "
                "or LM head."
            )
    elif paradigm == "response_based":
        # ResKD uses HF Trainer's default optimizer over the full student.
        # Both embeddings (upstream) and lm_head (directly upstream of the
        # loss) must update.
        assert any("embedding" in n for n in changed_names), (
            f"Rank {rank}: ResKD did not update any embedding parameter."
        )
        assert any("lm_head" in n for n in changed_names), (
            f"Rank {rank}: ResKD did not update the LM head."
        )

    # 6. Teacher still in eval mode
    assert not distiller.teacher_model.training, f"Rank {rank}: teacher should be in eval mode"

    # 7. Teacher device
    if expected_teacher_device is not None:
        dev = next(distiller.teacher_model.parameters()).device
        assert dev == expected_teacher_device, (
            f"Rank {rank}: teacher on {dev}, expected {expected_teacher_device}"
        )

    # 8. Cross-rank loss consistency
    if check_cross_rank:
        final_loss = torch.tensor(losses[-1], device=f"cuda:{rank}")
        all_losses = [torch.zeros_like(final_loss) for _ in range(world_size)]
        dist.all_gather(all_losses, final_loss)
        for r, other in enumerate(all_losses):
            ratio = final_loss / other
            assert 0.5 < ratio.item() < 2.0, (
                f"Rank {rank}: loss diverged from rank {r}. "
                f"self={final_loss.item():.4f}, other={other.item():.4f}"
            )


def _make_holistic_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


def _make_blockwise_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


def _make_response_args(tmpdir, rank, **extra):
    from silverspoon_kd.training_arguments import TrainingArguments

    defaults = {
        "output_dir": tmpdir,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "logging_steps": 1,
        "save_steps": 999,
        "learning_rate": 5e-3,
        "dataloader_num_workers": 0,
        "report_to": [],
        "remove_unused_columns": False,
        "local_rank": rank,
        "alpha": 0.0,
        "auto_device_match": True,
        "auto_dtype_match": True,
    }
    defaults.update(extra)
    return TrainingArguments(**defaults)


def _make_teacher_student(rank, teacher_to_device=None):
    """Create teacher + student. Teacher on CPU unless specified."""
    from tests.silverspoon_kd.conftest import SimpleModel

    teacher = SimpleModel(64, 128, 3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if teacher_to_device is not None:
        teacher.to(teacher_to_device)

    student = SimpleModel(64, 64, 3)
    student.to(f"cuda:{rank}")
    return teacher, student


def _make_alignments(teacher, student):
    from silverspoon_kd import create_alignments

    alignments = create_alignments(
        teacher_model=teacher,
        student_model=student,
        modules=r"layers\.\d+$",
        output_selector_index=None,
        auto_projector=True,
    )
    for a in alignments:
        a.auto_device_match = True
    return alignments


def _dataset(n=40):
    from tests.silverspoon_kd.conftest import DummyDataset

    return DummyDataset(num_samples=n, seq_len=16)


def _write_ds_config(tmpdir, rank):
    """Write a minimal DeepSpeed ZeRO-2 config file and return path.

    No ``optimizer`` block: HF Trainer passes a ``torch.optim.AdamW`` instance
    (and, for HKD, our ``_build_deepspeed_compatible_optimizer`` flat optimizer
    with per-alignment param_groups) to DeepSpeed as a client optimizer. This
    matches typical production usage and removes a hidden toolchain dependency
    (DeepSpeed's ``FusedAdam`` JIT-builds a CUDA extension that requires GCC
    >= 9, which not all clusters provide).
    """
    config = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 2},
    }
    path = os.path.join(tmpdir, f"ds_config_{rank}.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


def _make_explicit_projector_alignments(teacher, student):
    """Create alignments with explicit input/output projectors.

    Mirrors test_multi_gpu._make_alignments: output projector when dims
    mismatch, input projector for layers > 0 (where teacher output dim
    feeds into student input dim).
    """
    import torch.nn as nn

    from silverspoon_kd.alignments import Alignment
    from silverspoon_kd.alignments.projectors import GenericLinearProjector

    teacher_dim = teacher.hidden_dim
    alignments = []
    for i in range(min(teacher.num_layers, student.num_layers)):
        teacher_block = teacher.get_layer(i)
        student_block = student.get_layer(i)
        student_device = next(student_block.parameters()).device

        # Discover student output dim
        student_dim = None
        for mod in student_block.modules():
            if isinstance(mod, nn.Linear):
                student_dim = mod.out_features
                break

        output_projector = None
        if student_dim is not None and student_dim != teacher_dim:
            output_projector = nn.Linear(student_dim, teacher_dim).to(student_device)

        input_projector = None
        if i > 0 and student_dim is not None and student_dim != teacher_dim:
            input_projector = GenericLinearProjector(
                in_features=teacher_dim,
                out_features=student_dim,
                mode="input",
                apply_to_arg=0,
            ).to(student_device)

        alignment = Alignment(
            teacher_block=teacher_block,
            student_block=student_block,
            teacher_model_name="test_teacher",
            student_model_name="test_student",
            teacher_module_name=f"layers.{i}",
            student_module_name=f"layers.{i}",
            input_projector=input_projector,
            output_projector=output_projector,
            auto_device_match=True,
        )
        alignments.append(alignment)
    return alignments


def _spawn_with_retry(init_fn, nprocs, fn, tmp_path, max_retries=3):
    """Spawn distributed workers, retrying on port collisions (EADDRINUSE)."""
    for attempt in range(max_retries):
        port = _find_free_port()
        try:
            mp.spawn(
                init_fn,
                args=(nprocs, port, fn, str(tmp_path)),
                nprocs=nprocs,
                join=True,
            )
            return
        except Exception as exc:
            if "EADDRINUSE" in str(exc) and attempt < max_retries - 1:
                continue
            raise


def _spawn(fn, tmp_path):
    _spawn_with_retry(_init_process, 2, fn, tmp_path)


def _spawn_ds(fn, tmp_path):
    _spawn_with_retry(_init_process_ds, 2, fn, tmp_path)


_requires_deepspeed = pytest.mark.skipif(
    not _try_import_deepspeed(),
    reason="DeepSpeed not installed",
)


def _make_pp_teacher(rank, world_size, teacher):
    """Place teacher via PP on GPUs 2,3 (both ranks get own copy)."""
    from silverspoon_kd.distributed.strategies import place_teacher_pp

    teacher = place_teacher_pp(teacher, [2, 3], "cuda")
    return teacher


def _make_tp_teacher(rank, world_size, teacher):
    """TP-shard teacher on GPUs 2,3 and return."""
    from silverspoon_kd.distributed.strategies import (
        install_tp_device_hooks,
        parallelize_teacher_tp,
    )

    remapped = [rank + 2 for _ in range(1)]  # one teacher GPU per rank
    # TP requires len(remapped) == world_size
    remapped = [2, 3]
    teacher = parallelize_teacher_tp(teacher, remapped, "cuda")
    install_tp_device_hooks(teacher)
    return teacher


def _spawn_n(fn, tmp_path, nprocs):
    _spawn_with_retry(_init_process, nprocs, fn, tmp_path)


def _verify_eval(rank, metrics):
    """Verify evaluate() returns valid metrics."""
    assert isinstance(metrics, dict), f"Rank {rank}: evaluate() returned {type(metrics)}"
    assert len(metrics) > 0, f"Rank {rank}: evaluate() returned empty metrics"
    eval_loss = metrics.get("eval_loss")
    assert eval_loss is not None, f"Rank {rank}: no eval_loss in {metrics.keys()}"
    assert torch.isfinite(torch.tensor(eval_loss)), f"Rank {rank}: eval_loss={eval_loss}"
    assert eval_loss > 0, f"Rank {rank}: eval_loss={eval_loss} not positive"


def _write_ds3_config(tmpdir, rank):
    """Write a DeepSpeed ZeRO-3 config file and return path.

    See ``_write_ds_config`` for why we omit the ``optimizer`` block.
    """
    config = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 0,
            "stage3_param_persistence_threshold": 0,
        },
    }
    path = os.path.join(tmpdir, f"ds3_config_{rank}.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


def _assert_deepspeed_compatible_optimizer(rank, distiller, alignments):
    """Verify ``_build_deepspeed_compatible_optimizer`` actually got installed.

    Without this check, DeepSpeed could silently ignore our optimizer (e.g.
    if a future config grew an ``"optimizer"`` block again) and the test
    would still pass while the per-alignment LR tracking and grad-clipping
    code paths were dead. We assert the live optimizer wraps a single flat
    optimizer whose ``param_groups`` carry the per-alignment ``_alignment_name``
    tags that ``_clip_student_gradients`` and ``_track_student_learning_rates``
    rely on.
    """
    optimizer = distiller.optimizer
    # DeepSpeed wraps the client optimizer; the underlying torch optimizer is
    # exposed via ``.optimizer`` (DS engine attr) or directly when no wrap.
    base = getattr(optimizer, "optimizer", optimizer)
    param_groups = getattr(base, "param_groups", None)
    assert param_groups, f"Rank {rank}: optimizer has no param_groups"

    tagged_names = {g.get("_alignment_name") for g in param_groups if g.get("_alignment_name")}
    expected = {a.get_name() for a in alignments}
    assert tagged_names == expected, (
        f"Rank {rank}: expected param_groups tagged with {expected}, "
        f"got {tagged_names} (full groups: {param_groups})"
    )
