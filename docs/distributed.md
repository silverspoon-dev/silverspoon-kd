# Distributed Training

SilverSpoon-KD supports multi-GPU training for both student and teacher models.
This guide covers the available strategies, how to configure them, and how they
compose with each other.

## Overview

There are two independent axes of distribution:

1. **Teacher placement** — how the (frozen) teacher model is distributed across
   GPUs. Controlled by `TrainingArguments.teacher_placement`.
2. **Student distribution** — how the trainable student model is distributed.
   Uses HuggingFace Trainer's standard `fsdp` and `deepspeed` arguments.

These compose freely: you can shard the teacher with FSDP while training the
student with DeepSpeed, or pipeline-parallel the teacher across dedicated GPUs
while the student uses DDP on the remaining GPUs.

## Teacher placement

The `teacher_placement` parameter on `TrainingArguments` controls where and how
the teacher model lives during training. All distiller types support all
placement strategies.

### Replicated (default)

Each rank holds a full copy of the teacher. This is the simplest option and
requires no extra configuration:

```python
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    teacher_placement=None,  # or "replicated" — same thing
)
```

### Sharded (all-ranks FSDP)

FSDP-shards the teacher across all ranks, reducing per-GPU memory:

```python
args = TrainingArguments(
    output_dir="./output",
    teacher_placement="sharded",
)
```

The teacher is wrapped with `FULL_SHARD`. Wrapping granularity uses the
model's `_no_split_modules` attribute if available, otherwise falls back to a
size-based policy.

### Dedicated teacher GPUs (split-GPU)

For larger teachers, you can reserve specific GPUs exclusively for the teacher
and let the student use the remaining GPUs. This requires two things:

1. A `TeacherPlacement` object specifying which GPUs and which strategy.
2. A call to `setup_split_gpu()` **before** importing `torch`.

```python
# train.py — split-GPU setup
from silverspoon_kd.distributed import TeacherPlacement, setup_split_gpu

placement = TeacherPlacement(
    teacher_only_devices=[0, 1],  # physical GPU IDs (nvidia-smi)
    strategy="pp",  # "pp", "tp", or "sharded"
)
setup_split_gpu(placement)

# Now import torch and everything else
import torch
from silverspoon_kd import HolisticDistiller, TrainingArguments, create_alignments
# ...

args = TrainingArguments(
    output_dir="./output",
    teacher_placement=placement,
)
```

`setup_split_gpu` reorders `CUDA_VISIBLE_DEVICES` so that student GPUs get the
low CUDA indices (which Trainer/DDP/FSDP use via `LOCAL_RANK`) and teacher GPUs
follow:

```text
4-GPU system, teacher_only_devices=[0, 1]
→ CUDA_VISIBLE_DEVICES=2,3,0,1
    cuda:0 → physical GPU 2  (student)
    cuda:1 → physical GPU 3  (student)
    cuda:2 → physical GPU 0  (teacher)
    cuda:3 → physical GPU 1  (teacher)
```

Launch with `torchrun`, setting `--nproc_per_node` to the number of **student**
GPUs:

```bash
torchrun --nproc_per_node=2 train.py
```

#### Split-GPU strategies

| Strategy | Description |
|----------|-------------|
| `"pp"` | **Pipeline parallel.** Distributes teacher layers across dedicated GPUs using `device_map`. Tensors move between devices automatically via pre-forward hooks. Default strategy. |
| `"tp"` | **Tensor parallel.** Shards teacher weights across GPUs using PyTorch's DTensor. Requires the model to expose a `_tp_plan` attribute. |
| `"sharded"` | **FSDP.** Full-shards the teacher across dedicated GPUs only (separate process group from the student). |

For the sharded strategy, you can optionally specify `wrap_cls` to control FSDP
wrapping granularity:

```python
placement = TeacherPlacement(
    teacher_only_devices=[0, 1],
    strategy="sharded",
    wrap_cls="LlamaDecoderLayer",  # or a list: ["Block", "Layer"]
)
```

#### Passing placement as a dict

`teacher_placement` also accepts a plain dictionary, which is auto-converted to
a `TeacherPlacement`:

```python
args = TrainingArguments(
    output_dir="./output",
    teacher_placement={
        "teacher_only_devices": [0, 1],
        "strategy": "pp",
    },
)
```

## Student distribution

Student distribution uses HuggingFace Trainer's standard arguments. All
distiller types (Blockwise, Holistic, ResponseBased) support DDP, FSDP, and
DeepSpeed.

### DDP

DDP is the default when launching with `torchrun` on multiple GPUs. No extra
configuration is needed — Trainer handles wrapping and gradient synchronization
automatically.

```bash
torchrun --nproc_per_node=4 train.py
```

### FSDP

Enable FSDP via Trainer's `fsdp` argument:

```python
args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard",
    fsdp_config={
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
    },
)
```

SilverSpoon-KD handles the interaction between FSDP and per-alignment optimizers
automatically: optimizer creation is deferred until after FSDP wrapping
completes, and block references are synced to the wrapped modules.

### DeepSpeed

Pass a DeepSpeed config via `deepspeed`:

```python
args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",
)
```

Under DeepSpeed, SilverSpoon-KD builds a single flat optimizer (required by
ZeRO) with per-alignment parameter groups tagged internally, so per-alignment
learning rates and gradient clipping still work.

## Forward overlap

When the teacher and student reside on different physical GPUs, their forward
passes can run in parallel on separate CUDA streams. This is controlled by
`TrainingArguments.overlap_teacher_forward`:

```python
args = TrainingArguments(
    output_dir="./output",
    teacher_placement=placement,
    overlap_teacher_forward=True,
)
```

| Value | Behavior |
|-------|----------|
| `None` (default) | Auto-detect: enabled when teacher and student are on different CUDA devices |
| `True` | Force enable |
| `False` | Force disable |

The speedup depends on the distiller type:

- **Holistic / ResponseBased**: Teacher and student forward passes run fully in
  parallel. Typical speedup of 26--40% with a split-GPU teacher.
- **Blockwise**: Pipelined overlap — the teacher forward is queued on a
  separate stream and per-block CUDA events synchronize each student block with
  its corresponding teacher block. Typical speedup of ~10%.

## Combining strategies

Teacher placement and student distribution are independent and can be combined
freely. Some common configurations:

### Teacher sharded + Student FSDP

Both teacher and student sharded across all ranks — maximizes memory efficiency:

```python
args = TrainingArguments(
    output_dir="./output",
    teacher_placement="sharded",
    fsdp="full_shard",
)
```

### Split-GPU teacher + Student DDP + Forward overlap

Teacher on dedicated GPUs, student DDP on the rest, with overlapped forward
passes:

```python
from silverspoon_kd.distributed import TeacherPlacement, setup_split_gpu

placement = TeacherPlacement(teacher_only_devices=[0, 1], strategy="pp")
setup_split_gpu(placement)

import torch
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    teacher_placement=placement,
    overlap_teacher_forward=True,
)
```

```bash
torchrun --nproc_per_node=2 train.py  # 2 student GPUs
```

### Split-GPU teacher + Student FSDP

```python
placement = TeacherPlacement(teacher_only_devices=[0], strategy="pp")
setup_split_gpu(placement)

import torch
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    teacher_placement=placement,
    fsdp="full_shard",
    overlap_teacher_forward=True,
)
```

```bash
torchrun --nproc_per_node=3 train.py  # 3 student GPUs on a 4-GPU node
```

## Full example

A complete training script using split-GPU pipeline-parallel teacher placement
with student FSDP and forward overlap:

```python
# train.py
from silverspoon_kd.distributed import TeacherPlacement, setup_split_gpu

# 1. Reserve GPUs 0-1 for the teacher (must happen before torch import)
placement = TeacherPlacement(teacher_only_devices=[0, 1], strategy="pp")
setup_split_gpu(placement)

# 2. Now import everything else
import torch
from transformers import AutoModelForCausalLM
from silverspoon_kd import (
    HolisticDistiller,
    TrainingArguments,
    create_alignments,
)
from your_data import get_train_dataset

# 3. Load models
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

# 4. Create alignments
alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+",
    loss_function="mse",
    output_selector_index=0,
)

# 5. Configure training
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    teacher_placement=placement,
    overlap_teacher_forward=True,
    fsdp="full_shard",
)

# 6. Train
distiller = HolisticDistiller(
    teacher_model=teacher,
    alignments=alignments,
    train_dataset=get_train_dataset(),
    args=args,
)
distiller.train()
```

```bash
# Launch with 2 student GPUs (GPUs 2 and 3, after split)
torchrun --nproc_per_node=2 train.py
```
