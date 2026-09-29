# SilverSpoon-KD

**A flexible, modular knowledge distillation library for PyTorch.**

SilverSpoon-KD provides multiple distillation strategies for training smaller (student) models to match the representations of larger (teacher) models. It works with any `nn.Module` — from transformers and CNNs to custom architectures. Built on top of PyTorch and Hugging Face Transformers, it integrates directly with the HF `Trainer` API.

## Features

- **Multiple distillation strategies** -- holistic, blockwise, and response-based distillation
- **Flexible alignment system** -- pair arbitrary teacher and student modules with per-pair loss functions, optimizers, and projectors
- **Automatic projectors** -- shape mismatches between teacher and student are detected and resolved automatically
- **HF Trainer integration** -- all distillers extend `Trainer`, so logging, checkpointing, and mixed precision work out of the box
- **Multi-GPU & distributed training** -- flexible [teacher placement](distributed.md) (replicated, sharded, or dedicated GPUs), overlapped forward passes, and DDP/FSDP/DeepSpeed support for the student
- **Built-in loss functions** -- MSE, cosine similarity, smooth L1, KL divergence, JSD, contrastive, Mahalanobis, RelKD, and more
- **Profiling and metrics** -- FLOP counting, per-layer loss tracking, and optional WeightWatcher integration

## Installation

```bash
pip install silverspoon-kd
```

Or install from source with all dependencies:

```bash
git clone https://github.com/silverspoon-dev/silverspoon-kd.git
cd silverspoon-kd
pip install -e ".[all]"
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, Transformers >= 5.0 (< 6.0)

## Quick example

```python
from transformers import AutoModelForCausalLM
from silverspoon_kd import (
    BlockwiseDistiller,
    TrainingArguments,
    create_alignments,
)

teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+",
)

args = TrainingArguments(output_dir="./output", num_train_epochs=3)

distiller = BlockwiseDistiller(
    teacher_model=teacher,
    alignments=alignments,
    train_dataset=train_dataset,
    args=args,
)
distiller.train()
```

## Next steps

- [Getting Started](getting-started.md) -- a more detailed walkthrough of the core workflow
- [Concepts](concepts.md) -- how distillation strategies, alignments, and projectors work
- [Utilities](utilities.md) -- model preparation, pruning, parameter freezing, and checkpoint loading
- [API Reference](api/index.md) -- full reference for all classes and functions

```{toctree}
:maxdepth: 2
:hidden:

getting-started
concepts
distributed
utilities
api/index
```