<p align="center">
  <img src="https://raw.githubusercontent.com/silverspoon-dev/silverspoon-kd/main/docs/assets/favicon.png" width="120" alt="SilverSpoon-KD">
</p>
<h1 align="center">SilverSpoon-KD</h1>
<p align="center">A General-Purpose Toolkit for Knowledge Distillation</p>
<p align="center">
  <a href="https://pypi.org/project/silverspoon-kd/"><img src="https://img.shields.io/pypi/v/silverspoon-kd.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/silverspoon-kd/"><img src="https://img.shields.io/pypi/pyversions/silverspoon-kd.svg" alt="Python versions"></a>
  <a href="https://github.com/silverspoon-dev/silverspoon-kd/actions/workflows/ci.yml"><img src="https://github.com/silverspoon-dev/silverspoon-kd/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://kd.silverspoon.dev"><img src="https://img.shields.io/badge/docs-kd.silverspoon.dev-3f51b5.svg" alt="Documentation"></a>
  <a href="https://github.com/silverspoon-dev/silverspoon-kd/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

## Overview

SilverSpoon-KD is a library designed to simplify the process of knowledge distillation from a "teacher" model to a "student" model. It provides a variety of distillation strategies and powerful tools to align intermediate representations between the two models, without requiring modifications to the original model code.

## Key Features

*   **Multiple Distillation Strategies**:
    *   `ResponseBasedDistiller` (ResKD): A specialized distiller for matching logits, often using KL-divergence.
    *   `HolisticDistiller` (HKD): Distills knowledge using full forward passes through both models with cross-layer gradient flow.
    *   `BlockwiseDistiller` (BKD): Distills knowledge block-by-block for more granular supervision.
*   **Flexible Alignments**: Easily define alignments between arbitrary teacher and student layers to match intermediate features.
*   **Automatic Projectors**: Automatically inserts and trains projector layers (e.g., `nn.Linear`, `nn.Conv2d`) to match dimensions when teacher and student features are misaligned.
*   **Variety of Loss Functions**: Includes losses relevant to distillation such as MSE, Cosine Similarity, KL Divergence, JSD, Contrastive, Mahalanobis, RelKD, and more.
*   **Per-Alignment Loss Weighting**: Assign different weights to each alignment layer and optionally enable magnitude-aware normalization so gradient contributions match the specified weight ratios regardless of absolute loss magnitudes.
*   **Multi-GPU & Distributed Training**: Flexible teacher placement (replicated, FSDP-sharded, or dedicated GPUs with pipeline/tensor parallelism), overlapped teacher–student forward passes, and full support for DDP, FSDP, and DeepSpeed on the student side.
*   **Dynamic Hooking Engine**: Uses a powerful, non-intrusive engine to capture module inputs/outputs and to replace modules on-the-fly.

## Installation

```bash
pip install silverspoon-kd
```

> **Note:** SilverSpoon-KD is pre-1.0. The public API may change between minor versions; see the [changelog](https://github.com/silverspoon-dev/silverspoon-kd/blob/main/CHANGELOG.md) for details before upgrading.

## Quick Start

Here is an example of how to use SilverSpoon-KD for feature-based blockwise distillation.

```python
import torch
from transformers import AutoModelForCausalLM
from silverspoon_kd import (
    BlockwiseDistiller,
    TrainingArguments,
    create_alignments,
)
from your_data import get_train_dataset

# 1. Load teacher and student models
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

# 2. Create alignments with a regex pattern and a string loss name
#    This matches all "model.layers.<N>" modules in both teacher and student.
alignments = create_alignments(
    teacher_model=teacher_model,
    student_model=student_model,
    modules=r"model\.layers\.\d+",
    loss_function="mse",  # or "cosine", "kl_divergence", "jsd", "contrastive", etc.
    output_selector_index=0,  # extract the first element from tuple outputs
)

# 3. Configure training and start distillation
distiller = BlockwiseDistiller(
    teacher_model=teacher_model,
    alignments=alignments,
    train_dataset=get_train_dataset(),
    args=TrainingArguments(
        output_dir="./distillation_output",
        num_train_epochs=3,
        learning_rate=5e-5,
    ),
)

distiller.train()
```

> **Tip:** For config-driven workflows where the distiller type comes from a file or CLI argument, you can use the `Distiller` factory function instead — see [Utilities](#utilities).

### Deriving a Student from the Teacher with `reconfig_model`

If you want to create a smaller student with the same architecture, you can use `reconfig_model` to derive one from the teacher and `create_alignments` to automatically create alignments using regex patterns.

```python
import torch
from transformers import AutoModelForCausalLM
from silverspoon_kd import (
    BlockwiseDistiller,
    TrainingArguments,
    reconfig_model,
    create_alignments,
)
from your_data import get_train_dataset

# 1. Load the teacher model
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# 2. Create a smaller student by reconfiguring the teacher
#    This derives a new model with the same architecture but fewer attention heads
#    and a smaller intermediate size. Matching weights (e.g., embeddings) are copied over.
student_model = reconfig_model(
    teacher_model,
    name_or_path="my-student",
    diff={"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 1024},
    copy_matching_weights=True,
    freeze_copied_weights=True,
)

# 3. Create alignments with a regex pattern
#    This matches all "model.layers.<N>" modules in both teacher and student.
alignments = create_alignments(
    teacher_model=teacher_model,
    student_model=student_model,
    modules=r"model\.layers\.\d+",
    output_selector_index=0,  # extract the first element from tuple outputs
)

# 4. Configure training and start distillation
training_args = TrainingArguments(
    output_dir="./distillation_output",
    num_train_epochs=3,
    learning_rate=5e-5,
)

distiller = BlockwiseDistiller(
    teacher_model=teacher_model,
    alignments=alignments,
    train_dataset=get_train_dataset(),
    args=training_args,
)

distiller.train()
```

### Response-Based Distillation

For classic Hinton-style soft-label distillation on final logits, use `ResponseBasedDistiller`. Unlike the feature-based distillers above, it takes `student_model` and `teacher_model` directly — no alignments needed.

```python
from transformers import AutoModelForCausalLM
from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
from silverspoon_kd.losses import kl_divergence_loss
from your_data import get_train_dataset

teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")

distiller = ResponseBasedDistiller(
    student_model=student_model,
    teacher_model=teacher_model,
    soft_loss_fn=kl_divergence_loss(temperature=4.0),
    train_dataset=get_train_dataset(),
    args=TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        alpha=0.5,  # 50% soft (KL-div) loss, 50% hard (model's own CE) loss
    ),
)

distiller.train()
```

## Distillation Strategies

See [Concepts](https://kd.silverspoon.dev/concepts.html) for architecture diagrams and detailed explanations of alignments, projectors, and distillation strategies.

### Response-Based Knowledge Distillation (ResKD)
`ResponseBasedDistiller` is a specialized distiller for matching the teacher's and student's output logits. Unlike the feature-based distillers, it requires no alignments — just a teacher, a student, and a soft loss function (e.g. `kl_divergence_loss`, `jsd_loss`, or any custom callable).

### Holistic Knowledge Distillation (HKD)
`HolisticDistiller` runs full forward passes through both models and performs a single backward pass, allowing gradients to flow across layers. This is the most straightforward feature-based approach and works well when teacher and student architectures are similar.

### Blockwise Knowledge Distillation (BKD)
`BlockwiseDistiller` allows for a more fine-grained distillation process. The student is split into blocks that are trained in isolation: each student block receives the input captured at the corresponding teacher block and is trained, with its own optimizer, to reproduce that teacher block's output — an independent regression problem per block. Gradients never cross block boundaries, so every block gets direct supervision from the teacher, and `backward_per_block=True` in `TrainingArguments` keeps peak memory at a single block's worth.

## Loss Functions

All built-in losses follow the signature `(student_output, teacher_output) -> scalar`:

| Name | Aliases | Description |
|------|---------|-------------|
| `mse` | | Mean squared error (default) |
| `normalized_mse` | | Z-score normalized MSE (matches relational structure, ignores scale) |
| `cosine` | | Cosine similarity loss (1 - cos(s, t)) |
| `smooth_l1` | | Smooth L1 / Huber loss |
| `kl_divergence` | `kl_div` | KL divergence (with temperature scaling and optional chunking) |
| `jsd` | | Jensen-Shannon Divergence (with temperature and interpolation weight) |
| `logit_lens_kl` | | Project through LM head, then KL divergence on token distributions |
| `contrastive` | | InfoNCE-based contrastive distillation loss |
| `angular_magnitude` | | Decomposed angular (cosine) + magnitude (norm) loss |
| `mahalanobis_mse` | `mahal_mse` | MSE under a learned Mahalanobis metric |
| `mahalanobis_cosine` | `mahal_cosine` | Cosine loss under a learned Mahalanobis metric |
| `relkd_distance` | | RelKD distance-wise distillation |
| `relkd_angle` | | RelKD angle-wise distillation |
| `relkd_distance_angle` | `relkd_da` | Combined RelKD distance + angle loss |

Pass a loss name as a string to `create_alignments`, or use `get_loss_function` for more control:

```python
from silverspoon_kd import get_loss_function

loss_fn = get_loss_function("kl_div", temperature=2.0)
```

## Utilities

SilverSpoon-KD includes several utility functions for working with models:

- **`reconfig_model`** — Create a smaller student from a teacher by modifying config attributes (e.g., `num_attention_heads`, `intermediate_size`). Optionally copies and freezes matching weights.
- **`prune_model`** — Structured pruning using importance-based weight selection (requires `torch-pruning`). Automatically detects attention head structure and propagates pruning through coupled layers.
- **`freeze_parameters`** — Freeze parameters by regex pattern. Use `thaw_not_matched=True` to unfreeze everything else.
- **`Distiller`** — Factory function to create distillers by string name (e.g., `Distiller(distiller_type="blockwise", ...)`). Also accepts aliases: `"bkd"`, `"hkd"`, `"reskd"`.

```python
from silverspoon_kd import prune_model

student = prune_model(
    model=teacher_model,
    name_or_path="my-pruned-student",
    diff={"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 1024},
    example_inputs={"input_ids": sample_input_ids},
    output_transform=lambda x: x.logits,
)
```

## Documentation

Full documentation is hosted at [kd.silverspoon.dev](https://kd.silverspoon.dev/), including detailed guides on [Getting Started](https://kd.silverspoon.dev/getting-started.html), [Concepts](https://kd.silverspoon.dev/concepts.html), [Distributed Training](https://kd.silverspoon.dev/distributed.html), [Utilities](https://kd.silverspoon.dev/utilities.html), and the [API Reference](https://kd.silverspoon.dev/api/index.html).

## License

This project is licensed under the Apache License 2.0. Please see the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](https://github.com/silverspoon-dev/silverspoon-kd/blob/main/CONTRIBUTING.md) for the development setup and workflow, then open an issue or pull request.

## Citation

If you use SilverSpoon-KD in your research, please cite it (see also [CITATION.cff](https://github.com/silverspoon-dev/silverspoon-kd/blob/main/CITATION.cff)):

```bibtex
@software{silverspoon_kd,
  author  = {Davey, Xaver R.},
  title   = {SilverSpoon-KD: A General-Purpose Toolkit for Knowledge Distillation},
  year    = {2026},
  url     = {https://github.com/silverspoon-dev/silverspoon-kd},
  version = {0.1.0},
  license = {Apache-2.0},
}
```
