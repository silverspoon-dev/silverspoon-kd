# Getting Started

This guide walks through the core workflow of distilling a teacher model into a student model using SilverSpoon-KD.

## 1. Load teacher and student models

Any Hugging Face model (or plain `nn.Module`) can be used as teacher or student:

```python
from transformers import AutoModelForCausalLM

teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
```

## 2. Create alignments

Alignments define which teacher modules are paired with which student modules.
The simplest way is with `create_alignments`, which accepts regex patterns to
match module names:

```python
from silverspoon_kd import create_alignments

# Match all decoder layers by regex
alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+",
)
```

When teacher and student use different module naming conventions, pass a dictionary
where each **key** is a regex matching **teacher** module names and the
corresponding **value** is a replacement pattern for the **student** module name
(supports regex backreferences like `\1`, `\2`):

```python
alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules={
        # key = teacher pattern, value = student replacement
        r"model\.layers\.(\d+)": r"model\.blocks\.\1",
    },
)
```

In this example, the teacher's `model.layers.0` is paired with the student's
`model.blocks.0`, `model.layers.1` with `model.blocks.1`, and so on — each
matched pair becomes one alignment whose outputs are compared during training.

If your modules return tuples (common for attention layers), use `output_selector_index`
to select which element to compare:

```python
alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+\.self_attn",
    output_selector_index=0,  # compare first element of tuple output
)
```

## 3. Choose a distiller

SilverSpoon-KD provides three distillation strategies. Pick the one that fits your use case:

| Distiller | Description | When to use |
|-----------|-------------|-------------|
| `BlockwiseDistiller` | Trains each layer independently with its own optimizer | Default choice for feature-based distillation |
| `HolisticDistiller` | Full forward pass through both models, single backward pass | When cross-layer interactions matter |
| `ResponseBasedDistiller` | Classic Hinton-style soft-label distillation on final logits | When you only care about matching output probabilities |

:::{tip}
For `BlockwiseDistiller`, setting `backward_per_block=True` in
`TrainingArguments` runs the backward pass after each block instead of
accumulating all block losses first. This can reduce peak memory usage when
working with many aligned blocks.
:::

## 4. Configure training

`TrainingArguments` extends the Hugging Face `TrainingArguments` with a few
distillation-specific options:

```python
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    logging_steps=50,
)
```

## 5. Train

```python
from silverspoon_kd import BlockwiseDistiller

distiller = BlockwiseDistiller(
    teacher_model=teacher,
    alignments=alignments,
    train_dataset=train_dataset,
    args=args,
)
distiller.train()
```

## Response-based distillation

`ResponseBasedDistiller` implements Hinton-style soft-label distillation. Unlike the
feature-based distillers above, it compares final output logits directly — no
alignments are needed. Pass a `soft_loss_fn` to control how soft targets are
compared:

```python
from silverspoon_kd import ResponseBasedDistiller, TrainingArguments
from silverspoon_kd.losses import kl_divergence_loss

args = TrainingArguments(
    output_dir="./output",
    alpha=0.5,  # balance between soft and hard loss
)
distiller = ResponseBasedDistiller(
    student_model=student,
    teacher_model=teacher,
    soft_loss_fn=kl_divergence_loss(temperature=4.0, chunk_size=1024),
    train_dataset=train_dataset,
    args=args,
)
distiller.train()
```

Key parameters:

- **`soft_loss_fn`** — Any callable `(student_logits, teacher_logits) → scalar`,
  or a string from the loss registry (e.g. `"kl_div"`, `"jsd"`). Defaults to
  `kl_divergence_loss()`. Loss hyperparameters like `temperature` and `chunk_size`
  are configured on the loss function itself.
- **`alpha`** (in training args) — Weight for combining soft and hard losses.
  `0.0` = only soft loss, `1.0` = only hard loss. Default: `0.0`.
- **Hard loss** — When `alpha > 0`, the hard loss comes from the model's own
  `outputs.loss` (the standard HuggingFace convention).

:::{tip}
`soft_loss_fn` also accepts a string name from the loss registry, so
`soft_loss_fn="kl_divergence"` works as a shorthand. When using a string,
pass loss hyperparameters via `soft_loss_fn_kwargs`
(e.g., `soft_loss_fn_kwargs={"temperature": 4.0}`).
:::

:::{note}
When `alpha > 0`, ensure your dataset includes a `labels` column and that
your model computes loss when labels are provided.
:::

## 6. Load trained weights

After training, load the distilled weights back into the student model:

```python
from silverspoon_kd import load_student_weights_from_checkpoint

load_student_weights_from_checkpoint(
    student_model=student,
    checkpoint_dir="./output/checkpoint-1000",
    student_model_name="Qwen/Qwen3-4B",
)

student.save_pretrained("./distilled-student")
```

If the student was trained with projectors (because of dimension mismatches),
use `load_student_with_projectors_from_checkpoint` instead -- it will
reconstruct the projector layers automatically.

## Per-alignment loss weighting

Each alignment has a `loss_weight` that controls how much it contributes to the
total loss. This is useful for emphasizing later layers or a final LM-head
alignment in holistic distillation:

```python
from silverspoon_kd import TrainingArguments

# Give later layers more weight
for i, alignment in enumerate(alignments):
    alignment.loss_weight = (i + 1) / len(alignments)

# Enable magnitude-aware normalization so gradient contributions
# match the weight ratios regardless of absolute loss magnitudes
args = TrainingArguments(
    output_dir="./output",
    magnitude_aware_weighting=True,
    # further training arguments as needed
)
```

For `ResponseBasedDistiller`, `magnitude_aware_weighting` normalizes the soft
and hard loss components before applying `alpha` weights:

```python
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    alpha=0.5,
    magnitude_aware_weighting=True,
    # further training arguments as needed
)
```

## Custom loss functions

By default, alignments use MSE loss. You can substitute any loss from the
registry or provide your own:

```python
from silverspoon_kd import get_loss_function

cosine_loss = get_loss_function("cosine")

alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+",
    loss_function=cosine_loss,
)
```

Available built-in losses: `mse`, `normalized_mse`, `cosine`, `smooth_l1`,
`kl_divergence` (alias `kl_div`), `jsd`, `logit_lens_kl`, `contrastive`,
`angular_magnitude`, `mahalanobis_mse` (alias `mahal_mse`),
`mahalanobis_cosine` (alias `mahal_cosine`), `relkd_distance`, `relkd_angle`,
`relkd_distance_angle` (alias `relkd_da`).

:::{tip}
You can also pass loss names as strings directly to `create_alignments`
via the `loss_function` parameter (e.g., `loss_function="cosine"`), which avoids
the need to call `get_loss_function` separately. Loss-specific
hyperparameters can be passed alongside via `loss_function_kwargs`:

```python
alignments = create_alignments(
    teacher_model=teacher,
    student_model=student,
    modules=r"model\.layers\.\d+",
    loss_function="kl_divergence",
    loss_function_kwargs={"temperature": 3.0},
)
```
:::

## Multi-GPU training

When working with large models, you may need to distribute the teacher and/or
student across multiple GPUs. SilverSpoon-KD provides flexible options for this:

- **Teacher placement** — shard the teacher across all ranks to save memory,
  or dedicate specific GPUs to the teacher with pipeline, tensor, or FSDP
  parallelism.
- **Student distribution** — use DDP, FSDP, or DeepSpeed via standard
  HuggingFace Trainer arguments.
- **Forward overlap** — when teacher and student are on separate devices,
  their forward passes can run in parallel for a significant speedup.

```python
from silverspoon_kd import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    teacher_placement="sharded",  # FSDP-shard the teacher across all ranks
    fsdp="full_shard",  # also shard the student
)
```

See the [Distributed Training](distributed.md) guide for detailed configuration
options, split-GPU setups, and full examples.
