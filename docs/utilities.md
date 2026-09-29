# Utilities

SilverSpoon-KD provides several utility functions for model preparation, parameter
management, and checkpoint loading. All utilities are available from the top-level
package import.

## reconfig_model

Creates a smaller student model by modifying the teacher's configuration and
optionally copying matching weights.

```python
from silverspoon_kd import reconfig_model

student = reconfig_model(
    model=teacher,
    name_or_path="my-student",
    diff={
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1024,
    },
    copy_matching_weights=True,
    freeze_copied_weights=True,
)
```

**Parameters:**

- **`model`** — Source `PreTrainedModel` to derive the student from.
- **`name_or_path`** — The `name_or_path` assigned to the new model's config.
- **`diff`** — Dictionary of config attributes to change (e.g., `{"num_hidden_layers": 12}`).
  Any attribute on the model's config can be overridden.
- **`copy_matching_weights`** — If `True`, copies weights from the source model for
  all parameters where shapes match. Useful when changing some dimensions (e.g.,
  `num_attention_heads`) but keeping others (e.g., `hidden_size`), so embeddings,
  layer norms, and output heads are initialized from the teacher.
- **`freeze_copied_weights`** — If `True` (and `copy_matching_weights=True`), freezes
  all parameters that were copied. Only randomly-initialized parameters (those with
  shape mismatches) remain trainable.

## prune_model

Performs structured pruning using importance-based weight selection. Unlike
`reconfig_model` (which creates a new model from scratch), `prune_model` retains
the most important weights from the original model based on L2 magnitude.

Requires the `torch-pruning` package (`pip install torch-pruning`).

```python
from silverspoon_kd import prune_model

student = prune_model(
    model=teacher,
    name_or_path="my-pruned-student",
    diff={
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1024,
    },
    example_inputs={"input_ids": sample_input_ids},
    output_transform=lambda x: x.logits,
    freeze_copied_weights=True,
)
```

**Parameters:**

- **`model`** — Source `PreTrainedModel` to prune.
- **`name_or_path`** — The `name_or_path` assigned to the pruned model's config.
- **`diff`** — Dictionary of config attributes defining the target dimensions.
- **`example_inputs`** — Required. Sample inputs for `DependencyGraph` tracing.
  Must be passable to `model.forward()`.
- **`output_transform`** — Transform applied to model output for tracing
  (e.g., `lambda x: x.logits`).
- **`ignored_layers`** — Modules to exclude from pruning. Auto-detected if `None`
  (embeddings, LM heads).
- **`num_heads`** — Dict mapping attention projection modules to their head count.
  Auto-detected if `None`.
- **`out_channel_groups`** — Dict mapping fused modules (e.g., `gate_up_proj`) to
  their group count. Auto-detected if `None`.
- **`forward_fn`** — Custom forward function for `DependencyGraph` tracing.
- **`round_to`** — Round channel counts down to nearest multiple of this value,
  useful for GPU-aligned dimensions (e.g., `round_to=8`).
- **`freeze_copied_weights`** — If `True`, freezes parameters whose shapes were not
  changed by pruning. Structurally pruned parameters remain trainable.

**Two-pass algorithm:**

1. **Attention head pruning** — Q/K/V projections are pruned directly using known
   coupling rules (Q↔O, K↔V) to avoid dependency graph cycles caused by reshape
   operations in multi-head attention.
2. **Width pruning** — Non-attention modules (MLP layers, etc.) are pruned via
   Torch-Pruning's `DependencyGraph`, which handles architecture-agnostic coupling
   propagation.

## freeze_parameters

Freezes model parameters whose names match any of the provided regex patterns.

```python
from silverspoon_kd import freeze_parameters

# Freeze all embedding and layer norm parameters
freeze_parameters(model, [r"embed", r"layernorm", r"ln_"])

# Freeze specific layers and unfreeze everything else
freeze_parameters(
    model,
    [r"model\.layers\.[0-5]\."],
    thaw_not_matched=True,
)
```

**Parameters:**

- **`model`** — The model whose parameters to freeze.
- **`freeze_regex_patterns`** — List of regex strings. If a parameter name matches
  any pattern, it is set to `requires_grad=False`.
- **`thaw_not_matched`** — If `True`, parameters not matching any pattern are
  explicitly set to `requires_grad=True`.

## Distiller factory

The `Distiller` factory function returns the appropriate distiller instance
based on a string name:

```python
from silverspoon_kd import Distiller

distiller = Distiller(
    distiller_type="blockwise",
    teacher_model=teacher,
    alignments=alignments,
    train_dataset=train_dataset,
    args=args,
)
```

**Valid `distiller_type` values:**

| Name | Alias | Class |
|------|-------|-------|
| `"blockwise"` | `"bkd"` | `BlockwiseDistiller` |
| `"holistic"` | `"hkd"` | `HolisticDistiller` |
| `"response_based"` | `"reskd"` | `ResponseBasedDistiller` |

## fuse_projectors_into_module

After training with projectors (due to teacher/student dimension mismatches), you
can fuse the projectors into the module weights to eliminate runtime overhead during
inference:

```python
from silverspoon_kd import fuse_projectors_into_module

fused_module = fuse_projectors_into_module(
    module=student_layer,
    input_projector=input_proj,
    output_projector=output_proj,
)
```

Supports `nn.Linear` with `GenericLinearProjector` and `nn.Conv2d` with
`GenericConv2dProjector` (1x1 convolutions only).

## Checkpoint utilities

After training, load distilled weights back into the student model:

### load_student_weights_from_checkpoint

Loads the trained student weights from a distillation checkpoint into a student
model. A `BlockwiseDistiller` checkpoint holds only the trained blocks, and each
one is loaded into its module of the student (modules that were not aligned are
left untouched); a `HolisticDistiller` or `ResponseBasedDistiller` checkpoint
holds the full student state dict. Projectors are not loaded:

```python
from silverspoon_kd import load_student_weights_from_checkpoint

load_student_weights_from_checkpoint(
    student_model=student,
    checkpoint_dir="./output/checkpoint-1000",
    student_model_name="Qwen/Qwen3-4B",
)
student.save_pretrained("./distilled-student")
```

**Parameters:**

- **`student_model`** — The student to load weights into.
- **`checkpoint_dir`** — A checkpoint directory written by a distiller, e.g.
  `./output/checkpoint-1000`, containing `model.safetensors` (or
  `pytorch_model.bin`).
- **`student_model_name`** — The student label recorded on the alignments during
  training. `create_alignments` uses the student's `name_or_path`, so it must
  match exactly.
- **`strict`** — If `True` (default), missing or unexpected keys raise a
  `ValueError`; if `False`, they are only reported in the return value.

Returns a dict with `missing_keys` and `unexpected_keys`.

### load_student_with_projectors_from_checkpoint

Loads student blocks **and** their projector layers from a `BlockwiseDistiller`
checkpoint. Use this when the student was trained **with** projectors (dimension
mismatches between teacher and student). It takes the teacher and a student with
the trained architecture, works out which modules were aligned by inspecting the
checkpoint, and replaces each aligned teacher module in-place with
`[input projector, student block, output projector]`:

```python
from transformers import AutoModelForCausalLM
from silverspoon_kd import (
    load_student_with_projectors_from_checkpoint,
    reconfig_model,
)

# The teacher used during training, and a student with the same architecture
# as the one that was trained (here: a narrower model derived from the teacher).
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
student = reconfig_model(
    teacher,
    name_or_path="my-student",
    diff={"hidden_size": 512, "intermediate_size": 1536},
)

model_with_projectors = load_student_with_projectors_from_checkpoint(
    teacher_model=teacher,
    checkpoint_dir="./output/checkpoint-1000",
    student_model_name="my-student",
    student_model=student,
)
```

**Parameters:**

- **`teacher_model`** — The teacher model. It is modified in-place and returned.
- **`checkpoint_dir`** — A `BlockwiseDistiller` checkpoint directory, containing
  `model.safetensors` (or `pytorch_model.bin`) and, when projectors were trained,
  `projector_state.pt`.
- **`student_model_name`** — The student label recorded on the alignments during
  training. `create_alignments` uses the student's `name_or_path`, so it must
  match exactly.
- **`student_model`** — A student with the trained architecture and the same
  module structure as the teacher; its blocks provide the modules into which the
  checkpoint weights are loaded.
- **`device`** — Optional device for the loaded modules (defaults to the
  teacher's device).

The returned model can be used directly for inference, or its projectors can be
fused into the student blocks with `fuse_projectors_into_module`.
