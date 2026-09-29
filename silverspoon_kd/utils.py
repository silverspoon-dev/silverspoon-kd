"""Model reconfiguration, freezing, and structural pruning utilities."""

import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PreTrainedModel

# Optional dependency (see alignments/utils.py for the pattern).
if TYPE_CHECKING:
    import torch_pruning as tp

try:
    import torch_pruning as tp

    TORCH_PRUNING_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_PRUNING_AVAILABLE = False

logger = logging.getLogger(__name__)


def partial_summarize_layer_names(keys: list[str]) -> list[str]:
    """
    Groups a list of layer names into summarized patterns.
    Example: ['layers.0.attn.q_proj', 'layers.1.attn.q_proj'] -> ['layers.[0-1].attn.q_proj']
    """
    # Use a dictionary to group layer numbers by their surrounding prefix/suffix
    patterns = defaultdict(list)
    # This list will hold keys that don't match the layer pattern (e.g., 'model.norm.weight')
    unmatched_keys = []

    # Regex to find patterns like '...prefix.<number>.suffix...'
    # It captures three groups: (prefix, number, suffix)
    layer_pattern = re.compile(r"^(.*\.)(\d+)(\..*)$")

    for key in sorted(keys):
        match = layer_pattern.match(key)
        if match:
            prefix, number, suffix = match.groups()
            # Group numbers by their (prefix, suffix) tuple
            patterns[(prefix, suffix)].append(int(number))
        else:
            unmatched_keys.append(key)

    # --- Process the grouped patterns into summary strings ---
    summary_lines = []
    for (prefix, suffix), numbers in patterns.items():
        if not numbers:  # pragma: no cover
            continue

        # Create compact range string like "[0-15]" or "[0-2, 5, 8-10]"
        numbers.sort()
        ranges = []
        start = end = numbers[0]
        for num in numbers[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = num
        ranges.append(f"{start}-{end}" if start != end else str(start))

        range_str = f"[{','.join(ranges)}]"
        summary_lines.append(f"{prefix}{range_str}{suffix}")

    return sorted(summary_lines) + sorted(unmatched_keys)


def summarize_layer_names(keys: list[str]) -> list[str]:
    """
    Recursively summarizes layer names into compact patterns.
    Example: ['model.layers.0.attn.q_proj', 'model.layers.1.attn.q_proj']
    -> ['model.layers.[0-1].attn.q_proj']
    """
    summarized = partial_summarize_layer_names(list(keys))
    next_summarized = partial_summarize_layer_names(summarized)
    while len(next_summarized) < len(summarized):
        summarized = next_summarized
        next_summarized = partial_summarize_layer_names(summarized)
    summarized.sort()
    return summarized


def freeze_parameters(
    model: torch.nn.Module, freeze_regex_patterns: list[str], thaw_not_matched=False
):
    """
    Freezes parameters whose names match any of the provided regex patterns.
    Parameters not matching any pattern will be set to trainable (requires_grad=True).

    Args:
        model (torch.nn.Module): The model whose parameters are to be frozen.
        freeze_regex_patterns (List[str]): A list of regex strings. If a parameter's
                                           name matches any of these patterns, it will
                                           be set to requires_grad=False.
        thaw_not_matched (bool): If True, parameters not matching any pattern will be unfrozen.
    """
    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in freeze_regex_patterns]
    for name, param in model.named_parameters():
        should_freeze = any(pattern.search(name) for pattern in compiled_patterns)
        if should_freeze:
            param.requires_grad = False
        elif thaw_not_matched:
            param.requires_grad = True


def reconfig_model(
    model: PreTrainedModel,
    name_or_path: str,
    diff: dict | None = None,
    copy_matching_weights: bool = False,
    freeze_copied_weights: bool = False,
) -> nn.Module:
    """
    Reconfigures a model's configuration based on a provided dictionary of differences.
    Uses the same model class as the input model to ensure compatibility.

    Args:
        model: The source model to reconfigure from.
        name_or_path: The name_or_path to assign to the new model's config.
        diff: Dictionary of config attributes to change (e.g., {"num_hidden_layers": 12}).
        copy_matching_weights: If True, copies weights from the source model to the new model
                              for all parameters where shapes match. This is useful when changing
                              some dimensions (e.g., num_attention_heads, intermediate_size) but
                              keeping others (e.g., hidden_size), allowing embeddings, layer norms,
                              and output heads to be initialized from the source model.
        freeze_copied_weights: If True and copy_matching_weights is True, freezes (sets
                              requires_grad=False) all parameters that were copied from the
                              source model. This is ignored if copy_matching_weights is False.

    Returns:
        A new model with the modified configuration.
    """
    if diff is None:
        diff = {}
    if not diff:
        logger.warning("reconfig_model called with empty diff; output will be identical to source")
    config = deepcopy(model.config)
    config.name_or_path = name_or_path
    for key, value in diff.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.debug("Set config.%s to %s", key, value)
        else:
            logger.warning("Config has no attribute '%s'", key)
    # Use the same model class as the input model
    model_class = type(model)
    new_model = model_class(config)

    if copy_matching_weights:
        source_state = model.state_dict()
        target_state = new_model.state_dict()

        copied = []
        shape_mismatch = []

        for param_name, param_value in source_state.items():
            if param_name not in target_state:
                continue  # Parameter doesn't exist in new model (e.g., removed layers)

            if param_value.shape == target_state[param_name].shape:
                target_state[param_name] = param_value.clone()
                copied.append(param_name)
            else:
                shape_mismatch.append(param_name)

        new_model.load_state_dict(target_state)

        if freeze_copied_weights and copied:
            # Freeze all parameters that were copied from the source model
            # Convert parameter names to regex patterns (escape special chars)
            freeze_patterns = [re.escape(name) + "$" for name in copied]
            freeze_parameters(new_model, freeze_patterns)
            logger.debug("Froze %d copied parameters", len(copied))

        logger.info(
            "Weight transfer: copied %d parameters, %d shape mismatches",
            len(copied),
            len(shape_mismatch),
        )
        if copied:
            summarized_copied = summarize_layer_names(copied)
            lines = summarized_copied[:10]
            if len(summarized_copied) > 10:
                lines.append(f"... and {len(summarized_copied) - 10} more")
            logger.debug("Copied parameters:\n  %s", "\n  ".join(lines))
        if shape_mismatch:
            summarized_mismatch = summarize_layer_names(shape_mismatch)
            lines = summarized_mismatch[:10]
            if len(summarized_mismatch) > 10:
                lines.append(f"... and {len(summarized_mismatch) - 10} more")
            logger.debug(
                "Shape mismatches (randomly initialized):\n  %s",
                "\n  ".join(lines),
            )

    return new_model


_IGNORED_LAYER_SUFFIXES = ("lm_head", "embed_tokens", "wte", "wpe", "embed_positions")


def _auto_detect_ignored_layers(model: nn.Module) -> list[nn.Module]:
    """Find embedding and output head modules that should not be pruned."""
    ignored = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in _IGNORED_LAYER_SUFFIXES:
            ignored.append(module)
    return ignored


def _auto_detect_head_structure(
    model: nn.Module,
    config,
) -> tuple:
    """Detect attention head structure for head-aware pruning.

    Returns (num_heads dict, out_channel_groups dict).
    """
    num_heads: dict[nn.Module, int] = {}
    out_channel_groups: dict[nn.Module, int] = {}

    for name, module in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name not in ("self_attn", "attention"):
            continue

        # Look for separate Q/K/V projections
        q_proj = getattr(module, "q_proj", None)
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        qkv_proj = getattr(module, "qkv_proj", None)

        n_heads = getattr(config, "num_attention_heads", None)
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

        if n_heads is None:
            continue

        if q_proj is not None and isinstance(q_proj, nn.Linear):
            num_heads[q_proj] = n_heads
        if k_proj is not None and isinstance(k_proj, nn.Linear):
            num_heads[k_proj] = n_kv_heads if n_kv_heads is not None else n_heads
        if v_proj is not None and isinstance(v_proj, nn.Linear):
            num_heads[v_proj] = n_kv_heads if n_kv_heads is not None else n_heads
        if qkv_proj is not None and isinstance(qkv_proj, nn.Linear):
            num_heads[qkv_proj] = n_heads

        # Detect fused MLP gate_up_proj
        gate_up_proj = getattr(module, "gate_up_proj", None)
        if gate_up_proj is not None and isinstance(gate_up_proj, nn.Linear):
            out_channel_groups[gate_up_proj] = 2

    if not num_heads:
        logger.warning(
            "Could not auto-detect attention head structure; "
            "falling back to per-channel pruning for attention layers"
        )

    return num_heads, out_channel_groups


def _find_attention_blocks(model: nn.Module) -> list[nn.Module]:
    """Find attention block modules (self_attn, attention)."""
    blocks = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in ("self_attn", "attention"):
            blocks.append(module)
    return blocks


def _round_channels(channels: int, round_to: int | None) -> int:
    """Round channel count down to nearest multiple of round_to."""
    if round_to is None or round_to <= 1:
        return channels
    rounded = channels - (channels % round_to)
    return max(round_to, rounded)


def _kept_head_indices(scores: torch.Tensor, n_heads: int, target_channels: int) -> torch.Tensor:
    """Pick the channel indices to *keep* when pruning attention heads.

    Prefers removing whole heads when ``current - target`` is divisible by
    ``head_dim``; otherwise falls back to per-channel selection.  Always
    returns sorted indices so coupled projections can be sliced consistently.
    """
    current_channels = scores.shape[0]
    head_dim = current_channels // n_heads
    n_to_prune = current_channels - target_channels

    if n_to_prune % head_dim == 0:
        head_scores = scores.reshape(n_heads, head_dim).mean(dim=1)
        heads_to_keep = n_heads - (n_to_prune // head_dim)
        keep_heads = torch.argsort(head_scores, descending=True)[:heads_to_keep]
        return (
            torch.cat(
                [torch.arange(h.item() * head_dim, (h.item() + 1) * head_dim) for h in keep_heads]
            )
            .sort()
            .values
        )
    return torch.argsort(scores, descending=True)[:target_channels].sort().values


def _slice_projection_out_channels(proj: nn.Linear, keep_idxs: torch.Tensor) -> None:
    """Slice ``proj.weight`` / ``proj.bias`` along the output dimension."""
    proj.weight = nn.Parameter(proj.weight.data[keep_idxs])
    if proj.bias is not None:
        proj.bias = nn.Parameter(proj.bias.data[keep_idxs])
    proj.out_features = keep_idxs.shape[0]


def _prune_qo_coupling(
    name: str,
    q_proj: nn.Linear,
    o_proj: nn.Linear,
    target_q: nn.Linear,
    num_heads: dict[nn.Module, int],
    round_to: int | None,
) -> bool:
    """Prune the Q↔O coupling of one attention block in place.

    Returns ``True`` if anything was pruned, ``False`` otherwise.
    """
    if target_q.out_features >= q_proj.out_features:
        return False

    target_channels = _round_channels(target_q.out_features, round_to)
    n_heads_val = num_heads.get(q_proj) or q_proj.out_features
    original_features = q_proj.out_features

    # Average importance across the coupled Q (rows) and O (cols) weights.
    q_scores = torch.linalg.vector_norm(q_proj.weight.data, ord=2, dim=1)
    o_scores = torch.linalg.vector_norm(o_proj.weight.data, ord=2, dim=0)
    keep_idxs = _kept_head_indices((q_scores + o_scores) / 2, n_heads_val, target_channels)

    _slice_projection_out_channels(q_proj, keep_idxs)
    o_proj.weight = nn.Parameter(o_proj.weight.data[:, keep_idxs])
    o_proj.in_features = target_channels

    logger.debug("Pruned %s Q↔O: %d -> %d channels", name, original_features, target_channels)
    return True


def _prune_kv_coupling(
    name: str,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    target_k: nn.Linear,
    num_heads: dict[nn.Module, int],
    round_to: int | None,
) -> bool:
    """Prune the K↔V coupling of one attention block in place.

    Returns ``True`` if anything was pruned, ``False`` otherwise.
    """
    if target_k.out_features >= k_proj.out_features:
        return False

    target_channels = _round_channels(target_k.out_features, round_to)
    n_kv_heads_val = num_heads.get(k_proj) or k_proj.out_features
    original_features = k_proj.out_features

    k_scores = torch.linalg.vector_norm(k_proj.weight.data, ord=2, dim=1)
    v_scores = torch.linalg.vector_norm(v_proj.weight.data, ord=2, dim=1)
    keep_idxs = _kept_head_indices(k_scores + v_scores, n_kv_heads_val, target_channels)

    for proj in (k_proj, v_proj):
        _slice_projection_out_channels(proj, keep_idxs)

    logger.debug("Pruned %s K↔V: %d -> %d channels", name, original_features, target_channels)
    return True


def _prune_attention_heads(
    pruning_model: nn.Module,
    target_model: nn.Module,
    num_heads: dict[nn.Module, int],
    round_to: int | None = None,
) -> int:
    """Prune attention heads by direct weight selection.

    This avoids torch-pruning's DependencyGraph for attention projections,
    which hangs due to reshape-driven BFS cycles in the attention mechanism.
    Instead, we directly select the most important heads and slice the coupled
    projection weights (Q↔O, K↔V).

    Returns number of attention blocks pruned.
    """
    target_attn_map = {
        name: module
        for name, module in target_model.named_modules()
        if name.split(".")[-1] in ("self_attn", "attention")
    }

    pruned_count = 0
    for name, attn in pruning_model.named_modules():
        if name.split(".")[-1] not in ("self_attn", "attention"):
            continue
        target_attn = target_attn_map.get(name)
        if target_attn is None:
            continue

        block_pruned = False

        q_proj = getattr(attn, "q_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        target_q = getattr(target_attn, "q_proj", None)
        if q_proj is not None and o_proj is not None and target_q is not None:
            block_pruned |= _prune_qo_coupling(name, q_proj, o_proj, target_q, num_heads, round_to)

        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        target_k = getattr(target_attn, "k_proj", None)
        if k_proj is not None and v_proj is not None and target_k is not None:
            block_pruned |= _prune_kv_coupling(name, k_proj, v_proj, target_k, num_heads, round_to)

        if block_pruned:
            pruned_count += 1

    return pruned_count


def _remove_excess_layers(model: nn.Module, target_n: int) -> bool:
    """Truncate the transformer layer ModuleList to target_n layers.

    Returns:
        True if truncation succeeded, False otherwise.
    """
    source_n = getattr(model.config, "num_hidden_layers", None)
    if source_n is None:
        logger.warning("Model config has no num_hidden_layers attribute; skipping layer truncation")
        return False

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == source_n:
            # Truncate in-place
            del module[target_n:]
            model.config.num_hidden_layers = target_n  # type: ignore[assignment]
            logger.debug("Truncated %s from %d to %d layers", name, source_n, target_n)
            return True

    logger.warning(
        "Could not find ModuleList with %d elements matching num_hidden_layers; "
        "the pruned model may have more layers than intended",
        source_n,
    )
    return False


def _validate_pruned_dimensions(
    pruned: nn.Module,
    target: nn.Module,
) -> None:
    """Compare parameter shapes between pruned and target model, log warnings."""
    mismatches = []
    pruned_params = dict(pruned.named_parameters())
    for name, target_param in target.named_parameters():
        pruned_param = pruned_params.get(name)
        if pruned_param is None:
            mismatches.append(f"{name}: missing in pruned model")
        elif pruned_param.shape != target_param.shape:
            mismatches.append(f"{name}: {list(pruned_param.shape)} vs {list(target_param.shape)}")

    if mismatches:
        logger.warning(
            "Shape mismatches after pruning (%d parameters):\n  %s",
            len(mismatches),
            "\n  ".join(mismatches[:20]),
        )
    else:
        logger.debug("All parameter shapes match target model")


def _collect_attention_linear_modules(model: nn.Module) -> set:
    """Return the set of ``nn.Linear`` modules that live inside attention blocks.

    These are skipped by the DependencyGraph pass because their reshape-driven
    coupling causes torch-pruning's BFS to hang.
    """
    attn_modules: set = set()
    for block in _find_attention_blocks(model):
        for child in block.modules():
            if isinstance(child, nn.Linear):
                attn_modules.add(child)
    return attn_modules


def _build_target_out_channel_map(
    target_model: nn.Module,
    pruning_name_to_module: dict[str, nn.Module],
    ignored_set: set,
    attn_modules: set,
    round_to: int | None,
) -> dict[nn.Linear, int]:
    """Map each prunable ``nn.Linear`` to its desired (rounded) out-channel count.

    Skips ignored layers and attention-block linears (already pruned in pass 1).
    Only modules that need to *shrink* are included.
    """
    target_out: dict[nn.Linear, int] = {}
    for name, target_module in target_model.named_modules():
        if not isinstance(target_module, nn.Linear):
            continue
        pruning_module = pruning_name_to_module.get(name)
        if pruning_module is None or not isinstance(pruning_module, nn.Linear):
            continue
        if pruning_module in ignored_set or pruning_module in attn_modules:
            continue
        if target_module.out_features < pruning_module.out_features:
            target_out[pruning_module] = _round_channels(target_module.out_features, round_to)
    return target_out


def _coupled_importance_scores(
    module: nn.Module, dep_graph, pruner, ignored_set: set
) -> torch.Tensor:
    """L2 importance scores for ``module``, averaged with its coupled deps.

    Walks the DependencyGraph for one out-channel slot and averages the
    rows-or-columns L2 norms of every coupled ``nn.Linear`` whose score
    vector has the same length.  Coupled layers that don't line up with
    the target channel axis are ignored.
    """
    scores = torch.linalg.vector_norm(module.weight.data, ord=2, dim=1)
    probe_group = dep_graph.get_pruning_group(module, pruner.prune_out_channels, [0])
    coupled_scores = [scores]
    for dep, _ in probe_group:
        dep_module = dep.target.module
        if dep_module is module or dep_module in ignored_set:
            continue
        if not isinstance(dep_module, nn.Linear):
            continue
        axis = 1 if dep_graph.is_out_channel_pruning_fn(dep.handler) else 0
        dep_s = torch.linalg.vector_norm(dep_module.weight.data, ord=2, dim=axis)
        if dep_s.shape == scores.shape:
            coupled_scores.append(dep_s)
    if len(coupled_scores) > 1:
        return torch.stack(coupled_scores).mean(dim=0)
    return scores


def _pruned_indices_head_aware(
    scores: torch.Tensor, n_heads: int, n_to_prune: int, module_label: str
) -> torch.Tensor:
    """Indices to *prune* for a head-structured module.

    Falls back to per-channel pruning when ``n_to_prune`` doesn't land on a
    head boundary, with a warning.  Always returns sorted indices.
    """
    current_channels = scores.shape[0]
    head_dim = current_channels // n_heads

    if n_to_prune % head_dim != 0:
        logger.warning(
            "n_to_prune=%d not divisible by head_dim=%d for %s; "
            "falling back to per-channel pruning",
            n_to_prune,
            head_dim,
            module_label,
        )
        return torch.argsort(scores)[:n_to_prune].sort().values

    head_scores = scores.reshape(n_heads, head_dim).mean(dim=1)
    heads_to_prune = n_to_prune // head_dim
    worst_heads = torch.argsort(head_scores)[:heads_to_prune]
    return (
        torch.cat(
            [torch.arange(h.item() * head_dim, (h.item() + 1) * head_dim) for h in worst_heads]
        )
        .sort()
        .values
    )


def _pruned_indices_group_aware(
    scores: torch.Tensor,
    n_groups: int,
    current_channels: int,
    target_channels: int,
) -> torch.Tensor:
    """Indices to *prune* for a fused-channel-group module (e.g. gate_up_proj).

    Removes the same number of bottom-scoring channels from each fused group
    so that the group structure is preserved.
    """
    group_size = current_channels // n_groups
    target_group_size = target_channels // n_groups
    prune_per_group = group_size - target_group_size
    all_idxs = []
    for g in range(n_groups):
        start = g * group_size
        group_scores = scores[start : start + group_size]
        worst = torch.argsort(group_scores)[:prune_per_group]
        all_idxs.append(worst + start)
    return torch.cat(all_idxs).sort().values


def _select_prune_indices(
    module: nn.Module,
    scores: torch.Tensor,
    current_channels: int,
    target_channels: int,
    num_heads: dict[nn.Module, int],
    out_channel_groups: dict[nn.Module, int],
    module_label: str,
) -> torch.Tensor:
    """Dispatch to head-aware, group-aware, or per-channel pruning."""
    n_to_prune = current_channels - target_channels
    if module in num_heads:
        return _pruned_indices_head_aware(scores, num_heads[module], n_to_prune, module_label)
    if module in out_channel_groups:
        return _pruned_indices_group_aware(
            scores, out_channel_groups[module], current_channels, target_channels
        )
    return torch.argsort(scores)[:n_to_prune].sort().values


def _prune_one_width_group(
    module: nn.Linear,
    target_channels: int,
    dep_graph,
    num_heads: dict[nn.Module, int],
    out_channel_groups: dict[nn.Module, int],
    ignored_set: set,
    module_label: str,
) -> set:
    """Prune one module's out-channels via DG-propagated grouping.

    Returns the set of ``id(...)`` values for sibling modules that were
    pruned along the same out-channel axis (used by the caller to skip
    re-visiting them).
    """
    current_channels = module.out_features
    pruner = dep_graph.get_pruner_of_module(module)
    scores = _coupled_importance_scores(module, dep_graph, pruner, ignored_set)
    prune_idxs = _select_prune_indices(
        module,
        scores,
        current_channels,
        target_channels,
        num_heads,
        out_channel_groups,
        module_label,
    )

    # Use DG to propagate through coupled layers (down_proj, LayerNorm, etc.)
    group = dep_graph.get_pruning_group(module, pruner.prune_out_channels, prune_idxs.tolist())

    visited_ids: set = set()
    for dep, _ in group:
        if dep_graph.is_out_channel_pruning_fn(dep.handler):
            visited_ids.add(id(dep.target.module))

    group.prune()
    logger.debug("Pruned %s: %d -> %d channels", module_label, current_channels, target_channels)
    return visited_ids


def _freeze_unchanged_after_pruning(pruning_model: nn.Module, source_model: nn.Module) -> None:
    """Freeze parameters whose shape is unchanged from the source model.

    Pruned parameters (whose shapes shrank) remain trainable so the
    distillation step can re-fit them.
    """
    source_state = source_model.state_dict()
    unchanged = []
    pruned_params = []
    for name, param in pruning_model.named_parameters():
        source_param = source_state.get(name)
        if source_param is not None and param.shape == source_param.shape:
            unchanged.append(name)
        else:
            pruned_params.append(name)

    if not unchanged:
        return

    freeze_patterns = [re.escape(name) + "$" for name in unchanged]
    freeze_parameters(pruning_model, freeze_patterns)
    logger.info(
        "Froze %d unchanged parameters, %d pruned parameters remain trainable",
        len(unchanged),
        len(pruned_params),
    )
    if pruned_params:
        summarized = summarize_layer_names(pruned_params)
        lines = summarized[:10]
        if len(summarized) > 10:
            lines.append(f"... and {len(summarized) - 10} more")
        logger.debug("Trainable (pruned) parameters:\n  %s", "\n  ".join(lines))


def prune_model(
    model: PreTrainedModel,
    name_or_path: str,
    diff: dict | None = None,
    example_inputs: dict | None = None,
    output_transform: Callable | None = None,
    ignored_layers: list[nn.Module] | None = None,
    num_heads: dict[nn.Module, int] | None = None,
    out_channel_groups: dict[nn.Module, int] | None = None,
    forward_fn: Callable | None = None,
    round_to: int | None = None,
    freeze_copied_weights: bool = False,
) -> nn.Module:
    """Prune a model to a target config using importance-based weight selection.

    Uses Torch-Pruning's DependencyGraph to discover parameter couplings
    and propagate structural pruning correctly, while selecting which
    channels to keep based on L2 weight magnitude.

    Args:
        model: Source pretrained model to prune.
        name_or_path: The name_or_path to assign to the pruned model's config.
        diff: Dictionary of config attributes to change.
        example_inputs: Required inputs for DependencyGraph tracing.
        output_transform: Transform applied to model output for tracing
            (e.g., lambda x: x.logits).
        ignored_layers: Modules to exclude from pruning. Auto-detected if None.
        num_heads: Dict mapping attention projection modules to their head count.
            Auto-detected if None.
        out_channel_groups: Dict mapping fused modules to their group count.
            Auto-detected if None.
        forward_fn: Custom forward function for DependencyGraph tracing.
        round_to: Round channel counts down to nearest multiple of this value.
            Useful for GPU-aligned dimensions (e.g., round_to=4 or round_to=8).
        freeze_copied_weights: If True, freezes (sets requires_grad=False) all
            parameters whose shapes were not changed by pruning. Parameters that
            were structurally pruned remain trainable.

    Returns:
        A pruned copy of the model with target dimensions and retained weights.
    """
    if not TORCH_PRUNING_AVAILABLE:
        raise ImportError(
            "torch-pruning is required for prune_model. Install it with: pip install torch-pruning"
        )

    if example_inputs is None:
        raise ValueError(
            "example_inputs is required for DependencyGraph tracing. "
            "Provide sample inputs that can be passed to model.forward()."
        )

    if diff is None:
        diff = {}
    if not diff:
        logger.warning("prune_model called with empty diff; output will be identical to source")

    # Create target reference model for shape comparison
    target_model = reconfig_model(model, name_or_path, diff)

    # Prepare pruning model (deep copy of source)
    try:
        pruning_model = deepcopy(model)
    except (MemoryError, RuntimeError) as e:
        raise MemoryError(
            f"Failed to deep-copy model for pruning ({type(e).__name__}: {e}). "
            f"The model may be too large to duplicate in memory. Consider "
            f"moving the model to CPU first or freeing GPU memory."
        ) from e

    # Handle depth reduction before building DependencyGraph
    target_num_layers = diff.get("num_hidden_layers")
    if target_num_layers is not None:
        source_n = getattr(model.config, "num_hidden_layers", None)
        if (
            source_n is not None
            and target_num_layers < source_n
            and not _remove_excess_layers(pruning_model, target_num_layers)
        ):
            raise RuntimeError(
                f"Failed to truncate model from {source_n} to "
                f"{target_num_layers} layers. Could not find a "
                f"ModuleList with {source_n} elements."
            )

    # Auto-detect head structure if not provided
    if num_heads is None or out_channel_groups is None:
        detected_heads, detected_groups = _auto_detect_head_structure(pruning_model, model.config)
        if num_heads is None:
            num_heads = detected_heads
        if out_channel_groups is None:
            out_channel_groups = detected_groups
    assert num_heads is not None
    assert out_channel_groups is not None

    if ignored_layers is None:
        ignored_layers = _auto_detect_ignored_layers(pruning_model)
    ignored_set = set(ignored_layers)

    # --- Pass 1: Attention head pruning (direct weight selection) ---
    # Attention projections (Q/K/V) cause DependencyGraph.get_pruning_group()
    # to hang due to reshape-driven BFS cycles in the multi-head attention
    # mechanism. We handle them separately with known coupling rules:
    # Q↔O (same head dimension), K↔V (same KV head dimension).
    attn_pruned = _prune_attention_heads(pruning_model, target_model, num_heads, round_to)

    # --- Pass 2: Width pruning via DependencyGraph ---
    # For non-attention modules (MLP up/gate/down_proj, etc.), the DG works
    # correctly and provides architecture-agnostic coupling propagation.
    attn_modules = _collect_attention_linear_modules(pruning_model)
    dep_graph = tp.DependencyGraph().build_dependency(
        pruning_model,
        example_inputs=example_inputs,
        output_transform=output_transform,  # type: ignore[arg-type]
        forward_fn=forward_fn,  # type: ignore[arg-type]
        ignored_layers=ignored_layers,
    )

    pruning_name_to_module: dict[str, nn.Module] = dict(pruning_model.named_modules())
    target_out = _build_target_out_channel_map(
        target_model, pruning_name_to_module, ignored_set, attn_modules, round_to
    )

    module_to_name = {m: n for n, m in pruning_model.named_modules()}
    visited: set = set()
    width_pruned = 0
    for module, target_channels in target_out.items():
        if id(module) in visited:
            continue
        visited |= _prune_one_width_group(
            module,
            target_channels,
            dep_graph,
            num_heads,
            out_channel_groups,
            ignored_set,
            module_to_name.get(module, "unknown"),
        )
        width_pruned += 1

    # Finalize
    total_pruned = attn_pruned + width_pruned
    if total_pruned > 0:
        logger.info(
            "Pruned %d groups to match target config (%d attention, %d width)",
            total_pruned,
            attn_pruned,
            width_pruned,
        )
    else:
        logger.info("No width pruning needed")

    pruning_model.config = deepcopy(target_model.config)  # type: ignore[assignment]
    _validate_pruned_dimensions(pruning_model, target_model)

    if freeze_copied_weights:
        _freeze_unchanged_after_pruning(pruning_model, model)

    del target_model
    return pruning_model


class _ModelModuleNameMapper:
    def __init__(self, model):
        self.model = model
        # Create the reverse mapping: {Module_Object: "Name"}
        self.module_to_name: dict[nn.Module, str] = {
            module: name for name, module in model.named_modules()
        }

    def get_name(self, module) -> str:
        """Return the qualified name of ``module`` within the model."""
        name = self.module_to_name.get(module, None)
        if name is None:
            raise ValueError("Module not found in model.")
        return name

    def get_module(self, name):
        """Return the submodule registered under the qualified ``name``."""
        return self.model.get_submodule(name)


def _estimate_num_training_steps(train_dataset, training_args) -> int:
    """Estimate the number of optimizer steps ``Trainer.train`` will run.

    Mirrors the Trainer's own computation for a dataset with a length: the
    dataset is sharded across ``world_size`` processes, a partial final
    batch is kept unless ``dataloader_drop_last`` is set, and every
    ``gradient_accumulation_steps`` batches make one optimizer step.  Used
    as the horizon of the per-alignment schedulers when ``max_steps`` is
    not set.
    """
    try:
        num_samples = len(train_dataset)
    except TypeError as exc:
        raise ValueError(
            "train_dataset has no length, so the number of training steps cannot be "
            "estimated; set max_steps in TrainingArguments."
        ) from exc
    world_size = max(int(getattr(training_args, "world_size", 1) or 1), 1)
    samples_per_batch = training_args.per_device_train_batch_size * world_size
    if getattr(training_args, "dataloader_drop_last", False):
        batches_per_epoch = num_samples // samples_per_batch
    else:
        batches_per_epoch = math.ceil(num_samples / samples_per_batch)
    accumulation = max(int(training_args.gradient_accumulation_steps), 1)
    updates_per_epoch = max(math.ceil(batches_per_epoch / accumulation), 1)
    return math.ceil(training_args.num_train_epochs * updates_per_epoch)


def get_modules_by_names(
    model: torch.nn.Module, regex_patterns: list[str]
) -> list[torch.nn.Module]:
    """
    Given a model and a list of regex patterns, return a list of modules
    whose names match any of the patterns.

    Args:
        model (torch.nn.Module): The model to search for modules.
        regex_patterns (List[str]): A list of regex patterns to match against module names.

    Returns:
        List[torch.nn.Module]: A list of modules whose names match any of the patterns.
    """
    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in regex_patterns]
    modules = [
        module
        for name, module in model.named_modules()
        if any(pattern.search(name) for pattern in compiled_patterns)
    ]
    return modules
