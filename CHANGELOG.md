# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-29

Initial public release.

### Added

- Three distillation strategies built on the Hugging Face `Trainer`:
  `ResponseBasedDistiller` (logit matching), `HolisticDistiller` (full forward
  passes with cross-layer gradient flow), and `BlockwiseDistiller`
  (block-by-block supervision), plus the `Distiller` factory for
  config-driven selection.
- `create_alignments` for regex-based teacher/student layer alignment, with
  automatic projector insertion (`GenericLinearProjector`,
  `GenericConv2dProjector`) and post-training projector fusion.
- Loss registry with MSE, normalized MSE, cosine, smooth L1, KL divergence,
  JSD, logit-lens KL, contrastive (InfoNCE), angular-magnitude, Mahalanobis
  MSE/cosine, and RelKD distance/angle losses.
- Per-alignment loss weighting with optional magnitude-aware normalization.
- Teacher placement strategies (replicated, FSDP-sharded, dedicated GPUs with
  pipeline or tensor parallelism) and overlapped teacher/student forward
  passes; DDP, FSDP, and DeepSpeed support on the student side.
- Model preparation utilities: `reconfig_model`, `prune_model`
  (via `torch-pruning`), `freeze_parameters`, and checkpoint loaders that
  restore student weights with or without projectors.
- Optional integrations: Liger fused distillation losses, WeightWatcher
  metrics, memory diagnostics, and profiler hooks.

[Unreleased]: https://github.com/silverspoon-dev/silverspoon-kd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/silverspoon-dev/silverspoon-kd/releases/tag/v0.1.0
