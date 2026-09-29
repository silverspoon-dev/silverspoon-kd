# Tests

Unit and integration tests for the `silverspoon_kd` package. They use small
synthetic models (`SimpleModel`, `TPSimpleModel`) and a deterministic
`DummyDataset` defined in `tests/silverspoon_kd/conftest.py`, so the CPU suite
runs without downloading anything.

## Layout

```
tests/
├── conftest.py                 # --device option, cuda marker, device fixture
└── silverspoon_kd/
    ├── conftest.py             # model, dataset, alignment and training_args fixtures
    ├── alignments/             # Alignment, create_alignments, projectors, fusion, checkpoint loading
    ├── distillers/             # Blockwise/Holistic/ResponseBased distillers, factory, distributed runs
    ├── distributed/            # TeacherPlacement and setup_split_gpu
    ├── engines/                # ModuleCaptureEngine, profiler, memory diagnostics
    ├── losses/                 # every loss in LOSS_REGISTRY
    ├── optim/                  # composite optimizer and scheduler
    ├── test_public_api.py      # locks the top-level __all__
    ├── test_training_arguments.py
    ├── test_reconfig.py, test_pruning.py, test_utils.py
    └── test_clip_projectors.py
```

## Running

Install the package with its dev extras first (`pip install -e ".[dev]"`).

| Command | What it does |
|---|---|
| `make test` | CPU suite: `pytest tests/ -v --device=cpu`. This is what CI runs. |
| `make test-cuda` | Full suite on GPU with `pytest-xdist` (`--device=cuda -n 8`), including distributed tests. |
| `make test-cuda-only` | Only tests marked `cuda`, for GPU cluster jobs. |
| `make coverage` | CPU suite under `pytest-cov` with terminal, HTML and JSON reports; fails below 85% coverage. |

Run a subset by path or keyword as usual:

```bash
pytest tests/silverspoon_kd/losses -q --device=cpu
pytest tests/silverspoon_kd/distillers/test_blockwise_distiller.py -k checkpoint
```

## Device selection

`tests/conftest.py` adds a `--device` option with the choices `cpu` (default)
and `cuda`. The `device` fixture returns `torch.device("cuda")` only when
`--device=cuda` is passed *and* CUDA is available; otherwise it returns the
CPU device. Tests never pick the GPU on their own, so a plain `pytest`
invocation always runs on CPU.

## The `cuda` marker

Tests that need a GPU (multi-GPU placement, FSDP/DDP/DeepSpeed runs, device
placement) are decorated with `@pytest.mark.cuda`. Without `--device=cuda`
they are skipped at collection time, which is how the CPU-only CI job stays
green. Mark any new GPU-only test the same way; tests that need several GPUs
additionally guard on `torch.cuda.device_count()` with `skipif`.

## Writing tests

- Reuse the fixtures in `tests/silverspoon_kd/conftest.py` (`teacher_model`,
  `student_model`, `teacher_alignments`, `train_dataset`, `training_args`, ...).
- Put tests for a subpackage in the matching directory above.
- Public API additions must be exported from `silverspoon_kd/__init__.py` and
  listed in `__all__`; `test_public_api.py` enforces this.
