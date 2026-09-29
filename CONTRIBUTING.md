# Contributing to SilverSpoon-KD

Thank you for your interest in contributing. Bug reports, feature requests,
documentation fixes, and pull requests are all welcome.

## Releases and repository history

Each commit on `main` corresponds to one release and contains the complete
source tree for that version, so the history is a linear sequence of releases.
Pull requests are therefore not merged into `main` directly. Accepted changes
are integrated into the next release, credited in `CHANGELOG.md`, and the pull
request is closed with a reference to the release that contains them.

## Development setup

```bash
git clone https://github.com/silverspoon-dev/silverspoon-kd.git
cd silverspoon-kd
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Python 3.10 or newer is required. The pre-commit hook runs `ruff` on every
commit; the pre-push hook runs the CPU test suite.

## Checks

| Command | What it runs |
|---|---|
| `make check` | `ruff check` and `ruff format --check` |
| `make fix` | Auto-fix lint findings and reformat |
| `make typecheck` | `pyright` in `standard` mode over `silverspoon_kd/` |
| `make test` | CPU test suite (what CI runs) |
| `make test-cuda` | Full suite on GPU, including distributed tests |
| `make coverage` | CPU coverage report; fails below 85% |
| `make docs` | Build the Sphinx documentation into `docs/_build/html` |
| `make ci` | `check` + `typecheck` + `test` |

Before opening a pull request, `make ci` should pass locally. CI runs the
lint and typecheck jobs on Python 3.12 and the test job on Python 3.10, 3.11,
and 3.12.

## Pull request guidelines

- Keep each pull request focused on one change.
- Add or update tests for any behaviour you change. Tests that require a GPU
  must be marked with `@pytest.mark.cuda` so the CPU-only CI can skip them.
- `pyright` must report zero errors. Warnings for optional imports
  (`liger-kernel`, `weightwatcher`, `bitsandbytes`) are expected.
- Update the relevant page under `docs/` when you change public behaviour, and
  add a line under `## [Unreleased]` in `CHANGELOG.md`.
- Public API additions must be exported from `silverspoon_kd/__init__.py` and
  listed in `__all__`; `tests/silverspoon_kd/test_public_api.py` enforces this.

## Reporting bugs

Please include the versions of `silverspoon-kd`, `torch`, and `transformers`,
the distiller and teacher-placement strategy you used, and a minimal script
that reproduces the problem. For multi-GPU issues, include the launcher
command (`torchrun`, `accelerate launch`, or `deepspeed`) and the number of
processes.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the same license that covers the project.
