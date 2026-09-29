
docs:
	sphinx-build -b html -W --keep-going docs docs/_build/html

serve-docs:
	sphinx-autobuild docs docs/_build/html --port 8000 --open-browser

check:
	ruff check .
	ruff format --check .

typecheck:
	python -m pyright --pythonpath "$$(which python)"

# ``fix`` runs ruff in auto-modify mode for both lint AND formatting,
# matching what the pre-commit hook does on every commit.  Named
# ``fix`` rather than ``format`` because the lint pass also rewrites
# code in non-cosmetic ways (removes unused imports, simplifies
# expressions, modernises typing, etc.) — calling that "format" would
# be misleading.
fix:
	ruff check --fix .
	ruff format .

test:
	python -m pytest tests/ -v --device=cpu

test-cuda:
	python -m pytest tests/ -v --device=cuda -n 8 --dist=worksteal -p no:benchmark

# GPU-only: run only tests marked with @pytest.mark.cuda (distributed,
# device-placement, etc.).  Use this in GPU cluster jobs to avoid wasting
# GPU time on the ~85% of tests that are CPU-only.
test-cuda-only:
	python -m pytest tests/ -v --device=cuda -n 8 --dist=worksteal -p no:benchmark -m cuda

# Report formats shared by both coverage targets — add new reports
# (e.g. ``--cov-report=xml`` for Codecov) here and the CPU and CUDA
# variants both pick them up.
COV_FLAGS := --cov=silverspoon_kd --cov-report=term-missing --cov-report=html --cov-report=json

coverage:
	python -m pytest tests/ -v --device=cpu $(COV_FLAGS)

coverage-cuda:
	python -m pytest tests/ -v --device=cuda $(COV_FLAGS)

benchmark:
	python -m pytest benchmarks/ -v --benchmark-autosave --device=cuda $(ARGS)

benchmark-cpu:
	python -m pytest benchmarks/ -v --benchmark-autosave --device=cpu $(ARGS)

benchmark-compare:
	python -m pytest benchmarks/ -v --benchmark-only --benchmark-compare --device=cuda $(ARGS)

# Rebuild the concept diagrams (needs pdflatex and pdftocairo from poppler).
diagrams:
	cd docs/assets/diagrams && \
	for f in *.tex; do \
		pdflatex -interaction=nonstopmode -halt-on-error "$$f" > /dev/null && \
		pdftocairo -svg "$${f%.tex}.pdf" "$${f%.tex}.svg" && \
		rm -f "$${f%.tex}.pdf" "$${f%.tex}.aux" "$${f%.tex}.log"; \
	done

serve-coverage:
	python -m http.server 8001 -d htmlcov

clean:
	rm -rf htmlcov/ .coverage coverage.json
	rm -rf .pytest_cache/ .ruff_cache/
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +

ci: check typecheck test

# Build sdist + wheel into dist/ and validate the PyPI metadata.  Needs the
# ``release`` extra (``pip install -e ".[release]"``).
build:
	rm -rf build/ dist/
	python -m build
	python -m twine check --strict dist/*

ci-cuda: check typecheck test-cuda

.PHONY: \
	docs serve-docs serve-coverage \
	check typecheck fix \
	build \
	test test-cuda test-cuda-only \
	coverage coverage-cuda \
	benchmark benchmark-cpu benchmark-compare \
	ci ci-cuda \
	diagrams clean
