UV_PROJECT_ENVIRONMENT ?= .venv
RM := rm -f
RMDIR := rm -rf

.PHONY: all check ci clean docs format lint mypy pytest rstcheck serve

all: clean .venv lint mypy pytest

clean:
	$(RMDIR) .mypy_cache
	$(RMDIR) .pytest_cache
	$(RMDIR) .ruff_cache
	$(RMDIR) .venv
	$(RMDIR) docs/_build
	$(RMDIR) src/vaxflux/__pycache__
	$(RMDIR) src/vaxflux.egg-info
	$(RM) uv.lock

.venv:
	uv sync --all-extras
	uv pip install --editable .

rstcheck:
	$(UV_PROJECT_ENVIRONMENT)/bin/rstcheck --warn-unknown-settings --recursive docs/

docs/_build: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/sphinx-build -b html docs docs/_build

docs:
	@$(MAKE) rstcheck
	$(RMDIR) docs/_build
	@$(MAKE) docs/_build

serve: .venv docs/_build
	$(UV_PROJECT_ENVIRONMENT)/bin/python -m http.server --directory docs/_build

format: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format

check: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --fix

lint: format check

mypy: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy --strict .

pytest: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules

ci: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format --check
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --no-fix
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy .
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules --exitfirst
