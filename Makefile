UV_PROJECT_ENVIRONMENT ?= .venv
RM := rm -f
RMDIR := rm -rf

.PHONY: all clean format check mypy pytest ci

all: clean .venv format check mypy pytest

clean:
	$(RMDIR) .mypy_cache
	$(RMDIR) .pytest_cache
	$(RMDIR) .ruff_cache
	$(RMDIR) .venv
	$(RMDIR) src/vaxflux/__pycache__
	$(RMDIR) src/vaxflux.egg-info
	$(RM) uv.lock

.venv:
	uv sync --all-extras
	uv pip install --editable .

format: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format

check: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --fix

mypy: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy .

pytest: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules

ci: .venv
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff format --check
	$(UV_PROJECT_ENVIRONMENT)/bin/ruff check --no-fix
	$(UV_PROJECT_ENVIRONMENT)/bin/mypy .
	$(UV_PROJECT_ENVIRONMENT)/bin/pytest --doctest-modules --exitfirst
