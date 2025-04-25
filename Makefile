RMDIR := rm -rf

.PHONY: clean format check mypy pytest all

clean:
	$(RMDIR) .mypy_cache
	$(RMDIR) .pytest_cache
	$(RMDIR) .ruff_cache
	$(RMDIR) .venv
	$(RMDIR) src/vaxflux/__pycache__
	$(RMDIR) src/vaxflux.egg-info

.venv:
	uv sync --all-extras
	uv pip install --editable .

format: .venv
	./.venv/bin/ruff format

check: .venv
	./.venv/bin/ruff check --fix

mypy: .venv
	./.venv/bin/mypy .

pytest: .venv
	./.venv/bin/pytest

all: clean .venv format check mypy pytest
