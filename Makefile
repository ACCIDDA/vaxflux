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
	./.venv/bin/ruff format

check: .venv
	./.venv/bin/ruff check --fix

mypy: .venv
	./.venv/bin/mypy .

pytest: .venv
	./.venv/bin/pytest

ci: .venv
	./.venv/bin/ruff format --check
	./.venv/bin/ruff check --no-fix
	./.venv/bin/mypy .
	./.venv/bin/pytest --exitfirst
