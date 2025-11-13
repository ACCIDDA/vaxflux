# Default virtual environment location
rm    := "rm -f"
rmdir := "rm -rf"
cp    := "cp"
mv    := "mv"

# Default recipe: run all tasks
default: dev lint docs

# Run all development checks
dev: format check mypy pytest

# Run all linting checks
lint: cspell prettier yamllint

# Clean documentation artifacts
[unix]
clean-docs:
    {{rmdir}} docs/api
    {{rmdir}} site
    {{rm}} docs/changelog.md

# Clean all build artifacts and caches
[unix]
clean: clean-docs
    {{rmdir}} .mypy_cache
    {{rmdir}} .pytest_cache
    {{rmdir}} .ruff_cache
    {{rmdir}} .venv
    {{rmdir}} src/vaxflux/__pycache__
    {{rmdir}} src/vaxflux.egg-info
    {{rm}} uv.lock

# Generate API reference documentation
[unix]
api-reference:
    uv run python scripts/api-reference.py
    {{cp}} CHANGELOG.md docs/changelog.md

# Build complete documentation
docs: api-reference
    uv run mkdocs build --verbose --strict

# Serve documentation locally
serve: docs
    uv run mkdocs serve

# Format code with ruff
format:
    uv run ruff format

# Check and fix code issues with ruff
check:
    uv run ruff check --fix --unsafe-fixes

# Run mypy type checking
mypy:
    uv run mypy --strict .

# Run pytest tests
pytest:
    uv run pytest --doctest-modules

# Run CI checks (non-interactive)
ci:
    uv run ruff format --check
    uv run ruff check --no-fix
    uv run mypy --strict .
    uv run pytest --doctest-modules --exitfirst

npm-install:
    npm install

prettier: npm-install
    npm run prettier:fix

prettier-check: npm-install
    npm run prettier:check

[unix]
cspell: npm-install
    #!/usr/bin/env bash
    jq -S 'walk(if type == "array" then sort else . end) | .' cspell.json > .cspell.json
    {{mv}} .cspell.json cspell.json
    npm run cspell

yamllint:
    uvx yamllint --strict .
