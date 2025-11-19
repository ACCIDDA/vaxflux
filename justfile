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
[group('clean')]
clean-docs:
    {{rmdir}} docs/api
    {{rmdir}} site
    {{rm}} docs/changelog.md

# Clean all build artifacts and caches
[unix]
[group('clean')]
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
[group('docs')]
api-reference:
    uv run python scripts/api-reference.py
    {{cp}} CHANGELOG.md docs/changelog.md
    {{cp}} CONTRIBUTING.md docs/contributing.md
    {{cp}} README.md docs/index.md

# Build complete documentation
[group('docs')]
docs: api-reference
    uv run mkdocs build --verbose --strict

# Serve documentation locally
[group('docs')]
serve: docs
    uv run mkdocs serve

# Format code with ruff
[group('dev')]
format:
    uv run ruff format

# Check and fix code issues with ruff
[group('dev')]
check:
    uv run ruff check --fix --unsafe-fixes

# Run mypy type checking
[group('dev')]
mypy:
    uv run mypy --strict .

# Run pytest tests
[group('dev')]
pytest:
    uv run pytest --doctest-modules

# Run CI checks (non-interactive)
[group('ci')]
ci:
    uv run ruff format --check
    uv run ruff check --no-fix
    uv run mypy --strict .
    uv run pytest --doctest-modules --exitfirst

# Install npm dependencies
[group('lint')]
npm-install:
    npm install

# Run prettier to format files
[group('lint')]
prettier: npm-install
    npm run prettier:fix

# Run prettier to check file formatting
[group('lint')]
prettier-check: npm-install
    npm run prettier:check

# Run cspell spell checking
[unix]
[group('lint')]
cspell: npm-install
    #!/usr/bin/env bash
    jq -S 'walk(if type == "array" then sort else . end) | .' cspell.json > .cspell.json
    {{mv}} .cspell.json cspell.json
    npm run cspell

# Run yamllint on YAML files
[group('lint')]
yamllint:
    uvx yamllint --strict .
