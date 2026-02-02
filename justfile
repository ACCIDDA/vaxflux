# Default virtual environment location
rm    := "rm -f"
rmdir := "rm -rf"
cp    := "cp"
mv    := "mv"
mkdir := "mkdir -vp"

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
    {{rmdir}} docs/demos
    {{rmdir}} site
    {{rm}} docs/changelog.md
    {{rm}} docs/contributing.md
    {{rm}} docs/index.md

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
reference: venv
    {{cp}} CHANGELOG.md docs/changelog.md
    {{cp}} CONTRIBUTING.md docs/contributing.md
    {{cp}} README.md docs/index.md

# Generate demo documentation
[unix]
[group('docs')]
demos: venv
    {{mkdir}} docs/demos/
    {{cp}} demos/*.ipynb docs/demos/

# Build complete documentation
[group('docs')]
docs: venv reference demos
    uv run mkdocs build --verbose --strict

# Serve documentation locally
[group('docs')]
serve: venv docs
    uv run mkdocs serve

# Setup the venv
[group('dev')]
venv:
    uv sync --all-extras

# Format code with ruff
[group('dev')]
format: venv
    uv run ruff format

# Check and fix code issues with ruff
[group('dev')]
check: venv
    uv run ruff check --fix --unsafe-fixes

# Run mypy type checking
[group('dev')]
mypy: venv
    uv run mypy --strict .

# Run pytest tests
[group('dev')]
pytest: venv
    uv run pytest --doctest-modules

# Run demo tests
[unix]
[group('dev')]
demo-test:
    #!/usr/bin/env bash
    set -e # Exit immediately if a command exits with a non-zero status.
    echo "Setting up demo tests"
    {{rmdir}} demos/scripts/
    {{mkdir}} demos/scripts/
    uv run jupyter nbconvert --to script demos/*.ipynb
    {{mv}} demos/*.py demos/scripts/
    echo "Running demo test scripts"
    for file in demos/scripts/*.py; do
        echo "Running $file"
        time uv run python "$file"
    done
    echo "All demo scripts passed."

# Run CI checks (non-interactive)
[group('ci')]
ci: venv
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
yamllint: venv
    uv run yamllint --strict .
