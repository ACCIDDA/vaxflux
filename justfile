# Default virtual environment location
rm    := "rm -f"
rmdir := "rm -rf"
cp    := "cp"
mv    := "mv"
mkdir := "mkdir -vp"

# Default recipe: run all tasks
default: dev lint docs

# Run all development checks
dev: format check mypy cov

# Run all linting checks
lint: cspell prettier yamllint

# Clean documentation artifacts
[unix]
[group('clean')]
clean-docs:
    {{rmdir}} docs/demos
    {{rmdir}} site
    {{rm}} docs/changelog.md
    {{rm}} docs/contributing.md
    {{rm}} docs/index.md

# Clean all build artifacts and caches
[unix]
[group('clean')]
clean: clean-docs
    {{rmdir}} .*cache/
    {{rmdir}} .venv/
    {{rmdir}} node_modules/
    {{rmdir}} src/vaxflux/__pycache__/
    {{rmdir}} src/vaxflux.egg-info/
    {{rm}} uv.lock

# Generate API reference documentation
[unix]
[group('docs')]
reference:
    {{cp}} CHANGELOG.md docs/changelog.md
    {{cp}} CONTRIBUTING.md docs/contributing.md
    {{cp}} README.md docs/index.md

# Generate demo documentation
[unix]
[group('docs')]
demos:
    {{mkdir}} docs/demos/
    {{cp}} demos/*.ipynb docs/demos/

# Build complete documentation
[group('docs')]
docs: reference demos
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

# Run pytest with a coverage report
[group('dev')]
cov:
    uv run pytest --doctest-modules --cov=src/vaxflux --cov-report=term-missing

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
    TMP=$(mktemp)
    touch ${TMP}
    for file in demos/scripts/*.py; do
        awk 'NR==4{print "import matplotlib; matplotlib.use(\"Agg\")"}1' ${file} > ${TMP} && mv ${TMP} ${file}
        echo "Running $file"
        time uv run python "$file"
    done
    echo "All demo scripts passed."

# Run CI checks (non-interactive)
[group('ci')]
ci: quality ci-pytest

# Run CI quality checks
[group('ci')]
quality: ci-ruff ci-mypy

# Run CI ruff formatting and linting checks
[group('ci')]
ci-ruff:
    uv run ruff format --check
    uv run ruff check --no-fix

# Run CI mypy type checking
[group('ci')]
ci-mypy:
    uv run mypy --strict .

# Run CI pytest checks using the resolution from `UV_RESOLUTION`
[group('ci')]
ci-pytest:
    uv run --isolated --group dev --extra plot pytest --doctest-modules --exitfirst

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
    uv run yamllint --strict .

# Create a GitHub release
[confirm]
[unix]
[group('release')]
release +FLAGS='':
    uv run python scripts/release.py {{FLAGS}}
