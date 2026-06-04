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
    {{rmdir}} docs/images
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

# Generate API reference documentation
[unix]
[group('docs')]
reference: images
    {{cp}} CHANGELOG.md docs/changelog.md
    {{cp}} CONTRIBUTING.md docs/contributing.md
    {{cp}} README.md docs/index.md

# Generate static images for documentation
[unix]
[group('docs')]
images:
    uv run --extra plot python scripts/generate_docs_images.py

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
    uv run --group dev --extra demo jupyter nbconvert --to script demos/*.ipynb
    {{mv}} demos/*.py demos/scripts/
    echo "Running demo test scripts"
    TMP=$(mktemp)
    touch ${TMP}
    for file in demos/scripts/*.py; do
        awk 'NR==4{print "import matplotlib; matplotlib.use(\"Agg\")"}1' ${file} > ${TMP} && mv ${TMP} ${file}
        echo "Running $file"
        time uv run --group dev --extra demo python "$file"
    done
    echo "All demo scripts passed."

# Run CI checks (non-interactive)
[group('ci')]
ci: quality ci-pytest changelog-check

# Run CI quality checks
[group('ci')]
quality: ci-ruff ci-mypy

# Run CI ruff formatting and linting checks
[group('ci')]
ci-ruff:
    uv run --locked ruff format --check
    uv run --locked ruff check --no-fix

# Run CI mypy type checking
[group('ci')]
ci-mypy:
    uv run --locked --extra plot mypy --strict .

# Run CI pytest checks against the committed lockfile
[group('ci')]
ci-pytest:
    uv run --locked --isolated --group dev --extra plot pytest --doctest-modules --exitfirst

# Run CI minimum-version pytest checks without the committed lockfile
[unix]
[group('ci')]
ci-pytest-lowest-direct:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f uv.lock ]; then
        trap 'if [ -f uv.lock.bak ]; then mv uv.lock.bak uv.lock; fi' EXIT
        mv uv.lock uv.lock.bak
    fi
    env -u UV_LOCKED -u UV_FROZEN UV_RESOLUTION=lowest-direct \
        uv run --isolated --group dev --extra plot pytest --doctest-modules --exitfirst

# Check that CHANGELOG.md has been updated relative to origin/main
[unix]
[group('ci')]
changelog-check:
    #!/usr/bin/env bash
    set -euo pipefail
    BASE_BRANCH="${CHANGELOG_BASE_BRANCH:-main}"
    BASE_REF="origin/${BASE_BRANCH}"
    git fetch origin "${BASE_BRANCH}" --depth=1
    GIT_LOG="$(git log "${BASE_REF}"..HEAD --pretty=format:"%s %b")"
    SKIP_REGEX="no[[:space:]]+major[[:space:]]+changes"
    if echo "${GIT_LOG}" | tr '\n' ' ' | grep -Eiq "${SKIP_REGEX}"; then
        echo "Bypassing changelog check: 'no major changes' found in commit history"
        exit 0
    fi
    MERGE_BASE="$(git merge-base "${BASE_REF}" HEAD)"
    if ! git diff --name-only "${MERGE_BASE}" | grep -q "^CHANGELOG.md$"; then
        echo "Error: Please update CHANGELOG.md"
        exit 1
    fi

# Build sdist and wheel, then validate package metadata
[unix]
[group('build')]
build-check:
    {{rmdir}} dist/
    uv run python -m build
    uv run python -m twine check --strict dist/*

# Install the built wheel into a clean room and run tests against it
[unix]
[group('build')]
build-test:
    #!/usr/bin/env bash
    set -euo pipefail
    CLEANROOM="$(mktemp -d)"
    trap 'rm -rf "${CLEANROOM}"' EXIT
    uv export --locked --group dev --extra plot --no-emit-project --format requirements.txt --no-hashes --output-file "${CLEANROOM}/dev-requirements.txt" >/dev/null
    uv run python -m build --wheel --outdir "${CLEANROOM}/dist"
    uv venv --python "${UV_PYTHON_VERSION:-3.12}" "${CLEANROOM}/venv"
    uv pip install --python "${CLEANROOM}/venv/bin/python" "${CLEANROOM}"/dist/*.whl
    uv pip install --python "${CLEANROOM}/venv/bin/python" -r "${CLEANROOM}/dev-requirements.txt"
    cp -R tests "${CLEANROOM}/tests"
    (
        cd "${CLEANROOM}"
        "${CLEANROOM}/venv/bin/python" -m pytest --import-mode=importlib tests -q -x
    )

# Run all package build validations
[group('build')]
build-all: build-check build-test

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

# Validate release metadata without creating a GitHub release
[group('release')]
release-check:
    uv run python scripts/release.py

# Run the full local release preflight
[group('release')]
release-validate: build-all release-check
