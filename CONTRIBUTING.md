# Contributing

This document provides guidelines and instructions for contributing to
`vaxflux`, including development conventions and tips for best practices.

## Development Setup

### Prerequisites

- Python 3.11 or 3.12.
- [`uv`](https://docs.astral.sh/uv/) - Python package manager.
- [`npm`](https://www.npmjs.com/) - Node package manager for auxiliary
  formatting/linting tools.
- [`just`](https://just.systems/) - Command runner (optional but strongly
  recommended).

### Initial Setup

1. Clone the repository:

```shell
git clone git@github.com:ACCIDDA/vaxflux.git
cd vaxflux
```

2. Install dependencies:

```shell
uv sync --all-extras
```

This installs the package along with development, documentation, and optional
plotting dependencies used across local workflows. If you want to pin the
environment to a specific supported Python version you can also run:

```shell
uv sync --all-extras --python 3.12
```

Most commands in this repository are invoked through `uv run ...`, which will
also create or update the local environment as needed.

3. Verify your setup

```shell
just
```

This runs the default development checks:

- `ruff format` - Format code.
- `ruff check --fix` - Lint and auto-fix issues.
- `mypy --strict` - Type check with strict settings.
- `pytest --doctest-modules --cov=src/vaxflux --cov-report=term-missing` - Run
  tests including doctests with a local coverage report.
- `npm run cspell` - Spell check linting.
- `npm run prettier:fix` - Formatting for JSON/markdown/YAML files.
- `yamllint --strict` - Lint YAML files.
- `uv run mkdocs build --verbose --strict` - Build and check the mkdocs
  documentation.

It is recommended that you run this command frequently as you do development
work to catch issues early. You can also run `just --list` to list out the
available commands.

## Code Standards

### Code Style

[`ruff`](https://docs.astral.sh/ruff/) is used for both formatting and linting
python:

- **Formatting**: `ruff format` follows the Black code style.
- **Linting**: `ruff check` enforces code quality rules.

You can run `just` to automatically format and lint your code before committing
or you can run `just format` and `just check` to run just these steps.

### Docstring Style

[google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
comments and docstrings are used to document code.

## Testing

### Organization

Tests are organized to mirror the source code structure:

```
tests/
|-- {module}/
|   |-- {submodule}/
|   |   |-- test_{function}.py                   # Tests for individual functions
|   |   |-- test_{class_in_snake_case}_class.py  # Tests for classes
|   |-- test_{function}.py
```

**Examples**

- For a function called `vaxflux.foo.bar` the unit test file would be located at
  `tests/foo/test_bar.py`.
- For a class called `vaxflux.example.FizzBuzz` the unit test file would be
  located at `tests/example/test_fizz_buzz_class.py`.
- For a private function called `vaxflux._internal._utility` the unit test file
  would be located at `tests/_internal/test__utility.py`.

### Running Tests

Running all tests is a part of `just` and `just dev`. If you instead want to run
the Python test suite with a coverage report you can run `just cov`. Coverage is
currently for local visibility only; CI does not enforce a coverage threshold.
For more advanced invocations, please use `pytest` directly:

```shell
uv run pytest tests/{module}/ -v                    # Run all tests in a module
uv run pytest tests/{module}/test_{function}.py -v  # Run specific test file
```

For more information on how to invoke pytest please refer to the
[How to invoke pytest](https://docs.pytest.org/en/stable/how-to/usage.html)
documentation.

### Writing Tests

- Any public API should have unit tests that reaffirm the documentation's
  description.
- If possible unit tests should use `@pytest.mark.parametrize` for generality
  and ease of adding new test cases.
- Use descriptive test names that explain what is being tested. In the case of
  testing exceptions also the type of exception.
- For smaller helper functions, especially internal helpers, doctests are
  sufficient.

### Type Checking

All code must pass strict type checking with `mypy` which can be invoked with
`just mypy`. Note that `ruff` will catch missing type hints whereas `mypy` will
check that those type hints are correct and consistent.

## Documentation

`vaxflux` uses [MkDocs](https://www.mkdocs.org/) for documentation.

### Editing Documentation

Documentation files are located in the `docs/` directory. The documentation
structure is defined in `mkdocs.yml`, including the navigation structure.

### Viewing Documentation Locally

To preview documentation changes locally you can run `just serve` which will
build the documentation and start a local server at
`http://127.0.0.1:8000/vaxflux/`. To only build the documentation you can run
`just docs` which will generate the documentation site in the `site/` directory.

## Pull Request Process

### Before Submitting

1. Run CI checks locally using `just ci`. This runs the same checks that CI will
   run:

- `ruff format --check` - Verify code formatting (no auto-fix).
- `ruff check --no-fix` - Lint without modifications.
- `mypy --strict` - Type checking.
- `pytest --doctest-modules --exitfirst` - Run the test suite in a CI-style
  configuration.

In addition, you should run:

- `just docs` - Build the MkDocs documentation as CI does.
- `just lint` - Run `cspell`, `prettier`, and `yamllint`.

2. Update documentation if your changes affect user-facing functionality or add
   features that require usage guides. Also update the `CHANGELOG.md` with a
   description of your changes. Commits containing `no major changes` will skip
   the changelog check in CI.

3. Add tests for new functionality or bug fixes. Particularly for bug fixes, the
   test should be written before the fix and fail without the fix present.

### Submitting a Pull Request

1. Create a branch from `main` and make your changes on this branch using the
   code standards above. Please use clear and descriptive commit messages that
   explain the changes made and why. After your work is finished either push
   directly to `vaxflux` or your fork (depending on your permissions).

2. Create a pull request with:

- The motivation for and a clear description of the changes.
- Link any related issues (use "Fixes #XYZ" or "Closes #XYZ" to auto-close
  issues).
- Explicitly point out the relevant documentation changes.

### Pull Request Requirements

- PR CI runs:
- `docs` on Python 3.12.
- `quality` on Python 3.12.
- `tests` on Python 3.11 and 3.12 with both `highest` and `lowest-direct`
  dependency resolution.
- `changelog` on pull requests.
- A separate lint workflow runs `prettier`, `cspell`, and `yamllint`.
- Documentation must build successfully with `just docs`.
- Linting must also be successful.
- Branches must be up to date against `main` before merging and have a linear
  history. Only rebases are allowed for merging.

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Operating system and version.
- Python version (e.g., Python 3.12.1).
- `vaxflux` version as a git commit hash.
- A minimal, reproducible example that demonstrates the issue.
- An explanation of expected behavior vs actual behavior.
- Error messages or tracebacks if applicable.

### Feature Requests

For feature requests, please:

- Check existing issues to avoid duplicates.
- Clearly describe the use case and motivation.
- Provide examples of how the feature would be used.
- Be open to discussion about implementation approaches.

### Questions

For questions about using `vaxflux`:

- Check the documentation (which can be viewed locally with `just serve` for
  now).
- Search existing issues for similar questions.
- Open a new issue with the "question" label.
