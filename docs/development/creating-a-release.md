# Creating A Release

This guide covers the full `vaxflux` release process, from preparing the
changelog to running the manual GitHub Actions release workflow.

## Prerequisites

- You have write access to the
  [`ACCIDDA/vaxflux`](https://github.com/ACCIDDA/vaxflux) repository.
- The version has already been updated in `pyproject.toml` and
  `src/vaxflux/__init__.py`.
- The release notes for that version have already been added to `CHANGELOG.md`.
- Your local environment is synced with a supported Python version via
  `uv sync --locked --python 3.12 --group dev`.

## 1. Format The Changelog

The release workflow validates `CHANGELOG.md` using `scripts/release.py`, so the
changelog format must match what that script expects.

For a release, the top section of `CHANGELOG.md` must:

- Not contain a `## [Unreleased]` heading.
- Start with a heading in the form `## [X.Y.Z] - YYYY-MM-DD`.
- Use the current release version and the current date.
- Contain at least one non-empty bullet point.

Example:

```md
## [0.3.0] - 2026-04-06

Added:

- Added manual release workflow dispatch with package preflight validation.
- Added clean-room wheel installation tests for release preparation.

Changed:

...
```

If the top heading, date, or bullet formatting is wrong, `just release-check`
and the release workflow will fail before anything is published.

## 2. Run The Local Release Preflight

Use the `just release-validate` recipe to run the following checks locally:

- `build-check`: Build the source distribution and wheel, then run
  `twine check --strict` on the generated artifacts.
- `build-test`: Build the wheel, install it into a clean virtual environment,
  installs the pinned development test dependencies exported from `uv.lock`, and
  run the test suite against the installed wheel.
- `release-check`: Run `uv run python scripts/release.py` without `--create` to
  ensure the `CHANGELOG.md` is formatted correctly.

## 3. Run The Release Workflow

Releases are created through the manual GitHub Actions workflow in
`.github/workflows/release.yaml`. You can trigger it from the GitHub UI, but it
is preferred that the `gh` CLI is used to trigger it so appropriate options can
be provided.

If you are testing the workflow before the pull request is merged, add
`--ref <branch-name>` to the `gh workflow run` command. Without `--ref`, GitHub
dispatches the workflow definition from the repository's default branch, so any
unmerged `workflow_dispatch` inputs will not exist yet. Using
`--ref <branch-name>` means the workflow validates, builds, and publishes
artifacts from that branch, not from `main`. That is appropriate for dry runs
and TestPyPI testing before merge, but it should not be used for a real PyPI
release. Real releases should be triggered from `main` after the pull request
has been merged.

### Dry Run

Use this to validate the release end to end without publishing to TestPyPI or
PyPI, creating a GitHub release, or deploying docs:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/vaxflux \
  --ref <branch-name> \
  --field publish-target=none \
  --field create-github-release=false \
  --field deploy-docs=false
```

This runs the `validate` job only. It will still enforce the changelog and
release metadata checks through `just release-validate`.

This is the recommended way to test the workflow before merge.

### TestPyPI

Use this to publish the built artifacts to TestPyPI without creating the GitHub
release or deploying docs:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/vaxflux \
  --ref <branch-name> \
  --field publish-target=testpypi \
  --field create-github-release=false \
  --field deploy-docs=false
```

This is the safest way to test the release workflow against a real package index
before publishing to PyPI. To check that the package can be successfully
installed after uploading to TestPyPI you can run:

```shell
uv venv --python 3.12 /tmp/vaxflux-testpypi
uv pip install \
  --python /tmp/vaxflux-testpypi/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  vaxflux
```

The extra index is needed so dependencies can still be resolved from PyPI.

When testing from a branch, keep `create-github-release=false` and
`deploy-docs=false`. The documentation deployment workflow checks out `main`, so
it is not intended for pre-merge branch testing.

### Full PyPI Release

Use this to run the full release flow:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/vaxflux \
  --field publish-target=pypi \
  --field create-github-release=true \
  --field deploy-docs=true
```

This workflow runs, in order:

1. `just release-validate`.
2. Publish the built artifacts to PyPI.
3. Create the GitHub release.
4. Deploy the versioned documentation.

Run the full PyPI release flow only from `main`. Do not use `--ref` with a
feature branch for a real release.

## 4. Trusted Publishing Setup

The release workflow uses PyPI Trusted Publishing rather than a stored API
token. To enable publishing, configure the trusted publisher entry on TestPyPI
and PyPI for:

- Owner: `ACCIDDA`.
- Repository: `vaxflux`.
- Workflow file: `release.yaml`.

If that configuration is missing, the publish job will fail even if validation
passes.
