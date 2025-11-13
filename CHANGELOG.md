# Changes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Added:

- Added a CI check for updates to the `CHANGELOG.md` file. This check can be
  bypassed by including "no major changes" (case-insensitive) on the rare
  occasion that a change log update is not required. See
  [#87](https://github.com/ACCIDDA/vaxflux/issues/87).
- Added additional linting checks via `cspell`, `prettier`, and `yamllint` for
  spelling and JSON/markdown/YAML files as well as a linting workflow to run
  these checks regularly.

Changed:

- Switched from `sphinx` to `mkdocs` for documentation, see
  [#85](https://github.com/ACCIDDA/vaxflux/issues/85).
- Switched from `make` to `just` for task running, see
  [#86](https://github.com/ACCIDDA/vaxflux/issues/86).

Deprecated:

- ...

Removed:

- ...

Fixed:

- ...

Security:

- ...

## [0.1.0] - 2025-11-13

- Initial version of the package made publicly available centered around users
  creating a `vaxflux.uptake.SeasonalUptakeModel` instance with support for
  flexible curves using `vaxflux.curves`, date ranges using `vaxflux.dates`, and
  covariates using `vaxflux.covariates`.
