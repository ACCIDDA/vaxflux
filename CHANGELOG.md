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
- Added a `CONTRIBUTING.md` document that is also synced with the documentation
  site outlining how to get started as a developer and contribute, see
  [#75](https://github.com/ACCIDDA/vaxflux/issues/75).
- Added versioned documentation site hosted by GitHub pages at
  [accidda.github.io/vaxflux/](https://accidda.github.io/vaxflux/) that
  incorporates jupyter notebook demos, see
  [#14](https://github.com/ACCIDDA/vaxflux/issues/14),
  [#62](https://github.com/ACCIDDA/vaxflux/issues/62).
- Added `VaxfluxModel` and new `Curve` classes that are based on
  `jax`/`numpyro`. See the
  [Model Refactor milestone](https://github.com/ACCIDDA/vaxflux/milestone/3).
  - Added a custom and configurable prevalence penalty to decrease the
    likelihood of prevalence values beyond 1 from being sampled. This is
    configured via the `prevalence_penalty` argument to
    `add_observation_process`. See
    [#148](https://github.com/ACCIDDA/vaxflux/issues/148).
- Added `VaxfluxInferenceData` class that inherits from
  [`arviz.InferenceData`](https://python.arviz.org/en/stable/api/inference_data.html)
  and takes advantage of coords/dims to make the output easier to align and
  manipulate. See [#111](https://github.com/ACCIDDA/vaxflux/issues/111),
  [#141](https://github.com/ACCIDDA/vaxflux/issues/141).
- Added demonstration of `VaxfluxModel` class in the `demos/vaxflux-model.ipynb`
  notebook using synthetic data.

Changed:

- Switched from `sphinx` to `mkdocs` for documentation, see
  [#85](https://github.com/ACCIDDA/vaxflux/issues/85).
- Switched from `make` to `just` for task running, see
  [#86](https://github.com/ACCIDDA/vaxflux/issues/86).
- Generate documentation index page from `README.md` file, see
  [#71](https://github.com/ACCIDDA/vaxflux/issues/71).

Deprecated:

- ...

Removed:

- Dropped support for python 3.10, see
  [#90](https://github.com/ACCIDDA/vaxflux/issues/90).

Fixed:

- Corrected typo in the word "introduction" in the documentation navigation bar.
- Fixed the display of equations in the documentation, see
  [#123](https://github.com/ACCIDDA/vaxflux/issues/123).
- Fixed the display of the demo notebooks in the documentation using
  `mkdocs-jupyter` instead of converting notebooks to markdown files, see
  [#146](https://github.com/ACCIDDA/vaxflux/issues/146).

Security:

- ...

## [0.1.0] - 2025-11-13

- Initial version of the package made publicly available centered around users
  creating a `vaxflux.uptake.SeasonalUptakeModel` instance with support for
  flexible curves using `vaxflux.curves`, date ranges using `vaxflux.dates`, and
  covariates using `vaxflux.covariates`.
