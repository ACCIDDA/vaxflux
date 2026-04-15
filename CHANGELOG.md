# Changes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Added:

- Support for python 3.13. See
  [#103](https://github.com/ACCIDDA/vaxflux/issues/103).

Changed:

- Update the minimum required version for `numpy` from 1.17.0 to 2.1.0.

Deprecated:

- ...

Removed:

- ...

Fixed:

- Reduced the run time of the `demos/` notebooks by reducing the number of
  samples drawn in the `nis-frvm-survey.ipynb` notebook. See
  [#119](https://github.com/ACCIDDA/vaxflux/issues/119).

Security:

- ...

## [0.3.0] - 2026-04-07

Added:

- Added a `Curve.plot` method that is available when the "plot" optional
  dependency group is installed to quickly visualize curves. See
  [#176](https://github.com/ACCIDDA/vaxflux/issues/176).
- Added a class docstring to `VaxfluxModel` including a brief example. See
  [#135](https://github.com/ACCIDDA/vaxflux/issues/135).
- Added `VaxfluxObservations` container class for observations, see
  [#160](https://github.com/ACCIDDA/vaxflux/issues/160).
- Added ability to easily construct an observations instance, with
  `VaxfluxObservations.from_csv`, as well as a default model, with
  `VaxfluxModel.from_csv`, see
  [#112](https://github.com/ACCIDDA/vaxflux/issues/112).

Changed:

- Changed the expected output of `Covariate.sample` to either be a 1D
  `(num_seasons,)` or 2D `(num_seasons, num_categories)` array, removed the
  `Covariate.dims` method, and added the optional `Covariate.extra_dims` method
  to provide supplemental coordinates for latent variables.
- Updated the `pandas` dependency to be at least 3.0.0 and limit `mkdocs`
  dependency to less than 2.0.0.
- Added explanatory details to demo notebooks, see
  [#175](https://github.com/ACCIDDA/vaxflux/issues/175).
- Overhauled the `release.yaml` GitHub actions workflow to submit `vaxflux` to
  PyPI, see [#161](https://github.com/ACCIDDA/vaxflux/issues/161).

Fixed:

- Expanded documentation and unit tests for existing internal helpers. See
  [#33](https://github.com/ACCIDDA/vaxflux/issues/33),
  [#42](https://github.com/ACCIDDA/vaxflux/issues/42).

## [0.2.0] - 2026-02-04

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
  - Added `render_model` method to wrap
    [`numpyro.render_model`](https://num.pyro.ai/en/stable/utilities.html#render-model)
    to produce a graphic of the underlying model. Added an example of its usage
    to the `demos/vaxflux-model.ipynb` notebook. See
    [#156](https://github.com/ACCIDDA/vaxflux/issues/156).
  - Added `PooledGaussianCovariate`, `PartiallyPooledGaussianCovariate`,
    `SeasonVaryingPartiallyPooledGaussianCovariate` covariate classes to specify
    the priors/parameters for covariates. See
    [#106](https://github.com/ACCIDDA/vaxflux/issues/106).
- Added `VaxfluxInferenceData` class that inherits from
  [`arviz.InferenceData`](https://python.arviz.org/en/stable/api/inference_data.html)
  and takes advantage of coords/dims to make the output easier to align and
  manipulate. See [#111](https://github.com/ACCIDDA/vaxflux/issues/111),
  [#141](https://github.com/ACCIDDA/vaxflux/issues/141).
  - Added `plot_prior_predictive` and `plot_posterior_predictive` methods to
    easily plot the results of a model with sensible defaults for visualization.
    See [#159](https://github.com/ACCIDDA/vaxflux/issues/159),
    [#174](https://github.com/ACCIDDA/vaxflux/issues/174).
  - Added `prior_prevalence_scenarios` and `posterior_prevalence_scenarios`
    methods to produce a human readable set of projections for end of season
    prevalence ranges. See
    [#150](https://github.com/ACCIDDA/vaxflux/issues/150).
- Added demonstration of `VaxfluxModel` class in the `demos/vaxflux-model.ipynb`
  and `demos/season-only-vaxflux-model.ipynb` notebooks using synthetic data as
  well as in the `demos/nis-frvm-survey.ipynb` notebook with real data. See
  [#173](https://github.com/ACCIDDA/vaxflux/issues/173).

Changed:

- Switched from `sphinx` to `mkdocs` for documentation, see
  [#85](https://github.com/ACCIDDA/vaxflux/issues/85).
- Switched from `make` to `just` for task running, see
  [#86](https://github.com/ACCIDDA/vaxflux/issues/86).
- Generate documentation index page from `README.md` file, see
  [#71](https://github.com/ACCIDDA/vaxflux/issues/71).
- Added `noise` argument to `sample_dataset` that allows users to choose between
  normal or gamma distributed noise. Default is 'gamma' for backwards
  compatibility.
- Updated documentation site to reflect changes from the
  ["Model Refactor" milestone](https://github.com/ACCIDDA/vaxflux/milestone/3).
  See [#109](https://github.com/ACCIDDA/vaxflux/issues/109).
- Consolidated the `covariates`, `dates`, `interventions`, and `curves` modules
  as direct exports of `vaxflux` to simplify user imports.

Removed:

- Dropped support for python 3.10, see
  [#90](https://github.com/ACCIDDA/vaxflux/issues/90).
- Removed the `PyMC` based implementation of the vaxflux model, represented
  previously by the `UptakeModel` class, in favor of the `numpyro` based
  implementation. See [#172](https://github.com/ACCIDDA/vaxflux/issues/172).

Fixed:

- Corrected typo in the word "introduction" in the documentation navigation bar.
- Fixed the display of equations in the documentation, see
  [#123](https://github.com/ACCIDDA/vaxflux/issues/123).
- Fixed the display of the demo notebooks in the documentation using
  `mkdocs-jupyter` instead of converting notebooks to markdown files, see
  [#146](https://github.com/ACCIDDA/vaxflux/issues/146).
- Addressed warning from `pd.Categorical` call in private utility about non-null
  values being given that might not be in the `categories`. The fix was moving
  the validation of these values up before constructing the series. See
  [#154](https://github.com/ACCIDDA/vaxflux/issues/154).

## [0.1.0] - 2025-11-13

- Initial version of the package made publicly available centered around users
  creating a `vaxflux.uptake.SeasonalUptakeModel` instance with support for
  flexible curves using `vaxflux.curves`, date ranges using `vaxflux.dates`, and
  covariates using `vaxflux.covariates`.
