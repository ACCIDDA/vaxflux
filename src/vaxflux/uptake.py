"""Tools for constructing uptake models."""

__all__ = ("SeasonalUptakeModel",)


import copy
import itertools
import logging
import math
import sys
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vaxflux._util import (
    _coord_index,
    _coord_name,
    _pm_name,
    _validate_and_format_observations,
)
from vaxflux.covariates import (
    Covariate,
    CovariateCategories,
    _infer_covariate_categories_from_observations,
)
from vaxflux.curves import IncidenceCurve
from vaxflux.dates import DateRange, SeasonRange, _infer_ranges_from_observations

if sys.version_info[:2] >= (3, 11):
    from typing import Self
else:
    Self = Any


class SeasonalUptakeModel:
    """
    A generic class for uptake models.

    Attributes:
        curve: The incidence curve family to use.
        name: The name of the model, optional and only used for display.
        observations: The uptake dataset to use.

    """

    _observation_sigma = 1.0e-9

    def __init__(
        self,
        curve: IncidenceCurve,
        covariates: list[Covariate],
        observations: pd.DataFrame | None = None,
        covariate_categories: list[CovariateCategories] | None = None,
        season_ranges: list[SeasonRange] | None = None,
        date_ranges: list[DateRange] | None = None,
        epsilon: float = 5e-4,
        name: str | None = None,
    ) -> None:
        """
        Initialize an uptake model.

        Args:
            curve: The incidence curve family to use.
            covariates: The covariates to use.
            observations: The uptake dataset to use.
            covariate_categories: The covariate categories for the uptake scenarios or
                `None` to derive them from the observations.
            date_ranges: The date ranges for the uptake scenarios or `None` to derive
                them from the observations.
            season_ranges: The season ranges for the uptake scenarios or `None` to
                derive them from the observations.
            epsilon: The prior average for the standard deviation in observed uptake.
            name: The name of the model, optional and only used for display.

        Returns:
            None

        Raises:
            ValueError: If a covariate parameter is not present in the curve family
                parameters.

        """
        # Public attributes
        self.curve = copy.deepcopy(curve)
        self.name = name
        self.observations = _validate_and_format_observations(observations)

        # Private attributes
        self._covariates = copy.deepcopy(covariates)
        self._covariate_categories: list[CovariateCategories] = (
            copy.deepcopy(covariate_categories) if covariate_categories else list()
        )
        self._date_ranges: list[DateRange] = (
            copy.deepcopy(date_ranges) if date_ranges else list()
        )
        self._model: pm.Model | None = None
        self._season_ranges: list[SeasonRange] = (
            copy.deepcopy(season_ranges) if season_ranges else list()
        )
        self._epsilon = epsilon

        # Input validation
        if covariate_parameters_missing := {
            covariate.parameter for covariate in self._covariates
        } - set(self.curve.parameters):
            raise ValueError(
                "The following parameters were given in the covariates, "
                "but not present in the curve family parameters: "
                f"{sorted(covariate_parameters_missing)}."
            )
        if self.observations is not None and (
            covariate_covariates_missing := {
                covariate.covariate
                for covariate in self._covariates
                if covariate.covariate is not None
            }
            - set(self.observations.columns)
        ):
            raise ValueError(
                "The following covariates were given in the covariates, "
                "but not present in the observation columns: "
                f"{sorted(covariate_covariates_missing)}."
            )
        if self._epsilon <= 0 or math.isclose(self._epsilon, 0.0):
            raise ValueError(
                "The epsilon value must be greater than zero beyond floating point."
            )

        # Infer parameters from the observations
        self._covariate_categories = _infer_covariate_categories_from_observations(
            self.observations, self._covariates, self._covariate_categories
        )
        self._date_ranges = _infer_ranges_from_observations(
            self.observations, self._date_ranges, "date"
        )
        self._season_ranges = _infer_ranges_from_observations(
            self.observations, self._season_ranges, "season"
        )

    def coordinates(self) -> dict[str, list[str]]:
        """
        Get the coordinates for the uptake model.

        Returns:
            The coordinates for the uptake model.

        """
        coords = {}
        coords["season"] = [season.season for season in self._season_ranges]
        for season_range in self._season_ranges:
            n_days = (season_range.end_date - season_range.start_date).days + 1
            coords[_coord_name("season", season_range.season, "dates")] = [
                (season_range.start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(n_days)
            ]
        covariate_names = []
        for covariate_category in self._covariate_categories:
            covariate_names.append(covariate_category.covariate)
            categories = list(covariate_category.categories)
            coords[
                _coord_name("covariate", covariate_category.covariate, "categories")
            ] = categories
            coords[
                _coord_name(
                    "covariate", covariate_category.covariate, "categories", "limited"
                )
            ] = categories[1:]
        coords["covariate_names"] = covariate_names
        coords["parameters"] = list(self.curve.parameters)
        category_combinations = list(
            itertools.product(
                *[
                    coords[_coord_name("covariate", covariate_name, "categories")]
                    for covariate_name in coords["covariate_names"]
                ]
            )
        )
        coords["season_by_category"] = []
        for season_range in self._season_ranges:
            for category_combo in category_combinations:
                coords["season_by_category"].append(
                    _coord_name(
                        *(
                            ["season", season_range.season]
                            + [
                                item
                                for pair in zip(
                                    coords["covariate_names"], category_combo
                                )
                                for item in pair
                            ]
                        )
                    )
                )
        return coords

    # TODO: Split this method into smaller parts to address noqa issues.
    def build(self, debug: bool = False) -> Self:  # noqa: PLR0912, PLR0915
        """
        Build the uptake model.

        Args:
            debug: Whether to output debugging information as the model is built.

        Returns:
            The uptake model instance for chaining.

        """
        # Get the logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if debug else logging.CRITICAL + 1)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(stream_handler)

        # Preprocessing data
        logger.info(
            "Using %u date ranges for the uptake model.", len(self._date_ranges)
        )
        if self.observations is not None:
            logger.info(
                "Using %u observational date ranges for the uptake model.",
                len(self.observations),
            )
        else:
            logger.error(
                "No observations were provided, will not be able "
                "to calibrate model only sample from prior."
            )

        # Build the model
        coords = self.coordinates()
        self._model = pm.Model(coords=coords)
        with self._model:
            # Create the date indexes for each season
            date_indexes = {}
            for season_range in self._season_ranges:
                n_days = (season_range.end_date - season_range.start_date).days + 1
                date_indexes[season_range.season] = pm.Data(
                    _pm_name("season", season_range.season, "dates"), range(n_days)
                )
                logger.info(
                    "Added season %s to the model with %u days.",
                    season_range.season,
                    n_days,
                )

            # Create the covariate parameters
            params: dict[int, tuple[pm.Distribution, tuple[str, ...]]] = {}
            for covariate in self._covariates:
                name = _pm_name(covariate.parameter, covariate.covariate or "Season")
                params[hash((covariate.parameter, covariate.covariate))] = (
                    covariate.pymc_distribution(name, coords)
                )
                logger.info("Added covariate %s to the model.", name)

            # Create the summed parameters
            summed_params: dict[str, pm.Distribution] = {}
            category_combinations = list(
                itertools.product(
                    *[
                        coords[_coord_name("covariate", covariate_name, "categories")]
                        for covariate_name in coords["covariate_names"]
                    ]
                )
            )
            for category_combo in category_combinations:
                for season_range in self._season_ranges:
                    for param in coords["parameters"]:
                        param_components: list[pm.Data | pm.Deterministic] = []
                        result = params.get(hash((param, None)))
                        if result is not None:
                            dist, dims = result
                            if (
                                idx := _coord_index(
                                    dims, season_range.season, None, None, coords
                                )
                            ) is not None:
                                param_components.append(dist[idx])
                        for i, covariate_name in enumerate(coords["covariate_names"]):
                            result = params.get(hash((param, covariate_name)))
                            if result is None:
                                continue
                            dist, dims = result
                            if (
                                idx := _coord_index(
                                    dims,
                                    season_range.season,
                                    covariate_name,
                                    category_combo[i],
                                    coords,
                                )
                            ) is not None:
                                param_components.append(dist[idx])
                        name_args = [param, "season", season_range.season] + [
                            item
                            for pair in zip(coords["covariate_names"], category_combo)
                            for item in pair
                        ]
                        name = _pm_name(*name_args)
                        if param_components:
                            summed_params[name] = pm.Deterministic(
                                name, sum(param_components)
                            )
                        else:
                            summed_params[name] = pm.Data(name, 0.0)
                        logger.info("Added summed parameter %s to the model.", name)

            # Sample noise shape for each time series from exponential hyperprior
            epsilon = pm.Exponential(
                "epsilon", lam=1.0 / self._epsilon, dims=("season_by_category",)
            )

            # Generate the incidence time series at a daily level
            incidence = {}
            for category_combo in category_combinations:
                pm_cov_names = [
                    item
                    for pair in zip(coords["covariate_names"], category_combo)
                    for item in pair
                ]
                for season_range in self._season_ranges:
                    kwargs = {
                        param: summed_params[
                            _pm_name(
                                *[param, "season", season_range.season, *pm_cov_names]
                            )
                        ]
                        for param in coords["parameters"]
                    }
                    name_args = ["incidence", season_range.season, *pm_cov_names]
                    incidence[_coord_name(*name_args)] = pm.Gamma(
                        _pm_name(*name_args),
                        mu=self.curve.prevalence_difference(
                            date_indexes[season_range.season],
                            date_indexes[season_range.season] + 1.0,
                            **kwargs,
                        ),
                        sigma=epsilon[
                            coords["season_by_category"].index(
                                _coord_name(
                                    "season", season_range.season, *pm_cov_names
                                )
                            )
                        ],
                        dims=(_coord_name("season", season_range.season, "dates"),),
                    )
                    # Use custom potential to ensure prevalence is constrained to [0, 1]
                    prevalence = pt.cumsum(incidence[_coord_name(*name_args)])
                    constraint = (
                        pm.math.ge(prevalence, 0.0) * pm.math.le(prevalence, 1.0)
                    ).all()
                    pm.Potential(
                        _pm_name(*name_args, "prevalence", "constraint"),
                        pm.math.log(pm.math.switch(constraint, 1.0, 0.0)),
                    )

            # If observations are provided, add likelihoods
            if self.observations is not None:
                for obs_idx, row in self.observations.iterrows():
                    start_index = coords[
                        _coord_name("season", row["season"], "dates")
                    ].index(row["start_date"].strftime("%Y-%m-%d"))
                    end_index = coords[
                        _coord_name("season", row["season"], "dates")
                    ].index(row["end_date"].strftime("%Y-%m-%d"))
                    if row["type"] == "incidence":
                        incidence_name = [
                            "incidence",
                            row["season"],
                            *[
                                item
                                for cov_name in coords["covariate_names"]
                                for item in [cov_name, row[cov_name]]
                            ],
                        ]
                        incidence_series = incidence[_coord_name(*incidence_name)]
                        # PyMC does not allow for directly observing the sum of random
                        # vars, see
                        # https://discourse.pymc.io/t/sum-of-random-variables/1652.
                        # Instead, use a normal distribution with a very small standard
                        # deviation to approximate the sum.
                        pm.Normal(
                            name=_pm_name("observation", str(obs_idx)),
                            mu=incidence_series[start_index:end_index].sum(),
                            sigma=self._observation_sigma,
                            observed=row["value"],
                        )

        return self

    def sample_prior(
        self,
        draws: int = 500,
        var_names: Iterable[str] | None = None,
        random_seed: Any = 1,
        idata_kwargs: dict[str, Any] | None = None,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """
        Sample from the prior distribution of the model.

        See Also:
            [pymc.sample_prior_predictive](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html)

        """
        if self._model is None:
            raise AttributeError(
                "The `build` method must be called before `sample_prior`."
            )
        with self._model:
            prior = pm.sample_prior_predictive(
                draws=draws,
                var_names=var_names,
                random_seed=random_seed,
                return_inferencedata=True,
                idata_kwargs=idata_kwargs,
                compile_kwargs=compile_kwargs,
            )
        return prior

    def sample(
        self, random_seed: Any = 1, nuts_sampler: str = "blackjax", **kwargs: Any
    ) -> az.InferenceData:
        """
        Sample from the posterior distribution of the model.

        Args:
            random_seed: The random seed to use for sampling.
            nuts_sampler: The NUTS sampler to use.
            kwargs: Further keyword arguments to pass to the sampler. See
                [pymc.sample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)
                for more details.

        Notes:
            PyMC's built-in NUTS sampler has trouble compiling this model, so we use
            BlackJAX's NUTS sampler instead. This requires the `blackjax` package to be
            installed.

        Returns:
            The posterior samples as an InferenceData object.

        Raises:
            AttributeError: If the `build` method has not been called before this
                method.

        """
        if self._model is None:
            raise AttributeError("The `build` method must be called before `sample`.")
        with self._model:
            trace = pm.sample(
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                return_inferencedata=True,
                **kwargs,
            )
        return trace

    def add_observations(self, observations: pd.DataFrame) -> "SeasonalUptakeModel":
        """
        Create a copy of the uptake model with added observations.

        Args:
            observations: The observations to add to the model.

        Returns:
            A new instance of the uptake model with the added observations.

        """
        if self.observations is not None:
            raise ValueError("Observations have already been added to the model.")
        return SeasonalUptakeModel(
            curve=self.curve,
            covariates=self._covariates,
            observations=observations,
            covariate_categories=self._covariate_categories,
            season_ranges=self._season_ranges,
            date_ranges=self._date_ranges,
            epsilon=self._epsilon,
            name=self.name,
        )
