"""Tools for constructing uptake models."""

__all__ = ("SeasonalUptakeModel",)


import copy
import itertools
import logging
import math
import sys
from collections.abc import Iterable
from datetime import timedelta
from typing import Any, cast

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
from vaxflux.curves import Curve
from vaxflux.dates import DateRange, SeasonRange, _infer_ranges_from_observations
from vaxflux.interventions import (
    Implementation,
    Intervention,
    _check_interventions_and_implementations,
)

if sys.version_info[:2] >= (3, 11):
    from typing import Self
else:
    Self = Any


class SeasonalUptakeModel:
    """
    A generic class for uptake models.

    Attributes:
        curve: The curve family to use.
        name: The name of the model, optional and only used for display.
        observations: The uptake dataset to use.

    """

    def __init__(  # noqa: PLR0913
        self,
        curve: Curve,
        covariates: list[Covariate],
        observations: pd.DataFrame | None = None,
        covariate_categories: list[CovariateCategories] | None = None,
        interventions: list[Intervention] | None = None,
        implementations: list[Implementation] | None = None,
        season_ranges: list[SeasonRange] | None = None,
        date_ranges: list[DateRange] | None = None,
        epsilon: float = 5e-4,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an uptake model.

        Args:
            curve: The curve family to use.
            covariates: The covariates to use.
            observations: The uptake dataset to use.
            covariate_categories: The covariate categories for the uptake scenarios or
                `None` to derive them from the observations.
            interventions: A list of intervention modifiers to apply to the model or
                `None` to not apply interventions.
            implementations: A list of implementation modifiers to apply to the model
                or `None` to not apply implementations.
            date_ranges: The date ranges for the uptake scenarios or `None` to derive
                them from the observations.
            season_ranges: The season ranges for the uptake scenarios or `None` to
                derive them from the observations.
            epsilon: The prior average for the standard deviation in observed uptake.
            name: The name of the model, optional and only used for display.
            kwargs: Further keyword arguments to pass to the model.

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
        self._epsilon = epsilon
        self._implementations = (
            copy.deepcopy(implementations) if implementations else list()
        )
        self._interventions = copy.deepcopy(interventions) if interventions else list()
        self._kwargs = copy.deepcopy(kwargs)
        self._model: pm.Model | None = None
        self._season_ranges: list[SeasonRange] = (
            copy.deepcopy(season_ranges) if season_ranges else list()
        )
        self._trace: az.InferenceData | None = None

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
        _check_interventions_and_implementations(
            self._interventions,
            self._implementations,
            self._season_ranges,
            self._covariate_categories,
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
        coords["intervention_names"] = [
            intervention.name for intervention in self._interventions
        ]
        for intervention in self._interventions:
            implementation_coord_name = _coord_name(
                "intervention", intervention.name, "implementations"
            )
            coords[implementation_coord_name] = []
            for category_combo in category_combinations:
                for season_range in self._season_ranges:
                    for implementation in self._implementations:
                        if implementation.season == season_range.season and all(
                            (implementation.covariate_categories or {}).get(
                                covariate, category
                            )
                            == category
                            for covariate, category in zip(
                                covariate_names, category_combo
                            )
                        ):
                            coords[implementation_coord_name].append(
                                _coord_name(
                                    *(
                                        [
                                            "implementation",
                                            intervention.name,
                                            "season",
                                            season_range.season,
                                            *[
                                                item
                                                for pair in zip(
                                                    coords["covariate_names"],
                                                    category_combo,
                                                )
                                                for item in pair
                                            ],
                                        ]
                                    )
                                )
                            )
        return coords

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
                dist, dims = covariate.pymc_distribution(name, coords)
                params[hash((covariate.parameter, covariate.covariate))] = (dist, dims)
                logger.info(
                    "Added covariate %s to the model with shape %s.",
                    name,
                    dist.shape.eval(),
                )

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
                                logger.info(
                                    "Added season component to parameter %s "
                                    "for season %s from index %s of %s.",
                                    param,
                                    season_range.season,
                                    str(idx),
                                    str(dist),
                                )
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
                                logger.info(
                                    "Added %s, %s covariate component to parameter "
                                    "%s for season %s from index %s of %s with "
                                    "dimensions %s.",
                                    covariate_name,
                                    category_combo[i],
                                    param,
                                    season_range.season,
                                    str(idx),
                                    str(dist),
                                    str(dims),
                                )
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
                            logger.info(
                                "Added summed parameter %s to "
                                "the model with %u components.",
                                name,
                                len(param_components),
                            )
                        else:
                            summed_params[name] = pm.Data(name, 0.0)
                            logger.info(
                                "Added summed parameter %s to the "
                                "model with no components.",
                                name,
                            )

            # Add intervention implementations to the model
            interventions: dict[str, pm.Distribution] = {}
            for intervention in self._interventions:
                name = _pm_name("intervention", intervention.name, "implementations")
                dist, dims = intervention.pymc_distribution(name)
                interventions[intervention.name] = dist
                logger.info(
                    "Added intervention %s to the model with shape %s.",
                    name,
                    str(dist.shape.eval()),
                )

            # Sample noise shape for each time series from exponential hyperprior
            if self._kwargs.get("pooled_epsilon", False):
                # If pooled epsilon is used, use a single epsilon for all time series
                epsilon = pm.Exponential("epsilon", lam=1.0 / self._epsilon, shape=1)
            else:
                # If not pooled, use a different epsilon for each time series
                epsilon = pm.Exponential(
                    "epsilon", lam=1.0 / self._epsilon, dims=("season_by_category",)
                )
            logger.info(
                "Added epsilon to the model with shape %s.", epsilon.shape.eval()
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
                    season_dates_coord_name = _coord_name(
                        "season", season_range.season, "dates"
                    )
                    season_dates = coords[season_dates_coord_name]
                    kwargs = {
                        param: summed_params[
                            _pm_name(
                                *[param, "season", season_range.season, *pm_cov_names]
                            )
                        ]
                        for param in coords["parameters"]
                    }

                    # Determine modifiers to the curve kwargs due to interventions
                    modified_kwargs: dict[
                        str, list[pt.variable.TensorVariable[Any, Any]]
                    ] = {}
                    for param in coords["parameters"]:
                        for intervention in self._interventions:
                            if intervention.parameter == param:
                                for implementation in self._implementations:
                                    if (
                                        implementation.intervention == intervention.name
                                        and all(
                                            (
                                                implementation.covariate_categories
                                                or {}
                                            ).get(covariate, category)
                                            == category
                                            for covariate, category in zip(
                                                coords["covariate_names"],
                                                category_combo,
                                            )
                                        )
                                    ):
                                        if param not in modified_kwargs:
                                            modified_kwargs[param] = []
                                        implementation_coord_name = _coord_name(
                                            *(
                                                [
                                                    "implementation",
                                                    intervention.name,
                                                    "season",
                                                    season_range.season,
                                                    *[
                                                        item
                                                        for pair in zip(
                                                            coords["covariate_names"],
                                                            category_combo,
                                                        )
                                                        for item in pair
                                                    ],
                                                ]
                                            )
                                        )
                                        implementation_idx = coords[
                                            _coord_name(
                                                "intervention",
                                                intervention.name,
                                                "implementations",
                                            )
                                        ].index(implementation_coord_name)
                                        var = interventions[intervention.name][
                                            implementation_idx
                                        ]
                                        modified_kwargs[param].append(
                                            pm.math.switch(
                                                [
                                                    (
                                                        implementation.start_date
                                                        is None
                                                        or i
                                                        >= season_dates.index(
                                                            season_range.start_date.strftime(
                                                                "%Y-%m-%d"
                                                            )
                                                        )
                                                    )
                                                    and (
                                                        implementation.start_date
                                                        is None
                                                        or i
                                                        <= season_dates.index(
                                                            season_range.start_date.strftime(
                                                                "%Y-%m-%d"
                                                            )
                                                        )
                                                    )
                                                    for i in range(len(season_dates))
                                                ],
                                                0.0,
                                                var,
                                            )
                                        )

                    name_args = ["incidence", season_range.season, *pm_cov_names]
                    eps = (
                        epsilon
                        if self._kwargs.get("pooled_epsilon", False)
                        else epsilon[
                            coords["season_by_category"].index(
                                _coord_name(
                                    "season", season_range.season, *pm_cov_names
                                )
                            )
                        ]
                    )
                    evaled_kwargs = {
                        param: sum(modified_kwargs.get(param, [])) + kwargs[param]
                        for param in kwargs
                    }
                    incidence[_coord_name(*name_args)] = pm.Gamma(
                        _pm_name(*name_args),
                        mu=self.curve.prevalence_difference(
                            date_indexes[season_range.season],
                            date_indexes[season_range.season] + 1.0,
                            **evaled_kwargs,
                        ),
                        sigma=eps,
                        dims=(_coord_name("season", season_range.season, "dates"),),
                    )
                    logger.info(
                        "Added gamma distributed incidence %s "
                        "to the model with shape %s.",
                        _pm_name(*name_args),
                        str(incidence[_coord_name(*name_args)].shape.eval()),
                    )
                    # Use custom potential to ensure prevalence is constrained to [0, 1]
                    if self._kwargs.get("constrain_prevalence", True):
                        prevalence = pt.cumsum(incidence[_coord_name(*name_args)])  # type: ignore[no-untyped-call]
                        constraint = (
                            pm.math.ge(prevalence, 0.0) * pm.math.le(prevalence, 1.0)
                        ).all()
                        prevalence_constraint_name = _pm_name(
                            *name_args, "prevalence", "constraint"
                        )
                        pm.Potential(
                            prevalence_constraint_name,
                            pm.math.log(pm.math.switch(constraint, 1.0, 0.0)),
                        )
                        logger.info(
                            "Added prevalence constraint %s.",
                            prevalence_constraint_name,
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
                    logger.info(
                        "Observation index %u is %u in length.",
                        obs_idx,
                        end_index - start_index,
                    )
                    if row["type"] == "incidence":
                        pm_cov_names = [
                            item
                            for cov_name in coords["covariate_names"]
                            for item in [cov_name, row[cov_name]]
                        ]
                        incidence_name = ["incidence", row["season"], *pm_cov_names]
                        incidence_series = incidence[_coord_name(*incidence_name)]
                        # PyMC does not allow for directly observing the sum of
                        # random vars, see
                        # https://discourse.pymc.io/t/sum-of-random-variables/1652.
                        # Instead, use a normal distribution with a very small
                        # standard deviation to approximate the sum.
                        pm.Normal(
                            name=_pm_name("observation", str(obs_idx)),
                            mu=incidence_series[start_index : (end_index + 1)].sum(),
                            sigma=self._kwargs.get("observation_sigma", 1.0e-9),
                            observed=row["value"],
                        )
                        logger.info(
                            "Added observation %u to the model with approx normal "
                            "likelihood %s from %u and %u of %s time series.",
                            obs_idx,
                            _pm_name("observation", str(obs_idx)),
                            start_index,
                            end_index,
                            _pm_name(*incidence_name),
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
        return cast(az.InferenceData, prior)

    def sample(
        self, random_seed: Any = 1, nuts_sampler: str = "blackjax", **kwargs: Any
    ) -> Self:
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
            self._trace = pm.sample(
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                return_inferencedata=True,
                **kwargs,
            )
        return self

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
            **self._kwargs,
        )

    def dataframe(self) -> pd.DataFrame:
        """
        Get the posterior incidence data as a pandas DataFrame.

        Returns:
            A pandas DataFrame with the posterior incidence data, contains the columns
            'draw', 'chain', 'season', 'date', 'type', 'value, and the covariate names.

        Raises:
            AttributeError: If the `sample` method has not been called before this
                method.

        """
        if self._trace is None:
            raise AttributeError(
                "The `sample` method must be called before `dataframe`."
            )
        coords = self.coordinates()
        incidence_dataframes: list[pd.DataFrame] = []
        category_combinations = list(
            itertools.product(
                *[
                    coords[_coord_name("covariate", covariate_name, "categories")]
                    for covariate_name in coords["covariate_names"]
                ]
            )
        )
        for category_combo in category_combinations:
            pm_cov_names = [
                item
                for pair in zip(coords["covariate_names"], category_combo)
                for item in pair
            ]
            for season_range in self._season_ranges:
                name_args = ["incidence", season_range.season, *pm_cov_names]
                incidence_name = _pm_name(*name_args)
                tmp_df = (
                    self._trace.posterior[incidence_name]  # type: ignore[attr-defined]
                    .to_dataframe()
                    .reset_index()
                )
                tmp_df = tmp_df.rename(
                    columns={
                        _coord_name("season", season_range.season, "dates"): "date",
                        incidence_name: "value",
                    }
                )
                tmp_df["season"] = pd.Series(
                    len(tmp_df) * [season_range.season], dtype="string"
                )
                tmp_df["type"] = pd.Series(len(tmp_df) * ["incidence"], dtype="string")
                for cov_name, cov_value in zip(
                    coords["covariate_names"], category_combo
                ):
                    tmp_df[cov_name] = pd.Series(
                        len(tmp_df) * [cov_value], dtype="string"
                    )
                tmp_df["date"] = pd.to_datetime(tmp_df["date"])
                tmp_df = tmp_df[
                    ["draw", "chain", "season", "date"]
                    + coords["covariate_names"]
                    + ["type", "value"]
                ]
                incidence_dataframes.append(tmp_df)
        incidence_df = pd.concat(incidence_dataframes)
        return incidence_df
