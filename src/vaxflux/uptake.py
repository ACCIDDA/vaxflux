"""Tools for constructing uptake models."""

__all__ = ("SeasonalUptakeModel",)


from collections.abc import Iterable
import copy
from datetime import timedelta
import itertools
import logging
import sys
from typing import Any

import arviz as az
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import pymc as pm

from vaxflux._util import _coord_index, _coord_name, _pm_name
from vaxflux.covariates import (
    Covariate,
    CovariateCategories,
    _infer_covariate_categories_from_observations,
)
from vaxflux.curves import IncidenceCurve
from vaxflux.dates import DateRange, SeasonRange, _infer_ranges_from_observations


if sys.version_info >= (3, 11):
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

    def __init__(
        self,
        curve: IncidenceCurve,
        covariates: list[Covariate],
        observations: pd.DataFrame | None = None,
        covariate_categories: list[CovariateCategories] | None = None,
        season_ranges: list[SeasonRange] | None = None,
        date_ranges: list[DateRange] | None = None,
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
        self.observations = None if observations is None else observations.copy()
        if self.observations:
            for col in {"start_date", "end_date", "report_date"}.intersection(
                self.observations.columns
            ):
                if not is_datetime64_any_dtype(self.observations[col]):
                    self.observations[col] = pd.to_datetime(self.observations[col])

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

        # Input validation
        if covariate_parameters_missing := {
            covariate.parameter for covariate in self._covariates
        } - set(self.curve.parameters):
            raise ValueError(
                "The following parameters were given in the covariates, "
                "but not present in the curve family parameters: "
                f"{sorted(covariate_parameters_missing)}."
            )
        if self.observations and (
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
        return coords

    # TODO: Split this method into smaller parts to address noqa issues.
    def build(self, debug: bool = False) -> Self:  # noqa: PLR0912
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
        if self.observations:
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

            params: dict[int, tuple[pm.Distribution, tuple[str, ...]]] = {}
            for covariate in self._covariates:
                name = _pm_name(covariate.parameter, covariate.covariate or "Season")
                params[hash((covariate.parameter, covariate.covariate))] = (
                    covariate.pymc_distribution(name, coords)
                )
                logger.info("Added covariate %s to the model.", name)

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
                                param_components.append(dist[*idx])
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
                                param_components.append(dist[*idx])
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
                                *([param, "season", season_range.season] + pm_cov_names)
                            )
                        ]
                        for param in coords["parameters"]
                    }
                    name_args = ["incidence", season_range.season] + pm_cov_names
                    name = _pm_name(*name_args)
                    incidence[season_range.season] = pm.Deterministic(
                        name,
                        self.curve.prevalence_difference(
                            date_indexes[season_range.season],
                            date_indexes[season_range.season] + 1.0,
                            **kwargs,
                        ),
                        dims=(_coord_name("season", season_range.season, "dates"),),
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
