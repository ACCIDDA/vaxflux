"""Tools for constructing uptake models."""

__all__ = (
    "Covariate",
    "DateRange",
    "PooledCovariate",
    "SeasonalUptakeModel",
    "SeasonRange",
)


from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
from datetime import date, datetime
import sys
from typing import Any, Literal, NamedTuple, TypeVar, cast

import arviz as az
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pydantic import BaseModel, ConfigDict
import pymc as pm

from vaxflux.curves import IncidenceCurve


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


class Covariate(ABC, BaseModel):
    """
    Abstract class for the implementation of covariates in uptake models.

    Attributes:
        parameter: The parameter from the incidence curve family this covariate applies
            to.
        covariate: The name of the covariate in the dataset or `None` if the covariate
            is a seasonal effect.
    """

    parameter: str
    covariate: str | None

    @abstractmethod
    def pymc_distribution(
        self, name: str, coords: dict[str, list[str]]
    ) -> pm.Distribution:
        """
        Return a PyMC3 distribution for the covariate.

        Args:
            name: The name of the distribution.
            coords: The coordinates for the uptake model.
        """
        raise NotImplementedError


class PooledCovariate(Covariate):
    """
    A pooled covariate for uptake models.

    Attributes:
        distribution: The PyMC3 distribution to use for the covariate.
        distribution_kwargs: The keyword arguments to pass to the PyMC3 distribution.
    """

    distribution: str
    distribution_kwargs: dict[str, Any]

    def pymc_distribution(
        self, name: str, coords: dict[str, list[str]]
    ) -> pm.Distribution:
        """
        Return a PyMC3 distribution for the pooled covariate.

        Args:
            name: The name of the distribution.
            coords: The coordinates for the uptake model.
        """
        return getattr(pm, self.distribution)(
            name=name,
            **self.distribution_kwargs,
            shape=coords["season"],
        )


class SeasonRange(BaseModel):
    """
    A representation of a season range for uptake scenarios.

    Attributes:
        season: The name of the season for the season range.
        start_date: The start date of the season range, used to make seasonal dates
            relative.
        end_date: The end date of the season range, used for validation if given.

    """

    model_config = ConfigDict(frozen=True)

    season: str
    start_date: date
    end_date: date | None = None


class DateRange(BaseModel):
    """
    A representation of a date range for uptake scenarios.

    Attributes:
        season: The season for the date range.
        start_date: The start date of the date range.
        end_date: The end date of the date range.
        report_date: The report date of the date range.

    """

    model_config = ConfigDict(frozen=True)

    season: str
    start_date: date
    end_date: date
    report_date: date


class ObservationDateRow(NamedTuple):
    season: str
    start_date: datetime
    end_date: datetime
    report_date: datetime


class ObservationSeasonRow(NamedTuple):
    season: str
    season_start_date: datetime
    season_end_date: datetime


DateOrSeasonRange = TypeVar("DateOrSeasonRange", bound=DateRange | SeasonRange)


def _infer_ranges_from_observations(
    observations: pd.DataFrame | None,
    ranges: list[DateOrSeasonRange],
    mode: Literal["date", "season"],
) -> list[DateOrSeasonRange]:
    """
    Infer the date or season ranges from the observations.

    Args:
        observations: The uptake dataset to use.
        ranges: The date or season ranges for the uptake scenarios.
        mode: The mode of the inference, either "date" or "season".

    Returns:
        The inferred date or season ranges, depending on the `mode`.

    Raises:
        ValueError: If both `observations` and `ranges` are empty.
        ValueError: If the required columns are missing in the observations.
        ValueError: If the observed season ranges are not consistent with the explicit
            season ranges, only applicable for the season mode.
    """
    if not observations and not ranges:
        raise ValueError("At least one of `observations` or `ranges` is required.")
    cls = DateRange if mode == "date" else SeasonRange
    columns = {
        "date": {"season", "start_date", "end_date", "report_date"},
        "season": {"season", "season_start_date", "season_end_date"},
    }[mode]
    if observations:
        # Only observations
        if not ranges:
            if missing_columns := columns - set(observations.columns):
                raise ValueError(
                    f"Missing required columns in the observations: {missing_columns}."
                )
            observations_ranges = (
                observations[list(columns)]
                .drop_duplicates(ignore_index=True)
                .sort_values(list(columns), ignore_index=True)
            )
            return [
                cls(row._asdict())  # type: ignore
                for row in observations_ranges.itertuples(
                    index=False, name=f"Observation{mode.capitalize()}Row"
                )
            ]
        # Both ranges and observations
        if columns.issubset(observations.columns):
            observation_ranges: list[DateOrSeasonRange] = (
                _infer_ranges_from_observations(observations, [], mode)
            )
            if mode == "date":
                return list(set(ranges) | set(observation_ranges))
            if non_explicit_ranges := set(observation_ranges) - set(ranges):
                non_explicit_season_ranges = cast(set[SeasonRange], non_explicit_ranges)
                season_names = {season.season for season in non_explicit_season_ranges}
                raise ValueError(
                    "The observed season ranges are not consistent with the "
                    f"explicit season ranges, not accounting for: {season_names}."
                )
    # Only ranges
    return ranges


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
        return coords

    def build(self, debug: bool = False) -> Self:
        """
        Build the uptake model.

        Args:
            debug: Whether to output debugging information as the model is built.

        Returns:
            The uptake model instance for chaining.

        """
        self._model = pm.Model()
        with self._model:
            raise NotImplementedError
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
                return_idata=True,
                idata_kwargs=idata_kwargs,
                compile_kwargs=compile_kwargs,
            )
        return prior
