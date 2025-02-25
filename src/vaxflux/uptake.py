"""Tools for constructing uptake models."""

__all__ = (
    "Covariate",
    "DateRange",
    "PooledCovariate",
    "ScenarioDateRanges",
    "ScenarioSeasonRanges",
    "SeasonalUptakeModel",
    "SeasonRange",
)


from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import copy
from datetime import date
import sys
from typing import Any

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


class ScenarioSeasonRanges(BaseModel):
    """
    A representation of season ranges for uptake scenarios.

    Attributes:
        season_ranges: The set of season ranges for the uptake scenarios.
    """

    model_config = ConfigDict(frozen=True)

    season_ranges: set[SeasonRange]


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


class ScenarioDateRanges(BaseModel):
    """
    A representation of date ranges for uptake scenarios.

    Attributes:
        date_ranges: The set of date ranges for the uptake scenarios.
    """

    model_config = ConfigDict(frozen=True)

    date_ranges: set[DateRange]

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        season: str = "season",
        start_date: str = "start_date",
        end_date: str = "end_date",
        report_date: str = "report_date",
    ) -> Self:
        """
        Construct a set of date ranges for uptake scenarios from a DataFrame.

        Args:
            df: The DataFrame containing the date ranges.
            season: The column containing the season.
            start_date: The column containing the start date.
            end_date: The column containing the end date.
            report_date: The column containing the report date.

        Returns:
            The date ranges for the uptake scenarios.

        Raises:
            ValueError: If a required column is not found in the DataFrame.

        """
        for arg in ("season", "start_date", "end_date", "report_date"):
            if (col := locals().get(arg)) not in df.columns:
                raise ValueError(
                    f"Column '{col}' for `{arg}` not found in the DataFrame."
                )
        date_columns = [start_date, end_date, report_date]
        copied = False
        for col in date_columns:
            if not is_datetime64_any_dtype(df[col]):
                df = df if copied else df.copy()
                copied = True
                df[col] = pd.to_datetime(df[col])
        return cls(
            date_ranges={
                DateRange(
                    season=str(getattr(row, season)),
                    start_date=getattr(row, start_date).date(),
                    end_date=getattr(row, end_date).date(),
                    report_date=getattr(row, report_date).date(),
                )
                for row in df.itertuples(index=False)
            }
        )

    def union(self, other: "ScenarioDateRanges") -> "ScenarioDateRanges":
        """
        Compute the union of two sets of date ranges.

        Args:
            other: The other set of date ranges to compute the union with.

        Returns:
            The union of the two sets of date ranges.

        """
        return ScenarioDateRanges(
            date_ranges=set(self.date_ranges) | set(other.date_ranges)
        )


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
        season_ranges: Sequence[SeasonRange] | None = None,
        date_ranges: Sequence[DateRange] | None = None,
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

        # Private attributes
        self._covariates = copy.deepcopy(covariates)
        self._date_ranges: Sequence[DateRange] = (
            copy.deepcopy(date_ranges) if date_ranges else tuple()
        )
        self._model: pm.Model | None = None
        self._season_ranges: Sequence[SeasonRange] = (
            copy.deepcopy(season_ranges) if season_ranges else tuple()
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
        if not self.observations and not self._date_ranges:
            raise ValueError(
                "At least one of `observations` or `date_ranges` is required."
            )
        if not self.observations and not self._season_ranges:
            raise ValueError(
                "At least one of `observations` or `season_ranges` is required."
            )

    def coordinates(self) -> dict[str, list[str]]:
        """
        Get the coordinates for the uptake model.

        Returns:
            The coordinates for the uptake model.

        """
        return {}

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
