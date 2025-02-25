"""Tools for constructing uptake models."""

__all__ = (
    "DateRange",
    "ScenarioDateRanges",
    "SeasonalUptakeModel",
)


from collections.abc import Iterable
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
    """A representation of date ranges for uptake scenarios."""

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
        dataset: The uptake dataset to use.

    """

    curve: IncidenceCurve | None = None
    dataset: pd.DataFrame | None = None
    name: str | None = None

    def __init__(
        self,
        curve: IncidenceCurve,
        dataset: pd.DataFrame | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize an uptake model.

        Args:
            curve: The incidence curve family to use.
            dataset: The uptake dataset to use.
            name: The name of the model, optional and only used for display.

        Returns:
            None

        """
        # Public attributes
        self.curve = copy.deepcopy(curve)
        if dataset is not None:
            self.dataset = dataset.copy()
        self.name = name
        # Private attributes
        self._covariates: list[str] | None = None
        self._doses_administered: str | None = None
        self._model: pm.Model | None = None
        self._scenario_date_ranges: ScenarioDateRanges | None = None
        self._seasonal_parameters: dict[
            str, tuple[pm.Distribution, dict[str, Any], bool]
        ] = {}
        self._series: str | None = None

    def set_scenario_date_ranges(
        self, scenario_date_ranges: ScenarioDateRanges
    ) -> Self:
        """
        Set the scenario date ranges for the uptake model.

        Args:
            scenario_date_ranges: The scenario date ranges to set.

        Returns:
            The uptake model instance for chaining.

        """
        self._scenario_date_ranges = scenario_date_ranges
        return self

    def set_seasonal_parameter(
        self,
        parameter: str,
        distribution: pm.Distribution,
        distribution_kwargs: dict[str, Any],
        hierarchical: bool = True,
    ) -> Self:
        """
        Set a seasonal parameter for the uptake model.

        Args:
            parameter: The parameter to set.
            distribution: The distribution to use for the parameter.
            distribution_kwargs: The keyword arguments for the distribution.
            hierarchical: Whether the parameter is hierarchical.

        Returns:
            The uptake model instance for chaining.

        Raises:
            ValueError: If the parameter is not used by the curve family.

        """
        if self.curve is not None and parameter not in self.curve.parameters:
            raise ValueError(
                f"The parameter '{parameter}' is not used by the curve family."
            )
        self._seasonal_parameters[parameter] = (
            distribution,
            distribution_kwargs,
            hierarchical,
        )
        return self

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
