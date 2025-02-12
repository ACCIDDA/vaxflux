"""Tools for constructing uptake models."""

__all__ = (
    "DateRange",
    "ScenarioDateRanges",
    "SeasonalUptakeModel",
)


from collections.abc import Iterable, Sequence
import copy
from datetime import date
import sys
from typing import Any, NamedTuple

import arviz as az
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import pymc as pm

from vaxflux.curves import IncidenceCurve


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


class DateRange(NamedTuple):
    """
    A representation of a date range for uptake scenarios.

    Attributes:
        season: The season for the date range.
        start_date: The start date of the date range.
        end_date: The end date of the date range.
        report_date: The report date of the date range.

    """

    season: str
    start_date: date
    end_date: date
    report_date: date


class ScenarioDateRanges:
    """A representation of date ranges for uptake scenarios."""

    def __init__(self, date_ranges: Sequence[DateRange]) -> None:
        """
        Construct a set of date ranges for uptake scenarios.

        Args:
            date_ranges: The date ranges for the uptake scenarios.

        Returns:
            None

        Raises:
            ValueError: If duplicate date ranges are found in the input.

        """
        self._date_ranges = date_ranges
        unique_hashes = {hash(date_range) for date_range in date_ranges}
        if (len_unique_hashes := len(unique_hashes)) < (
            len_date_ranges := len(date_ranges)
        ):
            raise ValueError(
                f"Duplicate date ranges found in the input, given {len_date_ranges} "
                f"date ranges and {len_unique_hashes} unique date ranges."
            )

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
            [
                DateRange(
                    str(getattr(row, season)),
                    getattr(row, start_date).date(),
                    getattr(row, end_date).date(),
                    getattr(row, report_date).date(),
                )
                for row in df.itertuples(index=False)
            ]
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
        self._end_date: str | None = None
        self._model: pm.Model | None = None
        self._report_date: str | None = None
        self._season: str | None = None
        self._season_labels: Sequence[str] | None = None
        self._seasonal_parameters: dict[
            str, tuple[pm.Distribution, dict[str, Any], bool]
        ] = {}
        self._series: str | None = None
        self._start_date: str | None = None

    def date_columns(
        self, start_date: str, end_date: str, report_date: str | None = None
    ) -> Self:
        """
        Set the date columns for the uptake model.

        Args:
            start_date: The start date column in the dataset.
            end_date: The end date column in the dataset if relevant.
            report_date: The report date column in the dataset if relevant.

        Returns:
            The uptake model instance for chaining.

        """
        self._start_date = start_date
        self._end_date = end_date
        self._report_date = report_date
        return self

    def season_column(
        self, season: str, season_labels: Sequence[str] | None = None
    ) -> Self:
        """
        Set the season column for the uptake model.

        Args:
            season: The season column in the dataset.
            season_labels: The labels for the season column or `None` to be inferred
                from the dataset. If given they should be given in in ascending order,
                e.g. `('2021/22', '2022/23')`.

        Returns:
            The uptake model instance for chaining.

        """
        self._season = season
        self._season_labels = season_labels
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
