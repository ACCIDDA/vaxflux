"""Tools for constructing uptake models."""

__all__ = ("SeasonalUptakeModel",)


from collections.abc import Iterable
import copy
import sys
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm

from vaxflux.curves import IncidenceCurve


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

    def season_column(self, season: str) -> Self:
        """
        Set the season column for the uptake model.

        Args:
            season: The season column in the dataset.

        Returns:
            The uptake model instance for chaining.

        """
        self._season = season
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
