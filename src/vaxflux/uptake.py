"""Tools for constructing uptake models."""

__all__ = ("SeasonalUptakeModel",)


from collections.abc import Iterable
import copy
from typing import Any, Self

import arviz as az
import pandas as pd
import pymc as pm

from vaxflux.curves import IncidenceCurve


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
        self._date: str | None = None
        self._doses_administered: str | None = None
        self._model: pm.Model | None = None
        self._observation_type: str | None = None
        self._population: str | None = None
        self._rate: str | None = None
        self._report_date: str | None = None
        self._season: str | None = None

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
