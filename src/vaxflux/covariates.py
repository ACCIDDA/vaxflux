"""Functionality for covariates in uptake models."""

__all__ = ("Covariate", "CovariateCategories", "PooledCovariate")


from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
import pymc as pm


class CovariateCategories(BaseModel):
    """
    A representation of the categories for a covariate.

    Attributes:
        covariate: The name of the covariate.
        categories: The categories for the covariate.
    """

    model_config = ConfigDict(frozen=True)

    covariate: str
    categories: tuple[str, ...]

    @field_validator("categories", mode="after")
    @classmethod
    def _are_categories_unique(cls, categories: tuple[str]) -> tuple[str]:
        unique_categories = set()
        for category in categories:
            if category in unique_categories:
                raise ValueError(f"Category '{category}' is not unique.")
            unique_categories.add(category)
        return categories


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
            dims="season",
        )


def _infer_covariate_categories_from_observations(
    observations: pd.DataFrame | None,
    covariates: list[Covariate],
    covariate_categories: list[CovariateCategories],
) -> list[CovariateCategories]:
    """
    Infer the covariate categories from the observations.

    Args:
        observations: The uptake dataset to use if provided.
        covariates: The covariates to use.
        covariate_categories: The covariate categories for the uptake scenarios or
            `None` to derive them from the observations.

    Returns:
        The covariate categories for the uptake scenarios.

    Raises:
        ValueError: If a covariate is not in the observations.
    """
    covariate_names = {
        covariate.covariate
        for covariate in covariates
        if covariate.covariate is not None
    }
    if observations:
        # Only observations
        if not covariate_categories:
            if missing_columns := covariate_names - set(observations.columns):
                raise ValueError(
                    f"Missing required columns in the observations: {missing_columns}."
                )
            covariate_categories = [
                CovariateCategories(
                    covariate=covariate,
                    categories=tuple(sorted(observations[covariate].unique().tolist())),
                )
                for covariate in covariate_names
            ]
            return covariate_categories
        # Observations and covariate categories
        observation_covariate_categories = {
            covariate_category.covariate: covariate_category
            for covariate_category in _infer_covariate_categories_from_observations(
                observations, covariates, []
            )
        }
        if missing_covariates := set(
            covariate_category.covariate for covariate_category in covariate_categories
        ) - set(observation_covariate_categories.keys()):
            raise ValueError(
                f"Missing covariates in the observations: {missing_covariates}."
            )
        for covariate_category in covariate_categories:
            if covariate_category.covariate in observation_covariate_categories:
                observation_categories = observation_covariate_categories[
                    covariate_category.covariate
                ].categories
                if missing_categories := set(observation_categories) - set(
                    covariate_category.categories
                ):
                    raise ValueError(
                        f"Missing categories for '{covariate_category.covariate}' "
                        f"in the covariate categories: {missing_categories}."
                    )
    # Only covariate categories
    return covariate_categories
