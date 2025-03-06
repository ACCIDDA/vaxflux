"""Functionality for covariates in uptake models."""

__all__ = ("Covariate", "CovariateCategories", "PooledCovariate")


from abc import ABC, abstractmethod
from typing import Any

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
