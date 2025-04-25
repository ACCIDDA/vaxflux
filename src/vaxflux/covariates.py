"""Functionality for covariates in uptake models."""

__all__ = (
    "Covariate",
    "CovariateCategories",
    "GaussianCovariate",
    "GaussianRandomWalkCovariate",
    "PooledCovariate",
)


from abc import ABC, abstractmethod
from itertools import product
from typing import Annotated, Any

import pandas as pd
import pymc as pm
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vaxflux._util import ListOfFloats, _coord_name


class CovariateCategories(BaseModel):
    """
    A representation of the categories for a covariate.

    Attributes:
        covariate: The name of the covariate.
        categories: The categories for the covariate.
    """

    model_config = ConfigDict(frozen=True)

    covariate: str
    categories: Annotated[tuple[str, ...], Field(min_length=2)]

    @field_validator("categories", mode="after")
    @classmethod
    def _are_categories_unique(cls, categories: tuple[str]) -> tuple[str]:
        unique_categories = set()
        for category in categories:
            if category in unique_categories:
                raise ValueError(f"Category '{category}' is not unique.")
            unique_categories.add(category)
        return categories


def _covariate_categories_to_dict(
    covariate_categories: list[CovariateCategories],
) -> dict[str, list[str]]:
    """
    Convert the covariate categories to a dictionary.

    Args:
        covariate_categories: The covariate categories to convert.

    Returns:
        A dictionary with the covariate categories, where the keys are the covariate
        names and the values are the categories.

    Raises:
        ValueError: If the covariate names are not unique.

    Examples:
        >>> from pprint import pprint
        >>> from vaxflux.covariates import (
        ...     CovariateCategories,
        ...     _covariate_categories_to_dict,
        ... )
        >>> sex_cov = CovariateCategories(
        ...     covariate="sex", categories=("female", "male")
        ... )
        >>> age_cov = CovariateCategories(
        ...     covariate="age", categories=("youth", "adult", "senior")
        ... )
        >>> pop_density_cov = CovariateCategories(
        ...     covariate="population_density",
        ...     categories=("urban", "suburban", "rural")
        ... )
        >>> pprint(_covariate_categories_to_dict([sex_cov, age_cov, pop_density_cov]))
        {'age': ['youth', 'adult', 'senior'],
         'population_density': ['urban', 'suburban', 'rural'],
         'sex': ['female', 'male']}

    """
    covariate_categories_dict = {
        covariate_category.covariate: list(covariate_category.categories)
        for covariate_category in covariate_categories
    }
    if len(covariate_categories_dict) != len(covariate_categories):
        raise ValueError(
            "Covariate category `covariate` names must be unique to collapse to a dict."
        )
    return covariate_categories_dict


def _covariate_categories_product(
    covariate_categories: list[CovariateCategories],
) -> list[dict[str, str]]:
    """
    Create a product of the covariate categories.

    Args:
        covariate_categories: The covariate categories to use.

    Returns:
        A list of dictionaries with the product of the covariate categories.

    Examples:
        >>> from pprint import pprint
        >>> from vaxflux.covariates import (
        ...     CovariateCategories,
        ...     _covariate_categories_product
        ... )
        >>> sex_cov = CovariateCategories(
        ...     covariate="sex", categories=("male", "female")
        ... )
        >>> age_cov = CovariateCategories(
        ...     covariate="age", categories=("youth", "adult", "senior")
        ... )
        >>> pop_density_cov = CovariateCategories(
        ...     covariate="population_density",
        ...     categories=("urban", "suburban", "rural"),
        ... )
        >>> pprint(_covariate_categories_product([sex_cov, age_cov, pop_density_cov]))
        [{'age': 'youth', 'population_density': 'urban', 'sex': 'male'},
         {'age': 'youth', 'population_density': 'suburban', 'sex': 'male'},
         {'age': 'youth', 'population_density': 'rural', 'sex': 'male'},
         {'age': 'adult', 'population_density': 'urban', 'sex': 'male'},
         {'age': 'adult', 'population_density': 'suburban', 'sex': 'male'},
         {'age': 'adult', 'population_density': 'rural', 'sex': 'male'},
         {'age': 'senior', 'population_density': 'urban', 'sex': 'male'},
         {'age': 'senior', 'population_density': 'suburban', 'sex': 'male'},
         {'age': 'senior', 'population_density': 'rural', 'sex': 'male'},
         {'age': 'youth', 'population_density': 'urban', 'sex': 'female'},
         {'age': 'youth', 'population_density': 'suburban', 'sex': 'female'},
         {'age': 'youth', 'population_density': 'rural', 'sex': 'female'},
         {'age': 'adult', 'population_density': 'urban', 'sex': 'female'},
         {'age': 'adult', 'population_density': 'suburban', 'sex': 'female'},
         {'age': 'adult', 'population_density': 'rural', 'sex': 'female'},
         {'age': 'senior', 'population_density': 'urban', 'sex': 'female'},
         {'age': 'senior', 'population_density': 'suburban', 'sex': 'female'},
         {'age': 'senior', 'population_density': 'rural', 'sex': 'female'}]
        >>> pprint(_covariate_categories_product([sex_cov, age_cov]))
        [{'age': 'youth', 'sex': 'male'},
         {'age': 'adult', 'sex': 'male'},
         {'age': 'senior', 'sex': 'male'},
         {'age': 'youth', 'sex': 'female'},
         {'age': 'adult', 'sex': 'female'},
         {'age': 'senior', 'sex': 'female'}]
        >>> pprint(_covariate_categories_product([sex_cov]))
        [{'sex': 'male'}, {'sex': 'female'}]
        >>> pprint(_covariate_categories_product([age_cov]))
        [{'age': 'youth'}, {'age': 'adult'}, {'age': 'senior'}]
        >>> pprint(_covariate_categories_product([]))
        []

    """
    if not covariate_categories:
        return []
    names = []
    categories = []
    for covariate_category in covariate_categories:
        names.append(covariate_category.covariate)
        categories.append(covariate_category.categories)
    return [dict(zip(names, category)) for category in list(product(*categories))]


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
    ) -> tuple[pm.Distribution, tuple[str, ...]]:
        """
        Return a PyMC3 distribution for the covariate.

        Args:
            name: The name of the distribution already formatted.
            coords: The coordinates for the uptake model.

        Returns:
            A PyMC3 distribution describing the effect of the covariate along with the
            dims of the distribution.
        """
        raise NotImplementedError


class PooledCovariate(Covariate):
    """
    A pooled covariate for uptake models.

    Attributes:
        distribution: The PyMC3 distribution to use for the covariate.
        distribution_kwargs: The keyword arguments to pass to the PyMC3 distribution.

    Returns:
        A PyMC3 distribution describing the effect of the covariate along with the
        dims of the distribution.
    """

    distribution: str
    distribution_kwargs: dict[str, Any]

    def pymc_distribution(
        self, name: str, coords: dict[str, list[str]]
    ) -> tuple[pm.Distribution, tuple[str, ...]]:
        """
        Return a PyMC3 distribution for the pooled covariate.

        Args:
            name: The name of the distribution already formatted.
            coords: The coordinates for the uptake model.
        """
        return getattr(pm, self.distribution)(
            name=name,
            **self.distribution_kwargs,
            dims="season",
        ), ("season",)


class GaussianRandomWalkCovariate(Covariate):
    """
    A gaussian random walk covariate for uptake models.

    Attributes:
        init_mu: The initial mean for the gaussian random walk.
        mu: The drift for the gaussian random walk.
        sigma: The standard deviation for the gaussian random walk.
        eta: The shape parameter for the LKJ distribution for the covariance matrix of
            a multivariate gaussian random walk.

    Returns:
        A PyMC3 distribution describing the effect of the covariate along with the
        dims of the distribution.
    """

    init_mu: ListOfFloats
    mu: ListOfFloats
    sigma: ListOfFloats
    eta: Annotated[float, Field(gt=0.0)] = 1.0

    def pymc_distribution(
        self, name: str, coords: dict[str, list[str]]
    ) -> tuple[pm.Distribution, tuple[str, ...]]:
        """
        Return a PyMC3 distribution for the pooled covariate.

        Args:
            name: The name of the distribution already formatted.
            coords: The coordinates for the uptake model.

        Returns:
            A PyMC3 distribution describing the effect of the covariate along with the
            dims of the distribution.
        """
        if self.covariate is None:
            raise ValueError(
                "A covariate name is required for a gaussian random walk covariate."
            )
        categories_limited_coord_name = _coord_name(
            "covariate", self.covariate, "categories", "limited"
        )
        categories_limited = coords.get(categories_limited_coord_name)
        if not categories_limited:
            raise ValueError(
                f"Missing limited categories for '{self.covariate}' in the coordinates."
            )
        len_categories_limited = len(categories_limited)
        for param in ("init_mu", "mu", "sigma"):
            attr = getattr(self, param)
            if (len_attr := len(attr)) != len_categories_limited:
                raise ValueError(
                    f"The number of values for the '{param}' parameter, "
                    f"{len_attr}, must match the number of limited "
                    f"categories, {len_categories_limited}."
                )
        if len_categories_limited == 1:
            return pm.GaussianRandomWalk(
                name=name,
                mu=self.mu[0],
                sigma=self.sigma[0],
                init_dist=pm.Normal.dist(mu=self.init_mu[0], sigma=self.sigma[0]),
                dims="season",
            ), ("season",)
        packed_chol = pm.LKJCholeskyCov(
            name=f"{name}PackedChol",
            eta=self.eta,
            n=len_categories_limited,
            sd_dist=pm.Exponential.dist(lam=[1.0 / sigma for sigma in self.sigma]),
            compute_corr=False,
            store_in_trace=False,
        )
        chol = pm.expand_packed_triangular(len_categories_limited, packed_chol)
        init_dist = pm.MvNormal.dist(
            mu=self.init_mu, chol=chol, shape=(len_categories_limited,)
        )
        dims = ("season", categories_limited_coord_name)
        return pm.MvGaussianRandomWalk(
            name=name,
            mu=self.mu,
            chol=chol,
            init_dist=init_dist,
            dims=dims,
        ), dims


class GaussianCovariate(Covariate):
    """
    A gaussian random variable covariate for uptake models.

    Attributes:
        mu: Prior mean for the gaussian random variables.
        sigma: The standard deviation for the gaussian random variables.
        eta: The shape parameter for the LKJ distribution for the covariance matrix of
            a multivariate gaussian random walk.

    Returns:
        A PyMC3 distribution describing the effect of the covariate along with the
        dims of the distribution.
    """

    mu: ListOfFloats
    sigma: ListOfFloats
    eta: Annotated[float, Field(gt=0.0)] = 1.0

    def pymc_distribution(self, name, coords):
        """
        Return a PyMC3 distribution for the pooled covariate.

        Args:
            name: The name of the distribution already formatted.
            coords: The coordinates for the uptake model.

        Returns:
            A PyMC3 distribution describing the effect of the covariate along with the
            dims of the distribution.
        """
        if self.covariate is None:
            raise ValueError(
                "A covariate name is required for a gaussian random walk covariate."
            )
        categories_limited_coord_name = _coord_name(
            "covariate", self.covariate, "categories", "limited"
        )
        categories_limited = coords.get(categories_limited_coord_name)
        if not categories_limited:
            raise ValueError(
                f"Missing limited categories for '{self.covariate}' in the coordinates."
            )
        len_categories_limited = len(categories_limited)
        for param in ("mu", "sigma"):
            attr = getattr(self, param)
            if (len_attr := len(attr)) != len_categories_limited:
                raise ValueError(
                    f"The number of values for the '{param}' parameter, "
                    f"{len_attr}, must match the number of limited "
                    f"categories, {len_categories_limited}."
                )
        if len_categories_limited == 1:
            return pm.Normal(
                name=name,
                mu=self.mu[0],
                sigma=self.sigma[0],
                dims="season",
            ), ("season",)
        packed_chol = pm.LKJCholeskyCov(
            name=f"{name}PackedChol",
            eta=self.eta,
            n=len_categories_limited,
            sd_dist=pm.Exponential.dist(lam=[1.0 / sigma for sigma in self.sigma]),
            compute_corr=False,
            store_in_trace=False,
        )
        chol = pm.expand_packed_triangular(len_categories_limited, packed_chol)
        dims = ("season", categories_limited_coord_name)
        return pm.MvNormal(
            name=name,
            mu=self.mu,
            chol=chol,
            dims=dims,
        ), dims


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
    if observations is not None:
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
