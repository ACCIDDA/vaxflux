from abc import ABC, abstractmethod
from typing import cast

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pydantic import BaseModel, Field, field_validator

from vaxflux._types import NumericalArrayLike


class Covariate(ABC, BaseModel):
    """
    Abstract base class for covariates in vaxflux.

    Attributes:
        parameter: The name of the model parameter this covariate affects.
        covariate: The name of the covariate variable.
    """

    parameter: str
    covariate: str | None = None

    @property
    def prefix(self) -> str:
        """
        Get the prefix for the covariate based on the parameter name.

        Returns:
            The prefix string derived from the parameter name.
        """
        return self.parameter + (f"_{self.covariate}" if self.covariate else "_season")

    @abstractmethod
    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        """
        Abstract method to sample covariate values using model context.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        raise NotImplementedError

    @abstractmethod
    def dims(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
        category_coord_keys: tuple[str, str] | None,
    ) -> dict[str, list[str]]:
        """
        Return dims for covariate sampling and deterministic outputs.

        Args:
            season_names: Season names defined in the model.
            covariate_categories: Covariate categories for this covariate, or `None`
                if the covariate is seasonal.
            category_coord_keys: Tuple of (full_key, short_key) for covariate category
                coordinates, or `None` for seasonal covariates.

        Returns:
            A mapping of variable name to dimension names. The keys should include
            `self.prefix` and `f"covariate_values_{self.prefix}"` so ArviZ can
            attach coordinates correctly. For example, a seasonal covariate might
            return:

                {
                    "m_season": ["season"],
                    "covariate_values_m_season": ["season"],
                }

            While a non-seasonal covariate with categories might return:

                {
                    "m_age": ["age_categories_short"],
                    "covariate_values_m_age": ["age_categories"],
                }

        """
        raise NotImplementedError

    @field_validator("covariate", mode="after")
    @classmethod
    def _covariate_cannot_be_called_season(cls, v: str) -> str:
        """
        Validate that the covariate name is not 'season'.

        Args:
            v: The covariate name to validate.

        Returns:
            The validated covariate name.

        """
        if v == "season":
            msg = "covariate cannot be called 'season'."
            raise ValueError(msg)
        return v


class PooledGaussianCovariate(Covariate):
    r"""
    Covariate model using a pooled Gaussian approach.

    $$
    \begin{aligned}
    x_k &\sim \mathrm{Normal}(\mu, \sigma)
    \end{aligned}
    $$

    Attributes:
        mu: The mean of the Gaussian distribution.
        sigma: The standard deviation of the Gaussian distribution.

    """

    mu: float
    sigma: float = Field(gt=0.0)

    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        """
        Sample values from a Gaussian distribution defined by the mean and stddev.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        if covariate_categories is None:
            plate_size = len(season_names)
        else:
            plate_size = len(covariate_categories) - 1
        with numpyro.plate(f"covariate_{self.prefix}", plate_size):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(self.prefix, dist.Normal(self.mu, self.sigma)),
            )
        values = (
            sampled_values
            if covariate_categories is None
            else jnp.pad(sampled_values, (1, 0), mode="empty")
        )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", values),
        )

    def dims(
        self,
        *,
        season_names: list[str],  # noqa: ARG002
        covariate_categories: list[str] | None,
        category_coord_keys: tuple[str, str] | None,
    ) -> dict[str, list[str]]:
        if covariate_categories is None:
            return {
                self.prefix: ["season"],
                f"covariate_values_{self.prefix}": ["season"],
            }
        if category_coord_keys is None:
            msg = "category_coord_keys must be provided for non-seasonal covariates."
            raise ValueError(msg)
        full_key, short_key = category_coord_keys
        return {
            self.prefix: [short_key],
            f"covariate_values_{self.prefix}": [full_key],
        }


class PartiallyPooledGaussianCovariate(Covariate):
    r"""
    Covariate model using a pooled Gaussian approach.

    $$
    \begin{aligned}
    \mu_k &\sim \mathrm{Normal}(\mu_{\mu}, \mu_{\sigma}) \\\\
    \sigma_k &\sim \mathrm{HalfNormal}(\sigma) \\\\
    x_{s,k} &\sim \mathrm{Normal}(\mu_k, \sigma_k)
    \end{aligned}
    $$

    Attributes:
        mu_mu: The location parameter of the partially pooled mean.
        mu_sigma: The scale parameter of the partially pooled mean.
        sigma: The scale of the half-normal distribution for the standard deviation.
    """

    mu_mu: float
    mu_sigma: float = Field(gt=0.0)
    sigma: float = Field(gt=0.0)

    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        """
        Sample values from a Gaussian distribution defined by the mean and stddev.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        mu_sample = numpyro.sample(
            f"{self.prefix}_mu", dist.Normal(self.mu_mu, self.mu_sigma)
        )
        sigma_sample = numpyro.sample(
            f"{self.prefix}_sigma", dist.HalfNormal(self.sigma)
        )
        if covariate_categories is None:
            plate_size = len(season_names)
        else:
            plate_size = len(covariate_categories) - 1
        with numpyro.plate(f"covariate_{self.prefix}", plate_size):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(self.prefix, dist.Normal(mu_sample, sigma_sample)),
            )
        values = (
            sampled_values
            if covariate_categories is None
            else jnp.pad(sampled_values, (1, 0), mode="empty")
        )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", values),
        )

    def dims(
        self,
        *,
        season_names: list[str],  # noqa: ARG002
        covariate_categories: list[str] | None,
        category_coord_keys: tuple[str, str] | None,
    ) -> dict[str, list[str]]:
        if covariate_categories is None:
            return {
                self.prefix: ["season"],
                f"covariate_values_{self.prefix}": ["season"],
            }
        if category_coord_keys is None:
            msg = "category_coord_keys must be provided for non-seasonal covariates."
            raise ValueError(msg)
        full_key, short_key = category_coord_keys
        return {
            self.prefix: [short_key],
            f"covariate_values_{self.prefix}": [full_key],
        }


class SeasonVaryingPartiallyPooledGaussianCovariate(Covariate):
    r"""
    Season-varying covariate model using partial pooling across seasons.

    $$
    \begin{aligned}
    \mu_k &\sim \mathrm{Normal}(\mu_{\mu}, \mu_{\sigma}) \\\\
    \sigma_k &\sim \mathrm{HalfNormal}(\sigma) \\\\
    x_{s,k} &\sim \mathrm{Normal}(\mu_k, \sigma_k)
    \end{aligned}
    $$

    This covariate samples category effects that vary by season, while shrinking
    them toward category-level means.

    Attributes:
        mu_mu: The location parameter of the partially pooled mean.
        mu_sigma: The scale parameter of the partially pooled mean.
        sigma: The scale of the half-normal distribution for the standard deviation.
    """

    mu_mu: float
    mu_sigma: float = Field(gt=0.0)
    sigma: float = Field(gt=0.0)

    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        """
        Sample season-varying covariate values.

        Args:
            season_names: Season names in the model.
            covariate_categories: Covariate categories including the baseline.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        if covariate_categories is None:
            msg = "Season-varying covariates require covariate categories."
            raise ValueError(msg)
        num_seasons = len(season_names)
        num_categories = len(covariate_categories)
        if num_categories < 2:
            msg = "Season-varying covariates require at least 2 categories."
            raise ValueError(msg)
        mu_sample = numpyro.sample(
            f"{self.prefix}_mu",
            dist.Normal(self.mu_mu, self.mu_sigma),
        )
        sigma_sample = numpyro.sample(
            f"{self.prefix}_sigma",
            dist.HalfNormal(self.sigma),
        )
        with (
            numpyro.plate(f"covariate_{self.prefix}", num_categories - 1),
            numpyro.plate(f"{self.prefix}_season", num_seasons),
        ):
            sampled_values = numpyro.sample(
                self.prefix,
                dist.Normal(mu_sample, sigma_sample),
            )
        padded = jnp.pad(sampled_values, ((0, 0), (1, 0)), mode="constant")
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", padded),
        )

    def dims(
        self,
        *,
        season_names: list[str],  # noqa: ARG002
        covariate_categories: list[str] | None,
        category_coord_keys: tuple[str, str] | None,
    ) -> dict[str, list[str]]:
        if covariate_categories is None or category_coord_keys is None:
            msg = "Season-varying covariates require covariate categories."
            raise ValueError(msg)
        full_key, short_key = category_coord_keys
        return {
            self.prefix: ["season", short_key],
            f"covariate_values_{self.prefix}": ["season", full_key],
        }
