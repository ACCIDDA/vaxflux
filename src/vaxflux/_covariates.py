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

    Subclasses must implement `sample()` and adhere to the following shape contract:

    - Seasonal covariates (`covariate is None`): `sample()` must register and return a
      `numpyro.deterministic` site named `f"covariate_values_{self.prefix}"` with shape
      `(num_seasons,)`.
    - Categorical covariates (`covariate is not None`): `sample()` must register and
      return a `numpyro.deterministic` site named `f"covariate_values_{self.prefix}"`
      with shape `(num_seasons, num_categories)`.

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

        Args:
            season_names: Season names defined in the model.
            covariate_categories: All categories for this covariate including the
                baseline at index 0, or `None` for seasonal covariates.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        raise NotImplementedError

    def extra_dims(
        self,
        _season_coord: str,
        _category_short_coord: str | None,
    ) -> dict[str, list[str]]:
        """
        Return ArviZ dimension mappings for any additional numpyro sites.

        Returns dimension mappings for any additional numpyro sites this
        covariate registers beyond the standard `covariate_values_*`
        deterministic.

        The `covariate_values_*` site and the raw `{prefix}` sample site are
        handled automatically by `VaxfluxModel`; only override this method when
        `sample()` registers *extra* named sites (e.g. hyperparameter draws) that
        should carry labelled coordinates in the ArviZ output.

        Args:
            _season_coord: The name of the season coordinate (`"season"`).
            _category_short_coord: The name of the non-baseline-category
                coordinate for this covariate, or `None` for seasonal
                covariates.

        Returns:
            A mapping from numpyro site name to a list of coordinate dimension
            names, in the same format as the `dims` argument accepted by
            `arviz.from_numpyro`.  Return an empty dict (the default) if there
            are no additional sites to label.
        """
        return {}

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

    For seasonal covariates (`covariate is None`) this samples one value per season,
    producing shape `(num_seasons,)`.

    For categorical covariates (`covariate is not None`) this samples one value per
    non-baseline category, then broadcasts that constant effect across all seasons to
    produce shape `(num_seasons, num_categories)`. The baseline category (index 0)
    is always zero.

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
            with numpyro.plate(f"covariate_{self.prefix}", len(season_names)):
                sampled_values = cast(
                    "NumericalArrayLike",
                    numpyro.sample(self.prefix, dist.Normal(self.mu, self.sigma)),
                )
            return cast(
                "NumericalArrayLike",
                numpyro.deterministic(
                    f"covariate_values_{self.prefix}", sampled_values
                ),
            )
        with numpyro.plate(f"covariate_{self.prefix}", len(covariate_categories) - 1):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(self.prefix, dist.Normal(self.mu, self.sigma)),
            )
        padded = jnp.pad(sampled_values, (1, 0), mode="constant")
        inflated = jnp.broadcast_to(
            padded[jnp.newaxis, :],
            (len(season_names), len(covariate_categories)),
        )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", inflated),
        )


class PartiallyPooledGaussianCovariate(Covariate):
    r"""
    Covariate model using a partially pooled Gaussian approach.

    $$
    \begin{aligned}
    \mu_k &\sim \mathrm{Normal}(\mu_{\mu}, \mu_{\sigma}) \\\\
    \sigma_k &\sim \mathrm{HalfNormal}(\sigma) \\\\
    x_{s,k} &\sim \mathrm{Normal}(\mu_k, \sigma_k)
    \end{aligned}
    $$

    For seasonal covariates (`covariate is None`) this samples one value per season
    using shared scalar hyperpriors, producing shape `(num_seasons,)`.

    For categorical covariates (`covariate is not None`) this samples one value per
    non-baseline category using shared scalar hyperpriors, then broadcasts that
    constant effect across all seasons to produce shape
    `(num_seasons, num_categories)`. The baseline category (index 0) is always zero.

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
            with numpyro.plate(f"covariate_{self.prefix}", len(season_names)):
                sampled_values = cast(
                    "NumericalArrayLike",
                    numpyro.sample(self.prefix, dist.Normal(mu_sample, sigma_sample)),
                )
            return cast(
                "NumericalArrayLike",
                numpyro.deterministic(
                    f"covariate_values_{self.prefix}", sampled_values
                ),
            )
        with numpyro.plate(f"covariate_{self.prefix}", len(covariate_categories) - 1):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(self.prefix, dist.Normal(mu_sample, sigma_sample)),
            )
        padded = jnp.pad(sampled_values, (1, 0), mode="constant")
        inflated = jnp.broadcast_to(
            padded[jnp.newaxis, :],
            (len(season_names), len(covariate_categories)),
        )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", inflated),
        )


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
    them toward category-level means. It is only valid for categorical covariates
    (`covariate is not None`) and always returns shape
    `(num_seasons, num_categories)`.

    Attributes:
        mu_mu: The location parameter of the partially pooled mean.
        mu_sigma: The scale parameter of the partially pooled mean.
        sigma: The scale of the half-normal distribution for the standard deviation.

    Notes:
        Unlike covariates generally, this class does not support seasonal covariates
        since the effects are explicitly season-varying. Attempting to use this class
        with `covariate is None` will raise an error at model definition time due to
        the covariate name validation.
    """

    covariate: str
    mu_mu: float
    mu_sigma: float = Field(gt=0.0)
    sigma: float = Field(gt=0.0)

    def extra_dims(
        self,
        season_coord: str,
        category_short_coord: str | None,
    ) -> dict[str, list[str]]:
        """
        Return dims for the hyperparameter and raw-draw sites.

        Registers `{prefix}_mu`, `{prefix}_sigma`, and `{prefix}` with
        the appropriate coordinate names.
        """
        short = category_short_coord or season_coord
        return {
            f"{self.prefix}_mu": [short],
            f"{self.prefix}_sigma": [short],
            self.prefix: [season_coord, short],
        }

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
                Always non-`None` at runtime because `covariate` is required
                to be a non-empty string on this class, so the model always
                resolves and passes categories.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        # covariate_categories is always non-None here: the model only resolves
        # and passes categories when covariate is not None, and covariate is
        # required to be a str on this class.
        categories = cast("list[str]", covariate_categories)
        num_seasons = len(season_names)
        num_categories = len(categories)
        # Draw category-level parameters from hyper priors.
        with numpyro.plate(f"covariate_{self.prefix}", num_categories - 1):
            mu_sample = numpyro.sample(
                f"{self.prefix}_mu",
                dist.Normal(self.mu_mu, self.mu_sigma),
            )
            sigma_sample = numpyro.sample(
                f"{self.prefix}_sigma",
                dist.HalfNormal(self.sigma),
            )
        # Draw season-level effects per category from the category-level priors.
        with numpyro.plate(f"{self.prefix}_season", num_seasons):
            sampled_values = numpyro.sample(
                self.prefix,
                dist.Normal(mu_sample, sigma_sample).to_event(1),
            )
        padded = jnp.pad(sampled_values, ((0, 0), (1, 0)), mode="constant")
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(f"covariate_values_{self.prefix}", padded),
        )
