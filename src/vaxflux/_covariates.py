from abc import ABC, abstractmethod
from typing import cast

import numpyro
import numpyro.distributions as dist
from pydantic import BaseModel, Field

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
        return self.parameter + (f"_{self.covariate}" if self.covariate else "")

    def presample(self) -> None:
        """
        Hook method for any pre-sampling steps required before sampling.

        This method can be overridden by subclasses if needed.
        """

    @abstractmethod
    def sample(self) -> NumericalArrayLike:
        """
        Abstract method to sample the covariate values.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        raise NotImplementedError


class PartiallyPooledGaussianCovariate(Covariate):
    r"""
    Covariate model using a pooled Gaussian approach.

    $$ \\mu_i \\sim \\mathrm{Normal}(\\mu_0, \\mu_1) $$

    $$ \\sigma_i \\sim \\mathrm{Half}\\text{-}\\mathrm{Normal}(\\sigma) $$

    $$ x_i \\sim \\mathrm{Normal}(\\mu_i, \\sigma_i) $$

    Attributes:
        mu: A tuple representing the location and scale of the partially pooled mean.
        sigma: The scale of the half-normal distribution for the standard deviation.
    """

    mu: tuple[float, float]
    sigma: float = Field(gt=0.0)

    def presample(self) -> None:
        """
        Hook method for any pre-sampling steps required before sampling.

        This method can be overridden by subclasses if needed.
        """
        self._mu_sample = numpyro.sample(
            f"{self.prefix}_mu", dist.Normal(self.mu[0], self.mu[1])
        )
        self._sigma_sample = numpyro.sample(
            f"{self.prefix}_sigma", dist.HalfNormal(self.sigma)
        )

    def sample(self) -> NumericalArrayLike:
        """
        Sample values from a Gaussian distribution defined by the mean and stddev.

        Returns:
            A numerical array-like structure containing the sampled covariate values.
        """
        return cast(
            "NumericalArrayLike",
            numpyro.sample(
                self.prefix, dist.Normal(self._mu_sample, self._sigma_sample)
            ),
        )
