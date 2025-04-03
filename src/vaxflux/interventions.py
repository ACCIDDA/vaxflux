"""Functionality for applying interventions to the uptake of a vaccine."""

__all__ = ("Intervention",)


from typing import Any

import pymc as pm
from pydantic import BaseModel, Field


class Intervention(BaseModel):
    """
    A representation for the affect of an intervention on the uptake of a vaccine.

    Attributes:
        name: The name of the intervention, must be a lowercase alphanumeric string.
        parameter: The name of the parameter this intervention affects.
        distribution: The name of the distribution to use for the intervention.
        distribution_kwargs: The keyword arguments to pass to the distribution.
            Must be a dictionary of strings to values.

    Examples:
        >>> from vaxflux.interventions import Intervention
        >>> tv_ads = Intervention(
        ...     name="tv_ads",
        ...     parameter="m",
        ...     distribution="HalfNormal",
        ...     distribution_kwargs={"sigma": 0.1}
        ... )
        >>> tv_ads.name
        'tv_ads'
        >>> tv_ads.parameter
        'm'
        >>> tv_ads.distribution
        'HalfNormal'
        >>> tv_ads.distribution_kwargs
        {'sigma': 0.1}

    """

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    parameter: str
    distribution: str
    distribution_kwargs: dict[str, Any]

    def pymc_distribution(self, name: str) -> tuple[pm.Distribution, tuple[str, ...]]:
        """
        Return a PyMC3 distribution for the intervention.

        Args:
            name: The name of the distribution already formatted.
            coords: The coordinates for the uptake model.
        """
        return getattr(pm, self.distribution)(
            name=name,
            **self.distribution_kwargs,
            dims=f"intervention_{name}_implementations",
        )
