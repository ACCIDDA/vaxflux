"""Functionality for applying interventions to the uptake of a vaccine."""

__all__ = ("Implementation", "Intervention")

from datetime import date
from typing import Annotated, Any

import pymc as pm
from pydantic import BaseModel, ConfigDict, Field

InterventionName = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]


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
        ...     distribution_kwargs={"sigma": 0.1},
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

    model_config = ConfigDict(frozen=True)

    name: InterventionName
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


class Implementation(BaseModel):
    """
    A representation of an implementation of an intervention.

    Attributes:
        intervention: The name of the intervention being implemented.
        season: The season this implementation applies to.
        start_date: The start date of the implementation or `None` if to apply from the
            start of the season.
        end_date: The end date of the implementation or `None` if to apply to the end of
            the season.
        covariate_categories: The covariate categories this implementation applies to in
            the format {covariate: category}. If `None`, applies to all covariate
            categories.

    Examples:
        >>> from vaxflux.interventions import Implementation
        >>> tv_ads_targeted_at_males_2022_23 = Implementation(
        ...     intervention="tv_ads",
        ...     season="2022/23",
        ...     start_date=None,
        ...     end_date=None,
        ...     covariate_categories={"sex": "male"},
        ... )
        >>> tv_ads_targeted_at_males_2022_23.intervention
        'tv_ads'
        >>> tv_ads_targeted_at_males_2022_23.season
        '2022/23'
        >>> tv_ads_targeted_at_males_2022_23.start_date is None
        True
        >>> tv_ads_targeted_at_males_2022_23.end_date is None
        True
        >>> tv_ads_targeted_at_males_2022_23.covariate_categories
        {'sex': 'male'}
    """

    model_config = ConfigDict(frozen=True)

    intervention: InterventionName
    season: str
    start_date: date | None
    end_date: date | None
    covariate_categories: dict[str, str] | None
