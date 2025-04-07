"""Functionality for applying interventions to the uptake of a vaccine."""

__all__ = ("Implementation", "Intervention")

import warnings
from datetime import date
from typing import Annotated, Any

import pymc as pm
from pydantic import BaseModel, ConfigDict, Field, computed_field

from vaxflux._util import _coord_name
from vaxflux.covariates import CovariateCategories, _covariate_categories_to_dict
from vaxflux.dates import SeasonRange

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
        >>> tv_ads_targeted_at_males_2022_23.name
        'tv_ads_2022_23_sex_male'

    """

    model_config = ConfigDict(frozen=True)

    intervention: InterventionName
    season: str
    start_date: date | None
    end_date: date | None
    covariate_categories: dict[str, str] | None

    @computed_field  # type: ignore[misc]
    @property
    def name(self) -> str:
        """
        Get the name of the implementation.

        Returns:
            The name of the implementation, formatted as a string.
        """
        return _coord_name(
            *[
                self.intervention,
                self.season,
                *[
                    str(item)
                    for k in sorted((self.covariate_categories or {}).keys())
                    for item in [k, self.covariate_categories[k]]  # type: ignore
                ],
            ]
        )


def _check_interventions_and_implementations(
    interventions: list[Intervention],
    implementations: list[Implementation],
    season_ranges: list[SeasonRange],
    covariate_categories: list[CovariateCategories],
):
    """
    Check that the interventions and implementations are valid.

    Args:
        interventions: The list of interventions.
        implementations: The list of implementations.
        season_ranges: The list of season ranges.
        covariate_categories: The list of covariate categories.

    Raises:
        ValueError: If an implementation does not match an intervention.
        ValueError: If an implementation does not match a season.
        ValueError: If an implementation start date is before the season start date.
        ValueError: If an implementation end date is after the season end date.
        ValueError: If an implementation covariate is not a valid covariate.
        ValueError: If an implementation covariate category is not a valid category.

    Warnings:
        Warning: If an intervention does not have a corresponding implementation.
    """
    intervention_names = {i.name for i in interventions}
    implementation_intervention_names = {i.intervention for i in implementations}
    if (
        len(
            missing_intervention_names := implementation_intervention_names
            - intervention_names
        )
        > 0
    ):
        raise ValueError(
            "The following implementation intervention names do not match "
            f"any interventions: {', '.join(missing_intervention_names)}."
        )
    if (
        len(
            missing_implementation_names := intervention_names
            - implementation_intervention_names
        )
        > 0
    ):
        warnings.warn(
            f"The following interventions do not have a corresponding implementation, "
            f"affect will be unknown: {', '.join(missing_implementation_names)}.",
            RuntimeWarning,
        )
    season_map = {season.season: season for season in season_ranges}
    implementation_seasons = {i.season for i in implementations}
    if len(missing_season_names := implementation_seasons - season_map.keys()) > 0:
        raise ValueError(
            "The following implementation seasons do not match "
            f"any seasons: {', '.join(missing_season_names)}."
        )
    for implementation in implementations:
        if (
            implementation.start_date is not None
            and implementation.start_date < season_map[implementation.season].start_date
        ):
            raise ValueError(
                f"The start date of the implementation {implementation} is before "
                f"the start date of the season {implementation.season}."
            )
        if (
            implementation.end_date is not None
            and implementation.end_date > season_map[implementation.season].end_date
        ):
            raise ValueError(
                f"The end date of the implementation {implementation} is after "
                f"the end date of the season {implementation.season}."
            )
    covariate_categories_map = _covariate_categories_to_dict(covariate_categories)
    for implementation in implementations:
        if implementation.covariate_categories is not None:
            for covariate, category in implementation.covariate_categories.items():
                if covariate not in covariate_categories_map:
                    raise ValueError(
                        f"The covariate '{covariate}' is not a valid covariate "
                        f"category."
                    )
                if category not in covariate_categories_map[covariate]:
                    raise ValueError(
                        f"The category '{category}' is not a valid category for "
                        f"covariate '{covariate}'."
                    )
