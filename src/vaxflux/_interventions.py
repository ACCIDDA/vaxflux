"""Intervention definitions for numpyro/JAX models."""

import warnings
from datetime import date
from typing import Annotated, Any, cast

import numpyro
import numpyro.distributions as dist
from jax import Array as JaxArray
from pydantic import BaseModel, ConfigDict, Field, computed_field

from vaxflux._util import _coord_name
from vaxflux.covariates import CovariateCategories, _covariate_categories_to_dict
from vaxflux.dates import SeasonRange

InterventionName = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]


class Intervention(BaseModel):
    """A representation for the affect of an intervention on the uptake of a vaccine."""

    model_config = ConfigDict(frozen=True)

    #: The name of the intervention, must be a lowercase alphanumeric string.
    name: InterventionName

    #: The name of the parameter this intervention affects.
    parameter: str

    #: The name of the distribution to use for the intervention.
    distribution: str

    #: The keyword arguments to pass to the distribution. Must be a dictionary of
    #: strings to values.
    distribution_kwargs: dict[str, Any]

    def numpyro_distribution(self) -> dist.Distribution:
        """Return a numpyro distribution for the intervention."""
        return getattr(dist, self.distribution)(**self.distribution_kwargs)

    def sample(self, name: str) -> JaxArray:
        """Sample an intervention effect from its distribution."""
        return cast("JaxArray", numpyro.sample(name, self.numpyro_distribution()))


class Implementation(BaseModel):
    """A representation of an implementation of an intervention."""

    model_config = ConfigDict(frozen=True)

    #: The name of the intervention being implemented.
    intervention: InterventionName

    #: The season this implementation applies to.
    season: str

    #: The start date of the implementation or `None` if to apply from the start of the
    #: season.
    start_date: date | None

    #: The end date of the implementation or `None` if to apply to the end of the
    #: season.
    end_date: date | None

    #: The covariate categories this implementation applies to in the format
    #: `{covariate: category}`. If `None`, applies to all covariate categories.
    covariate_categories: dict[str, str] | None

    @computed_field  # type: ignore[prop-decorator]
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
                    for item in [k, self.covariate_categories[k]]  # type: ignore[index]
                ],
            ],
        )


def _check_interventions_and_implementations(
    interventions: list[Intervention],
    implementations: list[Implementation],
    season_ranges: list[SeasonRange],
    covariate_categories: list[CovariateCategories],
) -> None:
    """Check that the interventions and implementations are valid."""
    intervention_names = {i.name for i in interventions}
    implementation_intervention_names = {i.intervention for i in implementations}
    if (
        len(
            missing_intervention_names := implementation_intervention_names
            - intervention_names,
        )
        > 0
    ):
        msg = (
            "The following implementation intervention names do not match "
            f"any interventions: {', '.join(missing_intervention_names)}."
        )
        raise ValueError(
            msg,
        )
    if (
        len(
            missing_implementation_names := intervention_names
            - implementation_intervention_names,
        )
        > 0
    ):
        warnings.warn(
            f"The following interventions do not have a corresponding implementation, "
            f"affect will be unknown: {', '.join(missing_implementation_names)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    season_map = {season.season: season for season in season_ranges}
    implementation_seasons = {i.season for i in implementations}
    if len(missing_season_names := implementation_seasons - season_map.keys()) > 0:
        msg = (
            "The following implementation seasons do not match "
            f"any seasons: {', '.join(missing_season_names)}."
        )
        raise ValueError(
            msg,
        )
    for implementation in implementations:
        if (
            implementation.start_date is not None
            and implementation.start_date < season_map[implementation.season].start_date
        ):
            msg = (
                f"The start date of the implementation {implementation} is before "
                f"the start date of the season {implementation.season}."
            )
            raise ValueError(
                msg,
            )
        if (
            implementation.end_date is not None
            and implementation.end_date > season_map[implementation.season].end_date
        ):
            msg = (
                f"The end date of the implementation {implementation} is after "
                f"the end date of the season {implementation.season}."
            )
            raise ValueError(
                msg,
            )
    covariate_categories_map = _covariate_categories_to_dict(covariate_categories)
    for implementation in implementations:
        if implementation.covariate_categories is not None:
            for covariate, category in implementation.covariate_categories.items():
                if covariate not in covariate_categories_map:
                    msg = (
                        f"The covariate '{covariate}' is not a valid covariate "
                        f"category."
                    )
                    raise ValueError(
                        msg,
                    )
                if category not in covariate_categories_map[covariate]:
                    msg = (
                        f"The category '{category}' is not a valid category for "
                        f"covariate '{covariate}'."
                    )
                    raise ValueError(
                        msg,
                    )
