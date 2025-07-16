"""General purpose fitting utilities for vaxflux."""

__all__: tuple[str, ...] = ("least_squares_curve_fit",)

from datetime import datetime

import pandas as pd

from vaxflux._util import _validate_and_format_observations
from vaxflux.covariates import CovariateCategories, _covariate_categories_product
from vaxflux.curves import Curve
from vaxflux.dates import DateRange, SeasonRange, _infer_ranges_from_observations


def least_squares_curve_fit(
    curve: Curve,
    observations: pd.DataFrame,
    date_ranges: list[DateRange] | None = None,
    season_ranges: list[SeasonRange] | None = None,
    covariate_categories: list[CovariateCategories] | None = None,
) -> None:
    """
    Perform a least-squares curve fit to the provided observations.

    Args:
        curve: The curve family to use. Must implement the :obj:`vaxflux.curves.Curve`
            interface.
        observations: The uptake dataset to use.
        date_ranges: The date ranges for the uptake scenarios or `None` to derive
            them from the observations.
        season_ranges: The season ranges for the uptake scenarios or `None` to derive
            them from the observations.
        covariate_categories: The covariate categories to use for the fit or `None` to
            use the default categories.

    Raises:
        NotImplementedError: If the observations DataFrame contains values other than
            'prevalence' in the 'type' column.
    """
    # Initial input validation/formatting
    observations = _validate_and_format_observations(observations)
    if set(observations["type"].unique().tolist()) != {"prevalence"}:
        msg = (
            "Only 'prevalence' data is supported, 'incidence' and count equivalents "
            "are planned."
        )
        raise NotImplementedError(msg)
    date_ranges = _infer_ranges_from_observations(
        observations, date_ranges or [], "date"
    )
    season_ranges = _infer_ranges_from_observations(
        observations, season_ranges or [], "season"
    )
    covariate_categories = covariate_categories or []
    categories_prod = _covariate_categories_product(covariate_categories)
    if not categories_prod:
        msg = "No covariate categories provided, at least one is required."
        raise NotImplementedError(msg)
    # Fit each time series individually
    for season_range in season_ranges:
        for category_prod in categories_prod:
            observations_subset = observations.query(
                " & ".join(
                    "@cat == @val"
                    for cat, val in (
                        category_prod | {"season": season_range.season}
                    ).items()
                )
            )
            if observations_subset.empty:
                continue
            t = (
                observations_subset["end_date"]
                - datetime.combine(season_range.start_date, datetime.min.time())
            ).dt.days.to_numpy()
            y = observations_subset["value"].to_numpy()
            curve.propose_p0(t, y)

    # Loosely, need to:

    # 1) Split out `observations` by `category_prod`,
    # 2) Get the prevalence out from observations and the time steps as numpy arrays.
    # 3) Get the initial param proposal
    # 4) Do least squares! Store result
    # 5) .... Take that stored result put it in matrix form and decompose?
