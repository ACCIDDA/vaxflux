"""Unit tests for `_check_interventions_and_implementations`."""

from datetime import date

import pytest

from vaxflux import CovariateCategories, Implementation, Intervention, SeasonRange
from vaxflux._interventions import _check_interventions_and_implementations


@pytest.fixture
def example_inputs() -> tuple[
    list[Intervention],
    list[Implementation],
    list[SeasonRange],
    list[CovariateCategories],
]:
    """
    Example inputs for testing `_check_interventions_and_implementations`.

    Returns:
        A tuple containing lists of interventions, implementations, season ranges,
        and covariate categories for a hypothetical vaccination TV advertisement
        campaign.

    """
    interventions = [
        Intervention(
            name="tv_ads",
            parameter="m",
            distribution="Normal",
            distribution_kwargs={"loc": 0.0, "scale": 1.0},
        )
    ]
    implementations = [
        Implementation(
            intervention="tv_ads",
            season="2022/23",
            start_date=None,
            end_date=None,
            covariate_categories={"sex": "male"},
        )
    ]
    season_ranges = [
        SeasonRange(
            season="2022/23",
            start_date=date(2022, 1, 1),
            end_date=date(2022, 12, 31),
        )
    ]
    covariate_categories = [
        CovariateCategories(covariate="sex", categories=("female", "male"))
    ]
    return interventions, implementations, season_ranges, covariate_categories


@pytest.mark.parametrize("missing_name", ["radio_ads"])
def test_missing_intervention_names_raises_value_error(
    example_inputs: tuple[
        list[Intervention],
        list[Implementation],
        list[SeasonRange],
        list[CovariateCategories],
    ],
    missing_name: str,
) -> None:
    """Implementations without matching interventions raise a `ValueError`."""
    interventions, implementations, season_ranges, covariate_categories = example_inputs
    implementations = [
        Implementation(
            intervention=missing_name,
            season=implementations[0].season,
            start_date=implementations[0].start_date,
            end_date=implementations[0].end_date,
            covariate_categories=implementations[0].covariate_categories,
        )
    ]
    with pytest.raises(
        ValueError,
        match=(
            r"^The following implementation intervention names do not match "
            r"any interventions: .*.$"
        ),
    ):
        _check_interventions_and_implementations(
            interventions,
            implementations,
            season_ranges,
            covariate_categories,
        )


@pytest.mark.parametrize("missing_name", ["radio_ads"])
def test_missing_implementation_names_warns(
    example_inputs: tuple[
        list[Intervention],
        list[Implementation],
        list[SeasonRange],
        list[CovariateCategories],
    ],
    missing_name: str,
) -> None:
    """Interventions without implementations emit a warning."""
    interventions, implementations, season_ranges, covariate_categories = example_inputs
    interventions = [
        *interventions,
        Intervention(
            name=missing_name,
            parameter="m",
            distribution="Normal",
            distribution_kwargs={"loc": 0.0, "scale": 1.0},
        ),
    ]
    with pytest.warns(
        RuntimeWarning,
        match=(
            r"^The following interventions do not have a corresponding "
            r"implementation, affect will be unknown: .*.$"
        ),
    ):
        _check_interventions_and_implementations(
            interventions,
            implementations,
            season_ranges,
            covariate_categories,
        )


@pytest.mark.parametrize("missing_season", ["2023/24"])
def test_missing_season_names_raises_value_error(
    example_inputs: tuple[
        list[Intervention],
        list[Implementation],
        list[SeasonRange],
        list[CovariateCategories],
    ],
    missing_season: str,
) -> None:
    """Implementations with unknown seasons raise a `ValueError`."""
    interventions, implementations, season_ranges, covariate_categories = example_inputs
    implementations = [
        Implementation(
            intervention=implementations[0].intervention,
            season=missing_season,
            start_date=implementations[0].start_date,
            end_date=implementations[0].end_date,
            covariate_categories=implementations[0].covariate_categories,
        )
    ]
    with pytest.raises(
        ValueError,
        match=(
            r"^The following implementation seasons do not match "
            r"any seasons: .*.$"
        ),
    ):
        _check_interventions_and_implementations(
            interventions,
            implementations,
            season_ranges,
            covariate_categories,
        )


@pytest.mark.parametrize(
    ("start_date", "end_date", "match"),
    [
        (
            date(2021, 12, 31),
            None,
            r"^The start date of the implementation .* is before the start date",
        ),
        (
            None,
            date(2023, 1, 1),
            r"^The end date of the implementation .* is after the end date",
        ),
    ],
)
def test_implementation_date_bounds_raises_value_error(
    example_inputs: tuple[
        list[Intervention],
        list[Implementation],
        list[SeasonRange],
        list[CovariateCategories],
    ],
    start_date: date | None,
    end_date: date | None,
    match: str,
) -> None:
    """Implementation dates outside season bounds raise a `ValueError`."""
    interventions, implementations, season_ranges, covariate_categories = example_inputs
    implementations = [
        Implementation(
            intervention=implementations[0].intervention,
            season=implementations[0].season,
            start_date=start_date,
            end_date=end_date,
            covariate_categories=implementations[0].covariate_categories,
        )
    ]
    with pytest.raises(ValueError, match=match):
        _check_interventions_and_implementations(
            interventions,
            implementations,
            season_ranges,
            covariate_categories,
        )


@pytest.mark.parametrize(
    ("covariate_categories", "match"),
    [
        (
            {"age": "adult"},
            r"^The covariate 'age' is not a valid covariate category\.$",
        ),
        (
            {"sex": "other"},
            r"^The category 'other' is not a valid category for covariate 'sex'\.$",
        ),
    ],
)
def test_invalid_covariate_categories_raise_value_error(
    example_inputs: tuple[
        list[Intervention],
        list[Implementation],
        list[SeasonRange],
        list[CovariateCategories],
    ],
    covariate_categories: dict[str, str],
    match: str,
) -> None:
    """Invalid covariate categories raise a `ValueError`."""
    interventions, implementations, season_ranges, covariate_categories_list = (
        example_inputs
    )
    implementations = [
        Implementation(
            intervention=implementations[0].intervention,
            season=implementations[0].season,
            start_date=implementations[0].start_date,
            end_date=implementations[0].end_date,
            covariate_categories=covariate_categories,
        )
    ]
    with pytest.raises(ValueError, match=match):
        _check_interventions_and_implementations(
            interventions,
            implementations,
            season_ranges,
            covariate_categories_list,
        )
