"""Unit tests for `vaxflux.dates.daily_date_ranges` function."""

from datetime import date
from typing import Literal

import pytest

from vaxflux.dates import SeasonRange, DateRange, daily_date_ranges


@pytest.mark.parametrize("range_days", (-10, -1))
def test_daily_date_ranges_invalid_range_days_value_error(range_days: int) -> None:
    """Test `ValueError` is raised when `range_days` is less than 1."""
    with pytest.raises(
        ValueError,
        match="^The number of days for each daily date range must be at least 0.$",
    ):
        daily_date_ranges(
            SeasonRange(
                season="2020/21",
                start_date=date(2020, 10, 1),
                end_date=date(2021, 3, 31),
            ),
            range_days=range_days,
        )


@pytest.mark.parametrize("range_days", (2, 10, 14))
def test_daily_date_ranges_invalid_range_days_remainder_raise(range_days: int) -> None:
    """Test `ValueError` is raised when `remainder` is 'raise'."""
    season_range = SeasonRange(
        season="2020/21",
        start_date=date(2020, 10, 1),
        end_date=date(2021, 3, 31),
    )
    ndays = (season_range.end_date - season_range.start_date).days
    assert ndays % range_days != 0
    with pytest.raises(
        ValueError,
        match=(
            "^The number of days for each daily date range does not divide "
            f"evenly into the season range for {season_range.season}.$"
        ),
    ):
        daily_date_ranges(season_range, range_days=range_days, remainder="raise")


@pytest.mark.parametrize(
    "season_ranges",
    (
        [
            SeasonRange(
                season="Dec 2019",
                start_date=date(2019, 12, 1),
                end_date=date(2019, 12, 31),
            ),
        ],
        [
            SeasonRange(
                season="2020/21",
                start_date=date(2020, 10, 1),
                end_date=date(2021, 3, 31),
            ),
        ],
        [
            SeasonRange(
                season="2021/2022",
                start_date=date(2021, 9, 1),
                end_date=date(2022, 4, 30),
            ),
            SeasonRange(
                season="2022/2023",
                start_date=date(2022, 9, 1),
                end_date=date(2023, 4, 30),
            ),
        ],
        [
            SeasonRange(
                season="2022 Summer",
                start_date=date(2022, 6, 1),
                end_date=date(2022, 8, 31),
            ),
            SeasonRange(
                season="2022 Winter",
                start_date=date(2022, 12, 1),
                end_date=date(2023, 2, 28),
            ),
            SeasonRange(
                season="2023 Summer",
                start_date=date(2023, 6, 1),
                end_date=date(2023, 8, 31),
            ),
            SeasonRange(
                season="2023 Winter",
                start_date=date(2023, 12, 1),
                end_date=date(2024, 2, 29),
            ),
        ],
    ),
)
@pytest.mark.parametrize("range_days", (0, 1, 2, 5, 7))
@pytest.mark.parametrize("remainder", ("fill", "skip"))
def test_output_validation(
    season_ranges: list[SeasonRange],
    range_days: int,
    remainder: Literal["fill", "skip"],
) -> None:
    """Test output validation."""
    date_ranges = daily_date_ranges(
        season_ranges, range_days=range_days, remainder=remainder
    )
    assert isinstance(date_ranges, list)
    assert all(isinstance(date_range, DateRange) for date_range in date_ranges)
    expected_date_ranges = 0
    for season_range in season_ranges:
        season_ndays = (season_range.end_date - season_range.start_date).days + 1
        expected_date_ranges += season_ndays // (range_days + 1)
        if season_ndays % (range_days + 1) != 0 and remainder == "fill":
            expected_date_ranges += 1
    assert len(date_ranges) == expected_date_ranges
