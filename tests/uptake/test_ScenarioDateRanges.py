"""Unit tests for the `ScenarioDateRanges` class."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest

from vaxflux.uptake import DateRange, ScenarioDateRanges


@pytest.mark.parametrize(
    ("date_ranges", "unique_ranges"),
    (
        (
            (
                DateRange(
                    "2020/21", date(2020, 12, 1), date(2020, 12, 31), date(2021, 1, 1)
                ),
                DateRange(
                    "2020/21", date(2020, 12, 1), date(2020, 12, 31), date(2021, 1, 1)
                ),
            ),
            1,
        ),
        (
            (
                DateRange(
                    "2023-2024", date(2023, 12, 1), date(2023, 12, 31), date(2024, 1, 1)
                ),
                DateRange(
                    "2023-2024", date(2024, 1, 1), date(2024, 1, 31), date(2024, 2, 1)
                ),
                DateRange(
                    "2023-2024", date(2023, 12, 1), date(2023, 12, 31), date(2024, 1, 1)
                ),
                DateRange(
                    "2023-2024", date(2024, 1, 1), date(2024, 1, 31), date(2024, 2, 1)
                ),
            ),
            2,
        ),
        (
            (
                DateRange(
                    "SeasonA", date(2020, 12, 1), date(2020, 12, 31), date(2021, 1, 1)
                ),
                DateRange(
                    "SeasonB", date(2020, 12, 1), date(2020, 12, 31), date(2021, 1, 1)
                ),
                DateRange(
                    "SeasonA", date(2020, 12, 1), date(2020, 12, 31), date(2021, 1, 1)
                ),
            ),
            2,
        ),
    ),
)
def test_duplicate_date_ranges_value_error(
    date_ranges: Sequence[DateRange], unique_ranges: int
) -> None:
    """Test the `ScenarioDateRanges` class with duplicate date ranges."""
    with pytest.raises(
        ValueError,
        match=(
            f"^Duplicate date ranges found in the input, given {len(date_ranges)} "
            f"date ranges and {unique_ranges} unique date ranges.$"
        ),
    ):
        ScenarioDateRanges(date_ranges)


@pytest.mark.parametrize(
    ("df", "season", "start_date", "end_date", "report_date"),
    (
        (pd.DataFrame(), "season", "start_date", "end_date", "report_date"),
        (
            pd.DataFrame(data={"season": [1]}),
            "season",
            "start_date",
            "end_date",
            "report_date",
        ),
        (
            pd.DataFrame(data={"season": [1], "start_date": [date(2020, 1, 1)]}),
            "season",
            "start_date",
            "end_date",
            "report_date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": [1],
                    "start_date": [date(2020, 1, 1)],
                    "end_date": [date(2020, 1, 31)],
                }
            ),
            "season",
            "start_date",
            "end_date",
            "report_date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": [1],
                    "end_date": [date(2020, 1, 31)],
                    "report_date": [date(2020, 2, 1)],
                }
            ),
            "season",
            "start_date",
            "end_date",
            "report_date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": [1],
                    "start_date": [date(2020, 1, 1)],
                    "end_date": [date(2020, 1, 31)],
                }
            ),
            "season",
            "start_date",
            "end_date",
            "report_date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": [1],
                    "start_date": [date(2020, 1, 1)],
                    "end_date": [date(2020, 1, 31)],
                    "report_date": [date(2020, 2, 1)],
                }
            ),
            "season",
            "start_date",
            "end_date",
            "report_date_typo",
        ),
    ),
)
def test_missing_columns_value_error(
    df: pd.DataFrame, season: str, start_date: str, end_date: str, report_date: str
) -> None:
    """Test the `ScenarioDateRanges` class with missing columns."""
    arg, col = next(
        z
        for z in zip(
            ("season", "start_date", "end_date", "report_date"),
            (season, start_date, end_date, report_date),
        )
        if z[1] not in df.columns
    )
    with pytest.raises(
        ValueError,
        match=f"^Column '{col}' for `{arg}` not found in the DataFrame.$",
    ):
        ScenarioDateRanges.from_pandas(
            df,
            season=season,
            start_date=start_date,
            end_date=end_date,
            report_date=report_date,
        )
