"""Unit tests for the `ScenarioDateRanges` class."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest

from vaxflux.uptake import DateRange, ScenarioDateRanges


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


@pytest.mark.parametrize(
    ("scenario_date_ranges_one", "scenario_date_ranges_two", "expected"),
    (
        (
            ScenarioDateRanges(
                date_ranges={
                    DateRange(
                        season="2020/21",
                        start_date=date(2020, 12, 1),
                        end_date=date(2020, 12, 31),
                        report_date=date(2021, 1, 1),
                    )
                }
            ),
            ScenarioDateRanges(
                date_ranges={
                    DateRange(
                        season="2020/21",
                        start_date=date(2020, 12, 1),
                        end_date=date(2020, 12, 31),
                        report_date=date(2021, 1, 1),
                    )
                }
            ),
            ScenarioDateRanges(
                date_ranges={
                    DateRange(
                        season="2020/21",
                        start_date=date(2020, 12, 1),
                        end_date=date(2020, 12, 31),
                        report_date=date(2021, 1, 1),
                    )
                }
            ),
        ),
    ),
)
def test_union_method(
    scenario_date_ranges_one: ScenarioDateRanges,
    scenario_date_ranges_two: ScenarioDateRanges,
    expected: ScenarioDateRanges,
) -> None:
    """Test the `ScenarioDateRanges.union` method."""
    assert scenario_date_ranges_one.union(scenario_date_ranges_two) == expected
    assert scenario_date_ranges_two.union(scenario_date_ranges_one) == expected
