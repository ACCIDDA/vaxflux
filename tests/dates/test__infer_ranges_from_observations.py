"""Unit tests for the `vaxflux.dates.infer_ranges_from_observations` internal helper."""

from typing import Literal

import pandas as pd
import pytest

from vaxflux.dates import (
    _INFER_RANGES_REQUIRED_COLUMNS,
    DateOrSeasonRange,
    DateRange,
    SeasonRange,
    _infer_ranges_from_observations,
)


@pytest.mark.parametrize("mode", ["date", "season"])
def test_at_least_one_of_observations_or_ranges_required_value_error(
    mode: Literal["date", "season"],
) -> None:
    """Test that at least one of observations or ranges is provided."""
    with pytest.raises(
        ValueError, match="^At least one of `observations` or `ranges` is required.$"
    ):
        _infer_ranges_from_observations(None, [], mode)


@pytest.mark.parametrize(
    ("mode", "observations_columns"),
    [
        ("date", {"a", "b", "c"}),
        ("date", {"season", "start_date", "end_date"}),
        ("date", {"season", "season_start_date", "season_end_date"}),
        ("season", {"a", "b", "c"}),
        ("season", {"season", "season_start_date"}),
        ("season", {"season", "start_date", "end_date", "report_date"}),
    ],
)
def test_infer_ranges_from_observations_missing_columns_value_error(
    mode: Literal["date", "season"], observations_columns: set[str]
) -> None:
    """Test that the required columns are present in the observations."""
    missing_columns = _INFER_RANGES_REQUIRED_COLUMNS[mode] - set(observations_columns)
    assert len(missing_columns) > 0
    observations = pd.DataFrame(data={col: [] for col in observations_columns})
    with pytest.raises(
        ValueError,
        match=f"^Missing required columns in the observations: {missing_columns}.$",
    ):
        _infer_ranges_from_observations(observations, [], mode)


@pytest.mark.parametrize(
    ("ranges", "mode"),
    [
        (
            [
                DateRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    report_date="2023-12-31",
                )
            ],
            "date",
        ),
        (
            [
                DateRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    report_date="2023-12-31",
                ),
                DateRange(
                    season="2024",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    report_date="2024-12-31",
                ),
            ],
            "date",
        ),
        (
            [
                DateRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2023-11-30",
                    report_date="2023-12-01",
                ),
                DateRange(
                    season="2023/24",
                    start_date="2023-12-01",
                    end_date="2023-12-31",
                    report_date="2024-01-01",
                ),
                DateRange(
                    season="2023/24",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    report_date="2024-02-01",
                ),
                DateRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2024-11-30",
                    report_date="2024-12-01",
                ),
                DateRange(
                    season="2024/25",
                    start_date="2024-12-01",
                    end_date="2024-12-31",
                    report_date="2024-01-01",
                ),
                DateRange(
                    season="2024/25",
                    start_date="2025-01-01",
                    end_date="2025-01-31",
                    report_date="2025-02-01",
                ),
            ],
            "date",
        ),
        (
            [
                SeasonRange(
                    season="2023", start_date="2023-01-01", end_date="2023-12-31"
                )
            ],
            "season",
        ),
        (
            [
                SeasonRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                ),
                SeasonRange(
                    season="2024",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                ),
            ],
            "season",
        ),
        (
            [
                SeasonRange(
                    season="2023/24", start_date="2023-11-01", end_date="2024-01-31"
                ),
                SeasonRange(
                    season="2024/25", start_date="2024-11-01", end_date="2025-01-31"
                ),
                SeasonRange(
                    season="2025/26", start_date="2025-11-01", end_date="2026-01-31"
                ),
            ],
            "season",
        ),
    ],
)
def test_if_only_ranges_are_provided_input_matches_output(
    ranges: list[DateOrSeasonRange], mode: Literal["date", "season"]
) -> None:
    """Test that if only ranges are provided, the input matches the output."""
    assert _infer_ranges_from_observations(None, ranges, mode) == ranges
