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
        ValueError,
        match="^At least one of `observations` or `ranges` is required.$",
    ):
        _infer_ranges_from_observations(None, [], mode)


@pytest.mark.parametrize(
    ("mode", "observations_columns"),
    [
        ("date", {"a", "b", "c"}),
        ("date", {"season", "start_date", "end_date"}),
        ("date", {"season", "season_start_date", "season_end_date"}),
        ("season", {"a", "b", "c"}),
        ("season", {"season", "start_date"}),
        ("season", {"season", "report_date"}),
    ],
)
def test_infer_ranges_from_observations_missing_columns_value_error(
    mode: Literal["date", "season"],
    observations_columns: set[str],
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
    ("observations", "ranges"),
    [
        (
            pd.DataFrame(
                data={
                    "season": ["2023"],
                    "start_date": ["2023-01-01"],
                    "end_date": ["2023-12-31"],
                },
            ),
            [
                SeasonRange(
                    season="2024",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                ),
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023", "2024"],
                    "start_date": ["2023-01-01", "2024-01-01"],
                    "end_date": ["2023-12-31", "2024-12-31"],
                },
            ),
            [
                SeasonRange(
                    season="2024",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                ),
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023", "2024"],
                    "start_date": ["2023-01-01", "2024-01-01"],
                    "end_date": ["2023-12-31", "2024-12-31"],
                },
            ),
            [
                SeasonRange(
                    season="2025",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                ),
                SeasonRange(
                    season="2026",
                    start_date="2026-01-01",
                    end_date="2026-12-31",
                ),
            ],
        ),
    ],
)
def test_observation_season_ranges_inconsistent_with_explicit_ranges_value_error(
    observations: pd.DataFrame,
    ranges: list[SeasonRange],
) -> None:
    """Test that the observed season ranges are consistent with the explicit ranges."""
    season_names = ", ".join(
        sorted(set(observations["season"].unique()) - {r.season for r in ranges}),
    )
    with pytest.raises(
        ValueError,
        match=(
            "^The observed season ranges are not consistent with the "
            f"explicit season ranges, not accounting for: {season_names}.$"
        ),
    ):
        _infer_ranges_from_observations(observations, ranges, "season")


@pytest.mark.parametrize(
    ("observations", "ranges", "mode"),
    [
        (
            None,
            [
                DateRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    report_date="2023-12-31",
                ),
            ],
            "date",
        ),
        (
            None,
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
            None,
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
                    report_date="2025-01-01",
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
            pd.DataFrame(
                data={
                    "season": ["2023"],
                    "start_date": ["2023-01-01"],
                    "end_date": ["2023-12-31"],
                    "report_date": ["2023-12-31"],
                },
            ),
            [],
            "date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023"],
                    "start_date": ["2023-01-01"],
                    "end_date": ["2023-12-31"],
                    "report_date": ["2023-12-31"],
                },
            ),
            [
                DateRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    report_date="2023-12-31",
                ),
            ],
            "date",
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24", "2023/24"],
                    "start_date": ["2023-12-01", "2024-01-01"],
                    "end_date": ["2023-12-31", "2024-01-31"],
                    "report_date": ["2024-01-01", "2024-02-01"],
                },
            ),
            [
                DateRange(
                    season="2024/25",
                    start_date="2024-12-01",
                    end_date="2024-12-31",
                    report_date="2025-01-01",
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
            None,
            [
                SeasonRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                ),
            ],
            "season",
        ),
        (
            None,
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
            None,
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
                SeasonRange(
                    season="2025/26",
                    start_date="2025-11-01",
                    end_date="2026-01-31",
                ),
            ],
            "season",
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023"],
                    "start_date": ["2023-01-01"],
                    "end_date": ["2023-12-31"],
                },
            ),
            [],
            "season",
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023"],
                    "start_date": ["2023-01-01"],
                    "end_date": ["2023-12-31"],
                },
            ),
            [
                SeasonRange(
                    season="2023",
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                ),
            ],
            "season",
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24", "2024/25"],
                    "start_date": ["2023-11-01", "2024-11-01"],
                    "end_date": ["2024-01-31", "2025-01-31"],
                },
            ),
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
                SeasonRange(
                    season="2025/26",
                    start_date="2025-11-01",
                    end_date="2026-01-31",
                ),
            ],
            "season",
        ),
    ],
)
def test_ranges_is_subset_of_output_when_observations_provided(
    observations: pd.DataFrame | None,
    ranges: list[DateOrSeasonRange],
    mode: Literal["date", "season"],
) -> None:
    """Test that ranges are a subset of the output when observations are provided."""
    output = _infer_ranges_from_observations(observations, ranges, mode)
    assert isinstance(output, list)
    cls = DateRange if mode == "date" else SeasonRange
    assert all(isinstance(r, cls) for r in output)
    assert (
        (output == ranges)
        if observations is None
        else set(ranges).issubset(set(output))
    )


@pytest.mark.parametrize(
    ("observations", "ranges", "mode", "expected"),
    [
        # No observations, only ranges
        (
            None,
            [
                DateRange(
                    season="2023/24",
                    start_date="2023-12-01",
                    end_date="2023-12-31",
                    report_date="2024-01-01",
                ),
            ],
            "date",
            [
                DateRange(
                    season="2023/24",
                    start_date="2023-12-01",
                    end_date="2023-12-31",
                    report_date="2024-01-01",
                ),
            ],
        ),
        # No ranges, only observations
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24"],
                    "start_date": ["2023-12-01"],
                    "end_date": ["2023-12-31"],
                    "report_date": ["2024-01-01"],
                },
            ),
            [],
            "date",
            [
                DateRange(
                    season="2023/24",
                    start_date="2023-12-01",
                    end_date="2023-12-31",
                    report_date="2024-01-01",
                ),
            ],
        ),
        # Non-overlapping observations and ranges
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24"],
                    "start_date": ["2023-12-01"],
                    "end_date": ["2023-12-31"],
                    "report_date": ["2024-01-01"],
                },
            ),
            [
                DateRange(
                    season="2023/24",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    report_date="2024-02-01",
                ),
            ],
            "date",
            [
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
            ],
        ),
        # Ranges contained within observations
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24", "2023/24"],
                    "start_date": ["2023-12-01", "2024-01-01"],
                    "end_date": ["2023-12-31", "2024-01-31"],
                    "report_date": ["2024-01-01", "2024-02-01"],
                },
            ),
            [
                DateRange(
                    season="2023/24",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    report_date="2024-02-01",
                ),
            ],
            "date",
            [
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
            ],
        ),
        # Duplicated ranges contained within observations
        (
            pd.DataFrame(
                data={
                    "season": 10 * ["2023/24"],
                    "start_date": 10 * ["2023-12-01"],
                    "end_date": 10 * ["2023-12-31"],
                    "report_date": 10 * ["2024-01-01"],
                    "value": list(range(10)),
                },
            ),
            [],
            "date",
            [
                DateRange(
                    season="2023/24",
                    start_date="2023-12-01",
                    end_date="2023-12-31",
                    report_date="2024-01-01",
                ),
            ],
        ),
        # No observations, only ranges
        (
            None,
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
            ],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
            ],
        ),
        # No ranges, only observations
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24"],
                    "start_date": ["2023-11-01"],
                    "end_date": ["2024-01-31"],
                },
            ),
            [],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
            ],
        ),
        # Observations and ranges are overlapping
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24"],
                    "start_date": ["2023-11-01"],
                    "end_date": ["2024-01-31"],
                },
            ),
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
            ],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
            ],
        ),
        # Observations are contained within ranges
        (
            pd.DataFrame(
                data={
                    "season": ["2023/24"],
                    "start_date": ["2023-11-01"],
                    "end_date": ["2024-01-31"],
                },
            ),
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
            ],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
                SeasonRange(
                    season="2024/25",
                    start_date="2024-11-01",
                    end_date="2025-01-31",
                ),
            ],
        ),
        # Duplicated ranges contained within observations
        (
            pd.DataFrame(
                data={
                    "season": 10 * ["2023/24"],
                    "start_date": 10 * ["2023-11-01"],
                    "end_date": 10 * ["2024-01-31"],
                },
            ),
            [],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
            ],
        ),
        # Inferred season ranges from observations
        (
            pd.DataFrame(
                data={
                    "season": 3 * ["2023/24"],
                    "start_date": ["2023-11-01", "2023-12-01", "2024-01-01"],
                    "end_date": ["2023-11-30", "2023-12-31", "2024-01-31"],
                    "report_date": ["2023-12-01", "2024-01-01", "2024-02-01"],
                    "value": list(range(3)),
                },
            ),
            [],
            "season",
            [
                SeasonRange(
                    season="2023/24",
                    start_date="2023-11-01",
                    end_date="2024-01-31",
                ),
            ],
        ),
    ],
)
def test_exact_outputs_for_select_inputs(
    observations: pd.DataFrame | None,
    ranges: list[DateOrSeasonRange],
    mode: Literal["date", "season"],
    expected: list[DateOrSeasonRange],
) -> None:
    """Test that the exact outputs are returned for select inputs."""
    output = _infer_ranges_from_observations(observations, ranges, mode)
    assert isinstance(output, list)
    cls = DateRange if mode == "date" else SeasonRange
    assert all(isinstance(r, cls) for r in output)
    assert set(output) == set(expected)
