"""Unit tests for `VaxfluxObservations.to_season_ranges`."""

from datetime import date

import pandas as pd
import pytest

from vaxflux import SeasonRange
from vaxflux._vaxflux_observations import VaxfluxObservations


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        (
            pd.DataFrame(
                data={
                    "season": ["2023/2024"],
                    "start_date": ["2023-10-01"],
                    "end_date": ["2024-03-31"],
                    "type": ["incidence"],
                    "value": [0.2],
                },
            ),
            [
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 10, 1),
                    end_date=date(2024, 3, 31),
                )
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023/2024", "2023/2024", "2024/2025", "2024/2025"],
                    "start_date": [
                        "2023-11-01",
                        "2023-12-01",
                        "2024-10-01",
                        "2024-11-01",
                    ],
                    "end_date": [
                        "2023-11-30",
                        "2024-02-29",
                        "2024-10-31",
                        "2025-03-31",
                    ],
                    "type": ["incidence", "incidence", "incidence", "incidence"],
                    "value": [0.1, 0.2, 0.3, 0.4],
                },
            ),
            [
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 11, 1),
                    end_date=date(2024, 2, 29),
                ),
                SeasonRange(
                    season="2024/2025",
                    start_date=date(2024, 10, 1),
                    end_date=date(2025, 3, 31),
                ),
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2025", "2024", "2025", "2024"],
                    "start_date": [
                        "2025-01-02",
                        "2024-01-05",
                        "2025-01-01",
                        "2024-01-01",
                    ],
                    "end_date": [
                        "2025-01-09",
                        "2024-01-12",
                        "2025-01-16",
                        "2024-01-19",
                    ],
                    "type": ["incidence", "incidence", "incidence", "incidence"],
                    "value": [0.4, 0.1, 0.5, 0.2],
                },
            ),
            [
                SeasonRange(
                    season="2024",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 19),
                ),
                SeasonRange(
                    season="2025",
                    start_date=date(2025, 1, 1),
                    end_date=date(2025, 1, 16),
                ),
            ],
        ),
    ],
)
def test_to_season_ranges_extracts_season_bounds(
    observations: pd.DataFrame,
    expected: list[SeasonRange],
) -> None:
    """Season ranges are derived from the min start and max end per season."""
    assert (
        VaxfluxObservations.from_dataframe(observations).to_season_ranges() == expected
    )
