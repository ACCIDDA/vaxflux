"""Unit tests for `VaxfluxObservations.to_date_ranges`."""

from datetime import date

import pandas as pd
import pytest

from vaxflux import DateRange
from vaxflux._vaxflux_observations import VaxfluxObservations


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        (
            pd.DataFrame(
                data={
                    "season": ["2023/2024"],
                    "start_date": ["2023-10-01"],
                    "end_date": ["2023-10-31"],
                    "report_date": ["2023-11-01"],
                    "type": ["incidence"],
                    "value": [0.2],
                },
            ),
            [
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 10, 1),
                    end_date=date(2023, 10, 31),
                    report_date=date(2023, 11, 1),
                )
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2023/2024", "2023/2024", "2023/2024"],
                    "start_date": ["2023-10-01", "2023-10-01", "2023-11-01"],
                    "end_date": ["2023-10-31", "2023-10-31", "2023-11-30"],
                    "report_date": ["2023-11-01", "2023-11-01", "2023-12-01"],
                    "type": ["incidence", "incidence", "incidence"],
                    "value": [0.1, 0.1, 0.2],
                },
            ),
            [
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 10, 1),
                    end_date=date(2023, 10, 31),
                    report_date=date(2023, 11, 1),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 11, 1),
                    end_date=date(2023, 11, 30),
                    report_date=date(2023, 12, 1),
                ),
            ],
        ),
        (
            pd.DataFrame(
                data={
                    "season": ["2024/2025", "2023/2024", "2024/2025", "2023/2024"],
                    "start_date": [
                        "2024-11-01",
                        "2023-10-01",
                        "2024-10-01",
                        "2023-11-01",
                    ],
                    "end_date": [
                        "2024-11-30",
                        "2023-10-31",
                        "2024-10-31",
                        "2023-11-30",
                    ],
                    "report_date": [
                        "2024-12-01",
                        "2023-11-01",
                        "2024-11-01",
                        "2023-12-01",
                    ],
                    "type": ["incidence", "incidence", "incidence", "incidence"],
                    "value": [0.4, 0.1, 0.3, 0.2],
                },
            ),
            [
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 10, 1),
                    end_date=date(2023, 10, 31),
                    report_date=date(2023, 11, 1),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 11, 1),
                    end_date=date(2023, 11, 30),
                    report_date=date(2023, 12, 1),
                ),
                DateRange(
                    season="2024/2025",
                    start_date=date(2024, 10, 1),
                    end_date=date(2024, 10, 31),
                    report_date=date(2024, 11, 1),
                ),
                DateRange(
                    season="2024/2025",
                    start_date=date(2024, 11, 1),
                    end_date=date(2024, 11, 30),
                    report_date=date(2024, 12, 1),
                ),
            ],
        ),
    ],
)
def test_to_date_ranges_extracts_distinct_date_ranges(
    observations: pd.DataFrame,
    expected: list[DateRange],
) -> None:
    """Date ranges are deduplicated and sorted from the observations."""
    assert VaxfluxObservations.from_dataframe(observations).to_date_ranges() == expected
