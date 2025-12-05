"""Unit tests for the `VaxfluxModel.add_dates` method."""

from datetime import date

import pytest

from vaxflux._curves import LogisticCurve
from vaxflux._vaxflux_model import VaxfluxModel
from vaxflux.dates import DateRange


@pytest.mark.parametrize(
    ("args_factory", "expected_count"),
    [
        # Single date range
        (
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
            ),
            1,
        ),
        # Multiple date ranges as args
        (
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 8),
                    end_date=date(2023, 12, 14),
                    report_date=date(2023, 12, 15),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 15),
                    end_date=date(2023, 12, 21),
                    report_date=date(2023, 12, 22),
                ),
            ),
            3,
        ),
        # List of date ranges
        (
            lambda: (
                [
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 1),
                        end_date=date(2023, 12, 7),
                        report_date=date(2023, 12, 8),
                    ),
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 8),
                        end_date=date(2023, 12, 14),
                        report_date=date(2023, 12, 15),
                    ),
                ],
            ),
            2,
        ),
        # Tuple of date ranges
        (
            lambda: (
                (
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 1),
                        end_date=date(2023, 12, 7),
                        report_date=date(2023, 12, 8),
                    ),
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 8),
                        end_date=date(2023, 12, 14),
                        report_date=date(2023, 12, 15),
                    ),
                ),
            ),
            2,
        ),
        # Mixed args and sequences
        (
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                [
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 8),
                        end_date=date(2023, 12, 14),
                        report_date=date(2023, 12, 15),
                    ),
                    DateRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 15),
                        end_date=date(2023, 12, 21),
                        report_date=date(2023, 12, 22),
                    ),
                ],
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 22),
                    end_date=date(2023, 12, 28),
                    report_date=date(2023, 12, 29),
                ),
            ),
            4,
        ),
        # Empty call
        (lambda: (), 0),
        # Empty list
        (lambda: ([],), 0),
        # Adjacent date ranges (no overlap) - end date + 1 = next start date
        (
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 8),
                    end_date=date(2023, 12, 14),
                    report_date=date(2023, 12, 15),
                ),
            ),
            2,
        ),
        # Non-overlapping date ranges from different seasons
        (
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2024/2025",
                    start_date=date(2024, 12, 1),
                    end_date=date(2024, 12, 7),
                    report_date=date(2024, 12, 8),
                ),
            ),
            2,
        ),
    ],
    ids=[
        "single_date_range",
        "multiple_date_ranges_as_args",
        "list_of_date_ranges",
        "tuple_of_date_ranges",
        "mixed_args_and_sequences",
        "empty_call",
        "empty_list",
        "adjacent_date_ranges_no_overlap",
        "different_seasons_non_overlapping",
    ],
)
def test_add_dates_input_variations(args_factory: object, expected_count: int) -> None:
    """Test adding date ranges with various input patterns."""
    model = VaxfluxModel(curve=LogisticCurve())
    args = args_factory()  # type: ignore[operator]
    result = model.add_dates(*args)

    # Check method chaining works
    assert result is model
    # Check expected number of date ranges were added
    assert len(model._dates) == expected_count


def test_add_dates_preserves_existing_dates() -> None:
    """Test that add_dates preserves previously added date ranges."""
    model = VaxfluxModel(curve=LogisticCurve())
    dr1 = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 7),
        report_date=date(2023, 12, 8),
    )
    dr2 = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 8),
        end_date=date(2023, 12, 14),
        report_date=date(2023, 12, 15),
    )
    dr3 = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 15),
        end_date=date(2023, 12, 21),
        report_date=date(2023, 12, 22),
    )

    model.add_dates(dr1)
    initial_dates = model._dates.copy()

    model.add_dates(dr2, dr3)

    # Check initial date ranges are still there
    assert model._dates[0] == initial_dates[0]
    # Check new date ranges were added
    assert len(model._dates) == 3
    assert model._dates[1] == dr2
    assert model._dates[2] == dr3


@pytest.mark.parametrize(
    "invalid_arg",
    [
        "not a date range",
        123,
        45.6,
        {"season": "2023/2024"},
        None,
    ],
)
def test_add_dates_invalid_type_error(invalid_arg: object) -> None:
    """Test TypeError is raised when passing invalid types."""
    model = VaxfluxModel(curve=LogisticCurve())

    with pytest.raises(
        TypeError,
        match=r"Arguments must be DateRange objects or sequences of DateRange objects",
    ):
        model.add_dates(invalid_arg)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_item",
    ["not a date range", 123, {"key": "value"}],
    ids=["string", "int", "dict"],
)
def test_add_dates_sequence_with_invalid_items(invalid_item: object) -> None:
    """Test TypeError is raised when a sequence contains non-DateRange items."""
    model = VaxfluxModel(curve=LogisticCurve())
    dr = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 7),
        report_date=date(2023, 12, 8),
    )

    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be DateRange objects",
    ):
        model.add_dates([dr, invalid_item])  # type: ignore[list-item]


def test_add_dates_exact_duplicate_in_single_call() -> None:
    """Test ValueError is raised when duplicate date ranges are in a single call."""
    model = VaxfluxModel(curve=LogisticCurve())
    dr = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 7),
        report_date=date(2023, 12, 8),
    )

    with pytest.raises(
        ValueError,
        match=r"Duplicate date range found:",
    ):
        model.add_dates(dr, dr)


def test_add_dates_exact_duplicate_across_calls() -> None:
    """Test ValueError is raised when adding an exact duplicate date range."""
    model = VaxfluxModel(curve=LogisticCurve())
    dr = DateRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 7),
        report_date=date(2023, 12, 8),
    )

    model.add_dates(dr)

    with pytest.raises(
        ValueError,
        match=r"Date range already exists:",
    ):
        model.add_dates(dr)


@pytest.mark.parametrize(
    ("setup_dates", "test_dates_factory", "match_pattern"),
    [
        # Partial overlap in single call
        (
            None,
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 5),
                    end_date=date(2023, 12, 12),
                    report_date=date(2023, 12, 13),
                ),
            ),
            r"Overlapping date ranges found: '2023/2024' \(2023-12-01 to 2023-12-07\) "
            r"and '2023/2024' \(2023-12-05 to 2023-12-12\)\.",
        ),
        # One date range completely contains another
        (
            None,
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 31),
                    report_date=date(2024, 1, 1),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 10),
                    end_date=date(2023, 12, 15),
                    report_date=date(2023, 12, 16),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Same dates, different report date
        (
            None,
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 9),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Single day overlap
        (
            None,
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                ),
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 7),
                    end_date=date(2023, 12, 14),
                    report_date=date(2023, 12, 15),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Different seasons overlapping
        (
            None,
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 7),
                    report_date=date(2024, 1, 8),
                ),
                DateRange(
                    season="2024/2025",
                    start_date=date(2024, 1, 5),
                    end_date=date(2024, 1, 12),
                    report_date=date(2024, 1, 13),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Overlap with existing date range (across calls)
        (
            [
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2023, 12, 7),
                    report_date=date(2023, 12, 8),
                )
            ],
            lambda: (
                DateRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 5),
                    end_date=date(2023, 12, 12),
                    report_date=date(2023, 12, 13),
                ),
            ),
            r"Overlapping date ranges found: existing '2023/2024'",
        ),
    ],
    ids=[
        "partial_overlap_single_call",
        "complete_containment",
        "same_dates_different_report",
        "single_day_overlap",
        "different_seasons_overlapping",
        "overlap_with_existing",
    ],
)
def test_add_dates_overlapping_errors(
    setup_dates: list[DateRange] | None,
    test_dates_factory: object,
    match_pattern: str,
) -> None:
    """Test ValueError is raised for various overlapping date range scenarios."""
    model = VaxfluxModel(curve=LogisticCurve())

    # Add setup dates if provided (for testing across calls)
    if setup_dates:
        model.add_dates(*setup_dates)

    # Generate test dates and expect overlap error
    test_dates = test_dates_factory()  # type: ignore[operator]

    with pytest.raises(ValueError, match=match_pattern):
        model.add_dates(*test_dates)
