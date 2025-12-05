"""Unit tests for the `VaxfluxModel.add_seasons` method."""

from datetime import date

import pytest

from vaxflux._curves import LogisticCurve
from vaxflux._vaxflux_model import VaxfluxModel
from vaxflux.dates import SeasonRange


def test_add_seasons_single_season() -> None:
    """Test adding a single season."""
    model = VaxfluxModel(curve=LogisticCurve())
    season = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    result = model.add_seasons(season)

    # Check method chaining works
    assert result is model
    # Check season was added
    assert len(model._seasons) == 1
    assert model._seasons[0] == season


def test_add_seasons_multiple_seasons_as_args() -> None:
    """Test adding multiple seasons as individual arguments."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )
    season3 = SeasonRange(
        season="2025/2026",
        start_date=date(2025, 12, 1),
        end_date=date(2026, 3, 31),
    )

    result = model.add_seasons(season1, season2, season3)

    # Check method chaining works
    assert result is model
    # Check all seasons were added
    assert len(model._seasons) == 3
    assert model._seasons[0] == season1
    assert model._seasons[1] == season2
    assert model._seasons[2] == season3


def test_add_seasons_list() -> None:
    """Test adding seasons as a list."""
    model = VaxfluxModel(curve=LogisticCurve())
    seasons = [
        SeasonRange(
            season="2023/2024",
            start_date=date(2023, 12, 1),
            end_date=date(2024, 3, 31),
        ),
        SeasonRange(
            season="2024/2025",
            start_date=date(2024, 12, 1),
            end_date=date(2025, 3, 31),
        ),
    ]

    result = model.add_seasons(seasons)

    # Check method chaining works
    assert result is model
    # Check all seasons were added
    assert len(model._seasons) == 2
    assert model._seasons[0] == seasons[0]
    assert model._seasons[1] == seasons[1]


def test_add_seasons_tuple() -> None:
    """Test adding seasons as a tuple."""
    model = VaxfluxModel(curve=LogisticCurve())
    seasons = (
        SeasonRange(
            season="2023/2024",
            start_date=date(2023, 12, 1),
            end_date=date(2024, 3, 31),
        ),
        SeasonRange(
            season="2024/2025",
            start_date=date(2024, 12, 1),
            end_date=date(2025, 3, 31),
        ),
    )

    result = model.add_seasons(seasons)

    # Check method chaining works
    assert result is model
    # Check all seasons were added
    assert len(model._seasons) == 2
    assert model._seasons[0] == seasons[0]
    assert model._seasons[1] == seasons[1]


def test_add_seasons_mixed_args_and_sequences() -> None:
    """Test adding seasons with mixed individual and sequence arguments."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    seasons_list = [
        SeasonRange(
            season="2024/2025",
            start_date=date(2024, 12, 1),
            end_date=date(2025, 3, 31),
        ),
        SeasonRange(
            season="2025/2026",
            start_date=date(2025, 12, 1),
            end_date=date(2026, 3, 31),
        ),
    ]
    season4 = SeasonRange(
        season="2026/2027",
        start_date=date(2026, 12, 1),
        end_date=date(2027, 3, 31),
    )

    result = model.add_seasons(season1, seasons_list, season4)

    # Check method chaining works
    assert result is model
    # Check all seasons were added
    assert len(model._seasons) == 4
    assert model._seasons[0] == season1
    assert model._seasons[1] == seasons_list[0]
    assert model._seasons[2] == seasons_list[1]
    assert model._seasons[3] == season4


def test_add_seasons_empty_call() -> None:
    """Test calling add_seasons with no arguments."""
    model = VaxfluxModel(curve=LogisticCurve())
    result = model.add_seasons()

    # Check method chaining works
    assert result is model
    # Check no seasons were added
    assert len(model._seasons) == 0


def test_add_seasons_multiple_calls() -> None:
    """Test multiple calls to add_seasons accumulate."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )

    model.add_seasons(season1)
    assert len(model._seasons) == 1

    model.add_seasons(season2)
    assert len(model._seasons) == 2
    assert model._seasons[0] == season1
    assert model._seasons[1] == season2


def test_add_seasons_method_chaining() -> None:
    """Test method chaining with add_seasons."""
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )

    model = (
        VaxfluxModel(curve=LogisticCurve()).add_seasons(season1).add_seasons(season2)
    )

    assert len(model._seasons) == 2
    assert model._seasons[0] == season1
    assert model._seasons[1] == season2


@pytest.mark.parametrize(
    "invalid_arg",
    [
        "not a season",
        123,
        45.6,
        {"season": "2023/2024"},
        None,
    ],
)
def test_add_seasons_invalid_type_error(invalid_arg: object) -> None:
    """Test TypeError is raised when passing invalid types."""
    model = VaxfluxModel(curve=LogisticCurve())

    with pytest.raises(
        TypeError,
        match=(
            r"Arguments must be SeasonRange objects or sequences of SeasonRange objects"
        ),
    ):
        model.add_seasons(invalid_arg)  # type: ignore[arg-type]


def test_add_seasons_sequence_with_invalid_items() -> None:
    """Test TypeError is raised when a sequence contains non-SeasonRange items."""
    model = VaxfluxModel(curve=LogisticCurve())
    season = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )

    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be SeasonRange objects, got str\.",
    ):
        model.add_seasons([season, "not a season"])  # type: ignore[list-item]


def test_add_seasons_sequence_with_mixed_invalid_items() -> None:
    """Test TypeError is raised with various invalid items in a sequence."""
    model = VaxfluxModel(curve=LogisticCurve())
    season = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )

    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be SeasonRange objects",
    ):
        model.add_seasons([season, 123])  # type: ignore[list-item]


def test_add_seasons_empty_list() -> None:
    """Test adding an empty list."""
    model = VaxfluxModel(curve=LogisticCurve())
    result = model.add_seasons([])

    # Check method chaining works
    assert result is model
    # Check no seasons were added
    assert len(model._seasons) == 0


def test_add_seasons_preserves_existing_seasons() -> None:
    """Test that add_seasons preserves previously added seasons."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )
    season3 = SeasonRange(
        season="2025/2026",
        start_date=date(2025, 12, 1),
        end_date=date(2026, 3, 31),
    )

    model.add_seasons(season1)
    initial_seasons = model._seasons.copy()

    model.add_seasons(season2, season3)

    # Check initial seasons are still there
    assert model._seasons[0] == initial_seasons[0]
    # Check new seasons were added
    assert len(model._seasons) == 3
    assert model._seasons[1] == season2
    assert model._seasons[2] == season3


# Validation tests


def test_add_seasons_duplicate_names_in_single_call() -> None:
    """Test ValueError is raised when duplicate season names are in a single call."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_duplicate = SeasonRange(
        season="2023/2024",  # Same name as season1
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )

    with pytest.raises(
        ValueError,
        match=r"Duplicate season names found in new seasons: \['2023/2024'\]\.",
    ):
        model.add_seasons(season1, season2_duplicate)


def test_add_seasons_duplicate_names_across_calls() -> None:
    """Test ValueError is raised when adding a season with an existing name."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_duplicate = SeasonRange(
        season="2023/2024",  # Same name as season1
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )

    model.add_seasons(season1)

    with pytest.raises(
        ValueError,
        match=r"Season names already exist in the model: \['2023/2024'\]\.",
    ):
        model.add_seasons(season2_duplicate)


def test_add_seasons_multiple_duplicate_names() -> None:
    """Test ValueError lists all duplicate season names."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),
        end_date=date(2025, 3, 31),
    )
    season1_dup = SeasonRange(
        season="2023/2024",
        start_date=date(2025, 12, 1),
        end_date=date(2026, 3, 31),
    )
    season2_dup = SeasonRange(
        season="2024/2025",
        start_date=date(2026, 12, 1),
        end_date=date(2027, 3, 31),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Duplicate season names found in new "
            r"seasons: \['2023/2024', '2024/2025'\]\."
        ),
    ):
        model.add_seasons(season1, season2, season1_dup, season2_dup)


def test_add_seasons_overlapping_dates_in_single_call() -> None:
    """Test ValueError is raised when seasons have overlapping dates."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_overlapping = SeasonRange(
        season="2024 Winter",
        start_date=date(2024, 3, 1),  # Overlaps with season1
        end_date=date(2024, 5, 31),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Overlapping date ranges found: '2023/2024' \(2023-12-01 to 2024-03-31\) "
            r"and '2024 Winter' \(2024-03-01 to 2024-05-31\)\."
        ),
    ):
        model.add_seasons(season1, season2_overlapping)


def test_add_seasons_overlapping_dates_across_calls() -> None:
    """Test ValueError is raised when new season overlaps with existing season."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_overlapping = SeasonRange(
        season="2024 Winter",
        start_date=date(2024, 3, 1),  # Overlaps with season1
        end_date=date(2024, 5, 31),
    )

    model.add_seasons(season1)

    with pytest.raises(
        ValueError,
        match=(
            r"Overlapping date ranges found: '2023/2024' \(2023-12-01 to 2024-03-31\) "
            r"and '2024 Winter' \(2024-03-01 to 2024-05-31\)\."
        ),
    ):
        model.add_seasons(season2_overlapping)


def test_add_seasons_completely_overlapping_dates() -> None:
    """Test ValueError when one season completely contains another."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_contained = SeasonRange(
        season="2024 January",
        start_date=date(2024, 1, 1),  # Inside season1
        end_date=date(2024, 1, 31),
    )

    with pytest.raises(
        ValueError,
        match=r"Overlapping date ranges found:",
    ):
        model.add_seasons(season1, season2_contained)


def test_add_seasons_same_dates() -> None:
    """Test ValueError when seasons have identical date ranges."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024 A",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_same_dates = SeasonRange(
        season="2023/2024 B",
        start_date=date(2023, 12, 1),  # Same dates
        end_date=date(2024, 3, 31),
    )

    with pytest.raises(
        ValueError,
        match=r"Overlapping date ranges found:",
    ):
        model.add_seasons(season1, season2_same_dates)


def test_add_seasons_adjacent_dates_no_overlap() -> None:
    """Test that adjacent seasons (no overlap) are allowed."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_adjacent = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 4, 1),  # Day after season1 ends
        end_date=date(2024, 7, 31),
    )

    # Should not raise an error
    result = model.add_seasons(season1, season2_adjacent)

    assert result is model
    assert len(model._seasons) == 2
    assert model._seasons[0] == season1
    assert model._seasons[1] == season2_adjacent


def test_add_seasons_single_day_overlap() -> None:
    """Test ValueError when seasons overlap by a single day."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2_one_day_overlap = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 3, 31),  # Same as season1 end date
        end_date=date(2024, 7, 31),
    )

    with pytest.raises(
        ValueError,
        match=r"Overlapping date ranges found:",
    ):
        model.add_seasons(season1, season2_one_day_overlap)


def test_add_seasons_non_sequential_but_non_overlapping() -> None:
    """Test that non-sequential but non-overlapping seasons are allowed."""
    model = VaxfluxModel(curve=LogisticCurve())
    season1 = SeasonRange(
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 3, 31),
    )
    season2 = SeasonRange(
        season="2025/2026",
        start_date=date(2025, 12, 1),  # Gap between seasons
        end_date=date(2026, 3, 31),
    )
    season3 = SeasonRange(
        season="2024/2025",
        start_date=date(2024, 12, 1),  # Fills the gap
        end_date=date(2025, 3, 31),
    )

    # Should not raise an error
    result = model.add_seasons(season1, season2, season3)

    assert result is model
    assert len(model._seasons) == 3
