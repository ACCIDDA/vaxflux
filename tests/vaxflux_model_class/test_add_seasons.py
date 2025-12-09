"""Unit tests for the `VaxfluxModel.add_seasons` method."""

from datetime import date

import pytest

from vaxflux._curves import LogisticCurve
from vaxflux._vaxflux_model import VaxfluxModel
from vaxflux.dates import SeasonRange


@pytest.mark.parametrize(
    ("args_factory", "expected_count"),
    [
        # Single season
        (
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
            ),
            1,
        ),
        # Multiple seasons as args
        (
            lambda: (
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
                SeasonRange(
                    season="2025/2026",
                    start_date=date(2025, 12, 1),
                    end_date=date(2026, 3, 31),
                ),
            ),
            3,
        ),
        # List of seasons
        (
            lambda: (
                [
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
                ],
            ),
            2,
        ),
        # Tuple of seasons
        (
            lambda: (
                (
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
                ),
            ),
            2,
        ),
        # Mixed args and sequences
        (
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                [
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
                ],
                SeasonRange(
                    season="2026/2027",
                    start_date=date(2026, 12, 1),
                    end_date=date(2027, 3, 31),
                ),
            ),
            4,
        ),
        # Empty call
        (lambda: (), 0),
        # Empty list
        (lambda: ([],), 0),
        # Adjacent seasons (no overlap) - end date + 1 = next start date
        (
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2024/2025",
                    start_date=date(2024, 4, 1),
                    end_date=date(2024, 7, 31),
                ),
            ),
            2,
        ),
        # Non-sequential but non-overlapping seasons - gap between seasons
        (
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2025/2026",
                    start_date=date(2025, 12, 1),
                    end_date=date(2026, 3, 31),
                ),
                SeasonRange(
                    season="2024/2025",
                    start_date=date(2024, 12, 1),
                    end_date=date(2025, 3, 31),
                ),
            ),
            3,
        ),
    ],
)
def test_add_seasons_input_variations(
    args_factory: object, expected_count: int
) -> None:
    """Test adding seasons with various input patterns."""
    model = VaxfluxModel(curve=LogisticCurve())
    args = args_factory()  # type: ignore[operator]
    result = model.add_seasons(*args)

    # Check method chaining works
    assert result is model
    # Check expected number of seasons were added
    assert len(model._seasons) == expected_count


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


@pytest.mark.parametrize(
    ("setup_seasons", "test_seasons_factory", "match_pattern"),
    [
        # Single duplicate season name in one call
        (
            None,
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2024, 12, 1),
                    end_date=date(2025, 3, 31),
                ),
            ),
            r"Duplicate season names found in new seasons: \['2023/2024'\]\.",
        ),
        # Multiple duplicate season names in one call
        (
            None,
            lambda: (
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
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2025, 12, 1),
                    end_date=date(2026, 3, 31),
                ),
                SeasonRange(
                    season="2024/2025",
                    start_date=date(2026, 12, 1),
                    end_date=date(2027, 3, 31),
                ),
            ),
            (
                r"Duplicate season names found in new "
                r"seasons: \['2023/2024', '2024/2025'\]\."
            ),
        ),
        # Duplicate season name across calls
        (
            [
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                )
            ],
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2024, 12, 1),
                    end_date=date(2025, 3, 31),
                ),
            ),
            r"Season names already exist in the model: \['2023/2024'\]\.",
        ),
    ],
)
def test_add_seasons_duplicate_name_errors(
    setup_seasons: list[SeasonRange] | None,
    test_seasons_factory: object,
    match_pattern: str,
) -> None:
    """Test ValueError is raised for various duplicate season name scenarios."""
    model = VaxfluxModel(curve=LogisticCurve())

    # Add setup seasons if provided (for testing across calls)
    if setup_seasons:
        model.add_seasons(*setup_seasons)

    # Generate test seasons and expect duplicate name error
    test_seasons = test_seasons_factory()  # type: ignore[operator]

    with pytest.raises(ValueError, match=match_pattern):
        model.add_seasons(*test_seasons)


@pytest.mark.parametrize(
    ("setup_seasons", "test_seasons_factory", "match_pattern"),
    [
        # Partial overlap in single call
        (
            None,
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2024 Winter",
                    start_date=date(2024, 3, 1),
                    end_date=date(2024, 5, 31),
                ),
            ),
            r"Overlapping date ranges found: '2023/2024' \(2023-12-01 to 2024-03-31\) "
            r"and '2024 Winter' \(2024-03-01 to 2024-05-31\)\.",
        ),
        # One season completely contains another
        (
            None,
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2024 January",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Identical date ranges (different season names)
        (
            None,
            lambda: (
                SeasonRange(
                    season="2023/2024 A",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2023/2024 B",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Single day overlap
        (
            None,
            lambda: (
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                ),
                SeasonRange(
                    season="2024/2025",
                    start_date=date(2024, 3, 31),
                    end_date=date(2024, 7, 31),
                ),
            ),
            r"Overlapping date ranges found:",
        ),
        # Overlap with existing season (across calls)
        (
            [
                SeasonRange(
                    season="2023/2024",
                    start_date=date(2023, 12, 1),
                    end_date=date(2024, 3, 31),
                )
            ],
            lambda: (
                SeasonRange(
                    season="2024 Winter",
                    start_date=date(2024, 3, 1),
                    end_date=date(2024, 5, 31),
                ),
            ),
            r"Overlapping date ranges found: '2023/2024' \(2023-12-01 to 2024-03-31\) "
            r"and '2024 Winter' \(2024-03-01 to 2024-05-31\)\.",
        ),
    ],
)
def test_add_seasons_overlapping_errors(
    setup_seasons: list[SeasonRange] | None,
    test_seasons_factory: object,
    match_pattern: str,
) -> None:
    """Test ValueError is raised for various overlapping season scenarios."""
    model = VaxfluxModel(curve=LogisticCurve())

    # Add setup seasons if provided (for testing across calls)
    if setup_seasons:
        model.add_seasons(*setup_seasons)

    # Generate test seasons and expect overlap error
    test_seasons = test_seasons_factory()  # type: ignore[operator]

    with pytest.raises(ValueError, match=match_pattern):
        model.add_seasons(*test_seasons)
