"""Unit tests for the `vaxflux.dates.infer_ranges_from_observations` internal helper."""

from typing import Literal

import pytest

from vaxflux.dates import _infer_ranges_from_observations


@pytest.mark.parametrize("mode", ["season", "date"])
def test_at_least_one_of_observations_or_ranges_required_value_error(
    mode: Literal["season", "date"],
) -> None:
    """Test that at least one of observations or ranges is provided."""
    with pytest.raises(
        ValueError, match="^At least one of `observations` or `ranges` is required.$"
    ):
        _infer_ranges_from_observations(None, [], mode)
