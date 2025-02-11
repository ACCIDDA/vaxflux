"""Unit tests for the `SeasonalUptakeModel` class."""

import pytest

from vaxflux.curves import LogisticIncidenceCurve
from vaxflux.uptake import SeasonalUptakeModel


@pytest.mark.parametrize("start_date", ("period_start", "start_date"))
@pytest.mark.parametrize("end_date", ("period_end", "end_date"))
@pytest.mark.parametrize("report_date", (None, "date_posted"))
def test_date_columns_method(
    start_date: str, end_date: str, report_date: str | None
) -> None:
    """Test the `date_columns` method."""
    curve = LogisticIncidenceCurve()
    uptake_model = SeasonalUptakeModel(curve).date_columns(
        start_date, end_date, report_date
    )
    assert uptake_model._start_date == start_date
    assert uptake_model._end_date == end_date
    assert uptake_model._report_date == report_date


@pytest.mark.parametrize("season", ("season", "season_name", "year"))
def test_season_method(season: str) -> None:
    """Test the `season_column` method."""
    curve = LogisticIncidenceCurve()
    uptake_model = SeasonalUptakeModel(curve).season_column(season)
    assert uptake_model._season == season
