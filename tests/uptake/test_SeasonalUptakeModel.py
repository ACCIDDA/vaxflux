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
    """Mock test placeholder."""
    curve = LogisticIncidenceCurve()
    uptake_model = SeasonalUptakeModel(curve).date_columns(
        start_date, end_date, report_date
    )
    assert uptake_model._start_date == start_date
    assert uptake_model._end_date == end_date
    assert uptake_model._report_date == report_date
