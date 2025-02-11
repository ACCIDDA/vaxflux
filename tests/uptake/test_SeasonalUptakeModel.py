"""Unit tests for the `SeasonalUptakeModel` class."""

from typing import Any, TypedDict

import pymc as pm
import pytest

from vaxflux.curves import LogisticIncidenceCurve
from vaxflux.uptake import SeasonalUptakeModel


class SetSeasonalParameterKwargs(TypedDict):
    """Type for the `set_seasonal_parameter` method keyword arguments."""

    parameter: str
    distribution: Any
    distribution_kwargs: dict[str, Any]
    hierarchical: bool


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


@pytest.mark.parametrize("parameter", ("a", "b", "c"))
def test_set_seasonal_parameter_given_parameter_not_used_value_error(
    parameter: str,
) -> None:
    """Test the `set_seasonal_parameter` method."""
    curve = LogisticIncidenceCurve()
    assert parameter not in curve.parameters
    with pytest.raises(
        ValueError,
        match=f"^The parameter '{parameter}' is not used by the curve family.$",
    ):
        SeasonalUptakeModel(curve).set_seasonal_parameter(
            parameter, pm.Normal, {"mu": 0.0, "sigma": 1.0}
        )


@pytest.mark.parametrize(
    "first_call_kwargs",
    (
        {
            "parameter": "m",
            "distribution": pm.Normal,
            "distribution_kwargs": {"mu": 0.0, "sigma": 0.05},
            "hierarchical": True,
        },
        {
            "parameter": "r",
            "distribution": pm.Normal,
            "distribution_kwargs": {"mu": 0.7, "sigma": 0.1},
            "hierarchical": False,
        },
    ),
)
@pytest.mark.parametrize(
    "second_call_kwargs",
    (
        None,
        {
            "parameter": "r",
            "distribution": pm.Normal,
            "distribution_kwargs": {"mu": 1.2, "sigma": 0.3},
            "hierarchical": True,
        },
    ),
)
def test_set_seasonal_parameter_method(
    first_call_kwargs: SetSeasonalParameterKwargs,
    second_call_kwargs: SetSeasonalParameterKwargs | None,
) -> None:
    """Test the `set_seasonal_parameter` method."""
    curve = LogisticIncidenceCurve()
    uptake_model = SeasonalUptakeModel(curve)
    assert len(uptake_model._seasonal_parameters) == 0
    uptake_model.set_seasonal_parameter(**first_call_kwargs)
    assert len(uptake_model._seasonal_parameters) == 1
    assert uptake_model._seasonal_parameters[first_call_kwargs["parameter"]] == (
        first_call_kwargs["distribution"],
        first_call_kwargs["distribution_kwargs"],
        first_call_kwargs["hierarchical"],
    )
    if second_call_kwargs is not None:
        uptake_model.set_seasonal_parameter(**second_call_kwargs)
        assert len(uptake_model._seasonal_parameters) == (
            1
            if first_call_kwargs["parameter"] == second_call_kwargs["parameter"]
            else 2
        )
        assert uptake_model._seasonal_parameters[second_call_kwargs["parameter"]] == (
            second_call_kwargs["distribution"],
            second_call_kwargs["distribution_kwargs"],
            second_call_kwargs["hierarchical"],
        )
