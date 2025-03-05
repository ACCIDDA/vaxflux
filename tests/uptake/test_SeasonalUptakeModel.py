"""Unit tests for the `SeasonalUptakeModel` class."""

import pytest

from vaxflux.covariates import PooledCovariate
from vaxflux.curves import LogisticIncidenceCurve
from vaxflux.uptake import SeasonalUptakeModel


def test_covariate_parameters_missing_value_error() -> None:
    """Covariate parameter not present in curve parameters raises a value error."""
    curve = LogisticIncidenceCurve()
    assert "a" not in curve.parameters
    with pytest.raises(ValueError):
        SeasonalUptakeModel(
            curve,
            [
                PooledCovariate(
                    parameter="a",
                    covariate=None,
                    distribution="Normal",
                    distribution_kwargs={"mu": 0.0, "sigma": 1.0},
                ),
            ],
        )
