"""Unit tests for `VaxfluxInferenceData` helpers."""

from __future__ import annotations

from typing import Any, cast

import xarray as xr

from vaxflux._vaxflux_inference_data import VaxfluxInferenceData


def test_merged_posterior_prefers_posterior_predictive_values() -> None:
    """Test that posterior predictive values win when variables overlap."""
    idata = cast("Any", object.__new__(VaxfluxInferenceData))
    idata.posterior = xr.Dataset(
        {
            "incidence_2025_26_age_18_49_years": (("chain", "draw"), [[1.0]]),
            "posterior_only": (("chain", "draw"), [[10.0]]),
        },
        coords={"chain": [0], "draw": [0]},
    )
    idata.posterior_predictive = xr.Dataset(
        {
            "incidence_2025_26_age_18_49_years": (("chain", "draw"), [[2.0]]),
            "posterior_predictive_only": (("chain", "draw"), [[20.0]]),
        },
        coords={"chain": [0], "draw": [0]},
    )

    merged = cast("VaxfluxInferenceData", idata).merged_posterior

    assert merged["incidence_2025_26_age_18_49_years"].item() == 2.0
    assert merged["posterior_only"].item() == 10.0
    assert merged["posterior_predictive_only"].item() == 20.0
