"""Unit tests for the `VaxfluxModel.add_implementations` method."""

from datetime import date

import pytest

from vaxflux._curves import LogisticCurve
from vaxflux._interventions import Implementation, Intervention
from vaxflux._vaxflux_model import VaxfluxModel


def test_add_implementations_raises_when_intervention_missing() -> None:
    """Adding implementations without interventions raises a `ValueError`."""
    model = VaxfluxModel(curve=LogisticCurve())
    implementation = Implementation(
        intervention="tv_ads",
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 31),
        covariate_categories=None,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^The following implementation interventions have not been added "
            r"to the model: .*.$"
        ),
    ):
        model.add_implementations(implementation)


def test_add_implementations_allows_known_intervention() -> None:
    """Adding implementations with known interventions succeeds."""
    model = VaxfluxModel(curve=LogisticCurve())
    model.add_interventions(
        Intervention(
            name="tv_ads",
            parameter="m",
            distribution="Normal",
            distribution_kwargs={"loc": 0.0, "scale": 0.1},
        )
    )
    implementation = Implementation(
        intervention="tv_ads",
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2023, 12, 31),
        covariate_categories=None,
    )
    result = model.add_implementations(implementation)
    assert result is model
    assert model._implementations == [implementation]
