"""Unit tests for the `VaxfluxModel.add_observations` method."""

from datetime import date

import pandas as pd
import pytest

from vaxflux._curves import LogisticCurve
from vaxflux._vaxflux_model import VaxfluxModel


def test_add_observations_raises_when_already_set() -> None:
    """Adding observations twice raises a `ValueError`."""
    observations = pd.DataFrame.from_records(
        [
            {
                "season": "2022 thru 2023",
                "start_date": date(2022, 1, 1),
                "end_date": date(2022, 1, 31),
                "report_date": date(2022, 2, 1),
                "type": "incidence",
                "value": 0.2,
            },
            {
                "season": "2022 thru 2023",
                "start_date": date(2022, 2, 1),
                "end_date": date(2022, 2, 28),
                "report_date": date(2022, 3, 1),
                "type": "incidence",
                "value": 0.3,
            },
        ],
    )
    model = VaxfluxModel(curve=LogisticCurve())
    model.add_observations(observations)
    with pytest.raises(
        ValueError,
        match=r"^Observations have already been added to the model\.$",
    ):
        model.add_observations(observations)
