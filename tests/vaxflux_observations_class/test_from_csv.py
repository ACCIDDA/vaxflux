"""Unit tests for `VaxfluxObservations.from_csv`."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from vaxflux._vaxflux_observations import VaxfluxObservations


def test_from_csv_wraps_pandas_read_csv_and_forwards_args(tmp_path: Path) -> None:
    """`from_csv` forwards arguments to `pandas.read_csv` before validation."""
    csv_path = tmp_path / "observations.psv"
    pd.DataFrame(
        data={
            "season": ["2023/2024", "2023/2024"],
            "start_date": ["2023-10-01", "2023-10-08"],
            "end_date": ["2023-10-07", "2023-10-14"],
            "type": ["incidence", "incidence"],
            "value": [0.1, 0.2],
            "region": ["west", "east"],
        }
    ).to_csv(csv_path, sep="|", index=False)

    observations = VaxfluxObservations.from_csv(csv_path, sep="|")

    assert isinstance(observations, VaxfluxObservations)
    assert observations.covariate_columns == ["region"]
    assert is_datetime64_any_dtype(observations.data["start_date"])
    assert is_datetime64_any_dtype(observations.data["end_date"])
    assert is_datetime64_any_dtype(observations.data["report_date"])
    assert observations.data["value"].tolist() == [0.1, 0.2]
