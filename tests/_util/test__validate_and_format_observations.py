"""Unit tests for `vaxflux._util._validate_and_format_observations`."""

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype

from vaxflux._util import _validate_and_format_observations


def test_none_returns_none() -> None:
    """Providing `None` returns `None`."""
    assert _validate_and_format_observations(None) is None


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(),
        pd.DataFrame(data={"incidence": []}),
    ],
)
def test_zero_length_data_frame_value_error(observations: pd.DataFrame) -> None:
    """Providing a zero-length data frame raises a `ValueError`."""
    with pytest.raises(ValueError, match="^No observations provided.$"):
        _validate_and_format_observations(observations)


def test_incidence_column_not_in_columns_raises_not_implemented_error() -> None:
    """Providing a DataFrame without 'incidence' raises a `NotImplementedError`."""
    with pytest.raises(
        NotImplementedError,
        match=(
            "^Only 'incidence' data is supported, 'prevalence' "
            "and count equivalents are planned.$"
        ),
    ):
        _validate_and_format_observations(pd.DataFrame(data={"prevalence": [0]}))


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(data={"incidence": [0], "start_date": ["2025-01-01"]}),
        pd.DataFrame(data={"incidence": [0], "end_date": ["2025-01-31"]}),
        pd.DataFrame(data={"incidence": [0], "report_date": ["2025-01-31"]}),
    ],
)
def test_missing_required_columns_raises_value_error(
    observations: pd.DataFrame,
) -> None:
    """Providing a DataFrame missing required columns raises a `ValueError`."""
    with pytest.raises(
        ValueError,
        match="^The observations DataFrame is missing required columns: .*.$",
    ):
        _validate_and_format_observations(observations)


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(
            data={
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "incidence": [0],
            }
        ),
        pd.DataFrame(
            data={
                "incidence": [0.5, 0.5],
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-01", "2020-01-02"],
                "report_date": ["2020-01-01", "2020-01-02"],
            }
        ),
    ],
)
def test_returns_copy_of_original_data_frame(observations: pd.DataFrame) -> None:
    """The function uses a copy of the original data frame."""
    fmt_observations = _validate_and_format_observations(observations)
    assert fmt_observations is not observations
    assert fmt_observations.index is not observations.index
    assert fmt_observations.to_numpy() is not observations.to_numpy()


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(
            data={
                "incidence": [0.5, 0.5],
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-01", "2020-01-02"],
                "report_date": ["2020-01-01", "2020-01-02"],
            }
        ),
        pd.DataFrame(
            data={
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-02", "2020-01-03"],
                "other_date": ["2020-01-03", "2020-01-04"],
                "incidence": [0.5, 0.5],
            }
        ),
    ],
)
def test_date_columns_coerced_to_datetime64_type(observations: pd.DataFrame) -> None:
    """The date columns are coerced to `datetime64` type."""
    fmt_observations = _validate_and_format_observations(observations)
    assert all(
        is_datetime64_any_dtype(fmt_observations[col])
        for col in ["start_date", "end_date", "report_date"]
        if col in fmt_observations.columns
    )
