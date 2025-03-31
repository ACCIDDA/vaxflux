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
        pd.DataFrame(data={"season": ["2024/25"], "incidence": [0]}),
        pd.DataFrame(data={"start_date": ["2025-01-01"], "incidence": [0]}),
        pd.DataFrame(data={"end_date": ["2025-01-31"], "incidence": [0]}),
        pd.DataFrame(
            data={"season": ["2024/25"], "start_date": ["2025-01-01"], "incidence": [0]}
        ),
        pd.DataFrame(
            data={"season": ["2024/25"], "end_date": ["2025-01-31"], "incidence": [0]}
        ),
        pd.DataFrame(
            data={
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "incidence": [0],
            }
        ),
        pd.DataFrame(data={"other_date": ["2025-01-31"], "incidence": [0]}),
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
                "season": ["Winter 2021"],
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "incidence": [0],
            }
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "2/1/2021"],
                "end_date": ["1/31/2021", "2/28/2021"],
                "incidence": [0.55, 0.62],
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
                "season": ["2019-2020", "2019-2020"],
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-01", "2020-01-02"],
                "report_date": ["2020-01-01", "2020-01-02"],
                "incidence": [0.4, 0.5],
            }
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "1/8/2021"],
                "end_date": ["1/7/2021", "1/14/2021"],
                "other_date": ["1/8/2021", "1/15/2021"],
                "incidence": [0.6, 0.7],
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
